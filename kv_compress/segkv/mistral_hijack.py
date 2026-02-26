import torch
from typing import Optional
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.integrations.flash_attention import flash_attention_forward
from .segkv_cf_utils import init_segkv_cf
from .segkv_utils import init_segkv



def mistral_flash_attn2_forward_segkv_cf(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        q_len = input_shape[1]
        is_prefill = q_len > 1

        # GQA 处理
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if is_prefill:
            init_segkv_cf(self)
            k_comp, v_comp = self.kv_comp.update_kv(query_states, key_states, value_states)
            past_key_values.update(k_comp, v_comp, self.layer_idx, cache_kwargs)

        else:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # Decoding
            if q_len == 1 and self.kv_comp.is_compressed_stage1 and self.kv_comp.is_compressed_stage2:
                pass
            else:
                # semantic fine refine
                key_states, value_states = self.kv_comp.update_kv(query_states, key_states, value_states)

                past_key_values.layers[self.layer_idx].keys = key_states
                past_key_values.layers[self.layer_idx].values = value_states


    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights



def mistral_flash_attn2_forward_seg(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        ##############################
        # if prefill stage
        q_len = input_shape[1]
        is_prefill = q_len != 1

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if not is_prefill:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            init_segkv(self)
            key_states_compress, value_states_compress = self.kv_comp.update_kv(key_states, query_states, value_states)
            past_key_values.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        #############################

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights