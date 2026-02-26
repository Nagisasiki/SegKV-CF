import torch
import torch.nn.functional as F
from typing import Optional, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast,CausalLMOutputWithPast
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.integrations.flash_attention import flash_attention_forward
from .LAQ_Cache import DynamicCache_LAQ
from .LAQ_utils import LAQKVCluster
from transformers.utils import (
    logging,
)
logger = logging.get_logger(__name__)


def LlamaForCausalLM_forward_LAQ(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> CausalLMOutputWithPast:
    """
    LAQ-compatible forward that follows the transformers-style signature.
    Assumes helper functions/classes exist:
      - DynamicCache_LAQ()
      - get_softmax_max_tokenid(logits, tokenizer)
      - select_KV_lookahead_stage(full_past_kv, config)
      - select_KV_decoding_stage(full_past_kv, stage2_past_kv, new_token_len, lookahead_method, max_capacity_prompts, all_query, stage2_window_sizes)
    """
    # === defaults / flags ===
    use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", True)
    device = input_ids.device


    def calc_logits(input_ids_local, attention_mask_local, position_ids_local, past_kv_local, cache_position_local):
        outputs = self.model(
            input_ids=input_ids_local,
            attention_mask=attention_mask_local,
            position_ids=position_ids_local,
            past_key_values=past_kv_local,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position_local,
            **kwargs,
        )
        # outputs.last_hidden_state shape: (batch, seq_len, hidden)
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return outputs, logits


    kv_len = past_key_values.get_seq_length() if past_key_values is not None else 0

    # Main LAQ logic
    if kv_len == 0:

        past_key_values_local = DynamicCache_LAQ()

        outputs, logits = calc_logits(
            input_ids_local=input_ids,
            attention_mask_local=attention_mask,
            position_ids_local=position_ids,
            past_kv_local=past_key_values_local,
            cache_position_local=cache_position,
        )

        ori_logits = logits
        full_past_kv = outputs.past_key_values

        id0 = get_softmax_max_tokenid(logits)
        new_tokens = [int(id0)]

        stage1_decode_kv = select_KV_lookahead_stage(full_past_kv, self.config)
        past_key_values = stage1_decode_kv


        cur_attention_mask = attention_mask
        cur_position_ids = position_ids
        # iterate lookahead decode
        while len(new_tokens) < getattr(self.config, "lookahead_size", 0):

            next_id = torch.tensor([[new_tokens[-1]]], dtype=torch.long, device=device)
            cur_attention_mask = torch.cat((cur_attention_mask, torch.ones((cur_attention_mask.size(0), 1), dtype=torch.long, device=device)), dim=1)

            next_pos = torch.tensor([[cur_position_ids[0, -1] + 1]], dtype=torch.long, device=device)
            cur_position_ids = torch.cat((cur_position_ids, next_pos), dim=1)

            cache_position = torch.tensor([int(cache_position[-1]) + 1], dtype=torch.long, device=device)

            outputs, logits = calc_logits(
                input_ids_local=next_id,
                attention_mask_local=cur_attention_mask,
                position_ids_local=next_pos,
                past_kv_local=past_key_values,
                cache_position_local=cache_position,
            )

            new_id = get_softmax_max_tokenid(logits)
            past_key_values = outputs.past_key_values

            if new_id == 128001 or len(new_tokens) == self.config.lookahead_size:
                break
            new_tokens.append(int(new_id))


        # Stage2: select final KV set for decoding, based on full_past_kv and stage2 past_kv_local
        outputs.past_key_values = select_KV_decoding_stage(
            full_past_kv,
            past_key_values,
            len(new_tokens),
            getattr(self.config, "max_capacity", None),
            getattr(self.config, "all_query", None),
            getattr(self.config, "stage2_window_sizes", None),
        )

        logits = ori_logits
        # free heavy refs
        full_past_kv = None
        past_key_values_local = None
        torch.cuda.empty_cache()

    else:
        outputs, logits = calc_logits(
            input_ids_local=input_ids,
            attention_mask_local=attention_mask,
            position_ids_local=position_ids,
            past_kv_local=past_key_values,
            cache_position_local=cache_position,
        )



    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def get_softmax_max_tokenid(logits):
    new_logits = logits[:, -1, :].softmax(-1)
    confidence, idx = torch.max(new_logits, dim=-1)
    return idx


def select_KV_decoding_stage(full_past_kv, new_kv, lookahead_size, max_capacity_prompts, all_query,
                             stage2_window_sizes):
    kernel_sizes = 7
    pooling = "maxpool"
    window_sizes = stage2_window_sizes


    stage2_decode_kv = DynamicCache()
    for layer_idx in range(len(full_past_kv)):

        cluster = LAQKVCluster(
            window_size=window_sizes,
            max_capacity_prompt=max_capacity_prompts,
            kernel_size=kernel_sizes,
            pooling=pooling,
            merge=None,
            lookahead_size=lookahead_size
        )

        key_states_compress, value_states_compress = cluster.update_kv_LAQ(full_past_kv.key_cache[layer_idx],
                                                                           new_kv.temp_local_window_query[layer_idx],
                                                                           full_past_kv.value_cache[layer_idx], None,
                                                                           None)

        stage2_decode_kv.update(key_states_compress, value_states_compress, layer_idx)

    return stage2_decode_kv


def select_KV_lookahead_stage(full_past_kv, config):
    max_capacity_prompts = config.lookahead_max_capacity
    kernel_sizes = 7
    pooling = "maxpool"
    window_sizes = config.window_size

    stage1_decode_kv = DynamicCache_LAQ()
    # stage1_decode_kv.temp_local_window_query = full_past_kv.temp_local_window_query
    for layer_idx in range(len(full_past_kv)):

        cluster = LAQKVCluster(
            window_size=window_sizes,
            max_capacity_prompt=max_capacity_prompts,
            kernel_size=kernel_sizes,
            pooling=pooling,
            merge=None,
        )
        # print(full_past_kv.temp_local_window_query[layer_idx].shape)

        key_states_compress, value_states_compress = cluster.update_kv(full_past_kv.key_cache[layer_idx],
                                                                       full_past_kv.temp_local_window_query[layer_idx],
                                                                       full_past_kv.value_cache[layer_idx], None, None)
        stage1_decode_kv.update(full_past_kv.temp_local_window_query[layer_idx], key_states_compress,
                                value_states_compress, layer_idx, 16)
        # print(stage1_decode_kv.temp_local_window_query[layer_idx].shape)
        # input()

    return stage1_decode_kv


def llama_flash_attn2_forward_LAQ(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    LAQ-compatible attention forward adapted to your transformers-style signature.
    Preserves LAQ behaviors: pretraining_tp handling, RoPE via position_embeddings,
    DynamicCache/past_key_values updates, repeat_kv, optional flash attention path.
    Returns (attn_output, attn_weights) to match your current API.
    """
    # input shapes
    input_shape = hidden_states.shape[:-1]  # (batch, seq_len)
    bsz, q_len = input_shape
    # prepare view shapes like your original code
    hidden_shape = (*input_shape, -1, self.head_dim)

    # --- Q/K/V projection (support pretraining_tp slicing) ---
    if getattr(self.config, "pretraining_tp", 1) > 1 and hasattr(self.q_proj.weight, "split"):
        # split weights for tensor-parallel pretraining
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    # reshape & transpose to (batch, num_heads, seq_len, head_dim)
    query_states = query_states.view(hidden_shape).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
    key_states = key_states.view(hidden_shape).transpose(1, 2)      # (bsz, num_heads, q_len, head_dim)
    value_states = value_states.view(hidden_shape).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)

    # position embeddings (cos, sin) expected as tuple
    if position_embeddings is None:
        # Back-compat: if position embeddings missing, try to compute from internal rotary (less preferred)
        logger.warning_once(
            "position_embeddings is None — falling back to internal rotary (will be removed in future)."
        )
        # assume module has rotary_emb that takes (value_states, position_ids) like older code
        # If you do not have that helper, it's an exceptional case and you should pass position_embeddings.
        cos, sin = self.rotary_emb(value_states, kwargs.get("position_ids", None))
    else:
        cos, sin = position_embeddings

    # apply RoPE
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # repeat key/value if model uses grouped KV heads
    if getattr(self, "num_key_value_groups", 1) != 1:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Manage kv_seq_len bookkeeping and cache updates (LAQ-specific behavior)
    kv_seq_len = key_states.shape[-2]  # current kv length (without past)
    if past_key_values is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure changed — if you're doing autoregressive decoding with k/v caching, "
                f"initialize attention with a layer index (layer_idx)."
            )
        # if class tracks kv_seq_len attribute (LAQ style), use it; otherwise use past_key_values helper
        if hasattr(self, "kv_seq_len"):
            if getattr(self, "kv_seq_len", 0) != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_values.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_values.get_usable_length(kv_seq_len, self.layer_idx)

    # If past cache exists, update it (DynamicCache or static cache updates)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        # update sequence length counter on the module if present (matches LAQ logic)
        if not hasattr(self, "kv_seq_len"):
            self.kv_seq_len = 0
        self.kv_seq_len += q_len

        if isinstance(past_key_values, DynamicCache):
            # DynamicCache has an update(key, value, layer_idx, cache_kwargs) signature in LAQ
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            # older/static cache signature: update(query, key, value, layer_idx, window_size, cache_kwargs)
            # Some implementations expect query as first arg for certain cache update logic; pass query_states to be safe.
            key_states, value_states = past_key_values.update(query_states, key_states, value_states, self.layer_idx, getattr(self.config, "window_size", None), cache_kwargs)

        # mark seen tokens if supported
        try:
            past_key_values._seen_tokens = self.kv_seq_len
        except Exception:
            pass

    # transpose back to (batch, seq_len, head, head_dim) expected by attention kernels
    # query_states = query_states.transpose(1, 2)  # -> (bsz, q_len, num_heads, head_dim)
    # key_states = key_states.transpose(1, 2)      # -> (bsz, kv_len, num_heads, head_dim)
    # value_states = value_states.transpose(1, 2)  # -> (bsz, kv_len, num_heads, head_dim)

    # dtype handling: try to preserve intended proj dtype as in LAQ
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        # cast back
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    # reshape to original hidden dims and apply output projection
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    # if user API expects attn_weights even when not requested, keep as None if not available
    return attn_output, attn_weights
