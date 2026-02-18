import torch
from typing import Optional
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.integrations.flash_attention import flash_attention_forward
from .segmentkv_two_stages_utils import init_twostage_segmentkv



def llama_flash_attn2_forward(
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
        q_len = input_shape[1]
        is_prefill = q_len > 1

        # GQA 处理
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if is_prefill:
            # 初始化两阶段压缩器 (假设 init_twostage_segmentkv 会给 self 挂载 kv_comp)
            init_twostage_segmentkv(self)
            # 核心：执行 Prefill 压缩逻辑
            k_comp, v_comp = self.kv_comp.update_kv(query_states, key_states, value_states)

            # 更新 Transformers 的 Cache
            # 注意：此处 update 后的 key_states 会变成包含压缩部分的全局 KV
            # key_states, value_states = past_key_values.update(k_final, v_final, self.layer_idx)
            # prefill 使用 full kv 生成第一个token
            past_key_values.update(k_comp, v_comp, self.layer_idx)
        else:
            # Decoding 阶段
            if q_len == 1 and self.kv_comp.is_compressed_stage1 and self.kv_comp.is_compressed_stage2:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            else:
                # 将当前key_states, value_states, 和第一阶段压缩后的 key_comp 和 value_comp 结合，整体传入kv_comp进行update
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

                k_comp, v_comp = self.kv_comp.update_kv(query_states, key_states, value_states)

                # 此时，需要覆盖 Cache 中原本的 KV 序列
                past_key_values.layers[self.layer_idx].keys = k_comp
                past_key_values.layers[self.layer_idx].values = v_comp

                # 后续解码使用
                key_states, value_states = k_comp, v_comp

    # 计算注意力
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

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights