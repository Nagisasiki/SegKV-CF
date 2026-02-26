import torch
import torch.nn as nn
import math


# perform qk calculation and get indices
# this version will not update in inference mode
class SegKV:
    def __init__(self, window_size = 32, max_capacity = 1024, segments=None):
        self.max_capacity = max_capacity
        self.window_size = window_size

        assert segments is not None
        self.segments = segments

    def calcul_segments_scores(self, query_states, key_states):
        assert query_states.shape[-2] == key_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        assert bsz == 1
        device = query_states.device

        # calculate token score
        attn_weights = torch.matmul(query_states[..., -self.window_size :, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(device)
        attention_mask = mask[None, None, :, :]
        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        token_attn_weights = attn_weights[:, :, -self.window_size:, :-self.window_size].mean(dim=-2)

        # calculate segment score
        max_len = self.segments[-1]
        positions = torch.arange(max_len, device=device)
        token2seg = torch.searchsorted(self.segments, positions, right=True) - 1 # -1 because the index starts from 0.

        indices = token2seg.unsqueeze(0).expand(bsz, num_heads, -1)
        num_segments = self.segments.shape[0] - 1
        seg_scores = torch.zeros((bsz, num_heads, num_segments), device=device, dtype=attn_weights.dtype)
        seg_scores = seg_scores.scatter_add(dim=-1, index=indices, src=token_attn_weights)
        seg_len = torch.diff(self.segments)
        true_seg_scores = seg_scores / seg_len.unsqueeze(0)


        token_scores = true_seg_scores[..., token2seg]
        return seg_scores, token_scores, token_attn_weights

    def update_kv(self, key_states, query_states, value_states):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, seq_len, head_dim = query_states.shape

        if seq_len < self.max_capacity:
            return key_states, value_states
        else:

            _, token_scores, _ = self.calcul_segments_scores(query_states, key_states)

            indices = token_scores.topk(self.max_capacity - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states

def init_segkv(self):
    if not hasattr(self, "kv_comp"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'budget'):
            self.config.budget= 1024
        assert hasattr(self.config, 'segments')

    self.kv_comp = SegKV(
        window_size = self.config.window_size,
        max_capacity = self.config.budget,
        segments=self.config.segments,
        )