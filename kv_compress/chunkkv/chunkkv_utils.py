import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ChunkKV:
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, chunk_length = 10, kernel_size = 5):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.chunk_length = chunk_length

        self.save_indices = None
        self.save_scores = None


    def update_kv(self, key_states, query_states, value_states, num_key_value_groups=1):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            scores = attn_weights[:, :, -self.window_size:, : -self.window_size].mean(dim = -2)

            #######  The source code provided in the paper on GitHub is avg_pool.  #######
            scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

            #######  With num_key_value_groups set to 1, the following steps can be ignored.  #######
            # Average per group (https://github.com/FasterDecoding/SnapKV/issues/22)
            scores = scores.view(bsz, int(num_heads/num_key_value_groups), num_key_value_groups, q_len - self.window_size)
            scores = scores.mean(2)

            # Add back the observation window. Use max score to make sure the window is not pruned.
            global_scores = F.pad(scores, (0, self.window_size), value=scores.max().item())

            # 2. Calculate actual number of complete chunks and remaining tokens
            num_complete_chunks = q_len // self.chunk_length
            remaining_tokens = q_len % self.chunk_length

            # Reshape complete chunks for score calculation
            if num_complete_chunks > 0:
                main_scores = global_scores[..., : num_complete_chunks * self.chunk_length]
                main_chunk_scores = main_scores.sum(dim=1).view(-1, num_complete_chunks, self.chunk_length)   ## 所有头求均值？？
                main_chunk_scores = main_chunk_scores.mean(dim=-1)    # 块内求均值
            else:
                main_chunk_scores = torch.empty((global_scores.shape[0], 0), device=global_scores.device)

            # Handle remaining tokens if any
            if remaining_tokens > 0:
                remaining_scores = global_scores[..., -remaining_tokens:]
                remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
                chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
            else:
                chunk_scores = main_chunk_scores

            # 3. Calculate number of chunks to keep
            n_chunks_kept = int(self.max_capacity_prompt / self.chunk_length)
            top_chunks = chunk_scores.topk(n_chunks_kept, dim=-1)

            # 4. Create indices for selected chunks
            indices = []

            scores = []
            for chunk_idx in top_chunks.indices[0]:
                if chunk_idx < num_complete_chunks:
                    # For complete chunks
                    start_idx = chunk_idx * self.chunk_length
                    chunk_indices = torch.arange(start_idx, start_idx + self.chunk_length, device=key_states.device)
                else:
                    # For the remaining partial chunk
                    chunk_indices = torch.arange(num_complete_chunks * self.chunk_length, q_len, device=key_states.device)
                indices.append(chunk_indices)

                scores.append(chunk_scores[0, chunk_idx].expand(self.chunk_length))

            indices = torch.cat(indices).sort()[0]
            self.save_indices = indices
            self.save_scores = scores

            indices = indices.view(1, 1, -1, 1).expand(key_states.shape[0], key_states.shape[1], -1, head_dim)

            # 5. Use gather to collect selected keys and values
            keys = key_states.gather(2, indices).contiguous()
            values = value_states.gather(2, indices).contiguous()
            return keys, values


def init_chunkkv(self):
    if not hasattr(self, "kv_comp"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32                          # not mentioned in the paper, set to a general value of 32 here.
        if not hasattr(self.config, 'max_capacity'):
            self.config.max_capacity = 1024
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5                           # The source code provided in the paper on GitHub is kernel_size = 5.
        if not hasattr(self.config, 'chunk_length'):
            self.config.chunk_length = 10                         # According to paper B.4 Chunk Size, it is set to 10.

    self.kv_comp = ChunkKV(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity,
        kernel_size = self.config.kernel_size,
        chunk_length = self.config.chunk_length
        )