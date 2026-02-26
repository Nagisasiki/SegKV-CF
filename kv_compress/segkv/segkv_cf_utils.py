import torch
import torch.nn as nn
import math
import json
from pathlib import Path
import re


class SegKVCF:
    def __init__(self, suffix_window=32, coarse_budget=2048, prefix_window=4, budget=256, segments=None):
        self.coarse_budget = coarse_budget
        self.budget = budget
        self.suffix_window = suffix_window
        self.prefix_window = prefix_window

        assert segments is not None
        self.segments = segments

        # stage status
        self.is_compressed_stage1 = False
        self.is_compressed_stage2 = False
        self.kept_indices_stage1 = None
        self.token2seg_stage1 = None

        # current step
        self.gen_step = 0
        self.gen_query = None

    def _get_token2seg(self, num_tokens, bsz, num_heads):
        """Compute the sentence index for each token in the original sequence based on segment boundaries."""
        device = self.segments.device
        positions = torch.arange(num_tokens, device=device)
        # Use right=True so punctuation is assigned to the preceding sentence.
        token2seg = torch.searchsorted(self.segments, positions, right=True) - 1
        token2seg = token2seg.unsqueeze(0).expand(bsz, num_heads, -1)
        return token2seg

    def calcul_scores_with_query(self, query_states, key_states, token2seg_map, num_segments):
        """compute segment-level and token-level scores"""
        bsz, num_heads, q_len, head_dim = query_states.shape
        device = query_states.device

        # 1. Compute attention weights
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((q_len, q_len), torch.finfo(attn_weights.dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -q_len:, -q_len:] += mask[None, None, :, :]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 2. Aggregate into token scores (mean over the observation window)
        token_important = attn_weights[:, :, -q_len:, :-q_len].mean(dim=-2)

        if q_len == self.gen_step:
            token_important = token_important[:, :, :-self.suffix_window]


        # 3. Accumulate in segment.
        indices = token2seg_map
        seg_scores = torch.zeros((bsz, num_heads, num_segments), device=device, dtype=attn_weights.dtype)
        seg_scores = seg_scores.scatter_add(dim=-1, index=indices, src=token_important)

        # 4. Normalize by segment length
        ones = torch.ones_like(token_important)
        seg_counts = torch.zeros((bsz, num_heads, num_segments), device=device, dtype=attn_weights.dtype)
        seg_counts = seg_counts.scatter_add(dim=-1, index=indices, src=ones)
        seg_scores = seg_scores
        seg_scores = seg_scores / (seg_counts + 1e-6)


        # 5. Broadcast back to per-token scores
        token_scores = seg_scores.gather(dim=-1, index=indices)
        return token_scores

    def update_kv(self, query_states, key_states, value_states):
        bsz, num_heads, seq_len, head_dim = query_states.shape
        num_segments = self.segments.shape[0] - 1

        # seq < coarse_budget
        if seq_len < self.coarse_budget and not self.is_compressed_stage1:
            self.token2seg_stage1 = self._get_token2seg(seq_len - self.suffix_window, bsz, num_heads)
            self.is_compressed_stage1 = True
            return key_states, value_states

        # stage 1: semantic coarse selection
        if seq_len > 1 and not self.is_compressed_stage1:
            token2seg_full = self._get_token2seg(seq_len - self.suffix_window, bsz, num_heads)

            # Compute scores using the prompt suffix window
            token_scores = self.calcul_scores_with_query(
                query_states[:, :, -self.suffix_window:, :],
                key_states,
                token2seg_full,
                num_segments
            )

            indices = token_scores.topk(self.coarse_budget - self.suffix_window, dim=-1).indices

            k_past = key_states[:, :, :-self.suffix_window, :].gather(dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            v_past = value_states[:, :, :-self.suffix_window, :].gather(dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            key_states = torch.cat([k_past, key_states[:, :, -self.suffix_window:, :]], dim=2)
            value_states = torch.cat([v_past, value_states[:, :, -self.suffix_window:, :]], dim=2)

            # Cache the kept indices for stage 2 refinement
            self.kept_indices_stage1 = indices
            self.token2seg_stage1 = token2seg_full.gather(dim=-1, index=indices)
            self.is_compressed_stage1 = True

            return key_states, value_states

        # seq_len < budget
        current_key_len = key_states.shape[2]
        if seq_len == 1 and self.is_compressed_stage1 and not self.is_compressed_stage2 and current_key_len < self.budget:
            self.is_compressed_stage2 = True
            self.token2seg_stage1 = None
            return key_states, value_states

        # semantic fine refine
        if seq_len == 1 and self.is_compressed_stage1 and not self.is_compressed_stage2:
            self.gen_step = self.gen_step + 1
            if self.gen_query is  None:
                self.gen_query = query_states
            else:
                self.gen_query = torch.cat([self.gen_query, query_states], dim=-2)

            if self.gen_step == self.prefix_window:
                cur_window_size = self.suffix_window + self.prefix_window

                token_scores = self.calcul_scores_with_query(
                    self.gen_query,
                    key_states,
                    self.token2seg_stage1,
                    num_segments
                )

                indices = token_scores.topk(self.budget - cur_window_size, dim=-1).indices

                # final eviction
                k_final_past = key_states[:, :, :-cur_window_size, :].gather(dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
                v_final_past = value_states[:, :, :-cur_window_size, :].gather(dim=2, index=indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))

                key_states = torch.cat([k_final_past, key_states[:, :, -cur_window_size:, :]], dim=2)
                value_states = torch.cat([v_final_past, value_states[:, :, -cur_window_size:, :]], dim=2)


                self.is_compressed_stage2 = True
                self.kept_indices_stage1 = None
                self.token2seg_stage1 = None

                return key_states, value_states

            else:
                return key_states, value_states

        # After both stages, proceed with the normal decoding path
        return key_states, value_states


_SEP_DIR = Path(__file__).resolve().parent / "separator"

def create_separator_for_model(model_name, tokenizer):
    # QA
    qa_pattern = re.compile(r'^(?=.*[!.?\r\n])(?!^[\"%\'\)\]]+$)[!.?\r\n\"%\'\)\]]+$')

    # Code
    code_pattern = re.compile(r'^[ )\]{}\"\' ]*[;:\n][ ;:\n]*$')

    vocab = tokenizer.get_vocab()  # token -> id
    id_to_token = {v: k for k, v in vocab.items()}

    punc_qa = []
    punc_code = []

    for id_ in sorted(id_to_token.keys()):
        decoded = tokenizer.decode([id_])

        if qa_pattern.match(decoded):
            punc_qa.append(id_)

        if code_pattern.match(decoded):
            punc_code.append(id_)

    # Persist to separator/{model_name}.jsonl
    sep_path = _SEP_DIR / f"{model_name}.jsonl"
    sep_path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"punc_qa": punc_qa, "punc_code": punc_code}
    with sep_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return punc_qa, punc_code


def get_separator(model_name, tokenizer):
    sep_path = _SEP_DIR / f"{model_name}.jsonl"

    if sep_path.exists():
        with sep_path.open("r", encoding="utf-8") as f:
            line = f.readline().strip()
        obj = json.loads(line)
        punc_qa = obj.get("punc_qa")
        punc_code = obj.get("punc_code")
        return punc_qa, punc_code
    else:
        return create_separator_for_model(model_name, tokenizer)

def get_segments(input_ids, model_name, tokenizer, window_size=32):
    bs, seq_len = input_ids.shape
    assert bs == 1
    max_len = seq_len - window_size
    device = input_ids.device

    punc_qa, punc_code = get_separator(model_name, tokenizer)

    punc_qa = torch.tensor(punc_qa, device=device, dtype=input_ids.dtype)
    mask_qa = torch.isin(input_ids[0], punc_qa)
    punc_position_qa = torch.nonzero(mask_qa, as_tuple=False).reshape(-1)

    punc_code = torch.tensor(punc_code, device=device, dtype=input_ids.dtype)
    mask_code = torch.isin(input_ids[0], punc_code)
    punc_position_code = torch.nonzero(mask_code, as_tuple=False).reshape(-1)

    punc_position = punc_position_qa if punc_position_qa.sum() > punc_position_code.sum() else punc_position_code

    bounds = punc_position[punc_position < max_len] + 1  # +1 causes the punctuation mark to be placed in its corresponding paragraph.
    bounds = bounds.sort(descending=False).values

    start = torch.tensor([0], device=device)
    end = torch.tensor([max_len], device=device)
    if end != bounds[-1]:
        segments = torch.cat([start, bounds, end]).to(device)
    else:
        segments = torch.cat([start, bounds]).to(device)

    return segments


def init_segkv_cf(self):
    if not hasattr(self, "kv_comp"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'lookahead_steps'):
            self.config.lookahead_steps = 4
        if not hasattr(self.config, 'coarse_budget'):
            self.config.coarse_budget = 2048
        if not hasattr(self.config, 'budget'):
            self.config.budget = 128
        assert hasattr(self.config, 'segments')

    self.kv_comp = SegKVCF(
        suffix_window=self.config.window_size,
        coarse_budget=self.config.coarse_budget,
        prefix_window=self.config.lookahead_steps,
        budget=self.config.budget,
        segments=self.config.segments,
        )