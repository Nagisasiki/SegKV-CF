# SegKV-CF: Segment-Level KV Cache Eviction with Coarse-to-Fine Refinement

**Official (minimal) implementation for review** of the paper:
**SegKV-CF: Segment-Level KV Cache Eviction with Coarse-to-Fine Refinement**.

This repository currently provides the **LongBench prediction script** used to run inference with SegKV-CF.
Additional components (full evaluation pipeline, RULER scripts, and complete reproducibility utilities) will be added by **<2026-02-23>**.


---

## Introduction

Long-context inference in LLMs requires a KV cache whose memory footprint grows linearly with context length, creating substantial memory pressure and latency overhead. Existing KV cache eviction methods mitigate this by evicting tokens deemed unimportant, but they often (i) fail to preserve **semantic integrity** during eviction, and (i) rely on importance signals from prefilling that do not always align with **decoding-time** importance.

We propose **SegKV-CF**, a **training-free** KV cache eviction method that addresses both issues via **segment-level eviction** and a **coarse-to-fine, two-stage refinement** strategy:
- **Segment-level eviction**: Use separator-defined natural language boundaries as eviction units to better preserve semantic integrity.
- **Coarse-to-fine refinement**: Perform **semantic coarse selection** during prefilling to retain a high-recall candidate set, then apply **semantic fine refinement** using early response signals during decoding to keep only truly critical segments.

(See paper Sections 1–4 for details.)

---

## Quick Start

### Requirements

- `transformers==4.56.2`

### Installation

```bash

# Create your environment (conda / venv) then install deps
pip install -r requirements.txt
```

---

## Run LongBench Prediction

This repo currently supports running **LongBench prediction** with SegKV-CF via `pred.py`.

### Arguments

`pred.py` supports the following key arguments:

- `--model`: backbone model
  - choices: `llama-3.1-8b-instruct`, `mistral-7B-instruct-v0.2`
- `--compress`: enable KV compression/eviction (default: True)
- `--kv_comp`: KV compression method (default: `segment_two_stage`)
- `--window_size`: prompt suffix window size used for scoring (default: 32)
- `--max_capacity`: target KV budget / cache capacity (default: 128)
- `--lookahead_steps`: response prefix length (number of initial generated tokens) used for refinement (default: 2)

### Example Command

```bash
python pred.py   --model llama-3.1-8b-instruct   --compress True   --kv_comp segment_two_stage   --window_size 32   --max_capacity 128   --lookahead_steps 2
```

**Notes**
- `window_size` corresponds to the observation window at the end of the prompt (prompt suffix).
- `lookahead_steps` controls how many initial generated tokens are used to form response signals for the second-stage refinement.
- `max_capacity` is the final KV cache budget enforced after refinement.

---

## Reproducibility Status

- [x] LongBench prediction script (`pred.py`)
- [ ] Full LongBench evaluation scripts and result aggregation (**planned by <2026-02-23>**)
- [ ] RULER scripts (**planned by <2026-02-23>**)

