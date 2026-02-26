# SegKV-CF: Segment-Level KV Cache Eviction with Coarse-to-Fine Refinement

**Official implementation** of the paper:
**SegKV-CF: Segment-Level KV Cache Eviction with Coarse-to-Fine Refinement**.

This repository provides the **inference/prediction script** to run SegKV-CF.

---

## Introduction

![SegKV-CF Overview](imgs/framework.pdf)

Long-context inference in LLMs requires KV cache whose memory footprint grows linearly with context length, creating substantial memory pressure and latency overhead. Existing KV cache eviction methods mitigate this by evicting tokens deemed unimportant, but they often (i) fail to preserve **semantic integrity** during eviction, and (ii) only rely on importance signals from prefilling that do not always align with **decoding-time**.

We propose **SegKV-CF**, a **training-free** KV cache eviction method that addresses both issues via **segment-level eviction** and a **coarse-to-fine, two-stage refinement** strategy:

- **Segment-level eviction**: Use separator-defined natural language boundaries as eviction units to better preserve semantic integrity.
- **Coarse-to-fine refinement**: Perform **semantic coarse selection** during prefilling to retain a high-recall candidate set, then apply **semantic fine refinement** using early response signals during decoding to keep only truly critical segments.

---

## Quick Start

### Requirements

- `transformers==4.56.2`

### Installation

```bash
python pred.py --model llama-3.1-8b-instruct --kv_comp SegKV_CF

```
