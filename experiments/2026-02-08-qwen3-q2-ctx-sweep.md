# 2026-02-08 — Qwen3 Q2 context sweep

**Goal →** measure prefill drop‑off as context grows.

**Setup →**
- Qwen3‑Coder‑Next REAP‑40B A3B Q2_K_XL
- FA=1, K=q8_0, V=q4_0
- `n-cpu-moe 0`, `tensor-split 14/10`, `-b/-ub 128/64`
- llama‑bench prefill sweep

**Results (prefill)**
| ctx | tok/s |
|---:|---:|
| 32k | 180.82 |
| 64k | 175.98 |
| 96k | 171.39 |
| 128k | 166.92 |

**Generation**
- `tg256` ≈ **20.72 tok/s**
