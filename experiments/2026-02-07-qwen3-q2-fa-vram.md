# 2026-02-07 — Qwen3 Q2 (FA + V=q4_0) VRAM tuning

**Goal →** shrink KV cache and move MoE experts onto GPU for higher throughput.

**Setup →**
- Qwen3‑Coder‑Next REAP‑40B A3B Q2_K_XL
- FA=1, K=q8_0, V=q4_0, `-b/-ub 128/64`
- Dual RTX 3060 12GB, row‑split

**Metrics →**
| n-cpu-moe | tensor‑split | pp tok/s | tg tok/s | GPU0 / GPU1 (GiB) |
|---:|---:|---:|---:|---:|
| 27 | 18/6 | 64.7 | 16.7 | 6.7 / 5.6 |
| 20 | 18/6 | 76.0 | 17.5 | 8.7 / 5.6 |
| 16 | 18/6 | 84.6 | 18.2 | 9.8 / 5.6 |
| 12 | 18/6 | 92.2 | 18.8 | 10.9 / 5.6 |
| 12 | 16/8 | 89.5 | 18.5 | 9.4 / 7.3 |
| 8  | 16/8 | 92.4 | 12.8 | 10.5 / 7.3 |
| 12 | 14/10| 89.5 | 18.3 | 7.9 / 8.8 |
| 0  | 14/10| 136.9 | 20.8 | 11.2 / 8.8 |

**Conclusion →**
- Moving all MoE experts to GPU (n‑cpu‑moe 0) gave the best speed.
- 14/10 is the best balance across GPUs.
