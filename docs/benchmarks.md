# Benchmarks

## Context sweep (llama-bench)
**Config:** Qwen3‑Coder‑Next REAP‑40B A3B Q2_K_XL, FA=1, K=q8_0, V=q4_0, `n-cpu-moe 0`, `ts 14/10`, `-b 128 -ub 64`

| Prefill ctx | tok/s |
|---:|---:|
| 32k | 180.82 |
| 64k | 175.98 |
| 96k | 171.39 |
| 128k | 166.92 |

**Generation:** `tg256` ≈ **20.72 tok/s**

## gpt‑oss‑20b (GGUF) — single 3060
- **Model:** `ggml-org/gpt-oss-20b-GGUF` (default)
- **GPU:** single RTX 3060 12 GB
- **Generation:** **~66–68 tok/s** (≈2000 tokens in ~30.3s)

## Nemotron A/B/C runtime profile (chat-level, 5-task set)

| Profile | Runtime | Input tok | Output tok | Output tok/s |
|---|---:|---:|---:|---:|
| B_on (recommended) | 11.57s | 277 | 651 | 56.27 |
| B_off (fastest) | 6.36s | 282 | 338 | 53.14 |

- `B_on`: reasoning-budget `-1` + brief constrained reasoning prompt style.
- `B_off`: reasoning-budget `0`; faster but can show output-cleanliness artifacts.

See detailed breakdown in:
- `experiments/2026-02-12-nemotron-abc-executive-summary.md`
- `experiments/2026-02-12-nemotron-abc-timing-token-breakdown.md`

## Methodology
- Use **llama-bench** for prefill + gen
- Keep flags consistent (`-b/-ub`, `-ctk/-ctv`, `-fa`, `-ts`, `-ncmoe`)
- For chat-level comparisons, keep prompt set fixed and compare total runtime + token volume + effective tok/s

> Results are hardware‑specific; treat as comparative, not universal.
