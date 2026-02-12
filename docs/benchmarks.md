# Benchmarks

This page collects representative benchmark snapshots and the method used to keep comparisons fair.

## 1) Context sweep (`llama-bench`)

**Config**
- Model: Qwen3-Coder-Next REAP-40B A3B Q2_K_XL
- Flags: `-fa 1 -ctk q8_0 -ctv q4_0 -ncmoe 0 -ts 14/10 -b 128 -ub 64`

| Prefill context | tok/s |
|---:|---:|
| 32k | 180.82 |
| 64k | 175.98 |
| 96k | 171.39 |
| 128k | 166.92 |

**Generation (`tg256`)**: **~20.72 tok/s**

---

## 2) gpt-oss-20b (GGUF) on single RTX 3060

- **Model:** `ggml-org/gpt-oss-20b-GGUF`
- **GPU:** single RTX 3060 12GB
- **Generation:** **~66-68 tok/s** (about 2000 tokens in ~30.3s)

---

## 3) Nemotron profile comparison (chat-level, fixed 5-task set)

| Profile class | Runtime | Input tok | Output tok | Effective output tok/s |
|---|---:|---:|---:|---:|
| `reduced-thinking-balanced` (recommended) | 11.57s | 277 | 651 | 56.27 |
| `non-thinking-speed` (fastest) | 6.36s | 282 | 338 | 53.14 |

- `reduced-thinking-balanced`: reasoning-budget `-1` + constrained brief reasoning style.
- `non-thinking-speed`: reasoning-budget `0`; faster but more prone to output-cleanliness artifacts.

Detailed artifacts:
- `experiments/2026-02-12-nemotron-abc-executive-summary.md`
- `experiments/2026-02-12-nemotron-abc-timing-token-breakdown.md`

---

## Methodology (comparison discipline)

- Use **`llama-bench`** for prefill and generation throughput.
- Keep benchmark flags stable (`-b/-ub`, `-ctk/-ctv`, `-fa`, `-ts`, `-ncmoe`).
- For chat-level comparisons, keep prompt set fixed and compare:
  1. total runtime,
  2. token volume,
  3. effective tok/s.
- **Do not mix profile classes** (`reduced-thinking-balanced` vs `non-thinking-speed`) into one aggregate claim.

> Results are hardware-specific and should be treated as comparative guidance, not universal constants.
