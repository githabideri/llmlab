# Benchmarks

This page collects representative benchmark snapshots and the method used to keep comparisons fair.

## 1) Context sweep (`llama-bench`)

The following context sweep was run on Qwen3-Coder-Next REAP-40B A3B Q2_K_XL with:
`-fa 1 -ctk q8_0 -ctv q4_0 -ncmoe 0 -ts 14/10 -b 128 -ub 64`.

| Prefill context | tok/s |
|---:|---:|
| 32k | 180.82 |
| 64k | 175.98 |
| 96k | 171.39 |
| 128k | 166.92 |

Generation (`tg256`) is **~20.72 tok/s**.

---

## 2) gpt-oss-20b (GGUF) on single RTX 3060

On a single RTX 3060 12GB with `ggml-org/gpt-oss-20b-GGUF`, measured generation speed is **~66-68 tok/s** (about 2000 tokens in ~30.3s).

---

## 3) Nemotron profile comparison (chat-level, fixed 5-task set)

| Profile class | Runtime | Input tok | Output tok | Effective output tok/s |
|---|---:|---:|---:|---:|
| `reduced-thinking-balanced` (recommended) | 11.57s | 277 | 651 | 56.27 |
| `non-thinking-speed` (fastest) | 6.36s | 282 | 338 | 53.14 |

`reduced-thinking-balanced` uses reasoning-budget `-1` with constrained brief reasoning. `non-thinking-speed` uses reasoning-budget `0`; it is faster but more prone to output-cleanliness artifacts.

Detailed artifacts:
- `experiments/2026-02-12-nemotron-abc-executive-summary.md`
- `experiments/2026-02-12-nemotron-abc-timing-token-breakdown.md`

---

## Methodology (comparison discipline)

Comparison quality depends more on discipline than on raw throughput numbers. Keep benchmark flags stable (`-b/-ub`, `-ctk/-ctv`, `-fa`, `-ts`, `-ncmoe`) and use `llama-bench` for both prefill and generation throughput.

For chat-level runs, keep the prompt set fixed and compare the same three dimensions every time: total runtime, token volume, and effective tok/s. Do not merge `reduced-thinking-balanced` and `non-thinking-speed` into a single aggregate claim.

> Results are hardware-specific and should be treated as comparative guidance, not universal constants.
