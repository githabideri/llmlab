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

## 4) ik_llama.cpp vs llama.cpp (replacement models, preliminary)

A cross-runtime comparison campaign is in progress for:
- Qwen3-30B-A3B Q4_K_M
- Qwen3-Coder-30B-A3B-Instruct Q4_K_M
- DeepSeek-Coder-V2-Lite-Instruct Q5_K_M

### Confirmed so far
- ik GPU sweep completed for all three models on dual RTX 3060.
- ik CPU sweep completed on 5-thread CPU node.
- llama.cpp GPU baseline completed for both Qwen models; DeepSeek baseline reached the final prompt point before host connectivity interruption.

### Preliminary throughput snapshots (llama.cpp GPU baseline)

| Model | TG32 @ p512 | TG32 @ p32768 |
|---|---:|---:|
| Qwen3-30B-A3B Q4_K_M | 96.64 | 94.02 |
| Qwen3-Coder-30B-A3B-Instruct Q4_K_M | 95.90 | 92.36 |
| DeepSeek-Coder-V2-Lite-Instruct Q5_K_M | 28.35 | pending confirmation |

### ik sweep behavior note
DeepSeek on ik dual-GPU showed clear fill degradation in sweep mode, roughly from ~111 tok/s at low fill to ~65 tok/s near full fill at ctx 32768.

See detailed run log and closure checklist:
- `experiments/2026-02-12-ik-llama-cpp-vs-main-preliminary.md`

---

---

## 5) BeeLlama.cpp — Qwen3.6-27B Q5_K_S + DFlash (RTX 3090)

BeeLlama.cpp is a llama.cpp fork with DFlash speculative decoding. Benchmarks measured via live server API (`/v1/chat/completions`), not `llama-bench`. Model finishes naturally (`stop`). See [backend docs](docs/backend-beellama.md) and [cutover experiment](experiments/2026-06-19-beellama-dflash-cutover.md).

**Config:** Q5_K_S target + Q4_K_M DFlash drafter, `-ub 512`, `q5_0/q4_1` KV cache, `--reasoning on`, 160K context, RTX 3090 24 GB.

### Decode Speed by Workload

| Workload | Decode tok/s | Draft Acceptance | Output tokens | Speedup vs baseline |
|----------|-------------:|-----------------:|--------------:|-------------------:|
| Structured JSON (15 employees) | **80.2** | 33.3% | 3,792 | **2.17x** |
| Code (Python doubly-linked list) | **75.0** | 30.7% | 4,369 | **2.03x** |
| Free-form prose (essay) | **39.6** | 12.8% | 2,444 | **1.07x** |
| Baseline (no DFlash, from upstream) | ~37 | — | — | — |

DFlash is strongest on structured, repetitive output (JSON keys, code indentation, method signatures). Free-form prose sees minimal gain — creative text is hard to predict.

### Prefill Speed by Context Size

Unique prompts (no cross-request caching), 0 cached tokens, small decode (32 tokens) to isolate prefill:

| New Prompt Tokens | Prefill Speed | Time |
|-----------------:|-------------:|-----:|
| 19,280 | **791 tok/s** | 24.4s |
| 39,981 | **742 tok/s** | 53.9s |
| 81,383 | **618 tok/s** | 131.8s |

Prefill peaks around 19-20K tokens and degrades gracefully as context grows. DFlash does not accelerate prefill (decode-only optimization). For comparison, BeeLlama's published benchmark shows ~1229 tok/s at 20K with `--reasoning off` — our `--reasoning on` adds overhead but enables richer drafter context.

### Historical: Mainline llama.cpp (Q4_K_M, v616)

For reference, the previous mainline llama.cpp setup on the same hardware:

| Profile | PP @ 512 | PP @ 2048 | TG @ 128 | Context |
|---------|---------:|----------:|---------:|--------:|
| Default (131K, q8_0/f16) | 711 | 831 | 24.6 | 131K |
| Long context (204K, q8_0/q8_0) | 688 | 808 | 24.1 | 204K |

---

## Methodology (comparison discipline)

Comparison quality depends more on discipline than on raw throughput numbers. Keep benchmark flags stable (`-b/-ub`, `-ctk/-ctv`, `-fa`, `-ts`, `-ncmoe`) and use `llama-bench` for both prefill and generation throughput.

For chat-level comparisons, keep prompt set fixed and compare the same three dimensions every time: total runtime, token volume, and effective tok/s. Do not mix profile classes or runtime families into one aggregate claim without explicit labels.

> Results are hardware-specific and should be treated as comparative guidance, not universal constants.
