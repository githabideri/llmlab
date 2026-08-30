# Benchmarks

How we benchmark. This page keeps the **durable method** — flags and comparison discipline. Specific numbers are point-in-time snapshots and live in [reports/](../reports/README.md).

## Context sweep (`llama-bench`)

The standard sweep measures prefill and generation across growing context. Keep the flag set stable between runs so results are comparable (`-fa`, `-ctk/-ctv`, `-ts`, `-ncmoe`, `-b/-ub`).

Illustrative run (Qwen3-Coder-Next REAP-40B A3B Q2_K_XL, dual 3060, `-fa 1 -ctk q8_0 -ctv q4_0 -ncmoe 0 -ts 14/10 -b 128 -ub 64`):

| Prefill context | tok/s |
|---:|---:|
| 32k | 180.82 |
| 64k | 175.98 |
| 96k | 171.39 |
| 128k | 166.92 |

Generation (tg256) ~20.72 tok/s.

## Methodology (comparison discipline)

Comparison quality depends more on discipline than on raw throughput. Keep benchmark flags stable and use `llama-bench` for both prefill and generation. For chat-level comparisons, keep the prompt set fixed and compare the same three dimensions every time: total runtime, token volume, and effective tok/s. Do not mix profile classes or runtime families into one aggregate claim without explicit labels.

### The cache trap: every request needs a fresh nonce (2026-08)

Both serving stacks silently cache repeated prompts, which turns a "cold" benchmark into a warm one:

- **vLLM** has automatic prefix caching — a repeated fixture hits the KV cache and prefills at 10×+ the true cold rate (a "13K tok/s prefill" row in the 2026-08-30 campaign was a cache hit, not a regime).
- **llama.cpp** `--cache-prompt` does the same for repeated `/completion` prompts ("60× faster prefill" warm rows).

Rule: prepend a **unique nonce** to every prompt that must count as cold; repeat a fixture only when measuring warm behavior deliberately. Report cold and warm as separate rows — never average them.

### Decode rate under speculative decoding: usage tokens, not chunks (2026-08)

With MTP/DFlash, each streamed chunk carries multiple accepted tokens, so "tokens per chunk × chunks/sec" overstates the rate 3–5×. Compute decode t/s = **exact `completion_tokens` from the `/usage` chunk ÷ wall time from first token to end**. For llama.cpp, prefer the server's slot `print_timing` lines (it reports generation tokens directly); the server log's `draft acceptance` line is the MTP acceptance evidence.

### Proving GPU residency: PCIe counters, not timing (2026-08)

"The model fits in VRAM" is a bus claim — a streaming model decodes at a plausible steady rate too. Capture `nvidia-smi dmon -d 1 -s pumt` (columns include `rxpci`/`txpci` in MB/s; on some driver/LXC builds `-t`/`-T` per-line timestamps are absent, so write `REQ-START <epoch> <tag>` / `REQ-END …` markers to a side file and align rows by 1-second index). During decode: **~0–25 MB/s RX ⇒ resident; ≥1 GB/s ⇒ streaming from host RAM**. Always run the non-resident control (e.g. `--n-cpu-moe`) on the same card/model for the contrast. A weak PCIe link is usually *not* the LLM bottleneck — dual-GPU prefill on consumer cards is compute-bound (SM ~100%, PCIe well under link cap).

### First requests are not steady state (2026-08)

After server-up, send a tiny warmup request before the measured window (cudagraph capture, Triton JIT, FlashInfer plan build). A first long request can show a one-off JIT dip (seen on the 3090 vLLM node: 17.8 t/s first, 73 steady) — never quote it as the regime number. vLLM's `/health` returns **HTTP 200 with an empty body** (llama.cpp returns `{"status":"ok"}`) — match the status code, not the body.

> Results are hardware-specific — comparative guidance, not universal constants.

## Where the snapshots live

| Snapshot | Report |
|----------|--------|
| gpt-oss-20b on single 3060 | [2026-02-03](../reports/2026-02-03-gpt-oss-20b.md) |
| Nemotron profile (fixed 5-task set) | [2026-02-12 ABC](../reports/2026-02-12-nemotron-abc-executive-summary.md) |
| ik_llama.cpp vs llama.cpp | [2026-02-12](../reports/2026-02-12-ik-llama-cpp-vs-main-preliminary.md) |
| BeeLlama DFlash (Qwen3.6-27B) | [2026-06-19 cutover](../reports/2026-06-19-beellama-dflash-cutover.md) |
| Qwen3.6-35B-A3B dual 3060 (full config matrix, ~45 runs) | [2026-08-27](../reports/2026-08-27-qwen3.6-35b-a3b-dual-3060-optimization.md) |
| llama.cpp `-ub` tuning (single 3060, MoE+MTP) | [2026-08-28](../reports/2026-08-28-llama-cpp-ubatch-moe-single-gpu.md) |
| 35B on one 12 GB 3060 (residency proof) + 27B on the 3060 pair (ctx/concurrency/wall power) | [2026-08-30](../reports/2026-08-30-dual-3060-35b-squeeze-27b-node.md) |
