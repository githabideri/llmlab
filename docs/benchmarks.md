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
