# llmlab

A public, self‑contained notebook of a small-but-serious local LLM lab: configs, benchmarks, runbooks, and reproducible experiments.

## Why this repo
- Capture working configs (before they get lost)
- Keep benchmarks reproducible
- Share a lightweight, practical setup for running local models

## Hardware (as tested)
- **GPUs:** 2× RTX 3060 12GB
- **RAM:** ~26 GiB available to inference container
- **Storage:** 1 TB SSD for models, 250 GB NVMe for OS/root (ZFS)
- **OS:** Linux + ZFS

## Current best config (as of 2026‑02‑08)
- **Model:** Qwen3‑Coder‑Next REAP‑40B A3B **Q2_K_XL** (GGUF)
- **llama.cpp:** Flash‑Attn (all‑quants), **K=q8_0**, **V=q4_0**
- **Params:** `n-cpu-moe 0`, `tensor-split 14/10`, `ctx 128k`, `b/ub 128/64`
- **Perf:** prefill @128k **~166.9 tok/s**, `tg256` **~20.7 tok/s`

See **docs/benchmarks.md** for the full sweep.

## Repo map
```
README.md
LICENSE
/docs
  overview.md
  architecture.md
  runbook.md
  benchmarks.md
  troubleshooting.md
/experiments
  README.md
  2026-02-07-qwen3-q2-fa-vram.md
  2026-02-08-qwen3-q2-ctx-sweep.md
/scripts
  start_qwen3_q2.sh
  bench_ctx_sweep.sh
/config
  llama-server.example.env
```

## Safety note
This repo is **public**. Do not commit:
- hostnames / IPs
- SSH keys
- usernames / home paths
- private prompts or logs

Keep configs and paths **generic** and documented.
