# llmlab

Personal lab notes on running local LLMs on consumer GPUs: working configs, models that fell short, and numbers from real serving.

## The setup

The primary box runs **1× RTX 3090 24 GB + 2× RTX 3060 12 GB (48 GB)** on an **AMD Ryzen 5 5600X** — the 3090 came in early March 2026, and the CPU moved from a 7th-gen Intel i5-7400 to the 5600X in July 2026. Full specs, per-GPU deployment, and the rest of the fleet are under [docs/hardware](docs/hardware/README.md); serving layout and day-to-day operations under [architecture](docs/architecture.md) and [runbook](docs/runbook.md).

## What this is about

- **MoE and hybrid models** — the sweet spot for interactive use on limited VRAM: small active parameters keep generation fast while large total parameters keep quality up. Dense models are on the table now that 48 GB is available.
- **Agentic tool-calling** — models driving multi-step tool chains (search → fetch → analyze → file ops), not just chat.
- **Real serving metrics** — `llama-bench` at empty context is only a starting point; prompt cache, thinking tokens, and growing context change the numbers. Both are measured.
- **Multi-GPU without NVLink** — tensor-split tuning and compute-buffer analysis for squeezing context and parallelism out of consumer GPUs ([guide](docs/multi-gpu-tensor-split.md)).

## Currently serving

| Model | Quant | GPU | Context | Backend |
|-------|-------|-----|---------|---------|
| [Qwen3.8-27B](models/qwen3.8-27b-rtx3090.md) | Q4_K_M | 1× RTX 3090 | 160K | llama.cpp (stock) + MTP + vision |
| [Qwen3.5-35B-A3B](models/qwen3.5-35b-a3b.md) | UD-IQ3_S | 2× RTX 3060 (tensor-split 52/48) | 512K ×2 | llama.cpp + vision |

The quant is the highest-quality that still leaves 100K+ context headroom; each model card shows the full comparison with exact sizes. Per-model write-ups and every model tested live in [models/](models/README.md).

## What actually matters

- **`-sm layer` beats `-sm row`.** On PCIe multi-GPU without NVLink, split mode matters more than the model; `-sm layer` gives ~2.5× the throughput of `-sm row`.
- **`output.weight` lands on the last GPU.** In split-mode layer the output projection (~1+ GB) is hardcoded to the last GPU, creating asymmetric VRAM pressure that must be balanced with tensor-split ratios ([details](docs/multi-gpu-tensor-split.md)).
- **`--parallel N` shrinks compute buffers.** More slots mean smaller per-slot compute buffers, which frees VRAM for KV cache — but per-slot context shrinks proportionally.
- **Hybrid models use almost no KV cache.** Qwen3.5-35B-A3B has 40 layers but only 10 full-attention; the rest are linear attention with zero KV cache, so 128K context uses under 1 GB at q8_0/q4_0 ([per-model math](docs/kv-cache-sizing.md)).
- **Throughput degrades as context fills** — and differently by architecture:

| Arch | @0 | @16K | @32K | @64K | @64K drop |
|------|----|------|------|------|-----------|
| Mamba-2 (Nemotron) | 96 | 85 | 72 | 55 | −42% |
| MLA (GLM) | 71 | 54 | 45 | 33 | −53% |
| GQA (Qwen3) | 99 | 39 | 24 | 13 | −87% |

Mamba-2 holds up on its constant-time-attention promise; traditional GQA falls off a cliff. Real serving also runs 28–36% slower than `llama-bench` under load, and one long session recovered ~2× after context compaction.

## Where things live

- [models/](models/README.md) — per-model write-ups: architecture, quant rationale, speed vs context, agentic results, known issues.
- [docs/](docs/) — guides and reference: [multi-GPU tensor-split](docs/multi-gpu-tensor-split.md), [KV-cache sizing](docs/kv-cache-sizing.md), [architecture](docs/architecture.md), [runbook](docs/runbook.md), [troubleshooting](docs/troubleshooting.md), [hardware fleet](docs/hardware/README.md).
- [reports/](reports/README.md) — date-stamped investigations and deployments (snapshots, no maintenance).
- [benchmarks/](benchmarks/README.md) — benchmark harnesses (the OpenClaw ladder is legacy; future agent benchmarks target pi).
- [scripts/](scripts/) — context-sweep benchmarking, model info fetcher, server start scripts.
- [web/](web/README.md) — FastAPI + htmx dashboard for running and monitoring benchmarks.
