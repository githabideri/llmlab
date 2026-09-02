# llmlab

Personal lab notes on running local LLMs on consumer GPUs: working configs, models that fell short, and numbers from real serving.

## The setup

The primary box runs **1× RTX 3090 24 GB + 2× RTX 3060 12 GB (48 GB)** on an **AMD Ryzen 5 5600X** — the 3090 came in early March 2026, and the CPU moved from a 7th-gen Intel i5-7400 to the 5600X in July 2026. Full specs, per-GPU deployment, and the rest of the fleet are under [docs/hardware](docs/hardware/README.md); serving layout and day-to-day operations under [architecture](docs/architecture.md) and [runbook](docs/runbook.md).

## What this is about

- **MoE and hybrid models** — the sweet spot for interactive use on limited VRAM: small active parameters keep generation fast while large total parameters keep quality up. Dense models are on the table now that 48 GB is available.
- **Agentic tool-calling** — models driving multi-step tool chains (search → fetch → analyze → file ops), not just chat.
- **Real serving metrics** — `llama-bench` at empty context is only a starting point; prompt cache, thinking tokens, and growing context change the numbers. Both are measured.
- **Heterogeneous consumer-GPU inference** — not just "run on N cards": measuring where tensors actually live, what crosses PCIe, what spills to RAM/SSD, and what that costs — across automatic fitter placement, layer/tensor split, and compute-buffer pressure ([guide](docs/multi-gpu-model-placement.md)).

## Currently serving

| Model | Quant | GPU | Context | Backend |
|-------|-------|-----|---------|---------|
| [Qwen3.8-27B](models/qwen3.8-27b-rtx3090.md) | W4A16-AutoRound | 1× RTX 3090 | 160K | vLLM 0.27.1, MTP k=3, text-only (stock llama.cpp Q4_K_M+MTP kept as rollback) |
| [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b.md) | UD-IQ4_XS | 2× RTX 3060 (tensor-split 50/50) | 256K ×2 | llama.cpp + vision |

The quant is the highest-quality that still leaves 100K+ context headroom; each model card shows the full comparison with exact sizes. Per-model write-ups and every model tested live in [models/](models/README.md).

**Notable lab results (not serving):**

| Model | Quant | GPU | Result |
|-------|-------|-----|--------|
| [Qwen3.8-Flash-Next (Qwen4Exp: 125 B total / 6 B active, 51 B PLE table)](models/qwen3.8-flash-next.md) | Q2_K_XL | 3× (12+12+24 GB) | **30.2 t/s sustained decode** / ~390 pp/s (warm cache) / 35.3 t/s with MTP (+17 %, single run — provisional) — the fitter's selective expert spill beat all manual placement; on hold pending a production decision ([2026-09-02 report](reports/2026-09-02-qwen4exp-flash-next-three-gpu-campaign.md)) |
| [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b.md) | UD-IQ2_XXS | 1× 3060 | 2-bit fully resident on one 12 GB card: 43–81 t/s at 0.02 GB/s PCIe — the residency-proof baseline ([2026-08-30 report](reports/2026-08-30-dual-3060-35b-squeeze-27b-node.md)) |

## What actually matters

### Placement & residency

- **Placement comes before split tuning.** On current llama.cpp, the automatic fitter (no placement flags at all) selectively spills individual weight tensors and can beat hand-tuned `-ngl`/`--tensor-split` — on Qwen3.8-Flash-Next it raised sustained decode from ~21 to ~30 t/s on the mixed 12+12+24 GB box ([guide](docs/multi-gpu-model-placement.md), [2026-09-02](reports/2026-09-02-qwen4exp-flash-next-three-gpu-campaign.md)). Manual placement stays valuable when the fitter aborts or fully-resident homogeneous GPUs want tensor parallelism.
- **Row split has consistently been much slower than layer (or tensor) on this lab's PCIe-only systems** — across every architecture tested; the exact margin varies by test (~2.5× in the 2026-03 graph-mode work, 7.5–30× in the 2026-08 two-card test), and the [guide](docs/multi-gpu-model-placement.md) carries the numbers. The one exception noted there: tensor mode can *win* when the model is fully resident on equal cards.
- **`output.weight` lands on the last GPU.** In split-mode layer the output projection (~1+ GB) is hardcoded to the last GPU, creating asymmetric VRAM pressure that must be balanced with tensor-split ratios ([details](docs/multi-gpu-model-placement.md)).
- **"It fits in VRAM" is a bus claim — prove it with PCIe counters.** The 2-bit 35B sits fully resident on one 12 GB 3060 (43–81 t/s) with ~0.02 GB/s PCIe in decode, while its CPU-MoE twin streams 5.9 GB/s and runs 2× slower; decode timing alone can't tell the two apart ([2026-08-30](reports/2026-08-30-dual-3060-35b-squeeze-27b-node.md)).
- **A weak link isn't automatically the decode bottleneck.** The chipset-x4 3060 (PCIe 4.0, i.e. 32 GB/s — the generation matters) ran within a few percent of its x8 sibling at equal residency (2026-09) — with weights resident, layer-style placement barely moves data per token. That doesn't generalize to prefill-heavy or tensor-parallel workloads.

### Context & KV

- **`--parallel N` shrinks compute buffers.** More slots mean smaller per-slot compute buffers, which frees VRAM for KV cache — but per-slot context shrinks proportionally.
- **Hybrid models use almost no KV cache.** Qwen3.6-35B-A3B has 40 layers but only 10 full-attention; the rest are linear attention with zero KV cache, so 128K context uses under 1 GB at q8_0/q4_0 ([per-model math](docs/kv-cache-sizing.md)).
- **Ubatch is often an untuned knob** — raising `-ub` toward the VRAM limit gave +20–24% prompt processing on CPU-offloaded MoE with no generation penalty ([2026-08-28](reports/2026-08-28-llama-cpp-ubatch-moe-single-gpu.md)).
- **Throughput degrades as context fills** — and differently by architecture:

| Arch | @0 | @16K | @32K | @64K | @64K drop |
|------|----|------|------|------|-----------|
| Mamba-2 (Nemotron) | 96 | 85 | 72 | 55 | −42% |
| MLA (GLM) | 71 | 54 | 45 | 33 | −53% |
| GQA (Qwen3) | 99 | 39 | 24 | 13 | −87% |

Mamba-2 holds up on its constant-time-attention promise; traditional GQA falls off a cliff. Real serving also runs 28–36% slower than `llama-bench` under load, and one long session recovered ~2× after context compaction.

### Serving & speculation

- **MTP draft depth tops out at 2–3** — deeper drafts cost VRAM without measurable gain on 12 GB cards ([2026-08-27](reports/2026-08-27-qwen3.6-35b-a3b-dual-3060-optimization.md)).
- **Two 3060s serve a 27B dense model at ~80% of 3090 speed for the same wall power** — but only as a single-user node: the KV pool (126K tokens) fits 1.9× a 64K context, and 4× 16K contexts collapse per-request decode to ~16 t/s ([2026-08-30](reports/2026-08-30-dual-3060-35b-squeeze-27b-node.md)).

### Benchmark traps

- **Repeated prompts are warm prompts** — vLLM prefix caching and llama.cpp `--cache-prompt` both made a llama.cpp node look 2.7× faster than the 3090 in one campaign until every request got a unique nonce ([methodology](docs/benchmarks.md)).

## Where things live

- [models/](models/README.md) — per-model write-ups: architecture, quant rationale, speed vs context, agentic results, known issues.
- [docs/](docs/README.md) — methodology and reference, indexed by purpose and status: [model placement](docs/multi-gpu-model-placement.md), [benchmarking](docs/benchmarks.md), [KV-cache sizing](docs/kv-cache-sizing.md), [architecture](docs/architecture.md), [runbook](docs/runbook.md), [unit reference](docs/systemd.md), [hardware fleet](docs/hardware/README.md); frozen fork docs live under [docs/legacy/](docs/legacy/).
- [reports/](reports/README.md) — date-stamped investigations and deployments (snapshots, no maintenance).
- [benchmarks/](benchmarks/README.md) — benchmark harnesses (the March 2026 OpenClaw ladder is frozen under `benchmarks/legacy/`; future agent benchmarks target pi).
- [scripts/](scripts/) — small tooling (logged `llama-bench`, model-info fetcher); the older context-ladder harness is under `scripts/legacy/`.
- [web/](web/README.md) — FastAPI + htmx dashboard for running and monitoring benchmarks.
