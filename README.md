# llmlab

Personal lab notes on running local LLMs on consumer GPUs: working configs, models that fell short, and numbers from real serving.

## Current setup

The primary box runs **2× RTX 3090 24 GB (48 GB)** on an **AMD Ryzen 5 5600X**, plus a single-3060 backup box that doubles as an MTP inference endpoint, a laptop for iGPU/Vulkan experiments, and a secondary box with a 3060 and two Pascal cards. Specs, PCIe topology, and the upgrade history behind several of the results are under [docs/hardware](docs/hardware/README.md); serving layout under [architecture](docs/architecture.md), day-to-day operations under [runbook](docs/runbook.md).

## Currently serving

| Model | Quant | GPU | Context | Backend |
|-------|-------|-----|---------|---------|
| [Qwen3.8-27B](models/qwen3.8-27b-rtx3090.md) | W4A16-AutoRound | 2× RTX 3090 (vLLM TP2) | 256K | vLLM 0.28.0, MTP k=3, fp8 KV, 8192 batched, vision — since 2026-09-08 |
| [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b.md) | Q4_K_XL (interim) | 1× RTX 3060 (secondary box) | 128K | llama.cpp + MTP + vision — interim since the 3060 pair left the primary box 2026-09-08 |

The quants are the highest-quality that still leave 100K+ context headroom; each model card shows the full comparison with exact sizes. The 35B's home on the primary box (dual-3060 router, which also carried an abliterated Qwen3.8-27B on demand) was dismantled 2026-09-08 in favour of the second 3090 — see [models/legacy](models/legacy/) for that card. Per-model write-ups and every model tested live in [models/](models/README.md).

## Tools

- **[hub/](hub/README.md)** — live monitoring and control for the inference fleet: per-model generation (t/s) and prompt-processing (pp/s) throughput, vLLM latency percentiles, GPU telemetry, model load/unload, and a token-authed JSON API for agents.

  [![LLM Hub fleet overview: per-GPU state and live model throughput](hub/screenshots/llm-hub-overview-crop.png)](hub/README.md)

- **[web/](web/README.md)** — benchmark UI (legacy front-end for the context-ladder harness; current campaigns use per-campaign scripts).
- **[scripts/](scripts/)** — small tooling (logged `llama-bench`, model-info fetcher); the older context-ladder harness is under `scripts/legacy/`.
- **[agents/](agents/)** — agent-facing operational tooling (pi-style skills): [llmlab-bench](agents/llmlab-bench/SKILL.md) — how to *run* benchmark campaigns (pre-boot budgeting, version/flag drift, data discipline), the operational companion to [docs/benchmarks.md](docs/benchmarks.md).

## What the lab measures

- **Local model serving on consumer GPUs** — real inference performance and usability across dense, MoE, hybrid, and other architectures. Dense 27B-class models, small-active MoE, and hybrid SSM models are all in play; which one wins depends on the workload.
- **Serving performance** — generation throughput (t/s), prompt processing (pp/s), latency, context growth, prefix caching, concurrency, and speculative decoding (MTP).
- **Memory and placement** — quantization, VRAM residency, KV/compute buffers, PCIe traffic, and multi-GPU placement on heterogeneous cards ([guide](docs/multi-gpu-model-placement.md)).
- **Agent workloads** — models evaluated doing multi-step tool work (search → fetch → analyze → file ops), which stresses context growth and prompt caching far beyond short synthetic prompts.

Numbers here come from both controlled benchmarks and live serving. Placement work measures PCIe traffic and memory residency alongside throughput, and repeated-prompt tests use unique nonces where cache reuse would distort the result — which is why numbers in this repo often differ from a typical `llama-bench` screenshot ([methodology](docs/benchmarks.md)).

## Findings

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

- **The dense 27B went from ~35 t/s to 96–118 t/s on one 3090** — the clearest single-model arc in this lab. Qwen3.8-27B cut over on stock llama.cpp (Q4_K_M + MTP) at 34.7–37 tok/s sustained; the move to vLLM with the W4A16-AutoRound quant, MTP k=3, and fp8 KV took decode to 96–118 tok/s (prefill up to ~1,050 tok/s, 3/3 needles at ~155K). Qwen3.8-27B holds the primary slot because it was the best available model at its 2026-08 cutover; the 35B-A3B keeps the multi-slot and vision cases ([model card](models/qwen3.8-27b-rtx3090.md)).
- **MTP draft depth tops out at 2–3** — deeper drafts cost VRAM without measurable gain on 12 GB cards ([2026-08-27](reports/2026-08-27-qwen3.6-35b-a3b-dual-3060-optimization.md)).
- **Two 3060s serve a 27B dense model at ~80% of 3090 speed for the same wall power** — but only as a single-user node: the KV pool (126K tokens) fits 1.9× a 64K context, and 4× 16K contexts collapse per-request decode to ~16 t/s ([2026-08-30](reports/2026-08-30-dual-3060-35b-squeeze-27b-node.md)).
- **On the dual-3090 vLLM, the levers rank MTP ≫ CUDA graphs ≫ scheduler budget** — 2.1× from MTP k=3, 3.4× from piecewise graphs vs `--enforce-eager`, ~10% from the 8192 batched-token budget at 16K; MTP acceptance was stable (1.41–1.45 accepted tokens/verify) across the tested context depths. The old "slower at 16K than 2K" behaviour was context-dependent degradation plus a latency-biased budget, not a defect ([2026-09-10](reports/2026-09-10-dual3090-overnight-campaign.md)).
- **A 262K production context is safe but TTFT-bound** — 7 preemptions in a 30-minute 640K staged-admission soak, but under that deliberately saturated workload a 43K-token request observed 14–15 min *end-to-end* TTFT (queueing + preemption + prefill; not prefill speed). For agent traffic the context size is a latency decision, not a capacity one ([2026-09-10](reports/2026-09-10-dual3090-overnight-campaign.md)).

### Specialized inference (non-LLM)

- **2 GB Pascal cards run WhisperX at 10× realtime** — CTranslate2's int8 execution path makes the large-v3-turbo ASR model viable on CC 6.1 hardware with no Tensor Cores. The dual-GPU config is about avoiding CPU fallback (align on second card), not parallelism. See [2026-09-06 report](reports/2026-09-06-whisperx-pascal-dual-gpu-benchmark.md).

### Benchmark traps

- **Repeated prompts are warm prompts** — vLLM prefix caching and llama.cpp `--cache-prompt` both made a llama.cpp node look 2.7× faster than the 3090 in one campaign until every request got a unique nonce ([methodology](docs/benchmarks.md)).
- **"Complete" must be verified, not assumed** — a 256K ladder cell once recorded "complete" in 1 second (exit code 0, request died instantly); a swap-pressure canary with a 2 KB threshold sat below the machine's noise floor and invalidated the *winning* cell; and vLLM's `/health` is an **empty 200**, so body-grepping health checks silently report everything as down. Gates need noise-floor thresholds, plausibility checks (did the tokens actually move?), and code-based probes ([2026-09-10](reports/2026-09-10-dual3090-overnight-campaign.md)).
- **Dry-runs test control flow, not data-plane assumptions** — the 2026-09-10 package passed a full dry-run and then spent an hour fighting a wrong venv path, a quoting layer that ate JSON args, and a non-writable log directory, because none of those are visible until a real process touches the real filesystem as the real user ([2026-09-10](reports/2026-09-10-dual3090-overnight-campaign.md)).

## Notable experiments (not serving)

| Model | Quant | GPU | Result |
|-------|-------|-----|--------|
| [Qwen3.8-Flash-Next (Qwen4Exp: 125 B total / 6 B active, 51 B PLE table)](models/qwen3.8-flash-next.md) | Q2_K_XL | 3× (12+12+24 GB) | **30.2 t/s sustained decode**, ~390 pp/s (warm cache), 35.3 t/s with MTP (+17 %, single run, provisional). The fitter's selective expert spill beat all manual placement; on hold pending a production decision ([2026-09-02 report](reports/2026-09-02-qwen4exp-flash-next-three-gpu-campaign.md)) |
| [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b.md) | UD-IQ2_XXS | 1× 3060 | 2-bit fully resident on one 12 GB card: 43–81 t/s at 0.02 GB/s PCIe — the residency-proof baseline ([2026-08-30 report](reports/2026-08-30-dual-3060-35b-squeeze-27b-node.md)) |

## Repository map

- [models/](models/README.md) — per-model write-ups: architecture, quant rationale, speed vs context, agentic results, known issues. Production models are listed at the top; every other card is a frozen test record.
- [docs/](docs/README.md) — methodology and reference, indexed by purpose and status: [model placement](docs/multi-gpu-model-placement.md), [benchmarking](docs/benchmarks.md), [KV-cache sizing](docs/kv-cache-sizing.md), [architecture](docs/architecture.md), [runbook](docs/runbook.md), [unit reference](docs/systemd.md), [hardware fleet](docs/hardware/README.md); frozen fork docs live under [docs/legacy/](docs/legacy/).
- [reports/](reports/README.md) — date-stamped investigations and deployments (snapshots, no maintenance).
- [benchmarks/](benchmarks/README.md) — benchmark harnesses (the March 2026 OpenClaw ladder is frozen under `benchmarks/legacy/`; future agent benchmarks target pi).
