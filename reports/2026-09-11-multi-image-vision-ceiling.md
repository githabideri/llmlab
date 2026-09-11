# Multi-Image Vision Ceiling — Qwen3.8-27B on 2× RTX 3090

**Status:** research + campaign design, written for **external review** (the tail of this doc is an open question to the reader, not a settled result). The campaign itself is not run yet.

## Goal

Find the **true multi-image processing ceiling** of the 27B VLM on a 2× RTX 3090 (vLLM TP2) and turn that into a defensible production config: maximum KV context while keeping vision reliable at **more than 16 images** per request. This doc records (1) what we could verify about the model's own vision setup, (2) what others have published about the same failure mode, and (3) the campaign design + decision matrix we intend to run, with the open questions we'd like feedback on.

## Setup

- **Model:** `Qwen3.8-27B`, W4A16-AutoRound quant, served by **vLLM 0.28.0, TP2** on a dedicated LXC with two RTX 3090 (24 GB each, capped 250 W). Hybrid SSM/GDN architecture (17 of 66 layers are full attention). Same patch stack as the public [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090) recipe.
- **The incident that started this:** at `--gpu-memory-utilization 0.95` the idle box showed ~99% of each card used (only ~265 MiB genuinely free per card after CUDA-context/graph/fragmentation overhead). A single request with **64 images at 1024×1024 (67 Mpx total)** then crashed the engine: the vision-encoder activations plus the GDN/Mamba recurrent path pushed a routine 48 MiB allocation past the remaining headroom → EngineCore OOM, systemd restart. A slightly smaller **64×768×1024 (50 Mpx)** had survived at the same 0.95 — so the cliff is a memory-budget cliff, not an image-count limit.

The immediate fix (already in production): `--gpu-memory-utilization 0.93` + explicit image-size hints. The rest of this doc is about finding the real ceiling and choosing the config deliberately rather than by shrinking the image size until it stops crashing.

## What we verified about *our* model (not assumed)

We inspected the shipped `config.json` / `processor_config.json` rather than trusting blog posts, because the numbers differ between Qwen vision generations:

| Fact | Value | Source |
|---|---|---|
| `model_type` / arch | `qwen3_5` / `Qwen3_5ForConditionalGeneration` | `config.json` |
| vision tower | `qwen3_5_vision`, depth 27, hidden 1152, out 5120 | `vision_config` |
| vLLM processor | **`Qwen3VLProcessor`** | `processor_config.json` |
| base patch / merge | `patch_size 16`, `spatial_merge_size 2` → **32×32 px per token** | `vision_config` |
| **default** size bounds | `shortest_edge 65536`, **`longest_edge 16,777,216`** | `Qwen3VLProcessor` default |
| our override | `shortest_edge 65536`, `longest_edge 2,097,152` | launch script |

**Token-per-pixel is 32×32 = 1024 px/token**, which reproduces our live measurements exactly: 768×1024 → 768 tokens, 1024×1024 → 1024, 1448×1448 (2 Mpx) → 2048. This matches the Qwen3-VL "32×32" figure (vs Qwen2-VL's 14×14→28×28), so the **tokenization layer is confirmed Qwen3-VL-class** and those findings transfer to us. Two consequences fall out of the default bounds:

- The **default** `longest_edge` is **16 Mpx = 16,384 tokens per image** — the "unreasonably large" reservation documented in [vLLM issue #20123](https://github.com/vllm-project/vllm/issues/20123) (`16384×28×28 ≈ a 3584×3584 image`). Our `2 Mpx` override cuts that reservation **8×** (16384 → 2048 tokens/image). So `longest_edge` is *the* knob that simultaneously sets the real max image size **and** the memory vLLM profiles against.
- The **Qwen2 (28×28) vs Qwen3 (32×32) distinction is a real trap**: any config copied from a Qwen2-VL blog is ~15% off for a Qwen3-class model. We measured instead of assuming.

**What we could NOT infer and must measure:** the ViT *activation memory* per image (our tower is depth 27 / hidden 1152 — different from the models in the references), and the actual OOM path (ours dies in the **GDN/Mamba** recurrent allocation, which the Qwen3-VL writeups don't cover). Those are exactly what the campaign measures.

## What others have published (same failure mode)

- **The OOM is a memory-*profiling* reservation, and it's driven by the size hints.** vLLM's [Conserving Memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory/) page is explicit: *"The size hints affect memory profiling only. They shape the dummy inputs used to compute reserved activation sizes. They do not change how inputs are actually processed at inference time."* It names the three knobs: `limit_mm_per_prompt={"image": <count>}`, the `{count,width,height}` size hints, and `mm_processor_cache_gb` (default 4 GiB).
- **The default reservation is absurdly large.** [vLLM #20123](https://github.com/vllm-project/vllm/issues/20123): with only `limit_mm_per_prompt={"image":1}`, the profiler reserved *16,384 tokens/image*, and the reporter notes this *"blocks me from using multiple images as it will cause OOM during memory profiling."* Same class as our crash, seen on Qwen2.5-VL.
- **The profiler can wildly over-reserve.** [vLLM #27706](https://github.com/vllm-project/vllm/issues/27706) (ROCm) reported *"both expecting 256 GB when multimodal activated"* on 2B/3B models — evidence the MM memory model is not always well-calibrated, so measuring beats trusting the estimate.
- **Encoder parallelism is a separate lever.** AMD's [vLLM DP-vision writeup (Jan 2026)](https://rocm.blogs.amd.com/software-tools-optimization/vllm-dp-vision/README.html) describes `--mm-encoder-tp-mode data`: instead of TP-sharding the small ViT, it replicates the encoder on each GPU and load-balances the image batch — *"slightly higher memory usage in exchange for substantially better throughput and latency… if you are running multimodal models with tensor_parallel_size ≥ 4, this optimization deserves your attention."* We are TP2 and memory-tight, so we expect the *memory* side to bite; it's a throwaway variable to measure, not a prod default.
- **Closest analogs to our box.** (a) [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090) runs this exact 27B on a single 3090 — but **text-only**, which is how most 27B-on-24GB setups fit; it shows how little headroom the 27B leaves. (b) A [24 GB 27B recipe (May 2026)](https://dev.to/xreyrobertibm/qwen36-27b-vllm-hermes-on-24gb-vram-may-2026-recipe-5452) explicitly reaches for `--language-model-only` because the vision path is what doesn't fit. (c) A real multi-image Qwen3.6 config in the wild: `--mm-processor-kwargs '{"min_pixels": 313600, "max_pixels": 7840000}'` → 560²–2800² px, up to 100×100 tokens/image ([HF discussion](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/discussions/36)).

The takeaway that the references *do* let us take: **the binding constraint for multi-image is encoder-activation memory, a function of (image count × per-image pixels) and of the profiling reservation — not the KV cache.** The references give us the mechanism and the knobs; only our own measurement gives the numbers for this specific tower.

## The memory knobs (the full set)

| Knob | What it controls | Default |
|---|---|---|
| `--gpu-memory-utilization` | total headroom the engine may use → sets KV pool size | 0.90 here |
| `--limit-mm-per-prompt {"image":{"count":N}}` | hard cap on images per prompt (rejects beyond N) | 16 here |
| `--mm-processor-kwargs {"size":{"shortest_edge":S,"longest_edge":L}}` | real max image pixels **and** the profiling reservation | 65536 / 2097152 here |
| `mm_processor_cache_gb` | size of the shared-memory image cache | 4 GiB |
| `--mm-encoder-tp-mode` | `weights` (TP-shard ViT) vs `data` (replicate + batch-parallel) | `weights` here |

## The campaign design (3 axes, not 1)

The old plan was a 1-D count ladder. The research shows that's wrong — the ceiling is a **2-D surface** (count × pixel size) modulated by a **headroom** axis:

| Axis | Values | Rationale |
|---|---|---|
| **A — image envelope** | count {8,16,32,64} × per-image px {0.79 M, 1 M, 2 M} → 6 M–128 Mpx total | this is what OOMs the encoder |
| **B — headroom** | `--gpu-memory-utilization` {0.90, 0.93, 0.95} | multiplier on how far the encoder may spike before OOM |
| **C — encoder parallelism** (throwaway only) | `--mm-encoder-tp-mode` {weights, data} | measure its memory cost at TP2 |

**Method changes vs a naive sweep** (these are the things that made the first attempt misleading):
- **Set the size hints to match each test image.** Otherwise vLLM profiles against the 16 Mpx default (16,384 tok/image) and OOMs *during profiling*, before it ever reaches the real encoder ceiling — exactly the #20123 trap.
- **Record *how* it fails.** A clean per-request OOM/reject (recoverable) is a fundamentally different production outcome than an EngineCore crash (whole endpoint down for the ~3 min systemd restart). The prod decision depends on this distinction.
- **The controller runs on an independent endpoint**, never the one under test — a crash under test must not take the orchestrator down with it.
- **Capture a clean idle baseline per gpu-util level** (KV pool tokens + nvidia-smi free) *before* any request hits the endpoint, so the numbers are comparable (in-flight requests populate KV + the image cache and inflate the reading).

### Candidate production configs (the campaign measures each)

| ID | gpu-util | count | px/img | KV pool | enc-tp | what it probes |
|---|---|---|---|---|---|---|
| **C1** | 0.93 | 16 | 2 M | ~710 K | weights | **current prod** — the baseline |
| **C2** | 0.93 | 32 | 2 M | ~710 K | weights | double the count at current headroom? |
| **C3** | 0.90 | 32 | 2 M | ~678 K | weights | +headroom → does 32 hold? |
| **C4** | 0.90 | 64 | 1 M | ~678 K | weights | many mid-size images |
| **C5** | 0.95 | 16 | 2 M | ~747 K | weights | max KV; is 16 still safe at tight headroom? |
| **C6** | 0.90 | 32 | 2 M | ~678 K | **data** | does batch-DP help or cost at TP2? |
| **C7** | 0.90 | 32 | 4 M | ~678 K | weights | push resolution instead of count |

*(KV pool at a given gpu-util also depends on the mm config; the 0.95/0.93/0.90 clean-idle pools measured this session were 747,625 / 710,402 / 677,931 tokens at the count-16/2 M config, vs the 776,928-token pool documented for the pre-tuning config in the model card.)*

## The decision matrix (how we'll pick the prod config)

Rank the surviving cells on these goals — the first two are the user's, the rest are reliability criteria the research surfaced:

1. **Max KV context**
2. **Reliable vision at >16 images**
3. **Worst case = a rejected request, NOT an engine crash** (a crash is a ~3-min total outage for every consumer of the endpoint)
4. **TTFT at N images** (the encoder is the interactive bottleneck)
5. **Stable under a concurrent text+image mix** (the Mamba/SSM-state contention case)
6. **Recovery time if it does OOM**
7. **MM/prefix-cache hit under agent-loop reuse** (the real workload is agentic)

**Current expectation** (to be confirmed, not asserted): **C3** (0.90 + 32 images + 2 Mpx) — it trades ~4.5% of the KV pool for double the image count and buys a crash-vs-reject safety margin. C5 (max KV) is attractive on goal 1 but is the one most likely to fail goal 3. The data from the campaign decides.

## Open questions — feedback wanted

This is the part we're posting for outside eyes. Specifically we'd value a second opinion on:

1. **Is a crash-vs-reject distinction measurable and stable enough to drive a prod decision**, or is it too noisy at the boundary? (We plan to detect it via the health endpoint going dark + the systemd restart log, not from the request response alone.)
2. **Is `--mm-encoder-tp-mode data` worth the throwaway at TP2**, or is the "higher memory" penalty so certain at 2-way TP that we should skip C6?
3. **For goal 5 (concurrent text+image mix), what's the right minimal test** — we're leaning toward "N images in flight while M text requests decode," but we want to make sure we're measuring the actual Mamba-state contention and not just KV pressure.
4. **Are we missing a knob** — e.g., is there value in tuning `mm_processor_cache_gb` down (from 4 GiB) to claw back more KV, or does that just hurt the agent-loop cache-hit rate (goal 7)?
5. **Is 2 Mpx/image the right "real photo" upper bound** for the target use, or should the matrix include a 4 K (16 Mpx, the model's native max) cell to know the absolute ceiling even if we never serve it?

If you can push back on any of the framing above — especially #1 and #3 — that's the most useful thing you can do.
