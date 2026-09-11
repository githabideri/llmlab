# Multi-Image Vision Ceiling — Qwen3.8-27B on 2× RTX 3090

**Status (v2, 2026-09-11):** research + **staged** campaign design, reworked after external review (see the v1→v2 change log at the end). The campaign is **not run yet** — the build is approved; execution is gated to a scheduled window with the controller on an independent endpoint.

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

## The memory knobs (v2 — profiling vs processor separated)

The single structural fix from review: vLLM 0.28 has **two distinct mechanisms** that v1 conflated under “size hints.”

| Knob | Role | Here |
|---|---|---|
| `--gpu-memory-utilization` | total headroom the engine may use → sets the KV pool | 0.93 |
| **Profiling hints** `--limit-mm-per-prompt {"image":{"count":N,"width":W,"height":H}}` | dummy input used **only** to estimate reserved activation memory (does not change inference); `count` *also* is the hard per-prompt image cap | count 16; w/h matched to each test image |
| **Actual processor limits** `--mm-processor-kwargs {"size":{"shortest_edge":S,"longest_edge":L}}` | the **real** resize/tokenization envelope given to `Qwen3VLProcessor` (sets the actual max image size + token count) | 65536 / 2,097,152 |
| `--mm-processor-cache-gb` | host-side processed-input cache, replicated across engine processes — **not** a chunk of GPU KV (corrects v1 open question #4) | 4 GiB |
| `--mm-processor-cache-type` | `lru` (default) vs `shm` (TP workers share processed tensors) | shm |
| `--mm-encoder-tp-mode` | `weights` (TP-shard ViT) vs `data` (replicate + batch-parallel) | weights |
| `--disable-chunked-mm-input` | don’t split one MM item across chunked-prefill chunks | off (chunked on) |
| `--skip-mm-profiling` | skip the MM activation reservation, shifting memory responsibility to the operator — **diagnostic calibration only, never a prod config** | off |

For ordinary ceiling cells we **deliberately synchronize** the profiling hints (w/h) to each test image’s size, so the profiler isn’t reserving against the 16 Mpx default.

## The campaign design (v2 — sparse, staged, smart-stop)

Reworked after external review. Not a 72-cell factorial, but **seven staged, ~15 cells**, each gating the next. If the window runs short the cut order is **MUST A·B·C·D·G → SHOULD F → COULD E** (encoder `weights|data` is the first to go, *before* the agent-loop cache test).

| Stage | Tests | Class | Primary readout | Smart-stop |
|---|---|---|---|---|
| **A — memory surface** | sparse count×pixel cells centered on **equal-total-pixel pairs**, across gpu-util {0.90, 0.93}; + one `--skip-mm-profiling` calibration cell | throwaway | min **sampled** free VRAM per phase vs (actual visual tokens, image count) | — |
| **B — safe candidate** | the cell with *real margin* (not the largest that survived) | candidate | N=3 cold: 0 deaths, 0 restarts, 0 fatal alloc, sanity pass | fail → drop an envelope |
| **C — concurrency** | two **simultaneous** legal requests (start-barrier released); incl. one **asymmetric** case (near-ceiling + one ordinary) | throwaway | engine death? recovery? **request-active vs MM-execution overlap** | **if two individually-legal requests kill the engine → reject candidate, drop an envelope, rerun B→C** |
| **D — chunked-MM A/B** | `--disable-chunked-mm-input` on/off at the candidate envelope (the #41485 deepstack+chunk+prefix bug) | throwaway | correctness, OOM, TTFT, peak VRAM, ITL on concurrent decode | correctness corruption / death = **categorical**, not averaged |
| **E — encoder TP** | one safe `weights` vs `data` cell (below the boundary) | throwaway | startup VRAM/KV, peak during encode, vision TTFT, PCIe/NCCL | — |
| **F — cache policy + multi-turn** | LRU vs SHM; a 4-turn agent loop reusing 16 images (one changed, prefix changed) | throwaway | `vllm:mm_cache_hits_total`, encoder-cache, prefix-cache, per-stage time via `vllm bench mm-processor` | — |
| **G — mixed QoS** | one steady text decode + one N-image request | candidate-only | phase-resolved ITL (before / during-encode / during-prefill / after) + recovery | — |

**The architectural discriminator (Stage A).** Run **equal-total-pixel pairs**: `32×2M` vs `64×1M` (64 Mpx) and `16×2M` vs `32×1M` (32 Mpx). Same aggregate pixels; if peak memory differs, **per-image overhead / batching geometry** matters beyond total tokens. Regression uses **post-processor** visual tokens (not nominal Mpx — the processor resizes around patch geometry):

```
peak sampled VRAM ≈ base + A · actual_visual_tokens + B · image_count + residual
```

**The profiling-calibration cell (Stage A, `diagnostic_only: true`).** Run the same envelope with and without `--skip-mm-profiling`. The delta in reported KV pool estimates how conservative the reservation is; the real peak tells us how much of the cliff is phantom. **It can never win production selection** — it deliberately removes the reservation’s protection, so a surviving request is not evidence of a viable profile.

**The decision gate (replaces the v1 7-goal rank).** A cell is a production candidate **iff** at its worst supported envelope — N=3 cold, **0 EngineCore deaths + 0 systemd restarts + 0 fatal CUDA alloc + all token/image sanity pass**, **and** two simultaneous legal requests survive, **and** any rejection leaves `/health` + a live text completion + a subsequent vision completion working immediately. Survivors are then ranked on: **max KV · reliable >16 images · TTFT@N · QoS cost (phase-resolved) · agent-loop cache reuse**. If Stage C shows two individually-legal requests can kill the engine, **per-request capping is insufficient** — the fix becomes a **global MM admission-control budget**, not another `--gpu-memory-utilization`.

**Two-layer measurement (both on a monotonic timebase).**
- **External:** NVML `memory.free` per physical GPU at ~100 Hz — reported as **minimum *sampled* free VRAM** (a sub-10 ms allocator spike can fall between polls; a fresh process’s CUDA context alone eats ~300 MiB, so this must be read on the *running engine*, not a synthetic process).
- **Internal:** vLLM’s own peak if exposed (0.28 has no `/stats`/`/server_info` peak endpoint, so the throwaway build can add a `torch.cuda.max_memory_allocated` hook); plus the `/metrics` counters `vllm:mm_cache_hits_total` and `vllm:prefix_cache_hits_total` for the cache tests.
- Phase markers (request start → processor → encoder → LLM prefill → first token → end) are aligned to the same monotonic clock; even imperfect markers beat inferring phases after the fact.

**Two-tier concurrency evidence (Stage C).** *"Two requests submitted together" ≠ "two vision activation lifetimes overlapped."* The driver records **`request_active_overlap_ms`** (intersection of `[request_start, first_token]` — proves the requests were *in flight* together) and, where the server exposes encoder-phase markers *and* both requests were **engine-active** (not merely queued in HTTP/scheduler), **`mm_execution_overlap_ms`** (intersection of the encoder/MM-prefill windows). The conclusion must state which it is: `CONCURRENT-SAFE` (MM transients demonstrably coexisted), `IN-FLIGHT (UNVERIFIED)` (overlapped but MM overlap unknown — partial), or `invalid-conc` (didn't materially overlap — discarded, not averaged). A hard start barrier releases both at once; overlap is proven from monotonic timestamps so HTTP/scheduler jitter can't fake a near-sequential test into looking concurrent.

### The final A→G cell matrix

*(px = per-image processor envelope; **T** = throwaway, **C** = candidate; KV pools at the count-16/2 M config: 0.90→677,931 · 0.93→710,402 · 0.95→747,625.)*

| # | Stage | gpu-util | envelope | flags | class |
|---|---|---|---|---|---|
| 1 | A | 0.93 | 16 × 2M | base (current prod) | C (baseline) |
| 2 | A | 0.90 | 32 × 2M (64 Mpx) | base | T |
| 3 | A | 0.90 | 64 × 1M (64 Mpx) | base | T ← pair with #2 |
| 4 | A | 0.90 | 16 × 2M (32 Mpx) | base | T |
| 5 | A | 0.90 | 32 × 1M (32 Mpx) | base | T ← pair with #4 |
| 6 | A | 0.93 | 16 × 2M | `--skip-mm-profiling` | T (`diagnostic_only`) |
| 7 | A | 0.90 | 32 × 2M | `--skip-mm-profiling` | T (`diagnostic_only`) |
| 8 | B | (from A) | best-margin cell | base | C ×3 cold |
| 9 | C | (from B) | 2 × 16 × 1M (concurrent) | base | T |
| 10 | C | (from B) | 2 × 32 × 1M (concurrent) | base | T (if inside margin) |
| 11 | C | (from B) | 32 × 2M **∥** 2 × 1M (asymmetric) | base | T |
| 12 | D | (from B) | 32 × 1M: chunked-on vs `--disable-chunked-mm-input` | A/B | T |
| 13 | E | 0.90 | 32 × 1M: `weights` vs `--mm-encoder-tp-mode data` | A/B | T |
| 14 | F | (from B) | LRU vs SHM + 4-turn 16-image reuse | A/B | T |
| 15 | G | (from B) | steady text decode **∥** 1 N-image request | — | C (survivor only) |

**Not run tonight** (per review): the **16.8 Mpx (≈4096×4096 — *not* “4K”)** single-image native max, deferred to a later single-image capability probe; the **GPU power sweep** (the box just went one clean night after two unexplained crashes — don’t confound memory attribution); **MTP-depth / CUDA-graph** re-tests (separate, already characterized).

**Replication of the discriminator cells.** Rows 2–5 (the equal-total-pixel pairs) are **replicated at N=2 (N=3 if the window allows)** before any regression fit. A one-shot `base + A·visual_tokens + B·image_count` fit could be reading allocator/fragmentation fluctuation (150–300 MiB) rather than a real per-image effect; the spread across reps is what tells them apart.

## v1 → v2 changes (external review)

- **Profiling hints separated from actual processor limits** (v1 conflated them).
- **`mm_processor_cache_gb` removed from the GPU-KV hypothesis** — it’s host-side processor cache, not KV (v1 open question #4 answered: no).
- **“3-axis / 72-cell sweep” → sparse staged design** (7 stages, ~15 cells); the old C7 4 Mpx cell was the out-of-envelope probe, now dropped.
- **“4K” → 16.8 Mpx ≈ 4096×4096**; 16 Mpx moved out of the prod matrix to a later probe.
- **Crash moved from goal #3 (a rank) to a hard gate** (N=3, 0 deaths, 0 restarts, sanity + two-simultaneous).
- **Stage C split into two overlap tiers** — `request_active_overlap_ms` (in-flight) vs `mm_execution_overlap_ms` (encoder/MM-prefill actually coexisted); the safety claim depends on the distinction.
- **Equal-pixel discriminator cells replicated (N=2/3)** so the `B·image_count` term isn't an artifact of one noisy read.
- **`boundary_distance` defined mechanically** (gap in actual visual tokens to the nearest tested *failing* envelope under the same config tuple; **censored/unknown, never ∞, when no failure was observed above**); recovery probes no longer put a vision request between healthy cold reps (cache contamination).
- **Added:** Stage C asymmetric legal-load case; `--skip-mm-profiling` diagnostic cell; Stage D chunked-MM A/B; Stage F cache-policy + multi-turn reuse; two-layer (NVML + internal) sampling on a monotonic timebase; equal-pixel regression on **actual** visual tokens.
- **`vllm bench mm-processor` is comparability-gated:** usable for stage decomposition, but a headline serving number only if its processor/resize/hash/cache/request contract exactly matches the serving run.

*The v1 open questions are now folded in: #1 (crash-vs-reject) → the hard gate + two-simultaneous test; #2 (encoder-tp `data`) → Stage E, first to cut; #3 (concurrent-mix shape) → Stage G’s one-steady-decode + one-image-request; #4 (mm cache GB) → not a KV lever; #5 (4K/16 Mpx) → deferred single-image probe.*
