# Strata on Pascal (sm_61): the non-tensor-core prefill port, and a 96K q4_0 KV profile for the 6 GB laptop card

**2026-10-10/11** · experiment + deployment · the fleet's Pascal node (2017 HP OMEN 15, GTX 1060 6 GB Max-Q, i5-7300HQ)

## TL;DR

Strata's batched prefill path has always been gated to tensor-core GPUs (sm_75+) — on the Pascal laptop the engine's prompt handling ran through the decode window (~45 t/s, the engine's one-token-at-a-time path). This window (a) ported the non-tensor-core, batched prompt-attention kernel to sm_61, fixing a half-dimension bug in the upstream proof-of-concept along the way, and (b) used the resulting VRAM headroom to bring the laptop's 35B endpoint back to its full **96K context** — with q4_0 KV kept entirely in VRAM by shrinking the expert cache — after the first cutover's 8K auto-fit was rejected. The laptop's production engine is now Strata: **~24 t/s decode at 96K (llama.cpp's 2-bit baseline on the same card: ~15.3–19.7), prefill 97.6 t/s at 12.4K** (degrading with prompt width — the honest trade, measured below); the llama.cpp 96K config stays one rebuild away as rollback. A 12.4K needle test passes through the q4_0 KV.

## Background: the two gaps

1. **Prefill.** Strata (PR #1402, v0.1.41-era base) runs prompts through a fused batched path (`qsa_prompt_attn_batch`): one attention pass over the whole prompt per layer, K/V dequantized in-kernel. On sm_75+ it uses a tensor-core kernel; below that, `Prefill::supports()` returned false and the engine fell back to processing the prompt in the decode window — one token per pass. On the Pascal laptop that measured ~45 t/s prefill vs the ~200–290 t/s llama.cpp's optimized attention manages on the same box (the 2026-09-28 Phase-4 numbers). The upstream repo carried a proof-of-concept for the non-tensor-core variant on a `poc` branch; we ported it to the pinned production base instead.
2. **Context.** The laptop's 6 GB card had carried the 35B at 96K under llama.cpp (2-bit GGUF, 20 of 40 expert layers on GPU, q8 KV) since the 2026-09-28 Phase-4 cutover. Strata's auto-fit on a small card reserves a 500 MiB dense-attention/runtime floor and sizes the KV pool to whatever is left — first boot on the card landed at **8K context**. That was not a viable production endpoint (the box's primary consumer is a web UI that needs a large window), so the cutover had to solve the 96K fit before it was acceptable.

## Prerequisite: the RAM question, closed

Before building, the laptop's RAM was upgraded to a matched 2×8 GB DDR4-3200 pair (Samsung M471A1G44AB0), the last open experiment from the hardware profile (does the 7th-gen board interleave a matched pair?). It does not help: STREAM/triad stays at ~18–20 GB/s flat with no thread scaling — one channel of DDR4-2400 is 19.2 GB/s, so the box sits at its per-channel theoretical regardless of module matching. **The DRAM wall on this machine is CPU-side (memory-controller limit), not DRAM-side.** A pre-Strata re-baseline of the running llama.cpp 2-bit endpoint confirmed it: 16K-filled decode **15.29 t/s** (vs 15.26 on the mixed pair), short-context **19.72** (vs 20.08) — flat within run noise. RAM is no longer a lever on this box; the hardware profile is updated accordingly.

## Build provenance

Built on the laptop (NixOS 26.11, nvidia driver 580 LTSB, CUDA 12.9 — the last toolkit family that targets sm_61):

| Artifact | Contents | Binary SHA-256 (first 16 hex) |
|---|---|---|
| A — base | pinned upstream SHA `48534f3` (v0.1.41-era) | `3468b2913dde84ce` |
| B — +GEMM | A + v0.1.42's GEMM fix cherry-pick `cbbf5574` | `978bd8d149b6a088` |
| C — +kernel | B + the sm_61 batched-prefill kernel and dispatch (this report's port) | `25793e205243314f` |

Three A/B/C arms keep the effects separable: the GEMM fix moves decode slightly (24.6–24.9 → 26.0–26.4 t/s), the kernel moves prefill 4.7× and does not touch decode (25.5–25.9 t/s).

## The port

**What was gated.** `Prefill::supports()` rejected cc < 75 for the MoE hybrid family, and the dispatch in `qsa_prompt_attn_batch` had no non-tensor-core arm. Two edits: the sm_75 floor comes off for this model family, and a dense FP32 arm is dispatched when the card is < sm_75 and the layer is a full-attention (dense) one — the 30 linear-attention (DeltaNet) layers keep their fixed-state path and never enter the prompt-attention kernel.

**The kernel.** The upstream PoC processed only half of the 256-wide head dimension per thread group — a soft-max over a half dot-product, wrong whenever the omitted half carries signal (it always does in practice). The ported kernel covers the full head dimension: 128 threads split the 256 dims two-ways, FP32 online softmax (the running max/rescale form, so no second pass), a warp-shuffle reduction inside the 32-thread groups and a shared-memory exchange between the 8 groups per query, K/V dequantized in-kernel from all three pool formats the engine uses (fp16, int8 + scale, q4_0 block), with a host-streaming fallback when the pools are not in the expected layout. The attention scale is 1/√256, matching the decode kernel exactly. ~205 lines of CUDA; it compiles first-try on 12.9/sm_61 and the regression below shows no perf cost to the decode path.

**The parity gate, redefined.** Before the port we established the engine is **not bit-deterministic run-to-run** on this card: two identical A-base runs on the same prompt diverge at character 1043 (66% of the output matches). GPU GEMM rounding noise, not a bug. So strict greedy parity against the base build is not a usable correctness gate for *any* change on this engine — including the GEMM fix. The adapted gate: a change passes if (a) its output profile (where and how often it diverges from the base) sits inside the engine's own A-vs-A noise band, and (b) at least one same-length cross-pair is byte-identical. Result: C-vs-A divergence is statistically indistinguishable from A-vs-A; the B→C pair is byte-identical for 1,662 characters; all outputs coherent. No evidence of kernel bias. (The tensor-core branch on sm_75+ is untouched by this work.)

## Qualification window (C vs A vs B)

| | prefill | decode (warm) | VRAM | RAM | swap |
|---|---|---|---|---|---|
| A — base | 45.4 t/s (window path) | 24.6–24.9 t/s | 5411/6063 MiB | 8.8 GiB | 0 |
| B — +GEMM | 45 t/s (unchanged) | 26.0–26.4 t/s | 5411 | 8.8 | 0 |
| C — +kernel | **215.6 t/s @2.6K · 173.2 @5.3K** (4.7× / 3.8×) | 25.5–25.9 t/s (unchanged) | 5621 | 9.2 | 0 |

Cold first request: 20.1 t/s end-to-end with the kernel (prefill ~4 s) vs 13.0 (prefill ~17 s) without. 8K+ prompts on the 8K-fit config return HTTP 400 by design — which is exactly the cliff that sent us after the 96K salvage.

## The context cliff, and why streaming was the wrong direction

The 8K auto-fit came from the 500 MiB dense floor plus the KV pool sizing on a 6 GB card (8K × int8 KV for the 10 full-attention layers ≈ 82 MiB; the rest of the budget goes to the expert arena and the dense floor). The obvious salvage — *stream* most of the KV from RAM (`--kv-resident`) and keep only a resident window in VRAM — is the right tool for **sparse** attention models, and it is the wrong one here: Qwen3.6's 10 full-attention layers attend densely over the entire history, so a 96K prompt with a 32K resident window puts 64K × (576 B q4_0 / 1056 B int8) of KV per token on the laptop's measured ~6.3 GB/s PCIe link — a 360–660 MiB/token floor of ~10–17 t/s *before* expert traffic. The asset that moves densely at 96K is the KV itself; the experts are the sparse, routed side. So the split moves the other way: **shrink the VRAM expert cache and keep the entire 96K KV in VRAM.**

The numbers that make it close: 96K of KV costs ~990 MiB in int8 but **~540 MiB in q4_0** (576 B/cell/layer — the engine's #1188 work confirmed q4_0 + streaming is the correctness-clean KV-quant path; the k8/v4 mixed one was the broken one). The engine's expert arena on the card is 10,240 slots (~8.44 GiB: 6978 pinned-host + 3262 in-VRAM at the default budget); dropping the VRAM share from 3262 to **2600 slots** frees ~559 MiB of VRAM — enough for the 540 MiB of 96K q4_0 KV plus headroom.

## The 96K profile (what ships)

`--max-context 98304 --kv q4_0 --expert-cache 2600 --spec 4 --prefill auto --resident-budget-gib 10`

| | measured |
|---|---|
| Boot | 5,549/6,144 MiB VRAM (595 MiB headroom); 9.8/15 GiB RAM (6.30 GiB page-locked expert arena); no swap |
| Decode (839-prompt / 512-completion, warm ×3) | **23.8–25.5 t/s** — vs the llama.cpp 2-bit baseline on the same card: 19.72 short-ctx / 15.29 @16K-filled |
| Cache search | 2800 VRAM slots boots (515 MiB headroom) but measures 24.1/25.4 t/s — no gain over 2600; 2600 ships (more headroom, same speed) |
| Prefill | **164 t/s @0.8K · 97.6 @12.4K · 92.6 @13.2K · 66.4 @19.8K** — degrading with width (the naive kernel is O(width) per query; see below) |
| Quality | 12.4K needle-in-haystack (fact planted mid-text, asked at the end): **hit, exact date and name reproduced**; thinking output coherent throughout |

The 2800 result is worth stating plainly: the expert cache is not the decode bottleneck at this size — hit rate already covers the working set, and the extra slots sit idle. The int8 variant (96K int8 KV needs the cache down to ~2000 slots) was left unmeasured — q4_0 passes the needle and the perplexity cost is the documented one (upstream: +8%/+12% vs fp16 at 1K/8K); int8 stays available as a config flip for anyone who wants the quality tier.

**The prefill trade, honestly.** The kernel's per-query work grows with the KV width (online softmax over all preceding cells, 128 threads, no tiling), so prefill throughput degrades as the prompt grows: ~170–215 t/s at ≤5K, ~93–98 at 12–13K, ~66 at 20K, and a fresh 96K prefill lands in the multi-minute range — the same order as llama.cpp's 96K ceiling on this box (measured in the Phase-4 campaign), not a regression against it. The llama.cpp kernel has a much better constant (its 202–291 t/s at ≤16K) thanks to tiling/vectorization that a 1,280-core Pascal card can't use the same way; ours has a 4.7× lead over the engine's own window path and a 2–3× lead over llama.cpp at ≤2.6K prompts — which is the dominant case for the box's agent workload. The proper next step for long prompts is a tiled, shared-memory FlashAttention-style kernel for the non-tensor-core path (upstream work, not this report's scope); the reused-prefix path the server uses for incremental context is a separate, slower read path (~33 t/s) that an engine-side fix is more likely to land than a kernel one.

## Serving mechanics (the NixOS gotchas, for the record)

- **The NixOS `/lib64/ld-linux-x86-64.so.2` on this box is a stub (musl-flavoured loader shim), not a real glibc dynamic loader** — any patchelf-relocated binary that names it as interpreter will not run. Both the engine and the venv's python were patchelf'd to the *store* glibc's real loader (`…glibc-2.42-84/lib/ld-linux-x86-64.so.2`) with RPATH into a private `deps/` dir (CUDA 12.9 runtime/cuBLAS/cuBLASLt, libstdc++/libgcc_s). The venv's python is a **real file** (copy + patchelf), because a symlinked store python drags the stub loader back in. GC roots live in `/root/.nix/gcroots/` — `/nix/store` is read-only on this NixOS.
- **The 26.11 nixpkgs python3 packaging API churn** (the old `withs` is gone, `withPackages` returns an environment directory, not a binary, `override { extensions }` is broken for this use) is what pushed the runtime to the out-of-store venv pattern rather than a store derivation — the same pattern the box already used for its other two long-lived services.
- **`serve.server` binds loopback by default** — the `--host 0.0.0.0` is load-bearing for a tailnet-facing endpoint (the first "why can't the hub see this" was a 127.0.0.1 bind).
- **The hub's `snapshot.json` persists the model roster across restarts** — a config change (new ctx, new desc) is invisible until the snapshot is archived and the hub service restarted. One-click load/unload of this card is VRAM-exclusive with the box's second card (one 6 GB GPU); the switch script now stops the Strata service instead of unloading a llama.cpp model.

## Final state and rollback

The laptop's `:8080` is **Strata** (model id `qwen36-35b-strata`, 96K q4_0, the C binary above), the llama.cpp 2-bit 96K unit is **defined but not wanted** — rollback is one rebuild (restore its `wantedBy`, disable Strata) plus the one-click switch. The two engines are mutually exclusive on the card by construction. Open items: the tiled prefill kernel (upstream), the int8 96K variant (unmeasured), the MTP 3-arm A/B (deferred — MTP is not in this GGUF; a MTP-capable quant is a separate experiment), and filing the sm_61 prefill port upstream.

## Related

**Update (2026-10-11, same day — wording, not results):** two claims in this report need the sharper wording. (1) The decode figures (TL;DR's "~24 t/s decode at 96K" and the profile table's 23.8–25.5 t/s) are **short-context (839-prompt) measurements on the 96K-capable profile, not filled-96K decode** — filled-context decode is part of the pending INT8 A/B. (2) The 12.4K needle "passing through the q4_0 KV" establishes **basic retrieval correctness, not long-session quality**: a model can retrieve a planted fact while its distribution over ordinary reasoning tokens degrades. Strata's own KV-quality study (teacher-forced vs FP16) shows q4_0 with a real, context-growing precision cost (8K documents: top-1 88.0% vs INT8's 92.3%; KL 0.297 vs 0.114; ppl 8.65 vs 7.63; ~+12% perplexity), so **the INT8 96K A/B on this model is the production-quality gate** — q4_0 is provisional until it runs. (Those upstream numbers are not Qwen3.6 measurements: only 10 of 40 layers carry conventional KV here, which cuts both ways — less quantized mass, but the 10 dense layers are the model's only direct access to individual historical tokens.)

- [Model card](../models/qwen3.6-35b-a3b.md) — the Pascal laptop variant, its changelog, and the retained llama.cpp config
- [Phase 4 (2026-09-28)](2026-09-28-ista-2bit-100k-performance-ceiling-pascal-1060-phase4.md) — the llama.cpp 2-bit 96K ceiling this cutover measured against
- [Strata on the secondary 3060 (2026-10-09)](2026-10-09-strata-qwen36-35b-a3b-single-3060.md) — the same engine's A/B on the fleet's other 35B home (the sm_75+ reference numbers)
- [Pascal laptop hardware profile](../docs/hardware/pascal-laptop.md) — the DRAM wall (now closed: CPU-side) and the PCIe bandwidth the expert arena rides on

## Update (2026-10-12, same window — the reused-prefix path, diagnosed; not an intrinsic cost)

The INT8-vs-q4 long-session battery that followed this report (one deterministic fact-document grown 4K→92K, six machine-scored question categories per checkpoint, plus a 70K entity-binding set, run under both KV profiles) hit the worst case of the "separate, slower read path" mentioned above: at the 92K step the engine's reuse anchor sat at the last **periodic** checkpoint (55,296 tokens) and *every* question re-read the entire 13,636-token tail — the per-request log shows `55296 reused + 13636 read in 1075 s (12.7 t/s)`, i.e. ~18 minutes per question, both KV profiles.

The diagnosis, corrected after external review: this is **not** a hard checkpoint cap and not a broken read kernel. The engine has `--prompt-cache N` (default 6, root pinned, LRU above) and `--prompt-cache-every N` (default 16,384); the observed anchors (18,432 / 34,816 / 55,296) sit exactly on the 16,384 grid. The 92K document's end (68,932) was never a checkpoint because the battery's prompts are single user turns — there is no turn token at the document's end, so no turn-boundary checkpoint exists there. The read path's own efficiency is not the dominant term: the fresh batched kernel and the deep read run at comparable per-query-key-pair throughput (≈605K vs ≈788K pairs/ms), so the cost is the legitimate O(new × history) dense-attention work of the 10 full-attention layers, **multiplied by the re-read distance** — and that distance is a configuration artifact, not a physical one.

Upstream documents the identical failure mode and its fixes (Strata `docs/RESEARCH_RUNS.md`, a 262K document on a P100): through the server chat API with no turn boundary at the document's end, 55.5 s per follow-up question; with the boundary, 0.53 s median. The mechanisms — turn-boundary checkpoints (`--turn-token`), the server's `strata_prefix` pinned-prefix field (the document-end checkpoint is pinned, never LRU-evicted), and `--prompt-cache-tail` (one extra checkpoint at the last prefill chunk boundary) — **are all present in this build (48534f3)**; our serve configuration set none of them. Conclusions: (1) the multi-minute follow-up TTFTs in the battery are **not** to be read as the intrinsic cost of a normal continued chat on this card — the anchored re-run (pinned prefix, both KV profiles) is pending and will measure the real number, expected to fall from minutes to seconds; (2) the fresh-prefill width degradation of the ported kernel (164→97.6→66.4 t/s) remains the honest limitation and the right target for a tiled, shared-memory kernel — but no kernel can make a single-token append at 92K cheaper than O(history); only the anchor shrinks that work; (3) the per-request log (`"4 checkpoints"`) is not debuggable — a diagnostics request (anchor position, source, checkpoint set, read span, reason) goes with the sm_61 port to upstream.
