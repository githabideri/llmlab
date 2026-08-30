# vLLM Native CPU KV Offload — Silent No-Op on Hybrid SSM (GDN) Models

**Date:** 2026-08-30 · **Stack:** vLLM 0.27.1 · Qwen3.8-27B W4A16 (hybrid SSM + attention) · 1× RTX 3090 · fp8 KV · 163,840 ctx
**Verdict:** REVERTED — offload wrote to CPU RAM but never restored. Not a storage problem; it's scheduler-side. Don't retry it (including via lmcache) for this model.

## Goal

Qwen3.8-27B production serves at 163,840 ctx with prefix caching; the GPU KV pool holds **208,664 tokens** (fp8, ~1.27× one full session). Real workloads (90k–147k-token documents, multiple sessions per day) evict older sessions' prefixes, and each return costs a full recompute (~130 s at ~700 tok/s prefill). The question: can vLLM's **native CPU KV offloading** extend prefix-cache reach beyond the GPU pool — store evicted KV blocks in host RAM, restore on hit, skip recompute?

## Setup

Baseline production unit (dedicated LXC, RTX 3090, vLLM 0.27.1 from the public `syv-ai/qwen38-27b-rtx3090` recipe @ `999e264b`):

```
Environment=PORT=8080 CTX=long SPEC=mtp MAX_LEN=163840 PREFIX_CACHE=1
Environment="EXTRA_ARGS=--enable-auto-tool-choice --tool-call-parser qwen3_coder"
```

Offload test deltas:

```
EXTRA_ARGS+=--kv-offloading-size 12 --kv-offloading-backend native   # 12 GiB CPU store
LXC memory: 24 GiB → 40 GiB
CUDA_VISIBLE_DEVICES env + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# (required to clear vLLM's offload-connector vs. expandable-segments allocator conflict)
```

A/B/C test: three independent session documents, **90k / 80k / 70k tokens** (240k combined — deliberately larger than the 208,664-token GPU pool). Each session re-issued *after* the later sessions had evicted it, so a working offload tier must restore from CPU RAM instead of recomputing. Correctness probed with in-document markers at known positions.

## Observations

- **Write path worked.** CPU store filled healthily (836 MB stored in ~0.03 s for the first evicted chunk); logs showed the native connector active, no errors.
- **Read path never fired.** Offload hit rate: **0.0%** for the entire A/B/C run. Every session return triggered a full recompute (~130 s each) — identical to baseline, no faster, no restore logs.
- **Correctness was perfect.** All markers answered correctly — which is consistent with *nothing being restored*: the model recomputed from scratch every time. No corruption, no stale-KV artifacts. (A half-working tier would show exactly those — a good canary.)
- **Side cost:** the offload connector *shrank* the GPU pool: 208,664 → **180,842** tokens. So the configuration bought a dead CPU tier *and* lost ~13% of the GPU pool. Net negative.
- **Upstream corroboration.** vLLM [#38230](https://github.com/vllm-project/vllm/issues/38230) (HybridOffloadPlanner for mamba+attention) and [#49537](https://github.com/vllm-project/vllm/issues/49537) (SupportsHMA) — hybrid external KV cache is explicitly open work. The offload planner doesn't know how to schedule restore for layers with recurrent/SSM state, so it never emits a hit.

## Second experiment: MAX_NUM_SEQS 8 → 4

Hypothesis: fewer concurrent spec-decode recurrent-state slots would free pool capacity. **Zero effect** — pool identical at 208,664 tokens. Current config is already pool-optimal; pool size is a function of max-model-len and KV dtype, not seq slots.

## Conclusion

- **Do not use `--kv-offloading-backend native` (or lmcache) for hybrid SSM/GDN models on vLLM 0.27.1.** The blocker is scheduler-side (no hybrid-aware offload planner), not storage-side — so changing storage backends is a dead end; only an upstream planner fix helps.
- Effective cache hierarchy for this model stays **GPU-resident prefix cache only**: hits ≈ 2.9 s TTFT on a 90k-token document; misses recompute at ~700 tok/s.
- Reverted to baseline (config backup kept as `.bak-kv-offload` in the LXC). The LXC kept its 40 GiB RAM — it's harmless and prepositions the box for when hybrid-aware offload lands.
- **Follow-up:** when a hybrid-aware offload planner lands upstream (watch #38230 / #49537), re-run the A/B/C test (three documents > pool, marker probes, compare TTFT and hit-rate logs) — the 40 GiB LXC is already in place.

## Notes

- The first long prompt after any vLLM restart pays a one-time Triton JIT warmup storm (~60–120 s). Unrelated to this experiment, but it inflates "cold" timings if you don't account for it; a small `ExecStartPost` warmup request would smooth it if that ever matters.
- Deferred by design (separate questions, different blockers): DFlash2 drafter (downloaded, unused), INT8 activations (batch mode), 262k KVarN, SSD KV tiering, multimodal gateway.
