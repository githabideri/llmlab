# LMCache 0.5.0 RAM tier on the dual-3090 vLLM 27B: hard negative at engine init (HMA)

**Date:** 2026-09-15 (campaign ran 2026-09-14 11:18 → 2026-09-15 00:27 UTC)
**Node:** the vLLM node (2× RTX 3090, TP2; the always-on 27B production service)
**Model:** Qwen3.8-27B, `W4A16-AutoRound`, vLLM **0.28.0** (TP2, MTP k=3, fp8 KV, 262,144 ctx, 8192 batched tokens)
**Candidate add-on:** LMCache **0.5.0** (`LMCacheConnectorV1`) with a persistent 32 GiB host-RAM pool sidecar
**Result:** **hard negative — the connector cannot boot this model at all.** The question "does the RAM tier buy TTFT / is it safe to adopt" is answered *no, for a hard compatibility reason*: under vLLM 0.28 a KV connector that does not declare HMA support disables the hybrid KV-cache manager, and this model is hybrid (mamba/SSM layers), so engine-core init dies at KV-spec unification — **before the connector ever touches the pool**.

## Goal

The [2026-08-30 offload report](2026-08-30-vllm-cpu-kv-offload-hybrid-mamba-fails.md) found vLLM's *native* CPU KV offload writing to host RAM but never restoring on this model (0% hit rate — a silent no-op), and closed with "do not retry it (including via lmcache) for this model." This campaign was that warned-against path, attempted properly: the LMCache 0.5.0 connector with a persistent 32 GiB host-RAM pool, instrumented to either **prove** the RAM tier end-to-end (retrieval-log evidence, a TTFT effect vs a no-connector control, zero answer-corruption on buried-key retrieval — the adoption gate) or **pin down exactly where and why** it fails.

## Setup

- **Isolated venv** (byte-copy of the production venv + `lmcache==0.5.0`); the production venv and unit were untouched except for the documented stop/restore around each window.
- **Persistent pool sidecar**: a window-level supervisor process holding the 32 GiB host-RAM pool (health + reset endpoints) that *outlives* vLLM relaunches — the design premise: a return request served by a **fresh** engine (empty GPU prefix cache) can only be served by the RAM tier, which is the actual isolation proof.
- **Campaign cells** (10 window cycles over the run): S0 probe (records the engine's reported block size / connector provenance), 40K- and 90K-token prefix store→return batteries with **no-connector controls** and LMCache variants, an LRU-eviction leg, corruption batteries (M-C/M-B/M-D with a zero-count pre-gate), an evolving-conversation cell, and 8K re-measurement. Each battery cell runs 3 segments (vLLM relaunched between segments); a watchdog with a hard deadline guards every window, and production was live-verified (health + one real completion) after each exit.
- The window also restarted the node's LXC to apply a 60 GiB memory bump (32 GiB pool + engine headroom).

## Observations

**1. The failure is at engine init, earlier and harder than the documented silent no-op.**

The failure chain, reconstructed from the boot logs and the vLLM 0.28 source:

1. `--kv-transfer-config` with a KV connector → vLLM 0.28 checks `supports_hma(connector)` (`vllm/distributed/kv_transfer/v1/base.py`).
2. `LMCacheConnectorV1` does **not** subclass `SupportsHMA` — there is zero mamba/HMA code in the lmcache 0.5.0 vLLM adapter.
3. vLLM therefore sets `disable_hybrid_kv_cache_manager=True`, emitting the exact warning its config anticipates (`vllm/config/vllm.py`): *"hybrid SSM models (e.g. Jamba, Bamba) require HMA and will fail at startup without it."*
4. `get_kv_cache_groups` then runs `unify_hybrid_kv_cache_specs` on this model, whose config declares a mamba SSM dtype (boot log: "Padding mamba page size by 0.48%"). Mamba layers cannot be promoted to a full-attention spec → `ValueError: Failed to promote local KV cache specs to one unified type` → EngineCore init fails.

The engine dies during memory determination — the LMCache engine/pool is never initialized, so there is no "connector loaded but idle" ambiguity: the failure is one step *earlier* than the 08-30 finding (which at least reached a running engine with a populated-but-never-read pool). The no-connector control lane boots every time (the same engine minus the connector).

**2. Along the way, the vLLM 0.28 connector contract surfaced (independent of the HMA block), for anyone wiring LMCache 0.5.0 against this vLLM line:**

- The config key is `kv_connector_extra_config` (not the older `kv_connector_config`), with `lmcache.`-prefixed keys mapping to `LMCacheEngineConfig` fields.
- A `kv_role` (`kv_producer` / `kv_consumer` / `kv_both`) is now **required** whenever a connector is set — missing it is a pydantic validation error.
- The remote pool is addressed by a `remote_url` of the form `lm://host:port` (the v1 socket-protocol server), not a bare host/port pair.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (which this production stack uses) conflicts with a KV connector in 0.28's validation; `--enable-cumem-allocator` is the accepted escape hatch.
- The C-extension (`lmcache.c_ops`) fails to import against this torch build (undefined symbol `materialize_cow_storage`); the package falls back to pure Python with a warning — a real but non-fatal ABI gap in 0.5.0.

**3. The control lanes measured real work.** With the connector out of the picture, the no-connector lanes executed the full prefix batteries: 40K-token stores at ~30–69 s wall, and — notably — **return legs at ~1.5 s** once the stock auto-prefix-caching had seen the prefix within the same process. That is the stock APC doing its job, and it is the baseline any RAM-tier benefit would have to beat. (The campaign's wall-time plausibility floor was sized for first-touch legs and misfired on these fast return legs — a methodology lesson recorded in [docs/benchmarks.md](../docs/benchmarks.md).)

## Conclusion

- **Hard compatibility negative (the exact combination): LMCache 0.5.0 (`LMCacheConnectorV1`) cannot initialize Qwen3.8-27B under the tested vLLM 0.28.0 hybrid-KV path.** The hybrid KV-cache manager is disabled for a connector that doesn't advertise HMA, and hybrid KV specs cannot be unified, so engine init fails before the LMCache pool is ever touched. No configuration, chunk size, or role reaches past that check; the only "fix" would be a venv-level `SupportsHMA` impersonation, which would paper over a genuine capability gap (the connector has no mamba KV handling to offer) and was deliberately not done.
- **So, precisely:** *adoption on this stack = NO* (the compatibility prerequisite failed). *TTFT benefit = not tested.* *Restore correctness = not tested.* Not merely negative — the scientific core is **untestable** on this pair, and the verdict (`review-required`, no adoption evidence, no corruption evidence) encodes exactly that: nothing was measured, and nothing is claimed.
- **Retest trigger (narrow).** This closes **LMCache 0.5.0 + `LMCacheConnectorV1` on this stack** — not LMCache in general. A candidate connector is retest-eligible when it (1) advertises HMA to the tested vLLM version, (2) supports this model's hybrid mamba/attention cache groups, and (3) initializes it without disabling the hybrid manager. That may be near: **LMCache 0.5.5 on PyPI ships `LMCacheMPConnector`, which subclasses `SupportsHMA`** (verified in the sdist, 2026-09-15). Advertise-only is necessary, not sufficient, for this topology — conditions (2)–(3) are the actual gate.
- **Production impact: zero.** The candidate ran entirely in the isolated venv/sidecar; every window exit was live-verified (health 200 + a real completion) before the next one opened.
- **The next test**, if an HMA-advertising connector passes the (2)–(3) gate: a sidecar-less `kv_transfer_config` A/B on the two 3090s, with store/return legs in different restart-isolated segments so the return can only be served by the RAM tier.

## Related

- [2026-08-30 — vLLM CPU KV offload fails on hybrid mamba](2026-08-30-vllm-cpu-kv-offload-hybrid-mamba-fails.md) — the antecedent: native offload wrote but never restored; "do not retry, including via lmcache"
- [docs/benchmarks.md](../docs/benchmarks.md) — workload integrity: evidence of work, not wall-time proxies
- [Qwen3.8-27B model card](../models/qwen3.8-27b-rtx3090.md)
