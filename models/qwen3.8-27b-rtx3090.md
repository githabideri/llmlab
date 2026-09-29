# Qwen3.8-27B on the RTX 3090 pair

**Model:** Qwen3.8-27B (Dense, hybrid SSM + attention)  
**Tested Quantization:** W4A16-AutoRound (vLLM production), Q4_K_M (llama.cpp baseline/rollback), UD-Q4_K_XL (tested, heavier)  
**Hardware:** 2× RTX 3090 24 GB — vLLM tensor-parallel 2 since 2026-09-08 (single-3090 vLLM until then; single-3090 llama.cpp kept as dormant rollback)  
**Runtime:** vLLM 0.28.0 (production since 2026-09-08); 0.27.1 until then; llama.cpp 5f754ea retained as validated fallback  
**Status:** ✅ Production — vLLM TP2 (W4A16-AutoRound, MTP k=3, fp8 KV, 262,144 ctx, vision, **220 W/card** — interim since 2026-09-21, GPU 1 overheating, thermal service pending); the single-3090 llama.cpp Q4_K_M+MTP config below remains the documented rollback  
**Multimodal:** ✅ vLLM vision (weight-sharded vision tower, +0.86 GB across the pair) + llama.cpp fallback (mmproj BF16 on CPU via `--no-mmproj-offload`)  
**Supersedes:** Qwen3.6-27B BeeLlama deployment (see [`qwen3.6-27b-rtx3090.md`](qwen3.6-27b-rtx3090.md))

---

## Quick Facts

| Parameter | Value |
|-----------|-------|
| **Parameters** | ~27.3 billion |
| **Context Window** | 262,144 tokens (native) — deployed in full on the dual-3090 TP2 pool (710,402-token KV = 897 GPU blocks, block size 832 → ≈ 2.7× max concurrency); was 163,840 on the single 3090 |
| **Embedding Dimension** | 5120 |
| **Vocabulary Size** | 248,320 |
| **Quantization** | Q4_K_M (17.1 GB) |
| **Multimodal** | vLLM vision (weight-sharded across the pair) since 2026-09-08; llama.cpp fallback: mmproj BF16 ~0.9 GB CPU-resident |
| **Speculative Decoding** | MTP (draft-mtp, n-max 2, p-min 0.4) — no separate draft model |
| **Full-attention layers** | 17 of 66 (rest are SSM/hybrid) |

## Production performance (vLLM TP2 — measured)

**tgen (median of 3, cold unique-nonce prompts, [2026-09-10/11 campaigns](../reports/2026-09-11-dual3090-v2-operating-envelope.md)):**

| Workload | tgen |
|---|---|
| 2K → 1024 | **151.1 t/s** (IQR 8.6) |
| 16K → 1024, live endpoint | **147.8 t/s** (IQR 1.1) |
| 87K / 175K single-stream | 134 / 108 t/s |

**pp (prompt processing):** up to **~1,050 tok/s** cold on 16K prompts (live endpoint, measured 2026-09-21: 968.8 tok/s). Prefix-cache re-sends run far faster than cold (32K: 36.5 s cold → 1.22 s on a spaced re-send, ~30×) — those numbers are a cache-hit regime, not prefill speed.

**Live fleet (llm-hub, 7-day averages over active requests):** ~61 t/s tgen — real agent traffic with long contexts and thinking on; expectedly below the cold medians above. A single 3090 runs the same workload at 149.8 t/s — the second card is load-bearing for the KV pool/concurrency/vision envelope, not for text-decode speed (see the 09-11 report).

The llama.cpp baseline below is what the cutover replaced: the same model went from **~35 t/s to ~150 t/s** — the clearest single-model arc in this lab.

---

## llama.cpp Config (baseline / rollback)

**Runtime:** stock llama.cpp, commit `5f754ea` (unmodified upstream)  
**KV Cache:** q8_0 / q8_0  
**Batch:** `-b 512 -ub 64`, single slot

```bash
llama-server \
  --device CUDA0 \
  -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-BF16.gguf \
  --no-mmproj-offload \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.4 \
  -ngl all \
  -np 1 \
  --ctx-size 163840 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -b 512 -ub 64 \
  --flash-attn on \
  --fit off \
  --jinja \
  --reasoning on \
  --reasoning-preserve \
  --metrics \
  --mmap --mlock \
  --host 0.0.0.0 \
  --port 8080
```

### Why stock llama.cpp, not BeeLlama
Qwen3.8-27B ships with native MTP heads, so built-in `draft-mtp` speculative decoding replaces the BeeLlama DFlash draft model entirely (no separate ~1 GB drafter). At 160K context this leaves comfortable headroom (below), so there is no longer a memory reason to stay on the BeeLlama fork.

### Performance (measured 2026-08-15, Q4_K_M, 160K ctx)

| Metric | Value |
|--------|-------|
| Prefill (short prompt) | **~575 tok/s** |
| Prefill (131K prompt) | **~390 tok/s** |
| Generation (sustained) | **34.7–37 tok/s** |
| MTP draft acceptance | 66–94% (typ. ~75%) |
| Long-context recall | 3/3 needles at ~128K |
| Vision | Accurate (mmproj on CPU) |

### VRAM / headroom

| Quant | 160K headroom | Note |
|-------|--------------|------|
| UD-Q4_K_XL (17.9 GB) | **398 MiB** | Too tight for production |
| **Q4_K_M (17.1 GB) — deployed** | **1,138 MiB** | Chosen for production |

Under sustained load VRAM holds **23,235 MiB / 24 GB** and stays flat (verified through a real 131K-token workload).

### Why the feared quantized-KV FA pathology does not appear here
The upstream `f8f0a47a` "quantized-KV flash-attention scratch blowup" does **not** manifest on `5f754ea` for this model:
- Only 17 of 66 layers are full-attention (hybrid SSM arch), so KV is only ~64 KB/token at f16-equivalent accounting.
- At decode the **VEC** kernel is selected, which does not require an F16 K/V mirror for quantized KV.
- Verified **empirically**: flat VRAM through a genuine 131K-token workload (not assumed).

---

## Known Limits

- **CPU KV offload / RAM-tier extensions are not functional for this model**: on vLLM 0.27.1 the native offload stored to RAM but hit rate stayed 0% (no hybrid-aware offload planner, upstream #38230/#49537); on vLLM 0.28.0 **LMCache 0.5.0 fails even earlier** — its connector lacks `SupportsHMA`, vLLM disables the hybrid KV-cache manager, and this hybrid-SSM model dies at engine init (KV-spec unification) before the pool is touched. Do not retry *that combination* (0.5.0 `LMCacheConnectorV1`); the re-test trigger is a connector that (1) advertises HMA to the tested vLLM version, (2) supports this hybrid's mamba/attention groups, and (3) inits without disabling the hybrid manager — LMCache 0.5.5's `LMCacheMPConnector` subclasses `SupportsHMA` and is the first candidate to check. The native-offload path still waits on upstream #38230/#49537. See [2026-08-30-vllm-cpu-kv-offload-hybrid-mamba-fails.md](../reports/2026-08-30-vllm-cpu-kv-offload-hybrid-mamba-fails.md) and [2026-09-15-lmcache-ram-tier-vllm-dual3090.md](../reports/2026-09-15-lmcache-ram-tier-vllm-dual3090.md).
- **TP2 has no NVLink on this board** — collectives run over PCIe (NCCL; custom allreduce disabled), both cards CPU-direct Gen4 x8. Adequate for the current workload; a PCIe-bound TP2 is the thing a full campaign should characterise (pending).
- MTP requires the GGUF to include MTP heads (it does, natively for Qwen3.8).

---

## Changelog

### 2026-09-28/29: Re-paste done, dual engine restored, 250 W — and the "degraded card" was a hot slot
- **The 09-26 "electrically degraded" verdict was a misdiagnosis** (supersedes the entry below): the oscillating card (300–600 MHz at 79 °C under the 180 W cap) was simply sitting in the **hotter** of the board's two CPU slots — measured 10–20 °C hotter than the second slot *whichever card occupies it*, crossing the driver's 83 °C SW-thermal target while its twin in the cooler slot held ~1800 MHz at the same budget. A post-install card swap had put the original in that slot; no cap value could "fix" a position effect.
- The original card (**MSI VENTUS 3X 24G OC**) was pulled, re-pasted, and returned; the purchased card (**ASUS TUF-RTX3090-O24G**) stays in the box. The two were re-swapped in the window, so the freshly repasted card now sits in the **cooler** slot. The one genuine anomaly on the board: the VENTUS carries a **foreign GIGABYTE-family VBIOS** (`94.02.42.00.2F`, subsystem `1462:3881` — a previous owner's flash; GIGABYTE's own 24G 3090 series runs `94.02.42.40.xx`) — a curiosity, not a fault. The TUF's VBIOS (`94.02.42.00.B4`) is stock ASUS.
- **Dual TP2 is the steady state again**; the TP1 stopgap (`qwen3.8-27b`, 150k fp8, text-only) stays behind the mux but disabled. Power caps back to **250 W** on both cards (owner's call; the 180 W stopgap is retired). Post-window spot check: 74 °C @ 244 W (re-pasted card, cooler slot) vs 87 °C @ 209 W (TUF, hot slot) at light load.
- Clock note (closes a recurring question): the 2100 MHz (TUF) / 2130 MHz (VENTUS) `clocks.max.graphics` values are the **tops of the stock GA102 supported-clock P-state tables** — full 15 MHz-stepped tables down to the 210 MHz idle bin, stock 9751 MHz (19.5 Gbps) memory max — *not* modded-VBIOS clocks. Stock 3090s top out right there (community-measured: a STRIX OC 3090 "reaches 2100 but stabilises ~1980" at max power slider).

### 2026-09-26: One-card swappable serving — vllm-mux front, TP1 stopgap engine, 220 W restored
- The older 3090 (GPU 1) proved **electrically degraded**, not just thermally: under the 180 W stopgap cap it oscillated 300–600 MHz at 79 °C while its serviced twin held 1800 MHz at the same power budget — no cap value fixes that; the card goes for thermal/electrical service.
- The 180 W stopgap (same day, for the 83 °C SW-thermal derating the newer driver introduced) helped single-stream (ITL 57→41 ms) but made **multi-session** clearly worse (287 ms mean ITL / ~25 t/s at 3 reqs vs ~55–65 t/s at 220 W) — the owner restored **220 W** (the 250 W target stands after the card service).
- A **TP1 stopgap engine** now runs on the same node: the 0.28.0 head single-user profile on the serviced card (150k fp8 KV, MTP k=3, text-only, served name `qwen3.8-27b`). Mutually exclusive with TP2 — both engines share the serviced GPU, so at most one may boot at a time.
- Both engines now sit behind a **vllm-mux** on the public endpoint :8080 (the dual engine moved to :8083 behind it): **one hub card, two swappable model groups**, exclusive-GPU switching on demand (the first request for a non-resident engine evicts the other, ~2–4 min load) plus the hub's one-click load/unload (new hub kind `vllm-mux`; the card keeps the vLLM-native metrics of whichever engine is active). A both-services-enabled config crash-looped one LXC boot before the boot state was fixed to dual-only.
- Card-removal runbook: the LXC stays bootsafe with one GPU — every `/dev/nvidia*` bind entry is `optional`, the failure mode is a *non*-optional entry pointing at a missing device (the 2026-08-04 MIG-cap incident).

### 2026-09-21: Power limit 250 → 220 W (interim — GPU 1 running hot)
- GPU 1 (the original 3090) reads **81 °C at idle** (new card: 57 °C) at ~195 W drawn. `gpu-power-limits.service` reverted both cards to **220 W** until the card gets thermal service (paste + pads). 250 W (the 2026-09-08 setting, +6.2–6.5% prefill at equal stability) is the documented target again after the repaste. No performance claim at 220 W yet — the campaign numbers above were taken at 250 W; expect a few percent of prefill back after the repaste.

### 2026-09-16: Endpoint port 8082 → 8080, ctx-budget aliases added (undocumented drift made official)
- The serving endpoint moved 8082 → **8080** via an undocumented `Environment=PORT=8080` systemd drop-in (created 2026-09-15) — port 8082 was being taken by a host-local service. The move had no decision behind it and left every consumer that still said 8082 (the hub, the Prometheus scrape job, two agent clients) broken. On 2026-09-16 the port was made **officially 8080** and drift-proof: start-script default 8080, unit drop-in pins `PORT=8080`, unit description updated, all consumers re-pointed.
- The endpoint now advertises **7 names**: `qwen3.8-27b-dual` plus ctx-budget aliases `qwen3.8-27b-dual-{160,128,100,80,64,32}k` — the *same single endpoint* (one engine, one KV pool); the suffix is a client-side budget hint for consumers that want to state their context expectation. No config difference between names.
- KV pool note: the 0.28.0 engine resolves a block size of 832 for this hybrid model (not 1024), so the pool is 897 blocks = **710,402 tokens** (≈ 2.71× the 262,144 max), not the 776,928 logged at the 09-08 cutover. Same bytes, different block arithmetic — the "2×160K + 10×32K fits with headroom" conclusion is unchanged.

### 2026-09-11: Operating-envelope campaign — knee confirmed at k=3; TP1 text-decode parity (card freeability still open)
- Overnight v2 campaign ([report](../reports/2026-09-11-dual3090-v2-operating-envelope.md)): MTP envelope on the 8192 profile — k=0/2/3/4 = 67.1/113.4/151.1/167.6 t/s @16K; **knee at k=3** (k4 is the measured max, +11%, with a *narrower* IQR — so a knee, not a proven optimum; kept k3 as the conservative depth). Single-stream 16K on the live endpoint: 147.8 t/s (IQR 1.1). **Prefill interference measured: a 64K prefill in flight raises running-decode ITL 25.8 → 39.2 ms (+51.9%)** on the 8192 profile, with full recovery after — the measured **8192-side baseline** for the 8192-vs-2048 batch-budget A/B (the 2048 side is still to be measured; 48 h passive monitoring verdict due same day). **TP1 (single 3090) = 149.8 t/s ≈ TP2 147.8** at single-user 16K *text* decode → **no measurable text-decode benefit from the second card; whether it is operationally freeable depends on KV/concurrency/multimodal** (the 09-11 multi-image OOM is the counterexample — a rank can be irrelevant to text tgen yet load-bearing for the memory envelope). 32K prefix cache: 36.5 s cold → 1.22 s on a spaced re-send (~30×) while the near-immediate re-send missed — a real observation, mechanism not yet established (see the report's post-hoc note). Concurrency rows invalid by design (512-out vs 30 s overlap gate) — cells redesigned for the next window. First unattended campaign after the two 09-08/09-10 crashes: the box held, recovery stack unused.

### 2026-09-10: Profile attribution campaign — 8192 batched tokens promoted
- Overnight campaign (cloud-model-orchestrated; see [the report](../reports/2026-09-10-dual3090-overnight-campaign.md)): MTP k=3 ≈ **2.1×** decode vs none (151 vs 71.5 t/s @2K), `--enforce-eager` ≈ **3.4×** penalty, scheduler budget 2048→**8192** wins at 16K (150 vs 135–142). **Production unit switched to the 8192 profile** (script backed up; serving the fleet since). MTP acceptance 1.41–1.45 (stable across tested depths).
- 262K context re-confirmed safe (776K-token pool; 30-min 640K staged soak: 7 preemptions) but **TTFT-bound**: under the saturated 640K soak a 43K-token request observed 883–921 s *end-to-end* TTFT (≈47–49 tokens/s admission-to-first-token — queueing + preemption + prefill, not prefill throughput); ladder request-TTFT-implied 1.21K/0.86K tokens/s, decode 134/108 t/s at 87K/175K. The 256K ladder cell was a false-complete (exit-code-only acceptance) — no 256K data exists yet.
- The host hard-crashed once during the window (03:17 UTC, mid-restore; healthy 10 s heartbeat, no kernel/PCIe/IO/GPU trace, no power event on the rest of the network) — second unexplained reboot of this box in two days; see the hardware doc. The recovery stack (manifest on USB SSD + systemd auto-start + deadline watchdog) restored everything with zero manual touch.

### 2026-09-08: Dual-3090 TP2 goes production (3060 pair removed)
- The 3060 pair was removed from the box; a second RTX 3090 took the freed CPU x8 slot (the previous chipset-attached slot is now free for storage — de-congesting the chipset uplink that shared it with the model SSD).
- vLLM **0.28.0** (same `syv-ai` patch stack), **tensor-parallel 2**, `--max-model-len 262144`, fp8 KV, MTP k=3 (drafter capped at 163,840, same override as before), prefix caching, `qwen3` reasoning + `qwen3_xml` tool parsers, vision **enabled** (weight-sharded vision tower, +0.44 GiB per rank). Both cards capped at 250 W (220 W was the single-3090 setting; +6.2–6.5% prefill at equal stability). Memory per rank: 8.34 GiB consumed, 1.04 GiB peak activation, 0.59 GiB CUDA graphs (PIECEWISE, forced by spec-decode+FlashInfer as before) → **13.24 GiB KV per rank = 776,928-token pool**, 34.9 KiB/logical token — the 2×160K + 10×32K target (640K) fits with ~21% spare.
- Smoke suite all green: text, reasoning split, tool call, first-ever vision request on TP2, 16K prompt (~17K tok/s warm), 2/4 concurrency, 24.1% prefix-cache hit rate on identical prompts, MTP acceptance 73–87% (mean accepted length ~3.2–3.6). No NCCL/scheduler/preemption errors.
- The previous single-3090 vLLM (0.27.1, 163,840, text-only) and the llama.cpp 27B unit were disabled, not deleted — rollback path intact.
- **Campaign complete 2026-09-10** — profile attribution, production battery, and the 125B negative result are in [2026-09-10-dual3090-overnight-campaign](../reports/2026-09-10-dual3090-overnight-campaign.md); the pending campaign line below is closed.

### 2026-08-30: Dual-3060 node evaluated as a candidate second inference unit
- The two 3060s (normally serving 35B) ran this model under vLLM **TP2 + MTP k=4, fp8 KV, FlashInfer, 64K max len**: decode 87 / 77 / 67 / 57 / 57 t/s at 2K / 16K / 24K / 48K / 64K (3090 does 105 / 93 / — / — / 73 in the same cells), prefill 560–700 tok/s vs the 3090's ~1,000. Wall power ~340–370 W (the pair at ~220 W) — i.e. ~80% of 3090 speed at ~the same wall draw; the box idles at ~93 W.
- It is a **single-user machine**: KV pool is 126,390 tokens (64K = 1.93× max concurrency); 2×2K batches to ~97 aggregate t/s, 8×2K to ~191, but 4×16K collapses per-request decode to 3–46 t/s (mean 16) with MTP acceptance dropping to 17–39%.
- The experimental vLLM build (locally patched 0.27.1, same patch family as the 3090 production) was flaky on the pair: one FlashInfer drafter crash on the first long request, marginal OOM at `gpu-mem-util 0.93`; ran clean at 0.90 with a warmup request. Not deployed — see [the full report](../reports/2026-08-30-dual-3060-35b-squeeze-27b-node.md).

### 2026-08-30: CPU KV offload experiment — reverted
- Tested vLLM native CPU KV offload (12 GiB LXC store, LXC RAM 24 → 40 GiB) to extend prefix caching beyond the 208,664-token GPU pool. Write path healthy, **read-hit rate 0.0%** (hybrid SSM/GDN model — scheduler-side blocker, upstream #38230/#49537); GPU pool shrank 208,664 → 180,842 under the connector. Reverted to baseline (kept the 40 GiB LXC for a future re-test). `MAX_NUM_SEQS` 8 → 4 had zero pool effect — current config is pool-optimal. Also confirmed live: tool calling via `--enable-auto-tool-choice --tool-call-parser qwen3_coder`. See [the report](../reports/2026-08-30-vllm-cpu-kv-offload-hybrid-mamba-fails.md).

### 2026-08-21/22: Production runtime moved to vLLM
- vLLM 0.27.1 (PyTorch 2.13.0+cu130) now serves `Qwen3.8-27B-W4A16-AutoRound` (19.5 GB; int8 embed + MTP int4 "fast" prep), following the public recipe `syv-ai/qwen38-27b-rtx3090` @ `999e264b` (13 patches, all applied).
- MTP k=3 speculative decoding, fp8 KV (FlashInfer), 163,840 max len, prefix caching, reasoning parser (`qwen3`), tool calling (`qwen3_coder`), keyless, text-only (`--language-model-only`).
- Validated on the same RTX 3090: decode 96–118 tok/s (vs ~35 for llama.cpp Q4_K_M), prefill up to ~1,050 tok/s, 3/3 needles at ~155K, prefix-cache second turn 281 s → 2.8 s, PPL 8.095 (matches recipe reference), GSM8K 96.0% (n=200).
- The llama.cpp Q4_K_M + MTP config above is the validated fallback/rollback.

### 2026-08-15: Production cutover to Qwen3.8-27B (stock llama.cpp + MTP)
- Replaced Qwen3.6-27B BeeLlama (DFlash) on port 8080.
- Stock llama.cpp 5f754ea, Q4_K_M, 160K ctx, q8_0/q8_0 KV, MTP (n-max 2, p-min 0.4).
- Chose Q4_K_M over UD-Q4_K_XL for 1,138 MiB headroom (vs 398 MiB).
- Long-context 3/3 needles at ~128K; vision verified; 5/5 stability runs clean.
- BeeLlama Qwen3.6-27B service disabled (kept for rollback). See [`qwen3.6-27b-rtx3090.md`](qwen3.6-27b-rtx3090.md) for its history.

---

## Related Documentation

- **Systemd / server config:** [`docs/systemd.md`](../docs/systemd.md)
- **Predecessor (Qwen3.6-27B BeeLlama):** [`qwen3.6-27b-rtx3090.md`](qwen3.6-27b-rtx3090.md)
- **BeeLlama backend (historic):** [`docs/backend-beellama.md`](../docs/legacy/backend-beellama.md)
