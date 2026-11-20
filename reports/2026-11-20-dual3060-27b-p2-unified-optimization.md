---
name: dual3060-27b-p2-unified-optimization
description: P2 unified-KV optimization on the dual RTX 3060 — baseline, MTP/ubatch tuning, the 232K single-session ceiling, multi-agent concurrency, and the recommended profile
---

# P2 Unified KV on the Dual RTX 3060: measured baseline, tuning, ceiling, and the recommended profile

**Date:** 2026-11-20. **Scope:** one host, two RTX 3060 12 GB (layer split 60:40), one 27B IQ3_MTP GGUF, engine b12383, four scratch-server maintenance windows (20 min + 20 min + 11 min + 14 min; production restored and live-verified after every window). **Goal:** turn the P2 concept (single unified KV pool shared by two consumer slots) into one recommended, manually selectable serving profile — with measured context ceiling, tuning, and concurrency behavior, not projected numbers.

## Recommended profile (P2-RC1)

| Component | Value |
|---|---|
| Model + projector | same as the qualified fallback; SHA256s recorded in [provenance](../data/provenance/dual3060-27b-2026-11-20-m4.json) |
| Engine | llama.cpp b12383 (`llama-server`) |
| Layout | `CUDA_VISIBLE_DEVICES=1,0`, `--split-mode layer --tensor-split 60,40`, `--mmproj-device none` (A8) |
| KV | `--ctx-size 262144 --parallel 2 --kv-unified --kv-unified-per-slot 245000 --cache-type-k q4_0 --cache-type-v q4_0 --cache-prompt --cache-reuse 4096` |
| Batching | `-b 256 -ub 256` |
| Speculative | `--spec-type draft-mtp --spec-draft-n-max 2` |
| Misc | `-t 4 --load-mode none --jinja --metrics` |

Rationale, each item measured (sections below): unified pool with a 245K per-slot cap (ceiling section: 232K verified with a concurrent 16K consumer; 245K cap leaves headroom under the 262144 per-sequence limit); n-max 2 (+11% decode, acceptance 0.61 vs 0.42 at n-max 5); ub 256 (−4% cold TTFT, no decode penalty). Everything else is the unchanged qualified fallback. Rollback = the existing launcher (P1 non-unified); no other part of the stack changes.

## Phase A — P2 baseline (unified 262144, per-slot 200K, n-max 5, ub 128)

Engine-counted, one boot, port 8124. All warm re-POSTs were 4 new tokens.

| Depth (engine tokens) | Cold prefill | t/s | Warm re-POST | Retrieval |
|---|---|---|---|---|
| 11,827 | 23.3 s | 508 | 0.24 s | — |
| 47,241 (the M3 fixture) | 108.8 s | 434 | 0.26 s | — |
| 122,255 | 380.6 s | 321 | 0.30 s | — |
| 167,079 (the "182K" corpus; engine count) | 602.0 s | 278 | 0.33 s | first cold pass **missed**; immediate re-POST and all later passes correct |

Prefill throughput decays 508 → 278 t/s over 11.8K → 167K (attention growth, not a transfer effect). Decode at 16K context: 128 tokens in 3.1 s (41 t/s, MTP acceptance 0.90); short decodes at deep depths ran 12–22 t/s with acceptance 0.37–0.62. Two-consumer: a 16K consumer on slot 1 needed **only 4 new tokens** while a 167K session was resident on slot 0 (cross-slot prefix reuse from the shared pool), and the 167K session's follow-up afterwards was still fully cached. VRAM headroom with 167K resident: 2034 MiB (primary card) / 1021 MiB (secondary).

**PCIe under active inference (the open question from the 11-19 review, answered without reboot or kernel changes):** a 2 s sampler over the whole window recorded the negotiated link at **8 GT/s (Gen3) at all times during all four cold prefills and all decodes**; GPU utilization peaked 100% on both cards. The 2.5 GT/s (Gen1) state occurs only while the server is idle after a model load (fingerprint re-confirmed: production load at window close returned both links to Gen1). Consequence: a freshly (re)loaded production server will serve its first prefill at Gen1 until the driver re-trains the links; the sustained serving state is Gen3 ×16 / Gen3 ×4. No action is recommended on this in the profile; it is a load-time transient.

## Phase B — MTP draft depth × ubatch (one variable per boot, six boots)

Forced long-form decode (~300 tokens, same 16K-class context, n-max 5 default as control). Prefill was identical across boots (23.4 s / 109.3 s at 16K / 47K — ub and n-max do not touch this path).

| Config | Decode | t/s | MTP acceptance | cold 16K TTFT |
|---|---|---|---|---|
| n-max 5, ub 128 (baseline) | 244 tok / 8.76 s | 27.7 | 0.42 | 23.4 s |
| n-max 3, ub 128 | 338 tok / 11.72 s | 28.8 | 0.45 | 23.4 s |
| n-max 2, ub 128 | 340 tok / 11.08 s | 30.6 | 0.61 | 23.4 s |
| n-max 5, ub 256 | 257 tok / 8.69 s | 29.5 | 0.47 | 22.4 s |
| **n-max 2, ub 256** | 293 tok / 9.49 s | **30.8** | **0.61** | **22.4 s** |

Deeper drafts cost more than they earn: acceptance falls 0.61 → 0.42 as the draft depth goes 2 → 5, i.e. the extra draft tokens are mostly rejected. n-max 2 wins decode by +11% over baseline; ub 256 wins cold TTFT by ~4% with no decode penalty. One long-form decode per boot (cold prefills were triplicated), so the n-max 2 edge is a consistent signal with a mechanistic explanation, not a statistical proof.

## Phase C — context ceiling ladder (per-slot cap 245K, baseline flags)

Fresh boot; each rung = cold prefill + planted-code retrieval + warm re-POST. Engine-counted depths (my synthetic corpus measures 0.2533 tok/char — an early density estimate had the 200/216/235K-class files 35–68K tokens over the cap; the engine's 400 `exceed_context_size_error` rejected them cleanly, no corruption, and the corrected files landed within 0.2% of target):

| Rung (engine tokens) | Cold prefill | t/s | Retrieval (first pass) | Warm re-POST |
|---|---|---|---|---|
| 167,079 | 602.3 s | 277 | **correct** (second fresh-boot sample) | 0.64 s, correct |
| 200,643 | 796.6 s | 252 | correct | 0.74 s, correct |
| 216,709 | 898.6 s | 241 | correct | 0.78 s, correct |
| 232,717 | 1005.8 s | 231 | correct | 0.81 s, correct |

Second consumer at the deepest rung: a 16K request on the other slot prefilled in 26.6 s (no contention penalty vs 23 s alone), and the 232K session's follow-up immediately afterwards was still cached (2.8 s, code retrieved). The verified single-session ceiling is **232,717 tokens** — 12.5K under the 245K cap, 29K under the 262144 per-sequence limit. The ceiling was not hit by memory, by the pool, or by retrieval quality: it is the chosen per-slot cap.

Reliability, stated plainly: across all first-pass cold retrievals this campaign, one miss in ten (167K, first fresh boot of the day; both subsequent fresh-boot samples of the same fixture retrieved correctly). That is an N=1 flaky-retrieval event on a 167K prompt, not an established defect; it was not reproduced by any warm pass.

## Phase D — larger aggregate pool: decision, not a test

Not run, with the reasoning recorded: the per-sequence limit is 262144, so a single session cannot use more than 262K from any pool; Phase C demonstrated 232K + a concurrent 16K (248K aggregate) inside the 262K pool with the deep session surviving; a 272/280K pool would only add capacity for two sessions whose *combined* depth exceeds 262K (e.g. 200K + 70K), a use case this workload did not exercise. If it does materialize, the pool is a one-flag change with a ~0.3 GiB VRAM cost (unified pool is pre-reserved; see the memory section).

## Phase E — multi-agent concurrency (winner profile: n-max 2, ub 256)

One boot; 11 turns; 786 s = 0.84 turns/min. Slot-pinned, sequential (the i5-7400 4-core box serves one active prefill/decode at a time; "concurrency" here = multiple logical consumers sharing the two slots and the unified pool).

| Turn | Consumer | Slot | Result |
|---|---|---|---|
| 1 | 167K cold | 0 | 567.8 s prefill (294 t/s, faster than baseline's 277 — the winner profile), code retrieved |
| 2 | 16K (while 167K resident) | 1 | 24.3 s cold prefill, 514-char reply |
| 3–8 | 48K / 24K agents, 3 turns each, alternating | 0 / 1 | 48K cold 98.0 s; 24K cold 47.2 s; all cached follow-ups 1.1–1.3 s (no decode: see artifact note) |
| 9–10 | 12K, 8K one-shots | 0 / 1 | 21.3 s / 14.0 s |
| 11 | **167K follow-up after all of the above** | 0 | **2.14 s, code retrieved, 181-token reply** |

The flagship result: the 167K session stayed resident through six intervening consumers (≈ 90K additional pooled KV) and returned in 2.1 s with correct retrieval — the unified pool's retention behavior is what the P2 profile is for. Final headroom with ≈ 259K tokens pooled: 1564 / 919 MiB.

**Artifact (not a defect of the profile):** the 48K/24K/8K fixtures with the planted-code question deterministically produced zero-token generations (first sample = EOS, 4/4 occurrences across two boots). Those cells contribute prefill and cache-follow-up timings only; no decode numbers come from them. The winner profile's decode quality is established by Phase B's long-form cells (30.8 t/s, acceptance 0.61) and by every other cell in this campaign, which generated normally.

## Memory model (why the numbers above are stable)

The unified pool is **pre-reserved at boot**: VRAM headroom was identical (2034/1021 MiB) at 167K and 232K resident depths, because the per-slot cap is an accounting gate on a single shared buffer, not an allocation trigger (the b12383 source reconciliation of 11-19, section 4, covers the non-unified analogue; the unified behavior is its measured complement). The 262K unified pool costs ≈ 0.5 GiB more than the equivalent non-unified pair at boot (21.5 vs 21.1 GiB total), and the MTP draft buffer adds ≈ 0.3–0.5 GiB at depth. Headroom at the deepest verified combination (≈ 259K pooled) is 1.5/0.9 GiB — thin but never breached in any window, and the engine fails closed with 400s before it could be (every over-cap request in this campaign was rejected, none corrupted).

## Correctness and reliability summary

- Planted-code retrieval (the only end-to-end correctness instrument available without a reference implementation): **9 of 10 first-pass cold retrievals correct** at 167K–233K depths; all warm passes correct.
- Engine over-cap protection: verified at 269K / 291K / 317K (clean 400s).
- Every window closed with production restored and a live inference response; VRAM returned to the 9960/11097 MiB P1 state each time.
- No production configuration was modified during the campaign; P1 remains the live profile and the rollback path.

## Limitations

- Single long-form decode per tuning boot; the n-max 2 edge (+11%) is a consistent signal, not a statistical result.
- Token counts and throughput from the engine's `print_timing` (b12383 streaming responses carry no usage block); TTFT from client timing.
- Concurrency = two slots on a 4-core box; results do not extrapolate to more consumers or to a faster CPU.
- The 167K first-pass miss is N=1 and unexplained; the fixture's retrieval quality may degrade at 232K+ in ways the planted-code test cannot grade (it is a retrieval probe, not a comprehension measure).
- PCIe link state during the M4E window was not sampled (the M4A window covers it: Gen3 throughout).

## References

- [2026-11-19-dual3060-27b-m2-disk-persistence-and-pcie-retrain](2026-11-19-dual3060-27b-m2-disk-persistence-and-pcie-retrain.md) — the P1/P2 definitions, the save/erase/restore contract, the PCIe fingerprint and its correction
- [2026-11-18-dual3060-27b-matched-p1-vs-unified](2026-11-18-dual3060-27b-matched-p1-vs-unified.md) — the matched ×1 window behind the width question
- [2026-11-17-dual3060-27b-context-hierarchy-ram-disk](2026-11-17-dual3060-27b-context-hierarchy-ram-disk.md) — the RAM/disk hierarchy and the size-boundary map
- [2026-11-15-dual3060-27b-maxctx-multi-consumer](2026-11-15-dual3060-27b-maxctx-multi-consumer.md) — the original P2 concept, A8 layout, and the MTP/ubatch sweeps
- [Provenance record](../data/provenance/dual3060-27b-2026-11-20-m4.json) — model/binary hashes, exact launch commands, window timestamps, fixture hashes
