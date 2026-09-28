# Phase 4 — ISTA 2-bit 35B on the Pascal laptop: the ~100K performance-ceiling campaign and the production cutover (2026-09-28)

*Follows [Phase 2 — expert-residency campaign](2026-09-27-ista-2bit-expert-residency-pascal-1060-phase2.md) and [Phase 3 — adversarial quality gate](2026-09-28-ista-2bit-quality-gate-pascal-1060-phase3.md). Machine: the [Pascal laptop](../docs/hardware/pascal-laptop.md) (i5-7300HQ 4C/4T no-HT, GTX 1060 6 GB Max-Q, 24 GB single-channel-equivalent DDR4, 580 LTSB + CUDA 12.9 + 2026 llama.cpp moe-cache fork).*

## Goal

Phase 2 showed the 2-bit beats the 4-bit by ~47% on this DRAM-bandwidth-bound box; Phase 3 showed quality is a paired tie (38/54), with the 2-bit *better* at code/structured output. Phase 4 had one question: **what is the fastest robust configuration at ~100K context** — the context ceiling users actually need, not the theoretical 128K — across four levers the earlier phases never swept: **expert-cache size at two context points, ubatch/batch (prompt processing), flash-attention path, and KV-cache quantization** — then cut production over to the winner.

## Setup

- **Model:** `Qwen3.6-35B-A3B` ISTA GSQ-hybrid 2-bit (12.2 GB GGUF, Q2_0 family), text-only.
- **Server:** the 2026 `llama-server` in classic mode (single model, `--load-mode none`), campaign port 8099; **production (4-bit, port 8080) was stopped for the window** — the 6 GB card is single-tenant and this build has no unload/load HTTP API, so campaigns run with production offline and are restored afterwards.
- **Base flags** (from the Phase-2/3 shipping config): `--fit off -t 4 -np 1 -ngl all -ncmoe 20 --moe-early-router --decode-overlap --backend-sampling --phase-aware-workspace`.
- **Measurement protocol:** streaming chat completions, **thinking ON** (the production regime), `chat_template_kwargs` to control it. Decode rate (tg) = completion tokens ÷ (total − TTFT) on a **16K-token prompt, 256-token generation**; prompt rate (pp) from TTFT on 1K/2K/8K/16K prompts. Ladder points: one run each (Phase 2 established ±0.1 t/s run-to-run repeatability at fixed config); isolation points (C, E): 3-run medians.
- **VRAM rules:** peak ≤ 5,600 MiB = production-eligible; 5,600–5,920 = "risky" (works, no margin); > 5,920 or OOM = excluded. (Usable total: 5.92 GB.)
- **Two driver bugs fixed before data was trusted:** the 2026 server in `--models-preset` mode *is itself a router* (it needs `-m` for a classic single-model server — otherwise 0 models and every request 404s), and **thinking tokens stream as `reasoning_content`, so a small `max_tokens` can be entirely consumed by thinking** — a parser that only watches `content` sees "no content chunk" on a healthy server.
- **Rollback snapshot** taken before the first change (INI, units, dmesg-GPU, baseline inference, GPU state).

## Stage A — expert-cache ladder at 128K (q8_0 KV)

| cache | peak VRAM | tg (16K, thinking) | ok |
|-------|----------|--------------------|----|
| 24 | 5,095 MiB | 15.56 | yes |
| 32 | 5,335 | 16.73 | yes |
| 36 | 5,575 | 16.84 | yes |
| 40 | 5,575 | 17.53 | yes |
| 44 | 5,815 | 17.63 | **risky** |
| 48 | 5,815 | 17.84 | **risky** |

**The 128K cliff is between c40 and c44.** Gains flatten hard past c40 (+1.2% from c40→c48 at the price of all the remaining VRAM margin).

## Stage B — expert-cache ladder at 98,304 ctx (q8_0 KV)

| cache | peak VRAM | tg | ok |
|-------|----------|----|----|
| 32 | 4,967 | 16.60 | yes |
| 36 | 5,207 | 16.99 | yes |
| 40 | 5,207 | 17.33 | yes |
| 44 | 5,447 | 17.58 | yes |
| 48 | 5,447 | 17.60 | yes |
| 52 | 5,687 | 17.82 | **risky** |

**The 98K cliff is 4–8 slots further out (c48/c52)** — trading 30K of context buys ~8 more hot-expert slots. The catch: c48@98K (17.60) only beats c40@128K (17.53) by **0.4%** — inside run noise. The expert cache is **saturating around 40 slots**; beyond that the extra slots hold experts the routed set barely touches. **Context is free on this card** (128K adds ~1.15 GB of q8 KV for the 10 full-attention layers; the DeltaNet state is context-independent) — the 30K of context is worth more than the 0.4%.

## Stage C — `--decode-boundary-overlap` isolation (98K/c48, 3 runs each)

| variant | median tg | runs |
|---------|-----------|------|
| plain | 17.70 | 17.70 / 17.47 / 17.87 |
| boundary overlap | 17.90 | 17.90 / 17.74 / 17.95 |

+1.1% median with distributions that overlap run-to-run (run spread ±1.2%) → **not a reproducible win; flag dropped** (simpler production wins ties). The flag exists in the build (hidden from `--help`) and is safe, but it earns no place in the config.

## Stage D — ubatch / batch sweep (98K/c48, q8_0)

| ub | pp @ 2K | pp @ 16K | peak VRAM | ok |
|----|--------|----------|-----------|----|
| 512 (previous production) | 204.6 | 187.7 | 5,447 | yes |
| 768 | 203.1 | 205.6 | 5,483 | yes |
| 1024 | 204.8 | 237.3 | 5,525 | yes |
| 1536 | 205.2 | **243.5** | 5,591 | yes |
| 2048 | 205.7 | 275.4 | 5,637 | **risky** |
| 2048 with `-b 4096` / `-b 8192` | — | 276.2 / ≈276 | 5,637 | risky |

**Ubatch is the single biggest prompt-processing lever on this card: +47% pp from 512→2048 for +190 MiB VRAM** — Pascal's slow per-token path means the small default 512-token microbatch leaves the GPU starved during prefill. `-b` (outer batch) adds nothing once `ub ≥ 2048` (the microbatch is the binding chunk). Decoding is unaffected (ub only shapes prefill chunks). `ub 1536` is the production point: 97% of the ub2048 pp gain with 40 MiB more margin; `ub 2048` is the bench/edge option.

## Stage E — flash-attention path (98K/c48/ub1536, 3 runs each)

`-fa auto` (median 17.92) vs `-fa on` (median 17.82): identical within noise — **on this build the auto path already selects the same kernel for q8_0 KV on Pascal**. Ship `-fa on` explicitly (stable, self-documenting) rather than relying on the auto heuristic.

## Stage F — KV-cache quantization ladders (q5_0 / q4_0 at 98K and 128K)

| KV | ctx | best safe cache | peak VRAM | tg | Δtg vs q8 |
|----|-----|-----------------|-----------|----|-----------|
| q8_0 | 98K | c48 | 5,447 | 17.60 | — |
| q5_0 | 98K | c60 | 5,567 | 14.16 | **−20%** |
| q4_0 | 98K | c60 | 5,447 | 15.88 | **−12%** |
| q5_0 | 128K | c52 | 5,585 | 13.86 | −24% |
| q4_0 | 128K | c52 | 5,429 | 15.20 | −15% |

The KV quantization **does exactly what theory promised for VRAM** (q4 at 128K frees ~600 MiB — c40–c52 all fit, and the cache cliff moves out again) **and exactly what it promised against decode: it costs decode.** On this build q5 KV decodes ~20% slower than q8 and q4 ~12% slower, at *flat* pp — the coarser KV quantizations are simply slower attention kernels here (the q8 path is the fast one; note q4 even beats q5, i.e. this is a kernel-implementation characteristic, not a bandwidth effect). **Decision: q8_0 KV stays.** The q4-128K tier is documented, not adopted: it is a context-maximizing profile for the user who would rather have 128K and 52 cache slots at −15% decode, not a production point.

## The winner (promoted to production 2026-09-28)

| param | value |
|-------|-------|
| model | Qwen3.6-35B-A3B, **ISTA GSQ-hybrid 2-bit** (12.2 GB), text-only |
| context | **98,304 (96K)** — the "~100K" target; 128K costs ~230 MiB KV for 30K tokens the fleet doesn't use |
| KV cache | **q8_0 / q8_0** (q4/q5 measured 12–20% slower decode on this build) |
| expert cache | **48 slots** (98K cliff is c48/c52) |
| batch / ubatch | `-b 2048 -ub 1536` (pp @16K: 243.5; ub2048 = +13% pp, no margin) |
| flash-attn | `-fa on` (auto resolves to the same path; explicit is the stable choice) |
| rest | `--fit off --load-mode none -t 4 -np 1 -ngl all -ncmoe 20 --moe-early-router --decode-overlap --backend-sampling --phase-aware-workspace` |
| VRAM | **5,343 MiB steady / 5,591 peak** (329 MiB under the physical 5.92 GB ceiling) |
| host RSS | **~10.3 GB** (vs 15.6 GB for the 4-bit — a 5.3 GB difference that is what makes 16 GB RAM viable) |

**Measured:** ~17.6–17.9 t/s decode at 16K filled (thinking on); **19.6 t/s on short-context production traffic** (context adds no decode penalty, reconfirmed); ~243–245 t/s pp at 16K. Vs the outgoing 4-bit production (11.8–12.7 t/s, 167–186 t/s pp): **decode +42–50%, prompt processing +35–45%, 5.3 GB less host RAM.**

**Serving:** the 2026 `llama-server` in router mode (`--models-preset`, `load-on-startup`, resident at boot) as model `qwen36-35b-96k` on the hub's `llama-router` card; the outgoing 4-bit/128K profile is preserved as a one-file INI swap fallback (its ~15.6 GB host-RAM steady state is why it no longer fits comfortably on a 16 GB dual-channel box — the 2-bit does).

## Also settled in this phase

- **The Phase-3 "per-request RAM leak" is retracted.** A passive 60-second `/proc` sampler on the live production worker (hub polling at 30 req/min plus real traffic) ran 6.7 hours: RSS was **byte-stable at 15,630 MiB from the first sample** — a one-time ramp to a steady-state footprint, not per-request growth. The overnight 4-bit OOM was a *budget* problem (15.6 GB worker + OS + page cache > 24 GB during a heavy load spike), not an unbounded leak — and it is structurally fixed by the 2-bit cutover (~10.3 GB steady). The watchdog (tripwire at 17 GB, max-RSS child, GPU-idle-only, 1 h cooldown) stays as cheap insurance.
- **16 GB (2×8) feasibility:** host demand is context-independent (the KV lives in VRAM), so the 2-bit's ~10–13 GB fits all contexts on 16 GB with 2–3 GB margin; the 4-bit's ~15.6 GB does not. A dual-8 swap is the next experiment — decode is the one number that dual-channel should move (single-channel DRAM wall ≈ 19–20 GB/s measured; 2-bit demand ≈ 1.2 GB/token).

## Open items

1. **Dual-channel re-measure** (2×8 GB pending purchase): decode is the predicted mover; the 2-bit should roughly double until the next bottleneck (PCIe expert prefetch, 4-core serialization) appears. Only the measurement settles it.
2. **128K tier:** reachable via the q4-KV trade (−12–15% decode) or by accepting c36/ub1024 at q8 — a user-choice profile, not the default.
3. The 3-bit IQ3_XXS remains *unmeasured, not dominated*: no quality edge over the 4-bit and no speed edge over the 2-bit on this box.
