# llama.cpp mainline `781dbc5a` vs production `925e117` vs Strata: the Re-A/B on the 3060

**Date:** 2026-10-10
**Status:** measured three-way comparison, one window per arm; not a production cutover. As of writing, the box's production router is still build `925e1179`; the newer build is built, tested, and staged for adoption.
**Related:** [2026-10-09 Strata vs llama.cpp A/B](2026-10-09-strata-qwen36-35b-a3b-single-3060.md) (same fixtures, same box).

## Question

This box already runs both engines for the same model (see the 10-09 report: Strata's budgeted mode leads decode 1.2–1.5×; llama.cpp keeps its long-prompt TTFT edge). Since that report, two things changed on the llama.cpp side: the production build here (`925e1179`, 2026-08-26) is now ~6 weeks stale, and mainline merged PR #29887 on 2026-10-07 — the one mainline change aimed at exactly the mechanism behind the gap (a GPU cache for the MoE experts that live in host memory). Two questions: **how much of the measured gap does a fresh mainline build close on this box — and can the new feature be deployed on a 12 GB card at all?**

The PR, for the record: a community contribution (author `am17an`; a port of the qvac-fabric MoE cache). MUL_MAT_ID ops on host-resident experts run on the GPU against an **LRU** expert cache; only the misses are uploaded. It applies to **small batches only (≤32 tokens)** — i.e. the decode path; larger batches keep the regular offload. Layers with different expert layouts get separate banks. Off by default; `--moe-cache-mib N` enables it, with the author's experiments suggesting ~10% of the total expert size as a good budget. Its own benchmarks use a 93.7 GiB Q4_0 model on an RTX 4090/5090 with 6.4–19 GB of cache — a different machine class from this box.

## Setup

- **llama.cpp NEW (B):** mainline `781dbc5a` (2026-10-10), built in the same 4-core LXC as the production server with the production flags verbatim (Release, `GGML_CUDA`, `GGML_NATIVE`, gcc 13.3.0, CUDA 13.1). Served through the box's model-muxing proxy exactly like the production build. Note: the version banner no longer carries the commit stamp in this revision (`0.6.0-dev, commit unknown`) — provenance is the source tree, not the banner.
- **llama.cpp OLD (A):** the production build `925e1179` (2026-08-26), unchanged.
- **Strata (S):** unchanged from the 10-09 report (PR #1402 / v0.1.41 base, budgeted mode, on the host side of the box — the in-CT placement dies on long-prompt cuBLAS calls, a separate open issue).
- Same model as both earlier reports: Qwen3.6-35B-A3B UD-Q4_K_XL, 128K context, 28 of 40 MoE layers on CPU, MTP draft. Same fixtures and cells as the 10-09 A/B (C1 ~50-token prompt, C2 6.8K-token read, C3 13.5K-token read; streaming; one run per cell), measured at 10:36–12:00 UTC on 2026-10-10.

One comparability caveat: the two llama.cpp arms were measured in the LXC placement and the Strata arm on the host placement (forced by the CT cuBLAS anomaly). Decode cells are steady-state and the VRAM layouts are near-identical (~11.5–11.7 GB), so the decode comparison is sound; absolute TTFT carries a placement difference in both earlier reports already.

## Results

| Cell (streaming) | A: old `925e117` | B: new `781dbc5a` | S: Strata (10-09, same window family) |
|---|---|---|---|
| **C1** (~50 in, 128 out) — TTFT | 1.23 s | 1.55 s | 1.07–0.80 s |
| C1 — decode | 36.3 tok/s | **41.9 tok/s (+15%)** | **52.6–53.4 tok/s** |
| **C2** (6.8K in, 64 out) — TTFT | 8.0 s | 7.7 s | 10.9–16.3 s (cold 1st) |
| C2 — decode | 41.1 tok/s | **48.3 tok/s (+18%)** | **46.7–58.0 tok/s** |
| **C3** (13.5K in, 128 out) — TTFT | 13.9 s | 9.6 s | 10.3–11.0 s |
| C3 — decode | 37.1 tok/s | 37.5 tok/s (±0) | **53.0–58.7 tok/s** |
| Non-streaming C3 wall | 6.3 s | 6.0 s | 12.8 s |

The Strata column is the 10-09 measurement (the engine was re-run this window too; its C1 decode measured 53.4, in-band with the 10-09 figure). Ratios, new build vs Strata: **1.27× at C1, 1.11× at C2, 1.56× at C3** — versus 1.47×/1.31×/1.28–1.58× for the old build in the 10-09 report.

## Findings

1. **The rebuild narrows the decode gap but does not close it.** Short-prompt and 8K decode move from ~1.3–1.5× to ~1.1–1.3× in Strata's favor; at 13.5K the decode is unchanged (both builds ~37 tok/s — at that context length the bottleneck is no longer the expert path). llama.cpp keeps its long-prompt TTFT edge (9.6–8.0 s vs 10–16 s). The newest mainline is the better llama.cpp for this model, but on this box the two engines remain complementary (Strata for chat decode, llama.cpp for long-document first-token), not interchangeable.
2. **The new feature (`--moe-cache-mib`, PR #29887) is off by default and does not fit this card.** The llama.cpp arms above ran without it. It cannot be turned on here: with the model plus 128K KV loaded, ~575 MiB of the 12 GB is free, while the MTP draft context alone needs a 672 MiB allocation — a 384 MiB cache dies with `cudaMalloc failed: out of memory` at MTP-context creation, and the newer build **exits the entire router process on a failed model load** (the older build merely drops the model — an operational difference if you adopt it). The maximum that would fit is ≈128 MiB ≈ 2–3 Q4 experts (~20% of the per-token active set) — a marginal cache; the author's ~10%-of-experts budget for this model would be ~1.8 GB, roughly 3× the card's entire headroom. The feature's payoff assumes VRAM headroom (its benchmarks: 6.4–19 GB of cache); on a 12 GB card at 128K context it is structurally out of budget. Note also the mechanism gap versus the engine it would catch up to: the merged cache is a reactive LRU (misses uploaded on demand, small batches only), while the 10-09 report measured a profile-ranked cache with anticipatory streaming on the prompt path (hit rate climbing 58→86% within a run, only 4–12% of routed experts fetched over PCIe) — the merge covers one building block of that design, in its simpler form.
3. **Operational differences of the newer router (re-validation required before adoption).** The `/models/load` / `/models/unload` endpoints still exist (verified in source, `tools/server/server.cpp`), but one mux-driven exclusive switch in this window produced no switch-log lines at all and left the card wedged on the wrong engine until a manual restore — the box's muxing proxy (built against the older router) is not re-validated against the new one in either switch direction. The version banner drops the commit stamp. On-demand loading semantics differ slightly from the older build.

## Verdict

On this 31 GB / 4-core / RTX 3060 box, a fresh mainline build is a real but partial move: +15–18% decode at short and mid prompts, no change at 13.5K, and the long-prompt TTFT edge stays with llama.cpp. The merged expert-cache feature — the mainline change aimed at exactly this workload — cannot be deployed here (no VRAM headroom at 128K context), and it is a narrower mechanism than the one it would catch up to (reactive LRU vs profile-ranked with anticipatory streaming). Practical outcome: the two-engine status quo survives — production llama.cpp as default, Strata as the selectable chat-heavy alternate — and adopting the newer build is a normal maintenance decision gated on re-validating the muxing proxy, not a Strata retirement trigger.

*Sanitization: no internal host names, IPs, or machine identifiers appear in this report; "the box" is the 31 GB / i5-7400 / RTX 3060 LHR Proxmox host of the 10-09 report, and "the LXC" is its 4-core container placement for the production 35B.*
