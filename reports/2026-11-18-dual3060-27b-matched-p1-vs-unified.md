# Matched A/B on the dual-3060 27B: P1 as-is vs P1 + unified KV, one variable (2026-11-18)

Window 22:36–22:49 on the dual RTX 3060 12 GB box (B250 board, i5-7400
4C/4T, PVE 9.2.2; the second 3060 in the PCIEX4_1-class chipset port,
negotiating Gen3 ×1 — see the 2026-11-18 PCIe diagnosis and the model card).
Production was stopped for the window and live-verified after it. The point
of the window: isolate the **one variable** that differs between the
campaign's P1 (production) and P2 (the tried-and-reverted candidate) profiles
beyond ubatch/MTP knobs — the **unified KV pool** — and measure it against
P1 as-is, with everything else byte-identical.

## Setup

Both boots: the production launch flags verbatim — A8 layout
(`CUDA_VISIBLE_DEVICES=1,0` flip + `--tensor-split 60,40` +
`--mmproj-device none`), `--ctx-size 262144 --parallel 2`, q4_0/q4_0 KV,
`-b 256 -ub 128 -t 4`, `--load-mode none --jinja --metrics`,
`--spec-type draft-mtp` (default draft depth), llama.cpp b12383,
Qwen3.8-27B ISTA GSQ-RCO IQ3_S + MTP. Boot A = that exactly. Boot B = that
**plus `--kv-unified --kv-unified-per-slot 262144`** — the only difference.
Scratch port, systemd-scope isolation, readiness-gated (HTTP + model loaded +
real inference), watchdog with a 90-minute outer bound, production restore
and live verification at close. Total: 12 min 16 s, rc=0.

Fixtures: five seeded deterministic pseudo-text documents (plant a
verification code at ~40 % of the text; the model must recall it in a short
answer — a cheap, objective correctness check). Engine-measured lengths:
11,827 / 23,594 / 47,241 tokens (cold set) and two 23.6K-token documents
(agent pair). Cells per boot, on a fresh server: cold→warm repeats at all
three sizes, a 256-token solo decode, and two concurrent agents (simultaneous
32K-class prefills with code-recall answers, then simultaneous 128-token
decodes).

## Results (A = P1 as-is; B = P1 + unified KV; both at the ×1 condition)

| Cell | A first token | B first token | A end-to-end | B end-to-end |
|---|---|---|---|---|
| 11.8K cold prefill | 23.96 s | 23.90 s | 24.2 s | 24.5 s |
| 11.8K repeated (warm) | 0.455 s | 0.450 s | 0.99 s | 0.90 s |
| 23.6K cold prefill | 50.50 s | 50.61 s | 53.8 s | 53.9 s |
| 23.6K repeated | 0.491 s | 0.487 s | 0.78 s | 1.25 s |
| 47.2K cold prefill | 113.6 s | 114.1 s | 114.3 s | 114.8 s |
| 47.2K repeated | 0.578 s | 0.573 s | 4.21 s | 1.23 s |
| solo decode, 256 tok | — | — | 10.7 s (28.4 t/s) | 6.2 s (23.3 t/s) |
| two agents, 2×23.6K simultaneous prefill | — | — | wall 101.4 s | wall 106.2 s |
| two agents, simultaneous 128-tok decodes | — | — | wall 4.45 s | wall 7.09 s |

Cold prefill rates: 494 / 467 / 416 tok/s (A) vs 495 / 466 / 414 (B) —
indistinguishable. In every warm repeat, the engine processed **4 tokens**
(the API's `prompt_n` for the re-sent document): the full prefix was
reused in both configurations, first token in 0.45–0.58 s. Server logs show
why: repeated documents are routed back to the slot holding them by
longest-common-prefix similarity (`f_sim_best = 1.000`), cold documents by
LRU.

Two-agent prefill, per client: both boots show the same serialization —
agent a's first token at ~51 s (its own prefill), agent b's first token at
~100–101 s (b's prefill runs only after a's), b's 400-token answer then
completing at 28.8–31.8 t/s. Agent a's completion while b prefills took
77.2 s in A vs 106.2 s in B (a's decode ran at ~7 t/s in both, starved by
the concurrent prefill). All four code recalls correct in both boots; across
the whole window, **10/10 code-recall answers correct, zero failed turns**.

## Verdicts

1. **The unified KV pool is neutral on this box for cold prefill**
   (difference < 1 % at all three sizes) **and for prefix reuse** (identical
   warm behavior, full reuse in both). Reuse is a property of the prompt
   cache plus slot routing, not of pool unification.
2. **The decode and concurrency differences are single-run observations in
   the direction of the earlier P2 findings** (P2 measured ~15–20 % slower
   than P1 at 2×27K on 2026-11-15; the P2 canary of 2026-11-16/17 showed
   concurrent-stream starvation): solo decode 28.4 vs 23.3 t/s; the
   two-agent-decode wall 1.6× longer in B. One run each, not statistically
   established, no variable isolated beyond "unified on vs off" in those
   cells.
3. **P1 remains the production default.** This window was designed to find a
   reason to change it and found none: the one untested variable brought no
   prefill or reuse benefit.
4. **All November performance findings stand as ×1-condition results.** The
   window ran with the second 3060 at its measured Gen3 ×1 link (measured
   before and after the window; no in-window sample — the 5-minute external
   monitor's first clean tick landed 12 s after the window closed — and no
   event in the interval that can retrain a link). A true ×4 comparison
   still requires the physical fix (the box is not on the owner's site).
5. **The prefill link-utilization question is partially closed.** Decode
   traffic is far below any link ceiling (compute-bound, per the 2026-08-30
   "prove it with PCIe counters" lesson). For ≤47K prefill no transfer-rate
   sampling was taken in this window (an in-process sampler bug, recorded in
   the private notes) — arithmetic suggests low utilization, but that is not
   a measurement. The 167K-class case keeps its standing classification from
   2026-11-15: dmon observed 0.3–0.9 GB/s on this link, at or above the
   measured ×1 ceiling of 0.75 GB/s, so large prefills are "probably
   link-limited at ×1" — unconfirmed, and the one number that matters most
   for the onsite fix.

## What this unblocks

- A persistent 5-minute external PCIe monitor on the hub machine (polls the
  box over ssh: per-GPU negotiated width/generation, upstream root port and
  its link capability, freshness, change and baseline-deviation alerts;
  baseline is the documented ×1, not a hardcoded ×4) — in-window width
  evidence will be continuous from the next window on.
- The M2 workloads: 3–5-agent light/medium-load comparisons (fixed slots vs
  unified pool) and the restore-bottleneck isolation cells, using the same
  frozen fixtures and the same window protocol (scope + watchdog +
  live-verified restore), ~12 minutes of production time per window at this
  matrix size.

Raw per-cell JSON, both `-lv 3` server logs, the window log and the progress
trace are kept in the private repo (campaign `m3-*` files). Provenance:
`data/provenance/dual3060-27b-2026-11-18-m3-matched.json`.
