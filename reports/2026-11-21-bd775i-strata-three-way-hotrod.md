# BD775i three-way: Strata 0.1.41 vs upstream 0.1.45 vs the HotRod fork — the version upgrade is the finding

## Goal

Evaluate the public **Strata-HotRod** fork ([github.com/sangharshadhyeta/Strata-HotRod](https://github.com/sangharshadhyeta/Strata-HotRod), commit `d7f9dc4`) on the 48 GB / 12 GB node (RTX 3060, 8-core Zen4, single-channel DDR5), against the deployed Strata v0.1.41 and against plain upstream v0.1.45 — all on the production serve shape (32 GiB expert pin, int8 KV with the 20480-resident pool, batch 1, 208896 ctx, MTP drafter, O_DIRECT file tier) and the published IQ3_XXS pack. Pre-registered adoption bar (fixed before any result was seen): a candidate must be ≥20% faster sustained solo decode than the deployed build at **both** 128K and 200K, with no material quality regression and a clean 6 h soak. Concurrency: all cells single-client (the production lane is a single-user solo service); multi-client behavior is explicitly not in scope. The 48 GB regime is the intentional baseline; the 96 GB re-test (second 48 GB SO-DIMM, see the 2026-11-19 96 GB plan) is a later comparison, not a gate.

## Setup

Three Nix builds, deliberately identical underneath: same llama.cpp commit (`3cf03257`) in all three, nixpkgs CUDA 13.0, sm_86, the published IQ3_XXS GGUF (75.8 GB, two shards) with the native expert profile, the 9.4 GB processed pack and the BF16 MTP drafter. HotRod's own llama.cpp pin is the same commit, so the three-way isolates the Strata layer, not the backend.

Method:

- **Fresh engine process per measured cell** (no cross-cell state), three balanced orderings per arm (ABC / CAB / BAC), one 1024-token decode per cell on fixed public-domain text windows (32K / 128K / 200K occupied).
- **One-shot cells** (batched prefill, 16-token decode) at 4K / 16K / 64K.
- **Agent session**: a fixed five-turn conversation with accumulated history (21K-token opener, 256-token turns) to measure cached follow-up behavior.
- **Adaptive-state protocol (S2)**: HotRod learns per-machine (VRAM reserve, RAM reserve, online tuning, expert profile). Before every HotRod launch a calibration-start snapshot of the four learned-state files is restored and all four are hashed before/after. One file (the RAM reserve) still re-learned during runs, so HotRod numbers are "S2-start adaptive" — a labeled fallback, the only mode the engine actually supports.
- **H-ablation** (HotRod, 128K/200K, 2 reps each): H0 = stand-in routing disabled (exact routing, `STRATA_ROUTE_BUDGET=0`), H1 = fragile-token re-check disabled, H2 = full default — the decomposition of "approximation gain" vs "prefetch gain".

## Results

### Sustained solo decode (t/s, 1024-out, three balanced orderings; medians)

| occupied ctx | A: v0.1.41 (deployed) | B: v0.1.45 (upstream) | C: HotRod (S2-start) |
|---|---|---|---|
| 32K  | 21.2 / 22.3 / 23.3 → **22.3** | 30.8 / 30.9 (two of three) → **~30.9** | 24.3 / 26.9 / 25.0 → **25.0** |
| 128K | 20.1 / 21.0 / 21.0 → **21.0** | 29.2 / 29.1 / 29.1 → **29.1** | 18.1 / 16.1 / 16.6 → **16.6** |
| 200K | 21.1 / 22.8 / 22.7 → **22.7** | 28.0 / 27.9 / 27.9 → **27.9** | 11.2 / 11.8 / 11.1 → **11.2** |

Deployed-build deltas: B is **+38% / +39% / +23%** at 32K / 128K / 200K; C is **+12% / −21% / −51%**. The v0.1.41 column re-confirms the 2026-11-20 solo numbers (23.2 / 21.7 / 20.4; its 200K ran ~11% higher today, within day-level variance). B's long-context numbers are almost noise-free across orderings (±0.5%); A's 200K spread (~21.1–22.8) is the normal run-to-run band.

### Prefill

Fresh-prompt serve prefill (engine-measured tok/s, medians of the last three runs per cell):

| | 32K | 128K | 200K |
|---|---|---|---|
| A: v0.1.41 | 886 | 932 | 893 |
| B: v0.1.45 | 906 | 956 | 916 |
| C: HotRod | 709 | 794 | 737 |

B is a small, consistent +2.5% everywhere. C is **15–20% slower than the deployed build at every depth** on the chunked serve path.

One-shot prefill (whole prompt as one batch, cold process):

| tok/s | 4K | 16K | 64K |
|---|---|---|---|
| A: v0.1.41 | 111 | 250 | 356 |
| B: v0.1.45 | 114 | 246 | 345 |
| C: HotRod | **350** | **542** | **747** |

The inversion is the fork's clearest asset: HotRod is **2–3× faster on one-shot batched prefill** and cold-starts faster (~50 s to first token on 4K vs ~96 s for A, the one-shot TTFT including model load), while losing 15–20% on the small-chunk serve path. Its design (overlapped expert fetching / prefetch) evidently pays in the single-big-batch regime and is taxed per chunk in the serve regime; the H-ablation below probes how much of that is the machinery.

### Agent session (accumulated history, 21K-token opener, 256-token turns)

| | opener TTFT (s) | follow-up TTFT (s) | follow-up decode (t/s) |
|---|---|---|---|
| A | 26.5 | 0.82–0.96 | 17.5–23.8 |
| B | 30.5 | 0.52–0.60 | 29.5–35.5 |
| C | 33.1 | 0.58–0.65 | 24.2–28.1 |

Cached follow-ups are sub-second on all arms (the prefix path is healthy everywhere); at the agent's effective ~21.5K context the decode ladder tracks the main matrix (B ≈ 1.4–1.6× A, C in between at 32K-class but ahead of A by a wide margin only at short depth).

### H-ablation (attribution of HotRod's long-context deficit)

Early reading (128K, 2 reps each; the full 12-cell matrix follows in the next report): H0 (stand-in off, exact routing) 15.7 / 18.0 t/s ≈ H2 (full) 16.6–18.1. Disabling the approximation does **not** recover the deficit — the 128K gap to v0.1.41 exists in the fork's exact-routing mode too, so it is structural to the fork's engine on this node, not an artifact of approximate routing. (H0 also being only ≈ H2 says the stand-in mechanism is not what the fork buys you here at all.)

### The 2026-11-19 null, explained

The 11-19 engine comparison measured 0.1.45 as a null on this box — in a one-shot shape (no kv-resident pool, no MTP, 40K context cap). That shape is not where the 0.1.45 delta lives: in the production serve shape it is +23–39% across all depths, with the gap *widening* with context. The version delta is real, shape-dependent, and was missed by the wrong test shape.

## Conclusion

- **v0.1.45 (B) clears the pre-registered adoption bar on speed** at every depth (+38 / +39 / +23), with unusually stable long-context numbers. The remaining gates before the production swap are the quality battery (math/code probe set, paired per-probe deltas, no material regression) and the 6 h soak; the swap itself is a one-line flake pin change in the solo unit with a one-line rollback. The MTP draft-acceptance rate is lower in 0.1.45 than in 0.1.41 (43–60% vs 53–61% at 32K), which the net decode still beats — recorded so the acceptance metric is not misread in production logs.
- **HotRod (C) fails on this node at long context**: −21% at 128K, −51% at 200K, stable across all three orderings, and not attributable to its approximation mode. Its 32K gain (+12%) is below the bar. What it does have — 2–3× one-shot batched prefill and a fast cold start — is a genuine asset for batch/RAG-style workloads on this class of machine, and is recorded here even though it does not offset the solo-decode regression for the production lane.
- **IQ3_S on-label**: the 83.6 GB on-label quant (ISTA GSQ-RCO) pack is on the box's NVMe and will run as a separate result family (0.1.45 vs HotRod vs 0.1.41 on the on-label quant, with HotRod's shipped IQ3_S predictor) after this campaign closes.
- **Scope**: single-concurrency results, 48 GB regime, one 12 GB card. Nothing here claims multi-client behavior, and the 96 GB re-test may shift the trade (the arena-vs-pin balance flips there, per the 11-19 series).

## Update (post-publication append)

The full H0/H1/H2 matrix (128K and 200K, 2 reps each mode) and the quality-battery result land in the follow-up report; the 128K early reading above is the only ablation data in this snapshot.
