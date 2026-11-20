# BD775i single-user long-context decode: the 32 GiB pin is the ceiling, not the arena

## Goal

Fastest reliable single-user Qwen3.8-Flash-Next IQ3_XXS decode at 128K and
200K occupied context on the 48 GB / 12 GB node (RTX 3060, 8-core Zen4,
single-channel DDR5), with the trained MTP drafter, no quality loss, and
the existing two-consumer service preserved as a separate profile. The
standing two-consumer number (~10 tps at 200K) was explicitly not treated
as the hardware ceiling.

## Setup

Strata v0.1.41 (pinned Nix derivation, llama.cpp 3cf03257), the
published IQ3_XXS GGUF (75.8 GB, two shards) with the native expert
profile and the BF16 MTP drafter. All arms in exclusive GPU windows, one
engine process at a time, same model/quantization/tokenizer/routing
throughout (the two routing-approximation env vars do not exist in this
engine version; prefill CPU share off, unbuffered file-tier reads).
Six arms:

| arm | experts | KV | batch | MTP |
|---|---|---|---|---|
| A | 24 GiB pinned (the production config) | int8, streamed (kv-resident 20480) | 2 | yes |
| B | 24 GiB pinned | same | 1 | yes |
| C | 32 GiB pinned | same | 1 | yes |
| D | 36 GiB pinned | same | 1 | yes | (not run: see below)
| E | full arena (37.3 GiB pinned, no file tier) | int8 in VRAM | 1 | no |
| F | full arena | same | 1 | yes |
| Cnm | 32 GiB pinned | same | 1 | no | (MTP ablation)
| C2 | 32 GiB pinned | same | 2 | yes | (two-slot cost)

D (36 GiB pin) was not run: the 32 GiB arm already measured 41 GiB of
44 used with 3.1 GiB headroom, so a further +4 GiB of pinned experts
exceeds the machine's budget by measurement rather than guess; one
confirmed resource limit ends that line. The engine also clamps the
requested 40 GiB arena to 37.3 GiB (the part of the expert set the GPU's
own cache does not hold), so "arena" is 37.3 GiB of pinned RAM plus the
GPU cache.

## The solo matrix (sustained tok/s, medians of 3, 1024-out unless noted)

| occupied ctx | A (24p, b2) | B (24p, b1) | C (32p, b1) | E (arena, no MTP) | F (arena + MTP) |
|---|---|---|---|---|---|
| 32K | 10.5 | 13.2 | 23.2 | 20.4 | 21.6 |
| 64K | — | — | 20.2 | — | — |
| 128K | 10.2 | 12.3 | 21.7 | 19.7 | 19.4 |
| 192K | — | — | 20.4 | — | — |
| 200K | 10.1 | 11.1 | 21.6 | 19.7 | 19.6 |
| 200K, 512-out | 11.3 | 12.9 | 22.0 | 19.2 | 19.6 |

Run-to-run spread is 4 to 15 percent. The 24 GiB production pin is
~2x slower than every configuration that pins at least 32 GiB; the
arena (no file tier at all) is not faster than the 32 GiB pin — it is
5 to 10 percent slower at 128K/200K.

## Where the time goes (the attribution)

Per 1024-token decode, physical NVMe reads: ~50 GB at the 24 GiB pin
(arm A), ~3-5 GB at the 32 GiB pin (C/Cnm), ~20-50 MB in the arena
(E/F). At the NVMe's ~1.2 GB/s effective rate, the 24 GiB pin's file
traffic alone costs ~40 ms of every ~100 ms decode period, which is
exactly the ~10 tps that configuration produces; the 32 GiB pin's
residual ~4 ms of file traffic is a small fraction of the period.

The residual wall (19 to 23 tps, common to the 32-pin and arena
configurations, essentially flat from 32K to 200K) is the per-token GPU
expert pipeline: the 12 GB card's VRAM expert cache holds the hot
majority of the routed experts, and the engine reports that **20 to 22
percent of all routed-expert lookups are still streamed into that cache
over PCIe on each use** (decode expert-cache hit rate 67 to 72 percent).
Shrinking the PCIe share (pcie-frac 0, forcing the remainder onto the
CPU) measures ~35 percent slower in the earlier reproduction battery, so
the engine's default split is near its optimum; the card simply cannot
hold more of the ~6 B active-parameter expert set per token.

KV attention is not the limit: 200K runs at the same rate as 128K and
32K, and the int8 KV for 200K fits in the card's remaining VRAM without
shrinking the expert cache (peak VRAM is identical at every depth,
11.64 to 11.66 GB of 12.29).

**MTP at depth buys almost nothing here.** With the drafter on,
acceptance is 0.42 to 0.50 per draft token at 128K/200K, but the
verification cost scales with what it accepts, so the net effect is
+1 to 2 percent at 200K (32-pin: 22.0 with vs 21.3 without; arena:
19.8 vs 19.7) and +2 to 6 percent at 32K. The earlier campaign's
~1.7x MTP speedup at short prompts does not extrapolate to deep
context on this machine. The drafter is kept in the final profile
because it is the fast path for the short conversational turns that
alternate with long context, and it costs nothing at depth.

## The batch-2 question (is the penalty reversible?)

With one active client, reserving a second batch slot costs: at the
24 GiB pin, 10 to 21 percent (A vs B); at the 32 GiB pin, ~25 percent
at 128K and ~30 percent at 200K (C2: 16.4 / 15.0 vs C's 21.7 / 21.6) — the
second slot's pinned KV and its scheduling machinery squeeze the tighter
32 GiB ledger harder. With two simultaneous 512-token decoders on the
32-pin batch-2 config, the consumers get 2.1 and 9.9 tps respectively
(the smaller conversation is preempted into the slot) — much better
than the 24-pin two-consumer deployment (0.96 / 0.8 tps), but a
different service than the solo profile. Conclusion: the solo gain does
not survive a second reserved slot; the two profiles are genuinely
different machines' worth of behavior and are kept separate.

## Long-context correctness (finalist, 32-pin)

Thinking disabled, 1024 budget, answers checked against the text:

- 32K and 64K: "what is the stranger carrying?" -> "the identical New
  Zealand head" (ch. 11, ~18K in) — exact, both depths.
- 128K: the Ahab-name conversation (~70K in) -> attributes the name to
  Ahab's widowed mother and names the old squaw Tistig at Gay Head —
  both parts correct.
- 192K and 200K: the Tahitian sailors' passage (~150K in) ->
  "Heeva-Heeva" — the exact two-word name, both depths.
- A 1.7K-token synthetic code module with a planted sliding-window
  off-by-one: the model quotes the wrong line, explains both reported
  wrong values arithmetically, and gives the correct fix.
- A 300-token 15-day running-total arithmetic problem: correct running
  totals through the visible steps (final 146 verified by hand).

No depth-related degradation; the 200K answers come from regions of the
context that were in the prompt-cache (earlier turns) as well as the
fresh tail, so the KV of the whole window is being used.

## Stability (6 h soak, finalist)

Thirty 10-minute-apart continuation turns on two live conversations
(128K start and 200K start, each turn resending full history, 256-token
corrected-answer replies): **60/60 track-turns clean, zero errors, zero
OOM, PSI full never above 0.05, RAM flat at 41 GiB used / 3 GiB
headroom, VRAM 11.6/12.3**. Per-track decode stayed in 14.9 to 25.7
tps (median 20.6) across all 30 turns. The 200K conversation grew from
204,856 to 206,945 tokens (the 208,896 configured ceiling was never
touched); the 128K track from 128,051 to 130,016. The per-turn
replies remained correct Moby-Dick content throughout (the growing
history was re-prefilled or cache-matched every turn, so this is a
true live-long-context test, not a repeated identical prompt).

## Memory ledger of the fastest safe 200K configuration (arm C)

| item | size |
|---|---|
| pinned experts (32 GiB of the 35.98 GiB GPU-cache complement) | 32.0 GiB |
| streamed KV (208,896-cell budget, int8, non-resident window in pinned RAM) | ~2.5 GiB |
| MTP drafter (host side) | < 0.5 GiB |
| process + kernel/OS | ~3-6 GiB |
| **used / headroom** | **41 GiB / 3.1 of 44** |
| VRAM (weights + expert cache + KV + MTP, 200K) | 11.64 / 12.29 GB |

PSI full ~0, zero swap in all arms; the arena arms leave 1.1 to 2.1 GiB
headroom, the 24-pin 6 GiB.

## What the second 48 GB SO-DIMM will change (prediction, not measurement)

Dual-channel DDR5 roughly doubles host memory bandwidth (~41.6 to
~83 GB/s). Predicted effects, in decreasing confidence:

1. The 32 GiB pin configuration (this profile) improves further: the
   per-token decode still moves ~3-5 GB of file-tier bytes and the PLE
   lookup table crosses host memory, both of which are bandwidth-bound;
   a 10 to 25 percent gain on the solo 200K rate is the reasonable
   range.
2. The arena (37.3 GiB pin) improves little: its residual is the PCIe
   expert-cache refill (20 percent of lookups) and GPU compute, neither
   of which is host-bandwidth-bound. The 12 GB card, not the RAM, is
   the arena's wall.
3. With 96 GiB the 36/40 GiB pins become feasible (they were excluded
   here by the 44 GiB ledger), but this campaign's ordering suggests
   the 32 GiB pin already sits at the same ~22 tps wall the arena does,
   so extra pin is not expected to move that wall either.
4. Two deep sessions (the current two-consumer service) become
   comfortable again (the 4.6 GiB of their combined KV plus headroom
   fits without the 24-pin compromise), but the two-consumer decode
   rate is slot-decode-bound (no MTP in slots, preemption) and is not
   predicted to change materially.

The measurement that would settle all of this is the same arm matrix on
the 96 GB machine; nothing above is a measurement.

## Deployment (done 2026-11-20)

Two independently selectable NixOS profiles, one pinned engine
derivation (v0.1.41, unchanged):

- **Dual Long** — `strata-serve`, port 8080, the existing two-consumer
  config (24 GiB pin, batch 2, streamed KV, MTP). Running, unchanged
  by this campaign, verified after the work.
- **Fast Solo** — `strata-solo`, port 8081, arm C (32 GiB pin, batch 1,
  streamed KV, MTP). Not started by default; the switch script
  (`switch-profile {solo|dual}`) stops the active profile, starts the
  other, waits for readiness, and prints a warning that the engine
  restart invalidates every in-memory session (nothing in this
  configuration persists across a restart — the supported
  session-save mechanism is disabled with batch mode, upstream #465).
  Both directions were tested end-to-end after the soak: the solo
  profile served a 32K turn in 39 s, then the node was switched back
  to Dual Long, which is the running default.

## Reproduction

Engine v0.1.41 tag (the Nix derivation in the node's flake), the model
as published, `--resident-experts --resident-budget-gib 32 --kv int8
--kv-resident 20480 --max-context 208896 --batch 1 --mtp <pack>/mtp-bf16/rt`,
env `STRATA_UNBUFFERED_LOAD=1 STRATA_PREFILL_CPU_SHARE=0`, one client.
Every number above is a median of at least three runs in an exclusive
window; the per-cell records (wall, TTFT, sustained rate, MTP
acceptance, NVMe bytes, major faults, RAM/PSI/VRAM) are in the campaign
data directory.
