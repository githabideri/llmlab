# The 96 GB plan: what changes when the second SO-DIMM lands (report E)

## The hardware

One 48 GiB SO-DIMM becomes two, dual-channel DDR5-5200. Usable RAM
rises from ~44 GiB to ~90 GiB, and single-channel memory bandwidth
(~41.6 GB/s theoretical for DDR5-5200) roughly doubles to ~83 GB/s.
Everything in this report is a projection from the measured 48 GB
numbers; the plan ends with the exact measurements that settle each
question.

## The central reversal

The 48 GB fix (direct reads) exists because the 16 GiB file-tier
complement cannot fit in the page cache: 16 GiB complement vs ~15.3 GiB
headroom. At 96 GiB the same 24 GiB pin leaves ~60 GiB of headroom; the
complement fits with room to spare, the page cache holds it at 17.7
GB/s (measured warm-cache ceiling on this NVMe, 7.7x the cold-direct
rate), and the churn disappears. The direct-read win of 48 GB
therefore reverses at 96 GB: buffered (or the 96-correct auto) becomes
the better mode again. v0.1.45's `STRATA_FILE_CACHE_OLD_RULE` knob
restores the pre-#1194 sizing that is correct in this regime; the
cleaner fix is an upstream calibration of the keep-size to
complement/headroom. The plan pins whichever release carries the
calibration and measures both knobs, rather than assuming either.

## The three 96 GB configurations to measure

Same model, quantization, MTP, KV settings, batch 2, two deep
conversations (200K + 128K) as the 48 GB service:

| config | pin | file tier | expected prefill (200K fresh) | expected 2-session decode | RAM ledger |
|---|---|---|---|---|---|
| A: 24 pin (the 48 GB config, re-tested) | 24 GiB | 16 GiB warm-cached | ~300 to 400 s (cache absorbs the tail; ~1.7x faster than 48 GB direct) | ~1 to 5 tps per session (PCIe tail remains, but the 2x memory bandwidth halves its cost) | ~51 GiB used, ~35 GiB headroom |
| B: full arena (40 pin) | 40 GiB | none | ~110 to 150 s (the measured 800 to 1,020 tok/s arena rate; 422 s of 48 GB direct work becomes mostly compute) | the interesting case: all routed experts in RAM, no PCIe tail in slots, so the 0.9 to 5.6 tps two-consumer number should rise substantially (projection 10 to 20 tps per session; measure) | ~54 GiB used, ~30 GiB headroom — fits at 96, did not at 44 |
| C: 32 pin (middle) | 32 GiB | 8 GiB warm-cached | between A and B | between A and B | ~52 GiB used |

The headline prediction: **config B (full arena) becomes viable for
the two-session service at 96 GB** — it was the one config that
OOM'd at 48 (40 GiB pin + 5.6 GiB two-session KV + MTP + process + OS
= ~54 GiB > 44) and it is the only config that removes the PCIe
tail from the decode path, which is the remaining structural limit of
the 48 GB service. If config B's two-session decode lands at or above
what the consumers need (measured need: 200K and 128K conversations
with interactive turns), the 96 GB deployment is full arena, not a
scaled-up version of the 48 GB config. If it does not (slot scheduling
is the limit, not the experts), config A is the deployment and the
residual is an upstream issue (slot MTP, issue #465-adjacent work).

## Procedure

1. Install the second SO-DIMM, rebuild the NixOS hardware
   configuration (the dual-channel memory type), verify with
   `dmidecode`/`lscpu` and a STREAM-class bandwidth check (~80 GB/s
   sustained expected).
2. Rebuild the node with the candidate 96 GB config B first (arena,
   the existing 40 GiB pin value from the part-1 arena battery, MTP on,
   everything else as the 48 GB final config), in a separate pinned
   derivation; run the part-2 battery (32K/64K/128K/200K fresh, the
   two-consumer 512 battery, the 8-hour soak) against it.
3. If B's two-session decode disappoints, run the same battery on
   config A (24 pin, buffered/old-rule).
4. Deploy the winner; keep the loser as the rollback derivation.
5. The 48 GB direct config is then the previous-generation deployment,
   archived with its reports (A to D).

## What 96 GB does not fix

- The single-GPU MTP-in-slot limitation: if the two-consumer decode
  stays below what the consumers need even with all experts in RAM,
  the limit is the slot-decode design (no MTP acceptance in slots,
  one-at-a-time slot scheduling with preemption), which is upstream
  work, not hardware.
- The per-request prefill of a brand-new 200K session at arena rates
  is still ~110 to 150 s of compute; only the warm-cache and
  prefix-reuse paths make follow-ups cheap, as they do now.
- The model itself: the 32K-tok/s-class prefill of a larger future
  quant is out of scope; this plan is about the same model on the same
  GPU with double the memory.
