# The Pascal laptop's RAM swap: a mixed 2×8 set, and why "dual channel" needs a matched pair

*Machine: the [Pascal laptop](../docs/hardware/pascal-laptop.md) (i5-7300HQ 4C/4T no-HT, GTX 1060 6 GB Max-Q, NixOS, AC-only). Follows [Phase 4 — the 2-bit 96K production cutover](2026-09-28-ista-2bit-100k-performance-ceiling-pascal-1060-phase4.md), whose open item 1 was exactly this: "dual-channel re-measure (2×8 GB pending purchase)".*

## Goal

The laptop ran 24 GB of *asymmetric* RAM (16 GB + 8 GB, two different vendors) at effective single-channel bandwidth (~19–20 GB/s STREAM, no thread scaling — the DRAM wall behind the 35B decode numbers). The question the swap was supposed to answer: **does a symmetric 8+8 set engage dual channel (~35–38 GB/s, thread-scaling), and does the 2-bit decode move with it?**

## Setup

One new stick arrived: **SK hynix 8 GB, 1Rx8, 4 × 16 Gb x16 chips, DDR4-3200 class** (chip marking `H5…NAG6NC…-XNC`). It replaced the 16 GB Hynix stick; **the 8 GB Kingston stayed in its slot** — so the set that went in is *symmetric in capacity, asymmetric in vendor/SPD*. After the swap, DMI reports:

| slot | module | rated | configured |
|------|--------|-------|------------|
| top | SK Hynix 8 GB, 1 rank | 3200 MT/s | 2400 MT/s |
| under | Kingston 8 GB, 1 rank | 2667 MT/s | 2400 MT/s |

Both banks distinct (0 and 2), both 1.2 V. The i5-7300HQ (Kaby Lake) caps the board at DDR4-2400, so the 3200-class stick downclocks with no penalty.

## Commands

- `dmidecode -t 16` (physical memory array: 2 devices, 32 GB max, no interleave data exposed) and `-t 17` (per-module table above)
- STREAM: the same OpenMP binary as the 09-27 measurements (`N = 50 000 000`, best of 20 reps), run at 1/2/3/4 threads
- production re-measure on the **live endpoint** (no service change; the `llama-server` router + worker stayed up): streaming chat completions, model `qwen36-35b-96k`, thinking ON (`chat_template_kwargs`), 16K-token prompt (two copies of the 8K Phase-4 fixture) / 256-token generation, 3 runs, plus a short-context run; plus a prefill-only probe (16K, `max_tokens: 1`)
- `free` before/after, `nvidia-smi` for VRAM/clocks

## Observations

- **STREAM: no change.** ~20 GB/s at every thread count, no scaling — the *same* single-channel-equivalent profile as the old 16+8 set. (The old set's flex-mode expectation — 8 GB dual + 8 GB single — would have shown more than 20 GB/s; it didn't, and neither did the mixed 8+8.)
- **The board is nonetheless proven dual-capable.** A public Geekbench 5 result for the same laptop model (board "HP 838F", i5-7300HQ, 16 GB DDR4) reports **two memory channels at 1197 MT/s** ([browser.geekbench.com/v5/compute/6748576](https://browser.geekbench.com/v5/compute/6748576)), and NotebookCheck's review of the 7th-gen Omen 15 family describes the 16 GB configuration as dual-channel. So the open question narrowed to: *does this BIOS interleave two unmatched 8 GB modules at all, or only SPD-matched pairs?*
- **Decode did not move.** 16K filled (thinking on): **15.26 ± 0.02 t/s** over 3 runs — below the Phase-4 baseline of 17.6–17.9 t/s, but Phase 4 measured that on a *campaign* server while this number is on the *production* router-mode endpoint: not an identical regime, so the delta is not cleanly attributable to the RAM. Short context: **20.08 t/s** vs 19.6 — flat, which is exactly what the unchanged ~20 GB/s wall predicts.
- **Prefill is the odd one out: ~1648 t/s at 16K vs ~243 t/s in Phase 4** (same binary — built 09-27 — and the same shipped flags). That is a GPU-side difference between the 09-28 campaign window and today's fresh boot (driver/P-state state, not RAM); worth a like-for-like re-run before believing either number.
- **16 GB fits.** ~4.5–4.9 GB available with the 2-bit resident (10.3 GB RSS). The 4-bit 128K fallback (15.6 GB steady state) does *not* fit on 16 GB — as Phase 4 predicted; its INI stays as a document, not a working fallback, until RAM grows to 2×16.
- Boot note (box-specific, not RAM): this reboot came up with the reset RTC — the model has **no coin cell**, AC-only; NTP re-syncs the wall clock, which briefly makes `uptime` print a nonsense multi-year figure. Cosmetic.

## Metrics

| point | 09-28 (Phase 4) | 09-30 (mixed 8+8) | Δ |
|-------|-----------------|-------------------|---|
| STREAM TRIAD, 4 threads | ~19–20 GB/s, flat | 19.6 GB/s, flat | none |
| tg, 16K filled, thinking ON | 17.6–17.9 (campaign server) | **15.26 ± 0.02** (production endpoint) | −14% (protocol-confounded) |
| tg, short context, thinking ON | 19.6 (production) | **20.08** (production) | flat |
| pp, 16K (ub1536) | ~243 | **~1648** (first non-cached run) | +6.8× (GPU-side, unexplained) |
| host RAM available, 2-bit resident | ~8 GB (of 24) | **~4.5–4.9 GB (of 16)** | fits, tighter |

## Conclusion

**The swap bought capacity headroom, not bandwidth.** The DRAM wall is exactly where it was: the BIOS will not interleave an unmatched 2×8 pair (nor did it flex the 16+8). The board itself is proven dual-capable, so the definitive test is a **matched pair** — a second 8 GB stick identical to the one now in the top slot (the physical label's part number; DMI doesn't expose it). If that engages ~35–38 GB/s, the 2-bit decode should move until the next wall (PCIe expert prefetch or 4-core serialization) appears. Until then the 09-28 Phase-4 numbers remain the reference, and the prefill anomaly (243 → 1648 t/s) is a separate, GPU-side open question.
