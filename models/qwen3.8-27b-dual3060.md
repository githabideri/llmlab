# Qwen3.8-27B (ISTA GSQ-RCO 3-bit, MTP) on the dual-RTX-3060 box

**Model:** Qwen3.8-27B — dense ~27.3B, hybrid SSM + full attention (17 of 66 layers), native MTP head
**Quant:** ISTA DASLab [GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) **IQ3_S-mtp (3.0-3.4 bpw)**, 12.1 GB, with the release's own BF16 mmproj projector (0.93 GB; served on CPU)
**Hardware:** the secondary GPU server — Intel i5-7400 (4C/4T), 32 GB DDR4, two RTX 3060 12 GB (one on a CPU x16 slot, one on a chipset slot running x4), PHB, **no P2P**
**Runtime:** llama.cpp **b12383** (stock), `--split-mode layer`; MTP via `--spec-type draft-mtp` (shared-weight head; f16 draft KV, 1.02 GiB at 256K/2)
**Status:** ✅ **Fallback (boot-resident card default since 2026-11-15)** — the fleet's 27B primary is the [dual-3090 vLLM endpoint](qwen3.8-27b-rtx3090.md); this box serves when that one is unavailable. The 35B brain and the 27B cannot co-reside, so this endpoint is cold-startable in ~13 s when swapped in.

---

## Current production config (256K/2, measured 11-14, re-verified 11-15/16)

`--ctx-size 262144 --parallel 2`, layer split `1,1` (effective ~33/33), q4_0/q4_0 KV, MTP on, `-b 256 -ub 128 -t 4`, `--jinja --load-mode none`, projector on CPU. Idle VRAM 10.4/11.9 GB on the two cards; the lopsided 1.5 GiB / 40 MiB headroom split is the motivation for the campaign below.

## Max-context campaign (2026-11-15/16) — the Pareto set

Full measurements and method: [2026-11-15-dual3060-27b-maxctx-multi-consumer](../reports/2026-11-15-dual3060-27b-maxctx-multi-consumer.md). The three deployable profiles (adopting any is a two-line launch-script change; production was restored to the pre-campaign config after the campaign and stays there until the owner adopts one):

| profile | aggregate / per-session | the measured difference |
|---|---|---|
| **P1 Balanced multi-consumer** | 256K / 131K, 2 slots | GPU-order flip + 60,40 split + CPU projector → **1978/1012 MiB headroom on both cards** (was 1539/40) with no measurable decode penalty (24.8 → 24.8 t/s @16K); MTP n-max 2 (not 3) measured +14% decode (27.2 vs 23.8, acceptance 0.57 vs 0.37); `-ub 256` marginally the best prefill |
| **P2 Max context, flexible** | 256K unified, **per-session cap 200K** | `--kv-unified --kv-unified-per-slot 200000 --cache-prompt --cache-reuse 4096`: a 182K session plus a concurrent 30K session work with clean isolation; **conversation turns on the 182K session cost 1.6-2.1 s of prefill instead of 725-761 s** (99.97% cached); headroom 1704/874 |
| **P3 Largest stable aggregate** | **288K / 144K**, 2 slots | 128K-class retrieval verified; ~280 MiB post-deep headroom on the x4 card. 320K/2 boots but dies on its first prefill (`CUDA error: the resource allocation failed`) — the measured ceiling |

Negative results that closed levers: the MTP draft KV cannot be moved (`--spec-draft-device` is a measured byte-identical no-op for the built-in MTP; tensor override moves only ~40 MiB of head) and cannot be quantized (q8_0/q5_1/q4_0 draft KV OOMs at 256K and is dominated by f16 at 229K: +244 MiB, −4.5% decode). Slot count is not a memory lever in this build (the ctx size is the aggregate pool), and 4-way concurrent decode is plain time-slicing (~4.5 t/s per client vs 24 solo).

## Changelog (decision log)

- **2026-11-14:** qualified as the fleet's 27B fallback on this box — [campaign report](../reports/2026-11-14-fallback-dual-3060-27b.md) (engine verdict: llama.cpp layer split; vLLM TP2 CPU/PCIe-bound, PP2 W4A16 OOM; 130,690-token single-session edge; MTP 2× decode; reliability suite passed).
- **2026-11-15:** owner made it the box's **card default** (boot-resident; the 35B brain became on-demand on the same mux). Correction: the served projector is the GSQ-RCO release's own BF16 mmproj (sha256 `13cb7be…`); the 11-14 report had listed the Unsloth directory's file — both are BF16 projectors for this family.
- **2026-11-15/16:** max-context campaign (windows 19:45-21:07 and 21:28-00:15; production restored + live-verified after each). P1/P2/P3 measured and documented above; **production config deliberately unchanged** (still 1,1 + 256K/2) — adoption of P1/P2/P3 is an owner decision, rollback = delete the two A8 lines.
