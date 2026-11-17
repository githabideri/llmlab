# Qwen3.8-27B (ISTA GSQ-RCO 3-bit, MTP) on the dual-RTX-3060 box

**Model:** Qwen3.8-27B — dense ~27.3B, hybrid SSM + full attention (17 of 66 layers), native MTP head
**Quant:** ISTA DASLab [GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) **IQ3_S-mtp (3.0-3.4 bpw)**, 12.1 GB, with the release's own BF16 mmproj projector (0.93 GB; served on CPU)
**Hardware:** the secondary GPU server — Intel i5-7400 (4C/8T), 32 GB DDR4, two RTX 3060 12 GB (one on a CPU x16 slot, one on a chipset slot running x4), PHB, **no P2P**
**Runtime:** llama.cpp **b12383** (stock), `--split-mode layer`; MTP via `--spec-type draft-mtp` (shared-weight head; f16 draft KV, 1.02 GiB at 256K/2)
**Status:** ✅ **Fallback (boot-resident card default since 2026-11-15)** — the fleet's 27B primary is the [dual-3090 vLLM endpoint](qwen3.8-27b-rtx3090.md); this box serves when that one is unavailable. The 35B brain and the 27B cannot co-reside, so this endpoint is cold-startable in ~13 s when swapped in.

---

## Current production config (P1, since 2026-11-16)

The **P1 profile of the max-context campaign is the production default** (adopted 2026-11-16 after a canary of P2 found a concurrent-stream regression; P2 was tried as the default for one afternoon and reverted). `--ctx-size 262144 --parallel 2`, **A8 layout** (`CUDA_VISIBLE_DEVICES=1,0` GPU-order flip + `--tensor-split 60,40` + `--mmproj-device none`), q4_0/q4_0 KV, MTP on (default n-max 5), `-b 256 -ub 128 -t 4`, `--jinja --load-mode none`, non-unified KV. Idle VRAM 9934/10899 MiB (headroom 1978/1012); the pre-campaign 1,1 launcher is preserved for a full restoration. P2 (unified 200K/slot + `--cache-prompt --cache-reuse 4096`, ub 256, n-max 2) stays a documented candidate: it gave 1.6-2.1 s turns on a 182K conversation while pool-resident, but (a) two concurrent long prefills do not stream in real time (~225 s first byte on the second, burst delivery, one early close — attribution between `-ub 256` / `n-max 2` / per-slot cap still open) and (b) the 2026-11-17 context-hierarchy run found that in b12383 the RAM prompt cache cannot restore evicted states beyond the ~150 MiB checkpoint class, so a 182K turn is fast only while its state stays in the unified pool.

## Context-persistence findings (2026-11-17)

Full measurements: [2026-11-17-dual3060-27b-context-hierarchy-ram-disk](../reports/2026-11-17-dual3060-27b-context-hierarchy-ram-disk.md). The GPU→RAM→disk hierarchy, as it actually behaves in b12383:

- **RAM tier** (`--cache-ram` + `--cache-idle-slots`): saves idle prompt states (16K≈1.35 GB, 87K≈2.0 GB, 171K≈3.8 GB, 182K≈4.1 GB incl. MTP draft state; LRU-bounded), but **restores only the ≤~150 MiB checkpoint class in 2-3 s; full-state loads of 1.8-3.3 GB fail** (`failed to restore kv cache`) under both 4096 and 8192 MiB budgets. No cross-session contamination observed.
- **Disk tier** (`--slot-save-path`, `POST /slots/{id}?action=save|restore|erase`): works end to end — 16K (1.12 GB, 0.9 s restore) and 128K (4.19 GB, 4.5-11 s) snapshots round-trip, **survive a full server restart, and survive a 27B→35B→27B model switch** on the same cards. One follow-up on a restored 171K state wedged the slot for 8 h with no output (MTP spec-decode generation hang; intermittent; log preserved) — the failure mode to keep in mind before automating this.
- Practical consequence: long-conversation persistence on this build is a **disk-snapshot story, not a RAM story**; the RAM cache is a small-prefix accelerator that is on by default (8192 MiB) and costs nothing to leave on.

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
- **2026-11-15/16:** max-context campaign (windows 19:45-21:07 and 21:28-00:15; production restored + live-verified after each). P1/P2/P3 measured and documented above; **P1 adopted as the production default on 2026-11-16** (owner decision), P2 canaried the same day and reverted after the concurrent-stream regression (see the production-config section).
- **2026-11-16/17:** context-hierarchy mission (GPU→RAM→disk persistence + hub-observability scope): RAM-tier and disk-tier behavior of b12383 measured end to end (above); the disk tier works, the RAM tier restores only the small class, one restored-state generation hang documented. Production restored to P1 and live-verified at run close. The hub-observability lane (metrics/registry/alerts for the hierarchy) is the remaining piece; no production-facing change from this run beyond the P1/P2 history above.
