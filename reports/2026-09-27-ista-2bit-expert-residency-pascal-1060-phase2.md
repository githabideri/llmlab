# ISTA 2-bit expert-residency campaign on the Pascal laptop — Phase 2

**Date:** 2026-09-27
**Box:** 2017 HP OMEN 15 — i5-7300HQ (4C, SMT firmware-disabled), GTX 1060 6 GB Max-Q (Pascal cc 6.1), 24 GB DDR4 (asymmetric 16+8, effectively single-channel ~19–20 GB/s) — [hardware profile](../docs/hardware/pascal-laptop.md)
**Models:** Qwen3.6-35B-A3B (MoE, ~3B active) — `UD-IQ4_XS` (17 GB, the current production quant) vs the ISTA `GSQ-hybrid` 2-bit (11.38 GB: routed experts Q2_0, non-expert core Q8_0, MTP heads omitted)
**Engine:** the MoE-expert-cache fork of the 2026 llama.cpp (Pascal build; 580 LTSB + CUDA 12.9), `--fit off --load-mode none -t 4 -b 2048 -ub 512 -np 1`, q8_0 K/V. Follow-on to the [ISTA Q2_0 gate report](2026-09-27-ista-gsq-2bit-35b-q2-0-kaby-lake.md) and the [Pascal 128K deployment report](2026-09-27-pascal-1060-2026-llama-cpp-35b-128k.md).

## Goal

Does the 2-bit model's smaller CPU-resident expert mass beat the Q2_0 kernel's ~0.8× per-byte deficit on a DRAM-bandwidth-bound machine — i.e. does the ISTA 2-bit model beat the 4-bit production config, at 4K and at 128K, with what quality left over?

## Setup

Three measurement stages in one stop-window per stage (production router stopped, campaign servers on a test port, production restored after):

- **Controls** — fresh IQ4_XS runs with the exact production config (12-slot expert cache), 4K and 128K, engine-reported timings.
- **Stage A** — ISTA, expert cache *off*, `ncmoe N` swept 40→0 (4K ctx): static expert residency.
- **Stage B** — ISTA at the best Stage-A points with 12/24-slot expert cache.
- **Stage C** — the best Stage-B configs at 128K + a full host-memory envelope.
- **Quality** — 7 fixtures (arithmetic/reasoning, O(n) code, top-3 lakes by volume, long-context buried fact with a distractor date, exact-format instruction following, strict JSON, a low-quant-sensitive general-knowledge pair), identical sampling on both models (0.7 / 0.8 / top-k 20), in two regimes: thinking disabled (1024 max tokens, supplementary) and the production regime — thinking on, 4096 max tokens (primary).

## Commands

```sh
# static residency (Stage A) — note: NO cache flags
llama-server -m <gguf> --fit off --load-mode none -t 4 -c 4096 -b 2048 -ub 512 -np 1 \
  -ctk q8_0 -ctv q8_0 -ngl all -ncmoe 24 --host 127.0.0.1 --port 8099

# with hot cache (Stages B/C)
llama-server -m <gguf> --fit off --load-mode none -t 4 -c 128000 -b 2048 -ub 512 -np 1 \
  -ctk q8_0 -ctv q8_0 -ngl all -ncmoe 28 \
  --moe-expert-cache-size 24 --moe-early-router --decode-overlap \
  --backend-sampling --phase-aware-workspace --host 127.0.0.1 --port 8099
```

Decode: non-streaming `/v1/completions`, 256 tokens, fixed prompt, 2–3 warm reps, engine-reported `timings.predicted_ms/predicted_n` (no wall-clock noise). Memory envelope from `/proc/<pid>/status` + `smaps_rollup` + `vmstat` + `free` at load and after decode; `nvidia-smi` for VRAM/util/power.

## Observations

### The no-cache server path OOMs the 4-bit model

`--fit off -ngl all -ncmoe 20` *without* the cache feature set makes the server's whole-layer offloader place ~20 expert layers on the GPU — a 9.3 GB allocation that cannot fit in 6 GB. The hot-expert cache's budget-driven placement is therefore not a performance option on this card; **it is what makes the 4-bit production config load at all** (the 4053 MiB no-cache figure from the `llama-bench` tool came from bench's different offloader semantics, not the server's).

### Stage A — static residency: monotonic, cliff at 16 GPU layers

| ncmoe | GPU expert layers | VRAM (MiB) | decode (t/s) |
|---:|---:|---:|---:|
| 40 | 0 | 2,423 | 5.21 |
| 36 | 4 | 3,285 | 5.55 |
| 32 | 8 | 4,147 | 6.19 |
| 28 | 12 | 5,011 | 6.81 |
| **24** | **16** | **5,873** | **7.67** |
| 20 → 0 | 20+ | — | OOM at load (~6.9 GB requested) |

Decode rises monotonically and *accelerates* (+0.34 / +0.64 / +0.62 / +0.86 t/s per 4 layers): the per-token DRAM mass streamed from the CPU side shrinks faster than the kernel-rate deficit grows. Even the best static point stays 40% behind the 4-bit config — residency alone is not the lever.

### Stage B — the hot cache changes the regime (and ignores `ncmoe`)

The server logs it explicitly: **`--moe-expert-cache-size` routes expert tensors through the GPU LRU cache regardless of `--n-cpu-moe`** — with the cache on, the `ncmoe` value is inert, which is why every cache-12 point clusters:

| cache | decode (t/s) | VRAM (MiB) |
|---:|---:|---:|
| 12 | 16.77–16.86 (all ncmoe values) | 3,015 |
| **24** | **18.57–18.68** | **3,485** |

The 12-slot cache lifts the same model from 7.67 to 16.77 t/s (+119%); 24 slots add ~11% for 470 MiB. Repetition spread ≤0.1 t/s. **Result: 18.68 t/s at 4K vs 12.66 for the 4-bit production config, on 3.49 GB vs 4.67 GB VRAM.**

### Stage C — 128K

| cache | decode (t/s) | VRAM (MiB) | RSS (MiB) | Δ vs 4K (MiB) |
|---:|---:|---:|---:|---:|
| 12 | 16.87 | 4,171 | 10,584 | 1,156 |
| 24 | 18.65 | 4,641 | 10,584 | 1,156 |

The 4K→128K delta is identical for both points — it is the q8_0 KV of the 10 full-attention layers; **the linear-attention recurrent state is fixed-size (context-independent)**, so 128K costs no extra state. Both fit with 1.4–1.7 GB headroom; slot-level logs confirm 18.58 t/s steady decode with CUDA-graph reuse.

### Host-memory envelope

RSS ≈ 10.6 GB (≈10.4 GB private dirty — the pre-allocated CPU expert buffer), **12 GiB still MemAvailable, swap untouched** (36 MB used, `pswpout` unchanged across all snapshots, including model swaps). **16 GB (8+8) projection:** ~12.5–13.5 GB steady demand → fits with ~2–3 GB margin, workable but no room for a second resident model; 32 GB (16+16) is comfortable and — per the [09-27 report](2026-09-27-pascal-1060-2026-llama-cpp-35b-128k.md) — dual-channel RAM is the single biggest remaining *decode* lever, which the 2-bit model (2.6× less streamed byte count) benefits from proportionally more.

### Energy

GPU 27–30 W during the 2-bit decode cells (vs 29–30 W for the 4-bit at 4K and ~30 W at 128K); CPU package 13.3 W (RAPL, measured on the 4-bit — the 2-bit streams less, so the CPU side is equal-or-lower). Wall total in the same ~55–70 W band as the 4-bit; the 2-bit's win is tokens-per-second-per-watt, not watts.

## Metrics — the comparison that matters

| | 4-bit production (IQ4_XS, c12) | 2-bit (GSQ-hybrid, c24) | 2-bit (c12) |
|---|---:|---:|---:|
| decode @ 4K (t/s) | 12.66 | **18.68** | 16.86 |
| decode @ 128K (t/s) | 12.62 | **18.65** | 16.87 |
| VRAM @ 4K / 128K (MiB) | 3,595 / 4,881 | 3,485 / 4,641 | 3,015 / 4,171 |
| CPU RSS (GiB) | 16.2 | 10.6 | 10.6 |
| swap activity | 0 | 0 | 0 |

## Quality battery

**Primary (thinking on, 4096 max tokens):** 5 of 7 fixtures pass identically (arithmetic, code, exact-format, long-context, JSON — all correct on both models, including the distractor-date long-context and the strict JSON). The two divergences, both on the 2-bit side:

- *Top-3 lakes by volume:* the 2-bit model's thinking consumed the **entire 4096-token budget** (12.5k characters of reasoning, `finish=length`, zero answer); the 4-bit model's thinking on the same fixture is the same length (12.3k) but still leaves a 605-char answer (correct top-2, wrong third place).
- *General-knowledge pair:* the 2-bit answers the 1779 photosynthesis-experiment scientist as "**Jan Ingenhouz**" — the name is corrupted (missing final *s*); the 4-bit spells it correctly. The ice-density explanation is correct on both.

**Supplementary (thinking off, 1024 tokens):** 5/7 clean on both; the lakes question fails in *different* ways per quant (2-bit: right ranking, wrong volumes, self-corrects mid-answer; 4-bit: drops Superior for Malawi). No systematic degradation at the short-answer depth — but the primary regime shows the 2-bit's fragility concentrates exactly where it should: rare proper nouns and stopping-behavior under open-ended factual questions.

## Conclusion

1. **4K and 128K:** the 2-bit ISTA model beats the 4-bit production config by ~48% and ~47% respectively, with *less* VRAM and ~6 GB less host RAM. On a DRAM-bandwidth-bound box, bytes moved per token is the currency, and 2.6× fewer bytes beats a 0.8× kernel.
2. **The hot-expert cache is the enabler, not a tuning knob** — on this card the no-cache server path doesn't even load the 4-bit model, and with the cache on `ncmoe` is ignored. The only real cache question is slot count: 24 slots = +11% for 470 MiB.
3. **Quality:** no degradation on objective/format fixtures; two real 2-bit weaknesses surfaced (budget-eating runaway thinking on one open factual fixture; a corrupted proper noun). Whether those are 2-bit-specific or just this model's long-tail behavior needs a bigger battery — a small-battery result, not a verdict.
4. **Production candidate:** the 2-bit n28/c24 config (18.65 t/s @128K, 4.64 GB) is the leading candidate for this laptop's serving slot; the 4-bit config stays the fallback until a larger quality battery and the RAM upgrade (dual-channel, which multiplies the 2-bit's advantage) have both landed.
5. **Highest-value next experiments:** the 16+16 RAM swap re-measured on the 2-bit (the single biggest lever left); a 50–100 item quality battery with the 2-bit's two failure classes as the probe set; Q3-class quant upgrades for the CPU-resident experts.

*Raw cells, per-request JSON (engine `timings` + reasoning), placement and memory logs live in the private repo (they carry machine-internal identifiers and are not mirrored here).*
