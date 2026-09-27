# ISTA 2-bit (GSQ) Qwen3.6-35B-A3B on Kaby Lake: the Q2_0 gate and the 6 GB placement floor

**Date:** 2026-09-27
**Box:** HP OMEN 15 — i5-7300HQ (4 physical cores, **no Hyper-Threading on this unit**), GTX 1060 6 GB (Pascal), 24 GB DDR4 — [hardware profile](../docs/hardware/pascal-laptop.md)
**Model:** `Qwen3.6-35B-A3B-GSQ-hybrid.gguf` — ISTA-DASLab's 2-bit conversion (GGUF by chtisgit): 11.38 GiB, **routed experts Q2_0** (120 tensors), attention / linear-attention / shared-expert / LM-head **Q8_0** (251 tensors), 61 BF16, 301 F32; MTP head omitted; SHA-256 verified against the published values. Companion to [2026-09-25-ista-gsq-rcos-single-3060-q2-0-no-x86-simd](2026-09-25-ista-gsq-rcos-single-3060-q2-0-no-x86-simd.md) and the [Pascal 128K report](2026-09-27-pascal-1060-2026-llama-cpp-35b-128k.md).

## Question

The 09-25 3060 report found the 2-bit model's Q2_0 expert path running at **2.4 t/s all-CPU on an i3-9100** (no VNNI), i.e. scalar-ish despite the CPU having SSE/AVX2 — and concluded "2-bit belongs on machines with spare VRAM." Before discarding the 2-bit branch on this 4-core/6 GB laptop, two things had to be checked: (1) whether the 2026 llama.cpp base — not the 09-25 codacus-era build — has an x86 SIMD path for Q2_0 at all, and (2) what the model's **non-expert VRAM floor** actually is on a 6 GB card.

## Finding 1: the 09-25 "no x86 SIMD" result is stale for the 2026 base

The 2026-09-26 base (what this node builds) contains a **dedicated AVX2 Q2_0 kernel** (`ggml/src/ggml-cpu/arch/x86/quants.c`: `q2_0_madubs` under `#if defined __AVX2__`, with the `__F16C__`/FMA-3 variants). With the standard `-march=native` build, Kaby Lake's AVX2 path is active in the binary. So the 09-25 finding (Q2_0 ≈ 2.4 t/s ≈ 0.3× the Q2_K 10.7 on that i3-9100) predates this kernel and does not bound today's behaviour.

**Measured here** (all experts on CPU, `-t 4`, 128-token completion on the 2026 fork build): **4.87 t/s** — compared to the 4-bit all-CPU class at ~5–6 t/s on the same cores. The AVX2 kernel closes most of the 09-25 gap: 2-bit is now roughly **0.8×** the 4-bit all-CPU rate, not 0.3×. The 09-25 conclusion (2-bit CPU-inferior, fine where VRAM-resident) survives in direction but not in magnitude.

## Finding 2: the 6 GB placement floor is 2.34 GB — the branch is viable

Capacity probe (all 40 MoE layers' routed experts forced to CPU, 4K ctx, one slot, minimal batch):

| Buffer | Size |
|---|---:|
| CUDA0 **model buffer** (Q8_0 non-expert: attention, linear-attention, shared experts, embedding/LM head) | **2,031.3 MiB** |
| CUDA_Host model buffer (all Q2_0 routed experts) | 9,610.0 MiB |
| KV buffer (4096 cells, the 10 full-attention layers, q8_0 K/V) | 42.5 MiB |
| Recurrent state (linear-attention layers, on GPU) | 62.8 MiB |
| Compute/workspace | 91.4 MiB |
| **Total VRAM used** | **2,335 MiB** |

Two notes:

- The 2,335 MiB total with **zero** expert weights on the GPU leaves **~3.6 GB of headroom** on the 6 GB card for partial Q2_0 expert residency — each expert layer's three Q2_0 tensors are ~216 MiB, so a substantial expert subset can be GPU-resident. (An earlier probe with 20 of 40 expert layers GPU-eligible OOM'd at a 6.35 GiB *combined* allocation — that number is the non-expert core plus ~4.3 GB of expert weights, not a non-expert-only figure; the all-CPU-experts probe above is the correct floor.)
- The **linear-attention recurrent state sits on the GPU** (62.8 MiB at 4K ctx) and the 10 full-attention layers' KV will scale to ~1.4 GB at 128K — both need re-checking in a 128K probe; they compress the expert headroom at long context.

## Verdict

- **Placement: viable.** The 2-bit model fits the 6 GB card with room for expert residency — ISTA's choice to keep the non-expert core at 8-bit costs ~2.0 GB of VRAM, which this card can afford.
- **CPU speed: much less of a problem than 09-25 suggested.** The 2026 base's AVX2 Q2_0 kernel puts 2-bit at ~0.8× of 4-bit all-CPU on these cores, not ~0.3×.
- **Open question → Phase 2:** is there an expert-residency point (ncmoe 40 → 0, with the 12-slot hot-expert cache) where the 2-bit model's smaller CPU-resident mass and lower per-token DRAM traffic **crossover** the 4-bit baseline (12.05 t/s, t1–t4 flat)? The 2-bit CPU buffer is 9.6 GB vs 4-bit's ~15 GB; per active token the streamed mass is ~2.6× smaller, which on this DRAM-bound machine is exactly the lever that matters. Sweep: fixed 4K ctx, decode t/s + VRAM + GPU util + PCIe RX per residency step.

## Cross-microarchitecture note

| CPU (4-core class) | Q2_0 35B all-CPU | Engine era |
|---|---:|---|
| i3-9100 (Coffee Lake, no VNNI) | 2.4 t/s (0.3× of Q2_K) | 09-25 codacus-era build — scalar Q2_0 path |
| i5-7300HQ (Kaby Lake, AVX2) | **4.87 t/s (~0.8× of 4-bit)** | 2026 base — dedicated AVX2 Q2_0 kernel |

The same model, same quant, four months apart: the gap to the 4-bit class shrank from ~3× to ~1.25× purely from kernel coverage. 2-bit MoE is approaching CPU-viability on cheap 4-core hardware.

## Appendix

Raw placement log: `phase1/ista-floor.log` (private repo); completion JSON: `phase1/resp-ista.json`.
