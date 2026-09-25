# ISTA GSQ-RCO Flash-Next on a single 3060: Q2_0 has no x86 SIMD kernel, IQ2_XS halves the gap

**Date:** 2026-09-25
**Box:** backup/inference box — Intel i3-9100 (4C/4T, DDR4-2400 2-ch ≈ 38 GB/s), single RTX 3060 12 GB, 40 GB host RAM (LXC), NVMe
**Models tested:** ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO GGUFs — **Q2_0** (66.4 GB, 2-shard) and **IQ2_XS** (68 GB, 2-shard), both from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`
**Baseline:** the existing 2026-09-20 UD-Q2_K_XL deployment (14.3–14.9 t/s, 64-slot cache) on the same box

## Summary

The ISTA GSQ-RCO quantizations use per-tensor Gumbel-Softmax + Riemannian Constrained Optimization to pick the optimal quant type per tensor under a size budget. On x86, this backfires for the Q2_0 variant: **all three routed-expert projections (gate, up, down) are Q2_0**, and `GGML_TYPE_Q2_0` (type 42, PR #24448) has **no x86 SIMD kernel** — only a generic scalar implementation and an ARM NEON variant. The CPU floor collapses from 10.7 t/s (Q2_K) to 2.4 t/s (Q2_0 scalar), a 4.5× gap that no amount of MoE caching can close on a 12 GB card.

The IQ2_XS variant is substantially better: only `ffn_down_exps` (the 640-row output projection, locked to Q2_0 by its geometry) remains Q2_0. The gate and up projections (~2/3 of expert FLOPs) are **IQ4_XS** with full AVX2/AVX-512 kernels. This lifts the CPU floor to 5.3 t/s and the best cached config to **7.1–7.3 t/s** — still 2× behind the old UD-Q2_K_XL, but a 1.5× improvement over the Q2_0 variant.

**Verdict for this box:** the old UD-Q2_K_XL remains the best single-3060 model (14.3–14.9 t/s). The ISTA models belong on a machine with enough VRAM for GPU residency (the dual-3090: 37.6–40 GB transformer shard < 48 GB aggregate VRAM), where the x86 kernel gap becomes irrelevant.

## The Q2_0 x86 kernel gap

`GGML_TYPE_Q2_0` was added in llama.cpp PR #24448 with only the generic (C) `ggml_vec_dot_q2_0_q8_0` backend. In the codacus fork (`27c54b4b`, base `b10818`), `arch-fallback.h` maps `ggml_vec_dot_q2_0_q8_0_generic` → `ggml_vec_dot_q2_0_q8_0` for all x86 targets (SSE4, AVX2, AVX-512). The only architecture-specific Q2_0 kernel is in `arch/arm/quants.c` (NEON).

By contrast, `Q2_K` (type 8) has full AVX2/AVX-512 implementations in `arch/x86/quants.c` with maddubs-based dot products. The key structural difference: Q2_K's 2-bit packing uses a **stride-32** layout (each 2-bit set maps to a contiguous 32-value group of Q8_K), enabling a single wide maddubs per set. Q2_0 uses a **stride-4** layout (byte *b* → activation indices `[4b, 4b+1, 4b+2, 4b+3]`), which requires shuffling the activation data before the multiply.

### SSE2/SSSE3 kernel written during this session

A partial SSE kernel was implemented (appended to `arch/x86/quants.c` in the in-container codacus build): deinterleave Q8_0 with `pshufb`, extract 4 sets of 2-bit values via AND/SHR, 4× `maddubs` per 16-element group, subtract `sum(Q8_0)` for the unsigned→signed correction. Result: **1.5× over scalar** (2.4 → 3.3–3.8 t/s zero-cache). Not enough to close the gap to Q2_K, but a real improvement over the scalar-only path. Not yet submitted upstream.

## Tensor type distributions (shard 1, transformer)

The RCO allocation files (published in the HF repo under `tensor-allocation/`) show per-tensor types:

| Type | Q2_0 model (1235 tensors) | IQ2_XS model | x86 SIMD? |
|---|---:|---:|---|
| **Q2_0** | **205** (all gate/up/down experts + a few attn) | **54** (ffn_down_exps only) | **No** |
| IQ4_XS | 57 | **182** (gate + up experts) | Yes |
| Q3_K | 92 | 0 | Yes |
| IQ4_NL | 8 (PLE) | 37 (PLE) | Yes |
| Q4_K / Q5_K / Q6_K / Q4_0 / Q5_0 / Q8_0 | 94 | 79 | Yes |
| BF16 / F32 / F16 | 781 | 782 | Yes (FMA) |

In the Q2_0 model, **all three** expert projections are Q2_0 across all 40 MoE layers. In the IQ2_XS model, only `ffn_down_exps` remains Q2_0 (its 640-row geometry is incompatible with the 256-element-block K/I formats); gate and up are IQ4_XS.

## Results

All measurements: 32K context (unless noted), 256-token generations, 4 reps, warm average.
`-t 4`, `-fa on`, `--lazy-mode on`, `--no-sched-async-cpu` (unless noted).
codacus fork `27c54b4b` (build 10818), CUDA 13.1, `GGML_CUDA_FA=ON`.

### CPU floor (zero MoE cache, all experts on CPU)

| Model | tok/s (warm) | VRAM MiB | GPU W |
|---|---:|---:|---:|
| UD-Q2_K_XL (2026-09-20 baseline) | 10.7 | 4,677 | 42 |
| ISTA Q2_0 (scalar, stock) | 2.4–2.5 | 4,677 | 42 |
| ISTA Q2_0 (SSE, our kernel) | 3.3–3.8 | 4,677 | 42 |
| **ISTA IQ2_XS** | **5.3** | 4,707 | 47 |

### With 64-slot MoE hot cache

MoE profiles: Q2_0 traces (82,001 rows, 3 prompt families × 2 × 256 tok) and IQ2_XS traces (39,779 rows, 3 × 256 tok), generated with `llama-moe-trace`. Routing diverges between the two models from token 3 onwards, but the Q2_0 profile performs equivalently for IQ2_XS (7.3 vs 7.1).

| Model | Profile | tok/s (warm) | VRAM MiB | GPU W |
|---|---|---:|---:|---:|
| UD-Q2_K_XL (2026-09-20) | own (307K rows) | 14.3–14.9 | ~8,000 | 48 |
| ISTA Q2_0 (SSE) | Q2_0 (82K) | 4.4–5.3 | 8,747 | 44 |
| ISTA IQ2_XS | Q2_0 (82K) | 7.3 | 8,947 | 51 |
| ISTA IQ2_XS | IQ2_XS (39K) | 7.1 | 8,947 | 64 |
| ISTA IQ2_XS | IQ2_XS (39K), 80 slots | 6.4 | 10,003 | 50 |

### GPU placement sweep (Q2_0 model, no cache)

`--n-cpu-moe N` moves the first N layers' MoE experts to CPU; the rest go to GPU.

| n-cpu-moe | MoE on GPU | Result |
|---:|---:|---|
| 99 (all CPU) | 0 | 2.4–5.3 t/s (scalar/SSE) |
| 40 | 8 | 4.0–4.1 t/s, 10.0 GB VRAM |
| 36 / 32 / 30 | 12+ | **CUDA OOM** (12 GB ceiling) |

Each MoE layer costs ~667 MiB on GPU. The 12 GB card fits at most ~8 MoE layers (20% of 40). Even with 8 on GPU, the remaining 32 CPU layers dominate.

### IQ2_XS context sweep (64 slots, IQ2_XS profile)

| Context | tok/s (warm) | VRAM MiB | GPU W | 12 GB headroom |
|---:|---:|---:|---:|---:|
| 32K | 7.1–7.3 | 8,947 | 51–64 | 3.3 GB |
| 64K | 7.5–8.3 | 9,573 | 51–58 | 2.7 GB |
| 96K | 6.4–7.4 | 10,181 | 47–55 | 2.1 GB |

No degradation with context length — the Flash-Next architecture's mild KV growth (12/48 layers with sparse attention, Gated-DeltaNet state largely fixed) keeps VRAM growth linear and small.

### A/B variants (IQ2_XS, 64 slots, 32K)

| Variant | tok/s | VRAM MiB | Note |
|---|---:|---:|---|
| q8_0 KV + `--no-sched-async-cpu` (default) | 7.1–7.3 | 8,947 | best |
| f16 KV, no `--no-sched-async-cpu` | 6.9 | 9,397 | no improvement, more VRAM |
| hybrid: 8 GPU MoE + 32 cache slots | — | OOM | doesn't fit in 12 GB |

### Quality spot-check

IQ2_XS: "Capitals: France, Japan, Brazil" → "Paris, Tokyo, Brasília" ✓
Q2_0: "The capital of France is Paris." ✓
Both models produce correct, coherent output at all tested speeds.

## Why the numbers are still 2× behind

Three compounding factors:

1. **Q2_0 ffn_down_exps (1/3 of expert FLOPs).** Even in the IQ2_XS model, the 640-row output projection per expert is stuck at Q2_0 (no x86 SIMD). This is the single largest remaining CPU bottleneck. Each of the 512 experts × 40 layers × 10 active = 204,800 down-projection GEMVs per token sequence, all on the scalar/SSE path.

2. **IQ4_XS is not Q2_K.** Despite having full AVX2/AVX-512 kernels, IQ4_XS uses a lookup-table format (scale per 32-element sub-block, sign table, lookup grid) that is computationally more elaborate than Q2_K's simpler min+scale format. On the i3-9100's memory-bandwidth-limited DDR4, the higher per-element memory footprint of IQ4_XS (2.3125 bpw vs Q2_K's ~2.56 bpw but with more metadata) partially offsets the SIMD advantage.

3. **The 12 GB VRAM ceiling.** At most 8 of 40 MoE layers fit on GPU (~667 MiB each). The remaining 32 layers run entirely on CPU. The old UD model's Q2_K experts ran at 10.7 t/s on CPU; the IQ2_XS mix runs at 5.3. No amount of hot-caching (64–96 slots) covers 32 × 512 = 16,384 expert matrices.

## What would change the picture

- **A proper AVX2 Q2_0 kernel** (2 blocks per YMM, ~3× over the current SSE): would lift the Q2_0 model from ~5 to ~8 t/s and the IQ2_XS model from ~7 to ~9 t/s. Still not matching Q2_K, but closer.
- **The dual-3090 (48 GB aggregate VRAM):** the 37.6 GB Q2_0 transformer shard (or 40 GB IQ2_XS) fits entirely on GPU with `-ngl 99`. No experts on CPU → the x86 kernel gap is irrelevant. Q2_0 has CUDA MMQ/MMVQ kernels (including Ampere-specific configurations) in this llama.cpp build. This is the intended deployment target.
- **A surgical hybrid GGUF:** re-encode only the Q2_0 ffn_down_exps tensors (54 in IQ2_XS) to a CPU-friendly format (e.g., Q2_K if the 640-row geometry permits a non-256-block variant), while keeping the rest of ISTA's per-tensor allocation. Would preserve the 68 GB footprint while eliminating the last Q2_0 dependency.

## Artifacts

- **Model (Q2_0):** 66.4 GB, 2-shard GGUF + mmproj (866 MB). On the main GPU server and the backup box (IQ2_XS variant on the backup; Q2_0 on the main GPU server).
- **Model (IQ2_XS):** 68 GB, 2-shard GGUF + mmproj. On both machines.
- **SSE Q2_0 kernel:** appended to `ggml-cpu/arch/x86/quants.c` in the in-container codacus build. Not upstreamed.
- **MoE profiles:** Q2_0 (82K rows) and IQ2_XS (39K rows) trace CSVs, stored alongside the models.
- **Codacus fork:** `27c54b4b` (base `b10818`), built on both the backup box (CT) and the main GPU server.
- **Old UD-Q2_K_XL:** still available on the main GPU server (78.9 GB, 3-shard). Not currently deployed on the backup box (space was taken by the IQ2_XS model).

## Related reports

- [2026-09-20: Flash-Next single-3060 MoE cache baseline (UD-Q2_K_XL)](2026-09-20-flash-next-single-3060-moe-cache-backup.md)
- [2026-09-13: Flash-Next single-3090 codacus cache](2026-09-13-flash-next-single3090-codacus-cache.md)
- [2026-09-18: ISTA 3-bit 27B single-3060](2026-09-18-ista-3bit-27b-single-3060.md)
- [2026-09-02: Flash-Next three-GPU campaign](2026-09-02-qwen4exp-flash-next-three-gpu-campaign.md)
