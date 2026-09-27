# Qwen3.6-35B-A3B

**Base model:** Qwen3.6-35B-A3B — 35B MoE, ~3B active (8 routed + 1 shared expert), DeltaNet linear attention (10 of 40 layers full-attention)
**Current quant:** `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (unsloth dynamic quant, **MTP variant**), dir `qwen3.6-35b-a3b-mtp/`
**Vision projector:** `mmproj-F16.gguf` (CPU-resident)

## Current deployment (production)

> **Interim (since 2026-09-08):** the dual-3060 home below was dismantled — the 3060 pair left the primary box for a second RTX 3090 (dual-3090 vLLM TP2 for Qwen3.8-27B). The 35B now rides on the **secondary GPU server (i5-7400, single RTX 3060 LHR)** as the `Qwen3.6-35B-A3B-MTP` variant (Q4_K_XL + mmproj, 128K ctx, MTP, vision) — measured **~28–39 t/s tgen / ~630–780 t/s pp at 128K** ([2026-06-30 report](../reports/2026-06-30-qwen3.6-35b-a3b-mtp-single-3060.md) + the 2026-08-28 `-ub 2048` bump) — a shared card with document-AI work, so expect hub-mediated load/unload around those jobs — with the **single-3060 backup box** as the usual window fallback. The dual-3060 config below is preserved as the historical home; the dual-3090 campaign (09-10/11) confirmed the 27B on the 3090 pair, and the 35B stays interim on the secondary box for now.

### Pascal laptop variant (in production since 2026-09-27)

The 35B also rides on a **2017 HP OMEN 15** (i5-7300HQ, 4C/4T without Hyper-Threading — 4 worker threads; GTX 1060 6 GB Max-Q, 24 GB) as a headless, text-only, hub-managed 128K endpoint — the lab's Pascal test bed ([hardware profile](../docs/hardware/pascal-laptop.md)). Same Qwen3.6-35B-A3B base model/family as the 3060's, but a **different GGUF quant per machine**: the 3060 runs Q4_K_XL (+vision, MTP variant), the laptop runs **UD-IQ4_XS, text-only, no MTP** (MTP measured a net loss on this card: 12.7 → 11.6 t/s at 64K).

| Param | Value |
|-------|-------|
| Quant | UD-IQ4_XS (17 GB, no mmproj) |
| Hardware | 1× GTX 1060 6 GB Max-Q (Pascal 6.1, 1,280 cores, 60 W TGP) + i5-7300HQ (4C/4T, no HT) |
| Context | 128K, q8_0 K/V, single slot, 12 GB host-RAM-resident (24 GB machine) |
| Placement | `--load-mode none -ngl all -ncmoe 20` — 20 of 40 expert layers on GPU, rest on CPU |
| Engine | llama.cpp 2026 moe-cache fork: **12-slot expert cache** — the only configuration that makes 128K fast on a 6 GB card (frees ~1 GB of expert weights for the KV buffer); `--decode-overlap --backend-sampling --phase-aware-workspace --moe-early-router`, `-t 4` (one worker per physical core) |
| Serving | 2026 `llama-server` in **router mode** (`--models-preset` INI + `load-on-startup`) → llm-hub `llama-router` card: model row with live rates, one-click load/unload (202-accepted; ~5 min cold load on 4 cores) |
| Measured | **~11.8 t/s decode at 128K** (4.67 GB VRAM) · **~167–186 t/s pp** (512–2048) · decode energy: GPU 30–50 W (38.4 W avg over a 3,000-token run) + CPU package 13.3 W (RAPL package domain) → ~55–70 W at the wall (platform ~10–15 W not directly metered) |

Why it matters: the first 35B-class MoE served on Pascal in the fleet — 580 LTSB driver (last Pascal line) + CUDA 12.9 (last Pascal toolkit) + the 2026 engine, with the `cublasCreate_v2 = 3` crash forensics and the exact flag combination that resolves it ([2026-09-27 report](../reports/2026-09-27-pascal-1060-2026-llama-cpp-35b-128k.md)). Decode sits at the box's ~19–20 GB/s effective DRAM wall (asymmetric 16+8 RAM, single-channel-equivalent — see the profile); a 16+16 upgrade is the standing next step.

**2-bit candidate (evaluated 2026-09-27, Phase 2):** the ISTA GSQ-hybrid 2-bit quant beats this production config on the same card — **18.7 t/s vs 12.7 at both 4K and 128K** (19.5 @ 32 cache slots), 3.5/4.6 GB vs 4.7/4.9 GB VRAM, 10.6 vs 16.2 GB RSS — with two quality soft-spots (budget-eating runaway thinking on one open factual fixture; one corrupted proper noun). Cutover is gated on the larger adversarial quality battery; it is already viable on the current asymmetric RAM (the dual-channel swap is a separate performance experiment). Details in the [Phase-2 report](../reports/2026-09-27-ista-2bit-expert-residency-pascal-1060-phase2.md).

### Historic: dual-3060 home (dismantled 2026-09-08 — preserved as the historical config)

The MoE half of the primary GPU server — dual RTX 3060 on a mainline llama.cpp build (`/opt/llama.cpp-mainline`). Verified against the live unit `llama-server-qwen3.6-vision.service` (:8081), 2026-08-27.

| Param | Value |
|-------|-------|
| Quant | UD-IQ4_XS (MTP variant) |
| Hardware | 2× RTX 3060 12 GB (CUDA0, CUDA1) |
| Context | 256K (`-c 262144`), 2 parallel slots |
| Split | `--split-mode tensor --tensor-split 50,50` |
| KV cache | q8_0 (K) / q4_0 (V) |
| Batch | `-b 2048 -ub 1024`, flash-attn on |
| Spec decoding | `--spec-type draft-mtp --spec-draft-n-max 3` (acceptance 0.93–0.98) |
| Vision | mmproj F16, `--no-mmproj-offload --image-max-tokens 1024` |
| Unit | `llama-server-qwen3.6-vision.service` |

```bash
/opt/llama.cpp-mainline/build/bin/llama-server \
  --device CUDA0,CUDA1 \
  -m /mnt/models/gguf/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --mmproj /mnt/models/gguf/qwen3.6-35b-a3b-mtp/mmproj-F16.gguf --no-mmproj-offload --image-max-tokens 1024 \
  --host 0.0.0.0 --port 8081 \
  -c 262144 --parallel 2 \
  --split-mode tensor --tensor-split 50,50 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --batch-size 2048 --ubatch-size 1024 --flash-attn on \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --jinja --metrics \
  --cache-prompt --cache-ram 2048
```

KV cache is small because only 10 of 40 layers are full-attention; the rest are linear attention (DeltaNet) with zero KV, so 256K × 2 slots fits comfortably. See [KV-cache sizing](../docs/kv-cache-sizing.md).

> Throughput: ~100–114 t/s decode at 256K with MTP n=3 — full measured matrix in the [2026-08-27 optimization report](../reports/2026-08-27-qwen3.6-35b-a3b-dual-3060-optimization.md). The pre-optimization numbers below are from the earlier Qwen3.5 Q4_K_M work on the same hardware and are kept for reference.

## Historical: Qwen3.5-35B-A3B (Q4_K_M, 2026-02/03)

First deployed and evaluated as **Qwen3.5-35B-A3B (Q4_K_M)** on dual 3060 (24 GB, 98K) and later on 3×3060 (36 GB, vLLM PP=3). Superseded by the 3.6 IQ4_XS build above; kept for the fitment analysis and the tool-loop finding.

- **Base model:** [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- **Quant:** `Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf` (~20 GB) · **Vision:** `mmproj-Qwen_Qwen3.5-35B-A3B-f16.gguf`
- **KV cache:** ~7.5 KiB/token (10 attention layers)

### Why Q4_K_M fit on 24 GB (dual 3060)

The fit came from runtime pressure, not a lighter quant: `--ctx-size 98304`, `--split-mode layer`, `parallel=1`, and `--no-mmproj-offload`. Observed VRAM stayed just under the cliff — GPU0 ~11.9/12 GB, GPU1 ~11.3/12 GB.

```bash
llama-server \
  --model /mnt/models/gguf/qwen3.5-35b-a3b/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --mmproj /mnt/models/gguf/qwen3.5-35b-a3b/mmproj-Qwen_Qwen3.5-35B-A3B-f16.gguf \
  --no-mmproj-offload --ctx-size 98304 --parallel 1 \
  --split-mode layer --gpu-layers 99 \
  --cache-type-k q8_0 --cache-type-v q4_0 --flash-attn on --jinja
```

### Performance (3.5 Q4_K_M, March 2026, 3×3060)

A 15-minute production session (text + web research, native thinking, 21K→64K context, no compaction): PP ~836 tok/s avg (544–1016), TG ~42.4 tok/s avg (35–50). Context degradation was clean and predictable — ~49 tok/s <25K down to ~40 tok/s >40K (−20%); thinking tokens added ~15%. Vision preprocessing scaled with resolution (~0.3 s at 128px, ~32 s at 1024px).

### Tool-loop pathology (the one that matters)

Runaway repeated tool calls were observed on this model family — on mainline Q4_K_M in February, and later attributed to the BeeLlama DFlash drafter in June. The DFlash mechanism (a small draft model proposing tokens the target then verifies probabilistically) can reinforce a wrong continuation — e.g. the same bad file path — across 20+ repeated calls with no self-correction. Dedicated write-up: [`2026-06-19 DFlash cutover report`](../reports/2026-06-19-beellama-dflash-cutover.md).

### vLLM PP=3 profile (3× RTX 3060, 2026-03)

Official GPTQ-Int4 on vLLM 0.17.1, `--pipeline-parallel-size 3`, fp8 KV: post-warmup TG ~58–62 tok/s, ~1.91 GiB KV at 131K, high-concurrency run peaked ~130 tok/s output. Vision was disabled by config (`--language-model-only`); LMCache was incompatible (upstream #36771). Reference: [`2026-03-14 vLLM PP=3 experiment`](../reports/2026-03-14-qwen3.5-35b-a3b-vllm-pp3-concurrency.md).

## References

- 3.5 preliminary (loop failures): [`2026-02-26`](../reports/2026-02-26-qwen3.5-35b-a3b-llmlab-preliminary.md)
- 3.5 24 GB vision retest: [`2026-03-03`](../reports/2026-03-03-qwen3.5-35b-a3b-24gb-vision-retest.md)
- DFlash tool-loop write-up: [`2026-06-19`](../reports/2026-06-19-beellama-dflash-cutover.md)
- 2-bit squeeze onto one 12 GB 3060 (residency proof, decode ladder): [`2026-08-30`](../reports/2026-08-30-dual-3060-35b-squeeze-27b-node.md)
- ISTA 2-bit Q2_0 on Kaby Lake + 6 GB placement floor (Pascal laptop, 2026 base has the AVX2 Q2_0 kernel): [`2026-09-27`](../reports/2026-09-27-ista-gsq-2bit-35b-q2-0-kaby-lake.md)
- ISTA 2-bit expert-residency Phase 2 on the Pascal laptop (128K + quality battery; 2-bit beats the 4-bit production config by ~47–48%): [`2026-09-27`](../reports/2026-09-27-ista-2bit-expert-residency-pascal-1060-phase2.md)

## Changelog

- **2026-09-27 (Phase 2):** ISTA GSQ-hybrid 2-bit evaluated on the Pascal laptop (follow-up campaign, production untouched): with the 24-slot hot-expert cache it decodes **18.68 t/s at 4K and 18.65 t/s at 128K vs 12.66/12.62 for the shipping 4-bit config**, using 3.49/4.64 GB vs 4.67/4.88 GB VRAM and 10.6 vs 16.2 GB RSS — the 2-bit's smaller CPU-resident expert mass wins on this DRAM-bandwidth-bound box. Findings: the no-cache server offloader path OOMs the 4-bit model (the hot cache is the 6 GB enabler, not a tuning knob), the cache ignores `--n-cpu-moe`, the linear-attention state is fixed-size (128K adds only the 10 full-attention layers' KV), zero swap, 12 GiB headroom (16 GB projected to fit). Quality: 5/7 objective fixtures identical; two 2-bit soft-spots (runaway thinking eating a 4096 budget on one open factual fixture; "Jan Ingenhouz"). **Decision: 2-bit is the production candidate; cutover gated on a larger quality battery + the RAM upgrade.** Details: [2026-09-27 Phase-2 report](../reports/2026-09-27-ista-2bit-expert-residency-pascal-1060-phase2.md).
- **2026-09-28 (Phase 2 corrections, post-review):** the published Phase-2 report was corrected on review: the expert byte ratio is **~1.5–1.6×** (measured: IQ4 experts ≈ 15.3 GB vs ~9.4 GB CPU-resident 2-bit), not "2.6×"; the 16 GB fit is *projected, untested*; the 2-bit CPU-power claim is a hypothesis; the cutover gate is **the adversarial quality battery only** (the 2-bit is already viable on the current asymmetric 24 GB; dual-channel RAM is a separate performance experiment); Stage D added (c28: 19.2–19.4 t/s / 4.9 GB, c32: 19.5 t/s / 4.9 GB @ 128K — gains flatten past 24 slots); and the CPU is the i5-7300HQ **4C/4T no-HT** (ARK: Total Threads 4, HT No; CPUID F6/0x9E/S9, uCode 0x84 — the `ht` CPUID bit is a die-level capability and does not imply 8 threads; earlier "8T-capable / SMT-firmware-disabled" wording was wrong).
- **2026-09-27:** **Pascal laptop variant added, in production** — UD-IQ4_XS text-only, 128K q8 KV, moe-cache fork (12-slot expert cache + `-ncmoe 20` + `-t 4`, no MTP) on the HP OMEN 15 (GTX 1060 6 GB Max-Q, i5-7300HQ 4C no-HT, 24 GB); ~11.8 t/s decode at 128K, 4.67 GB VRAM; 2026 llama-server in router mode → llm-hub `llama-router` card with one-click load/unload (202-accepted, ~5 min cold load). The 580 LTSB + CUDA 12.9 + 2026-llama.cpp Pascal build story, the `cublasCreate_v2 = 3` crash (config-specific: manual `-ngl` + mmap; resolved by load-mode/ncmoe), the MTP net-loss measurement, and the 12-slot expert cache as the 6 GB/128K enabler are in the [2026-09-27 report](../reports/2026-09-27-pascal-1060-2026-llama-cpp-35b-128k.md). Box facts (incl. the single-channel-equivalent RAM) in the [hardware profile](../docs/hardware/pascal-laptop.md).
- **2026-02-25/26:** 3.5 initial evaluation; loop-failure finding.
- **2026-03-03:** 3.5 24 GB text+tools+vision viability (tuned profile); reasoning-loop fix (dropped `--reasoning-format deepseek`).
- **2026-03-04:** 3.5 production performance benchmarked on 3×3060 (36 GB).
- **2026-03-14/15:** 3.5 vLLM PP=3 profile + observations.
- **2026-08 (this update):** card renamed to 3.6; current deployment is Qwen3.6-35B-A3B UD-IQ4_XS (256K, dual 3060, 2 slots, mainline build). 3.5 Q4_K_M work moved to Historical.
- **2026-08-30:** IQ2_XXS squeeze tested on **one** 12 GB 3060 (temporarily taking the card out of the production 2-GPU unit): fully resident — 10.03 GiB GGUF (`Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`, SHA256 `2e8f5f70…bef`) → 11,307–11,320 MiB measured VRAM, ~0.02 GB/s PCIe in decode vs 5.9 GB/s for its CPU-MoE twin — 43–81 t/s decode across 2K–64K, 99 t/s with the MTP-variant build (11.0 GiB, `627e2b05…03`) at the VRAM cliff (~16K budget only). 2-bit quality not yet evaluated — candidate, not deployment. See the [report](../reports/2026-08-30-dual-3060-35b-squeeze-27b-node.md).
