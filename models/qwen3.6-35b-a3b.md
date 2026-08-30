# Qwen3.6-35B-A3B

**Base model:** Qwen3.6-35B-A3B — 35B MoE, ~3B active (8 routed + 1 shared expert), DeltaNet linear attention (10 of 40 layers full-attention)
**Current quant:** `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (unsloth dynamic quant, **MTP variant**), dir `qwen3.6-35b-a3b-mtp/`
**Vision projector:** `mmproj-F16.gguf` (CPU-resident)

## Current deployment (production)

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

## Changelog

- **2026-02-25/26:** 3.5 initial evaluation; loop-failure finding.
- **2026-03-03:** 3.5 24 GB text+tools+vision viability (tuned profile); reasoning-loop fix (dropped `--reasoning-format deepseek`).
- **2026-03-04:** 3.5 production performance benchmarked on 3×3060 (36 GB).
- **2026-03-14/15:** 3.5 vLLM PP=3 profile + observations.
- **2026-08 (this update):** card renamed to 3.6; current deployment is Qwen3.6-35B-A3B UD-IQ4_XS (256K, dual 3060, 2 slots, mainline build). 3.5 Q4_K_M work moved to Historical.
- **2026-08-30:** IQ2_XXS squeeze tested on **one** 12 GB 3060 (temporarily taking the card out of the production 2-GPU unit): fully resident — 10.03 GiB GGUF (`Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`, SHA256 `2e8f5f70…bef`) → 11,307–11,320 MiB measured VRAM, ~0.02 GB/s PCIe in decode vs 5.9 GB/s for its CPU-MoE twin — 43–81 t/s decode across 2K–64K, 99 t/s with the MTP-variant build (11.0 GiB, `627e2b05…03`) at the VRAM cliff (~16K budget only). 2-bit quality not yet evaluated — candidate, not deployment. See the [report](../reports/2026-08-30-dual-3060-35b-squeeze-27b-node.md).
