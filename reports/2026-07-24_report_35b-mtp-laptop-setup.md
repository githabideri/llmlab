---
title: "Running Qwen3.6-35B at 28 tok/s on a Laptop — MTP + Vulkan iGPU"
date: 2026-07-24
tags:
  - llama-cpp
  - mtp
  - qwen3.6
  - vulkan
  - laptop
  - amd-igpu
  - benchmark
type: report
status: done
---

# Running Qwen3.6-35B at 28 tok/s on a Laptop — MTP + Vulkan iGPU

## Overview

This report documents running a 35B-parameter MoE model (Qwen3.6-35B-A3B) at **28.7 tokens/second** on a consumer laptop with integrated graphics — no discrete GPU required. The key is combining three techniques: **Multi-Token Prediction (MTP)**, **APEX quantization**, and **KV cache quantization** on AMD's Vulkan iGPU backend.

**Hardware:** AMD Ryzen 7 7840U (8C/16T), Radeon 780M iGPU, 90 GB DDR5 — a mid-range 2024 laptop.

**TL;DR:** MTP (speculative decoding using auxiliary prediction heads) gives 13× speedup over baseline. APEX quantization beats Q4_K_M on MoE models. q8_0 KV cache is essential on iGPU — q4_0 kills performance via dequantization overhead.

## Why This Matters

35B models were previously impractical on laptops without discrete GPUs. Even with quantization, generation speeds of 2-4 tok/s made interactive use painful. MTP changes this by predicting multiple tokens per forward pass, dramatically reducing the number of expensive attention computations.

## Setup

### Model

- **Base:** Qwen3.6-35B-A3B (MoE, 35B total / 3B active parameters)
- **Variant:** [SC117's Native MTP Preserved APEX GGUF](https://huggingface.co/SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF)
- **Quant:** APEX I-Compact (17 GB) — MoE-aware mixed precision
- **mmproj:** F16 multimodal projector (858 MB) for vision support

### llama.cpp Build

- **Version:** b483-555881e (July 2026 master)
- **Backend:** Vulkan (`GGML_VULKAN=ON`), HIP disabled
- **Features:** Server + Web UI + MTP support
- **Key PRs:** Lower submit threshold for small AMD GPUs (PR #25240), q8_0 KV cache dequant optimization (PR #25493)

### Configuration

```ini
[model]
model = Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-I-Compact.gguf
mmproj = mmproj-Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-F16.gguf
ctx-size = 65536
n-gpu-layers = 99
cache-type-k = q8_0
cache-type-v = q8_0
flash-attn = on
spec-type = draft-mtp
spec-draft-n-max = 3
batch-size = 2048
ubatch-size = 2048
threads = 8
threads-batch = 8
```

### Key Parameters Explained

| Parameter | Value | Why |
|-----------|-------|-----|
| `n-gpu-layers = 99` | All layers on GPU | Unified memory on iGPU — no PCIe transfer penalty |
| `cache-type-k/v = q8_0` | 8-bit KV cache | q4_0 causes 92% slowdown on iGPU (dequant overhead). q8_0 halves memory with +0.004 perplexity |
| `spec-type = draft-mtp` | MTP speculative decoding | Uses auxiliary prediction heads in the model. Requires MTP-specific GGUF |
| `spec-draft-n-max = 3` | Predict 3 tokens per step | 91% acceptance rate at this setting |
| `batch-size = 2048` | Larger batch | Saturates iGPU compute units during prompt processing |
| `threads = 8` | Physical cores only | SMT threads add overhead without benefit for matrix math |

## Benchmarks

| Configuration | Eval tok/s | Prompt tok/s | MTP Acceptance |
|---------------|-----------|-------------|----------------|
| **MTP + APEX + q8_0 + ngl99** | **28.68** | 22.29 | 91%, mean len 3.62 |
| Non-MTP Q4_K_M + q8_0 + ngl99 | 2.14 | 6.75 | — |
| Non-MTP Q4_K_M + q4_0 + split | 1.30 | 4.71 | — |

**MTP + APEX is 13× faster** than the non-MTP Q4_K_M baseline.

### What MTP Does

Multi-Token Prediction uses auxiliary "draft" prediction heads trained alongside the main transformer. Instead of generating one token per forward pass, the model predicts 3-4 tokens speculatively, then verifies them in a single attention pass. At 91% acceptance with mean length 3.62, roughly 3.6 out of 4 tokens are accepted per step — effectively 3.6× fewer attention computations.

**MTP requires GGUF files with grafted prediction heads.** Standard GGUFs won't work — you need MTP-converted variants.

## KV Cache Quantization — The Hidden Killer

KV cache quantization is the most impactful and most misunderstood optimization for iGPU setups.

**The problem:** Quantized KV caches must be dequantized during attention computation. On integrated GPUs sharing system RAM, the dequantization overhead dominates:
- `q4_0` cache: Saves memory but requires expensive dequantization → **1-2 tok/s**
- `q8_0` cache: Halves memory with minimal dequant overhead → **28 tok/s with MTP**
- `f16` cache: No dequant cost but too heavy for 64K context on 90 GB RAM

This is specific to iGPU/Vulkan. Discrete GPUs with fast VRAM handle q4_0 dequant much better. The research from [OmniForge](https://omniforge.online/blog/your-local-llm-is-slow-because-of-five-config-flags) confirms up to 92% slowdown at long context with q4_0 on integrated graphics.

## APEX vs Q4_K_M

APEX (Adaptive Precision Expert) quantization is MoE-aware — it uses higher precision for active expert parameters and lower precision for inactive ones. On Qwen3.6-35B-A3B:

- **APEX I-Compact:** 17 GB, better perplexity than Q4_K_M at smaller size
- **Q4_K_M:** 21 GB, uniform quantization, slower and larger

The MoE-aware approach gives better quality at smaller size, which matters for iGPU where memory bandwidth is the bottleneck.

## Router Mode + Desktop Integration

For managing multiple models, llama.cpp's router mode (`--models-preset`) loads models on-demand with LRU eviction. We built a [Cinnamon panel applet](https://github.com/githabideri/cinnamon-llama-router-applet) for direct load/unload control from the desktop panel.

## Reproducing This Setup

1. **Get llama.cpp b483+** with Vulkan support:
   ```bash
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp && mkdir build && cd build
   cmake .. -DGGML_VULKAN=ON -DLLAMA_BUILD_SERVER=ON
   cmake --build . --config Release -j$(nproc)
   ```

2. **Download MTP model** from [SC117's repo](https://huggingface.co/SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF)

3. **Create `models.ini`** with the configuration above

4. **Run router:**
   ```bash
   llama-server --models-preset models.ini --models-max 2 --host 127.0.0.1 --port 8082
   ```

5. **Test:** Send a chat completion request to `http://127.0.0.1:8082/v1/chat/completions`

## Caveats

- **MTP GGUFs are rare.** Only a handful of models have MTP heads grafted. The SC117 heretic variant is one of the public ones.
- **Vulkan MTP bug (#23199)** affects dual ROCm+Vulkan builds. Pure Vulkan (`GGML_HIP=OFF`) is unaffected.
- **Memory pressure:** 35B MoE models need ~17-21 GB. With router mode and `--models-max 2`, expect 35-40 GB peak with two models loaded.
- **Prompt eval is slower** (~22 tok/s) than generation (~28 tok/s) on iGPU — this is hardware-bound.

## Conclusion

35B models are now practical on mid-range laptops with integrated graphics, thanks to MTP speculative decoding. The combination of MTP (13× speedup), APEX quantization (better quality/size), and q8_0 KV cache (avoiding dequant bottleneck) makes interactive use viable at 28 tok/s.

At these speeds the model is usable for actual work — coding assistance, research, tool-calling — not just a demo.

## References

- llama.cpp MTP implementation: [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)
- MTP technical overview: [eeshansrivastava89/gist](https://gist.github.com/eeshansrivastava89/85797104af34181944bfd1360d69e8af)
- KV cache quantization research: [OmniForge guide](https://omniforge.online/blog/your-local-llm-is-slow-because-of-five-config-flags)
- AMD iGPU benchmarks: [StochasticSandbox 780M deep dive](https://stochasticsandbox.com/posts/deep-dive-mini-pc-local-ai-2026-04-04/)
- Router API documentation: [glukhov.org unload guide](https://www.glukhov.org/llm-hosting/llama-cpp/unload-llama-cpp-router-models/)
- MTP model: [SC117/Qwen3.6-35B-A3B MTP APEX GGUF](https://huggingface.co/SC117/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-APEX-GGUF)
- Cinnamon applet: [github.com/githabideri/cinnamon-llama-router-applet](https://github.com/githabideri/cinnamon-llama-router-applet)
