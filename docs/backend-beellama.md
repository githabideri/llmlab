# BeeLlama.cpp — Backend Setup and Features

**Purpose:** Document the BeeLlama.cpp fork — a performance-focused llama.cpp derivative with DFlash speculative decoding and TurboQuant/TCQ KV-cache compression.

**Upstream:** <https://github.com/Anbeeld/beellama.cpp>

---

## Overview

**BeeLlama.cpp** is a fork of llama.cpp that keeps the familiar `llama-server` tooling and OpenAI-compatible API, then adds:

| Feature | What it does |
|---------|-------------|
| **DFlash speculative decoding** | Small draft model cross-attends to target hidden states via a ring buffer, proposes tokens ahead, target verifies in one forward pass. Up to 4-5× speedup on structured output. |
| **TurboQuant / TCQ KV-cache** | Trellis-coded quantization for KV cache at 2-3 bits. 3-7× more context in the same VRAM. |
| **Reasoning-loop protection** | Detects repeated hidden reasoning output and intervenes. |
| **Adaptive draft control** | Server adjusts active draft depth at runtime based on real-time acceptance rates. |
| **`--mmproj-gpu-swap`** | Solves VRAM tension between vision encoder and speculative decoding on constrained GPUs. |

Not all features are used in our setup. We primarily leverage **DFlash** for decode speedup and **asymmetric KV cache** (`q5_0`/`q4_1`) for quality/size trade.

### How DFlash Works

1. Target model captures hidden states into a ring buffer
2. Draft model (~1 GB, separate GGUF) cross-attends to the most recent `--spec-dflash-cross-ctx` tokens
3. Draft proposes tokens; target verifies them in a single forward pass
4. Accepted tokens are "free"; rejected tokens rollback to last accepted state

**Key difference from MTP:** DFlash works with any unmodified model GGUF — no specially converted MTP GGUFs needed. The draft model is fully separate from the target.

### DFlash vs MTP (brief comparison)

| | DFlash | MTP |
|---|--------|-----|
| Speedup on structured output | 3-5× | 2-3× |
| Speedup on prose | 1-2× | 1.5-2× |
| Requires special GGUF | No | Yes |
| Draft model size | ~1 GB (separate) | Baked into weights |
| Adaptive depth | Yes | Fixed |

On CUDA hardware the gap narrows because DFlash uses a GPU-side ring buffer. On bandwidth-limited hardware (AMD CPU + integrated GPU), MTP can win because prediction heads ride along for free during weight reads.

---

## Build Configuration

We build from source for our CUDA 12.5 + RTX 3090 (compute 8.6) setup.

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build -j
```

| Flag | Purpose |
|------|---------|
| `GGML_CUDA=ON` | Enable CUDA backend |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | Required for TurboQuant and TCQ cache types |
| `CMAKE_CUDA_ARCHITECTURES=86` | Target RTX 3090 (Ada, compute 8.6) |
| `CMAKE_BUILD_TYPE=Release` | Optimized build |

Building from source with `-DGGML_NATIVE=ON` may give a small additional performance benefit, but is not currently used.

**Binary location:** `build/bin/llama-server` (shared library build, resolves CUDA deps at runtime).

---

## Model Files Required

DFlash needs **three** files (vs one for standard llama.cpp):

| File | Purpose | Size | Source |
|------|---------|------|--------|
| Target GGUF | Main model (e.g., Qwen3.6-27B-Q5_K_S) | ~18 GB | [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) or [bartowski/Qwen_Qwen3.6-27B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.6-27B-GGUF) |
| DFlash draft GGUF | Draft model for speculative decoding | ~1 GB | [Anbeeld/Qwen3.6-27B-DFlash-GGUF](https://huggingface.co/Anbeeld/Qwen3.6-27B-DFlash-GGUF) |
| mmproj GGUF | Vision projector (optional) | ~885 MB | [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) |

The draft model shares the target's token embedding and LM head at runtime, so the GGUF file only contains DFlash-specific weights.

---

## Launch Command (Precision Combo)

From the [official quickstart](https://github.com/Anbeeld/beellama.cpp/blob/main/docs/quickstart-qwen36-dflash.md), tuned for a 24 GB GPU:

```bash
llama-server \
  -m /path/to/Qwen3.6-27B-Q5_K_S.gguf \
  --mmproj /path/to/mmproj-Qwen_Qwen3.6-27B-f16.gguf \
  --no-mmproj-offload \
  --spec-draft-model /path/to/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-dflash-cross-ctx 1024 \
  -ngl all \
  --spec-draft-ngl all \
  --kv-unified \
  -np 1 \
  -b 2048 -ub 512 \
  --ctx-size 163840 \
  --cache-type-k q5_0 --cache-type-v q4_1 \
  --flash-attn on \
  --jinja \
  --mmap --mlock \
  --no-host \
  --reasoning on \
  --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.6 --top-k 20 --top-p 1.0 --min-p 0.0 \
  --host 0.0.0.0 \
  --port 8080
```

### Key Flags Explained

| Flag | Value | What it controls |
|------|-------|------------------|
| `--spec-type` | `dflash` | Enables DFlash speculative decoding |
| `--spec-draft-model` | path | DFlash draft model to load |
| `--spec-draft-ngl` | `all` | Offload all draft layers to GPU |
| `--spec-dflash-cross-ctx` | `1024` | How many tokens of target hidden state the drafter sees. Higher = more context for cross-attention, lower = saves VRAM |
| `-ngl` | `all` | Offload all target model layers to GPU. **Required** for Qwen3.5/3.6 models — without it, fused GDN tensors fall back to CPU and disable GDN entirely. See [llama.cpp #24712](https://github.com/ggml-org/llama.cpp/issues/24712) |
| `-b` | `2048` | Logical batch size for prompt evaluation |
| `-ub` | `512` | Physical microbatch size. Critical for prefill speed. 512 is the sweet spot for 24 GB VRAM with 160K context + vision. |
| `--cache-type-k` | `q5_0` | K cache quantization. `q5_0` is the recommended default for Q5_K_S targets. |
| `--cache-type-v` | `q4_1` | V cache quantization. Keeps cache footprint reasonable while preserving better tail behavior than `q4_0`. |
| `--no-mmproj-offload` | — | Run vision projector on CPU, freeing GPU VRAM. Skip on macOS (unified memory). |
| `--kv-unified` | — | Single KV buffer shared across server slots |
| `--no-host` | — | Bypass host buffer, allowing extra buffers to be used |
| `--reasoning` | `on` | Enable reasoning output handling — thinking tokens give the drafter richer context for better predictions |
| `--chat-template-kwargs` | `{"preserve_thinking":true}` | Preserve thinking tokens across turns for better output quality and stronger drafter predictions |

### Environment Variables (DFlash Debugging)

| Variable | Values | Purpose |
|----------|--------|---------|
| `GGML_DFLASH_PROFILE` | `0`, `1`/`default` | `1` enables summary, replay, copy, and verify timing. Add `prefill` or `trace` for deeper profiling |
| `GGML_DFLASH_DEBUG` | `0`, `1` | Enable DFlash debug logs (prefill route, capture decisions) |
| `GGML_DFLASH_KV_CACHE_MODE` | `both`, `k`, `v`, `off` | Control which KV cache the drafter keeps |

---

## KV Cache Quantization Presets

From [Anbeeld's KV Cache Quantization Benchmarks](https://anbeeld.com/articles/kv-cache-quantization-benchmarks-for-long-context):

| K / V | % of bf16 size | 99.9% precision | Use case |
|-------|---------------:|----------------:|----------|
| `q8_0` / `q6_0` | 46.9% | 94.33% | Recommended high-end preset |
| `q5_0` / `q4_1` | 32.8% | 96.50% | **Our default** — good quality/size trade |
| `q5_0` / `q4_0` | 31.3% | 96.18% | Slightly smaller, same tier |
| `q4_0` / `q4_0` | 28.1% | 94.34% | VRAM-constrained |
| `turbo3_tcq` / `turbo3_tcq` | 20.3% | 86.06% | 3-bit TCQ — quality drops noticeably |
| `turbo3_tcq` / `turbo2_tcq` | 17.2% | ~80% | Asymmetric 2.75 bpv — tight VRAM |
| `turbo2_tcq` / `turbo2_tcq` | 14.1% | 63.53% | Last resort, rough summarization only |

**Our choice:** `q5_0`/`q4_1` — asymmetric K-first allocation. K cache gets higher precision because attention quality depends more on K than V. Same footprint as `q4_1`/`q4_1` but better tail behavior.

---

## Known Issues

### Fused GDN + `--fit on` (llama.cpp upstream)

Without `-ngl all`, llama.cpp's auto-fit (`--fit on`) cannot handle fused GDN tensors in Qwen3.5/3.6 architecture. Result: layer 0 assigned to CPU, GDN disabled, ~3× speed regression.

- [llama.cpp #24712](https://github.com/ggml-org/llama.cpp/issues/24712) — GDN warning, layer 0 on CPU
- [llama.cpp #20492](https://github.com/ggml-org/llama.cpp/issues/20492) — 3× regression with `--fit on`
- [llama.cpp #20436](https://github.com/ggml-org/llama.cpp/issues/20436) — Fused GDN kernel cache anti-pattern

**Fix:** Always use `-ngl all` explicitly for Qwen3.5/3.6 models.

### CUDA FA_ALL_QUANTS slowdown on Blackwell

Issue [#67](https://github.com/Anbeeld/beellama.cpp/issues/67) (Jun 2026): 3× slower generation with `GGML_CUDA_FA_ALL_QUANTS=ON` on Blackwell GPUs (RTX PRO 6000, CUDA 13.3). Not our setup — we're on Ada (RTX 3090), CUDA 12.5. No reports of this on Turing/Ada.

### Vulkan DFlash limitation

On Vulkan, DFlash falls back to a CPU ring buffer path (vs GPU ring on CUDA). TCQ cache types are CUDA-only. Not relevant for our CUDA setup.

---

## When to Use BeeLlama vs Mainline llama.cpp

### Use BeeLlama when:
- You want DFlash speculative decoding (2-5× on structured output)
- You need TCQ KV-cache compression for extreme context lengths
- You want reasoning-loop protection
- You're on CUDA and want the best decode performance

### Stick with mainline llama.cpp when:
- You need the broadest backend support (Vulkan, SYCL, Metal with full features)
- You don't need speculative decoding
- You want the largest community and most up-to-date upstream fixes
- You're on non-CUDA hardware where DFlash falls back to CPU

---

## References

| Resource | URL |
|----------|-----|
| BeeLlama.cpp (main fork) | <https://github.com/Anbeeld/beellama.cpp> |
| BeeLlama args reference | <https://github.com/Anbeeld/beellama.cpp/blob/main/docs/beellama-args.md> |
| BeeLlama features comparison | <https://github.com/Anbeeld/beellama.cpp/blob/main/docs/beellama-features.md> |
| Quickstart: Qwen 3.6 DFlash | <https://github.com/Anbeeld/beellama.cpp/blob/main/docs/quickstart-qwen36-dflash.md> |
| BeeLlama build docs | <https://github.com/Anbeeld/beellama.cpp/blob/main/docs/build.md> |
| DFlash draft model (HF) | <https://huggingface.co/Anbeeld/Qwen3.6-27B-DFlash-GGUF> |
| Target model Q5_K_S (HF) | <https://huggingface.co/bartowski/Qwen_Qwen3.6-27B-GGUF> |
| KV cache quant benchmarks | <https://anbeeld.com/articles/kv-cache-quantization-benchmarks-for-long-context> |
| DFlash on Strix Halo benchmark | <https://sleepingrobots.com/dreams/beellama-dflash-strix-halo/> |
| spiritbuun/buun-llama-cpp (TCQ + DFlash origin) | <https://github.com/spiritbuun/buun-llama-cpp> |
| TCQ paper (HuggingFace dataset) | <https://huggingface.co/datasets/spiritbuun/turboquant-tcq-kv-cache> |
| llama.cpp GDN issues | [#24712](https://github.com/ggml-org/llama.cpp/issues/24712), [#20492](https://github.com/ggml-org/llama.cpp/issues/20492), [#20436](https://github.com/ggml-org/llama.cpp/issues/20436) |
| BeeLlama CUDA FA slowdown (Blackwell) | <https://github.com/Anbeeld/beellama.cpp/issues/67> |
