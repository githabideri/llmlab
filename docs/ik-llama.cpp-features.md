# ik_llama.cpp Features and Optimizations

**Purpose:** Document CPU-specific optimizations in ik_llama.cpp fork

## Overview

**ik_llama.cpp** is a fork of llama.cpp focused on CPU performance optimizations. It's particularly valuable for CPU-only inference setups where GPU acceleration isn't available.

### Key Differentiators

| Feature | llama.cpp | ik_llama.cpp |
|---------|-----------|---------------|
| CPU Flash Attention | Partial | Full (iqk kernels) |
| IQK Quantizations | Basic | Extended (IQ1_XXS through IQ5_KS) |
| MoE Optimizations | Standard | Fused operations |
| AVX2 Utilization | Yes | Enhanced |

---

## CPU Flash Attention (iqk)

### How It Works on CPU

**Traditional attention:** O(n²) complexity, computes full attention matrix

**Flash Attention:** Blocks computation to reduce memory access, O(n) with better cache locality

### ik_llama.cpp Implementation

Uses specialized `iqk_mul_mat` kernels with AVX2 optimizations:

1. **Reduced memory bandwidth:** Processes attention in tiles, keeping data in L1/L2 cache
2. **AVX2 vectorization:** 256-bit SIMD instructions process 8 floats or 32 int8 values per cycle
3. **Fused operations:** Combines Q×K^T and softmax into single kernel, avoiding intermediate memory writes
4. **Pre-computed tables:** For quantized models, uses lookup tables for faster matrix multiplication

### Performance Gains

| Operation | Typical Improvement |
|-----------|---------------------|
| Prompt processing | 20-40% faster on AVX2 CPUs |
| Generation | 10-20% faster due to better KV cache handling |
| Long context (>4k tokens) | Especially beneficial where attention dominates |

---

## IQK Flash Attention Kernels

The `iqk` (integer quantized kernel) implementation includes:

- **iqk_mul_mat.cpp:** Core quantized matrix multiplication
- **iqk_flash_attn.cpp:** Flash Attention for CPU
- **Specialized kernels** for different block sizes (64×64, 128×128, 192×192, 256×256, 576×512)

These are automatically selected based on model architecture and sequence length.

---

## SOTA Quantizations

ik_llama.cpp supports extended quantization types:

### Ultra-Low Bit (1-2 bit)
- **IQ1_XXS/IQ1_S/IQ1_M/IQ1_KS:** Ultra-low bit with excellent quality

### Trellis-Based
- **IQ2_KS/IQ2_K/IQ3_KS:** Trellis-based quantizations

### High-Performance
- **IQ4_KS/IQ5_KS:** High-performance 4-5 bit quantizations

### Standard
- All standard quants: Q4_K_M, Q5_K_M, Q6_K, Q8_0, etc.

---

## MoE Optimizations

For Mixture-of-Experts models (like Qwen3.5-35B-A3B):

- **Fused MoE operations:** Combines routing + expert computation
- **Tensor overrides:** Allows partial GPU offload for large MoE models
- **Better expert selection:** Optimized gating network evaluation

---

## Build Configuration

```bash
cmake . -B build \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=ON \
  -DGGML_CCACHE=ON  # Optional, for faster builds
```

### Flags Explained

| Flag | Purpose |
|------|---------|
| `GGML_NATIVE=ON` | Enables AVX2/AVX512/FMA optimizations for your CPU |
| `LLAMA_CURL=ON` | Enables HuggingFace model downloads via `--hf-repo` |
| `GGML_CCACHE=ON` | Speeds up rebuilds with compilation cache |

---

## When to Use ik_llama.cpp

### Recommended For:
- ✅ CPU-only inference setups
- ✅ AVX2-capable CPUs (most modern x86-64)
- ✅ Long context workloads (>4k tokens)
- ✅ Systems with memory bandwidth constraints

### Consider vanilla llama.cpp for:
- ⚠️ GPU-heavy setups (benefits are smaller)
- ⚠️ CPUs without AVX2 (very old hardware)

---

## References

- **ik_llama.cpp:** https://github.com/ikawrakow/ik_llama.cpp
- **llama.cpp:** https://github.com/ggml-org/llama.cpp
- **Flash Attention paper:** https://arxiv.org/abs/2205.14135
- **IQK quantization:** https://github.com/ikawrakow/ik_llama.cpp/discussions/8

---

*Source: Migrated from locmox-private/llama-cpp-lxc-setup.md (2026-04-11)*
