# Qwen3-Coder-Next-REAP-40B-A3B

**Base model:** [lovedheart/Qwen3-Coder-Next-REAP-40B-A3B](https://huggingface.co/lovedheart/Qwen3-Coder-Next-REAP-40B-A3B)  
**GGUF quants:** [mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-GGUF](https://huggingface.co/mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-GGUF) (static K_M/Q2/etc.)  
**GGUF imatrix:** [mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-i1-GGUF](https://huggingface.co/mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-i1-GGUF) (IQ quants)

## Detailed Quant Reports

- **[i1-IQ3_M](./qwen3-coder-next-reap-40b-a3b-iq3m.md)** — Full benchmark + agentic testing (2026-02-14)
  - Speed: 22.4 tok/s empty, 18-19 tok/s real-world
  - VRAM: 21.0 GB @ 262k context (2×12GB GPUs)
  - Status: ✅ Recommended for production
- Q2_K_XL — Baseline quant (testing pending)

## Architecture

- **Type:** Hybrid DeltaNet + Gated Attention MoE
- **Total params:** 40B (3B activated per token)
- **Layers:** 48 total
  - **12 attention layers** (require KV cache)
  - **36 DeltaNet layers** (linear attention, no KV cache)
- **Layout:** 12 × (3 × DeltaNet-MoE → 1 × Attention-MoE)
- **KV heads:** 2 (GQA with 16 Q heads = 8:1 ratio)
- **Head dimension:** 256
- **Hidden dimension:** 2048
- **MoE:** 256 experts (10 active + 1 shared)
- **Native context:** 262,144 tokens
- **Compression:** REAP (50% expert pruning from 512→256)

## KV Cache Characteristics

**Key insight:** Only 12 of 48 layers use attention → **75% smaller KV cache** than standard transformers!

### Memory per token (measured 2026-02-07)

| Cache Config | KiB/token | 64k ctx | 128k ctx | 196k ctx | 262k ctx (max) |
|--------------|-----------|---------|----------|----------|----------------|
| **f16/f16** (unquantized) | 24 | 1.5 GB | 3.0 GB | 4.6 GB | 6.1 GB |
| **q8/q8** | 12 | 768 MB | 1.5 GB | 2.3 GB | 3.1 GB |
| **q8/q4** (recommended) | **~10** | **640 MB** | **1.25 GB** | **1.9 GB** | **2.6 GB** |
| **q4/q4** (aggressive) | 6 | 384 MB | 768 MB | 1.2 GB | 1.5 GB |

**Recommendation:** `--cache-type-k q8_0 --cache-type-v q4_0` balances quality + VRAM.

### Calculation breakdown

**Unquantized (f16/f16):**
- K: 12 layers × 2 kv_heads × 256 dim × 2 bytes = **12 KiB/tok**
- V: 12 layers × 2 kv_heads × 256 dim × 2 bytes = **12 KiB/tok**
- **Total: 24 KiB/tok**

**With q8/q4:**
- K (q8_0): 12 × 2 × 256 × 1 byte = **6 KiB/tok**
- V (q4_0): 12 × 2 × 256 × 0.5 byte = **3 KiB/tok**
- **Total: ~9 KiB/tok** (+ 1 KiB overhead = 10 KiB empirical)

## Quant Comparison

### Static quants (K_M/Q2/etc.)
- Standard quantization, uniform bit allocation
- Fast, predictable
- **Available:** Q2_K, Q2_K_XL, Q3_K_M, Q4_K_M, IQ4_XS, etc.

### IQ (imatrix) quants
- Importance-weighted quantization
- Better quality-per-byte
- **Faster on NVIDIA GPUs** (optimized CUDA kernels)
- **Available:** i1-IQ2_M, i1-IQ3_XXS, i1-IQ3_M, i1-IQ4_XS, i1-Q4_K_M

### Our collection (at `/mnt/models/gguf/qwen3-coder-next/`)

| File | Size | Type | Notes |
|------|------|------|-------|
| `Qwen3-Coder-Next-REAP-40B-A3B-Q2_K_XL.gguf` | 19 GB | Static Q2 | Original baseline |
| `Qwen3-Coder-Next-REAP-40B-A3B.i1-IQ3_M.gguf` | 18.2 GB | imatrix | **Downloaded 2026-02-14, ready for bench** |
| `Qwen3-Coder-Next-IQ4_XS.gguf` | 40 GB | imatrix | Full 60B variant (not REAP) |
| `Qwen3-Coder-Next-Q2_K.gguf` | 28 GB | Static | Full 60B variant |
| `Qwen3-Coder-Next-Q4_K_M.gguf` | 23 GB | Static | Full 60B variant |
| `Qwen3-Coder-Next-UD-TQ1_0.gguf` | 19 GB | UltraDeep | Experimental quant |

## Benchmarking Plan

### Test matrix (2026-02-14)

**Model:** `Qwen3-Coder-Next-REAP-40B-A3B.i1-IQ3_M.gguf`  
**Baseline:** `Qwen3-Coder-Next-REAP-40B-A3B-Q2_K_XL.gguf`

**Context levels to test:**
- 32k (baseline small)
- 64k (typical usage)
- 128k (heavy usage)
- 196k (current server config)
- 262k (native max)

**Metrics:**
- Prompt processing (tok/s)
- Generation speed fresh→filled
- VRAM footprint per GPU
- Quality (subjective, multi-turn coding task)

**Cache configs to compare:**
- q8/q4 (current standard)
- q8/q8 (quality check)
- q4/q4 (VRAM-constrained)

## Expected Performance (RTX 3060 12GB + GTX 1050 2GB)

Based on 2026-02-07 measurements with Q2_K_XL + q8/q4 cache:

| Setting | PP (tok/s) | TG (tok/s) | GPU0 | GPU1 | CPU offload |
|---------|------------|------------|------|------|-------------|
| 48k ctx, n-cpu-moe=0, tensor-split 14/10 | ~135 | ~21 | 7.9 GB | 8.8 GB | 0.6 GB |
| 64k ctx, same | ~137 | ~21 | ~8.5 GB | ~9.0 GB | 0.6 GB |
| 128k ctx, same | ~133 | ~21 | ~10.5 GB | ~9.5 GB | 0.6 GB |

**IQ3_M expectations:**
- Slightly slower PP (~5-10% due to irregular bit patterns)
- Similar TG (memory-bound anyway)
- **Better quality** at same size
- Potentially faster with newer CUDA kernel optimizations

---

**Status:** Ready for benchmark (i1-IQ3_M downloaded 2026-02-14 02:05 UTC)
