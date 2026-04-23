# Qwen3.6-27B on RTX 3090

**Model:** Qwen3.6-27B (Dense)  
**Tested Quantization:** Q4_K_M  
**Hardware:** 1× RTX 3090 24 GB  
**Runtime:** llama.cpp v616 (c0159f9)  
**Status:** ✅ Production-ready (single GPU, long-context profile, text-only)
**Multimodal:** ⚠️ Tested and verified, not deployed in active service

---

## Quick Facts

| Parameter | Value |
|-----------|-------|
| **Architecture** | qwen36 (hybrid recurrent + attention) |
| **Parameters** | 27 billion |
| **Context Window** | 262,144 tokens (native), 204,800 tested |
| **Embedding Dimension** | 5120 |
| **Vocabulary Size** | 248,320 |
| **Quantization Tested** | Q4_K_M |
| **Multimodal** | ⚠️ Tested, not deployed (mmproj, f16, ~885 MiB) |

---

## Validated Configurations

Three profiles were benchmarked and validated on a single RTX 3090 24 GB.

### 1. Best Default (Text)

```bash
llama-server \
  --device CUDA0 \
  --model Qwen3.6-27B-Q4_K_M.gguf \
  --ctx-size 131072 \
  --gpu-layers all \
  --parallel 1 \
  --flash-attn on \
  --batch-size 2048 \
  --ubatch-size 128 \
  --cache-type-k q8_0 \
  --cache-type-v f16 \
  --metrics
```

| Metric | Value |
|--------|-------|
| PP @ 512 tokens | 711 tok/s |
| PP @ 2048 tokens | 831 tok/s |
| TG @ 128 tokens | 24.6 tok/s |
| Peak VRAM | ~22.3 GiB |

### 2. Multimodal (Tested, Not Deployed)

Same as default, plus:

> **Note:** This profile was benchmarked and verified working, but is **not** the currently active service. The running systemd unit does not include `--mmproj`.

```bash
  --mmproj mmproj-Qwen_Qwen3.6-27B-f16.gguf
```

| Metric | Value |
|--------|-------|
| PP @ 2048 tokens | 834 tok/s |
| TG @ 128 tokens | 25.7 tok/s |
| Peak VRAM | ~23.4 GiB |

Image input verified via `/v1/chat/completions` with inline image data.

### 3. Long Context

```bash
llama-server \
  --device CUDA0 \
  --model Qwen3.6-27B-Q4_K_M.gguf \
  --ctx-size 204800 \
  --gpu-layers all \
  --parallel 1 \
  --flash-attn on \
  --batch-size 2048 \
  --ubatch-size 128 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --metrics
```

| Metric | Value |
|--------|-------|
| PP @ 512 tokens | 688 tok/s |
| PP @ 2048 tokens | 808 tok/s |
| TG @ 128 tokens | 24.1 tok/s |
| Peak VRAM | ~22.8 GiB |

**Key detail:** `ctx-size 204800` only fits with `q8_0/q8_0` KV cache — `q8_0/f16` fails with CUDA OOM on KV allocation.

---

## Benchmark Findings

### Ubatch

| ubatch | PP @ 2048 | TG | Peak VRAM |
|--------|----------|-----|-----------|
| 128 | 851 tok/s | 25.5 tok/s | ~22.3 GiB |
| 512 | 831 tok/s | 24.6 tok/s | ~22.6 GiB |
| 1024 | 799 tok/s | 25.2 tok/s | ~23.1 GiB |

**Winner:** `ubatch 128` — best throughput and lowest VRAM.

### Batch

| batch | ubatch | PP @ 2048 | TG |
|-------|--------|----------|-----|
| 1024 | 512 | 821 tok/s | 24.3 tok/s |
| 2048 | 512 | 831 tok/s | 24.6 tok/s |
| 4096 | 512 | 809 tok/s | 24.1 tok/s |

**Winner:** `batch 2048` — best overall compromise.

### KV Cache Comparison

| K-cache | V-cache | PP @ 2048 | TG | Fits 204800? |
|---------|---------|----------|-----|-------------|
| q8_0 | f16 | 834 tok/s | 25.0 tok/s | ❌ CUDA OOM |
| q8_0 | q4_0 | 835 tok/s | 24.6 tok/s | ❌ CUDA OOM |
| q8_0 | q8_0 | 818 tok/s | 24.5 tok/s | ✅ ~22.8 GiB |
| f16 | f16 | — | — | ❌ compute-buffer OOM |

**Takeaway:** `q8_0/f16` is the best default at 131K context. `q8_0/q8_0` is the only profile that enables 204K context with full GPU offload on 24 GB VRAM.

---

## Known Limits

- **204K context** requires `q8_0/q8_0` KV cache (not `q8_0/f16`)
- **No multi-GPU split** tested — this model is designed for single 24 GB GPU
- **Long-context profile** is slightly slower (~3-5% TG reduction) vs default profile at 131K

---

## Changelog

### 2026-04-23: Initial Report
- Benchmarked Q4_K_M on single RTX 3090
- Established three profiles: text default, multimodal, long-context
- Validated mmproj for image input
- Tested ubatch/batch/KV cache combinations

---

## Related Documentation

- **Qwen3.5-27B (3× RTX 3060):** See `qwen3.5-27b.md`
- **Multi-GPU Tensor-Split:** See `docs/multi-gpu-tensor-split.md`
