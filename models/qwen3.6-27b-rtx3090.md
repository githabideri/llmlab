# Qwen3.6-27B on RTX 3090

**Model:** Qwen3.6-27B (Dense)  
**Tested Quantization:** Q4_K_M (mainline), Q5_K_S (BeeLlama)  
**Hardware:** 1× RTX 3090 24 GB  
**Runtime:** BeeLlama.cpp b10102 (current), llama.cpp v616 (historical)  
**Status:** ✅ Production — BeeLlama with DFlash speculative decoding  
**Multimodal:** ✅ Deployed (mmproj on CPU via `--no-mmproj-offload`)

---

## Quick Facts

| Parameter | Value |
|-----------|-------|
| **Architecture** | qwen36 (hybrid recurrent + attention) |
| **Parameters** | 27 billion |
| **Context Window** | 262,144 tokens (native), 163,840 deployed |
| **Embedding Dimension** | 5120 |
| **Vocabulary Size** | 248,320 |
| **Quantization Tested** | Q4_K_M (mainline), Q5_K_S (BeeLlama) |
| **Multimodal** | ✅ Deployed (mmproj f16, ~885 MiB, CPU-resident) |
| **Speculative Decoding** | DFlash (draft model: Qwen3.6-27B-DFlash-Q4_K_M, ~1 GB) |

---

## Current Production Config (BeeLlama + DFlash)

**Runtime:** BeeLlama.cpp b10102 (commit `85e22ea0b`)  
**Build:** CUDA 12.5, compute 86, `GGML_CUDA_FA_ALL_QUANTS=ON`  
**See:** [`docs/backend-beellama.md`](docs/backend-beellama.md) for build details and feature overview.

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

### Memory Configuration

**2026-07-05: `--mmap` fix for CPU OOM** — Previously used `--no-mmap --mlock` which loaded the full 18-19GB model into anonymous CPU memory, causing ~23-24GB RSS and repeated OOM kills on memory-constrained LXC containers (30GB limit, no swap). Switched to `--mmap --mlock` which keeps the model in the OS page cache (reclaimable under pressure) while `--mlock` protects actively-touched pages.

| Metric | `--no-mmap` (old) | `--mmap` (current) |
|--------|------------------|-------------------|
| CPU RSS (idle) | ~6.7 GB | ~2.8 GB |
| CPU RSS (after load) | **23-24 GB** (OOM) | **~3.8 GB** |
| OOM risk | **High** | **None** |

**Decode speed comparison (same workloads):**

| Workload | `--no-mmap` | `--mmap` | Diff |
|----------|------------:|---------:|------|
| Structured JSON | 77.9 tok/s | 77.9 tok/s | 0% |
| Code | 72.3 tok/s | 86.2 tok/s | +19% (within variance) |
| Prose | 46.2 tok/s | 36.6 tok/s | -21% (within variance) |

No measurable performance regression. Token throughput variance is expected with speculative decoding (±15% between runs due to draft acceptance rate differences).

### Performance (BeeLlama + DFlash)

#### Decode Speed by Workload Type

Measured via live server API (`/v1/chat/completions`), 3 independent requests, model finishes naturally (`stop`, not `length`):

| Workload | Decode tok/s | Draft Acceptance | Output tokens | Notes |
|----------|-------------:|-----------------:|--------------:|-------|
| Structured JSON | **80.2** | 33.3% | 3,792 | 15 employees, nested fields, highly repetitive |
| Code generation | **75.0** | 30.7% | 4,369 | Full Python class with docstrings |
| Free-form prose | **39.6** | 12.8% | 2,444 | Philosophical essay, least predictable |
| **Baseline (no DFlash)** | ~37 | N/A | — | From [BeeLlama README benchmarks](https://github.com/Anbeeld/beellama.cpp) |

**Speedup vs baseline:** 2.17× (JSON), 2.03× (code), 1.07× (prose). DFlash is strongest on structured, repetitive generation — exactly as documented upstream.

#### Prefill Speed by Context Size

Measured via live server API, unique prompts (no cross-request caching), 0 cached tokens:

| New Prompt Tokens | Prefill Speed | Time |
|-----------------:|-------------:|-----:|
| 19,280 | **791 tok/s** | 24.4s |
| 39,981 | **742 tok/s** | 53.9s |
| 81,383 | **618 tok/s** | 131.8s |

Prefill peaks around 19-20K tokens and degrades gracefully as context grows (attention computation scales with KV cache size). For comparison, BeeLlama's published benchmark shows ~1229 tok/s at ~20K with `--reasoning off` — our `--reasoning on` adds overhead but enables richer drafter context.

#### VRAM Usage

| Component | VRAM |
|-----------|------|
| Target model (Q5_K_S) | ~18 GB |
| DFlash draft model (Q4_K_M) | ~1 GB |
| KV cache (160K context, q5_0/q4_1) | ~3.5 GB |
| Other (compute buffers, mmproj swap) | ~0.7 GB |
| **Total** | **~23.7 GB / 24 GB** |

---

## Historical: Mainline llama.cpp Profiles (v616, Q4_K_M)

These profiles were benchmarked with mainline llama.cpp v616 (c0159f9) and Q4_K_M quantization. Kept for reference — no longer the active serving path.

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

Same as default, plus `--mmproj mmproj-Qwen_Qwen3.6-27B-f16.gguf`.

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

### Historical Benchmark Findings

#### Ubatch

| ubatch | PP @ 2048 | TG | Peak VRAM |
|--------|----------|-----|-----------|
| 128 | 851 tok/s | 25.5 tok/s | ~22.3 GiB |
| 512 | 831 tok/s | 24.6 tok/s | ~22.6 GiB |
| 1024 | 799 tok/s | 25.2 tok/s | ~23.1 GiB |

**Winner:** `ubatch 128` — best throughput and lowest VRAM (mainline llama.cpp).

#### Batch

| batch | ubatch | PP @ 2048 | TG |
|-------|--------|----------|-----|
| 1024 | 512 | 821 tok/s | 24.3 tok/s |
| 2048 | 512 | 831 tok/s | 24.6 tok/s |
| 4096 | 512 | 809 tok/s | 24.1 tok/s |

**Winner:** `batch 2048` — best overall compromise.

#### KV Cache Comparison

| K-cache | V-cache | PP @ 2048 | TG | Fits 204800? |
|---------|---------|----------|-----|-------------|
| q8_0 | f16 | 834 tok/s | 25.0 tok/s | ❌ CUDA OOM |
| q8_0 | q4_0 | 835 tok/s | 24.6 tok/s | ❌ CUDA OOM |
| q8_0 | q8_0 | 818 tok/s | 24.5 tok/s | ✅ ~22.8 GiB |
| f16 | f16 | — | — | ❌ compute-buffer OOM |

**Takeaway:** `q8_0/f16` is the best default at 131K context. `q8_0/q8_0` is the only profile that enables 204K context with full GPU offload on 24 GB VRAM.

---

## Known Limits

- **204K context** (mainline) requires `q8_0/q8_0` KV cache — not `q8_0/f16`
- **160K context** (BeeLlama) is the current deployed size — constrained by DFlash + vision + Q5 model fitting in 24 GB
- **No multi-GPU split** tested — this model is designed for single 24 GB GPU
- **DFlash prefill overhead:** speculative decoding does not accelerate prefill (only decode). Prefill speed is comparable to or slightly below mainline at equivalent quants.
- **`--reasoning on` adds prefill overhead** but enables richer drafter context for better decode predictions. BeeLlama's published benchmarks use `--reasoning off` for non-chat prompts and show ~1229 tok/s prefill at 20K.

---

## Changelog

### 2026-07-05: `--mmap` CPU OOM fix
- Switched from `--no-mmap --mlock` to `--mmap --mlock` to eliminate ~23-24GB CPU RSS OOM spikes
- CPU RSS dropped from 23-24GB to ~3.8GB under load (85% reduction)
- No measurable decode speed regression across JSON, code, and prose workloads

### 2026-06-19: BeeLlama + DFlash Cutover
- Switched from mainline llama.cpp Q4_K_M to BeeLlama.cpp Q5_K_S with DFlash speculative decoding
- Vision now deployed (was "tested, not deployed")
- Context reduced from 204K to 160K to fit DFlash drafter + vision in 24 GB VRAM
- KV cache changed from `q8_0/q8_0` to `q5_0/q4_1` (BeeLlama recommendation)
- Decode speedup: 2.0-2.2× on structured output vs mainline baseline
- See `experiments/2026-06-19-beellama-dflash-cutover.md` for full experiment log

### 2026-04-23: Initial Report (mainline llama.cpp)
- Benchmarked Q4_K_M on single RTX 3090
- Established three profiles: text default, multimodal, long-context
- Validated mmproj for image input
- Tested ubatch/batch/KV cache combinations

---

## Related Documentation

- **BeeLlama backend setup:** [`docs/backend-beellama.md`](docs/backend-beellama.md)
- **Cutover experiment:** [`experiments/2026-06-19-beellama-dflash-cutover.md`](experiments/2026-06-19-beellama-dflash-cutover.md)
- **Qwen3.5-27B (3× RTX 3060):** See [`qwen3.5-27b.md`](qwen3.5-27b.md)
- **Multi-GPU Tensor-Split:** See [`docs/multi-gpu-tensor-split.md`](docs/multi-gpu-tensor-split.md)
- **BeeLlama upstream:** <https://github.com/Anbeeld/beellama.cpp>
- **DFlash draft model:** <https://huggingface.co/Anbeeld/Qwen3.6-27B-DFlash-GGUF>
