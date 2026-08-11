# DeepSeek-V4-Flash-162B (REAP K144)

**Base model:** [0xSero/DeepSeek-V4-Flash-162B](<https://huggingface.co/0xSero/DeepSeek-V4-Flash-162B>)
**GGUF variant:** [0xSero/DeepSeek-V4-Flash-162B-GGUF](<https://huggingface.co/0xSero/DeepSeek-V4-Flash-162B-GGUF>)
**Architecture:** DeepSeek V4 Flash (MoE, REAP-pruned K144)
**Quant used:** `DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf` (49 GB)
**Tested hardware:** AMD Ryzen 7 7840U + Radeon 780M iGPU (96 GB unified RAM)

## Quick Facts

| Param | Value |
|-------|-------|
| Total parameters | 162B |
| Active parameters | ~4B (6 routed + 1 shared expert) |
| Experts per layer | 144 (REAP-pruned from 256) |
| Layers | 43 |
| Quantization | Q2-REAP-ds4 (DS4/DwarfStar-specific) |
| KV cache | f16 K / f16 V (q8_0 K-cache produces incorrect output) |
| Tested backend | llama.cpp b483 + Vulkan (RADV) |
| Current stable profile | `ctx=16384`, `-ngl 99`, `-ncmoe 999`, `-t 6`, `load-mode none` |

## What This Model Is

REAP (Routing-Enhanced Activation Pruning) variant of [deepseek-ai/DeepSeek-V4-Flash](<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>). The 162B version trades some capacity for higher prefill speed and a more conservative memory profile.

This is a **DS4/DwarfStar-specific GGUF** — it uses specialized tensor layouts and metadata from the [antirez/ds4](<https://github.com/antirez/ds4>) runtime. Standard llama.cpp may not load it without DSV4 support (b483+ required).

## Working Config

```bash
llama-server \
  -m /path/to/DeepSeek-V4-Flash-Spark-Mini-Q2-REAP-ds4.gguf \
  --load-mode none \
  --no-host \
  --no-warmup \
  -ngl 99 \
  -ncmoe 999 \
  -t 6 \
  -tb 8 \
  -c 16384 \
  -np 1 \
  -b 2048 \
  -ub 1024 \
  -fa on \
  -ctk f16 \
  -ctv f16 \
  --jinja
```

**Critical flags:**
- `--load-mode none --no-host` — eliminates mmap/GTT duplication that causes OOM on 96 GB RAM
- `--no-warmup` — avoids transient allocation peak during startup
- `-ncmoe 999` — keeps all routed experts on CPU (GPU experts hurt performance)
- `-t 6` — optimal for CPU/GPU coordination on shared DDR5

## Performance

### Baseline (Ryzen 7 7840U + Radeon 780M)

| Metric | Value |
|--------|-------|
| **Decode throughput** | 3.1-3.4 tok/s |
| **Prefill throughput** | 6.5 tok/s (short prompt) |
| **Load time** | ~45 seconds |
| **RAM usage** | ~50 GB peak (no swap pressure) |

### Configuration Impact

| Change | Effect |
|--------|--------|
| `--load-mode none --no-host` (vs mmap) | +74% decode (1.9 → 3.3 tok/s) |
| `powerprofilesctl set performance` | +60% prefill (4.2 → 6.5 tok/s) |
| `-t 6` (vs `-t 8`) | +5% decode |
| `-t 4` | -10% decode |
| `-t 12` | -12% decode |
| `-ncmoe 40` (GPU experts) | -6% decode |
| `-fa on/-fa off` | Neutral at short context |
| `--no-repack` | -67% decode (critical: keep repacking enabled) |

### Prefill by Prompt Length

| Prompt Tokens | Prefill tok/s | Notes |
|--------------|---------------|-------|
| 619 | 22.9 | Baseline |
| 621 | 21.0 | Warm repeat |
| 1235 | 14.8 | ~2× prompt |
| 8192 | ❌ timed out | Request exceeded 600s timeout |

Prefill degrades with prompt length (~35% drop at 2× tokens). 8K prompt timed out — likely needs chunked prefill or longer timeout. Second run (same day) showed degraded results (10-11 tok/s at 620 tokens), suggesting router/model state variance between sessions.

## Why This Config

### Loading Path

The default mmap path causes OOM on 96 GB RAM due to Vulkan buffer duplication during startup:
- mmap + Vulkan upload → temporary 80+ GB peak → OOM
- `--load-mode none --no-host` → ~50 GB peak → fits

This matches a known llama.cpp loader regression (post-July 2026 `--load-mode` refactor) where CPU-offloaded MoE + Vulkan + no-mmap creates staging buffer duplication.

### Expert Placement

All routed experts on CPU (`-ncmoe 999`) is optimal:
- GPU offload of experts adds synchronization overhead
- CPU matmuls on Zen 4 are competitive for sparse expert selection
- DDR5 bandwidth is shared between CPU and iGPU anyway

### Thread Count

6 threads is the sweet spot for CPU/GPU coordination:
- Fewer threads → CPU underutilized
- More threads → DDR5 contention with iGPU
- 12 threads clearly hurts (memory bandwidth saturation)

## Bottlenecks

1. **Lightning Indexer not supported on Vulkan** — falls back to CPU, adds latency
2. **Fused DSV4 HC ops not supported on Vulkan** — pre/comb/post operations fall back
3. **DDR5 bandwidth contention** — CPU and iGPU share the same memory controller
4. **Q8_0 Vulkan kernels** — may be suboptimal on RDNA3 vs lower-bit quants
5. **Power profile** — `power-saver` mode significantly limits GPU clocks

## KV Cache Warning

**Do not use `--cache-type-k q8_0`** — this produces incorrect output on DeepSeek V4. The quantized K-cache forces `self_k_rot` which breaks the sparse CSA/HCA/Lightning-Indexer attention path.

Safe combinations:
- `-ctk f16 -ctv f16` — correct, verified
- `-ctk f16 -ctv q8_0` — correct, verified
- `-ctk q8_0` — **incorrect/gibberish output**

## Context Degradation

Pending full sweep. Expected pattern based on architecture:
- V4's specialized sparse attention should degrade less than traditional GQA
- KV cache pressure will increase with context
- Prefill speed is expected to drop significantly at 8K+ tokens

## Caveats

- **DS4-specific GGUF** — requires llama.cpp b483+ with DSV4 support. Older builds will fail to load.
- **Q2 quantization** — most aggressive quant; quality vs Q4/Q6 unknown for this model.
- **No vision support** — text-only model.
- **Power profile matters** — `power-saver` mode can halve prefill speed on laptops.
- **Swap pressure** — startup can fill 8 GB swap; disable swap or use `--load-mode none` to avoid.

## Verdict

**🟡 LAB / EXPERIMENTAL**

- ✅ Remarkable for hardware: 162B MoE on a laptop CPU+iGPU at 3+ tok/s
- ✅ Stable with correct config (`load-mode none`, `-ncmoe 999`, `-t 6`)
- ⚠️ DS4-specific GGUF limits compatibility
- ⚠️ Q2 quant quality unverified
- ⚠️ Prefill speed needs optimization for real-world use
- ❌ Not production-ready without further testing

## DS4/ROCm on gfx1103 — Build Success, Runtime Blocker

DS4's ROCm backend **compiles and links for gfx1103** after ~10 lines of compatibility patches. The rocWMMA whitelist is conservative — gfx1103 supports WMMA instructions (verified by runtime test).

**Remaining blocker:** Spark-Mini shape (`n_expert=144`) not in DS4's runtime shape table. Adding `DS4_SHAPE_SPARK_MINI` would enable testing.

See: `../reports/2026-08-12-ds4-rocm-gfx1103-build-success.md`

## References

- [0xSero/DeepSeek-V4-Flash-162B](<https://huggingface.co/0xSero/DeepSeek-V4-Flash-162B>) — base model
- [0xSero/DeepSeek-V4-Flash-162B-GGUF](<https://huggingface.co/0xSero/DeepSeek-V4-Flash-162B-GGUF>) — GGUF quant
- [antirez/ds4](<https://github.com/antirez/ds4>) — DwarfStar runtime/converter
- [0xSero/deepseek-spark](<https://github.com/0xSero/deepseek-spark>) — Spark deployment
- [REAP paper](<https://arxiv.org/abs/2510.13999>) — Routing-Enhanced Activation Pruning
- [llama.cpp DSV4 support](<https://github.com/ggml-org/llama.cpp/issues/25382>) — V4 quantized K-cache bug
- [llama.cpp loader regression](<https://github.com/ggml-org/llama.cpp/issues/26110>) — `--load-mode` refactor impact on MoE

## Changelog

- **2026-08-11:** Initial evaluation on Ryzen 7 7840U + Radeon 780M. Config optimized, performance baseline established. Prefill sweep pending.
