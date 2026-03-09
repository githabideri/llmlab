# Qwen3.5-27B Dense Model Report

**Model:** Qwen3.5-27B (Dense, NOT the 35B-A3B MoE variant)  
**Tested Quantization:** Q5_K_XL (importance-based)  
**Hardware:** 3× RTX 3060 12GB (36GB total VRAM)  
**Status:** ✅ Production-ready for multi-GPU inference

---

## Quick Facts

| Parameter | Value |
|-----------|-------|
| **Architecture** | qwen35 (hybrid recurrent + attention) |
| **Parameters** | 27 billion |
| **Layers** | 64 total (48 DeltaNet recurrent + 16 full-attention) |
| **Context Window** | 262,144 tokens |
| **Embedding Dimension** | 5120 |
| **Attention Heads** | 24 (4 KV heads, GQA 6:1) |
| **Vocabulary Size** | 248,320 |
| **Quantization Tested** | Q5_K_XL (importance-based, Q5_K→Q6_K gradient) |
| **Model Size (Q5_K_XL)** | ~19.2 GiB on disk, ~18.4 GiB on GPU |
| **VRAM Requirement** | 19GB minimum (single GPU), 36GB tested (3×12GB) |

### Architecture Details

Qwen3.5-27B uses a **hybrid architecture**:
- **48 recurrent layers** (75%): Gated DeltaNet (linear attention) with SSM components
- **16 full-attention layers** (25%): Standard transformer attention at positions 3, 7, 11, 15, 19, ..., 63 (every 4th layer)

This design significantly reduces KV cache requirements compared to pure transformer models.

---

## Performance

All benchmarks performed on 3× RTX 3060 12GB with optimal configuration (see Multi-GPU Configuration below).

### Prompt Processing (PP) Speed

| Context Depth | PP tok/s | Notes |
|---------------|----------|-------|
| **32K tokens** | **441** | Excellent for medium-context workloads |
| **128K tokens** | **355** | Still very fast at extended context |
| **250K tokens** | **270** | Solid performance at near-max context |

**Key insight:** Performance degrades gracefully with context depth due to flash-attention optimization and efficient KV cache utilization.

### Text Generation (TG) Speed

| Context State | TG tok/s | Notes |
|---------------|----------|-------|
| **Fresh context (0K)** | **60-62** | Real serving speed with 131K allocated context |
| **Mid-context (14K)** | **85** | Improved speed with warmed KV cache |
| **Low parallel usage** | **60-84** | Typical range for interactive serving |

**Note:** `llama-bench` reports ~96 tok/s, but real serving with allocated KV cache shows 60-62 tok/s at low context. This is expected overhead for production configurations with large context windows.

---

## KV Cache Characteristics

One of Qwen3.5-27B's major advantages is its **extremely efficient KV cache usage**:

| Metric | Value | Explanation |
|--------|-------|-------------|
| **Layers using KV cache** | **16/64 (25%)** | Only full-attention layers; recurrent layers use fixed-size state |
| **KV cache per token** | **~12 KB** | With `q8_0` K-cache / `q4_0` V-cache |
| **Total KV @ 262K ctx** | **~3.1 GB** | Compare to ~10.8 GB for equivalent 64-layer transformer |
| **KV cache per layer** | **416 MiB** | Per full-attention layer at max context |

### Why This Matters

A traditional 27B transformer with 64 attention layers would require **~10.8 GB** of KV cache at 262K context. Qwen3.5's hybrid architecture reduces this by **~72%**, making it significantly more VRAM-efficient for long-context workloads.

**Distribution of KV cache across GPUs** depends on how many full-attention layers land on each GPU (controlled by `--tensor-split`).

---

## Multi-GPU Configuration

### Production Config (3× RTX 3060 12GB)

```bash
llama-server \
  --model Qwen3.5-27B-UD-Q5_K_XL.gguf \
  --parallel 3 \
  --ctx-size 393216 \
  --tensor-split 0.30,0.37,0.33 \
  --split-mode layer \
  --gpu-layers 99 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q4_0 \
  --jinja \
  --metrics
```

**Configuration explanation:**
- **`--parallel 3`**: 3 concurrent slots × 131K context each = 393K total
- **`--ctx-size 393216`**: Total context pool across all slots
- **`--tensor-split 0.30,0.37,0.33`**: Carefully balanced to account for `output.weight` on last GPU
- **`--split-mode layer`**: Mandatory for multi-GPU without NVLink
- **`--cache-type-k q8_0 --cache-type-v q4_0`**: Recommended KV cache quantization (minimal quality loss)

### VRAM Allocation (Production Config)

| GPU | Model Layers | Model Weight | KV Cache | Compute Buffer | Extra | Total Used | Free |
|-----|--------------|-------------|----------|----------------|-------|------------|------|
| **GPU0** | 0-19 (20) | 5,318 MiB | 2,080 MiB | 2,496 MiB | 884 MiB (mmproj) | ~11,600 MiB | **688 MiB** |
| **GPU1** | 20-43 (24) | 6,385 MiB | 2,496 MiB | 1,212 MiB | — | ~11,659 MiB | **629 MiB** |
| **GPU2** | 44-63 (20) + output | 6,690 MiB | 2,080 MiB | 1,559 MiB | — | ~11,445 MiB | **843 MiB** |

**Key observations:**
- All GPUs have **600-850 MiB free** — safe operating margin
- GPU2 includes the **output.weight tensor** (~1,060 MiB, hardcoded to last GPU in llama.cpp)
- mmproj (multimodal projection) always lands on GPU0
- Compute buffers are determined by model graph structure, not tensor-split ratios

---

## Capacity Matrix

Different parallelism configurations for 3× RTX 3060 12GB setup:

| Config | Slots | Context per Slot | Total Context | Min VRAM Free | Status | Use Case |
|--------|-------|------------------|---------------|---------------|--------|----------|
| **parallel 2, ctx 262144** | 2 | 131K | 262K | ~1,796 MiB | ✅ Very safe | Low concurrency, max headroom |
| **parallel 3, ctx 393216** | 3 | 131K | 393K | **629 MiB** | ✅ **Production** | Balanced concurrency + context |
| **parallel 3, ctx 480000** | 3 | 160K | 480K | 199 MiB | ⚠️ Too tight | Not recommended |
| **parallel 4, ctx 480000** | 4 | 120K | 480K | 189 MiB | ⚠️ Too tight | Risky margin |
| **parallel 4, ctx 524288** | 4 | 131K | 524K | — | ❌ SEGV | Exceeds capacity |

**Recommendation:** Use `parallel 3` with `ctx-size 393216` for optimal balance of concurrency and stability.

### Scaling `--parallel`

Increasing `--parallel` has two effects:
1. **Reduces compute buffers** (inversely proportional) — more slots = smaller per-slot buffers
2. **Pre-allocates more KV cache** — each slot reserves its share of total context

At `parallel 4`, compute buffers shrink enough to theoretically fit, but CUDA overhead and flash-attention scratch space push it over the edge.

---

## Known Issues

### 1. Output Weight Placement

**Issue:** The `output.weight` tensor (~1,060 MiB for this model) is **hardcoded to the last GPU** in llama.cpp's split-mode layer implementation (`llama-model.cpp:2762`).

**Impact:** When using `--tensor-split`, you cannot distribute this tensor. The last GPU always bears this extra load.

**Solution:** Reduce the last GPU's layer share to compensate. For 3× GPUs with equal capacity, a split like `0.30,0.37,0.33` balances the load better than an even `0.33,0.33,0.34`.

### 2. Flash-Attention CUDA OOM at Tight Margins

**Issue:** At VRAM margins <200 MiB, flash-attention scratch space allocation can trigger CUDA OOM errors even though static VRAM appears sufficient.

**Symptoms:**
- Server crashes with `SIGABRT` during request processing
- Error logs show `flash_attn_ext` allocation failure
- Occurs at specific context depths (~130K in our testing)

**Solution:** Maintain **>500 MiB free VRAM** per GPU as safety margin. Use the `parallel 3 × 131K` configuration, not tighter variants.

### 3. Layer Boundary Discretization

**Issue:** Small changes to tensor-split ratios (e.g., 0.34 → 0.35) may produce **identical layer allocations** due to discrete layer boundaries.

**Example:** Both `0.28,0.38,0.34` and `0.27,0.38,0.35` produce the same GPU2 allocation in our testing.

**Impact:** You cannot fine-tune VRAM distribution with arbitrary precision. Changes must cross a layer boundary to take effect.

**Solution:** When optimizing tensor-split, test with `nvidia-smi` to verify actual allocations, not just theoretical calculations.

---

## KV Cache Quantization

### Recommended Configuration

```bash
--cache-type-k q8_0 --cache-type-v q4_0
```

**Rationale:**
- **K-cache is 7× more sensitive** to quantization than V-cache (empirically measured)
- `q8_0` for K-cache: virtually no quality degradation
- `q4_0` for V-cache: minor quality impact, significant VRAM savings
- This combination saves ~1.4 GB VRAM vs f16 with minimal perplexity increase

### Alternative Configurations

| Config | VRAM Savings | Quality Impact | Recommendation |
|--------|-------------|----------------|----------------|
| `k:f16 / v:f16` | 0 GB (baseline) | None | Only if VRAM abundant |
| `k:q8_0 / v:q8_0` | ~0.7 GB | Negligible | Safe alternative |
| **`k:q8_0 / v:q4_0`** | **~1.0 GB** | **Very small** | **✅ Recommended** |
| `k:q4_0 / v:q4_0` | ~1.4 GB | Moderate | Test for your use case |
| `k:q8_0 / v:q3_0` | ~1.1 GB | Small (V tolerant) | ⚠️ Probably fine, untested |
| `k:q3_0 / v:q3_0` | ~1.5 GB | **Significant (K degradation)** | ❌ Not recommended |

**Do NOT use `q3_0` for K-cache.** K projections have higher information density than V projections, and aggressive quantization causes meaningful retrieval degradation.

### Special Consideration for Hybrid Architecture

Since only **16/64 layers** use KV cache in Qwen3.5-27B, the absolute VRAM savings from aggressive KV quantization are **proportionally smaller** than in pure transformer models:
- **Pure transformer equivalent:** ~10.8 GB KV cache → q4_0 saves ~8 GB
- **Qwen3.5-27B:** ~3.1 GB KV cache → q4_0 saves ~2.3 GB

The quality requirements remain the same (K-cache still needs precision), but the VRAM ROI is reduced.

---

## Multi-GPU Topology Considerations

### PCIe Lane Distribution

On our test system (typical consumer motherboard):
- **GPU0:** PCIe x16 from CPU (full bandwidth)
- **GPU1:** PCIe x4 from chipset (reduced bandwidth)
- **GPU2:** PCIe x4 from chipset (reduced bandwidth)

**Impact on tensor-split:**
- PCIe bandwidth affects **prompt processing throughput** (layer-to-layer transfers)
- Does **NOT affect VRAM allocation** or model fit
- GPU0 can handle slightly more compute load without becoming a bottleneck

### Split-Mode Requirements

```bash
--split-mode layer  # MANDATORY for multi-GPU without NVLink
```

**Never use `--split-mode row`** on PCIe-only multi-GPU setups — it requires constant GPU-to-GPU communication and kills throughput. Layer-split keeps each layer on one GPU, minimizing cross-GPU traffic.

---

## Changelog

### 2026-03-08: Production Configuration Established
- Tested parallel 2/3/4 configurations extensively
- Established `parallel 3 × 131K` as optimal balance
- Discovered `output.weight` placement gotcha
- Validated KV cache quantization (`q8_0`/`q4_0`)
- Identified flash-attention OOM threshold (<500 MiB free)

---

## Related Documentation

- **General tensor-split optimization:** See `docs/multi-gpu-tensor-split.md`
- **Hardware profile:** See `docs/hardware/triple-3060.md`
- **Model inventory:** See `models/inventory-gpu.md`
