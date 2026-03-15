# Hardware Profile: 3× RTX 3060 12GB

**Configuration:** Triple GPU consumer setup  
**Total VRAM:** 36 GB (3 × 12,288 MiB)  
**Use Case:** Multi-GPU LLM inference with llama.cpp and vLLM  
**Status:** Production-validated for models up to ~27B dense / ~35B MoE

---

## Hardware Specifications

### GPU Configuration

| Slot | GPU Model | VRAM | PCIe Connection | Bandwidth | Notes |
|------|-----------|------|-----------------|-----------|-------|
| **GPU0** | RTX 3060 12GB | 12,288 MiB | x16 from CPU | Full | Primary compute GPU |
| **GPU1** | RTX 3060 12GB | 12,288 MiB | x4 from chipset | Reduced | Secondary |
| **GPU2** | RTX 3060 12GB | 12,288 MiB | x4 from chipset | Reduced | Secondary |
| *(slot 4)* | *(empty)* | — | x1 Gen 3 available | — | Future expansion possible |

**Total VRAM:** 36,864 MiB (36 GB)

### PCIe Topology

```
CPU (PCIe lanes)
 └─ GPU0 [PCIe x16]  ← Full bandwidth from CPU

Chipset (PCIe lanes)
 ├─ GPU1 [PCIe x4]   ← Shared chipset bandwidth
 └─ GPU2 [PCIe x4]   ← Shared chipset bandwidth
```

**Implications:**
- **GPU0:** Best for compute-heavy layers (input processing, first layers)
- **GPU1/GPU2:** Equal priority, slightly slower inter-GPU transfers during prompt processing
- **No NVLink:** Must use `--split-mode layer` (row-split unusably slow on PCIe)

### System Context

- **Host Type:** LXC container on Proxmox VE
- **GPU variant:** GA106 (Lite Hash Rate), compute capability 8.6
- **CUDA Toolkit:** 12.5.1
- **Driver:** 580.105.08
- **PCIe:** Gen 3, x16/x4/x4 (idle power saving drops to Gen 1)
- **Cooling:** Open case (no side panel), standard fans — GPU temps 37–49°C at idle

---

## Validated Model Configurations

### Qwen3.5-27B Q5_K_XL (Production)

**Model Stats:**
- Parameters: 27B (dense, hybrid recurrent + attention)
- Quant: Q5_K_XL
- Model size: ~19.2 GiB
- Architecture: 48 DeltaNet + 16 full-attention layers

**Configuration:**
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
  --cache-type-v q4_0
```

**VRAM Allocation:**

| Component | GPU0 | GPU1 | GPU2 | Notes |
|-----------|------|------|------|-------|
| **Model layers** | 5,318 MiB | 6,385 MiB | 5,630 MiB | 20/24/20 layers |
| **output.weight** | — | — | 1,060 MiB | Hardcoded to last GPU |
| **KV cache** | 2,080 MiB | 2,496 MiB | 2,080 MiB | 5/6/5 full-attn layers |
| **Compute buffers** | 2,496 MiB | 1,212 MiB | 1,559 MiB | Graph-determined |
| **mmproj** | 884 MiB | — | — | Vision encoder (GPU0 only) |
| **CUDA overhead** | ~380 MiB | ~380 MiB | ~380 MiB | Runtime, fragmentation |
| **Total used** | ~11,600 MiB | ~11,659 MiB | ~11,445 MiB | |
| **Free VRAM** | **688 MiB** | **629 MiB** | **843 MiB** | Safe margins |

**Capacity:** 3 concurrent slots × 131K context each = 393K total  
**Status:** ✅ Stable under load, validated to 194K actual tokens per slot

**Performance:**
- **PP:** 441 tok/s @32K, 355 tok/s @128K, 270 tok/s @250K
- **TG:** 60-62 tok/s (fresh context), 85 tok/s (warm)

### Qwen3.5-35B-A3B Q4_K_M (Previous Config)

**Model Stats:**
- Parameters: 35B (MoE, 30B active)
- Quant: Q4_K_M
- Model size: ~23 GiB
- Architecture: 64 layers, 3 experts, all active

**Configuration:**
```bash
llama-server \
  --model Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --parallel 1 \
  --ctx-size 262144 \
  --tensor-split 0.28,0.36,0.36 \
  --split-mode layer \
  --gpu-layers 99 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q4_0
```

**VRAM Allocation:**

| Component | GPU0 | GPU1 | GPU2 |
|-----------|------|------|------|
| **Model layers** | ~7.2 GB | ~7.8 GB | ~8.0 GB |
| **KV cache** | ~1.7 GB | ~2.5 GB | ~2.5 GB |
| **Compute buffers** | ~2.5 GB | ~1.2 GB | ~1.6 GB |
| **Total used** | ~11.4 GB | ~11.5 GB | ~12.1 GB |
| **Free VRAM** | ~0.9 GB | ~0.8 GB | ~0.2 GB |

**Capacity:** 1 slot × 262K context  
**Status:** ⚠️ Tight fit, <500 MiB margin on GPU2 (replaced by 27B config)

**Notes:**
- This configuration was the production baseline before Qwen3.5-27B optimization
- Demonstrated that 24GB MoE models are at the capacity limit for this hardware
- Single-slot only (no headroom for `--parallel >1`)

### vLLM with Tensor Parallel (Validated)

vLLM works on this 3× GPU setup using tensor parallelism (TP=3), despite the asymmetric PCIe topology.

**Validated model:** Qwen3-30B-A3B-GPTQ-Int4

**Configuration:**
```bash
vllm serve Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --tensor-parallel-size 3 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92
```

**Notes:**
- TP=3 works across all 3 GPUs despite mixed x16/x4 PCIe lanes
- GPTQ-Int4 quantization required (FP16 is too large for 36 GB)
- Shorter context limits than llama.cpp (vLLM's memory management is less granular)
- FlashInfer JIT compilation on first request can take 60-90s (cold start)
- Suitable for GPTQ/AWQ quantized models where llama.cpp doesn't support the format
- Not currently used in production (llama.cpp is faster for our workloads), but available as an option

---

## Capacity Planning Matrix

### What Fits on 3× RTX 3060 12GB

| Model Size (on GPU) | Quant | Example Model | Fit? | Max Context | Parallel Slots | Notes |
|---------------------|-------|---------------|------|-------------|----------------|-------|
| **~13 GB** | Q5_K_M | LLaMA-2-13B | ✅ Excellent | 262K+ | 4-6 | Plenty of headroom |
| **~19 GB** | Q5_K_XL | Qwen3.5-27B (dense) | ✅ Good | 393K (3×131K) | 3 | Validated config |
| **~22 GB** | Q4_K_M | Nemotron-30B-A3B | ✅ Tight | 262K | 1-2 | <1GB headroom |
| **~24 GB** | Q4_K_M | Qwen3.5-35B-A3B | ⚠️ Very tight | 262K | 1 | <500MB margin |
| **~28 GB+** | Any | 70B+ models | ❌ No fit | — | — | Exceeds capacity |

**General guidelines:**
- **<19 GB model:** Comfortable fit, multi-slot serving viable
- **19-22 GB model:** Fits well, 2-3 slots possible with tuning
- **22-24 GB model:** Tight fit, single-slot only, <500 MiB margins
- **>24 GB model:** Does not fit (requires 4× GPUs or larger GPUs)

### Quantization Trade-offs

For a 27B parameter dense model:

| Quant | Model Size | Fit | TG Speed | Quality | Recommendation |
|-------|------------|-----|----------|---------|----------------|
| **Q2_K** | ~11 GB | ✅ Easy | Fastest | Poor | Avoid (quality too low) |
| **Q4_K_M** | ~15 GB | ✅ Comfortable | Fast | Good | Viable for speed-critical use |
| **Q5_K_M** | ~18 GB | ✅ Good | Medium-fast | Very good | Balanced choice |
| **Q5_K_XL** | ~19 GB | ✅ Tight | Medium | Excellent | Best quality, tight fit |
| **Q6_K** | ~22 GB | ⚠️ Very tight | Medium-slow | Near-f16 | Only if quality critical |
| **Q8_0** | ~28 GB | ❌ No fit | Slow | Near-f16 | Exceeds capacity |

**Recommendation:** Q5_K_M or Q5_K_XL for 27B models — best balance of quality and capacity.

---

## VRAM Budget Breakdown

Understanding where VRAM goes helps optimize configurations.

### Static Allocations (Model Load)

| Component | Size | Location | Notes |
|-----------|------|----------|-------|
| **Model weights** | ~18-24 GB | Distributed by tensor-split | Bulk of VRAM usage |
| **output.weight** | ~1-2 GB | Last GPU (hardcoded) | Additional load on GPU2 |
| **mmproj (if multimodal)** | ~0.9 GB | GPU0 (first GPU) | Vision encoder |
| **Token embeddings** | — | CPU (not GPU) | Minimal VRAM impact |

### Dynamic Allocations (Context-Dependent)

| Component | Formula | Notes |
|-----------|---------|-------|
| **KV cache** | `(n_layers_on_gpu × ctx_size × kv_bytes_per_token)` | Pre-allocated at server start |
| **Compute buffers** | Graph-determined | GPU0: ~2.5 GB, others: ~1.2-1.6 GB |
| **Flash-attn scratch** | Allocated on-demand | Requires 200-500 MiB extra per GPU |

### Example: Qwen3.5-27B Q5_K_XL @ 393K Context (3 Slots)

| Component | GPU0 | GPU1 | GPU2 | Total |
|-----------|------|------|------|-------|
| Model weights | 5,318 | 6,385 | 5,630 | 17,333 MiB |
| output.weight | — | — | 1,060 | 1,060 MiB |
| KV cache | 2,080 | 2,496 | 2,080 | 6,656 MiB |
| Compute buffers | 2,496 | 1,212 | 1,559 | 5,267 MiB |
| mmproj | 884 | — | — | 884 MiB |
| CUDA overhead | ~380 | ~380 | ~380 | ~1,140 MiB |
| **Total** | **~11,600** | **~11,659** | **~11,445** | **~32,500 MiB** |
| **Free** | **688** | **629** | **843** | **~4,364 MiB** |

**Utilization:** 88.2% (32.5 GB / 36.9 GB)  
**Headroom:** 11.8% (~4.4 GB distributed)

---

## Optimization Strategies

### 1. Tensor-Split Balancing

**Goal:** Distribute VRAM load evenly across all GPUs.

**Key insight:** The last GPU (GPU2) bears an extra ~1-2 GB from `output.weight`. Compensate by reducing its layer share.

**Example progression:**
```bash
# Naive even split → GPU2 overloaded:
--tensor-split 0.33,0.33,0.34
# GPU0: 1,819 free, GPU1: 1,984 free, GPU2: 721 free (BOTTLENECK)

# Balanced split → even headroom:
--tensor-split 0.30,0.37,0.33
# GPU0: 688 free, GPU1: 629 free, GPU2: 843 free (BALANCED)
```

**Methodology:** See `docs/multi-gpu-tensor-split.md` for detailed optimization workflow.

### 2. KV Cache Quantization

**Default:** `--cache-type-k q8_0 --cache-type-v q4_0`

**Savings:** ~1.0 GB vs f16 KV cache (minimal quality loss)

**Alternatives:**
- `k:f16 / v:f16` — No savings, maximum quality
- `k:q8_0 / v:q8_0` — ~0.7 GB savings, negligible quality impact
- `k:q4_0 / v:q4_0` — ~1.4 GB savings, moderate quality impact (test your use case)

**Do NOT use `q3_0` for K-cache** — K projections are sensitive to quantization, degrades retrieval quality significantly.

### 3. Parallel Slot Tuning

Trade concurrency for total context capacity:

| Parallel | Compute Buffers | Total KV | Free VRAM | Use Case |
|----------|-----------------|----------|-----------|----------|
| **2** | Larger (~1.3-2.5 GB) | Moderate | ~1.8 GB | Max single-context depth |
| **3** | Medium (~1.0-2.5 GB) | High | ~0.6 GB | Balanced (recommended) |
| **4** | Small (~0.8-2.5 GB) | Very high | <0.2 GB | Max concurrency (risky) |

**Recommendation:** `--parallel 3` for Qwen3.5-27B — good balance of concurrency and stability.

### 4. Model Selection

**For this hardware, prioritize:**
1. **Hybrid architectures** (Qwen3.5, GLM-4, models with DeltaNet/Mamba layers) — lower KV cache overhead
2. **Importance-based quants** (Q5_K_XL, Q4_K_XL) — better quality at lower sizes
3. **Dense models over MoE** if <30B params — MoE only wins at larger scales

**Avoid:**
- Pure transformer 27B+ models (KV cache too large)
- Q6_K / Q8_0 quants (exceed capacity without quality benefit)
- Models >24 GB on GPU (no headroom for safe operation)

---

## Limitations and Gotchas

### 1. No NVLink

**Impact:** Must use `--split-mode layer`. Row-split reduces throughput by 10-50×.

**Workaround:** None — this is a hardware limitation. Layer-split is the only viable mode.

### 2. Asymmetric PCIe Bandwidth

**Impact:** GPU1 and GPU2 are slower than GPU0 for prompt processing due to x4 PCIe lanes.

**Mitigation:** Place more compute-heavy layers on GPU0 if prompt processing speed is critical. For text generation (TG), bandwidth matters less.

### 3. CUDA Overhead and Fragmentation

**Observation:** `nvidia-smi` reports ~300-400 MiB "missing" VRAM per GPU beyond accounted allocations.

**Cause:** CUDA runtime overhead, memory fragmentation, driver buffers.

**Implication:** Budget ~400 MiB per GPU for overhead when capacity planning.

### 4. Flash-Attention Scratch Space

**Problem:** Even with sufficient "free" VRAM shown by `nvidia-smi`, flash-attention may trigger CUDA OOM during inference.

**Cause:** Scratch space is allocated on-demand, not at model load.

**Solution:** Maintain >500 MiB free VRAM per GPU as safety margin. Test under actual load, not just idle state.

### 5. 4th GPU Slot Constraint

**Current:** 3× RTX 3060 = 36 GB  
**Potential:** 4× RTX 3060 = 48 GB

**Limitation:** The 4th slot is PCIe x1 on this motherboard — likely insufficient bandwidth for LLM inference, though untested. Even the x4 slots already show bottlenecks at high context loads.

**Status:** Impractical without motherboard upgrade. Current system maxes out at 3 usable GPUs.

---

## Comparison to Alternatives

### vs Single RTX 3090 24GB

| Metric | 3× RTX 3060 (36GB) | 1× RTX 3090 (24GB) |
|--------|--------------------|--------------------|
| **Total VRAM** | 36 GB | 24 GB |
| **Max model size** | ~24 GB | ~20 GB |
| **TG speed** | 60-85 tok/s | 80-120 tok/s (faster per GPU) |
| **PP speed** | 270-441 tok/s | 500-800 tok/s (better bandwidth) |
| **Cost** | ~$900 (3× $300) | ~$1,200 (used market) |
| **Power draw** | ~510W (3× 170W) | ~350W |

**Trade-off:** 3× 3060 provides more total VRAM and better cost-per-GB, but single 3090 is faster per operation and more power-efficient.

### vs 2× RTX 4060 Ti 16GB

| Metric | 3× RTX 3060 (36GB) | 2× RTX 4060 Ti (32GB) |
|--------|--------------------|-----------------------|
| **Total VRAM** | 36 GB | 32 GB |
| **Architecture** | Ampere (3000 series) | Ada (4000 series) |
| **TG speed** | 60-85 tok/s | ~100-130 tok/s (newer arch) |
| **PP speed** | 270-441 tok/s | ~600-900 tok/s (better bandwidth) |
| **Cost** | ~$900 | ~$1,000 (2× $500) |

**Trade-off:** 4060 Ti is faster but has 4 GB less total VRAM. Better for speed-critical workloads; 3× 3060 better for fitting larger models.

---

## Future Expansion Considerations

### Adding a 4th GPU

**Current limitation:** Slot 4 is PCIe x1 (insufficient bandwidth).

**Requirements for 4th GPU:**
- Motherboard with 4× usable PCIe slots (x4 or better)
- PSU with 4× PCIe power connectors (current: be quiet! 700W — likely sufficient)
- Case with adequate cooling for 4× GPUs (monitor temps)

**Benefit:** 48 GB total VRAM → fits 70B Q4_K_M models or 30B Q6_K models comfortably.

### Upgrading Individual GPUs

**Most realistic option:** Replace one or more RTX 3060 12GB with an RTX 3090 24GB.

- 1× 3090 + 2× 3060 = 48 GB (mixed config, needs tensor-split adjustment)
- 1× 3090 alone = 24 GB (simpler, faster per-token, less total VRAM)
- Austrian used market: €700-900 for a 3090 (prices fluctuate — watch willhaben/eBay Kleinanzeigen for deals)

**Other options:**
- **RTX 4090 24GB** — faster architecture, same 24 GB VRAM, ~€1,800+
- **A6000 48GB** — workstation card, 48 GB in one slot, ~€3,500+ (used ~€2,000)

**Trade-off:** A single large GPU is simpler (no tensor-split), but multi-GPU gives more total VRAM per euro.

---

## Recommended Workflows

### For Production Serving

**Configuration:**
```bash
--parallel 3 --ctx-size 393216 --tensor-split 0.30,0.37,0.33
```

**Model:** Qwen3.5-27B Q5_K_XL or similar ~19GB model  
**Capacity:** 3 concurrent users × 131K context each  
**Stability:** ✅ Validated under sustained load

### For Experimentation / Model Testing

**Configuration:**
```bash
--parallel 1 --ctx-size 262144 --tensor-split 0.30,0.37,0.33
```

**Benefit:** Maximum headroom (~3-4 GB free), can test models up to 24 GB  
**Use case:** Evaluating new models, running benchmarks, debugging

### For Maximum Context Depth

**Configuration:**
```bash
--parallel 2 --ctx-size 262144 --tensor-split 0.30,0.37,0.33
```

**Benefit:** Reduced compute buffers = more VRAM for context  
**Use case:** Single-user scenarios requiring deepest possible context

---

## Changelog

### 2026-03-08: Qwen3.5-27B Optimization
- Established `--tensor-split 0.30,0.37,0.33` as optimal balance
- Validated 3-slot serving (parallel 3 × 131K context)
- Documented output.weight hardcoding gotcha
- Tested ceiling for parallel 2/3/4 configurations

### 2026-02-XX: Qwen3.5-35B-A3B Baseline
- Demonstrated 24 GB MoE fit with `--parallel 1`
- Identified <500 MiB margins as OOM risk threshold

---

## Related Documentation

- **Tensor-split optimization guide:** `docs/multi-gpu-tensor-split.md`
- **Qwen3.5-27B model report:** `models/qwen3.5-27b.md`
- **GPU model inventory:** `models/inventory-gpu.md`
