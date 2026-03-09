# Multi-GPU Tensor-Split Optimization Guide

**Scope:** General methodology for optimizing `--tensor-split` on multi-GPU llama.cpp setups  
**Platform:** llama.cpp with `--split-mode layer`  
**Target:** Consumer multi-GPU systems without NVLink (PCIe only)

---

## What Is Tensor-Split?

The `--tensor-split` flag in llama.cpp controls how model layers are distributed across GPUs. You provide a **ratio vector** that determines the proportion of layers assigned to each GPU.

**Example:**
```bash
llama-server --tensor-split 0.30,0.37,0.33 --gpu-layers 99
```

This distributes layers across 3 GPUs in a 30:37:33 ratio.

### What Gets Distributed

When you enable GPU offloading (`--gpu-layers N`), llama.cpp splits:
- ✅ **Model layer weights** (attention, FFN, norms) — distributed by tensor-split
- ✅ **KV cache** — distributed based on which layers land on which GPU
- ✅ **Intermediate activations** — follow the layer that produces them

### What Does NOT Get Distributed

Several components have **fixed placement** regardless of tensor-split ratios:
- ❌ **Token embeddings** (`token_embd.weight`) — always stays on **CPU** (minimal benefit to offload)
- ❌ **Output projection** (`output.weight`, lm_head) — **hardcoded to last GPU** (see gotcha #1 below)
- ❌ **Multimodal projections** (mmproj) — lands on **first GPU** (GPU0)
- ❌ **Compute buffers** — size determined by model graph structure, not ratios (see gotcha #2)

---

## Key Gotchas

### 1. Output Weight Hardcoded to Last GPU

**Source:** `llama-model.cpp:2762`
```cpp
// assign the output layer
pimpl->dev_output = get_layer_buft_list(n_layer);
```

The output projection tensor (also called `lm_head` or `output.weight`) is **always placed on the last GPU** in split-mode layer, regardless of your tensor-split ratios.

**Size:** Approximately `model_size / n_layers` (often 1-2 GB for large models)

**Impact:**
- The last GPU bears an **extra 1-2 GB load** compared to middle GPUs
- An "equal" split like `--tensor-split 0.33,0.33,0.34` will leave the last GPU with **less free VRAM** than the others
- You must **compensate** by reducing the last GPU's layer share

**Example (Qwen3.5-27B on 3× 12GB GPUs):**
- Model size: ~19.2 GB, 64 layers → output.weight ≈ 1,060 MiB
- Even split `0.33,0.33,0.34` → GPU2 gets 22 layers + output = ~7.1 GB total
- Adjusted split `0.30,0.37,0.33` → GPU2 gets 20 layers + output = ~6.7 GB total
- **Result:** GPU2 free VRAM improves from 237 MiB → 1,465 MiB

### 2. Compute Buffers Are Fixed by Model Graph

**Misconception:** "If I give GPU0 30% of layers, it should use 30% of VRAM."

**Reality:** Compute buffers for intermediate activations are determined by the **model's computation graph**, not by tensor-split ratios.

**Typical pattern:**
- **GPU0:** Largest compute buffer (handles input processing)
- **GPU1-N:** Smaller, relatively uniform buffers

**Example (3× GPU system):**
- GPU0: 2,496 MiB compute buffer
- GPU1: 1,212 MiB compute buffer
- GPU2: 1,559 MiB compute buffer

These values are **constant** regardless of whether you use `--tensor-split 0.28,0.36,0.36` or `0.30,0.37,0.33`.

**How `--parallel` affects buffers:**
- Compute buffers scale **inversely** with `--parallel N`
- `--parallel 2` roughly halves buffer sizes
- `--parallel 4` roughly quarters them
- This allows you to trade concurrency for total context capacity

### 3. Layer Boundary Discretization

Tensor-split ratios map to **discrete layer boundaries**. Small ratio changes may not move any layers.

**Example:**
```bash
# Both produce identical GPU2 allocation:
--tensor-split 0.27,0.38,0.35  # 64 layers / 0.35 = layer 44 cutoff
--tensor-split 0.28,0.38,0.34  # Still layer 44 cutoff
```

The mapping is:
```
position = layer_index / (n_layers + 1)
assigned_gpu = first GPU where cumsum(tensor_split) > position
```

Since this is discrete, ratios that don't cross a layer boundary produce identical allocations.

**Implication:** You cannot fine-tune VRAM balance with arbitrary precision. Changes must move at least one layer to have any effect.

---

## Ceiling Testing Methodology

How to find the maximum usable context for your specific model + hardware configuration.

### Step 1: Start with Even Split

Begin with an even distribution:
```bash
# 3 GPUs:
--tensor-split 0.33,0.33,0.34

# 2 GPUs:
--tensor-split 0.5,0.5
```

Load the model with a moderately large context (e.g., `--ctx-size 131072`) and check VRAM allocation:
```bash
nvidia-smi
```

### Step 2: Identify the Bottleneck GPU

Look for the GPU with the **least free VRAM** at idle (model loaded, no requests active).

**Example output:**
```
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|      0  RTX 3060         Off  | 00000000:01:00.0 Off |                  N/A |
|      Memory Usage: 10469MiB / 12288MiB                                      |
|      1  RTX 3060         Off  | 00000000:02:00.0 Off |                  N/A |
|      Memory Usage: 10304MiB / 12288MiB                                      |
|      2  RTX 3060         Off  | 00000000:03:00.0 Off |                  N/A |
|      Memory Usage: 11567MiB / 12288MiB    <-- BOTTLENECK (only 721 MiB free) |
+-----------------------------------------------------------------------------+
```

In this example, **GPU2 is the bottleneck** with only 721 MiB free.

### Step 3: Adjust Ratios

**Strategy:** Give less to the bottleneck GPU, more to others.

```bash
# Before: 0.33,0.33,0.34 (even split)
# After:  0.30,0.37,0.33 (shift load from GPU2 to GPU1)
```

**Guidelines:**
- Shift in increments of 0.03-0.05 initially (maps to ~2-3 layers on a 64-layer model)
- Remember: last GPU bears the output.weight penalty (~1-2 GB)
- Target: all GPUs have roughly equal free VRAM

### Step 4: Verify with Actual Prompt Fill

**Critical:** KV cache is pre-allocated at model load, but **flash-attention scratch space is allocated on-demand** during request processing.

Even if `nvidia-smi` shows sufficient free VRAM at idle, you may hit CUDA OOM during actual inference.

**Test procedure:**
1. Load the model with your new tensor-split
2. Send a request that fills the context to your target depth (e.g., 128K tokens)
3. Monitor for CUDA OOM errors in server logs
4. If OOM occurs, reduce context size or adjust split further

**Safe margin:** Maintain **>500 MiB free VRAM** per GPU after model load to avoid flash-attention OOM.

### Step 5: Iterate

Repeat steps 2-4 until:
- All GPUs have roughly balanced free VRAM (within ~300 MiB of each other)
- No CUDA OOM errors occur during actual inference at target context depth
- Worst-case free VRAM is >500 MiB

---

## Effect of `--parallel N`

The `--parallel` flag controls how many concurrent inference slots the server maintains. It has a **major impact on VRAM allocation**.

### How It Works

Each slot gets an equal share of the total context:
```bash
--ctx-size 393216 --parallel 3
# → 3 slots × 131,072 tokens each
```

**VRAM components affected:**
1. **KV cache:** Scales linearly with `--parallel` (more slots = more KV cache)
2. **Compute buffers:** Scale **inversely** with `--parallel` (more slots = smaller per-slot buffers)

### Example: Qwen3.5-27B on 3× RTX 3060

| Config | Slots | Ctx/Slot | GPU0 Buffer | GPU1 Buffer | GPU2 Buffer | Min Free VRAM |
|--------|-------|----------|-------------|-------------|-------------|---------------|
| `parallel 2, ctx 262144` | 2 | 131K | 1,348 MiB | 711 MiB | 1,047 MiB | ~1,796 MiB |
| `parallel 3, ctx 393216` | 3 | 131K | 1,070 MiB | 574 MiB | 844 MiB | ~629 MiB |
| `parallel 4, ctx 524288` | 4 | 131K | ~800 MiB | ~430 MiB | ~633 MiB | SEGV (OOM) |

**Trade-off:**
- **Higher `--parallel`:** More concurrency, smaller buffers, **less total context capacity**
- **Lower `--parallel`:** Less concurrency, larger buffers, **more total context capacity**

### Finding the Sweet Spot

For a given model + GPU setup:
1. Start with `--parallel 2` to maximize headroom
2. Incrementally increase `--parallel` while monitoring VRAM
3. Stop when minimum free VRAM drops below 500 MiB
4. Back off by 1 for safety margin

---

## PCIe Topology Considerations

### Asymmetric Lane Distributions

Typical consumer motherboard:
- **GPU0:** PCIe x16 from CPU (slot 1)
- **GPU1:** PCIe x4 from chipset (slot 2)
- **GPU2:** PCIe x4 from chipset (slot 3)

### Impact on Tensor-Split

**VRAM allocation:** PCIe lane width has **NO impact** on how VRAM is distributed. A GPU in an x4 slot can hold the same model layers as one in an x16 slot.

**Prompt processing throughput:** GPUs in x4 slots will be **slower at layer-to-layer transfers** during prompt ingestion (PP). This affects speed but not capacity.

**Recommendation:** If you must distribute unevenly due to the output.weight gotcha, prefer placing **more layers on the x16 GPU** (GPU0) since it has better bandwidth for intermediate activations.

---

## Split-Mode: Layer vs Row

```bash
--split-mode layer  # One layer = one GPU (minimal cross-GPU traffic)
--split-mode row    # Tensors split across GPUs (requires constant communication)
```

### For Multi-GPU WITHOUT NVLink

**Use `--split-mode layer`** exclusively.

**Why:**
- Row-split requires frequent GPU-to-GPU tensor transfers
- PCIe bandwidth is ~10-100× slower than NVLink
- Row-split on PCIe can reduce throughput by **10-50×**

**Benchmark (2-GPU PCIe x16 + x4, same model):**
- `-sm layer`: 60-62 tok/s text generation
- `-sm row`: 2-8 tok/s text generation (unusably slow)

### For Multi-GPU WITH NVLink

Row-split becomes viable if you have:
- NVLink bridges connecting GPUs (V100, A100, H100)
- High-bandwidth GPU-to-GPU interconnect (>100 GB/s)

Even then, layer-split is often still faster due to reduced communication overhead.

---

## Practical Examples

### Example 1: 2× RTX 3060 12GB

**Model:** 19B parameter, Q5_K_M (~13 GB)  
**Goal:** Maximize context window  
**Initial attempt:** `--tensor-split 0.5,0.5`

```
nvidia-smi output:
GPU0: 10,024 MiB used, 2,264 MiB free
GPU1: 10,856 MiB used, 1,432 MiB free  <-- bottleneck (output.weight here)
```

**Adjustment:** Shift load to GPU0: `--tensor-split 0.55,0.45`

```
GPU0: 10,692 MiB used, 1,596 MiB free
GPU1: 10,188 MiB used, 2,100 MiB free
```

**Result:** Worst-case free VRAM improved from 1,432 → 1,596 MiB (+164 MiB headroom).

### Example 2: 3× RTX 3060 12GB

**Model:** Qwen3.5-27B Q5_K_XL (~19.2 GB)  
**Goal:** Fit with multi-slot serving  
**Initial attempt:** `--tensor-split 0.33,0.33,0.34 --parallel 3 --ctx-size 393216`

```
nvidia-smi output:
GPU0: 10,469 MiB used, 1,819 MiB free
GPU1: 10,304 MiB used, 1,984 MiB free
GPU2: 11,567 MiB used,   721 MiB free  <-- CRITICAL (output.weight + heavy layers)
```

**Problem:** GPU2 only has 721 MiB free → flash-attention OOM at ~130K context during actual inference.

**Adjustment:** Redistribute to balance load: `--tensor-split 0.30,0.37,0.33`

```
GPU0: 11,600 MiB used,   688 MiB free
GPU1: 11,659 MiB used,   629 MiB free  <-- new bottleneck (acceptable)
GPU2: 11,445 MiB used,   843 MiB free
```

**Result:** Worst-case free VRAM improved from 721 → 629 MiB (more balanced), and GPU2 now has 843 MiB free. System is stable at full 131K context under load.

**Why 629 MiB became the bottleneck:** GPU1 gained layers from both GPU0 and GPU2, but it's acceptable because all GPUs are now within a safe margin.

### Example 3: 4× RTX 3060 12GB

**Model:** 35B parameter MoE, Q4_K_M (~24 GB)  
**Goal:** Maximize slots  
**Configuration:** `--tensor-split 0.25,0.25,0.25,0.25 --parallel 4`

```
Initial test:
GPU0: 9,200 MiB used
GPU1: 9,100 MiB used
GPU2: 9,100 MiB used
GPU3: 9,800 MiB used  <-- output.weight here
```

**Adjustment:** `--tensor-split 0.27,0.26,0.26,0.21`

```
Balanced result:
GPU0: 9,500 MiB used, 2,788 MiB free
GPU1: 9,300 MiB used, 2,988 MiB free
GPU2: 9,300 MiB used, 2,988 MiB free
GPU3: 9,200 MiB used, 3,088 MiB free
```

**Result:** Successfully fit 4-slot serving with >2.7 GB free per GPU.

---

## Troubleshooting

### "Model loads but crashes during inference"

**Symptom:** `nvidia-smi` shows plenty of free VRAM, but server crashes with CUDA OOM during request processing.

**Cause:** Flash-attention scratch space is allocated on-demand, not at model load time.

**Solution:**
- Maintain >500 MiB free VRAM per GPU at idle
- Reduce `--ctx-size` or `--parallel` if needed
- Test with actual context fill, not just model load

### "GPU utilization is uneven during PP"

**Symptom:** During prompt processing, some GPUs sit idle while others work.

**Cause:** PCIe bandwidth bottleneck on GPUs with fewer lanes (x4 vs x16).

**Solution:** This is expected with asymmetric PCIe topologies. It affects **speed** but not **capacity**. If PP speed is critical, consider placing more compute-heavy layers (early layers) on the x16 GPU.

### "Changing tensor-split doesn't affect VRAM"

**Symptom:** Adjusting ratios from `0.34` to `0.35` produces no visible change in `nvidia-smi`.

**Cause:** Layer boundary discretization — both ratios map to the same layer cutoff.

**Solution:** Make larger adjustments (0.03-0.05) to ensure you cross a layer boundary. Verify by checking how many layers landed on each GPU in the server startup log.

### "Last GPU always runs out of VRAM first"

**Symptom:** No matter what tensor-split you use, GPU N (last) is always the bottleneck.

**Cause:** Output.weight is hardcoded to the last GPU.

**Solution:** Significantly reduce the last GPU's share — often 5-10% less than an even split.

---

## Summary Checklist

When optimizing tensor-split for a new model:

- [ ] Start with even split for initial load
- [ ] Check `nvidia-smi` to identify bottleneck GPU
- [ ] Account for output.weight on last GPU (~1-2 GB penalty)
- [ ] Adjust tensor-split to balance free VRAM across all GPUs
- [ ] Test with actual context fill (not just idle load)
- [ ] Maintain >500 MiB free VRAM per GPU for flash-attention
- [ ] Use `--split-mode layer` for PCIe multi-GPU setups
- [ ] Consider `--parallel N` to trade concurrency for capacity
- [ ] Iterate until worst-case free VRAM is stable and safe

---

## References

- **llama.cpp source:** `llama-model.cpp:2762` (output.weight placement)
- **Split-mode documentation:** https://github.com/ggml-org/llama.cpp/blob/master/docs/backend.md
- **Model-specific example:** See `models/qwen3.5-27b.md` for detailed worked example
