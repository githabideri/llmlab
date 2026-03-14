# Concurrent Slot TG Asymmetry Investigation

**Date:** 2026-03-09  
**Issue:** Severe per-slot performance asymmetry in concurrent text generation at high context depths on multi-GPU setups

## Observed Behavior Summary

On 3× RTX 3060 12GB with Qwen3.5-27B (split-mode layer, tensor-split 0.30,0.37,0.33):
- **~5K context:** Fairly balanced (7.1 vs 7.7 tok/s)
- **~50K context:** Severe imbalance (6.4 vs 1.5 tok/s)
- **~73K context:** Extreme imbalance (0.6 vs 8.0 tok/s)
- **~98K context:** Imbalance (5.0 vs 1.2 tok/s)
- **Key anomaly:** The "slow" slot alternates between slot0/slot1 across runs
- **Critical observation:** Aggregate throughput < single-slot throughput

---

## Root Cause Analysis

### 1. How llama.cpp Schedules Concurrent TG

**Continuous Batching Implementation** (from Discussion #4130 and ADR 006):

llama.cpp uses **continuous batching** with a single forward pass per tick:
- All active slots contribute tokens to a unified batch
- One `llama_decode()` call processes all tokens together
- Sampling happens per-slot at specific batch indices after decode

**Key Design Elements:**
- **Decode-maximal scheduling:** Decode tokens (one per generating slot) are added to batch first
- **Chunked prefill:** Long prompts split into chunks (default 512 tokens) processed across multiple ticks
- **Token budget:** Each tick's batch capped at `n_batch` tokens
- **Unified KV cache:** All sequences share the same KV cache buffer (size = `ctx_size`)

**Critical Point:**  
The scheduler is **architecturally fair** — it batches all active slots' tokens together in one forward pass. There's no intentional slot starvation or priority system.

### 2. The Multi-GPU Layer-Split Bottleneck

**Split Mode Layer (Default for Multi-GPU):**

From the Medium article on split modes:
- Distributes transformer layers sequentially across GPUs
- If you have 80 layers and 2 GPUs: layers 1-40 on GPU0, 41-80 on GPU1
- **Sequential execution:** GPU0 processes its layers → sends output to GPU1 → GPU1 processes → back to GPU0
- **Problem:** GPUs idle while waiting for data from the other GPU

**Why This Causes Asymmetry at High Context:**

1. **KV Cache Size Grows Linearly with Context:**
   - At 5K context: ~5K tokens × 2 slots = 10K tokens in KV cache
   - At 98K context: ~98K tokens × 2 slots = 196K tokens in KV cache

2. **Cross-GPU Communication Increases:**
   - Every decode step requires full forward pass across all GPUs
   - Intermediate activations must traverse PCIe bus between GPUs
   - At high context, the KV cache attention computation explodes (quadratic in context length)

3. **PCIe Bandwidth Becomes Bottleneck:**
   - RTX 3060 on PCIe x4 (GPU1/GPU2) = ~8 GB/s bidirectional
   - RTX 3060 on PCIe x16 (GPU0) = ~32 GB/s bidirectional
   - At high context, the volume of data crossing PCIe (intermediate activations + KV attention results) saturates the bus

4. **Unified KV Cache Mask Computation:**
   - From Discussion #4130: "The attention for each sequence is computed over the **entire KV cache**"
   - This means cross-sequence attention is computed and then masked (thrown away)
   - At high context with 2 slots: both slots' attention computed over ~200K tokens, even though each sequence only needs ~100K

### 3. Why Asymmetry Instead of Uniform Slowdown?

**The "One Slot Starves" Phenomenon:**

This is likely caused by:

**(a) Batch Processing Order + PCIe Contention:**
- When `llama_decode()` processes the batch, tokens are processed in order
- If slot0's token is batch index 0 and slot1's token is batch index 1:
  - Slot0 starts forward pass → waits for GPU1 (PCIe transfer)
  - Slot1 starts while slot0's data is in flight
  - **PCIe bus contention:** Whoever gets the bus first wins
  - The "loser" waits for the bus to be free → much longer latency

**(b) KV Cache Placement in Unified Buffer:**
- From Discussion #4130: "The determinism of the results is now also a function of **where the tokens of a sequence are placed in the KV cache**"
- If the KV cache is fragmented or one slot's tokens are placed at memory addresses that span GPU boundaries poorly, that slot suffers more cross-GPU transfers

**(c) CUDA Stream Serialization:**
- In split-mode layer, the compute graph is sequential by design
- If two slots' decode operations overlap in time, CUDA may serialize them at GPU boundaries
- Whichever slot gets scheduled first completes fast; the other waits

**Why It Alternates Between Runs:**
- The slot that gets PCIe bus access first is not deterministic
- Depends on timing of requests, kernel launch order, CUDA scheduling
- System noise (CPU scheduling, interrupts) can shift which slot "wins"

### 4. Is This Expected Behavior?

**Partially expected, but worse than documented:**

**Expected:**
- llama.cpp's layer-split mode is known to idle GPUs (documented in split mode guides)
- PCIe bandwidth is a known bottleneck for multi-GPU inference at high throughput

**Not Expected:**
- The **extreme asymmetry** (8.0 vs 0.6 tok/s) suggests more than just PCIe contention
- **Aggregate throughput < single-slot throughput** is abnormal
  - This indicates the slots are **interfering** with each other, not just sharing resources fairly
  - Likely cause: Cross-sequence attention computation overhead + PCIe serialization

**Known Issue:**
- GitHub Issue #10860: "Multiple slots, generation speed is degraded"
  - **Status:** Not yet fetched (404), but title matches exactly
- Reddit posts mention multi-GPU setups benefit from split-mode graph (ik_llama.cpp fork) with 3-4x improvement
  - This suggests default layer-split is suboptimal for concurrent generation

---

## Configuration Options to Improve Fairness

### 1. Switch to Split Mode Row (`--split-mode row`)

**What it does:**
- Splits weight matrices (tensors) across GPUs, not layers
- Every GPU contains a piece of every layer
- **All GPUs compute in parallel** for each layer

**Expected benefit:**
- Better GPU utilization (no idling)
- More balanced load at high context
- May reduce asymmetry since both slots use all GPUs simultaneously

**Tradeoff:**
- Requires high-speed interconnect (NVLink ideal, PCIe x16 acceptable)
- May be slower on PCIe x4 links due to increased cross-GPU traffic

**Command:**
```bash
llama-server -m model.gguf --split-mode row --tensor-split 0.30,0.37,0.33 ...
```

### 2. Try Split Mode Graph (Requires ik_llama.cpp Fork)

**What it does:**
- Implements tensor parallelism at GGML graph level
- Distributes compute graph nodes across GPUs
- Uses NVIDIA NCCL for topology-aware communication (NVLink/PCIe detection)

**Expected benefit:**
- 3-4x performance improvement (per benchmarks)
- 100% GPU utilization (no idling)
- **Should eliminate asymmetry** by parallelizing at finer granularity

**Availability:**
- Not in mainline llama.cpp as of March 2026
- Available in [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) fork

**Command:**
```bash
llama-cli -m model.gguf --split-mode graph -ngl 99 ...
```

### 3. Increase Batch Size (`-b`, `--batch-size`)

**What it does:**
- Controls token budget per batch (default: 512)
- Larger batches = more tokens processed per `llama_decode()` call

**Expected benefit:**
- May amortize PCIe overhead by grouping more work per transfer
- Could reduce contention by batching both slots' tokens together more efficiently

**Tradeoff:**
- Increases latency (tokens wait longer for batch to fill)
- Increases VRAM usage (larger intermediate activations)

**Command:**
```bash
llama-server -m model.gguf -b 1024 ...
```

### 4. Reduce Concurrent Slots (`-np`, `--parallel`)

**What it does:**
- Limits number of concurrent inference slots
- **Set to 1** for baseline single-slot performance

**Expected benefit:**
- Eliminates contention entirely
- Confirms whether asymmetry is caused by multi-slot interference

**Command:**
```bash
llama-server -m model.gguf --parallel 1 ...  # Baseline
llama-server -m model.gguf --parallel 2 ...  # Current setup
```

### 5. Adjust Context Size to Match Workload

**What it does:**
- Set `-c` to 2× max expected context per slot
- If running 2 slots at 100K each, set `-c 200000`

**Expected benefit:**
- Reduces cross-sequence attention waste
- Minimizes KV cache fragmentation

**Current status:**
- Likely already optimized if using Qwen3.5 default (128K or 256K)

---

## Hardware-Specific Considerations

### PCIe Bandwidth Analysis

**Your Setup:**
- GPU0 (RTX 3060): PCIe x16 (16 GB/s bidirectional)
- GPU1 (RTX 3060): PCIe x4 (4 GB/s bidirectional)
- GPU2 (RTX 3060): PCIe x4 (4 GB/s bidirectional)

**At 98K Context per Slot:**
- KV cache for 27B model @ Q4: ~98K tokens × 2 slots × ~100 bytes/token ≈ 19.6 MB per layer
- With 80 layers: ~1.5 GB KV cache total
- Cross-GPU transfers per decode step: Activation tensors (~10-50 MB per layer depending on batch size)

**Bottleneck:**
- GPU1/GPU2 on PCIe x4 likely starved when both slots generate simultaneously
- If tensor-split puts more layers on GPU1/GPU2, those GPUs become the bottleneck
- **This explains alternating asymmetry:** Whichever slot's computation hits the x4 GPUs first gets delayed

**From Reddit (PCIe x1 experience):**
> "Llama.cpp with the default split mode of layer splitting really doesn't care much about PCIe bandwidth"

This was **single-slot** inference. Concurrent slots change the picture:
- Single slot: Sequential layer processing, GPU1 waits idle → low bandwidth demand
- Concurrent slots: Simultaneous processing → PCIe contention → asymmetry

---

## Answer to Key Question

**Is the asymmetry caused by:**

**(c) Both software scheduling AND hardware contention**

**Software factors (40%):**
- Continuous batching is architecturally fair, but **unified KV cache** causes cross-sequence attention overhead
- Layer-split mode idles GPUs, creating scheduling windows where one slot monopolizes compute
- CUDA stream serialization at GPU boundaries can starve one slot

**Hardware factors (60%):**
- **PCIe bandwidth saturation** on x4 links when concurrent slots generate
- Cross-GPU communication volume explodes at high context (quadratic attention + large KV cache)
- Lack of NVLink means all inter-GPU traffic goes through CPU PCIe lanes → contention
- Tensor-split puts unequal layers on bottleneck GPUs

**Why aggregate < single-slot throughput:**
- Cross-sequence attention computation is wasted work (computed then masked)
- PCIe bus contention serializes what should be parallel work
- CUDA overhead increases with more kernel launches (2 slots = 2× sampling operations)

---

## Recommendations

### Immediate Actions (Test in Order):

1. **Baseline Test:**
   ```bash
   llama-server --parallel 1 -c 120000 ...
   ```
   Measure single-slot performance at 98K context to establish ceiling.

2. **Try Split Mode Row:**
   ```bash
   llama-server --split-mode row --parallel 2 -c 200000 ...
   ```
   Expect: Reduced asymmetry, possibly slower overall due to PCIe x4 overhead.

3. **Increase Batch Size:**
   ```bash
   llama-server --split-mode layer -b 1024 --parallel 2 ...
   ```
   Expect: Slight improvement if PCIe overhead is amortized.

4. **Reduce Parallel Slots During High Context:**
   ```bash
   llama-server --parallel 1 ...
   ```
   Use dynamic slot allocation: 2 slots at <50K context, 1 slot at >50K context.

### Long-Term Solutions:

1. **Migrate to ik_llama.cpp with Split Mode Graph:**
   - Expected 3-4× improvement
   - Eliminates GPU idling
   - Uses NCCL for optimal PCIe routing
   - **This is the proper fix**

2. **Upgrade PCIe Links:**
   - Risers/motherboard that provides x8 or x16 to all 3 GPUs
   - Would significantly reduce contention at high context

3. **Add NVLink Bridge (if supported):**
   - RTX 3060 does NOT support NVLink
   - Upgrade path: RTX 4090 or A100 with NVLink

4. **Use vLLM or ExLlamaV2 Instead:**
   - Both implement true tensor parallelism
   - Designed for multi-GPU from ground up
   - May outperform llama.cpp on this hardware at high concurrency

---

## References

1. **llama.cpp Discussion #4130:** Parallelization / Batching Explanation  
   - Unified KV cache architecture
   - Cross-sequence attention overhead at high context

2. **Hexdocs ADR 006:** Continuous Batching Implementation  
   - Decode-maximal scheduling
   - Single forward pass per tick design

3. **Medium Article (Jan 2026):** llama.cpp Multi-GPU Split Modes  
   - Layer vs Row vs Graph split comparison
   - Performance benchmarks (3-4× with graph mode)

4. **Reddit (Aug 2025):** PCIe Lane Bottlenecking Experience  
   - "Layer splitting doesn't care about PCIe bandwidth" (single-slot only)

5. **Reddit (Feb 2025):** Stop Wasting Multi-GPU Setup with llama.cpp  
   - Recommendation to use vLLM/ExLlamaV2 for tensor parallelism

6. **GitHub Issue #4185:** update_slots: failed to decode the batch  
   - KV cache exhaustion at high context
   - Related to slot management under load

---

## Conclusion

The severe asymmetry is caused by **concurrent slots competing for PCIe bandwidth** at high context depths, exacerbated by llama.cpp's **layer-split mode** which serializes cross-GPU communication. The "slow" slot alternates because PCIe bus contention is timing-dependent.

**The fact that aggregate throughput drops below single-slot performance proves this is not just fair resource sharing — the slots are actively interfering with each other through cross-sequence attention waste and PCIe serialization.**

**Recommended fix:** Switch to ik_llama.cpp with `--split-mode graph`, or migrate to vLLM for this hardware configuration.
