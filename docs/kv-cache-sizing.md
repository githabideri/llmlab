# KV Cache Sizing & Quantization

**Purpose:** How to compute KV cache memory from model architecture, choose cache quantization, and understand quality/speed tradeoffs.

---

## Formula

KV cache only exists for **full_attention layers**. Linear attention (DeltaNet, SSM, Mamba) has zero KV cache — it's O(1) in context.

**Per-token cache (all full_attention layers combined):**

```
K_bytes = num_full_attn_layers × num_kv_heads × head_dim × bytes_per_element_K
V_bytes = num_full_attn_layers × num_kv_heads × head_dim × bytes_per_element_V
```

| Quant type | bytes/element |
|------------|---------------|
| f16 | 2 |
| q8_0 | 1 |
| q4_0 | 0.5 |

**Total KV cache = (K_bytes + V_bytes) × ctx_size × parallel_slots**

---

## Worked Examples

### Qwen3.6-35B-A3B (our primary MoE)

Architecture from [config.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json):
- 40 layers: `[linear, linear, linear, full]` × 10 → **10 full_attention, 30 linear_attention**
- `num_kv_heads: 2`, `head_dim: 256`
- Per token: K = 10 × 2 × 256 = 5120 elements, V = 5120 elements

| Cache type | Total/token | 4K ctx | 32K ctx | 128K ctx | 256K ctx |
|------------|-------------|--------|---------|----------|----------|
| f16/f16 | 20 KiB | 80 MiB | 625 MiB | 2.5 GiB | 5 GiB |
| q8_0/q4_0 | 7.5 KiB | 30 MiB | 234 MiB | 938 MiB | 1.88 GiB |
| q4_0/q4_0 | 5 KiB | 20 MiB | 156 MiB | 625 MiB | 1.25 GiB |

**Key insight:** Despite being a "35B" model, the KV cache is tiny because 30/40 layers are linear attention. A dense 35B model with 64 layers would have ~6× the cache.

### Qwen3.6-27B (hybrid dense)

Architecture from [config.json](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json):
- 64 layers: `[linear, linear, linear, full]` × 16 → **16 full_attention, 48 linear_attention**
- `num_kv_heads: 4`, `head_dim: 256`
- Per token: K = 16 × 4 × 256 = 16384 elements, V = 16384 elements

| Cache type | Total/token | 4K ctx | 32K ctx | 128K ctx | 160K ctx |
|------------|-------------|--------|---------|----------|----------|
| f16/f16 | 64 KiB | 256 MiB | 2 GiB | 8 GiB | 10 GiB |
| q8_0/q4_0 | 24 KiB | 96 MiB | 768 MiB | 3 GiB | 3.75 GiB |
| q4_0/q4_0 | 16 KiB | 64 MiB | 512 MiB | 2 GiB | 2.5 GiB |

**Key insight:** Even this "dense" model is hybrid — 48/64 layers are linear attention with zero KV cache. Still, 16 full_attention layers with 4 KV heads means the cache is 2× the 35B-A3B per layer. On RTX 3090 (24 GB), 160K context at q8_0/q4_0 uses ~3.75 GiB for KV cache alone.

---

## Quantization Quality & Speed

### Quality

Sources: [r/LocalLLaMA discussion](https://www.reddit.com/r/LocalLLaMA/comments/1q97081/quantized_kv_cache/), [DGX Spark benchmark](https://github.com/Memoriant/dgx-spark-kv-cache-benchmark), [llama.cpp TurboQuant discussion](https://github.com/ggml-org/llama.cpp/discussions/20969)

- **K-cache is more sensitive than V-cache.** K projections are used for attention scoring — quantization error directly affects which tokens get attended to.
- **q8_0 for K:** "Free quality" — <0.1% perplexity delta vs f16 on every model measured. Indistinguishable from f16 in practice.
- **q4_0 for V:** Minimal quality loss. V-cache stores value vectors that get weighted-summed — quantization error averages out.
- **q4_0 for K:** Noticeable quality loss at long context (>64K). Retrieval accuracy degrades. **Do NOT use q3_0 or lower for K-cache.**

**Our default across all services:** `--cache-type-k q8_0 --cache-type-v q4_0`

### Speed

KV cache quantization saves memory but adds per-token dequantization overhead during decode:

| Context | f16 | q8_0/q4_0 | q4_0/q4_0 | Source |
|---------|-----|-----------|-----------|--------|
| ~6K | baseline | ~baseline | -1% | [DGX Spark benchmark](https://github.com/Memoriant/dgx-spark-kv-cache-benchmark) (Nemotron-30B) |
| ~24K | baseline | ~baseline | -5% | Same |
| ~110K | baseline | -34% | -37% | Same |

- **Prompt processing is unaffected** — all tokens processed in parallel, dequantization cost amortized across the batch.
- **Decode speed degrades at long context** — each generated token must dequantize the full KV cache. At 110K+, q4_0 is measurably slower.
- **On high-bandwidth GPUs (RTX 3060+, 360 GB/s GDDR6), the speed tax matters less** than on unified memory (GB10, 273 GB/s LPDDR5X) where bandwidth isn't the bottleneck.

**Practical rule:** Use q8_0/q4_0 for everything. Switch to q4_0/q4_0 only when you need every last MiB of VRAM for context and can tolerate the decode speed tax at long context.

---

## Where the KV Cache Lives

The KV cache follows the layer assignment:
- Layers on GPU → KV cache in VRAM
- Layers on CPU → KV cache in system RAM

For our setups:
- **wgpx15 (dual 3060, tensor-split):** All layers on GPU → entire KV cache in VRAM, split across GPUs
- **llama-backup (single 3060, --n-cpu-moe):** Dense layers (including all full_attention) on GPU → KV cache in VRAM. Only expert weights move to CPU.

---

## TurboQuant (turbo3/turbo4)

Google Research's extreme KV cache quantization — under 3 bits per value with near-zero quality loss. Implemented in [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) and [Madreag's CUDA fork](https://github.com/Madreag/turbo3-cuda).

**Status:** Not yet merged to mainline llama.cpp. CUDA path works on SM 8.6+ (Ampere+). First SM 121 (Blackwell) results available [here](https://github.com/Memoriant/dgx-spark-kv-cache-benchmark).

**Not used in any production service yet.** Worth watching for future context scaling.

---

## Related

- `hardware/triple-3060.md` — VRAM budget breakdown with KV cache allocations
- `llama-cpp-systemd.md` — Service configs with cache quantization flags
- `runbook.md` — Operational reference
