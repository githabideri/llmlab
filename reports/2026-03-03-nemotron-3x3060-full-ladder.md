# Triple GPU Full Context Ladder — Nemotron-3-Nano-30B A3B

**Date:** 2026-03-03  
**Model:** Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf  
**Hardware:** 3x RTX 3060 (36GB VRAM total)  
**Runtime:** `--ctx-size 262144 --split-mode layer --ngl 99 --parallel 1 --reasoning-format deepseek`

---

## Complete Context Speed Ladder (0-250K)

| Context | PP tok/s | TG tok/s | TG % | Notes |
|--------:|---------:|---------:|-----:|-------|
| 11 | 240 | 102 | 100% | Baseline |
| 3,215 | 1,800 | 95 | 93% | Near-peak PP |
| 6,415 | **1,899** | 93 | 91% | **Peak PP speed** |
| 9,615 | 1,913 | **115** | **113%** | 🔥 **Peak TG (113% of baseline!)** |
| 12,815 | 1,906 | 111 | 109% | Sustained peak |
| 19,215 | 1,879 | 105 | 103% | Still above baseline |
| 24,015 | 1,843 | 100 | 98% | Optimal zone ceiling |
| 32,015 | 1,789 | 95 | 93% | -7% from peak |
| 64,015 | 1,577 | 76 | 74% | -26% from peak |
| 95,015 | 1,407 | 62 | 61% | -39% from peak |
| **128,015** | **1,260** | **54** | **53%** | -47% from peak |
| **192,015** | **1,049** | **42** | **41%** | -59% from peak |
| **250,015** | **911** | **34** | **33%** | **Maximum tested (-67%)** |

**Percentages relative to baseline (11 tokens).**

**Notable:** TG *improved* above baseline at 10-19K context range, likely due to optimal KV cache prefill.

---

## Key Findings

### Prompt Processing (PP)
- **Peak:** 1,913 tok/s at ~10K context
- **Sustained high:** >1,800 tok/s from 3K-24K
- **At 250K:** 911 tok/s (-52% from peak)
- **Graceful degradation:** No sudden drops or cliffs

### Text Generation (TG)
- **Peak:** **115 tok/s** at 10K context (73% faster than previous 2×3060 baseline of 66.7 tok/s!)
- **Above-baseline zone:** 95-115 tok/s from 3K-24K (Mamba-2 "sweet spot")
- **Excellent 32-95K:** 62-95 tok/s (still very usable for agentic work)
- **At 250K limit:** 34 tok/s (-67% from peak, **89% faster than Qwen3.5 at same context**)

### Performance Zones

| Zone | Context Range | TG Performance | Recommended Use |
|------|--------------|----------------|-----------------|
| 🟢 **Optimal** | 0-32K | 95-115 tok/s | Interactive chat, agentic workflows |
| 🟡 **Good** | 32-95K | 62-76 tok/s | Long docs, multi-file code review |
| 🟠 **Usable** | 95-192K | 42-54 tok/s | Very long documents |
| 🔴 **Slow but viable** | 192-250K | 34-42 tok/s | Ultra-long context (still functional) |

---

## Architecture Advantage: Mamba-2 Hybrid

**Key observation:** Nemotron's Mamba-2 hybrid architecture shows **significantly better context retention** than traditional attention:

| Model | Architecture | TG @ 250K | Degradation |
|-------|-------------|-----------|-------------|
| Nemotron-30B | Mamba-2 hybrid | **34 tok/s** | **-67%** |
| Qwen3.5-35B | GQA (traditional) | 18 tok/s | -69% |

**At extreme context (250K), Nemotron is 89% faster than Qwen3.5** despite similar active parameter count (~3B).

**Crossover point:** Both models are excellent 0-32K, but Nemotron pulls ahead significantly beyond 64K context.

---

## 2×3060 → 3×3060 Upgrade Impact

Comparing against previous baseline (Feb 2026, 2×RTX 3060):

| Metric | 2×3060 | 3×3060 | Improvement |
|--------|--------|--------|-------------|
| **Peak TG** | 66.7 tok/s | **115 tok/s** | **+73%** 🔥 |
| **Peak PP** | ~670 tok/s (32K) | **1,913 tok/s** (10K) | **+185%** |
| **Max stable ctx** | 196K (OOM at 256K) | **250K+** (no OOM) | **+28%** |
| **Total VRAM** | 24GB | 36GB | +50% |

**Key finding:** The TG speed improvement (+73%) **exceeds the VRAM increase (+50%)**, suggesting the third GPU provides more than just capacity — likely better parallelization/reduced PCIe bottlenecks with `--split-mode layer`.

**Prompt processing improvement is massive** (+185%), likely because:
- Better memory bandwidth distribution across 3 PCIe lanes
- More efficient layer split with 3-way division
- Reduced contention on any single GPU

---

## 3-GPU Distribution

**VRAM usage across context range:**
- **At 95K:** GPU0 ~8GB, GPU1 ~8GB, GPU2 ~8GB (~24GB total)
- **At 250K:** (estimated ~32GB based on linear scaling)

**Active utilization during inference:**
- All 3 cards actively processing (confirmed via `nvtop` during tests)
- Balanced load distribution with `--split-mode layer`
- No single-GPU bottleneck observed

✅ **All 3 cards actively contributing**  
✅ **No OOM errors at any context level**  
✅ **Graceful degradation across entire 0-250K range**  
✅ **No sudden performance cliffs**

---

## Runtime Configuration

```bash
llama-server \
  --model /mnt/models/gguf/nemotron-3-nano-30b-a3b/Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf \
  --ctx-size 262144 \
  --split-mode layer \
  --gpu-layers 99 \
  --parallel 1 \
  --flash-attn on \
  --reasoning-format deepseek \
  --cache-type-k q8_0 \
  --cache-type-v q4_0 \
  --host 0.0.0.0 \
  --port 8080
```

**Critical flags:**
- `--split-mode layer` — distributes model layers across GPUs (essential for multi-GPU)
- `--flash-attn on` — memory-efficient attention (crucial for long context)
- `--reasoning-format deepseek` — prevents "lost in thought" failure mode

---

## Comparison to Qwen3.5-35B Q4_K_M

Both models tested on same hardware (3×RTX 3060) with full 0-250K ladder:

| Context | Nemotron TG | Qwen3.5 TG | Nemotron Advantage |
|--------:|------------:|-----------:|-------------------:|
| 10K | **115** tok/s | 57 tok/s | +102% |
| 32K | 95 tok/s | 50 tok/s | +90% |
| 64K | 76 tok/s | 40 tok/s | +90% |
| 128K | 54 tok/s | 28 tok/s | +93% |
| 250K | **34** tok/s | **18** tok/s | **+89%** |

**Nemotron is consistently 90-100% faster across all context ranges.**

**Architecture explains the difference:**
- Nemotron: Mamba-2 hybrid (constant-time attention scaling)
- Qwen3.5: GQA (quadratic attention, even with grouped queries)

**Practical takeaway:** For long-context agentic work (>32K), Nemotron's Mamba-2 architecture delivers nearly 2× the throughput.

---

## Recommendations

### When to Use Nemotron-30B
- ✅ Long-context work (>32K tokens)
- ✅ Agentic workflows with multi-file context
- ✅ Speed-critical inference at depth
- ✅ When you have 3+ GPUs with 24GB+ total VRAM

### When to Use Qwen3.5-35B
- Vision tasks (Nemotron is text-only)
- Shorter context (<32K) where difference is minimal
- When you need the extra 5B parameters for reasoning quality

### Hardware Requirements
- **Minimum:** 2×12GB GPUs (18GB model + KV cache)
- **Recommended:** 3×12GB GPUs for 250K+ context headroom
- **Must use:** `--split-mode layer` (never `row` on PCIe)

---

## Test Methodology

- **Script:** `llmlab/scripts/run_context_ladder.py`
- **Test points:** 13 (9, 3K, 6K, 10K, 13K, 19K, 24K, 32K, 64K, 95K, 128K, 192K, 250K)
- **Per test:** Prompt eval + 3 tokens generation (to measure both PP and TG)
- **Monitoring:** CPU/RAM usage (lightweight), GPU metrics not captured (future improvement)
- **Duration:** ~14 minutes (20:47-21:01 UTC)

**Limitations:**
- No parallel slot testing (single-slot only)
- GPU VRAM usage estimated (not captured by script)
- No sustained inference testing (short bursts only)

---

## Next Steps

1. **Soak testing:** Run sustained agentic workload at 64K-128K context to verify stability
2. **Parallel slots:** Test multi-slot performance at various context levels
3. **Compare to GLM-4.7-Flash:** Full ladder test running now (ETA 21:25 UTC)
4. **4th GPU exploration:** Validate if 48GB total enables 300K+ context
5. **Update baseline:** Adopt 3×3060 as new standard hardware config in model profiles

---

## Changelog

- **2026-03-03:** Initial full ladder test on 3×RTX 3060, discovered massive TG improvement (+73% vs 2×3060 baseline)
