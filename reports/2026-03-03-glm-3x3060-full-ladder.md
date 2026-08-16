# Triple GPU Full Context Ladder — GLM-4.7-Flash

**Date:** 2026-03-03  
**Model:** GLM-4.7-Flash-UD-Q4_K_XL.gguf  
**Hardware:** 3x RTX 3060 (36GB VRAM total)  
**Runtime:** `--ctx-size 262144 --split-mode layer --ngl 99 --parallel 1 --reasoning-format deepseek`

---

## Visual Evidence: MLA Architecture Fingerprinting

GLM's Multi-head Latent Attention (MLA) shows **distinctive utilization patterns** that differ between processing phases:

### Prompt Processing (PP) — Extreme Zig-Zag Pattern

<img src="assets/2026-03-03-glm-nvtop-zigzag-pattern.png" width="800" alt="GLM Prompt Processing - Extreme alternating GPU spikes">

**Characteristics:**
- **Aggressive 0-100% alternating spikes** across all GPUs
- GPUs alternate in waves rather than parallel continuous load
- Memory-bandwidth bound, bursty parallel processing
- VRAM: ~30GB total (11.1GB + 9.6GB + 9.2GB)

### Text Generation (TG) — Gentler Wave Pattern

<img src="assets/2026-03-03-glm-nvtop-tgen-pattern.png" width="800" alt="GLM Text Generation - Sustained waves">

**Characteristics:**
- **More sustained 25-75% utilization** (less extreme swings)
- GPU1: 28%, GPU2: 70%, GPU3: 17% (less synchronized)
- Compute-bound sequential token generation
- Steadier load compared to PP phase

**Key insight:** MLA architecture shows **different GPU utilization signatures for PP vs TG**, suggesting sequential wave processing rather than parallel streaming.

---

## Complete Context Speed Ladder (0-128K)

**Note:** Tests beyond 128K failed due to VRAM constraints. GLM's MLA architecture has higher VRAM requirements than Nemotron/Qwen3.5.

**Failure details:**
- **192K:** CUDA OOM (KV cache allocation failed, needed ~857 MB more)
- **250K:** Exceeded auto-reduced context window (server: 202K, not 262K)
- **Root cause:** GLM's context was auto-reduced from 262K → 202K due to insufficient VRAM for full KV cache

| Context | PP tok/s | TG tok/s | TG % | Notes |
|--------:|---------:|---------:|-----:|-------|
| 10 | 105 | 66 | 100% | Baseline |
| 3,213 | **1,562** | 60 | 91% | **Peak PP speed** |
| 6,413 | 1,499 | 53 | 80% | High efficiency |
| 9,613 | 1,298 | 48 | 73% | Optimal zone |
| 12,813 | 1,138 | 43 | 65% | Degradation begins |
| 19,213 | 903 | 40 | 61% | -39% from peak |
| 24,013 | 777 | 37 | 56% | -44% from peak |
| 32,013 | 635 | 32 | 48% | -52% from peak |
| 64,013 | 365 | 21 | 32% | -68% from peak |
| 95,013 | 258 | 16 | 24% | -76% from peak |
| **128,013** | **196** | **13** | **20%** | **-80% from peak** |

**Percentages relative to baseline (10 tokens).**

---

## Key Findings

### Prompt Processing (PP)
- **Peak:** 1,562 tok/s at 3K context
- **High efficiency zone:** >1,000 tok/s from 3K-13K
- **Steep degradation:** Drops to 196 tok/s at 128K (-87% from peak)
- **Failed at extreme depth:** 192K+ tests timed out or OOM'd

### Text Generation (TG)
- **Peak:** 66 tok/s at baseline (10 tokens)
- **Rapid decline:** Drops to 48 tok/s by 10K context (-27%)
- **Severe degradation:** Only 13 tok/s at 128K (-80% from peak)
- **Worst among tested models** at high context (see comparison below)

### Performance Zones

| Zone | Context Range | TG Performance | Recommended Use |
|------|--------------|----------------|-----------------|
| 🟢 **Optimal** | 0-10K | 48-66 tok/s | Short chats, simple tasks |
| 🟡 **Acceptable** | 10-32K | 32-48 tok/s | Medium docs, basic agentic |
| 🟠 **Slow** | 32-95K | 16-32 tok/s | Long docs (if patient) |
| 🔴 **Very slow** | 95-128K | 13-16 tok/s | Avoid for interactive use |
| ❌ **Failed** | 128K+ | N/A | Timeout/OOM on 3×3060 |

**Practical limit on 3×3060:** ~95K context (beyond that, performance becomes unusable for interactive work).

---

## Architecture Characteristics: MLA (Multi-head Latent Attention)

Based on visual evidence from nvtop screenshots:

**PP Phase (Prompt):**
- Extreme alternating GPU spikes (0-100%)
- Memory-bandwidth limited
- Bursty parallel ingestion
- Suggests sequential layer processing with wave distribution

**TG Phase (Generation):**
- Gentler sustained waves (25-75%)
- Compute-bound
- Less synchronized across GPUs
- Sequential token-by-token generation pattern

**Compared to other architectures (3×3060):**

| Model | Architecture | PP Pattern | TG Pattern |
|-------|-------------|------------|------------|
| GLM-4.7 | MLA | **Extreme zig-zag** | Gentle waves |
| Nemotron-30B | Mamba-2 | [Screenshot pending] | [Screenshot pending] |
| Qwen3.5-35B | GQA | [Screenshot pending] | [Screenshot pending] |

---

## 3-GPU Distribution

**VRAM usage (from screenshots):**
- GPU1: 11.1 GB
- GPU2: 9.6 GB  
- GPU3: 9.2 GB
- **Total: ~30 GB**

**Utilization patterns:**
- PP: Alternating 0-100% spikes
- TG: Mixed 17-70% sustained loads

---

## Runtime Configuration

```bash
llama-server \
  --model /mnt/models/gguf/glm-4.7-flash/GLM-4.7-Flash-UD-Q4_K_XL.gguf \
  --ctx-size 262144 \           # Configured
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

**Note:** Server auto-reduced context to **202,752 tokens** (from configured 262,144) due to insufficient VRAM for full KV cache with GLM's MLA architecture on 3×3060 (36GB).

---

## Comparison to Nemotron & Qwen3.5 (3×RTX 3060)

All three models tested on same hardware with full context ladder:

| Context | GLM-4.7 TG | Nemotron-30B TG | Qwen3.5-35B TG | Winner |
|--------:|-----------:|----------------:|---------------:|--------|
| 10K | 48 tok/s | **115 tok/s** | 57 tok/s | **Nemotron (+140%)** |
| 32K | 32 tok/s | **95 tok/s** | 50 tok/s | **Nemotron (+197%)** |
| 64K | 21 tok/s | **76 tok/s** | 40 tok/s | **Nemotron (+262%)** |
| 95K | 16 tok/s | **62 tok/s** | 33 tok/s | **Nemotron (+288%)** |
| 128K | **13 tok/s** | **54 tok/s** | 28 tok/s | **Nemotron (+315%)** |
| 250K | ❌ failed | **34 tok/s** | 18 tok/s | **Nemotron** |

**Key takeaways:**
1. **Nemotron dominates at all context ranges** (2-4× faster than GLM)
2. **GLM has worst context retention** of the three (-80% @ 128K vs Nemotron -47%)
3. **GLM fails beyond 128K** on 3×3060 (Nemotron/Qwen3.5 both reach 250K)
4. **Architecture matters:** Mamba-2 (Nemotron) >> GQA (Qwen3.5) >> MLA (GLM) for long context

### Degradation Comparison

| Model | Architecture | TG @ 128K | Degradation |
|-------|-------------|-----------|-------------|
| **Nemotron-30B** | Mamba-2 | **54 tok/s** | **-47%** |
| Qwen3.5-35B | GQA | 28 tok/s | -53% |
| **GLM-4.7** | MLA | **13 tok/s** | **-80%** |

**GLM's MLA architecture shows the steepest degradation** under long context load, despite being the smallest model (4.7B active vs ~3B for Nemotron/Qwen3.5).

### VRAM Efficiency Comparison

**Maximum stable context on 3×3060 (36GB VRAM):**

| Model | Architecture | Max Context | KV Cache Efficiency |
|-------|-------------|------------:|---------------------|
| Nemotron-30B | Mamba-2 | **250K+** | ✅ **Best** (constant-time) |
| Qwen3.5-35B | GQA | 250K+ | ✅ Good |
| **GLM-4.7** | MLA | **~128K** | ❌ **Worst** (high per-token VRAM) |

**Key finding:** GLM's MLA has **significantly higher VRAM requirements per token** than Mamba-2 or GQA, limiting practical context on consumer GPUs.

Server auto-reduced from 262K → 202K configured context due to insufficient VRAM for full KV cache.

---

## Test Methodology

- **Script:** `llmlab/scripts/run_context_ladder.py`
- **Test points attempted:** 13 (9, 3K, 6K, 10K, 13K, 19K, 24K, 32K, 64K, 95K, 128K, 192K, 250K)
- **Test points completed:** 11 (0-128K range, 192K & 250K failed)
- **Per test:** Prompt eval + 3 tokens generation
- **Visual monitoring:** nvtop screenshots captured during PP and TG phases
- **Duration:** 38 minutes (21:10-21:48 UTC)
- **Failure mode:** Tests beyond 128K timed out or OOM'd (context too large for 3×3060 with this model)

---

## Next Steps

1. Complete full numerical comparison vs Nemotron and Qwen3.5
2. Capture comparison screenshots for Nemotron and Qwen3.5 (PP + TG phases)
3. Update GLM model profile with 3×3060 results
4. Document MLA architectural advantages/disadvantages vs Mamba-2 and GQA

---

## Changelog

- **2026-03-03:** Initial experiment with visual GPU utilization fingerprinting (MLA architecture analysis)
