# Experiment Assets

Visual evidence from benchmarks and testing sessions.

## 2026-03-03: Triple GPU Context Ladder Tests

### GLM-4.7-Flash nvtop Screenshots

**Files:** 
- `2026-03-03-glm-nvtop-zigzag-pattern.png` — **Prompt Processing (PP)**
- `2026-03-03-glm-nvtop-tgen-pattern.png` — **Text Generation (TG)**

**What they show:**

**Screenshot 1 (PP - Prompt Processing):**
- **Extreme zig-zag pattern** (0% → 100% → 0% swings)
- GPUs alternate aggressively during prompt ingestion
- Memory-bandwidth bound, bursty parallel processing
- VRAM: ~30GB total (11.1GB + 9.6GB + 9.2GB)

**Screenshot 2 (TG - Text Generation):**
- **Gentler wave pattern** (25-75% utilization range)
- More sustained load: GPU1 28%, GPU2 70%, GPU3 17%
- Sequential token generation, compute-bound
- Less synchronized than PP phase

**Why it matters:**
Demonstrates that GLM's MLA architecture has **different utilization signatures for PP vs TG**:
- **PP (prompt)**: Extreme alternating spikes (memory-bandwidth limited)
- **TG (generation)**: Moderate sustained waves (compute-bound)

This is architecturally different from:
- **Nemotron (Mamba-2)**: Expected smooth continuous utilization (comparison TBD)
- **Qwen3.5 (GQA)**: (comparison TBD)

MLA's bursty pattern suggests:
- Multi-head Latent Attention processes in sequential waves
- Layer-wise distribution across GPUs
- Different optimization strategy than Mamba-2's constant-time attention

Related experiments:
- `experiments/2026-03-03-glm-3x3060-full-ladder.md` (when complete)
- `experiments/2026-03-03-nemotron-3x3060-full-ladder.md`
- `experiments/2026-03-03-qwen3.5-35b-q4km-context-ladder.md`

### Nemotron-3-Nano-30B nvtop Screenshots

**Files:**
- `2026-03-03-nemotron-nvtop-pp-pattern.png` — **Prompt Processing (PP)**
- `2026-03-03-nemotron-nvtop-tg-pattern.png` — **Text Generation (TG)**

**What they show:**

**Screenshot 1 (PP - Prompt Processing):**
- **Sustained 30-60% GPU utilization** (GPU1: 41%, GPU2: 57%, GPU3: 32%)
- More variation in utilization patterns
- Smooth waves, not extreme spikes
- VRAM: ~8.8GB, 7.7GB, 7.9GB per GPU

**Screenshot 2 (TG - Text Generation):**
- **Similar 30-60% sustained load** (GPU1: 40%, GPU2: 51%, GPU3: 28%)
- Slightly steadier patterns than PP
- Smooth continuous utilization
- VRAM: ~8.8GB, 7.7GB, 7.9GB per GPU

**Key finding:** **Nemotron shows remarkably similar patterns for PP vs TG**, unlike GLM's dramatic difference (0-100% spikes vs 25-75% waves). This demonstrates **Mamba-2's constant-time attention advantage** — more predictable, consistent GPU utilization regardless of processing phase.

### Architecture Comparison (Visual Evidence)

| Model | PP Pattern | TG Pattern | Consistency |
|-------|-----------|-----------|-------------|
| **GLM (MLA)** | **0-100% extreme spikes** | 25-75% waves | ❌ **Highly variable** |
| **Nemotron (Mamba-2)** | **30-60% smooth waves** | **30-60% smooth waves** | ✅ **Very consistent** |
| Qwen3.5 (GQA) | [Pending] | [Pending] | [Pending] |

**Implication:** Mamba-2's consistent utilization pattern suggests better hardware efficiency and more predictable performance characteristics compared to traditional attention mechanisms.
