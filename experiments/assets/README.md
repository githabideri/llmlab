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

### Qwen3.5-35B nvtop Screenshot

**File:** `2026-03-03-qwen35-nvtop-pattern.png`

**What it shows:**
- **Moderate 25-50% GPU utilization** (GPU1: 32%, GPU2: 36%, GPU3: 31%)
- Gentle wave patterns with some variation
- More oscillation than Nemotron, less extreme than GLM
- VRAM: ~10.3GB, 9.0GB, 8.9GB per GPU (~28.2GB total)

**Pattern characteristics:**
- Not as smooth as Nemotron (more waves)
- Not as spiky as GLM (no extreme 0-100% swings)
- **Middle ground between constant-time (Mamba-2) and burst (MLA)**

### Architecture Comparison (Visual Evidence)

| Model | GPU Pattern | Utilization Range | Consistency | Architecture |
|-------|------------|-------------------|-------------|--------------|
| **GLM (MLA)** | **Extreme spikes (PP)** / Waves (TG) | 0-100% (PP), 25-75% (TG) | ❌ **Highly variable** | Multi-head Latent Attention |
| **Qwen3.5 (GQA)** | **Moderate waves** | 25-50% sustained | 🟡 **Moderate** | Grouped Query Attention |
| **Nemotron (Mamba-2)** | **Smooth sustained** | 30-60% consistent | ✅ **Very consistent** | Mamba-2 hybrid (constant-time) |

**Visual fingerprints discovered:**
1. **MLA (GLM):** Extreme alternating bursts — memory-bandwidth limited, sequential wave processing
2. **GQA (Qwen3.5):** Moderate sustained waves — traditional attention with some optimization
3. **Mamba-2 (Nemotron):** Smooth consistent utilization — constant-time attention, best hardware efficiency

**Implication:** Visual evidence confirms architectural trade-offs:
- **GLM (MLA):** Highest VRAM cost, most variable GPU load, steepest degradation
- **Qwen3.5 (GQA):** Balanced middle ground, good VRAM efficiency, moderate degradation
- **Nemotron (Mamba-2):** Best VRAM efficiency, most consistent load, shallowest degradation
