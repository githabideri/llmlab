# Nemotron-3-Nano-30B-A3B

NVIDIA's Nemotron-3-Nano, a 30B parameter MoE model with ~3B active parameters.

## Quick Facts

| Param | Value |
|-------|-------|
| Parameters | 30B total, ~3B active (MoE) |
| Architecture | Mixture of Experts |
| Quant tested | IQ4_NL (18GB on disk) |
| Context window | 131,072 tokens |
| VRAM requirement | **2x 12GB GPUs** (requires tensor split) |

## Performance

Tested on 2x RTX 3060 (12GB each) with `--tensor-split 1,1`:

| Metric | Value |
|--------|-------|
| Prompt eval (32k ctx) | 670 tok/s |
| Generation (fresh) | 66.7 tok/s |
| Generation (filled ctx) | ~58 tok/s |

## Known Issues

### 🚨 "Lost in Thought" Failure Mode

**Severity:** Medium — causes silent message drops

**Symptom:** 
- Model generates thinking/reasoning tokens
- Emits stop token without producing visible output
- Server returns HTTP 200 (no error)
- User sees nothing — message never delivered

**Trigger:** 
- Action-type prompts requiring multi-step planning
- Examples: "do steps 1-4", "install X and configure Y"

**Observed rate:** ~1.4% of turns (2/148 in testing)

**Technical details:**
```
stopReason: "stop"
content: [
  { type: "thinking", thinking: "...1000+ tokens of planning..." }
  // NO text output
  // NO tool calls
]
```

**Potential mitigations (untested):**
1. Use `--reasoning-format none` to disable thinking
2. Increase `max_tokens` to ensure output budget after thinking
3. Prompt engineering: prefix with "Respond with your first action:"
4. Gateway-side: detect thinking-only responses and retry

### Single-GPU Trap

**Symptom:** Slow generation, high CPU usage, GPU underutilized

**Cause:** 18GB model doesn't fit in single 12GB GPU, spills to RAM

**Fix:** Must use `--tensor-split 1,1` (or similar) to split across GPUs

## Recommended Config

```bash
llama-server \
  --model /path/to/Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf \
  --ctx-size 131072 \
  --n-gpu-layers -1 \
  --tensor-split 1,1 \
  --flash-attn 1 \
  -b 128 -ub 64 \
  --jinja \
  --host 0.0.0.0 --port 8080
```

**Required flags:**
- `--tensor-split 1,1` — split across 2 GPUs (adjust ratio for unequal VRAM)
- `--flash-attn 1` — recommended for long context

## Benchmark Results

Tested with llmlab OpenClaw benchmark (L0-L4 + DOC-QA):

| Case | Result |
|------|--------|
| L0: Read/write | ✅ Pass |
| L1: Config summary | ✅ Pass |
| L2: Config patch + JSON | ✅ Pass |
| L3: Bench parse | ✅ Pass |
| L4: Tool chain | ✅ Pass |
| DOC-QA: Citations | ✅ Pass |

Model handles structured tasks well when it doesn't hit the "lost in thought" failure.

## Hardware Tested

- 2x NVIDIA RTX 3060 12GB
- Intel i5-7400
- 24GB system RAM
- llama.cpp (latest as of 2026-02-09)

## Changelog

- **2026-02-09:** Initial profile. Discovered "lost in thought" failure mode. Documented tensor-split requirement.
