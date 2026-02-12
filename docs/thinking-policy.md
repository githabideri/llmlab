# Thinking/Reasoning Control Policy for Local LLM Operations

> Practical guidelines for when to enable, limit, or disable reasoning ("thinking") in local LLM deployments. Applies to reasoning-capable models (Qwen3, DeepSeek-R1 distills, Nemotron-style).

> **Operational update (2026-02-12):** current llmlab default for Nemotron on both `llama-cpp` and `llama-local` is the measured "B" profile (brief constrained reasoning with `--reasoning-budget -1`) for consistency across backends during comparative runs.

## Background

Reasoning models generate internal chain-of-thought (CoT) tokens inside `<think>...</think>` blocks before producing the final answer. These tokens:

- **Cost latency linearly** — each thinking token = one generation step. On CPU (~5-15 tok/s for 7-8B Q4), 500 thinking tokens = 30-100s extra. On dual 3060 GPU (~60-80 tok/s), the same = 6-8s.
- **Consume KV cache** — thinking tokens occupy context slots, reducing effective context for the actual task.
- **Improve quality on hard tasks** — math, multi-step logic, code debugging. But for simple tasks (extraction, formatting, classification), thinking adds overhead with zero quality gain ("overthinking problem" — see OptimalThinkingBench, 2025).
- **Can degrade simple tasks** — DeepSeek-R1 repo notes that skipping `<think>` entirely can hurt, but excessive reasoning on trivial prompts wastes resources and occasionally produces worse conclusions.

## Control Mechanisms

### llama.cpp (llama-server)
- No native `thinking_budget` parameter (as of early 2026).
- **Disable thinking:** Use system prompt: `Do not use <think> blocks. Answer directly.` — works for Qwen3 in non-thinking mode.
- **Limit thinking:** Use `max_tokens` conservatively (thinking + answer combined), or implement a logit processor if using Python bindings.
- **Qwen3 specific:** Append `/no_think` or set `enable_thinking: false` in chat template kwargs (requires template support).

### vLLM
- Native support via `--reasoning-parser qwen3` (or `deepseek_r1`).
- Per-request: `"enable_thinking": false` in request body.
- Server-wide default: `--default-chat-template-kwargs '{"enable_thinking": false}'`
- Thinking content returned separately in `reasoning_content` field.

### Agent Frameworks (OpenAI-compatible API)
- Most frameworks pass `extra_body` or model-specific params.
- For OpenClaw/similar: set thinking level at the session or model config level.

## Policy Matrix

### CPU Room (llama.cpp, CPU-only, ~5-15 tok/s TG)

| Task Type | Thinking Setting | Rationale |
|-----------|-----------------|-----------|
| Simple Q&A, extraction, formatting | **OFF** | 0 benefit, huge latency cost (30-100s wasted) |
| Classification, routing | **OFF** | Deterministic tasks don't benefit from CoT |
| Summarization | **OFF** | Quality difference negligible, latency unacceptable |
| Tool use / function calling | **OFF or MINIMAL** (≤50 tokens) | Brief planning helps tool selection; long chains timeout |
| Code generation (simple) | **OFF** | Direct generation faster and equivalent quality |
| Math / logic puzzles | **LOW** (≤200 tokens) | Some benefit, but cap strictly to avoid 2+ min waits |
| Complex multi-step reasoning | **Consider skipping on CPU** | Even with thinking, CPU latency makes this impractical for interactive use |

**CPU default: Thinking OFF.** Only enable for explicit reasoning tasks, always with a strict token budget.

### GPU Room (dual RTX 3060, llama.cpp or vLLM, ~40-80 tok/s TG)

| Task Type | Thinking Setting | Rationale |
|-----------|-----------------|-----------|
| Simple Q&A, extraction, formatting | **OFF** | Still no benefit; saves 5-10s |
| Classification, routing | **OFF** | Unnecessary overhead |
| Summarization | **OFF or MINIMAL** | Marginal benefit at best |
| Tool use / function calling | **LOW** (≤100 tokens) | Brief planning improves tool chain accuracy |
| Code generation | **LOW** (≤200 tokens) | Thinking helps with architecture decisions |
| Math / multi-step logic | **ON** (≤500 tokens) | Sweet spot: real quality gain, acceptable latency (~6-12s) |
| Complex reasoning / analysis | **ON** (≤1000 tokens) | Full benefit, ~12-25s overhead is tolerable |
| Agentic multi-turn workflows | **LOW** (≤150 tokens per turn) | Cumulative thinking tokens across turns can explode context |

**GPU default: Thinking LOW (~100-200 tokens).** Enough for planning, not enough to overthink.

## Thinking Budget Guidelines

| Level | Token Budget | Use When |
|-------|-------------|----------|
| **OFF** | 0 | Simple/extractive tasks, CPU inference, latency-critical |
| **MINIMAL** | ≤50 | Quick planning for tool calls, CPU with tolerance |
| **LOW** | 100-200 | General-purpose with GPU, moderate complexity |
| **MEDIUM** | 200-500 | Math, code architecture, multi-step reasoning (GPU only) |
| **HIGH** | 500-1000+ | Hard problems where quality >> latency (GPU only) |

## Key Tradeoffs

```
Quality gain vs. thinking tokens (approximate):

High  │          ╭──────────── Complex reasoning
      │        ╭─╯
      │      ╭─╯
      │    ╭─╯              ← Diminishing returns zone
      │  ╭─╯
      │╭─╯
Low   │╯─────────────────── Simple tasks (flat line)
      └──────────────────────
      0    200   500   1000   tokens
```

- **Simple tasks:** Quality is flat regardless of thinking budget. Every token is waste.
- **Complex tasks:** Most gain in first 200-500 tokens. Diminishing returns after that.
- **CPU penalty multiplier:** ~5-10x vs GPU. A 500-token thinking budget costs 30-100s on CPU vs 6-12s on GPU.

## A/B Experiment Design

### Goal
Quantify quality-latency tradeoff for thinking levels on llmlab hardware.

### Setup

**Independent Variable:** Thinking level (OFF / LOW-100 / LOW-200 / MED-500)

**Control Variables:**
- Same model + quantization (e.g., Qwen3-8B-Q4_K_M)
- Same prompt set
- Same temperature (0.0 for reproducibility)
- Same hardware config per room

**Test Prompt Categories (5 prompts each, 20 total):**
1. **Simple extraction** — "Extract the date from: ..." 
2. **Classification** — "Classify this support ticket: ..."
3. **Tool/function call** — "Look up the weather in ..." (structured output)
4. **Code generation** — "Write a Python function to ..." (medium complexity)
5. **Math/logic** — "Solve: ..." (multi-step)

### Metrics

| Metric | How to Measure |
|--------|---------------|
| **Total latency (ms)** | Wall clock from request to completion |
| **Thinking tokens** | Count tokens in `<think>...</think>` block |
| **Answer tokens** | Count tokens outside think block |
| **Correctness** | Manual scoring 0-2 (wrong/partial/correct) per prompt |
| **Answer quality** | Blind ranking across conditions (1=best, 4=worst) |
| **Total tokens/s** | (thinking + answer tokens) / latency |

### Procedure

1. Run each prompt × each thinking level × each room (CPU, GPU) = 20 prompts × 4 levels × 2 rooms = **160 runs**
2. Log: `{room, model, thinking_level, prompt_id, prompt_category, total_latency_ms, thinking_tokens, answer_tokens, tg_tok_s}`
3. Score correctness independently (ideally blinded to condition)
4. Compute per-category:
   - Mean latency by thinking level
   - Mean correctness by thinking level  
   - Quality-adjusted throughput: `correctness_score / latency_seconds`

### Expected Outcomes

- **Simple tasks:** OFF ≈ LOW ≈ MED in correctness, OFF wins on latency → OFF is optimal
- **Complex tasks:** MED > LOW > OFF in correctness, with diminishing returns
- **CPU room:** Even LOW may be impractical for interactive use on hard tasks
- **GPU room:** LOW is the sweet spot for general-purpose; MED justified for reasoning tasks

### Decision Rule

For each task category, pick the thinking level that maximizes:
```
utility = correctness_score - α × latency_seconds
```
Where α = latency penalty weight (suggest α=0.1 for GPU, α=0.3 for CPU to reflect user patience).

---

*Last updated: 2026-02-12*
*For llmlab internal use. Hardware: dual RTX 3060 12GB (GPU room), CPU-only (CPU test room).*
