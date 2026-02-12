# Nemotron Thinking Gradient A/B/C Tuning (2026-02-12)

## Goal
Find a practical middle ground for Nemotron reasoning output (target ~1–2:1 reasoning:text) without killing quality.

## Setup
- Host: `llama-cpp` (2x RTX 3060)
- Model: `Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf`
- Server flags baseline:
  - `--reasoning-format deepseek`
  - tested with `--reasoning-budget -1` and `--reasoning-budget 0`
- Prompt set (5 prompts): greeting, factual, REST-vs-GraphQL, arithmetic, practical NAS backup plan.

## Config Matrix
### Budget = -1 (reasoning enabled)
- **A_baseline_on**: neutral system prompt
- **B_brief_on**: system prompt constrains reasoning to max 1–2 short sentences
- **C_prompt_off**: system prompt `detailed thinking off`

### Budget = 0 (reasoning disabled server-side)
- **A_baseline_budget0**
- **B_brief_budget0**
- **C_prompt_off_budget0**

## Results

### 1) Budget -1 (weighted ratio across full run)
| Config | Reasoning tokens | Visible tokens | Weighted ratio | Avg latency | Quality proxy |
|---|---:|---:|---:|---:|---:|
| A_baseline_on | 573 | 390 | **1.47** | 3.36s | 3/5 |
| B_brief_on | 373 | 263 | **1.42** | **2.31s** | **5/5** |
| C_prompt_off | 472 | 412 | **1.15** | 3.11s | **5/5** |

Observations:
- All three are already near target band in weighted ratio.
- **B_brief_on** gave the best speed/quality balance in this prompt set.
- No visible `</think>`/planner leakage in this run set.

### 2) Budget 0
| Config | Reasoning tokens | Visible tokens | Weighted ratio | Avg latency | Quality proxy | Leak count |
|---|---:|---:|---:|---:|---:|---:|
| A_baseline_budget0 | 0 | 562 | 0.00 | 2.04s | 4/5 | 3/5 |
| B_brief_budget0 | 0 | 325 | 0.00 | **1.27s** | 5/5 | 2/5 |
| C_prompt_off_budget0 | 0 | 485 | 0.00 | 1.77s | 5/5 | 3/5 |

Important caveat:
- Despite zero extracted reasoning tokens, multiple outputs still leaked malformed/stray reasoning artifacts (e.g. duplicated answer with `</think>`, and one planner-text style answer).
- So budget=0 is fastest, but still not fully “clean” for chat output quality.

### 3) `chat_template_kwargs` probe (budget -1)
Quick probe with `chat_template_kwargs: {"reasoning_effort": ...}` (2 prompts):

- Prompt: REST vs GraphQL
  - base: 2.24
  - low: **1.17**
  - medium: 1.70
  - high: 1.99
- Prompt: NAS backup plan
  - base: 0.27
  - low: 0.44
  - medium: 0.43
  - high: 0.31

Interpretation:
- There is **some** signal that `reasoning_effort=low` can reduce reasoning on reasoning-heavy prompts.
- Effect is inconsistent across prompt types; not yet reliable as a sole control.

## Conclusion
Best current “less not none” candidate:
1. Keep server at `--reasoning-budget -1`
2. Use constrained system prompt (B profile: brief reasoning cap)
3. Optionally test `chat_template_kwargs.reasoning_effort=low` per workload
4. Keep strict output sanitization guardrails in client path (to catch malformed leaked planner/tag artifacts)

Best speed-only mode:
- `--reasoning-budget 0` is fastest, but currently still shows occasional output artifacts; not ideal for high-trust factual chat.

## Artifacts
- Raw run data:
  - `llmlab/experiments/2026-02-12-nemotron-abc-budget-minus1.json`
  - `llmlab/experiments/2026-02-12-nemotron-abc-budget-0.json`
