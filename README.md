# llmlab

Public notebook for a practical local-LLM lab: reproducible configs, benchmark methods, and operator-grade run notes.

## What this repo is for
- Keep working inference configs from getting lost
- Compare model/runtime changes with repeatable measurements
- Share what actually works (and what breaks) in small-GPU setups

## Current operating profile (2026-02-12)

### Primary model/runtime
- **Model:** Nemotron-3-Nano-30B-A3B (IQ4_NL GGUF)
- **Runtime:** llama.cpp (`llama-server`)
- **Hardware class tested:** dual 12GB GPUs + CPU fallback node

### Recommended reasoning profile (“less, not none”)
- **Server:** `--reasoning-format deepseek --reasoning-budget -1 --jinja`
- **Prompt/system style:** brief constrained reasoning (B profile)
- **Why:** best observed balance of speed + quality in A/B/C tests

### Speed-only profile
- `--reasoning-budget 0` is faster, but can leak malformed planner/tag artifacts in output.

### At-a-glance profile table

| Profile class | Core setting | Strength | Main tradeoff |
|---|---|---|---|
| `reduced-thinking-balanced` | `--reasoning-budget -1` + constrained brief reasoning | Better speed/quality balance | Still some hidden reasoning token overhead |
| `non-thinking-speed` | `--reasoning-budget 0` / think-off style | Lowest latency | Higher risk of malformed planner/tag leakage and weaker factual robustness |

### Run-comparison rule (important)
- Keep profile classes separate when reading results:
  - `non-thinking-speed` (budget `0` / think-off style)
  - `reduced-thinking-balanced` (budget `-1` + constrained brief reasoning)
- Do **not** mix these into one aggregate performance/quality claim.

**Run-tag example (use in notes/experiments):**

```yaml
profile_class: reduced-thinking-balanced
reasoning_budget: -1
reasoning_style: constrained-brief
```

### Recent postmortem snapshot (Feb 12, reduced-thinking era)
- Session type: chat-orchestration heavy (“wild ride”), not a pure quality benchmark
- Tool-call reliability: **33 calls**, **31 success**, **2 failures**
  - one channel-resolution (`message`) error
  - one killed monitored process (`process`, SIGKILL)
- Takeaway: reliability remained high overall; failures were environment/control-plane class.

See:
- `models/nemotron-3-nano-30b-a3b.md`
- `experiments/2026-02-12-nemotron-thinking-gradient-abc.md`
- `experiments/2026-02-12-nemotron-abc-executive-summary.md`

## In-progress benchmark campaign (ik_llama.cpp → llama.cpp baseline)

A replacement-model benchmark pass is now underway for:
- Qwen3-30B-A3B
- Qwen3-Coder-30B-A3B-Instruct
- DeepSeek-Coder-V2-Lite-Instruct

Current state:
- ik GPU suite: complete on all three models (dual 3060)
- ik CPU suite: complete
- regular llama.cpp GPU baseline: mostly complete; final DeepSeek point pending confirmation after a CT327 connectivity interruption

Working notes:
- `experiments/2026-02-12-ik-llama-cpp-vs-main-preliminary.md`

## Repo map

```text
README.md
LICENSE
/docs
  overview.md
  architecture.md
  runbook.md
  benchmarks.md
  troubleshooting.md
  thinking-policy.md
/models
  nemotron-3-nano-30b-a3b.md
/experiments
  README.md
  2026-02-09-nemotron-thinking-fix.md
  2026-02-12-nemotron-thinking-gradient-abc.md
  2026-02-12-nemotron-abc-executive-summary.md
/scripts
  start_qwen3_q2.sh
  bench_ctx_sweep.sh
/config
  llama-server.example.env
```

## Safety note
This repo is public. Do **not** commit:
- private hostnames/IPs
- credentials/tokens/SSH keys
- personal transcripts or sensitive logs

Keep examples generic and reusable.
