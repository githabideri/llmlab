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

See:
- `models/nemotron-3-nano-30b-a3b.md`
- `experiments/2026-02-12-nemotron-thinking-gradient-abc.md`
- `experiments/2026-02-12-nemotron-abc-executive-summary.md`

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
