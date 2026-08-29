# llmlab Scripts

Small active tooling; the March 2026 context-ladder harness lives in `legacy/`.

## Active

- **`run_llama_bench_logged.sh`** — runs `llama-bench` while capturing env, command line, and full output to a timestamped log (default `benchmarks/legacy/openclaw/logs/`, override with `LOGDIR=`).
- **`fetch_model_info.py`** — model metadata fetcher for reports.
- **`forensics/.env.example`** — env template for the forensics runbook.
- **`results/`** — historical ladder outputs (data only, March 2026).

## legacy/

The context-ladder benchmarking from the March 2026 campaigns: `run_context_ladder.py` (+ `_fixed`), `run_concurrency_test.py`, `bench_ctx_sweep.sh`, `start_qwen3_q2.sh`, `example.env`. **Frozen** — superseded: current campaigns use per-campaign scripts documented in the [reports](../reports/). The [web UI](../web/) drives `run_context_ladder.py` and still works against this directory if needed.
