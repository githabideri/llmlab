# llmlab Scripts

Reusable benchmarking and testing scripts for local LLM evaluation.

## Context Ladder Benchmark

Tests model performance across various context lengths (0-250K tokens).

### Setup

1. Copy example config:
```bash
cd scripts
cp example.env .env
```

2. Edit `.env` with your settings:
   - `LLAMA_ENDPOINT`: Your llama.cpp server URL
   - `LLAMA_SSH_HOST`: SSH alias/hostname for monitoring
   - `MODEL_NAME`: Model identifier for output files

3. Make script executable:
```bash
chmod +x run_context_ladder.py
```

### Usage

```bash
# Run with default .env config
./run_context_ladder.py

# Or specify custom config
LLAMA_ENDPOINT=http://other-host:8080/v1/completions ./run_context_ladder.py
```

### Output

Creates three files in `results/` directory:

- **JSON** (`YYYYMMDD-HHMMSS-model-name.json`): Complete data with all metrics
- **CSV** (`YYYYMMDD-HHMMSS-model-name.csv`): Spreadsheet-friendly format
- **Markdown** (`YYYYMMDD-HHMMSS-model-name.md`): Human-readable report

### Metrics Captured

- **Performance:** PP/TG tok/s at each context level
- **Timing:** Total time, prompt/generation milliseconds
- **CPU/RAM:** Process CPU % and memory usage (via SSH)
- **GPU:** VRAM usage, utilization, power draw (via nvidia-smi)

### Test Points

Default ladder (13 points):
- 64, 3K, 6K, 10K, 13K, 19K, 24K, 32K, 64K, 95K, 128K, 192K, 250K tokens

Edit `DEFAULT_TEST_POINTS` in script to customize.

### Requirements

- Python 3.6+
- `curl` command available
- SSH access to llama.cpp host
- `nvidia-smi` on target host (for GPU stats)
- Optional: `python-dotenv` for .env file support

```bash
pip install python-dotenv  # optional but recommended
```

### Example Output

```
Testing ~6400 tokens context...
    6410 ctx: PP=1257.70 tok/s, TG= 58.80 tok/s (CPU: 95.2%, RAM: 2834.5 MB)
```

### Notes

- Each test uses `cache_prompt=false` for accurate cold measurements
- Small delay (2s) between tests to let system stabilize
- Failed tests are skipped and logged
- All timestamps in UTC
