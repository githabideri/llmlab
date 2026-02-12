# Runbook

## llama.cpp — start (Nemotron recommended profile)

```bash
llama-server \
  --model /path/to/Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf \
  --ctx-size 120000 \
  --parallel 1 \
  --slot-save-path /path/to/slots \
  --split-mode row --tensor-split 1,1 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --flash-attn on \
  --jinja \
  --reasoning-format deepseek \
  --reasoning-budget -1 \
  --host 0.0.0.0 --port 8080
```

## Alternate speed mode (lower trust)

```bash
# Same as above, but:
--reasoning-budget 0
```

Use only when latency matters more than output cleanliness.

## Health check

```bash
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/v1/models | head
```

## Common adjustments
- **Lower memory:** reduce `--ctx-size`, reduce `--parallel`, or tune KV cache types
- **Balance GPUs:** adjust `--tensor-split A,B`
- **Reasoning volume:** keep `--reasoning-format deepseek`; tune prompt policy before forcing budget 0

## Stop

```bash
pkill -x llama-server
```

## Legacy benchmark profile (Qwen REAP 40B)

```bash
llama-server \
  --model /mnt/models/gguf/qwen3-coder-next/Qwen3-Coder-Next-REAP-40B-A3B-Q2_K_XL.gguf \
  --ctx-size 131072 -b 128 -ub 64 \
  --n-gpu-layers -1 --split-mode row --tensor-split 14,10 \
  --n-cpu-moe 0 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --flash-attn 1 --fit off --no-warmup --jinja \
  --host 0.0.0.0 --port 8080
```
