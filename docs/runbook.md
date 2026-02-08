# Runbook

## llama.cpp — start (Qwen3 Q2 current best)
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

## Health check
```bash
curl -sS http://127.0.0.1:8080/v1/models | head
```

## Common adjustments
- **Lower memory:** reduce `--ctx-size`, `-b/-ub`, or increase `--n-cpu-moe`
- **Balance GPUs:** adjust `--tensor-split A,B`

## Stop
```bash
pkill -x llama-server
```
