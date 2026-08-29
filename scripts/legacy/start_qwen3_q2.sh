#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/models/gguf/qwen3-coder-next/Qwen3-Coder-Next-REAP-40B-A3B-Q2_K_XL.gguf}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} \
llama-server \
  --model "$MODEL_PATH" \
  --ctx-size 131072 -b 128 -ub 64 \
  --n-gpu-layers -1 --split-mode row --tensor-split 14,10 \
  --n-cpu-moe 0 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --flash-attn 1 --fit off --no-warmup --jinja \
  --host 0.0.0.0 --port 8080
