#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/models/gguf/qwen3-coder-next/Qwen3-Coder-Next-REAP-40B-A3B-Q2_K_XL.gguf}"

llama-bench \
  -m "$MODEL_PATH" \
  -p 32768,65536,98304,131072 -n 256 \
  -b 128 -ub 64 -ctk q8_0 -ctv q4_0 -fa 1 \
  -ngl 99 -sm row -ts 14/10 -ncmoe 0 -t 4 \
  -r 1 --no-warmup -o md
