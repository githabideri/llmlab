#!/usr/bin/env bash
set -euo pipefail

STAMP=$(date -u +"%Y%m%dT%H%M%SZ")
LOGDIR="${LOGDIR:-benchmarks/openclaw/logs}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/llama-bench-$STAMP.log"

# capture env + command
{
  echo "# llama-bench log ($STAMP UTC)"
  echo "# host: $(hostname)"
  echo "# pwd: $(pwd)"
  echo "# cmd: $*"
  echo "# env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" 
  echo "# uname: $(uname -a)"
  echo "# ---" 
} | tee "$LOG"

# run and capture stdout/stderr
( set -x; "$@" ) >>"$LOG" 2>&1 || {
  echo "# exit: $?" >>"$LOG";
  exit 1;
}

