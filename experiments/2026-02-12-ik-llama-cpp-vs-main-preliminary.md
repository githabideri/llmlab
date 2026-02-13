# ik_llama.cpp vs llama.cpp (preliminary wrap-up, handoff for tomorrow)

## Goal
Run a controlled benchmark pass on three replacement MoE targets in `ik_llama.cpp`, then mirror the same GPU suite on regular `llama.cpp` to estimate runtime delta.

## Setup
- GPU host: dual RTX 3060 12GB (`llama-cpp`)
- CPU host: 5-thread CPU-only node (`llama-local`)
- ik build: `1fdbc0d`
- main llama.cpp build: `91ea44e`
- Models:
  - `Qwen3-30B-A3B-Q4_K_M.gguf`
  - `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
  - `DeepSeek-Coder-V2-Lite-Instruct-Q5_K_M.gguf`

## Commands (high level)
- ik GPU sweep (`llama-sweep-bench`) with explicit dual-GPU flags:
  - `-ngl 999 --n-cpu-moe 0 -sm layer -ts 1,1 -dev CUDA0,CUDA1`
- ik CPU sweep (`llama-sweep-bench`) on `llama-local` with `-t 5`
- main llama.cpp GPU baseline (`llama-bench`) with matching model set and prompt sweep:
  - prompt points: `512, 4096, 8192, 16384, 32768`

## Observations
- ik dual-GPU path was validated as real offload (not CPU fallback):
  - `offloaded 49/49 layers to GPU` observed on Qwen runs
  - substantial CUDA buffers allocated on both GPUs
- Transient process/orchestration issues happened during setup (SIGKILL/timeouts/flag parsing hiccups), then stabilized.
- At wrap-up time, CT327 became intermittently unreachable via SSH (timeout during banner exchange), so the final baseline point collection was interrupted for live reporting.

## Metrics (captured so far)

### ik_llama.cpp — GPU (dual 3060)
- All three models reached `MODEL_DONE` in `/tmp/ik_gpu_ctx_degrade.log`:
  - Qwen3-30B-A3B
  - Qwen3-Coder-30B-A3B-Instruct
  - DeepSeek-Coder-V2-Lite-Instruct
- DeepSeek @ high fill (ctx 32768 run window):
  - TG roughly declines from ~111 tok/s (low fill) to ~65 tok/s near full fill.

### ik_llama.cpp — CPU (5 threads)
- Full CPU pass completed (`ik CPU bench v2 done`).
- DeepSeek @ ctx 8192:
  - TG starts around ~13.6 tok/s at low fill,
  - drifts mostly through ~9–12 tok/s,
  - with deeper-fill dips lower.

### llama.cpp (main) — GPU baseline (captured portions)
- Qwen3-30B-A3B (full prompt sweep completed):
  - TG32: `96.64` (p512), `96.31` (p4096), `96.06` (p8192), `95.67` (p16384), `94.02` (p32768)
- Qwen3-Coder-30B-A3B (full prompt sweep completed):
  - TG32: `95.90` (p512), `95.52` (p4096), `95.48` (p8192), `95.05` (p16384), `92.36` (p32768)
- DeepSeek-V2-Lite (partial captured before connectivity issue):
  - TG32: `28.35` (p512), `20.36` (p4096), `41.44` (p8192), `40.17` (p16384)
  - p32768 was in progress when host connectivity degraded.

## Conclusion (preliminary)
- ik GPU and CPU suites are both usable and produced the expected fill-degradation behavior.
- Mainline llama.cpp baseline is largely captured for Qwen models; DeepSeek baseline needs final verification/closure after host recovery.

## Next (tomorrow)
1. Reconnect to CT327 and confirm whether DeepSeek `p32768` finished in baseline log.
2. Produce final side-by-side table:
   - `runtime (ik vs main) × model × prompt/fill point`
   - delta in `%` for TG/PP.
3. Add a compact “operator recommendation” (fastest stable profile per model).
