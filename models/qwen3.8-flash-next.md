# Qwen3.8-Flash-Next (Qwen4Exp) on 3× Consumer GPUs

**Model:** Qwen3.8-Flash-Next — "Qwen4Exp" MoE: 125 B total, ~6 B active per token, 48 layers, plus a 51 B-parameter per-layer lookup table (PLE; `per_layer_token_embd.weight`) that is streamed per-token from host memory  
**Tested Quantization:** Unsloth `UD-Q2_K_XL` (3-shard GGUF, 78.9 GB) + `mtp-…-shared-Q8_0` draft (2.8 GB). GGUF **v3** header format — only llama.cpp at/after PR #27742 (`6c84c7d5`) can load it.  
**Hardware:** GPU server, 3 cards: RTX 3060 12 GB (chipset, PCIe 4.0 x4), RTX 3060 12 GB (CPU, x8), RTX 3090 24 GB (CPU, x16)  
**Runtime:** llama.cpp master `b81c99b4` (fitter auto-placement), PR #28136 build retained for direct-read PLE  
**Status:** ⚠️ **Tested, NOT in production** — the config uses all three GPUs, which collides with the 3090's production vLLM service. Frozen test record; see the campaign report for the full numbers.

---

## Quick Facts

| Parameter | Value |
|-----------|-------|
| **Architecture** | `qwen4exp` (MoE, per-layer PLE lookup, SSM/hyper-connection/indexer subsystems in the GGUF) |
| **Parameters** | 125 B total / ~6 B active; PLE table 51 B params (RAM/SSD-resident) |
| **Layers** | 48 |
| **GGUF size** | 78.9 GB (UD-Q2_K_XL) — non-PLE weights 45.9 GB (experts 0.89 GB/layer), fits 48.6 GB aggregate VRAM |
| **Context tested** | 32,768 (32K; 128K/262K gated on RAM) |
| **Best measured** | **30.2 t/s sustained decode** (finalist median — the canonical number; 35.3 t/s with MTP is a single clean run, promising, not established) · **~390 prefill t/s under warm-cache conditions only** (34–437 across PLE page-cache states) · ~40 t/s on a prompt-cache hit (a different serving regime, not the model's decode speed) |
| **Per-token traffic (decode)** | PCIe RX 10–50 KB · SSD 2–14 KB · GPU power 68–142 W → **~0.5–1.7 Wh/1K tokens, GPU-only** (dmon sum, no idle subtraction, no wall meter) |
| **Speculative decoding** | MTP: +17 % (single run — promising, not established; soak before any production use). N-gram: pathological on the tested code fixture (0.13 t/s; prose neutral) — do not enable router-wide without content gating. |

---

## Config (fitter auto-placement — the only working path on this build)

**KV Cache:** q8_0 / q8_0 · **Batch:** `-b 4096 -ub 1024` · **Threads:** 6 / batch 12 · single slot

```bash
/opt/llama.cpp-master/build/bin/llama-server \
  --device CUDA0,CUDA1,CUDA2 \
  -m /path/to/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  --ctx-size 32768 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 4096 --ubatch-size 1024 \
  --threads 6 --threads-batch 12 \
  --flash-attn on --jinja
# optional MTP: --spec-type draft-mtp --spec-draft-model /path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
```

- **No `--gpu-layers`, no `--tensor-split`, no `--override-tensor`.** The fitter (`common_fit_params`) auto-places: all 48 layers' non-expert weights + shared experts on the GPUs, selective expert spill to RAM, PLE auto-pinned to CPU (VRAM landed at 11.2 / 10.7 / 23.0 GiB).
- On this build, any explicit placement flag **disables** the fitter and the resulting hand-balancing OOMs (12.9 GB demanded from a 12 GB card). `--cpu-moe` / `--n-cpu-moe N` fail the same way.
- Only the **last** `--override-tensor` flag is honored (deprecation: use comma-separated values) — if you must pin tensors, combine them in one flag.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` is required on this box (default FASTEST_FIRST maps cells to the wrong physical cards).
- Fitter VRAM margin knob: `--fit-target <MiB>` (default 1024; 512–3072 made no difference in testing).

## Stage-1 baseline (for reference)

Explicit-flag mode (build `6c84c7d5`): `--gpu-layers 44 --split-mode layer --tensor-split 1,1,2` with `--override-tensor per_layer_token_embd.weight=CPU --load-mode mmap` → 23.6 t/s decode / 72 prefill. Whole-layer spill is the prefill killer; the knee is ngl 44 (3090 nearly full), ngl 48+ OOMs.

## Caveats

- Prefill numbers are **page-cache-state dependent** (34–437 t/s across cache states); a true-cold test needs hypervisor-level `drop_caches`.
- MTP was verified on a single clean 256-token run (+17 %); longer soak recommended before production.
- Energy figures are the sum of the three GPUs' dmon power readings (no wall-plug meter on this box).

## Links

- Campaign report: [`reports/2026-09-02-qwen4exp-flash-next-three-gpu-campaign.md`](../reports/2026-09-02-qwen4exp-flash-next-three-gpu-campaign.md)
- Methodology: [`docs/benchmarks.md`](../docs/benchmarks.md), [`docs/multi-gpu-model-placement.md`](../docs/multi-gpu-model-placement.md)
- llama.cpp PR #27742 (Qwen4Exp), PR #28136 (direct-read lazy PLE)
