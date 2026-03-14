# 2026-03-14 — ik_llama.cpp Graph Mode on 3× RTX 3060 (Non-NVLink)

## Goal
Test ik_llama.cpp's `--split-mode graph` (true tensor parallelism) as a replacement for mainline llama.cpp's sequential layer mode on a 3× RTX 3060 12GB setup with mixed PCIe lanes.

## Setup

### Hardware
- 3× RTX 3060 12GB (36GB total)
- GPU0: PCIe x16 (CPU-direct), GPU1+GPU2: PCIe x4 (chipset)
- No NVLink, no P2P support between GPUs

### Software
- ik_llama.cpp v102 (commit `c2b8e95`)
- Model: Qwen3.5-35B-A3B Q4_K_M (~20GB)
- Vision projector: mmproj-f16 (858 MB)

### Serving config
```bash
llama-server \
  --split-mode graph \
  --ctx-size 393216 --parallel 3 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --ubatch-size 128 --flash-attn on --jinja
```

**Key constraint:** `--tensor-split` is incompatible with `--split-mode graph` (crashes with exp/shexp mismatch).

## Results

### Speed comparison (single-slot TG)

| Backend | TG @ 0K | TG @ 32K | Notes |
|---------|---------|----------|-------|
| Mainline (layer mode) | 59 tok/s | 48 tok/s | Stable at all depths |
| ik_llama.cpp (graph mode) | ~42–60 tok/s | ~46 tok/s | 3–4× faster than mainline's 13.5 tok/s baseline* |

*Mainline baseline was measured at `--parallel 3` with `--tensor-split`; graph mode doesn't use tensor-split and achieves higher throughput through true tensor parallelism.

### VRAM distribution (graph mode)
- GPU0: 11303/12288 MiB (tight — mmproj lives here)
- GPU1: 9949/12288 MiB
- GPU2: 10551/12288 MiB

### Benchmark data
- `scripts/results/20260314-121519-Qwen3.5-35B-A3B-Q4_K_M-ik_llama-graph.*` — 0–32K context ladder
- `scripts/results/20260314-134817-Qwen3.5-35B-A3B-Q4_K_M-ik-graph-no-reuse.*` — with `--no-graph-reuse`

## Bugs Found

### 1. VRAM Leak (graph reuse) — FIXABLE ✅
- Compute graph caching (`graph reuse`) leaks ~1GB VRAM per request
- **Fix:** `--no-graph-reuse` eliminates leak completely (80 requests, zero drift)
- Speed impact negligible
- Upstream: [ik_llama.cpp #1232](https://github.com/ikawrakow/ik_llama.cpp/issues/1232)

### 2. Host crash at 64K+ context — NOT FIXABLE ❌
- At 64K+ tokens, cross-GPU transfers trigger PCIe data corruption
- Corrupted GPU state → Xid 31/79 fault → PCIe bus hang → full host reboot
- Root cause: non-P2P PCIe topology (GPU1/2 on chipset x4 lanes)
- This is a **hardware limitation**, not a software bug

### Workarounds tested (all failed for 64K+)
- `--no-graph-reuse` — fixes leak, not crash
- NCCL 2.29.7 — already linked, no effect
- `--ubatch-size 64` — still crashes
- `ggml_set_inplace` patch from mainline #19816 — not applicable (ik uses different code path)

## Mainline Baseline (same day)

For comparison, ran full context ladder + concurrency on mainline llama.cpp (layer mode):

| Metric | 0K | 32K | 64K | 128K |
|--------|-----|------|------|-------|
| PP (tok/s) | 67 | 168 | 308 | 594 |
| TG (tok/s) | 59 | 48 | 42 | 33 |

Concurrency (layer mode): **severe slot starvation** — one slot gets 37–57 tok/s, others freeze at 0.4 tok/s. This is the fundamental layer-mode sequential GPU processing limitation.

Results: `scripts/results/20260314-142906-*-mainline-layer.*`, `scripts/results/20260314-144044-*-mainline-layer-concurrency.*`

## Conclusion

**Graph mode delivers real 3–4× speedup** but is unusable on non-P2P PCIe topologies at 64K+ context. The VRAM leak is fixable; the host crash is not (without hardware change).

**Hardware fix:** CPU-direct PCIe for all GPUs (e.g., Threadripper/HEDT platform with 64 lanes, or reducing to single large GPU like RTX 3090 on x16).

## Related
- Stability investigation: private workspace notes
- Upstream issues: [#1232](https://github.com/ikawrakow/ik_llama.cpp/issues/1232), [#854](https://github.com/ikawrakow/ik_llama.cpp/issues/854)
- Mainline reference: [ggml-org/llama.cpp #20052](https://github.com/ggml-org/llama.cpp/issues/20052) (non-P2P PCIe corruption)
