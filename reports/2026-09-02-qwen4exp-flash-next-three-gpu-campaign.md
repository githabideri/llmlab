# Qwen4Exp Flash-Next on Three Consumer GPUs — Two-Stage Campaign

**Date:** 2026-09-02 (overnight campaign, ~13 h, two gated stages, 87 launch attempts / 46 complete runs)  
**Machine:** GPU server (Ryzen 5 5600X, 64 GB RAM) with three Ampere cards: RTX 3060 12 GB (chipset, PCIe 4.0 x4), RTX 3060 12 GB (CPU root port, x8 negotiated), RTX 3090 24 GB (CPU x16)  
**Model:** Qwen3.8-Flash-Next — the "Qwen4Exp" MoE: 125 B total / 6 B active per token, 48 layers, plus a 51 B-parameter per-layer lookup table (PLE)  
**GGUF:** Unsloth `Qwen3.8-Flash-Next-UD-Q2_K_XL` (3 shards, 78.9 GB) + `mtp-…-shared-Q8_0` draft (2.8 GB)  
**Status of result:** tested, **not** in production (see Conclusion)

---

## Goal

1. **Stage 1 — baseline.** First ever run of this architecture on consumer hardware. Establish the GPU-layer residency knee, a tensor-split comparison, a 3-rep determinism baseline, and per-token PCIe / SSD / energy numbers at a fixed 2K-prompt / 256-token shape.
2. **Stage 2 — hypothesis test.** Stage 1's prefill was terrible (72 tokens/s) because whole layers were spilling to RAM (each spilled layer drags its ~0.9 GB of experts with it). Test whether **selective expert spill** — keeping all 48 layers' non-expert weights on the GPUs while the fitter streams a few layers' experts from RAM — can kill the prefill penalty, and measure the secondary knobs (ubatch, threads, PLE direct-read, MTP, n-gram).

## Setup

- **Engines:** Stage 1 on llama.cpp at the exact PR #27742 merge commit (`6c84c7d5`) — the first mainline build that can load the new `qwen4exp` GGUF (v3 header format). The box's production build (b10642) **cannot load this model at all** (unknown architecture). Stage 2 on current master `b81c99b4` after a parity check, plus a PR #28136 build (`90555796`) for the direct-read comparison.
- **Method (fixed for every cell):** 32 K context, q8_0 KV cache, flash attention, `--parallel 1`, deterministic prompt fixture (~2,113 tokens, nonce-prefixed per pass, temp 0 / seed 1234), 256 completion tokens, one warm pass then one measured pass per run.
- **Instrumentation per run:** `nvidia-smi dmon` at 1 Hz (per-GPU power, SM %, framebuffer, PCIe RX/TX), `vmstat` 1 Hz (page cache, swap, SSD sectors), `iostat`, request-start/end epoch markers, server log with per-request timings.
- **Page cache was never dropped** (host-level `drop_caches` needs the hypervisor; not permitted unattended). Most runs are therefore cache-warm for the PLE table; "cold" means first-touch of a given prompt's rows, not an empty OS cache. This bounds the cold-prefill story (see Observations 6).
- **Topology gotcha (bitten, fixed):** CUDA's default `FASTEST_FIRST` device ordering silently mapped the single-GPU cells to the wrong physical cards. Exporting `CUDA_DEVICE_ORDER=PCI_BUS_ID` and addressing everything by bus ID is mandatory on this box — two early cells had to be invalidated and re-run.

### The two champion commands

```bash
# Stage-1 champion (build 6c84c7d5; explicit PLE pin, layer split)
llama-server --device CUDA0,CUDA1,CUDA2 \
  -m /mnt/models/gguf/qwen3.8-flash-next-ud-q2/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  --override-tensor per_layer_token_embd.weight=CPU --load-mode mmap \
  --split-mode layer --tensor-split 1,1,2 --gpu-layers 44 \
  --ctx-size 32768 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 2048 --ubatch-size 256 \
  --flash-attn on --jinja

# Stage-2 finalist (master b81c99b4; NO placement flags — the fitter decides)
/opt/llama.cpp-master/build/bin/llama-server --device CUDA0,CUDA1,CUDA2 \
  -m /mnt/models/gguf/qwen3.8-flash-next-ud-q2/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  --ctx-size 32768 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 4096 --ubatch-size 1024 \
  --threads 6 --threads-batch 12 \
  --flash-attn on --jinja
# + MTP:  --spec-type draft-mtp --spec-draft-model /mnt/models/gguf/qwen3.8-flash-next-ud-q2/mtp-...-Q8_0.gguf
```

## Observations

1. **The model fits.** Non-PLE weights total 45.9 GB (batched experts 42.9 GB = 0.89 GB/layer, per-layer non-expert only 2.95 GB) — inside 48.6 GB aggregate VRAM. The 51 B PLE table stays in RAM/SSD and is streamed row-by-row; per measured token it costs only **10–50 KB of PCIe RX and 2–14 KB of SSD** (median, decode window) — the table is served from page cache, the SSD only backfills.
2. **Stage-1 knee at 44 layers.** Decode rose monotonically 9.2 → 21.3 t/s from ngl 24 → 44 (3090 at 21.6/24 GB at 44); ngl 48 and ngl all OOM on the 3090 (single >24 GB allocation). Best tensor split was plain `1,1,2` (23.6 t/s) over `1,1,2.2` and `1,1,2.4`.
3. **The master build's fitter is the whole game.** Giving master *no* placement flags at all (`common_fit_params`) auto-produces its own selective expert spill (VRAM 11.2 / 10.7 / 23.0 GiB) — decode **33.4 t/s** (+46 % vs the same config on master with explicit flags) and prefill 206 → ~390 with ubatch 1024.
4. **Explicit placement knobs are counterproductive on this build.** Any `--gpu-layers` or `--tensor-split` aborts the fitter ("tensor_split already set by user"); `--cpu-moe` / `--n-cpu-moe N` then misbalance buffers and demand 12.9 GB from a 12 GB card, every time. And master uses **only the last `--override-tensor` flag** (deprecation: comma-separated) — a combined flag is required. The fitter is currently the only working path to selective spill.
5. **PLE needs no override on master** — it is auto-placed to CPU. The stage-1 `--override-tensor …=CPU` pin was only needed on the #27742 build.
6. **Prefill is cache-bound, not compute-bound, and the "5.4×" needs a footnote.** Prefill speed on this model is dominated by PLE row faults: on a cold-ish cache it ran at 34–174 tokens/s, fully warmed it ran 387–437. The stage-1→finalist "5.4×" mixes (a) a faster master PLE path (~2.4×), (b) fitter placement (~1.5–2× via ub1024 geometry), and (c) residual page-cache state. Treat cross-stage pp/s comparisons as order-of-magnitude.
7. **Warm-cache decode is ~40 t/s** (0.17 s TTFT on a cache hit) in both mmap and `--lazy-mode on-direct` PLE modes — PR #28136 showed **no measurable advantage** here because the cache never went cold.
8. **Speculation: MTP works, n-gram does not.** `draft-mtp` with the shared Q8_0 draft: **35.3 t/s (+17 %)**, clean. N-gram speculation is content-toxic: 29.2 t/s on prose (neutral) but **0.13 t/s on the code fixture** — effectively dead. If n-gram is ever enabled router-side, gate it away from code.
9. **Secondary knobs:** ubatch 256→1024 roughly doubled prefill on fitter placement at a ~1–3 t/s decode cost (2,048/4,096 give no further gain); host threads 6 / batch-threads 12 was the best of three (32.8 vs 29.0/29.1); `--fit-target` margin 512–3072 MiB barely changes the fitter's placement; `GGML_CUDA_FORCE_MMQ=1` neutral; flash-attn-off is structurally impossible with q8_0 KV.

## Metrics

### Stage 1 (build `6c84c7d5`, explicit PLE pin, layer split, ub256) — decode t/s

| Config | Decode | Prefill | TTFT |
|---|---|---|---|
| ngl 24 / 28 / 32 / 36 (ALL3) | 9.2 / 10.0 / 11.6 / 13.5 | 29–38 | 56–78 s |
| ngl 40 / 44 (ALL3) | 17.5 / 21.3 | 52 / 72.5 | 41 / 29 s |
| ngl 48 / all | OOM (3090) | — | — |
| **best tensor split `1,1,2` @ ngl44** | **23.6** | 72.5 | 29.1 s |
| champion reps ×3 | 21.5 / 21.2 / 21.0 (median 21.2) | 72.4–72.9 | 29.1 s |
| topology @ common ngl32: 3090+x8 / all3 / 3090+x4 / 3090-only | 12.2 / 12.0 / 11.9 / 9.7 (OOM>24) | 37–45 | 47–61 s |

### Stage 2 (master `b81c99b4`, fitter auto-placement, b4096/ub1024, t6/tb12) — decode t/s / prefill

| Config | Decode | Prefill | TTFT |
|---|---|---|---|
| 2A: stage-1 champion config, on master | 22.9 | 174.6 | 12.4 s |
| 2B: **fitter auto-placement** (ub256) | **33.4** | 206.3 | 10.5 s |
| fit-target 512 / 2048 / 3072 | 31.7 / 33.1 / 32.7 | ~390 | 5.6 s |
| ubatch 256 / 512 / 1024 / 2048 / 4096 | 32.4 / 31.8 / 31.0 / 30.6 / 29.1 | 207 / 148 / 170 / 437 / 177 | 5.1–14.6 s |
| **finalist reps ×3** (median) | **30.2** (29.0–30.3) | **~390** | 5.6–5.7 s |
| finalist + MTP (draft-mtp) | **35.3** | 336 | 6.6 s |
| warm-cache decode (both PLE modes) | ~40 | (cache hit) | 0.17 s |
| PLE direct-read (PR #28136) vs mmap | 386.4 / 411.9 vs 389.7 / 409.0 pp (cold/diverse) | no measurable edge | — |

### Per-token data movement (decode window, 1 Hz dmon/vmstat, medians)

| Quantity | Stage 1 (ngl44) | Stage 2 finalist |
|---|---|---|
| PCIe RX (steady) | 37–200 MB/s | **37–42 MB/s** |
| PCIe RX per token | 0.05–1.5 MB | **0.01–0.05 MB** |
| SSD reads per token | 5–18 KB | **2–14 KB** |
| GPU power (3 cards, decode) | ~150–450 W (noisy 1 Hz) | **68–142 W** |
| Energy per 1K tokens (GPU-only, above nothing — absolute) | ~2–15 Wh/1K | **~0.5–1.7 Wh/1K** |

At Austrian list price (~€0.30/kWh) that is on the order of **€0.2–0.5 per 1M generated tokens in decode**, excluding the box's ~30–60 W idle floor (no wall-plug meter on this box — GPU power sum only).

**Determinism:** champion reps 21.0–21.5 t/s (±1.5 %); finalist reps 28.98–30.26 (±2 %). One stage-1 rep's first attempt was invalidated (host swap touched 170 MiB during model page-in) and cleanly re-run; the campaign enforced no-swap-or-restart.

## Conclusion

- The Qwen4Exp Flash-Next architecture **runs well on three consumer Ampere cards** once the model's non-PLE weights (45.9 GB) are all kept resident: **30 t/s decode / ~390 prefill tokens/s / 35 t/s with MTP**, sub-cent energy per 1K tokens, and only a trickle of PCIe/SSD traffic per token for the 51 B table.
- On this build generation, **let the fitter place the model** — every explicit placement knob (`-ngl`, `--tensor-split`, `--cpu-moe`) is either counterproductive or unavailable. Stage-1's whole-layer knee (ngl 44) was simply the best possible within explicit-flag mode.
- **Production deployment is on hold:** the fitter config uses all three cards, i.e. it cannibalizes the 3090 that currently serves vLLM in production. Deciding between "one 27 B dense" and "one 125 B MoE at 30 t/s" is an owner call.
- **Next:** supervised true-cold PLE test (`drop_caches` on the hypervisor) to give PR #28136 a fair shot; a longer MTP soak before any production use (upstream has reported recurrent-state rollback bugs); and the SGLang lane (native MTP + PLE offload) as the second engine for comparison.

## References

- llama.cpp PR #27742 (Qwen4Exp support, merge `6c84c7d5`, 2026-08-27); PR #28136 (direct-read lazy PLE); issues #27964, #27953, #27835
- Unsloth `Qwen3.8-Flash-Next-GGUF` (Q2_K_XL, MTP shared-Q8_0)
- Full per-run artifacts (dmon, vmstat, iostat, pidstat, server logs, manifests): retained on the GPU server's model volume under `bench/2026-09-02-qfn-stage1|2/` (private repo; see homelab report `2026-09-02_benchmark_qfn-3gpu-campaign.md`)
