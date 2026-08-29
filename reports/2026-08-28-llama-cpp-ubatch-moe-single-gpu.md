# llama.cpp `-ub` (ubatch) tuning for MoE + MTP inference on a single 12 GB GPU

**Date:** 2026-08-28
**Category:** Experiment / benchmark
**Hardware:** RTX 3060 12 GB (compute 8.6), Intel i3-9100 4C/4T, 48 GB RAM, PCIe Gen3 x16
**Stack:** llama.cpp b9850 (4f31eed), CUDA, `llama-server`, single slot (`-np 1`)

---

## Goal

On a 12 GB single-GPU box serving a 35B-A3B MoE model (Q4_K_XL, 22.8 GB) with
`--n-cpu-moe 28` (12 expert layers on GPU, 28 on CPU), MTP draft decoding (n=2),
128K context, and `--no-mmap`, prompt-processing speed had stopped improving while
generation was fine. The question: is the ubatch (`-ub`) an untuned knob?

## Setup

Everything held constant across the sweep — only `-ub` changes:

```
llama-server -m Qwen3.6-35B-A3B-MTP-GGUF \
  -ngl 99 --n-cpu-moe 28 \
  -c 131072 -ctk q8_0 -ctv q4_0 \
  -b 4096 -ub <512|1024|1536|2048> \
  --flash-attn on --no-mmap -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 2
```

Prompts: tokenizer-verified synthetic prompts of ~8K and ~105K tokens (byte-truncated
code-like filler + a continuation instruction so the model generates the full
`n_predict`). 256 generated tokens, temperature 0, fixed seed. Single slot, no other
load. The 105K prompt puts the KV cache at realistic 128K-class fill, which matters
because the VRAM budget for the compute buffers shrinks as KV grows.

## Observations / metrics

| `-ub` | pp @8K | pp @105K fill | tg (both) | MTP acceptance @105K | peak VRAM |
|------:|-------:|-------------:|----------:|---------------------:|----------:|
| 512   | 345 t/s | — | 30.7 t/s | — | 10.7 GB |
| 1024  | 561 t/s | — | 34.7 t/s | — | 11.0 GB |
| **1536** (was production) | 676 t/s | 572 t/s | 30.0–30.6 t/s | 0.938 | 11.4 GB |
| **2048** | **836 t/s (+24%)** | **695 t/s (+21.6%)** | 30.3–31.9 t/s (neutral) | 0.943 | 11.8 GB |

- **Prompt processing scales super-linearly with ubatch** in this regime:
  512 → 1024 → 1536 → 2048 is 345 → 561 → 676 → 836 t/s at 8K.
- **Generation is unaffected** (within run-to-run noise) — MTP draft acceptance is
  unchanged (0.938 → 0.943).
- **VRAM cost is a near-constant ~340 MiB per ubatch step** (the compute/pp
  buffers), which fits here because only 1 slot is used and the 12 GB card had
  ~840 MiB of headroom at 1536.

## Why ubatch matters for MoE + CPU offload

Each ubatch is one graph evaluation: the GPU work for its tokens is batched into
fused kernels while the CPU-side expert compute (the 28 CPU layers, the dominant
cost per token) is amortized over the whole batch. Small ubatches mean more,
smaller graph launches and more per-batch CPU scheduling/sync overhead on a weak
4-core CPU — exactly the profile of this box. Larger ubatches also let flash
attention and the quantized GEMMs run at higher arithmetic intensity. The classic
counter-pressure is VRAM (compute buffers scale with batch×ubatch) and latency
granularity (a 2048 ubatch on a 128-token turn still processes in a few sub-batches,
so interactive latency is unaffected).

## Conclusion

For this configuration, **`-ub 2048` is the better production default**:
+21–24% prompt-processing speed at both short (8K) and long (105K) fills, with
neutral generation speed and MTP quality, at the cost of ~340 MiB VRAM. It was
adopted on the 128K production endpoint running this model.

## Caveats

- Single-slot serving (`-np 1`). With more concurrent slots, KV per slot shrinks
  and the VRAM math changes; re-measure before applying.
- 12 GB is near the ceiling: headroom went from ~840 to ~430 MiB. If you have more
  VRAM, sweep upward (3072, 4096) to find the knee — we did not, since 12 GB binds
  first. If you have less, 1536 (or 1024) may be your ceiling.
- Numbers are for Qwen3.6-35B-A3B UD-Q4_K_XL with `--n-cpu-moe 28` and MTP n=2.
  The direction (bigger ubatch → faster pp on CPU-offloaded MoE) should hold for
  other MoE models on similar hardware, but the optimum ubatch depends on the
  CPU/GPU split, quantization, and context — sweep per deployment.
- The sweep measured the whole prompt in one request; streaming-first-token
  behavior was unaffected at 2048 in practice, but if you need strict TTFT
  guarantees under load, verify with your real prompt distribution.

## Related

- [Qwen3.6-35B-A3B-MTP on Single RTX 3060 12GB](2026-06-30-qwen3.6-35b-a3b-mtp-single-3060.md) — the original deployment: `--n-cpu-moe` mechanics and sweep
- [Qwen3.6-35B-A3B on two RTX 3060s](2026-08-27-qwen3.6-35b-a3b-dual-3060-optimization.md) — the multi-GPU end of the same model family
- [KV cache sizing](../docs/kv-cache-sizing.md) — why 128K KV dominates the VRAM budget
