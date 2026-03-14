# Qwen3.5-35B-A3B

**Base model:** [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)  
**Architecture:** DeltaNet linear attention + MoE (35B total, ~3B active)  
**Quant used:** `Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf` (≈20 GB on disk)  
**Vision projector:** `mmproj-Qwen_Qwen3.5-35B-A3B-f16.gguf` (858 MB)

## Quick Facts

| Param | Value |
|-------|-------|
| Total parameters | 35B |
| Active parameters | ~3B (8 routed + 1 shared expert) |
| KV cache | ~7.5 KiB/token (10 attention layers) |
| Tested hardware | 2× RTX 3060 12GB (24GB total) |
| Current stable profile | `ctx=98,304`, `-sm layer`, `--no-mmproj-offload`, `parallel=1` |
| Text+tools+vision on 24GB | ✅ **Works** (with tuned profile) |

## What changed (important)

### Earlier result (2026-02)
- We observed **vision OOM** while trying to run Q4_K_M + mmproj on 24GB.
- We also observed repeated tool-call loops in real agentic sessions.

### Retest result (2026-03-03)
- We re-ran Qwen3.5 as a single text+tools+vision model on the same 24GB setup.
- It **fits and runs** when using a tighter memory profile:

```bash
llama-server \
  --model /mnt/models/gguf/qwen3.5-35b-a3b/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --mmproj /mnt/models/gguf/qwen3.5-35b-a3b/mmproj-Qwen_Qwen3.5-35B-A3B-f16.gguf \
  --no-mmproj-offload \
  --ctx-size 98304 \
  --parallel 1 \
  --split-mode layer \
  --gpu-layers 99 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --flash-attn on --jinja
```

## Why it fits on 24GB now

The key was not changing the model quant; it was changing runtime pressure points:

1. **Lowered context to 96k (`--ctx-size 98304`)**  
   Lower KV/cache footprint than 131k+ profiles.

2. **Used `split-mode layer`**  
   Better memory balance for this dual-3060 PCIe setup.

3. **Kept `parallel=1`**  
   Avoids extra slot overhead.

4. **Used `--no-mmproj-offload`**  
   Prevents additional projector-offload pressure on already-tight VRAM.

Observed runtime memory during active service stayed just under the cliff:
- GPU0: ~11.9 GB / 12 GB
- GPU1: ~11.3 GB / 12 GB

## Performance

### Production Serving (March 2026)

Tested on 3× RTX 3060 12GB (36GB total) with native thinking mode (no `--reasoning-format` flag).

**Configuration:**
```bash
--ctx-size 262144 --parallel 1 --split-mode layer 
--gpu-layers 99 --cache-type-k q8_0 --cache-type-v q4_0 
--flash-attn on --jinja
```

**15-minute production session** (text + web research tasks):

| Metric | Value | Range |
|--------|-------|-------|
| **PP speed** | 835.9 tok/s avg | 544-1016 tok/s |
| **TG speed** | 42.4 tok/s avg | 35-50 tok/s |
| **Total requests** | 12 | |
| **Context range** | 21K → 64K | No compaction needed |

### Speed Degradation by Context

| Context Range | Avg TG Speed | Change from Baseline |
|---------------|--------------|---------------------|
| **<25K** | 49.4 tok/s | Baseline |
| **36-40K** | 42.3 tok/s | -14% |
| **>40K** | 39.7 tok/s | -20% |

**Key findings:**
- Clean 20% degradation curve from fresh to 64K context (predictable, stable)
- Web content fetches caused +14K and +23K context spikes
- Thinking token overhead ~15% (all responses included thinking content)
- No crashes, hangs, or errors during extended session

### Vision Processing

Vision preprocessing time scales with input resolution:
- ~128px image class: ~0.3 s
- ~512px image class: ~5.3 s
- ~1024px image class: ~32 s

**Operational takeaway:** Keep interactive images small/medium unless high latency is acceptable.

### Benchmark Comparison

| Metric | Expected (llama-bench) | Actual (serving) | Delta |
|--------|------------------------|------------------|-------|
| PP speed | 900-1100 tok/s | 836 tok/s | -7% to -24% |
| TG speed (fresh) | 55-60 tok/s | 49 tok/s | -10% to -18% |
| TG speed (filled) | 45-50 tok/s | 35 tok/s | -22% to -30% |

Delta explained by: server overhead + thinking tokens + context depth overhead.

## vLLM PP=3 profile (3× RTX 3060)

### Tested model/engine
- **Model:** `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (official)
- **Engine:** vLLM 0.17.1
- **Parallelism:** `--pipeline-parallel-size 3`
- **KV dtype:** `fp8_e4m3`

### Stable high-throughput profile

```bash
--max-model-len 131072 \
--max-num-seqs 3 \
--max-num-batched-tokens 8192 \
--performance-mode interactivity \
--mamba-cache-mode align --mamba-block-size 8 \
--enable-prefix-caching --enable-chunked-prefill \
--gpu-memory-utilization 0.88 \
--compilation-config '{"cudagraph_capture_sizes": [1,2,3]}'
```

### Key observations
- **Post-warmup TG:** ~58–62 tok/s (single request)
- **KV cache size:** 88,032 tokens (reported by vLLM)
- **Available KV cache memory:** ~1.91 GiB
- **Reported max concurrency @ 131K context:** 2.48×
- Warmup can be slow due to JIT + CUDAGraph capture.

### Concurrency benchmark highlights
- `8K in / 256 out`, 50 prompts, 10 RPS:
  - Peak concurrent requests: **50**
  - Output throughput: **54.3 tok/s avg**, **130 tok/s peak**
  - Total token throughput: **1790 tok/s**

Reference experiment: `../experiments/2026-03-14-qwen3.5-35b-a3b-vllm-pp3-concurrency.md`

## Agentic behavior

- Historical risk remains: we have prior sessions with runaway repeated tool calls.
- In short retest prompts, tool usage was sane (single-call behavior where expected), but this is not yet enough to declare it fully stable under long mixed workloads.

## Verdict

**🟡 PILOT / LAB-ONLY**

- ✅ Strong capability: one model for text + tools + vision on 24GB is possible.
- ⚠️ Reliability caveat: tool-loop pathology still requires guardrails and soak observation.
- ✅ Good fit for controlled llmlab experiments.
- ❌ Not yet “set-and-forget production” without loop controls.

## References

- Preliminary run with loop failures: `../experiments/2026-02-26-qwen3.5-35b-a3b-llmlab-preliminary.md`
- 24GB vision retest: `../experiments/2026-03-03-qwen3.5-35b-a3b-24gb-vision-retest.md`

## Changelog

- **2026-02-25/26:** Initial evaluation and loop-failure finding.
- **2026-03-03 (early):** Retest confirms 24GB text+tools+vision viability with tuned runtime profile; verdict updated to pilot/lab-only.
- **2026-03-03 (late):** Fixed reasoning loop issue by removing `--reasoning-format deepseek` flag (incompatible with Qwen3.5). Native thinking mode works correctly.
- **2026-03-04:** Production serving performance benchmarked on 3×RTX 3060 (36GB). Added detailed speed metrics, context degradation analysis, and thinking overhead quantification.
- **2026-03-14:** Added vLLM PP=3 deployment profile (official GPTQ-Int4), CUDAGraph tuning, KV/concurrency metrics, and high-concurrency benchmark results.
