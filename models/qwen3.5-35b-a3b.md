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

## Performance notes (retest)

From live serving logs at ~32k prompt depth:
- Prompt eval: ~725–984 tok/s (cache/checkpoint dependent)
- Generation eval: ~44–45 tok/s

Vision preprocessing time scales hard with input size:
- ~128px image class: ~0.3 s
- ~512px image class: ~5.3 s
- ~1024px image class: ~32 s

**Operational takeaway:** keep interactive images small/medium unless high latency is acceptable.

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
- **2026-03-03:** Retest confirms 24GB text+tools+vision viability with tuned runtime profile; verdict updated to pilot/lab-only.
