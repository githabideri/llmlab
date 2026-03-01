# Qwen3.5-35B-A3B

**Base model:** [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)  
**Architecture:** DeltaNet linear attention + Mixture of Experts (35B total, 3B active)  
**Quant tested:** Q4_K_M (19.7 GB on disk)  
**Context window:** 131,072 tokens

## Quick Facts

| Param | Value |
|-------|-------|
| Total parameters | 35B |
| Active parameters | 3B (MoE) |
| Architecture | DeltaNet (linear attention) + MoE |
| Context window | 131,072 tokens |
| Quant tested | Q4_K_M (19.7 GB) |
| VRAM requirement | 2× 12GB GPUs (text-only, vision OOMs) |

## Evaluation (2026-02-25)

### Deployment

- Downloaded Q4_K_M (19.7 GB) to CT327 (2× RTX 3060 12GB)
- Required llama.cpp rebuild for Qwen3.5 arch support
- **Vision OOM**: mmproj + pipeline parallelism exceeded 24GB → text-only deployment
- **Template crisis**: Qwen3.5 chat template rejects `tool`/`function` roles → fixed with `--chat-template chatml` (11 min to diagnose)
- Config: `-ngl 99 -sm layer -fa 1 -ctk q8_0 -ctv q4_0 --chat-template chatml`

### Speed (llama-bench tg128, ctx 131072)

| Context Depth | TG tok/s |
|--------------:|---------:|
| 0 | ~95 |
| 2K | ~90 |
| 4K | ~85 |
| 8K | ~75 |
| 16K | ~60 |
| 32K | ~45 |

**DeltaNet speed hypothesis was WRONG.** Expected DeltaNet's O(1) per-step attention to beat Nemotron's Mamba-2 at long context. In practice, **Nemotron was faster at ALL context depths** on our hardware. See `notes/models/qwen3.5-vs-nemotron-speed-hypothesis.md`.

### Agentic Testing

**CRITICAL FAILURE: Infinite tool-call loops.**

In llmlab testing, Qwen3.5 entered a loop of 25+ identical `web_search` calls with the same query, never converging. The pattern:
1. Makes a tool call (e.g. `web_search("llama.cpp weather API")`)
2. Gets results
3. Immediately makes the same call again
4. Repeats indefinitely until context fills up or timeout

This happened consistently across multiple sessions and prompts. The model cannot be used for agentic work without a circuit breaker.

### Verdict

**❌ NOT SUITABLE for production agentic use.**

| Criterion | Result |
|-----------|--------|
| Speed | ⚠️ Good but NOT better than Nemotron (DeltaNet advantage not realized on PCIe GPUs) |
| Tool calling | ❌ Infinite loop pathology — unusable without external circuit breaker |
| Template | ⚠️ Requires ChatML workaround (native template rejects tool roles) |
| Vision | ❌ OOMs on 24GB with mmproj |
| Context degradation | ⚠️ Steeper than Nemotron despite DeltaNet claims |

### Why DeltaNet Wasn't Faster

- DeltaNet's O(1) attention doesn't help when the bottleneck is PCIe data transfer between GPUs
- `-sm layer` means each GPU processes complete layers — attention mechanism efficiency is secondary to inter-GPU communication
- The model's 19.7 GB weight size creates more transfer overhead than Nemotron's smaller footprint
- Full analysis: `notes/models/qwen3-vs-nemotron-speed-analysis.md`

## References

- Full evaluation: `notes/models/qwen3.5-35b-a3b-eval-2026-02-25.md`
- Memory analysis: `notes/models/qwen3.5-memory-analysis-2026-02-25.md`
- Speed hypothesis: `notes/models/qwen3.5-vs-nemotron-speed-hypothesis.md`
- Session report: `notes/models/qwen35-llmlab-session-report-2026-02-26.md`
- Experiment log: `experiments/2026-02-26-qwen3.5-35b-a3b-llmlab-preliminary.md`

## Changelog

- **2026-02-25:** Initial evaluation. Template workaround, speed benchmarks, tool-loop failure documented.
- **2026-02-26:** Session report from llmlab testing confirms tool-loop pathology.
