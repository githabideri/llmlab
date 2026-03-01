# ZwZ-4B

**Base model:** [inclusionAI/ZwZ-4B](https://huggingface.co/inclusionAI/ZwZ-4B)  
**GGUF quant repo:** [mradermacher/ZwZ-4B-i1-GGUF](https://huggingface.co/mradermacher/ZwZ-4B-i1-GGUF)  
**Architecture:** `qwen3_vl` (multimodal, derived from Qwen3-VL-4B)  
**File tested:** `ZwZ-4B-i1-Q6_K.gguf` (~3.38 GB)

## Quick Facts

| Param | Value |
|-------|-------|
| Total parameters | 4B (dense) |
| Architecture | Qwen3-VL derivative (multimodal) |
| Context window | 131,072 tokens |
| Quant tested | i1-Q6_K (3.38 GB) |
| KV cache/token | 54 KiB (`36 × 8 × 128 × 1.5`) |
| Max context @24GB | ~390K (fits easily) |

## Evaluation (2026-02-23)

### Fitment

All practical quants fit comfortably in 24GB with 100K+ context:

| Quant | File Size | + KV @100K | Total | Verdict |
|-------|----------:|----------:|------:|---------|
| i1-Q6_K | 3.38 GB | 5.1 GB | 8.5 GB | ✅ |
| i1-Q4_K_M | 2.67 GB | 5.1 GB | 7.8 GB | ✅ |
| i1-IQ4_NL | 2.43 GB | 5.1 GB | 7.5 GB | ✅ |

### Speed

Tested on 2× RTX 3060 12GB:

| Context Depth | TG tok/s |
|--------------:|---------:|
| 64 (fresh) | 77 |
| 100K | 11 |

### Assessment

**🟡 MAYBE — parked for later investigation.**

- Reddit claimed "best small agentic model" but no evidence found
- Architecture is multimodal (`qwen3_vl`), not a pure text-agent model
- 77 tok/s at fresh context is good for 4B
- Steep degradation to 11 tok/s at 100K
- Not benchmarked on our agentic benchmark suite (L0-L4)

### Decision

Parked. The multimodal architecture raises questions about whether it's optimized for text-only agentic use. Would need L0-L4 benchmark suite results before considering production deployment.

## References

- Quick eval: `notes/llama-cpp/zwz-4b-quick-eval-2026-02-23.md`

## Changelog

- **2026-02-23:** Initial evaluation. Downloaded, benchmarked speed, fitment calculated. Verdict: MAYBE.
