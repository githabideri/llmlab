# Nanbeige4.1-3B

**Base model:** [Nanbeige/Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B)  
**GGUF quants tested:** mradermacher i1 + Q8 variants  
**Date tested:** 2026-02-23

## Quick Facts

| Param | Value |
|---|---|
| Family | LLaMA-style dense |
| Params | ~3B |
| `n_ctx_train` | 262,144 |
| KV cache | ~24 KiB/token (very efficient for long context) |
| Tested hardware | 2× RTX 3060 12GB |
| Runtime flags | `-sm layer -fa 1 -ctk q8_0 -ctv q4_0` |

## Quants on disk

| Quant | File |
|---|---:|
| i1-IQ4_NL | 2.2G |
| i1-Q6_K | 3.1G |
| Q8_0 | 3.9G |

## Performance Summary (24GB setup)

### 100k context (ingest-heavy check)

| Quant | PP tok/s | TG tok/s |
|---|---:|---:|
| i1-IQ4_NL | 1324.76 | **19.76** |
| i1-Q6_K | 1362.09 | 18.93 |
| Q8_0 | **1368.76** | 18.26 |

### Near-max context check

| Quant | Context | PP tok/s | TG tok/s |
|---|---:|---:|---:|
| i1-IQ4_NL | 261k | **559.92** | **8.26** |
| i1-Q6_K | 261k | 545.94 | 8.06 |
| Q8_0 | 261k | 549.63 | 7.94 |

## Interpretation

- Q6/Q8 can prefill faster around ~100k.
- At very deep context (192k → 261k), **i1-IQ4_NL is consistently best** on this hardware.
- Differences are not massive, but stable and repeatable.

## Status

🟢 **Perf baseline established**  
🟡 **Quality eval still pending** (must be completed before final default quant decision)

## References

- Experiment log: `llmlab/experiments/2026-02-23-nanbeige4.1-3b-quant-sweep.md`
- Raw datasets: `nanbeige_bench_complete.json`, `nanbeige_highctx_complete.json`
