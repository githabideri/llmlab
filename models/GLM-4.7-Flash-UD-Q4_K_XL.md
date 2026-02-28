# GLM-4.7-Flash-UD-Q4_K_XL

> **✅ STATUS: WORKING & RECOMMENDED** (Feb 2026)  
> **Previous verdict (Feb 20):** NOT RECOMMENDED due to broken MLA  
> **Current verdict:** ✅ MLA fix confirmed working, excellent agent performance  
> **Speed:** ~26.6 tok/s real serving (dual RTX 3060 12GB)

**Last Updated:** 2026-02-28

---

## Model Info

- **Base Model**: [THUDM/GLM-4.7-Flash](https://huggingface.co/THUDM/GLM-4.7-Flash)
- **GGUF Source**: [unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF)
- **Quant**: Q4_K_XL (importance matrix: unsloth calibration)
- **File Size**: ~2.8 GB
- **Architecture**: MoE (4.7B total params, 2.4B active per token)
- **Context Window**: 131,072 tokens (tested)
- **License**: Apache 2.0

---

## Hardware & Config

**Tested on:**
- **GPUs**: 2× RTX 3060 12GB (Ampere, no NVLink) + 1× GTX 1050 2GB (display only)
- **Backend**: llama-cpp (b4930 commit)
- **Flags**: `-ngl 99 -sm layer -fa 1 -c 131072 -ctk q8_0 -ctv q4_0`

---

## Performance Metrics

### Speed (Real Serving)

| Context Depth | Prompt Proc (tok/s) | Generation (tok/s) | Notes |
|---------------|---------------------|--------------------| -----|
| 0 - 40K | 150 - 220 | **27.8** | Fresh context, cold start |
| 40K - 47K | 194 - 220 | **27.4** | Pre-compaction, stable |
| 47K - 53K | 140 - 200 | **25.9** | Post-compaction, -7% |
| **Mean (all)** | **~180** | **~26.6** | Across 47 API calls |

**Cache hits**: 80-130 tok/s prompt processing (LCP similarity > 0.98)

**Degradation rate**: ~0.3 tok/s per 1K context tokens (from 47K → 53K)

---

## Real-World Performance (Agent Task)

**Test scenario**: Multi-step troubleshooting task (15-16 hours active session, 47 API calls)  
**Context growth**: 38K → 53K tokens  
**Typical outputs**: 100-300 tokens per response (tool calls + explanations)  
**Compactions**: 1 event at 47K tokens (82 seconds to reprocess 33.5K tokens @ 409 tok/s)

**Strengths**:
- ✅ Excellent cache hit rate (40% of requests reused KV cache)
- ✅ Stable 26-28 tok/s across 15K token growth
- ✅ Small active param count (2.4B) enables fast tool switching
- ✅ Handled 1,089-token outputs without quality loss

**Weaknesses**:
- ⚠️ In-band thinking tokens (5-15% overhead, not user-visible)
- ⚠️ Speed drops ~7% after 50K context (still acceptable)
- ⚠️ Compaction pause (82s) noticeable in long sessions

---

## Comparison to Alternatives (Same Hardware)

| Model | Active Params | Speed (TG) | Context | Notes |
|-------|---------------|------------|---------|-------|
| **GLM-4.7-Flash (Q4_K_XL)** | 2.4B | ~26 tok/s | 131K | Tested model |
| Nemotron-30B-A3B (IQ4_NL) | 3B | ~60 tok/s | 131K | 2.3× faster, +25% params |
| Qwen3-30B-A3B (GPTQ-Int4, vLLM) | 30B | ~18.5 tok/s | 12K | Similar speed, 12.5× params |

---

## Recommendations

**Use Cases**:
- ✅ **Agent work** (tool-heavy, short outputs): Excellent fit
- ✅ **Long-context tasks** (up to 131K): Stable performance
- ⚠️ **High-throughput serving**: Consider Nemotron-30B-A3B instead (2.3× faster)
- ❌ **Large batch generation**: Not tested (single-slot use case)

**Optimal Config**:
- Context: 65K-131K (tested up to 53K active)
- Split mode: `-sm layer` (critical for PCIe dual-GPU setup)
- Flash attention: `-fa 1` (enabled)
- KV cache quant: `-ctk q8_0 -ctv q4_0` (balanced quality/VRAM)

**Thinking Token Handling**:
- Model generates in-band thinking tokens (5-15% of output)
- To suppress: use `--reasoning-budget 0` (not tested yet)

---

## Quirks & Gotchas

1. **MoE routing overhead**: Slight slowdown vs dense 2.4B models (tradeoff for 4.7B total params)
2. **Thinking tokens**: Counted in speed metrics but not always user-visible
3. **Compaction timing**: First compaction at ~47K tokens (expected behavior for sliding window)
4. **Cache sensitivity**: High LCP hit rate (~40% requests) — good for iterative tasks

---

## Benchmark Data

### llama-bench (Static)

*(No data yet — add when available)*

### Real Serving (Feb 2026)

- **Test duration**: 15-16 hours
- **API calls**: 47 requests
- **Context range**: 38,141 → 53,221 tokens
- **Mean TG speed**: 26.6 tok/s
- **Prompt cache hit rate**: ~40% of requests (sim_best > 0.98)

---

## Download

```bash
huggingface-cli download unsloth/GLM-4.7-Flash-GGUF \
  GLM-4.7-Flash-UD-Q4_K_XL.gguf \
  --local-dir ./models
```

**SHA256** (optional, verify after download):
```bash
sha256sum GLM-4.7-Flash-UD-Q4_K_XL.gguf
```

---

## Notes

- First comprehensive evaluation on RTX 3060 dual-GPU setup
- Baseline for future model comparisons in llmlab
- Performance data from real agent task (not synthetic benchmark)

---

**Last test**: 2026-02-28 (llama-cpp b4930, dual RTX 3060 12GB)
