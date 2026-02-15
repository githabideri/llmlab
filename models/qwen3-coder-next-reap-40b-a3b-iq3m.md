# Qwen3-Coder-Next-REAP-40B-A3B (i1-IQ3_M)

**Source:** [mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-i1-GGUF](https://huggingface.co/mradermacher/Qwen3-Coder-Next-REAP-40B-A3B-i1-GGUF)  
**Quant:** i1-IQ3_M (imatrix-quantized, 18.2 GB on disk)  
**Base model:** [lovedheart/Qwen3-Coder-Next-REAP-40B-A3B](https://huggingface.co/lovedheart/Qwen3-Coder-Next-REAP-40B-A3B)

A 40B parameter hybrid DeltaNet + Gated Attention MoE model with 3B active parameters per token.

## Quick Facts

| Param | Value |
|-------|-------|
| Parameters | 40B total, ~3B active (MoE) |
| Architecture | Hybrid DeltaNet + Attention MoE |
| Quant tested | i1-IQ3_M (imatrix) |
| File size | 18.2 GB |
| Context window | 262,144 tokens (native) |
| VRAM requirement | **2x 12GB GPUs** (requires tensor split) |
| Attention layers | 12 of 48 (75% KV cache reduction) |

## Performance

Tested on 2x RTX 3060 (12GB each) with `--tensor-split 1,1`:

### Speed Metrics (Empty Cache)

| Metric | Value |
|--------|-------|
| Prompt eval (varied) | 56-102 tok/s (prompt-size dependent) |
| Generation (fresh) | **22.4-22.5 tok/s** |
| Generation (filled ctx) | See degradation section |

### Context Window Stability

| Target | VRAM | KV Cache | Result |
|--------|------|----------|--------|
| 32k | 18.2 GB | 320 MB | ✅ Stable |
| 64k | 18.5 GB | 640 MB | ✅ Stable |
| 128k | 19.3 GB | 1.25 GB | ✅ Stable |
| 196k | 20.1 GB | 1.9 GB | ✅ Stable |
| **262k (native max)** | **21.0 GB** | **2.6 GB** | ✅ **Stable** |

**Sweet spot on 2×RTX 3060:** 128k-196k for production use (leaves headroom). 262k fits but uses most available VRAM.

## Prefill Degradation Analysis

**Critical finding:** Performance degrades significantly with filled cache, non-linearly with context size.

### Empty Cache Baseline

| Context | TG Speed | Note |
|---------|----------|------|
| All sizes | 22.4-22.5 tok/s | Consistent across 32k-262k |

### Degradation Results (Filled Cache)

| Context | Fill % | Tokens Cached | TG Speed | vs Baseline |
|---------|--------|---------------|----------|-------------|
| **64k** | 77% | 50,001 | **18.5 tok/s** | **-17%** |
| **128k** | 39% | 50,925 | **18.4 tok/s** | **-18%** |
| **196k** | 26% | 50,925 | **18.5 tok/s** | **-18%** |
| **262k** | 39% | 101,817 | **15.3 tok/s** | **-32%** ⚠️ |

**Key insights:**
1. **Degradation plateaus at ~18 tok/s** for moderate fills (~50k tokens)
2. **Fill percentage matters more than absolute context size** until ~100k+ tokens
3. **Heavy fills (100k+) show severe degradation** (-32% at 262k with 101k tokens)

**Production guidance:**
- For **light usage** (<50k cached): expect ~18-19 tok/s
- For **heavy usage** (100k+ cached): expect ~15-16 tok/s
- **Recommended for production:** 128k-196k context (predictable ~18% degradation)

## Agentic Capabilities

Tested as an autonomous agent (LocalBot session, Feb 14 afternoon, 7h 12min duration):

### Task Performance

**Workload:** Web research, browser automation, multi-language interaction (English/German)

| Task Class | Attempts | Success | Notes |
|------------|----------|---------|-------|
| Weather lookups | 1 | ✅ | Vienna forecast accurate |
| Browser automation | 1 | ✅ | orf.at screenshot (3 retries needed) |
| Web article research | 3 | ✅✅✅ | Wired.com GPT-4o article + detailed follow-up |
| **Total** | **9** | **8/9** | **89% success rate** |

**Quality highlights:**
- **Research synthesis:** Exceptional detail on OpenClaw Wired article (captured humor, technical nuance, key plot points)
- **Tool chaining:** Smooth navigate → snapshot → web_fetch → summarize workflows
- **Error recovery:** Self-corrected screenshot path issues after retry
- **Multilingual:** Natural German/English switching (acknowledged German weakness when prompted)

**Known issues:**
- **Screenshot delivery:** Took 3 attempts (file path assumptions not verified upfront)
- **Overconfidence:** Claimed "screenshot sent" before file existence check
- **German language quality:** Model self-acknowledged limitation

### Behavioral Notes (Feb 14)

- **Error recovery works:** Fixed screenshot issues without explicit guidance
- **Research quality is strong:** Detailed, well-structured article summaries
- **Tool reliability needs monitoring:** 78% first-attempt success vs ~85% Nemotron baseline
- **Context retention:** Maintained coherence across 40+ message session

## Comparison: i1-IQ3_M vs Q2_K_XL

| Metric | i1-IQ3_M | Q2_K_XL | Winner |
|--------|----------|---------|--------|
| File size | 18.2 GB | 19.0 GB | **i1-IQ3_M** (-5%) |
| TG speed (empty) | 22.4 tok/s | ~21.0 tok/s* | **i1-IQ3_M** |
| VRAM (128k) | 19.3 GB | ~19.5 GB* | **i1-IQ3_M** |
| Quality (subjective) | TBD | TBD | *Needs eval* |

*From 2026-02-07 measurements

**Recommendation:** Use i1-IQ3_M — same/better speed, smaller file, expected quality improvement from imatrix quantization.

## Recommended Config

```bash
llama-server \
  --model /path/to/Qwen3-Coder-Next-REAP-40B-A3B.i1-IQ3_M.gguf \
  --ctx-size 131072 \
  --n-gpu-layers -1 \
  --tensor-split 1,1 \
  --cache-type-k q8_0 \
  --cache-type-v q4_0 \
  --flash-attn on \
  --split-mode row \
  --n-cpu-moe 0 \
  --jinja \
  --host 0.0.0.0 --port 8080
```

**Required flags:**
- `--tensor-split 1,1` — split across 2 GPUs (adjust ratio for unequal VRAM)
- `--cache-type-k q8_0 --cache-type-v q4_0` — recommended KV cache compression
- `--flash-attn on` — critical for long context performance
- `--n-cpu-moe 0` — keep all MoE experts on GPU

## Benchmark Results

### Empty-Cache Speed Tests (Feb 14)
✅ **Pass** — All context sizes 32k-262k ran stable at 22.4-22.5 tok/s

### Prefill Degradation Tests (Feb 14)
✅ **Complete** — Documented degradation curve from 0-100k+ tokens

### Agentic Task Suite (Feb 14, afternoon session)
⚠️ **Partial** — 89% success on limited test set (web research, browser automation)

**Not yet tested:**
- [ ] Structured benchmarks (L0-L4 suite)
- [ ] Coding task quality comparison vs Q2_K_XL
- [ ] Multi-step tool chains (complex workflows)
- [ ] Long-context reasoning tasks
- [ ] Error edge cases
- [ ] Reasoning/thinking behavior (if applicable to this model)

## Known Issues

### 🚨 Screenshot Path Handling

**Severity:** Low — causes retry overhead, not data loss

**Symptom:**
- Browser screenshot attempts succeed server-side
- Agent claims "screenshot sent" before verifying file exists
- Subsequent attempts fail due to incorrect path assumptions
- Resolves after 2-3 retries with self-correction

**Trigger:**
- Browser automation tasks requiring screenshot delivery
- Initial file path assumptions not verified

**Observed rate:** 1/1 screenshot tasks required retries (limited sample)

**Mitigations:**
- Model eventually self-corrects
- No user intervention needed
- Consider adding path verification prompt for browser tasks

### 📉 Degradation at High Fill

**Severity:** Medium — impacts production context size decisions

**Symptom:**
- Performance drops 17-32% when KV cache fills
- Non-linear: ~18% at 50k tokens, ~32% at 100k+ tokens
- Affects all context sizes once absolute token count exceeds ~50k

**Trigger:**
- Long conversations or document processing
- Cached tokens > 50k

**Observed:** Consistent across multiple test runs

**Mitigations:**
1. **Use 128k-196k context** for production (predictable ~18% degradation)
2. **Avoid 262k** unless necessary (severe degradation at high fills)
3. **Monitor token counts** and consider session resets for long workflows
4. **Set expectations:** Real-world speed is ~18-19 tok/s, not 22-23 tok/s

### 🌍 German Language Quality

**Severity:** Low — model self-acknowledges limitation

**Symptom:**
- User feedback: "german is not your strength"
- Model agreed when prompted
- Switched to English successfully

**Trigger:**
- German-language news summarization task

**Observed rate:** 1/1 German tasks (limited sample)

**Mitigations:**
- Model handles English-language tasks well
- Multilingual capability exists but may need evaluation
- Consider English-first workflows for this model

## Hardware Tested

- 2x NVIDIA RTX 3060 12GB
- Intel i5-7400
- 24GB system RAM
- llama.cpp (latest as of 2026-02-14)

## Test Coverage Status

**✅ Complete:**
- Empty-cache speed benchmarks (all context sizes)
- VRAM scaling measurements
- Prefill degradation curve (0-100k+ tokens)
- Basic agentic tasks (web research, browser automation)

**⏳ Partial:**
- Agentic reliability (89% on limited sample, needs broader testing)
- Multilingual capability (German weakness noted, needs systematic eval)

**❌ Not tested:**
- Structured benchmarks (L0-L4 suite)
- Coding task quality vs baseline
- Long-context reasoning
- Thinking/reasoning behavior (if supported)
- Tool-call reliability under stress
- Multi-step complex workflows

## Changelog

- **2026-02-14:** Initial profile based on empty-cache benchmarks (32k-262k), prefill degradation tests (64k-262k), and afternoon agentic session (7h 12min, 9 tasks, 89% success). Documented screenshot path handling issue, degradation severity, German language weakness. Test coverage: speed/VRAM complete, agentic partial, structured benchmarks pending.
