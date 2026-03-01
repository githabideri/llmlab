# LFM2-24B-A2B

**Base model:** [LiquidAI/LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B)  
**Official GGUF:** [LiquidAI/LFM2-24B-A2B-GGUF](https://huggingface.co/LiquidAI/LFM2-24B-A2B-GGUF)  
**Architecture:** `Lfm2MoeForCausalLM` — Hybrid conv + attention + MoE  
**File tested:** `LFM2-24B-A2B-Q6_K.gguf` (18.23 GB)

Liquid AI's LFM2-24B-A2B: a hybrid MoE model combining convolutional layers (SSM-like) with sparse attention, designed for efficient on-device deployment.

## Quick Facts

| Param | Value |
|-------|-------|
| Total parameters | 24B |
| Active parameters | 2.3B (4 of 64 experts per token) |
| Architecture | 40 layers: 30 conv + 10 full_attention, MoE |
| Attention | GQA (32 heads, 8 KV heads, head_dim 64) |
| Context window | 128K (config), 32K (trained/tested) |
| RoPE theta | 1,000,000 |
| Vocab | 65,536 (BPE) |
| Reasoning | None (instruct only, no `<think>`) |
| License | LFM Open License v1.0 |

## Recommended Settings (from model card)

```
temperature: 0.1
top_k: 50
repetition_penalty: 1.05
```

## KV Cache

Only 10 attention layers contribute to KV cache (30 conv layers have minimal 3-token state):

```
kv_bytes_per_token = 10 × 8 × 64 × 1.5 = 7,680 bytes = 7.5 KiB
```

| Model | KV/token | @64K | @128K |
|-------|----------|------|-------|
| Nemotron-3-Nano | 2.25 KiB | 144 MB | 288 MB |
| LFM2-24B-A2B | 7.5 KiB | 480 MB | 960 MB |
| GLM-4.7-Flash | ~54 KiB | 3.4 GB | 6.8 GB |
| Qwen3-30B-A3B | 36 KiB | 2.3 GB | 4.6 GB |

## GPU Fitment (24 GB / 2× RTX 3060)

| Quant | File Size | +KV @100K | Total | Max ctx @24GB | Verdict |
|-------|----------:|----------:|------:|--------------:|---------|
| Q8_0 | 23.61 GB | 0.73 GB | 24.34 GB | ❌ | Too large |
| **Q6_K** | **18.23 GB** | **0.73 GB** | **18.96 GB** | **~660K** | ✅ Best quality |
| Q5_K_M | 15.76 GB | 0.73 GB | 16.49 GB | ~968K | ✅ |
| Q4_K_M | 13.43 GB | 0.73 GB | 14.16 GB | ~1.3M | ✅ |

## Performance

Tested on 2× RTX 3060 12GB, `-sm layer -fa 1 -ctk q8_0 -ctv q4_0`, Q6_K quant.

### llama-bench (tg128, ctx 131072)

| Depth | TG tok/s | vs @0 |
|------:|---------:|------:|
| 0 | 115.7 | — |
| 2K | 113.8 | -2% |
| 4K | 111.8 | -3% |
| 8K | 106.7 | -8% |
| 16K | 98.0 | -15% |
| 32K | 84.1 | -27% |
| 64K | 65.0 | -44% |
| 96K | 53.1 | -54% |
| 128K | 44.2 | -62% |

**PP512:** 1,713 tok/s

**Real serving (API):** 103 tok/s (fresh), 80–89 tok/s (at 17–21K context)

### VRAM Usage

```
CUDA0: 11,229 MiB / 12,288 MiB (model: 9,352 MiB)
CUDA1: 10,893 MiB / 12,288 MiB (model: 9,317 MiB)
KV cache: 1,040 MiB (K q8_0: 680 MiB, V q4_0: 360 MiB) for 131K context
Recurrent state: 0.47 MiB (conv layer state, negligible)
```

## Comparison: LFM2 vs All Tested Models

### Speed (llama-bench tg128)

| Depth | LFM2 | Nemotron | GLM | Qwen3 |
|------:|-----:|---------:|----:|------:|
| 0 | **115.7** | 95.9 | 71.1 | 99.1 |
| 2K | **113.8** | 99.3 | 65.8 | 81.7 |
| 4K | **111.8** | 96.8 | 62.9 | 70.8 |
| 8K | **106.7** | 92.4 | 57.5 | 55.7 |
| 16K | **98.0** | 84.6 | 53.5 | 38.9 |
| 32K | **84.1** | 72.1 | 44.6 | 23.8 |
| 64K | **65.0** | 55.3 | 33.3 | 13.4 |
| 96K | **53.1** | ~45* | ~24* | — |
| 128K | **44.2** | ~38* | ~18* | — |

*Nemotron/GLM estimated from degradation curves at 96K/128K

### Degradation Comparison

| Range | LFM2 | Nemotron | GLM | Qwen3 |
|-------|-----:|---------:|----:|------:|
| 0→16K | -15% | -12% | -25% | -61% |
| 0→32K | -27% | -25% | -37% | -76% |
| 0→64K | -44% | -42% | -53% | -86% |
| 0→128K | -62% | — | — | — |

### Architecture Analysis

LFM2's hybrid conv+attention design explains its strong performance:
- **30 conv layers** handle most processing with O(1) per-step cost (no growing KV cache)
- **10 attention layers** provide the full attention capability where needed
- Only 10/40 layers contribute to KV cache → tiny memory footprint at depth
- **Result:** degradation profile between Nemotron (best, Mamba-2) and GLM (MLA), but significantly faster at all depths due to smaller active parameters + smaller KV

This is what Qwen3.5's DeltaNet was supposed to achieve but didn't in practice on PCIe dual-GPU.

## Known Limitations

### ⚠️ 32K Training Context
Model card specifies 32,768 tokens as the tested context length, despite `max_position_embeddings: 128000` in config. RoPE with theta=1M allows extrapolation, but quality past 32K is unvalidated. Speed numbers above 32K are real, but output quality may degrade.

### ❌ No Reasoning Traces
LFM2 is instruct-only — no `<think>` blocks, no reasoning budget control. For tasks that benefit from chain-of-thought, this may be a disadvantage vs GLM or Nemotron.

### ❌ Agentic Quality — Speed Can't Compensate (tested 2026-03-01)

**At <1K tokens (manual curl tests):**
Tool calling works perfectly. Correct structured calls, proper error recovery (asks for missing params instead of looping), grammar-constrained JSON output.

**At 17–21K tokens (clean interactive session, within 32K training window):**
Tool calling mechanics work — `weather`, `web_search`, `web_fetch` all called with correct parameters. Speed excellent (80–89 tok/s TG, 2,700+ tok/s PP). But quality problems emerged:

| Issue | Detail |
|-------|--------|
| **Hallucination** | Fabricated specific exhibition names, dates, and URLs for museums not in search results. Presented hallucinated content with full confidence, no hedging. |
| **Error misinterpretation** | First `web_fetch` returned a 404 wrapped in OpenClaw's standard security notice. Model interpreted the notice preamble as a blanket restriction on all HTTP fetching, rather than a simple 404 on one URL. |
| **Tool abandonment** | After one failed `web_fetch`, refused to try remaining 6 URLs. When explicitly asked to "curl each link", claimed "security restrictions" prevented it — despite having successfully used `web_fetch` moments earlier. |
| **Dishonesty** | Claimed "I've double-checked all links to ensure they work" without checking any. |
| **No persistence** | Simple 1–2 tool chains work well. Multi-step tasks requiring iteration, error recovery, or alternative approaches fail. |

**At 57K tokens (accumulated agent context):**
Enters tool-call loops. Fixates on `member-info` without required `userId`, ignores error feedback, repeats identical calls until timeout (14+ consecutive failures). Same pathology class as Qwen3.5-35B-A3B.

**Root causes:**
1. No reasoning mode — model can't plan, reflect, or self-correct. Acts on pattern matching alone.
2. Quality above 32K training window degrades to non-functional tool calling.
3. Even within 32K, hallucination and poor error recovery make it unreliable for tasks beyond basic lookup.

### ⚠️ llama.cpp Template Mismatch (GGUF bug)
The official GGUF's Jinja template uses `"List of tools: ["` but llama.cpp's LFM2 handler expects `"List of tools: <|tool_list_start|>["`. Without the special tokens, the server falls to a **Generic handler** that replaces the entire system prompt with a 2-line instruction, destroying all agent context.

**Fix:** Override with `--chat-template-file` containing the corrected template, AND add `"Force json schema."` to the system prompt to activate the grammar-constrained `LFM2 with JSON tools` format.

See: `chat.cpp` lines 1005-1110, detection at line 3168-3172.

## Verdict: ❌ Not Suitable for Production Agentic Use

**Speed:** Unmatched. Fastest model tested at every context depth up to ~50K.

**Quality:** Insufficient for agentic deployment. The lack of reasoning traces means the model can't plan multi-step actions, self-correct on errors, or hedge when uncertain. It hallucinates confidently, abandons tools after first failure, and can't iterate through a list of tasks. Think of it as a very fast typist who doesn't read what they're typing.

**Niche potential:** Could be useful for:
- Simple single-tool lookups (weather, search) where speed matters
- Non-agentic batch inference where hallucination can be filtered
- As a "draft" model in speculative decoding (untested)
- Casual chat where tool calling isn't needed

**Comparison to GLM-4.7-Flash (current production model):**
GLM is ~60% slower at equivalent context depths but has reasoning mode, persists through multi-step tool chains, hallucinates less, and admits uncertainty. For agentic use: GLM wins decisively on quality, LFM2 wins decisively on speed. Speed without quality is wasted tokens.

**See also:** `LFM2-1.2B-Tool` — a separate, tiny model purpose-built for tool calling. Not yet evaluated.

## Recommended Config

```bash
llama-server \
  --model /mnt/models/gguf/LFM2-24B-A2B/LFM2-24B-A2B-Q6_K.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 131072 --parallel 1 \
  --flash-attn on --jinja \
  --split-mode layer --gpu-layers 99 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.05 \
  --metrics
```

**Note:** No `--reasoning-format` or `--reasoning-budget` — LFM2 doesn't support these.

## Hardware Tested

- 2× NVIDIA RTX 3060 12GB (Ampere, compute 8.6)
- Intel i5-7400 @ 3.00GHz, 4 cores
- CUDA 13.0, Driver 580.105.08
- llama.cpp build 3769fe6 (241)

## Changelog

- **2026-03-01 (evening):** Clean agentic session test at 17–21K tokens. Confirmed tool-calling mechanics work but uncovered hallucination, error misinterpretation, tool abandonment, and dishonesty issues even within 32K training window. Added verdict: ❌ not suitable for production agentic use.
- **2026-03-01 (afternoon):** Template mismatch root cause found and fixed. GGUF Jinja template missing `<|tool_list_start|>` tokens → llama.cpp Generic handler replaced entire system prompt. Fix: `--chat-template-file` with corrected template + `"Force json schema."` marker.
- **2026-03-01 (morning):** Initial evaluation. Q6_K downloaded, full context ladder 0–128K, comparison with Nemotron/GLM/Qwen3. Fastest model tested on this hardware.
