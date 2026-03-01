# GLM-4.7-Flash

**Base model:** [THUDM/GLM-4.7-Flash](https://huggingface.co/THUDM/GLM-4.7-Flash)  
**GGUF quant:** [bartowski/GLM-4.7-Flash-GGUF](https://huggingface.co/bartowski/GLM-4.7-Flash-GGUF) (UltraDeep dynamic quants)  
**File tested:** `GLM-4.7-Flash-UD-Q4_K_XL.gguf` (17.5 GB on disk)

Zhipu AI's GLM-4.7-Flash, a 30B parameter MoE model optimized for tool-calling and agentic tasks.

## Quick Facts

| Param | Value |
|-------|-------|
| Parameters | ~30B total (MoE, active params undisclosed) |
| Architecture | Mixture of Experts, Multi-head Latent Attention (MLA) |
| Quant tested | **UD-Q4_K_XL** (17.5 GB, UltraDeep dynamic quant) |
| Context window | 131,072 tokens (tested) |
| VRAM requirement | **2× 12GB GPUs** (requires split-mode layer) |
| Reasoning | Built-in `<think>` blocks, always generated (no disable mechanism) |

## Upstream Benchmarks

| Benchmark | GLM-4.7-Flash | Qwen3-30B-A3B | GPT-OSS-20B |
|-----------|------:|------:|------:|
| AIME 25 | **91.6** | 85.0 | 91.7 |
| SWE-bench Verified | **59.2** | 22.0 | 34.0 |
| τ²-Bench (tool-calling) | **79.5** | 49.0 | 47.7 |

GLM-4.7-Flash leads significantly on code and tool-calling benchmarks.

## Performance

Tested on 2× RTX 3060 (12GB each), `-sm layer -fa 1 -ctk q8_0 -ctv q4_0`:

### llama-bench (tg128, ctx 131072)

| Context Depth | TG tok/s | PP tok/s |
|--------------:|---------:|---------:|
| 0 | 71.1 | — |
| 2K | 65.8 | — |
| 4K | 62.9 | — |
| 8K | 57.5 | — |
| 16K | 53.5 | — |
| 32K | 44.6 | — |
| 64K | 33.3 | — |

**PP512:** 1,426 tok/s (empty cache)

### Context Degradation

| Range | Degradation |
|-------|-------------|
| 0 → 32K | -37% |
| 0 → 64K | **-53%** |
| 0 → 107K | ~-66% (measured from real serving) |

**Comparison:**
- Nemotron-3-Nano: -42% at 64K (best) 👑
- GLM-4.7-Flash: -53% at 64K
- Qwen3-30B-A3B: -86% at 64K (worst)

GLM sits between Nemotron and Qwen3 for context degradation. Usable up to ~50K context for interactive work.

### Real-World Serving Speed

Measured from a 79-request production session (llama-server logs, ctx 131072 allocated):

| Context Depth | llama-bench | Actual Serving | Delta |
|--------------:|------:|------:|------:|
| ~37K | ~47 (extrapolated) | 30.0 tok/s | -36% |
| ~54K | ~37 (extrapolated) | 24.2 tok/s | -35% |
| ~113K | ~20 (extrapolated) | 14.4 tok/s | -28% |

**Real-world serving is 28-36% slower than llama-bench.** Causes:
- Full 131K context KV cache pre-allocated (VRAM pressure)
- Server overhead (HTTP, JSON, tokenization, sampling)
- Thinking token generation (not present in benchmarks)
- Prompt cache management

### Speed Across a Full Session

79-request session with 1 compaction event:

**Phase 1 (context 59K → 113K, 33 requests):**
- Start: 23.0 tok/s → End: 14.4 tok/s (-37%)
- Rate: -0.16 tok/s per 1K context tokens

**Phase 2 (context 37K → 54K, 44 requests, post-compaction):**
- Start: 30.0 tok/s → End: 24.2 tok/s (-19%)
- Rate: -0.34 tok/s per 1K context tokens

**Compaction recovery:** 14.4 → 30.0 tok/s (**2.1× improvement**)

### PP Speed (Real-World, with Prompt Cache)

Prompt cache hit rates were excellent (`sim_best > 0.99` on most requests). Only new tokens needed processing:

| New Tokens | PP tok/s | Note |
|-----------:|---------:|------|
| 8-50 | 56-124 | Small incremental updates |
| 150-500 | 119-200 | Medium batches |
| 1,000-2,500 | 150-313 | Large batches |
| 7,800+ | 153 | Very large (cache miss) |
| 58,787 (cold start) | 514 | Full context load |

## Thinking Token Behavior

GLM-4.7-Flash generates `<think>` blocks **unconditionally**. There is no `--reasoning-budget` or `--reasoning-format` equivalent to suppress them in llama.cpp.

### Overhead Measured from Production Session

| Response Type | Thinking Overhead |
|---------------|------------------:|
| Short factual answer | ~0.6-0.7× output |
| Tool call decision | ~6× output |
| Complex instructions | ~1.0-1.2× output |
| Memory/file operations | ~0.1× output |

**Average thinking overhead: ~20-40% of total generated tokens.** This means effective user-visible throughput is lower than the TG tok/s numbers suggest.

For tool-call-heavy agentic workflows (many short tool calls), thinking overhead is proportionally larger.

## MLA (Multi-head Latent Attention) Notes

GLM-4.7-Flash uses MLA, which was broken in llama.cpp until January 2026:
- **Fixed in PRs #18986 and #18936** (merged Jan 21, 2026)
- llama.cpp builds before this date will show ~8 tok/s (MLA fallback path)
- Builds after: full speed (~71 tok/s at empty context)
- **Minimum recommended build:** any build after 2026-01-21

### KV Cache per Token

GLM's MLA architecture uses ~54 KB per token for KV cache (measured empirically). This is higher than simpler architectures:

| Model | KV Cache/Token | At 64K context |
|-------|------:|------:|
| Nemotron-3-Nano | 2.25 KiB | 144 MB |
| Qwen3-30B-A3B | 36 KiB | 2.3 GB |
| GLM-4.7-Flash (MLA) | ~54 KB | ~3.4 GB |

GLM's larger KV cache contributes to its faster degradation at deep context compared to Nemotron.

## Agentic Capabilities

Tested in a 40-minute interactive session (~12 user interactions, ~25 tool calls):

| Capability | Result | Notes |
|------------|--------|-------|
| Web research (fetch + summarize) | ✅ Excellent | Multi-source synthesis, no hallucinations |
| Tool chaining (search → fetch → analyze) | ✅ Solid | Smooth multi-step workflows |
| CLI tool installation (npm/npx) | ✅ Good | Handled permission errors, found alternatives |
| File system operations (read/edit) | ✅ Good | Correct file edits with proper tool params |
| Debugging (PATH, permissions) | ✅ Good | Systematic investigation approach |
| Error recovery | ⚠️ Adequate | Needed user guidance for some edge cases |

### Quality Highlights
- **Research quality excellent**: Thorough web research with accurate summarization
- **No hallucinations observed**: All factual claims were grounded in fetched content
- **Tool parameter handling**: Correctly used `old_string`/`new_string` for edits after initial error

### Observed Limitations
- **Cannot disable thinking**: Always generates `<think>` blocks, adding latency
- **Search granularity**: Sometimes answered from prior context instead of searching
- **Permission awareness**: Didn't immediately recognize `clawdbot` vs `root` user context

## Comparison: GLM vs Nemotron vs Qwen3

All on 2× RTX 3060 12GB, `-sm layer -fa 1 -ctk q8_0 -ctv q4_0`:

### Speed (llama-bench tg128)

| Context | Qwen3 tok/s | Nemotron tok/s | GLM tok/s |
|--------:|------:|------:|------:|
| 0 | **99.1** | 95.9 | 71.1 |
| 2K | 81.7 | **99.3** | 65.8 |
| 4K | 70.8 | **96.8** | 62.9 |
| 8K | 55.7 | **92.4** | 57.5 |
| 16K | 38.9 | **84.6** | 53.5 |
| 32K | 23.8 | **72.1** | 44.6 |
| 64K | 13.4 | **55.3** | 33.3 |

### Verdict

| Use Case | Best Model | Why |
|----------|------------|-----|
| Short context (<2K) | Qwen3 | Fastest at empty cache |
| Medium context (2-50K) | Nemotron | Best speed retention |
| Long context (50K+) | Nemotron | 55 tok/s at 64K vs GLM's 33 |
| Code quality | GLM | SWE-bench 59.2% vs Qwen3 22% |
| Tool calling quality | GLM | τ²-Bench 79.5% vs others <50% |
| Interactive sessions | Nemotron | Predictable speed, reasoning control |
| One-shot research tasks | GLM | High quality at fresh context |

**Recommendation:** Use GLM for fresh-context subagent tasks (research, code review, analysis) where quality matters and context stays below ~40K. Use Nemotron for interactive sessions that grow to 100K+.

## Recommended Config

```bash
llama-server \
  --model /path/to/GLM-4.7-Flash-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 131072 --parallel 1 \
  --flash-attn on --jinja \
  --split-mode layer --gpu-layers 99 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --temp 1.0 --top-p 0.95 --min-p 0.01 --repeat-penalty 1.0
```

**Critical flags:**
- `--split-mode layer` — **mandatory** for PCIe dual-GPU (2.5× faster than `row`)
- `--flash-attn on` — required to fit in VRAM
- `--gpu-layers 99` — full GPU offload
- No `--reasoning-format` or `--reasoning-budget` — GLM doesn't support these

## Known Issues

### 🧠 Uncontrollable Thinking Tokens

**Severity:** Medium — adds 20-40% latency overhead

**Symptom:** Every response includes `<think>` blocks that consume generation budget. Cannot be disabled via llama-server flags.

**Impact:**
- Effective user-visible throughput is ~20-30% lower than measured TG tok/s
- For tool-call-heavy workflows, thinking overhead is proportionally larger
- At deep context (100K+), thinking at 14 tok/s means noticeable delays

**Mitigations:**
- None available in llama.cpp (GLM's chat template forces thinking)
- Accept the overhead and plan context budget accordingly
- For speed-critical tasks, use Nemotron with `--reasoning-budget 0` instead

### 📉 Steep Degradation at Deep Context

**Severity:** High — limits interactive usability

**Symptom:** TG speed drops from 71 tok/s (empty) to ~14 tok/s (113K context) in real serving.

**Root cause:** Large MLA KV cache (~54 KB/token) + full 131K allocation pressure

**Mitigations:**
1. Keep sessions below 50K context for interactive use
2. Use aggressive compaction settings
3. Use Nemotron for long-context workloads

### ⚠️ llama.cpp Build Requirement

**Severity:** Low (one-time)

**Symptom:** Builds before 2026-01-21 show ~8 tok/s due to broken MLA support.

**Fix:** Use any build after 2026-01-21 (PRs #18986, #18936)

## Hardware Tested

- 2× NVIDIA RTX 3060 12GB (Ampere, compute 8.6)
- Intel i5-7400 @ 3.00GHz, 4 cores
- 26GB system RAM
- CUDA 13.0, Driver 580.105.08
- llama.cpp build 91ea44e (Feb 2, 2026)

## Production Status

**As of 2026-03-01:** GLM-4.7-Flash is the **default model for all LocalBot agents** (localbot-labmaster, localbot-fraktalia, localbot-llmlab, localbot-planning, localbot-polis, ht). Fallback: Nemotron-30B-A3B on CPU (llama-local).

Previous default was Nemotron-3-Nano-30B-A3B, switched 2026-03-01 after multi-account Matrix fix enabled stable agent routing.

### NO_REPLY Quirk
GLM consistently appends `NO_REPLY` to the end of real responses (doesn't follow the "entire message" instruction). Required a gateway-level fix: `isSilentReplyText()` changed to strict exact-match `^\s*NO_REPLY\s*$`.

## Changelog

- **2026-03-01:** Promoted to default for all LocalBot agents. NO_REPLY suppression fix documented.
- **2026-02-20:** Initial profile. llama-bench context fill-up benchmarks (0-64K). Full 79-request production session analysis with per-request metrics, compaction analysis, thinking overhead measurement. Three-model comparison (GLM vs Nemotron vs Qwen3). MLA history documented.
