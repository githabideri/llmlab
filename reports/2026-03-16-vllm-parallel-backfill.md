# vLLM Parallel Backfill Stress Test

**Date:** 2026-03-16
**Hardware:** 3× RTX 3060 12GB (36GB total), vLLM PP=3
**Model:** Qwen3.5-35B-A3B-GPTQ-Int4
**Task:** Write daily notes for 6 agents × 7 days = 42 notes

## Results Summary

| Version | Strategy | Concurrency | Success | Duration | Key Issue |
|---------|----------|-------------|---------|----------|-----------|
| v1 | Per-agent, all parallel | 39 | 0% | 300s timeout | KV cache overwhelmed |
| v2 | Per-agent, 6 parallel | 6 | 0% | 300s timeout | PP overhead per agent |
| v3 | Per-agent, 3 parallel | 3 | ~30% | 600s+ | Slow TTFT, timeouts |
| v4 | Single-agent, 6 parallel | 6 | 95% | 34 min | Prefix cache leverage |
| v5 | Single-agent + memory pre-inject | 6 | 100% | 69s | Full coverage + quality |

## vLLM Metrics (Full Session)

| Metric | Baseline | Final | Notes |
|--------|----------|-------|-------|
| Prompt tokens processed | 1.24M | 17.5M | +16.3M total |
| Generation tokens | 7.5K | 128K | +120.5K total |
| Prefix cache hits | 0 | 11.6M | 66% of all PP tokens cached |
| Requests completed | — | 313 | Zero failures at vLLM level |
| Preemptions | 0 | 0 | Never hit KV pressure |
| Peak KV cache usage | — | 44% | Plenty of headroom |
| Peak running requests | — | 3 | Sweet spot for PP=3 |
| Peak waiting requests | — | 5 | Queue drained steadily |

## Key Findings

### 1. Prefix Cache is the #1 Lever for Batch Workloads

When multiple requests share the same system prompt (agent identity), vLLM's automatic prefix caching eliminates redundant prompt processing. Measured impact:

- **v3 (per-agent):** Each of 6 agents has unique system prompt → 0% cache sharing → avg TTFT 68s
- **v4 (single-agent):** All requests share one system prompt → 88% cache hit → dramatic TTFT reduction
- **v5 (single-agent + warm):** With primer request → 96% cache hit → 69s for 42 requests total

**Architectural takeaway:** For batch operations, route through a single agent identity. The PP savings are enormous (6× reduction in prompt processing).

### 2. Optimal Concurrency: 2 Per GPU

On 3× RTX 3060 with PP=3:
- **3 concurrent (1/GPU):** Stable but underutilizes scheduling
- **6 concurrent (2/GPU):** Sweet spot — 44% KV cache, zero preemptions, steady queue drain
- **39 concurrent:** KV cache explosion → cascading timeouts → 0% completion

### 3. PP:TG Ratio Reveals the Bottleneck

Agent workloads are heavily prompt-dominated:
- **v3:** 213:1 PP:TG ratio — almost all GPU time spent on prompt ingestion
- **v4+v5:** 82:1 PP:TG ratio — prefix caching cuts PP work by 2.6×

With typical agent system prompts of 30-50K tokens, prompt processing dominates wall-clock time. Prefix caching directly attacks this.

### 4. Quality Requires Evidence Injection

Model quality correlated with available data:
- Agents with rich workspace files (detailed memory, session logs) → 7-10/10 quality
- Agents with sparse workspaces → defaulted to "quiet day" templates → 1/10

**Fix:** Pre-grep MEMORY.md for relevant dates and inject excerpts into the prompt. Model can't hallucinate "no activity" when evidence is in-context.

## TTFT Distribution (Full Session, 313 Requests)

- Average TTFT: 42s (skewed by early v1-v3 runs)
- v4+v5 TTFT: significantly lower due to prefix caching
- 16% of requests had TTFT >80s (mostly v1-v3 era)

## Hardware Observations

- VRAM stable throughout: 11.1-11.3 / 10.7-10.8 / 11.7 GiB per GPU
- GPU utilization: 99-100% during active processing, drops during PP pauses
- GPU2 occasionally showed 0% utilization during PP phases (pipeline bubble in PP=3)
- No VRAM growth or leaks over 2+ hours of sustained load

## Recommendations

1. **Batch tasks:** Always use single-agent routing for prefix cache benefits
2. **Concurrency:** Cap at 2× GPU count (6 for 3-GPU setup)
3. **Timeouts:** 600s is marginal; 900s recommended for complex agent tasks
4. **Quality:** Pre-inject relevant memory/context excerpts; don't rely on model to search
5. **Monitoring:** Track `prefix_cache_hits_total` as primary efficiency metric
