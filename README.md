# llmlab

A hobbyist's notebook for running local LLMs on consumer GPUs — focused on what actually works, not what benchmarks promise.

## What this is

We run agentic LLM workloads on **2× RTX 3060 (24 GB VRAM)** and document everything: configs that work, models that don't, performance numbers from real serving (not just `llama-bench`), and the weird edge cases you only find by actually using these things.

**Focus areas:**
- **MoE (Mixture of Experts) models** — the sweet spot for interactive use on limited VRAM. Small active parameters = fast generation, large total parameters = good quality. Dense models are on the table too, but ~10 tok/s generation isn't thrilling for interactive work.
- **Agentic tool-calling** — not just chat, but models driving multi-step tool chains (web search → fetch → analyze → file ops). We test with [OpenClaw](https://github.com/openclaw/openclaw), which is demanding enough that if a model works here, it'll work in any general-purpose agentic setup.
- **Real serving metrics** — `llama-bench` numbers are a starting point. Real-world serving with prompt caches, thinking tokens, and growing context tells a different story. We measure both.

## Current setup

### Hardware
- **GPU server:** 2× RTX 3060 12 GB (24 GB total), Intel i5-7400, llama.cpp
- **CPU fallback:** Intel i5-8400T, 64 GB DDR4-2667, llama.cpp
- **Runtime:** llama.cpp (`llama-server`) with `-sm layer -fa 1 -ctk q8_0 -ctv q4_0`

### Active model (March 2026)
- **[GLM-4.7-Flash](models/glm-4.7-flash.md)** (Q4_K_XL, 17.5 GB) — 30B MoE with MLA, strong tool-calling (τ²-Bench 79.5%), 71→33 tok/s (0→64K context)
- **Fallback:** [Nemotron-3-Nano-30B-A3B](models/nemotron-3-nano-30b-a3b.md) (IQ4_NL) on CPU at ~5 tok/s

### Key finding: `-sm layer` is everything
On PCIe dual-GPU without NVLink, split-mode matters more than the model itself. `-sm layer` gives 2.5× the throughput of `-sm row`. If you have multiple consumer GPUs, this is the single most important flag.

## What we've learned

### Models tested

| Model | Active | Speed @0 | Verdict | Notes |
|-------|--------|----------|---------|-------|
| [GLM-4.7-Flash](models/glm-4.7-flash.md) | MoE ~4B | 71 tok/s | ✅ Production | Best tool-calling quality, always-on thinking |
| [Nemotron-3-Nano-30B](models/nemotron-3-nano-30b-a3b.md) | MoE 3B | 96 tok/s | ✅ Production | Best speed retention at depth (Mamba-2), controllable reasoning |
| [Qwen3.5-35B-A3B](models/qwen3.5-35b-a3b.md) | MoE 3B | ~95 tok/s | ❌ Failed | Infinite tool-call loops. DeltaNet NOT faster than Mamba-2 in practice |
| [Nanbeige4.1-3B](models/nanbeige4.1-3b.md) | Dense 3B | ~80 tok/s | ❌ Failed | Leaks `<think>` blocks, can't disable reasoning |
| [ZwZ-4B](models/zwz-4b.md) | Dense 4B | 77 tok/s | 🟡 Parked | Multimodal arch, untested for agentic |
| [Qwen3-Coder-REAP](models/qwen3-coder-next-reap-40b-a3b.md) | MoE 3B | ~90 tok/s | 🟡 Mixed | Good code, context degradation issues |

### Context degradation (the number that actually matters)

`llama-bench` at empty context is marketing. Here's what happens as context fills:

| Model | @0 | @16K | @32K | @64K | Degradation pattern |
|-------|------|------|------|------|---------------------|
| Nemotron (Mamba-2) | 96 | 85 | 72 | 55 | **-42% @64K** — best |
| GLM (MLA) | 71 | 54 | 45 | 33 | -53% @64K |
| Qwen3 (GQA) | 99 | 39 | 24 | 13 | -87% @64K — worst |

Nemotron's Mamba-2 architecture genuinely delivers on the "constant-time attention" promise. Qwen3's traditional GQA falls off a cliff.

### Real serving vs benchmarks

From a 79-request GLM production session:
- **Benchmark says:** 71 tok/s at empty context
- **Real serving:** 30 tok/s at 37K context, 14.4 tok/s at 113K
- **Gap:** 28-36% slower than `llama-bench` (server overhead, prompt cache, thinking tokens, KV pressure)
- **Compaction helps:** speed recovered 14.4 → 30.0 tok/s (2.1×) after context compaction

## Repo structure

```
models/          8 model profiles (GLM, Nemotron, Qwen3.5, ZwZ, Nanbeige, ...)
experiments/     14 experiment logs (context sweeps, quant comparisons, speed tests)
benchmarks/      Agentic benchmark suite (L0-L4: read/write → config → tool chains)
docs/            Architecture, runbook, systemd setup, troubleshooting, thinking policy
scripts/         Context sweep benchmarking, model info fetcher, server start scripts
```

### Model profiles (`models/`)

Each model gets a full write-up: architecture, quant selection rationale, speed at various context depths, real-world serving numbers, agentic capability test results, known issues, and recommended config. Not just "it works" — *how* it works, *where* it breaks, and *why*.

### Experiments (`experiments/`)

Raw experiment logs with goal → setup → commands → observations → metrics → conclusion. Covers context sweep benchmarks, quant quality comparisons, thinking budget tuning (A/B/C profiles), CPU vs GPU comparisons, and more.

### Benchmark suite (`benchmarks/`)

Five-level agentic benchmark:
- **L0:** Basic file read/write
- **L1:** Config summarization
- **L2:** Config patching (structured edit)
- **L3:** Benchmark output parsing (complex extraction)
- **L4:** Multi-step tool chain (search → fetch → analyze → write)

Designed to stress real agentic capabilities, not trivia or chat fluency.

## Quant selection philosophy

We pick the **highest-quality quant that fits 24 GB with 100K+ context headroom**. Every model profile includes a transparent comparison table showing all viable quants with exact sizes and math. No "just use Q4" hand-waving.

KV cache per token varies wildly between architectures (Nemotron: 2.25 KiB, GLM: ~54 KiB, Qwen3: 36 KiB) and dominates fitment at large context more than model size itself.

## Safety note

This repo is public. Do **not** commit:
- Private hostnames, IPs, or infrastructure details
- Credentials, tokens, or SSH keys
- Personal transcripts or sensitive logs

Keep examples generic and reproducible.

## Contributing

This is primarily a personal lab notebook, but issues and discussions are welcome if you're running similar hardware and have findings to share. The more data points on consumer GPU setups, the better.
