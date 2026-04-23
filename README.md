# llmlab

A hobbyist's notebook for running local LLMs on consumer GPUs — focused on what actually works, not what benchmarks promise.

## What this is

We run agentic LLM workloads on **2× RTX 3060 12 GB + 1× RTX 3090 24 GB (48 GB VRAM)** and document everything: configs that work, models that don't, performance numbers from real serving (not just `llama-bench`), and the weird edge cases you only find by actually using these things.

> *We started with a dual 3060 setup (24 GB) and added an RTX 3090 (24 GB) in early March 2026 — unlocking dense 27B models and multi-GPU parallel serving.*

**Focus areas:**
- **MoE and hybrid models** — the sweet spot for interactive use on limited VRAM. Small active parameters = fast generation, large total parameters = good quality. Dense models are on the table too now that we have 48 GB to play with.
- **Agentic tool-calling** — not just chat, but models driving multi-step tool chains (web search → fetch → analyze → file ops). We test with [OpenClaw](https://github.com/openclaw/openclaw), which is demanding enough that if a model works here, it'll work in any general-purpose agentic setup.
- **Real serving metrics** — `llama-bench` numbers are a starting point. Real-world serving with prompt caches, thinking tokens, and growing context tells a different story. We measure both.
- **Multi-GPU optimization** — tensor-split tuning, compute buffer analysis, and [practical guides](docs/multi-gpu-tensor-split.md) for squeezing maximum context and parallelism out of consumer GPUs without NVLink.

## Hardware and serving setup

### Hardware
- **GPU server:** 2× RTX 3060 12 GB + 1× RTX 3090 24 GB (48 GB total), Intel i5-7400, PCIe x16 + x4 + x4 — [detailed hardware profile](docs/hardware/triple-3060.md)
- **CPU fallback:** Intel i5-8400T, 64 GB DDR4-2667, llama.cpp

### Runtime profiles
- **llama.cpp** remains the reference path for GGUF-based serving, tensor-split tuning, and long-context fitment work.
- **vLLM** is now part of the documented production story as well, especially for GPTQ-based serving and higher aggregate concurrency with native prefix caching.

### Validated highlights
- **[Qwen3.6-27B](models/qwen3.6-27b-rtx3090.md)** on llama.cpp (Dense, Q4_K_M, ~16 GB) — **single RTX 3090**, long-context profile at 204K context.
- **[Qwen3.5-27B](models/qwen3.5-27b.md)** on llama.cpp (Dense, Q5_K_XL, ~19 GB) — `--parallel 3 --ctx-size 393216` = **3 concurrent sessions × 131K context**.
- **[Qwen3.5-35B-A3B](models/qwen3.5-35b-a3b.md)** on vLLM (GPTQ-Int4, PP=3) — dedicated validation in [the PP=3 experiment](experiments/2026-03-14-qwen3.5-35b-a3b-vllm-pp3-concurrency.md).
- **CPU fallback:** [Nemotron-3-Nano-30B-A3B](models/nemotron-3-nano-30b-a3b.md).
- **Earlier 24 GB-era baselines:** [GLM-4.7-Flash](models/glm-4.7-flash.md), [Qwen3.5-35B-A3B](models/qwen3.5-35b-a3b.md).

### Key findings

**`-sm layer` is everything.** On PCIe multi-GPU without NVLink, split-mode matters more than the model itself. `-sm layer` gives 2.5× the throughput of `-sm row`. If you have multiple consumer GPUs, this is the single most important flag.

**`output.weight` lands on the last GPU.** In llama.cpp's split-mode layer, the output projection (~1+ GB) is hardcoded to the last GPU. This creates asymmetric VRAM pressure that must be compensated with tensor-split ratios. See our [multi-GPU tensor-split guide](docs/multi-gpu-tensor-split.md).

**`--parallel N` shrinks compute buffers.** More slots = smaller per-slot compute buffers, which frees VRAM for KV cache. On our 2×3060 setup, going from parallel 1→3 freed enough headroom for 3× the concurrent sessions at 131K each. The tradeoff: per-slot context shrinks proportionally.

## What we've learned

### Models tested

| Model | Main validated serving path | Arch | Verdict | Notes |
|-------|-----------------------------|------|---------|-------|
| [Qwen3.6-27B](models/qwen3.6-27b-rtx3090.md) | llama.cpp | Hybrid (recurrent+attn) | ✅ Production | Single RTX 3090, 204K long-context, multimodal tested |
| [Qwen3.5-35B-A3B](models/qwen3.5-35b-a3b.md) | vLLM + llama.cpp | MoE | ✅/🟡 Mixed by backend | Important current model: validated on vLLM PP=3; historically tighter and riskier on 24 GB llama.cpp configs |
| [Gemma 4 26B-A4B](models/gemma-4-26b-a4b.md) | llama.cpp | Hybrid attention / A4B MoE-style sparse activation | 🟡 Pilot | Dual-3060 text-only path validated; `q8_0/q8_0` KV was dramatically better than `q8_0/q4_0` on this setup |
| [GLM-4.7-Flash](models/glm-4.7-flash.md) | llama.cpp | MoE ~4B | ✅ Production | Best tool-calling quality in earlier 24 GB-era work |
| [Nemotron-3-Nano-30B](models/nemotron-3-nano-30b-a3b.md) | llama.cpp / CPU fallback | MoE (Mamba-2) | ✅ Production | Excellent speed retention at depth; useful fallback profile |
| [LFM2-24B-A2B](models/lfm2-24b-a2b.md) | llama.cpp | MoE | ❌ Failed | Extremely fast but unreliable for agentic work |
| [Nanbeige4.1-3B](models/nanbeige4.1-3b.md) | llama.cpp | Dense | ❌ Failed | Leaks `<think>` blocks, can't disable reasoning |
| [ZwZ-4B](models/zwz-4b.md) | llama.cpp | Dense | 🟡 Parked | Multimodal arch, untested for agentic |
| [Qwen3-Coder-REAP](models/qwen3-coder-next-reap-40b-a3b.md) | llama.cpp | MoE | 🟡 Mixed | Good code, context degradation issues |

### Context degradation (the number that actually matters)

`llama-bench` at empty context is marketing. Here's what happens as context fills:

| Model | @0 | @16K | @32K | @64K | Degradation pattern |
|-------|------|------|------|------|---------------------|
| Nemotron (Mamba-2) | 96 | 85 | 72 | 55 | **-42% @64K** — best |
| GLM (MLA) | 71 | 54 | 45 | 33 | -53% @64K |
| Qwen3 (GQA) | 99 | 39 | 24 | 13 | -87% @64K — worst |

Nemotron's Mamba-2 architecture genuinely delivers on the "constant-time attention" promise. Qwen3's traditional GQA falls off a cliff. Qwen3.5-27B's hybrid approach (mostly recurrent) should sit closer to Nemotron — benchmarks pending.

### Real serving vs benchmarks

From a 79-request GLM production session:
- **Benchmark says:** 71 tok/s at empty context
- **Real serving:** 30 tok/s at 37K context, 14.4 tok/s at 113K
- **Gap:** 28-36% slower than `llama-bench` (server overhead, prompt cache, thinking tokens, KV pressure)
- **Compaction helps:** speed recovered 14.4 → 30.0 tok/s (2.1×) after context compaction

## Docs & guides

| Guide | Description |
|-------|-------------|
| [Multi-GPU Tensor-Split](docs/multi-gpu-tensor-split.md) | How to optimize layer distribution across GPUs for llama.cpp — ceiling testing, `output.weight` gotcha, `--parallel` effects |
| [Hardware: Triple 3060](docs/hardware/triple-3060.md) | Our 2×3060 + 3090 setup — validated llama.cpp configs, vLLM PP=3 note, VRAM budgets, capacity planning |
| [Architecture](docs/architecture.md) | System architecture overview |
| [Runbook](docs/runbook.md) | Start/stop servers, common operations |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |
| [Qwen3.5-35B-A3B vLLM PP=3 experiment](experiments/2026-03-14-qwen3.5-35b-a3b-vllm-pp3-concurrency.md) | Dedicated write-up for the validated vLLM deployment profile and concurrency behavior |

## Repo structure

```
models/          Model profiles (GLM, Nemotron, Qwen3.5-27B, Qwen3.5-35B, LFM2, ...)
experiments/     Experiment logs (context sweeps, quant comparisons, speed tests)
benchmarks/      Agentic benchmark suite (L0-L4: read/write → config → tool chains)
docs/            Guides, runbook, troubleshooting, hardware profiles
scripts/         Context sweep benchmarking, model info fetcher, server start scripts
web/             Web UI for running and monitoring benchmarks
```

### Web UI (`web/`)

**Interactive benchmark dashboard** built with FastAPI + htmx, styled to match llama.cpp's web interface:
- 🤖 Model and server status monitoring
- 🚀 Run context ladder benchmarks with live streaming output
- 📈 Browse historical test results
- ⚙️ View current server configuration

**Quick start:**
```bash
cd web
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./start.sh
```
Then open http://localhost:8000 — see [web/README.md](web/README.md) for details.

### Model profiles (`models/`)

Each model gets a full write-up: architecture, quant selection rationale, speed at various context depths, real-world serving numbers, agentic capability test results, known issues, and recommended config. Not just "it works" — *how* it works, *where* it breaks, and *why*.

### Benchmark suite (`benchmarks/`)

Five-level agentic benchmark:
- **L0:** Basic file read/write
- **L1:** Config summarization
- **L2:** Config patching (structured edit)
- **L3:** Benchmark output parsing (complex extraction)
- **L4:** Multi-step tool chain (search → fetch → analyze → write)

Designed to stress real agentic capabilities, not trivia or chat fluency.

## Quant selection philosophy

We pick the **highest-quality quant that fits with 100K+ context headroom**. Every model profile includes a transparent comparison table showing all viable quants with exact sizes and math. No "just use Q4" hand-waving.

KV cache per token varies wildly between architectures (Nemotron: 2.25 KiB, Qwen3.5-27B: ~12 KiB for 16/64 attn layers, GLM: ~54 KiB, Qwen3: 36 KiB) and dominates fitment at large context more than model size itself.

## Hardware history

| Period | Setup | VRAM | Key models |
|--------|-------|------|------------|
| Jan–Feb 2026 | 2× RTX 3060 12 GB | 24 GB | GLM-4.7-Flash, Nemotron-30B, Qwen3.5-35B-A3B |
| Mar 2026+ | 2× RTX 3060 + 1× RTX 3090 | 48 GB | Qwen3.5-27B (dense), 3-slot parallel serving |
| Apr 2026 | 1× RTX 3090 24 GB | 24 GB | Qwen3.6-27B (dense), 204K long-context |

The RTX 3090 addition opened up dense models and multi-session serving that wasn't feasible at 24 GB. See [hardware profile](docs/hardware/triple-3060.md) for the full story.

## Safety note

This repo is public. Do **not** commit:
- Private hostnames, IPs, or infrastructure details
- Credentials, tokens, or SSH keys
- Personal transcripts or sensitive logs

Keep examples generic and reproducible.

## Contributing

This is primarily a personal lab notebook, but issues and discussions are welcome if you're running similar hardware and have findings to share. The more data points on consumer GPU setups, the better.
