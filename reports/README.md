# Reports

Date-stamped findings — investigations, deployments, benchmarks, hardware forensics, and infrastructure notes. Each file is a **snapshot of what we knew at the time**; no future maintenance needed. If something changes, write a new report rather than editing an old one.

> **Format note:** these are *not* homelab reports. There is **no YAML frontmatter** and **no `<type>_` filename prefix** — naming is just `YYYY-MM-DD-<kebab-slug>.md`. (The private homelab repo uses a different, frontmatter-based convention; its `report` skill is explicitly not to be applied here.)

## Categories

- **Model deployment** — configs, sweeps, and performance numbers for a specific model + hardware combo
- **Experiment / benchmark** — a single test run: *Goal → Setup → Commands → Observations → Metrics → Conclusion*
- **Hardware** — PCIe forensics, BIOS gotchas, thermal / bandwidth analysis
- **Infrastructure** — serving setups, container configs, multi-machine ops

The directory is flat and date-sorted. Use the per-run structure above for experiments/benchmarks; deployments and forensics are free-form.

## Recent

- `2026-08-30-dual-3060-35b-squeeze-27b-node.md` — 2-bit 35B proven resident on one 12 GB 3060 (PCIe counters, 43–81 t/s); 27B dense on the 3060 pair at ~80% of 3090 speed for the same wall draw, with the ctx-degradation curve, concurrency cliff, and the cache-artifact retraction
- `2026-08-30-vllm-cpu-kv-offload-hybrid-mamba-fails.md` — vLLM native CPU KV offload writes but never restores on hybrid SSM/GDN models (0% hit rate, pool shrank); scheduler-side blocker, reverted
- `2026-08-28-llama-cpp-ubatch-moe-single-gpu.md` — `-ub 2048` gives +22% prompt-processing for MoE+MTP on a single 12 GB GPU, tg-neutral
- `2026-08-27-qwen3.6-35b-a3b-dual-3060-optimization.md` — two 3060s, 50/50 tensor split, 100+ t/s
- `2026-08-12-ds4-rocm-gfx1103-build-success.md` — DS4 ROCm gfx1103 build success
- `2026-07-24-35b-mtp-laptop-setup.md` — Qwen3.6-35B at 28 tok/s on a laptop (MTP + Vulkan iGPU)
- `2026-06-30-qwen3.6-35b-a3b-mtp-single-3060.md` — MoE offload mechanics, n-cpu-moe sweep, config for single RTX 3060 12GB

## Highlights

- `2026-03-14-ik-llama-cpp-graph-mode-multi-gpu.md` — graph mode: 3–4× speedup, VRAM leak fix, PCIe crash root cause on non-P2P topology
- `2026-03-14-qwen3.5-35b-a3b-vllm-pp3-concurrency.md` — vLLM PP=3 deployment + concurrency benchmarks on 3× RTX 3060
- `2026-03-09-concurrent-slot-asymmetry-investigation.md` — llama.cpp concurrent slot starvation (layer mode)
- `2026-06-19-beellama-dflash-cutover.md` — BeeLlama DFlash cutover (now historic; see the Qwen3.8-27B report)
