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

- `2026-08-12-ds4-rocm-gfx1103-build-success.md` — DS4 ROCm gfx1103 build success
- `2026-07-24-35b-mtp-laptop-setup.md` — Qwen3.6-35B at 28 tok/s on a laptop (MTP + Vulkan iGPU)
- `2026-06-30-qwen3.6-35b-a3b-mtp-single-3060.md` — MoE offload mechanics, n-cpu-moe sweep, config for single RTX 3060 12GB

## Highlights

- `2026-03-14-ik-llama-cpp-graph-mode-multi-gpu.md` — graph mode: 3–4× speedup, VRAM leak fix, PCIe crash root cause on non-P2P topology
- `2026-03-14-qwen3.5-35b-a3b-vllm-pp3-concurrency.md` — vLLM PP=3 deployment + concurrency benchmarks on 3× RTX 3060
- `2026-03-09-concurrent-slot-asymmetry-investigation.md` — llama.cpp concurrent slot starvation (layer mode)
- `2026-06-19-beellama-dflash-cutover.md` — BeeLlama DFlash cutover (now historic; see the Qwen3.8-27B report)
