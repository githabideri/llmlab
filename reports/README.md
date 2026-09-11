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

- `2026-09-11-multi-image-vision-ceiling.md` — research + campaign design for the **multi-image** ceiling on the 27B (written for external review): verified our model is `Qwen3VLProcessor` with **32×32 px/token** (patch 16×merge 2) and a **16 Mpx / 16,384-token default** that our 2 Mpx override cuts 8×; the 0.95 crash was an **encoder-activation → GDN/Mamba** OOM, not a count limit; the size-hints / `mm_processor_cache_gb` / `--mm-encoder-tp-mode` knob set (with #20123 / #27706 as the same failure mode elsewhere); a 3-axis campaign (count×pixel × gpu-util × encoder-tp) and a 7-goal decision matrix with the open questions we want feedback on

- `2026-09-11-dual3090-v2-operating-envelope.md` — operating-envelope campaign (cloud-model-orchestrated; box held all night — first unattended run after the two crashes, recovery stack on standby, zero intervention): **MTP knee at k=3** (67/113/151/168 t/s for k=0/2/3/4 on the 8192 profile — k4 only +11%, prod left untouched), **a 64K prefill in flight degrades running decode ~52%** (ITL 25.8→39.2 ms, full recovery after — the key 8192-vs-2048 input), **TP1 ≈ TP2** (149.8 vs 147.8 t/s at single-user load → one 3090 freeable for a second service), 32K prefix-cache 30× win (timing-sensitive); dev-phase field fixes (usage parsing, E2BIG, TP1 contract) logged by the orchestrator

- `2026-09-10-dual3090-overnight-campaign.md` — first unattended overnight run on the new 2×3090 box, orchestrated by a cloud model decoupled from the fleet: 27B profile fully attributed (MTP = 2.1×, eager = 3.4× penalty, 8192-batched wins at 16K → promoted to production), 640K soak (7 preemptions; 14–15 min TTFT per 43K-token prompt — the real cost of 262K), 128K ladder real / 256K a documented false-complete; Flash-Next 125B confirmed a **wall** on the homogeneous pair (25.75 GB unsplitable layer buffer > 24.37 GB card, all split modes, two builds); the host **hard-crashed mid-run with zero software trace** (second unexplained reboot in two days — PSU prime suspect) and the three-layer recovery design absorbed it with zero data loss
- `2026-09-07-llm-hub-adaptive-vision-routing.md` — four-day silent vision outage (static primary/fallback on a `--models-max 1` router; evicted model + slot-pinned fallback) → llm-hub advisor endpoint `GET /api/vision/route`: read-only, payload-free, two capacity classes (IMMEDIATE > QUEUED), filter-then-score, no backend mutation in Phase 1; measured fleet vision speeds (27B = 2× image tokens = worst route, not second-best); advisor-not-gateway rationale
- `2026-09-06-whisperx-pascal-dual-gpu-benchmark.md` — WhisperX large-v3-turbo (int8_float32) on 2× 2 GB Pascal (GTX 1050 + GT 1030, CC 6.1): ASR at 10.7× RT, full pipeline 7.1× RT; GPU placement proven per-stage; X-vector diarization replaces pyannote (322× faster, 31 MiB vs 1.5 GB)
- `2026-09-02-qwen4exp-flash-next-three-gpu-campaign.md` — 125 B Qwen4Exp MoE (51 B PLE table) on 3060/3060/3090: the master-build fitter's auto expert-spill beats every manual placement knob (30.2 t/s decode, ~390 prefill, +17 % MTP, ~1 Wh/1K tokens); last-`--override-tensor`-wins and fitter-disabling pitfalls; n-gram is code-toxic
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
