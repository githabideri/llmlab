# Experiments

Use this structure for each run:

**Goal → Setup → Commands → Observations → Metrics → Conclusion → Next**

## Recent key runs
- `2026-03-14-ik-llama-cpp-graph-mode-multi-gpu.md` — ik_llama.cpp graph mode testing: 3–4× speedup, VRAM leak fix, PCIe crash root cause on non-P2P topology
- `2026-03-14-qwen3.5-35b-a3b-vllm-pp3-concurrency.md` — vLLM PP=3 deployment + concurrency benchmarks on 3× RTX 3060
- `2026-03-09-concurrent-slot-asymmetry-investigation.md` — llama.cpp concurrent slot starvation analysis (layer mode)
- `2026-03-03-qwen3.5-35b-a3b-24gb-vision-retest.md` — confirms tuned 24GB text+tools+vision profile for Qwen3.5 (with caveats)
- `2026-02-12-ik-llama-cpp-vs-main-preliminary.md` — replacement-model benchmark pass (ik vs main), partial baseline closure + tomorrow handoff
- `2026-02-12-nemotron-thinking-gradient-abc.md` — A/B/C reasoning profile comparison
- `2026-02-12-nemotron-abc-executive-summary.md` — operator summary (time + tokens + tok/s)
- `2026-02-12-nemotron-abc-timing-token-breakdown.md` — full per-step timing/token breakdown
- `2026-02-09-nemotron-thinking-fix.md` — reasoning format mitigation notes
