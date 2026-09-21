# docs/

Methodology and reference — the repo's only **maintained** documentation surface. `reports/` are frozen point-in-time snapshots: if a fact has changed, fix it here, don't patch a report.

| Doc | Purpose | Status |
|-----|---------|--------|
| [architecture](architecture.md) | Fleet layout: what runs where, storage, GPU topology | Active |
| [runbook](runbook.md) | Day-to-day ops: health, restart, rollback, MTP debugging, symptom→fix index | Active |
| [systemd](systemd.md) | Serving unit reference (vLLM 27B, llama.cpp 35B, 3060-box 27B D-CFR, model-mux) — what the units contain; the runbook says how to operate them | Active |
| [benchmarks](benchmarks.md) | Durable benchmarking method and comparison discipline | Active |
| [multi-gpu-model-placement](multi-gpu-model-placement.md) | Placement strategy: fitter vs manual, layer/tensor/row, heterogeneous balancing, expert spill, PCIe validation (renamed from multi-gpu-tensor-split, 2026-09) | Active |
| [kv-cache-sizing](kv-cache-sizing.md) | KV memory math per architecture; quant tradeoffs | Active |
| [thinking-policy](thinking-policy.md) | When to enable/limit/disable reasoning in serving | Active |
| [cutover-checklist](cutover-checklist.md) | Reusable model/runtime swap checklist | Active |
| [forensics-runbook](forensics-runbook.md) | Crash/freeze evidence collection (intentionally generic) | Active |
| [llama-cpp-grammar-workaround](llama-cpp-grammar-workaround.md) | Build-specific grammar repetition-threshold fix | Reference |
| [cpu-performance-tips](cpu-performance-tips.md) | CPU-only inference tuning (fleet is GPU; kept as general advice) | Reference |
| [hardware/](hardware/README.md) | Fleet profiles: physical box specs and what's deployed | Active (index) |
| [legacy/backend-beellama](legacy/backend-beellama.md) | BeeLlama DFlash backend — out of production since 2026-08-15 | Frozen |
| [legacy/ik-llama.cpp-features](legacy/ik-llama.cpp-features.md) | ik_llama.cpp CPU fork features — never in production here | Frozen |

Frozen docs are kept for reference and deliberately not updated; a banner at the top of each says what superseded it.

## Fact ownership (where each mutable thing lives)

Docs rot when a *mutable* fact is described in more than one place — a 2026-09 power-limit change sat stale in six of them for two weeks, and a port move in three. The rule: **one owner per fact class; everything else links.** The sanctioned mirrors are the index tables (top README "Currently serving", `models/README.md`, the `hardware/README.md` index row) — a mirror row is updated **in the same commit as the source change**, never as an afterthought.

| Fact class | Owner |
|---|---|
| Box & physical (CPU, RAM, GPU lineup, PCIe, **power limits**, reliability) | [hardware/<box>.md](hardware/README.md) profiles — index row mirrors the current lineup only |
| Model serving (quant, backend, ctx, **port/endpoint**, status, **measured performance**) | the [model card](../models/README.md) — its **changelog is the decision log**; volatile values it doesn't own are pointers, not copies |
| Unit mechanics (flags, env, paths) | [systemd](systemd.md) — current values it doesn't own (power limits) are pointers to the hardware profile |
| Layout (which box runs which service) | [architecture](architecture.md) — shape only, no numbers (ports are serving facts owned by the cards; profile tables mirror them) |
| Day-to-day ops (commands, symptom → fix) | [runbook](runbook.md) |
| Methodology (bench method, KV math, placement, thinking policy) | the other docs here |
| Frozen evidence (campaigns, incidents) | [reports/](../reports/README.md) — never rewritten; supersede-append only |

**Verify volatile values before writing them.** Before a doc claims watts, a port, a GPU lineup, or a t/s figure, check it live: `nvidia-smi` (limits/temps), `systemctl`/`curl` (units/ports), the llm-hub or a quick request (served-model numbers). Out-of-band changes (the 250→220 W drop happened outside any doc) are exactly how the multi-copy staleness is born; a 30-second check is the fix.
