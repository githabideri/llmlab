# docs/

Methodology and reference — the repo's only **maintained** documentation surface. `reports/` are frozen point-in-time snapshots: if a fact has changed, fix it here, don't patch a report.

| Doc | Purpose | Status |
|-----|---------|--------|
| [architecture](architecture.md) | Fleet layout: what runs where, storage, GPU topology | Active |
| [runbook](runbook.md) | Day-to-day ops: health, restart, rollback, MTP debugging, symptom→fix index | Active |
| [llama-cpp-systemd](llama-cpp-systemd.md) | Unit-file reference — what the units contain (the runbook says how to operate them) | Active |
| [benchmarks](benchmarks.md) | Durable benchmarking method and comparison discipline | Active |
| [multi-gpu-tensor-split](multi-gpu-tensor-split.md) | `--tensor-split` tuning; when to use layer vs tensor (2026-08) vs row (never on PCIe) | Active |
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
