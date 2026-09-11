# Hardware Profile: GPU Server

**Configuration:** 2× RTX 3090 24 GB (48 GB VRAM)  
**Use Case:** Primary multi-GPU LLM inference (vLLM tensor-parallel 2)  
**Status:** Active — Qwen3.8-27B on the dual-3090 vLLM (since 2026-09-08); the 35B-A3B interim home is the secondary box

---

## Specifications

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 5 5600X (6C/12T) |
| Motherboard | MSI MEG X570 ACE |
| RAM | 64 GB (2× 32 GB DDR4-2933 SO-DIMM via SO-DIMM-to-DIMM adapters; 4×8 GB at 2133 before 2026-08-26) |
| System disk | 512 GB SATA SSD (ZFS rpool) — chipset-attached |
| Model disk | 1 TB WD Green SATA SSD (ext4) — chipset-attached, shares the 8 GB/s uplink with GPU 2 (mount name `/mnt/usb-ssd` is historical) |
| GPU 0 | RTX 3090 24 GB (new, 2026-09-08) — CPU PCIe 4.0 x8, bus 2d |
| GPU 1 | RTX 3090 24 GB (original) — CPU PCIe 4.0 x8, bus 2e (PCI_E1; the board allocates x8 per CPU slot) |
| Total VRAM | 48 GB |

> **PCIe note:** idle GPUs report Gen1 links — this is normal NVIDIA power management; they retrain to full speed under load.

## What runs where

| GPU | Workload | Port |
|-----|----------|------|
| 2× RTX 3090 (TP2) | Qwen3.8-27B — vLLM 0.28.0 (W4A16-AutoRound, MTP k=3, 262K fp8 KV, vision), 250 W/card, one dedicated LXC seeing both cards | 8082 |

Both cards are CPU-direct Gen4 x8 and share one vLLM engine (tensor-parallel 2); there is no NVLink on this board, so TP collectives run over PCIe (NCCL). The freed chipset slot de-congested the chipset uplink that used to be shared with the model SSD. The former dual-3060 llama.cpp unit (35B + on-demand 27B-Uncensored) was dismantled with the cards — its consumer fleet moved to the secondary/backup boxes. A full dual-3090 benchmark campaign (and the 35B placement decision) follows.

## Reliability notes (2026-09)

- **Two unexplained hard crashes in two days** (2026-09-08 and 2026-09-10). The 09-08 one killed an 8-hour benchmark window and a 92 GB model load. The 09-10 one hit mid-campaign while the box was *measurably healthy* (10-second telemetry heartbeat: both GPUs idle at 49/64 °C, no memory pressure, load 7 — ten seconds before the journal ended mid-sentence). No kernel panic captured (kdump not installed), no MCE/EDAC/GPU-Xid/hung-task/IO errors, SMART passed, no power event on the rest of the network. **Prime suspect: the power path (PSU/cable) or an unlogged motherboard fault.** Both crashes sat adjacent to the heaviest mixed VRAM + host-RAM workloads this box has run — correlation, not proof.
- **Mitigations in place:** a 10-second telemetry sampler (v2: PCIe gen/width, swap, error-marker counters; ~4 h ring buffer) so any future death leaves a last heartbeat; **kdump armed** (a kernel crash now produces a vmcore in `/var/crash` — no reboot needed because `crashkernel` was already on the PVE command line); **rasdaemon** active (on this X570 board EDAC/PCIe-AER are not exposed, so its value is MCE persistence); a **boot-forensics collector** that, after any abnormal boot, auto-bundles the previous journal, dmesg, kdump, rasdaemon, SMART and — via Home Assistant — the **wall-power window** from the smart plug (the plug telemetry was the decisive evidence for both known crashes: continuous power at 10 s resolution through both death windows → both deaths are internal to the box); campaign recovery designed to survive a host death (manifest on the USB model SSD, systemd auto-start of the production unit, deadline watchdog). **Verified in practice on 2026-09-11:** the first unattended campaign after the crashes ran to completion with the whole stack on standby and zero intervention — the next crash will be the one that is diagnosable. **Still open:** a physical PSU inspection (cable count/daisy-chains, label/serial, connector condition, wall path) and a power sweep of the GPU power limits — the software-visible picture is in the private repo.

- **Qwen3.8-27B (dual 3090, vLLM TP2):** [model card](../../models/qwen3.8-27b-rtx3090.md)
- **Qwen3.6-35B-A3B (interim: secondary box):** [model card](../../models/qwen3.6-35b-a3b.md)

## Methodology

- **Placement (fitter vs manual, split modes, `output.weight`, `--parallel`):** [multi-gpu-model-placement](../multi-gpu-model-placement.md)
- **KV-cache budgeting:** [kv-cache-sizing](../kv-cache-sizing.md)

## History

- **2026-03 era:** this machine ran an i5-7400 with 3× RTX 3060 (36 GB). The 3-GPU vLLM PP=3 and triple-GPU validation write-ups remain in [reports](../../reports/) — `2026-03-14-qwen3.5-35b-a3b-vllm-pp3-concurrency.md`, `2026-03-03-triple-gpu-validation.md`.
- **2026-07-17:** swapped to Ryzen 5 5600X + 1×3090 / 2×3060 (48 GB).
- **2026-08-21:** Qwen3.8-27B production moved to vLLM 0.27.1 in a dedicated LXC; stock llama.cpp (5f754ea, Q4_K_M + MTP) kept as dormant rollback.
- **2026-08-26:** RAM 4×8 GB (2133 MT/s) → 2× 32 GB DDR4-2933 SO-DIMM via SO-DIMM-to-DIMM adapters (64 GB), plus BIOS rework (downgrade to 1R0, CSM boot, WOL re-armed).
- **2026-09-08:** 3060 pair removed; second RTX 3090 installed in the freed CPU x8 slot (board now 2× 3090, both CPU-direct x8, 48 GB). Qwen3.8-27B production became vLLM 0.28.0 **tensor-parallel 2** on both cards (262K ctx, vision, 250 W/card). The chipset-attached GPU slot is free again — the chipset uplink no longer contends with the model SSD.
