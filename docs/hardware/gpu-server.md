# Hardware Profile: GPU Server

**Configuration:** 2× RTX 3090 24 GB (48 GB VRAM)  
**Use Case:** Primary multi-GPU LLM inference (vLLM tensor-parallel 2)  
**Status:** Active — Qwen3.8-27B on the dual-3090 vLLM (since 2026-09-08); the 35B-A3B interim home is the secondary box  
**Last verified:** 2026-11-07/08 (drive inventory: only the Intel 660p remains, the 11-06 "Samsung rpool" reading was wrong; card swap; P2P/NCCL measured; power caps 220 W)

---

## Specifications

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 5 5600X (6C/12T) |
| Motherboard | MSI MEG X570 ACE |
| RAM | 64 GB (2× 32 GB DDR4-2933 SO-DIMM via SO-DIMM-to-DIMM adapters; 4×8 GB at 2133 before 2026-08-26) |
| System disk | 512 GB **Intel 660p** M.2 NVMe (ZFS rpool, ~16 % used, healthy) — chipset PCIe 3.0 x4 (bus 22) — **the only NVMe left in the box**: the Samsung PM991 (retired NixOS LUKS drive, clean) and the failed **WD SN570 2 TB** (retired PVE host's LVM, 100 media errors, kept in stasis, never written) were removed 2026-11-07; the pre-reinstall SATA rpool lives as read-only `oldrpool` |
| Model disk | 1 TB WD Green SATA SSD (ext4, `/mnt/internal-ssd`) — chipset-attached, ~550 MB/s; the fast model lane is a ZFS dataset on the rpool NVMe (`models-nvme`, 300 GB) holding working/serving copies, the SATA disk the cold copies + catalog |
| GPU 0 | RTX 3090 24 GB (ASUS TUF, re-pasted 2026-09-28) — CPU PCIe 4.0 x8, bus 2d (lower CPU slot, cooler) |
| GPU 1 | RTX 3090 24 GB (MSI VENTUS, original, re-pasted 2026-09-28) — CPU PCIe 4.0 x8, bus 2e (PCI_E1, upper CPU slot, structurally hotter) |
| Total VRAM | 48 GB |

> **PCIe note:** idle GPUs report Gen1 links — this is normal NVIDIA power management; they retrain to full speed under load. The only NVMe sits on chipset PCIe 3.0 x4 (bus 22); the other M.2 slots are empty (2026-11-07); only the GPUs are CPU-direct (buses 2d/2e); the CPU M.2 slot (PCIe 4.0) is empty. **The two GPUs have no P2P path** (separate CPU root ports, no NVLink, IOMMU translated + ACS): vLLM's collectives run over a NCCL shared-memory transport (measured; details in Reliability).

## What runs here

| GPU | Workload | Port |
|-----|----------|------|
| 2× RTX 3090 (TP2) | Qwen3.8-27B — **live 2026-11-08: vLLM 0.32 PTH per-token-head INT8 KV (reversible trial; 0.28.0 fp8 is the one-step rollback)**, one dedicated LXC seeing both cards (config, power, numbers: [model card](../../models/qwen3.8-27b-rtx3090.md), [2026-11-08 report](../../reports/2026-11-08-dual3090-engine-ladder-pth-int8-kv.md)) | 8080 (vllm-mux front; engine at 8083) |

Both cards are CPU-direct Gen4 x8 and share one vLLM engine (tensor-parallel 2); there is no NVLink on this board, so TP collectives run over PCIe (NCCL). The freed chipset slot de-congested the chipset uplink that used to be shared with the model SSD. The former dual-3060 llama.cpp unit (35B + on-demand 27B-Uncensored) was dismantled with the cards — its consumer fleet moved to the secondary/backup boxes; the box's remaining llama.cpp unit is the dormant 27B rollback. The dual-3090 benchmark campaign landed 2026-09-10/11 ([reports](../../reports/)); the 35B interim home is the secondary box (see its [card](../../models/qwen3.6-35b-a3b.md)).

## Reliability notes

- **Power limits: 220 W on both cards (current).** 250 W was re-enabled in the 09-28/29 window, but the live caps read 220 W at the 2026-11-07 audit (250 W had evidently been reverted on the box; the docs and the box disagreed, as the card changelog of 2026-09-28/29 itself notes). A matched 3-repetition 16K control measured **+4.4 % decode at 220 → 250 W** (157.92 → 164.80 t/s, peaks 74/67 °C, no thermal event) — a real small lever; the box stays at 220 W until the open-air riser rehouse makes slot position moot, because the upper-slot card already runs at the 83 °C SW-thermal target at 220 W.
- **The +10–20 °C offset follows the PCIe slot, not the card — confirmed in every layout tested, including the 2026-11-07 physical swap.** After the 11-07 swap the MSI VENTUS sits in the hot upper CPU slot (CUDA 1, `2e:00`: 86–89 °C under sustained load with intermittent SW slowdown) and the ASUS TUF in the cooler lower slot (CUDA 0, `2d:00`: 60–62 °C at the same caps). The 09-28 story ("the degraded card was a position effect") stands; the current layout is the deliberate one (the purchased card in the cooler slot). Structural fix: the planned riser-rig relocation.
- **GPU communication floor (measured 2026-11-07/08, model stopped, read-only):** `cudaDeviceCanAccessPeer` is **false in both directions** (separate CPU root ports, IOMMU translated, ACS on) — no CUDA P2P. vLLM therefore runs custom all-reduce disabled on a **PYNCCL shared-memory (SHM/direct/direct) transport** — host memory, not TCP. Standalone two-rank allreduce: 48.7 µs @32 KiB, 4.37 ms @32 MiB (≈7.7 GB/s payload). A **one-boot `amd_iommu=off` experiment is fully designed (boot state captured; rollback = rebooting the original entry) but shelved** as its own maintenance window — and the measured 102.8 t/s without P2P shows it is an optimization, not a prerequisite. Do not run wildcard `setpci` ACS-disabling scripts on a hypervisor host; do not force NCCL distance variables to create peer access CUDA reports unavailable.
- **NVMe inventory (2026-11-07):** two secondary drives were removed from the box — the retired NixOS LUKS drive (Samsung PM991, clean, kept) and a **failed WD SN570 2 TB with 100 media errors** (a retired PVE host's LVM; data disposable; kept in stasis, never written). The box now runs its ZFS root on the single boot NVMe (Intel 660p 512 GB, ~16 % used, healthy). No inference-path impact.
- **Two unexplained hard crashes in two days** (2026-09-08 and 2026-09-10). The 09-08 one killed an 8-hour benchmark window and a 92 GB model load. The 09-10 one hit mid-campaign while the box was *measurably healthy* (10-second telemetry heartbeat: both GPUs idle at 49/64 °C, no memory pressure, load 7 — ten seconds before the journal ended mid-sentence). No kernel panic captured (kdump not installed at the time), no MCE/EDAC/GPU-Xid/hung-task/IO errors, SMART passed, no power event on the rest of the network. **Prime suspect: the power path (PSU/cable) or an unlogged motherboard fault.** Both crashes sat adjacent to the heaviest mixed VRAM + host-RAM workloads this box has run — correlation, not proof.
- **Mitigations in place:** a 10-second telemetry sampler (v2: PCIe gen/width, swap, error-marker counters; ~4 h ring buffer) so any future death leaves a last heartbeat; **kdump armed**; **rasdaemon** active (on this X570 board EDAC/PCIe-AER are not exposed, so its value is MCE persistence); a **boot-forensics collector** that, after any abnormal boot, auto-bundles the previous journal, dmesg, kdump, rasdaemon, SMART and — via Home Assistant — the **wall-power window** from the smart plug (the plug telemetry was the decisive evidence for both known crashes: continuous power at 10 s resolution through both death windows → both deaths are internal to the box); campaign recovery designed to survive a host death (manifest on the USB model SSD, systemd auto-start of the production unit, deadline watchdog) — **verified in practice on 2026-09-11** (first unattended campaign after the crashes ran to completion, zero intervention). **Still open:** a physical PSU inspection (cable count/daisy-chains, label/serial, connector condition, wall path). The GPU thermal service of the original is **done** (both cards re-pasted 2026-09-28).
- **Watchdog (container-level):** the restart-on-`/health`-failure watchdog had a **clock-domain bug** (systemd monotonic **µs** compared against a 900-**s** grace, epoch-s **s** markers aged against the µs counter) that restarted the engine mid-load on one 2026-11-07 boot; fixed in place (one epoch-seconds domain, µs→s conversion). General trap: `/proc/uptime`×1e6, systemd monotonic stamps, and `date +%s` are three different clock domains.

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
