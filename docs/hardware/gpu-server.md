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
