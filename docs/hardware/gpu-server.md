# Hardware Profile: GPU Server

**Configuration:** 1× RTX 3090 24 GB + 2× RTX 3060 12 GB (48 GB VRAM)  
**Use Case:** Primary multi-GPU LLM inference (llama.cpp)  
**Status:** Active — Qwen3.8-27B on the 3090, Qwen3.6-35B-A3B on the dual 3060

---

## Specifications

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 5 5600X (6C/12T) |
| Motherboard | MSI MEG X570 ACE |
| RAM | 64 GB (2× 32 GB DDR4-2933 SO-DIMM via SO-DIMM-to-DIMM adapters; 4×8 GB at 2133 before 2026-08-26) |
| System disk | 512 GB SATA SSD (ZFS rpool) |
| Model disk | ~1 TB SSD (ZFS) |
| GPU 0 | RTX 3090 24 GB — CPU PCIe 4.0 x16 |
| GPU 1 | RTX 3060 12 GB — CPU PCIe 4.0 x8 |
| GPU 2 | RTX 3060 12 GB — chipset PCIe 4.0 x4 |
| Total VRAM | 48 GB |

> **PCIe note:** idle GPUs report Gen1 links — this is normal NVIDIA power management; they retrain to full speed under load.

## What runs where

| GPU | Workload | Port |
|-----|----------|------|
| RTX 3090 | Qwen3.8-27B — vLLM 0.27.1 (W4A16-AutoRound, MTP k=3, 160K fp8 KV, text-only), dedicated LXC | 8080 |
| 2× RTX 3060 | Qwen3.6-35B-A3B (UD-IQ4_XS, MTP variant), tensor-split 50/50 + vision, MTP n=3, 256K ×2 | 8081 |

The 3090 is a standalone single-GPU endpoint in its own LXC; the two 3060s share a 256K-context MoE model across a 50/50 tensor-split.

- **Qwen3.8-27B (3090):** [model card](../../models/qwen3.8-27b-rtx3090.md)
- **Qwen3.6-35B-A3B (dual 3060):** [model card](../../models/qwen3.6-35b-a3b.md)

## Methodology

- **Placement (fitter vs manual, split modes, `output.weight`, `--parallel`):** [multi-gpu-model-placement](../multi-gpu-model-placement.md)
- **KV-cache budgeting:** [kv-cache-sizing](../kv-cache-sizing.md)

## History

- **2026-03 era:** this machine ran an i5-7400 with 3× RTX 3060 (36 GB). The 3-GPU vLLM PP=3 and triple-GPU validation write-ups remain in [reports](../../reports/) — `2026-03-14-qwen3.5-35b-a3b-vllm-pp3-concurrency.md`, `2026-03-03-triple-gpu-validation.md`.
- **2026-07-17:** swapped to Ryzen 5 5600X + 1×3090 / 2×3060 (48 GB).
- **2026-08-21:** Qwen3.8-27B production moved to vLLM 0.27.1 in a dedicated LXC; stock llama.cpp (5f754ea, Q4_K_M + MTP) kept as dormant rollback.
- **2026-08-26:** RAM 4×8 GB (2133 MT/s) → 2× 32 GB DDR4-2933 SO-DIMM via SO-DIMM-to-DIMM adapters (64 GB), plus BIOS rework (downgrade to 1R0, CSM boot, WOL re-armed).
