# Hardware Profile: GPU Server

**Configuration:** 1× RTX 3090 24 GB + 2× RTX 3060 12 GB (48 GB VRAM)  
**Use Case:** Primary multi-GPU LLM inference (llama.cpp, vLLM)  
**Status:** Active — Qwen3.8-27B on the 3090, Qwen3.6-35B-A3B on the dual 3060

---

## Specifications

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 5 5600X (6C/12T) |
| Motherboard | MSI MEG X570 ACE |
| RAM | 32 GB (4×8 GB DDR4-2133) |
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
| RTX 3090 | Qwen3.8-27B-Q4_K_M + MTP, single-GPU | 8080 |
| 2× RTX 3060 | Qwen3.6-35B-A3B (UD-IQ4_XS), tensor-split 50/50 + vision | 8081 |

The 3090 is a standalone single-GPU endpoint; the two 3060s share a 256K-context MoE model across a 50/50 tensor-split.

- **Qwen3.8-27B (3090):** [model card](../../models/qwen3.8-27b-rtx3090.md)
- **Qwen3.6-35B-A3B (dual 3060):** [model card](../../models/qwen3.5-35b-a3b.md) *(card still titled 3.5 — see open item)*

## Methodology

- **Tensor-split / `output.weight` / `--parallel`:** [multi-gpu-tensor-split](../multi-gpu-tensor-split.md)
- **KV-cache budgeting:** [kv-cache-sizing](../kv-cache-sizing.md)

## History

- **2026-03 era:** this machine ran an i5-7400 with 3× RTX 3060 (36 GB). The 3-GPU vLLM PP=3 and triple-GPU validation write-ups remain in [reports](../../reports/) — `2026-03-14-qwen3.5-35b-a3b-vllm-pp3-concurrency.md`, `2026-03-03-triple-gpu-validation.md`.
- **2026-07-17:** swapped to Ryzen 5 5600X + 1×3090 / 2×3060 (48 GB).
