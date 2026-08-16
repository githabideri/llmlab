# Hardware

Inference machines in the fleet, by role. Each profile covers the physical box — CPU, board, GPUs, RAM, disks, PCIe topology — and what's deployed on it. Serving configs live in the [model cards](../../models/); general GPU methodology lives in [multi-GPU tensor-split](../multi-gpu-tensor-split.md) and [KV-cache sizing](../kv-cache-sizing.md).

| Machine | CPU | GPUs (VRAM) | RAM | Role | Status |
|---------|-----|-------------|-----|------|--------|
| [GPU server](gpu-server.md) | AMD Ryzen 5 5600X (6C/12T) | 1× RTX 3090 + 2× RTX 3060 (48 GB) | 32 GB | Primary multi-GPU inference | Active |
| [Backup / inference box](backup-single-3060.md) | Intel i3-9100 (4C) | 1× RTX 3060 (12 GB) | 32 GB | Data backup + llama.cpp MTP endpoint | Active (endpoint idle) |
| [Laptop](laptop.md) | AMD Ryzen 7 7840U (8C/16T) | Radeon 780M iGPU | 96 GB | Vulkan / iGPU experiments | Active |
| Secondary GPU server | Intel i5-7400 | RTX 3060 + GTX 1050 + GTX 1030 | — | (planned, not yet operational) | Planned |
