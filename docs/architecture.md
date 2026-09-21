# Architecture

**Compute layout**
- **Host:** Proxmox + ZFS
- **Containers:**
  - **vLLM (Qwen3.8-27B production):** vLLM 0.28.0, W4A16-AutoRound, **tensor-parallel 2 over both RTX 3090s** (since 2026-09-08), MTP k=3, fp8 KV, 262,144 ctx, vision — dedicated LXC, one engine for both cards (220 W per card, interim — see [hardware/gpu-server.md](hardware/gpu-server.md))
  - **llama-cpp (dormant rollback only):** the box's remaining llama.cpp unit is the single-3090 Qwen3.8-27B Q4_K_M + MTP rollback (disabled, kept on disk). The 35B/27B/125B llama.cpp serving moved off this box: 35B interim on the secondary box's single 3060, plus the backup box (both behind their model-mux); BeeLlama.cpp is historic/rollback only.

**Storage layout**
- **NVMe (OS/root):** ZFS datasets, container rootfs
- **SSD (models):** `/mnt/models`
  - `gguf/` (GGUF weights)
  - `cache/` (llama.cpp cache)
  - `hf/` (HF cache)

**GPU topology**
- 2x RTX 3090 24GB — both CPU-direct PCIe 4.0 x8 (no NVLink on this board); vLLM TP2, one dedicated LXC sees both cards

**Key assumptions**
- Batch sizes tuned for single-user latency (ubatch raised to the VRAM limit where possible — see [2026-08-28 ubatch report](../reports/2026-08-28-llama-cpp-ubatch-moe-single-gpu.md))
- Qwen3.8-27B: vLLM 0.28.0 TP2 is the production path (since 2026-09-08; 0.27.1 single-3090 until then); the single-3090 stock llama.cpp 5f754ea (Q4_K_M + MTP) unit is kept disabled as the documented rollback. BeeLlama (DFlash) is historic only.
- **Serving config and performance numbers live on the model cards** ([models/](../models/README.md)); per-box hardware facts (GPU layout, power limits, reliability) live in [hardware/](hardware/README.md). This file is layout only — if a number is stated here, it is one that belongs to the shape of the machine, not to a service.

> If you adapt this, replace paths with your local equivalents.
