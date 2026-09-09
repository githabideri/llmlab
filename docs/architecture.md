# Architecture

**Compute layout**
- **Host:** Proxmox + ZFS
- **Containers:**
  - **vLLM (Qwen3.8-27B production):** vLLM 0.28.0, W4A16-AutoRound, **tensor-parallel 2 over both RTX 3090s** (since 2026-09-08), MTP k=3, fp8 KV, 262,144 ctx, vision — dedicated LXC, one engine for both cards (250 W per card)
  - **llama-cpp (35B + fallbacks):** llama.cpp mainline 4f31eed (the primary box's dual-3060 unit was dismantled 2026-09-08 for the second 3090; 35B-A3B is now interim on the secondary box's single 3060, plus the backup box in its window), BeeLlama.cpp (historic/rollback)

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

> If you adapt this, replace paths with your local equivalents.
