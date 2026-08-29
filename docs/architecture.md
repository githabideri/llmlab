# Architecture

**Compute layout**
- **Host:** Proxmox + ZFS
- **Containers:**
  - **vLLM (Qwen3.8-27B production):** vLLM 0.27.1, W4A16-AutoRound, MTP k=3, fp8 KV — dedicated LXC on the 3090
  - **llama-cpp (35B + fallbacks):** llama.cpp mainline 4f31eed (production: Qwen3.6-35B-A3B on the dual 3060; stock 5f754ea Qwen3.8-27B kept as dormant rollback), BeeLlama.cpp (historic/rollback)

**Storage layout**
- **NVMe (OS/root):** ZFS datasets, container rootfs
- **SSD (models):** `/mnt/models`
  - `gguf/` (GGUF weights)
  - `cache/` (llama.cpp cache)
  - `hf/` (HF cache)

**GPU topology**
- 1x RTX 3090 24GB (single-GPU: Qwen3.8-27B, vLLM + MTP, dedicated LXC)
- 2x RTX 3060 12GB (multi-GPU: tensor-split `--split-mode tensor --tensor-split 50,50`, Qwen3.6-35B-A3B + MTP n=3)

**Key assumptions**
- Batch sizes tuned for single-user latency (ubatch raised to the VRAM limit where possible — see [2026-08-28 ubatch report](../reports/2026-08-28-llama-cpp-ubatch-moe-single-gpu.md))
- Qwen3.8-27B: vLLM 0.27.1 is the production path (since 2026-08-21); stock llama.cpp 5f754ea (Q4_K_M + MTP) is the documented rollback. BeeLlama (DFlash) is historic only.

> If you adapt this, replace paths with your local equivalents.
