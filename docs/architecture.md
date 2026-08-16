# Architecture

**Compute layout**
- **Host:** Proxmox + ZFS
- **Containers:**
  - **llama-cpp (primary inference host):** llama.cpp (production, Qwen3.8-27B + MTP), BeeLlama.cpp (historic/rollback)
  - **vLLM** (optional UI/serving)

**Storage layout**
- **NVMe (OS/root):** ZFS datasets, container rootfs
- **SSD (models):** `/mnt/models`
  - `gguf/` (GGUF weights)
  - `cache/` (llama.cpp cache)
  - `hf/` (HF cache)

**GPU topology**
- 1x RTX 3090 24GB (single-GPU: Qwen3.8-27B, llama.cpp + MTP)
- 2x RTX 3060 12GB (multi-GPU: layer-split via `--split-mode layer`, `--tensor-split A/B`)

**Key assumptions**
- Batch sizes tuned for single-user latency
- llama.cpp (stock 5f754ea) is the production path (Qwen3.8-27B + native MTP). BeeLlama (DFlash) is historic/rollback; llama.cpp was also the reference path.

> If you adapt this, replace paths with your local equivalents.
