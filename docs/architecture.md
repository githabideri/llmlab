# Architecture (sanitized)

**Compute layout**
- **Host:** Proxmox + ZFS
- **Containers:**
  - **llama.cpp** (inference)
  - **vLLM** (optional UI/serving)
  - **Ollama** (optional quick‑run service)

**Storage layout**
- **NVMe (OS/root):** ZFS datasets, container rootfs
- **SSD (models):** `/mnt/models`
  - `gguf/` (GGUF weights)
  - `cache/` (llama.cpp cache)
  - `hf/` (HF cache)

**GPU topology**
- Dual RTX 3060 12GB, row‑split across GPUs (`--split-mode row`, `--tensor-split A/B`)

**Key assumptions**
- 10 GbE or local access not required (single‑host)
- Batch sizes tuned for single‑user latency

> If you adapt this, replace paths with your local equivalents.
