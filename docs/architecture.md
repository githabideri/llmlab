# Architecture (sanitized)

**Compute layout**
- **Host:** Proxmox + ZFS
- **Containers:**
  - **CT 327 (llama-cpp):** Primary inference — BeeLlama.cpp (production), llama.cpp (reference/fallback)
  - **vLLM** (optional UI/serving)
  - **Ollama** (optional quick‑run service)

**Storage layout**
- **NVMe (OS/root):** ZFS datasets, container rootfs
- **SSD (models):** `/mnt/models`
  - `gguf/` (GGUF weights)
  - `cache/` (llama.cpp cache)
  - `hf/` (HF cache)

**GPU topology**
- 1x RTX 3090 24GB (single-GPU: Qwen3.6-27B, BeeLlama + DFlash)
- 2x RTX 3060 12GB (multi-GPU: layer-split via `--split-mode layer`, `--tensor-split A/B`)

**Key assumptions**
- Batch sizes tuned for single-user latency
- BeeLlama is production path for structured output (code, JSON, tool calls). llama.cpp remains reference path and fallback.

> If you adapt this, replace paths with your local equivalents.
