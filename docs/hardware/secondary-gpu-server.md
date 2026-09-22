# Hardware Profile: Secondary GPU Server

**Configuration:** 1× RTX 3060 12 GB (x16 slot, host-side) + 2× 2 GB Pascal (GTX 1050, GT 1030) VFIO-passthrough to a 4-vCPU VM  
**Use Case:** Auxiliary GPU host — host-level llama.cpp 35B endpoint on the 3060; WhisperX ASR on the Pascal VM  
**Status:** Active

---

## Specifications

| Component | Spec |
|-----------|------|
| CPU | Intel i5-7400 (4C/4T @ 3.0 GHz, no HT) |
| RAM | 32 GB (4× 8 GB) |
| Board | Gigabyte B250-HD3P (ATX, Intel B250) |
| System disk | 466 GB Samsung 860 EVO 500 GB NVMe (Proxmox 9.2; ~49 G root LV + models LV) |
| GPU 1 | RTX 3060 12 GB **LHR** — x16 slot, host-side (driver 595.x); power limit at the card default (170 W, no custom limit); idle draw ~15 W; **PCIe link observed running at 2.5 GT/s (Gen1) against an 8 GT/s (Gen2) slot cap** |
| GPU 2/3 | GTX 1050 2 GB + GT 1030 2 GB (Pascal, CC 6.1) — VFIO-passthrough (pcie=1) to the 4-vCPU / 8 GB VM; `power.draw` not reported by nvidia-smi on these cards (null-tolerant handling, see the WhisperX report) |
| iGPU | Intel UHD 630 (unused by GPU workloads) |

## What's deployed

- **3060 (host-side):** the llama.cpp `Qwen3.6-35B-A3B-MTP` endpoint — the 35B's interim home since the primary box's dual-3060 setup was dismantled 2026-09-08. Serving facts (quant, ctx, flags, measured performance) live on the [35B model card](../../models/qwen3.6-35b-a3b.md); the process runs host-level (no container), loopback-only.
- **Pascal VM (4 vCPU, 8 GB):** WhisperX `faster-whisper-large-v3-turbo` ASR at ~10.7× realtime using both 2 GB cards, with X-vector diarization — benchmarked in the [2026-09-06 report](../../reports/2026-09-06-whisperx-pascal-dual-gpu-benchmark.md).

## Why it exists

This is the original main GPU server, demoted when the Ryzen 5 5600X box took over that role and relocated to the second site in 2026-08. It lives on as the fleet's auxiliary GPU machine: a second 35B endpoint (redundancy with the backup box) and the WhisperX host.

## Related

- **35B serving config:** [models/qwen3.6-35b-a3b.md](../../models/qwen3.6-35b-a3b.md)
- **Pascal VM benchmark:** [reports/2026-09-06-whisperx-pascal-dual-gpu-benchmark.md](../../reports/2026-09-06-whisperx-pascal-dual-gpu-benchmark.md)
- **Fleet index:** [README.md](README.md)
