# Hardware Profile: Backup / Inference Box

**Configuration:** 1× RTX 3060 12 GB in the x16 slot  
**Use Case:** Dual-use — data backup target + llama.cpp MTP inference endpoint  
**Status:** Active (data backup always on; inference endpoint idle as of last check)

---

## Specifications

| Component | Spec |
|-----------|------|
| CPU | Intel i3-9100 (4C) |
| RAM | 32 GB |
| System disk | 256 GB NVMe (ZFS) |
| Storage | 4× 16 TB HDD (backup target) |
| GPU | RTX 3060 12 GB — CPU PCIe x16 |

## Why it exists

The x16 slot was vacant and a single inference machine is a single point of failure, so the backup box doubles as a secondary llama.cpp MTP endpoint on the 3060. Many people run a single 3060, so the results here are broadly relevant.

## Related

- **Single-3060 MTP config:** [2026-06-30 single RTX 3060 report](../../reports/2026-06-30-qwen3.6-35b-a3b-mtp-single-3060.md)
