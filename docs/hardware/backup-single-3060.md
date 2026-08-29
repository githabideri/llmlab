# Hardware Profile: Backup / Inference Box

**Configuration:** 1× RTX 3060 12 GB in the x16 slot  
**Use Case:** Dual-use — Proxmox backup host (power-saved nightly backup source) + llama.cpp MTP inference endpoint  
**Status:** Active (endpoint in production use as fallback inference)

---

## Specifications

| Component | Spec |
|-----------|------|
| CPU | Intel i3-9100 (4C) |
| RAM | 48 GB |
| System disk | 238 GB NVMe (ZFS) |
| Backup | nightly source for a Proxmox Backup Server VM (PBS datastore lives on a separate host); the former 4× 16 TB local HDDs are removed |
| GPU | RTX 3060 12 GB — CPU PCIe x16 |

## Why it exists

The x16 slot was vacant and a single inference machine is a single point of failure, so the box doubles as a secondary llama.cpp MTP endpoint on the 3060.

## Related

- **Single-3060 MTP config:** [2026-06-30 single RTX 3060 report](../../reports/2026-06-30-qwen3.6-35b-a3b-mtp-single-3060.md)
