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

> **PCIe note:** the idle GPU reports a Gen1 link — normal NVIDIA power management; it retrains to full speed under load (the 2026-06-30 report walked through exactly this). Not a finding.

## Power management (measured 2026-10-26)

Loaded idle (Strata resident, CUDA context held): **P8, 210/405 MHz, ~5.9 W** on the card, PCIe Gen1 link — the driver's floor while a context exists, stable for 15+ min; keeping the 11.4 GB of VRAM shows **no detectable additional GPU draw** (5.82 W with no context vs 5.93 W loaded, inside the card's ±0.2 W wobble). Runtime D3 did not occur under the tested conditions (live context, persistence mode on, no wake path on the device); which layer blocks it is an inference, not a measurement. No tested driver-side lever improves on this: `nvidia-pstated` measured a no-op (+0.2 W), clock locks *raise* idle (they pin the P-state above P8), and the 170 W power cap is never near (86 W peak under load). Memory-clock tiers the driver exposes: 405 / 810 / 5001 / 7301 / 7501 MHz. **Wall (plug meter via HA, same window): 63–65 W across all idle states, 124–184 W in generation bursts** — this is a dual-purpose host (four disks, backup jobs, Proxmox), and they, not the GPU, are most of that draw. Details, the full state matrix and the corrections to the first version: [2026-10-26 loaded-idle power report](../../reports/2026-10-26-rtx3060-strata-loaded-idle-power.md).

## Why it exists

The x16 slot was vacant and a single inference machine is a single point of failure, so the box doubles as a secondary llama.cpp MTP endpoint on the 3060.

## Related

- **Single-3060 MTP config:** [2026-06-30 single RTX 3060 report](../../reports/2026-06-30-qwen3.6-35b-a3b-mtp-single-3060.md)
- **Loaded-idle power floor:** [2026-10-26 report](../../reports/2026-10-26-rtx3060-strata-loaded-idle-power.md)
