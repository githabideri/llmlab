# Hardware Profile: Laptop

**Configuration:** AMD Ryzen 7 7840U (8C/16T), Radeon 780M iGPU  
**Use Case:** Vulkan / iGPU inference experiments (portable)  
**Status:** Active

---

## Specifications

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 7840U (8C/16T, Zen 4) |
| GPU | Radeon 780M (integrated) |
| RAM | 96 GB DDR5 |

## Notes

Runs llama.cpp via the Vulkan backend on the iGPU. MTP gives large speedups where available; q8_0 KV is the practical choice on iGPU (q4_0 carries dequantization overhead).

## Related

- [2026-07-24 Qwen3.6-35B at 28 tok/s (MTP + Vulkan iGPU)](../../reports/2026-07-24-35b-mtp-laptop-setup.md)
- [2026-08-12 DS4 ROCm gfx1103 build](../../reports/2026-08-12-ds4-rocm-gfx1103-build-success.md)
