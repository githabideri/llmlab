# Triple GPU Validation — Qwen3.5-35B Q4_K_M

**Date:** 2026-03-03  
**Model:** Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf  
**Hardware:** 3x RTX 3060 (36GB VRAM total)  
**Runtime config:** `--ctx-size 98304 --split-mode layer --ngl 99 --parallel 1`

---

## Context Speed Ladder

| Context | PP tok/s | TG tok/s | Notes |
|--------:|---------:|---------:|-------|
| 9 | 134.95 | 59.02 | Fresh KV cache |
| 3,210 | 1,242.41 | 59.27 | Optimal PP zone |
| 6,410 | 1,257.70 | 58.80 | Peak PP speed |
| 9,610 | 1,262.39 | 57.18 | Sustained high PP |
| 12,810 | 1,259.69 | 53.86 | TG starts degrading |
| 19,210 | 1,221.45 | 51.02 | -14% TG from peak |
| 24,010 | 1,200.81 | 48.88 | -17% TG from peak |

**Key findings:**
- **PP (prompt processing):** Peaks at ~1,260 tok/s (6K-12K context), stays above 1,200 tok/s even at 24K
- **TG (text generation):** Stable ~59 tok/s at low-mid context, degrades gracefully to ~49 tok/s at 24K (-17%)
- **3-GPU distribution working perfectly:** All cards active during inference (verified via nvtop)

---

## OpenClaw Parcours Results

| Test | Status | Details |
|------|--------|---------|
| L0 | ✅ PASS | Read/write sanity (ping.json) |
| L1 | ✅ PASS | Provider extraction (llama-cpp/Qwen3.5/98304) |
| L2-L4 | 🔄 Pending | (Functional tests) |

---

## Hardware Verification

**GPU topology:**
```
GPU 0: 7.0 GB / 12 GB (PCIe Gen3 x16)  — CPU slot
GPU 1: 6.6 GB / 12 GB (PCIe Gen3 x4)   — Chipset slot 1
GPU 2: 6.7 GB / 12 GB (PCIe Gen3 x4)   — Chipset slot 2
```

**Live utilization during test:**
- GPU0: 31% util, 58.71 W
- GPU1: 28% util, 52.99 W  
- GPU2: 36% util, 65.71 W

**Total model footprint:** ~20.3 GB across 3 cards ✅

---

## Comparison: 2-GPU vs 3-GPU

**VRAM distribution:**
- **2-GPU:** ~13.6 GB (11.9 + 11.3 per card near ceiling)
- **3-GPU:** ~20.3 GB (7.0 + 6.6 + 6.7, comfortable headroom)

**Speed:** (To be measured with identical config on 2 GPUs for direct comparison)

**Capacity unlocked:** 
- Qwen3.5 Q5_K_M or Q6_K now possible (wouldn't fit on 2 GPUs)
- Higher context ceiling feasible (tested to 24K, could push to 96K runtime limit)

---

## Next Steps

1. ✅ Verify 3-GPU passthrough working
2. ✅ Run context speed ladder (0-24K)
3. 🔄 Complete OpenClaw Parcours (L2-L4 + DOC-QA)
4. 🔄 Test higher quant (Q5_K_M or Q6_K)
5. 🔄 Push context to limits (~96K or beyond)
6. 🔄 Generate comparison report with 2-GPU baseline

**Status:** Phase A (baseline validation) ~70% complete
