# Hardware Profile: Pascal Laptop

**Configuration:** 2017 HP OMEN 15 (ce-series) repurposed as a headless single-GPU inference node — GTX 1060 6 GB Max-Q (Pascal) + 4-core Kaby Lake CPU  
**Use Case:** Qwen3.6-35B-A3B 128K endpoint (MoE hybrid CPU/GPU) + the lab's Pascal test bed (driver/CPU-kernel/PCIe work that the 3060/3090 boxes can't do)  
**Status:** Active (in production since 2026-09-27)

---

## Specifications

| Component | Spec |
|-----------|------|
| CPU | Intel i5-7300HQ (Kaby Lake, 45 W) — **4 cores, 4 threads, no Hyper-Threading** (Intel ARK: Total Cores 4 / Total Threads 4 / HT No; the HT-equipped sibling of the 7th-gen mobile family is the i7-7700HQ). CPUID identity: **Family 6 / Model 0x9E / Stepping 9, microcode 0x84**, brand string "i5-7300HQ @ 2.50 GHz" (base 2.50 / max 3.50 GHz, 6 MB L3). Note: the CPUID `ht` capability bit is *set* on this SKU (die-level capability; also set on other known-4T 7300HQ units) — it does **not** indicate 8 threads; the SMT topology leaves report 1 logical processor per core and `smt/control` is `notsupported`. So `-t 4` is one worker per core and `-t 8` is 2× oversubscription |
| RAM | 24 GB DDR4-2400, **asymmetric 16 + 8 flex** (one SK Hynix 16 GB, one Kingston 8 GB) — see bandwidth note below |
| GPU | NVIDIA GeForce **GTX 1060 6 GB with Max-Q Design** (Pascal GP106, compute 6.1): **10 SMs = 1,280 CUDA cores**, 192-bit bus @ 4004 MHz (192 GB/s), 5.92 GB usable, 1.5 MB L2, 60 W TGP |
| PCIe | laptop x8 slot, Gen3 capable; **negotiates Gen3 x8 under sustained load** (drops to Gen2/Gen1 when idle — normal power management, not a finding; see the backup-box note). Measured host↔device transfer: **~6.3 GB/s per direction** (pinned ≈ pageable, single-process sustained) |
| Storage | 128 GB NVMe (single partition, models + system) |
| Power | AC-only: the battery was removed and the CR2032 CMOS cell is dead, so the machine must stay on AC (BIOS resets on total power loss) |
| Network | headless; wired ethernet on the home network |
| OS | NixOS (flakes-based), kernel 6.18 |
| Driver / toolkit | NVIDIA **580.178.04** (the 580 LTSB line — the last Pascal-compatible driver series, supported through mid-2028) + **CUDA 12.9** — CUDA 12.x is the final toolkit family capable of targeting Pascal/sm_61; 13.x dropped it |
| Engine | llama.cpp 2026 (September base) — nixpkgs-24.05 gcc 13.2 toolchain, `-march=native` CPU (AVX2/FMA/F16C, no AVX-512/VNNI), `sm_61` CUDA |

## Notes

- **The CPU has no Hyper-Threading to hide behind.** The i5-7300HQ is a 4C/4T part (the 7th-gen *i5* mobile SKUs ship without HT; the i7-7700HQ is the HT variant). All thread-count results in this lab's MoE work that used this box are physical-core counts; the measured `-t 4` > `-t 8` gap is plain oversubscription.
- **The 8+16 RAM runs at effective single-channel bandwidth.** STREAM (1/2/3/4 threads): ~17–20 GB/s read+write with **no thread scaling** — one DDR4-2400 channel's theoretical is 19.2 GB/s, true dual-channel would be ~35–38 GB/s and would scale with threads. The mechanism (whether the BIOS interleaves the asymmetric DIMM set) is not verifiable from the OS on this platform — no edac, no rank data in DMI — so the measurement is the fact and the BIOS behaviour remains a hypothesis. Consequence: the CPU-resident half of MoE decode sits at a ~19–20 GB/s DRAM wall, which the 35B decode numbers confirm arithmetically. A symmetric set (2×8 for the 12 GB 2-bit model, or 2×16 for the 17 GB IQ4_XS) is the single highest-leverage hardware upgrade on this machine (as of 2026-09-28 the 2×8 purchase is pending; with the 2-bit production model's ~10 GB steady host footprint, 16 GB fits every context — the 4-bit's ~15.6 GB would not.)
- **Thermals are not the constraint.** During a 5-minute 3000-token decode: CPU held 3.07–3.11 GHz across all cores (no throttle; 50–86 °C vs 100 °C critical), GPU 99% util at 38–50 W against the 60 W TGP (not power-throttled; ≤74 °C). The CPU and GPU share a cooling system, so raising GPU power would trade against CPU headroom — but there is no thermal headroom problem at the current limits.
- **The Max-Q 1060 is a full GP106** (10 SMs / 1,280 cores) — the Max-Q suffix is only the power/clock profile. Don't confuse it with the 3 GB GTX 1060 (GP107, 640 cores) or with older notes in the fleet that misread this card's size.
- The `cublasCreate_v2 = 3` crash seen with manual `-ngl` + mmap on this driver/build is config-specific (resolved by the load-mode/ncmoe combination used in production — see the 2026-09-27 report); the 580 driver itself is otherwise stable on the Pascal card.

## Related

- **Served model:** [Qwen3.6-35B-A3B card](../../models/qwen3.6-35b-a3b.md) (Pascal laptop variant)
- **Build & 128K placement:** [2026-09-27 Pascal 1060 report](../../reports/2026-09-27-pascal-1060-2026-llama-cpp-35b-128k.md)
- **Q2_0 CPU-kernel check on this CPU:** [2026-09-27 ISTA 2-bit Q2_0 report](../../reports/2026-09-27-ista-gsq-2bit-35b-q2-0-kaby-lake.md)
- The other Pascal card in the fleet (1050 + 1030, ASR only): [secondary GPU server profile](secondary-gpu-server.md), [WhisperX report](../../reports/2026-09-06-whisperx-pascal-dual-gpu-benchmark.md)
