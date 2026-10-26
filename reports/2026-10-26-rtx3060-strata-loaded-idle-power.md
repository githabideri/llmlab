# 2026-10-26 Loaded-idle power on the single-3060 box: a fully loaded but inactive Strata engine sits at P8, ~5.9 W, and 11.4 GB of VRAM costs less than a tenth of a watt

**Date:** 2026-10-26 (one ~2.5 h window; the box was booted for this and left running)
**Category:** Experiment / benchmark (measurement campaign; **no permanent changes** — every control was reverted and the production endpoint re-verified live)
**Hardware:** backup / inference box: Intel i3-9100 (4C/4T, DDR4-2400 2-channel), 48 GB RAM, single **RTX 3060 12 GB** in the CPU-direct PCIe 3.0 x16 slot; Proxmox host with an Ubuntu 24.04 inference container running the box's production Strata endpoint (v0.1.41, Qwen3.8-Flash-Next IQ2_XS, 98K context, 19 GiB resident expert budget, MTP spec 4, KV int8, mmap — the [2026-10-08 winner](2026-10-08-strata-v0141-single-3060.md), unchanged). Driver **595.71.05** (distribution package, **proprietary** kernel module, GSP firmware active, persistence mode on).
**Question:** can a fully loaded but *inactive* Strata engine keep its CUDA context and VRAM while drawing less power? What is the floor, and which of the usual levers (P-state daemon, clock locks, power limit, PCIe ASPM, runtime D3) actually move it on this platform?

**TL;DR**

- The loaded-idle floor is **P8 at 5.8–6.2 W on the card, stable for 15+ min** (no drift with time since last generation). Holding **11.47 GB of VRAM costs <0.15 W** against the driver-only state (5.82 W at 4 MiB) — at P8 the VRAM is essentially free; the floor is the board's P8 floor, not the memory.
- **No driver-side lever improves on it.** `nvidia-pstated` (source-reviewed, built, 10-min run): **+0.2 W** (its forced P8 pins the graphics clock at 420 MHz instead of 210), no errors, clean restore. Clock locking is strictly worse at idle: `-lmc 5001` pins the card in **P3 (19.8 W idle, decode −10–15%)**; `-lgc 1350` pins **P5 (+5 W idle)** while saving 21 W at peak with zero t/s cost — the only active-power lever found, net-negative for an idle-dominant duty cycle. The 170 W power cap is meaningless (86 W peak).
- **Runtime D3 is unreachable while keeping the context** on this desktop platform (the PCIe device exposes no wake path; the driver parks it `active` forever, suspended-time 0). The only way below ~6 W with the model usable is releasing the CUDA context itself (engine-level unload-on-idle) — out of scope, recorded as a follow-up.
- PCIe idles at **Gen1 x16** (the documented idle-link state) and retrains to **Gen3 in under 2 s** under load. ASPM L0s/L1 is advertised but disabled at the link; the host kernel runs `pcie_aspm=default`. The one untested lever is a `pcie_aspm` cmdline change + reboot — not done here.
- **For the always-on node question this informs (BD775i-class mini-PC):** budget **~6 W GPU + ~16–21 W CPU package (RAPL) + 10–20 W board/NIC/disk (unmeasured) ≈ 35–45 W wall** for the always-loaded state. The GPU is a small minority of that; the platform's own idle draw dominates, and the loaded-vs-unloaded GPU difference at the wall is roughly 5–10 W.

**Measurement gap (stated, not hidden):** the box has **no AC telemetry** — its power path is a smart plug with no wattage channel, and the host's node_exporter RAPL collector is broken on this kernel (7.0.14), so Prometheus has no CPU-power series for it. Everything below is per-component: GPU watts from NVML, CPU package watts from RAPL `energy_uj` deltas, sampled every 2 s with pstate/graphics+memory clocks/PCIe-gen/VRAM/util on the same cadence. Wall power = GPU + CPU package + the unmeasured board, and every claim in this report is phrased per-component. The 10-08 campaign had no watt capture at all; this is strictly more, and the follow-up list ends with "add a wall meter."

## States, commands, results

Fixed across the whole window: model/quant (same shards as 10-08), engine binary and version, driver, expert budget, MTP settings, context geometry, prompt and generation lengths. The benchmark is a fixed ~150-token prompt plus a unique nonce per rep (greedy), 3 reps of prefill-only (`max_tokens 1`) and 3 of 256-token decode, against the production endpoint. The A0/A1 stop+restore of the serve unit was registered per the box's prod-service-change discipline; the unit file was never touched and the endpoint was re-verified after each restore (model re-loads in ~60 s, mmap).

| State | What | GPU avg (W) | GPU max | P-state | gr/mem MHz | PCIe | VRAM | CPU pkg avg (W)¹ | Duration |
|---|---|---|---|---|---|---|---|---|---|
| **A0** | driver loaded, no context (serve stopped) | 5.82 | 5.96 | P8 | 210/405 | Gen1 | 4 MiB | 40.1² | 90 s |
| **A1** | Strata loaded, idle | **5.93** | 6.07 | P8 | 210/405 | Gen1 | 11473 MiB | 21.3 | 5 min |
| **A2** | active generation (3 prefill + 3 decode) | 33.8 (window) | **86.3** | P2 when busy | 1777–1890 / 7301 | Gen3 when busy | 11473 | 33.1 | 2 min |
| **A3** | post-generation, resident | 5.93 | 6.20 | P8 | 210/405 | Gen1 | 11605 | 15.9 | 15.5 min |
| **B1** | A1 + `nvidia-pstated` (forced-P8 daemon) | 6.13 | 6.48 | P8 (forced) | **420**/405 | Gen1 | 11605 | 18.8 | 10 min |
| **D1** | A2 with `-lmc 5001` (memory clock locked) | 40.2 (window) | 80.9 | P3 when busy | 1882 / 5001 | Gen3 | — | 30.5 | 2.5 min |
| **D2** | A2 with `-lgc 1350` (graphics clock locked) | 22.3 (window) | **65.3** | P2 when busy | 1350 / 7301 | Gen3 | — | 25.4 | 2.5 min |
| B0′ | 5-min clean check after B1 (caught one real production request, peak 75 W) | 14.2 (window) | 75.2 | P8↔P2 | — | Gen1↔Gen3 | — | 25.1 | 5 min |

¹ RAPL package power on the i3-9100, 2 s deltas. ² Elevated by unrelated host workload during that 90 s (the box also runs the backup source); the GPU column is what the campaign is about.

**Throughput, same benchmark** (the cold first rep is included, it happened):

| Config | Prefill (s) | Decode 256 tok (t/s) | GPU peak |
|---|---|---|---|
| A2, no lock | 4.43 cold / 1.85 / 1.89 warm | 18.3 / 26.9 / 25.2 | 86.3 W |
| D1, mem @5001 | 2.01 / 2.01 / 2.01 | 22.8 / 23.8 / 25.4 | 80.9 W |
| D2, grc @1350 | 1.91 / 1.89 / 1.88 | 23.8 / 27.1 / 26.4 | 65.3 W |

Decode is host-side on this box (the 10-08 attribution: the 4 cores do the expert compute, the 3060 does <1.6% of per-token time), which is why D2's clock ceiling costs *nothing* in t/s — the SMs were never the bottleneck and the natural 1777–1890 MHz sits below the card's 2100 maximum anyway. D1 costs 10–15% because decode *is* memory-bandwidth-bound and 5001 MHz is below the 7301 the driver picks under load.

**Integrated energy** (trapezoid over the 2 s samples): idle states (A1/A3) burn **~21–22 W combined (GPU + CPU package) ≈ 0.22 Wh/h**; the A2 window averages 60 W over its 2 min (mostly idle gaps between reps) and delivers ~1200 tokens ≈ **~6 J/token window-average**, i.e. the active segments run at ~50–60 W GPU + ~30 W CPU.

## Observations (the mechanism, evidence vs hypothesis)

1. **P8 is the floor with a context, and the driver gets there by itself.** In the A2 window 28 of 59 samples (all the inter-request gaps) are P8 at 210/405 MHz — the card downclocks within ~2 s of the last CUDA work. Any daemon's only useful action is to force exactly the state the driver already chose; B1 demonstrates the residual (the forced P8 holds the SM clock at 420 instead of letting it settle to 210, +0.2 W). Verdict: the daemon exists for cards that stick in high P-states after load; this one does not.
2. **VRAM residency is free at P8.** A0 (4 MiB) vs A1 (11473 MiB): 5.82 vs 5.93 W, same clocks. The ~0.1 W difference is within the sampler's own noise (the card wobbles 5.7–6.2 W anyway).
3. **No time drift.** 15.5 min post-generation (A3) is statistically identical to fresh-load idle (A1). Whatever "the engine has been running a while" effect exists, it is not in the power trace.
4. **Runtime D3: not on this platform.** The PCIe device shows `power/control=auto` and `runtime_enabled`, but `wakeup=disabled` and `runtime_status=active` with `runtime_suspended_time=0` — a desktop board without PCIe PME cannot suspend the device, and with persistence mode on there is no window where the driver would try. The context is the reason the card exists electrically; dropping to D3 means dropping the context.
5. **PCIe link behaviour matches the 2026-06-30 finding and is now instrumented:** Gen1 x16 at idle, Gen3 x16 under load, retrain in <2 s. LnkCap advertises ASPM L0s+L1; LnkCtl has it disabled (VBIOS/driver side); the host kernel parameter is `default`. Untested lever: `pcie_aspm=force` (cmdline + reboot).
6. **`nvidia-pstated` implementation notes** (43 commits, last 2025-11-02, maintained; built from source for this test): it drives state with NVML (temperature/utilization) plus a dlsym'd `NvAPI_GPU_SetForcePstate` — the call **is** supported on this GeForce (returns NVAPI_OK, unlike what one would assume for a desktop card). Its exit path hardcodes `enter_pstate(16)` as "restore automatic management": on a datacenter GPU P16 is a real high state; on this GeForce P16 does not exist and the same NVAPI_OK return is how "back to auto" works. No persistent configuration is written by the daemon; nothing to clean up but the process.
7. **Clock locking moves the P-state, not just the clocks** — the reason both D-arms raise idle: locking memory at 5001 forbids the P8 pair (mem 405), so the selector parks at P3 (19.8 W); locking graphics at 1350 forbids the 210 MHz idle pair, so it parks at P5 (10.7 W). There is no "active-only" clock mode in NVML.
8. **The supported memory-clock ladder on this card:** 405 / 810 / 5001 / 7301 / 7501 MHz, each with its own graphics range (the `-q` Supported Clocks table; `-lmc` accepts any tier, the earlier assumption that only 7501 was settable was wrong).

## Conclusion

1. **The campaign's question, answered:** yes — a loaded, inactive Strata engine keeps its context and VRAM at the card's lowest power state. On this box that is **5.9 W on the GPU** (P8, 210/405 MHz, Gen1 link), and the VRAM in it costs less than 0.15 W. The box-level idle (GPU + CPU package) is ~21–22 W before the unmeasured board.
2. **Nothing driver-side gets you lower while keeping the context.** The daemon no-ops (measured, not inferred); clock locks raise idle by 5–14 W to buy active power you mostly don't spend (86 W peak against a 170 W cap); the only config worth keeping in the back pocket is D2 (`-lgc 1350`) *if* the duty cycle ever flips to generation-heavy, and even then it costs +5 W per idle minute. Recommended: **deploy nothing** — the stock driver behaviour is already the optimum for this duty cycle.
3. **The only way below the P8 floor with a usable model** is (a) engine-level unload-on-idle (release the CUDA context; the mmap design makes reload ~60 s, measured in this window, and the model files are already page-cached) or (b) a platform whose PCIe path actually suspends the device (D3 with PME wake — a server-class board, and even then the context must be gone). Both are follow-ups, not findings.
4. **For the BD775i always-on question:** the GPU contributes roughly a fifth of the loaded-idle wall draw; the CPU package and the board are the bigger terms. A low-idle mini-PC choice beats any GPU tuning in this class.

## Follow-ups (out of scope here)

- `pcie_aspm` cmdline experiment (needs a reboot window on the box; the link-control evidence above is the pre-test).
- Engine-level idle-unload mode in Strata (release context after N minutes, re-init on demand; measure the 60 s re-init against the ~0.2 Wh/h it saves… which is the actual arithmetic: it saves **less** than one hour of the D2-style gap per unload cycle — probably a responsiveness feature, not an energy one).
- Fix the host's power telemetry: a RAPL collector that works on kernel 7.0 (or a 10-line exporter for `energy_uj`) so Prometheus gets the CPU series, and a wall meter with a real wattage channel for absolute calibration.

## Raw data and commands

The 2 s CSVs (8 files, columns `ts, gpu_w, pstate, gr_mhz, mem_mhz, util_pct, vram_mib, pcie_gen, cpu_busy_pct, cpu_pkg_w`) live in the private repo under `temp/power-campaign-2026-10-26/` alongside the sampler script; they are ephemeral and regenerate with the commands below.

```bash
# sampler (PVE host): nvidia-smi query + RAPL delta every 2 s
nvidia-smi --query-gpu=power.draw_w,pstate,clocks.gr,clocks.mem,utilization.gpu,memory.used,pcie.link.gen.current --format=csv,noheader
# CPU package watts: Δ(/sys/class/powercap/intel-rapl:0/energy_uj)/Δt  ·  busy%: /proc/stat

# nvidia-pstated (built from source, main branch, 2025-11):
#   repo src/{main.c,nvapi.c,nvapi.h,nvml.h,utils.c} + NVIDIA R555-OpenSource SDK headers
#   + system nvml.h (header-only dev package) → gcc -O2 ... -lnvidia-ml -ldl
#   run: pstated -i 0 -psh 0 -psl 8 -ut 0 -ibs 30 -tt 80 -si 100

# clock locks (all reverted after use: -rmc / -rgc):
nvidia-smi -lmc 5001    # D1: memory tier lock (5001 of 405/810/5001/7301/7501)
nvidia-smi -lgc 1350    # D2: graphics lock (of 210–2100, 7 MHz steps)

# benchmark: fixed ~150-token prompt + nonce, greedy,
#   3× POST /v1/chat/completions max_tokens=1  (prefill)
#   3× POST /v1/chat/completions max_tokens=256 (decode)
# against the box's production Strata endpoint (:8181)
```
