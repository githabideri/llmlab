# 2026-10-26 Loaded-idle power on the single-3060 box: a fully loaded but inactive Strata engine sits at P8, ~5.9 W, and keeping 11.4 GB of VRAM shows no detectable additional GPU draw

**Date:** 2026-10-26 (one ~2.5 h window; the box was booted for this and left running)
**Category:** Experiment / benchmark (measurement campaign; **no permanent changes** — every control was reverted and the production endpoint re-verified live)
**Hardware:** backup / inference box: Intel i3-9100 (4C/4T, DDR4-2400 2-channel), 48 GB RAM, **four** hard disks, single **RTX 3060 12 GB** in the CPU-direct PCIe 3.0 x16 slot; Proxmox host with an Ubuntu 24.04 inference container running the box's production Strata endpoint (v0.1.41, Qwen3.8-Flash-Next IQ2_XS, 98K context, 19 GiB resident expert budget, MTP spec 4, KV int8, mmap — the [2026-10-08 winner](2026-10-08-strata-v0141-single-3060.md), unchanged). The box is a dual-purpose machine: backup source *and* inference host, which matters for every wall-level number below. Driver **595.71.05** (distribution package, **proprietary** kernel module, GSP firmware active, persistence mode on).
**Question:** can a fully loaded but *inactive* Strata engine keep its CUDA context and VRAM while drawing less power? What is the floor, and which of the usual levers (P-state daemon, clock locks, power limit, PCIe ASPM, runtime D3) actually move it on this platform?

**TL;DR**

- The loaded-idle floor is **P8 at 5.8–6.2 W on the card, stable for 15+ min** (no drift with time since last generation). The driver-only state (4 MiB) draws 5.82 W and the loaded state (11473 MiB) 5.93 W — a 0.11 W difference, inside the card's own ±0.2 W wobble, so **no detectable additional GPU power** comes from keeping 11.47 GB allocated. The idle floor is the P8 board floor in both states; the difference is not zero, only below this measurement's noise.
- **No tested power-management change improves loaded idle.** `nvidia-pstated` (source-reviewed, built, 10-min run): **+0.2 W** (its forced P8 pins the graphics clock at 420 MHz instead of 210), no errors, clean restore. Clock locking *raises* idle: `-lmc 5001` parks the card in **P3 (19.8 W idle)** with a −5 to −15% decode regression in the short test; `-lgc 1350` parks **P5 (+5 W idle)** and cuts the *peak* from 86.3 W to 65.3 W with no visible regression in the three-rep test — a peak number, not an average-power or general no-cost claim. The 170 W power cap is never near (86 W peak).
- **Runtime D3 did not occur under the tested conditions.** With a live context and persistence mode on, the PCIe device stayed `active` the whole time (`runtime_suspended_time=0`, `wakeup=disabled`). Whether that is a hard platform limit (desktop boards generally have no PCIe PME wake) or a driver/GSP policy choice, the observations do not disambiguate; it is recorded as an inference, not a measurement.
- PCIe idles at **Gen1 x16** (the documented idle-link state) and retrains to **Gen3 in under 2 s** under load. ASPM L0s/L1 is advertised but disabled at the link; the host kernel runs `pcie_aspm=default`. The one untested lever is a `pcie_aspm` cmdline change + reboot — not done here.
- **At the wall** (the box's smart-plug power meter via Home Assistant, ~10 s cadence): idle windows sit at **63–65 W** (the owner's one-off external observation of ~60 W falls in this band; its workload state is unspecified), and generation bursts take the whole host to **124–184 W**. The four disks and the Proxmox host are the dominant, noisiest consumers; at the wall the campaign's levers are indistinguishable, so the per-component numbers are what the conclusions rest on. These wall figures describe *this* dual-purpose deployment; they do not transfer to a bare always-on mini-PC.

**Power measurement, three sources at three resolutions:** (1) **NVML** `power.draw_w`, 2 s cadence, ~0.1 W resolution — the GPU; (2) **RAPL** CPU-package `energy_uj` deltas, 2 s — the i3-9100 package; (3) **AC wall** — the box's power path is a smart plug *with* a power meter, surfaced in Home Assistant (power plus a cumulative energy counter; ~10 s cadence, 1 W resolution, history merged). The owner's ~60 W one-off is an external observation of that same plug: duration and workload state unspecified, it lands inside the meter's 55–64 W idle band. Two caveats: there is **no synchronized AC series** at the 2 s sampler cadence (HA history merging drops points), and the wall is a *whole-host* figure (disks, Proxmox, backup jobs, container, PSU losses), not the bare board. Cross-checks: the plug's energy counter reads 62–64 W/hour over the session blocks, and the benchmark spikes land in the meter exactly where the sampler puts the requests. One provenance note: the box's clock ran an hour ahead of the meter's timezone (stale DST after 2026-10-25); the two series were aligned by absolute epoch, verified against the power-on event (energy counter resuming) and the spike alignment. Separately, the host's node_exporter RAPL collector is broken on this kernel (7.0.14), so Prometheus has no CPU-power series for this box — a follow-up, not a campaign blocker.

## States, commands, results

Fixed across the whole window: model/quant (same shards as 10-08), engine binary and version, driver, expert budget, MTP settings, context geometry, prompt and generation lengths. The benchmark is a fixed ~150-token prompt plus a unique nonce per rep (greedy), 3 reps of prefill-only (`max_tokens 1`) and 3 of 256-token decode, against the production endpoint. The A0 stop+restore of the serve unit was registered per the box's prod-service-change discipline; the unit file was never touched and the endpoint was re-verified after each restore (model re-loads in ~60 s, mmap). Times are UTC; the sampler CSVs carry epoch timestamps.

| State | What | GPU avg (W) | GPU max | P-state | gr/mem MHz | PCIe | VRAM | CPU pkg avg (W)¹ | Wall avg/max (W)² | Duration |
|---|---|---|---|---|---|---|---|---|---|---|
| **A0** | driver loaded, no context (serve stopped) | 5.82 | 5.96 | P8 | 210/405 | Gen1 | 4 MiB | 40.1³ | 64 / 72⁴ | 90 s |
| **A1** | Strata loaded, idle | **5.93** | 6.07 | P8 | 210/405 | Gen1 | 11473 MiB | 21.3 | 63.8 / 79 | 5 min |
| **A2** | active generation (3 prefill + 3 decode) | 33.8 (window) | **86.3** | P2 when busy | 1777–1890 / 7301 | Gen3 when busy | 11473 | 33.1 | **134.8 / 184** | 2 min |
| **A3** | post-generation, resident | 5.93 | 6.20 | P8 | 210/405 | Gen1 | 11605 | 15.9 | 64.4 / 85 | 15.5 min |
| **B1** | A1 + `nvidia-pstated` (forced-P8 daemon) | 6.13 | 6.48 | P8 (forced) | **420**/405 | Gen1 | 11605 | 18.8 | 63.6 / 78 | 10 min |
| **B0′** | 5-min clean check after B1; first 3.9 min clean P8 (5.91 W), then the D1 lock was applied mid-window; the trailing 60–75 W GPU events are D1's own first request (44 s window overlap), not production traffic | 14.2 (window) | 75.2 | P8 → P3 (D1 lock) | 210 → 1882 / 405 → 5001 | Gen1 → Gen3 | 11605 | 25.1 | 64 / 76 (pre-lock) | 5 min |
| **D1** | A2 with `-lmc 5001` (memory clock locked) | 40.2 (window) | 80.9 | P3 when busy | 1882 / 5001 | Gen3 | — | 30.5 | **126.8 / 174** | 2.5 min |
| **D2** | A2 with `-lgc 1350` (graphics clock locked) | 22.3 (window) | **65.3** | P2 when busy | 1350 / 7301 | Gen3 | — | 25.4 | **128.3 / 161** | 2.5 min |

¹ RAPL package power on the i3-9100, 2 s deltas. ² The plug meter via Home Assistant, ~10 s, merged: whole host, not the bare board. ³ Elevated by unrelated host workload during that 90 s (a CPU-side host process; the wall did not spike, so it was not a disk storm). ⁴ Only two meter points in that 90 s window — order-of-magnitude only.

**Throughput, same benchmark** (the cold first rep is included; it happened):

| Config | Prefill (s) | Decode 256 tok (t/s) | GPU peak |
|---|---|---|---|
| A2, no lock | 4.43 cold / 1.85 / 1.89 warm | 18.3 / 26.9 / 25.2 | 86.3 W |
| D1, mem @5001 | 2.01 / 2.01 / 2.01 | 22.8 / 23.8 / 25.4 | 80.9 W |
| D2, grc @1350 | 1.91 / 1.89 / 1.88 | 23.8 / 27.1 / 26.4 | 65.3 W |

Decode is host-side on this box (the 10-08 attribution: the 4 cores do the expert compute, the 3060 does <1.6% of per-token time). D2's 1350 MHz ceiling showed no visible regression in this three-rep test (warm-decode mean 26.4 vs 26.1 t/s) — consistent with the SMs not being the measured bottleneck (natural 1777–1890 MHz, below the 2100 max), but a short single-workload test is not evidence of zero cost in general. D1 regressed decode by −5 to −15% (22.8–25.4 vs 18.3–26.9 t/s, warm 23.8/25.4 vs 26.9/25.2); that is consistent with the reduced memory bandwidth (5001 vs 7301 MHz), but the lock also changed the P-state the card ran in (P3, graphics 1882), so the mechanism is a hypothesis, not an isolation.

**Integrated energy** (trapezoid over the 2 s samples): A3 (15.5 min) consumed **1.528 Wh on the GPU + 3.765 Wh on the CPU package** — ~21.8 W combined, i.e. **~22 Wh per hour** at that rate (A1's equivalent is ~27 Wh/h, its CPU figure including some host load). The A2 window burned 1.135 Wh GPU + 0.877 Wh CPU = 2.01 Wh total for 771 generated tokens (3×256 decode + 3×1 prefill) → **~10 J/token combined, window-average; ~5 J/token GPU-only**. The window is ~53% active (31/59 samples above 30 W GPU), so active-segment rates are higher: GPU>30 W samples average ~110 W combined. The wall averaged 135 W in the same window, but that includes host disks and PSU losses at load, so no J/token is derived from it.

## Observations (the mechanism, evidence vs hypothesis)

1. **P8 is the floor with a context, and the driver gets there by itself.** In the A2 window 28 of 59 samples (all the inter-request gaps) are P8 at 210/405 MHz — the card downclocks within ~2 s of the last CUDA work. Any daemon's only useful action is to force exactly the state the driver already chose; B1 demonstrates the residual (the forced P8 holds the SM clock at 420 instead of letting it settle to 210, +0.2 W). Verdict: the daemon exists for cards that stick in high P-states after load; this one does not.
2. **VRAM residency: no detectable additional draw.** A0 (4 MiB) vs A1 (11473 MiB): 5.82 vs 5.93 W, same clocks — 0.11 W, inside the card's own ±0.2 W wobble measured within each window. What the data supports is "no detectable difference", not a precise per-GB cost.
3. **No time drift.** 15.5 min post-generation (A3) is statistically identical to fresh-load idle (A1). Whatever "the engine has been running a while" effect exists, it is not in the power trace.
4. **Runtime D3 did not occur.** With a live context and persistence mode on, the PCIe device showed `power/control=auto`, `runtime_enabled`, but stayed `runtime_status=active` with `runtime_suspended_time=0` and `wakeup=disabled` for the whole campaign. Those observations establish that *this driver/platform combination, under these conditions, does not suspend*; they do not establish which layer is the blocker (no PME on the desktop board, persistence-mode policy, or GSP behaviour would all be compatible). A context is a precondition for staying out of D3 in any case, so "below the P8 floor with a usable model" still means "release the context".
5. **PCIe link behaviour matches the 2026-06-30 finding and is now instrumented:** Gen1 x16 at idle, Gen3 x16 under load, retrain in <2 s. LnkCap advertises ASPM L0s+L1; LnkCtl has it disabled (VBIOS/driver side); the host kernel parameter is `default`. Untested lever: `pcie_aspm=force` (cmdline + reboot).
6. **`nvidia-pstated` implementation notes** (43 commits, last 2025-11-02, maintained; built from source for this test): it drives state with NVML (temperature/utilization) plus a dlsym'd `NvAPI_GPU_SetForcePstate` — the call **is** supported on this GeForce (returns NVAPI_OK, unlike what one would assume for a desktop card). Its exit path hardcodes `enter_pstate(16)` as "restore automatic management": on a datacenter GPU P16 is a real high state; on this GeForce P16 does not exist and the same NVAPI_OK return is how "back to auto" works. No persistent configuration is written by the daemon; nothing to clean up but the process.
7. **Clock locking moves the P-state, not just the clocks** — the reason both D-arms raise idle: locking memory at 5001 forbids the P8 pair (mem 405), so the selector parks at P3 (19.8 W); locking graphics at 1350 forbids the 210 MHz idle pair, so it parks at P5 (10.7 W). There is no "active-only" clock mode in NVML.
8. **The supported memory-clock ladder on this card:** 405 / 810 / 5001 / 7301 / 7501 MHz, each with its own graphics range (the `-q` Supported Clocks table; `-lmc` accepts any tier, the earlier assumption that only 7501 was settable was wrong).
9. **The wall is host-dominated and campaign-noise is host traffic.** The meter's idle band (55–64 W) is stable across A0/A1/A3/B1 — at the wall the driver-only vs loaded states are indistinguishable (64.0 vs 63.8 vs 64.4 W), and D1 vs D2 are indistinguishable too (126.8 vs 128.3 W, raw series overlapping). The generation bursts (A2 to 184 W, D1 to 174 W, D2 to 161 W) exceed the GPU+CPU sum plus the idle residual by 35–50 W at the peaks: host disks and PSU efficiency at load, not separable from this data. No live production request was observed during the campaign; the one apparent "external" 75 W GPU event (B0′) turned out to be the D1 benchmark's first request, whose sampler overlapped B0′ by 44 s.

## Conclusion

1. **The campaign's question, answered:** yes — a loaded, inactive Strata engine keeps its context and VRAM at the card's lowest power state. On this box that is **5.9 W on the GPU** (P8, 210/405 MHz, Gen1 link), with no detectable additional draw from the 11.47 GB of VRAM. The box-level idle (GPU + CPU package) is ~22–27 W; the whole-host wall is 63–65 W, the extra ~40 W being the disks, the Proxmox host and their inefficiencies.
2. **Nothing driver-side gets you lower while keeping the context.** The daemon no-ops (measured, not inferred); clock locks raise idle by 5–14 W to buy *peak* reduction you mostly don't need (86 W peak against a 170 W cap); D2 (`-lgc 1350`) is the only config with a plausible use case if the duty cycle ever flips to generation-heavy, and even then it costs +5 W per idle minute for a 21 W peak cut whose average-power benefit is not established. Recommended: **deploy nothing** — the stock driver behaviour is already the optimum for this duty cycle.
3. **The only tested way below the P8 floor with a usable model is releasing the CUDA context** (engine-level unload-on-idle; the mmap design makes reload ~60 s, measured in this window, with the model files already page-cached). Whether a platform can additionally reach D3 with a context *gone* was not tested here.
4. **The wall numbers are deployment-specific.** This box is a backup source with four disks, and that is most of what its 63–65 W idle wall is. A bare always-on inference mini-PC is not measurable from these numbers; if a candidate node (BD775i-class) gets built, run the same recipe on it (2 s sampler + plug meter, both in the appendix). What does transfer is the card-level result: the loaded context itself costs ~5.9 W at P8, on whatever board it sits.

## Follow-ups (out of scope here)

- `pcie_aspm` cmdline experiment (needs a reboot window on the box; the link-control evidence above is the pre-test).
- Engine-level idle-unload mode in Strata (release context after N minutes, re-init on demand). The energy arithmetic: the measured loaded-vs-driver-only GPU difference is ≤0.15 W, and the CPU-package difference is confounded by host load (the engine process sits inside the 15.9–21.3 W package and is not separable), so the saving per idle hour is at most a few Wh at best — against a ~60 s reload. It is a responsiveness feature, not an energy one.
- Close the telemetry gaps: a 2 s wall series (direct plug polling or a higher-rate logger; HA's merged history is too coarse to synchronize against the sampler), a per-device or job-correlated study of the host's ~40 W idle residual (four disks vs Proxmox vs backup jobs vs PSU), and a RAPL collector that works on kernel 7.0 (or a 10-line `energy_uj` exporter) so Prometheus gets the CPU series.

## Raw data and commands

The 2 s CSVs (8 files, columns `ts, gpu_w, pstate, gr_mhz, mem_mhz, util_pct, vram_mib, pcie_gen, cpu_busy_pct, cpu_pkg_w`) and the wall series (raw JSON of the plug meter's power and energy entities for the session, plus the per-window extracts above) live in the private repo under `temp/power-campaign-2026-10-26/` alongside the sampler script; they are ephemeral and regenerate with the commands below.

```bash
# sampler (PVE host): nvidia-smi query + RAPL delta every 2 s
nvidia-smi --query-gpu=power.draw_w,pstate,clocks.gr,clocks.mem,utilization.gpu,memory.used,pcie.link.gen.current --format=csv,noheader
# CPU package watts: Δ(/sys/class/powercap/intel-rapl:0/energy_uj)/Δt  ·  busy%: /proc/stat

# AC wall: the box's smart-plug power meter, surfaced in Home Assistant
# (power entity ~10 s cadence + cumulative kWh counter as cross-check).
# NOTE: the box's clock ran an hour ahead of the meter's timezone during the
# session (stale DST); align by absolute epoch, not by clock.

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
