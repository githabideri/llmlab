# 2026-10-07 QFN OptLlama O11 re-run: the 42 MB/s re-reads are the PLE table — page faults strace cannot see — and the 98K prefill is I/O-dominated

**Date:** 2026-10-07 (run window 11:33–13:02 UTC; analysis the same evening)
**Category:** Experiment / benchmark
**Hardware:** backup / inference box — Intel i3-9100 (4C/4T, DDR4-2400 2-channel ≈ 38 GB/s), single RTX 3060 12 GB, 46.8 GB host RAM, **no swap**, NVMe. This run in a container with a **42 GiB *hard* cgroup memory limit** (the [10-05 run](2026-10-05-optllama-moe-cache-qfn-single-3060.md) ran the same figure as a soft/advisory limit); host and GPU sampled at 1 s / 10 s, per-process I/O at 2 s.
**Model:** Qwen3.8-Flash-Next, `GSQ-RCO-IQ2_XS` — two shards: `…-00001-of-00002.gguf` **39,225,954,592 B (36.4 GiB, expert table + dense)**, `…-00002-of-00002.gguf` **28,800,138,432 B (26.8 GiB, the PLE / n-gram table)**; MTP draft `mtp-Qwen3.8-Flash-Next-Q8_0` 2,786,568,256 B (2.6 GiB). Total model 70,812,661,280 B = **65.9 GiB**.

**TL;DR**

- **The O11 regime (owner-verified commit `925933801`, 22 888 MiB pin, 32 cache groups, the owner's full flag set) reproduces**: 16K decode **8.83 / 11.48** (mean 10.16) and 98K decode **6.06 / 6.36** (mean 6.21) — the same cluster as the 10-06 O11 result (9.9/9.8, 6.6/6.8); canaries 3/3 at both contexts; the 42 GiB hard guard **never fired** (cgroup peak 39.8 GiB; host available floor 1.24 GiB).
- **The per-file attribution left open by 10-05 is settled**: a 300 s `strace` window over a 98K decode shows **zero `read()` calls** — but `/proc/pid/io` shows **27–42 MB/s of *storage* reads in every one of the four 98K phases** while `rchar` (bytes actually read via `read()`) sits perfectly flat. The bytes therefore arrive as **mmap page faults against the 28.8 GB PLE shard**, not as `read()` of the expert table. 10-05's "~42 MB/s sustained" measurement was correct; its attribution was open, and this closes it.
- **`strace` is the wrong instrument for this question** — it counts syscalls, and a page fault is not a syscall from the process's point of view. The counter that sees page-fault I/O is `/proc/pid/io`: `read_bytes` − `rchar` = bytes fetched from storage that did not go through `read()`.
- **The 98K prefill is I/O-dominated**: TTFT 832–892 s ≈ the ~25–38 GB of PLE rows each 98K request re-faults at 27–42 MB/s (the 6.26 GiB ZFS ARC cannot retain the 26.8 GiB table, so each request re-reads most of it). The 16K phases show no such signature.
- **The PLE cannot be made resident on this box by any configuration**: whole model 65.9 GiB > the 54.5 GiB of memory containers that exist (39.8 cgroup + 8.4 VRAM + 6.3 ARC), and the 26.8 GiB table exceeds every single one of them. This matches the design intent from the primary sources: Codacus (10-05 update) — the ~28 GB table "gets streamed from the SSD as they're needed, and you don't need to fit them in RAM at all".

## Goal

The 10-05 report's update (2026-10-06) left exactly one open measurement: *item 3* — the 10-05 campaign measured ~42 MB/s of sustained `read_bytes` during 90K decodes but never attributed the bytes to a file; a per-fd `strace` sample during a 90K decode was proposed as the decider and "armed but its sampler did not survive the window". This run re-executes the same O11 regime with (a) that `strace` window, and (b) a 2 s-cadence `/proc/pid/io` sampler running across the entire 98K section, so the attribution can be settled from the data rather than inferred.

## Setup / Commands

Same O11 configuration as 10-06, verbatim: build `925933801` (2026-09-14, the owner-verified `moe-cache` commit),

```
llama-server -m …/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --moe-expert-cache-size 32 --moe-expert-cache-host-pinned-mb 22888 \
  --load-mode none --lazy-mode on -ctk q8_0 -ctv q8_0 -kvo --cache-ram 0 \
  --ple-prefetch --phase-aware-workspace --live-context-workspace \
  -t 4 -c 98304 --port 8090
```

(`-m` names shard 1 only; the engine derives shard 2 from the filename — shard 2 is never in `-m`, never copied, only mapped.) Samplers: per-process `/proc/pid/io` at 2 s for the whole 98K section (3,598 s) plus a dedicated 300 s window; `strace -f -e trace=read,pread,pread64` over 300 s of the second 98K decode; cgroup `memory.peak`/`memory.current` (1 s and 10 s), host `MemAvailable` (1 s), `nvidia-smi` (mem MiB). Bench: 256 out-tokens per rep, 2 reps per context (16K, 98K), 3 numeric canaries per context; the 35B service on the box was not running during the window.

## Results

**Throughput (t/s; r1 / r2, mean):**

| Context | Prefill (s) | Decode |
|---|---:|---:|
| 16K | 135.0–140.8 | **8.83 / 11.48 (10.16)** |
| 98K | **832.1–891.8** | **6.06 / 6.36 (6.21)** |

Canaries 3/3 at both contexts; HTTP 200 on all 4 reps.

**The I/O the `strace` window could not see** (`/proc/pid/io`, 2 s cadence, the whole 98K section; `read_bytes` = bytes fetched from storage, `rchar` = bytes via `read()`):

| Phase (wall) | `read_bytes` rate | Δ over phase |
|---|---:|---:|
| 98K r1, first half | 40.3 MB/s | 35.5 GB |
| 98K r1, second half | 42.4 MB/s | 38.0 GB |
| 98K r2, first half | 39.9 MB/s | 35.5 GB |
| 98K r2, second half | 26.9 MB/s | 24.7 GB (4.7-min zero plateau at phase start — the ARC-hit window) |

`rchar` is **flat at 39,235,447,630 B** for the entire section — and ≈ exactly the shard-1 size (39,225,954,592 B) plus ~10 MB of miscellaneous reads. The shard-1 `read()` copy (the `--load-mode none` materialization) was complete by the first 98K sample; the 300 s `strace` window inside it confirms it: `preads=0 bytes=0 fds=0`. So **every one of the ~130 GB of section I/O bytes is mmap page faults**, and the only mapped file large enough to be the target is shard 2, the 28.8 GB PLE table: each 98K request re-faults ~25–38 GB of it. Cumulative for the run: ~123 GB (load + 16K phases) + ~130 GB (98K section) ≈ **253 GB of storage reads in 1.5 h**.

**Memory map (measured):**

| Piece | Size | Where it lives |
|---|---:|---|
| Pinned expert groups (32 of 48) | 22,888 MiB = 24,031,437,824 B (22.3 GiB) | mlock'd anon — non-reclaimable |
| Staged groups (16) | 16 × 773,326,400 B ≈ 11.5 GiB | anon, copied on demand (admission log: `source_budget_exhausted`) |
| Dense + MTP | ~4.7 GB | split cgroup / GPU |
| Cgroup `memory.peak` | **42,777,223,168 B = 39.8 GiB** | the true peak (1 s and 10 s samplers agree) |
| VRAM peak | 8,593 MiB (8.4 GiB) | of 12 GB |
| PLE (shard 2) | 26.8 GiB | **never in RAM** — mmap'd, page-faulted, cached only in the host's 6.26 GiB ZFS ARC |
| Host | 46.8 GiB total | available floor 1.24–1.27 GiB, never OOM |

Note the distinction from 10-05's "35.4 GB expert table": the 39.8 GiB here is the *cgroup peak* of the whole container. (The 10-second sampler's maximum *current* reading is 37,969,200,000 B = 35.4 GiB — a sampling artifact; the cgroup's own `memory.peak` counter is the number to quote, and it says 39.8.)

## Observations

1. **The 10-05 "~42 MB/s during 90K decode" is real and is the PLE table, not the expert table.** The per-phase rates here (27–42 MB/s) reproduce it exactly, and the `rchar`-flat / `read_bytes`-rising split attributes it to page faults on shard 2. The expert table is a full anonymous copy in RAM (its bytes were all `read()` during load) and contributes zero storage I/O after load. The 10-05 "weaker explanation" the update flagged — scattered lazy page faults on the 90K PLE rows — is the one that was right.
2. **The 98K prefill is I/O-dominated, which is why its TTFT is 832–892 s.** 25–38 GB of PLE rows at 27–42 MB/s is 10–15 minutes of pure fault-handling, which is exactly the observed prefill length. Compute is a small fraction of it; the 16K phases (TTFT 135–141 s, no I/O signature) are the compute-bound reference. On this box class, deep-context TTFT ≈ *table size touched* ÷ *SSD read bandwidth* — a term no engine flag can remove, only a larger ARC, a faster SSD, or a smaller table.
3. **Residency arithmetic says the PLE was never resident here, by construction.** Containers that exist: cgroup 39.8 GiB + VRAM 8.4 GiB + ARC 6.3 GiB = **54.5 GiB < 65.9 GiB** (whole model); the PLE alone (26.8 GiB) exceeds every single container. No configuration of this engine changes that — the pin budget only ever applies to the expert shard, `--ple-prefetch` is a read-ahead (it is what produced the 4.7-min ARC-hit plateau), and `--lazy-mode` is about on-demand *mapping*, not residency. Even a hypothetical "free 10 GiB by lowering the pin" would not help: the table is cached in the host's ZFS ARC, whose 6.26 GiB cap is a host sysctl, not a pool the process can claim.
4. **The hard guard did its job invisibly.** 39.8 GiB peak against the 42 GiB limit leaves 2.1 GiB of headroom; the host floor stayed at 1.24–1.27 GiB and the host never paginated (no swap). Contrast the 09-23 incident (full pin, no limit → host death): the cgroup cap converts the worst case into a container OOM.
5. **The gap to the 40+ t/s class is the pinned-experts + cores profile, re-confirmed.** Strata (the engine behind the "40+ t/s on a 6-core Ryzen" reference: [Niko1221/Strata](https://github.com/Niko1221/Strata)) pins the *experts* in RAM (its table says 48 GB RAM for IQ2_XS; 64 GB is Codacus's IQ3_XXS number) and SSD-streams the PLE — the same division of labor this box is trying to run at 46 GB, 4 cores, PCIe 3.0 class. Same commit, same 22 888 MiB budget: 6.21 t/s here vs the owner's 33.7–36.2 (12 cores / 62.1 GiB) — a 5.4–5.6× gap that tracks cores × PCIe generation, not RAM capacity.

## Method note (general)

- **Syscall tracing cannot see page-fault I/O.** `strace` counts `read`/`pread` invocations; a page fault on a mapped file generates no such call from the process. A "no/zero file I/O" claim based on `strace` is a claim about *syscalls only*. The storage-byte counters are `/proc/pid/io` (`read_bytes`: bytes fetched from storage; `rchar`: bytes via `read()`) or the block-device stats — and **`read_bytes` − `rchar` is precisely the page-fault component**.
- **A "X fits in RAM/VRAM" claim needs the arithmetic, not the log line**: sum the measured containers that exist (cgroup peak, VRAM peak, ARC/page-cache cap) and compare to size(X). "The model loaded in 3 s" is not evidence of residency — with `--load-mode none` the 39 GB copy ran for ~30 minutes after that line.

## Energy

No power sampler ran this session (gap against the 10-05 convention). Closest reference, same regime: 10-05's campaign average 74.6 W GPU-only, 77–85 W in the active cells; a 98K 256-token rep here (~42 min at ~80 W) ≈ **35 Wh per 1K generated tokens, GPU-only**, an extrapolation from those figures, not a measurement.

## Conclusion

- **Closes the 10-05 update, item 3**: the sustained ~42 MB/s re-reads during deep-context decode are **page faults on the PLE/n-gram shard** (mmap, `read()`-free), not expert streaming. The expert table, once materialized by `--load-mode none`, is a pure-RAM resident for the life of the process.
- **The O11 regime is the confirmed configuration for this box**: reproducible 10.16 / 6.21 t/s at 16K / 98K under a hard 42 GiB guard with 2.1 GiB of headroom, canary-clean, no host impact. It is the top of the in-box range, and nothing left in the flag space changes the binding terms (4-core staging, 6.26 GiB ARC, PCIe 3.0).
- **The 98K prefill (≈15 min) is an I/O tax of the quant's table size on this box's SSD + ARC** — the lever is hardware (faster SSD / more ARC) or a smaller table (a lighter quant), not an engine flag.
- **40+ t/s on this model stays the Strata/Codacus machine class**: 48–64 GB RAM for the pinned experts + 6+ modern cores + a newer PCIe generation. On this box the corrected ceiling stands: ≈ 10 t/s at 16K, ≈ 6 t/s at 98K, prefill I/O-bound at depth.

**Supersedes (closes):** the [2026-10-05 report](2026-10-05-optllama-moe-cache-qfn-single-3060.md), Update (2026-10-06) item 3 (per-file attribution of the 42 MB/s decode re-reads — open) and the O11 "sampler did not survive the window" note.
