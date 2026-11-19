# BD775i Strata prefill: the low-RAM optimization matrix (report B)

## Scope

Every lever this box can reach, measured on the installed model (Qwen
3.8-Flash-Next, GGUF `gsq-rco` IQ3_XXS, 75.8 GB across two shards, 12 GB
GPU, 44 GiB RAM, single-channel DDR5, NVMe). The regime is the
two-session service: 24 GiB of experts pinned in RAM (2,304 of 11,520
routed expert slots), the remaining 16 GiB of experts in the file tier,
MTP on, batch 2, int8 KV, kv-resident 20,480. One engine at a time; no
other GPU workload; the same llama.cpp commit throughout
(3cf03257, 1.72.1-equivalent).

Headline: the lever that mattered is the file-tier I/O mode, not any of
the chunk, stager, budget, or engine levers. In this regime, direct
reads of the 16 GiB complement are 2.4 to 2.5x faster on prefill, with
decode unchanged.

## The mechanism, in one paragraph

With the 24 GiB pin, the engine's file tier must serve 16 GiB of
experts per pass. The auto policy computes how much page-cache to keep
(`file_cache_keeps`) with a floor of 3 GiB (5 percent of the file tier)
and a ceiling of 15 percent. With 44 GiB of RAM the computed value
stays at the floor: 3 GiB. The complement (16 GiB) is therefore larger
than the usable page-cache headroom (roughly 15 GiB), so the page cache
cannot hold it. Every prefill then evicts and re-reads the same blocks
in a storm: measured 1,234 GB of physical NVMe reads for 467 GB of
logical expert reads on one 200K-token prompt, 962K major faults,
10 to 15 percent full PSI. Direct reads bypass the cache entirely: each
byte is read once, major faults drop to near zero, and physical reads
fall to about half. The effective expert-read rate is roughly 1.2 GB/s
in both modes (small random blobs over PCIe-attached NVMe), so the
bottleneck is storage volume, not bandwidth, and removing the
re-reads is worth 2.4x. The 3 GiB / 5 percent floor is miscalibrated
for this regime: it is right when the complement is small (full-cache
machines, where keeping 3 GiB warm helps) and wrong when the
complement exceeds the headroom, where the cache becomes churn.
Upstream issue #1194 documents the same auto-selection problem from
the other direction (an older release flipped unbuffered on
unconditionally and regressed prefill 40 to 80 percent).

## The matrix

All cells: 64 or 32 output tokens (pre-fill dominated), median of the
runs shown, exclusive window, cold NVMe state per arm where the arm
changes I/O behavior.

### I/O mode x context length

| arm (I/O) | 32K | 64K |
|---|---|---|
| auto (buffered) | 207 / 222 / 200 tps (n=3) | 233 / 230 / 230 tps (n=3) |
| direct | 556 / 543 / 543 tps (n=3) | 581 tps (n=1) |
| ratio (direct/auto) | 2.4x to 2.5x | 2.5x |

NVMe attribution per 32K prefill: buffered 248 to 251 GB physical
(953K major faults); direct 123 GB physical (2.9K major faults).

### chunk size (both I/O modes, 32K)

| chunks | auto (buffered) | direct |
|---|---|---|
| 4096 | 118 / 118 / 125 tps (n=3) | 301 / 285 / 290 tps (n=3) |
| 8192 (auto) | 207 / 222 / 200 tps (n=3) | 556 / 543 / 543 tps (n=3) |
| 16384 (auto) | = 8192 arm | = 8192 arm |
| 32768 (auto) | = 8192 arm | = 8192 arm |

4096 chunks are 1.7 to 1.8x slower in both modes (more expert
re-loads per token batch). 8192 is the local optimum; 16K and 32K
chunks produce no further gain, so the engine's auto value (8192 at
this model shape) is the right default.

### GGUF stager (buffered only, since direct ignores it)

| stager | 32K | 64K |
|---|---|---|
| auto (SSD profile) | 207 / 222 / 200 (n=3) | 233 / 230 / 230 (n=3) |
| forced RAM profile | 217 (n=1) | 273 (n=1) |

The forced-RAM stager sits inside the buffered range at 32K and 20
percent above at 64K with n=1; not distinguishable from run-to-run
variation given the buffered mode's 10 percent spread. The stager
profile is not the lever.

### expert pin budget (direct I/O, 32K)

| pinned | 32K prefill | solo decode (D8) |
|---|---|---|
| 24 GiB (2,304 slots) | 543 / 556 / 543 tps (n=3) | 9.7 to 10.0 tps (n=3) |
| 20 GiB (1,920 slots) | 526 / 514 tps (n=2) | 9.1 to 9.5 tps |
| 16 GiB (1,536 slots) | 379 tps (n=1) | 5.8 tps |

Larger pins are faster (less to read) and the 24 GiB value is the
largest that fits two deep sessions plus MTP plus process in 44 GiB.
The deployed budget stands.

### 200K context (served, two-session config)

| state | wall | effective |
|---|---|---|
| cold (fresh engine, direct) | 422.5 s | 485 tps |
| prefix-warm (128K of the same text prefilled minutes earlier) | 212 s | 960 tps |
| prefix-warm, 204,851 of 204,856 tokens reused | 0.16 s | (cache-hit path) |

The warm path shows what a stable conversation costs after the first
turn: five tokens of work. The cold path is the restart cost, and it is
the number a consumer pays when an engine restart loses the session.

## What this box cannot reach

- **Full-arena mode** (40 GiB pin + 40 GiB file cache) needs ~80 GB of
  RAM and cannot coexist with two deep sessions; it remains the
  one-session benchmarking regime (800 to 1020 tps there).
- **More than 24 GiB pinned** does not fit two sessions.
- **Decode in two-session load** (0.9 to 5.6 tps per consumer) is
  slot-decode-bound (no MTP in slots, 18.8 percent of routed experts
  over PCIe at the 24 GiB pin) and no I/O lever touches it; it moves
  with RAM (96 GiB) and/or upstream slot-MTP work.
- **Session persistence across an engine restart** is not available
  with batch 2 (slot save/restore is disabled whenever parallel
  requests are enabled; upstream #465). The cost of a lost session is
  one full re-prefill: 422 s at 200K in direct mode.

## Upstream fixes evaluated

- **#1194** (buffered/direct auto-selection): the v0.1.41 auto logic
  picks buffered here (complement 39.9 GiB < 40 GiB available at the
  load-time check, 10.5 < 40 at the KV allocation check), so the
  unconditional-unbuffered regression of the older release is not
  active. But the 3 GiB / 5 percent `file_cache_keeps` floor still
  makes buffered the wrong choice in this regime. Fixed locally with
  `STRATA_UNBUFFERED_LOAD=1`; the general fix is to let the
  floor/ceiling scale with the complement size, which v0.1.45's
  `STRATA_FILE_CACHE_OLD_RULE` knob approximates for 96 GiB machines.
- **#3060** (`STRATA_RAM_ADAPT`): new in v0.1.45, explicitly a no-op
  with batch (this service), so not applicable.
- **#1670 / #3976** (layer-major prefill for Qwen single-GPU): not in
  any pinned release; the 4 to 6x figure is an unmerged proposer
  branch on a 6 GB laptop. Watch item.
- **#1614** (shared KV pool): relevant to the 96 GiB two-session
  regime; watch item.

## Reproduction

Engine v0.1.41 (tag), llama.cpp 3cf03257, the model as published, the
config as in report D. The one environment variable that matters:
`STRATA_UNBUFFERED_LOAD=1`. Every other variable tested was either
ineffective (stager, chunk beyond 8K, prefetch, pool workers) or
harmful (kv-resident below 20,480, 4K chunks).
