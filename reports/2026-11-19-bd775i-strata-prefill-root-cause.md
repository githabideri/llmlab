# Root-cause analysis — low-RAM prefill on the 48 GB BD775i node

## The gap

Full expert arena (40 GiB pinned, no file tier): prefill ~800 to 1,020
tok/s. Deployed two-session config (24 GiB pinned + 16 GiB file tier,
int8 KV, kv-resident 20,480): prefill ~200 to 233 tok/s at 32K/64K in
the same config served to two consumers. The gap is about 4.5x, and it
is not a compute gap: the per-token GPU/CPU work is the same model.
(measured: one-shot baseline battery; served two-session battery,
32K/64K/128K)

## Where the time goes (measured)

Per 32K-token prompt, deployed config, exclusive window:

- Logical expert bytes read from file during the prompt: 60.7 GB,
  identical across all I/O arms (routing is unchanged by the read path).
  ~1.85 GB per 1K prompt tokens; the ratio rises to ~2.3 GB/Ktok at
  200K as deeper prompts touch more expert diversity (measured in the
  served 200K turn: 467 GB logical).
- Buffered file tier (v0.1.41 auto selection, confirmed "through the
  file cache" at both decision points in the engine log): 233 to 251 GB
  of physical NVMe reads per run for ~76 GB of logical file traffic,
  2.5 to 3.3x amplification, 114K to 962K major faults (the 200K turn:
  1,234 GB physical for 467 GB logical).
- Direct reads (`STRATA_UNBUFFERED_LOAD=1`): 118.5 to 123 GB physical
  per 32K run, ~1.07x the logical; ~2.9K major faults.

The NVMe does 2.3 GB/s sequential cold, 1.9 GB/s cold-direct; the
effective expert-read rate in both arms is ~1.2 GB/s (small random
blobs), i.e. both arms run the same pipe. The buffered arm is 2.4 to
2.8x slower in the served regime because it moves 2.5x to 2.7x the
bytes: pages evicted mid-prompt are re-faulted from the device later.
The warm-cache ceiling (17.7 GB/s, 7.7x the cold-direct rate) is what
the numbers would look like if the cache could hold the complement; it
cannot, in this regime.

**Attribution: the low-RAM prefill in this configuration is
storage-volume-bound via page-cache churn. Not bandwidth-bound, not
CPU-bound, and not the #1194 O_DIRECT regression.**

## Why the auto rule is miscalibrated here

The v0.1.41 auto policy (`file_cache_keeps`) keeps a 3 GiB / 5 percent
floor of the file tier in the page cache when headroom allows, on the
assumption that the kept pages are the ones that come back. That
assumption holds on 20 to 32 GB boxes where the complement (the part of
the experts not pinned) is smaller than the usable headroom and a small
hot set dominates the repeats. Here the complement (16 GiB) is larger
than the headroom (~15.3 GiB after the 24 GiB pin, MTP, and process
memory): the entire complement is hot by construction (it is the
profile-ranked tail of the expert set, and a 200K prompt routes through
most of it). A cache that cannot hold what comes back does not keep the
hot set, it churns it: every kept page evicts another page that comes
back again, and the device serves the same blocks over and over. When
complement is at or above headroom, direct reads are the lower-volume
path: each byte is read once and nothing is evicted. Measured: 2.4 to
2.5x prefill in one-shot arms (n=3 each), 2.4 to 2.8x in the served
two-session battery (32K/64K/128K, one run each per arm at serve
scale), decode unchanged (9.1 to 10.8 vs 9.7 to 10.0 tps solo, n=3).

## The history that made this subtle

Upstream issue #1194 is the same trade in the opposite direction: an
older release flipped the file tier to unbuffered unconditionally and
regressed prefill 40 to 80 percent on machines where the cache *does*
fit (small complement). The subsequent fix (prefer the cache, 3 GiB
floor) is correct for those machines and wrong for this one. The
v0.1.45 release added `STRATA_FILE_CACHE_OLD_RULE` to restore the old
behavior explicitly, which is a knob, not a calibration. The general
fix is to make the keep-size a function of the complement relative to
headroom, not a floor of the tier size.

## Why full arena is 4.5x faster (the control case)

The arena pins all 40 GiB of experts: zero file traffic after the
one-time ~33 s load at 1.2 GiB/s (measured). Prefill then runs at the
GPU/CPU-compute rate (~1,000 tok/s). Its cost: 40 GiB pinned leaves too
little for two deep sessions plus MTP, which is why the two-session
config exists at all.

## Why pure mmap differs from pinned

Pure mmap (no pin) leaves the whole file tier to the page cache: with
nothing else pinning 24 GiB beside it, the cache can hold the
complement, and a warm 64K prefill reads only 13.5 GB physical
(measured). That is the 17.7 GB/s regime. Its decode is ~2x slower than
pinned (11 tps at 64K vs 9.9 to 10.5, part-1 battery) because the
complement competes with KV and process memory for the whole cache and
the decode path takes the hits; with the pin, decode reads 24 GiB from
real RAM at memory speed and only the 16 GiB tail touches the files.
Mapped also wins one-shot 200K prefill (253 vs 200 tok/s) for the same
reason. The two-session service chose the pin for decode and headroom;
this campaign measures the price of that choice and removes the churn
without giving the pin up.

## What the fix does and does not change

`STRATA_UNBUFFERED_LOAD=1` on the deployed config: prefill 2.4 to 2.8x
faster at every context length tested (32K to 200K, one-shot and
served), decode unchanged, major faults down ~40x, physical NVMe volume
down ~2x. It does not change the 2-session decode rate (slot-decode
bound, no MTP in slots, 18.8 percent of routed experts over PCIe at the
24 GiB pin), does not change the 96 GiB picture (there the complement
fits and the trade reverses, report E), and does not change what an
engine restart costs a session (one full re-prefill: 422 s at 200K,
direct; slot save/restore is unavailable with batch, upstream #465).
