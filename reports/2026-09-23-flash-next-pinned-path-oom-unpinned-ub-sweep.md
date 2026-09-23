# Qwen3.8-Flash-Next on a single 3060: why the pinned ubatch matrix can't run on this box, and what the unpinned path measures

**Date:** 2026-09-23
**Category:** Experiment / benchmark
**Hardware:** backup/inference box — Intel i3-9100 (4C/4T, DDR4-2400 ≈ 38 GB/s), single RTX 3060 12 GB, 40 GiB LXC soft limit on a 46 GiB host with no swap
**Model:** Qwen3.8-Flash-Next (Qwen4Exp) Unsloth `UD-Q2_K_XL` (3-shard GGUF, 78.9 GB), codacus llama.cpp fork `27c54b4b` build (`build-fable`)

---

## Goal

The follow-up to the [2026-09-20 single-3060 MoE hot-cache report](2026-09-20-flash-next-single-3060-moe-cache-backup.md): a ubatch/slots matrix (`GGML_CUDA_REGISTER_HOST=1` + `GGML_SCHED_PREFETCH_EXPERTS=1`) to find the highest sustained prefill rate at 64K context. The pinned cells crashed the host three times (00:01 / 03:43 / 07:57 UTC). Two questions: why does the pinned path crash this box, and what does the unpinned (prod-like) path actually measure?

## What the pinned path does — confirmed in fork source

`GGML_CUDA_REGISTER_HOST=1` makes the CUDA backend call `cudaHostRegister(buffer, size, Portable|ReadOnly)` (`ggml-cuda.cu:4878`) on the **mmap'd expert weight buffers**. That page-locks every touched page: it becomes a permanent charge of the container's cgroup and is **not reclaimable** by the kernel.

A single ~9.6K-token prefill touches nearly all experts, so one request pins tens of GB. The RAM guard measured the CT cgroup **flat at 40.7–42.8 GB for over two hours** (flat, not oscillating — the non-reclaimable signature) against a 40 GiB soft limit, on a 46 GiB no-swap host. The kernel then thrashed: `systemd-journald: Under memory pressure, flushing caches` → SIGABRT/SIGKILL of host services → hard reboot. Three crashes in one night, all correlated with pinned QFN loads.

The production preset (`qwen38-qfn-preset.ini`) is the **same layer split** as the matrix cells (`ngl 99`, `n-cpu-moe 99`, 64 slots, 64K ctx) — the only difference is that it does *not* set the env vars. Its expert reads go through reclaimable mmap page cache + the hot-expert VRAM profile + incremental `--cache-reuse 256` + serialized `--no-sched-async-cpu`: the cgroup churns at a few GB and never pins, so it is slow (fork README: ~6–7 GB/s bounce-buffer expert upload) but alive.

**Consequence:** pinned-path cells at ≥9.6K context are impossible on this host; fast-path numbers need a bigger-RAM box. A RAM guard could only ever *fire* against pinning, never protect — it was a band-aid.

## Unpinned quick test (this session)

Prod-equivalent flags (the adopted B64 cell minus `-fit off`), **no** `GGML_CUDA_REGISTER_HOST` / `GGML_SCHED_PREFETCH_EXPERTS`, on non-prod ports with the three resident services stopped (unit files untouched; everything restored and re-verified afterwards). Host activity lock held for the whole window; a 5-second logger on the host recorded host `MemAvailable` + CT cgroup `memory.current`.

```bash
llama-server \
  -m /path/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  --moe-cache-profile /path/q2-merged.csv --moe-cache-slots 64 \
  -ngl 99 --n-cpu-moe 99 -t 4 \
  --load-mode mmap -fa on -ctk q8_0 -ctv q8_0 \
  -c 65536 -np 1 --cache-reuse 256 --no-sched-async-cpu -b 2048 -ub <512|1024> --jinja \
  --host 127.0.0.1 --port 1808x
```

Bench: `bench-client.py <url> <model> 32 2 long4k-b.txt` — 9,640-token reversed-interleaved fixture (no KV prefix reuse), `max_tokens=32`, reps 2; prefill = prompt_tokens/TTFT.

| Cell | Config | Cold prefill @ 9.6K | Decode t/s | VRAM |
|------|--------|--------------------|-----------|------|
| U1 | ub **512** (prod config) | **45.8 t/s** (TTFT 210.4 s) | 8.4–11.1 | 11.4 GB |
| U2 | ub **1024** | **26.1 t/s** (TTFT 369.3 s) | 8.6–11.3 | 11.6 GB |

KV-reuse reps (`--cache-reuse 256`): TTFT ~0.4 s — the warm-cache path is unaffected by either ub.

## Observations

- **Stable.** No crash across ~15 min of heavy prefill. Host `MemAvailable` stayed ~30 GiB; after each prefill the CT cgroup held ~31 GiB of *page cache* — flat, but reclaimable (the kernel evicted it on demand instead of thrashing). That is the exact contrast with the pinned path at similar cgroup levels.
- **Prefill within the baseline band.** U1's 45.8 t/s sits just under the 2026-09-20 warm band (48–53 t/s) — consistent with a post-reboot, partially cold expert page cache plus a freshly booted NAS VM on the same host.
- **Decode below the warm baseline** (8.4–11.1 vs 14.3–14.9 t/s): same page-cache-state explanation; flags are identical to the adopted cell, so this is state, not config. Worth re-measuring once the working set is fully warm.
- **Larger ub is counterproductive on the unpinned path here.** ub 1024 does 26.1 t/s vs 45.8 for ub 512 — the opposite of the [2026-08-28 ub sweep](2026-08-28-llama-cpp-ubatch-moe-single-gpu.md) (different model, `--no-mmap`, MTP). On a 4-core i3 feeding unpinned mmap expert uploads, bigger upload batches stall the pipeline instead of amortizing it.

## Conclusion

- The pinned fast path (`GGML_CUDA_REGISTER_HOST=1`) is **unsafe on this host** at any meaningful context: one prefill page-locks ~40 GB against a 40 GiB cgroup limit on a no-swap 46 GiB box. It needs a bigger-RAM machine.
- The unpinned path is **stable and already near-optimal at the prod setting**: ub 512 beats larger ub here, so there is no ub-sweep gain to harvest on this box.
- Follow-ups if full numbers are wanted: (a) a 64K unpinned soak (~25 min per rep at ~45 t/s) for the sustained-PP question; (b) decode re-measurement with a warm expert page cache; (c) the pinned matrix on a bigger-RAM host.
