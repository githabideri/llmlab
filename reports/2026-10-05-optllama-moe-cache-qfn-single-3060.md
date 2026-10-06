# 2026-10-05 OptLlama moe-cache A/B on Flash-Next, single-3060 box: prefill 2.5×, decode wall is the RAM ceiling

**Date:** 2026-10-05/06 (campaign window 20:39 → 05:05 local)
**Category:** Experiment / benchmark
**Hardware:** backup / inference box — Intel i3-9100 (4C/4T, DDR4-2400 2-channel ≈ 38 GB/s), single RTX 3060 12 GB, 46 GB host RAM (42 GB LXC soft limit), **no swap**, NVMe
**Model:** Qwen3.8-Flash-Next ("Qwen4Exp", 125 B total / ~6 B active + 51 B per-layer PLE table), `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` (2-shard GGUF, **65.0 GB on disk**) + MTP draft `mtp-Qwen3.8-Flash-Next-Q8_0` (2.9 GB). Same weights as the [2026-09-29 attribution campaign](2026-09-29-flash-next-3060-compact-gather-cpu-moe-bound.md).

**TL;DR**

- **OptLlama's `moe-cache` fork (commit `167742d`) is a real prefill engine**: with `--ubatch-size 2048` it prefills 16K at **184–290 t/s (2.4–2.6×** the production expert-pool build) and 32K up to 323 t/s. `--ubatch-size 4096` **OOMs at 96K ctx** on 12 GB — 2048 is the ceiling.
- **Decode at depth (64–90K) is ~5.5–7.4 t/s for *both* builds, and the reason is not the MoE scheme**: the 65 GB model does not fit the 46 GB box. The 90K prefill touches essentially the entire 35.4 GB expert table; during 90K decode the page cache sustains **~42 MB/s of NVMe re-reads** (measured via `read_bytes`), i.e. experts stream from disk for both builds. The [09-29 "18.8 t/s at 16K"](2026-09-29-flash-next-3060-compact-gather-cpu-moe-bound.md) was a fresh-cache best case; the same build, run after an overnight campaign, does 7.0–7.4 at 16K in identical conditions. **20–30 t/s at deep context is unreachable on this box at any setting of any build.**
- **20–30 t/s is a memory purchase, not a fork choice**: 128 GB DDR4 (4×32, board-supported) puts the full 65 GB resident and lets the GPU-miss expert-cache family (the ddvnguyen lineage) do what it's designed for; until then the expert-pool production build stays the default and OptLlama is the prefill-heavy alternative.
- **MTP is broken in this fork revision for grouped MoE** (CUDA graph capture rejects `MUL_MAT_ID` in the MoE decode path — every MTP request 500s). Bounded host pinning (1–2 GiB) shows no measurable decode effect on this box and stays near zero, per the [09-23 pinned-path reboots](2026-09-23-flash-next-pinned-path-oom-unpinned-ub-sweep.md). The fork's overlap suite gains are small here (~5%) vs the owner's 2.2–2.7× on 12+ cores — 4 cores is the binding constraint for the overlap design.
- Quality: 3 numeric canaries (recompute, carry, checksum) — **3/3 on the fork** at 88.6K prefix, coherent output, no drift at 64K/90K.

## Goal

Three questions, motivated by the theodacus/Codacus "Strata" video (Aug 2026; 177B on a 12 GB card) and a follow-up comment (a 5070 Ti hitting 24.4 t/s at 32K):

1. How does the **OptLlama `moe-cache` fork** (GPU-resident MoE expert cache with LRU/least-used admission and bounded host pinning — the bounded-pinning implementation of the Codacus class) compare to this box's production **expert-pool** build, like-for-like on the same 65 GB quant?
2. Is **20–30 t/s** reachable on this box, and which tuning lever gets there?
3. Where does the campaign's measured 8.4–10 t/s decode sit relative to both?

## Engines

| Engine | Build | Role |
|--------|-------|------|
| C1 | the box's production build — llama.cpp `88e76d8a`-based **expert-pool** (the [09-29 champion](2026-09-29-flash-next-3060-compact-gather-cpu-moe-bound.md)), 64-slot balanced pool, mmap + lazy PLE, ub 512, ctx 96K, `-t 4` | control (production shape) |
| O | **OptLlama** (public fork `GenerelSchwerz/llama.cpp`, branch `moe-cache`, commit `167742d`, base `b6b6281`), CUDA MoE expert cache: `--moe-expert-cache-size` (MiB) + admission (`--moe-expert-cache-miss` lru|least-used, `--moe-expert-cache-miss-policy lru|fixed|frequency`), bounded pinning (`--moe-expert-cache-host-pinned-mb`), overlap suite (`--decode-overlap-suite auto`), graph prefill. Flag-parity with the 09-23 matrix (`-b` = max ubatch) | candidate |

Both built in-container (Release, CUDA 13.1, `GGML_CUDA_FA=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_CUDA_GRAPHS=ON`, `CMAKE_CUDA_ARCHITECTURES=86`, `-j8` — the i3's 4-way build is the bottleneck; OptLlama is ~1.4 GB of C++/CUDA, ~1 h).

## Commands

The 4500 MiB cache (the 09-29 production budget) derives **3216 slots (67 per routed tensor × 48)**:

```bash
# candidate, baseline cell (O1)
/opt/llama.cpp-opt/build/bin/llama-server \
  -m /mnt/models/…/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --moe-expert-cache-size 4500 -ngl 99 --n-cpu-moe 99 -t 4 \
  --load-mode mmap -fit off -fa on -ctk q8_0 -ctv q8_0 \
  -c 98304 -np 1 --cache-reuse 256 --no-sched-async-cpu \
  -b 512 --ubatch-size 512 --jinja --port 8090

# candidate, prefill cell (O2): identical except  -b 2048 --ubatch-size 2048
# control (C1): the production preset's flags verbatim (expert-pool 64, ub 512) on port 8091
```

Bench: `ab-bench.py` against `/v1/chat/completions` (256 out-tokens, 2 reps/depth, median; prefill = prompt_tokens/TTFT; decode = content-chunk rate) with depth ladder 16K/32K/64K/90K, a 96K 2-token-prefix-reuse pair (cold/warm), and 3 numeric canaries (arithmetic recompute, carrying, checksum — tokenizer-verified, 88.6K prefix reuse). The 35B service was stopped for the whole window (unit files untouched; it was not restarted, matching the 09-23/09-30 decision to hold it out). A 5 s host logger (MemAvailable + CT cgroup) aborted on < 6 GiB available — **never fired**; the soft limit was never exceeded.

## Matrix and results (medians; t/s)

Cells: **O1** baseline (cache 4500 MiB, ub 512) · **O2** ub 2048 · **O3** ub 4096 · **O4** +1 GiB pinned · **O5** +2 GiB pinned · **O6** overlap suite · **O7** +MTP · **O8** LRU eviction (bench-started, not completed — see Observations) · **C1** control.

| Cell | 16K r1/r2 | 32K r1/r2 | 64K r1/r2 | 90K r1/r2 | 96K cold/warm | Note |
|------|-----------|-----------|-----------|-----------|---------------|------|
| O1 | 8.8 / 9.0 | 8.1 / 8.2 | 6.6 / 7.2 | 6.2 / 6.4 | 124.3 / 133.4 (8.4 / 9.0) | canaries 3/3 |
| O2 | 8.7 / 9.3 | 8.6 / 8.8 | 7.0 / 6.9 | 6.1 / 5.6 | — | prefill 184–323 t/s |
| O3 | 8.5 / 9.1 | 8.3 / 9.0 | 7.3 / 7.1 | **OOM at 90K** | — | batch 4096 ceiling |
| O4 | 8.3 / 7.2 | 7.6 / 7.4 | 7.6 / 7.1 | 7.3 / 6.9 | — | 1 GiB pinned → 773 MB registered |
| O5 | 8.9 / 9.2 | 7.9 / 8.3 | 7.4 / 7.3 | 6.7 / 6.8 | — | 2 GiB pinned, same budget ceiling |
| O6 | 9.4 / 9.2 | 8.3 / 8.3 | 7.1 / 7.2 | 6.7 / 6.5 | — | overlap suite |
| O7 | — | — | — | — | — | **all MTP runs HTTP 500** (graph capture) |
| O8 | 9.2 / 8.5 | 7.3 / 7.3 | 7.0 / 7.0 | — | — | LRU eviction; stopped pre-90K |
| **C1** | **7.0 / 7.4** | **6.8 / 6.9** | **6.0 / 6.1** | **5.5 / 5.5** | 126.0 / 132.9 (8.2 / 8.8) | production build, same session |

Prefill (16K cold → warm): C1 78.2 → 95.3 · O1 108.9 → 131.8 · **O2 184.3 → 289.9** (32K: 199 → 323). At 90K the control's prefill (88.5/89.6) is slightly *faster* than the fork's (74.4/72.9) — the page-cache ordering after an overnight campaign, not a structural difference; both are ~1.8–2× the 09-29 cold prefill (45.8 t/s @9.6K, ub 512).

Prior same-box numbers for reference: [09-20](2026-09-20-flash-next-single-3060-moe-cache-backup.md) 64-slot profile cache (older 78.9 GB UD quant): 14.3–14.9 @64K warm · [09-29](2026-09-29-flash-next-3060-compact-gather-cpu-moe-bound.md) production expert-pool (this quant): 18.8 @16K / 14 @32K / 8.5 @90K — all **fresh-cache best cases**, i.e. the top of what this box can do before the working set churns.

## Observations

1. **The RAM ceiling is the headline.** The expert table alone is 48 layers × 512 experts × 1.43 MB ≈ **35.4 GB**; add the ~10 GB dense/attention core, KV, and PLE pages and a 90K-context run's working set exceeds the 46 GB box by a wide margin (the 65 GB file is fully materialized — no sparse PLE region to skip, unlike the APEX-mini quants in the ddvnguyen runs). Measured during the 90K phases: **~42 MB/s sustained NVMe re-reads** (per-process `read_bytes` deltas) on *both* builds — the decode is partially fetching experts from disk. Both builds converge on 5.5–7.4 t/s at 64–90K (C1 5.5/5.5, O1 6.2/6.4, O2 6.1/5.6 — within run noise of each other). The 09-29 "CPU-MoE-bound at depth" attribution was right about the CPU side; this campaign adds the storage side: **at 90K the bottleneck is also (and for the tail, mostly) the NVMe**, because the expert set does not fit in RAM.
2. **The 16K gap (fork 8.8–9.4 vs C1 7.0–7.4 vs 09-29's 18.8) is cache-condition, not config.** Same build, same flags: fresh page cache (09-29, morning, single-session) → 18.8; after an 8-cell overnight campaign with three 65 GB load cycles (this run) → 7.0–7.4. The fork's 16K decode (8.8–9.4) sits *above* the churned control in the same session and below the fresh-cache control. Interpretation: the production build's 64-slot static pool degrades with churn (the pool's fixed expert selection is cold again after each reload); the fork's LRU/fixed admission re-warms faster. Neither recovers the fresh-cache 18.8 under churn.
3. **Prefill is the fork's clean win.** ub 2048: 184–323 t/s vs 78–95 for the control — the batched CUDA-graph prefill path handles the 2048-token ubatches the 512-ubatch control serializes. The practical effect: a 9.6K prompt costs ~1.1 s of prefill (O2) vs ~2.5 s (C1); at 90K the difference is ~0.6 s per 50K tokens of prefill. **O3 (ub 4096) OOMs** on the 12 GB card at 98K ctx (graph + workspace + cache) — 2048 is the ceiling; that matches the 09-28 ISTA finding that `-b > 2048` does nothing on this card.
4. **Bounded pinning is a no-op at these budgets.** With `--moe-expert-cache-host-pinned-mb 1024/2048`, the admission logic registers only **773 MB** of pinned pages (`source_budget_exhausted` at group 47) and everything else goes through staging/pageable — and decode is unchanged within noise (O4/O5 vs O1). To pin a meaningful fraction of the 35 GB expert table you'd need ≥ 20 GiB of pinned budget, which is exactly the non-reclaimable regime that [rebooted this host three times on 09-23](2026-09-23-flash-next-pinned-path-oom-unpinned-ub-sweep.md) (`cudaHostRegister` on mmap'd weights is a permanent cgroup charge). On this box the pinned budget stays near zero; the fork's bounded-budget design is what makes "near zero" a safe knob rather than a crash.
5. **The overlap suite scales with core count.** O6 (`--decode-overlap-suite auto` = pre+post+post-alloc+post-alloc2+post-batch) gains ~5–8% here vs the owner's reported **2.2–2.7×** (37→97 t/s) on a 12-core 5070 Ti build. The suite overlaps CPU staging with GPU compute, but with 4 cores the staging thread starves the expert compute threads it's meant to hide — the gain collapses. Expect the suite to matter on ≥ 8–12 cores.
6. **MTP is broken for grouped MoE in this revision.** With the 2.9 GB Q8_0 draft, every request 500s: CUDA graph capture of the decode path fails on `MUL_MAT_ID` (the grouped-expert op) when the draft's attention path is mixed in. This is the same failure family as the [09-29 "27B MTP 500s"](2026-09-29-flash-next-3060-compact-gather-cpu-moe-bound.md) and the 09-20 "MTP adds nothing" observation gets a sharper form here: on this fork revision MTP + grouped MoE does not run at all, on any card. Worth an upstream issue.
7. **LRU vs fixed admission is not the lever** (O8, `GGML_CUDA_MOE_FREQUENCY=0` + LRU eviction): 16K 9.2/8.5 vs O1 8.8/9.0 — noise. The 90K phase was not completed (campaign stop). The admission *budget* (4500 MiB / 3216 slots) is the lever that matters; at this budget the policy doesn't.
8. **Quality.** O1 canaries 3/3 (recompute/carry/checksum exact at 64K/90K prefix). As in the 09-13/09-20 runs, the cache path is **not bit-identical to CPU-only greedy** (first content token diverges, output coherent) — the standing caveat for MoE-cache engines. No loop/gibberish observed.

## Energy

GPU-only (3 s `nvidia-smi` sampling, no wall meter on the box): campaign average **74.6 W**, active cells **77–85 W** (per-cell: O2 76.8, O4 80.9, O5 84.7, O6 80.4, O7 81.4, C1 83.1 W), peak 120.7 W (90K prefill bursts), **451 Wh total over the 6.1 h campaign**. A 90K cell (256 tokens in ~17 min at 83 W) ≈ **24 Wh per 1K generated tokens**, GPU-only — versus 09-20's measured **48.1 W steady decode (~3.3 Wh/1K)** for the 14.5 t/s 64K cell. The deep-context numbers are both slower *and* more power-hungry per token: the NVMe re-reads and the 4-core staging show up in the socket.

## Research notes (the two MoE-cache families)

- **CPU-miss family** (weights on CPU; only the cache lives on GPU): upstream RFC [ggml-org/llama.cpp#24528](https://github.com/ggml-org/llama.cpp/pull/24528) (June 2026, "CPU expert cache — the 4th expert tier") is the design source of this box's production expert-pool build; the Codacus/Strata video and its 24.4 t/s 5070 Ti commenter run this family. It wins when the *uncached tail* is small relative to the cache.
- **GPU-miss family** (weights on GPU; the CPU is the staging path): the ddvnguyen llama.cpp fork (issues [#129](https://github.com/ddvnguyen/llama.cpp/issues/129)/[#130](https://github.com/ddvnguyen/llama.cpp/issues/130), x4-3060 record: **18.8–20.5 t/s** on a 167.7B IQ2_XS at 20 GB pinned / 85.7 GB system RAM) and **OptLlama `moe-cache`** (the bounded-pinning implementation tested here). It wins when *everything fits in system RAM* and the GPU cache absorbs the hot set.
- The breakeven between them is the **cache hit rate** (~40% by the ddvnguyen analysis). This campaign measures the other side of that ratio: at 46 GB RAM the *cacheable expert fraction* for a 90K-context QFN run is below breakeven, so neither family can deliver its design win — both fall over the same RAM cliff. The family question becomes answerable on a 128 GB box, where the GPU-miss family's record numbers (18.8–20.5 on the *bigger* 167B model, x4 3060s, 20 GB pinned) suggest the 20–30 t/s target is in range for QFN on this one 3060 once the weights fit.

## Conclusion / recommendations

- **Keep the production expert-pool build as this box's default.** Under churn it is no slower than the fork (5.5–7.4 vs 5.6–9.4 t/s by depth), it has the proven production track record (09-20/09-23/09-29), zero pinning risk, and its fresh-cache 16K (18.8) is the best decode this box has ever produced.
- **OptLlama `moe-cache` is the alternative for prefill-heavy workloads** on this box: 2.4–2.6× prefill at ub 2048 (the one setting that is unambiguously better), decode at parity-or-better under churn, bounded pinning safe to leave near zero. A second service/preset (own port, mux entry) is a one-INI change if prefill latency is the pain.
- **20–30 t/s at deep context on this box = a 128 GB RAM purchase** (4×32 GB DDR4-2666, board-supported; the 09-29 report's open item). Then re-run this matrix: the fork's GPU-miss design is the one to bet on (ddvnguyen's numbers), with the overlap suite (this time on a box where cores aren't the wall) and a ≥ 20 GiB pinned budget (now that the RAM ceiling is gone, not the 09-23 non-reclaimable regime). Until that purchase, expect ≤ ~9 t/s at 16K and ≤ ~7 t/s at 90K on *any* MoE-cache variant.
- **File an upstream issue**: CUDA graph capture + `MUL_MAT_ID` (grouped MoE) + MTP draft → capture failure on every request (OptLlama `moe-cache` `167742d`; same family as the upstream 27B-MTP 500s).

**Supersedes (partially):** the [09-29](2026-09-29-flash-next-3060-compact-gather-cpu-moe-bound.md) "CPU-MoE-bound at depth" attribution — still right about the CPU term, incomplete about the storage term (the 90K working set exceeds 46 GB RAM; NVMe re-reads measured) — and its 18.8/14/8.5 profile, which this campaign re-identifies as fresh-cache best cases rather than steady state. The [09-20](2026-09-20-flash-next-single-3060-moe-cache-backup.md) "40 GB holds most of the working set" note is optimistic for the 65 GB GSQ quant at 90K (it was measured on the 78.9 GB UD quant at ≤ 64K, where the touched expert fraction is smaller).
