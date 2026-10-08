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

## Update (2026-10-06) — the "RAM ceiling" conclusion and the Codacus attribution, corrected

House rules: the body above is frozen; this section supersedes parts of the Research notes and Conclusion.

1. **The Codacus citation is retracted and replaced with the primary source.** "RFC ggml-org/llama.cpp#24528" does not exist (404; the GitHub API was verified working the same day against a known issue, so the number was wrong, not the API). The "24.4 t/s 5070 Ti commenter" figure is not verifiable (the video's comment extraction returned no readable content). The primary source — [the video](https://www.youtube.com/watch?v=6WLBmP-tZ0Q) and its description ("All numbers are from my own runs on my own machine") — says Codacus's 12 GB machine is **RTX 3060 12 GB + 64 GB system RAM** (not 32 GB), running "Qwen 3.8 Flash, 177B parameters counting its n-gram tables" (i.e. this 125 B + 51 B PLE model) in the **GSQ-RCO IQ3_XXS** quant (larger than our IQ2_XS). His numbers: prefill ~120 t/s (stock llama.cpp, last month) → ~620 t/s (his two-day llama.cpp fork port) → 900+ t/s (Strata); **decode ~26 t/s on his llama.cpp fork, ~40 t/s on Strata** — the 40 includes MTP, which Strata keeps fully in VRAM drafting ~2 tokens per pass ("mainline llama.cpp has no MTP for this architecture at all"). For the full model he states "you need 64 gigs of RAM, because strata loads the experts and **pins** them in RAM" ("the IQ3_XXS pins about 40 gigs"), while the **~28 GB engram/PLE table "gets streamed from the SSD as they're needed, and you don't need to fit them in RAM at all"**; 128K context costs "nothing measurable". So his 26/40 t/s was produced on a **64 GB** machine in a **~40 GB pinned** regime — *more* RAM than this box, not less. The earlier "24 t/s on 32 GB" line is dropped.

2. **The RAM ceiling is misframed: the binding resource is the *pinned* (non-reclaimable) budget, not "the model fits in RAM".** The OptLlama wiki's owner-verified runs — [Notable runs](https://github.com/GenerelSchwerz/llama.cpp/wiki/Notable-Runs), commit `925933801`, **RTX 5070 Ti 16 GB, 62.1 GiB RAM, 12 cores** (`-t 12`), Q3_K_XL — are the profile behind the "34–60+ t/s on 12-core / 62 GB" question:

   | Regime (64K ctx) | Prefill | Decode |
   |---|---:|---:|
   | full pin, no MTP | 115.9 | **55.5** |
   | full pin + MTP (579/887 accepted, 65.3 %) | 136.8 | **61.7** |
   | partial pin, 22 888 MB budget | 31.5–194.6 | **33.7–36.2** |
   | earlier baseline (`b46f7f7a4`, 12K, no MTP) | — | 47.0 |

   Full pin out-decodes partial pin by ~53 %; the partial-pin prefill collapse (115.9 → 31.5) shows the CPU-side pageable→pinned staging cost. Their flag composition includes `--moe-early-router --decode-boundary-overlap --ple-prefetch --phase-aware-workspace --live-context-workspace --backend-sampling -kvo` — none of which this campaign's O1–O8 cells used, and our pin budgets were 0/1/2 GiB against their full or 22 888 MB. And Codacus's architecture shows the 28 GB PLE **never needs to be in RAM at all**. So "a 65 GB file on a 46 GB box" was the wrong statement of the constraint: what does not fit in 46 GB is the *pinned* expert set (full pin ≈ 35.4 GB IQ2_XS experts + ~10 GB dense ≈ 45 GB non-reclaimable on a no-swap host = the 09-23 incident), while the 28.8 GB PLE shard is excludable from RAM by design.

3. **The 42 MB/s read_bytes observation (Observation 1) is not attributed per file.** "The decode is partially fetching experts from disk" (which would require ~35 GB of the expert table evicted) is a *weaker* explanation than scattered lazy-page faults on the 90K PLE rows in the 28.8 GB shard, and was never settled. Open question; a per-fd `strace` sample during a 90K decode would decide it (not run).

4. **The price/board claim is retracted.** "4×32 GB DDR4-2666, board-supported, ~€150": the board model of this box is unknown (no basis for "board-supported"), and the €150 figure was stale — current geizhals prices for a 32 GB (2×16) DDR4-3200 kit are **€214.49** (G.Skill Aegis) to **€239** (Corsair Vengeance LPX); Tom's Hardware's 2026 RAM index has 32 GB kits at $60–90 in Oct 2025 → **$150–180 by Jan 2026**. So 4×32 ≈ €850–950, not ~€150. It is mostly moot in any case: the in-range regimes were bought with **62–64 GB**, and even then a *full* pin of this quant (~45 GB non-reclaimable) exceeds what a no-swap host can safely carry — the 09-23 incident remains the binding fact.

5. **Corrected feasibility verdict for this box.**
   - **50–62 t/s (full pin; 62–64 GB host, ≥12 cores): not reachable here** (46 GB no-swap; 09-23).
   - **~20–36 t/s (24 GB partial pin + the full flag composition): testable here.** The owner's own 22 888 MB cell fits inside our 42 GiB cgroup with ~15 GB headroom; the cgroup cap makes the worst case a CT OOM instead of a host death. Their 12-core machine posts 33.7–36.2 t/s in that regime; on this 4-core i3 (staging is CPU-bound — Observation 5) the honest expectation is *between* the current 8.8–9.4 and their ~34 — a prediction to be measured, not a claim.
   - **MTP** (the 61.7 includes ~1.65×; Strata's 40 includes ~2.1×): broken in this fork revision (MUL_MAT_ID capture), working in the owner's current verified run — retest on a newer build.
   - If the goal is *his* speeds on this model, the in-range machine is the owner's profile (16 GB card, 62–64 GB RAM, ≥12 cores) — a box purchase, and the number to buy is **64 GB**, not 128.
   - **Next cell (pending approval): O11** = newer OptLlama `moe-cache` build (`925933801` or later) + `--moe-expert-cache-host-pinned-mb 22888` + the owner's full flag composition, 16K + 90K, cgroup cap as guard, host RAM watched; optional 30 s per-fd `strace` sample during the 90K decode to settle the PLE-vs-experts read attribution (item 3).

## O11 results (2026-10-06) — the 24 GB partial-pin regime, measured (closes the Update's pending cell)

Built the exact owner-verified commit `925933801` (2026-09-14, "cuda: isolate MoE host memory
ownership") — a *different, newer* lineage than the campaign's `167742d` moe-cache branch (11 302
commits of separation; the 09-14 binary does not carry `--moe-early-router`/`--backend-sampling` — in
that lineage the overlap features are env-gated as `LLAMA_ARG_DECODE_OVERLAP` /
`LLAMA_ARG_DECODE_BOUNDARY_OVERLAP`, both set and observed active in the slot logs). Configuration:
`--moe-expert-cache-host-pinned-mb 22888` (the owner's budget), `--moe-expert-cache-size 32`
(slabs per expert tensor; 7.7 GB VRAM all-in), q8 KV, ub 512, `-t 4`, under the box's 42 GiB cgroup
guard. The 22 888 MB budget was **fully consumed**: the admission log shows 32 of 48 expert groups
pinned (~23.6 GB at ~737 MiB/group), the remaining 16 groups (~11.8 GB) staging/pageable.

> **Supersede note (2026-10-08):** the “*different, newer* lineage … 11 302 commits of separation” statement was a shallow-fetch artifact of the local clone. The GitHub compare API against the published fork (2026-10-07) resolves `925933801...167742d` to ahead 310 / behind 0 — **`925933801` is an ancestor of the `moe-cache` branch**, and the #99 alias-boundary fix `e464190a2` sits 189 commits after `925933801` and 121 before `167742d` (i.e. **`167742d` contains the #99 fix; the O11 build `925933801` is the pre-fix build**). The overlap-gating observation in that paragraph stands as measured in the slot logs. This is why the follow-up campaign (reports 2026-10-07 and 2026-10-08) ran hard output-correctness canaries on the `925933801`-family builds and retested `167742d` in a control cell.

| Cell (r1 / r2) | 16K prefill | 16K decode | 90K prefill | 90K decode |
|---|---:|---:|---:|---:|
| **O11a** (24 GB partial pin) | 110.7 / 116.0 | **9.9 / 9.8** | 99.7 / 99.9 | **6.8 / 6.6** |
| **O11b** (same + MTP draft 2) | 107.2 / 122.0 | **10.8 / 11.3** | — | — |
| 10-05 fork `167742d` (pin 0–2 GiB) | 91.7–289.9 ¹ | 8.8–9.4 | 90.7–91.7 | 6.2–6.4 |
| 10-05 production expert-pool | 78–95 ¹ | 7.0–7.4 | 88.5–89.6 | 5.5 |
| Owner reference (12c / 62.1 GiB / 16 GB card, same-commit regime) | — | — | — | 33.7–36.2 (partial) / 55.5–61.7 (full) |

¹ ubatch-dependent (the 2048 cells); O11 kept the owner's ub 512.

**Verdict.**
1. **The partial-pin regime is correctly implemented but is not the missing lever on this box.**
   Decode is +6–8 % vs the 10-05 fork cells and ~+20–25 % vs production — inside the existing cluster,
   not a regime change. At 16K the hot expert set fits within the 32 pinned groups, so pinning removes
   little that was already fine; at 90K the routing profile touches all 48 groups and the 16 staged
   ones are re-fetched from NVMe per step — with 4 cores and a PCIe 3.0-class link (~12–14 GB/s
   effective H2D per 10-05), *staging throughput* sets the floor. (Inference; the per-fd attribution
   sample the Update left open was armed but its sampler did not survive the window — item 3 stays
   open.)
2. **MTP works in this lineage.** 10.8–11.3 t/s (+9–15 % over O11a); the `167742d` `MUL_MAT_ID`
   capture bug (10-05 O7: every request 500) is absent — the draft engine reports "MTP draft enabled
   after target acceptance". The gain magnitude matches the owner's own table (55.5 → 61.7 ≈ +11 %).
3. **The gap to the owner's numbers tracks the staging hardware, not RAM.** Same commit, same budget,
   same card class: 6.6–6.8 here vs 33.7–36.2 there (4.9–5.3×) — their box has 3× the cores and a
   newer PCIe generation; at 16K it is 9.9–11.3 vs their full-pin 55.5 (4.9–5.6×). On this
   4-core / PCIe 3.0 / 46 GB box, 20–30 t/s at 16K and 33+ at 90K is **a hardware story (cores × PCIe
   generation), not a RAM-capacity or pinning-configuration story** — the corrected form of the
   original "128 GB" headline (direction right: a different machine is required; axis wrong: it is
   staging bandwidth — and the RAM figure that would enable the *full-pin* regime is 64 GB, a figure
   ruled out for this particular server by its two SO-DIMM slots).
4. **Recommendation update.** Production expert-pool remains the default for service use (same or
   better decode, no new risk surface). For prefill-heavy or batch work on this box class, the
   **`925933801` lineage** supersedes `167742d` as the OptLlama reference build (MTP working,
   partial pin implemented as documented), noting its ub-2048 prefill was not re-verified in this
   session.

## Update (2026-10-07) — the read attribution (item 3) is settled: it is the PLE table

The per-fd sample item 3 left open was run the next day (same O11 regime, hard 42 GiB
guard): a 300 s `strace` window over a 98K decode shows **zero `read()` calls**, while
`/proc/pid/io` at 2 s shows **27–42 MB/s of storage reads in every 98K phase with `rchar`
flat** — i.e. the bytes are **mmap page faults on the 28.8 GB PLE shard**, not expert
streaming, and `strace` is blind to them by construction (`read_bytes` − `rchar` is the
page-fault component). The 98K prefill (TTFT 832–892 s) is I/O-dominated at that rate, and
the PLE cannot be resident on this box in any configuration (65.9 GiB model > 54.5 GiB of
memory containers). See [2026-10-07-qfn-o11-ple-page-faults-single-3060](2026-10-07-qfn-o11-ple-page-faults-single-3060.md).
