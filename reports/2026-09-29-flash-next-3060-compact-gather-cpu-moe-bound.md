# 2026-09-29 Flash-Next on the single-3060 box: decode at depth is CPU-MoE-bound — compact-gather is correct but inert here

Companion to [`2026-09-28-flash-next-sparse-qsa-f16-kv-single-3060.md`](2026-09-28-flash-next-sparse-qsa-f16-kv-single-3060.md). Same machine, same model (ISTA-DASLab `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS`, IQ2_XS), same placement (balanced expert-pool 64, `budget 4500 MB`, `-ngl 99 -ncmoe 99 -fit off -t 4`, single slot, no LCP, `lazy-mode on`). This campaign re-adjudicates the 09-28 f16 finding and answers the attribution question: **what actually causes the decode depth-decay on this box, and does the compact-gather decode path (the 09-28 candidate) fix it?**

**TL;DR**

- The 09-28 "+28% f16 KV" at 16K was a **cold-vs-warm artifact**. Cold-vs-cold: q8 14.3 vs f16 15.1 t/s (~6%, within run noise). A real but small f16 edge appears at 64K (12.6 vs ~10).
- **Mixed KV is harmful**: K=f16/V=q8 collapses to 9.0/6.8/4.6 t/s at 16/32/64K. V=f16 doesn't even load (12 GB). Use all-q8 or all-f16.
- **The QSA indexer is ~0% of decode cost.** Ablating its entire score+top-k pass (fixed selection, identical FA work) changes nothing at 16K or 90K.
- **Compact-gather decode is functionally correct (3/3 canary retrieval at every depth) but gives no speedup on this machine** (90K: 8.1 t/s gathered vs 8.3–9.1 masked). Eliminating ~97% of the per-token K/V+mask traffic moves nothing — **decode at depth is CPU-MoE-bound**, not attention-bound. The GPU is off the critical path.
- The upstream compact-gather branch ([`4d6ef5a`](https://github.com/ggml-org/llama.cpp/pull/28734), 2× at 225K on a 5×3090) does not reproduce its gain here — the machine class, not the algorithm, decides whether this lever matters.
- Ubatch (256→1024) is a wash for decode (8.5–9.5 at 90K); it only moves prefill (65→109 t/s).
- Net: the depth decay (16K 18.8 → 90K ~8.5) is **context-dependent CPU expert-pool behaviour** (hit-rate 86.4% → 82.0%, measured 09-27), not anything the attention path can fix. On this box, "long controller" = plain expert-pool + q8_0; compact-gather is the right component for GPU-bound cards (3090-class), not this one.

## Setup

| | |
|---|---|
| host | backup server, i3-9100 (4C/4T, DDR4-2400), RTX 3060 12 GB, 42 GB LXC RAM |
| model | ISTA-DASLab GSQ-RCO IQ2_XS (65 GB, 2 shard), native ctx 262144 |
| placement | `-ngl 99 -ncmoe 99 -fit off -mec 64` (balanced expert-pool), pool budget 4500 MB, 1 GiB rail |
| KV | q8_0 both sides (controls); f16 / mixed variants as noted |
| context | 98304; tests at 16K/32K/64K/90K depth, 1 slot, `-sps 0`, `--no-cache-idle-slots`, `-t 4` |
| canary | 3 retrieval codes embedded at 10/50/90% of the prompt; model is a thinking model — answers live in `reasoning_content`, `max_tokens 600` |

Three build trees: **expertpool** (known-good, 09-27), **idx** (expertpool + ablation + compact-gather env-gated patches), **compact** (upstream `4d6ef5a` on newer master, built for reference).

## Results (t/s, decode; pp = prompt-processing)

### Controls, n=3 (q8_0, ub 512)

| depth | v1 (cold) | v2 (warm) | v3 (warm) | canary |
|---|---|---|---|---|
| 16K | 14.3 | 18.8 | 18.6 | 3/3 |
| 64K | 11.4 | 10.0 | 10.1 | 3/3 |
| 90K | 6.2 | 9.1 | 8.3 | 3/3 |

### KV type (v1; f16 can't run 90K on 12 GB — OOM)

| depth | q8_0 (v1) | f16 | K=f16/V=q8 |
|---|---|---|---|
| 16K | 14.3 | 15.1 | **9.0** |
| 32K | — | 14.0 | **6.8** |
| 64K | 11.4 | **12.6** (3/3) | **4.6** (3/3) |

### Ubatch (q8_0, v1/cold)

| depth | ub 256 | ub 512 | ub 768 | ub 1024 |
|---|---|---|---|---|
| 16K | 12.1 | 14.5 | 15.3 | 14.5 |
| 64K | 10.2 | 11.2 | 10.2 | 10.9 |
| 90K | 9.3 | 8.7 | 9.5 | 8.5 |
| 90K pp | 78.9 | 87.1 | 107.2 | 109.3 |

### Attribution (idx tree; canary 3/3 on every run)

| run | change | 16K | 64K | 90K |
|---|---|---|---|---|
| R1 | QSA indexer ablated (score+top-k removed; fixed selection, identical FA work) | 14.2 | — | 8.5 |
| R2 | compact-gather (gather the 2051 selected K/V rows + their visibility values into 2304-row f16 buffers, padded to the MMA stride; dense FA on 2304 instead of masked dense over 90K) | 15.0 | 11.4 | 8.1 |
| CTRL | (baseline, above) | 14.3–18.8 | 10.0–11.4 | 6.2–9.1 |

R1 ≈ R2 ≈ CTRL at every depth: **neither the indexer nor the attention body is on the critical path.** The gather port is the per-cell top-k (2051) of this model's QSA; the upstream branch uses block-level top-k (128-wide blocks + `extra_cells`) on a newer master, but the essential mechanism — dequantize only the selected rows via `get_rows`, mask only those, run dense FA on the compact buffer — is the same. Our port adds a per-cell gathered visibility mask because in this base the selected set can contain invisible cells (the newer base's block selection cannot).

## Why it's CPU-bound

Per decode token at 90K: 12 QSA layers read 90K × (256 B K + 256 B V) ≈ 46 MB of KV + 180 KB of mask from VRAM. The 3060's 360 GB/s makes that ~0.15 ms — and R2 proved it: cutting that traffic ~97× (2304 rows) changed nothing. What remains per token is the 48-layer MoE: ~192 expert matvecs in IQ2_XS across 4 CPU cores at DDR4-2400, of which the resident 64-expert pool on the GPU covers the hot experts and the rest fall to CPU. Measured 09-27: pool hit-rate drops from 86.4% (shallow) to 82.0% at 90K — context widens the active-expert set, more experts fall to the slower path, decode decays. 18.8 → 8.5 t/s from 16K to 90K tracks that, and nothing we changed in the attention path moved it. (On a 3090-class box with faster or more cores, or a larger GPU pool, the same attention optimizations become the binding lever — hence the upstream 2× on 5×3090.)

## Cold/warm method note

09-28 compared a *cold* f16 first-request against a *warm* q8 baseline and reported +28%. This campaign's n=3 controls make the artifact visible: q8 itself runs 14.3 cold vs 18.8 warm at 16K (+32%). Any single-run A/B across a server reload is unreliable on this machine; compare cold-to-cold or warm-to-warm.

## PLE / page-cache (hetero context)

The "90K" hetero prompt (mixed prose/code/JSON/logs/tables) — note the truncation region tokenizes at ~2.7 chars/token, so the "90K" file is actually **74,749 tokens** (fits the 98304 ctx with headroom); the earlier 261K-token full file exceeded the ctx and failed instantly. Runs on the plain expert-pool binary, 98304 ctx, q8_0, pool 4500 MB: 2× cold (fresh server each), 1 "warm" (3rd request, same server), and a growing-context sequence.

| run | prompt (tokens) | pp | tg | gen_n |
|---|---|---|---|---|
| C1 cold (page cache cold) | 74,749 | 95.0 | 6.7 | 16 |
| C2 cold (fresh server, cache warm) | 74,749 | 95.4 | 6.9 | 16 |
| C3 "warm" (same server, identical prefix) | **4 processed** (74,745 served from slot KV) | — | 6.0 | 16 |
| I1 growing | 13,091 | 74.9 | — | 2 |
| I2 growing | 20,583 | 80.5 | — | 1 |
| I3 growing | 41,083 | 90.4 | 6.1 | 16 |

Findings:

- **No page-cache/PLE penalty on prefill**: fully-cold vs warm-page-cache fresh servers run 95.0 vs 95.4 t/s. (The 35B-class "137" issue on this LXC is not a PLE throughput effect; see memory below.)
- **Slot KV persistence works**: the third identical request processed 4 tokens — llama-server's single slot kept the 74,745-token prefix in VRAM across requests (even with `-sps 0`). Re-ingesting an unchanged context after a server restart is the expensive case; within a server's life it is free.
- **Memory is the real constraint**: peak RSS (VmHWM) plateaued at **39.0 GB across every phase** (93% of the 42 GB LXC limit, no swap) — dominated by the 65 GB model's mmap working set, not by context size (39 GB at both 13K and 75K prompts). Any overlapping large allocation (e.g. the 35B service loading at the same time) tips the cgroup OOM-killer — that is the `Killed`/137 recurrence mechanism seen twice during this campaign. Mitigation is operational: never overlap two large model loads in this LXC, and `posix_fadvise(DONTNEED)` the model shards between loads (`/proc/sys/vm/drop_caches` is not reachable from inside the LXC — `/proc/sys` is read-only).
- **Power**: 91–96 W during prefill, ~13–22 W idle/decode.
- The 1–16-token generations (immediate EOS on non-conversational hetero content) make tg values uninformative here by design; the pp/RSS/power columns are the point.

## Profiles (this box, IQ2_XS)

| profile | context | config | decode | notes |
|---|---|---|---|---|
| FAST | ≤16K | expert-pool 64 + q8_0, ub 512 | 18.6–18.8 (warm) | 3/3 canary |
| BALANCED | ≤32K | same | ~14 (09-27) | 4.2× the codacus baseline |
| LONG CONTROLLER | 90K | same | 8.3–9.1 | 3/3 canary; compact-gather optional (no gain here, correct output) |

Compact-gather and the indexer ablation are env-gated in the `idx` build (`LLAMA_QSA_GATHER_FA`, `LLAMA_ABLATE_INDEXER`); the gather port is the piece worth carrying upstream for GPU-bound cards — the upstream branch's block top-k plus this base's per-cell selection disagree on what "selected" means, so a merged version needs the gathered-mask handling.

> **Update (2026-10-05):** the 18.8/14/8.5 profile is re-identified as a *fresh-cache best case* — the 65 GB model's 90K-context working set exceeds this box's 46 GB RAM, so deep-context decode also streams experts from NVMe (~42 MB/s re-reads measured, both MoE-cache families converge at ~6 t/s there), and the same production build does 7.0–7.4 t/s at 16K after an overnight load campaign. The CPU-MoE attribution stands for the CPU term; the storage term is the missing half. See [2026-10-05-optllama-moe-cache-qfn-single-3060](2026-10-05-optllama-moe-cache-qfn-single-3060.md).
