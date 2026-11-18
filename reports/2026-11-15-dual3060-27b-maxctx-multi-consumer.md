# Qwen3.8-27B on a dual-RTX-3060: maximum context, balanced memory, and flexible multi-consumer serving

**Date:** 2026-11-15, 19:45-21:07 CET, and the overnight window 21:28-00:15 CET (crossing into 11-16) · **Status:** campaign complete — every stage measured, Pareto frontier mapped, three deployable profiles with exact launch lines, production restored and live-verified after each window · **Companion:** the [2026-11-14 qualification report](2026-11-14-fallback-dual-3060-27b.md) established that llama.cpp layer splitting is the only engine that works on this box and qualified the 256K/2 + MTP configuration; this campaign pushes that qualified baseline to its practical limits.

## Goal

The 11-14 campaign left the box with a working but lopsided memory layout: at the 256K aggregate / 2-slot configuration with MTP on, the CPU-direct x16 card kept ~1.5-2.7 GiB of headroom while the chipset-connected x4 card ran with 36-92 MiB free. This campaign asked four questions, in dependency order:

1. Can the layout be *balanced* — adequate allocatable headroom on both cards — without touching the total pool or the model?
2. What is the *largest stable aggregate pool* (288K, 320K?), and what is usable per session?
3. Can the KV pool be *flexible per consumer* — one session holding 200K while a second holds 32K, with correct isolation and reliable prefix reuse?
4. What do MTP draft-length, ubatch, thread count, and slot count actually do to decode/prefill once the layout is fixed?

The 128K×2 MTP server that production was running was the *floor*, not the target. Everything below was measured on the box, in authorized maintenance windows, with a detached out-of-band watchdog and a frozen, hash-verified baseline as the control.

## 1. The machine and the frozen baseline

Same machine as 11-14: Intel i5-7400 (4C/4T), 32 GB DDR4, two RTX 3060 12 GB — GPU 0 (LHR) on a CPU PCIe 3.0 x16 slot, GPU 1 (EVGA PX1) on a chipset x16 slot electrically running at x4 — no P2P. The NVIDIA driver moved from the 580.x series to 595.71.05 between the 11-14 qualification and this campaign; all numbers here are on the new driver, and all 11-14 numbers that re-measure the same configuration reproduced within noise.

Frozen and hash-verified before anything was changed:

- Engine: **llama.cpp b12383** (commit `c56be86`), the 11-14 winning build, unmodified.
- Model: **ISTA GSQ-RCO IQ3_S-mtp**, 12.1 GB, sha256 `58fd8267…` (matches the Hugging Face source).
- Projector: **the GSQ-RCO release's own BF16 mmproj**, sha256 `13cb7be…` — see the correction append to the 11-14 report: its §2 listed the *Unsloth* directory's projector hash (`83ee4f4f…`); the serving instance has used the GSQ-RCO one all along. Both are BF16 projectors for this model family; the one actually served is the latter.
- The production configuration: `--ctx-size 262144 --parallel 2`, layer split `1,1`, q4_0/q4_0 KV, MTP on (f16 draft), CPU projector, `-b 256 -ub 128 -t 4`, `--jinja --load-mode none`.

Every experimental configuration ran on a scratch loopback port; production was stopped only while a watchdog was armed and restored + live-verified at every window close (both times within the same hour).

## 2. Why one card was almost full (the allocation breakdown)

With the production 1,1 split, 256K/2 and MTP on, the two cards hold (GiB, from the build's own memory-breakdown tables):

| component | card 0 (x16) | card 1 (x4) | movable? |
|---|---|---|---|
| layer weights (IQ3_S mixed precision) | 4.75 | 6.08 — includes the **output projection, which the build hardcodes to the last GPU** | by layer ratio only |
| main KV (256K/2, q4_0) | 2.93 | 2.88 | follows the full-attention layer share (~even at 1,1) |
| MTP cluster: nextn head (0.33) + **draft KV (1.02, f16, one full-attention layer)** + draft compute (0.18) + recurrent share | — | **~2.4, almost entirely on the last GPU** | not by any flag in this build (§4, §5) |
| compute buffers | 1.20 | 1.19 + 0.18 | sized by the graph, not the ratio |
| mmproj (0.89, BF16) | CPU | CPU | auto-placement goes to the *first* GPU when left to the build (§3) |
| **used / free at peak** | **10.37 / 1.54** | **11.87 / 0.04** | |

Two facts explain everything that follows. First, the aggregate requirement is ~21.1 GiB of ~23.8 GiB: there is ~2.7 GiB of total headroom, so the problem is *placement*, not capacity. Second, the MTP cluster is pinned to the last GPU and the output projection is pinned to the last GPU too, so whatever the ratio, the second-listed card always carries the bigger, less-movable mass.

## 3. Phase A — placement sweep (8 configurations)

| id | change vs production | boot | free (card 0 / card 1) | 16K story decode |
|---|---|---|---|---|
| A0 | none (control) | ok | 1539 / 40 MiB | 24.8 t/s |
| A1 | tensor split 55,45 | ok | 525 / 1058 | shift only |
| A2 | tensor split 60,40 | boots (124 / 1978 idle), **OOM during the first 16K cell** | — | — |
| A3 | tensor split 65,35 | **OOM at boot**: the auto-placed mmproj (888 MiB) goes to the first GPU, which at 65,35 is full | — | — |
| A4/A4b | `--spec-draft-device 0` / `CUDA0` | arg error / accepted but **a byte-identical no-op** (the built-in MTP context is created from the main params; this flag governs a separate draft *model*) | — | — |
| A5 | tensor override moving the `blk.64.nextn.*` head to card 0 | ok | 1609 / 314 — recovers 274 MiB on card 1; the 1024 MiB draft KV does **not** follow the head | 42.4 (6-token sample) |
| A6 | ordinal flip (`CUDA_VISIBLE_DEVICES=1,0`) | ok — exact mirror: the tight card becomes the x16 one (40 / 1539) | — | proves the last-GPU pinning |
| A7 | MTP off (control) | ok | 2009 / 2022 — both cards ~2 GiB | 20.8 (6-token) — MTP is worth 2.1× decode here |
| **A8** | **flip + 60,40 + `--mmproj-device none`** | **ok** | **1978 / 1012 — both cards ≥ 1 GiB** | **24.82 (no penalty vs A0)** |
| A9 | flip + 55,45 + `--mmproj-device none` | ok | 1104 / 1886 | 24.22 |

Three conclusions. (1) The 60/40 "OOM wall" from the naive sweep is not the ratio — it is the mmproj auto-placement onto whichever card is first; forcing the projector to CPU unblocks the aggressive ratios. (2) Neither of the two levers that looked like they should move the MTP draft actually do in this build: `--spec-draft-device` is inert for the built-in MTP (measured byte-identical), and the tensor override moves only the ~40 MiB of head tensors that are not shared with the main model. (3) The ordinal flip is the one lever that moves the entire MTP cluster plus the output projection — and pointing it at the x16 card is exactly the direction the goal wants: **A8 is the memory-balanced placement, with zero measurable decode penalty** (24.82 vs 24.81 t/s on an identical 450-token cell; draft acceptance 0.40, retrieval intact).

PCIe was checked under load throughout: at 16K-class prefill the x4 card peaks around 0.3-0.9 GB/s (roughly a quarter of a gen3 x4 link), with both GPUs at 100% SM — the layout is compute-bound, and the x4 link is not the demonstrated constraint (A8 lowers the x4 bursts further by moving the MTP cluster to the x16 card).

## 4. Phase B — MTP draft-KV quantization: rejected with data

The draft KV (1024 MiB f16 at 256K/2) was the most movable-looking piece, so the obvious lever was quantizing it (`--spec-draft-type-k/-v`):

- At 256K/2 all three options (q8_0, q5_1, q4_0) **fail at boot**: the quantized-draft path's 1076 MiB compute peak no longer fits the headroom the layout leaves at MTP-creation time.
- At the smaller 229K pool the same flags boot — and are *worse* than f16: the block-quantized draft KV costs **+244 MiB** (dequant/scratch buffers) and decodes **4.5% slower** (24.52 → 23.42 t/s, same seed, paired).

The f16 draft KV stays a fixed cost of this build's MTP. The memory levers that actually work are **pool size** and, for the aggregate, the A8 placement.

## 5. Pool size and the context ceiling

| aggregate pool | per session | boot | deep test (A8 mapping) | free post-deep (c0/c1) | verdict |
|---|---|---|---|---|---|
| 229376/2 | 114688 | ok | 110K-class retrieval ok, decode 14.8 t/s | 2434 / 1286 | safe, most headroom |
| **262144/2** | **131072** | ok | 110K-class: 343 s prefill (320 t/s), 14.7 t/s decode, acceptance 0.38, retrieval ok | **1930 / 780** | **the stable baseline** |
| **294912/2** | **147456 (144K)** | ok | 128K-class: 440.7 s prefill (298 t/s), 14.8 t/s decode, **retrieval ok** | 1426 / 276 | **works; thin headroom** |
| 327680/2 | 163840 (160K) | boots with 2-4 MiB free on card 1 | **dies on the first prefill: `ggml-cuda.cu:113: CUDA error: the resource allocation failed`** (reproduced twice) | — | the ceiling: boots, not usable |

Two notes. The per-session 262144 native limit is not exceeded anywhere (320K/2 = 160K per session). And a `256K/1` configuration is *not* a memory lever in this build: the ctx size is the aggregate pool, so one slot of 262144 holds exactly the same KV as two slots of 131072 — slot count changes concurrency and per-session caps, not memory.

## 6. Phase C — unified KV: flexible multi-consumer allocation (the main functional result)

b12383 ships `--kv-unified --kv-unified-per-slot N`: one shared KV pool with per-session caps. All four per-slot caps (131072 / 200000 / 224000 / 262144) boot on the 256K pool with MTP on and the A8 mapping; unified KV costs ~250 MiB more than the non-unified layout at the same pool.

**The 200K + 32K test (per-slot cap 200000), two clients in flight simultaneously:**

| client | prompt | TTFT | decode | retrieval |
|---|---|---|---|---|
| A — the large session | 182,596 tokens (~178K-class) | 678.5 s (269 t/s effective) | 10.9 t/s while co-active (26.5 t/s solo) | **planted code found at 178K depth** |
| B — the secondary | 29,929 tokens | 66.9 s | 3.5 t/s while A decodes | code found, no cross-contamination (each answer contained only its own code) |

One session above 128K with a second session holding independent context: **yes, with correct isolation**. The cost is honest GPU time-slicing: while the big session decodes, the small one gets ~3.5 of its ~24 t/s.

**Prefix reuse.** Without any cache flag, b12383 retains an idle slot's KV only for a short while: a request immediately repeated hits the retained KV (16K TTFT 31 s → 0.46 s on four back-to-back repeats), but after a few minutes of idle the slot is cleared (a 7-minute-later 32K follow-up re-prefilled fully, `cached_tokens=0`). With the build's prompt-cache pair — `--cache-prompt --cache-reuse 4096` — reuse becomes explicit and durable:

| probe | cold | follow-up (full history resent) |
|---|---|---|
| 32K session | 66.2 s | **1.16 s, `cached_tokens=29,838`** |
| **200K session** | **724-728 s** | **1.64-2.14 s, `cached_tokens=182,619-182,836` (99.97%), reply was exactly the planted code** |

A 182K conversation turn costs ~1.6 s of prefill instead of ~12 minutes — a 440× reduction. One caveat that cost a failed request: with `--cache-prompt` on a *non-unified* 256K/2 layout the server caps each slot at half the pool (131072), so a 200K request gets a clean `400: request (182542 tokens) exceeds the available context size (131072)`; the 200K class requires the unified 200000 per-slot layout.

## 7. Phase E — slot-count concurrency (2/3/4 slots, 256K pool, A8)

Four simultaneous 16K requests (256-token decodes) per configuration:

| slots | per-client TTFT (s) | concurrent decode (t/s/client) |
|---|---|---|
| 2 | 91.8 / 31.9 / 92.2 / 127.6 | 18.5, 12.3, 4.1 (two clients queued behind the two active slots) |
| 3 | 122.9 / 93.1 / 31.9 / 85.8 | 4.7, 4.5, 5.0 |
| 4 | 90.8 / 120.0 / 31.9 / 62.0 | 4.6, 4.1, 4.7 |

Reading: per-client decode throughput degrades roughly linearly with the number of *simultaneously decoding* sessions (4-way ≈ 4.5 t/s each vs 24 solo), which is plain GPU time-slicing, not a defect. More slots do not make prefill *faster in aggregate*; they change who gets served first — the last-started client waits 3-4× a solo TTFT regardless. Retention after the burst was intact (an immediate repeat hit the kept KV, TTFT ~2 s). The practical rule for this box: **two concurrent agent-class sessions is the sweet spot; three or four work but each one gets a fourth of the decode speed.**

## 8. Phase F — tuning on the winner (A8, 256K/2, 16K class)

| knob | measured | conclusion |
|---|---|---|
| `--spec-draft-n-max` 2 / 3 (current) / 5 | 27.17 / 23.76 / 23.20 t/s (stochastic 450-token cell); draft acceptance 0.57 / 0.37 / 0.26 | **shorter draft wins** — at this model's acceptance, longer drafts cost more verification than they save; n-max 2 is +14% |
| `-ub` 64 / 128 (current) / 256 | TTFT 37.6 / 30.9 / 29.4 s | **256 is marginally the best prefill**, no penalty seen |
| `-t` 4 (current) / 8 (all logical cores) | 24.1 / ~24 t/s | no difference — the CPU is not in the loop |

## 9. Vision and reliability

- **Projector artifact:** the served GSQ-RCO BF16 mmproj matches its frozen hash; with it on the CPU (A8's `--mmproj-device none`) a two-color test image (red square left, blue square right, gray background) was described exactly right in 3.1 s at 67 prompt tokens, and the following 16K text request was unaffected — vision does not destabilize the text path.
- **Reliability:** 32/32 planted-code retrievals across the dedicated reliability cells (5 identical repeats, plus 5+5+3 repeats of story and low-entropy cells on the 256K, 229K and 288K finalists), at every pool size tested.
- **Energy (GPU power from 1 Hz sampling, not wall):** a 16K request round trip ≈ 2.8 Wh at ~200 W total; the 110K prefill + decode ≈ 31 Wh; the 128K-class ≈ 41 Wh; a 4×16K concurrent burst ≈ 23 Wh. Per-card peaks 117-144 W — both cards stay under their TDP throughout.

## 10. Pareto frontier and the deployable configurations

Production was restored to its pre-campaign configuration after each window and is not changed by this report. The three profiles below are the measured Pareto set; adopting any of them is a one-file change to the launch script (the A8 placement is two lines: the `CUDA_VISIBLE_DEVICES=1,0` environment assignment and the `--tensor-split 60,40 --mmproj-device none` flags), and rolling back means deleting them.

**P1 — Balanced multi-consumer (recommended default).** `--ctx-size 262144 --parallel 2`, A8 placement, MTP with `--spec-draft-n-max 2`, `-ub 256`: 1978/1012 MiB headroom at idle (1930/780 post-deep), 24.8-27 t/s decode at 16K, 14.8 t/s at 128K, two full 131K sessions.

**P2 — Maximum context, flexible consumers.** P1 plus `--kv-unified --kv-unified-per-slot 200000 --cache-prompt --cache-reuse 4096`: any session may grow to 200K, a 200K+32K mix works with clean isolation, and conversation turns on a 182K session cost ~1.6 s instead of ~12 min. Idle headroom 1704/874 MiB.

**P3 — Largest stable aggregate.** P1 with `--ctx-size 294912`: 144K per session, 128K-class retrieval verified, at the price of ~280 MiB of post-deep headroom on the x4 card. This is the largest pool that survives real use; 320K/2 dies on its first prefill.

| profile | aggregate | per session | idle free (c0/c1) | 16K decode | 128K decode |
|---|---|---|---|---|---|
| P1 | 256K | 131K | 1978/1012 | 24.8-27.2 | 14.8 |
| P2 | 256K (unified) | 200K | 1704/874 | 24.8-27.2 | n/a (200K-class verified) |
| P3 | 288K | 144K | 1474/508 | 24.9 | 14.8 |

## 11. Failure registry (exact reasons)

| configuration | failure | exact reason |
|---|---|---|
| 65,35 layer split, mmproj auto | OOM at boot | `allocating 887.99 MiB on device 0: cudaMalloc failed` — mmproj lands on the first GPU, which is full |
| `-devd 0` | arg error | `invalid device: 0` (device names, not ordinals) |
| `-devd CUDA0` | no-op | accepted; memory layout byte-identical to the control |
| draft KV q8_0/q5_1/q4_0 at 256K/2 | OOM at MTP creation | `allocating 1076.00 MiB on device 1: cudaMalloc failed` (compute-pp peak) |
| 320K/2 | crash on first prefill | `ggml-cuda.cu:113: CUDA error: the resource allocation failed` — 2-4 MiB of headroom on the tight card |
| `--cache-prompt` at non-unified 256K/2 with a 200K request | 400 | `request (182542 tokens) exceeds the available context size (131072 tokens)` |

## 12. Method

Every cell planted a 5-character code in a generated word corpus (3.9 chars/token) and retrieved it by asking for a ~350-word story ending in the code — correctness at depth, not just throughput. Cells were run cold (unique nonce per run) through the OpenAI-compatible endpoint with streaming and usage; decode t/s from first-to-last token; draft acceptance from the build's `print_timing` lines at `-lv 5`; VRAM by `nvidia-smi` at idle and post-cell; PCIe by 1 Hz `dmon` samples. Repetition: finalists measured 5× (the 288K finalist 3×). Windows were guarded by a detached systemd timer that kills the scratch ports and restarts + live-verifies production regardless of what the campaign session does; both windows closed well inside the guard with production restored and verified within minutes. Raw logs (server output at `-lv 5`, dmon captures, per-cell JSON) are kept in the private repo under the campaign directory.

**Corrections carried from this campaign:** the 11-14 report's §2 listed the Unsloth directory's projector hash; the served file is the GSQ-RCO release's own BF16 mmproj (`13cb7be…`). Its "33/33 layer split" description of the production config is the effective layer split of the 1,1 ratio on this model, and its card-1 headroom figure (35-41 MiB) re-measures as 40-92 MiB on the 595.71 driver. See the append to that report.

## Addendum 2026-11-18 (PCIe link correction)

The machine description's “GPU 1 (EVGA PX1) on a chipset x16 slot
electrically running at x4” is corrected by the 2026-11-18 link
investigation (model card Provenance): the slot's port is ×4-capable, but
the card's **negotiated** width has never been recorded as ×4 and is
measured **Gen3 ×1 (0.75 GB/s)**; it has most likely been ×1 since the
11-14 installation. Consequences:

- The “PCIe was checked under load: the x4 card peaks around 0.3–0.9
  GB/s (roughly a quarter of a gen3 x4 link) … the x4 link is not the
  demonstrated constraint” analysis is **unsupported as stated**: those
  bursts sit at or above the measured ×1 clean ceiling, so prefill may
  have been link-limited on the second card. The measured tok/s numbers
  stand as ×1-condition results; the ×4 potential is unmeasured (a small
  matched re-run after any physical repair is the only way to size it).
- **Preserved:** the placement conclusions (A0 vs A8 VRAM balance), the
  matched 16K decode result (24.8 tok/s in both), and the memory/ceiling
  findings — decode moves ~4% of even a ×1 link, so width does not enter
  them.
