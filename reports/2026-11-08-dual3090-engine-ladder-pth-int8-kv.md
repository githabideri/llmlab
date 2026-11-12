# Dual-3090 engine ladder: vLLM 0.28 → 0.30 → 0.32 with PTH INT8 KV, and why "160 t/s" and "80 t/s" are both true

**2026-11-07/08** · experiment / deployment · the vLLM node (an LXC on the main GPU server, Ryzen 5 5600X, 2× RTX 3090 24 GB, no NVLink; the always-on [Qwen3.8-27B](../models/qwen3.8-27b-rtx3090.md) production service) · vLLM 0.28.0 (production) / 0.30.0 / 0.32.0 (isolated environments)

**TL;DR**

- A two-day audit campaign on the dual-3090 box compared three serving profiles for Qwen3.8-27B W4A16-AutoRound on **identical checkpoint, vision, and 262K context**: the production vLLM 0.28.0 (fp8 KV), an isolated vLLM 0.30 (MTP v2, fp8), and an isolated **vLLM 0.32 with PTH per-token-head INT8 KV** (the per-token-head quantized-KV attention path from the community HyperQwen patch stack). On the same frozen 20K/80K *reasoning* fixture the ladder is **83/78 → 95-104/85-87 → ~100/100.8-102.8 tok/s** — the INT8-KV profile is the strongest balanced candidate: +29-32 % warm 80K decode over production, full vision/long-context/concurrency validation passed, prices being a 9-10 % slower cold long-prefill and 4 validated concurrent sequences instead of 16.
- **The whole "was it faster before?" mystery resolves to benchmark text, not hardware.** The campaign's own benchmark client feeds a repeated lighthouse paragraph to raw `/v1/completions` at temperature 0 — unusually predictable continuation — where MTP draft acceptance runs **~95 %**. Real auto-tool agent replay on the same engine accepts **~53 %** of drafts (measured A/B/A). The 150-168 t/s campaign numbers and the ~60-90 t/s agent experience are two regimes of the same engine; neither is a regression. Probabilistic draft sampling (a flag the production dual launcher had silently omitted) added **+11-14 %** on the stochastic fixture — a one-line change worth having.
- **Public KV-pool numbers that look contradictory are both right.** The 1.32M-token pool in a widely shared dual-3090 profile and the 560,900-token figure on a Hugging Face model card differ in KV precision (INT4 vs FP8), drafter precision, and reserved budget — not in what is possible. The raw arithmetic: 16 of 64 layers are full-attention with 4 KV heads × 256 dim, so a token costs 32,768 B of KV across the TP2 pair at FP8 and 16,384 B (plus small scales) at INT4. A 1.32M-token pool at INT4 is ~20.8 GiB of attention payload across both GPUs, not 40 GiB.
- **P2P is unavailable at the CUDA level on this board** (separate CPU root ports, no NVLink, IOMMU translated + ACS); vLLM runs custom-all-reduce disabled on a PYNCCL shared-memory transport (measured ~7.7 GB/s at 32 MiB allreduce). A decisive one-boot `amd_iommu=off` experiment to test whether peer access can be unlocked is fully designed (boot state captured, rollback = rebooting the original entry) but shelved as a separate maintenance window — and the data says it is not a ceiling: the same box decodes 102.8 t/s without P2P.
- Two housekeeping findings with public relevance: the watchdog that restarts a wedged engine had a **clock-domain bug** (monotonic microseconds compared against a seconds grace) that restarted the engine mid-load on one boot — fixed, and a general trap for any `systemd`+`/proc/uptime` watchdog; and the board's **PCIe slot asymmetry** (the upper CPU slot runs 10-20 °C hotter whichever card sits there) was re-confirmed by a physical card swap, which is the real thermal story on this class of board.

## Goal

The production 27B endpoint on this box had been at vLLM 0.28.0 since the dual-3090 cutover. Three questions, in order:

1. **Is the current profile behind what the same hardware can do?** Upstream vLLM and the community patch stack (HyperQwen) have moved (0.29/0.30/0.32, fused GDN decode, PTH per-token-head INT8 KV, MTP v2), and a public dual-3090 profile (YYTbit) claims 145-153 t/s warm on similar hardware. Which of that is reachable here, on this board's topology and this checkpoint, with vision and 262K context intact?
2. **Why do the measured numbers disagree with lived experience?** The 2026-09 campaigns recorded 148-151 t/s; the live-fleet 7-day average has been ~61 t/s, and users report "it felt faster before". Is there a regression, and if not, what explains the gap?
3. **What is the GPU-communication floor?** The two cards have no NVLink and sit on separate CPU root ports; how much does that actually cost, and is there a safe, testable path to P2P?

The campaign was run as bounded, self-restoring experiments (each profile in an isolated environment, the production service restored after every arm), plus a read-only hardware pass. The winning profile was kept as a **reversible trial** on the production service (a single systemd drop-in; removal restores the 0.28 profile) — the state this report documents.

## Setup

| | |
|---|---|
| **Box** | main GPU server: AMD Ryzen 5 5600X, MSI MEG X570 ACE, 64 GB RAM, 2× RTX 3090 24 GB, both on CPU-direct PCIe 4.0 x8 (buses 2d/2e, separate root ports), **no NVLink**, IOMMU translated + ACS, no P2P (measured). GPU power caps 220 W each (a 250 W control is in the metrics below) |
| **Node** | a dedicated LXC seeing both GPUs; the production service behind the fleet's model-mux |
| **Model** | Qwen3.8-27B W4A16-AutoRound (the production checkpoint, unchanged in every arm; 64 layers, 16 full-attention + 48 SSM/hybrid, 4 KV heads, head dim 256, one native MTP layer) |
| **Profiles** | **A (production):** vLLM 0.28.0 (syv-ai patch stack, CUDA 12), fp8 KV, MTP k=3 (probabilistic drafts from the 11-07 fix), 8192 batched tokens, 16 seqs · **B:** vLLM 0.30.0 (isolated venv, CUDA 13), MTP v2, fp8 KV, 11.5 GiB/card KV budget, 674,763-token shared pool · **C (the live trial):** vLLM 0.32.0 (HyperQwen commit `6ab9bd0` + 57 declared patches + KVarN, isolated venv, CUDA 13), **PTH CUSTOM attention with INT8 per-token-head KV for target and MTP drafter**, BF16 recurrent (SSM) state with the CUDA GDN decode kernel, V2 model runner, fine-grained (16-token) shared Mamba prefix checkpoint, 2048 batched tokens, 4 seqs, 673,093-token pool |
| **Constants** | 262,144 max model length, vision on, prefix caching, async scheduling, TP2, keyless, same endpoint |

**Fixed workload (the one number-to-number comparison):** two frozen code-review fixtures (20,116 and 80,117 server prompt tokens, real code and file contents, not filler), target sampling temperature 1 / top_p 0.95 / top_k 20 / seed 42, 768 generated tokens (all reasoning), one cold + three warm repeats per context and profile; rates computed from server `usage` plus streamed first/last chunk times. **Separate, clearly distinct:** the legacy low-entropy benchmark text (repeated lighthouse paragraph, raw completions, temperature 0) used by the 2026-09 campaigns, and the auto-tool agent replay (eight turns, real tool schemas, tool execution simulated).

## Method

- Each profile ran in its **own isolated venv** under a throwaway systemd drop-in on the production unit; a controller script ran the fixed workload per arm, captured the engine's `/metrics` deltas (draft/accepted counters, prefix-cache hits/queries, KV usage) and 1-second GPU telemetry per request, then **restored the original service and re-verified `/health`** before the next arm. Nothing in the production venv, launcher, or checkpoint was edited (backups kept for the two deliberate exceptions below).
- **Long-context validation** (per promoted candidate): cold 150K and 255K marker-retrieval prompts, three concurrent 180K prompts, two-image vision OCR, mixed short/long latency, prefix-cache sibling forks and branch junctions, parser/tool modes, and a post-soak repeat. All markers had to come back exactly.
- **P2P/NCCL pass (read-only, model stopped):** `nvidia-smi topo`, CUDA `cudaDeviceCanAccessPeer` both directions, NCCL 2.29.7 channel transport inspection, and a standalone two-rank synchronized PyTorch allreduce across payload sizes.
- **Deliberate, documented exceptions** (the only persistent changes of the campaign): (1) the production 0.28 launcher gained `draft_sample_method=probabilistic` in its MTP config (the recipe's own single-user launcher sets it; the dual one had silently omitted it) — measured +11-14 %; (2) the engine watchdog's clock-domain bug was fixed in place. The profile C trial above is a drop-in, not a promotion.

## Observations

### The engine ladder (frozen 20K/80K reasoning fixture, medians)

| Profile | Shared KV pool | 20K warm decode | 80K warm decode | Cold 150K / 255K TTFT | Verdict |
|---|---:|---:|---:|---|---|
| A · vLLM 0.28 (production) | 696,938 | 83.0 | 77.9-78.3 | 156.1 s / 338.7 s | the safe baseline |
| B · vLLM 0.30 MTP v2, fp8 | 674,763 | 95.0 (104.0 with BF16 state) | 85.4-87.5 | 158 s / 341 s | validated; the middle option |
| B′ · vLLM 0.30 INT4 KV (DFlash2, block-128/BLHNC) | 985,025-1,089,812 | 89.9-100.2 | **59.9-62.9** | not run | **rejected** — 80K collapse, transient 367 MB allocation failures, near-full device memory |
| **C · vLLM 0.32 PTH INT8 (live trial)** | **673,093** | ~100 | **100.8-102.8** | 172.0 s / 376.2 s | **kept as the live trial** |

Reading the table: C beats A by ~29-32 % on warm 80K decode and ~20 % at 20K; it is 18-19 % faster than the best 0.30 fp8 profile at 80K. The price is paid on the cold first token of long prompts (172 s vs 156 s at 150K, 376 s vs 339 s at 255K — 9-10 % slower, the INT8 per-token-head quant/dequant on the prefill path) and on concurrency (4 validated active sequences vs 16; three concurrent 180K prompts all passed, with end-to-end times of 246.6 / 471.9 / 674.1 s — severe prefill/decode interference, no allocation failures). For a workload of one long retrieval prompt followed by a short answer, the cold TTFT penalty dominates; for the actual agent regime (one long cold first turn, many cached follow-ups) the decode gain dominates, which is what the first real agent session on the trial showed: a cold 147,617-token first prompt generated 1,317 tokens in 166.9 s end-to-end, then six follow-ups (149-155K prompts) returned 224-2,400 tokens each in 4.88-27.11 s (~81 generated t/s across the set, prefix-cache hit rate 87 %, zero queued requests after the session).

**Short-context latency is where the small batch budget pays back.** A mixed test (short prompt arriving while a long one runs) measured first-token at 14.2-14.6 s on the 8192-batch profiles but **2.6-2.7 s on the 2048-batch profiles** (0.30 and 0.32 alike). The 8192 budget that won the 2026-09 single-stream 16K test starves an interactive request that has to wait behind a chunked prefill; 2048 does not. The 0.32 startup log itself warns that 2048 may be suboptimal with 3 MTP draft tokens — a 4096/8192 re-comparison, measuring cold TTFT and mixed latency rather than decode alone, is the next tuning step.

### The acceptance reality check (160 vs 80)

Same engine, same checkpoint, two text regimes — measured as server counter deltas (drafts, accepted, per-position), not estimates:

| Text regime | 0.28 | 0.30 (re-probed) |
|---|---:|---:|
| Repeated lighthouse paragraph, raw completions, temperature 0 (the 2026-09 campaign benchmark) | **95.5 %** draft acceptance | 95.0 % |
| Auto-tool agent replay, 8 turns, stochastic target sampling | **54.3 %** | 53.1 % |

Per-position acceptance on the agent replay fell 69 % / 49 % / 36 % across the three draft positions. So the campaign's 150-168 t/s (and this audit's own 157.9-160.5 t/s re-measurement of the 16K-target cell at 220 W — the cell's actual prompts were ~21.8K tokens, off-spec by the client's own flag, but the 220/250 W and pre/post-swap runs are matched to each other) and the ~60-90 t/s agent experience are **the same engine reporting two different draft-acceptance regimes**. The 2026-09 campaigns' "16K decode 147.8 t/s" is a low-entropy number; it is not an agent-throughput promise, and no regression was found: single-stream at equal power cap reproduced, and the felt-slow period of 2026-09 is already explained by the driver's SW-thermal derating of the hot slot (see the hardware doc). Two method corollaries, now part of the [benchmarking method](../docs/benchmarks.md): always report MTP acceptance next to tgen, and never port a temperature-0 low-entropy rate to a stochastic agent workload.

The one-line fix of the campaign: the production dual launcher had omitted `draft_sample_method`, so it drafted **greedy** while targeting a stochastic sampler — the recipe's own single-user launcher sets `probabilistic`. With probabilistic drafts (which retain the draft distribution for standard rejection sampling) the frozen fixture moved 83.0 → 92.1 (20K) and 77.9 → 88.6 (80K) warm, +11-14 %, canaries intact. A custom reduced-head "local argmax" variant was A/B/A'd and gave no measured gain (acceptance identical to the matching requests); it was reverted.

### The KV-pool arithmetic (why 560K and 1.32M are both right)

This model has **16 of 64 layers as full attention**, 4 KV heads, head dim 256. Raw attention-cache cost per token across the TP2 pair:

```
16 layers × 2 (K and V) × 4 heads × 256 × bytes-per-value
FP8:  32,768 B/token → 8.0 GiB per 262,144 tokens
INT4: 16,384 B/token (+ ~512 B/token of per-token-head scales) → ~4.1 GiB per 262,144
```

So a **1.32M-token** pool's raw FP8 payload *would* be ~40.3 GiB — but the published 1.32M profile (YYTbit, same hardware class, also no NVLink/P2P) uses **INT4 target KV + BF16 draft KV + FP16 recurrent state** with a different drafter (DFlash2 k7) and checkpoint: at INT4 that pool is ~20.8 GiB of attention payload **aggregated across both GPUs**, with the reported 23.3 GiB/GPU being total device allocation (weights + state + graphs + reserve), not KV alone. The Hugging Face 560,900-token figure is an FP8-KV configuration of a different checkpoint. Pool size is therefore a function of KV precision × reserved budget × runtime profiling — not a property of the checkpoint — and none of the public numbers imply a 1.3M-token *request* (the per-request ceiling is the 262,144 `max_model_len` in every profile, ours included). Our own measured pools: 696,938 (0.28), 674,763 (0.30 at an explicit 11.5 GiB/card KV budget — the auto-sized first attempt reported 727,033 and OOM'd on a 20K prefill), 673,093 (0.32 at 11.75 GiB/card INT8). All three hold the full 262K with ~2.5-2.7× the max single request; the pool is shared by the pair, not per-GPU.

### GPU communication: measured floor, prepared experiment

- `nvidia-smi topo`: both cards on **separate CPU root ports (PHB)**, no NVLink. CUDA `cudaDeviceCanAccessPeer`: **false in both directions**; NVML peer check `CNS`. vLLM accordingly disables custom all-reduce and uses **PYNCCL** (NCCL 2.29.7), whose channels negotiated a **shared-memory (SHM/direct/direct) transport** — host memory, not TCP, and no evidence of a broken setup (the Socket plugin's init lines are not the data path).
- Standalone two-rank synchronized allreduce (50 reps/size, both ranks, Python + synchronize overhead included): 52.1 µs @1 KiB, 48.7 µs @32 KiB, 87.5 µs @256 KiB, 191 µs @1 MiB, 1.13 ms @8 MiB, 4.37 ms @32 MiB (~7.7 GB/s payload at 32 MiB). These are host-staging baselines, not per-layer collective costs, and not an available speedup.
- **The prepared one-boot experiment** (not run; needs a maintenance window with console access): NVIDIA documents that a translated IOMMU domain and ACS can block PCIe P2P on bare metal, so the decisive test is booting once with `amd_iommu=off` (the less-disruptive comparison is an identity-domain `iommu=pt`), verifying the actual cmdline and IOMMU state, re-testing CUDA peer access, then — only if it appears — a P2P data-integrity check, the same NCCL transport/collective suite, and the same inference profile at equal power cap, requiring correctness, no Xid/AER errors, and a *measured* benefit before any persistent change. Rollback is rebooting the original entry. Two explicit no-gos: do not run the documentation's wildcard `setpci` ACS-disable script on a hypervisor host, and do not force an NCCL distance variable to "create" peer access that CUDA reports unavailable. The measured 102.8 t/s with SHM transport is the evidence that P2P, if it ever unlocks, is an optimization on this box, not a prerequisite.

### Housekeeping findings (brief)

- **Watchdog clock domains.** The restart-on-`/health`-failure watchdog computed the unit's age in **monotonic microseconds** (systemd's `ExecMainStartTimestampMonotonic`) and compared it against a 900-**second** grace, and aged epoch-second markers against the microsecond counter. On the post-maintenance boot the first timer run (~90 s in) "reported" an active age of 77,525,188 s and restarted a still-initializing engine; the second automatic start loaded fine, but the 15-minute grace was defeated. Fixed by converting to seconds before comparing and using one epoch-seconds domain for all markers. General trap: any watchdog mixing `/proc/uptime`·1e6, systemd monotonic stamps, and `date +%s` has three clock domains in one script.
- **Slot asymmetry, confirmed twice.** With the cards physically swapped, the card in the upper CPU slot ran 86-89 °C with intermittent SW-thermal slowdown under sustained load while its twin in the lower slot sat at 60-62 °C at equal 220 W caps — the same ~10-20 °C asymmetry the pre-swap layout showed for the *other* card. The heat follows the **slot**, and at the 83 °C driver target the upper-slot card derates under long workloads. Power cap: a matched 3-repetition 16K-target test gave 157.92 t/s at 220 W vs 164.80 at 250 W (+4.4 % decode, peaks 74/67 °C, no thermal event) — a real small lever, not a regression explanation; the box stays at 220 W until the planned open-air riser rehouse makes slot position moot.
- **Storage.** The host's two secondary NVMe drives (a retired NixOS LUKS drive and a failed 2 TB drive with 100 media errors) were removed after SMART forensics; the box now runs its ZFS root on the single boot NVMe. No inference-path impact (model storage is on the root pool's dataset).

## Metrics

**Decode (frozen 20K/80K reasoning fixture, stochastic sampling, medians):** as in the ladder table. Post-soak repeat on the trial: 119.9 (20K) / 100.8 (80K) — stable, not a first-run artifact.

**Prompt processing:**

| Profile | ~21.8K prompt | 150K cold (TTFT-derived) | 255K cold (TTFT-derived) |
|---|---:|---:|---:|
| 0.28 | **~1,310 t/s** (65,366 fresh tokens / 49.9 s, engine prefill metrics) | ~962 | ~753 |
| 0.32 PTH trial | **~1,365-1,370 t/s** (3 reps, same cell, 2026-11-08) · **~1,216-1,257 at the 32K cell** (actual ~43.7K) | ~873 | ~678 |

The INT8 profile's cold-prefill tax (9-10 %) shows at long context, not at 16-32K. TTFT-derived rates include queueing; the 21.8K cell is engine-prefill-metric exact. Wall power was not metered in this campaign; both cards sat at the 220 W cap (213-219 W drawn) during sustained decode, the same envelope as the 2026-09 campaigns.

**Concurrency (0.32 trial, three simultaneous 180K prompts):** all exact-marker checks passed; TTFT 225.1 / 450.5 / 673.6 s; E2E 246.6 / 471.9 / 674.1 s. Two-long-requests-in-flight admission is the interference regime; the 4-sequence cap means a fourth concurrent user queues rather than co-decodes.

## Conclusion

1. **vLLM 0.32 with PTH per-token-head INT8 KV is the best measured profile for this checkpoint on consumer dual-3090 hardware without NVLink**: ~+30 % warm long-context decode over the 0.28 production baseline, the full 262K/vision/concurrency validation passing, with the price stated (cold long TTFT −9-10 %, 4 validated concurrent sequences). It is deployed here as a **reversible trial** (one drop-in; the 0.28 profile is the one-step rollback), which is the honest status: strong, validated on the campaign's gates, awaiting enough real multi-session agent use to decide promotion. The 0.30 MTP-v2 fp8 profile is the middle candidate if INT8 KV precision ever becomes a quality concern; the INT4-KV high-pool profiles are a *capacity* play, not a speed one, and failed the long-context throughput and stability bars.
2. **The fleet's speed disagreement is closed as a benchmark-text artifact, not a hardware regression.** Report acceptance with tgen; treat the low-entropy 150-168 t/s as what it is (a ceiling for predictable text at temperature 0), and the ~60-100 t/s stochastic/agent band as the operative expectation. The probabilistic-drafting one-liner (+11-14 %) was kept.
3. **The board's communication floor is known and bounded**: no P2P, SHM-transport NCCL, measured collective latencies, and a fully specified (shelved) one-boot IOMMU experiment for anyone who wants to test the last variable. It is not what limits this box — decode at 102.8 t/s without P2P is the proof.
4. Next, in order of leverage: 2048→4096/8192 batch-budget re-comparison (the startup warning, measured against cold TTFT and mixed latency), `MAX_SEQS=8` validation for real multi-session use, the riser-rig cooling work, then the P2P boot experiment if it is ever wanted.

## References

- [Qwen3.8-27B model card](../models/qwen3.8-27b-rtx3090.md) (the 2026-09-10/11 campaigns it extends; its 148-151 t/s rows are the low-entropy regime) · [GPU server hardware profile](../docs/hardware/gpu-server.md) · [benchmarking method](../docs/benchmarks.md) (the acceptance-rate note added by this work)
- [HyperQwen](https://github.com/syv-ai/HyperQwen) (vLLM 0.30/0.32 patch stack; [PTH per-token-head FA](https://github.com/syv-ai/HyperQwen/blob/main/pthfa/README.md), [long-context notes](https://github.com/syv-ai/HyperQwen/blob/main/docs/long-context.md))
- [YYTbit/qwen3.8-27b-3090](https://github.com/YYTbit/qwen3.8-27b-3090) (the public 1.32M-pool dual-3090 profile compared in the pool arithmetic) · [tonyd2wild's SGLang-vs-vLLM dual-3090](https://github.com/tonyd2wild/Qwen3.8-27B-SGLang-vs-vLLM-2x3090) (NVLink-equipped, different quants — a context, not a comparison) · [ExLlamaV3 DFlash2 on Qwen3.8-27B](https://github.com/r0b0tlab/qwen38-exl3-dflash2) (the short-context-headline caution)
- NVIDIA: [multi-GPU systems (P2P/IOMMU/ACS)](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html) · [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
