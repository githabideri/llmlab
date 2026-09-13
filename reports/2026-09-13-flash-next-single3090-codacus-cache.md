# Flash-Next Q2 on 1× RTX 3090 — Codacus CPU-Expert / Hot-Cache Baseline

**Model:** Qwen3.8-Flash-Next 125B MoE (arch `qwen4exp`; ~6 B active/token) · Unsloth `UD-Q2_K_XL` GGUF
**Hardware:** 1× RTX 3090 24 GB (the other 3090 on the main GPU server left idle), Ryzen 5 5600X, 48 GB RAM (configurable; no swap)
**Software:** `thecodacus/llama.cpp` fork, branch `perf`, commit `27c54b4bbcefadedcec6397477cc2e866c1db716` (built unchanged)
**Date:** measurements 2026-09-12 15:51–19:35 UTC; report compiled 2026-09-13

## Summary

On a **single** RTX 3090, Qwen3.8-Flash-Next Q2 decodes at **~17–18 t/s** with the CPU-expert baseline (S0, hot cache off). A **128-slot GPU hot-expert cache** lifts that to **~26.5–27 t/s** (+49–58%), using ~15.8 GB of the 24 GB card. The cache path is **deterministic within a configuration** but **not bit-identical to the cache-off path** under greedy decoding (temperature 0, fixed seed): S0 and the 128-slot configuration diverge from the **first generated content token**, even though the observed outputs stayed coherent. Because the strict bit-exact correctness gate failed, **MTP was intentionally not tested**.

This is a real, viable **fallback** that does **not** depend on the dual-GPU tensor-split path (which is a separate, still-unresolved question — see the companion [dual-3090 report](2026-09-10-dual3090-overnight-campaign.md) and its 09-12 re-test note). Whether the cache-induced divergence is benign is the open question; it is **not** established by this run.

## Why this experiment

The dual-3090 tensor-split work is the direct path to running the full 125B across both GPUs, but it is blocked on a build that correctly shards the `qwen4exp` tensors (see the companion report). While that is being diagnosed, a **single-3090** test answers a different, orthogonal question:

> If we deliberately use the **Codacus-style hybrid** — dense/common path on GPU, experts largely on CPU, a small set of "hot" experts cached in GPU VRAM, PLE host-backed — on **one** 24 GB 3090 with the artifact we already have, what performance do we get?

That is a clean baseline/fallback that stands even if the dual-GPU path never works, and it isolates the hybrid architecture's value from the tensor-split problem.

## The design under test

`thecodacus/llama.cpp` (branch `perf`, built **unchanged** for this run — no #28569, no tensor split, no replica cache, no graph merges) implements the hybrid:

- **Dense / non-expert weights** on the GPU (`-ngl 99`).
- **Experts** on CPU (the fork's CPU-MoE path; async CPU split left at fork default).
- **A hot-expert cache**: a fixed number of expert *slots* pinned in GPU VRAM, populated from a routing **profile** so the most-likely-used experts are served from GPU instead of the CPU each step (`--moe-cache-profile … --moe-cache-slots N`).
- **PLE** (the ~51 B per-layer embedding table, ~28.8 GB serialized) host/mmap-backed — it is *not* expected to consume VRAM.
- Optional **MTP** (draft speculative decoding).

The public reference that motivated this line was roughly an RTX 3060 (12 GB) running Flash-Next IQ3 with CPU experts + a hot-expert cache + MTP at ~16.6 t/s baseline and ~24.4 t/s with 56 hot slots + MTP. That is **not** directly comparable to this run (different GPU, different quant, different cache size, different MTP state); it is used only as architectural evidence that a single consumer GPU plus host-backed experts can be useful.

## Hardware & software provenance

- **Single-GPU, verified:** only `2D:00.0` (one 3090) was used (`--device CUDA0`); the sibling `2E:00.0` was confirmed idle (8 MiB, 0 % util) in the before/after `nvidia-smi` CSVs. The main GPU server's two physical 3090s are **shared** by the vLLM node and this llama.cpp node (no GPU partitioning), so the vLLM production service was **stopped for the window** to release the shared GPUs and **restored at the end** (restoration proved below).
- **RAM is a near-edge constraint, and it is a knob not a wall:** the llama.cpp node's LXC was configured at **48 GiB** (no cgroup cap beyond that, no swap) — below the 64 GB the brief assumed. The run reached **~48.7 GiB** of host RAM, i.e. at the edge of that *configurable* limit (a 60 GiB allocation was used in the 09-12 window and is available to the host). So host RAM — not just VRAM — is a real constraint here, and this argues for tuning cache/profile memory against measured host pressure and, if the profile grows, for raising the allocation toward 60 GiB in future runs.
- **`nvidia-smi` caveat:** on this box the `nvidia-smi` used-count shows a **~23 GB/card host-side reading that does not correspond to allocatable VRAM** and does not constrain `cudaMalloc` (the 4.8 GB baseline and the ~19 GB cache pack both allocated successfully against it). Allocation truth below comes from the CUDA allocator and the fork's own fit reporting, not the raw `nvidia-smi` used-count.
- **Fork build (Release):** `llama-server` `ef954d405912341f8a7a501260923d06aca7e678f7fe2ac00701f9e4a56b5968`; `libllama-server-impl` `2987821f8a77e39d02afbba5f56cb96043b0e265096739a58ce345f68a8c417d`; `libllama.so.0.3.0` `a4ee8a38dced980c70d450ed34d8043c05d5219eedba6bc6bb37cf77695db4b1`. Build flags: `-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_COMPRESSION_MODE=size -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON -DGGML_NATIVE=ON`; gcc 13.3.0, CUDA 12.0.140, driver 580.95.05.
- **Model artifact (no downloads):** shard 1 `a4f3b21e77353999829f2f767e9ac21ce9c71d29a74f2cc9eda48c9bf23c8b86`, shard 2 `2e3bf1ee7d2a04e261e9f342a2d968f696cce5941d082b0e434deb9b1edc12c6`, shard 3 `ec8c106759fdf4f463039c34c0707718d7d8908d53d892bd4f002e71620803f9`; MTP `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6`.

**Exact command** (variants):
```
env CUDA_DEVICE_ORDER=PCI_BUS_ID build-fable/bin/llama-server \
  -m Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  --device CUDA0 -ngl 99 -ncmoe 99 -c 8192 -np 1 -ctk q8_0 -ctv q8_0 \
  -b 4096 -ub 1024 -t 6 -tb 12 -fa on -lm mmap --jinja \
  [--moe-cache-profile <profile>/q2-merged.csv --moe-cache-slots 128|224] \
  [-md mtp/Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 1] \
  --host 127.0.0.1 --port 18100
```
S0 = no cache/profile args; S1a = `--moe-cache-slots 128`; S1b = `--moe-cache-slots 224` (capacity probe only); S2 = the MTP args (blocked).

**Measurement protocol:** streaming `/v1/completions`, temperature 0, seed 42, `stream:true`. Decode t/s = (SSE chunk count − 1) / (wall − TTFT); prefill t/s = prompt tokens / TTFT. One warm-up rep + 3 measured per prompt, reported as **medians over reps 1–3** (rep 0 is the cold, no-prompt-cache prefill). Two prompt families: **P1** (short prose, natural EOS at 264–288 tokens) and **P2** (code/reasoning review, 1025-token prompt, EOS-capped).

## Routing profile & cache capacity

The hot-cache profile was built from **291,840 decode trace rows** — 3 workload families × 4 prompts × 512 trace tokens: coding (103,995), prose (87,268), reasoning (100,577); prefill rows (negative position) are excluded by the fork's own parser. Merged into one file, SHA-256 `7f260a28538df160ae3313852084d0e28187d4008bddf4daa53e7b6059b8afe7`. Routing is heavily skewed (layer 0's top experts: 93(423), 272(386), 338(338), 349(308), 403(304)) — the concentration that makes a small hot cache worthwhile.

Cache cost, **measured from the allocator** (not GGUF arithmetic): **85.76 MiB per slot** (1,874,240 B/layer × 48 layers). The pack is allocated all-or-nothing:

| Slots | Total VRAM (base + pack) | Result |
|---|---|---|
| 192 | 21,226 MiB | ok |
| 200 | 21,914 MiB | ok |
| 208 | 22,600 MiB | ok |
| **224** | **23,974 MiB** (4,782 base + 19,192 pack) | **max-safe** (a first-prompt test passed; ~366 MiB headroom) |
| 256 | ~26.7 GB total | **`cudaMalloc` OOM** (pack alone ~21,975 MiB) |

S1a's **128 slots** sits at ~60 % of the max-safe slot count (~15.8 GB total), a deliberately comfortable margin rather than the 224-slot edge.

## Results

| Cell | Cache | Slots | MTP | Decode t/s (median) | Cold prefill t/s (rep 0) | GPU MiB | CPU % (med) | Host RAM |
|---|---|---|---|---|---|---|---|---|
| **S0** baseline | off | 0 | off | **17.07 (P1) / 17.80 (P2)** | 4.44 / 4.32 | 4,819 | 58 | ~48.7 GiB |
| **S1a** | on | 128 | off | **27.00 (P1) / 26.45 (P2)** | prompt-cache hit | 15,785 | 48 (GPU util 36 avg / 85 max) | ~48.7 GiB |
| S1b | on | 224 | off | *not benchmarked — branch stopped at the correctness gate* | — | (23,974 probe) | — | — |
| S2 | — | — | **blocked** | not run (gated by S1a correctness failure) | — | — | — | — |

Individual runs (medians over reps 1–3; rep 0 is warm-up): S0 P1 decode [10.44, 15.60, 17.49, **17.07**]; S0 P2 [9.17, 18.80, 17.29, **17.80**]; S1a P1 [13.00, 27.00, 26.77, 29.63 → **27.00**]; S1a P2 [13.24, 25.83, 27.79, 26.45 → **26.45**]. (P1's natural stop shifted 288 → 264 tokens between S0 and S1a — see the correctness section; the decode *rate* comparison is unaffected because it is per-token.)

All text outputs were coherent: S0 produced clean prose and a full, correct code review; S1a produced a full, valid code review (correct analysis plus two fixed versions). **Restoration proof at window close:** the vLLM service was back up (cold-start health 200 in ~145 s), a live text canary and a 2-image vision canary both passed, and the node's other llama.cpp services were left running and untouched throughout.

## Correctness gate — measured vs. hypothesized

**Measured (proven):**

1. The path is **deterministic within a configuration**: repeated S0 runs produce identical output (identical SHA across reps 1–3); repeated S1a runs likewise.
2. **S0 and S1a are not bit-identical to each other.** Under identical prompt / temperature 0 / seed 42, they **diverge from the first generated content token** (P1's common prefix is only ~55 characters; P2 diverges at the opening structure).
3. The observed outputs remained **coherent** in both configurations (no corruption, no nonsense).
4. The **strict bit-exact gate therefore failed**, so — by the run's own protocol — the cache branch was stopped and **MTP was not tested**.

**Hypothesis (not established):** the most plausible mechanism is that the hot (GPU) and cold (CPU) expert paths use **different kernels / floating-point accumulation orders**, so at Q2 quantization tiny logit-level differences flip a **near-tie** greedy winner. This is a reasonable explanation, but it is **not proven** by this run, and this report does not state it as fact. Equally, it is *not* established that the divergence is harmless — that is exactly what the proposed follow-ups test.

## Interpretation

- **The cache speedup is real and measured**: +49–58 % decode (17 → 26.5–27 t/s) from a 128-slot hot cache on a single 24 GB card, at ~15.8 GB VRAM.
- **Strict numerical equivalence is not**: the hot-cache path is not bit-identical to the no-cache path, so a bit-exact acceptance criterion rejects it.
- **Semantic correctness is still unknown**: outputs were coherent, but coherence is not proof of equivalence — a near-tie flip can be benign *or* a subtle regression, and this run cannot tell which.
- **The single-3090 hybrid is technically viable** and is the only bit-reproducible configuration at **~17–18 t/s** (S0). That is below a 25–35 t/s "comfortable planner/reviewer" band; at <25 t/s it is attractive only because **quality dominates** for a 125B/6B-active reasoning model.
- **224 slots is too close to the full card** (~23.97 of 24.58 GB) for comfortable deployment; **128 slots is the more practical operating point**, and the near-edge **48 GiB host RAM** (no swap) makes that margin important too.

## Relation to the Codacus 3060 reference

The published reference was ~3060 12 GB, IQ3, CPU experts, hot cache, MTP, ~24.4 t/s (56 slots). This run is a **3090 24 GB, Q2, 128-slot cache, no MTP, ~26.5–27 t/s**. These are **not** a like-for-like GPU-speedup comparison (different GPU, quant, cache size, and MTP state). The only valid takeaway is directional: **the same broad hybrid architecture scales to useful performance on a 3090.**

## Relation to the dual-3090 tensor work (kept separate)

This is a **different experiment** from the dual-3090 line and the two should not be merged into one "Flash-Next doesn't fit / cache is broken" narrative:

- **09-10** — the *tested build family* could not load the 125B on the pair (`layer` mode → 25.75 GB one-card buffer; `tensor`/`row` unimplemented). A build limitation, not a proven capacity limit.
- **09-12 B1** — the #28569 tensor re-test was **inconclusive**: an unsplit ~25.4 GB allocation failed on one 24 GB card. (A later clarification established the ~23 GB `nvidia-smi` reading was *not* production-vLLM contamination; see the companion report's note.)
- **09-13 (this)** — a **single**-3090 hybrid that sidesteps tensor splitting entirely (one GPU + CPU experts + hot cache). It is a valid baseline and a valid cache-speedup measurement; it fails the strict bit-exact gate.

## Next tests (proposed — **not** run here)

1. **Numerical / logit divergence characterization** (highest value): for several representative prompts around the first S0↔S1a divergence, capture top-1 and top-5 tokens, logprobs, and the margin between the top candidates. A tiny-epsilon near-tie flip supports the accumulation-order hypothesis; large score changes or unrelated candidates point to a genuine cache/remapping problem.
2. **Small semantic-equivalence set** (20–30 prompts spanning code review, debugging, planning, reasoning, prose): compare S0 vs. S1a-128 on factual correctness, instruction following, code correctness, corruption/repetition, and answer quality/length. This does **not** prove numerical equivalence; it answers whether the cache's numerical variation *materially hurts* the planner/reviewer use case.
3. **MTP** — only after the divergence is characterized and a correctness standard for the hot-cache path is chosen (adding MTP on top of an uncharacterized split would make attribution worse).
4. **Slot-count tuning around 128–192** (not 224 by default), and — if the profile grows — **raise the node's RAM toward 60 GiB** before pushing the cache further.

## Decision (two-track)

- **Strict bit-exact standard:** S0 (no cache, ~17–18 t/s) is the accepted, reproducible configuration; the hot-cache path is **rejected**.
- **Operational planner/reviewer standard:** the 128-slot hot-cache path (**~26.5–27 t/s**) remains an **open candidate** — attractive, but pending the logit-characterization and small semantic-equivalence checks above.

These are two legitimate standards; this run does not collapse them into one verdict.

## Claim status

- **Measured:** S0/S1a decode and prefill rates, per-rep values, GPU VRAM, per-slot 85.76 MiB, the 192/200/208/224/256 capacity ladder, within-config determinism, S0≠S1a divergence from the first token, output coherence, ~48.7 GiB host RAM, single-GPU usage (2D active / 2E idle), vLLM stop/restore, restoration canaries.
- **Inferred / hypothesized (not fact):** that the S0↔S1a divergence is caused by hot/cold expert accumulation-order differences flipping near-tie logits; that the divergence is benign.
- **Still open:** whether the divergence is benign (numeric + semantic characterization); MTP (never run); whether raising RAM to 60 GiB changes the operating envelope; and, separately, whether the dual-3090 tensor path can ever be made to work.

## References

- [2026-09-02 — Qwen4Exp Flash-Next on three consumer GPUs](2026-09-02-qwen4exp-flash-next-three-gpu-campaign.md) (the 30.2 t/s mixed-GPU baseline)
- [2026-09-10 — dual-3090 overnight campaign](2026-09-10-dual3090-overnight-campaign.md) (the dual-GPU wall and the 09-12 tensor re-test it supersedes for that topology)
- [Qwen3.8-Flash-Next model card](../models/qwen3.8-flash-next.md) (architecture, PLE sizing, non-PLE weight budget)
- `thecodacus/llama.cpp` fork (branch `perf`) — [github.com/thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp), tested at `27c54b4bbcefadedcec6397477cc2e866c1db716`
- [GPU server hardware doc](../docs/hardware/gpu-server.md) (reliability notes)
