# The BD775i as an always-on Qwen3.8-Flash-Next node: the v0.1.41 baseline (memory, KV, MTP, batch)

**Date:** 2026-11-18
**Machine:** [the BD775i](../docs/hardware/bd775i.md) (Ryzen 7 7745HX 8-core, 1×48 GiB DDR5 single channel, RTX 3060 12 GB, 512 GB NVMe, NixOS 26.11)
**Engine:** Strata v0.1.41 (upstream git tag, pinned by SHA256), built as a NixOS derivation
**Model:** ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS (75.8 GB, two GGUF shards + native pack)

## Goal

The previous report covered first contact (5-token prompt, ~14.4 t/s). This one is the baseline profile of the same node: how the engine's memory modes, KV formats, context depth, the official MTP drafter and the two-session batch mode behave on this box, measured with a full per-run resource contract, so the final always-on configuration (a 200K-token conversation running alongside a 128K one) can be chosen on evidence. The campaign ran one machine, one frozen engine, one frozen model; every number below is a median over the repetitions listed, with the range where the repetitions differed.

## Setup

**Freeze.** The run is reproducible from hashes: the engine binary (SHA256 in the campaign manifest), both model shards, the pack (`dense.bin`, index, expert profile, tokenizer), and the corpus (Moby-Dick, Project Gutenberg, 505,821 tokens, SHA256-checked against the archive). The environment (NixOS 26.11, kernel 6.18, driver 595.104.02, CUDA 13.0) is part of the manifest.

**Measurement contract.** Every cell records: wall time; the engine's own prefill/decode lines (tokens, tok/s, TTFT, draft acceptance); per-run `rchar`/block reads from `/proc/<pid>/io`; major/minor page faults; the whole-disk NVMe sector counter (whole-disk only: an earlier version of the harness summed the disk line *and* the partition lines, which double-counted every number, so the campaign's original "96 GB cold read" is actually ~48 GB and all figures below are the corrected ones); GPU memory peak, utilisation, power (avg/max), SM clock and PCIe link state on a 2 s poll series; per-core CPU busy %; memory PSI (pressure some/avg10) per sample and pre/post; pinned bytes (VmLck), RSS high-water mark; kernel error lines since the run; and the parsed engine log (PCIe probe, arena load rate, expert-cache size, prefill CPU-share state, per-chunk expert attribution, slot/batch lines). Verdicts: PASS (exact token count, coherent output), PASS_REVIEW, INVALID (coherence or a resource anomaly), RESOURCE_LIMIT (OOM/CUDA), HARNESS_FAILURE.

**Cells.** Prompts are the corpus prefix plus a random nonce (fresh tokens, no prompt-cache overlap between runs); depths 8K/32K/64K with 1024 output tokens (3 reps), 32K cold (page cache dropped, 3 reps) and 32K warm (1); the memory-mode matrix (full arena, mapped in-place, 16/24 GiB page-locked budget) at 32K and 64K; KV formats int8/k8v4/q4_0 at 32K; the single-session depth ladder 96K/128K/160K/192K/224K with int8 KV; KV streaming (`--kv-resident`, engine minimum 20480 cells) at 128K/200K/224K; the MTP drafter at 32K; and two-session batch probes. The engine's default KV is fp16; the depth ladder and the service use int8 unless stated.

## Commands

```
strata --serve --pack /var/lib/models/.../pack --native <shard1> \
  --expert-profile <profile> [--resident-experts --resident-budget-gib N | --mmap-experts] \
  --prefill auto --spec 4 --kv int8 [--kv-resident 20480] --max-context 208896 \
  [--batch 2] [--mtp /var/lib/models/.../mtp-bf16/rt]
```

(one-shot cells are the same without `--serve`, driven by the campaign harness; the MTP directory is the output of the upstream `tools/mtp_fetch.py` → `tools/mtp_pack.py` → `tools/mtp_rt.py` pipeline, described below.)

## Observations

### 1. The 40 GB arena is the box's defining constraint

The engine's default ("auto") mode copies the 39.97 GiB expert arena into a **pinned** (page-locked, CUDA-registered) anonymous buffer. On a 44 GiB-RAM box that copy and its 40 GB file cache cannot coexist: the page cache is evicted while the arena fills, so *every* fresh process re-reads ~48 GB from NVMe (measured 47.9–49.4 GB per run, 30–33 s at 1.22–1.33 GiB/s). There is no warm steady state in this mode; dropping the page cache before a run changes nothing (0.8–0.9 GB dropped, because it is already gone). The pinned arena also leaves only ~4 GB for everything else, which is why the two-session target (below) cannot run in this mode at all.

### 2. Memory mode: arena is fastest, but only at shallow depth

Decode at 32K (1024 output tokens, medians): full arena **20.8** t/s (20.3–22.8), 24 GiB page-locked 10.5, 16 GiB page-locked 10.3, mapped in-place (page cache only) **7.3**. The pinned-budget modes read 222–251 GB from NVMe per 32K run (a profile pass plus the 16–18 GiB file-backed complement churning), and that gets worse with depth: at 64K the order flips, mapped (11.0 t/s, ~13 GB NVMe/run) beats 24 GiB (9.9) and 16 GiB (8.6; 354 GB NVMe/run). The page cache's LRU is a better working-set manager than the profile-frozen pin for this model. `STRATA_IO_PREFETCH` (whole-blob preads + read-ahead threads) did not help here (7.2 mapped, 9.8 res16, versus 7.3/10.3 without); the staged variant is slower still. Pool size: the default 7 workers beat both 6 and 8.

### 3. KV format is free at these depths; KV streaming has a hard floor

At 32K the three int8-class formats decode at the same rate (int8 19.2, k8v4 21.5, q4_0 21.9 t/s, 2 reps each), so the 23% (k8v4) or 50% (q4) KV savings come at no measured speed cost; the quality cost of q4_0 is documented upstream as +8–12% perplexity on long documents. The default fp16 KV costs 2× the memory of int8 with no speed benefit, so int8 is the service format. `--kv-resident N` streams everything but the N most recent cells to pinned RAM (2.47 GiB at the 208,896-cell context, 1.16 GiB at 135,168); the engine's minimum is 20480 cells in this version.

**The arena + KV streaming combination OOMs at 200K and above.** 40 GB pinned arena + 2.76 GiB KV pin + process ≈ 45–46 GB > 44: the engine is killed by the global OOM killer at 200K/224K but boots fine at 128K (1.16 GiB KV pin; 15.5 t/s decode). That is the measurable cliff, and it is why the two-session service uses a non-arena expert mode.

### 4. Single-session depth: 224K holds

With the arena and int8 KV (no streaming), the ladder is flat: 15.3 (96K), 17.6 (128K), 15.0 (160K), 15.8 (192K), 15.0 t/s (224K); prefill stays at 859–994 tok/s at every depth (the 32K number was 801–817 with fp16 KV), TTFT 101–268 s. The model's native context is 262,144, so this configuration runs 85% of it as a single conversation. KV streaming in the mapped mode at 200K/224K works but the KV pin evicts the page-cached experts: 4.4–5.7 t/s. For a *single* deep conversation, plain arena + int8 KV (no streaming) is the fastest thing the box does.

### 5. The official MTP drafter is a 1.7× and packs to 0.89 GB

The ISTA GGUF ships no MTP head, but the upstream checkpoint (Qwen/Qwen3.8-Flash-Next, 360 GB BF16) carries the 31-tensor MTP module, and the engine's own `tools/mtp_fetch.py` downloads exactly those tensors by HTTP range from a pinned revision with per-tensor SHA256 (5.21 GB total; `verify` re-checks every tensor). `tools/mtp_pack.py` quantises the 512 routed experts to Q2_0 (the same block format as the main model's experts) and keeps the dense part BF16: 0.89 GB GGUF, unpacked by `tools/mtp_rt.py` to the runtime files (675 MB experts + 111 MB dense + 162 KB draft-vocab). The engine loads it in 0.68 s.

At 32K (512 output tokens, 2 reps): suffix-draft only 19.2 t/s; with MTP **30.9–33.4 t/s** (draft acceptance 0.46–0.66, 2.4–2.95 tokens per 6-token window). The English/code draft-vocab subset's head costs 68 MiB of VRAM (the full CJK-inclusive subset is ~180 MiB); with everything else loaded the card still shows 408 MiB free. Disabling the prefill CPU-share (default ON in this version for chunks < 1024 tokens) both speeds the MTP arm and raises acceptance (33.35 vs ~31 t/s), so the service configuration sets `STRATA_PREFILL_CPU_SHARE=0`.

### 6. Two sessions at once: it works, with slot hand-off

The engine's `--batch 2` keeps two conversations live: the first request runs on the fast solo path (with MTP drafts); when a second arrives, the first is moved into a batch slot (prompt cache holds its state, nothing is re-read) and the new one runs solo; slots decode one token per window without MTP by default and interleave with the solo stream. Measured with two 32K sessions, 128 tokens each, B starting 20 s after A: both completed (61.9 s and 64.0 s end-to-end; B's prompt read reused A's cache prefix where the two prompts shared content, as expected for the same corpus). The cost is a smaller expert cache (1.43 GiB of VRAM goes to the slot sessions) and the slower slot decode; on this 12 GB card that is the documented ~10–25% per-request penalty, and it is the price of nobody waiting for the other.

### 7. Greedy output is not byte-reproducible on this box

The upstream-documented exactness recipe (`STRATA_IQ_MT_MIN=1`, `--pcie-frac 0`, `--adapt-swaps 0`) did not produce identical outputs here: two 128-token greedy runs diverged at token 3 (106/128 different). The same two-run comparison with only the prefill CPU-share disabled came out identical, but n=2 is not a proof. The cost of the settings was measured regardless: `--pcie-frac 0` costs ~35% decode (the PCIe expert share is a real feature on this box), the static expert tier ~17%, the prefill-share switch ~0. The campaign's determinism contract records the first divergent token and the differing count per pair and classifies divergence (execution nondeterminism vs task failure) instead of demanding identity.

## Metrics

| Cell (all: int8 KV unless noted; 32K = 32,768-token prompt) | reps | decode t/s (median, range) | prefill t/s | TTFT s | NVMe GB/run | VRAM peak |
|---|---|---|---|---|---|---|
| 8K, arena, fp16 KV | 3 | 32.3 (29.3–34.2) | 806 | 11.0 | 47.9 | 11.55 GB |
| 32K, arena, fp16 KV | 3 | 20.4 (19.7–26.2) | 816 | 41.0 | 48.6 | 11.55 GB |
| 64K, arena, fp16 KV | 3 | 28.6 (20.4–30.0) | 801 | 82.6 | 49.4 | 11.57 GB |
| 32K, arena, int8 | 2 | 19.2 (18.9–19.6) | 1021 | 33.5 | 45.0 | 11.55 GB |
| 32K, arena, k8v4 | 2 | 21.5 (19.9–23.1) | 948 | 35.6 | 44.5 | 11.56 GB |
| 32K, arena, q4_0 | 2 | 21.9 (21.0–22.8) | 970 | 34.8 | 45.0 | 11.56 GB |
| 32K, mapped | 3 | 7.3 (7.2–7.5) | 816 | 41.0 | 6–50 | 11.55 GB |
| 32K, 16 GiB pinned | 3 | 10.3 (9.2–10.4) | 228 | 169 | 222–234 | 11.59 GB |
| 32K, 24 GiB pinned | 3 | 10.5 (10.5–10.7) | 206 | 194 | 248–251 | 11.60 GB |
| 64K, mapped | 2 | 11.0 (10.0–12.0) | 702 | 94.4 | 13.5 | 11.55 GB |
| 64K, 16 GiB pinned | 2 | 8.6 | 232 | 308 | 354 | 11.59 GB |
| 64K, 24 GiB pinned | 2 | 9.9 (9.6–10.2) | 217 | 339 | 396 | 11.60 GB |
| 96K, arena, int8 | 1 | 15.3 | 994 | 100.9 | 47 | 11.55 GB |
| 128K, arena, int8 | 1 | 17.6 | 966 | 136.6 | 47 | 11.56 GB |
| 160K, arena, int8 | 1 | 15.0 | 943 | 174.8 | 48 | 11.55 GB |
| 192K, arena, int8 | 1 | 15.8 | 904 | 218.5 | 48 | 11.55 GB |
| 224K, arena, int8 | 1 | 15.0 | 859 | 267.9 | 49 | 11.55 GB |
| 128K, arena, int8 + kv-resident 20480 | 2 | 15.5 (15.4–15.6) | 960 | 138.6 | 47.5 | 11.55 GB |
| 200K, mapped, int8 + kv-resident 20480 | 1 | 5.5 | 656 | 313 | 49 | 11.55 GB |
| 224K, mapped, int8 + kv-resident 20480 | 1 | 4.4 | 818 | 282 | 49 | 11.56 GB |
| 200K, mapped, k8v4 + kv-resident 20480 | 1 | 5.7 | 788 | 261 | 49 | 11.55 GB |
| 32K, arena, int8, **MTP q2_0** (prefill share off) | 2 | 31.8–33.4 (2 reps) | — | — | 45–46 | 11.57 GB |
| 2×32K, batch 2, int8 + kv-resident 20480 | 1 | both done: 61.9 s / 64.0 s end-to-end for 128 tokens each | 805–895 | — | — | 11.54 GB |

The 200K+128K two-session target (the assignment's headline) ran after this snapshot and is reported separately.

## Conclusion

The BD775i runs this model well at shallow depth (20–32 t/s decode, 800–1000 tok/s prefill, ~48 GB of NVMe per cold start) and reaches 224K single-session with int8 KV, but the 40 GB pinned arena makes it impossible to host two deep conversations: the two-session service must use a non-arena expert mode, where the page-cache (mapped) mode is the better deep-context citizen even though the arena wins at 32K. The official MTP drafter is worth its 0.89 GB (a 1.7× decode gain at 32K with 0.46–0.66 acceptance). Greedy reproducibility is not achievable with the documented knobs on this box, and every fresh process pays a ~48 GB NVMe read regardless of page-cache state, which is the single most important operational fact for the service (restarts, updates and crash recovery all pay it).

**Not covered here (final report):** the 200K + 128K two-session target run, the 8 h stability soak, the quality fixtures (MTP/q4 perplexity and comprehension), and the NixOS deployment.
