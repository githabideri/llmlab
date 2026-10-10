# Strata expert-offload vs llama.cpp on a 31 GB / RTX 3060 host: Qwen3.6-35B-A3B A/B

**Date:** 2026-10-09
**Status:** measured, single-machine A/B; not a production cutover. The budgeted (low-RAM) mode was measured twice; the full-arena mode could not be measured on this host — it OOM'd the host during arena fill (Findings 1 and 3).
**Engine under test:** Strata, PR #1402 (branch `qwen36-35b-support`, base v0.1.41), built locally (`CMAKE_CUDA_ARCHITECTURES=86`), serve mode with OpenAI-compatible API. (The upstream repo rewrote its history on 2026-10-06 — PR #1276 — so no specific commit hash is cited; build from the branch tip of that era.)
**Baseline:** the same box's production llama.cpp server for the same model (build `925e1179`)

## Question

The PR claims 2× decode and 6× prefill over llama.cpp for 30B-class MoE on consumer hardware (reference box: 6-core, 62 GB, RTX 4070 Ti, 32K context, full arena). This box is smaller in every dimension: a 4-core/4-thread i5-7400, 31 GB RAM, RTX 3060 LHR 12 GB (sm_86), on a Proxmox 9 host that also runs a virtual machine and several containers. Can the engine even run here, do the ratios hold — and in which of the engine's two memory modes?

## Setup

Same model file for both engines: Qwen3.6-35B-A3B, UD-Q4_K_XL (22.85 GB GGUF; 40 blocks, 256 experts / 8 active per token, MTP draft block; the expert arena within it is 18.29 GiB).

- **llama.cpp (A):** 128K context, k8v4 KV, flash attention, batch 4096 / ubatch 2048, MTP speculative draft (n_max 2), 28 of 40 MoE layers on the CPU, fully in RAM (no-mmap), behind a model-muxing proxy. This is the live production configuration; it ran throughout the baseline phase, so the baseline was measured with zero downtime, in steady state (model resident for hours). It was also re-measured inside the second Strata window (a note after the table covers that copy's fresh-load warmup).
- **Strata (B):** 128K context, int8 KV, profile-ranked VRAM expert cache (6.2 GB resident + 3.3 GB borrowed by the prompt path; 449 MiB of the 12 GB left free with everything loaded), MTP draft head with an English-only draft vocab, `--spec 4 --spec-min-p 0.5`. Two of the engine's memory modes exist: a **budgeted mode** (`--resident-budget-gib N`: a pinned host-RAM complement of the experts the VRAM cache does not hold, the rest read in-place from the model file on a ~300 MB/s SATA disk) and a **full arena** (no budget: all 18.29 GiB of experts page-locked, disk out of the inference path). The first window used the 14 GiB budget; the second used 20 GiB — and as Findings 1 explains, both pin the identical 12.09 GiB complement, since the budget only sizes that complement (12.09 GiB total) and the next size up is the whole arena.

Cells (streaming): C1 short prompt (31-token fixture, 128 out), C2 6.8K-token document read (64 out), C3 13.5K-token document read (128 out), plus three canary questions at temperature 0.3 for output comparison (a semantic check — the PR's parity claim is for greedy, and B prompts carried a nonce tag that adds ~17 tokens, so a token-level parity test was not possible here). One run per cell. Non-streaming rows were measured but are excluded from the table: the baseline server reuses its KV cache across requests (its 13.5K non-streaming request completed in 6.6 s — faster than the same server's 10.3 s streaming first-token on the identical fixture), while Strata performs no prefix reuse (every prompt logged "0 reused"), so that row is not comparable between the two.

## Commands

Reproducible as: check out the PR's branch (`qwen36-35b-support`, v0.1.41 base), build with `CMAKE_CUDA_ARCHITECTURES=86`, pack the existing GGUF with `tools/iq_pack.py --model <gguf> --out <pack> --compat-bf16` (all 613 tensors native: 252 expert, 140 dense, 0 compat), then serve. The serve config used (JSON):

```json
{
  "model": "<Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf>",
  "pack": "<pack dir>",
  "serve": ["--host", "<ip>", "--port", "8181", "--api-key", "<key>",
            "--expert-cache", "auto", "--mtp", "<draft-head args>",
            "--spec", "4", "--spec-min-p", "0.5",
            "--max-context", "128K", "--kv", "int8",
            "--resident-budget-gib", "14"]
}
```

The full-arena mode is the same config with `--resident-budget-gib` removed. Two launch constraints from the field: the process must run under `LimitMEMLOCK=infinity` (e.g. `systemd-run --property=LimitMEMLOCK=infinity`) or the arena's `cudaHostRegister` fails under an 8 MB memlock cap and the engine sits idle without surfacing the error; and the server binds only `--host`, not loopback. Before a launch, pre-warming the model file into the page cache makes the first prefill of the window hit cache instead of disk. Cells: OpenAI chat-completions, streaming, one run per fixture; prefill/decode rates from the engine's per-batch log lines, TTFT and client-side decode from the OpenAI client. Wall power is not instrumented on this host (no meter on the box or the card), so there are no energy figures; rates only.

## Results

| Cell (streaming) | llama.cpp (A, steady state) | Strata budget 14 (B) | Strata budget 20 (B20) |
|---|---|---|---|
| **C1 short** (~50 in, 128 out) — TTFT | 1.21 s | 1.07 s | 0.80 s |
| C1 — prefill (engine) | ~40 t/s (est.) | 59.4 t/s | 71.6 t/s |
| C1 — decode | 35.7 tok/s | **52.6 tok/s (1.47×)** | **49.3 tok/s (1.38×)** |
| **C2 6.8K read** (64 out) — TTFT | 3.23 s | 10.90 s (cold 1st) | 16.28 s (cold 1st) |
| C2 — prefill (engine) | ~2100 t/s (est.) | 636.6 → 1434.2 t/s (2nd req) | 419.9 → 1252.7 t/s (2nd req) |
| C2 — decode | 39.6 tok/s | 46.7 tok/s (1.18×) | 58.0 tok/s (1.46×) |
| **C3 13.5K read** (128 out) — TTFT | 10.29 s | 10.83 s | 11.04 s |
| C3 — prefill (engine) | ~1311 t/s (est.) | 1261.3 t/s | 1232.6 t/s |
| C3 — decode | 41.5 tok/s | 53.0 tok/s (1.28×) | 50.9 tok/s (1.23×) |
| **Canaries** (21–40 in, 64 out, temp 0.3) — TTFT | 1.86–2.14 s | 0.55–0.78 s | 1.50–3.00 s (cold window) |
| Canaries — prefill (engine) | — | 69.6–88.1 t/s | 26.9–47.1 t/s |
| Canaries — decode | — | 59.8–66.1 tok/s | 30.1–62.1 tok/s |
| MTP draft acceptance | (n_max 2; not logged) | 82–98 % per cell | 61–100 % per cell, median ~88 % |

**Full arena (no budget): no cells measured on this host** — the arena fill was killed by a host-level out-of-memory (Finding 1). The "cold 1st" prefill rows are the first 6.8K request of each window: after the arena's pinned copy evicts the model file's cached pages, the first prefill re-reads the file tier from disk; the second request runs from the re-warmed cache. The canary rows vary because they were measured with and without streaming across the runs.

The llama.cpp column is the steady-state production baseline (model resident for hours, zero-downtime measurement). A second llama.cpp measurement taken inside the B20 window (model loaded 55 s earlier) matched the 13.5K cell (10.21 s vs 10.29 s) but was elevated on the two shorter streaming cells (1.93 s / 8.17 s vs 1.21 s / 3.23 s, with decode correspondingly depressed to 23.6 / 37.5 tok/s) — llama.cpp's own fresh-load warmup, not cross-engine contention (no other GPU process was present in that window). The ratios above use the steady-state column; against the fresh-load copy, B20's C1 decode would look like 2.1× instead of 1.38×.

Canary texts match semantically across engines (same answers; temperature 0.3 plus the nonce tag account for wording drift). Qwen3.6 emits its answer in the `reasoning_content` channel before `content` — token counts above include both.

## Findings

1. **The engine runs on a 4-core/31 GB/3060 box — in budgeted mode.** Whether full-arena mode runs is a property of what else is in the host's RAM. The full arena (18.29 GiB of experts page-locked, process peak ~20 GB) was attempted in a clean window with 22 GB free; it reached 19.2 GiB of pinned RSS with ~2.5 GB of headroom left, and a co-located virtual machine's memory growth then pushed the host into a **global out-of-memory** that killed the engine (the kernel correctly picked the 19.2 GB test process, not the production one; nothing else was harmed, and the production server was restored and re-verified). Two genuinely failed early windows are worth knowing for operators: (a) under an 8 MB `memlock` hard limit the arena's `cudaHostRegister` fails with EPERM and the engine does not surface the error — it sits idle until the server's startup timer kills it; run it under `LimitMEMLOCK=infinity`. (b) page-locked memory cannot be swapped: with a ~9 GB host baseline, a 20 GB arena leaves too little headroom for the host's other consumers, and the "available memory" floor is misleading because most of it is reclaimable file cache — a 2.5 GB floor still OOMs on a non-reclaimable request.
2. **Decode is 1.2–1.5× faster, short-context first-token ~2.4–3.9× faster (warm window) — and it holds across two independent windows and two baseline copies.** The engine's decode cache hit rate climbed ~58 % → 86 % within a run (profile-ranked hot experts in VRAM); only 4–12 % of routed experts were fetched over PCIe. The MTP spec window (up to 6, min-p 0.5) accepted 61–100 % of drafts per cell (median ~88 %), inside the PR's claimed 0.84–0.96 band.
3. **Prefill is mode-specific, and on this host only the file-tier behavior is measurable.** In budgeted mode, steady-state prefill (after the window's first 6.8K request re-warms the cache) runs at 1250–1430 t/s — the same band as the production server's 1300–2100 t/s — so at 13.5K the two are roughly even. The visible gap is the **cold first request of each window** (420–640 t/s vs ~2100 t/s for the server on the same cell): a page-cache management cost of the pinned-arena design, not a sustained compute deficit. The full arena — the mode the PR's 6× prefill number presumes, with the file tier out of the path entirely — does not fit this host beside its other memory consumers (Finding 1), so its prefill number is the one this A/B could not produce.
4. **Footprint:** 11.7 GB of the 12 GB VRAM used (identical to the production server after restore; the two are mutually exclusive on this card). Budgeted host-RAM footprint ~14–15 GB, with a measured floor of ~6 GB available — comfortable. Full arena ~20 GB — does not fit this host as configured (Finding 1).

## Verdict

On a 31 GB / 4-core / RTX 3060 Proxmox host, Strata is a **clear win for chat-style workloads** in its budgeted mode (short prompts, long answers: +18–47 % decode across the two windows, ~2.4–3.9× short-prompt first-token in the warm window, MTP acceptance matching the PR's claims) and **roughly even with the production server on long-document reads once the window's first request has re-warmed its file tier**, with a 3–5× cold-start penalty on that first request. The full-arena mode — the only configuration that would take prefill into the compute-bound regime the PR benchmarks — **cannot be measured on this host**: its ~20 GB of non-swappable RAM does not fit beside the host's other consumers, and the attempt ended in a host-level OOM (recovered cleanly; production verified after). For this box's mixed workload, the production llama.cpp configuration stays the better default; a Strata endpoint is a feasible alternate for chat-heavy use (same VRAM footprint, a stop/start swap), and the full-arena numbers remain open for a dedicated 32 GB+ machine or a maintenance window with the host's other memory consumers out of the way.

*Sanitization: no internal host names, IPs, or machine identifiers appear in this report; "this box" is a 31 GB / i5-7400 / RTX 3060 Proxmox host, and "the co-located virtual machine" is its largest non-35B memory consumer.*
