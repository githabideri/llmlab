# The two 3060s, repurposed: 35B squeezed onto one 12 GB card, 27B served on the pair

**Date:** 2026-08-30
**Category:** Experiment / benchmark
**Status:** Measured on the main box (Ryzen 5 5600X, 1× RTX 3090 + 2× RTX 3060). The 3090 serves Qwen3.8-27B in production; the two 3060s serve Qwen3.6-35B-A3B. This campaign temporarily stopped the 35B unit, repurposed the 3060 pair, and benchmarked both directions of the trade-off. All numbers below were re-run independently of the original campaign to confirm them.

**TL;DR**

- A 35B MoE (3B active) in 2-bit **fully fits on one 12 GB 3060** — 11.3 GB VRAM, and it is *provably* resident: ~0.02 GB/s PCIe in decode vs 5.9–6.7 GB/s for its CPU-MoE twin on the same card. 81 t/s at 2K, 43 t/s at 64K context.
- The same two 3060s run a 27B dense model (vLLM, TP2 + MTP) at **77–87 t/s decode / 560–700 tok/s prefill** — matching a 3090 within a few percent while drawing less power (wall ~340 W vs ~370 W for the 3090 serving the same model).
- The 27B node is a **single-user machine**: 4× concurrent 16K contexts collapses per-request decode to ~16 t/s, and the KV pool (126K tokens) fits at most two 64K contexts.
- Two of the campaign's earlier claims were **retractions in disguise**: "llama.cpp 27B beats the 3090" and "3090 at 13K prefill tok/s" were both prefix-cache artifacts. Cold, the 3090 wins 2.7× on decode.

---

## Goal

Answer three questions with the 3060 pair:

1. Does the 35B MoE actually fit *on one* 12 GB card (and is it truly resident, or secretly streaming)?
2. How does a 27B dense model behave on the pair — context degradation, concurrency, power — as a candidate **second inference node** next to the 3090?
3. How do all of this compare against the 3090 reference, at the wall, not just per-GPU?

## Setup

| Node | GPUs | Model | Backend |
|---|---|---|---|
| Reference | 1× 3090 24 GB | Qwen3.8-27B W4A16-AutoRound (19.5 GB) | vLLM 0.27.1, MTP k=4, fp8 KV, FlashInfer, 64K max len |
| 27B candidate | 2× 3060 12 GB | same | vLLM 0.27.1, **TP2**, MTP k=4, fp8 KV, FlashInfer, 64K max len, `gpu-mem-util 0.90` |
| 35B squeeze | 1× 3060 12 GB | Qwen3.6-35B-A3B UD-IQ2_XXS (10.03 GiB file; see addendum) | llama.cpp mainline `925e117`, `-ngl 99`, flash-attn, q8_0 KV, MTP variant tested |
| 35B control | 1× 3060 12 GB | same | same + `--n-cpu-moe 99` (experts in host RAM) |

Method discipline (full write-up in [docs/benchmarks.md](../docs/benchmarks.md)): cold requests only, each with a **unique nonce** (vLLM prefix cache and llama.cpp `--cache-prompt` both silently cache repeated fixtures); decode rate = exact `completion_tokens` from `/usage` divided by decode wall — **never chunk counts** (speculative decoding ships multiple tokens per chunk, which makes any chunk-derived "tok/s" look 3–5× faster); per-request wall-clock markers aligned against 1 Hz `nvidia-smi dmon` (power, SM, VRAM, **PCIe RX/TX**); wall power from the PDU/smart-plug feeding the box.

Fixtures: deterministic text files of ~2K / 16K / 24K / 32K / 48K / 64K tokens (prose, ~0.72 tokens/char).

## Part 1 — 35B MoE on one 12 GB 3060

**Config:** `-m Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf -ngl 99 -c <tier> -ctk q8_0 -ctv q8_0 --flash-attn on -b 2048 -ub 1024 -np 1` (fresh server per context tier, no `--cache-prompt`).

### The decode ladder (single user, 256-token completions, cold)

| Context | Prefill tok/s | Decode t/s | Decode power (dmon) | PCIe RX during decode |
|---:|---:|---:|---:|---:|
| 2K | ~1,100 | **81.5** | ~57 W (mixed) | ~16 MB/s |
| 16K | ~1,100 | **68.6** | ~108 W (SM 98%) | ~21 MB/s |
| 32K | ~1,050 | **57.7** | — | — |
| 64K | ~930 | **43.2** | — | — |

(Prefill runs ~1–1.4K tok/s across the whole ladder — the re-run came out a touch faster than the original campaign, 1.1–1.45K vs 0.93–1.1K — and stays nearly context-free thanks to the linear-attention hybrid, unlike GQA models which fall off a cliff here.)

### Proving residency: PCIe, not timing

A streaming model decodes at a *plausible* steady rate too, so "it fits in VRAM" is a bus claim. Same card, same model, two configs, one 16K-context request each:

| Config | PCIe RX (decode) | PCIe TX | SM | Power | Decode |
|---|---:|---:|---:|---:|---:|
| fully resident (`-ngl 99`) | **21 MB/s** | 41 MB/s | 98% | 108 W | 68.6 t/s |
| CPU-MoE (`--n-cpu-moe 99`) | **5,870 MB/s** | 520 MB/s | 42% | 45 W | 35 t/s |

Resident is **~1.8–2× faster** *and* draws more power (all layers run on-GPU; the offloaded variant idles the GPU while the 5600X pushes 6 GB/s over PCIe). The 2-bit quant is what makes this work: 11.3 GB of the 12 GB card, with the 64K KV cache included. The MTP variant (k=3) adds ~99 t/s decode but pushes VRAM to the cliff — it only reliably boots with ~16K context budget, not 64K.

**Takeaway:** a 2-bit 35B on a single 12 GB card is a real, resident, ~43–81 t/s deployment — at the cost of 2-bit quality (not yet evaluated against the IQ4_XS production unit; open item).

## Part 2 — 27B dense on the 3060 pair (vLLM TP2 + MTP)

**Config:** `vllm serve Qwen3.8-27B-W4A16-AutoRound-fast --tensor-parallel-size 2 --speculative-config '{"method":"mtp","num_speculative_tokens":4}' --kv-cache-dtype fp8 --attention-backend FLASHINFER --gpu-memory-utilization 0.90 --max-model-len 65536 --max-num-batched-tokens 2048`.

### Context degradation (single stream, cold)

| Context | Cold TTFT | Prefill tok/s | Decode t/s |
|---:|---:|---:|---:|
| 2K | 2.9 s | ~700 | **87** |
| 16K | 23.9 s | ~670 | **77** |
| 24K | 37.1 s | ~650 | **67** |
| 48K | 81.2 s | ~590 | **57** |
| 64K | 113.9 s | ~560 | **57** |

Two regimes: prefill degrades gently (chunked prefill at 2K tokens/step keeps it ~600–700 tok/s to 64K — the 3090 does ~1,000), and decode holds 77 t/s through 16K, then falls to a ~57 t/s **plateau** from 48K up. MTP acceptance: 46–61% (4-token drafts) single-stream; it degrades under concurrency (below).

### The concurrency cliff

| Scenario | Per-request decode (t/s) | Aggregate output (t/s) | Wall power |
|---|---|---:|---:|
| 1× 2K | 87 | 87 | ~200 W |
| 2× 2K | 31 / 66 | ~97 | ~220 W |
| 4× 2K | 17 / 22 / 31 / 53 | ~122 | ~230 W |
| 8× 2K | 11–69 (mean 24) | ~191 | ~250 W |
| 2× 16K | 9 / 60 | ~35 | ~340 W (incl. prefill) |
| 4× 16K | **3 / 5 / 9 / 46** (mean 16) | ~32 | ~345 W (incl. prefill) |

The shape: short prompts batch well in aggregate (8×2K delivers ~2× single-stream throughput), but **16K contexts do not share** — chunked prefill serializes the prompts (TTFTs stack to 25–100 s) and the 4-way 16K decode batch collapses to single-digit per-request rates while the GPUs sit at 99% SM. MTP acceptance fell from ~50% to 17–39% in these windows, compounding the loss.

**KV ceiling:** the pool is **126,390 tokens** (fp8 KV, 0.90 util). At 64K that is 1.93× — a 64K stream is effectively the only tenant; two 48K contexts fit; four 16K fit on paper but are the cliff row above.

### 3090 vs the pair, at the wall

The box idles at **~93–98 W** (5600X + three idle GPUs + NVMe). Measured from the smart plug feeding the server:

| Serving | 3090 power (dmon) | 3060-pair power (dmon) | Wall (plug) | Decode |
|---|---:|---:|---:|---:|
| 27B on 3090 (vLLM MTP) | ~219 W | ~40 W (35B unit was up) | ~330–345 W | 91–105 t/s |
| 27B on 2×3060 (vLLM TP2+MTP) | ~13 W (idle) | ~220 W | **~340–370 W** (peaks 460) | 57–87 t/s |
| 35B on 1×3060 (llama.cpp, resident) | ~13 W | ~110 W (one card) | ~210–230 W (decode) | 43–81 t/s |
| llama.cpp 27B on 2×3060 (tensor+MTP, sanity) | ~13 W | ~200 W | ~310 W | 35 t/s (16K, cold) |

So the pair serves the 27B at **~80% of 3090 decode speed for roughly the same wall draw** — and llama.cpp with tensor split manages the same model at ~2/3 the wall but 1/3 the speed (compute-bound on two consumer cards; the vLLM+MTP path is the one that closes the gap).

### A caution about the experimental vLLM build

The 27B runs use a **locally patched vLLM 0.27.1** (embedding-quant routing + split-KV spec-decode attention; the 3090 production uses the same patch family). It is *flaky* on the 3060 pair: the first long request crashed once in the FlashInfer drafter path, and `--gpu-memory-utilization 0.93` hit a marginal OOM (worker at 11.85/11.87 GiB). It ran reliably at 0.90 with a small warmup request first. Anyone standing this node up should budget for both.

## Part 3 — the retraction: how caching made llama.cpp look 2.7× faster than a 3090

The original campaign first reported "llama.cpp 27B (dual 3060) beats the 3090 vLLM node." Cold re-measurement on both:

| Cold, 16K prompt | Prefill tok/s | Decode t/s |
|---|---:|---:|
| 3090, vLLM + MTP | ~1,000 | **93** |
| dual 3060, llama.cpp + MTP | 618 | **35** |

The phantom came from two cache mechanisms the harness had not neutralized: llama.cpp `--cache-prompt` (the "60× faster prefill" warm rows were cache hits at 48–81K tok/s) and vLLM's automatic prefix caching (a "13K tok/s prefill" row on the 3090 was a warm cache hit, not a regime). A repeat of any fixture is a warm test until you prepend a nonce. Both traps, plus the chunk-rate trap, are now in [docs/benchmarks.md](../docs/benchmarks.md).

## Open items

- **2-bit quality** — IQ2_XXS 35B is fast but untested on real tasks; the 12 GB squeeze is only as good as its quality eval against the IQ4_XS unit.
- **27B node not deployed** — candidate only; current production stays 3090=27B / 3060 pair=35B.
- The 27B-on-3060-pair prefill (560–700 tok/s) lags the 3090 (~1,000) — chunk-size sweep (`max-num-batched-tokens`) not yet done; may be part of the gap.

## Data

- [`reports/assets/2026-08-30-27b-dual3060-ctx-conc.csv`](assets/2026-08-30-27b-dual3060-ctx-conc.csv) — the 27B ladder + concurrency cells (dev run, independent re-run, and wall power per row).
- [`reports/assets/2026-08-30-35b-single3060-ladder.csv`](assets/2026-08-30-35b-single3060-ladder.csv) — 35B decode ladder incl. PCIe RX/TX and power for the residency proof.

## Addendum (same day, later) — exact artifacts behind the 12 GB claim

The "~9 GB" in the setup table above was a quant-weight estimate, not the file. For reproducibility, the exact artifacts:

| Artifact | Size (bytes) | SHA256 | Notes |
|---|---:|---|---|
| `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` (used for all ladder rows) | 10,756,586,464 (10.03 GiB) | `2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef` | the non-MTP build; matches the campaign's 10.76 GB figure |
| `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` (MTP variant) | 11,819,399,456 (11.0 GiB) | `627e2b05f83088448e27861ec392d56efc2b178783cae5adfa4ad2d988441203` | the build behind the 99 t/s MTP row — 1 GiB larger, and the one that hits the VRAM cliff |

So "fits on 12 GB" really means: **10.03 GiB weights + q8_0 KV + compute buffers = 11,307–11,320 MiB measured VRAM on the 12,288 MiB card** — a 900–1,000 MiB margin, not a 3 GB one. Same-named files across the two builds differ in size; a reproduction should verify SHA256 rather than filename.

## References

- 35B production unit (IQ4_XS, dual 3060): [`models/qwen3.6-35b-a3b.md`](../models/qwen3.6-35b-a3b.md), its optimization matrix [`2026-08-27`](2026-08-27-qwen3.6-35b-a3b-dual-3060-optimization.md)
- 35B single-3060 history (CPU-MoE mechanics): [`2026-06-30`](2026-06-30-qwen3.6-35b-a3b-mtp-single-3060.md)
- 27B on 3090: [`models/qwen3.8-27b-rtx3090.md`](../models/qwen3.8-27b-rtx3090.md)
- Method: [`docs/benchmarks.md`](../docs/benchmarks.md)
