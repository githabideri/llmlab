# Qwen3.8-Flash-Next on a single 3060: the MoE hot-cache recipe, measured on the backup box

**Date:** 2026-09-20
**Box:** backup / inference box — Intel i3-9100 (4 cores / 4 threads, 3.6–4.2 GHz, DDR4-2400 2-channel ≈ 38 GB/s), single RTX 3060 12 GB (PCIe), 40 GB host RAM (LXC soft limit), NVMe, 1 GbE
**Model:** Qwen3.8-Flash-Next ("Qwen4Exp", 125 B total / ~6 B active + 51 B per-layer PLE table), Unsloth `UD-Q2_K_XL` (3-shard GGUF, 78.9 GB) + `mtp-…-shared-Q8_0` draft (2.8 GB). All four files SHA-256-verified against the copies measured in the [2026-09-13 single-3090 run](2026-09-13-flash-next-single3090-codacus-cache.md).

## Summary

A single 12 GB RTX 3060 plus 40 GB of host RAM serves the full 125B model at **~14.3–14.9 t/s decode** (64K context, warm) using the CPU-experts + GPU hot-expert-cache hybrid. The winning cell is a **64-slot MoE hot cache** (`--moe-cache-slots 64`, ~5.5 GB of the 12 GB card) on the codacus llama.cpp fork (`27c54b4b`, base `b10818`): decode is **+36 %** over the no-cache CPU-experts baseline on this box, and 9.6K-token prompts prefill at **48–53 t/s** (the 40 GB-RAM "full-speed prompt reading" tier — no cliff). An 80-slot cache decodes ~5 t/s faster but **crashes (CUDA OOM) on long-prompt prefill** on a 12 GB card; 64 slots is the safe ceiling. MTP adds nothing measurable (the CPU, not the GPU, is the decode bottleneck here). The model is now an **on-demand roster member** behind the box's model-mux (the 35B stays resident; the mux evicts and reloads in ~60–120 s).

The public reference that motivated this run: theodacus's video (Aug 2026) running the same hybrid on a 5600X + 3060 + 61 GB: ~16.5–17.3 t/s stock, **24.4 t/s with a 48-slot cache**. Our i3-9100 has roughly half that memory bandwidth, and we got proportionally less: 10.7 → 14.5 t/s for the cache.

## Engines

| Engine | Build | Role |
|--------|-------|------|
| A | llama.cpp mainline `b81c99b4` (fitter auto-placement; the 2026-09-02 three-GPU champion commit) | baselines |
| B | codacus fork `27c54b4b` (perf branch; base `b10818`) with `--moe-cache-profile`/`--moe-cache-slots` | tuned config |

Both built in-container (Release, CUDA 13.1, `GGML_CUDA_FA=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_CUDA_GRAPHS=ON`). The box's existing production build (`925e1179`) predates `qwen4exp` support (PR #27742, 2026-08-27) and cannot load the GGUF at all.

The hot cache is profile-driven: 12 generation traces (3 prompt families × 4, 512 tokens each, `llama-moe-trace`) → **307,776 expert-access rows** merged into one CSV (`9b381b2e…`); the fork ranks experts and pins the hottest N slot-groups (85.76 MiB each at Q2) in reserved VRAM.

## Commands (the adopted B64 cell)

```bash
llama-server \
  -m /path/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  --moe-cache-profile /path/q2-merged.csv --moe-cache-slots 64 \
  -ngl 99 --n-cpu-moe 99 --no-sched-async-cpu -t 4 \
  --load-mode mmap -fit off -fa on -ctk q8_0 -ctv q8_0 \
  -c 65536 -np 1 --cache-reuse 256 -b 2048 -ub 512 --jinja
```

`-t 4` = one thread per physical core (the video's key lesson; 8/12 threads regress). Served as a router (`--models-max 1`, preset INI, on-demand load) on a dedicated port, fronted by the box's model-mux for exclusive-GPU switching with the resident 35B and the on-demand 27B.

## Results (medians over warm reps; P1 = short prose, P2 = code; decode = SSE content-chunk rate)

| Cell | Config | Decode t/s | Prefill | VRAM | Note |
|------|--------|-----------|---------|------|------|
| A1 | mainline, **fitter** auto-placement | **11.7** (11.56–11.94) | 226–399 t/s (150 tok) | 10.7 GB | fitter beats hand flags here |
| A2 | mainline, video-official flags (`-ngl 99 -ncmoe 99`) | 10.8 | 112–385 t/s | 5.3 GB | the video's "stock" layout |
| B0 | fork, no cache (S0) | 10.7 | 150–373 t/s | 5.0 GB | matches A2 — CPU-bound floor |
| B1 | fork, **48-slot** cache | P1 10.8 / P2 **12.9** | 9.6K prompt: 48–57 t/s | 9.1 GB | the video's slot count |
| B2 | fork, **80-slot** cache | **15.8** / 14.6 | **OOM at 9.6K prefill** (CUDA `cuMemCreate` failure, server aborts) | 11.8 GB | fastest, **not adoptable** on 12 GB |
| B3 | fork, 48-slot + MTP draft (n-max 1) | 12.4 / 12.8 | — | 9.4 GB | ≈ neutral vs B1 — CPU-bound |
| **B64** | fork, **64-slot + 64K ctx** — **adopted** | **P1 14.5** (12.3–14.9) / **P2 14.3** (13.2–14.0) | 9.6K: 48–53 t/s | 11.4 GB | best decode that survives long prompts |

Cold numbers (reported for honesty): first load after boot ≈ 55–120 s (81.7 GB from NVMe); the first generations right after a 35B↔125B switch run **5–9 t/s** and climb to ~14 as the 81.7 GB PLE/expert working set re-fills the page cache (40 GB RAM holds most of it). TTFT warm: 0.3–0.7 s (short prompt).

**Context ceiling:** 64K is the maximum that loads with the 64-slot cache (128K fails at context creation; the KV is small — q8_0 K/V, ~10 KB/token — but the prefill *workspace* plus cache plus weights must all coexist on 12 GB). Prefill at 40 GB RAM shows no cliff (48–53 t/s at 9.6K), in line with the video's RAM-tier curve (20 GB = cliff, 40 GB = full speed, 61 GB = slightly more).

**Energy (decode, steady state):** GPU **48.1 W** average (1 Hz `dmon`, 4×300-token generations) → ≈ **0.09 Wh per 1K generated tokens, GPU-only** (no wall-plug meter on this box; the i3-9100's CPU power is unmeasured). Compare the three-GPU campaign's 0.5–1.7 Wh/1K across *three* cards.

**Quality:** one sampled completion per configuration — coherent prose, `think` blocks present, no looping/gibberish observed. The fork's hot-cache path is **not bit-identical to the cache-off path** under greedy decoding (established in the 2026-09-13 3090 run; divergence from the first content token, coherent output) — that caveat carries over. No trap-lab run was done here.

## Comparison to prior measurements

| Run | GPU | Cache | Decode |
|-----|-----|-------|--------|
| this report (B64, adopted) | 1× 3060 12 GB (i3-9100, 40 GB RAM) | 64 slots | **14.3–14.9 t/s** |
| theodacus video (reference) | 1× 3060 (5600X, 61 GB) | 48 slots + MTP | 24.4 t/s |
| [2026-09-13](2026-09-13-flash-next-single3090-codacus-cache.md) | 1× 3090 24 GB | S0 / 128 slots | 17–18 / 26.5–27 t/s |
| [2026-09-02](2026-09-02-qwen4exp-flash-next-three-gpu-campaign.md) | 2× 3060 + 3090 | fitter, no hot cache | 30.2 t/s |

The bandwidth scaling is the whole story: 5600X (51 GB/s) → 24.4, i3-9100 (38 GB/s, and it feeds both the CPU-expert path *and* the PLE stream) → 14.5. The 3090's 26.5–27 (128 slots in 24 GB) sits between, as expected.

## Decision

- **Adopted** as an on-demand roster model (64K ctx, 64-slot cache, no MTP): useful for "big-brain" queries that don't need the 35B's speed or the 27B's latency — the box's first 125B-serving path. On-demand only: it takes the whole 12 GB card and the mux's exclusive switch (~60–120 s including the first cold load).
- **Not adopted:** 80-slot (OOM risk on long prompts), MTP (neutral on a CPU-bound card; bit-divergence path), 128K ctx (won't load at 64 slots).
- The 32→40 GB RAM bump (soft LXC limit, 512 MB swap kept) was necessary and is retained: it is what makes the 40 GB "full-speed prefill" tier reachable.

## Reproducibility notes

- Q2_K_XL shard hashes (match the 2026-09-13 set): `a4f3b21e…`, `2e3bf1ee…`, `ec8c1067…` (shards 1–3), MTP `5ff54097…`.
- Profile CSV: 307,776 rows / 13.7 MB, `9b381b2e…`; regenerate with the 12-prompt trace harness (~30 min on this box) if the model or fork changes.
- Decode measurement = one SSE content chunk per generated token under this build's streaming contract (same method as 2026-09-13); the server's own `timings` object cross-checked within ~20 % during mux-path runs.
