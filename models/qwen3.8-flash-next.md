# Qwen3.8-Flash-Next (Qwen4Exp)

**Model:** Qwen3.8-Flash-Next — "Qwen4Exp" MoE: 125 B total, ~6 B active per token, 48 layers, plus a 51 B-parameter per-layer lookup table (PLE; `per_layer_token_embd.weight`) that is streamed per-token from host memory  
**Tested Quantization:** Unsloth `UD-Q2_K_XL` (3-shard GGUF, 78.9 GB) + `mtp-…-shared-Q8_0` draft (2.8 GB). GGUF **v3** header format — only llama.cpp at/after PR #27742 (`6c84c7d5`) can load it.  
**Status:** ✅ **Served on the backup/inference box (single 3060) — engine swapped to Strata v0.1.41 on 2026-10-08** (IQ2_XS, 98K, 19 GiB resident budget; the 35B service is offline for the duration of the Strata window, they cannot share the 42 GiB cgroup). The llama.cpp config below was the live one since 2026-09-20 and is now a frozen record. The three-GPU config further below remains a **frozen test record**.

---

## Strata v0.1.41 production config (live, since 2026-10-08)

**Hardware:** backup/inference box — i3-9100 (4C/4T), RTX 3060 12 GB, 46.8 GB RAM, no swap, NVMe, 42 GiB hard cgroup limit. **Engine:** [Strata](https://github.com/Niko1221/Strata) v0.1.41 (`fb58e0d`, CUDA 13.1): experts **mmap'd from the NVMe** with a **19 GiB page-locked resident budget** (#1250), 3748-slot / 5.04 GiB auto GPU expert cache, PLE table SSD-streamed, in-VRAM MTP draft, KV int8, 98K context. **Config:** `--spec 4 --spec-min-p 0.5 --kv int8 --mmap-experts --resident-budget-gib 19 --expert-cache auto --prefill auto --max-context 98304`, env `STRATA_STAGE_PIN=0 STRATA_PREFILL_CPU_SHARE=0`, own unit (port 8181), `Restart=on-failure`.

**Measured (2026-10-08, medians of 3; n=5 confirmations pending):** resident ladder **16 GiB 19.81 → 17 GiB 21.82 → 19 GiB 22.85 → 21 GiB 22.01 t/s** @16K (32K geometry; peak at 19 GiB) · **23.02 t/s @90K / 17.72 @16K** (98K geometry, +24.5% over the in-window 16 GiB re-run) · prefill 2.2–4.6 ms/tok warm, **181.8 s true-cold 98K** (faster than the 16 GiB engine's 240.7 s — the 19th GiB changes the cold fault-in pattern) · first-token overhead ~2.5 s on short interactive requests. Canary battery byte-identical to the llama.cpp-era baseline at every arm; zero OOM/allocfail (cgroup max 38.9 GiB).

**Reading `/metrics`:** the live reading is `spec: 6 / mtp_max: 4 / lookup: 3` — the reported `spec` is the effective verify window, not the flag (`--spec 4` plus the default drafter widening), so this is not config drift. Field semantics: [docs/strata.md](../docs/strata.md).

**Window semantics:** while this unit runs, the box's 35B service is **offline** (42 GiB cgroup — co-residency impossible; same exclusivity as the old model-mux switch, now a design decision). Stop the unit + start the 35B to give the box back. MTP is mandatory in v0.1.41 for this quant (`--spec 0` is refused; spec 2 measures 15.03 vs 19.81 for spec 4 in serve mode).

Full campaign: [`reports/2026-10-08-strata-v0141-single-3060.md`](../reports/2026-10-08-strata-v0141-single-3060.md); predecessor: [`reports/2026-10-08-strata-v0139-single-3060.md`](../reports/2026-10-08-strata-v0139-single-3060.md).

---

## llama.cpp single-3060 config (live 2026-09-20 → 2026-10-08; frozen record)

**Hardware:** backup/inference box — i3-9100 (4C/4T, DDR4-2400 2-ch ≈ 38 GB/s), RTX 3060 12 GB, 40 GB RAM. **Engine:** codacus llama.cpp fork `27c54b4b` (base `b10818`). **Config:** CPU experts + **64-slot MoE hot cache** (profile: 12 traces → 307,776 access rows) + 64K ctx (the card's ceiling at this slot count; 128K won't load, 80 slots OOM on long-prompt prefill) + `-t 4` (one thread per core), no MTP (measured neutral on this CPU-bound card).

**Measured:** **14.3–14.9 t/s decode** warm (vs 10.7 no-cache on this box; the video reference: 24.4 on a 5600X/61 GB), **48–53 t/s** at 9.6K-token prefill (40 GB-RAM tier, no cliff), GPU 48 W in decode (≈0.09 Wh/1K tokens GPU-only). First load ~1–2 min (81.7 GB from NVMe); the first minutes after a 35B↔125B mux switch run 5–9 t/s until the PLE/expert page cache re-fills. Text-only (no vision). On-demand only: takes the whole 12 GB card; the box's model-mux handles the exclusive switch with the resident 35B.

Full numbers, cell matrix, and the OOM/crash findings: [`reports/2026-09-20-flash-next-single-3060-moe-cache-backup.md`](../reports/2026-09-20-flash-next-single-3060-moe-cache-backup.md). The fork's hot-cache path is not bit-identical to cache-off under greedy decoding (see the 2026-09-13 single-3090 report).

**Update (2026-09-23):** the pinned ubatch path (`GGML_CUDA_REGISTER_HOST=1`) is **not runnable on this host** — it page-locks mmap'd expert pages, pinning the CT cgroup at ~40 GB against the 40 GiB soft limit and hard-rebooting the 46 GiB no-swap box ([report](../reports/2026-09-23-flash-next-pinned-path-oom-unpinned-ub-sweep.md)). Unpinned quick test: the prod **ub 512** setting (45.8 t/s cold prefill @9.6K) already beats ub 1024 (26.1 t/s) on this box, so there is no ub gain to harvest here; fast-path numbers need a bigger-RAM host.

---

## Three-GPU campaign (frozen record)

**Hardware:** GPU server, 3 cards: RTX 3060 12 GB (chipset, PCIe 4.0 x4), RTX 3060 12 GB (CPU, x8), RTX 3090 24 GB (CPU, PCIe 4.0 x8 in the 3-GPU config). **Runtime:** llama.cpp master `b81c99b4` (fitter auto-placement), PR #28136 build retained for direct-read PLE.

## Quick Facts

| Parameter | Value |
|-----------|-------|
| **Architecture** | `qwen4exp` (MoE, per-layer PLE lookup, SSM/hyper-connection/indexer subsystems in the GGUF) |
| **Parameters** | 125 B total / ~6 B active; PLE table 51 B params (RAM/SSD-resident) |
| **Layers** | 48 |
| **GGUF size** | 78.9 GB (UD-Q2_K_XL) — non-PLE weights 45.9 GB (experts 0.89 GB/layer), fits 48.6 GB aggregate VRAM |
| **Context tested** | 32,768 (32K; 128K/262K gated on RAM) |
| **Best measured** | **30.2 t/s sustained decode** (finalist median — the canonical number; 35.3 t/s with MTP is a single clean run, promising, not established) · **~390 prefill t/s under warm-cache conditions only** (34–437 across PLE page-cache states) · ~40 t/s on a prompt-cache hit (a different serving regime, not the model's decode speed) |
| **Per-token traffic (decode)** | PCIe RX 10–50 KB · SSD 2–14 KB · GPU power 68–142 W → **~0.5–1.7 Wh/1K tokens, GPU-only** (dmon sum, no idle subtraction, no wall meter) |
| **Speculative decoding** | MTP: +17 % (single run — promising, not established; soak before any production use). N-gram: pathological on the tested code fixture (0.13 t/s; prose neutral) — do not enable router-wide without content gating. |

---

## Config — fitter auto-placement (the only working path on the `b81c99b4` build)

**KV Cache:** q8_0 / q8_0 · **Batch:** `-b 4096 -ub 1024` · **Threads:** 6 / batch 12 · single slot

```bash
/opt/llama.cpp-master/build/bin/llama-server \
  --device CUDA0,CUDA1,CUDA2 \
  -m /path/to/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  --ctx-size 32768 --parallel 1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 4096 --ubatch-size 1024 \
  --threads 6 --threads-batch 12 \
  --flash-attn on --jinja
# optional MTP: --spec-type draft-mtp --spec-draft-model /path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
```

- **No `--gpu-layers`, no `--tensor-split`, no `--override-tensor`.** The fitter (`common_fit_params`) auto-places: all 48 layers' non-expert weights + shared experts on the GPUs, selective expert spill to RAM, PLE auto-pinned to CPU (VRAM landed at 11.2 / 10.7 / 23.0 GiB).
- On this build, any explicit placement flag **disables** the fitter and the resulting hand-balancing OOMs (12.9 GB demanded from a 12 GB card). `--cpu-moe` / `--n-cpu-moe N` fail the same way.
- Only the **last** `--override-tensor` flag is honored (deprecation: use comma-separated values) — if you must pin tensors, combine them in one flag.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` is required on this box (default FASTEST_FIRST maps cells to the wrong physical cards).
- Fitter VRAM margin knob: `--fit-target <MiB>` (default 1024; 512–3072 made no difference in testing).

## Stage-1 baseline (for reference)

Explicit-flag mode (build `6c84c7d5`): `--gpu-layers 44 --split-mode layer --tensor-split 1,1,2` with `--override-tensor per_layer_token_embd.weight=CPU --load-mode mmap` → 23.6 t/s decode / 72 prefill. Whole-layer spill is the prefill killer; the knee is ngl 44 (3090 nearly full), ngl 48+ OOMs.

## Caveats

- Prefill numbers are **page-cache-state dependent** (34–437 t/s across cache states); a true-cold test needs hypervisor-level `drop_caches`.
- MTP was verified on a single clean 256-token run (+17 %); longer soak recommended before production.
- Energy figures are the sum of the three GPUs' dmon power readings (no wall-plug meter on this box).

## ISTA GSQ-RCO variants (evaluated 2026-09-25, not deployed)

ISTA-DASLab published per-tensor GSQ-RCO quantizations of this model (Q2_0, IQ2_XS, IQ3_XXS). On the single-3060 backup box the Q2_0 variant is **not viable** (no optimized x86 Q2_0 vec-dot kernel in the tested codacus build; VNNI PR open but N/A on Coffee Lake i3-9100 → 2.4 t/s CPU floor) and the IQ2_XS variant reaches **7.1–7.3 t/s** (64-slot cache) — 2× behind the existing UD-Q2_K_XL deployment. The ISTA models' intended target is a machine with enough VRAM for full GPU residency (e.g. dual-3090, 48 GB aggregate). Details: [`reports/2026-09-25-ista-gsq-rcos-single-3060-q2-0-no-x86-simd.md`](../reports/2026-09-25-ista-gsq-rcos-single-3060-q2-0-no-x86-simd.md).

**Update (2026-09-28):** the IQ2_XS variant under the **expert-pool** branch (per-tensor 64-slot balanced pool) reaches **21.1 t/s at 32K** — 4.2× over the codacus baseline and ahead of the UD-Q2_K_XL deployment's 14.3–14.9 on this box — so the "2× behind" statement above applies only to the codacus build, not to expert-pool placement. At depth the picture narrows: **f16 KV lifts 16K decode to 19.2 t/s (+28%) but brings no gain at 64K and OOMs at 96K ctx on 12 GB** (64K works; exact ceiling unmapped — q8_0 remains the deep-context KV), the **sparse-attention path is structurally unreachable for this model's (512, 256) QSA head config** in current upstream, and the primary identified per-token cost at 64K–90K is the **QSA indexer (O(n_kv) per token)**, with dense full-KV attention remaining a possible co-contributor. Details: [`reports/2026-09-28-flash-next-sparse-qsa-f16-kv-single-3060.md`](../reports/2026-09-28-flash-next-sparse-qsa-f16-kv-single-3060.md).

**Update (2026-09-29):** the two load-bearing 09-28 claims were re-adjudicated with n=3 cold/warm controls and attribution runs — and **both change**. (1) The "f16 KV +28% at 16K" was a **cold-vs-warm measurement artifact**: cold-vs-cold, q8 14.3 vs f16 15.1 t/s (~6%, noise); a real but small f16 edge appears only at 64K (12.6 vs ~10), and mixed KV (K=f16/V=q8) is actively harmful (9.0/6.8/4.6 t/s), so **q8_0 stays the default at every depth on this box**. (2) The **QSA indexer is ~0% of decode cost**: ablating its entire score+top-k pass changed nothing at 16K or 90K, and a working **compact-gather decode port** (gather the 2051 selected K/V rows + their visibility values into 2304-row f16 buffers; dense FA on 2304 instead of masked dense over the full n_kv — canary-correct at every depth) also changed nothing: 90K stays 8.1–9.1 t/s. **Decode at depth on this 4-core box is CPU-MoE-bound** — the GPU attention path (indexed, masked-dense, or compact-gather) is off the critical path, which is why the upstream 5×3090 2× compact-gather gain does not reproduce here. Final profiles: 16K ~18.8 (warm), 32K ~14, 90K ~8.5 t/s — all expert-pool 64 + q8_0. Also settled: the LXC runs the 65 GB model at ~39 GB peak RSS (93% of the 42 GB limit, no swap) — overlapping two large loads OOM-kills; and llama-server's single slot keeps an unchanged prefix in VRAM across requests (a 74,749-token repeat cost 4 tokens). Details: [`reports/2026-09-29-flash-next-3060-compact-gather-cpu-moe-bound.md`](../reports/2026-09-29-flash-next-3060-compact-gather-cpu-moe-bound.md). The profile was **promoted to the on-demand production unit the same day** (expert-pool build + IQ2_XS + 96K preset; 54 s cold load, exclusive-GPU auto-switch verified both directions against the resident 35B).

## Links

- **Update (2026-10-08):** engine swapped llama.cpp → **Strata v0.1.41** (see the production section above): IQ2_XS, 98K, 19 GiB resident budget, ~2.7× the deep-context decode of the expert-pool profile (23.02 vs ~8.5 t/s @90K) and ~2.5–5× warm prefill; the llama.cpp config above is the frozen record. [Report](../reports/2026-10-08-strata-v0141-single-3060.md).

- Campaign report: [`reports/2026-09-02-qwen4exp-flash-next-three-gpu-campaign.md`](../reports/2026-09-02-qwen4exp-flash-next-three-gpu-campaign.md)
- Methodology: [`docs/benchmarks.md`](../docs/benchmarks.md), [`docs/multi-gpu-model-placement.md`](../docs/multi-gpu-model-placement.md)
- llama.cpp PR #27742 (Qwen4Exp), PR #28136 (direct-read lazy PLE)
