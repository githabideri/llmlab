# Qwen3.8-27B (ISTA GSQ-RCO 3-bit) on the single-3060 boxes

**Model:** Qwen3.8-27B — dense ~27.3B, hybrid SSM + full attention (17 of 66 layers), native MTP head
**Quant:** ISTA DASLab [GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) **IQ3_XXS (3.02 bpw)** `-mtp` GGUF (+ the 2-bit IQ2_S sibling, tested and retired)
**Hardware:** 1× RTX 3060 12 GB — two identical boxes: the secondary GPU server (Intel i5-7400, 20 GB LXC limit) and the backup/inference box (Intel i3-9100, 40 GB LXC limit)
**Runtime:** llama.cpp `925e1179` + the **D-CFR** patch (GDN transactional replay — removes the MTP draft's redundant recurrent-state copies so 64K fits at 3-bit), env `LLAMA_GDN_TRANSACTIONAL_REPLAY=1 GGML_OP_OFFLOAD_MIN_BATCH=2`
**Status:** ✅ **On-demand** on both boxes since 2026-09-19 — the user-facing 27B, fronted by each box's model-mux (port 8081, [unit reference](../docs/systemd.md)) alongside the resident 35B. The 98K no-MTP variant is parked in a dormant preset (off-roster); the 2-bit model is retired.
**Supersedes:** the dual-3060 abliterated 27B deployment (dismantled 2026-09-08; see [legacy card](legacy/qwen3.8-27b-uncensored-dual3060.md)) as the 3060-side 27B home

---

## Single-3060 config (live, on-demand, since 2026-09-19)

**Config:** 64K ctx (the MTP ceiling), MTP n-max 2 (accept ~60% at ISTA sampling), `-b 256 -ub 128`, **q4_0/q4_0 KV**, mmproj on CPU (`--mmproj-device none`), ISTA instruct sampling (temp 0.7 / top_p 0.80 / top_k 20 / presence 1.5) with no-think (`chat_template_kwargs: {"enable_thinking": false}`), single slot.

**Measured:** **25–29 t/s decode** (22.5–29.8 on the i3-9100 box), prefill ~300–425 tok/s, MTP mean accepted length ~2.2, **11.7 GB VRAM** (577 MiB headroom), vision working at ~26 t/s. No-MTP at 98K: 15.8–19.8 t/s (parked, off-roster). First load is a ~10 GB no-mmap disk read (tens of seconds to a couple of minutes); a mux-triggered 35B↔27B switch runs ~30–60 s per direction. **Live fleet (llm-hub, 7 days after shipping) corroborates the campaign:** ~24–28 t/s on real agent traffic on the secondary box. (The backup box's 7-day average reads lower — 15.6 t/s — from sparse traffic; the mux's synthesized counters are a coarse approximation, so the controlled campaign numbers stay authoritative.)

**Ceilings (measured, 12 GB card, 3-bit):**

| Regime | Ceiling | Why it stops |
|---|---|---|
| MTP n2, D-CFR build | **64K** | 67K loads (515 MiB headroom) but first-token OOMs in every batch/n-max/KV variant — the decode spike, not steady state |
| MTP n2, production build | 48K | 57K loads, first-decode OOM; 64K doesn't even load |
| no-MTP, production build | **98K** | 100K loads, first-decode OOM at b256 *and* b128 |

KV cost on this hybrid at this build is ~12 MiB/token *per context* (target + draft each) — see [docs/kv-cache-sizing.md](../docs/kv-cache-sizing.md).

**Sampling matters.** The model's own instruct recommendation is temp 0.7 / top_p 0.80 / top_k 20 / presence 1.5. llama.cpp's defaults (1.0 / 0.95 / 0.05 / 0.0) are Qwen *thinking-mode* values: the jinja template then defaults to thinking, a small `max_tokens` reads like an empty reply (the budget goes to `reasoning_content`), and MTP acceptance runs ~84% instead of ~60%.

**Quality:** 3-bit passes a 5-item battery (arithmetic, in-place merge code, fact-check, 16K buried fact, vision) that the 2-bit sibling fails on code and fact-check — the 2-bit "too stupid" feel was quantization-caused, not a sampling artifact. That's why the 3-bit is the shipped model and the 2-bit was retired (file kept on disk).

---

## Known limits

- **One model at a time per card.** The 12 GB card holds exactly one of {35B, 27B, 125B-Flash-Next}; the [model-mux](../docs/systemd.md#llama-mux-port-8081-both-3060-boxes-since-2026-09-19) performs the exclusive switch (unload → VRAM check → load → poll-until-really-loaded). Loading the 27B takes the resident 35B (and its consumers) offline for the switch duration.
- **q8_0 K-cache is not usable on this model/build** — the K quantization pass runs CPU-bound (7.6 t/s prefill, GPU idle); q4_0/q4_0 is the working KV.
- **mmproj must be on CPU at 64K** (GPU OOMs); vision still works.
- **Never put the 35B on the D-CFR build** — it costs the MoE ~85% prefill. The 8090 service is 27B-only.
- **D-CFR build counters freeze** (per-token metrics stall while the GPU decodes; only the decode counter moves) — the mux (v2.2+) is the telemetry authority for this model and synthesizes its `/metrics`; expect hub dashboards to show mux-derived, not engine-derived, numbers here.
- The two boxes' D-CFR builds (identical source) disagree on MTP-context allocation timing (one OOMs a 99-layer single-mode load, the other tolerates it) — stay on the routed config, where the preset sizes the MTP context.
- 100K at 3-bit would need a 24 GB card or RAM KV offload (not measured; LXC RAM limits are 20/40 GB).

## Deployment

Both boxes, same shape:

| Port | Service | What |
|---|---|---|
| 8080 | `llama-server.service` (router, production 925 build) | 35B-MTP resident at boot + the dormant 98K 3-bit section |
| 8090 | `llama-dcfr.service` (D-CFR build, 27B-only) | **3-bit 64K MTP — the user-facing 27B**, on-demand |
| 8081 | `llama-mux.service` | single endpoint per card; exclusive-GPU switching + telemetry authority for the 27B |

Full unit reference: [docs/systemd.md](../docs/systemd.md). The campaign that established all of the above: [reports/2026-09-18-ista-3bit-27b-single-3060.md](../reports/2026-09-18-ista-3bit-27b-single-3060.md).

## Links

- 3090 production 27B (vLLM): [qwen3.8-27b-rtx3090.md](qwen3.8-27b-rtx3090.md)
- Dual-3060 27B (vLLM TP2, frozen): [2026-08-30 report](../reports/2026-08-30-dual-3060-35b-squeeze-27b-node.md)
- 35B on the same cards: [qwen3.6-35b-a3b.md](qwen3.6-35b-a3b.md) · 125B on the backup box: [qwen3.8-flash-next.md](qwen3.8-flash-next.md)
- D-CFR patch: [github.com/kadenball/qwen38-27b-rtx3060-dcfr](https://github.com/kadenball/qwen38-27b-rtx3060-dcfr)
