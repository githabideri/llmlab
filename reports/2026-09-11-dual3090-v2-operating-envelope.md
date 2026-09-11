# Dual-3090 v2 Campaign: Operating Envelope — MTP Knee, Prefill Interference, TP1 Parity (2026-09-11)

**Status: complete.** Night run on the dual-3090 GPU server (Ryzen 5 5600X, 2× RTX 3090,
250 W/card, no NVLink), cloud-model-orchestrated; the runner process owned all
orchestration. Window 01:16–03:24 CEST; the box **held all night** (the first unattended
campaign since the two 09-08/09-10 unexplained crashes — the kdump + heartbeat +
auto-restore stack stood by unused; see the [hardware doc](../docs/hardware/gpu-server.md)).

## Goal

Answer the operating-envelope questions the 09-10 attribution campaign left open:

1. **Where is the MTP knee** on the promoted 8192-token scheduler profile? (prod runs k=3)
2. **Does a big prefill stall running decodes?** (the 8192-vs-2048 decision was pending
   this number)
3. **Is one 3090 freeable?** (TP1 vs TP2 at single-user load)
4. **Agent-session behavior:** prefix-cache hits/misses at 32K, concurrency overlap proof.

## Setup

- Qwen3.8-27B-W4A16-AutoRound, vLLM 0.28.0 (same patched stack as production),
  262K context, fp8 KV, prefix caching, vision on
- Block A: four throwaway TP2 instances at MTP k = 0/2/3/4 (8192 batched tokens each),
  canary-gated, ports 18020 — production untouched
- Block B: the **live production endpoint** (TP2, k=3, 8192) — QOS interference,
  concurrency, 16K decode, 32K prefix cells; production never stopped
- Block C: one TP1 instance (single 3090) for the freeable-card question
- Client: custom SSE harness (this repo, `bench-vllm.py`) — four wall-clock timestamps per
  request, strictly separate TTFT / E2E / tokens/s / TPOT / ITL, exact usage-based token
  closure (±2 % prompt gate), and a concurrency **overlap proof**: a cell is
  `invalid-conc` unless every request pair provably decoded concurrently
- Medians of 3 reps, unique nonces (cold), IQR reported

## Results

### MTP envelope (16K ctx → 1024 out, 8192 batched)

| k (speculative tokens) | median decode (t/s) | IQR | TTFT (s) |
|---:|---:|---:|---:|
| 0 | 67.1 | 1.9 | ~15.6 |
| 2 | 113.4 | 0.3 | ~15.5 |
| **3 (production)** | **151.1** | 8.6 | ~15.4 |
| 4 | 167.6 | 1.1 | ~15.4 |

**The knee is at k=3.** k0→2 = 1.69×, k2→3 = 1.33×, k3→4 = **1.11×** — k=4 buys +11% for
one more speculative token (more VRAM/power per step). **Production already runs the
optimal depth; nothing was changed.**

### Prefill-vs-decode interference (the 8192 question, measured on the live endpoint)

An interactive decode stream (16K ctx, 20K-token output) ran while two 64K prefills were
injected at +30 s and +60 s:

| decode inter-token latency | before | during | after |
|---|---:|---:|---:|
| ITL (ms) | 25.8 | **39.2** | 26.0 |

**Running agent streams lose ~52% of their token rate while a 64K prefill is in flight,
and recover fully afterwards.** The injected prefills themselves: first one 78.6 s to
first token (it was queued behind the live decode), second 115.2 s. Whether this is
specific to the 8192-token budget (vs the old 2048) is the next experiment — the passive
48 h monitoring of the 8192 profile was due to land around this report's date, and a
2048-vs-8192 A/B follows if the owner reverts.

### Single-stream decode & prefix caching (live endpoint)

- 16K → 1024: **147.8 t/s** median (147.0–148.1, IQR 1.1), TTFT 16.7 s
- 32K prefix profile: cold 36.5 s TTFT; a **second request with the same 32K prefix after
  a short gap: 1.22 s (~30× win)**; an immediate re-send and a different prompt both paid
  the full 36–37 s. The cache win is real but **timing-sensitive** (the near-immediate
  re-send missed).

### TP1 vs TP2 (is a card freeable?)

Single-3090 instance, same config, 16K → 1024: **149.8 t/s** vs TP2's 147.8 t/s —
**parity at single-user load**. The second 3090 adds nothing for one user at this model
size; **one 3090 can be released for a second service.**

### Concurrency (2/4/8 on the live endpoint)

All three rows came back `invalid-conc` **by design**: the cells requested 512-token
outputs (~4 s streams) under a 30 s minimum-overlap gate — a self-contradiction in the
matrix, which the client's honesty gate surfaced instead of reporting fake overlap
numbers. Aggregates for reference only (C2/C4/C8 ≈ 135/173/192 t/s aggregate, 114/122/59
t/s per-request medians). The cells were redesigned (2048-token outputs, 8 s gate) for
the next window.

### Power (per-GPU, dmon sampling during the cells)

Decode-heavy cells averaged **171–185 W per GPU** (peaks at the 250 W cap during prefill
bursts); the box idles at ~93 W. A full wall-power (smart-plug) integration per phase is
in the private dataset.

## Notes (honest record of what the run itself found and fixed)

- The client had two data-quality bugs (this vLLM build omits `usage` on the final stream
  chunk; large payloads overflowed the runner's manifest-embedding) and the TP1 cell a
  contract bug (wrong speculative-mode value for the target start script). The
  cloud-model orchestrator fixed them in the field under an explicit dev-phase mandate
  (log before/after, selftest-proof, never change what is measured), re-ran the affected
  cells, and left a full change log. This is a development-phase run: the strict
  "hands off the scripts" regime is reserved for once the package survives clean runs
  unmodified.
- The in-campaign QOS row had a timing gap (the 2048-token decode finished before the
  injections); the valid numbers above come from a supplementary standalone QOS run
  against the same live endpoint.
- One detail: the 09-10 and 09-11 manifests share a results directory (append-only);
  per-window separation is a v2.1 item.

## Conclusion

1. **MTP k=3 is the knee** — production is at the optimum; k=4 is not worth it.
2. **Big prefills stall running decodes ~52% on the 8192 profile** — the central
   interference number for the batch-budget decision (and for any multi-tenant use of
   this endpoint).
3. **TP1 ≈ TP2 at single-user load** — one 3090 is freeable for a second service.
4. **30× prefix-cache win at 32K, timing-sensitive** — exactly the agent-session pattern.
5. The **reliability stack earned its keep by doing nothing**: first unattended overnight
   campaign after two unexplained host crashes, zero intervention needed.

Related: [2026-09-10 attribution campaign](2026-09-10-dual3090-overnight-campaign.md) ·
[model card](../models/qwen3.8-27b-rtx3090.md) · [hardware profile](../docs/hardware/gpu-server.md)
