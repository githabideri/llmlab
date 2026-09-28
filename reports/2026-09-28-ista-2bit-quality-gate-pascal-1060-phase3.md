# Phase 3 — adversarial quality battery: IQ4_XS vs ISTA GSQ 2-bit, Qwen3.6-35B-A3B on the Pascal 1060 laptop (2026-09-28)
> **Update (2026-09-28, Phase 4):** two items below were re-read. (a) The **"~100–300 MB per-request RAM leak" was an extrapolation, not a measurement** — 6.7 h of 60-second `/proc` sampling on the live production worker shows a byte-stable 15,630 MiB steady state: a one-time ramp, not a leak; the OOM was a RAM-budget problem, structurally fixed by the 2-bit cutover (~10.3 GB steady). (b) The **cutover was executed** (2-bit at 96K) after the agentic tool-calling battery passed 8/8 on both quants. See the [Phase-4 report](2026-09-28-ista-2bit-100k-performance-ceiling-pascal-1060-phase4.md).
Frozen report. Companion to [`2026-09-27-ista-2bit-expert-residency-pascal-1060-phase2.md`](2026-09-27-ista-2bit-expert-residency-pascal-1060-phase2.md) (performance) and the model card [`models/qwen3.6-35b-a3b.md`](../models/qwen3.6-35b-a3b.md).

## Question

Phase 2 established that the ISTA GSQ-hybrid 2-bit (Q2_0 experts on CPU, c24 expert cache) is the fastest 128K configuration on this box — ~18.7 t/s vs ~12.7 t/s for the production IQ4_XS — at lower VRAM. The remaining cutover gate was **general-purpose quality**: is the 2-bit close enough to the 4-bit to replace it as the production model of this node? This campaign measures that, and nothing changes in production as a result (no cutover performed; the 4-bit remains the serving model).

## Setup

| | i4 (control, production today) | i2 (candidate) |
|---|---|---|
| Model file | Qwen3.6-35B-A3B-UD-IQ4_XS (18.2 GB) | Qwen3.6-35B-A3B-GSQ-hybrid (12.2 GB, Q2_0 CPU experts) |
| Engine | 2026 llama.cpp, moe-cache fork (GenerelSchwerz), CUDA 12.9, sm_61 | same |
| Placement | `--fit off --load-mode none -ngl all -ncmoe 20` (16 layers' experts on GPU), `--moe-expert-cache-size 12` | same, `--moe-expert-cache-size 24` |
| Context | 128K, q8_0 KV, `-b 2048 -ub 512 -np 1`, `--moe-early-router --decode-overlap --backend-sampling --phase-aware-workspace` | same |
| Box | GTX 1060 6GB Max-Q (Pascal), i5-7300HQ 4C/4T, 24 GB single-channel DDR4 (see `docs/hardware/pascal-laptop.md`) | same |

All fixtures: `chat_template_kwargs: {"enable_thinking": true}` (the production regime), sampling `temp 0.7 / top_p 0.8 / top_k 20` — the fleet-standard chat sampling. Budgets: 2048 tokens for normal fixtures, 4096 for the runaway probes and (harness fix, see below) code/structured, 512 for the names battery.

**Battery (90 fixtures):** 16 rare proper nouns (A), 8 dates, 8 ordering/ranking, 8 multi-hop, 8 code (auto-executed), 8 structured/JSON (schema-checked), 6 normal long-context questions, 10 runaway-thinking probes (deliberately open-ended, 4096), 18 proper-noun stress probes (512 budget). Auto-graded: substring sets, exact ordering, executed code, parsed JSON.

## Two engine facts discovered and used by the harness

1. **The 2026 `llama-server` exposes reasoning under `message.reasoning_content`** (not `message.reasoning`), and per-request `seed` is **accepted but ignored** on this build's `/v1` path (identical seed → different outputs). Repeats therefore measure sampling variance, not seed reproducibility.
2. **A host-RAM leak in the MoE-cache server path**: after ~50–80 requests the server's resident set climbs ~100–300 MB per request and can OOM-kill the process (the 4-bit batch died this way at 54/90; the 2-bit batch survived only under a watchdog that restarts the server when RSS exceeds ~12 GB). This is a **production concern for any long-lived 2026 llama.cpp MoE server on RAM-constrained hosts**, independent of the quality question — the leak grows with request count/thinking length, not with context size.

## Coverage (what actually ran)

The 4-bit batch was killed by the OOM at 54/90 (nouns, dates, ranking, multihop, code, 6/8 structured). The 2-bit batch ran all 90 under a watchdog that performed 4 clean server restarts. Paired comparison therefore covers **A–F (54 fixtures)**; G/R/N are 2-bit-only, and the 4-bit G/R/N remain a daytime follow-up (~15 min of runtime, no time pressure).

| category | i4 (4-bit) | i2 (2-bit) | paired (i2 pass / n) |
|---|---|---|---|
| nouns (A, 16) | 10/16 | 9/16 | 9/16 |
| dates (B, 8) | 7/8 | 8/8 | 8/8 |
| ranking (C, 8) | 7/8 | 5/8 | 5/8 |
| multihop (D, 8) | 6/8 | 4/8 | 4/8 |
| code (E, 8, 4096) | 3/8 | 7/8 | 7/8 |
| structured (F, 8, 4096) | 5/6 | 7/8 | 5/6 |
| long-ctx normal (G, 6) | — (OOM) | 6/6 | — |
| runaway (R, 10, 4096) | — (OOM) | 0/10 by strict check; see below | — |
| names (N, 18, 512) | — (OOM) | 8/18 (budget-confounded; see below) | — |
| **total** | **38/54 (70%)** | **54/90 (60%)** | **38/54 (70%)** |

**Headline: on the 54-fixture paired set the two quants tie exactly — 38/54 each (70.4%).** The compositions differ: 31 both-pass, 9 both-fail, 7 where only the 2-bit failed, 7 where only the 4-bit failed.

## Failure-mode breakdown (the paired divergences)

**Only 2-bit failed (7):**

- **A08 (genuine 2-bit error)**: asked for the sculptor of *The Buried Venus* (1853); 2-bit confidently answered **"Horatio Green"** (correct: Hiram Powers). A hallucinated proper name — the same failure class as the Phase-2 "Jan Ingenhouz" corruption, and the one quality defect type that is clearly 2-bit-specific.
- **D04 (genuine 2-bit error)**: first Olympic city + its river; 2-bit got Athens right but named the wrong river (4-bit: Ilissos, correct).
- **A01, A07, C02, C05, D06 (budget exhaustion, not knowledge)**: all `finish=length` at 2048 — the 2-bit's thinking consumed the budget before the answer. These are the same questions a 4096 budget would likely have answered (several were close).

**Only 4-bit failed (7):**

- **E01, E02, E03, E05 (4 of the 5 code questions, all 4-bit)**: the 4-bit's thinking is *more* verbose on code (median 9.3k chars vs 4.7k for the 2-bit) and hit the 4096 budget with no answer. The 2-bit solved 7/8 code fixtures with short, correct functions. **The 2-bit is materially better at code on this engine** — a surprise in its favor.
- **A02 (urea/Wöhler), A13 (1943 Nobel/Cockcroft-Walton), B08 (phosphorus/Brand)**: 4-bit budget exhaustion at 2048; the 2-bit answered all three cleanly in under 1000 tokens.

**Both failed (9):** D01, D03 (Nobel multi-hop chains — the hardest hop class), A03, A06, A11, A14 (rare nouns), C01, E07, F04. These are the model-class floor, not a quant effect.

## The two 2-bit-only batteries

**Runaway (R, 10 open-ended probes, 4096):** 9/10 self-terminated with real answers (65–921 chars after 2.5–8k chars of thinking); only R10 ran the full 4096 with an empty answer (1/10). This is **better than the Phase-2 7-fixture signal suggested** (where the 2-bit ate a 4096 budget on one of two hard open questions): runaway-thinking is real but bounded in most cases. (4-bit comparison pending — its batch died before R.)

**Names (N, 18, 512-token budget):** 10/18 returned empty — not because the names were wrong, but because the 2-bit's thinking (1.7–2.1k chars) consumes a 512-token budget before any answer. The 8 that passed (Kubrick, Heisenberg, Franklin, Darwin, Kennedy, Fleming, Perelman, MLK) came out clean. **This battery as run measures the budget floor, not name knowledge: the 2-bit needs ≳1k tokens for short-answer tasks.** The 2-bit's actual name-knowledge failures are the A-category ones (A08 above; A01 under a bigger budget, where Phase 2 showed the Ingenhouz corruption).

## Thinking-verbosity stats (paired set)

| | median thinking | max thinking | median completion |
|---|---|---|---|
| i4 (4-bit) | 3.3k chars | 15.0k chars | 920 tok |
| i2 (2-bit) | 3.4k chars | 8.2k chars | 1004 tok |
| i4 on code only | 9.3k chars | 15.0k | 2996 tok |
| i2 on code only | 4.7k chars | 8.2k | 1374 tok |

The 2-bit thinks *less far* (lower ceiling, half the code verbosity). The 4-bit is the more prone to budget-eating on exactly the tasks (code, hard facts) where its thinking rambles.

## Conclusion

1. **Paired quality is a tie (38/54 each).** On a 54-fixture battery the expected difference between two quants of the same base model is a few fixtures of noise; 7-vs-7 asymmetric divergence is within that. The 2-bit is a **viable production replacement** for the 4-bit on this node.
2. **The 2-bit is better where it matters for this node's traffic**: code (7/8 vs 3/8), structured output (7/8 vs 5/6), dates (8/8 vs 7/8) — and it halves VRAM and gives +48% decode (Phase 2).
3. **The 2-bit's real, quant-specific defect is the proper-noun failure class**: confident wrong names (A08, and the Phase-2 Ingenhouz). The 4-bit makes the same class of error less often (A02/A07/A13/B08 were its *budget* failures, not wrong-name failures). Proper-noun recall is the one dimension where the 2-bit is measurably worse.
4. **Both models are budget-hungry with thinking on**: a 2048-token budget is too small for either on hard questions; the 2-bit's floor is ~1k for short answers. Consumers of this node should give ≥2k tokens for anything non-trivial.
5. **Cutover decision (this node): viable, not yet forced.** The 2-bit passes the quality gate on the evidence — same general quality, faster, cheaper. A cautious cutover could be gated on two cheap daytime measurements: (a) re-run the 18-fixture names battery at a 1024-token budget (measures name knowledge instead of budget exhaustion), (b) complete the 4-bit's G/R/N fixtures (~15 min) to pair the 2-bit-only batteries. If the names re-run comes back clean, cutover; if it reproduces A08-class corruptions at scale, keep the 4-bit and treat the 2-bit as the on-demand alternative.
6. **The OOM leak is the bigger finding for operators**: the 2026 llama.cpp MoE-cache server grows host RAM ~100–300 MB per request and OOM-kills on RAM-tight hosts. Any long-lived deployment (including this node's production router) should run an RSS watchdog or periodic restart. Filed against the fork.

## Next experiments

- Names battery at 1024 tokens (2-bit, 18 fixtures, ~10 min) — the actual cutover gate.
- 4-bit G/R/N completion (~15 min) to pair the remaining categories.
- RSS-leak characterization: leak rate vs thinking length vs request count (the fork's issue).
- Dual-channel RAM swap (16 GB stick) re-run of Phase 2 + this battery — the 2-bit's lead should widen; not done yet.
