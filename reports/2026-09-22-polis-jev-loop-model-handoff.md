# Polis Jev-loop: cross-model A/B environment ready, 27B doubt-arbiter stall, vision-Jev feasibility ask

**Date:** 2026-09-22
**Category:** Handoff
**Status:** Environment built and verified (venvs, weights, runner, corpus on a dedicated CPU LXC). Full A/B run and the two open questions below are the ask.

**TL;DR** — The polis mod repo's decision loop (421M Laya noul pre-filter + 27B doubt-arbiter over typed game-state questions) has now accumulated **7 labeled sets (~110 decision rows with oracle actions)** across six harness passes, with a characterized failure mode: the 421M's p-band overlaps between correct and faulty proposals in the *abstract-comparative* regime, and the 27B doubt-arbiter stalls on fault rows in a prompt-resistant way. A **CPU A/B container is provisioned and verified** (4 model venvs: Laya 421M, Decider 2B, SemIf 4B, NanoJev 0.6B; weights downloaded; batch runner + corpus in place) to test whether any of the 2026-09-20 open Jev-family models beats 421M on *our* decision class. Additionally: **decider-2b-vision** (same project) is a candidate for a vision-Jev on the 2×2 GB Pascal box — feasibility question for the GPU bench side.

## 1. What the polis loop is (one paragraph)

A bot in VintageStory executes a mission (travel → act → return). Each step, the harness serializes the bot's state to a structured text block (task, phase, facts like `marker_present=yes`, carrying, last_action, proposed action). Two model tiers sit on it:

- **Reflex (fast, every step):** Laya 421M answers a `noul` — "does the proposed action match the current phase?" — at τ=0.35 (calibrated on the labeled sets). ~1.1 s per call on the 2-core CPU box.
- **Doubt-arbiter (slow, on noul "no" or low confidence):** 27B (Qwen3.8 on the vLLM node, thinking off, ~120 ms) answers a `choice` over the mission action set.

Measured behavior so far: missions complete 2/2 with the loop (25 s, 4 injected faults, 2 corrected); the reflex short-circuits 6/10 steps in the pass-5 set. The decision-class findings (concrete-observable vs abstract-comparative regimes, threshold drift, p-band overlap 0.35–0.48 correct vs 0.32–0.43 faulty) are documented in the mod repo's report `docs/reports/2026-09-22-bot-cargo-and-decision-harness.md` and the openjev use-case `polis-action-noul`.

## 2. The A/B environment (built and verified)

Dedicated **CPU LXC** (4 cores / 4 GB, unprivileged, on the PVE host with the polis container):

| Model | venv | Weights | Interface used |
|---|---|---|---|
| Laya 421M (production baseline) | `vlaya` (py 3.13, torch 2.14) | convaiinnovations/laya | `Router().predict(state, {q: {t: "choice", crit: {…}}})` — same wire format as the openjev endpoint |
| Decider 2B (Qwen3.5-2B-Base) | `vdec` (py 3.12, forced by numpy<2 pin) | Mapika/decider-2b | `Decider(path).decide(state, [{question, options}])` → per-option probs, one forward pass |
| SemIf 4B (frozen Qwen3.5-4B, logit readout) | `vsem` | Qwen/Qwen3.5-4B | `score(model, tokenizer, row, metadata)` — chat template + A/B/C… slot tokens |
| NanoJev 0.6B (Qwen-based parallel decisions) | `vnano` | C-Tianyu/NanoJev | **secondary** — its question schema is game-specialized (ASCII window + 4 boolean questions); adapter not yet written |

The batch runner (`ab-runner.py`) feeds every labeled row as a **choice question over the mission action set** (oracle = ground truth) and reports top-1 accuracy, p(oracle), multiclass Brier, latency, and calibration split (mean confidence on correct vs incorrect). The 7 labeled sets from the mod repo's `data/` are staged in the container; usage is in the file header.

**The ask (llmlab side):** run the A/B — `laya`, `decider`, `semif` on the full corpus (decider 2B int8 on 4 CPU cores: expect ~1–2 s/question, ~110 rows ≈ 30–40 min, fine detached). Deliverable: the per-model table, to answer: *does a 2B/4B model fix the p-band overlap in the concrete-observable regime (where the 421M is our production reflex), and at what latency/cost?* This decides whether the reflex tier stays 421M (fast, calibrated) with 2B as the arbiter, or whether a 2B model replaces the 421M entirely. A second, deeper question for the decision-classifier docs: our judge rows contain **"better-than-oracle" cases** (27B chose an action the oracle label says is wrong, but the loop later proved the 27B's action was what actually completed the mission) — evidence that the oracle granularity (per-step action labels) is too coarse for fault-injection rows; see §3.

## 3. 27B doubt-arbiter stall (measured, prompt-resistant)

On injected-fault rows (proposed action contradicts the phase), the 27B reliably answers the *safe* action (`wait`) instead of the corrective one, even when the state text explicitly contains the failure evidence. Three prompt reformulations tested (evidence block, explicit "the last action failed" line, diff-since-last-step line) did not change the distribution meaningfully; the loop instead recovered via a **max-stall valve** (force the phase-correct action after N stalls). Interpretation offered to the decision-classifier line: in the abstract-comparative regime the model's conservative prior dominates failure evidence in the state text — i.e. this is the same regime class where Laya inverts (see the noul findings), now observed at the 27B tier. The "better-than-oracle" rows suggest the *label* is the coarser artifact, not always the model.

## 4. Vision-Jev feasibility (GPU bench side)

The polis loop wants a cheap visual check (marker visible? crop still there? has the frame changed at all — the last question was the missing wedge detector on 2026-09-21). Candidate: **decider-2b-vision** (same project as §2's Decider): 2B Qwen3.5-based, same typed wire format, image questions, 59 ms median on a datacenter GPU; int4 ≈ 1.3–1.5 GB → fits **one** of the 2 GB consumer GPUs (sm_61, Pascal, int4/int8 only) that also host the WhisperX service.

Questions: (a) does the Qwen3.5-VL architecture load and run on a Pascal-era CUDA toolchain at all? (b) measured latency for one 768-px image question at int4 on a 2 GB sm_61 card? (c) does it do perceptual frame-diffing when given two frames, or does the "changed?" question need side-by-side prompt layout? Design doc: polis repo `docs/design/vision-jev-2026-09-22.md`.

## 5. Assets & pointers

- Labeled sets (7): polis repo `data/labeled-set-2026-09-22-*.json` (rows: `state_text`, `oracle`, `proposal`, `phase`, per-model ms/probs from the run).
- Openjev use cases: `polis-action-noul`, `polis-harvest-noul` (saved on the local endpoint).
- Runner + corpus on the CPU LXC: `/opt/jevab/` (`ab-runner.py`, `corpus/data/`, `weights/`, per-model venvs).
- Model provenance: the 2026-09-20 "Open Jev Models" roundup (Witteveen) — Laya, NanoJev, Decider, SemIf (ex-OpenJev), Nimble 9B and DiffusionGemma 26B excluded from CPU A/B (too heavy), JevBench at benchmarkheaven.com for external baselines.
- Polis loop report: `docs/reports/2026-09-22-bot-cargo-and-decision-harness.md` (mod repo).

## Addendum 2026-09-22 evening (supersedes §2 status for Laya; hosting change for §2)

**Laya 421M A/B complete** (41-row merged corpus, 27 mine + 14 harvest):
top-1 **mine 63.0% / harvest 0.0%** (overall 41.5%), mean p_oracle 0.335
(mine 0.386, harvest 0.238), Brier 0.641, conf correct 0.481 vs wrong 0.378,
**10.4 s/question** on the 4-core CPU box. Two independent disqualifiers for
the live-loop role: (1) domain shift — Laya was trained in our loop on
mine-format states and has never seen harvest-format ones (0/14, as expected
for 421M out-of-distribution); (2) latency — 10 s/decision is two orders
above the loop's 1–2 s cadence. Implication for any general-purpose Jev:
one classifier per state schema, or training on the merged corpus.

**Hosting change.** Decider 2B and Qwen3.5-4B (SemIf base) do not fit the
4-core/4G A/B box (bf16 2B ≈ 4G weights alone; 4B ≈ 8G) and Decider's
gated-delta-net kernels are triton/GPU-only. Both now run on the 2x24G box,
**CPU bf16** (64G RAM, 8 cores); for Decider the `fla` import is suppressed
so transformers takes its pure-torch reference GDR path. Same corpus, same
runner, same metrics; latency is per-box, which is the deployment-relevant
number. Weights were still downloading at writing time (slow CDN line).
