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

## Addendum 2026-09-24 (supersedes the §2/addendum-09-22 Decider/SemIf hosting and "results pending" status)

**Decider 2B is now the Jev candidate.** Both pending runs completed on a
dedicated 4-core/16G CPU measurement container (no GPU; the vLLM node stays
pure production — bench work no longer sits on it; the 2026-09-22/23 attempt
on it never ran: its weight downloads stalled, and the host was reinstalled
the next day). Weights live on the host's shared model store, not in the
container. Same 41-row corpus, same runner and metrics as the Laya run;
both models ran CPU bf16 on transformers' pure-torch reference
gated-delta-net path, so the latencies below are reference-path numbers.

| model | top-1 | p(oracle) mean | Brier | p(oracle)>0.5: correct / wrong |
|---|---|---|---|---|
| Laya 421M (09-22) | 0.415 | 0.335 | 0.641 | 25/27 · 0/14 |
| **Decider 2B** | **0.707** | **0.655** | **0.426** | **27/29 · 0/12** |
| SemIf 4B | 0.463 | 0.000 | 1.000 | 0/19 · 0/22 |

By mission (Decider): mine top-1 0.815, harvest 0.500 (vs Laya's 0.630 / 0.000).

1. **Decider 2B fixes the p-band overlap** that disqualified Laya for the
   concrete-observable regime (mean p(oracle) 0.870 on correct rows vs 0.136
   on wrong; the >0.5 split is 27/29 vs 0/12). Its 12 remaining errors are
   one systematic habit — choosing `goto_base` where the oracle is
   `goto_target` on travel-phase rows — a single fixable bias, not a
   capacity wall.
2. **SemIf 4B is not a measurement.** The letter-slot readout produced a
   degenerate distribution (p(oracle) exactly 0 on all 41 rows, Brier 1.0,
   all mass on one argmax action). Most plausibly a harness/model-build
   mismatch (or bf16-CPU collapse), not "the 4B is bad" — no claim is made
   about 4B quality. Follow-up: pin the transformers version to the SemIf
   development era, verify the letter-slot mapping against the model's
   tokenizer, fp32 probe.
3. **Latency is the open deployment question.** Reference-path CPU: Decider
   18.4 s/question on the 4-core box (Laya's 10.4 s from the 09-22 addendum
   is the same story — choice-type questions on small CPUs; the loop's live
   noul questions run ~1.1 s on the 2-core reflex box). A per-step reflex
   needs the fast `fla`/`causal_conv1d` kernels on a GPU or a quantized
   GGUF on CPU; that campaign is next before the 2B is called "the Jev".

Caveats carried from the 09-22 addendum: oracle labels are coarse by
design, and this measures choice-type questions only (the live noul gate is
a different question type).

## Addendum 2026-09-24 (second session) — quantization floor measured; first valid 4B

Both open threads from the previous addendum closed on the same 4-core/16G CPU
measurement container.

**Quantization floor for the Decider 2B readout.** The 2B fine-tune was
converted to GGUF (llama.cpp, 2026-09-24) and re-scored on the same 41-row
corpus through the C library: identical prompt construction, full-vocab logit
row at the answer slot, `softmax(logits/1.3)` over the letter tokens.

| weights | top-1 | p(oracle) mean | Brier | s/question (4-core) |
|---|---|---|---|---|
| bf16 torch (reference) | 0.707 | 0.655 | 0.426 | 18.4 |
| **Q8_0 (2.0 GB)** | **0.707** | **0.658** | **0.420** | **3.6** |
| Q4_K_M (1.27 GB) | 0.537 | 0.545 | 0.580 | 4.1 |

8-bit is numerically indistinguishable from bf16 on this readout (Δtop-1 = 0,
Δp(oracle) = +0.002) while running 5.2× faster — the Decider authors'
"quantization is below evaluation noise" claim holds on CPU at 8-bit, and
8-bit is therefore the quantization floor. 4-bit visibly degrades this
fine-grained letter-logit contrast (−17 pp top-1) even though 4-bit is
harmless for ordinary text generation: this readout is more quantization
sensitive than generation. One side effect of 8-bit: on the 12
systematically-wrong rows (the travel-phase `goto_base` bias), the *wrong*
choice becomes confident (0.75) where bf16 was flat (0.14); top-1 and Brier
are unchanged, so threshold-on-p(oracle) decisions are unaffected, but
"confidence in the choice" is not stable across quantizations. Conversion
gotcha for anyone replicating: the Qwen3.5 GGUF converter assumes MTP draft
tensors and must be given `--no-mtp` for this MTP-less fine-tune, or it writes
a phantom 25th block and the file fails to load.

**SemIf 4B: first valid measurement — the previous degeneracy was a runtime
artifact.** Running the same frozen 4B through its own llama.cpp backend
(fp32-accumulated decode, not the bf16-torch path that collapsed) gives a
healthy readout: 99.3 % of the vocabulary softmax mass lands on the four
letter answer slots, p(oracle) 0.298 (was exactly 0), conf right 0.544 vs
wrong 0.535. The judgment itself is genuinely weak: top-1 **0.463** (chance
0.25) and p(oracle) 0.298 — the 4×-parameter frozen base is *below* the 2B
task-tuned Decider (0.707 / 0.658) on this readout, at 3× the per-question
CPU latency. The "bigger frozen model" leg of the A/B is now answered with a
valid measurement: for this narrow one-pass decision, the tuned small model
wins.

**Deployment outlook.** On the 4-core measurement box, Q8_0 runs 3.6 s per
question — a 5.2× speedup over the bf16 reference but still short of a
per-step reflex. The remaining tiers on the menu: a 12-thread desktop CPU
(expect roughly a further halving) and a 12 GB consumer GPU with the
authors' FP8 + CUDA-graph path (their B300 numbers: 3.2 ms p50; on consumer
hardware expect 10–30 ms). The reflex-layer wiring (Decider Q8 in the
two-tier loop) is the next experiment; the box choice follows whichever tier
is measured first.

## Addendum 2026-09-25 (3rd session): deployed on a 12 GB consumer GPU; the reflex measures ~290 ms

**Placement.** The Decider now runs on the **12 GB consumer GPU (RTX 3060) box
behind the inference hub** — the box's model card, explicitly *not* the 24 GB
production pair (which serves the 27B vLLM production service) and not the
second 3060 box (owner's preference). The Q8_0 GGUF sits in that box's shared
model store; in the model router it is the third model alongside two
27B/35B models, with automatic load-on-demand and eviction (load ≈ 6 s,
evict ≈ 4 s — the 2B and the 35B don't both fit the 12 GB card, so they
alternate). The hub card and its Prometheus series picked the model up without
any new scrape targets.

**The HTTP readout protocol (the non-obvious part).** On llama.cpp main
(build 925e1179), the OpenAI `top_logprobs` body key only feeds the *chat*
path; on `/v1/completions` the raw `n_probs` body key must be sent. With
`logprobs: true, n_probs: 8192, max_tokens: 1` the server returns the **raw
full-vocabulary T=1 softmax** for the final slot (partial-sort over the whole
row, no temperature, no sampler filtering). The Decider's T=1.3 readout is
recovered exactly from it:

    p_T(i) = p_1(i)^(1/1.3) / Σ_j p_1(j)^(1/1.3)      (over the option letters)

(the partition-function constant cancels in the ratio) — so a stock
llama.cpp server is a fully faithful Decider endpoint; no custom in-process
client is needed in production.

**Validation.** The same 41-row corpus re-run over HTTP/CUDA against the
C-API/CPU Q8 reference: per-row Δp(oracle) 0.001–0.03 (one outlier 0.086, a
row both runs chose correctly); top-1 identical on all 41 rows;
conf-correct/conf-wrong 0.861/0.134 (CPU: 0.856/0.141). **Q8 on the GPU is
the same floor as Q8 on the CPU** — kernel/float noise, not a new variable.

**Latency — the reflex is real.** ~**290 ms per question end-to-end** from a
LAN client (RTT + ~70–100 ms prefill for the 130-token harness prompt): ~12×
faster than the 4-core CPU Q8 run and 4× inside the ≪1 s target. The
remaining work on the two-tier loop is wiring, not science: Decider readout as
the per-step reflex, the 421M noul as the cheap veto, the 27B as the
escalation judge. Caveat for anyone replicating on a shared 12 GB card: the
2B is *exclusive* with the 35B — a request for the other model pays a one-time
~10–60 s switch.

## Addendum 2026-09-25 (fourth session, overnight): the three-tier loop runs live

The wiring finished. The Polis testbed game now runs a per-step decision
loop with three tiers, cheapest first:

    policy proposes an action
      -> 421M noul pre-filter (anchored yes/no; ~1.4 s on 2 CPU cores)
             no  -> 27B doubt-arbiter (thinking off, ~270 ms on the GPU pair)
             yes -> 2B Decider choice readout (p over the action set)
                    p < tau_dec              -> 27B
                    p >= tau_dec and noul >= tau_strong -> execute (short-circuit)
                    p >= tau_dec but noul < tau_strong  -> 27B  (the strong-gate)

**The strong-gate is the interesting addition.** A first live run exposed a
*consensus hole*: the injected skip-goal fault got noul-yes at 0.37 (over the
0.35 veto threshold) **and** the Decider's known travel-phase bias at
p(goto_base)=0.76 — two "no doubt" signals agreeing on a confidently wrong
action, with the 27B never consulted. Both p's are individually "healthy"
bands; the failure is that no tier asked the expensive one. Making the 27B
run on *borderline* consensus (noul < 0.6) as well as on outright doubt
closes it, at the cost of more 27B calls. The 27B at ~270 ms (thinking off)
is cheap enough that this is a sensible default; the short-circuit rate is a
calibration target, not a safety property. This is a general property of
cascades with independent biases: **gating must be on the evidence of doubt
in each tier, not on each tier's self-reported confidence.**

**Measured (same fixture as the A/B corpus: mine and harvest missions,
8-block targets, fault injection at chosen phases):**

| mission | steps | wall | notes |
|---|---|---|---|
| harvest | 2 | 16 s | skip-goal fault: 421M no (0.21) -> 27B **jumped straight to `harvest_target`** (the action includes its own approach) -> crop harvested -> goal |
| mine | 4 | 76 s | travel-skip caught by the 27B after a 0.37-yes/0.76 Decider consensus (the strong-gate's reason for existing); tool-drop fault: 27B repeated the tool-less `mine_target` (failed), the loop re-proposed `give_tool` next step, 27B endorsed it, mine then succeeded |
| mine (5-option action set) | 2 | 12 s | 27B goal-first again: `mine_target` from base in one action |

**Findings that will outlive this game:**

1. **Goal-first vs phase-first judges.** When an action's execution includes
   its own approach (goto is part of mine/harvest), the 27B routinely
   completes a whole multi-phase mission in one action. Phase-relative
   oracles under-count this; mission completion is the honest metric.
   A judge that optimizes for the *task* beats one that optimizes for the
   *phase* whenever the action space permits the shortcut.
2. **Conditional rules are not substitution rules.** The 27B will endorse a
   *proposed* `give_tool` action but will not proactively replace a
   tool-less `mine_target` proposal, even with an explicit "never mine
   without a pickaxe" instruction. It follows conditional instructions
   about the *proposed* action, not counterfactual substitution. Reword
   judge rules in the substitution form ("no pickaxe in inventory ⇒ answer
   give_tool") and the correction appears at the same step.
3. **The 2B generalizes to a new option as a detector, not a
   selector.** Adding `give_tool` as a fifth letter to the mine action
   set, measured on a 30-row labeled grid built around the live no-tool
   states (12 rows with `give_tool` proposed, 12 with a tool-less
   `mine_target` proposed — the arbiter's own failure mode — and 6
   with-pickaxe controls): the controls stay at 0.921–0.926 (no
   5-option drift in the trained readout), and p(`give_tool`) separates
   cleanly by state — 0.255 (no tool, give_tool proposed), 0.060 (no
   tool, mine proposed), 0.009 (with tool). But the *choice* is
   `mine_target` on 30/30: the trained phase action owns the argmax, and
   the new option only ever receives probability mass. In the loop this
   is exactly the right shape for a doubt signal — a proposed
   `give_tool` reads 0.255 < tau_dec, so the cascade escalates to the
   arbiter (which endorses it: the live self-repair path) — but it means
   the 2B does not flag the *other* no-tool case (a proposed tool-less
   mine reads 0.843, "confirmed"): its own tier's failure mode stays
   with the veto tier and the deterministic last-resort repair. The
   generated rows are the fine-tune input to close that gap.
4. **The 421M's live noul distribution is lower than its corpus band**
   (0.21–0.48 across these runs vs up to ~0.9 in the offline sets), so the
   strong-gate sends most steps to the 27B until the threshold is
   re-derived from labeled rows. The 421M remains what the 09-22 verdict
   said it is: a conservative net whose *rejections* are informative and
   whose *acceptances* need a second opinion.

**Interim deployment:** the Decider served these runs from the 4-core CPU
box (the A/B box) via a thin HTTP wrapper over the validated in-process
llama.cpp Q8 readout — bit-identical to the batch results, ~3–4 s/row. The
12 GB GPU card endpoint (addendum 3rd session, ~290 ms/row) is one URL
change away once the testbed's tailnet node finishes its one-click
approval; the loop's code is endpoint-agnostic.

## Addendum 2026-09-25 (fifth session, morning): goal-first dominance, granularity, and what the tiers actually do

**Status: measured.** Four more labeled runs this morning (8 in total
across the overnight + morning sessions, incl. a 24-block fixture — a
`--dist` parameter now scales the fixture distance for world variety),
all on the same three-tier loop:

1. **Goal-first is the dominant live pattern, not an exception.** On
   every 27B contact in the last six runs, the arbiter consumed the
   fixture in step 1 from the base (one action, 24 blocks of walking
   included, 25 s wall). Because actions carry their own approach, a
   multi-phase mission collapses to two steps (act, return) and
   phase-scoped fault injections often never fire — the phase is
   skipped. The 27B is the *task*-completing tier: give it a reachable
   goal and it will do it in one shot.
2. **Action granularity sets which tier does the work.** With
   self-approaching (coarse) actions, the small tiers mostly see *veto*
   opportunities (a proposed skip-goal) and *return/done* confirmations
   — precisely the decisions the 421M noul question was built for —
   while goal selection goes to the 27B. The 2B's corpus strength
   (phase selection, 70.7% top-1) has little live surface on this
   action set; its live role is the calibrated p(proposal) that feeds
   the strong-gate. Making the small tiers *drive* would require
   decomposing actions (goto and mine as separate actions), which
   trades steps and latency for small-tier utilization. On a
   ~272-ms 27B endpoint the coarse action set is the better deal.
3. **The loop's floor is the deterministic policy, not a failure mode.**
   With both model tiers unreachable (endpoint defaults pointed at
   unresolvable placeholder hosts — since fixed), the loop ran on
   policy alone and still completed the mine mission in 5 steps; the
   give_tool oracle proposal did the tool repair. The model tiers add
   judgment on top of a policy that already completes missions.
4. **Thresholds re-derived from the 21 dual-p rows (15 carrying both
   readings).** The false-yes ceiling (an injected skip-goal the 2B
   confirmed at p 0.76 while Laya said 0.37-yes, twice) is 0.371; the
   correct floor is 0.400. The strong-gate threshold now defaults to
   0.40 (advisory: a thin 0.029 gap on N=15) — short-circuits 9/15
   rows vs 4/15 at the old 0.6 with zero short-circuit errors, and
   every observed false consensus still escalates. The Laya yes
   threshold was likewise set to its 09-22 derived 0.35 (faulty
   0.29–0.32 vs correct 0.36–0.39) instead of a placeholder.
5. **Imperative-form tool rules did not change goal-first behavior.**
   Rewriting the judge's tool rule as a strict decision procedure
   (check inventory first; no pickaxe ⇒ give_tool, even after a failed
   mine) was necessary for correctness but the 27B still jumps to the
   terminal action from base when both facts hold. The rule governs
   *which* action; it does not re-impose the phase sequence.

**Net picture of the live loop:** a 421M veto net whose rejections are
informative, a 2B providing the calibrated confirmation that gates
short-circuits, and a 27B that completes the whole mission when either
small tier raises doubt. Two of the five original design assumptions
held (veto quality, healthy new-option generalization); two did not
(goal-first granularity, live Laya distribution); the design adapted
by making the 27B the default decision-maker and the small tiers the
fast, cheap exception handlers.
