# Decision classifiers (one-pass, Laya-class)

Designing, calibrating, and evolving one-pass decision classifiers: 421M-class bidirectional encoder models (Laya / Jev-style — unstructured state in, typed probabilistic decisions out, one forward pass, no text generation). This is the repo's **methodology home** for anything in that class; frozen evidence lives in [reports/](../reports/README.md), the agent-facing design view in [agents/jev-designer](../agents/jev-designer/SKILL.md).

## What you're programming

The reference implementation here is [Laya 0.1.6](https://huggingface.co/convaiinnovations/laya): ModernBERT-large (395M) + a decision head (2 transformer layers, option-marker scorer, act/escalate head), 421M total, trained with **RLCD** — reinforcement learning against strictly proper scoring rules, so *reporting honest probabilities is the only way to maximise reward*. Self-hosted in this fleet via [openjev/](../openjev/README.md) — a Jev-schema endpoint + playground on a 2-core CPU container (the prompter's system prompt in `openjev/server.py` is an operational distillation of this document); one call ≈ 0.5–5 s, batch ≤ 25, no auth (LAN-only).

- **Hard budget:** 512 tokens per question (state + instructions + options combined). Long documents → pre-extract; the model cannot read a 4k-token ticket.
- Every question is an independent question about the *same* state, answered in one parallel forward pass. No chaining between questions; a dependent question is a second call.
- **Three question types:** `choice` (2–8 options, 10 is the ceiling), `score` (true ordinals only), `noul` (calibrated yes/no — the most reliable primitive).
- The model card's own caveat, taken seriously: *"Laya is a fast base to specialise, not a zero-shot decision engine"* — the base checkpoint sits near the majority-class baseline on typed-decisions zero-shot; the 0.766 benchmark score belongs to the fine-tuned checkpoint. Zero-shot success on *easy* question classes (below) is real but bounded.

## Design rules (each one measured on this fleet, not guessed)

1. **noul is the workhorse, with an explicit default anchor.** "Yes only if <concrete trigger>; **no otherwise** — <typical no case> is no." Measured: with the anchor, 1–3% false positives on ordinary texts; without, 24–76%.
2. **Never use a score scale for presence of a feature.** A 3-level presence scale drifted to *medium on every input, under every wording* (the middle-level drift). The same concept as a noul scored 4/4.
3. **Add an escape option** when the input may match none of the options (`not_comparable`, `none`, `no match`) — measured: a forced guess at 44% became 90% abstention. Omit it only when some answer always applies.
4. **Gate on confidence — within its regime.** See the regime map below: `< ~0.5 = undetermined` is the right *default*, but whether confidence can gate at all depends on the question's trigger class, and thresholds are per-use-case, not universal.
5. **2–8 options; 10 is the ceiling.** Clean cases degrade from ~90% toward 35–70% as option count grows; merge related labels instead of adding.
6. **Labels short, unnumbered.** Numbered prefixes ("1: Very Negative") measurably degrade scale reading.
7. **Observable triggers beat abstract judgments.** "Is this clickbait?" was unreliable (false-positived a dry headline at 85%); rewording toward observable phrases helped but never fully fixed it. Some abstract concepts simply exceed a 421M — if a design needs multi-hop reasoning, arithmetic, or long documents, it doesn't fit; say so and route to the 27B.
8. **One concept per question; `instructions` is one operational sentence.** No "you are an expert" filler. snake_case names.
9. **States realistic and short** (< ~400 tokens).

## The regime map

Whether confidence is a usable signal is a property of the **question's trigger class**, not its wording. Two measured regimes, one out-of-bounds class:

| Regime | Example | Measured behavior |
|---|---|---|
| **Concrete-observable** — the answer is readable from surface facts in the state ("is a refund explicitly requested?", "is the proposed action valid given these facts?") | noul pre-filters, action-validity checks | Confidence is a real signal. **Per-use-case thresholds work**: the internal decision-loop use case gates a noul pre-filter at p ≥ 0.60 (deliberately *not* the default 0.5) and short-circuits most steps at ~2 s with the 27B engaged only on the unsure minority (details with that project's publication). |
| **Abstract-comparative** — the answer requires comparing two descriptions against each other ("do claim A and claim B contradict?") | the [doc-staleness detector](#the-doc-staleness-application) | noul **inverts** (answers the implicit "do these match?" — measured in both polarities across two wordings, so a type problem, not wording); 2-way is *worse* than 3-way (the escape option helps the model commit); an explicit subject prefix fixes labels but collapses confidence; and **confidence cannot gate** — correct answers arrive at 1–65%. The reframe that restores the concrete regime: make the comparison *observable* (NLI on concrete value mismatches, not on vibes). |
| **Multi-hop / subject-aliasing** — two names for one knob, or facts spread across documents | q8_0 vs q4_0 as "the same quant" | Out of capability at 421M. Route to the 27B/human; do not keep rewording. |

**Tier-placement rule** (measured v3→v4 in the internal decision loop): the bigger, more conservative model belongs as a **doubt-arbiter** — called only when the cheap tier is uncertain — not as a **veto** in series with everything. A veto-placed 27B stalled *correct* actions; a doubt-arbiter 27B waited only on visible failure evidence ("informed conservatism, not reflexive"). And the 27B-side contract that makes the tier cheap: thinking off, single-word answers, a few hundred ms.

## The doc-staleness application

The canonical comparative-regime case study: [reports/2026-09-21-laya-doc-staleness.md](../reports/2026-09-21-laya-doc-staleness.md). The pattern in one line: **stale = an NLI contradiction between a non-owner doc's claim and the owner's fact, where "owner" comes from the [fact-ownership map](README.md#fact-ownership-where-each-mutable-thing-lives).** A deterministic pipeline extracts volatile lines (watts, ports, temps, t/s, ctx, versions, quants, GPUs), resolves owners, excludes frozen zones, and pairs claim↔fact; Laya judges each pair; **confident-contradicted (conf ≥ 0.5) = advisory flag only — a human/agent adjudicates, never auto-fix; everything else routes to the 27B or clears.** Measured: 5 confident flags, all true, zero false, across 284 judgments — including one genuine human miss, on a corpus that had just been scrubbed clean (the negative control held).

## Calibration protocol (when a gate misbehaves)

1. **Fit a temperature per (question type, option count) on your own data before trusting probabilities.** The Laya card is explicit: it "ships over-confident" (mean ECE 0.466 pre-fit) and refitting one temperature per (type, option count) moves it to ~0.081 — "do this on your own data before trusting the probabilities."
2. **Derive thresholds from an oracle-labeled set, don't guess them.** The cascade literature is unanimous ([TMLS](https://www.tmls.nyc/research/model-routing-cascades), [UCCI](https://arxiv.org/html/2605.18796), [BARGAIN-A-M](https://arxiv.org/pdf/2509.02896)): the escalation threshold "cannot be guessed; it must be derived from evaluation data that pairs the confidence signal with actual correctness, and re-derived as models and traffic change." Recipe: a few hundred pairs labeled by the 27B oracle (humans touch the disagreements), try every observed confidence value as the threshold, pick the one meeting your target. Expect **per-class thresholds** — output classes are calibrated differently (e.g. "consistent" commits, "contradicted" hedges).
3. **Data discipline.** Label with the **oracle model (27B) + human adjudication — not with the model being calibrated**: same-model pseudo-labeling is textbook confirmation bias, and the filter that makes it tempting (confidence) is precisely what may be broken. Split related records together (by document/subject, not randomly) so near-duplicates can't leak across train/test. Tune on a dev split; confirm on a held-out split **once per pass** — if dev improves and held-out degrades, the pass didn't happen.
4. **Controls.** Every evaluation run is sandwiched: a **negative control** (a known-clean corpus must produce zero confident alarms) and a **positive control** (a known-stale corpus must produce the known alarms — for this repo that's a pre-fix git-history checkout). One knob per pass; an append-only pass log records what changed and what moved.

## Fine-tune path (open — decision pending, not started)

The only route from *alarm* to *pre-filter* (clearing the boring majority locally). Candidates, in decreasing order of operational friction:

1. **Laya's own fine-tune notebook** — 2×T4, ~1.2k cases / 6k decisions, hybrid policy-gradient + cross-entropy, built-in post-training temperature fit; measured 0.362 → 0.766 (clears the 0.735 teacher ceiling). Keeps the existing endpoint unchanged.
2. **An SFT NLI fine-tune of the same backbone** — [tasksource/ModernBERT-large-nli](https://huggingface.co/tasksource/ModernBERT-large-nli) (0.4B; MNLI 0.89, "better than llama 3.1 8B Instruct on ANLI and FOLIO"). Standard supervised head → routine LoRA, standard softmax → temperature scaling applies directly, and it may **beat zero-shot Laya without any training** — so it gets a zero-shot fixture test before anything is trained.
3. **A [jevlike](https://github.com/vinnylarouge/jevlike)-style option scorer** — frozen pretrained encoder + small supervised head, CPU-trainable, reports ECE; keeps the Jev-shaped interface.

Whichever is chosen, the mined corpus serves three times: training labels, the calibration set for the *current* model, and the held-out evaluation. Training hardware: the main GPU server (Ryzen 5 5600X) is the natural host for anything GPU; a small head trains on CPU.

## References

- [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) (model card) · [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) (repo + fine-tune notebook) · [PyPI](https://pypi.org/project/laya/)
- [tasksource/ModernBERT-large-nli](https://huggingface.co/tasksource/ModernBERT-large-nli) · [Answer.AI on ModernBERT instruct](https://www.answer.ai/posts/2025-02-10-modernbert-instruct.html)
- [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike) · [featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev)
- [TypeSafe Jev / System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [reproductions tracker](https://huggingface.co/spaces/multimodalart/jev-reproductions-tracker)
- Cascades: [TMLS](https://www.tmls.nyc/research/model-routing-cascades) · [UCCI](https://arxiv.org/html/2605.18796) · [BARGAIN-A-M](https://arxiv.org/pdf/2509.02896)
- Doc-drift prior art: [jbrockSTL/doc-drift](https://github.com/jbrockSTL/doc-drift) · [validate-consistency](https://tonsofskills.com/skills/validate-consistency/)
