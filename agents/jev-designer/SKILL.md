---
name: jev-designer
description: >-
  Design and debug one-pass decision classifiers (Laya/Jev class, 421M): choosing
  the question type (choice/score/noul), observable-trigger wording, confidence
  regimes and per-use-case thresholds, doubt-arbiter vs veto tier placement,
  calibration (temperature fit, oracle-labeled thresholds), and the fine-tune
  decision. Use when: "build me a classifier for X", "why does the small model
  hedge / invert / force a guess on this", "the gate is letting bad answers
  through", "should we fine-tune", or reviewing any use-case design against the
  measured rules. The record is docs/decision-classifiers.md.
---

# jev-designer — Designing One-Pass Decision Classifiers

A decision-classifier use case is a *typed question with an evidence budget
attached*: one 512-token state, a few independent questions, one forward pass,
calibrated probabilities out, no text generation. The operational companion is
[docs/decision-classifiers.md](../../docs/decision-classifiers.md) — the
measured rules and the regime map; this skill is the design procedure.

## 1. Start with the trigger, not the question

Ask: **is the answer readable from observable surface facts in the state?**

- **Yes** (presence of a phrase, a value mismatch, a concrete condition in the
  state) → the 421M's home turf. Confidence is a real signal; design a gate.
- **No — it requires comparing two descriptions, or multi-hop inference across
  documents** → the comparative/multi-hop regime: noul will *invert* (it
  answers "do these match?", not your question), and confidence cannot gate
  (correct answers arrive at 1–65%). Two options, in order: (a) **reframe to
  an observable comparison** (e.g. NLI on concrete value mismatches instead of
  "is this stale?"), or (b) **route the tier to the 27B** and say so. Do not
  spend iterations rewording an abstract question — wording cannot fix a type
  problem (measured: two wordings, both polarities, same failure).

## 2. Choose the question type

1. **noul** — the default. Calibrated yes/no, the most reliable primitive.
   *Always* write the explicit default anchor: "yes only if <concrete
   trigger>; no otherwise — <typical no case> is no." Without the anchor:
   measured 24–76% false positives; with it: 1–3%.
2. **choice** — 2–8 options (10 is the ceiling; beyond that accuracy collapses
   as options share the fixed token budget). Add an explicit escape option
   (`not_comparable`, `none`) whenever the input may match nothing. Labels
   short and unnumbered.
3. **score** — *true ordinals only* (severity, intensity with clear rungs).
   Never for presence of a feature (measured: drifts to the middle level on
   every input).

One concept per question; `instructions` is one operational sentence; snake_case
names; keep the state < ~400 tokens and realistic.

## 3. Place the tiers

- **Cheap tier (421M) carries the common case.** It answers first, fast,
  locally; its confidence decides what escalates.
- **The big model (27B) is a doubt-arbiter, never a veto.** Placed *in series
  with everything* as a veto, it stalls correct actions (measured failure).
  Placed *only on the cheap tier's uncertainty*, its conservatism is informed —
  it waits on visible evidence, not reflex.
- **27B-side contract:** thinking off, single-word answers, a few hundred ms —
  the big brain answering like the small one, so the tier is cheap to call.
- **Thresholds are per-use-case.** The default `<0.5 = undetermined` is a
  starting point, not a law; derive the actual threshold on your own
  oracle-labeled data (see 4). We run a noul pre-filter at p ≥ 0.60 in the
  interactive decision loop precisely because 0.5 was wrong for that use case.

## 4. Calibrate before you trust (and re-derive when anything changes)

- **Fit one temperature per (question type, option count) on your own data.**
  The Laya card is explicit that the model ships over-confident (ECE 0.466 →
  0.081 after per-(type, options) refit) — "do this on your own data before
  trusting the probabilities."
- **Derive the threshold from an oracle-labeled set.** A few hundred pairs
  labeled by the 27B oracle (humans adjudicate disagreements); try every
  observed confidence value as the gate; pick the one meeting your target.
  Expect **per-class** thresholds — output classes calibrate differently.
  Re-derive whenever the model or the question changes.
- **Never label with the model being calibrated** (same-model pseudo-labeling =
  confirmation bias, and the confidence filter that makes it tempting is
  precisely what may be broken). Split related records together (by
  document/subject) so near-duplicates can't leak across splits.

## 5. Debug protocol (which way to move)

- **Wrong answer at high confidence** → the question is mis-worded or
  mis-typed. Reword or split; check the trigger is observable (rule 1).
- **Wrong answer at low confidence** → the model already told you it was
  guessing. **Reframe the question type before the wording** (measured:
  score→noul reframe fixed what three wording iterations failed on).
- **Right labels, uselessly low confidence** → the comparative regime. You
  have an alarm, not a gate: keep the high-confidence-alarm contract (advisory
  flag, human adjudicates), route the rest up, and treat the fix as the
  fine-tune path, not more wording.
- **Needs multi-hop, arithmetic, or > 512 tokens** → out of capability. Stop,
  name what it would take (fine-tune, bigger model), route it.

## 6. The fine-tune decision (last, and only on evidence)

Fine-tuning is the route from *alarm* to *pre-filter*, but it is a small
project, not a tuning knob. Before committing to it: (1) run the **SFT NLI
fine-tune of the same backbone** ([tasksource/ModernBERT-large-nli](https://huggingface.co/tasksource/ModernBERT-large-nli),
0.4B) *zero-shot* on the task's fixture set — it may beat zero-shot Laya and
remove the need entirely; (2) if training is still warranted, pick the target
on measurements — Laya's own notebook (2×T4, keeps the endpoint), LoRA on the
SFT NLI (easiest supervised path), or a [jevlike](https://github.com/vinnylarouge/jevlike)
scorer (CPU-trainable, Jev-shaped); (3) the mined corpus serves three times —
training labels, calibration set, held-out evaluation. Details and status:
[docs/decision-classifiers.md, § Fine-tune path](../../docs/decision-classifiers.md).
