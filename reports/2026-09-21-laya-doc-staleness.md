# Laya 421M as a documentation-staleness detector — NLI on claim pairs

**Date:** 2026-09-21
**Category:** Experiment
**Status:** Measured. Laya 0.1.6 (421M ModernBERT-large decision model, [HF](https://huggingface.co/convaiinnovations/laya)) self-hosted on a 2-core CPU container, called through a System One-shaped batch endpoint (≤ 25 states per call, one forward pass per state). Corpus: this repo. The bigger tier referenced below is the always-on 27B (the vLLM node, 2× RTX 3090 TP2, via the llm-hub — [hub/](../hub/README.md)).

**TL;DR** — "Is this doc stale?" is an abstract judgment, which a 421M cannot do reliably. The reframe that works is **NLI on claim pairs**: premise = the fact-ownership owner's current fact, hypothesis = the claim in a non-owner doc. Laya as a *precision-first alarm* (confident-contradicted → advisory flag; everything else → 27B/human) measured **5 confident flags, all true, zero false** across 284 judgments — including one flag that was a *genuine human miss* the same session's cleanup had overlooked. Confidence cannot gate in this comparative regime (correct answers arrive at 1–65%), so it is an alarm, not a pre-filter, until the fine-tune path is built.

## Goal

After the 2026-09 drift episode in this repo — a 250→220 W power-limit change sitting stale in **six** docs for two weeks, a port move in three, a dead "multi-slot" claim — the structural fix was the **fact-ownership map** ([docs/README.md](../docs/README.md): one owner per mutable fact class) plus the no-copy / verify-live / cards-as-decision-log rules. What the ownership map + regex cannot reach is **paraphrase-level semantic staleness**: the 35B card's changelog says "multi-slot home dismantled 2026-09-08", while a non-owner index row still says "multi-slot + vision" — no shared token, no pattern match. The question: can a 421M local model detect that class?

## Setup

**The reframe.** The direct question ("does this doc need updating?") failed in earlier Laya testing: abstract concepts exceed a 421M (the "clickbait" probe false-positived a dry headline at 85% — see [docs/decision-classifiers.md](../docs/decision-classifiers.md), rule 7). The workable reframe is NLI on *concrete* claim pairs: premise = the owner's current fact, hypothesis = the non-owner claim, relation = contradiction. This is not an arbitrary choice — Laya's backbone is ModernBERT, and NLI is exactly the task class that family is benchmarked on: [tasksource/ModernBERT-large-nli](https://huggingface.co/tasksource/ModernBERT-large-nli) (same 0.4B backbone, SFT-trained) is "very good at reasoning tasks (better than llama 3.1 8B Instruct on ANLI and FOLIO)". Short texts, observable value mismatches — the class the backbone was measured on.

**The use case** (`doc-stale`, saved on the local endpoint; questions JSON needs an explicit `type` per question or the server 500s): one **3-way choice** question over the state `CLAIM: <doc line> | FACT: <owner fact>` with options `{contradicted, consistent, not_comparable}`, plus a noul control in the first variant. All questions answer in a single forward pass.

**The fixture corpus:** 22 claim/fact pairs mined from *this repo's actual 2026-09 incidents* — 10 contradicted (8082↔8080, 250 W↔220 W, multi-slot↔single-slot, 1×3090+2×3060↔2×3090, llama.cpp↔vLLM, 64K↔256K, q8_0↔q4_0, dual↔single slot, ~15↔~25–29 t/s, Q4_K_M↔W4A16), 6 consistent (incl. one paraphrase pair), 3 different-subject, 3 fuzzy (dated changelog entries).

**Three variants measured** (each a full batch, ~40–80 s on the 2-core box):
- **v1**: 3-way choice + noul control
- **v2**: 2-way choice (consistent + not_comparable merged) + affirmative noul ("is it outdated?")
- **v3**: v1's 3-way + an explicit `SUBJECT:` prefix in the state

**The live sweep** (deterministic pipeline; the script lives in the private companion repo, this report is the public record): extract volatile lines per the ownership map (8 fact classes: watts, port, °C, t/s, ctx-K, version, quant, GPU), resolve each line's owner (model cards + hardware profiles), exclude frozen zones (dated changelog entries, `Historic:`/`Dismantled`/`Rollback` sections, frozen model cards), pair claim↔fact by shared fact class + subject tokens (≤ 3 facts per claim), batch to the endpoint. Verdict contract: confident-contradicted (confidence ≥ 0.5) = **FLAG, advisory only**; everything else = cleared or undetermined → 27B/human. Unpaired lines are reported as a coverage measure, not findings.

## Commands

```bash
# fixture / sweep batches (System One-shaped endpoint, one call per pair)
POST /v1/batch        # {"states": [≤25], "questions": {...}}  → per-state answers
# the local endpoint wraps the same API with a use-case catalog and persistence;
# a full sweep of 109 pairs runs in ~5 minutes on the 2-core box
```

## Observations

- **3-way choice is the only usable question shape.** v1's label is right on 16 of 19 solid pairs, and across **all 38 judgments of all three variants there were zero high-confidence errors**.
- **noul is inverted on comparative questions.** Confident "no" on 9/10 true contradictions; confident "yes" on 5/6 *identical* pairs — in both polarities, across two wordings. It answers the implicit "do these match?", not "is this contradicted?". Type problem, not wording — rewording cannot fix it.
- **2-way is worse than 3-way** (undetermined 58% → 79%): the `not_comparable` escape was *helping* the model commit on same-subject pairs, not siphoning probability off them.
- **Explicit `SUBJECT:` prefix** fixes the subject-aliasing mislabels (q8_0/q4_0, Q4_K_M/W4A16 now come out contradicted) but collapses confidence everywhere (most < 25%) — net negative.
- **Confidence cannot gate in this regime.** Correct answers arrive at 1–65% confidence, so the standard `<0.5 = undetermined` gate releases only ~40% of fixture pairs and **97% of real doc lines**. Markdown normalization (stripping `**`, `|`, `[]()`, hashes before pairing) did not move the number — the low confidence is not a formatting artifact.
- **Capability boundary:** subject aliasing (two option names for the same knob) and multi-hop facts exceed the 421M; those stay in 27B/human territory.

### v1 table (the winning shape; conf = model confidence)

| # | pair | choice (top% / conf) | verdict | noul (yes% / conf) |
|---|------|----------------------|---------|--------------------|
| 1 | 8082 vs 8080 | contradicted 58 / 11 | undetermined | 6 / 94 ← false negative |
| 2 | 250 W vs 220 W | contradicted 63 / 16 | undetermined | 15 / 85 ← false negative |
| 3 | multi-slot vs single | contradicted 87 / 56 | **flag** | 4 / 96 |
| 4 | 1×3090+2×3060 vs 2×3090 | contradicted 57 / 28 | undetermined | 33 / 67 |
| 5 | llama.cpp vs vLLM | contradicted 66 / 23 | undetermined | 28 / 72 |
| 6 | 64K vs 256K | contradicted 70 / 26 | undetermined | 11 / 89 |
| 7 | q8_0 vs q4_0 | not_comparable 35 / 0 | undetermined | 5 / 95 |
| 8 | dual vs single slot | contradicted 90 / 65 | **flag** | 84 / 84 ✓ |
| 9 | ~15 vs ~25–29 t/s | contradicted 63 / 16 | undetermined | 6 / 94 |
| 10 | Q4_K_M vs W4A16 | not_comparable 44 / 8 | undetermined | 18 / 82 |
| 11 | vLLM 0.28.0 (identical) | consistent 96 / 82 | cleared | 96 / 96 ← false positive |
| 12 | MTP k=3 (identical) | consistent 85 / 53 | cleared | 80 / 80 ← false positive |
| 13 | port 8080 (identical) | consistent 70 / 26 | undetermined | 48 / 52 |
| 14 | fp8 KV (identical) | consistent 83 / 48 | undetermined | 94 / 94 ← false positive |
| 15 | 2-bit retired (identical) | consistent 94 / 76 | cleared | 96 / 96 ← false positive |
| 16 | 81 °C (paraphrase) | consistent 98 / 89 | cleared | 97 / 97 ← false positive |
| 17 | RAM vs MTP (diff subject) | not_comparable 71 / 37 | undetermined | 0 / 100 ✓ |
| 18 | WhisperX vs 220 W (diff subject) | not_comparable 89 / 63 | cleared | 0 / 100 ✓ |
| 19 | model vs GPU temp (diff subject) | not_comparable 84 / 50 | cleared | 3 / 97 ✓ |
| 20 | dated 250 W vs current 220 W | consistent 40 / 1 | undetermined | 14 / 86 |
| 21 | "keeps the vision case" vs all-have-vision | not_comparable 56 / 10 | undetermined | 13 / 87 |
| 22 | "moved to vLLM in Sep" vs since 09-08 | not_comparable 49 / 8 | undetermined | 44 / 56 |

v2: 4 correct / 0 wrong / 15 undetermined. v3: 4 / 0 / 15.

### Live sweeps on the (just-scrubbed) corpus — a ready-made negative control

The corpus had been cleaned in the same session, so a correct detector must flag nothing.

| run | pairs | FLAG (confident contradicted) | cleared | undetermined | unpaired |
|-----|-------|-------------------------------|---------|--------------|----------|
| 1 (raw lines) | 109 | **1 — true** | 2 | 106 (97%) | 25 |
| 2 (post-fix + normalization) | 109 | 0 | 3 | 106 (97%) | 25 |

- **Run 1's flag was a genuine miss we had overlooked**: `models/README.md`'s in-production list still labelled the 35B "multi-slot + vision" (legend included) although its multi-slot dual-3060 home was dismantled 2026-09-08 — the top README had been fixed that morning, the model index had not. Fixed the same day ([`35eec06`](https://github.com/githabideri/llmlab/commit/35eec06): row/legend → "vision"; the 35B card's preserved dual-3060 table moved under a frozen `### Historic:` heading). Run 2 then produced **zero flags** — the contract held.
- **Totals:** 284 judgments (66 fixture + 218 sweep) → **5 confident flags, all true, zero false.**
- **The 25 unpaired lines are a coverage report, not findings** — mostly methodology lines (sizing tables) with no owner per the map; correct to leave alone.

## Conclusion

Laya 421M is **not a final judge** for documentation staleness — confidence cannot gate in the comparative regime, and the always-on 27B stays the arbiter of record. It **is** a cheap, local, **precision-first alarm**: confident-contradicted (conf ≥ 0.5) → advisory flag that a human/agent adjudicates and fixes; everything else → 27B/human. The measured value is the confident catch *between* production changes — one full sweep is ~5 minutes on a 2-core box — with a clean corpus producing zero flags and a dirty one producing only true ones (5/5 here, including a real human miss).

Caveat on the sample: n = 22 fixture pairs; the "100% precision" rests on 5 confident alarms. These numbers are directional, not statistical.

Turning the alarm into a real triage **pre-filter** (clearing the boring 97% locally instead of routing it up) is the open fine-tune path — see [docs/decision-classifiers.md](../docs/decision-classifiers.md), § Fine-tune path.

### Open questions

1. **Fine-tune** (the only route to pre-filter value): mine a few hundred pairs from this repo, label with the 27B oracle (not the 421M itself — same-model pseudo-labeling carries confirmation bias), train a ~1B-class scorer (Laya's fine-tune notebook, or the SFT NLI fine-tune of the same backbone, or a jevlike-style option scorer), then re-derive the gates. A small project.
2. **27B routing layer** for the undetermined minority (design only; no code yet).
3. **Triggering** — event-driven on owner-doc change vs. periodic sweep — deliberately deferred; a sweep is ~5 min, so the agent that changes a fact can run it as a follow-up step.
4. **Owner gap:** the secondary 3060 box has no hardware profile, so its serving lines come out unpaired.

## Prior art

The bigger-model world built event-driven versions of this pattern; this is the small-model variant (local, CPU, advisory-only):

- **[jbrockSTL/doc-drift](https://github.com/jbrockSTL/doc-drift)** — per-PR GitHub Action: LLM compares the diff against configured doc sources, posts a comment, optionally fails the build (gated on a 0.75 self-reported-confidence threshold — a regime that works for generative LLMs and not, as measured here, for a 421M decision head).
- **[validate-consistency](https://tonsofskills.com/skills/validate-consistency/)** (000-jeremy plugin) — the same architecture at the policy level: a per-fact-class **authority registry** (`sot-map.yaml`), deterministic checks block, LLM-judged findings are advisory-only, and the truth invariant "raw findings are candidate evidence, not truth, until a human adjudicates them". Our ownership map is this pattern applied to a personal repo.
- **dosu.dev's Claude Code pattern** — a repo file maps source areas → affected doc pages; on merge, an agent opens a follow-up doc-update PR or explains why none is needed.
- **LLM cascades** — the 421M→27B tiering is the standard cheap-proxy/oracle cascade; the literature is unanimous that the escalation threshold must be derived from oracle-labeled evaluation data, not guessed (TMLS survey note; [UCCI](https://arxiv.org/html/2605.18796) "most deployed routers use uncalibrated confidence scores"; [BARGAIN-A-M](https://arxiv.org/pdf/2509.02896) for per-class thresholds).

## References

- https://huggingface.co/convaiinnovations/laya — Laya model card (backbone, RLCD training, "ships over-confident … do this on your own data before trusting the probabilities", fine-tune notebook)
- https://github.com/NandhaKishorM/laya — Laya repo (the fine-tune notebook: 2×T4, hybrid policy-gradient + cross-entropy, post-training temperature fit)
- https://huggingface.co/tasksource/ModernBERT-large-nli — the SFT NLI fine-tune of the same backbone family
- https://github.com/vinnylarouge/jevlike — train a one-pass option-attention scorer on `{context, options, label}` JSONL (CPU-capable, reports ECE)
- https://github.com/featherless-ai/simple-jev — Jev-endpoint fine-tuning with a teacher model supplying missing labels
- https://typesafe.ai/blog/introducing-system-one-models-and-jev — the commercial original
- https://huggingface.co/spaces/multimodalart/jev-reproductions-tracker — the reproductions ecosystem
- https://github.com/jbrockSTL/doc-drift · https://tonsofskills.com/skills/validate-consistency/ — doc-drift prior art
