# Mirage evaluation: frozen scoring rules

Version 1.1, frozen 2026-11-06 before the first scored run (v1.0 draft
pre-run). Any change to these rules after the run started invalidates the
results; if a rule proves unworkable during the run, the whole corpus is
re-run under the new rules.

## v1.1 amendment (found during dry-run, before any scoring call)

- **Engine output truncation.** `cactus_describe` (engine 4.0.0, model 2-bit
  Mirage) returns a positive status code on every call; the observed codes
  track the output-token budget actually applied (default 154; requesting
  256 or 1536 still applies 154; requesting 64 applies 64). When generation
  hits that budget before the JSON is closed, the Python library does NOT
  raise (it only raises on negative codes) and hands back the truncated
  byte string, which then fails `json.loads` (`JSONDecodeError: Expecting
  value`). This is a library/engine defect: no JSON validation, no stop
  sequence handling, no error on budget exhaustion. Truncation is
  deterministic per image (same code and same byte length on every retry).
  It hits dense-text images systematically (all 5 tested documents failed
  with `sections=["objects","text"]`; two also failed with `objects` only).
- **Handling (frozen):** every scored call is re-validated as JSON in the
  harness. On parse failure the call is retried once (the retry result is
  stored alongside); the record is then graded as a hard failure
  (`engine_truncated`). For OCR tasks the truncated byte string is
  additionally salvaged with a regex over complete `"string"` fields and
  scored as a *partial* CER, clearly labeled. The truncation rate per
  image kind is a first-class reliability metric in the report.
- The 154-token budget means long outputs (dense documents, many objects)
  cannot complete in this release; that limitation is reported as a
  product defect, not worked around by re-prompting.

## Call conventions (identical for every scored call)

- Library: `cactus-compute==4.0.1` (PyPI), model `Cactus-Compute/mirage`
  from Hugging Face, model file SHA-256
  `4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237`.
- `CACTUS_TELEMETRY=0` for every process.
- One `Mirage()` instance per run process; no per-call knobs exist in the
  C ABI (verified: only `load`, `models`, `last_error`, `describe`, …).
- `describe`: called with `sections=["objects", "text"]`, all other settings
  at library defaults. `max_tokens`, `max_objects`, `max_text_lines`,
  `conf`, `text_conf` are left at defaults (not overridable to useful
  values before a call in 4.0.1 — see report).
- `extract`: called with the task's form; the hidden `-` (abstain) option is
  part of the enum and is treated as an abstention, never as a wrong answer
  and never as a correct answer.
- Images are passed to the library at their native size (no pre-resize).

## Grading per task type

- **presence** — the model answers a yes/no question via
  `extract(form={field: enum("yes","no")})` (the `-` abstain option is
  implicit). Correct iff argmax option == GT. Abstain is counted separately
  and is neither TP nor FP. Per-query and per-image abstain rates are
  reported; an image-level abstain (all queries abstained) is a "no
  decision" for that image.
- **choice** — 4-way enum via `extract`. Correct iff argmax == GT.
  Abstain as above.
- **detect** — `describe` objects. Each GT object (name = "<color>
  <shape>") is matched to the single best unmatched predicted box with
  IoU >= 0.5 (greedy, descending predicted score). A match also requires the
  predicted object's `description` text to contain the color word AND the
  shape word (e.g. "a red square"); if the description lacks one, the box
  still counts as a *box* match for bbox-precision/recall, and is scored
  separately for *name* correctness (name recall reported alongside bbox
  recall). Overlap-free synthetic scenes use the same rule.
- **count** — bounded integer via `extract` with `lo`/`hi` from the task.
  Correct iff the predicted integer equals the GT count. The 0 returned by
  the smoke test for a 2-object image would grade as wrong. Abstain as above.
- **relation** — 4-way enum (left of / right of / above / below) from
  `extract`. Exact match.
- **ocr (doc)** — `describe` text lines, concatenated in order, normalized
  (lowercase, whitespace collapsed). Character error rate (CER) against the
  GT text normalized the same way (edit distance / len(GT)). Additionally:
  line recall = fraction of GT lines with a predicted line whose edit
  distance ratio is < 0.5.
- **text_presence** — normalized substring match of the query against the
  predicted text (and, separately, against the object descriptions).
  A *true* query must be found (TP); a *false* query must NOT be found
  (TN). A false query found is a hallucination and is always reported.
- **ui_presence** — per candidate element: `describe` object descriptions;
  the candidate counts as "seen" if any predicted object description
  contains the candidate string (case-insensitive). GT present/absent as in
  presence; "seen but GT absent" = hallucinated element.
- **determinism** — 5 consecutive `describe` calls on the same image.
  Raw-JSON identity is checked after stripping the volatile fields
  (`latency`, `tokens`, `wall_ms`, any timestamp). Stable-identity rate =
  fraction of images for which all 5 stripped JSONs are byte-identical.
- **perf** — 5 `describe` calls per image after a 3-call warmup on
  unrelated images. p50/p95 of per-call wall time; process RSS and HWM from
  `/proc/self/status` sampled after the block; thread count from
  `/proc/self/status` (Threads). Cold start is measured in a separate
  process: import time, model load time, first-call time, RSS at rest.

## Aggregate metrics reported

Per task type: accuracy, abstain rate, and (for presence) TP/FP/FN/TN
split by image kind (everyday / crowded / negative / synthetic). For
detect: bbox precision/recall/F1 (IoU 0.5) and name recall. For OCR: mean
CER and line recall, split by font size bucket (>=30px vs <30px). For
perf: p50, p95, median RSS, max HWM, thread count, cold-load time,
first-call time, model file size.

## What is deliberately NOT claimed

- No calibration: predicted probabilities are compared only via argmax and
  never interpreted as calibrated likelihoods (smoke test showed a 0.60
  confidence on a known-wrong count).
- No transfer claim: Phase A results on these images are evidence only
  about these images; Vintage-Street-specific capability is Phase B.
- No absolute quality claim: scores are relative to the frozen GT in
  `manifest/` and the rules above, not to human perception.

## Post-run note (appended 2026-11-06, after run `run-20261106T233645Z`)

The v1.1 text above is unchanged (its sha256
`1fa51a545c32d3641db7b0aa5b21192a15d353c3c588e40fcb02f86e31ce42e0` is recorded
in the run's `meta.json`). The *mechanism label* in the truncation amendment
("default budget 154") was an over-interpretation of the dry run; the
operational handling (JSON re-validation, one retry, hard-fail grading,
OCR salvage, truncation rate as a first-class metric) was executed exactly
as written and stands. The full-run data refutes the fixed-154 reading:

- failing calls end at **150–436** output tokens (varies per image);
  successful calls reach **1534** against the config's 1536-token budget;
- the dry-run probes re-read consistently: requesting `max_tokens=64` did
  apply 64 (the budget is real and honored), while the tested document
  images ended at ~154 tokens whether 256 or 1536 was requested — the
  model stopping early on those images, not a cap being applied.

The defect is therefore **premature termination**: generation ends with the
JSON still open, silently (positive return code, no error), deterministically
per image, concentrated on text-bearing inputs. See the report's "Engine
truncation defect" section for the full measurements.
