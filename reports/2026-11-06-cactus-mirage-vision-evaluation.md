# Cactus Compute "Mirage" 21 MB vision model — first independent evaluation

Date: 2026-11-06 (run `run-20261106T233645Z`) · Environment: 4-core CPU LXC (Intel
i5-8500T, 12 GiB RAM), Linux x86_64, Python 3.12.3 · Model: `Cactus-Compute/mirage`
@ `76c1e9e53381aca8fc3843c636216c96ff220122` · Engine `libcactus.so` 4.0.0 (sha256
`89ad23bd…3e0bf9bc574c7`) · Library `cactus-compute` 4.0.1 (PyPI) ·
`CACTUS_TELEMETRY=0`

**TL;DR — Mirage (a 21 MiB / 51.3M-parameter 2-bit vision-encoder + decoder that
runs entirely on a laptop CPU) is the first sub-100 MB on-device vision model we
have measured that gives *useful* structured answers at all: it describes simple
scenes correctly, answers everyday object-presence questions at 95% recall and
identifies the main subject of everyday photos 21/25 (0.84), runs at 1.9–6.6 s
per call with ~190 MB RSS, and is byte-for-byte deterministic. But it fails four
of the nine measured capability classes outright — object counting (2/21),
spatial relations (0/4), OCR (12/12 calls truncated by a silent premature-
termination defect, mean partial CER 0.41), and — worst — *negative* presence: the model
never answers "no", so 36/49 absent-object queries come back as confident
"yes" hallucinations. A 14% silent-truncation rate (the engine ends
generation with an unclosed JSON on text-bearing inputs; see the defect
section) is a first-class reliability problem of this release (documented,
not worked around silently). Verdict: **PROMISING** for low-stakes scene
description and subject-level presence on constrained devices, **NOT
SUITABLE** for counting, spatial reasoning, OCR, or any use that depends on
a trustworthy "no".**

This is a single-session technical evaluation (Phase A of the two-phase task;
Phase B, the Vintage Stone game-screenshot fit for the Polis project, is a
separate report in that project's repo). Nothing here is a benchmark for any
fleet service; the model was run in isolation on the agent host's CPU and the
host was otherwise idle during the timed calls.

## Goal

Cactus Compute published Mirage on 2026-11-05 (a week old at run time): a
~22 MB vision model with object detection + segmentation + OCR + structured
"form" extraction, claimed to run on a mobile CPU. The questions:

1. **Provenance** — where do the weights and engine come from, what exactly did
   we run, and is it reproducible?
2. **Performance** — latency, memory, threads, determinism on a 4-core CPU.
3. **Quality** — nine capability classes against a frozen ground-truth corpus
   of 110 images, with a pre-registered scoring rule set.
4. **Reliability** — the engine's behavior at its output budget (see §Engine
   truncation defect), and whether reported probabilities are usable.

## Provenance (A1)

| Item | Value |
|---|---|
| Model repo | `Cactus-Compute/mirage` (Hugging Face), Apache-2.0 |
| Pinned commit | `76c1e9e53381aca8fc3843c636216c96ff220122` |
| `mirage.cact` | 21,611,043 B (20.6 MiB); sha256 `4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237` |
| `mirage.safetensors` | 216,370,504 B (fp32 master copy, 206 MB) |
| Architecture (from `config.json`) | 51.3M params; 2-bit weights (embedding/mhc 4-bit); 512-px canvas; 16 body + 12 attention + 8 decoder layers; 8192-token vocab; CTC text reader (320 px read width) |
| Engine | `libcactus.so` 4.0.0, 1,933,336 B, sha256 `89ad23bdcfa5b8e62137175eb297664933d75a3095fc44a24043e0bf9bc574c7`, fetched and pinned by the library into `~/.cache/cactus/` |
| Library | `cactus-compute` 4.0.1 (PyPI); C ABI entry `cactus_describe` |
| API | `Mirage().describe(image, **settings)`; `extract(image, form)`; `form` fields take `enum` / bounded `integer` / `string`; a hidden `-` option carries abstention |
| Languages | en, de, fr, es, it, nl, pl (English used here) |

The 216 MB fp32 master and the 21 MiB 2-bit export are both public in the same
repo; only the `.cact` is used at inference. Apache-2.0 applies to both the
model and the engine repo (`cactus-compute/cactus`, PR #187 merged 2026-11-05
added Mirage).

## Setup (A3)

**Corpus: 110 images, 450 tasks**, frozen before any model call (rules in
`benchmarks/mirage/RULES.md` v1.1, sha256 `1fa51a54…e0`):

| Source | n | What it measures |
|---|---:|---|
| Synthetic shapes (34) | 8 subkinds: simple (2–4), crowded (8–12), small, faint, pairs, counts, noise, blank | detect (IoU ≥ 0.5), count, presence, spatial relation — GT is exact by construction |
| Synthetic documents (12) | 2–8 lines, controlled text | OCR (CER, line recall), text presence |
| Synthetic UI (14) | fake windows/HUDs/mobile | UI-element presence (precision-focused) |
| Wikimedia Commons (43) | 25 everyday subjects, 12 crowded scenes, 6 negative (no target) | presence (incl. negatives), choice (subject ID), determinism, perf — every image visually verified and human-labeled (`review-commons.jsonl`, 47 records) |
| Own nvtop screenshots (4) | terminal UI | OCR/text presence |

Commons images are referenced by URL + sha256 in the manifest (27 MB, not
committed); the synthetic set (2.2 MB) is committed and regenerable from seed.
Ground truth for the six Polis game screenshots used in Phase B is *not* part
of this corpus (separate project, separate report).

**Protocol.** One sequential process; cold-start probe in a fresh subprocess
(import, construct, first call, at-rest RSS); one warmup call; then every task
as a raw `cactus_describe` C call with the exact settings shown in
`run_bench.py` (no library-level retries, no normalization). Each call is
recorded verbatim (code, raw bytes, parse status, wall ms, RSS, threads) to
`results/<run>/raw.jsonl`; scoring is a separate pass (`grade.py`) against the
frozen manifest. `CACTUS_TELEMETRY=0` throughout.

**One protocol bug, caught and corrected.** The first pass asked
yes/no/choice questions through a form field named `"answer"` — a neutral name
that carries no question content. The documented usage (and the A/B test in
§Observations) is that **the form field name is the question** (e.g.
`{"Is there a cat in this image?": {"enum": ["yes","no"]}}`). With `"answer"`,
the model answered an unspecified default question: it said "no" (p≈0.64) to
everything. The 141 affected tasks (presence/choice/relation/count) were
re-run on 2026-11-06 ~01:15 with the question as the field name, and those
records are the ones graded below. This is a lesson, not a footnote: a
structured-extraction model that takes questions by field name will silently
answer the wrong question if the name is generic.

## Observations (A4)

### Presence — 91 tasks: 42 "is there X" (GT yes), 49 "is there X" (GT no)

| (GT, predicted) | n |
|---|---:|
| (yes, yes) | 40 |
| (yes, abstain) | 2 |
| (no, yes) | **36** |
| (no, abstain) | 13 |
| (no, no) | **0** |

Present objects: 95% recall, 5% abstention, zero misses. Absent objects: the
model **never says "no"**. It either hallucinates the object (36/49 = 73%) or
abstains (13/49). By kind: everyday 24/24 present-subject queries answered
correctly with 1 abstention; crowded scenes are the worst offender (24/36
queries false-yes); the 6 negative images (designed to contain nothing of the
target) yield 5 false-yes and 13 abstentions. A yes-biased "is X there?" model
is safe only where a false "yes" is cheap; it cannot serve as a negative check.

### Choice (subject identification) — 25 tasks: 21 correct, 4 wrong, 0 abstain (0.84)

Everyday subject-ID with 4 options ("what is the main subject?"). This is the
strongest single number in the whole run and the one that justifies the
PROMISING verdict.

### Spatial relation — 4 tasks: 0/4

Two-shape scenes, "where is A relative to B?" with 4 options. Every answer
wrong; the probability mass is diffuse (e.g. GT below: left-of 0.45 / below
0.31). Spatial reasoning is not a capability this release carries.

### Count — 21 tasks: 2/21 (0.10)

Bounded-integer forms (1–12 / 1–6). For 19 of 21 images with 2–12 objects the
model answers **1** — with p(1) = 0.56–0.70, i.e. confidently wrong. The only
correct answers are the two genuinely-single-object images. Asking the
question by field name (vs the v1 bare "count" field) spread the
distribution (p(1) 0.97 → 0.56–0.70) but did not change the argmax. Note
the contrast with §Presence: on these same scenes `describe` *finds* most of
the objects (box recall 0.72 overall, essentially perfect on simple scenes) —
the model can see them but cannot count them.

### Detect — 23 images: box F1 0.79 (precision 0.89, recall 0.72)

IoU ≥ 0.5 box matching, name matching separate. Simple scenes: boxes essentially
perfect (8/8 images, 2–4 objects each). Crowded scenes (8–12 objects): recall
degrades as objects overlap and shrink — missed boxes are the small/occluded
ones, and the 64-detection budget is not the limit. Of the 81 boxes that
matched a ground-truth object, only 19 carried the exact expected name: the
model returns fluent composites ("red rectangle" for a GT "red square").

### OCR — 12 documents: 12/12 calls truncated; no complete result

Every document call returned a JSON string that the engine cut off mid-value
(§Engine truncation defect). The salvage rule (grade only complete `"string"`
fields that appear before the cut) recovers a mean **partial CER of 0.41**
across the recoverable line fragments — i.e. even the text that does come
through is substantially wrong (expected at 2-bit quantization plus a 320 px
reader width). For document text this release is a non-starter; short
high-contrast labels are the only plausible OCR niche.

### Text presence — 50 queries (25 images × 2): 8 true-found, 3 missed, 0 hallucinated (0.73 true-recall)

"Does the image contain the text X?" 28 queries died with their calls on the
defect below. Of the 11 absent-text queries whose calls survived, all 11 came
back clean "no" — and none of the 50 queries ever produced a hallucinated
"yes" for text that is not there.

### UI-element presence — 38 candidates: 6 found, 8 missed, 20 correctly rejected, 0 false, 4 call-failures (precision 1.0, recall 0.43)

Same shape: nothing it claims is ever wrong, but it finds less than half.

### Describe — 284 calls: 221 complete, 63 truncated

The general scene-description call is the workhorse and it mostly works:
simple and everyday scenes come back with 5–8 named objects, bounding boxes
(0–1000 scale), scores, coarse position and a color/shape description, plus
any visible text. On the game screenshots (Phase B) the same call ran clean at
230–300 output tokens with no truncation at all — truncation is concentrated
where the text section has many long lines (all 12 synthetic documents; plus a
handful of dense real-world scenes).

### Determinism — 5/5 groups byte-identical

Five everyday images, 5 calls each, same process: the raw output bytes are
identical across all repeats. The engine is deterministic on this hardware —
a property the Jev-style decision loops in this fleet depend on, and a useful
one for regression testing (a fixture diff is a signal, not noise).

### Calibration — not usable as-is

The library documentation claims a post-hoc calibration head in the shipped
weights, but `extract()` returns raw decode probabilities, and those are not
calibrated in the places that matter: count says "1" at 0.97 when the answer
is 8–12; presence says "yes" at 0.6–0.9 when the object is absent; the abstain
channel `-` is used unevenly (abstains on 73% of negatives but only 5% of
positives). Any threshold discipline (the fleet's Jev pattern) would have to be
built from scratch on a labeled corpus per question type; the shipped numbers
cannot be trusted as probabilities.

## Engine truncation defect (measured, first-class)

On some inputs the engine **ends generation before the JSON is closed** and
the library hands the truncated string back with a positive return code and no
error of any kind — the `JSONDecodeError` the caller sees comes from the
library's own `json.loads` inside `describe()`, not from an error path. The
caller cannot distinguish "the model finished" from "the model stopped
c halfway" by anything the library exposes. Measured properties:

- **63/450 calls (14.0%)** fail to parse; every failure is a mid-value
  `JSONDecodeError: Expecting value` at a stable offset.
- **Deterministic per image**: same output-token count and byte length on
  every retry (we retry exactly once per RULES v1.1; the retry never
  differs).
- **Far below the stated budget**: the model config carries a 1536-token
  output budget; successful calls in this run reached 1534 tokens, while the
  failing ones ended at 150–436. This is premature termination, not a budget
  cap. (Whether the `max_tokens` option changes it was not tested in this
  run.)
- **Concentrated on text-bearing inputs**: all 12 synthetic documents on
  every one of their 3–8 calls, 2 synthetic mobile-UI images, 4 everyday and
  2 crowded real-world scenes. Simple shape scenes: 0.

Because this is a product defect and not a property of our test, it is
reported here as a first-class reliability metric rather than excluded. Any
deployment of this release must (a) treat every call as potentially
unparseable, (b) budget for a parse-retry, and (c) track a truncation rate.
We have not opened an upstream issue; the engine repo and this report make
the reproduction trivial.

## Performance (A2/A4)

Fresh-process cold start, measured in three fresh processes (mini-test,
re-run, and the two resume runs; `cold.json` per run directory, warm page
cache): import 0.07–0.12 s, construct + model load 0.72–0.75 s, first call
2.19–2.26 s, at-rest RSS ≈ 159 MiB.

Warm steady state (100 timed calls, 20 images × 5; 25 of the 100 were re-run
in a fresh process after an unrelated runner crash — flagged in `meta.json`,
their first per-image call may include residual warmup):

| | ms |
|---|---:|
| min / p50 / p95 / max | 1894 / 2707 / 4558 / 6607 |
| by image class | everyday 2.0–3.5 · crowded 1.9–3.4 · negative 2.4–4.4 · docs 2.5–6.6 · UI 1.9–4.7 · shapes 2.0–3.5 |

RSS plateaus at 151–193 MiB (the ~164 MiB at rest plus working buffers); the
engine uses 5 threads and ignores `OMP_NUM_THREADS`. On the 4-core i5-8500T
that is ~4× slower than the "mobile CPU" target class's interactive comfort
zone, but it fits the Vision-Jev-style *escalation* role (a few calls per
mission step, one at a time) on exactly the constrained hardware this model
was made for.

## Reproducing

```
cd benchmarks/mirage
python3 -m venv /tmp/mv && /tmp/mv/bin/pip install cactus-compute==4.0.1
/tmp/mv/bin/python gen_synthetic.py      # 63 deterministic images
/tmp/mv/bin/python fetch_commons.py      # 43 Wikimedia images (needs network)
/tmp/mv/bin/python build_manifest.py     # manifest/images.jsonl + tasks.jsonl
/tmp/mv/bin/python run_bench.py          # ~25 min on a 4-core CPU; results/<run>/
/tmp/mv/bin/python grade.py <run>        # metrics.json
```

Manifest sha256s and the engine/model hashes are in `results/<run>/meta.json`;
`raw.jsonl` is the complete evidence (450 calls, verbatim).

## Conclusion

Mirage is the first model in this size class we have measured that returns
*structured, useful* vision output on a laptop CPU instead of a lossy
embedding or a single caption. For scene description, subject identification,
and "is the main object there?" on everyday imagery, it is genuinely
promising — 0.84 subject-ID, 95% presence recall, deterministic, 190 MB,
no GPU, Apache-2.0, 21 MiB to ship to a phone.

It is not a general VLM. Counting, spatial relations, and OCR fail at the
capability level, not the threshold level; the never-answers-"no" presence
behavior and the 14% silent-truncation defect make it unsafe as an unsupervised
sensor. In this fleet it belongs (if at all) in the same escalation slot as
the Jev models — a cheap pre-veto or second opinion, never the arbiter — and
that exact fit question is what Phase B answered for the Polis project.

Verification level: **locally verified** (this machine, this run; all calls
reproducible from the pinned hashes above; no production system was touched).
