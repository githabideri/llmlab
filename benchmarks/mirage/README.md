# mirage — Cactus Compute Mirage benchmark

One-off evaluation harness for the Cactus Compute **Mirage** vision model
(21 MiB, 51.3M params, 2-bit; `Cactus-Compute/mirage` @ `76c1e9e53381aca8fc3843c636216c96ff220122`;
engine `libcactus.so` 4.0.0; library `cactus-compute==4.0.1`). Not a standing
campaign: a frozen corpus, one run, one report
([reports/2026-11-06-cactus-mirage-vision-evaluation.md](../../reports/2026-11-06-cactus-mirage-vision-evaluation.md)).

## Layout

| Path | What |
|---|---|
| `RULES.md` | **Frozen** scoring rules (v1.1, sha256 in `meta.json`). Read before touching anything. |
| `gen_synthetic.py` | Deterministic corpus generation (shapes / documents / UI, 63 images, seed in file). |
| `fetch_commons.py` | Wikimedia Commons fetcher (43 images); writes `common/` + `common-manifest.jsonl` (URL + sha256, images not committed). |
| `review-commons.jsonl` | Human ground-truth labels for the Commons images (47 records: subjects, presence, counts, text). |
| `build_manifest.py` | Builds `manifest/images.jsonl` (110) and `manifest/tasks.jsonl` (450) from the three sources. |
| `run_bench.py` | The runner: cold-start probe + one raw `cactus_describe` call per task, verbatim to `results/<run>/raw.jsonl`. `--tasks` / `--name` for subsets and resume runs. |
| `grade.py` | Separate scoring pass → `metrics.json`. Pure function of `raw.jsonl` + `tasks.jsonl`. |
| `phaseB_polis.py` | Phase-B probe set (6 Vintage Stone screenshots, side-by-side pair) → `results/phaseB-polis/`. |

## Protocol notes (the two that bit)

1. **The form field name is the question.** `extract(img, {"answer": ...})` makes
   the model answer an unspecified default question. Questions must be the
   field name (`{"Is there a cat?": {"enum": ["yes","no"]}}`). A/B evidence in
   the report; the 141 affected tasks were re-run after the catch.
2. **Every call must be re-validated as JSON.** On some inputs the engine
   ends generation with the JSON still open (premature termination; silent —
   positive return code, no error; deterministic per image; 14% of this
   run's calls). `run_bench.py` records the parse failure and (after one
   retry) the full raw output in `raw_full`; `grade.py` only salvages
   complete `"string""` fields. Mechanism history: RULES.md v1.1 labelled
   this a "154-token budget" from the dry run; the post-run note corrects
   it.

## Reproducing

```
python3 -m venv /tmp/mv && /tmp/mv/bin/pip install cactus-compute==4.0.1
/tmp/mv/bin/python gen_synthetic.py
/tmp/mv/bin/python fetch_commons.py     # needs network
/tmp/mv/bin/python build_manifest.py
/tmp/mv/bin/python run_bench.py         # ~25 min, 4-core CPU; results/<run>/
/tmp/mv/bin/python grade.py <run>
```

`CACTUS_TELEMETRY=0` (set in `run_bench.py`). One process, sequential calls,
no retries except the single truncation retry. ~25 min wall on a 4-core
i5-8500T.
