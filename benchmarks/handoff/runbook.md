# Executor runbook — GPU benchmark campaigns (platform v0)

One page, in order. Everything you need is in the **brief** the owner sends
(read it first — it names the window id, the prod unit, and the freeze hashes).
You never need to know what any command does; you follow this.

## 0. Before you touch anything

1. Read the brief. Write the **window id** down; it appears in every command.
2. Run the preflight:
   `python3 /<bundle>/benchmarks/campaign.py prepare /<bundle>/campaigns/<SPEC> /<profile>`
   The brief is regenerated; if qualification (Q1–Q10) is not green, STOP and
   say so. Do not proceed.
3. Before the `run`, clear the campaign package's bytecode cache
   (`find /<bundle>/benchmarks -name __pycache__ -exec rm -rf {} +`) and re-run
   the prepare. A fresh bundle extraction is always safe; a *redeployed* one
   (new commit over an old extracted copy) can leave stale `.pyc` files that
   Python may load instead of the new source (bit us in the 2026-09-13
   dogfood). The clear makes the deployed copy unambiguously current.
3. If the owner is not available and the window deadline is inside 30 minutes,
   do not start. Ask.

## 1. Open the window (one command)

`python3 /<bundle>/benchmarks/campaign.py run /<bundle>/campaigns/<SPEC> /<profile> --backend real`

What happens, in the order the machine enforces (you cannot reorder it):
watchdog armed (deadline now running) → prod unit stopped (unit name from the
brief) → temp changes applied (rendered from the verified PVE table) →
readiness probes → cells → **the single exit path**: undo → prod start →
live-verify → disarm.

If the process dies for any reason, the watchdog finishes the exit. The
restore result file is the ground truth:
`cat /<results>/RESTORE-RESULT.md` — read the `VERDICT:` line.

## 2. When a cell fails

Read the cell's `verdict.json`. The class tells you exactly what to do:

| class | you may | you must not |
|---|---|---|
| `RETRYABLE_FAILURE` / `RETRYABLE_INFRA` | re-run the same cell (≤ the policy's attempts) | change any science field |
| `RESOURCE_LIMIT` | nothing — it's a data point; continue | re-quantize, change ctx/bs, "try again smaller" |
| `HARNESS_FAILURE` | apply ONE repair from the brief's allowed list (path/syntax/quoting/parse/fixture-copy) to a copy, re-qualify, continue | touch anything in `requires_owner` |
| `BUILD_DEFECT_ASSERT` | nothing — it needs a build; that is an owner decision | "fix" the build inline |
| `UNKNOWN` | collect the log, notify the owner, STOP the campaign | guess |
| `SAFETY_ABORT` | nothing — the host was touched; the campaign is over | resume |
| `EXPECTED_NEGATIVE` (a documented class) | nothing — the campaign's question is answered; the final is `expected-negative`, which is a success, not a review | |

If a repair is needed, it is declared: edit the copy in
`/tmp/harness-repair/`, re-run qualification, and the next attempt's
`meta.json` records both hashes (scientific unchanged, implementation new).
A repair that changes the scientific hash fails the final assertion by
construction — you cannot make that mistake silently.

## 3. Close the window (only if the runner is already dead)

The runner always closes the window itself. If it died, you wait for the
watchdog (it cannot take longer than the brief's deadline). You do not stop
prod manually. You do not re-run the runner before reading
`RESTORE-RESULT.md`.

## 4. After

- `cat /<results>/RESTORE-RESULT.md` — send the VERDICT line to the owner.
- `cat /<run-dir>/final.json` — send `final`, `valid`, and the cell table.
- Stop. The owner runs ingestion and writes the report. You hand over the
  run directory and the events file. You did not "fix" the numbers, you did
  not skip a cell, you did not re-quantize: if any of those happened, the
  brief's allowed list did not contain it, and you would have been blocked.
