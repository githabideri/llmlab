# benchmarks/

The campaign platform for GPU inference benchmarking: one code path that runs
a whole multi-night campaign as a **frozen bundle** on any target in the fleet,
with the failure modes of six past campaigns encoded as a qualification matrix
that must be green before a window may open.

(The March 2026 OpenClaw ladder is frozen under `legacy/`; it is not part of
this platform.)

## Layout

```
campaign.py            the only entry point (prepare / qualify / deploy / run / ingest)
spec.py                campaign spec: JSONC, validation, the scientific hash
freeze.py              freeze.json: implementation hash + scientific hash + bundle hashes
verdict.py             the verdict taxonomy (PASS ... EXPECTED_NEGATIVE ... SAFETY_ABORT)
classifier/            log + response-shape classifier, with per-class provenance
window.py              one enter (watchdog first), one idempotent exit, always
preflight.py           EXPECTED / OBSERVED / MATCH|DRIFT table before the window opens
runner.py              the state machine: attempts, resume, gates, stop conditions
host_failure.py        host-failure stop gate (Xid / MCE / panic / reboot)
dialects.py            per-PVE-version command tables (only verified syntax renders)
backends/              Backend interface: real (SSH) and fixture (in-process)
clients/               bench-llama.py, bench-vllm.py, mm_driver.py (field-proven)
qualify.py             the P0/P1 qualification matrix (14 scenarios)
fixtures/failures/     the historical failure corpus (sanitized) + replay expectations
handoff/runbook.md     the one-page executor runbook
campaigns/             dogfood campaign spec
```

## The two hashes

Every run carries two hashes: the **implementation hash** (this git commit —
what the code *is*) and the **scientific hash** (the sha256 of the spec's
science fields: model, quant, workload, matrix, verdict policy — what the
experiment *means*). A bounded, logged repair may change the implementation
hash. If a repair touches a science field, the final assertion fails by
construction and the run is marked invalid regardless of its numbers. The
scientific hash is recomputed at the end of every run and compared.

## Qualification

`python3 campaign.py qualify` runs the real runner against the fixture backend:
a real in-process HTTP server speaking the llama.cpp `/completion` SSE contract,
fault-injected failure shapes (OOM at the documented wall, meta-backend assert,
unsupported architecture, empty-200 health, malformed SSE, host reboot), and a
replay of the historical failure corpus. Each P0 row is the recurrence of a
specific observed failure from the 2026-09 campaign series:

| # | scenario (the incident it prevents) |
|---|---|
| Q1 | clean cell: real client over real sockets, metrics, determinism |
| Q2 | documented wall → `EXPECTED_NEGATIVE`, campaign stops DONE — the 09-12 "needs human review at 2 am" gate failure |
| Q3 | MiB-sized OOM below the wall → `RESOURCE_LIMIT` (the 09-12 unit-blind matcher) |
| Q4 | meta-backend assert outranks the OOM it triggers → `HARNESS_FAILURE`, not the wall (09-12 B1) |
| Q5 | empty-body 200 `/health`: readiness is the HTTP code, never the body (09-02 / 09-10) |
| Q6 | hard crash mid-campaign → watchdog restores, a fresh runner resumes (09-10) |
| Q7 | restore idempotent across a double exit (09-10's non-converging exit) |
| Q8 | host reboot mid-campaign → stop, forensics, no auto-resume (09-10) |
| Q9 | PVE dialect: only tabled command syntax renders; hand-rolled forms are rejected (09-12 maint-fn) |
| Q10 | false-complete: a 2-token 50 ms answer fails the sized gates → `INVALID` (09-10) |
| Q11–Q14 | P1: malformed SSE, unsupported-arch control cell, no-usage stream shape, corpus replay |

P0 must be 10/10 for a bundle to be frozen. P1 warns.

## Repair policy

Declared in the spec, enforced in the runner: a small allowlist of repair
classes (path, command syntax, parser exception, log-format support, missing
fixture copy, quoting) may be applied by the executor against a *copy*; every
change is re-qualified and recorded in the attempt's `meta.json` with both
hashes. Everything touching a science field requires the owner. The forbidden
list (delete evidence, force a pass, skip a cell, rewrite a previous attempt)
is structural: attempts are immutable directories.
