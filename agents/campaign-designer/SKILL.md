---
name: campaign-designer
description: >-
  Design GPU inference measurement campaigns: the science (what cells answer the
  question, what gates mean, what a negative result is), the platform contract
  (benchmarks/ runner: freeze, verdict engine, effect gates, window + watchdog,
  repair policy), measurement-method discipline (exact tokens, median-of-N,
  cold/warm, residency, provenance), and the review lessons (PASS-only estimators,
  below-floor vs unverifiable, honest degradation to NOT_ESTABLISHED). Use when
  planning or reviewing any multi-stage measurement campaign, or when writing a
  campaign spec for the llmlab campaign platform.
---

# campaign-designer — Designing Inference Measurement Campaigns

A campaign is a *question with evidence attached*. This skill is the design
view: what to prove, in what order, with which gates, and what it means when
anything fails. On the llmlab platform ([benchmarks/](../../benchmarks/README.md))
the spec is data and the runner is the enforcer — a campaign designed with this
skill compiles to a spec that the platform runs unattended. The operational
companion is [docs/benchmarks.md](../../docs/benchmarks.md).

## 1. Start with the question, not the hardware

Every campaign answers ONE scientific question ("does mechanism X provide
useful effect under config Y?"). From it, derive:

- **Cells in dependency order.** The first cell validates that the mechanism
  you intend to measure is actually ON (S0 pattern): assert the system's own
  log (placement line, allocation line, no-fallback pattern) plus a live
  probe. Nothing after S0 is interpretable until S0 passes — a throughput
  number produced by a silently-fallen-back mechanism is not data.
- **Paired comparisons, not guessed absolutes.** When the effect of a
  mechanism is the question, measure ON vs OFF under identical conditions
  (seed, prompt, decode, hardware, co-residency) and compare. Absolute
  performance floors are sanity checkers (they catch degenerate runs —
  hangs, lost tokens, broken streams), they are never the instrument that
  judges usefulness — absolute numbers from a *different* configuration
  regime (single vs tensor, other card, other build) have no calibration
  there and importing them manufactures both false PASSes and false
  "the mechanism failed" verdicts.
- **Expected negatives declared up front.** A mechanism known to be
  unsupported in this build/arch is a *documented negative* (a completed
  campaign that answered its question), not a failure. Say so per cell.
- **Dependencies are the smallest mechanism that expresses them.** The
  platform's `effect_gate` is the one dependency construct: a dependent cell
  runs only if a declared floor over *persisted* evidence of earlier cells
  clears. Do not invent workflow languages; do not let a dependent cell
  start on an unverified premise.

## 2. Gates: know which kind you're writing

For each gate answer: *"why does crossing this invalidate the result?"*

| Kind | Example | Meaning of crossing |
|---|---|---|
| Measurement contract | exact token count, within-config determinism | the measurement is broken → INVALID |
| Sanity floor | `min_wall_s`, cheap `min_tps` — **REVIEW-grade only** | the run *looks* degenerate → REVIEW_REQUIRED; generous on purpose, it does NOT predict performance, and it is never the primary no-op detector (see §3) |
| Comparison | effect gate floor, paired differences | the *question's answer* → below floor is a RESULT (expected-negative), not a defect |
| Interpretation criterion | divergence thresholds, coherence review | reported in the analysis, never enforced — the value is the finding |

Unsized gates are how false-completes pass (the 256K-ladder incident: a
gate that never fired is not a gate). Gates are **per-cell data** — the same
gate value is right for a 35B 512-decode cell and wrong for a 0.5B 32-decode
cell; copying a gate across workloads is a known defect class.

## 3. Evidence discipline

- **A result exists only when it is a persisted row.** Client JSON with
  exact counts, full generated text, server log, classification, verdict —
  per attempt, never rewritten. Numbers read off a terminal or quoted in
  chat are not data.
- **Exact tokens, server-side.** Take token counts from the engine's
  reported numbers, cross-checked against the client's own count. Under
  speculative decoding a streamed chunk is NOT a token; SSE chunk count as a
  token counter is a defect.
- **Median of N, keep every attempt.** MLPerf's rule: N independent runs,
  score the median. Failed/invalid attempts stay on disk as evidence — and
  they are excluded from estimators (below).
- **Integrity over timing (2026-09 lesson).** Prefer *direct evidence that the
  declared work happened* over performance-based plausibility proxies. The
  per-request **work contract** — declared ≈ client-encoded ≈
  server-observed prompt tokens (tolerance `max(32, 1%)`) plus server-side
  request-occurrence evidence — is the validity gate; wall-time floors are
  weak anomaly detectors and **must never reject the phenomenon being
  measured**. Counterexample (LMCache campaign): a cell-level `min_wall_s`
  misfired on 1.5 s *cached returns* (the signal) and invalidated three
  healthy controls, while *missing* a never-sent 60 K "filler" (1.7 s) that
  token accounting would have caught. A request's **kind** (`first_touch`,
  `cached_return`, `pressure`, …) selects *which evidence is required* —
  not which time constants apply. Timing floors on return legs are always
  wrong: the speedup *is* the result.
- **Cold vs warm is a first-class axis.** Repeated fixtures silently hit
  prefix caches and fake prefill; a unique nonce per request for cold,
  deliberate repeats for warm. If the question involves first-use behavior,
  a dropped page cache is the honest cold condition and the restore path
  becomes the test — size the restore budget from observed cold starts.
- **Provenance is part of the result.** Model artifact (filename, bytes,
  SHA256 of what the log says it loaded), the FULL build stack (upstream
  base + PRs + local patches + flags), driver, engine, co-resident process
  inventory. "Which file" matters as much as "which model"; same-named
  files differ between builds, and official artifacts are not
  automatically the best ones.
- **Residency and power are claims that need instruments.** "GPU-resident"
  is a PCIe-counter claim (decode timings look plausible either way);
  wall-power comparisons integrate at the plug with state-hold semantics
  (history coalesces flat stretches — a fixed-dt integrator undercounts).

## 4. Review lessons (2026-09-13 external review — the unattended-execution rules)

These cost a review round; encode them in every campaign you design:

1. **Dependent science uses only valid evidence.** An estimator over
   persisted attempts must select on the attempt's verdict (class == PASS).
   A failed attempt (1 t/s, harness error) is preserved as evidence but
   never feeds a median. Record WHICH attempt IDs were used so a reviewer
   can recompute the comparison verbatim.
2. **Below-threshold and unverifiable are different verdicts.** A measured
   gain under the declared floor is a *result* (the hypothesis is weak; the
   campaign completed: expected-negative). A missing/malformed premise
   (no PASS evidence, missing metric, malformed evidence) is the *absence
   of a result* (review-required). Conflating them lets "we couldn't
   measure" masquerade as "it doesn't work" — fatal when nobody is
   watching.
3. **Best-effort parsing degrades to NOT_ESTABLISHED, never to a
   valid-looking zero/equality.** When a response shape is unknown (token
   records differ across engine versions; a field may be int / str / dict /
   candidate-list), unknown normalizes to *nothing*: the position is not
   comparable, the metric is null, the output says `not_established`. An
   integer token stream and a text stream are different evidence; comparing
   them (or letting `None == None`) manufactures apparent identity.
4. **Identity claims are bounded.** A bounded sample can only support
   "no divergence observed in the tested sequences" — with the count of
   compared positions and the count of not-established ones next to it.
   Never "bit-identical" from a sample; and a 100%-overlap result on a
   mechanism with KNOWN divergence is "not established at this length",
   not proof.
5. **A bad cell cannot be masked by a good one.** Campaign-level folding is
   conservative: any HARNESS_FAILURE / INVALID / UNKNOWN cell makes the
   campaign review-required, before considering any cell's documented
   negative. "We got a result" and "an instrument broke" are different
   outcomes.
6. **Regressions exercise the lowest real boundary that reproduces the
   failure.** The platform's qualification matrix maps the fixture backend
   (local processes, real sockets, a simulated watchdog) onto each failure
   class; only semantics the fixture cannot express (real SSH transport,
   real host failure hardware, cold restore under load) are reserved for
   one-off live drills. After a fix: targeted regression + full matrix.
   Another live production window is NOT the acceptance criterion — it is
   the escape hatch, used only when a fix changes watchdog / window
   lifecycle / prod stop-start / restore / backend / destructive-host
   semantics.

## 5. The platform contract (what your design compiles to)

The llmlab campaign platform enforces, and your design should therefore
declare, rather than reimplement:

- **Freeze**: bundle = `git archive` of the llmlab commit; `freeze.json`
  records the *scientific* hash (spec semantics), the *implementation* hash
  (bundle), model SHA, and the freeze time. A repair that touches a science
  field fails its assertion by construction; implementation fixes are
  re-frozen and logged.
- **Window + watchdog**: one `enter` (stop prod, apply temp changes, arm an
  out-of-band watchdog) / one idempotent `exit` (undo in reverse, restore
  prod, live-check with a real completion, disarm). The watchdog is
  owned by the operator (NOT shipped in the bundle — it must work when the
  bundle is corrupt), fires on heartbeat staleness or deadline, and can
  only start the prod unit, never the campaign. Arm-time verification: the
  watchdog's own sensor is live-checked while prod is known-healthy, before
  prod is stopped; a broken sensor refuses the window.
- **Verdict engine**: per attempt — classify (the failure log, with
  observed / inferred / not-established labeling), then gates, then verdict
  (PASS / EXPECTED_NEGATIVE / INVALID / HARNESS_FAILURE / RESOURCE_LIMIT /
  RETRYABLE_* / UNKNOWN / SAFETY_ABORT). `UNKNOWN` → review-required; it is
  never auto-passed.
- **Repair policy as data**: `allowed` (harness-local, agent may apply +
  re-qualify) / `requires_owner` (thresholds, workload, model, build-when-
  scientific) / `forbidden` (delete evidence, skip cells, rewrite attempts).
  A bounded, logged, re-qualified repair that lets the campaign finish is a
  *success* of the design.
- **Host-failure detection**: a kernel-log scan scoped to the window's own
  epoch (an old boot line is not evidence against the run), plus crash-dir
  and uptime checks. True positives stop the campaign, collect forensics,
  restore prod, and never auto-resume.
- **Stop policy**: campaign deadline, per-cell wall ceiling, disk floor —
  and the documented-negative stop: when a declared expected-negative
  cell fires, the campaign's question is answered; stopping is the success
  path.

## 6. Method details that cost incidents (durable)

- **Don't touch production.** Never stop the endpoint that serves the
  campaign controller's own model; move the controller to an independent
  endpoint first and verify with a live request. Never aim a
  memory-exhaustion test at an endpoint anything else depends on — it is a
  denial of service, not a load test.
- **Health shapes differ per engine**: JSON `{"status":"ok"}` (llama.cpp,
  spacing varies — match the JSON, not the string) vs HTTP-200-empty-body
  (vLLM — match the code). Readiness = the right shape, not any 200.
- **Warm before you measure**: first requests pay graph-capture/JIT costs
  and can show one-off transient crashes; a discarded warmup rep is part of
  the design, and it is recorded as such.
- **Pre-boot VRAM budgeting**: weights (per rank if TP) + draft + graph
  overhead + KV at target context. If it doesn't fit, DON'T boot — a
  boot-OOM is a data point that defines the config family's ceiling, and
  the exact allocation-shortfall line is the evidence. Some engines crash
  in the loader instead of refusing an oversized pool; budget before boot
  rather than learning from the crash.
- **Flag/version drift is a discipline, not a table**: pin the build
  (commit/version), read its `--help` fresh, record the mapping in the
  campaign notes. Tables of specific builds rot; the discipline doesn't.
  (Same for dialects: only verified command syntax renders in a PVE
  maintenance script, per version × guest type — an unverified syntax is a
  prepare-time error, not a runtime surprise.)
- **Hybrid/recurrent models**: capacity from the engine's startup
  profiler, not from attention-KV formulas; recurrent state and page
  allocation add engine-specific overhead.
- **Co-residency is a measured variable**: shared cores depress decode 2–3×;
  record the inventory per cell and compare only matched co-residency.
- **Concurrency**: report per-request AND aggregate; under chunked prefill
  per-request decode collapses while the GPU sits at 100%. Distinguish
  batchable-short-prompt from long-context workloads; compute the KV
  ceiling from the engine's startup log before running.
- **Speculative decoding needs a correctness gate before any of its
  numbers are accepted**: coherent on-topic output at short AND deep
  context, no garble/runaway. A high token rate over corrupted text is not
  a result. Its acceptance statistics live in the server log, not the
  client.
- **Scheduling**: gate against the host's own time and known noise windows
  (nightly backups), polling with short sleeps — and verify the host's
  timezone first (one rack has observed disagreeing offsets).

## 7. Public boundary

This repo is public (Gitea + GitHub). Campaign content that lands here:
dated reports (frozen; supersede-append allowed, no silent edits), living
`docs/` as the only maintained surface, no internal identifiers (hostnames,
IPs, container numbers, secrets) — a sanitizer sweep with a checked-in
substitution table runs before every push. Campaign *specs* for internal
hardware live in the private homelab repo; this repo holds the platform,
the methods, and sanitized results.

## 8. Lanes: chop first, paved when proven (2026-09-16)

Before writing a paved campaign for a mechanism that has never booted on the
target stack (new connector, new engine version, first-of-kind feature), run
the **chop lane** first: a brief-driven session where the agent has free hand
*inside declared hard boundaries* to make the mechanism work, with an
append-only decisions log and a stop rule. Discovery failures (engine-arg
validation walls, config shapes, allocator conflicts) are only visible at
boot — paying a full prod-impact window to learn one line of error at a time
is the anti-pattern this lane exists to kill. The probe is the cheapest chop
instrument: same command, tiny memory reservation, scratch port — validation
walls fire identically, and nothing touches production.

The paved campaign then **freezes the chop lane's working state** and
measures it. Design the paved spec from the *proven* configuration, not from
the hoped-for one. `docs/benchmarks.md` ("Execution lanes") is the reference
for the contract between the two.
