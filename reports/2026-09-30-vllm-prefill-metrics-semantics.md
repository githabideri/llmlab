# vLLM prefill metrics in the llm-hub: why "5,000 t/s pp" was wrong, what /metrics actually does during a chunked prefill, and the three-number semantics the hub now uses (2026-09-30)

*Scope: the hub's vLLM metrics path (`hub/llm-hub.py`), its UI, and the
Prometheus export. The llama.cpp path is unchanged except for documentation.
The measured engine: the in-production 27B W4A16 model on the dual-RTX-3090
node (TP2, 250 W caps), vLLM 0.28.0, behind the hub's vllm-mux.*

## The three problems

1. **A 5,000 t/s "pp/s" reading** that no benchmark on this hardware supports
   (~1.1–1.3K t/s is the measured prefill speed of this model).
2. **No numbers at all during an initial prefill** — a 28-second prefill
   showed only a dim `prefill…` cell.
3. **`degraded` state from normal operation** — one response hitting its
   max-tokens cap, or a 94 %-full KV pool on a legitimate long context,
   turned the card red.

## The root cause (probed, not assumed)

The hub's old vLLM pp/s was `Δ(computed prompt tokens) / Δ(2 s poll gap)`.
The probe (a novel ~33K-token prompt, 300 ms sampling of the engine's
`/metrics`, both direct and through the mux) established the counter
behaviour of this vLLM build:

| Question | Answer (vLLM 0.28.0) |
|---|---|
| Does `prompt_tokens_by_source_total{source="local_compute"}` advance *during* a chunked prefill? | **No** — flat for all 47 in-flight samples of a 28 s prefill; one jump of the full 33,377 tokens at first token |
| Does `iteration_tokens_total` move during prefill? | No — one burst (the prefill counted as one iteration) at the end |
| Does `generation_tokens_total` stay flat until decode? | Yes (Δ=0 until the first output token) |
| When do `request_prefill_time_seconds` / `request_prefill_kv_computed_tokens` first change? | At request **completion** (after the decode tokens), not at first token. Only the TTFT histogram moves at first token |
| Does the mux's `/metrics` forwarding batch or delay? | No — 1,056 direct-vs-mux counter comparisons, 1 mismatch (an intra-tick scrape-ordering artifact), zero fetch errors |
| Accounting consistency? | Exact: Δ computed = Δ prefill-KV-computed = Δ prompt-total = 33,377 (0 cached) |

So the "5,000 t/s" was the full prefill lump divided by the 2 s poll gap in
which it landed — a measurement artifact of the same class the llama.cpp
branch already fixed against its child clocks (2026-09-19). And the official
practice matches the correction: vLLM's own reference dashboard computes
token throughput as a windowed `rate()` of a monotone counter, and its
production dashboard has **no** short-window prefill-throughput panel at
all; the per-request phase histograms are the project's own
"prefill performance" surface.

The only signals that *do* move in-flight are `num_requests_running` and
`kv_cache_usage_perc` (the latter in stepped ramp as blocks are allocated).

## The three numbers (and the rule: never merge them again)

**A. Prefill execution speed — the headline `pp/s`.**
`Δ(request_prefill_kv_computed_tokens) / Δ(request_prefill_time_seconds)`
over the completed requests in the hub's 5-minute window. The engine's own
per-request phase clocks; cached/transferred prompt tokens excluded;
independent of the arrival pattern. This is the number comparable to a
controlled prefill benchmark — the probe's own request: 33,377 tokens /
28.15 s = **1,186 t/s** (wall-clock: 1,181 t/s, 0.4 % apart). It carries a
sample count and an age tag once the window's last completed prefill is
> 1 min old, so a stale figure never masquerades as current. Caveat,
documented in the UI: under concurrency it is an *aggregate ratio* over
completed requests, so it can sit below the wall rate of a single request
when the window mixes sizes.

**B. Prompt compute throughput — the operational figure.**
The computed-prompt counter, exported as a monotone Prometheus counter
(`hub_model_prompt_tokens_computed_total`, re-baselined; an engine restart
or mux engine switch is a counter *reset*), plus a hub-computed 1-minute
average gauge (`hub_model_prompt_compute_throughput_tokens_per_second`).
The 1-minute window is deliberate: the counter's arrivals are lumps, so a
short window is *more* wrong, not less (5 s would show a 33K prefill as a
~6K t/s spike for the whole minute). This answers "how much prefill work
did the server do per wall second", not "how fast does it prefill".
`rate(hub_model_prompt_tokens_computed_total[1m])` is the canonical
PromQL form.

**C. Live in-flight prefill — an honest timer, no rate.**
While a request is in the prefill phase (requests running but neither the
generation counter nor the TTFT histogram advanced on the last poll) the
card shows `prefill Ns…` plus the live kv/req cells. **No rate is shown or
invented**: this build exposes no token counter until first token, so any
in-flight number would be fabricated. Under concurrent load the detector
can be masked by other requests' decode; that is accepted over a false
number. (The 30-minute sparkline's pp lane now plots the 1-minute work
rate for vLLM, and the child-clock rate for llama.cpp.)

Supporting diagnostic: the 5-minute `prompt work` line in the card —
*total requested* vs *computed* vs *cache-served* prompt tokens. On this
cache-hot engine the split is ~99:1:0.8 (observed: 1.18M requested,
16.7K computed, 98.6 % cache-served) — which is exactly why the raw
counter must never feed a performance number again.

## What changed

**`hub/llm-hub.py`** — the vLLM poller:
- `tpp` for vLLM models is now the 5-min phase-clock prefill speed (it was
  a 2 s wall-rate of the lumpy counter); the 1-min work-rate gauge is a new
  field (`pp_tput_1m`) with reset/gap re-baselining of its rolling ring;
  new fields: `prompt_work`, `live_prefill`, `prefill_speed_age`,
  `prompt_tokens_computed_raw`.
- A mux engine switch now clears *all* short-window working state
  (previously only the 5-min ring), so no rate, age, or detection timer
  straddles two engines.
- Prometheus: new `hub_model_prompt_tokens_computed_total` (counter),
  `hub_model_prompt_compute_throughput_tokens_per_second` (1-min),
  `hub_model_prefill_in_flight`, `hub_model_prefill_speed_age_seconds`;
  `hub_model_{prefill,decode}_avg_seconds` **renamed**
  `hub_model_{prefill,decode}_{p50,p95}_seconds` (they were quantiles,
  never means); `hub_model_prompt_tokens_per_second` is now
  **llama.cpp-only** (its semantics were always correct — child-clock
  rate). The old combined gauge is retired, not redefined.

**`hub/ui/index.html`** — the `pp/s` cell shows the phase speed (with the
age tag); the prefill cell is the honest timer; the kv cell gains a
pressure chip at ≥ 90 %; the length-limited chip is now explicitly
neutral. **State derivation** (the source of truth for both UI and README):

| state | new rule |
|---|---|
| busy | waiting > 0 ‖ deferred > 0 ‖ kv ≥ 75 % (kv ≥ 90 % adds a pressure chip) |
| degraded | a preemption in the window ‖ stalled engine ‖ failed `/metrics` parse ‖ queue building against kv ≥ 90 % |

`finish_reason=length` and KV occupancy alone no longer affect the state
at all — one (or 100 %) length-limited responses and a 95 %-full pool with
no queue or preemptions are `busy`/`active`, never `degraded`.

**Consumers** (in the monitoring repo): the `homelab:llm_prompt_per_second`
recording rule now joins `hub_model_prompt_tokens_per_second` (llama.cpp)
`or rate(hub_model_prompt_tokens_computed_total[5m])` (vLLM); the
gpu-llm panel description documents that the two branches answer
different questions; the metrics catalog and agent query guide carry the
new series.

**Tests** — `hub/test_vllm_metrics.py` (12 hub tests + the UI state matrix
run in node against the real `modelState` function): the 2 s burst
artifact, phase-speed correctness, the cached-prompt split, counter reset,
mux switch re-baselining, initial prefill, length/100 %-length, high-KV,
preemption, missing phase histograms, and the Prometheus surface
(monotone counter, reset as a drop, retired names gone).

## Migration note

- `hub_model_prompt_tokens_per_second`: vLLM series **disappear** on
  upgrade (they were the artifact); llama.cpp series are unchanged.
- `hub_model_prefill_avg_seconds` / `hub_model_decode_avg_seconds`:
  **renamed** to `*_p50_seconds` (+ new `*_p95_seconds`). Dashboards and
  alerts referencing the old names need the rename.
- New canonical vLLM series: `hub_model_prompt_tokens_computed_total`
  (counter), `hub_model_prompt_compute_throughput_tokens_per_second`,
  `hub_model_prefill_speed_tokens_per_second` (already existed),
  `hub_model_prefill_in_flight`, `hub_model_prefill_speed_age_seconds`.

## Verification

- Unit: 12/12 hub tests + 9/9 UI state cases.
- Live, through the deployed hub: cold ~6.7K-token prompt (0 cached)
  completed in 5.3 s wall (≈1.26K t/s wall) with the phase-speed cell
  reporting the blended completed-request ratio (no 5–7K spike anywhere);
  identical-prompt rerun (75 % cache-served) with flat numbers; a
  max-tokens completion with no degraded inputs; the counter monotone
  across scrapes; the 33K-token probe request: 1,186 t/s phase vs 1,181
  t/s wall.

## Addendum (same day, after external review): the restart-window bug, the recording-rule split, MTP and request shape

An external review of this change (ChatGPT, forwarding the commits) confirmed
the core semantics and found one real bug plus a batch of follow-ups, all
fixed the same day:

**1. Restart-state bug (the one that mattered).** The counter-reset
detection re-baselined the 1-min computed-token ring, but on an engine
process restart *with the same model ID* the 5-minute window (phase
histograms), the live-prefill detection and the phase age were **not**
cleared — pre-restart phase sums would have sat as window baselines until
they aged out. The first round's reset test passed only because its
pre-restart sample carried zero phase data (exactly the gap). Now any core
counter decrease clears all engine-derived rolling state (as a mux engine
switch does) and re-baselines on the restart sample; the test now uses a
live pre-restart world (900 s / 1.1M KV-computed) and asserts the
post-restart window is the clean world-only ratio (5,000 KV / 4 s =
1,250 t/s, no straddle).

**2. The recording rule is split.** `homelab:llm_prompt_per_second` joined
llama.cpp's *execution speed* with vLLM's *wall-clock work rate* under one
name — the very mixing this fix exists to end. It is retired in favour of
`homelab:llm_prefill_speed_tokens_per_second` (vLLM phase clocks; llama.cpp
absent by design — it has no phase histograms) and
`homelab:llm_prompt_compute_throughput_tokens_per_second` (one quantity:
tokens per wall second, child-clock for llama.cpp, 5-min `rate()` of the
computed counter for vLLM). Dashboard: the old panel became "Prompt compute
(tok/s, wall)"; the new "Prefill execution speed (tok/s)" panel sits above
it. Metrics catalog and agent query guide updated.

**3. Wording tightened.** The 1-min gauge no longer claims "true average
work rate" — it *amortizes lumpy counter updates over a stable wall-clock
interval to represent operational prompt-compute throughput* (a 33K-token
prefill arriving as one 28 s lump reads ~553 t/s there while its phase
speed is ~1,186 — the two stay apart on purpose). The prefill-speed age
tag is labelled "how long ago the last completed request contributed a
prefill measurement" (the histograms are sampled at request completion,
not TTFT — which this report's probe established).

**4. External KV transfer modelled.** The prompt-work split is now
requested / computed / local-cache-served / **external KV transfer**
(`prompt_tokens_by_source{source="external_kv_transfer"}`; zero on the
current build, present for LMCache/disaggregated serving), with the
cache-served figure as local+external.

**5. Prometheus contract.** The `/metrics` exposition now declares `# HELP`
/ `# TYPE` for every series (static header); the counter-ness of
`hub_model_prompt_tokens_computed_total` is stated, not implied by the
suffix.

**6. MTP telemetry (the one vllm-monitor idea worth adopting).** The
engine's official counters (`spec_decode_num_{drafts,draft_tokens,
accepted_tokens}_total`, per-draft-position) are reduced over the window:
acceptance = accepted/draft tokens, mean acceptance length =
`1 + accepted/drafts` (the +1 is the bonus token vLLM documents), and
per-draft-position acceptance. Card: `mtp [5m]` (window deltas, not
lifetime ratios); Prometheus: `hub_model_spec_acceptance`,
`hub_model_spec_accept_length`. Live on the k=3 model: 55 % acceptance,
2.65 tok/step, position curve 71/50/37 % — position 2 still earns its
verification cost, k=3 stays. (vllm-monitor's raw `prompt_tokens_total`
"Prompt Tokens/s" was deliberately *not* copied — on a ~99 % cache-hit
workload it is the old bug in another body.)

**7. Request shape.** Window means of the engine's per-request
`request_prompt_tokens` / `request_generation_tokens` histograms — the
context that makes latency interpretable (a TTFT p95 jump means different
things at 4K vs 45K average input; the live window read ~95K in, 430 out).
Card: `req shape [5m]`; Prometheus: `hub_model_prompt_tokens_mean`,
`hub_model_generation_tokens_mean`.

Tests: 17 hub unit tests (the old A–J plus the genuine restart scenario,
MTP, request shape, external KV, TYPE/HELP/surface) and the 9-case UI
state matrix, all passing.

## Addendum 2 (same day) — round-3 correctness pass

A second external review of the committed work found three remaining
semantic problems. All fixed, tested, and deployed:

1. **The throughput rule was still mixing unlike quantities.** The
   round-2 rule's llama leg used the hub's *child-clock* prompt rate
   (tokens / engine prompt-phase seconds) — a *speed* — under the
   "tokens per wall second" name. The fix normalizes the *concepts*:
   `homelab:llm_prefill_speed_tokens_per_second` now carries **both
   engines' prompt execution speeds** (vLLM phase clocks OR llama.cpp
   child clocks — both divide by the engine's own prefill clock; the
   dashboard's speed graph is now cross-engine, with the caveat that the
   token bases differ slightly: KV-computed vs prompt tokens).
   `homelab:llm_prompt_compute_throughput_tokens_per_second` is
   **vLLM-only** (rate of the monotonic computed counter); a llama leg
   appears only when the hub exports a monotonic llama prompt-work
   counter — a speed must not masquerade as a wall-clock work rate.
2. **External-KV accounting.** The test used an impossible partition
   (400 + 2800 + 800 = 4000 against 3000 requested); vLLM maintains
   `computed + local_cache + external_kv == requested`. The test now
   uses a valid partition, the hub sanity-checks the identity (±2%)
   and flags **accounting drift** instead of displaying an impossible
   split, and the percentage attached to the combined "cache-served"
   value is the **combined** (local + external) ratio, with the
   local-only ratio split out only when external KV is non-zero.
3. **The `hub_model_tokens_per_second` HELP was false for vLLM.** It
   claimed a decode-phase clock; vLLM's value is Δtokens / wall-clock
   (aggregate service throughput). The HELP now states the
   engine-dependent clock per engine kind, and the UI tooltip matches.
   (A vLLM decode-phase-seconds counter does not exist in this build,
   so a true vLLM decode speed is not derivable from /metrics — the
   per-request decode figure is the phase-histogram one in
   diagnostics.)

Also: request-shape means now divide by **each histogram's own request
count** (parsed separately; they normally agree, and parallel sampling
can diverge upstream), and `hub_model_preemptions_total` is declared a
counter (it is a monotonic engine-lifetime counter). 18 hub tests +
9 UI state cases.
