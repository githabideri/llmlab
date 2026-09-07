# LLM hub adaptive vision routing — project design brief

**Date:** 2026-09-07
**Category:** Infrastructure (design)
**Status:** Phase 1 (read-only advisor) implemented in `hub/` in this commit; Phases 2–3 are preliminary research tracks, deliberately not implemented.

**TL;DR**

- A four-day vision outage exposed that a *static* vision primary/fallback list cannot survive a small fleet's real dynamics: one of the two "redundant" models on the primary node had been evicted by the router's `--models-max 1` residency limit, and the fallback shared the same node *and* had its only `parallel=1` slot monopolized by long-context text prefills (50–76K tokens, tens to hundreds of seconds each). Every image description silently failed.
- The fix is a **read-only endpoint advisor** in the existing `llm-hub` service: a new authenticated `GET /api/vision/route` that returns the best currently usable vision endpoint (plus alternates and machine-readable reason codes), computed from state the hub already polls — model load state, input modalities, busy/deferred request counts, slot parallelism, GPU telemetry, and freshness.
- The advisor is a **decision-maker, not a proxy**: the vision consumer (e.g. the `pi-vision-handoff` Pi extension) receives the recommendation and sends the OpenAI-compatible request **directly** to the chosen server. No image or prompt payload ever enters the hub; the hub staying down never breaks inference (consumers keep a static emergency fallback).
- The hub already derives almost every signal needed (`n_proc`, `n_deferred`, `busy_slots`, prompt/generation rates, GPU util/VRAM); Phase 1 adds **input-modality capture** (currently dropped from `/v1/models`) and a `--parallel` extraction, then a pure, unit-testable `route_vision()` function plus the endpoint, metrics, and config.
- **Phase 1 mutates nothing** — no model load/unload, no slot save/restore, no preemption. A previously proposed "auto-heal evicted vision model" rule is explicitly *not* included: on a `--models-max 1` router, auto-loading one model implicitly evicts another, so "auto-load only" is not non-destructive. Visibility and alerting come first.
- Measured fleet vision speed (1920×1080 fixture, thinking off): the dual-3060 35B MoE is 2–3× faster than either single-3060 35B and ~3× faster than the 27B, which also consumes **2× the image tokens** for the same picture (2,057 vs 1,025) because of its different vision tower. "Same host = next-best candidate" is false; the policy encodes the measured order.

---

## 1. Motivation: the incident

The fleet's coding-agent harness describes images with a vision model (the
[pi-vision-handoff](https://github.com/monotykamary/pi-vision-handoff) extension for
Pi: images are intercepted, described by a configured vision model, and replaced
with text before a text-only chat model sees them). Its configuration was:

```
primary  = node A / 35B MoE
fallback = node A / 27B dense
```

Node A runs the llama.cpp **router mode** with `--models-max 1`: the two models
(35B-A3B MoE at ~11 GB VRAM, 27B dense at ~10.8 GB) do not both fit across the
two 12 GB 3060s, so loading one evicts the other. The two "redundant" vision
models were therefore **mutually exclusive** — they could never be the two legs
of a failover chain at the same time.

While the 27B happened to be the loaded one (evicting the 35B during an earlier
manual test and never restored), every primary call failed immediately with
`400 "model is not loaded"`, and the fallback — the very model that had
*caused* the primary to vanish — timed out at 120 s. It timed out not because
the description was intrinsically that slow, but because a long-context text
session on the same node kept pinning the 27B's single execution slot with
50–76K-token prefills taking 73–150 s each (visible in the server journal as
`cancel task` lines matching the client-side timeouts 1:1). For roughly four
days, every image in those agent conversations came back "description
unavailable", and nothing in the fleet reported it.

Root causes, in order:

1. **Static routing** — the consumer had no knowledge of live model residency or slot pressure.
2. **Same-failure-domain fallback** — primary and fallback shared one node *and* one residency slot.
3. **No repair or alerting** — the hub could see the 35B was evicted, but is an observer; nothing acted, and no alert defined "usable vision capacity" as a monitored quantity.
4. **No per-node availability policy** — one node is powered off outside a nightly window, another shares its GPU with a long-running document-AI workload; a static list cannot encode either fact.

## 2. Measured fleet vision capability

Same 1920×1080 synthetic fixture (screenshot-like content), one pixel
nonced per run to defeat `--cache-prompt`, `chat_template_kwargs:
{enable_thinking: false}`, three warm runs each. Node identities are
anonymized; hardware per `hardware/` docs.

| Node | Model (as deployed) | Image tokens for the fixture | Time-to-first-token | Decode |
|------|---------------------|------------------------------|---------------------|--------|
| A — inference LXC, dual RTX 3060 (tensor split, `--models-max 1`, 35B `parallel=2` / 27B `parallel=1`, both CPU mmproj) | Qwen3.6-35B-A3B (MoE, ~3B active) | 1,025 | **12.2 s** | **~110 t/s** |
| A (same node) | Qwen3.8-27B dense | **2,057** | 37.5 s | ~34 t/s |
| B — backup/inference box, single RTX 3060 (powered on during a nightly window only) | Qwen3.6-35B-A3B | 1,025 | 26.6 s | ~35 t/s |
| C — secondary GPU server (i5-7400), single RTX 3060 shared with document-AI containers | Qwen3.6-35B-A3B | 1,025 | 31 s | ~28 t/s |

Consequences the policy must respect:

- **A/35B is the clear preferred route** (dual-GPU prefill, MTP decode, `parallel=2` so a second concurrent description can run).
- **A/27B is the *worst* vision route in the fleet despite sitting on the strongest host**: 2× the vision tokens (different vision tower → different patch/merge ratio), dense compute, single slot, and it competes with the very long-context text sessions that cause the outages. It is an *emergency* route, not a "next-best" one.
- All vision encoders in the fleet run **CPU mmproj** (`no-mmproj-offload` in every preset, chosen to keep symmetric tensor splits at large ctx). The TTFB above is dominated by mmproj encoding + vision-token prefill; expect the same relative order for real screenshots (which were 5–7K prompt tokens in the incident, i.e. ~2–4× slower than this 1080p fixture's 1K).

## 3. Architectural decision: advisor, not proxy

```
vision consumer (e.g. pi-vision-handoff)
      |
      | 1. GET /api/vision/route?images=N&max_width=..&max_height=..
      v
   llm-hub (decides, from asynchronously polled state)
      |
      | 2. JSON: chosen {server,url,model} + alternates + reason codes
      v
vision consumer
      |
      | 3. direct OpenAI-compatible multimodal request
      v
chosen inference server
```

Properties this preserves:

- **Hub failure ≠ inference failure.** The consumer falls back to its existing
  static model list (today's behavior) when the advisor is unreachable.
- **No payload in the hub.** No image bytes, no prompt, no SSE/streaming
  forwarding, no cancellation propagation, no credentials for backends
  (they're keyless on the trust domain). The hub stays stdlib-only and
  restart-safe.
- **One owner of routing policy.** Consumers remain deliberately dumb
  resolvers: they build request-shape metadata, GET the route, send the
  request, try the advisor's alternates on immediate failure, then the static
  emergency fallback. Fleet topology changes (preferred node, weights, power
  windows, contention thresholds, new vLLM/SGLang vision backends) require
  **no consumer-side changes**.

The pattern mirrors the Endpoint-Picker separation in the
[Kubernetes Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)
and the Filter → Score → Pick scheduling of [llm-d](https://llm-d.ai/) —
except our picker returns a *recommendation* to the client instead of a proxy
forwarding the request, which is the right size for a 3–4 node fleet.

### Why not a full LLM gateway

A in-path gateway (LiteLLM, vLLM production-stack router, SGLang model
gateway) would give consumers one permanent URL — but it concentrates
credentials, request bodies, plugin/tooling APIs, and a data-plane lifecycle
(SSE, retries, partial responses) in one service, and makes that service a
hard dependency of the inference path. The dependency surface matters: the
[March 2026 LiteLLM PyPI supply-chain compromise](https://github.com/BerriAI/litellm/issues/24518)
(malicious `1.82.7`/`1.82.8` releases; [PyPI incident report](https://blog.pypi.org/posts/2026-04-02-incident-report-litellm-telnyx-supply-chain-attack/),
[Microsoft analysis of gateway risk](https://www.microsoft.com/en-us/security/blog/2026/08/26/when-ai-infrastructure-becomes-target-securing-gateways-control-points/))
and separate 2026 gateway application vulnerabilities
([GitHub security advisories](https://github.com/BerriAI/litellm/security/advisories))
are not an argument against LiteLLM's routing *concepts* — they are an
argument for keeping our attack surface at the size of one authenticated
JSON endpoint. (The supply-chain event and the application vulnerabilities
are distinct incidents and should be treated as such.)

## 4. Selection algorithm (Phase 1)

The policy is deliberately boring: **eligibility first, capacity class, then a
simple ordering** — not a tuned scoring DSL.

**Normalization.** Each loaded-model candidate is normalized to:

```
identity:      server, model, api url, backend kind
capabilities:  input_modalities (from /v1/models; config override allowed),
               reasoning control (consumer-side, not a routing input)
health:        server online, state age (s), metrics ok
capacity:      parallelism (--parallel), busy_slots, requests_processing,
               requests_deferred  →  has immediate capacity?
accelerator:   max GPU util %, memory used/total (sidecar), telemetry age
performance:   prompt t/s, generation t/s (already hub-derived)
policy:        vision tier (config), enabled
```

Unknowns stay unknown (null) — never silently zero.

**Eligibility filters (in order):** server online → state fresh (age ≤
`stale_after_s`) → model loaded → image-capable → enabled in policy.

**Capacity class.** Two classes, decided before any fine-grained scoring:

- `IMMEDIATE` — eligible, and (parallelism unknown, or busy slots < parallelism, and nothing deferred);
- `QUEUED` — eligible but the execution capacity is fully occupied
  (the incident's exact signature: `parallel=1`, one long prefill in flight).

Any `IMMEDIATE` candidate outranks any `QUEUED` candidate. A `QUEUED`
candidate is only ever returned when no `IMMEDIATE` one exists, and the
response says so in `reason_codes`.

**Ordering within a class.** (1) configured tier (measured-vision-performance
order: A/35B=0, B/35B=5, C/35B=7, A/27B=20 by default), then (2) a small
deduction for GPU contention (sustained sidecar utilization/VRAM pressure
above threshold — relevant to node C's shared GPU), then (3) a deduction for
queue depth and recent prompt-processing load. GPU utilization is an
*ordering* signal only; it never decides eligibility.

**Response (contract):**

```json
{
  "ok": true,
  "policy_version": 1,
  "choice": {"name": "node-a", "url": "http://…:8081", "model": "…",
             "api": "openai-chat", "score": 0.94, "state_age_ms": 713},
  "reason": "preferred tier; loaded; image-capable; capacity available",
  "reason_codes": ["loaded", "image_capable", "preferred_tier", "capacity_available"],
  "alternates": [ …same shape… ],
  "rejected": [ …name + machine-readable reject code (debug surface)… ],
  "ts": 1788812345
}
```

No free-text in metric labels; reason codes are a closed vocabulary. The
contract is intentionally not llama.cpp-specific: `api` is a string
(`openai-chat` today; `vllm`/`sglang` later) and candidates are the normalized
structure above, so a future vision-capable vLLM or SGLang node plugs in
without a schema change.

**Freshness.** The hub polls every 2 s (`OFFLINE_AFTER=15` → offline after
~30 s of failures). The route endpoint does **not** trigger a synchronous
re-poll: that would inherit multi-second backend HTTP timeouts into an
advisor expected to answer in <50 ms, and would break the async monitoring
architecture. Instead, state age is carried in the response
(`state_age_ms`) and stale state is rejected/penalized by policy. Known
limitation: the shared polling loop runs servers sequentially with a 4 s
HTTP timeout per fetch, so a wedged backend can delay a cycle past the
nominal interval — recorded here, not fixed in Phase 1.

## 5. Workload profile: vision requests disable reasoning

The description workload (screenshot → text for a text-only model) does not
need chain-of-thought, but Qwen3-family templates think by default, and
measured, a thinking response consumed the entire token budget before any
visible content. The *consumer* therefore sends description requests with
reasoning disabled (e.g. `chat_template_kwargs: {enable_thinking: false}`,
the form current llama.cpp builds accept) plus an explicit output cap — the
existing pi-vision-handoff config already carries `thinking: false`. The
hub's routing contract does **not** expose any backend-specific reasoning
field; "reasoning: off" is a consumer workload preference each backend
integration translates. Because the 35B uses MTP speculative decoding,
alternating thinking/non-thinking requests against the exact pinned build
should be re-tested for state/correctness regressions (see upstream MTP
inter-request state issue [ggml-org/llama.cpp#26425](https://github.com/ggml-org/llama.cpp/issues/26425)).

## 6. What Phase 1 deliberately does not do

- **No backend mutation of any kind** — no load/unload (a `--models-max 1`
  load implicitly evicts the other model, so even "just load the preferred
  vision model" is a fleet mutation requiring a designed desired-state
  policy), no slot save/restore/erase, no preemption, no proxying.
- **No auto-healing.** The earlier draft's "advisor triggers a load of the
  evicted preferred model" is withdrawn: with one residency slot, restoring
  the preferred 35B by auto-loading it would implicitly unload whatever was
  intentionally resident. A separate, explicitly-requested project may later
  define desired-state reconciliation (per-model desired resident, operator
  suppression, cooldown, evict-conflict awareness). Until then: **alerting**
  ("no immediately usable vision route") plus one-click load in the existing
  hub UI solves the four-day-invisibility problem without silent mutation.
- **No `/slots` integration** (see §7 Phase 2).

## 7. Phase plan

**Phase 1 — read-only advisor (this commit).**
`hub/vision_routing.py` (new, importable — underscore name, since the
repo's dashed filenames are direct-execution scripts, not modules): pure
`route_vision(candidates, request, policy) → result` with no I/O.
`hub/llm-hub.py`: capture `architecture.input_modalities` and `--parallel`
from the existing `/v1/models` fetch (the raw entry was already downloaded
and discarded), build normalized candidates, add authenticated
`GET /api/vision/route`, add vision metrics to `/metrics`, read the new
`vision` config block. `hub/test_vision_routing.py`: stdlib `unittest`
covering the acceptance matrix below. `config.example.json` + `README.md`
updated.

Acceptance tests (unit level, no network):

1. All three 35B candidates loaded and idle → A/35B chosen, preferred-tier reason.
2. B offline (outside power window) → never chosen, not even as alternate.
3. A/35B evicted (27B resident) → A not mutated; best loaded B/C chosen;
   if none, honest degraded result (`ok: false` with reject codes), never a mutation.
4. Hub unavailable → consumer-side static fallback (documented consumer contract, not testable here).
5. C's sidecar shows sustained contention → C demoted, still usable as last resort.
6. A candidate with `parallel=1` and its slot occupied (the incident) → an idle alternate wins; if it is the *only* loaded vision model, it is returned with `queued` stated in the reason.
7. State older than `stale_after_s` → rejected/penalized, age exposed.
8. TOCTOU (state changes between advice and request) → consumer contract: try alternate, then static fallback. The advisor cannot eliminate the race; the contract bounds its cost.

**Phase 2 — better passive signals (preliminary, not started).**
Read-only `/slots` polling (slot state, in-use context, possibly direct
long-prefill evidence); explicit free-slot normalization; smoothing of GPU
contention; recent-latency/TTFT estimation from existing metrics. Decision
criterion: does it change any routing decision the current signals get
wrong? No slot *mutation* without a separate, explicitly approved design.

**Phase 3 — long-context cache/KV preservation (experimental, gated).**
The fleet's 50–76K-token sessions pay a full prefill every turn; a future
optimization is treating cached context as a *scheduling resource* (HOT:
resident in slot; WARM: RAM prompt cache; COLD: nothing). Relevant llama.cpp
mechanisms: `--cache-ram`, `--cache-idle-slots`, context checkpoints,
unified KV, slot save/restore (see
[`/slots` in the server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[KV-copy feature request #27616](https://github.com/ggml-org/llama.cpp/issues/27616),
[RAM cache correctness #27148](https://github.com/ggml-org/llama.cpp/issues/27148)).
**Gating:** upstream slot save/restore has open correctness problems on
hybrid/recurrent architectures — [#25913](https://github.com/ggml-org/llama.cpp/issues/25913)
(save reports success but yields zero prompt reuse on Qwen3.6-35B-A3B) and
[#28194](https://github.com/ggml-org/llama.cpp/issues/28194) (restore yields no
KV reuse on Qwen3.8-27B; checkpoints not persisted) — both open as of this
writing and both about exactly our models. Native `--cache-ram` behavior on
the pinned build must be benchmarked before any custom hub-managed swap.
Do not attempt to preempt an actively processing request.

**Later, optional.** Multi-backend vision (vLLM/SGLang candidates into the
same normalized structure — SGLang's
[cache-aware policy](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs)
and [router monitoring](https://github.com/sgl-project/sglang/blob/main/experimental/sgl-router/monitoring/README.md)
are the reference material for cache-affinity + load routing); desired-state
model reconciliation (§6); a thin proxy mode only if a permanent single-URL
consumer requirement ever materializes (it would reuse `route_vision()` and
stay a separate opt-in data-plane component).

## 8. Rejected alternatives

| Alternative | Why rejected (Phase 1) |
|---|---|
| Client-side smart picking (consumer polls each node itself) | Distributes fleet knowledge (power windows, GPU sharing, residency) into every harness; N consumers × N nodes of polling. |
| In-path LLM gateway (LiteLLM / vLLM router / SGLang gateway) | Larger dependency, credential, and data-plane surface than one JSON endpoint; makes the gateway an inference SPOF. |
| K8s Gateway API Inference Extension / llm-d | Right ideas, wrong scale; introduces a whole platform for a 3–4 node fleet. |
| Keep static list, just fix the ordering (35B → backup-35B) | Fixes *this* outage, not the class: no freshness, no capacity awareness, breaks again on the next eviction or power window. |
| Auto-heal (advisor triggers load of evicted preferred model) | Not non-destructive under `--models-max 1`; see §6. |

## 9. Risks

- **TOCTOU** between advice and request (mitigated by alternates + static fallback; bounded, not eliminated).
- **Stale state** after a wedged backend delays the polling cycle (mitigated by `state_age_ms` + staleness policy; recorded limitation in §4).
- **Modality metadata quirks:** older llama.cpp builds reported `input_modalities: ["text"]` for a model whose mmproj was actually loaded (see the 27B model card). The current build reports correctly for both node-A models (verified 2026-09-07), but the policy therefore accepts a per-model `vision` override in config so a metadata regression degrades to "known vision model", not "blind".
- **One consumer's long text session** can still monopolize a node's slots — routing avoids it, only fleet scheduling (Phases 2–3) fixes it.

## 10. Open questions

1. Does the current pinned build's `/slots` expose enough to distinguish "long prefill in flight" from "decoding" cheaply enough to be worth polling? (Phase 2 decides empirically.)
2. What is the right staleness default — 60 s is a guess; the polling loop's tail latency under a wedged backend will inform it.
3. When vLLM/SGLang vision nodes appear, is request-shape-based tiering (many/large images → biggest ctx) enough, or does cache-affinity routing become worth the complexity?
4. Advisory-scoped token vs. reusing the master token (master works; a read-only token is hardening, not a requirement).

## 11. References

Local (this repo): [`hub/README.md`](../hub/README.md), [`hub/llm-hub.py`](../hub/llm-hub.py), [`hub/config.example.json`](../hub/config.example.json), [models/](../models/) (dual-3060 router mode + the 27B card with the modality quirk), [docs/](../docs/) (hardware, architecture). Monitoring dashboards/alerts for the new `hub_vision_*` metrics live in the [monitoring-stack](https://github.com/githabideri/monitoring-stack) repo.

Upstream / prior art:
- llama.cpp — [`tools/server/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) (`/v1/models`, `/metrics`, `/slots`, `--cache-ram`, `--cache-idle-slots`, reasoning controls); issues [#25913](https://github.com/ggml-org/llama.cpp/issues/25913), [#28194](https://github.com/ggml-org/llama.cpp/issues/28194), [#27616](https://github.com/ggml-org/llama.cpp/issues/27616), [#27148](https://github.com/ggml-org/llama.cpp/issues/27148), [#26425](https://github.com/ggml-org/llama.cpp/issues/26425)
- Routing prior art: [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension), [llm-d](https://github.com/llm-d/llm-d) (scheduling docs: [overview](https://llm-d.ai/docs/architecture/core/router/epp/scheduling), [dev](https://llm-d.ai/docs/dev/architecture/core/router/epp/scheduling)), [vLLM production-stack load-aware routing](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/loadaware-routing.html), [SGLang model gateway](https://github.com/sgl-project/sglang) (cache-aware policy, router monitoring, [scheduler metrics](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_components/metrics_reporter.py))
- Consumers: [Pi](https://pi.dev/) + [pi-vision-handoff](https://github.com/monotykamary/pi-vision-handoff) ([pi.dev package page](https://pi.dev/packages/pi-vision-handoff)); [Hermes agent](https://github.com/NousResearch/hermes-agent) ([configuration docs](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/configuration.md), `auxiliary.vision`); [oh-my-pi](https://github.com/can1357/oh-my-pi) ([AI package docs](https://github.com/can1357/oh-my-pi/blob/main/packages/ai/README.md)); DSH community vision integrations [dsh-vision](https://github.com/zoahdev/dsh-vision), [dsh-vision-provider](https://github.com/libinyam/dsh-vision-provider) (a third, `dsh-open-eyes`, was unresolvable via the public API at writing time)
- Gateway risk: [LiteLLM incident #24518](https://github.com/BerriAI/litellm/issues/24518), [advisories](https://github.com/BerriAI/litellm/security/advisories), [PyPI post-mortem](https://blog.pypi.org/posts/2026-04-02-incident-report-litellm-telnyx-supply-chain-attack/), [Microsoft on AI gateway risk](https://www.microsoft.com/en-us/security/blog/2026/08/26/when-ai-infrastructure-becomes-target-securing-gateways-control-points/)

All external links verified reachable on 2026-09-07.

## 12. Phase 1 implementation plan (as implemented in this commit)

```
hub/
    vision_routing.py     NEW  pure policy module (route_vision, normalization,
                               reason codes, defaults) — importable, stdlib-only
    llm-hub.py            CHG  input_modalities + --parallel captured in
                               _poll_router; Server.api() exposes them;
                               new GET /api/vision/route (authenticated);
                               hub_vision_* metrics in /metrics; "vision"
                               config block (enabled, stale_after_s,
                               gpu_busy_threshold, per-model tier/override)
    test_vision_routing.py NEW  stdlib unittest, acceptance matrix §7
    config.example.json   CHG  vision policy block + per-model vision tier
    README.md             CHG  endpoint, contract, auth, policy, security note
```

Verification level of this report's claims: fleet numbers **measured live**
2026-09-07 against the running nodes (read-only probes); upstream issues
**verified open** via the GitHub API the same day; Phase 1 code **unit-tested**
(stdlib unittest, 8 acceptance cases) but **not yet deployed** to the running
hub — deployment is a separate, deliberate step (ansible role + config update
on the private side).
