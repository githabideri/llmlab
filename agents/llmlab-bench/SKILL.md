---
name: llmlab-bench
description: >-
  Plan and run LLM inference benchmark campaigns against llama.cpp/vLLM endpoints on
  consumer-GPU boxes — don't-touch-production rule, pre-boot VRAM budgeting, vLLM/llama.cpp
  version & flag-drift tables, GDN/hybrid state accounting, MTP speculative decoding,
  nonce cold/warm discipline, exact-token decode rates, PCIe residency proof, wall-power
  integration, per-run artifact set, median-of-N data discipline with row sanity filters,
  artifact provenance. Use when benchmarking models or planning multi-stage measurement
  campaigns.
---

# llmlab-bench — Benchmarking Inference on Consumer GPUs

The operational companion to [docs/benchmarks.md](../../docs/benchmarks.md) (durable measurement
method). This skill is the campaign view: what to verify before boot, what to record while
running, and what disqualifies a number. The learnings here cost real incidents and real GPU
hours — follow them.

## Golden rule: don't touch production

- **Never benchmark-stop an endpoint that currently provides the campaign controller's own
  model.** Before entering an exclusive GPU maintenance phase, move the controlling
  agent/session to an independent endpoint or a cloud model, and verify that path with a live
  request. A controller whose provider is the endpoint under test dies at `maint.sh enter` —
  and with it the cheapest recovery path (the 2026-09-08 incident pattern).
- **Every destructive/exclusive phase needs an out-of-band restore path whose execution does
  not depend on the agent session surviving** — an independent deadline watchdog (owned-PID
  kills only, prod-unit start, `/health` + live-completion proof) that a normal campaign
  completion simply disarms. The watchdog is the difference between "wake up with the endpoint
  alive" and "hope the orchestrator's final `finally:` ran".

- Benchmark a **throwaway server instance on a spare port** (e.g. 8093) started from the same
  build/model, **not** the production service.
- If the measured config cannot run side-by-side (VRAM contention), measure **inside a verified
  maintenance window**: stop production, run, restore, and confirm `{"status":"ok"}` **plus a live
  completion** before declaring done.
- **Flag parity**: read the *live* unit (`systemctl cat <service>`) and match its buffers,
  offload, MTP, and KV-quant settings. (In the Aug 2026 fork campaign, "production" was
  benchmarked with the wrong mmproj offload for three parts, and only `systemctl cat` caught it.)
- The throwaway instance needs no API key (local client) and no metrics, but does need
  `--cache-prompt` etc. to mirror the unit.
- **Never send a memory-exhaustion test (near-ceiling context, huge KV) at an endpoint anything
  else depends on.** It is a denial of service, not a "load test": a 92K-prompt test against a
  24GB production card OOM'd the server and took its dependent clients down with it. Exhaustion
  tests run on a dedicated instance only — or in a proper maintenance window.

## Endpoint mechanics

- **Health check returns JSON**: `curl http://<host>:<port>/health` → `{"status":"ok"}`. Match
  the **whole string**, never `1`/`0` like an HTTP code. vLLM's `/health` returns **HTTP 200 with
  an empty body** — match the status code, not the body. The two engines differ.
- **Long prompts**: use `POST /completion` (`prompt` field) from a Python client. Never
  `llama-cli -f bigprompt.txt` — it floods the TUI, and E2IG does **not** insert the prompt.
- **Verify prompt tokenization** before comparing: `llama-tokenize -m <gguf> < prompt.txt |
  tr ' ' '\n' | grep -c .` (a byte-cutoff prompt ≠ a token-cutoff one; counts are only comparable
  if both cut at token boundaries).
- **First requests are not steady state.** Send a tiny warmup request after server-up and before
  the measured window (cudagraph capture, Triton JIT, FlashInfer plan build). A first *long*
  request can also show a one-off JIT dip or a single transient crash in the FlashInfer drafter
  path — never quote it as the regime number.
- **Speculative decode (MTP/DFlash) invalidates chunk-count t/s** — each streamed chunk carries
  multiple tokens. Compute decode t/s = exact `completion_tokens` (from `/usage`) ÷ decode wall
  (first token → done); for llama.cpp prefer the slot `print_timing` lines on a fresh server
  (no `--cache-prompt`).
- **MTP acceptance lives in the server log** (`draft acceptance: n/N, x%`), not the client —
  capture the log per cell.

## Cold vs warm is a first-class axis

Repeated fixtures silently hit vLLM's prefix cache and llama.cpp `--cache-prompt`, faking
prefill (the Aug 2026 dual-3060 campaign's "llama beats 3090" and "3090 at 13K prefill" were both
cache artifacts). Prepend a unique nonce per request for cold; repeat only deliberately for warm.

## Pre-boot VRAM budgeting

Before booting, compute the budget: weights (per rank if TP) + draft/MTP weights + graph
overhead + KV pool at the target context. If it doesn't fit, **don't boot — a boot-OOM is a data
point**, and it defines the context ceiling of that config family. Measured on 12GB cards with a
W4 27B-class hybrid: 64K never boots (vLLM 0.27.x *and* 0.28.x); 0.28.x piecewise cudagraphs + MTP
don't fit at 17K *or even 8K* context (eager-only is the workable regime). Log the exact shortfall
(`CUDA out of memory. Tried to allocate … GiB`) — it is the OOM-margin evidence the report needs.

## Version & flag drift — re-verify per build

Engine flag sets rot. Before composing any campaign, pin the build (commit hash / pip version),
check the current `--help`, and record the mapping in the campaign notes.

**llama.cpp mainline (2026-09, build 925e1179):** `--kv-quant-type` → `--cache-type-k/v`;
`--sm` → `--split-mode`; `--experimental` and `--cont` are gone; MTP is
`--spec-type draft-mtp` (the head ships in the `-mtp` GGUF). Long context needs
`--cache-type-k/v q4_0`: with it, a W4 27B fits the full 262K window on a 24GB card at ~22.2GB.
The community's living tuning record for Qwen3.8-27B MTP is
[sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) — 53 configs across a decade of
silicon; start from `--spec-draft-n-max 2` (measured +33–145% decode depending on card).

**vLLM 0.27.x vs 0.28.x** (observed on 27B-class hybrid, 2×12GB TP2 and 1×24GB):

| | 0.27.1 | 0.28.0 (HEAD) |
|---|---|---|
| cudagraphs + MTP, 12GB cards | profiling OOMs — `--enforce-eager` mandatory | PIECEWISE works on 24GB-class; on 12GB doesn't fit at 17K *or* 8K ctx → eager-only |
| MTP acceptance (same model/config) | ~4.6–4.8 | ~3.7–3.8 (lower, but graphs available on big cards) |
| known bugs | loader nondeterminism on W4-AutoRound-class checkpoints (`embed_tokens.weight_packed` ValueError — verify file integrity before blaming the loader; retries eventually pass); **generation garble, fixed by PR #40914** — [reported here](https://github.com/MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090): stock 0.27.1 fails 13/15 of a garble battery, 0/15 with the patch | — |
| ops notes | `python -m vllm` may lack `__main__` in some builds — use the venv's `bin/vllm`; the flashinfer **python and cubin packages must be the same version pair** or the verify path crashes at certain draft depths | `VLLM_USE_FLASHINFER_SAMPLER=0` on consumer setups |

Before trusting any 0.27.x number, verify whether #40914 is in the venv (git log / source of the
pip install) and run a small loop/garble battery first.

## Hybrid (GDN) state accounting

For hybrid GDN/Mamba + attention models, **do not estimate capacity from attention KV alone**:
recurrent state and hybrid-page allocation add engine-specific overhead. Use the engine's
startup profiler as authoritative. On the current Qwen3.8-27B vLLM 0.28.0 TP2 deployment,
13.24 GiB/rank yields 776,928 logical tokens — ≈35.7 KiB/logical token aggregate
(≈17.9 KiB/rank) versus the 32 KiB architectural fp8 attention-KV floor (~12% overhead).
That is an **effective cache-allocation measurement, not a standalone GDN-state measurement**
(and "per token/slot" phrasing is misleading for it). The older "305 B/token/slot (~0.3 GB per
1K)" figure was a 1000× unit slip and could not be traced to a real measurement — removed.
Size pools from the pool the engine *reports* at startup, not from a KV-only formula.

## Co-resident CPU accounting

On a shared box, the instance under test shares cores with other vLLM/llama processes. Measured:
decode rates depressed 2–3× when a production 3090 server co-resided on the same cores as the
test instance. Record a co-resident process inventory (ps/pidstat) per cell; compare numbers only
against runs with matched co-residency. Idle-host baselines are not comparable to contended
campaign numbers — say which one each number is.

## Data discipline

- **A result exists only when it is a row in the campaign's CSV/JSON.** Numbers read off a live
  terminal or quoted in a chat are not data; the runner appends rows, or the cell doesn't count.
- **Row sanity filter:** flag 0-token completions and sub-second decodes. They are infrastructure
  artifacts (e.g. a health server answering a benchmark port), not physics. Flag, don't silently
  drop — a filtered row is evidence something was wrong.
- **Median of N, never best-of-N.** MLPerf's rule is the model: N independent runs, score the
  median; "running multiple iterations of N runs to try find the lowest one is against the
  spirit" ([training rules](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc)).
  Keep **all** attempts, including failures, in an attempt log — inconsistencies in the log are
  how bad runs get caught.
- **One timebase.** Verify each host's timezone at campaign start — boxes in one lab disagree
  (UTC vs CEST observed in a single rack). All artifact timestamps in **UTC**; record each host's
  offset in the campaign notes.

## Wall power & energy: measure at the plug, integrate state-holds

Per-GPU dmon power misses the ~90 W box base (CPU + RAM + NVMe + idle GPUs), so
"2×3060 vs 3090" comparisons only make sense at the wall (this lab's plug: both draw ~340 W —
a per-GPU-only analysis would have missed that). Use any watts timeseries you have (smart plug,
PDU, HA sensor) — and integrate state-holds, because history systems **coalesce unchanged states
into one point**: a naive fixed-dt integrator silently drops flat stretches and undercounts by
~25%. Correct model: each reading holds until the next change → `E = Σ watts_i × (t_{i+1} − t_i)`,
clipped to the phase window. Validate against idle hours (baseline W → expected Wh/h), then
report **Wh per phase + "above idle"**. A full night of campaign reproduction on this box costs
~0.26 kWh — measuring is nearly free next to serving.

## Concurrency: report aggregate AND per-request

One number is meaningless — under chunked prefill, prompt time stacks across concurrent requests
and per-request decode can collapse while the GPU sits at 99% SM. Measure a small grid
(c1/2/4/8 × 2K and 16K prompts, threaded client, nonce per request) and report: per-request
t/s per stream, **aggregate** output t/s, stacked TTFTs, and how MTP acceptance moves (dropped
46–61% → 17–39% at 4×16K in the 2026-08-30 campaign). Compute the KV ceiling from vLLM's startup
log (`Available KV cache memory … total tokens`) before running. Distinguish *batchable
short-prompt* workloads (aggregate scales, per-request falls) from *long-context* ones (they
serialize — the cliff on the dual-3060 node was 4×16K).

## Proving residency (PCIe, not just timing)

"Fully GPU-resident" is a bus claim: verify with PCIe counters, because decode timings look
plausible either way (a streaming model still decodes at a steady rate). On Aug 2026 runs this
separated a 43–82 t/s resident MoE (~11–24 MB/s PCIe RX in decode) from its CPU-MoE twin
(6.7 GB/s RX, 35 t/s) on the same card.

- Capture: `nvidia-smi dmon -d 1 -s pumt` — `rxpci`/`txpci` in MB/s. Some driver/LXC builds
  reject `-t`/`-T`; write `REQ-START <epoch> <tag>` / `REQ-END` markers to a side file and align
  rows by index (1 s cadence).
- During decode (SM high, mem high): **~0–25 MB/s RX ⇒ resident**; **≥1 GB/s ⇒
  weight/expert streaming from host** (MoE `--n-cpu-moe` always shows this).
- Always run the non-resident control on the same model/card.
- Weak PCIe links (x4/x8) were **not** the LLM bottleneck in this lab's measured dual-GPU runs
  (compute-bound: SM ~100%, PCIe well under link cap). Treat this as a measured observation to
  re-verify — keep dmon residency evidence in every run — rather than doctrine, especially after
  a topology change.

## Artifact provenance

"Which file" matters as much as "which model". Record the exact GGUF/safetensors filename + byte
size + SHA256 **of what the server log says it loaded**, not an estimate. Same-named files differ
between builds (an MTP variant of the same model was 1.06 GiB larger). And official artifacts are
not automatically the best: the official Qwen3.8-27B NVFP4 was **5.5× slower TTFT** (276 ms vs
50 ms) than an older community build — same engine, same 5090, same settings — and returned more
invalid structured responses
([neroued/ninfer#38](https://github.com/Neroued/ninfer/issues/38)).

## Engine gotchas

- **Ollama + agent-style clients** on Qwen3.8-27B can fail with
  `500 system message must be at the beginning` — an API-layer system-message placement issue,
  model-specific ([ollama#17754](https://github.com/ollama/ollama/issues/17754)).
- **llama.cpp `--split-mode tensor` has no memory fit check** in mainline: an oversized pool
  crashes `load_model` instead of refusing. Verify fit *before* TP runs (Pre-boot budgeting
  above) — a 32K pool on a ~11GB/card 27B split dies in the loader, not at startup.
- **vLLM first long request** can crash once in the FlashInfer drafter path (transient) and/or
  show a one-off JIT dip — the warmup request above avoided both on this lab's build (evidence
  for this build, not a universal guarantee).

## Scheduling / gating

Gate timed runs by **host** time (verify its timezone first), against your box's known noise
windows (nightly backups, large transfers) — detect the windows (systemd timers/cron on the
host), poll, and wait. Poll with **short sleeps (≤10 s)** rather than long blind `sleep N`.

## Per-run artifact set

Tag everything consistently (`<campaign>-<tag>`), and capture in parallel:

1. Client: full response (tokens, timings, generated text) → JSON + pretty log
2. Server: journal/log → **MTP acceptance** lives here, not in the client
3. `dmon -s puvmt` (2 s) — CPU%, GPU util, VRAM **plus `t` (PCIe RX/TX) for residency evidence**
4. `pidstat -t 1` — per-core + per-thread (and a co-resident process inventory)
5. VRAM sampler: `nvidia-smi --query-gpu=memory.used --format=csv,noheader` @ 4 s (peak =
   OOM-margin evidence)
6. Wall-clock markers per request (`REQ-START/END <epoch> <tag>`) so dmon rows align to decode
   vs prefill windows
7. Wall power: the box's timeseries source for the whole window — idle baseline + per-phase
   averages
8. Model provenance: filename + byte size + SHA256 of the exact artifact loaded (from the server
   log), plus the engine build hash

## Determinism discipline

- `temp 0` (+ `seed 1234` for /completion) for content comparison.
- Compare **content**, not token-ID arrays (`token_data` came back empty in current builds).
- **Expected divergence across compute paths** (different offload/split/quant layouts) is FP
  re-association, not a bug — document where/when it appears; don't hide it or chase it.

## Reporting

- Public, sanitized results → dated report in this repo (`reports/` conventions: no frontmatter,
  sanitizer sweep, both remotes).
- Internal details (identifiers, exact units, incident forensics) → your private repo.
- State the verification level honestly: *implemented / locally verified / live verified /
  remotely verified* — never collapse to "works".

## Speculative-decoding correctness gate (mandatory)

A performance row for a speculative configuration (MTP / ngram / DFlash / any drafter) is
**invalid until the generated output passes a content-sanity check**: coherent on-topic text at
both short and long context, no multilingual garbling, no runaway repetition, clean
stop/formatting. High token rate alone is not evidence of a valid result — community
Flash-Next work has produced impressive counters while emitting corrupted text beyond short
context on certain MTP implementations. Run the gate (a fixed prompt battery: short prose,
long-prompt continuation, structured output) at short and deep context for every new
speculative path before accepting its throughput numbers.

## Patch-stack provenance

For fast-moving experimental engines, "the commit" is not the provenance: record the **full
stack** — upstream base SHA + every applied PR SHA + local patches + build flags + CUDA/GGML
version + which MTP / top-k / fitter implementation the binary contains. Two binaries of the
same "version family" can behave radically differently (community Flash-Next stacks are
literally master + expert-cache PR + mmap fix + MTP PR + a local safety gate). Put the stack
next to the model artifact provenance above — it is part of the result, not a footnote.
