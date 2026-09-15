# Benchmarks

How we benchmark. This page keeps the **durable method** — flags and comparison discipline. Specific numbers are point-in-time snapshots and live in [reports/](../reports/README.md). The design companion for agent-run campaigns (what to prove, in what order, which gates mean what, and the unattended-execution rules) is [agents/campaign-designer](../agents/campaign-designer/SKILL.md). Multi-night GPU campaigns now run on the [campaign platform](../benchmarks/README.md): one frozen bundle, a qualification matrix built from the 2026-09 failure history, and a documented-negative gate that turns a predicted negative result into a success path instead of a 2 a.m. review.

## Context sweep (`llama-bench`)

The standard sweep measures prefill and generation across growing context. Keep the flag set stable between runs so results are comparable (`-fa`, `-ctk/-ctv`, `-ts`, `-ncmoe`, `-b/-ub`).

Illustrative run (Qwen3-Coder-Next REAP-40B A3B Q2_K_XL, dual 3060, `-fa 1 -ctk q8_0 -ctv q4_0 -ncmoe 0 -ts 14/10 -b 128 -ub 64`):

| Prefill context | tok/s |
|---:|---:|
| 32k | 180.82 |
| 64k | 175.98 |
| 96k | 171.39 |
| 128k | 166.92 |

Generation (tg256) ~20.72 tok/s.

## Methodology (comparison discipline)

Comparison quality depends more on discipline than on raw throughput. Keep benchmark flags stable and use `llama-bench` for both prefill and generation. For chat-level comparisons, keep the prompt set fixed and compare the same three dimensions every time: total runtime, token volume, and effective tok/s. Do not mix profile classes or runtime families into one aggregate claim without explicit labels.

### The cache trap: every request needs a fresh nonce (2026-08)

Both serving stacks silently cache repeated prompts, which turns a "cold" benchmark into a warm one:

- **vLLM** has automatic prefix caching — a repeated fixture hits the KV cache and prefills at 10×+ the true cold rate (a "13K tok/s prefill" row in the 2026-08-30 campaign was a cache hit, not a regime).
- **llama.cpp** `--cache-prompt` does the same for repeated `/completion` prompts ("60× faster prefill" warm rows).

Rule: prepend a **unique nonce** to every prompt that must count as cold; repeat a fixture only when measuring warm behavior deliberately. Report cold and warm as separate rows — never average them.

### Decode rate under speculative decoding: usage tokens, not chunks (2026-08)

With MTP/DFlash, each streamed chunk carries multiple accepted tokens, so "tokens per chunk × chunks/sec" overstates the rate 3–5×. Compute decode t/s = **exact `completion_tokens` from the `/usage` chunk ÷ wall time from first token to end**. For llama.cpp, prefer the server's slot `print_timing` lines (it reports generation tokens directly); the server log's `draft acceptance` line is the MTP acceptance evidence.

### Proving GPU residency: PCIe counters, not timing (2026-08)

"The model fits in VRAM" is a bus claim — a streaming model decodes at a plausible steady rate too. Capture `nvidia-smi dmon -d 1 -s pumt` (columns include `rxpci`/`txpci` in MB/s; on some driver/LXC builds `-t`/`-T` per-line timestamps are absent, so write `REQ-START <epoch> <tag>` / `REQ-END …` markers to a side file and align rows by 1-second index). During decode: **~0–25 MB/s RX ⇒ resident; ≥1 GB/s ⇒ streaming from host RAM**. Always run the non-resident control (e.g. `--n-cpu-moe`) on the same card/model for the contrast. A weak PCIe link is usually *not* the LLM bottleneck — dual-GPU prefill on consumer cards is compute-bound (SM ~100%, PCIe well under link cap).

### GPU identity and topology discipline (2026-09)

**Never identify GPUs by CUDA ordinal alone.** CUDA's default device order (`FASTEST_FIRST`) can silently map a run to the wrong physical card — in the 2026-09 Qwen4Exp campaign it did: four non-3-GPU cells (one smoke, three topology) ran on the wrong hardware and were invalidated and re-run after postload VRAM exposed it. For every multi-GPU campaign:

- Record per device: **UUID, PCI bus ID, model name, CUDA ordinal, negotiated link width** (lspci under load, not idle).
- Run with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and address devices by bus ID end-to-end (manifest, runner, reports).
- Note that negotiated width can differ from the slot's maximum (x8 in an x16 slot); that's a topology fact, not a fault.

### Result-record schema (2026-09)

Compact vocabulary for campaign reports, so cross-campaign comparison stays sane (documented convention, not code):

```
Campaign provenance
- model + quant + SHA256
- backend + exact commit/build
- CUDA/driver
- GPU UUID + PCI bus ID + active link width (per device)

Placement
- fitter or manual
- split mode / split ratios / ngl
- per-device model/KV/compute allocation
- free VRAM after load + minimum during request

Performance
- prompt tokens / completion tokens
- pp/s · tg/s · TTFT · total latency

Data movement
- PCIe RX/TX MB/token · SSD KB/token (where relevant) · host RSS · swap activity

Efficiency
- GPU-only Wh/1K tokens · wall energy when a meter exists

Cache state
- cold / warm / prefix-cache hit · whether OS page cache was controlled
```

### First requests are not steady state (2026-08)

After server-up, send a tiny warmup request before the measured window (cudagraph capture, Triton JIT, FlashInfer plan build). A first long request can show a one-off JIT dip (seen on the 3090 vLLM node: 17.8 t/s first, 73 steady) — never quote it as the regime number. vLLM's `/health` returns **HTTP 200 with an empty body** (llama.cpp returns `{"status":"ok"}`) — match the status code, not the body.

> Results are hardware-specific — comparative guidance, not universal constants.

### Workload integrity: evidence that the work happened, not how fast it did (2026-09-15)

A wall-time floor is a **proxy** for proof that the declared workload happened, and a proxy only works in the layer where it actually holds. The LMCache campaign produced two counterexamples in one dataset: a **return leg is fast by construction** (a cached 40 K prefix in 1.5 s — that speed *is* the signal), so any cell-level or even per-kind time floor kills the measurement the campaign exists to make; and a "**60 K filler**" row at 1.7 s (~35 K tok/s, physically impossible) was a request that was **never sent** — which the wall floor did *not* catch, because the same cell also contained rows that passed.

The invariant was never "60 K takes more than 30 s". It was "**the 60 K request really contains ~60 K tokens and reached the server.**" Gates encode that with a three-class architecture:

1. **Integrity (hard → `INVALID`)** — the per-request **work contract**: `declared_prompt_tokens ≈ client_encoded_tokens ≈ server_observed_prompt_tokens` (tolerance `max(32, 1%)` to absorb BOS/chat-template drift), plus the request seen in server-side evidence, the expected decode count, and persisted artifacts. This is the only class that answers "did the declared work happen", and it is **hardware-speed-independent** — a 40 K store finishing in 2 s on future hardware is *valid*, not implausible.
2. **Mechanism (experiment-specific → `EXPECTED_NEGATIVE` / `NOT_ESTABLISHED` / `INVALID` per the spec)** — the mechanism under study must be shown to have *participated*: pool allocated, lookups occurred, retrieved chunks, APC-hit tokens, eviction counts.
3. **Plausibility (soft → `REVIEW_REQUIRED`, never auto-INVALID)** — absurd wall times, throughput several orders beyond a reference population, zero GPU utilization. Anomaly *flags*, not verdicts — and never applied to return legs, where an extreme outlier may be the phenomenon itself.

Every critical quantity carries **provenance** (`{value, source}`: client-tokenizer, vllm-usage, server-log, client-monotonic) so the runner evaluates facts rather than treating one client JSON as the oracle. A request's **kind** (`first_touch`, `cached_return`, `pressure`, …) selects *which evidence is required* — not which time constants apply. Brand-new hardware with no baseline is a non-problem: the engine's own token accounting validates the workload, and the wall time is free to be whatever physics produced.

## Where the snapshots live

| Snapshot | Report |
|----------|--------|
| LMCache 0.5.0 RAM tier on the dual-3090 27B (hard negative: HMA) | [2026-09-15](../reports/2026-09-15-lmcache-ram-tier-vllm-dual3090.md) |
| gpt-oss-20b on single 3060 | [2026-02-03](../reports/2026-02-03-gpt-oss-20b.md) |
| Nemotron profile (fixed 5-task set) | [2026-02-12 ABC](../reports/2026-02-12-nemotron-abc-executive-summary.md) |
| ik_llama.cpp vs llama.cpp | [2026-02-12](../reports/2026-02-12-ik-llama-cpp-vs-main-preliminary.md) |
| BeeLlama DFlash (Qwen3.6-27B) | [2026-06-19 cutover](../reports/2026-06-19-beellama-dflash-cutover.md) |
| Qwen3.6-35B-A3B dual 3060 (full config matrix, ~45 runs) | [2026-08-27](../reports/2026-08-27-qwen3.6-35b-a3b-dual-3060-optimization.md) |
| llama.cpp `-ub` tuning (single 3060, MoE+MTP) | [2026-08-28](../reports/2026-08-28-llama-cpp-ubatch-moe-single-gpu.md) |
| 35B on one 12 GB 3060 (residency proof) + 27B on the 3060 pair (ctx/concurrency/wall power) | [2026-08-30](../reports/2026-08-30-dual-3060-35b-squeeze-27b-node.md) |
