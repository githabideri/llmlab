# Nemotron-3-Nano-30B-A3B

NVIDIA's Nemotron-3-Nano, a 30B parameter MoE model with ~3B active parameters.

## Quick Facts

| Param | Value |
|-------|-------|
| Parameters | 30B total, ~3B active (MoE) |
| Architecture | Mixture of Experts |
| Quant tested | IQ4_NL (18GB on disk) |
| Context window | 131,072 tokens |
| VRAM requirement | **2x 12GB GPUs** (requires tensor split) |

## Performance

Tested on 2x RTX 3060 (12GB each) with `--tensor-split 1,1`:

| Metric | Value |
|--------|-------|
| Prompt eval (32k ctx) | 670 tok/s |
| Generation (fresh) | 66.7 tok/s |
| Generation (filled ctx) | ~58 tok/s |

## Context Window

| Target | Result |
|--------|--------|
| 131k (default) | ✅ Stable |
| 196k | ✅ Stable (tested with 4 parallel slots) |
| 256k | ❌ OOM on 2×12GB setup |

Sweet spot on 2×RTX 3060: **196k context** with 4 slots.

## Known Issues

### 🚨 "Lost in Thought" Failure Mode

**Severity:** Medium — causes silent message drops

**Symptom:** 
- Model generates thinking/reasoning tokens
- Emits stop token without producing visible output
- Server returns HTTP 200 (no error)
- User sees nothing — message never delivered

**Trigger:** 
- Action-type prompts requiring multi-step planning
- Examples: "do steps 1-4", "install X and configure Y"

**Observed rate:** ~1.4% of turns (2/148 in testing)

**Technical details:**
```
stopReason: "stop"
content: [
  { type: "thinking", thinking: "...1000+ tokens of planning..." }
  // NO text output
  // NO tool calls
]
```

**Mitigations:**

1. ✅ **`--reasoning-format deepseek`** (recommended) — Extracts thinking into separate `reasoning_content` field, keeping visible output in `content`. Deployed and monitoring.
2. `--reasoning-format none` — Disables thinking entirely (loses reasoning capability)
3. Increase `max_tokens` to ensure output budget after thinking
4. Prompt engineering: prefix with "Respond with your first action:"
5. Gateway-side: detect thinking-only responses and retry

```bash
# Recommended server flags for reasoning
llama-server \
  --reasoning-format deepseek \
  --reasoning-budget -1 \
  # ... other flags
```

### Single-GPU Trap

**Symptom:** Slow generation, high CPU usage, GPU underutilized

**Cause:** 18GB model doesn't fit in single 12GB GPU, spills to RAM

**Fix:** Must use `--tensor-split 1,1` (or similar) to split across GPUs

## Recommended Config

```bash
llama-server \
  --model /path/to/Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf \
  --ctx-size 131072 \
  --n-gpu-layers -1 \
  --tensor-split 1,1 \
  --flash-attn 1 \
  -b 128 -ub 64 \
  --jinja \
  --host 0.0.0.0 --port 8080
```

**Required flags:**
- `--tensor-split 1,1` — split across 2 GPUs (adjust ratio for unequal VRAM)
- `--flash-attn 1` — recommended for long context

## Benchmark Results

Tested with llmlab OpenClaw benchmark (L0-L4 + DOC-QA):

| Case | Result |
|------|--------|
| L0: Read/write | ✅ Pass |
| L1: Config summary | ✅ Pass |
| L2: Config patch + JSON | ✅ Pass |
| L3: Bench parse | ✅ Pass |
| L4: Tool chain | ✅ Pass |
| DOC-QA: Citations | ✅ Pass |

Model handles structured tasks well when it doesn't hit the "lost in thought" failure.

## Agentic Capabilities

Tested as an autonomous agent (multi-turn sessions with tool access):

| Capability | Result |
|------------|--------|
| Tool calling (exec, web) | ✅ Reliable |
| Multi-step planning | ✅ Works (watch for lost-in-thought) |
| API integration (weather, etc.) | ✅ Correct data extraction |
| Following prompt updates | ✅ Picks up instruction changes mid-session |
| Voice→action pipeline | ✅ Transcribe + act in one turn |

Model handles agentic workloads well when `--reasoning-format deepseek` is applied.

### Behavioral Notes (Feb 10)

- **Tool-use is real, not performative:** it executed shell commands (venv + pip) and surfaced missing system deps instead of claiming success.
- **Environment constraints are handled honestly:** it reported permission errors when `apt-get` failed rather than fabricating completion.
- **Web-search hygiene gap:** it sometimes answers from prior knowledge unless explicitly prompted to search/cite.

### Wild-ride Session Addendum (Feb 12, reduced-thinking era)

Chat-orchestration-heavy postmortem sample (not a quality benchmark):

- Tool calls: **33 total**
- Successes: **31**
- Failures: **2**
  - `message`: channel-resolution error (`Unknown channel`)
  - `process`: monitored process ended with `SIGKILL`

Operational interpretation: tool reliability remained high overall, and failures were environment/control-plane class rather than model-internal reasoning failures.

### Non-thinking Mode Results (Feb 12)

> Scope note: this section reflects a targeted no-thinking speed snapshot.
> It is not directly comparable to later reduced-thinking production-profile runs
> (`--reasoning-budget -1` + constrained brief-reasoning prompt style).

Dedicated no-thinking validation run after clean reset (`/lbn llmlab`) on GPU Nemotron.

**Runtime config in test:**
- Model: `llama-cpp/Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf`
- OpenClaw session thinking: `off`
- llama-server: `--reasoning-format deepseek --reasoning-budget 0`

**Observed speed (chat-level):**
- 20 prompt→answer pairs
- Average: **~8.15s**
- Median: **~6.75s**
- P90: **~12.8s**

**Tool-call reliability in this run window:**

| Tool | Calls | Outcome |
|------|------:|---------|
| `web_search` | 2 | ✅ success |
| `web_fetch` | 2 | ✅ success |
| `browser` | 1 | ⚠️ failed (browser relay/tab unavailable), fallback path used |
| **Total** | **5** | **4 successful, 1 understandable environment failure** |

**Quality findings:**
- **Reasoning/planner leakage occurred** in visible answers (internal planning text surfaced in output on multiple turns).
- **Hallucination risk increased** for factual prompts when not forced through tool-grounded retrieval.
- **Link hygiene degraded** in some responses (invalid/404 URLs were produced in one segment).

**Interpretation:**
- `think=off` gives excellent latency, but quality/reliability can drop for open-ended factual tasks.
- Best used for lightweight chat or tightly tool-grounded workflows; less ideal for freeform factual synthesis without guardrails.

### Reasoning Gradient Tuning (A/B/C, Feb 12)

Follow-up tuning compared three profiles on the same 5-prompt set:
- **A**: baseline reasoning on
- **B**: constrained brief reasoning prompt ("max 1–2 short sentences")
- **C**: prompt-level thinking-off directive

#### With `--reasoning-budget -1`
| Profile | Weighted reasoning:text | Avg latency | Quality proxy |
|---|---:|---:|---:|
| A | 1.47:1 | 3.36s | 3/5 |
| B | **1.42:1** | **2.31s** | **5/5** |
| C | 1.15:1 | 3.11s | 5/5 |

#### With `--reasoning-budget 0`
- Fastest, but not perfectly clean: occasional malformed `</think>`/planner-style output still leaked into visible text.
- Practical takeaway: budget=0 is speed mode, not quality mode.

#### `chat_template_kwargs.reasoning_effort` probe
- `reasoning_effort=low` reduced reasoning on a reasoning-heavy prompt, but behavior was inconsistent across prompt types.
- Treat as experimental/secondary control until verified stable for Nemotron templates.

**Current best "less-not-none" strategy:**
1. keep `--reasoning-budget -1`
2. apply constrained brief-thinking prompt profile (B)
3. add task routing (`off` for trivial/speed-critical; constrained-on for medium/hard)

### Reduced-thinking Operating Profile (Adopted, Feb 12 evening)

After A/B/C tuning, operational profile was switched to reduced-thinking mode for better speed/quality balance:

- `--reasoning-format deepseek`
- `--reasoning-budget -1`
- brief constrained reasoning prompt style
- OpenClaw-side reasoning remains enabled (`reasoning: true`), with prompt-level constraining

**How to compare runs correctly:**
- compare reduced-thinking runs only against other reduced-thinking runs
- do not merge reduced-thinking outcomes into no-thinking baseline tables
- tag runs by profile class in notes/reports:
  - `non-thinking-speed` (e.g. budget `0` / think-off style)
  - `reduced-thinking-balanced` (budget `-1` + constrained brief reasoning)

## Hardware Tested

- 2x NVIDIA RTX 3060 12GB
- Intel i5-7400
- 24GB system RAM
- llama.cpp (latest as of 2026-02-09)

## Changelog

- **2026-02-12:** Added run-class scope note for no-thinking section, added reduced-thinking operating profile guidance, and added wild-ride postmortem reliability addendum (33 calls, 31 success, 2 env/control failures).
- **2026-02-12:** Added A/B/C reasoning-gradient results (timing/token split), adopted brief-constrained B profile as recommended default (`--reasoning-budget -1`, constrained prompt style).
- **2026-02-12:** Added dedicated non-thinking run results section (speed profile, tool-call success/failure breakdown, leakage + hallucination caveats).
- **2026-02-10:** Added `--reasoning-format deepseek` mitigation (deployed). Updated context window findings (196k stable, 256k OOMs). Confirmed agentic task completion + behavioral notes.
- **2026-02-09:** Initial profile. Discovered "lost in thought" failure mode. Documented tensor-split requirement.
