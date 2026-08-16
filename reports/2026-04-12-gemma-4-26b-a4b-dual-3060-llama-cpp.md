# 2026-04-12 — Gemma 4 26B-A4B on llama.cpp (dual RTX 3060, text-only)

## Goal
Validate whether Gemma 4 26B-A4B can serve as a practical text-only local model on a 24 GB class setup (2× RTX 3060 12 GB) using mainline-ish `llama.cpp`, and identify the serving profile that actually works under real prompt load.

---

## Setup

### Hardware
- 2× RTX 3060 12 GB (24 GB total VRAM)
- Consumer PCIe multi-GPU setup without NVLink

### Software
- `llama.cpp` Gemma-capable build
- Binary: `llama-server`
- Model: `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`
- API mode: OpenAI-compatible `/v1/chat/completions`

### Common launch profile
Unless otherwise stated, tests used:

```bash
llama-server \
  -m gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8081 \
  --device CUDA0,CUDA1 \
  --split-mode layer \
  --gpu-layers all \
  --parallel 1 \
  --flash-attn on \
  --batch-size 512 \
  --ubatch-size 256 \
  --jinja \
  --reasoning off \
  --reasoning-budget 0 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --ctx-checkpoints 0 \
  --cache-ram 0 \
  --metrics --slots --perf
```

All comparative runs used the same ~4.5K-token prompt and a structured response task unless explicitly marked as a tiny smoke test.

---

## Findings

### 1. Initial failure was a build/architecture mismatch, not VRAM fit
Early load attempts failed with:

```text
unknown model architecture: 'gemma4'
```

The GGUF file itself loaded far enough for metadata inspection, so the blocker was missing Gemma 4 support in the earlier build, not a corrupt model or obvious fit failure.

### 2. Thinking mode had to be disabled explicitly
With a Gemma-capable build, the server could answer requests, but default behavior produced pathological chat responses:
- output consumed by `reasoning_content`
- empty normal `content`
- effectively useless chat completions despite HTTP 200 responses

The stable text-only fix was:

```bash
--reasoning off \
--reasoning-budget 0 \
--chat-template-kwargs '{"enable_thinking":false}'
```

### 3. The original 8K baseline looked broken because CPU thread settings were wrong
At:
- `ctx-size=8192`
- tiny prompt (`39` tokens)
- tiny completion (`26` tokens)
- `threads > 1`

observed generation throughput was only about **2 tok/s**, which looked pathological.

Forcing:

```bash
--threads 1 --threads-batch 1
```

immediately changed the picture.

#### 8K tiny smoke test, no-thinking, `threads=1`, `threads-batch=1`
- Prompt: `39` tokens
- Completion: `26` tokens
- **PP:** `200.75 tok/s`
- **TG:** `35.10 tok/s`

This established that the decode path itself was not fundamentally broken.

### 4. Long-prompt performance separated into two different problems
Using a real prompt (~4517 tokens) showed that:
- the severe slowdown was **not** mostly from reserving 32K context
- **prefill** and **decode** had different best knobs

#### Control: 8K server, same ~4517-token prompt, `q8_0/q4_0`, `threads=1`, `threads-batch=1`
- **PP:** `12.06 tok/s`
- **TG:** `5.09 tok/s`

#### 32K server, same prompt, `q8_0/q4_0`, `threads=1`, `threads-batch=1`
- **PP:** `12.46 tok/s`
- **TG:** `5.16 tok/s`

Interpretation: the collapse was mostly **real prompt/live-context cost**, not merely 32K reservation overhead.

### 5. `threads-batch` is the main prefill lever on this box
Keeping `threads=1` for decode and increasing only `threads-batch` materially improved prompt processing.

#### 32K, same ~4517-token prompt, `q8_0/q4_0`
| Threads | Threads-batch | PP tok/s | TG tok/s |
|--------:|--------------:|---------:|---------:|
| 1 | 1 | 12.46 | 5.16 |
| 1 | 4 | 30.48 | 4.97 |
| 1 | 6 | 32.07 | 5.05 |
| 1 | 8 | 32.29 | 5.01 |

Practical plateau: **`threads-batch=4..6`**. Higher values provided little additional gain.

### 6. With `q8_0/q4_0`, 32K and 64K were usable but unimpressive under real load
#### 32K, `q8_0/q4_0`, `threads=1`, `threads-batch=6`
- Prompt: `4527` tokens
- Completion: `142` tokens
- **PP:** `33.71 tok/s`
- **TG:** `4.81 tok/s`

#### 64K, `q8_0/q4_0`, `threads=1`, `threads-batch=6`
- Prompt: `4527` tokens
- Completion: `155` tokens
- **PP:** `41.14 tok/s`
- **TG:** `4.81 tok/s`

This profile was stable and text output quality was clean, but decode remained modest for agentic use.

### 7. The big surprise: `q8_0/q8_0` transformed the profile
A final A/B was run at 64K, keeping the same successful no-thinking profile and only changing V-cache from `q4_0` to `q8_0`.

#### 64K, `q8_0/q8_0`, `threads=1`, `threads-batch=6`
- Prompt: `4527` tokens
- Completion: `193` tokens
- **PP:** `2261.79 tok/s`
- **TG:** `58.85 tok/s`

Load-time buffers also changed dramatically:

| Profile | CUDA0 compute | CUDA1 compute | CUDA_Host compute |
|---------|--------------:|--------------:|------------------:|
| 64K, `q8_0/q4_0` | 616.72 MiB | 554.48 MiB | 3453.67 MiB |
| 64K, `q8_0/q8_0` | 357.54 MiB | 412.98 MiB | 271.67 MiB |

This was not a small tuning win. It was a qualitative shift from “works, but slow” to “actually fast.”

---

## Interpretation

### What failed
- Older `llama.cpp` build without Gemma 4 architecture support
- Default/thinking-oriented chat behavior for text serving
- Multi-thread decode on this hardware profile (`threads > 1`) causing severe slowdown
- `q8_0/q4_0` KV choice producing a surprisingly poor real-serving path for this model/config

### What worked
- Text-only no-thinking profile
- `threads=1` for decode stability and speed
- `threads-batch=4..6` for prefill recovery
- `q8_0/q8_0` KV cache, which massively improved both PP and TG in the tested 64K configuration

### What this suggests
For Gemma 4 26B-A4B on this class of dual-12GB setup, **KV cache dtype choice is not a second-order detail**. It appears to materially alter the runtime path and overall serving behavior. The common llama.cpp advice of using `q8_0/q4_0` as a safe default does **not** generalize cleanly to this model on this hardware.

---

## Recommended text-only pilot profile

```bash
llama-server \
  -m gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8081 \
  --device CUDA0,CUDA1 \
  --split-mode layer \
  --gpu-layers all \
  --ctx-size 65536 \
  --parallel 1 \
  --flash-attn on \
  -ctk q8_0 -ctv q8_0 \
  --batch-size 512 \
  --ubatch-size 256 \
  --threads 1 \
  --threads-batch 6 \
  --jinja \
  --reasoning off \
  --reasoning-budget 0 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --ctx-checkpoints 0 \
  --cache-ram 0 \
  --metrics --slots --perf
```

---

## Caveats
- These tests were **text-only**. No `mmproj` or vision path was exercised here.
- The large `q8_0/q8_0` improvement is real in these server measurements, but should still be treated as a **validated profile on this setup**, not a universal Gemma 4 law.
- Multi-request concurrency was not evaluated here.
- The system was not tested with prompt caching, checkpoints, or thinking mode once the fast path was identified.

---

## Conclusion
Gemma 4 26B-A4B is viable on a dual RTX 3060 text-only setup in `llama.cpp`, but only once the serving profile is tuned for this model/hardware combination.

The key lessons were:
1. disable thinking for normal text serving,
2. keep decode at `threads=1`,
3. raise only `threads-batch` for prefill,
4. and, most importantly, **do not assume `q8_0/q4_0` is the right KV configuration**.

On this setup, the winning 64K text profile used **`q8_0/q8_0`** and delivered roughly **2262 tok/s prefill** and **58.9 tok/s decode** on a ~4.5K prompt with a ~193-token completion.
