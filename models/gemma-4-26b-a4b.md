# Gemma 4 26B-A4B

**Model:** Gemma 4 26B-A4B Instruct  
**Tested Quantization:** `UD-Q4_K_M`  
**Hardware:** 2× RTX 3060 12GB (24 GB total VRAM)  
**Status:** 🟡 Pilot-viable for text-only serving on llama.cpp

---

## Quick Facts

| Parameter | Value |
|-----------|-------|
| **Architecture** | `gemma4` |
| **Model family** | Gemma 4 |
| **Parameters** | 26B total / 4B active (A4B) |
| **Context Window** | 262,144 tokens |
| **Quant tested** | `UD-Q4_K_M` |
| **Serving path tested** | llama.cpp, dual-GPU layer split |
| **Primary use tested** | text-only OpenAI-compatible chat serving |

---

## Summary verdict

Gemma 4 26B-A4B can run well on a 24 GB dual-RTX-3060 class setup, but the default-looking profile was misleadingly bad.

Three issues had to be solved before the model became useful:
1. a Gemma-4-capable `llama.cpp` build,
2. explicitly disabling thinking mode for text serving,
3. and choosing a KV cache configuration that does not kneecap the runtime.

The most important empirical result from this setup was that **`q8_0/q8_0` KV cache dramatically outperformed `q8_0/q4_0`** in real server runs.

---

## Working text-only profile

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

### Why this profile
- **`threads=1`** fixed severe decode slowdown seen with a more CPU-heavy profile.
- **`threads-batch=6`** recovered prompt-processing speed without harming decode.
- **`q8_0/q8_0`** was the decisive change that turned the model from “stable but sluggish” into “actually fast.”
- **No-thinking mode** was required to avoid responses being spent in hidden reasoning content.

---

## Performance

### Tiny smoke test (8K, near-empty context)
Using the no-thinking profile with `threads=1`, `threads-batch=1`:

| Prompt tokens | Completion tokens | PP tok/s | TG tok/s |
|--------------:|------------------:|---------:|---------:|
| 39 | 26 | 200.75 | 35.10 |

This confirmed that Gemma 4 was capable of healthy decode speed on the hardware once the bad thread profile was removed.

### Real prompt load (~4.5K tokens)
All runs below used the same repeated-text prompt and a structured short analysis task.

| Context | KV cache | Threads / batch | Prompt tokens | Completion tokens | PP tok/s | TG tok/s |
|--------:|---------|----------------:|--------------:|------------------:|---------:|---------:|
| 32K | `q8_0/q4_0` | `1 / 1` | 4517 | 20 | 12.46 | 5.16 |
| 32K | `q8_0/q4_0` | `1 / 4` | 4517 | 20 | 30.48 | 4.97 |
| 32K | `q8_0/q4_0` | `1 / 6` | 4517 | 20 | 32.07 | 5.05 |
| 32K | `q8_0/q4_0` | `1 / 8` | 4517 | 20 | 32.29 | 5.01 |
| 32K | `q8_0/q4_0` | `1 / 6` | 4527 | 142 | 33.71 | 4.81 |
| 64K | `q8_0/q4_0` | `1 / 6` | 4527 | 155 | 41.14 | 4.81 |
| **64K** | **`q8_0/q8_0`** | **`1 / 6`** | **4527** | **193** | **2261.79** | **58.85** |

### Interpretation

#### `threads-batch` plateau
For `q8_0/q4_0`, prompt processing improved sharply from `threads-batch=1` to `4`, then mostly plateaued by `6..8`.

#### 32K vs 64K under the slow KV profile
With `q8_0/q4_0`, 64K was not catastrophically worse than 32K for a ~4.5K prompt, but decode stayed around **~5 tok/s**, which is only modestly usable.

#### KV cache choice changed everything
The 64K `q8_0/q8_0` run produced a completely different performance envelope. On this setup, it was not a marginal tradeoff; it was the difference between a sluggish pilot and a responsive one.

---

## Memory observations

### 64K with `q8_0/q4_0`
| Buffer | Size |
|--------|-----:|
| CUDA0 compute | 616.72 MiB |
| CUDA1 compute | 554.48 MiB |
| CUDA_Host compute | 3453.67 MiB |

### 64K with `q8_0/q8_0`
| Buffer | Size |
|--------|-----:|
| CUDA0 compute | 357.54 MiB |
| CUDA1 compute | 412.98 MiB |
| CUDA_Host compute | 271.67 MiB |

The host-side compute buffer collapse when switching to `q8_0/q8_0` was especially notable and aligned with the large throughput improvement.

---

## Known issues and gotchas

### 1. Old builds may not recognize Gemma 4 at all
Earlier `llama.cpp` builds failed with:

```text
unknown model architecture: 'gemma4'
```

This is a hard compatibility failure, not a tuning issue.

### 2. Default chat behavior can look “working” while still being unusable
Without explicit no-thinking settings, the server can return HTTP 200 responses while spending output budget in reasoning fields rather than visible assistant content.

### 3. “Safe default” KV quant advice did not transfer cleanly
The common `q8_0/q4_0` KV recommendation worked technically but performed badly on this tested configuration. Do not assume the usual llama.cpp KV defaults are correct for Gemma 4 without measuring them.

---

## Recommendation

For a dual-12GB consumer GPU setup:
- **Use a Gemma-4-capable build** of `llama.cpp`
- **Disable thinking** for text-only server use
- **Start with `threads=1`**
- **Tune prefill with `threads-batch`** separately from decode
- **Test `q8_0/q8_0` first**, even if you would normally reach for `q8_0/q4_0`

At least on this hardware, the best validated text-serving profile was **64K context, `threads=1`, `threads-batch=6`, and `q8_0/q8_0` KV cache**.

---

## Related experiment

Full write-up: [`experiments/2026-04-12-gemma-4-26b-a4b-dual-3060-llama-cpp.md`](../experiments/2026-04-12-gemma-4-26b-a4b-dual-3060-llama-cpp.md)
