# BeeLlama.cpp DFlash Cutover — Qwen3.6-27B on RTX 3090

**Date:** 2026-06-19
**Goal:** Replace mainline llama.cpp Q4_K_M with BeeLlama.cpp Q5_K_S + DFlash speculative decoding on the RTX 3090. Validate decode speedup, prefill performance, and vision support.
**Setup:** BeeLlama.cpp b10102 (commit `85e22ea0b`), CUDA 12.5, RTX 3090 24 GB, compute 86.
**Result:** Success — DFlash delivers 2-2.2x decode speedup on structured output. Prefill at 791 tok/s (19K tokens). Vision deployed.

---

## Background

Our Qwen3.6-27B was running on mainline llama.cpp v616 with Q4_K_M quantization (~24-25 tok/s decode, no speculative decoding). BeeLlama.cpp offers DFlash speculative decoding — a small draft model (~1 GB) that cross-attends to the target's hidden states and proposes tokens ahead for batch verification.

**Upstream claims** (from [BeeLlama README](https://github.com/Anbeeld/beellama.cpp)): up to 4.4x speedup on structured generation (code, JSON), ~2x on multi-turn coding, ~1x on free-form prose. All on RTX 3090, same model family.

---

## Pre-Flight State

| Service | Port | Model | GPU |
|---------|------|-------|-----|
| llama-server (mainline) | 8080 | Qwen3.6-27B-Q4_K_M | CUDA0 (3090) |
| llama-server (mainline) | 8081 | Qwen3.6-35B-A3B-IQ4_XS | CUDA1+2 (dual 3060) |
| llama-swap | 8090 | proxy | — |

Old 27B service stopped. BeeLlama started manually on port 8080 (same port, seamless for llama-swap proxy). Old service kept enabled for boot safety.

---

## Session 1 — Config Failure (Discarded)

First launch used an incomplete config missing critical flags:

| Missing Flag | Impact |
|--------------|--------|
| `-ngl all` | GDN warning — layer 0 fell to CPU, fused GDN disabled. ~11 tok/s instead of 30+ |
| `--kv-unified` | Memory inefficiency |
| `-ub 512` | Used 128 — ~4x slower prefill |
| `--reasoning on` | Poorer drafter context |

Root cause: [llama.cpp #24712](https://github.com/ggml-org/llama.cpp/issues/24712) — fused GDN tensors + `--fit on` (auto-fit) cannot handle Qwen3.5/3.6 architecture. Fix: always use `-ngl all` explicitly.

Session abandoned. Config corrected from [official quickstart](https://github.com/Anbeeld/beellama.cpp/blob/main/docs/quickstart-qwen36-dflash.md).

---

## Session 2 — Corrected Config, Benchmarks

### Launch Command

```bash
llama-server \
  -m Qwen3.6-27B-Q5_K_S.gguf \
  --mmproj mmproj-Qwen_Qwen3.6-27B-f16.gguf \
  --no-mmproj-offload \
  --spec-draft-model Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-dflash-cross-ctx 1024 \
  -ngl all --spec-draft-ngl all \
  --kv-unified -np 1 \
  -b 2048 -ub 512 \
  --ctx-size 163840 \
  --cache-type-k q5_0 --cache-type-v q4_1 \
  --flash-attn on --jinja \
  --no-mmap --mlock --no-host \
  --reasoning on \
  --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.6 --top-k 20 --top-p 1.0 --min-p 0.0 \
  --host 0.0.0.0 --port 8080
```

Server started, model loaded, `/health` returned `ok`, no GDN warnings in logs.

### Decode Benchmarks

Three workloads, measured via `/v1/chat/completions`. Model finishes naturally (`stop`), not truncated.

| Workload | Decode tok/s | Draft Proposed | Draft Accepted | Acceptance | Output tokens |
|----------|-------------:|---------------:|---------------:|-----------:|--------------:|
| Structured JSON (15 employees) | **80.2** | 9,471 | 3,152 | 33.3% | 3,792 |
| Code (Python doubly-linked list) | **75.0** | 11,676 | 3,582 | 30.7% | 4,369 |
| Free-form prose (essay) | **39.6** | 12,501 | 1,602 | 12.8% | 2,444 |
| **Baseline (no DFlash)** | ~37 | — | — | — | — |

**Speedup vs baseline:** 2.17x (JSON), 2.03x (code), 1.07x (prose).

**Observations:**
- DFlash excels on structured/repetitive output (JSON keys, code indentation, method signatures)
- Prose sees minimal gain — creative text is hard to predict, most drafts rejected
- Matches upstream claims closely (their README shows 4.4x on "task store module", 1.94x on multi-turn coding)
- Lower absolute speedup than upstream's best case (163-181 tok/s) — likely due to `--reasoning on` overhead and different prompt structure

### Prefill Benchmarks

Unique prompts (no cross-request caching), 0 cached tokens. Small decode (32 tokens) to isolate prefill.

| New Prompt Tokens | Prefill Speed | Time |
|-----------------:|-------------:|-----:|
| 19,280 | **791 tok/s** | 24.4s |
| 39,981 | **742 tok/s** | 53.9s |
| 81,383 | **618 tok/s** | 131.8s |

**Observations:**
- Prefill peaks around 19-20K tokens
- Graceful degradation as context grows (attention scales with KV cache)
- Below upstream's published ~1229 tok/s at 20K — they use `--reasoning off` for benchmark prompts
- DFlash does not accelerate prefill (only decode). Speculative decoding is a decode-phase optimization.

### VRAM Usage

| Component | VRAM |
|-----------|------|
| Target model (Q5_K_S) | ~18 GB |
| DFlash draft (Q4_K_M) | ~1 GB |
| KV cache (160K, q5_0/q4_1) | ~3.5 GB |
| Other (compute, mmproj swap) | ~0.7 GB |
| **Total** | **~23.7 GB / 24 GB** |

Vision projector loaded on CPU (`--no-mmproj-offload`) — no VRAM cost, latency cost on image input.

---

## Comparison: Old vs New

| Metric | Old (mainline Q4_K_M) | New (BeeLlama Q5_K_S + DFlash) | Change |
|--------|----------------------|--------------------------------|--------|
| Decode (structured) | ~25 tok/s | **80 tok/s** | +220% |
| Decode (code) | ~25 tok/s | **75 tok/s** | +200% |
| Decode (prose) | ~25 tok/s | **40 tok/s** | +60% |
| Prefill @ 20K | ~808 tok/s (Q4, -ub 128) | **791 tok/s** (Q5, -ub 512) | ~equal |
| Context | 204K | 160K | -22% |
| Vision | Tested, not deployed | Deployed | + |
| Quant | Q4_K_M | Q5_K_S | Higher quality |
| KV cache | q8_0/q8_0 | q5_0/q4_1 | Smaller, asymmetric |

**Tradeoff:** Lost 44K of context (204K to 160K) to fit DFlash drafter + vision. Gained 2-3x decode speed on our primary workloads (code, JSON, tool calls). Net positive for agentic use.

---

## Conclusions

1. **DFlash delivers** — 2-2.2x on structured output matches expectations. Prose sees minimal gain (expected).
2. **Prefill is solid** — 791 tok/s at 19K, comparable to mainline. No regression.
3. **Vision works** — deployed with `--no-mmproj-offload`, no VRAM cost.
4. **Context tradeoff accepted** — 160K is sufficient for our agentic workloads. 44K loss is the cost of DFlash + vision on 24 GB.
5. **Config matters** — `-ngl all` is required for Qwen3.5/3.6. Missing it causes silent GDN fallback to CPU.

---

## Operational Note: Observed Speculative Decoding Artifact

During this session, the agent running through the BeeLlama endpoint exhibited a repetitive error pattern: when using a file-edit tool, it generated an incorrect file path and repeated the exact same failed edit call **over 20 times** without self-correcting.

This behavior was not observed with the previous mainline llama.cpp setup (Q4_K_M, no speculative decoding).

### Hypothesis

DFlash speculative decoding works by having a small draft model propose tokens ahead, which the target model then verifies in batch. From the target model's perspective, drafted tokens are indistinguishable from self-generated tokens — there is no metadata marking their origin. Verification is probabilistic: "would I have picked this token?" If yes, accept.

The theory is:

1. The draft model learned an incorrect path pattern from earlier in the conversation (where it appeared in tool output)
2. On each edit attempt, the drafter proposed the same incorrect path as the next tokens
3. The target model verified: "plausible path given conversation context" — accepted
4. The edit failed (file not found at that path), the error was returned
5. The cycle repeated — the drafter again proposed the same path, the target again accepted

This would be speculative decoding working exactly as designed — optimizing for "most likely next token" rather than "correct answer." The drafter is good at predicting what the target will write, including the target's errors. The verification threshold is not "is this correct?" but "is this the most likely continuation?"

### Caveats

- This is an observed correlation, not a proven causal chain. The target model cannot introspect whether a token was drafted or self-generated.
- The conversation context (prior tool output showing incorrect paths) provided strong priors for the wrong path.
- This may be specific to the interaction between tool-use patterns, path generation, and the draft model's cross-attention context window.
- It may not reproduce under different workloads or with different draft models.

### Takeaway

If confirmed, this illustrates a subtle risk of speculative decoding in agentic/tool-use settings: the drafter can reinforce the target's mistakes by drafting the same wrong continuation repeatedly, and the target has no internal mechanism to detect that it is being led. The loop only breaks through external intervention (user correction) or context shift.

Worth monitoring in production. If it recurs, mitigations could include: reducing `--spec-dflash-cross-ctx` (less context for the drafter to learn error patterns), disabling speculation during tool-call generation, or adding external loop detection.

---

## Resources

| Resource | URL |
|----------|-----|
| BeeLlama.cpp fork | <https://github.com/Anbeeld/beellama.cpp> |
| Quickstart: Qwen 3.6 DFlash | <https://github.com/Anbeeld/beellama.cpp/blob/main/docs/quickstart-qwen36-dflash.md> |
| DFlash draft model | <https://huggingface.co/Anbeeld/Qwen3.6-27B-DFlash-GGUF> |
| KV cache quant benchmarks | <https://anbeeld.com/articles/kv-cache-quantization-benchmarks-for-long-context> |
| llama.cpp GDN issue | <https://github.com/ggml-org/llama.cpp/issues/24712> |
