# LFM3-1.7B on vLLM at the 3060: prefix caching works to the 262K roof, thinking is narration, decode is bandwidth physics

**2026-11-07** · experiment / deployment · dedicated LXC on the secondary GPU server (i5-7400, RTX 3060 12 GB) · vLLM `0.33.0rc1.dev987+g8176238c9` (main, commit `8176238c9`, 2026-11-06) · LFM3-1.7B BF16, 262K context · companion to [2026-11-06-lfm3-qad-1.7b-2gb-fleet-matrix](2026-11-06-lfm3-qad-1.7b-2gb-fleet-matrix.md) (same model, llama.cpp route)

**TL;DR**

- SGLang — the vendor's primary serving path for LFM3 — has no LFM3 yet (WIP PR sgl-project/sglang#46858, unmerged and non-mergeable as of 2026-11-07). vLLM main merged LFM3 text on 2026-11-05/06 and treats the 13 width-7 causal-conv layers as short-conv "mamba-style" inner state *with state copy for the hybrid block manager* — the checkpoint/restore mechanism the llama.cpp route lacks. vLLM is therefore the only working LFM3 prefix-caching path, and **it works: no cliff anywhere up to the 262K roof** (258K prefix re-reads in 0.81 s warm, 45.9× the cold 37.2 s; 100K = 55×, 200K = 77×).
- The 12 GB card holds a **313,545-token KV pool** at 0.90 utilization (6.48 GiB): 3.36 GiB BF16 weights + ~1.0 GiB engine overhead (activations + CUDA graphs). The "only ~310K" is not a config bug — vLLM pads the hybrid's mamba state to the attention page size (433.33%) plus 2+4 padding layers, inflating the effective cost to ~21.4 KiB/token (12 KiB of real full-attention KV). It cannot be meaningfully grown on this card without lower-precision weights (Ampere has no FP8 tensor cores; the vendor's FP8/NVFP4 LFM3 builds target other silicon).
- **The 1.7B text checkpoint barely uses its `
` block on vLLM.** ~27 requests across two server configurations (A/B: the serving flags are neutral), three samplings, five prompt families (math, logic, explicit step-by-step, strict JSON, multi-turn with a think block planted in history): zero open tags. It "thinks" as plain-text narration, sometimes closed by a bare `
`. The vendor's own card concedes the mode is "inconsistent". The `qwen3` reasoning parser still delivers the clean split: narration lands in `reasoning_content`, the answer in `content`, no stray tags in either; tool calling (`qwen3_coder`) and strict JSON work.
- Decode on this card is bandwidth physics, not engine tuning: 336 GB/s ÷ 3.46 GB BF16 ≈ **97 t/s ceiling**; measured 87.8 (small context, 90% of ceiling) and 60–67 t/s in a live ~80K-token session (KV reads add up). The llama.cpp QAD route's 239 t/s (1.05 GB; ceiling ~320) stays the decode champion; vLLM BF16 is the long-session / cached-context engine.
- A trap found the hard way: capping `max_tokens` at 512 while the model is mid-narration returns a `finish=stop` response with **both fields empty** (the vendor's docs warn about exactly this). Agent deployments need a 4K–16K cap, not a small one.

## Goal

The llama.cpp route for LFM3 (11-06 report) re-prefills the entire conversation history on every turn: the hybrid conv state cannot be checkpointed, so prefix caching is dead on that engine (llama.cpp PR #24891 is the open attempt at it). For a 50K+ token agentic session that is the whole cost of being slow. The question: **does vLLM give LFM3 real prefix caching on a 12 GB consumer card — and if so, what is the complete cost/benefit picture** (cache depth, VRAM accounting, thinking-mode behavior, tool calling, decode speed, and how it all compares against the llama.cpp route on the same card)?

Secondary: a reproducible build path for vLLM-main on a 4-core box without a multi-hour local compile, and a characterization of LFM3-1.7B's thinking mode under a production serving stack.

## Hardware

| Box | CPU | GPU | VRAM | Driver | Notes |
|---|---|---|---|---|---|
| "LFM3 node" — dedicated LXC on the secondary GPU server (i5-7400; the 11-06 "3060 node" is a second container on the same host) | 4 vCPU / 8 GB | RTX 3060 (sm_86) | 12 GB | 595-series (CUDA 13-generation host driver) | window-only box (off by default); the card is shared with the host's production 35B service, so each run stops that service, runs, and restores it (the 35B was offline ~15 min per window and smoke-checked after) |

## Setup

- **Model:** `LiquidAI/LFM3-1.7B` BF16 (3.46 GB single safetensors; 24 layers = 13 gated short-conv (L=7) + 6 full-attention + 5 sliding-window (w=256); hidden 2048, 16 heads, 4 KV heads, head_dim 128; vocab 128K; max position 262144).
- **Engine:** vLLM main, commit `8176238c9` (2026-11-06). LFM3 text merged 2026-11-05/06; the model class carries `MambaStateCopyFunc/Calculator` + `ShortConv` + `IsHybrid`/`HasInnerState` — vLLM's hybrid block manager checkpoints the conv state so prefixes are restorable. The current *released* vLLM (0.33.0) predates the merge; main is required.
- **Build (the reusable trick, ~5 min on 4 cores instead of hours):** vLLM publishes **precompiled wheels per commit** at `wheels.vllm.ai/<commit-sha>/cu130/` (one 239 MB wheel, `cp38-abi3`, with all prebuilt extensions and SASS for sm_75…sm_120, verified to include sm_86). Install with `--no-deps` into a venv holding `torch 2.14.0+cu130` plus vLLM's `requirements/cuda.txt` (the pinned `torch==2.14.0` rejects the local `+cu130` wheel under `==`, so deps must be satisfied out-of-band). A from-source build on this 4-core box would have taken hours (561 CUDA translation units).
- **Container gotchas (for anyone reproducing in an unprivileged LXC with GPU passthrough):** (1) bind-mounting the host's driver user-space files breaks `apt` transactions (cross-device backup-link failure) — copy the host's `libcuda.so.1`/`libnvidia-ml.so.1` into the container's library path and `ldconfig`; the CUDA compute API negotiates loosely enough that torch sees the GPU even though NVML still reports a version mismatch. (2) No `tmux` in the container → launch the server under `nohup`.
- **Server flags (final config):**

```bash
vllm serve /path/to/LFM3-1.7B --served-model-name lfm3-1.7b \
  --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --chat-template-content-format openai
```

The last three flags are the vendor's published LFM3 recipe (HF LFM3-VL-2B card's serve command; docs.liquid.ai's vLLM page; and the vendor's answer in the LFM3-1.7B HF discussion #4, which specifically chose `qwen3` over `qwen3-thinking` because the latter "incorrectly placed the final answer in reasoning_content").

## Commands

```bash
# prefix-cache ladder (two identical rounds; wall-clock ratio = the evidence)
for K in 13000 24000 31000 39000 47000 100000 200000 258000; do
  # fill ~4.28 chars/token; round 1 = cold, round 2 = warm
done

# engine accounting (from the startup log)
grep -E "Available KV cache memory|Model loading took|Graph capturing|GPU KV cache size|Padding mamba|Free memory on device" vllm.log

# thinking A/B: identical probes against (a) the vendor-flag server and (b) a bare
# server (no parser/tool flags); count emitted think tags in raw output
```

## Observations

### Prefix caching: it works, and there is no cliff

Same prompt twice, 8 output tokens, thinking as configured:

| prefix (tokens) | cold | warm | speedup |
|---|---|---|---|
| 12.9K | 0.32 s | 0.07 s | 4.4× |
| 23.5K | 1.70 s | 0.10 s | 17.5× |
| 31.3K | 1.59 s | 0.12 s | 13.6× |
| 39.2K | 1.75 s | 0.14 s | 13.0× |
| 47.0K | 1.92 s | 0.15 s | 12.5× |
| 100,327 | 15.98 s | 0.29 s | 55.0× |
| 201,362 | 47.83 s | 0.62 s | 76.7× |
| 257,957 | 37.18 s | 0.81 s | 45.9× |

Warm re-reads stay essentially flat (0.07–0.81 s) all the way to the model's 262,144-token roof — the number that matters for the "agentic session gets slower forever" complaint. Cross-request prefix sharing works too (a 60K-char prompt ran 0.08 s cold against a just-served 55K-char prompt: the shared head was served from cache). One anomaly: the very first large request after server start showed no speedup and inflated `prompt_tokens` accounting — a warmup artifact; every repeat at the same or larger size cached normally. A live agentic session observed in the engine log agrees: prefix-cache hit rate 68–78% across turns, ~26% of the pool in use (~80K tokens), 6.4K t/s on cold chunks, 60–67 t/s decode.

### The VRAM story: why "only" ~310K tokens

From the engine's own startup accounting (0.90 utilization on 12,288 MiB):

| Component | Size |
|---|---|
| Free at startup (after CUDA context) | 11.63 GiB |
| Utilization target (0.90) | 10.47 GiB |
| Weights (BF16; 3.22 GiB on disk) | 3.36 GiB |
| Non-torch overhead (torch/CUDA) | 0.18 GiB |
| Peak activations | 0.44 GiB |
| CUDA graphs | 0.41 GiB |
| **KV cache pool** | **6.48 GiB → 313,545 tokens** (1.20× concurrency at 262K) |
| nvidia-smi used | 10.63 GB |

Per-token cost: the 6 full-attention layers contribute 2 × 4 KV heads × 128 head_dim × 2 B = **12 KiB/token** (matches the llama.cpp route's measured 12,288 B/token). The 5 sliding-window layers hold only their 256-token window (2.6 MiB fixed per request) and the 13 conv layers carry ~315 KiB of state per request (fixed) — neither grows with length. But vLLM block-aligns the hybrid: the startup log reports "Padding mamba page size by 433.33% to ensure that mamba page size and attention page size are exactly equal" plus 2 and 4 padding layers, which inflates the effective per-token cost to ~21.4 KiB (6.48 GiB ÷ 313,545). The pool is therefore exactly "what fits", not a misconfiguration — and the practical cap for a single 262K request is the 1.18–1.20× concurrency the log prints.

### Thinking mode: the model narrates, it does not bracket

The vendor's position (model card): reasoning is **on by default**; `enable_thinking=False` "can be inconsistent, especially for reasoning-heavy tasks"; reasoning from turns before the latest user message is dropped unless `preserve_thinking=True`. Measured with the vendor flag set:

- **Zero open `
` tags in ~27 requests.** Two server configurations (vendor flags vs bare), samplings 0.1/top_k 50/rep 1.1, 1.0, and 1.0/top_p 0.95/top_k 20 (the vendor's math preset), prompt families: a two-train word problem, a syllogism, an explicit "think step by step" question, a strict-JSON demand, and multi-turn history with a genuine think block planted on the prior assistant turn. The model's "thinking" is plain-text narration ("The user says: …", "We need to solve…"), sometimes terminated by a bare `
` before the answer, sometimes not. Planting a think block in history did not induce the open tag (a control without `preserve_thinking` behaved identically) — this is the checkpoint's base habit under this stack, not a config effect. (The A/B against the bare server was the control that proved the flags change nothing about generation.)
- **The parser still earns its keep.** Whenever the model emits a close tag, `
` is the split point: everything before it lands in `reasoning_content`, the answer in `content`, and neither contains a stray tag. When it doesn't, the whole response is `content` — which is why the bare-server runs "leaked" narration into the answer while the configured server keeps the two channels clean in every case observed. `usage.completion_tokens_details.reasoning_tokens` is populated, so consumers can see the split even when the narration was minimal.
- **Tool calling and structured output work** with the vendor's `qwen3_coder` parser: a weather tool call came back as a structured `tool_calls` entry (`{"city":"Vienna"}`), and a strict-JSON demand returned exactly `{"color":"red","count":3}`.
- **The cap trap:** with a 512-token cap, a request that enters narration returns `finish_reason=stop` with *empty* `reasoning_content` and *empty* `content` — the response is silently lost (the narration fills the cap before any answer starts). The vendor's docs warn: "leave max_tokens unset unless you mean to cap output, since a low cap cuts off the reasoning models mid-thought." For agent use: 4K–16K cap, not less.

## Metrics (measured)

Decode, same card, both engines (the physics: RTX 3060 = 336 GB/s GDDR6):

| Engine / quant | weight bytes | theoretical ceiling | measured (small ctx) | measured (live ~80K session) |
|---|---|---|---|---|
| vLLM, BF16 | 3.46 GB | ~97 t/s | **87.8 t/s** (90% of ceiling) | **60–67 t/s** (KV reads add to the per-token traffic) |
| llama.cpp, QAD-Q4_0 | 1.05 GB | ~320 t/s | **239 t/s** (11-06 report) | — (no prefix caching; every turn re-prefills) |

A 3090-class card (936 GB/s) would ceiling at ~270 t/s in BF16 — the "300 t/s LFM3" numbers in circulation are faster-card or 4-bit numbers, unreachable on a 3060 in BF16.

Prefix-cache ladder as above; KV pool 313,545 tokens; engine startup log lines cited inline in Observations.

## Status

- **Deployed as a window-only container** on the secondary GPU server (off by default; started for a session, stopped afterwards). The production 35B service was stopped/started around each window and smoke-checked after (one window also surfaced a dead-but-logged-in state of a second service on the host — restored; details in the private repo). The box is reachable from the web UI through a persistent SSH tunnel on the UI host; when the container is off, the model simply reports unreachable (no ghost endpoint).
- **Not measured:** wall energy (no meter on this path — same gap as the 11-06 3060 windows), and the model's *sustained* multi-hour agentic behavior (the live-session observations above are from one user session of ~10 minutes).
- **Open:** llama.cpp PR #24891 (hybrid-state prefix caching for the quant route) is the other path; when it lands, the llama.cpp route gets caching too and the choice becomes decode speed vs cache depth rather than caching vs no-caching. FP8/NVFP4 vLLM builds of LFM3 exist but are irrelevant on Ampere (no FP8 tensor cores; the vendor's quantized cards target newer silicon).

## Conclusion

For LFM3 long-session work on a 12 GB Ampere card, **vLLM is the only engine with working prefix caching, and it works all the way to the 262K roof** — a 258K-token prefix re-reads in 0.81 s, and a live 50K+ agentic session keeps 70–80% of every turn's prompt out of the prefill path. The price is the BF16 bandwidth ceiling on this card (60–90 t/s decode, set by 336 GB/s, not by the engine) and a KV pool that the hybrid's block-alignment padding holds to ~310K tokens. The llama.cpp QAD route remains the faster decoder (239 t/s) for short sessions. The 1.7B's "thinking" under production serving is mostly plain-text narration with an unreliable bracket — consistent with the vendor's own "inconsistent" wording and with its positioning of this checkpoint as a fast local *agent* model; deployments should set a generous output cap, route `reasoning_content` back in multi-turn history, and treat structured `
` blocks as a bonus rather than a contract.
