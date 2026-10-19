# Ling-3.0-tiny

**Base model:** [InclusionAI/Ling-3.0-tiny](https://huggingface.co/InclusionAI/Ling-3.0-tiny) (Ling v3.0 generation, InclusionAI)
**Date tested:** 2026-10-19
**Status:** in production (on-demand) — since 2026-10-19

## Quick Facts

| Param | Value |
|---|---|
| Family | BailingMoE3 (Qwen3-derived), native 128K |
| Params | 7.9B total / **1.3B active** (128 routed experts, 8 routed + 1 shared per token; 4:1 GQA) |
| Attention | 3:1 **KDA–MLA hybrid** (18 KDA + 6 MLA layers out of 24) — KV cost is ~5× smaller than a comparable dense 7B |
| Quant | Q4_K_M (4.8 GB GGUF) |
| Context | **128K native**, q8_0 K/V (128K ≈ 0.45 GB); YaRN available to 256K (unused) |
| Capabilities | text, **thinking on by default**, **tool calling** (`ling3` grammar parser, `enable_thinking`/`reasoning_effort` in the toolset) |
| Runtime | upstream llama.cpp `f174553` (2026-10, BailingMoE3 support), sm_61 (CUDA 12.9), router mode, `--jinja --flash-attn` |
| Tested hardware | Pascal laptop — GTX 1060 6 GB Max-Q, 16 GB DDR4-3200 |

## Memory footprint (6 GB card)

| Item | VRAM |
|---|---:|
| Q4_K_M weights (mmap, hot in page cache) | ~4.5 GB |
| 128K q8_0 KV (allocated at load) | ~0.45 GB |
| CUDA/context overhead | ~0.45 GB |
| **Total, steady** | **~5.4 GB** (0.6 GB headroom) |

The KV is so small because only 6 of 24 layers pay a per-token price (MLA, compressed to 576 B/token); the 18 KDA layers keep a fixed recurrent state. Q5_K_M would not fit; Q4_K_M is the practical top quant.

## Performance (GTX 1060 6 GB, measured 2026-10-19)

| Workload | Result |
|---|---|
| Prefill, short prompt (36 t) | ~220 t/s |
| Prefill, 25K of a 75K prompt | ~190 t/s |
| Prefill, full **75,291-token** prompt | **99 t/s avg** (760 s), degrading to ~60–80 t/s in the tail (O(width) kernel, as everywhere) |
| Decode, short ctx | **~64 t/s** |
| Decode, 75K ctx | ~23 t/s |
| Needle @75K | retrieved verbatim from the middle of the context (the answer arrived in the thinking stream before the 60-token budget was spent) |

For reference, the 2-bit 35B on the same card prefilled 12.4K at 97.6 t/s — this 1.3B-active model prefills at 75K width faster than the 35B does at 12K.

## Production notes

- Served on the Pascal laptop behind a small stdlib-python **model mux** that fronts two VRAM-exclusive engines on one endpoint (the 2-bit 35B and this model); a chat request or an explicit load evicts the other, waits for the VRAM, and loads (a 202 ack keeps callers within a 2 s control budget). The laptop is headless-cockpit (tmux) with the display blanked; both models are also reachable through the llm-hub from anywhere on the tailnet.
- **Thinking is on by default and eats the token budget**: a 60-token `max_tokens` can be fully consumed by reasoning before any content is emitted. Size `max_tokens` for the model's thinking, or pass `enable_thinking: false` (via the toolset) for terse calls.
- No multimodal, no MTP in this build (the BailingMoE3 branch implements the KDA/MLA compute but not speculative decoding).
- 128K is a hard ceiling at this quant on 6 GB; the 256K YaRN stretch needs the KV to stay small — possible (it would cost ~0.9 GB) but untested.
