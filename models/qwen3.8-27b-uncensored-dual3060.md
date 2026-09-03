# Qwen3.8-27B-Uncensored on Dual RTX 3060

**Model:** Qwen3.8-27B-Uncensored (abliterated, [JonathanColetti](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF))  
**Base:** orcarouter/Qwen3.8-27B-Uncensored → Qwen3.8-27B (dense, hybrid SSM + attention)  
**Tested Quantization:** Q4_K_M (15.66 GB)  
**Hardware:** 2× RTX 3060 12 GB (tensor split 50/50)  
**Runtime:** llama.cpp router mode (`--models-preset`), commit `925e1179`  
**Status:** ✅ Production — on-demand load (not auto-loaded; the 35B-A3B is the default)  
**Multimodal:** ✅ mmproj F16 (885 MB) on GPU  
**Censored:** No — abliterated (Heretic method), 12/100 refusals on test set, MMLU −0.2, mean score −0.5 (within noise)  
**Supersedes:** nothing — this is a second model on the same node, not a replacement

---

## Quick Facts

| Parameter | Value |
|-----------|-------|
| **Parameters** | ~27.3 billion |
| **Context Window** | 262,144 tokens (native), 65,536 deployed |
| **Quantization** | Q4_K_M (15.66 GB) |
| **Multimodal** | mmproj F16 (885 MB), on GPU (not offloaded) |
| **Speculative Decoding** | MTP (draft-mtp, n-max 3) |
| **VRAM (measured)** | 11,390 + 10,250 MiB across dual 3060 (92.7% / 83.4%) |
| **Uncensored** | Yes — 12/100 refusals (vs ~100% for base Qwen3.8) |

---

## Router Mode Deployment

This model is deployed via **llama.cpp router mode** alongside the 35B-A3B on the same dual-3060 node. Only one model is in VRAM at a time (`--models-max 1`). The 27B is loaded on-demand via:

```bash
# Load (evicts the 35B)
curl -X POST http://<gpu-server>:8081/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b-uncensored"}'

# Switch back
curl -X POST http://<gpu-server>:8081/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"}'
```

`--no-models-autoload` prevents the 27B from loading unless explicitly requested. Cold load from SATA SSD: ~30–40 s; warm (page cache): ~5–10 s.

### Preset INI (per-model section)

```ini
[qwen38-27b-uncensored]
model = /path/to/Qwen3.8-27B-Uncensored-Q4_K_M.gguf
mmproj = /path/to/mmproj-Qwen3.8-27B-Uncensored-F16.gguf
ctx-size = 65536
# No load-on-startup, no no-mmproj-offload (mmproj stays on GPU)
```

Shared `[*]` section provides: `device = CUDA0,CUDA1`, `split-mode = tensor`, `tensor-split = 50,50`, `cache-type-k q8_0`, `cache-type-v q4_0`, `batch-size 2048`, `ubatch-size 1024`, `flash-attn on`, `spec-type draft-mtp`, `spec-draft-n-max 3`, `parallel 2`, `jinja`, `metrics`, `cache-prompt`, `cache-ram 2048`.

---

## Performance (measured 2026-09-03, initial / cold)

| Metric | Value |
|--------|-------|
| Prefill (57-token prompt) | ~104 tok/s |
| Decode (10 tokens, MTP active) | ~33 tok/s |
| MTP draft acceptance | 6/8 (75%) |
| VRAM | 11.4 + 10.3 GB (dual 3060) |

> **Note:** these are cold numbers from the first request after model load. Warm throughput (after JIT warmup and page cache) is expected to be significantly higher. The 35B-A3B (MoE, smaller active params) runs at 100–114 t/s on the same hardware; the 27B dense model is expected to be slower per-token due to all 27B params being active.

---

## Why This Model

- **Q4_K_M is the ceiling quant for dual 3060 with vision.** FP8 (28.75 GB) doesn't fit (14.4 GB/GPU > 12 GB). Q4_K_M at 15.66 GB → ~7.8 GB/GPU → ~4.2 GB/GPU KV headroom.
- **MTP is inline** (native to the GGUF, no separate draft model).
- **mmproj F16** is a separate 885 MB file. On GPU (not offloaded) — the 27B has enough headroom, unlike the 35B which uses `--no-mmproj-offload` (CPU) at 96% VRAM.
- **Abliteration quality:** 12/100 refusals (vs 0/100 for aggressive abliteration like HauhauCS). MMLU delta −0.2, mean benchmark delta −0.5 — within noise of the base model.
- **Open access:** no HuggingFace gating (unlike orcarouter's GGUF repo).

---

## Known Limits

- **One model at a time:** loading the 27B evicts the 35B (and vice versa). Agents (OpenClaw/Dolly) that depend on the 35B will have requests queue during the transition.
- **Dense, not MoE:** all 27B params active per token, so per-token decode is slower than the 35B-A3B (only 3B active). Expect ~30–60 t/s vs 100–114 t/s.
- **64K context** deployed (vs 256K for the 35B) to fit more KV headroom.
- **Vision via mmproj on GPU** — the `/models` endpoint reports `input_modalities: ["text"]` despite the mmproj loading successfully (llama.cpp metadata quirk). Image input works via the API.

---

## Changelog

### 2026-09-03: Deployed in router mode
- Downloaded Q4_K_M (15.66 GB) + mmproj F16 (885 MB) from JonathanColetti HF.
- Added to the 3060 router preset INI. Tested: load (evicts 35B), text inference (MTP 75%), switch-back (35B restored).
- Cold load ~30 s from SATA SSD. Warm loads faster (page cache).

---

## Related Documentation

- **35B-A3B on the same node:** [`qwen3.6-35b-a3b.md`](qwen3.6-35b-a3b.md)
- **27B on RTX 3090 (vLLM, text-only):** [`qwen3.8-27b-rtx3090.md`](qwen3.8-27b-rtx3090.md)
- **Dual 3060 optimization:** [`../reports/2026-08-27-qwen3.6-35b-a3b-dual-3060-optimization.md`](../reports/2026-08-27-qwen3.6-35b-a3b-dual-3060-optimization.md)
