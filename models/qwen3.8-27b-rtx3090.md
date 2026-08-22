# Qwen3.8-27B on RTX 3090

**Model:** Qwen3.8-27B (Dense, hybrid SSM + attention)  
**Tested Quantization:** W4A16-AutoRound (vLLM production), Q4_K_M (llama.cpp baseline/rollback), UD-Q4_K_XL (tested, heavier)  
**Hardware:** 1× RTX 3090 24 GB  
**Runtime:** vLLM 0.27.1 (production since 2026-08-21); llama.cpp 5f754ea retained as validated fallback  
**Status:** ✅ Production — vLLM (W4A16-AutoRound, MTP k=3, fp8 KV, 163840 ctx, keyless, text-only); llama.cpp Q4_K_M+MTP config below remains the documented rollback  
**Multimodal:** ✅ Deployed (mmproj BF16 on CPU via `--no-mmproj-offload`)  
**Supersedes:** Qwen3.6-27B BeeLlama deployment (see [`qwen3.6-27b-rtx3090.md`](qwen3.6-27b-rtx3090.md))

---

## Quick Facts

| Parameter | Value |
|-----------|-------|
| **Parameters** | ~27.3 billion |
| **Context Window** | 262,144 tokens (native), 163,840 deployed |
| **Embedding Dimension** | 5120 |
| **Vocabulary Size** | 248,320 |
| **Quantization** | Q4_K_M (17.1 GB) |
| **Multimodal** | ✅ Deployed (mmproj BF16, ~0.9 GB, CPU-resident) |
| **Speculative Decoding** | MTP (draft-mtp, n-max 2, p-min 0.4) — no separate draft model |
| **Full-attention layers** | 17 of 66 (rest are SSM/hybrid) |

---

## llama.cpp Config (baseline / rollback)

**Runtime:** stock llama.cpp, commit `5f754ea` (unmodified upstream)  
**KV Cache:** q8_0 / q8_0  
**Batch:** `-b 512 -ub 64`, single slot

```bash
llama-server \
  --device CUDA0 \
  -m /path/to/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /path/to/mmproj-BF16.gguf \
  --no-mmproj-offload \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.4 \
  -ngl all \
  -np 1 \
  --ctx-size 163840 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -b 512 -ub 64 \
  --flash-attn on \
  --fit off \
  --jinja \
  --reasoning on \
  --reasoning-preserve \
  --metrics \
  --mmap --mlock \
  --host 0.0.0.0 \
  --port 8080
```

### Why stock llama.cpp, not BeeLlama
Qwen3.8-27B ships with native MTP heads, so built-in `draft-mtp` speculative decoding replaces the BeeLlama DFlash draft model entirely (no separate ~1 GB drafter). At 160K context this leaves comfortable headroom (below), so there is no longer a memory reason to stay on the BeeLlama fork.

### Performance (measured 2026-08-15, Q4_K_M, 160K ctx)

| Metric | Value |
|--------|-------|
| Prefill (short prompt) | **~575 tok/s** |
| Prefill (131K prompt) | **~390 tok/s** |
| Generation (sustained) | **34.7–37 tok/s** |
| MTP draft acceptance | 66–94% (typ. ~75%) |
| Long-context recall | 3/3 needles at ~128K |
| Vision | Accurate (mmproj on CPU) |

### VRAM / headroom

| Quant | 160K headroom | Note |
|-------|--------------|------|
| UD-Q4_K_XL (17.9 GB) | **398 MiB** | Too tight for production |
| **Q4_K_M (17.1 GB) — deployed** | **1,138 MiB** | Chosen for production |

Under sustained load VRAM holds **23,235 MiB / 24 GB** and stays flat (verified through a real 131K-token workload).

### Why the feared quantized-KV FA pathology does not appear here
The upstream `f8f0a47a` "quantized-KV flash-attention scratch blowup" does **not** manifest on `5f754ea` for this model:
- Only 17 of 66 layers are full-attention (hybrid SSM arch), so KV is only ~64 KB/token at f16-equivalent accounting.
- At decode the **VEC** kernel is selected, which does not require an F16 K/V mirror for quantized KV.
- Verified **empirically**: flat VRAM through a genuine 131K-token workload (not assumed).

---

## Known Limits

- **160K context** is the deployed size; the model's native 262K needs more VRAM than a single 3090 has with q8 KV.
- **No multi-GPU split** — designed for a single 24 GB GPU.
- MTP requires the GGUF to include MTP heads (it does, natively for Qwen3.8).

---

## Changelog

### 2026-08-21/22: Production runtime moved to vLLM
- vLLM 0.27.1 (PyTorch 2.13.0+cu130) now serves `Qwen3.8-27B-W4A16-AutoRound` (19.5 GB; int8 embed + MTP int4 "fast" prep), following the public recipe `syv-ai/qwen38-27b-rtx3090` @ `999e264b` (13 patches, all applied).
- MTP k=3 speculative decoding, fp8 KV (FlashInfer), 163,840 max len, prefix caching, reasoning parser (`qwen3`), tool calling (`qwen3_coder`), keyless, text-only (`--language-model-only`).
- Validated on the same RTX 3090: decode 96–118 tok/s (vs ~35 for llama.cpp Q4_K_M), prefill up to ~1,050 tok/s, 3/3 needles at ~155K, prefix-cache second turn 281 s → 2.8 s, PPL 8.095 (matches recipe reference), GSM8K 96.0% (n=200).
- The llama.cpp Q4_K_M + MTP config above is the validated fallback/rollback.

### 2026-08-15: Production cutover to Qwen3.8-27B (stock llama.cpp + MTP)
- Replaced Qwen3.6-27B BeeLlama (DFlash) on port 8080.
- Stock llama.cpp 5f754ea, Q4_K_M, 160K ctx, q8_0/q8_0 KV, MTP (n-max 2, p-min 0.4).
- Chose Q4_K_M over UD-Q4_K_XL for 1,138 MiB headroom (vs 398 MiB).
- Long-context 3/3 needles at ~128K; vision verified; 5/5 stability runs clean.
- BeeLlama Qwen3.6-27B service disabled (kept for rollback). See [`qwen3.6-27b-rtx3090.md`](qwen3.6-27b-rtx3090.md) for its history.

---

## Related Documentation

- **Systemd / server config:** [`docs/llama-cpp-systemd.md`](../docs/llama-cpp-systemd.md)
- **Predecessor (Qwen3.6-27B BeeLlama):** [`qwen3.6-27b-rtx3090.md`](qwen3.6-27b-rtx3090.md)
- **BeeLlama backend (historic):** [`docs/backend-beellama.md`](../docs/backend-beellama.md)
