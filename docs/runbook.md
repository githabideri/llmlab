# Runbook

## Qwen3.8-27B — Production (Port 8080, RTX 3090)

**Runtime:** vLLM 0.27.1 (W4A16-AutoRound, MTP k=3, fp8 KV, 160K ctx, keyless, text-only) in a **dedicated LXC** on the GPU server host — unit `llama-vllm-qwen3.8-27b.service` inside that LXC.

```bash
# From the Proxmox host (the vLLM LXC):
pct exec <vllm-lxc-id> -- systemctl status llama-vllm-qwen3.8-27b
pct exec <vllm-lxc-id> -- journalctl -u llama-vllm-qwen3.8-27b -f
# Health (from inside the LXC or over the LAN):
curl -s http://localhost:8080/health

# Rollback to stock llama.cpp (Q4_K_M + native MTP, dormant config)
# see models/qwen3.8-27b-rtx3090.md for the full llama.cpp command
```

> **History:** 2026-06-19: BeeLlama Qwen3.6-27B (DFlash) production → 2026-08-15: stock llama.cpp 5f754ea Q4_K_M+MTP → 2026-08-21: vLLM 0.27.1 cutover (current). The llama.cpp unit is kept dormant; the BeeLlama unit is a disk-only backup. See [`llama-cpp-systemd.md`](llama-cpp-systemd.md) and [`models/qwen3.6-27b-rtx3090.md`](../models/qwen3.6-27b-rtx3090.md).

### MTP Debugging (Qwen3.8 production — vLLM)

```bash
# vLLM spec-decoding stats in logs (acceptance, draft tokens)
pct exec <vllm-lxc-id> -- journalctl -u llama-vllm-qwen3.8-27b | grep -iE 'spec|acceptance'
# Per-request timing via the OpenAI API usage fields
```

### MTP Debugging (llama.cpp 35B dual-3060)

```bash
# Check draft acceptance in logs (MTP: 'draft acceptance' / 'mean len')
journalctl -u llama-server-qwen3.6-vision | grep -i 'draft acceptance\|mean len'
# Metrics: per-slot timings + /metrics
curl -s http://localhost:8081/metrics
```

### DFlash Debugging (Historic — BeeLlama only)

```bash
# Enable DFlash profiling (set before starting service or via env override)
export GGML_DFLASH_PROFILE=1    # summary + timing
export GGML_DFLASH_DEBUG=1      # prefill route, capture decisions
export GGML_DFLASH_KV_CACHE_MODE=both  # control drafter KV cache

# Check draft acceptance in logs
journalctl -u beellama-qwen3.6-27b | grep -i 'draft\|dflash'
```

### Common Issues (symptom → fix)

| Symptom | Check | Fix |
|---------|-------|-----|
| ~11 tok/s (not 40+) | GDN warning in logs | Add `-ngl all` — required for Qwen3.5/3.6 |
| Slow prefill | `-ub` flag | Raise `-ub` up to the VRAM limit (1024–2048) — see [2026-08-28 ubatch report](../reports/2026-08-28-llama-cpp-ubatch-moe-single-gpu.md) |
| CUDA OOM on start | `nvidia-smi` | Reduce `--ctx-size` or check for other processes on GPU |
| Draft acceptance 0% | `--spec-draft-model` path | Verify draft GGUF exists and is correct model |
| "failed to parse grammar" with tool schemas | build's repetition threshold | Raise `MAX_REPETITION_THRESHOLD` — [grammar workaround](llama-cpp-grammar-workaround.md) |
| "Flash Attention was auto, set to disabled" | build flags | Rebuild llama.cpp with FA all-quants support |

Unit-level failures (won't start, metrics missing, OOM patterns) are in [llama-cpp-systemd.md → Troubleshooting](llama-cpp-systemd.md#troubleshooting).

## llama.cpp — start (Nemotron recommended profile)

```bash
llama-server \
  --model /path/to/Nemotron-3-Nano-30B-A3B-IQ4_NL.gguf \
  --ctx-size 196608 \
  --parallel 1 \
  --slot-save-path /path/to/slots \
  --split-mode row --tensor-split 1,1 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --flash-attn on \
  --jinja \
  --reasoning-format deepseek \
  --reasoning-budget -1 \
  --host 0.0.0.0 --port 8080
```

## llama.cpp — start (Qwen3.6-35B-A3B, dual 3060)

Normally run via the `llama-server-qwen3.6-vision.service` unit; manual equivalent (matches the live unit, 2026-08-27):

```bash
llama-server \
  --device CUDA0,CUDA1 \
  --model /mnt/models/gguf/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --mmproj /mnt/models/gguf/qwen3.6-35b-a3b-mtp/mmproj-F16.gguf --no-mmproj-offload --image-max-tokens 1024 \
  --ctx-size 262144 \
  --parallel 2 \
  --split-mode tensor --tensor-split 50,50 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --batch-size 2048 --ubatch-size 1024 \
  --flash-attn on \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --jinja \
  --host 0.0.0.0 --port 8081
```

See the [model card](../models/qwen3.6-35b-a3b.md) for the full config and the earlier Qwen3.5 Q4_K_M (24 GB) profile.
- This configuration keeps dual 3060 usage below the 12GB/card cliff while preserving text+tools+vision.

## Alternate speed mode (lower trust)

```bash
# Same as above, but:
--reasoning-budget 0
```

Use only when latency matters more than output cleanliness.

## Health check

```bash
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/v1/models | head
```

## Common adjustments
- **Lower memory:** reduce `--ctx-size`, reduce `--parallel`, or tune KV cache types (see `kv-cache-sizing.md` for per-model calculations)
- **Balance GPUs:** adjust `--tensor-split A,B`
- **Reasoning volume:** keep `--reasoning-format deepseek`; tune prompt policy before forcing budget 0

## Stop

```bash
pkill -x llama-server
```

## Legacy benchmark profile (Qwen REAP 40B)

```bash
llama-server \
  --model /mnt/models/gguf/qwen3-coder-next/Qwen3-Coder-Next-REAP-40B-A3B-Q2_K_XL.gguf \
  --ctx-size 131072 -b 128 -ub 64 \
  --n-gpu-layers -1 --split-mode row --tensor-split 14,10 \
  --n-cpu-moe 0 \
  --cache-type-k q8_0 --cache-type-v q4_0 \
  --flash-attn 1 --fit off --no-warmup --jinja \
  --host 0.0.0.0 --port 8080
```
