# LLM Serving — systemd Unit Reference

**Services (current):**
- `llama-vllm-qwen3.8-27b.service` — Qwen3.8-27B, **vLLM** (production since 2026-08-21, port 8080)
- `llama-server-qwen3.6-vision.service` — Qwen3.6-35B-A3B, llama.cpp, dual 3060 (production, port 8081)
- `llama-server.service` — Qwen3.6-35B-A3B MTP, llama.cpp, backup box (port 8080)
- `llama-qwen3.8-27b.service` — Qwen3.8-27B, llama.cpp (**dormant fallback** for the vLLM endpoint)
- plus historic units (BeeLlama DFlash, old longctx/reference configs)

**Logs:** `journalctl -u <service>`

> This file is the **unit reference** (what the units contain). Day-to-day operations — status, restart, rollback, debugging — live in [runbook.md](runbook.md).

---

## vLLM — Qwen3.8-27B (Port 8080, RTX 3090)

**Service:** `llama-vllm-qwen3.8-27b.service` — **production since 2026-08-21** (supersedes the llama.cpp 27B unit below)  
**Placement:** dedicated LXC on the GPU-server host (Ubuntu 24.04); from the host: `pct exec <lxc-id> -- systemctl status llama-vllm-qwen3.8-27b`  
**Runtime:** vLLM 0.27.1 (PyTorch cu130, Python 3.12) · **Model:** Qwen3.8-27B W4A16-AutoRound (19.5 GB; int8 embed + MTP int4 "fast" prep)  
**Effective config:** fp8 KV (FlashInfer) · MTP speculative decoding k=3 · max-model-len 163,840 · gpu-mem-util 0.93 · max-num-seqs 8 · prefix caching + mamba-align resume · `--reasoning-parser qwen3` · `--language-model-only` · tool-call parser · **keyless**

```ini
[Unit]
Description=vLLM Qwen3.8-27B W4A16 MTP fp8-KV 163k prefix-cache (RTX 3090)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/qwen38-vllm
Environment=HOME=/root
Environment=PORT=8080
Environment=CTX=long
Environment=SPEC=mtp
Environment=MAX_LEN=163840
Environment=PREFIX_CACHE=1
Environment="EXTRA_ARGS=--enable-auto-tool-choice --tool-call-parser qwen3_coder"
ExecStart=/bin/bash /opt/qwen38-vllm/single-user/start_qwen.sh
Restart=on-failure
RestartSec=10
TimeoutStartSec=25min
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

> ⚠️ **Do not replace the start script wholesale** — it carries the MTP/fp8/13-patch wiring (verified by the recipe's `verify.sh`). Add flags via `EXTRA_ARGS`, then restart (~3 min warm). The 35B vision unit's `ExecStartPre` polls this endpoint's :8080 health, so a vLLM restart transiently blocks the 35B boot chain.

---

## llama.cpp units

### Qwen3.8-27B (Port 8080, RTX 3090 — dormant fallback)

**Service:** `llama-qwen3.8-27b.service`  
**Unit:** `/etc/systemd/system/llama-qwen3.8-27b.service`  
**Status:** ⏸ **Dormant fallback** — since 2026-08-21 the 27B endpoint on :8080 is served by vLLM 0.27.1 in a separate LXC (see [runbook](runbook.md)); this unit stays on disk as the validated llama.cpp rollback  
**GPU:** RTX 3090 24GB (CUDA0)  
**Context:** 160K, q8_0/q8_0 KV, MTP speculative decoding  
**Cutover:** 2026-08-15 (replaced BeeLlama Qwen3.6-27B; stock llama.cpp 5f754ea); superseded by vLLM 2026-08-21

```ini
[Unit]
Description=llama.cpp Qwen3.8-27B MTP Vision - RTX 3090 (160K ctx, q8 KV)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp-20260815-5f754ea
Environment=CUDA_VISIBLE_DEVICES=0
Environment=LD_LIBRARY_PATH=/opt/llama.cpp-20260815-5f754ea
ExecStart=/opt/llama.cpp-20260815-5f754ea/llama-server \
  --device CUDA0 \
  -m /mnt/models/gguf/qwen3.8-27b/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /mnt/models/gguf/qwen3.8-27b/mmproj-BF16.gguf \
  --no-mmproj-offload \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.4 \
  -ngl all \
  -np 1 \
  --ctx-size 163840 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
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
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

### Qwen3.6-35B-A3B Vision (Port 8081, Dual RTX 3060)

**Service:** `llama-server-qwen3.6-vision.service`  
**Status:** ✅ Active, enabled — **current production** (optimization campaign 2026-08-27, [report](../reports/2026-08-27-qwen3.6-35b-a3b-dual-3060-optimization.md))  
**Model:** Qwen3.6-35B-A3B-UD-IQ4_XS (MTP variant) + vision F16 — mainline llama.cpp, `/opt/llama.cpp-mainline`  
**GPU:** 2× RTX 3060 12GB (tensor-split 50,50)  
**Context:** 256K, 2 parallel slots, MTP n=3 (acceptance 0.93–0.98)

Verified core of the live unit (`systemctl cat` for the full file):

```ini
[Service]
ExecStartPre=/usr/local/sbin/wait-for-8080-health   # gates on the 27B endpoint health before start
ExecStart=/opt/llama.cpp-mainline/build/bin/llama-server \
  --device CUDA0,CUDA1 \
  -m /mnt/models/gguf/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --mmproj /mnt/models/gguf/qwen3.6-35b-a3b-mtp/mmproj-F16.gguf \
  --no-mmproj-offload \
  --image-max-tokens 1024 \
  --host 0.0.0.0 \
  --port 8081 \
  -c 262144 \
  --parallel 2 \
  --split-mode tensor \
  --tensor-split 50,50 \
  --cache-type-k q8_0 \
  --cache-type-v q4_0 \
  --batch-size 2048 \
  --ubatch-size 1024 \
  --flash-attn on \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --jinja \
  --metrics \
  --cache-prompt \
  --cache-ram 2048
```

**Drop-in:** a sequence/QOS drop-in previously gated this unit on the 27B service; since the 27B moved to vLLM in a separate LXC (2026-08-21), those `Requires=`/`After=` lines are commented out — the `ExecStartPre` health poll is the live gate. Re-engage the drop-in only when rolling 27B back to llama.cpp.

### Qwen3.6-35B-A3B-MTP (Port 8080, RTX 3060, backup box)

**Host:** backup / inference box (LXC on a Proxmox host)  
**Service:** `llama-server.service`  
**Unit:** `/etc/systemd/system/llama-server.service`  
**Status:** ✅ Active, enabled  
**Deployed:** 2026-07-02  
**Model:** Qwen3.6-35B-A3B-UD-Q4_K_XL (22 GB, MoE, MTP)  
**GPU:** RTX 3060 12GB, CUDA 13.1, llama.cpp b9850  
**Context:** 128K, hybrid CPU+GPU offload

```ini
[Unit]
Description=llama.cpp Qwen3.6-35B-A3B-MTP (RTX 3060, 128K ctx)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp
Environment=LD_LIBRARY_PATH=/opt/llama.cpp/build/bin
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/opt/llama.cpp/build/bin/llama-server \
  -m /mnt/models/qwen3.6-35b-a3b-mtp-q4/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --mmproj /mnt/models/qwen3.6-35b-a3b-mtp-q4/mmproj-F16.gguf \
  --no-mmproj-offload \
  --image-min-tokens 1024 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 --n-cpu-moe 28 \
  -c 131072 \
  -ctk q8_0 -ctv q4_0 \
  -b 4096 -ub 2048 \
  --flash-attn on \
  --no-mmap \
  -np 1 \
  --jinja \
  --reasoning-preserve \
  --host 0.0.0.0 \
  --port 8080 \
  --metrics
Restart=on-failure
RestartSec=10
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

**VRAM:** ~11.7 GiB idle / ~11.8 peak at 128K (`-ub 2048` since 2026-08-28; ~430 MiB headroom) | **CPU RAM:** ~15–18 GiB of the LXC limit (host has 48 GB)

### BeeLlama Qwen3.6-27B (Historic — replaced 2026-08-15)

**Service:** `beellama-qwen3.6-27b.service`  
**Status:** ❌ Disabled, inactive — kept for rollback (unit + model files untouched)  
**GPU:** RTX 3090 24GB (CUDA0)  
**Context:** 160K, DFlash speculative decoding (BeeLlama.cpp b10102)  
**Cutover:** 2026-06-19 → replaced by `llama-qwen3.8-27b.service` 2026-08-15

```ini
[Unit]
Description=BeeLlama Qwen3.6-27B with DFlash (3090, 160K ctx)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/beellama.cpp
Environment=LD_LIBRARY_PATH=/opt/beellama.cpp/build/bin
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/opt/beellama.cpp/build/bin/llama-server \
  --device CUDA0 \
  -m /mnt/models/gguf/qwen3.6-27b/Qwen3.6-27B-Q5_K_S.gguf \
  --mmproj /mnt/models/gguf/qwen3.6-27b/mmproj-Qwen_Qwen3.6-27B-f16.gguf \
  --no-mmproj-offload \
  --spec-draft-model /mnt/models/gguf/qwen3.6-27b-dflash/Qwen3.6-27B-DFlash-Q4_K_M.gguf \
  --spec-type dflash \
  --spec-dflash-cross-ctx 1024 \
  -ngl all \
  --spec-draft-ngl all \
  --kv-unified \
  -np 1 \
  -b 2048 -ub 512 \
  --ctx-size 163840 \
  --cache-type-k q5_0 --cache-type-v q4_1 \
  --flash-attn on \
  --jinja \
  --mmap --mlock \
  --reasoning on \
  --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.6 --top-k 20 --top-p 1.0 --min-p 0.0 \
  --host 0.0.0.0 \
  --port 8080
Restart=on-failure
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

### Legacy Service (Rollback)

**Service:** `llama-server-qwen3.6-27b-longctx.service`  
**Status:** Enabled, inactive — kept for boot safety / rollback  
**Model:** Qwen3.6-27B-Q4_K_M (mainline llama.cpp, 204K context)

### Legacy Reference Unit

**Service:** `llama-server.service`  
**Status:** Disabled — old reference config (GLM-4.7-Flash), no longer used

```ini
[Unit]
Description=llama.cpp server with slot persistence
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/llama.cpp
ExecStart=/opt/llama.cpp/build/bin/llama-server \
  --model /mnt/models/gguf/glm-4.7-flash/GLM-4.7-Flash-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 131072 \
  --parallel 1 \
  --slot-save-path /mnt/models/cache/llama-cpp/slots \
  --reasoning-format deepseek \
  --reasoning-budget -1 \
  --flash-attn on \
  --jinja \
  --split-mode layer \
  --gpu-layers 99 \
  --cache-type-k q8_0 \
  --cache-type-v q4_0 \
  --metrics
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/llama-server.log
StandardError=append:/var/log/llama-server.log

[Install]
WantedBy=multi-user.target
```

---

## Key Flags

### Performance
- `--split-mode layer` — Split model across GPUs by layer (critical for dual 3060)
- `--gpu-layers 99` — Offload all layers to GPU
- `--flash-attn on` — Enable flash attention for speed
- `--cache-type-k q8_0` — K cache quantization (1 byte/element)
- `--cache-type-v q4_0` — V cache quantization (0.5 byte/element)

### Context & Slots
- `--ctx-size 131072` — 128K context window
- `--parallel 1` — Single slot (reduces memory pressure)
- `--slot-save-path /mnt/models/cache/llama-cpp/slots` — Persistent slot state

### Reasoning
- `--reasoning-format deepseek` — DeepSeek-style `<think>` tags
- `--reasoning-budget -1` — Unlimited reasoning tokens
- `--jinja` — Use Jinja2 chat templates from model

### Observability
- `--metrics` — **Enable Prometheus metrics at /metrics endpoint** ✅
  - Added 2026-02-24 (was missing before)
  - Exposes: `prompt_tokens_seconds` (pp tok/s), `predicted_tokens_seconds` (tg tok/s)

---

## Management Commands

```bash
# Check status
systemctl status llama-server

# Start/stop/restart
systemctl start llama-server
systemctl stop llama-server
systemctl restart llama-server

# Reload after config changes
systemctl daemon-reload && systemctl restart llama-server

# View logs
journalctl -u llama-server -f
# Or direct file
tail -f /var/log/llama-server.log

# Check health
curl http://localhost:8080/health

# Query metrics
curl http://localhost:8080/metrics
```

---

## Changing Models

1. Edit service file:
   ```bash
   nano /etc/systemd/system/llama-server.service
   ```

2. Update `--model` path

3. Reload and restart:
   ```bash
   systemctl daemon-reload && systemctl restart llama-server
   ```

4. Wait for model load (~30 sec), then check:
   ```bash
   curl http://localhost:8080/health
   ```

---

## Metrics Endpoint

**URL:** http://localhost:8080/metrics

**Key metrics:**
- `llamacpp:prompt_tokens_total` — Total prompt tokens processed
- `llamacpp:tokens_predicted_total` — Total generation tokens
- `llamacpp:prompt_seconds_total` — Cumulative prompt processing time
- `llamacpp:tokens_predicted_seconds_total` — Cumulative generation time
- `llamacpp:prompt_tokens_seconds` — **Current prompt processing speed (tok/s)**
- `llamacpp:predicted_tokens_seconds` — **Current generation speed (tok/s)**

**Example query:**
```bash
curl -s http://localhost:8080/metrics | grep tokens_seconds
# llamacpp:prompt_tokens_seconds 670.5
# llamacpp:predicted_tokens_seconds 45.2
```

---

## Model History

| Date | Service | Model | Notes |
|------|---------|-------|-------|
| 2026-08-15 | `llama-qwen3.8-27b` | Qwen3.8-27B-Q4_K_M + MTP | Production cutover, stock llama.cpp 5f754ea |
| 2026-06-19 | `beellama-qwen3.6-27b` | Qwen3.6-27B-Q5_K_S + DFlash | Production cutover, BeeLlama.cpp b10102 |
| 2026-04-23 | `llama-server-qwen3.6-27b-longctx` | Qwen3.6-27B-Q4_K_M | Mainline llama.cpp, 204K context (now rollback only) |
| 2026-02-24 | `llama-server` | GLM-4.7-Flash Q4_K_XL | Old reference config (disabled) |
| 2026-02-23 | — | ZwZ-4B Q6_K | Vision model test |
| 2026-02-19 | — | Nemotron-30B-A3B IQ4_NL | MoE baseline |
| 2026-02-19 | — | Qwen3-30B-A3B Q4_K_M | Dense 30B test |

---

## Troubleshooting

### Server won't start
```bash
# Check logs
tail -50 /var/log/llama-server.log

# Check GPU memory
nvidia-smi

# Verify model file exists
ls -lh /path/to/model/
```

### OOM / Slow performance
- Reduce `--ctx-size` (e.g., 65536)
- Check `--parallel` (lower = less memory)
- Verify `--split-mode layer` is set
- Monitor with `nvidia-smi` during inference

### Metrics not showing up
- Ensure `--metrics` flag is present in ExecStart
- Reload: `systemctl daemon-reload && systemctl restart llama-server`
- Check endpoint: `curl http://localhost:8080/metrics`

---

## Related Docs
- Model specs: `llmlab/models/*.md`
- Benchmarking: `llmlab/benchmarks/`
- SSH config: `notes/llama-cpp/ssh-and-cuda.md`
