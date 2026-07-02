# llama.cpp / BeeLlama.cpp systemd Service Configuration

**Services:** `beellama-qwen3.6-27b.service` (production), `llama-server-qwen3.6-vision.service` (vision), plus legacy units  
**Logs:** `journalctl -u <service>`

---

## Production Services

### BeeLlama Qwen3.6-27B (Port 8080, RTX 3090)

**Service:** `beellama-qwen3.6-27b.service`  
**Unit:** `/etc/systemd/system/beellama-qwen3.6-27b.service`  
**Status:** ✅ Active, enabled  
**Cutover:** 2026-06-19 (replaced mainline llama.cpp Q4_K_M)

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
  --no-mmap --mlock \
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

### Qwen3.6-35B-A3B Vision (Port 8081, Dual RTX 3060)

**Service:** `llama-server-qwen3.6-vision.service`  
**Status:** ✅ Active, enabled  
**Model:** Qwen3.6-35B-A3B-UD-IQ4_XS + Vision (mainline llama.cpp)

### Qwen3.6-35B-A3B-MTP (Port 8080, RTX 3060, llama-backup)

**Host:** llama-backup (LXC, private homelab)  
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
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 --n-cpu-moe 28 \
  -c 131072 \
  -ctk q8_0 -ctv q4_0 \
  -b 4096 -ub 1536 \
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

**VRAM:** ~9.9 / 12.3 GiB | **CPU RAM:** ~15-18 / 20 GiB

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
