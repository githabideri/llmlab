# llama.cpp systemd Service Configuration

**Service:** `llama-server.service`  
**Logs:** `/var/log/llama-server.log`

---

## Service File Location

`/etc/systemd/system/llama-server.service`

## Current Configuration

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

| Date | Model | Notes |
|------|-------|-------|
| 2026-02-24 | GLM-4.7-Flash Q4_K_XL | Current (MLA, 4.7B active) |
| 2026-02-23 | ZwZ-4B Q6_K | Vision model test |
| 2026-02-19 | Nemotron-30B-A3B IQ4_NL | MoE baseline |
| 2026-02-19 | Qwen3-30B-A3B Q4_K_M | Dense 30B test |

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
