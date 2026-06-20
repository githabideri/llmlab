# Backend/Model Cutover Checklist

Reusable checklist for swapping models or runtimes (llama.cpp → BeeLlama, model A → model B, quant change, etc.).

**Principle:** Old service stays enabled until new is verified. Rollback is one command.

---

## Pre-Flight

- [ ] **Read upstream docs** — fetch official config/quickstart before writing your own
- [ ] **Pre-flight checks on GPU server:**
  ```bash
  # What's running
  systemctl list-units --type=service --state=running | grep -iE 'llama|beellama'
  # Ports in use
  ss -tlnp | grep -E '808[019]|581'
  # GPU state
  nvidia-smi
  # Disk space (for new models)
  df -h /mnt/models
  ```
- [ ] **New model files present** — verify paths, sizes, checksums if downloaded
- [ ] **Binary ready** — build resolved, no missing deps (`ldd` check if shared lib)
- [ ] **Config validated against upstream** — compare flags side-by-side with official quickstart/docs

## Launch (Manual, Not Enabled)

- [ ] **Stop old service** (keep **enabled**, just stop):
  ```bash
  systemctl stop <old-service>
  # Verify process actually died
  ps aux | grep llama-server | grep -v grep
  ```
- [ ] **Launch new server manually** (no `systemd enable` yet):
  ```bash
  /path/to/llama-server <all flags> &
  ```
- [ ] **Validate launch:**
  ```bash
  curl -s http://localhost:8080/health
  # Check logs for warnings — especially GDN warnings, CUDA OOM, layer fallback
  ```
- [ ] **Key flags present** (Qwen3.5/3.6 specific):
  - [ ] `-ngl all` (required — without it, fused GDN falls back to CPU)
  - [ ] `--kv-unified`
  - [ ] `-ub 512` (not 128 — 4× prefill difference)
  - [ ] `--reasoning on` (richer drafter context for DFlash)

## Benchmark

- [ ] **Decode speed** — run at least one structured output test (JSON/code)
- [ ] **Prefill speed** — run with a known prompt size (e.g., 20K tokens)
- [ ] **VRAM usage** — `nvidia-smi` shows expected budget, no OOM
- [ ] **DFlash active** (if applicable) — check logs for `draft_n`, `draft_n_accepted`
- [ ] **Results match expectations** — compare against upstream benchmarks or old baseline

## Systemd Service

- [ ] **Write service unit** to `/etc/systemd/system/<name>.service`
- [ ] **Key fields present:**
  - [ ] `Restart=on-failure`
  - [ ] `RestartSec=5`
  - [ ] `LimitNOFILE=65536` (BeeLlama)
  - [ ] `Environment=CUDA_VISIBLE_DEVICES=N` (isolate GPU)
- [ ] **Test via systemd:**
  ```bash
  # Kill manual process, start via systemd
  systemctl start <new-service>
  systemctl status <new-service>
  curl -s http://localhost:8080/health
  ```
- [ ] **Enable new service:**
  ```bash
  systemctl enable <new-service>
  ```
- [ ] **Old service state:**
  - [ ] Old service still **enabled** (for boot safety / rollback)
  - [ ] Old service **inactive** (not running)

## Client Config Audit

> This is the step that was missed on the BeeLlama cutover. Every config that references the changed port/model needs updating.

- [ ] **Identify all configs that point to the changed port:**
  - [ ] `~/.pi/agent/models.json` (pi CLI + pi-web — shared)
  - [ ] `~/.pi/agent/settings.json` (defaultModel)
  - [ ] `~/.pi-matrix-agent/agent/models.json` (Matrix bot — separate)
  - [ ] `/etc/llama-swap/config.yaml` (model router on GPU server)
  - [ ] Any other services/scripts that hit `192.168.0.27:8080` or `:8081`
- [ ] **Update fields that changed:**
  - [ ] Model ID (quant name)
  - [ ] Model display name
  - [ ] Context window
  - [ ] Input types (text vs text+image)
  - [ ] Service name references (in llama-swap metadata)
- [ ] **Verify with a live request:**
  ```bash
  curl -s http://192.168.0.27:8080/v1/chat/completions \
    -H "Authorization: Bearer <key>" \
    -d '{"model":"<new-model-id>","messages":[{"role":"user","content":"hi"}],"max_tokens":10}'
  # Check: model in response matches new model, fingerprint is correct
  ```

## Documentation

- [ ] **Update model card** (`llmlab/models/<model>.md`) — add changelog entry
- [ ] **Update host doc** (`infra/<location>/<host>.md`) — service table, GPU workload
- [ ] **Update systemd doc** (`llmlab/docs/llama-cpp-systemd.md`) — add new service, update model history
- [ ] **Write experiment log** (`llmlab/experiments/YYYY-MM-DD-<name>.md`) — what was tried, results
- [ ] **Write report** (`reports/YYYY-MM-DD_report_<name>.md`) — if operation was significant
- [ ] **Check for stale references** — grep for old model names / service names / port assignments

## Rollback Plan

```bash
# If anything goes wrong:
systemctl stop <new-service>
systemctl start <old-service>
curl -s http://localhost:8080/health  # should be ok
```

---

## Quick Reference: What Changed on Each Cutover

| Date | What | Configs Updated |
|------|------|-----------------|
| 2026-06-19 | BeeLlama DFlash cutover | `~/.pi/agent/models.json`, `~/.pi/agent/settings.json`, `~/.pi-matrix-agent/agent/models.json`, `/etc/llama-swap/config.yaml` |

> Add rows here after each cutover so the next person knows what to check.
