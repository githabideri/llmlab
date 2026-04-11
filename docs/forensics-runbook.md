# Crash/Freeze Forensics Runbook (Generic)

This runbook is intentionally generic (no machine- or container-specific identifiers).

## Goal
Collect enough evidence to distinguish between:
1) application/runtime crashes,
2) kernel/driver hangs,
3) host hard resets/power events.

---

## Configuration (local only, not committed)

Create a local file at `scripts/forensics/.env` from `scripts/forensics/.env.example`.

Example keys:
- `FORENSICS_TAG` (short label for this host)
- `FORENSICS_OUTDIR` (e.g. `/var/log/forensics`)
- `GPU_SAMPLER_UNIT`
- `FREEZE_TIMER_UNIT`
- `BOOT_SNAPSHOT_UNIT`
- `MODEL_SERVICE_UNIT`
- `MODEL_HEALTH_URL`

---

## Collectors

### 1) Boot snapshot collector
Capture on each boot:
- previous-boot kernel log
- warning-level logs
- filtered kernel signals (OOM, watchdog, GPU/Xid, PCIe/AER, lockups)
- `nvidia-smi -q` snapshot
- metadata (`last -x`, kernel version, uptime)

### 2) Freeze sampler (periodic, e.g. every 10s)
Capture:
- uptime/load/memory
- GPU summary
- key storage SMART lines
- recent dmesg tail

### 3) GPU live sampler (1s CSV)
CSV should include at minimum:
- timestamp
- GPU index/name
- power/temp/utilization
- memory used/total
- clocks/link info
- host loadavg

---

## Validation checklist (must pass)

```bash
systemctl is-enabled "$GPU_SAMPLER_UNIT" "$FREEZE_TIMER_UNIT" "$BOOT_SNAPSHOT_UNIT"
systemctl is-active  "$GPU_SAMPLER_UNIT" "$FREEZE_TIMER_UNIT"
```

```bash
LATEST=$(ls -1t "$FORENSICS_OUTDIR"/gpu-live-*.csv | head -n1)
head -n2 "$LATEST"
tail -n5 "$LATEST"
```

Expected: rows are updating with real values (not empty/static headers only).

---

## Kdump (kernel crash dumps)

Install and configure:
- `kdump-tools`, `kexec-tools`, `crash`, `makedumpfile`
- kernel cmdline includes `crashkernel=<size>`

Verify after reboot:

```bash
cat /proc/cmdline
kdump-config status
```

`kdump-config status` must report ready.

---

## Stress correlation workflow

1. Record test start/end epoch.
2. Run the stress shape under test.
3. Parse GPU live CSV within that time window.
4. Correlate with model service logs (`journalctl -u "$MODEL_SERVICE_UNIT"`).
5. Summarize:
   - max-used/min-free per GPU
   - max inter-GPU imbalance
   - exact failure signature (OOM / invalid config / ABRT / timeout / connection close)

---

## Interpretation guide

- **Only app/service crashes, host alive:** likely runtime/kernel-driver interaction or model config limits.
- **Both services/containers disappear at once + unclean shutdown markers:** host reset/hang/power path likely.
- **No explicit kernel panic text does not exclude power loss** (hard power events often leave minimal logs).

---

## Recommended next step sequence

1. Ensure all collectors validated as above.
2. Reboot once to activate kdump cmdline changes.
3. Confirm `kdump-config status` is ready.
4. Run one controlled repro.
5. If host-level event repeats, inspect crashdump artifacts first, then decide hardware vs software branch.
