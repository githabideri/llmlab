# wgpx15 crash/freeze forensics runbook (2026-03-24)

## Scope
This runbook captures the actual state and tooling used to investigate:
- sudden host reboots (both CT327 + CT329 down together)
- CT329 llama-server aborts/freezes under 4-way long-context stress

Host: `wgpx15` (Proxmox)
CTs: `327` (Qwen3.5-27B), `329` (Qwen3.5-35B-A3B)

---

## 1) Evidence observed so far

### Host-level crash indicators
- `last -x` shows multiple `crash` terminations (not clean shutdown entries).
- Boot journals include: `system.journal ... uncleanly shut down`.
- No clean shutdown/reboot sequence immediately before those boot boundaries.

Interpretation: at least some events are host hard resets/hangs (not only app crash).

### CT329 app-level failure indicators
Under 4 concurrent long-prefill stress:
- requests die with `Remote end closed connection without response`
- service logs show recurrent:
  - `CUDA error: invalid configuration argument`
  - `Main process exited ... status=6/ABRT`

Interpretation: CT329 model server/driver path can fail independently of full host reboot.

---

## 2) Forensic collectors enabled on wgpx15

### A) Boot snapshot collector
- Unit: `wgpx15-crash-snapshot.service` (enabled)
- Script: `/usr/local/sbin/wgpx15-crash-snapshot.sh`
- Output dir: `/var/log/wgpx15-forensics`
- Captures at boot:
  - prev boot kernel log + filtered kernel lines
  - prev/this boot warning logs
  - `nvidia-smi` snapshots
  - metadata + `last -x`

### B) Freeze telemetry sampler (10s)
- Timer: `wgpx15-freeze-sample.timer` (enabled)
- Service: `wgpx15-freeze-sample.service`
- Script: `/usr/local/sbin/wgpx15-freeze-sample.sh`
- Output dir: `/var/log/wgpx15-freeze`
- Captures:
  - uptime/load/memory
  - nvidia-smi summary
  - nvme SMART key lines
  - dmesg tail

### C) GPU live sampler (1s)
- Unit: `wgpx15-gpu-sampler.service` (enabled)
- Script: `/usr/local/sbin/wgpx15-gpu-sampler.sh`
- Output dir: `/var/log/wgpx15-forensics/gpu-live-*.csv`
- CSV fields:
  `ts_utc,gpu_index,name,pstate,power_draw_w,power_limit_w,temp_c,gpu_util_pct,mem_util_pct,mem_used_mb,mem_total_mb,sm_clock_mhz,mem_clock_mhz,pcie_gen,pcie_width,loadavg_1m`

---

## 3) Stress test with numeric VRAM balance (post-fix sampler)

Window: `START_EPOCH=1774359719`, `END_EPOCH=1774359872`
Shape: CT329, 4 parallel, long-prefill prompt

Result:
- all 4 requests failed at ~152s with remote-close
- CT329 journal showed `CUDA error: invalid configuration argument` + ABRT

Sampler-derived VRAM numbers (during failure window):
- GPU1 (3060): max used **11229 MB**, min free **1059 MB**, avg used **10797 MB**
- GPU2 (3060): max used **10669 MB**, min free **1619 MB**, avg used **10254 MB**
- Peak imbalance: **562 MB**

Conclusion from numbers:
- imbalance is real (~0.56 GB)
- but failures still happen with >1 GB free on tighter GPU, so imbalance alone is not sufficient cause
- likely a combined concurrency/context + flash-attn/CUDA path issue

---

## 4) Kdump setup status (for kernel crash dumps)

Installed:
- `kdump-tools`, `kexec-tools`, `crash`, `makedumpfile`

Configured:
- `/etc/default/grub` now includes `crashkernel=768M`
- `kdump-tools.service` enabled

Current status:
- **NOT ready until next host reboot** (current `/proc/cmdline` still lacks `crashkernel=`)
- after reboot, verify with `kdump-config status`

---

## 5) Quick operator checklist

### Verify collectors
```bash
systemctl status wgpx15-crash-snapshot.service
systemctl status wgpx15-freeze-sample.timer
systemctl status wgpx15-gpu-sampler.service
```

### Verify outputs
```bash
ls -lah /var/log/wgpx15-forensics | tail
ls -lah /var/log/wgpx15-freeze | tail
```

### Verify kdump readiness (after reboot)
```bash
cat /proc/cmdline
kdump-config status
```

### Correlate a stress run with sampler window
1. save start/end epoch around test
2. parse `gpu-live-*.csv` for that interval
3. compare with `journalctl -u llama-server.service`

---

## 6) Next recommended step

After confirming a maintenance window, reboot wgpx15 once to activate `crashkernel=768M`, then run one controlled repro to capture whether host-level failures yield kdump artifacts.
