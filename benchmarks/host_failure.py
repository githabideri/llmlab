"""host_failure.py — campaign-level HOST-FAILURE STOP GATE (backend-driven).

The distinction that matters (Sep 8 / Sep 10 history):
  * an OOM / request-level failure is a NORMAL experiment outcome — campaigns
    are designed to provoke and cleanly capture those;
  * a physical-host failure is NOT.

If the host hard-reboots, or shows MCE / NVIDIA Xid / kernel panic / PCIe-AER,
or becomes unreachable, the runner STOPS the whole campaign: preserve forensics,
restore production, and do NOT auto-resume. A human reviews the evidence.

Ported from mm-image-max (9/9 selftests) and re-wired over the Backend so the
same detector runs in qualification (injected probe text) and live (SSH).
Read-only by construction.
"""
import re

# signal label -> regex applied to the combined kernel-log body
PATTERNS = {
    "mce": r"mce:|Machine Check Exception|\bMCE: ?\d",
    "xid": r"Xid \(|\bNVRM: Xid|\bXid \d+",
    "pci_aer": r"\bAER\b|pcieport.*error|PCIe Bus Error|AER:.*(Corrected|Uncorrected)",
    "panic": r"Kernel panic|kernel BUG|kernel BUG at|\bOops:|segmentation fault in kernel",
}

PROBE = (
    "echo '===KERN==='; "
    "journalctl -k --no-pager 2>/dev/null | tail -4000; "
    "dmesg 2>/dev/null | tail -4000; "
    "echo '===VMCORE==='; "
    "ls -1 /var/crash 2>/dev/null; "
    "echo '===BTIME==='; "
    "awk '/btime/{print $2}' /proc/stat; "
    "echo '===UP==='; "
    "uptime -s 2>/dev/null"
)


def _search(pattern, body):
    return re.search(pattern, body, re.I) is not None


def _section(body, label):
    m = re.search(rf"==={label}===\n(.*?)(?:\n===|\Z)", body, re.S)
    return m.group(1).strip() if m else ""


class HostFailureDetector:
    def __init__(self, backend, t0_btime=None):
        self.backend = backend
        self.t0_btime = t0_btime

    def record_t0(self):
        """Capture the boot time once at window start (a mid-campaign change means
        the host rebooted — the 09-10 crash era)."""
        try:
            self.t0_btime = self.backend.host_btime()
        except Exception:
            self.t0_btime = None
        return self.t0_btime

    def check(self):
        """Read-only probe. Returns {detected, reasons[], evidence{}}. Never mutates."""
        res = {"detected": False, "reasons": [], "evidence": {}}
        try:
            rc, out, err = self.backend.host_probe()
        except Exception as e:
            res["detected"] = True
            res["reasons"].append("host-unreachable")
            res["evidence"]["error"] = f"{type(e).__name__}: {e}"
            return res
        kern = _section(out, "KERN") or out
        for label, pat in PATTERNS.items():
            if _search(pat, kern):
                res["detected"] = True
                res["reasons"].append(label)
        btime = _section(out, "BTIME").split("\n")[-1].strip()
        res["evidence"]["btime_now"] = btime
        if self.t0_btime is not None and btime.isdigit() and int(btime) != int(self.t0_btime):
            res["detected"] = True
            res["reasons"].append("host-rebooted")
        res["evidence"]["uptime_since"] = _section(out, "UP")
        return res

    def forensic_collect(self, cell_dir):
        """Gather the evidence a human needs. Best-effort; never takes the campaign down."""
        import os
        os.makedirs(cell_dir, exist_ok=True)
        try:
            rc, out, _ = self.backend.host_probe()
            with open(os.path.join(cell_dir, "host-forensics.txt"), "w") as f:
                f.write(out)
        except Exception:
            pass
