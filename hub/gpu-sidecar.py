#!/usr/bin/env python3
"""llm-hub GPU sidecar: nvidia-smi -> JSON over HTTP.

Stdlib only. One instance per GPU host (LXC, VM, or bare metal). The llm-hub
poller fetches this to get GPU utilization — llama.cpp and vLLM both publish
token metrics but NOT GPU utilization. One nvidia-smi call per poll (2 s).

Deploy: systemd unit gpu-sidecar.service (see gpu-sidecar.service in this dir).
Trust model: no auth, same trust domain as the keyless inference servers
(LAN/tailnet only). Port 9421.
"""
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("GPU_SIDECAR_PORT", "9421"))
# optional index filter, via CLI ("--only 2") or env (GPU_SIDECAR_INDICES=2,3):
# in a container that only sees passed-through GPUs, nvidia-smi still lists
# every host GPU; pass the index(es) that are actually yours.
FILTER = {int(x) for x in os.environ.get("GPU_SIDECAR_INDICES", "").split(",") if x.strip()}
args = sys.argv[1:]
if args and args[0] == "--only":
    FILTER = {int(x) for part in args[1:] for x in part.split(",") if x.strip()}
QUERY = (
    "index,name,utilization.gpu,memory.used,memory.total,"
    "temperature.gpu,power.draw"
)


def _f(x):
    """float or None — Pascal/older drivers report [N/A] for some fields (power.draw)."""
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def sample():
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in out.stdout.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 7:
                continue
            idx = int(float(p[0]))
            if FILTER and idx not in FILTER:
                continue
            gpus.append({
                "index": idx,
                "name": p[1],
                "util_pct": _f(p[2]),
                "mem_used_mib": _f(p[3]),
                "mem_total_mib": _f(p[4]),
                "temp_c": _f(p[5]),
                "power_w": _f(p[6]),
            })
        # honest failure reporting: a non-zero exit (driver fault, XID) or an
        # empty GPU list must be ok:false — the hub keeps the last GOOD sample
        # and flips gpus_stale in the UI after 30 s, instead of showing a
        # silent card with no GPU rows.
        if out.returncode != 0:
            err = (out.stderr or out.stdout).strip()[:200] or f"nvidia-smi rc={out.returncode}"
            return {"ok": False, "error": err, "ts": time.time(), "gpus": []}
        if not gpus:
            return {"ok": False, "error": "no GPUs reported (driver fault or wrong --only filter)",
                    "ts": time.time(), "gpus": []}
        return {"ok": True, "host": os.uname().nodename, "ts": time.time(), "gpus": gpus}
    except Exception as e:
        return {"ok": False, "error": str(e), "ts": time.time(), "gpus": []}


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        d = sample()
        if self.path in ("/", "/gpus"):
            self._json(d)
        elif self.path == "/health":
            self._json({"ok": d["ok"], "ts": time.time()})
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
