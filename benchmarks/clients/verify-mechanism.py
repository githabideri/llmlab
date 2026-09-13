#!/usr/bin/env python3
"""verify-mechanism.py — S0-style mechanism validator client (stdlib only).

The campaign asks: did the mechanism we intend to measure actually turn on?
A throughput number taken on a server that silently fell back to another
placement/mechanism is not data. This client proves, from the server's own
log plus a live probe, that:
  1. the server is alive and serving;
  2. every positive pattern the spec expects is present (tensor split init,
     cache allocation line, per-device placement, ...);
  3. no negative pattern (silent fallback / init failure) is present;
  4. a minimal completion actually completes (mechanism not just announced).

The patterns are SPEC data (--expect label=regex, --forbid label=regex), so
the same client validates any mechanism; the platform's client contract
(script + --json-out) absorbs it unchanged. Exit 0 iff valid, and
--json-out always writes the evidence either way.
"""
import argparse
import json
import re
import urllib.request


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--url", required=True)
    a.add_argument("--server-log", required=True)
    a.add_argument("--expect", action="append", default=[],
                   help="label=regex that MUST appear in the server log")
    a.add_argument("--forbid", action="append", default=[],
                   help="label=regex that MUST NOT appear (silent fallback)")
    a.add_argument("--prompt", default="Say ok")
    a.add_argument("--n-predict", type=int, default=8)
    a.add_argument("--json-out", required=True)
    a = a.parse_args()

    checks, valid = {}, True

    def parse(kind):
        out = []
        for e in a.__dict__[kind]:
            label, _, rx = e.partition("=")
            out.append((label, rx))
        return out

    try:
        log = open(a.server_log, errors="replace").read()
    except OSError as e:
        checks["server_log_readable"] = False
        valid = False
        log = ""

    for label, rx in parse("expect"):
        m = re.search(rx, log, re.M | re.I)
        checks[f"expect:{label}"] = bool(m)
        if not m:
            valid = False
    for label, rx in parse("forbid"):
        m = re.search(rx, log, re.M | re.I)
        checks[f"forbid:{label}"] = m.group(0)[:120] if m else None
        if m:
            valid = False

    # numeric evidence from the canonical cache-allocation line, if any
    cm = re.search(r"(\d+)\s+layers\s*x\s*(\d+)\s+slots[^,]*,\s*([\d.]+)\s*M.iB\s+uploaded",
                   log, re.I)
    cache = None
    if cm:
        cache = {"layers": int(cm.group(1)), "slots": int(cm.group(2)),
                 "mib_uploaded": float(cm.group(3))}

    # live probe: the mechanism must serve, not merely announce
    comp = None
    try:
        req = urllib.request.Request(
            a.url.rstrip("/") + "/completion",
            data=json.dumps({"prompt": a.prompt,
                             "n_predict": a.n_predict}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode(errors="replace")
            comp = {"status": r.status, "bytes": len(body),
                    "ok": r.status == 200}
    except Exception as e:
        comp = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    checks["live_completion"] = bool(comp and comp.get("ok"))
    if not (comp and comp.get("ok")):
        valid = False

    json.dump({"valid": valid, "checks": checks, "cache": cache,
               "completion": comp}, open(a.json_out, "w"), indent=2)
    print(json.dumps({"valid": valid, "checks": checks, "cache": cache}))
    raise SystemExit(0 if valid else 2)


if __name__ == "__main__":
    main()
