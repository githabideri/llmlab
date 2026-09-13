#!/usr/bin/env python3
"""
mm-driver.py — request driver for the multi-image vision ceiling campaign.

Build now, selftest now. Execution is a separate, gated step (`--run`) that is
NOT exercised here: the selftest touches only a local mock on 127.0.0.1, never
the production endpoint and never a GPU.

Control paths (all selftestable without production):
  * fixture generation + SHA manifests (deterministic, varied aspect ratios)
  * image-count / visual-token closure (32x32 post-processor math)
  * two-tier concurrency: request_active_overlap_ms vs mm_execution_overlap_ms
  * recovery-plan sequencing (no cache contamination between cold reps)
  * outcome detection: healthy / reject / crash  (PID + systemd restart + /health)
  * mechanical boundary_distance (censored when no failure observed above)
  * Stage-B scoring: min-margin x repeatability; diagnostic_only guard
  * per-cell artifact schema (min_sampled_free, peak_corroborated)

VRAM terminology: external NVML numbers are "minimum sampled free VRAM"; a value
is only ever called a "peak" if an internal allocator metric corroborates it.

Usage:
  mm-driver.py --selftest
  mm-driver.py --make-fixtures <outdir> [--seed 1]
  mm-driver.py --run ...            # GATED — driven by runner.py in a window
"""
import argparse
import base64
import hashlib
import json
import os
import struct
import sys
import threading
import time
import urllib.request
import zlib

PATCH_MERGE = 32
PX_PER_TOKEN = PATCH_MERGE ** 2
MIN_OVERLAP_MS = 250


# --- fixtures ----------------------------------------------------------------
def _png(w, h, rgb):
    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + ch(b"IDAT", zlib.compress(raw))
            + ch(b"IEND", b""))


def expected_visual_tokens(w, h):
    return (w // PATCH_MERGE) * (h // PATCH_MERGE)


class Fixture:
    def __init__(self, name, w, h, data):
        self.name, self.w, self.h = name, w, h
        self.data = data
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.px = w * h
        self.vtok = expected_visual_tokens(w, h)

    def b64(self):
        return base64.b64encode(self.data).decode()


def make_fixtures(outdir, seed=1, specs=None):
    os.makedirs(outdir, exist_ok=True)
    if specs is None:
        specs = [("a_768x1024", 768, 1024), ("b_1024x1024", 1024, 1024),
                 ("c_1408x1472", 1408, 1472), ("d_1536x864", 1536, 864), ("e_864x1536", 864, 1536)]
    fixtures, manifest = [], []
    for name, w, h in specs:
        rgb = tuple((seed * 7 + c * 13 + i * (w + h)) % 256
                    for c, i in zip((0, 1, 2), (w, h, hash(name) % 512)))
        data = _png(w, h, rgb)
        p = os.path.join(outdir, name + ".png")
        with open(p, "wb") as f:
            f.write(data)
        fx = Fixture(name, w, h, data)
        fixtures.append(fx)
        manifest.append({"name": name, "sha256": fx.sha256, "w": w, "h": h,
                         "px": fx.px, "aspect": round(w / h, 4), "vtok": fx.vtok, "path": p})
    mpath = os.path.join(outdir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump({"seed": seed, "images": manifest}, f, indent=2)
    return fixtures, mpath


# --- closure -----------------------------------------------------------------
def closure(expected_vtok, usage):
    img = usage.get("image_tokens") or usage.get("prompt_tokens_details", {}).get("image_tokens")
    ok = img is not None and expected_vtok > 0 and abs(img - expected_vtok) / expected_vtok <= 0.02
    return {"expected_vtok": expected_vtok, "returned_image_tokens": img,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens"), "vtok_reconciles": bool(ok)}


# --- HTTP client -------------------------------------------------------------
def _read_sse(resp):
    first_mono, usage, comp, n = None, {}, 0, 0
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        try:
            obj = json.loads(body)
        except ValueError:
            continue
        if first_mono is None and obj.get("choices"):
            first_mono = time.monotonic_ns()
            n += 1
        if obj.get("usage"):
            usage = obj["usage"]
            comp = usage.get("completion_tokens", comp)
    return first_mono, usage, comp, n


def chat(base_url, model, content, max_tokens, stream=True, timeout=300):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "stream": stream}).encode()
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    start = time.monotonic_ns()
    r = {"start_mono": start, "first_token_mono": None, "end_mono": None,
         "ok": False, "status": None, "usage": {}, "completion_tokens": 0,
         "n_chunks": 0, "engine_active": False}
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        r["status"] = resp.status
        if stream:
            ft, usage, comp, n = _read_sse(resp)
            r.update(first_token_mono=ft, usage=usage, completion_tokens=comp, n_chunks=n)
        else:
            r["usage"] = json.loads(resp.read().decode()).get("usage", {})
            r["first_token_mono"] = time.monotonic_ns()
        r["ok"] = True
    except urllib.error.HTTPError as e:
        r["status"] = e.code
        try:
            r["error"] = e.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
    except Exception as e:
        r["error"] = repr(e)
    r["end_mono"] = time.monotonic_ns()
    return r


# --- concurrency: start barrier + two-tier overlap ---------------------------
def fire_concurrent(base_url, model, contents, release_at_mono):
    barrier = threading.Event()
    results = []
    lock = threading.Lock()

    def worker(i, content):
        time.sleep(max(0, release_at_mono - time.monotonic()))
        barrier.wait()
        r = chat(base_url, model, content, max_tokens=1, stream=True)
        r["idx"] = i
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker, args=(i, c)) for i, c in enumerate(contents)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(results, key=lambda r: r["start_mono"])


def active_window(r):
    end = r.get("first_token_mono") or r.get("end_mono")
    return (r["start_mono"], end) if end and end >= r["start_mono"] else None


def _max_overlap_ms(windows):
    best = 0
    for a in range(len(windows)):
        for b in range(a + 1, len(windows)):
            s1, e1 = windows[a]
            s2, e2 = windows[b]
            best = max(best, min(e1, e2) - max(s1, s2))
    return best / 1e6


def overlap_proof(results, min_overlap_ms=MIN_OVERLAP_MS):
    """Two-tier concurrency evidence.
      request_active_overlap_ms — intersection of [start, first_token]. Proves the
        requests were IN-FLIGHT together. Does NOT prove the expensive
        encoder/MM-prefill transients coexisted.
      mm_execution_overlap_ms   — intersection of encoder/MM-prefill phase windows,
        only when the server exposed encoder-phase markers AND both requests were
        engine-active (not just sitting in the HTTP/scheduler queue). None (=unknown)
        otherwise. 'submitted together' != 'MM activation lifetimes overlapping'."""
    active = [w for w in (active_window(r) for r in results) if w]
    ra = _max_overlap_ms(active) if len(active) >= 2 else 0.0
    ra_valid = ra >= min_overlap_ms

    enc = [(r["encoder_start"], r["encoder_end"]) for r in results
           if r.get("encoder_start") is not None and r.get("encoder_end") is not None]
    both_active = len(results) >= 2 and all(r.get("engine_active", False) for r in results)
    if len(enc) >= 2 and both_active:
        me = _max_overlap_ms(enc)
        me_valid = me >= min_overlap_ms
        detail = f"request-active {ra:.0f}ms; MM-execution {me:.0f}ms (both engine-active)"
    else:
        me, me_valid = None, None
        detail = (f"request-active {ra:.0f}ms; MM-execution UNKNOWN "
                  f"({'no encoder markers' if len(enc) < 2 else 'not both engine-active'})")
    return {"request_active_overlap_ms": round(ra, 1), "request_active_valid": bool(ra_valid),
            "mm_execution_overlap_ms": (round(me, 1) if me is not None else None),
            "mm_execution_valid": me_valid, "detail": detail}


def stage_c_conclusion(op):
    if not op["request_active_valid"]:
        return "invalid-conc: requests did not materially overlap; measurement discarded"
    if op["mm_execution_valid"] is True:
        return "CONCURRENT-SAFE: two vision activation lifetimes demonstrably overlapped"
    if op["mm_execution_valid"] is False:
        return "IN-FLIGHT ONLY: overlapped but MM transients did not demonstrably coexist — not a full pass"
    return "IN-FLIGHT (UNVERIFIED): overlapped; MM-execution overlap unknown — partial"


# --- recovery sequencing (avoid contaminating cold reps) ----------------------
def recovery_plan(rep_index, n_reps, outcome):
    """Healthy non-final reps: health + text only (a vision probe would mutate MM
    cache / allocator / fragmentation before the next cold rep). Final healthy
    rep: one dedicated small vision probe. Any reject/crash: full text+vision
    immediately (recovery is then what's being tested)."""
    if outcome in ("reject", "crash"):
        return ["health", "text", "vision"]
    if rep_index == n_reps - 1:
        return ["health", "text", "vision"]
    return ["health", "text"]


# --- outcome detection --------------------------------------------------------
def detect_outcome(res, sys_probe):
    before, after = sys_probe["before"], sys_probe["after"]
    pid_changed = before["pid"] != after["pid"]
    restart_changed = after["restart"] > before["restart"]
    health_bad = after.get("health") != 200
    status = res.get("status")
    if status is None:
        if pid_changed or restart_changed or health_bad:
            return "crash", "no response + engine change"
        return "reject", "no response, engine intact (transient)"
    if status == 200:
        if pid_changed or restart_changed or health_bad:
            return "crash", f"200 but engine changed (pid {pid_changed}, restart {restart_changed}, health {after.get('health')})"
        return "healthy", "200, engine intact"
    if 400 <= status < 500:
        if pid_changed or restart_changed or health_bad:
            return "crash", f"{status} with engine change"
        return "reject", f"clean {status} reject, engine intact"
    if pid_changed or restart_changed or health_bad:
        return "crash", f"{status} with engine change"
    return "reject", f"{status} transient, engine intact"


# --- boundary distance (mechanical, censored when unknown) --------------------
def boundary_distance(candidate_vtok, failing_vtoks):
    """Gap in actual visual tokens from this candidate to the nearest *tested
    failing* envelope. Censored (None) — never infinity — when no failure was
    observed at/above the candidate."""
    above = [v for v in failing_vtoks if v is not None and v >= candidate_vtok]
    return (min(above) - candidate_vtok) if above else None


# --- Stage-B candidate scoring ------------------------------------------------
def stage_b_score(cells, observed_failures):
    """cells: [{id, visual_tokens, config_key, margins_mib, diagnostic_only,
    all_hard_gate_pass?}]. observed_failures: [(config_key, visual_tokens)] for
    tested cells that crashed. Rank: promotable first (hard-gate pass on all reps
    AND not diagnostic_only), then by worst-of-N margin (stable beats spiky),
    then by spread. boundary_distance is computed mechanically per config_key and
    reported (censored when no failure above)."""
    out = []
    for c in cells:
        ms = c.get("margins_mib", [])
        all_pass = c.get("all_hard_gate_pass", (len(ms) > 0 and all(m > 0 for m in ms)))
        min_m = min(ms) if ms else -1
        spread = (max(ms) - min(ms)) if len(ms) >= 2 else 0
        bound = boundary_distance(c.get("visual_tokens", 0),
                                  [v for (k, v) in observed_failures if k == c.get("config_key")])
        promotable = bool(all_pass) and not c.get("diagnostic_only", False)
        out.append({"id": c["id"], "all_pass": bool(all_pass), "min_mib": min_m,
                    "spread_mib": spread, "boundary_mtok": bound,
                    "boundary_unknown": bound is None, "promotable": promotable,
                    "diagnostic_only": c.get("diagnostic_only", False)})
    out.sort(key=lambda x: (not x["promotable"], -x["min_mib"], x["spread_mib"], x["boundary_unknown"]))
    for i, o in enumerate(out):
        o["rank"] = i + 1
    return out


# --- phase markers -------------------------------------------------------------
def phase_markers(res, sys_probe):
    s = res["start_mono"]
    ft = res.get("first_token_mono") or res["end_mono"]
    e = res["end_mono"]
    mid = (s + ft) // 2
    return {"request_start": s, "encoder_start": res.get("encoder_start", s),
            "encoder_end": res.get("encoder_end", mid), "prefill_start": mid,
            "first_token": ft, "request_end": e,
            "engine_pid_before": sys_probe["before"]["pid"],
            "engine_pid_after": sys_probe["after"]["pid"]}


def markers_ordered(m):
    seq = [m["request_start"], m["encoder_start"], m["encoder_end"],
           m["prefill_start"], m["first_token"], m["request_end"]]
    return all(a <= b for a, b in zip(seq, seq[1:]))


# --- system probe (pluggable) -------------------------------------------------
class SysProbe:
    def __init__(self, base_url=None, mock=None):
        self.base_url, self.mock = base_url, mock

    def snapshot(self):
        if self.mock is not None:
            return self.mock()
        import subprocess
        pid = None
        try:
            out = subprocess.check_output(["pgrep", "-f", "vllm serve"], text=True).split()
            pid = int(out[0]) if out else None
        except Exception:
            pass
        try:
            restart = int(subprocess.check_output(
                ["systemctl", "show", "-p", "NRestarts", "--value", "vllm-dual"], text=True).strip())
        except Exception:
            restart = 0
        health = None
        if self.base_url:
            try:
                health = urllib.request.urlopen(self.base_url + "/health", timeout=5).status
            except Exception:
                health = 0
        return {"pid": pid, "restart": restart, "health": health}

    def before_after(self):
        return {"before": self.snapshot(), "after": self.snapshot()}


# --- artifact schema ----------------------------------------------------------
ARTIFACT_SCHEMA = {
    "cell": ["id", "stage", "gpu_util", "envelope", "flags", "diagnostic_only", "class", "config_key"],
    "baseline": ["kv_pool_tokens", "idle_free_mib", "engine_pid", "restart_count"],
    "reps": ["rep", "outcome", "requested_images", "processor_image_items", "visual_tokens",
             "completion_tokens", "ttft_ms", "min_sampled_free_mib", "peak_corroborated",
             "engine_pid_after", "restart_after", "health_after", "recovery", "recovery_probes"],
    "concurrency": ["requests", "request_active_overlap_ms", "request_active_valid",
                    "mm_execution_overlap_ms", "mm_execution_valid", "conclusion"],
    "mm_metrics": ["mm_cache_hits_delta", "prefix_cache_hits_delta", "per_turn"],
    "score": ["min_margin_mib", "spread_mib", "boundary_mtok", "boundary_unknown", "rank", "promotable"],
}


def write_artifact(outdir, artifact):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"cell-{artifact['cell']['id']}.json")
    with open(p, "w") as f:
        json.dump(artifact, f, indent=2)
    return p


# --- selftest ----------------------------------------------------------------
def selftest(mock_base=None):
    P = []

    def check(name, cond, detail=""):
        P.append((name, bool(cond), detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))

    print("== 1. fixtures (determinism, geometry, tokens) ==")
    import tempfile
    d = tempfile.mkdtemp()
    fx1, _ = make_fixtures(os.path.join(d, "r1"), seed=7)
    fx2, _ = make_fixtures(os.path.join(d, "r2"), seed=7)
    check("deterministic SHAs", all(a.sha256 == b.sha256 for a, b in zip(fx1, fx2)))
    check("varied aspect ratios (>=3)", len({f.h / f.w for f in fx1}) >= 3)
    check("vtok 768x1024=768", expected_visual_tokens(768, 1024) == 768)
    check("vtok 1448x1448=2025", expected_visual_tokens(1448, 1448) == 2025)

    print("== 2. closure ==")
    check("reconciles matching image tokens", closure(768, {"prompt_tokens": 8, "image_tokens": 768})["vtok_reconciles"])
    check("flags mismatched image tokens", not closure(768, {"prompt_tokens": 8, "image_tokens": 100})["vtok_reconciles"])

    print("== 3. two-tier overlap (request-active vs MM-execution) ==")
    def mk(start_ms, ft_ms, enc=None, active=True):
        s = start_ms * 1e6
        r = {"start_mono": s, "first_token_mono": s + ft_ms * 1e6, "end_mono": s + (ft_ms + 500) * 1e6,
             "engine_active": active}
        if enc:
            r["encoder_start"], r["encoder_end"] = s + enc[0] * 1e6, s + enc[1] * 1e6
        return r
    a, b = mk(0, 2000), mk(100, 2000)
    op = overlap_proof([a, b])
    check("request-active overlap valid (1900ms)", op["request_active_valid"] and op["request_active_overlap_ms"] > 1500, op["detail"])
    check("MM-execution UNKNOWN when no encoder markers", op["mm_execution_overlap_ms"] is None)
    check("conclusion = IN-FLIGHT (UNVERIFIED) without encoder evidence",
          stage_c_conclusion(op) == "IN-FLIGHT (UNVERIFIED): overlapped; MM-execution overlap unknown — partial", stage_c_conclusion(op))
    # both engine-active with overlapping ENCODE windows -> CONCURRENT-SAFE
    a2 = mk(0, 2000, enc=(0, 1500), active=True)
    b2 = mk(200, 2000, enc=(100, 1500), active=True)   # enc overlap ~1400ms
    op2 = overlap_proof([a2, b2])
    check("MM-execution overlap valid with encoder markers + both active",
          op2["mm_execution_valid"] is True and op2["mm_execution_overlap_ms"] > 800, op2["detail"])
    check("conclusion = CONCURRENT-SAFE when MM transients overlapped",
          stage_c_conclusion(op2) == "CONCURRENT-SAFE: two vision activation lifetimes demonstrably overlapped", stage_c_conclusion(op2))
    # engine-queued (not active) -> not a full pass
    a3 = mk(0, 2000, enc=(0, 1500), active=True)
    b3 = mk(200, 2000, enc=(100, 1500), active=False)
    op3 = overlap_proof([a3, b3])
    check("not CONCURRENT-SAFE when one request never became engine-active",
          stage_c_conclusion(op3) in ("IN-FLIGHT ONLY: overlapped but MM transients did not demonstrably coexist — not a full pass",
                                      "IN-FLIGHT (UNVERIFIED): overlapped; MM-execution overlap unknown — partial"), stage_c_conclusion(op3))
    c1, c2 = mk(0, 200), mk(900, 200)
    check("back-to-back -> invalid-conc", not overlap_proof([c1, c2])["request_active_valid"])

    print("== 4. recovery sequencing (no cold-rep contamination) ==")
    check("healthy mid-rep: health+text only (no vision)", recovery_plan(0, 3, "healthy") == ["health", "text"])
    check("healthy final rep: adds one vision probe", recovery_plan(2, 3, "healthy") == ["health", "text", "vision"])
    check("crash rep: full text+vision immediately", recovery_plan(0, 3, "crash") == ["health", "text", "vision"])
    check("reject rep: full recovery sequence", recovery_plan(1, 3, "reject") == ["health", "text", "vision"])

    print("== 5. outcome detection ==")
    def sp(pb, rb, pa, ra, ha):
        return {"before": {"pid": pb, "restart": rb}, "after": {"pid": pa, "restart": ra, "health": ha}}
    check("200+intact=healthy", detect_outcome({"status": 200}, sp(1, 3, 1, 3, 200))[0] == "healthy")
    check("200 but restarted=crash", detect_outcome({"status": 200}, sp(1, 3, 999, 4, 0))[0] == "crash")
    check("clean 400=reject", detect_outcome({"status": 400}, sp(1, 3, 1, 3, 200))[0] == "reject")
    check("no-response + engine change=crash", detect_outcome({"status": None}, sp(1, 3, 999, 4, 0))[0] == "crash")

    print("== 6. phase-marker ordering ==")
    m = phase_markers({"start_mono": 1000, "first_token_mono": 5000, "end_mono": 9000}, sp(1, 0, 1, 0, 200))
    check("monotonic phase markers", markers_ordered(m))

    print("== 7. mechanical boundary_distance (censored, not infinity) ==")
    check("failure above -> positive distance", boundary_distance(65536, [65536, 98304]) == 0
          and boundary_distance(32768, [65536]) == 32768)
    check("no failure above -> censored (None, not inf)", boundary_distance(98304, [65536]) is None)
    ranked = stage_b_score(
        [{"id": "S_stable", "visual_tokens": 65536, "config_key": "k1", "margins_mib": [700, 710, 695]},
         {"id": "S_spiky", "visual_tokens": 65536, "config_key": "k1", "margins_mib": [1100, 250, 260]},
         {"id": "S_diag", "visual_tokens": 98304, "config_key": "k1", "margins_mib": [900, 910, 905], "diagnostic_only": True},
         {"id": "S_top", "visual_tokens": 131072, "config_key": "k1", "margins_mib": [800, 810, 795]}],
        observed_failures=[("k1", 98304)])
    order = [r["id"] for r in ranked]
    check("stable ranks above spiky (min-margin key)", order.index("S_stable") < order.index("S_spiky"), f"{order}")
    check("diagnostic_only never promotable", not [r for r in ranked if r["id"] == "S_diag"][0]["promotable"])
    check("boundary reported mechanically (S_stable dist to 98304 = 32768)",
          [r for r in ranked if r["id"] == "S_stable"][0]["boundary_mtok"] == 32768)
    check("censored boundary flagged when no failure above (S_top > 98304)",
          [r for r in ranked if r["id"] == "S_top"][0]["boundary_unknown"] is True)

    print("== 8. post-request recovery probes (mock) ==")
    if mock_base:
        r = chat(mock_base, "m", [{"type": "text", "text": "hi"}], max_tokens=1, stream=True)
        check("tiny text completion vs mock", r["ok"] and r["status"] == 200, f"{r['status']} {r.get('error','')}")

    print("== 9. artifact schema (round-trip, min_sampled_free not peak) ==")
    art = {"cell": {"id": "A3", "stage": "A", "gpu_util": 0.90, "envelope": {"count": 64, "total_mpx": 67.1},
                    "flags": {}, "diagnostic_only": False, "class": "T", "config_key": "k1"},
           "baseline": {"kv_pool_tokens": 677931, "idle_free_mib": [502, 499], "engine_pid": 100, "restart_count": 3},
           "reps": [{"rep": 0, "outcome": "healthy", "requested_images": 64, "processor_image_items": None,
                     "visual_tokens": 65536, "completion_tokens": 1, "ttft_ms": 4200.0,
                     "min_sampled_free_mib": [40, 42], "peak_corroborated": False,
                     "engine_pid_after": 100, "restart_after": 3, "health_after": 200,
                     "recovery": {"text_ok": True, "vision_ok": True}, "recovery_probes": ["health", "text"]}],
           "concurrency": {"requests": [], "request_active_overlap_ms": 0, "request_active_valid": False,
                           "mm_execution_overlap_ms": None, "mm_execution_valid": None, "conclusion": "n/a"},
           "mm_metrics": {"mm_cache_hits_delta": 0, "prefix_cache_hits_delta": 0, "per_turn": []},
           "score": {"min_margin_mib": 40, "spread_mib": 2, "boundary_mtok": None, "boundary_unknown": True,
                     "rank": 2, "promotable": True}}
    p = write_artifact(d, art)
    rl = json.load(open(p))
    missing = [k for k in ARTIFACT_SCHEMA if k not in rl]
    check("all top-level schema keys present", not missing, f"missing={missing}")
    check("processor_image_items is nullable (unknown, not synthesized)",
          rl["reps"][0]["processor_image_items"] is None)
    check("min_sampled_free + peak_corroborated False (no false 'peak')",
          rl["reps"][0]["min_sampled_free_mib"] == [40, 42] and rl["reps"][0]["peak_corroborated"] is False)

    print(f"\n== SELFTEST SUMMARY: {sum(1 for _, c, _ in P if c)}/{len(P)} passed ==")
    failed = [n for n, c, _ in P if not c]
    if failed:
        print("FAILED:", failed)
    return not failed


# --- mock server (selftest only) ---------------------------------------------
def _start_mock(port):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json", sse=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype if not sse else "text/event-stream")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, b"ok", "text/plain")
            elif self.path == "/v1/models":
                self._send(200, json.dumps({"data": [{"id": "m"}]}).encode())
            else:
                self._send(404, b"{}")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/v1/chat/completions":
                usage = {"prompt_tokens": 8000, "completion_tokens": 1,
                         "prompt_tokens_details": {"image_tokens": 768}}
                if body.get("stream"):
                    chunks = [{"choices": [{"delta": {"content": "x"}}]},
                              {"choices": [{"delta": {}}], "usage": usage}]
                    self._send(200, ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode(), sse=True)
                else:
                    self._send(200, json.dumps({"usage": usage, "choices": [{"message": {"content": "x"}}]}).encode())
            else:
                self._send(404, b"{}")

    srv = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--make-fixtures", default=None, metavar="OUTDIR")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run", nargs="*", default=None)
    a = ap.parse_args()
    if a.selftest:
        base, _ = _start_mock(0)
        sys.exit(0 if selftest(base) else 1)
    elif a.make_fixtures:
        _, mp = make_fixtures(a.make_fixtures, seed=a.seed)
        print("fixtures + manifest:", mp)
    elif a.run:
        print("GATED: --run is driven by runner.py in an approved window. Refusing inline.", file=sys.stderr)
        sys.exit(2)
    else:
        ap.print_help()
