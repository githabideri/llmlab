#!/usr/bin/env python3
"""llm-hub — inference fleet control & observability. Single file, stdlib only.

Polls every configured llama.cpp / vLLM server every 2 s:
  * llama.cpp router: GET /v1/models (catalog + load state),
                      GET /metrics?model=<id> per loaded model
  * vLLM:             GET /metrics
  * GPUs:             per-machine nvidia-smi sidecar (see gpu-sidecar.py)

Serves:
  * GET  /api/servers       fleet JSON for agents/automation (auth)
  * GET  /api/models        flat model list across servers (auth)
  * POST /api/models/load   {server, model} — proxies to the router (auth)
  * POST /api/models/unload {server, model} — proxies to the router (auth)
  * GET  /api/vision/route  advisor: best vision endpoint for an upcoming
                           multimodal request — read-only, zero mutation (auth)
  * POST /auth              {token} or Bearer header -> 7-day session cookie
  * GET  /metrics           Prometheus text (public)
  * GET  /health            liveness + poller freshness (public)
  * GET  /                  UI (unauthenticated shell; data requires auth)

State is in-memory: a 30-min sparkline ring and 5-min rolling windows per
server, plus an optional periodic snapshot to STATE_DIR/snapshot.json.

The hub is NEVER in the inference request path — clients talk directly to the
inference servers; the hub only watches them. Losing the hub loses visibility,
not inference.

Auth: master Bearer token (TOKEN file, chmod 600) for agents/CLI; the browser
exchanges it once via POST /auth for an HMAC-signed HttpOnly session cookie
(7 days). A missing token file fails boot (LLM_HUB_ALLOW_NO_AUTH=1 for a
deliberate unauthenticated dev deploy). /metrics and /health are public
(topology-level data).

Env overrides (all optional, for testing):
  LLM_HUB_CONFIG        config path      (default /etc/llm-hub/config.json)
  LLM_HUB_TOKEN         token file       (default /etc/llm-hub/token)
  LLM_HUB_STATE         state dir        (default /var/lib/llm-hub)
  LLM_HUB_UI            ui dir           (default /opt/llm-hub/ui)
  LLM_HUB_LAT_WINDOW    percentile window seconds (default 300)
  LLM_HUB_COOKIE_SECURE add `Secure` to the session cookie (1/true; default off
                        so plain-HTTP local testing works — set it behind HTTPS)
  LLM_HUB_ALLOW_NO_AUTH fail open on a missing token file (1/true) — dev
                        escape hatch; the default is fail-closed at boot
"""
import collections
import hashlib
import hmac
import http.client
import json
import math
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import vision_routing  # pure policy module, same directory (underscore: importable)

CONFIG_PATH = os.environ.get("LLM_HUB_CONFIG", "/etc/llm-hub/config.json")
TOKEN_PATH = os.environ.get("LLM_HUB_TOKEN", "/etc/llm-hub/token")
STATE_DIR = os.environ.get("LLM_HUB_STATE", "/var/lib/llm-hub")
UI_DIR = os.environ.get("LLM_HUB_UI", "/opt/llm-hub/ui")

POLL_INTERVAL = 2
RING_SECONDS = 30 * 60              # sparkline ring retention
SNAPSHOT_EVERY = 60                 # seconds between state snapshots
OFFLINE_AFTER = 15                  # consecutive failed polls (30 s) before "offline"
HTTP_TIMEOUT = 4                    # per upstream fetch
SIDECAR_TIMEOUT = 7                 # sidecar runs nvidia-smi (~1 s), allow margin
RATE_DECAY = 30                     # measured rates decay to idle after this
LAT_WINDOW = int(os.environ.get("LLM_HUB_LAT_WINDOW", "300"))
WIN_SAMPLES = max(2, LAT_WINDOW // POLL_INTERVAL)   # window ring buffer size
COOKIE_SECURE = os.environ.get("LLM_HUB_COOKIE_SECURE", "").lower() in ("1", "true", "yes")

# vLLM latency histograms (emitted per engine/model; buckets incl. +Inf).
VLLM_HISTS = (
    ("ttft",  "time_to_first_token_seconds"),
    ("tpot",  "inter_token_latency_seconds"),
    ("e2e",   "e2e_request_latency_seconds"),
    ("queue", "request_queue_time_seconds"),
)

CONFIG = {}
TOKEN = ""
NO_AUTH = False
last_tick = None                    # poller heartbeat (API-visible)

# Vision advisor: decisions served via GET /api/vision/route (counter only —
# the decision itself is recomputed on demand, stateless).
VISION_LOCK = threading.Lock()
VISION_ROUTES_TOTAL = collections.defaultdict(int)   # (server, model) -> picks


# ---------------------------------------------------------------------------
# HTTP / Prometheus text helpers
# ---------------------------------------------------------------------------

def _http(method, url, timeout, obj=None):
    """GET/POST via http.client. Returns (status, body_text); (0, None) on
    fetch failure; non-2xx return their status + body.

    (Not urllib: it forces `Connection: close`, and uvicorn-backed servers
    — vLLM / llama.cpp router — stall sending the body on that, so the 4 s
    poll-timeout fires on a response that curl gets in 10 ms.)"""
    p = urllib.parse.urlsplit(url)
    host = p.hostname
    port = p.port or (443 if p.scheme == "https" else 80)
    path = p.path or "/"
    if p.query:
        path += "?" + p.query
    conn = (http.client.HTTPSConnection(host, port, timeout=timeout)
            if p.scheme == "https" else
            http.client.HTTPConnection(host, port, timeout=timeout))
    try:
        headers = {"Accept-Encoding": "identity"}
        data = None
        if obj is not None:
            data = json.dumps(obj).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=data, headers=headers)
        r = conn.getresponse()
        return r.status, r.read().decode("utf-8", "replace")
    except Exception:
        return 0, None
    finally:
        conn.close()


def http_json(url, timeout=HTTP_TIMEOUT):
    return _http("GET", url, timeout)


def http_post_json(url, obj, timeout=HTTP_TIMEOUT):
    return _http("POST", url, timeout, obj)


def parse_prom(text):
    """One scalar per metric name: *_total -> sum across label sets,
    everything else -> max (gauge semantics)."""
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        try:
            head, _, val = line.rpartition(" ")
            name, _, _ = head.partition("{")
            v = float(val)
        except (ValueError, IndexError):
            continue
        if name.endswith("_total"):
            out[name] = out.get(name, 0.0) + v
        else:
            out[name] = max(out.get(name, 0.0), v)
    return out


def parse_buckets(text):
    """{hist_base_name: {le: cumulative_count}} for every *_bucket series,
    summed across the other labels (engine, model_name)."""
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        head, _, lab = line.partition("{")
        if not head.endswith("_bucket"):
            continue
        m = re.search(r'le="([^"]+)"', lab)
        parts = line.split()
        if not m or len(parts) < 2:
            continue
        try:
            le = float(m.group(1))
            v = float(parts[1])
        except ValueError:
            continue
        d = out.setdefault(head[: -len("_bucket")], {})
        d[le] = d.get(le, 0.0) + v
    return out


def parse_labeled(text, name, label):
    """{label_value: value} for one metric's one label, summed across all
    other label series. (Gauge semantics: last sample wins per value.)"""
    out = {}
    prefix = name + "{"
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        m = re.search(re.escape(label) + r'="([^"]*)"', line)
        parts = line.split()
        if not m or len(parts) < 2:
            continue
        try:
            v = float(parts[1])
        except ValueError:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
    return out


def hist_quantile(buckets, q):
    """Linear-interpolated quantile from a CUMULATIVE delta histogram
    {le: count}. Returns seconds, or None when there were no observations."""
    finite = sorted((le, c) for le, c in buckets.items() if math.isfinite(le))
    total = buckets.get(float("inf"), finite[-1][1] if finite else 0.0)
    if total <= 0:
        return None
    target = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in finite:
        if c >= target:
            span = c - prev_c
            if span <= 0:
                return le
            return prev_le + (le - prev_le) * (target - prev_c) / span
        prev_le, prev_c = le, c
    return finite[-1][0] if finite else None


def _counter_delta(base, cur):
    """Delta of one monotonic counter over a window. None on missing/reset."""
    if base is None or cur is None or cur < base - 1e-9:
        return None
    return cur - base


def _bucket_deltas(base, cur):
    """Delta histogram {le: d} for a rolling window; None on counter reset or
    a changed bucket set (engine restart, vLLM upgrade)."""
    if base is None or cur is None or set(base) != set(cur):
        return None
    out = {}
    for le in cur:
        d = cur[le] - base[le]
        if d < -1e-9:
            return None
        out[le] = d
    return out


def get_metric(prom, name, prefixes=("llamacpp:", "llamacpp_")):
    """Counter/gauge lookup by bare name, trying known metric prefixes
    (llama.cpp ships both `llamacpp:` and `llamacpp_`; vLLM: `vllm:`)."""
    for p in prefixes:
        if p + name in prom:
            return prom[p + name]
    return None


def _ctx_from_args(entry):
    """--ctx-size N from a router /v1/models status.args list (or None)."""
    args = (entry.get("status") or {}).get("args") or []
    for i, a in enumerate(args):
        if a == "--ctx-size" and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                return None
    return None


def _parallel_from_args(entry):
    """--parallel N from a router /v1/models status.args list (or None).
    Slot capacity the vision advisor's IMMEDIATE/QUEUED split needs."""
    args = (entry.get("status") or {}).get("args") or []
    for i, a in enumerate(args):
        if a == "--parallel" and i + 1 < len(args):
            try:
                return int(args[i + 1])
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class Server:
    def __init__(self, cfg):
        self.name = cfg["name"]
        self.kind = cfg["kind"]            # llama-router | vllm | gpu-only
        self.url = cfg.get("url")          # gpu-only servers have none
        self.desc = cfg.get("description", "")
        self.model_hints = cfg.get("models", {})   # per-model ctx/desc hints
        self.sidecar_url = cfg.get("sidecar")     # per-server override
        self.models = {}                   # model_id -> state dict
        self.gpus = []
        self.gpus_ts = None
        self.gpus_stale = False
        self.online = False
        self.miss_streak = 0
        self.last_seen = None
        self.last_err = None
        self._lock = threading.Lock()      # guards .models (poller writes, API reads)
        self._counters = {}                # (mid, key) -> rate state
        self._spec_ts = {}                 # mid -> last spec-activity ts
        # rolling 5-min windows: deque of (ts, buckets, counters)
        self._winbuf = collections.deque(maxlen=WIN_SAMPLES)
        # llama.cpp per-model window: mid -> deque of (ts, {cached, new})
        self._wins = {}
        # sparkline ring: (t, tgen_sum, gpu_max, ttft_p95_ms|None)
        self.ring = collections.deque(maxlen=RING_SECONDS // POLL_INTERVAL)

    # -- sparkline ----------------------------------------------------------
    def ring_push(self, tgen, gpu, ttft_p95_ms):
        now = time.time()
        self.ring.append((now, tgen, gpu, ttft_p95_ms))
        while self.ring and now - self.ring[0][0] > RING_SECONDS:
            self.ring.popleft()

    # -- rate computation (ported from v1 — child-clock, idle decay) --------
    def _rate(self, mid, key, value, now, secs=None, gauge=None):
        """t/s from counter deltas, honest by construction:
        - llama.cpp gen: tokens / generation-seconds (child's own
          tokens_predicted_seconds clock) -> physically bounded, immune to
          poll gaps and counter-restart artifacts (vLLM: wall clock, its
          counters are process-lifetime cumulative)
        - optional gauge (llama.cpp prompt_tokens_seconds, a per-prefill
          average) wins over wall division when a prefill just happened
        - a counter *decrease* (child restart / model reload) re-baselines
          instead of spiking
        - a measured rate shows for at most RATE_DECAY s after the counter
          last moved, then decays to None (idle)"""
        if value is None:
            return None
        prev = self._counters.get((mid, key))
        rate = change = None
        if prev is not None:
            if value < prev["v"]:
                rate, change = None, None            # restart: clean re-baseline
            elif value > prev["v"]:
                dv = value - prev["v"]
                if secs is not None and prev.get("s") is not None:
                    ds = secs - prev["s"]
                    if ds > 0:
                        rate = round(dv / ds, 1)
                if rate is None and gauge is not None and gauge > 0:
                    rate = round(gauge, 1)
                if rate is None:
                    dt = now - prev["t"]
                    if dt > 0:
                        rate = round(dv / dt, 1)
                change = now
        pr = prev.get("rate") if prev else None
        pc = prev.get("change") if prev else None
        self._counters[(mid, key)] = {
            "v": value, "t": now, "s": secs,
            "rate": rate if rate is not None else pr,
            "change": change if change is not None else pc,
        }
        e = self._counters[(mid, key)]
        if e["rate"] is None or e["change"] is None:
            return None
        return e["rate"] if now - e["change"] <= RATE_DECAY else None

    # -- polling ------------------------------------------------------------
    def poll(self, now):
        if self.kind == "gpu-only":
            return                        # no API; online state comes from the sidecar
        try:
            if self.kind == "vllm":
                self._poll_vllm(now)
            else:
                self._poll_router(now)
        except Exception as e:
            self.miss_streak += 1
            self.last_err = f"{self.name}: {e}"[:200]
            if self.miss_streak >= OFFLINE_AFTER:
                self.online = False
        else:
            self.miss_streak = 0
            self.online = True
            self.last_seen = now
            self.last_err = None

    def _poll_vllm(self, now):
        status, body = http_json(f"{self.url}/metrics")
        if status != 200 or not body:
            raise RuntimeError(f"/metrics HTTP {status or 'timeout'}")
        p = parse_prom(body)
        gen = get_metric(p, "generation_tokens_total", prefixes=("vllm:", "vllm_"))
        prompt = get_metric(p, "prompt_tokens_total", prefixes=("vllm:", "vllm_"))
        req_running = int(get_metric(p, "num_requests_running",
                                     prefixes=("vllm:", "vllm_")) or 0)
        req_waiting = int(get_metric(p, "num_requests_waiting",
                                     prefixes=("vllm:", "vllm_")) or 0)
        kv = get_metric(p, "kv_cache_usage_perc", prefixes=("vllm:", "vllm_"))
        if kv is not None and kv > 1.0:       # some builds expose 0..100
            kv /= 100.0
        kv = round(kv, 3) if kv is not None else None

        # ---- rolling 5-min window: snapshot, then compute deltas ----------
        buckets = parse_buckets(body)
        finish = parse_labeled(body, "vllm:request_success_total", "finished_reason")
        sleep = parse_labeled(body, "vllm:engine_sleep_state", "sleep_state")
        wait_reason = parse_labeled(body, "vllm:num_requests_waiting_by_reason", "reason")
        counters = {("f:" + r): v for r, v in finish.items()}
        counters["ph"] = get_metric(p, "prefix_cache_hits_total",
                                    prefixes=("vllm:", "vllm_")) or 0.0
        counters["pq"] = get_metric(p, "prefix_cache_queries_total",
                                    prefixes=("vllm:", "vllm_")) or 0.0
        counters["pre"] = get_metric(p, "num_preemptions_total",
                                     prefixes=("vllm:", "vllm_")) or 0.0
        self._winbuf.append((now, buckets, counters))

        latency = finish_win = cache_hit = preempt_win = None
        if len(self._winbuf) >= 2:
            ts0, b0, c0 = self._winbuf[0]
            win = max(1, round(now - ts0))
            latency = {"window_s": win}
            for key, base in VLLM_HISTS:
                # histogram series carry the vllm: prefix; resolve once
                full = next((n for n in (base, "vllm:" + base,
                                          "vllm_" + base) if n in b0 or n in buckets), base)
                d = _bucket_deltas(b0.get(full), buckets.get(full))
                latency[key] = None if d is None else {
                    "p50": hist_quantile(d, 0.50),
                    "p95": hist_quantile(d, 0.95),
                    "p99": hist_quantile(d, 0.99),
                }
            fd = {k: _counter_delta(c0.get(k), v)
                  for k, v in counters.items() if k.startswith("f:")}
            if fd and not any(v is None for v in fd.values()):
                total = int(sum(fd.values()))
                if total > 0:
                    stop = int(fd.get("f:stop") or 0)
                    length = int(fd.get("f:length") or 0)
                    abort = int(fd.get("f:abort") or 0)
                    finish_win = {
                        "window_s": win, "total": total, "stop": stop,
                        "length": length, "abort": abort,
                        "other": max(0, total - stop - length - abort),
                    }
            dh = _counter_delta(c0.get("ph"), counters["ph"])
            dq = _counter_delta(c0.get("pq"), counters["pq"])
            if dh is not None and dq is not None and dq > 0:
                cache_hit = round(dh / dq, 3)
            pd = _counter_delta(c0.get("pre"), counters["pre"])
            preempt_win = int(pd) if pd is not None else None

        sleep_state = next((k for k, v in sleep.items() if v >= 1), None)
        wr = None
        if req_waiting > 0 and any(v > 0 for v in wait_reason.values()):
            wr = max(wait_reason.items(), key=lambda kv: kv[1])[0]

        with self._lock:
            st = self.models.setdefault("(vllm)", {
                "id": "(vllm)", "kind": "vllm", "desc": self.desc,
                "loaded": True,
            })
            st.update({
                "tgen": self._rate("(vllm)", "gen", gen, now),
                "tpp": self._rate("(vllm)", "prompt", prompt, now),
                "spec_accept": None,
                "req_running": req_running, "req_waiting": req_waiting,
                "kv_used": kv,
                "latency": latency, "finish": finish_win,
                "cache_hit": cache_hit, "preempt_win": preempt_win,
                "preempt_total": int(get_metric(p, "num_preemptions_total",
                                                prefixes=("vllm:", "vllm_")) or 0),
                "sleep": sleep_state, "wait_reason": wr,
                "metrics_ok": gen is not None,
            })

    def _poll_router(self, now):
        # fetch first (network, no lock), apply under the lock afterwards so
        # handler threads reading api() are never blocked on upstream I/O
        status, body = http_json(f"{self.url}/v1/models")
        if status != 200 or not body:
            raise RuntimeError(f"/v1/models HTTP {status or 'timeout'}")
        try:
            listing = json.loads(body).get("data", [])
        except json.JSONDecodeError:
            raise RuntimeError("/v1/models bad json")
        fetched = []
        for m in listing:
            mid = m.get("id")
            if not mid:
                continue
            loaded = m.get("status", {}).get("value") == "loaded"
            p = None
            if loaded:
                _, mt = http_json(
                    f"{self.url}/metrics?model={urllib.request.quote(mid)}")
                p = parse_prom(mt) if mt else None
            fetched.append((mid, loaded, p, m))
        with self._lock:
            for mid, loaded, p, entry in fetched:
                hint = self.model_hints.get(mid, {})
                ctx = _ctx_from_args(entry) or hint.get("ctx")
                # catalog metadata the vision advisor needs: input modalities
                # (newer llama.cpp builds report [text,image] for mmproj models)
                # and slot capacity (--parallel in the launch args).
                arch = entry.get("architecture") or {}
                mods = arch.get("input_modalities")
                par = _parallel_from_args(entry)
                st = self.models.setdefault(mid, {
                    "id": mid, "kind": "llama.cpp", "ctx": ctx,
                    "desc": hint.get("desc", ""),
                })
                st["loaded"] = loaded
                if ctx:
                    st["ctx"] = ctx
                st["input_modalities"] = list(mods) if isinstance(mods, list) else None
                st["parallelism"] = par
                st["desc"] = hint.get("desc") or st.get("desc") or ""
                if not loaded or p is None:
                    st.update({"tgen": None, "tpp": None, "spec_accept": None,
                               "cache_hit": None, "n_proc": None,
                               "n_deferred": None, "busy_slots": None,
                               "metrics_ok": False})
                    continue
                # parser-compat probe: the two counters every hub rate
                # depends on. False = backend reachable but counter names broke
                st["metrics_ok"] = (
                    get_metric(p, "tokens_predicted_total") is not None
                    and get_metric(p, "prompt_tokens_total") is not None)
                st["tgen"] = self._rate(
                    mid, "gen", get_metric(p, "tokens_predicted_total"), now,
                    secs=get_metric(p, "tokens_predicted_seconds_total"))
                st["tpp"] = self._rate(
                    mid, "prompt", get_metric(p, "prompt_tokens_total"), now,
                    gauge=get_metric(p, "prompt_tokens_seconds"))
                a = get_metric(p, "spec_decode_num_accepted_tokens_total")
                d = get_metric(p, "spec_decode_num_draft_tokens_total")
                if a is not None and d is not None:
                    prev = self._counters.get((mid, "spec"))
                    if prev is None:
                        st["spec_accept"] = None
                    elif a < prev[0] or d < prev[1]:
                        st["spec_accept"] = None                 # child restart
                    elif a > prev[0] and d > prev[1]:
                        st["spec_accept"] = round((a - prev[0]) / (d - prev[1]), 3)
                        self._spec_ts[mid] = now
                    elif now - self._spec_ts.get(mid, 0) > RATE_DECAY:
                        st["spec_accept"] = None                 # idle decay
                    self._counters[(mid, "spec")] = (a, d, now)
                else:
                    st["spec_accept"] = None
                # rolling 5-min prompt-cache hit rate (cached vs non-cached
                # tokens; prompt_tokens_total EXCLUDES cached by definition)
                cache_win = None
                win = self._wins.setdefault(
                    mid, collections.deque(maxlen=WIN_SAMPLES))
                win.append((now, {
                    "cached": get_metric(p, "prompt_tokens_cached_total") or 0.0,
                    "new": get_metric(p, "prompt_tokens_total") or 0.0,
                }))
                if len(win) >= 2:
                    s0, s1 = win[0][1], win[-1][1]
                    dc = _counter_delta(s0["cached"], s1["cached"])
                    dn = _counter_delta(s0["new"], s1["new"])
                    if dc is not None and dn is not None:
                        tot = dc + dn
                        if tot > 0:
                            cache_win = round(dc / tot, 3)
                st.update({
                    "cache_hit": cache_win,
                    "n_proc": int(get_metric(p, "requests_processing") or 0),
                    "n_deferred": int(get_metric(p, "requests_deferred") or 0),
                    "busy_slots": get_metric(p, "n_busy_slots_per_decode"),
                })

    # -- gpu sidecar ---------------------------------------------------------
    def poll_gpus(self, now):
        if not self.sidecar_url:
            return
        status, body = http_json(self.sidecar_url, timeout=SIDECAR_TIMEOUT)
        data = None
        if status == 200:
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = None
        if data is not None and data.get("ok"):
            self.gpus = data.get("gpus", [])
            self.gpus_ts = now
            self.gpus_stale = False
            self.last_gpu_err = None
            if self.kind == "gpu-only":
                self.online = True
                self.miss_streak = 0
                self.last_seen = now
            return
        err = (f"sidecar HTTP {status or 'timeout'}" if data is None
               else data.get("error", "sidecar ok:false"))
        if self.kind == "gpu-only":
            self.miss_streak += 1
            if self.miss_streak >= OFFLINE_AFTER:
                self.online = False
        else:
            # keep last GOOD sample; flag stale after 30 s (nvidia-smi may be
            # wedged — the hub must not show a silent card with no GPU rows)
            if self.gpus and now - (self.gpus_ts or 0) > 30:
                self.gpus_stale = True
        self.last_gpu_err = err

    # -- api -----------------------------------------------------------------
    def api(self, now):
        with self._lock:
            models = []
            for st in self.models.values():
                m = dict(st)
                m["stale"] = not self.online
                models.append(m)
        spark = [[int(t), g, u, l] for (t, g, u, l) in self.ring]
        return {
            "name": self.name, "kind": self.kind, "url": self.url,
            "description": self.desc, "online": self.online,
            "last_seen": self.last_seen, "last_err": self.last_err,
            "gpus": self.gpus, "gpus_stale": self.gpus_stale,
            "models": models, "spark": spark,
        }


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _master_token_ok(auth_header):
    if NO_AUTH:
        return True
    return bool(TOKEN) and hmac.compare_digest(
        auth_header or "", "Bearer " + TOKEN)


SESSION_TTL = 7 * 86400          # 7 days, fixed from issuance


def _session_ok(cookie_header):
    """Validate an HMAC-signed session cookie (exp.sig)."""
    if not cookie_header or not TOKEN:
        return False
    for part in cookie_header.split(";"):
        k, _, v = part.strip().partition("=")
        if k != "hub_session" or not v:
            continue
        exp, _, sig = v.partition(".")
        if not sig:
            return False
        expect = hmac.new(TOKEN.encode(), exp.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, sig):
            return False
        try:
            return int(exp) > time.time()
        except ValueError:
            return False
    return False


def _session_cookie():
    exp = str(int(time.time() + SESSION_TTL))
    sig = hmac.new(TOKEN.encode(), exp.encode(), hashlib.sha256).hexdigest()
    c = f"hub_session={exp}.{sig}; Path=/; HttpOnly; SameSite=Lax; Max-Age={SESSION_TTL}"
    if COOKIE_SECURE:
        c += "; Secure"    # behind an HTTPS proxy; off by default for plain-HTTP local use
    return c


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Vision advisor (read-only; the hub never mutates the backends for it)
# ---------------------------------------------------------------------------

def _vision_candidates(now):
    """Normalized vision-route candidates from the current server state.
    Read-only: touches nothing, polls nothing. The route decision itself
    lives in vision_routing.route_vision (pure, unit-tested)."""
    cands = []
    for s in SERVERS:
        if s.kind == "gpu-only":
            continue
        age = None if s.last_seen is None else now - s.last_seen
        utils = [g.get("util_pct") for g in (s.gpus or [])
                 if isinstance(g.get("util_pct"), (int, float))]
        for m in s.api(now)["models"]:
            vcfg = (s.model_hints.get(m["id"]) or {}).get("vision") or {}
            cands.append(vision_routing.normalize_candidate({
                "server": s.name, "model": m["id"], "url": s.url,
                "kind": m.get("kind"), "online": s.online,
                "state_age_s": age, "loaded": m.get("loaded"),
                "input_modalities": m.get("input_modalities"),
                "vision_override": vcfg.get("image_capable"),
                "enabled": vcfg.get("enabled", True),
                "tier": vcfg.get("tier"),
                "parallelism": m.get("parallelism"),
                "busy_slots": m.get("busy_slots"),
                "n_proc": m.get("n_proc"), "n_deferred": m.get("n_deferred"),
                "gpu_util_max": max(utils) if utils else None,
                "gpus_stale": s.gpus_stale,
                "tpp": m.get("tpp"),
            }))
    return cands


def _vision_policy():
    p = dict(vision_routing.DEFAULT_POLICY)
    p.update(CONFIG.get("vision") or {})
    return p


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llm-hub/1.3"

    def _send(self, code, body, ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra:
            for k, v in extra:
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        return _master_token_ok(self.headers.get("Authorization")) or \
               _session_ok(self.headers.get("Cookie"))

    def log_message(self, fmt, *args):
        pass   # quiet; the audit log below covers control actions

    def _audit(self, what, detail, code):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            with open(os.path.join(STATE_DIR, "audit.log"), "a") as f:
                f.write(json.dumps({
                    "ts": time.time(), "what": what, "detail": detail,
                    "code": code, "ip": self.client_address[0],
                }) + "\n")
        except OSError:
            pass

    def _vision_route(self):
        """GET /api/vision/route — advisor (auth). Read-only: answers from
        asynchronously polled state in well under a second; it never loads
        or unloads anything and never proxies the request. The caller sends
        the OpenAI-compatible multimodal request directly to choice.url and
        keeps its own static emergency fallback for when the hub is down
        (documented in README.md). Query params: images, max_width,
        max_height (accepted; Phase 1 scoring is shape-independent)."""
        q = urllib.parse.parse_qs(
            self.path.split("?", 1)[1] if "?" in self.path else "")

        def _i(key, default=None):
            v = (q.get(key) or [None])[0]
            try:
                return int(v) if v is not None else default
            except ValueError:
                return default

        request = {"images": max(0, _i("images", 0) or 0),
                   "max_width": _i("max_width"),
                   "max_height": _i("max_height")}
        try:
            res = vision_routing.route_vision(_vision_candidates(time.time()),
                                             request, _vision_policy())
        except ValueError as e:
            self._audit("vision.route", {"error": str(e)}, 500)
            return self._send(500, {"error": f"vision policy misconfigured: {e}"})
        if res["choice"]:
            with VISION_LOCK:
                VISION_ROUTES_TOTAL[(res["choice"]["server"],
                                     res["choice"]["model"])] += 1
        self._audit("vision.route",
                    {"ok": res["ok"],
                     "choice": (res["choice"]["server"] + "/" + res["choice"]["model"]
                                if res["choice"] else None),
                     "reason_codes": res["reason_codes"]}, 200)
        self._send(200, res)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._ui("index.html", "text/html; charset=utf-8")
        elif path.startswith("/ui/"):
            self._ui(path[4:], "application/octet-stream")
        elif path == "/api/servers":
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            self._send(200, {"ts": time.time(), "poll_tick": last_tick,
                             "servers": [s.api(time.time()) for s in SERVERS]})
        elif path == "/api/models":
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            out = []
            for s in SERVERS:
                for m in s.api(time.time())["models"]:
                    row = dict(m)
                    row["server"] = s.name
                    row["server_online"] = s.online
                    out.append(row)
            self._send(200, {"ts": time.time(), "models": out})
        elif path == "/api/vision/route":
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            self._vision_route()
        elif path == "/auth":
            self._send(200 if self._authed() else 401, {"ok": self._authed()})
        elif path == "/metrics":
            self._send(200, prom_text(), "text/plain; version=0.0.4")
        elif path == "/health":
            now = time.time()
            age = None if last_tick is None else now - last_tick
            self._send(200, {"ok": age is not None and age < 30,
                             "poll_tick": last_tick,
                             "poll_age_s": round(age, 1) if age is not None else None,
                             "ts": now, "servers": len(SERVERS)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode() if length else "{}"
        except (ValueError, OSError):
            body = "{}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}

        if path == "/auth":
            tok = self.headers.get("Authorization", "")[len("Bearer "):] \
                if (self.headers.get("Authorization") or "").startswith("Bearer ") \
                else payload.get("token", "")
            if (TOKEN and hmac.compare_digest(tok, TOKEN)) or NO_AUTH:
                return self._send(200, {"ok": True},
                                  extra=[("Set-Cookie", _session_cookie())])
            return self._send(401, {"error": "bad token"})

        if path in ("/api/models/load", "/api/models/unload"):
            if not self._authed():
                return self._send(401, {"error": "unauthorized"})
            server_name = payload.get("server")
            model = payload.get("model")
            s = next((x for x in SERVERS if x.name == server_name), None)
            if s is None:
                return self._send(404, {"error": f"unknown server {server_name!r}"})
            if s.url is None:   # gpu-only: no inference API to proxy to
                return self._send(400, {"error": f"server {server_name!r} has no inference API"})
            if not model:
                return self._send(400, {"error": "missing model"})
            endpoint = "load" if path.endswith("/load") else "unload"
            # the router parses "model" for BOTH load and unload
            status, resp = http_post_json(
                f"{s.url}/models/{endpoint}", {"model": model})
            self._audit(f"model {endpoint}", f"{server_name}/{model}", status)
            return self._send(status if status else 502,
                              {"upstream_status": status, "response": resp[:500]})
        self._send(404, {"error": "not found"})

    def _ui(self, rel, ctype):
        root = os.path.realpath(UI_DIR)
        fp = os.path.realpath(os.path.join(root, rel))
        if not (fp == root or fp.startswith(root + os.sep)):
            return self._send(400, {"error": "bad path"}, "text/plain")
        if not os.path.isfile(fp):
            return self._send(404, {"error": "not found"})
        with open(fp, "rb") as f:
            data = f.read()
        extra = [] if rel == "index.html" else [("Cache-Control", "max-age=300")]
        self._send(200, data, ctype, extra)


# ---------------------------------------------------------------------------
# Prometheus export
# ---------------------------------------------------------------------------

def _f(v):
    """Prometheus float formatting; None -> line omitted (see _g)."""
    return None if v is None else repr(float(v))


def _g(L, name, lab, v):
    """Append a gauge line; unknown (None) values are omitted rather than NaN
    — a NaN would poison avg()/sum() over the whole series in PromQL, while an
    absent sample shows as a gap, which is what "no data" means. A measured
    zero is still exported as 0."""
    f = _f(v)
    if f is not None:
        L.append(f"{name}{lab} {f}")


def prom_text():
    L = []
    L.append("# --- llm-hub fleet ---")
    L.append("hub_servers_total " + str(len(SERVERS)))
    for s in SERVERS:
        lab = f'{{server="{s.name}",kind="{s.kind}"}}'
        L.append(f"hub_server_online{lab} {1.0 if s.online else 0.0}")
        if s.gpus:
            L.append(f'hub_server_gpu_count{{server="{s.name}"}} {len(s.gpus)}')
            L.append(f'hub_server_gpus_stale{{server="{s.name}"}} '
                     f"{1.0 if s.gpus_stale else 0.0}")
        for g in s.gpus:
            gl = f'{{server="{s.name}",gpu="{g.get("name", "")[:40]}"}}'
            _g(L, "hub_gpu_utilization_pct", gl, g.get('util_pct'))
            _g(L, "hub_gpu_memory_used_mib", gl, g.get('mem_used_mib'))
            _g(L, "hub_gpu_memory_total_mib", gl, g.get('mem_total_mib'))
            _g(L, "hub_gpu_temp_c", gl, g.get('temp_c'))
            _g(L, "hub_gpu_power_w", gl, g.get('power_w'))
        for m in s.models.values():
            ml = f'{{server="{s.name}",model="{m["id"]}"}}'
            L.append(f"hub_model_loaded{ml} {1.0 if m.get('loaded') else 0.0}")
            _g(L, "hub_model_tokens_per_second", ml, m.get('tgen'))
            _g(L, "hub_model_prompt_tokens_per_second", ml, m.get('tpp'))
            if m.get("kind") == "vllm":
                if m.get("loaded"):
                    L.append(f"hub_vllm_metrics_ok{ml} "
                             f"{1.0 if m.get('metrics_ok') else 0.0}")
                _g(L, "hub_model_requests_running", ml, m.get('req_running'))
                _g(L, "hub_model_requests_waiting", ml, m.get('req_waiting'))
                _g(L, "hub_model_kv_cache_used", ml, m.get('kv_used'))
                lat = m.get("latency") or {}
                for key in ("ttft", "tpot", "e2e", "queue"):
                    q = lat.get(key) or {}
                    for qn in ("p50", "p95", "p99"):
                        _g(L, f"hub_model_{key}_{qn}_seconds", ml, q.get(qn))
                _g(L, "hub_model_preemptions_total", ml, m.get('preempt_total'))
                _g(L, "hub_model_preemptions_window", ml, m.get('preempt_win'))
                fin = m.get("finish") or {}
                _g(L, "hub_model_finish_total_window", ml, fin.get('total'))
                _g(L, "hub_model_finish_length_window", ml, fin.get('length'))
                _g(L, "hub_model_finish_abort_window", ml, fin.get('abort'))
                _g(L, "hub_model_prefix_cache_hit", ml, m.get('cache_hit'))
                if m.get("sleep"):
                    L.append(f"hub_model_engine_asleep{ml} "
                             f"{0.0 if m.get('sleep') == 'awake' else 1.0}")
            else:
                if m.get("loaded"):
                    L.append(f"hub_llama_metrics_ok{ml} "
                             f"{1.0 if m.get('metrics_ok') else 0.0}")
                _g(L, "hub_model_requests_processing", ml, m.get('n_proc'))
                _g(L, "hub_model_requests_deferred", ml, m.get('n_deferred'))
                _g(L, "hub_model_busy_slots", ml, m.get('busy_slots'))
                _g(L, "hub_model_prompt_cache_hit", ml, m.get('cache_hit'))
            _g(L, "hub_model_spec_acceptance", ml, m.get('spec_accept'))
    # --- vision advisor: route recomputed from current state at scrape time ---
    now = time.time()
    try:
        cands = _vision_candidates(now)
        vres = vision_routing.route_vision(cands, {}, _vision_policy())
    except ValueError:
        cands, vres = None, None
    if vres is None:
        L.append("hub_vision_route_available 0.0")
    else:
        L.append(f"hub_vision_route_available {1.0 if vres['ok'] else 0.0}")
        rej = {r["name"]: r["code"] for r in vres["rejected"]}
        for c in cands:
            lab = f'{{server="{c["server"]}",model="{c["model"]}"}}'
            code = rej.get(c["server"] + "/" + c["model"])
            L.append(f"hub_vision_candidate_eligible{lab} "
                     f"{0.0 if code else 1.0}")
            if code:
                L.append(f'hub_vision_candidate_reject_code'
                         f'{{server="{c["server"]}",model="{c["model"]}",'
                         f'code="{code}"}} 1.0')
            if vres["choice"] and vres["choice"]["server"] == c["server"] \
                    and vres["choice"]["model"] == c["model"]:
                L.append(f"hub_vision_candidate_selected{lab} 1.0")
                L.append(f"hub_vision_choice_score{lab} {vres['choice']['score']}")
        for s in SERVERS:
            if s.last_seen is not None:
                L.append(f'hub_vision_state_age_seconds{{server="{s.name}"}} '
                         f"{now - s.last_seen:.1f}")
    with VISION_LOCK:
        for (sv, md), n in sorted(VISION_ROUTES_TOTAL.items()):
            L.append(f'hub_vision_routes_total'
                     f'{{server="{sv}",model="{md}"}} {n}')
    L.append("# generated " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------

def snapshot():
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        now = time.time()
        data = {"ts": now, "servers": {}}
        for s in SERVERS:
            a = s.api(now)
            data["servers"][s.name] = {k: a[k] for k in
                                       ("name", "online", "last_seen", "gpus", "models")}
        tmp = os.path.join(STATE_DIR, "snapshot.json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(STATE_DIR, "snapshot.json"))
    except OSError:
        pass


def restore():
    try:
        with open(os.path.join(STATE_DIR, "snapshot.json")) as f:
            data = json.load(f)
        entries = data.get("servers")
        if isinstance(entries, list):            # v1 snapshot format
            entries = {e.get("name"): e for e in entries if isinstance(e, dict)}
        for name, sd in (entries or {}).items():
            s = next((x for x in SERVERS if x.name == name), None)
            if not s:
                continue
            s.online = sd.get("online", False)
            s.last_seen = sd.get("last_seen")
            s.gpus = sd.get("gpus", [])
            s.gpus_ts = data.get("ts") if sd.get("gpus") else None
            for m in sd.get("models", []):
                s.models[m["id"]] = m
    except (OSError, json.JSONDecodeError):
        pass


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

SERVERS = []


def poller():
    global last_tick
    while True:
        t0 = time.time()
        now = t0
        for s in SERVERS:
            try:
                s.poll(now)
                s.poll_gpus(now)
            except Exception as e:
                s.last_err = f"poller exception: {e}"[:200]
            tgen_sum = sum(m.get("tgen") or 0 for m in s.models.values())
            gpu_max = max((g.get("util_pct") or 0) for g in s.gpus) if s.gpus else None
            ttft_p95_ms = None
            for m in s.models.values():
                ttft = (m.get("latency") or {}).get("ttft") or {}
                if ttft.get("p95") is not None:
                    ttft_p95_ms = round(ttft["p95"] * 1000)
                    break
            s.ring_push(tgen_sum if tgen_sum > 0 else None, gpu_max, ttft_p95_ms)
        last_tick = time.time()
        if int(last_tick) % SNAPSHOT_EVERY < POLL_INTERVAL:
            snapshot()
        time.sleep(max(0.5, POLL_INTERVAL - (time.time() - t0)))


def load_config():
    global CONFIG, TOKEN, SERVERS, NO_AUTH
    with open(CONFIG_PATH) as f:
        CONFIG = json.load(f)
    TOKEN = _read_token() or ""
    NO_AUTH = os.environ.get("LLM_HUB_ALLOW_NO_AUTH", "") in ("1", "true", "True")
    if not TOKEN and not NO_AUTH:
        print(f"config error: no token at {TOKEN_PATH} — create one, or set "
              "LLM_HUB_ALLOW_NO_AUTH=1 for a deliberate unauthenticated (dev) "
              "deploy", file=sys.stderr)
        sys.exit(1)
    if not TOKEN:
        print(f"WARNING: LLM_HUB_ALLOW_NO_AUTH=1 — API is UNAUTHENTICATED "
              f"(anyone with network access can read AND load/unload models)",
              file=sys.stderr)
    errs = []
    seen = set()
    for c in CONFIG.get("servers", []):
        n = c.get("name")
        if not n:
            errs.append("server entry missing 'name'")
            continue
        if n in seen:
            errs.append(f"duplicate server name: {n!r}")
        seen.add(n)
        if c.get("kind") not in ("llama-router", "vllm", "gpu-only"):
            errs.append(f"{n}: kind must be 'llama-router', 'vllm' or 'gpu-only'")
        elif c.get("kind") != "gpu-only" and not c.get("url"):
            errs.append(f"{n}: kind '{c.get('kind')}' requires 'url'")
    for sc in CONFIG.get("gpu_sidecars", []):
        if not isinstance(sc, dict) or not sc.get("url"):
            errs.append(f"gpu_sidecars entry needs a 'url': {sc!r}")
            continue
        if sc.get("server") not in seen:
            errs.append(f"gpu_sidecars references unknown server "
                        f"{sc.get('server')!r}")
    if errs:    # fail fast at boot instead of silently misrouting a typo'd kind
        print("config errors in " + CONFIG_PATH + ":", file=sys.stderr)
        for e in errs:
            print("  - " + e, file=sys.stderr)
        sys.exit(1)
    SERVERS = [Server(c) for c in CONFIG.get("servers", [])]
    # gpu_sidecars list (name -> url) overrides per-server "sidecar" keys
    for sc in CONFIG.get("gpu_sidecars", []):
        s = next((x for x in SERVERS if x.name == sc.get("server")), None)
        if s is not None:
            s.sidecar_url = sc["url"]
    restore()


def _read_token():
    try:
        with open(TOKEN_PATH) as f:
            return f.read().strip()
    except OSError:
        return None


def main():
    load_config()
    port = int(CONFIG.get("port", 8443))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=poller, daemon=True)
    t.start()
    print(f"llm-hub: {len(SERVERS)} servers, port {port}, "
          f"token={'set' if TOKEN else 'MISSING'}", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
