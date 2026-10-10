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
CONTROL_VERDICT_S = 2.0             # how long to wait for a load/unload verdict
SIDECAR_TIMEOUT = 7                 # sidecar runs nvidia-smi (~1 s), allow margin
RATE_DECAY = 30                     # measured rates decay to idle after this
LAT_WINDOW = int(os.environ.get("LLM_HUB_LAT_WINDOW", "300"))
WIN_SAMPLES = max(2, LAT_WINDOW // POLL_INTERVAL)   # window ring buffer size
# vLLM prompt-compute throughput gauge window. The engine's computed-prompt
# counter advances in LUMPS (one jump per finished prefill — probed on the
# 0.28.0 dual-3090: flat through a 28 s prefill, +33K at first token), so a
# short window turns each lump into a fake spike; one minute amortizes the
# lumpy updates over a stable wall-clock interval to represent operational
# prompt-compute throughput. That is a work-rate, NOT the execution speed —
# a 33K-token prefill arriving as one 28 s lump reads ~553 t/s here while
# its phase-clock speed is ~1,186 t/s; the two are kept apart on purpose.
# See hub/README "What the numbers mean".
PP_TPUT_WINDOW = 60
COOKIE_SECURE = os.environ.get("LLM_HUB_COOKIE_SECURE", "").lower() in ("1", "true", "yes")

# vLLM latency histograms (emitted per engine/model; buckets incl. +Inf).
# "tpot" is the per-request time-per-output-token histogram (the canonical
# SLO metric: decode_time / (n_tokens - 1)); "itl" is the token-weighted
# inter-token latency histogram (grows with batch size — see README
# "What the numbers mean"). A build missing either just omits that row.
VLLM_HISTS = (
    ("ttft",  "time_to_first_token_seconds"),
    ("tpot",  "request_time_per_output_token_seconds"),
    ("itl",   "inter_token_latency_seconds"),
    ("e2e",   "e2e_request_latency_seconds"),
    ("queue", "request_queue_time_seconds"),
)
# Per-request phase histograms, sampled on request FINISH (a long-running
# request contributes its phase stats late): wall-clock prefill time
# (scheduled -> first token), decode time (first -> last token), and the
# KV tokens actually computed in prefill (cached tokens excluded).
VLLM_PHASE_HISTS = (
    ("prefill", "request_prefill_time_seconds"),
    ("decode",  "request_decode_time_seconds"),
    ("pfkv",    "request_prefill_kv_computed_tokens"),
)

# KV pool history: peak-usage windows shown in the UI, and how far back the
# hub keeps (t, kv_used) samples (2 s poll; time-trimmed to the longest
# window, deque maxlen as backstop).
KV_PEAK_WINDOWS = ((600, "10m"), (1800, "30m"), (3600, "1h"), (86400, "24h"))
KV_HIST_SECONDS = 86400
KV_HIST_MAX = 43200

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


def _control_post(url, obj):
    """Fire a control POST (model load/unload) and wait up to
    CONTROL_VERDICT_S for the upstream's verdict. Fast upstreams (the
    router's /models/load|unload handlers) answer in milliseconds; their
    200/4xx is reported as-is. Orchestration frontends that block during
    an exclusive-GPU switch (e.g. a mux that unloads the other backend
    and waits for VRAM) never answer within the budget: the request has
    already been sent and the action runs upstream, so return 202 and
    let state polling be authoritative instead of dying on the client
    timeout. (0, None) = upstream unreachable."""
    p = urllib.parse.urlsplit(url)
    host = p.hostname
    port = p.port or (443 if p.scheme == "https" else 80)
    path = p.path or "/"
    if p.query:
        path += "?" + p.query
    conn = (http.client.HTTPSConnection(host, port, timeout=CONTROL_VERDICT_S)
            if p.scheme == "https" else
            http.client.HTTPConnection(host, port, timeout=CONTROL_VERDICT_S))
    try:
        conn.connect()
    except (TimeoutError, OSError):
        return 0, None   # upstream unreachable: the action did not run
    try:
        conn.request("POST", path, body=json.dumps(obj).encode(),
                     headers={"Content-Type": "application/json"})
    except OSError:
        return 0, None   # request not even delivered: the action did not run
    try:
        r = conn.getresponse()
        return r.status, r.read().decode("utf-8", "replace")
    except (TimeoutError, http.client.HTTPException, OSError):
        return 202, ("no verdict within %.0f s; action runs upstream — "
                     "state polling is authoritative" % CONTROL_VERDICT_S)
    finally:
        conn.close()


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


def parse_engine_json(text):
    """Adapter for engines whose /metrics is a JSON dashboard, not
    Prometheus text (Strata serve, 2026-10-08: the hub's llama-router path
    called parse_prom() on the JSON body, got {}, and the UI flagged
    'metrics degraded' although every number the engine reports was
    right there). Maps the engine's cumulative totals onto the counter
    names the router poll path reads, so the same child-clock rate,
    restart re-baseline, and stall machinery applies unchanged:

      totals.output_tokens   -> tokens_predicted_total      (decode clock: decode_ms)
      totals.prompt_tokens   -> prompt/cached split (the engine's prompt
                                counter INCLUDES reused tokens; the hub's
                                tpp + cache_hit math wants the split, so
                                both sides are derived here)
      totals.prompt_ms       -> prompt clock
      totals.drafts_offered/ -> spec_decode_num_draft/accepted_tokens_total
        accepted
      live.queued            -> requests_processing

    Keys carry the llamacpp: prefix get_metric() probes. Returns None for
    any body that is not such a dashboard, so callers fall through to
    parse_prom(). Counters are engine-lifetime cumulative; a restart
    resets them and the downstream _rate() re-baselines on the decrease.

    In-flight smoothing: the engine commits `totals` only when a request
    completes, but the `live` object tracks the request in flight
    (`generated` tokens, `tok_s_mean`). While state == "generating" the
    adapter adds the in-flight share to the decode counter and clock
    (generated / tok_s_mean), so the hub's _rate() sees a ticking counter
    and reports the engine's own mean rate instead of going quiet for the
    whole response. At completion the engine absorbs the same tokens into
    totals; if the absorbed clock lands below the last synthesized value
    _rate() treats it as a restart and re-baselines for one poll —
    self-healing. The prompt clock stays totals-based: during a long
    fresh prefill both clocks freeze, so the (display-only) dual-clock
    stall flag can false-positive until the request completes.
    """
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    t = obj.get("totals")
    if not isinstance(t, dict) or t.get("output_tokens") is None:
        return None

    def f(x):
        return float(x or 0)

    def sec(ms):
        return f(ms) / 1000.0

    prompt = f(t.get("prompt_tokens"))
    reused = f(t.get("reused"))
    live = obj.get("live") or {}
    out = f(t.get("output_tokens"))
    decode_s = sec(t.get("decode_ms"))
    if live.get("state") == "generating":
        gen = f(live.get("generated"))
        out += gen
        tmean = f(live.get("tok_s_mean"))
        if tmean > 0:
            decode_s += gen / tmean
    return {
        "llamacpp:tokens_predicted_total": out,
        "llamacpp:tokens_predicted_seconds_total": decode_s,
        "llamacpp:prompt_tokens_total": max(prompt - reused, 0.0),
        "llamacpp:prompt_tokens_cached_total": reused,
        "llamacpp:prompt_tokens_seconds_total": sec(t.get("prompt_ms")),
        "llamacpp:spec_decode_num_draft_tokens_total": f(t.get("drafts_offered")),
        "llamacpp:spec_decode_num_accepted_tokens_total": f(t.get("drafts_accepted")),
        "llamacpp:requests_processing": f(live.get("queued")),
    }


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


def parse_sum(text, name):
    """Sum of all value samples of one metric across its label sets
    (parse_prom's max() would undercount a _sum series with several
    label combinations). None when the metric is absent."""
    prefix, plain = name + "{", name + " "
    total = None
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith(plain) or line.startswith(prefix):
            try:
                v = float(line.rpartition(" ")[2])
            except ValueError:
                continue
            total = (total or 0.0) + v
    return total


def parse_kv_pool(text):
    """KV pool capacity from vLLM's cache_config_info gauge (0.28+ exposes
    the whole cache config as labels, value == 1): pool size in tokens,
    block count, and max concurrency. None on builds that don't expose it."""
    line = next((l for l in text.splitlines()
                 if l.startswith("vllm:cache_config_info{")
                 and not l.startswith("#")), None)
    if line is None:
        return None
    def g(key):
        m = re.search(key + r'="([^"]+)"', line)
        if not m or m.group(1) == "None":
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None
    pool = {"size_tokens": g("kv_cache_size_tokens"),
            "num_blocks": g("num_gpu_blocks"),
            "max_concurrency": g("kv_cache_max_concurrency")}
    # config context for the UI line (strings; None when absent)
    def s(key):
        m = re.search(key + r'="([^"]+)"', line)
        return m.group(1) if m and m.group(1) not in ("None", "False") else None
    pool["block_size"] = g("block_size")
    pool["kv_dtype"] = s("cache_dtype")
    pool["offload"] = s("kv_offloading_backend")
    pool["sliding_window"] = g("sliding_window")
    return pool if any(v is not None for v in pool.values()) else None


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
        # per-model opt-in list: which model ids get the dual-clock stall
        # predicate (see the stall detection block in poll()). Absent/empty
        # = classic single-clock behaviour for every model on this server.
        self.dual_clock_models = cfg.get("dual_clock") or []
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
        self._rawgen = {}                  # mid -> (tokens_predicted_total, ts last moved)
        self._rawprompt = {}               # mid -> (prompt_tokens_total, ts last moved)
        self._spec_ts = {}                 # mid -> last spec-activity ts
        # rolling 5-min windows: deque of (ts, buckets, counters)
        self._winbuf = collections.deque(maxlen=WIN_SAMPLES)
        self._kv_hist = collections.deque(maxlen=KV_HIST_MAX)
        # llama.cpp per-model window: mid -> deque of (ts, {cached, new})
        self._wins = {}
        # vLLM working state (per model id):
        # _lc_hist: (ts, computed-prompt counter) 1-min ring -> throughput gauge
        # _live_prefill: prefill-detection start ts (None = not prefilling)
        # _vlast: raw (gen, ttft-count, prefill-KV) from the last poll
        # _phase_ts: when the completed-prefill figures last advanced (age)
        self._lc_hist = {}
        self._live_prefill = {}
        self._vlast = {}
        self._phase_ts = {}
        # sparkline ring: (t, tgen_sum, gpu_max, ttft_p95_ms|None, tpp_sum)
        # (ttft stays in the ring so the UI tooltip can show it; the graph
        #  lanes are tok/s, pp/s, gpu% — a 5-min-window percentile made a
        #  flat, hard-to-read lane, and it is better read on hover)
        self.ring = collections.deque(maxlen=RING_SECONDS // POLL_INTERVAL)

    # -- sparkline ----------------------------------------------------------
    def ring_push(self, tgen, gpu, ttft_p95_ms, tpp):
        now = time.time()
        # the poller is the only writer, but api() reads the ring from
        # handler threads — hold the lock so a mutation can never land
        # mid-iteration (deque raises RuntimeError on that; the poller
        # never holds the lock when it gets here, so no deadlock)
        with self._lock:
            self.ring.append((now, tgen, gpu, ttft_p95_ms, tpp))
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
            elif self.kind == "vllm-mux":
                self._poll_vllmmux(now)
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
        self._apply_vllm_metrics("(vllm)", body, now, self.desc)

    def _apply_vllm_metrics(self, mid, body, now, desc=None):
        """Parse a vLLM-native /metrics body into per-model entry `mid`.
        Shared by the `vllm` kind (one aggregate "(vllm)" entry) and the
        `vllm-mux` kind (attributed to the model whose engine is active —
        the mux forwards the active engine's native /metrics verbatim).
        When the attributed mid changes (an engine switch), the rolling
        window buffer is cleared so 5-min deltas never straddle two
        engines; per-(mid,key) counter rates re-baseline via _rate's
        restart detection (value < prev -> clean re-baseline)."""
        if getattr(self, "_metrics_mid", None) not in (None, mid):
            self._winbuf.clear()
            # an engine switch changes the counter world: no queue, rate or
            # age may straddle two engines
            self._lc_hist.clear()
            self._live_prefill.clear()
            self._vlast.clear()
            self._phase_ts.clear()
        self._metrics_mid = mid
        p = parse_prom(body)
        gen = get_metric(p, "generation_tokens_total", prefixes=("vllm:", "vllm_"))
        # vLLM's prompt_tokens_total counts each request's FULL prompt
        # length — prefix-cache hits included — so its raw rate is
        # inflated whenever the cache is hot (13.7M counted vs 0.8M
        # computed on the dual-3090 with a 95% hit rate). The GPU's
        # actual prefill work is the computed share: prompt_tokens_by_source
        # {source="local_compute"} when the build exposes it, else the
        # equivalent queries - hits (both monotone non-decreasing, so
        # _rate's restart re-baseline stays safe).
        src = parse_labeled(body, "vllm:prompt_tokens_by_source_total", "source")
        if "local_compute" in src:
            pp = src["local_compute"]
        else:
            pq = get_metric(p, "prefix_cache_queries_total",
                            prefixes=("vllm:", "vllm_")) or 0.0
            ph = get_metric(p, "prefix_cache_hits_total",
                            prefixes=("vllm:", "vllm_")) or 0.0
            pp = max(0.0, pq - ph)
        # the two other prompt-token quantities (diagnostic only — never
        # read as performance): full requested prompt length, and the
        # prefix-cache share of it
        pt = get_metric(p, "prompt_tokens_total", prefixes=("vllm:", "vllm_"))
        lh = src.get("local_cache_hit")
        if lh is None:
            lh = get_metric(p, "prefix_cache_hits_total",
                            prefixes=("vllm:", "vllm_"))
        ext = src.get("external_kv_transfer")
        req_running = int(get_metric(p, "num_requests_running",
                                     prefixes=("vllm:", "vllm_")) or 0)
        req_waiting = int(get_metric(p, "num_requests_waiting",
                                     prefixes=("vllm:", "vllm_")) or 0)
        kv = get_metric(p, "kv_cache_usage_perc", prefixes=("vllm:", "vllm_"))
        if kv is not None and kv > 1.0:       # some builds expose 0..100
            kv /= 100.0
        kv = round(kv, 3) if kv is not None else None

        # KV pool capacity + recent peak usage. The usage gauge is a
        # point-in-time snapshot (and low whenever idle, because the pool
        # is sized by VRAM — ~2.7x max ctx on the dual-3090), so the hub
        # keeps its own (t, kv) history to answer "how full does it get
        # and when" across 10m/30m/1h/24h windows.
        pool = parse_kv_pool(body)
        if kv is not None:
            self._kv_hist.append((now, kv))
            cut = now - KV_HIST_SECONDS
            while self._kv_hist and self._kv_hist[0][0] < cut:
                self._kv_hist.popleft()
        peaks = {}
        for secs, label in KV_PEAK_WINDOWS:
            w = [v for t, v in self._kv_hist if t >= now - secs]
            peaks[label] = round(max(w), 3) if w else None

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
        counters["gen"] = gen
        counters["lc"] = pp
        counters["lh"] = lh if lh is not None else 0.0
        counters["pt"] = pt if pt is not None else 0.0
        ttftn = parse_sum(body, "vllm:time_to_first_token_seconds_count")
        if ttftn is None:                       # older builds: bare names
            ttftn = parse_sum(body, "time_to_first_token_seconds_count")
        counters["ttftn"] = ttftn if ttftn is not None else 0.0
        for key, base in VLLM_PHASE_HISTS:
            s = parse_sum(body, "vllm:" + base + "_sum")
            if s is None:                       # older builds: bare names
                s = parse_sum(body, base + "_sum")
            counters["sum:" + key] = s

        # spec decoding / MTP (official vLLM counters; windowed below)
        counters["mtp_d"] = get_metric(p, "spec_decode_num_drafts_total",
                                       prefixes=("vllm:", "vllm_"))
        counters["mtp_dt"] = get_metric(p, "spec_decode_num_draft_tokens_total",
                                        prefixes=("vllm:", "vllm_"))
        counters["mtp_acc"] = get_metric(p, "spec_decode_num_accepted_tokens_total",
                                         prefixes=("vllm:", "vllm_"))
        for mm in re.finditer(
                r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*?position="(\d+)"[^}]*\}\s+([0-9.eE+-]+)',
                body):
            counters["mtp_pos" + mm.group(1)] = float(mm.group(2))
        # per-request shape (prompt/generation token histograms)
        counters["rp_sum"] = parse_sum(body, "vllm:request_prompt_tokens_sum")
        counters["rg_sum"] = parse_sum(body, "vllm:request_generation_tokens_sum")
        # per-histogram request counts, parsed SEPARATELY: the two histograms
        # count the same requests today, but that is an engine invariant, not
        # something the hub should assume (parallel sampling makes request-
        # vs sequence accounting diverge upstream).
        counters["reqc"] = parse_sum(body, "vllm:request_prompt_tokens_count")
        counters["reqc_g"] = parse_sum(body, "vllm:request_generation_tokens_count")
        counters["ext"] = ext

        # ---- engine-restart detection (before any window math) ----------
        # The computed-token counter decreasing means the vLLM process
        # restarted with the SAME model ID (a mux engine switch changes the
        # attributed mid and is handled above). Every engine-derived
        # working state is then a different counter world: the 5-min
        # window (phase histograms), the 1-min ring, the live-prefill
        # detection, the phase age — clear it all and re-baseline on this
        # sample. Refusing the single negative delta alone is NOT enough:
        # pre-restart phase sums would otherwise sit as window baselines
        # until they age out (minutes of blank speed).
        ppq = self._lc_hist.get(mid)
        lc_reset = (ppq is not None and len(ppq) > 0
                    and (pp is None or pp < ppq[-1][1] - 1e-9))
        if lc_reset:
            self._winbuf.clear()
            self._lc_hist.pop(mid, None)
            self._live_prefill.pop(mid, None)
            self._vlast.pop(mid, None)
            self._phase_ts.pop(mid, None)
        self._winbuf.append((now, buckets, counters))

        latency = finish_win = cache_hit = preempt_win = phase = None
        prompt_work = spec = req_shape = None
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
            # per-request phase stats over the window (histograms are
            # sampled on request FINISH): p50/p95/p99 of the phase
            # histogram + the _sum deltas -> real prefill/decode speeds
            # independent of the request-arrival pattern.
            ph = {}
            for key, base in VLLM_PHASE_HISTS:
                full = next((n for n in (base, "vllm:" + base,
                                          "vllm_" + base)
                             if n in b0 or n in buckets), base)
                d = _bucket_deltas(b0.get(full), buckets.get(full))
                sd = _counter_delta(c0.get("sum:" + key),
                                    counters.get("sum:" + key))
                if d is None and sd is None:
                    ph[key] = None
                else:
                    ph[key] = {
                        "p50": hist_quantile(d, 0.50) if d else None,
                        "p95": hist_quantile(d, 0.95) if d else None,
                        "p99": hist_quantile(d, 0.99) if d else None,
                        "n": int(d.get(float("inf"), 0)) if d else None,
                        "sum": sd,
                    }
            if any(ph.values()):
                pf, dd, pk = ph.get("prefill"), ph.get("decode"), ph.get("pfkv")
                prefill_speed = decode_speed = None
                if (pf and pf.get("sum") and pf["sum"] > 0
                        and pk is not None and pk.get("sum") is not None):
                    prefill_speed = round(pk["sum"] / pf["sum"], 1)
                gen_d = _counter_delta(c0.get("gen"), counters.get("gen"))
                if dd and dd.get("sum") and dd["sum"] > 0 and gen_d is not None:
                    decode_speed = round(gen_d / dd["sum"], 1)
                phase = {
                    "window_s": win, "prefill": pf, "decode": dd,
                    "pfkv": pk, "prefill_speed": prefill_speed,
                    "decode_speed": decode_speed,
                }
            else:
                phase = None

            # 5-min prompt-work breakdown (diagnostic, not performance):
            # full requested prompt vs what the GPU computed vs what the
            # prefix cache served. On a cache-hot fleet these differ ~19x,
            # which is exactly why only `computed` may feed a rate.
            d_pt = _counter_delta(c0.get("pt"), counters.get("pt"))
            d_lc = _counter_delta(c0.get("lc"), counters.get("lc"))
            d_lh = _counter_delta(c0.get("lh"), counters.get("lh"))
            d_ext = _counter_delta(c0.get("ext"), counters.get("ext"))
            if d_pt is not None and d_lc is not None and d_pt >= 0:
                served = (int((d_lh or 0) + (d_ext or 0))
                          if (d_lh is not None or d_ext is not None)
                          else None)
                # vLLM's prompt-token accounting invariant:
                #   computed + local_cache + external_kv == requested
                # (external = 0 on non-LMCache builds). A violation is a
                # version/accounting drift — flag it instead of displaying
                # an impossible partition as fact.
                drift = None
                if served is not None and d_pt > 0:
                    resid = abs(d_pt - d_lc - served)
                    drift = bool(resid / d_pt > 0.02)
                prompt_work = {
                    "window_s": win,
                    "total": int(d_pt),
                    "computed": int(d_lc),
                    "cached": int(d_lh) if d_lh is not None else None,
                    "ext": int(d_ext) if d_ext is not None else None,
                    "served": served,
                    # the % shown next to the COMBINED served value is the
                    # combined ratio; the local-only ratio stays separate
                    # (they differ exactly when external KV is in play).
                    "hit_ratio": (round(served / d_pt, 3)
                                  if served is not None and d_pt > 0
                                  else None),
                    "local_ratio": (round(d_lh / d_pt, 3)
                                    if d_lh is not None and d_pt > 0
                                    else None),
                    "drift": drift,
                }
            else:
                prompt_work = None

            # spec decoding / MTP over the window (ratios of WINDOW deltas,
            # not lifetime ratios): acceptance = accepted/draft tokens;
            # mean acceptance length = 1 + accepted/drafts (the +1 is the
            # bonus token vLLM documents). Per-draft-position acceptance
            # answers "is k=3 still worth it": a position accepting near 0
            # pays verification cost for nothing.
            dd_ = _counter_delta(c0.get("mtp_d"), counters.get("mtp_d"))
            ddt = _counter_delta(c0.get("mtp_dt"), counters.get("mtp_dt"))
            daa = _counter_delta(c0.get("mtp_acc"), counters.get("mtp_acc"))
            spec = None
            if dd_ is not None and dd_ > 0 and ddt is not None and daa is not None:
                pos = {}
                for i in range(8):
                    dp = _counter_delta(c0.get(f"mtp_pos{i}"),
                                        counters.get(f"mtp_pos{i}"))
                    if dp is not None:
                        pos[str(i)] = round(dp / dd_, 3)
                spec = {
                    "window_s": win,
                    "drafts": int(dd_),
                    "acceptance": round(daa / ddt, 3) if ddt > 0 else None,
                    "mean_len": round(1 + daa / dd_, 2),
                    "pos": pos or None,
                }

            # request shape over the window (context for latency numbers:
            # a TTFT p95 jump means different things at 4K vs 45K avg input).
            # Each average divides by ITS OWN histogram's request count —
            # the two counts normally agree, and baking that in would be a
            # bug waiting for parallel sampling / unusual shapes.
            d_rp = _counter_delta(c0.get("rp_sum"), counters.get("rp_sum"))
            d_rg = _counter_delta(c0.get("rg_sum"), counters.get("rg_sum"))
            d_rc = _counter_delta(c0.get("reqc"), counters.get("reqc"))
            d_rcg = _counter_delta(c0.get("reqc_g"), counters.get("reqc_g"))
            req_shape = None
            if (d_rc is not None and d_rc > 0) or (
                    d_rcg is not None and d_rcg > 0):
                req_shape = {
                    "window_s": win,
                    "n": int(max(d_rc or 0, d_rcg or 0)),
                    "avg_in": (round(d_rp / d_rc)
                               if d_rp is not None and d_rc else None),
                    "avg_out": (round(d_rg / d_rcg)
                                if d_rg is not None and d_rcg else None),
                }

        # ---- 1-min prompt-compute throughput (lump-safe gauge) ----------
        # the counter's arrivals are per-prefill lumps (see PP_TPUT_WINDOW):
        # a rolling 1-min ring amortizes the lumpy counter updates over a
        # stable wall-clock interval to represent OPERATIONAL prompt-compute
        # throughput — deliberately NOT the execution speed (a 33K-token
        # prefill arriving as one 28-s lump reads ~553 t/s here, half its
        # 1,181 t/s phase speed; the two stay apart on purpose). A counter
        # reset (engine restart) was handled above (full re-baseline); a
        # gap (>3 polls without a sample — not the active engine) just
        # clears the ring.
        lhq = self._lc_hist.setdefault(
            mid, collections.deque(maxlen=PP_TPUT_WINDOW // POLL_INTERVAL))
        if not lc_reset and lhq and now - lhq[-1][0] > 15:
            lhq.clear()
        if pp is not None:
            lhq.append((now, pp))
        pp_tput_1m = None
        if len(lhq) >= 2:
            t_a, v_a = lhq[0]
            t_b, v_b = lhq[-1]
            if t_b - t_a >= 30 and v_b >= v_a:
                pp_tput_1m = round((v_b - v_a) / (t_b - t_a), 1)

        # ---- live in-flight prefill (detected, not measured) -----------
        # This build exposes no token counter during a chunked prefill
        # (probe 2026-09-30: local_compute flat for 28 s, one jump at
        # first token), so a live prefill is DETECTED — requests running
        # while neither the generation counter nor the TTFT histogram
        # advanced on the last poll — and shown as an honest timer. No
        # rate is invented for it; the completed-request phase speed
        # (above) is the only number that may be read as prefill
        # performance.
        vl = self._vlast.get(mid)
        gen_moved = (vl is not None and gen is not None
                     and vl.get("gen") is not None and gen != vl["gen"])
        ttft_moved = (vl is not None and vl.get("ttftn") is not None
                      and counters.get("ttftn") is not None
                      and counters["ttftn"] != vl["ttftn"])
        if req_running > 0 and not gen_moved and not ttft_moved:
            start = self._live_prefill.get(mid) or now
        else:
            start = None
        self._live_prefill[mid] = start
        # prefill-speed age: when the completed-prefill numerator (the
        # computed-KV histogram sum) last advanced
        pks = counters.get("sum:pfkv")
        if (pks is not None and vl is not None and vl.get("pks") is not None
                and pks != vl["pks"] and not lc_reset):
            self._phase_ts[mid] = now
        self._vlast[mid] = {"gen": gen, "ttftn": counters.get("ttftn"),
                            "pks": pks, "ts": now}
        prefill_speed_age = (int(now - self._phase_ts[mid])
                             if mid in self._phase_ts else None)
        live_prefill = ({"active": True, "elapsed_s": int(now - start)}
                        if start else None)

        sleep_state = next((k for k, v in sleep.items() if v >= 1), None)
        wr = None
        if req_waiting > 0 and any(v > 0 for v in wait_reason.values()):
            wr = max(wait_reason.items(), key=lambda kv: kv[1])[0]

        with self._lock:
            st = self.models.setdefault(mid, {
                "id": mid, "kind": "vllm",
                "desc": desc if desc is not None else self.desc,
                "loaded": True,
            })
            # config-owned metadata refreshes every poll — a snapshot-restored
            # dict must not pin a stale desc (llama branch does the same)
            if desc is not None:
                st["desc"] = desc
            st.update({
                "tgen": self._rate(mid, "gen", gen, now),
                # headline pp/s for a vLLM model is the completed-request
                # phase speed (computed KV tokens / vLLM's PREFILL-phase
                # seconds over the window) — the physical number, immune
                # to counter arrival patterns. The old wall-rate of the
                # computed counter (one 2 s delta) is gone: the counter
                # jumps in lumps at prefill completion, so that rate read
                # 5,000 t/s for a prefill that took 28 s (probe 2026-09-30).
                # The operational work-rate lives in pp_tput_1m.
                "tpp": phase.get("prefill_speed") if phase else None,
                "pp_tput_1m": pp_tput_1m,
                "prompt_work": prompt_work,
                "live_prefill": live_prefill,
                "prefill_speed_age": prefill_speed_age,
                "prompt_tokens_computed_raw": pp,
                "spec": spec,
                "req_shape": req_shape,
                "spec_accept": (spec["acceptance"] if spec else None),
                "req_running": req_running, "req_waiting": req_waiting,
                "kv_used": kv,
                "kv_used_tokens": (round(kv * pool["size_tokens"]) if
                                   kv is not None and pool and pool.get("size_tokens")
                                   else None),
                "kv_pool": pool, "kv_peaks": peaks,
                "latency": latency, "finish": finish_win, "phase": phase,
                "cache_hit": cache_hit, "preempt_win": preempt_win,
                "preempt_total": int(get_metric(p, "num_preemptions_total",
                                                prefixes=("vllm:", "vllm_")) or 0),
                "sleep": sleep_state, "wait_reason": wr,
                "metrics_ok": gen is not None,
            })

    def _poll_vllmmux(self, now):
        # vllm-mux: one front endpoint (vllm-mux) in front of N vLLM
        # engines that cannot be resident together. Catalog + load state
        # come from the mux's synthesized /v1/models (OpenAI-style, with a
        # status value); engine metrics come from the mux's /metrics
        # forward — the ACTIVE engine's native vLLM counters, attributed
        # to the loaded model. The UI renders router-style loaded/idle
        # chips + load/unload buttons (kind != "vllm") while the per-model
        # vLLM metrics block still applies (model kind stays "vllm").
        status, body = http_json(f"{self.url}/v1/models")
        if status != 200 or not body:
            raise RuntimeError(f"/v1/models HTTP {status or 'timeout'}")
        try:
            listing = json.loads(body).get("data", [])
        except json.JSONDecodeError:
            raise RuntimeError("/v1/models bad json")
        catalog = {}
        for m in listing:
            mid = m.get("id")
            if mid:
                catalog[mid] = (m.get("status") or {}).get("value") == "loaded"
        # Display is config-driven: ctx aliases of one engine (same
        # systemd unit, all names loaded together) are not shown as
        # individually loadable rows — the card shows the switch units
        # (engine bundle + stopgap). With no hints configured, fall back
        # to the raw mux catalog. All names of one engine are loaded
        # together, so the loaded row == the resident engine's row.
        if self.model_hints:
            display = {mid: catalog.get(mid, False) for mid in self.model_hints}
        else:
            display = dict(catalog)
        active = next((mid for mid, l in display.items() if l), None)
        with self._lock:
            # drop entries this card doesn't show (e.g. the "(vllm)"
            # aggregate restored from a previous plain-vllm config)
            for mid in [m for m in self.models if m not in display]:
                del self.models[mid]
            for mid, loaded in display.items():
                hint = self.model_hints.get(mid) or {}
                st = self.models.setdefault(mid, {
                    "id": mid, "kind": "vllm",
                    "ctx": hint.get("ctx"), "desc": hint.get("desc", ""),
                })
                st["loaded"] = loaded
                if hint.get("ctx"):
                    st["ctx"] = hint["ctx"]
                if hint.get("desc"):
                    st["desc"] = hint["desc"]
                if not loaded:
                    st.update({"tgen": None, "tpp": None, "spec_accept": None,
                               "pp_tput_1m": None, "prompt_work": None,
                               "live_prefill": None, "prefill_speed_age": None,
                               "prompt_tokens_computed_raw": None,
                               "req_running": None, "req_waiting": None,
                               "kv_used": None, "kv_used_tokens": None,
                               "kv_pool": None, "kv_peaks": None,
                               "latency": None, "finish": None, "phase": None,
                               "cache_hit": None, "preempt_win": None,
                               "sleep": None, "wait_reason": None,
                               "metrics_ok": False})
                    self._live_prefill[mid] = None
            # Re-key: the loop above keeps first-seen positions (the boot
            # snapshot resurrects a dead process's order), so re-key into
            # the current display (config) order each poll.
            self.models = {mid: self.models[mid] for mid in display
                           if mid in self.models}
        if active is None:
            return                      # all idle: no metrics source
        status, body = http_json(f"{self.url}/metrics")
        if status == 200 and body:
            # pass only the hint desc (nullable): _apply_vllm_metrics updates
            # st["desc"] with it, and an absent hint leaves the existing
            # value (creation default = the server desc) alone
            self._apply_vllm_metrics(
                active, body, now, (self.model_hints.get(active) or {}).get("desc"))
        else:
            with self._lock:
                if active in self.models:
                    self.models[active]["metrics_ok"] = False

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
                p = parse_engine_json(mt) if mt else None
                if p is None:
                    p = parse_prom(mt) if mt else None
            fetched.append((mid, loaded, p, m))
        with self._lock:
            # prune ghost rows: models absent from both the upstream roster
            # and the config hints (e.g. an ID folded at an identity boundary
            # in front of the card). Same rule the vllm-mux branch applies;
            # without it, a model removed from the config resurrects from
            # the boot snapshot and lingers as a stale idle row forever.
            # Counter/rate state is left alone: if the ID reappears, _rate()
            # re-baselines on the counter decrease.
            known = {mid for mid, _, _, _ in fetched} | set(self.model_hints)
            for mid in [m for m in self.models if m not in known]:
                del self.models[mid]
            # Re-key for display: config order (the vllm-mux branch's rule),
            # else upstream roster order. The setdefault loop above keeps
            # first-seen positions for the process's lifetime — the boot
            # snapshot resurrects a dead process's order, so a roster or
            # config reorder otherwise reaches the UI only on a cold start
            # (2026-10-10: a reordered card row sat at the bottom of its
            # list for the whole process lifetime).
            by_mid = dict(self.models)
            if self.model_hints:
                order = ([mid for mid in self.model_hints]
                         + [mid for mid in by_mid if mid not in self.model_hints])
            else:
                order = [mid for mid, _, _, _ in fetched]
            self.models = {mid: by_mid[mid] for mid in order if mid in by_mid}
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
                               "stalled": False, "metrics_ok": False})
                    continue
                # parser-compat probe: the two counters every hub rate
                # depends on. False = backend reachable but counter names broke
                st["metrics_ok"] = (
                    get_metric(p, "tokens_predicted_total") is not None
                    and get_metric(p, "prompt_tokens_total") is not None)
                st["tgen"] = self._rate(
                    mid, "gen", get_metric(p, "tokens_predicted_total"), now,
                    secs=get_metric(p, "tokens_predicted_seconds_total"))
                # tpp: prompt_tokens_total EXCLUDES cached tokens on every
                # honest source (mainline engine; mux v2.3 synthesis) — the
                # cached share lives in prompt_tokens_cached_total. Rate it on
                # the child's own prompt clock (prompt-seconds total; builds
                # disagree on the name) so a big prefill landing inside one
                # poll interval divides by its real processing time, not the
                # 2 s poll gap (the ~16,500 "pp/s" artifact behind the 2026-09-19
                # big-prompt misreading; same child-clock principle as
                # tgen and as the vLLM computed-prefill fix).
                tpp_secs = (get_metric(p, "prompt_tokens_seconds_total")
                            or get_metric(p, "prompt_seconds_total"))
                st["tpp"] = self._rate(
                    mid, "prompt", get_metric(p, "prompt_tokens_total"), now,
                    secs=tpp_secs, gauge=get_metric(p, "prompt_tokens_seconds"))
                # stall detection (2 variants, per-model gated by the
                # server config's "dual_clock" list):
                # - classic single clock: some builds (observed on the ISTA
                #   D-CFR dev branch) FREEZE their per-token counters
                #   mid-session while the slot keeps decoding — generation
                #   continues but every counter-fed number goes stale and
                #   the rates below silently vanish. GPU busy + generation
                #   counter not moving = stalled: surface it explicitly
                #   instead of a blank "active" row.
                # - dual clock: llama.cpp engines whose honest metrics the
                #   hub reads directly. tokens_predicted_total stays flat
                #   through a prefill BY DESIGN (it counts generated
                #   tokens), so the single clock false-positived on every
                #   long prefill (2026-09-29: the QFN 35K-token request was
                #   flagged "stalled" the whole time its prompt was still
                #   being processed at full rate). There the prompt clock
                #   (prompt_tokens_total, which advances per prompt ubatch)
                #   must also be frozen 30+ s before the flag is set.
                raw = get_metric(p, "tokens_predicted_total")
                if raw is not None:
                    rg = self._rawgen.get(mid)
                    if rg is None or raw != rg[0]:
                        self._rawgen[mid] = (raw, now)
                rg = self._rawgen.get(mid)
                rp = None
                if mid in self.dual_clock_models:
                    rawp = get_metric(p, "prompt_tokens_total")
                    if rawp is not None:
                        pr = self._rawprompt.get(mid)
                        if pr is None or rawp != pr[0]:
                            self._rawprompt[mid] = (rawp, now)
                        rp = self._rawprompt.get(mid)
                gpumax = max((g.get("util_pct") or 0) for g in self.gpus) \
                    if self.gpus else 0.0
                if mid in self.dual_clock_models:
                    st["stalled"] = bool(
                        rg is not None and now - rg[1] > 30
                        and rp is not None and now - rp[1] > 30
                        and gpumax > 10)
                else:
                    st["stalled"] = bool(
                        rg is not None and now - rg[1] > 30 and gpumax > 10)
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
            # No sidecar configured for this server. A non-gpu-only server (a real
            # llama.cpp/vLLM box) with no sidecar has NO live GPU source, so any
            # gpus it carries (e.g. restored from the boot snapshot) can never be
            # refreshed. Flag them stale instead of silently re-presenting a frozen
            # value as fresh telemetry with gpus_stale=0 — the failure that hid a
            # flat GPU line for a whole day. gpu-only servers are defined by their
            # sidecar, so there is nothing to mark if they have none.
            if self.kind != "gpu-only" and self.gpus and not self.gpus_stale:
                self.gpus_stale = True
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
            # under the same lock: the poller's ring_push must not mutate
            # while we iterate (deque raises RuntimeError on concurrent
            # mutation; a handler thread used to 502 /metrics on that)
            spark = [[int(t), g, u, l, p] for (t, g, u, l, p) in self.ring]
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
    server_version = "llm-hub/1.5"

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
            # Control actions get their own short verdict budget: fast
            # upstreams report their 200/4xx, slow ones (orchestrated
            # switches) are acknowledged with 202 and the state poller
            # shows the result. An unreachable upstream is a clean 502,
            # never a crashed handler.
            status, resp = _control_post(f"{s.url}/models/{endpoint}",
                                         {"model": model})
            self._audit(f"model {endpoint}", f"{server_name}/{model}", status)
            return self._send(status if status else 502,
                              {"upstream_status": status,
                               "response": (resp[:500]
                                            if isinstance(resp, (bytes, str))
                                            else None)})
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


PROM_META = {
    # exposition contract for every series the hub emits (TYPE + HELP)
    "hub_servers_total": ("gauge", "Configured hub upstream servers."),
    "hub_server_online": ("gauge", "1 = server reachable on its last poll (stale/absent semantics: the frozen gauges drop when this is 0)."),
    "hub_server_gpu_count": ("gauge", "Number of GPUs reported by the server sidecar."),
    "hub_server_gpus_stale": ("gauge", "1 = the per-GPU point gauges are withheld (sidecar unreachable) — a frozen value must never masquerade as a current reading."),
    "hub_gpu_utilization_pct": ("gauge", "GPU utilization, percent of peak (nvidia-smi style)."),
    "hub_gpu_memory_used_mib": ("gauge", "GPU memory used, MiB."),
    "hub_gpu_memory_total_mib": ("gauge", "GPU memory total, MiB."),
    "hub_gpu_temp_c": ("gauge", "GPU temperature, deg C."),
    "hub_gpu_power_w": ("gauge", "GPU power draw, W."),
    "hub_model_loaded": ("gauge", "1 = model currently loaded on its engine."),
    "hub_model_tokens_per_second": ("gauge", "Generation tokens per second — engine-dependent clock: llama.cpp = tokens / the engine's own decode-phase seconds (child clocks, a true DECODE EXECUTION SPEED); vLLM = delta(tokens) / wall-clock poll interval — AGGREGATE SERVICE THROUGHPUT (this vLLM build exposes no decode-phase-seconds counter). Same series name, two different quantities per engine kind; the UI's engine column disambiguates."),
    "hub_model_prompt_tokens_per_second": ("gauge", "llama.cpp ONLY: prompt tokens / the engine's own prompt-phase seconds (child clocks) — a defensible speed. vLLM has no such counter (its prompt counter arrives in per-prefill lumps) — for vLLM, prompt work is the computed_tokens_total counter + the compute_throughput gauge below."),
    "hub_model_prompt_tokens_computed_total": ("counter", "vLLM computed prompt tokens, monotonic per engine lifetime (prefix-cache hits EXCLUDED). A decrease is a counter reset (engine process restart / mux engine switch) in the Prometheus sense. Rate it at your own window — 5m for a stable number, 1m for responsiveness."),
    "hub_model_prompt_compute_throughput_tokens_per_second": ("gauge", "vLLM: computed prompt tokens per WALL second over the trailing minute (amortized lumps). Operational workload rate, deliberately NOT the execution speed — a 33K-token prefill arriving as one 28 s lump reads ~553 t/s here while its phase-clock speed is ~1,186 t/s."),
    "hub_model_prefill_in_flight": ("gauge", "vLLM: 1 = a prefill is in flight right now (requests running with no first token yet on the last poll). A detector flag, not a rate — this build exposes no token counter during a chunked prefill."),
    "hub_model_prefill_speed_age_seconds": ("gauge", "vLLM: seconds since the last completed request contributed a prefill measurement (phase histograms are sampled at request COMPLETION, not TTFT). Above ~60 s, treat the speed as stale."),
    "hub_vllm_metrics_ok": ("gauge", "1 = the vLLM metric names the hub expects were present (a 0 flags a vLLM version rename instead of silent zeros)."),
    "hub_llama_metrics_ok": ("gauge", "1 = llama.cpp metrics parsed with the expected names."),
    "hub_model_requests_running": ("gauge", "vLLM requests in the running state."),
    "hub_model_requests_waiting": ("gauge", "vLLM requests waiting (queued)."),
    "hub_model_requests_processing": ("gauge", "llama.cpp requests currently processing."),
    "hub_model_requests_deferred": ("gauge", "llama.cpp requests deferred by n_parallel (the router's wait queue)."),
    "hub_model_busy_slots": ("gauge", "llama.cpp decode slots in use."),
    "hub_model_kv_cache_used": ("gauge", "vLLM KV pool occupancy, 0..1."),
    "hub_model_kv_used_tokens": ("gauge", "vLLM KV pool occupancy in tokens."),
    "hub_model_kv_pool_tokens": ("gauge", "vLLM KV pool capacity in tokens (cache_config)."),
    "hub_model_kv_pool_blocks": ("gauge", "vLLM KV pool capacity in blocks (cache_config)."),
    "hub_model_kv_pool_max_concurrency": ("gauge", "vLLM max concurrent contexts the pool supports (cache_config)."),
    "hub_model_kv_peak": ("gauge", "vLLM KV occupancy peak over the hub's own sampling window (10m/30m/1h/24h; the Prometheus scrape is the longer-term store)."),
    "hub_model_ttft_p50_seconds": ("gauge", "Time to first token, window p50 (vLLM histogram)."),
    "hub_model_ttft_p95_seconds": ("gauge", "Time to first token, window p95 (vLLM histogram)."),
    "hub_model_ttft_p99_seconds": ("gauge", "Time to first token, window p99 (vLLM histogram)."),
    "hub_model_tpot_p50_seconds": ("gauge", "Time per output token, window p50."),
    "hub_model_tpot_p95_seconds": ("gauge", "Time per output token, window p95."),
    "hub_model_tpot_p99_seconds": ("gauge", "Time per output token, window p99."),
    "hub_model_itl_p50_seconds": ("gauge", "Inter-token latency, window p50."),
    "hub_model_itl_p95_seconds": ("gauge", "Inter-token latency, window p95."),
    "hub_model_itl_p99_seconds": ("gauge", "Inter-token latency, window p99."),
    "hub_model_e2e_p50_seconds": ("gauge", "End-to-end request latency, window p50."),
    "hub_model_e2e_p95_seconds": ("gauge", "End-to-end request latency, window p95."),
    "hub_model_e2e_p99_seconds": ("gauge", "End-to-end request latency, window p99."),
    "hub_model_queue_p50_seconds": ("gauge", "Queue time before running, window p50."),
    "hub_model_queue_p95_seconds": ("gauge", "Queue time before running, window p95."),
    "hub_model_queue_p99_seconds": ("gauge", "Queue time before running, window p99."),
    "hub_model_prefill_p50_seconds": ("gauge", "Per-request PREFILL phase duration, window p50 (vLLM phase histograms; quantiles, not means — renamed from the misleading *_avg_seconds)."),
    "hub_model_prefill_p95_seconds": ("gauge", "Per-request PREFILL phase duration, window p95."),
    "hub_model_decode_p50_seconds": ("gauge", "Per-request DECODE phase duration, window p50."),
    "hub_model_decode_p95_seconds": ("gauge", "Per-request DECODE phase duration, window p95."),
    "hub_model_prefill_computed_tokens_window": ("gauge", "vLLM prefill KV-computed tokens of completed requests in the window (numerator of the prefill speed; lumpy by construction)."),
    "hub_model_prefill_speed_tokens_per_second": ("gauge", "vLLM prefill EXECUTION SPEED: prefill KV-computed tokens / prefill-phase seconds over completed requests in the window. The performance figure — cached/transferred tokens excluded; independent of the request-arrival pattern."),
    "hub_model_decode_speed_tokens_per_second": ("gauge", "vLLM decode speed from the phase clocks (generation tokens / decode-phase seconds over the window)."),
    "hub_model_preemptions_total": ("counter", "vLLM preemptions, engine lifetime (monotonic; resets with the engine process — a decrease is a counter reset in the Prometheus sense)."),
    "hub_model_preemptions_window": ("gauge", "vLLM preemptions in the window (a preemption alone degrades the hub state — it is an engine impairment, not a request outcome)."),
    "hub_model_finish_total_window": ("gauge", "Requests finished in the window (any reason)."),
    "hub_model_finish_length_window": ("gauge", "Requests finished by hitting the token limit in the window (a normal request outcome — NOT a health signal)."),
    "hub_model_finish_abort_window": ("gauge", "Requests aborted in the window."),
    "hub_model_prefix_cache_hit": ("gauge", "vLLM prefix-cache hit ratio over the window (hits/queries)."),
    "hub_model_prompt_cache_hit": ("gauge", "llama.cpp prompt-cache hit ratio over the window."),
    "hub_model_engine_asleep": ("gauge", "1 = the vLLM engine reports a non-awake sleep state (e.g. after auto-SHM/swap sleep)."),
    "hub_model_spec_acceptance": ("gauge", "vLLM spec-decoding (MTP) draft-token acceptance, window delta: accepted draft tokens / draft tokens. A ratio of window deltas, not a lifetime ratio."),
    "hub_model_spec_accept_length": ("gauge", "vLLM spec-decoding mean acceptance length, window delta: 1 + accepted/drafts (the +1 is the bonus token)."),
    "hub_model_prompt_tokens_mean": ("gauge", "Mean prompt length (tokens) of COMPLETED requests in the window (request_prompt_tokens histogram, divided by its own request count). Context for reading TTFT."),
    "hub_model_generation_tokens_mean": ("gauge", "Mean generation length (tokens) of COMPLETED requests in the window (request_generation_tokens histogram, divided by its own request count)."),
    "hub_vision_route_available": ("gauge", "1 = a vision-capable model is currently loaded (route recomputed at scrape time)."),
    "hub_vision_candidate_eligible": ("gauge", "1 = vision candidate passes every eligibility rule (0 = rejected, see reject_code)."),
    "hub_vision_candidate_reject_code": ("gauge", "1 = the vision candidate was rejected, labeled with the rule code."),
    "hub_vision_candidate_selected": ("gauge", "1 = this candidate is the current vision route choice."),
    "hub_vision_choice_score": ("gauge", "Current vision route choice score (0..100)."),
    "hub_vision_state_age_seconds": ("gauge", "Seconds since the hub last saw each server's state (the route is only as fresh as this)."),
    "hub_vision_routes_total": ("counter", "Vision route resolutions, lifetime (increments on state change or first observation)."),
}


def prom_text():
    L = []
    for name, (typ, help) in PROM_META.items():
        L.append(f"# HELP {name} {help}")
        L.append(f"# TYPE {name} {typ}")
    L.append("# --- llm-hub fleet ---")
    L.append("hub_servers_total " + str(len(SERVERS)))
    for s in SERVERS:
        lab = f'{{server="{s.name}",kind="{s.kind}"}}'
        L.append(f"hub_server_online{lab} {1.0 if s.online else 0.0}")
        if s.gpus:
            L.append(f'hub_server_gpu_count{{server="{s.name}"}} {len(s.gpus)}')
            L.append(f'hub_server_gpus_stale{{server="{s.name}"}} '
                     f"{1.0 if s.gpus_stale else 0.0}")
        # Honest stale/absent: while the sidecar is unreachable (host off / down
        # / wedge), the last sample is FROZEN, not live. Emit the per-GPU point
        # gauges only while fresh; when stale, drop them (a Grafana gap = "no
        # observation") and let hub_server_gpus_stale carry the meaning. A
        # frozen value must never masquerade as a current reading — the failure
        # behind the 2026-09-19 flat-GPU incident. (idle != 0; absent == no data.)
        if s.gpus and not s.gpus_stale:
            for g in s.gpus:
                gl = f'{{server="{s.name}",gpu="{g.get("name", "")[:40]}",idx="{g.get("index", "")}"}}'
                _g(L, "hub_gpu_utilization_pct", gl, g.get('util_pct'))
                _g(L, "hub_gpu_memory_used_mib", gl, g.get('mem_used_mib'))
                _g(L, "hub_gpu_memory_total_mib", gl, g.get('mem_total_mib'))
                _g(L, "hub_gpu_temp_c", gl, g.get('temp_c'))
                _g(L, "hub_gpu_power_w", gl, g.get('power_w'))
        for m in s.models.values():
            ml = f'{{server="{s.name}",model="{m["id"]}"}}'
            L.append(f"hub_model_loaded{ml} {1.0 if m.get('loaded') else 0.0}")
            _g(L, "hub_model_tokens_per_second", ml, m.get('tgen'))
            if m.get("kind") == "vllm":
                # prompt WORK, as a Prometheus counter (monotonic per engine
                # lifetime; engine restart / mux switch = counter reset in
                # the Prometheus sense). Consumers pick their own window:
                #   rate(hub_model_prompt_tokens_computed_total[1m])
                # Prefix-cache hits are EXCLUDED. (Retires the old
                # hub_model_prompt_tokens_per_second 2 s wall-rate gauge,
                # whose lump arrivals read 5,000 t/s for a 28 s prefill —
                # see reports/2026-09-30-vllm-prefill-metrics-semantics.md.)
                _g(L, "hub_model_prompt_tokens_computed_total", ml,
                   m.get('prompt_tokens_computed_raw'))
                _g(L, "hub_model_prompt_compute_throughput_tokens_per_second",
                   ml, m.get('pp_tput_1m'))
                _g(L, "hub_model_prefill_in_flight", ml,
                   1.0 if (m.get('live_prefill') or {}).get('active') else 0.0)
                _g(L, "hub_model_prefill_speed_age_seconds", ml,
                   m.get('prefill_speed_age'))
            else:
                # llama.cpp: child-clock rate (prompt tokens / the engine's
                # own prompt seconds) — physically bounded, keep the name
                _g(L, "hub_model_prompt_tokens_per_second", ml, m.get('tpp'))
            if m.get("kind") == "vllm":
                if m.get("loaded"):
                    L.append(f"hub_vllm_metrics_ok{ml} "
                             f"{1.0 if m.get('metrics_ok') else 0.0}")
                _g(L, "hub_model_requests_running", ml, m.get('req_running'))
                _g(L, "hub_model_requests_waiting", ml, m.get('req_waiting'))
                _g(L, "hub_model_kv_cache_used", ml, m.get('kv_used'))
                _g(L, "hub_model_kv_used_tokens", ml, m.get('kv_used_tokens'))
                kpool = m.get("kv_pool") or {}
                _g(L, "hub_model_kv_pool_tokens", ml, kpool.get('size_tokens'))
                _g(L, "hub_model_kv_pool_blocks", ml, kpool.get('num_blocks'))
                _g(L, "hub_model_kv_pool_max_concurrency", ml,
                   kpool.get('max_concurrency'))
                for wl, pv in (m.get("kv_peaks") or {}).items():
                    if pv is not None:
                        L.append(f'hub_model_kv_peak{{server="{s.name}",'
                                 f'model="{m["id"]}",window="{wl}"}} {pv}')
                lat = m.get("latency") or {}
                for key in ("ttft", "tpot", "itl", "e2e", "queue"):
                    q = lat.get(key) or {}
                    for qn in ("p50", "p95", "p99"):
                        _g(L, f"hub_model_{key}_{qn}_seconds", ml, q.get(qn))
                ph = m.get("phase") or {}
                if ph.get("window_s"):
                    pf, dd, pk = ph.get("prefill"), ph.get("decode"), ph.get("pfkv")
                    # p50/p95, not "avg": these are quantiles of the
                    # per-request phase-time histogram (renamed from
                    # *_avg_seconds 2026-09-30 — the old label implied a
                    # mean the hub never computed)
                    _g(L, "hub_model_prefill_p50_seconds", ml,
                       pf.get("p50") if pf else None)
                    _g(L, "hub_model_prefill_p95_seconds", ml,
                       pf.get("p95") if pf else None)
                    _g(L, "hub_model_decode_p50_seconds", ml,
                       dd.get("p50") if dd else None)
                    _g(L, "hub_model_decode_p95_seconds", ml,
                       dd.get("p95") if dd else None)
                    _g(L, "hub_model_prefill_computed_tokens_window", ml,
                       pk.get("sum") if pk else None)
                    # PREFFILL speed: computed KV tokens / vLLM's PREFILL-
                    # phase seconds over completed requests in the window
                    # — the performance figure (vs the throughput gauges
                    # above, which are work per wall-clock second)
                    _g(L, "hub_model_prefill_speed_tokens_per_second", ml,
                       ph.get("prefill_speed"))
                    _g(L, "hub_model_decode_speed_tokens_per_second", ml,
                       ph.get("decode_speed"))
                spec = m.get("spec") or {}
                _g(L, "hub_model_spec_acceptance", ml, spec.get("acceptance"))
                _g(L, "hub_model_spec_accept_length", ml, spec.get("mean_len"))
                rs = m.get("req_shape") or {}
                _g(L, "hub_model_prompt_tokens_mean", ml, rs.get("avg_in"))
                _g(L, "hub_model_generation_tokens_mean", ml, rs.get("avg_out"))
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
                # normalize stale snapshot shapes: the vLLM entry was
                # created without kind/desc by the v1-era setdefault and
                # a restored dict suppresses the current creation
                # defaults (setdefault never overwrites), which sent the
                # Prometheus export down the llama branch and hid the
                # vision advisor's kind. Re-derive from the server kind.
                m.setdefault("kind", "vllm" if s.kind in ("vllm", "vllm-mux") else "llama.cpp")
                # desc is config-owned metadata, not a measurement: a hint
                # change must reach a restored dict (same rule as the pollers'
                # per-poll refresh), so prefer the current hint over the
                # stored value.
                hint = (s.model_hints or {}).get(m["id"]) or {}
                m["desc"] = hint.get("desc") or m.get("desc") or s.desc
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
            # sparkline pp lane: llama.cpp child-clock rate; vLLM the 1-min
            # prompt-compute throughput (the headline cell is the separate
            # phase-clock prefill speed — the two branches stay separate)
            tpp_sum = sum(((m.get("pp_tput_1m") if m.get("kind") == "vllm"
                            else m.get("tpp")) or 0)
                          for m in s.models.values())
            gpu_max = max((g.get("util_pct") or 0) for g in s.gpus) if s.gpus else None
            ttft_p95_ms = None
            for m in s.models.values():
                ttft = (m.get("latency") or {}).get("ttft") or {}
                if ttft.get("p95") is not None:
                    ttft_p95_ms = round(ttft["p95"] * 1000)
                    break
            s.ring_push(tgen_sum if tgen_sum > 0 else None, gpu_max,
                        ttft_p95_ms, tpp_sum if tpp_sum > 0 else None)
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
        if c.get("kind") not in ("llama-router", "vllm", "vllm-mux", "gpu-only"):
            errs.append(f"{n}: kind must be 'llama-router', 'vllm', 'vllm-mux' or 'gpu-only'")
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
