# LLM Hub — inference fleet control & observability

A single-file, stdlib-only Python service that watches a fleet of
llama.cpp / vLLM inference servers and gives it one control surface:
a dark terminal-style web UI, a JSON API for agents, and a
Prometheus-shaped `/metrics` export.

**The hub is never in the inference request path.** Clients talk directly to
the inference servers; the hub only watches them (and proxies load/unload
commands on demand). Losing the hub loses visibility, not inference.

## Files

| File | Purpose |
|------|---------|
| `llm-hub.py` | the hub (poller + HTTP API + Prometheus export + UI server) |
| `gpu-sidecar.py` | per-GPU-host `nvidia-smi` → JSON sidecar (the hub polls it) |
| `ui/index.html` | the web UI (single file, no build step, no framework) |
| `config.example.json` | config template |
| `llm-hub.service` | systemd unit for the hub |
| `gpu-sidecar.service` | systemd unit for the sidecar |
| `screenshots/` | UI screenshots (redacted) |

No dependencies beyond CPython stdlib. No Docker, no JS framework, no
database. One process per role.

## Quick start

```sh
mkdir -p hubrun && cd hubrun
cp ../config.example.json config.json && $EDITOR config.json   # your servers + port
head -c 32 /dev/urandom | od -An -tx1 | tr -d " \n" > token && chmod 600 token
python3 ../llm-hub.py
# or with the env overrides:  LLM_HUB_CONFIG=… LLM_HUB_TOKEN=… LLM_HUB_UI=… LLM_HUB_STATE=…
```

Then open `http://localhost:8443/` and paste the token into the unlock form.
For GPU telemetry, run `gpu-sidecar.py` on each GPU host first.
For the systemd layout, see [Deploy](#deploy).

## What it shows

Per server, every 2 s:

- **llama.cpp routers**: per-model tokens/s (generation) and
  prefill rate (child-clock based, so it's physically bounded and decays to
  idle after 30 s of no counter movement), MTP/spec-decode acceptance,
  5-min rolling prompt-cache hit rate, requests in-flight / deferred, busy
  decode slots. Load state from the router's own `/v1/models` catalog.
- **vLLM**: tokens/s, **latency percentiles (p50/p95/p99) for TTFT, TPOT,
  E2E, and queue time computed from the native histograms over a rolling
  5-minute window** (bucket deltas — not lifetime cumulative), request
  counts, KV cache usage, prefix-cache hit rate (window), preemptions
  (window + total), engine sleep state, and wait-reason breakdown.
- **GPUs**: per-card utilization, memory, temperature, power, from the
  per-host sidecar.

Model rows are collapsible: the collapsed line carries the rate + the key
load signals; expanding shows the full diagnostics block. Node state is
derived: `offline / idle / active / busy / degraded` — idle is the common
state and looks calm, degraded is reserved for real problems
(KV ≥ 90%, length-limited responses, preemptions under load).

## API

| Endpoint | Auth | Purpose |
|----------|------|---------|
| `GET /api/servers` | token/cookie | full fleet JSON (UI + agents) |
| `GET /api/models` | token/cookie | flat model list with server attribution (agents) |
| `POST /api/models/load` | token/cookie | `{server, model}` → proxies to router |
| `POST /api/models/unload` | token/cookie | same, unload |
| `POST /auth` | Bearer master token | exchange token for 7-day session cookie |
| `GET /auth` | token/cookie | convenience: is my session valid? |
| `GET /metrics` | public | Prometheus text (topology-level data) |
| `GET /health` | public | liveness + poller freshness |

### Auth model

- **Agents/CLI**: master Bearer token (one file, `chmod 600`).
- **Browser**: exchanges the master token once via `POST /auth` for an
  HMAC-signed `HttpOnly` session cookie (7-day sliding). The UI shell is
  public; data requires auth. The page itself never contains the token.

`/metrics` and `/health` are public by design — they expose server names,
model names, and rates (topology-level), not prompts or payloads. If that's
too much for your network, put the hub behind an auth proxy.

## Config

`/etc/llm-hub/config.json` (see `config.example.json`):

```json
{
  "port": 8443,
  "servers": [
    {"name": "…", "kind": "llama-router", "url": "http://host:8081",
     "sidecar": "http://host:9421", "description": "…"},
    {"name": "…", "kind": "vllm", "url": "http://host:8080",
     "sidecar": "http://host:9421", "description": "…"},
    {"name": "…", "kind": "gpu-only", "sidecar": "http://host:9421",
     "description": "GPU telemetry only"}
  ]
}
```

`kind` is `"llama-router"` (a llama.cpp server exposing the OpenAI-style
`/v1/models` catalog + per-model `/metrics?model=`), `"vllm"`, or
`"gpu-only"` (no inference API; online state comes from the sidecar).
`sidecar` is the per-host `gpu-sidecar` URL (default assumption: same host
as the server). A `gpu_sidecars` list of `{server, url}` entries may also be
used; it overrides the per-server keys.

Per-model hints are supported via a `models` object on a server entry:

```json
{"name": "…", "kind": "llama-router", "url": "…",
 "models": {"some-model": {"ctx": 262144, "desc": "notes"}}}
```

`ctx` is otherwise taken from the router's `--ctx-size` launch args when
available.

### Environment overrides (testing)

| Var | Default |
|-----|---------|
| `LLM_HUB_CONFIG` | `/etc/llm-hub/config.json` |
| `LLM_HUB_TOKEN` | `/etc/llm-hub/token` |
| `LLM_HUB_STATE` | `/var/lib/llm-hub` |
| `LLM_HUB_UI` | `/opt/llm-hub/ui` |
| `LLM_HUB_LAT_WINDOW` | `300` (percentile window seconds) |

## Deploy

```sh
# hub host
mkdir -p /opt/llm-hub/ui /etc/llm-hub /var/lib/llm-hub
install llm-hub.py /opt/llm-hub/
install ui/index.html /opt/llm-hub/ui/
install config.json /etc/llm-hub/
install -m 600 token /etc/llm-hub/token
systemctl enable --now llm-hub

# each GPU host
install gpu-sidecar.py /opt/llm-hub/
systemctl enable --now gpu-sidecar
```

The hub is unprivileged; put TLS in front (reverse proxy) if you expose it.

## Prometheus export

`/metrics` emits `hub_*` gauges: server online state, per-GPU
utilization/memory/temperature/power, per-model tokens/s, vLLM latency
percentiles (seconds), request counts, KV usage, cache hit rates,
preemptions, and engine sleep state. Idle/unknown values are `NaN`, not `0`.
This is the stable scrape surface for a Grafana stack (separate concern —
the hub does not require Prometheus).

## State

In-memory: a 30-minute sparkline ring and 5-minute rolling windows per
server, plus a periodic `snapshot.json` in the state dir (restored on boot
so the UI isn't blank after a hub restart). No historical storage — for
long-term trends, scrape `/metrics`.

## Security assumptions

- Designed for a **trusted homelab / admin network**. Not intended to be
  internet-exposed without an external secure access layer (reverse proxy
  with TLS + auth, VPN, etc.).
- One master token protects the whole fleet (read + load/unload control).
  Rotate by replacing the token file and restarting.
- The session cookie is HMAC-signed with the master token: anyone who
  obtains the token file can mint valid cookies.
- Load/unload is a **proxied passthrough** to the router's native
  `/models/load|/models/unload` — same capability as calling the router
  directly, audited to `audit.log` in the state dir.
- No CSRF token: state-changing endpoints accept `Authorization: Bearer`
  or the `HttpOnly` session cookie (SameSite=Lax), which covers the
  same-origin UI and agent usage; a cross-origin form post to a GET-less
  POST endpoint is the classic residual gap if you expose the hub
  publicly.
- Secrets never leave the hub host: the API never forwards upstream
  payloads beyond a 500-char excerpt of the load/unload response.

## Known limitations

- **History is intentionally in-memory and short-lived**: a 30-min
  sparkline ring and 5-min rolling windows per server. After a hub restart,
  only the last `snapshot.json` is restored. For long-term trends, scrape
  `/metrics`.
- vLLM latency percentiles need at least one observation inside the rolling
  window; with no recent requests they show `—` (no stale values).
- vLLM is treated as one logical model per server (`(vllm)`); multi-model
  vLLM deployments are not distinguished.
- Percentiles are interpolated from histogram bucket deltas — resolution is
  bounded by the engine's bucket edges.
- The UI is a single page with a full 2 s refresh; it is built for a few
  dozen nodes, not hundreds.
- No TLS of its own — put a reverse proxy in front for anything beyond a
  trusted LAN/tailnet.

## Screenshots

- [Fleet view (desktop)](screenshots/llm-hub-fleet.png) — one card per
  server: GPU rows, per-model rates, derived state (IDLE/ACTIVE/BUSY/…).
- [Expanded diagnostics](screenshots/llm-hub-diagnostics.png) — click a
  model name for the latency percentile table (TTFT/TPOT/E2E/queue ×
  p50/p95/p99 over the 5-min window), finish reasons, preemptions, cache.
- [Mobile layout](screenshots/llm-hub-mobile.png).

Node names/descriptions in the screenshots are redacted.
