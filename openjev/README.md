# openjev — local Jev-compatible one-pass decision endpoint + playground

A self-hosted stand-in for TypeSafe's API-only **Jev / System One** model:
unstructured text *state* in → typed probabilistic *decisions* out
(`choice` / `score` / `noul`), in one forward pass. The model is
[Laya](https://huggingface.co/convaiinnovations/laya) (ModernBERT-large 395M backbone +
RLCD-trained decision head, 421M params, Apache-2.0) — it *scores*, it never generates
text (`usage.output_tokens` is always 0). Typical call: 0.5–5 s on 2 CPU cores.

This directory is the **service implementation**. The companion surfaces in this repo:

- [docs/decision-classifiers.md](../docs/decision-classifiers.md) — the *methodology*:
  the measured design rules, the confidence-regime map, tier placement, calibration and
  the fine-tune path. The playground's built-in **prompter** carries an operational
  distillation of that document as its system prompt (server-side, see `AGENT_SYSTEM`
  in `server.py`); when a finding lands in the doc, the prompter prompt and the
  built-in use cases below are where it becomes actionable.
- [agents/jev-designer](../agents/jev-designer/SKILL.md) — the agent-facing skill view
  of the same record.
- [reports/2026-09-21-laya-doc-staleness.md](../reports/2026-09-21-laya-doc-staleness.md)
  — the canonical comparative-regime case study (the `doc-stale` saved use case).

## Layout

| File | What |
|---|---|
| `server.py` | The whole service: FastAPI app, Laya preload, saved-use-case persistence, the Jev-schema API, the prompter passthrough. Stdlib + `laya`/`fastapi`/`uvicorn` only. |
| `usecases.py` | Curated built-in use cases (read-only; saving one forks it under a new id). |
| `ui/index.html` | The playground — single page, no build step: case catalog, freeform editor, prompter chat, latency benchmark, confidence-gate control. Served straight from disk (UI-only changes need no restart). |
| `data/usecases.json` | Created on first save (atomic write); survives restarts. Not in git. |

## API

| Endpoint | What |
|---|---|
| `GET /` | Playground UI |
| `GET /api/health` | `state: loading\|ready`, load time, last-call ms, RSS |
| `GET /api/usecases` · `GET /api/usecases/{id}` | Catalog (each entry has `origin: builtin\|saved`) |
| `POST /api/usecases` · `PUT/DELETE /api/usecases/{id}` | Persist / edit / delete saved cases (grammar-validated; builtins 409) |
| `POST /api/run` | `{"usecase": "<id>", "state": "…"}` → Jev-schema answer — the one-call form for other services/agents |
| `POST /v1/systemone` | **Jev drop-in**: `{"state", "questions"}` inline — same request/response shape as TypeSafe's API; repoint any Jev client's base URL here. Normalizes Jev-style input (message-list state, `options`/`labels`/`criteria` keys, choice-list questions) to Laya grammar. |
| `POST /v1/batch` | One `questions` map over many `states` (sequential, cap 25, per-state error isolation) |
| `POST /api/bench` | Latency table (per-run ms, median, p95, per-question cost) |
| `GET /api/agent/models` | Design-model fleet from the hub, priority-ordered |
| `POST /api/agent/chat` | The **prompter**: `{key, messages}` → SSE stream; the model designs a classifier and ends with a fenced `usecase` block the UI can apply |

**Gate on `confidence` — within its regime.** The confidence regime of a question
(concrete-observable vs comparative vs multi-hop) decides whether that number can
gate at all; the regime map and the measured thresholds live in
[docs/decision-classifiers.md](../docs/decision-classifiers.md). The RLCD
`act_probability` field is pinned at 1.0 — never gate on it.

## Configuration (env)

| Variable | Default | Meaning |
|---|---|---|
| `OPENJEV_DATA` | `<app dir>/data/usecases.json` | Where saved use cases persist |
| `OPENJEV_HUB_URL` | *(unset → prompter disabled)* | llm-hub base URL (see [../hub](../hub/README.md)); serves the prompter's design models |
| `OPENJEV_HUB_TOKEN_FILE` | `~/.llm-hub-token` | File with the hub bearer token (chmod 600) |
| `OPENJEV_HUB_PRIORITY` | `{}` | JSON map server-name → priority for the prompter's model picker (lower = preferred) |
| `OPENJEV_LAN_PREFIXES` | *(empty = everything http(s) reachable)* | Comma-separated URL prefixes counted as directly reachable from this process — models outside them (e.g. Tailscale-only, when the host has no TUN) are shown but disabled |

`HF_HUB_OFFLINE=1` on the unit is deliberate: it bounds the cold-boot window
(a network-checked HuggingFace boot once took 165 s; with a warm cache the model
preloads in ~30 s).

## Running

```bash
python3 -m venv venv
# CPU box: install the CPU torch wheel first (the default wheel ships CUDA and is ~900 MB and useless here)
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install -r requirements.txt
```

Then run it under any supervisor. Example systemd unit (the deployment in this
lab is a 2C/4G unprivileged LXC, `MemoryMax=3.5G` — the venv + 1 GB model needs
the headroom; a 2G unit got OOM-killed):

```ini
[Unit]
Description=openjev — local Jev-compatible decision endpoint (Laya) + playground
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/opt/openjev/app
Environment=LANG=C.UTF-8
Environment=HF_HUB_DISABLE_TELEMETRY=1
Environment=HF_HUB_DISABLE_PROGRESS_BARS=1
Environment=HF_HUB_OFFLINE=1
# prompter (all optional — without HUB_URL the /api/agent/* routes answer 503)
Environment=OPENJEV_HUB_URL=http://<hub>:8443
Environment=OPENJEV_HUB_PRIORITY={"vllm-3090": 1}
# Environment=OPENJEV_LAN_PREFIXES=http://<lan-network>
ExecStart=/opt/openjev/venv/bin/python /opt/openjev/app/server.py
Restart=on-failure
RestartSec=5
MemoryMax=3.5G

[Install]
WantedBy=multi-user.target
```

No auth by default — this is a LAN/Tailscale-only instrument, by design. If you
expose it publicly, put a Bearer-token middleware in front of all routes first
(the [hub](../hub/README.md) token pattern is the template).

## Deployment notes (from running this)

- The server binds before the model is loaded; inference routes answer
  `503 {"state": "loading"}` until ready — keep any reverse proxy in front
  tolerant of that, the UI itself polls `/api/health`.
- `if __name__ == "__main__"` must stay at the **end** of `server.py` —
  `uvicorn.run()` blocks; code appended after it silently 404s.
- A 500 from this service carries its diagnostic in the HTTP detail string,
  not in the journal (FastAPI `HTTPException` path). Read the detail.
