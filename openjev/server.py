"""openjev — local Jev-compatible decision endpoint + playground.

Endpoints:
  GET  /                playground web UI
  GET  /api/health      instrument status (state: loading|ready, load, last call, rss)
  GET  /api/usecases    curated use-case catalog
  POST /v1/systemone    Jev-schema compatible API (drop-in for TypeSafe /v1/systemone)
  POST /api/bench       speed benchmark (runs x n_questions)
  GET  /api/agent/models   llm-hub fleet, priority-ordered (prompter picker)
  POST /api/agent/chat     streamed classifier-design chat (SSE) against a hub model

Boot model: the server binds immediately; the Laya model preloads in a
background thread (tens of seconds with a warm HF cache, longer when the
model still has to be downloaded). While loading, /v1/systemone and /api/bench
answer 503 {state: loading} — the UI rail shows the power-on state instead
of the whole site 502-ing through a proxy.

Configuration (env, all optional):
  OPENJEV_DATA             saved-use-case file (default: <this dir>/data/usecases.json)
  OPENJEV_HUB_URL          llm-hub base URL (prompter only; without it /api/agent/* is 503)
  OPENJEV_HUB_TOKEN_FILE   file holding the hub bearer token (default: ~/.llm-hub-token)
  OPENJEV_HUB_PRIORITY     priority per hub server: compact "name:1,name:2" list (unit-friendly) or JSON map
  OPENJEV_LAN_PREFIXES     comma-separated URL prefixes treated as directly reachable
"""
import json
import os
import re
import statistics
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import laya
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from usecases import USECASES

_agent = None
_load_sec = None
_last_ms = None
_state = "loading"
_boot = time.time()


def _preload():
    global _agent, _load_sec, _state
    t0 = time.time()
    _agent = laya.load("convaiinnovations/laya")
    _load_sec = round(time.time() - t0, 1)
    _state = "ready"


# ---------------------------------------------------------------------------
# Saved use cases: JSON persistence so classifiers can be created/edited/run
# by id — for agents, debugging, and reuse by other homelab services.
# Builtin cases (usecases.py) are read-only; saving one forks it.
# ---------------------------------------------------------------------------
SAVED_FILE = Path(os.environ.get("OPENJEV_DATA", str(Path(__file__).parent / "data" / "usecases.json")))
_saved = {}
_saved_lock = threading.Lock()


def _load_saved():
    global _saved
    try:
        data = json.loads(SAVED_FILE.read_text())
        if isinstance(data, dict):
            _saved = data
    except (FileNotFoundError, ValueError):
        _saved = {}


def _persist_saved():
    SAVED_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SAVED_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(_saved, indent=1))
    tmp.replace(SAVED_FILE)


def _slug(s):
    s = re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")
    return s or "case"


def _unique_id(base):
    uid, n = _slug(base), 2
    while uid in {u["id"] for u in USECASES} or uid in _saved:
        uid = f"{base}-{n}"
        n += 1
    return uid


@asynccontextmanager
async def lifespan(_app):
    _load_saved()
    threading.Thread(target=_preload, daemon=True).start()
    yield


app = FastAPI(title="openjev", lifespan=lifespan)


def _rss_mb():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024)
    except Exception:  # noqa: BLE001
        pass
    return None


def get_agent():
    if _state != "ready" or _agent is None:
        raise HTTPException(503, {"state": "loading",
                                  "detail": "model still loading (first boot can take a few minutes); retry shortly"})
    return _agent


def normalize_state(state):
    """Jev state = string | dict | message list → Laya state (string | dict)."""
    if isinstance(state, (str, dict)):
        return state
    if isinstance(state, list):
        parts = []
        for m in state:
            if isinstance(m, dict):
                parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
            else:
                parts.append(str(m))
        return "\n".join(parts)
    return str(state)


def normalize_questions(questions):
    """Accept Jev-style or Laya-style question maps; emit Laya grammar.

    Laya score questions require `criteria` (list); Jev-style clients call the
    same thing `labels`. Without this mapping a `labels`-only score question
    reaches Laya with criteria=None and crashes enumerate() (500). The
    prompter's model writes `labels` sometimes, so accept all three keys.
    """
    out = {}
    for name, q in questions.items():
        q = dict(q)
        for k in ("options", "labels"):
            if k in q and "criteria" not in q:
                q["criteria"] = q.pop(k)
        q.pop("options", None)
        q.pop("labels", None)
        crit = q.get("criteria")
        t = q.get("type")
        if t == "choice" and isinstance(crit, list):
            q["criteria"] = {str(c): str(c) for c in crit}
        if t in ("choice", "score") and not crit:
            raise HTTPException(422, f"question '{name}' ({t}) needs criteria/options/labels")
        if t == "score" and not isinstance(crit, list):
            q["criteria"] = [str(c) for c in crit]
        out[name] = q
    return out


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(Path(__file__).parent / "ui" / "index.html", media_type="text/html")


@app.get("/api/health")
def health():
    return {
        "model": "laya-rl-agent",
        "params_m": 421,
        "state": _state,
        "load_sec": _load_sec,
        "last_ms": _last_ms,
        "uptime_sec": round(time.time() - _boot),
        "rss_mb": _rss_mb(),
    }


@app.get("/api/usecases")
def usecases():
    """Catalog: builtin cases + saved (user/agent-created) ones, each with an origin."""
    out = [dict(u, origin="builtin") for u in USECASES]
    with _saved_lock:
        out += [dict(u, origin="saved") for u in _saved.values()]
    return out


@app.get("/api/usecases/{uid}")
def get_usecase(uid):
    for u in USECASES:
        if u["id"] == uid:
            return dict(u, origin="builtin")
    with _saved_lock:
        if uid in _saved:
            return dict(_saved[uid], origin="saved")
    raise HTTPException(404, f"no use case '{uid}'")


@app.post("/api/usecases")
def create_usecase(body: dict):
    """Save a classifier. Body: {id?, title?, summary?, state, questions}."""
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise HTTPException(422, "need non-empty 'questions' map")
    try:
        normalize_questions(questions)  # validates grammar (raises 422)
    except HTTPException:
        raise
    if body.get("state") is None:
        raise HTTPException(422, "need 'state' (the default/example input)")
    if body.get("id") and any(u["id"] == body["id"] for u in USECASES):
        raise HTTPException(409, f"'{body['id']}' is a builtin case — pick a different id to fork it")
    uid = _unique_id(body.get("id") or body.get("title") or "saved-case")
    entry = {
        "id": uid,
        "title": str(body.get("title") or uid.replace("-", " ").title()),
        "summary": str(body.get("summary") or ""),
        "state": body["state"],
        "questions": questions,
    }
    with _saved_lock:
        _saved[uid] = entry
        _persist_saved()
    return dict(entry, origin="saved")


@app.put("/api/usecases/{uid}")
def update_usecase(uid, body: dict):
    with _saved_lock:
        if uid not in _saved:
            if any(u["id"] == uid for u in USECASES):
                raise HTTPException(409, f"'{uid}' is a builtin case — pick a new id to fork it")
            raise HTTPException(404, f"no saved use case '{uid}'")
    if body.get("state") is None:
        raise HTTPException(422, "need 'state'")
    questions = body.get("questions")
    if questions is not None:
        if not isinstance(questions, dict) or not questions:
            raise HTTPException(422, "need non-empty 'questions' map")
        try:
            normalize_questions(questions)
        except HTTPException:
            raise
    with _saved_lock:
        e = dict(_saved[uid])
        e["state"] = body["state"]
        if body.get("title") is not None:
            e["title"] = str(body["title"])
        if body.get("summary") is not None:
            e["summary"] = str(body["summary"])
        if questions is not None:
            e["questions"] = questions
        _saved[uid] = e
        _persist_saved()
    return dict(e, origin="saved")


@app.delete("/api/usecases/{uid}")
def delete_usecase(uid):
    with _saved_lock:
        if uid not in _saved:
            if any(u["id"] == uid for u in USECASES):
                raise HTTPException(409, f"'{uid}' is a builtin case and cannot be deleted")
            raise HTTPException(404, f"no saved use case '{uid}'")
        del _saved[uid]
        _persist_saved()
    return {"deleted": uid}


@app.post("/api/run")
def run_by_id(body: dict):
    """Run a named classifier on a state — the one-call form for agents and
    other homelab services: {usecase: "beauty-review", state: "..."}."""
    uid = body.get("usecase")
    state = body.get("state")
    if not uid or state is None:
        raise HTTPException(422, "need 'usecase' (id) and 'state'")
    uc = next((u for u in USECASES if u["id"] == uid), None)
    if uc is None:
        with _saved_lock:
            uc = _saved.get(uid)
    if uc is None:
        raise HTTPException(404, f"no use case '{uid}' — see GET /api/usecases")
    try:
        res = _run_systemone(state, uc["questions"])
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"model error: {e}")
    if isinstance(res, dict):
        res["usecase"] = uid
    return JSONResponse(res)


def _run_systemone(state, questions):
    """Shared core: normalize (Jev -> Laya grammar) + one forward pass."""
    t0 = time.time()
    res = get_agent().system_one(normalize_state(state), normalize_questions(questions))
    if isinstance(res, dict):
        res["server_ms"] = round((time.time() - t0) * 1000)
        global _last_ms
        _last_ms = res["server_ms"]
    return res


@app.post("/v1/systemone")
async def systemone(request: dict):
    """Jev-schema compatible. Request: {model?, state, questions} → {answers, usage, ...}."""
    state = request.get("state")
    questions = request.get("questions")
    if state is None or not isinstance(questions, dict) or not questions:
        raise HTTPException(422, "need 'state' and non-empty 'questions' map")
    try:
        return JSONResponse(_run_systemone(state, questions))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — surface model errors to the client
        raise HTTPException(500, f"model error: {e}")


@app.post("/v1/batch")
async def systemone_batch(request: dict):
    """One questions map over many states — for review mining etc.

    {states: [...], questions: {...}} → {results: [{state, answers, usage,
    server_ms, error?}], total_ms}. States run sequentially (CPU box), cap 25
    (25 × ~1–5 s). A bad state errors out per-entry without killing the batch.
    """
    states = request.get("states")
    questions = request.get("questions")
    if not isinstance(states, list) or not states:
        raise HTTPException(422, "need non-empty 'states' list")
    if not isinstance(questions, dict) or not questions:
        raise HTTPException(422, "need 'questions' map")
    t0 = time.time()
    results = []
    for s in states[:25]:
        t = time.time()
        entry = {"state": s}
        try:
            r = _run_systemone(s, questions)
            entry.update({"answers": r.get("answers"), "usage": r.get("usage"),
                          "server_ms": round((time.time() - t) * 1000)})
        except Exception as e:  # noqa: BLE001 — per-state isolation
            entry["error"] = str(e)
        results.append(entry)
    return {"model": "laya-rl-agent", "results": results, "total_ms": round((time.time() - t0) * 1000)}


@app.post("/api/bench")
async def bench(request: dict):
    runs = min(int(request.get("runs", 5)), 25)
    nq = int(request.get("n_questions", 10))
    assert 1 <= nq <= 100
    state = (
        "Ticket #7741: The nightly export job finished in 41 minutes instead of the usual 8. "
        "The warehouse dashboard shows stale numbers; the finance team is blocked on the month-end report. "
        "Ops suspects a connection-pool exhaustion after last week's patch. No user-facing impact yet."
    )
    questions = {
        f"q{i}": {"type": "noul", "instructions": f"Does this ticket mention issue facet {i}?"}
        for i in range(nq)
    }
    agent = get_agent()
    times = []
    usage = None
    for _ in range(runs):
        t0 = time.time()
        res = agent.system_one(state, questions)
        dt = (time.time() - t0) * 1000
        times.append(round(dt, 1))
        if usage is None and isinstance(res, dict):
            usage = res.get("usage")
    times_sorted = sorted(times)

    def pct(p):
        k = max(0, min(len(times_sorted) - 1, int(round(p / 100 * (len(times_sorted) - 1)))))
        return round(times_sorted[k], 1)

    return {
        "n_questions": nq,
        "runs": times,
        "median_ms": round(statistics.median(times), 1),
        "p95_ms": pct(95),
        "min_ms": round(min(times), 1),
        "per_question_median_ms": round(statistics.median(times) / nq, 1),
        "usage": usage,
        "model_load_sec": _load_sec,
    }



# ---------------------------------------------------------------------------
# Prompter agent: designs classifiers by chat, served by llm-hub models.
# ---------------------------------------------------------------------------

# The prompter (classifier-design chat) is served by a larger model behind an
# llm-hub (see ../hub) or any fleet manager exposing GET /api/servers and
# GET /api/models. Everything is optional: without OPENJEV_HUB_URL the
# /api/agent/* routes answer 503 and the UI's prompter panel reports that.
HUB_URL = os.environ.get("OPENJEV_HUB_URL", "")
HUB_TOKEN_FILE = Path(os.environ.get("OPENJEV_HUB_TOKEN_FILE", str(Path.home() / ".llm-hub-token")))
# Display/selection priority per hub server name (lower = preferred).
# Compact list "name:1,name:2" (unit-friendly: no quotes, no spaces) or a JSON map.
# URL prefixes that count as directly reachable from this process (e.g. the
# local LAN range, when this box can't use Tailscale). Models served only
# outside these prefixes are shown but disabled. Empty = all http(s) reachable.
_LAN_PREFIXES = [p for p in os.environ.get("OPENJEV_LAN_PREFIXES", "").split(",") if p]


def _parse_priority(spec):
    spec = (spec or "").strip()
    if not spec:
        return {}
    try:
        v = json.loads(spec)
        if isinstance(v, dict):
            return {str(k): int(x) for k, x in v.items()}
    except ValueError:
        pass
    out = {}
    for part in spec.split(","):
        name, _, prio = part.strip().partition(":")
        if name:
            out[name] = int(prio or 9)
    return out


HUB_PRIORITY = _parse_priority(os.environ.get("OPENJEV_HUB_PRIORITY", ""))

AGENT_SYSTEM = """You are the openjev prompter: a design assistant for one-pass classifiers.
The user describes a decision they want to automate; you discuss it with them and, when ready, emit a ready-to-run use case.
Every design rule below was measured on a real deployment of this model, not guessed.

TARGET RUNTIME (what your finished classifier runs on):
- Laya: a 421M-param bidirectional transformer (ModernBERT) with a decision head. It answers typed questions about a text state in ONE forward pass. It never generates text.
- Question types: "choice" (pick 1 of N options), "score" (place on an ordered scale), "noul" (calibrated yes/no probability 0-1).
- Hard budget: 512 tokens PER QUESTION (state + question + options combined). Keep states short.

WHAT IT DOES WELL (measured on this box):
- Short states (reviews, emails, tickets, comments, headlines, log lines) under ~400 tokens; ~0.5-5 s per call on 2 CPU cores.
- Commonsense semantics with operational definitions: sentiment polarity, spam/phishing, toxicity, intent, routing, "is X asked/mentioned".
- noul is its most reliable type: explicit-trigger yes/no with an explicit default anchor scores clean (measured 4/4 on a 4-text battery; a presence noul WITHOUT the anchor false-positived at 76% on an ordinary description).
- Calibrated confidence on every answer: high on clear cases, low when genuinely ambiguous. That honesty is a feature - low confidence is the model telling the truth.

WHAT IT DOES NOT DO (reframe or advise a bigger model; do not fight these):
- long documents (over the 512-token budget), arithmetic, multi-step/multi-hop reasoning, rare-jargon domains, span extraction (it only picks from the options you give it - precompute candidates and pass them as options).
- Subtle ABSTRACT judgments phrased without observable triggers (measured: "is this clickbait?" false-positived a dry headline at 85%; rewording around observable markers helped but never fully fixed it - such concepts may simply exceed the 421M, in which case say so).
- The base checkpoint sits near the majority-class baseline on hard zero-shot tasks; its headline benchmark score belongs to the fine-tuned one. Treat surprising zero-shot quality as a lucky question class, not the norm.

CONFIDENCE REGIMES (measured — the most important part of the design): whether confidence can gate is a property of the question's TRIGGER CLASS, not its wording.
1. CONCRETE-OBSERVABLE — the answer is readable from surface facts in the state ("is a refund explicitly requested?", "is the proposed action valid given these facts?"). Confidence is a real signal; per-use-case thresholds work. Design for this regime whenever the decision allows it.
2. ABSTRACT-COMPARATIVE — the answer requires comparing two descriptions against each other (claim vs fact, summary vs source). Measured: noul INVERTS (it answers the implicit "do these match?" — a type problem, not a wording one); 2-way is WORSE than 3-way (the escape option helps the model commit); an explicit subject prefix fixes labels but collapses confidence; and confidence CANNOT gate (correct answers arrive at 1-65% confidence). The reframe that restores the concrete regime: make the comparison OBSERVABLE (e.g. NLI on concrete value mismatches, not on vibes). In this regime the use case is a PRECISION-FIRST ALARM: confident-contradicted -> advisory flag for a human or the bigger tier; never auto-fix. Everything else routes to the bigger model or clears.
3. MULTI-HOP / SUBJECT-ALIASING — two names for one thing, or facts spread across documents. Out of capability at 421M. Route to the bigger model or a human; do not keep rewording.

TIER PLACEMENT (when the design pairs with a bigger model, e.g. a 27B): put the bigger model as a DOUBT-ARBITER — called only when the cheap tier is uncertain — never as a VETO in series with everything. A veto-placed big model stalls correct actions; a doubt-arbiter one waits only on visible failure evidence. Keep the big-model contract cheap: thinking off, single-word or very short answers.

DESIGN RULES (each one was measured, not guessed):
- One concept per question; snake_case names. "instructions": one operational sentence, unambiguous; no "you are an expert" filler.
- choice: 2-8 short mutually exclusive labels; key is "options" (list), plus a "criteria" map (label -> short description) whenever labels could be confused. 10 options is the practical ceiling - measured: clean cases degrade from ~90% to 35-70% as the option count grows; merge related labels instead of adding.
- Escape hatch: for CLASSIFICATION questions where the input may not match any option, add one explicit abstain option ("not_discussed", "none", "no match") so the model can abstain instead of inventing the least-wrong label (measured: a logistics-only review went from a 44% forced guess to 90% abstention). Omit it only for judgments where some answer always applies (severity, sentiment). Know the cost: the residual siphons some probability off even clean cases (measured 87% -> 68%) - tell the user their pipeline should gate on confidence.
- score: ONLY for true ordinals (severity, intensity) with clear rungs; 3-5 short UNNUMBERED levels low to high in "criteria" (numbered prefixes like "1: Very Negative" measurably degraded scale reading). NEVER for "presence of a feature": measured, a 3-level presence scale drifted to the middle level on every input under every wording, while the same concept as a noul scored 4/4.
- noul: "Yes only if <concrete trigger>; no otherwise - <typical no case> is no." The explicit default anchor is what makes presence-detection reliable (measured: with it, 1-3% on ordinary texts; without, 24-76%).
- Design so your "good" cases come out high-confidence. The API returns confidence per answer (the RLCD act_probability field is pinned at 1.0 - do not gate on it). The UI's default gate is 50% - but tell the user which regime applies (see above): gating is a control in the concrete-observable regime, and only an information channel in the comparative one.
- Always include a realistic short example "state" so the user can test immediately.

ITERATION PROTOCOL (the user runs your use case and reports back):
- Wrong answer at HIGH confidence = the question is mis-worded - fix the wording or split the question.
- Wrong answer at LOW confidence = the model was already telling you it was guessing - reframe rather than force: change the question TYPE before the wording (measured: a score->noul reframe fixed a case that three wording iterations failed on).
- If a design needs multi-hop reasoning, arithmetic, or documents over budget: say plainly that this 421M model cannot carry it and name what it would take (a fine-tune, a bigger model) - that is a useful answer, not a failure.

OUTPUT CONTRACT:
- When the design is ready, end your reply with a fenced block exactly in this form:
```usecase
{"state": "...", "questions": {"name": {"type": "noul", "instructions": "..."}}}
```
The UI offers a button to load that block into the editor. One usecase block per message; keep the JSON strictly valid. After the block, in one line, name the regime of each question (concrete-observable vs comparative) and say what to do with confidence accordingly (gate on it, or treat it as alarm-only and route undetermined answers to the bigger tier)."""


def _hub_get(path):
    if not HUB_URL:
        raise HTTPException(503, "llm-hub not configured (set OPENJEV_HUB_URL)")
    if not HUB_TOKEN_FILE.is_file():
        raise HTTPException(503, "llm-hub token not configured on this box")
    tok = HUB_TOKEN_FILE.read_text().strip()
    req = urllib.request.Request(HUB_URL + path, headers={"Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=6) as r:
        return json.load(r)


def _vllm_served_name(url):
    """vLLM models are anonymous in the hub ((vllm)); ask the endpoint itself."""
    if not url:
        return None
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=4) as r:
            data = json.load(r)
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        return ids[0] if ids else None
    except Exception:  # noqa: BLE001
        return None


@app.get("/api/agent/models")
def agent_models():
    """llm-hub fleet, priority-ordered; only directly-reachable, online, loaded rows are selectable."""
    servers = {s["name"]: s for s in _hub_get("/api/servers")["servers"]}
    out = []
    for m in _hub_get("/api/models")["models"]:
        srv = servers.get(m["server"], {})
        url = srv.get("url")
        model = m["id"]
        note = ""
        if model == "(vllm)" and url:
            model = _vllm_served_name(url) or "(vllm)"
        direct = not _LAN_PREFIXES or (bool(url) and any(url.startswith(p) for p in _LAN_PREFIXES))
        if not url or not direct:
            note = "not directly reachable from this box"
        elif not srv.get("online"):
            note = "server offline"
        elif not m.get("loaded"):
            note = "model not loaded"
        label = f"{m['server']} · {model}"
        if m.get("tgen"):
            label += f" · {m['tgen']:.0f} t/s"
        out.append({
            "key": f"{m['server']}/{m['id']}",
            "server": m["server"],
            "model": model,
            "label": label,
            "url": url,
            "loaded": m.get("loaded", False),
            "online": srv.get("online", False),
            "reachable": direct and bool(srv.get("online")) and bool(m.get("loaded")),
            "note": note,
            "priority": HUB_PRIORITY.get(m["server"], 9),
            "desc": (m.get("desc") or "")[:160],
        })
    out.sort(key=lambda x: (x["priority"], not x["loaded"], x["key"]))
    return out


@app.post("/api/agent/chat")
def agent_chat(request: dict):
    """Streamed OpenAI-compatible chat against the chosen hub model (SSE passthrough)."""
    msgs = request.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise HTTPException(422, "need a non-empty 'messages' list")
    models = {m["key"]: m for m in agent_models()}
    m = models.get(request.get("key", ""))
    if not m:
        raise HTTPException(404, "unknown model key")
    if not m["reachable"]:
        raise HTTPException(409, m["note"] or "model not currently reachable")
    payload = {
        "model": m["model"],
        "messages": [{"role": "system", "content": AGENT_SYSTEM}] + msgs[-12:],
        "stream": True,
        "temperature": 0.4,
        "max_tokens": 1600,
    }
    # qwen3.8-27b-dual is a Qwen3 thinking model: left alone it spends the whole
    # token budget on reasoning deltas and emits zero content (the 2026-09-19
    # "(no data received)" bug). A design chat doesn't need CoT, so switch it off.
    if "qwen" in m["model"].lower():
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        m["url"].rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — connection errors, 5xx, timeouts
        raise HTTPException(502, f"upstream {m['server']} error: {e}")

    def gen():
        finish = None
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if chunk.get("error"):
                    yield f"data: {json.dumps({'err': str(chunk['error'])})}\n\n"
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("reasoning"):
                    yield f"data: {json.dumps({'r': delta['reasoning']})}\n\n"
                if delta.get("content"):
                    yield f"data: {json.dumps({'t': delta['content']})}\n\n"
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
            yield f"data: {json.dumps({'done': True, 'finish': finish})}\n\n"
        finally:
            resp.close()

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8781, log_level="info")

