#!/usr/bin/env python3
"""Adaptive vision routing policy for llm-hub (pure decision logic).

The llm-hub advisor (`GET /api/vision/route`) recommends the best currently
usable vision endpoint for a consumer that is about to send an OpenAI-compatible
multimodal request. This module is the policy: it is deliberately **pure** —
no HTTP, no logging, no locking, no config loading — so the decision logic is
unit-testable in isolation (see test_vision_routing.py) and future backends
(vLLM/SGLang vision) plug in through the normalized candidate shape, not through
llama.cpp metric names.

The advisor performs **zero backend mutation**: it never loads/unloads models,
saves/restores slots, or preempts requests. It reads asynchronously collected
state (the hub polls every ~2 s) and recommends; the consumer sends the actual
request directly to the chosen endpoint and keeps its own static emergency
fallback for when the hub is unreachable. (On a `--models-max 1` router,
auto-loading one model implicitly evicts the other, so even "just restore the
preferred vision model" is a fleet mutation requiring its own designed
policy — deliberately out of scope here.)

Design: hard eligibility filters, then a two-class capacity split
(IMMEDIATE outranks QUEUED, always), then a simple ordering (tier, GPU
contention, load). Not a tuned scoring DSL — the motivating outage was a
knowledge failure ("the model isn't loaded", "the fallback's only slot is
pinned by a 68K prefill"), not an optimization failure.

File name uses an underscore (importable module); the repo's dashed filenames
(llm-hub.py, gpu-sidecar.py) are direct-execution scripts, not modules.
"""

POLICY_VERSION = 1

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

DEFAULT_POLICY = {
    "policy_version": POLICY_VERSION,
    "enabled": True,
    # reject candidates whose server state is older than this (seconds).
    # The hub polls every 2 s; the poll loop can stretch when a backend
    # hangs, so a route must never sit on state that's minutes old.
    "stale_after_s": 60,
    # sidecar GPU utilization (%) at/above which a candidate's accelerator is
    # considered contended (ordering penalty, never a hard exclusion — the
    # llama.cpp process may still have free capacity on a busy card).
    "gpu_busy_threshold": 80.0,
}

# Order-of-magnitude sanity bounds for validate_policy() — this is a
# homelab advisor, not a config compiler; anything inside the bounds is
# accepted, anything outside is a typo.
_POLICY_BOUNDS = {
    "stale_after_s": (1, 3600),
    "gpu_busy_threshold": (1, 100),
}


def validate_policy(policy):
    """Return a normalized copy of `policy` (defaults merged), or raise
    ValueError on type/bounds violations. Pure — safe to call per request."""
    if not isinstance(policy, dict):
        raise ValueError("policy must be a dict")
    p = dict(DEFAULT_POLICY)
    p.update(policy)
    if not isinstance(p["enabled"], bool):
        raise ValueError("vision.enabled must be a boolean")
    for key, (lo, hi) in _POLICY_BOUNDS.items():
        v = p[key]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi:
            raise ValueError(f"vision.{key} must be a number in [{lo}, {hi}]")
    if not isinstance(p["policy_version"], int) or p["policy_version"] < 1:
        raise ValueError("policy_version must be a positive integer")
    return p


# ---------------------------------------------------------------------------
# Normalized candidate
# ---------------------------------------------------------------------------

# Candidate dict shape (all optional unless noted; unknowns stay None):
#
#   server            str   (required)  hub server name
#   model             str   (required)  model id
#   url               str   (required)  base URL to send requests to
#   kind              str   "llama.cpp" | "vllm" | ... (informational)
#   online            bool  (required)  server reachable in latest poll cycle
#   state_age_s       float|None        age of the server's last good state
#   loaded            bool  (required)  model resident in the backend
#   input_modalities  [str]|None        from /v1/models architecture metadata
#   vision_override   bool|None         per-model config: force image-capable
#                                       True/False (None = trust modalities)
#   enabled           bool  per-model config enable (default True)
#   tier              int   lower = preferred (default 10); encode measured
#                           vision performance, not host identity
#   parallelism       int|None          slots the model runs with
#   busy_slots        int|None          slots busy right now
#   n_proc            int|None          requests currently being processed
#   n_deferred        int|None          requests waiting behind them
#   gpu_util_max      float|None        max sidecar utilization across the
#                                       server's GPUs (percent)
#   gpus_stale        bool              sidecar data older than its freshness
#                                       window (penalty not applied)
#   tpp               float|None        recent prompt tokens/s
#   tgen              float|None        recent generation tokens/s


def normalize_candidate(c):
    """Validate/normalize one candidate dict. Missing keys become None (or
    their documented defaults); unknown keys are dropped. Pure."""
    if not isinstance(c, dict):
        raise ValueError("candidate must be a dict")
    out = {}
    for k in ("server", "model", "url"):
        v = c.get(k)
        if not isinstance(v, str) or not v:
            raise ValueError(f"candidate.{k} must be a non-empty string")
        out[k] = v
    for k in ("kind", "input_modalities"):
        out[k] = c.get(k)
    for k in ("online", "loaded", "enabled", "gpus_stale"):
        out[k] = bool(c.get(k, k == "enabled"))
    for k in ("state_age_s", "busy_slots", "gpu_util_max", "tpp", "tgen"):
        v = c.get(k)
        out[k] = v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    for k in ("n_proc", "n_deferred"):
        v = c.get(k)
        out[k] = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    v = c.get("parallelism")
    out["parallelism"] = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None
    v = c.get("vision_override")
    out["vision_override"] = bool(v) if isinstance(v, bool) else None
    v = c.get("tier")
    out["tier"] = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else 10
    if out["input_modalities"] is not None and not isinstance(out["input_modalities"], list):
        out["input_modalities"] = None
    return out


# ---------------------------------------------------------------------------
# Eligibility / capacity
# ---------------------------------------------------------------------------

# Closed vocabulary — these strings appear in the public response and in
# metric-adjacent docs; new codes are additive only.
REJECT = {
    "server_offline": "server not reachable in the latest poll cycle",
    "stale_state": "server state older than the freshness limit",
    "not_loaded": "model not currently resident in the backend",
    "no_image_modality": "model does not report image input (and no config override)",
    "policy_disabled": "candidate disabled in the vision policy",
}
REASONS = {
    "loaded": "model resident in the backend",
    "image_capable": "reports image input modality",
    "immediate_capacity": "execution capacity free — request will not join a queue",
    "queued": "execution capacity fully occupied — request will join a queue",
    "only_queued_available": "no immediately available candidate; best queued one returned",
    "preferred_tier": "lowest configured vision tier among eligible candidates",
    "gpu_contention": "shared accelerator under sustained load (ordering penalty)",
}


def image_capable(c):
    if c["vision_override"] is not None:
        return c["vision_override"]
    mods = c["input_modalities"]
    if mods is None:
        # Metadata absent. llama.cpp: treat as capable — older builds
        # reported ["text"] (or nothing) for models with a loaded mmproj,
        # so absence is not proof of text-only. vLLM: treat as NOT capable —
        # its /v1/models does not expose modalities at all, and a text-only
        # engine must not win a vision route; opt a vision-capable vLLM
        # model in explicitly via vision: {image_capable: true}.
        return c.get("kind") != "vllm"
    return "image" in mods


def capacity_class(c):
    """'immediate' | 'queued' | 'unknown'.

    Known-busy signals, any of which puts a candidate in QUEUED:
      - requests in flight >= parallelism (the incident's exact signature:
        parallel=1, one long prefill in flight),
      - busy slots >= parallelism,
      - anything deferred (a queue already exists).
    Unknown capacity (metrics absent) stays IMMEDIATE-with-penalty: a loaded
    model is probably usable, and an advisor that can't prove saturation
    should not pretend it can.
    """
    if c["n_deferred"] is not None and c["n_deferred"] > 0:
        return "queued"
    par = c["parallelism"]
    if par is not None:
        if c["n_proc"] is not None and c["n_proc"] >= par:
            return "queued"
        if c["busy_slots"] is not None and c["busy_slots"] >= par:
            return "queued"
    if par is None and c["n_proc"] is None and c["busy_slots"] is None:
        return "unknown"
    return "immediate"


def _reject(c, now, policy, rejected):
    """First failing filter wins (documented order). Returns the reject code
    or None if eligible; records the rejection for the response."""
    code = None
    if not c["online"]:
        code = "server_offline"
    elif c["state_age_s"] is not None and c["state_age_s"] > policy["stale_after_s"]:
        code = "stale_state"
    elif not c["loaded"]:
        code = "not_loaded"
    elif not image_capable(c):
        code = "no_image_modality"
    elif not c["enabled"]:
        code = "policy_disabled"
    if code is not None:
        rejected.append({"name": f"{c['server']}/{c['model']}", "code": code,
                         "detail": REJECT[code]})
    return code


def _score(c, policy):
    """Informational 0..1 score. Ordering is done by (class, tier, penalty
    flags) below; the score is the same ingredients expressed as one number
    so consumers/dashboards can display it. Deductions:
      tier            0.02 × tier      (0 → 0.0, 5 → 0.10, 7 → 0.14, 20 → 0.40)
      GPU contention  0.10             (sustained util >= threshold, fresh data)
      requests in     0.10             (n_proc > 0)
      recent prefill  0.05             (tpp > 0 — work has been flowing in)
    """
    s = 1.0
    s -= 0.02 * c["tier"]
    if (c["gpu_util_max"] is not None and not c["gpus_stale"]
            and c["gpu_util_max"] >= policy["gpu_busy_threshold"]):
        s -= 0.10
    if c["n_proc"] is not None and c["n_proc"] > 0:
        s -= 0.10
    if c["tpp"] is not None and c["tpp"] > 0:
        s -= 0.05
    return round(max(0.0, min(1.0, s)), 3)


def _gpu_contended(c, policy):
    return (c["gpu_util_max"] is not None and not c["gpus_stale"]
            and c["gpu_util_max"] >= policy["gpu_busy_threshold"])


def _sort_key(c, policy):
    cls = capacity_class(c)
    return (
        0 if cls in ("immediate", "unknown") else 1,   # IMMEDIATE/unknown first
        c["tier"],
        1 if _gpu_contended(c, policy) else 0,
        1 if (c["n_proc"] is not None and c["n_proc"] > 0) else 0,
        c["server"], c["model"],                        # deterministic tail
    )


def _choice_payload(c, cls):
    return {
        "server": c["server"],
        "model": c["model"],
        "url": c["url"],
        "api": "openai-chat",
        "kind": c["kind"] or "unknown",
        "capacity": cls if cls != "unknown" else "immediate",
        "score": _score(c, DEFAULT_POLICY),  # recomputed by caller with real policy
        "state_age_ms": None if c["state_age_s"] is None
                      else int(round(c["state_age_s"] * 1000)),
    }


# ---------------------------------------------------------------------------
# The route decision
# ---------------------------------------------------------------------------

def route_vision(candidates, request=None, policy=None):
    """Pick the best vision endpoint from normalized candidates.

    candidates  [candidate]   normalized dicts (see normalize_candidate)
    request     dict|None     {"images": N, "max_width": W, "max_height": H}
                              accepted in the contract; Phase 1 scoring is
                              request-shape-independent (all candidates serve
                              any single request) — the field exists so the
                              contract doesn't break when that changes.
    policy      dict|None     validated per validate_policy (defaults merged)

    Returns the public response dict (ok / policy_version / choice / reason /
    reason_codes / alternates / rejected). No I/O of any kind.
    """
    policy = validate_policy(DEFAULT_POLICY if policy is None else policy)
    request = request or {}
    if not policy["enabled"]:
        return {"ok": False, "policy_version": policy["policy_version"],
                "choice": None, "reason": "vision routing disabled in hub config",
                "reason_codes": ["disabled"], "alternates": [], "rejected": []}
    if not isinstance(candidates, (list, tuple)):
        raise ValueError("candidates must be a list of candidate dicts")
    candidates = [normalize_candidate(c) for c in candidates]
    if not candidates:
        return {"ok": False, "policy_version": policy["policy_version"],
                "choice": None,
                "reason": "no vision-capable backend registered with the hub",
                "reason_codes": ["no_candidates"], "alternates": [], "rejected": []}

    rejected = []
    eligible = []
    for c in candidates:
        if _reject(c, None, policy, rejected) is None:
            eligible.append(c)

    if not eligible:
        codes = sorted({r["code"] for r in rejected}) or ["no_candidates"]
        return {"ok": False, "policy_version": policy["policy_version"],
                "choice": None,
                "reason": "no usable vision route: " + ", ".join(codes),
                "reason_codes": codes, "alternates": [], "rejected": rejected}

    eligible.sort(key=lambda c: _sort_key(c, policy))
    choice_c = eligible[0]
    choice_cls = capacity_class(choice_c)
    choice = _choice_payload(choice_c, choice_cls)
    choice["score"] = _score(choice_c, policy)

    codes = ["loaded", "image_capable"]
    if choice_cls == "queued":
        codes += ["queued", "only_queued_available"]
    else:
        codes.append("immediate_capacity")
    if choice_c["tier"] == min(c["tier"] for c in eligible):
        codes.append("preferred_tier")
    if _gpu_contended(choice_c, policy):
        codes.append("gpu_contention")

    reason_bits = []
    if choice_cls == "queued":
        reason_bits = [REASONS["only_queued_available"],
                       f"{choice_c['server']} is the best of the loaded candidates but "
                       f"its capacity is occupied — expect queueing"]
    else:
        if "preferred_tier" in codes:
            reason_bits.append(REASONS["preferred_tier"])
        reason_bits.append(REASONS["immediate_capacity"])
        if _gpu_contended(choice_c, policy):
            reason_bits.append(REASONS["gpu_contention"])

    alternates = [_choice_payload(c, capacity_class(c)) | {"score": _score(c, policy)}
                 for c in eligible[1:3]]

    return {
        "ok": True,
        "policy_version": policy["policy_version"],
        "choice": choice,
        "reason": "; ".join(reason_bits),
        "reason_codes": codes,
        "alternates": alternates,
        "rejected": rejected,
    }
