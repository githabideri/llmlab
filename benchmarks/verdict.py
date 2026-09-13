"""verdict.py — verdict taxonomy + decision engine (platform core).

Top-level cell verdict classes:

    PASS                 gate-valid result
    EXPECTED_NEGATIVE    a documented negative outcome hit (the campaign's question
                         was ANSWERED — this is a success path, never a page)
    INVALID              measured but untrustworthy (invalid-conc, invalid-swapped,
                         false-complete, token mismatch)
    RETRYABLE_FAILURE    the measured system failed in a transient way (server died,
                         connection reset)
    RETRYABLE_INFRA      infrastructure hiccup outside the measured system (port in
                         use, sampler death)
    RESOURCE_LIMIT       OOM / cgroup-oom / disk — not the documented negative
    HARNESS_FAILURE      OUR code failed (crash, misparse) — not a model verdict;
                         feeds the bounded repair lane
    UNKNOWN              unrecognized failure shape — REVIEW_REQUIRED, pages the owner
    SAFETY_ABORT         stop_policy / watchdog — pages the owner (URGENT)

Campaign-level finals:
    completed | expected-negative | review-required | safety-abort |
    stopped-host-failure | aborted-*

Design rule baked in (the 2026-09-12 lesson): an outcome that matches the spec's
`documented_negative` classes is EXPECTED_NEGATIVE and the runner stops DONE. It is
never "needs human review", because a correct predicted result is not an error.
"""

PASS = "PASS"
EXPECTED_NEGATIVE = "EXPECTED_NEGATIVE"
INVALID = "INVALID"
RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
RETRYABLE_INFRA = "RETRYABLE_INFRA"
RESOURCE_LIMIT = "RESOURCE_LIMIT"
HARNESS_FAILURE = "HARNESS_FAILURE"
UNKNOWN = "UNKNOWN"
SAFETY_ABORT = "SAFETY_ABORT"

CELL_CLASSES = (PASS, EXPECTED_NEGATIVE, INVALID, RETRYABLE_FAILURE, RETRYABLE_INFRA,
                RESOURCE_LIMIT, HARNESS_FAILURE, UNKNOWN, SAFETY_ABORT)

# classifier class -> cell verdict, when the cell's own measurement produced no data
CLASS_TO_VERDICT = {
    "WALL_VRAM_FIT": None,          # resolved against spec.documented_negative in decide()
    "BUILD_DEFECT_ASSERT": HARNESS_FAILURE,
    "UNSUPPORTED_ARCH": None,       # often a documented negative (control cells)
    "CUDA_OOM": RESOURCE_LIMIT,
    "C_GROUP_OOM": RESOURCE_LIMIT,
    "HOST_KERNEL_ERROR": SAFETY_ABORT,
    "MODEL_LOAD_FAIL": RETRYABLE_FAILURE,
    "WRONG_GPU": SAFETY_ABORT,      # identity drift: never a measurable outcome
    "HTTP_EMPTY_200": HARNESS_FAILURE,
    "SSE_MALFORMED": HARNESS_FAILURE,
    "SSE_NO_USAGE": HARNESS_FAILURE,
    "TOKEN_MISMATCH": INVALID,
}


def decide(cell_cfg, classifier_result, gates, client_result, attempt_wall_s):
    """Map one attempt's evidence to a cell verdict.

    cell_cfg:        the spec cell dict (expects, documented_negative, ...)
    classifier_result: classify() output (or None when the server never produced
                       a failure log)
    gates:           list of (ok: bool, label: str, detail: str) from the cell's
                     plausibility gates
    client_result:   the client's JSON result dict (or None on client failure)
    attempt_wall_s:  wall time of the attempt in seconds

    Returns (verdict, reason).
    """
    # 1) a harness-side crash of our own tooling is never a model verdict
    if client_result is None and classifier_result is None and attempt_wall_s < 2:
        return HARNESS_FAILURE, "no client result and no server failure evidence"

    # 2) documented-negative check runs BEFORE the generic mapping: the spec says
    #    which classes are the expected answer for THIS cell.
    class_ = (classifier_result or {}).get("class")
    doc_neg = (cell_cfg.get("expects") or {}).get("documented_negative") or []
    if class_ and class_ in doc_neg:
        return EXPECTED_NEGATIVE, f"documented negative {class_} matched: " \
            + ", ".join((classifier_result or {}).get("observed", [])[:2])

    # 3) plausibility gates — only when the client actually measured something:
    #    a server that died at load time has no measurement to gate (the class
    #    decides it). Gating a load-failure log on min_wall_s is how the 09-12
    #    night turned the documented wall into 'needs human review'.
    if client_result is not None:
        for ok, label, detail in gates:
            if not ok:
                return INVALID, f"gate failed: {label} ({detail})"

    # 4) client measured data: trust it unless it self-reports invalid
    if client_result is not None:
        if client_result.get("valid") is False:
            return INVALID, "client reported invalid (see client result flags)"
        if class_ and class_ in ("TOKEN_MISMATCH",):
            return INVALID, "classifier: " + class_
        if class_ is None:
            return PASS, "measured, gates pass"

    # 5) no client data: fall back to what the server log says
    if class_ is not None:
        v = CLASS_TO_VERDICT.get(class_)
        if v is not None:
            return v, f"classifier: {class_}"
        # unmapped class with no data: unknown shape
        return UNKNOWN, f"unmapped classifier class {class_} with no client data"
    return UNKNOWN, "no client data and no classifier evidence"


def campaign_final(cell_verdicts, host_failure=None):
    """Fold per-cell verdicts into a campaign-level final."""
    if host_failure:
        return "stopped-host-failure"
    vs = list(cell_verdicts.values())
    if any(v == SAFETY_ABORT for v in vs):
        return "safety-abort"
    if any(v in (RETRYABLE_FAILURE, RETRYABLE_INFRA, UNKNOWN) for v in vs):
        return "review-required"
    if all(v in (PASS, EXPECTED_NEGATIVE) for v in vs) and vs:
        if all(v == EXPECTED_NEGATIVE for v in vs):
            return "expected-negative"
        return "completed"
    if any(v == EXPECTED_NEGATIVE for v in vs) and vs:
        return "expected-negative"   # documented result reached; partial data ok
    return "review-required"


NOTIFY_LEVELS = {
    "completed": "INFO",
    "expected-negative": "INFO",
    "review-required": "ATTENTION",
    "safety-abort": "URGENT",
    "stopped-host-failure": "URGENT",
    "aborted-signal": "ATTENTION",
    "aborted-failure": "URGENT",
}
