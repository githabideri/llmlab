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
    PLAUSIBILITY_ANOMALY measured with COMPLETE integrity evidence, but a
                         plausibility gate (the wall floor) fired anyway —
                         the numbers are kept for review, never silent
    UNKNOWN              unrecognized failure shape — REVIEW_REQUIRED, pages the owner
    SAFETY_ABORT         stop_policy / watchdog — pages the owner (URGENT)

Campaign-level finals:
    completed | expected-negative | review-required | safety-abort |
    stopped-host-failure | aborted-*

Design rule baked in (the 2026-09-12 lesson): an outcome that matches the spec's
`documented_negative` classes is EXPECTED_NEGATIVE and the runner stops DONE. It is
never "needs human review", because a correct predicted result is not an error.

Integrity rule baked in (the 2026-09-14 lesson, generalized): a workload the
spec declared is verified by EVIDENCE (evidence.py) before any plausibility
gate runs. A wall floor may only demote a complete-evidence attempt to
PLAUSIBILITY_ANOMALY; it is INVALID again only when the workload itself is
unverifiable (the LADDER-256k shape: no evidence at all and an impossible
wall).  Timing is a hint; integrity is fact.
"""

PASS = "PASS"
EXPECTED_NEGATIVE = "EXPECTED_NEGATIVE"
INVALID = "INVALID"
RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
RETRYABLE_INFRA = "RETRYABLE_INFRA"
RESOURCE_LIMIT = "RESOURCE_LIMIT"
HARNESS_FAILURE = "HARNESS_FAILURE"
PLAUSIBILITY_ANOMALY = "PLAUSIBILITY_ANOMALY"
UNKNOWN = "UNKNOWN"
SAFETY_ABORT = "SAFETY_ABORT"

CELL_CLASSES = (PASS, EXPECTED_NEGATIVE, INVALID, RETRYABLE_FAILURE, RETRYABLE_INFRA,
                RESOURCE_LIMIT, HARNESS_FAILURE, PLAUSIBILITY_ANOMALY, UNKNOWN,
                SAFETY_ABORT)

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
    "KV_CONFIG_INVALID": HARNESS_FAILURE,   # launch-config defect (09-16 run #7)
    "TOKEN_MISMATCH": INVALID,
    # the LMCache classes are evidence-based (the client declares them from
    # the store/retrieval logs); None = "documented-negative-oriented": they
    # resolve against the cell's expects.documented_negative in decide(), and
    # an undeclared-in-spec occurrence degrades to UNKNOWN (review) below.
    "LMCACHE_CONNECTOR_FALLBACK": None,
    "LMCACHE_NO_HIT": None,
    "LMCACHE_RESTORE_CORRUPTION": None,
    "LMCACHE_PERSISTENCE_SUBPAGE": None,
}


def decide(cell_cfg, classifier_result, gates, client_result, attempt_wall_s,
           contract=None):
    """Map one attempt's evidence to a cell verdict.

    cell_cfg:        the spec cell dict (expects, documented_negative, ...)
    classifier_result: classify() output (or None when the server never produced
                       a failure log)
    gates:           list of (ok: bool, label: str, detail: str) from the cell's
                     plausibility gates
    client_result:   the client's JSON result dict (or None on client failure)
    attempt_wall_s:  wall time of the attempt in seconds
    contract:        evidence.evaluate() output for this attempt (or None)

    Returns (verdict, reason).
    """
    # 1) a harness-side crash of our own tooling is never a model verdict
    if client_result is None and classifier_result is None and attempt_wall_s < 2:
        return HARNESS_FAILURE, "no client result and no server failure evidence"

    # 2) WORKLOAD CONTRACT (integrity) — before anything else scientific: if the
    #    declared workload did not demonstrably happen, no downstream reading of
    #    this attempt is valid.
    c_status = (contract or {}).get("status")
    if c_status == "VIOLATED":
        return INVALID, "workload contract violated: " + (contract or {}).get("detail", "")
    if c_status == "UNVERIFIABLE":
        wall_violated = any(not ok and label == "min_wall_s" for ok, label, _ in gates)
        if wall_violated:
            return INVALID, ("workload unverifiable AND wall implausible "
                             "(LADDER-256k last resort): "
                             + (contract or {}).get("detail", ""))
        return HARNESS_FAILURE, ("workload contract unverifiable (evidence gap, "
                                 "not a scientific fact): "
                                 + (contract or {}).get("detail", ""))

    # 3) documented-negative check runs BEFORE the generic mapping: the spec says
    #    which classes are the expected answer for THIS cell. The class can come
    #    from the failure-log classifier OR be DECLARED BY THE CLIENT from
    #    evidence (e.g. a store that wrote but never served a retrieval line,
    #    or a wrong buried-key answer on a logged hit) — evidence-based
    #    verdicts the log shape alone cannot express.
    class_ = (classifier_result or {}).get("class") \
        or (client_result or {}).get("declared_class")
    declared = class_ is not None and not (classifier_result or {}).get("class")
    doc_neg = (cell_cfg.get("expects") or {}).get("documented_negative") or []
    if class_ and class_ in doc_neg:
        return EXPECTED_NEGATIVE, f"documented negative {class_} matched: " \
            + ", ".join((classifier_result or {}).get("observed", [])[:2])

    # 4) plausibility gates — only when the client actually measured something:
    #    a server that died at load time has no measurement to gate (the class
    #    decides it). Gating a load-failure log on min_wall_s is how the 09-12
    #    night turned the documented wall into 'needs human review'.
    #    With SATISFIED contract evidence, a firing wall floor is a REVIEW
    #    anomaly, not INVALID — the evidence says the workload happened.
    if client_result is not None:
        for ok, label, detail in gates:
            if not ok:
                if label == "min_wall_s" and c_status == "SATISFIED":
                    return PLAUSIBILITY_ANOMALY, ("plausibility: wall floor "
                                                  "fired with complete integrity "
                                                  "evidence — numbers kept for "
                                                  "review: " + detail)
                return INVALID, f"gate failed: {label} ({detail})"

    # 5) client measured data: trust it unless it self-reports invalid
    if client_result is not None:
        if client_result.get("valid") is False:
            return INVALID, "client reported invalid (see client result flags)"
        if class_ and class_ in ("TOKEN_MISMATCH",):
            return INVALID, "classifier: " + class_
        if class_ is None:
            return PASS, "measured, gates pass"
        # a class DECLARED BY THE CLIENT (evidence-based) that the spec did not
        # document for this cell: never a silent PASS, never a silent negative.
        # (Classifier-sourced classes keep their table mapping, step 5.)
        if declared:
            return UNKNOWN, f"client declared {class_} (not a documented " \
                            f"negative for this cell)"

    # 6) no client data: fall back to what the server log says
    if class_ is not None:
        v = CLASS_TO_VERDICT.get(class_)
        if v is not None:
            return v, f"classifier: {class_}"
        # documented-negative-oriented (or declared-without-data): the spec
        # did not predict this for this cell — review, never assume
        return UNKNOWN, f"class {class_} is documented-negative-oriented or " \
            f"client-declared; the spec did not declare it for this cell — review"
    return UNKNOWN, "no client data and no classifier evidence"


def campaign_final(cell_verdicts, host_failure=None):
    """Fold per-cell verdicts into a campaign-level final.

    Conservative order (2026-09-13 external review): a campaign in which
    some cell FAILED as a harness/measurement problem must never collapse
    to an INFO-level 'expected-negative' just because another cell
    produced a documented negative. 'We got a result' and 'one of our
    instruments broke' are different outcomes; the latter needs review.
    A SKIPPED cell (its pre-gate failed or was unverifiable) is likewise a
    decision that needs eyes, not an invisible gap.
    """
    if host_failure:
        return "stopped-host-failure"
    vs = list(cell_verdicts.values())
    if any(v == SAFETY_ABORT for v in vs):
        return "safety-abort"
    if any(v in (RETRYABLE_FAILURE, RETRYABLE_INFRA, UNKNOWN,
                 HARNESS_FAILURE, INVALID, PLAUSIBILITY_ANOMALY, "SKIPPED") for v in vs):
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
