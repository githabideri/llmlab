"""evidence.py — the workload contract (INTEGRITY layer of the verdict ladder).

One question per attempt, answered with evidence rather than time constants:

    did the workload that was declared actually happen?

Three values (tri-valued, like the effect gate):

  SATISFIED      every required concept has at least one accepted source
                 and all observed values are within tolerance of the
                 declaration.
  VIOLATED       evidence WAS captured and it contradicts the declaration:
                 an observed token count far from declared, or a server log
                 that is present but contains no record of the request at
                 all.  This is the 256k no-op shape — proven, not guessed.
  UNVERIFIABLE   the pipeline could not produce the required evidence
                 (no usage in the response, no server log pulled).  This is
                 a harness problem, never a scientific fact: the verdict
                 engine maps it to HARNESS_FAILURE, with one exception —
                 if the wall clock is ALSO implausible for the declared
                 work, the combined signal is the LADDER-256k shape and the
                 attempt is INVALID (the wall floor's only remaining job:
                 last resort when no evidence exists at all).

Evidence concepts (spec cell may add/remove via `workload_contract.require`;
defaults: server_prompt_tokens + request_seen):

  server_prompt_tokens   the engine's own count of the prompt it processed.
                         Sources per engine are the accepted sets in
                         evidence_policy.jsonc: *-usage (server-computed,
                         client-transcribed from the response) and
                         *-server-log (independent of the client process).
  client_encoded_tokens  what the client actually serialized (fixture
                         manifest / tokenizer count).  A client that
                         declares 60k but encodes 200 fails this — even
                         when the server is perfectly honest about the 200.
  request_seen           an independent record that the request reached the
                         engine (access-log line per request).

Absence is not equivalence: a missing value is UNVERIFIABLE; a captured
zero-record is VIOLATED; a different-but-declared source is perfectly valid
(the accepted sets exist precisely so the gate never assumes which channel
an engine uses).
"""
import json
import os
import re

POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "evidence_policy.jsonc")

DEFAULT_TOLERANCE = {"absolute": 32, "relative": 0.01}
DEFAULT_REQUIRE = ["server_prompt_tokens", "request_seen"]


def load_policy():
    text = open(POLICY_PATH).read()
    text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
    return json.loads(text)


def _tolerance(declared, tol_cfg):
    a = tol_cfg.get("absolute", DEFAULT_TOLERANCE["absolute"])
    r = tol_cfg.get("relative", DEFAULT_TOLERANCE["relative"])
    return max(a, r * max(declared or 0, 0))


def evaluate(cell, client_data, server_log_text, engine):
    """Evaluate one attempt's evidence against its declared workload.

    cell:            spec cell (requests[].prompt_tokens, workload_contract)
    client_data:     the client's JSON result (rows carry declared_prompt_tokens,
                     client_encoded_tokens, server_usage_prompt_tokens)
    server_log_text: the attempt's server log(s) concatenated, or None when
                     none could be pulled
    engine:          spec-level engine name (vllm | llama_cpp) — selects the
                     accepted source sets; None -> NOT_APPLICABLE

    Returns {status, detail, rows: [...], seen: bool|None, log_counts: [...]}.
    """
    out = {"status": "NOT_APPLICABLE", "detail": "no declared workload or no engine",
           "rows": [], "seen": None, "log_counts": []}
    if not engine:
        return out

    requests = cell.get("requests") or []
    rows = (client_data or {}).get("rows") or (client_data or {}).get("runs") or []
    declared = []
    for i in range(max(len(rows), len(requests))):
        d = None
        if i < len(rows):
            d = rows[i].get("declared_prompt_tokens")
        if d is None and i < len(requests):
            d = (requests[i] or {}).get("prompt_tokens")
        if d is not None:
            declared.append(int(d))
    if not declared:
        out["detail"] = "no request declares a prompt size"
        return out

    wc = cell.get("workload_contract") or {}
    require = wc.get("require") or DEFAULT_REQUIRE
    tol_cfg = dict(DEFAULT_TOLERANCE)
    tol_cfg.update(wc.get("tolerance") or {})

    policy = load_policy().get(engine) or {}
    accept = {k: list(v) for k, v in policy.items() if k not in ("log_patterns",)}
    accept.update(wc.get("accept") or {})

    pats = (policy.get("log_patterns") or {})
    seen = None
    log_counts = []
    if server_log_text is not None:
        seen = any(re.search(pats.get("request_seen") or r"$^", line)
                   for line in server_log_text.splitlines())
        m = pats.get("server_prompt_tokens")
        if m:
            log_counts = [int(x) for x in re.findall(m, server_log_text)]

    violations = []
    unverifiable = []
    row_out = []

    for i, row in enumerate(rows):
        d = row.get("declared_prompt_tokens")
        if d is None and i < len(declared):
            d = declared[i]
        if d is None:
            continue
        tol = _tolerance(int(d), tol_cfg)
        row_ev = []
        # (1) client-constructed source
        enc = row.get("client_encoded_tokens")
        if enc is not None:
            row_ev.append({"value": int(enc), "source": "client-constructed"})
        # (2) server-computed, client-transcribed (usage field in the response)
        u = row.get("server_usage_prompt_tokens")
        if u is not None:
            row_ev.append({"value": int(u), "source": engine + "-usage"})
        # (3) server-log-derived (the client parsed the engine's own log)
        lg = row.get("server_log_prompt_tokens")
        if lg is not None:
            row_ev.append({"value": int(lg), "source": engine + "-server-log"})
        for ev in row_ev:
            if abs(ev["value"] - int(d)) > tol:
                violations.append(f"row {i}: declared {d} vs {ev['source']} "
                                  f"{ev['value']} (tol {tol:.0f})")
        row_out.append({"index": i, "declared": int(d), "tolerance": tol,
                        "evidence": row_ev,
                        "violation": any("row %d" % i in v for v in violations)})

    # independent log counts: every one must match SOME declared row
    for c in log_counts:
        if not any(abs(c - d) <= _tolerance(d, tol_cfg) for d in declared):
            violations.append(f"server log observed {c} prompt tokens, "
                              f"matching no declared workload {declared}")

    if "server_prompt_tokens" in require:
        all_covered = (log_counts or
                       (row_out and all(
                           any(e["source"] != "client-constructed"
                               for e in r["evidence"])
                           for r in row_out)))
        if not all_covered:
            unverifiable.append("no accepted source for server_prompt_tokens "
                                f"(accepted: {accept.get('server_prompt_tokens')})")
    if "client_encoded_tokens" in require:
        if not any(r["evidence"] and
                   any(e["source"] == "client-constructed" for e in r["evidence"])
                   for r in row_out):
            unverifiable.append("no client_encoded_tokens (client has no "
                                "authoritative count for this plan)")

    if "request_seen" in require:
        if server_log_text is None:
            unverifiable.append("server log not pulled (request_seen unknown)")
        elif not seen:
            violations.append("server log is present but contains no record "
                              "of the request (no-op proven)")

    out["seen"] = seen
    out["log_counts"] = log_counts
    out["rows"] = row_out
    if violations:
        out["status"] = "VIOLATED"
        out["detail"] = "; ".join(violations[:3])
    elif unverifiable:
        out["status"] = "UNVERIFIABLE"
        out["detail"] = "; ".join(unverifiable[:3])
    else:
        out["status"] = "SATISFIED"
        out["detail"] = (f"{len(row_out)} rows within tolerance "
                         f"(seen={seen}, log_counts={len(log_counts)})")
    return out
