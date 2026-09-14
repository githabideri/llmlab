#!/usr/bin/env python3
"""verify-lmcache.py — LMCache L1-tier campaign client (stdlib only).

Two modes:

  --mode probe   S0. Reads the vLLM boot log + the LMCache sidecar log,
                 extracts the hybrid block size N, the KV pool size, the
                 connector module actually loaded (provenance, never
                 cargo-culted) and the LMCache version; checks expect/
                 forbid patterns (SPEC DATA). Writes client.json plus
                 probe.json (N, pool, versions) which the platform uses to
                 size every later launch. Declares LMCACHE_CONNECTOR_FALLBACK
                 when the connector loaded but the sidecar never saw a
                 store for the probe request (the 2026-08-30 silent no-op
                 shape — a DOCUMENTED NEGATIVE, a completed campaign).

  --mode run     The battery. Executes a request plan (a JSON file: a list
                 of segments, each a list of requests referencing fixtures)
                 against /v1/chat/completions. The platform drives the
                 vLLM relaunches between segments (--segment k --segments N);
                 this client only measures. Per request: exact usage tokens,
                 TTFT/E2E walls (strictly separate), stream integrity. Per
                 segment: the LMCache sidecar log slice is parsed for
                 retrieval evidence (a TTFT drop WITHOUT a retrieval line is
                 not a hit — LMCACHE_NO_HIT on must-hit legs). Buried keys:
                 a wrong answer on a logged retrieval is
                 LMCACHE_RESTORE_CORRUPTION; on a full recompute it is a
                 fixture/model defect, reported separately. The final
                 segment merges everything into client.json.

Honesty rules (2026-09-13 review, applied throughout):
  * log-shape parsing is best-effort: an UNRECOGNIZED shape yields
    not_established evidence (null), never a fake zero / fake hit;
  * the correctness criterion for GDN hybrids is buried-fact / score-level
    equivalence, not token identity (bit-exact is not expected for
    cached-vs-fresh on these architectures — official docs);
  * a declared_class in client.json is a finding the spec may or may not
    have documented: the verdict engine decides (documented => completed
    negative; undeclared => review), this client only reports what it saw.

  --mode selftest  the parsers prove themselves on synthetic logs before any
                   window (the 2026-09-11 lesson: the harness must prove
                   itself before it may touch the machine).
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

# --------------------------------------------------------------------------
# log parsing (best-effort; every branch that cannot recognize its input
# returns None -> the caller records NOT_ESTABLISHED, never a zero)
# --------------------------------------------------------------------------

def parse_pool_tokens(vllm_log_text):
    """KV pool size from the vLLM startup line(s). Shape varies across
    versions; several known forms, otherwise None."""
    for rx in (r"GPU KV cache size:?\s+(\d+)\s+tokens",
               r"total tokens[=: ]+(\d+)",
               r"KV cache memory[^\n]*?(\d[\d,]*)\s+logical tokens"):
        m = re.search(rx, vllm_log_text)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def parse_block_size(vllm_log_text):
    m = re.search(r"attention block size to (\d+) tokens", vllm_log_text)
    return int(m.group(1)) if m else None


def parse_connector(vllm_log_text):
    """What connector module actually loaded (provenance). The bare name is
    NOT the module — on vLLM >= 0.20 the bare name can resolve to vLLM's
    built-in connector, which is the 08-30 no-op."""
    for rx in (r"lmcache\.integration\.vllm\.\w+",
               r"[Kk][Vv][_-]?[Cc]onnector[=:\s]+['\"]?[\w.]+",
               r"\bconnector[=:\s]+['\"]?[\w.]+"):
        m = re.search(rx, vllm_log_text)
        if m:
            return m.group(0)
    return None


def parse_retrievals(lmcache_slice, since_byte=0):
    """Retrieval events from the sidecar log slice.
    Returns a list of {tokens, ts} or None when no line matches any known
    shape (NOT_ESTABLISHED — the caller must not read that as zero hits).
    """
    events = []
    recognized = False
    for line in lmcache_slice.splitlines():
        m = re.search(r"retriev\w*\s*[=: ]+\S*?(\d[\d,]*)\s+tokens?", line,
                      re.I)
        if m:
            recognized = True
            try:
                events.append({"tokens": int(m.group(1).replace(",", "")),
                               "line": line.strip()[:200]})
            except ValueError:
                continue
    if not recognized:
        return None
    return events


def parse_stores(lmcache_slice):
    """Store/write events — the 'did it even go in' evidence."""
    events = []
    recognized = False
    for line in lmcache_slice.splitlines():
        m = re.search(r"(store|writ\w*|put)\w*\s*[=: ]+\S*?(\d[\d,]*)\s+tokens?",
                      line, re.I)
        if m:
            recognized = True
            events.append({"tokens": int(m.group(2).replace(",", "")),
                           "line": line.strip()[:200]})
    if not recognized:
        return None
    return events


def parse_capacity(lmcache_slice):
    """Bytes stored / evictions — planning numbers, reported not gated."""
    out = {"bytes_stored": None, "evictions": None}
    m = re.search(r"(?:total|resident)[\s:=]*(\d[\d,]*)\s*(?:bytes|B\b)",
                  lmcache_slice, re.I)
    if m:
        out["bytes_stored"] = int(m.group(1).replace(",", ""))
    m = re.search(r"evict\w*\s*[=: ]+(\d+)", lmcache_slice, re.I)
    if m:
        out["evictions"] = int(m.group(1))
    return out


# --------------------------------------------------------------------------
# HTTP (vLLM OpenAI-compatible; streaming + usage)
# --------------------------------------------------------------------------

def chat(url, prompt, decode, seed=0, temperature=0.0):
    body = json.dumps({"model": "default", "messages": [{"role": "user",
                         "content": prompt}],
                       "stream": True,
                       "stream_options": {"include_usage": True},
                       "max_tokens": decode,
                       "temperature": temperature,
                       "seed": seed}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    t_first = None
    content = []
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                j = json.loads(payload)
            except ValueError:
                continue
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    if t_first is None:
                        t_first = time.monotonic()
                    content.append(delta["content"])
    t_end = time.monotonic()
    return {"content": "".join(content),
            "ttft_s": round((t_first - t0), 3) if t_first else None,
            "e2e_s": round(t_end - t0, 3),
            "usage": usage,
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "prompt_tokens": (usage or {}).get("prompt_tokens")}


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------

def do_probe(a):
    def read(p):
        try:
            return open(p, errors="replace").read()
        except OSError:
            return ""
    vllm = read(a.vllm_log)
    lm = read(a.lmcache_log)
    checks, valid = {}, True
    for e in a.expect:
        label, _, rx = e.partition("=")
        m = re.search(rx, vllm + lm, re.M | re.I)
        checks[f"expect:{label}"] = m.group(0)[:160] if m else None
        if not m:
            valid = False
    for e in a.forbid:
        label, _, rx = e.partition("=")
        m = re.search(rx, vllm + lm, re.M | re.I)
        checks[f"forbid:{label}"] = m.group(0)[:160] if m else None
        if m:
            valid = False
    n = parse_block_size(vllm)
    pool = parse_pool_tokens(vllm)
    conn = parse_connector(vllm)
    stores = parse_stores(lm)
    # the 08-30 shape: connector loaded, but nothing was ever stored
    declared = None
    if conn and stores == [] and any("LMCache" in c for c in (conn or "")):
        declared = "LMCACHE_CONNECTOR_FALLBACK"
        valid = False
    out = {"valid": valid, "checks": checks,
           "block_size_N": n, "kv_pool_tokens": pool,
           "connector": conn,
           "stores_seen": len(stores) if stores is not None else None,
           "not_established": {"kv_pool": pool is None,
                               "stores": stores is None,
                               "block_size": n is None},
           }
    if declared:
        out["declared_class"] = declared
    json.dump(out, open(a.json_out, "w"), indent=2)
    if a.probe_out and n is not None:
        probe = {"N": n, "pool_tokens": pool, "connector": conn,
                 "lmcache_version": a.lmcache_version or "unknown"}
        json.dump(probe, open(a.probe_out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in
                      ("valid", "block_size_N", "kv_pool_tokens", "connector")}))
    raise SystemExit(0 if valid else 2)


def do_run(a):
    plan = json.load(open(a.plan))
    seg = a.segment if a.segment is not None else 1
    n_seg = a.segments if a.segments is not None else len(plan)
    seg_plan = plan[seg - 1]
    if not isinstance(seg_plan, dict):
        seg_plan = {"requests": seg_plan}
    fixtures = seg_plan.get("fixtures") or {}
    lm_since = 0
    if a.lmcache_log and os.path.exists(a.lmcache_log):
        lm_since = os.path.getsize(a.lmcache_log)
    rows, keys, hits, declared = [], [], {}, None
    for rq in seg_plan["requests"]:
        fx = rq.get("fixture")
        if fx:
            if a.fixture_dir and fx in fixtures:
                p = os.path.join(a.fixture_dir, fixtures[fx])
            else:
                p = os.path.join(a.fixture_dir or ".", fx)
            prompt = open(p, errors="replace").read()
            if rq.get("nonce"):
                prompt += f"\n\n[NONCE-{a.nonce or 'x'}-{seg}-{rq.get('tag', 'r')}]\n"
        else:
            prompt = rq.get("text", "Say ok")
        res = chat(a.url, prompt, int(rq.get("decode", 64)),
                   seed=int(rq.get("seed", 0)),
                   temperature=float(rq.get("temperature", 0.0)))
        row = {"tag": rq.get("tag"), "fixture": fx,
               "nonce": bool(rq.get("nonce")),
               "ttft_s": res["ttft_s"], "e2e_s": res["e2e_s"],
               "completion_tokens": res["completion_tokens"],
               "prompt_tokens": res["prompt_tokens"]}
        rows.append(row)
        # retrieval evidence for this request (log slice grown since before)
        if a.lmcache_log:
            with open(a.lmcache_log, errors="replace") as f:
                f.seek(lm_since)
                slice_ = f.read()
            lm_since = os.path.getsize(a.lmcache_log)
            ev = parse_retrievals(slice_)
            row["retrieval_evidence"] = (
                "not_established" if ev is None
                else ({"tokens": sum(e["tokens"] for e in ev),
                       "events": len(ev)} if ev else {"tokens": 0}))
        # buried key
        if rq.get("key"):
            kq = chat(a.url, rq["key"]["query"], int(rq.get("key_decode", 64)),
                      seed=0)
            expected = rq["key"]["expected"]
            ok = expected in kq["content"]
            keys.append({"tag": rq.get("tag"), "expected": expected,
                         "got": kq["content"][:300], "correct": ok,
                         "on_logged_retrieval": row.get(
                             "retrieval_evidence", {}) not in
                            (None, "not_established")
                             and isinstance(row.get("retrieval_evidence"), dict)
                             and row["retrieval_evidence"].get("tokens", 0) > 0
                             if isinstance(row.get("retrieval_evidence"), dict)
                             else False})
            if not ok:
                ev = row.get("retrieval_evidence")
                if isinstance(ev, dict) and ev.get("tokens", 0) > 0:
                    declared = "LMCACHE_RESTORE_CORRUPTION"
                else:
                    keys[-1]["defect_class"] = (
                        "fixture-or-model defect on a NON-logged leg "
                        "(not a cache corruption)")
        if rq.get("must_hit") and a.lmcache_log:
            ev = row.get("retrieval_evidence")
            if not (isinstance(ev, dict) and ev.get("tokens", 0) > 0):
                declared = declared or "LMCACHE_NO_HIT"
    seg_out = {"segment": seg, "segments": n_seg, "rows": rows, "keys": keys,
               "lmcache_capacity": parse_capacity(
                   open(a.lmcache_log, errors="replace").read())
                   if a.lmcache_log else None}
    if declared:
        seg_out["declared_class"] = declared
    seg_file = a.json_out.replace("client.json", f"seg-{seg}.json")
    json.dump(seg_out, open(seg_file, "w"), indent=2)
    if seg == n_seg:
        # merge all segments into the final client.json
        merged = {"valid": True, "rows": [], "keys": [],
                  "restore_corruption_count": 0, "segments": n_seg}
        for k in range(1, n_seg + 1):
            sf = a.json_out.replace("client.json", f"seg-{k}.json")
            try:
                d = json.load(open(sf))
            except (OSError, ValueError):
                merged["valid"] = False
                continue
            merged["rows"].extend(d.get("rows") or [])
            merged["keys"].extend(d.get("keys") or [])
            for kx in d.get("keys") or []:
                if not kx.get("correct"):
                    merged["restore_corruption_count"] += (
                        1 if kx.get("on_logged_retrieval") else 0)
            if d.get("declared_class") == "LMCACHE_RESTORE_CORRUPTION":
                merged["declared_class"] = "LMCACHE_RESTORE_CORRUPTION"
            elif d.get("declared_class") == "LMCACHE_NO_HIT" and \
                    "declared_class" not in merged:
                merged["declared_class"] = "LMCACHE_NO_HIT"
        merged["return_leg_ttft_s"] = (
            sorted(r["ttft_s"] for r in merged["rows"]
                   if r.get("ttft_s") and str(r.get("tag", "")).startswith("ret"))
            or None)
        json.dump(merged, open(a.json_out, "w"), indent=2)
    print(json.dumps({"segment": seg, "rows": len(rows),
                      "declared": declared}))
    raise SystemExit(0)


def do_selftest(a):
    """The parsers prove themselves on synthetic logs (before any window)."""
    ok = True
    vllm = ("INFO 09-13 [core.py] Setting attention block size to 784 tokens\n"
            "INFO 09-13 [worker.py] GPU KV cache size: 776928 tokens\n")
    assert parse_block_size(vllm) == 784
    assert parse_pool_tokens(vllm) == 776928
    lm_good = ("2026-09-13 retrieved 120000 tokens for session A\n"
               "2026-09-13 stored 120000 tokens (object group: kv)\n")
    ev = parse_retrievals(lm_good)
    st = parse_stores(lm_good)
    ok &= (ev is not None and ev[0]["tokens"] == 120000
           and st is not None and st[0]["tokens"] == 120000)
    # an unrecognized shape must be None (NOT zero)
    ev2 = parse_retrievals("2026-09-13 some unrelated line\n")
    st2 = parse_stores("2026-09-13 some unrelated line\n")
    ok &= (ev2 is None and st2 is None)
    # connector provenance: the module path, not the bare name
    assert parse_connector("using lmcache.integration.vllm.lmcache_mp_connector")
    assert parse_connector("connector: vllm.v1.core.kv_connector") is not None
    print(json.dumps({"valid": bool(ok)}))
    raise SystemExit(0 if ok else 2)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--mode", choices=["probe", "run", "selftest"],
                   default="run")
    a.add_argument("--url", default="http://127.0.0.1:8082")
    a.add_argument("--plan", default=None)
    a.add_argument("--segment", type=int, default=None)
    a.add_argument("--segments", type=int, default=None)
    a.add_argument("--nonce", default=None)
    a.add_argument("--fixture-dir", default=None)
    a.add_argument("--vllm-log", default=None)
    a.add_argument("--lmcache-log", default=None)
    a.add_argument("--lmcache-version", default=None)
    a.add_argument("--expect", action="append", default=[])
    a.add_argument("--forbid", action="append", default=[])
    a.add_argument("--probe-out", default=None)
    a.add_argument("--json-out", required=True)
    a = a.parse_args()
    if a.mode == "probe":
        do_probe(a)
    elif a.mode == "run":
        do_run(a)
    else:
        do_selftest(a)


if __name__ == "__main__":
    main()
