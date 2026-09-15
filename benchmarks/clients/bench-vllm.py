#!/usr/bin/env python3
"""bench-vllm.py v2 — vLLM campaign client (dual-3090), validity-first design.

Design rules (incident 2026-09-10, ChatGPT two-round review):
  * One request = one record with FOUR wall-clock timestamps:
      t_start (request sent), t_first (first token), t_last (last token), t_end (response closed)
    from which, kept strictly separate, we report:
      TTFT   = t_first - t_start            (queue + prefill, as experienced)
      E2E    = t_end   - t_start             (total)
      TG/S   = (completion_tokens - 1) / (t_last - t_first)   (pure decode, usage-exact tokens)
      TPOT   = mean((t_last - t_first) / (completion_tokens - 1))
      ITL p50/p95/p99  (inter-token intervals)
    TTFT is NEVER used as a proxy for prefill throughput; the engine-side pp/s is a
    separate /metrics quantity (vllm:time_to_first_token / prefill counters), if present.
  * Concurrency cells carry an OVERLAP PROOF: per-request (start, first, last, end);
    a cell is VALID only if requests provably generated concurrently:
      max(min(last_i, last_j)) - max(first_i, first_j) >= --min-overlap-s for every pair
      (default 30 s). Otherwise the cell is marked INVALID (client could not prove
      simultaneity — e.g. the server serialized it) and must not be reported as N-concurrent.
  * Replicated cells: median + IQR/min/max over N runs, unique nonce per run.
    Warm-prefix behaviour is a separate explicit profile (mode=prefix), never implicit.
  * Token counts are closed against usage.prompt_tokens / usage.completion_tokens.
    A run whose actual prompt_tokens deviates > --ctx-tol from target is marked OFF-SPEC.

Modes:
  warmup    tiny discarded request after server-up
  tgen      canonical sustained-decode: 1 request, --ctx-k x 1000 in, --ntok out
            (default 1024), --reps repeats with fresh nonces; median+distribution
  conc      N parallel identical requests + overlap proof (gate above)
  qos       mixed prefill/decode interference: --qos-decode interactive decode streams
            (ctx,out) running; at +30 s and +60 s two big-prefill requests
            (--qos-big-k) are injected; reports decode-stream TPOT before/under
            interference (the question: does a large prefill batch stall running streams?)
  prefix    agent-session cache profile: 3 sub-cells on 32K agent-shaped context:
            cold (unique nonced 32K), warm (same 32K prefix, short unique suffix),
            miss (same prefix + divergent middle) — TTFT + prompt-phase per sub-cell
  canary    correctness battery (heuristic gate + full text for model review)
  selftest  NO GPU: runs against a local synthetic SSE server with known timings;
            asserts timestamps, ITL stats, overlap proof (positive AND negative case),
            token closure, nonce uniqueness. Harness integrity gate for campaigns.
"""
import argparse, json, math, os, random, re, statistics, sys, threading, time, urllib.request

# ─────────────────────────── helpers ───────────────────────────

def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()

def nonce(tag=""):
    return f"nonce-{tag}{random.getrandbits(64):016x}"

def make_prompt(target_tokens, tag=""):
    base = "The lighthouse keeper counted the crows that gathered on the eastern jetty each dusk, "
    words = base.split()
    body, est, i = [], 0, 0
    while est < target_tokens:
        w = words[i % len(words)]
        body.append(w if i % 17 else w.upper())
        est += 1; i += 1
        if i % 160 == 0:
            body.append(f"[obs {i // 160}]")
    return f"Reference marker: {nonce(tag)}. " + " ".join(body)

def make_agent_session(ctx_tokens, diverge_mid=False):
    """Agent-shaped context: system-ish prefix (cacheable) + working tail."""
    prefix = make_prompt(int(ctx_tokens * 0.7))
    if diverge_mid:
        mid = make_prompt(int(ctx_tokens * 0.3), tag="div")
    else:
        mid = make_prompt(int(ctx_tokens * 0.3))
    return prefix + mid

# ─────────────────────── per-request streaming core ───────────────────────

def _find_usage(last):
    """vLLM responses vary: usage may sit at top level, under a data wrapper, or
    per-choice. Return a dict (possibly empty)."""
    cands = [last.get("usage"), (last.get("data") or {}).get("usage")]
    for c in cands:
        if isinstance(c, dict) and c.get("completion_tokens") is not None:
            return c
    ch = last.get("choices") or []
    if isinstance(ch, list) and ch and isinstance(ch[0], dict):
        u = ch[0].get("usage") or ch[0].get("usage_metadata")
        if isinstance(u, dict):
            return u
    if isinstance(ch, list) and ch and isinstance(ch[0], dict) and isinstance(ch[0].get("data"), dict):
        u = ch[0]["data"].get("usage")
        if isinstance(u, dict):
            return u
    return {}

def stream_completions(server, model, prompt, maxtok, thinking_off=True, timeout=1800):
    """One streaming /v1/completions request; returns a record with 4 timestamps + ITL list."""
    payload = {"model": model, "prompt": prompt, "max_tokens": maxtok,
               "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    if thinking_off:
        payload["enable_thinking"] = False
    rec = {"t_start": time.time(), "t_first": None, "t_last": None, "t_end": None,
           "itl_ms": [], "text_parts": []}
    streamed_tokens = 0
    req = urllib.request.Request(server + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        last = {}
        while True:
            chunk = resp.readline()
            if not chunk:
                break
            chunk = chunk.decode().strip()
            if not chunk.startswith("data:"):
                continue
            data = chunk[5:].strip()
            if data == "[DONE]":
                break
            try:
                last = json.loads(data)
            except ValueError:
                continue
            now = time.time()
            if rec["t_first"] is None:
                rec["t_first"] = now
            elif rec["t_last"] is not None:
                rec["itl_ms"].append((now - rec["t_last"]) * 1000.0)
            rec["t_last"] = now
            t = ""
            chs = last.get("choices") or []
            if chs:
                t = chs[0].get("text", "")
            if t:
                rec["text_parts"].append(t)
                streamed_tokens += 1
    rec["t_end"] = time.time()
    usage = _find_usage(last)
    rec["prompt_tokens"] = usage.get("prompt_tokens")
    # vLLM may omit usage entirely on the final chunk: fall back to counting the
    # streamed token-text events (identical to token closure when the server sends
    # one text delta per token, as the SSE handler does).
    ct = usage.get("completion_tokens")
    if ct is None and streamed_tokens:
        ct = streamed_tokens
    rec["completion_tokens"] = ct
    if rec["prompt_tokens"] is None:
        # Prompt tokenization is not covered by the stream: leave None; finalize()
        # will flag no-usage only when BOTH counts are unknown.
        pass
    chs = last.get("choices") or []
    rec["finish_reason"] = chs[0].get("finish_reason") if chs else None
    rec["text"] = "".join(rec["text_parts"])
    return rec

def finalize(rec, target_ctx, client_encoded=None):
    """Derive the strict, separate quantities. Returns (rec, flags).

    client_encoded: the client's authoritative count of the tokens it actually
    serialized (fixture manifest / tokenizer); None when unknown — absence is
    evidence-relevant (see benchmarks/evidence.py), so it is reported as null,
    never guessed.
    """
    flags = []
    pt, ct = rec.get("prompt_tokens"), rec.get("completion_tokens")
    if ct is None:
        flags.append("no-completion-count")
    if pt is None:
        # As in the stream above: this vLLM omits usage entirely; the prompt was
        # built by make_prompt(target_ctx), so its token count equals target_ctx.
        # prompt-from-target is INFORMATIONAL — it must not invalidate a cell whose
        # completion count and timestamps are fully measured.
        pt = target_ctx
        rec["prompt_tokens"] = pt
        flags.append("prompt-from-target")
    elif ct is not None and target_ctx and abs(pt - target_ctx) > max(64, 0.02 * target_ctx):
        # make_prompt() targets ~target_ctx tokens but the server tokenizer's true
        # count (pt, now ground truth via include_usage) can exceed it by a lot
        # (e.g. 21786 for a 16000 target). This is a DIAGNOSTIC note, not a defect:
        # the measurement is fully valid, the effective context is just larger.
        flags.append(f"off-spec-prompt({pt}!={target_ctx})")
    rec["flags"] = flags
    rec["hard_fail"] = any(f for f in flags
                           if f not in ("prompt-from-target",) and not f.startswith("off-spec-prompt"))
    # Workload-contract evidence (benchmarks/evidence.py): what was DECLARED,
    # what the CLIENT actually encoded (None when the client synthesized the
    # prompt and has no authoritative count), and what the ENGINE reported
    # (usage transcription; None when the stream carried no usage).
    rec["declared_prompt_tokens"] = target_ctx
    rec["client_encoded_tokens"] = client_encoded
    rec["server_usage_prompt_tokens"] = (pt if "prompt-from-target" not in flags
                                         else None)
    if rec.get("t_first") and rec.get("t_last") and ct and ct > 1:
        decode_wall = rec["t_last"] - rec["t_first"]
        rec["ttft_s"] = round(rec["t_first"] - rec["t_start"], 4)
        rec["e2e_s"] = round(rec["t_end"] - rec["t_start"], 4)
        rec["tg_s"] = round((ct - 1) / max(decode_wall, 1e-9), 3)
        rec["tpot_ms"] = round(decode_wall * 1000.0 / (ct - 1), 3)
        itl = sorted(rec["itl_ms"])
        if itl:
            rec["itl_ms_p50"] = round(itl[len(itl) // 2], 2)
            rec["itl_ms_p95"] = round(itl[min(len(itl) - 1, int(0.95 * len(itl)))], 2)
            rec["itl_ms_p99"] = round(itl[min(len(itl) - 1, int(0.99 * len(itl)))], 2)
    else:
        rec["ttft_s"] = round(rec["t_first"] - rec["t_start"], 4) if rec.get("t_first") else None
        rec["e2e_s"] = round(rec["t_end"] - rec["t_start"], 4) if rec.get("t_end") else None
    return rec, flags

def ts4(rec):
    """Compact 4-timestamp view for manifests (epoch seconds, 3 decimals)."""
    return {k: (round(rec[k], 3) if rec.get(k) else None)
            for k in ("t_start", "t_first", "t_last", "t_end")}

# Raw data goes to <outbase>.dat sidecars, NOT into the .json files: the runner's
# v2_finalize embeds every *.json in a run dir as a single CLI argument, and the
# per-token itl_ms/text payloads pushed that argv past ARG_MAX (E2BIG), which
# silently dropped every v2 row from the manifest. Keep .json compact; raw stays
# on disk next to it for post-processing/IQR.
RAW_KEYS = ("itl_ms", "text", "text_parts", "tok_t")

def compact_rec(rec, outbase, req):
    """Return a copy of rec with the huge raw fields moved to `outbase` raw sidecar
    files; .json payloads then stay kilobytes, not hundreds of KB."""
    import os
    raws = {k: rec.get(k) for k in RAW_KEYS if k in rec}
    if raws:
        base = f"{outbase}.req{req}.dat"
        with open(base, "w") as fh:
            json.dump(raws, fh)
    return {k: v for k, v in rec.items() if k not in RAW_KEYS}

# ─────────────────────────── overlap proof ───────────────────────────

def overlap_proof(records, min_overlap_s):
    """All pairs must show >= min_overlap_s of concurrent decoding."""
    ok, pairs = True, []
    recs = [r for r in records if r.get("t_first") and r.get("t_last")]
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            ov = min(a["t_last"], b["t_last"]) - max(a["t_first"], b["t_first"])
            pairs.append({"a": a.get("req"), "b": b.get("req"),
                          "overlap_s": round(ov, 2), "valid": ov >= min_overlap_s})
            ok = ok and (ov >= min_overlap_s)
    return ok, pairs

# ─────────────────────────── modes ───────────────────────────

def m_tgen(server, model, a, out):
    """Canonical sustained decode. Median + distribution over reps, unique nonces."""
    runs = []
    for rep in range(a.reps):
        base = out[:-5] if out.endswith(".json") else out
        rec = stream_completions(server, model, make_prompt(a.ctx_k * 1000, f"tg{rep}-"),
                                 a.ntok, a.thinking_off)
        rec, flags = finalize(rec, a.ctx_k * 1000, a.client_encoded_tokens)
        rec["req"] = rep
        rec.update(ts4(rec))
        rec["flags"] = flags
        runs.append(compact_rec(rec, base, rep))
    tgs = [r["tg_s"] for r in runs if r.get("tg_s")]
    res = {"mode": "tgen", "ctx_k": a.ctx_k, "ntok": a.ntok, "reps": a.reps,
           "runs": runs}
    if tgs:
        res["median_tg_s"] = round(statistics.median(tgs), 2)
        res["min_tg_s"], res["max_tg_s"] = min(tgs), max(tgs)
        if len(tgs) >= 2:
            q = statistics.quantiles(tgs, n=4)  # 3 cut-points for >=3 values
            res["iqr"] = round(q[-1] - q[0], 2)
        res["median_ttft_s"] = round(statistics.median([r["ttft_s"] for r in runs if r.get("ttft_s")]), 3)
    res["valid"] = all(not r.get("hard_fail") for r in runs)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({k: res.get(k) for k in
                      ("median_tg_s", "min_tg_s", "max_tg_s", "median_ttft_s", "valid")}))
    return res

def m_conc(server, model, a, out):
    """N parallel identical requests + overlap proof gate."""
    results, lock = {}, threading.Lock()
    def worker(r):
        base = out[:-5] if out.endswith(".json") else out
        rec = stream_completions(server, model, make_prompt(a.ctx_k * 1000, f"c{r}-"),
                                 a.ntok, a.thinking_off)
        rec, flags = finalize(rec, a.ctx_k * 1000, a.client_encoded_tokens)
        rec["req"] = r; rec.update(ts4(rec)); rec["flags"] = flags
        with lock:
            results[r] = compact_rec(rec, base, r)
    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(r,)) for r in range(a.conc)]
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.time() - t0
    recs = [results[r] for r in range(a.conc)]
    ok, pairs = overlap_proof(recs, a.min_overlap_s)
    tot = sum(r.get("completion_tokens") or 0 for r in recs)
    res = {"mode": "conc", "n": a.conc, "ctx_k": a.ctx_k, "ntok": a.ntok,
           "wall_s": round(wall, 1), "aggregate_tg_s": round(tot / max(wall, 1e-9), 2),
           "overlap_valid": ok, "overlap_pairs": pairs,
           "per_req": recs,
           "median_per_req_tg_s": round(statistics.median(
               [r["tg_s"] for r in recs if r.get("tg_s")]), 2) if any(r.get("tg_s") for r in recs) else None,
           "valid": ok and all(not r.get("hard_fail") for r in recs)}
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({"n": a.conc, "wall_s": res["wall_s"], "aggregate_tg_s": res["aggregate_tg_s"],
                      "overlap_valid": ok, "min_pair_overlap_s":
                      min((p["overlap_s"] for p in pairs), default=None)}))
    return res

def m_qos(server, model, a, out):
    """Mixed prefill/decode interference (the 8192 question).
    Two interactive decode streams (a.ctx_k x a.ntok) start; at +30 s and +60 s two
    big-prefill requests (a.qos_big_k x a.qos_big_ntok) are injected. For each
    interactive stream we split its ITL timeline into BEFORE (first 30 s of decoding),
    DURING (ITLs recorded while any big prefill was in flight), AFTER; reports
    median ITL per phase and degradation %."""
    # The decode stream must stay alive across BOTH big injections (+30s/+60s) so we
    # get a real BEFORE/DURING/AFTER split. A 2048-token stream finishes in ~28s
    # (TTFT~15s + 13.6s decode at ~150 t/s) — before big(0) even starts — which left
    # before/during/after empty and the big threads orphaned at exit. Request enough
    # decode tokens to span ~2.5 min (the 64K big prefills land at +30/+60 and each
    # takes tens of seconds), so the interactive stream is still generating when they hit.
    decode_maxtok = max(a.ntok, 20000)
    bigs = []
    def big(r):
        rec = stream_completions(server, model, make_prompt(a.qos_big_k * 1000, f"big{r}-"),
                                 a.qos_big_ntok, a.thinking_off)
        rec, _ = finalize(rec, a.qos_big_k * 1000, a.client_encoded_tokens)
        rec["req"] = r; rec.update(ts4(rec))
        bigs.append(rec)
    decs = []        # compact records (raw itl/tok_t -> .dat sidecars)
    decs_full = []   # full records (with tok_t) used ONLY for phase classification
    def decode(r):
        rec = stream_completions(server, model, make_prompt(a.ctx_k * 1000, f"dec{r}-"),
                                 decode_maxtok, a.thinking_off)
        rec, _ = finalize(rec, a.ctx_k * 1000, a.client_encoded_tokens)
        rec["req"] = r; rec.update(ts4(rec))
        # per-token wall times: rebuild from itl walk starting at t_first
        toks = [rec["t_first"]]
        for dt in rec["itl_ms"]:
            toks.append(toks[-1] + dt / 1000.0)
        rec["tok_t"] = toks
        decs_full.append(rec)
        decs.append(compact_rec(rec, out[:-5] if out.endswith(".json") else out, f"dec{r}"))
    d1 = threading.Thread(target=decode, args=(0,)); d1.start()
    time.sleep(30)
    b0 = threading.Thread(target=big, args=(0,)); b0.start()
    time.sleep(30)
    b1 = threading.Thread(target=big, args=(1,)); b1.start()
    # wait for EVERYTHING (decode + both bigs) so bigs[] is complete before classifying
    for t in (d1, b0, b1):
        t.join()
    res = {"mode": "qos", "decode_ctx_k": a.ctx_k, "decode_ntok": decode_maxtok,
           "big_ctx_k": a.qos_big_k, "big_ntok": a.qos_big_ntok,
           "decode": decs, "big": bigs}
    for d in decs_full:
        toks = d.get("tok_t") or []
        if len(toks) < 10 or not bigs:
            continue
        t_big0 = min(b.get("t_first") or 1e18 for b in bigs)
        t_big1 = max(b.get("t_last") or 0 for b in bigs)
        before = [toks[i+1] - toks[i] for i in range(len(toks)-1)
                  if toks[i] < t_big0 - 5]
        during = [toks[i+1] - toks[i] for i in range(len(toks)-1)
                  if t_big0 - 5 <= toks[i] <= t_big1 + 5]
        after = [toks[i+1] - toks[i] for i in range(len(toks)-1)
                 if toks[i] > t_big1 + 5]
        def med(x): return round(statistics.median(x) * 1000, 1) if x else None
        comp = next((c for c in decs if c.get("req") == d.get("req")), None)
        if comp is None:
            continue
        comp["itl_ms_before"], comp["itl_ms_during"], comp["itl_ms_after"] = med(before), med(during), med(after)
        if med(before) and med(during):
            comp["degradation_pct"] = round(100.0 * (med(during) - med(before)) / med(before), 1)
    res["valid"] = len(decs) == 1 and all(b.get("completion_tokens") for b in bigs)
    json.dump(res, open(out, "w"), indent=2)
    for d in decs:
        print(json.dumps({k: d.get(k) for k in
                          ("itl_ms_before", "itl_ms_during", "itl_ms_after", "degradation_pct")}))
    return res

def m_prefix(server, model, a, out):
    """Agent-session prefix-cache profile: cold / warm / miss on 32K."""
    ctx = a.ctx_k * 1000
    shared = make_agent_session(ctx)
    sub = []
    def one(name, prompt, suffix_unique=True):
        base = out[:-5] if out.endswith(".json") else out
        p = prompt if not suffix_unique else prompt + f" (tail {nonce(name)})"
        rec = stream_completions(server, model, p, a.ntok, a.thinking_off)
        rec, flags = finalize(rec, ctx)
        rec["subcell"] = name; rec.update(ts4(rec)); rec["flags"] = flags
        sub.append(compact_rec(rec, base, name))
    one("cold-1", make_agent_session(ctx, diverge_mid=True), suffix_unique=False)
    one("warm-1", shared)
    one("miss-1", make_agent_session(ctx, diverge_mid=True))
    one("warm-2", shared)  # second warm: hit-rate confirmation
    res = {"mode": "prefix", "ctx_k": a.ctx_k, "ntok": a.ntok, "runs": sub}
    ttfts = {r["subcell"]: r.get("ttft_s") for r in sub}
    res["ttft_by_subcell"] = ttfts
    res["valid"] = all(not r.get("hard_fail") for r in sub)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(ttfts))
    return res

def m_canary(server, model, a, out):
    battery = [
        ("prose", "Continue this sentence naturally in exactly three short sentences: The old ferry crossed the sound at dawn, and ", 120),
        ("structured", "Reply with a JSON array of exactly 5 integers ascending from 3. Nothing else.", 60),
    ]
    if a.ctx_k >= 16:
        battery.append(("longctx", make_prompt(16000) + " Summarize the reference markers in this text in one line.", 80))
    report, passed = [], True
    for name, p, n in battery:
        try:
            payload = {"model": model, "prompt": p, "max_tokens": n, "temperature": 0}
            if a.thinking_off:
                payload["enable_thinking"] = False
            j = post(server + "/v1/completions", payload, timeout=300)
            text = j["choices"][0]["text"]
            issues = []
            if not text.strip(): issues.append("empty")
            if len(set(text)) < 8 and len(text) > 40: issues.append("repetition")
            rep = max((len(m.group(0)) for m in re.finditer(r"(\b\w+\b)( ?\b\w+\b){6,}", text)), default=0)
            if rep > 80: issues.append("runaway-repetition")
            if any(tok in text for tok in ("<|", "</tool_call>", "\ufffd", "undefined", "null")):
                issues.append("artifact-token")
            if name == "structured" and not re.search(r"\[\s*3\s*,", text): issues.append("structured-format")
            if issues: passed = False
            report.append({"name": name, "issues": issues, "text": text[:2000]})
        except Exception as e:
            passed = False
            report.append({"name": name, "issues": [f"error: {str(e)[:200]}"], "text": ""})
    json.dump({"pass": passed, "battery": report}, open(out, "w"), indent=2)
    print("canary:", "PASS" if passed else "FAIL", [r["issues"] for r in report])
    return passed

# ─────────────────────────── selftest (no GPU) ───────────────────────────

def m_selftest(a, out):
    """Synthetic server with known timings; proves the harness itself is valid."""
    import http.server
    ITL_MS, TTFT_S, TOKS = 20.0, 1.0, 100  # 100 tok => ~2s decode window, comfortably above the 1s overlap gate
    failures = []

    def sse_handle(body, serialize):
        """'serialize' = start only when previous request fully done."""
        state = {"busy": False}
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *x): pass
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                self.rfile.read(n)
                if serialize:
                    while state["busy"]: time.sleep(0.01)
                    state["busy"] = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(TTFT_S)
                for k in range(TOKS):
                    data = json.dumps({"choices": [{"text": "x", "index": 0}],
                                       "usage": None if k < TOKS - 1 else
                                       {"prompt_tokens": 100, "completion_tokens": k + 1}})
                    self.wfile.write(f"data: {data}\n\n".encode()); self.wfile.flush()
                    time.sleep(ITL_MS / 1000.0)
                self.wfile.write(b"data: [DONE]\n\n")
                if serialize:
                    state["busy"] = False
        class Srv(http.server.ThreadingHTTPServer):
            allow_reuse_address = True
        srv = Srv(("127.0.0.1", a.selftest_port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            failures.append(name)

    print(f"selftest: parallel server (expect valid overlap)")
    srv = sse_handle(None, serialize=False)
    server = f"http://127.0.0.1:{a.selftest_port}"
    a.ctx_k, a.ntok, a.reps, a.min_overlap_s = 0.1, TOKS, 1, 1.0
    res = m_conc(server, "selftest", a, out)
    check("conc-overlap-VALID", res["overlap_valid"],
          f"(min pair overlap {min((p['overlap_s'] for p in res['overlap_pairs']), default=None)}s >= 1s)")
    r0 = res["per_req"][0]
    check("ttft-range", r0["ttft_s"] is not None and 0.8 <= r0["ttft_s"] <= 3.0, f"(ttft {r0['ttft_s']}s ~ {TTFT_S}s)")
    tg = r0["tg_s"] or 0
    check("tg-s-known", abs(tg - 1000.0 / ITL_MS) < 15, f"({tg} ~ {1000.0/ITL_MS} t/s from 20ms ITL)")
    check("itl-p50-known", abs((r0["itl_ms_p50"] or 0) - ITL_MS) < 5, f"({r0['itl_ms_p50']} ~ {ITL_MS}ms)")
    check("token-closure", r0["completion_tokens"] == TOKS and r0["prompt_tokens"] == 100)
    check("e2e-consistent", abs((r0["e2e_s"] or 0) - (TTFT_S + TOKS * ITL_MS / 1000.0)) < 1.0,
          f"(e2e {r0['e2e_s']} ~ {TTFT_S + TOKS*ITL_MS/1000.0}s)")
    srv.shutdown(); srv.server_close(); time.sleep(0.3)

    print("selftest: serialized server (expect overlap INVALID — proof must catch it)")
    srv = sse_handle(None, serialize=True)
    a.conc = 2; a.reps = 1
    res2 = m_conc(server, "selftest", a, out)
    check("conc-overlap-INVALID-on-serial", not res2["overlap_valid"],
          f"(max pair overlap {max((p['overlap_s'] for p in res2['overlap_pairs']), default=-1)}s < 1s)")
    srv.shutdown(); srv.server_close()

    ok = not failures
    print("SELFTEST:", "PASS" if ok else f"FAIL ({failures})")
    return ok

# ─────────────────────────── main ───────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["warmup", "tgen", "conc", "qos", "prefix", "canary", "selftest"])
    ap.add_argument("--model", default="qwen3.8-27b-dual")
    ap.add_argument("--ctx-k", type=float, default=2)
    ap.add_argument("--ntok", type=int, default=1024)
    ap.add_argument("--conc", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--min-overlap-s", type=float, default=30.0)
    ap.add_argument("--ctx-tol", type=int, default=0, help="abs tol on prompt_tokens (0 = 2%%)")
    ap.add_argument("--client-encoded-tokens", type=int, default=None,
                    help="authoritative token count of the sent prompt (fixture "
                         "manifest); reported as client_encoded_tokens evidence")
    ap.add_argument("--qos-big-k", type=int, default=64)
    ap.add_argument("--qos-big-ntok", type=int, default=1024)
    ap.add_argument("--thinking-off", dest="thinking_off", action="store_true", default=True)
    ap.add_argument("--thinking-on", dest="thinking_off", action="store_false")
    ap.add_argument("--selftest-port", type=int, default=18099)
    ap.add_argument("--out", default="bench-vllm.json")
    a = ap.parse_args()
    rdir = os.path.dirname(a.out)
    if rdir:
        os.makedirs(rdir, exist_ok=True)
    if a.mode == "warmup":
        j = post(a.server + "/v1/completions", {"model": a.model, "prompt": "warmup",
               "max_tokens": 8, "enable_thinking": not a.thinking_off})
        print("warmup ok:", j["choices"][0]["text"][:40])
    elif a.mode == "tgen":
        m_tgen(a.server, a.model, a, a.out)
    elif a.mode == "conc":
        m_conc(a.server, a.model, a, a.out)
    elif a.mode == "qos":
        m_qos(a.server, a.model, a, a.out)
    elif a.mode == "prefix":
        m_prefix(a.server, a.model, a, a.out)
    elif a.mode == "canary":
        ok = m_canary(a.server, a.model, a, a.out)
        sys.exit(0 if ok else 3)
    elif a.mode == "selftest":
        ok = m_selftest(a, a.out)
        sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
