"""bench-llama.py — llama.cpp /completion benchmark client (platform client).

Ported from the field-proven llama.cpp bench client (2026-09-12 campaign), deduplicated
(the original carried three copies of the log-parser from iterative field edits;
this keeps the version main() actually used).

Measures per request: TTFT, per-chunk decode t/s (token-weighted, correct under
MTP), total wall, full-text sha256 (determinism canary). If a --server-log is
given, the server's own per-request summary line wins over client estimates
(server-side exact numbers). If /metrics exposes spec-decode counters, reports
acceptance (MTP cells).

Stdlib only.
"""
import argparse, json, hashlib, re, sys, time, urllib.request


def http_post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def run_one(base, prompt, decode, rep, seed):
    t0 = time.monotonic()
    body = {"prompt": prompt, "n_predict": decode, "stream": True,
            "temperature": 0, "top_k": 1, "seed": seed, "cache_prompt": False}
    resp = http_post(base + "/completion", body)
    ttft = None; ntok = 0; chunks = []
    first_tok_t = None; last_tok_t = None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            break
        j = json.loads(d)
        tok = j.get("content", "")
        if tok:
            if first_tok_t is None:
                first_tok_t = time.monotonic(); ttft = first_tok_t - t0
            last_tok_t = time.monotonic()
            # token count from the frame if present, else one per chunk
            ntok += j.get("token_id_count", 1) if "token_id" in j or "token_ids" in j else 1
            chunks.append(tok)
    total = time.monotonic() - t0
    text = "".join(chunks)
    span = (last_tok_t - first_tok_t) if (first_tok_t and last_tok_t and last_tok_t > first_tok_t) else 0.0
    out = {
        "rep": rep, "ttft_s": round(ttft, 4) if ttft else None,
        "decode_tokens": ntok, "decode_span_s": round(span, 4),
        "decode_tps": round(ntok / span, 3) if span > 0 else None,
        "total_s": round(total, 4),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "len_chars": len(text),
    }
    return out, text


def server_log_line(path, before_marker_count):
    """Parse the latest 'prompt eval ... | eval time = X ms | N tokens' line from a
    running llama-server log. Returns (prompt_tokens, prompt_ms, gen_tokens, gen_ms, tps).
    llama.cpp vintages differ slightly; we accept several shapes."""
    if not path:
        return None
    try:
        data = open(path, errors="replace").read()
    except OSError:
        return None
    lines = [l for l in data.splitlines() if "eval time" in l or ("prompt eval" in l and "tokens" in l)]
    if len(lines) <= before_marker_count:
        return None
    line = lines[-1]
    m = re.search(r"prompt eval time = (\d+\.?\d*) ms", line)
    g = re.search(r"eval time = (\d+\.?\d*) ms", line)
    t = re.search(r"(\d+) tokens in (\d+\.?\d*) ms", line) or re.search(r"(\d+) tokens \| (\d+\.?\d*) t/s", line)
    if not (g and t):
        return None
    tok, val = int(t.group(1)), float(t.group(2))
    tps = val if "t/s" in line else tok * 1000.0 / val
    return (int(m.group(1)) if m else None, float(g.group(1)), tok, val, tps)


def metrics_spec(base):
    try:
        txt = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode("utf-8", "replace")
    except Exception:
        return None
    acc = rej = 0
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) != 2:
            continue
        # spec-decode accept/reject counters (names vary by vintage; take all spec*)
        if "spec" in p[0] and ("accept" in p[0] or "reject" in p[0] or "draft" in p[0]):
            try:
                v = float(p[1])
            except ValueError:
                continue
            if "accept" in p[0]:
                acc += v
            elif "reject" in p[0]:
                rej += v
    return (acc, rej)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--decode", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="")
    ap.add_argument("--json-out")
    ap.add_argument("--server-log", default=None,
                    help="llama-server log file; exact decode numbers parsed from its per-request summary line")
    ap.add_argument("--save-text", default=None, help="write full generated text to this path")
    ap.add_argument("--warmup", type=int, default=0,
                    help="run N throwaway reps before the measured reps (page-cache/"
                         "graph warmup); discarded, but recorded as the 'warmup' count")
    a = ap.parse_args()
    prompt = open(a.prompt_file).read()
    rows = []
    full = []
    texts = []

    for _w in range(a.warmup):  # discarded; the measured reps are the data
        try:
            run_one(a.url, prompt, a.decode, -1, a.seed)
        except Exception as e:
            print(f"warmup rep failed: {type(e).__name__}: {e}", flush=True)

    def log_count():
        if not a.server_log:
            return 0
        try:
            return sum(1 for l in open(a.server_log, errors="replace")
                       if "eval time" in l or ("prompt eval" in l and "tokens" in l))
        except OSError:
            return 0

    for r in range(a.reps):
        before = log_count()
        t0 = time.monotonic()
        row, text = run_one(a.url, prompt, a.decode, r, a.seed + r)
        texts.append(text)
        row["wall_s"] = round(time.monotonic() - t0, 4)
        truth = server_log_line(a.server_log, before)
        if truth:  # server-side exact numbers win over client-side estimates
            row["prompt_tokens"] = truth[0]
            row["prompt_ms"] = truth[1]
            row["gen_tokens"] = truth[2]
            row["gen_ms"] = truth[3]
            row["server_decode_tps"] = round(truth[4], 3)
            row["server_prefill_tps"] = round(truth[2] * 1000.0 / truth[1], 2) if truth[1] else None
            row["source"] = "server-log"
        rows.append(row)
        full.append(row["sha256"])
        row2 = {k: row[k] for k in ("rep", "ttft_s", "decode_tps", "server_decode_tps", "sha256") if k in row}
        print(f"rep {r}: {row2}", flush=True)
    uniq = len(set(full))
    det = "OK" if uniq == 1 else ("SELF-REPRO" if uniq > 1 else "?")
    if len(full) == 1:
        det = "SINGLE"
    res = {"tag": a.tag, "url": a.url, "prompt_file": a.prompt_file,
           "prompt_chars": len(prompt), "decode": a.decode, "reps": a.reps,
           "warmup": a.warmup,
           "rows": rows, "determinism": det, "sha_set": sorted(set(full)),
           "valid": det in ("OK", "SINGLE") or det == "?"}
    print(f"determinism: {det} ({uniq} unique of {len(full)})", flush=True)
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=1)
    if a.save_text and texts:
        open(a.save_text, "w").write(texts[-1])  # last rep; reps are deterministic
    # spec counters (best effort, for MTP cells)
    m = metrics_spec(a.url)
    if m and (m[0] or m[1]):
        rate = m[0] / (m[0] + m[1]) if (m[0] + m[1]) else None
        print(f"spec-counters(lifetime): accept={m[0]:.0f} reject={m[1]:.0f} rate={rate}")
        res["spec_counters_lifetime"] = {"accept": m[0], "reject": m[1], "rate": rate}
        if a.json_out:
            json.dump(res, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
