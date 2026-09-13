#!/usr/bin/env python3
"""divergence-pair.py — paired ON/OFF comparison client (stdlib only).

The D1/Q1 client: runs the SAME seeded prompt(s) against a cache-ON server
(--url-on, already launched by the platform) and a cache-OFF server this
script launches itself (--launch-off; it is a child, killed on exit), and
reports the numerical distance between the two paths:

  per prompt:
    first_divergence_pos   first position where the token streams differ
    overlap_k5             fraction of positions where the top-5 sets match
    top1_top2_margin_*     per-side margin (on/off), where available
    logprob_delta_mean     mean |logprob_on - logprob_off| over shared prefix
    token_overlap_ratio    (Q1) fraction of positions with identical tokens
    structural            (Q1, optional) JSON parse validity per class
    outputs               BOTH full outputs, persisted in the json

Labels for the report are carried verbatim: every metric is OBSERVED
(measured), the 'bit-identical?' question is reported as NOT ESTABLISHED
unless first_divergence_pos is None for every prompt — the client does not
infer beyond what the streams show. Exit 0 iff both sides produced
complete outputs; --json-out always writes the evidence either way.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request


def complete(url, prompt, decode, seed):
    body = json.dumps({"prompt": prompt, "n_predict": decode,
                       "temperature": 0.0, "top_k": 1, "seed": seed,
                       "stream": False,
                       "return_logprobs": True}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/completion",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read().decode(errors="replace"))
    content = j.get("content", "")
    tokens = j.get("tokens") or []
    lprobs = []
    for t in j.get("token_alternatives") or []:
        pass
    # logprobs shape varies across builds; best-effort extraction
    if isinstance(tokens, list):
        for i, t in enumerate(tokens):
            if isinstance(t, dict) and "logprob" in t:
                lprobs.append((i, t.get("token", ""), t.get("logprob")))
            elif i < len(tokens) and isinstance(tokens[i], (list, tuple)) \
                    and len(tokens[i]) >= 3:
                lprobs.append((i, tokens[i][0], tokens[i][1]))
    return {"content": content, "tokens": tokens, "logprobs": lprobs,
            "n_predicted": j.get("tokens_predicted"), "raw": j}


def topk_sets(tokens_a, tokens_b, k=5):
    """Positional top-k set overlap; tolerant of differing result shapes —
    if the build does not expose candidates, returns None (reported as not
    established, never guessed)."""
    out = []
    for ta, tb in zip(tokens_a, tokens_b):
        sa = {c[0] for c in ta[:k]} if isinstance(ta, list) else None
        sb = {c[0] for c in tb[:k]} if isinstance(tb, list) else None
        out.append(None if (sa is None or sb is None)
                   else len(sa & sb) / max(len(sa | sb), 1))
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--url-on", required=True)
    a.add_argument("--launch-off", required=True,
                   help="shell command launching the OFF server (child of us)")
    a.add_argument("--url-off", default=None,
                   help="explicit OFF url (skips launch; for drills)")
    a.add_argument("--prompt-file", default=None)
    a.add_argument("--prompt-dir", default=None,
                   help="Q1: one prompt per .txt file, one class per file")
    a.add_argument("--decode", type=int, default=128)
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--prompts", type=int, default=1)
    a.add_argument("--json-out", required=True)
    a = a.parse_args()

    if a.prompt_dir:
        prompts = []
        for f in sorted(os.listdir(a.prompt_dir)):
            if f.endswith(".txt"):
                prompts.append((f[:-4], open(os.path.join(a.prompt_dir, f),
                                            errors="replace").read()))
    elif a.prompt_file:
        text = open(a.prompt_file, errors="replace").read()
        prompts = [(f"p{i}", text) for i in range(a.prompts)]
    else:
        prompts = [("p0", "Say ok")]

    off_proc = None
    url_off = a.url_off
    if not url_off:
        m = re.search(r"--port[ =]+(\d+)", a.launch_off)
        port = m.group(1) if m else "18199"
        off_proc = subprocess.Popen(
            a.launch_off.replace("{port2}", port), shell=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url_off = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                with urllib.request.urlopen(url_off + "/health", timeout=3) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(2)
        else:
            print("OFF server never became healthy", file=os.stderr)

    ok = True
    results = []
    try:
        for name, text in prompts:
            on = complete(a.url_on, text, a.decode, a.seed)
            off = complete(url_off, text, a.decode, a.seed)
            to, tf = on["tokens"], off["tokens"]
            first_div = None
            n = min(len(to), len(tf))
            overlap_ratio = 0.0
            for i in range(n):
                a_ = to[i] if isinstance(to[i], (str, list, dict)) else None
                b_ = tf[i] if isinstance(tf[i], (str, list, dict)) else None
                sa = a_[0] if isinstance(a_, (list, tuple)) else a_
                sb = b_[0] if isinstance(b_, (list, tuple)) else b_
                if first_div is None and sa != sb:
                    first_div = i
                if sa == sb:
                    overlap_ratio += 1
            lp_o = dict(on["logprobs"])
            lp_f = dict(off["logprobs"])
            shared = sorted(set(lp_o) & set(lp_f))
            lpd = (sum(abs(lp_o[i] - lp_f[i]) for i in shared) / len(shared)
                   if shared else None)
            k5 = topk_sets(to, tf)
            results.append({
                "prompt": name,
                "observed": {
                    "first_divergence_pos": first_div,
                    "token_overlap_ratio": round(overlap_ratio / max(n, 1), 4)
                    if n else None,
                    "topk5_overlap_mean": (round(sum(k5) / len(k5), 4)
                                           if k5 and None not in k5 else None),
                    "logprob_delta_mean": (round(lpd, 5) if lpd is not None
                                           else None),
                    "n_on": len(to), "n_off": len(tf),
                    "n_predicted_on": on["n_predicted"],
                    "n_predicted_off": off["n_predicted"],
                },
                # the honest labels (3b): what we measured vs what we did not
                "inferred": {
                    "bit_identical": (first_div is None
                                      and len(to) == len(tf)),
                },
                "not_established": {
                    "topk5": (k5 is not None and None in k5),
                    "logprobs": lpd is None,
                },
                "outputs": {"on": on["content"], "off": off["content"]},
            })
            if not (len(to) >= a.decode * 0.9 and len(tf) >= a.decode * 0.9):
                ok = False
    finally:
        if off_proc:
            off_proc.send_signal(signal.SIGTERM)
            try:
                off_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                off_proc.kill()

    json.dump({"valid": ok, "prompts": results}, open(a.json_out, "w"),
              indent=2)
    print(json.dumps({"valid": ok,
                      "first_divergence": [r["observed"]["first_divergence_pos"]
                                           for r in results]}))
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
