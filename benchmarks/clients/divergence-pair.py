#!/usr/bin/env python3
"""divergence-pair.py — paired ON/OFF comparison client (stdlib only).

The D1/Q1 client: runs the SAME seeded prompt(s) against a cache-ON server
(--url-on, already launched by the platform) and a cache-OFF server this
script launches itself (--launch-off; it is a child, killed on exit), and
reports the numerical distance between the two paths.

Honesty rules (2026-09-13 external review) — this client is for an
UNATTENDED campaign, so it must never manufacture a result:

1. TOKEN NORMALIZATION. Token fields come in different shapes across
   llama.cpp builds (int id / str / dict record / candidate list). Each
   shape is normalized explicitly; an UNKNOWN shape normalizes to None and
   its positions are reported as not_established — never compared, never
   'equal by accident'.
2. IDENTITY WORDING. A bounded sample can only show what it sampled: the
   output says `no_divergence_observed_in_tested_sequences`, and only when
   every tested position was comparable. It never says "bit-identical".
3. METRICS ARE EVIDENCE OR NULL. top-k / logprob metrics are computed only
   when the response actually exposes that data in a recognized shape;
   otherwise they are null + not_established. Unknown shapes degrade to
   MISSING evidence, never fake zero / fake equality.
4. BOTH FULL OUTPUTS are persisted in the json either way.

Exit 0 iff both sides produced complete outputs; --json-out always writes.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import time
import urllib.request


def norm(t):
    """Normalize one token-field element. Returns (kind, value) or None.

    kind in {"id", "str", "tok", "cand"}; None = unknown shape
    (NOT ESTABLISHED, by design — see honesty rule 1)."""
    if isinstance(t, int):
        return ("id", t)
    if isinstance(t, str):
        return ("str", t)
    if isinstance(t, dict):
        if "id" in t and isinstance(t.get("id"), int):
            return ("id", t["id"])
        if "token" in t and isinstance(t.get("token"), str):
            return ("tok", t["token"])
        if "text" in t and isinstance(t.get("text"), str):
            return ("tok", t["text"])
        return None
    if isinstance(t, (list, tuple)) and len(t) > 0:
        inner = norm(t[0])
        if inner is not None:
            return ("cand", inner[1])  # first candidate only
        return None
    return None


def comparable(a, b):
    """Two normalized positions are comparable only when their KINDS match
    (an int-id stream and a text stream are different evidence)."""
    na, nb = norm(a), norm(b)
    if na is None or nb is None:
        return None
    if na[0] != nb[0]:
        return None
    return (na[1], nb[1])


def topk_sets(tokens_a, tokens_b, k=5):
    """Per-position top-k set overlap, ONLY from a recognized candidate
    structure (list/tuple of >=2 entries, each with a comparable head).
    Returns per-position overlaps, or None when the shape is not
    recognized (reported as not_established — never guessed)."""
    out = []
    recognized = 0
    for ta, tb in zip(tokens_a, tokens_b):
        sa, sb = None, None
        if isinstance(ta, (list, tuple)) and len(ta) >= 2 and \
                isinstance(tb, (list, tuple)) and len(tb) >= 2:
            sa = {norm(c)[1] for c in ta[:k] if norm(c) is not None}
            sb = {norm(c)[1] for c in tb[:k] if norm(c) is not None}
            if sa and sb:
                recognized += 1
                out.append(len(sa & sb) / max(len(sa | sb), 1))
        else:
            out.append(None)
    if recognized == 0:
        return None  # the endpoint did not expose candidates in a known shape
    return out


def complete(url, prompt, decode, seed):
    body = json.dumps({"prompt": prompt, "n_predict": decode,
                       "temperature": 0.0, "top_k": 1, "seed": seed,
                       "stream": False}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/completion",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode(errors="replace"))


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
            to = on.get("tokens") or []
            tf = off.get("tokens") or []
            n = min(len(to), len(tf))

            # pairwise positions: compared vs not-established (honesty 1)
            first_div = None
            compared = 0
            not_est = 0
            same = 0
            for i in range(n):
                pair = comparable(to[i], tf[i])
                if pair is None:
                    not_est += 1
                    continue
                compared += 1
                if pair[0] == pair[1]:
                    same += 1
                elif first_div is None:
                    first_div = i

            # logprobs: only positions where BOTH sides expose a logprob
            # for the same index; anything else is missing evidence.
            lp_o = {i: t.get("logprob") for i, t in enumerate(to)
                    if isinstance(t, dict) and "logprob" in t}
            lp_f = {i: t.get("logprob") for i, t in enumerate(tf)
                    if isinstance(t, dict) and "logprob" in t}
            shared = sorted(i for i in (set(lp_o) & set(lp_f))
                            if lp_o[i] is not None and lp_f[i] is not None)
            lpd = (sum(abs(lp_o[i] - lp_f[i]) for i in shared) / len(shared)
                   if shared else None)

            k5 = topk_sets(to, tf)

            all_comparable = compared == n and n > 0
            results.append({
                "prompt": name,
                "observed": {
                    "n_on": len(to), "n_off": len(tf),
                    "positions_compared": compared,
                    "positions_not_established": not_est,
                    "first_divergence_pos": first_div
                    if all_comparable else None,
                    "identical_positions": same,
                    "token_overlap_ratio": (round(same / compared, 4)
                                            if compared else None),
                    "topk5_overlap_mean": (round(sum(k5) / len(k5), 4)
                                           if k5 and None not in k5 else None),
                    "logprob_delta_mean": (round(lpd, 5)
                                           if lpd is not None else None),
                },
                "inferred": {
                    # the strongest claim a bounded sample supports:
                    "no_divergence_observed_in_tested_sequences":
                        (all_comparable and first_div is None),
                },
                "not_established": {
                    "identity": not (all_comparable and first_div is None),
                    "topk5": (k5 is None or None in k5),
                    "logprobs": lpd is None,
                    "positions": not_est,
                },
                "outputs": {"on": on.get("content", ""),
                            "off": off.get("content", "")},
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
                                           for r in results],
                      "no_divergence_observed": [
                          r["inferred"]["no_divergence_observed_in_tested_sequences"]
                          for r in results]}))
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
