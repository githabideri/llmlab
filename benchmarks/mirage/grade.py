#!/usr/bin/env python3
"""Grade a run directory (raw.jsonl) against manifest/ under RULES.md.

Writes metrics.json + prints a summary table. Read-only over the run.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_tasks_file = "tasks.jsonl"

VOLATILE_KEYS = {"latency", "tokens", "wall_ms", "ms", "time"}


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    iw = max(0.0, x1 - x0); ih = max(0.0, y1 - y0)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def edit_distance(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def cer(pred, gt):
    if not gt:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, gt) / len(gt)


def salvage_strings(raw_prefix):
    """Best-effort: pull complete 'string' values out of truncated JSON."""
    if isinstance(raw_prefix, str):
        raw_prefix = raw_prefix.encode("utf-8", "replace")
    out = []
    for m in re.finditer(rb'"string"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_prefix or b""):
        out.append(m.group(1).decode("utf-8", "replace"))
    return out


def strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items()
                if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_volatile(x) for x in obj]
    return obj


def argmax(field):
    probs = field.get("probabilities") or {}
    val = field.get("value")
    if val is None and probs:
        val = max(probs, key=probs.get)
    return val, probs


def grade(recs, images):
    by_id = {r["task_id"]: r for r in recs}
    tasks = load_jsonl(os.path.join(HERE, "manifest", _tasks_file))
    imgs = {i["id"]: i for i in images}

    stats = {
        "presence": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "abstain": 0,
                     "fail": 0, "by_kind": {}},
        "choice": {"correct": 0, "wrong": 0, "abstain": 0, "fail": 0},
        "relation": {"correct": 0, "wrong": 0, "abstain": 0, "fail": 0},
        "count": {"correct": 0, "wrong": 0, "abstain": 0, "fail": 0,
                  "errors": []},
        "detect": {"n": 0, "box_tp": 0, "box_fn": 0, "box_fp": 0,
                   "name_tp": 0, "name_fn": 0, "truncated": 0,
                   "by_subkind": {}},
        "ocr": {"n": 0, "truncated": 0, "parsed": 0, "cers": [],
                "line_recall": [], "partial_cers": []},
        "text_presence": {"tp": 0, "fp_halluc": 0, "fn_miss": 0,
                          "abstain": 0, "fail": 0, "queries": 0},
        "ui_presence": {"tp": 0, "fp_halluc": 0, "fn_miss": 0, "fail": 0,
                        "candidates": 0},
        "describe": {"n": 0, "ok": 0, "truncated": 0},
        "determinism": {"groups": 0, "identical_groups": 0,
                        "stable_fields_groups": 0,
                        "per_image": {}},
        "perf": {"n": 0, "walls": [], "rss": [], "threads": set(),
                 "by_image": {}},
        "engine_truncation": {"total_calls": 0, "truncated": 0,
                              "by_image": {}},
    }

    # determinism: group perf-free 'determinism' tasks by image
    det = {}
    for r in recs:
        if r["type"] == "determinism":
            det.setdefault(r["image_id"], []).append(r)
    for iid, rs in det.items():
        rs.sort(key=lambda r: r["task_id"])
        raws = [json.dumps(r["parsed"], sort_keys=True) if r["parsed"] is not None
                else ("ERR:" + str(r["error"])) for r in rs]
        stripped = [json.dumps(strip_volatile(r["parsed"]), sort_keys=True)
                    if r["parsed"] is not None else ("ERR:" + str(r["error"]))
                    for r in rs]
        def objs_only(x):
            try:
                d = json.loads(x)
                if isinstance(d, dict) and "objects" in d:
                    return json.dumps(
                        [[o.get("name"), o.get("bbox")] for o in d["objects"]],
                        sort_keys=True)
            except Exception:
                pass
            return x
        objs = [objs_only(x) for x in stripped]
        stats["determinism"]["groups"] += 1
        stats["determinism"]["per_image"][iid] = {
            "full_identity": len(set(raws)) == 1,
            "stable_identity": len(set(stripped)) == 1,
            "objects_identity": len(set(objs)) == 1,
            "n": len(rs),
        }
        if len(set(raws)) == 1:
            stats["determinism"]["identical_groups"] += 1
        if len(set(stripped)) == 1:
            stats["determinism"]["stable_fields_groups"] += 1

    for t in tasks:
        r = by_id.get(t["task_id"])
        if r is None:
            continue
        img = imgs[t["image_id"]]
        if img.get("kind") == "commons":
            kind = img.get("slot", "commons").split("-")[0]
        else:
            kind = img.get("kind", "synthetic")
        sub = img.get("subkind", kind)
        parsed = r["parsed"]
        truncated = parsed is None
        stats["engine_truncation"]["total_calls"] += 1
        if truncated:
            stats["engine_truncation"]["truncated"] += 1
            stats["engine_truncation"]["by_image"][img["id"]] = \
                stats["engine_truncation"]["by_image"].get(img["id"], 0) + 1

        ty = t["type"]

        if ty in ("describe", "ocr", "text_presence", "ui_presence",
                  "detect", "perf"):
            stats["describe"]["n"] += 1
            if not truncated:
                stats["describe"]["ok"] += 1
            else:
                stats["describe"]["truncated"] += 1

        if ty == "presence":
            s = stats["presence"]
            s["by_kind"].setdefault(kind, {"tp": 0, "fp": 0, "fn": 0,
                                           "tn": 0, "abstain": 0,
                                           "fail": 0})
            k = s["by_kind"][kind]
            if truncated:
                s["fail"] += 1; k["fail"] += 1
                continue
            val, probs = argmax(parsed["form"][t["query"]])
            gt = t["gt"]["answer"]
            if val == "-":
                s["abstain"] += 1; k["abstain"] += 1
            elif val == gt:
                key = "tp" if gt == "yes" else "tn"
                s[key] += 1; k[key] += 1
            else:
                key = "fp" if val == "yes" else "fn"
                s[key] += 1; k[key] += 1

        elif ty in ("choice", "relation"):
            s = stats[ty]
            if truncated:
                s["fail"] += 1
                continue
            val, _ = argmax(parsed["form"][t["gt"]["question"]])
            if val == "-":
                s["abstain"] += 1
            elif val == t["gt"]["answer"]:
                s["correct"] += 1
            else:
                s["wrong"] += 1

        elif ty == "count":
            s = stats["count"]
            if truncated:
                s["fail"] += 1
                continue
            field = next(iter(parsed["form"].values()))
            val = field.get("value")
            if val == "-":
                s["abstain"] += 1
            elif val == t["gt"]["answer"]:
                s["correct"] += 1
            else:
                s["wrong"] += 1
                s["errors"].append({"image": img["id"],
                                    "pred": val, "gt": t["gt"]["answer"],
                                    "probs": field.get("probabilities")})

        elif ty == "detect":
            s = stats["detect"]
            s["n"] += 1
            if truncated:
                s["truncated"] += 1
                continue
            w, h = img["size"]
            gt_objs = t["gt"]["objects"]
            gboxes = [[b[0] * 1000 / w, b[1] * 1000 / h,
                       b[2] * 1000 / w, b[3] * 1000 / h]
                      for b in [o["bbox"] for o in gt_objs]]
            gt_names = [o["name"] for o in gt_objs]
            pred = sorted(parsed.get("objects") or [],
                          key=lambda o: -o.get("score", 0))
            matched = [False] * len(gt_objs)
            for p in pred:
                pb = p.get("bbox")
                if not pb:
                    continue
                best, bi = 0.0, -1
                for gi, gb in enumerate(gboxes):
                    if matched[gi]:
                        continue
                    v = iou(pb, gb)
                    if v > best:
                        best, bi = v, gi
                if bi >= 0 and best >= 0.5:
                    matched[bi] = True
                    s["box_tp"] += 1
                    name_ok = False
                    gname = gt_names[bi].lower()
                    pname = (p.get("name") or "").lower()
                    desc = p.get("description") or {}
                    dstr = json.dumps(desc).lower()
                    for word in gname.split():
                        if word not in pname and word not in dstr:
                            name_ok = False
                            break
                    else:
                        name_ok = True
                    if name_ok:
                        s["name_tp"] += 1
                    else:
                        s["name_fn"] += 1
                else:
                    s["box_fp"] += 1
            for gi, m in enumerate(matched):
                if not m:
                    s["box_fn"] += 1
                    s["name_fn"] += 1
            sk = s["by_subkind"].setdefault(sub,
                                            {"n": 0, "box_tp": 0, "box_fn": 0,
                                             "box_fp": 0, "truncated": 0})
            sk["n"] += 1
            if truncated:
                sk["truncated"] += 1

        elif ty == "ocr":
            s = stats["ocr"]
            s["n"] += 1
            gt_text = " ".join(t["gt"]["text"])
            gt_n = norm(gt_text)
            if truncated:
                s["truncated"] += 1
                parts = salvage_strings(r.get("raw_full") or r.get("raw_prefix"))
                if parts:
                    s["partial_cers"].append(cer(norm(" ".join(parts)), gt_n))
                continue
            s["parsed"] += 1
            pred = " ".join(x.get("string", "") for x in
                            (parsed.get("text") or []))
            s["cers"].append(cer(norm(pred), gt_n))
            # line recall
            gt_lines = [norm(x) for x in t["gt"]["text"] if norm(x)]
            pred_lines = [norm(x.get("string", "")) for x in
                          (parsed.get("text") or []) if norm(x.get("string", ""))]
            hits = 0
            used = [False] * len(pred_lines)
            for gl in gt_lines:
                for pj, pl in enumerate(pred_lines):
                    if used[pj]:
                        continue
                    d = edit_distance(gl, pl)
                    if d <= 0.5 * max(1, len(gl)):
                        hits += 1
                        used[pj] = True
                        break
            s["line_recall"].append(hits / len(gt_lines) if gt_lines else 1.0)

        elif ty == "text_presence":
            s = stats["text_presence"]
            s["queries"] += 2
            if truncated:
                s["fail"] += 2
                continue
            text = " ".join(x.get("string", "") for x in
                            (parsed.get("text") or []))
            onames = " ".join((o.get("name") or "") + " " +
                              json.dumps(o.get("description") or {})
                              for o in (parsed.get("objects") or []))
            hay = norm(text + " " + onames)
            if norm(t["true_query"]) in hay:
                s["tp"] += 1
            else:
                s["fn_miss"] += 1
            if norm(t["false_query"]) in hay:
                s["fp_halluc"] += 1

        elif ty == "ui_presence":
            s = stats["ui_presence"]
            if truncated:
                s["fail"] += 2
                continue
            onames = " ".join((o.get("name") or "") + " " +
                              json.dumps(o.get("description") or {})
                              for o in (parsed.get("objects") or [])).lower()
            for c in t["candidates"]:
                s["candidates"] += 1
                seen = c.lower() in onames
                want = c in t["gt"]["present"]
                if seen and want:
                    s["tp"] += 1
                elif seen and not want:
                    s["fp_halluc"] += 1
                elif not seen and want:
                    s["fn_miss"] += 1

        elif ty == "perf":
            s = stats["perf"]
            s["n"] += 1
            s["walls"].append(r["wall_ms"])
            if r.get("rss_kb"):
                s["rss"].append(r["rss_kb"])
            if r.get("threads"):
                s["threads"].add(r["threads"])
            s["by_image"].setdefault(img["id"], []).append(r["wall_ms"])

    # finalize
    p = stats["presence"]
    tot = p["tp"] + p["fp"] + p["fn"] + p["tn"]
    stats["presence"]["accuracy"] = (p["tp"] + p["tn"]) / tot if tot else None
    for name in ("choice", "relation"):
        s = stats[name]
        n = s["correct"] + s["wrong"]
        s["accuracy"] = s["correct"] / n if n else None
    s = stats["count"]
    n = s["correct"] + s["wrong"]
    s["accuracy"] = s["correct"] / n if n else None
    d = stats["detect"]
    prec = d["box_tp"] / (d["box_tp"] + d["box_fp"]) if d["box_tp"] + d["box_fp"] else None
    rec = d["box_tp"] / (d["box_tp"] + d["box_fn"]) if d["box_tp"] + d["box_fn"] else None
    d["precision"] = prec
    d["recall"] = rec
    d["f1"] = (2 * prec * rec / (prec + rec)
               if prec is not None and rec is not None
               and prec + rec > 0 else None)
    o = stats["ocr"]
    o["mean_cer"] = sum(o["cers"]) / len(o["cers"]) if o["cers"] else None
    o["mean_line_recall"] = (sum(o["line_recall"]) / len(o["line_recall"])
                             if o["line_recall"] else None)
    o["mean_partial_cer"] = (sum(o["partial_cers"]) / len(o["partial_cers"])
                             if o["partial_cers"] else None)
    tp_, fp_ = stats["text_presence"]["tp"], stats["text_presence"]["fp_halluc"]
    stats["text_presence"]["true_recall"] = (
        tp_ / (tp_ + stats["text_presence"]["fn_miss"])
        if tp_ + stats["text_presence"]["fn_miss"] else None)
    u = stats["ui_presence"]
    stats["ui_presence"]["precision"] = (
        u["tp"] / (u["tp"] + u["fp_halluc"])
        if u["tp"] + u["fp_halluc"] else None)
    perf = stats["perf"]
    perf["walls_ms_sorted"] = sorted(perf["walls"])
    n = len(perf["walls_ms_sorted"])
    if n:
        perf["p50_ms"] = perf["walls_ms_sorted"][n // 2]
        perf["p95_ms"] = perf["walls_ms_sorted"][min(n - 1, int(n * 0.95))]
        perf["min_ms"] = perf["walls_ms_sorted"][0]
        perf["max_ms"] = perf["walls_ms_sorted"][-1]
    perf["rss_kb_min"] = min(perf["rss"]) if perf["rss"] else None
    perf["rss_kb_max"] = max(perf["rss"]) if perf["rss"] else None
    perf["threads"] = sorted(perf["threads"])
    e = stats["engine_truncation"]
    e["rate"] = e["truncated"] / e["total_calls"] if e["total_calls"] else None

    return stats


def main():
    if len(sys.argv) < 2:
        runs = sorted(d for d in os.listdir(os.path.join(HERE, "results"))
                      if os.path.isdir(os.path.join(HERE, "results", d)))
        if not runs:
            print("no run found", file=sys.stderr)
            return 1
        run = runs[-1]
        print(f"latest run: {run}")
    else:
        run = sys.argv[1]
    run_dir = os.path.join(HERE, "results", run)
    tasks_file = "tasks.jsonl"
    if "--tasks" in sys.argv:
        tasks_file = sys.argv[sys.argv.index("--tasks") + 1]
    recs = load_jsonl(os.path.join(run_dir, "raw.jsonl"))
    images = load_jsonl(os.path.join(HERE, "manifest", "images.jsonl"))
    global _tasks_file
    _tasks_file = tasks_file
    stats = grade(recs, images)
    stats["run"] = run
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    stats["meta"] = meta
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
