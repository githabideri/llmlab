#!/usr/bin/env python3
"""Merge the three sources (synthetic manifest, commons manifest, own images)
into the frozen benchmark manifests:

  manifest/images.jsonl  one record per image: id, kind, path, sha256, size,
                         license/attribution (commons only), gt (from the
                         generator or the human review file)
  manifest/tasks.jsonl   one record per model call: task id, image id, type,
                         params, and the ground-truth payload for grading

Deterministic: same inputs -> byte-identical outputs. Re-run after any
corpus change BEFORE scoring; never change these files after a run started.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "manifest")
os.makedirs(MAN, exist_ok=True)

OWN_REL = {
    "own-nvtop-glm": "reports/assets/2026-03-03-glm-nvtop-tgen-pattern.png",
    "own-nvtop-nemotron": "reports/assets/2026-03-03-nemotron-nvtop-tg-pattern.png",
    "own-nvtop-qwen35": "reports/assets/2026-03-03-qwen35-nvtop-pattern.png",
    "own-nvtop-vllm-pcie": "reports/assets/vllm-pp3-nvtop-pcie-2026-03-17.png",
}
# repo-relative; the llmlab checkout root is two levels up
LAB_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OWN = {k: os.path.join(LAB_ROOT, v) for k, v in OWN_REL.items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build():
    images, tasks = [], []
    tseq = [0]

    def add_task(image_id, type_, gt, **params):
        tseq[0] += 1
        tasks.append({"task_id": f"t{tseq[0]:04d}", "image_id": image_id,
                      "type": type_, "gt": gt, **params})

    def presence_q(obj):
        return f"Is there a {obj} in this image?"

    def review_for(slot):
        return REVIEW.get(slot, {})

    # ---------------- synthetic ----------------
    for m in load_jsonl(os.path.join(HERE, "images", "synthetic-manifest.jsonl")):
        iid = m["id"]
        p = m["path"] if os.path.isabs(m["path"]) else os.path.join(HERE, m["path"])
        rel = os.path.relpath(p, HERE)
        images.append({"id": iid, "kind": "synthetic", "subkind": m["subkind"],
                       "path": rel, "sha256": sha256(p),
                       "size": m["size"], "gt": m["gt"]})
        sk = m["subkind"]
        gt = m["gt"]
        if sk.startswith("ui-"):
            pool = ["button", "toggle", "menu item", "minimap", "crosshair",
                    "bar", "app icon", "content pane"]
            names = [e.get("name") for e in gt["elements"]]
            present = [p for p in pool if p in names]
            absent = [p for p in pool if p not in names][:2]
            cands = (present[:2] + absent)[:4]
            add_task(iid, "ui_presence",
                     {"present": present[:2], "absent": absent},
                     candidates=cands)
        if sk in ("doc",) or sk.startswith("ui-"):
            if sk == "doc":
                add_task(iid, "ocr", {"text": gt["text"]})
            text = gt.get("text") or []
            if text:
                tline = text[int(hashlib.md5(iid.encode()).hexdigest(), 16) % len(text)]
                fline = " ".join(tline.split()[:-1] + ["zebra"])
                add_task(iid, "text_presence",
                         {"true_query": tline, "false_query": fline},
                         true_query=tline, false_query=fline)
        if sk.startswith("shapes-"):
            if sk == "shapes-pairs":
                add_task(iid, "detect", {"objects": gt["objects"]})
                add_task(iid, "relation",
                         {"answer": gt["relation_a_to_b"],
                          "question": f"Object A is a {gt['objects'][0]['name']} and object B is a {gt['objects'][1]['name']}. Where is A relative to B?"},
                         options=["left of", "right of", "above", "below"])
            elif sk == "shapes-counts":
                add_task(iid, "count", {"answer": len(gt["objects"])},
                         lo=1, hi=6)
            elif sk in ("shapes-simple", "shapes-crowded"):
                add_task(iid, "detect", {"objects": gt["objects"]})
                add_task(iid, "count", {"answer": len(gt["objects"])},
                         lo=1, hi=12)
            elif sk == "shapes-small":
                add_task(iid, "detect", {"objects": gt["objects"]})
                add_task(iid, "presence", {"answer": "yes"},
                         query=presence_q(gt["objects"][0]["name"]))
            elif sk == "shapes-faint":
                add_task(iid, "presence", {"answer": "yes"},
                         query=presence_q(gt["objects"][0]["name"]))
            elif sk == "shapes-blank":
                add_task(iid, "presence", {"answer": "no"},
                         query=presence_q("red square"))
        if sk in ("noise", "gradient"):
            add_task(iid, "presence", {"answer": "no"},
                     query=presence_q("red square"))

    # ---------------- commons (human-reviewed) ----------------
    for m in load_jsonl(os.path.join(HERE, "common-manifest.jsonl")):
        slot = m["slot"]
        rv = review_for(slot)
        iid = f"com-{slot}"
        p = m["path"] if os.path.isabs(m["path"]) else os.path.join(HERE, m["path"])
        rel = os.path.relpath(p, HERE)
        images.append({"id": iid, "kind": "commons", "slot": slot,
                       "search": m.get("search"), "path": rel,
                       "sha256": m.get("sha256"),
                       "license": m.get("license"),
                       "attribution": m.get("attribution"),
                       "url": m.get("url"), "fetched_url": m.get("fetched_url"),
                       "reviewed": rv.get("verified", False),
                       "review_notes": rv.get("notes"),
                       "gt": rv})
        if slot.startswith("everyday"):
            subj = rv["subject"]
            add_task(iid, "presence", {"answer": "yes"},
                     query=presence_q(subj))
            add_task(iid, "choice", {"answer": subj,
                                    "question": f"What is the main subject of this image?"},
                     options=[subj] + rv["distractors"])
            if "gt_text" in rv:
                t = rv["gt_text"][0]
                add_task(iid, "text_presence",
                         {"true_query": t, "false_query": rv["fake_text"][0]},
                         true_query=t, false_query=rv["fake_text"][0])
        elif slot.startswith("crowded"):
            for obj in rv["present"]:
                add_task(iid, "presence", {"answer": "yes"}, query=presence_q(obj))
            for obj in rv["absent"]:
                add_task(iid, "presence", {"answer": "no"}, query=presence_q(obj))
        elif slot.startswith("negative"):
            for obj in rv["absent"]:
                add_task(iid, "presence", {"answer": "no"}, query=presence_q(obj))

    # ---------------- own (describe only) ----------------
    for slot, path in OWN.items():
        rv = review_for(slot)
        images.append({"id": f"own-{slot}", "kind": "own", "slot": slot,
                       "path": OWN_REL[slot], "sha256": sha256(path),
                       "reviewed": rv.get("verified", False)})

    # ---------------- describe_full for everything ----------------
    for img in images:
        add_task(img["id"], "describe", {"note": "raw full response, no GT"})

    # ---------------- determinism block (5 everyday images x 5 repeats) ----------------
    det = [f"com-everyday-{i:02d}" for i in range(5)]
    for iid in det:
        for _ in range(5):
            add_task(iid, "determinism", {"note": "raw full response, hash-compared"})

    # ---------------- performance subset (20 images x 5 reps) ----------------
    perf = [
        "com-everyday-00", "com-everyday-01", "com-everyday-02",
        "com-everyday-03", "com-everyday-05", "com-everyday-15",
        "com-crowded-00", "com-crowded-05", "com-crowded-07",
        "com-crowded-09", "com-negative-00", "com-negative-02",
        "syn-shapes-simple-1", "syn-shapes-crowded-11",
        "syn-doc-101", "syn-doc-112", "syn-ui-window-201",
        "syn-ui-hud-207", "syn-shapes-small-21", "syn-shapes-blank-61",
    ]
    for iid in perf:
        for _ in range(5):
            add_task(iid, "perf", {"note": "timed describe; rss/hwm/threads recorded"})

    with open(os.path.join(MAN, "images.jsonl"), "w") as fh:
        for i in images:
            fh.write(json.dumps(i) + "\n")
    with open(os.path.join(MAN, "tasks.jsonl"), "w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    print(f"images: {len(images)}  tasks: {len(tasks)}")


REVIEW = {}

def main():
    global REVIEW
    with open(os.path.join(HERE, "review-commons.jsonl")) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                REVIEW[json.loads(line)["slot"]] = json.loads(line)
    build()

if __name__ == "__main__":
    main()
