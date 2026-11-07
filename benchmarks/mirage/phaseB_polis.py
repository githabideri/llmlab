#!/usr/bin/env python3
"""Phase B: Mirage on the six Polis harvest screenshots (offline, after Phase A).

Each frame gets the question battery from docs/design/vision-jev.md adapted to
what the frames actually contain (see the session findings on the rye/bush
distinction). Everything is recorded raw; nothing is interpreted here.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# The Polis checkout's harvest proofs dir (sibling submodule by default;
# override with POLIS_PROOFS if the checkout lives elsewhere).
PROOFS = os.environ.get(
    "POLIS_PROOFS",
    os.path.join(HERE, "..", "..", "..", "polis", "docs", "proofs",
                 "2026-09-22-harvest"),
)
OUT = os.path.join(HERE, "results", "phaseB-polis")
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("CACTUS_TELEMETRY", "0")

import cactus
from cactus import Mirage
from cactus.agent import mirage as M

m = Mirage()
lib = m._bind()

def raw_call(path, settings):
    rgb, w, h = M._image(path)
    t0 = time.perf_counter()
    code = lib.cactus_describe(rgb, w, h, json.dumps(settings).encode(),
                               None, None, m._buffer, len(m._buffer))
    wall = (time.perf_counter() - t0) * 1000.0
    n = m._buffer.raw.find(b"\x00")
    raw = bytes(m._buffer.raw[:n])
    parsed = None
    err = None
    try:
        parsed = json.loads(raw)
    except Exception as e:
        err = f"{e.__class__.__name__}: {e}"
    return code, raw, parsed, err, wall

FRAMES = [
    "viz8-01-before-story.png",
    "viz8-02-before-cropclose.png",
    "viz8-04-at-crop.png",
    "viz8-05-after-crop-gone.png",
    "viz8-06-return-mid.png",
    "viz8-07-at-base-carrying.png",
]

# The question battery. Field names are short; the description carries the
# actual question the model is asked to answer about the image.
BATTERY = {
    "describe": {"sections": ["objects", "text"]},
    "bush_visible": {"form": {"bush_visible": {
        "enum": ["yes", "no"],
        "description": "Is there a green leafy bush with small orange tips "
                       "inside or next to the stone-walled farmland plot?"}}},
    "rye_visible": {"form": {"rye_visible": {
        "enum": ["yes", "no"],
        "description": "Is there a cluster of yellow rye plants inside the "
                       "stone-walled farmland plot?"}}},
    "facing": {"form": {"facing": {
        "enum": ["stone_wall", "crop_or_plant", "farmland", "empty_ground",
                 "sky", "other"],
        "description": "What is directly in front of the camera at the "
                       "crosshair?"}}},
}

def main():
    recs = []
    # warmup (not recorded)
    raw_call(os.path.join(PROOFS, FRAMES[0]), {"sections": ["objects"]})
    for fname in FRAMES:
        path = os.path.join(PROOFS, fname)
        for qname, settings in BATTERY.items():
            code, raw, parsed, err, wall = raw_call(path, settings)
            rec = {
                "frame": fname,
                "question": qname,
                "code": code,
                "parsed": parsed,
                "error": err,
                "wall_ms": round(wall, 1),
            }
            recs.append(rec)
            with open(os.path.join(OUT, "phaseB-raw.jsonl"), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{fname} {qname}: code={code} parsed={parsed is not None} "
                  f"err={err} {wall:.0f}ms", flush=True)
    # changed-frame probe: 04 and 05 side by side
    from PIL import Image
    a = Image.open(os.path.join(PROOFS, FRAMES[2])).convert("RGB")
    b = Image.open(os.path.join(PROOFS, FRAMES[3])).convert("RGB")
    pair = Image.new("RGB", (a.width, a.height * 2 + 4))
    pair.paste(a, (0, 0))
    pair.paste(b, (0, a.height + 4))
    pair_path = os.path.join(OUT, "pair-04-05.png")
    pair.save(pair_path)
    settings = {"form": {"changed": {
        "enum": ["yes", "no"],
        "description": "Are the two images different from each other?"}}}
    code, raw, parsed, err, wall = raw_call(pair_path, settings)
    rec = {"frame": "pair-04-05", "question": "changed", "code": code,
           "parsed": parsed, "error": err, "wall_ms": round(wall, 1)}
    with open(os.path.join(OUT, "phaseB-raw.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"pair-04-05 changed: code={code} parsed={parsed} err={err}",
          flush=True)
    print("done ->", OUT)

if __name__ == "__main__":
    main()
