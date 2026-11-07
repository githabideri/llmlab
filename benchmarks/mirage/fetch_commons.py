#!/usr/bin/env python3
"""Fetch real-world photos from Wikimedia Commons for the Mirage corpus.

License policy: permissive licenses only (CC0, CC-BY, CC-BY-SA). Images are
downloaded at run time and referenced by URL + SHA-256; they are NOT committed
to the public repo (reference-only, to stay clear of redistribution
questions). Attribution is recorded per image.
"""
import json
import os
import sys
import time
import hashlib
import urllib.parse
import urllib.request
import urllib.error

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia UA policy: application name/version + purpose + contact (public URL)
UA = ("MirageEvalCorpus/1.0 (research image corpus, llmlab benchmark; "
      "https://github.com/githabideri/llmlab)")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common")
os.makedirs(OUT, exist_ok=True)

ALLOWED_LICENSES = ("CC0-1.0", "CC-BY-1.0", "CC-BY-2.0", "CC-BY-2.5", "CC-BY-3.0",
                    "CC-BY-4.0", "CC-BY-SA-3.0", "CC-BY-SA-4.0",
                    "no known copyright restrictions", "Public Domain Mark 1.0",
                    "CC-PDDC", "CC-BY-2.5")

EVERYDAY = [
    "cat", "dog", "horse", "car", "bicycle", "chair", "table", "bottle",
    "coffee cup", "laptop computer", "book", "apple fruit", "banana", "bird",
    "sailing boat", "bridge", "wall clock", "umbrella", "guitar", "bus",
    "train", "airplane", "television set", "mobile phone", "flower",
]
CROWDED = [
    "busy street market", "crowded market street", "busy intersection people",
    "market stall vegetables", "crowded square", "busy city street people",
    "street fair crowd", "fish market people", "busy harbour", "crowded train station",
    "street festival crowd", "night market people",
]
NATURAL_NEGATIVE = [
    "clear blue sky", "calm lake water", "empty desert dunes",
    "empty green field", "snowfield plain", "empty beach sand",
]


def api(params, retries=4):
    params = dict(params, format="json")
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  429 from Commons API, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def find_image(search, min_w=300, max_w=4000, min_h=200, max_h=3000, tries=20):
    q = f"{search} filetype:bitmap"  # api() encodes once
    d = api({
        "action": "query",
        "generator": "search",
        "gsrsearch": q,
        "gsrnamespace": 6,
        "gsrlimit": tries,
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": 1600,
    })
    if d.get("error"):
        print(f"  api error: {d['error']}", file=sys.stderr)
    pages = (d.get("query") or {}).get("pages") or {}
    cands = []
    rejected = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [None])[0]
        if not ii:
            continue
        w, h = ii.get("width", 0), ii.get("height", 0)
        em = ii.get("extmetadata") or {}
        raw_lic = (em.get("LicenseShortName") or {}).get("value") or \
                  (em.get("LicenseShortName") or {}).get("val") or ""
        lic = raw_lic.replace(" ", "-")  # "CC BY-SA 4.0" -> "CC-BY-SA-4.0"
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            rejected.append((p.get("title"), f"size {w}x{h}"))
            continue
        if lic not in ALLOWED_LICENSES:
            rejected.append((p.get("title"), f"license {lic!r}"))
            continue
        url = ii.get("url")
        if not url:
            continue
        cands.append({
            "title": p.get("title"),
            "url": url,
            "thumburl": ii.get("thumburl"),
            "license": raw_lic or lic,
            "attribution": ((em.get("Artist") or {}).get("val") or "")[:120],
            "width": w, "height": h,
            "description": p.get("description", ""),
        })
    if rejected and not cands:
        print(f"  all rejected: {rejected[:4]}", file=sys.stderr)
    return cands


def fetch_one(slot, search):
    os.makedirs(os.path.join(OUT, slot), exist_ok=True)
    cands = find_image(search)
    if not cands:
        return None
    c = cands[0]
    safe = search.replace(" ", "_")
    ext = ".jpg" if (c.get("thumburl") or c["url"]).lower().endswith((".jpg", ".jpeg")) else ".png"
    path = os.path.join(OUT, slot, f"{slot}-{safe}{ext}")
    fetch_url = c.get("thumburl") or c["url"]
    req = urllib.request.Request(fetch_url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except Exception as e:
        print(f"  download failed: {e}", file=sys.stderr)
        return None
    if len(data) > 4 * 1024 * 1024:
        print(f"  skip >4MB: {c['title']}", file=sys.stderr)
        return None
    with open(path, "wb") as fh:
        fh.write(data)
    rec = dict(c, slot=slot, search=search,
               path=os.path.relpath(path, os.path.dirname(OUT)),
               fetch_url=fetch_url, fetched_url=c["url"],
               sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return rec


def main():
    plan = [(f"everyday-{i:02d}", s) for i, s in enumerate(EVERYDAY)] + \
           [(f"crowded-{i:02d}", s) for i, s in enumerate(CROWDED)] + \
           [(f"negative-{i:02d}", s) for i, s in enumerate(NATURAL_NEGATIVE)]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "common-manifest.jsonl")
    n = 0
    with open(out, "w") as fh:
        for slot, search in plan:
            rec = fetch_one(slot, search)
            if rec is None:
                print(f"!! no image for {slot} ({search})", file=sys.stderr)
            else:
                fh.write(json.dumps(rec) + "\n")
                n += 1
                print(f"ok {slot}: {rec['title']} ({rec['license']}, "
                      f"{rec['width']}x{rec['height']}, {rec['bytes']} B)")
            time.sleep(2.0)
    print(f"total {n}/{len(plan)}")


if __name__ == "__main__":
    main()
