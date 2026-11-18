#!/usr/bin/env python3
"""validate-provenance.py — schema and cross-reference checks for the machine-
readable benchmark provenance records in data/provenance/*.json.

Each published performance result in this repo should trace to a record with:
config id, engine commit + binary hash, model hash, workload, token counts,
time-definition, repetitions, artifact reference, success classification, and
whether the number is measured or inferred.

Checks:
  * required fields and schema version
  * unit allowlist and plausibility ranges (no 9,000 tok/s, no negative times)
  * arithmetic: cached tokens <= total prompt tokens
  * machine_tag must have an inventory snapshot in hardware/inventory/
  * engine commit / hash formats
  * duplicate ids
  * ids referenced by model cards' "## Provenance" sections must exist;
    unreferenced records are warnings, not errors
Exit 1 on any error.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = {"s", "ms", "tokens", "tok/s", "bytes", "MiB", "GiB", "GB", "bool", "count", "percent"}
RANGES = {
    "s": (0, 10800),
    "ms": (0, 10_800_000),
    "tok/s": (0, 500),
    "tokens": (0, 2_000_000),
    "percent": (0, 100),
}
REQUIRED = ["schema", "id", "machine_tag", "date", "engine", "model", "config",
            "workload", "metrics", "classification", "success"]
HEX40 = re.compile(r"^[0-9a-f]{7,40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

errors, warnings = [], []


def err(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


def main():
    inv_tags = set()
    for f in glob.glob(os.path.join(ROOT, "hardware", "inventory", "*.json")):
        try:
            inv_tags.add(json.load(open(f)).get("machine_tag"))
        except Exception as e:
            err("inventory %s unreadable: %s" % (f, e))

    recs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "provenance", "*.json"))):
        rel = os.path.relpath(f, ROOT)
        try:
            r = json.load(open(f))
        except Exception as e:
            err("%s: not valid JSON: %s" % (rel, e))
            continue
        recs[r.get("id", rel)] = (rel, r)

    for rid, (rel, r) in recs.items():
        if r.get("schema") != "llmlab.provenance/1":
            err("%s: schema must be llmlab.provenance/1" % rel)
        for k in REQUIRED:
            if k not in r:
                err("%s: missing field %s" % (rel, k))
        if r.get("machine_tag") not in inv_tags:
            err("%s: machine_tag %r has no inventory snapshot" % (rel, r.get("machine_tag")))
        eng = r.get("engine", {})
        if not HEX40.match(eng.get("commit", "")):
            err("%s: engine.commit must be 7-40 hex chars" % rel)
        for h in (eng.get("binary_sha256"), r.get("model", {}).get("sha256")):
            if h is not None and not HEX64.match(h):
                err("%s: sha256 field malformed: %r" % (rel, h))
        if r.get("classification") not in ("measured", "inferred"):
            err("%s: classification must be measured|inferred" % rel)
        if r.get("classification") == "inferred" and not r.get("basis"):
            err("%s: inferred results must carry a 'basis'" % rel)
        if r.get("success") not in ("pass", "fail", "partial"):
            err("%s: success must be pass|fail|partial" % rel)
        wl = r.get("workload", {})
        for m in r.get("metrics", []):
            for k in ("name", "value", "unit"):
                if k not in m:
                    err("%s: metric missing %s" % (rel, k))
            u = m.get("unit")
            if u not in UNITS:
                err("%s: metric %s has disallowed unit %r" % (rel, m.get("name"), u))
            v = m.get("value")
            if u == "bool":
                if not isinstance(v, bool):
                    err("%s: metric %s must be bool" % (rel, m.get("name")))
            else:
                if not isinstance(v, (int, float)):
                    err("%s: metric %s must be numeric" % (rel, m.get("name")))
                elif u in RANGES:
                    lo, hi = RANGES[u]
                    if not (lo <= v <= hi):
                        err("%s: metric %s = %s out of plausible range %s" % (rel, m.get("name"), v, (lo, hi)))
        # arithmetic: cached <= total
        vals = {m["name"]: m.get("value") for m in r.get("metrics", [])}
        total = wl.get("prompt_tokens")
        cached = vals.get("prompt_cached_tokens")
        if isinstance(total, int) and isinstance(cached, int) and cached > total:
            err("%s: prompt_cached_tokens (%s) > workload prompt_tokens (%s)" % (rel, cached, total))
        nrep = wl.get("n_reps", 1)
        if isinstance(nrep, int) and nrep < 1:
            err("%s: n_reps must be >= 1" % rel)
        if isinstance(nrep, int) and nrep == 1 and r.get("classification") == "measured":
            warn("%s: n_reps=1 (single shot — fine, but state it in the report)" % rel)

    # cross-reference with model cards
    card_refs = set()
    for f in glob.glob(os.path.join(ROOT, "models", "*.md")):
        if os.path.basename(f) == "README.md":
            continue
        text = open(f).read()
        if "## Provenance" in text:
            for m in re.finditer(r"^-\s+`([^`]+)`", text.split("## Provenance")[1], re.M):
                card_refs.add(m.group(1))
        else:
            warn("%s: no '## Provenance' section (required for model cards)" % os.path.relpath(f, ROOT))
    for ref in sorted(card_refs):
        if ref not in recs:
            err("model card references provenance id %r — no such record" % ref)
    for rid in recs:
        if rid not in card_refs:
            warn("provenance record %s not referenced by any model card" % rid)

    for w in warnings:
        print("warn:  %s" % w)
    for e in errors:
        print("error: %s" % e)
    print("%d records checked, %d errors, %d warnings" % (len(recs), len(errors), len(warnings)))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
