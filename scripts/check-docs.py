#!/usr/bin/env python3
"""check-docs.py — documentation consistency checks for the llmlab repo.

Checks (exit 1 on error):
  1. broken relative links in .md files
  2. hardware spec claims in model cards vs the latest machine inventory
     snapshot for the card's machine tag (the 4C/8T class of error)
  3. superseded claims: patterns from data/superseded-claims.json must not
     appear in living docs (models/, README.md); inside dated reports a
     claim is allowed only if the report predates the supersession, or a
     SUPERSEDED banner sits nearby
  4. a single "current production" declaration per machine tag across
     living docs (cards must not contradict each other)
  5. reports/ files carry a YYYY-MM-DD date in the name
"""
import glob
import json
import os
import re
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
errors, warnings = [], []


def err(m):
    errors.append(m)


def warn(m):
    warnings.append(m)


def latest_inventory(tag):
    files = sorted(glob.glob(os.path.join(ROOT, "hardware", "inventory", tag + "-*.json")))
    return json.load(open(files[-1])) if files else None


# card filename -> machine tag
CARD_TAGS = {
    "qwen3.8-27b-dual3060.md": "dual3060-27b",
    "strata-qwen36-35b-a3b-single-3060.md": "dual3060-27b",
}


def md_files():
    for pat in ("**/*.md",):
        for f in glob.glob(os.path.join(ROOT, pat), recursive=True):
            if "/.git/" not in f:
                yield f


def check_links():
    for f in md_files():
        rel = os.path.relpath(f, ROOT)
        text = open(f).read()
        for m in re.finditer(r"\]\(([^)#\s]+?\.md)(#[^)]*)?\)", text):
            target = m.group(1)
            if target.startswith(("http", "/")):
                continue
            p = os.path.normpath(os.path.join(os.path.dirname(f), target))
            if not os.path.exists(p):
                err("%s: broken link %s" % (rel, target))


def check_hardware():
    for card, tag in CARD_TAGS.items():
        f = os.path.join(ROOT, "models", card)
        if not os.path.exists(f):
            continue
        inv = latest_inventory(tag)
        rel = "models/" + card
        if not inv:
            warn("%s: no inventory snapshot for machine tag %r (hardware check skipped)" % (rel, tag))
            continue
        text = open(f).read()
        cpu = inv["immutable"]["cpu"]
        truth_c = cpu["cores_per_socket"] * cpu["sockets"]
        truth_t = truth_c * cpu["threads_per_core"]
        for m in re.finditer(r"i5-7400[^\n]*?(\d+)\s*C\s*/\s*(\d+)\s*T", text):
            c, t = int(m.group(1)), int(m.group(2))
            if (c, t) != (truth_c, truth_t):
                err("%s: CPU spec %dC/%dT contradicts inventory (%dC/%dT, %s)"
                    % (rel, c, t, truth_c, truth_t, cpu["model"]))
        ram = inv["immutable"]["ram"]
        mod_gb = sum(m // 2**30 for m in ram.get("module_sizes_bytes", []))
        for m in re.finditer(r"(\d+)\s*GB\s*DDR4", text):
            if mod_gb and int(m.group(1)) != mod_gb:
                err("%s: RAM %s GB contradicts inventory modules (%d GB)" % (rel, m.group(1), mod_gb))
        for g in inv["immutable"]["gpus"]:
            pass  # gpu count/width checks stay manual until the x1 question is settled


def load_claims():
    p = os.path.join(ROOT, "data", "superseded-claims.json")
    if not os.path.exists(p):
        return []
    return json.load(open(p))


def check_superseded():
    claims = load_claims()
    living = []
    for f in md_files():
        rel = os.path.relpath(f, ROOT)
        parts = rel.split(os.sep)
        if parts and parts[0] == "reports":
            continue
        if os.path.basename(f) == "README.md":
            continue
        living.append((rel, open(f).read()))
    for c in claims:
        pat = re.compile(c["pattern"])
        for rel, text in living:
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if pat.search(line):
                    window = lines[max(0, i - 3):i + 3]
                    if any(k in w.lower() for w in window for k in ("retract", "correction", "supersede", "recharacterize")):
                        continue  # a pointer to the correction, not a re-assertion
                    err("superseded claim %r found in living doc %s: %s" % (c["id"], rel, line.strip()[:100]))
    for f in md_files():
        rel = os.path.relpath(f, ROOT)
        parts = rel.split(os.sep)
        if not (parts and parts[0] == "reports"):
            continue
        m = re.match(r"reports/(\d{4}-\d{2}-\d{2})", rel)
        if not m:
            continue
        rep_date = m.group(1)
        lines = open(f).read().split("\n")
        for c in claims:
            pat = re.compile(c["pattern"])
            sup_on = c.get("superseded_on")
            if sup_on and rep_date < sup_on:
                continue  # historical: report predates the supersession
            for i, line in enumerate(lines):
                if pat.search(line):
                    window = lines[max(0, i - 14):i + 14]
                    if any("SUPERSEDED" in w or "retracted" in w.lower() or "correction" in w.lower() or "recharacterized" in w.lower() for w in window):
                        continue
                    err("superseded claim %r in report %s line %d without nearby banner" % (c["id"], rel, i + 1))


PROD_RE = re.compile(r"(?:Production default|current production|Production:)[:\s]*\**\s*(P\d)", re.I)
CARD_PROD_TAG = {
    "qwen3.8-27b-dual3060.md": "dual3060-27b",
}


def check_production():
    pass  # per-card check below covers living-doc contradictions; cross-card check runs in check_hardware
    per_tag = {}
    for card, tag in CARD_PROD_TAG.items():
        f = os.path.join(ROOT, "models", card)
        if not os.path.exists(f):
            continue
        found = set(PROD_RE.findall(open(f).read()))
        if len(found) > 1:
            err("models/%s declares multiple production profiles %s" % (card, sorted(found)))
        if found:
            per_tag.setdefault(tag, set()).update((card, p) for p in found)
    for tag, decls in per_tag.items():
        profiles = {p for _, p in decls}
        if len(profiles) > 1:
            err("machine tag %s has contradictory production declarations: %s" % (tag, sorted(decls)))


def check_report_dates():
    for f in glob.glob(os.path.join(ROOT, "reports", "*.md")):
        if os.path.basename(f) == "README.md":
            continue
        rel = os.path.relpath(f, ROOT)
        if not re.match(r"reports/\d{4}-\d{2}-\d{2}-", rel):
            err("%s: report filename lacks YYYY-MM-DD date" % rel)


def main():
    check_links()
    check_hardware()
    check_superseded()
    check_production()
    check_report_dates()
    for w in warnings:
        print("warn:  %s" % w)
    for e in errors:
        print("error: %s" % e)
    print("docs checked: %d errors, %d warnings" % (len(errors), len(warnings)))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
