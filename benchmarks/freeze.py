"""freeze.py — freeze.json: the immutable identity of a prepared campaign.

Baked at prepare time, verified at window start (T0) on every target. Any
mismatch aborts BEFORE the first destructive step.

Two hashes, deliberately separate (the repair-lane contract):

  scientific_hash     sha256 over the spec's science fields (spec.py) — what is
                      measured. The runner recomputes this after any repair and
                      refuses to continue if it moved.
  implementation_hash the llmlab git commit the bundle was built from — the
                      code. A bounded repair produces a NEW implementation hash
                      (a new commit / patch record) and is requalified.

Plus: spec file sha, bundle tarball sha, model/build artifact shas, profile id,
the qualification result, and the prepare timestamp. FROZEN.json (mm-image-max)
is the ancestor of this file; the per-file hash set below is the same
non-circular trick (freeze.json excludes itself).
"""
import hashlib
import json
import os
import time


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def bake(campaign_id, spec_path, spec_obj, llmlab_dir, profile_id,
         qualification, bundle_sha=None, model_sha=None, build_sha=None,
         campaigns_payload=None):
    """Write freeze.json into the campaign's bundle dir. Returns the dict.

    campaigns_payload: optional {relpath: sha256} of the shipped campaign
    payload (the spec copy + plan files). It is (a) a science input — the
    plans are part of the scientific projection — and (b) verified on the
    target by deploy.push (keys namespaced "campaigns/…").
    """
    from . import spec as specmod
    frozen = {
        "schema": 3,
        "campaign_id": campaign_id,
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "llmlab_dir": llmlab_dir,
        "llmlab_commit": _git_head(llmlab_dir),
        "scientific_hash": specmod.scientific_hash(spec_obj, campaigns_payload),
        "implementation_hash": _git_head(llmlab_dir),   # the commit IS the impl identity
        "spec_path": os.path.basename(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "bundle_sha256": bundle_sha,                    # set by deploy if a tarball is built
        "profile_id": profile_id,
        "model_sha256": model_sha,
        "build_sha256": build_sha,
        "qualification": qualification,                 # {"p0": "10/10 PASS", "p1": "4/4", ...}
    }
    # P0 (09-14): the per-file hash set is part of the freeze. Every earlier
    # caller blanked it, which made every downstream verifier vacuous
    # (deploy push compared against nothing; the bundle never got checked).
    frozen["file_hashes"] = tree_hashes(llmlab_dir)
    if campaigns_payload:
        frozen["campaigns_payload"] = dict(sorted(campaigns_payload.items()))
        for k, v in sorted(campaigns_payload.items()):
            frozen["file_hashes"]["campaigns/" + k] = v
    return frozen


def tree_hashes(llmlab_dir, bench_rel="benchmarks"):
    """Per-file sha256 of the benchmarks tree (excludes freeze.json — non-circular)."""
    root = os.path.join(llmlab_dir, bench_rel)
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            if f == "freeze.json" or f.endswith(".pyc") or "__pycache__" in p:
                continue
            out[os.path.relpath(p, root)] = sha256_file(p)
    return out


def _git_head(d):
    import subprocess
    try:
        p = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        return p.stdout.strip() or None
    except Exception:
        return None


def verify(frozen, live):
    """live: {llmlab_commit, spec_sha256, tree_hashes, bundle_sha256}.
    Returns (ok, problems). Any problem => abort the window."""
    problems = []
    if frozen.get("llmlab_commit") and live.get("llmlab_commit") and \
            frozen["llmlab_commit"] != live["llmlab_commit"]:
        problems.append(f"llmlab commit — frozen {frozen['llmlab_commit'][:12]}…, "
                        f"live {live['llmlab_commit'][:12]}…")
    if frozen.get("spec_sha256") and live.get("spec_sha256") and \
            frozen["spec_sha256"] != live["spec_sha256"]:
        problems.append(f"campaign spec changed — frozen {frozen['spec_sha256'][:12]}…, "
                        f"live {live['spec_sha256'][:12]}…")
    if frozen.get("bundle_sha256") and live.get("bundle_sha256") and \
            frozen["bundle_sha256"] != live["bundle_sha256"]:
        problems.append("bundle tarball sha mismatch")
    for f, want in (frozen.get("file_hashes") or {}).items():
        got = (live.get("tree_hashes") or {}).get(f)
        if got is None:
            problems.append(f"bundle file missing: {f}")
        elif got != want:
            problems.append(f"bundle file CHANGED: {f}")
    return (not problems), problems


def write(campaign_dir, frozen):
    path = os.path.join(campaign_dir, "freeze.json")
    with open(path, "w") as f:
        json.dump(frozen, f, indent=2)
        f.write("\n")
    return path


def read(campaign_dir):
    with open(os.path.join(campaign_dir, "freeze.json")) as f:
        return json.load(f)
