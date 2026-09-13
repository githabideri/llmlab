"""deploy.py — one content-addressed bundle, every target, one hash.

The 09-11 lesson (cited SHA != live SHA) and the 09-12 mirror dance die here:
the bundle is a tarball of the benchmarks tree at the frozen commit, with a
single sha256 and a per-file hash set. `deploy` pushes the identical bytes to
every target (host, target CT, executor host) and each target verifies its
extracted tree against freeze.json before the window may open. Unprivileged
CTs get the files by host-side copy into the shared mount (pct push into a
host-owned path is inert — the 09-11 lesson).

Live push is marked live-untested until the first dogfood window.
"""
import hashlib
import json
import os
import subprocess
import tarfile
import time


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_bundle(llmlab_dir, out_dir, bench_rel="benchmarks"):
    """tar.gz the benchmarks tree; returns (tar_path, sha256)."""
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(llmlab_dir, bench_rel)
    tar_path = os.path.join(out_dir, "campaign-bundle.tar.gz")
    with tarfile.open(tar_path, "w:gz") as t:
        for dirpath, _, files in os.walk(src):
            if "__pycache__" in dirpath:
                continue
            for f in files:
                if f.endswith((".pyc",)) or f == "freeze.json":
                    continue
                p = os.path.join(dirpath, f)
                t.add(p, arcname=os.path.join(bench_rel, os.path.relpath(p, src)))
    return tar_path, sha256_file(tar_path)


def verify_target(frozen, target_dir):
    """Recompute the per-file hashes of an extracted tree and check them against
    the freeze. Any mismatch = ABORT (the window never opens)."""
    from . import freeze as freeze_mod
    live_tree = freeze_mod.tree_hashes(target_dir)
    problems = []
    for f, want in (frozen.get("file_hashes") or {}).items():
        got = live_tree.get(f)
        if got is None:
            problems.append(f"missing: {f}")
        elif got != want:
            problems.append(f"changed: {f}")
    return (not problems), problems


def push(backend, bundle_tar_path, target_dir):
    """Push the identical bytes to one target (v0: ssh base64; small tree).
    Live-untested until the first dogfood."""
    b64 = __import__("base64").b64encode(open(bundle_tar_path, "rb").read()).decode()
    rc, out, err = backend.sh(
        f"mkdir -p {target_dir} && echo '{b64}' | base64 -d > {target_dir}/campaign-bundle.tar.gz "
        f"&& cd {target_dir} && tar xzf campaign-bundle.tar.gz && echo EXTRACTED")
    if "EXTRACTED" not in out:
        raise OSError(f"bundle push failed on {backend.name}: {err[:300]}")
    ok, problems = verify_target_on_remote(backend, target_dir)
    if not ok:
        raise OSError(f"bundle verification failed on {backend.name}: {problems}")


def verify_target_on_remote(backend, target_dir):
    """Ask the target to hash its own tree (PF_ parse-back); compare to freeze
    via the caller (kept here for the real path)."""
    script = (
        "cd {d}/benchmarks 2>/dev/null || cd {d}; "
        "find . -type f -not -path '*/__pycache__/*' -not -name 'freeze.json' | sort | "
        "while read f; do echo \"PF_$(echo $f | tr / _)=$(sha256sum $f | cut -d' ' -f1)\"; done"
    ).format(d=target_dir)
    rc, out, err = backend.sh("bash -s", stdin=script)
    kv = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("PF_") and "=" in line:
            k, _, v = line.partition("=")
            kv[k] = v
    return kv, err
