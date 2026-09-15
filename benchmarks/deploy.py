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


def push(backend, bundle_tar_path, target_dir, frozen):
    """Push the identical bytes to one target, then verify the extracted tree.

    Transport: b64 over ssh STDIN (a full bundle's b64 far exceeds
    MAX_ARG_STRLEN in an argv element — the 09-14 E2BIG shape).
    Verify: the remote hashes its own tree; every file in the freeze must
    be present and match, and no unexpected file may appear. A freeze
    WITHOUT file_hashes is refused — verifying against nothing is how the
    09-11 'cited SHA != live SHA' lesson was silently neutered (09-14: the
    entire campaign ran on a bundle nobody pushed, because this function
    had no call site and its verifier could not fail).
    """
    if not (frozen.get("file_hashes") or {}):
        raise OSError("bundle push refused: freeze has no file_hashes — "
                      "nothing to verify against")
    import base64
    local_sha = sha256_file(bundle_tar_path)
    b64 = base64.b64encode(open(bundle_tar_path, "rb").read()).decode()
    rc, out, err = backend.sh(
        f"mkdir -p {target_dir} && base64 -d > {target_dir}/campaign-bundle.tar.gz && "
        f"cd {target_dir} && sha256sum campaign-bundle.tar.gz | cut -d' ' -f1 && "
        f"tar xzf campaign-bundle.tar.gz && echo EXTRACTED",
        stdin=b64 + "\n")
    if "EXTRACTED" not in out:
        raise OSError(f"bundle push failed on {backend.name}: {err[:300]}")
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if not lines or lines[0] != local_sha:
        raise OSError(f"bundle sha mismatch on {backend.name}: "
                      f"local {local_sha[:12]}... remote {lines[0][:12] if lines else '?'}...")
    kv = remote_tree_hashes(backend, target_dir)
    problems = []
    for f, want in sorted(frozen["file_hashes"].items()):
        got = kv.get(f)
        if got is None:
            problems.append(f"missing: {f}")
        elif got != want:
            problems.append(f"changed: {f}")
    for f in sorted(kv):
        if f not in frozen["file_hashes"]:
            problems.append(f"unexpected: {f}")
    if problems:
        raise OSError(f"bundle verification failed on {backend.name}: "
                      f"{problems[:10]}")
    return local_sha


def remote_tree_hashes(backend, target_dir):
    """Ask the target to hash its own extracted tree (PF_ parse-back).

    Keys match freeze.tree_hashes: paths relative to the benchmarks root
    (the bundle extracts to <target_dir>/benchmarks/...)."""
    script = (
        f"cd {target_dir}/benchmarks 2>/dev/null || exit 3; "
        "find . -type f -not -path '*/__pycache__/*' -not -name 'freeze.json' | sort | "
        "while read f; do h=$(sha256sum \"$f\" | cut -d' ' -f1); "
        "echo \"PF_ $f $h\"; done"
    )
    rc, out, err = backend.sh("bash -s", stdin=script)
    if rc != 0:
        raise OSError(f"remote tree hashing failed on {backend.name}: {err[:200]}")
    kv = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("PF_ "):
            rel, _, h = line[4:].partition(" ")
            kv[rel.lstrip("./")] = h.strip()
    return kv
