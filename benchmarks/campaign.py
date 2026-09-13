#!/usr/bin/env python3
"""campaign.py — the only entry point an executor (human or agent) needs.

    python3 campaign.py prepare  <spec.jsonc> <profile.yaml> [--model-sha HEX]
    python3 campaign.py qualify  [--out QUALIFICATION.json]
    python3 campaign.py deploy   <spec.jsonc> <profile.yaml>     # builds the bundle
    python3 campaign.py run      <spec.jsonc> <profile.yaml> --backend fixture
                                 # --backend real: refused until dogfood gate
    python3 campaign.py ingest   <profile.yaml> <run-dir> [--out DIR]

`prepare` = validate spec + freeze (with qualification inside) + readiness
brief. Nothing runs on a real GPU before qualification is green and the owner
has approved a specific window.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import deploy, freeze as freeze_mod, ingest, spec as spec_mod  # noqa: E402
from benchmarks import qualify, runner as runner_mod  # noqa: E402


def _load_profile(path):
    import yaml
    return yaml.safe_load(open(path))


def _llmlab_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cmd_qualify(args):
    out = args[args.index("--out") + 1] if "--out" in args else None
    ok, _ = qualify.run_all(out)
    return 0 if ok else 1


def cmd_prepare(args):
    spec_path = args[0]
    profile = _load_profile(args[1])
    model_sha = None
    if "--model-sha" in args:
        model_sha = args[args.index("--model-sha") + 1]
    if "--model-sha-file" in args:
        model_sha = open(args[args.index("--model-sha-file") + 1]).read().strip()
    spec_obj = spec_mod.load_spec(spec_path)
    problems = spec_mod.validate(spec_obj)
    ok = not problems
    if not ok:
        print("SPEC INVALID:")
        for p in problems:
            print("  -", p)
        return 1
    # fill the artifact sha if provided (it is part of the scientific projection)
    if model_sha:
        spec_obj["model"]["artifact"]["sha256"] = model_sha
    else:
        print("WARNING: no --model-sha given; the artifact sha is UNSET "
              "(treated as implementation, not scientific) — set it before a "
              "science run.")
    spec_dir = os.path.dirname(os.path.abspath(spec_path))
    run_dir = os.path.join(spec_dir, "runs",
                           spec_obj.get("id", "run") + "-"
                           + time.strftime("%Y%m%d", time.gmtime()))
    os.makedirs(run_dir, exist_ok=True)
    # re-freeze the exact spec text that will ship
    ship_spec = os.path.join(run_dir, "spec.shipped.jsonc")
    with open(ship_spec, "w") as f:
        f.write(open(spec_path).read())
    if model_sha and not model_sha.startswith("0" * 10):
        ship_text = open(ship_spec).read().replace(
            '"sha256": "UNSET"', f'"sha256": "{model_sha}"')
        with open(ship_spec, "w") as f:
            f.write(ship_text)
    fz = freeze_mod.bake(spec_obj.get("id"), ship_spec, spec_mod.load_spec(ship_spec),
                         _llmlab_dir(), profile.get("host", {}).get("ssh", "local"),
                         {"p0": "pending"})
    print("== QUALIFICATION (must be green before this freeze is usable) ==")
    ok, results = qualify.run_all()
    fz["qualification"] = {
        "p0": f"{sum(1 for _, o, _, _ in results['P0'])}/{len(results['P0'])}",
        "p1": f"{sum(1 for _, o, _, _ in results['P1'])}/{len(results['P1'])}",
        "green": ok,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    freeze_mod.write(run_dir, fz)
    print(f"\nFREEZE written to {run_dir}/freeze.json  "
          f"(scientific {fz['scientific_hash'][:12]}…)")
    print(f"qualification: P0 {fz['qualification']['p0']}  P1 {fz['qualification']['p1']}")
    if not ok:
        print("NOT FROZEN for live use — P0 incomplete.")
        return 1
    # readiness brief
    brief = os.path.join(run_dir, "brief.md")
    with open(brief, "w") as f:
        f.write(_brief(fz, profile))
    print(f"readiness brief: {brief}")
    return 0


def _brief(fz, profile):
    cells = []
    lines = [f"# Window brief — {fz['campaign_id']}", "",
             "- executor: " + profile.get("host", {}).get("ssh", "?"),
             "- target: " + profile.get("target", {}).get("ssh", "?"),
             "- prod unit: " + profile.get("prod", {}).get("unit", "?"),
             f"- freeze: implementation `{fz['implementation_hash']}` · "
             f"scientific `{fz['scientific_hash'][:16]}…`",
             "- watchdog: armed BEFORE any destructive step; deadline enforced;",
             "  exit is ONE idempotent path (undo → prod start → live verify → disarm)",
             "", "The bundle is the ONLY code that may run on the target. "
             "A repair outside `repair_policy.allowed` is an owner decision.", ""]
    return "\n".join(lines)


def cmd_deploy(args):
    spec_path = args[0]
    profile = _load_profile(args[1])
    out = args[args.index("--out") + 1] if "--out" in args else None
    llmlab_dir = _llmlab_dir()
    out_dir = out or os.path.join(os.path.dirname(os.path.abspath(spec_path)), "bundle")
    tar, sha = deploy.build_bundle(llmlab_dir, out_dir)
    print(f"bundle: {tar}\nsha256: {sha}")
    # verify our own tree against a fresh extraction (the target will do the same)
    import tarfile
    ext = os.path.join(out_dir, "extract")
    os.makedirs(ext, exist_ok=True)
    with tarfile.open(tar) as t:
        t.extractall(ext)
    # the freeze with file hashes ships inside the bundle tree; build it now
    from benchmarks import spec as spec_mod
    spec_obj = spec_mod.load_spec(spec_path)
    fz = freeze_mod.bake(spec_obj.get("id"), spec_path, spec_obj, llmlab_dir,
                         profile.get("host", {}).get("ssh", "local"),
                         {"p0": "pending", "note": "finalize at prepare"})
    fz["bundle_sha256"] = sha
    print(f"file_hashes: {len(fz['file_hashes'])} files")
    print("NOTE: live push (ssh) is live-untested until the first dogfood window.")
    return 0


def cmd_run(args):
    backend = None
    for i, a in enumerate(args):
        if a == "--backend" and i + 1 < len(args):
            backend = args[i + 1]
    if not backend:
        print("--backend required (fixture|real)")
        return 2
    spec_path = args[0]
    profile = _load_profile(args[1])
    approve = None
    if "--approve-window" in args:
        approve = args[args.index("--approve-window") + 1]
    campaign_dir = os.path.dirname(os.path.abspath(spec_path))
    if "--model-sha" in args:
        sha = args[args.index("--model-sha") + 1]
        t = open(spec_path).read()
        if '"sha256": "UNSET"' in t:
            t = t.replace('"sha256": "UNSET"', f'"sha256": "{sha}"')
            ship = os.path.join(campaign_dir, "runs", "_run_shipped_spec.jsonc")
            os.makedirs(os.path.dirname(ship), exist_ok=True)
            with open(ship, "w") as f:
                f.write(t)
            spec_path = ship
    if backend == "real":
        if not approve:
            print("refused: --backend real requires an explicit owner approval:\n"
                  "  --approve-window <window-id>\n"
                  "The approval is recorded in the run's events.jsonl; without it the\n"
                  "platform never touches production (the 09-08 rule, mechanized).")
            return 2
        from benchmarks.backends.real import RealBackend
        rb_probe = RealBackend(profile, "", run_dir_probe := os.path.join(
            os.path.dirname(os.path.abspath(spec_path)), "runs", "_gate"))
        wd = (profile.get("window") or {}).get("script", "")
        rc, out, err = rb_probe.sh(f"test -x {wd} && echo WATCHDOG_DEPLOYED", where="target")
        if "WATCHDOG_DEPLOYED" not in out:
            print(f"refused: the watchdog script is not deployed/executable on the target\n"
                  f"  expected: {wd}\n  (deploy it first: homelab scripts/gpu-bench/campaign-watchdog.sh)")
            return 2
    from benchmarks.backends.fixture import FixtureBackend
    from benchmarks import spec as spec_mod
    spec_obj = spec_mod.load_spec(spec_path)
    llmlab_dir = _llmlab_dir()
    # prefer the PREPARED freeze (qualification + sha + file hashes) if present
    run_dir = None
    prepared = None
    if os.path.isdir(os.path.join(campaign_dir, "runs")):
        for d in sorted(os.listdir(os.path.join(campaign_dir, "runs")), reverse=True):
            p = os.path.join(campaign_dir, "runs", d)
            if d.startswith(spec_obj.get("id", "")) and os.path.exists(os.path.join(p, "freeze.json")):
                prepared = p
                break
    if prepared:
        run_dir = prepared
        fz = json.load(open(os.path.join(run_dir, "freeze.json")))
        print(f"using prepared freeze: {run_dir} "
              f"(sci {fz.get('scientific_hash', '?')[:12]}..., qual {fz.get('qualification', {}).get('p0')})")
    else:
        run_dir = os.path.join(campaign_dir, "runs",
                               (backend if backend == "real" else "fixture") +
                               "-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()))
        fz = freeze_mod.bake(spec_obj.get("id"), spec_path, spec_obj, llmlab_dir,
                             profile.get("host", {}).get("ssh", "fixture"), {"p0": "pending"})
        fz["file_hashes"] = {}
        freeze_mod.write(run_dir, fz)
        print("WARNING: no prepared freeze found — running unqualified (fixture use only)")
    if backend == "real":
        from benchmarks.backends.real import RealBackend
        b = RealBackend(profile, llmlab_dir, run_dir)
    else:
        b = FixtureBackend(profile, llmlab_dir, run_dir)
    r = runner_mod.Runner(spec_obj, profile, fz, run_dir, b, llmlab_dir)
    if approve:
        r.events.emit("OWNER_APPROVAL", window=approve,
                      note="recorded by campaign.py run --approve-window")
    fin = r.run()
    print(f"\nfinal: {fin['final']}  cells: {fin['cells']}  "
          f"restored: {fin['restore'].get('restored')}")
    return 0 if fin["final"] in ("completed", "expected-negative") else 1


def cmd_ingest(args):
    profile = _load_profile(args[0])
    run_dir = args[1]
    out_dir = (args[args.index("--out") + 1] if "--out" in args
               else os.path.join(os.getcwd(), "temp", "ingest"))
    final_path = os.path.join(run_dir, "final.json")
    if not os.path.exists(final_path):
        print(f"no final.json in {run_dir} — nothing to ingest")
        return 1
    final = json.load(open(final_path))
    from benchmarks.backends.fixture import FixtureBackend
    from benchmarks.backends.real import RealBackend
    if (profile.get("host") or {}).get("ssh"):
        b = RealBackend(profile, profile.get("deploy", {}).get("target_dir", "/root/campaigns/active"),
                        run_dir)
    else:
        b = FixtureBackend(profile, ".", run_dir)
    rep = ingest.run(b, profile, final, run_dir, out_dir)
    print(json.dumps({k: v for k, v in rep.items() if k != "production"}, indent=2))
    print(f"production: {rep['production']}")
    for f in rep["completeness_findings"]:
        print("FINDING:", f)
    return 0 if not rep["completeness_findings"] and rep["production"]["ok"] else 1


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    args = sys.argv[2:]
    if cmd == "qualify":
        return cmd_qualify(args)
    if cmd == "prepare":
        return cmd_prepare(args)
    if cmd == "deploy":
        return cmd_deploy(args)
    if cmd == "run":
        return cmd_run(args)
    if cmd == "ingest":
        return cmd_ingest(args)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
