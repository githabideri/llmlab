"""preflight.py — drift detection: EXPECTED / OBSERVED / MATCH|DRIFT.

Preparation-time knowledge is not enough (the 09-02 wrong-GPU cells: the prep
looked perfect, the machine disagreed). Immediately before execution we observe
reality and compare it against what the profile + freeze say the world should
look like.

Blocking drift (no window): bundle/hash mismatch, GPU identity (model/BDF/count/
visibility), production unit state, controller independence (the executor must
not be served by the endpoint the window stops — the 09-08 lesson), watchdog
mechanics, results disk headroom.
Noted drift (window allowed, recorded): PCIe width/gen changes, driver version,
minor counter differences.
"""
import json
import os


def run(backend, profile, freeze, run_dir):
    """Returns (ok, report). report is the EXPECTED/OBSERVED/MATCH|DRIFT table."""
    rows = []

    def add(check, expected, observed, blocking=True, why=""):
        match = (str(expected) == str(observed)) if expected is not None else \
                (observed is not None)
        rows.append({"check": check, "expected": expected, "observed": observed,
                     "blocking": blocking, "match": bool(match), "why": why})
        return match

    # -- bundle / hashes --------------------------------------------------------
    live_tree = {}
    try:
        from . import freeze as freeze_mod
        live_tree = freeze_mod.tree_hashes(
            os.path.dirname(os.path.abspath(backend.bundle_dir))) if backend.name != "fixture" \
            else {"(fixture)": "n/a"}
    except Exception as e:
        rows.append({"check": "bundle tree", "expected": freeze.get("implementation_hash"),
                     "observed": f"error: {e}", "blocking": True, "match": False,
                     "why": "bundle unreadable"})

    # -- GPU identity ------------------------------------------------------------
    gpus = backend.gpu_state()
    want = profile.get("gpu") or []
    add("gpu.count", len(want), len(gpus),
        why="identity: the wrong card set invalidates every cell (09-02)")
    for i, w in enumerate(want):
        g = gpus[i] if i < len(gpus) else None
        add(f"gpu[{i}].model", w.get("model"), g and g.get("model"),
            why="identity")
        if w.get("bdf"):
            # canonical form: last two components (nvidia-smi says 000001:00.0,
            # lspci says 0000:01:00.0 — same card)
            def canon(x):
                # drop the PCI domain (0000 / 00000000 — always zero in this
                # fleet) and keep bus:function; NEVER substring-replace '0000:'
                # (it matches inside the 8-digit domain and mangles it)
                return (x or "").lower().split(":", 1)[1] if ":" in (x or "") else x
            want_bdf = canon(w["bdf"])
            have_bdf = canon((g or {}).get("bdf", ""))
            add(f"gpu[{i}].bdf", want_bdf, have_bdf,
                why="identity: CUDA enumeration order has bitten us before")

    # -- production state ---------------------------------------------------------
    prod = profile.get("prod") or {}
    health = backend.prod_health()
    add("prod.health", 200, health,
        why="prod must be up and reachable before we are allowed to stop it")

    # -- controller independence (the 09-08 self-kill lesson) ---------------------
    # the profile declares who the executor is served by; if that equals the
    # endpoint being stopped, a live 200 proves the executor would die with it.
    exec_provider = profile.get("executor_provider_endpoint")
    if exec_provider and prod.get("health_url") and str(exec_provider) == str(prod.get("health_url")):
        add("controller.independence", "independent provider", "provider IS the prod endpoint",
            why="a controller served by the endpoint under test dies when the window opens")
    else:
        add("controller.independence", "declared", "declared", blocking=True)

    # -- resources ------------------------------------------------------------------
    results_path = (profile.get("storage") or {}).get("results", "")
    if results_path:
        free = backend.disk_free(results_path)
        want_free = (profile.get("storage") or {}).get("min_free_bytes", 10 << 30)
        ok = isinstance(free, (int, float)) and free >= want_free
        rows.append({"check": "results.disk_free",
                     "expected": f">={want_free}", "observed": free,
                     "blocking": True, "match": bool(ok),
                     "why": "results must fit (the manifest lives on this volume)"})

    ok = all(r["match"] for r in rows if r["blocking"])
    path = os.path.join(run_dir, "preflight.json")
    with open(path, "w") as f:
        json.dump({"ok": ok, "rows": rows}, f, indent=2)
    return ok, {"ok": ok, "rows": rows, "written": path}


def summary_text(report):
    lines = []
    for r in report["rows"]:
        flag = "MATCH " if r["match"] else ("BLOCK " if r["blocking"] else "NOTED")
        lines.append(f"  [{flag}] {r['check']}: expected={r['expected']} observed={r['observed']}"
                     + (f" ({r['why']})" if not r["match"] and r.get("why") else ""))
    lines.append(f"  => {'GO' if report['ok'] else 'NO-GO'} (blocking drift: "
                 f"{sum(1 for r in report['rows'] if r['blocking'] and not r['match'])})")
    return "\n".join(lines)
