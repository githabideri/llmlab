"""qualify.py — the deploy gate: the REAL runner, exercised against the fixture.

This is the difference between 2026-09-10 (a dry-run proved the state machine and
then the night was spent on a wrong venv path and a quoting layer) and 2026-09-12
(cleanest night: the Python package had a selftest that executed the real code).
There is no parallel fake runner here: `Runner` runs against `FixtureBackend`,
the clients run as real subprocesses against a real in-process HTTP server, and
the historical failure corpus is replayed through the real classifier.

P0 (must all pass for a bundle to be FROZEN) — each row is the recurrence of a
specific observed failure:

   Q1  clean cell, real client over real sockets, metrics + determinism
   Q2  documented wall (25.4 GB alloc, no assert) -> EXPECTED_NEGATIVE, STOP-done,
       no "needs human review"                     [2026-09-12 gate failure]
   Q3  MiB-sized OOM below the wall -> RESOURCE_LIMIT (unit-blind matcher fix)
       [2026-09-12 classify.sh]
   Q4  meta-backend ASSERT + OOM -> BUILD_DEFECT_ASSERT / HARNESS_FAILURE, NOT
       the wall (precedence)                       [2026-09-12 B1, re-clarified 09-13]
   Q5  empty-body 200 /health -> readiness by HTTP code, cell proceeds
       [2026-09-02/09-10 body-grepping probes]
   Q6  runner hard-crash mid-campaign -> watchdog fires, restore verified,
       fresh runner RESUMES: cell 1 skipped, cell 2 runs [09-10 crash era]
   Q7  restore called twice -> idempotent: prod started once, watchdog disarmed
       once                                      [09-10 non-converging exit]
   Q8  host reboot mid-campaign (btime change) -> stop, forensics, restore,
       NO auto-resume                             [09-10 host crash]
   Q9  PVE dialect replay: temp changes rendered only from the verified command
       table; the 09-12 hand-rolled forms are rejected [2026-09-12 maint-fn.sh]
   Q10 false-complete: 2 tokens in 50 ms against min_wall_s/min_tokens gates
       -> INVALID, never PASS                     [09-10 LADDER-256k]

P1 (strongly recommended; a failure warns, does not block the freeze):
   Q11 malformed SSE frame -> HARNESS_FAILURE      [09-02 stage-1 client bugs]
   Q12 unsupported-arch control cell with documented_negative -> EXPECTED_NEGATIVE
       [09-12 B0]
   Q13 no-usage stream shape -> classifier SSE_NO_USAGE [09-10 usage lesson]

Usage:  qualify.py [--bundle DIR] [--out DIR]      exit 0 iff all P0 pass
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_mod(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------- helpers

class _Crash(BaseException):
    """A hard crash: not an Exception, so the runner's exit path never runs —
    exactly the mid-campaign death the watchdog exists for."""


def _spec(cells, doc_neg=None, stop_on=False, gates=None, client="llama"):
    client_block = {
        "llama": {"script": "clients/bench-llama.py",
                  "args": ["python3", "{bundle}/clients/bench-llama.py",
                           "--url", "http://127.0.0.1:{port}",
                           "--prompt-file", "{cell}/prompt.txt",
                           "--decode", "64", "--reps", "2", "--seed", "1",
                           "--json-out", "{cell}/client.json"]},
        "vllm": {"script": "clients/bench-vllm.py",
                 "args": ["python3", "{bundle}/clients/bench-vllm.py",
                          "--server", "http://127.0.0.1:{port}",
                          "--mode", "tgen", "--ctx-k", "0.064", "--ntok", "64",
                          "--reps", "2", "--min-overlap-s", "1",
                          "--out", "{cell}/client.json"]},
    }[client]
    return {
        "id": "qualify",
        "question": "platform qualification (no science)",
        "model": {"name": "fixture-1b", "quant": "Q4_K_M",
                  "artifact": {"source": "fixture", "sha256": "0" * 64}},
        "workload": {"client": "llama"},
        "client": client_block,
        "matrix": cells,
        "verdict_policy": {
            "retry": {"max": 2, "on": ["RETRYABLE_FAILURE", "RETRYABLE_INFRA"]},
            "stop_review": ["UNKNOWN"],
        },
        "repair_policy": {
            "allowed": ["runtime_path", "command_syntax", "parser_exception",
                        "log_format_support", "missing_fixture_copy", "quoting"],
            "requires_owner": ["metric_threshold", "workload", "expected_outcome",
                               "model", "quant", "benchmark_duration", "build_when_scientific"],
            "forbidden": ["delete_evidence", "invalid_to_pass", "skip_required_cell",
                          "rewrite_previous_attempt"],
        },
        "stop_policy": {"deadline_h": 0.05, "cell_max_s": 120},
    }


def _cell(cid, doc_neg=None, gates=None, stop_on=False):
    c = {"cell": cid,
         "launch": {"binary": "/opt/<build>/bin/llama-server",
                    "args": ["-m", "{model}", "-ngl", "99", "-c", "4096",
                             "--port", "{port}", "--host", "127.0.0.1"]},
         "requests": [{"kind": "completion", "prompt_tokens": 32, "decode": 64, "reps": 2}],
         "gates": gates or {"min_wall_s": 0.1, "min_tokens": 32, "min_tps": 1,
                            "require_determinism": True}}
    if doc_neg is not None:
        c["expects"] = {"documented_negative": doc_neg}
        c["stop_on_expected_negative"] = bool(stop_on)
    return c


def _profile(tmp, fault="ok"):
    return {
        "id": "fixture",
        "host": {"ssh": "fixture-host"},
        "target": {"ssh": "fixture-target"},
        "pve": {"version": 9, "dialect": "pve9"},
        "gpu": [{"model": "RTX 3060 12GB", "bdf": "00:01.0"}],
        "prod": {"unit": "fixture-prod",
                 "stop": "systemctl stop fixture-prod",
                 "start": "systemctl start fixture-prod",
                 "health_url": "http://127.0.0.1:8080/health",
                 "live_check_cmd": "curl -s fixture-live && echo LIVE_OK"},
        "storage": {"results": os.path.join(tmp, "results"),
                    "min_free_bytes": 1 << 30,
                    "model_path": "/mnt/<models>/gguf/fixture-1b.gguf"},
        "window": {"dir": os.path.join(tmp, "watchdog"),
                   "script": "/usr/local/bin/campaign-watchdog.sh",
                   "lease": os.path.join(tmp, "campaign.lock"),
                   "heartbeat": os.path.join(tmp, "campaign-hb"),
                   "restore_result": os.path.join(tmp, "results", "RESTORE-RESULT.md")},
        "deploy": {"target_dir": os.path.join(tmp, "bundle")},
        "readiness": {"attempts": 5, "sleep": 0.2},
        "ports": {"base": 18100},
    }


def _ctx(tmp, fault, spec=None, profile=None):
    """Assemble (runner, backend, run_dir, freeze) for a scenario."""
    from benchmarks import freeze as freeze_mod, runner as runner_mod
    from benchmarks.backends.fixture import FixtureBackend
    profile = profile or _profile(tmp, fault)
    spec = spec or _spec([_cell("c1")])
    run_dir = os.path.join(tmp, "runs", spec["id"] + "-" + fault)
    os.makedirs(run_dir, exist_ok=True)
    spec_file = os.path.join(tmp, "spec.yaml")
    import json as _json
    with open(spec_file, "w") as f:
        f.write(_json.dumps(spec, indent=2))
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    fz = freeze_mod.bake("qualify-" + fault, spec_file, spec,
                         bundle_dir, "fixture", {"p0": "pending"})
    fz["file_hashes"] = {}
    freeze_mod.write(run_dir, fz)
    b = FixtureBackend(profile, bundle_dir, run_dir, fault=fault)
    r = runner_mod.Runner(spec, profile, fz, run_dir, b, bundle_dir, max_attempts=1)
    return r, b, run_dir, fz, spec, profile


P0 = []
P1 = []


def p0(name, failure):
    P0.append((name, failure))
    return failure


def p1(name, failure):
    P1.append((name, failure))
    return failure


# ------------------------------------------------------------------ P0 scenarios

def q1_clean(tmp):
    from benchmarks.runner import Runner
    r, b, run_dir, fz, spec, prof = _ctx(tmp, "ok")
    fin = r.run()
    ok = (fin["final"] == "completed" and fin["cells"].get("c1") == "PASS"
          and fin["valid"] and not fin["restore"].get("restored") is False
          and fin["restore"]["restored"] is True)
    a = os.path.join(run_dir, "attempts", "c1-01")
    art_ok = all(os.path.exists(os.path.join(a, f))
                 for f in ("launch.sh", "meta.json", "prompt.txt", "client.json",
                           "verdict.json", "nvml.csv"))
    cli = json.load(open(os.path.join(a, "client.json")))
    det_ok = cli.get("determinism") in ("OK", "SINGLE")
    return ok and art_ok and det_ok, \
        f"final={fin['final']} cell={fin['cells'].get('c1')} artifacts={art_ok} det={det_ok}"


def q2_documented_wall(tmp):
    r, b, run_dir, fz, spec, prof = _ctx(tmp, "fail-oom-25g",
                                         spec=_spec([_cell("wall",
                                                           doc_neg=["WALL_VRAM_FIT"],
                                                           stop_on=True)]))
    fin = r.run()
    ok = (fin["final"] == "expected-negative"
          and fin["cells"].get("wall") == "EXPECTED_NEGATIVE"
          and fin["restore"]["restored"] is True)
    return ok, f"final={fin['final']} cell={fin['cells'].get('wall')} (want expected-negative, NOT review)"


def q3_mib_oom(tmp):
    from benchmarks.classifier import engine
    # the 09-12 classifier bug: 'MiB' sizes were invisible -> 'generic OOM'
    res = engine.classify("allocating 8589.93 MiB on device 0: cudaMalloc failed: out of memory")
    ok = res["class"] == "CUDA_OOM"
    r, b, run_dir, fz, spec, prof = _ctx(tmp, "fail-oom-mib")
    fin = r.run()
    ok = ok and fin["cells"].get("c1") == "RESOURCE_LIMIT"
    return ok, f"classifier={res['class']} cell={fin['cells'].get('c1')} (want CUDA_OOM/RESOURCE_LIMIT)"


def q4_meta_assert(tmp):
    from benchmarks.classifier import engine
    log = ("ggml-backend-meta.cpp:1760: GGML_ASSERT(meta_buf_ctx->bufs[i]) failed\n"
           "allocating 24224.99 MiB on device 0: cudaMalloc failed: out of memory\n"
           "alloc_tensor_range: failed to allocate CUDA0 buffer of size 25401738496")
    res = engine.classify(log)
    ok = res["class"] == "BUILD_DEFECT_ASSERT"     # assert outranks the 25.4 GB OOM
    r, b, run_dir, fz, spec, prof = _ctx(tmp, "fail-assert")
    fin = r.run()
    v = fin["cells"].get("c1")
    ok = ok and v == "HARNESS_FAILURE" and not fin["final"] == "expected-negative"
    return ok, f"classifier={res['class']} cell={v} (want BUILD_DEFECT_ASSERT/HARNESS_FAILURE, not the wall)"


def q5_empty_200_health(tmp):
    from benchmarks.classifier import engine
    res = engine.classify_response("health", "")
    ok = res["class"] == "HTTP_EMPTY_200"
    r, b, run_dir, fz, spec, prof = _ctx(tmp, "prod-health-empty200")
    fin = r.run()
    ok = ok and fin["final"] == "completed" and fin["cells"].get("c1") == "PASS"
    return ok, f"health-shape={res['class']} final={fin['final']} cell={fin['cells'].get('c1')} (empty 200 must proceed)"


def q6_crash_watchdog_resume(tmp):
    """Hard crash mid-campaign: the watchdog restores; a fresh runner resumes
    (cell 1 skipped as complete, cell 2 runs)."""
    from benchmarks import runner as runner_mod
    from benchmarks.backends.fixture import FixtureBackend
    from benchmarks import freeze as freeze_mod
    spec = _spec([_cell("c1"), _cell("c2")])
    spec_file = os.path.join(tmp, "spec.yaml")
    import json as _json
    with open(spec_file, "w") as f:
        f.write(_json.dumps(spec, indent=2))
    run_dir = os.path.join(tmp, "runs", "q6-crash")
    os.makedirs(run_dir, exist_ok=True)
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    fz = freeze_mod.bake("q6", spec_file, spec, bundle_dir, "fixture",
                         {"p0": "pending"})
    fz["file_hashes"] = {}
    freeze_mod.write(run_dir, fz)
    b1 = FixtureBackend(_profile(tmp), bundle_dir, run_dir, fault="ok")
    # crash after the first cell completes: run cell 1 manually, then hard-kill
    r1 = runner_mod.Runner(spec, _profile(tmp), fz, run_dir, b1, bundle_dir, max_attempts=1)
    r1.run.__globals__["_Crash"] = _Crash
    try:
        # monkeypatch: after the first cell, die hard (no exit path runs)
        orig = r1._run_cell
        def crash_after_first(cell, idx):
            v = orig(cell, idx)
            if cell["cell"] == "c1":
                r1.cells_done[cell["cell"]] = v
                r1._state_save()
                raise _Crash()
            return v
        r1._run_cell = crash_after_first
        r1.run()
        crashed = False
    except _Crash:
        crashed = True
    # the watchdog fires (deadline/lease still held)
    b1.simulate_watchdog_fire()
    restored = b1.watchdog.get("fired") and b1.prod_state == "active"
    rr = os.path.join(b1.p["window"]["restore_result"])
    rr_ok = os.path.exists(rr) and "VERDICT: RESTORED" in open(rr).read()
    # a FRESH runner (fresh process, same run dir) resumes
    b2 = FixtureBackend(_profile(tmp), bundle_dir, run_dir, fault="ok")
    r2 = runner_mod.Runner(spec, _profile(tmp), fz, run_dir, b2, bundle_dir, max_attempts=1)
    fin = r2.run()
    resumed = ("CELL_SKIPPED" in [json.loads(l)["event"] for l in open(os.path.join(run_dir, "events.jsonl"))]
               and fin["cells"].get("c1") == "PASS" and fin["cells"].get("c2") == "PASS"
               and fin["final"] == "completed")
    ok = crashed and restored and rr_ok and resumed
    return ok, f"crashed={crashed} watchdog_restored={restored} result_file={rr_ok} resume={resumed} final={fin['final']}"


def q7_restore_idempotent(tmp):
    from benchmarks.window import Window
    from benchmarks.backends.fixture import FixtureBackend
    from benchmarks import freeze as freeze_mod
    spec = _spec([_cell("c1")])
    spec_file = os.path.join(tmp, "spec.yaml")
    import json as _json
    with open(spec_file, "w") as f:
        f.write(_json.dumps(spec, indent=2))
    run_dir = os.path.join(tmp, "runs", "q7")
    os.makedirs(run_dir, exist_ok=True)
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    fz = freeze_mod.bake("q7", spec_file, spec, bundle_dir, "fixture",
                         {"p0": "pending"})
    fz["file_hashes"] = {}
    freeze_mod.write(run_dir, fz)
    b = FixtureBackend(_profile(tmp), bundle_dir, run_dir)
    w = Window(b, _profile(tmp), run_dir, deadline_h=1)
    w.enter()          # the 09-10 state: prod was down when exit ran twice
    r1 = w.exit("first")
    r2 = w.exit("second")
    starts = sum(1 for e in b.events() if e == "prod_start")
    disarms = sum(1 for e in b.events() if e == "disarm_watchdog")
    ok = (r1["restored"] is True and r2["restored"] is True
          and starts == 1 and disarms == 1 and r1 == r2)
    return ok, f"prod_start x{starts} disarm x{disarms} (want 1/1); r1={r1['restored']} r2={r2['restored']}"


def q8_host_reboot(tmp):
    from benchmarks.backends.fixture import FixtureBackend
    from benchmarks import freeze as freeze_mod
    spec = _spec([_cell("c1"), _cell("c2")])
    spec_file = os.path.join(tmp, "spec.yaml")
    import json as _json
    with open(spec_file, "w") as f:
        f.write(_json.dumps(spec, indent=2))
    run_dir = os.path.join(tmp, "runs", "q8")
    os.makedirs(run_dir, exist_ok=True)
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    prof = _profile(tmp)
    fz = freeze_mod.bake("q8", spec_file, spec, bundle_dir, "fixture",
                         {"p0": "pending"})
    fz["file_hashes"] = {}
    freeze_mod.write(run_dir, fz)
    b = FixtureBackend(prof, bundle_dir, run_dir, fault="reboot-mid")
    b.reboot_after_checks = 1          # first btime read is T0; after that it jumps
    from benchmarks import runner as runner_mod
    r = runner_mod.Runner(spec, prof, fz, run_dir, b, bundle_dir, max_attempts=1)
    fin = r.run()
    forensics = os.path.exists(os.path.join(run_dir, "host-forensics.txt"))
    no_resume = fin["final"] == "stopped-host-failure" and "c2" not in fin["cells"]
    ok = no_resume and forensics and fin["restore"]["restored"] is True
    return ok, f"final={fin['final']} cells={fin['cells']} forensics={forensics} (want stop, no auto-resume, prod restored)"


def q9_pve_dialect(tmp):
    from benchmarks import dialects
    good = dialects.render("pve9", "memory_set", "lxc", id=382, mi=61440)
    ok = (good == "pct set 382 --memory 61440")
    # the 09-12 hand-rolled forms must be unrenderable (not in the table)
    bad1 = dialects.render("pve9", "memory_kv", "lxc", id=382, mi=61440)   # 'memory:NNN' style
    bad2 = dialects.render("pve9", "restart", "lxc")                      # 'pct restart'
    ok = ok and bad1 is None and bad2 is None
    # dogfood #2 (2026-09-13): the guest axis. VMs use qm, and a pct command
    # must NOT validate for a vm profile (and vice versa).
    vm_good = dialects.render("pve9", "memory_set", "vm", id=135, mi=9216)
    ok = (ok and vm_good == "qm set 135 --memory 9216"
          and dialects.validate("pve9", vm_good, "vm")
          and not dialects.validate("pve9", vm_good, "lxc")
          and not dialects.validate("pve9", "pct set 135 --memory 9216", "vm")
          and not dialects.validate("pve9", "qm set 135 memory:9216", "vm"))
    # and the rendered form must flow through apply_temp_changes
    from benchmarks.backends.fixture import FixtureBackend
    from benchmarks import freeze as freeze_mod
    spec = _spec([_cell("c1")])
    spec_file = os.path.join(tmp, "spec.yaml")
    import json as _json
    with open(spec_file, "w") as f:
        f.write(_json.dumps(spec, indent=2))
    run_dir = os.path.join(tmp, "runs", "q9")
    os.makedirs(run_dir, exist_ok=True)
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    prof = _profile(tmp)
    prof["temp_changes"] = [{"cmd": good, "undo": "pct set 382 --memory 16384"}]
    fz = freeze_mod.bake("q9", spec_file, spec, bundle_dir, "fixture",
                         {"p0": "pending"})
    fz["file_hashes"] = {}
    freeze_mod.write(run_dir, fz)
    b = FixtureBackend(prof, bundle_dir, run_dir)
    from benchmarks.window import Window
    w = Window(b, prof, run_dir, deadline_h=1, temp_changes=prof["temp_changes"])
    w.enter()
    applied = [e for e in b.events() if e.startswith("apply_temp_changes")]
    ok = ok and any(good in a for a in applied)
    return ok, f"rendered={good!r} bad1={bad1} bad2={bad2} applied={applied[:1]}"


def q10_false_complete(tmp):
    r, b, run_dir, fz, spec, prof = _ctx(
        tmp, "tiny-fast",
        spec=_spec([_cell("c1", gates={"min_wall_s": 0.5, "min_tokens": 32, "min_tps": 1})]))
    fin = r.run()
    ok = (fin["cells"].get("c1") == "INVALID" and fin["final"] != "completed")
    return ok, f"cell={fin['cells'].get('c1')} final={fin['final']} (2 tokens in 50 ms must be INVALID, not PASS)"


# ------------------------------------------------------------------ P1 scenarios

def q11_malformed_sse(tmp):
    from benchmarks.classifier import engine
    res = engine.classify_response("sse", "Expecting value: line 1 column 2 (char 1)")
    ok = res["class"] == "SSE_MALFORMED"
    r, b, run_dir, fz, spec, prof = _ctx(tmp, "sse-malformed")
    fin = r.run()
    ok = ok and fin["cells"].get("c1") == "HARNESS_FAILURE"
    return ok, f"classifier={res['class']} cell={fin['cells'].get('c1')} (malformed stream = our client broke)"


def q12_arch_control(tmp):
    r, b, run_dir, fz, spec, prof = _ctx(
        tmp, "fail-arch",
        spec=_spec([_cell("ctrl", doc_neg=["UNSUPPORTED_ARCH"], stop_on=True)]))
    fin = r.run()
    ok = fin["final"] == "expected-negative" and fin["cells"].get("ctrl") == "EXPECTED_NEGATIVE"
    return ok, f"final={fin['final']} cell={fin['cells'].get('ctrl')} (control cell documents the unsupported arch)"


def q13_no_usage(tmp):
    from benchmarks.classifier import engine
    res = engine.classify_response("usage", None)
    ok = res["class"] == "SSE_NO_USAGE"
    return ok, f"classifier={res['class']} (no usage -> token closure unavailable is a known shape)"


def q14_corpus_replay(tmp):
    """The historical failure corpus — every fixture must classify to its .expect."""
    from benchmarks.classifier import engine
    corpus = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "failures")
    failures = []
    for f in sorted(os.listdir(corpus)):
        if not (f.endswith(".log") or f.endswith(".resp")):
            continue
        expect = open(os.path.join(corpus, f + ".expect")).read().strip()
        if f.endswith(".log"):
            got = engine.classify(open(os.path.join(corpus, f)).read())["class"]
        else:
            kind = f.split(".")[0].split("-")[0]      # health / sse / usage
            body = open(os.path.join(corpus, f)).read().strip()
            got = engine.classify_response(kind, body)["class"]
        if got != expect:
            failures.append(f"{f}: got {got}, want {expect}")
    return (not failures), (", ".join(failures) if failures else f"{len(os.listdir(corpus)) // 2} fixtures")


# Q24-Q27 test bodies (spliced into qualify.py by the maintainers) —
# LMCache-campaign platform features: sidecar lifecycle, segment relaunch,
# zero-count effect gate, probe substitution + isolation reset + declared
# documented-negative.
def _mk2(tmp, cells):
    import json as _j
    from benchmarks import runner as runner_mod
    from benchmarks.backends.fixture import FixtureBackend
    bundle = os.path.join(tmp, "bundle")
    os.makedirs(os.path.join(bundle, "clients"), exist_ok=True)
    with open(os.path.join(bundle, "clients", "test-client.py"), "w") as f:
        f.write(_client_recorder())
    spec = {"id": "t", "owner": "t",
            "model": {"name": "m", "quant": "q",
                      "artifact": {"source": "gguf", "sha256": "0" * 64}},
            "workload": {"client": "x"},
            "client": {"script": "clients/test-client.py",
                       "args": ["python3", "{bundle}/clients/test-client.py",
                                "--url", "http://127.0.0.1:{port}",
                                "--json-out", "{cell}/client.json"]},
            "matrix": cells,
            "sidecar": {"script": "/opt/cam/sidecar.sh"},
            "verdict_policy": {"retry": {"max": 1, "on": ["RETRYABLE_FAILURE"]},
                               "stop_review": ["UNKNOWN"]},
            "stop_policy": {"deadline_h": 0.1, "cell_max_s": 60}}
    prof = _profile(tmp)
    prof["readiness"] = {"attempts": 20, "sleep": 0.1}
    prof["window"].update({"health_wait_s": 5})
    run_dir = os.path.join(tmp, "runs", "t")
    b2 = FixtureBackend(prof, bundle, run_dir)
    r2 = runner_mod.Runner(spec, prof, {}, run_dir, b2, bundle, max_attempts=1)
    return b2, r2


def _lmcache_fixture_campaign(tmp, spec_extra, client_src, fault="ok"):
    import json as _json
    from benchmarks import runner as runner_mod
    from benchmarks.backends.fixture import FixtureBackend
    bundle = os.path.join(tmp, "bundle")
    os.makedirs(os.path.join(bundle, "clients"), exist_ok=True)
    with open(os.path.join(bundle, "clients", "test-client.py"), "w") as f:
        f.write(client_src)
    cells = spec_extra.pop("cells")
    spec = {"id": "t", "owner": "t",
            "model": {"name": "m", "quant": "q",
                      "artifact": {"source": "gguf", "sha256": "0" * 64}},
            "workload": {"client": "x"},
            "client": {"script": "clients/test-client.py",
                       "args": ["python3", "{bundle}/clients/test-client.py",
                                "--url", "http://127.0.0.1:{port}",
                                "--json-out", "{cell}/client.json"]},
            "matrix": cells,
            "verdict_policy": {"retry": {"max": 1, "on": ["RETRYABLE_FAILURE"]},
                               "stop_review": ["UNKNOWN"]},
            "stop_policy": {"deadline_h": 0.1, "cell_max_s": 60}}
    spec.update(spec_extra)
    prof = _profile(tmp)
    prof["readiness"] = {"attempts": 20, "sleep": 0.1}
    prof["window"].update({"health_wait_s": 5})
    run_dir = os.path.join(tmp, "runs", "t")
    b = FixtureBackend(prof, bundle, run_dir, fault=fault)
    r = runner_mod.Runner(spec, prof, {}, run_dir, b, bundle, max_attempts=1)
    r.run()
    return b, r


def _client_recorder(declared=None, probe=None):
    import json as _j
    d = _j.dumps(declared or {})
    p = _j.dumps(probe or {})
    src = ('import argparse, json, urllib.request\n'
           'a = argparse.ArgumentParser()\n'
           'a.add_argument("--url"); a.add_argument("--json-out")\n'
           'a.add_argument("--segment", default=None); a.add_argument("--segments", default=None)\n'
           'a.add_argument("--dirty", action="store_true")\n'
           'a = a.parse_args()\n'
           'body = json.dumps({"prompt": "hi", "n_predict": 4}).encode()\n'
           'req = urllib.request.Request(a.url + "/completion", data=body,\n'
           '                             headers={"Content-Type": "application/json"})\n'
           'urllib.request.urlopen(req, timeout=30).read()\n'
           'res = {"valid": True, "rows": [{"decode_tps": 10.0, "decode_tokens": 4,\n'
           '        "wall_s": 0.5, "seg": a.segment}], "segment": a.segment}\n'
           + d + '.update(res)\n'
           + 'if ' + p + ':\n'
           + '    json.dump(' + p + ', open(a.json_out.replace("client.json", "probe.json"), "w"))\n'
           'json.dump(res, open(a.json_out, "w"))\n'
           'if a.segment is not None:\n'
           "    json.dump(res, open(a.json_out.replace('client.json', 'seg-' + str(a.segment) + '.json'), 'w'))\n"
           'print("ok")\n')
    return src


def q24_sidecar_lifecycle(tmp):
    # G1 (2026-09-13 LMCache): the window-level sidecar outlives per-cell
    # restarts; it dies exactly once — at the window's exit path — and the
    # watchdog fire path kills it too.
    cell = {"cell": "c1", "launch": {"binary": "/bin/true",
                                     "args": ["--port", "{port}"]},
            "requests": [{"kind": "completion", "prompt_tokens": 16,
                          "decode": 4, "reps": 1}],
            "gates": {"min_wall_s": 0.1}}
    b, r = _lmcache_fixture_campaign(
        tmp, {"cells": [cell],
              "sidecar": {"script": "/opt/cam/sidecar.sh",
                          "health_url": "http://127.0.0.1:1"}},
        _client_recorder())
    def idx(log, prefix):
        for i, e in enumerate(log):
            if e.startswith(prefix):
                return i
        return -1
    log = b.events()
    ok = (idx(log, "arm_watchdog") >= 0
          and idx(log, "arm_watchdog") < idx(log, "sidecar_start")
          and idx(log, "sidecar_stop") >= 0
          and idx(log, "sidecar_stop") < idx(log, "prod_start")
          and sum(1 for e in log if e.startswith("sidecar_stop")) == 1)
    # watchdog fire while the sidecar is alive (the 09-10 shape): simulated
    # after the first cell completes, with the sidecar still up.
    cell2 = dict(cell, cell="c2")
    b2 = None
    def mk():
        nonlocal b2
        return _mk2(tmp, [cell, cell2])
    b2, r2 = mk()
    orig = r2._run_cell
    def fire_after_first(c, idx):
        v = orig(c, idx)
        r2.cells_done[c["cell"]] = v
        if c["cell"] == "c1":
            b2.simulate_watchdog_fire()
        return v
    r2._run_cell = fire_after_first
    r2.run()
    l2 = b2.events()
    fired = any(e == "watchdog_fire" for e in l2)
    stop_after = fired and idx(l2, "sidecar_stop") > idx(l2, "watchdog_fire") \
        and idx(l2, "sidecar_stop") >= 0
    ok = ok and fired and stop_after
    return ok, (f"normal: armed<sidecar<exit-kill(x1) before prod_start; "
                f"fire-path: fired={fired} sidecar-killed-after-fire={stop_after}")


def q25_segments_relaunch(tmp):
    # the B-cell shape: segments: N => the cell's server is relaunched N
    # times (fresh state each time) while the sidecar persists; the client
    # runs once per segment with --segment/--segments.
    cell = {"cell": "c1", "segments": 3,
            "launch": {"binary": "/bin/true",
                       "args": ["--port", "{port}"]},
            "requests": [{"kind": "completion", "prompt_tokens": 16,
                          "decode": 4, "reps": 1}],
            "gates": {"min_wall_s": 0.1}}
    b, r = _lmcache_fixture_campaign(
        tmp, {"cells": [cell],
              "sidecar": {"script": "/opt/cam/sidecar.sh"}},
        _client_recorder())
    log = b.events()
    launches = [e.split(": ", 1)[1] for e in log if e.startswith("launch:")]
    kills = [e for e in log if e.startswith("kill pid=")]
    clients = [e for e in log if e.startswith("run_client")]
    ok = (len(launches) == 3 and len(kills) == 3
          and sum(1 for e in log if e.startswith("sidecar_start")) == 1
          and sum(1 for e in log if e.startswith("sidecar_stop")) == 1
          and len(clients) == 3 and len(set(launches)) == 3)
    seg_ok = 0
    for k in (1, 2, 3):
        sf = os.path.join(tmp, "runs", "t", "attempts", "c1-01",
                          f"seg-{k}.json")
        try:
            if json.load(open(sf)).get("segment") == str(k):
                seg_ok += 1
        except (OSError, ValueError):
            pass
    ok = ok and seg_ok == 3
    return ok, f"launches={len(launches)} kills={len(kills)} " \
               f"sidecar=1/1 distinct-scripts={len(set(launches))} " \
               f"segment-seen={seg_ok}/3"


def q26_zero_count_gate(tmp):
    # G3: the M phase gates on a ZERO condition over multiple cells.
    # all zero -> PASS; any non-zero -> VIOLATED (a result); a listed cell
    # without PASS evidence -> UNVERIFIABLE (absence of a result).
    from benchmarks import runner as runner_mod
    from benchmarks.backends.fixture import FixtureBackend
    spec = _spec([_cell("dep")])
    r = runner_mod.Runner(spec, _profile(tmp), {}, tmp,
                          FixtureBackend(_profile(tmp), ".", tmp), ".")
    _write_attempt(tmp, "b1", 1, "PASS", [])
    open(os.path.join(tmp, "attempts", "b1-01", "client.json"), "w").write(
        '{"restore_corruption_count": 0}')
    _write_attempt(tmp, "c1", 1, "PASS", [])
    open(os.path.join(tmp, "attempts", "c1-01", "client.json"), "w").write(
        '{"restore_corruption_count": 0}')
    eg = {"kind": "zero_count", "over": ["b1", "c1"],
          "metric": "restore_corruption_count"}
    s, d, u = r._effect_gate(eg)
    open(os.path.join(tmp, "attempts", "c1-01", "client.json"), "w").write(
        '{"restore_corruption_count": 2}')
    s2, d2, _ = r._effect_gate(eg)
    open(os.path.join(tmp, "attempts", "c1-01", "verdict.json"), "w").write(
        '{"class": "HARNESS_FAILURE"}')
    s3, d3, _ = r._effect_gate(eg)
    ok = (s == "PASS" and s2 == "VIOLATED" and s3 == "UNVERIFIABLE"
          and u.get("b1") == ["b1-01"] and "2" in d2)
    return ok, f"all-zero={s} non-zero={s2} no-evidence={s3} used={u}"


def q27_probe_isolation_declared(tmp):
    # G2/G4: (a) the S0 probe writes probe.json; the next cell's launch
    # consumes {probe_N}. (b) isolation_reset runs between cells. (c) a
    # client-declared documented-negative class is a completed negative;
    # the same class undeclared for the cell is review-required (never a
    # silent PASS, never a silent negative).
    s0 = {"cell": "s0", "launch": {"binary": "/bin/true",
                                   "args": ["--port", "{port}"]},
          "requests": [{"kind": "completion", "prompt_tokens": 16,
                        "decode": 4, "reps": 1}],
          "gates": {"min_wall_s": 0.1},
          "client": {"script": "clients/test-client.py",
                     "args": ["python3", "{bundle}/clients/test-client.py",
                              "--url", "http://127.0.0.1:{port}",
                              "--json-out", "{cell}/client.json"]}}
    c1 = {"cell": "c1",
          "launch": {"binary": "/bin/true",
                     "args": ["--port", "{port}", "--chunk", "{probe_N}"]},
          "requests": [{"kind": "completion", "prompt_tokens": 16,
                        "decode": 4, "reps": 1}],
          "gates": {"min_wall_s": 0.1}}
    b, r = _lmcache_fixture_campaign(
        tmp, {"cells": [s0, c1],
              "isolation_reset": "echo reset-sentinel"},
        _client_recorder(probe={"N": 784, "pool_tokens": 776928}))
    launch2 = open(os.path.join(tmp, "runs", "t", "attempts", "c1-01",
                                "launch.sh")).read()
    ok_a = "--chunk" in launch2 and "784" in launch2
    ok_b = any("reset-sentinel" in e for e in b.events())
    from benchmarks import verdict as v
    cls, _ = v.decide({"expects": {"documented_negative": ["LMCACHE_NO_HIT"]}},
                      None, [(True, "g", "ok")],
                      {"valid": True, "rows": [],
                       "declared_class": "LMCACHE_NO_HIT"}, 10)
    cls2, _ = v.decide({"expects": {}}, None, [(True, "g", "ok")],
                       {"valid": True, "rows": [],
                        "declared_class": "LMCACHE_NO_HIT"}, 10)
    ok_c = (cls == v.EXPECTED_NEGATIVE and cls2 == v.UNKNOWN)
    # a CLASSIFIER-sourced class with client data still maps through the
    # table (Q11 behavior must survive the declared-class change)
    cls3, _ = v.decide({"expects": {}}, {"class": "SSE_MALFORMED"},
                       [(True, "g", "ok")], {"valid": True, "rows": []}, 10)
    ok = ok_a and ok_b and ok_c and cls3 == v.HARNESS_FAILURE
    return ok, (f"probe={ok_a} isolation={ok_b} "
                f"declared-documented={cls} declared-undeclared={cls2} "
                f"classifier-sourced={cls3}")

def q28_pre_gate(tmp):
    # M-C runs and leaves a NON-ZERO corruption count in its PASS client.json;
    # M-B (pre_gate zero_count over M-C) must be SKIPPED (recorded) and the
    # campaign final must be review-required — not expected-negative, not a
    # silent gap.
    mc = {"cell": "mc", "launch": {"binary": "/bin/true",
                                   "args": ["--port", "{port}"]},
          "requests": [{"kind": "completion", "prompt_tokens": 16,
                        "decode": 4, "reps": 1}],
          "gates": {"min_wall_s": 0.1},
          "client": {"script": "clients/test-client.py",
                     "args": ["python3", "{bundle}/clients/test-client.py",
                              "--url", "http://127.0.0.1:{port}",
                              "--json-out", "{cell}/client.json",
                              "--dirty"]}}
    mb = {"cell": "mb",
          "launch": {"binary": "/bin/true", "args": ["--port", "{port}"]},
          "requests": [{"kind": "completion", "prompt_tokens": 16,
                        "decode": 4, "reps": 1}],
          "gates": {"min_wall_s": 0.1},
          "pre_gate": {"kind": "zero_count", "over": ["mc"],
                       "metric": "restore_corruption_count"}}
    src = (_client_recorder()
           + 'import sys\n'
           + 'if "--dirty" in sys.argv:\n'
           + '    d = json.load(open(a.json_out))\n'
           + "    d['restore_corruption_count'] = 1\n"
           + "    json.dump(d, open(a.json_out, 'w'))\n")
    b, r = _lmcache_fixture_campaign(tmp, {"cells": [mc, mb]}, src)
    st = json.load(open(os.path.join(tmp, "runs", "t", "cell-state.json")))
    fs = open(os.path.join(tmp, "runs", "t", "final.json")).read()
    ok = (st.get("mb", {}).get("status") == "SKIPPED"
          and st["mb"]["gate"]["status"] == "VIOLATED"
          and '"review-required"' in fs)
    return ok, (f"skip={st.get('mb', {}).get('status')} "
                f"gate={st.get('mb', {}).get('gate', {}).get('status')} "
                f"final={fs.strip()[:80]}")


p0("Q1  clean cell, real client over real sockets", q1_clean)
p0("Q2  documented wall -> EXPECTED_NEGATIVE, stop-done (09-12 gate)", q2_documented_wall)
p0("Q3  MiB OOM below wall -> RESOURCE_LIMIT (09-12 unit-blind matcher)", q3_mib_oom)
p0("Q4  meta-assert outranks the OOM it triggers (09-12 B1 / 09-13)", q4_meta_assert)
p0("Q5  empty-200 /health: readiness by code, not body (09-02/09-10)", q5_empty_200_health)
p0("Q6  hard crash -> watchdog restore -> resume (09-10 crash era)", q6_crash_watchdog_resume)
p0("Q7  restore idempotent across double exit (09-10 non-converging exit)", q7_restore_idempotent)
p0("Q8  host reboot mid-campaign: stop, forensics, no auto-resume (09-10)", q8_host_reboot)
p0("Q9  PVE dialect: only verified syntax renders (09-12 maint-fn)", q9_pve_dialect)
p0("Q10 false-complete gate: tiny/fast output is INVALID (09-10 LADDER-256k)", q10_false_complete)
p1("Q11 malformed SSE -> HARNESS_FAILURE (09-02 client bugs)", q11_malformed_sse)
p1("Q12 unsupported-arch control cell -> EXPECTED_NEGATIVE (09-12 B0)", q12_arch_control)
p1("Q13 no-usage stream shape -> SSE_NO_USAGE (09-10 usage lesson)", q13_no_usage)
p1("Q14 historical failure corpus replay", q14_corpus_replay)

def q15_slow_start(tmp):
    r, b, run_dir, fz, spec, prof = _ctx(tmp, "slow-start")
    prof["readiness"] = {"attempts": 20, "sleep": 0.2}
    fin = r.run()
    ok = fin["cells"].get("c1") == "PASS"
    return ok, f"cell={fin['cells'].get('c1')} final={fin['final']} (early 503 polls must not fail readiness)"
p1("Q15 slow-start health (2026-09-13 dogfood: all() readiness bug)", q15_slow_start)

def q16_cmd_with_json_braces(tmp):
    # the 2026-09-13 dogfood: str.format() ate the JSON braces of a
    # live-check command (KeyError: 'model') and aborted the restore loop.
    from benchmarks.backends.real import RealBackend
    prof = {"host": {"ssh": "x"}, "target": {"ssh": "x"},
            "prod": {"live_check_cmd": "curl -d '{\"model\":\"x\"}' && echo LIVE_OK"}}
    b = RealBackend(prof, "", "/tmp")
    got = b._prod("live_check_cmd")
    ok = got == prof["prod"]["live_check_cmd"]
    return ok, f"_prod passthrough: {got[:60]!r} (braces must survive unformatted)"
p1("Q16 profile commands with JSON braces survive _prod (2026-09-13 dogfood)", q16_cmd_with_json_braces)

def q17_arm_time_live_check_refusal(tmp):
    # the 2026-09-13 dogfood drill: a subtly-broken live_check_cmd made the
    # watchdog spin silently for 30 min at fire time (false ATTENTION).
    # The window must verify its sensor at arm time — with prod still healthy
    # — and refuse to open, disarming the watchdog.
    r, b, run_dir, fz, spec, prof = _ctx(tmp, "live-check-broken")
    b.prod_live = False          # the sensor is broken
    fin = r.run()
    ok = ("prod_stop" not in b.op_log and
          "disarm_watchdog" in b.op_log and
          fin["final"] == "aborted-failure")
    return ok, f"final={fin['final']} prod_stopped={'prod_stop' in b.op_log} disarmed={'disarm_watchdog' in b.op_log} (refuse with prod still up)"
p1("Q17 arm-time live-check verification refuses a broken sensor (2026-09-13 dogfood drill)", q17_arm_time_live_check_refusal)

def q18_kernel_scan_scoped_to_window(tmp):
    # the 2026-09-13 VM dogfood: an UNSCOPED kernel scan matched an
    # 8-week-old 'pcieport: AER: enabled' boot line and killed a healthy
    # window (the raw dmesg tail: no timestamps, full ring buffer). With a
    # t0 the probe must be journalctl-scoped to the window; the historical
    # form is only for pre-t0 callers. (The semantic proof — old AER lines
    # not firing live — is the VM window itself.)
    from benchmarks import host_failure
    scoped = host_failure.probe_cmd(since_epoch=1700000000)
    hist = host_failure.probe_cmd()
    ok = ("--since @1700000000" in scoped and "dmesg" not in scoped
          and "dmesg" in hist and host_failure.PROBE == hist)
    return ok, ("scoped since=%s dmesg-in-scoped=%s hist-dmesg=%s"
                % ("--since" in scoped, "dmesg" in scoped, "dmesg" in hist))
p0("Q18 kernel-log scan is window-scoped when a t0 exists (2026-09-13 VM dogfood)", q18_kernel_scan_scoped_to_window)

def q19_temp_change_undone_not_reapplied(tmp):
    # the 2026-09-13 VM dogfood: exit "undid" a temp change by re-running its
    # CMD — the host was left modified after every window. The undo must run
    # the entry's UNDO command, in reverse order.
    r, b, run_dir, fz, spec, prof = _ctx(tmp, "ok")
    prof["temp_changes"] = [{"cmd": "bump-to-9216", "undo": "restore-to-8192"}]
    fin = r.run()
    undo_events = [e for e in b.op_log if e.startswith("undo_temp_changes")]
    applied = [e for e in b.op_log if e.startswith("apply_temp_changes")]
    ok = (fin["final"] == "completed"
          and (fin.get("restore") or {}).get("restored") is True) and undo_events and \
         "restore-to-8192" in undo_events[0] and "bump-to-9216" not in undo_events[0]
    return ok, f"applied={applied[:1]} undo={undo_events[:1]} final={fin['final']}"
p1("Q19 temp changes are undone (not re-applied) at exit (2026-09-13 VM dogfood)", q19_temp_change_undone_not_reapplied)

def q20_temp_change_reQUIRES_undo(tmp):
    # external review blocker on 5b78803: the old undo path fell back to the
    # forward cmd when "undo" was missing — the exact bug it was fixing.
    # Invariant: every temp_change must have an explicit undo; prepare
    # (via check_temp_changes — the function prepare calls) rejects
    # cmd-without-undo, so the unsafe state is prepare-invalid and the host
    # is untouched (prepare never opens a window).
    from benchmarks import dialects
    prof_ok = {"host": {"pve_dialect": "pve9", "pve_guest": "vm"},
               "temp_changes": [{"cmd": "qm set 135 --memory 9216",
                                 "undo": "qm set 135 --memory 8192"}]}
    prof_bad = {"host": {"pve_dialect": "pve9", "pve_guest": "vm"},
                "temp_changes": [{"cmd": "qm set 135 --memory 9216"}]}
    ok = (dialects.check_temp_changes(prof_ok) == []
          and any("without undo" in p for p in dialects.check_temp_changes(prof_bad))
          and dialects.check_temp_changes({"temp_changes": []}) == [])
    return ok, "cmd+undo accepted; cmd-without-undo rejected; empty list clean"

def _write_attempt(base, cell, rep, verdict_class, client_rows, malformed=False):
    d = os.path.join(base, "attempts", f"{cell}-{rep:02d}")
    os.makedirs(d, exist_ok=True)
    if malformed:
        open(os.path.join(d, "client.json"), "w").write("{ not json")
    else:
        json.dump({"rows": client_rows}, open(os.path.join(d, "client.json"), "w"))
    json.dump({"class": verdict_class, "reason": "q-fixture"},
              open(os.path.join(d, "verdict.json"), "w"))
    return d

def q21_effect_gate_three_outcomes(tmp):
    # external review (B): PASS / BELOW_FLOOR / UNVERIFIABLE are DIFFERENT
    # verdicts. "cache bought only +2%" (a result) != "we could not measure
    # the cache gain" (absence of a result) — the runner maps them to
    # expected-negative vs review-required respectively.
    from benchmarks import runner as runner_mod
    from benchmarks.backends.fixture import FixtureBackend
    spec = _spec([_cell("dep")])
    r = runner_mod.Runner(spec, _profile(tmp), {}, tmp, FixtureBackend(_profile(tmp), ".", tmp), ".")
    _write_attempt(tmp, "base", 1, "PASS", [{"decode_tps": 100.0}, {"decode_tps": 102.0}])
    _write_attempt(tmp, "treat", 1, "PASS", [{"decode_tps": 150.0}, {"decode_tps": 148.0}])
    eg = {"base": "base", "treat": "treat", "metric": "decode_tps", "min_relative_gain_pct": 30}
    s_above, d1, u1 = r._effect_gate(eg)
    s_below, d2, _ = r._effect_gate(dict(eg, min_relative_gain_pct=60))
    s_unver, d3, u3 = r._effect_gate({"base": "ghost", "treat": "treat",
                                      "metric": "decode_tps", "min_relative_gain_pct": 1})
    # metric missing from a PASS attempt, and malformed client evidence
    _write_attempt(tmp, "m1", 1, "PASS", [{"other": 1}])
    _write_attempt(tmp, "treat", 2, "PASS", [])
    s_miss, d4, _ = r._effect_gate({"base": "m1", "treat": "treat",
                                    "metric": "decode_tps", "min_relative_gain_pct": 1})
    _write_attempt(tmp, "bad", 1, "PASS", None, malformed=True)
    s_mal, d5, _ = r._effect_gate({"base": "bad", "treat": "treat",
                                   "metric": "decode_tps", "min_relative_gain_pct": 1})
    ok = (s_above == "PASS" and s_below == "BELOW_FLOOR" and s_unver == "UNVERIFIABLE"
          and s_miss == "UNVERIFIABLE" and s_mal == "UNVERIFIABLE"
          and u1.get("base") == ["base-01"] and u1.get("treat") == ["treat-01"])
    return ok, (f"above={s_above} below={s_below} no-evidence={s_unver} "
                f"metric-missing={s_miss} malformed={s_mal}")
def q22_effect_gate_pass_only_estimator(tmp):
    # external review (A), verbatim case: an INVALID 1 t/s attempt must not
    # pollute the median. base-01 INVALID (1 t/s), base-02 PASS (20 t/s),
    # treat-01 PASS (22 t/s), 5% floor -> the comparison is 20 vs 22 ->
    # gate PASSES; the invalid attempt stays persisted but excluded.
    from benchmarks import runner as runner_mod
    from benchmarks.backends.fixture import FixtureBackend
    spec = _spec([_cell("dep")])
    r = runner_mod.Runner(spec, _profile(tmp), {}, tmp, FixtureBackend(_profile(tmp), ".", tmp), ".")
    _write_attempt(tmp, "base", 1, "INVALID", [{"decode_tps": 1.0}])
    _write_attempt(tmp, "base", 2, "PASS", [{"decode_tps": 20.0}])
    _write_attempt(tmp, "treat", 1, "PASS", [{"decode_tps": 22.0}])
    status, detail, used = r._effect_gate(
        {"base": "base", "treat": "treat", "metric": "decode_tps",
         "min_relative_gain_pct": 5})
    # sanity: the naive all-attempts estimator would see median(base)=1.0
    # vs 22 -> +2100%? no — with the 1 t/s included the base median is 1.0
    # only if it sorts first; the point is the invalid attempt must not be
    # among the attempts used AT ALL.
    ok = (status == "PASS"
          and used == {"base": ["base-02"], "treat": ["treat-01"]}
          and "base-02" in detail and "base-01" not in detail
          and "20" in detail and "22" in detail)
    # the INVALID attempt is still on disk (evidence preserved)
    preserved = os.path.exists(os.path.join(tmp, "attempts", "base-01", "client.json"))
    ok = ok and preserved
    return ok, f"status={status} used={used} preserved={preserved}"

def q23_campaign_final_mixed_sets(tmp):
    # external review (C): HARNESS_FAILURE and INVALID must not collapse to
    # an INFO-level expected-negative just because another cell produced a
    # documented negative.
    from benchmarks import verdict as v
    ok = (v.campaign_final({"a": v.EXPECTED_NEGATIVE, "b": v.HARNESS_FAILURE})
          == "review-required"
          and v.campaign_final({"a": v.EXPECTED_NEGATIVE, "b": v.INVALID})
          == "review-required"
          and v.campaign_final({"a": v.EXPECTED_NEGATIVE, "b": v.UNKNOWN})
          == "review-required"
          and v.campaign_final({"a": v.PASS, "b": v.EXPECTED_NEGATIVE})
          == "completed"
          and v.campaign_final({"a": v.EXPECTED_NEGATIVE, "b": v.EXPECTED_NEGATIVE})
          == "expected-negative"
          and v.campaign_final({"a": v.SAFETY_ABORT, "b": v.HARNESS_FAILURE})
          == "safety-abort")
    return ok, "mixed sets resolve conservatively; pure-negative and completed unchanged"

p0("Q20 every temp_change requires an explicit undo; prepare rejects the rest (external-review blocker)", q20_temp_change_reQUIRES_undo)
p0("Q21 effect gate: PASS / BELOW_FLOOR / UNVERIFIABLE are distinct outcomes (external review B)", q21_effect_gate_three_outcomes)
p0("Q22 effect gate: the estimator uses ONLY PASS attempts (external review A)", q22_effect_gate_pass_only_estimator)
p0("Q23 campaign final: a bad cell can never be masked by another cell's negative (external review C)", q23_campaign_final_mixed_sets)
# Q29 test body (spliced into qualify.py by the maintainers) —
# window rearm after a target restart: bounded ssh-readiness wait,
# deadline preservation, and a clean refusal when the target never comes back.
def q29_rearm_wait(tmp):
    from benchmarks import runner as runner_mod
    from benchmarks.backends.fixture import FixtureBackend
    from benchmarks.window import Window
    bundle = os.path.join(tmp, "bundle")
    os.makedirs(os.path.join(bundle, "clients"), exist_ok=True)
    with open(os.path.join(bundle, "clients", "test-client.py"), "w") as f:
        f.write(_client_recorder())
    cell = {"cell": "c1", "launch": {"binary": "/bin/true",
                                     "args": ["--port", "{port}"]},
            "requests": [{"kind": "completion", "prompt_tokens": 16,
                          "decode": 4, "reps": 1}],
            "gates": {"min_wall_s": 0.1},
            "client": {"script": "clients/test-client.py",
                       "args": ["python3", "{bundle}/clients/test-client.py",
                                "--url", "http://127.0.0.1:{port}",
                                "--json-out", "{cell}/client.json"]}}
    spec = {"id": "t", "owner": "t",
            "model": {"name": "m", "quant": "q",
                      "artifact": {"source": "gguf", "sha256": "0" * 64}},
            "workload": {"client": "x"},
            "client": {"script": "clients/test-client.py",
                       "args": ["python3", "{bundle}/clients/test-client.py",
                                "--url", "http://127.0.0.1:{port}",
                                "--json-out", "{cell}/client.json"]},
            "matrix": [cell],
            "verdict_policy": {"retry": {"max": 1, "on": ["RETRYABLE_FAILURE"]},
                               "stop_review": ["UNKNOWN"]},
            "stop_policy": {"deadline_h": 0.1, "cell_max_s": 60}}

    def _build(ping_seq):
        prof = _profile(tmp)
        prof["readiness"] = {"attempts": 20, "sleep": 0.05}
        prof["window"].update({"health_wait_s": 5,
                               "rearm_wait_s": 5, "rearm_poll_s": 0.05,
                               "permanent_steps": [
                                   {"cmd": "pct stop 999",
                                    "reason": "test lxc restart",
                                    "rearm_watchdog": True}]})
        run_dir = os.path.join(tmp, "runs", "t")
        b = FixtureBackend(prof, bundle, run_dir)
        b._ping_sequence = list(ping_seq)
        r = runner_mod.Runner(spec, prof, {}, run_dir, b, bundle,
                              max_attempts=1)
        return b, r

    # (a) target comes back after two failed pings: rearm with SAME deadline
    b, r = _build([False, False, True, True, True])
    d0 = []
    orig_arm = b.arm_watchdog

    def spy(deadline_epoch, lease_path, heartbeat_path, undo_manifest,
            prod_desc):
        d0.append(deadline_epoch)
        orig_arm(deadline_epoch, lease_path, heartbeat_path, undo_manifest,
                 prod_desc)

    b.arm_watchdog = spy
    r.run()
    arms = [e for e in b.op_log if e in ("arm_watchdog", "rearm_watchdog")]
    assert arms.count("rearm_watchdog") == 1, f"expected exactly one rearm: {arms}"
    assert len(d0) >= 2 and len(set(d0)) == 1, \
        f"deadline not preserved across rearm: {d0}"
    assert b.watchdog["deadline"] == d0[0]
    fs = open(os.path.join(tmp, "runs", "t", "final.json")).read()
    assert "PASS" in fs, fs

    # (b) target never comes back: bounded wait then a clean refusal
    b2, r2 = _build([False] * 50)
    r2.profile["window"]["rearm_wait_s"] = 0.3
    r2.profile["window"]["rearm_poll_s"] = 0.05
    w = Window(r2.backend, r2.profile, r2.run_dir, deadline_h=0.1)
    raised = None
    try:
        w.enter()
    except RuntimeError as e:
        raised = e
    assert raised and "not ssh-reachable" in str(raised), \
        f"expected clean refusal, got: {raised!r}"
    return True, (f"rearm={arms.count('rearm_watchdog')} "
                  f"deadline-preserved={b.watchdog['deadline'] == d0[0]} "
                  f"refusal=clean")

p0("Q24 window sidecar: survives per-cell restarts; killed on exit and on watchdog fire (LMCache G1)", q24_sidecar_lifecycle)
p0("Q25 segments: per-segment server relaunch (fresh state), sidecar persists, client sees its segment (LMCache G1/B)", q25_segments_relaunch)
p0("Q26 zero-count effect gate: PASS / VIOLATED / UNVERIFIABLE with attempt provenance (LMCache G3)", q26_zero_count_gate)
p1("Q27 probe substitution + isolation reset + client-declared doc-negative vs review (LMCache G2/G4)", q27_probe_isolation_declared)
p1("Q28 pre-gate: conditional cell runs on clean evidence, recorded-SKIPPED on a dirty gate, final=review-required (LMCache M-B/M-D)", q28_pre_gate)
p0("Q29 window rearm after target restart: bounded ssh wait, same-deadline rearm, clean refusal (permanent-step LXC restart: rearm after target reboot)", q29_rearm_wait)



def run_all(out=None):
    results = {"P0": [], "P1": []}
    for tier, items in (("P0", P0), ("P1", P1)):
        tmp = tempfile.mkdtemp(prefix=f"qualify-{tier.lower()}-")
        for name, fn in items:
            t0 = time.time()
            try:
                ok, detail = fn(tmp)
            except Exception as e:
                import traceback
                ok, detail = False, f"scenario raised: {type(e).__name__}: {e}\n" \
                    + "".join(traceback.format_exception(type(e), e, e.__traceback__))[-800:]
            results[tier].append((name, ok, detail, round(time.time() - t0, 1)))
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({results[tier][-1][3]}s)  {detail}")
    shutil.rmtree(tmp, ignore_errors=True)
    p0_pass = sum(1 for _, ok, _, _ in results["P0"] if ok)
    p1_pass = sum(1 for _, ok, _, _ in results["P1"] if ok)
    print(f"\n== QUALIFICATION: P0 {p0_pass}/{len(P0)} (must be 10/10 to freeze), "
          f"P1 {p1_pass}/{len(P1)} (warn-only) ==")
    return p0_pass == len(P0), results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ok, results = run_all(a.out)
    if a.out:
        slim = {"P0": [{"name": n, "pass": o, "detail": d} for n, o, d, _ in results["P0"]],
                "P1": [{"name": n, "pass": o, "detail": d} for n, o, d, _ in results["P1"]]}
        with open(a.out, "w") as f:
            json.dump(slim, f, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
