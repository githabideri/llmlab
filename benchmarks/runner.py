"""runner.py — the campaign state machine (ONE implementation, local or remote).

Everything the six nights taught us lives here, in code rather than prose:

  * a cell is an ATTEMPT: attempts/<cell>-<n>/ with launch.sh (script-file,
    byte-exact — never inline `bash -c`), prompts (nonced), client JSON,
    server.log, telemetry, verdict.json. An attempt is never rewritten; a
    retry is a new directory (the `rewrite_previous_attempt` class is
    structurally impossible).
  * the watchdog is armed before the first destructive step; the ONE exit
    path (window.exit) runs on every terminal transition, idempotently
    (the 09-10 non-converging exit is gone).
  * resume: a cell whose latest attempt verdict is PASS/EXPECTED_NEGATIVE is
    skipped on re-launch; a stale lease or dead runner resumes from the
    manifest, never from memory.
  * a host failure stops the campaign AFTER recovery: forensics, restore,
    no auto-resume (the 09-10 crash era).
  * a documented negative (the spec says which classes are the expected
    answer) is a SUCCESS path — EXPECTED_NEGATIVE, campaign final
    `expected-negative`, never "needs human review" (the 09-12 failure).
  * the scientific hash is recomputed at the end: if a repair moved it, the
    run is invalid no matter what the numbers say.
"""
import hashlib
import json
import os
import random
import statistics
import threading
import time

from . import preflight, verdict as verdict_mod, evidence as evidence_mod
from .classifier import engine as classifier
from .host_failure import HostFailureDetector
from .window import Window

PROMPT_WORDS = ("The lighthouse keeper counted the crows that gathered on the "
                "eastern jetty each dusk, and noted their numbers in a ledger "
                "bound in salt-stiffened leather.").split()


def make_prompt(target_tokens, tag=""):
    nonce = f"nonce-{tag}{random.getrandbits(64):016x}"
    body, i = [], 0
    while len(body) < target_tokens:
        body.append(PROMPT_WORDS[i % len(PROMPT_WORDS)])
        i += 1
    return f"Reference marker: {nonce}. " + " ".join(body)


class _Events:
    """Append-only event trail (events.jsonl)."""

    def __init__(self, path):
        self.path = path
        self.seq = 0
        if os.path.exists(path):
            self.seq = sum(1 for _ in open(path))

    def emit(self, event, **kw):
        self.seq += 1
        rec = {"seq": self.seq, "event": event,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        rec.update(kw)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


class _Heartbeat(threading.Thread):
    def __init__(self, window, interval=30):
        super().__init__(daemon=True)
        self.window = window
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(self.interval):
            try:
                self.window.heartbeat_tick()
            except Exception:
                pass

    def stop(self):
        self._stop.set()


class Runner:
    def __init__(self, spec, profile, freeze, run_dir, backend, bundle_dir,
                 max_attempts=2):
        self.spec = spec
        self.profile = profile
        self.freeze = freeze
        self.run_dir = run_dir
        self.backend = backend
        self.bundle_dir = bundle_dir
        self.max_attempts = max_attempts
        self.events = _Events(os.path.join(run_dir, "events.jsonl"))
        self.cells_done = {}
        self.window = None
        self._hb = None
        self._port_base = int(profile.get("ports", {}).get("base", 18100))

    # ------------------------------------------------------------------ run
    def run(self):
        os.makedirs(self.run_dir, exist_ok=True)
        self.backend.ensure_run_dir()
        self._state_init()
        final = "aborted-failure"
        try:
            # (1) preflight: blocking drift = no window
            self.events.emit("RUN_STARTED", campaign_id=self.freeze.get("campaign_id"))
            ok, report = preflight.run(self.backend, self.profile, self.freeze, self.run_dir)
            self.events.emit("PREFLIGHT_" + ("PASSED" if ok else "DRIFT"),
                             blocking=0 if ok else sum(1 for r in report["rows"]
                                                       if r["blocking"] and not r["match"]))
            if not ok:
                self.events.emit("NO_WINDOW_OPENED", reason="blocking drift")
                self._finish("aborted-preflight", "preflight drift")
                return self._final()
            # (2) window: watchdog first, then prod stop
            hf = HostFailureDetector(self.backend)
            hf.record_t0()
            self.window = Window(self.backend, self.profile, self.run_dir,
                                 deadline_h=self.spec.get("stop_policy", {}).get("deadline_h", 2.0),
                                 temp_changes=self._temp_changes())
            self.window.enter()
            self.events.emit("WINDOW_ARMED")
            self._sidecar = None
            sc = self.spec.get("sidecar")
            if sc:
                # the window-level sidecar (2026-09-13 LMCache campaign): a
                # target process that outlives per-cell server restarts.
                # Launched once, after the window is armed; killed at exit;
                # its pid joins the watchdog's kill manifest.
                log_t = self.backend.target_path(os.path.join(self.run_dir, "sidecar.log"))
                self._sidecar = self.backend.start_sidecar(
                    sc["script"], log_t,
                    health_url=sc.get("health_url"))
                try:
                    with open(os.path.join(self.run_dir, "undo-manifest.json")) as f:
                        man = json.load(f)
                    man["sidecar_pid"] = self._sidecar.get("pid")
                    with open(os.path.join(self.run_dir, "undo-manifest.json"), "w") as f:
                        json.dump(man, f, indent=2)
                except OSError:
                    pass
                self.events.emit("SIDECAR_STARTED", pid=self._sidecar.get("pid"))
            self._hb = _Heartbeat(self.window)
            self._hb.start()
            # (3) cells
            for idx, cell in enumerate(self.spec["matrix"]):
                if self.cells_done.get(cell["cell"]):
                    self.events.emit("CELL_SKIPPED", cell=cell["cell"], reason="complete")
                    continue
                # isolation reset between cells (2026-09-13 LMCache review):
                # the sidecar's store persists across cells BY DESIGN inside a
                # cell (that is the property under test), but must not leak
                # scientific state between cells (B's store must not serve C).
                if idx > 0 and (self.spec.get("isolation_reset")):
                    self.events.emit("ISOLATION_RESET")
                    self.backend.sh(self.spec["isolation_reset"], where="target",
                                    timeout=300)
                # host-failure stop gate: before EVERY cell
                hres = hf.check()
                if hres.get("detected"):
                    hf.forensic_collect(self.run_dir)
                    self.events.emit("HOST_FAILURE", reasons=hres["reasons"],
                                     stopped_before=cell["cell"])
                    self._finish("stopped-host-failure", "host failure: "
                                                        + ",".join(hres["reasons"]))
                    return self._final()
                # stop conditions (disk headroom; a watchdog death is the watchdog's
                # own job — it fires, we do not double-drive it)
                disk = (self.profile.get("storage") or {}).get("results")
                if disk and self.backend.disk_free(disk) < \
                        (self.profile.get("storage") or {}).get("min_free_bytes", 1 << 30):
                    self.events.emit("STOP_CONDITION", reason="results disk low")
                    self._finish("safety-abort", "results disk below floor")
                    return self._final()

                # effect gate (2026-09-13 flashnext review): the ONE small
                # declarative dependency. The verdict engine is per-attempt
                # and cannot see across cells, so a cross-cell comparison
                # ("MTP is scientifically meaningless unless S1 shows a
                # useful effect") is evaluated HERE, before the dependent
                # cell runs, from the PERSISTED evidence of the compared
                # cells. Single kind, no branching: below the declared
                # floor the campaign stops (expected-negative) and the
                # dependent cell is recorded NOT_RUN — a reviewer reconstructs
                # why it never ran from this event + the two client.jsons.
                eg = cell.get("effect_gate") or {}
                if eg:
                    status, detail, used = self._effect_gate(eg)
                    if status != "PASS":
                        self.events.emit("EFFECT_GATE", cell=cell["cell"],
                                         status=status, detail=detail,
                                         attempts=used)
                        self.cells_done[cell["cell"]] = "NOT_RUN"
                        # BELOW_FLOOR is a RESULT (the declared effect is
                        # absent) -> expected-negative. UNVERIFIABLE is the
                        # ABSENCE of a result (we could not measure) ->
                        # review-required. "cache bought only +2%" != "we
                        # could not measure the cache gain".
                        # BELOW_FLOOR (measured gain under the floor) and
                        # VIOLATED (a zero-condition gate saw a non-zero
                        # count) are RESULTS; UNVERIFIABLE is the absence of
                        # a result.
                        final = ("expected-negative"
                                 if status in ("BELOW_FLOOR", "VIOLATED")
                                 else "review-required")
                        self._finish(final, f"effect gate on {cell['cell']}: {detail}")
                        return self._final()
                verdict = self._run_cell(cell, idx)
                self.cells_done[cell["cell"]] = verdict
                if verdict == verdict_mod.EXPECTED_NEGATIVE and \
                        cell.get("stop_on_expected_negative"):
                    # the documented wall: the campaign's question is answered.
                    # GATE — stop DONE. This is success, not review.
                    self.events.emit("GATE_EXPECTED_NEGATIVE", cell=cell["cell"])
                    self._finish("expected-negative", f"documented negative on {cell['cell']}")
                    return self._final()
        except (KeyboardInterrupt, SystemExit):
            self.events.emit("ABORTED", reason="signal")
            final = "aborted-signal"
            self._finish(final, "signal")
            return self._final()
        except Exception as e:
            if os.environ.get("RUNNER_TB"):
                import traceback
                traceback.print_exc()
            self.events.emit("ABORTED", reason=repr(e))
            final = "aborted-failure"
            # if prod is still up (window never entered) the watchdog is a stray
            # timer — disarm it; if prod is down, exit() owns the restore
            try:
                if self.window and not self.window.entered:
                    self.window.b.disarm_watchdog(self.window.lease)
                elif self.window and self.window.entered:
                    self.window.exit("aborted: " + repr(e)[:120])
            except Exception:
                pass
            self._finish(final, repr(e))
            return self._final()
        final = verdict_mod.campaign_final(self.cells_done)
        self._finish(final, "all cells done")
        return self._final()

    # --------------------------------------------------------------- cells
    def _record_cell_state(self, cell, state):
        """Persist a cell's final state without an attempt (skips)."""
        f = os.path.join(self.run_dir, "cell-state.json")
        st = {}
        if os.path.exists(f):
            try:
                st = json.load(open(f))
            except ValueError:
                st = {}
        st[cell["cell"]] = state
        json.dump(st, open(f, "w"), indent=2)

    def _run_cell(self, cell, idx):
        cid = cell["cell"]
        # pre_gate: this cell runs only if an earlier cell's PERSISTED
        # evidence satisfies a condition (2026-09-13 LMCache: M-B/M-D run
        # only after a clean M-C). VIOLATED / UNVERIFIABLE skips the cell
        # with a recorded, reviewable decision — never a silent omission,
        # never a run.
        pg = cell.get("pre_gate")
        if pg:
            gs, gd, gu = self._effect_gate(pg)
            if gs != "PASS":
                self._record_cell_state(cell, {
                    "status": "SKIPPED", "attempts": 0,
                    "gate": {"kind": pg.get("kind"), "status": gs,
                             "detail": gd, "used": gu},
                })
                self.events.emit("CELL_SKIPPED", cell=cid, gate=gs,
                                 detail=gd)
                return "SKIPPED"
        verdict = verdict_mod.UNKNOWN
        for attempt in range(1, self.max_attempts + 1):
            self.events.emit("ATTEMPT_STARTED", cell=cid, attempt=attempt)
            a = self._run_attempt(cell, idx, attempt)
            verdict = a["verdict"]["class"]
            if verdict in (verdict_mod.PASS, verdict_mod.EXPECTED_NEGATIVE,
                           verdict_mod.SAFETY_ABORT, verdict_mod.INVALID):
                break
            # RETRYABLE_*: one more attempt; HARNESS_FAILURE: do not retry the same
            # broken code — record the repair need and move on
            if verdict == verdict_mod.HARNESS_FAILURE:
                self.events.emit("REPAIR_NEEDED", cell=cid, attempt=attempt,
                                 note=a["verdict"].get("reason", ""))
                break
            self.events.emit("ATTEMPT_RETRY", cell=cid, attempt=attempt,
                             verdict=verdict, next=attempt + 1)
        self.events.emit("CELL_" + verdict.upper(), cell=cid, attempts=attempt)
        return verdict

    def _run_attempt(self, cell, idx, attempt):
        a_dir = os.path.join(self.run_dir, "attempts", f"{cell['cell']}-{attempt:02d}")
        os.makedirs(a_dir, exist_ok=True)
        # The attempt lives in TWO places: on the target (where the workload runs)
        # and locally (the evidence of record). The backend maps between them
        # (identity for the fixture, deploy-dir mirror for real).
        t_dir = self.backend.target_path(a_dir)
        port = self._port_base + idx
        cell_paths = {"cell": t_dir, "port": str(port),
                      "bundle": self.backend.bundle_path(),
                      "plan": str(cell.get("plan", ""))}
        server_log = os.path.join(a_dir, "server.log")
        t_server_log = self.backend.target_path(server_log)

        # (1) prompt (nonced — the cache trap), both places
        ptok = int(cell.get("requests", [{}])[0].get("prompt_tokens", 64))
        with open(os.path.join(a_dir, "prompt.txt"), "w") as f:
            f.write(make_prompt(ptok, f"{cell['cell']}{attempt}-"))
        self.backend.put_file(os.path.join(a_dir, "prompt.txt"),
                              self.backend.target_path(os.path.join(a_dir, "prompt.txt")))

        # (2) immutable launch spec as a SCRIPT FILE (byte-exact; no inline bash -c).
        # A cell may declare "segments": N — the platform then boots the cell's
        # server N times (kill + relaunch between segments) and runs the client
        # once per segment (--segment k --segments N). Each segment sees a
        # FRESH server (clean GPU/APC state by construction) while any
        # window-level sidecar keeps running across the relaunches — the
        # "vLLM restarts, the RAM store survives" experiment, expressed as a
        # platform property instead of client-side process fiddling.
        launch = cell["launch"]
        model_path = (self.profile.get("storage") or {}).get("model_path", "{model}")
        n_seg = int(cell.get("segments", 1))
        seg_logs = []
        client_data, fail_log = None, ""
        t0 = time.time()
        for seg in range(1, n_seg + 1):
            c_paths = dict(cell_paths, segment=str(seg), segments=str(n_seg),
                           **self._probe_paths())
            # probe keys consumed by the launch must EXIST: a launch that
            # silently drops its {probe_N} would size the cache with an
            # unverified chunk size. Missing = HARNESS_FAILURE, loudly.
            need = set()
            for a in launch["args"]:
                need |= set(__import__("re").findall(r"\{(\w+)\}", a))
            missing = [k for k in sorted(need)
                       if k.startswith("probe_") and k not in c_paths]
            if missing:
                fail_log = ("probe key missing: " + ",".join(missing)
                            + " (the probe cell must have written probe.json)")
                self._pull(os.path.join(a_dir, "server.log"))
                break
            args = [a.format(model=model_path, **c_paths) for a in launch["args"]]
            script = launch["binary"].format(model=model_path)
            if n_seg == 1:
                log_name, launch_name = "server.log", "launch.sh"
            else:
                log_name, launch_name = f"vllm-{seg}.log", f"launch-seg{seg}.sh"
            seg_logs.append(log_name)
            t_log = self.backend.target_path(os.path.join(a_dir, log_name))
            launch_sh = (f"#!/bin/sh\n# IMMUTABLE per-segment launch spec (generated; do not hand-edit)\n"
                         f"# cell={cell['cell']} attempt={attempt} segment={seg}/{n_seg}\n"
                         + " ".join(_q(a) for a in [script] + args)
                         + f" > {t_log} 2>&1" + "\n")
            self.backend.write_script(self.backend.target_path(os.path.join(a_dir, launch_name)),
                                      launch_sh)
            with open(os.path.join(a_dir, launch_name), "w") as f:
                f.write(launch_sh)      # local copy = evidence of what was sent
            with open(os.path.join(a_dir, "meta.json"), "w") as f:
                json.dump({"cell": cell["cell"], "attempt": attempt, "port": port,
                           "segment": seg, "segments": n_seg,
                           "launch": [script] + args,
                           "freeze": {"scientific_hash": self.freeze.get("scientific_hash"),
                                      "implementation_hash": self.freeze.get("implementation_hash")}},
                          f, indent=2)
            # (3) launch + readiness (the HTTP CODE is the gate — empty-200 contract)
            handle = None
            try:
                handle = self.backend.launch(self.backend.target_path(os.path.join(a_dir, launch_name)),
                                             t_log)
            except Exception as e:
                fail_log = "launch failed: " + repr(e)
            if handle is not None:
                ready = self.backend.wait_ready(
                    f"http://127.0.0.1:{port}/health",
                    attempts=int(self.profile.get("readiness", {}).get("attempts", 6)),
                    sleep=float(self.profile.get("readiness", {}).get("sleep", 0.3)))
                # 'became ready at some point' is the contract (all() would
                # demand every poll — including pre-load ones — to be 200)
                if any(ready):
                    # (4) telemetry, then the real client over real sockets
                    telem = self.backend.start_telemetry(t_dir,
                                                         self.spec.get("stop_policy", {}).get("cell_max_s", 600))
                    try:
                        client_data = self._run_client(cell, t_dir, port,
                                                       seg=seg, n_seg=n_seg)
                    except Exception as ce:
                        _write(os.path.join(a_dir, "client-error.json"),
                               {"error": repr(ce)})
                        fail_log = fail_log or self._pull(os.path.join(a_dir, log_name))
                    finally:
                        self.backend.stop_telemetry(telem)
                        self.backend.kill(handle)   # the sidecar (if any) survives
                else:
                    self.backend.kill(handle)
                    fail_log = self._pull(os.path.join(a_dir, log_name))
                    if not fail_log:
                        fail_log = "server never became ready (no load log)"
            # the probe file (written by a client, e.g. the S0 N-probe) is
            # evidence that later cells' launches may consume
            self._pull_probe(a_dir)
        for log_name in seg_logs:          # the server log(s) are evidence either way
            self._pull(os.path.join(a_dir, log_name))
        for f in ("nvml.csv", "client.json"):
            if not os.path.exists(os.path.join(a_dir, f)):
                self._pull(os.path.join(a_dir, f))
        wall = time.time() - t0
        _write(os.path.join(a_dir, "attempt.json"),
               {"wall_s": round(wall, 3), "client": _slim(client_data)})

        # (4.5) workload contract: did the declared workload demonstrably happen?
        # Evaluated BEFORE the plausibility gates; evidence sources per engine
        # come from evidence_policy.jsonc (accepted sets), so the gate never
        # assumes which channel an engine reports through.
        contract = None
        if self.spec.get("engine"):
            log_text = None
            for log_name in (["server.log"] + list(seg_logs)):
                p = os.path.join(a_dir, log_name)
                if os.path.exists(p):
                    chunk = open(p, errors="replace").read()
                    log_text = (log_text + "\n" if log_text else "") + chunk
            contract = evidence_mod.evaluate(cell, client_data, log_text,
                                             self.spec["engine"])
        # (5) classify (only when there is a failure log), then gates, then verdict
        cls = None
        if fail_log and fail_log != "server never became ready (no load log)":
            cls = classifier.classify(fail_log)
            _write(os.path.join(a_dir, "classification.json"), cls)
        gates = self._gates(cell, client_data, wall)
        v_class, reason = verdict_mod.decide(cell, cls, gates, client_data, wall,
                                             contract=contract)
        v = {"class": v_class, "reason": reason,
             "gates": {g[1]: g[0] for g in gates},
             "contract": contract,
             "observed": (cls or {}).get("observed", []),
             "inferred": (cls or {}).get("inferred", []),
             "not_established": (cls or {}).get("not_established", [])}
        _write(os.path.join(a_dir, "verdict.json"), v)
        return {"verdict": v}

    def _pull(self, local_path):
        # fetch a target file to the local evidence dir; returns its text
        try:
            t = self.backend.target_path(local_path)
            if t != local_path:
                self.backend.get_file(t, local_path)
        except OSError:
            pass
        return _read(local_path)

    def _run_client(self, cell, a_dir_target, port, seg=None, n_seg=None):
        # cell-level client OVERRIDES the spec-level one (S0 probe cell,
        # M-battery cells) — a per-cell contract is the finer one
        c = cell.get("client") or self.spec.get("client")
        argv = [a.format(cell=a_dir_target, port=port,
                         bundle=self.backend.bundle_path(),
                         plan=str(cell.get("plan", ""))) for a in c["args"]]
        # segment args are appended ONLY for multi-segment cells: existing
        # clients (bench-llama and friends) do not accept them and a single-
        # segment cell must see exactly the old argv shape
        if seg is not None and n_seg and int(n_seg) > 1:
            argv += ["--segment", str(seg), "--segments", str(n_seg)]
        return self.backend.run_client(argv)

    def _probe_paths(self):
        """{probe_<key>: value} substitutions for launch args, from the most
        recent probe.json a client wrote (the S0 probe: the engine's reported
        hybrid block size N, pool size, ...). N is a BOOT MECHANIC, not a
        science field — but the value is persisted as evidence per cell."""
        p = os.path.join(self.run_dir, "probe.json")
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            return {}
        return {f"probe_{k}": str(v) for k, v in d.items()}

    def _pull_probe(self, a_dir):
        local = os.path.join(self.run_dir, "probe.json")
        try:
            self.backend.get_file(os.path.join(a_dir, "probe.json"), local)
        except OSError:
            pass

    def _gates(self, cell, client_data, wall):
        """Plausibility gates, SIZED per cell (unsized gates are how false-completes
        pass — the LADDER-256k lesson). All return (ok, label, detail)."""
        g = cell.get("gates") or {}
        gates = []
        rows = (client_data or {}).get("rows") or (client_data or {}).get("runs") or []
        tps_key = ("decode_tps" if rows and "decode_tps" in (rows[0] if rows else {})
                   else "tg_s")
        tps = [r.get(tps_key) for r in rows if r.get(tps_key) is not None]
        toks = [r.get("decode_tokens") or r.get("completion_tokens") or 0 for r in rows]
        walls = [r.get("wall_s") or r.get("e2e_s") or 0 for r in rows]
        if "min_wall_s" in g:
            gates.append((min(walls or [wall]) >= g["min_wall_s"],
                          "min_wall_s", f"min wall {min(walls or [wall])}s < {g['min_wall_s']}s"))
        if "max_wall_s" in g:
            gates.append((max(walls or [wall]) <= g["max_wall_s"],
                          "max_wall_s", f"wall {max(walls or [wall])}s > {g['max_wall_s']}s"))
        if "min_tokens" in g:
            ok = bool(toks) and min(toks) >= g["min_tokens"]
            gates.append((ok, "min_tokens", f"min tokens {min(toks) if toks else 0} < {g['min_tokens']}"))
        if "min_tps" in g:
            med = statistics.median(tps) if tps else 0
            gates.append((med >= g["min_tps"], "min_tps",
                          f"median {tps_key} {med} < {g['min_tps']}"))
        if g.get("require_determinism"):
            det = (client_data or {}).get("determinism", "OK")
            gates.append((det in ("OK", "SINGLE"), "determinism", f"sha256 determinism={det}"))
        if g.get("require_overlap"):
            ok = (client_data or {}).get("overlap_valid", False)
            gates.append((ok, "overlap_proof",
                          f"overlap_valid={ok} (min {g.get('overlap_s_min', '?')}s per pair)"))
        return gates

    def _temp_changes(self):
        return self.profile.get("temp_changes") or []

    def _effect_gate(self, eg):
        """Evaluate a declared cross-cell comparison from persisted evidence.
        eg: {base: <cell>, treat: <cell>, metric: <client json key>,
            min_relative_gain_pct: <float>}.

        Returns (status, detail, used) with status in:
          PASS         treat >= base*(1+floor) — the dependent cell runs
          BELOW_FLOOR  measured gain below the declared floor — a scientific /
                       engineering RESULT (expected-negative; the campaign
                       answers its own question)
          UNVERIFIABLE the premise could not be measured (no PASS evidence,
                       missing/malformed metric, base == 0) — NOT a result:
                       the campaign goes to review-required

        Estimator discipline (2026-09-13 external review): only attempts
        whose verdict.json says class == PASS feed the estimator. A failed /
        invalid attempt (a 1 t/s HARNESS_FAILURE) is preserved as evidence
        but must not drag a median. The detail records which attempt IDs
        were used, so a reviewer can recompute the comparison verbatim.
        """
        import glob
        import json as _json
        import statistics
        if eg.get("kind") == "zero_count":
            # 2026-09-13 LMCache review: the M phase gates on a ZERO condition
            # (no restoration corruption across the target-only battery) —
            # "is it safe" is a different question than "does it help", so it
            # is a different gate. PASS: every listed cell's PASS evidence
            # shows a zero count. VIOLATED: any non-zero count (a RESULT: the
            # phase is blocked and the campaign completes with that answer).
            # UNVERIFIABLE: any listed cell has no PASS evidence carrying the
            # metric (absence of a result -> review).
            def cell_count(name, metric):
                files = sorted(glob.glob(os.path.join(
                    self.run_dir, "attempts", f"{name}-*", "client.json")))
                vals = []
                for f in files:
                    a_dir = os.path.dirname(f)
                    try:
                        vd = _json.load(open(os.path.join(a_dir, "verdict.json")))
                    except (OSError, ValueError):
                        continue
                    if vd.get("class") != "PASS":
                        continue
                    try:
                        c = _json.load(open(f))
                    except (OSError, ValueError):
                        continue
                    v = c.get(metric)
                    if v is not None:
                        vals.append((os.path.basename(a_dir), v))
                return vals
            all_used, bad = {}, False
            counts = {}
            for name in eg["over"]:
                vv = cell_count(name, eg["metric"])
                all_used[name] = [a for a, _ in vv]
                if not vv:
                    return ("UNVERIFIABLE",
                            f"zero-count gate: {eg['metric']} has no PASS "
                            f"evidence from cell {name}", all_used)
                counts[name] = max(v for _, v in vv)
                bad = bad or max(v for _, v in vv) > 0
            detail = f"zero-count gate {eg['metric']}: " + \
                ", ".join(f"{n}={v}" for n, v in counts.items())
            if bad:
                return "VIOLATED", detail + " — non-zero count; dependent phase blocked", all_used
            return "PASS", detail + " — clean", all_used

        def cell_metric(name, metric):
            files = sorted(glob.glob(
                os.path.join(self.run_dir, "attempts", f"{name}-*", "client.json")))
            vals, used = [], []
            for f in files:
                a_dir = os.path.dirname(f)
                a_id = os.path.basename(a_dir)
                # verdict gate: only PASS attempts are scientific evidence
                try:
                    vd = _json.load(open(os.path.join(a_dir, "verdict.json")))
                except (OSError, ValueError):
                    continue
                if vd.get("class") != "PASS":
                    continue
                try:
                    c = _json.load(open(f))
                except (OSError, ValueError):
                    continue  # malformed evidence is not evidence
                rows = c.get("rows") or []
                mv = [r[metric] for r in rows if isinstance(r, dict)
                      and r.get(metric) is not None]
                if mv:  # median of the reps, then median across attempts
                    vals.append(statistics.median(mv))
                    used.append(a_id)
            if not vals:
                return None, []
            return statistics.median(vals), used
        base_v, base_used = cell_metric(eg["base"], eg["metric"])
        treat_v, treat_used = cell_metric(eg["treat"], eg["metric"])
        detail = (f"{eg['metric']}: base={base_v if base_v is not None else 'unverifiable'}"
                  f" ({len(base_used)} PASS attempt(s): {base_used}) "
                  f"treat={treat_v if treat_v is not None else 'unverifiable'}"
                  f" ({len(treat_used)} PASS attempt(s): {treat_used})")
        if base_v is None or treat_v is None or base_v == 0:
            return ("UNVERIFIABLE", detail +
                    " — no valid comparison exists; the dependent cell was NOT_RUN "
                    "and the campaign is review-required",
                    {eg["base"]: base_used, eg["treat"]: treat_used})
        gain_pct = 100.0 * (treat_v - base_v) / base_v
        floor = float(eg["min_relative_gain_pct"])
        detail = f"{eg['metric']}: base={base_v:.4g} treat={treat_v:.4g} " \
                 f"gain={gain_pct:+.1f}% (floor {floor:g}%) | " + detail
        used = {eg["base"]: base_used, eg["treat"]: treat_used}
        if gain_pct >= floor:
            return "PASS", detail, used
        return ("BELOW_FLOOR", detail +
                " — the declared effect is not there; the dependent cell was NOT_RUN "
                "(expected-negative)", used)

    # -------------------------------------------------------------- finish
    def _finish(self, final, reason):
        if self._hb:
            self._hb.stop()
        # the sidecar is campaign-owned: it dies with the campaign, before the
        # prod restore (the watchdog's kill manifest carries its pid too, so
        # the fire path kills it as well)
        if getattr(self, "_sidecar", None):
            try:
                self.backend.stop_sidecar(self._sidecar)
                self.events.emit("SIDECAR_STOPPED",
                                 pid=self._sidecar.get("pid"))
            except Exception as e:
                self.events.emit("SIDECAR_STOP_FAILED", detail=repr(e)[:160])
            self._sidecar = None
        restore = {}
        if self.window is not None:
            restore = self.window.exit(reason)
        self._final_json(final, reason, restore)
        self.events.emit("RUN_ENDED", final=final,
                         restored=restore.get("restored", None))
        try:
            from . import verdict as vmod
            level = vmod.NOTIFY_LEVELS.get(final, "ATTENTION")
            from . import notify as notify_mod
            self.backend.notify(level, notify_mod.message_for(
                final, self.cells_done, restore), self.freeze.get("campaign_id"))
        except Exception:
            pass

    def _final_json(self, final, reason, restore):
        # the scientific hash must not have moved (repair-lane invariant)
        from . import spec as specmod
        sci_now = specmod.scientific_hash(self.spec)
        sci_moved = sci_now != self.freeze.get("scientific_hash")
        obj = {
            "campaign_id": self.freeze.get("campaign_id"),
            "final": final,
            "reason": reason,
            "cells": self.cells_done,
            "restore": restore,
            "scientific_hash_frozen": self.freeze.get("scientific_hash"),
            "scientific_hash_now": sci_now,
            "scientific_hash_moved": sci_moved,
            "valid": (final in ("completed", "expected-negative")) and not sci_moved,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(os.path.join(self.run_dir, "final.json"), "w") as f:
            json.dump(obj, f, indent=2)
        return obj

    def _final(self):
        p = os.path.join(self.run_dir, "final.json")
        return json.load(open(p)) if os.path.exists(p) else {"final": "aborted-failure"}

    def _state_init(self):
        p = os.path.join(self.run_dir, "state.json")
        if os.path.exists(p):
            st = json.load(open(p))
            self.cells_done = st.get("cells", {})
        else:
            with open(p, "w") as f:
                json.dump({"campaign_id": self.freeze.get("campaign_id")}, f)

    def _state_save(self):
        with open(os.path.join(self.run_dir, "state.json"), "w") as f:
            json.dump({"campaign_id": self.freeze.get("campaign_id"),
                       "cells": self.cells_done}, f)


def _q(s):
    import shlex
    return shlex.quote(s)


def _read(path):
    try:
        return open(path, errors="replace").read()
    except OSError:
        return ""


def _slim(obj):
    if not isinstance(obj, dict):
        return obj
    keep = {k: v for k, v in obj.items()
            if k not in ("text", "text_parts", "itl_ms")}
    return keep


def _write(path, obj):
    with open(path, "w") as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, indent=2)
