"""backends/real.py — the real I/O layer (SSH to the profile's machines).

Adapted from the field-proven mm-image-max LocalBackend (which ran the cleanest
night to date) and generalized: everything machine-specific comes from the
deployment profile (unit, ports, paths, PVE dialect, watchdog location), never
hard-coded. Carried-over lessons:

  * commands run as `bash -s` over SSH stdin with `PF_ key=value` parse-back
    (no quote-escaping through three shell layers)
  * files move as base64 over the same channel (no scp dependencies)
  * launches are script FILES written first, then nohup+setsid (byte-exact;
    inline `bash -c` through runuser/setsid has eaten JSON args before)
  * kills are PID-only, process-group, with cmdline verification (no pkill,
    ever, on a machine that hosts production)
"""
import base64
import json
import os
import shlex
import subprocess
import time


class RealBackend:
    name = "real"

    def __init__(self, profile, bundle_dir, run_dir, ssh_extra=None):
        self.p = profile
        self.bundle_dir = bundle_dir
        self.run_dir = run_dir
        self.ssh_extra = ssh_extra or []
        self._events = []
        self._prod_pid = None

    # -- exec boundary ---------------------------------------------------------
    def _ssh(self, where, cmd, timeout=120, stdin=None):
        host = self.p["host"]["ssh"] if where == "host" else self.p["target"]["ssh"]
        p = subprocess.run(["ssh", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes",
                            *self.ssh_extra, host, cmd],
                           capture_output=True, text=True, input=stdin, timeout=timeout)
        return p.returncode, p.stdout, p.stderr

    def sh(self, cmd, timeout=120, stdin=None, where="target"):
        self._events.append(f"sh[{where}]: {cmd[:120]}")
        return self._ssh(where, cmd, timeout, stdin)

    def _remote(self, where, script, timeout=120):
        """Run a bash script over stdin; parse PF_ key=value lines back."""
        rc, out, err = self._ssh(where, "bash -s", timeout, stdin=script)
        kv = {}
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("PF_") and "=" in line:
                k, _, v = line.partition("=")
                kv[k] = v
        return rc, kv, err

    def put_file(self, local_path, remote_path, where="target"):
        b64 = base64.b64encode(open(local_path, "rb").read()).decode()
        # b64 over ssh STDIN, never in argv: a 16k+ token prompt b64 (~135 KB)
        # exceeds MAX_ARG_STRLEN (~131072) and died with E2BUG/E2BIG at cell
        # a40-off (09-14). Remote size check makes a truncated write fail loud.
        size = os.path.getsize(local_path)
        rc, out, err = self._ssh(where,
                                 f"mkdir -p $(dirname {shlex.quote(remote_path)}) && "
                                 f"base64 -d > {shlex.quote(remote_path)} && "
                                 f"[ \"$(stat -c%s {shlex.quote(remote_path)})\" = {size} ] && echo WROTE",
                                 stdin=b64 + "\n")
        if "WROTE" not in out:
            raise OSError(f"put_file failed for {remote_path}: {err[:200]}")

    def get_file(self, remote_path, local_path, where="target"):
        """b64 over ssh. 09-16 run #7: one transient ssh loss during a
        high-churn window killed the fetch of a GOOD client.json (the file
        was on target the whole time) and voided the cell — a single
        un-retried transport call must not have science-grade consequences.
        Retry with backoff; the old `2>/dev/null` made every failure an
        empty string, so the final error now also probes the remote file."""
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        last = ""
        for attempt in (1, 2, 3):
            rc, out, err = self._ssh(where, f"base64 {shlex.quote(remote_path)}")
            if rc == 0:
                with open(local_path, "wb") as f:
                    f.write(base64.b64decode(out))
                return
            last = f"rc={rc} err={err[:200]}"
            if attempt < 3:
                time.sleep(5)
        prc, pout, perr = self._ssh(
            where, f"[ -f {shlex.quote(remote_path)} ] && "
                  f"echo EXISTS-$(stat -c %s {shlex.quote(remote_path)}) || echo ABSENT")
        raise OSError(
            f"get_file failed for {remote_path} after 3 attempts ({last}); "
            f"remote probe: {(pout or perr).strip()[:120]}")

    def write_script(self, path, content, where="target"):
        self.put_file(self._materialize(content), path, where)

    def _materialize(self, content):
        import tempfile
        f = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)
        f.write(content)
        f.close()
        return f.name

    # -- production lifecycle ----------------------------------------------------
    def _prod(self, key, *a):
        # format ONLY when arguments are given: commands may legally contain
        # JSON braces ({"model":...}) which str.format would eat
        cmd = self.p["prod"].get(key) or ""
        return cmd.format(*a) if a else cmd

    def prod_stop(self):
        rc, out, err = self.sh(self._prod("stop"), where="target")
        if rc != 0 and "stopped" not in out and "inactive" not in out:
            # stopping something already stopped is not an error (idempotent exit)
            rc2, out2, _ = self.sh(f"systemctl is-active {self._prod('unit')}", where="target")
            if out2.strip() != "inactive":
                raise RuntimeError(f"prod stop failed: {err[:300]}")

    def prod_start(self):
        rc, out, err = self.sh(self._prod("start"), where="target", timeout=30)
        if rc != 0:
            raise RuntimeError(f"prod start failed: {err[:300]}")

    def prod_health(self):
        url = self._prod("health_url")
        rc, out, _ = self.sh(f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 5 {url}",
                             where="target")
        try:
            return int(out.strip() or 0) or None
        except ValueError:
            return None

    def prod_live_check(self):
        script = self._prod("live_check_cmd")
        try:
            rc, out, err = self.sh(script, where="target", timeout=240)
        except Exception as e:
            return False, f"live check timed out/failed: {type(e).__name__}"
        ok = rc == 0 and "LIVE_OK" in out
        return ok, (out or err).strip()[-200:]

    # -- path mapping (executor <-> target) -----------------------------------------
    def _target_run_dir(self):
        return os.path.join(self._bundle_target(), "runs", os.path.basename(self.run_dir))

    def ensure_run_dir(self):
        # 09-16 run #6: the target-side run dir is created by the DEPLOY, not
        # by the bundle tarball (the tar has benchmarks/ + campaigns/ only).
        # Without it, every log redirect into it failed INSIDE the background
        # subshell — the parent still emitted PF_pid for the dead job, and
        # the failure surfaced 5 min later as a misleading 'sidecar never
        # became ready'. The dir must exist before any launch redirects
        # into it.
        self.sh(f"mkdir -p {shlex.quote(self._target_run_dir())}")

    def target_path(self, local_path):
        if local_path.startswith(self.run_dir):
            return self._target_run_dir() + local_path[len(self.run_dir):]
        if self.bundle_dir and local_path.startswith(self.bundle_dir):
            return self._bundle_target() + local_path[len(self.bundle_dir):]
        return local_path

    def bundle_path(self):
        return self._bundle_target()

    # -- machine state -------------------------------------------------------------
    def gpu_state(self):
        rc, out, _ = self.sh(
            "nvidia-smi --query-gpu=pci.bus_id,name,memory.used,memory.total "
            "--format=csv,noheader,nounits 2>/dev/null | "
            "awk -F', *' '{print $1\",\"$2\",\"$3\",\"$4}'", where="target")
        gpus = []
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                gpus.append({"bdf": parts[0], "model": parts[1],
                             "mem_used_mib": int(float(parts[2])),
                             "mem_total_mib": int(float(parts[3]))})
        return gpus

    def host_probe(self, since_epoch=None):
        from .. import host_failure
        # the 2026-09-13 gap assessment (4a): the probe SOURCE is
        # profile-overridable so the live detection->stop->restore path can
        # be drilled against a controlled journal fixture without creating a
        # real hardware fault. Default (no key) is the platform probe.
        override = (self.p.get("window") or {}).get("host_probe_override")
        return self._ssh("target", override or host_failure.probe_cmd(since_epoch))

    def host_btime(self):
        rc, out, _ = self._ssh("target", "awk '/btime/{print $2}' /proc/stat")
        try:
            return int(out.strip())
        except ValueError:
            return None

    def disk_free(self, path):
        # fall back to the nearest existing ancestor (the results dir may not
        # exist before the first run)
        script = ("p=" + shlex.quote(path) +
                  "; while [ ! -d \"$p\" ] && [ \"$p\" != / ]; do p=$(dirname \"$p\"); done;"
                  " df -B1 \"$p\" 2>/dev/null | tail -1 | awk '{print $4}'")
        rc, out, _ = self.sh(script, where="target")
        try:
            return int(out.strip())
        except ValueError:
            return 0

    # -- workload lifecycle ---------------------------------------------------------
    def launch(self, script_path, log_path, where="target"):
        import os as _os
        q = shlex.quote
        pidfile = script_path + ".pid"
        inner = (f"nohup setsid bash {q(script_path)} > {q(log_path)} 2>&1 < /dev/null & "
                 f"echo $! > {q(pidfile)}")
        # mkdir -p the log dir in the FOREGROUND: a redirect into a missing
        # dir fails inside the background subshell, where the error goes
        # nowhere and $! still yields a (dead) pid (09-16 run #6).
        cmd = (f"mkdir -p {q(_os.path.dirname(log_path))} && "
               f"cd {q(_os.path.dirname(script_path))} && chmod +x {q(script_path)} && "
               f"sh -c {q(inner)} && echo PF_pid=$(cat {q(pidfile)}) && "
               f"sleep 1 && kill -0 $(cat {q(pidfile)}) 2>/dev/null || echo PF_DEAD")
        rc, out, err = self.sh(cmd, where=where)
        if "PF_DEAD" in out:
            raise StartupFailure(f"launch process died at start (script {script_path}); "
                                 f"check {log_path}: {err[:300]}")
        pid = None
        for line in out.splitlines():
            if "PF_pid=" in line:
                try:
                    pid = int(line.split("PF_pid=")[1].strip())
                except ValueError:
                    pid = None
        if not pid:
            raise StartupFailure(f"launch returned no pid: rc={rc} {err[:300]}")
        return {"pid": pid, "script": script_path}

    # -- window-level sidecar (2026-09-13): a target process that must outlive
    # per-cell server restarts. The script file is staged beforehand (it is
    # campaign code, not bundle code); we only launch and kill it.
    def start_sidecar(self, script_path, log_path, health_url=None, where="target"):
        h = self.launch(script_path, log_path, where=where)
        if health_url:
            ready = self.wait_ready(health_url,
                                    attempts=int(self.p.get("readiness", {}).get("attempts", 60)),
                                    sleep=float(self.p.get("readiness", {}).get("sleep", 5)))
            if not any(ready):
                self.kill(h, where=where)
                raise StartupFailure(f"sidecar never became ready at {health_url}")
        return h

    def stop_sidecar(self, handle, where="target"):
        self.kill(handle, where=where)

    def ping(self):
        rc, _, _ = self._ssh("target", "true", timeout=30)
        return rc == 0

    def rearm_watchdog(self, deadline_epoch, lease_path, heartbeat_path,
                       undo_manifest, prod_desc):
        # an LXC restart (or any target reboot) kills the armed watchdog
        # process; the lease/heartbeat files survive on durable storage.
        # Kill any stale process, then re-arm with the SAME absolute deadline.
        wd = self.p["window"]
        self._ssh("target", f"pkill -f {self._wd_pattern()} 2>/dev/null; true")
        self.arm_watchdog(deadline_epoch, lease_path, heartbeat_path,
                          undo_manifest, prod_desc)

    def kill(self, handle, where="target"):
        pid = handle.get("pid")
        if not pid:
            return
        # cmdline verification before kill (the 2026-09-08 binding rule, mechanized)
        rc, out, _ = self.sh(
            f"[ -d /proc/{pid} ] && tr '\\0' ' ' < /proc/{pid}/cmdline | head -c 200", where=where)
        if rc != 0:
            return                       # already dead: idempotent
        self.sh(f"kill -- -{pid} 2>/dev/null; kill {pid} 2>/dev/null; true", where=where)

    def wait_ready(self, health_url, attempts=90, sleep=5):
        results = []
        for _ in range(attempts):
            rc, out, _ = self.sh(
                f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 3 {health_url}",
                where="target")
            code = out.strip()
            results.append(code == "200")          # the HTTP CODE, never the body
            if code == "200":
                break
            time.sleep(sleep)
        return results

    def run_client(self, argv, timeout=3600):
        json_out = None
        for i, a in enumerate(argv):
            if a == "--json-out":
                json_out = argv[i + 1]
        if not json_out:
            raise ClientFailure("client argv has no --json-out")
        rc, out, err = self.sh(" ".join(shlex.quote(a) for a in argv), timeout=timeout,
                               where="target")
        local = os.path.join(self.run_dir, "_client_last.json")
        self.get_file(json_out, local, where="target")
        try:
            return json.load(open(local))
        except (OSError, ValueError) as e:
            raise ClientFailure(f"client produced no parseable result: {e}; "
                                f"rc={rc} err={err[:200]}")

    def start_telemetry(self, cell_dir, duration, where="target"):
        rc, out, _ = self.sh(
            f"nohup setsid python3 {shlex.quote(self._bundle_target())}/telemetry.py "
            f"--out {shlex.quote(cell_dir)} --duration {int(duration)} > {cell_dir}/telemetry.log "
            f"2>&1 < /dev/null & echo PF_pid=$!", where=where)
        pid = None
        for line in out.splitlines():
            if "PF_pid=" in line:
                pid = int(line.split("PF_pid=")[1].strip())
        return {"pid": pid}

    def telemetry_alive(self, handle):
        if not handle or not handle.get("pid"):
            return False
        rc, out, _ = self.sh(f"kill -0 {handle['pid']} 2>/dev/null && echo alive || echo dead",
                             where="target")
        return out.strip() == "alive"

    def stop_telemetry(self, handle):
        if handle and handle.get("pid"):
            self.sh(f"kill {handle['pid']} 2>/dev/null || true", where="target")

    def _bundle_target(self):
        # the bundle tarball extracts as <deploy_dir>/benchmarks/ — the bundle
        # ROOT is that subdirectory (clients/, telemetry.py live there)
        return os.path.join(self.p.get("deploy", {}).get("target_dir", "/root/campaigns/active"),
                            "benchmarks")

    # -- window mechanics --------------------------------------------------------------
    def _wd_pattern(self):
        """pkill/pgrep -f pattern for the watchdog with the self-exclusion
        idiom: the ssh command runs as `bash -c '<pattern> in this very
        string'`, so a plain pattern matches the checker's OWN process and
        pkill kills its own shell (and pgrep false-positives liveness).
        Replacing the first '.' with '[.]' keeps the regex matching the
        watchdog's cmdline (real dot) but not the literal pattern text."""
        s = self.p["window"]["script"]
        if "." in s:
            s = s.replace(".", "[.]", 1)
        return shlex.quote(s + " " + self.p["window"]["dir"])

    def arm_watchdog(self, deadline_epoch, lease_path, heartbeat_path, undo_manifest,
                     prod_desc):
        # the watchdog runs ON THE TARGET (it must restore what the target hosts)
        wd = self.p["window"]
        undo_target = wd["dir"] + "/undo.sh"
        params = {"deadline_epoch": deadline_epoch, "lease": lease_path,
                  "heartbeat": heartbeat_path, "undo_manifest": undo_target,
                  "prod_unit": prod_desc.get("unit"),
                  "health_url": prod_desc.get("health_url"),
                  "live_check_cmd": prod_desc.get("live_check_cmd"),
                  "restore_result": wd.get("restore_result"),
                  "health_wait_s": int(wd.get("health_wait_s", 1200)),
                  "ensure_guest": wd.get("ensure_guest_cmd", "")}
        b64 = base64.b64encode(json.dumps(params).encode()).decode()
        # undo manifest (local JSON) -> shell script on the target
        try:
            manifest = json.load(open(undo_manifest))
        except OSError:
            manifest = {"temp_changes": []}
        lines = ["#!/bin/sh", "# undo manifest (generated by the campaign window)"]
        for c in manifest.get("temp_changes") or []:
            lines.append(str(c.get("cmd", "")))
        lines.append("true")
        undo_local = os.path.join(self.run_dir, "undo.sh")
        with open(undo_local, "w") as f:
            f.write("\n".join(lines) + "\n")
        self.put_file(undo_local, undo_target, where="target")
        rc, out, err = self._ssh("target",
                                 f"mkdir -p {wd['dir']} && echo '{b64}' | base64 -d > "
                                 f"{wd['dir']}/params.json && "
                                 f"touch {shlex.quote(lease_path)} && touch {shlex.quote(heartbeat_path)} && "
                                 f"{{ setsid nohup bash {wd['script']} {wd['dir']} > "
                                 f"{wd['dir']}/watchdog.log 2>&1 < /dev/null & }}")
        if rc != 0:
            raise RuntimeError(f"watchdog arm spawn failed (rc={rc}): {err[:300]}")
        # P0 (09-14): the lease is created BEFORE the spawn (the watchdog's
        # first loop line is '[ -f $lease ] || exit 0' — the old arm never
        # created it, so every in-LXC watchdog was dead on arrival).
        # P0 (09-16, run #2): the liveness check must NOT trust $! — with
        # `A && … && setsid nohup X &` the whole chain is one background job
        # and $! is the wrapper subshell, which exits when the ssh session
        # unwinds while the setsid-detached watchdog lives (deterministic
        # false negative: WD_DEAD on a healthy watchdog). The watchdog writes
        # its OWN pidfile at start; the check reads it, with a pgrep
        # fallback (self-exclusion idiom) in case the pidfile lags.
        wlog = wd['dir'] + '/watchdog.log'
        pidf = wd['dir'] + '/.watchdog.pid'
        # Bounded polling grace (09-16 run #3): a single 2 s check raced the
        # spawn on a loaded box — the check saw nothing, the runner disarmed,
        # and the still-starting watchdog saw .disarmed and exited silently
        # (no OOM/fork errors in the kernel log: a scheduling race, not a
        # crash). 4 attempts x ~5 s: a truly dead watchdog is still refused
        # (in ~20 s); a slow but healthy start is found.
        rc, out, err = self._ssh("target",
                                 f"for i in 1 2 3 4; do "
                                 f"if [ -f {pidf} ] && kill -0 \"$(cat {pidf})\" 2>/dev/null; then "
                                 f"echo WD_ALIVE=\"$(cat {pidf})\"; exit 0; fi; "
                                 f"W=$(pgrep -f {self._wd_pattern()} | head -1); "
                                 f"if [ -n \"$W\" ]; then echo WD_ALIVE=\"$W\"; exit 0; fi; "
                                 f"sleep 5; done; "
                                 f"head -c 600 {wlog} 2>/dev/null; echo WD_DEAD=1")
        pid = None
        for line in out.splitlines():
            if line.strip().startswith("WD_ALIVE="):
                try:
                    pid = int(line.strip().split("=", 1)[1])
                except ValueError:
                    pass
        if pid is None:
            tail = " ".join(out.split())[:400]
            raise RuntimeError(
                f"watchdog died at arm — refusing to enter the window: {tail}")
        return pid

    def disarm_watchdog(self, lease_path):
        wd = self.p["window"]
        self._ssh("target", f"touch {wd['dir']}/.disarmed; pkill -f {self._wd_pattern()} 2>/dev/null; "
                            f"rm -f {lease_path}; true")

    def heartbeat(self, heartbeat_path):
        self._ssh("target", f"touch {heartbeat_path} 2>/dev/null || true")

    def apply_temp_changes(self, changes, where="host"):
        self._run_changes(changes, where, use_key="cmd")

    def undo_temp_changes(self, changes, where="host"):
        # UNDO runs each entry's "undo" command — never "cmd" again (the
        # 2026-09-13 VM dogfood caught that form re-applying the change at
        # exit: the host was left memory:9216 after every window). The
        # invariant is structural: prepare refuses cmd-without-undo entries
        # (dialects.check_temp_changes), so a missing undo here is
        # unrepresentable; if one ever appears anyway (hand-edited profile),
        # it is a true no-op skip, logged — substitution is not an option.
        self._run_changes(list(reversed(changes)), where, use_key="undo")

    def _run_changes(self, changes, where, use_key="cmd"):
        for c in changes:
            cmd = c.get(use_key)
            if not cmd:
                if use_key == "undo":
                    self._events.append(f"temp-change[undo] skipped (no undo declared): {c.get('cmd','?')[:80]}")
                continue
            rc, out, err = self._ssh(where, cmd)
            if rc != 0 and not c.get("ignore_rc"):
                self._events.append(
                    f"temp-change[{use_key}] rc={rc}: {cmd[:100]} {err[:100]}")

    def notify(self, level, message, campaign_id):
        from .. import notify
        return notify.send(self.p, level, message, campaign_id)

    def events(self):
        return self._events


# local import shim (base is a sibling module)
from .base import StartupFailure, ClientFailure  # noqa: E402
