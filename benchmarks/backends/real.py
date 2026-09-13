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
        rc, out, err = self._ssh(where,
                                 f"mkdir -p $(dirname {shlex.quote(remote_path)}) && "
                                 f"echo '{b64}' | base64 -d > {shlex.quote(remote_path)} && echo WROTE")
        if "WROTE" not in out:
            raise OSError(f"put_file failed for {remote_path}: {err[:200]}")

    def get_file(self, remote_path, local_path, where="target"):
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        rc, out, err = self._ssh(where, f"base64 {shlex.quote(remote_path)} 2>/dev/null")
        if rc != 0:
            raise OSError(f"get_file failed for {remote_path}: {err[:200]}")
        with open(local_path, "wb") as f:
            f.write(base64.b64decode(out))

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

    def host_probe(self):
        from .. import host_failure
        return self._ssh("target", host_failure.PROBE)

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
        cmd = (f"cd {q(_os.path.dirname(script_path))} && chmod +x {q(script_path)} && "
               f"sh -c {q(inner)} && echo PF_pid=$(cat {q(pidfile)})")
        rc, out, err = self.sh(cmd, where=where)
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
                  "health_wait_s": int(wd.get("health_wait_s", 1200))}
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
                                 f"setsid nohup bash {wd['script']} {wd['dir']} > "
                                 f"{wd['dir']}/watchdog.log 2>&1 < /dev/null & "
                                 f"echo PF_armed=$!")
        for line in out.splitlines():
            if line.strip().startswith("PF_armed="):
                return int(line.strip().split("=", 1)[1])
        raise RuntimeError(f"watchdog arm failed: {err[:300]}")

    def disarm_watchdog(self, lease_path):
        wd = self.p["window"]
        self._ssh("target", f"touch {wd['dir']}/.disarmed; pkill -f {shlex.quote(wd['script'])} 2>/dev/null; "
                            f"rm -f {lease_path}; true")

    def heartbeat(self, heartbeat_path):
        self._ssh("target", f"touch {heartbeat_path} 2>/dev/null || true")

    def apply_temp_changes(self, changes, where="host"):
        self._run_changes(changes, where)

    def undo_temp_changes(self, changes, where="host"):
        self._run_changes(list(reversed(changes)), where)

    def _run_changes(self, changes, where):
        for c in changes:
            rc, out, err = self._ssh(where, c["cmd"])
            if rc != 0 and not c.get("ignore_rc"):
                # a failed temp change is not fatal before the first cell (pre-apply),
                # but must be reported; the undo list still records the intended state
                self._events.append(f"temp-change rc={rc}: {c['cmd'][:100]} {err[:100]}")

    def notify(self, level, message, campaign_id):
        from .. import notify
        return notify.send(self.p, level, message, campaign_id)

    def events(self):
        return self._events


# local import shim (base is a sibling module)
from .base import StartupFailure, ClientFailure  # noqa: E402
