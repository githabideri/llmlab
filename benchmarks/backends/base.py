"""backends/base.py — the Backend contract.

One interface, two implementations (RealBackend = SSH to the profile's machines,
FixtureBackend = in-process synthetic target). The runner, preflight, window,
host-failure detector and qualify all program against THIS, so qualification
executes the exact same code paths as a live run (no `if DRY_RUN` branch exists
anywhere — the 2026-09-10 lesson that a dry-run proved the state machine but not
the data plane).

Everything the runner might need from the outside world is a method here:
command execution, file transfer, production lifecycle, GPU/host state, workload
launch/kill, telemetry, watchdog, and the temporary resource changes a window
may make (and their undo).
"""
import json


class StartupFailure(Exception):
    """The workload process could not be started (port in use, binary missing, ...)."""


class ClientFailure(Exception):
    """The client produced no usable result record."""


class WindowError(Exception):
    """A window transition failed (arm/stop/start/verify)."""


class Backend:
    # -- identity / state ------------------------------------------------------
    name = "abstract"

    # -- raw I/O ---------------------------------------------------------------
    def sh(self, cmd, timeout=120, stdin=None):
        """Execute a command (on the target by default). -> (rc, stdout, stderr)."""
        raise NotImplementedError

    def put_file(self, local_path, remote_path, where="target"):
        raise NotImplementedError

    def get_file(self, remote_path, local_path, where="target"):
        raise NotImplementedError

    # -- production lifecycle ---------------------------------------------------
    def prod_stop(self):
        raise NotImplementedError

    def prod_start(self):
        raise NotImplementedError

    def prod_health(self):
        """Return the HTTP status code of the prod health endpoint (NOT the body —
        the empty-200 contract). None if unreachable."""
        raise NotImplementedError

    def prod_live_check(self):
        """A live completion against prod. -> (ok: bool, detail: str)."""
        raise NotImplementedError

    # -- machine state -----------------------------------------------------------
    def gpu_state(self):
        """-> list of {bdf, model, mem_used_mib, mem_total_mib}."""
        raise NotImplementedError

    def host_probe(self):
        """Read-only host probe for the host-failure detector. -> (rc, text, err)."""
        raise NotImplementedError

    def host_btime(self):
        """Boot time in seconds since epoch (reboot detection). -> int|None."""
        raise NotImplementedError

    def disk_free(self, path):
        raise NotImplementedError

    # -- workload lifecycle (the script-file pattern, never inline) ---------------
    def write_script(self, path, content, where="target"):
        """Write a script file (the byte-exact launch pattern; inline `bash -c`
        through nested shells has eaten JSON args before)."""
        raise NotImplementedError

    def launch(self, script_path, log_path, where="target"):
        """nohup+setsid a script file. -> handle {pid} or raise StartupFailure."""
        raise NotImplementedError

    def kill(self, handle, where="target"):
        """Kill by PID, process-group, after cmdline verification (never pkill)."""
        raise NotImplementedError

    def wait_ready(self, health_url, attempts=90, sleep=5):
        """Poll a /health-style endpoint. -> bool (all-ready)."""
        raise NotImplementedError

    def run_client(self, argv, timeout=3600):
        """Run a client script (subprocess), parse its --json-out. -> dict or
        raise ClientFailure."""
        raise NotImplementedError

    def start_telemetry(self, cell_dir, duration, where="target"):
        raise NotImplementedError

    def telemetry_alive(self, handle):
        raise NotImplementedError

    def stop_telemetry(self, handle):
        raise NotImplementedError

    # -- window mechanics ---------------------------------------------------------
    def arm_watchdog(self, deadline_epoch, lease_path, heartbeat_path,
                     undo_manifest, prod_desc):
        """Idempotent. The watchdog is OUT-OF-BUNDLE (host-deployed); this only
        hands it its parameters. MUST succeed before anything destructive."""
        raise NotImplementedError

    def disarm_watchdog(self, lease_path):
        raise NotImplementedError

    def heartbeat(self, heartbeat_path):
        raise NotImplementedError

    def apply_temp_changes(self, changes, where="host"):
        """Profile-declared temporary changes (memory bump, ...). Each change dict
        carries its own undo (the dialect-rendered command)."""
        raise NotImplementedError

    def undo_temp_changes(self, changes, where="host"):
        raise NotImplementedError

    # -- notification -------------------------------------------------------------
    def notify(self, level, message, campaign_id):
        raise NotImplementedError

    def target_path(self, local_path):
        # map an executor-local path to its mirror on the target (identity by default)
        return local_path

    def bundle_path(self):
        # where the deployed bundle root lives (executor-local for the fixture)
        return self.bundle_dir

    def events(self):
        """Ordered op log (fixture records it; real returns [])."""
        return []


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
