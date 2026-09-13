"""backends/fixture.py — the synthetic target for qualification.

The whole point (the 2026-09-10 lesson): qualification must execute the REAL
runner + REAL client code against something that behaves like the machine.
This backend provides:

  * an in-process local HTTP server that speaks enough of llama.cpp /v1
    (/completion SSE) and /health to drive `clients/bench-llama.py` UNMODIFIED
    over real sockets, with fault-injected behaviors (OOM logs, assert logs,
    empty-200 health, malformed SSE, no-usage streams, tiny/fast responses);
  * a simulated production unit (stop/start/health/live-check) with state;
  * a fake target filesystem (a tmp dir tree) so script-file writes, client
    runs and file pulls exercise the same code paths as SSH;
  * in-memory watchdog arm/disarm/heartbeat with a manual fire() that runs
    the SAME restore sequence the runner's exit uses (idempotency tests);
  * an op/event log so tests can assert on ORDERING (watchdog before stop,
    restore on every exit, no auto-resume after host failure, ...).

Nothing here knows about llmlab internals beyond the client script names —
it models a machine, not the campaign.
"""
import http.server
import json
import os
import shlex
import socketserver
import subprocess
import tempfile
import threading
import time
import urllib.request

from .base import StartupFailure, ClientFailure


def _http_code(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception:
        return None, None


class FixtureServer:
    """One synthetic inference server with a fault mode."""

    def __init__(self, port, fault="ok", tokens=64, itl_ms=10.0, ttft_ms=200.0,
                 usage=False):
        self.port = port
        self.fault = fault
        self.tokens = tokens
        self.itl_ms = itl_ms
        self.ttft_ms = ttft_ms
        self.usage = usage
        self.log_path = None
        self.alive = True
        self._srv = None
        self._thread = None

    def start(self, log_path):
        import time as _t
        self.log_path = log_path
        mode, toks, itl, ttft, usage, fault, start_t = (self.fault, self.tokens,
                                                        self.itl_ms, self.ttft_ms,
                                                        self.usage, self.fault, _t.time())

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/health":
                    if fault == "slow-start" and _t.time() - start_t < 1.0:
                        # still loading: the readiness loop must survive failed
                        # early polls (a real model load takes minutes)
                        self.send_response(503)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.end_headers()
                    # empty body on purpose: vLLM >=0.2 /health is an empty 200
                    return
                if self.path == "/metrics":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"# no metrics in the fixture\n")
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                self.rfile.read(n)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                # llama.cpp /completion SSE: one JSON frame per token
                time.sleep(ttft / 1000.0)
                for k in range(toks):
                    frame = {"content": f"t{k} ", "token_id": k}
                    if fault == "sse-malformed" and k == toks // 2:
                        self.wfile.write(b"data: {not valid json at all\n\n")
                        continue
                    self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(itl / 1000.0)
                self.wfile.write(b"data: [DONE]\n\n")

        class Srv(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._srv = Srv(("127.0.0.1", self.port), H)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

        if fault in ("fail-assert", "fail-oom-mib", "fail-oom-25g", "fail-arch"):
            # the server dies at load time: write its failure log, close the port
            self._die_with_log()

    def _die_with_log(self):
        lines = {
            "fail-assert": [
                "llama_model_loader: - init from /mnt/<models>/gguf/m-00001-of-00003.gguf",
                "/opt/<build>/ggml/src/ggml-backend-meta.cpp:1760: GGML_ASSERT(meta_buf_ctx->bufs[i]) failed",
                "0.02.444.804 E ggml_backend_cuda_buffer_type_alloc_buffer: "
                "allocating 24224.99 MiB on device 0: cudaMalloc failed: out of memory",
                "0.02.444.809 E alloc_tensor_range: failed to allocate CUDA0 buffer of size 25401738496",
            ],
            "fail-oom-mib": [
                "0.01.100.100 E ggml_backend_cuda_buffer_type_alloc_buffer: "
                "allocating 8589.93 MiB on device 0: cudaMalloc failed: out of memory",
                "0.01.100.110 E alloc_tensor_range: failed to allocate CUDA0 buffer of size 9000000000",
            ],
            "fail-oom-25g": [
                "llama_model_loader: - init from /mnt/<models>/gguf/m-00001-of-00003.gguf",
                "0.03.200.100 E ggml_backend_cuda_buffer_type_alloc_buffer: "
                "allocating 24224.99 MiB on device 0: cudaMalloc failed: out of memory",
                "0.03.200.110 E alloc_tensor_range: failed to allocate CUDA0 buffer of size 25401738496",
            ],
            "fail-arch": [
                "0.00.133.494 E llama_model_load_from_file_impl: failed to load model",
                "0.00.133.500 E cmn  common_init_: LLAMA_SPLIT_MODE_TENSOR not implemented "
                "for architecture '<experimental>'",
                "0.00.134.086 E srv  llama_server: exiting due to model loading error",
            ],
        }[self.fault]
        with open(self.log_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        self.alive = False
        time.sleep(0.05)
        self._srv.shutdown()
        self._srv.server_close()

    def stop(self):
        if self.alive and self._srv:
            self.alive = False
            try:
                self._srv.shutdown()
                self._srv.server_close()
            except Exception:
                pass


class FixtureBackend:
    """A fake machine: target fs, prod unit, watchdog, and fault-injected servers."""

    name = "fixture"

    def __init__(self, profile, bundle_dir, run_dir, fault="ok", client="bench-llama",
                 prod_live=True, btime=None):
        self.p = profile
        self.bundle_dir = bundle_dir
        self.run_dir = run_dir
        self.fault = fault
        self.client = client
        # a broken live-check sensor (qualify Q17): the arm-time verification
        # must refuse the window while prod is still healthy
        self.prod_live = prod_live and fault != "live-check-broken"
        self.btime = btime if btime is not None else int(time.time()) - 86400
        self.root = tempfile.mkdtemp(prefix="fixture-target-")
        self.op_log = []
        self.prod_state = "active"
        self.watchdog = {"armed": False, "fired": False, "deadline": None,
                         "pid": None, "disarmed": False}
        self.temp_changes = []
        self.servers = []
        self._port = 18100
        self.rebooted = False

    # -- path mapping (the fixture's "target" is its own tmp tree) ----------------
    def target_path(self, local_path):
        return local_path

    def bundle_path(self):
        return self.bundle_dir

    # -- target fs ---------------------------------------------------------------
    def sh(self, cmd, timeout=120, stdin=None, where="target"):
        self.op_log.append(f"sh[{where}]: {cmd[:140]}")
        # the fixture only needs to understand a few command shapes; everything
        # else is recorded and succeeds (the point is ordering + client runs)
        if where == "host" and "watchdog" in cmd:
            return 0, "PF_armed=777", ""
        if where == "host" and cmd.strip().startswith("touch"):
            return 0, "", ""
        if "base64" in cmd and " > " in cmd:
            # file write over the ssh channel (put_file)
            return 0, "WROTE", ""
        return 0, "", ""

    def put_file(self, local_path, remote_path, where="target"):
        self.op_log.append(f"put[{where}]: {remote_path}")
        base = self.root if where == "target" else os.path.join(self.root, "_host")
        dest = os.path.join(base, remote_path.lstrip("/"))
        os.makedirs(os.path.dirname(dest) or base, exist_ok=True)
        with open(local_path, "rb") as fsrc, open(dest, "wb") as fdst:
            fdst.write(fsrc.read())

    def get_file(self, remote_path, local_path, where="target"):
        base = self.root if where == "target" else os.path.join(self.root, "_host")
        src = os.path.join(base, remote_path.lstrip("/"))
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(src, "rb") as fsrc, open(local_path, "wb") as fdst:
            fdst.write(fsrc.read())

    def write_script(self, path, content, where="target"):
        dest = os.path.join(self.root if where == "target" else os.path.join(self.root, "_host"),
                            path.lstrip("/"))
        os.makedirs(os.path.dirname(dest) or self.root, exist_ok=True)
        with open(dest, "w") as f:
            f.write(content)
        self.op_log.append(f"write_script[{where}]: {path}")

    # -- production ----------------------------------------------------------------
    def prod_stop(self):
        self.op_log.append("prod_stop")
        self.prod_state = "inactive"

    def prod_start(self):
        self.op_log.append("prod_start")
        self.prod_state = "active"

    def prod_health(self):
        if self.prod_state != "active":
            return None
        if self.fault == "prod-health-empty200":
            return 200          # empty body on purpose (the vLLM contract)
        return 200

    def prod_live_check(self):
        ok = self.prod_state == "active" and self.prod_live
        return ok, f"fixture prod live={ok}"

    # -- machine state ----------------------------------------------------------------
    def gpu_state(self):
        return [{"bdf": "00:01.0", "model": "RTX 3060 12GB",
                 "mem_used_mib": 1024, "mem_total_mib": 12288}]

    def host_probe(self):
        kern = ""
        if self.rebooted:
            self.btime += 3600
        if self.fault == "host-xid":
            kern = "\n[  99.1] NVRM: Xid (PCI:0000:2d:00): 79, Graphics Engine Exception\n"
        elif self.fault == "host-mce":
            kern = "\n[  12.3] mce: [Hardware Error]\n"
        body = (f"===KERN===\n{f'[  ok  ] boot' if not self.rebooted else '[  ok  ] reboot'}\n"
                f"{kern}===VMCORE===\n===BTIME===\n{self.host_btime()}\n"
                f"===UP===\n2026-09-13 00:00:00\n")
        return 0, body, ""

    def host_btime(self):
        # 'reboot-mid' fault: after N reads the host has (re)booted
        if self.fault == "reboot-mid" and getattr(self, "reboot_after_checks", None):
            self._probe_n = getattr(self, "_probe_n", 0) + 1
            if self._probe_n > self.reboot_after_checks and not getattr(self, "_reboot_done", False):
                self.btime += 3600
                self._reboot_done = True
        return self.btime

    def disk_free(self, path):
        return 1 << 30

    # -- workload (real servers, real clients, fake plumbing) --------------------------
    def launch(self, script_path, log_path, where="target"):
        self.op_log.append(f"launch: {script_path}")
        # the port comes from the IMMUTABLE launch script (byte-exact contract)
        port = None
        base = self.root if where == "target" else os.path.join(self.root, "_host")
        sp = os.path.join(base, script_path.lstrip("/"))
        try:
            m = __import__("re").search(r"--port\s+(\d+)", open(sp).read())
            if m:
                port = int(m.group(1))
        except OSError:
            pass
        if port is None:
            port = self._next_port()
        fail_set = ("fail-assert", "fail-oom-mib", "fail-oom-25g", "fail-arch")
        srv = FixtureServer(port, fault=self.fault,
                            tokens=(0 if self.fault in fail_set else
                                    2 if self.fault == "tiny-fast" else 64),
                            itl_ms=(5.0 if self.fault == "tiny-fast" else 10.0),
                            ttft_ms=(10.0 if self.fault == "tiny-fast" else 150.0))
        if self.fault in ("ok", "sse-malformed", "sse-no-usage", "tiny-fast", "slow-start"):
            if self.fault == "tiny-fast":
                srv.tokens, srv.itl_ms, srv.ttft_ms = 2, 5.0, 10.0
            srv.start(log_path)
        else:
            srv.start(log_path)          # dies at load (writes its failure log)
        self.servers.append(srv)
        return {"pid": 4000 + len(self.servers), "port": port, "server": srv,
                "log": log_path}

    def _next_port(self):
        self._port += 1
        return self._port

    def kill(self, handle, where="target"):
        self.op_log.append(f"kill pid={handle.get('pid')}")
        srv = handle.get("server")
        if srv:
            srv.stop()

    def wait_ready(self, health_url, attempts=90, sleep=5):
        results = []
        for _ in range(attempts):
            code, _ = _http_code(health_url)
            ok = code == 200
            results.append(ok)
            if ok:
                break
            if not any(s.alive for s in self.servers):
                break            # the server is dead; don't burn the full window
            time.sleep(min(sleep, 0.2))
        return results

    def run_client(self, argv, timeout=3600):
        self.op_log.append(f"run_client: {argv[:3]}...")
        # resolve bundle-relative scripts against the LOCAL bundle (the fixture
        # runs the client here, where the real client code + real sockets live)
        cmd = []
        for a in argv:
            if a.startswith("{bundle}/"):
                cmd.append(os.path.join(self.bundle_dir, a[len("{bundle}/"):]))
            else:
                cmd.append(a)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        json_out = None
        for i, a in enumerate(argv):
            if a == "--json-out" and i + 1 < len(argv):
                json_out = argv[i + 1]
        if not json_out or p.returncode != 0 or not os.path.exists(json_out):
            raise ClientFailure(f"client rc={p.returncode} err={p.stderr[:300]}")
        return json.load(open(json_out))

    # -- telemetry ----------------------------------------------------------------------
    def start_telemetry(self, cell_dir, duration, where="target"):
        os.makedirs(cell_dir, exist_ok=True)
        with open(os.path.join(cell_dir, "nvml.csv"), "w") as f:
            f.write("gpu,used,free\n0,1024,11264\n")
        return {"pid": 9000}

    def telemetry_alive(self, handle):
        return self.fault != "telemetry-dead"

    def stop_telemetry(self, handle):
        pass

    # -- window mechanics -----------------------------------------------------------------
    def arm_watchdog(self, deadline_epoch, lease_path, heartbeat_path, undo_manifest,
                     prod_desc):
        self.op_log.append("arm_watchdog")
        self.watchdog.update(armed=True, deadline=deadline_epoch, pid=777,
                             disarmed=False, lease=lease_path,
                             heartbeat=heartbeat_path,
                             undo=undo_manifest, prod_desc=prod_desc)

    def disarm_watchdog(self, lease_path):
        self.op_log.append("disarm_watchdog")
        self.watchdog.update(disarmed=True, armed=False)

    def heartbeat(self, heartbeat_path):
        pass

    def apply_temp_changes(self, changes, where="host"):
        self.op_log.append(f"apply_temp_changes: {[c.get('cmd') for c in changes]}")
        self.temp_changes.extend(changes)

    def undo_temp_changes(self, changes, where="host"):
        self.op_log.append(f"undo_temp_changes: {[c.get('cmd') for c in changes]}")
        self.temp_changes = []

    def notify(self, level, message, campaign_id):
        self.op_log.append(f"notify[{level}]: {message[:80]}")
        self.notified = getattr(self, "notified", [])
        self.notified.append((level, message))

    # -- test hooks (used by qualify only, never by the runner) ----------------------------
    def simulate_reboot(self):
        self.rebooted = True
        self.btime += 3600

    def simulate_watchdog_fire(self):
        """The deadline hit with the lease still held: kill the workload, undo
        temp changes, start prod, verify, write RESTORE-RESULT.md, disarm."""
        wd = self.watchdog
        if not wd.get("armed") or wd.get("disarmed"):
            return
        wd["fired"] = True
        self.op_log.append("watchdog_fire")
        for s in self.servers:
            s.stop()
        self.undo_temp_changes(self.temp_changes)
        self.prod_start()
        health = self.prod_health()
        live, _ = self.prod_live_check()
        ok = (health == 200) and live
        out = self.p.get("window", {}).get("restore_result")
        if out:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            with open(out, "w") as f:
                f.write(f"VERDICT: {'RESTORED' if ok else 'ATTENTION'}\n"
                        f"health={health} live={live}\n")
        wd.update(disarmed=True, armed=False)

    def events(self):
        return self.op_log
