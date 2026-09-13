"""window.py — the maintenance window: one enter, ONE exit, always.

The 2026-09-10 residual flaw was an exit path that didn't converge (watchdog
still armed, lease still held 75 min after restore). The fix, mechanized here:
there is exactly one exit function, it is called from every terminal path
(complete / expected-negative / review / safety-abort / host-failure / signal /
exception), and it is idempotent — calling it twice converges to the same
state (prod up once, watchdog disarmed once, lease released once).

Ordering is the other half (the 09-10 "scarier variant"): the watchdog is
armed BEFORE the first destructive step, and if anything after the prod stop
fails, we restore immediately — never leave production down with no backstop.
"""
import json
import os
import time


class Window:
    def __init__(self, backend, profile, run_dir, deadline_h=2.0,
                 temp_changes=None, heartbeat_thread=None):
        self.b = backend
        self.p = profile
        self.run_dir = run_dir
        self.deadline_h = deadline_h
        self.temp_changes = temp_changes or []
        # restore patience is PROFILE-DRIVEN: a 35B cold start is minutes, not
        # seconds — a 60 s wait turns a healthy restore into a false ATTENTION
        # (and the watchdog is already disarmed, so it can't clean up after).
        prod = profile.get("prod") or {}
        self.restore_wait_s = int(prod.get("restore_wait_s", 600))
        self.restore_poll_s = float(prod.get("restore_poll_s", 10))
        self.lease = profile.get("window", {}).get("lease", "/run/campaign.lock")
        self.heartbeat = profile.get("window", {}).get("heartbeat", "/run/campaign-heartbeat")
        self.entered = False
        self.closed = False
        self.prod_down = False   # exit() only restores what was actually stopped
        self._hb_thread = heartbeat_thread

    # -- enter: watchdog FIRST ------------------------------------------------
    def enter(self):
        if self.entered:
            return
        deadline = int(time.time()) + int(self.deadline_h * 3600)
        prod_desc = {
            "unit": (self.p.get("prod") or {}).get("unit"),
            "health_url": (self.p.get("prod") or {}).get("health_url"),
            "live_check_cmd": (self.p.get("prod") or {}).get("live_check_cmd"),
            "results_dir": self.run_dir,
        }
        # (1) watchdog armed before ANYTHING destructive — always a restore path
        manifest = os.path.join(self.run_dir, "undo-manifest.json")
        with open(manifest, "w") as f:
            json.dump({"temp_changes": self.temp_changes}, f, indent=2)
        self.b.arm_watchdog(deadline, self.lease, self.heartbeat, manifest, prod_desc)
        # (1b) verify the watchdog's sensor BEFORE trusting the watchdog: a live
        # check that fails while prod is known-healthy means the watchdog will
        # spin silently at fire time (2026-09-13 dogfood drill: quoting broke the
        # grep pattern -> 30 min of silent retries, false ATTENTION). Refusing
        # here, with prod still up, is the cheap state.
        if prod_desc["live_check_cmd"]:
            ok, detail = self.b.prod_live_check()
            if not ok:
                self.b.disarm_watchdog(self.lease)
                raise RuntimeError(
                    f"arm-time live check failed ({detail}) — prod is not serving; "
                    "refusing to open the window")
        # (2) temp resource changes (memory bump, ...) — profile-declared, undone on exit
        if self.temp_changes:
            self.b.apply_temp_changes(self.temp_changes)
        # (3) stop production; on ANY post-stop failure, restore immediately.
        # prod_down is set in BOTH branches: a stop that raised may still have
        # stopped (ssh timeout after the stop landed), and exit() must not
        # assume otherwise.
        try:
            self.b.prod_stop()
            self.prod_down = True
        except Exception:
            self.prod_down = True
            self._restore_only()
            raise
        self.entered = True

    # -- heartbeat (runner runs this in a thread while the window is open) ------
    def heartbeat_tick(self):
        if self.entered and not self.closed:
            self.b.heartbeat(self.heartbeat)

    # -- the ONE exit path ------------------------------------------------------
    def exit(self, reason):
        """Idempotent. Restores production, verifies, disarms, releases.
        Returns {restored, health, live} — never raises."""
        if self.closed:
            return self._last_result
        self.closed = True
        # undo temp changes first (reversed), then start prod, then verify
        if self.temp_changes:
            try:
                self.b.undo_temp_changes(self.temp_changes)
            except Exception:
                pass
        if not self.prod_down:
            # prod was never stopped (arm-time refusal, ...): there is nothing
            # to restore — a restore loop here would spin on whatever the
            # live check reports (2026-09-13: 600 s of waiting for a broken
            # sensor after an arm refusal). Disarm and record N/A.
            try:
                self.b.disarm_watchdog(self.lease)
            except Exception:
                pass
            result = {"reason": reason, "restored": None, "health": None,
                      "live": None, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            self._last_result = result
            out = os.path.join(self.run_dir, "RESTORE-RESULT.md")
            with open(out, "w") as f:
                f.write(f"VERDICT: N/A — prod was never stopped (reason: {reason})\n")
                f.write(f"utc: {result['utc']}\n")
            return result
        health = None
        live = False
        deadline = time.time() + self.restore_wait_s
        try:
            self.b.prod_start()
            while time.time() < deadline:            # non-fatal health wait
                health = self.b.prod_health()
                if health == 200:
                    live, _ = self.b.prod_live_check()
                    if live:
                        break
                time.sleep(self.restore_poll_s)
        except Exception:
            health, live = None, False
        # disarm + release ALWAYS happen, whatever the restore outcome
        try:
            self.b.disarm_watchdog(self.lease)
        except Exception:
            pass
        restored = (health == 200) and live
        result = {"reason": reason, "restored": restored, "health": health,
                  "live": live, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self._last_result = result
        out = os.path.join(self.run_dir, "RESTORE-RESULT.md")
        with open(out, "w") as f:
            f.write(f"VERDICT: {'RESTORED' if restored else 'ATTENTION'}\n")
            f.write(f"reason: {reason}\nhealth: {health}\nlive_completion: {live}\n")
            f.write(f"utc: {result['utc']}\n")
        return result

    def _restore_only(self):
        """The enter-failure path: prod down is unacceptable without a backstop."""
        try:
            self.b.prod_start()
        except Exception:
            pass
