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
        self.lease = profile.get("window", {}).get("lease", "/run/campaign.lock")
        self.heartbeat = profile.get("window", {}).get("heartbeat", "/run/campaign-heartbeat")
        self.entered = False
        self.closed = False
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
        self.b.arm_watchdog(deadline, self.lease, self.heartbeat,
                            os.path.join(self.run_dir, "undo-manifest.json"), prod_desc)
        with open(os.path.join(self.run_dir, "undo-manifest.json"), "w") as f:
            json.dump({"temp_changes": self.temp_changes}, f, indent=2)
        # (2) temp resource changes (memory bump, ...) — profile-declared, undone on exit
        if self.temp_changes:
            self.b.apply_temp_changes(self.temp_changes)
        # (3) stop production; on ANY post-stop failure, restore immediately
        try:
            self.b.prod_stop()
        except Exception:
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
        health = None
        live = False
        try:
            self.b.prod_start()
            for _ in range(30):            # non-fatal health wait (cold starts take minutes)
                health = self.b.prod_health()
                if health == 200:
                    live, _ = self.b.prod_live_check()
                    if live:
                        break
                time.sleep(2)
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
