"""notify.py — owner notification policy (severity, not cleverness).

Scientific ambiguity must not page a human at 2 am; operational risk must.
Levels (mapped from verdict.campaign_final in verdict.NOTIFY_LEVELS):

  INFO       — campaign completed / expected negative / clean retry
  ATTENTION  — harness failure, unknown classification, safe stop, review needed
  URGENT     — restore failed, watchdog fired with ATTENTION, machine unreachable

The channel is a scoped webhook token from the (private) deployment profile —
the cloud executor never holds credentials; it calls notify() with a level and
a message, and the mapping is platform code, not agent judgment.
"""
import json
import urllib.request


def send(profile, level, message, campaign_id):
    """Fire-and-mostly-forget: a dead notification path must never kill the
    campaign (it is logged, not raised)."""
    n = (profile or {}).get("notify") or {}
    url = n.get("webhook_url")
    if not url:
        return {"sent": False, "reason": "no webhook configured"}
    payload = json.dumps({"level": level, "campaign": campaign_id,
                          "message": message}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {n.get('token', '')}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return {"sent": True, "status": r.status}
    except Exception as e:
        return {"sent": False, "reason": f"{type(e).__name__}: {e}"}


def message_for(final, cell_summary, restore):
    """One-paragraph morning report, assembled by the platform (the owner reads,
    the owner does not assemble)."""
    cells = ", ".join(f"{k}={v}" for k, v in cell_summary.items())
    restore_txt = ("production restored and live-verified" if restore.get("restored")
                   else f"production RESTORE INCOMPLETE (health={restore.get('health')}, "
                        f"live={restore.get('live')})")
    return (f"campaign {final}: {cells}; {restore_txt}. "
            f"See final.json for details.")
