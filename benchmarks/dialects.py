"""dialects.py — verified per-PVE-version command tables.

The 2026-09-12 maintenance script hand-rolled `pct set <id> memory:NNN` and
`pct restart <id>` — neither is a real Proxmox command. The failure cost the
window its memory bump and its container unit-restore (done manually, both
directions). The fix is structural: temporary changes are RENDERED from this
table at prepare time; there is no free-text path. A template that is not in
the table renders None, and prepare fails loud on None.

`validate` is the inverse: a raw command is acceptable only if it equals a
table rendering — the qualification matrix (Q9) exercises both directions.
"""
import json
import os

_TABLE = None


def _load():
    global _TABLE
    if _TABLE is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "backends", "pve_dialects.jsonc")
        text = open(path).read()
        text = "\n".join(l for l in text.splitlines()
                         if not l.strip().startswith("//"))
        _TABLE = json.loads(text)
    return _TABLE


def render(dialect, op, **kw):
    """Render a tabled command. Returns the command string, or None if the
    (dialect, op) pair is not a verified entry — callers must treat None as a
    prepare-time failure."""
    t = _load().get(dialect) or {}
    tpl = t.get(op)
    if tpl is None:
        return None
    out = tpl
    for k, v in kw.items():
        out = out.replace("{%s}" % k, str(v))
    if "{" in out:
        return None          # unrendered placeholder = caller bug, refuse
    return out


def validate(dialect, raw_cmd):
    """A raw command is acceptable only if it is exactly a table rendering
    (placeholders may be filled with a single non-space token)."""
    import re
    t = (_load().get(dialect) or {})
    for op, tpl in t.items():
        pat = re.escape(tpl)
        for k in re.findall(r"\{([a-z_]+)\}", tpl):
            pat = pat.replace(re.escape("{%s}" % k), r"(\S+)", 1)
        if re.match("^" + pat + "$", raw_cmd.strip()):
            return True
    return False


def ops(dialect):
    return sorted((_load().get(dialect) or {}).keys())
