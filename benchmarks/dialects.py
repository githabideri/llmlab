"""dialects.py — verified per-PVE-version command tables.

The 2026-09-12 maintenance script hand-rolled `pct set <id> memory:NNN` and
`pct restart <id>` — neither is a real Proxmox command. The table is keyed on
(version, guest): pct for LXC, qm for VM (the guest dimension was missing
until the 2026-09-13 dogfood #2 hit a VM target), and prepare() refuses any
profile temp-change that is not an exact rendering of the table. The failure cost the
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


def _guest_table(dialect, guest):
    """Table for a (version, guest) pair. Tolerates the pre-2026-09-13 flat
    shape (ops at the top level, LXC-only) as an LXC table."""
    t = _load().get(dialect) or {}
    if guest in t and isinstance(t[guest], dict) and "memory_set" in t[guest]:
        return t[guest]
    if guest == "lxc" and "memory_set" in t:
        return t
    return {}


def render(dialect, op, guest, **kw):
    """Render a tabled command. Returns the command string, or None if the
    (dialect, guest, op) triple is not a verified entry — callers must treat
    None as a prepare-time failure. guest is "lxc" or "vm" (pct vs qm)."""
    tpl = (_guest_table(dialect, guest) or {}).get(op)
    if tpl is None:
        return None
    out = tpl
    for k, v in kw.items():
        out = out.replace("{%s}" % k, str(v))
    if "{" in out:
        return None          # unrendered placeholder = caller bug, refuse
    return out


def validate(dialect, raw_cmd, guest):
    """A raw command is acceptable only if it is exactly a table rendering for
    this (dialect, guest) pair (placeholders may be filled with a single
    non-space token). The 2026-09-12 contract, enforced in prepare()."""
    import re
    t = _guest_table(dialect, guest) or {}
    for op, tpl in t.items():
        pat = re.escape(tpl)
        for k in re.findall(r"\{([a-z_]+)\}", tpl):
            pat = pat.replace(re.escape("{%s}" % k), r"(\S+)", 1)
        if re.match("^" + pat + "$", raw_cmd.strip()):
            return True
    return False


def ops(dialect, guest):
    return sorted((_guest_table(dialect, guest) or {}).keys())
