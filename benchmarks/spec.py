"""spec.py — campaign spec: loading, validation, scientific hash.

A campaign spec is JSON (with `//` line comments allowed — jsonc). The split
that matters (the 2026-09-12 lesson in one function):

  * the SCIENTIFIC hash covers everything that defines WHAT is measured: model,
    quant, build-when-scientific, workload, matrix, gates, expectations, reps.
  * the IMPLEMENTATION hash is the llmlab commit the runner was frozen from.

A bounded repair changes the implementation hash only. The runner recomputes
the scientific hash after any repair and refuses to continue if it moved.

`campaign prepare` fails loud on: missing sha fields (prepare fills them from
the live artifacts), gates not sized per cell, documented_negative classes not
in the classifier registry, and repair classes outside the known set.
"""
import copy
import hashlib
import json
import re

# --- jsonc (JSON + // comments + trailing commas) ------------------------------

def _strip_jsonc(text):
    out = []
    for line in text.splitlines():
        # strip a whole-line or trailing // comment that is not inside a string
        in_str = False
        esc = False
        cut = None
        for i, ch in enumerate(line):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                    cut = i
                    break
        if cut is not None:
            line = line[:cut]
        out.append(line)
    return "\n".join(out).replace(",\n}", "\n}").replace(",\n]", "\n]")


def load_spec(path):
    with open(path) as f:
        return json.loads(_strip_jsonc(f.read()))


# --- validation -----------------------------------------------------------------

KNOWN_REPAIR_ALLOWED = {
    "runtime_path", "command_syntax", "parser_exception", "log_format_support",
    "missing_fixture_copy", "quoting",
}
KNOWN_REPAIR_OWNER = {
    "metric_threshold", "workload", "expected_outcome", "model", "quant",
    "benchmark_duration", "build_when_scientific",
}
KNOWN_REPAIR_FORBIDDEN = {
    "delete_evidence", "invalid_to_pass", "skip_required_cell", "rewrite_previous_attempt",
}

from .classifier import registry as _reg


def validate(spec, strict=True):
    """Return a list of problems (empty = valid)."""
    problems = []
    for field in ("id", "question", "model", "matrix", "verdict_policy"):
        if field not in spec:
            problems.append(f"missing required field: {field}")
    art = spec.get("model", {}).get("artifact", {})
    if strict:
        if not art.get("sha256") or art["sha256"] in ("", "FILL"):
            problems.append("model.artifact.sha256 must be filled (prepare does this "
                            "against the live artifact; a spec without it is not runnable)")
        for i, cell in enumerate(spec.get("matrix", [])):
            if "cell" not in cell and "id" not in cell:
                problems.append(f"matrix[{i}] has no cell id")
            gates = cell.get("gates") or {}
            if "min_wall_s" not in gates and "decode_tps_min" not in gates and \
                    "min_completion_tokens" not in gates:
                problems.append(f"matrix[{i}] ({cell.get('cell') or cell.get('id')}) has no "
                                f"sized gates — an unsized gate is how false-completes pass")

    rp = spec.get("repair_policy") or {}
    for bucket, known in (("allowed", KNOWN_REPAIR_ALLOWED),
                          ("requires_owner", KNOWN_REPAIR_OWNER),
                          ("forbidden", KNOWN_REPAIR_FORBIDDEN)):
        for k in rp.get(bucket, []):
            if k not in known:
                problems.append(f"repair_policy.{bucket}: unknown class '{k}' "
                                f"(known: {sorted(known)})")

    dn = set()
    for cell in spec.get("matrix", []):
        for c in (cell.get("expects") or {}).get("documented_negative") or []:
            if c not in _reg.CLASSES:
                problems.append(f"cell {cell.get('cell') or cell.get('id')}: documented_negative "
                                f"class '{c}' not in the classifier registry")
            dn.add(c)
    return problems


# --- scientific hash -------------------------------------------------------------

# fields that define WHAT is measured (the repair lane may not touch any of these)
SCIENCE_FIELDS = ("question", "model", "workload", "matrix", "verdict_policy",
                  "stop_policy", "reps_default")


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _project_science(spec):
    proj = {k: spec[k] for k in SCIENCE_FIELDS if k in spec}
    proj.pop("matrix", None)
    # the matrix's science: everything except launch mechanics
    slim = []
    for cell in spec.get("matrix", []):
        c = copy.deepcopy(cell)
        c.pop("launch", None)
        slim.append(c)
    proj["matrix"] = slim
    return proj


def scientific_hash(spec):
    return hashlib.sha256(_canonical(_project_science(spec)).encode()).hexdigest()


def implementation_hash(llmlab_commit):
    """The frozen llmlab commit IS the implementation identity (bundle = git tree)."""
    return llmlab_commit
