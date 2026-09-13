"""classifier/engine.py — failure-log classification with evidence discipline.

The classifier maps server-log text to a single class plus a three-part evidence
record (the measured/hypothesized separation that past postmortems had to do by
hand):

    observed          — quoted log lines that triggered the match (facts)
    inferred          — derived statements (labelled as such, never as measured)
    not_established   — plausible explanations this run CANNOT establish

Precedence matters and encodes the 2026-09-12 lesson: a build-defect ASSERTION
line outranks the OOM lines it triggers (the assert is the root cause; the OOM is
its symptom). Unit-blind matching is the historical failure mode, so all size
evidence is normalized to decimal GB before thresholding.

The pattern table is data (registry.py) so it is reviewable line-by-line; the
historical failure corpus (fixtures/failures/) is the regression suite that keeps
every pattern honest.
"""
import re

from . import registry

_GB = 1000.0 ** 3


def normalize_alloc_mb(value):
    """Parse a size string like '24224.99 MiB' / '25.40 GB' / '25401738496' and
    return decimal GB (the 09-12 matcher died on exactly this: it only knew
    GB|GiB and saw a MiB-sized OOM as 'generic')."""
    m = re.search(r"([\d.]+)\s*(GiB|MiB|KB|B|GB|MB)?", value.strip())
    if not m:
        return None
    v = float(m.group(1))
    unit = m.group(2)
    if unit in (None, "B"):
        return v / _GB
    return v * {"KiB": 1e-6, "MiB": 1e-3, "GiB": 1.073741824,
                "KB": 1e-3, "MB": 1e0, "GB": 1.0}[unit]


def _lines(text):
    return [l.rstrip("\n") for l in text.splitlines() if l.strip()]


def _first(text, pattern, flags=re.I):
    for l in _lines(text):
        if re.search(pattern, l, flags):
            return l
    return None


def _mk(cls, observed, inferred, not_established):
    return {"class": cls, "observed": observed,
            "inferred": inferred, "not_established": not_established}


def classify(text):
    """Classify a server log. Returns
    {class, observed[], inferred[], not_established[]}.
    class is one of registry.CLASSES or 'UNKNOWN' (which pages the owner)."""
    if text is None:
        return {"class": "UNKNOWN", "observed": [], "inferred": ["no log captured"],
                "not_established": []}

    out = {"class": "UNKNOWN", "observed": [], "inferred": [], "not_established": []}

    # ---- precedence 1: build/toolchain assertion (root cause of many OOMs) ----
    line = _first(text, r"GGML_ASSERT\s*\(.*\)\s*failed|Assertion .* failed|assert .* failed")
    if line:
        out["class"] = "BUILD_DEFECT_ASSERT"
        out["observed"].append(line.strip())
        out["inferred"].append("a debug/toolchain assertion fired inside the build; "
                               "subsequent allocation failures are likely its symptom")
        return out

    # ---- precedence 2: architecture / feature unsupported (control-cell shape) ----
    line = _first(text, r"not implemented for architecture|not supported for architecture|"
                        r"unsupported architecture|no (support|implementation) for arch")
    if line:
        out["class"] = "UNSUPPORTED_ARCH"
        out["observed"].append(line.strip())
        m = re.search(r"architecture\s*'([^']+)'", line)
        if m:
            out["inferred"].append(f"this build does not implement the required feature "
                                   f"for arch '{m.group(1)}'")
        return out

    # ---- precedence 3: GPU identity drift (a cell on the wrong GPU is unmeasurable) ----
    line = _first(text, r"invalid device ordinal|no CUDA-capable device|CUDA error: invalid device|"
                        r"device count .*less than requested", flags=0)
    if line:
        out["class"] = "WRONG_GPU"
        out["observed"].append(line.strip())
        return out

    # ---- precedence 4: allocation failures, size-normalized ----
    oom_lines = [l for l in _lines(text)
                 if re.search(r"cudaMalloc failed|out of memory|OOM[ _-]killed|"
                              r"failed to allocate .*buffer", l, re.I)]
    alloc_gb = []
    for l in oom_lines:
        for m in re.finditer(r"([\d.,]+\s*(?:GiB|MiB|KB|GB|MB|B))\s*(?:on device|buffer of size|allocating)", l, re.I):
            g = normalize_alloc_mb(m.group(1).replace(",", ""))
            if g:
                alloc_gb.append((g, l))
    if oom_lines:
        # host-visible OOM killer lines are a different class (cgroup-scoped)
        line = _first(text, r"oom-kill|Out of memory: Killed|memory cgroup out of memory|"
                            r"Memory cgroup out of memory")
        if line and not re.search(r"cudaMalloc", line, re.I):
            out["class"] = "C_GROUP_OOM"
            out["observed"].append(line.strip())
            return out
        max_gb = max((g for g, _ in alloc_gb), default=None)
        if max_gb is not None and max_gb >= registry.WALL_MIN_GB:
            # the documented wall shape: a single-device allocation above the
            # documented frame size, with no higher-precedence root marker
            out["class"] = "WALL_VRAM_FIT"
            g, l = max(alloc_gb, key=lambda t: t[0])
            out["observed"].append(l.strip())
            out["inferred"].append(f"single-device allocation of {g:.2f} GB exceeds the "
                                   f"documented {registry.WALL_MIN_GB} GB wall threshold")
            out["not_established"].append("which exact tensor/buffer is unshardable "
                                          "(see the build's fit reporting)")
        else:
            out["class"] = "CUDA_OOM"
            out["observed"].append(oom_lines[0].strip())
            if max_gb is not None:
                out["inferred"].append(f"largest allocation attempted {max_gb:.2f} GB "
                                       f"(below the {registry.WALL_MIN_GB} GB wall threshold)")
        return out

    # ---- precedence 5: kernel-level host errors (never a cell outcome) ----
    line = (_first(text, r"Xid \(|NVRM: Xid") or _first(text, r"mce:|Machine Check Exception")
            or _first(text, r"Kernel panic|kernel BUG") or _first(text, r"AER:.*(Uncorrected|Corrected)"))
    if line:
        out["class"] = "HOST_KERNEL_ERROR"
        out["observed"].append(line.strip())
        return out

    # ---- precedence 6: generic model load failure (retryable, unknown shape) ----
    line = _first(text, r"failed to load model|could not load model|model loading error")
    if line:
        out["class"] = "MODEL_LOAD_FAIL"
        out["observed"].append(line.strip())
        out["not_established"].append("the specific load failure cause (see full log)")
        return out

    out["observed"] = [l.strip() for l in _lines(text)[-3:]]  # tail as context
    out["inferred"].append("no known failure pattern matched")
    return out


def classify_response(kind, body):
    """Classify the shape of an HTTP/SSE response — the class family that the
    09-02 and 09-10 nights turned into days of quoting/health-debug.

    The rule that mattered: an empty 200 from a health endpoint is a CONTRACT
    (newer vLLM returns no body). Readiness is the HTTP code, never the body.
    """
    obs = ["response kind=" + str(kind)]
    if kind == "health":
        if not body:
            return _mk("HTTP_EMPTY_200", obs,
                       ["newer vLLM (>=0.2) /health returns an empty 200; "
                        "readiness must be the HTTP code, not the body"],
                       ["the server's actual readiness semantics"])
        if isinstance(body, str) and "error" in body.lower():
            return _mk("HTTP_ERROR", obs,
                       ["health endpoint returned an error body: " + body[:120]], [])
        return _mk("OK", obs, [], [])
    if kind == "sse":
        if not body or (isinstance(body, str) and not body.strip()):
            return _mk("SSE_NO_USAGE", obs,
                       ["stream carried no usage object — token closure unavailable"],
                       ["whether the stream completed"])
        if isinstance(body, str):
            try:
                body = __import__("json").loads(body)
            except Exception as e:
                return _mk("SSE_MALFORMED", obs,
                           ["stream frame is not a JSON object: " + type(e).__name__,
                            "the harness's SSE parser broke on this stream shape; "
                            "this is a client bug, not a server one"],
                           ["server behavior"])
        if not isinstance(body, dict):
            return _mk("SSE_MALFORMED", obs,
                       ["stream frame type: " + type(body).__name__],
                       ["server behavior"])
        if "usage" not in body:
            return _mk("SSE_NO_USAGE", obs,
                       ["stream completed without a usage object — "
                        "completion-token count is not closed"],
                       ["exact decoded token count"])
        return _mk("OK", obs, [], [])
    if kind == "usage":
        if not body:
            return _mk("SSE_NO_USAGE", obs, ["no usage object present"],
                       ["exact decoded token count"])
        if (isinstance(body, dict) and "usage" in body) or \
           (isinstance(body, str) and chr(34)+"usage"+chr(34) in body):
            return _mk("OK", obs, [], [])
        return _mk("SSE_NO_USAGE", obs,
                   ["usage shape not recognized: " + str(body)[:80]],
                   ["exact decoded token count"])
    return _mk("UNKNOWN", obs, [],
               ["unrecognized response kind: " + str(kind)])
