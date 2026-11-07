#!/usr/bin/env python3
"""Execute the frozen Mirage task list (manifest/tasks.jsonl) and write raw
results. One engine instance for the whole run; CACTUS_TELEMETRY=0.

Output: results/run-<UTCstamp>/{raw.jsonl, perf.json, cold.json, meta.json}

raw.jsonl line:
  {task_id, image_id, type, code, bytes_len, wall_ms, rss_kb, threads,
   parsed (dict | null), error (str | null), raw_prefix (first 256 B on failure)}

The engine's cactus_describe returns a positive "code" on every call (values
observed 64..219; the code tracks the output-token budget actually applied,
default 154, clamped at 154 even when higher values are requested). When
generation hits the budget before the JSON is closed, the library still
returns the truncated byte string (it only raises on negative codes), so
every call is re-validated here as JSON. Truncation is recorded, retried
once (deterministic failures repeat), and graded as a hard failure by
grade.py. See RULES.md.
"""
import json
import os
import sys
import time
import hashlib

os.environ["CACTUS_TELEMETRY"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
RUN_NAME = "run-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
TASKS_FILE = "tasks.jsonl"


def rss_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def nthreads():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def load_manifest(name):
    with open(os.path.join(HERE, "manifest", name)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    global RUN, RUN_NAME, TASKS_FILE
    args = sys.argv[1:]
    if "--tasks" in args:
        TASKS_FILE = args[args.index("--tasks") + 1]
    if "--name" in args:
        RUN_NAME = args[args.index("--name") + 1]
    RUN = os.path.join(HERE, "results", RUN_NAME)
    os.makedirs(RUN, exist_ok=True)
    t_start = time.time()
    images = {i["id"]: i for i in load_manifest("images.jsonl")}
    tasks = load_manifest(TASKS_FILE)

    from cactus import Mirage
    from cactus.agent import mirage as M

    def raw_call(path, settings):
        if not os.path.isabs(path):
            path = os.path.join(HERE, path)  # repo-relative manifest paths
        rgb, w, h = M._image(path)
        t0 = time.perf_counter()
        code = lib.cactus_describe(rgb, w, h, json.dumps(settings).encode(),
                                   None, None, m._buffer, len(m._buffer))
        wall = (time.perf_counter() - t0) * 1000.0
        n = m._buffer.raw.find(b"\x00")
        raw = bytes(m._buffer.raw[:n])
        parsed = None
        err = None
        try:
            parsed = json.loads(raw)
        except Exception as e:
            err = f"{e.__class__.__name__}: {e}"
        return code, raw, parsed, err, wall

    def call_for(task, img):
        t = task["type"]
        if t in ("describe", "ocr", "text_presence", "ui_presence", "detect",
                 "determinism", "perf"):
            return raw_call(img["path"], {"sections": ["objects", "text"]})
        if t == "presence":
            return raw_call(img["path"],
                            {"form": {task["query"]: {"enum": ["yes", "no"]}}})
        if t in ("choice", "relation"):
            return raw_call(img["path"],
                            {"form": {task["gt"]["question"]: {"enum": task["options"]}}})
        if t == "count":
            return raw_call(img["path"],
                            {"form": {"How many objects are in the image?":
                                      {"type": "integer",
                                       "minimum": task["lo"],
                                       "maximum": task["hi"]}}})
        raise ValueError(f"unknown task type {t}")

    # ---- cold-start probe (fresh process) ----
    cold = subprocess_cold_probe()

    # ---- engine ----
    t0 = time.perf_counter()
    import cactus as _c
    m = Mirage()
    lib = m._bind()
    load_s = time.perf_counter() - t0

    meta = {
        "run": RUN,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_start": t_start,
        "python": sys.version,
        "cactus_version": getattr(_c, "__version__", "4.0.1(pip)"),
        "model_file": m.weights,
        "model_sha256": sha256(m.weights),
        "engine_path": find_engine_so(),
        "engine_sha256": sha256(find_engine_so()),
        "model_load_s": load_s,
        "rss_after_load_kb": rss_kb(),
        "threads_after_load": nthreads(),
        "host": hostinfo(),
        "n_images": len(images),
        "n_tasks": len(tasks),
        "manifest_sha256": {
            "images.jsonl": sha256(os.path.join(HERE, "manifest", "images.jsonl")),
            "tasks.jsonl": sha256(os.path.join(HERE, "manifest", TASKS_FILE)),
            "RULES.md": sha256(os.path.join(HERE, "RULES.md")),
        },
        "env": {k: os.environ.get(k) for k in
                ("CACTUS_TELEMETRY", "OMP_NUM_THREADS", "HF_TOKEN")},
        "cold": cold,
    }

    out = open(os.path.join(RUN, "raw.jsonl"), "w")
    perf_calls = []
    records = []

    def flush():
        out.seek(0)
        out.truncate(0)
        for rec in records:
            out.write(json.dumps(rec) + "\n")
        out.flush()

    def record(task, img, code, raw, parsed, err, wall):
        rec = {
            "task_id": task["task_id"], "image_id": img["id"],
            "type": task["type"], "code": code, "bytes_len": len(raw),
            "wall_ms": round(wall, 1), "rss_kb": rss_kb(),
            "threads": nthreads(), "parsed": parsed, "error": err,
        }
        if err:
            rec["raw_prefix"] = raw[:256].decode("utf-8", "replace")
            rec["raw_full"] = raw.decode("utf-8", "replace")
        records.append(rec)
        return rec

    # warmup: 3 describes on a fixed image, not scored
    warm_img = images["com-everyday-00"]
    for _ in range(3):
        raw_call(warm_img["path"], {"sections": ["objects", "text"]})

    n = 0
    for task in tasks:
        img = images[task["image_id"]]
        code, raw, parsed, err, wall = call_for(task, img)
        rec = record(task, img, code, raw, parsed, err, wall)
        if err:
            # one retry (truncation is deterministic in practice; the retry
            # result is kept alongside, the first failure stays the record)
            code2, raw2, parsed2, err2, wall2 = call_for(task, img)
            rec["retry"] = {"code": code2, "bytes_len": len(raw2),
                            "wall_ms": round(wall2, 1), "parsed": parsed2,
                            "error": err2}
        if task["type"] == "perf":
            perf_calls.append(rec)
        n += 1
        if n % 25 == 0:
            flush()
            print(f"  {n}/{len(tasks)} done ({time.strftime('%H:%M:%S')})", flush=True)
    flush()
    out.close()

    meta["wall_total_s"] = round(time.time() - t_start, 1)
    meta["rss_end_kb"] = rss_kb()
    meta["hwm_kb"] = hwm_kb()
    meta["threads_end"] = nthreads()
    with open(os.path.join(RUN, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    with open(os.path.join(RUN, "perf.json"), "w") as fh:
        json.dump({"calls": perf_calls}, fh)

    with open(os.path.join(RUN, "cold.json"), "w") as fh:
        json.dump(cold, fh, indent=1)

    print(f"\nrun complete: {n} tasks, {meta['wall_total_s']}s -> {RUN}")


def hwm_kb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_engine_so():
    import cactus.agent.runtime as rt
    so = os.path.join(os.path.dirname(os.path.abspath(rt.__file__)),
                      "..", "engine", "4.0.0")
    # the runtime resolves the .so from the HF cache; locate it there
    cache = os.path.expanduser("~/.cache/cactus/engine")
    for root, dirs, files in os.walk(cache):
        for f in files:
            if f == "libcactus.so":
                return os.path.join(root, f)
    return None


def hostinfo():
    import platform
    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "online": len(os.sched_getaffinity(0)),
    }


def subprocess_cold_probe():
    """Fresh process: import, construct Mirage, first describe, at rest."""
    import subprocess
    script = r"""
import json, time, sys, os
t0 = time.time()
import cactus
t1 = time.time()
from cactus import Mirage
m = Mirage()
t2 = time.time()
import resource
from cactus.agent import mirage as M
rgb, w, h = M._image(sys.argv[1])
lib = m._bind()
code = lib.cactus_describe(rgb, w, h,
    json.dumps({"sections": ["objects", "text"]}).encode(),
    None, None, m._buffer, len(m._buffer))
n = m._buffer.raw.find(b"\x00")
t3 = time.time()
rss = None
try:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1])
except Exception:
    pass
print(json.dumps({
    "import_s": round(t1 - t0, 3),
    "construct_load_s": round(t2 - t1, 3),
    "first_call_s": round(t3 - t2, 3),
    "first_call_code": code,
    "first_call_bytes": n,
    "first_call_parsed": json.loads(bytes(m._buffer.raw[:n])) is not None,
    "rss_kb": rss,
}))
"""
    img = None
    for cand in (os.path.join(HERE, "images", "syn-shapes-simple-1.png"),):
        if os.path.exists(cand):
            img = cand
            break
    if img is None:
        img = os.path.join(HERE, "images", os.listdir(os.path.join(HERE, "images"))[0])
    p = subprocess.run([sys.executable, "-c", script, img],
                       env=dict(os.environ, CACTUS_TELEMETRY="0"),
                       capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        return {"error": p.stderr[-2000:]}
    return json.loads(p.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    main()
