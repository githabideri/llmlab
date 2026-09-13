"""telemetry.py — per-cell resource samplers for campaign cells.

Three independent layers, each to its own CSV, all on a monotonic timebase so
they align with request/phase markers:

  nvml.csv     GPU memory used/free per device (pynvml if importable, else
               nvidia-smi; default 1 Hz — the campaign default; a 100 Hz mode
               exists for spike work, per the mm-image-max sampler)
  cgroup.csv   the target's cgroup memory usage + swap (2 s)
  disk.csv     results-volume usage + inode count (2 s)

It is a *measurement* tool only: it reads counters, never sends inference
requests and never allocates GPU memory. Safe to run alongside a live endpoint.
A dead sampler invalidates the cell (the runner checks liveness before and
after each request — a lost measurement is a lost measurement, not an estimate).

Usage (run ON the machine that owns the GPUs, inside the target's namespace):
  telemetry.py --out /path/to/cell-dir --duration 300 \
      [--hz 1] [--cgroup /sys/fs/cgroup] [--results-disk /mnt/models] \
      [--metrics-url http://127.0.0.1:8082/metrics]
  telemetry.py --analyze /path/to/cell-dir
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request


def get_nvidia_sampler():
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]

        def sample():
            out = []
            for h in handles:
                mi = pynvml.nvmlDeviceGetMemoryInfo(h)
                out.append((mi.used, mi.free))
            return out
        return "pynvml", sample
    except Exception:
        def sample():
            txt = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout
            out = []
            for line in txt.strip().splitlines():
                if not line:
                    continue
                u, f = (float(x) for x in line.split(","))
                out.append((int(u * 1024 * 1024), int(f * 1024 * 1024)))
            return out
        return "nvidia-smi", sample


def _cgroup_file(base, names):
    """cgroup v2 first, v1 fallback."""
    for n in names:
        p = os.path.join(base, n)
        if os.path.exists(p):
            return p
    return None


def cgroup_sample(cg):
    try:
        usage = int(open(os.path.join(cg, "memory.current")).read().strip())
        swap = int(open(os.path.join(cg, "memory.swap.current")).read().strip())
        return (usage, swap)
    except Exception:
        p = _cgroup_file(cg, ["memory/memory.usage_in_bytes", "memory.usage_in_bytes"])
        if not p:
            return (0, 0)
        try:
            return (int(open(p).read().strip()), 0)
        except Exception:
            return (0, 0)


def disk_sample(path):
    st = os.statvfs(path)
    return (st.f_blocks * st.f_frsize, (st.f_blocks - st.f_bfree) * st.f_frsize, st.f_files, st.f_ffree)


def loop(hz, duration, out, sample_fn, header, stop):
    t_end = time.monotonic() + duration
    iters = 0
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(header)
        while not stop.is_set() and time.monotonic() < t_end:
            t0 = time.monotonic()
            try:
                row = sample_fn()
            except Exception:
                row = None
            if row is not None:
                w.writerow([time.monotonic_ns()] + list(row))
                f.flush()
            iters += 1
            dt = time.monotonic() - t0
            s = (1.0 / hz) - dt
            if s > 0:
                stop.wait(min(s, 1.0))
    print(f"TELEMETRY {os.path.basename(out)} iters={iters} duration={duration}s", flush=True)


def metrics_poll(url, interval, out, stop):
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["t_mono_ns", "metric", "value"])
        while not stop.is_set():
            try:
                body = urllib.request.urlopen(url, timeout=5).read().decode()
            except Exception:
                stop.wait(interval)
                continue
            tn = time.monotonic_ns()
            for line in body.splitlines():
                if line.startswith(("vllm:mm_cache_hits_total", "vllm:prefix_cache_hits_total",
                                    "llama_spec", "vllm:time_to_first_token")):
                    try:
                        name, val = line.rsplit(" ", 1)
                        w.writerow([tn, name, float(val)])
                    except ValueError:
                        pass
            f.flush()
            stop.wait(interval)


def analyze(cell_dir):
    for name in ("nvml.csv", "cgroup.csv", "disk.csv"):
        p = os.path.join(cell_dir, name)
        if not os.path.exists(p):
            continue
        rows = [r for r in csv.reader(open(p)) if r and r[0] != name.split(".")[0]]
        if not rows:
            print(f"{name}: no samples")
            continue
        if name == "nvml.csv":
            by_gpu = {}
            for r in rows:
                by_gpu.setdefault(r[1], []).append(int(r[3]))
            for g, frees in sorted(by_gpu.items()):
                print(f"{name} GPU{g}: n={len(frees)} min_free={min(frees)//1048576}MiB "
                      f"median_free={int(statistics.median(frees))//1048576}MiB")
        elif name == "cgroup.csv":
            usages = [int(r[1]) for r in rows]
            swaps = [int(r[2]) for r in rows]
            print(f"{name}: n={len(rows)} max_mem={max(usages)//(1024**2)}MiB "
                  f"max_swap={max(swaps)//(1024**2)}MiB")
        else:
            used = [int(r[2]) for r in rows]
            print(f"{name}: n={len(rows)} max_used={max(used)//(1024**3)}GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="cell dir (writes nvml.csv, cgroup.csv, disk.csv)")
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--cgroup", default="/sys/fs/cgroup", help="cgroup root for the target")
    ap.add_argument("--results-disk", default=None, help="path on the results volume to monitor")
    ap.add_argument("--metrics-url", default=None)
    ap.add_argument("--metrics-interval", type=float, default=2.0)
    ap.add_argument("--analyze", default=None, help="analyze a cell dir instead of sampling")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.analyze)
        return
    os.makedirs(a.out, exist_ok=True)
    stop = threading.Event()
    backend, nvml = get_nvidia_sampler()
    threading.Thread(target=loop, args=(a.hz, a.duration, os.path.join(a.out, "nvml.csv"),
                                        nvml, ["gpu", "used", "free"], stop), daemon=True).start()
    threading.Thread(target=loop, args=(0.5, a.duration, os.path.join(a.out, "cgroup.csv"),
                                        lambda: cgroup_sample(a.cgroup),
                                        ["cgroup_mem_bytes", "cgroup_swap_bytes"], stop), daemon=True).start()
    if a.results_disk:
        threading.Thread(target=loop, args=(0.5, a.duration, os.path.join(a.out, "disk.csv"),
                                            lambda: disk_sample(a.results_disk),
                                            ["disk_total_bytes", "disk_used_bytes",
                                             "inodes_total", "inodes_free"], stop), daemon=True).start()
    if a.metrics_url:
        threading.Thread(target=metrics_poll, args=(a.metrics_url, a.metrics_interval,
                                                    os.path.join(a.out, "metrics.csv"), stop),
                         daemon=True).start()
    print(f"TELEMETRY started out={a.out} nvml_backend={backend} duration={a.duration}s", flush=True)
    t_end = time.monotonic() + a.duration
    while time.monotonic() < t_end:
        time.sleep(1)
    stop.set()
    print("TELEMETRY done", flush=True)


if __name__ == "__main__":
    main()
