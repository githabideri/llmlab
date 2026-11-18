#!/usr/bin/env python3
"""machine-inventory.py — canonical, sanitized hardware inventory for llmlab.

Run ON the machine (or via ssh: `ssh host 'python3 -' < scripts/machine-inventory.py --tag TAG`).
Produces a machine-readable manifest separating IMMUTABLE hardware facts from
TIME-VARYING measurements, keyed by a neutral machine tag (no host names, IPs,
MACs, or serials — the public-safe invariant this repo is held to; --private
adds host identity for the private homelab copy only).

Usage:
  machine-inventory.py --tag dual3060-27b [--out FILE] [--private]

The output is a timestamped snapshot. Supersede by adding a new file, never by
rewriting history: a changed immutable block in a new snapshot is a hardware
change and must be discussed, not silently absorbed.
"""
import argparse
import datetime
import json
import re
import subprocess


def sh(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout or ""
    except Exception:
        return ""


def parse_cpu():
    d = {}
    for line in sh("lscpu").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    cores = int(d.get("Core(s) per socket", "0") or 0)
    sockets = int(d.get("Socket(s)", "0") or 0)
    threads = int(d.get("Thread(s) per core", "0") or 0)
    model = d.get("Model name", "")
    m = re.search(r"(i\d-\d+|Xeon[\w .-]+|Ryzen[\w .-]+|EPYC[\w .-]+)", model)
    return {
        "model": m.group(1).strip() if m else model,
        "sockets": sockets,
        "cores_per_socket": cores,
        "threads_per_core": threads,
        "logical_cpus": int(d.get("CPU(s)", "0") or 0),
        "features": [f for f in d.get("Flags", "").split()
                     if f in ("avx2", "avx512f", "avx512_vnni", "fma", "sse4_2")],
        "hyperthreading": threads > 1,
    }


def parse_ram():
    total_kb = 0
    for line in sh("free -k").splitlines():
        if line.startswith("Mem:"):
            total_kb = int(line.split()[1])
    modules = []
    for line in sh("dmidecode -t memory 2>/dev/null").splitlines():
        s = line.strip()
        if s.startswith("Size:"):
            g = re.match(r"Size:\s*(\d+) GB", s)
            if g:
                modules.append(int(g.group(1)) * 2**30)
    return {
        "total_bytes": total_kb * 1024,
        "module_sizes_bytes": sorted(modules, reverse=True),
        "swap": sh("swapon --show 2>/dev/null").strip(),
    }


def parse_gpus():
    out = sh("nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id,"
             "pcie.link.gen.max,pcie.link.width.current,driver_version,compute_cap "
             "--format=csv,noheader,nounits 2>/dev/null")
    gpus = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 8:
            continue
        slot = p[3].split(":")[1] + ":00.0"
        parent = sh("lspci -P 2>/dev/null | grep %s | sed 's/.*<- //; s/ .*//'" % slot).strip()
        pt = sh("lspci 2>/dev/null | grep %s" % parent).strip() if parent else ""
        gpus.append({
            "index": int(p[0]),
            "model": p[1],
            "mem_bytes": int(float(p[2])) * 2**20,
            "pci_slot": slot,
            "pcie_gen": int(float(p[4])),
            "pcie_width": int(float(p[5])),
            "upstream_bridge": ("cpu-host-bridge" if re.search(r"Host bridge", pt)
                                else "chipset-pci-bridge" if re.search(r"PCI bridge", pt)
                                else "unknown"),
            "driver_version": p[6],
            "compute_cap": p[7],
        })
    return gpus


def parse_storage():
    disks = []
    seen = set()
    for line in sh("lsblk -bdn -o NAME,SIZE,TYPE 2>/dev/null").splitlines():
        p = [x.strip() for x in line.split()]
        if len(p) >= 3 and p[2] == "disk" and p[0] not in seen:
            seen.add(p[0])
            disks.append({"device": p[0], "size_bytes": int(p[1])})
    for line in sh("lsblk -dn -o NAME,MODEL 2>/dev/null").splitlines():
        p = [x.strip() for x in line.split()]
        if len(p) >= 2:
            for d in disks:
                if d["device"] == p[0]:
                    d["model"] = " ".join(p[1:]).strip()
    mounts = []
    for line in sh("df -BG --output=source,size,avail 2>/dev/null").splitlines()[1:]:
        p = [x.strip() for x in line.split()]
        if len(p) >= 3 and p[0].startswith(("/dev", "/var", "/opt")):
            mounts.append({"source": p[0],
                           "size_gb": int(p[1].rstrip("G") or 0),
                           "avail_gb": int(p[2].rstrip("G") or 0)})
    return disks, mounts


def parse_pci_vga():
    """All VGA controllers from lspci, with LnkCap/LnkSta (static PCIe facts),
    including cards the active driver does not expose (e.g. passed through to
    VMs)."""
    out = []
    for line in sh("lspci 2>/dev/null").splitlines():
        if "VGA compatible" not in line or ":" not in line:
            continue
        bus, rest = line.split(":", 1)
        fcn = rest.strip().split()[0]
        a = bus.strip() + ":" + fcn
        d = sh("lspci -vv -s %s 2>/dev/null" % a)
        cap = re.search(r"LnkCap:.*?Speed ([\d.]+)GT/s.*?Width (x\d+)", d, re.S)
        sta_line = re.search(r"LnkSta:\s*(.+)", d)
        sta = None
        if sta_line:
            s = sta_line.group(1)
            g = re.search(r"Speed ([\d.]+)GT/s.*?Width (x\d+)", s, re.S)
            if g:
                sta = {"speed_gts": float(g.group(1)), "width": g.group(2),
                        "downgraded": "downgraded" in s, "disabled": "disabled" in s}
        out.append({
            "pci_slot": a,
            "description": rest.strip(),
            "lnkcap": {"speed_gts": float(cap.group(1)), "width": cap.group(2)} if cap else None,
            "lnksta": sta,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="neutral machine tag, e.g. dual3060-27b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    gpus = parse_gpus()
    doc = {
        "schema": "llmlab.machine-inventory/1",
        "machine_tag": a.tag,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "immutable": {
            "cpu": parse_cpu(),
            "ram": parse_ram(),
            "gpus": gpus,
            "pci_vga_all": parse_pci_vga(),
            "disks": parse_storage()[0],
        },
        "time_varying": {
            "os": sh("lsb_release -ds 2>/dev/null").strip()
                  or sh("grep PRETTY /etc/os-release | cut -d= -f2").strip().strip('"'),
            "kernel": sh("uname -r").strip(),
            "driver": gpus[0]["driver_version"] if gpus else None,
            "mounts": parse_storage()[1],
        },
    }
    if a.private:
        doc["host_identity"] = {
            "hostname": sh("hostname").strip(),
            "ips": sh("hostname -I 2>/dev/null").split(),
            "board_serial": sh("dmidecode -t system 2>/dev/null | grep 'Serial Number' | awk '{print $3}'").strip(),
        }
    out = json.dumps(doc, indent=2) + "\n"
    if a.out:
        open(a.out, "w").write(out)
        print("wrote", a.out)
    else:
        print(out, end="")


if __name__ == "__main__":
    main()
