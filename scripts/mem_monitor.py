#!/usr/bin/env python3
"""Sample the memory of a process tree (Linux) once per second until it exits: PSS / RSS summed over the process
and its children (e.g. the app + llama_server), system MemAvailable, swap used and SoC temperature.

    python scripts/mem_monitor.py --pid 1234 --json runs/mem.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def children(pid: int) -> list[int]:
    out = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                if int((p / "stat").read_text().rsplit(")", 1)[1].split()[1]) == pid:
                    out.append(int(p.name))
            except (OSError, IndexError, ValueError):
                pass
    return out


def tree(pid: int) -> list[int]:
    todo, seen = [pid], []
    while todo:
        p = todo.pop()
        seen.append(p)
        todo += children(p)
    return seen


def rollup(pid: int) -> dict:
    d = {}
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()[1:]:
            k, v = line.split(":", 1)
            d[k] = int(v.split()[0])  # kB
        d["name"] = Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        pass
    return d


def meminfo() -> dict:
    d = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        d[k] = int(v.split()[0])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    samples, t0 = [], time.time()
    while os.path.exists(f"/proc/{args.pid}"):
        procs = {p: rollup(p) for p in tree(args.pid)}
        mi = meminfo()
        try:
            temp = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
        except OSError:
            temp = None
        by_name: dict[str, dict] = {}
        for r in procs.values():
            if "Pss" in r:
                b = by_name.setdefault(r["name"], {"pss_mb": 0.0, "rss_mb": 0.0})
                b["pss_mb"] += r["Pss"] / 1024
                b["rss_mb"] += r["Rss"] / 1024
        samples.append({
            "t": round(time.time() - t0, 1),
            "pss_mb": round(sum(r.get("Pss", 0) for r in procs.values()) / 1024, 1),
            "rss_mb": round(sum(r.get("Rss", 0) for r in procs.values()) / 1024, 1),
            "swap_mb": round(sum(r.get("Swap", 0) for r in procs.values()) / 1024, 1),
            "by_name": {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in by_name.items()},
            "mem_available_mb": round(mi["MemAvailable"] / 1024, 1),
            "sys_swap_used_mb": round((mi["SwapTotal"] - mi["SwapFree"]) / 1024, 1),
            "temp_c": temp,
        })
        time.sleep(args.interval)
    peak = max(samples, key=lambda s: s["pss_mb"]) if samples else {}
    summary = {"samples": len(samples), "peak_pss_mb": peak.get("pss_mb"), "peak_rss_mb": max((s["rss_mb"] for s in samples), default=None),
               "peak_by_name": peak.get("by_name"), "min_mem_available_mb": min((s["mem_available_mb"] for s in samples), default=None),
               "max_sys_swap_used_mb": max((s["sys_swap_used_mb"] for s in samples), default=None),
               "max_temp_c": max((s["temp_c"] or 0 for s in samples), default=None)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"summary": summary, "samples": samples}, indent=0))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
