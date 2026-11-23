#!/usr/bin/env python3
"""Benchmark the vision pipeline (and optionally one LLM advice call) on this device.

    python scripts/benchmark.py --source samples/synthetic --frames 200
    python scripts/benchmark.py --source datasets/mvtec/bottle/test --set vision.anomaly.method=patchcore --llm

Prints per-stage latency p50/p95, throughput, and peak RSS as a markdown table for the report.
"""

from __future__ import annotations

import argparse
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgeqc.advisor.service import build_advisors  # noqa: E402
from edgeqc.capture import open_source  # noqa: E402
from edgeqc.config import load_config  # noqa: E402
from edgeqc.vision import InspectionPipeline  # noqa: E402


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # bytes on macOS, KiB on Linux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--device", help="e.g. configs/devices/pi4.yaml")
    ap.add_argument("--source", default="samples/synthetic")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--llm", action="store_true", help="also time one advisor call")
    args = ap.parse_args()

    cfg = load_config(args.config, [*args.set, f"app.source={args.source}"], device=args.device)
    pipe = InspectionPipeline(cfg)
    src = open_source(cfg["app"]["source"], loop=True)
    lat: dict[str, list[float]] = {}
    last = None
    t0 = None
    for i in range(args.warmup + args.frames):
        frame = src.read()
        if i == args.warmup:
            t0 = time.perf_counter()
        res = pipe.inspect(frame)
        if i >= args.warmup:
            for k, v in res.latency_ms.items():
                lat.setdefault(k, []).append(v)
            if res.defects:
                last = res
    wall = time.perf_counter() - t0

    print(f"\n### Vision benchmark — {platform.machine()} / {platform.platform()}")
    print(f"anomaly={type(pipe.anomaly).__name__}, classifier={'YOLO' if pipe.classifier else 'off'}, "
          f"backend={cfg['vision']['backend']}, frames={args.frames}, frame={frame.shape[1]}x{frame.shape[0]}\n")
    print("| stage | p50 ms | p95 ms | mean ms |\n|---|---|---|---|")
    for k, v in lat.items():
        a = np.array(v)
        print(f"| {k} | {np.percentile(a, 50):.2f} | {np.percentile(a, 95):.2f} | {a.mean():.2f} |")
    print(f"\nThroughput: {args.frames / wall:.1f} FPS · peak RSS {peak_rss_mb():.0f} MB")

    if args.llm and last is not None:
        llm, rule = build_advisors(cfg)
        adv = llm or rule
        t = time.perf_counter()
        text = adv.advise(last)
        print(f"\n### Advisor ({adv.name}) latency: {time.perf_counter() - t:.2f} s\n{text}")


if __name__ == "__main__":
    main()
