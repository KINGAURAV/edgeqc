#!/usr/bin/env python3
"""Per-stage latency of one inspection, as the app runs it: capture (image decode) -> anomaly -> classifier ->
fusion/severity -> overlay render -> JPEG encode -> SQLite log. Reports median / p90 in ms for the 1-core
(pinned) sessions and, if vision.boost_threads is set, the boosted sessions (LLM idle).

    python scripts/profile_stages.py --config configs/profiles/steel.yaml --device configs/devices/pi4.yaml \
        --source samples/neu_val --frames 60
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from edgeqc.capture import open_source  # noqa: E402
from edgeqc.config import load_config  # noqa: E402
from edgeqc.storage import InspectionLog  # noqa: E402
from edgeqc.vision import InspectionPipeline  # noqa: E402
from edgeqc.vision.overlay import render  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def run(pipeline, source, frames, cfg, store):
    ui = cfg["ui"]
    t = {k: [] for k in ("capture", "anomaly", "classifier", "fusion_severity", "render", "jpeg", "log", "total")}
    for _ in range(3):  # warm-up
        pipeline.inspect(source.read())
    for _ in range(frames):
        t0 = time.perf_counter()
        frame = source.read()
        t1 = time.perf_counter()
        res = pipeline.inspect(frame)
        t2 = time.perf_counter()
        heat = pipeline.last_anomaly.heatmap if pipeline.last_anomaly else None
        img = render(frame, res, heat, 0.0, int(ui.get("display_max_side", 640)))
        t3 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(ui.get("jpeg_quality", 80))])
        t4 = time.perf_counter()
        store.log(res)
        t5 = time.perf_counter()
        t["capture"].append((t1 - t0) * 1e3)
        for k in ("anomaly", "classifier", "fusion_severity"):
            t[k].append(res.latency_ms.get(k, 0.0))
        t["render"].append((t3 - t2) * 1e3)
        t["jpeg"].append((t4 - t3) * 1e3)
        t["log"].append((t5 - t4) * 1e3)
        t["total"].append((t5 - t0) * 1e3)
    return {k: {"median": round(statistics.median(v), 1), "p90": round(pct(v, 0.9), 1)} for k, v in t.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--device")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--source")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--json")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set, device=args.device)
    source = open_source(args.source or cfg["app"]["source"], loop=True)
    vcpus = cfg["vision"].get("cpu_affinity")
    if vcpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, vcpus)
    pipeline = InspectionPipeline(cfg)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        store = InspectionLog(Path(tmp), "none")
        pipeline.set_boost(False)
        out["pinned"] = run(pipeline, source, args.frames, cfg, store)
        if pipeline.can_boost:
            pipeline.set_boost(True)
            out["boost"] = run(pipeline, source, args.frames, cfg, store)
        store.close()
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
