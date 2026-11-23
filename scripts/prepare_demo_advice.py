#!/usr/bin/env python3
"""Fill the advice cache for a demo sequence before recording: every non-PASS defect signature in the image folder
gets a Llama reply (one at a time, nothing dropped), saved to <data_dir>/advice_cache.json. The app then shows
Llama advice instantly for these parts.

    python scripts/prepare_demo_advice.py --config configs/profiles/steel.yaml --source samples/demo/steel \
        --set app.data_dir=runs/demo_steel [--device configs/devices/pi4.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from edgeqc.advisor import AdvisorService  # noqa: E402
from edgeqc.advisor.service import signature  # noqa: E402
from edgeqc.config import load_config  # noqa: E402
from edgeqc.types import Verdict  # noqa: E402
from edgeqc.vision import InspectionPipeline  # noqa: E402


def main():
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device")
    ap.add_argument("--source", required=True)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set + ["advisor.pregenerate=false", "vision.cpu_affinity=null",
                                         "advisor.cpu_affinity=null"], device=a.device)
    pipeline = InspectionPipeline(cfg)
    svc = AdvisorService(cfg, lambda *x: None)
    if svc.llm is None:
        print("No Llama artefacts found: the demo will use rule-based advice.")
        return
    files = sorted(p for p in Path(a.source).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    done = set()
    for f in files:
        r = pipeline.inspect(cv2.imread(str(f)))
        if r.verdict == Verdict.PASS:
            continue
        sig = signature(r)
        if sig in done or svc.cached(r):
            continue
        done.add(sig)
        t = time.perf_counter()
        text, engine, _ = svc.advise_sync(r, stream=False)
        svc._deliver(sig, None, text, engine, time.perf_counter() - t)  # caches + saves Llama replies to disk
        print(f"{f.name}: {engine} {time.perf_counter() - t:.0f}s | {text.splitlines()[0]}", flush=True)
    svc.close()
    print(f"Advice ready for {len(done)} defect signatures.")


if __name__ == "__main__":
    main()
