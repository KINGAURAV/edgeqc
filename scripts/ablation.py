#!/usr/bin/env python3
"""Ablation of the hybrid vision pipeline (app verdicts, desktop models, 4 threads).

steel (NEU-DET val, 361 defective images): full pipeline vs classifier only vs anomaly (classical) only.
  caught = verdict != PASS; typed = the right NEU class is named.
MVTec (full test sets): classical anomaly detector vs PatchCore. caught = recall, flagged good = false-alarm rate.

    python scripts/ablation.py --json runs/report_bench/ablation.json
"""
import argparse, glob, json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2  # noqa: E402
from edgeqc.config import load_config  # noqa: E402
from edgeqc.vision import InspectionPipeline  # noqa: E402

NEU = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def steel(sets):
    p = InspectionPipeline(load_config("configs/profiles/steel.yaml", sets))
    n = caught = typed = 0
    for f in sorted(glob.glob("datasets/neu_yolo/images/val/*.jpg")):
        t = NEU[int(open(f.replace("images", "labels").replace(".jpg", ".txt")).readline().split()[0])]
        r = p.inspect(cv2.imread(f))
        n += 1
        caught += r.verdict.value != "PASS"
        ty = [d for d in r.defects if d.label in NEU]
        typed += bool(ty) and sorted(ty, key=lambda d: (-d.severity.rank, -d.confidence))[0].label == t
    return {"images": n, "caught": round(caught / n, 3), "typed_correctly": round(typed / n, 3)}


def mvtec(cat, sets):
    p = InspectionPipeline(load_config(f"configs/profiles/mvtec_{cat}.yaml", sets))
    g = b = ng = nb = 0
    for f in sorted(glob.glob(f"datasets/mvtec_full/{cat}/test/*/*.png")):
        flag = p.inspect(cv2.imread(f)).verdict.value != "PASS"
        if "/good/" in f:
            ng += 1; g += flag
        else:
            nb += 1; b += flag
    return {"caught": round(b / nb, 3), "flagged_good": round(g / ng, 3)}


def main():
    logging.disable(logging.WARNING)
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    out = {"steel": {
        "hybrid (deployed)": steel([]),
        "classifier only": steel(["vision.anomaly.classical_z_threshold=1000000000"]),
        "anomaly only (classical)": steel(["vision.classifier.enabled=false"]),
    }, "mvtec": {}}
    print(json.dumps(out["steel"], indent=1), flush=True)
    for cat in ["bottle", "metal_nut", "tile", "grid", "screw"]:
        out["mvtec"][cat] = {"PatchCore (deployed)": mvtec(cat, []),
                             "classical": mvtec(cat, ["vision.anomaly.method=classical"])}
        print(cat, out["mvtec"][cat], flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
