#!/usr/bin/env python3
"""Add the smoothed-image-score threshold (image_threshold_smooth) to an existing PatchCore model's meta, and
print the test-set operating points for threshold scales.

The app's image score is the max of the blurred heatmap when the meta has this key (see PatchCoreDetector).
Calibration uses only the model's held-out good training images (holdout.json), same rule as the other
thresholds: max over the holdout x 1.05. New models get the key from train_anomaly.py / optimize_patchcore.py.

    python scripts/calibrate_smooth_score.py --model models/anomaly/pi4/screw/patchcore_fused_int8.onnx \
        --category-dir datasets/mvtec_full/screw [--holdout-json models/anomaly/screw/holdout.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--category-dir", required=True, help="MVTec-style dir with train/good and test/")
    ap.add_argument("--meta", help="meta to update (default: meta_<model stem>.json next to the model)")
    ap.add_argument("--holdout-json", help="default: holdout.json next to the model")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    from edgeqc.vision.anomaly import PatchCoreDetector
    from edgeqc.vision.backends import make_backend

    model = Path(args.model)
    meta_path = Path(args.meta) if args.meta else model.parent / f"meta_{model.stem}.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("image_threshold_smooth", None)
    tmp = meta_path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(meta))
    det = PatchCoreDetector(make_backend("auto", model, args.threads), model.parent / "memory_bank.npy", tmp,
                            image_gate=None)
    tmp.unlink()
    k = int(det.blur_sigma * 4) | 1

    def smooth_max(path: Path) -> float:
        d = det.distance_map(cv2.imread(str(path), cv2.IMREAD_COLOR)).astype(np.float32)
        heat = cv2.resize(d, (det.size, det.size), interpolation=cv2.INTER_LINEAR)
        return float(cv2.GaussianBlur(heat, (k, k), det.blur_sigma).max())

    root = Path(args.category_dir)
    names = json.loads(Path(args.holdout_json or model.parent / "holdout.json").read_text())
    hold = [root / "train" / "good" / n for n in names]
    thr = max(smooth_max(p) for p in hold) * 1.05
    meta["image_threshold_smooth"] = thr
    meta["smooth_holdout_images"] = len(hold)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"{meta_path}: image_threshold_smooth = {thr:.4f} ({len(hold)} holdout images)")

    test = sorted(p for p in (root / "test").rglob("*") if p.suffix.lower() in EXTS)
    if test:
        from sklearn.metrics import roc_auc_score

        y = np.array([p.parent.name != "good" for p in test])
        s = np.array([smooth_max(p) for p in test]) / thr
        print(f"  test: image AUROC {roc_auc_score(y, s):.4f}  ({int(y.sum())} defective, {int((~y).sum())} good)")
        for scale in (1.0, 0.97, 0.95, 0.93, 0.91, 0.90, 0.89, 0.88, 0.87, 0.85):
            print(f"  scale {scale:.2f}: recall {(s[y] >= scale).mean():6.1%}  false alarms {(s[~y] >= scale).mean():6.1%}")


if __name__ == "__main__":
    main()
