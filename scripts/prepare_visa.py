#!/usr/bin/env python3
"""Convert VisA (Amazon Visual Anomaly, CC BY 4.0) categories to the MVTec layout used by the training scripts.

    VisA/<obj>/Data/Images/{Normal,Anomaly}/*.JPG + Masks/Anomaly/*.png + split_csv/1cls.csv
 →  datasets/visa/<obj>/train/good, test/good, test/anomaly, ground_truth/anomaly/<name>_mask.png

Images are downscaled to at most --max-side px (VisA boards are ~1.5k px; models run at 224-288 px).

    python scripts/prepare_visa.py --visa datasets/visa_dl --out datasets/visa --objects pcb1 pcb2 pcb3 pcb4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def resize(img, max_side, interp):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp) if s < 1 else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visa", required=True, help="folder holding split_csv/ and the object folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--objects", nargs="+", default=["pcb1", "pcb2", "pcb3", "pcb4"])
    ap.add_argument("--max-side", type=int, default=1024)
    a = ap.parse_args()
    root, out = Path(a.visa), Path(a.out)
    rows = list(csv.DictReader(open(root / "split_csv" / "1cls.csv")))
    for obj in a.objects:
        n = {}
        for r in (r for r in rows if r["object"] == obj):
            split = "train" if r["split"] == "train" else "test"
            kind = "good" if r["label"] == "normal" else "anomaly"
            if split == "train" and kind != "good":
                continue
            src = root / r["image"]
            name = src.stem + ".png"
            dst = out / obj / split / kind / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), resize(cv2.imread(str(src)), a.max_side, cv2.INTER_AREA))
            if kind == "anomaly" and r.get("mask"):
                m = cv2.imread(str(root / r["mask"]), cv2.IMREAD_GRAYSCALE)
                m = (resize(m, a.max_side, cv2.INTER_NEAREST) > 0).astype("uint8") * 255
                md = out / obj / "ground_truth" / "anomaly" / (src.stem + "_mask.png")
                md.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(md), m)
            n[(split, kind)] = n.get((split, kind), 0) + 1
        print(obj, {f"{s}/{k}": v for (s, k), v in sorted(n.items())})


if __name__ == "__main__":
    main()
