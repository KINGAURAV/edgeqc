#!/usr/bin/env python3
"""Generate synthetic steel-like surface images with known defects.

These are for out-of-the-box demos and smoke tests before real datasets and models
are in place. They are NOT for reporting accuracy.

    python scripts/make_synthetic_samples.py --out samples/synthetic --n 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def texture(rng: np.random.Generator, size: int) -> np.ndarray:
    base = rng.normal(128, 18, (size, size)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=6, sigmaY=1.2)  # rolling-direction streaks
    base += np.linspace(-10, 10, size, dtype=np.float32)[None, :]  # illumination gradient
    base += rng.normal(0, 3, (size, size)).astype(np.float32)
    return base


def add_scratch(img, rng):
    s = img.shape[0]
    x0, y0 = rng.integers(20, s - 20, 2)
    ang = rng.uniform(0, np.pi)
    L = rng.integers(s // 4, s // 2)
    x1, y1 = int(np.clip(x0 + L * np.cos(ang), 0, s - 1)), int(np.clip(y0 + L * np.sin(ang), 0, s - 1))
    cv2.line(img, (int(x0), int(y0)), (x1, y1), float(img.mean() - 55), 2, cv2.LINE_AA)
    return "scratches", [int(min(x0, x1)), int(min(y0, y1)), abs(x1 - int(x0)) + 2, abs(y1 - int(y0)) + 2]


def add_spot(img, rng):
    s = img.shape[0]
    c = tuple(int(v) for v in rng.integers(30, s - 30, 2))
    r = int(rng.integers(4, 9))
    cv2.circle(img, c, r, float(img.mean() - 60), -1, cv2.LINE_AA)
    return "pitted_surface", [c[0] - r, c[1] - r, 2 * r, 2 * r]


def add_patch(img, rng):
    s = img.shape[0]
    x, y = (int(v) for v in rng.integers(20, s - 80, 2))
    w, h = (int(v) for v in rng.integers(30, 60, 2))
    mask = np.zeros_like(img)
    cv2.ellipse(mask, (x + w // 2, y + h // 2), (w // 2, h // 2), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 4)
    img += mask * 45
    return "patches", [x, y, w, h]


def add_crack(img, rng):
    s = img.shape[0]
    pts = [rng.integers(40, s - 40, 2)]
    for _ in range(6):
        pts.append(np.clip(pts[-1] + rng.integers(-18, 19, 2) + np.array([12, 0]), 0, s - 1))
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], False, float(img.mean() - 65), 1, cv2.LINE_AA)
    x, y, w, h = cv2.boundingRect(pts)
    return "crack", [int(x), int(y), int(w), int(h)]


DEFECTS = {"scratch": add_scratch, "spot": add_spot, "patch": add_patch, "crack": add_crack}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/synthetic")
    ap.add_argument("--n", type=int, default=12, help="images per class")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    labels = {}
    for cls in ["good", *DEFECTS]:
        (out / cls).mkdir(parents=True, exist_ok=True)
        for i in range(args.n):
            img = texture(rng, args.size)
            boxes = []
            if cls != "good":
                boxes.append(DEFECTS[cls](img, rng))
            gray = np.clip(img, 0, 255).astype(np.uint8)
            path = out / cls / f"{cls}_{i:03d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            labels[str(path.relative_to(out))] = [{"label": lbl, "bbox": b} for lbl, b in boxes]
    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"Wrote {len(labels)} images to {out}")


if __name__ == "__main__":
    main()
