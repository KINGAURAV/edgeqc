#!/usr/bin/env python3
"""Convert NEU-DET (Pascal VOC XML) into YOLO format.

    python scripts/prepare_neu_det.py --src datasets/NEU-DET --out datasets/neu_yolo

It finds every *.xml under --src and matches each one to its image by filename.
If the source has "validation" folders they become the val split; otherwise a
seeded 80/20 split is used.
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    imgs = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".bmp", ".png"}}
    xmls = sorted(src.rglob("*.xml"))
    has_val = any("valid" in str(x).lower() for x in xmls)
    random.seed(args.seed)
    n = {"train": 0, "val": 0}
    for x in xmls:
        root = ET.parse(x).getroot()
        stem = Path(root.findtext("filename", x.stem)).stem
        img = imgs.get(stem) or imgs.get(x.stem)
        if img is None:
            continue
        W, H = int(root.find("size/width").text), int(root.find("size/height").text)
        split = ("val" if "valid" in str(x).lower() else "train") if has_val else \
                ("val" if random.random() < 0.2 else "train")
        lines = []
        for obj in root.findall("object"):
            name = obj.findtext("name").strip()
            if name not in CLASSES:
                continue
            b = obj.find("bndbox")
            x0, y0, x1, y1 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            lines.append(f"{CLASSES.index(name)} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                         f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, out / "images" / split / img.name)
        (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines))
        n[split] += 1
    (out / "neu.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"train={n['train']} val={n['val']} → {out / 'neu.yaml'}")


if __name__ == "__main__":
    main()
