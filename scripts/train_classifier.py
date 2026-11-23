#!/usr/bin/env python3
"""Train YOLOv8n on NEU-DET and export it to ONNX for the edge pipeline.

    python scripts/train_classifier.py --data datasets/neu_yolo/neu.yaml --epochs 100

Ultralytics is AGPL-3.0, which is fine for this open-source challenge repo.
For Hailo, compile the exported ONNX to .hef with the Hailo Dataflow Compiler (docs/HAILO.md).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="models/classifier/yolov8n_neu.onnx")
    ap.add_argument("--name", default="neu_yolov8n", help="run name under runs/train")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--close-mosaic", type=int, default=15)
    ap.add_argument("--hyp", action="append", default=[],
                    help="extra Ultralytics train args, e.g. --hyp flipud=0.5 --hyp hsv_s=0")
    args = ap.parse_args()

    import torch
    import yaml
    from ultralytics import YOLO

    device = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
    model = YOLO(args.model)
    model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=device,
                project="runs/train", name=args.name, exist_ok=True, patience=args.patience,
                cos_lr=True, close_mosaic=args.close_mosaic, seed=args.seed, deterministic=True,
                **{k: yaml.safe_load(v) for k, v in (h.split("=", 1) for h in args.hyp)})
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=device)
    print(f"mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, args.out)
    print(f"Exported → {args.out}. Enable it with --set vision.classifier.enabled=true")


if __name__ == "__main__":
    main()
