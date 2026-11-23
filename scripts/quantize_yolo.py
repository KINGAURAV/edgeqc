#!/usr/bin/env python3
"""INT8 post-training static quantisation of the YOLO ONNX model, plus mAP and latency comparison on this device.

    python scripts/quantize_yolo.py --model models/classifier/yolov8n_neu.onnx --data datasets/neu_yolo/neu.yaml

Calibration uses training images, preprocessed exactly as edgeqc.vision.classifier does (letterbox, RGB, /255).
The detection head (last model.N block: box/class decoding, concat, sigmoid) stays FP32. Quantising it
collapses box precision, so only the backbone and neck convolutions go to INT8.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def head_nodes(model_path: Path) -> list[str]:
    import onnx

    g = onnx.load(str(model_path)).graph
    blocks = [int(m.group(1)) for n in g.node if (m := re.match(r"/model\.(\d+)/", n.name))]
    last = max(blocks)
    return [n.name for n in g.node if n.name.startswith(f"/model.{last}/")]


def quantize(model: Path, calib_imgs: list[Path], size: int, out: Path, quant_head: bool) -> Path:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from edgeqc.vision.classifier import letterbox

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib_imgs)

        def get_next(self):
            p = next(self.it, None)
            if p is None:
                return None
            img, *_ = letterbox(cv2.imread(str(p)), size)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return {"images": np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0}

    import onnx
    from onnx import version_converter

    m = onnx.load(str(model))
    if m.opset_import[0].version < 13:  # per-channel DequantizeLinear(axis=...) needs opset >= 13
        m = version_converter.convert_version(m, 13)
    src = out.with_name("_opset13.onnx")
    onnx.save(m, str(src))
    pre = out.with_name("_pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    src.unlink(missing_ok=True)
    quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv"],
                    nodes_to_exclude=[] if quant_head else head_nodes(model))
    pre.unlink(missing_ok=True)
    return out


def latency(model: Path, imgs: list[Path], size: int, threads: int) -> tuple[float, float]:
    from edgeqc.vision.backends import OnnxBackend
    from edgeqc.vision.classifier import YoloDetector

    det = YoloDetector(OnnxBackend(model, threads), [str(i) for i in range(6)], size)
    frames = [cv2.imread(str(p)) for p in imgs]
    for f in frames[:5]:
        det(f)
    lat = []
    for f in frames:
        t = time.perf_counter()
        det(f)
        lat.append((time.perf_counter() - t) * 1e3)
    return statistics.median(lat), float(np.percentile(lat, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quant-head", action="store_true", help="also quantise the detection head (not advised)")
    args = ap.parse_args()

    import yaml
    from ultralytics import YOLO

    model = Path(args.model)
    data = yaml.safe_load(open(args.data))
    droot = Path(data["path"])
    train_imgs = sorted((droot / data["train"]).glob("*.jpg"))
    rng = np.random.default_rng(0)
    calib = [train_imgs[i] for i in rng.choice(len(train_imgs), min(args.calib, len(train_imgs)), replace=False)]
    val_imgs = sorted((droot / data["val"]).glob("*.jpg"))

    out = model.with_name(model.stem + "_int8.onnx")
    quantize(model, calib, args.imgsz, out, args.quant_head)

    for m in (model, out):
        r = YOLO(str(m), task="detect").val(data=args.data, imgsz=args.imgsz, batch=1, device="cpu",
                                            plots=False, verbose=False)
        p50, p95 = latency(m, val_imgs[:120], args.imgsz, args.threads)
        print(f"{m.name:28s} mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f}  "
              f"p50={p50:.2f}ms p95={p95:.2f}ms  size={m.stat().st_size / 1e6:.1f}MB", flush=True)


if __name__ == "__main__":
    main()
