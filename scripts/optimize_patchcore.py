#!/usr/bin/env python3
"""Build edge-optimised PatchCore models and compare them on the MVTec test set (on this device).

    python scripts/optimize_patchcore.py --model-dir models/anomaly/metal_nut --category-dir datasets/mvtec/metal_nut

Variants written next to the original artefacts:
    patchcore_fused.onnx       uint8 NHWC RGB in → patch-distance map out. ImageNet normalisation, ResNet-18
                               features AND the memory-bank kNN in one graph (one ONNX Runtime call, Arm-optimised
                               MLAS kernels for the kNN matmul instead of whatever BLAS numpy was built with).
    patchcore_fused_int8.onnx  same, with the backbone convolutions statically quantised to INT8 (QDQ,
                               per-channel). The kNN stays FP32 so distances keep their resolution.
Evaluation uses the app's own PatchCoreDetector, so latency includes resize + post-processing.
The backbone is the frozen ImageNet ResNet-18, so no retraining is needed, only memory_bank.npy + meta.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def fused_module(model_dir: Path):
    """uint8 NHWC RGB (1, S, S, 3) → patch-distance map (1, h, w): normalisation + ResNet-18 + kNN in one module.
    Shared by the ONNX (here) and ExecuTorch (scripts/export_executorch_vision.py) exports."""
    import torch
    import torch.nn as nn

    from train_anomaly import FeatureExtractor

    class FusedPatchCore(nn.Module):
        def __init__(self, bank: np.ndarray):
            super().__init__()
            self.fe = FeatureExtractor(model_dir / "projection.npz").eval()
            b = torch.from_numpy(bank.astype(np.float32))
            self.register_buffer("bank_t", b.t().contiguous())                # (C, M)
            self.register_buffer("bank_sq", (b ** 2).sum(1)[None, :])         # (1, M)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

        def forward(self, x):                                                  # (1, S, S, 3) uint8 RGB
            x = (x.permute(0, 3, 1, 2).float() - self.mean) / self.std
            f = self.fe(x)                                                     # (1, C, h, w)
            h, w = f.shape[-2:]
            f = f.flatten(2)[0].t()                                            # (P, C)
            d2 = (f * f).sum(1, keepdim=True) - 2.0 * (f @ self.bank_t) + self.bank_sq
            d = torch.sqrt(torch.clamp(d2.min(1).values, min=0.0))
            return d.view(1, h, w)

    return FusedPatchCore(np.load(model_dir / "memory_bank.npy")).eval()


def build_fused(model_dir: Path, size: int) -> Path:
    import torch

    m = fused_module(model_dir)
    out = model_dir / "patchcore_fused.onnx"
    dummy = torch.zeros(1, size, size, 3, dtype=torch.uint8)
    kw = {"dynamo": False} if "dynamo" in torch.onnx.export.__code__.co_varnames else {}
    torch.onnx.export(m, dummy, out, input_names=["image"], output_names=["dist"], opset_version=17, **kw)
    return out


def build_int8(fused: Path, calib: list[Path], size: int) -> Path:
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(paths_to_inputs(calib, size))

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"image": x}

    import onnx

    pre = fused.with_name("_pre.onnx")
    quant_pre_process(str(fused), str(pre), skip_symbolic_shape=True)
    # The PCA projection (1x1 conv, if any) stays FP32, like the kNN: it maps straight into distances.
    proj_nodes = [n.name for n in onnx.load(str(pre)).graph.node if "/proj/" in n.name]
    out = fused.with_name("patchcore_fused_int8.onnx")
    quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv"], nodes_to_exclude=proj_nodes, calibrate_method=CalibrationMethod.Percentile,
                    extra_options={"CalibPercentile": 99.99})
    pre.unlink(missing_ok=True)
    return out


def paths_to_inputs(paths, size):
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        yield rgb[None]


def recalibrate(model_path: Path, model_dir: Path, holdout: list[Path], threads: int) -> Path:
    """Re-derive thresholds for this model variant on the SAME held-out good images and with the same rule
    as train_anomaly.py (INT8 shifts the distance scale slightly, so FP32 thresholds don't transfer)."""
    from edgeqc.vision.anomaly import PatchCoreDetector
    from edgeqc.vision.backends import make_backend

    meta = json.loads((model_dir / "meta.json").read_text())
    size, sigma = meta.get("input_size", 256), meta.get("blur_sigma", 4.0)
    det = PatchCoreDetector(make_backend("auto", model_path, threads), model_dir / "memory_bank.npy",
                            model_dir / "meta.json", size)
    dists = [det.distance_map(cv2.imread(str(p), cv2.IMREAD_COLOR)) for p in holdout]
    k = int(sigma * 4) | 1
    heats = [cv2.GaussianBlur(cv2.resize(d.astype(np.float32), (size, size)), (k, k), sigma) for d in dists]
    pix = np.concatenate([h.ravel() for h in heats])
    new = {**{k_: v for k_, v in meta.items() if k_ not in ("operating_points",)},
           "image_threshold": float(max(d.max() for d in dists) * 1.05),
           "image_threshold_smooth": float(max(h.max() for h in heats) * 1.05),
           "pixel_threshold": float(np.percentile(pix, 99.9) * 1.05),
           "calibrated_for": model_path.name, "holdout_images": len(holdout)}
    out = model_dir / f"meta_{model_path.stem}.json"
    out.write_text(json.dumps(new, indent=2))
    return out


def evaluate(model_path: Path, model_dir: Path, test: list[Path], threads: int, meta_path: Path) -> dict:
    from sklearn.metrics import roc_auc_score

    from edgeqc.vision.anomaly import PatchCoreDetector
    from edgeqc.vision.backends import make_backend

    meta = json.loads(meta_path.read_text())
    det = PatchCoreDetector(make_backend("auto", model_path, threads), model_dir / "memory_bank.npy",
                            meta_path, meta.get("input_size", 256))
    imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in test]
    for im in imgs[:5]:
        det(im)                                                                # warm-up
    scores, lat = [], []
    for im in imgs:
        t = time.perf_counter()
        scores.append(det(im).score)
        lat.append((time.perf_counter() - t) * 1e3)
    y = np.array([0 if p.parent.name == "good" else 1 for p in test])
    s = np.array(scores)
    return {"image_auroc": round(float(roc_auc_score(y, s)), 4),
            "recall@thr": round(float((s[y == 1] >= 1.0).mean()), 3),
            "fpr@thr": round(float((s[y == 0] >= 1.0).mean()), 3),
            "recall@0.9thr": round(float((s[y == 1] >= 0.9).mean()), 3),
            "fpr@0.9thr": round(float((s[y == 0] >= 0.9).mean()), 3),
            "p50_ms": round(statistics.median(lat), 2),
            "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "size_mb": round(model_path.stat().st_size / 1e6, 1),
            "image_threshold": round(meta["image_threshold"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--category-dir", required=True, help="needs test/ (eval) and train/good (INT8 calibration)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--calib", type=int, default=20)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--holdout-dir", help="held-out good images (see holdout.json) → per-variant thresholds")
    args = ap.parse_args()

    md, cd = Path(args.model_dir), Path(args.category_dir)
    size = json.loads((md / "meta.json").read_text()).get("input_size", 256)
    calib = sorted(p for p in (cd / "train" / "good").glob("*") if p.suffix.lower() in EXTS)[:args.calib]
    test = sorted(p for p in (cd / "test").rglob("*") if p.suffix.lower() in EXTS)

    if not args.skip_build:
        fused = build_fused(md, size)
        build_int8(fused, calib, size)

    holdout = sorted(p for p in Path(args.holdout_dir).glob("*") if p.suffix.lower() in EXTS) \
        if args.holdout_dir else []
    rows = {}
    for name in ("feature_extractor.onnx", "patchcore_fused.onnx", "patchcore_fused_int8.onnx",
                 "patchcore_fused_et.pte", "patchcore_fused_et_int8.pte"):
        if (md / name).exists():
            meta_path = recalibrate(md / name, md, holdout, args.threads) if holdout else md / "meta.json"
            rows[name] = evaluate(md / name, md, test, args.threads, meta_path)
            print(f"{md.name:10s} {name:28s} {rows[name]}", flush=True)
    (md / "optimization.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
