#!/usr/bin/env python3
"""Export the vision models to ExecuTorch (.pte, XNNPACK delegate): the same Arm runtime as the Llama advisor.

    # PatchCore (fused: normalisation + ResNet-18 + kNN), FP32 and INT8 (backbone only, PT2E static per-channel)
    .venv-et/bin/python scripts/export_executorch_vision.py patchcore --model-dir models/anomaly/metal_nut \
        --category-dir datasets/mvtec/metal_nut
    # YOLOv8 detector from the trained PyTorch weights
    .venv-et/bin/python scripts/export_executorch_vision.py yolo --weights runs/.../best.pt --imgsz 320 \
        --calib-dir datasets/neu_yolo/images/train --out models/classifier/yolov8n_neu

Run in the ExecuTorch env (.venv-et, has torch + executorch + torchao). The .pte files run anywhere the
`executorch` pip package (runtime) is installed, e.g. the Raspberry Pi; edgeqc picks ExecuTorchBackend by the
.pte extension. A sidecar <name>.json records the output names the detectors look at.
Thresholds for the new PatchCore variants are re-calibrated with scripts/optimize_patchcore.py --skip-build
(it evaluates every patchcore_*.onnx/.pte in the model dir).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def lower_and_save(model, example: tuple, out: Path, output_names: list[str]) -> Path:
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    ep = torch.export.export(model, example)
    prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    out.write_bytes(prog.buffer)
    out.with_suffix(".json").write_text(json.dumps({"output_names": output_names}))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return out


def quantize(model, example: tuple, calib: list[tuple]):
    """PT2E static INT8 of a whole module (XNNPACK quantizer, symmetric per-channel weights)."""
    import torch
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (XNNPACKQuantizer,
                                                                          get_symmetric_quantization_config)
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    q = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
    gm = prepare_pt2e(torch.export.export(model, example).module(), q)
    with torch.no_grad():
        for x in calib:
            gm(*x)
    return convert_pt2e(gm)


def patchcore(args):
    import torch

    from optimize_patchcore import fused_module, paths_to_inputs

    md, cd = Path(args.model_dir), Path(args.category_dir)
    size = json.loads((md / "meta.json").read_text()).get("input_size", 256)
    m = fused_module(md)
    example = (torch.zeros(1, size, size, 3, dtype=torch.uint8),)
    lower_and_save(m, example, md / "patchcore_fused_et.pte", ["dist"])
    # INT8: quantise only the backbone (its own graph, calibrated on normalised good images), then put it back
    # in place of m.fe. Normalisation and the kNN distances (mul/sub/matmul/min/sqrt) stay FP32, as in ONNX.
    calib_paths = sorted(p for p in (cd / "train" / "good").glob("*") if p.suffix.lower() in EXTS)[:args.calib]
    norm = lambda u8: (u8.permute(0, 3, 1, 2).float() - m.mean) / m.std        # noqa: E731
    calib = [(norm(torch.from_numpy(x)),) for x in paths_to_inputs(calib_paths, size)]
    m.fe = quantize(m.fe, (norm(example[0]),), calib)
    lower_and_save(m, example, md / "patchcore_fused_et_int8.pte", ["dist"])


def yolo(args):
    import torch
    from ultralytics import YOLO

    from edgeqc.vision.classifier import letterbox

    det = YOLO(args.weights).model.float().eval().fuse()
    head = det.model[-1]
    head.export, head.format = True, "executorch"     # export-mode output: (1, 4 + nc, anchors), like the ONNX

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            y = self.m(x)
            return y[0] if isinstance(y, (list, tuple)) else y

    m = Wrap(det).eval()
    s = args.imgsz
    example = (torch.zeros(1, 3, s, s),)
    out = Path(args.out)
    lower_and_save(m, example, out.with_name(out.name + "_et.pte"), ["output0"])
    imgs = sorted(p for p in Path(args.calib_dir).glob("*") if p.suffix.lower() in EXTS)
    rng = np.random.default_rng(0)
    calib = []
    for p in [imgs[i] for i in rng.choice(len(imgs), min(args.calib, len(imgs)), replace=False)]:
        img, *_ = letterbox(cv2.imread(str(p)), s)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        calib.append((torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0),))
    qm = quantize(m, example, calib)
    lower_and_save(qm, example, out.with_name(out.name + "_et_int8.pte"), ["output0"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("patchcore")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--category-dir", required=True, help="needs train/good for INT8 calibration")
    p.add_argument("--calib", type=int, default=20)
    y = sub.add_parser("yolo")
    y.add_argument("--weights", required=True)
    y.add_argument("--imgsz", type=int, default=320)
    y.add_argument("--calib-dir", required=True)
    y.add_argument("--calib", type=int, default=100)
    y.add_argument("--out", required=True, help="output path stem, e.g. models/classifier/yolov8n_neu")
    args = ap.parse_args()
    {"patchcore": patchcore, "yolo": yolo}[args.cmd](args)


if __name__ == "__main__":
    main()
