"""Stage B: supervised defect classifier.

YoloDetector runs a YOLOv8 detection model exported to ONNX (or HEF) and returns
typed boxes. ShapeTyper names anomaly regions the classifier did not explain,
using only their geometry.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..types import Detection, Region
from .backends import InferenceBackend


def letterbox(bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, left, top


class YoloDetector:
    def __init__(self, backend: InferenceBackend, labels: list[str], input_size: int = 320,
                 conf_threshold: float = 0.35, iou_threshold: float = 0.45,
                 class_conf: dict[str, float] | None = None):
        self.backend, self.labels = backend, labels
        self.size, self.conf, self.iou = input_size, conf_threshold, iou_threshold
        # Per-class confidence thresholds (default: conf_threshold). NEU crazing is detected but with low confidence
        # (median 0.33), so one global threshold misses it; see docs/RESULTS.md section 1.
        self.thr = np.array([(class_conf or {}).get(l, conf_threshold) for l in labels], np.float32)

    def __call__(self, bgr: np.ndarray) -> list[Detection]:
        img, r, padx, pady = letterbox(bgr, self.size)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.backend.raw_uint8_nhwc:
            x = rgb[None]
        else:  # Ultralytics ONNX: float32 NCHW in [0, 1]
            x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        out = self.backend.run(x)[0]
        return self.postprocess(out, r, padx, pady, bgr.shape[:2])

    def postprocess(self, out: np.ndarray, r: float, padx: int, pady: int, hw: tuple[int, int]) -> list[Detection]:
        pred = out[0]
        if pred.shape[0] == 4 + len(self.labels):  # (4+nc, N) → (N, 4+nc)
            pred = pred.T
        scores = pred[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.thr[cls]
        if not keep.any():
            return []
        pred, cls, conf = pred[keep], cls[keep], conf[keep]
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        H, W = hw
        x0 = np.clip((cx - bw / 2 - padx) / r, 0, W - 1)
        y0 = np.clip((cy - bh / 2 - pady) / r, 0, H - 1)
        x1 = np.clip((cx + bw / 2 - padx) / r, 0, W - 1)
        y1 = np.clip((cy + bh / 2 - pady) / r, 0, H - 1)
        boxes = np.stack([x0, y0, x1 - x0, y1 - y0], 1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), float(self.thr.min()), self.iou)
        idx = np.array(idx).reshape(-1)
        return [
            Detection(label=self.labels[int(cls[i])], confidence=float(conf[i]),
                      bbox=tuple(int(v) for v in boxes[i]))  # type: ignore[arg-type]
            for i in idx
        ]


class ShapeTyper:
    """Name an unexplained anomaly region by its geometry."""

    def __init__(self, linear_aspect: float = 3.5, area_frac: float = 0.02):
        self.linear_aspect, self.area_frac = linear_aspect, area_frac

    def __call__(self, region: Region) -> str:
        _, _, w, h = region.bbox
        aspect = max(w, h) / max(1, min(w, h))
        if aspect >= self.linear_aspect:
            return "anomaly_linear"
        if region.area_frac >= self.area_frac:
            return "anomaly_area"
        return "anomaly_spot"
