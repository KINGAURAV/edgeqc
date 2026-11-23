"""Stage A: unsupervised anomaly detection.

Both detectors return an AnomalyOutput whose heatmap is normalised so that
1.0 == the calibrated decision threshold. Regions are connected components
above the threshold, in original-image coordinates.

- ClassicalAnomalyDetector: OpenCV-only, no training. Robust z-score of local deviation
  from a median-filtered background. Works for uniform textures (steel, fabric, film).
- PatchCoreDetector: PatchCore-lite. A CNN backbone (ONNX/HEF) gives patch features,
  and the anomaly score is the kNN distance to a coreset memory bank of good-part
  features (scripts/train_anomaly.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..types import Region
from .backends import InferenceBackend, preprocess


@dataclass
class AnomalyOutput:
    score: float  # image-level, 1.0 == threshold
    heatmap: np.ndarray  # float32 (h, w) at the detector's working resolution, 1.0 == threshold
    #                      (region bboxes are already in original-image coordinates; resize the heatmap for display)
    regions: list[Region] = field(default_factory=list)


def extract_regions(heatmap: np.ndarray, min_area_frac: float,
                    out_hw: tuple[int, int] | None = None) -> list[Region]:
    """Connected components of heatmap >= 1. If out_hw is given, boxes are scaled to that image size."""
    h, w = heatmap.shape
    sy, sx = (out_hw[0] / h, out_hw[1] / w) if out_hw else (1.0, 1.0)
    mask = (heatmap >= 1.0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area / (h * w) < min_area_frac:
            continue
        peak = float(heatmap[labels == i].max())
        box = (int(x * sx), int(y * sy), max(1, int(round(bw * sx))), max(1, int(round(bh * sy))))
        regions.append(Region(bbox=box, score=peak, area_frac=bw * bh / (h * w)))
    return regions


class ClassicalAnomalyDetector:
    def __init__(self, input_size: int = 256, z_threshold: float = 7.0, bg_kernel: int = 31,
                 min_region_area_frac: float = 0.0008):
        self.size = input_size
        self.z_threshold = z_threshold
        self.bg_kernel = bg_kernel | 1  # must be odd
        self.min_area = min_region_area_frac

    def __call__(self, bgr: np.ndarray) -> AnomalyOutput:
        H, W = bgr.shape[:2]
        gray = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        bg = cv2.medianBlur(gray, self.bg_kernel)
        diff = cv2.GaussianBlur(np.abs(gray.astype(np.float32) - bg.astype(np.float32)), (5, 5), 0)
        med = float(np.median(diff))
        mad = float(np.median(np.abs(diff - med))) + 1e-3
        z = (diff - med) / (1.4826 * mad)
        heat = (z / self.z_threshold).astype(np.float32)
        score = float(np.percentile(heat, 99.9))
        return AnomalyOutput(score=score, heatmap=heat, regions=extract_regions(heat, self.min_area, (H, W)))


class PatchCoreDetector:
    def __init__(self, backend: InferenceBackend, memory_bank: str | Path, meta: str | Path,
                 input_size: int = 256, min_region_area_frac: float = 0.0008,
                 threshold_scale: float = 1.0, image_gate: float | None = 1.0):
        """image_gate: report regions only if image_score >= image_gate (None = always). The pixel threshold
        alone flagged 21-54% of good MVTec test parts (a few pixels just over it); the image threshold is the
        holdout-calibrated operating point the recall/FPR numbers are measured at."""
        self.image_gate = image_gate
        self.backend = backend
        self.size = input_size
        self.min_area = min_region_area_frac
        # Fused models (scripts/optimize_patchcore.py) take uint8 RGB and output the patch-distance map
        # directly (normalisation + kNN inside the graph), so the memory bank isn't needed here.
        self.fused = getattr(backend, "output_names", [None])[0] == "dist"
        if not self.fused:
            self.bank = np.load(memory_bank).astype(np.float32)  # (M, C)
            self.bank_sq = (self.bank ** 2).sum(1)[None, :]
        with open(meta) as f:
            m = json.load(f)
        # threshold_scale < 1 trades false alarms for fewer misses (operating point; see meta "operating_points")
        self.pixel_threshold = float(m["pixel_threshold"]) * threshold_scale
        self.image_threshold = float(m["image_threshold"]) * threshold_scale
        # Image score = max of the SMOOTHED heatmap when the meta has a threshold for it (image_threshold_smooth,
        # same holdout rule). One noisy patch can't set it, unlike the raw max patch distance: on the MVTec test
        # sets (Pi models) AUROC 0.965 -> 0.981 for screw and 0.911 -> 0.954 for grid. Older metas: raw max.
        smooth = m.get("image_threshold_smooth")
        self.smooth_threshold = float(smooth) * threshold_scale if smooth else None
        self.blur_sigma = float(m.get("blur_sigma", 4.0))
        # The model was trained/exported/calibrated at a fixed resolution; that wins over the config value.
        self.size = int(m.get("input_size", self.size))

    def _patch_distances(self, feats: np.ndarray) -> np.ndarray:
        if feats.ndim == 4:
            feats = feats[0]
        if self.backend.raw_uint8_nhwc:  # Hailo returns HWC
            feats = feats.transpose(2, 0, 1)
        c, h, w = feats.shape
        f = feats.reshape(c, -1).T.astype(np.float32)  # (P, C)
        d2 = (f ** 2).sum(1, keepdims=True) - 2.0 * f @ self.bank.T + self.bank_sq
        return np.sqrt(np.maximum(d2.min(1), 0.0)).reshape(h, w)

    def distance_map(self, bgr: np.ndarray) -> np.ndarray:
        """Raw nearest-neighbour patch distances (h, w), before thresholds are applied."""
        if self.fused:
            return self.backend.run(preprocess(bgr, self.size, raw_uint8_nhwc=True))[0][0]
        x = preprocess(bgr, self.size, self.backend.raw_uint8_nhwc)
        return self._patch_distances(self.backend.run(x)[0])

    def __call__(self, bgr: np.ndarray) -> AnomalyOutput:
        H, W = bgr.shape[:2]
        dist = self.distance_map(bgr)
        # Same resolution and blur as the threshold calibration in train_anomaly.py
        heat = cv2.resize(dist.astype(np.float32), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        k = int(self.blur_sigma * 4) | 1
        heat = cv2.GaussianBlur(heat, (k, k), self.blur_sigma)
        if self.smooth_threshold is not None:
            image_score = float(heat.max()) / self.smooth_threshold
        else:
            image_score = float(dist.max()) / self.image_threshold
        heat = heat / self.pixel_threshold
        if self.image_gate is not None and image_score < self.image_gate:
            return AnomalyOutput(score=image_score, heatmap=heat)  # heatmap still shown, no defect reported
        return AnomalyOutput(score=image_score, heatmap=heat, regions=extract_regions(heat, self.min_area, (H, W)))
