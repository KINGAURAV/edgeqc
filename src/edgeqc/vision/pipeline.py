"""Vision pipeline: anomaly (A) + classifier (B), then fusion, severity and verdict."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from ..config import resolve_path
from ..types import InspectionResult
from .anomaly import AnomalyOutput, ClassicalAnomalyDetector, PatchCoreDetector
from .backends import make_backend
from .classifier import ShapeTyper, YoloDetector
from .fusion import fuse
from .severity import SeverityEngine

log = logging.getLogger(__name__)


def _backend(v: dict, model_path: str):
    return make_backend(v["backend"], resolve_path(model_path), v.get("num_threads", 4),
                        v.get("ort_spinning", False), int(v.get("boost_threads", 0) or 0))


class InspectionPipeline:
    def __init__(self, cfg: dict):
        v = cfg["vision"]
        self.product = cfg["app"]["product"]
        self.anomaly = self._build_anomaly(v)
        self.classifier = self._build_classifier(v)
        self.typer = ShapeTyper()
        self.severity = SeverityEngine(cfg["severity"])
        self.iou_match = float(v["fusion"]["iou_match"])
        self.frame_id = 0
        self.last_anomaly: Optional[AnomalyOutput] = None

    @staticmethod
    def _build_anomaly(v: dict):
        a = v["anomaly"]
        if a["method"] == "patchcore":
            try:
                backend = _backend(v, a["model_path"])
                det = PatchCoreDetector(backend, resolve_path(a["memory_bank"]), resolve_path(a["meta"]),
                                        a["input_size"], a["min_region_area_frac"],
                                        a.get("threshold_scale", 1.0), a.get("image_gate", 1.0))
                log.info("Anomaly: PatchCore (%s)", type(backend).__name__)
                return det
            except FileNotFoundError as e:
                log.warning("PatchCore artefacts missing (%s); falling back to classical detector", e)
        log.info("Anomaly: classical OpenCV detector")
        return ClassicalAnomalyDetector(a["input_size"], a["classical_z_threshold"],
                                        a["classical_bg_kernel"], a["min_region_area_frac"])

    @staticmethod
    def _build_classifier(v: dict) -> Optional[YoloDetector]:
        c = v["classifier"]
        if not c.get("enabled"):
            log.info("Classifier: disabled (anomaly-only mode)")
            return None
        try:
            backend = _backend(v, c["model_path"])
        except FileNotFoundError as e:
            log.warning("Classifier model missing (%s); running anomaly-only", e)
            return None
        log.info("Classifier: YOLO (%s), %d classes", type(backend).__name__, len(c["labels"]))
        # A static model input size wins over the config (models are exported for one size; see quantize_yolo.py)
        shape = getattr(backend, "input_shape", None)
        size = shape[-1] if shape and isinstance(shape[-1], int) else c["input_size"]
        return YoloDetector(backend, c["labels"], size, c["conf_threshold"], c["iou_threshold"],
                            c.get("class_conf_thresholds"))

    @property
    def can_boost(self) -> bool:
        return any(getattr(b, "boost_session", None) is not None for b in self._backends())

    def _backends(self):
        return [m.backend for m in (self.anomaly, self.classifier) if m is not None and hasattr(m, "backend")]

    def set_boost(self, on: bool) -> None:
        """Use the multi-thread sessions (vision.boost_threads) while the Llama advisor is idle."""
        for b in self._backends():
            if hasattr(b, "boost"):
                b.boost = on

    def inspect(self, bgr: np.ndarray) -> InspectionResult:
        t0 = time.perf_counter()
        an = self.anomaly(bgr)
        t1 = time.perf_counter()
        dets = self.classifier(bgr) if self.classifier else []
        t2 = time.perf_counter()
        defects = fuse(an.regions, dets, bgr.shape[:2], self.typer, self.iou_match)
        verdict = self.severity.apply(defects)
        t3 = time.perf_counter()
        self.frame_id += 1
        self.last_anomaly = an
        return InspectionResult(
            frame_id=self.frame_id,
            timestamp=time.time(),
            product=self.product,
            verdict=verdict,
            image_score=an.score,
            defects=defects,
            latency_ms={"anomaly": (t1 - t0) * 1e3, "classifier": (t2 - t1) * 1e3,
                        "fusion_severity": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3},
            image_hw=bgr.shape[:2],
        )
