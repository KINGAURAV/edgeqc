"""Fuse anomaly regions (Stage A) with classifier detections (Stage B) into defects."""

from __future__ import annotations

from ..types import BBox, Defect, Detection, Region
from .classifier import ShapeTyper


def iou(a: BBox, b: BBox) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def fuse(regions: list[Region], detections: list[Detection], image_hw: tuple[int, int],
         typer: ShapeTyper, iou_match: float = 0.1) -> list[Defect]:
    H, W = image_hw
    used: set[int] = set()
    defects: list[Defect] = []
    for det in detections:
        best_j, best_iou = -1, 0.0
        for j, reg in enumerate(regions):
            v = iou(det.bbox, reg.bbox)
            if v > best_iou:
                best_j, best_iou = j, v
        area = det.bbox[2] * det.bbox[3] / (H * W)
        if best_j >= 0 and best_iou >= iou_match:
            used.add(best_j)
            defects.append(Defect(det.label, det.bbox, det.confidence, regions[best_j].score, area, "fused"))
        else:
            defects.append(Defect(det.label, det.bbox, det.confidence, 0.0, area, "classifier"))
    for j, reg in enumerate(regions):
        if j not in used:
            defects.append(Defect(typer(reg), reg.bbox, 0.0, reg.score, reg.area_frac, "anomaly"))
    return defects
