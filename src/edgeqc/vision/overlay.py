"""Draw inspection results onto frames for the dashboard and demo video."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..types import InspectionResult, Severity, Verdict

SEV_COLORS = {Severity.MINOR: (0, 215, 255), Severity.MAJOR: (0, 120, 255), Severity.CRITICAL: (0, 0, 230)}
VERDICT_COLORS = {Verdict.PASS: (60, 170, 60), Verdict.REVIEW: (0, 190, 255), Verdict.REJECT: (40, 40, 220)}


def render(bgr: np.ndarray, result: InspectionResult, heatmap: Optional[np.ndarray] = None,
           fps: Optional[float] = None, max_side: int = 0) -> np.ndarray:
    """max_side > 0: draw on a copy downscaled to that size (boxes are scaled). Drawing on a 1024 px frame
    costs ~50 ms on a Pi 4; the dashboard and the saved evidence image don't need more than ~640 px."""
    s = max_side / max(bgr.shape[:2]) if max_side and max(bgr.shape[:2]) > max_side else 1.0
    out = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else bgr.copy()
    if heatmap is not None:
        if heatmap.shape[:2] != out.shape[:2]:
            heatmap = cv2.resize(heatmap, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
        h = np.clip(heatmap / 2.0, 0, 1)  # 1.0 threshold → mid colour
        colored = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
        mask = (heatmap >= 0.6)[..., None]
        out = np.where(mask, cv2.addWeighted(out, 0.55, colored, 0.45, 0), out)

    scale = max(0.4, min(out.shape[:2]) / 600)
    thick = max(1, int(round(scale * 2)))
    for d in result.defects:
        x, y, w, h = (int(round(v * s)) for v in d.bbox)
        color = SEV_COLORS[d.severity]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thick)
        tag = f"{d.label} {d.severity.value[:3]}"
        if d.confidence:
            tag += f" {d.confidence:.2f}"
        cv2.putText(out, tag, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, color, thick)

    banner_h = int(28 * scale)
    cv2.rectangle(out, (0, 0), (out.shape[1], banner_h), VERDICT_COLORS[result.verdict], -1)
    text = f"{result.verdict.value}  score {result.image_score:.2f}  {result.latency_ms.get('total', 0):.0f} ms"
    if fps:
        text += f"  {fps:.1f} FPS"
    cv2.putText(out, text, (6, int(banner_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), thick)
    return out
