"""Core data types shared by the vision, advisory, storage and UI layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

BBox = tuple[int, int, int, int]  # x, y, w, h in original image pixels


class Severity(str, Enum):
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return {"MINOR": 1, "MAJOR": 2, "CRITICAL": 3}[self.value]

    @staticmethod
    def max(a: "Severity", b: "Severity") -> "Severity":
        return a if a.rank >= b.rank else b


class Verdict(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


@dataclass
class Region:
    """Anomalous region from the anomaly detector."""

    bbox: BBox
    score: float  # peak anomaly strength in region, 1.0 == calibrated threshold
    area_frac: float


@dataclass
class Detection:
    """Typed box from the defect classifier."""

    label: str
    confidence: float
    bbox: BBox


@dataclass
class Defect:
    label: str
    bbox: BBox
    confidence: float  # classifier confidence (0 for anomaly-only)
    anomaly_score: float  # 0 if classifier-only
    area_frac: float
    source: str  # fused | classifier | anomaly
    severity: Severity = Severity.MINOR

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class InspectionResult:
    frame_id: int
    timestamp: float
    product: str
    verdict: Verdict
    image_score: float
    defects: list[Defect] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    advice: Optional[str] = None
    inspection_id: Optional[int] = None  # set by storage
    image_hw: Optional[tuple[int, int]] = None  # (H, W) of the inspected frame

    @property
    def worst_severity(self) -> Optional[Severity]:
        if not self.defects:
            return None
        worst = self.defects[0].severity
        for d in self.defects[1:]:
            worst = Severity.max(worst, d.severity)
        return worst

    def to_dict(self) -> dict:
        return {
            "inspection_id": self.inspection_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "product": self.product,
            "verdict": self.verdict.value,
            "image_score": round(self.image_score, 3),
            "defects": [d.to_dict() for d in self.defects],
            "latency_ms": {k: round(v, 2) for k, v in self.latency_ms.items()},
            "advice": self.advice,
        }
