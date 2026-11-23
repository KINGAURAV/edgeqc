"""Transparent severity rules and verdict. The LLM never makes this decision."""

from __future__ import annotations

from ..types import Defect, Severity, Verdict


class SeverityEngine:
    def __init__(self, cfg: dict):
        self.area_major = float(cfg["area_frac"]["major"])
        self.area_critical = float(cfg["area_frac"]["critical"])
        self.score_major = float(cfg["score"]["major"])
        self.score_critical = float(cfg["score"]["critical"])
        self.base = {k: Severity(v) for k, v in cfg.get("class_criticality", {}).items()}
        self.default = Severity(cfg.get("default", "MINOR"))

    def grade(self, d: Defect) -> Severity:
        sev = self.base.get(d.label, self.default)
        if d.area_frac >= self.area_critical or d.anomaly_score >= self.score_critical:
            return Severity.CRITICAL
        if d.area_frac >= self.area_major or d.anomaly_score >= self.score_major:
            sev = Severity.max(sev, Severity.MAJOR)
        return sev

    def apply(self, defects: list[Defect]) -> Verdict:
        for d in defects:
            d.severity = self.grade(d)
        if any(d.severity.rank >= Severity.MAJOR.rank for d in defects):
            return Verdict.REJECT
        if defects:
            return Verdict.REVIEW
        return Verdict.PASS
