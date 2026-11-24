"""Deterministic fallback advisor built from the defect knowledge base.

Used when ExecuTorch artefacts are missing or the LLM fails or times out, and as a
baseline in the report when comparing it with the Llama advice.
"""

from __future__ import annotations

from ..types import InspectionResult
from .prompts import kb_entry, log_line


class RuleBasedAdvisor:
    name = "rule"

    def __init__(self, kb: dict):
        self.kb = kb

    def advise(self, result: InspectionResult) -> str:
        if not result.defects:
            return "No defects detected. Part OK."
        worst = sorted(result.defects, key=lambda d: (-d.severity.rank, -d.area_frac))[0]
        e = kb_entry(self.kb, result.product, worst.label) or {}
        causes = e.get("causes") or ["unknown"]
        actions = e.get("actions") or ["route part to manual inspection"]
        return (f"Cause: {worst.label} ({worst.severity.value}), likely {causes[0]}.\n"
                f"Action: {actions[0]}.\n"
                f"Check: {actions[1] if len(actions) > 1 else causes[-1]}.\n"
                f"Log: {log_line(result)}")
