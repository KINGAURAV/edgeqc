#!/usr/bin/env python3
"""Run the Llama/ExecuTorch advisor on canned inspection findings and print the prompt, reply and latency.

    python scripts/llm_smoke_test.py            # uses advisor.* paths from configs/default.yaml
    python scripts/llm_smoke_test.py --print-prompt-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgeqc.advisor.prompts import build_llama3_prompt, is_grounded  # noqa: E402
from edgeqc.advisor.service import build_advisors, load_kb  # noqa: E402
from edgeqc.config import load_config  # noqa: E402
from edgeqc.types import Defect, InspectionResult, Severity, Verdict  # noqa: E402

CASES = [
    InspectionResult(1, time.time(), "steel_strip", Verdict.REJECT, 2.7, [
        Defect("scratches", (40, 60, 180, 8), 0.82, 2.7, 0.03, "fused", Severity.MAJOR)]),
    # same class again, different frame: exercises llama_server's class-notes KV cache (second cache level)
    InspectionResult(4, time.time(), "steel_strip", Verdict.REJECT, 3.2, [
        Defect("scratches", (300, 120, 90, 6), 0.74, 3.2, 0.02, "fused", Severity.MAJOR)]),
    InspectionResult(2, time.time(), "steel_strip", Verdict.REVIEW, 1.3, [
        Defect("anomaly_spot", (100, 100, 12, 12), 0.0, 1.3, 0.002, "anomaly", Severity.MINOR)]),
    InspectionResult(3, time.time(), "steel_strip", Verdict.REJECT, 4.1, [
        Defect("crazing", (10, 10, 120, 120), 0.77, 4.1, 0.14, "fused", Severity.CRITICAL),
        Defect("inclusion", (200, 50, 20, 14), 0.55, 0.0, 0.003, "classifier", Severity.MAJOR)]),
]
for _c in CASES:
    _c.image_hw = (200, 400)  # example frame size so prompts include location


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--device", help="e.g. configs/devices/pi4.yaml (threads / core pinning)")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--print-prompt-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set, device=args.device)
    if args.print_prompt_only:
        print(build_llama3_prompt(CASES[0], load_kb(cfg["advisor"]["knowledge_base"])))
        return
    llm, rule = build_advisors(cfg)
    adv = llm or rule
    print(f"Advisor engine: {adv.name}")
    if hasattr(adv, "warm_up"):
        t = time.perf_counter()
        adv.warm_up()   # model load + static prefix prefill, as the app does at start-up
        print(f"warm-up {time.perf_counter() - t:.1f}s")
    for case in CASES:
        t = time.perf_counter()
        text = adv.advise(case)
        dt = time.perf_counter() - t
        ok = "grounded" if is_grounded(text, case) else "UNGROUNDED -> app would use rule fallback"
        st = getattr(getattr(adv, "server", None), "last_stats", {}) or {}
        tok = (f" — prompt {st['prompt_tokens'] - st['cached_prefix_tokens']} new tokens ({st['cached_prefix_tokens']} cached), {st['generated_tokens']} "
               f"generated, TTFT {st['ttft_ms'] / 1e3:.1f}s") if st else ""
        print(f"\n--- case {case.frame_id} ({case.verdict.value}) — {dt:.2f}s{tok} — {ok}\n{text}", flush=True)


if __name__ == "__main__":
    main()
