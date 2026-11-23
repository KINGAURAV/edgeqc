#!/usr/bin/env python3
"""Evaluate the QC advisor on 20 scripted inspection findings (all 10 knowledge-base classes, single and multi-defect).

For every case it records the reply, latency, TTFT / token counts (llama_server) and automatic checks:
  format     - exactly the three lines Cause / Action / Check
  grounded   - the app's own guard (is_grounded); if False the app shows the rule-based advice instead
  names_type - mentions the worst defect type (or a word of it)
  cause_kb   - the Cause line restates one of the knowledge-base causes for the detected classes
  action_kb  - Action or Check restates one of the knowledge-base actions
  no_foreign - names no defect class that is absent from the prompt
  no_numbers - invents no number that is absent from the prompt
  verdict_ok - does not tell the operator to pass / release a REJECT or REVIEW part
The rule-based advisor is scored on the same checks as a baseline.

    python scripts/eval_advisor.py --device configs/devices/pi4.yaml --out runs/advisor_eval
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgeqc.advisor.prompts import build_llama3_prompt, is_grounded, split_log_entry  # noqa: E402
from edgeqc.advisor.rule_based import RuleBasedAdvisor  # noqa: E402
from edgeqc.advisor.service import build_advisors, load_kb  # noqa: E402
from edgeqc.config import load_config  # noqa: E402
from edgeqc.types import Defect, InspectionResult, Severity, Verdict  # noqa: E402

S, M, C = Severity.MINOR, Severity.MAJOR, Severity.CRITICAL
HW = (200, 400)


def _d(label, bbox, sev, area, src="fused", conf=0.7, score=2.0):
    return Defect(label, bbox, conf if src != "anomaly" else 0.0, score if src != "classifier" else 0.0, area, src, sev)


# (product, verdict, image score, defects)
CASES = [
    ("steel_strip", Verdict.REJECT, 4.1, [_d("crazing", (10, 10, 380, 180), C, 0.62)]),
    ("steel_strip", Verdict.REVIEW, 1.4, [_d("crazing", (20, 120, 90, 60), S, 0.07, "classifier")]),
    ("steel_strip", Verdict.REJECT, 2.9, [_d("inclusion", (200, 50, 20, 14), M, 0.004)]),
    ("steel_strip", Verdict.REVIEW, 1.2, [_d("inclusion", (330, 160, 12, 10), S, 0.002, "classifier")]),
    ("steel_strip", Verdict.REJECT, 3.3, [_d("patches", (40, 30, 160, 120), M, 0.24)]),
    ("steel_strip", Verdict.REVIEW, 1.5, [_d("patches", (300, 10, 80, 60), S, 0.06, "classifier")]),
    ("steel_strip", Verdict.REJECT, 3.0, [_d("pitted_surface", (150, 70, 90, 60), M, 0.07)]),
    ("steel_strip", Verdict.REVIEW, 1.3, [_d("pitted_surface", (20, 140, 40, 40), S, 0.02, "classifier")]),
    ("steel_strip", Verdict.REJECT, 3.6, [_d("rolled-in_scale", (0, 60, 400, 70), M, 0.35)]),
    ("steel_strip", Verdict.REVIEW, 1.2, [_d("rolled-in_scale", (250, 150, 60, 30), S, 0.02, "classifier")]),
    ("steel_strip", Verdict.REJECT, 2.7, [_d("scratches", (40, 60, 180, 8), M, 0.03)]),
    ("steel_strip", Verdict.REJECT, 4.4, [_d("scratches", (0, 90, 400, 10), C, 0.05)]),
    ("metal_nut", Verdict.REJECT, 5.2, [_d("crack", (120, 40, 30, 90), C, 0.03)]),
    ("bottle", Verdict.REJECT, 3.8, [_d("crack", (180, 20, 12, 70), C, 0.01)]),
    ("screw", Verdict.REJECT, 2.2, [_d("anomaly_linear", (60, 90, 120, 10), M, 0.015, "anomaly")]),
    ("bottle", Verdict.REVIEW, 1.3, [_d("anomaly_spot", (210, 110, 14, 12), S, 0.002, "anomaly")]),
    ("tile", Verdict.REJECT, 2.6, [_d("anomaly_area", (20, 20, 200, 150), M, 0.38, "anomaly")]),
    ("steel_strip", Verdict.REJECT, 4.0, [_d("crazing", (10, 10, 200, 180), C, 0.45),
                                          _d("inclusion", (300, 50, 20, 14), M, 0.003, "classifier")]),
    ("steel_strip", Verdict.REJECT, 2.8, [_d("scratches", (40, 160, 300, 8), M, 0.03),
                                          _d("anomaly_spot", (350, 20, 12, 12), S, 0.002, "anomaly")]),
    ("steel_strip", Verdict.REJECT, 3.1, [_d("pitted_surface", (60, 40, 80, 60), M, 0.06),
                                          _d("patches", (250, 100, 120, 90), S, 0.13, "classifier")]),
]

STOP = {"the", "and", "for", "with", "from", "into", "this", "that", "part", "check", "verify", "inspect",
        "unknown", "needs", "possible", "may", "after", "before", "later", "during", "affected", "be"}
PASS_WORDS = re.compile(r"\b(pass(es|ed)? the part|release the part|accept the part|ship|no action (is )?needed|"
                        r"part is (ok|fine|acceptable))\b")


def words(s: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", s.lower().replace("_", " ").replace("-", " "))
            if len(w) > 3 and w not in STOP}


def overlap(line: str, phrases: list[str], need: float = 0.5) -> bool:
    lw = words(line)
    return any(len(words(p)) and len(words(p) & lw) / len(words(p)) >= need for p in phrases)


def lines_of(reply: str) -> dict[str, str]:
    out = {}
    for ln in reply.splitlines():
        k = ln.split(":", 1)[0].strip().lower()
        if k in ("cause", "action", "check") and ":" in ln:
            out[k] = ln.split(":", 1)[1]
    return out


def score(reply: str, res: InspectionResult, kb: dict, prompt: str) -> dict:
    advice, _ = split_log_entry(reply)
    ls = lines_of(advice)
    body = [ln for ln in advice.splitlines() if ln.strip()]
    labels = [d.label for d in res.defects]
    causes = [c for lab in labels for c in kb.get(lab, {}).get("causes", [])]
    actions = [a for lab in labels for a in kb.get(lab, {}).get("actions", [])]
    worst = sorted(res.defects, key=lambda d: (-d.severity.rank, -d.area_frac))[0].label
    low = advice.lower().replace("_", " ").replace("-", " ")
    wl = worst.replace("_", " ").replace("-", " ")
    p_low = prompt.lower().replace("_", " ").replace("-", " ")
    foreign = [k for k in kb if not k.startswith("anomaly")
               and re.search(r"\b" + re.escape(k.replace("_", " ").replace("-", " ").rstrip("es").rstrip("s")), low)
               and k.replace("_", " ").replace("-", " ").rstrip("es").rstrip("s") not in p_low]
    nums = [n for n in re.findall(r"\d+(?:\.\d+)?", advice) if n not in prompt]
    return {
        "format": len(body) == 3 and set(ls) == {"cause", "action", "check"},
        "grounded": is_grounded(reply, res),
        "names_type": wl in low or any(w in low for w in wl.split() if len(w) > 3 and w != "anomaly"),
        "cause_kb": overlap(ls.get("cause", ""), causes),
        "action_kb": overlap(ls.get("action", "") + " " + ls.get("check", ""), actions),
        "no_foreign": not foreign,
        "no_numbers": not nums,
        "verdict_ok": not PASS_WORDS.search(advice.lower()),
        "_foreign": foreign, "_numbers": nums,
    }


CHECKS = ["format", "grounded", "names_type", "cause_kb", "action_kb", "no_foreign", "no_numbers", "verdict_ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--device")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default="runs/advisor_eval")
    ap.add_argument("--rule-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set, device=args.device)
    kb = load_kb(cfg["advisor"]["knowledge_base"])
    rule = RuleBasedAdvisor(kb)
    llm = None if args.rule_only else build_advisors(cfg)[0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    warm = None
    if llm is not None:
        t = time.perf_counter()
        llm.warm_up()
        warm = time.perf_counter() - t
        print(f"warm-up {warm:.1f}s", flush=True)

    rows = []
    for i, (prod, verdict, sc, defects) in enumerate(CASES, 1):
        res = InspectionResult(i, time.time(), prod, verdict, sc, defects)
        res.image_hw = HW
        prompt = build_llama3_prompt(res, kb)
        row = {"case": i, "product": prod, "verdict": verdict.value,
               "defects": [f"{d.label}/{d.severity.value}" for d in defects]}
        r_text = rule.advise(res)
        row["rule"] = {"text": r_text, **score(r_text, res, kb, prompt)}
        if llm is not None:
            t = time.perf_counter()
            text = llm.advise(res)
            dt = time.perf_counter() - t
            st = dict(getattr(getattr(llm, "server", None), "last_stats", {}) or {})
            row["llm"] = {"text": text, "latency_s": round(dt, 2), "stats": st, **score(text, res, kb, prompt)}
            print(f"[{i:2d}] {dt:5.1f}s ttft {st.get('ttft_ms', 0) / 1e3:4.1f}s "
                  f"gen {st.get('generated_tokens')} | " + " ".join(k for k in CHECKS if not row["llm"][k]) +
                  f"\n{text}\n", flush=True)
        rows.append(row)

    def summary(key):
        rs = [r[key] for r in rows if key in r]
        return {k: sum(bool(r[k]) for r in rs) for k in CHECKS} | {"n": len(rs)}

    summ = {"rule": summary("rule"), "warm_up_s": warm}
    if llm is not None:
        lat = sorted(r["llm"]["latency_s"] for r in rows)
        ttft = sorted(r["llm"]["stats"].get("ttft_ms", 0) / 1e3 for r in rows)
        gen = [r["llm"]["stats"].get("generated_tokens", 0) for r in rows]
        dec = [(r["llm"]["latency_s"] - r["llm"]["stats"].get("ttft_ms", 0) / 1e3) / max(r["llm"]["stats"].get("generated_tokens", 1) - 1, 1)
               for r in rows]
        summ["llm"] = summary("llm") | {
            "latency_median_s": lat[len(lat) // 2], "latency_max_s": lat[-1],
            "ttft_median_s": ttft[len(ttft) // 2], "ttft_min_s": ttft[0], "ttft_max_s": ttft[-1],
            "gen_tokens_mean": sum(gen) / len(gen), "decode_tok_per_s": round(1 / (sum(dec) / len(dec)), 2)}
    (out / "results.json").write_text(json.dumps({"summary": summ, "cases": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
