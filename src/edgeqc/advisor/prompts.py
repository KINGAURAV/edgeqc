"""Prompt construction for the Llama 3.x Instruct advisor.

The LLM receives *structured findings* and *knowledge-base notes* for the detected
classes (grounding), never the raw image. The pass/fail decision has already been made by
the severity engine, so the LLM explains it and recommends actions.
"""

from __future__ import annotations

from ..types import InspectionResult

SYSTEM_PROMPT = (
    "You are an on-device quality-control assistant on a factory floor. "
    "You receive machine-vision inspection findings and reference notes. "
    "Reply with exactly three short lines and nothing else:\n"
    "Cause: <the most likely cause, chosen from the reference notes>\n"
    "Action: <immediate action for this part>\n"
    "Check: <process check to prevent recurrence>\n"
    "Write in plain words. Do not copy the findings fields (no 'type=', 'area='). "
    "Use only the reference notes and findings. Do not invent measurements. Do not change the verdict."
)

# One worked example (a class not used in the smoke tests) so the 1B model learns the style.
EXAMPLE_USER = (
    "Reference notes:\n- patches: Irregular dull regions on the strip surface. Causes: uneven pickling; "
    "residual oil or emulsion. Actions: check pickling bath concentration; inspect roll coolant filters.\n\n"
    "Findings:\nProduct: steel_strip\nVerdict: REVIEW\nAnomaly score: 1.60 (1.00 = threshold)\n"
    "Defect 1: type=patches, severity=MINOR, area=1.2% of part, detected_by=classifier, "
    "location=bottom-right"
)
EXAMPLE_ASSISTANT = (
    "Cause: Likely uneven pickling leaving a dull patch on the surface.\n"
    "Action: Hold the part for manual review and confirm the patch visually.\n"
    "Check: Check pickling bath concentration and roll coolant filters."
)

# The 'Log:' line holds only facts the app already has (verdict, severity, type, zone), so it is written by
# log_line() instead of being generated: ~20% fewer decoded tokens on the Pi and an exact log entry.
REQUIRED_FIELDS = ("Cause:", "Action:")
RAW_DUMP_MARKERS = ("type=", "severity=", "area=", "detected_by=", "location=", "confidence=", "defect 1:")


def split_log_entry(text: str) -> tuple[str, str | None]:
    """Separate the 'Log:' line (stored as the log entry) from the operator advice."""
    advice, log_entry = [], None
    for line in text.splitlines():
        if line.strip().lower().startswith("log:") and log_entry is None:
            log_entry = line.split(":", 1)[1].strip()
        else:
            advice.append(line)
    return "\n".join(advice).strip(), log_entry


def is_grounded(reply: str, result: InspectionResult) -> bool:
    """Cheap guard that the reply follows the format and refers to at least one detected defect type."""
    reply, _ = split_log_entry(reply)  # judge only the model's own text, not the appended log line
    low = reply.lower()
    if not all(f.lower() in low for f in REQUIRED_FIELDS):
        return False
    if any(m in low for m in RAW_DUMP_MARKERS):  # parroted the findings instead of advising
        return False
    text = reply.lower().replace("_", " ").replace("-", " ")
    names = {d.label.lower().replace("_", " ").replace("-", " ") for d in result.defects}
    # "anomaly" alone is too generic, unless the finding IS an unclassified anomaly (anomaly_spot etc.)
    words = {w for n in names for w in n.split() if len(w) > 3 and (w != "anomaly" or n.startswith("anomaly"))}
    # stems too ("patches" -> "patch", "scratches" -> "scratch"): the model writes "a patch", "patching"
    stems = {w[:-2] if w.endswith("es") and len(w) > 6 else w.rstrip("s") for w in words}
    return any(n in text for n in names) or any(w in text for w in words | stems) or not names


def describe_location(bbox, image_hw) -> str:
    """Coarse 3x3 zone name plus pixel box, e.g. 'top-left (x=12,y=40,w=80,h=6)'."""
    x, y, w, h = bbox
    H, W = image_hw
    cx, cy = (x + w / 2) / W, (y + h / 2) / H
    row = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "middle")
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    zone = "center" if (row, col) == ("middle", "center") else f"{row}-{col}"
    return f"{zone} (x={x},y={y},w={w},h={h})"


def log_line(result: InspectionResult) -> str:
    """Deterministic inspection-log sentence (shared by the LLM and rule-based advisors)."""
    if not result.defects:
        return f"{result.verdict.value} - no defects, anomaly score {result.image_score:.2f}."
    worst = sorted(result.defects, key=lambda d: (-d.severity.rank, -d.area_frac))[0]
    where = describe_location(worst.bbox, result.image_hw).split(" (")[0] if result.image_hw else "on part"
    others = f" (+{len(result.defects) - 1} more)" if len(result.defects) > 1 else ""
    return (f"{result.verdict.value} - {worst.severity.value} {worst.label} at {where}{others}, "
            f"anomaly score {result.image_score:.2f}.")


def summarize_findings(result: InspectionResult, max_defects: int = 4) -> str:
    lines = [f"Product: {result.product}", f"Verdict: {result.verdict.value}",
             f"Anomaly score: {result.image_score:.2f} (1.00 = threshold)"]
    for i, d in enumerate(sorted(result.defects, key=lambda d: -d.severity.rank)[:max_defects], 1):
        lines.append(
            f"Defect {i}: type={d.label}, severity={d.severity.value}, "
            f"area={d.area_frac * 100:.1f}% of part, detected_by={d.source}"
            # zone only: pixel boxes and confidences cost ~20 prompt tokens per defect and the model must not
            # repeat them anyway
            + (f", location={describe_location(d.bbox, result.image_hw).split(' (')[0]}" if result.image_hw else "")
        )
    return "\n".join(lines)


def kb_notes(result: InspectionResult, kb: dict) -> str:
    seen, notes = set(), []
    for d in result.defects:
        if d.label in seen or d.label not in kb:
            continue
        seen.add(d.label)
        e = kb[d.label]
        notes.append(f"- {d.label}: {e.get('description', '')} Causes: {'; '.join(e.get('causes', []))}. "
                     f"Actions: {'; '.join(e.get('actions', []))}.")
    return "\n".join(notes) if notes else "- (no reference notes for these defect types)"


def llama3_static_prefix() -> str:
    """The part of every advice prompt that never changes (system prompt + worked example), ~75% of the
    tokens. llama_server prefills it into the KV cache once and reuses it, so each request only prefills
    the findings. It ends on a special token (<|eot_id|>), so tokenizing prefix and rest separately gives
    the same tokens as tokenizing the whole prompt."""
    return (
        "<|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM_PROMPT + "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n" + EXAMPLE_USER + "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n" + EXAMPLE_ASSISTANT + "<|eot_id|>"
    )


def llama3_user_turn(user: str) -> str:
    return ("<|start_header_id|>user<|end_header_id|>\n\n" + user + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")


def llama3_prompt_parts(result: InspectionResult, kb: dict) -> tuple[str, str]:
    """(head, tail) of the advice prompt. head = static prefix + the reference notes for the detected defect
    classes: the same for consecutive defects of the same class, so llama_server keeps its KV cache (second
    cache level) and only prefills tail = the per-frame findings. Notes come before findings for that reason."""
    head = (llama3_static_prefix() + "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Reference notes:\n{kb_notes(result, kb)}\n\n")
    tail = (f"Findings:\n{summarize_findings(result)}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")
    return head, tail


def build_llama3_prompt(result: InspectionResult, kb: dict) -> str:
    """Llama 3 chat template. <|begin_of_text|> is omitted here; the runner prepends BOS
    because we pass --num_bos=1 (ExecuTorch v1.5.1 llama_main defaults to num_bos=0)."""
    return "".join(llama3_prompt_parts(result, kb))
