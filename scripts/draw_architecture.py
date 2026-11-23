#!/usr/bin/env python3
"""Draw docs/figures/architecture.png (report figure)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
VIS, LLM, APP = "#cde2fb", "#fde0d2", "#e9e8e4"
VIS_E, LLM_E, APP_E = "#2a78d6", "#eb6834", "#8a8984"

fig, ax = plt.subplots(figsize=(11, 5.0))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.set_xlim(0, 110); ax.set_ylim(-4, 46); ax.axis("off")


def box(x, y, w, h, text, fc, ec, size=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(x1, y1, x2, y2, c=INK2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3, shrinkA=2, shrinkB=2))


# lanes
ax.add_patch(FancyBboxPatch((14, 24.5), 62, 20, boxstyle="round,pad=0.3,rounding_size=1.5", fc="none", ec=VIS_E, lw=1, ls="--"))
ax.text(15.5, 42.6, "Vision pipeline · ONNX Runtime (INT8) · Arm CPU core 0 (+ cores 1-3 while Llama is idle)",
        fontsize=8, color=VIS_E, va="center")
ax.add_patch(FancyBboxPatch((40, -3), 68, 22.5, boxstyle="round,pad=0.3,rounding_size=1.5", fc="none", ec=LLM_E, lw=1, ls="--"))
ax.text(41.5, -1.4, "Advisory pipeline · ExecuTorch (XNNPACK, 8da4w) · Arm CPU cores 1-3", fontsize=8, color=LLM_E, va="center")

box(1, 29, 10, 9, "Camera /\nimage folder", APP, APP_E)
box(17, 33.5, 22, 6.5, "Anomaly detector\nPatchCore-lite / classical", VIS, VIS_E)
box(17, 26, 22, 6.5, "Defect classifier\nYOLOv8n (NEU-DET)", VIS, VIS_E)
box(44, 29, 13, 8.5, "Fusion +\nseverity rules", VIS, VIS_E)
box(61, 29, 13, 8.5, "PASS /\nREVIEW /\nREJECT", VIS, VIS_E, bold=True)
box(81, 33, 27, 8, "Dashboard (FastAPI)\nstream · alerts · advice · ack", APP, APP_E)
box(81, 23.5, 27, 7, "SQLite log + evidence images\nCSV export", APP, APP_E)

box(43, 4, 20, 10.5, "Advisor service\nsignature cache · queue\nidle pre-generation", LLM, LLM_E)
box(67, 4, 18, 10.5, "Prompt: findings +\nknowledge-base notes", LLM, LLM_E)
box(89, 4, 18, 10.5, "llama_server\nLlama 3.2 1B (.pte)\nKV cache · streaming", LLM, LLM_E, bold=False)

arrow(11, 35, 17, 36.5); arrow(11, 32, 17, 29.5)
arrow(39, 36.5, 44, 34); arrow(39, 29.5, 44, 32)
arrow(57, 33.2, 61, 33.2)
arrow(74, 35, 81, 37); arrow(74, 31.5, 81, 27.5)
arrow(67.5, 29, 55, 14.6, VIS_E)
ax.text(57, 21.5, "non-PASS\nfindings", fontsize=7.5, color=INK2, ha="center")
arrow(63, 9.2, 67, 9.2); arrow(85, 9.2, 89, 9.2)
arrow(98, 14.6, 98, 23.3, LLM_E)
ax.text(96.8, 21.6, "advice (grounding-checked)", fontsize=7.5, color=INK2, va="center", ha="right")
fig.tight_layout()
fig.savefig("docs/figures/architecture.png", dpi=180)
