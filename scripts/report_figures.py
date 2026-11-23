#!/usr/bin/env python3
"""Report figures from the deployed app pipeline (same configs, models and thresholds as the app).

  steel:  image-level confusion matrix on the NEU-DET val split (true class vs the app's named worst defect,
          'PASS' = nothing reported)
  mvtec:  ROC of the image score per category (full MVTec AD test sets) with the app's operating point

    python scripts/report_figures.py --device configs/devices/pi4.yaml --out docs/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from edgeqc.config import load_config  # noqa: E402
from edgeqc.vision import InspectionPipeline  # noqa: E402

INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
NEU = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CATS = ["bottle", "metal_nut", "tile", "grid", "screw"]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "text.color": INK})


def cfg_for(profile: str, device: str | None, threads: int):
    cfg = load_config(profile, [f"vision.num_threads={threads}", "vision.boost_threads=0",
                                "vision.cpu_affinity=null"], device=device)
    return cfg


def steel(device, threads, out: Path) -> dict:
    p = InspectionPipeline(cfg_for("configs/profiles/steel.yaml", device, threads))
    cols = NEU + ["untyped", "PASS"]  # untyped = flagged by the anomaly detector only, no class named
    cm = np.zeros((len(NEU), len(cols)), int)
    for f in sorted(glob.glob("datasets/neu_yolo/images/val/*.jpg")):
        t = int(open(f.replace("images", "labels").replace(".jpg", ".txt")).readline().split()[0])
        r = p.inspect(cv2.imread(f))
        typed = [d for d in r.defects if d.label in NEU]
        if typed:
            pred = sorted(typed, key=lambda d: (-d.severity.rank, -d.confidence))[0].label
        else:
            pred = "PASS" if r.verdict.value == "PASS" else "untyped"
        cm[t, cols.index(pred)] += 1
    acc = np.trace(cm[:, :len(NEU)]) / cm.sum()
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.imshow(norm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", [SURF, "#cde2fb", BLUE, "#104281"]),
              vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(cols)), [c.replace("_", " ") for c in NEU] + ["anomaly,\nno type", "PASS\n(missed)"],
                  rotation=30, ha="right")
    ax.set_yticks(range(len(NEU)), [c.replace("_", " ") for c in NEU])
    ax.set_xlabel("App: worst defect named")
    ax.set_ylabel("NEU-DET label")
    ax.set_title(f"Steel line, NEU-DET val ({cm.sum()} images): image-level accuracy {acc:.1%}", fontsize=9, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "steel_confusion.png", dpi=180)
    plt.close(fig)
    return {"accuracy": round(float(acc), 4), "images": int(cm.sum()), "matrix": cm.tolist(), "columns": cols}


def roc(scores, labels):
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / (1 - y).sum()])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def mvtec(device, threads, out: Path) -> dict:
    res = {}
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.7), sharey=True)
    for ax, cat in zip(axes, CATS):
        p = InspectionPipeline(cfg_for(f"configs/profiles/mvtec_{cat}.yaml", device, threads))
        sc, lab, flag = [], [], []
        for f in sorted(glob.glob(f"datasets/mvtec_full/{cat}/test/*/*.png")):
            r = p.inspect(cv2.imread(f))
            sc.append(r.image_score)
            lab.append(0 if "/good/" in f else 1)
            flag.append(r.verdict.value != "PASS")
        lab, flag = np.array(lab), np.array(flag)
        fpr, tpr, auc = roc(sc, lab)
        rec, far = flag[lab == 1].mean(), flag[lab == 0].mean()
        res[cat] = {"auroc": round(auc, 3), "recall": round(float(rec), 3), "false_alarm": round(float(far), 3),
                    "defective": int(lab.sum()), "good": int((lab == 0).sum())}
        ax.plot([0, 1], [0, 1], color=GRID, lw=1)
        ax.plot(fpr, tpr, color=BLUE, lw=2, solid_capstyle="round")
        ax.plot([far], [rec], "o", ms=8, color=ORANGE, mec=SURF, mew=2, zorder=3)
        ax.set_title(f"{cat.replace('_', ' ')}  AUROC {auc:.3f}", fontsize=9, loc="left")
        ax.annotate(f"app: {rec:.1%} caught\n{far:.1%} false alarms", (far, rec), xytext=(0.32, 0.12),
                    textcoords="axes fraction", fontsize=7.5, color=INK2)
        ax.set_xlabel("False-alarm rate (good parts)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        print(cat, res[cat], flush=True)
    axes[0].set_ylabel("Defects caught (recall)")
    fig.suptitle("MVTec AD test sets: image-score ROC, dot = app operating point" +
                 (" (Pi 4 models)" if device else " (desktop models)"), fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / ("mvtec_roc_pi4.png" if device else "mvtec_roc_desktop.png"), dpi=180)
    plt.close(fig)
    return res


def main():
    logging.disable(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--only", choices=["steel", "mvtec"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    if a.only != "mvtec":
        summary["steel"] = steel(a.device, a.threads, out)
        print("steel", summary["steel"]["accuracy"], flush=True)
    if a.only != "steel":
        summary["mvtec"] = mvtec(a.device, a.threads, out)
    tag = "pi4" if a.device else "desktop"
    (out / f"figures_{tag}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
