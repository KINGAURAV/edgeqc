#!/usr/bin/env python3
"""Train PatchCore-lite on good images, calibrate thresholds, evaluate on MVTec AD, export ONNX.

    python scripts/train_anomaly.py --category-dir datasets/mvtec/bottle --out models/anomaly/bottle

Outputs (consumed by edgeqc.vision.anomaly.PatchCoreDetector):
    feature_extractor.onnx   ResNet-18 layer2+layer3 → (1, 384, S/8, S/8) local-avg-pooled features
    memory_bank.npy          coreset of good-patch features (M, 384) float32
    meta.json                thresholds + eval metrics

Requires: pip install -e '.[train]'
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


class FeatureExtractor(nn.Module):
    """ResNet-18 layer2+layer3, 3x3 local average. With a projection (projection.npz from --proj-dim), a 1x1
    conv maps the 384-dim patch features onto their top-D principal components, so the memory bank and the
    kNN matmul (~40% of Pi 4 latency) shrink by 384/D."""

    def __init__(self, projection: str | Path | None = None):
        super().__init__()
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.layer2, self.layer3 = m.layer2, m.layer3
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.proj = None
        if projection and Path(projection).exists():
            z = np.load(projection)
            self.set_projection(torch.from_numpy(z["components"]), torch.from_numpy(z["mean"]))

    def set_projection(self, comps: torch.Tensor, mean: torch.Tensor) -> None:
        """comps (C, D) orthonormal columns, mean (C,): f -> (f - mean) @ comps."""
        c, d = comps.shape
        self.proj = nn.Conv2d(c, d, 1)
        with torch.no_grad():
            self.proj.weight.copy_(comps.t().reshape(d, c, 1, 1))
            self.proj.bias.copy_(-(mean @ comps))

    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = F.interpolate(self.layer3(f2), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = self.pool(torch.cat([f2, f3], 1))
        return self.proj(f) if self.proj is not None else f


def load(path: Path, size: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def images(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in EXTS)


@torch.no_grad()
def extract(model, paths, size, device, bs=16, rot: int = 0) -> torch.Tensor:
    """rot=k also adds the good images rotated by 90°..k*90° (exact, no interpolation): for parts placed at
    any orientation (screw), so the memory bank covers orientations the training set under-samples."""
    feats = []
    for i in range(0, len(paths), bs):
        x = torch.from_numpy(np.stack([load(p, size) for p in paths[i:i + bs]]))
        for k in range(rot + 1):
            feats.append(model(torch.rot90(x, k, (2, 3)).contiguous().to(device)).cpu())
    return torch.cat(feats)  # (N, C, h, w)


@torch.no_grad()
def greedy_coreset(x: torch.Tensor, k: int, proj_dim: int = 128, device="cpu") -> torch.Tensor:
    """k-center greedy on a random projection (PatchCore §3.2)."""
    n = x.shape[0]
    if k >= n:
        return torch.arange(n)
    proj = torch.randn(x.shape[1], proj_dim) / proj_dim ** 0.5
    z = (x @ proj).to(device)
    idx = [random.randrange(n)]
    mind = torch.cdist(z, z[idx]).squeeze(1)
    for _ in range(k - 1):
        j = int(mind.argmax())
        idx.append(j)
        mind = torch.minimum(mind, torch.cdist(z, z[j:j + 1]).squeeze(1))
    return torch.tensor(idx)


@torch.no_grad()
def patch_dist(feats: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    n, c, h, w = feats.shape
    f = feats.permute(0, 2, 3, 1).reshape(-1, c)
    d = torch.cat([torch.cdist(f[i:i + chunk], bank).min(1).values for i in range(0, len(f), chunk)])
    return d.reshape(n, h, w)


def to_heat(dmap: np.ndarray, size: tuple[int, int], sigma: float) -> np.ndarray:
    heat = cv2.resize(dmap, size, interpolation=cv2.INTER_LINEAR)
    k = int(sigma * 4) | 1
    return cv2.GaussianBlur(heat, (k, k), sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category-dir", required=True, help="MVTec-style dir with train/good (and optionally test/, ground_truth/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--coreset", type=int, default=4096, help="memory bank size (smaller = faster on Pi)")
    ap.add_argument("--holdout", type=float, default=0.1, help="fraction of good images for threshold calibration")
    ap.add_argument("--blur-sigma", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rot-aug", type=int, default=0, choices=[0, 1, 3],
                    help="add 90° (1) or 90/180/270° (3) rotated good images to the memory bank")
    ap.add_argument("--proj-dim", type=int, default=0,
                    help="PCA-project patch features to this many dims (0 = keep 384); cuts kNN cost on the edge")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root, out = Path(args.category_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor().eval().to(device)

    good = images(root / "train" / "good")
    random.shuffle(good)
    n_hold = max(2, int(len(good) * args.holdout))
    hold, train = good[:n_hold], good[n_hold:]
    (out / "holdout.json").write_text(json.dumps([p.name for p in hold]))  # for re-calibrating optimised variants
    print(f"{len(train)} train / {len(hold)} holdout good images, device={device}")

    feats = extract(model, train, args.size, device, rot=args.rot_aug)
    c = feats.shape[1]
    flat = feats.permute(0, 2, 3, 1).reshape(-1, c)
    if args.proj_dim and args.proj_dim < c:
        # PCA on the good patches; the projection becomes a 1x1 conv inside the extractor (and every export).
        sub = flat[torch.randperm(len(flat))[:200_000]].double()
        mean = sub.mean(0)
        _, ev, vt = torch.linalg.svd(sub - mean, full_matrices=False)
        comps = vt[:args.proj_dim].t().float().contiguous()
        kept = float((ev[:args.proj_dim] ** 2).sum() / (ev ** 2).sum())
        np.savez(out / "projection.npz", components=comps.numpy(), mean=mean.float().numpy())
        model.set_projection(comps, mean.float())
        model.proj.to(device)
        flat = (flat - mean.float()) @ comps
        c = args.proj_dim
        print(f"PCA {feats.shape[1]} -> {c} dims ({kept:.1%} variance)")
    elif (out / "projection.npz").exists():
        (out / "projection.npz").unlink()
    print(f"{flat.shape[0]} patches → coreset {args.coreset}")
    bank = flat[greedy_coreset(flat, args.coreset, device=device)].contiguous()
    np.save(out / "memory_bank.npy", bank.numpy().astype(np.float32))

    # Calibrate on held-out good images
    hd = patch_dist(extract(model, hold, args.size, device), bank).numpy()
    image_threshold = float(hd.reshape(len(hd), -1).max(1).max() * 1.05)
    pix = np.concatenate([to_heat(m, (args.size, args.size), args.blur_sigma).ravel() for m in hd])
    pixel_threshold = float(np.percentile(pix, 99.9) * 1.05)
    # Threshold for the smoothed image score (max of the blurred heatmap), which the app uses when present
    image_threshold_smooth = float(max(to_heat(m, (args.size, args.size), args.blur_sigma).max() for m in hd) * 1.05)
    meta = {"image_threshold": image_threshold, "image_threshold_smooth": image_threshold_smooth,
            "pixel_threshold": pixel_threshold,
            "blur_sigma": args.blur_sigma, "input_size": args.size, "coreset": int(bank.shape[0]),
            "feature_dim": int(c), "category": root.name}

    # Evaluate on MVTec test split if present
    test_dir = root / "test"
    if test_dir.exists():
        from sklearn.metrics import roc_auc_score

        tpaths = images(test_dir)
        td = patch_dist(extract(model, tpaths, args.size, device), bank).numpy()
        y = np.array([0 if p.parent.name == "good" else 1 for p in tpaths])
        s = td.reshape(len(td), -1).max(1)
        meta["image_auroc"] = float(roc_auc_score(y, s))
        pred = s >= image_threshold
        meta["image_recall_at_threshold"] = float(pred[y == 1].mean())
        meta["image_fpr_at_threshold"] = float(pred[y == 0].mean())
        # Operating points (diagnostic only: they use test labels, so report them as a trade-off curve,
        # not as the deployed calibration). threshold_ratio is relative to the calibrated image_threshold.
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y, s)
        meta["operating_points"] = [
            {"max_fpr": t, "recall": float(tpr[fpr <= t].max()),
             "threshold_ratio": float(thr[fpr <= t][tpr[fpr <= t].argmax()] / image_threshold)}
            for t in (0.0, 0.05, 0.10)]
        gt_dir = root / "ground_truth"
        if gt_dir.exists():
            ys, ss = [], []
            for p, m in zip(tpaths, td):
                heat = to_heat(m, (64, 64), args.blur_sigma / 4)
                if p.parent.name == "good":
                    gt = np.zeros((64, 64), np.uint8)
                else:
                    g = cv2.imread(str(gt_dir / p.parent.name / f"{p.stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
                    gt = (cv2.resize(g, (64, 64), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
                ys.append(gt.ravel()); ss.append(heat.ravel())
            meta["pixel_auroc"] = float(roc_auc_score(np.concatenate(ys), np.concatenate(ss)))
        print({k: round(v, 4) for k, v in meta.items() if "auroc" in k or "at_threshold" in k})
        for op in meta["operating_points"]:
            print(f"  FPR<={op['max_fpr']:.2f}: recall {op['recall']:.3f} at {op['threshold_ratio']:.2f}x threshold")

    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    dummy = torch.randn(1, 3, args.size, args.size)
    # TorchScript-based exporter (dynamo=False): plain static graph that ONNX Runtime on the Pi
    # and the Hailo Dataflow Compiler both accept; newer torch defaults to the dynamo exporter.
    kw = {"dynamo": False} if "dynamo" in torch.onnx.export.__code__.co_varnames else {}
    torch.onnx.export(model.cpu(), dummy, out / "feature_extractor.onnx", input_names=["input"],
                      output_names=["features"], opset_version=17, **kw)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
