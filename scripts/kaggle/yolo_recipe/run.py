"""Kaggle GPU kernel: NEU-DET training-recipe study (same as scripts/hpc/sweep_yolo_recipe.sbatch).

3 seeds x {base: 100 epochs, recipe: 200 epochs + close_mosaic 30 + grayscale-suited aug}, YOLOv8n @ 256 px.
Runs two variants at a time when the session has 2 GPUs (T4 x2). Results: /kaggle/working/results.txt + best.pt files.
"""
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Kernel has no internet (unverified account): install from the private "edgeqc-yolo-deps" dataset
# (wheels + yolov8n.pt), no dependency resolution; torch/opencv/etc. come from the Kaggle image.
deps = next(p.parent for p in Path("/kaggle/input").rglob("yolov8n.pt"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                *map(str, deps.glob("*.whl"))], check=True)
os.environ["YOLO_OFFLINE"] = "1"
WEIGHTS = str(deps / "yolov8n.pt")

import torch  # noqa: E402

src = next(p.parent.parent for p in Path("/kaggle/input").rglob("train") if p.parent.name == "images")
data = Path("/kaggle/tmp/neu_yolo")
if not data.exists():
    shutil.copytree(src, data)
(data / "neu.yaml").write_text(
    f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n  0: crazing\n  1: inclusion\n  2: patches\n"
    "  3: pitted_surface\n  4: rolled-in_scale\n  5: scratches\n")
out = Path("/kaggle/working")

VARIANTS = []
for seed in range(3):
    VARIANTS.append((f"neu_v8n256_base_s{seed}", seed, dict(epochs=100)))
for seed in range(3):
    VARIANTS.append((f"neu_v8n256_recipe_s{seed}", seed,
                     dict(epochs=200, close_mosaic=30, hsv_h=0.0, hsv_s=0.0, flipud=0.5, mixup=0.1)))

TRAIN = r"""
import json, sys
from ultralytics import YOLO
name, seed, data, extra, weights = sys.argv[1], int(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4]), sys.argv[5]
extra.setdefault("close_mosaic", 15)
m = YOLO(weights)
m.train(data=data, imgsz=256, batch=64, device=0, project="/kaggle/working/runs", name=name, exist_ok=True,
        patience=100, cos_lr=True, seed=seed, deterministic=True, workers=2, plots=False, amp=False, **extra)
b = YOLO(f"/kaggle/working/runs/{name}/weights/best.pt")
r = b.val(data=data, imgsz=256, device=0, verbose=False, plots=False)
line = (f"PERCLASS {name} map50={r.box.map50:.4f} map50-95={r.box.map:.4f} "
        + " ".join(f"{r.names[c]}={ap:.3f}" for c, ap in zip(r.box.ap_class_index, r.box.ap50)))
print(line)
open("/kaggle/working/results.txt", "a").write(line + "\n")
"""


def run(job):
    i, (name, seed, extra) = job
    gpu = str(i % max(1, torch.cuda.device_count()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    import json
    with open(out / f"{name}.log", "w") as log:
        subprocess.run([sys.executable, "-c", TRAIN, name, str(seed), str(data / "neu.yaml"), json.dumps(extra), WEIGHTS],
                       env=env, stdout=log, stderr=subprocess.STDOUT)
    print("done", name, flush=True)


print("GPUs:", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available():
    subprocess.run("nvidia-smi || true", shell=True)
    sys.exit("No GPU in this session (Kaggle needs a phone-verified account for GPUs/internet).")
with ThreadPoolExecutor(max(1, torch.cuda.device_count())) as ex:
    list(ex.map(run, enumerate(VARIANTS)))
for f in sorted(out.glob("*.log")):
    print("====", f.name, "\n", "".join(open(f).readlines()[-15:]))
print(open(out / "results.txt").read() if (out / "results.txt").exists() else "NO RESULTS")
