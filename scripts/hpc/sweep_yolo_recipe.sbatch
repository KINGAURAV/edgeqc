#!/bin/bash
# NEU-DET training-recipe study (literature: aug suited to grayscale steel, longer schedule), 3 seeds each so a
# difference can be told from seed noise (~1 pt). Same YOLOv8n @ 256 px as deployed: zero inference cost.
#   sbatch scripts/hpc/sweep_yolo_recipe.sbatch
#SBATCH --job-name=edgeqc-yolo-recipe
#SBATCH --partition=gpu_v100_1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --array=0-5%2
#SBATCH --output=runs/slurm/%x-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))
if (( SLURM_ARRAY_TASK_ID < 3 )); then
  NAME="neu_v8n256_base_s$SEED"; EXTRA=(--epochs 100)
else
  NAME="neu_v8n256_recipe_s$SEED"
  EXTRA=(--epochs 200--close-mosaic 30 --hyp hsv_h=0 --hyp hsv_s=0 --hyp flipud=0.5 --hyp mixup=0.1)
fi
echo "variant: $NAME"
python scripts/train_classifier.py --data datasets/neu_yolo/neu.yaml --model yolov8n.pt --imgsz 256 \
  --batch 64 --seed "$SEED" --patience 100 --name "$NAME" --out "models/classifier_sweep/$NAME.onnx" "${EXTRA[@]}"
python - "$NAME" <<'PY'
import sys
from ultralytics import YOLO
m = YOLO(f"runs/train/{sys.argv[1]}/weights/best.pt")
r = m.val(data="datasets/neu_yolo/neu.yaml", imgsz=256, verbose=False)
names = r.names
print("PERCLASS", sys.argv[1], f"map50={r.box.map50:.4f}",
      " ".join(f"{names[c]}={ap:.3f}" for c, ap in zip(r.box.ap_class_index, r.box.ap50)))
PY
