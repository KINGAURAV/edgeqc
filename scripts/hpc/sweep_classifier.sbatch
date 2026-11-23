#!/bin/bash
# Compare YOLO variants on NEU-DET (accuracy vs edge cost). One variant per array task:
#   sbatch scripts/hpc/sweep_classifier.sbatch
#SBATCH --job-name=edgeqc-yolo-sweep
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=runs/slurm/%x-%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
VARIANTS=("yolov8n.pt 320 200" "yolov8n.pt 640 150" "yolov8s.pt 320 150" "yolo11n.pt 320 200")
read -r MODEL IMGSZ EPOCHS <<< "${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
NAME="neu_${MODEL%.pt}_${IMGSZ}_e${EPOCHS}"
echo "variant: $NAME"
python scripts/train_classifier.py --data datasets/neu_yolo/neu.yaml --model "$MODEL" --imgsz "$IMGSZ" \
  --epochs "$EPOCHS" --batch 64 --name "$NAME" --out "models/classifier_sweep/$NAME.onnx"
