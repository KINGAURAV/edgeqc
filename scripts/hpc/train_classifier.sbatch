#!/bin/bash
# YOLOv8n on NEU-DET → models/classifier/yolov8n_neu.onnx
#   sbatch scripts/hpc/train_classifier.sbatch            (from /scratch/<user>/edgeqc)
#   EPOCHS=150 sbatch scripts/hpc/train_classifier.sbatch
# Check which queue starts soonest: sbatch --test-only -p <partition> ... (multi-partition lists are rejected here)
#SBATCH --job-name=edgeqc-yolo
# override: sbatch -p gpu_h100_4 | gpu_h200_8 -c 4 (H200 max 4 CPUs/GPU; V100 unsupported by the CUDA 13 torch build)
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -c "import sys, cv2, torch; print('python', sys.executable, sys.version.split()[0], '| cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"

[[ -f datasets/neu_yolo/neu.yaml ]] || python scripts/prepare_neu_det.py --src datasets/NEU-DET --out datasets/neu_yolo
python scripts/train_classifier.py --data datasets/neu_yolo/neu.yaml \
  --epochs "${EPOCHS:-100}" --batch "${BATCH:-64}" --imgsz "${IMGSZ:-320}"
