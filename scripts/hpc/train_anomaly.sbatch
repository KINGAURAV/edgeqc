#!/bin/bash
# PatchCore-lite for each MVTec AD category → models/anomaly/<category>/
#   sbatch scripts/hpc/train_anomaly.sbatch                        (default categories)
#   CATS="metal_nut screw" sbatch scripts/hpc/train_anomaly.sbatch
#   CPU-only (when GPU queues are long; ResNet-18 PatchCore is light enough):
#   sbatch -p big_compute --gres=none -c 128 --mem=128G scripts/hpc/train_anomaly.sbatch   (QOS needs >=128 CPUs)
#SBATCH --job-name=edgeqc-patchcore
# override: sbatch -p gpu_h100_4 | gpu_h200_8 -c 4 (H200 max 4 CPUs/GPU; V100 unsupported by the CUDA 13 torch build)
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -c "import sys, cv2, torch; print('python', sys.executable, sys.version.split()[0], '| cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "No GPU: running on ${SLURM_CPUS_PER_TASK:-?} CPUs"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

for c in ${CATS:-metal_nut screw grid bottle tile}; do
  echo ">> $c"
  python scripts/train_anomaly.py --category-dir "datasets/mvtec/$c" --out "models/anomaly/$c" \
    --coreset "${CORESET:-4096}"
done
# One-line summary of the eval metrics per category
python - <<'EOF'
import json, pathlib
for m in sorted(pathlib.Path("models/anomaly").glob("*/meta.json")):
    d = json.loads(m.read_text()); print(m.parent.name, {k: v for k, v in d.items() if not isinstance(v, (list, dict))})
EOF
