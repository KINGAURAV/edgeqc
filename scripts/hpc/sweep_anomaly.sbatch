#!/bin/bash
# PatchCore resolution / memory-bank sweep.
#   sbatch scripts/hpc/sweep_anomaly.sbatch                                   (accuracy sweep, hard categories)
#   CATS="bottle metal_nut screw grid tile" CFGS="256:2048 224:4096 224:2048 192:2048" \
#     sbatch scripts/hpc/sweep_anomaly.sbatch                                 (edge-speed sweep, Pi 4)
#SBATCH --job-name=edgeqc-patchcore-sweep
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
for c in ${CATS:-grid screw tile}; do
  for cfg in ${CFGS:-256:8192 320:4096 320:8192 384:8192}; do
    S=${cfg%%:*}; K=${cfg##*:}
    echo ">> $c size=$S coreset=$K"
    python scripts/train_anomaly.py --category-dir "datasets/mvtec/$c" --out "models/anomaly_sweep/${c}_s${S}_k${K}" \
      --size "$S" --coreset "$K" 2>&1 | grep -E "auroc|FPR<="
  done
done
