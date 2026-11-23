#!/bin/bash
# Upload datasets/neu_yolo as a PRIVATE Kaggle dataset and run scripts/kaggle/yolo_recipe/run.py on a Kaggle GPU.
# Also needs the private dataset <user>/edgeqc-yolo-deps (ultralytics wheels + yolov8n.pt): kernels on an
# unverified account have no internet. Needs the Kaggle CLI and your token in ~/.kaggle (kaggle.json or access_token).
#   bash scripts/kaggle/launch_yolo_recipe.sh          # upload + push kernel
#   kaggle kernels status  <user>/edgeqc-yolo-recipe   # poll
#   kaggle kernels output  <user>/edgeqc-yolo-recipe -p runs/kaggle_yolo_recipe
set -euo pipefail
cd "$(dirname "$0")/../.."
USER_NAME=$(kaggle config view | awk -F': ' '/username/{print $2}')
[ -n "$USER_NAME" ] || { echo "kaggle username not found (kaggle config view)"; exit 1; }
STAGE=$(mktemp -d)

# 1. private dataset (skipped if it already exists)
if ! kaggle datasets status "$USER_NAME/neu-yolo" >/dev/null 2>&1; then
  mkdir -p "$STAGE/ds"
  (cd datasets && zip -qr "$STAGE/ds/neu_yolo.zip" neu_yolo/images neu_yolo/labels)
  cat > "$STAGE/ds/dataset-metadata.json" <<EOF
{"title": "neu-yolo", "id": "$USER_NAME/neu-yolo", "licenses": [{"name": "other"}]}
EOF
  kaggle datasets create -p "$STAGE/ds"          # private by default
  echo "waiting for dataset to be ready..."; sleep 60
fi

# 2. GPU kernel
mkdir -p "$STAGE/k"; cp scripts/kaggle/yolo_recipe/run.py "$STAGE/k/"
cat > "$STAGE/k/kernel-metadata.json" <<EOF
{"id": "$USER_NAME/edgeqc-yolo-recipe", "title": "edgeqc-yolo-recipe", "code_file": "run.py",
 "language": "python", "kernel_type": "script", "is_private": true, "enable_gpu": true, "machine_shape": "NvidiaTeslaT4",
 "enable_internet": true, "dataset_sources": ["$USER_NAME/neu-yolo", "$USER_NAME/edgeqc-yolo-deps"]}
EOF
kaggle kernels push -p "$STAGE/k"
echo "pushed: https://www.kaggle.com/code/$USER_NAME/edgeqc-yolo-recipe"
