#!/usr/bin/env bash
# One-time: create the training venv on the HPC (run on the login node; it needs internet).
#   cd /scratch/<user>/edgeqc && bash scripts/hpc/setup_env.sh
# Uses a standalone CPython (via uv) stored on shared /scratch, so it's identical on every
# node. Node system Pythons differ: /usr/bin/python3 is 3.6 on GPU nodes, and python3.11 is not everywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."
SCR="$(dirname "$PWD")"
export UV_PYTHON_INSTALL_DIR="$SCR/uv-python" UV_CACHE_DIR="$SCR/uv-cache"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

uv python install 3.11
rm -rf .venv && uv venv --python 3.11 --python-preference only-managed .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -e '.[train,dev]' onnx onnxruntime onnxslim onnxscript kaggle six
python -c "import sys, torch, cv2, ultralytics; print(sys.executable, sys.version.split()[0], '| torch', torch.__version__, '| cv2', cv2.__version__, '| ultralytics', ultralytics.__version__)"
mkdir -p runs/slurm datasets models
