#!/usr/bin/env bash
# Sync code to the HPC, or bring trained models back.
#   scripts/hpc/sync.sh push    # code + configs → HPC (no venvs, datasets, LLM weights, third_party)
#   scripts/hpc/sync.sh pull    # models/{anomaly,classifier} + training runs ← HPC
# Override with HPC_HOST=user@host HPC_DIR=/scratch/user/edgeqc
set -euo pipefail
HOST="${HPC_HOST:?set HPC_HOST=user@hpc-login-host}"
DIR="${HPC_DIR:?set HPC_DIR=/scratch/user/edgeqc}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

case "${1:-}" in
  push)
    ssh "$HOST" "mkdir -p '$DIR'"
    rsync -az \
      --exclude '.venv*/' --exclude 'third_party/' --exclude 'datasets/' --exclude 'runs/' \
      --exclude 'models/llm/' --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '*.egg-info/' \
      --exclude 'samples/synthetic/' \
      "$ROOT/" "$HOST:$DIR/"
    ;;
  pull)
    mkdir -p "$ROOT/models" "$ROOT/runs/hpc"
    for d in anomaly classifier; do   # one remote source per call (macOS openrsync)
      if ssh "$HOST" "test -d '$DIR/models/$d'" 2>/dev/null; then rsync -az "$HOST:$DIR/models/$d" "$ROOT/models/"; fi
    done
    rsync -az --include '*/' --include '*.png' --include '*.csv' --include '*.json' --include '*.yaml' \
      --include '*.log' --exclude '*' "$HOST:$DIR/runs/" "$ROOT/runs/hpc/" 2>/dev/null || true
    ;;
  *) echo "usage: $0 push|pull"; exit 1 ;;
esac
