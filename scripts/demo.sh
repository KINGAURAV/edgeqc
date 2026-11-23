#!/usr/bin/env bash
# Demo launcher (video recording). Plays a curated "virtual conveyor" from samples/demo/ at a readable pace.
#
#   scripts/demo.sh steel            # NEU steel strip: 6 defect types, each comes back once (advice reuse is instant)
#   scripts/demo.sh bottle           # MVTec bottle: good parts PASS, unseen defects flagged by anomaly detection
#   scripts/demo.sh steel --fresh    # forget saved advice first, so the Llama visibly writes every first reply
#   scripts/demo.sh steel --camera 0 # live USB camera instead of the image folder
#   scripts/demo.sh steel --prepare  # before recording: the Llama writes advice for every demo defect once (~5 min
#                                    # on a Pi 4), so the recorded run shows Llama advice instantly
#
# Extra args after the product go to the app (e.g. --set app.target_fps=2). On a Raspberry Pi 4 the Pi 4 device
# profile (core pinning, Pi-size models) is used automatically. Dashboard: http://<device-ip>:8000
set -euo pipefail
cd "$(dirname "$0")/.."
PRODUCT="${1:-steel}"; shift || true
FRESH=0; PREPARE=0; SOURCE=""; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --fresh) FRESH=1 ;;
    --prepare) PREPARE=1 ;;
    --camera) SOURCE="$2"; shift ;;
    *) EXTRA+=("$1") ;;
  esac
  shift
done

case "$PRODUCT" in
  steel)  CFG=configs/profiles/steel.yaml ;;
  bottle|metal_nut|tile|grid|screw) CFG=configs/profiles/mvtec_$PRODUCT.yaml ;;
  *) echo "unknown product: $PRODUCT"; exit 1 ;;
esac
[ -z "$SOURCE" ] && SOURCE="samples/demo/$PRODUCT"
[ -e "$SOURCE" ] || [[ "$SOURCE" =~ ^[0-9]+$ ]] || { echo "no demo images at $SOURCE"; exit 1; }

DEVICE=()
if grep -qs "Raspberry Pi 4" /proc/device-tree/model; then DEVICE=(--device configs/devices/pi4.yaml); fi

DATA="runs/demo_$PRODUCT"
if [ "$FRESH" = 1 ]; then rm -rf "$DATA"; fi   # also clears the saved advice cache (advice_cache.json)
mkdir -p "$DATA"

PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
if [ "$PREPARE" = 1 ]; then
  exec "$PY" scripts/prepare_demo_advice.py --config "$CFG" "${DEVICE[@]}" --source "$SOURCE" --set app.data_dir="$DATA"
fi
IP=$(hostname -I 2>/dev/null | awk '{print $1}'); IP="${IP:-localhost}"
echo "EdgeQC demo: $PRODUCT from $SOURCE ${DEVICE[*]:-}"
echo "Dashboard: http://$IP:8000   (Llama warm-up takes ~40 s on a Pi 4; Ctrl+C to stop)"
exec "$PY" -m edgeqc.app --config "$CFG" "${DEVICE[@]}" --source "$SOURCE" \
  --set app.data_dir="$DATA" --set app.target_fps=1 --set app.idle_target_fps=1 "${EXTRA[@]}"
