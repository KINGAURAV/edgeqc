#!/usr/bin/env bash
# Report measurements on the Raspberry Pi 4 (run from the project root, nothing else heavy running):
#   1. advisor evaluation on 20 scripted cases (Llama on cores 1-3, as in the app)
#   2. per-stage latency for every profile (1 pinned core and 4-core boost)
#   3. full app (web UI on, cold advice cache) for steel and screw with memory sampling
# Output: runs/report_bench/
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=runs/report_bench
DEV=configs/devices/pi4.yaml
mkdir -p "$OUT"

echo "== 1. advisor eval $(date)"
$PY scripts/eval_advisor.py --device "$DEV" --out "$OUT/advisor_eval" > "$OUT/advisor_eval.log" 2>&1

echo "== 2. stage latency $(date)"
$PY scripts/profile_stages.py --config configs/profiles/steel.yaml --device "$DEV" --source samples/neu_val \
    --frames 60 --json "$OUT/stages/steel.json" > "$OUT/stages_steel.log" 2>&1
for c in bottle metal_nut tile grid screw; do
  $PY scripts/profile_stages.py --config "configs/profiles/mvtec_$c.yaml" --device "$DEV" --source "samples/mvtec/$c" \
      --frames 40 --json "$OUT/stages/$c.json" > "$OUT/stages_$c.log" 2>&1
done

for prof in steel screw; do
  echo "== 3. full app $prof $(date)"
  if [ "$prof" = steel ]; then cfg=configs/profiles/steel.yaml; src=samples/neu_val; else cfg=configs/profiles/mvtec_screw.yaml; src=samples/mvtec/screw; fi
  rm -rf "$OUT/app_$prof"
  $PY -m edgeqc.app --config "$cfg" --device "$DEV" --source "$src" --max-frames 900 \
      --set app.data_dir="$OUT/app_$prof" > "$OUT/app_$prof.log" 2>&1 &
  APP=$!
  $PY scripts/mem_monitor.py --pid $APP --json "$OUT/mem_$prof.json" > "$OUT/mem_$prof.log" 2>&1
  wait $APP
done
echo "== done $(date)"
