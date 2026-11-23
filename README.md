# EdgeQC: On-Device Industrial Defect Detection & QC Assistant

**Arm Bharat AI-SoC Challenge 2026-27 · Problem Statement 5.** Real-time visual inspection that runs fully on an Arm
edge device. **Target and test device: Raspberry Pi 4 Model B (4 GB, Cortex-A72)**; it also runs on Apple Silicon
and other aarch64 Linux. It detects and classifies surface defects, grades their severity, alerts the operator with
recommendations from **Llama 3.2 running on ExecuTorch**, and logs every inspection, with no cloud dependency.

```
frame ─▶ [A] anomaly detector (PatchCore-lite | classical) ─┐
      └▶ [B] defect classifier (YOLOv8n, NEU-DET)  ─────────┴▶ fusion ▶ severity rules ▶ PASS / REVIEW / REJECT
                     ONNX Runtime on Arm CPU  (HailoRT on NPU: untested)                      │
                                                                                              ▼
            SQLite log + images  ◀──  dashboard (MJPEG, alerts, ack)  ◀──  async Llama 3.2 1B advisor
                                                                           (ExecuTorch .pte, XNNPACK, Arm CPU)
```

The vision and advisory pipelines are **separate**: vision runs in ONNX Runtime on the Arm CPU, Llama runs through
ExecuTorch (XNNPACK) on the Arm CPU. Design: [docs/PLAN.md](docs/PLAN.md). Measured results:
[docs/RESULTS.md](docs/RESULTS.md).

Measured on the Raspberry Pi 4 (details and conditions in RESULTS.md section 5):

| | Pi 4 result |
|---|---|
| Steel line (classical anomaly + YOLOv8n INT8) | 5 FPS while the Llama is idle, 2 FPS while it writes advice; correct defect type named for 97.5% of NEU-DET val images |
| MVTec AD (PatchCore-lite INT8), 5 categories | 3.4 FPS (screw) to 4.4 FPS (bottle) while the Llama is idle; 1.7 FPS on bottle while it writes; 88-100% of defects caught at 0-10% false alarms |
| Llama 3.2 1B advice | 23-30 s for a new defect type (42-68 s while vision is running), first words streamed after 6-11 s; instant for a defect type seen before (advice is cached per type and severity) |


## Quick start (no models needed)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python scripts/make_synthetic_samples.py          # virtual conveyor of synthetic steel parts
python -m edgeqc.app                              # open http://localhost:8000
pytest -q
```
Out of the box, the app uses the classical OpenCV anomaly detector and the rule-based advisor. Add the real models as
you build them:

| Component | How | Enable |
|---|---|---|
| PatchCore-lite (MVTec AD) | `pip install -e '.[train]'`, `python scripts/train_anomaly.py --category-dir datasets/mvtec/bottle --out models/anomaly/bottle`, then `python scripts/optimize_patchcore.py` (INT8 fused model) | `--config configs/profiles/mvtec_bottle.yaml` (also `metal_nut`, `tile`, `grid`, `screw`) |
| YOLOv8n (NEU-DET) | `python scripts/prepare_neu_det.py ...`, `python scripts/train_classifier.py --data datasets/neu_yolo/neu.yaml`, then `python scripts/quantize_yolo.py` (INT8) | `--config configs/profiles/steel.yaml` |
| Llama 3.2 1B on ExecuTorch | [docs/EXECUTORCH_LLAMA.md](docs/EXECUTORCH_LLAMA.md) / `scripts/setup_executorch.sh` | automatic once the `.pte`, tokenizer and `llama_server` (or `llama_main`) exist |
| Raspberry Pi 4 tuning | smaller PatchCore models, vision/Llama core split | add `--device configs/devices/pi4.yaml` to any profile |
| Hailo NPU (optional, **untested**: no hardware) | [docs/HAILO.md](docs/HAILO.md) | `--set vision.backend=hailo` |

## Useful commands
```bash
python -m edgeqc.app --config configs/profiles/steel.yaml --device configs/devices/pi4.yaml   # on the Pi 4
python -m edgeqc.app --source 0                                   # USB camera
python -m edgeqc.app --source picamera                            # Pi camera (Picamera2)
python -m edgeqc.app --headless --max-frames 300 --export-csv runs/log.csv
python scripts/benchmark.py --source samples/synthetic --frames 200 --llm
python scripts/llm_smoke_test.py
```

## Repository layout
```
configs/   default.yaml · defect_kb.yaml (LLM grounding) · profiles/{steel,mvtec_<category>}.yaml · devices/pi4.yaml
src/edgeqc/
  capture/   camera (latest frame) / Picamera2 / video / image-folder sources
  vision/    backends (ONNX Runtime, ExecuTorch, Hailo) · anomaly · classifier · fusion · severity · pipeline · overlay
  advisor/   prompts · ExecuTorch Llama client · rule fallback · async service (per-signature advice cache)
  storage/   SQLite inspection log, CSV export
  ui/        FastAPI dashboard (offline, no CDN)
  app.py     main loop / CLI
native/    llama_server.cpp: resident ExecuTorch Llama runner with prompt KV caches and token streaming
scripts/   synthetic samples · dataset prep · training · INT8 export (optimize_patchcore, quantize_yolo) ·
           ExecuTorch setup · benchmark · LLM smoke test · hpc/ (Slurm jobs) · kaggle/ (GPU notebook job)
tests/     26 unit/smoke tests (run without any trained models)
docs/      PLAN.md · RESULTS.md · LITERATURE.md · EXECUTORCH_LLAMA.md · DATASETS.md · HPC.md · HAILO.md
```

## Licences
Code: MIT (TBD by team). MVTec AD is CC BY-NC-SA 4.0. Llama 3.2 is under the Llama 3.2 Community License.
Ultralytics YOLO is AGPL-3.0.
