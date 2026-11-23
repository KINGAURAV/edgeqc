# Models

Tracked in git (deployed, INT8, ~80 MB): everything the app needs except the Llama.

| File | Used by |
|---|---|
| `classifier/yolov8n_neu_256_int8.onnx` | steel profile (YOLOv8n, NEU-DET, 256 px) |
| `anomaly/{bottle,metal_nut,tile,grid,screw_s320}/patchcore_fused_int8.onnx` + `meta_patchcore_fused_int8.json` | MVTec profiles, desktop |
| `anomaly/pi4/<category>/patchcore_fused_int8.onnx` + meta (+ `projection.npz`) | MVTec profiles with `--device configs/devices/pi4.yaml` |

The fused models contain the memory bank, so `memory_bank.npy` is not needed at run time.

Not tracked (too large or licence-restricted), rebuild them:
- **Llama 3.2 1B Instruct `.pte` (765 MB) + `tokenizer.model`**: accept the Llama 3.2 licence on Hugging Face, then
  follow `docs/EXECUTORCH_LLAMA.md` (export 8da4w with ExecuTorch, build `llama_main` / `native/llama_server`).
  Without it the app runs with the rule-based advisor.
- FP32 models, memory banks, sweep variants: `scripts/train_classifier.py`, `scripts/train_anomaly.py`,
  `scripts/optimize_patchcore.py` (datasets: `docs/DATASETS.md`).
