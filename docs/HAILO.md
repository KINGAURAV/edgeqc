# Optional: vision on the Hailo NPU (Raspberry Pi AI HAT+ / AI HAT+ 2)

> **Not used and never run.** Our device is a Raspberry Pi 4, and the AI HAT+ needs the Pi 5's PCIe connector.
> `HailoBackend` is written against the HailoRT documentation only. Do not claim NPU results in the report; all
> measured vision numbers are ONNX Runtime on the Arm CPU.

If you use an AI HAT+, **only the vision models move to the NPU**. Llama keeps running on the Pi's Arm CPU through
ExecuTorch. These are two independent pipelines:

| | Vision | Advisory |
|---|---|---|
| Runtime | HailoRT (`hailo_platform`), `.hef` | ExecuTorch `llama_server` / `llama_main`, `.pte` |
| Silicon | Hailo-8L / Hailo-8 / Hailo-10H NPU (on a Pi 5) | The Pi's Arm CPU |
| Code | `src/edgeqc/vision/backends.py::HailoBackend` | `src/edgeqc/advisor/executorch_llama.py` |

## Steps
1. On an **x86 Linux host**, install the Hailo Dataflow Compiler (Hailo Developer Zone).
2. Compile the ONNX models with a calibration set of about 64–512 real images:
   - `models/anomaly/<cat>/feature_extractor.onnx` → `.hef` (the backbone-only export; the fused `patchcore_fused*.onnx` models the profiles use contain the kNN and are for ONNX Runtime). Fold ImageNet normalisation into the model (normalization layer in the model script), so the input is uint8 NHWC. The kNN then runs in NumPy on the CPU (`PatchCoreDetector._patch_distances`).
   - `models/classifier/yolov8n_neu.onnx` → `.hef`. Keep the raw output head (`(1, 4+nc, N)`), or adapt `YoloDetector.postprocess` if you use Hailo's on-chip NMS.
3. On the Pi: `sudo apt install hailo-all`, then reboot and check with `hailortcli fw-control identify`.
4. Run:
   ```bash
   python -m edgeqc.app --set vision.backend=hailo \
       --set vision.anomaly.model_path=models/anomaly/<cat>/feature_extractor.hef \
       --set vision.classifier.model_path=models/classifier/yolov8n_neu.hef
   ```

`HailoBackend` targets the HailoRT InferModel API (≥ 4.18). It is **untested until we have hardware**, so validate the
output shapes (NHWC) against `PatchCoreDetector._patch_distances` and `YoloDetector.postprocess`.
