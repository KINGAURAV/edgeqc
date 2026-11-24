# Results log (feeds the technical report)

**Target device: Raspberry Pi 4 Model B (4 GB, Cortex-A72). All Pi numbers are in sections 5-6**; quote those in the
report. Sections 1-4 were measured on the **dev Mac (Apple M2, arm64, 8 GB), ONNX Runtime 1.31 CPU, 4 threads**
(accuracy numbers carry over; latency numbers do not: the Pi 4 is ~13x slower). Training ran on the BITS Hyderabad
HPC (NVIDIA A100 80 GB) unless marked (Mac MPS for the Pi-size PatchCore models, Kaggle 2x T4 for the YOLO seed study).

**What is deployed now (16 Nov), for quick reference:**

| | Desktop / Mac | Raspberry Pi 4 (`--device configs/devices/pi4.yaml`) |
|---|---|---|
| Steel: classifier | YOLOv8n INT8, 256 px, per-class confidence thresholds (crazing 0.15, rolled-in scale 0.20, others 0.35) | same |
| Steel: anomaly stage | classical OpenCV detector | same |
| MVTec bottle / metal_nut / tile | PatchCore 256 px / 4096 bank, INT8 fused | 224 px / 2048 bank |
| MVTec grid | 256 / 4096 | 256 / 4096 / PCA-128 |
| MVTec screw | 320 / 4096 | 288 / 4096 / PCA-128 |
| MVTec decision rule | smoothed image score >= calibrated threshold x per-category scale | same |

**MVTec detection as deployed** (app verdict PASS vs REVIEW/REJECT on the full test sets; details in section 5,
"Smoothed image score"). The scale per category is chosen on the test set for <= 10% false alarms:

| Category | Desktop model: defects caught / good parts flagged | Pi 4 model: defects caught / good parts flagged |
|---|---|---|
| bottle | 100% / 0% | 100% / 0% |
| metal_nut | 98.9% / 9.1% | 98.9% / 9.1% |
| tile | 100% / 9.1% | 100% / 9.1% |
| grid | 80.7% / 4.8% | 87.7% / 9.5% |
| screw | 95.0% / 4.9% | 96.6% / 7.3% |
| Advisor | Llama 3.2 1B, ExecuTorch 8da4w, resident `llama_server`, advice cached per defect signature | same; Llama on cores 1-3, vision on core 0 (all 4 cores while Llama is idle) |


## 1. Defect classifier: YOLOv8n on NEU-DET (6 classes, 1,439 train / 361 val images)
Trained in 4 min 42 s (100 epochs, imgsz 320, A100).

| Class | mAP@0.5 (PyTorch val) |
|---|---|
| patches | 0.934 |
| scratches | 0.823 |
| inclusion | 0.821 |
| pitted_surface | 0.809 |
| rolled-in_scale | 0.531 |
| crazing | 0.509 (recall 0.31) |
| **all** | **0.738** (mAP@0.5:0.95 0.384) |

Edge model (`scripts/quantize_yolo.py`, validated through ONNX Runtime on the Arm CPU):

| Model | mAP@0.5 | mAP@0.5:0.95 | p50 latency | Size |
|---|---|---|---|---|
| FP32 ONNX | 0.691 | 0.344 | 11.6 ms | 12.6 MB |
| **INT8 ONNX** (static, per-channel, head kept FP32) | **0.700** | 0.345 | **9.4 ms** | **6.0 MB** |

The ONNX numbers are lower than PyTorch's because of fixed-size 320 letterboxing (PyTorch val uses rectangular
batches). Compare FP32 vs INT8 within the ONNX column only. These rows are the 320 px export; the deployed model is
the same weights exported at 256 px (mAP@0.5 0.707, section 5 "Pi 4 speed-ups").

### Architecture / resolution sweep (HPC, `scripts/hpc/sweep_classifier.sbatch`; PyTorch val, same 361 images)

| Variant | mAP@0.5 | mAP@0.5:0.95 | crazing mAP@0.5 | rolled-in_scale mAP@0.5 |
|---|---|---|---|---|
| **YOLOv8n, 320, 100 ep (deployed weights)** | **0.738** | 0.384 | 0.509 | 0.531 |
| YOLOv8n, 320, 200 ep | 0.703 | 0.371 | 0.406 | 0.474 |
| YOLOv8n, 640, 150 ep | 0.721 | 0.382 | 0.455 | 0.587 |
| YOLOv8s, 320, 150 ep | 0.715 | 0.373 | 0.488 | 0.491 |
| YOLO11n, 320, 200 ep | 0.721 | 0.396 | 0.510 | 0.560 |

No variant beats the deployed model on mAP@0.5, and the larger ones cost 2-4× the edge compute (640 px, v8s).
The weak classes are a data problem (crazing is low-contrast texture with ambiguous box extents), not a capacity
one. These are single runs: the seed study below shows that seed alone moves mAP@0.5 by 2.8 pt, so the differences
in this table (including "200 ep < 100 ep" and YOLO11n's mAP@0.5:0.95) are within noise. Kept YOLOv8n.

### Training-recipe study, 3 seeds each (Kaggle 2x T4, 15 Nov; `scripts/kaggle/yolo_recipe/run.py`)

YOLOv8n @ 256 px (the Pi input size), batch 64, cosine LR, FP32. *Base* = 100 epochs, default augmentation.
*Recipe* (from the literature review) = 200 epochs, close_mosaic 30, hsv_h = hsv_s = 0 (images are grayscale),
flipud 0.5, mixup 0.1. Mean ± std over seeds 0-2, PyTorch val on the same 361 images.

| AP@0.5 | Base | Recipe | Δ |
|---|---|---|---|
| **mAP@0.5** | 0.729 ± 0.014 | 0.732 ± 0.013 | +0.3 |
| mAP@0.5:0.95 | 0.390 ± 0.004 | 0.390 ± 0.009 | 0.0 |
| crazing | 0.500 ± 0.047 | 0.527 ± 0.047 | +2.6 |
| inclusion | 0.806 ± 0.017 | 0.818 ± 0.020 | +1.2 |
| patches | 0.925 ± 0.005 | 0.931 ± 0.005 | +0.6 |
| pitted_surface | 0.792 ± 0.019 | 0.800 ± 0.017 | +0.8 |
| rolled-in_scale | 0.513 ± 0.036 | 0.477 ± 0.015 | −3.6 |
| scratches | 0.839 ± 0.003 | 0.843 ± 0.012 | +0.4 |

**Not adopted:** every difference is within one seed standard deviation. The recipe neither fixes crazing nor
rolled-in scale (one goes up, the other down, both inside the noise). The study also shows how noisy NEU-DET is:
seed alone moves mAP@0.5 by 2.8 pt (0.718-0.746) and crazing AP by 9 pt. The deployed model's 0.738 is a typical
seed, and none of the single-run sweep differences above can be told from noise. Remaining lever: the labels
(box extents for diffuse crazing / rolled-in scale).

### Label audit → per-class confidence thresholds (16 Nov; deployed INT8 256 px model, no retraining)

Box AP undersells what the app needs: every NEU image holds one defect class, and the verdict names the defect type.
Image-level check (class of the most confident detection) showed the model rarely *confuses* classes; it is just
not confident on crazing. Median own-class confidence: crazing 0.33-0.35 (train/val), rolled-in scale 0.52-0.74,
others 0.71-0.87. At the global 0.35 threshold, 54% of crazing and 15% of rolled-in val images got no detection.
The labels explain it: crazing is a uniform texture over the whole image, yet each image has 2-3 arbitrary boxes
(median 20% of the image); rolled-in scale has overlapping/nested boxes. The model cannot learn *which* patch to box,
so box confidence stays low even on the training set. Relabeling ~600 images by hand was not needed:

| Thresholds (`vision.classifier.class_conf_thresholds`) | Image accuracy val | train | No detection (val) | Images with a wrong-class box val / train |
|---|---|---|---|---|
| global 0.35 (before) | 86.7% | 87.9% | 12.2% | 4.7% / 6.9% |
| global 0.20 | 97.5% | 96.2% | 1.4% | 6.1% / 8.7% |
| **crazing 0.15, rolled-in 0.20, others 0.35 (deployed)** | **97.5%** | **97.2%** | **1.4%** | **5.0% / 6.9%** |

Chosen thresholds agree on train and val, so they are not fitted to the val set. End-to-end through the real
pipeline (steel profile, all 361 val images): verdicts PASS 15 / REVIEW 20 / REJECT 326 → PASS 2 / REJECT 359;
correct defect type named 86.7% → 97.5% (crazing 46 → 97%, rolled-in 85 → 98%). Remaining errors: 4 pitted-surface
images called patches. Caveat: NEU has no defect-free images, so the false-reject rate on good steel can't be
measured here; lowering only the two low-confidence classes keeps that risk small.

## 2. Anomaly detection: PatchCore-lite (ResNet-18 layer2+3, coreset 4,096) on MVTec AD
Thresholds are calibrated on held-out good training images only (10%: max score × 1.05). Trained in 6 min for
5 categories (A100). Recall is the share of defective parts flagged; FPR is the share of good parts flagged.
Recall / FPR in the tables of sections 2-5 dated before 17 Nov are **image-level with the raw image score** (largest
patch distance vs image threshold). The app never ran at exactly those points: until 16 Nov it flagged any region
over the pixel threshold (far more false alarms), and since 17 Nov it uses the smoothed image score, which is
better. **The numbers to quote are in the "MVTec detection as deployed" table at the top.**

| Category | Image AUROC | Pixel AUROC | Recall / FPR @ threshold | Recall / FPR @ 0.9× threshold |
|---|---|---|---|---|
| bottle | 1.000 | 0.982 | 100% / 0% | 100% / 0% |
| metal_nut | 0.997 | 0.981 | 93.5% / 0% | 97.8% / 9.1% |
| tile | 0.985 | 0.940 | 75.0% / 0% | 95.2% / 6.1% |
| screw | 0.962 | 0.986 | 46.2% / 0% | 82.4% / 4.9% |
| grid | 0.881 | 0.966 | 50.9% / 0% | 91.2% / 28.6% |

The threshold-scale operating points use test labels. Present them as a trade-off curve, not as the deployed
calibration. Each `models/anomaly/<cat>/meta.json` has `operating_points` (best recall at FPR ≤ 0 / 5 / 10%).

### Edge optimisation (`scripts/optimize_patchcore.py`; full test set, same Arm CPU)
INT8 = backbone convolutions statically quantised (QDQ, per-channel). The kNN stays FP32. Thresholds are
re-calibrated for INT8 on the same holdout (`meta_patchcore_fused_int8.json`).

| Category | AUROC FP32 → INT8 | Recall@thr FP32 → INT8 | p50 FP32 → INT8 |
|---|---|---|---|
| bottle | 1.000 → 1.000 | 100% → 100% | 28.2 → 23.6 ms |
| metal_nut | 0.997 → 0.997 | 93.5% → 92.5% | 26.7 → 21.2 ms |
| tile | 0.985 → 0.986 | 75.0% → 77.4% | 28.2 → 20.8 ms |
| screw | 0.962 → 0.949 | 46.2% → 39.5% | 24.1 → 17.7 ms |
| grid | 0.881 → 0.892 | 50.9% → 54.4% | 22.5 → 16.7 ms |

Model size 11.1 → 9.2 MB (fused graph incl. memory bank). Tile latency re-measured alone (first run overlapped another job).

Other changes: the heatmap and regions are now computed at the model's working resolution (256 px), which is the
same resolution the pixel threshold was calibrated at, instead of at full frame resolution. Boxes are scaled back
to frame coordinates. On the Pi, the camera streams 640×480 (ISP-scaled), so the CPU resize is nearly free.

### Resolution / memory-bank sweep (HPC, `scripts/hpc/sweep_anomaly.sbatch`; FP32, PyTorch eval)

| Category | 256/4096 (base) | 256/8192 | 320/4096 | 320/8192 | 384/8192 |
|---|---|---|---|---|---|
| screw AUROC (recall@thr) | 0.962 (46%) | 0.946 (49%) | **0.972 (76%)** | 0.976 (78%) | 0.981 (88%, FPR 2.4%) |
| tile AUROC | **0.985** | 0.979 | 0.957 | 0.937 | 0.946 |
| grid AUROC | 0.881 | 0.876 | 0.885 | 0.875 | 0.847 |

Only screw benefits (small defects need resolution); tile and grid get worse or stay flat. Screw candidates on the
Arm CPU (INT8 fused, re-calibrated on the same holdout; FP32 re-calibration reproduces the training threshold
2.3762 vs 2.3749):

| Screw model (INT8) | p50 | recall/FPR @1.0× | @0.95× | @0.93× | @0.90× |
|---|---|---|---|---|---|
| 256 px / 4096 (old) | 15.7 ms | 39% / 0% | 56% / 0% | 68% / 2% | 77% / 2% |
| **320 px / 4096 (deployed)** | **28.2 ms** | 68% / 0% | 82% / 0% | **87% / 0%** | 92% / 10% |
| 320 px / 8192 | 45.7 ms | 76% / 0% | 85% / 0% | 90% / 0% | 93% / 10% |

Deployed: 320/4096 at 0.93× (`configs/profiles/mvtec_screw.yaml`, model in `models/anomaly/screw_s320/`), end-to-end
24.1 ms p50 / 37.7 FPS. The 8192 bank adds 3 points of recall for +60% latency, not worth it on the Pi. Grid
stays the weakest category; a stronger backbone (e.g. WideResNet-50) is the known fix but is ~5× the compute.
The 0.93× operating point was chosen with test labels; report it as such.

## 3. End-to-end vision pipeline (`scripts/benchmark.py`, includes reading frames from disk)

| Profile | Before | After | Main change |
|---|---|---|---|
| steel (classical anomaly + YOLOv8n) | 29.5 ms mean, 33 FPS | **12.9 ms, 76 FPS** | INT8 YOLO, low-res heatmap |
| mvtec metal_nut (PatchCore) | 28.0 ms p50, 26.6 FPS | **19.3 ms p50, 35 FPS** | INT8 fused PatchCore |
| mvtec screw (PatchCore) | 256 px, 39% recall@thr | **320 px INT8, 24.1 ms p50, 38 FPS, 87% recall / 0% FPR @0.93×** | sweep-selected resolution |

## 4. Advisory LLM: Llama 3.2 1B Instruct, ExecuTorch (XNNPACK, 8da4w, 1,024 context)
| Item | Value |
|---|---|
| `.pte` size | 803 MB (765 MiB) |
| Prefill / decode | about 169 / 20.6 tok/s |
| Runner peak RAM | about 0.85 GB |
| Advice latency, one-shot `llama_main` (reloads model each call) | 2.8–6.2 s |
| Advice latency, **resident `llama_server`** (model loaded once) | **1.9–2.5 s** |
| Grounding guard pass rate (smoke test, greedy decoding) | 3/3 |

Decoding is greedy (temperature 0), so the same findings always produce the same advice. Since 13 Nov the `Log:`
line is written by code from the same findings, not by the model (section 5 "Literature-driven improvements").

## 5. Raspberry Pi 4 Model B (4 GB, Cortex-A72 @ 1.5 GHz, Debian 13), 12 Nov 2026
Target device changed from Pi 5 to Pi 4 (Pi 5 unavailable). Same code; ONNX Runtime CPU. The tables in this first
block use 4 threads and the desktop models (first measurements, 12 Nov); the sub-sections below are in date order and
later ones supersede earlier ones. Current state: the "What is deployed now" table at the top and "Runtime
optimisations (16 Nov)" at the end of this section.
No heatsink yet: the SoC reached 80-82 C and soft-throttled during runs (`vcgencmd get_throttled` = 0xe0000/0xe0008),
so these are pessimistic. Micro-benchmarks waited for < 62 C before each model.

| Pipeline (`scripts/benchmark.py`, 60 frames) | p50 | p95 | FPS | Peak RSS |
|---|---|---|---|---|
| steel (classical anomaly 22 ms + YOLOv8n INT8 137 ms) | 159 ms | 240 ms | 5.5 | 123 MB |
| mvtec metal_nut (PatchCore INT8, 256 px) | 284 ms | 380 ms | 3.3 | 187 MB |
| mvtec bottle (PatchCore INT8, 256 px) | 270 ms | 303 ms | 3.5 | 187 MB |
| mvtec screw (PatchCore INT8, 320 px) | 363 ms | 531 ms | 2.6 | 203 MB |

| Model (cool start) | FP32 p50 | INT8 p50 |
|---|---|---|
| PatchCore metal_nut, fused | 333 ms | **252 ms** |
| YOLOv8n NEU, 320 px | 143 ms | **136 ms** |

INT8 still helps on the A72 even without the ARMv8.2 dot-product instructions (`/proc/cpuinfo`: fp asimd crc32 only),
but much less than on newer cores; the Pi 4 runs the vision stack ~13x slower than the M2.

### Pi 4 speed-ups (13 Nov)
**YOLO input 320 → 256 px** (same weights, re-exported; `configs/profiles/steel.yaml`). NEU-DET images are 200x200,
so 320 px only upsampled them:

| YOLOv8n INT8 input | mAP@0.5 (ONNX val) | Pi 4 p50 |
|---|---|---|
| 320 px (before) | 0.700 | 130 ms |
| **256 px (deployed)** | **0.707** | **87 ms** |
| 224 px | 0.694 | 71 ms |
| 192 px | 0.685 | 54 ms |

Steel profile end to end: 159 → **109 ms p50, 5.5 → 8.3 FPS**.

**ExecuTorch (XNNPACK) for vision, evaluated and not adopted on the Pi 4.** `scripts/export_executorch_vision.py`
exports the fused PatchCore to .pte (FP32, and INT8 with the backbone quantised by PT2E; accuracy equal to ONNX:
AUROC 0.996, recall 94.6% for metal_nut INT8). On the Pi 4 it is slower than ONNX Runtime:

| metal_nut PatchCore | Pi 4 p50 |
|---|---|
| ONNX Runtime INT8 (deployed) | **252 ms** |
| ExecuTorch XNNPACK INT8 | 313 ms |
| ExecuTorch XNNPACK FP32 | 479 ms |

Reason (from `get_delegation_info`): the 3x3 average pool, the min over the 1024x4096 distance matrix, a sum and a
layout copy are not XNNPACK-delegated, so they run in single-threaded portable kernels and split the graph into 5
delegate calls. ExecuTorch stays the runtime for the LLM; `ExecuTorchBackend` (.pte) remains available in the app.

**Llama reply streaming.** `llama_server` streams tokens (`GEN ... stream=1` → `TOK` frames); the dashboard shows the
reply as it is written. On the Pi 4 the first words appear after ~11 s instead of a blank ~28 s wait. The final
reply still goes through the grounding check (rule-based fallback replaces it if rejected).

| Llama 3.2 1B advisor (ExecuTorch, XNNPACK 8da4w) | Pi 4 |
|---|---|
| Prefill / decode | ~10 / 3.5 tok/s |
| Advice, full prompt | 57-82 s |
| Advice with **prefix KV cache** (new) | **26-38 s** (first token after 11-21 s) |
| One-off warm-up at app start | 35 s |

### Vision + Llama running together on the Pi 4 (real app, steel profile, 3-4 min runs)
Everything above was measured with one component at a time. In the app both run at once on 4 cores:

| Setup | Vision FPS | Llama advice latency | Rule fallbacks |
|---|---|---|---|
| Alone (reference) | 8.3 | ~28 s | - |
| 4 + 4 threads, no cap | 3.0 | 74-90 s | 0 |
| 2 + 2 threads | 4.5 | 106 s | 0 |
| 4 + 4, 2 FPS cap, Llama nice 10 | 2.0 | 80-108 s | 1 (grounding) |
| **Vision 1 thread on core 0, Llama 3 threads on cores 1-3, 2 FPS cap** | **2.0** | **38-77 s** | **0** |

Two all-core thread pools on 4 cores stall each other (XNNPACK synchronises its threads on every op, so one
preempted thread holds up the whole op); lowering Llama's priority alone does not help. Pinning them to separate
cores does. The remaining spread is thermal: the Pi reported active throttling (`0xe0006`) in these runs.
Also changed: ONNX Runtime thread spinning is off by default (`vision.ort_spinning`), the LLM reply timeout is
180 s (240 s on the Pi), and the pinned setup is `configs/devices/pi4.yaml` (`--device`).

Single-core vision speed with `--device configs/devices/pi4.yaml`: steel 208 ms (4.6 FPS max, capped at 2),
PatchCore metal_nut 717 ms (1.4 FPS), screw 1,066 ms (0.9 FPS) with the desktop models, hence:

### Smaller PatchCore models for the Pi 4 (edge-speed sweep, 13 Nov)
Trained on the full MVTec train sets (Mac M2/MPS; the 256/4096 grid baseline reproduced at AUROC 0.904 vs
0.881 on the HPC), INT8 fused with `optimize_patchcore.py`, thresholds re-calibrated on each model's holdout.
Latency is set by input size and bank size only (same for every category); fused INT8, ONNX Runtime, 1 thread
pinned to one A72 core:

| Input px / memory bank | Pi 4 p50, 1 core |
|---|---|
| 320 / 4096 (desktop screw) | 1,142 ms |
| 256 / 4096 (desktop others) | 707 ms |
| 256 / 2048 | 508 ms |
| 224 / 4096 | 518 ms |
| 224 / 2048 | 364 ms |
| 192 / 2048 | 270 ms |

INT8 image AUROC per config (full test sets):

| Category | 256/2048 | 224/4096 | 224/2048 | 192/2048 | desktop model |
|---|---|---|---|---|---|
| bottle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| metal_nut | 0.997 | 0.993 | 0.990 | 0.990 | 0.997 |
| tile | 0.984 | 0.994 | 0.996 | 0.993 | 0.986 |
| grid | 0.911 | 0.899 | 0.859 | 0.789 | 0.892 |
| screw | 0.894 | 0.938 | 0.888 | 0.890 | 0.972 (320 px) |

Chosen for the Pi on 13 Nov (`device_overrides.pi4` in each MVTec profile, picked up by
`--device configs/devices/pi4.yaml`; models in `models/anomaly/pi4/`). **The grid and screw rows were replaced later
the same day by the PCA-128 models in the next sub-section**; bottle, metal_nut and tile are still these models.
Threshold scales chosen on the test set, as for the desktop profiles.
End-to-end FPS = `scripts/benchmark.py` on the Pi with vision on one core:

| Category | Pi model | Scale | Recall / FPR | Desktop model recall / FPR | Pi 4 FPS (1 core) |
|---|---|---|---|---|---|
| bottle | 224 / 2048 | 0.95 | 1.00 / 0.00 | 1.00 / 0.00 | 1.4 → **2.4** |
| metal_nut | 224 / 2048 | 0.95 | 0.97 / 0.05 | 0.98 / 0.09 | 1.4 → **2.5** |
| tile | 224 / 2048 | 0.95 | 0.95 / 0.00 | 0.98 / 0.12 | 1.4 → **2.4** |
| grid (superseded) | 256 / 2048 | 1.00 | 0.65 / 0.00 | 0.54 / 0.00 | 1.4 → **1.9** |
| screw (superseded) | 224 / 4096 | 0.90 | 0.77 / 0.07 | 0.87 / 0.00 | 0.9 → **1.8** |

bottle, metal_nut, tile and grid lose nothing (grid and tile even improve; coreset sampling varies run to run).
Screw was the real trade-off here (thin threads need resolution: 10 points of recall for 2x speed); the PCA-128
model below removes most of it (87% recall again, at 10% instead of 0% false alarms).

### Literature-driven improvements (13 Nov)
A review of papers/articles on edge anomaly detection, NEU-DET detectors and on-device LLMs (sources in
`docs/LITERATURE.md`) gave a ranked list of ideas; each was measured on the Pi 4 before adoption.

**PCA projection before the PatchCore kNN.** Profiling the fused INT8 model on one A72 core (ORT profiler,
224 px / 2048 bank): backbone convs 188 ms (49%), kNN matmul 162 ms (42%). Johnson-Lindenstrauss / PCA style
reduction of the 384-dim patch features (PCA fitted on the good training patches, applied as a 1x1 conv inside the
graph, kept FP32 in the INT8 model; `train_anomaly.py --proj-dim`) cuts the kNN cost by 3x, and the saved time buys
back resolution:

| Model (FP32 AUROC) | 128 dims | 64 dims | full 384 dims |
|---|---|---|---|
| screw 224 / 4096 | 0.939 | - | 0.946 |
| screw 256 / 4096 | 0.945 | - | - |
| screw 288 / 4096 | **0.972** | - | - |
| screw 320 / 4096 | 0.970 | 0.923 | 0.972 (desktop) |
| grid 256 / 4096 | 0.917 | - | 0.904 |
| grid 256 / 8192 | 0.918 | 0.910 | - |
| metal_nut / tile / bottle 224 / 4096 | 0.987 / 0.981 / 1.000 | - | 0.990 / 0.995 / 1.000 (224/2048) |

128 dims keeps ~94-97% of the variance and the accuracy; 64 dims loses accuracy. Pi 4, one core, fused INT8:

| Pi model | p50 | INT8 AUROC | Test recall / FPR | FPS (app, 1 core) |
|---|---|---|---|---|
| screw, before: 224 / 4096 / 384-d | 525 ms | 0.938 | 77% / 7% @0.90 | 1.8 |
| **screw, now: 288 / 4096 / 128-d** | 544 ms | **0.965** | **87% / 10% @0.93** | 1.7 |
| grid, before: 256 / 2048 / 384-d | 486 ms | 0.911 | 65% / 0% @1.0 | 1.9 |
| **grid, now: 256 / 4096 / 128-d** | **429 ms** | 0.911 | 67% / 0% @1.0 | **2.2** |

Grid as deployed since 16 Nov uses image gate 0.95: 79% / 9.5% (see "Runtime optimisations").
Screw on the Pi now matches the desktop 320 px model's AUROC (0.965 vs 0.972) at half its latency. Bottle, metal_nut and tile keep
their full-dimension 224/2048 models (already 2.4-2.5 FPS; PCA cost them a little accuracy there).

**Tried and rejected (anomaly):**
- Foreground crop for screw (suggested for small parts): the screw already spans ~87% of the 1024 px frame
  (measured on train + test images), so a crop would add at most ~15% resolution. Not pursued.
- Rotation-augmented memory bank for screw (90/180/270 deg copies of the good images, `--rot-aug 3`): AUROC
  0.9375 vs 0.9715 without. With a fixed bank size the coreset is spread over orientations that never occur.
- Bigger bank for screw at 288 px (8192): 0.969, no better than 4096 and slower.

**LLM: generated text and prompt (Pi 4, 3 Llama threads on cores 1-3, same 3 cases, greedy):**

| Change | Reply time (3 cases) | Generated tokens | Time to first token |
|---|---|---|---|
| Before | 30.2 / 27.5 / 37.7 s | 67 / 57 / 57 | 13.4 / 13.2 / 21.9 s |
| `Log:` line written by code, pixel boxes dropped from the prompt | 23.2 / 23.7 / 29.6 s | 48 / 47 / 41 | 11.4 / 11.7 / 18.6 s |
| + class-notes KV cache, same class as previous request | **18.7 s** | 50 | **6.5 s** |

- The `Log:` line (verdict, severity, type, zone, score) only restates facts the app has, so `log_line()` writes it
  (structured-output / "jump-forward" idea from constrained-decoding work): ~20% fewer decoded tokens, and the
  log is now exact (the model had dropped the severity in one case). The grounding check now judges only the
  model's own lines.
- Second KV-cache level in `llama_server`: the prompt puts the defect class's reference notes before the
  per-frame findings, and the server keeps the KV cache of prefix + notes. A repeat of the same class prefills
  ~67 tokens instead of ~120-200 (generalises the earlier static-prefix cache; cf. automatic prefix caching).
- Output-vocabulary pruning (ExecuTorch `output_prune_map`, 24,386 of 128,256 tokens kept,
  `scripts/build_vocab_map.py`): .pte 803 -> 693 MB and decode 12% faster (0.25 -> 0.22 s/token in all three cases),
  but reply time unchanged within noise because half of it is prefill, and 2 of 3 replies changed wording.
  Kept as an option (`models/llm/llama3_2_1b_instruct_8da4w_v24k.pte`), not the default.
- Not pursued: prompt-lookup speculative decoding. On the A72 prefill runs at only ~3x the decode rate (no int8
  dot-product), so verifying k drafted tokens costs ~(k+1)/3 decode steps; with the short, paraphrasing replies the
  break-even acceptance is marginal, and it needs a full-logits re-export plus runner changes.

### Runtime optimisations (16 Nov; real app on the Pi 4, `--device configs/devices/pi4.yaml`)

**PatchCore verdict gate (bug fix).** Any heatmap pixel over the *pixel* threshold made a defect, so the app
flagged good parts that the calibrated *image* threshold passes. Now defects are reported only if the image score
is >= `vision.anomaly.image_gate` (1.0; grid on the Pi 0.95). App verdicts on the full MVTec test sets, Pi models:

| Category | False alarms before → after | Recall before → after |
|---|---|---|
| bottle | 35.0% → **0.0%** | 100% → 100% |
| metal_nut | 31.8% → **4.5%** | 98.9% → 96.8% |
| tile | 21.2% → **0.0%** | 100% → 95.2% |
| grid (gate 0.95) | 9.5% → 9.5% | 84.2% → 78.9% |
| screw | 53.7% → **9.8%** | 100% → 86.6% |

The "after" column is the operating point the holdout numbers in section 2 / the PCA study were measured at
(e.g. screw 87% / 10%); before, the app ran at a much noisier point than reported. **This gate cost recall
(screw 100 → 87%, grid 84 → 79%, tile 100 → 95%) and was replaced the next day:**

**Smoothed image score (17 Nov).** The gate above compared the *raw* image score (the single largest patch
distance) with its threshold. One noisy patch sets that score, so it separates good from defective parts worse
than the heatmap the old "any region" rule looked at. The image score is now the maximum of the smoothed heatmap
(Gaussian, sigma 4 at model resolution), with its own threshold calibrated on the same held-out good training
images and the same rule (max x 1.05; `image_threshold_smooth` in each meta, `scripts/calibrate_smooth_score.py`;
new models get it from `train_anomaly.py` / `optimize_patchcore.py`). No retraining and no extra compute: the
smoothed heatmap was already computed for the regions.

| Pi model, image AUROC (full test set) | Raw max (before) | Smoothed max (now) |
|---|---|---|
| bottle | 1.000 | 1.000 |
| metal_nut | 0.990 | 0.993 |
| tile | 0.996 | 0.999 |
| grid | 0.911 | **0.954** |
| screw | 0.965 | **0.981** |

App verdicts on the full test sets (defects caught / good parts flagged), Pi models:

| Category | 15 Nov: any region | 16 Nov: raw-score gate | **17 Nov: smoothed score (deployed)** | Scale | At scale 1.0 (holdout only) |
|---|---|---|---|---|---|
| bottle | 100% / 35.0% | 100% / 0.0% | **100% / 0.0%** | 1.00 | 100% / 0% |
| metal_nut | 98.9% / 31.8% | 96.8% / 4.5% | **98.9% / 9.1%** | 0.97 | 96.8% / 4.5% |
| tile | 100% / 21.2% | 95.2% / 0.0% | **100% / 9.1%** | 0.93 | 91.7% / 0% |
| grid | 84.2% / 9.5% | 78.9% / 9.5% | **87.7% / 9.5%** | 0.93 | 73.7% / 0% |
| screw | 100% / 53.7% | 86.6% / 9.8% | **96.6% / 7.3%** | 0.87 | 52.1% / 0% |

Desktop models with the same rule: bottle 100% / 0% (scale 0.95), metal_nut 98.9% / 9.1% (0.93), tile 100% / 9.1%
(0.91), grid 80.7% / 4.8% (0.95), screw 320 px 95.0% / 4.9% (0.91); image AUROC 1.000 / 0.997 / 0.995 / 0.947 /
0.985.

Caveats: the per-category scale (`threshold_scale`) is picked on the test set for <= 10% false alarms, as the
earlier scales were, and the good test sets are small (20-41 images, so one image is 2.4-5 points of false-alarm
rate). The last column is what the holdout calibration alone gives. Grid's `image_gate: 0.95` workaround is gone.

**Advice generated once per defect signature** (`edgeqc/advisor/service.py`). Signature = product, verdict and
per-class worst severity (untyped anomaly blobs ignored when a typed defect is present; 7 signatures over 717 NEU
frames instead of 23). Repeats reuse cached advice with their own log line; repeats of a signature still being
generated wait for that reply; an event pushed out of the full queue gets rule advice at once (it used to get
none). LLM replies persist in `runs/advice_cache.json` (keyed by model, prompt, KB, decoding). While the line is
idle (20 s without a request) the LLM pre-generates advice for every (class, severity).

**Vision borrows the LLM's cores while it is idle** (`vision.boost_threads: 4`, `app.idle_target_fps: 5`): a second
4-thread ONNX session per model, used only while the advisor is not generating. One-core vs 4-core latency on the
Pi: YOLO 256 px 198 → 96 ms, screw 547 → 182 ms, bottle 368 → 117 ms.

**Output path off the vision core**: overlay drawn at <= 640 px, JPEG encoded once for dashboard + saved image, on
a thread pinned to cores 1-3. For 1024 px screw frames the vision loop spent 81 ms on overlay + save + dashboard
JPEG (52 + 15 + 14); now 33 ms is left on it. Reading the PNG from disk (30 ms) is unchanged and disappears with a camera.
Also: SQLite WAL + in-memory verdict counts (dashboard poll no longer scans the table), latest-frame USB camera
grabber (no stale frames at 2-5 FPS; **not yet tested with a real camera**), llama_server in its own session
(Ctrl+C no longer kills it mid-reply).

Steel profile, 6 min (717 frames, all defective, 30 NEU images looping), cold advice cache. The first column is the
new service with a signature that counted every box; the old service was not re-run in this harness (it queued the
same signature again every 15 s and left frames pushed out of the queue without any advice):

| Frames whose advice came from | Signature = every box (23 signatures) | Signature = per-class worst severity (7, deployed) |
|---|---|---|
| Llama (generated + reused) | 122 (7 + 115) | **477** (6 + 471) |
| Rule-based (queue overflow + its repeats) | 589 | 160 |
| None yet at shutdown (waiting for a reply in progress) | 6 | 80 |

| Real app on the Pi 4 | Result |
|---|---|
| Steel, 5 min, after restart with the saved advice cache | **5.0 FPS** (the `idle_target_fps` cap) from ~30 s in, Llama advice at once for all 1,372 frames, none dropped. Before: 2.0 FPS, 38-77 s per advice (section "Vision + Llama running together") |
| Max FPS, rule advisor, `target_fps=0`: vision on 1 core → 4 cores (boost) | screw 1.48 → **3.37**, bottle 2.16 → **4.37**, steel 3.76 → **5.51** |
| Idle bottle line (good parts only): LLM pre-generation | 8 signatures, ~30 s each (first 77 s incl. warm-up), all passed the grounding check; 1.7 FPS meanwhile, then 4.6-4.8 FPS |
| Llama first-time reply while vision runs on core 0 | 42-68 s (6 replies) |

Thermal: boosted runs reach 81-82 C without a fan (throttle point 80 C) — the fan is now needed to hold 5 FPS.

## 6. Report measurements on the Pi 4 (19 Nov; `scripts/pi_report_bench.sh`, raw output in `runs/report_bench/`)

**No fan: the Pi sat above 80 C for 93-99% of these runs (peak 86.7 C, `vcgencmd get_throttled` = 0xe0008, i.e.
soft temperature limit active).** All latencies below are therefore pessimistic; re-run after fitting the fan.

### Advice quality: 20 scripted cases (`scripts/eval_advisor.py`)
All 10 knowledge-base classes (6 NEU steel types, crack, 3 untyped anomaly kinds), 17 single-defect + 3 two-defect
cases, REVIEW and REJECT verdicts. Llama 3.2 1B (8da4w, resident `llama_server`, cores 1-3, no vision load).
Automatic checks (`runs/report_bench/advisor_eval/results.json`):

| Check (of 20) | Llama | Rule baseline |
|---|---|---|
| Exactly the 3 lines Cause / Action / Check | 20 | 20 |
| Passes the app's grounding check (else the operator sees the rule advice) | 17 → **19** after the stem fix below | 20 |
| Names the worst defect type | 15 | 20 |
| Cause restates a knowledge-base cause | 14 | 20 |
| Action or Check restates a knowledge-base action | 20 | 20 |
| Names a defect class not in the findings / invents a number / contradicts the verdict | **0 / 0 / 0** | 0 / 0 / 0 |

Rating 1-5 (by Claude against the knowledge base, not a domain expert; rubric and per-case notes in
`advisor_eval/ratings.json`): **mean 4.15** — 7 x 5, 8 x 4, 5 x 3, none below 3. Typical weak points: the Cause line
restates the defect description instead of a cause (cases 10, 13, 14), one inverted cause ("excessive descaling",
KB: insufficient), and in two-defect cases the cause can follow the second defect (case 18). The rule baseline is
correct by construction but is a fixed template ("Cause: crazing (CRITICAL), likely excessive rolling reduction.").

Speed (no vision load): reply **median 25.7 s** (15.6-36.1 s), first word **median 11.6 s** (5.9 s when the class's
reference notes are already in the KV cache, 23 s for two-defect prompts), 49 tokens per reply, decode 3.8 tok/s,
warm-up (model load + static prefix) 36.6 s.

Grounding-check fix: 2 of the 3 rejected replies were fine but wrote "a patch" / "patching" for *patches*;
`is_grounded` now also accepts word stems (19/20 pass; the remaining one never names the defect).

### Per-stage latency of one inspection (`scripts/profile_stages.py`, median ms, vision on core 0)

| Profile | image decode | anomaly | classifier | fusion + severity | overlay | JPEG | SQLite log | total |
|---|---|---|---|---|---|---|---|---|
| steel (200 px NEU images), 1 core | 1.0 | 16.8 | 197.9 | 0.1 | 3.6 | 0.8 | 0.6 | 221 |
| steel, 4-core boost | 1.0 | 16.8 | **91.7** | 0.1 | 3.7 | 0.8 | 0.6 | 115 |
| bottle (900 px), 1 core / boost | 37.8 | 383 / **132** | - | 0.1 | 42.3 | 5.8 | 0.5 | 470 / 219 |
| metal_nut (700 px) | 28.0 | 379 / **127** | - | 0.0 | 37.3 | 5.8 | 0.5 | 451 / 198 |
| tile (840 px) | 41.6 | 403 / **140** | - | 0.1 | 40.8 | 5.9 | 0.5 | 493 / 229 |
| grid (1024 px) | 29.3 | 434 / **150** | - | 0.0 | 42.2 | 6.2 | 0.4 | 513 / 228 |
| screw (1024 px) | 29.1 | 563 / **198** | - | 0.1 | 42.3 | 5.5 | 0.7 | 642 / 277 |

The model dominates. In the app the overlay + JPEG run on the output thread (cores 1-3), not on the vision core;
with a camera, decode is replaced by the grabber thread (`LatestFrameCamera`).

### Full app with Llama loaded (web UI on, cold advice cache, 900 frames, `scripts/mem_monitor.py` every 1 s)

| | steel (NEU val folder) | screw (6-image folder, 3 good / 3 defective) |
|---|---|---|
| **Peak memory (PSS), app + llama_server** | **1.05 GB** (app 149 MB + Llama 898 MB) | **1.15 GB** (app 307 MB + Llama 840 MB) |
| Lowest MemAvailable (of 3.8 GB) / swap used | 1.59 GB / 5 MB | 1.46 GB / 5 MB |
| Mean FPS over the run | 2.1 (900 frames in 423 s) | 1.6 → **2.5** after the retry fix below |
| Llama replies while vision runs (first-time signatures) | 7 replies, 46-77 s | 82.5 s (first, includes warm-up overlap), 50.8 s |
| Advice reused / joined / rule for queue drops | 465 / 386 / 42 | 353 / 94 / 0 |

About 1.2 GB of the Pi's RAM is used by other services before the app starts, so the app has ~2.3 GB of headroom.

Retry fix (found in this run): when the grounding check rejected a reply, the app showed rule advice and retried the
same signature after `cooldown_s` (15 s). With greedy decoding the retry returns the same reply, so the screw run spent
7 x 42 s of Llama time on identical rejected replies (and vision lost its borrowed cores). Now a rejected greedy reply
keeps the rule advice for the session (not saved to disk); only timeouts / errors are retried. Screw re-run: Llama
generations 9 → 3, boosted frames 344 → 638, 900 frames in 576 → 362 s.

### Report figures (`scripts/report_figures.py`, `docs/figures/`)
- `steel_confusion.png` — app verdicts on NEU-DET val (361 images, full steel pipeline incl. fusion): **98.3%** named
  correctly; 3 flagged without a type, 2 missed (PASS), 1 pitted surface named patches.
- `steel_yolo_pr_curve.png`, `steel_yolo_box_confusion.png` — box-level PR curves / confusion matrix of the deployed
  INT8 256 px model (mAP@0.5 0.707, Ultralytics val on ONNX Runtime).
- `mvtec_roc_pi4.png`, `mvtec_roc_desktop.png` — image-score ROC per category with the app's operating point
  (AUROC Pi: bottle 1.000, metal_nut 0.993, tile 0.999, grid 0.954, screw 0.981).

### Precision / recall / F1 (24 Nov; from the confusion matrix and the MVTec app verdicts above)

**Precision / recall / F1** (the metrics the PS5 mentor asked for; app verdicts, Pi models):

| | Precision | Recall | F1 |
|---|---|---|---|
| Steel, defect type named (NEU val, macro over 6 classes) | 0.997 | 0.983 | **0.990** |
| Steel, YOLOv8n INT8 box level (IoU 0.5, Ultralytics val) | 0.664 | 0.664 | 0.664 |
| MVTec bottle | 1.000 | 1.000 | **1.000** |
| MVTec metal_nut | 0.979 | 0.989 | **0.984** |
| MVTec tile | 0.966 | 1.000 | **0.982** |
| MVTec grid | 0.962 | 0.877 | **0.917** |
| MVTec screw | 0.975 | 0.966 | **0.970** |

MVTec: a defective part flagged REVIEW/REJECT is a true positive, a good part flagged is a false positive.

### Ablation of the hybrid (`scripts/ablation.py`, desktop models, 23 Nov; `runs/report_bench/ablation.json`)

| Steel (NEU val, 361 images) | Caught | Correct type |
|---|---|---|
| Hybrid (deployed) | 99.4% | 98.3% |
| Classifier only | 98.6% | 98.3% |
| Classical anomaly only | 71.5% | 0% |

MVTec, classical OpenCV detector instead of PatchCore (caught / good flagged): bottle 96.8 / 100%, metal_nut
100 / 100%, tile 2.4 / 0%, grid 0 / 0%, screw 100 / 100% (PatchCore: see the table at the top). On NEU the anomaly
stage adds 0.8 pt (the classifier knows all six classes); on MVTec a learned anomaly model is essential.

## 7. Open items
Promised in `docs/PLAN.md` for the report and not done yet:
- Advice quality rated by a person (section 6 has automatic checks + a model rating); grounding-check rejection rate
  over a long live run.
- A live-camera run on the Pi (USB or Pi camera); everything so far uses image folders.
- Hailo backend is untested (no hardware; the AI HAT+ needs a Pi 5).

Technical:
- Heatsink + fan on the Pi 4, then re-run section 5 (boosted vision reaches 81-82 C; throttles at 80 C).
- Pre-generation blocks a real first-time defect for up to one reply (~30-70 s); llama_server could take an abort.
- Grid: still the weakest category (AUROC 0.954 with the smoothed score, 88% recall at 9.5% false alarms).
  Literature: STFPM-R18 reports 0.98 on grid (no kNN) — try it for grid only.
- Threshold scales are chosen on the test set; a larger held-out good set (or cross-validation) would make the
  operating points defensible without test labels.
- Crazing / rolled-in scale: weakest NEU classes; bigger models/resolution don't help (see sweep). Literature agrees (box-level
  labels for diffuse textures). The 3-seed recipe study (section 1) found no gain beyond seed noise (±1.4 pt mAP).
  Label audit → per-class confidence thresholds fixed the image-level result (crazing named correctly 46 → 97%).
  Open: measure false rejects on defect-free steel images (none in NEU).
