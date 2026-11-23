# EdgeQC — Plan & Architecture

**Arm Bharat AI-SoC Challenge 2026-27 · Problem Statement 5: Real-time industrial defect detection & quality-control assistant**
Phase 1 (Open Qualifying) deadline: **23 Nov 2026** — source on GitHub + demo video + technical report.

> **Status, 16 Nov.** This is the plan written on 4 Nov, kept for the design rationale; places where the build
> turned out differently are marked "Now:". The target device is a **Raspberry Pi 4 Model B (4 GB, Cortex-A72)**,
> not a Pi 5. Measured results are in [RESULTS.md](RESULTS.md); what is still missing for the report is listed in
> its "Open items".

---

## 1. What we are building

An on-device QC assistant that watches parts on a line and, for every frame:

1. **Detects** surface anomalies (anything that doesn't look like a good part, including never-seen defects).
2. **Classifies** the defect type and assigns a **severity** (MINOR / MAJOR / CRITICAL).
3. Issues a **verdict**: PASS / REVIEW / REJECT.
4. **Alerts** the operator with an actionable recommendation written by **Llama 3.2 1B Instruct running on-device through ExecuTorch**.
5. **Logs** every inspection (SQLite + annotated images, CSV export).
6. Works **offline**: no network or cloud calls at runtime.

### Design decisions

| Decision | Choice | Why |
|---|---|---|
| Vision approach | **Hybrid**: unsupervised anomaly detector, then a supervised defect classifier, fused together | The anomaly stage catches unknown defects. The classifier names known ones. Anything flagged but unnamed goes to REVIEW, which is the safe default for QC. |
| Anomaly model | PatchCore-lite (ResNet-18 layer2+3 features, coreset memory bank, kNN distance) | Strong MVTec AD results. Trains from good parts only. The backbone is the only NN, and it exports cleanly to ONNX and HEF. |
| Classifier | YOLOv8n trained on NEU-DET (6 steel defect classes) | Small, fast, exports to ONNX/HEF. NEU gives bounding boxes. |
| Severity | Transparent rule engine (class criticality, defect area, anomaly strength) | Auditable. Operators and judges can see why a part was rejected. The LLM never decides pass or fail. |
| LLM role | **Advisory only.** It takes structured findings plus knowledge-base notes and writes the cause, action and process check | Keeps the safety-relevant decision deterministic and grounds the LLM to reduce hallucination. |
| LLM runtime | Llama 3.2 1B Instruct → ExecuTorch `.pte` (XNNPACK, 8da4w quant) → resident `llama_server` (our runner on ExecuTorch's `TextLLMRunner`; `llama_main` is the fallback) | Required by the PS. The 1B model at 4-bit is 803 MB. Now: on the Pi 4 it decodes ~3.5 tok/s, so a reply takes 23-30 s (42-68 s while vision runs). |
| Real-time | Vision runs in the main loop. The LLM runs asynchronously in a worker thread; advice is generated once per defect signature and reused | LLM latency must never stall inspection. Planned ≥ 5–10 FPS (Pi 5). Now, Pi 4: 5 FPS on steel and 3.4-4.4 FPS on MVTec while the Llama is idle, ~2 FPS while it generates. |
| UI | FastAPI + one static HTML page: MJPEG stream, verdict, advice, log, ack button | Opens from any phone or PC on the factory LAN with no app install. |
| Fallbacks | Classical OpenCV anomaly detector and a rule-based advisor | The app runs end-to-end on day 1 with no trained models, and the system degrades gracefully. |

---

## 2. Architecture

```
                         ┌──────────────────────── VISION PIPELINE (real-time) ─────────────────────────┐
 Camera / video /        │                                                                              │
 image folder  ──frame──▶│  Preprocess ─▶ [A] Anomaly detector ──heatmap+regions──┐                     │
 (OpenCV/Picamera2)      │              (PatchCore-lite | classical)              ├─▶ Fusion ─▶ Severity ─▶ Verdict
                         │           ─▶ [B] Defect classifier ──typed boxes───────┘   (IoU)     rules      PASS/REVIEW/REJECT
                         │              (YOLOv8n)                                                       │
                         │  Backend: ONNX Runtime (Arm CPU)  ── or ──  HailoRT (Hailo-8/8L NPU)         │
                         └───────────────────────────────┬──────────────────────────────────────────────┘
                                                         │ InspectionResult (structured)
                         ┌───────────────────────────────┼──────────────────────────────────────────────┐
                         │                               ▼                                              │
                         │  SQLite log + images   Web dashboard (MJPEG, alerts, ack)                    │
                         │                               ▲                                              │
                         │                               │ advice text (async)                          │
                         │  ┌──────────── ADVISORY PIPELINE (async, Arm CPU only) ───────────────┐      │
                         │  │ queue ─▶ prompt builder (+ defect KB grounding) ─▶ ExecuTorch      │      │
                         │  │          Llama 3.2 1B Instruct (.pte, XNNPACK, 4-bit)              │      │
                         │  │          fallback: rule-based advisor                              │      │
                         │  └────────────────────────────────────────────────────────────────────┘      │
                         └──────────────────────────────────────────────────────────────────────────────┘
```

### Two separate pipelines (as the PS asks us to document)

| | Vision pipeline | Advisory pipeline |
|---|---|---|
| Models | PatchCore backbone (ResNet-18), YOLOv8n | Llama 3.2 1B Instruct |
| Runtime | ONNX Runtime on CPU (INT8 models); HailoRT on the Hailo NPU (`.hef`) is coded but untested | **ExecuTorch** (`llama_server` / `llama_main`, XNNPACK delegate) |
| Hardware | Arm Cortex-A72 (Pi 4): core 0, plus cores 1-3 while the Llama is idle. Hailo NPU path untested (the AI HAT+ needs a Pi 5) | Arm Cortex-A72 CPU, cores 1-3 |
| Timing | Synchronous, per frame | Asynchronous, per defect event |
| Code | `src/edgeqc/vision/` | `src/edgeqc/advisor/` |

Switching vision to Hailo is meant to be a config change (`vision.backend: hailo` and `.hef` model paths) with the advisory pipeline unchanged; this path has never been run on hardware. ExecuTorch was also tried for vision and was slower than ONNX Runtime on the Pi 4 (RESULTS.md section 5), so it stays the LLM runtime only.

### Fusion logic
- A classifier box that overlaps an anomaly region becomes **fused**: it keeps the class label from B and the anomaly strength from A.
- A classifier box with no anomaly region keeps source **classifier**. It is reported, and its severity comes from the class.
- An anomaly region with no classifier box becomes `anomaly_linear` / `anomaly_spot` / `anomaly_area`, based on its shape. It goes to at least REVIEW because the classifier doesn't know it.

### Severity → verdict
- Base severity comes from class criticality (configurable per product, e.g. `crazing: MAJOR`).
- Severity is raised to MAJOR or CRITICAL by defect area fraction or by anomaly score relative to threshold.
- Any MAJOR or CRITICAL → **REJECT**. Any MINOR or unknown anomaly → **REVIEW**. Otherwise → **PASS**.
- Now: PatchCore reports regions only when the image score (max of the smoothed heatmap) reaches its calibrated threshold (`vision.anomaly.image_gate`, `threshold_scale`), and the YOLO confidence threshold is per class (`class_conf_thresholds`).

---

## 3. Repository layout

```
configs/            default.yaml (all tunables), defect_kb.yaml (causes/actions grounding), profiles/, devices/pi4.yaml
src/edgeqc/
  capture/          frame sources: USB/CSI camera, Picamera2, video, image folder
  vision/           backends (ONNX/ExecuTorch/Hailo), anomaly, classifier, fusion, severity, pipeline, overlay
  advisor/          prompt builder, ExecuTorch Llama client, rule fallback, async worker with advice cache
  storage/          SQLite inspection log + CSV export
  ui/               FastAPI server + static dashboard
  app.py            main loop / CLI
native/             llama_server.cpp (resident ExecuTorch Llama runner: KV-cache reuse, streaming)
scripts/            dataset prep, training/export, Llama→ExecuTorch export, benchmark, synthetic samples, hpc/, kaggle/
tests/              unit + smoke tests (run without any trained models)
docs/               this plan, setup guides, report material
```

---

## 4. Milestones (plan of 4 Nov; status as of 16 Nov)

| Dates | Milestone | Exit criteria | Status |
|---|---|---|---|
| **4–6 Nov** | Scaffold, environment, datasets | `pytest` green; app runs end-to-end on synthetic samples with classical detector + rule advisor; MVTec AD + NEU-DET downloaded | Done |
| **6–9 Nov** | Llama on ExecuTorch (Mac) | ExecuTorch built, Llama 3.2 1B Instruct exported to `.pte` (8da4w), `llama_main` produces advice from our prompt; tokens/s recorded | Done (4 Nov) |
| **6–10 Nov** | Vision models | PatchCore-lite trained on 3–5 MVTec categories (image AUROC reported); YOLOv8n on NEU-DET (mAP@0.5 reported); both exported to ONNX and wired into the pipeline | Done (5 categories) |
| **~10 Nov** | Mentor doubt session | Questions ready (see §9) | Not recorded here |
| **10–14 Nov** | Integration + hardening | Fusion tuned, severity thresholds calibrated, async advisor stable, dashboard polished, logs/CSV verified | Done; advisor reworked 16 Nov |
| **14–17 Nov** | Edge optimisation | INT8 ONNX (static quant) for vision, thread tuning, benchmark table (latency p50/p95, FPS, RAM) on the device | Done on the Pi 4 (RESULTS.md section 5); full-app RAM with Llama not yet measured |
| **17–21 Nov** | Deliverables | Demo video (3–5 min), technical report, README polish, reproducibility check from a clean clone | **Open** |
| **22–23 Nov** | Buffer + submit | GitHub public, release tag, submission form | **Open** (the project is not yet a git repository) |

## 5. Suggested task split (adjust to team size)

| Owner | Area |
|---|---|
| A | Anomaly detection: PatchCore training/eval/export, threshold calibration |
| B | Defect classifier: NEU-DET prep, YOLO training/export, fusion tuning |
| C | Llama + ExecuTorch: build, export, quantisation, prompt design, latency |
| D | App/UI/logging, benchmarking, demo video, report editing |

## 6. How we will be judged (PS5 mentoring video and slides, Arm Education, 2 Nov 2026)
The slides' "suggested evaluation lens". Every deliverable should provide evidence for each one:

| Lens (slide wording) | Our evidence |
|---|---|
| **Detection quality**: misses vs false alarms | Recall (miss rate) and false-alarm rate at the operating threshold, not only AUROC; threshold sweep / precision-recall curve; mAP@0.5 (NEU-DET); confusion matrix; hybrid ablation |
| **Real-time on device**: measured latency, FPS | `scripts/benchmark.py` table (p50/p95 latency, FPS, RAM) **on the target device** |
| **Useful advice**: grounded in findings | KB-grounded prompt with structured findings; automatic grounding/format guard (`advisor/prompts.py::is_grounded`) with the rejection rate reported; 20 scripted cases scored 1–5, Llama vs rule baseline (**not done yet**) |
| **Operator UX**: clear alerts, easy log | Dashboard: colour-coded verdict, boxes with location, advice streamed as it is written (instant for a defect type seen before), ack button, log table with one log entry per inspection, CSV export |
| **Rigour**: both runtimes documented | Report has separate sections for the vision runtime (ONNX Runtime CPU / HailoRT NPU) and the advisory runtime (ExecuTorch on the Arm CPU) |

Deliverables, as the slides define "good": **Source code**, runnable and documented. **Application**, *working on the
target device*. **Demo video** showing detection, alerts and logging. **Technical report** covering design, runtimes,
results and **limits**.

Other guidance from the video and slides:
- "Give Llama **structured findings** (defect class, severity, location, confidence), not raw pixels. It explains the defect, suggests actions and **writes the log entry**." `advisor/prompts.py` sends the class, severity, zone and area plus KB notes, and Llama replies `Cause / Action / Check`. **Deviation since 13 Nov:** the `Log:` line (stored in the `log_entry` column) is written by code (`log_line()`) from the same findings, not by Llama, and the prompt no longer carries pixel boxes or confidences. This cut reply time by ~22% on the Pi 4 and made the log exact (the model had dropped the severity in one test). The slide wording asks for Llama to write the log entry, so either say this openly in the report or switch it back; see question 7 in §9.
- Anomaly detection is listed "for few-defect-sample settings", which supports our hybrid choice.
- If Hailo is used, the report must document the HailoRT (vision) and ExecuTorch (Llama) pipelines **clearly and separately**.
- Datasets mentioned: MVTec AD, NEU surface defect, KolektorSDD. Optional tools: OpenCV, Hailo SDK, YOLO / EfficientDet, anomaly-detection frameworks.

## 7. Metrics for the report
- **Detection quality**: image AUROC and pixel AUROC (MVTec), mAP@0.5 (NEU-DET), and a confusion matrix for defect types.
- **System**: per-stage latency (preprocess / anomaly / classifier / fusion), end-to-end FPS, peak RAM, and model sizes.
- **LLM**: time-to-first-token, tokens/s, `.pte` size, RAM; quality check of the advice on 20 scripted cases (relevance / actionability, scored 1–5).
- **Ablations**: classical vs PatchCore, FP32 vs INT8, classifier on/off (the value of the hybrid).

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **No Pi 5 before the deadline (HIGH: the slides say the application must work "on the target device")** | Resolved 12 Nov with a **Raspberry Pi 4 Model B (4 GB)**: everything runs on it, at 2-5 FPS instead of the planned 5-10 and with 23-68 s Llama replies (cached per defect type). It has no heatsink or fan and throttles at 80 C; a fan is the cheapest remaining speed-up |
| Llama weights are gated (Meta licence) | Request access on Hugging Face / llama.com **today**; approval can take hours to days |
| ExecuTorch build friction | Pin one ExecuTorch release tag; follow its `examples/models/llama` README exactly; the rule advisor keeps the demo alive meanwhile |
| LLM too slow | 1B model, 4-bit, async; advice is not on the critical path. Now: prompt KV caches in `llama_server`, token streaming, advice cached per defect signature (kept across restarts) and pre-generated while the line is idle |
| Dataset domain gap vs demo | Demo with dataset images in a "virtual conveyor" (image-folder source) plus a live camera on printed samples |
| YOLO licence (Ultralytics AGPL-3.0) | Acceptable for an open-source challenge repo; noted in README |

## 9. Questions for the mentor session
1. Is a demo on an Arm64 Mac / generic aarch64 Linux acceptable for Phase 1 if a Pi 5 is not yet available?
2. Is Llama 3.2 1B acceptable, or do they expect 3B or a specific variant?
3. Must the Llama `.pte` be run through the C++ `llama_main` runner, or are the ExecuTorch Python bindings fine?
4. Preferred dataset(s) for evaluation, and is a multi-dataset hybrid valued?
5. Report format or length expectations? (The slides say: design, runtimes, results, limits.)
6. For "misses vs false alarms", is there a preferred operating point, e.g. recall ≥ 99% at minimum false alarms?
7. Is it acceptable that the one-line log entry is templated by code from the structured findings (Llama writes the cause, action and check), or must Llama write the log entry itself?

## 10. Technical report outline (mirrors the slide "design, runtimes, results, limits")
1. Problem and domain choice
2. **Design**: architecture, hybrid vision, defect record schema, severity rules, prompt design and grounding
3. **Runtimes**: (a) vision pipeline (ONNX Runtime on CPU / HailoRT on NPU); (b) advisory pipeline (Llama 3.2 1B, ExecuTorch, XNNPACK, 8da4w). Kept as separate sections
4. **Results**: misses vs false alarms, AUROC/mAP, latency/FPS/RAM on device, LLM tokens/s, advice-quality scores
5. **Limits**: dataset domain gap, threshold sensitivity, LLM latency and hallucination risk, lighting dependence; future work

## 11. Demo video outline (3–5 min)
1. Problem and architecture slide (two pipelines).
2. Live dashboard: a stream of good parts → PASS.
3. Defective parts → boxes plus heatmap, type, severity, REJECT; Llama advice streams in (first words after 6-11 s on the Pi 4 for a new defect type; at once for a type seen before).
4. Unknown defect → REVIEW (shows the value of the hybrid).
5. Operator acknowledges; show the SQLite log and CSV export.
6. Benchmarks table; airplane mode / no network to show it runs fully on-device.
