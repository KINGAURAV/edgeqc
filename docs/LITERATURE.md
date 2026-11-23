# Literature review → what we tried (13 Nov 2026)

Three short reviews (papers, arXiv, benchmark READMEs, vendor docs) were done for the three parts of EdgeQC, each
with a ranked list of ideas. Measured outcomes are in `docs/RESULTS.md` ("Literature-driven improvements").
Citations were checked against the sources on 19 Nov 2026 (arXiv PDFs, READMEs, model cards); corrections from that check are
applied below. Rows marked *abstract only* could not be checked against the full text.

## 1. Anomaly detection on an Arm CPU (PatchCore-lite)

| Source | Key point for us |
|---|---|
| Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022, arxiv.org/abs/2106.08265 | p=3 neighbourhood and layer2+3 are the right defaults; coreset 1% costs screw ~1.7 pt; JL random projection is used only for coreset selection |
| anomalib benchmark READMEs (open-edge-platform/anomalib) | PatchCore R18: grid 0.947, screw 0.943; STFPM R18: grid 0.982 / screw 0.835; EfficientAD-S: grid 1.000 / screw 0.960 |
| Batzner et al., *EfficientAD*, WACV 2024, arxiv.org/abs/2303.14535 | strong on grid/screw but heavy: EfficientAD-S is 8M parameters / ~76 GFLOPs (Table 16); pretrained teacher in anomalib releases |
| Stropeni et al., *Efficient Visual Anomaly Detection at the Edge* (PatchCore-Edge / PatchCore-Lite), 2026, arxiv.org/abs/2603.20288 | MobileNetV2 exact-kNN PatchCore ~11x faster than WRN-50 PatchCore on an x86 CPU (345 → 30 ms, image AUROC 0.98 → 0.95, Table II); a product-quantised bank hurts (0.95 → 0.86 MVTec, 0.86 → 0.68 VisA) and is slower (30 → 130 ms) |
| Bonazzi et al., *TinyGLASS*, AICAS 2026, arxiv.org/abs/2603.16451 | ResNet-18 layer2+3 features, INT8 on the IMX500 (Raspberry Pi AI Camera on a Pi 5); INT8 costs 0.4 pt image AUROC (94.6 → 94.2) and 2.0 pt pixel AUROC (Table II) |
| Defard et al., *PaDiM*, ICPR 2020 workshops, arxiv.org/abs/2011.08785 | random selection of 100 of 448 feature dims costs little accuracy (pixel AUROC 97.1 → 96.7, Table II) → motivates dimension reduction. Note: in PaDiM's Gaussian model PCA-100 did *worse* than random (93.5); our PCA is in front of a kNN and we measured it ourselves (below) |

Tried: **PCA 384→128 before the kNN (adopted: screw 0.938 → 0.965 at the same Pi latency, grid 12% faster)**,
foreground crop (not useful: screw fills the frame), rotation-augmented bank (worse). Open: STFPM-R18 for grid,
EfficientAD-S (likely too slow on one A72 core).

## 2. Steel-defect detector (YOLOv8n, NEU-DET)

| Source | Key point for us |
|---|---|
| Ezeji & Rijal, 2026 (unrefereed preprint), arxiv.org/abs/2606.07659 | YOLOv8n..l within 0.9 pt mAP50 (73.2–74.1, single 50-epoch runs); crazing / rolled-in scale stay at ~48–56% AP regardless of size (Table V) |
| hazemaaa/steel-defect-yolo11n (Hugging Face model card) | test mAP50 0.783 ± 0.011 over 5 seeds; the card treats differences < ~3 pt as ties (±1.4 pt noise floor on 180 test images) |
| Ma et al., *A lightweight algorithm for steel surface defect detection using improved YOLOv8*, Sci. Rep. 15:8966, 2025 (PMC11910532) | cleanest ablation: GhostNet ±0 mAP50 at −37% FLOPs (8.1 → 5.0 GFLOPs); SIoU +1.1; crazing improves only 45.6 → 47.5 AP even with all changes |
| FusionKD (Information Fusion; VLM → detector distillation), YOLO-DCF (dual distillation), YOLOv11-EMD (Mathematics 13:2769, 2025) — *abstract only* | FusionKD 77.8–80.0 NEU-DET mAP50; YOLO-DCF +1.6 pt; YOLOv11-EMD +1.6 pt with multi-stage transfer from NEU-DET + Severstal. Gains are below the ~3 pt seed noise above |
| YOLOBench, 2023, arxiv.org/abs/2307.13901 | YOLO latency on Pi 4 CPU (TFLite/XNNPACK) |

Matches our own sweep (bigger model / resolution doesn't help). Tried (Kaggle, 15 Nov): 3 seeds × {baseline,
grayscale-suited aug (hsv_h = hsv_s = 0, flipud 0.5, mixup 0.1) + 200 epochs + close_mosaic 30} →
**not adopted**: mAP@0.5 0.729 ± 0.014 vs 0.732 ± 0.013, crazing +2.6 / rolled-in scale −3.6, all within seed noise
(RESULTS.md section 1). Label audit (16 Nov): crazing boxes are arbitrary patches of a whole-image texture → low box confidence, not
confusion; per-class confidence thresholds (crazing 0.15, rolled-in 0.20) raise image-level accuracy 86.7 → 97.5%
without retraining (RESULTS.md section 1).

## 3. Llama 3.2 1B on a Cortex-A72 (ExecuTorch / XNNPACK)

| Source | Key point for us |
|---|---|
| XNNPACK source tree (KleidiAI kernels) | KleidiAI 4-bit kernels need dotprod/i8mm/SME — none on A72; 8da4w uses plain-NEON qd8-f32-qb4w kernels |
| ExecuTorch Llama README; Meta, *Introducing quantized Llama models*, Oct 2024 | SpinQuant / QAT-LoRA 1B use the same 8da4w scheme: quality, not speed |
| ExecuTorch `output_prune_map` (examples/models/llama/source_transformation/prune_vocab.py); Fedorov et al., *Llama Guard 3-1B-INT4*, 2024, arxiv.org/abs/2411.17713 | output-head pruning to a task vocabulary |
| Saxena, *Prompt Lookup Decoding*, 2023 (github.com/apoorvumang/prompt-lookup-decoding); Fu, Bailis, Stoica, Zhang, *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*, ICML 2024, arxiv.org/abs/2402.02057 | draft-free speculative decoding; gains measured on GPUs |
| LMSYS, *Fast JSON decoding / compressed FSM (jump-forward)*, 2024 | don't decode text the program already knows |
| Yoon et al., *LiteRT-Optimized INT8 LLM for Raspberry Pi4 Deployment*, ICCVW 2025 (supplemental, Table 3) | llama.cpp INT8 Llama-3.2-1B 2.72 tok/s on Pi 4 (6.86 on Pi 5) → switching runtime isn't a clear win (our reading) |

Tried: **code-written log line + slimmer prompt (adopted: −22% reply time)**, **class-notes KV cache (adopted: TTFT
11 → 6.5 s on same-class repeats)**, output-vocab pruning (decode −12%, end-to-end within noise → optional).
Not pursued: prompt-lookup decoding (A72 prefill is only ~3x decode speed, so verification is expensive).
