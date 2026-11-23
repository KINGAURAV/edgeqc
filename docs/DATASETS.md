# Datasets

Everything goes under `datasets/` (git-ignored).

On the dev Mac, `datasets/mvtec/` holds only a **20-image-per-category sample**; do not train or evaluate on it.
The full MVTec copy used for the Pi-size models and all test-set numbers is `datasets/mvtec_full/` (5 categories,
copied from the HPC). `samples/` holds the small image sets the profiles use as demo sources.

| Dataset | Use in EdgeQC | Get it |
|---|---|---|
| **MVTec AD** | Stage A (PatchCore) training on `train/good`, and image/pixel AUROC on `test` + `ground_truth` | <https://www.mvtec.com/company/research/datasets/mvtec-ad>. Accept the CC BY-NC-SA 4.0 licence, then download per category. Unpack to `datasets/mvtec/<category>/` |
| **NEU-DET** (NEU surface defect, steel, 6 classes, boxes) | Stage B (YOLOv8n) training and mAP | Kaggle: `kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database -p datasets/ --unzip`, then `python scripts/prepare_neu_det.py --src datasets/NEU-DET --out datasets/neu_yolo` |
| KolektorSDD2 (optional) | Defect-free *and* defective industrial surfaces. Useful to train Stage A for a surface profile, since NEU-DET has no defect-free images | <https://www.vicos.si/resources/kolektorsdd2/> |

## How the hybrid maps onto datasets
- **MVTec profiles** (`configs/profiles/mvtec_<category>.yaml` for bottle, metal_nut, tile, grid, screw; `mvtec.yaml` is the untuned FP32 starting template): PatchCore trained on one category's good images. With no classifier, unexplained regions become `anomaly_linear/spot/area` and go to REVIEW/REJECT based on severity.
- **Steel profile** (`configs/profiles/steel.yaml`): YOLOv8n (NEU-DET) names the defects. Stage A is the classical detector (no training); it catches defect types that are not in NEU's 6 classes. PatchCore for steel needs defect-free steel images (KolektorSDD2 negatives or your own captures) and has **not been trained**. The same gap means the false-reject rate on good steel is unmeasured: NEU-DET has no defect-free images.

Recommended MVTec categories for the demo: `bottle`, `metal_nut`, `screw`, `grid`, `tile`. They are industrial-looking and quick to train.
