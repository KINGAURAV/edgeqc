import numpy as np

from edgeqc.types import Defect, Detection, Region, Severity, Verdict
from edgeqc.vision import InspectionPipeline
from edgeqc.vision.classifier import ShapeTyper, YoloDetector
from edgeqc.vision.fusion import fuse, iou
from edgeqc.vision.severity import SeverityEngine


def test_pipeline_good_vs_defective(cfg, surfaces):
    good, scratched, spotted = surfaces
    pipe = InspectionPipeline(cfg)
    r_good = pipe.inspect(good)
    assert r_good.verdict == Verdict.PASS, r_good.defects
    for img in (scratched, spotted):
        r = pipe.inspect(img)
        assert r.verdict != Verdict.PASS
        assert r.defects and all(d.source == "anomaly" for d in r.defects)
    assert set(r.latency_ms) >= {"anomaly", "classifier", "total"}


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 10, 10)) - 50 / 150) < 1e-9


def test_fusion_sources():
    regions = [Region((10, 10, 20, 20), 2.0, 0.01), Region((200, 200, 100, 4), 1.5, 0.004)]
    dets = [Detection("scratches", 0.9, (12, 12, 18, 18)), Detection("inclusion", 0.6, (400, 400, 10, 10))]
    out = fuse(regions, dets, (500, 500), ShapeTyper())
    by = {d.label: d for d in out}
    assert by["scratches"].source == "fused" and by["scratches"].anomaly_score == 2.0
    assert by["inclusion"].source == "classifier"
    assert by["anomaly_linear"].source == "anomaly"


def test_severity_rules(cfg):
    eng = SeverityEngine(cfg["severity"])
    minor = Defect("scratches", (0, 0, 5, 5), 0.9, 1.1, 0.001, "fused")
    big = Defect("scratches", (0, 0, 5, 5), 0.9, 1.1, 0.5, "fused")
    crit_cls = Defect("crack", (0, 0, 5, 5), 0.9, 0.0, 0.001, "classifier")
    assert eng.grade(minor) == Severity.MINOR
    assert eng.grade(big) == Severity.CRITICAL
    assert eng.grade(crit_cls) == Severity.CRITICAL
    assert eng.apply([]) == Verdict.PASS
    assert eng.apply([minor]) == Verdict.REVIEW
    assert eng.apply([minor, crit_cls]) == Verdict.REJECT


def test_yolo_postprocess_decodes_and_unletterboxes():
    class Dummy:
        raw_uint8_nhwc = False

    det = YoloDetector(Dummy(), ["a", "b"], input_size=320, conf_threshold=0.5)
    out = np.zeros((1, 6, 3), np.float32)
    out[0, :4, 0] = [160, 160, 40, 20]
    out[0, 5, 0] = 0.9  # class b
    out[0, :4, 1] = [100, 100, 10, 10]
    out[0, 4, 1] = 0.2  # below threshold
    # image 640x640 → letterbox ratio 0.5, no padding
    dets = det.postprocess(out, r=0.5, padx=0, pady=0, hw=(640, 640))
    assert len(dets) == 1 and dets[0].label == "b"
    assert dets[0].bbox == (280, 300, 80, 40)


def test_yolo_per_class_threshold():
    class Dummy:
        raw_uint8_nhwc = False

    out = np.zeros((1, 6, 2), np.float32)
    out[0, :4, 0] = [100, 100, 20, 20]
    out[0, 4, 0] = 0.2  # class a, low confidence
    out[0, :4, 1] = [200, 200, 20, 20]
    out[0, 5, 1] = 0.2  # class b, low confidence
    det = YoloDetector(Dummy(), ["a", "b"], input_size=320, conf_threshold=0.5, class_conf={"a": 0.1})
    dets = det.postprocess(out, r=1.0, padx=0, pady=0, hw=(320, 320))
    assert [d.label for d in dets] == ["a"]


def test_patchcore_image_gate_suppresses_pixel_noise(tmp_path):
    import json

    from edgeqc.vision.anomaly import PatchCoreDetector

    class Fused:
        raw_uint8_nhwc = False
        output_names = ["dist"]

        def __init__(self, peak):
            self.peak = peak

        def run(self, x):
            d = np.zeros((1, 32, 32), np.float32)
            d[0, 10:14, 10:14] = self.peak
            return [d]

    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"pixel_threshold": 1.0, "image_threshold": 10.0, "input_size": 64, "blur_sigma": 0.5}))
    img = np.zeros((64, 64, 3), np.uint8)
    quiet = PatchCoreDetector(Fused(5.0), "", meta, 64)          # pixels over threshold, image score 0.5
    assert quiet(img).regions == []
    assert PatchCoreDetector(Fused(5.0), "", meta, 64, image_gate=None)(img).regions
    assert PatchCoreDetector(Fused(12.0), "", meta, 64)(img).regions  # image score 1.2 → reported


def test_patchcore_smoothed_image_score_ignores_single_noisy_patch(tmp_path):
    import json

    from edgeqc.vision.anomaly import PatchCoreDetector

    class Fused:
        raw_uint8_nhwc = False
        output_names = ["dist"]

        def __init__(self, d):
            self.d = d

        def run(self, x):
            return [self.d[None]]

    spike = np.zeros((32, 32), np.float32)
    spike[5, 5] = 12.0                       # one hot patch: raw max 12, smoothed max small
    blob = np.zeros((32, 32), np.float32)
    blob[10:20, 10:20] = 6.0                 # a real 10x10 region: smoothed max ~6
    base = {"pixel_threshold": 1.0, "image_threshold": 10.0, "input_size": 64, "blur_sigma": 4.0}
    raw, smooth = tmp_path / "raw.json", tmp_path / "smooth.json"
    raw.write_text(json.dumps(base))
    smooth.write_text(json.dumps({**base, "image_threshold_smooth": 3.0}))
    img = np.zeros((64, 64, 3), np.uint8)
    assert PatchCoreDetector(Fused(spike), "", raw, 64)(img).score >= 1.0       # raw max: flagged by one patch
    assert PatchCoreDetector(Fused(spike), "", smooth, 64)(img).score < 1.0     # smoothed: not flagged
    out = PatchCoreDetector(Fused(blob), "", smooth, 64)(img)
    assert out.score >= 1.0 and out.regions
