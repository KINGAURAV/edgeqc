from edgeqc.config import load_config


def test_device_overrides_apply_only_for_named_device(tmp_path):
    prof = tmp_path / "p.yaml"
    prof.write_text("vision:\n  anomaly:\n    model_path: big.onnx\n"
                    "device_overrides:\n  pi4:\n    vision:\n      anomaly:\n        model_path: small.onnx\n")
    dev = tmp_path / "d.yaml"
    dev.write_text("device:\n  name: pi4\n")
    assert load_config(prof)["vision"]["anomaly"]["model_path"] == "big.onnx"
    cfg = load_config(prof, device=dev)
    assert cfg["vision"]["anomaly"]["model_path"] == "small.onnx"
    assert "device_overrides" not in cfg
