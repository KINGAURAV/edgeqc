import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def cfg(tmp_path):
    from edgeqc.config import load_config

    return load_config(overrides=[f"app.data_dir={tmp_path / 'runs'}", "advisor.engine=rule"])


@pytest.fixture
def surfaces():
    """(good, scratched, spotted) synthetic BGR images, generated the same way as the demo samples."""
    from make_synthetic_samples import add_scratch, add_spot, texture

    rng = np.random.default_rng(42)
    to_bgr = lambda f: cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # noqa: E731
    good = texture(rng, 320)
    scratched = texture(rng, 320)
    add_scratch(scratched, rng)
    spotted = texture(rng, 320)
    add_spot(spotted, rng)
    return to_bgr(good), to_bgr(scratched), to_bgr(spotted)
