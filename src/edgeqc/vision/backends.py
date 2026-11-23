"""Inference backends for the VISION pipeline.

- OnnxBackend       : ONNX Runtime on the Arm CPU (Raspberry Pi, Apple Silicon, any aarch64 Linux).
- ExecuTorchBackend : ExecuTorch .pte with the XNNPACK delegate (Arm NEON kernels), same runtime as the Llama advisor.
                      Built by scripts/export_executorch_vision.py. Needs the `executorch` pip package (runtime only).
- HailoBackend      : HailoRT on the Hailo-8/8L/10H NPU (Raspberry Pi 5 AI HAT+ / AI HAT+ 2).

The backend is picked from the model file extension (.onnx / .pte / .hef).
NOTE: Llama does not run here; it runs through ExecuTorch's LLM runner (edgeqc/advisor).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np


class InferenceBackend(Protocol):
    #: True if the model expects raw uint8 NHWC input (normalisation baked into the model, typical for HEF);
    #: False if it expects float32 NCHW ImageNet-normalised input (typical for ONNX exports).
    raw_uint8_nhwc: bool

    def run(self, x: np.ndarray) -> list[np.ndarray]: ...


class OnnxBackend:
    raw_uint8_nhwc = False

    def __init__(self, model_path: str | Path, num_threads: int = 4, spinning: bool = False,
                 boost_threads: int = 0):
        """boost_threads > num_threads: also build a session with that many threads, used while `boost` is
        True (the app sets it while the Llama advisor is idle, so vision can borrow the LLM's cores)."""
        import onnxruntime as ort

        def session(threads: int):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            # ORT worker threads busy-wait between ops by default. With the Llama runner sharing the same 4
            # cores that spinning steals its CPU time, so it is off unless asked for (vision.ort_spinning).
            opts.add_session_config_entry("session.intra_op.allow_spinning", "1" if spinning else "0")
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

        self.session = session(num_threads)
        self.boost_session = None
        if boost_threads > num_threads:
            # The vision thread may be pinned to one core (vision.cpu_affinity); ORT's pool threads inherit the
            # creating thread's mask, so widen it while the boost session spawns its threads.
            pinned = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
            if pinned is not None:
                os.sched_setaffinity(0, range(os.cpu_count() or 1))
            try:
                self.boost_session = session(boost_threads)
            finally:
                if pinned is not None:
                    os.sched_setaffinity(0, pinned)
        self.boost = False
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape   # static exports: e.g. [1, 3, 256, 256]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        s = self.boost_session if (self.boost and self.boost_session is not None) else self.session
        return s.run(None, {self.input_name: x})


class ExecuTorchBackend:
    """ExecuTorch program (.pte, XNNPACK-delegated). A sidecar <model>.json from the exporter carries the
    output names (e.g. ["dist"] for the fused PatchCore model), which the detectors use to pick their path."""

    raw_uint8_nhwc = False

    def __init__(self, model_path: str | Path, num_threads: int = 4):
        import json

        import torch
        from executorch.runtime import Runtime

        try:  # XNNPACK threadpool size (the pybindings default to all cores)
            from executorch.extension.pybindings import portable_lib

            portable_lib._unsafe_reset_threadpool(num_threads)
        except Exception:  # noqa: BLE001 — older wheels: keep the default
            pass
        self._torch = torch
        self.method = Runtime.get().load_program(Path(model_path)).load_method("forward")
        side = Path(model_path).with_suffix(".json")
        self.output_names = json.loads(side.read_text()).get("output_names", ["output"]) if side.exists() else ["output"]

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        outs = self.method.execute([self._torch.from_numpy(np.ascontiguousarray(x))])
        return [o.numpy() for o in outs]


class HailoBackend:
    """HailoRT InferModel API (HailoRT >= 4.18). UNTESTED until we have an AI HAT+.

    Compile models to .hef with the Hailo Dataflow Compiler (on an x86 host) with the
    ImageNet normalisation folded into the model, so the input is uint8 NHWC.
    Outputs are returned in NHWC; callers that expect NCHW feature maps must transpose
    (see PatchCoreDetector).
    """

    raw_uint8_nhwc = True

    def __init__(self, hef_path: str | Path, timeout_ms: int = 1000):
        from hailo_platform import VDevice  # type: ignore

        self.timeout_ms = timeout_ms
        self.vdevice = VDevice()
        self.infer_model = self.vdevice.create_infer_model(str(hef_path))
        self.configured = self.infer_model.configure()
        self.output_names = [o.name for o in self.infer_model.outputs]

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        bindings = self.configured.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(x[0] if x.ndim == 4 else x))
        outs = {}
        for name in self.output_names:
            shape = self.infer_model.output(name).shape
            outs[name] = np.empty(shape, dtype=np.float32)
            bindings.output(name).set_buffer(outs[name])
        self.configured.run([bindings], self.timeout_ms)
        return [outs[n][None] for n in self.output_names]


def make_backend(kind: str, model_path: str | Path, num_threads: int = 4, spinning: bool = False,
                 boost_threads: int = 0) -> InferenceBackend:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if kind == "hailo" or model_path.suffix == ".hef":
        return HailoBackend(model_path)
    if kind == "executorch" or model_path.suffix == ".pte":
        return ExecuTorchBackend(model_path, num_threads=num_threads)
    return OnnxBackend(model_path, num_threads=num_threads, spinning=spinning, boost_threads=boost_threads)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(bgr: np.ndarray, size: int, raw_uint8_nhwc: bool) -> np.ndarray:
    """Resize (square) and normalise a BGR frame for the given backend layout."""
    import cv2

    rgb = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    if raw_uint8_nhwc:
        return rgb[None]
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])
