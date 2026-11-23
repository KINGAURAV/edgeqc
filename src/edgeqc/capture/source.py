"""Frame sources: USB/CSI camera (OpenCV), Raspberry Pi camera (Picamera2), video file, image folder.

An image folder acts as a "virtual conveyor" for demos and benchmarks.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator, Optional, Protocol

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class FrameSource(Protocol):
    def read(self) -> Optional[np.ndarray]: ...
    def close(self) -> None: ...


class OpenCVSource:
    def __init__(self, spec: int | str, loop: bool = False):
        self.spec, self.loop = spec, loop
        self.cap = cv2.VideoCapture(spec)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {spec!r}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok and self.loop and isinstance(self.spec, str):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()


class LatestFrameCamera:
    """Live OpenCV camera that always returns the NEWEST frame.

    A USB camera delivers 30 FPS into a driver queue; an inspection loop running at 2-5 FPS that calls
    cap.read() gets frames that are seconds old. A background thread keeps calling grab() (dequeue only, no
    decode), and read() decodes just the latest grabbed frame. read() waits for a frame newer than the last one
    it returned, so the same frame is never inspected twice."""

    def __init__(self, index: int, size: tuple[int, int] | None = None, timeout_s: float = 2.0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if size:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.timeout_s = timeout_s
        self._lock, self._new = threading.Lock(), threading.Condition()
        self._seq, self._read_seq, self._ok = 0, 0, True
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._grab_loop, name="camera-grab", daemon=True)
        self._thread.start()

    def _grab_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                ok = self.cap.grab()
            with self._new:
                self._ok = ok
                if ok:
                    self._seq += 1
                self._new.notify_all()
            if not ok:
                self._stop.wait(0.05)

    def read(self) -> Optional[np.ndarray]:
        with self._new:
            if not self._new.wait_for(lambda: self._seq > self._read_seq or self._stop.is_set(), self.timeout_s):
                return None
            self._read_seq = self._seq
        with self._lock:
            ok, frame = self.cap.retrieve()
        return frame if ok else None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.cap.release()


class PiCameraSource:
    """Raspberry Pi CSI camera via Picamera2 (only importable on Raspberry Pi OS)."""

    def __init__(self, size: tuple[int, int] = (1280, 960)):
        from picamera2 import Picamera2  # type: ignore

        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(main={"size": size, "format": "RGB888"}))
        self.cam.start()

    def read(self) -> Optional[np.ndarray]:
        # Picamera2's "RGB888" format is BGR byte order in memory, which is what OpenCV expects.
        return self.cam.capture_array()

    def close(self) -> None:
        self.cam.stop()


class ImageFolderSource:
    def __init__(self, folder: str | Path, loop: bool = True):
        self.paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if not self.paths:
            raise RuntimeError(f"No images found in {folder}")
        self.loop = loop
        self._it = self._iter()
        self.current_path: Optional[Path] = None

    def _iter(self) -> Iterator[Path]:
        while True:
            yield from self.paths
            if not self.loop:
                return

    def read(self) -> Optional[np.ndarray]:
        for p in self._it:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                self.current_path = p
                return img
        return None

    def close(self) -> None:
        pass


def open_source(spec: int | str, loop: bool = True, camera_size: tuple[int, int] | None = None) -> FrameSource:
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return LatestFrameCamera(int(spec), tuple(camera_size) if camera_size else None)
    if spec == "picamera":
        # The Pi ISP scales in hardware, so asking for a small stream makes the CPU resize nearly free
        return PiCameraSource(tuple(camera_size) if camera_size else (640, 480))
    from ..config import resolve_path

    path = resolve_path(spec)
    if path.is_dir():
        return ImageFolderSource(path, loop=loop)
    if path.is_file():
        return OpenCVSource(str(path), loop=loop)
    raise FileNotFoundError(f"Source not found: {spec} (resolved to {path})")
