"""Operator dashboard: FastAPI with an MJPEG live stream, JSON state and acknowledge endpoints.

All assets are served locally (no CDN), so the dashboard works on an offline factory LAN.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from ..storage import InspectionLog

STATIC = Path(__file__).parent / "static"


class SharedState:
    """Thread-safe handoff between the inspection loop and HTTP handlers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg: Optional[bytes] = None
        self.latest: Optional[dict] = None
        self.latest_alert: Optional[dict] = None
        self.fps: float = 0.0
        self.engines: dict = {}

    def update_frame(self, frame: np.ndarray, result: dict, fps: float) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        self.update_jpeg(buf.tobytes() if ok else None, result, fps)

    def update_jpeg(self, jpeg: Optional[bytes], result: dict, fps: float) -> None:
        """Same as update_frame with an already-encoded frame (the app encodes once for UI + saved image)."""
        with self.lock:
            if jpeg is not None:
                self.jpeg = jpeg
            self.latest, self.fps = result, fps
            if result["verdict"] != "PASS":
                self.latest_alert = result

    def set_advice(self, inspection_id: int, text: str, engine: str, latency_s: float,
                   partial: bool = False) -> None:
        """partial=True: reply still being generated (streamed); the final call replaces it."""
        with self.lock:
            if self.latest_alert and self.latest_alert.get("inspection_id") == inspection_id:
                self.latest_alert = {**self.latest_alert, "advice": text, "advice_engine": engine,
                                     "advice_latency_s": round(latency_s, 2), "advice_partial": partial}


def create_app(state: SharedState, store: InspectionLog) -> FastAPI:
    app = FastAPI(title="EdgeQC")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/stream")
    def stream():
        def gen():
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            last = None
            while True:
                with state.lock:
                    jpeg = state.jpeg
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield boundary + jpeg + b"\r\n"
                time.sleep(0.03)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/state")
    def api_state():
        with state.lock:
            return {"latest": state.latest, "alert": state.latest_alert, "fps": round(state.fps, 1),
                    "engines": state.engines, "stats": store.stats()}

    @app.get("/api/recent")
    def api_recent(n: int = 25, only_defects: bool = True):
        return store.recent(n, only_defects)

    @app.post("/api/ack/{inspection_id}")
    def api_ack(inspection_id: int, operator: str = "operator"):
        if not store.acknowledge(inspection_id, operator):
            raise HTTPException(404, "inspection not found")
        return {"ok": True}

    return app
