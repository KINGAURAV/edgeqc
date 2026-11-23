"""EdgeQC main loop: capture → vision pipeline → log → async Llama advice → dashboard.

Examples:
    python -m edgeqc.app                                   # synthetic samples, web UI on :8000
    python -m edgeqc.app --source 0                        # USB camera
    python -m edgeqc.app --source datasets/mvtec/screw/test --set vision.anomaly.method=patchcore
    python -m edgeqc.app --headless --max-frames 200       # no UI, benchmark-style run
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import threading
import time

import cv2

from .advisor import AdvisorService
from .advisor.service import pregeneration_candidates
from .capture import open_source
from .config import load_config, resolve_path
from .storage import InspectionLog
from .types import InspectionResult, Verdict
from .vision import InspectionPipeline
from .vision.overlay import render

log = logging.getLogger("edgeqc")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EdgeQC on-device QC assistant")
    p.add_argument("--config", help="YAML config overriding configs/default.yaml")
    p.add_argument("--source", help="camera index, 'picamera', video file or image folder")
    p.add_argument("--device", help="device tuning on top of --config, e.g. configs/devices/pi4.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config key, e.g. --set vision.backend=hailo")
    p.add_argument("--headless", action="store_true", help="disable web UI and window")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = run forever)")
    p.add_argument("--export-csv", help="write the inspection log to this CSV on exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _pin_current_thread(cpus) -> None:
    if cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)


class OutputWorker:
    """Draws the overlay, JPEG-encodes it ONCE and hands the bytes to the saved-image file and the dashboard.

    Runs on its own thread, pinned off the vision core: on a Pi 4 this work is ~100 ms per 1024 px MVTec frame
    (overlay 52, encode for UI 14, encode+write for the log 15), which used to sit on the vision loop.
    Frames are drawn at ui.display_max_side (default 640 px). A non-PASS frame (saved as evidence) waits for a
    queue slot; a PASS frame is dropped if the worker is behind, which only skips a dashboard refresh."""

    def __init__(self, state, max_side: int, quality: int, cpus=None):
        self.state, self.max_side, self.quality, self.cpus = state, max_side, quality, cpus
        self.q: queue.Queue = queue.Queue(maxsize=4)
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="output", daemon=True)
        self._thread.start()

    def submit(self, frame, result: InspectionResult, heat, fps: float, image_path, ui_dict) -> None:
        item = (frame, result, heat, fps, image_path, ui_dict)
        try:
            if image_path:
                self.q.put(item, timeout=5)
            else:
                self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        _pin_current_thread(self.cpus)
        while True:
            item = self.q.get()
            if item is None:
                return
            frame, result, heat, fps, image_path, ui_dict = item
            try:
                annotated = render(frame, result, heat, fps, self.max_side)
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                jpeg = buf.tobytes() if ok else None
                if image_path and jpeg:
                    with open(image_path, "wb") as f:
                        f.write(jpeg)
                if self.state is not None:
                    self.state.update_jpeg(jpeg, ui_dict, fps)
            except Exception:  # noqa: BLE001 — display/logging must never stop the line
                log.exception("output worker failed on frame %s", result.frame_id)

    def close(self) -> None:
        self.q.put(None)
        self._thread.join(timeout=5)


def run(cfg: dict, headless: bool = False, max_frames: int = 0, export_csv: str | None = None) -> dict:
    app_cfg, ui_cfg = cfg["app"], cfg["ui"]
    vcpus = cfg["vision"].get("cpu_affinity")
    all_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    other_cpus = None  # cores for the web/output threads: everything except the vision cores
    if vcpus and all_cpus:
        other_cpus = [c for c in all_cpus if c not in vcpus] or None
    store = InspectionLog(resolve_path(app_cfg["data_dir"]), app_cfg["save_images"])

    state = None
    if not headless and ui_cfg["web"]:
        import uvicorn

        from .ui.server import SharedState, create_app

        state = SharedState()
        server = uvicorn.Server(uvicorn.Config(create_app(state, store), host=ui_cfg["host"],
                                               port=int(ui_cfg["port"]), log_level="warning"))

        def serve():
            _pin_current_thread(other_cpus)  # HTTP/MJPEG work stays off the vision core
            server.run()
        threading.Thread(target=serve, name="web", daemon=True).start()
        log.info("Dashboard: http://localhost:%s", ui_cfg["port"])

    # Camera grabber thread starts before the vision thread pins itself, so it isn't confined to the vision core
    source = open_source(app_cfg["source"], loop=app_cfg["loop_source"], camera_size=app_cfg.get("camera_size"))
    # Optional core split (Linux): pin the vision loop to its own cores so it doesn't stall the LLM's threads
    # (advisor.cpu_affinity). Set before the pipeline starts so ONNX Runtime's threads inherit it; the
    # vision.boost_threads sessions are created with all cores (see OnnxBackend).
    if vcpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, vcpus)
        log.info("Vision pinned to cores %s", vcpus)
    pipeline = InspectionPipeline(cfg)

    def on_advice(result: InspectionResult, text: str, engine: str, latency_s: float) -> None:
        store.set_advice(result.inspection_id, text, engine, latency_s)
        if state:
            state.set_advice(result.inspection_id, text, engine, latency_s)
        log.info("[advice #%s via %s, %.1fs] %s", result.inspection_id, engine, latency_s, text.replace("\n", " | "))

    t_submit: dict[int, float] = {}  # inspection_id → submit time, for the live "generating…" latency

    def on_partial(result: InspectionResult, text: str, engine: str) -> None:
        if state:
            state.set_advice(result.inspection_id, text, engine,
                             time.monotonic() - t_submit.get(result.inspection_id, time.monotonic()), partial=True)

    advisor = AdvisorService(cfg, on_advice, on_partial=on_partial if state else None)
    if cfg["advisor"].get("pregenerate", True):
        advisor.pregenerate(pregeneration_candidates(cfg, cfg["vision"]["classifier"]["labels"]
                                                     if pipeline.classifier else []))
    boost = pipeline.can_boost
    if state:
        state.engines = {"vision": f"{type(pipeline.anomaly).__name__}"
                                   f"{' + YOLO' if pipeline.classifier else ''} / {cfg['vision']['backend']}"
                                   f"{' (+boost)' if boost else ''}",
                         "advisor": advisor.engine_name}

    window = bool(ui_cfg["window"] and not headless)
    need_output = state is not None or app_cfg["save_images"] != "none"
    output = OutputWorker(state, int(ui_cfg.get("display_max_side", 640)), int(ui_cfg.get("jpeg_quality", 80)),
                          other_cpus) if (need_output and not window) else None
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    period = 1.0 / float(app_cfg["target_fps"]) if app_cfg["target_fps"] else 0.0
    idle_fps = app_cfg.get("idle_target_fps")
    idle_period = (1.0 / float(idle_fps) if idle_fps else 0.0) if idle_fps is not None else period
    fps, n, boosted_frames = 0.0, 0, 0
    try:
        while not stop.is_set():
            t_start = time.perf_counter()
            boosted = boost and not advisor.busy  # borrow the LLM's cores only while it is idle
            if boost:
                pipeline.set_boost(boosted)
                boosted_frames += boosted
            frame = source.read()
            if frame is None:
                break
            result = pipeline.inspect(frame)
            heat = pipeline.last_anomaly.heatmap if (ui_cfg["show_heatmap"] and pipeline.last_anomaly) else None
            if window:  # desktop preview: draw here (imshow must run on the main thread)
                annotated = render(frame, result, heat, fps)
                store.log(result, annotated)
            else:
                image_path = store.image_path_for(result) if output else None
                store.log(result, image_path=image_path)
            cached = None
            if result.verdict != Verdict.PASS:
                t_submit[result.inspection_id] = time.monotonic()
                if len(t_submit) > 256:
                    t_submit.pop(next(iter(t_submit)))
                if not advisor.submit(result):
                    # Same defect signature as earlier advice: reuse it now (with this frame's log line). If it
                    # is still being generated, on_advice delivers it for this inspection when ready.
                    cached = advisor.cached(result)
                    if cached:
                        result.advice = cached[0]
                        store.set_advice(result.inspection_id, cached[0], f"{cached[1]} (repeat)", 0.0)
            d = None
            if state:
                d = result.to_dict()
                if cached:
                    d.update(advice_engine=f"{cached[1]} (repeat)", advice_latency_s=0.0)
            if window:
                if state:
                    state.update_frame(annotated, d, fps)
                cv2.imshow("EdgeQC", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif output:
                output.submit(frame, result, heat, fps, image_path, d)
            n += 1
            if max_frames and n >= max_frames:
                break
            dt = time.perf_counter() - t_start
            target = idle_period if boosted else period
            if target > dt:
                time.sleep(target - dt)
            inst = 1.0 / max(time.perf_counter() - t_start, 1e-6)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
    finally:
        source.close()
        if output:
            output.close()
        time.sleep(0.2)
        advisor.close()
        stats = store.stats()
        stats["advisor"] = dict(advisor.stats)
        if boost:
            stats["boosted_frames"] = boosted_frames
        if output and output.dropped:
            stats["ui_frames_dropped"] = output.dropped
        if export_csv:
            log.info("Exported log to %s", store.export_csv(export_csv))
        store.close()
        if window:
            cv2.destroyAllWindows()
    log.info("Done: %s", stats)
    return stats


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    overrides = list(args.set)
    if args.source is not None:
        overrides.append(f"app.source={args.source}")
    cfg = load_config(args.config, overrides, device=args.device)
    run(cfg, headless=args.headless, max_frames=args.max_frames, export_csv=args.export_csv)


if __name__ == "__main__":
    main()
