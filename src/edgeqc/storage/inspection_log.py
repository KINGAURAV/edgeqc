"""SQLite inspection log with annotated image snapshots and CSV export."""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..advisor.prompts import split_log_entry
from ..types import InspectionResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    frame_id INTEGER,
    product TEXT,
    verdict TEXT NOT NULL,
    image_score REAL,
    worst_severity TEXT,
    n_defects INTEGER,
    defects_json TEXT,
    latency_json TEXT,
    image_path TEXT,
    advice TEXT,
    log_entry TEXT,
    advice_engine TEXT,
    advice_latency_s REAL,
    acknowledged_by TEXT,
    acknowledged_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_ts ON inspections(ts);
"""


class InspectionLog:
    def __init__(self, data_dir: str | Path, save_images: str = "non_pass"):
        self.dir = Path(data_dir)
        (self.dir / "images").mkdir(parents=True, exist_ok=True)
        self.save_images = save_images
        self.db = sqlite3.connect(self.dir / "inspections.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        # WAL + NORMAL: a commit is an append to the WAL, not a full fsync of the DB (SD-card friendly, ~2 FPS
        # of inserts). A power cut can lose the last few rows but never corrupts the DB.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(inspections)")}
        if "log_entry" not in cols:  # migrate DBs created before log_entry existed
            self.db.execute("ALTER TABLE inspections ADD COLUMN log_entry TEXT")
        self.lock = threading.Lock()
        # Verdict counts kept in memory: the dashboard polls stats() every second, and a GROUP BY over a
        # table that grows by ~170k rows/day at 2 FPS would hold the lock the inspection loop needs.
        self._counts = {r[0]: r[1] for r in self.db.execute("SELECT verdict, COUNT(*) FROM inspections GROUP BY verdict")}

    def image_path_for(self, result: InspectionResult) -> Optional[str]:
        """Where this result's annotated image goes under the save_images policy (None = not saved)."""
        if not (self.save_images == "all" or (self.save_images == "non_pass" and result.verdict.value != "PASS")):
            return None
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(result.timestamp))
        return str(self.dir / "images" / f"{name}_{result.frame_id:06d}_{result.verdict.value}.jpg")

    def log(self, result: InspectionResult, annotated: Optional[np.ndarray] = None,
            image_path: Optional[str] = None) -> int:
        """Insert a row. annotated: write it now under the save policy. image_path: record a path that the
        caller writes itself (the app's output thread writes the already-encoded dashboard JPEG there)."""
        if annotated is not None and image_path is None:
            image_path = self.image_path_for(result)
            if image_path:
                cv2.imwrite(image_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        worst = result.worst_severity
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO inspections (ts, frame_id, product, verdict, image_score, worst_severity, n_defects,"
                " defects_json, latency_json, image_path) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (result.timestamp, result.frame_id, result.product, result.verdict.value, result.image_score,
                 worst.value if worst else None, len(result.defects),
                 json.dumps([d.to_dict() for d in result.defects]), json.dumps(result.latency_ms), image_path))
            self.db.commit()
            result.inspection_id = cur.lastrowid
            self._counts[result.verdict.value] = self._counts.get(result.verdict.value, 0) + 1
        return result.inspection_id

    def set_advice(self, inspection_id: int, text: str, engine: str, latency_s: float) -> None:
        """Store advisor output; its 'Log:' line (written by code, prompts.log_line) goes to the log_entry column."""
        advice, log_entry = split_log_entry(text)
        with self.lock:
            self.db.execute("UPDATE inspections SET advice=?, log_entry=?, advice_engine=?, advice_latency_s=?"
                            " WHERE id=?", (advice, log_entry, engine, latency_s, inspection_id))
            self.db.commit()

    def acknowledge(self, inspection_id: int, operator: str = "operator") -> bool:
        with self.lock:
            cur = self.db.execute("UPDATE inspections SET acknowledged_by=?, acknowledged_ts=? WHERE id=?",
                                  (operator, time.time(), inspection_id))
            self.db.commit()
            return cur.rowcount > 0

    def recent(self, n: int = 50, only_defects: bool = False) -> list[dict]:
        q = "SELECT * FROM inspections"
        if only_defects:
            q += " WHERE verdict != 'PASS'"
        q += " ORDER BY id DESC LIMIT ?"
        with self.lock:
            rows = self.db.execute(q, (n,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["defects"] = json.loads(d.pop("defects_json") or "[]")
            d["latency_ms"] = json.loads(d.pop("latency_json") or "{}")
            out.append(d)
        return out

    def stats(self) -> dict:
        with self.lock:
            counts = dict(self._counts)
        total = sum(counts.values())
        return {"total": total, **{k: counts.get(k, 0) for k in ("PASS", "REVIEW", "REJECT")},
                "yield": (counts.get("PASS", 0) / total) if total else None}

    def export_csv(self, path: str | Path) -> Path:
        path = Path(path)
        with self.lock:
            rows = self.db.execute("SELECT * FROM inspections ORDER BY id").fetchall()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            if rows:
                w.writerow(rows[0].keys())
                w.writerows([tuple(r) for r in rows])
        return path

    def close(self) -> None:
        self.db.close()
