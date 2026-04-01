from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue
import os
import time
import cv2
import numpy as np

from config import ALERT_COOLDOWN_SECONDS, SNAPSHOT_DIR
from utils.logger import setup_logging

log = setup_logging()


@dataclass
class AlertEvent:
    camera_id: int
    camera_label: str
    track_ids: list
    frame: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)


class AlertManager:
    def __init__(self):
        self._queue: queue.Queue[AlertEvent] = queue.Queue()
        self._last_alert: dict[int, float] = {}

        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        threading.Thread(
            target=self._worker,
            daemon=True,
            name="AlertWorker"
        ).start()

        log.info("AlertManager ready (email=OFF)")

    def trigger(self, cam_id: int, label: str, track_ids: list, frame: np.ndarray):
        now = time.time()

        if now - self._last_alert.get(cam_id, 0) < ALERT_COOLDOWN_SECONDS:
            return

        self._last_alert[cam_id] = now

        self._queue.put(
            AlertEvent(cam_id, label, list(track_ids), frame.copy())
        )

    def _worker(self):
        while True:
            try:
                ev = self._queue.get(timeout=1)
                self._handle_event(ev)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                log.error("AlertManager error: %s", exc, exc_info=True)

    def _handle_event(self, ev: AlertEvent):
        ts = ev.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        ids = ", ".join(str(t) for t in ev.track_ids)

        log.warning(
            "INTRUSION | %-20s (cam %d) | tracks [%s] | %s",
            ev.camera_label, ev.camera_id, ids, ts
        )

        self._save_snapshot(ev)

    def _save_snapshot(self, ev: AlertEvent) -> str:
        ts = ev.timestamp.strftime("%Y%m%d_%H%M%S")
        path = f"{SNAPSHOT_DIR}/cam{ev.camera_id}_{ts}.jpg"

        cv2.imwrite(path, ev.frame)

        log.info("Snapshot saved -> %s", path)

        return path