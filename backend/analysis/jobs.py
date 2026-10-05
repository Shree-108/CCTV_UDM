"""Background runner for video analyses (one at a time: the tracker is stateful and the
CPU/GPU is the bottleneck). Images run synchronously in the request."""
from __future__ import annotations

import queue
import threading
import traceback

from backend.analysis.pipeline import LiveView, Models, analyze_video
from backend.core.settings import Settings
from backend.database.db import DB
from backend.detection.grid import AnalysisArea


class JobRunner:
    def __init__(self, db: DB, settings: Settings, models: Models):
        self.db, self.settings, self.models = db, settings, models
        self.q: queue.Queue = queue.Queue()
        self.live: dict[str, LiveView] = {}
        self._cancel: set[str] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="video-worker", daemon=True)
        self._thread.start()

    def submit(self, aid: str, media: dict, area: AnalysisArea, s: dict) -> None:
        self.live[aid] = LiveView()
        self.q.put((aid, media, area, s))

    def cancel(self, aid: str) -> None:
        self._cancel.add(aid)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                aid, media, area, s = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if aid in self._cancel:
                self.db.update_analysis(aid, state="failed", error="Cancelled before start")
                continue
            self.db.update_analysis(aid, state="running")
            try:
                models = self.models.get(s)
                analyze_video(self.db, aid, media, area, s, models, self.live[aid],
                              should_stop=lambda: aid in self._cancel or self._stop.is_set())
            except Exception as e:  # noqa: BLE001 - reported to the UI
                traceback.print_exc()
                self.db.update_analysis(aid, state="failed", error=str(e))
            finally:
                self._cancel.discard(aid)
                lv = self.live.get(aid)
                if lv:
                    lv.jpeg = b""       # free memory; the annotated video is on disk now

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
