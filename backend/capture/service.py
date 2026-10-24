"""Capture service: take one frame from a camera's source, keep the original, and analyse it
with the existing uniform pipeline. Plus the scheduler loop that does this for every enabled
camera at each time slot.

Guarantees
  * one capture at a time per camera (a busy camera is skipped, never queued twice)
  * a scheduled slot is captured at most once per camera, also across restarts
  * the original frame is saved before analysis; an analysis failure never loses it
  * every capture is recorded (done / capture_failed / analysis_failed) with its error
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone

import cv2

from backend.analysis.jobs import JobRunner
from backend.analysis.media import new_upload_path, probe
from backend.analysis.pipeline import Models
from backend.analysis.runner import AnalysisFailed, full_frame_area, start_analysis
from backend.capture.schedule import current_slot, next_slot, parse_hhmm, slots_per_day
from backend.capture.sources import CaptureError, dhash, grab, hamming
from backend.core.settings import Settings
from backend.database.db import DB, now_iso

log = logging.getLogger(__name__)
SIMILAR_BITS = 4      # dHash distance at or below which a frame counts as "similar to previous"
TICK_SECONDS = 5.0


class CaptureBusy(RuntimeError):
    pass


def _utc_iso(local_naive: datetime) -> str:
    return local_naive.astimezone(timezone.utc).isoformat(timespec="seconds")


class CaptureService:
    def __init__(self, db: DB, settings: Settings, models: Models, jobs: JobRunner):
        self.db, self.settings, self.models, self.jobs = db, settings, models, jobs
        self._inflight: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_tick: str | None = None

    # ------------------------------------------------------------------ one capture
    def capture_now(self, cam_id: str, trigger: str = "manual", slot_at: str | None = None) -> dict:
        cam = self.db.camera(cam_id)
        if cam is None:
            raise KeyError(cam_id)
        with self._lock:
            if cam_id in self._inflight:
                raise CaptureBusy(f"{cam_id} is already capturing")
            self._inflight.add(cam_id)
        try:
            return self._capture(cam, trigger, slot_at)
        finally:
            with self._lock:
                self._inflight.discard(cam_id)

    def _capture(self, cam: dict, trigger: str, slot_at: str | None) -> dict:
        t0 = time.time()
        local = datetime.now()
        cap_id = f"cap-{local:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.db.add_capture({"id": cap_id, "camera_id": cam["id"], "store_id": cam["store_id"],
                             "captured_at": now_iso(), "slot_at": slot_at, "trigger": trigger, "status": "running"})
        cfg = self.settings.get()
        try:
            frame, ref, cursor = grab(cam, float(cfg["video_step_seconds"]))
        except CaptureError as e:
            if e.next_cursor is not None:
                self.db.set_camera_cursor(cam["id"], e.next_cursor)
            log.warning("capture %s on %s failed: %s", cap_id, cam["id"], e)
            self.db.update_capture(cap_id, status="capture_failed", error=str(e), duration_ms=int((time.time() - t0) * 1000))
            return self.db.capture(cap_id)
        self.db.set_camera_cursor(cam["id"], cursor)

        # keep the original frame before anything else can fail
        h = dhash(frame)
        prev = self.db.last_capture(cam["id"])
        similar = int(prev is not None and hamming(prev["frame_hash"], h) <= SIMILAR_BITS)
        filename = f"{cam['id']}_{local:%Y%m%d_%H%M%S}.jpg"
        path = new_upload_path(filename)
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        media = probe(path, filename, "image", path.stat().st_size)
        self.db.add_media(media)
        self.db.update_capture(cap_id, source_ref=ref, frame_hash=h, similar_to_previous=similar, media_id=media["id"])

        area = cam.get("area") or full_frame_area(cam["name"])
        extra = {"source": "capture", "store_id": cam["store_id"], "camera_id": cam["id"], "capture_id": cap_id,
                 "captured_at": self.db.capture(cap_id)["captured_at"]}
        try:
            aid = start_analysis(self.db, self.settings, self.models, self.jobs, media, area, extra)
            fields = {"status": "done", "analysis_id": aid}
        except AnalysisFailed as e:
            fields = {"status": "analysis_failed", "analysis_id": e.aid, "error": str(e)}
        except ValueError as e:                      # invalid camera area
            fields = {"status": "analysis_failed", "error": f"Invalid camera area: {e}"}
        if fields["status"] != "done":
            log.warning("capture %s on %s: %s", cap_id, cam["id"], fields.get("error"))
        self.db.update_capture(cap_id, duration_ms=int((time.time() - t0) * 1000), **fields)
        return self.db.capture(cap_id)

    # ------------------------------------------------------------------ schedule
    def _window(self) -> tuple[int, object, object]:
        cfg = self.settings.get()
        return int(cfg["capture_interval_minutes"]), parse_hhmm(cfg["capture_active_start"]), parse_hhmm(cfg["capture_active_end"])

    def tick(self, now: datetime | None = None) -> list[dict]:
        """Capture the current slot for every enabled camera that does not have it yet.
        `now` is naive local time (injectable for tests). Runs captures one after another."""
        now = now or datetime.now()
        self.last_tick = now_iso()
        if not self.settings.get()["capture_enabled"]:
            return []
        interval, start, end = self._window()
        slot = current_slot(now, interval, start, end)
        if slot is None:
            return []
        slot_at = _utc_iso(slot)
        done = []
        for cam in self.db.cameras():
            if self._stop.is_set():
                break
            if not cam["enabled"] or self.db.capture_slot_taken(cam["id"], slot_at):
                continue
            try:
                done.append(self.capture_now(cam["id"], "schedule", slot_at))
            except CaptureBusy:
                continue
            except Exception as e:  # noqa: BLE001 - the loop must never die
                log.exception("scheduled capture on %s failed: %s", cam["id"], e)
        return done

    def _loop(self) -> None:
        log.info("capture scheduler started")
        while not self._stop.wait(TICK_SECONDS):
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001
                log.exception("capture scheduler tick failed: %s", e)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="capture-scheduler", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def status(self) -> dict:
        cfg = self.settings.get()
        interval, start, end = self._window()
        now = datetime.now()
        cur = current_slot(now, interval, start, end)
        nxt = next_slot(now, interval, start, end)
        return {
            "enabled": cfg["capture_enabled"], "scheduler_running": self._thread is not None and self._thread.is_alive(),
            "interval_minutes": interval, "active_start": cfg["capture_active_start"], "active_end": cfg["capture_active_end"],
            "slots_per_day": slots_per_day(interval, start, end),
            "in_window": cur is not None, "current_slot": _utc_iso(cur) if cur else None, "next_slot": _utc_iso(nxt),
            "timezone": "local time of this PC", "inflight": sorted(self._inflight), "last_tick": self.last_tick,
            "sources": "local folder/video only (live CCTV not enabled)",
        }
