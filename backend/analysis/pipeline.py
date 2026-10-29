"""Uniform analysis pipeline for uploaded images and videos.

Uploaded media -> selected CCTV tile -> YOLO person detection (+ ByteTrack for video)
 -> store ROI filter -> upper-body crop -> uniform classifier -> confidence check
 -> temporal voting (video) -> compliance decision -> alerts/evidence -> annotated media + report
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from backend.analysis import compliance
from backend.analysis.annotate import caption, draw_people, draw_roi, draw_status, enlarge, fit_width, tile_scale
from backend.analysis.temporal import PresenceTimeline
from backend.core.config import ANALYSES, TRACKER_CFG
from backend.core.model_config import INFERENCE, PERSON_MODEL, uniform_model_path
from backend.core.settings import CRF
from backend.database.db import DB
from backend.detection.grid import AnalysisArea
from backend.detection.person import PersonDetector, select_device
from backend.reports.writer import write_reports
from backend.tracking.memory import TrackMemory
from backend.uniform.classifier import UniformDetector

OUTPUT_MAX_W = 1280
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- models
class Models:
    """Lazily loaded; reloaded when the device / untrained mode / uniform weights change."""

    def __init__(self):
        self._lock = threading.Lock()
        self.image_lock = threading.Lock()
        self._key = None
        self.tracker_detector: PersonDetector | None = None
        self.image_detector: PersonDetector | None = None
        self.uniform: UniformDetector | None = None
        self.error: str | None = None

    def get(self, s: dict) -> "Models":
        device = select_device(s["gpu_acceleration"])
        path = uniform_model_path()
        mtime = path.stat().st_mtime if path.exists() else 0
        key = (device, s["untrained_model_mode"], str(path), mtime)
        with self._lock:
            if key != self._key:
                try:
                    self.tracker_detector = PersonDetector(PERSON_MODEL, device, TRACKER_CFG)
                    self.image_detector = PersonDetector(PERSON_MODEL, device)
                    self.uniform = UniformDetector(path, device, s["untrained_model_mode"])
                    self._key, self.error = key, None
                except Exception as e:  # noqa: BLE001 - surfaced in /api/health
                    self.error = str(e)
                    raise
        return self

    def warmup(self, s: dict) -> None:
        """One dummy inference at startup so the first real upload is not slowed by model initialisation."""
        t0 = time.time()
        with self.image_lock:
            self.image_detector.detect(np.zeros((480, 640, 3), np.uint8), s["person_confidence"], s["inference_size"])
            if self.uniform and self.uniform.backend:
                self.uniform.backend.predict([np.full((96, 64, 3), 127, np.uint8)])
        log.info("Models warmed up in %.1fs", time.time() - t0)

    @property
    def device(self) -> str:
        return self.tracker_detector.device if self.tracker_detector else "cpu"


def fmt_clock(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60:02d}:{sec % 60:02d}"


# --------------------------------------------------------------------------- shared step
def _analyse_tile(tile, area: AnalysisArea, s: dict, detector: PersonDetector, uniform: UniformDetector, tracked: bool):
    fn = detector.track if tracked else detector.detect
    everyone = fn(tile, s["person_confidence"], s["inference_size"])
    inside = [p for p in everyone if area.contains(p.box)]      # outside the ROI -> never analysed
    uniform.classify(tile, inside, s["uniform_confidence"], s["non_uniform_confidence"],
                     s["min_person_height_px"], occluders=[p.box for p in everyone])
    return inside


def _annotated(frame, tile, area: AnalysisArea, people, smoothed: bool):
    """Output = the selected camera tile only (enlarged if small), with ROI, boxes, IDs,
    labels and confidence. The other DVR cameras are not part of the result."""
    k = tile_scale(tile.shape[1])
    t = enlarge(tile, k)
    draw_roi(t, area.roi_pixels(), k)
    draw_people(t, people, smoothed, k)
    caption(t, area.label, whole_frame=tile.shape[:2] == frame.shape[:2])
    return t


def _encode(img: np.ndarray, q: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else b""


def _label_name(label: str) -> str:
    return {"uniform": "UNIFORM", "non_uniform": "NON-UNIFORM", "uncertain": "UNCERTAIN"}[label]


@dataclass
class LiveView:
    """In-memory progress of a running video analysis (polled by the UI)."""
    jpeg: bytes = b""
    snapshot: dict | None = None


# --------------------------------------------------------------------------- image
def analyze_image(db: DB, aid: str, media: dict, area: AnalysisArea, s: dict, models: Models) -> None:
    t0 = time.time()
    out_dir = ANALYSES / aid
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(media["path"])
    if img is None:
        raise ValueError("Could not read the image file")
    area.bind(img.shape[1], img.shape[0])
    tile = area.crop(img)
    with models.image_lock:
        people = _analyse_tile(tile, area, s, models.image_detector, models.uniform, tracked=False)
    for p in people:    # single image: the frame label is final
        p.smoothed_label, p.smoothed_conf = p.label, p.conf
    labels = [p.label for p in people]
    status = compliance.decide(labels, INFERENCE["overall_rule"])
    n_u = labels.count("uniform")
    out = _annotated(img, tile, area, people, smoothed=False)
    draw_status(out, compliance.TITLE[status], compliance.COLOR[status],
                f"{len(people)} in store area · {n_u} uniformed")
    out = fit_width(out, 1600)
    annotated = out_dir / "annotated.jpg"
    cv2.imwrite(str(annotated), out, [cv2.IMWRITE_JPEG_QUALITY, 90])

    persons = [{
        "track_id": p.track_id, "label": p.label, "confidence": round(p.conf, 4), "det_confidence": round(p.det_conf, 4),
        "first_seen_sec": None, "last_seen_sec": None, "frames": 1,
        "uniform_votes": int(p.label == "uniform"), "non_uniform_votes": int(p.label == "non_uniform"),
        "uncertain_votes": int(p.label == "uncertain"), "note": p.reason,
        "box": json.dumps([round(v, 1) for v in p.box]),
    } for p in people]
    db.set_persons(aid, persons)

    n_alerts = 0
    if status == compliance.NO_UNIFORM:
        db.add_alert({"analysis_id": aid, "kind": "no_uniform", "severity": "crit", "title": "NO UNIFORM DETECTED",
                      "message": compliance.message(status, n_u, len(people)), "media_time_sec": None,
                      "people_inside": len(people), "uniformed": 0,
                      "confidence": max((p.conf for p in people if p.label == "non_uniform"), default=None),
                      "evidence": str(annotated), "filename": media["filename"]})
        n_alerts = 1

    uni = [p.conf for p in people if p.label == "uniform"]
    summary = dict(
        progress=1.0, finished_at=_now(), status=status, people=len(people), uniformed=n_u,
        non_uniform=labels.count("non_uniform"), uncertain=labels.count("uncertain"),
        avg_uniform_conf=float(np.mean(uni)) if uni else None, max_uniform_conf=max(uni) if uni else None,
        processing_sec=round(time.time() - t0, 2), frames_total=1, frames_processed=1, alerts=n_alerts,
        uniform_obs=n_u, non_uniform_obs=labels.count("non_uniform"), uncertain_obs=labels.count("uncertain"),
        classifier=models.uniform.backend_name, model_version=models.uniform.model_version, device=models.device,
    )
    db.update_analysis(aid, **summary)
    files = write_reports(db, aid, out_dir, annotated=annotated, keyframe=annotated)
    db.update_analysis(aid, state="done", files=files)      # "done" only ever appears together with its files


# --------------------------------------------------------------------------- video
def analyze_video(db: DB, aid: str, media: dict, area: AnalysisArea, s: dict, models: Models,
                  live: LiveView, should_stop: Callable[[], bool]) -> None:
    out_dir = ANALYSES / aid
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(media["path"])
    if not cap.isOpened():
        raise ValueError("Could not open the video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = fps if fps > 1 else 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    area.bind(W, H)
    step = s["frame_skip"] or max(1, round(fps / s["analysis_fps"]))

    detector, uniform = models.tracker_detector, models.uniform
    detector.reset()
    memory = TrackMemory(s["vote_window"], s["temporal_confirmation_frames"])
    timeline = PresenceTimeline(s["presence_gap_tolerance"])
    raw_path = out_dir / "annotated_raw.mp4"
    writer = None
    t0 = time.time()
    last_db = 0.0
    idx = processed = 0
    obs = {"uniform": 0, "non_uniform": 0, "uncertain": 0}
    # alert state
    absence, prev_t, episode_alerted = 0.0, None, False
    n_alerts = 0
    violation_frame: tuple[int, bytes, float, int] | None = None     # (people, jpeg, t, n_non)
    key_frame: tuple[float, bytes] | None = None                     # best uniform evidence
    lowconf: dict[int, tuple[float, bytes, float]] = {}              # track -> (max P(uniform), jpeg, t)
    current_status = compliance.NO_PERSON
    # seconds of footage needed to fill the vote window before a "no uniform" state is shown
    confirm_window = max(2.0, s["vote_window"] * step / fps)

    try:
        while not should_stop():
            if idx % step:
                if not cap.grab():                                   # skipped: not decoded
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            idx += 1
            processed += 1

            tile = area.crop(frame)
            people = _analyse_tile(tile, area, s, detector, uniform, tracked=True)
            memory.update(people, t)
            present = any(p.smoothed_label == "uniform" for p in people)
            timeline.update(t, present)
            for p in people:
                obs[p.label] += 1

            # ---- absence of a confirmed uniform while people are inside (occlusion-tolerant)
            dt = 0.0 if prev_t is None else min(t - prev_t, 5.0)
            prev_t = t
            if timeline.present_at(t):
                absence, episode_alerted = 0.0, False
            elif people:
                absence += dt

            # ---- current store status for the live view / burned-in bar. While the vote
            # window is still filling it is MONITORING - never "no uniform" from a few frames.
            monitoring = False
            if timeline.present_at(t):
                current_status = compliance.COMPLIANT
            elif not people:
                current_status = compliance.NO_PERSON
            elif absence < confirm_window:
                current_status, monitoring = compliance.UNCERTAIN, True
            elif any(p.smoothed_label == "non_uniform" for p in people):
                current_status = compliance.NO_UNIFORM
            else:
                current_status = compliance.UNCERTAIN

            out = _annotated(frame, tile, area, people, smoothed=True)
            n_u_now = sum(p.smoothed_label == "uniform" for p in people)
            draw_status(out, "MONITORING · confirming uniform…" if monitoring else compliance.TITLE[current_status],
                        compliance.COLOR[current_status],
                        f"{fmt_clock(t)}  ·  people in store area: {len(people)}  ·  uniformed: {n_u_now}")
            out = fit_width(out, OUTPUT_MAX_W)
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (w, h))
            writer.write(out)
            jpeg = None

            # ---- sustained absence -> NO UNIFORM alert (never from a single frame)
            if people and not timeline.present_at(t):
                n_non = sum(p.smoothed_label == "non_uniform" for p in people)
                if violation_frame is None or (n_non, len(people)) > (violation_frame[3], violation_frame[0]):
                    jpeg = jpeg or _encode(out)
                    violation_frame = (len(people), jpeg, t, n_non)
                if absence >= s["absence_alert_seconds"] and not episode_alerted and n_non > 0:
                    jpeg = jpeg or _encode(out)
                    ev = out_dir / f"evidence_{n_alerts + 1:02d}.jpg"
                    ev.write_bytes(jpeg)
                    db.add_alert({"analysis_id": aid, "kind": "no_uniform", "severity": "crit",
                                  "title": "NO UNIFORMED PERSON DETECTED",
                                  "message": f"No confirmed uniform for {absence:.0f}s while people were inside the store area.",
                                  "media_time_sec": round(t, 2), "people_inside": len(people), "uniformed": 0,
                                  "confidence": max((p.smoothed_conf for p in people if p.smoothed_label == "non_uniform"), default=None),
                                  "evidence": str(ev), "filename": media["filename"]})
                    n_alerts += 1
                    episode_alerted = True

            # ---- evidence candidates
            score = sum(p.smoothed_conf for p in people if p.smoothed_label == "uniform")
            if score > 0 and (key_frame is None or score > key_frame[0]):
                jpeg = jpeg or _encode(out)
                key_frame = (score, jpeg)
            for p in people:
                pu = p.extra.get("uniform_prob")
                if pu is not None and 0.40 <= pu < s["uniform_confidence"] and pu > lowconf.get(p.track_id, (0,))[0]:
                    jpeg = jpeg or _encode(out)
                    lowconf[p.track_id] = (pu, jpeg, t)

            # ---- live progress (UI polls this)
            tracks = memory.people(s["min_track_observations"])
            verdicts = [memory.verdict(r)[0] for r in tracks]
            live.jpeg = _encode(out, 70)
            live.snapshot = {
                "frame": idx, "frames_total": total, "processed": processed,
                "progress": round(idx / total, 4) if total else None,
                "people": len(tracks), "uniformed": verdicts.count("uniform"),
                "non_uniform": verdicts.count("non_uniform"), "uncertain": verdicts.count("uncertain"),
                "people_now": len(people), "current_status": current_status, "monitoring": monitoring,
                "processing_fps": round(processed / max(time.time() - t0, 1e-6), 1),
                "media_time_sec": round(t, 2), "alerts": n_alerts,
            }
            if time.time() - last_db > 1.0:
                db.update_analysis(aid, progress=live.snapshot["progress"] or 0, live=live.snapshot, frames_processed=processed)
                last_db = time.time()
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    stopped = should_stop()
    elapsed = time.time() - t0
    tracks = memory.people(s["min_track_observations"])
    persons, labels, uni_confs = [], [], []
    for r in tracks:
        label, conf = memory.verdict(r)
        labels.append(label)
        if label == "uniform":
            uni_confs.append(conf)
        persons.append({
            "track_id": r.track_id, "label": label, "confidence": round(conf, 4), "det_confidence": round(r.best_det_conf, 4),
            "first_seen_sec": round(r.first_seen, 2), "last_seen_sec": round(r.last_seen, 2), "frames": r.observations,
            "uniform_votes": r.counts["uniform"], "non_uniform_votes": r.counts["non_uniform"],
            "uncertain_votes": r.counts["uncertain"], "note": "", "box": None,
        })
        # persistent uncertainty for this person (many frames, never confident) -> LOW CONFIDENCE
        if (s["low_confidence_alerts"] and label == "uncertain" and r.track_id in lowconf
                and r.observations >= 2 * s["min_track_observations"]):
            pu, jpeg, lt = lowconf[r.track_id]
            ev = out_dir / f"lowconf_{r.track_id:02d}.jpg"
            ev.write_bytes(jpeg)
            db.add_alert({"analysis_id": aid, "kind": "low_confidence", "severity": "warn", "title": "LOW CONFIDENCE",
                          "message": f"Person #{r.track_id:02d} · Uniform confidence {pu * 100:.0f}% "
                                     f"(below the {s['uniform_confidence'] * 100:.0f}% threshold) across {r.observations} frames",
                          "media_time_sec": round(lt, 2), "people_inside": None, "uniformed": None,
                          "confidence": round(pu, 4), "track_id": r.track_id, "evidence": str(ev),
                          "filename": media["filename"]})
            n_alerts += 1
    db.set_persons(aid, persons)
    status = compliance.decide(labels, INFERENCE["overall_rule"])

    # a confirmed whole-video NO UNIFORM verdict always has evidence, even for clips shorter than the threshold
    if status == compliance.NO_UNIFORM and not _alerts_of(db, aid, "no_uniform") and violation_frame:
        ev = out_dir / f"evidence_{n_alerts + 1:02d}.jpg"
        ev.write_bytes(violation_frame[1])
        db.add_alert({"analysis_id": aid, "kind": "no_uniform", "severity": "crit", "title": "NO UNIFORMED PERSON DETECTED",
                      "message": compliance.message(status, 0, len(labels)), "media_time_sec": round(violation_frame[2], 2),
                      "people_inside": violation_frame[0], "uniformed": 0, "confidence": None,
                      "evidence": str(ev), "filename": media["filename"]})
        n_alerts += 1

    annotated = _to_h264(raw_path, out_dir / "annotated.mp4", CRF[s["output_video_quality"]])
    keyframe = out_dir / "keyframe.jpg"
    kf = key_frame[1] if key_frame else (violation_frame[1] if violation_frame else live.jpeg)
    if kf:
        keyframe.write_bytes(kf)

    db.update_analysis(
        aid, progress=1.0, finished_at=_now(), status=status, people=len(labels),
        uniformed=labels.count("uniform"), non_uniform=labels.count("non_uniform"), uncertain=labels.count("uncertain"),
        avg_uniform_conf=float(np.mean(uni_confs)) if uni_confs else None, max_uniform_conf=max(uni_confs) if uni_confs else None,
        duration_sec=round((total or idx) / fps, 2), processing_sec=round(elapsed, 2), frames_total=total or idx,
        frames_processed=processed, processing_fps=round(processed / max(elapsed, 1e-6), 2),
        longest_presence_sec=round(timeline.longest, 2), alerts=n_alerts,
        uniform_obs=obs["uniform"], non_uniform_obs=obs["non_uniform"], uncertain_obs=obs["uncertain"],
        classifier=uniform.backend_name, model_version=uniform.model_version, device=models.device,
        error="Stopped by user before the end of the video" if stopped else None,
        live={**(live.snapshot or {}), "segments": [[round(a, 2), round(b, 2)] for a, b in timeline.segments]},
    )
    files = write_reports(db, aid, out_dir, annotated=annotated, keyframe=keyframe if kf else None)
    db.update_analysis(aid, state="done", files=files)      # "done" only ever appears together with its files


def _alerts_of(db: DB, aid: str, kind: str):
    rows, _ = db.alerts(limit=1, analysis_id=aid, kind=kind)
    return rows


def _now() -> str:
    from backend.database.db import now_iso
    return now_iso()


def _to_h264(src: Path, dst: Path, crf: int) -> Path:
    """OpenCV's mp4v does not play in browsers; re-encode to H.264 with the bundled ffmpeg."""
    if not src.exists():
        return src
    try:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(dst)], check=True, timeout=3600)
        src.unlink(missing_ok=True)
        return dst
    except Exception as e:  # noqa: BLE001
        print(f"[warn] H.264 conversion failed ({e}); serving mp4v")
        return src
