"""Extract the planned frames from a video file - one frame in memory at a time.

The video is decoded strictly in order from its first frame, in ONE pass (samples in frame order);
frames between samples are decoded but not converted or kept. Seeking straight to a frame number is
not used: in containers without a seek index (NVR exports: HEVC in MPEG-PS) the decoder then starts in
the middle of a group of pictures, without its reference pictures, and returns grey frames - or a
different frame than the one asked for. The frame number is the position in decode order, and the
timestamp is the frame's presentation time measured from the first frame (NVR exports have recording
gaps, so frame_number / fps is wrong there); a frame without a timestamp of its own is placed after the
last frame that had one, at the frame rate.

Every decoded sample is checked (frame_quality.problem) before it is saved: a corrupted or blank frame
is a failure (error_category frame_unreadable), never an image - so it is never analysed or reported
as "no person detected". It is not replaced by a neighbouring frame. The image is written exactly as
decoded (PNG = lossless; JPG optional): nothing is drawn on it, cropped, resized or otherwise changed.

Outputs (never overwritten - every job has its own folder):
  storage/extractions/<job>/frames/<image files>
  storage/extractions/<job>/image_manifest.csv   successfully saved frames only (written row by row)
  storage/extractions/<job>/failures.csv         samples that could not be extracted

Timestamps: `video_timestamp_*` is the position inside the video file. `recording_timestamp` is a
real clock time and is filled ONLY when the recording start time of the video was supplied.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Callable

import cv2

from backend.core.storage import StorageBackend, get_storage, rel  # noqa: F401 - rel re-exported
from backend.extraction import frame_quality
from backend.extraction.plan import Sample, clock_time, hhmmss

MANIFEST_FIELDS = ["image_id", "extraction_job_id", "source_video", "store_id", "camera_id",
                   "video_timestamp_seconds", "video_timestamp_hhmmss", "frame_number", "sampling_strategy",
                   "sampling_interval_seconds", "image_filename", "original_image_path", "image_width",
                   "image_height", "extraction_status",
                   # supporting fields (after the required ones)
                   "source_video_id", "sample_type", "random_seed", "recording_start_time", "recording_timestamp"]
MANIFEST_NAME = "image_manifest.csv"
LEGACY_MANIFEST_NAME = "manifest.csv"        # jobs created before the file was renamed
WINDOWS_MAX_PATH = 260
FAILURE_FIELDS = ["image_id", "extraction_job_id", "source_video", "stage", "sample_type", "target_seconds",
                  "target_hhmmss", "target_frame", "error_category", "error_message"]
FRAME_UNREADABLE = "frame_unreadable"        # error_category of a decoded but corrupted / blank frame
# columns written by jobs before the schema change -> current names
_LEGACY_MANIFEST = {"timestamp_seconds": "video_timestamp_seconds", "timestamp_hhmmss": "video_timestamp_hhmmss",
                    "image_path": "original_image_path"}


def row_meta(job: dict) -> dict:
    """Per-job values repeated on every manifest row. Unknown store / camera / recording start stay blank."""
    strategy = job.get("sampling_strategy") or "interval"
    seed = job.get("random_seed")
    return {"extraction_job_id": job["id"], "source_video": job["source_video"], "source_video_id": job["media_id"],
            "store_id": job.get("store_id") or "", "camera_id": job.get("camera_id") or "",
            "sampling_strategy": strategy,
            "sampling_interval_seconds": job["interval_seconds"] if strategy == "interval" else "",
            "random_seed": seed if seed is not None else "",
            "recording_start_time": job.get("recording_start_time") or ""}


def normalize_manifest_row(row: dict, meta: dict) -> dict:
    """A manifest row in the current schema - also for files written by older jobs."""
    for old, new in _LEGACY_MANIFEST.items():
        if old in row and new not in row:
            row[new] = row.pop(old)
    for k, v in meta.items():
        if row.get(k) in (None, ""):
            row[k] = v
    if not row.get("recording_timestamp"):
        row["recording_timestamp"] = clock_time(row["recording_start_time"], float(row["video_timestamp_seconds"]))
    return row


def _safe(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-")[:40] or "video"


def write_failure(path: Path) -> tuple[str, str]:
    """(error_category, message) for an image that could not be written - names the Windows
    path-length limit when that is the cause."""
    n = len(str(path.resolve()))
    if os.name == "nt" and n >= WINDOWS_MAX_PATH:
        return "path_too_long", (f"the image could not be written: the file path is {n} characters, over the Windows "
                                 f"limit of {WINDOWS_MAX_PATH}; move the project (or SU_STORAGE_DIR) to a shorter folder")
    return "write_failed", "the image could not be written (disk full, no permission, or the folder was removed?)"


def write_failure_reason(path: Path) -> str:
    return write_failure(path)[1]


def image_id(job_id: str, s: Sample) -> str:
    tag = {"initial": "I", "random": "R"}.get(s.sample_type, "S")
    return f"{job_id}-{tag}{s.index:04d}"


class SequentialReader:
    """Frames of an opened cv2.VideoCapture in decode order, from the first one - never by seeking.
    Frames must be asked for in increasing order (the same frame twice is allowed)."""
    STOPPED = "stopped"

    def __init__(self, cap, fps: float):
        self.cap, self.fps = cap, fps
        self.next = 0                   # number of the frame the next grab() decodes
        self.origin_ms = 0.0            # decoder clock at frame 0 (NVR exports can start at "26 h")
        self.known = (0, 0.0)           # (frame, seconds) of the last frame that had its own timestamp
        self.last: tuple | None = None  # (frame, image, seconds) of the last frame returned

    def _grab(self) -> float | None:
        """Decode the next frame; its time in seconds from frame 0, or None at the end of the video."""
        if not self.cap.grab():
            return None
        n, ms = self.next, self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.next += 1
        if n == 0:
            self.origin_ms = ms
            return 0.0
        if ms > 0:                                    # 0 = this frame carries no timestamp
            t = (ms - self.origin_ms) / 1000
            if t > self.known[1]:
                self.known = (n, t)
                return t
        return self.known[1] + (n - self.known[0]) / self.fps

    def read(self, number: int, should_stop: Callable[[], bool] = lambda: False) -> tuple:
        """(image, seconds from frame 0, error) of frame `number`; error is "" when the image is there and
        STOPPED when should_stop() asked to stop while decoding towards it."""
        if self.last and self.last[0] == number:
            return self.last[1], self.last[2], ""
        if number < self.next:
            return None, None, f"frame {number} was already passed (frames are read in order)"
        while self.next < number:
            if self.next % 250 == 0 and should_stop():
                return None, None, self.STOPPED
            if self._grab() is None:
                return None, None, f"the video ends after {self.next} decodable frames, before frame {number}"
        t = self._grab()
        if t is None:
            return None, None, f"the video ends after {self.next} decodable frames, before frame {number}"
        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            return None, None, "the decoder returned no image for this frame"
        self.last = (number, frame, t)
        return frame, t, ""


def extract(video_path: str, meta: dict, out_dir: Path, fps: float, samples: list[Sample], image_format: str = "png",
            on_progress: Callable[[int, int, int], None] | None = None,
            should_stop: Callable[[], bool] = lambda: False, storage: StorageBackend | None = None) -> dict:
    """`meta` = row_meta(job). Returns {"extracted", "failed", "failures": [...], "stopped": bool}."""
    storage = storage or get_storage()
    job_id, source_video = meta["extraction_job_id"], meta["source_video"]
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)          # a job folder is never reused
    ext = ".png" if image_format == "png" else ".jpg"
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if ext == ".png" else [cv2.IMWRITE_JPEG_QUALITY, 95]
    stem = f"{_safe(Path(source_video).stem)}_{meta['source_video_id'][:8]}"
    failures: list[dict] = []
    extracted = 0
    stopped = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("The video could not be opened (missing codec or unreadable file)")
    reader = SequentialReader(cap, fps)
    with open(out_dir / MANIFEST_NAME, "w", newline="", encoding="utf-8") as mf, \
            open(out_dir / "failures.csv", "w", newline="", encoding="utf-8") as ff:
        mw = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        fw = csv.DictWriter(ff, fieldnames=FAILURE_FIELDS)
        mw.writeheader(); fw.writeheader()
        mf.flush(); ff.flush()
        try:
            for n, s in enumerate(sorted(samples, key=lambda x: (x.target_frame, x.index)), 1):
                if should_stop():
                    stopped = True
                    break
                iid = image_id(job_id, s)

                def fail(category: str, message: str, seconds: float | None = None) -> None:
                    """`seconds` = the decoded frame's real time when it is known (an unreadable frame was
                    decoded); otherwise the planned time (frame / fps) is all there is."""
                    at = s.target_seconds if seconds is None else round(seconds, 3)
                    row = {"image_id": iid, "extraction_job_id": job_id, "source_video": source_video,
                           "stage": "extraction", "sample_type": s.sample_type, "target_seconds": at,
                           "target_hhmmss": hhmmss(at), "target_frame": s.target_frame,
                           "error_category": category, "error_message": message}
                    fw.writerow(row); ff.flush()
                    failures.append(row)

                frame = None
                try:
                    frame, t, err = reader.read(s.target_frame, should_stop)
                except cv2.error as e:                        # decoder error on this frame only
                    fail("decoder_error", f"decoder error: {e.msg if hasattr(e, 'msg') else e}")
                else:
                    if err == reader.STOPPED:
                        stopped = True
                        break
                    if err:
                        fail("no_frame", err)
                    elif why := frame_quality.problem(frame):
                        fail(FRAME_UNREADABLE, why, t)    # never saved, so never analysed
                        frame = None
                if frame is not None:
                    actual = s.target_frame
                    tag = {"initial": "initial", "random": f"r{s.index:04d}"}.get(s.sample_type, f"s{s.index:04d}")
                    name = f"{stem}_{job_id[-6:]}_{tag}_f{actual:07d}_t{hhmmss(t).replace(':', '').replace('.', '')}{ext}"
                    path = frames_dir / name
                    try:
                        written = cv2.imwrite(str(path), frame, params)
                    except cv2.error:
                        written = False
                    if not written:
                        fail(*write_failure(path))
                    else:
                        h, w = frame.shape[:2]
                        mw.writerow({k: meta[k] for k in meta} | {
                            "image_id": iid, "video_timestamp_seconds": round(t, 3),
                            "video_timestamp_hhmmss": hhmmss(t), "frame_number": actual, "image_filename": name,
                            "original_image_path": storage.put(path, "original"), "image_width": w, "image_height": h,
                            "extraction_status": "ok", "sample_type": s.sample_type,
                            "recording_timestamp": clock_time(meta["recording_start_time"], t)})
                        mf.flush()
                        extracted += 1
                del frame
                if on_progress:
                    on_progress(n, extracted, len(failures))
        finally:
            cap.release()
    return {"extracted": extracted, "failed": len(failures), "failures": failures, "stopped": stopped}
