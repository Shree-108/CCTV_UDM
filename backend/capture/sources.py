"""Frame sources for scheduled capture.

  folder : the next image of a folder (sorted by name, wraps around) - e.g. saved CCTV screenshots
  video  : the frame at the next position of a video file (advances `video_step_seconds`, wraps)

Both are LOCAL test sources. Live CCTV (RTSP / NVR) is deliberately not available until
authorised access and the recorder details are confirmed.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.core.config import IMAGE_EXT, VIDEO_EXT

SOURCE_TYPES = ("folder", "video")


class CaptureError(RuntimeError):
    """The frame could not be obtained. `next_cursor` lets one bad file be skipped next time."""

    def __init__(self, message: str, next_cursor: float | None = None):
        super().__init__(message)
        self.next_cursor = next_cursor


def validate_source(source_type: str, source: str) -> str:
    """Return the resolved path, or raise ValueError with a user-facing message."""
    if source_type not in SOURCE_TYPES:
        raise ValueError("source_type must be 'folder' or 'video' (live CCTV is not enabled yet)")
    p = Path(source).expanduser()
    if not p.is_absolute():
        from backend.core.config import ROOT
        p = ROOT / p
    p = p.resolve()
    if source_type == "folder":
        if not p.is_dir():
            raise ValueError(f"Folder not found: {source}")
        if not _images(p):
            raise ValueError("The folder contains no JPG/PNG/BMP/WEBP images")
    else:
        if not p.is_file():
            raise ValueError(f"Video file not found: {source}")
        if p.suffix.lower() not in VIDEO_EXT:
            raise ValueError("Unsupported video type")
    return str(p)


def _images(folder: Path) -> list[Path]:
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXT)


def grab(camera: dict, video_step_seconds: float) -> tuple[np.ndarray, str, float]:
    """Return (frame BGR, source reference, next cursor)."""
    st, src, cursor = camera["source_type"], Path(camera["source"]), float(camera.get("cursor") or 0)
    if st == "folder":
        if not src.is_dir():
            raise CaptureError(f"Source folder not found: {src.name}")
        files = _images(src)
        if not files:
            raise CaptureError("Source folder has no images")
        i = int(cursor) % len(files)
        frame = cv2.imread(str(files[i]))
        if frame is None:
            raise CaptureError(f"Could not read image {files[i].name}", next_cursor=i + 1)
        return frame, files[i].name, float(i + 1)
    if st == "video":
        cap = cv2.VideoCapture(str(src))
        try:
            if not cap.isOpened():
                raise CaptureError(f"Could not open video {src.name}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = frames / fps if frames and fps else 0.0
            pos = cursor % duration if duration > 0 else 0.0
            cap.set(cv2.CAP_PROP_POS_MSEC, pos * 1000.0)
            ok, frame = cap.read()
            if not ok:                      # some containers cannot seek precisely: fall back to the start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame, pos = (*cap.read(), 0.0)
            if not ok or frame is None:
                raise CaptureError(f"No frame could be read from {src.name}")
            return frame, f"{src.name}@{pos:.1f}s", pos + video_step_seconds
        finally:
            cap.release()
    raise CaptureError(f"Unsupported source type: {st}")


def dhash(frame: np.ndarray) -> str:
    """64-bit difference hash: near-identical frames give hashes a few bits apart."""
    g = cv2.cvtColor(cv2.resize(frame, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")
