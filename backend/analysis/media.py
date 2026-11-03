"""Upload handling: save the file, read its metadata, extract a preview frame and
suggest the CCTV grid layout."""
from __future__ import annotations

import uuid
from pathlib import Path

import cv2

from backend.core.config import IMAGE_EXT, UPLOADS, VIDEO_EXT
from backend.database.db import now_iso
from backend.detection.grid import detect_layout
from backend.extraction import frame_quality

PREVIEW_SCAN_FRAMES = 400


def kind_of(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    return "image" if ext in IMAGE_EXT else "video" if ext in VIDEO_EXT else None


def _packets(path: Path) -> tuple[int, float] | None:
    """(video packets, seconds from the first packet to the last) read WITHOUT decoding (OpenCV raw stream
    mode; well under a second even for a 200 MB file), or None when the backend cannot do that."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_FORMAT, -1):
            return None
        n, first, last = 0, None, 0.0
        while cap.grab():
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if ms > 0:
                first = ms if first is None else min(first, ms)
                last = max(last, ms)
            n += 1
        return n, ((last - first) / 1000 if first is not None else 0.0)
    finally:
        cap.release()


def _real_length(cap, path: Path, reported: int, fps: float) -> tuple[int, float | None]:
    """(frames, duration). Some NVR exports carry a wrong length in the container (e.g. 26 h for a 3-minute
    clip, or 2900 frames for 2538), so samples planned from it fall past the real end, and recording gaps
    make frames / fps shorter than the real duration. The packets are counted instead; when they disagree
    with the container the last one is not counted (cut exports end with an incomplete one) and the
    duration is taken from the packet timestamps."""
    counted = _packets(path)
    if counted is None:                          # no raw mode: count the frames that actually decode
        n = reported
        if reported > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            n = 0
            while cap.grab():
                n += 1
        return n, (n / fps if fps and n else None)
    packets, span = counted
    if packets == reported:
        return reported, (reported / fps if fps and reported else None)
    n = max(packets - 1, 0)
    return n, (span + 1 / fps if fps and n else None)


def probe(path: Path, filename: str, kind: str, size: int) -> dict:
    mid = path.stem
    meta = {"id": mid, "created_at": now_iso(), "kind": kind, "filename": filename, "path": str(path),
            "size_bytes": size, "width": None, "height": None, "fps": None, "frames": None, "duration_sec": None}
    samples = []
    if kind == "image":
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError("This image could not be decoded")
        samples = [img]
    else:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("This video could not be opened (unsupported codec?)")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames, duration = _real_length(cap, path, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), fps)
        meta.update(fps=round(fps, 2) if fps else None, frames=frames or None,
                    duration_sec=round(duration, 2) if duration else None)
        for frac in (0.1, 0.4, 0.7):                 # preview / layout only: a seek can return a broken frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames * frac))
            ok, f = cap.read()
            if ok and frame_quality.problem(f) is None:
                samples.append(f)
        cap.release()
        if not samples:                              # the first readable frame, decoded in order
            cap, first = cv2.VideoCapture(str(path)), None
            for _ in range(PREVIEW_SCAN_FRAMES):
                ok, f = cap.read()
                if not ok:
                    break
                first = f if first is None else first
                if frame_quality.problem(f) is None:
                    samples = [f]
                    break
            cap.release()
            if not samples and first is not None:
                samples = [first]
        if not samples:
            raise ValueError("No frames could be read from this video")
    h, w = samples[0].shape[:2]
    meta.update(width=w, height=h, suggested_layout=detect_layout(samples))
    cv2.imwrite(str(UPLOADS / f"{mid}_preview.jpg"), samples[0], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return meta


def new_upload_path(filename: str) -> Path:
    return UPLOADS / f"{uuid.uuid4().hex[:16]}{Path(filename).suffix.lower()}"
