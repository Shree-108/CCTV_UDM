"""Upload handling: save the file, read its metadata, extract a preview frame and
suggest the CCTV grid layout."""
from __future__ import annotations

import uuid
from pathlib import Path

import cv2

from backend.core.config import IMAGE_EXT, UPLOADS, VIDEO_EXT
from backend.database.db import now_iso
from backend.detection.grid import detect_layout


def kind_of(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    return "image" if ext in IMAGE_EXT else "video" if ext in VIDEO_EXT else None


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
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        meta.update(fps=round(fps, 2) if fps else None, frames=frames or None,
                    duration_sec=round(frames / fps, 2) if fps and frames else None)
        for frac in (0.1, 0.4, 0.7):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames * frac))
            ok, f = cap.read()
            if ok:
                samples.append(f)
        if not samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = cap.read()
            if ok:
                samples.append(f)
        cap.release()
        if not samples:
            raise ValueError("No frames could be read from this video")
    h, w = samples[0].shape[:2]
    meta.update(width=w, height=h, suggested_layout=detect_layout(samples))
    cv2.imwrite(str(UPLOADS / f"{mid}_preview.jpg"), samples[0], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return meta


def new_upload_path(filename: str) -> Path:
    return UPLOADS / f"{uuid.uuid4().hex[:16]}{Path(filename).suffix.lower()}"
