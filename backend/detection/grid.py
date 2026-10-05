"""CCTV grid (multi-camera DVR layout) selection and store-interior ROI.

Only the selected camera tile is ever passed to the detector, so people and
objects in the other tiles can never be detected. The optional ROI is a
polygon inside that tile (normalised 0..1 tile coordinates): only people whose
reference point lies inside it count towards the compliance result.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# id -> (label, rows, cols)
LAYOUTS: dict[str, tuple[str, int, int]] = {
    "1x1": ("Single view", 1, 1),
    "2x2": ("2 × 2 grid", 2, 2),
    "1x2": ("1 × 2 side by side", 1, 2),
    "2x1": ("2 × 1 stacked", 2, 1),
    "2x3": ("2 × 3 grid", 2, 3),
    "3x3": ("3 × 3 grid", 3, 3),
    "4x4": ("4 × 4 grid", 4, 4),
}

Rect = tuple[int, int, int, int]


def grid_cells(rows: int, cols: int) -> list[list[float]]:
    """Normalised cells, row-major (Camera 1 = top-left)."""
    return [[c / cols, r / rows, (c + 1) / cols, (r + 1) / rows] for r in range(rows) for c in range(cols)]


def _seam_strength(gray: np.ndarray, pos: int, axis: int) -> float:
    last = gray.shape[axis] - 1
    a_i, b_i = max(pos - 2, 0), min(pos + 1, last)
    if axis == 1:
        a, b, line = gray[:, a_i], gray[:, b_i], gray[:, pos]
    else:
        a, b, line = gray[a_i, :], gray[b_i, :], gray[pos, :]
    a, b, line = a.astype(np.float32), b.astype(np.float32), line.astype(np.float32)
    jump = np.abs(a - b)                     # content changes across the seam
    border = np.abs(line - (a + b) / 2)      # or a separator line is drawn on it
    return float(np.percentile(np.maximum(jump, border), 60))


def detect_layout(frames: list[np.ndarray]) -> str:
    """Suggest a layout id from seam lines at tile borders (the user can override)."""
    if not frames:
        return "1x1"
    grays = [cv2.cvtColor(cv2.resize(f, (640, 360)), cv2.COLOR_BGR2GRAY) for f in frames]
    best, best_rank = "1x1", 0.0
    rng = np.random.default_rng(0)
    for lid, (_, rows, cols) in LAYOUTS.items():
        if rows == cols == 1:
            continue
        ratios = []
        for g in grays:
            h, w = g.shape
            base = np.median([_seam_strength(g, int(x), 1) for x in rng.integers(20, w - 20, 6)] +
                             [_seam_strength(g, int(y), 0) for y in rng.integers(20, h - 20, 6)]) + 2.0
            seams = [_seam_strength(g, round(w * c / cols), 1) for c in range(1, cols)] + \
                    [_seam_strength(g, round(h * r / rows), 0) for r in range(1, rows)]
            ratios.append(min(seams) / base)   # every seam must be visible
        score = float(np.median(ratios))
        if score > 3.0:                        # accepted: prefer the layout with more tiles
            rank = rows * cols + min(score, 50) / 100
            if rank > best_rank:
                best, best_rank = lid, rank
    return best


@dataclass
class AnalysisArea:
    rect_norm: list[float]                   # selected tile, normalised full-frame [x1, y1, x2, y2]
    roi: list[list[float]] | None = None     # polygon inside the tile, normalised tile coords
    label: str = "Camera 1"
    reference_point: str = "bottom_center"
    frame_w: int = 0
    frame_h: int = 0

    def bind(self, w: int, h: int) -> "AnalysisArea":
        self.frame_w, self.frame_h = w, h
        return self

    @property
    def rect(self) -> Rect:
        x1, y1, x2, y2 = self.rect_norm
        W, H = self.frame_w, self.frame_h
        r = (int(round(x1 * W)), int(round(y1 * H)), int(round(x2 * W)), int(round(y2 * H)))
        return max(0, r[0]), max(0, r[1]), min(W, r[2]), min(H, r[3])

    @property
    def tile_size(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.rect
        return x2 - x1, y2 - y1

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.rect
        return frame[y1:y2, x1:x2]

    def roi_pixels(self) -> np.ndarray | None:
        if not self.roi or len(self.roi) < 3:
            return None
        w, h = self.tile_size
        return np.array([[x * w, y * h] for x, y in self.roi], dtype=np.float32)

    def contains(self, box) -> bool:
        poly = self.roi_pixels()
        if poly is None:
            return True
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = {"bottom_center": y2 - 0.03 * (y2 - y1),
              "upper_center": y1 + 0.3 * (y2 - y1)}.get(self.reference_point, (y1 + y2) / 2)
        return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


def validate_area(rect: list[float], roi: list[list[float]] | None) -> None:
    if len(rect) != 4:
        raise ValueError("grid rectangle needs 4 values")
    x1, y1, x2, y2 = rect
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1) or (x2 - x1) < 0.03 or (y2 - y1) < 0.03:
        raise ValueError("invalid or too small camera area")
    if roi is not None:
        if len(roi) < 3:
            raise ValueError("ROI needs at least 3 points")
        if any(len(p) != 2 or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1) for p in roi):
            raise ValueError("ROI points must be normalised (0..1)")
