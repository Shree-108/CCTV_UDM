"""Burned-in annotations for annotated images / videos / evidence frames."""
from __future__ import annotations

import cv2
import numpy as np

# BGR, matched to the dashboard palette
C = {
    "uniform": (153, 211, 52),      # --ok  #34d399
    "non_uniform": (113, 113, 248),  # --crit #f87171
    "uncertain": (36, 191, 251),     # --warn #fbbf24
    "cyan": (238, 211, 34),          # --info #22d3ee
    "green": (153, 211, 52), "red": (113, 113, 248), "yellow": (36, 191, 251), "gray": (147, 120, 100),
}
TEXT = {"uniform": "UNIFORM", "non_uniform": "NON-UNIFORM", "uncertain": "UNCERTAIN"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, lines: list[str], x: int, y: int, color, scale: float, below: bool = False) -> None:
    th = int(22 * scale) + 5
    w = max(cv2.getTextSize(s, FONT, scale, 1)[0][0] for s in lines) + 10
    top = y if below else max(0, y - th * len(lines))
    x = max(0, min(x, img.shape[1] - w))
    cv2.rectangle(img, (x, top), (x + w, top + th * len(lines)), color, -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (x + 5, top + th * (i + 1) - 7), FONT, scale, (15, 12, 10), 1, cv2.LINE_AA)


def draw_people(tile: np.ndarray, people: list, smoothed: bool, k: float = 1.0) -> None:
    """`k` = factor the tile was enlarged by (boxes are in original tile pixels)."""
    scale = max(0.45, min(0.75, tile.shape[1] / 1300))
    thick = max(2, tile.shape[1] // 450)
    for p in people:
        key = p.smoothed_label if smoothed else p.label
        conf = p.smoothed_conf if smoothed else p.conf
        pending = p.extra.get("pending") if smoothed else None
        c = C[key]
        x1, y1, x2, y2 = (int(v * k) for v in p.box)
        cv2.rectangle(tile, (x1, y1), (x2, y2), c, thick)
        if pending:   # votes still being collected: never shown as a verdict
            lab, n, need = pending
            status = f"{'UNIFORM?' if lab == 'uniform' else 'CHECKING'} {n}/{need}"
        else:
            status = TEXT[key] + (f" {conf * 100:.0f}%" if conf > 0 else "")
        _label(tile, [f"Person #{p.track_id:02d}", status], x1, y1, c, scale, below=y1 < 40)


def draw_roi(tile: np.ndarray, poly: np.ndarray | None, k: float = 1.0) -> None:
    if poly is None:
        return
    pts = (poly * k).astype(np.int32)
    ov = tile.copy()
    cv2.fillPoly(ov, [pts], C["cyan"])
    cv2.addWeighted(ov, 0.10, tile, 0.90, 0, dst=tile)
    cv2.polylines(tile, [pts], True, C["cyan"], max(2, tile.shape[1] // 500), cv2.LINE_AA)
    x, y = int(pts[:, 0].min()), int(pts[:, 1].max())
    cv2.putText(tile, "STORE INTERIOR ROI", (x + 6, y - 8), FONT, 0.5, C["cyan"], 1, cv2.LINE_AA)


def tile_scale(tile_w: int, target: int = 1280) -> float:
    """Small DVR tiles are enlarged (max 2x) so labels stay legible in the output."""
    return float(min(2.0, max(1.0, target / max(tile_w, 1))))


def enlarge(tile: np.ndarray, k: float) -> np.ndarray:
    if k == 1.0:
        return tile.copy()
    return cv2.resize(tile, (int(tile.shape[1] * k), int(tile.shape[0] * k)), interpolation=cv2.INTER_LINEAR)


def caption(img: np.ndarray, label: str, whole_frame: bool) -> None:
    """Top-left caption naming the analysed camera area."""
    text = "FULL FRAME - STORE INTERIOR" if whole_frame else f"{label.upper()} - STORE INTERIOR (other cameras ignored)"
    _label(img, [text], 6, 6, C["cyan"], 0.5, below=True)


def draw_status(img: np.ndarray, text: str, color: str, sub: str = "") -> None:
    h, w = img.shape[:2]
    bar = 30 if not sub else 48
    ov = img.copy()
    cv2.rectangle(ov, (0, h - bar), (w, h), (26, 21, 6), -1)
    cv2.addWeighted(ov, 0.82, img, 0.18, 0, dst=img)
    cv2.circle(img, (16, h - bar + 15), 6, C[color], -1, cv2.LINE_AA)
    cv2.putText(img, text, (30, h - bar + 21), FONT, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (30, h - 9), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def fit_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
