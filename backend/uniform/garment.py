"""Garment crop for the uniform classifier (crop_mode "garment"): the SHIRT region located from body keypoints.

Why: the original crop (classifier.upper_body_box: 12-55 % of the person box height) assumes a standing,
full-body box. Behind a counter, seated or bending, the box ends at the waist and that band lands on the head
and the shelves - uniformed staff were scored non_uniform. Here the shirt is located from the person's own
shoulders and hips (YOLO11 pose, models/person/yolo11n-pose.pt), so front, back, side, seated and bending
people all get their shirt - collar, shoulders, sleeves near the torso and the stripes - and not the face.

    garment_region(person_crop, pose) -> (x1, y1, x2, y2, source)   source: pose | fallback
    prepare(crop)                     -> 224 x 224 letterboxed image (aspect ratio kept, grey padding)

The SAME functions cut the training crops (training/udm_dataset.py) and the inference crops
(backend/uniform/classifier.py, crop_mode: garment), so training and inference always match. The original
torso band stays the default (crop_mode: torso_band) for the production v0.2.0 model.
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

SIZE = 224                          # classifier input: square, letterboxed
PAD = (114, 114, 114)
KP_CONF = 0.35                      # keypoint visibility threshold
L_SH, R_SH, L_EL, R_EL, L_HIP, R_HIP = 5, 6, 7, 8, 11, 12
POSE_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "person" / "yolo11n-pose.pt"


class PoseModel:
    """YOLO11n-pose on one person crop at a time (lazy, thread-safe)."""

    def __init__(self, weights: Path = POSE_WEIGHTS, device: str = "cpu"):
        from ultralytics import YOLO
        self.model, self.device = YOLO(str(weights)), device
        self.lock = threading.Lock()

    def keypoints(self, person: np.ndarray) -> np.ndarray | None:
        """(17, 3) x, y, conf in person-crop pixels of the most central / largest skeleton, or None."""
        with self.lock:
            r = self.model.predict(person, imgsz=320, conf=0.25, device=self.device, verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return None
        h, w = person.shape[:2]
        boxes = r.boxes.xyxy.cpu().numpy()
        cx, cy = w / 2, h / 2
        score = [(b[2] - b[0]) * (b[3] - b[1]) - 2 * abs((b[0] + b[2]) / 2 - cx) * h - abs((b[1] + b[3]) / 2 - cy) * w
                 for b in boxes]
        i = int(np.argmax(score))
        xy = r.keypoints.xy.cpu().numpy()[i]
        cf = r.keypoints.conf.cpu().numpy()[i] if r.keypoints.conf is not None else np.ones(len(xy))
        return np.concatenate([xy, cf[:, None]], axis=1)


def _fallback(h: int, w: int) -> tuple[int, int, int, int]:
    """No usable keypoints: a band by box shape. Standing full-body boxes (tall) -> the upper body;
    boxes cut at the waist / seated (squat) -> the lower part of the box, below the head."""
    aspect = h / max(w, 1)
    y1, y2 = (0.12, 0.58) if aspect >= 2.2 else (0.22, 0.95)
    return int(0.04 * w), int(y1 * h), int(0.96 * w), int(y2 * h)


def garment_region(person: np.ndarray, pose: PoseModel | None) -> tuple[int, int, int, int, str]:
    """Shirt box inside a person crop: shoulders -> hips, widened to the arms near the torso."""
    h, w = person.shape[:2]
    kp = pose.keypoints(person) if pose is not None else None
    if kp is not None:
        vis = kp[:, 2] >= KP_CONF
        sh = [kp[i, :2] for i in (L_SH, R_SH) if vis[i]]
        hips = [kp[i, :2] for i in (L_HIP, R_HIP) if vis[i]]
        if sh:
            sh_a = np.array(sh)
            top = float(sh_a[:, 1].min())
            sw = float(np.ptp(sh_a[:, 0])) if len(sh) == 2 else 0.0
            if hips:
                bottom = float(np.mean([p[1] for p in hips]))
                torso = max(bottom - top, 1.0)
            else:                                     # hips hidden (counter / table / frame edge): the box bottom
                torso = max(sw * 1.4, 0.35 * h)
                bottom = min(h, top + torso)
            width = max(sw, 0.55 * torso)
            xs = [p[0] for p in sh] + [p[0] for p in hips] + [kp[i, 0] for i in (L_EL, R_EL) if vis[i]]
            cx = float(np.mean([p[0] for p in sh] + [p[0] for p in hips]))
            x1 = min(min(xs), cx - width / 2) - 0.15 * width
            x2 = max(max(xs), cx + width / 2) + 0.15 * width
            y1 = top - 0.18 * torso                    # the collar sits above the shoulder points
            y2 = bottom + 0.05 * torso
            box = (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))
            if box[2] - box[0] >= 8 and box[3] - box[1] >= 8:
                return (*box, "pose")
    return (*_fallback(h, w), "fallback")


def prepare(crop: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Aspect-ratio-preserving resize into a size x size square (grey padding) - no stretching of the stripes."""
    h, w = crop.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    img = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.full((size, size, 3), PAD, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = img
    return out
