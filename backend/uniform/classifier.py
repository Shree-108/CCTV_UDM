"""Uniform classification on the person's UPPER-BODY crop only (never on background).

Pipeline per person:  person box -> visibility checks -> upper-body crop
                      -> classifier -> uniform / non_uniform / uncertain

Only pixels inside the person's torso region are ever classified, so shelves,
posters, packaging and background patterns cannot count as "uniform".

Backends
  * TrainedClassifier  - Ultralytics YOLO-cls model trained with training/train.py
                         (the production path).
  * StripePatternBaseline - hand-crafted vertical-stripe analysis used only until a
                         trained model exists. It is a stop-gap, not a substitute
                         for training on real CCTV data.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from pathlib import Path as resolve_path

CLASSES = ("uniform", "non_uniform", "uncertain")

# Upper-body region as fractions of the person box (shared with training so
# training crops and production crops look identical).
TORSO_Y = (0.12, 0.55)
TORSO_X = (0.10, 0.90)


def upper_body_box(box: Sequence[float]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (int(x1 + TORSO_X[0] * w), int(y1 + TORSO_Y[0] * h),
            int(x1 + TORSO_X[1] * w), int(y1 + TORSO_Y[1] * h))


def upper_body_crop(frame: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = upper_body_box(box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


def _overlap_fraction(a: Sequence[float], b: Sequence[float]) -> float:
    """Fraction of box a covered by box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def visibility_issue(box: Sequence[float], others: list[Sequence[float]],
                     frame_shape: tuple, min_height: int) -> str | None:
    """Return a reason string if the upper body is not reliably visible."""
    H, W = frame_shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if h < min_height:
        return "too small / far"
    if y1 <= 2 and h < 0.6 * H:
        return "upper body cut off"
    if h / max(w, 1) < 0.9:
        return "partially visible"
    torso = upper_body_box(box)
    for o in others:
        # a person whose feet are lower in the image is standing in front
        if o[3] > y2 and _overlap_fraction(torso, o) > 0.35:
            return "occluded by another person"
    return None


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class TrainedClassifier:
    name = "trained"

    def __init__(self, path: str, device: str):
        from ultralytics import YOLO
        self.model = YOLO(path, task="classify")
        self.device = device
        self.names = {int(k): v for k, v in self.model.names.items()}

    def predict(self, crops: list[np.ndarray]) -> list[dict[str, float]]:
        if not crops:
            return []
        results = self.model.predict(crops, device=self.device, verbose=False, imgsz=224)
        out = []
        for r in results:
            probs = r.probs.data.cpu().numpy()
            out.append({self.names[i]: float(p) for i, p in enumerate(probs)})
        return out


class StripePatternBaseline:
    """Scores narrow, regular, VERTICAL, low-saturation (black/white) stripes that
    run coherently down the torso. Plain black/white, horizontal stripes, checks,
    prints and coloured shirts all score low. Accuracy on distant/low-res CCTV is
    limited - train the real classifier."""

    name = "stripe-baseline"
    SIZE = 96

    def features(self, crop: np.ndarray) -> dict[str, float]:
        img = cv2.resize(crop, (self.SIZE, self.SIZE), interpolation=cv2.INTER_AREA)
        sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32) / 255.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.equalizeHist(gray.astype(np.uint8)).astype(np.float32)
        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).mean()
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).mean()

        def periodicity(profiles: np.ndarray) -> float:
            # share of AC power held by the 3 strongest stripe-band frequencies
            # (3..40 cycles per crop): ~1 for regular stripes, low for noise/prints
            p = profiles - profiles.mean(axis=1, keepdims=True)
            spec = np.abs(np.fft.rfft(p, axis=1)) ** 2
            total = spec[:, 1:].sum(axis=1) + 1e-6
            top3 = np.sort(spec[:, 3:41], axis=1)[:, -3:].sum(axis=1)
            return float((top3 / total).mean())

        rows, cols = gray, gray.T
        # high-pass across x so shading / folds / lighting gradients don't count
        hp = gray - cv2.blur(gray, (9, 1))
        col_profile = hp.mean(axis=0)
        rc = hp - hp.mean(axis=1, keepdims=True)
        cp = col_profile - col_profile.mean()
        denom = np.linalg.norm(rc, axis=1) * np.linalg.norm(cp) + 1e-6
        coherence = float(np.clip((rc @ cp) / denom, -1, 1).mean())  # same stripes all the way down?
        return {
            "saturation": float(np.median(sat)),
            "vertical_ratio": float(gx / (gx + gy + 1e-6)),
            "edge_energy": float((gx + gy) / 255.0),
            "vertical_periodicity": periodicity(rows),
            "horizontal_periodicity": periodicity(cols),
            "coherence": coherence,
            "contrast": float(gray.std() / 128.0),
        }

    def score(self, f: dict[str, float]) -> float:
        # Regular periodicity across the torso is the key signal: lapels, ties,
        # zips and folds give vertical edges but not a regular stripe spectrum.
        s = (6.0 * (f["vertical_ratio"] - 0.62)
             - 20.0 * max(0.0, 0.55 - f["vertical_ratio"])   # edges must be mostly vertical
             + 16.0 * (f["vertical_periodicity"] - 0.50)
             - 15.0 * max(0.0, f["horizontal_periodicity"] - 0.62)
             + 2.0 * (f["coherence"] - 0.20)
             - 25.0 * max(0.0, f["saturation"] - 0.28)
             - 25.0 * max(0.0, 0.10 - f["edge_energy"])
             - 6.0 * max(0.0, 0.25 - f["contrast"]))
        return float(1.0 / (1.0 + np.exp(-s)))

    def predict(self, crops: list[np.ndarray]) -> list[dict[str, float]]:
        out = []
        for c in crops:
            p = self.score(self.features(c))
            out.append({"uniform": p, "non_uniform": 1.0 - p, "uncertain": 0.0})
        return out


# --------------------------------------------------------------------------- #
class UniformDetector:
    MIN_CROP_W = 24  # narrower torsos cannot resolve stripes reliably

    def __init__(self, model_path: str | None, device: str = "cpu", untrained_mode: str = "baseline"):
        path = resolve_path(model_path) if model_path else None
        self.trained = path is not None and path.exists()
        self.untrained_mode = untrained_mode
        if self.trained:
            self.backend = TrainedClassifier(str(path), device)
        elif untrained_mode == "baseline":
            self.backend = StripePatternBaseline()
        else:
            self.backend = None   # "disabled": no shirt is judged, everyone is UNCERTAIN

    @property
    def backend_name(self) -> str:
        return self.backend.name if self.backend else "not-trained"

    @property
    def status(self) -> dict:
        if self.trained:
            return {"state": "ready", "label": "Trained classifier", "backend": "trained"}
        if self.backend:
            return {"state": "not_trained", "label": "MODEL NOT TRAINED — stripe-pattern baseline (indicative only)",
                    "backend": "stripe-baseline"}
        return {"state": "not_trained", "label": "MODEL NOT TRAINED — uniform classification disabled",
                "backend": "not-trained"}

    BLUR_MIN = 6.0   # Laplacian variance of the torso crop below this -> too blurred

    def classify(self, tile: np.ndarray, people: list, uniform_conf: float,
                 non_uniform_conf: float, min_height: int, occluders: list | None = None) -> None:
        """Set label / conf / reason on each Person (all inside the ROI).
        `occluders` = every detected box in the tile (incl. outside the ROI), used
        to spot people hidden behind someone else."""
        boxes = occluders if occluders is not None else [p.box for p in people]
        if self.backend is None:
            for p in people:
                p.label, p.conf, p.reason = "uncertain", 0.0, "uniform model not trained"
            return
        crops, targets = [], []
        for p in people:
            others = [b for b in boxes if b is not p.box and tuple(b) != tuple(p.box)]
            issue = visibility_issue(p.box, others, tile.shape, min_height)
            crop = None if issue else upper_body_crop(tile, p.box)
            if crop is not None and crop.shape[1] < self.MIN_CROP_W:
                issue, crop = "low resolution", None
            if crop is not None:
                g = cv2.cvtColor(cv2.resize(crop, (96, 96)), cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(g, cv2.CV_32F).var() < self.BLUR_MIN:
                    issue, crop = "too blurred", None
            if crop is None:
                p.label, p.conf, p.reason = "uncertain", 0.0, issue or "no crop"
                continue
            crops.append(crop)
            targets.append(p)

        for p, probs in zip(targets, self.backend.predict(crops)):
            pu = probs.get("uniform", 0.0)
            pn = probs.get("non_uniform", max(0.0, 1.0 - pu - probs.get("uncertain", 0.0)))
            p.extra["uniform_prob"] = pu
            if pu >= uniform_conf:
                p.label, p.conf = "uniform", pu
            elif pn >= non_uniform_conf:
                p.label, p.conf = "non_uniform", pn
            else:
                p.label, p.conf, p.reason = "uncertain", max(pu, pn), "low confidence"
