"""Uniform classification on the person's UPPER-BODY crop only (never on background).

Pipeline per person:  person box -> visibility checks -> upper-body crop
                      -> classifier -> uniform / non_uniform / uncertain

Only pixels inside the person's torso region are ever classified, so shelves,
posters, packaging and background patterns cannot count as "uniform".

Backends (model.backend in config/uniform_config.yaml)
  * TrainedClassifier  - yolo_cls: Ultralytics YOLO-cls model trained with training/train.py
                         (the production path).
  * TorchvisionClassifier - torchvision: EfficientNet-B0 / MobileNetV3 checkpoint trained with
                         training/train_torchvision.py (Phase B candidates; same crop, same class_map).
  * StripePatternBaseline - hand-crafted vertical-stripe analysis used only until a
                         trained model exists. It is a stop-gap, not a substitute
                         for training on real CCTV data.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from backend.core.model_config import INFERENCE, MODEL, model_identity, rel, version_label

log = logging.getLogger(__name__)

CLASSES = ("uniform", "non_uniform", "uncertain")

# Upper-body region as fractions of the person box (shared with training so
# training crops and production crops look identical).
TORSO_Y = (0.12, 0.55)
TORSO_X = (0.10, 0.90)
# squatter person boxes: the torso band would land on the head (config inference.min_box_aspect, 0 = off)
MIN_BOX_ASPECT = float(INFERENCE.get("min_box_aspect", 0) or 0)


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
    if h / max(w, 1) < MIN_BOX_ASPECT:
        return "seated / leaning / cut off - shirt position unclear"
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
        self.model = YOLO(path, task=MODEL["task"])
        # check the real network, not the task name (stripped checkpoints carry no task metadata)
        arch = type(self.model.model).__name__
        if arch != "ClassificationModel":
            raise ValueError(f"{Path(path).name} contains a {arch}, but the uniform model must be an Ultralytics "
                             f"classification model (YOLO-cls, trained with training/train.py)")
        if self.model.task != "classify":   # .pt without train_args: Ultralytics mis-guesses the task
            log.warning("%s has no task metadata - using it as a classifier", Path(path).name)
            self.model.task = "classify"
        self.device = device
        self.imgsz = int(MODEL["imgsz"])
        cmap = MODEL["class_map"]
        names = {int(k): v for k, v in self.model.names.items()}
        unknown = sorted(set(names.values()) - set(cmap))
        if unknown:
            raise ValueError(f"model classes {unknown} are not in class_map of config/uniform_config.yaml")
        if "uniform" not in {cmap[n] for n in names.values()}:
            raise ValueError("no model class maps to 'uniform' in config/uniform_config.yaml class_map")
        self.labels = {i: cmap[n] for i, n in names.items()}   # model index -> app label
        self.crop_size = MODEL.get("crop_size")                # the crop shape the weights were trained on

    def predict(self, crops: list[np.ndarray]) -> list[dict[str, float]]:
        if not crops:
            return []
        if self.crop_size:
            crops = [cv2.resize(c, self.crop_size, interpolation=cv2.INTER_AREA) for c in crops]
        results = self.model.predict(crops, device=self.device, verbose=False, imgsz=self.imgsz)
        out = []
        for r in results:
            probs = r.probs.data.cpu().numpy()
            d: dict[str, float] = {}
            for i, p in enumerate(probs):
                d[self.labels[i]] = d.get(self.labels[i], 0.0) + float(p)
            out.append(d)
        return out


class TorchvisionClassifier:
    """EfficientNet-B0 / MobileNetV3 candidate (backend/uniform/torch_backbones.py checkpoint)."""
    name = "trained"

    def __init__(self, path: str, device: str):
        from backend.uniform import torch_backbones as tb
        self.model, self.meta = tb.load(path, device)
        self.device = device
        cmap = MODEL["class_map"]
        unknown = sorted(set(self.meta["classes"]) - set(cmap))
        if unknown:
            raise ValueError(f"model classes {unknown} are not in class_map of config/uniform_config.yaml")
        self.labels = {i: cmap[n] for i, n in enumerate(self.meta["classes"])}
        if "uniform" not in self.labels.values():
            raise ValueError("no model class maps to 'uniform' in config/uniform_config.yaml class_map")
        self.crop_size = MODEL.get("crop_size")

    def predict(self, crops: list[np.ndarray]) -> list[dict[str, float]]:
        if not crops:
            return []
        import torch
        from backend.uniform import torch_backbones as tb
        if self.crop_size:
            crops = [cv2.resize(c, self.crop_size, interpolation=cv2.INTER_AREA) for c in crops]
        x = tb.to_tensor(crops, self.meta["input_size"], self.meta["mean"], self.meta["std"]).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(x), dim=1).cpu().numpy()
        out = []
        for row in probs:
            d: dict[str, float] = {}
            for i, p in enumerate(row):
                d[self.labels[i]] = d.get(self.labels[i], 0.0) + float(p)
            out.append(d)
        return out


def load_backend(path: str, device: str):
    """The classifier named by model.backend (yolo_cls | torchvision)."""
    kind = MODEL.get("backend") or "yolo_cls"
    if kind == "torchvision":
        return TorchvisionClassifier(path, device)
    return TrainedClassifier(path, device)


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

    def __init__(self, model_path: str | Path | None, device: str = "cpu", untrained_mode: str = "baseline"):
        path = Path(model_path) if model_path else None
        self.trained = path is not None and path.exists()
        self.untrained_mode = untrained_mode
        self.identity = model_identity() if path is not None else None
        if self.trained:
            self.backend = load_backend(str(path), device)
            log.info("Uniform model loaded: %s (%s) on %s", rel(path), version_label(self.identity), device)
            if self.identity and self.identity["verified"] is False:
                log.error("Uniform model checksum MISMATCH: %s is not the approved weights (expected sha256 %s)",
                          rel(path), self.identity["expected_sha256"])
        else:
            log.warning("Trained uniform model NOT FOUND at %s - using %s", rel(path) if path else "-",
                        "stripe-pattern baseline" if untrained_mode == "baseline" else "no classifier (all UNCERTAIN)")
            # "disabled": no shirt is judged, everyone is UNCERTAIN
            self.backend = StripePatternBaseline() if untrained_mode == "baseline" else None

    @property
    def backend_name(self) -> str:
        return self.backend.name if self.backend else "not-trained"

    @property
    def model_version(self) -> str:
        if self.trained and self.identity:
            return version_label(self.identity)
        return f"{self.backend_name} (trained model missing)"

    @property
    def status(self) -> dict:
        ident = self.identity or model_identity()
        base = {"model": ident, "version": self.model_version}
        if self.trained:
            if ident["verified"] is False:
                return base | {"state": "mismatch", "backend": "trained",
                               "label": f"WRONG MODEL FILE — {ident['path']} does not match the approved checksum"}
            return base | {"state": "ready", "label": f"Trained classifier {ident['name']} v{ident['version']}",
                           "backend": "trained"}
        missing = f"MODEL FILE MISSING — restore {ident['expected_path']}"
        if self.backend:
            return base | {"state": "not_trained", "label": f"{missing} (stripe baseline, indicative only)",
                           "backend": "stripe-baseline"}
        return base | {"state": "not_trained", "label": f"{missing} (uniform classification disabled)",
                       "backend": "not-trained"}

    BLUR_MIN = 6.0   # Laplacian variance of the torso crop below this -> too blurred

    def _garment_crop(self, tile: np.ndarray, box, others: list, min_height: int) -> tuple[np.ndarray | None, str | None]:
        """crop_mode garment: the shirt from body keypoints, letterboxed. Seated / leaning / bending people are
        classified when their shirt is visible; only too small, occluded, hidden or blurred shirts are not judged."""
        from backend.uniform.garment import PoseModel, garment_region, prepare
        x1, y1, x2, y2 = box
        H, W = tile.shape[:2]
        if y2 - y1 < min_height:
            return None, "too small / far"
        torso = upper_body_box(box)
        for o in others:
            if o[3] > y2 and _overlap_fraction(torso, o) > 0.35:
                return None, "occluded by another person"
        mx, my = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
        a, b = max(0, int(x1 - mx)), max(0, int(y1 - my))
        c, d = min(W, int(x2 + mx)), min(H, int(y2 + my))
        person = tile[b:d, a:c]
        if person.size == 0:
            return None, "no crop"
        if getattr(self, "_pose", None) is None:
            self._pose = PoseModel(device=getattr(self.backend, "device", "cpu"))
        gx1, gy1, gx2, gy2, _src = garment_region(person, self._pose)
        crop = person[gy1:gy2, gx1:gx2]
        if crop.size == 0 or crop.shape[1] < self.MIN_CROP_W or crop.shape[0] < self.MIN_CROP_W:
            return None, "shirt not visible / low resolution"
        g = cv2.cvtColor(cv2.resize(crop, (96, 96)), cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(g, cv2.CV_32F).var() < self.BLUR_MIN:
            return None, "too blurred"
        return prepare(crop), None

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
        garment = MODEL.get("crop_mode") == "garment"
        for p in people:
            others = [b for b in boxes if b is not p.box and tuple(b) != tuple(p.box)]
            if garment:
                crop, issue = self._garment_crop(tile, p.box, others, min_height)
                if crop is None:
                    p.label, p.conf, p.reason = "uncertain", 0.0, issue or "no crop"
                    continue
                crops.append(crop)
                targets.append(p)
                continue
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
            p.extra["probs"] = probs              # raw classifier output (training/diagnose.py)
            if pu >= uniform_conf:
                p.label, p.conf = "uniform", pu
            elif pn >= non_uniform_conf:
                p.label, p.conf = "non_uniform", pn
            else:
                p.label, p.conf, p.reason = "uncertain", max(pu, pn), "low confidence"
