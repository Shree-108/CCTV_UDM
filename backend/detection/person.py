"""Ultralytics YOLO person detection (COCO class 0), with ByteTrack tracking for video."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PERSON_CLASS = 0


def select_device(pref: str = "auto") -> str:
    """gpu_acceleration setting: auto | on | off."""
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if pref == "off":
        return "cpu"
    if pref == "on" and not cuda:
        return "cpu"   # requested but unavailable: fall back, reported via /api/health
    return "cuda:0" if cuda else "cpu"


def gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@dataclass
class Person:
    track_id: int
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in TILE pixel coordinates
    det_conf: float
    label: str = "uncertain"                 # per-frame: uniform | non_uniform | uncertain
    conf: float = 0.0
    reason: str = ""
    smoothed_label: str = "uncertain"        # after temporal voting
    smoothed_conf: float = 0.0
    extra: dict = field(default_factory=dict)


class PersonDetector:
    def __init__(self, weights: Path, device: str, tracker_cfg: Path | None = None):
        from ultralytics import YOLO
        # missing official weights -> Ultralytics downloads them to this exact path (not the working directory)
        weights.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(weights))
        self.device = device
        self.extra = {"half": True} if device.startswith("cuda") else {}
        self.tracker_cfg = str(tracker_cfg) if tracker_cfg else "bytetrack.yaml"

    def reset(self) -> None:
        self.model.predictor = None   # fresh tracker state -> IDs restart at 1

    @staticmethod
    def _people(res, tracked: bool) -> list[Person]:
        b = res.boxes
        if b is None or len(b) == 0:
            return []
        xyxy, conf = b.xyxy.cpu().numpy(), b.conf.cpu().numpy()
        if tracked:
            if b.id is None:
                return []
            ids = b.id.cpu().numpy().astype(int)
        else:
            ids = np.arange(1, len(xyxy) + 1)
        return [Person(int(i), tuple(map(float, bx)), float(c)) for i, bx, c in zip(ids, xyxy, conf)]

    def track(self, img: np.ndarray, conf: float, imgsz: int) -> list[Person]:
        res = self.model.track(img, persist=True, tracker=self.tracker_cfg, classes=[PERSON_CLASS],
                               conf=conf, imgsz=imgsz, device=self.device, verbose=False, **self.extra)[0]
        return self._people(res, tracked=True)

    def detect(self, img: np.ndarray, conf: float, imgsz: int) -> list[Person]:
        res = self.model.predict(img, classes=[PERSON_CLASS], conf=conf, imgsz=imgsz,
                                 device=self.device, verbose=False, **self.extra)[0]
        return self._people(res, tracked=False)
