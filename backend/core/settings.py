"""Runtime settings (editable from the Settings page, persisted to storage/settings.json)."""
from __future__ import annotations

import copy
import json
import threading
from typing import Any

from backend.core.config import SETTINGS_PATH
from backend.core.model_config import DEVICE_TO_GPU, INFERENCE as D

# key -> (default, type, min, max, advanced, description)
# Model-related defaults come from config/uniform_config.yaml (inference:).
SPEC: dict[str, tuple] = {
    # ------------------------------------------------ basic
    "person_confidence": (float(D.get("person_confidence", 0.40)), float, 0.05, 0.95, False, "Minimum confidence for a person detection"),
    "uniform_confidence": (float(D.get("uniform_confidence", 0.70)), float, 0.50, 0.99, False, "Minimum probability to call a shirt UNIFORM; weaker results become UNCERTAIN"),
    "analysis_fps": (5.0, float, 0.5, 30.0, False, "Video frames analysed per second of footage"),
    "inference_size": (int(D.get("inference_size", 640)), int, 320, 1920, False, "Resolution the selected camera tile is resized to for detection"),
    "temporal_confirmation_frames": (3, int, 1, 15, False, "Uniform votes needed before a tracked person is confirmed as UNIFORM"),
    "gpu_acceleration": (DEVICE_TO_GPU[D["device"]], str, None, None, False, "auto | on | off"),
    "output_video_quality": ("medium", str, None, None, False, "low | medium | high"),
    "absence_alert_seconds": (30.0, float, 3.0, 3600.0, False, "Seconds with people inside but no confirmed uniform before a NO UNIFORM alert"),
    # ------------------------------------------------ advanced
    "non_uniform_confidence": (float(D.get("non_uniform_confidence", 0.60)), float, 0.50, 0.99, True, "Minimum probability to call a shirt NON-UNIFORM"),
    "frame_skip": (0, int, 0, 120, True, "Analyse every Nth frame (0 = derive from analysis FPS)"),
    "vote_window": (5, int, 1, 30, True, "Recent classifications remembered per tracked person"),
    "presence_gap_tolerance": (3.0, float, 0.0, 30.0, True, "Seconds a confirmed uniform may be hidden (occlusion) without breaking presence"),
    "min_track_observations": (3, int, 1, 50, True, "Tracks seen fewer times are treated as detector noise"),
    "min_person_height_px": (int(D.get("min_person_height_px", 50)), int, 10, 1000, True, "People shorter than this (tile pixels) are UNCERTAIN"),
    "roi_reference_point": ("bottom_center", str, None, None, True, "bottom_center | center | upper_center"),
    "untrained_model_mode": (D.get("untrained_model_mode", "baseline"), str, None, None, True,
                             "When no trained uniform model exists: baseline (stripe heuristic, indicative) | disabled (everyone UNCERTAIN)"),
    "low_confidence_alerts": (True, bool, None, None, True, "Raise LOW CONFIDENCE alerts for persistently uncertain people in videos"),
}
CHOICES = {
    "gpu_acceleration": ("auto", "on", "off"),
    "output_video_quality": ("low", "medium", "high"),
    "roi_reference_point": ("bottom_center", "center", "upper_center"),
    "untrained_model_mode": ("baseline", "disabled"),
}
CRF = {"low": 32, "medium": 26, "high": 20}


class Settings:
    def __init__(self):
        self._lock = threading.RLock()
        self._data = {k: v[0] for k, v in SPEC.items()}
        self._roi_preset: dict | None = None
        if SETTINGS_PATH.exists():
            try:
                raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                self._data.update(self._clean(raw.get("settings", {})))
                self._roi_preset = raw.get("roi_preset")
            except (ValueError, OSError):
                pass

    @staticmethod
    def _clean(changes: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in changes.items():
            if k not in SPEC:
                continue
            default, typ, lo, hi, _, _ = SPEC[k]
            if typ is bool:
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            else:
                try:
                    v = typ(v)
                except (TypeError, ValueError):
                    raise ValueError(f"{k}: invalid value {v!r}")
            if lo is not None and not (lo <= v <= hi):
                raise ValueError(f"{k} must be between {lo} and {hi}")
            if k in CHOICES and v not in CHOICES[k]:
                raise ValueError(f"{k} must be one of {', '.join(CHOICES[k])}")
            out[k] = v
        return out

    def get(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        clean = self._clean(changes)
        with self._lock:
            merged = {**self._data, **clean}
            if merged["non_uniform_confidence"] > 0.99 or merged["uniform_confidence"] < 0.5:
                raise ValueError("invalid thresholds")
            self._data = merged
            self._save()
            return copy.deepcopy(self._data)

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._data = {k: v[0] for k, v in SPEC.items()}
            self._save()
            return copy.deepcopy(self._data)

    @property
    def roi_preset(self) -> dict | None:
        with self._lock:
            return copy.deepcopy(self._roi_preset)

    def save_roi_preset(self, preset: dict | None) -> None:
        with self._lock:
            self._roi_preset = preset
            self._save()

    def _save(self) -> None:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps({"settings": self._data, "roi_preset": self._roi_preset}, indent=2),
                                 encoding="utf-8")

    @staticmethod
    def schema() -> list[dict]:
        return [{"key": k, "default": v[0], "type": v[1].__name__, "min": v[2], "max": v[3], "advanced": v[4],
                 "description": v[5], "choices": list(CHOICES.get(k, ())) or None} for k, v in SPEC.items()]
