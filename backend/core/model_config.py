"""Uniform model configuration (config/uniform_config.yaml).

Every path in the config is resolved against the project root, so a cloned project
finds its model on any PC. Also identifies the exact weights file in use (version +
SHA-256) so each deployment can show which model it runs.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any

import yaml

from backend.core.config import ROOT

log = logging.getLogger(__name__)

CONFIG_PATH = Path(os.environ.get("SU_UNIFORM_CONFIG", ROOT / "config" / "uniform_config.yaml"))
LABELS = ("uniform", "non_uniform", "uncertain")
DEVICE_TO_GPU = {"auto": "auto", "cpu": "off", "cuda": "on", "gpu": "on"}


def _project_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def rel(p: Path) -> str:
    """Project-relative display path (forward slashes) - never a PC-specific path."""
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.name


def load() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    m, inf = cfg.setdefault("model", {}), cfg.setdefault("inference", {})
    m.setdefault("name", "uniform-cls")
    m["version"] = str(m.get("version", "unversioned"))
    m.setdefault("path", "models/uniform_best.pt")
    m.setdefault("legacy_paths", [])
    m["sha256"] = (m.get("sha256") or "").strip().lower()
    m.setdefault("task", "classify")
    m.setdefault("imgsz", 224)
    cs = m.get("crop_size")
    if cs is not None and not (isinstance(cs, list) and len(cs) == 2 and all(isinstance(v, int) and v > 0 for v in cs)):
        raise ValueError(f"{CONFIG_PATH.name}: model.crop_size must be [width, height] in pixels or null")
    m["crop_size"] = tuple(cs) if cs else None
    m["class_map"] = m.get("class_map") or {k: k for k in LABELS}
    bad = {v for v in m["class_map"].values() if v not in LABELS}
    if bad:
        raise ValueError(f"{CONFIG_PATH.name}: class_map values must be one of {LABELS}, got {sorted(bad)}")
    cfg.setdefault("person_detector", {}).setdefault("path", "models/person/yolo11n.pt")
    if os.environ.get("SU_DEVICE"):
        inf["device"] = os.environ["SU_DEVICE"]
    inf.setdefault("device", "auto")
    if inf["device"] not in DEVICE_TO_GPU:
        raise ValueError(f"{CONFIG_PATH.name}: inference.device must be auto, cpu or cuda")
    inf.setdefault("overall_rule", "any_uniform")
    cap = cfg.setdefault("capture", {})
    cap.setdefault("enabled", True)
    cap.setdefault("store_id", "STORE-001")
    cap.setdefault("interval_minutes", 15)
    cap.setdefault("active_start", "09:00")
    cap.setdefault("active_end", "21:00")
    cap.setdefault("video_step_seconds", 60)
    ext = cfg.setdefault("extraction", {})
    ext.setdefault("default_interval_seconds", 30)
    ext.setdefault("default_strategy", "random_segments")
    ext.setdefault("default_sample_count", 4)
    ext.setdefault("image_format", "png")
    ext.setdefault("max_frames", 5000)
    if inf["overall_rule"] not in ("any_uniform", "all_uniform"):
        raise ValueError(f"{CONFIG_PATH.name}: inference.overall_rule must be any_uniform or all_uniform")
    return cfg


CFG = load()
MODEL = CFG["model"]
INFERENCE = CFG["inference"]
CAPTURE = CFG["capture"]
EXTRACTION = CFG["extraction"]
PERSON_MODEL = _project_path(CFG["person_detector"]["path"])
UNIFORM_MODEL = _project_path(os.environ.get("SU_UNIFORM_MODEL") or MODEL["path"])   # canonical location


def uniform_model_path() -> Path:
    """The weights file to load: the configured path, else a legacy location (warned)."""
    if UNIFORM_MODEL.exists() or os.environ.get("SU_UNIFORM_MODEL"):
        return UNIFORM_MODEL
    for p in map(_project_path, MODEL["legacy_paths"]):
        if p.exists():
            log.warning("Uniform model found at legacy path %s - move it to %s", rel(p), rel(UNIFORM_MODEL))
            return p
    return UNIFORM_MODEL


_sha_cache: dict[tuple[str, float, int], str] = {}
_sha_lock = threading.Lock()


def sha256_of(p: Path) -> str:
    st = p.stat()
    key = (str(p), st.st_mtime, st.st_size)
    with _sha_lock:
        if key not in _sha_cache:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(2 ** 20), b""):
                    h.update(chunk)
            _sha_cache[key] = h.hexdigest()
        return _sha_cache[key]


def model_identity() -> dict[str, Any]:
    """What the dashboard / reports show about the uniform model on this installation."""
    p = uniform_model_path()
    out = {"name": MODEL["name"], "version": MODEL["version"], "status": MODEL.get("status") or None,
           "dataset_version": MODEL.get("dataset_version") or None, "path": rel(p),
           "expected_path": rel(UNIFORM_MODEL), "present": p.exists(), "sha256": None,
           "crop_size": list(MODEL["crop_size"]) if MODEL["crop_size"] else None,
           "expected_sha256": MODEL["sha256"] or None, "verified": None}
    if p.exists():
        out["sha256"] = sha256_of(p)
        if MODEL["sha256"]:
            out["verified"] = out["sha256"] == MODEL["sha256"]
    return out


def version_label(identity: dict | None = None) -> str:
    i = identity or model_identity()
    if not i["present"]:
        return f"no trained model ({i['expected_path']} missing)"
    # the crop shape is part of the label so results made before / after the 2026-10-29 preprocessing fix
    # stay distinguishable in the stored results (same weights, different input)
    parts = [i.get("status"), "crop {}x{}".format(*i["crop_size"]) if i.get("crop_size") else None]
    tag = f" ({', '.join(p for p in parts if p)})" if any(parts) else ""
    return f"{i['name']} v{i['version']}{tag} · sha256 {i['sha256'][:12]}"
