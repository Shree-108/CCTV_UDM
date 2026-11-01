"""Candidate models: every training run saves its weights under a NEW versioned folder - never over the
production file (models/uniform_best.pt, or whatever model.path in config/uniform_config.yaml names).

    models/candidates/<name>-<YYYYmmdd-HHMMSS>/uniform_best.pt
    models/candidates/<name>-<YYYYmmdd-HHMMSS>/model_card.json   checksum, classes, preprocessing,
                                                                  thresholds, dataset, training config,
                                                                  code commit, evaluation (when run)

A candidate only becomes active through `python -m training.register_model --activate <folder>`, after
its evaluation has been reviewed. The previous model file stays where it is, so a rollback is the same
command pointing at the previous model.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from backend.core.model_config import INFERENCE, MODEL, ROOT, rel, sha256_of

CANDIDATES = ROOT / "models" / "candidates"
CARD = "model_card.json"


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save_candidate(best: Path, name: str, info: dict) -> Path:
    """Copy `best` into a new candidate folder with its model card. Returns the folder."""
    from ultralytics import YOLO
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = CANDIDATES / f"{name}-{stamp}"
    folder.mkdir(parents=True, exist_ok=False)                       # never reuse / overwrite a candidate
    dest = folder / "uniform_best.pt"
    shutil.copy2(best, dest)
    card = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "candidate",
            "weights": rel(dest), "sha256": sha256_of(dest), "architecture": "Ultralytics YOLO-cls",
            "classes": {int(k): v for k, v in YOLO(str(dest), task="classify").names.items()},
            "preprocessing": {"crop": "upper_body_box (backend/uniform/classifier.py)",
                              "crop_size": list(MODEL["crop_size"]) if MODEL["crop_size"] else None,
                              "imgsz": MODEL["imgsz"]},
            "thresholds_at_training": {"uniform_confidence": INFERENCE["uniform_confidence"],
                                       "non_uniform_confidence": INFERENCE["non_uniform_confidence"]},
            "code_commit": _commit(), **info}
    write_card(folder, card)
    return folder


def read_card(folder: Path) -> dict:
    p = folder / CARD
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_card(folder: Path, card: dict) -> None:
    (folder / CARD).write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
