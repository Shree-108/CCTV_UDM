"""Static paths and environment configuration."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[2]
if load_dotenv:
    load_dotenv(ROOT / ".env")

STORAGE = Path(os.environ.get("SU_STORAGE_DIR", ROOT / "storage"))
UPLOADS = STORAGE / "uploads"
ANALYSES = STORAGE / "analyses"        # one folder per analysis: annotated media, evidence, reports
MODELS = Path(os.environ.get("SU_MODELS_DIR", ROOT / "models"))
# Model weight paths + inference defaults live in config/uniform_config.yaml (backend/core/model_config.py)
UNIFORM_EVAL = MODELS / "uniform" / "eval" / "report.json"
TRACKER_CFG = ROOT / "backend" / "tracking" / "bytetrack_store.yaml"
DB_PATH = Path(os.environ.get("SU_DB_PATH", STORAGE / "sightline_uniform.db"))
SETTINGS_PATH = STORAGE / "settings.json"

CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3010,http://127.0.0.1:3010").split(",") if o.strip()]
MAX_IMAGE_MB = int(os.environ.get("MAX_IMAGE_UPLOAD_MB", "25"))
MAX_VIDEO_MB = int(os.environ.get("MAX_VIDEO_UPLOAD_MB", "1024"))

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".ts"}

for d in (UPLOADS, ANALYSES, MODELS / "person"):
    d.mkdir(parents=True, exist_ok=True)
