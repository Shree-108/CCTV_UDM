"""Create and run one analysis - shared by the upload API and the capture service, so an
uploaded image and a scheduled capture go through exactly the same pipeline."""
from __future__ import annotations

import uuid

from backend.analysis.jobs import JobRunner
from backend.analysis.pipeline import Models, analyze_image
from backend.core.settings import Settings
from backend.database.db import DB, now_iso
from backend.detection.grid import AnalysisArea, validate_area


class AnalysisFailed(RuntimeError):
    def __init__(self, aid: str, message: str):
        super().__init__(message)
        self.aid = aid


def full_frame_area(label: str = "Full frame") -> dict:
    return {"rect": [0.0, 0.0, 1.0, 1.0], "roi": None, "label": label, "layout": "1x1", "cell": 0}


def start_analysis(db: DB, settings: Settings, models: Models, jobs: JobRunner, media: dict, area: dict,
                   extra: dict | None = None) -> str:
    """Images are analysed synchronously (raises AnalysisFailed, the failed row is kept);
    videos are queued. `extra` adds traceability fields (source, store_id, camera_id, ...).
    Raises ValueError for an invalid area."""
    validate_area(area["rect"], area.get("roi"))
    cfg = settings.get()
    aa = AnalysisArea(rect_norm=area["rect"], roi=area.get("roi"), label=area.get("label") or "Camera",
                      reference_point=cfg["roi_reference_point"])
    aid = f"{now_iso()[:10].replace('-', '')}-{uuid.uuid4().hex[:8]}"
    db.add_analysis({"id": aid, "created_at": now_iso(), "media_id": media["id"], "kind": media["kind"],
                     "filename": media["filename"], "state": "queued", "area": area, "settings": cfg,
                     "duration_sec": media["duration_sec"], "frames_total": media["frames"],
                     "source": "upload", **(extra or {})})
    if media["kind"] == "image":
        db.update_analysis(aid, state="running")
        try:
            analyze_image(db, aid, media, aa, cfg, models.get(cfg))
        except Exception as e:  # noqa: BLE001 - recorded on the analysis, never loses the input
            db.update_analysis(aid, state="failed", error=str(e))
            raise AnalysisFailed(aid, str(e)) from e
    else:
        jobs.submit(aid, media, aa, cfg)
    return aid
