"""REST API for the Sightline Uniform Compliance dashboard."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.analysis.compliance import TITLE
from backend.analysis.media import kind_of, new_upload_path, probe
from backend.analysis.runner import AnalysisFailed, start_analysis
from backend.core.config import MAX_IMAGE_MB, MAX_VIDEO_MB, UNIFORM_EVAL, UPLOADS
from backend.core.model_config import INFERENCE, MODEL, PERSON_MODEL, model_identity, rel
from backend.core.settings import Settings
from backend.database.db import now_iso
from backend.detection.grid import LAYOUTS, grid_cells, validate_area
from backend.detection.person import gpu_available, select_device
from backend.capture import cctv_adapter
from backend.extraction.results import storage_status

router = APIRouter(prefix="/api")


def st(request: Request):
    return request.app.state.app_state


def _404(what: str):
    raise HTTPException(404, f"{what} not found")


# ------------------------------------------------------------------ health / model status
@router.get("/health")
def health(request: Request):
    s = st(request)
    cfg = s.settings.get()
    if s.models.uniform:
        uniform = s.models.uniform.status
    else:   # models failed to load: describe the file without loading it
        ident = model_identity()
        uniform = {"state": "not_trained", "backend": "not-loaded", "model": ident, "version": "not loaded",
                   "label": "Uniform model could not be loaded" if ident["present"]
                   else f"MODEL FILE MISSING — restore {ident['expected_path']}"}
    uniform |= {"classes": sorted(set(MODEL["class_map"].values())), "overall_rule": INFERENCE["overall_rule"],
                "uniform_confidence": cfg["uniform_confidence"], "non_uniform_confidence": cfg["non_uniform_confidence"],
                "config": "config/uniform_config.yaml"}
    evaluation = None
    if UNIFORM_EVAL.exists():
        import json
        try:
            evaluation = json.loads(UNIFORM_EVAL.read_text(encoding="utf-8"))
        except ValueError:
            evaluation = None
    return {
        "status": "ok" if not s.models.error else "degraded",
        "model_error": s.models.error,
        "person_detector": {"state": "ready" if PERSON_MODEL.exists() or s.models.tracker_detector else "missing",
                            "model": PERSON_MODEL.name, "path": rel(PERSON_MODEL)},
        "uniform_classifier": uniform,
        "tracking": {"state": "ready", "tracker": "ByteTrack"},
        "gpu": {"available": gpu_available(), "setting": cfg["gpu_acceleration"],
                "device": select_device(cfg["gpu_acceleration"])},
        "evaluation": evaluation,
        "queue": s.jobs.q.qsize(),
        "live_cctv": "not_implemented",
        "cctv_adapter": cctv_adapter.status(),
        "storage": storage_status(),
    }


# ------------------------------------------------------------------ settings
@router.get("/settings")
def get_settings(request: Request):
    s = st(request).settings
    return {"settings": s.get(), "schema": Settings.schema(), "roi_preset": s.roi_preset}


@router.put("/settings")
def put_settings(request: Request, body: dict):
    try:
        return {"settings": st(request).settings.update(body)}
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.post("/settings/reset")
def reset_settings(request: Request):
    return {"settings": st(request).settings.reset()}


class AreaIn(BaseModel):
    rect: list[float] = Field(description="Selected camera tile, normalised [x1,y1,x2,y2] of the full frame")
    roi: Optional[list[list[float]]] = Field(default=None, description="Polygon inside the tile, normalised")
    label: str = "Camera 1"
    layout: str = "1x1"
    cell: Optional[int] = None


@router.put("/roi-preset")
def save_roi_preset(request: Request, area: AreaIn):
    try:
        validate_area(area.rect, area.roi)
    except ValueError as e:
        raise HTTPException(422, str(e))
    st(request).settings.save_roi_preset(area.model_dump())
    return {"ok": True, "roi_preset": area.model_dump()}


@router.delete("/roi-preset")
def clear_roi_preset(request: Request):
    st(request).settings.save_roi_preset(None)
    return {"ok": True}


@router.get("/layouts")
def layouts():
    return [{"id": k, "label": v[0], "rows": v[1], "cols": v[2], "cells": grid_cells(v[1], v[2])} for k, v in LAYOUTS.items()]


# ------------------------------------------------------------------ media upload
async def receive(file: UploadFile, path, kind: str) -> tuple[int, str]:
    """Stream an upload to `path` within the size limit for its kind; returns (size, sha256 hex).
    Nothing half-written is kept on failure."""
    limit = (MAX_IMAGE_MB if kind == "image" else MAX_VIDEO_MB) * 2 ** 20
    size = 0
    digest = hashlib.sha256()
    try:
        with open(path, "wb") as f:
            while chunk := await file.read(2 ** 20):
                size += len(chunk)
                if size > limit:
                    f.close()
                    path.unlink(missing_ok=True)
                    raise HTTPException(413, f"File too large (max {limit // 2 ** 20} MB for {kind}s)")
                f.write(chunk)
                digest.update(chunk)
    except OSError as e:                         # disk full, permissions
        path.unlink(missing_ok=True)
        raise HTTPException(507, f"The file could not be saved on the server ({e.strerror or e.__class__.__name__})")
    return size, digest.hexdigest()


@router.post("/media")
async def upload_media(request: Request, file: UploadFile = File(...)):
    kind = kind_of(file.filename or "")
    if not kind:
        raise HTTPException(415, "Unsupported file type. Use JPG, PNG, BMP, WEBP, MP4, AVI, MOV, MKV, M4V or WEBM.")
    path = new_upload_path(file.filename)
    size, digest = await receive(file, path, kind)
    try:
        meta = probe(path, file.filename, kind, size)
    except ValueError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(422, str(e))
    db = st(request).db
    meta["sha256"] = digest
    db.add_media(meta)
    return _media_out(meta) | {"previously_received": _previously_received(db, meta)}


def _previously_received(db, meta: dict) -> list[dict]:
    """Earlier deliveries of the SAME file (same SHA-256) and their analysis jobs. The new upload is kept
    as its own record (nothing is merged or deleted); the user decides whether to analyse it again."""
    out = []
    for m in db.media_by_sha256(meta["sha256"], exclude=meta["id"]):
        jobs = [{"job_id": j["id"], "state": j["state"], "created_at": j["created_at"]}
                for j in db.extractions(media_id=m["id"], limit=20)]
        out.append({"media_id": m["id"], "filename": m["filename"], "received_at": m["created_at"], "jobs": jobs})
    return out


def _media_out(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "path"} | {"preview_url": f"/api/media/{m['id']}/preview.jpg",
                                                         "file_url": f"/api/media/{m['id']}/file"}


@router.get("/media/{mid}")
def get_media(request: Request, mid: str):
    db = st(request).db
    m = db.media(mid) or _404("Media")
    return _media_out(m) | {"previously_received": _previously_received(db, m) if m.get("sha256") else []}


@router.get("/media/{mid}/preview.jpg")
def media_preview(mid: str):
    p = UPLOADS / f"{Path(mid).name}_preview.jpg"
    return FileResponse(p) if p.exists() else _404("Preview")


@router.get("/media/{mid}/file")
def media_file(request: Request, mid: str):
    m = st(request).db.media(mid) or _404("Media")
    return FileResponse(m["path"], filename=m["filename"])


# ------------------------------------------------------------------ analyses
class AnalysisIn(BaseModel):
    media_id: str
    area: AreaIn


@router.post("/analyses")
def create_analysis(request: Request, body: AnalysisIn):
    s = st(request)
    media = s.db.media(body.media_id) or _404("Media")
    try:
        aid = start_analysis(s.db, s.settings, s.models, s.jobs, media, body.area.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))
    except AnalysisFailed as e:
        raise HTTPException(500, f"Analysis failed: {e}")
    return analysis_detail(request, aid)


def _analysis_out(a: dict) -> dict:
    files = a.get("files") or {}
    a = {k: v for k, v in a.items() if k not in ("files",)}
    a["status_title"] = TITLE.get(a.get("status") or "", None)
    a["files"] = {k: f"/api/analyses/{a['id']}/files/{k}" for k, v in files.items() if v}
    a["preview_url"] = f"/api/media/{a['media_id']}/preview.jpg"
    return a


@router.get("/analyses")
def list_analyses(request: Request, limit: int = 50, offset: int = 0, kind: Optional[str] = None,
                  status: Optional[str] = None, q: Optional[str] = None, camera_id: Optional[str] = None,
                  store_id: Optional[str] = None, source: Optional[str] = None):
    rows, total = st(request).db.analyses(min(limit, 500), offset, kind, status, q, camera_id=camera_id,
                                          store_id=store_id, source=source)
    return {"analyses": [_analysis_out(r) for r in rows], "total": total}


@router.get("/analyses/{aid}")
def analysis_detail(request: Request, aid: str):
    s = st(request)
    a = s.db.analysis(aid) or _404("Analysis")
    out = _analysis_out(a)
    lv = s.jobs.live.get(aid)
    if a["state"] in ("queued", "running") and lv and lv.snapshot:
        out["live"] = lv.snapshot
    out["live_frame_url"] = f"/api/analyses/{aid}/live.jpg" if a["state"] == "running" else None
    out["persons"] = s.db.persons(aid)
    alerts, _ = s.db.alerts(limit=100, analysis_id=aid)
    out["alert_list"] = [_alert_out(x) for x in alerts]
    return out


@router.get("/analyses/{aid}/live.jpg")
def live_frame(request: Request, aid: str):
    lv = st(request).jobs.live.get(aid)
    if not lv or not lv.jpeg:
        raise HTTPException(404, "No live frame")
    return Response(lv.jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/analyses/{aid}/cancel")
def cancel_analysis(request: Request, aid: str):
    s = st(request)
    a = s.db.analysis(aid) or _404("Analysis")
    if a["state"] not in ("queued", "running"):
        raise HTTPException(409, "Analysis is not running")
    s.jobs.cancel(aid)
    return {"ok": True}


DOWNLOAD_NAMES = {"annotated": "annotated", "keyframe": "annotated_frame", "report_json": "report",
                  "report_csv": "people", "report_txt": "report"}


@router.get("/analyses/{aid}/files/{name}")
def analysis_file(request: Request, aid: str, name: str, download: bool = False):
    a = st(request).db.analysis(aid) or _404("Analysis")
    p = (a.get("files") or {}).get(name)
    if not p or not Path(p).exists():
        _404("File")
    stem = Path(a["filename"]).stem
    fname = f"{stem}_{DOWNLOAD_NAMES.get(name, name)}{Path(p).suffix}"
    return FileResponse(p, filename=fname if download else None,
                        content_disposition_type="attachment" if download else "inline")


# ------------------------------------------------------------------ alerts
def _alert_out(x: dict) -> dict:
    x = dict(x)
    x["evidence_url"] = f"/api/alerts/{x['id']}/evidence.jpg" if x.get("evidence") else None
    x.pop("evidence", None)
    return x


@router.get("/alerts")
def list_alerts(request: Request, state: Optional[str] = None, kind: Optional[str] = None,
                limit: int = 100, offset: int = 0):
    db = st(request).db
    rows, total = db.alerts(min(limit, 500), offset, state, kind)
    new_total = db.alerts(limit=1, state="new")[1]
    return {"alerts": [_alert_out(r) for r in rows], "total": total, "new": new_total}


@router.get("/alerts/{alert_id}/evidence.jpg")
def alert_evidence(request: Request, alert_id: int):
    a = st(request).db.alert(alert_id) or _404("Alert")
    if not a.get("evidence") or not Path(a["evidence"]).exists():
        _404("Evidence")
    return FileResponse(a["evidence"], media_type="image/jpeg")


class AlertState(BaseModel):
    state: str


@router.post("/alerts/{alert_id}/state")
def set_alert_state(request: Request, alert_id: int, body: AlertState):
    if body.state not in ("new", "acknowledged", "resolved"):
        raise HTTPException(422, "state must be new, acknowledged or resolved")
    db = st(request).db
    db.alert(alert_id) or _404("Alert")
    db.set_alert_state(alert_id, body.state)
    return {"ok": True}


# ------------------------------------------------------------------ overview / analytics / search
@router.get("/overview")
def overview(request: Request):
    db = st(request).db
    latest = db.latest_done()
    recent, total = db.analyses(limit=8)
    alerts, _ = db.alerts(limit=5)
    return {
        "latest": analysis_detail(request, latest["id"]) if latest else None,
        "recent": [_analysis_out(r) for r in recent], "analyses_total": total,
        "alerts_new": db.alerts(limit=1, state="new")[1], "alerts_total": db.alerts(limit=1)[1],
        "recent_alerts": [_alert_out(a) for a in alerts],
        "running": [_analysis_out(r) for r in db.analyses(limit=5)[0] if r["state"] in ("queued", "running")],
    }


@router.get("/analytics")
def analytics(request: Request, start: Optional[str] = None):
    db = st(request).db
    out = db.analytics(start)
    today = now_iso()[:10]
    out["today"] = db.analytics(f"{today}T00:00:00")
    out["today"].pop("by_day", None)
    return out


@router.get("/search")
def search(request: Request, q: str):
    db = st(request).db
    rows, _ = db.analyses(limit=8, q=q)
    alerts, _ = db.alerts(limit=200)
    ql = q.lower()
    hits = [a for a in alerts if ql in (a.get("filename") or "").lower() or ql in a["title"].lower()][:8]
    return {"analyses": [_analysis_out(r) for r in rows], "alerts": [_alert_out(a) for a in hits]}


# ------------------------------------------------------------------ maintenance
@router.delete("/analyses/{aid}")
def delete_analysis(request: Request, aid: str):
    """Removes an analysis and its generated files (the uploaded source file is kept)."""
    from backend.core.config import ANALYSES
    s = st(request)
    a = s.db.analysis(aid) or _404("Analysis")
    if a["state"] in ("queued", "running"):
        raise HTTPException(409, "Cancel the analysis first")
    with s.db._lock, s.db._conn() as c:
        c.execute("DELETE FROM alerts WHERE analysis_id=?", (aid,))
        c.execute("DELETE FROM persons WHERE analysis_id=?", (aid,))
        c.execute("DELETE FROM analyses WHERE id=?", (aid,))
    shutil.rmtree(ANALYSES / aid, ignore_errors=True)
    return {"ok": True}
