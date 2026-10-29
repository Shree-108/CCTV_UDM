"""Frame extraction from uploaded videos (the upload itself is the existing POST /api/media)."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.extraction import business_csv, master_csv, reports
from backend.extraction.detect import RESULT_FIELDS, normalize
from backend.extraction.extractor import row_meta
from backend.extraction.runner import EXTRACTIONS, DuplicateJob, ExtractionRejected, manifest_path, plan_for, read_manifest

router = APIRouter(prefix="/api")


def st(request: Request):
    return request.app.state.app_state


def _404(what: str):
    raise HTTPException(404, f"{what} not found")


class ExtractionIn(BaseModel):
    media_id: str
    sampling_strategy: Optional[Literal["interval", "random", "random_segments"]] = Field(
        default=None, description="Default: interval when interval_seconds is given, otherwise Settings (random_segments)")
    interval_seconds: Optional[float] = Field(default=None, gt=0, le=86400,
                                              description="interval: seconds between scheduled samples (15 min = 900)")
    sample_count: Optional[int] = Field(default=None, ge=1, description="random strategies: number of screenshots (default: Settings, 4)")
    random_seed: Optional[int] = Field(default=None, ge=0, le=2 ** 31 - 1,
                                       description="random: seed (omit to generate one; it is recorded on the job)")
    store_id: Optional[str] = Field(default=None, max_length=64, description="Leave empty when unknown")
    camera_id: Optional[str] = Field(default=None, max_length=64, description="Leave empty when unknown")
    recording_start_time: Optional[str] = Field(default=None, max_length=40,
                                                description="ISO 8601 clock time of the first frame, only if known")
    include_initial: Optional[bool] = Field(default=None, description="Also save the frame at 00:00:00 (sample_type 'initial', "
                                            "not counted as one of the screenshots). Default: on for interval, off for random")
    new_run: bool = Field(default=False, description="Extract again even if an identical finished job exists")
    run_detection: bool = Field(default=False, description="Run the uniform model on the frames when extraction finishes")
    threshold: Optional[float] = Field(default=None, ge=0.5, le=0.99, description="Detection threshold (default: Settings)")


def _job_out(j: dict) -> dict:
    base = f"/api/extractions/{j['id']}"
    plan = {k: v for k, v in (j.get("plan") or {}).items() if k not in ("targets", "samples")}
    strategy = j.get("sampling_strategy") or "interval"
    return j | {"include_initial": bool(j["include_initial"]), "plan": plan, "sampling_strategy": strategy,
                "samples": _sample_status(j),
                "interval_seconds": j["interval_seconds"] if strategy == "interval" else None,
                "source": j.get("source") or "upload",
                "failures": [f | {"error_message": f.get("error_message") or f.get("reason", "")}
                             for f in j.get("failures") or []],
                "manifest_url": f"{base}/image_manifest.csv", "failures_url": f"{base}/failures.csv"}


def _sample_status(j: dict) -> list[dict] | None:
    """The selected timestamps of a (small) job, each with saved | failed | pending | not_extracted."""
    samples = (j.get("plan") or {}).get("samples")
    if samples is None:
        return None
    saved = {r["image_id"]: r for r in read_manifest(j)}
    failed = {f["image_id"] for f in j.get("failures") or []}
    out = []
    for x in samples:
        m = saved.get(x["image_id"])
        status = ("saved" if m else "failed" if x["image_id"] in failed
                  else "pending" if j["state"] in ("queued", "running") else "not_extracted")
        out.append(x | {"status": status, "frame_number": int(m["frame_number"]) if m else None,
                        "video_timestamp_hhmmss": m["video_timestamp_hhmmss"] if m else None})
    return out


def _job_dir(job: dict) -> Path:
    return EXTRACTIONS / job["id"]


@router.get("/media/{mid}/extraction-plan")
def extraction_plan(request: Request, mid: str, interval_seconds: Optional[float] = None, include_initial: bool = True,
                    sampling_strategy: Literal["interval", "random", "random_segments"] = "interval", sample_count: Optional[int] = None,
                    random_seed: Optional[int] = None):
    """Exactly which samples a job would extract, with a plain-language explanation."""
    s = st(request)
    media = s.db.media(mid) or _404("Media")
    cfg = s.settings.get()
    if sampling_strategy != "interval" and sample_count is None:
        sample_count = int(cfg["extraction_default_sample_count"])
    try:
        _, summary = plan_for(media, interval_seconds, include_initial, int(cfg["extraction_max_frames"]),
                              sampling_strategy, sample_count, random_seed)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return summary


@router.post("/extractions", status_code=202)
def create_extraction(request: Request, body: ExtractionIn):
    s = st(request)
    media = s.db.media(body.media_id) or _404("Media")
    try:
        job, reused = s.extractions.create(media, body.interval_seconds, body.include_initial, body.new_run,
                                           body.run_detection, body.threshold, strategy=body.sampling_strategy,
                                           sample_count=body.sample_count, seed=body.random_seed,
                                           store_id=body.store_id, camera_id=body.camera_id,
                                           recording_start_time=body.recording_start_time)
    except DuplicateJob as e:
        raise HTTPException(409, {"message": str(e), "job_id": e.job["id"]})
    except ValueError as e:                      # ExtractionRejected and plan errors
        raise HTTPException(422, str(e))
    return _job_out(job) | {"reused": reused}


@router.get("/extractions")
def list_extractions(request: Request, media_id: Optional[str] = None, limit: int = 50):
    return {"extractions": [_job_out(j) for j in st(request).db.extractions(media_id, min(limit, 200))]}


@router.get("/extractions/{eid}")
def get_extraction(request: Request, eid: str):
    return _job_out(st(request).db.extraction(eid) or _404("Extraction"))


@router.post("/extractions/{eid}/cancel")
def cancel_extraction(request: Request, eid: str):
    s = st(request)
    j = s.db.extraction(eid) or _404("Extraction")
    if j["state"] not in ("queued", "running"):
        raise HTTPException(409, "This extraction is not running")
    s.extractions.cancel(eid)
    return {"ok": True}


_read_manifest = read_manifest


@router.get("/extractions/{eid}/frames")
def list_frames(request: Request, eid: str, limit: int = 200, offset: int = 0):
    """Successfully extracted frames (from image_manifest.csv), with image URLs."""
    j = st(request).db.extraction(eid) or _404("Extraction")
    rows = _read_manifest(j)
    page = rows[offset: offset + min(limit, 1000)]
    return {"total": len(rows), "frames": [r | {"url": f"/api/extractions/{eid}/frames/{r['image_filename']}"} for r in page]}


@router.get("/extractions/{eid}/frames/{filename}")
def frame_image(request: Request, eid: str, filename: str):
    j = st(request).db.extraction(eid) or _404("Extraction")
    if filename not in {r["image_filename"] for r in _read_manifest(j)}:   # only files listed in this job's manifest
        _404("Frame")
    p = _job_dir(j) / "frames" / filename
    return FileResponse(p) if p.exists() else _404("Frame")


# ------------------------------------------------------------------ business-facing uniform compliance CSV
def _compliance_run(db, eid: str, run_id: Optional[str]) -> tuple[dict, dict | None]:
    """(extraction, run): the given run, or the latest run that processed frames (None = not analysed yet)."""
    ext = db.extraction(eid) or _404("Extraction")
    if run_id:
        run = db.detection_run(run_id)
        if not run or run["extraction_id"] != eid:
            _404("Detection run")
        return ext, run
    return ext, next((r for r in db.detection_runs(eid) if r["processed"] > 0), None)


def _detection_status(s, ext: dict, run: dict | None) -> tuple[dict | None, bool, str]:
    """(latest run of any state, detection requested but not created yet, how to describe unanalysed shots)."""
    latest = next(iter(s.db.detection_runs(ext["id"])), None)
    pending = s.extractions.detection_requested(ext["id"])
    return latest, pending, business_csv.unanalysed_status(run, latest, pending)


@router.get("/extractions/{eid}/uniform-compliance")
def uniform_compliance(request: Request, eid: str, run_id: Optional[str] = None):
    """Preview of uniform_compliance_job_<job>.csv: one record per sampled screenshot."""
    s = st(request)
    ext, run = _compliance_run(s.db, eid, run_id)
    latest, pending, unanalysed = _detection_status(s, ext, run)
    q = f"?run_id={run['id']}" if run else ""
    return {"job_id": eid, "screenshots_sampled": ext["planned"], "extracted": ext["extracted"],
            "extraction_state": ext["state"], "run_id": run["id"] if run else None,
            "run_state": run["state"] if run else None,
            # the newest detection run of ANY state (queued / running / failed too) - the UI keeps polling while
            # detection is requested or active, so it never stops at "not analysed" before the model has run
            "detection_state": latest["state"] if latest else None, "detection_error": latest["error"] if latest else None,
            "detection_pending": pending, "file_name": business_csv.file_name(eid),
            "saved": bool(run) and business_csv.saved_path(ext, run).exists(),
            "csv_url": f"/api/extractions/{eid}/uniform_compliance.csv{q}",
            "fields": business_csv.BUSINESS_FIELDS, "records": business_csv.records(ext, run, unanalysed)}


@router.get("/extractions/{eid}/uniform_compliance.csv")
def uniform_compliance_csv(request: Request, eid: str, run_id: Optional[str] = None, download: bool = False):
    """The saved snapshot of the run, or (older runs / not analysed yet) generated now from the stored files."""
    s = st(request)
    ext, run = _compliance_run(s.db, eid, run_id)
    name = business_csv.file_name(eid)
    p = business_csv.saved_path(ext, run) if run else None
    if p and p.exists():
        return FileResponse(p, media_type="text/csv", filename=name if download else None,
                            content_disposition_type="attachment" if download else "inline")
    headers = {"Content-Disposition": f'attachment; filename="{name}"'} if download else {}
    return Response(business_csv.to_csv(ext, run, _detection_status(s, ext, run)[2]), media_type="text/csv", headers=headers)


# ------------------------------------------------------------------ cumulative master CSV (all videos)
@router.get("/uniform-compliance/master")
def master_status(request: Request):
    """Rows / videos in uniform_compliance_master.csv, jobs not yet appended and the last write error."""
    return master_csv.status(st(request).db)


@router.post("/uniform-compliance/master/sync")
def master_sync(request: Request):
    """Append finished jobs that are missing (e.g. after a failed write). Safe to repeat: never duplicates."""
    db = st(request).db
    return master_csv.sync(db) | {"status": master_csv.status(db)}


@router.get("/uniform-compliance/master.csv")
def master_download(download: bool = False):
    p = master_csv.master_path()
    if not p.exists():
        raise HTTPException(404, "No video has been processed yet, so the master CSV does not exist")
    return FileResponse(p, media_type="text/csv", filename=master_csv.MASTER_NAME if download else None,
                        content_disposition_type="attachment" if download else "inline")


# ------------------------------------------------------------------ annotated video (full-video analysis pass)
FULL_FRAME = {"rect": [0.0, 0.0, 1.0, 1.0], "roi": None, "label": "Camera 1", "layout": "1x1", "cell": None}


def _video_ok(p: Path) -> bool:
    """The annotated file exists, is not empty and decodes (checked before a link is shown)."""
    if not p.is_file() or p.stat().st_size == 0:
        return False
    import cv2
    cap = cv2.VideoCapture(str(p))
    try:
        return cap.isOpened() and cap.read()[0]
    finally:
        cap.release()


def _annotated_video(s, ext: dict) -> dict:
    rows, _ = s.db.analyses(limit=500, kind="video")
    mine = [a for a in rows if a["media_id"] == ext["media_id"]]           # newest first
    a = next((x for x in mine if x["state"] in ("queued", "running")), None) or next(iter(mine), None)
    out = {"job_id": ext["id"], "media_id": ext["media_id"], "analysis_id": None, "state": "not_started",
           "progress": 0.0, "available": False, "video_url": None, "download_url": None, "error": None,
           "note": ("A separate full-video pass with the person tracker and the same uniform classifier. It is not "
                    "part of the four-screenshot result or the CSV files; the original video is not changed.")}
    if not s.db.media(ext["media_id"]):
        return out | {"state": "unavailable", "error": "The source video is no longer stored, so it cannot be annotated"}
    if not a:
        return out
    out |= {"analysis_id": a["id"], "state": a["state"], "progress": a.get("progress") or 0.0, "error": a.get("error")}
    p = (a.get("files") or {}).get("annotated")
    if a["state"] == "done":
        if p and _video_ok(Path(p)):
            base = f"/api/analyses/{a['id']}/files/annotated"
            out |= {"available": True, "video_url": base, "download_url": f"{base}?download=true",
                    "playable_in_browser": Path(p).name == "annotated.mp4"}       # H.264; annotated_raw.mp4 = mp4v
        else:
            out["error"] = "The analysis finished but its annotated video is missing or unreadable"
    return out


@router.get("/extractions/{eid}/annotated-video")
def annotated_video(request: Request, eid: str):
    s = st(request)
    return _annotated_video(s, s.db.extraction(eid) or _404("Extraction"))


@router.post("/extractions/{eid}/annotated-video", status_code=202)
def create_annotated_video(request: Request, eid: str):
    """Start the full-video analysis of this job's video (or return the one already running / finished)."""
    from backend.analysis.runner import AnalysisFailed, start_analysis
    s = st(request)
    ext = s.db.extraction(eid) or _404("Extraction")
    cur = _annotated_video(s, ext)
    if cur["state"] in ("queued", "running") or cur["available"] or cur["state"] == "unavailable":
        return cur
    media = s.db.media(ext["media_id"])
    try:
        start_analysis(s.db, s.settings, s.models, s.jobs, media, dict(FULL_FRAME),
                       {"store_id": ext.get("store_id"), "camera_id": ext.get("camera_id")})
    except (ValueError, AnalysisFailed) as e:
        raise HTTPException(422, str(e))
    return _annotated_video(s, ext)


@router.get("/extractions/{eid}/{name}.csv")
def extraction_csv(request: Request, eid: str, name: str, download: bool = False):
    if name not in ("image_manifest", "manifest", "failures"):          # "manifest" = older URL, same file
        _404("File")
    j = st(request).db.extraction(eid) or _404("Extraction")
    p = _job_dir(j) / "failures.csv" if name == "failures" else manifest_path(j)
    name = "failures" if name == "failures" else "image_manifest"
    if not p.exists():
        _404("File")
    stem = Path(j["source_video"]).stem
    return FileResponse(p, media_type="text/csv", filename=f"{stem}_{eid}_{name}.csv" if download else None,
                        content_disposition_type="attachment" if download else "inline")


# ------------------------------------------------------------------ uniform detection on extracted frames
class DetectIn(BaseModel):
    threshold: Optional[float] = Field(default=None, ge=0.5, le=0.99,
                                       description="P(uniform) needed for a target-shirt detection (default: Settings)")
    new_run: bool = False


def _run_out(r: dict) -> dict:
    base = f"/api/detection-runs/{r['id']}"
    return r | {"results_csv_url": f"{base}/detection_results.csv", "results_url": f"{base}/results",
                "screenshot_summary_url": f"{base}/screenshot_summary.csv",
                "compliance_details_url": f"{base}/uniform_compliance_details.csv"}


def _run_dir(r: dict) -> Path:
    return EXTRACTIONS / r["extraction_id"] / "detections" / r["id"]


def _results(r: dict, ext: dict | None) -> list[dict]:
    p = _run_dir(r) / "detection_results.csv"
    if not p.exists():
        return []
    meta = row_meta(ext) if ext else None
    with open(p, newline="", encoding="utf-8") as f:
        return [normalize(row, r, meta) for row in csv.DictReader(f)]


@router.post("/extractions/{eid}/detect", status_code=202)
def start_detection(request: Request, eid: str, body: DetectIn):
    s = st(request)
    job = s.db.extraction(eid) or _404("Extraction")
    try:
        run, reused = s.extractions.create_detection(job, body.threshold, body.new_run)
    except DuplicateJob as e:
        raise HTTPException(409, {"message": str(e), "run_id": e.job["id"]})
    except ExtractionRejected as e:
        raise HTTPException(422, str(e))
    return _run_out(run) | {"reused": reused}


@router.get("/extractions/{eid}/detections")
def list_detection_runs(request: Request, eid: str):
    s = st(request)
    s.db.extraction(eid) or _404("Extraction")
    return {"runs": [_run_out(r) for r in s.db.detection_runs(eid)]}


@router.get("/detection-runs/{rid}")
def get_detection_run(request: Request, rid: str):
    return _run_out(st(request).db.detection_run(rid) or _404("Detection run"))


@router.post("/detection-runs/{rid}/cancel")
def cancel_detection_run(request: Request, rid: str):
    s = st(request)
    r = s.db.detection_run(rid) or _404("Detection run")
    if r["state"] not in ("queued", "running"):
        raise HTTPException(409, "This detection run is not running")
    s.extractions.cancel(rid)
    return {"ok": True}


@router.get("/detection-runs/{rid}/results")
def detection_results(request: Request, rid: str):
    """detection_results.csv grouped per image, with URLs of the original and annotated images."""
    db = st(request).db
    r = db.detection_run(rid) or _404("Detection run")
    images: dict[str, dict] = {}
    for row in _results(r, db.extraction(r["extraction_id"])):
        img = images.get(row["image_id"])
        if img is None:
            img = images[row["image_id"]] = {
                k: row[k] for k in ("image_id", "extraction_job_id", "detection_run_id", "source_video", "store_id",
                                    "camera_id", "video_timestamp_hhmmss", "recording_timestamp", "prediction",
                                    "processing_status", "review_status", "review_reason", "model_version")}
            img["video_timestamp_seconds"] = float(row["video_timestamp_seconds"])
            img["detection_count"] = int(row["detection_count"] or 0)
            img["person_count"] = int(row["person_count"]) if row["person_count"] not in ("", None) else None
            img["original_url"] = f"/api/extractions/{r['extraction_id']}/frames/{Path(row['original_image_path']).name}"
            img["annotated_url"] = (f"/api/detection-runs/{rid}/annotated/{Path(row['annotated_image_path']).name}"
                                    if row["annotated_image_path"] else None)
            img["detections"] = []
        if row["detection_type"] == "uniform_shirt" and row["confidence"]:
            img["detections"].append({"confidence": float(row["confidence"]),
                                      "bbox": [int(row[k]) for k in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")]})
    return {"fields": RESULT_FIELDS, "images": list(images.values())}


@router.get("/detection-runs/{rid}/annotated/{filename}")
def annotated_image(request: Request, rid: str, filename: str):
    r = st(request).db.detection_run(rid) or _404("Detection run")
    if filename not in {Path(x["annotated_image_path"]).name for x in _results(r, None) if x["annotated_image_path"]}:
        _404("Annotated image")
    p = _run_dir(r) / "annotated" / filename
    return FileResponse(p, media_type="image/jpeg") if p.exists() else _404("Annotated image")


@router.get("/detection-runs/{rid}/{name}.csv")
def run_report_csv(request: Request, rid: str, name: str, download: bool = False):
    """screenshot_summary.csv / uniform_compliance_details.csv of one run: the snapshot written when
    the run finished, or (older runs) generated now from the stored files."""
    if name == "detection_results":
        return detection_results_csv(request, rid, download)
    if name not in reports.REPORT_NAMES:
        _404("File")
    db = st(request).db
    r = db.detection_run(rid) or _404("Detection run")
    p = _run_dir(r) / f"{name}.csv"
    disp = f"{rid}_{name}.csv" if download else None
    if p.exists():
        return FileResponse(p, media_type="text/csv", filename=disp, content_disposition_type="attachment" if download else "inline")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=reports.fields(name))
    w.writeheader()
    w.writerows(reports.build(name, db.extraction(r["extraction_id"]), r))
    headers = {"Content-Disposition": f'attachment; filename="{disp}"'} if download else {}
    return Response(buf.getvalue(), media_type="text/csv", headers=headers)


def detection_results_csv(request: Request, rid: str, download: bool = False):
    r = st(request).db.detection_run(rid) or _404("Detection run")
    p = _run_dir(r) / "detection_results.csv"
    if not p.exists():
        _404("File")
    return FileResponse(p, media_type="text/csv", filename=f"{rid}_detection_results.csv" if download else None,
                        content_disposition_type="attachment" if download else "inline")
