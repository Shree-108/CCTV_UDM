"""Results dashboard API: real results across all processed videos + combined CSV exports."""
from __future__ import annotations

import csv
import io
import time
from typing import Literal, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response

from backend.capture import cctv_adapter
from backend.extraction import reports
from backend.extraction import results as R

router = APIRouter(prefix="/api")
Outcome = Literal["UNIFORM_DETECTED", "NO_UNIFORM_DETECTED", "REVIEW_REQUIRED", "PROCESSING_FAILED"]
ProcStatus = Literal["ok", "failed", "annotation_failed"]
DateQ = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="Recording date (only jobs with a known recording start time)")


def st(request: Request):
    return request.app.state.app_state


@router.get("/frame-results")
def frame_results(request: Request, source_video: Optional[str] = None, camera_id: Optional[str] = None,
                  store_id: Optional[str] = None, date: Optional[str] = DateQ, outcome: Optional[Outcome] = None,
                  processing_status: Optional[ProcStatus] = None, limit: int = 200, offset: int = 0):
    """Summary + per-video jobs + filtered per-image results. The summary follows the video /
    camera / store / date filters only, so the totals do not change when the table is filtered
    by result or status."""
    db = st(request).db
    videos, images = R.collect(db, source_video, camera_id, store_id, date)
    all_videos, _ = R.collect(db) if (source_video or camera_id or store_id or date) else (videos, images)
    table = [i for i in images if (not outcome or i["prediction"] == outcome)
             and (not processing_status or i["processing_status"] == processing_status)]
    failures = R.processing_failure_rows(db, source_video, camera_id, store_id)
    return {
        "summary": R.summarize(videos, images) | {"processing_failures": len(failures)},
        "videos": videos,
        "source_videos": sorted({v["source_video"] for v in all_videos}),
        "cameras": sorted({v["camera_id"] for v in all_videos if v["camera_id"]}),
        "stores": sorted({v["store_id"] for v in all_videos if v["store_id"]}),
        "recording_dates": sorted({(v["recording_start_time"] or "")[:10] for v in all_videos if v["recording_start_time"]}),
        "images": table[offset: offset + min(limit, 1000)], "images_total": len(table),
        "hourly": R.hourly(R.hourly_rows(db, source_video, camera_id, store_id, date)),
        "failures": failures[:200], "failures_total": len(failures),
        "storage": R.storage_status(),
        "cctv": cctv_adapter.status(),
        "counting_note": "Counts are of sampled frames (images), not unique people.",
    }


def _csv(rows: list[dict], fields: list[str], name: str) -> Response:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}_{time.strftime("%Y%m%d_%H%M%S")}.csv"'})


@router.get("/frame-results/image_manifest.csv")
@router.get("/frame-results/manifest.csv", include_in_schema=False)          # older URL
def export_manifest(request: Request, source_video: Optional[str] = None, camera_id: Optional[str] = None,
                    store_id: Optional[str] = None, date: Optional[str] = DateQ):
    return _csv(R.manifest_rows(st(request).db, source_video, camera_id, store_id, date),
                R.MANIFEST_EXPORT_FIELDS, "image_manifest")


@router.get("/frame-results/detection_results.csv")
def export_detections(request: Request, source_video: Optional[str] = None, outcome: Optional[Outcome] = None,
                      processing_status: Optional[ProcStatus] = None, camera_id: Optional[str] = None,
                      store_id: Optional[str] = None, date: Optional[str] = DateQ):
    rows = R.detection_rows(st(request).db, source_video, outcome, processing_status, camera_id, store_id, date)
    return _csv(rows, R.RESULTS_EXPORT_FIELDS, "detection_results")


@router.get("/frame-results/screenshot_summary.csv")
def export_screenshot_summary(request: Request, source_video: Optional[str] = None, camera_id: Optional[str] = None,
                              store_id: Optional[str] = None, date: Optional[str] = DateQ):
    rows = R.report_rows(st(request).db, "screenshot_summary", source_video, camera_id, store_id, date)
    return _csv(rows, reports.SUMMARY_FIELDS, "screenshot_summary")


@router.get("/frame-results/uniform_compliance_details.csv")
def export_compliance_details(request: Request, source_video: Optional[str] = None, camera_id: Optional[str] = None,
                              store_id: Optional[str] = None, date: Optional[str] = DateQ):
    rows = R.report_rows(st(request).db, "uniform_compliance_details", source_video, camera_id, store_id, date)
    return _csv(rows, reports.DETAIL_FIELDS, "uniform_compliance_details")


@router.get("/frame-results/processing_failures.csv")
def export_failures(request: Request, source_video: Optional[str] = None, camera_id: Optional[str] = None,
                    store_id: Optional[str] = None):
    rows = R.processing_failure_rows(st(request).db, source_video, camera_id, store_id)
    return _csv(rows, R.FAILURE_EXPORT_FIELDS, "processing_failures")


@router.get("/frame-results/hourly_summary.csv")
def export_hourly(request: Request, source_video: Optional[str] = None, camera_id: Optional[str] = None,
                  store_id: Optional[str] = None, date: Optional[str] = DateQ):
    rows = R.hourly_rows(st(request).db, source_video, camera_id, store_id, date)
    return _csv(rows, R.HOURLY_FIELDS, "hourly_summary")
