"""Results across all processed videos, read from the files the jobs actually wrote:
extractions (image_manifest.csv, failures.csv) and, per extraction, its LATEST detection run that
processed frames (detection_results.csv). Nothing here is estimated or cached.

Datasets (all joinable on image_id / extraction_job_id / detection_run_id)
  image_manifest        one row per successfully extracted image
  detection_results     one row per person found (or one row per image without people / that failed)
  processing_failures   failed extraction samples, failed inferences / annotations, failed jobs and runs
  hourly_summary        per job and hour: expected / successful / failed samples, analysed images, outcomes

Counting rules
  * image counts are of SAMPLED FRAMES, never of people: one person seen in 10 frames counts in
    10 images, and two people in one frame count once;
  * `detections` is the number of target-shirt detections summed over images - also not people;
  * NO_UNIFORM_DETECTED means no target shirt passed the threshold in that image; it is not a
    confirmed uniform violation;
  * hours are clock hours only when the recording start time of the video was supplied;
    otherwise they are hours of VIDEO TIME (offset from the start of the file). Hours without
    footage never appear.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

from backend.core.storage import get_storage
from backend.database.db import DB
from backend.extraction.detect import (NO_UNIFORM_DETECTED, PROCESSING_FAILED, RESULT_FIELDS, REVIEW_REQUIRED,
                                       UNIFORM_DETECTED, normalize)
from backend.extraction.extractor import MANIFEST_FIELDS, row_meta
from backend.extraction.plan import hhmmss, parse_recording_start
from backend.extraction.runner import EXTRACTIONS, expected_targets, read_failures, read_manifest
from backend.extraction.screenshots import remark

ACTIVE = ("queued", "running")
UNFINISHED = ("cancelled", "interrupted", "failed")


def latest_processed_run(db: DB, extraction_id: str) -> dict | None:
    runs = db.detection_runs(extraction_id)              # newest first
    return next((r for r in runs if r["processed"] > 0), None)


def latest_run(db: DB, extraction_id: str) -> dict | None:
    runs = db.detection_runs(extraction_id)
    return runs[0] if runs else None


def results_rows(run: dict, ext: dict | None = None) -> list[dict]:
    p = EXTRACTIONS / run["extraction_id"] / "detections" / run["id"] / "detection_results.csv"
    if not p.exists():
        return []
    meta = row_meta(ext) if ext else None
    with open(p, newline="", encoding="utf-8") as f:
        return [normalize(r, run, meta) for r in csv.DictReader(f)]


def _date_of(ts: str) -> str:
    return ts[:10] if ts else ""


def jobs(db: DB, source_video: str | None = None, camera_id: str | None = None, store_id: str | None = None) -> list[dict]:
    out = []
    for ext in db.extractions(limit=1000):
        if source_video and ext["source_video"] != source_video:
            continue
        if camera_id and (ext.get("camera_id") or "") != camera_id:
            continue
        if store_id and (ext.get("store_id") or "") != store_id:
            continue
        out.append(ext)
    return out


def collect(db: DB, source_video: str | None = None, camera_id: str | None = None, store_id: str | None = None,
            date: str | None = None) -> tuple[list[dict], list[dict]]:
    """(videos, images). `videos`: one entry per extraction job; `images`: one per processed frame.
    `date` (YYYY-MM-DD) keeps only images whose real recording time is on that date."""
    videos, images = [], []
    for ext in jobs(db, source_video, camera_id, store_id):
        run = latest_processed_run(db, ext["id"])
        last = latest_run(db, ext["id"])
        strategy = ext.get("sampling_strategy") or "interval"
        videos.append({
            "extraction_id": ext["id"], "source_video": ext["source_video"], "media_id": ext["media_id"],
            "store_id": ext.get("store_id"), "camera_id": ext.get("camera_id"),
            "recording_start_time": ext.get("recording_start_time"), "source": ext.get("source") or "upload",
            "created_at": ext["created_at"], "state": ext["state"], "sampling_strategy": strategy,
            "interval_seconds": ext["interval_seconds"] if strategy == "interval" else None,
            "sample_count": ext.get("sample_count"), "random_seed": ext.get("random_seed"),
            "duration_seconds": (ext.get("plan") or {}).get("duration_seconds"), "planned": ext["planned"],
            "extracted": ext["extracted"], "extraction_failed": ext["failed"], "extraction_error": ext["error"],
            "progress": ext["progress"],
            "run_id": run["id"] if run else None, "run_state": run["state"] if run else None,
            "processed": run["processed"] if run else 0, "threshold": run["threshold"] if run else None,
            "model_version": run["model_version"] if run else None,
            # the most recent run, even if it processed nothing (e.g. refused because the model is missing)
            "last_run_state": last["state"] if last else None, "last_run_error": last["error"] if last else None,
            "last_run_progress": last["progress"] if last else None,
        })
        if not run:
            continue
        per_image: dict[str, dict] = {}
        for row in results_rows(run, ext):
            im = per_image.get(row["image_id"])
            if im is None:
                im = per_image[row["image_id"]] = {
                    "extraction_id": ext["id"], "run_id": run["id"], "image_id": row["image_id"],
                    "source_video": row["source_video"], "store_id": row["store_id"], "camera_id": row["camera_id"],
                    "timestamp": row["video_timestamp_hhmmss"],
                    "timestamp_seconds": float(row["video_timestamp_seconds"]),
                    "recording_timestamp": row["recording_timestamp"], "prediction": row["prediction"],
                    "detection_count": int(row["detection_count"]) if row["detection_count"] not in ("", None) else 0,
                    "person_count": int(row["person_count"]) if row["person_count"] not in ("", None) else None,
                    "confidences": [], "people": [],
                    "processing_status": row["processing_status"], "review_status": row["review_status"],
                    "review_reason": row["review_reason"], "model_version": row["model_version"],
                    "threshold": float(row["confidence_threshold"]), "sampling_strategy": strategy,
                    "original_image_path": row["original_image_path"],
                    "annotated_image_path": row["annotated_image_path"],
                    "original_url": f"/api/extractions/{ext['id']}/frames/{Path(row['original_image_path']).name}",
                    "annotated_url": (f"/api/detection-runs/{run['id']}/annotated/{Path(row['annotated_image_path']).name}"
                                      if row["annotated_image_path"] else None),
                }
            if row["detection_type"]:
                im["people"].append({"type": row["detection_type"], "reason": row.get("person_reason") or "",
                                     "confidence": float(row["confidence"]) if row["confidence"] else None,
                                     "bbox": [int(row[k]) for k in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")]
                                     if row["bbox_xmin"] not in ("", None) else None})
            if row["detection_type"] == "uniform_shirt" and row["confidence"]:
                im["confidences"].append(float(row["confidence"]))
        for im in per_image.values():
            im["remark"] = remark([p["type"] for p in im["people"]], im["prediction"], im["processing_status"],
                                  [p["reason"] for p in im["people"]])
        images.extend(i for i in per_image.values() if not date or _date_of(i["recording_timestamp"]) == date)
    images.sort(key=lambda i: (i["source_video"], i["extraction_id"], i["timestamp_seconds"]))
    return videos, images


def summarize(videos: list[dict], images: list[dict]) -> dict:
    out = {
        "videos": len({v["source_video"] for v in videos}), "extraction_jobs": len(videos),
        "frames_extracted": sum(v["extracted"] for v in videos),
        "extraction_failures": sum(v["extraction_failed"] for v in videos),
        "frames_processed": len(images),
        "frames_analysed": sum(i["prediction"] != PROCESSING_FAILED for i in images),
        "uniform_images": sum(i["prediction"] == UNIFORM_DETECTED for i in images),
        "no_uniform_images": sum(i["prediction"] == NO_UNIFORM_DETECTED for i in images),
        "review_images": sum(i["prediction"] == REVIEW_REQUIRED for i in images),
        "failed_images": sum(i["prediction"] == PROCESSING_FAILED for i in images),
        "detections": sum(i["detection_count"] for i in images),
        "people_detected": sum(i["person_count"] or 0 for i in images),
        "incomplete_jobs": sum(v["state"] in ACTIVE + UNFINISHED or (v["last_run_state"] or "") in ACTIVE + UNFINISHED
                               for v in videos),
    }
    out["frames_not_processed"] = out["frames_extracted"] - out["frames_processed"]
    return out


# ------------------------------------------------------------------ hourly summary
HOURLY_FIELDS = ["source_video", "extraction_job_id", "detection_run_id", "store_id", "camera_id", "hour_basis",
                 "hour_start", "hour_end", "expected_samples", "successful_samples", "failed_samples",
                 "analysed_images", "processing_failed_images", "uniform_images", "no_uniform_images",
                 "review_required_images", "target_shirt_detections", "person_detections", "summary_status"]


def _bucket(start: datetime | None, seconds: float) -> tuple[str, str]:
    if start:
        h = (start + timedelta(seconds=seconds)).replace(minute=0, second=0, microsecond=0)
        return h.isoformat(), (h + timedelta(hours=1)).isoformat()
    i = int(seconds // 3600)
    return f"video {hhmmss(i * 3600)[:8]}", f"video {hhmmss((i + 1) * 3600)[:8]}"


def hourly_rows(db: DB, source_video: str | None = None, camera_id: str | None = None,
                store_id: str | None = None, date: str | None = None) -> list[dict]:
    """One row per job and hour that has footage samples. Grouped by real clock hour when the
    recording start time is known, otherwise by hour of video time."""
    out = []
    for ext in jobs(db, source_video, camera_id, store_id):
        start = parse_recording_start(ext.get("recording_start_time"))
        if date and not start:
            continue
        run = latest_processed_run(db, ext["id"])
        rows: dict[tuple, dict] = {}

        def row(t: float) -> dict:
            b = _bucket(start, t)
            return rows.setdefault(b, {
                "source_video": ext["source_video"], "extraction_job_id": ext["id"],
                "detection_run_id": run["id"] if run else "", "store_id": ext.get("store_id") or "",
                "camera_id": ext.get("camera_id") or "", "hour_basis": "recording_time" if start else "video_relative",
                "hour_start": b[0], "hour_end": b[1], "expected_samples": 0, "successful_samples": 0,
                "failed_samples": 0, "analysed_images": 0, "processing_failed_images": 0, "uniform_images": 0,
                "no_uniform_images": 0, "review_required_images": 0, "target_shirt_detections": 0,
                "person_detections": 0})

        for t in expected_targets(ext, db.media(ext["media_id"])):
            row(float(t))["expected_samples"] += 1
        for m in read_manifest(ext):
            row(float(m["video_timestamp_seconds"]))["successful_samples"] += 1
        for fl in read_failures(ext):
            row(float(fl["target_seconds"]))["failed_samples"] += 1
        if run:
            seen = set()
            for r in results_rows(run, ext):
                if r["image_id"] in seen:
                    continue
                seen.add(r["image_id"])
                x = row(float(r["video_timestamp_seconds"]))
                if r["prediction"] == PROCESSING_FAILED:
                    x["processing_failed_images"] += 1
                    continue
                x["analysed_images"] += 1
                x[{UNIFORM_DETECTED: "uniform_images", NO_UNIFORM_DETECTED: "no_uniform_images",
                   REVIEW_REQUIRED: "review_required_images"}[r["prediction"]]] += 1
                x["target_shirt_detections"] += int(r["detection_count"] or 0)
                x["person_detections"] += int(r["person_count"] or 0)
        for x in rows.values():
            if date and not x["hour_start"].startswith(date):
                continue
            x["summary_status"] = _hour_status(x, ext, run)
            out.append(x)
    out.sort(key=lambda r: (r["source_video"], r["extraction_job_id"], r["hour_start"]))
    return out


def _hour_status(x: dict, ext: dict, run: dict | None) -> str:
    if ext["state"] in ACTIVE or (run and run["state"] in ACTIVE):
        return "in_progress"
    if x["successful_samples"] + x["failed_samples"] < x["expected_samples"]:
        return "extraction_incomplete"
    if not run:
        return "not_analysed"
    if x["analysed_images"] + x["processing_failed_images"] < x["successful_samples"]:
        return "analysis_incomplete"
    if x["failed_samples"] or x["processing_failed_images"]:
        return "complete_with_failures"
    return "complete"


def hourly(rows: list[dict]) -> dict:
    if not rows:
        return {"available": False, "reason": "No extraction job matches these filters yet."}
    bases = {r["hour_basis"] for r in rows}
    basis = ("recording clock time" if bases == {"recording_time"}
             else "video time (hours from the start of each file; the recording clock time is not known)"
             if bases == {"video_relative"} else "recording clock time where known, otherwise video time")
    return {"available": True, "basis": basis, "rows": rows}


# ------------------------------------------------------------------ processing failures
FAILURE_EXPORT_FIELDS = ["stage", "extraction_job_id", "detection_run_id", "image_id", "source_video", "store_id",
                         "camera_id", "video_timestamp_seconds", "video_timestamp_hhmmss", "target_frame",
                         "original_image_path", "error_category", "error_message", "retry_status"]


def _extraction_category(row: dict) -> str:
    if row.get("error_category"):
        return row["error_category"]
    msg = row.get("reason", "")                          # older failures.csv files
    return ("decoder_error" if msg.startswith("decoder error") else "no_frame" if "no frame" in msg
            else "path_too_long" if "Windows limit" in msg else "write_failed")


def _inference_category(r: dict) -> tuple[str, str]:
    if r["processing_status"] == "annotation_failed":
        return "annotation", "annotation_write_failed"
    if (r["review_reason"] or "").startswith("original image could not be read"):
        return "inference", "image_unreadable"
    return "inference", "inference_error"


def processing_failure_rows(db: DB, source_video: str | None = None, camera_id: str | None = None,
                            store_id: str | None = None) -> list[dict]:
    """Every failed operation that left a record. retry_status: retried_ok / retried_failed when
    ANOTHER job of the same video (or another run of the same extraction) handled the same frame /
    image, otherwise not_retried. Nothing is retried automatically. (Job times are stored to the
    second, so "another" is used rather than "a later" one.)"""
    out = []
    blank = {"detection_run_id": "", "image_id": "", "video_timestamp_seconds": "", "video_timestamp_hhmmss": "",
             "target_frame": "", "original_image_path": ""}
    for ext in jobs(db, source_video, camera_id, store_id):
        base = {"extraction_job_id": ext["id"], "source_video": ext["source_video"],
                "store_id": ext.get("store_id") or "", "camera_id": ext.get("camera_id") or ""}
        other_jobs = [j for j in db.extractions(media_id=ext["media_id"], limit=200) if j["id"] != ext["id"]]
        failures = read_failures(ext)
        if ext["state"] == "failed" and ext["error"] and not ext["extracted"] and not failures:
            out.append(base | blank | {"stage": "extraction_job", "error_category": "job_failed",
                                       "error_message": ext["error"],
                                       "retry_status": "retried_ok" if any(j["state"] == "done" for j in other_jobs)
                                       else "not_retried"})
        if failures:
            other_frames = {int(m["frame_number"]) for j in other_jobs for m in read_manifest(j)}
            for fl in failures:
                out.append(base | blank | {
                    "stage": "extraction", "image_id": fl["image_id"], "video_timestamp_seconds": fl["target_seconds"],
                    "video_timestamp_hhmmss": fl["target_hhmmss"], "target_frame": fl["target_frame"],
                    "error_category": _extraction_category(fl),
                    "error_message": fl.get("error_message") or fl.get("reason", ""),
                    "retry_status": "retried_ok" if int(fl["target_frame"]) in other_frames else "not_retried"})
        runs = db.detection_runs(ext["id"])
        results = {r["id"]: results_rows(r, ext) for r in runs}
        for run in runs:
            others = [r for r in runs if r["id"] != run["id"]]
            if run["state"] == "failed" and run["error"] and not run["processed"]:
                out.append(base | blank | {"stage": "detection_run", "detection_run_id": run["id"],
                                           "error_category": "run_refused_or_failed", "error_message": run["error"],
                                           "retry_status": "retried_ok" if any(r["state"] == "done" and r["processed"]
                                                                               for r in others) else "not_retried"})
            seen = set()
            for r in results[run["id"]]:
                if r["processing_status"] == "ok" or r["image_id"] in seen:
                    continue
                seen.add(r["image_id"])
                stage, cat = _inference_category(r)
                retries = [x for o in others for x in results[o["id"]] if x["image_id"] == r["image_id"]]
                retry = ("not_retried" if not retries else "retried_ok"
                         if any(x["processing_status"] == "ok" for x in retries) else "retried_failed")
                out.append(base | blank | {
                    "stage": stage, "detection_run_id": run["id"], "image_id": r["image_id"],
                    "video_timestamp_seconds": r["video_timestamp_seconds"],
                    "video_timestamp_hhmmss": r["video_timestamp_hhmmss"],
                    "original_image_path": r["original_image_path"], "error_category": cat,
                    "error_message": r["review_reason"], "retry_status": retry})
    return out


def report_rows(db: DB, name: str, source_video: str | None = None, camera_id: str | None = None,
                store_id: str | None = None, date: str | None = None) -> list[dict]:
    """screenshot_summary / uniform_compliance_details over all jobs (latest run that processed frames)."""
    from backend.extraction import reports
    out = []
    for ext in jobs(db, source_video, camera_id, store_id):
        out += [r for r in reports.build(name, ext, latest_processed_run(db, ext["id"]))
                if not date or r["recording_date"] == date]
    return out


def storage_status() -> dict:
    return get_storage().status()


# ------------------------------------------------------------------ combined CSV exports
MANIFEST_EXPORT_FIELDS = MANIFEST_FIELDS
RESULTS_EXPORT_FIELDS = RESULT_FIELDS


def manifest_rows(db: DB, source_video: str | None = None, camera_id: str | None = None,
                  store_id: str | None = None, date: str | None = None) -> list[dict]:
    out = []
    for ext in jobs(db, source_video, camera_id, store_id):
        out += [r for r in read_manifest(ext) if not date or _date_of(r["recording_timestamp"]) == date]
    return out


def detection_rows(db: DB, source_video: str | None = None, outcome: str | None = None,
                   processing_status: str | None = None, camera_id: str | None = None, store_id: str | None = None,
                   date: str | None = None) -> list[dict]:
    out = []
    for ext in jobs(db, source_video, camera_id, store_id):
        run = latest_processed_run(db, ext["id"])
        if not run:
            continue
        for r in results_rows(run, ext):
            if outcome and r["prediction"] != outcome:
                continue
            if processing_status and r["processing_status"] != processing_status:
                continue
            if date and _date_of(r["recording_timestamp"]) != date:
                continue
            out.append(r)
    return out
