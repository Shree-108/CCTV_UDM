"""Uniform compliance CSV reports, derived from the stored records of a job and its detection run
(image_manifest.csv + detection_results.csv). No new decision rule: every value comes from the
existing classifier output.

  screenshot_summary.csv          one row per successfully extracted screenshot
  uniform_compliance_details.csv  one row per person found in a screenshot (+ one row per screenshot
                                  whose processing failed, so failures are never silent)

Per person, the existing classifier gives one of three labels:
  uniform      P(uniform) >= threshold (default 0.70)                  -> UNIFORM_DETECTED
  non_uniform  P(non_uniform) >= non-uniform threshold (default 0.60)  -> NON_UNIFORM_DETECTED
  uncertain    neither, or the person could not be judged             -> UNCERTAIN (sent to review)
The model judges the SHIRT APPEARANCE only - not logos, exact colours or the full dress code - so
"compliant" / "non-compliant" in these reports means "uniform shirt detected / not detected" and
nothing more. A processing failure is reported as a failure, never as non-compliance.

Dates: `recording_date` and `recording_timestamp` are filled only when the recording start time of
the video was supplied; otherwise they are blank and `recording_date_status` says "unknown". The
processing date is never used as a recording date.

S3 is not implemented: `original_s3_key` / `annotated_s3_key` are always blank.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from backend.extraction.detect import PROCESSING_FAILED, normalize
from backend.extraction.extractor import row_meta
from backend.extraction.runner import EXTRACTIONS, read_manifest

SUMMARY_FIELDS = ["image_id", "extraction_job_id", "source_video_filename", "store_id", "camera_id", "recording_date",
                  "video_timestamp_seconds", "video_timestamp_hhmmss", "recording_timestamp", "screenshot_filename",
                  "original_image_path", "original_s3_key", "extraction_status", "number_of_people_detected",
                  "number_of_people_classified", "number_of_uniform_compliant", "number_of_uniform_non_compliant",
                  "number_of_uncertain_predictions", "number_of_processing_failures", "overall_image_status",
                  # links / context (after the requested columns)
                  "recording_date_status", "detection_run_id", "annotated_image_path", "confidence_threshold",
                  "model_version", "review_reason"]
DETAIL_FIELDS = ["image_id", "extraction_job_id", "detection_run_id", "store_id", "camera_id", "recording_date",
                 "video_timestamp_hhmmss", "screenshot_filename", "person_index", "person_confidence",
                 "uniform_prediction", "uniform_confidence_score", "confidence_threshold", "compliance_status",
                 "review_status", "annotated_image_path", "annotated_s3_key", "model_version", "processing_status",
                 # links / context
                 "recording_date_status", "video_timestamp_seconds", "recording_timestamp", "source_video_filename",
                 "original_image_path", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "review_reason"]
REPORT_NAMES = ("screenshot_summary", "uniform_compliance_details")
PREDICTION = {"uniform_shirt": "UNIFORM_DETECTED", "not_uniform": "NON_UNIFORM_DETECTED", "uncertain": "UNCERTAIN"}
COMPLIANCE = {"uniform_shirt": "UNIFORM_DETECTED", "not_uniform": "NON_UNIFORM_DETECTED", "uncertain": "REVIEW_REQUIRED"}


def _dates(row: dict) -> dict:
    ts = row.get("recording_timestamp") or ""
    return {"recording_timestamp": ts, "recording_date": ts[:10], "recording_date_status": "known" if ts else "unknown"}


def run_rows(run: dict, ext: dict) -> list[dict]:
    p = EXTRACTIONS / run["extraction_id"] / "detections" / run["id"] / "detection_results.csv"
    if not p.exists():
        return []
    meta = row_meta(ext)
    with open(p, newline="", encoding="utf-8") as f:
        return [normalize(r, run, meta) for r in csv.DictReader(f)]


def summary_rows(ext: dict, run: dict | None) -> list[dict]:
    """One row per screenshot in the job's manifest. Without a run (or for screenshots the run did
    not reach) the detection columns are blank and the status is NOT_ANALYSED."""
    by_image: dict[str, list[dict]] = {}
    for r in run_rows(run, ext) if run else []:
        by_image.setdefault(r["image_id"], []).append(r)
    out = []
    for m in read_manifest(ext):
        rows = by_image.get(m["image_id"], [])
        base = {"image_id": m["image_id"], "extraction_job_id": m["extraction_job_id"],
                "source_video_filename": m["source_video"], "store_id": m["store_id"], "camera_id": m["camera_id"],
                "video_timestamp_seconds": m["video_timestamp_seconds"], "video_timestamp_hhmmss": m["video_timestamp_hhmmss"],
                "screenshot_filename": m["image_filename"], "original_image_path": m["original_image_path"],
                "original_s3_key": "", "extraction_status": m["extraction_status"], **_dates(m),
                "number_of_people_detected": "", "number_of_people_classified": "", "number_of_uniform_compliant": "",
                "number_of_uniform_non_compliant": "", "number_of_uncertain_predictions": "",
                "number_of_processing_failures": "", "overall_image_status": "NOT_ANALYSED",
                "detection_run_id": run["id"] if run and rows else "", "annotated_image_path": "",
                "confidence_threshold": run["threshold"] if run and rows else "",
                "model_version": run["model_version"] if run and rows else "", "review_reason": ""}
        if rows:
            first = rows[0]
            types = [r["detection_type"] for r in rows if r["detection_type"]]
            failed = first["prediction"] == PROCESSING_FAILED
            legacy = first["person_count"] in ("", None) and not failed      # older run: only target shirts were stored
            base |= {"overall_image_status": first["prediction"], "annotated_image_path": first["annotated_image_path"],
                     "review_reason": first["review_reason"],
                     "number_of_processing_failures": int(first["processing_status"] != "ok")}
            if not failed and not legacy:
                base |= {"number_of_people_detected": int(first["person_count"]),
                         "number_of_people_classified": sum(t in ("uniform_shirt", "not_uniform") for t in types),
                         "number_of_uniform_compliant": types.count("uniform_shirt"),
                         "number_of_uniform_non_compliant": types.count("not_uniform"),
                         "number_of_uncertain_predictions": types.count("uncertain")}
            elif legacy:
                base["number_of_uniform_compliant"] = types.count("uniform_shirt")
        out.append(base)
    return out


def detail_rows(ext: dict, run: dict | None) -> list[dict]:
    """One row per person found; a screenshot whose processing failed gets one failure row."""
    if not run:
        return []
    names = {m["image_id"]: m["image_filename"] for m in read_manifest(ext)}
    out = []
    for r in run_rows(run, ext):
        failed = r["prediction"] == PROCESSING_FAILED
        if not r["detection_type"] and not failed:
            continue                                           # screenshot without people: nothing to list
        t = r["detection_type"]
        out.append({"image_id": r["image_id"], "extraction_job_id": r["extraction_job_id"],
                    "detection_run_id": r["detection_run_id"], "store_id": r["store_id"], "camera_id": r["camera_id"],
                    "video_timestamp_hhmmss": r["video_timestamp_hhmmss"],
                    "screenshot_filename": names.get(r["image_id"], Path(r["original_image_path"]).name),
                    "person_index": r["detection_index"], "person_confidence": r["person_confidence"],
                    "uniform_prediction": PREDICTION.get(t, ""), "uniform_confidence_score": r["confidence"],
                    "confidence_threshold": r["confidence_threshold"],
                    "compliance_status": PROCESSING_FAILED if failed else COMPLIANCE.get(t, ""),
                    "review_status": r["review_status"], "annotated_image_path": r["annotated_image_path"],
                    "annotated_s3_key": "", "model_version": r["model_version"],
                    "processing_status": r["processing_status"], **_dates(r),
                    "video_timestamp_seconds": r["video_timestamp_seconds"], "source_video_filename": r["source_video"],
                    "original_image_path": r["original_image_path"], "bbox_xmin": r["bbox_xmin"],
                    "bbox_ymin": r["bbox_ymin"], "bbox_xmax": r["bbox_xmax"], "bbox_ymax": r["bbox_ymax"],
                    "review_reason": r["review_reason"]})
    return out


def build(name: str, ext: dict, run: dict | None) -> list[dict]:
    return summary_rows(ext, run) if name == "screenshot_summary" else detail_rows(ext, run)


def fields(name: str) -> list[str]:
    return SUMMARY_FIELDS if name == "screenshot_summary" else DETAIL_FIELDS


def write_run_reports(ext: dict, run: dict) -> None:
    """Snapshot of both reports in the run's folder, written when the run finishes."""
    out = EXTRACTIONS / ext["id"] / "detections" / run["id"]
    for name in REPORT_NAMES:
        tmp = out / f".{name}.csv.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields(name))
            w.writeheader()
            w.writerows(build(name, ext, run))
        os.replace(tmp, out / f"{name}.csv")            # atomic: readers see the old state or the whole file
