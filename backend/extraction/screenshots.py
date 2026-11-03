"""One structured result per SAMPLED screenshot of an extraction job, with a human-readable remark.

Everything comes from the files the job and its detection run wrote (the plan's samples,
image_manifest.csv, failures.csv, detection_results.csv) - nothing is re-run or estimated.
Every planned sample gets a record, also when its extraction failed, when nobody was found, or
when the detection run could not process the image.

Per person the remark uses the existing classifier label:
  uniform_shirt  P(uniform) >= threshold                -> "detected wearing the designated uniform shirt"
  not_uniform    P(non_uniform) >= non-uniform threshold -> "detected without the designated uniform shirt"
  uncertain      neither / could not be judged           -> "requires manual review"
The model judges shirt appearance only, so the remarks say nothing about logos, exact colours or the
rest of the dress code. A failed step is reported as a failure, never as non-compliance.
Persons are numbered in the order of detection_results.csv (detection_index); the numbers are per
screenshot and do not identify anyone across screenshots.
"""
from __future__ import annotations

from backend.extraction.detect import PROCESSING_FAILED
from backend.extraction.extractor import FRAME_UNREADABLE  # failure category = status: corrupted / blank frame
from backend.extraction.plan import clock_time, hhmmss
from backend.extraction.reports import run_rows
from backend.extraction.runner import read_failures, read_manifest

PERSON_REMARK = {"uniform_shirt": "detected wearing the designated uniform shirt",
                 "not_uniform": "detected without the designated uniform shirt",
                 "uncertain": "requires manual review"}
EXTRACTION_FAILED, NOT_ANALYSED = "extraction_failed", "not_analysed"
# a saved screenshot without a prediction because detection is still to come / ended without processing it
ANALYSIS_PENDING, DETECTION_FAILED = "pending", "detection_failed"


def remark(types: list[str], prediction: str, processing_status: str, reasons: list[str] | None = None) -> str:
    """`types` = detection_type of each person, in detection_index order; `reasons` = why each uncertain
    person could not be judged (person_reason; '' when unknown, e.g. runs made before it was recorded)."""
    if processing_status == EXTRACTION_FAILED:
        return "Uniform status could not be determined because the screenshot could not be extracted."
    if processing_status == FRAME_UNREADABLE:
        return ("Uniform status could not be determined because the video frame at this position is unreadable "
                "(corrupted or blank); no prediction was made.")
    if processing_status == NOT_ANALYSED:
        return "Not analysed yet: no detection run has processed this screenshot."
    if processing_status == ANALYSIS_PENDING:
        return "Analysis pending: the uniform model has not processed this screenshot yet."
    if processing_status == DETECTION_FAILED:
        return "Uniform status could not be determined because the detection run failed before analysing this screenshot."
    if prediction == PROCESSING_FAILED or processing_status == "failed":
        return "Uniform status could not be determined because processing failed."
    if not types:
        text = "No person detected in this screenshot."
    else:
        why = reasons or [""] * len(types)
        text = "; ".join(f"Person {i} {PERSON_REMARK.get(t, 'requires manual review')}"
                         + (f" ({r})" if t == "uncertain" and r else "")
                         for i, (t, r) in enumerate(zip(types, why), 1)) + "."
    if processing_status == "annotation_failed":
        text += " (The annotated copy could not be saved.)"
    return text


def _record(base: dict, rows: list[dict] | None, run: dict | None) -> dict:
    status = base.pop("_status")
    if rows is None:
        return base | {"prediction": "", "processing_status": status, "people_detected": None,
                       "uniform_count": None, "non_uniform_count": None, "uncertain_count": None,
                       "confidence_scores": [], "people": [], "review_reason": "", "annotated_image_path": "",
                       "detection_run_id": "", "model_version": "", "confidence_threshold": None,
                       "remark": remark([], "", status)}
    first = rows[0]
    people = sorted((r for r in rows if r["detection_type"]), key=lambda r: int(r["detection_index"] or 0))
    types = [r["detection_type"] for r in people]
    failed = first["prediction"] == PROCESSING_FAILED
    counted = not failed and first["person_count"] not in ("", None)       # older runs stored target shirts only
    score = lambda v: float(v) if v not in ("", None) else None             # noqa: E731
    return base | {
        "prediction": first["prediction"], "processing_status": first["processing_status"],
        "people_detected": int(first["person_count"]) if counted else None,
        "uniform_count": types.count("uniform_shirt") if counted else None,
        "non_uniform_count": types.count("not_uniform") if counted else None,
        "uncertain_count": types.count("uncertain") if counted else None,
        "confidence_scores": [score(r["confidence"]) for r in people],      # uniform classifier P(uniform) per person
        "people": [{"person": i, "type": r["detection_type"], "confidence": score(r["confidence"]),
                    "person_confidence": score(r["person_confidence"]), "reason": r.get("person_reason") or ""}
                   for i, r in enumerate(people, 1)],
        "review_reason": first["review_reason"], "annotated_image_path": first["annotated_image_path"],
        "detection_run_id": run["id"] if run else "", "model_version": first["model_version"],
        "confidence_threshold": score(first["confidence_threshold"]),
        "remark": remark(types, first["prediction"], first["processing_status"],
                         [r.get("person_reason") or "" for r in people]),
    }


def screenshot_results(ext: dict, run: dict | None, unanalysed: str = NOT_ANALYSED) -> list[dict]:
    """One record per sampled screenshot of `ext`, in time order; `run` = the detection run to report.
    `unanalysed` = what a saved screenshot without a prediction is: NOT_ANALYSED (no detection requested),
    ANALYSIS_PENDING (detection queued / running) or DETECTION_FAILED (the run ended without it)."""
    by_image: dict[str, list[dict]] = {}
    for r in run_rows(run, ext) if run else []:
        by_image.setdefault(r["image_id"], []).append(r)
    saved = {m["image_id"]: m for m in read_manifest(ext)}
    failed = {f["image_id"]: f for f in read_failures(ext) if f.get("image_id")}
    planned = (ext.get("plan") or {}).get("samples")
    if planned is None:                         # large / older jobs do not keep a per-sample list
        planned = [{"image_id": i} for i in [*saved, *(k for k in failed if k not in saved)]]
    out = []
    for s in planned:
        iid = s["image_id"]
        m, f = saved.get(iid), failed.get(iid)
        secs = float(m["video_timestamp_seconds"] if m else f["target_seconds"] if f else s["target_seconds"])
        base = {"image_id": iid, "extraction_job_id": ext["id"], "source_video": ext["source_video"],
                "store_id": ext.get("store_id") or "", "camera_id": ext.get("camera_id") or "",
                "image_filename": m["image_filename"] if m else "",
                "original_image_path": m["original_image_path"] if m else "",
                "video_timestamp_seconds": secs,
                "video_timestamp_hhmmss": (m["video_timestamp_hhmmss"] if m else "") or hhmmss(secs),
                "recording_timestamp": (m.get("recording_timestamp") if m else "")
                or clock_time(ext.get("recording_start_time"), secs),
                "frame_number": int(m["frame_number"]) if m else int(f["target_frame"]) if f else s.get("target_frame"),
                "sample_type": s.get("sample_type") or (m or f or {}).get("sample_type", ""),
                "extraction_status": "saved" if m else "failed" if f else "not_extracted",
                "extraction_error": (f or {}).get("error_message", ""),
                "_status": unanalysed if m else FRAME_UNREADABLE
                if (f or {}).get("error_category") == FRAME_UNREADABLE else EXTRACTION_FAILED}
        out.append(_record(base, by_image.get(iid) if m else None, run))
    return sorted(out, key=lambda x: x["video_timestamp_seconds"])
