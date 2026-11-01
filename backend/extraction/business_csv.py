"""The business-facing uniform compliance CSV of one extraction job (format of "Sample CCTV CSV.xlsx").

Exactly six columns, in this order, and ONE row per sampled screenshot (never one per person):
  Store Name        the store name / ID the user supplied, as typed; blank when unknown
  Date              recording date (YYYY-MM-DD) - only when the recording start time was supplied;
                    blank otherwise (the processing date is never used)
  Time Stamp        position of the frame in the video, HH:MM:SS (not a clock time)
  img name          the extracted screenshot file the model analysed; blank when extraction failed
  confidence score  the uniform classifier's P(uniform) per person as a percentage, e.g. "92.5%", or
                    "Person 1: 92.5%; Person 2: 41.0%" for several people. Never the person detector's
                    confidence, and not a calibrated probability. Blank when there is no prediction.
  remark (Example)  generated from the screenshot's actual detections (backend.extraction.screenshots)

Person-level detail stays in detection_results.csv / uniform_compliance_details.csv.
"""
from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from urllib.parse import quote

from backend.extraction.runner import EXTRACTIONS
from backend.extraction.screenshots import ANALYSIS_PENDING, DETECTION_FAILED, NOT_ANALYSED, screenshot_results

BUSINESS_FIELDS = ["Store Name", "Date", "Time Stamp", "img name", "confidence score", "remark (Example)"]
UNSCORED = "not scored"          # a person the classifier could not score (e.g. too small) among scored people


def file_name(job_id: str) -> str:
    return f"uniform_compliance_job_{job_id}.csv"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def confidence_text(scores: list[float | None]) -> str:
    if not any(s is not None for s in scores):
        return ""
    if len(scores) == 1:
        return _pct(scores[0])
    return "; ".join(f"Person {i}: {_pct(s) if s is not None else UNSCORED}" for i, s in enumerate(scores, 1))


def business_row(r: dict) -> dict:
    """One CSV row from a screenshots.screenshot_results record."""
    return {"Store Name": r["store_id"] or "", "Date": (r["recording_timestamp"] or "")[:10],
            "Time Stamp": r["video_timestamp_hhmmss"][:8], "img name": r["image_filename"],
            "confidence score": confidence_text(r["confidence_scores"]), "remark (Example)": r["remark"]}


def unanalysed_status(run: dict | None, latest: dict | None, pending: bool = False) -> str:
    """How to describe saved screenshots that have no prediction (see screenshots.screenshot_results)."""
    if run or not (latest or pending):
        return NOT_ANALYSED
    if pending or latest["state"] in ("queued", "running"):
        return ANALYSIS_PENDING
    return DETECTION_FAILED if latest["state"] in ("failed", "cancelled", "interrupted") else NOT_ANALYSED


THUMB_WIDTH = 360                 # hover previews: resized in memory by the image endpoints, never saved


def evidence_links(job_id: str, r: dict) -> dict:
    """URLs of the evidence of ONE CSV row (a screenshots.screenshot_results record). They are addressed by
    job id + image id + the run that produced the row - never by the file name alone - and the image
    endpoints only serve files listed in that job's manifest / that run's results."""
    name, ann, rid = r["image_filename"], r["annotated_image_path"], r["detection_run_id"]
    frame = f"/api/extractions/{quote(job_id)}/frames/{quote(name)}" if name else None
    return {"original_url": frame, "thumb_url": f"{frame}?w={THUMB_WIDTH}" if frame else None,
            "annotated_url": f"/api/detection-runs/{quote(rid)}/annotated/{quote(Path(ann).name)}" if ann and rid else None,
            "evidence_url": f"/api/extractions/{quote(job_id)}/evidence/{quote(r['image_id'])}"
                            + (f"?run_id={quote(rid)}" if rid else "")}


def records(ext: dict, run: dict | None, unanalysed: str = NOT_ANALYSED) -> list[dict]:
    """Preview records: the six CSV values plus links to the screenshot evidence."""
    out = []
    for r in screenshot_results(ext, run, unanalysed):
        out.append({"row": business_row(r), "job_id": ext["id"], "run_id": run["id"] if run else None,
                    "image_id": r["image_id"], "processing_status": r["processing_status"],
                    "prediction": r["prediction"], "extraction_status": r["extraction_status"],
                    "recording_timestamp": r["recording_timestamp"], **evidence_links(ext["id"], r)})
    return out


def to_csv(ext: dict, run: dict | None, unanalysed: str = NOT_ANALYSED) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=BUSINESS_FIELDS)
    w.writeheader()
    w.writerows(business_row(r) for r in screenshot_results(ext, run, unanalysed))
    return buf.getvalue()


def saved_path(ext: dict, run: dict) -> Path:
    return EXTRACTIONS / ext["id"] / "detections" / run["id"] / file_name(ext["id"])


def write(ext: dict, run: dict) -> Path:
    """Snapshot in the run's own folder (a new folder per run, so earlier reports are never overwritten)."""
    p = saved_path(ext, run)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(to_csv(ext, run), encoding="utf-8", newline="")
    os.replace(tmp, p)
    return p
