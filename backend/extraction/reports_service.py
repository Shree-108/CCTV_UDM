"""Latest-analysis, monthly and historical business reports - all derived from persisted records.

Unit: one analysis = one extraction job (its sampled screenshots) + the detection run that reports it.
The reporting run is the one recorded in the master CSV ledger when the job is already in the master
file, otherwise the job's latest finished run - so the latest, monthly and historical views always show
the same rows as uniform_compliance_master.csv. Rows use the six business columns (business_csv).

Analysis status
  completed     a detection run finished ("done")
  failed        extraction failed / was cancelled / interrupted, or the latest run failed and none finished
  in_progress   extraction or detection queued / running
  not_analysed  screenshots saved, but detection was never run (not counted as failed)

Months ("YYYY-MM", only months that contain real records - none are invented)
  completion (default)  analyses by the date their run finished (failed ones: when they stopped);
                        "videos received" by the date the video was received.
  recording             analyses whose recording start time was supplied, by that recording date;
                        analyses without a known recording date are left out of this view.
Dates are converted to the server's local time zone before taking the month (REPORT_TZ, default:
the time zone of this PC), so a run that finished at 23:30 local time on 31 October is in October.
"""
from __future__ import annotations

from datetime import datetime, tzinfo

from backend.extraction import master_csv
from backend.extraction.business_csv import BUSINESS_FIELDS, business_row, records as job_evidence
from backend.extraction.screenshots import screenshot_results

ACTIVE = ("queued", "running")
STOPPED = ("failed", "cancelled", "interrupted")
GROUPINGS = ("completion", "recording")
REPORT_TZ: tzinfo | None = None          # None = the server's local time zone


def _local(ts: str | None) -> datetime | None:
    if not ts:
        return None
    d = datetime.fromisoformat(ts)
    return d.astimezone(REPORT_TZ) if d.tzinfo else d


def _month(ts: str | None) -> str | None:
    d = _local(ts)
    return f"{d:%Y-%m}" if d else None


def analysis(db, ext: dict, ledger: dict[str, dict]) -> dict:
    """Everything the reports need about one analysis, from the database and its stored files."""
    runs = db.detection_runs(ext["id"])
    entry = ledger.get(ext["id"])
    run = next((r for r in runs if r["id"] == entry["run_id"]), None) if entry else master_csv.eligible_run(db, ext)
    latest = runs[0] if runs else None
    media = db.media(ext["media_id"]) or {}
    if ext["state"] in ACTIVE or (latest and latest["state"] in ACTIVE):
        status = "in_progress"
    elif run:
        status = "completed"
    elif ext["state"] in STOPPED or (latest and latest["state"] in STOPPED):
        status = "failed"
    else:
        status = "not_analysed"
    shots = screenshot_results(ext, run) if run else []
    error = None
    if status == "failed":
        error = (latest or {}).get("error") or ext.get("error") or f"extraction {ext['state']}"
    ended = run["finished_at"] if run else (latest or {}).get("finished_at") or ext.get("finished_at")
    return {
        "job_id": ext["id"], "run_id": run["id"] if run else None, "media_id": ext["media_id"],
        "source_video": ext["source_video"], "source": ext.get("source") or "upload",
        "store_id": ext.get("store_id"), "camera_id": ext.get("camera_id"),
        "recording_start_time": ext.get("recording_start_time"),
        "recording_date": (ext.get("recording_start_time") or "")[:10] or None,
        "received_at": media.get("created_at") or ext["created_at"], "size_bytes": media.get("size_bytes"),
        "duration_seconds": media.get("duration_sec"), "created_at": ext["created_at"],
        "started_at": ext.get("started_at"), "completed_at": run["finished_at"] if run else None,
        "ended_at": ended if status in ("completed", "failed") else None,
        "status": status, "error": error, "sampling_strategy": ext.get("sampling_strategy") or "interval",
        "random_seed": ext.get("random_seed"), "screenshots_planned": ext["planned"],
        "screenshots_extracted": ext["extracted"],
        "screenshots_analysed": sum(s["processing_status"] == "ok" for s in shots),
        "uniform": sum(s["prediction"] == "UNIFORM_DETECTED" for s in shots),
        "no_uniform": sum(s["prediction"] == "NO_UNIFORM_DETECTED" for s in shots),
        "review": sum(s["prediction"] == "REVIEW_REQUIRED" for s in shots),
        "processing_failed": sum(s["prediction"] == "PROCESSING_FAILED" for s in shots),
        "in_master_csv": bool(entry), "_ext": ext, "_run": run, "_shots": shots,
    }


def all_analyses(db) -> list[dict]:
    ledger = master_csv.included_jobs()
    jobs = sorted(db.extractions(limit=100_000), key=lambda j: (j["created_at"], j["id"]))
    return [analysis(db, j, ledger) for j in jobs]



def _month_key(a: dict, by: str) -> str | None:
    if by == "recording":
        return a["recording_date"][:7] if a["recording_date"] else None
    return _month(a["ended_at"])


def rows(a: dict) -> list[dict]:
    """Business rows of a completed analysis (exactly what the master CSV holds for it)."""
    return [business_row(s) for s in a["_shots"]] if a["status"] == "completed" else []


def _ordered(items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda a: (a["completed_at"] or "", a["job_id"]))


def public(a: dict) -> dict:
    return {k: v for k, v in a.items() if not k.startswith("_")}


def overview(db) -> dict:
    items = all_analyses(db)
    done = _ordered([a for a in items if a["status"] == "completed"])
    latest = done[-1] if done else None
    newest = max(items, key=lambda a: (a["created_at"], a["job_id"])) if items else None
    received = db.media_list(kind="video")
    return {
        "summary": {"videos_received": len(received), "analyses": len(items),
                    "analyses_completed": len(done), "analyses_failed": sum(a["status"] == "failed" for a in items),
                    "analyses_in_progress": sum(a["status"] == "in_progress" for a in items),
                    "analyses_not_analysed": sum(a["status"] == "not_analysed" for a in items),
                    "screenshots_extracted": sum(a["screenshots_extracted"] for a in items),
                    "screenshots_analysed": sum(a["screenshots_analysed"] for a in items),
                    "business_records": sum(len(rows(a)) for a in done)},
        "latest": latest_report(latest) if latest else None,
        # the newest job, when it is not the latest completed one (failed / still running) - never shown as its result
        "newest_other": public(newest) if newest and newest is not latest and newest["status"] != "completed" else None,
        "months": {by: months(items, received, by)["months"] for by in GROUPINGS},
        "grouping_rules": RULES,
    }


RULES = {"completion": "Analyses by the date their run finished (failed analyses: the date they stopped); "
                       "videos received by the date the video was received. Server local time.",
         "recording": "Only analyses whose recording start time was supplied, by that recording date."}


def latest_report(a: dict) -> dict:
    ev = job_evidence(a["_ext"], a["_run"])
    return public(a) | {"fields": BUSINESS_FIELDS, "records": ev,
                        "csv_url": f"/api/extractions/{a['job_id']}/uniform_compliance.csv?run_id={a['run_id']}"}


def months(items: list[dict], received: list[dict], by: str) -> dict:
    keys = {_month_key(a, by) for a in items} - {None}
    if by == "completion":
        keys |= {_month(m["created_at"]) for m in received} - {None}
    return {"by": by, "months": sorted(keys, reverse=True)}


def month_report(db, month: str, by: str = "completion") -> dict:
    items = all_analyses(db)
    mine = [a for a in items if _month_key(a, by) == month]
    done = _ordered([a for a in mine if a["status"] == "completed"])
    if by == "completion":
        received = len([m for m in db.media_list(kind="video") if _month(m["created_at"]) == month])
    else:
        received = len({a["media_id"] for a in mine})
    recs = [r for a in done for r in rows(a)]
    return {
        "month": month, "by": by, "rule": RULES[by], "videos_received": received, "analyses": len(mine),
        "analyses_completed": len(done), "analyses_failed": sum(a["status"] == "failed" for a in mine),
        "analyses_in_progress": sum(a["status"] == "in_progress" for a in mine),
        "screenshots_extracted": sum(a["screenshots_extracted"] for a in mine),
        "screenshots_analysed": sum(a["screenshots_analysed"] for a in mine),
        "uniform": sum(a["uniform"] for a in done), "no_uniform": sum(a["no_uniform"] for a in done),
        "review": sum(a["review"] for a in done), "processing_failed": sum(a["processing_failed"] for a in done),
        "business_records": len(recs), "fields": BUSINESS_FIELDS,
        "analyses_list": [public(a) for a in sorted(mine, key=lambda a: (a["ended_at"] or "", a["job_id"]), reverse=True)],
        "csv_url": f"/api/reports/months/{month}.csv?by={by}",
        "_rows": recs,
    }


def history_records(db, month: str | None = None, by: str = "completion", store: str | None = None,
                    video: str | None = None, result: str | None = None, status: str | None = None,
                    q: str | None = None) -> list[dict]:
    """Every business row of every completed analysis (= the master CSV), with what it belongs to."""
    out = []
    for a in _ordered([a for a in all_analyses(db) if a["status"] == "completed"]):
        if month and _month_key(a, by) != month:
            continue
        if store and (a["store_id"] or "") != store:
            continue
        if video and a["source_video"] != video:
            continue
        for s, row in zip(a["_shots"], rows(a)):
            if result and s["prediction"] != result:
                continue
            if status and s["processing_status"] != status:
                continue
            if q and q.lower() not in " ".join(row.values()).lower() + a["source_video"].lower():
                continue
            out.append({"row": row, "job_id": a["job_id"], "run_id": a["run_id"], "source_video": a["source_video"],
                        "image_id": s["image_id"], "prediction": s["prediction"], "processing_status": s["processing_status"],
                        "month": _month_key(a, by), "completed_at": a["completed_at"], "in_master_csv": a["in_master_csv"]})
    return out
