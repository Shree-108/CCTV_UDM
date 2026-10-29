"""The cumulative uniform compliance CSV: ONE file for every successfully processed video.

Same six columns, same order and same row rules as the per-job file (backend.extraction.business_csv):
  Store Name, Date, Time Stamp, img name, confidence score, remark (Example)
One row per sampled screenshot. Time Stamp is the position of the frame in the source video (HH:MM:SS);
Date is the recording date only when the recording start time was supplied, otherwise blank.

  storage/exports/uniform_compliance_master.csv          the business file (header written once)
  storage/exports/uniform_compliance_master.ledger.jsonl internal: which job / screenshot each append
                                                         came from, and the file size after it

Rules
  * Rows are only ever APPENDED. Earlier rows are never rewritten or reordered.
  * A screenshot is included once: the key is (extraction job id, image id), kept in the ledger - never
    as an extra CSV column. Retrying a job, a second detection run on the same screenshots, or a
    repeated sync appends nothing.
  * Only finished detection runs (state "done") are appended. A run that failed on every image, was
    cancelled or interrupted is not; its job is appended once a later run finishes.
  * One writer at a time: a lock inside the process plus an OS file lock (released by the OS if the
    process dies).
  * Crash safety: the ledger records the file size after every append. Bytes after the last recorded
    size belong to an append whose ledger entry was never written (a crash in between); they are cut
    off and those rows are appended again, so a crash can neither lose nor duplicate rows. A file that
    is SHORTER than the ledger says was changed outside the application: nothing is written and the
    error is reported until someone restores it.
  * Local disk only. S3 is not implemented (backend.core.storage); nothing is uploaded anywhere.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path

from backend.core.config import STORAGE
from backend.core.storage import rel
from backend.database.db import now_iso
from backend.extraction.business_csv import BUSINESS_FIELDS, business_row
from backend.extraction.screenshots import screenshot_results

log = logging.getLogger(__name__)
EXPORTS = STORAGE / "exports"
MASTER_NAME = "uniform_compliance_master.csv"
_lock = threading.Lock()
_last_error: dict | None = None


class MasterCsvError(RuntimeError):
    """The master CSV could not be written; the job's own results are unaffected."""


def master_path() -> Path:
    return EXPORTS / MASTER_NAME


def ledger_path() -> Path:
    return EXPORTS / "uniform_compliance_master.ledger.jsonl"


@contextmanager
def _file_lock():
    EXPORTS.mkdir(parents=True, exist_ok=True)
    with open(EXPORTS / ".uniform_compliance_master.lock", "a+b") as f:
        if os.name == "nt":
            import msvcrt
            while True:
                f.seek(0)
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)     # retries for ~10 s, then raises
                    break
                except OSError:
                    continue
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def _ledger() -> list[dict]:
    p = ledger_path()
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:          # a torn last line (crash while writing it): the append it
                break                             # describes is cut off from the CSV and redone
    return out


def _recover(entries: list[dict]) -> None:
    """Make the CSV match the ledger (see "Crash safety" above)."""
    p = master_path()
    expected = entries[-1]["end_offset"] if entries else 0
    size = p.stat().st_size if p.exists() else 0
    if size < expected:
        raise MasterCsvError(f"{MASTER_NAME} is shorter than its ledger records ({size} < {expected} bytes): it was "
                             "changed outside the application. Restore it from a backup; nothing was written.")
    if size > expected:
        log.warning("%s: removing %d bytes of an append that was never recorded (interrupted write)",
                    MASTER_NAME, size - expected)
        with open(p, "r+b") as f:
            f.truncate(expected)
            f.flush()
            os.fsync(f.fileno())
    lp = ledger_path()                     # drop a torn trailing ledger line, if any
    if lp.exists():
        good = "".join(json.dumps(e) + "\n" for e in entries)
        if lp.read_text(encoding="utf-8") != good:
            lp.write_text(good, encoding="utf-8")


def _rows_bytes(rows: list[dict], header: bool) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=BUSINESS_FIELDS)
    if header:
        w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def append(ext: dict, run: dict) -> dict:
    """Append the screenshots of one finished job. Returns {"status": appended | already_included |
    not_eligible, "rows": n}. Raises MasterCsvError when the file cannot be written."""
    global _last_error
    if run.get("state") != "done":
        return {"status": "not_eligible", "rows": 0}
    try:
        with _lock, _file_lock():
            entries = _ledger()
            _recover(entries)
            seen = {(e["job_id"], i) for e in entries for i in e["image_ids"]}
            new = [r for r in screenshot_results(ext, run) if (ext["id"], r["image_id"]) not in seen]
            if not new:
                return {"status": "already_included", "rows": 0}
            p = master_path()
            start = p.stat().st_size if p.exists() else 0
            with open(p, "ab") as f:
                f.write(_rows_bytes([business_row(r) for r in new], header=start == 0))
                f.flush()
                os.fsync(f.fileno())
                end = f.tell()
            entry = {"job_id": ext["id"], "run_id": run["id"], "source_video": ext["source_video"],
                     "image_ids": [r["image_id"] for r in new], "rows": len(new), "start_offset": start,
                     "end_offset": end, "appended_at": now_iso()}
            with open(ledger_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
    except MasterCsvError as e:
        _last_error = {"job_id": ext["id"], "run_id": run.get("id"), "message": str(e), "at": now_iso()}
        raise
    except OSError as e:                  # disk full, permissions, file locked by another program, ...
        _last_error = {"job_id": ext["id"], "run_id": run.get("id"), "at": now_iso(),
                       "message": f"{MASTER_NAME} could not be written ({e.strerror or e.__class__.__name__}). "
                                  "The job's own results are saved; its rows are appended on the next sync."}
        raise MasterCsvError(_last_error["message"]) from e
    _last_error = None
    log.info("%s: appended %d rows of %s (run %s)", MASTER_NAME, len(new), ext["id"], run["id"])
    return {"status": "appended", "rows": len(new)}


def eligible_run(db, ext: dict) -> dict | None:
    """The run whose rows go into the master file: the latest finished run of the job."""
    return next((r for r in db.detection_runs(ext["id"]) if r["state"] == "done" and r["processed"] > 0), None)


def _jobs(db) -> list[dict]:
    return sorted(db.extractions(limit=100_000), key=lambda j: (j["created_at"], j["id"]))


def sync(db) -> dict:
    """Append every finished job that is not in the master file yet, oldest first (backfill at start-up,
    and recovery after a failed append). Never raises; a failure is reported by status()."""
    appended = rows = 0
    for ext in _jobs(db):
        run = eligible_run(db, ext)
        if not run:
            continue
        try:
            r = append(ext, run)
        except MasterCsvError as e:
            log.error("master CSV sync stopped: %s", e)
            break
        if r["status"] == "appended":
            appended, rows = appended + 1, rows + r["rows"]
    return {"jobs_appended": appended, "rows_appended": rows}


def status(db) -> dict:
    """What the file contains and what is still missing - never claims more than is on disk."""
    p = master_path()
    entries = _ledger()
    included = {e["job_id"] for e in entries}
    pending = [{"job_id": j["id"], "source_video": j["source_video"]} for j in _jobs(db)
               if j["id"] not in included and eligible_run(db, j)]
    shown = rel(p)                         # project-relative; never a full path outside the project
    shown = shown if not Path(shown).is_absolute() else f"<storage>/exports/{MASTER_NAME}"
    return {"file_name": MASTER_NAME, "path": shown, "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else 0, "rows": sum(e["rows"] for e in entries),
            "videos": len(included), "fields": BUSINESS_FIELDS,
            "updated_at": entries[-1]["appended_at"] if entries else None, "pending_jobs": pending,
            "last_error": _last_error, "storage": "local disk (S3 is not implemented)",
            "csv_url": "/api/uniform-compliance/master.csv"}


def included_jobs() -> dict[str, dict]:
    """job id -> its ledger entries' summary (rows, run, appended_at)."""
    out: dict[str, dict] = {}
    for e in _ledger():
        x = out.setdefault(e["job_id"], {"rows": 0, "run_id": e["run_id"], "appended_at": e["appended_at"]})
        x["rows"] += e["rows"]
    return out
