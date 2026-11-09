"""Cart batches from S3: every video in s3://<bucket>/<MM-YYYY>/<DD-MM-YYYY>/<cart>/input_data/ goes
through the quick sampled analysis (the same ExtractionRunner path as the dashboard, so the jobs also
reach the master CSV). Results go to the processing day's output folder (today in Asia/Kolkata, not the
input day folder), each video's rows appended and verified (s3_output.py):

    <MM-YYYY>/<DD-MM-YYYY>/card_<N>/output_data/card<N>_YYYY_MM_DD.csv          the day's CSV
    <MM-YYYY>/<DD-MM-YYYY>/card_<N>/output_data/card<N>_YYYY_MM_DD_pic<K>.jpg   each row's detection image

The cart's <cart>.xlsx (one row per sampled screenshot of all its videos) is written locally only.

The recording time is read from CCTV file names like 20261029_20261029183009_20261029183120_183009.mp4.
Only videos are analysed (screenshots are sampled from videos); other files are listed as skipped.
A screenshot that could not be analysed gets a blank score - it is never reported as non-uniform.
Everything else stays local: the jobs' own files (storage/extractions/) and, per batch,
storage/s3_batches/<id>/summary.json, <cart>/<cart>.xlsx and <cart>/pictures.csv (which picN is which screenshot).
One batch runs at a time; batches are kept in memory and in summary.json.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from datetime import date, datetime
from pathlib import Path

from backend.analysis.media import kind_of, new_upload_path, probe
from backend.core import s3
from backend.core.config import STORAGE
from backend.database.db import DB, now_iso
from backend.extraction import cart_report
from backend.extraction.runner import ExtractionRunner
from backend.extraction.screenshots import DETECTION_FAILED, screenshot_results

log = logging.getLogger(__name__)
BATCHES = STORAGE / "s3_batches"
POLL_SECONDS = 1.0
XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(2 ** 20):
            h.update(chunk)
    return h.hexdigest()


class S3BatchRunner:
    def __init__(self, db: DB, extractions: ExtractionRunner, client_factory=s3.client):
        self.db, self.extractions, self.client_factory = db, extractions, client_factory
        self.batches: dict[str, dict] = {}
        self.q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="s3-batch-worker", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ API
    def check(self, bucket: str) -> dict:
        try:
            return s3.check(self.client_factory(), bucket)
        except s3.S3Unavailable as e:
            return {"ok": False, "bucket": bucket, "credentials": None, "message": str(e)}

    def create(self, day: date, carts: list[str], bucket: str, threshold: float | None = None) -> dict:
        bid = f"s3-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        batch = {"id": bid, "created_at": now_iso(), "finished_at": None, "state": "queued", "error": None,
                 "bucket": bucket, "date": f"{day:%d-%m-%Y}", "threshold": threshold,
                 "carts": [{"cart": c, "state": "queued", "error": None, "report": None,
                            "input": f"s3://{bucket}/{s3.input_prefix(day, c)}",
                            "output": f"s3://{bucket}/{s3.daily_csv_key(s3.today(), c)}", "files": []}
                           for c in carts]}
        with self._lock:
            self.batches[bid] = batch
        self._save(batch)
        self.q.put((bid, day))
        return self.get(bid)

    def get(self, bid: str) -> dict | None:
        with self._lock:
            b = self.batches.get(bid)
            return copy.deepcopy(b) if b else None

    def list(self) -> list[dict]:
        with self._lock:
            return [copy.deepcopy(b) for b in sorted(self.batches.values(), key=lambda b: b["id"], reverse=True)]

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------ worker
    def _save(self, batch: dict) -> None:
        d = BATCHES / batch["id"]
        d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            text = json.dumps(batch, indent=2)
        (d / "summary.json").write_text(text, encoding="utf-8")

    def _set(self, obj: dict, **fields) -> None:
        with self._lock:
            obj.update(fields)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                bid, day = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = self.batches[bid]
            self._set(batch, state="running")
            try:
                s3c = self.client_factory()
                for cart in batch["carts"]:
                    self._run_cart(s3c, batch, cart, day)
                failed = all(c["state"] == "failed" for c in batch["carts"])
                self._set(batch, state="failed" if failed else "done",
                          error="Every cart failed (see each cart's error)" if failed else None)
            except Exception as e:  # noqa: BLE001 - recorded on the batch
                log.exception("S3 batch %s failed", bid)
                self._set(batch, state="failed", error=str(e))
            self._set(batch, finished_at=now_iso())
            self._save(batch)

    def _run_cart(self, s3c, batch: dict, cart: dict, day: date) -> None:
        bucket = batch["bucket"]
        self._set(cart, state="running")
        try:
            keys = s3.list_keys(s3c, bucket, s3.input_prefix(day, cart["cart"]))
        except Exception as e:  # noqa: BLE001
            self._set(cart, state="failed", error=f"Could not list {cart['input']}: {e}")
            return
        if not keys:
            self._set(cart, state="failed", error=f"No files found in {cart['input']}")
            return
        records: list[dict] = []
        for key in keys:
            if self._stop.is_set():
                break
            f = {"key": key, "name": key.rsplit("/", 1)[-1], "state": "queued", "error": None,
                 "recording_start_time": None, "extraction_id": None, "detection_run_id": None, "screenshots": 0}
            with self._lock:
                cart["files"].append(f)
            if kind_of(f["name"]) != "video":
                self._set(f, state="skipped", error="Not a video file; screenshots are sampled from videos only")
                continue
            try:
                records += self.analyse_file(s3c, batch["bucket"], batch["threshold"], cart["cart"], f,
                                             day=day)
            except Exception as e:  # noqa: BLE001 - one bad file must not stop the cart
                log.exception("S3 batch %s: %s failed", batch["id"], key)
                self._set(f, state="failed", error=str(e))
            self._save(batch)

        counts = {s: sum(f["state"] == s for f in cart["files"]) for s in ("done", "failed", "skipped")}
        summary = {"files": len(cart["files"]), "analysed": counts["done"], "failed": counts["failed"],
                   "skipped": counts["skipped"], "screenshots": len(records)}
        if not records:
            self._set(cart, state="failed", summary=summary,
                      error="No screenshot could be taken from the videos in input_data (see the files)")
            return
        rows = cart_report.rows(cart["cart"], records)
        local = BATCHES / batch["id"] / cart["cart"]
        xlsx = cart_report.write_xlsx(local / f"{cart['cart']}.xlsx", rows)
        with open(local / "pictures.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["img name", "source_video", "image_filename", "original_image_path"])
            w.writerows([r["img"], r["source_video"], r["image_filename"], r["original_image_path"]] for r in rows)
        # the xlsx stays local (only the CSV goes to S3); report = the day's CSV the videos' rows went to
        csv_uris = sorted({f["s3_csv"] for f in cart["files"] if f.get("s3_csv")})
        self._set(cart, state="done", summary=summary, report=csv_uris[0] if csv_uris else None,
                  error=None if not counts["failed"] else f"{counts['failed']} video(s) could not be analysed (see the files)")

    def analyse_file(self, s3c, bucket: str, threshold: float | None, cart: str, f: dict, *,
                     s3_output: bool = True, day: date | None = None) -> list[dict]:
        """Download s3://bucket/f["key"] and analyse it; returns its screenshot records
        (screenshots.screenshot_results). Also used by the admin uploads (s3_upload.py, s3_output=False: they
        upload their own CSV). `f` is updated while it runs: state downloading -> processing -> done / failed,
        media_id, extraction_id, ... With s3_output the video's CSV is uploaded to
        the processing day's card_<N>/output_data/ CSV and the file is "done" only once it is verified there (f["s3_csv"])."""
        self._set(f, state="downloading")
        path = new_upload_path(f["name"])
        try:
            s3c.download_file(bucket, f["key"], str(path))
            media = probe(path, f["name"], "video", path.stat().st_size)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        media["sha256"] = _sha256(path)
        self.db.add_media(media)

        start = cart_report.recording_start(f["name"])
        self._set(f, state="processing", media_id=media["id"], recording_start_time=start)
        job, _ = self.extractions.create(media, None, None, new_run=True, run_detection=True,
                                         threshold=threshold, strategy=None, store_id=cart,
                                         recording_start_time=start, source="s3", s3_output=s3_output,
                                         s3_bucket=bucket, s3_cart=cart, s3_day=f"{day:%d-%m-%Y}" if day else None)
        self._set(f, extraction_id=job["id"])
        ext, run = self._wait(job["id"])
        if run:
            self._set(f, detection_run_id=run["id"])
        if ext["state"] != "done":
            error = f"Screenshot extraction {ext['state']}: {ext.get('error') or 'no detail'}"
            # every sample failed (e.g. all frames unreadable): still one report row each, never dropped
            records = screenshot_results(ext, None) if ext["state"] == "failed" and ext.get("failed") else []
            if not records:
                raise RuntimeError(error)
            self._set(f, state="failed", error=error, screenshots=len(records))
            return records
        records = screenshot_results(ext, run if run and run["state"] == "done" else None, DETECTION_FAILED)
        self._set(f, screenshots=len(records))
        if not run or run["state"] != "done":         # listed with a blank score, never as non-uniform
            self._set(f, state="failed", error=f"Uniform detection {run['state'] if run else 'not started'}: "
                                               f"{(run or {}).get('error') or 'no detail'}")
            return records
        out = self.db.s3_output(run["id"]) if s3_output else None
        if s3_output and (not out or out["state"] != "completed"):
            self._set(f, state="failed", error="The video's CSV could not be uploaded to S3: "
                                               f"{(out or {}).get('error') or 'the upload did not finish'}")
        else:
            self._set(f, state="done", s3_csv=f"s3://{out['bucket']}/{out['key']}" if out else None)
        return records

    def _wait(self, jid: str) -> tuple[dict, dict | None]:
        """Until the extraction, its automatic detection run and (S3 output) the run's CSV upload have finished."""
        while True:
            ext = self.db.extraction(jid)
            runs = self.db.detection_runs(jid)
            out = self.db.s3_output(runs[0]["id"]) if ext.get("s3_output") and runs and runs[0]["state"] == "done" else None
            busy = (ext["state"] in ("queued", "running") or self.extractions.detection_requested(jid)
                    or (runs and runs[0]["state"] in ("queued", "running"))
                    or (out is not None and out["state"] not in ("completed", "failed")))
            if not busy or self._stop.is_set():
                return ext, runs[0] if runs else None
            time.sleep(POLL_SECONDS)
