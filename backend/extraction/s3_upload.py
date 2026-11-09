"""Admin video uploads (POST /api/s3/uploads): one video for one card goes to S3, through the existing
quick sampled analysis, and its report goes back to S3 as a CSV.

    queued -> uploading -> processing -> generating_report -> uploading_output -> uploading_images -> completed
                                                                                                    (or failed)

  uploading          the received video is written to <MM-YYYY>/<DD-MM-YYYY>/cart_<N>/input_data/<name>
                     with a conditional PUT (never over an existing object) and verified (size + SHA-256)
  processing         S3BatchRunner.analyse_file: download from S3, extraction + uniform detection
                     (the same path as the cart batches, so the job also reaches the master CSV)
  generating_report  cart_report rows (the cart_<N>.xlsx columns) written as this video's own CSV (local; the
                     admin portal shows / downloads it)
  uploading_output   the rows are appended to the processing day's CSV (today in Asia/Kolkata, never the
                     video's date) <MM-YYYY>/<DD-MM-YYYY>/card_<N>/output_data/card<N>_YYYY_MM_DD.csv - the day's first
                     job creates it (s3.append_csv: earlier rows kept, conditional write, numbering continues
                     after the highest pic<K>) - and verified
  uploading_images   each row's detection image next to the CSV, card<N>_YYYY_MM_DD_pic<K>.jpg (never replaced;
                     s3_output.upload_images)
  completed          only after the CSV and the images in S3 are verified

Duplicates and retries: the same video (SHA-256) for the same card is one job. Uploading it again returns
that job; if it had failed, the new copy is used to retry it. A retry skips the stages already done:
a verified input is not uploaded again and a finished analysis is reused, so the master CSV never gets
the same video twice. A failed or unfinished analysis is never reported (no CSV is uploaded for it).
"""
from __future__ import annotations

import csv
import logging
import queue
import threading
import uuid
from datetime import date, datetime
from pathlib import Path

from backend.core import s3
from backend.core.config import STORAGE
from backend.database.db import DB, now_iso
from backend.extraction import cart_report
from backend.extraction.extractor import FRAME_UNREADABLE
from backend.extraction.s3_batch import S3BatchRunner
from backend.extraction.s3_output import append_rows, upload_images
from backend.extraction.screenshots import DETECTION_FAILED, screenshot_results

log = logging.getLogger(__name__)
UPLOADS_DIR = STORAGE / "s3_uploads"
INCOMING = UPLOADS_DIR / "incoming"
ACTIVE = ("queued", "uploading", "processing", "generating_report", "uploading_output", "uploading_images")
VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo",
               ".mkv": "video/x-matroska", ".webm": "video/webm", ".mpg": "video/mpeg", ".mpeg": "video/mpeg",
               ".ts": "video/mp2t"}


class UploadRejected(ValueError):
    """The request cannot be accepted; the message is shown to the admin as is."""


def card_cart(card_id: int) -> tuple[str, str]:
    """(cart folder, store name) of a Card ID listed in config/carts.yaml: 50 -> ("cart_50", "IOCL Khandwa naka")."""
    cart = s3.cart_name(str(card_id))
    carts = cart_report.known_carts()
    if cart not in carts:
        known = ", ".join(str(cart_report.cart_id(c)) for c in carts) or "none"
        raise UploadRejected(f"Card ID {card_id} is not configured in config/carts.yaml (configured: {known})")
    return cart, carts[cart]


def summarise(records: list[dict]) -> dict:
    people = [p["type"] for r in records for p in r["people"]]
    return {"screenshots": len(records),
            "unreadable_frames": sum(r["processing_status"] == FRAME_UNREADABLE for r in records),
            "people": len(people), "uniform": people.count("uniform_shirt"), "not_uniform": people.count("not_uniform"),
            "manual_review": sum(t not in ("uniform_shirt", "not_uniform") for t in people)}


class S3UploadRunner:
    def __init__(self, db: DB, batches: S3BatchRunner, client_factory=None):
        self.db, self.batches, self.client_factory = db, batches, client_factory
        self.live: dict[str, dict] = {}          # job id -> analysis progress of the running job
        self.q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="s3-upload-worker", daemon=True)
        self._thread.start()

    def client(self):
        # the batch runner's factory by default, so both use the same S3 client (and the same test stand-in)
        return (self.client_factory or self.batches.client_factory)()

    # ------------------------------------------------------------------ API
    def create(self, *, card_id: int, path: Path, filename: str, size: int, sha256: str, day: date,
               day_source: str, bucket: str, user: str | None) -> tuple[dict, str]:
        """Returns (job, how): "created", "duplicate" (the same video is already queued / done for this card;
        the received copy is discarded) or "retried" (it had failed: retried with the received copy)."""
        cart, store = card_cart(card_id)
        with self._lock:
            old = self.db.s3_upload_by_sha256(sha256, cart)
            if old and old["state"] != "failed":
                path.unlink(missing_ok=True)
                return self.get(old["id"]), "duplicate"
            if old:
                keep = not old["input_verified"] and not (old["local_path"] and Path(old["local_path"]).exists())
                if keep:
                    self.db.update_s3_upload(old["id"], local_path=str(path))
                else:
                    path.unlink(missing_ok=True)
                self._requeue(old)
                return self.get(old["id"]), "retried"
            jid = f"upl-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
            name = s3.safe_filename(filename)
            now = now_iso()
            self.db.add_s3_upload({
                "id": jid, "created_at": now, "updated_at": now, "created_by": user, "card_id": card_id, "cart": cart,
                "store_name": store, "filename": filename, "s3_name": name, "size_bytes": size, "sha256": sha256,
                "day": f"{day:%d-%m-%Y}", "day_source": day_source, "bucket": bucket,
                "input_key": s3.input_prefix(day, cart) + name,
                # the processing day's CSV (today, Asia/Kolkata - not the video's date) its rows are appended to
                "output_key": s3.daily_csv_key(s3.today(), cart),
                "state": "queued", "attempts": 1, "input_verified": 0, "local_path": str(path)})
        self.q.put(jid)
        return self.get(jid), "created"

    def retry(self, jid: str) -> dict:
        with self._lock:
            job = self.db.s3_upload(jid)
            if not job:
                raise KeyError(jid)
            if job["state"] != "failed":
                raise UploadRejected(f"Only a failed job can be retried (this one is {job['state']})")
            if not job["input_verified"] and not (job["local_path"] and Path(job["local_path"]).exists()):
                raise UploadRejected("The video never reached S3 and the received copy is gone: upload the file again")
            self._requeue(job)
        return self.get(jid)

    def _requeue(self, job: dict) -> None:
        self.db.update_s3_upload(job["id"], state="queued", error=None, failed_stage=None, finished_at=None,
                                 attempts=job["attempts"] + 1)
        self.q.put(job["id"])

    def get(self, jid: str) -> dict | None:
        job = self.db.s3_upload(jid)
        return self.public(job) if job else None

    def list(self, limit: int = 50) -> list[dict]:
        return [self.public(j) for j in self.db.s3_uploads(limit)]

    def public(self, job: dict) -> dict:
        """What the admin sees: no local paths; live analysis progress while processing."""
        out = {k: v for k, v in job.items() if k not in ("local_path", "report_path")}
        out["input_verified"] = bool(job["input_verified"])
        out["s3_input"] = f"s3://{job['bucket']}/{job['input_key']}"
        out["s3_output"] = f"s3://{job['bucket']}/{job['output_key']}"
        out["report_url"] = f"/api/s3/uploads/{job['id']}/report.csv" if job["state"] == "completed" else None
        out["analysis"] = None
        if job["state"] == "processing":
            f = self.live.get(job["id"], {})
            eid = f.get("extraction_id") or job["extraction_id"]
            ext = self.db.extraction(eid) if eid else None
            runs = self.db.detection_runs(eid) if eid else []
            out["analysis"] = {"step": "detecting" if runs else "extracting" if ext else f.get("state") or "starting",
                               "extraction_progress": ext["progress"] if ext else None,
                               "detection_progress": runs[0]["progress"] if runs else None}
        return out

    def report_rows(self, jid: str) -> list[dict] | None:
        job = self.db.s3_upload(jid)
        if not job or job["state"] != "completed" or not job["report_path"] or not Path(job["report_path"]).exists():
            return None
        with open(job["report_path"], newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def report_path(self, jid: str) -> Path | None:
        job = self.db.s3_upload(jid)
        if not job or job["state"] != "completed" or not job["report_path"]:
            return None
        p = Path(job["report_path"])
        return p if p.exists() else None

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                jid = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run(jid)
            except Exception:  # noqa: BLE001 - _run records its own failures; this is a last resort
                log.exception("S3 upload %s crashed", jid)

    def _run(self, jid: str) -> None:
        job = self.db.s3_upload(jid)
        if not job or job["state"] != "queued":
            return
        set_ = lambda **f: self.db.update_s3_upload(jid, **f)  # noqa: E731
        stage = "uploading"
        try:
            s3c = self.client()
            bucket = job["bucket"]
            if not job["input_verified"]:
                set_(state=stage)
                local = Path(job["local_path"] or "")
                if not local.is_file():
                    raise RuntimeError("The received video is missing on the server: upload the file again")
                ctype = VIDEO_TYPES.get(local.suffix.lower(), "application/octet-stream")
                s3.put_new(s3c, bucket, job["input_key"], local, job["sha256"], ctype)
                s3.verify(s3c, bucket, job["input_key"], job["size_bytes"], job["sha256"])
                set_(input_verified=1, local_path=None)
                local.unlink(missing_ok=True)

            stage = "processing"
            set_(state=stage)
            records = self._records(s3c, self.db.s3_upload(jid))

            stage = "generating_report"
            set_(state=stage)
            # this video's own report (what the admin portal shows and downloads)
            out = cart_report.write_csv(UPLOADS_DIR / jid / f"{job['cart']}_{Path(job['s3_name']).stem}_{jid}.csv",
                                        cart_report.rows(job["cart"], records))
            set_(report_path=str(out), summary=summarise(records))

            first = job.get("csv_first_row")
            if not first:                            # once the rows are verified in S3 they are never appended again
                stage = "uploading_output"
                set_(state=stage)
                # appended to the day's CSV; report_sha256 = SHA-256 of that CSV in S3 once this job's rows are in it
                first = append_rows(s3c, bucket, job["output_key"], job["cart"], records,
                                    UPLOADS_DIR / jid / job["output_key"].rsplit("/", 1)[-1],
                                    already=job.get("report_sha256"), on_prepared=lambda size, sha: set_(report_sha256=sha))
                set_(csv_first_row=first)

            stage = "uploading_images"
            set_(state=stage)
            n = upload_images(s3c, bucket, job["output_key"], cart_report.rows(job["cart"], records, first), jid)
            set_(state="completed", images_uploaded=n, finished_at=now_iso())
        except Exception as e:  # noqa: BLE001 - shown to the admin with the stage it failed in
            log.warning("S3 upload %s failed while %s: %s", jid, stage, e)
            set_(state="failed", failed_stage=stage, error=str(e) or e.__class__.__name__, finished_at=now_iso())

    def _records(self, s3c, job: dict) -> list[dict]:
        """Screenshot records of the video: a finished earlier analysis of this job is reused (retries);
        otherwise the video is downloaded from S3 and analysed. Raises unless detection finished."""
        if job["extraction_id"]:
            ext = self.db.extraction(job["extraction_id"])
            runs = self.db.detection_runs(job["extraction_id"])
            if ext and ext["state"] == "done" and runs and runs[0]["state"] == "done":
                return screenshot_results(ext, runs[0], DETECTION_FAILED)
        f = {"key": job["input_key"], "name": job["s3_name"], "state": "queued", "error": None,
             "recording_start_time": None, "extraction_id": None, "detection_run_id": None, "screenshots": 0}
        self.live[job["id"]] = f
        try:
            records = self.batches.analyse_file(s3c, job["bucket"], None, job["cart"], f, s3_output=False)
        finally:
            self.live.pop(job["id"], None)
            self.db.update_s3_upload(job["id"], media_id=f.get("media_id"), extraction_id=f.get("extraction_id"),
                                     detection_run_id=f.get("detection_run_id"))
        if f["state"] != "done":
            raise RuntimeError(f["error"] or "The analysis did not finish")
        return records
