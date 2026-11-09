"""Every analysed video's CSV rows and detection images go to S3: the last stage of the quick sampled analysis.

When a detection run of a job created with S3 output (every new job from Video Analysis, the Results
Dashboard, the cart batches and the CCTV adapter - not the admin uploads, which run the same steps in
s3_upload.py) finishes, its results are uploaded:

    queued -> generating_report -> uploading -> uploading_images -> completed     (or failed, with the stage)

  generating_report  cart_report rows of the run (the same columns as the admin uploads' CSV and the
                     cart_<N>.xlsx) written to storage/extractions/<job>/detections/<run>/s3_report.csv
  uploading          the rows are appended to the processing day's CSV
                     s3://<bucket>/<MM-YYYY>/<DD-MM-YYYY>/card_<N>/output_data/card<N>_YYYY_MM_DD.csv
                     The day is TODAY in Asia/Kolkata when the upload first starts - never the video's date.
                     The day's first run creates the file; later runs append (s3.append_csv: download, add the
                     rows below, save to the same key; one header, earlier rows kept byte for byte, conditional
                     write). Numbering continues after the highest pic<K> already in the CSV or in the folder.
                     Verified: size + SHA-256 of the merged file (local copy: .../<run>/s3_daily.csv).
                     csv_first_row records that the rows are in, so they are never appended twice.
  uploading_images   each row's detection image (the annotated screenshot) goes next to the CSV as
                     card<N>_YYYY_MM_DD_pic<K>.jpg (pic<K> = the row's img name) with s3.put_new - never over
                     an existing object (a different one already there -> ..._pic<K>_<run id>.jpg); verified
  completed          only after that

The cart: the one chosen with the job (or the job's store ID when it is a cart listed in config/carts.yaml).
When it is unknown the upload fails with a message, and POST /api/detection-runs/{id}/s3-output/retry retries
it (optionally with the Card ID). A retry never appends the same rows twice. Credentials come only from the
project's .env (backend.core.s3) and are never logged or returned. Uploads that were in progress when the
server stopped are resumed at start-up.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import queue
import threading
from pathlib import Path

from backend.core import s3
from backend.core.config import ROOT
from backend.database.db import DB, now_iso
from backend.extraction import cart_report
from backend.extraction.runner import EXTRACTIONS
from backend.extraction.screenshots import DETECTION_FAILED, screenshot_results

log = logging.getLogger(__name__)
ACTIVE = ("queued", "generating_report", "uploading", "uploading_images")
LOCAL_NAME = "s3_report.csv"          # this run's rows, kept in the run's folder
DAILY_LOCAL_NAME = "s3_daily.csv"     # the day's CSV as this run wrote it to S3 (earlier rows + this run's)
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


class OutputRejected(ValueError):
    """The request cannot be accepted; the message is shown as is."""


def resolve_cart(ext: dict) -> str:
    if ext.get("s3_cart"):
        return ext["s3_cart"]
    store = (ext.get("store_id") or "").strip()
    try:
        cart = s3.cart_name(store) if store else None
    except ValueError:
        cart = None
    if cart and cart in cart_report.known_carts():
        return cart
    raise RuntimeError("The S3 cart folder is unknown: the job has no Card ID"
                       + (f" and its store ID {store!r} is not a cart listed in config/carts.yaml" if store else "")
                       + ". Retry with the Card ID.")


def describe_error(e: Exception) -> str:
    """A message for the user; never contains credentials (botocore errors do not include them)."""
    try:
        from botocore.exceptions import EndpointConnectionError, NoCredentialsError, PartialCredentialsError
    except ImportError:                                   # boto3 missing: S3Unavailable says so
        return str(e) or e.__class__.__name__
    if isinstance(e, (NoCredentialsError, PartialCredentialsError)):
        return ("No AWS credentials found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (and AWS_REGION) in the "
                "project's .env, then retry.")
    if isinstance(e, EndpointConnectionError):
        return f"S3 could not be reached ({e}); check the network, then retry."
    return str(e) or e.__class__.__name__


# ---------------------------------------------------------------------- shared with s3_upload.py
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(2 ** 20):
            h.update(chunk)
    return h.hexdigest()


def _existing(p: str | None) -> Path | None:
    if not p:
        return None
    path = Path(p) if Path(p).is_absolute() else ROOT / p
    return path if path.is_file() else None


def detection_image(row: dict) -> Path | None:
    """The image uploaded for a report row: the annotated screenshot, else the plain one; None when neither
    exists (e.g. an unreadable frame that was never saved)."""
    return _existing(row.get("annotated_image_path")) or _existing(row.get("original_image_path"))


def highest_image_pic(s3c, bucket: str, csv_key: str) -> int:
    """Highest pic<K> among the day's images already next to the CSV (<csv stem>_pic<K>...). Read-only."""
    prefix, name = csv_key.rsplit("/", 1)
    stem = name.rsplit(".", 1)[0] + "_"
    return s3.highest_pic(k for k in s3.list_keys(s3c, bucket, f"{prefix}/{stem}"))


def upload_images(s3c, bucket: str, csv_key: str, report_rows: list[dict], unique: str) -> int:
    """Upload each row's detection image next to the day's CSV as <csv stem>_<img name><ext>. Never replaces
    an object: when a different file already has that name, <csv stem>_<img name>_<unique><ext> is used.
    Running it again is harmless (an identical object is left as it is). Returns the number of images in S3."""
    prefix, name = csv_key.rsplit("/", 1)
    stem = name.rsplit(".", 1)[0]
    n = 0
    for r in report_rows:
        src = detection_image(r)
        if src is None:
            continue
        ext = src.suffix.lower()
        sha, size = _sha256(src), src.stat().st_size
        for key in (f"{prefix}/{stem}_{r['img']}{ext}", f"{prefix}/{stem}_{r['img']}_{unique}{ext}"):
            try:
                s3.put_new(s3c, bucket, key, src, sha, IMAGE_TYPES.get(ext, "application/octet-stream"))
            except s3.ObjectConflict:
                continue
            s3.verify(s3c, bucket, key, size, sha)
            n += 1
            break
        else:
            raise RuntimeError(f"No free name for the image of {r['img']} in s3://{bucket}/{prefix}/")
    return n


def appended_first_row(daily_local: Path, n_rows: int) -> int:
    """Number of this run's first row in the day's CSV, from the local copy of what was written to S3
    (this run's rows are its last n_rows)."""
    with open(daily_local, newline="", encoding="utf-8-sig") as fh:
        total = sum(1 for _ in csv.reader(fh)) - 1          # minus the header
    return total - n_rows + 1


def append_rows(s3c, bucket: str, key: str, cart: str, records: list[dict], daily_local: Path, *,
                already: str | None, on_prepared) -> int:
    """Append the records' rows to the day's CSV (created by the day's first run) and verify it. Numbering
    continues after the highest pic<K> in the CSV or among the images already in the folder.
    Returns the number of the first appended row."""
    used: dict = {}

    def make(first: int) -> list[list]:
        used["first"] = first
        return [cart_report.csv_row(r) for r in cart_report.rows(cart, records, first)]

    size, sha = s3.append_csv(s3c, bucket, key, cart_report.HEADERS, make, daily_local, already=already,
                              on_prepared=on_prepared, floor=highest_image_pic(s3c, bucket, key))
    s3.verify(s3c, bucket, key, size, sha)
    # make() was not called when an earlier attempt's write had already landed: count from the local copy
    return used.get("first") or appended_first_row(daily_local, len(records))


class S3OutputRunner:
    def __init__(self, db: DB, client_factory=s3.client):
        self.db, self.client_factory = db, client_factory
        self.q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="s3-output-worker", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ hooks (ExtractionRunner.on_run_done)
    def prepare(self, rid: str) -> None:
        """Register the run's upload (state queued) if its job has S3 output. Called BEFORE the run is
        marked done, so a done run of such a job always has its upload record."""
        run = self.db.detection_run(rid)
        ext = self.db.extraction(run["extraction_id"]) if run else None
        if not ext or not ext.get("s3_output"):
            return
        now = now_iso()
        self.db.add_s3_output({"run_id": rid, "extraction_id": ext["id"], "created_at": now, "updated_at": now,
                               "state": "queued", "attempts": 1, "bucket": ext.get("s3_bucket") or s3.bucket_name(),
                               "folder": s3.OUTPUT_FOLDER})

    def submit(self, rid: str) -> None:
        o = self.db.s3_output(rid)
        if o and o["state"] == "queued":
            self.q.put(rid)

    def resume(self) -> int:
        """At start-up: restart uploads that were in progress, register the ones that were never registered."""
        n = 0
        for rid in self.db.s3_outputs_to_resume():
            o = self.db.s3_output(rid)
            if o:
                self.db.update_s3_output(rid, state="queued")
            else:
                self.prepare(rid)
            self.submit(rid)
            n += 1
        return n

    # ------------------------------------------------------------------ API
    def get(self, rid: str) -> dict | None:
        o = self.db.s3_output(rid)
        return self.public(o) if o else None

    @staticmethod
    def public(o: dict) -> dict:
        """What the API returns: no local paths, never any credential."""
        out = {k: v for k, v in o.items() if k != "report_path"}
        out["s3_uri"] = f"s3://{o['bucket']}/{o['key']}" if o.get("key") else None
        out["verified"] = o["state"] == "completed"
        return out

    def retry(self, rid: str, card_id: int | None = None) -> dict:
        """Retry a failed upload. A Card ID given here replaces the job's (only while the run's rows are not
        yet in a CSV in S3). The day stays the processing day chosen when the upload first started."""
        with self._lock:
            o = self.db.s3_output(rid)
            if not o:
                raise KeyError(rid)
            if o["state"] != "failed":
                raise OutputRejected(f"Only a failed upload can be retried (this one is {o['state']})")
            cart = None
            if card_id is not None:
                try:
                    cart = s3.cart_name(str(card_id))
                except ValueError as e:
                    raise OutputRejected(str(e))
                if cart not in cart_report.known_carts():
                    raise OutputRejected(f"Card ID {card_id} is not configured in config/carts.yaml")
                if o.get("csv_first_row") and cart != o.get("cart"):
                    raise OutputRejected(f"This run's rows are already in s3://{o['bucket']}/{o['key']}; "
                                         "its Card ID can no longer be changed")
            fields = {"state": "queued", "error": None, "failed_stage": None, "finished_at": None,
                      "attempts": o["attempts"] + 1}
            if cart and cart != o.get("cart"):
                self.db.update_extraction(o["extraction_id"], s3_cart=cart)
                # new destination: rebuild the CSV (its cart_id / Store Name columns depend on the cart)
                fields.update(cart=None, day=None, day_source=None, key=None, report_path=None, size_bytes=None,
                              sha256=None)
            self.db.update_s3_output(rid, **fields)
        self.q.put(rid)
        return self.get(rid)

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------ worker
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                rid = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run(rid)
            except Exception:  # noqa: BLE001 - _run records its own failures; this is a last resort
                log.exception("S3 output of %s crashed", rid)

    def _run(self, rid: str) -> None:
        o = self.db.s3_output(rid)
        if not o or o["state"] != "queued":
            return
        set_ = lambda **f: self.db.update_s3_output(rid, **f)  # noqa: E731
        stage = "generating_report"
        try:
            set_(state=stage)
            run = self.db.detection_run(rid)
            ext = self.db.extraction(o["extraction_id"])
            if not run or not ext or run["state"] != "done":
                raise RuntimeError("The analysis did not finish; no report is uploaded for it")
            records = screenshot_results(ext, run, DETECTION_FAILED)
            if not records:
                raise RuntimeError("The analysis produced no screenshot to report")
            run_dir = EXTRACTIONS / ext["id"] / "detections" / rid
            if not o.get("key"):
                cart = resolve_cart(ext)
                day = s3.today()                    # the processing day (Asia/Kolkata), never the video's date
                # this run's own rows, kept locally (short name: the run folder is already deep - Windows MAX_PATH)
                local = cart_report.write_csv(run_dir / LOCAL_NAME, cart_report.rows(cart, records))
                key = s3.daily_csv_key(day, cart)   # a retry keeps the key chosen the first time
                o = o | {"cart": cart, "key": key}
                set_(cart=cart, day=f"{day:%d-%m-%Y}", day_source="processing_date", folder=s3.OUTPUT_FOLDER,
                     key=key, report_path=str(local))

            first = o.get("csv_first_row")
            if not first and o.get("day"):
                # rows not in S3 yet: always the current layout for the day chosen first (a job that failed
                # under an earlier layout, e.g. cart50/..., is filed in card_<N>/output_data/ on retry)
                key = s3.daily_csv_key(s3.parse_day(o["day"]), o["cart"])
                if key != o["key"]:
                    o = o | {"key": key, "sha256": None, "size_bytes": None}
                    set_(key=key, folder=s3.OUTPUT_FOLDER, sha256=None, size_bytes=None)

            s3c = self.client_factory()
            if not first:                           # once the rows are verified in S3 they are never appended again
                stage = "uploading"
                set_(state=stage)
                first = append_rows(s3c, o["bucket"], o["key"], o["cart"], records, run_dir / DAILY_LOCAL_NAME,
                                    already=o.get("sha256"),
                                    on_prepared=lambda size, sha: set_(sha256=sha, size_bytes=size))
                set_(csv_first_row=first)

            stage = "uploading_images"
            set_(state=stage)
            n = upload_images(s3c, o["bucket"], o["key"], cart_report.rows(o["cart"], records, first), rid)
            set_(state="completed", images_uploaded=n, finished_at=now_iso())
            log.info("S3 output of %s verified: s3://%s/%s (rows from pic%d, %d images)",
                     rid, o["bucket"], o["key"], first, n)
        except Exception as e:  # noqa: BLE001 - shown with the stage it failed in; Retry resumes it
            msg = describe_error(e)
            log.warning("S3 output of %s failed while %s: %s", rid, stage, msg)
            set_(state="failed", failed_stage=stage, error=msg, finished_at=now_iso())
