"""Frame-extraction jobs and uniform-detection runs on extracted frames: validation, duplicate
prevention, background execution, cancel.

  * One worker thread, separate from the video-analysis worker (extraction never blocks analysis).
  * At most one queued/running extraction per video; an identical finished job is returned
    instead of extracting again (unless the caller asks for a new run).
  * Every job writes to its own new folder, so results are never overwritten.
  * Jobs that were queued/running when the server stopped are marked "interrupted" at start-up.
  * S3 output: every finished detection run of a job created with s3_output (the default) is handed to
    `on_run_done` (backend/extraction/s3_output.py), which uploads its CSV to the cart's output_data folder.
"""
from __future__ import annotations

import csv
import io
import logging
import queue
import re
import secrets
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from backend.core import s3
from backend.core.config import STORAGE
from backend.core.settings import Settings
from backend.database.db import DB, now_iso
from backend.extraction import detect
from backend.extraction.extractor import (LEGACY_MANIFEST_NAME, MANIFEST_NAME, extract, image_id,
                                          normalize_manifest_row, rel, row_meta)
from backend.extraction.plan import (RANDOM_STRATEGIES, describe, hhmmss, parse_recording_start, plan, plan_random,
                                     plan_segments)

log = logging.getLogger(__name__)
EXTRACTIONS = STORAGE / "extractions"
MAX_FPS = 240.0
STRATEGIES = ("interval", "random", "random_segments")
MAX_LISTED_SAMPLES = 50          # jobs up to this size keep a per-sample list (shown with its status)
ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
STORE_PATTERN = re.compile(r"^[A-Za-z0-9._ -]{1,64}$")   # the store name/ID as the user typed it, e.g. "store id 21"


class ExtractionRejected(ValueError):
    """The request is not valid (shown to the user as is)."""


class DuplicateJob(RuntimeError):
    def __init__(self, job: dict):
        super().__init__(f"An extraction for this video is already {job['state']} ({job['id']})")
        self.job = job


def validate_video(media: dict) -> None:
    if media["kind"] != "video":
        raise ExtractionRejected("Frame extraction needs a video file (this upload is an image)")
    if not media.get("fps") or media["fps"] <= 0 or media["fps"] > MAX_FPS:
        raise ExtractionRejected("The video's frame rate could not be read; re-export the video (e.g. MP4/H.264) and upload it again")
    if not media.get("duration_sec") or media["duration_sec"] <= 0:
        raise ExtractionRejected("The video's duration could not be read; re-export the video (e.g. MP4/H.264) and upload it again")
    if not media.get("width") or not media.get("height") or media["width"] < 16 or media["height"] < 16:
        raise ExtractionRejected("The video's resolution could not be read")
    if not Path(media["path"]).exists():
        raise ExtractionRejected("The uploaded video file is missing on disk; upload it again")


def plan_for(media: dict, interval_seconds: float | None, include_initial: bool, max_frames: int,
             strategy: str = "interval", sample_count: int | None = None, seed: int | None = None):
    """(samples, summary). For random sampling without a seed the samples only serve to count -
    the real seed is generated when the job is created."""
    validate_video(media)
    d, fps, n = media["duration_sec"], media["fps"], media["frames"]
    if strategy in RANDOM_STRATEGIES:
        if not sample_count or not (1 <= sample_count <= max_frames):
            raise ExtractionRejected(f"The number of screenshots must be between 1 and {max_frames}")
        fn = plan_segments if strategy == "random_segments" else plan_random
        try:
            samples = fn(d, fps, n, sample_count, 0 if seed is None else seed, include_initial)
        except ValueError as e:
            raise ExtractionRejected(str(e))
        summary = describe(d, fps, n, None, include_initial, samples, strategy, seed, sample_count)
    elif strategy == "interval":
        if interval_seconds is None or not (1.0 <= interval_seconds <= 86400.0):
            raise ExtractionRejected("The interval must be between 1 second and 24 hours")
        samples = plan(d, fps, n, interval_seconds, include_initial)
        summary = describe(d, fps, n, interval_seconds, include_initial, samples)
    else:
        raise ExtractionRejected("The sampling strategy must be 'interval', 'random' or 'random_segments'")
    summary["max_frames"] = max_frames
    summary["allowed"] = 0 < len(samples) <= max_frames
    if not samples:
        summary["reason"] = "This interval gives no frames. Turn on the initial frame or use a shorter interval."
    elif len(samples) > max_frames:
        summary["reason"] = f"{len(samples)} frames is more than the limit of {max_frames}. Use a longer interval."
    return samples, summary


def manifest_path(job: dict) -> Path:
    """image_manifest.csv, or manifest.csv for jobs created before the rename."""
    d = EXTRACTIONS / job["id"]
    return d / MANIFEST_NAME if (d / MANIFEST_NAME).exists() or not (d / LEGACY_MANIFEST_NAME).exists() else d / LEGACY_MANIFEST_NAME


def _complete_rows(p: Path) -> list[dict]:
    """Rows of a CSV that a running job may still be writing: a header or last line that is not complete
    yet (no newline) is left out instead of being parsed half-written."""
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    if not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
    reader = csv.DictReader(io.StringIO(text, newline=""))
    return list(reader) if reader.fieldnames else []


def read_manifest(job: dict) -> list[dict]:
    """Rows of the job's image manifest in the current schema (older files are converted)."""
    p = manifest_path(job)
    if not p.exists():
        return []
    meta = row_meta(job)
    return [normalize_manifest_row(r, meta) for r in _complete_rows(p)]


def read_failures(job: dict) -> list[dict]:
    return _complete_rows(EXTRACTIONS / job["id"] / "failures.csv")


def expected_targets(job: dict, media: dict | None) -> list[float]:
    """Target times of every planned sample. Older jobs did not store them: re-plan from the video
    (deterministic), or fall back to the samples that were actually attempted."""
    targets = (job.get("plan") or {}).get("targets")
    if targets is not None:
        return targets
    if media and (job.get("sampling_strategy") or "interval") == "interval":
        try:
            samples = plan(media["duration_sec"], media["fps"], media["frames"], job["interval_seconds"],
                           bool(job["include_initial"]))
            return [x.target_seconds for x in samples]
        except ValueError:
            pass
    return ([float(r["video_timestamp_seconds"]) for r in read_manifest(job)]
            + [float(r["target_seconds"]) for r in read_failures(job)])


def clean_metadata(store_id: str | None, camera_id: str | None, recording_start_time: str | None) -> dict:
    """Validate optional traceability fields. Nothing is invented: missing values stay None."""
    out = {}
    for k, v, pattern, allowed in (("store_id", store_id, STORE_PATTERN, "letters, digits, spaces, '.', '_' and '-'"),
                                   ("camera_id", camera_id, ID_PATTERN, "letters, digits, '.', '_' and '-'")):
        v = (v or "").strip()
        if v and not pattern.match(v):
            raise ExtractionRejected(f"{k} may contain {allowed} only (max 64)")
        out[k] = v or None
    try:
        start = parse_recording_start(recording_start_time)
    except ValueError as e:
        raise ExtractionRejected(str(e))
    out["recording_start_time"] = start.isoformat() if start else None
    return out


def clean_s3_destination(enabled: bool, bucket: str | None, cart: str | None, day: str | None) -> dict:
    """Extraction columns of the job's S3 output. A missing cart is resolved when the CSV is uploaded
    (store_id) - never guessed here. `day` (the input day folder) is only recorded: the output always goes
    to the processing day's folder (s3_output.py)."""
    if not enabled:
        return {"s3_output": 0}
    try:
        cart = s3.cart_name(cart) if cart and cart.strip() else None
        day = f"{s3.parse_day(day):%d-%m-%Y}" if day and day.strip() else None
    except ValueError as e:
        raise ExtractionRejected(str(e))
    return {"s3_output": 1, "s3_bucket": (bucket or "").strip() or s3.bucket_name(), "s3_cart": cart,
            "s3_day": day}


class ExtractionRunner:
    def __init__(self, db: DB, settings: Settings, models=None):
        self.db, self.settings, self.models = db, settings, models
        # S3 output stage (S3OutputRunner): prepare(run_id) before a run is marked done, submit(run_id) after
        self.on_run_done = None
        self.q: queue.Queue = queue.Queue()
        self._cancel: set[str] = set()
        self._auto_detect: dict[str, float | None] = {}   # extraction id -> threshold (None = settings default)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="extraction-worker", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ create
    def create(self, media: dict, interval_seconds: float | None, include_initial: bool | None, new_run: bool = False,
               run_detection: bool = False, threshold: float | None = None, *, strategy: str | None = "interval",
               sample_count: int | None = None, seed: int | None = None, store_id: str | None = None,
               camera_id: str | None = None, recording_start_time: str | None = None,
               source: str = "upload", s3_output: bool = True, s3_bucket: str | None = None,
               s3_cart: str | None = None, s3_day: str | None = None) -> tuple[dict, bool]:
        """Returns (job, reused). Raises ExtractionRejected or DuplicateJob. With `run_detection` the
        uniform model is run on the frames as soon as the extraction finishes.

        An identical finished job (same video, strategy, settings and metadata) is reused unless
        `new_run`. A random job is identical only when the same seed was given explicitly.

        With `s3_output` (default) the CSV of every finished detection run is uploaded to
        s3://<bucket>/<MM-YYYY>/<DD-MM-YYYY>/card_<N>/output_data/ (the processing day) and verified; the admin uploads pass False
        because they upload their own CSV."""
        cfg = self.settings.get()
        if strategy is None:                 # API caller gave no strategy: interval if an interval was given
            strategy = "interval" if interval_seconds is not None else cfg["extraction_default_strategy"]
        if strategy not in STRATEGIES:
            raise ExtractionRejected("The sampling strategy must be 'interval', 'random' or 'random_segments'")
        if strategy in RANDOM_STRATEGIES and sample_count is None:
            sample_count = int(cfg["extraction_default_sample_count"])
        meta = clean_metadata(store_id, camera_id, recording_start_time)
        dest = clean_s3_destination(s3_output, s3_bucket, s3_cart, s3_day)
        if include_initial is None:          # the initial frame is extra: off by default for the random strategies
            include_initial = strategy == "interval"
        explicit_seed = seed is not None
        if strategy in RANDOM_STRATEGIES:
            interval_seconds = None
            if seed is None:
                seed = secrets.randbelow(2 ** 31)
        samples, summary = plan_for(media, interval_seconds, include_initial, int(cfg["extraction_max_frames"]),
                                    strategy, sample_count, seed)
        if not summary["allowed"]:
            raise ExtractionRejected(summary["reason"])
        summary["targets"] = [x.target_seconds for x in samples]        # expected samples (hourly summary)
        fmt = cfg["extraction_image_format"]
        with self._lock:
            for j in self.db.extractions(media_id=media["id"], limit=200):
                if j["state"] in ("queued", "running"):
                    raise DuplicateJob(j)
                same = ((j.get("sampling_strategy") or "interval") == strategy
                        and all((j.get(k) or None) == (v or None) for k, v in dest.items())
                        and bool(j["include_initial"]) == include_initial and j["image_format"] == fmt
                        and all((j.get(k) or None) == v for k, v in meta.items()))
                if strategy == "interval":
                    same = same and abs(j["interval_seconds"] - interval_seconds) < 1e-9
                else:
                    same = (same and explicit_seed and j.get("random_seed") == seed
                            and j.get("sample_count") == sample_count)
                if same and j["state"] == "done" and not new_run:
                    return j, True
            jid = f"ext-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
            out = EXTRACTIONS / jid
            job = {"id": jid, "created_at": now_iso(), "media_id": media["id"], "source_video": media["filename"],
                   "state": "queued", "interval_seconds": interval_seconds if strategy == "interval" else 0,
                   "include_initial": int(include_initial), "image_format": fmt, "planned": len(samples),
                   "output_dir": rel(out), "plan": summary, "failures": [], "sampling_strategy": strategy,
                   "sample_count": sample_count if strategy in RANDOM_STRATEGIES else None,
                   "random_seed": seed if strategy in RANDOM_STRATEGIES else None, "source": source, **meta, **dest}
            if len(samples) <= MAX_LISTED_SAMPLES:       # the selected timestamps, shown with their status
                summary["samples"] = [{"image_id": image_id(jid, x), "sample_type": x.sample_type,
                                       "target_seconds": x.target_seconds, "target_hhmmss": hhmmss(x.target_seconds),
                                       "target_frame": x.target_frame} for x in samples]
            self.db.add_extraction(job)
            if run_detection:
                self._auto_detect[jid] = threshold
        self.q.put(("extract", (jid, media, samples, out, fmt)))
        return self.db.extraction(jid), False

    def detection_requested(self, jid: str) -> bool:
        """Detection was asked for with the extraction and the run has not been created yet."""
        return jid in self._auto_detect

    def cancel(self, jid: str) -> None:
        """Works for extraction jobs and detection runs (ids are unique across both)."""
        self._cancel.add(jid)

    # ------------------------------------------------------------------ detection runs
    def model_check(self) -> tuple[str, str]:
        """(model_version, sha256) of the TRAINED uniform model, or ExtractionRejected. The stripe
        baseline is never used for detection runs."""
        if self.models is None:
            raise ExtractionRejected("Models are not available")
        cfg = self.settings.get()
        try:
            m = self.models.get(cfg)
        except Exception as e:  # noqa: BLE001
            raise ExtractionRejected(f"The models could not be loaded: {e}")
        u = m.uniform
        st = u.status
        if not u.trained:
            raise ExtractionRejected(f"Trained uniform model weights not found ({st['model']['expected_path']}). "
                                     "Restore them; detection was not run and no substitute model was used.")
        if st["state"] == "mismatch":
            raise ExtractionRejected(f"{st['model']['path']} does not match the approved checksum in "
                                     "config/uniform_config.yaml; detection was not run.")
        return u.model_version, st["model"].get("sha256") or ""

    def create_detection(self, extraction: dict, threshold: float | None = None, new_run: bool = False) -> tuple[dict, bool]:
        """Queue a detection run on an extraction's frames. Returns (run, reused)."""
        cfg = self.settings.get()
        thr = float(threshold if threshold is not None else cfg["uniform_confidence"])
        if not (0.5 <= thr <= 0.99):
            raise ExtractionRejected("The threshold must be between 0.50 and 0.99")
        if extraction["state"] in ("queued", "running"):
            raise ExtractionRejected("The extraction has not finished yet")
        frames = read_manifest(extraction)
        if not frames:
            raise ExtractionRejected("This extraction has no saved frames to analyse")
        version, sha = self.model_check()
        with self._lock:
            for r in self.db.detection_runs(extraction["id"]):
                if r["state"] in ("queued", "running"):
                    raise DuplicateJob(r)
                if (not new_run and r["state"] == "done" and abs(r["threshold"] - thr) < 1e-9 and r["model_sha256"] == sha
                        and r["total"] == len(frames)):
                    return r, True
            rid = f"det-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
            out = EXTRACTIONS / extraction["id"] / "detections" / rid
            run = {"id": rid, "extraction_id": extraction["id"], "created_at": now_iso(), "state": "queued",
                   "threshold": thr, "model_version": version, "model_sha256": sha, "total": len(frames),
                   "output_dir": rel(out)}
            self.db.add_detection_run(run)
        self.q.put(("detect", (rid, frames, out, thr, version)))
        return self.db.detection_run(rid), False

    def _detect(self, rid: str, frames: list[dict], out: Path, thr: float, version: str) -> None:
        if rid in self._cancel:
            self.db.update_detection_run(rid, state="cancelled", finished_at=now_iso())
            return
        if out.exists():
            raise RuntimeError(f"Output folder already exists: {rel(out)}")
        self.db.update_detection_run(rid, state="running", started_at=now_iso())
        cfg = self.settings.get()
        models = self.models.get(cfg)
        last = [0.0]
        keys = ("processed", "succeeded", "failed", "uniform_images", "no_uniform_images", "review_images", "detections")

        def progress(st: dict) -> None:
            now = time.time()
            if now - last[0] >= 0.5 or st["processed"] == len(frames):
                last[0] = now
                self.db.update_detection_run(rid, progress=round(st["processed"] / len(frames), 4),
                                             **{k: st[k] for k in keys})

        st = detect.run(frames, out, models, cfg, thr, version, on_progress=progress,
                        should_stop=lambda: rid in self._cancel or self._stop.is_set(), run_id=rid)
        state = "cancelled" if st["stopped"] else "done"
        fields = {"state": state, "finished_at": now_iso(), **{k: st[k] for k in keys}}
        if state == "done":
            fields["progress"] = 1.0
            if st["succeeded"] == 0:
                fields.update(state="failed", error="Inference failed on every image (see detection_results.csv)")
        if st["processed"]:
            try:                     # report snapshot next to detection_results.csv - BEFORE the run is marked
                from backend.extraction import business_csv                   # finished, so it is never read half-written
                from backend.extraction.reports import write_run_reports
                run = self.db.detection_run(rid) | fields
                ext = self.db.extraction(run["extraction_id"])
                write_run_reports(ext, run)
                business_csv.write(ext, run)
            except Exception:  # noqa: BLE001 - the reports can always be regenerated from the stored files
                log.exception("could not write the CSV reports of %s", rid)
        if fields["state"] == "done":
            try:                     # cumulative master CSV, also before "done" (a failure is reported by
                from backend.extraction import master_csv                      # its status and retried by sync)
                run = self.db.detection_run(rid) | fields
                master_csv.append(self.db.extraction(run["extraction_id"]), run)
            except Exception:  # noqa: BLE001
                log.exception("could not append %s to the master CSV", rid)
        hook = self.on_run_done if fields["state"] == "done" else None
        if hook:
            try:                     # the upload record exists before "done", so nobody sees a done run
                hook.prepare(rid)    # whose S3 output has not been registered yet
            except Exception:  # noqa: BLE001 - resumed at the next start-up (db.s3_outputs_to_resume)
                log.exception("could not register the S3 output of %s", rid)
        self.db.update_detection_run(rid, **fields)
        if hook:
            hook.submit(rid)
        log.info("detection %s %s: %d images, %d ok, %d failed, %d detections", rid, fields["state"],
                 st["processed"], st["succeeded"], st["failed"], st["detections"])

    # ------------------------------------------------------------------ run
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                kind, args = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            jid = args[0]
            try:
                if kind == "extract":
                    self._run(*args)
                    self._after_extraction(jid)
                else:
                    self._detect(*args)
            except Exception as e:  # noqa: BLE001 - recorded on the job
                log.exception("%s %s failed", kind, jid)
                if kind == "extract":
                    self.db.update_extraction(jid, state="failed", error=str(e), finished_at=now_iso())
                else:
                    self.db.update_detection_run(jid, state="failed", error=str(e), finished_at=now_iso())
            finally:
                self._cancel.discard(jid)

    def _after_extraction(self, jid: str) -> None:
        if jid not in self._auto_detect:
            return
        try:
            self._start_auto_detection(jid, self._auto_detect[jid])
        finally:                 # only once the run exists, so detection_requested() never has a gap
            self._auto_detect.pop(jid, None)

    def _start_auto_detection(self, jid: str, thr: float | None) -> None:
        job = self.db.extraction(jid)
        if not job or job["state"] != "done":
            return
        try:
            self.create_detection(job, thr)
        except (ExtractionRejected, DuplicateJob) as e:   # e.g. model missing: record it visibly
            log.warning("automatic detection for %s not started: %s", jid, e)
            self.db.add_detection_run({"id": f"det-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
                                       "extraction_id": jid, "created_at": now_iso(), "finished_at": now_iso(),
                                       "state": "failed", "threshold": float(thr or self.settings.get()["uniform_confidence"]),
                                       "total": job["extracted"], "error": str(e)})

    def _run(self, jid: str, media: dict, samples, out: Path, fmt: str) -> None:
        if jid in self._cancel:
            self.db.update_extraction(jid, state="cancelled", finished_at=now_iso())
            return
        self.db.update_extraction(jid, state="running", started_at=now_iso())
        last = [0.0]

        def progress(done: int, ok: int, failed: int) -> None:
            now = time.time()
            if now - last[0] >= 0.5 or done == len(samples):
                last[0] = now
                self.db.update_extraction(jid, progress=round(done / len(samples), 4), extracted=ok, failed=failed)

        if out.exists():                                  # never write into an existing folder
            raise RuntimeError(f"Output folder already exists: {rel(out)}")
        res = extract(media["path"], row_meta(self.db.extraction(jid)), out, float(media["fps"]), samples, fmt,
                      on_progress=progress, should_stop=lambda: jid in self._cancel or self._stop.is_set())
        state = "cancelled" if res["stopped"] else "done"
        err = None
        if not res["stopped"] and res["extracted"] == 0:
            state, err = "failed", "No frame could be extracted (see failures.csv)"
        elif not res["stopped"] and res["failed"]:
            err = (f"Only {res['extracted']} of {len(samples)} screenshots could be extracted; {res['failed']} failed "
                   "(see failures.csv)")
        fields = {"state": state, "error": err, "extracted": res["extracted"], "failed": res["failed"],
                  "failures": res["failures"][:500], "finished_at": now_iso()}
        if state != "cancelled":
            fields["progress"] = 1.0
        self.db.update_extraction(jid, **fields)
        log.info("extraction %s %s: %d extracted, %d failed", jid, state, res["extracted"], res["failed"])

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
