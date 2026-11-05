"""SQLite storage for analyses, per-person results and alerts. Every number shown in
the UI (overview, analytics, history, alerts) is read from here - nothing is invented."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS media (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL, filename TEXT NOT NULL,
  path TEXT NOT NULL, size_bytes INTEGER, width INTEGER, height INTEGER, fps REAL,
  frames INTEGER, duration_sec REAL, suggested_layout TEXT
);
CREATE TABLE IF NOT EXISTS analyses (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, finished_at TEXT, media_id TEXT NOT NULL,
  kind TEXT NOT NULL, filename TEXT NOT NULL,
  state TEXT NOT NULL,                  -- queued | running | done | failed
  progress REAL DEFAULT 0, error TEXT,
  status TEXT,                          -- compliant | no_uniform | uncertain | no_person
  people INTEGER, uniformed INTEGER, non_uniform INTEGER, uncertain INTEGER,
  avg_uniform_conf REAL, max_uniform_conf REAL,
  duration_sec REAL, processing_sec REAL, frames_total INTEGER, frames_processed INTEGER,
  processing_fps REAL, longest_presence_sec REAL, alerts INTEGER DEFAULT 0,
  uniform_obs INTEGER, non_uniform_obs INTEGER, uncertain_obs INTEGER,
  area TEXT, settings TEXT, classifier TEXT, model_version TEXT, device TEXT,
  live TEXT,                            -- JSON progress snapshot while running
  files TEXT                            -- JSON {annotated, keyframe, report_json, report_csv, report_txt}
);
CREATE TABLE IF NOT EXISTS persons (
  analysis_id TEXT NOT NULL, track_id INTEGER NOT NULL, label TEXT NOT NULL, confidence REAL,
  det_confidence REAL, first_seen_sec REAL, last_seen_sec REAL, frames INTEGER,
  uniform_votes INTEGER, non_uniform_votes INTEGER, uncertain_votes INTEGER, note TEXT, box TEXT,
  PRIMARY KEY (analysis_id, track_id)
);
CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, analysis_id TEXT NOT NULL,
  kind TEXT NOT NULL,                   -- no_uniform | low_confidence
  severity TEXT NOT NULL,               -- crit | warn
  title TEXT NOT NULL, message TEXT, media_time_sec REAL, people_inside INTEGER, uniformed INTEGER,
  confidence REAL, track_id INTEGER, evidence TEXT, filename TEXT,
  state TEXT NOT NULL DEFAULT 'new'     -- new | acknowledged | resolved
);
CREATE TABLE IF NOT EXISTS cameras (
  id TEXT PRIMARY KEY,                  -- e.g. CAM-01
  store_id TEXT NOT NULL,               -- e.g. STORE-001
  name TEXT NOT NULL,
  source_type TEXT NOT NULL,            -- folder | video (local sources; rtsp after CCTV access)
  source TEXT NOT NULL,                 -- local path of the test source
  area TEXT,                            -- JSON {rect, roi, label, layout, cell}: camera tile + store ROI
  enabled INTEGER NOT NULL DEFAULT 1,
  cursor REAL NOT NULL DEFAULT 0,       -- folder: next image index | video: next position (s)
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS captures (
  id TEXT PRIMARY KEY,
  camera_id TEXT NOT NULL, store_id TEXT NOT NULL,
  captured_at TEXT NOT NULL,            -- UTC ISO time the frame was taken
  slot_at TEXT,                         -- scheduled slot (UTC ISO); NULL for manual captures
  trigger TEXT NOT NULL,                -- schedule | manual
  status TEXT NOT NULL,                 -- running | done | capture_failed | analysis_failed
  error TEXT,
  source_ref TEXT,                      -- which file / video position the frame came from
  frame_hash TEXT,                      -- 64-bit dHash (hex) of the frame
  similar_to_previous INTEGER,          -- 1 if nearly identical to this camera's previous frame
  media_id TEXT, analysis_id TEXT,
  duration_ms INTEGER
);
CREATE TABLE IF NOT EXISTS extractions (
  id TEXT PRIMARY KEY,                  -- ext-YYYYMMDD-HHMMSS-xxxxxx
  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
  media_id TEXT NOT NULL, source_video TEXT NOT NULL,
  state TEXT NOT NULL,                  -- queued | running | done | failed | cancelled | interrupted
  interval_seconds REAL NOT NULL, include_initial INTEGER NOT NULL, image_format TEXT NOT NULL,
  planned INTEGER NOT NULL,             -- samples in the plan (initial + scheduled)
  extracted INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
  progress REAL NOT NULL DEFAULT 0,
  error TEXT,
  output_dir TEXT,                      -- project-relative: storage/extractions/<id>
  plan TEXT,                            -- JSON summary of the plan (counts + explanation)
  failures TEXT                         -- JSON list of failed samples (also in failures.csv)
);
CREATE INDEX IF NOT EXISTS ix_analyses_created ON analyses(created_at);
CREATE INDEX IF NOT EXISTS ix_alerts_created ON alerts(created_at);
CREATE INDEX IF NOT EXISTS ix_captures_camera ON captures(camera_id, captured_at);
CREATE INDEX IF NOT EXISTS ix_extractions_media ON extractions(media_id, created_at);
CREATE TABLE IF NOT EXISTS frame_detection_runs (
  id TEXT PRIMARY KEY,                  -- det-YYYYMMDD-HHMMSS-xxxxxx
  extraction_id TEXT NOT NULL,
  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
  state TEXT NOT NULL,                  -- queued | running | done | failed | cancelled | interrupted
  threshold REAL NOT NULL,              -- P(uniform) needed for a target-shirt detection
  model_version TEXT, model_sha256 TEXT,
  total INTEGER NOT NULL,               -- images to process (extracted frames)
  processed INTEGER NOT NULL DEFAULT 0, succeeded INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
  uniform_images INTEGER NOT NULL DEFAULT 0, no_uniform_images INTEGER NOT NULL DEFAULT 0,
  review_images INTEGER NOT NULL DEFAULT 0, detections INTEGER NOT NULL DEFAULT 0,
  progress REAL NOT NULL DEFAULT 0, error TEXT,
  output_dir TEXT                       -- project-relative: storage/extractions/<ext>/detections/<run>
);
CREATE INDEX IF NOT EXISTS ix_detruns_extraction ON frame_detection_runs(extraction_id, created_at);
CREATE TABLE IF NOT EXISTS s3_uploads (    -- admin video uploads to S3 (backend/extraction/s3_upload.py)
  id TEXT PRIMARY KEY,                  -- upl-YYYYMMDD-HHMMSS-xxxxxx
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL, finished_at TEXT, created_by TEXT,
  card_id INTEGER NOT NULL, cart TEXT NOT NULL, store_name TEXT,
  filename TEXT NOT NULL,               -- as uploaded by the admin
  s3_name TEXT NOT NULL,                -- safe file name used in the S3 key
  size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
  day TEXT NOT NULL,                    -- S3 day folder, DD-MM-YYYY
  day_source TEXT NOT NULL,             -- file_name | selected
  bucket TEXT NOT NULL, input_key TEXT NOT NULL, output_key TEXT NOT NULL,
  state TEXT NOT NULL,                  -- queued | uploading | processing | generating_report | uploading_output | completed | failed
  failed_stage TEXT,                    -- the state in which it failed
  error TEXT,
  attempts INTEGER NOT NULL DEFAULT 1,
  input_verified INTEGER NOT NULL DEFAULT 0,  -- 1 = the video is in S3 with the same size and SHA-256
  local_path TEXT,                      -- received video until it is verified in S3 (then deleted)
  media_id TEXT, extraction_id TEXT, detection_run_id TEXT,
  summary TEXT,                         -- JSON counts of the report
  report_path TEXT, report_sha256 TEXT
);
CREATE INDEX IF NOT EXISTS ix_s3_uploads_sha ON s3_uploads(sha256, cart);
"""

# Columns added after the first release (additive migrations, applied on start-up).
ANALYSIS_COLUMNS = {
    "model_version": "TEXT",
    "source": "TEXT",          # upload | capture   (NULL = upload, rows created before captures existed)
    "store_id": "TEXT",
    "camera_id": "TEXT",
    "capture_id": "TEXT",
    "captured_at": "TEXT",     # UTC ISO time the frame was taken (captures); NULL for uploads
}

# Extraction-job columns added with random sampling and store/camera traceability.
EXTRACTION_COLUMNS = {
    "sampling_strategy": "TEXT",     # interval | random   (NULL = interval, rows created before random sampling)
    "sample_count": "INTEGER",       # random: number of random samples requested
    "random_seed": "INTEGER",        # random: seed that reproduces the same samples
    "store_id": "TEXT",              # as supplied by the user / CCTV adapter; NULL = unknown
    "camera_id": "TEXT",
    "recording_start_time": "TEXT",  # real clock time of the video's first frame, if known (ISO 8601)
    "source": "TEXT",                # upload | cctv_adapter   (NULL = upload)
}

# Media columns added later. sha256 = content hash of the received file, used to tell the user that the
# same video was already received (duplicate delivery). The database is backed up before it is added.
MEDIA_COLUMNS = {
    "sha256": "TEXT",
}

JSON_COLS = {"area", "settings", "live", "files", "plan", "failures", "summary"}
CAMERA_COLS = {"id", "store_id", "name", "source_type", "source", "area", "enabled", "cursor"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(analyses)")}
            for col, typ in ANALYSIS_COLUMNS.items():
                if col not in cols:
                    c.execute(f"ALTER TABLE analyses ADD COLUMN {col} {typ}")
            c.execute("CREATE INDEX IF NOT EXISTS ix_analyses_camera ON analyses(camera_id, created_at)")
            cols = {r[1] for r in c.execute("PRAGMA table_info(extractions)")}
            for col, typ in EXTRACTION_COLUMNS.items():
                if col not in cols:
                    c.execute(f"ALTER TABLE extractions ADD COLUMN {col} {typ}")
            cols = {r[1] for r in c.execute("PRAGMA table_info(media)")}
            missing = {k: v for k, v in MEDIA_COLUMNS.items() if k not in cols}
            if missing and c.execute("SELECT COUNT(*) FROM media").fetchone()[0]:
                self._backup(c, "pre-media-sha256")
            for col, typ in missing.items():
                c.execute(f"ALTER TABLE media ADD COLUMN {col} {typ}")
            c.execute("CREATE INDEX IF NOT EXISTS ix_media_sha256 ON media(sha256)")

    def _backup(self, c: sqlite3.Connection, tag: str) -> Path:
        """Consistent copy of the whole database next to it, before a schema change."""
        dst = self.path.with_name(f"{self.path.stem}.{tag}-{datetime.now():%Y%m%d-%H%M%S}.db")
        with sqlite3.connect(dst) as b:
            c.backup(b)
        return dst

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    @staticmethod
    def _row(r: sqlite3.Row | None) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        for k in JSON_COLS & d.keys():
            if d[k]:
                d[k] = json.loads(d[k])
        return d

    def _write(self, sql: str, args: tuple | list = ()) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute(sql, args)
            return cur.lastrowid

    # ------------------------------------------------------------------ media
    def add_media(self, m: dict) -> None:
        self._write(f"INSERT INTO media ({','.join(m)}) VALUES ({','.join('?' * len(m))})", list(m.values()))

    def media(self, mid: str) -> dict | None:
        with self._conn() as c:
            return self._row(c.execute("SELECT * FROM media WHERE id=?", (mid,)).fetchone())

    def media_list(self, kind: str | None = None) -> list[dict]:
        with self._conn() as c:
            q = "SELECT * FROM media" + (" WHERE kind=?" if kind else "") + " ORDER BY created_at"
            return [self._row(r) for r in c.execute(q, (kind,) if kind else ())]

    def media_by_sha256(self, sha256: str, exclude: str | None = None) -> list[dict]:
        with self._conn() as c:
            return [self._row(r) for r in c.execute("SELECT * FROM media WHERE sha256=? AND id<>? ORDER BY created_at",
                                                    (sha256, exclude or ""))]

    # ------------------------------------------------------------------ analyses
    def add_analysis(self, a: dict) -> None:
        a = {k: (json.dumps(v) if k in JSON_COLS and v is not None else v) for k, v in a.items()}
        self._write(f"INSERT INTO analyses ({','.join(a)}) VALUES ({','.join('?' * len(a))})", list(a.values()))

    def update_analysis(self, aid: str, **fields: Any) -> None:
        if not fields:
            return
        vals = [json.dumps(v) if k in JSON_COLS and v is not None else v for k, v in fields.items()]
        self._write(f"UPDATE analyses SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", [*vals, aid])

    def analysis(self, aid: str) -> dict | None:
        with self._conn() as c:
            return self._row(c.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone())

    def analyses(self, limit: int = 50, offset: int = 0, kind: str | None = None, status: str | None = None,
                 q: str | None = None, start: str | None = None, camera_id: str | None = None,
                 store_id: str | None = None, source: str | None = None) -> tuple[list[dict], int]:
        where, args = ["1=1"], []
        for col, val in (("kind", kind), ("status", status), ("camera_id", camera_id), ("store_id", store_id)):
            if val:
                where.append(f"{col}=?"); args.append(val)
        if source:
            where.append("COALESCE(source, 'upload')=?"); args.append(source)
        if q:
            where.append("filename LIKE ?"); args.append(f"%{q}%")
        if start:
            where.append("created_at>=?"); args.append(start)
        w = " AND ".join(where)
        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM analyses WHERE {w}", args).fetchone()[0]
            rows = c.execute(f"SELECT * FROM analyses WHERE {w} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                             [*args, limit, offset]).fetchall()
        return [self._row(r) for r in rows], total

    def latest_done(self) -> dict | None:
        with self._conn() as c:
            return self._row(c.execute("SELECT * FROM analyses WHERE state='done' ORDER BY finished_at DESC LIMIT 1").fetchone())

    def interrupt_running(self) -> None:
        """Jobs that were running when the server stopped cannot resume."""
        self._write("UPDATE analyses SET state='failed', error='Server restarted during analysis' "
                    "WHERE state IN ('queued','running')")

    # ------------------------------------------------------------------ persons
    def set_persons(self, aid: str, rows: list[dict]) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM persons WHERE analysis_id=?", (aid,))
            for r in rows:
                r = {"analysis_id": aid, **r}
                c.execute(f"INSERT INTO persons ({','.join(r)}) VALUES ({','.join('?' * len(r))})", list(r.values()))

    def persons(self, aid: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM persons WHERE analysis_id=? ORDER BY track_id", (aid,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["box"] = json.loads(d["box"]) if d.get("box") else None
            out.append(d)
        return out

    # ------------------------------------------------------------------ alerts
    def add_alert(self, a: dict) -> int:
        a = {"created_at": now_iso(), **a}
        return self._write(f"INSERT INTO alerts ({','.join(a)}) VALUES ({','.join('?' * len(a))})", list(a.values()))

    def alerts(self, limit: int = 100, offset: int = 0, state: str | None = None, kind: str | None = None,
               analysis_id: str | None = None, start: str | None = None) -> tuple[list[dict], int]:
        where, args = ["1=1"], []
        for col, val in (("state", state), ("kind", kind), ("analysis_id", analysis_id)):
            if val:
                where.append(f"{col}=?"); args.append(val)
        if start:
            where.append("created_at>=?"); args.append(start)
        w = " AND ".join(where)
        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM alerts WHERE {w}", args).fetchone()[0]
            rows = c.execute(f"SELECT * FROM alerts WHERE {w} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                             [*args, limit, offset]).fetchall()
        return [dict(r) for r in rows], total

    def alert(self, alert_id: int) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return dict(r) if r else None

    def set_alert_state(self, alert_id: int, state: str) -> None:
        self._write("UPDATE alerts SET state=? WHERE id=?", (state, alert_id))

    # ------------------------------------------------------------------ cameras
    def cameras(self) -> list[dict]:
        with self._conn() as c:
            return [self._camera(r) for r in c.execute("SELECT * FROM cameras ORDER BY store_id, id")]

    def camera(self, cam_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM cameras WHERE id=?", (cam_id,)).fetchone()
        return self._camera(r) if r else None

    @staticmethod
    def _camera(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["area"] = json.loads(d["area"]) if d.get("area") else None
        d["enabled"] = bool(d["enabled"])
        return d

    def upsert_camera(self, cam: dict) -> None:
        """Insert or update; unknown keys are ignored. `area` is stored as JSON."""
        cam = {k: v for k, v in cam.items() if k in CAMERA_COLS}
        if "area" in cam:
            cam["area"] = json.dumps(cam["area"]) if cam["area"] is not None else None
        if "enabled" in cam:
            cam["enabled"] = int(bool(cam["enabled"]))
        now = now_iso()
        with self._lock, self._conn() as c:
            exists = c.execute("SELECT 1 FROM cameras WHERE id=?", (cam["id"],)).fetchone()
            if exists:
                sets = {k: v for k, v in cam.items() if k != "id"} | {"updated_at": now}
                c.execute(f"UPDATE cameras SET {', '.join(f'{k}=?' for k in sets)} WHERE id=?",
                          [*sets.values(), cam["id"]])
            else:
                row = cam | {"created_at": now, "updated_at": now}
                c.execute(f"INSERT INTO cameras ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                          list(row.values()))

    def set_camera_cursor(self, cam_id: str, cursor: float) -> None:
        self._write("UPDATE cameras SET cursor=? WHERE id=?", (cursor, cam_id))

    def delete_camera(self, cam_id: str) -> None:
        """Removes the camera only; its captures and analyses are kept as history."""
        self._write("DELETE FROM cameras WHERE id=?", (cam_id,))

    # ------------------------------------------------------------------ captures
    def add_capture(self, cap: dict) -> None:
        self._write(f"INSERT INTO captures ({','.join(cap)}) VALUES ({','.join('?' * len(cap))})", list(cap.values()))

    def update_capture(self, cap_id: str, **fields: Any) -> None:
        if fields:
            self._write(f"UPDATE captures SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                        [*fields.values(), cap_id])

    def capture(self, cap_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM captures WHERE id=?", (cap_id,)).fetchone()
        return dict(r) if r else None

    def captures(self, limit: int = 100, offset: int = 0, camera_id: str | None = None, store_id: str | None = None,
                 status: str | None = None, start: str | None = None, end: str | None = None) -> tuple[list[dict], int]:
        where, args = ["1=1"], []
        for col, val in (("camera_id", camera_id), ("store_id", store_id), ("status", status)):
            if val:
                where.append(f"{col}=?"); args.append(val)
        if start:
            where.append("captured_at>=?"); args.append(start)
        if end:
            where.append("captured_at<?"); args.append(end)
        w = " AND ".join(where)
        with self._conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM captures WHERE {w}", args).fetchone()[0]
            rows = c.execute(f"SELECT * FROM captures WHERE {w} ORDER BY captured_at DESC, id DESC LIMIT ? OFFSET ?",
                             [*args, limit, offset]).fetchall()
        return [dict(r) for r in rows], total

    def last_capture(self, cam_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM captures WHERE camera_id=? AND frame_hash IS NOT NULL "
                          "ORDER BY captured_at DESC, id DESC LIMIT 1", (cam_id,)).fetchone()
        return dict(r) if r else None

    def capture_slot_taken(self, cam_id: str, slot_at: str) -> bool:
        """True if this camera already has a capture for this scheduled slot (no duplicates after restarts)."""
        with self._conn() as c:
            return c.execute("SELECT 1 FROM captures WHERE camera_id=? AND slot_at=?",
                             (cam_id, slot_at)).fetchone() is not None

    def interrupt_captures(self) -> None:
        self._write("UPDATE captures SET status='capture_failed', error='Server restarted during capture' "
                    "WHERE status='running'")

    # ------------------------------------------------------------------ frame extractions
    def add_extraction(self, e: dict) -> None:
        e = {k: (json.dumps(v) if k in JSON_COLS and v is not None else v) for k, v in e.items()}
        self._write(f"INSERT INTO extractions ({','.join(e)}) VALUES ({','.join('?' * len(e))})", list(e.values()))

    def update_extraction(self, eid: str, **fields: Any) -> None:
        if fields:
            vals = [json.dumps(v) if k in JSON_COLS and v is not None else v for k, v in fields.items()]
            self._write(f"UPDATE extractions SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", [*vals, eid])

    def extraction(self, eid: str) -> dict | None:
        with self._conn() as c:
            return self._row(c.execute("SELECT * FROM extractions WHERE id=?", (eid,)).fetchone())

    def extractions(self, media_id: str | None = None, limit: int = 50) -> list[dict]:
        sql, args = "SELECT * FROM extractions", []
        if media_id:
            sql += " WHERE media_id=?"; args.append(media_id)
        with self._conn() as c:
            return [self._row(r) for r in c.execute(f"{sql} ORDER BY created_at DESC, id DESC LIMIT ?", [*args, limit])]

    # ------------------------------------------------------------------ uniform detection on extracted frames
    def add_detection_run(self, d: dict) -> None:
        self._write(f"INSERT INTO frame_detection_runs ({','.join(d)}) VALUES ({','.join('?' * len(d))})", list(d.values()))

    def update_detection_run(self, rid: str, **fields: Any) -> None:
        if fields:
            self._write(f"UPDATE frame_detection_runs SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                        [*fields.values(), rid])

    def detection_run(self, rid: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM frame_detection_runs WHERE id=?", (rid,)).fetchone()
        return dict(r) if r else None

    def detection_runs(self, extraction_id: str) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM frame_detection_runs WHERE extraction_id=? "
                                               "ORDER BY created_at DESC, id DESC", (extraction_id,))]

    def interrupt_detection_runs(self) -> None:
        self._write("UPDATE frame_detection_runs SET state='interrupted', error='Server stopped during detection; "
                    "results written before that are kept in detection_results.csv' WHERE state IN ('queued','running')")

    def interrupt_extractions(self) -> None:
        """Jobs that were queued/running when the server stopped. Frames already saved stay listed in
        their manifest; the job itself cannot resume."""
        self._write("UPDATE extractions SET state='interrupted', error='Server stopped during extraction; "
                    "frames saved before that are kept in the manifest' WHERE state IN ('queued','running')")

    # ------------------------------------------------------------------ admin uploads to S3
    def add_s3_upload(self, u: dict) -> None:
        u = {k: (json.dumps(v) if k in JSON_COLS and v is not None else v) for k, v in u.items()}
        self._write(f"INSERT INTO s3_uploads ({','.join(u)}) VALUES ({','.join('?' * len(u))})", list(u.values()))

    def update_s3_upload(self, uid: str, **fields: Any) -> None:
        fields["updated_at"] = now_iso()
        vals = [json.dumps(v) if k in JSON_COLS and v is not None else v for k, v in fields.items()]
        self._write(f"UPDATE s3_uploads SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", [*vals, uid])

    def s3_upload(self, uid: str) -> dict | None:
        with self._conn() as c:
            return self._row(c.execute("SELECT * FROM s3_uploads WHERE id=?", (uid,)).fetchone())

    def s3_uploads(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            return [self._row(r) for r in c.execute("SELECT * FROM s3_uploads ORDER BY created_at DESC, id DESC LIMIT ?",
                                                     (limit,))]

    def s3_upload_by_sha256(self, sha256: str, cart: str) -> dict | None:
        """The latest upload of the same video (same content) for the same cart."""
        with self._conn() as c:
            return self._row(c.execute("SELECT * FROM s3_uploads WHERE sha256=? AND cart=? ORDER BY created_at DESC, "
                                       "id DESC LIMIT 1", (sha256, cart)).fetchone())

    def interrupt_s3_uploads(self) -> None:
        """Uploads that were in progress when the server stopped: failed in that stage; Retry resumes them."""
        self._write("UPDATE s3_uploads SET failed_stage=state, state='failed', updated_at=?, "
                    "error='Server stopped during ' || replace(state, '_', ' ') || '; use Retry' "
                    "WHERE state NOT IN ('completed','failed')", (now_iso(),))

    # ------------------------------------------------------------------ analytics
    def analytics(self, start: str | None = None) -> dict:
        w, args = ("WHERE state='done'" + (" AND created_at>=?" if start else "")), ([start] if start else [])
        with self._conn() as c:
            agg = dict(c.execute(f"""
                SELECT COUNT(*) AS analyses,
                       SUM(status IN ('compliant','no_uniform','uncertain')) AS with_people,
                       SUM(status='compliant') AS compliant, SUM(status='no_uniform') AS no_uniform,
                       SUM(status='uncertain') AS uncertain_status, SUM(status='no_person') AS no_person,
                       SUM(people) AS people, SUM(uniformed) AS uniformed, SUM(non_uniform) AS non_uniform,
                       SUM(uncertain) AS uncertain, SUM(alerts) AS alerts,
                       SUM(uniform_obs) AS uniform_obs, SUM(non_uniform_obs) AS non_uniform_obs,
                       SUM(uncertain_obs) AS uncertain_obs,
                       SUM(kind='video') AS videos, SUM(kind='image') AS images,
                       SUM(processing_sec) AS processing_sec, SUM(duration_sec) AS footage_sec
                FROM analyses {w}""", args).fetchone())
            # average uniform confidence over confirmed uniformed PEOPLE (not analyses)
            conf = c.execute(f"""SELECT AVG(p.confidence) FROM persons p JOIN analyses a ON a.id=p.analysis_id
                                 {w.replace('state', 'a.state').replace('created_at', 'a.created_at')} AND p.label='uniform'""",
                             args).fetchone()[0]
            by_day = [dict(r) for r in c.execute(f"""
                SELECT substr(created_at,1,10) AS day, COUNT(*) AS analyses,
                       SUM(status='compliant') AS compliant,
                       SUM(status IN ('compliant','no_uniform','uncertain')) AS with_people
                FROM analyses {w} GROUP BY day ORDER BY day""", args)]
        agg = {k: (v or 0) for k, v in agg.items()}
        agg["avg_uniform_confidence"] = conf
        agg["compliance_rate"] = (agg["compliant"] / agg["with_people"]) if agg["with_people"] else None
        agg["by_day"] = by_day
        return agg
