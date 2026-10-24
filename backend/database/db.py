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
CREATE INDEX IF NOT EXISTS ix_analyses_created ON analyses(created_at);
CREATE INDEX IF NOT EXISTS ix_alerts_created ON alerts(created_at);
"""

JSON_COLS = {"area", "settings", "live", "files"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._conn() as c:
            c.executescript(SCHEMA)
            # columns added after the first release
            cols = {r[1] for r in c.execute("PRAGMA table_info(analyses)")}
            if "model_version" not in cols:
                c.execute("ALTER TABLE analyses ADD COLUMN model_version TEXT")

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
                 q: str | None = None, start: str | None = None) -> tuple[list[dict], int]:
        where, args = ["1=1"], []
        if kind:
            where.append("kind=?"); args.append(kind)
        if status:
            where.append("status=?"); args.append(status)
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
