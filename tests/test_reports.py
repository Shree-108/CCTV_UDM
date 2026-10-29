"""Latest / monthly / historical business reports (backend.extraction.reports_service, /api/reports).

The suite shares one database, so these tests move their own runs into 2031 (finished_at) to get months
that contain only their analyses."""
import csv
import io
import sqlite3
from datetime import timedelta, timezone

import pytest

from backend.database.db import DB
from backend.extraction import detect, master_csv
from backend.extraction import reports_service as RS
from tests.test_compliance_reports import Person, run_job
from tests.test_extraction import MEDIA
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)

HEADER = ["Store Name", "Date", "Time Stamp", "img name", "confidence score", "remark (Example)"]


@pytest.fixture
def utc(monkeypatch):
    monkeypatch.setattr(RS, "REPORT_TZ", timezone.utc)


@pytest.fixture
def one_uniform(monkeypatch):
    monkeypatch.setattr(detect, "classify_frame", lambda img, models, cfg, thr: [Person((100, 80, 220, 460), "uniform", 0.9)])


def table(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


def finish_at(db, run_id: str, ts: str) -> None:
    db._write("UPDATE frame_detection_runs SET finished_at=? WHERE id=?", (ts, run_id))


def job_rows(client, job_id: str) -> list[list[str]]:
    return table(client.get(f"/api/extractions/{job_id}/uniform_compliance.csv").text)[1:]


@pytest.fixture
def three_in_2031(client, real_model, one_uniform, utc):  # noqa: F811
    db = client.app.state.app_state.db
    j1, r1 = run_job(client, {"random_seed": 101, "store_id": "store id 1", "camera_id": "CAM-R1",
                              "recording_start_time": "2031-03-05T09:00:00"})
    j2, r2 = run_job(client, {"random_seed": 102, "store_id": "store id 2", "camera_id": "CAM-R2"})
    j3, r3 = run_job(client, {"random_seed": 103, "camera_id": "CAM-R3"})
    finish_at(db, r1["id"], "2031-01-31T23:59:59+00:00")        # last second of January
    finish_at(db, r2["id"], "2031-02-01T00:00:00+00:00")        # first second of February
    finish_at(db, r3["id"], "2031-02-15T10:00:00+00:00")
    return db, (j1, r1), (j2, r2), (j3, r3)


def test_month_boundaries_and_monthly_csv(client, three_in_2031):
    db, (j1, _), (j2, _), (j3, _) = three_in_2031
    jan = client.get("/api/reports/months/2031-01").json()
    feb = client.get("/api/reports/months/2031-02").json()
    assert [a["job_id"] for a in jan["analyses_list"]] == [j1["id"]]
    assert sorted(a["job_id"] for a in feb["analyses_list"]) == sorted([j2["id"], j3["id"]])
    assert (jan["analyses_completed"], jan["analyses_failed"], jan["screenshots_extracted"], jan["screenshots_analysed"]) == (1, 0, 4, 4)
    assert (feb["analyses_completed"], feb["uniform"], feb["no_uniform"], feb["review"], feb["business_records"]) == (2, 8, 0, 0, 8)
    assert jan["videos_received"] == 0                  # received in 2026, not in January 2031 (receipt date rule)
    assert "completion" in jan["by"] and "finished" in jan["rule"]

    # the monthly CSV: exact schema, ALL rows of the month (not just the latest four), in completion order
    r = client.get("/api/reports/months/2031-02.csv", params={"download": "true"})
    assert 'filename="uniform_compliance_2031-02.csv"' in r.headers["content-disposition"]
    header, *rows = table(r.text)
    assert header == HEADER and rows == job_rows(client, j2["id"]) + job_rows(client, j3["id"])
    assert table(client.get("/api/reports/months/2031-01.csv").text)[1:] == job_rows(client, j1["id"])

    # recording-date grouping: only the analysis whose recording time was supplied, in ITS month
    rec = client.get("/api/reports/months/2031-03", params={"by": "recording"}).json()
    assert [a["job_id"] for a in rec["analyses_list"]] == [j1["id"]] and rec["business_records"] == 4
    assert all(row[1] == "2031-03-05" for row in table(client.get("/api/reports/months/2031-03.csv", params={"by": "recording"}).text)[1:])
    ov = client.get("/api/reports/overview").json()
    assert {"2031-01", "2031-02"} <= set(ov["months"]["completion"]) and "2031-03" in ov["months"]["recording"]
    assert "2031-03" not in ov["months"]["completion"]

    # months without records are not invented
    assert client.get("/api/reports/months/2031-04").status_code == 404
    assert client.get("/api/reports/months/2031-04.csv").status_code == 404
    assert client.get("/api/reports/months/2031-13").status_code == 422


def test_month_uses_the_report_time_zone(client, three_in_2031, monkeypatch):
    db, (j1, _), _, _ = three_in_2031
    monkeypatch.setattr(RS, "REPORT_TZ", timezone(timedelta(hours=5, minutes=30)))     # IST
    # 23:59:59 UTC on 31 January is 05:29:59 on 1 February in India
    assert j1["id"] in [a["job_id"] for a in client.get("/api/reports/months/2031-02").json()["analyses_list"]]
    jan = client.get("/api/reports/months/2031-01")
    assert jan.status_code == 404 or j1["id"] not in [a["job_id"] for a in jan.json()["analyses_list"]]


def test_latest_analysis_and_a_failed_newest_job(client, three_in_2031, monkeypatch):
    db, _, _, (j3, r3) = three_in_2031
    ov = client.get("/api/reports/overview").json()
    latest = ov["latest"]
    assert (latest["job_id"], latest["run_id"], latest["status"]) == (j3["id"], r3["id"], "completed")
    assert latest["store_id"] is None and latest["recording_date"] is None          # unknown metadata stays unknown
    assert len(latest["records"]) == 4 and all(client.get(x["original_url"]).status_code == 200 for x in latest["records"])
    # the latest-analysis CSV holds that analysis only
    header, *rows = table(client.get(latest["csv_url"]).text)
    assert header == HEADER and rows == job_rows(client, j3["id"])
    assert {r[0] for r in rows} == {""}                                             # store unknown -> blank
    before = ov["summary"]

    def crash(*a):
        raise RuntimeError("simulated inference crash")
    monkeypatch.setattr(detect, "classify_frame", crash)
    jf, rf = run_job(client, {"random_seed": 104, "camera_id": "CAM-R4"})
    assert rf["state"] == "failed"
    ov = client.get("/api/reports/overview").json()
    assert ov["latest"]["job_id"] == j3["id"]                                      # still the last SUCCESSFUL one
    # the newest job is reported separately (in this shared test database another test leaves a job dated
    # 2099, so look the failed one up by its month), never as the latest result
    assert ov["newest_other"]["status"] != "completed"
    db.update_extraction(jf["id"], created_at="2099-12-31T00:00:00+00:00")
    newest = client.get("/api/reports/overview").json()["newest_other"]
    assert newest["job_id"] == jf["id"] and newest["status"] == "failed" and "Inference failed" in newest["error"]
    month = RS._month(client.get(f"/api/detection-runs/{rf['id']}").json()["finished_at"])
    listed = {a["job_id"]: a for a in client.get(f"/api/reports/months/{month}").json()["analyses_list"]}
    assert listed[jf["id"]]["status"] == "failed"
    s = ov["summary"]
    assert s["analyses_failed"] == before["analyses_failed"] + 1 and s["analyses_completed"] == before["analyses_completed"]
    assert s["analyses"] == before["analyses"] + 1 and s["videos_received"] == before["videos_received"] + 1
    assert s["screenshots_extracted"] == before["screenshots_extracted"] + 4
    assert s["screenshots_analysed"] == before["screenshots_analysed"]               # none of the failed ones
    assert jf["id"] not in {r["job_id"] for r in client.get("/api/reports/records", params={"limit": 5000}).json()["records"]}


def test_historical_records_are_the_master_csv(client, real_model, one_uniform, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(master_csv, "EXPORTS", tmp_path / "exports")
    run_job(client, {"random_seed": 111, "camera_id": "CAM-R5"})
    master_csv.sync(client.app.state.app_state.db)
    master = table(client.get("/api/uniform-compliance/master.csv").text)
    hist = client.get("/api/reports/records", params={"limit": 5000}).json()
    assert master[0] == HEADER and hist["fields"] == HEADER
    assert sorted(master[1:]) == sorted([list(r["row"].values()) for r in hist["records"]])
    assert hist["total"] == len(master) - 1
    one = client.get("/api/reports/records", params={"result": "UNIFORM_DETECTED", "q": "person 1 detected"}).json()
    assert one["total"] and all(r["prediction"] == "UNIFORM_DETECTED" for r in one["records"])


# ------------------------------------------------------------------ receipt / idempotency / migration
def test_same_video_delivered_twice_is_recognised(client):
    data = (MEDIA / "dvr_grid.mp4").read_bytes()
    a = client.post("/api/media", files={"file": ("dvr_grid.mp4", data)}).json()
    b = client.post("/api/media", files={"file": ("store_cam_copy.mp4", data)}).json()
    assert a["id"] != b["id"] and a["sha256"] == b["sha256"] and len(a["sha256"]) == 64
    assert a["id"] in {p["media_id"] for p in b["previously_received"]}                # told, not merged
    assert a["id"] in {p["media_id"] for p in client.get(f"/api/media/{b['id']}").json()["previously_received"]}
    c = client.post("/api/media", files={"file": ("other.mp4", data + b"\x00")})        # different bytes
    assert c.status_code in (200, 422)
    if c.status_code == 200:
        assert c.json()["previously_received"] == []
    assert "path" not in b                                                             # no server paths exposed


def test_database_is_backed_up_before_the_media_migration(tmp_path):
    p = tmp_path / "old.db"
    with sqlite3.connect(p) as c:
        c.execute("CREATE TABLE media (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL, filename TEXT NOT NULL,"
                  " path TEXT NOT NULL, size_bytes INTEGER, width INTEGER, height INTEGER, fps REAL, frames INTEGER,"
                  " duration_sec REAL, suggested_layout TEXT)")
        c.execute("INSERT INTO media VALUES ('m1','2026-01-01T00:00:00+00:00','video','a.mp4','x',1,2,2,25,10,0.4,'1x1')")
    db = DB(p)
    backups = list(tmp_path.glob("old.pre-media-sha256-*.db"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as c:
        assert [r[0] for r in c.execute("SELECT id FROM media")] == ["m1"]
        assert "sha256" not in {r[1] for r in c.execute("PRAGMA table_info(media)")}
    assert db.media("m1")["sha256"] is None and db.media("m1")["filename"] == "a.mp4"   # existing record kept
    DB(p)                                                                                 # second start: no new backup
    assert len(list(tmp_path.glob("old.pre-media-sha256-*.db"))) == 1
