"""Cumulative uniform_compliance_master.csv, the annotated-video link and the simplified navigation.

Each test points the master CSV at its own folder (the API session is shared by the whole suite)."""
import csv
import io
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.extraction import detect, master_csv
from backend.extraction.business_csv import BUSINESS_FIELDS
from tests.test_compliance_reports import Person, csv_rows, run_job
from tests.test_extraction import MEDIA
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)

HEADER = ["Store Name", "Date", "Time Stamp", "img name", "confidence score", "remark (Example)"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def master_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(master_csv, "EXPORTS", tmp_path / "exports")
    monkeypatch.setattr(master_csv, "_last_error", None)
    return tmp_path / "exports"


@pytest.fixture
def one_uniform(monkeypatch):
    """Deterministic model output: one person in the target uniform at 92.5 %."""
    monkeypatch.setattr(detect, "classify_frame", lambda img, models, cfg, thr: [Person((100, 80, 220, 460), "uniform", 0.925)])


def table(p: Path) -> list[list[str]]:
    return list(csv.reader(io.StringIO(p.read_text(encoding="utf-8"))))


def ledger_jobs() -> list[str]:
    return [e["job_id"] for e in master_csv._ledger()]


def three_jobs(client, seed=0):
    return [run_job(client, {"random_seed": seed + 1, "store_id": "store id 21", "camera_id": "CAM-M1",
                             "recording_start_time": "2026-10-01T09:30:00"}),
            run_job(client, {"random_seed": seed + 2, "store_id": "STORE-002", "camera_id": "CAM-M2"}),
            run_job(client, {"random_seed": seed + 3, "camera_id": "CAM-M3"})]          # store unknown


# ------------------------------------------------------------------ cumulative appends
def test_three_videos_append_to_one_file(client, real_model, one_uniform, master_dir):  # noqa: F811
    jobs = three_jobs(client)
    p = master_csv.master_path()
    rows = table(p)
    assert rows[0] == HEADER == BUSINESS_FIELDS                 # exact six columns, in order
    assert sum(r == HEADER for r in rows) == 1                  # header written once
    body = rows[1:]
    assert len(body) == 12 and all(len(r) == 6 for r in body)   # 4 screenshots x 3 videos
    assert ledger_jobs() == [j["id"] for j, _ in jobs]           # appended in processing order
    for k, (j, run) in enumerate(jobs):
        man = csv_rows(client, j["manifest_url"])
        part = [dict(zip(HEADER, r)) for r in body[4 * k: 4 * k + 4]]
        assert [x["img name"] for x in part] == [m["image_filename"] for m in man]
        assert [x["Time Stamp"] for x in part] == [m["video_timestamp_hhmmss"][:8] for m in man]
        assert all(x["confidence score"] == "92.5%" for x in part)
        assert all(x["remark (Example)"] == "Person 1 detected wearing the designated uniform shirt." for x in part)
    stores = [r[0] for r in body]
    dates = [r[1] for r in body]
    assert stores == ["store id 21"] * 4 + ["STORE-002"] * 4 + [""] * 4      # unknown store stays blank
    assert dates == ["2026-10-01"] * 4 + [""] * 8                             # date only when it is known

    # the master file equals the three per-video files concatenated (same rows, same rules)
    per_video = [client.get(f"/api/extractions/{j['id']}/uniform_compliance.csv").text for j, _ in jobs]
    assert [r for t in per_video for r in list(csv.reader(io.StringIO(t)))[1:]] == body

    # download = the complete file, byte for byte
    r = client.get("/api/uniform-compliance/master.csv", params={"download": "true"})
    assert r.status_code == 200 and r.content == p.read_bytes()
    assert 'filename="uniform_compliance_master.csv"' in r.headers["content-disposition"]
    s = client.get("/api/uniform-compliance/master").json()
    assert (s["rows"], s["videos"], s["exists"], s["fields"], s["last_error"]) == (12, 3, True, HEADER, None)
    assert s["path"].endswith("exports/uniform_compliance_master.csv") and "S3 is not implemented" in s["storage"]
    assert not {x["job_id"] for x in s["pending_jobs"]} & {j["id"] for j, _ in jobs}

    # retry: a second detection run of the same job appends nothing, and neither does a sync
    before = p.read_bytes()
    r = client.post(f"/api/extractions/{jobs[0][0]['id']}/detect", json={"new_run": True})
    assert r.status_code == 202
    rid = r.json()["id"]
    for _ in range(400):
        if client.get(f"/api/detection-runs/{rid}").json()["state"] not in ("queued", "running"):
            break
        time.sleep(0.25)
    assert client.get(f"/api/detection-runs/{rid}").json()["state"] == "done"
    assert p.read_bytes() == before                      # the retried run appended nothing
    client.post("/api/uniform-compliance/master/sync")   # (may add OTHER tests' finished jobs - never these)
    after = p.read_bytes()
    assert after.startswith(before)                      # append-only
    names = [r[3] for r in table(p)[1:] if r[3]]          # blank = no image (extraction failed / frame unreadable)
    assert len(names) == len(set(names)) and ledger_jobs().count(jobs[0][0]["id"]) == 1


def test_concurrent_appends_do_not_interleave(client, real_model, one_uniform, master_dir):  # noqa: F811
    db = client.app.state.app_state.db
    jobs = three_jobs(client, seed=10)
    for f in master_dir.iterdir():                     # start from an empty folder, then append from 9 threads
        f.unlink()
    pairs = [(db.extraction(j["id"]), db.detection_run(r["id"])) for j, r in jobs] * 3
    errors = []

    def go(ext, run):
        try:
            master_csv.append(ext, run)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=go, args=x) for x in pairs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    rows = table(master_csv.master_path())
    assert rows[0] == HEADER and sum(r == HEADER for r in rows) == 1
    body = rows[1:]
    assert len(body) == 12                               # each job once, despite 3 attempts each
    names = [r[3] for r in body]
    assert len(set(names)) == 12
    for k in range(3):                                   # every job's four rows are contiguous
        assert len({n.split("_r0")[0] for n in names[4 * k: 4 * k + 4]}) == 1
    assert sorted(ledger_jobs()) == sorted(j["id"] for j, _ in jobs)


def test_interrupted_append_is_repaired_not_duplicated(client, real_model, one_uniform, master_dir):  # noqa: F811
    db = client.app.state.app_state.db
    j1, r1 = run_job(client, {"random_seed": 31, "camera_id": "CAM-M4"})
    p = master_csv.master_path()
    good = p.read_bytes()
    with open(p, "ab") as f:                             # rows written, ledger entry never written (crash)
        f.write(b"half,a,row")
    j2, r2 = run_job(client, {"random_seed": 32, "camera_id": "CAM-M4"})
    rows = table(p)
    assert p.read_bytes().startswith(good) and b"half,a,row" not in p.read_bytes()
    assert len(rows) == 1 + 8 and ledger_jobs() == [j1["id"], j2["id"]]

    # a file shorter than its ledger was changed outside the app: refuse to write, report it
    p.write_bytes(good[:20])
    with pytest.raises(master_csv.MasterCsvError):
        master_csv.append(db.extraction(j2["id"]), db.detection_run(r2["id"]) | {"id": "det-x"})
    assert p.read_bytes() == good[:20]
    assert "shorter than its ledger" in client.get("/api/uniform-compliance/master").json()["last_error"]["message"]


def test_write_failure_is_reported_and_recovered(client, real_model, one_uniform, master_dir, monkeypatch):  # noqa: F811
    blocked = master_dir.parent / "not_a_folder"
    blocked.write_text("x")                              # the export folder cannot be created
    monkeypatch.setattr(master_csv, "EXPORTS", blocked)
    j, run = run_job(client, {"random_seed": 41, "camera_id": "CAM-M5"})
    assert run["state"] == "done"                        # the job's own results are kept
    assert client.get(f"/api/extractions/{j['id']}/uniform_compliance.csv").status_code == 200
    s = client.get("/api/uniform-compliance/master").json()
    assert s["exists"] is False and s["last_error"]["job_id"] == j["id"]
    assert j["id"] in {x["job_id"] for x in s["pending_jobs"]}
    assert client.get("/api/uniform-compliance/master.csv").status_code == 404     # no fake download

    monkeypatch.setattr(master_csv, "EXPORTS", master_dir)                        # problem fixed: recover
    out = client.post("/api/uniform-compliance/master/sync").json()
    assert out["status"]["last_error"] is None and j["id"] not in {x["job_id"] for x in out["status"]["pending_jobs"]}
    assert j["id"] in ledger_jobs()


def test_failed_runs_are_not_appended(client, master_dir):
    """Suite default: trained weights missing -> the run fails -> nothing goes into the master CSV."""
    j, run = run_job(client, {"random_seed": 51, "camera_id": "CAM-M6"})
    assert run["state"] == "failed"
    assert master_csv.sync(client.app.state.app_state.db)["rows_appended"] >= 0
    assert j["id"] not in ledger_jobs()
    assert j["id"] not in {x["job_id"] for x in client.get("/api/uniform-compliance/master").json()["pending_jobs"]}


def test_csv_escaping_round_trips():
    rows = [{"Store Name": 'Store "A", Pune', "Date": "", "Time Stamp": "00:01:02", "img name": "café_t000062000.png",
             "confidence score": "Person 1: 92.5%; Person 2: not scored", "remark (Example)": "line one\nline two, ünïcode"}]
    back = list(csv.DictReader(io.StringIO(master_csv._rows_bytes(rows, header=True).decode("utf-8"))))
    assert back == rows


# ------------------------------------------------------------------ evidence and annotated video
def test_evidence_images_match_their_timestamps(client, real_model, one_uniform, master_dir):  # noqa: F811
    j, run = run_job(client, {"random_seed": 61, "camera_id": "CAM-M7"})
    media = client.app.state.app_state.db.media(j["media_id"])
    cap = cv2.VideoCapture(media["path"])
    recs = client.get(f"/api/extractions/{j['id']}/uniform-compliance").json()["records"]
    for rec, m in zip(recs, csv_rows(client, j["manifest_url"])):
        r = client.get(rec["original_url"])
        assert r.status_code == 200
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(m["frame_number"]))
        ok, frame = cap.read()
        assert ok and np.array_equal(img, frame)                     # the saved frame IS the frame at that time
        assert rec["row"]["Time Stamp"] == m["video_timestamp_hhmmss"][:8]
        assert re.search(rf"_t{int(round(float(m['video_timestamp_seconds']) * 1000)):09d}\.", rec["row"]["img name"])
        assert client.get(rec["annotated_url"]).status_code == 200
    cap.release()


def test_annotated_video_only_when_the_file_exists(client):
    j, _ = run_job(client, {"random_seed": 71, "camera_id": "CAM-M8"})
    v = client.get(f"/api/extractions/{j['id']}/annotated-video").json()
    assert (v["state"], v["available"], v["video_url"], v["download_url"]) == ("not_started", False, None, None)
    v = client.post(f"/api/extractions/{j['id']}/annotated-video").json()
    assert v["analysis_id"] and v["state"] in ("queued", "running", "done")
    assert client.post(f"/api/extractions/{j['id']}/annotated-video").json()["analysis_id"] == v["analysis_id"]  # no 2nd pass
    for _ in range(600):
        v = client.get(f"/api/extractions/{j['id']}/annotated-video").json()
        if v["state"] not in ("queued", "running"):
            break
        time.sleep(0.25)
    assert v["state"] == "done" and v["available"] is True
    r = client.get(v["download_url"])
    assert r.status_code == 200 and len(r.content) > 1000 and "attachment" in r.headers["content-disposition"]
    media = client.app.state.app_state.db.media(j["media_id"])
    assert Path(media["path"]).exists()                              # the original is untouched
    a = client.app.state.app_state.db.analysis(v["analysis_id"])
    Path(a["files"]["annotated"]).unlink()                           # output lost -> no link is offered
    v = client.get(f"/api/extractions/{j['id']}/annotated-video").json()
    assert v["available"] is False and v["download_url"] is None and "missing or unreadable" in v["error"]


def test_corrupt_video_is_rejected_at_upload(client):
    r = client.post("/api/media", files={"file": ("broken.mp4", b"\x00\x00\x00\x18ftypmp42" + b"not a video" * 200)})
    assert r.status_code == 422 and "could not be opened" in r.json()["detail"]


# ------------------------------------------------------------------ navigation
def test_navigation_is_result_oriented_and_routes_are_kept():
    src = (ROOT / "frontend" / "components" / "shell" / "nav.ts").read_text(encoding="utf-8")
    groups = re.findall(r"\{ label: '([^']+)'(, hidden: true)?, items: \[(.*?)\] \}", src, re.S)
    visible = [h for _, hidden, body in groups if not hidden for h in re.findall(r"href: '([^']+)'", body)]
    hidden = [h for _, hid, body in groups if hid for h in re.findall(r"href: '([^']+)'", body)]
    assert visible == ["/analysis/video", "/results", "/s3-upload", "/alerts", "/settings", "/help", "/cameras"]
    assert {"/", "/analysis/image", "/extract", "/analytics"} <= set(hidden)
    assert re.search(r"href: '/cameras'[^}]*comingSoon: true", src)            # still marked unavailable
    for h in hidden + visible[:-1]:                                            # every page still has its route
        assert (ROOT / "app" / h.strip("/") / "page.tsx").exists(), h
