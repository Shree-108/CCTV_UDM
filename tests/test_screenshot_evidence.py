"""Every business-CSV row links to the exact screenshot it was made from (job id + image id + run id).

Two jobs of the same video are run with different classifier outputs, so a row that opened the other
job's evidence would show the wrong result. Storage here is the temporary test folder (conftest)."""
import csv
import io
import time

import cv2
import numpy as np
import pytest

from backend.extraction import detect, master_csv
from backend.extraction.runner import EXTRACTIONS
from tests.test_compliance_reports import Person, run_job
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)

HEADER = ["Store Name", "Date", "Time Stamp", "img name", "confidence score", "remark (Example)"]


def uniform(monkeypatch):
    monkeypatch.setattr(detect, "classify_frame", lambda img, models, cfg, thr: [Person((100, 80, 220, 460), "uniform", 0.9)])


def nobody(monkeypatch):
    monkeypatch.setattr(detect, "classify_frame", lambda img, models, cfg, thr: [])


def records(client, job_id):
    return client.get(f"/api/extractions/{job_id}/uniform-compliance").json()["records"]


def frame_file(job_id, name):
    return EXTRACTIONS / job_id / "frames" / name


@pytest.fixture
def two_jobs(client, real_model, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(master_csv, "EXPORTS", tmp_path / "exports")
    uniform(monkeypatch)
    a = run_job(client, {"random_seed": 201, "store_id": "store A"})
    nobody(monkeypatch)
    b = run_job(client, {"random_seed": 202, "store_id": "store B"})
    master_csv.sync(client.app.state.app_state.db)
    return a, b


def test_each_row_opens_its_own_screenshot(client, two_jobs):
    for (job, run), expected in zip(two_jobs, ("UNIFORM_DETECTED", "NO_UNIFORM_DETECTED")):
        recs = records(client, job["id"])
        assert len(recs) == 4
        for r in recs:
            assert r["job_id"] == job["id"] and r["run_id"] == run["id"]
            assert f"/extractions/{job['id']}/evidence/{r['image_id']}?run_id={run['id']}" in r["evidence_url"]
            e = client.get(r["evidence_url"]).json()
            assert (e["job_id"], e["run_id"], e["image_id"], e["prediction"]) == (job["id"], run["id"], r["image_id"], expected)
            assert e["row"] == r["row"] and e["original"]["filename"] == r["row"]["img name"]
            assert e["original"]["available"] and e["annotated"]["available"]
            # the image served for this row is the job's own saved file, byte for byte
            img = client.get(r["original_url"])
            assert img.status_code == 200 and img.content == frame_file(job["id"], r["row"]["img name"]).read_bytes()
            ann = client.get(r["annotated_url"])
            assert ann.status_code == 200 and f"/detection-runs/{run['id']}/" in r["annotated_url"]
            # hover thumbnail: a small JPEG made in memory from the same file - nothing new on disk
            thumb = client.get(r["thumb_url"])
            assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
            assert cv2.imdecode(np.frombuffer(thumb.content, np.uint8), cv2.IMREAD_COLOR).shape[1] <= 360
        assert sorted(p.name for p in (EXTRACTIONS / job["id"] / "frames").iterdir()) == sorted(r["row"]["img name"] for r in recs)


def test_identical_file_names_in_two_videos_stay_apart(client, two_jobs):
    (a, _), (b, _) = two_jobs
    ra, rb = records(client, a["id"])[0], records(client, b["id"])[0]
    same = ra["row"]["img name"]
    # give job B's screenshot exactly job A's file name (different content), as another video could
    frame_file(b["id"], rb["row"]["img name"]).rename(frame_file(b["id"], same))
    cv2.imwrite(str(frame_file(b["id"], same)), np.full((120, 160, 3), 40, np.uint8))
    mp = EXTRACTIONS / b["id"] / "image_manifest.csv"
    rows = list(csv.DictReader(io.StringIO(mp.read_text(encoding="utf-8"))))
    for row in rows:
        if row["image_id"] == rb["image_id"]:
            row["image_filename"] = same
    with open(mp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    rb = next(r for r in records(client, b["id"]) if r["image_id"] == rb["image_id"])
    assert rb["row"]["img name"] == ra["row"]["img name"] == same
    ia, ib = client.get(ra["original_url"]).content, client.get(rb["original_url"]).content
    assert ia == frame_file(a["id"], same).read_bytes() and ib == frame_file(b["id"], same).read_bytes() and ia != ib
    ea, eb = client.get(ra["evidence_url"]).json(), client.get(rb["evidence_url"]).json()
    assert (ea["job_id"], ea["prediction"]) == (a["id"], "UNIFORM_DETECTED")
    assert (eb["job_id"], eb["prediction"]) == (b["id"], "NO_UNIFORM_DETECTED")


def test_historical_rows_keep_their_links_through_filters_and_reruns(client, two_jobs, monkeypatch):
    (a, ra), (b, rb) = two_jobs
    hist = client.get("/api/reports/records", params={"limit": 5000}).json()["records"]
    mine = [r for r in hist if r["job_id"] in (a["id"], b["id"])]
    assert len(mine) == 8
    for h in mine:
        e = client.get(h["evidence_url"]).json()
        assert (e["job_id"], e["run_id"], e["image_id"], e["row"]) == (h["job_id"], h["run_id"], h["image_id"], h["row"])
    # a server-side filter returns a subset whose links still belong to their own rows
    only_b = client.get("/api/reports/records", params={"result": "NO_UNIFORM_DETECTED", "store": "store B"}).json()["records"]
    assert {r["image_id"] for r in only_b if r["job_id"] == b["id"]} == {r["image_id"] for r in records(client, b["id"])}
    assert all(r["row"]["Store Name"] == "store B" and r["prediction"] == "NO_UNIFORM_DETECTED" for r in only_b)
    assert all(client.get(r["evidence_url"]).json()["image_id"] == r["image_id"] for r in only_b)

    # a new detection run on job A (different result) does not change what A's historical rows open
    nobody(monkeypatch)
    run2 = client.post(f"/api/extractions/{a['id']}/detect", json={"new_run": True}).json()
    for _ in range(400):
        if client.get(f"/api/detection-runs/{run2['id']}").json()["state"] not in ("queued", "running"):
            break
        time.sleep(0.25)
    again = [r for r in client.get("/api/reports/records", params={"limit": 5000}).json()["records"] if r["job_id"] == a["id"]]
    assert {r["run_id"] for r in again} == {ra["id"]}                     # the master-CSV run, not the newest
    for h in again:
        e = client.get(h["evidence_url"]).json()
        assert (e["run_id"], e["prediction"]) == (ra["id"], "UNIFORM_DETECTED")
        assert f"/detection-runs/{ra['id']}/" in h["annotated_url"]


def test_missing_screenshot_is_reported_not_replaced(client, two_jobs):
    (a, _), _ = two_jobs
    r = records(client, a["id"])[1]
    frame_file(a["id"], r["row"]["img name"]).unlink()
    e = client.get(r["evidence_url"]).json()
    assert e["original"]["available"] is False and e["original"]["filename"] == r["row"]["img name"]
    assert e["prediction"] == "UNIFORM_DETECTED"                         # the stored result is still shown
    assert client.get(r["original_url"]).status_code == 404 and client.get(r["thumb_url"]).status_code == 404


def test_evidence_cannot_be_read_across_jobs_or_outside_storage(client, two_jobs):
    (a, ra), (b, rb) = two_jobs
    xa, xb = records(client, a["id"])[0], records(client, b["id"])[0]
    assert client.get(f"/api/extractions/{a['id']}/evidence/{xb['image_id']}").status_code == 404          # B's image via A
    assert client.get(f"/api/extractions/{a['id']}/evidence/{xa['image_id']}", params={"run_id": rb["id"]}).status_code == 404
    assert client.get(f"/api/extractions/{a['id']}/frames/{xb['row']['img name']}").status_code == 404       # B's file via A
    ann = xb["annotated_url"].rsplit("/", 1)[1]
    assert client.get(f"/api/detection-runs/{ra['id']}/annotated/{ann}").status_code == 404
    for bad in ("..%2F..%2Fimage_manifest.csv", "..%5C..%5Cimage_manifest.csv", "%2E%2E"):
        assert client.get(f"/api/extractions/{a['id']}/frames/{bad}").status_code == 404
    assert client.get(f"/api/extractions/{a['id']}/frames/{xa['row']['img name']}", params={"w": 10_000}).status_code == 422


def test_business_csv_schema_is_unchanged(client, two_jobs):
    (a, _), _ = two_jobs
    job_csv = list(csv.reader(io.StringIO(client.get(f"/api/extractions/{a['id']}/uniform_compliance.csv").text)))
    master = list(csv.reader(io.StringIO(client.get("/api/uniform-compliance/master.csv").text)))
    assert job_csv[0] == HEADER and master[0] == HEADER
    assert all(len(row) == 6 for row in job_csv + master)
    assert client.get("/api/reports/records").json()["fields"] == HEADER
