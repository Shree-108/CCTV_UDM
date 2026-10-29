"""Regression: an extracted screenshot reaches the uniform model and its ACTUAL prediction is saved.

The bug: the job page stopped refreshing as soon as extraction finished - before the detection run
existed / had processed anything - so it stayed at "Not analysed yet ... no detection run" with blank
scores although the model ran a moment later. The API now reports the detection state (and whether
detection was requested), the UI keeps polling while it is pending, and screenshots without a
prediction say WHY (pending / detection failed / not requested) instead of all reading "not analysed"."""
import csv
import io
import threading
import time

from backend.extraction import detect
from backend.extraction.screenshots import ANALYSIS_PENDING, DETECTION_FAILED, remark
from tests.test_compliance_reports import Person, csv_rows
from tests.test_extraction import MEDIA
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)


def start_job(client, body):
    m = client.post("/api/media", files={"file": ("dvr_grid.mp4", (MEDIA / "dvr_grid.mp4").read_bytes())}).json()
    return client.post("/api/extractions", json={"media_id": m["id"], "run_detection": True, "new_run": True, **body}).json()


def wait_for(fn, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError("condition not reached")


def test_screenshots_reach_the_model_and_predictions_are_saved(client, real_model, monkeypatch):  # noqa: F811
    gate, seen = threading.Event(), []

    def model(img, models, cfg, thr):            # the inference call for one screenshot
        seen.append(img.shape)
        gate.wait(30)                            # hold the run so the "in between" state can be observed
        if len(seen) == 2:
            return [Person((40, 40, 140, 400), "uniform", 0.93), Person((300, 50, 400, 420), "non_uniform", 0.12)]
        return [Person((100, 80, 220, 460), "uniform", 0.88)] if len(seen) != 3 else []
    monkeypatch.setattr(detect, "classify_frame", model)
    j = start_job(client, {"random_seed": 201, "store_id": "store id 7", "camera_id": "CAM-INF"})
    url = f"/api/extractions/{j['id']}/uniform-compliance"

    # extraction done, detection running but nothing processed yet: NOT "not analysed", and polling must go on
    c = wait_for(lambda: (x := client.get(url).json())["extraction_state"] == "done" and x["detection_state"] == "running" and x)
    assert c["run_id"] is None and not any(r["row"]["confidence score"] for r in c["records"])
    assert all(r["processing_status"] == ANALYSIS_PENDING for r in c["records"])
    assert all(r["row"]["remark (Example)"].startswith("Analysis pending") for r in c["records"])
    csv_now = list(csv.reader(io.StringIO(client.get(f"/api/extractions/{j['id']}/uniform_compliance.csv").text)))
    assert all(r[5].startswith("Analysis pending") and r[4] == "" for r in csv_now[1:])          # never a fake 0 %

    gate.set()
    c = wait_for(lambda: (x := client.get(url).json())["run_state"] == "done" and x)
    assert len(seen) == 4                                                    # each saved screenshot inferred ONCE
    run = client.get(f"/api/detection-runs/{c['run_id']}").json()
    assert run["processed"] == 4 and run["succeeded"] == 4 and run["model_version"]
    rows = [r["row"] for r in c["records"]]
    assert [r["confidence score"] for r in rows] == ["88.0%", "Person 1: 93.0%; Person 2: 12.0%", "", "88.0%"]
    assert rows[1]["remark (Example)"] == ("Person 1 detected wearing the designated uniform shirt; "
                                           "Person 2 detected without the designated uniform shirt.")
    assert rows[2]["remark (Example)"] == "No person detected in this screenshot."
    man = csv_rows(client, j["manifest_url"])
    det = csv_rows(client, run["results_csv_url"])
    for r, m in zip(rows, man):                                              # prediction <-> screenshot <-> time
        assert r["img name"] == m["image_filename"] and r["Time Stamp"] == m["video_timestamp_hhmmss"][:8]
    assert {d["detection_run_id"] for d in det} == {run["id"]} and {d["model_version"] for d in det} == {run["model_version"]}
    assert {d["extraction_job_id"] for d in det} == {j["id"]}
    # the job's CSV, the master CSV and the dashboard show the same real values
    saved = list(csv.reader(io.StringIO(client.get(f"/api/extractions/{j['id']}/uniform_compliance.csv").text)))[1:]
    assert saved == [list(r.values()) for r in rows]
    master = list(csv.reader(io.StringIO(client.get("/api/uniform-compliance/master.csv").text)))
    assert all(r in master for r in saved)
    frames = client.get("/api/frame-results", params={"camera_id": "CAM-INF"}).json()["images"]
    assert sorted(f["prediction"] for f in frames) == ["NO_UNIFORM_DETECTED", "UNIFORM_DETECTED", "UNIFORM_DETECTED", "UNIFORM_DETECTED"]


def test_detection_failure_is_not_reported_as_not_analysed_or_non_compliant(client):
    """Suite default: trained weights missing -> the automatic run is refused and recorded as failed."""
    j = start_job(client, {"random_seed": 202, "camera_id": "CAM-INF2"})
    url = f"/api/extractions/{j['id']}/uniform-compliance"
    c = wait_for(lambda: (x := client.get(url).json())["detection_state"] == "failed" and x)
    assert "weights not found" in c["detection_error"] and c["run_id"] is None and not c["detection_pending"]
    for r in c["records"]:
        assert r["processing_status"] == DETECTION_FAILED and r["row"]["confidence score"] == ""
        assert "detection run failed" in r["row"]["remark (Example)"] and "without" not in r["row"]["remark (Example)"]


def test_remark_wording_for_unanalysed_screenshots():
    assert remark([], "", ANALYSIS_PENDING).startswith("Analysis pending")
    assert "detection run failed" in remark([], "", DETECTION_FAILED)
