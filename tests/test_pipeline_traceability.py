"""End-to-end traceability: sampling strategies, store / camera / recording-time metadata, hourly
summary, processing failures and the CCTV adapter interface."""
import csv
import time

import cv2
import pytest

from backend.capture import cctv_adapter
from backend.extraction import extractor
from backend.extraction.plan import plan_random
from tests.test_extraction import MEDIA, upload, wait
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)


def csv_rows(client, url, **params):
    return list(csv.DictReader(client.get(url, params=params).text.splitlines()))


def wait_run(client, eid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        runs = client.get(f"/api/extractions/{eid}/detections").json()["runs"]
        if runs and runs[0]["state"] not in ("queued", "running"):
            return runs[0]
        time.sleep(0.25)
    raise AssertionError("detection did not finish")


# ------------------------------------------------------------------ random sampling
def test_random_plan_is_reproducible():
    a = plan_random(20.0, 25.0, 500, 5, 42, include_initial=True)
    assert a == plan_random(20.0, 25.0, 500, 5, 42, include_initial=True)         # same seed -> same frames
    assert [s.target_frame for s in a[1:]] != [s.target_frame for s in plan_random(20.0, 25.0, 500, 5, 43, True)[1:]]
    assert a[0].sample_type == "initial" and {s.sample_type for s in a[1:]} == {"random"}
    frames = [s.target_frame for s in a[1:]]
    assert frames == sorted(frames) and len(set(frames)) == 5 and 0 not in frames and max(frames) <= 499
    with pytest.raises(ValueError):
        plan_random(20.0, 25.0, 500, 500, 1, False)                              # only 499 frames after frame 0


def test_random_extraction_records_seed_and_reuses_only_with_same_seed(client):
    m = upload(client, "dvr_grid.mp4")
    body = {"media_id": m["id"], "sampling_strategy": "random", "sample_count": 3, "random_seed": 7,
            "include_initial": True, "new_run": True}
    j = wait(client, client.post("/api/extractions", json=body).json()["id"])
    assert j["state"] == "done" and j["sampling_strategy"] == "random" and j["random_seed"] == 7
    assert j["sample_count"] == 3 and j["interval_seconds"] is None and j["extracted"] == 4
    rows = csv_rows(client, j["manifest_url"])
    expected = [s.target_frame for s in plan_random(20.0, 25.0, 500, 3, 7, True)]
    assert [int(r["frame_number"]) for r in rows] == expected
    assert [r["sample_type"] for r in rows] == ["initial", "random", "random", "random"]
    assert {r["sampling_strategy"] for r in rows} == {"random"} and {r["random_seed"] for r in rows} == {"7"}
    assert {r["sampling_interval_seconds"] for r in rows} == {""}
    again = client.post("/api/extractions", json=body | {"new_run": False}).json()
    assert again["reused"] is True and again["id"] == j["id"]                     # same seed + settings -> reused
    # without a seed a new one is generated, recorded, and reproduces that job's frames
    j2 = wait(client, client.post("/api/extractions", json={k: v for k, v in body.items() if k != "random_seed"}).json()["id"])
    assert j2["id"] != j["id"] and isinstance(j2["random_seed"], int)
    seed_frames = [s.target_frame for s in plan_random(20.0, 25.0, 500, 3, j2["random_seed"], True)]
    assert [int(r["frame_number"]) for r in csv_rows(client, j2["manifest_url"])] == seed_frames
    # bad requests
    assert client.post("/api/extractions", json=body | {"sample_count": 0}).status_code == 422
    assert client.post("/api/extractions", json={"media_id": m["id"], "sampling_strategy": "interval"}).status_code == 422  # interval missing


# ------------------------------------------------------------------ metadata, clock time, hourly, filters
def test_store_camera_and_recording_time_flow_through(client, real_model):  # noqa: F811
    m = upload(client, "dvr_grid.mp4")
    meta = {"store_id": "STORE-TEST", "camera_id": "CAM-TEST", "recording_start_time": "2026-10-25T13:59:50"}
    j = wait(client, client.post("/api/extractions", json={"media_id": m["id"], "interval_seconds": 5, "include_initial": True,
                                                          "run_detection": True, "new_run": True, **meta}).json()["id"])
    run = wait_run(client, j["id"])
    assert j["store_id"] == "STORE-TEST" and j["camera_id"] == "CAM-TEST" and run["state"] == "done"
    man = csv_rows(client, j["manifest_url"])
    assert {(r["store_id"], r["camera_id"]) for r in man} == {("STORE-TEST", "CAM-TEST")}
    assert [r["recording_timestamp"] for r in man] == ["2026-10-25T13:59:50.000", "2026-10-25T13:59:55.000",
                                                       "2026-10-25T14:00:00.000", "2026-10-25T14:00:05.000"]
    assert [r["video_timestamp_hhmmss"] for r in man] == ["00:00:00.000", "00:00:05.000", "00:00:10.000", "00:00:15.000"]
    res = csv_rows(client, run["results_csv_url"])
    assert {(r["store_id"], r["camera_id"]) for r in res} == {("STORE-TEST", "CAM-TEST")}
    assert {r["image_id"] for r in res} == {r["image_id"] for r in man}         # every image analysed, linked

    # hourly summary: real clock hours, because the recording start time is known
    hours = [h for h in csv_rows(client, "/api/frame-results/hourly_summary.csv", camera_id="CAM-TEST")
             if h["extraction_job_id"] == j["id"]]
    assert [(h["hour_basis"], h["hour_start"], h["expected_samples"], h["successful_samples"], h["analysed_images"])
            for h in hours] == [("recording_time", "2026-10-25T13:00:00", "2", "2", "2"),
                                ("recording_time", "2026-10-25T14:00:00", "2", "2", "2")]
    assert {h["summary_status"] for h in hours} == {"complete"}

    # dashboard filters by camera and recording date use the stored records
    d = client.get("/api/frame-results", params={"camera_id": "CAM-TEST"}).json()
    assert "CAM-TEST" in d["cameras"] and "2026-10-25" in d["recording_dates"]
    assert {v["camera_id"] for v in d["videos"]} == {"CAM-TEST"} and d["summary"]["frames_extracted"] == sum(
        v["extracted"] for v in d["videos"])
    mine = [i for i in d["images"] if i["extraction_id"] == j["id"]]
    assert len(mine) == 4 and mine[0]["recording_timestamp"] == "2026-10-25T13:59:50.000"
    assert client.get("/api/frame-results", params={"camera_id": "CAM-TEST", "date": "2026-10-24"}).json()["images_total"] == 0
    assert client.get("/api/frame-results", params={"date": "25-10-2026"}).status_code == 422

    # invalid metadata is refused, never silently changed
    bad = {"media_id": m["id"], "interval_seconds": 5, "new_run": True}
    assert client.post("/api/extractions", json=bad | {"camera_id": "CAM 01; drop"}).status_code == 422
    assert client.post("/api/extractions", json=bad | {"recording_start_time": "yesterday 2pm"}).status_code == 422


# ------------------------------------------------------------------ processing failures
def test_processing_failures_with_retry_status(client, monkeypatch):
    m = upload(client, "dvr_grid.mp4")
    real = cv2.VideoCapture

    class Flaky:                                          # the decoder gives no image for frame 250 (10 s)
        def __init__(self, *a):
            self.c, self.pos = real(*a), -1

        def isOpened(self):
            return self.c.isOpened()

        def grab(self):
            self.pos += 1
            return self.c.grab()

        def retrieve(self):
            return (False, None) if self.pos == 250 else self.c.retrieve()

        def get(self, prop):
            return self.c.get(prop)

        def release(self):
            self.c.release()
    body = {"media_id": m["id"], "interval_seconds": 5, "include_initial": True, "new_run": True,
            "camera_id": "CAM-FAIL", "run_detection": True}
    monkeypatch.setattr(extractor.cv2, "VideoCapture", Flaky)
    j1 = wait(client, client.post("/api/extractions", json=body).json()["id"])
    monkeypatch.undo()
    wait_run(client, j1["id"])                            # suite pins a missing model: the run is refused
    pf = csv_rows(client, "/api/frame-results/processing_failures.csv", camera_id="CAM-FAIL")
    ext_fail = [r for r in pf if r["stage"] == "extraction"]
    assert len(ext_fail) == 1 and ext_fail[0]["video_timestamp_hhmmss"] == "00:00:10.000"
    assert ext_fail[0]["error_category"] == "no_frame" and ext_fail[0]["retry_status"] == "not_retried"
    refused = [r for r in pf if r["stage"] == "detection_run"]
    assert len(refused) == 1 and "weights not found" in refused[0]["error_message"]
    # a later job of the same video extracts that frame -> the failure is marked as retried
    j2 = wait(client, client.post("/api/extractions", json=body | {"run_detection": False}).json()["id"])
    assert j2["failed"] == 0
    pf = csv_rows(client, "/api/frame-results/processing_failures.csv", camera_id="CAM-FAIL")
    assert [r["retry_status"] for r in pf if r["stage"] == "extraction"] == ["retried_ok"]
    hours = csv_rows(client, "/api/frame-results/hourly_summary.csv", camera_id="CAM-FAIL")
    h1 = next(h for h in hours if h["extraction_job_id"] == j1["id"])
    assert (h1["expected_samples"], h1["successful_samples"], h1["failed_samples"], h1["summary_status"]) == \
        ("4", "3", "1", "not_analysed")
    d = client.get("/api/frame-results", params={"camera_id": "CAM-FAIL"}).json()
    assert d["failures_total"] == len(pf) and d["summary"]["processing_failures"] == len(pf)


# ------------------------------------------------------------------ CCTV adapter interface
def test_cctv_adapter_is_honest_and_feeds_the_same_pipeline(client):
    st = cctv_adapter.status()
    assert st["state"] == "not_configured" and st["adapter"] is None and len(st["requirements"]) >= 4
    assert client.get("/api/health").json()["cctv_adapter"]["state"] == "not_configured"
    with pytest.raises(cctv_adapter.CctvNotConfigured):
        cctv_adapter.get_adapter()
    s = client.app.state.app_state
    clip = cctv_adapter.VideoClip(path=MEDIA / "dvr_grid.mp4", store_id="STORE-AD", camera_id="CAM-AD",
                                  recording_start_time="2026-10-25T09:00:00", source_ref="local test file")
    job, reused = cctv_adapter.ingest_clip(s.db, s.extractions, clip, interval_seconds=10, include_initial=True)
    j = wait(client, job["id"])
    assert not reused and j["state"] == "done" and j["source"] == "cctv_adapter" and j["extracted"] == 2
    man = csv_rows(client, j["manifest_url"])
    assert {(r["store_id"], r["camera_id"]) for r in man} == {("STORE-AD", "CAM-AD")}
    assert [r["recording_timestamp"] for r in man] == ["2026-10-25T09:00:00.000", "2026-10-25T09:00:10.000"]
    with pytest.raises(ValueError):
        cctv_adapter.ingest_clip(s.db, s.extractions, cctv_adapter.VideoClip(path=MEDIA / "store_uniform.jpg"))
