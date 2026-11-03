"""Frame extraction: exact plan arithmetic, faithful frames, manifest, failures, duplicates."""
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.extraction import extractor
from backend.extraction.extractor import MANIFEST_FIELDS
from backend.extraction.plan import describe, hhmmss, plan

MEDIA = Path(__file__).parent / "media"      # dvr_grid.mp4: 500 frames @ 25 fps = 20 s


# ------------------------------------------------------------------ plan arithmetic
def test_15_minute_video_at_15_minute_interval_is_not_four_images():
    s = plan(900.0, 25.0, 22500, 900.0, include_initial=True)
    assert [(x.sample_type, x.target_seconds) for x in s] == [("initial", 0.0)]      # one image, not four
    assert plan(900.0, 25.0, 22500, 900.0, include_initial=False) == []
    d = describe(900.0, 25.0, 22500, 900.0, True, s)
    assert d["total"] == 1 and d["scheduled"] == 0 and "No scheduled sample" in d["explanation"]


def test_plan_counts():
    assert len(plan(901.0, 25.0, 22525, 900.0, True)) == 2              # 00:00 + 15:00
    s = plan(3600.0, 25.0, 90000, 900.0, True)
    assert [x.target_seconds for x in s] == [0, 900, 1800, 2700]        # 60 min at 15 min: 1 initial + 3 scheduled
    assert len(plan(900.0, 25.0, 22500, 30.0, False)) == 29             # 30 s ... 14:30
    s30 = plan(900.0, 25.0, 22500, 30.0, True)                          # initial frame kept separate
    assert [x.sample_type for x in s30].count("initial") == 1 and len(s30) == 30
    assert [x.target_seconds for x in s30 if x.sample_type == "scheduled"] == [30.0 * k for k in range(1, 30)]
    s = plan(20.0, 25.0, 500, 5.0, True)
    assert [x.target_frame for x in s] == [0, 125, 250, 375]
    assert hhmmss(3725.5) == "01:02:05.500"
    with pytest.raises(ValueError):
        plan(20.0, 25.0, 500, 0, True)


# ------------------------------------------------------------------ helpers
def upload(client, name):
    r = client.post("/api/media", files={"file": (name, (MEDIA / name).read_bytes())})
    assert r.status_code == 200, r.text
    return r.json()


def wait(client, eid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/extractions/{eid}").json()
        if j["state"] not in ("queued", "running"):
            return j
        time.sleep(0.2)
    raise AssertionError("extraction did not finish")


def manifest(client, j):
    text = client.get(j["manifest_url"]).text
    return list(csv.DictReader(text.splitlines()))


# ------------------------------------------------------------------ API end to end
def test_extract_frames_exactly(client):
    m = upload(client, "dvr_grid.mp4")
    plan_r = client.get(f"/api/media/{m['id']}/extraction-plan", params={"interval_seconds": 5, "include_initial": True}).json()
    assert plan_r["total"] == 4 and plan_r["scheduled"] == 3 and plan_r["allowed"]
    r = client.post("/api/extractions", json={"media_id": m["id"], "interval_seconds": 5, "include_initial": True})
    assert r.status_code == 202, r.text
    j = wait(client, r.json()["id"])
    assert j["state"] == "done" and j["planned"] == 4 and j["extracted"] == 4 and j["failed"] == 0
    rows = manifest(client, j)
    assert list(rows[0].keys()) == MANIFEST_FIELDS                       # required fields, then sample_type
    assert [int(x["frame_number"]) for x in rows] == [0, 125, 250, 375]
    assert [float(x["video_timestamp_seconds"]) for x in rows] == [0.0, 5.0, 10.0, 15.0]
    assert [x["video_timestamp_hhmmss"] for x in rows] == ["00:00:00.000", "00:00:05.000", "00:00:10.000", "00:00:15.000"]
    assert {x["extraction_job_id"] for x in rows} == {j["id"]} and {x["source_video_id"] for x in rows} == {m["id"]}
    assert {x["sampling_strategy"] for x in rows} == {"interval"} and {float(x["sampling_interval_seconds"]) for x in rows} == {5.0}
    # nothing invented: no store / camera / recording time was supplied
    assert {x["store_id"] for x in rows} == {x["camera_id"] for x in rows} == {""}
    assert {x["recording_start_time"] for x in rows} == {x["recording_timestamp"] for x in rows} == {""}
    assert [x["sample_type"] for x in rows] == ["initial", "scheduled", "scheduled", "scheduled"]
    assert {x["extraction_status"] for x in rows} == {"ok"}
    assert len({x["image_filename"] for x in rows}) == 4 and len({x["image_id"] for x in rows}) == 4
    assert all(x["source_video"] == "dvr_grid.mp4" and m["id"][:8] in x["image_filename"] for x in rows)
    assert all(int(x["image_width"]) == 1280 and int(x["image_height"]) == 960 for x in rows)
    # every saved image is pixel-identical to the decoder's frame at that frame number
    cap = cv2.VideoCapture(str(MEDIA / "dvr_grid.mp4"))
    decoded, idx = {}, 0
    wanted = {int(x["frame_number"]) for x in rows}
    while idx <= max(wanted):
        ok, f = cap.read()
        if idx in wanted:
            decoded[idx] = f
        idx += 1
    cap.release()
    for x in rows:
        img = cv2.imdecode(np.frombuffer(client.get(f"/api/extractions/{j['id']}/frames/{x['image_filename']}").content,
                                         np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(img, decoded[int(x["frame_number"])])
    assert client.get(j["failures_url"]).text.strip().startswith("image_id,")   # header only
    frames = client.get(f"/api/extractions/{j['id']}/frames").json()
    assert frames["total"] == 4
    # the file on disk is image_manifest.csv; the older /manifest.csv URL serves the same file
    out = Path(client.app.state.app_state.db.extraction(j["id"])["output_dir"])
    out = out if out.is_absolute() else extractor.ROOT / out
    assert j["manifest_url"].endswith("/image_manifest.csv")
    assert (out / "image_manifest.csv").exists() and not (out / "manifest.csv").exists()
    assert client.get(j["manifest_url"].replace("image_manifest.csv", "manifest.csv")).text == client.get(j["manifest_url"]).text


def test_legacy_manifest_name_still_read(client, tmp_path, monkeypatch):
    from backend.extraction import runner
    monkeypatch.setattr(runner, "EXTRACTIONS", tmp_path)
    (tmp_path / "old-job").mkdir()
    old_header = ("image_id,source_video,image_filename,image_path,timestamp_seconds,timestamp_hhmmss,frame_number,"
                  "image_width,image_height,extraction_status,sample_type")
    (tmp_path / "old-job" / "manifest.csv").write_text(old_header + "\n" + "a,v.mp4,x.png,p/x.png,5,00:00:05.000,125,1,1,ok,scheduled\n",
                                                       encoding="utf-8")
    job = {"id": "old-job", "source_video": "v.mp4", "media_id": "m1", "interval_seconds": 5.0}
    assert runner.manifest_path(job).name == "manifest.csv"
    row = runner.read_manifest(job)[0]                          # older columns are converted to the current schema
    assert row["image_id"] == "a" and row["original_image_path"] == "p/x.png" and row["video_timestamp_seconds"] == "5"
    assert row["extraction_job_id"] == "old-job" and row["sampling_strategy"] == "interval" and row["camera_id"] == ""
    assert runner.manifest_path({"id": "new-job"}).name == "image_manifest.csv"


def test_write_failure_reason_names_long_paths(monkeypatch):
    monkeypatch.setattr(extractor.os, "name", "nt")
    long = Path("C:/" + "d" * 300 + "/frame.png")
    assert "over the Windows limit of 260" in extractor.write_failure_reason(long)
    assert "disk full" in extractor.write_failure_reason(Path("C:/x/frame.png"))


def test_duplicates_and_no_overwrite(client):
    m = upload(client, "dvr_grid.mp4")
    body = {"media_id": m["id"], "interval_seconds": 10, "include_initial": False}
    j1 = wait(client, client.post("/api/extractions", json=body).json()["id"])
    again = client.post("/api/extractions", json=body).json()
    assert again["reused"] is True and again["id"] == j1["id"]                   # identical finished job reused
    j2 = wait(client, client.post("/api/extractions", json=body | {"new_run": True}).json()["id"])
    assert j2["id"] != j1["id"] and j2["output_dir"] != j1["output_dir"]       # new folder, first one untouched
    assert len(manifest(client, j1)) == len(manifest(client, j2)) == 1         # 10 s only (20 s is after the last frame)
    # an active job for the same video blocks a second one
    db = client.app.state.app_state.db
    db.add_extraction({"id": "ext-busy", "created_at": "2099-01-01T00:00:00+00:00", "media_id": m["id"],
                       "source_video": "dvr_grid.mp4", "state": "running", "interval_seconds": 3, "include_initial": 1,
                       "image_format": "png", "planned": 7})
    try:
        r = client.post("/api/extractions", json=body | {"interval_seconds": 3})
        assert r.status_code == 409 and r.json()["detail"]["job_id"] == "ext-busy"
    finally:
        db.update_extraction("ext-busy", state="done")


def test_rejections(client):
    img = upload(client, "store_uniform.jpg")
    assert client.post("/api/extractions", json={"media_id": img["id"], "interval_seconds": 30}).status_code == 422
    v = upload(client, "dvr_grid.mp4")
    r = client.post("/api/extractions", json={"media_id": v["id"], "interval_seconds": 900, "include_initial": False})
    assert r.status_code == 422 and "no frames" in r.json()["detail"]        # 20 s video, 15 min interval
    assert client.post("/api/extractions", json={"media_id": v["id"], "interval_seconds": 0}).status_code == 422
    client.put("/api/settings", json={"extraction_max_frames": 2})
    try:
        assert client.post("/api/extractions", json={"media_id": v["id"], "interval_seconds": 1}).status_code == 422
    finally:
        client.post("/api/settings/reset")
    assert client.get("/api/extractions/nope").status_code == 404
    assert client.post("/api/media", files={"file": ("x.mp4", b"not a video")}).status_code == 422


def test_failed_frames_are_recorded_not_faked(client, monkeypatch):
    m = upload(client, "dvr_grid.mp4")
    real = cv2.VideoCapture

    class Flaky:                                   # the decoder gives no image for frame 250 (10 s)
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
    monkeypatch.setattr(extractor.cv2, "VideoCapture", Flaky)
    j = wait(client, client.post("/api/extractions", json={"media_id": m["id"], "interval_seconds": 5,
                                                          "include_initial": True, "new_run": True}).json()["id"])
    assert j["state"] == "done" and j["extracted"] == 3 and j["failed"] == 1
    rows = manifest(client, j)
    assert [int(x["frame_number"]) for x in rows] == [0, 125, 375]               # no fake record for 10 s
    fails = list(csv.DictReader(client.get(j["failures_url"]).text.splitlines()))
    assert len(fails) == 1 and fails[0]["target_hhmmss"] == "00:00:10.000" and fails[0]["target_frame"] == "250"
    assert fails[0]["stage"] == "extraction" and fails[0]["error_category"] == "no_frame" and fails[0]["error_message"]
    assert j["failures"][0]["error_message"]
    assert "Only 3 of 4 screenshots could be extracted" in j["error"]           # shortfall is stated, not silent


def test_cancel_and_restart(client, monkeypatch):
    s = client.app.state.app_state
    m = upload(client, "dvr_grid.mp4")
    monkeypatch.setattr(s.extractions, "_run", lambda *a: time.sleep(1.0))      # hold the worker busy
    r1 = client.post("/api/extractions", json={"media_id": m["id"], "interval_seconds": 2, "include_initial": True,
                                               "new_run": True}).json()
    s.extractions.cancel(r1["id"])
    assert client.post(f"/api/extractions/{r1['id']}/cancel").status_code == 200
    time.sleep(1.5)
    monkeypatch.undo()
    # a job that was "running" when the server stopped is reported as interrupted, not done
    s.db.add_extraction({"id": "ext-stuck", "created_at": "2099-01-01T00:00:00+00:00", "media_id": "x",
                         "source_video": "v.mp4", "state": "running", "interval_seconds": 30, "include_initial": 1,
                         "image_format": "png", "planned": 3})
    s.db.interrupt_extractions()
    assert s.db.extraction("ext-stuck")["state"] == "interrupted"
    s.db.update_extraction(r1["id"], state="cancelled")


def test_cancel_mid_run_keeps_partial_manifest(client, monkeypatch):
    import backend.extraction.runner as runner
    m = upload(client, "dvr_grid.mp4")
    orig = runner.extract
    calls = {"n": 0}

    def stop_after_two(*a, **k):
        def should_stop():                                 # the cancel arrives just before the 3rd sample
            calls["n"] += 1
            return calls["n"] >= 3
        k["should_stop"] = should_stop
        return orig(*a, **k)
    monkeypatch.setattr(runner, "extract", stop_after_two)
    r = client.post("/api/extractions", json={"media_id": m["id"], "interval_seconds": 2, "include_initial": True,
                                              "new_run": True}).json()
    j = wait(client, r["id"])
    assert j["state"] == "cancelled" and j["planned"] == 10 and j["extracted"] == 2
    rows = manifest(client, j)                              # exactly the two frames that were saved, nothing else
    assert [int(x["frame_number"]) for x in rows] == [0, 50]
