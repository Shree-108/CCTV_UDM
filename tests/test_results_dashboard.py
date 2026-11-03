"""Results dashboard API: complete workflow upload -> extraction -> detection -> summary -> CSV exports."""
import csv
import time

import numpy as np

from backend.extraction import results as R
from backend.extraction.extractor import MANIFEST_FIELDS
from backend.extraction.detect import RESULT_FIELDS
from tests.test_frame_detection import empty_scene, make_video, real_model, wait  # noqa: F401  (fixture re-used)


def _workflow(client, tmp_path, name, n_seconds=3):
    frames = [empty_scene(70 + 40 * i) for i in range(n_seconds)]
    video = make_video(tmp_path / name, frames)
    m = client.post("/api/media", files={"file": (name, video.read_bytes())}).json()
    ext = client.post("/api/extractions", json={"media_id": m["id"], "interval_seconds": 1, "include_initial": True,
                                                "run_detection": True, "new_run": True}).json()
    ext = wait(client, f"/api/extractions/{ext['id']}")
    for _ in range(200):                                       # detection starts after extraction
        runs = client.get(f"/api/extractions/{ext['id']}/detections").json()["runs"]
        if runs and runs[0]["state"] not in ("queued", "running"):
            return ext, runs[0]
        time.sleep(0.25)
    raise AssertionError("detection did not finish")


def test_complete_workflow_to_csv_export(client, real_model, tmp_path):
    ext, run = _workflow(client, tmp_path, "dash_workflow.mp4")
    assert ext["state"] == "done" and ext["extracted"] == 3 and run["state"] == "done" and run["processed"] == 3

    d = client.get("/api/frame-results", params={"source_video": "dash_workflow.mp4"}).json()
    s = d["summary"]
    assert s["frames_extracted"] == 3 and s["frames_processed"] == 3 and s["frames_not_processed"] == 0
    assert s["uniform_images"] + s["no_uniform_images"] + s["review_images"] == 3
    assert s["no_uniform_images"] == 3 and s["detections"] == 0               # empty scene: no shirt found
    assert d["images_total"] == 3 and {i["prediction"] for i in d["images"]} == {"NO_UNIFORM_DETECTED"}
    assert [i["timestamp"] for i in d["images"]] == ["00:00:00.000", "00:00:01.000", "00:00:02.000"]
    im = d["images"][0]
    assert client.get(im["original_url"]).status_code == 200 and client.get(im["annotated_url"]).status_code == 200
    assert "dash_workflow.mp4" in d["source_videos"]
    v = next(x for x in d["videos"] if x["extraction_id"] == ext["id"])
    assert v["processed"] == 3 and v["run_state"] == "done" and v["duration_seconds"] > 2.9

    # filters: result + processing status narrow the table, not the totals
    f = client.get("/api/frame-results", params={"source_video": "dash_workflow.mp4", "outcome": "UNIFORM_DETECTED"}).json()
    assert f["images_total"] == 0 and f["summary"]["frames_processed"] == 3
    f = client.get("/api/frame-results", params={"source_video": "dash_workflow.mp4", "processing_status": "ok"}).json()
    assert f["images_total"] == 3
    assert client.get("/api/frame-results", params={"outcome": "BOGUS"}).status_code == 422

    # CSV exports
    man = list(csv.DictReader(client.get("/api/frame-results/image_manifest.csv", params={"source_video": "dash_workflow.mp4"}).text.splitlines()))
    assert len(man) == 3 and list(man[0].keys()) == MANIFEST_FIELDS
    res_r = client.get("/api/frame-results/detection_results.csv", params={"source_video": "dash_workflow.mp4"})
    assert res_r.headers["content-disposition"].startswith("attachment")
    res = list(csv.DictReader(res_r.text.splitlines()))
    assert len(res) == 3 and list(res[0].keys()) == RESULT_FIELDS
    assert {r["detection_count"] for r in res} == {"0"} and {r["confidence"] for r in res} == {""}
    assert {r["detection_run_id"] for r in res} == {run["id"]} and {r["extraction_job_id"] for r in res} == {ext["id"]}
    # every result joins to exactly one manifest row on image_id, with the same path and timestamp
    by_id = {m["image_id"]: m for m in man}
    for r in res:
        assert r["original_image_path"] == by_id[r["image_id"]]["original_image_path"]
        assert r["video_timestamp_seconds"] == by_id[r["image_id"]]["video_timestamp_seconds"]
    # CSV counts == database counts of the run
    assert sum(r["prediction"] == "NO_UNIFORM_DETECTED" for r in res) == run["no_uniform_images"] == 3

    # short video, no recording start time: one hourly row, in video time
    assert d["hourly"]["available"] and "video time" in d["hourly"]["basis"]
    (h,) = [r for r in d["hourly"]["rows"] if r["extraction_job_id"] == ext["id"]]
    assert (h["hour_basis"], h["hour_start"], h["expected_samples"], h["successful_samples"], h["analysed_images"],
            h["summary_status"]) == ("video_relative", "video 00:00:00", 3, 3, 3, "complete")


def test_hourly_buckets():
    from datetime import datetime
    assert R._bucket(None, 3599.9) == ("video 00:00:00", "video 01:00:00")
    assert R._bucket(None, 3600) == ("video 01:00:00", "video 02:00:00")
    start = datetime.fromisoformat("2026-10-25T13:45:00")
    assert R._bucket(start, 0) == ("2026-10-25T13:00:00", "2026-10-25T14:00:00")
    assert R._bucket(start, 15 * 60) == ("2026-10-25T14:00:00", "2026-10-25T15:00:00")
    assert R.hourly([])["available"] is False


def test_storage_status_is_honest(client, monkeypatch):
    st = client.get("/api/health").json()["storage"]
    assert st["mode"] == "local" and st["s3"]["configured"] is False and st["s3"]["uploads_enabled"] is False
    monkeypatch.setenv("SU_S3_BUCKET", "example-bucket")
    s3 = client.get("/api/frame-results").json()["storage"]["s3"]
    assert s3["configured"] is True and s3["uploads_enabled"] is False and "not implemented" in s3["note"]


def test_model_missing_is_visible_on_dashboard(client, tmp_path):
    ext, run = _workflow(client, tmp_path, "dash_nomodel.mp4", n_seconds=2)      # suite pins a missing model
    assert run["state"] == "failed" and "weights not found" in run["error"]
    d = client.get("/api/frame-results", params={"source_video": "dash_nomodel.mp4"}).json()
    v = d["videos"][0]
    assert v["extracted"] == 2 and v["processed"] == 0 and "weights not found" in v["last_run_error"]
    assert d["summary"]["frames_extracted"] == 2 and d["summary"]["frames_processed"] == 0
    assert d["summary"]["frames_not_processed"] == 2
