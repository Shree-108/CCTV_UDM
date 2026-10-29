"""Uniform detection on extracted frames: real weights required, outcome rules, one row per
detection, failures isolated, originals unchanged."""
import csv
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import backend.analysis.pipeline as pipeline
import backend.core.model_config as mc
from backend.extraction import detect
from backend.extraction.detect import RESULT_FIELDS

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "tests" / "media"
REAL_WEIGHTS = ROOT / "models" / "uniform_best.pt"          # the committed trained model (v0.2.0)
REFERENCE = ROOT / "data" / "reference" / "uniform_reference_02.png"   # private staff photo, git-ignored


def make_video(path: Path, images: list[np.ndarray], seconds_each: float = 1.0, fps: int = 10) -> Path:
    """Short MP4 whose content changes every `seconds_each` - each image letterboxed onto 640x480."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    for img in images:
        h, wd = img.shape[:2]
        k = min(640 / wd, 480 / h)
        canvas = np.full((480, 640, 3), 60, np.uint8)
        small = cv2.resize(img, (int(wd * k), int(h * k)))
        y, x = (480 - small.shape[0]) // 2, (640 - small.shape[1]) // 2
        canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
        for _ in range(int(seconds_each * fps)):
            w.write(canvas)
    w.release()
    return path


def wait(client, url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(url).json()
        if j["state"] not in ("queued", "running"):
            return j
        time.sleep(0.25)
    raise AssertionError(f"{url} did not finish")


def extract(client, video: Path, interval=1, run_detection=False):
    r = client.post("/api/media", files={"file": (video.name, video.read_bytes())})
    assert r.status_code == 200, r.text
    mid = r.json()["id"]
    j = client.post("/api/extractions", json={"media_id": mid, "interval_seconds": interval, "include_initial": True,
                                              "run_detection": run_detection, "new_run": True}).json()
    return wait(client, f"/api/extractions/{j['id']}")


def run_detection(client, ext, **body):
    r = client.post(f"/api/extractions/{ext['id']}/detect", json=body)
    assert r.status_code == 202, r.text
    return wait(client, f"/api/detection-runs/{r.json()['id']}")


def rows(client, run):
    return list(csv.DictReader(client.get(run["results_csv_url"]).text.splitlines()))


@pytest.fixture()
def real_model(client, monkeypatch):
    """Use the committed trained weights (the test suite otherwise pins a missing model)."""
    models = client.app.state.app_state.models
    monkeypatch.setattr(pipeline, "uniform_model_path", lambda: REAL_WEIGHTS)
    monkeypatch.setattr(mc, "uniform_model_path", lambda: REAL_WEIGHTS)
    models._key = None
    yield models
    models._key = None


# ------------------------------------------------------------------ weights required
def test_refuses_without_trained_weights(client, tmp_path):
    ext = extract(client, make_video(tmp_path / "blank.mp4", [np.full((480, 640, 3), 128, np.uint8)] * 2))
    r = client.post(f"/api/extractions/{ext['id']}/detect", json={})
    assert r.status_code == 422 and "weights not found" in r.json()["detail"]   # no silent substitute
    assert client.get(f"/api/extractions/{ext['id']}/detections").json()["runs"] == []
    # automatic detection after extraction records the problem visibly
    ext2 = extract(client, make_video(tmp_path / "blank2.mp4", [np.full((480, 640, 3), 90, np.uint8)] * 2),
                   run_detection=True)
    time.sleep(0.5)
    runs = client.get(f"/api/extractions/{ext2['id']}/detections").json()["runs"]
    assert len(runs) == 1 and runs[0]["state"] == "failed" and "weights not found" in runs[0]["error"]


def test_refuses_unreadable_weights(client, tmp_path, monkeypatch):
    """A corrupt weights file is refused like a missing one - never replaced by another model."""
    bad = tmp_path / "uniform_best.pt"
    bad.write_bytes(b"not a model")
    models = client.app.state.app_state.models
    monkeypatch.setattr(pipeline, "uniform_model_path", lambda: bad)
    monkeypatch.setattr(mc, "uniform_model_path", lambda: bad)
    models._key = None
    try:
        ext = extract(client, make_video(tmp_path / "blank3.mp4", [np.full((480, 640, 3), 70, np.uint8)] * 2))
        r = client.post(f"/api/extractions/{ext['id']}/detect", json={})
        assert r.status_code == 422, r.text
        assert client.get(f"/api/extractions/{ext['id']}/detections").json()["runs"] == []
        print(f"\n  corrupt weights refused with: {r.json()['detail']}")
    finally:
        models._key = None


def test_older_result_files_are_normalized():
    old = {"image_id": "a", "capture_timestamp": "00:01:02.500", "processing_status": "inference_failed",
           "prediction": "REVIEW_REQUIRED", "confidence": ""}
    run = {"id": "det-1", "extraction_id": "ext-1", "threshold": 0.7}
    row = detect.normalize(dict(old), run, {"store_id": "STORE-9", "camera_id": ""})
    assert row["video_timestamp_hhmmss"] == "00:01:02.500" and float(row["video_timestamp_seconds"]) == 62.5
    assert row["processing_status"] == "failed" and row["prediction"] == "PROCESSING_FAILED"
    assert row["extraction_job_id"] == "ext-1" and row["detection_run_id"] == "det-1" and row["confidence_threshold"] == 0.7
    assert row["store_id"] == "STORE-9" and row["camera_id"] == "" and "capture_timestamp" not in row


# ------------------------------------------------------------------ real model
def test_no_detection_case(client, real_model, tmp_path):
    """Frames without people: NO_UNIFORM_DETECTED, one row each, detection fields blank."""
    blank = np.full((480, 640, 3), 128, np.uint8)
    ext = extract(client, make_video(tmp_path / "empty_scene.mp4", [blank, blank]))
    run = run_detection(client, ext)
    assert run["state"] == "done" and run["total"] == 2 and run["succeeded"] == 2 and run["failed"] == 0
    assert run["no_uniform_images"] == 2 and run["detections"] == 0
    assert "yatrikart-uniform-cls" in run["model_version"]
    rs = rows(client, run)
    assert list(rs[0].keys()) == RESULT_FIELDS
    assert len(rs) == 2 and {r["prediction"] for r in rs} == {"NO_UNIFORM_DETECTED"}
    for r in rs:
        assert r["detection_count"] == "0" and r["confidence"] == "" and r["bbox_xmin"] == ""
        assert r["processing_status"] == "ok" and r["review_status"] == "unreviewed" and r["annotated_image_path"]
    assert [r["video_timestamp_hhmmss"] for r in rs] == ["00:00:00.000", "00:00:01.000"]
    assert [float(r["video_timestamp_seconds"]) for r in rs] == [0.0, 1.0]
    assert {r["extraction_job_id"] for r in rs} == {ext["id"]} and {r["detection_run_id"] for r in rs} == {run["id"]}
    assert {float(r["confidence_threshold"]) for r in rs} == {run["threshold"]}
    assert {r["person_count"] for r in rs} == {"0"} and {r["detection_type"] for r in rs} == {""}
    manifest = {f["image_id"]: f for f in client.get(f"/api/extractions/{ext['id']}/frames").json()["frames"]}
    for r in rs:                                                    # every result links back to its source frame
        assert r["original_image_path"] == manifest[r["image_id"]]["original_image_path"]
        assert r["video_timestamp_seconds"] == manifest[r["image_id"]]["video_timestamp_seconds"]
        assert (ROOT / r["annotated_image_path"]).exists() and r["annotated_image_path"] != r["original_image_path"]


@pytest.mark.skipif(not REFERENCE.exists(), reason="private uniform reference photo not available on this machine")
def test_successful_inference_on_uniform_photo(client, real_model, tmp_path):
    ref = cv2.imread(str(REFERENCE))
    ext = extract(client, make_video(tmp_path / "staff.mp4", [ref, ref]))
    frame = client.get(f"/api/extractions/{ext['id']}/frames").json()["frames"][0]
    orig = ROOT / frame["original_image_path"] if not Path(frame["original_image_path"]).is_absolute() else Path(frame["original_image_path"])
    before = hashlib.sha256(orig.read_bytes()).hexdigest()
    run = run_detection(client, ext, threshold=0.7)
    assert run["state"] == "done" and run["uniform_images"] >= 1 and run["detections"] >= 1
    rs = [r for r in rows(client, run) if r["detection_type"] == "uniform_shirt"]
    r = rs[0]
    assert float(r["confidence"]) >= 0.7 and int(r["detection_count"]) >= 1
    assert 0 <= int(r["bbox_xmin"]) < int(r["bbox_xmax"]) <= 640 and 0 <= int(r["bbox_ymin"]) < int(r["bbox_ymax"]) <= 480
    assert hashlib.sha256(orig.read_bytes()).hexdigest() == before                 # original unchanged
    res = client.get(run["results_url"]).json()["images"][0]
    assert client.get(res["annotated_url"]).status_code == 200 and client.get(res["original_url"]).status_code == 200
    print(f"\n  real model on the uniform reference photo: {[(x['prediction'], x['confidence']) for x in rows(client, run)]}")


# ------------------------------------------------------------------ result structure + failure isolation
class FakePerson:
    def __init__(self, box, label, conf, reason=""):
        self.box, self.label, self.conf, self.reason, self.extra = box, label, conf, reason, {"uniform_prob": conf}


def test_one_row_per_detection_and_failures_isolated(client, real_model, tmp_path, monkeypatch):
    imgs = [np.full((480, 640, 3), v, np.uint8) for v in (40, 90, 140, 190)]
    ext = extract(client, make_video(tmp_path / "four.mp4", imgs))                # frames at 0,1,2,3 s
    calls = {"n": 0}

    def fake(img, models, cfg, threshold):
        calls["n"] += 1
        if calls["n"] == 1:                                                         # two shirts in one image
            return [FakePerson((50, 50, 150, 400), "uniform", 0.95), FakePerson((300, 60, 420, 420), "uniform", 0.81),
                    FakePerson((500, 80, 600, 400), "non_uniform", 0.2)]
        if calls["n"] == 2:
            raise RuntimeError("simulated inference crash")
        return [FakePerson((100, 100, 140, 160), "uncertain", 0.0, "too small / far")]
    monkeypatch.setattr(detect, "classify_frame", fake)
    # make the last frame unreadable
    frames = client.get(f"/api/extractions/{ext['id']}/frames").json()["frames"]
    lp = frames[-1]["original_image_path"]
    last = ROOT / lp if not Path(lp).is_absolute() else Path(lp)
    last.write_bytes(b"corrupt")
    run = run_detection(client, ext)
    assert run["state"] == "done" and run["total"] == 4 and run["processed"] == 4
    assert run["succeeded"] == 2 and run["failed"] == 2 and run["detections"] == 2
    assert run["uniform_images"] == 1 and run["review_images"] == 1              # failures are not "review"
    rs = rows(client, run)
    first = [r for r in rs if r["image_id"] == frames[0]["image_id"]]
    # one row per person found, all linked to the same image; detection_count = target shirts only
    assert len(first) == 3 and {r["detection_count"] for r in first} == {"2"} and {r["person_count"] for r in first} == {"3"}
    assert [r["detection_type"] for r in first] == ["uniform_shirt", "uniform_shirt", "not_uniform"]
    assert [float(r["confidence"]) for r in first] == [0.95, 0.81, 0.2] and [r["detection_index"] for r in first] == ["1", "2", "3"]
    assert {r["prediction"] for r in first} == {"UNIFORM_DETECTED"}
    by_id = {r["image_id"]: r for r in rs}
    crashed = by_id[frames[1]["image_id"]]
    assert crashed["processing_status"] == "failed" and crashed["prediction"] == "PROCESSING_FAILED"
    assert crashed["confidence"] == crashed["detection_count"] == crashed["bbox_xmin"] == crashed["annotated_image_path"] == ""
    assert "simulated inference crash" in crashed["review_reason"]
    assert by_id[frames[2]["image_id"]]["prediction"] == "REVIEW_REQUIRED"           # person not judgeable
    assert "too small" in by_id[frames[2]["image_id"]]["review_reason"]
    assert by_id[frames[3]["image_id"]]["processing_status"] == "failed"
    assert by_id[frames[3]["image_id"]]["prediction"] == "PROCESSING_FAILED"           # never NO_UNIFORM_DETECTED
    assert "could not be read" in by_id[frames[3]["image_id"]]["review_reason"]
    # the failures are listed in processing_failures.csv with their stage and category
    pf = list(csv.DictReader(client.get("/api/frame-results/processing_failures.csv",
                                        params={"source_video": "four.mp4"}).text.splitlines()))
    mine = {r["image_id"]: r for r in pf if r["detection_run_id"] == run["id"]}
    assert mine[frames[1]["image_id"]]["error_category"] == "inference_error"
    assert mine[frames[3]["image_id"]]["error_category"] == "image_unreadable"
    assert {r["retry_status"] for r in mine.values()} == {"not_retried"} and {r["stage"] for r in mine.values()} == {"inference"}


def test_duplicate_run_reused_and_threshold_validated(client, real_model, tmp_path):
    ext = extract(client, make_video(tmp_path / "dup.mp4", [np.full((480, 640, 3), 100, np.uint8)]))
    a = run_detection(client, ext, threshold=0.8)
    b = client.post(f"/api/extractions/{ext['id']}/detect", json={"threshold": 0.8}).json()
    assert b["reused"] is True and b["id"] == a["id"]
    c = run_detection(client, ext, threshold=0.8, new_run=True)
    assert c["id"] != a["id"] and c["output_dir"] != a["output_dir"]                # new folder, old kept
    assert client.post(f"/api/extractions/{ext['id']}/detect", json={"threshold": 0.3}).status_code == 422
