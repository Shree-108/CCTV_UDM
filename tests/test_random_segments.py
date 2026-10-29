"""Default sampling: four screenshots, one random frame from each quarter of the video ("random across video")."""
import csv

import cv2
import numpy as np
import pytest

from backend.extraction.plan import plan_segments, segments
from tests.test_extraction import wait

FPS = 5
MINUTES = 20


@pytest.fixture(scope="module")
def video_20min(tmp_path_factory):
    """A real 20:00 MP4 (small and 5 fps so it is quick to write); each frame shows its own number."""
    path = tmp_path_factory.mktemp("v20") / "cctv_20min.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 120))
    for i in range(MINUTES * 60 * FPS):
        img = np.full((120, 160, 3), (i * 7) % 255, np.uint8)
        cv2.putText(img, str(i), (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        w.write(img)
    w.release()
    return path


# ------------------------------------------------------------------ planner
def test_one_distinct_frame_per_quarter():
    frames = MINUTES * 60 * FPS                                          # 6000 frames, last = 5999
    s = plan_segments(MINUTES * 60.0, FPS, frames, 4, 1234, include_initial=False)
    assert len(s) == 4 and {x.sample_type for x in s} == {"random"}
    parts = segments(0, frames - 1, 4)
    assert parts == [(0, 1499), (1500, 2999), (3000, 4499), (4500, 5999)]
    assert all(lo <= x.target_frame <= hi for x, (lo, hi) in zip(s, parts))  # one per quarter, inside the video
    assert len({x.target_frame for x in s}) == 4
    assert all(0 <= x.target_seconds <= (frames - 1) / FPS for x in s)
    assert s == plan_segments(MINUTES * 60.0, FPS, frames, 4, 1234, False)            # reproducible
    assert s != plan_segments(MINUTES * 60.0, FPS, frames, 4, 1235, False)            # seed matters
    with_initial = plan_segments(MINUTES * 60.0, FPS, frames, 4, 1234, True)
    assert [x.sample_type for x in with_initial] == ["initial"] + ["random"] * 4       # initial is extra
    assert 0 not in [x.target_frame for x in with_initial[1:]]                         # never drawn twice


def test_too_short_is_an_error_not_fewer_images():
    with pytest.raises(ValueError, match="too short for 4 distinct screenshots"):
        plan_segments(0.6, FPS, 3, 4, 1, False)
    assert len(plan_segments(0.8, FPS, 4, 4, 1, False)) == 4                          # exactly enough frames


# ------------------------------------------------------------------ end to end on a 20-minute video
def test_20_minute_video_gives_exactly_four_reproducible_screenshots(client, video_20min):
    m = client.post("/api/media", files={"file": (video_20min.name, video_20min.read_bytes())}).json()
    assert abs(m["duration_sec"] - 1200) < 0.5, m

    # defaults only (what START PROCESSING sends when nothing is changed): random across video, 4, no initial frame
    plan = client.get(f"/api/media/{m['id']}/extraction-plan", params={"sampling_strategy": "random_segments",
                                                                       "include_initial": False}).json()
    assert plan["total"] == 4 and plan["initial"] == 0 and len(plan["segments"]) == 4
    r = client.post("/api/extractions", json={"media_id": m["id"]})
    assert r.status_code == 202, r.text
    j = wait(client, r.json()["id"], timeout=120)
    assert j["state"] == "done" and j["sampling_strategy"] == "random_segments" and j["sample_count"] == 4
    assert j["planned"] == j["extracted"] == 4 and j["failed"] == 0 and j["error"] is None
    assert isinstance(j["random_seed"], int) and j["include_initial"] is False
    assert [x["status"] for x in j["samples"]] == ["saved"] * 4

    rows = list(csv.DictReader(client.get(j["manifest_url"]).text.splitlines()))
    assert len(rows) == 4 and len({x["image_id"] for x in rows}) == 4 and len({x["image_filename"] for x in rows}) == 4
    secs = [float(x["video_timestamp_seconds"]) for x in rows]
    assert all(q * 300 <= t < (q + 1) * 300 for q, t in enumerate(secs)), secs      # one per 5-minute quarter
    assert all(x["video_timestamp_hhmmss"] and x["sample_type"] == "random" for x in rows)
    assert [int(x["frame_number"]) for x in rows] == [x["target_frame"] for x in j["samples"]]
    # the saved image is the decoded frame at that position, unchanged
    cap = cv2.VideoCapture(str(video_20min))
    for x in rows:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(x["frame_number"]))
        ok, f = cap.read()
        img = cv2.imdecode(np.frombuffer(client.get(f"/api/extractions/{j['id']}/frames/{x['image_filename']}").content,
                                         np.uint8), cv2.IMREAD_COLOR)
        assert ok and np.array_equal(img, f)
    cap.release()

    # the recorded seed + same settings reproduce the same timestamps
    j2 = wait(client, client.post("/api/extractions", json={"media_id": m["id"], "random_seed": j["random_seed"],
                                                           "new_run": True}).json()["id"], timeout=120)
    rows2 = list(csv.DictReader(client.get(j2["manifest_url"]).text.splitlines()))
    assert j2["id"] != j["id"] and [x["video_timestamp_seconds"] for x in rows2] == [x["video_timestamp_seconds"] for x in rows]
    again = client.post("/api/extractions", json={"media_id": m["id"], "random_seed": j["random_seed"]}).json()
    # an identical finished job is reused (j and j2 have the same seed and settings - either may be returned)
    assert again["reused"] is True and again["id"] in (j["id"], j2["id"])

    # the count stays configurable; fixed interval is still available
    j6 = wait(client, client.post("/api/extractions", json={"media_id": m["id"], "sample_count": 6, "new_run": True}).json()["id"], timeout=120)
    assert j6["extracted"] == 6 and len(j6["samples"]) == 6
    ji = client.post("/api/extractions", json={"media_id": m["id"], "interval_seconds": 300, "new_run": True}).json()
    ji = wait(client, ji["id"], timeout=120)
    assert ji["sampling_strategy"] == "interval" and ji["include_initial"] is True and ji["extracted"] == 4   # 0,5,10,15 min


def test_short_video_is_refused_with_a_reason(client, tmp_path):
    path = tmp_path / "three_frames.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 120))
    for i in range(3):
        w.write(np.full((120, 160, 3), 60 * i, np.uint8))
    w.release()
    m = client.post("/api/media", files={"file": (path.name, path.read_bytes())}).json()
    r = client.post("/api/extractions", json={"media_id": m["id"]})
    assert r.status_code == 422 and "too short for 4 distinct screenshots" in r.json()["detail"]
