"""Frames are decoded in order, timed by the video's own clock and checked before they are saved.

The cart NVR exports are HEVC in MPEG-PS with a keyframe every ~2 s and gaps in the recording. Seeking
straight to a frame number there returned grey frames (or a different frame), and a frame after a gap has
no reference pictures at all. A corrupted frame must never be saved, analysed or reported as "no person
detected"; every saved screenshot must be the real decoded frame at the recorded time."""
import csv
import re
import subprocess
import time

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from backend.analysis.media import probe
from backend.extraction import cart_report, detect, frame_quality
from backend.extraction.extractor import FRAME_UNREADABLE, extract
from backend.extraction.plan import Sample
from backend.extraction.screenshots import remark
from tests.test_compliance_reports import Person
from tests.test_extraction import MEDIA
from tests.test_frame_detection import empty_scene, make_video, real_model, wait  # noqa: F401  (fixture re-used)
from tests.test_s3_batch import BUCKET, DAY, FakeS3, fake_s3, local_sheet, wait_batch  # noqa: F401  (fixture re-used)

FPS = 20


@pytest.fixture(scope="module")
def nvr_video(tmp_path_factory):
    """12 s of a moving test pattern as HEVC in MPEG-PS with a keyframe every 40 frames (like the cart NVRs),
    with 5 % of the stream cut out after 45 %: a recording gap whose following frames lack their references."""
    d = tmp_path_factory.mktemp("nvr")
    src, out = d / "src.mpg", d / "nvr_like.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
           f"testsrc2=size=320x240:rate={FPS}", "-t", "12", "-c:v", "libx265", "-x265-params",
           "keyint=40:min-keyint=40:scenecut=0:bframes=0:log-level=error", "-pix_fmt", "yuv420p", "-f", "mpeg", str(src)]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        pytest.skip("this ffmpeg build cannot encode HEVC")
    data, packs, i = src.read_bytes(), [], 0
    while (i := data.find(b"\x00\x00\x01\xba", i)) >= 0:                  # MPEG-PS pack headers
        packs.append(i)
        i += 4
    out.write_bytes(data[:packs[len(packs) * 45 // 100]] + data[packs[len(packs) * 50 // 100]:])
    return out


@pytest.fixture(scope="module")
def reference(nvr_video):
    """Every frame decoded in order (OpenCV), with ffmpeg's own presentation times - the ground truth."""
    cap, frames = cv2.VideoCapture(str(nvr_video)), []
    while (r := cap.read())[0]:
        frames.append(r[1])
    log = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(nvr_video), "-vf", "showinfo",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    pts = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", log)]
    assert len(pts) == len(frames)
    bad = [n for n, f in enumerate(frames) if frame_quality.problem(f)]
    assert bad, "the cut should leave frames without their reference pictures"
    return frames, [p - pts[0] for p in pts], bad


def meta(job_id="ext-test-000001"):
    return {"extraction_job_id": job_id, "source_video": "nvr_like.mp4", "source_video_id": "abcdef0123456789",
            "store_id": "cart_50", "camera_id": "", "sampling_strategy": "random_segments",
            "sampling_interval_seconds": "", "random_seed": 7, "recording_start_time": ""}


def run_extract(nvr_video, tmp_path, frames, job_id="ext-test-000001"):
    samples = [Sample(i, "random", round(f / FPS, 6), f) for i, f in enumerate(frames, 1)]
    res = extract(str(nvr_video), meta(job_id), tmp_path / job_id, FPS, samples)
    manifest = list(csv.DictReader(open(tmp_path / job_id / "image_manifest.csv", encoding="utf-8")))
    failures = list(csv.DictReader(open(tmp_path / job_id / "failures.csv", encoding="utf-8")))
    return res, manifest, failures


# ------------------------------------------------------------------ the frame check itself
def test_frame_check_accepts_real_pictures_and_rejects_broken_ones():
    scene = cv2.resize(cv2.imread(str(MEDIA / "dvr_grid.jpg")), (640, 480))     # a real CCTV picture
    assert frame_quality.problem(scene) is None
    gray = cv2.cvtColor(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert frame_quality.problem(gray) is None                                  # a grayscale (IR) picture is fine
    assert frame_quality.problem((scene * 0.15).astype(np.uint8)) is None      # so is a dark one
    fill = np.full((480, 640, 3), 128, np.uint8)                               # decoder grey fill ...
    cv2.rectangle(fill, (200, 150), (330, 470), (90, 110, 140), 3)            # ... with a ghost outline
    assert "grey fill" in frame_quality.problem(fill)
    half = scene.copy()
    half[160:] = 128                                                            # two thirds of the picture missing
    assert "grey fill" in frame_quality.problem(half)
    assert "blank" in frame_quality.problem(np.zeros((480, 640, 3), np.uint8))
    assert frame_quality.problem(None) and frame_quality.problem(np.zeros((0, 0, 3), np.uint8))


# ------------------------------------------------------------------ probe
def test_probe_counts_the_frames_that_exist(nvr_video, reference):
    frames, pts, _ = reference
    cap = cv2.VideoCapture(str(nvr_video))
    header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    m = probe(nvr_video, nvr_video.name, "video", nvr_video.stat().st_size)
    assert header > len(frames)                                   # the container claims frames that are not there
    assert len(frames) - 1 <= m["frames"] <= len(frames)          # never a frame that does not exist
    assert abs(m["duration_sec"] - (pts[-1] + 1 / FPS)) <= 0.15   # from the timestamps, gap included


# ------------------------------------------------------------------ extraction
def test_screenshots_are_the_real_frames_at_their_real_time(nvr_video, reference, tmp_path):
    frames, pts, bad = reference
    good = [n for n in range(len(frames)) if n not in bad]
    after_gap = next(n for n in good if n > bad[-1])               # first readable frame after the gap
    wanted = sorted({0, good[len(good) // 4], bad[0], bad[len(bad) // 2], after_gap, good[-1]})
    res, manifest, failures = run_extract(nvr_video, tmp_path, wanted)

    saved = {int(r["frame_number"]): r for r in manifest}
    assert sorted(saved) == [n for n in wanted if n not in bad]
    assert res["extracted"] == len(saved) and res["failed"] == len(failures)
    for n, row in saved.items():
        img = cv2.imread(str(tmp_path / "ext-test-000001" / "frames" / row["image_filename"]))
        assert np.array_equal(img, frames[n]), f"frame {n} is not the frame decoded at that position"
        assert abs(float(row["video_timestamp_seconds"]) - pts[n]) < 0.002, (n, row["video_timestamp_seconds"], pts[n])
    # the gap is in the timestamps: frame number / fps would put the frame after it too early
    assert float(saved[after_gap]["video_timestamp_seconds"]) - after_gap / FPS > 0.3
    secs = [float(r["video_timestamp_seconds"]) for r in manifest]
    assert secs == sorted(secs) and len(set(secs)) == len(secs)   # chronological, no duplicates

    # corrupted frames: a failure with its reason, no image file, nothing to analyse
    assert {int(f["target_frame"]) for f in failures} == {n for n in wanted if n in bad}
    assert {f["error_category"] for f in failures} == {FRAME_UNREADABLE}
    assert all("grey fill" in f["error_message"] for f in failures)
    for f in failures:                       # the real time of the decoded frame, not frame / fps
        assert abs(float(f["target_seconds"]) - pts[int(f["target_frame"])]) < 0.002, (f, pts[int(f["target_frame"])])
    files = sorted(p.name for p in (tmp_path / "ext-test-000001" / "frames").iterdir())
    assert files == sorted(r["image_filename"] for r in manifest)


def test_same_video_same_frames_every_time(nvr_video, reference, tmp_path):
    frames, _, bad = reference
    wanted = [5, 60, bad[0] - 1, 180, len(frames) - 3]
    _, a, fa = run_extract(nvr_video, tmp_path, wanted, "ext-test-00000a")
    _, b, fb = run_extract(nvr_video, tmp_path, wanted, "ext-test-00000b")
    key = ("frame_number", "video_timestamp_seconds", "video_timestamp_hhmmss", "image_width", "image_height")
    assert [[r[k] for k in key] for r in a] == [[r[k] for k in key] for r in b]
    for ra, rb in zip(a, b):
        ia = (tmp_path / "ext-test-00000a" / "frames" / ra["image_filename"]).read_bytes()
        ib = (tmp_path / "ext-test-00000b" / "frames" / rb["image_filename"]).read_bytes()
        assert ia == ib
    assert [f["target_frame"] for f in fa] == [f["target_frame"] for f in fb]


def test_frame_past_the_real_end_is_a_failure(nvr_video, reference, tmp_path):
    frames, _, _ = reference
    res, manifest, failures = run_extract(nvr_video, tmp_path, [10, len(frames) + 5])
    assert [int(r["frame_number"]) for r in manifest] == [10]
    assert failures[0]["error_category"] == "no_frame" and "the video ends after" in failures[0]["error_message"]


# ------------------------------------------------------------------ dashboard quick analysis + reports
def test_unreadable_frames_are_reported_as_unreadable_not_no_person(client, real_model, nvr_video, reference,  # noqa: F811
                                                                    monkeypatch):
    calls = []
    monkeypatch.setattr(detect, "classify_frame", lambda img, *a: calls.append(img.shape) or [])
    m = client.post("/api/media", files={"file": (nvr_video.name, nvr_video.read_bytes())}).json()
    j = client.post("/api/extractions", json={"media_id": m["id"], "interval_seconds": 1, "include_initial": True,
                                              "run_detection": True, "new_run": True}).json()
    j = wait(client, f"/api/extractions/{j['id']}")
    assert j["state"] == "done" and j["failed"] > 0 and j["extracted"] > 0
    for _ in range(400):                                                    # detection starts after extraction
        c = client.get(f"/api/extractions/{j['id']}/uniform-compliance").json()
        if c["run_state"] == "done":
            break
        time.sleep(0.25)
    assert c["run_state"] == "done", c
    recs = c["records"]
    assert len(recs) == j["planned"] and len(calls) == j["extracted"]       # the model never saw a broken frame
    broken = [r for r in recs if r["processing_status"] == FRAME_UNREADABLE]
    assert len(broken) == j["failed"]
    for r in broken:
        assert r["prediction"] == "" and r["row"]["confidence score"] == "" and r["row"]["img name"] == ""
        assert "unreadable" in r["row"]["remark (Example)"] and "No person" not in r["row"]["remark (Example)"]
    assert {r["prediction"] for r in recs if r not in broken} == {"NO_UNIFORM_DETECTED"}
    csv_rows = list(csv.DictReader(client.get(f"/api/extractions/{j['id']}/uniform_compliance.csv").text.splitlines()))
    assert len(csv_rows) == len(recs) and sum("unreadable" in r["remark (Example)"] for r in csv_rows) == len(broken)


def test_report_wording_for_unreadable_frames():
    rec = {"processing_status": FRAME_UNREADABLE, "people": []}
    assert cart_report.score(rec) is None                                       # never 0
    assert cart_report.remark(rec).startswith("frame unreadable")
    assert "unreadable" in remark([], "", FRAME_UNREADABLE)


def test_s3_cart_report_keeps_rows_of_a_video_with_only_unreadable_frames(client, real_model, fake_s3,  # noqa: F811
                                                                          tmp_path, monkeypatch):
    monkeypatch.setattr(detect, "classify_frame", lambda *a: [Person((100, 80, 220, 460), "uniform", 0.9)])
    grey = np.full((480, 640, 3), 128, np.uint8)                               # nothing but decoder grey fill
    broken = make_video(tmp_path / "broken.mp4", [grey] * 4).read_bytes()
    fine = make_video(tmp_path / "fine.mp4", [empty_scene(90 + 30 * i) for i in range(4)]).read_bytes()
    fake = fake_s3(FakeS3({f"{DAY}/cart_50/input_data/a_broken.mp4": broken, f"{DAY}/cart_50/input_data/b_fine.mp4": fine}))
    b = wait_batch(client, client.post("/api/s3/process", json={"date": "30-10-2026", "carts": ["50"]}).json()["id"])
    c = b["carts"][0]
    assert c["state"] == "done" and c["summary"]["screenshots"] == 8 and c["summary"]["failed"] == 1
    files = {f["name"]: f for f in c["files"]}
    assert files["a_broken.mp4"]["state"] == "failed" and files["a_broken.mp4"]["screenshots"] == 4
    header, *rows = local_sheet(b["id"], "cart_50")
    assert len(rows) == 8
    unreadable = [r for r in rows if "frame unreadable" in r[6]]
    assert len(unreadable) == 4 and all(r[5] is None for r in unreadable)    # blank score, never 0
    assert all("person1 wearing the dress" in r[6] and r[5] == 90 for r in rows if r not in unreadable)
