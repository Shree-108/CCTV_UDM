"""Video Analysis and the Results Dashboard run the SAME quick sampled analysis (4 random frames).

The backend service is POST /api/media + POST /api/extractions (run_detection) -> ExtractionRunner ->
extractor.extract (seeks to the 4 planned frames) -> detect.run (existing model on those 4 images).
Both pages render the same frontend panel (ProcessPanel), which is what calls that service."""
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.extraction import detect, extractor, master_csv
from tests.test_compliance_reports import Person, csv_rows, run_job
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "frontend" / "features"


def src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ------------------------------------------------------------------ one shared workflow
def test_both_pages_render_the_same_sampled_analysis_panel():
    video_page = src(ROOT / "app" / "analysis" / "video" / "page.tsx")
    assert "features/analysis/VideoAnalysisPage" in video_page
    va = src(FRONT / "analysis" / "VideoAnalysisPage.tsx")
    dash = src(FRONT / "results" / "ResultsDashboard.tsx")
    assert "import { ProcessPanel } from '../results/ResultsDashboard'" in va and "<ProcessPanel" in va
    assert re.search(r"export function ProcessPanel", dash) and "<ProcessPanel" in dash
    # the panel's start button is Configure, which creates the extraction job WITH detection
    assert "<Configure" in dash and "run_detection: detectAfter" in src(FRONT / "extraction" / "ExtractionPage.tsx")
    # quick sampled analysis is the default; the full-video mode stays reachable and keeps its own links
    assert re.search(r"const full = params\.get\('mode'\) === 'full' \|\| params\.has\('analysis'\) \|\| params\.has\('media'\)", va)
    assert "<AnalysisWorkspace mode=\"video\" />" in va
    for stage in ("Received", "Validating", "Sampling frames", "Analyzing", "Saving results", "Completed", "Failed"):
        assert f"'{stage}'" in dash, stage


def test_quick_mode_reads_and_classifies_only_the_four_frames(client, real_model, monkeypatch):  # noqa: F811
    reads, calls = {"n": 0}, {"n": 0}
    real = extractor.cv2.VideoCapture

    class CountingCapture:                       # every decoded frame the EXTRACTOR asks for (the upload probe's
        def __init__(self, *a):                  # 3 preview reads use the same cv2 module and are not counted)
            self._c = real(*a)

        def read(self):
            if sys._getframe(1).f_code.co_filename == extractor.__file__:
                reads["n"] += 1
            return self._c.read()

        def __getattr__(self, k):
            return getattr(self._c, k)

    def fake(img, models, cfg, thr):
        calls["n"] += 1
        return [Person((100, 80, 220, 460), "uniform", 0.9)]
    monkeypatch.setattr(extractor.cv2, "VideoCapture", CountingCapture)
    monkeypatch.setattr(detect, "classify_frame", fake)
    media_frames = 500                            # tests/media/dvr_grid.mp4
    j, run = run_job(client, {"sampling_strategy": "random_segments", "sample_count": 4, "random_seed": 81})
    assert run["state"] == "done" and run["processed"] == 4
    assert reads["n"] == 4 < media_frames         # seeks to each planned frame; never decodes the whole video
    assert calls["n"] == 4                        # the model runs on the 4 screenshots only

    # four distinct frames, one per quarter, seed and timestamps persisted on the job
    db = client.app.state.app_state.db
    job = db.extraction(j["id"])
    assert job["random_seed"] == 81 and job["sampling_strategy"] == "random_segments"
    samples = job["plan"]["samples"]
    frames = [s["target_frame"] for s in samples]
    assert len(set(frames)) == 4 and frames == sorted(frames)
    for k, f in enumerate(frames):
        assert k * media_frames // 4 <= f < (k + 1) * media_frames // 4
    man = csv_rows(client, j["manifest_url"])
    assert [int(m["frame_number"]) for m in man] == frames
    assert [m["video_timestamp_hhmmss"] for m in man] == [s["target_hhmmss"] for s in samples]
    # the same seed re-plans the same frames (reproducible)
    plan = client.get(f"/api/media/{j['media_id']}/extraction-plan",
                      params={"sampling_strategy": "random_segments", "sample_count": 4, "random_seed": 81,
                              "include_initial": False}).json()
    assert plan["first_timestamps"] == [s["target_seconds"] for s in samples]


def test_results_survive_an_application_restart(client, real_model, monkeypatch):  # noqa: F811
    monkeypatch.setattr(detect, "classify_frame", lambda img, models, cfg, thr: [Person((100, 80, 220, 460), "uniform", 0.9)])
    j, run = run_job(client, {"random_seed": 91, "store_id": "store id 91", "camera_id": "CAM-RESTART"})
    before = client.get("/api/frame-results", params={"camera_id": "CAM-RESTART"}).json()
    rows_before = client.get(f"/api/extractions/{j['id']}/uniform-compliance").json()["records"]
    assert j["id"] in {e["job_id"] for e in master_csv._ledger()}

    from backend.main import create_app            # a second, fresh application on the same database + storage
    with TestClient(create_app(preload=False)) as again:
        after = again.get("/api/frame-results", params={"camera_id": "CAM-RESTART"}).json()
        assert after["summary"] == before["summary"] and len(after["images"]) == 4
        assert again.get(f"/api/extractions/{j['id']}/uniform-compliance").json()["records"] == rows_before
        st = again.get("/api/uniform-compliance/master").json()
        assert j["id"] not in {x["job_id"] for x in st["pending_jobs"]}
        master_text = again.get("/api/uniform-compliance/master.csv").text
        assert all(r["row"]["img name"] in master_text for r in rows_before)
        for r in rows_before:
            assert again.get(r["original_url"]).status_code == 200


@pytest.mark.parametrize("name", ["notes.txt", "clip.exe"])
def test_unsupported_formats_are_refused(client, name):
    r = client.post("/api/media", files={"file": (name, b"not a video")})
    assert r.status_code == 415
