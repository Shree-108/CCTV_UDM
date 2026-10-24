"""Scheduled capture (Phase 2): slots, local sources, capture -> analysis traceability,
no duplicate slots, failures never lose the original frame."""
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.capture import sources
from backend.capture.schedule import current_slot, next_slot, parse_hhmm, slots_per_day
from backend.capture.service import CaptureBusy

MEDIA = Path(__file__).parent / "media"
T = parse_hhmm


# ------------------------------------------------------------------ schedule (pure)
@pytest.mark.parametrize("now,cur,nxt", [
    ("08:59", None, "25 09:00"), ("09:00", "09:00", "25 09:15"), ("09:14", "09:00", "25 09:15"),
    ("20:45", "20:45", "26 09:00"), ("20:59", "20:45", "26 09:00"), ("21:00", None, "26 09:00"),
])
def test_slots_9_to_21_every_15(now, cur, nxt):
    t = datetime(2026, 10, 25, int(now[:2]), int(now[3:]))
    c = current_slot(t, 15, T("09:00"), T("21:00"))
    assert (c.strftime("%H:%M") if c else None) == cur
    assert next_slot(t, 15, T("09:00"), T("21:00")).strftime("%d %H:%M") == nxt


def test_slots_per_day_and_uneven_interval():
    assert slots_per_day(15, T("09:00"), T("21:00")) == 48       # 4 per hour, last 20:45
    assert slots_per_day(45, T("09:00"), T("21:00")) == 16       # 09:00 ... 20:15
    assert current_slot(datetime(2026, 1, 1, 20, 59), 45, T("09:00"), T("21:00")).strftime("%H:%M") == "20:15"


def test_overnight_window():
    s, e = T("22:00"), T("06:00")
    assert current_slot(datetime(2026, 1, 2, 1, 7), 15, s, e) == datetime(2026, 1, 2, 1, 0)
    assert current_slot(datetime(2026, 1, 2, 12, 0), 15, s, e) is None
    assert next_slot(datetime(2026, 1, 2, 5, 50), 15, s, e) == datetime(2026, 1, 2, 22, 0)


# ------------------------------------------------------------------ sources
def _folder(tmp_path, names=("store_uniform.jpg", "dvr_grid.jpg")) -> Path:
    d = tmp_path / "cam"
    d.mkdir()
    for n in names:
        shutil.copy(MEDIA / n, d / n)
    return d


def test_folder_source_cycles_and_wraps(tmp_path):
    d = _folder(tmp_path)
    cam = {"source_type": "folder", "source": str(d), "cursor": 0}
    refs = []
    for _ in range(3):
        frame, ref, cam["cursor"] = sources.grab(cam, 60)
        assert frame is not None and frame.ndim == 3
        refs.append(ref)
    assert refs == ["dvr_grid.jpg", "store_uniform.jpg", "dvr_grid.jpg"]     # sorted, wraps around


def test_video_source_steps_and_wraps():
    cam = {"source_type": "video", "source": str(MEDIA / "dvr_grid.mp4"), "cursor": 0}
    _, r1, c1 = sources.grab(cam, 7)
    _, r2, c2 = sources.grab(cam | {"cursor": c1}, 7)
    assert r1.endswith("@0.0s") and r2.endswith("@7.0s") and c2 == 14
    _, r3, _ = sources.grab(cam | {"cursor": 10_000}, 7)                    # beyond the end -> wraps
    assert "@" in r3


def test_source_validation(tmp_path):
    with pytest.raises(ValueError):
        sources.validate_source("rtsp", "rtsp://example")                    # live CCTV not enabled
    with pytest.raises(ValueError):
        sources.validate_source("folder", str(tmp_path / "missing"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        sources.validate_source("folder", str(tmp_path / "empty"))
    with pytest.raises(ValueError):
        sources.validate_source("video", str(MEDIA / "store_uniform.jpg"))


def test_dhash_similarity():
    img = cv2.imread(str(MEDIA / "store_uniform.jpg"))
    noisy = np.clip(img.astype(int) + 3, 0, 255).astype(np.uint8)
    other = cv2.imread(str(MEDIA / "dvr_grid.jpg"))
    assert sources.hamming(sources.dhash(img), sources.dhash(noisy)) <= 4
    assert sources.hamming(sources.dhash(img), sources.dhash(other)) > 4


# ------------------------------------------------------------------ API + service
@pytest.fixture()
def cam(client, tmp_path):
    d = _folder(tmp_path, ("store_uniform.jpg",))
    r = client.post("/api/cameras", json={"id": "CAM-T1", "source_type": "folder", "source": str(d)})
    assert r.status_code == 201, r.text
    yield r.json()
    client.delete("/api/cameras/CAM-T1")


def svc(client):
    return client.app.state.app_state.capture


def test_camera_defaults_and_validation(client, cam, tmp_path):
    assert cam["store_id"] == "STORE-001" and cam["name"] == "CAM-T1" and cam["enabled"] and cam["area"] is None
    assert client.post("/api/cameras", json={"id": "CAM-T1", "source_type": "folder", "source": cam["source"]}).status_code == 409
    assert client.post("/api/cameras", json={"id": "bad id!", "source_type": "folder", "source": cam["source"]}).status_code == 422
    assert client.post("/api/cameras", json={"id": "CAM-X", "source_type": "rtsp", "source": "rtsp://x"}).status_code == 422
    assert client.post("/api/cameras", json={"id": "CAM-X", "source_type": "folder", "source": str(tmp_path / "nope")}).status_code == 422


def test_manual_capture_is_analysed_and_traceable(client, cam):
    r = client.post("/api/cameras/CAM-T1/capture")
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["status"] == "done" and c["trigger"] == "manual" and c["slot_at"] is None
    assert c["source_ref"] == "store_uniform.jpg" and len(c["frame_hash"]) == 16 and c["similar_to_previous"] == 0
    assert client.get(c["image_url"]).status_code == 200                     # original frame kept
    a = client.get(f"/api/analyses/{c['analysis_id']}").json()
    assert a["state"] == "done" and a["source"] == "capture" and a["camera_id"] == "CAM-T1"
    assert a["store_id"] == "STORE-001" and a["capture_id"] == c["id"] and a["captured_at"] == c["captured_at"]
    assert a["model_version"] and a["files"]["annotated"]
    listed = client.get("/api/analyses", params={"camera_id": "CAM-T1", "source": "capture"}).json()
    assert listed["total"] >= 1 and all(x["camera_id"] == "CAM-T1" for x in listed["analyses"])
    # the same frame again is flagged as similar (still analysed: it is a valid observation)
    c2 = client.post("/api/cameras/CAM-T1/capture").json()
    assert c2["status"] == "done" and c2["similar_to_previous"] == 1


def test_scheduler_one_capture_per_slot(client, cam):
    s = svc(client)
    at_0915 = datetime(2026, 10, 25, 9, 16)
    first = s.tick(at_0915)
    assert [c["camera_id"] for c in first] == ["CAM-T1"] and first[0]["trigger"] == "schedule"
    assert s.tick(datetime(2026, 10, 25, 9, 29)) == []                        # same slot: no duplicate
    assert len(s.tick(datetime(2026, 10, 25, 9, 30))) == 1                    # next slot
    assert s.tick(datetime(2026, 10, 25, 21, 5)) == []                        # outside the window
    caps = client.get("/api/captures", params={"camera_id": "CAM-T1"}).json()["captures"]
    assert len({c["slot_at"] for c in caps if c["trigger"] == "schedule"}) == 2


def test_scheduler_respects_switches(client, cam):
    s = svc(client)
    client.put("/api/cameras/CAM-T1", json={"enabled": False})
    assert s.tick(datetime(2026, 10, 25, 10, 1)) == []
    client.put("/api/cameras/CAM-T1", json={"enabled": True})
    client.put("/api/settings", json={"capture_enabled": False})
    try:
        assert s.tick(datetime(2026, 10, 25, 10, 1)) == []
    finally:
        client.put("/api/settings", json={"capture_enabled": True})
    assert len(s.tick(datetime(2026, 10, 25, 10, 1))) == 1


def test_capture_failure_is_recorded(client, cam):
    for f in Path(cam["source"]).iterdir():
        f.unlink()                                                            # source emptied after setup
    c = client.post("/api/cameras/CAM-T1/capture").json()
    assert c["status"] == "capture_failed" and "no images" in c["error"] and c["media_id"] is None


def test_analysis_failure_keeps_original(client, cam, monkeypatch):
    import backend.analysis.runner as runner

    def boom(*a, **k):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(runner, "analyze_image", boom)
    c = client.post("/api/cameras/CAM-T1/capture").json()
    assert c["status"] == "analysis_failed" and "model exploded" in c["error"]
    assert client.get(c["image_url"]).status_code == 200                     # original frame still there
    a = client.get(f"/api/analyses/{c['analysis_id']}").json()
    assert a["state"] == "failed" and a["camera_id"] == "CAM-T1"


def test_busy_camera_is_not_captured_twice(client, cam):
    s = svc(client)
    s._inflight.add("CAM-T1")
    try:
        with pytest.raises(CaptureBusy):
            s.capture_now("CAM-T1")
        assert client.post("/api/cameras/CAM-T1/capture").status_code == 409
        assert s.tick(datetime(2026, 10, 25, 11, 0)) == []                    # scheduler skips a busy camera
    finally:
        s._inflight.discard("CAM-T1")


def test_restart_marks_running_captures_failed(client, cam):
    db = client.app.state.app_state.db
    db.add_capture({"id": "cap-stuck", "camera_id": "CAM-T1", "store_id": "STORE-001",
                    "captured_at": "2026-10-25T00:00:00+00:00", "trigger": "schedule", "status": "running"})
    db.interrupt_captures()
    assert db.capture("cap-stuck")["status"] == "capture_failed"


def test_delete_camera_keeps_history(client, cam):
    c = client.post("/api/cameras/CAM-T1/capture").json()
    assert client.delete("/api/cameras/CAM-T1").json()["ok"]
    assert client.get("/api/cameras/CAM-T1").status_code == 404
    assert client.get(f"/api/analyses/{c['analysis_id']}").status_code == 200
    assert client.get("/api/captures", params={"camera_id": "CAM-T1"}).json()["total"] >= 1


def test_capture_status(client):
    st = client.get("/api/capture/status").json()
    assert st["interval_minutes"] == 15 and st["active_start"] == "09:00" and st["active_end"] == "21:00"
    assert st["slots_per_day"] == 48 and st["scheduler_running"] is False    # disabled in tests
