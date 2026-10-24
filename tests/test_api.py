"""End-to-end API tests with real model inference on synthetic CCTV media.

tests/media/dvr_grid.*: a 2x2 DVR layout. Camera 1 (top-left) is a street/store scene in
which one person wears a painted black-and-white vertically striped shirt; Camera 4
(bottom-right) is the SAME scene without the striped shirt; Camera 3 is empty.
"""
import time

from pathlib import Path

MEDIA = Path(__file__).resolve().parent / "media"

CAM1 = [0, 0, 0.5, 0.5]
CAM2 = [0.5, 0, 1, 0.5]
CAM3 = [0, 0.5, 0.5, 1]
CAM4 = [0.5, 0.5, 1, 1]


def upload(client, name):
    with open(MEDIA / name, "rb") as f:
        r = client.post("/api/media", files={"file": (name, f)})
    assert r.status_code == 200, r.text
    return r.json()


def analyse(client, media_id, rect, roi=None, label="Camera 1"):
    r = client.post("/api/analyses", json={"media_id": media_id, "area": {"rect": rect, "roi": roi, "label": label, "layout": "2x2"}})
    assert r.status_code == 200, r.text
    return r.json()


def test_health_reports_model_status(client):
    h = client.get("/api/health").json()
    assert h["person_detector"]["state"] == "ready"
    assert h["uniform_classifier"]["state"] == "not_trained"      # tests pin SU_UNIFORM_MODEL to a missing file
    assert h["live_cctv"] == "not_implemented"


def test_upload_probe_and_layout_suggestion(client):
    m = upload(client, "dvr_grid.jpg")
    assert m["kind"] == "image" and m["width"] == 1280 and m["suggested_layout"] == "2x2"
    assert client.get(m["preview_url"]).status_code == 200
    v = upload(client, "dvr_grid.mp4")
    assert v["kind"] == "video" and v["frames"] == 500 and v["fps"] == 25 and v["suggested_layout"] == "2x2"


def test_rejects_unsupported_file(client):
    r = client.post("/api/media", files={"file": ("notes.txt", b"hello")})
    assert r.status_code == 415


def test_image_only_selected_grid_is_analysed(client):
    m = upload(client, "dvr_grid.jpg")
    a1 = analyse(client, m["id"], CAM1)
    assert a1["state"] == "done" and a1["status"] == "compliant" and a1["uniformed"] == 1 and a1["people"] == 4
    a4 = analyse(client, m["id"], CAM4, label="Camera 4")
    assert a4["status"] == "no_uniform" and a4["uniformed"] == 0
    assert any(al["kind"] == "no_uniform" and al["evidence_url"] for al in a4["alert_list"])
    a3 = analyse(client, m["id"], CAM3, label="Camera 3")
    assert a3["status"] == "no_person" and a3["people"] == 0
    for a in (a1, a4):
        for k in ("annotated", "report_json", "report_csv", "report_txt"):
            assert client.get(a["files"][k]).status_code == 200


def test_roi_excludes_people_outside(client):
    m = upload(client, "dvr_grid.jpg")
    # the striped-shirt person stands at the right edge of camera 1 -> left 85% of the tile excludes them
    roi = [[0, 0], [0.85, 0], [0.85, 1], [0, 1]]
    a = analyse(client, m["id"], CAM1, roi=roi)
    assert a["status"] == "no_uniform" and a["uniformed"] == 0 and a["people"] >= 1


def test_invalid_area_rejected(client):
    m = upload(client, "dvr_grid.jpg")
    r = client.post("/api/analyses", json={"media_id": m["id"], "area": {"rect": [0.5, 0, 0.2, 1]}})
    assert r.status_code == 422


def test_video_tracking_temporal_and_reports(client):
    v = upload(client, "dvr_grid.mp4")
    a = analyse(client, v["id"], CAM1)
    aid = a["id"]
    deadline = time.time() + 240
    seen_live = False
    while time.time() < deadline:
        a = client.get(f"/api/analyses/{aid}").json()
        if a["state"] == "running" and a.get("live"):
            seen_live = True
            assert a["live"]["frames_total"] == 500
        if a["state"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert a["state"] == "done", a.get("error")
    assert seen_live
    assert a["status"] == "compliant" and a["uniformed"] == 1
    assert a["people"] == 4                         # tracked: 4 people, not 4 x 100 frames
    assert a["frames_processed"] == 100             # 25 fps / 5 analysed fps
    assert a["longest_presence_sec"] >= 15          # 2 s shirt swap bridged by tracking history
    assert client.get(a["files"]["annotated"]).headers["content-type"] == "video/mp4"
    assert client.get(a["files"]["keyframe"]).status_code == 200
    uniform = [p for p in a["persons"] if p["label"] == "uniform"]
    assert len(uniform) == 1 and uniform[0]["uniform_votes"] > uniform[0]["non_uniform_votes"]


def test_alert_state_and_analytics(client):
    alerts = client.get("/api/alerts").json()
    assert alerts["total"] >= 1
    aid = alerts["alerts"][0]["id"]
    assert client.get(f"/api/alerts/{aid}/evidence.jpg").status_code == 200
    assert client.post(f"/api/alerts/{aid}/state", json={"state": "acknowledged"}).status_code == 200
    an = client.get("/api/analytics").json()
    assert an["analyses"] >= 5 and an["compliant"] >= 2 and 0 < an["compliance_rate"] < 1
    ov = client.get("/api/overview").json()
    assert ov["latest"] is not None and len(ov["recent"]) >= 1


def test_settings_validation(client):
    assert client.put("/api/settings", json={"uniform_confidence": 1.5}).status_code == 422
    r = client.put("/api/settings", json={"uniform_confidence": 0.8})
    assert r.status_code == 200 and r.json()["settings"]["uniform_confidence"] == 0.8
    client.post("/api/settings/reset")
