"""screenshot_summary.csv and uniform_compliance_details.csv: four sampled screenshots, one row per
person, failures reported as failures, unknown metadata left blank."""
import csv
import time

import pytest

from backend.extraction import detect
from backend.extraction.reports import DETAIL_FIELDS, SUMMARY_FIELDS
from tests.test_extraction import MEDIA, wait
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)


class Person:
    def __init__(self, box, label, p_uniform, reason="", det_conf=0.9):
        self.box, self.label, self.reason, self.det_conf = box, label, reason, det_conf
        self.conf = p_uniform if label == "uniform" else 1 - p_uniform
        self.extra = {"uniform_prob": p_uniform}


def csv_rows(client, url, **params):
    r = client.get(url, params=params)
    assert r.status_code == 200, r.text
    return list(csv.DictReader(r.text.splitlines()))


def run_job(client, body, timeout=120):
    m = client.post("/api/media", files={"file": ("dvr_grid.mp4", (MEDIA / "dvr_grid.mp4").read_bytes())}).json()
    j = wait(client, client.post("/api/extractions", json={"media_id": m["id"], "run_detection": True, "new_run": True,
                                                          **body}).json()["id"])
    t0 = time.time()
    while time.time() - t0 < timeout:
        runs = client.get(f"/api/extractions/{j['id']}/detections").json()["runs"]
        if runs and runs[0]["state"] not in ("queued", "running"):
            return j, runs[0]
        time.sleep(0.25)
    raise AssertionError("detection did not finish")


def test_reports_from_four_screenshots(client, real_model, monkeypatch):  # noqa: F811
    calls = {"n": 0}

    def fake(img, models, cfg, threshold):         # deterministic model output per screenshot (in time order)
        calls["n"] += 1
        if calls["n"] == 1:                         # three people: uniform, not uniform, could not be judged
            return [Person((40, 40, 140, 400), "uniform", 0.95), Person((300, 50, 400, 420), "non_uniform", 0.10),
                    Person((600, 60, 650, 140), "uncertain", 0.55, "too small / far")]
        if calls["n"] == 2:
            raise RuntimeError("simulated inference crash")
        if calls["n"] == 3:
            return []                               # nobody in the screenshot
        return [Person((100, 80, 220, 460), "uniform", 0.81)]
    monkeypatch.setattr(detect, "classify_frame", fake)
    j, run = run_job(client, {"random_seed": 99, "store_id": "STORE-RPT", "camera_id": "CAM-RPT"})
    assert j["extracted"] == 4 and run["state"] == "done" and run["processed"] == 4

    man = {m["image_id"]: m for m in csv_rows(client, j["manifest_url"])}
    summ = csv_rows(client, run["screenshot_summary_url"])
    assert list(summ[0].keys()) == SUMMARY_FIELDS
    # 1. one row per sampled screenshot, linked to the right image and timestamp
    assert len(summ) == 4 and [s["image_id"] for s in summ] == list(man)
    for s in summ:
        m = man[s["image_id"]]
        assert (s["video_timestamp_seconds"], s["video_timestamp_hhmmss"], s["screenshot_filename"], s["original_image_path"]) == \
            (m["video_timestamp_seconds"], m["video_timestamp_hhmmss"], m["image_filename"], m["original_image_path"])
        assert s["detection_run_id"] == run["id"] and s["extraction_job_id"] == j["id"]
        # 4. supplied ids kept; unknown recording date stays blank and is marked unknown; no S3
        assert (s["store_id"], s["camera_id"], s["recording_date"], s["recording_timestamp"], s["recording_date_status"],
                s["original_s3_key"]) == ("STORE-RPT", "CAM-RPT", "", "", "unknown", "")
    a, b, c, d = summ
    assert (a["number_of_people_detected"], a["number_of_people_classified"], a["number_of_uniform_compliant"],
            a["number_of_uniform_non_compliant"], a["number_of_uncertain_predictions"], a["number_of_processing_failures"],
            a["overall_image_status"]) == ("3", "2", "1", "1", "1", "0", "UNIFORM_DETECTED")
    # 5. failed frame: a failure, not non-compliance, and no invented counts
    assert (b["overall_image_status"], b["number_of_processing_failures"], b["number_of_people_detected"],
            b["number_of_uniform_non_compliant"]) == ("PROCESSING_FAILED", "1", "", "")
    assert "simulated inference crash" in b["review_reason"]
    assert (c["number_of_people_detected"], c["overall_image_status"]) == ("0", "NO_UNIFORM_DETECTED")
    assert (d["number_of_uniform_compliant"], d["overall_image_status"]) == ("1", "UNIFORM_DETECTED")

    det = csv_rows(client, run["compliance_details_url"])
    assert list(det[0].keys()) == DETAIL_FIELDS
    # 2. one row per person (3 + 1), plus one failure row for the failed screenshot; none for the empty one
    by_img = {}
    for r in det:
        by_img.setdefault(r["image_id"], []).append(r)
    assert [len(by_img.get(s["image_id"], [])) for s in summ] == [3, 1, 0, 1]
    first = by_img[a["image_id"]]
    assert [(r["person_index"], r["uniform_prediction"], r["compliance_status"], r["uniform_confidence_score"])
            for r in first] == [("1", "UNIFORM_DETECTED", "UNIFORM_DETECTED", "0.95"),
                                ("2", "NON_UNIFORM_DETECTED", "NON_UNIFORM_DETECTED", "0.1"),
                                ("3", "UNCERTAIN", "REVIEW_REQUIRED", "0.55")]
    assert {r["person_confidence"] for r in first} == {"0.9"} and {r["confidence_threshold"] for r in det} == {str(run["threshold"])}
    (fail,) = by_img[b["image_id"]]
    assert (fail["compliance_status"], fail["processing_status"], fail["uniform_prediction"], fail["person_index"]) == \
        ("PROCESSING_FAILED", "failed", "", "")
    # 3. every person row points to its screenshot, timestamp, run and annotated evidence
    for r in det:
        s = next(x for x in summ if x["image_id"] == r["image_id"])
        assert (r["video_timestamp_hhmmss"], r["screenshot_filename"], r["detection_run_id"]) == \
            (s["video_timestamp_hhmmss"], s["screenshot_filename"], run["id"])
        assert r["annotated_image_path"] == s["annotated_image_path"] and r["annotated_s3_key"] == ""
        assert r["recording_date"] == "" and r["recording_date_status"] == "unknown"
    assert all(r["annotated_image_path"] for r in det if r["processing_status"] == "ok")

    # the dashboard (combined) export contains exactly the same rows for this camera
    assert csv_rows(client, "/api/frame-results/screenshot_summary.csv", camera_id="CAM-RPT") == summ
    assert csv_rows(client, "/api/frame-results/uniform_compliance_details.csv", camera_id="CAM-RPT") == det
    # 6. the existing exports still work
    for name in ("image_manifest", "detection_results", "processing_failures", "hourly_summary"):
        assert client.get(f"/api/frame-results/{name}.csv", params={"camera_id": "CAM-RPT"}).status_code == 200


def test_known_recording_start_fills_the_date(client, real_model, monkeypatch):  # noqa: F811
    monkeypatch.setattr(detect, "classify_frame", lambda *a: [Person((40, 40, 140, 400), "uniform", 0.9)])
    j, run = run_job(client, {"random_seed": 5, "camera_id": "CAM-DATE", "recording_start_time": "2026-10-25T23:59:50"})
    summ = csv_rows(client, run["screenshot_summary_url"])
    assert len(summ) == 4 and {s["recording_date_status"] for s in summ} == {"known"}
    for s in summ:                                  # clock time = start + video position (crosses midnight)
        assert s["recording_timestamp"][:10] == s["recording_date"] in ("2026-10-25", "2026-10-26")
    assert {s["store_id"] for s in summ} == {""}                     # not supplied -> blank, never a default
    assert csv_rows(client, "/api/frame-results/uniform_compliance_details.csv", camera_id="CAM-DATE", date="2026-10-26") == \
        [r for r in csv_rows(client, run["compliance_details_url"]) if r["recording_date"] == "2026-10-26"]


def test_real_model_one_row_per_person(client, real_model):  # noqa: F811
    """The real two-stage model on the 2x2 DVR test video: several people per screenshot."""
    j, run = run_job(client, {"random_seed": 3})
    assert run["state"] == "done" and run["processed"] == 4
    summ = csv_rows(client, run["screenshot_summary_url"])
    det = csv_rows(client, run["compliance_details_url"])
    assert len(summ) == 4 and max(int(s["number_of_people_detected"]) for s in summ) >= 2
    for s in summ:
        mine = [r for r in det if r["image_id"] == s["image_id"]]
        assert len(mine) == int(s["number_of_people_detected"])           # one row per detected person
        assert int(s["number_of_people_classified"]) + int(s["number_of_uncertain_predictions"]) == len(mine)
        assert sum(r["uniform_prediction"] == "UNIFORM_DETECTED" for r in mine) == int(s["number_of_uniform_compliant"])
        assert [r["person_index"] for r in mine] == [str(i) for i in range(1, len(mine) + 1)]
    print("\n  real model, people per screenshot:", [(s["video_timestamp_hhmmss"], s["number_of_people_detected"],
                                                     s["number_of_uniform_compliant"], s["number_of_uniform_non_compliant"],
                                                     s["number_of_uncertain_predictions"]) for s in summ])


def test_reports_without_a_detection_run(client):
    """Suite default: the trained model is missing -> no run processed; screenshots are NOT_ANALYSED."""
    j, run = run_job(client, {"random_seed": 1, "camera_id": "CAM-NORUN"})
    assert run["state"] == "failed"
    summ = csv_rows(client, "/api/frame-results/screenshot_summary.csv", camera_id="CAM-NORUN")
    assert len(summ) == 4 and {s["overall_image_status"] for s in summ} == {"NOT_ANALYSED"}
    assert {s["number_of_people_detected"] for s in summ} == {""}
    assert csv_rows(client, "/api/frame-results/uniform_compliance_details.csv", camera_id="CAM-NORUN") == []
    with pytest.raises(AssertionError):
        csv_rows(client, f"/api/detection-runs/{run['id']}/unknown_report.csv")
