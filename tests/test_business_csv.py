"""Business-facing uniform compliance CSV (format of "Sample CCTV CSV.xlsx") and the per-screenshot
remarks: six columns in order, one row per sampled screenshot, values from the stored model output."""
import csv

import cv2

from backend.extraction import business_csv, detect
from backend.extraction.business_csv import BUSINESS_FIELDS, confidence_text
from backend.extraction.runner import EXTRACTIONS
from backend.extraction.screenshots import EXTRACTION_FAILED, NOT_ANALYSED, remark
from tests.test_compliance_reports import Person, csv_rows, run_job
from tests.test_extraction import MEDIA
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)

HEADER = ["Store Name", "Date", "Time Stamp", "img name", "confidence score", "remark (Example)"]


def raw_csv(client, url, **params):
    r = client.get(url, params=params)
    assert r.status_code == 200, r.text
    return r, list(csv.reader(r.text.splitlines()))


# ------------------------------------------------------------------ pure rules
def test_remarks_follow_the_model_output():
    assert remark(["uniform_shirt"], "UNIFORM_DETECTED", "ok") == "Person 1 detected wearing the designated uniform shirt."
    assert remark(["not_uniform"], "NO_UNIFORM_DETECTED", "ok") == "Person 1 detected without the designated uniform shirt."
    assert remark(["uncertain"], "REVIEW_REQUIRED", "ok") == "Person 1 requires manual review."
    assert remark(["uncertain"], "REVIEW_REQUIRED", "ok", ["occluded by another person"]) == \
        "Person 1 requires manual review (occluded by another person)."
    assert remark(["uniform_shirt", "uncertain"], "UNIFORM_DETECTED", "ok") == \
        "Person 1 detected wearing the designated uniform shirt; Person 2 requires manual review."
    assert remark([], "NO_UNIFORM_DETECTED", "ok") == "No person detected in this screenshot."
    assert remark([], "PROCESSING_FAILED", "failed") == "Uniform status could not be determined because processing failed."
    assert "could not be extracted" in remark([], "", EXTRACTION_FAILED)
    assert remark([], "", NOT_ANALYSED).startswith("Not analysed yet")
    # never a non-compliance claim for a failure or an uncertain person
    for r in (remark([], "PROCESSING_FAILED", "failed"), remark(["uncertain"], "REVIEW_REQUIRED", "ok")):
        assert "without" not in r


def test_confidence_format():
    assert confidence_text([0.925]) == "92.5%"
    assert confidence_text([0.925, 0.41]) == "Person 1: 92.5%; Person 2: 41.0%"
    assert confidence_text([0.9, None]) == "Person 1: 90.0%; Person 2: not scored"
    assert confidence_text([]) == confidence_text([None]) == ""
    assert BUSINESS_FIELDS == HEADER


# ------------------------------------------------------------------ end to end
def test_business_csv_four_screenshots(client, real_model, monkeypatch):  # noqa: F811
    calls = {"n": 0}

    def fake(img, models, cfg, threshold):          # deterministic model output per screenshot (time order)
        calls["n"] += 1
        if calls["n"] == 1:
            return [Person((100, 80, 220, 460), "uniform", 0.925, det_conf=0.97)]
        if calls["n"] == 2:                          # several people: uniform, not uniform, could not be judged
            return [Person((40, 40, 140, 400), "uniform", 0.95), Person((300, 50, 400, 420), "non_uniform", 0.10),
                    Person((600, 60, 650, 140), "uncertain", 0.55, "too small / far")]
        return []                                    # nobody in the screenshot
    imread = cv2.imread
    monkeypatch.setattr(detect, "classify_frame", fake)
    monkeypatch.setattr(detect.cv2, "imread", lambda p, *a: None if "_r0004_" in str(p) else imread(p, *a))
    j, run = run_job(client, {"random_seed": 21, "store_id": "store id 21", "camera_id": "CAM-BIZ",
                              "recording_start_time": "2026-10-01T01:00:00"})
    assert j["store_id"] == "store id 21"                              # kept exactly as typed
    assert j["extracted"] == 4 and run["state"] == "done" and run["processed"] == 4
    man = csv_rows(client, j["manifest_url"])

    # saved against the job, in the run's own folder, with the job-specific name
    saved = EXTRACTIONS / j["id"] / "detections" / run["id"] / f"uniform_compliance_job_{j['id']}.csv"
    assert saved.exists()
    r, table = raw_csv(client, f"/api/extractions/{j['id']}/uniform_compliance.csv", download="true")
    assert f'filename="uniform_compliance_job_{j["id"]}.csv"' in r.headers["content-disposition"]
    assert r.content == saved.read_bytes()
    header, *rows = table
    assert header == HEADER and len(rows) == 4 and all(len(x) == 6 for x in rows)
    rows = [dict(zip(header, x)) for x in rows]
    for row, m in zip(rows, man):                                       # one row per screenshot, in time order
        assert row["Store Name"] == "store id 21"
        assert row["Date"] == "2026-10-01"                              # recording date, not the processing date
        assert row["Time Stamp"] == m["video_timestamp_hhmmss"][:8] and len(row["Time Stamp"]) == 8
        assert row["img name"] == m["image_filename"] and row["img name"].endswith(".png")
    a, b, c, d = rows
    # scores are the classifier's P(uniform) - not the person detector's 0.97 / 0.9
    assert (a["confidence score"], a["remark (Example)"]) == ("92.5%", "Person 1 detected wearing the designated uniform shirt.")
    assert b["confidence score"] == "Person 1: 95.0%; Person 2: 10.0%; Person 3: 55.0%"
    assert b["remark (Example)"] == ("Person 1 detected wearing the designated uniform shirt; Person 2 detected without the "
                                     "designated uniform shirt; Person 3 requires manual review (too small / far).")
    assert (c["confidence score"], c["remark (Example)"]) == ("", "No person detected in this screenshot.")
    assert (d["confidence score"], d["remark (Example)"]) == \
        ("", "Uniform status could not be determined because processing failed.")
    assert d["img name"] == man[3]["image_filename"]                    # the failed image is still listed

    # dashboard preview: the same four records, linked to their evidence
    p = client.get(f"/api/extractions/{j['id']}/uniform-compliance").json()
    assert (p["job_id"], p["screenshots_sampled"], p["run_id"], p["saved"], p["fields"]) == (j["id"], 4, run["id"], True, HEADER)
    assert [x["row"] for x in p["records"]] == rows
    assert [x["processing_status"] for x in p["records"]] == ["ok", "ok", "ok", "failed"]
    for x in p["records"]:
        assert client.get(x["original_url"]).status_code == 200
    assert all(client.get(x["annotated_url"]).status_code == 200 for x in p["records"][:3])
    assert p["records"][3]["annotated_url"] is None
    # the Results dashboard shows the remark for every analysed screenshot
    frames = client.get("/api/frame-results", params={"camera_id": "CAM-BIZ"}).json()["images"]
    assert [f["remark"] for f in frames] == [x["remark (Example)"] for x in rows]

    # technical exports unchanged: still one row per person / per screenshot
    assert len(csv_rows(client, run["screenshot_summary_url"])) == 4
    assert len(csv_rows(client, run["compliance_details_url"])) == 1 + 3 + 1     # 4 people + 1 failure row
    assert len(csv_rows(client, run["results_csv_url"])) == 1 + 3 + 1 + 1

    # an older run without the saved snapshot: generated on request, same content
    saved.unlink()
    assert client.get(f"/api/extractions/{j['id']}/uniform_compliance.csv").content == r.content
    # a new job never overwrites the previous one
    j2, run2 = run_job(client, {"random_seed": 22, "store_id": "store id 21", "camera_id": "CAM-BIZ"})
    assert j2["id"] != j["id"]
    assert (EXTRACTIONS / j2["id"] / "detections" / run2["id"] / business_csv.file_name(j2["id"])).exists()
    assert client.get(f"/api/extractions/{j['id']}/uniform-compliance").json()["records"][0]["row"] == a


def test_business_csv_without_run_and_unknown_metadata(client):
    """Suite default: the trained model is missing -> nothing analysed; nothing is invented."""
    j, run = run_job(client, {"random_seed": 7, "camera_id": "CAM-BIZ-NORUN"})
    assert run["state"] == "failed"
    _, (header, *rows) = raw_csv(client, f"/api/extractions/{j['id']}/uniform_compliance.csv")
    assert header == HEADER and len(rows) == 4
    for row in (dict(zip(header, x)) for x in rows):
        assert (row["Store Name"], row["Date"], row["confidence score"]) == ("", "", "")
        # the automatic run was refused (weights missing): said so, never "non-compliant", never a score
        assert row["img name"] and "detection run failed" in row["remark (Example)"]
    p = client.get(f"/api/extractions/{j['id']}/uniform-compliance").json()
    assert p["run_id"] is None and p["saved"] is False and len(p["records"]) == 4


def test_store_name_validation(client):
    m = client.post("/api/media", files={"file": ("dvr_grid.mp4", (MEDIA / "dvr_grid.mp4").read_bytes())}).json()
    r = client.post("/api/extractions", json={"media_id": m["id"], "store_id": "store<21>", "random_seed": 1})
    assert r.status_code == 422 and "store_id" in r.json()["detail"]
    r = client.post("/api/extractions", json={"media_id": m["id"], "camera_id": "cam 1", "random_seed": 1})
    assert r.status_code == 422                                          # camera ids keep the strict format
