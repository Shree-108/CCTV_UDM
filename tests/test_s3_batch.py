"""S3 cart batches (POST /api/s3/process) against an in-memory S3 stand-in: which keys are read, what
is written to output_data, and that missing credentials / failed analyses are reported, never hidden."""
import io
from datetime import datetime, time as clock
import time

import openpyxl
import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from backend.core import s3
from backend.extraction import cart_report, detect
from tests.test_compliance_reports import Person
from tests.test_extraction import MEDIA
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)

HEADER = ["cart_id", "Store Name ", "Date ", "Time Stamp ", "img name ", "confidence score ", "remark (Example)"]
BUCKET = "yk-cctv-prediction"
DAY = "10-2026/30-10-2026"


class FakeS3:
    def __init__(self, objects: dict[str, bytes], credentials: bool = True, bucket_exists: bool = True):
        self.objects, self.credentials, self.bucket_exists = dict(objects), credentials, bucket_exists
        self.content_types: dict[str, str] = {}

    def head_bucket(self, Bucket):
        if not self.credentials:
            raise NoCredentialsError()
        if not self.bucket_exists:
            raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class Pager:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in fake.objects if k.startswith(Prefix)]}
        return Pager()

    def download_file(self, bucket, key, path):
        with open(path, "wb") as f:
            f.write(self.objects[key])

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        with open(path, "rb") as f:
            self.objects[key] = f.read()
        self.content_types[key] = (ExtraArgs or {}).get("ContentType")

    def keys(self, prefix):
        return sorted(k for k in self.objects if k.startswith(prefix))


@pytest.fixture()
def fake_s3(client, monkeypatch):
    def install(fake):
        monkeypatch.setattr(client.app.state.app_state.s3_batches, "client_factory", lambda: fake)
        return fake
    return install


def wait_batch(client, bid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        b = client.get(f"/api/s3/process/{bid}").json()
        if b["state"] not in ("queued", "running"):
            return b
        time.sleep(0.25)
    raise AssertionError("batch did not finish")


def test_paths_and_validation(client, fake_s3):
    from datetime import date
    d = s3.parse_day("30-10-2026")
    assert s3.input_prefix(d, "cart_78") == "10-2026/30-10-2026/cart_78/input_data/"
    assert s3.output_prefix(d, "cart_78") == "10-2026/30-10-2026/cart_78/output_data/"
    assert d == date(2026, 10, 30) and s3.cart_name("78") == "cart_78" and s3.cart_name("cart_46") == "cart_46"
    fake_s3(FakeS3({}))
    assert client.post("/api/s3/process", json={"date": "2026-10-30", "carts": ["cart_78"]}).status_code == 422
    assert client.post("/api/s3/process", json={"date": "30-10-2026", "carts": ["../x"]}).status_code == 422
    assert client.post("/api/s3/process", json={"date": "30-10-2026", "carts": []}).status_code == 422


def test_missing_credentials_are_reported(client, fake_s3):
    fake_s3(FakeS3({}, credentials=False))
    st = client.get("/api/s3/status").json()
    assert st["ok"] is False and st["credentials"] is False and "AWS_ACCESS_KEY_ID" in st["message"]
    r = client.post("/api/s3/process", json={"date": "30-10-2026", "carts": ["cart_78"]})
    assert r.status_code == 503 and "No AWS credentials" in r.json()["detail"]
    fake_s3(FakeS3({}, bucket_exists=False))
    r = client.post("/api/s3/process", json={"date": "30-10-2026", "carts": ["cart_78"]})
    assert r.status_code == 503 and "does not exist" in r.json()["detail"]


def _rec(types, scores=None, status="ok"):
    scores = scores or [None] * len(types)
    return {"processing_status": status, "people": [{"type": t, "confidence": c} for t, c in zip(types, scores)]}


def test_report_wording_and_score():
    u, n, q = "uniform_shirt", "not_uniform", "uncertain"
    assert cart_report.remark(_rec([u])) == "person1 wearing the dress"
    assert cart_report.remark(_rec([n])) == "person1 not wearing the dress"
    assert cart_report.remark(_rec([])) == "no person detected in frame"
    assert cart_report.remark(_rec([u, u])) == "both persons are wearing the dress"
    assert cart_report.remark(_rec([u, u, u])) == "all 3 persons are wearing the dress"
    assert cart_report.remark(_rec([u, n, u])) == "person2 is not wearing the dress, others wearing"
    assert cart_report.remark(_rec([n, n, u])) == "person1 and person2 are not wearing the dress, others wearing"
    assert cart_report.remark(_rec([u, q])) == "person2 needs manual review, others wearing"   # never "not wearing"
    assert cart_report.remark(_rec([], status="failed")).startswith("uniform status could not be determined")
    assert cart_report.score(_rec([u, n], [0.9, 0.1])) == 50
    assert cart_report.score(_rec([])) == 0 and cart_report.score(_rec([], status="detection_failed")) is None
    assert cart_report.recording_start("20261029_20261029183009_20261029183120_183009.mp4") == "2026-10-29T18:30:09"
    assert cart_report.recording_start("video1.mp4") is None
    assert cart_report.cart_id("cart_50") == 50 and cart_report.store_name("cart_50") == "IOCL Khandwa naka"


def sheet(fake, key):
    ws = openpyxl.load_workbook(io.BytesIO(fake.objects[key])).active
    return [[c.value for c in r] for r in ws.iter_rows()]


def test_cart_batch_end_to_end(client, real_model, fake_s3, monkeypatch):  # noqa: F811
    monkeypatch.setattr(detect, "classify_frame", lambda *a: [Person((100, 80, 220, 460), "uniform", 0.9),
                                                              Person((300, 50, 400, 420), "non_uniform", 0.1)])
    video = (MEDIA / "dvr_grid.mp4").read_bytes()
    timed = "20261029_20261029183009_20261029183120_183009.mp4"
    fake = fake_s3(FakeS3({f"{DAY}/cart_50/input_data/": b"",                    # folder placeholder
                           f"{DAY}/cart_50/input_data/video1.mp4": video,          # no recording time in the name
                           f"{DAY}/cart_50/input_data/{timed}": video,
                           f"{DAY}/cart_50/input_data/notes.txt": b"hello",
                           f"{DAY}/cart_46/input_data/cam1.mp4": video}))          # a cart that was not asked for
    r = client.post("/api/s3/process", json={"date": "30-10-2026", "carts": ["50", "cart_78"]})
    assert r.status_code == 202, r.text
    b = wait_batch(client, r.json()["id"])
    assert b["state"] == "done" and [c["cart"] for c in b["carts"]] == ["cart_50", "cart_78"]
    c50, c78 = b["carts"]
    out = f"{DAY}/cart_50/output_data/"
    assert c50["state"] == "done" and c50["report"] == f"s3://{BUCKET}/{out}cart_50.xlsx"
    assert c50["summary"] == {"files": 3, "analysed": 2, "failed": 0, "skipped": 1, "screenshots": 8}
    assert c78["state"] == "failed" and "No files found" in c78["error"]
    # ONE file per cart, nothing for carts that were not asked for
    assert fake.keys(out) == [f"{out}cart_50.xlsx"]
    assert not fake.keys(f"{DAY}/cart_46/output_data/") and not fake.keys(f"{DAY}/cart_78/output_data/")

    header, *rows = sheet(fake, f"{out}cart_50.xlsx")
    assert header == HEADER and len(rows) == 8
    timed_rows, untimed = rows[:4], rows[4:]                  # known recording times first, in time order
    for i, (cid, store, day, t, img, score, remark) in enumerate(rows, 1):
        assert (cid, store, img, score) == (50, "IOCL Khandwa naka", f"pic{i}", 50)
        assert remark == f"{i}. person2 is not wearing the dress, others wearing"
    for row in timed_rows:                                    # date and clock time from the file name
        assert row[2] == datetime(2026, 10, 29) and isinstance(row[3], clock) and row[3] >= clock(18, 30, 9)
    assert [r[3] for r in timed_rows] == sorted(r[3] for r in timed_rows)
    for row in untimed:                                       # unknown recording time: never invented
        assert row[2] is None and row[3].startswith("video 00:")
    ext = client.get(f"/api/extractions/{c50['files'][0]['extraction_id']}").json()
    assert ext["source"] == "s3" and ext["store_id"] == "cart_50"


def test_failed_analysis_is_not_non_uniform(client, fake_s3):
    """Without trained weights detection is refused: the screenshots are listed with a blank score."""
    fake = fake_s3(FakeS3({f"{DAY}/cart_9/input_data/cam1.mp4": (MEDIA / "dvr_grid.mp4").read_bytes()}))
    b = wait_batch(client, client.post("/api/s3/process", json={"date": "30-10-2026", "carts": ["cart_9"]}).json()["id"])
    (c,) = b["carts"]
    assert c["files"][0]["state"] == "failed" and "weights not found" in c["files"][0]["error"]
    assert fake.keys(f"{DAY}/cart_9/output_data/") == [f"{DAY}/cart_9/output_data/cart_9.xlsx"]
    header, *rows = sheet(fake, f"{DAY}/cart_9/output_data/cart_9.xlsx")
    assert rows and all(r[1] is None and r[5] is None and "could not be determined" in r[6] for r in rows)


def test_test_runs_upload_next_to_output_data_never_over_it(client, real_model, fake_s3, monkeypatch):  # noqa: F811
    monkeypatch.setattr(detect, "classify_frame", lambda *a: [Person((100, 80, 220, 460), "uniform", 0.9)])
    prod = f"{DAY}/cart_50/output_data/cart_50.xlsx"
    fake = fake_s3(FakeS3({f"{DAY}/cart_50/input_data/video1.mp4": (MEDIA / "dvr_grid.mp4").read_bytes(),
                           prod: b"the existing report"}))
    for bad in ("output_data", "input_data", "../output_data", "a/b", ""):
        r = client.post("/api/s3/process", json={"date": "30-10-2026", "carts": ["50"], "output_folder": bad})
        assert r.status_code == 422, bad
    r = client.post("/api/s3/process", json={"date": "30-10-2026", "carts": ["50"], "output_folder": "output_data_test"})
    assert r.status_code == 202 and r.json()["output_folder"] == "output_data_test"
    b = wait_batch(client, r.json()["id"])
    (c,) = b["carts"]
    assert c["state"] == "done" and c["report"] == f"s3://{BUCKET}/{DAY}/cart_50/output_data_test/cart_50.xlsx"
    assert fake.objects[prod] == b"the existing report"                       # production report untouched
    assert fake.keys(f"{DAY}/cart_50/output_data_test/") == [f"{DAY}/cart_50/output_data_test/cart_50.xlsx"]
    assert len(sheet(fake, f"{DAY}/cart_50/output_data_test/cart_50.xlsx")) == 5   # header + 4 screenshots
