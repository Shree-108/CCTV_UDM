"""S3 cart batches (POST /api/s3/process) against an in-memory S3 stand-in: which keys are read, what
is written to output_data, and that missing credentials / failed analyses are reported, never hidden."""
import csv
import io
import json
import time

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from backend.core import s3
from backend.extraction import detect
from tests.test_compliance_reports import Person
from tests.test_extraction import MEDIA
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)

HEADER = ["Store Name", "Date", "Time Stamp", "img name", "confidence score", "remark (Example)"]
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


def test_cart_batch_end_to_end(client, real_model, fake_s3, monkeypatch):  # noqa: F811
    monkeypatch.setattr(detect, "classify_frame", lambda *a: [Person((100, 80, 220, 460), "uniform", 0.925)])
    video = (MEDIA / "dvr_grid.mp4").read_bytes()
    fake = fake_s3(FakeS3({f"{DAY}/cart_78/input_data/": b"",                    # folder placeholder
                           f"{DAY}/cart_78/input_data/cam1.mp4": video,
                           f"{DAY}/cart_78/input_data/notes.txt": b"hello",
                           f"{DAY}/cart_46/input_data/cam1.mp4": video}))          # a cart that was not asked for
    r = client.post("/api/s3/process", json={"date": "30-10-2026", "carts": ["78", "cart_50"]})
    assert r.status_code == 202, r.text
    b = wait_batch(client, r.json()["id"])
    assert b["state"] == "done" and [c["cart"] for c in b["carts"]] == ["cart_78", "cart_50"]
    c78, c50 = b["carts"]
    assert c78["state"] == "done" and c78["output"] == f"s3://{BUCKET}/{DAY}/cart_78/output_data/"
    assert c78["summary"] == {"files": 2, "analysed": 1, "failed": 0, "skipped": 1}
    vid, txt = sorted(c78["files"], key=lambda f: f["name"])
    assert (vid["name"], vid["state"], txt["state"]) == ("cam1.mp4", "done", "skipped")
    assert c50["state"] == "failed" and "No files found" in c50["error"]

    out = f"{DAY}/cart_78/output_data/"
    keys = fake.keys(out)
    assert f"{out}cam1/uniform_compliance.csv" in keys and f"{out}cam1/detection_results.csv" in keys
    assert f"{out}uniform_compliance_cart_78_30-10-2026.csv" in keys and f"{out}processing_summary.json" in keys
    annotated = [k for k in keys if k.startswith(f"{out}cam1/annotated/")]
    assert annotated and len(annotated) == vid["screenshots"]
    assert fake.content_types[f"{out}cam1/uniform_compliance.csv"] == "text/csv"
    assert not fake.keys(f"{DAY}/cart_46/output_data/") and not fake.keys(f"{DAY}/cart_50/output_data/")

    rows = list(csv.reader(io.StringIO(fake.objects[f"{out}uniform_compliance_cart_78_30-10-2026.csv"].decode())))
    assert rows[0] == HEADER and len(rows) == 1 + vid["screenshots"]
    for row in (dict(zip(HEADER, x)) for x in rows[1:]):
        assert row["Store Name"] == "cart_78" and row["Date"] == ""          # recording time unknown: never invented
        assert row["confidence score"] == "92.5%"
    assert json.loads(fake.objects[f"{out}processing_summary.json"])["cart"] == "cart_78"
    ext = client.get(f"/api/extractions/{vid['extraction_id']}").json()
    assert ext["source"] == "s3" and ext["store_id"] == "cart_78"


def test_failed_analysis_uploads_no_results(client, fake_s3):
    """Without trained weights detection is refused: the video is failed, nothing claims non-uniform."""
    fake = fake_s3(FakeS3({f"{DAY}/cart_9/input_data/cam1.mp4": (MEDIA / "dvr_grid.mp4").read_bytes()}))
    b = wait_batch(client, client.post("/api/s3/process", json={"date": "30-10-2026", "carts": ["cart_9"]}).json()["id"])
    (c,) = b["carts"]
    assert c["state"] == "failed" and c["files"][0]["state"] == "failed"
    assert "weights not found" in c["files"][0]["error"]
    assert fake.keys(f"{DAY}/cart_9/output_data/") == [f"{DAY}/cart_9/output_data/processing_summary.json"]
