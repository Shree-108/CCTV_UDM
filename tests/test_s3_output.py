"""Every analysed video's CSV rows and detection images go to S3 (backend/extraction/s3_output.py): the Video
Analysis / Results Dashboard path (POST /api/extractions) against an in-memory S3 stand-in.

    s3://<bucket>/<MM-YYYY>/<DD-MM-YYYY>/card_50/output_data/card50_YYYY_MM_DD.csv          (today, Asia/Kolkata)
    s3://<bucket>/<MM-YYYY>/<DD-MM-YYYY>/card_50/output_data/card50_YYYY_MM_DD_pic<K>.jpg

The day's first run creates the CSV and later runs append (one header, earlier rows intact, no duplicates on
retries); numbering continues after the highest existing pic<K>; nothing is replaced or deleted; no cart50/."""
import csv
import io
import time
from datetime import date, datetime

import pytest

from backend.core import s3
from backend.extraction import detect, master_csv
from tests.test_compliance_reports import Person
from tests.test_extraction import MEDIA
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)
from tests.test_s3_batch import BUCKET, HEADER, FakeS3

NVR = "20261104_20261104215126_20261104215614_192126.mp4"     # recording 04-11-2026 21:51:26
TODAY = s3.today()                                            # the processing day decides the folder
OUT = f"{TODAY:%m-%Y}/{TODAY:%d-%m-%Y}/card_50/output_data/"
STEM = f"card50_{TODAY:%Y_%m_%d}"
KEY = f"{OUT}{STEM}.csv"


def image(i: int, suffix: str = "") -> str:
    return f"{OUT}{STEM}_pic{i}{suffix}.jpg"


def images(fake):
    return [k for k in fake.keys(OUT) if k.endswith(".jpg")]


def rows_of(data: bytes) -> tuple[list[str], list[list[str]]]:
    header, *rows = list(csv.reader(io.StringIO(data.decode())))
    return header, rows


@pytest.fixture()
def fake(client, monkeypatch):
    f = FakeS3({})
    monkeypatch.setattr(client.app.state.app_state.s3_batches, "client_factory", lambda: f)
    return f


@pytest.fixture(autouse=True)
def model_and_master(client, real_model, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(master_csv, "EXPORTS", tmp_path / "exports")
    monkeypatch.setattr(master_csv, "_last_error", None)
    monkeypatch.setattr(detect, "classify_frame", lambda *a: [Person((100, 80, 220, 460), "uniform", 0.9)])


def start(client, name=NVR, **body):
    m = client.post("/api/media", files={"file": (name, (MEDIA / "dvr_grid.mp4").read_bytes())})
    assert m.status_code == 200, m.text
    r = client.post("/api/extractions", json={"media_id": m.json()["id"], "run_detection": True, "new_run": True,
                                              "sampling_strategy": "random_segments", "sample_count": 4, **body})
    assert r.status_code == 202, r.text
    return r.json()["id"]


def finished(client, eid, timeout=120):
    """The job once its status is no longer "processing" (analysis AND the S3 upload have ended)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/extractions/{eid}").json()
        if j["status"] != "processing":
            return j
        time.sleep(0.25)
    raise AssertionError(f"{eid} did not finish")


def test_layout_and_processing_day():
    d = date(2026, 11, 8)
    assert s3.output_prefix(d, "cart_50") == "11-2026/08-11-2026/card_50/output_data/"
    assert s3.daily_csv_name(d, "cart_50") == "card50_2026_11_08.csv"
    assert s3.daily_csv_key(d, "cart_50") == "11-2026/08-11-2026/card_50/output_data/card50_2026_11_08.csv"
    assert s3.today() == datetime.now(s3.IST).date() and s3.IST.utcoffset(None).total_seconds() == 5.5 * 3600
    assert s3.highest_pic(["x/card50_2026_11_08_pic12.jpg", "card50_2026_11_08_pic3_det-1.jpg", "a.csv"]) == 12
    assert s3.highest_pic([["50", "s", "", "t", "pic7", "", "7. person1 ..."]]) == 7


def test_client_uses_only_env_credentials(monkeypatch):
    """No keys in the environment (.env): no client - ~/.aws files or an instance role are never used."""
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(s3.S3Unavailable, match="AWS_ACCESS_KEY_ID"):
        s3.client()


def test_run1_creates_the_daily_csv_and_images(client, fake):
    eid = start(client, card_id=50)
    j = finished(client, eid)
    o = j["s3_output"]
    run = client.get(f"/api/extractions/{eid}/detections").json()["runs"][0]
    assert j["status"] == "completed" and o["state"] == "completed" and o["verified"] is True
    assert o["key"] == KEY and o["s3_uri"] == f"s3://{BUCKET}/{KEY}" and fake.puts[0] == KEY
    # filed under TODAY, although the file name says the video was recorded on 04-11-2026; no cart50/
    assert (o["cart"], o["day"], o["day_source"], o["folder"]) == (
        "cart_50", f"{TODAY:%d-%m-%Y}", "processing_date", "output_data")
    assert set(fake.objects) == {KEY, *[image(i) for i in range(1, 5)]} and not fake.keys("cart50")
    assert fake.content_types[KEY] == "text/csv" and o["size_bytes"] == len(fake.objects[KEY]) > 0
    assert "report_path" not in o and run["s3_output"] == o                        # no local paths in the API
    header, rows = rows_of(fake.objects[KEY])
    assert header == HEADER and len(rows) == 4                                     # the job's actual results
    assert all(r[0] == "50" and r[1] == "IOCL Khandwa naka" and r[2] == "" and r[3].startswith("video 00:")
               and r[5] == "90" and "person1 wearing the dress" in r[6] for r in rows)
    assert [r[4] for r in rows] == ["pic1", "pic2", "pic3", "pic4"]
    assert images(fake) == [image(i) for i in range(1, 5)] and o["images_uploaded"] == 4
    assert all(fake.content_types[k] == "image/jpeg" and fake.objects[k][:2] == b"\xff\xd8" for k in images(fake))
    assert o["csv_first_row"] == 1


def test_run2_same_day_appends_to_the_same_file(client, fake):
    first = finished(client, start(client, card_id=50))["s3_output"]
    before = fake.objects[KEY]
    first_images = {k: fake.objects[k] for k in images(fake)}
    second = finished(client, start(client, card_id=50))["s3_output"]
    assert second["state"] == "completed" and second["key"] == first["key"] == KEY
    after = fake.objects[KEY]
    assert after.startswith(before)                                                # old rows intact
    header, rows = rows_of(after)
    assert header == HEADER and after.decode().count("cart_id,") == 1            # a single header
    assert len(rows) == 8 and len({tuple(r) for r in rows}) == 8                   # no duplicate rows
    assert [r[4] for r in rows] == [f"pic{i}" for i in range(1, 9)]               # numbering continues
    assert second["csv_first_row"] == 5 and second["size_bytes"] == len(after)
    assert images(fake) == sorted(image(i) for i in range(1, 9))
    assert all(fake.objects[k] == v for k, v in first_images.items())             # earlier images untouched


HISTORY = ("cart_id,Store Name ,Date ,Time Stamp ,img name ,confidence score ,remark (Example)\r\n"
           '50,IOCL Khandwa naka,04-11-2026,08:00:00,pic1,75,"1. person1 wearing the dress, earlier hour"\r\n'
           "50,IOCL Khandwa naka,04-11-2026,08:00:30,pic2,,2. frame unreadable\r\n").encode()


def test_existing_csv_and_images_continue_from_the_last_pic(client, fake):
    """Today's folder already holds the CSV and pic1-pic8 (earlier runs): the new rows and images start at pic9,
    and nothing that was there changes."""
    rows8 = HISTORY + b"".join(f"50,IOCL Khandwa naka,,video,pic{i},100,{i}. ok\r\n".encode() for i in range(3, 9))
    old = {KEY: rows8, **{image(i): f"old picture {i}".encode() for i in range(1, 9)}}
    fake.objects.update(old)
    o = finished(client, start(client, card_id=50))["s3_output"]
    assert o["state"] == "completed" and o["csv_first_row"] == 9 and o["images_uploaded"] == 4
    assert fake.objects[KEY].startswith(rows8)
    _, rows = rows_of(fake.objects[KEY])
    assert [r[4] for r in rows] == [f"pic{i}" for i in range(1, 13)] and rows[8][6].startswith("9. ")
    assert all(fake.objects[k] == v for k, v in old.items() if k != KEY)          # no image replaced
    assert images(fake) == sorted(image(i) for i in range(1, 13))


def test_numbering_skips_past_images_with_higher_numbers(client, fake):
    """The CSV has 2 rows but pic1-pic5 images exist (e.g. from an older run): the new rows start at pic6."""
    fake.objects[KEY] = HISTORY
    for i in range(1, 6):
        fake.objects[image(i)] = b"older"
    o = finished(client, start(client, card_id=50))["s3_output"]
    assert o["state"] == "completed" and o["csv_first_row"] == 6
    _, rows = rows_of(fake.objects[KEY])
    assert [r[4] for r in rows] == ["pic1", "pic2", "pic6", "pic7", "pic8", "pic9"]
    assert all(fake.objects[image(i)] == b"older" for i in range(1, 6)) and image(6) in fake.objects


def test_image_name_collision_keeps_the_old_object(tmp_path):
    from backend.extraction import s3_output
    img = tmp_path / "a_annotated.jpg"
    img.write_bytes(b"\xff\xd8 new picture")
    fake = FakeS3({image(3): b"someone else's picture"})
    n = s3_output.upload_images(fake, BUCKET, KEY, [{"img": "pic3", "annotated_image_path": str(img)}], "det-x")
    assert n == 1 and fake.objects[image(3)] == b"someone else's picture"
    assert fake.objects[image(3, "_det-x")] == b"\xff\xd8 new picture"


def test_daily_csv_with_other_columns_is_not_changed(client, fake):
    fake.objects[KEY] = b"a,b\r\n1,2\r\n"
    o = finished(client, start(client, card_id=50))["s3_output"]
    assert o["state"] == "failed" and o["failed_stage"] == "uploading" and "other columns" in o["error"]
    assert fake.objects[KEY] == b"a,b\r\n1,2\r\n" and not fake.puts


def test_concurrent_write_is_merged_not_lost(client, fake):
    """Another writer appends between our read and our write: the conditional PUT fails, the file is read
    again and our rows go after theirs."""
    fake.objects[KEY] = HISTORY
    other = b'50,IOCL Khandwa naka,04-11-2026,09:00:00,pic3,80,3. written meanwhile\r\n'
    real_get, raced = fake.get_object, []

    def get_object(**kw):
        r = real_get(**kw)
        if not raced:                                  # first read: the object then changes under us
            raced.append(1)
            fake.objects[KEY] += other
        return r
    fake.get_object = get_object
    o = finished(client, start(client, card_id=50))["s3_output"]
    assert o["state"] == "completed"
    data = fake.objects[KEY]
    assert data.startswith(HISTORY + other)
    _, rows = rows_of(data)
    assert len(rows) == 7 and [r[4] for r in rows[3:]] == ["pic4", "pic5", "pic6", "pic7"]
    assert images(fake) == [image(i) for i in range(4, 8)]


def test_lost_put_response_does_not_append_twice(tmp_path):
    """The PUT lands but its response is lost: the append is recognised as done, and a later attempt that
    knows the SHA-256 it wrote does not add the rows again - even after another run appended."""
    key = "11-2026/06-11-2026/card_50/output_data/card50_2026_11_06.csv"
    fake = FakeS3({key: HISTORY})
    real_put = fake.put_object

    def put_then_fail(**kw):
        real_put(**kw)
        raise ConnectionError("response lost")
    fake.put_object = put_then_fail

    def new(first):
        return [[50, "IOCL Khandwa naka", "06-11-2026", "10:00:00", f"pic{first}", 90, f"{first}. ok"]]
    size, sha = s3.append_csv(fake, BUCKET, key, HEADER, new, tmp_path / "d.csv")
    ours = HISTORY + b"50,IOCL Khandwa naka,06-11-2026,10:00:00,pic3,90,3. ok\r\n"
    assert fake.objects[key] == ours
    fake.put_object = real_put
    assert s3.append_csv(fake, BUCKET, key, HEADER, new, tmp_path / "d.csv", already=sha) == (size, sha)
    s3.append_csv(fake, BUCKET, key, HEADER, lambda f: [[50, "x", "", "video", f"pic{f}", 1, f"{f}. other"]],
                  tmp_path / "other.csv")
    after_other = fake.objects[key]
    s3.append_csv(fake, BUCKET, key, HEADER, new, tmp_path / "d.csv", already=sha)
    assert fake.objects[key] == after_other and after_other.startswith(ours)
    _, rows = rows_of(after_other)
    assert [r[4] for r in rows] == ["pic1", "pic2", "pic3", "pic4"]


def test_job_failed_under_the_cart50_layout_is_filed_in_card_50_on_retry(client, fake):
    """Jobs that failed while the destination was cart50/... (never created) go to card_50/output_data/."""
    s = client.app.state.app_state
    fake.fail_put = lambda key: True
    j = finished(client, start(client, card_id=50))
    rid = j["s3_output"]["run_id"]
    old = f"cart50/{TODAY:%Y_%m}/{TODAY:%d}/cart50_{TODAY:%Y_%m_%d}.csv"
    s.db.update_s3_output(rid, key=old, folder="cart50", sha256=None, size_bytes=None)
    fake.fail_put = lambda key: False
    assert client.post(f"/api/detection-runs/{rid}/s3-output/retry", json={}).status_code == 202
    o = finished(client, j["id"])["s3_output"]
    assert o["state"] == "completed" and o["key"] == KEY and o["folder"] == "output_data"
    assert not fake.keys("cart50") and len(rows_of(fake.objects[KEY])[1]) == 4


def test_no_extra_output_folders_and_validation(client, fake):
    o = finished(client, start(client, "video1.mp4", card_id=50, s3_output_folder="output_data_test"))["s3_output"]
    assert o["state"] == "completed" and o["key"] == KEY
    assert not [k for k in fake.objects if "output_data_test" in k or k.startswith("cart50")]
    for bad in ({"s3_day": "2026-10-30"}, {"card_id": 4242}):
        m = client.post("/api/media", files={"file": ("v.mp4", (MEDIA / "dvr_grid.mp4").read_bytes())}).json()
        r = client.post("/api/extractions", json={"media_id": m["id"], "run_detection": True, **bad})
        assert r.status_code == 422, bad


def test_unknown_card_fails_visibly_and_retry_with_card(client, fake):
    j = finished(client, start(client, "video1.mp4"))            # no card
    o = j["s3_output"]
    assert j["status"] == "failed" and o["state"] == "failed" and o["failed_stage"] == "generating_report"
    assert "Card ID" in o["error"] and not fake.puts
    rid = o["run_id"]
    assert client.post(f"/api/detection-runs/{rid}/s3-output/retry", json={"card_id": 4242}).status_code == 409
    r = client.post(f"/api/detection-runs/{rid}/s3-output/retry", json={"card_id": 50})
    assert r.status_code == 202, r.text
    j = finished(client, j["id"])
    assert j["status"] == "completed" and j["s3_output"]["key"] == KEY and j["s3_output"]["attempts"] == 2
    assert client.post(f"/api/detection-runs/{rid}/s3-output/retry", json={}).status_code == 409   # not failed


def test_upload_failure_is_not_success_and_retry_adds_rows_once(client, fake):
    fake.fail_put = lambda key: True
    j = finished(client, start(client, card_id=50))
    o = j["s3_output"]
    assert j["status"] == "failed" and o["failed_stage"] == "uploading" and "S3 is down" in o["error"]
    assert not o["verified"] and not fake.objects and o["key"] and o["s3_uri"]   # kept for the retry
    fake.fail_put = lambda key: False
    client.post(f"/api/detection-runs/{o['run_id']}/s3-output/retry", json={})
    o2 = finished(client, j["id"])["s3_output"]
    assert o2["state"] == "completed" and o2["key"] == o["key"] and o2["sha256"] == o["sha256"]
    assert len(rows_of(fake.objects[KEY])[1]) == 4


def test_image_failure_retry_does_not_append_the_rows_again(client, fake):
    fake.fail_put = lambda key: key.endswith(".jpg")
    j = finished(client, start(client, card_id=50))
    o = j["s3_output"]
    assert j["status"] == "failed" and o["failed_stage"] == "uploading_images" and o["csv_first_row"] == 1
    csv_bytes = fake.objects[KEY]
    fake.objects[KEY] += b"50,IOCL Khandwa naka,,video 00:00:01,pic5,80,5. another run meanwhile\r\n"
    fake.fail_put = lambda key: False
    assert client.post(f"/api/detection-runs/{o['run_id']}/s3-output/retry", json={}).status_code == 202
    o2 = finished(client, j["id"])["s3_output"]
    assert o2["state"] == "completed" and o2["images_uploaded"] == 4
    assert fake.objects[KEY].startswith(csv_bytes) and len(rows_of(fake.objects[KEY])[1]) == 5
    assert images(fake) == [image(i) for i in range(1, 5)]


def test_missing_credentials_message(client, monkeypatch):
    from botocore.exceptions import NoCredentialsError

    class NoCreds(FakeS3):
        def put_object(self, **kw):
            raise NoCredentialsError()
    monkeypatch.setattr(client.app.state.app_state.s3_batches, "client_factory", lambda: NoCreds({}))
    o = finished(client, start(client, card_id=50))["s3_output"]
    assert o["state"] == "failed" and "No AWS credentials found" in o["error"] and "AWS_SECRET" in o["error"]


def test_interrupted_upload_resumes_at_start_up(client, fake):
    s = client.app.state.app_state
    j = finished(client, start(client, card_id=50))
    rid = j["s3_output"]["run_id"]
    # as if the server stopped mid-upload, before the CSV was verified
    s.db.update_s3_output(rid, state="uploading", csv_first_row=None, sha256=None, size_bytes=None)
    fake.objects.clear()
    assert s.s3_outputs.resume() >= 1
    assert finished(client, j["id"])["s3_output"]["state"] == "completed" and KEY in fake.objects


def test_admin_uploads_do_not_get_a_second_csv(client):
    """Admin uploads run their own CSV / image stages (s3_upload.py): their analysis jobs have no S3 output."""
    from backend.extraction.runner import clean_s3_destination
    assert clean_s3_destination(False, None, "50", None) == {"s3_output": 0}
