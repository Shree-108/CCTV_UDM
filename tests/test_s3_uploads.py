"""Admin video uploads (POST /api/s3/uploads) against an in-memory S3 stand-in: login, validation, the S3
input / output keys, the job states, failures per stage, retries and duplicate prevention."""
import base64
import csv
import hashlib
import io
import time

import pytest
from botocore.exceptions import ClientError

from backend.core import auth, s3
from backend.extraction import detect, master_csv, s3_upload
from tests.test_compliance_reports import Person
from tests.test_extraction import MEDIA
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)
from tests.test_s3_batch import BUCKET, HEADER, FakeS3

NVR = "20261104_20261104215126_20261104215614_192126.mp4"     # recording 04-11-2026 21:51:26
DAY = "11-2026/04-11-2026"
IN = f"{DAY}/cart_50/input_data/{NVR}"
OUT = f"{DAY}/cart_50/output_data/"
VIDEO = (MEDIA / "dvr_grid.mp4").read_bytes()


class FakeS3Put(FakeS3):
    """Adds conditional PUT with a SHA-256 check and HEAD with checksums, as S3 does them."""
    def __init__(self, objects=None, **kw):
        super().__init__(objects or {}, **kw)
        self.checksums: dict[str, str] = {}
        self.puts: list[str] = []
        self.fail_put = lambda key: False

    def put_object(self, Bucket, Key, Body, ContentType, IfNoneMatch=None, ChecksumSHA256=None, Metadata=None):
        if self.fail_put(Key):
            raise ClientError({"Error": {"Code": "InternalError", "Message": "S3 is down"}}, "PutObject")
        if IfNoneMatch == "*" and Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        data = Body.read()
        if ChecksumSHA256 != base64.b64encode(hashlib.sha256(data).digest()).decode():
            raise ClientError({"Error": {"Code": "BadDigest"}}, "PutObject")
        self.objects[Key], self.checksums[Key], self.content_types[Key] = data, ChecksumSHA256, ContentType
        self.puts.append(Key)

    def head_object(self, Bucket, Key, ChecksumMode=None):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        h = {"ContentLength": len(self.objects[Key])}
        if Key in self.checksums:
            h["ChecksumSHA256"] = self.checksums[Key]
        return h


@pytest.fixture()
def fake(client, monkeypatch):
    f = FakeS3Put()
    monkeypatch.setattr(client.app.state.app_state.s3_batches, "client_factory", lambda: f)
    return f


@pytest.fixture()
def admin(client):
    client.cookies.clear()
    r = client.post("/api/auth/login", json={"username": "admin", "password": "test-password"})
    assert r.status_code == 200, r.text
    yield client
    client.cookies.clear()


@pytest.fixture()
def master_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(master_csv, "EXPORTS", tmp_path / "exports")
    monkeypatch.setattr(master_csv, "_last_error", None)
    return tmp_path / "exports"


@pytest.fixture()
def states(client, monkeypatch):
    """Every state an upload job is set to, in order."""
    db, seen = client.app.state.app_state.db, []
    orig = db.update_s3_upload

    def spy(uid, **f):
        if "state" in f:
            seen.append(f["state"])
        return orig(uid, **f)
    monkeypatch.setattr(db, "update_s3_upload", spy)
    return seen


def post(client, name=NVR, data=VIDEO, card_id="50", date=None):
    form = {"card_id": card_id} | ({"date": date} if date is not None else {})
    return client.post("/api/s3/uploads", data=form, files={"file": (name, data, "video/mp4")})


def wait(client, uid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/s3/uploads/{uid}").json()
        if j["state"] in ("completed", "failed"):
            return j
        time.sleep(0.25)
    raise AssertionError("upload job did not finish")


def master_rows(master_dir):
    p = master_dir / master_csv.MASTER_NAME
    return len(p.read_text(encoding="utf-8").splitlines()) - 1 if p.exists() else 0


def uniform_and_not(monkeypatch):
    monkeypatch.setattr(detect, "classify_frame", lambda *a: [Person((100, 80, 220, 460), "uniform", 0.9),
                                                              Person((300, 50, 400, 420), "non_uniform", 0.1)])


# ------------------------------------------------------------------ 1. admin authentication
def test_admin_login(client, fake, monkeypatch):
    client.cookies.clear()
    auth._failures.clear()
    for path in ("/api/auth/me", "/api/s3/cards", "/api/s3/uploads"):
        assert client.get(path).status_code == 401, path
    assert post(client).status_code == 401
    assert client.post("/api/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "root", "password": "test-password"}).status_code == 401
    r = client.post("/api/auth/login", json={"username": "admin", "password": "test-password"})
    assert r.status_code == 200 and r.json() == {"username": "admin"}
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "test-password" not in cookie
    assert client.get("/api/auth/me").json() == {"username": "admin"}
    assert client.get("/api/s3/cards").json() == {"cards": [{"card_id": 50, "cart": "cart_50", "store_name": "IOCL Khandwa naka"}]}

    token = client.cookies.get(auth.COOKIE)                       # a tampered or expired token is refused
    client.cookies.set(auth.COOKIE, token[:-2] + ("AA" if not token.endswith("AA") else "BB"))
    assert client.get("/api/auth/me").status_code == 401
    client.cookies.set(auth.COOKIE, auth.make_token("admin", now=time.time() - 13 * 3600))
    assert client.get("/api/auth/me").status_code == 401

    client.cookies.set(auth.COOKIE, token)
    client.post("/api/auth/logout")
    client.cookies.clear()
    assert client.get("/api/auth/me").status_code == 401

    for _ in range(auth.MAX_FAILURES):                            # brute force is slowed down
        client.post("/api/auth/login", json={"username": "admin", "password": "nope"})
    assert client.post("/api/auth/login", json={"username": "admin", "password": "test-password"}).status_code == 429
    auth._failures.clear()

    monkeypatch.setenv("SU_ADMIN_PASSWORD_HASH", "")              # not configured: closed, never open
    assert client.post("/api/auth/login", json={"username": "admin", "password": "test-password"}).status_code == 503
    assert client.get("/api/s3/cards").status_code == 503


def test_password_hash_roundtrip():
    h = auth.hash_password("correct horse", iterations=1000)
    assert auth.verify_password("correct horse", h) and not auth.verify_password("wrong", h)
    assert not auth.verify_password("x", "garbage")


# ------------------------------------------------------------------ 2-3. upload and Card ID validation
def test_invalid_uploads_are_refused(admin, fake, monkeypatch):
    assert post(admin, name="photo.jpg").status_code == 415
    assert post(admin, name="notes.txt", data=b"x").status_code == 415
    for card in ("51", "0", "abc", "-50"):
        assert post(admin, card_id=card).status_code == 422, card
    r = post(admin, name="clip.mp4")                               # no recording date in the name, none chosen
    assert r.status_code == 422 and "date" in r.json()["detail"]
    assert post(admin, name="clip.mp4", date="2026-11-04").status_code == 422
    r = post(admin, name="broken.mp4", data=b"not a video at all", date="04-11-2026")
    assert r.status_code == 422
    from backend.api import routes
    monkeypatch.setattr(routes, "MAX_VIDEO_MB", 0)
    assert post(admin).status_code == 413
    assert not list(s3_upload.INCOMING.glob("*")), "a refused upload must not leave files behind"
    assert fake.puts == []                                         # nothing reached S3


def test_s3_unavailable_is_reported(admin, client, monkeypatch):
    monkeypatch.setattr(client.app.state.app_state.s3_batches, "client_factory", lambda: FakeS3Put(credentials=False))
    r = post(admin)
    assert r.status_code == 503 and "No AWS credentials" in r.json()["detail"]


def test_safe_filename():
    assert s3.safe_filename(NVR) == NVR
    assert s3.safe_filename("..\\..\\etc/pass wd?.MP4") == "pass_wd.mp4"
    assert s3.safe_filename("...mp4") == "video.mp4"


# ------------------------------------------------------------------ 4-9. end to end
def test_upload_process_and_report(admin, fake, real_model, master_dir, states, monkeypatch):  # noqa: F811
    uniform_and_not(monkeypatch)
    r = post(admin)
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["how"] == "created" and job["card_id"] == 50 and job["cart"] == "cart_50"
    assert job["day"] == "04-11-2026" and job["day_source"] == "file_name"
    assert job["s3_input"] == f"s3://{BUCKET}/{IN}"
    assert job["output_key"] == f"{OUT}cart_50_{NVR[:-4]}_{job['id']}.csv"
    assert "local_path" not in job and "report_path" not in job

    done = wait(admin, job["id"])
    assert done["state"] == "completed", done
    assert states == ["uploading", "processing", "generating_report", "uploading_output", "completed"]
    # input: the uploaded bytes, verified; output: exactly one CSV, nothing else in output_data
    assert fake.objects[IN] == VIDEO and fake.content_types[IN] == "video/mp4" and done["input_verified"]
    assert fake.keys(OUT) == [done["output_key"]] and fake.content_types[done["output_key"]] == "text/csv"
    header, *rows = list(csv.reader(io.StringIO(fake.objects[done["output_key"]].decode())))
    assert header == HEADER and len(rows) == 4
    for i, (cid, store, day, t, img, score, remark) in enumerate(rows, 1):
        assert (cid, store, day, img, score) == ("50", "IOCL Khandwa naka", "04-11-2026", f"pic{i}", "50")
        assert t >= "21:51:26" and remark == f"{i}. person2 is not wearing the dress, others wearing"
    assert done["summary"] == {"screenshots": 4, "unreadable_frames": 0, "people": 8, "uniform": 4,
                               "not_uniform": 4, "manual_review": 0}
    # the admin portal sees the rows and can download the same CSV
    assert [r["img name "] for r in done["report_rows"]] == ["pic1", "pic2", "pic3", "pic4"]
    dl = admin.get(done["report_url"])
    assert dl.status_code == 200 and dl.content == fake.objects[done["output_key"]]
    assert any(u["id"] == job["id"] for u in admin.get("/api/s3/uploads").json()["uploads"])
    assert not list(s3_upload.INCOMING.glob("*"))                   # local copy deleted once verified in S3
    assert master_rows(master_dir) == 4

    # the same video again: the same job, nothing uploaded or analysed twice
    again = post(admin)
    assert again.status_code == 200 and again.json()["how"] == "duplicate" and again.json()["id"] == job["id"]
    assert fake.puts.count(IN) == 1 and len(fake.keys(OUT)) == 1 and master_rows(master_dir) == 4
    assert not list(s3_upload.INCOMING.glob("*"))


def test_processing_failure_uploads_no_report(admin, fake):
    """Without trained weights detection is refused: failed in 'processing', the video stays in S3, no CSV."""
    name = "20261104_20261104222309_20261104222440_195309.mp4"
    job = wait(admin, post(admin, name=name, data=VIDEO + b"\3").json()["id"])
    assert job["state"] == "failed" and job["failed_stage"] == "processing" and "weights not found" in job["error"]
    assert job["input_verified"] and f"{DAY}/cart_50/input_data/{name}" in fake.objects
    assert fake.keys(OUT) == [] and job["report_url"] is None and job["report_rows"] is None
    assert admin.get(f"/api/s3/uploads/{job['id']}/report.csv").status_code == 404


# ------------------------------------------------------------------ 10. retries and duplicates
def test_retry_resumes_without_duplicates(admin, fake, real_model, master_dir, monkeypatch):  # noqa: F811
    uniform_and_not(monkeypatch)
    fake.fail_put = lambda key: key.endswith(".csv")                # output upload fails
    job = wait(admin, post(admin, data=VIDEO + b"\0").json()["id"])  # (another video than the test above)
    assert job["state"] == "failed" and job["failed_stage"] == "uploading_output" and "S3 is down" in job["error"]
    assert fake.keys(OUT) == [] and master_rows(master_dir) == 4     # the analysis itself finished

    fake.fail_put = lambda key: False
    r = admin.post(f"/api/s3/uploads/{job['id']}/retry")
    assert r.status_code == 202 and r.json()["attempts"] == 2
    done = wait(admin, job["id"])
    assert done["state"] == "completed" and done["extraction_id"] == job["extraction_id"]   # analysis reused
    assert fake.puts.count(job["input_key"]) == 1 and fake.keys(OUT) == [done["output_key"]]
    assert master_rows(master_dir) == 4                                                   # not appended twice
    r = admin.post(f"/api/s3/uploads/{job['id']}/retry")
    assert r.status_code == 409 and "Only a failed job" in r.json()["detail"]


def test_failed_upload_is_retried_by_uploading_again(admin, fake):
    fake.fail_put = lambda key: "input_data" in key
    data = VIDEO + b"\1"
    job = wait(admin, post(admin, data=data).json()["id"])
    assert job["state"] == "failed" and job["failed_stage"] == "uploading" and not job["input_verified"]
    fake.fail_put = lambda key: False
    again = post(admin, data=data)
    assert again.status_code == 202 and again.json()["how"] == "retried" and again.json()["id"] == job["id"]
    job = wait(admin, job["id"])
    assert job["input_verified"] and fake.objects[job["input_key"]] == data
    assert len(list(s3_upload.INCOMING.glob("*"))) == 0


def test_existing_input_is_never_overwritten(admin, fake):
    fake.objects[IN] = b"another recording with the same name"     # no checksum: not provably identical
    job = wait(admin, post(admin, data=VIDEO + b"\2").json()["id"])
    assert job["state"] == "failed" and job["failed_stage"] == "uploading" and "not overwritten" in job["error"]
    assert fake.objects[IN] == b"another recording with the same name" and fake.keys(OUT) == []


def test_put_new_is_create_only(tmp_path):
    f, p = FakeS3Put(), tmp_path / "r.csv"
    p.write_bytes(b"a,b\n")
    sha = hashlib.sha256(b"a,b\n").hexdigest()
    assert s3.put_new(f, BUCKET, "k.csv", p, sha, "text/csv") is True
    assert s3.put_new(f, BUCKET, "k.csv", p, sha, "text/csv") is False             # identical: left as is
    s3.verify(f, BUCKET, "k.csv", 4, sha)
    p.write_bytes(b"a,c\n")
    with pytest.raises(s3.ObjectConflict):
        s3.put_new(f, BUCKET, "k.csv", p, hashlib.sha256(b"a,c\n").hexdigest(), "text/csv")
    assert f.objects["k.csv"] == b"a,b\n"
    with pytest.raises(RuntimeError):
        s3.verify(f, BUCKET, "k.csv", 4, "0" * 64)


def test_interrupted_jobs_fail_in_their_stage(client):
    db = client.app.state.app_state.db
    now = "2026-11-04T00:00:00+00:00"
    db.add_s3_upload({"id": "upl-test-interrupt", "created_at": now, "updated_at": now, "card_id": 50, "cart": "cart_50",
                      "filename": "a.mp4", "s3_name": "a.mp4", "size_bytes": 1, "sha256": "0" * 64, "day": "04-11-2026",
                      "day_source": "selected", "bucket": BUCKET, "input_key": "i", "output_key": "o",
                      "state": "generating_report"})
    db.interrupt_s3_uploads()
    j = db.s3_upload("upl-test-interrupt")
    assert j["state"] == "failed" and j["failed_stage"] == "generating_report" and "use Retry" in j["error"]
