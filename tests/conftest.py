"""Tests run against a throw-away storage directory (never the real database)."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_TMP = Path(tempfile.mkdtemp(prefix="su_test_"))
os.environ["SU_STORAGE_DIR"] = str(_TMP)
os.environ["SU_DB_PATH"] = str(_TMP / "test.db")
# Pipeline tests are written against the deterministic stripe baseline (their test media is a
# synthetic striped patch), so they must not depend on whichever trained weights are installed.
os.environ["SU_UNIFORM_MODEL"] = str(_TMP / "no_trained_model.pt")
# No background scheduler in tests: scheduling is tested by calling CaptureService.tick(now) directly.
os.environ["SU_CAPTURE_SCHEDULER"] = "off"
# Never the real bucket or AWS keys from the project's .env (load_dotenv does not override these).
for _k in ("SU_S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    os.environ[_k] = ""
# ...nor ~/.aws credentials / profiles or an instance role: no test can reach a real bucket (every finished
# analysis uploads its CSV to S3, so tests that need S3 patch the client factory with a stand-in).
os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(_TMP / "no-aws-credentials")
os.environ["AWS_CONFIG_FILE"] = str(_TMP / "no-aws-config")
os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
os.environ.pop("AWS_PROFILE", None)
# A test admin account (never the one in the project's .env). Password: "test-password" (few iterations: fast).
os.environ["SU_ADMIN_USERNAME"] = "admin"
os.environ["SU_ADMIN_PASSWORD_HASH"] = ("pbkdf2_sha256$1000$00112233445566778899aabbccddeeff$"
                                        + __import__("hashlib").pbkdf2_hmac("sha256", b"test-password",
                                          bytes.fromhex("00112233445566778899aabbccddeeff"), 1000).hex())
os.environ["SU_SESSION_SECRET"] = "test-session-secret"
os.environ["SU_COOKIE_SECURE"] = "0"

import pytest  # noqa: E402

MEDIA = ROOT / "tests" / "media"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from backend.main import create_app
    with TestClient(create_app()) as c:
        yield c


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
