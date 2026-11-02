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
