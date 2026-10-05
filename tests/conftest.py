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
