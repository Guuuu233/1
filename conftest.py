"""Pytest configuration.

Offline CI gate: force an isolated SQLite database before any test imports
api.database / api.main so tests cannot touch a real checkout database.
"""
from __future__ import annotations

import os
import shutil
import tempfile

_ORIGINAL_DATABASE_URL = os.environ.get("DATABASE_URL")
_DB_DIR = tempfile.mkdtemp(prefix="ta-pytest-")
os.environ["DATABASE_URL"] = "sqlite:///" + _DB_DIR + "/pytest.db"


def pytest_sessionstart(session):
    from api.database import init_db

    init_db()


def pytest_sessionfinish(session, exitstatus):
    """Restore the caller's DATABASE_URL and remove the temp SQLite dir."""
    if _ORIGINAL_DATABASE_URL is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = _ORIGINAL_DATABASE_URL
    shutil.rmtree(_DB_DIR, ignore_errors=True)
