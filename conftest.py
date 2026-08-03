"""Pytest configuration.

Offline CI gate: force an isolated SQLite database before any test imports
api.database / api.main so tests cannot touch a real checkout database.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from datetime import date, timedelta
from unittest.mock import patch

import pytest

_ORIGINAL_DATABASE_URL = os.environ.get("DATABASE_URL")
_DB_DIR = tempfile.mkdtemp(prefix="ta-pytest-")
os.environ["DATABASE_URL"] = "sqlite:///" + _DB_DIR + "/pytest.db"

# The offline test suite runs without a TA_APP_SECRET_KEY and explicitly opts
# into the insecure built-in default key (DAV-66 startup guard). Production
# deployments must set TA_APP_SECRET_KEY or both the API and the scheduler
# refuse to start. setdefault keeps an externally provided value winning.
os.environ.setdefault("TA_ALLOW_DEFAULT_SECRET", "1")

_OFFLINE_STOCK_MAP = {
    "贵州茅台": "600519.SH",
    "宁德时代": "300750.SZ",
}
_OFFLINE_REVERSE_STOCK_MAP = {
    code: name for name, code in _OFFLINE_STOCK_MAP.items()
}

_OFFLINE_TRADE_START = date(2024, 1, 1)
_OFFLINE_TRADE_END = date(2027, 12, 31)
_OFFLINE_TRADE_DATES = [
    day
    for day in (
        _OFFLINE_TRADE_START + timedelta(days=offset)
        for offset in range((_OFFLINE_TRADE_END - _OFFLINE_TRADE_START).days + 1)
    )
    if day.weekday() < 5
]


def pytest_sessionstart(session):
    from api.database import init_db

    init_db()


@pytest.fixture(autouse=True)
def _seed_offline_stock_map():
    """Warm api.main's stock-map cache deterministically for every test."""
    from api import main as main_mod

    saved = (
        main_mod._cn_stock_map,
        main_mod._cn_stock_reverse_map,
        main_mod._cn_stock_map_loaded_at,
    )
    main_mod._cn_stock_map = dict(_OFFLINE_STOCK_MAP)
    main_mod._cn_stock_reverse_map = dict(_OFFLINE_REVERSE_STOCK_MAP)
    main_mod._cn_stock_map_loaded_at = time.time()
    try:
        yield
    finally:
        (
            main_mod._cn_stock_map,
            main_mod._cn_stock_reverse_map,
            main_mod._cn_stock_map_loaded_at,
        ) = saved


@pytest.fixture(autouse=True)
def _offline_trade_calendar(request):
    """Replace AkShare calendar fetches with a deterministic offline table.

    trade_calendar's own fallback tests manage cache/fetch failures explicitly
    and are excluded so their assertions keep seeing an empty calendar.
    """
    module_name = getattr(request.module, "__name__", "")
    if module_name.endswith("test_trade_date_fallback"):
        yield
        return

    from tradingagents.dataflows import trade_calendar as tc

    tc.clear_cn_trade_date_cache()
    with patch.object(
        tc,
        "_fetch_cn_trade_dates_from_akshare",
        return_value=list(_OFFLINE_TRADE_DATES),
    ):
        yield
    tc.clear_cn_trade_date_cache()


def pytest_sessionfinish(session, exitstatus):
    """Restore the caller's DATABASE_URL and remove the temp SQLite dir."""
    if _ORIGINAL_DATABASE_URL is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = _ORIGINAL_DATABASE_URL
    shutil.rmtree(_DB_DIR, ignore_errors=True)
