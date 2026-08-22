"""Test fixtures for tests/ directory.

Provides unified date/trade-calendar isolation fixtures to ensure tests
run deterministically on any natural day (weekends, holidays, intraday).
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

CN_TZ = ZoneInfo("Asia/Shanghai")

# A fixed known trading day: a historical Monday (2026-08-17) with post-close timestamp.
FROZEN_TRADE_DATE = "2026-08-17"
FROZEN_TRADE_DATE_OBJ = date(2026, 8, 17)
FROZEN_TRADE_DATETIME = datetime(2026, 8, 17, 16, 0, 0, tzinfo=CN_TZ)


def _apply_frozen_trade_date(monkeypatch):
    """Patch all trade_calendar time/date entrypoints and module imports."""
    from tradingagents.dataflows import trade_calendar as tc

    monkeypatch.setattr(tc, "now_cn", lambda: FROZEN_TRADE_DATETIME)
    monkeypatch.setattr(tc, "cn_today_str", lambda: FROZEN_TRADE_DATE)

    # Patch any already-imported modules across tradingagents, api, tests, and conftest
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name.startswith(("tradingagents", "api", "tests", "conftest")):
            if hasattr(mod, "cn_today_str"):
                monkeypatch.setattr(mod, "cn_today_str", lambda: FROZEN_TRADE_DATE)
            if hasattr(mod, "now_cn"):
                monkeypatch.setattr(mod, "now_cn", lambda: FROZEN_TRADE_DATETIME)


@pytest.fixture
def frozen_trade_date(monkeypatch):
    """Pytest fixture that stubs today and now_cn to a fixed known trading day (Monday 2026-08-17 16:00:00).

    Guarantees that:
    - cn_today_str() == "2026-08-17"
    - now_cn() == datetime(2026, 8, 17, 16, 0, 0, tzinfo=CN_TZ)
    - is_cn_trading_day("2026-08-17") == True
    - cn_market_phase() == "post_close"
    - is_historical_analysis_date("2026-08-17") == False
    - is_historical_analysis_date("2026-08-14") == True
    - unavailable_analysis_date_reason("2026-08-17") == None
    """
    _apply_frozen_trade_date(monkeypatch)
    return FROZEN_TRADE_DATE
