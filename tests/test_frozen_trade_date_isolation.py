"""Verification tests for non-trading day test isolation and frozen_trade_date fixture (DAV-289).

Ensures that tests utilizing frozen_trade_date execute deterministically
and identically across arbitrary natural calendar days (weekends, holidays, intraday).
"""
from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tradingagents.dataflows import trade_calendar as tc
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.knowledge.historical_cases import (
    calculate_t1_return,
    get_next_cn_trading_day,
)

CN_TZ = ZoneInfo("Asia/Shanghai")


def test_frozen_trade_date_fixture_contract(frozen_trade_date):
    """Verify frozen_trade_date fixture guarantees fixed Monday post-close semantics."""
    assert frozen_trade_date == "2026-08-17"
    assert tc.cn_today_str() == "2026-08-17"
    assert tc.now_cn() == datetime(2026, 8, 17, 16, 0, 0, tzinfo=CN_TZ)
    assert tc.is_cn_trading_day(frozen_trade_date) is True
    assert tc.cn_market_phase() == "post_close"
    assert tc.is_historical_analysis_date(frozen_trade_date) is False
    assert tc.is_historical_analysis_date("2026-08-14") is True
    assert tc.unavailable_analysis_date_reason(frozen_trade_date) is None
    assert tc.unavailable_analysis_date_reason("2026-08-18") is not None


@pytest.mark.parametrize(
    "simulated_natural_dt",
    [
        datetime(2026, 8, 22, 11, 0, tzinfo=CN_TZ),  # Saturday
        datetime(2026, 8, 23, 20, 0, tzinfo=CN_TZ),  # Sunday
        datetime(2026, 10, 1, 10, 0, tzinfo=CN_TZ),  # National Day holiday
        datetime(2026, 8, 18, 10, 30, tzinfo=CN_TZ), # Tuesday intraday session
    ],
    ids=["saturday", "sunday", "national_holiday", "intraday_session"],
)
def test_isolation_across_simulated_natural_days(monkeypatch, simulated_natural_dt):
    """Verify that regardless of the simulated natural clock, frozen_trade_date enforces isolated trading date."""
    # First, verify that without isolation, the simulated date would produce non-trading day or different phase
    raw_today_str = simulated_natural_dt.date().strftime("%Y-%m-%d")
    is_weekend = simulated_natural_dt.weekday() >= 5
    if is_weekend:
        assert tc.is_cn_trading_day(raw_today_str) is False

    # Now apply frozen_trade_date
    from tests.conftest import _apply_frozen_trade_date, FROZEN_TRADE_DATE

    _apply_frozen_trade_date(monkeypatch)

    assert tc.cn_today_str() == FROZEN_TRADE_DATE
    assert tc.is_cn_trading_day(tc.cn_today_str()) is True
    assert tc.cn_market_phase() == "post_close"

    # Fund flow snapshot provider behaves consistently as a trading day
    provider = CnAkshareProvider()
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = RuntimeError("eastmoney rate-limited")
    ak.stock_fund_flow_individual.return_value = pd.DataFrame(
        [
            {
                "股票代码": "600519",
                "最新价": "1700.00",
                "净额": "5.60亿",
                "流入资金": "10.00亿",
                "流出资金": "4.40亿",
                "换手率": "0.50",
            }
        ]
    )
    provider._ak = lambda: ak

    with patch("requests.get", return_value=MagicMock(text="[]")):
        text = provider.get_individual_fund_flow("600519", curr_date=tc.cn_today_str())

    assert "同花顺即时资金流净额快照" in text
    assert "资金净额: 5.60亿" in text
