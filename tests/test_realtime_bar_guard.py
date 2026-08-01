"""Daily bars must not contain an intraday quote row."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider


def _hist_with_today() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-07-27", "2026-07-28", "2026-07-29"]),
            "Open": [1.0, 1.1, 1.2],
            "High": [1.2, 1.3, 1.4],
            "Low": [0.9, 1.0, 1.1],
            "Close": [1.1, 1.2, 1.3],
            "Volume": [100, 110, 120],
        }
    )


def test_drop_incomplete_today_bar_during_market_session():
    provider = CnAkshareProvider()
    with patch(
        "tradingagents.dataflows.trade_calendar.cn_today_str",
        return_value="2026-07-29",
    ), patch(
        "tradingagents.dataflows.trade_calendar.cn_market_phase",
        return_value="in_session",
    ):
        out = provider._drop_incomplete_today_bar(_hist_with_today(), "2026-07-29")

    assert list(out["Date"].dt.strftime("%Y-%m-%d")) == ["2026-07-27", "2026-07-28"]


def test_drop_incomplete_today_bar_during_lunch_break():
    provider = CnAkshareProvider()
    with patch(
        "tradingagents.dataflows.trade_calendar.cn_today_str",
        return_value="2026-07-29",
    ), patch(
        "tradingagents.dataflows.trade_calendar.cn_market_phase",
        return_value="lunch_break",
    ):
        out = provider._drop_incomplete_today_bar(_hist_with_today(), "2026-07-29")

    assert list(out["Date"].dt.strftime("%Y-%m-%d")) == ["2026-07-27", "2026-07-28"]


def test_drop_incomplete_today_bar_keeps_closed_session_bar():
    provider = CnAkshareProvider()
    with patch(
        "tradingagents.dataflows.trade_calendar.cn_today_str",
        return_value="2026-07-29",
    ), patch(
        "tradingagents.dataflows.trade_calendar.cn_market_phase",
        return_value="closed",
    ):
        out = provider._drop_incomplete_today_bar(_hist_with_today(), "2026-07-29")

    assert len(out) == 3


class _HistoricalAk:
    def stock_zh_a_hist(self, **_kwargs):
        return _hist_with_today()

    def stock_individual_spot_xq(self, **_kwargs):
        raise AssertionError("daily history must not fetch Xueqiu intraday bars")


def test_fetch_history_does_not_append_xueqiu_intraday_bar():
    provider = CnAkshareProvider()
    provider._ak = lambda: _HistoricalAk()
    with patch(
        "tradingagents.dataflows.trade_calendar.cn_today_str",
        return_value="2026-07-29",
    ), patch(
        "tradingagents.dataflows.trade_calendar.cn_market_phase",
        return_value="in_session",
    ):
        out = provider._fetch_hist_df("600519", "2026-07-27", "2026-07-29")

    assert list(out["Date"].dt.strftime("%Y-%m-%d")) == ["2026-07-27", "2026-07-28"]
