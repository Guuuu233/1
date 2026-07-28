"""Refuse realtime bars when calendar/date is not trustworthy."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows import trade_calendar as tc
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider


class _SpotAk:
    def __init__(self, items: dict):
        self.items = items

    def stock_individual_spot_xq(self, symbol, token=None):
        return pd.DataFrame(
            {
                "item": list(self.items.keys()),
                "value": list(self.items.values()),
            }
        )


class _Provider(CnAkshareProvider):
    def __init__(self, items: dict):
        self._items = items

    def _ak(self):
        return _SpotAk(self._items)

    def _xq_symbol(self, symbol: str) -> str:
        return f"SH{symbol}" if str(symbol).startswith("6") else f"SZ{symbol}"


def _hist_without_today() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-07-27", "2026-07-28"]),
            "Open": [1.0, 1.1],
            "High": [1.2, 1.3],
            "Low": [0.9, 1.0],
            "Close": [1.1, 1.2],
            "Volume": [100, 110],
        }
    )


def test_fetch_realtime_row_refuses_unparseable_time(caplog):
    items = {
        "时间": "not-a-date",
        "今开": 10,
        "最高": 11,
        "最低": 9,
        "现价": 10.5,
        "成交量": 1000,
    }
    with caplog.at_level("WARNING"):
        rt = _Provider(items)._fetch_realtime_row_unlocked("600519")
    assert rt.empty
    assert any("unparseable spot time" in r.message for r in caplog.records)
    assert any("not-a-date" in r.message for r in caplog.records)


def test_fetch_realtime_row_refuses_missing_time_without_defaulting_today(caplog):
    items = {
        # no 时间 key
        "今开": 10,
        "最高": 11,
        "最低": 9,
        "现价": 10.5,
        "成交量": 1000,
    }
    with caplog.at_level("WARNING"):
        rt = _Provider(items)._fetch_realtime_row_unlocked("600519")
    assert rt.empty
    assert any("unparseable spot time" in r.message for r in caplog.records)


def test_fetch_realtime_row_accepts_parseable_time():
    items = {
        "时间": "2026-07-29 14:30:00",
        "今开": 10,
        "最高": 11,
        "最低": 9,
        "现价": 10.5,
        "成交量": 1000,
    }
    rt = _Provider(items)._fetch_realtime_row_unlocked("600519")
    assert not rt.empty
    assert pd.to_datetime(rt.iloc[0]["Date"]).strftime("%Y-%m-%d") == "2026-07-29"


def test_maybe_append_refuses_when_calendar_unavailable(caplog):
    provider = _Provider(
        {
            "时间": "2026-07-29 14:30:00",
            "今开": 10,
            "最高": 11,
            "最低": 9,
            "现价": 10.5,
            "成交量": 1000,
        }
    )
    hist = _hist_without_today()
    tc.clear_cn_trade_date_cache()
    with patch.object(tc, "_fetch_cn_trade_dates_from_akshare", side_effect=RuntimeError("down")):
        with patch(
            "tradingagents.dataflows.providers.cn_akshare_provider.cn_today_str",
            return_value="2026-07-29",
        ):
            with caplog.at_level("WARNING"):
                out = provider._maybe_append_realtime_row(
                    "600519", hist, "2026-07-29", assume_locked=True
                )
    assert len(out) == len(hist)
    assert list(pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")) == [
        "2026-07-27",
        "2026-07-28",
    ]
    assert any("trade calendar unavailable" in r.message for r in caplog.records)


def test_maybe_append_accepts_valid_today_bar_on_trading_day():
    # seed calendar with today
    tc.clear_cn_trade_date_cache()
    from datetime import date

    days = [
        date(2026, 7, 27),
        date(2026, 7, 28),
        date(2026, 7, 29),
    ]
    tc._TRADE_DATES_CACHE["dates"] = days
    tc._TRADE_DATES_CACHE["dates_set"] = set(days)
    tc._TRADE_DATES_CACHE["loaded_at"] = 1e18

    provider = _Provider(
        {
            "时间": "2026-07-29 14:30:00",
            "今开": 10,
            "最高": 11,
            "最低": 9,
            "现价": 10.5,
            "成交量": 1000,
        }
    )
    hist = _hist_without_today()
    with patch(
        "tradingagents.dataflows.providers.cn_akshare_provider.cn_today_str",
        return_value="2026-07-29",
    ), patch(
        "tradingagents.dataflows.providers.cn_akshare_provider.cn_market_phase",
        return_value="in_session",
    ):
        out = provider._maybe_append_realtime_row(
            "600519", hist, "2026-07-29", assume_locked=True
        )
    assert len(out) == 3
    assert pd.to_datetime(out.iloc[-1]["Date"]).strftime("%Y-%m-%d") == "2026-07-29"
