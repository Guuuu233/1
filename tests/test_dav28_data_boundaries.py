"""DAV-28 data boundary regressions.

Everything here is offline: fake providers/frames and patched clocks.
"""

from __future__ import annotations

from datetime import datetime

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.providers.cn_baostock_provider import CnBaoStockProvider
from tradingagents.dataflows.providers.cn_investoday_provider import CnInvestodayProvider


def _ak_hist_df(volume_col: bool = True, amount_col: bool = False) -> pd.DataFrame:
    data = {
        "Date": ["2026-07-28", "2026-07-28", "bad", "2026-07-29"],
        "Open": [1.0, 1.0, float("nan"), 1.2],
        "High": [1.3, 1.3, float("nan"), 1.4],
        "Low": [0.9, 0.9, float("nan"), 1.1],
        "Close": [1.2, 1.2, float("nan"), 1.3],
    }
    if volume_col:
        data["Volume"] = [100.0, 300.0, float("nan"), 120.0]
    if amount_col:
        data["Amount"] = [1e6, 1e6, float("nan"), 1.2e6]
        data["成交额"] = [1e6, 1e6, float("nan"), 1.2e6]
    df = pd.DataFrame(data)
    # Deliberately unsorted: worst-case vendor shape.
    return df.iloc[[3, 0, 2, 1]].reset_index(drop=True)


# ── AkShare normalization ────────────────────────────────────────────────


def test_akshare_normalize_dedup_sort_and_drop_bad_future_rows():
    provider = CnAkshareProvider()
    out = provider._normalize_hist_df(_ak_hist_df())

    assert list(out.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    # 2026-07-28 duplicated -> unique; bad row and unsorted normalized.
    assert list(out["Date"].dt.strftime("%Y-%m-%d")) == ["2026-07-28", "2026-07-29"]
    assert list(out["Volume"]) == [300.0, 120.0]


def test_akshare_normalize_does_not_use_amount_as_volume():
    provider = CnAkshareProvider()
    # Amount present but Volume absent must fail, never alias Amount into Volume.
    with pytest.raises(ValueError):
        provider._normalize_hist_df(_ak_hist_df(volume_col=False, amount_col=True))


# ── Investoday/BaoStock completed-bar protection ─────────────────────────


def test_investoday_keeps_only_completed_bars_during_session():
    rows = [
        {"date": "2026-07-27", "openPrice": 1, "highPrice": 2, "lowPrice": 0.9, "closePrice": 1.5, "volume": 10},
        {"date": "2026-07-28", "openPrice": 1, "highPrice": 2, "lowPrice": 0.9, "closePrice": 1.5, "volume": 10},
        {"date": "2026-07-29", "openPrice": 1, "highPrice": 2, "lowPrice": 0.9, "closePrice": 1.5, "volume": 10},
    ]
    provider = CnInvestodayProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"), \
         patch.object(provider, "_fetch_paged_list", return_value=rows), \
         patch("tradingagents.dataflows.trade_calendar.cn_today_str", return_value="2026-07-29"), \
         patch("tradingagents.dataflows.trade_calendar.cn_market_phase", return_value="in_session"):
        out = provider.get_stock_data("600519.SH", "2026-07-27", "2026-07-29")

    assert "2026-07-27" in out
    assert "2026-07-28" in out
    assert "2026-07-29" not in out


def test_baostock_keeps_only_completed_bars_during_session():
    hist = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-07-27", "2026-07-28", "2026-07-29"]),
            "Open": [1.0, 1.1, 1.2],
            "High": [1.3, 1.4, 1.5],
            "Low": [0.9, 1.0, 1.1],
            "Close": [1.1, 1.2, 1.3],
            "Volume": [100, 110, 120],
        }
    )
    provider = CnBaoStockProvider()
    provider._fetch_hist_df = MagicMock(return_value=hist)
    with patch("tradingagents.dataflows.trade_calendar.cn_today_str", return_value="2026-07-29"), \
         patch("tradingagents.dataflows.trade_calendar.cn_market_phase", return_value="in_session"):
        out = provider.get_stock_data("600519", "2026-07-27", "2026-07-29")

    assert "2026-07-27" in out
    assert "2026-07-28" in out
    assert "2026-07-29" not in out


# ── DataCollector: date filtering before indicators/VPA/prompt ─────────


def test_fetch_all_filters_dedupes_and_sorts_before_prompt_and_vpa():
    from tradingagents.graph import data_collector as dc

    csv_rows = "\n".join(
        [
            "date,open,high,low,close,volume",
            "2026-07-29,1.2,1.3,1.1,1.2,120",  # future > trade_date
            "2026-07-28,1.0,1.1,0.9,1.1,110",
            "2026-07-28,1.0,1.1,0.9,1.1,999",  # duplicate
            "bad,1.0,1.1,0.9,1.1,110",
        ]
    )
    realtime_context = {
        "status": "not_applicable",
        "source": None,
        "quote_as_of": None,
        "retrieved_at": None,
        "error": None,
        "quote": None,
    }
    seen_vpa: list[pd.DataFrame] = []

    def fake_vpa(df, window=20):
        seen_vpa.append(df.copy())
        return "ok"

    patch_targets = {
        name: (lambda **kwargs: "safe")
        for name in [
            "get_stock_data", "get_indicators", "get_fundamentals", "get_balance_sheet",
            "get_cashflow", "get_income_statement", "get_news", "get_global_news",
            "get_insider_transactions", "get_board_fund_flow", "get_individual_fund_flow",
            "get_lhb_detail", "get_zt_pool", "get_hot_stocks_xq", "get_restricted_release",
            "get_share_pledge", "get_earnings_forecast", "get_shareholder_count",
            "get_margin_trading", "get_northbound_flow",
        ]
    }
    patch_targets["get_stock_data"] = lambda **kwargs: csv_rows
    with patch.multiple("tradingagents.graph.data_collector", **patch_targets), \
         patch.object(dc, "_fetch_realtime_context", return_value=realtime_context), \
         patch.object(dc, "_compute_vpa_indicators", side_effect=fake_vpa):
        result = dc._fetch_all("600519", "2026-07-28")

    assert len(seen_vpa) == 1
    df = seen_vpa[0]
    # Only completed bars <= trade_date, deduplicated, sorted.
    assert list(df["date"].astype(str)) == ["2026-07-28"]
    assert list(df["volume"]) == [110.0]
    # Prompt input must not contain the future bar or duplicate.
    assert "2026-07-29" not in result["stock_data"]
    assert result["stock_data"].count("2026-07-28") <= 1


# ── Trade-calendar analysis-date semantics ─────────────────────────────


def test_unavailable_analysis_date_reason_rejects_missing_invalid_future():
    from tradingagents.dataflows import trade_calendar as tc

    assert tc.unavailable_analysis_date_reason(None)
    assert tc.unavailable_analysis_date_reason("")
    assert tc.unavailable_analysis_date_reason("not-a-date")
    with patch(
        "tradingagents.dataflows.trade_calendar.now_cn",
        return_value=datetime(2026, 7, 29, 12, 0, 0),
    ):
        assert tc.unavailable_analysis_date_reason("2026-07-30")

    today_ok = tc.unavailable_analysis_date_reason("2026-07-29")
    assert today_ok is None
