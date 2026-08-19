"""DAV-28 data boundary regressions.

Everything here is offline: fake providers/frames and patched clocks.
"""

from __future__ import annotations

from datetime import datetime
import io

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.providers.cn_baostock_provider import CnBaoStockProvider
from tradingagents.dataflows.providers.cn_investoday_provider import CnInvestodayProvider


def _ak_hist_df(
    volume_col: bool = True,
    amount_col: bool = False,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    data = {
        "Date": ["2026-07-28", "2026-07-28", "bad", "2026-07-29"],
        "Open": [1.0, 1.0, float("nan"), 1.2],
        "High": [1.3, 1.3, float("nan"), 1.4],
        "Low": [0.9, 0.9, float("nan"), 1.1],
        "Close": [1.2, 1.2, float("nan"), 1.3],
    }
    if volume_col:
        data["Volume"] = volumes if volumes is not None else [100.0, 100.0, float("nan"), 120.0]
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
    # Identical 2026-07-28 duplicates collapse; bad row and unsorted normalized.
    assert list(out["Date"].dt.strftime("%Y-%m-%d")) == ["2026-07-28", "2026-07-29"]
    assert list(out["Volume"]) == [100.0, 120.0]


def test_akshare_normalize_rejects_conflicting_duplicate_dates():
    provider = CnAkshareProvider()
    # Same date, different Volume -> vendor row order must not decide silently.
    with pytest.raises(ValueError, match="duplicate daily bars"):
        provider._normalize_hist_df(
            _ak_hist_df(volumes=[100.0, 300.0, float("nan"), 120.0])
        )


class _MultiSourceAk:
    """Fake akshare with several history sources for the provider's fallback chain.

    One source yields a duplicate-bar conflict; reaching any other source is a
    test failure. Call counts are recorded so the Eastmoney retry loop can be
    proven to stop at the first conflicting attempt.
    """

    def __init__(self, conflict_source: str, empty_sources: set[str] | None = None):
        self._conflict = conflict_source
        self._empty = empty_sources or set()
        self.calls: list[str] = []

    def _maybe(self, source: str) -> pd.DataFrame:
        self.calls.append(source)
        if source == self._conflict:
            return _ak_hist_df(volumes=[100.0, 300.0, float("nan"), 120.0])
        if source in self._empty:
            return pd.DataFrame()
        raise AssertionError(
            f"{source} must not be called after a duplicate-bar conflict"
        )

    def fund_etf_hist_sina(self, **_kwargs):
        return self._maybe("fund_etf_hist_sina")

    def fund_etf_hist_em(self, **_kwargs):
        return self._maybe("fund_etf_hist_em")

    def stock_zh_a_hist(self, **_kwargs):
        return self._maybe("stock_zh_a_hist")

    def stock_zh_a_daily(self, **_kwargs):
        return self._maybe("stock_zh_a_daily")

    def stock_zh_a_hist_tx(self, **_kwargs):
        return self._maybe("stock_zh_a_hist_tx")


@pytest.mark.parametrize(
    "symbol,conflict_source,empty_sources",
    [
        # Eastmoney conflicts; Sina/Tencent must not be reached.
        ("600519", "stock_zh_a_hist", None),
        # ETF branch: Sina conflicts before ETF-EM or the stock sources are tried.
        ("510300", "fund_etf_hist_sina", None),
        # ETF branch: Sina is empty, then ETF-EM conflicts; no stock-source switch.
        ("510300", "fund_etf_hist_em", {"fund_etf_hist_sina"}),
    ],
)
def test_akshare_duplicate_conflict_propagates_without_source_switch(
    symbol, conflict_source, empty_sources
):
    """A duplicate-bar conflict inside the real AkShare provider must propagate.

    The provider's internal ``except Exception`` fallback chain must not swallow
    ``DuplicateBarConflictError`` and switch (or retry) another data source:
    every alternative source's row order is equally arbitrary, so the conflict
    has to surface to the router as an explicit unavailable refusal.
    """
    from tradingagents.dataflows.trade_calendar import DuplicateBarConflictError

    provider = CnAkshareProvider()
    fake_ak = _MultiSourceAk(conflict_source, empty_sources)
    provider._ak = lambda: fake_ak

    with pytest.raises(DuplicateBarConflictError, match="duplicate daily bars"):
        provider._fetch_hist_df(symbol, "2026-07-01", "2026-07-28")

    expected_calls = (
        ["fund_etf_hist_sina", conflict_source]
        if "fund_etf_hist_sina" in (empty_sources or set())
        else [conflict_source]
    )
    assert fake_ak.calls == expected_calls


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

    fresh_df = pd.read_csv(io.StringIO(out), comment="#")
    dates = fresh_df["Date"].astype(str).tolist()
    assert "2026-07-27" in dates
    assert "2026-07-28" in dates
    assert "2026-07-29" not in dates


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

    fresh_df = pd.read_csv(io.StringIO(out), comment="#")
    dates = fresh_df["Date"].astype(str).tolist()
    assert "2026-07-27" in dates
    assert "2026-07-28" in dates
    assert "2026-07-29" not in dates


def test_investoday_rejects_conflicting_duplicate_dates():
    rows = [
        {"date": "2026-07-28", "openPrice": 1, "highPrice": 2, "lowPrice": 0.9, "closePrice": 1.5, "volume": 10},
        {"date": "2026-07-28", "openPrice": 1, "highPrice": 2, "lowPrice": 0.9, "closePrice": 1.5, "volume": 999},
    ]
    provider = CnInvestodayProvider()
    with pytest.raises(ValueError, match="duplicate daily bars"):
        provider._iv_adjusted_rows_to_df(rows)


class _FakeBaostockResultSet:
    error_code = "0"
    error_msg = ""
    fields = ["date", "open", "high", "low", "close", "volume"]

    def __init__(self, rows: list[list[str]]):
        self._rows = list(rows)
        self._i = 0

    def next(self) -> bool:
        if self._i < len(self._rows):
            return True
        return False

    def get_row_data(self) -> list[str]:
        row = self._rows[self._i]
        self._i += 1
        return row


def test_baostock_rejects_conflicting_duplicate_dates():
    provider = CnBaoStockProvider()
    rs = _FakeBaostockResultSet(
        [
            ["2026-07-28", "1.0", "1.3", "0.9", "1.2", "100"],
            ["2026-07-28", "1.0", "1.3", "0.9", "1.2", "999"],
        ]
    )
    fake_bs = MagicMock()
    fake_bs.query_history_k_data_plus.return_value = rs
    with patch.object(provider, "_session", return_value=nullcontext(fake_bs)):
        with pytest.raises(ValueError, match="duplicate daily bars"):
            provider._fetch_hist_df("600519", "2026-07-01", "2026-07-28")


# ── DataCollector: date filtering before indicators/VPA/prompt ─────────


def test_fetch_all_filters_dedupes_and_sorts_before_prompt_and_vpa():
    from tradingagents.graph import data_collector as dc

    csv_rows = "\n".join(
        [
            "date,open,high,low,close,volume",
            "2026-07-29,1.2,1.3,1.1,1.2,120",  # future > trade_date
            "2026-07-28,1.0,1.1,0.9,1.1,999",
            "2026-07-28,1.0,1.1,0.9,1.1,999",  # identical duplicate
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
            "get_stock_data", "get_cn_indices", "get_global_indices", "get_major_assets",
            "get_indicators", "get_fundamentals", "get_balance_sheet",
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
    assert list(df["volume"]) == [999.0]
    # Prompt input must not contain the future bar or duplicate row.
    fresh_df = pd.read_csv(io.StringIO(result["stock_data"]), comment="#")
    assert list(fresh_df["date"]) == ["2026-07-28"]
    assert list(fresh_df["volume"]) == [999.0]
    assert "2026-07-29" not in result["stock_data"]
    assert "# as-of: 2026-07-28" in result["stock_data"]


def _fetch_all_with_csv(
    csv_rows: str,
    trade_date: str = "2026-07-28",
    *,
    vpa_must_not_run: bool = True,
):
    """Run _fetch_all with a fake CSV stock_data payload (fully offline)."""
    from tradingagents.graph import data_collector as dc

    realtime_context = {
        "status": "not_applicable",
        "source": None,
        "quote_as_of": None,
        "retrieved_at": None,
        "error": None,
        "quote": None,
    }
    patch_targets = {
        name: (lambda **kwargs: "safe")
        for name in [
            "get_stock_data", "get_cn_indices", "get_global_indices", "get_major_assets",
            "get_indicators", "get_fundamentals", "get_balance_sheet",
            "get_cashflow", "get_income_statement", "get_news", "get_global_news",
            "get_insider_transactions", "get_board_fund_flow", "get_individual_fund_flow",
            "get_lhb_detail", "get_zt_pool", "get_hot_stocks_xq", "get_restricted_release",
            "get_share_pledge", "get_earnings_forecast", "get_shareholder_count",
            "get_margin_trading", "get_northbound_flow",
        ]
    }
    patch_targets["get_stock_data"] = lambda **kwargs: csv_rows
    if vpa_must_not_run:
        vpa_patch = patch.object(
            dc, "_compute_vpa_indicators",
            side_effect=AssertionError("VPA must not run without valid daily data"),
        )
    else:
        vpa_patch = patch.object(dc, "_compute_vpa_indicators", return_value="ok")
    with patch.multiple("tradingagents.graph.data_collector", **patch_targets), \
         patch.object(dc, "_fetch_realtime_context", return_value=realtime_context), \
         vpa_patch:
        return dc._fetch_all("600519", trade_date)


def test_fetch_all_unavailable_when_all_rows_invalid():
    csv_rows = "\n".join(
        [
            "date,open,high,low,close,volume",
            "bad,1.0,1.1,0.9,1.1,110",
            "also-bad,2.0,2.1,1.9,2.1,120",
        ]
    )
    result = _fetch_all_with_csv(csv_rows)

    assert result["stock_data"].startswith("【数据获取失败】")
    assert "无有效完整日线数据" in result["stock_data"]
    assert "bad,1.0" not in result["stock_data"]
    assert result["indicators"]["close_50_sma"] == "无数据"
    assert result["vpa_indicators"] == "VPA 数据不足"
    assert result["market_data_context"]["daily"]["completeness"] == "unavailable"


def test_fetch_all_unavailable_when_columns_missing():
    # No Volume column: must not leak a partial OHLC frame to indicators/VPA.
    csv_rows = "date,open,high,low,close\n2026-07-28,1.0,1.1,0.9,1.1\n"
    result = _fetch_all_with_csv(csv_rows)

    assert result["stock_data"].startswith("【数据获取失败】")
    assert "无有效完整日线数据" in result["stock_data"]
    assert result["indicators"]["close_50_sma"] == "无数据"
    assert result["vpa_indicators"] == "VPA 数据不足"
    assert result["market_data_context"]["daily"]["completeness"] == "unavailable"


def test_fetch_all_unavailable_when_only_future_rows():
    csv_rows = "date,open,high,low,close,volume\n2026-07-29,1.2,1.3,1.1,1.2,120\n"
    result = _fetch_all_with_csv(csv_rows, trade_date="2026-07-28")

    assert result["stock_data"].startswith("【数据获取失败】")
    assert "2026-07-29" not in result["stock_data"]
    assert result["indicators"]["close_50_sma"] == "无数据"
    assert result["vpa_indicators"] == "VPA 数据不足"
    assert result["market_data_context"]["daily"]["completeness"] == "unavailable"


def test_fetch_all_preserves_source_comments_and_as_of():
    csv_rows = "\n".join(
        [
            "# Stock data for 600519 from 2026-07-01 to 2026-07-28",
            "# Total records: 1",
            "",
            "date,open,high,low,close,volume",
            "2026-07-28,1.0,1.1,0.9,1.1,999",
        ]
    )
    result = _fetch_all_with_csv(csv_rows, vpa_must_not_run=False)

    assert "# Stock data for 600519 from 2026-07-01 to 2026-07-28" in result["stock_data"]
    assert "# as-of: 2026-07-28" in result["stock_data"]
    assert "# normalized: sorted, deduped, date<=as-of, OHLCV columns" in result["stock_data"]
    assert result["market_data_context"]["daily"] == {
        "as_of": "2026-07-28",
        "completeness": "completed",
    }


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


def test_route_to_vendor_refuses_future_and_missing_as_of_without_provider_call():
    from tradingagents.dataflows import interface

    # Future stock-data analysis must be refused before any vendor fallback.
    future_refusal = interface._as_of_refusal(
        "get_stock_data", ("600519", "2026-01-01", "2030-01-01"), {}
    )
    assert future_refusal and "未来数据" in future_refusal

    # Missing stock-data end date is not allowed either.
    missing_refusal = interface._as_of_refusal(
        "get_stock_data", ("600519", "2026-01-01", None), {}
    )
    assert missing_refusal and "缺少分析日期" in missing_refusal

    # Positional curr_date methods are guarded too (fundamentals/collected data).
    positional_future_refusal = interface._as_of_refusal(
        "get_fundamentals", ("600519", "2030-01-01"), {}
    )
    assert positional_future_refusal and "未来数据" in positional_future_refusal

    # A live realtime quote with no curr_date is still allowed (dashboard live path).
    live_ok = interface._as_of_refusal("get_realtime_quotes", (["600519"],), {})
    assert live_ok is None


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((["600519"],), {"curr_date": ""}),
        ((["600519"],), {"curr_date": "not-a-date"}),
        ((["600519"],), {"curr_date": "2030-01-01"}),
        ((["600519"], ""), {}),
        ((["600519"], "not-a-date"), {}),
        ((["600519"], "2030-01-01"), {}),
    ],
)
def test_route_realtime_refuses_bad_curr_date_without_provider_call(args, kwargs):
    from tradingagents.dataflows import interface as iface

    provider = MagicMock()
    provider.get_realtime_quotes.side_effect = AssertionError(
        "provider must not be called"
    )
    with patch.object(iface, "_registry") as reg, \
         patch.object(iface, "get_vendor", return_value="cn_akshare"):
        reg.list_names.return_value = ["cn_akshare"]
        reg.get.return_value = provider
        out = iface.route_to_vendor("get_realtime_quotes", *args, **kwargs)

    assert out.startswith("【数据获取失败】")
    assert "本项不可用" in out
    provider.get_realtime_quotes.assert_not_called()
    reg.get.assert_not_called()


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((["600519"],), {}),
        ((["600519"],), {"curr_date": None}),
        ((["600519"], None), {}),
    ],
)
def test_route_realtime_missing_curr_date_still_routes_to_provider(args, kwargs):
    from tradingagents.dataflows import interface as iface

    provider = MagicMock()
    provider.get_realtime_quotes.return_value = "## live quotes"
    with patch.object(iface, "_registry") as reg, \
         patch.object(iface, "get_vendor", return_value="cn_akshare"):
        reg.list_names.return_value = ["cn_akshare"]
        reg.get.return_value = provider
        out = iface.route_to_vendor("get_realtime_quotes", *args, **kwargs)

    assert out == "## live quotes"
    provider.get_realtime_quotes.assert_called_once()


# ── Router duplicate-bar-conflict semantics ────────────────────────────


def test_route_duplicate_conflict_is_explicit_unavailable_without_fallback():
    """A duplicate-bar conflict must refuse loudly, not fall back to another vendor.

    Falling back would silently mask the conflict by re-reading the same date
    from a different source, whose row order is equally arbitrary. The router
    therefore treats ``DuplicateBarConflictError`` as an explicit refusal and
    never calls the next provider in the chain.
    """
    from tradingagents.dataflows import interface as iface
    from tradingagents.dataflows.trade_calendar import DuplicateBarConflictError

    assert issubclass(DuplicateBarConflictError, ValueError)

    conflicting = MagicMock()
    conflicting.is_placeholder = False
    conflicting.get_stock_data.side_effect = DuplicateBarConflictError(
        "duplicate daily bars with conflicting OHLCV, cannot choose "
        "deterministically for date(s): 2026-07-28"
    )
    second = MagicMock()
    second.is_placeholder = False
    second.get_stock_data.side_effect = AssertionError(
        "fallback provider must not be called after a duplicate-bar conflict"
    )
    with patch.object(iface, "_registry") as reg, \
         patch.object(iface, "get_vendor", return_value="cn_akshare"):
        reg.list_names.return_value = ["cn_akshare", "cn_investoday"]
        reg.get.side_effect = lambda name: {
            "cn_akshare": conflicting,
            "cn_investoday": second,
        }[name]
        out = iface.route_to_vendor(
            "get_stock_data", "600519", "2026-07-01", "2026-07-28"
        )

    assert out.startswith("【数据获取失败】")
    assert "duplicate daily bars with conflicting OHLCV" in out
    assert "本项不可用" in out
    conflicting.get_stock_data.assert_called_once()
    second.get_stock_data.assert_not_called()


def test_route_real_akshare_conflict_propagates_without_vendor_fallback():
    """End-to-end: the real AkShare provider's source-switch conflict reaches the router.

    The provider's internal multi-source fallback must re-raise
    ``DuplicateBarConflictError`` instead of swallowing it and switching to
    Sina/Tencent; the router then refuses loudly and never calls the next vendor.
    """
    from tradingagents.dataflows import interface as iface

    provider = CnAkshareProvider()
    fake_ak = _MultiSourceAk(conflict_source="stock_zh_a_hist")
    provider._ak = lambda: fake_ak

    second = MagicMock()
    second.is_placeholder = False
    second.get_stock_data.side_effect = AssertionError(
        "fallback vendor must not be called after a real AkShare duplicate conflict"
    )
    with patch.object(iface, "_registry") as reg, \
         patch.object(iface, "get_vendor", return_value="cn_akshare"):
        reg.list_names.return_value = ["cn_akshare", "cn_investoday"]
        reg.get.side_effect = lambda name: {
            "cn_akshare": provider,
            "cn_investoday": second,
        }[name]
        out = iface.route_to_vendor(
            "get_stock_data", "600519", "2026-07-01", "2026-07-28"
        )

    assert out.startswith("【数据获取失败】")
    assert "duplicate daily bars with conflicting OHLCV" in out
    assert "本项不可用" in out
    # Only the Eastmoney source was contacted, once; Sina/Tencent untouched.
    assert fake_ak.calls == ["stock_zh_a_hist"]
    second.get_stock_data.assert_not_called()


def test_route_generic_value_error_still_falls_back():
    """Only the duplicate-conflict subclass is a hard refusal.

    A plain ``ValueError`` is a provider-level data/parsing problem and keeps the
    pre-existing fallback contract, so unrelated failures can still be retried on
    the next vendor without conflating them with duplicate-bar conflicts.
    """
    from tradingagents.dataflows import interface as iface
    from tradingagents.dataflows.providers.base import ProviderResourcePolicy

    broken = MagicMock()
    broken.is_placeholder = False
    broken.get_stock_data.side_effect = ValueError("vendor returned unexpected shape")
    second = MagicMock()
    second.is_placeholder = False
    second.get_stock_data.return_value = "## fallback csv"
    with patch.object(iface, "_registry") as reg, \
         patch.object(iface, "get_vendor", return_value="cn_akshare"):
        reg.list_names.return_value = ["cn_akshare", "cn_investoday"]
        reg.get.side_effect = lambda name: {
            "cn_akshare": broken,
            "cn_investoday": second,
        }[name]
        # DAV-44: each vendor carries its own resource policy (retry budget).
        # max_retries=0 keeps the original "generic error is not retried on the
        # same vendor" expectation from the pre-DAV-44 router contract.
        reg.resource_policy.return_value = ProviderResourcePolicy(
            timeout_seconds=1.0, max_retries=0, max_concurrency=1
        )
        out = iface.route_to_vendor(
            "get_stock_data", "600519", "2026-07-01", "2026-07-28"
        )

    assert out == "## fallback csv"
    broken.get_stock_data.assert_called_once()
    second.get_stock_data.assert_called_once()
