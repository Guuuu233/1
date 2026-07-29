"""3c-2: internal providers must not soft-default missing analysis dates to today."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.providers.cn_investoday_provider import CnInvestodayProvider


FAIL_MARKERS = ("【数据获取失败】", "缺少")


def _assert_missing_date_failure(text: str) -> None:
    assert isinstance(text, str) and text.strip()
    assert any(m in text for m in FAIL_MARKERS)
    # Must not look like a successful data table / live dump
    assert "涨停家数" not in text
    assert "融资余额" not in text
    assert "预告类型" not in text
    assert "股票简称" not in text


@pytest.fixture
def ak_provider() -> CnAkshareProvider:
    p = CnAkshareProvider()
    # Any network touch is a regression: missing date must fail before I/O.
    p._ak = MagicMock(side_effect=AssertionError("must not call network without date"))
    return p


def test_akshare_lhb_missing_date(ak_provider):
    out = ak_provider.get_lhb_detail("600519", date=None)
    _assert_missing_date_failure(out)
    assert "龙虎榜" in out or "date" in out.lower() or "curr_date" in out


def test_akshare_lhb_empty_string_date(ak_provider):
    out = ak_provider.get_lhb_detail("600519", date="")
    _assert_missing_date_failure(out)


def test_akshare_zt_pool_missing_date(ak_provider):
    out = ak_provider.get_zt_pool(date=None)
    _assert_missing_date_failure(out)


def test_akshare_margin_missing_date(ak_provider):
    out = ak_provider.get_margin_trading("600519", curr_date=None)
    _assert_missing_date_failure(out)


def test_akshare_restricted_missing_date(ak_provider):
    # restricted still constructs source after normalize; patch normalize only
    ak_provider._normalize_symbol = MagicMock(return_value="600519")
    # _ak will be called only after date check — ensure date check first
    # restore _ak to not raise until after (date check returns first)
    out = ak_provider.get_restricted_release("600519", curr_date=None)
    _assert_missing_date_failure(out)


def test_akshare_earnings_forecast_missing_date(ak_provider):
    ak_provider._normalize_symbol = MagicMock(return_value="600519")
    out = ak_provider.get_earnings_forecast("600519", curr_date=None)
    _assert_missing_date_failure(out)


def test_akshare_insider_missing_date(ak_provider):
    out = ak_provider.get_insider_transactions("600519", curr_date=None)
    _assert_missing_date_failure(out)
    assert "内部层不得默认今天" in out or "缺少 curr_date" in out


def test_investoday_financial_missing_date():
    p = CnInvestodayProvider()
    p._require_api_key = MagicMock(return_value="k")
    p._normalize_stock_code = MagicMock(return_value="600519")
    p._fetch_paged_list = MagicMock(
        side_effect=AssertionError("must not fetch without curr_date")
    )
    for method, title in (
        (p.get_balance_sheet, "资产负债表"),
        (p.get_cashflow, "现金流量表"),
        (p.get_income_statement, "利润表"),
    ):
        out = method("600519", freq="quarterly", curr_date=None)
        _assert_missing_date_failure(out)
        assert title in out or "curr_date" in out


def test_investoday_insider_missing_date():
    p = CnInvestodayProvider()
    p._require_api_key = MagicMock(return_value="k")
    p._normalize_stock_code = MagicMock(return_value="600519")
    p._fetch_paged_list = MagicMock(
        side_effect=AssertionError("must not fetch without curr_date")
    )
    out = p.get_insider_transactions("600519", curr_date=None)
    _assert_missing_date_failure(out)


def test_investoday_insider_live_window_uses_curr_date_not_now():
    """Live path (today) must end the API window at curr_date, not wall-clock now."""
    p = CnInvestodayProvider()
    p._require_api_key = MagicMock(return_value="k")
    p._normalize_stock_code = MagicMock(return_value="600519")
    p._resolve_base_url = MagicMock(return_value="https://example.invalid")
    captured: dict = {}

    def _capture(path, params, api_key, base_url):
        captured.update(params)
        return []

    p._fetch_paged_list = _capture  # type: ignore[assignment]

    with patch(
        "tradingagents.dataflows.providers.cn_investoday_provider.snapshot_historical_refusal",
        return_value=None,
    ):
        out = p.get_insider_transactions("600519", curr_date="2026-07-29")

    assert captured.get("endDate") == "2026-07-29"
    assert "未获取到数据" in out or "Insider" in out
