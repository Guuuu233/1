"""Earnings-forecast report period = latest closed disclosure window."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from tradingagents.dataflows.financial_announce import (
    resolve_earnings_forecast_report_period,
)
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider


@pytest.mark.parametrize(
    "curr_date,expected",
    [
        ("2026-07-29", "20260630"),  # H1 window closed Jul 15
        ("2026-07-15", "20260630"),  # exact close day
        ("2026-07-14", "20260331"),  # day before H1 close → still Q1
        ("2026-04-15", "20260331"),
        ("2026-04-14", "20251231"),  # before Q1 close → prior annual
        ("2026-01-31", "20251231"),
        ("2026-01-30", "20250930"),
        ("2026-10-15", "20260930"),
        ("2026-02-01", "20251231"),
    ],
)
def test_resolve_closed_window_period(curr_date, expected):
    assert resolve_earnings_forecast_report_period(curr_date) == expected


def test_resolve_rejects_unparseable():
    with pytest.raises(ValueError):
        resolve_earnings_forecast_report_period("not-a-date")


def _provider_capturing_date(df: pd.DataFrame | None = None):
    p = CnAkshareProvider()
    ak = MagicMock()
    ak.stock_yjyg_em.return_value = (
        df if df is not None else pd.DataFrame(columns=["股票代码", "预告类型", "业绩变动", "业绩变动原因", "公告日期"])
    )
    p._ak = MagicMock(return_value=ak)
    p._normalize_symbol = MagicMock(return_value="600519")
    return p, ak


def test_provider_queries_h1_not_prior_annual_in_july():
    p, ak = _provider_capturing_date(
        pd.DataFrame(
            [
                {
                    "股票代码": "600519",
                    "预告类型": "预增",
                    "业绩变动": "+10%",
                    "业绩变动原因": "seasonal",
                    "公告日期": "2026-07-10",
                }
            ]
        )
    )
    out = p.get_earnings_forecast("600519", curr_date="2026-07-29")
    ak.stock_yjyg_em.assert_called_once_with(date="20260630")
    assert "查询报告期 = 20260630" in out
    assert "2026H1" in out
    assert "预增" in out


def test_provider_empty_market_pool_is_unknown_not_no_forecast():
    p, ak = _provider_capturing_date(pd.DataFrame())  # empty market pool
    out = p.get_earnings_forecast("600519", curr_date="2026-07-29")
    assert "查询报告期 = 20260630" in out
    assert "未知" in out or "数据获取失败" in out
    assert "确认无预告" not in out


def test_provider_ticker_absent_is_confirmed_no_forecast():
    # Non-empty market pool but ticker missing
    df = pd.DataFrame(
        [
            {
                "股票代码": "000001",
                "预告类型": "预增",
                "业绩变动": "+1%",
                "业绩变动原因": "x",
                "公告日期": "2026-07-01",
            }
        ]
    )
    p, ak = _provider_capturing_date(df)
    out = p.get_earnings_forecast("600519", curr_date="2026-07-29")
    assert "查询报告期 = 20260630" in out
    assert "确认无预告" in out or "暂无业绩" in out
    assert "未知" not in out


def test_existing_announce_date_truncation_still_works():
    df = pd.DataFrame(
        [
            {
                "股票代码": "600519",
                "预告类型": "预增",
                "业绩变动": "+10%",
                "业绩变动原因": "test",
                "公告日期": "2026-4-5",
            },
            {
                "股票代码": "600519",
                "预告类型": "略增",
                "业绩变动": "+5%",
                "业绩变动原因": "old",
                "公告日期": "2026-03-20",
            },
        ]
    )
    p, ak = _provider_capturing_date(df)
    out = p.get_earnings_forecast("600519", curr_date="2026-04-01")
    # On 2026-04-01, closed window is still prior annual 20251231 (Q1 closes Apr 15)
    ak.stock_yjyg_em.assert_called_once_with(date="20251231")
    assert "2026-4-5" not in out
    assert "2026-03-20" in out
