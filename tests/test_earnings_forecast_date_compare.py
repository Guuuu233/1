"""3c-1: earnings_forecast truncates by datetime, not string compare."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider


def _provider_with_df(df: pd.DataFrame) -> CnAkshareProvider:
    p = CnAkshareProvider()
    ak = MagicMock()
    ak.stock_yjyg_em.return_value = df
    p._ak = MagicMock(return_value=ak)
    p._normalize_symbol = MagicMock(return_value="600519")
    return p


def test_unpadded_announce_date_not_kept_by_string_quirk():
    """'2026-4-5' must not survive via string compare against '2026-04-01'."""
    df = pd.DataFrame(
        [
            {
                "股票代码": "600519",
                "预告类型": "预增",
                "业绩变动": "+10%",
                "业绩变动原因": "test",
                "公告日期": "2026-4-5",  # unpadded; string '>' vs 2026-04-01 is True wrongly path-dependent
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
    p = _provider_with_df(df)
    out = p.get_earnings_forecast("600519", curr_date="2026-04-01")
    assert "2026-4-5" not in out
    assert "2026-03-20" in out


def test_compact_yyyymmdd_announce_date_parsed():
    df = pd.DataFrame(
        [
            {
                "股票代码": "600519",
                "预告类型": "预增",
                "业绩变动": "+10%",
                "业绩变动原因": "x",
                "公告日期": "20260405",
            },
            {
                "股票代码": "600519",
                "预告类型": "略增",
                "业绩变动": "+5%",
                "业绩变动原因": "y",
                "公告日期": "20260315",
            },
        ]
    )
    p = _provider_with_df(df)
    out = p.get_earnings_forecast("600519", curr_date="2026-04-01")
    assert "20260405" not in out  # after cutoff
    assert "20260315" in out


def test_unparseable_announce_date_skipped():
    df = pd.DataFrame(
        [
            {
                "股票代码": "600519",
                "预告类型": "预增",
                "业绩变动": "+10%",
                "业绩变动原因": "x",
                "公告日期": "not-a-date",
            }
        ]
    )
    p = _provider_with_df(df)
    out = p.get_earnings_forecast("600519", curr_date="2026-04-01")
    assert "not-a-date" not in out
    assert "无可用预告记录" in out or "暂无业绩" in out
