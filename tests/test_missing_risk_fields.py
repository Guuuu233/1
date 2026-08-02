"""Missing risk fields must not default to zero / 'safe'."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.trade_calendar import cn_today_str


class _PledgeAk:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def stock_gpzy_pledge_ratio_em(self):
        return self.df


class _MarginAk:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def stock_margin_detail_sse(self, date=None):
        return self.df

    def stock_margin_detail_szse(self, date=None):
        return self.df


class _PledgeProvider(CnAkshareProvider):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def _ak(self):
        return _PledgeAk(self._df)

    def _normalize_symbol(self, symbol: str) -> str:
        return str(symbol).zfill(6)


class _MarginProvider(CnAkshareProvider):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def _ak(self):
        return _MarginAk(self._df)

    def _normalize_symbol(self, symbol: str) -> str:
        return str(symbol).zfill(6)


def test_share_pledge_missing_ratio_is_unassessed_not_safe():
    df = pd.DataFrame(
        {
            "股票代码": ["600519"],
            # 质押比例 column absent on purpose
            "质押笔数": [3],
            "所属行业": ["白酒"],
        }
    )
    out = _PledgeProvider(df).get_share_pledge("600519", curr_date=cn_today_str())
    assert "未排查" in out
    assert "安全" not in out
    assert "质押比例" in out


def test_share_pledge_blank_ratio_is_unassessed():
    df = pd.DataFrame(
        {
            "股票代码": ["600519"],
            "质押比例": [""],
            "质押笔数": [3],
            "所属行业": ["白酒"],
        }
    )
    out = _PledgeProvider(df).get_share_pledge("600519", curr_date=cn_today_str())
    assert "未排查" in out
    assert "安全" not in out


def test_share_pledge_present_ratio_still_thresholds():
    df = pd.DataFrame(
        {
            "股票代码": ["600519"],
            "质押比例": ["12.5"],
            "质押笔数": [2],
            "所属行业": ["白酒"],
        }
    )
    out = _PledgeProvider(df).get_share_pledge("600519", curr_date=cn_today_str())
    assert "整体质押比例：12.5%" in out
    assert "未排查" not in out


def test_margin_missing_balance_is_unassessed_not_zero():
    df = pd.DataFrame(
        {
            "标的证券代码": ["600519"],
            # 融资余额 missing
            "融资买入额": [100],
            "融券余量": [1],
        }
    )
    provider = _MarginProvider(df)
    with patch(
        "tradingagents.dataflows.providers.cn_akshare_provider.fetch_with_date_fallback"
    ) as mock_fb:
        # Exercise the inner _fetch_one path by calling it through a real fallback
        # with a fixed day window.
        from tradingagents.dataflows.trade_calendar import DateFetchResult

        def _run(fetch_fn, date_str, max_back=5, start_offset=0):
            data = fetch_fn("2026-07-27")
            return DateFetchResult(
                ok=True,
                data=data,
                as_of="2026-07-27",
                request_date=date_str,
                attempted=["2026-07-27"],
            )

        mock_fb.side_effect = _run
        out = provider.get_margin_trading("600519", curr_date="2026-07-29")

    assert "未排查" in out
    assert "融资余额" in out
    assert "融资余额: 0" not in out


def test_margin_present_fields_render_values():
    df = pd.DataFrame(
        {
            "标的证券代码": ["600519"],
            "融资余额": [12345],
            "融资买入额": [100],
            "融券余量": [1],
        }
    )
    provider = _MarginProvider(df)
    with patch(
        "tradingagents.dataflows.providers.cn_akshare_provider.fetch_with_date_fallback"
    ) as mock_fb:
        from tradingagents.dataflows.trade_calendar import DateFetchResult

        def _run(fetch_fn, date_str, max_back=5, start_offset=0):
            data = fetch_fn("2026-07-27")
            return DateFetchResult(
                ok=True,
                data=data,
                as_of="2026-07-27",
                request_date=date_str,
                attempted=["2026-07-27"],
            )

        mock_fb.side_effect = _run
        out = provider.get_margin_trading("600519", curr_date="2026-07-29")

    assert "融资余额: 12345" in out
    assert "未排查" not in out
