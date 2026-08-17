"""Regression: time-series row selection must not depend on vendor row order.

Covers both the shared helper and the three call sites fixed in commit 1a:
get_news / get_individual_fund_flow / get_shareholder_count.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.dataflows.utils import chronological, take_latest
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider


def _shareholder_rows() -> pd.DataFrame:
    """Enough history so latest-4 excludes ancient 2013 rows."""
    return pd.DataFrame(
        {
            "股东户数统计截止日": [
                "2013-03-22",
                "2013-03-31",
                "2013-06-30",
                "2013-09-30",
                "2025-03-31",
                "2025-06-30",
                "2025-09-30",
                "2025-12-31",
                "2026-03-31",
            ],
            "股东户数公告日期": [
                "2013-03-29",
                "2013-04-18",
                "2013-08-31",
                "2013-10-16",
                "2025-04-30",
                "2025-08-13",
                "2025-10-30",
                "2026-04-17",
                "2026-04-25",
            ],
            "股东户数-本次": [69331, 68539, 53045, 75369, 192430, 220658, 238512, 255892, 243159],
            "股东户数-增减比例": [39.0, -1.1, -22.6, 42.0, -7.4, 14.6, 8.0, 7.2, -4.9],
            "户均持股市值": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )


def _asc_shareholder_df() -> pd.DataFrame:
    return _shareholder_rows()


def _desc_shareholder_df() -> pd.DataFrame:
    return _shareholder_rows().iloc[::-1].reset_index(drop=True)


def _news_rows() -> pd.DataFrame:
    """Six dated news rows; titles encode chronological labels for assertions."""
    return pd.DataFrame(
        {
            "新闻标题": ["t0", "t1", "t2", "t3", "t4", "t5"],
            "文章来源": ["s"] * 6,
            "新闻内容": ["c"] * 6,
            "新闻链接": ["http://x"] * 6,
            "发布时间": [
                "2026-07-20 11:00:00",
                "2026-07-28 21:23:51",  # newest
                "2026-07-21 09:00:00",
                "2026-07-28 16:34:03",
                "2026-07-22 10:00:00",
                "2026-07-28 12:08:00",
            ],
        }
    )


def _asc_news_df() -> pd.DataFrame:
    return _news_rows().sort_values("发布时间").reset_index(drop=True)


def _desc_news_df() -> pd.DataFrame:
    return _news_rows().sort_values("发布时间", ascending=False).reset_index(drop=True)


def _fund_flow_rows() -> pd.DataFrame:
    dates = pd.date_range("2026-01-20", periods=20, freq="B")
    return pd.DataFrame(
        {
            "日期": dates.date,
            "主力净流入-净额": list(range(1, 21)),
        }
    )


def test_take_latest_asc_and_desc_same_result():
    asc = pd.DataFrame({"d": ["2020-01-01", "2021-01-01", "2022-01-01"], "v": [1, 2, 3]})
    desc = asc.iloc[::-1].reset_index(drop=True)
    a = take_latest(asc, "d", 2)
    b = take_latest(desc, "d", 2)
    assert list(a["v"]) == [3, 2]
    assert list(b["v"]) == [3, 2]


def test_chronological_renders_oldest_to_newest():
    latest = take_latest(
        pd.DataFrame({"d": ["2020-01-01", "2022-01-01", "2021-01-01"], "v": [1, 3, 2]}),
        "d",
        3,
    )
    shown = chronological(latest, "d")
    assert list(shown["v"]) == [1, 2, 3]


def test_take_latest_rejects_bad_n_and_missing_col():
    df = pd.DataFrame({"d": ["2020-01-01"], "v": [1]})
    with pytest.raises(KeyError):
        take_latest(df, "missing", 1)
    with pytest.raises(ValueError):
        take_latest(df, "d", 0)


class _ShareholderFixtureProvider(CnAkshareProvider):
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def _ak(self):
        class _Ak:
            def __init__(self, frame):
                self._frame = frame

            def stock_zh_a_gdhs_detail_em(self, symbol):
                return self._frame.copy()

        return _Ak(self._frame)


class _NewsFixtureProvider(CnAkshareProvider):
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def _ak(self):
        class _Ak:
            def __init__(self, frame):
                self._frame = frame

            def stock_news_em(self, symbol):
                return self._frame.copy()

        return _Ak(self._frame)


class _FundFlowFixtureProvider(CnAkshareProvider):
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def _ak(self):
        class _Ak:
            def __init__(self, frame):
                self._frame = frame

            def stock_individual_fund_flow(self, stock, market):
                return self._frame.copy()

        return _Ak(self._frame)


@pytest.mark.parametrize("frame_factory", [_asc_shareholder_df, _desc_shareholder_df])
def test_shareholder_count_returns_latest_periods_chronological(frame_factory):
    """Function-level: asc and desc inputs yield the same latest-4, oldest→newest."""
    provider = _ShareholderFixtureProvider(frame_factory())
    text = provider.get_shareholder_count("600519", curr_date="2026-07-28")
    assert "2026-03-31" in text
    assert "2025-12-31" in text
    assert "2025-09-30" in text
    assert "2025-06-30" in text
    assert "2013-03-22" not in text
    assert "2013-03-31" not in text
    # render order: older first, newest last
    assert text.index("2025-06-30") < text.index("2025-09-30") < text.index("2026-03-31")


@pytest.mark.parametrize("frame_factory", [_asc_news_df, _desc_news_df])
def test_news_selects_latest_and_renders_chronologically(frame_factory):
    """Function-level get_news: asc/desc input → latest window, ascending display."""
    provider = _NewsFixtureProvider(frame_factory())
    text = provider.get_news("600519", "2026-07-01", "2026-07-28")
    # oldest among selected first, newest last
    pos_t0 = text.find("### t0 ")  # 2026-07-20
    pos_t5 = text.find("### t5 ")  # 2026-07-28 12:08
    pos_t1 = text.find("### t1 ")  # 2026-07-28 21:23 newest
    assert pos_t0 != -1 and pos_t5 != -1 and pos_t1 != -1
    assert pos_t0 < pos_t5 < pos_t1


@pytest.mark.parametrize(
    "frame",
    [
        _fund_flow_rows(),
        _fund_flow_rows().iloc[::-1].reset_index(drop=True),
    ],
)
def test_fund_flow_latest_five_independent_of_order(frame):
    """Function-level fund flow: asc/desc same latest 5, chronological render."""
    provider = _FundFlowFixtureProvider(frame)
    latest = sorted(frame["日期"])[-1]
    second = sorted(frame["日期"])[-2]
    oldest = sorted(frame["日期"])[0]
    text = provider.get_individual_fund_flow("600519", curr_date=str(latest))
    assert str(latest) in text
    assert str(second) in text
    assert str(oldest) not in text
    # Header may mention latest data day; compare order only inside the table body.
    body = text.split("：", 1)[-1]
    assert body.index(str(second)) < body.index(str(latest))


def test_northbound_deprecated_no_network_and_no_old_dates():
    """Commit 1b: daily northbound is hard-deprecated; no vendor call path."""

    class _Probe(CnAkshareProvider):
        def _ak(self):
            raise AssertionError("get_northbound_flow must not touch akshare")

    text = _Probe().get_northbound_flow("600519", curr_date="2026-07-28")
    assert "停止披露" in text
    assert "不可用" in text
    assert "季度" in text
    assert "2017-03-16" not in text
    assert "2017" not in text
