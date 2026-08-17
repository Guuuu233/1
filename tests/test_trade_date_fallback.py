"""Shared CN trading calendar + date-scoped fetch fallback.

Covers:
- normalize_to_trading_day only rolls backward
- calendar hard-fails instead of weekday degradation for date queries
- fetch_with_date_fallback success / full-failure messaging
- function-level wiring for margin / lhb / zt_pool
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows import trade_calendar as tc
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider


@pytest.fixture(autouse=True)
def _clear_calendar_cache():
    tc.clear_cn_trade_date_cache()
    yield
    tc.clear_cn_trade_date_cache()


def _seed_calendar(days: list[str]) -> None:
    dates = [date.fromisoformat(d) for d in days]
    tc._TRADE_DATES_CACHE["dates"] = dates
    tc._TRADE_DATES_CACHE["dates_set"] = set(dates)
    tc._TRADE_DATES_CACHE["loaded_at"] = 1e18  # far future: never expire in tests


# ── normalize / calendar ──────────────────────────────────────────────


def test_normalize_weekend_rolls_back_to_friday():
    _seed_calendar(
        [
            "2026-07-24",
            "2026-07-27",
            "2026-07-28",
            "2026-07-29",
            "2026-07-30",
            "2026-07-31",
        ]
    )
    assert tc.normalize_to_trading_day("2026-07-26") == "2026-07-24"  # Sunday -> Friday
    assert tc.normalize_to_trading_day("2026-07-25") == "2026-07-24"  # Saturday -> Friday


def test_normalize_holiday_rolls_back_not_forward():
    # Simulate National Day gap: 10-01..10-07 closed, last session 09-30
    _seed_calendar(
        [
            "2026-09-29",
            "2026-09-30",
            "2026-10-08",
            "2026-10-09",
        ]
    )
    assert tc.normalize_to_trading_day("2026-10-03") == "2026-09-30"
    assert tc.normalize_to_trading_day("2026-10-01") == "2026-09-30"
    # never later than input
    for d in ("2026-09-30", "2026-10-03", "2026-10-08"):
        out = tc.normalize_to_trading_day(d)
        assert out <= d


def test_normalize_trading_day_identity():
    _seed_calendar(["2026-07-27", "2026-07-28", "2026-07-29"])
    assert tc.normalize_to_trading_day("2026-07-28") == "2026-07-28"


def test_normalize_never_later_than_input_property():
    _seed_calendar(
        [
            "2026-07-20",
            "2026-07-21",
            "2026-07-22",
            "2026-07-23",
            "2026-07-24",
            "2026-07-27",
            "2026-07-28",
            "2026-07-29",
        ]
    )
    for d in (
        "2026-07-20",
        "2026-07-25",
        "2026-07-26",
        "2026-07-28",
        "2026-07-29",
    ):
        out = tc.normalize_to_trading_day(d)
        assert out <= d


def test_normalize_hard_fails_when_calendar_unavailable():
    # empty cache + loader returns empty -> hard fail, no weekday degradation
    tc.clear_cn_trade_date_cache()

    def _boom():
        raise RuntimeError("network down")

    with patch.object(tc, "_fetch_cn_trade_dates_from_akshare", side_effect=_boom), \
         patch.object(tc, "_fetch_cn_trade_dates_from_fuyao", side_effect=_boom):
        with pytest.raises(tc.TradeCalendarUnavailableError) as ei:
            tc.normalize_to_trading_day("2026-07-26")
    assert "交易日历不可用" in str(ei.value)


# ── 日历 fallback：akshare 失败 → fuyao 对照 → 缓存兜底 ───────────────


@pytest.mark.parametrize("hour", [8, 10, 12])
def test_resolve_default_fails_closed_when_both_calendars_fail(hour):
    tc.clear_cn_trade_date_cache()
    frozen = datetime(2026, 8, 12, hour, 0, tzinfo=tc.CN_TZ)
    with patch.object(tc, "_fetch_cn_trade_dates_from_akshare", side_effect=RuntimeError("ak down")), \
         patch.object(tc, "_fetch_cn_trade_dates_from_fuyao", side_effect=RuntimeError("fuyao down")):
        with pytest.raises(tc.TradeCalendarUnavailableError):
            tc.resolve_cn_analysis_date(None, now=frozen)


def test_resolve_default_rejects_stale_cached_calendar_before_close():
    _seed_calendar(["2026-07-20", "2026-07-21"])
    frozen = datetime(2026, 8, 12, 10, 0, tzinfo=tc.CN_TZ)
    with patch.object(tc, "_fetch_cn_trade_dates_from_akshare", side_effect=RuntimeError("ak down")), \
         patch.object(tc, "_fetch_cn_trade_dates_from_fuyao", side_effect=RuntimeError("fuyao down")):
        with pytest.raises(tc.TradeCalendarUnavailableError, match="早于请求日"):
            tc.resolve_cn_analysis_date(None, now=frozen)


def test_resolve_explicit_weekend_preserves_calendar_rollback():
    _seed_calendar(["2026-08-07", "2026-08-10", "2026-08-11"])
    assert tc.resolve_cn_analysis_date("2026-08-09", explicit=True) == "2026-08-07"


def test_calendar_falls_back_to_fuyao_when_akshare_fails():
    """akshare 交易日历失败 → 用 fuyao 近一年日历作在线对照，并写入缓存。"""
    fuyao_dates = [date(2026, 8, 3), date(2026, 8, 4), date(2026, 8, 5)]
    with patch.object(
        tc, "_fetch_cn_trade_dates_from_akshare", side_effect=RuntimeError("akshare down")
    ), patch.object(
        tc, "_fetch_cn_trade_dates_from_fuyao", return_value=fuyao_dates
    ):
        dates, dates_set = tc._load_cn_trade_dates()
    assert dates == fuyao_dates
    assert dates_set == set(fuyao_dates)
    assert tc._TRADE_DATES_CACHE["dates"] == fuyao_dates


def test_calendar_uses_cached_dates_when_akshare_and_fuyao_fail():
    """akshare 与 fuyao 均失败 → 用此前成功日历（已过期缓存）兜底。"""
    cached = [date(2026, 8, 3), date(2026, 8, 4)]
    tc._TRADE_DATES_CACHE["dates"] = cached
    tc._TRADE_DATES_CACHE["dates_set"] = set(cached)
    tc._TRADE_DATES_CACHE["loaded_at"] = 0.0  # 超过 TTL，强制重新加载
    with patch.object(
        tc, "_fetch_cn_trade_dates_from_akshare", side_effect=RuntimeError("akshare down")
    ), patch.object(
        tc, "_fetch_cn_trade_dates_from_fuyao", side_effect=RuntimeError("fuyao down")
    ):
        dates, dates_set = tc._load_cn_trade_dates()
    assert dates == cached
    assert dates_set == set(cached)


def test_calendar_returns_empty_when_all_sources_fail_and_no_cache():
    """akshare 与 fuyao 均失败且无缓存 → 空容器；require_cn_trade_dates 显式报错。"""
    tc.clear_cn_trade_date_cache()
    with patch.object(
        tc, "_fetch_cn_trade_dates_from_akshare", side_effect=RuntimeError("akshare down")
    ), patch.object(
        tc, "_fetch_cn_trade_dates_from_fuyao", side_effect=RuntimeError("fuyao down")
    ):
        dates, dates_set = tc._load_cn_trade_dates()
        assert dates == []
        assert dates_set == set()
        with pytest.raises(tc.TradeCalendarUnavailableError):
            tc.require_cn_trade_dates()


# ── fetch_with_date_fallback ──────────────────────────────────────────


def test_fetch_fallback_succeeds_on_third_day_and_sets_as_of():
    _seed_calendar(
        [
            "2026-07-22",
            "2026-07-23",
            "2026-07-24",
            "2026-07-27",
            "2026-07-28",
            "2026-07-29",
        ]
    )
    calls: list[str] = []

    def fetch_fn(day: str):
        calls.append(day)
        if len(calls) < 3:
            raise tc.DateDataUnavailable(f"{day} not ready")
        return f"payload-{day}"

    result = tc.fetch_with_date_fallback(fetch_fn, "2026-07-29", max_back=5)
    assert result.ok is True
    assert result.as_of == "2026-07-27"
    assert result.request_date == "2026-07-29"
    assert result.data == "payload-2026-07-27"
    assert calls == ["2026-07-29", "2026-07-28", "2026-07-27"]
    header = result.date_header()
    assert "【数据日期】2026-07-27" in header
    assert "请求 2026-07-29" in header
    assert "已回退" in header


def test_fetch_fallback_all_fail_includes_attempted_range():
    _seed_calendar(
        [
            "2026-07-22",
            "2026-07-23",
            "2026-07-24",
            "2026-07-27",
            "2026-07-28",
            "2026-07-29",
        ]
    )

    def fetch_fn(day: str):
        raise tc.DateDataUnavailable(f"{day} empty")

    result = tc.fetch_with_date_fallback(fetch_fn, "2026-07-29", max_back=5)
    assert result.ok is False
    assert "已尝试 2026-07-29 至 2026-07-23 共 5 个交易日，均无数据" in result.error
    assert result.attempted == [
        "2026-07-29",
        "2026-07-28",
        "2026-07-27",
        "2026-07-24",
        "2026-07-23",
    ]


def test_fetch_fallback_calendar_unavailable_message():
    tc.clear_cn_trade_date_cache()
    with patch.object(
        tc,
        "_fetch_cn_trade_dates_from_akshare",
        side_effect=RuntimeError("down"),
    ), patch.object(
        tc,
        "_fetch_cn_trade_dates_from_fuyao",
        side_effect=RuntimeError("down"),
    ):
        result = tc.fetch_with_date_fallback(lambda d: d, "2026-07-29", max_back=3)
    assert result.ok is False
    assert "交易日历不可用" in result.error


# ── function-level: margin / lhb / zt ─────────────────────────────────


class _SeqAk:
    """Minimal akshare stub that serves per-date tables or raises."""

    def __init__(self, tables: dict[str, pd.DataFrame | Exception]):
        self.tables = tables
        self.calls: list[tuple[str, str]] = []

    def _get(self, api: str, day_yyyymmdd: str):
        day = f"{day_yyyymmdd[:4]}-{day_yyyymmdd[4:6]}-{day_yyyymmdd[6:8]}"
        self.calls.append((api, day))
        val = self.tables.get(day)
        if isinstance(val, Exception):
            raise val
        if val is None:
            raise ValueError("Length mismatch: Expected axis has 0 elements")
        return val.copy()

    def stock_margin_detail_sse(self, date: str):
        return self._get("margin_sse", date)

    def stock_margin_detail_szse(self, date: str):
        return self._get("margin_szse", date)

    def stock_lhb_detail_em(self, start_date: str, end_date: str):
        return self._get("lhb", start_date)

    def stock_zt_pool_em(self, date: str):
        return self._get("zt", date)


class _FixtureProvider(CnAkshareProvider):
    def __init__(self, ak: _SeqAk):
        self._ak_obj = ak

    def _ak(self):
        return self._ak_obj


def test_get_margin_trading_rolls_back_and_exposes_as_of_date():
    _seed_calendar(
        [
            "2026-07-22",
            "2026-07-23",
            "2026-07-24",
            "2026-07-27",
            "2026-07-28",
            "2026-07-29",
        ]
    )
    ok_df = pd.DataFrame(
        {
            "标的证券代码": ["600519"],
            "融资余额": [100],
            "融资买入额": [10],
            "融券余量": [1],
        }
    )
    ak = _SeqAk(
        {
            "2026-07-29": ValueError("empty assign"),
            "2026-07-28": ValueError("empty assign"),
            "2026-07-27": ok_df,
        }
    )
    text = _FixtureProvider(ak).get_margin_trading("600519", curr_date="2026-07-29")
    assert "【数据日期】2026-07-27" in text
    assert "请求 2026-07-29" in text
    assert "已回退" in text
    assert "融资余额: 100" in text
    assert "2026-07-27" in text
    assert [c[1] for c in ak.calls] == ["2026-07-29", "2026-07-28", "2026-07-27"]


def test_get_margin_trading_all_fail_lists_attempted_range():
    _seed_calendar(
        [
            "2026-07-22",
            "2026-07-23",
            "2026-07-24",
            "2026-07-27",
            "2026-07-28",
            "2026-07-29",
        ]
    )
    ak = _SeqAk({})  # every day raises
    text = _FixtureProvider(ak).get_margin_trading("600519", curr_date="2026-07-29")
    assert "【数据获取失败】" in text
    assert "融资融券数据获取失败" in text
    assert "已尝试 2026-07-29 至 2026-07-23 共 5 个交易日，均无数据" in text


def test_get_zt_pool_rolls_back_and_labels_actual_date(monkeypatch):
    _seed_calendar(
        [
            "2026-07-22",
            "2026-07-23",
            "2026-07-24",
            "2026-07-27",
            "2026-07-28",
            "2026-07-29",
        ]
    )
    monkeypatch.setattr(tc, "now_cn", lambda: datetime(2026, 7, 29, 10, 0, 0))
    pool = pd.DataFrame({"代码": ["000001", "600000"], "连板数": [1, 2]})
    ak = _SeqAk(
        {
            "2026-07-29": pd.DataFrame(),  # empty -> DateDataUnavailable
            "2026-07-28": pool,
        }
    )
    # empty DataFrame is truthy-check empty -> raise
    with patch.object(tc, "now_cn", return_value=datetime(2026, 7, 29, 12, 0, tzinfo=tc.CN_TZ)):
        text = _FixtureProvider(ak).get_zt_pool("2026-07-29")
    assert "【数据日期】2026-07-28" in text
    assert "请求 2026-07-29" in text
    assert "涨停家数：2" in text


def test_get_lhb_detail_rolls_back_until_market_table_available():
    _seed_calendar(
        [
            "2026-07-22",
            "2026-07-23",
            "2026-07-24",
            "2026-07-27",
            "2026-07-28",
            "2026-07-29",
        ]
    )
    # day1 empty market, day2 market present but symbol not listed (stop), 
    # Use day with symbol hit to assert success path with rollback note.
    hit = pd.DataFrame(
        {
            "代码": ["600519"],
            "名称": ["贵州茅台"],
            "买入额": [1.0],
        }
    )
    ak = _SeqAk(
        {
            "2026-07-29": pd.DataFrame(),
            "2026-07-28": hit,
        }
    )
    text = _FixtureProvider(ak).get_lhb_detail("600519", "2026-07-29")
    assert "【数据日期】2026-07-28" in text
    assert "请求 2026-07-29" in text
    assert "龙虎榜明细（2026-07-28）" in text


def test_is_cn_trading_day_hard_fails_without_explicit_fallback():
    tc.clear_cn_trade_date_cache()
    with patch.object(tc, "_fetch_cn_trade_dates_from_akshare", side_effect=RuntimeError("down")), \
         patch.object(tc, "_fetch_cn_trade_dates_from_fuyao", side_effect=RuntimeError("down")):
        with pytest.raises(tc.TradeCalendarUnavailableError):
            tc.is_cn_trading_day("2026-07-26")


def test_is_cn_trading_day_soft_fallback_requires_explicit_true(caplog):
    import logging

    tc.clear_cn_trade_date_cache()
    with patch.object(tc, "_fetch_cn_trade_dates_from_akshare", side_effect=RuntimeError("down")), \
         patch.object(tc, "_fetch_cn_trade_dates_from_fuyao", side_effect=RuntimeError("down")):
        with caplog.at_level(logging.WARNING):
            assert tc.is_cn_trading_day("2026-07-28", allow_weekday_fallback=True) is True  # Tuesday
            assert tc.is_cn_trading_day("2026-07-26", allow_weekday_fallback=True) is False  # Sunday
    assert any("Mon-Fri fallback" in r.message for r in caplog.records)


def test_normalize_rejects_stale_calendar_cache():
    """Cached max trade day 60 days before request must hard-fail, not snap to old day."""
    _seed_calendar(
        [
            "2026-05-20",
            "2026-05-21",
            "2026-05-22",
            "2026-05-25",  # newest ~65 days before 2026-07-29
        ]
    )
    with pytest.raises(tc.TradeCalendarUnavailableError) as ei:
        tc.normalize_to_trading_day("2026-07-29")
    assert "缓存最新交易日" in str(ei.value)
    assert "2026-05-25" in str(ei.value)


def test_normalize_allows_calendar_within_staleness_bound():
    _seed_calendar(
        [
            "2026-07-20",
            "2026-07-21",
            "2026-07-22",  # 7 days before 07-29
        ]
    )
    assert tc.normalize_to_trading_day("2026-07-29") == "2026-07-22"
