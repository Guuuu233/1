"""3c-3: structural guards against undated / analysis-date-blind providers."""

from __future__ import annotations

import inspect
import re
from datetime import timedelta
from typing import Callable
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tradingagents.dataflows.providers.base import BaseMarketDataProvider
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.providers.cn_investoday_provider import CnInvestodayProvider
from tradingagents.dataflows.providers.registry import build_default_registry
from tradingagents.dataflows.trade_calendar import cn_today_str, now_cn


DATE_PARAM_NAMES = {
    "curr_date",
    "date",
    "start_date",
    "end_date",
    "trade_date",
    "begin_date",
    "as_of",
}

# Methods that are intentionally date-blind (raw paging / pure transport).
# Adding a new get_* without a date param requires an explicit whitelist entry.
TIMELESS_GET_METHODS = {
    # Low-level sina live stream page fetch; historical refuse is enforced on
    # get_global_news / router, not on the raw pager.
    "get_sina_global_news",
}


def _has_date_param(fn: Callable) -> bool:
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return False
    return any(p in DATE_PARAM_NAMES or p.endswith("_date") for p in params)


def test_base_realtime_quotes_accepts_curr_date():
    sig = inspect.signature(BaseMarketDataProvider.get_realtime_quotes)
    assert "curr_date" in sig.parameters


def test_all_time_sensitive_get_methods_have_date_param():
    """Every provider get_* that returns time-sensitive data must take a date."""
    registry = build_default_registry()
    violations: list[str] = []
    for name in registry.list_names():
        provider = registry.get(name)
        assert provider is not None
        for attr in dir(provider):
            if not attr.startswith("get_"):
                continue
            if attr in TIMELESS_GET_METHODS:
                continue
            fn = getattr(provider, attr)
            if not callable(fn):
                continue
            # Skip inherited abstract raises that are not overridden? still check.
            if not _has_date_param(fn):
                violations.append(f"{provider.name}.{attr}")
    assert violations == [], (
        "time-sensitive get_* missing date param (add param or whitelist): "
        + ", ".join(sorted(violations))
    )


def test_whitelist_entries_actually_exist_and_lack_date():
    """Whitelist must stay honest: each entry exists on some provider and has no date."""
    registry = build_default_registry()
    found: set[str] = set()
    for name in registry.list_names():
        provider = registry.get(name)
        for attr in TIMELESS_GET_METHODS:
            if hasattr(provider, attr):
                found.add(attr)
                fn = getattr(provider, attr)
                assert not _has_date_param(fn), (
                    f"{attr} has a date param — remove from TIMELESS whitelist"
                )
    missing = TIMELESS_GET_METHODS - found
    assert not missing, f"whitelist methods not found on any provider: {missing}"


# --- missing curr_date must fail (no data) ---------------------------------

def _fail_if_data(text: str) -> None:
    assert isinstance(text, str) and text.strip()
    assert "【数据获取失败】" in text or "缺少" in text
    # Must not look like a successful table dump
    assert "股票简称" not in text
    assert "涨停家数" not in text


def test_missing_date_methods_fail_not_data():
    p = CnAkshareProvider()
    p._ak = MagicMock(side_effect=AssertionError("no network without date"))

    cases = [
        lambda: p.get_lhb_detail("600519", date=None),
        lambda: p.get_zt_pool(date=None),
        lambda: p.get_margin_trading("600519", curr_date=None),
        lambda: p.get_restricted_release("600519", curr_date=None),
        lambda: p.get_earnings_forecast("600519", curr_date=None),
        lambda: p.get_insider_transactions("600519", curr_date=None),
        lambda: p.get_shareholder_count("600519", curr_date=None),
        lambda: p.get_individual_fund_flow("600519", curr_date=None),
        lambda: p.get_balance_sheet("600519", curr_date=None),
    ]
    for call in cases:
        _fail_if_data(call())

    inv = CnInvestodayProvider()
    inv._require_api_key = MagicMock(return_value="k")
    inv._normalize_stock_code = MagicMock(return_value="600519")
    inv._fetch_paged_list = MagicMock(
        side_effect=AssertionError("no network without date")
    )
    for call in (
        lambda: inv.get_balance_sheet("600519", curr_date=None),
        lambda: inv.get_cashflow("600519", curr_date=None),
        lambda: inv.get_income_statement("600519", curr_date=None),
        lambda: inv.get_insider_transactions("600519", curr_date=None),
    ):
        _fail_if_data(call())


# --- two historical days → different date upper bounds --------------------

DATE_RE = re.compile(r"(?<!\d)(20\d{2}[-/]?\d{1,2}[-/]?\d{1,2})(?!\d)")


def _norm_date_token(tok: str) -> str | None:
    s = tok.replace("/", "-")
    if re.fullmatch(r"\d{8}", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    parts = s.split("-")
    if len(parts) != 3:
        return None
    try:
        y, m, d = (int(parts[0]), int(parts[1]), int(parts[2]))
        return f"{y:04d}-{m:02d}-{d:02d}"
    except ValueError:
        return None


def _date_upper_bound(text: str) -> str | None:
    found = []
    for m in DATE_RE.findall(text):
        n = _norm_date_token(m)
        if n:
            found.append(n)
    return max(found) if found else None


def _three_tables() -> dict[str, pd.DataFrame]:
    """Minimal real-shaped sina tables for two-date upper-bound check."""
    # Report periods with effective announce dates via A4:
    # 2025-03-31 → statutory 2025-04-30 (IS/CF YoY-refreshed)
    # 2025-12-31 → max within window 2026-04-25
    # 2026-03-31 → 2026-04-25
    rows_bs = pd.DataFrame(
        {
            "报告日": ["20250331", "20251231", "20260331"],
            "公告日期": ["2025-04-29", "2026-04-25", "2026-04-25"],
            "资产总计": [1.0, 2.0, 3.0],
            "负债合计": [0.4, 0.5, 0.6],
        }
    )
    rows_is = pd.DataFrame(
        {
            "报告日": ["20250331", "20251231", "20260331"],
            "公告日期": ["2026-04-25", "2026-04-17", "2026-04-25"],  # YoY refresh on Q1'25
            "归属于母公司所有者的净利润": [10.0, 20.0, 30.0],
        }
    )
    rows_cf = rows_is.copy()
    return {"资产负债表": rows_bs, "利润表": rows_is, "现金流量表": rows_cf}


class _FinFixtureProvider(CnAkshareProvider):
    def __init__(self, tables: dict[str, pd.DataFrame]):
        self._tables = tables

    def _ak(self):
        tables = self._tables

        class _Ak:
            def stock_financial_report_sina(self, stock, symbol):
                df = tables.get(symbol)
                if df is None:
                    raise ValueError("missing")
                return df.copy()

            def stock_financial_abstract(self, symbol):
                return pd.DataFrame()

            def stock_individual_info_em(self, symbol):
                return pd.DataFrame({"item": ["股票简称"], "value": ["贵州茅台"]})

            def stock_individual_basic_info_xq(self, symbol):
                return pd.DataFrame()

        return _Ak()

    def _fetch_company_info_em_fallback(self, code):
        return pd.DataFrame()


def test_two_historical_dates_yield_different_upper_bounds(monkeypatch):
    """If analysis date is consumed, later analysis day must admit later periods.

    Catches the recurring bug where curr_date is accepted but ignored.
    """
    monkeypatch.setattr(
        "tradingagents.dataflows.providers.cn_akshare_provider.cn_today_str",
        lambda: "2026-07-29",
    )
    provider = _FinFixtureProvider(_three_tables())

    early = "2025-11-15"
    late = "2026-04-30"
    early_text = provider.get_income_statement("600519", curr_date=early)
    late_text = provider.get_income_statement("600519", curr_date=late)

    # Both should succeed with some periods (not hard-fail)
    assert "财务数据截至" in early_text or "报告日" in early_text
    assert "财务数据截至" in late_text or "报告日" in late_text

    # Late analysis must include 2025FY / 2026Q1 periods that early cannot see.
    assert "20251231" not in early_text and "2025-12-31" not in early_text
    assert "20260331" not in early_text
    assert ("20251231" in late_text) or ("20260331" in late_text) or ("2026Q1" in late_text)

    # Report-period tokens present in the body (YYYYMMDD) must differ across analysis days.
    period_re = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
    early_periods = set(period_re.findall(early_text))
    late_periods = set(period_re.findall(late_text))
    assert early_periods, early_text[:300]
    assert late_periods, late_text[:300]
    assert late_periods != early_periods, (
        f"two analysis dates produced identical report periods: {early_periods}"
    )
    assert max(late_periods) > max(early_periods)
    # No report period after the analysis day should appear (period end itself may
    # be after announce; we check effective visibility via membership above).


def test_two_historical_collects_differ_in_date_upper_bound():
    """Live e2e: full collect on two analysis days must not share the same max date.

    Opt-in: set RUN_LIVE_DATA_TESTS=1 (hits akshare / network).
    """
    import os
    if os.getenv("RUN_LIVE_DATA_TESTS") != "1":
        pytest.skip("set RUN_LIVE_DATA_TESTS=1 for live dual-date collect guard")

    from tradingagents.graph.data_collector import _fetch_all

    d1 = "2025-11-15"
    d2 = "2026-04-30"
    today = cn_today_str()
    if today < d2:
        pytest.skip("today before late analysis date")

    pool1 = _fetch_all("600519", d1)
    pool2 = _fetch_all("600519", d2)

    def pool_upper(pool: dict) -> str | None:
        ups = []
        for val in pool.values():
            text = val if isinstance(val, str) else str(val)
            ub = _date_upper_bound(text)
            if ub and ub <= (d2 if pool is pool2 else d1):
                # only count dates not after that pool's analysis day
                analysis = d1 if pool is pool1 else d2
                if ub <= analysis:
                    ups.append(ub)
        return max(ups) if ups else None

    # Stronger: extract max date strictly <= analysis day per pool
    def pool_upper_for(pool: dict, analysis: str) -> str | None:
        ups = []
        for val in pool.values():
            text = val if isinstance(val, str) else str(val)
            for m in DATE_RE.findall(text):
                n = _norm_date_token(m)
                if n and n <= analysis:
                    ups.append(n)
        return max(ups) if ups else None

    u1 = pool_upper_for(pool1, d1)
    u2 = pool_upper_for(pool2, d2)
    assert u1 is not None and u2 is not None
    assert u1 != u2, (
        f"two historical collects share identical date upper bound {u1}; "
        "some path likely ignores analysis date"
    )
    assert u2 > u1
