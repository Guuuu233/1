"""A4 effective-announce-date resolution + financial statement cutoff (commit 3a)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tradingagents.dataflows.financial_announce import (
    DROPPED_YOY_PROMPT_NOTE,
    LATE_FILING_GRACE_DAYS,
    PATH_DROPPED_YOY_REFRESH,
    PATH_MAX_WITHIN_WINDOW,
    PATH_STATUTORY_FALLBACK,
    build_effective_announce_map,
    filter_abstract_period_columns,
    filter_financial_df_by_effective_announce,
    financial_cutoff_header,
    resolve_effective_announce_date,
    statutory_disclosure_deadline,
)
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider


# ── statutory deadlines ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "period, expected",
    [
        ("20251231", date(2026, 4, 30)),  # annual → next Apr 30
        ("20250630", date(2025, 8, 31)),  # H1 → Aug 31
        ("20250331", date(2025, 4, 30)),  # Q1 → Apr 30
        ("20250930", date(2025, 10, 31)),  # Q3 → Oct 31
        ("20241231", date(2025, 4, 30)),
    ],
)
def test_statutory_disclosure_deadline(period, expected):
    assert statutory_disclosure_deadline(period) == expected


# ── resolve_effective_announce_date with real 600519-shaped fixtures ─


def test_resolve_20250331_uses_bs_within_statutory_not_refreshed_is():
    """IS/CF 2025-03-31 ann=2026-04-25 is YoY-refresh; BS 2025-04-30 stays."""
    result = resolve_effective_announce_date(
        "20250331",
        ["20250430", "20260425", "20260425"],
    )
    assert result.effective_date == date(2025, 4, 30)
    assert result.path == PATH_MAX_WITHIN_WINDOW
    assert date(2025, 4, 30) in result.kept_announce_dates
    assert date(2026, 4, 25) in result.discarded_announce_dates


def test_resolve_20240930_keeps_original_q3_announce():
    result = resolve_effective_announce_date(
        "20240930",
        ["20241026", "20251030", "20251030"],
    )
    assert result.effective_date == date(2024, 10, 26)
    assert result.path == PATH_MAX_WITHIN_WINDOW


def test_resolve_20251231_max_within_window():
    """FY2025: BS 20260425 and IS/CF 20260417 both ≤ 2026-04-30 → max."""
    result = resolve_effective_announce_date(
        "20251231",
        ["20260425", "20260417", "20260417"],
    )
    assert result.effective_date == date(2026, 4, 25)
    assert result.path == PATH_MAX_WITHIN_WINDOW
    assert result.discarded_announce_dates == ()


def test_resolve_all_refreshed_drops_yoy_constructed():
    """CONSTRUCTED fixture: every announce date is ~1y past statutory (YoY).

    Not real 600519 2024Q1 data (see test_resolve_real_20240331_keeps_bs_ann).
    All values beyond grace window → dropped_yoy_refresh, effective=statutory.
    """
    result = resolve_effective_announce_date(
        "20240331",
        ["20250430", "20250430", "20250430"],  # constructed: all ~1y late
    )
    assert result.effective_date == date(2024, 4, 30)
    assert result.path == PATH_DROPPED_YOY_REFRESH
    assert result.kept_announce_dates == ()
    assert date(2025, 4, 30) in result.discarded_announce_dates
    assert result.grace_days == LATE_FILING_GRACE_DAYS


def test_resolve_real_20240331_keeps_bs_ann_not_statutory_only():
    """REAL 600519-shaped 2024Q1: BS 2024-04-27 ≤ deadline, IS/CF refreshed.

    Must keep 2024-04-27 via max_within_window — if this falls back to
    statutory-only, A4 is silently degrading.
    """
    result = resolve_effective_announce_date(
        "20240331",
        ["20240427", "20250430", "20250430"],  # real 600519 probe: BS/IS/CF
    )
    assert result.effective_date == date(2024, 4, 27)
    assert result.path == PATH_MAX_WITHIN_WINDOW
    assert date(2024, 4, 27) in result.kept_announce_dates
    assert date(2025, 4, 30) in result.discarded_announce_dates
    assert result.effective_date != result.statutory_deadline


def test_resolve_empty_announce_dates_uses_statutory():
    result = resolve_effective_announce_date("20250630", [])
    assert result.effective_date == date(2025, 8, 31)
    assert result.path == PATH_STATUTORY_FALLBACK


def test_resolve_688981_real_late_filing_uses_actual_announce_not_statutory():
    """REAL 688981-shaped late first filing: all three anns 8–15d past deadline.

    Old hard-cut A4 would discard them and leak via statutory_fallback
    (system thinks data public on 10-31 when actual filing was 11-08).
    Windowed A4 must keep the real announce date.
    """
    # 2024Q3 statutory = 2024-10-31; first filing 2024-11-08 (+8d)
    result = resolve_effective_announce_date(
        "20240930",
        ["20241108", "20241108", "20241108"],
    )
    assert result.effective_date == date(2024, 11, 8)
    assert result.path == PATH_MAX_WITHIN_WINDOW
    assert result.effective_date != result.statutory_deadline
    assert date(2024, 11, 8) in result.kept_announce_dates
    assert result.discarded_announce_dates == ()


def test_resolve_688981_mixed_late_and_yoy_keeps_late_drops_yoy():
    """688981 2024Q1 shape: one real late ann (+10d) + YoY refresh (~374d).

    Effective must be the real late date, not statutory and not YoY.
    """
    # statutory 2024-04-30; real late 2024-05-10; yoy 2025-05-09
    result = resolve_effective_announce_date(
        "20240331",
        ["20240510", "20250509", "20250509"],
    )
    assert result.effective_date == date(2024, 5, 10)
    assert result.path == PATH_MAX_WITHIN_WINDOW
    assert date(2024, 5, 10) in result.kept_announce_dates
    assert date(2025, 5, 9) in result.discarded_announce_dates


def test_resolve_600519_q1_2025_still_drops_yoy_is_cf():
    """Regression: 600519 Q1'25 effective remains 2025-04-30 (IS/CF 2026-04-25 dropped)."""
    result = resolve_effective_announce_date(
        "20250331",
        ["20250430", "20260425", "20260425"],
    )
    assert result.effective_date == date(2025, 4, 30)
    assert result.path == PATH_MAX_WITHIN_WINDOW
    assert date(2026, 4, 25) in result.discarded_announce_dates


# ── table filter ────────────────────────────────────────────────────


def _three_tables() -> dict[str, pd.DataFrame]:
    """Minimal 600519-shaped rows covering the user-verified periods."""
    periods = [
        # report, BS ann, IS ann, CF ann
        ("20250331", "20250430", "20260425", "20260425"),
        ("20241231", "20260417", "20260417", "20260417"),  # note: may be refreshed FY
        ("20240930", "20241026", "20251030", "20251030"),
        ("20240630", "20240809", "20250813", "20250813"),
        ("20251231", "20260425", "20260417", "20260417"),
        ("20260331", "20260425", "20260425", "20260425"),
    ]
    bs = pd.DataFrame(
        {
            "报告日": [p[0] for p in periods],
            "公告日期": [p[1] for p in periods],
            "总资产": list(range(100, 100 + len(periods))),
        }
    )
    inc = pd.DataFrame(
        {
            "报告日": [p[0] for p in periods],
            "公告日期": [p[2] for p in periods],
            "归属于母公司所有者的净利润": list(range(200, 200 + len(periods))),
        }
    )
    cf = pd.DataFrame(
        {
            "报告日": [p[0] for p in periods],
            "公告日期": [p[3] for p in periods],
            "经营活动产生的现金流量净额": list(range(300, 300 + len(periods))),
        }
    )
    return {"资产负债表": bs, "利润表": inc, "现金流量表": cf}


def test_filter_income_before_and_after_q1_2025_announce():
    tables = _three_tables()
    eff = build_effective_announce_map(tables)
    # Before 2025-04-30, 2025Q1 must be excluded even though raw IS ann is 2026.
    filtered, latest = filter_financial_df_by_effective_announce(
        tables["利润表"], eff, "2025-04-20"
    )
    periods = set(filtered["报告日"].astype(str))
    assert "20250331" not in periods
    assert "20240930" in periods

    # On 2025-04-30, 2025Q1 becomes available via BS/statutory path.
    filtered2, latest2 = filter_financial_df_by_effective_announce(
        tables["利润表"], eff, "2025-04-30"
    )
    periods2 = set(filtered2["报告日"].astype(str))
    assert "20250331" in periods2
    assert latest2 is not None
    assert latest2.report_period == "20250331"
    assert latest2.effective_date == date(2025, 4, 30)


def test_filter_does_not_leak_unannounced_fy_via_report_period():
    """Report period 20260331 with effective 2026-04-25 must not leak on 2026-04-20."""
    tables = _three_tables()
    eff = build_effective_announce_map(tables)
    filtered, _ = filter_financial_df_by_effective_announce(
        tables["资产负债表"], eff, "2026-04-20"
    )
    periods = set(filtered["报告日"].astype(str))
    assert "20260331" not in periods
    assert "20251231" not in periods  # effective 2026-04-25


def test_financial_cutoff_header_mentions_period_and_effective_date():
    tables = _three_tables()
    eff = build_effective_announce_map(tables)
    _, latest = filter_financial_df_by_effective_announce(
        tables["资产负债表"], eff, "2026-07-28"
    )
    header = financial_cutoff_header(latest, "2026-07-28")
    assert "财务数据截至" in header
    assert "2026Q1" in header or "20260331" in header
    assert "2026-04-25" in header
    # latest period for this fixture is not dropped_yoy → no caveat
    assert DROPPED_YOY_PROMPT_NOTE not in header


def test_financial_cutoff_header_appends_yoy_disclaimer():
    """dropped_yoy_refresh latest period must surface the statutory-floor caveat."""
    result = resolve_effective_announce_date(
        "20241231",
        ["20260417", "20260417", "20260417"],  # all ~1y late vs 2025-04-30
    )
    assert result.path == PATH_DROPPED_YOY_REFRESH
    header = financial_cutoff_header(result, "2025-06-01")
    assert "财务数据截至" in header
    assert DROPPED_YOY_PROMPT_NOTE in header
    assert "首次公告日" in header
    assert "法定披露截止日" in header


def test_financial_cutoff_header_yoy_disclaimer_forced_flag():
    """Even if latest path is max_within_window, explicit flag still appends note."""
    result = resolve_effective_announce_date(
        "20250331",
        ["20250430", "20260425", "20260425"],
    )
    assert result.path == PATH_MAX_WITHIN_WINDOW
    plain = financial_cutoff_header(result, "2025-05-01")
    forced = financial_cutoff_header(result, "2025-05-01", yoy_disclaimer=True)
    assert DROPPED_YOY_PROMPT_NOTE not in plain
    assert DROPPED_YOY_PROMPT_NOTE in forced


def test_abstract_columns_filtered_by_effective_map():
    tables = _three_tables()
    eff = build_effective_announce_map(tables)
    abstract = pd.DataFrame(
        {
            "选项": ["按报告期"],
            "指标": ["净利润"],
            "20260331": [1.0],
            "20251231": [2.0],
            "20250930": [3.0],
            "20250331": [4.0],
        }
    )
    filtered, latest = filter_abstract_period_columns(abstract, eff, "2025-05-01")
    assert "20250331" in filtered.columns
    assert "20260331" not in filtered.columns
    assert "20251231" not in filtered.columns
    assert latest is not None
    assert latest.report_period == "20250331"


# ── provider-level behavior ─────────────────────────────────────────


class _FinFixtureProvider(CnAkshareProvider):
    def __init__(self, tables: dict[str, pd.DataFrame], abstract: pd.DataFrame | None = None):
        self._tables = tables
        self._abstract = abstract

    def _ak(self):
        tables = self._tables
        abstract = self._abstract

        class _Ak:
            def stock_financial_report_sina(self, stock, symbol):
                df = tables.get(symbol)
                if df is None:
                    raise ValueError("missing")
                return df.copy()

            def stock_financial_abstract(self, symbol):
                if abstract is None:
                    raise ValueError("no abstract")
                return abstract.copy()

            def stock_individual_info_em(self, symbol):
                return pd.DataFrame({"item": ["股票简称"], "value": ["贵州茅台"]})

            def stock_individual_basic_info_xq(self, symbol):
                return pd.DataFrame()

            def stock_financial_abstract_new_ths(self, symbol, indicator):
                # No announce date — used only to prove historical refuse.
                return pd.DataFrame({"report_date": ["2025-12-31"], "净利润": [1]})

        return _Ak()

    def _fetch_company_info_em_fallback(self, code):
        return pd.DataFrame()


def test_provider_income_truncation_today_vs_90d_ago(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.dataflows.providers.cn_akshare_provider.cn_today_str",
        lambda: "2026-07-28",
    )
    tables = _three_tables()
    provider = _FinFixtureProvider(tables)

    today_text = provider.get_income_statement("600519", curr_date="2026-07-28")
    assert "财务数据截至" in today_text
    assert "20260331" in today_text or "2026Q1" in today_text

    hist_text = provider.get_income_statement("600519", curr_date="2026-04-20")
    assert "20260331" not in hist_text
    assert "20251231" not in hist_text
    # 2025Q1 should still be present via effective 2025-04-30
    assert "20250331" in hist_text


def test_provider_missing_curr_date_refuses_financials(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.dataflows.providers.cn_akshare_provider.cn_today_str",
        lambda: "2026-07-28",
    )
    provider = _FinFixtureProvider(_three_tables())
    text = provider.get_balance_sheet("600519", curr_date=None)
    assert "缺少 curr_date" in text
    assert "不可用" in text


def test_provider_historical_refuses_ths_fallback(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.dataflows.providers.cn_akshare_provider.cn_today_str",
        lambda: "2026-07-28",
    )

    class _EmptySina(_FinFixtureProvider):
        def _ak(self):
            class _Ak:
                def stock_financial_report_sina(self, stock, symbol):
                    raise RuntimeError("sina down")

                def stock_financial_abstract_new_ths(self, symbol, indicator):
                    return pd.DataFrame({"report_date": ["20251231"], "净利润": [9]})

            return _Ak()

    provider = _EmptySina({})
    text = provider.get_cashflow("600519", curr_date="2026-01-15")
    assert "历史日期" in text or "无公告日" in text
    assert "不可用" in text
    assert "20251231" not in text or "备用" not in text


def test_shareholder_count_requires_curr_date():
    class _P(CnAkshareProvider):
        def _ak(self):
            raise AssertionError("must not call vendor without curr_date")

    text = _P().get_shareholder_count("600519", curr_date=None)
    assert "数据获取失败" in text or "缺少 curr_date" in text
    assert "前视" in text or "不可用" in text or "未排查" in text or "获取失败" in text


def test_fund_flow_requires_curr_date_and_oor_message():
    class _P(CnAkshareProvider):
        def _ak(self):
            class _Ak:
                def stock_individual_fund_flow(self, stock, market):
                    return pd.DataFrame(
                        {
                            "日期": ["2026-07-20", "2026-07-21", "2026-07-22"],
                            "主力净流入-净额": [1, 2, 3],
                        }
                    )

            return _Ak()

    missing = _P().get_individual_fund_flow("600519", curr_date=None)
    assert "缺少 curr_date" in missing

    # Analysis date before available range → explicit OOR, not empty table.
    oor = _P().get_individual_fund_flow("600519", curr_date="2025-01-02")
    assert "120" in oor
    assert "超出" in oor
    assert "不可用" in oor

    ok = _P().get_individual_fund_flow("600519", curr_date="2026-07-22")
    assert "2026-07-22" in ok
    assert "截至于 2026-07-22" in ok
