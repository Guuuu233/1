"""Comprehensive tests for multi-horizon return settlement pipeline (V-01-2 / DAV-830).

Covers all 8 red-team scenarios specified in DAV-830:
RT-1: T+1 停牌 / 一字涨停无法买入 -> unexecutable_entry；不得假装以开盘价成交。
RT-2: 持有期内停牌（T+4 有价、T+5 无价、T+6 有价） -> suspension；不得仅凭 T+5 单点缺失判停牌（须双侧证据）。
RT-3: 目标日恰逢节假日 / 周末 / 缺行 -> 按交易日历滚动到候选日；与旧 iloc[hold_days-1] 的差异负例证明。
RT-4: 价格缺失 / 供应商返回结构异常 -> data_missing / provider_failure；不得当 0 收益。
RT-5: 分红/送转 -> 走 ReturnType=total_return + cash_dividend_total / split_ratio_total 字段；
     不新增状态值；缺分红数据不得静默按 0 计。
RT-6: 旧报告（无 profile / legacy） -> 缺省仍走 DEFAULT_HOLD_DAYS = 5；不得把旧 T+5 结果当成 T+10/T+40。
RT-7: 新 profile（显式 short T+10 / medium T+40） -> 正确解析；未成熟样本标 pending_due，禁止缩短 40 日窗口。
RT-8: 跨档 / 跨配置样本混池 -> 按 profile 隔离，不得混算。
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from api.database import ReportDB
from api.services import backtest_service as bt
from api.services import calibration_service as cal
from tests.test_horizon_return_labels import (
    FIXTURE_EXTENDED_2024,
    FIXTURE_JAN_2024,
    FIXTURE_SPRING_FESTIVAL_2024,
)
from tradingagents.dataflows import return_labels as rl
from tradingagents.dataflows.trade_calendar import trading_days_forward
from tradingagents.graph.horizon_profile import (
    HORIZON_MEDIUM,
    HORIZON_PROFILE_V1,
    HORIZON_SHORT,
)


# ==============================================================================
# RT-1: T+1 停牌 / 一字涨停无法买入 -> unexecutable_entry；不得假装以开盘价成交
# ==============================================================================


class TestRT1UnexecutableEntry:
    """RT-1: Verifies unexecutable entry detection at T+1."""

    def test_t1_suspended_returns_unexecutable_entry_no_fake_fill(self):
        """T+1 suspended (is_suspended=True or volume=0) must yield unexecutable_entry."""
        # T is 2024-02-01, T+1 is 2024-02-02
        bars = {
            "2024-02-01": {"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "volume": 1000},
            "2024-02-02": {
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "volume": 0.0,
                "is_suspended": True,
            },
            "2024-02-23": {"open": 11.0, "high": 11.5, "low": 10.8, "close": 11.2, "volume": 1500},
        }

        res = rl.resolve_horizon_return_label(
            symbol="600519.SH",
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=True,
            bar_data=bars,
            direction="BUY",
        )

        assert res["outcome_status"] == rl.OutcomeStatus.UNEXECUTABLE_ENTRY.value
        assert res["executable_entry_date"] == "2024-02-02"
        assert res["entry_executable_price"] is None, "Must not fabricate execution fill on suspended day"
        assert res["entry_price"] is None
        assert res["exit_price"] is None
        assert res["return_pct"] is None
        assert res["evaluation_eligible"] is False
        assert res["is_direction_hit"] is None

    def test_t1_limit_up_locked_on_buy_returns_unexecutable_entry_no_fake_fill(self):
        """T+1 one-word limit-up (open == high == low == limit_up) cannot be bought."""
        bars = {
            "2024-02-01": {"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.0, "volume": 1000},
            "2024-02-02": {
                "open": 11.0,
                "high": 11.0,
                "low": 11.0,
                "close": 11.0,
                "volume": 500,
                "limit_up": 11.0,
            },
            "2024-02-23": {"open": 12.0, "high": 12.5, "low": 11.8, "close": 12.0, "volume": 1500},
        }

        res = rl.resolve_horizon_return_label(
            symbol="600519.SH",
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=True,
            bar_data=bars,
            direction="BUY",
        )

        assert res["outcome_status"] == rl.OutcomeStatus.UNEXECUTABLE_ENTRY.value
        assert res["entry_executable_price"] is None, "Must NOT pretend to fill at Open on limit-up locked day"
        assert res["return_pct"] is None
        assert res["evaluation_eligible"] is False

    def test_t1_normal_tradable_bar_fills_and_evaluates(self):
        """When T+1 is tradable, fills at executable price and proceeds to evaluation."""
        bars = {
            "2024-02-01": {"open": 10.0, "high": 10.5, "low": 9.8, "close": 10.0, "volume": 1000},
            "2024-02-02": {"open": 10.2, "high": 10.6, "low": 10.1, "close": 10.4, "volume": 1200},
            "2024-02-23": {"open": 11.0, "high": 11.5, "low": 10.8, "close": 11.22, "volume": 1500},
        }

        res = rl.resolve_horizon_return_label(
            symbol="600519.SH",
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=True,
            bar_data=bars,
            direction="BUY",
        )

        assert res["outcome_status"] == rl.OutcomeStatus.EVALUATED_OK.value
        assert res["entry_executable_price"] == 10.2
        assert res["exit_price"] == 11.22
        # (11.22 - 10.2) / 10.2 * 100 = 10.0%
        assert res["return_pct"] == 10.0
        assert res["evaluation_eligible"] is True
        assert res["is_direction_hit"] is True


# ==============================================================================
# RT-2: 持有期内停牌（T+4 有价、T+5 无价、T+6 有价） -> suspension；须双侧证据
# ==============================================================================


class TestRT2SuspensionBilateralEvidence:
    """RT-2: Verifies bilateral evidence requirement for suspension vs data_missing."""

    def test_suspension_with_bilateral_evidence_returns_suspension(self):
        """When exit window is missing prices but subsequent trading day resumes, assert suspension."""
        # Signal: 2024-02-01. Target T+10: 2024-02-23. Rolls: 2024-02-26, 2024-02-27.
        # Subsequent day T+13: 2024-02-28 exists with valid price (bilateral evidence!)
        bars = {
            "2024-02-01": {"close": 10.0, "volume": 1000},
            "2024-02-02": {"open": 10.0, "close": 10.0, "volume": 1000},
            # 2024-02-23 (T+10) missing
            # 2024-02-26 (T+11) missing
            # 2024-02-27 (T+12) missing
            # Subsequent trading day T+13 (2024-02-28) exists with positive price:
            "2024-02-28": {"open": 10.5, "close": 10.5, "volume": 2000},
        }

        res = rl.resolve_horizon_return_label(
            symbol="600519.SH",
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-28",
            as_of_market_closed=True,
            bar_data=bars,
        )

        assert res["outcome_status"] == rl.OutcomeStatus.SUSPENSION.value
        assert res["actual_exit_date"] is None
        assert res["return_pct"] is None
        assert res["evaluation_eligible"] is False

    def test_unilateral_missing_without_subsequent_evidence_returns_data_missing(self):
        """When prices end at T+4 with no subsequent evidence (pipeline truncation), assert data_missing."""
        # Only prices up to T+1 exist; vendor data truncated with no subsequent proof
        bars = {
            "2024-02-01": {"close": 10.0, "volume": 1000},
            "2024-02-02": {"open": 10.0, "close": 10.0, "volume": 1000},
            # No prices for target or any subsequent days
        }

        res = rl.resolve_horizon_return_label(
            symbol="600519.SH",
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-28",
            as_of_market_closed=True,
            bar_data=bars,
        )

        # Strictly forbidden to assert suspension without bilateral evidence!
        assert res["outcome_status"] == rl.OutcomeStatus.DATA_MISSING.value
        assert res["return_pct"] is None
        assert res["evaluation_eligible"] is False


# ==============================================================================
# RT-3: 目标日恰逢节假日 / 周末 / 缺行 -> 按交易日历滚动；负例证明
# ==============================================================================


class TestRT3TradingCalendarRollVsNaiveIlocDrift:
    """RT-3: Verifies calendar rolling and provides negative proof of old iloc drift."""

    def test_spring_festival_calendar_skips_holiday_counterexample(self):
        """Prove exact trading calendar skips 2024 Spring Festival vs naive weekday counterexample."""
        window = rl.resolve_horizon_calendar_window(
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=True,
        )

        # 1. Trading calendar T+10 lands exactly on 2024-02-23 (Fri)
        assert window.target_calendar_date == "2024-02-23"

        # 2. Counterexample: naive weekday addition cur.weekday() < 5 lands on 2024-02-15 (closed holiday)
        assert window.target_calendar_date != "2024-02-15"

    def test_target_date_missing_rolls_to_candidate_date(self):
        """When target date is suspended, rolls to first available candidate date."""
        bars = {
            "2024-02-01": {"close": 10.0, "volume": 1000},
            "2024-02-02": {"open": 10.0, "close": 10.0, "volume": 1000},
            # 2024-02-23 (T+10) suspended:
            "2024-02-23": {"open": 0.0, "close": 0.0, "volume": 0.0, "is_suspended": True},
            # 2024-02-26 (T+11, roll 1) tradable:
            "2024-02-26": {"open": 11.5, "close": 11.5, "volume": 1500},
        }

        res = rl.resolve_horizon_return_label(
            symbol="600519.SH",
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-26",
            as_of_market_closed=True,
            bar_data=bars,
        )

        assert res["outcome_status"] == rl.OutcomeStatus.EVALUATED_OK.value
        assert res["target_calendar_date"] == "2024-02-23"
        assert res["actual_exit_date"] == "2024-02-26"
        assert res["roll_days_used"] == 1
        assert res["exit_price"] == 11.5
        assert res["return_pct"] == 15.0

    def test_negative_proof_old_iloc_slicing_drifts_on_missing_rows(self):
        """Negative proof: old iloc[hold_days - 1] produces silent date drift when rows are missing."""
        # Suppose a vendor CSV has a missing row (e.g. 2024-01-04 is missing due to a pause)
        csv_with_gap = """date,close
2024-01-02,102.0
2024-01-03,103.0
2024-01-05,105.0
2024-01-08,108.0
2024-01-09,109.0
2024-01-10,110.0
"""
        # Canonical calendar: Jan 2, Jan 3, Jan 4, Jan 5, Jan 8 (Jan 8 is 5th trading day!)
        # Old iloc[5 - 1] = iloc[4] takes row index 4 -> 2024-01-09 (price 109.0) — DRIFT!
        # With calendar-aware _get_price_after:
        with patch("tradingagents.dataflows.interface.route_to_vendor", return_value=csv_with_gap):
            # Calendar-aware with explicit trading days:
            canonical_trading_days = FIXTURE_JAN_2024
            cal_price = bt._get_price_after(
                "600519.SH",
                "2024-01-01",
                5,
                trading_days=canonical_trading_days,
                max_roll_days=0,
            )
            # The 5th trading day from 2024-01-01 in FIXTURE_JAN_2024 is 2024-01-08 (price 108.0)
            assert cal_price == 108.0
            assert cal_price != 109.0, "Calendar resolution must not drift to row index 4 (109.0)"


# ==============================================================================
# RT-4: 价格缺失 / 供应商返回结构异常 -> data_missing / provider_failure
# ==============================================================================


class TestRT4DataMissingAndProviderFailure:
    """RT-4: Verifies fail-closed behavior on missing prices and provider failures (never 0 return)."""

    def test_provider_exception_returns_provider_failure_not_zero_return(self):
        """Provider exception must return provider_failure, strictly never 0.0 return."""
        def failing_fetcher(sym, dt):
            raise ConnectionError("Upstream vendor endpoint 504 Gateway Timeout")

        res = rl.resolve_horizon_return_label(
            symbol="600519.SH",
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=True,
            price_fetcher=failing_fetcher,
        )

        assert res["outcome_status"] == rl.OutcomeStatus.PROVIDER_FAILURE.value
        assert res["return_pct"] is None, "Must NOT treat provider failure as 0% return"
        assert res["return_pct"] != 0.0
        assert res["evaluation_eligible"] is False

    def test_missing_price_data_returns_data_missing_not_zero_return(self):
        """Missing prices without bilateral proof returns data_missing, strictly never 0.0 return."""
        res = rl.resolve_horizon_return_label(
            symbol="600519.SH",
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=True,
            bar_data={},  # completely empty
        )

        assert res["outcome_status"] == rl.OutcomeStatus.DATA_MISSING.value
        assert res["return_pct"] is None, "Must NOT treat missing data as 0% return"
        assert res["return_pct"] != 0.0
        assert res["evaluation_eligible"] is False


# ==============================================================================
# RT-5: 分红/送转 -> ReturnType=total_return + cash_dividend / split_ratio 字段
# ==============================================================================


class TestRT5TotalReturnDividendAndSplit:
    """RT-5: Verifies total_return calculation with dividend/split fields and fail-closed missing handling."""

    def test_total_return_with_cash_dividend_and_stock_split(self):
        """Total return calculation includes cash dividends and stock splits."""
        bars = {
            "2024-02-01": {"close": 10.0, "volume": 1000},
            "2024-02-02": {"open": 10.0, "close": 10.0, "volume": 1000},
            "2024-02-23": {"open": 11.0, "close": 11.0, "volume": 1500},
        }
        # Cash dividend 0.50 RMB/share on 2024-02-19
        dividends = {"2024-02-19": 0.50}
        # Stock split 10送10 (ratio 2.0) on 2024-02-20
        splits = {"2024-02-20": 2.0}

        res = rl.resolve_horizon_return_label(
            symbol="600519.SH",
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=True,
            bar_data=bars,
            return_type=rl.ReturnType.TOTAL_RETURN.value,
            dividend_data=dividends,
            split_data=splits,
        )

        assert res["outcome_status"] == rl.OutcomeStatus.EVALUATED_OK.value
        assert res["cash_dividend_total"] == 0.50
        assert res["split_ratio_total"] == 2.0
        # effective_exit = 11.0 * 2.0 + 0.50 = 22.50
        # total_return = (22.50 - 10.0) / 10.0 * 100 = 125.0%
        assert res["return_pct"] == 125.0
        assert res["evaluation_eligible"] is True

        # Invariant: OutcomeStatus remains exact 7 values; NO CASH_DIVIDEND or SPLIT enum added
        assert len(rl.OutcomeStatus) == 7
        assert "cash_dividend" not in [item.value for item in rl.OutcomeStatus]
        assert "split" not in [item.value for item in rl.OutcomeStatus]

    def test_total_return_missing_dividend_data_fails_closed(self):
        """When total_return is requested but dividend_data is None, fail-closed (never silently 0)."""
        bars = {
            "2024-02-01": {"close": 10.0, "volume": 1000},
            "2024-02-02": {"open": 10.0, "close": 10.0, "volume": 1000},
            "2024-02-23": {"open": 11.0, "close": 11.0, "volume": 1500},
        }

        res = rl.resolve_horizon_return_label(
            symbol="600519.SH",
            signal_date="2024-02-01",
            horizon="short",
            trading_days=FIXTURE_SPRING_FESTIVAL_2024,
            as_of="2024-02-23",
            as_of_market_closed=True,
            bar_data=bars,
            return_type=rl.ReturnType.TOTAL_RETURN.value,
            dividend_data=None,  # Missing dividend data
        )

        assert res["outcome_status"] == rl.OutcomeStatus.DATA_MISSING.value
        assert res["return_pct"] is None, "Missing dividend data must NOT silently compute as 0 dividend"
        assert res["evaluation_eligible"] is False


# ==============================================================================
# RT-6: 旧报告（无 profile / legacy） -> 缺省仍走 DEFAULT_HOLD_DAYS = 5
# ==============================================================================


class TestRT6LegacyReportsPreserveDefaultT5:
    """RT-6: Verifies legacy calls preserve DEFAULT_HOLD_DAYS = 5 and T+5 semantics."""

    def test_backtest_submit_defaults_to_t5_when_horizon_omitted(self):
        """backtest_service.submit defaults to hold_days=5 when horizon is omitted."""
        job_id = bt.submit(
            user_id="u1",
            symbol="600519.SH",
            start_date="2024-01-02",
            end_date="2024-01-10",
            selected_analysts=["market"],
        )
        job = bt.get_job(job_id, "u1")
        assert job is not None
        assert job["hold_days"] == 5
        assert job["horizon"] is None
        assert bt.DEFAULT_BACKTEST_HOLD_DAYS == 5

    def test_calibration_service_defaults_to_t5_when_horizon_omitted(self):
        """calibration_service preserves DEFAULT_HOLD_DAYS = 5."""
        assert cal.DEFAULT_HOLD_DAYS == 5


# ==============================================================================
# RT-7: 新 profile（显式 short T+10 / medium T+40） -> pending_due 不缩短
# ==============================================================================


class TestRT7NewProfileShortT10MediumT40PendingDue:
    """RT-7: Verifies canonical short T+10 and medium T+40 offsets and no window shortening."""

    def test_short_and_medium_profile_offsets(self):
        assert rl.PRIMARY_EVAL_OFFSET_SHORT == 10
        assert rl.PRIMARY_EVAL_OFFSET_MEDIUM == 40
        assert HORIZON_PROFILE_V1[HORIZON_SHORT]["primary_eval_offset"] == 10
        assert HORIZON_PROFILE_V1[HORIZON_MEDIUM]["primary_eval_offset"] == 40

    def test_medium_t40_unmatured_sample_returns_pending_due_without_truncation(self):
        """Unmatured medium T+40 sample at day 20 returns pending_due; strictly no window shortening."""
        # 20 trading days have elapsed from 2024-01-02 to 2024-01-30 (index 19)
        # Needs 40 trading days (2024-03-05)
        res = rl.resolve_horizon_return_label(
            symbol="600519.SH",
            signal_date="2024-01-02",
            horizon="medium",
            trading_days=FIXTURE_EXTENDED_2024,
            as_of="2024-01-30",  # Only 20 days elapsed
            as_of_market_closed=True,
        )

        assert res["outcome_status"] == rl.OutcomeStatus.PENDING_DUE.value
        assert res["evaluation_eligible"] is False
        assert res["return_pct"] is None
        assert res["target_calendar_date"] == FIXTURE_EXTENDED_2024[40]


# ==============================================================================
# RT-8: 跨档 / 跨配置样本混池 -> 按 profile 隔离，不得混算
# ==============================================================================


class TestRT8CrossProfilePoolIsolation:
    """RT-8: Verifies strict pool isolation across horizons and profiles."""

    def test_calibration_admissibility_isolates_short_and_medium_pools(self):
        """_is_admissible_calibration_report strictly isolates short vs medium vs legacy."""
        rep_short = ReportDB(
            id="rep-short-1",
            status="completed",
            symbol="600519.SH",
            trade_date="2024-01-02",
            probability=0.7,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={
                "horizon": "short",
                "profile_id": "horizon_profile_v1",
                "mode": "single_horizon",
                "horizons_explicit": True,
            },
        )
        rep_medium = ReportDB(
            id="rep-med-1",
            status="completed",
            symbol="600519.SH",
            trade_date="2024-01-02",
            probability=0.7,
            analysis_status="VALID",
            trade_action="BUY",
            result_data={
                "horizon": "medium",
                "profile_id": "horizon_profile_v1",
                "mode": "single_horizon",
                "horizons_explicit": True,
            },
        )

        # 1. Querying short accepts short, strictly rejects medium
        admissible_short_for_short, _, _, _ = cal._is_admissible_calibration_report(
            rep_short, target_horizon="short"
        )
        assert admissible_short_for_short is True

        admissible_med_for_short, _, _, _ = cal._is_admissible_calibration_report(
            rep_medium, target_horizon="short"
        )
        assert admissible_med_for_short is False, "Short evaluation must NOT admit medium samples"

        # 2. Querying medium accepts medium, strictly rejects short
        admissible_med_for_med, _, _, _ = cal._is_admissible_calibration_report(
            rep_medium, target_horizon="medium"
        )
        assert admissible_med_for_med is True

        admissible_short_for_med, _, _, _ = cal._is_admissible_calibration_report(
            rep_short, target_horizon="medium"
        )
        assert admissible_short_for_med is False, "Medium evaluation must NOT admit short samples"

        # 3. Default legacy query (target_horizon is None) rejects explicit medium and explicit short
        admissible_med_for_legacy, _, _, _ = cal._is_admissible_calibration_report(
            rep_medium, target_horizon=None
        )
        assert admissible_med_for_legacy is False, "Legacy T+5 must NOT admit explicit medium samples"

        admissible_short_for_legacy, _, _, _ = cal._is_admissible_calibration_report(
            rep_short, target_horizon=None
        )
        assert admissible_short_for_legacy is False, "Legacy T+5 must NOT admit explicit short samples"
