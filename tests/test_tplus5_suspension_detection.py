"""Unit and regression tests for Track DAV-779: T+5 Suspension Detection.

Distinguishes between objective lack of price (e.g. stock suspended on T+5)
and market data pipeline failure / vendor truncation (data_missing).

Test requirements (DAV-779 contract & Codex/David bilateral evidence review):
1. True suspension is an internal sequence hole, not an end-of-series truncation.
   Bilateral evidence required: T+5 has no price, but BOTH adjacent trading days
   T+4 (preceding) and T+6 (subsequent) exist with valid positive prices -> suspension.
2. Truncation protection: if vendor sequence ends at T+4 (no T+6), it cannot be
   distinguished from pipeline truncation/lag -> must assert suspension=False and data_missing.
3. Whole sequence missing / unobtainable -> data_missing (included in due denominator).
4. Sequence normal with valid T+5 price -> due_and_evaluated (normal).
5. Unverifiable boundary cases -> data_missing (fail-closed to due denominator).
6. Existing explicit is_suspended=True passthrough remains effective.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tradingagents.agents.utils.shadow_credit import (
    T_PLUS_5_STATUS_DATA_MISSING,
    T_PLUS_5_STATUS_DUE_AND_EVALUATED,
    T_PLUS_5_STATUS_NOT_APPLICABLE,
    T_PLUS_5_STATUS_PENDING_DUE,
    T_PLUS_5_STATUS_SUSPENSION,
    VALID_T_PLUS_5_STATUSES,
    backfill_tplus5_shadow_for_report,
    backfill_tplus5_shadow_for_reports,
    detect_tplus5_suspension,
    fetch_close_prices_safe,
)
from tradingagents.dataflows.trade_calendar import trading_days_forward


def _build_v2_report(
    symbol: str = "600519.SH",
    trade_date: str = "2026-08-03",
    winner: str = "bull",
    entry_price: float = 100.0,
    is_suspended: bool | None = None,
    existing_t5_price: float | None = None,
) -> dict:
    """Helper to build a qualifying completed v2 report fixture."""
    return {
        "id": f"rep-{symbol}-{trade_date}",
        "symbol": symbol,
        "trade_date": trade_date,
        "status": "completed",
        "industry": "白酒",
        "result_data": {
            "symbol": symbol,
            "trade_date": trade_date,
            "industry": "白酒",
            "protocol_version": "v2_structured",
            "market_report": "市场分析",
            "sentiment_report": "情绪分析",
            "news_report": "新闻分析",
            "fundamentals_report": "基本面分析",
            "smart_money_report": "主力资金分析",
            "volume_price_report": "量价分析",
            "macro_report": "宏观分析",
            "analyst_traces": [{"agent": "bull", "verdict": "看多"}],
            "data_gaps": [],
            "investment_debate_state": {
                "protocol_version": "v2_structured",
                "claims": [
                    {
                        "claim_id": "INV-1",
                        "speaker_key": "Bull",
                        "stance": "bullish",
                        "claim": "主力资金净流入",
                        "status": "verified",
                        "is_verified": True,
                    }
                ],
                "claim_evidence_summary": {
                    "INV-1": {"speaker_key": "Bull", "counts": {"verified": 1, "total": 1}, "decision": "adopt"},
                },
                "manager_verdict": {
                    "winner": winner,
                    "direction": "看多" if winner == "bull" else ("看空" if winner == "bear" else "中性"),
                    "entry": f"{entry_price:.2f}元",
                    "consistency_check_passed": True,
                    "failed_checks": [],
                },
                "round_messages": [
                    {"speaker_key": "Bull", "stance": "bullish", "model_name": "deepseek-r1"},
                    {"speaker_key": "Bear", "stance": "bearish", "model_name": "qwen-max"},
                    {"speaker_key": "Research Manager", "is_verdict": True, "model_name": "gpt-4o"},
                ],
                "feature_flags": {
                    "v2_debate_enabled": True,
                    "shadow_credit_enabled": True,
                    "credit_weighting_enabled": False,
                },
            },
            "manager_verdict": {
                "winner": winner,
                "direction": "看多" if winner == "bull" else ("看空" if winner == "bear" else "中性"),
                "entry": f"{entry_price:.2f}元",
                "consistency_check_passed": True,
                "failed_checks": [],
            },
            "is_suspended": is_suspended,
            "t_plus_5_price": existing_t5_price,
        },
    }


class TestTPlus5StatusConstants:
    """Test suite ensuring t_plus_5_status domain is centrally managed (AGENTS.md §5)."""

    def test_constants_defined_and_expected_values(self):
        assert T_PLUS_5_STATUS_DUE_AND_EVALUATED == "due_and_evaluated"
        assert T_PLUS_5_STATUS_PENDING_DUE == "pending_due"
        assert T_PLUS_5_STATUS_DATA_MISSING == "data_missing"
        assert T_PLUS_5_STATUS_SUSPENSION == "suspension"
        assert T_PLUS_5_STATUS_NOT_APPLICABLE == "not_applicable"

    def test_valid_statuses_set(self):
        expected = {"due_and_evaluated", "pending_due", "data_missing", "suspension", "not_applicable"}
        assert VALID_T_PLUS_5_STATUSES == expected


class TestDetectTPlus5SuspensionHelper:
    """Direct tests for detect_tplus5_suspension bilateral evidence logic."""

    def test_bilateral_evidence_t4_and_t6_present_t5_missing_returns_true(self):
        """True suspension hole: T+4 and T+6 both present with prices, T+5 missing."""
        # 2026-08-03 (Mon) -> T1: 08-04, T2: 08-05, T3: 08-06, T4: 08-07, T5: 08-10, T6: 08-11
        quotes = {
            "2026-08-03": 10.0,
            "2026-08-04": 10.2,
            "2026-08-05": 10.1,
            "2026-08-06": 10.3,
            "2026-08-07": 10.5,  # T+4 adjacent trading day present
            # 2026-08-10 (T+5) missing
            "2026-08-11": 10.7,  # T+6 subsequent trading day present (bilateral evidence!)
        }
        res = detect_tplus5_suspension(
            symbol="600519.SH",
            trade_date_str="2026-08-03",
            t5_date="2026-08-10",
            quotes_map=quotes,
        )
        assert res is True

    def test_codex_david_reproducer_vendor_truncated_at_t4_returns_false(self):
        """Codex/David reproducer: vendor data truncated at T+4 (no T+6) must return False."""
        T0 = "2026-09-01"
        fwd = trading_days_forward(T0, 5)  # fwd[3]=09-07 (T+4), fwd[4]=09-08 (T+5)
        # 供应商数据刚好截断在 T+4, T+5 及以后没返回:
        qa = {fwd[0]: 10.0, fwd[1]: 10.1, fwd[2]: 10.2, fwd[3]: 10.3}  # 只到 T+4
        # 必须断言 False，防止把管道截断/延迟洗成合法停牌
        assert detect_tplus5_suspension("X", T0, fwd[-1], qa) is False

    def test_t5_present_returns_false(self):
        quotes = {
            "2026-08-03": 10.0,
            "2026-08-07": 10.5,
            "2026-08-10": 11.0,  # T+5 present
            "2026-08-11": 11.2,
        }
        res = detect_tplus5_suspension(
            symbol="600519.SH",
            trade_date_str="2026-08-03",
            t5_date="2026-08-10",
            quotes_map=quotes,
        )
        assert res is False

    def test_empty_quotes_returns_false(self):
        res = detect_tplus5_suspension(
            symbol="600519.SH",
            trade_date_str="2026-08-03",
            t5_date="2026-08-10",
            quotes_map={},
        )
        assert res is False

    def test_only_t0_present_without_bilateral_days_returns_false(self):
        quotes = {"2026-08-03": 10.0}
        res = detect_tplus5_suspension(
            symbol="600519.SH",
            trade_date_str="2026-08-03",
            t5_date="2026-08-10",
            quotes_map=quotes,
        )
        assert res is False

    def test_t4_present_but_t6_price_nonpositive_returns_false(self):
        # T+6 has price 0.0 or negative -> invalid bilateral evidence
        quotes = {
            "2026-08-07": 10.5,
            "2026-08-11": 0.0,
        }
        res = detect_tplus5_suspension(
            symbol="600519.SH",
            trade_date_str="2026-08-03",
            t5_date="2026-08-10",
            quotes_map=quotes,
        )
        assert res is False

    def test_t6_present_but_t4_missing_returns_false(self):
        # T+4 missing -> missing preceding anchor
        quotes = {
            "2026-08-11": 10.7,
        }
        res = detect_tplus5_suspension(
            symbol="600519.SH",
            trade_date_str="2026-08-03",
            t5_date="2026-08-10",
            quotes_map=quotes,
        )
        assert res is False

    def test_custom_calendar_supported(self):
        # Custom trading calendar with 6 days forward
        custom_cal = [
            "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06",
            "2026-08-07", "2026-08-10", "2026-08-11",
        ]
        quotes = {"2026-08-07": 15.0, "2026-08-11": 15.5}  # T+4 and T+6 exist
        res = detect_tplus5_suspension(
            symbol="600519.SH",
            trade_date_str="2026-08-03",
            t5_date="2026-08-10",
            quotes_map=quotes,
            trading_calendar=custom_cal,
        )
        assert res is True


class TestTPlus5SuspensionDetectionIntegration:
    """End-to-end integration tests through backfill_tplus5_shadow_for_report."""

    def test_requirement_1_bilateral_evidence_marked_suspension(self):
        """Case 1: T+5 missing, but T+4 and T+6 BOTH exist -> suspension, excluded from due."""
        report = _build_v2_report(trade_date="2026-08-03", entry_price=10.0)
        # 2026-08-03 -> T+4 is 2026-08-07, T+5 is 2026-08-10, T+6 is 2026-08-11
        prices = {
            "2026-08-03": 10.0,
            "2026-08-04": 10.1,
            "2026-08-05": 10.2,
            "2026-08-06": 10.3,
            "2026-08-07": 10.4,  # T+4 present
            # 2026-08-10 (T+5) missing -> suspended on T+5
            "2026-08-11": 10.6,  # T+6 present (bilateral evidence!)
        }
        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15", price_series=prices)

        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_SUSPENSION
        assert res["is_suspended"] is True
        assert res["is_t_plus_5_due"] is False
        assert res["t_plus_5_evaluated"] is False
        assert res["t_plus_5_direction_hit"] is None
        assert res["t_plus_5_price"] is None
        assert res["_backfill_status"] == "suspension"
        assert res["result_data"]["is_suspended"] is True
        assert res["result_data"]["t_plus_5_status"] == T_PLUS_5_STATUS_SUSPENSION
        assert res["result_data"]["is_t_plus_5_due"] is False

    def test_requirement_2_vendor_truncated_at_t4_marked_data_missing(self):
        """Case 2: Vendor data truncated at T+4 (no T+6) -> data_missing (fail-closed to due)."""
        T0 = "2026-09-01"
        report = _build_v2_report(trade_date=T0, entry_price=10.0)
        fwd = trading_days_forward(T0, 5)
        # Only up to T+4
        qa = {fwd[0]: 10.0, fwd[1]: 10.1, fwd[2]: 10.2, fwd[3]: 10.3}
        res = backfill_tplus5_shadow_for_report(report, as_of="2026-09-15", price_series=qa)

        # Must NOT be marked suspension (no bilateral evidence!)
        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DATA_MISSING
        assert res["is_t_plus_5_due"] is True
        assert res["t_plus_5_evaluated"] is True
        assert res.get("is_suspended") is not True
        assert res["_backfill_status"] == "data_missing"

    def test_requirement_2_vendor_fetch_csv_with_bilateral_evidence_marked_suspension(self):
        """Case 2b: Vendor CSV returns T0..T+4 and T+6 but misses T+5 -> suspension."""
        report = _build_v2_report(trade_date="2026-08-03", entry_price=10.0)
        # Frozen CSV fixture: includes 08-03..08-07 and 08-11, misses 08-10
        vendor_csv = (
            "date,open,high,low,close,volume\n"
            "2026-08-03,10.0,10.2,9.9,10.0,1000\n"
            "2026-08-04,10.0,10.3,10.0,10.1,1100\n"
            "2026-08-05,10.1,10.4,10.1,10.2,1200\n"
            "2026-08-06,10.2,10.5,10.2,10.3,1300\n"
            "2026-08-07,10.3,10.6,10.3,10.4,1400\n"
            "2026-08-11,10.4,10.7,10.4,10.5,1500\n"
        )
        with patch("tradingagents.dataflows.interface.route_to_vendor", return_value=vendor_csv):
            res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15")

        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_SUSPENSION
        assert res["is_suspended"] is True
        assert res["is_t_plus_5_due"] is False
        assert res["t_plus_5_price"] is None
        assert res["_backfill_status"] == "suspension"

    def test_requirement_2_vendor_fetch_csv_truncated_at_t4_marked_data_missing(self):
        """Case 2c: Vendor CSV ends at T+4 (truncated) -> data_missing."""
        report = _build_v2_report(trade_date="2026-08-03", entry_price=10.0)
        # Frozen CSV fixture truncated at 08-07
        vendor_csv = (
            "date,open,high,low,close,volume\n"
            "2026-08-03,10.0,10.2,9.9,10.0,1000\n"
            "2026-08-07,10.3,10.6,10.3,10.4,1400\n"
        )
        with patch("tradingagents.dataflows.interface.route_to_vendor", return_value=vendor_csv):
            res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15")

        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DATA_MISSING
        assert res["is_t_plus_5_due"] is True
        assert res.get("is_suspended") is not True

    def test_requirement_3_entire_sequence_missing_marked_data_missing(self):
        """Case 3: Vendor / series returns nothing -> data_missing, included in due."""
        report = _build_v2_report(trade_date="2026-08-03", entry_price=10.0)
        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15", price_series={})

        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DATA_MISSING
        assert res["is_t_plus_5_due"] is True
        assert res["t_plus_5_evaluated"] is True
        assert res.get("is_suspended") is not True
        assert res["_backfill_status"] == "data_missing"

    def test_requirement_4_sequence_normal_with_t5_price_marked_due_and_evaluated(self):
        """Case 4: Both entry and T+5 price present -> due_and_evaluated."""
        report = _build_v2_report(trade_date="2026-08-03", entry_price=10.0)
        prices = {
            "2026-08-03": 10.0,
            "2026-08-07": 10.4,
            "2026-08-10": 11.5,
            "2026-08-11": 11.7,
        }
        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15", price_series=prices)

        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DUE_AND_EVALUATED
        assert res["is_t_plus_5_due"] is True
        assert res["t_plus_5_evaluated"] is True
        assert res["t_plus_5_price"] == 11.5
        assert res["t_plus_5_direction_hit"] is True
        assert res["_backfill_status"] == "hit"

    def test_requirement_5_unverifiable_boundary_only_t0_present_falls_to_data_missing(self):
        """Case 5a: Only T0 is in prices -> data_missing."""
        report = _build_v2_report(trade_date="2026-08-03", entry_price=10.0)
        prices = {"2026-08-03": 10.0}
        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15", price_series=prices)

        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DATA_MISSING
        assert res["is_t_plus_5_due"] is True
        assert res["_backfill_status"] == "data_missing"

    def test_requirement_5_unverifiable_boundary_dates_outside_window_falls_to_data_missing(self):
        """Case 5b: Only dates from wrong periods present -> data_missing."""
        report = _build_v2_report(trade_date="2026-08-03", entry_price=10.0)
        prices = {"2025-01-02": 10.0, "2025-01-03": 10.1}
        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15", price_series=prices)

        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DATA_MISSING
        assert res["is_t_plus_5_due"] is True
        assert res["_backfill_status"] == "data_missing"

    def test_requirement_5_unverifiable_boundary_get_price_fn_none_without_series(self):
        """Case 5c: get_price_fn returns None without price series -> data_missing."""
        report = _build_v2_report(trade_date="2026-08-03", entry_price=10.0)
        res = backfill_tplus5_shadow_for_report(
            report,
            as_of="2026-08-15",
            get_price_fn=lambda sym, t0, t5: None,
        )

        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DATA_MISSING
        assert res["is_t_plus_5_due"] is True
        assert res["_backfill_status"] == "data_missing"

    def test_requirement_6_explicit_is_suspended_passthrough_still_effective(self):
        """Case 6: Explicit is_suspended=True passes through directly without series."""
        report = _build_v2_report(trade_date="2026-08-03", entry_price=10.0, is_suspended=True)
        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15", price_series={})

        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_SUSPENSION
        assert res["is_suspended"] is True
        assert res["is_t_plus_5_due"] is False
        assert res["_backfill_status"] == "suspension"


class TestBatchBackfillStatsWithSuspension:
    """Batch backfill statistics test verifying suspension count and denominator exclusion."""

    def test_batch_backfill_counts_and_excludes_suspension_from_due(self):
        r_eval = _build_v2_report(symbol="600519.SH", trade_date="2026-08-03", entry_price=10.0)
        r_susp = _build_v2_report(symbol="000001.SZ", trade_date="2026-08-03", entry_price=20.0)
        r_trunc = _build_v2_report(symbol="000002.SZ", trade_date="2026-08-03", entry_price=30.0)
        r_pend = _build_v2_report(symbol="300750.SZ", trade_date="2026-08-20", entry_price=40.0)

        prices_map = {
            "600519.SH": {"2026-08-03": 10.0, "2026-08-10": 11.0},
            # 000001.SZ has bilateral evidence: T+4 (08-07) and T+6 (08-11) present, T+5 (08-10) missing -> suspension
            "000001.SZ": {"2026-08-03": 20.0, "2026-08-07": 20.5, "2026-08-11": 20.8},
            # 000002.SZ truncated at T+4 (08-07), NO T+6 -> data_missing (fail-closed!)
            "000002.SZ": {"2026-08-03": 30.0, "2026-08-07": 30.2},
            # 300750.SZ: 2026-08-20 with as_of 2026-08-15 -> pending_due
            "300750.SZ": {},
        }

        updated, stats = backfill_tplus5_shadow_for_reports(
            [r_eval, r_susp, r_trunc, r_pend],
            as_of="2026-08-15",
            price_series_map=prices_map,
        )

        assert stats["total_scanned"] == 4
        assert stats["qualifying_v2_count"] == 4
        assert stats["suspension_count"] == 1
        assert stats["pending_due_count"] == 1
        assert stats["data_missing_count"] == 1
        assert stats["evaluated_count"] == 1
        # Due denominator must be evaluated (1) + data_missing (1) = 2.
        # Suspension (1) and pending_due (1) must be EXCLUDED from due_count!
        assert stats["due_count"] == 2
        assert stats["completeness_rate"] == 0.5  # 1 / 2 = 0.5
