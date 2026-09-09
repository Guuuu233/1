"""Tests for T+5 shadow backfill idempotency and decoupling of price and entry.

Addresses DAV-782 / DAV-789:
Decoupling price existence from entry existence. Samples with valid t_plus_5_price
must not be degraded to data_missing due to missing or non-numeric entry price.
Repeated backfill on samples with valid price must be strictly idempotent.
"""

from typing import Any, Dict, Optional
import pytest

from tradingagents.agents.utils.shadow_credit import (
    PROTOCOL_VERSION_V2_STRUCTURED,
    T_PLUS_5_STATUS_DATA_MISSING,
    T_PLUS_5_STATUS_DUE_AND_EVALUATED,
    T_PLUS_5_STATUS_PENDING_DUE,
    T_PLUS_5_STATUS_SUSPENSION,
    backfill_tplus5_shadow_for_report,
    backfill_tplus5_shadow_for_reports,
)


def _build_test_report(
    *,
    symbol: str = "600001.SH",
    trade_date: str = "2026-08-03",
    winner: str = "bull",
    entry_price: Optional[float] = 10.0,
    raw_entry_str: Optional[str] = None,
    existing_t5_price: Optional[float] = None,
    existing_t5_status: Optional[str] = None,
    existing_t5_hit: Optional[bool] = None,
    is_suspended: bool = False,
    status: str = "completed",
) -> Dict[str, Any]:
    """Helper to build a completed v2 report with customizable entry and T+5 fields."""
    if raw_entry_str is not None:
        entry_field = raw_entry_str
    elif entry_price is not None:
        entry_field = f"{entry_price:.2f}元"
    else:
        entry_field = None

    direction = "看多" if winner == "bull" else ("看空" if winner == "bear" else "中性")

    rep: Dict[str, Any] = {
        "id": f"rep-{symbol}-{trade_date}",
        "symbol": symbol,
        "trade_date": trade_date,
        "status": status,
        "industry": "白酒",
        "result_data": {
            "symbol": symbol,
            "trade_date": trade_date,
            "industry": "白酒",
            "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
            "market_report": "市场分析正文内容",
            "sentiment_report": "情绪分析正文内容",
            "news_report": "新闻分析正文内容",
            "fundamentals_report": "基本面分析正文内容",
            "smart_money_report": "主力资金分析正文内容",
            "volume_price_report": "量价分析正文内容",
            "macro_report": "宏观分析正文内容",
            "analyst_traces": [{"agent": "bull", "verdict": "看多"}],
            "data_gaps": [],
            "investment_debate_state": {
                "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
                "claims": [
                    {
                        "claim_id": "INV-1",
                        "speaker_key": "Bull",
                        "stance": "bullish",
                        "claim": "主力资金持续净流入",
                        "status": "verified",
                        "is_verified": True,
                    },
                ],
                "claim_evidence_summary": {
                    "INV-1": {"speaker_key": "Bull", "counts": {"verified": 1, "total": 1}, "decision": "adopt"},
                },
                "manager_verdict": {
                    "winner": winner,
                    "direction": direction,
                    "entry": entry_field,
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
                "direction": direction,
                "entry": entry_field,
                "consistency_check_passed": True,
                "failed_checks": [],
            },
            "is_suspended": is_suspended,
            "t_plus_5_price": existing_t5_price,
            "t_plus_5_status": existing_t5_status,
            "t_plus_5_direction_hit": existing_t5_hit,
        },
    }
    if existing_t5_price is not None:
        rep["t_plus_5_price"] = existing_t5_price
    if existing_t5_status is not None:
        rep["t_plus_5_status"] = existing_t5_status
    if existing_t5_hit is not None:
        rep["t_plus_5_direction_hit"] = existing_t5_hit

    return rep


class TestRedTeamScenarios:
    """Mandatory Red Team verification scenarios (RT-1 to RT-5)."""

    def test_rt1_valid_price_with_none_entry_preserves_price_no_data_missing(self):
        """RT-1: Sample has valid t_plus_5_price, entry is None -> retains price, not degraded to data_missing."""
        report = _build_test_report(
            symbol="601012.SH",
            trade_date="2026-08-03",
            winner="tie",
            entry_price=None,
            existing_t5_price=15.6,
            existing_t5_status=T_PLUS_5_STATUS_DUE_AND_EVALUATED,
            existing_t5_hit=False,
        )

        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15")

        # Must preserve price and status, NOT degraded to data_missing
        assert res["t_plus_5_price"] == 15.6
        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DUE_AND_EVALUATED
        assert res["_backfill_status"] != "data_missing"
        assert res["_backfill_status"] == "miss"
        assert res["t_plus_5_direction_hit"] is False

    def test_rt1_valid_price_with_non_numeric_entry_preserves_price_no_data_missing(self):
        """RT-1: Sample has valid t_plus_5_price, entry is non-numeric text -> retains price, not degraded."""
        report = _build_test_report(
            symbol="688981.SH",
            trade_date="2026-08-03",
            winner="bear",
            entry_price=None,
            raw_entry_str="逢反抽122.80减仓",
            existing_t5_price=128.7,
            existing_t5_status=T_PLUS_5_STATUS_DUE_AND_EVALUATED,
            existing_t5_hit=True,
        )

        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15")

        assert res["t_plus_5_price"] == 128.7
        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DUE_AND_EVALUATED
        assert res["_backfill_status"] != "data_missing"
        assert res["_backfill_status"] == "hit"
        assert res["t_plus_5_direction_hit"] is True

    def test_rt1_valid_price_with_missing_entry_calculates_hit_when_trade_date_in_prices(self):
        """RT-1: Sample has valid t_plus_5_price, entry is None, but price_series provides trade_date -> computes hit."""
        report = _build_test_report(
            symbol="600001.SH",
            trade_date="2026-08-03",
            winner="bull",
            entry_price=None,
            existing_t5_price=12.0,
        )
        prices = {"2026-08-03": 10.0, "2026-08-10": 12.0}

        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15", price_series=prices)

        assert res["t_plus_5_price"] == 12.0
        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DUE_AND_EVALUATED
        assert res["t_plus_5_direction_hit"] is True
        assert res["_backfill_status"] == "hit"

    def test_rt1c_partial_price_series_without_t5_preserves_existing_price(self):
        """RT-1c: Valid existing price + partial price_series (no T+5 date) -> retains price, not degraded."""
        report = _build_test_report(
            symbol="601012.SH",
            trade_date="2026-08-03",
            winner="tie",
            entry_price=None,
            existing_t5_price=15.6,
            existing_t5_status=T_PLUS_5_STATUS_DUE_AND_EVALUATED,
            existing_t5_hit=False,
        )
        # partial price series containing only T0 date, not T+5 (2026-08-10)
        partial_prices = {"2026-08-03": 15.0}

        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15", price_series=partial_prices)

        assert res["t_plus_5_price"] == 15.6
        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DUE_AND_EVALUATED
        assert res["_backfill_status"] != "data_missing"
        assert res["_backfill_status"] == "miss"
        assert res["t_plus_5_direction_hit"] is False

    def test_rt1d_get_price_fn_none_preserves_existing_price(self):
        """RT-1d: Valid existing price + get_price_fn returning None -> retains price, not degraded."""
        report = _build_test_report(
            symbol="601012.SH",
            trade_date="2026-08-03",
            winner="tie",
            entry_price=None,
            existing_t5_price=15.6,
            existing_t5_status=T_PLUS_5_STATUS_DUE_AND_EVALUATED,
            existing_t5_hit=False,
        )

        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15", get_price_fn=lambda *a: None)

        assert res["t_plus_5_price"] == 15.6
        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DUE_AND_EVALUATED
        assert res["_backfill_status"] != "data_missing"
        assert res["_backfill_status"] == "miss"
        assert res["t_plus_5_direction_hit"] is False

    def test_rt2_price_present_entry_present_behavior_unchanged(self):
        """RT-2: Sample has valid price and valid entry -> normal behavior unchanged."""
        report = _build_test_report(
            symbol="600001.SH",
            trade_date="2026-08-03",
            winner="bull",
            entry_price=10.0,
        )
        prices = {"2026-08-03": 10.0, "2026-08-10": 11.5}

        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15", price_series=prices)

        assert res["t_plus_5_price"] == 11.5
        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DUE_AND_EVALUATED
        assert res["t_plus_5_direction_hit"] is True
        assert res["_backfill_status"] == "hit"

    def test_rt3_no_price_no_entry_remains_data_missing(self):
        """RT-3: Sample has no price and no entry -> genuinely data_missing (unaffected)."""
        report = _build_test_report(
            symbol="600001.SH",
            trade_date="2026-08-03",
            winner="bull",
            entry_price=None,
            existing_t5_price=None,
        )
        # Empty price series so T+5 price is not available
        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15", price_series={})

        assert res["t_plus_5_price"] is None
        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_DATA_MISSING
        assert res["_backfill_status"] == "data_missing"
        assert res["t_plus_5_direction_hit"] is None

    def test_rt4_idempotent_repeated_backfill(self):
        """RT-4: Repeated backfill on sample with valid price leaves price and status unchanged."""
        report = _build_test_report(
            symbol="600001.SH",
            trade_date="2026-08-03",
            winner="bull",
            entry_price=None,
            existing_t5_price=15.0,
            existing_t5_status=T_PLUS_5_STATUS_DUE_AND_EVALUATED,
            existing_t5_hit=True,
        )

        res1 = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15")
        res2 = backfill_tplus5_shadow_for_report(res1, as_of="2026-08-15")
        res3 = backfill_tplus5_shadow_for_report(res2, as_of="2026-08-15")

        assert res1["t_plus_5_price"] == 15.0
        assert res2["t_plus_5_price"] == 15.0
        assert res3["t_plus_5_price"] == 15.0

        assert res1["t_plus_5_status"] == T_PLUS_5_STATUS_DUE_AND_EVALUATED
        assert res2["t_plus_5_status"] == T_PLUS_5_STATUS_DUE_AND_EVALUATED
        assert res3["t_plus_5_status"] == T_PLUS_5_STATUS_DUE_AND_EVALUATED

        assert res1["t_plus_5_direction_hit"] == res2["t_plus_5_direction_hit"] == res3["t_plus_5_direction_hit"] == True
        assert res1["_backfill_status"] == res2["_backfill_status"] == res3["_backfill_status"] == "hit"

    def test_rt5_suspended_sample_determination_unaffected(self):
        """RT-5: Suspended sample (DAV-779 logic) remains suspension and is not affected."""
        report = _build_test_report(
            symbol="600001.SH",
            trade_date="2026-08-03",
            winner="bull",
            entry_price=10.0,
            is_suspended=True,
        )

        res = backfill_tplus5_shadow_for_report(report, as_of="2026-08-15")

        assert res["t_plus_5_status"] == T_PLUS_5_STATUS_SUSPENSION
        assert res["t_plus_5_price"] is None
        assert res["_backfill_status"] == "suspension"
        assert res["t_plus_5_direction_hit"] is None


class TestBatchBackfillDecoupling:
    """Test batch backfill accounting with decoupled price/entry."""

    def test_batch_backfill_preserves_rt1_samples(self):
        """Batch backfill does not count RT-1 samples as data_missing."""
        reports = [
            # RT-1: Has price, missing entry
            _build_test_report(
                symbol="600001.SH",
                trade_date="2026-08-03",
                winner="tie",
                entry_price=None,
                existing_t5_price=15.6,
                existing_t5_status=T_PLUS_5_STATUS_DUE_AND_EVALUATED,
                existing_t5_hit=True,
            ),
            # RT-2: Normal sample
            _build_test_report(
                symbol="600002.SH",
                trade_date="2026-08-03",
                winner="bull",
                entry_price=10.0,
            ),
            # RT-3: True data_missing
            _build_test_report(
                symbol="600003.SH",
                trade_date="2026-08-03",
                winner="bull",
                entry_price=None,
                existing_t5_price=None,
            ),
        ]

        prices_map = {
            "600002.SH": {"2026-08-03": 10.0, "2026-08-10": 11.0},
        }

        updated, stats = backfill_tplus5_shadow_for_reports(
            reports,
            as_of="2026-08-15",
            price_series_map=prices_map,
        )

        # 600001.SH must be evaluated, NOT data_missing
        assert updated[0]["t_plus_5_price"] == 15.6
        assert updated[0]["t_plus_5_status"] == T_PLUS_5_STATUS_DUE_AND_EVALUATED
        assert updated[0]["t_plus_5_direction_hit"] is True

        # Summary stats
        assert stats["total_scanned"] == 3
        assert stats["qualifying_v2_count"] == 3
        assert stats["evaluated_count"] == 2  # 600001 and 600002
        assert stats["data_missing_count"] == 1  # only 600003
        assert stats["hit_count"] == 2  # both hit
        assert stats["completeness_rate"] == round(2 / 3, 4)
