"""Tests for H1b sample pool analysis_status filtering and excluded_counts (DAV-783 / D-009 §5).

Contract Requirements:
1. Filter strictly based on analysis_status, not winner (prevents winner leakage from ABSTAIN/etc.).
2. Excluded samples must be categorized and counted into `excluded_counts`:
   legacy_null, abstain, invalid_run, data_error, no_trade, wait.
3. Legacy samples with analysis_status IS NULL must be excluded and counted as legacy_null.
4. When qualifying samples drop below 60, dimension_n fails honestly (no relaxing filter).
5. Qualifying VALID samples preserve existing statistical calculation integrity.
6. All tests use in-memory fixtures; no network calls, no production DB access.
"""

from typing import Any, Mapping, Optional
import pytest

from tradingagents.agents.utils.shadow_credit import (
    PROTOCOL_VERSION_V2_STRUCTURED,
    evaluate_h1b_system_gates,
    filter_v2_completed_reports,
    is_qualifying_v2_report,
)


def make_report(
    *,
    report_id: str = "rep-test",
    symbol: str = "600519.SH",
    industry: str = "白酒",
    trade_date: str = "2026-08-01",
    status: str = "completed",
    analysis_status: Optional[str] = "VALID",
    trade_action: Optional[str] = "BUY",
    winner: str = "bull",
    direction: str = "看多",
    protocol_version: str = PROTOCOL_VERSION_V2_STRUCTURED,
    claims: Optional[list] = None,
    shadow_credit_metrics: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Helper to construct a report dict fixture with full v2 debate structure."""
    if claims is None:
        claims = [
            {
                "claim_id": "c1",
                "speaker": "Bull Researcher",
                "stance": "bullish",
                "status": "verified",
                "claim": f"Bull argument for {symbol}",
            }
        ]

    rep = {
        "id": report_id,
        "symbol": symbol,
        "industry": industry,
        "trade_date": trade_date,
        "status": status,
        "protocol_version": protocol_version,
        "manager_verdict": {
            "winner": winner,
            "direction": direction,
            "claim_evidence_summary": {
                "c1": {
                    "speaker": "Bull Researcher",
                    "stance": "bullish",
                    "counts": {"verified": 1, "refuted": 0},
                }
            },
        },
        "claims": claims,
    }
    if analysis_status is not None:
        rep["analysis_status"] = analysis_status
    if trade_action is not None:
        rep["trade_action"] = trade_action

    if shadow_credit_metrics is not None:
        rep["shadow_credit_metrics"] = dict(shadow_credit_metrics)

    return rep


class TestH1bAnalysisStatusFilter:
    """Test suite for analysis_status filtering and excluded_counts contract."""

    def test_valid_buy_qualifies(self):
        """analysis_status=VALID with trade_action=BUY qualifies and enters pool."""
        rep = make_report(analysis_status="VALID", trade_action="BUY", winner="bull")
        assert is_qualifying_v2_report(rep) is True
        filtered = filter_v2_completed_reports([rep])
        assert len(filtered) == 1
        assert filtered[0]["id"] == rep["id"]

    def test_valid_sell_qualifies(self):
        """analysis_status=VALID with trade_action=SELL qualifies and enters pool."""
        rep = make_report(analysis_status="VALID", trade_action="SELL", winner="bear", direction="看空")
        assert is_qualifying_v2_report(rep) is True
        filtered = filter_v2_completed_reports([rep])
        assert len(filtered) == 1

    def test_valid_hold_qualifies(self):
        """analysis_status=VALID with trade_action=HOLD qualifies and enters pool."""
        rep = make_report(analysis_status="VALID", trade_action="HOLD", winner="tie", direction="中性")
        assert is_qualifying_v2_report(rep) is True
        filtered = filter_v2_completed_reports([rep])
        assert len(filtered) == 1

    def test_valid_no_trade_excluded_and_counted(self):
        """analysis_status=VALID with trade_action=NO_TRADE must be excluded and counted as no_trade."""
        rep = make_report(analysis_status="VALID", trade_action="NO_TRADE", winner="bull")
        assert is_qualifying_v2_report(rep) is False
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["no_trade"] == 1

    def test_valid_wait_excluded_and_counted(self):
        """analysis_status=VALID with trade_action=WAIT must be excluded and counted as wait."""
        rep = make_report(analysis_status="VALID", trade_action="WAIT", winner="bear")
        assert is_qualifying_v2_report(rep) is False
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["wait"] == 1

    def test_abstain_excluded_and_counted(self):
        """analysis_status=ABSTAIN must be excluded and counted as abstain."""
        rep = make_report(analysis_status="ABSTAIN", trade_action="WAIT", winner="tie")
        assert is_qualifying_v2_report(rep) is False
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["abstain"] == 1

    def test_abstain_winner_leakage_prevented_like_6decef3d(self):
        """ABSTAIN sample with manager_verdict.winner='bear' (like 6decef3d) must be excluded.

        It must NOT leak into the bear sample count.
        """
        rep = make_report(
            report_id="6decef3d-test",
            analysis_status="ABSTAIN",
            trade_action="NO_TRADE",
            winner="bear",
            direction="看空",
        )
        assert is_qualifying_v2_report(rep) is False
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["abstain"] == 1

        # Even if evaluated directly in gate evaluation, excluded samples must not leak into side count
        gate_res = evaluate_h1b_system_gates(filtered)
        assert gate_res["matrix"]["dimension_side"]["details"]["bear_samples"] == 0

    def test_invalid_run_excluded_and_counted(self):
        """analysis_status=INVALID_RUN must be excluded and counted as invalid_run."""
        rep = make_report(analysis_status="INVALID_RUN", trade_action="NO_TRADE", winner="bull")
        assert is_qualifying_v2_report(rep) is False
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["invalid_run"] == 1

    def test_data_error_excluded_and_counted(self):
        """analysis_status=DATA_ERROR must be excluded and counted as data_error."""
        rep = make_report(analysis_status="DATA_ERROR", trade_action="NO_TRADE", winner="tie")
        assert is_qualifying_v2_report(rep) is False
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["data_error"] == 1

    def test_legacy_null_excluded_and_counted(self):
        """Pre-D-009 legacy reports with analysis_status=None must be excluded and counted as legacy_null."""
        rep = make_report(analysis_status=None, trade_action=None, winner="bull")
        assert is_qualifying_v2_report(rep) is False
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["legacy_null"] == 1

    def test_mixed_pool_excluded_counts_total_integrity(self):
        """Verify excluded_counts total sum + qualifying count equals total input count exactly."""
        mixed_pool = []
        # Qualifying:
        # 4 VALID + BUY
        for i in range(4):
            mixed_pool.append(make_report(report_id=f"buy-{i}", analysis_status="VALID", trade_action="BUY", winner="bull"))
        # 3 VALID + SELL
        for i in range(3):
            mixed_pool.append(make_report(report_id=f"sell-{i}", analysis_status="VALID", trade_action="SELL", winner="bear"))
        # 2 VALID + HOLD
        for i in range(2):
            mixed_pool.append(make_report(report_id=f"hold-{i}", analysis_status="VALID", trade_action="HOLD", winner="tie"))
        # Excluded:
        # 5 VALID + WAIT (wait)
        for i in range(5):
            mixed_pool.append(make_report(report_id=f"wait-{i}", analysis_status="VALID", trade_action="WAIT", winner="bull"))
        # 3 VALID + NO_TRADE (no_trade)
        for i in range(3):
            mixed_pool.append(make_report(report_id=f"notrade-{i}", analysis_status="VALID", trade_action="NO_TRADE", winner="bear"))
        # 4 ABSTAIN (abstain)
        for i in range(4):
            mixed_pool.append(make_report(report_id=f"abstain-{i}", analysis_status="ABSTAIN", trade_action="WAIT", winner="tie"))
        # 2 INVALID_RUN (invalid_run)
        for i in range(2):
            mixed_pool.append(make_report(report_id=f"inv-{i}", analysis_status="INVALID_RUN", trade_action="NO_TRADE", winner="bull"))
        # 1 DATA_ERROR (data_error)
        mixed_pool.append(make_report(report_id="err-0", analysis_status="DATA_ERROR", trade_action="NO_TRADE", winner="bear"))
        # 6 legacy NULL (legacy_null)
        for i in range(6):
            mixed_pool.append(make_report(report_id=f"legacy-{i}", analysis_status=None, trade_action=None, winner="bull"))

        total_input = len(mixed_pool)
        assert total_input == (4 + 3 + 2) + (5 + 3 + 4 + 2 + 1 + 6)  # 9 qualifying + 21 excluded = 30

        filtered, excluded = filter_v2_completed_reports(mixed_pool, return_excluded_counts=True)
        assert len(filtered) == 9
        assert excluded["legacy_null"] == 6
        assert excluded["abstain"] == 4
        assert excluded["invalid_run"] == 2
        assert excluded["data_error"] == 1
        assert excluded["no_trade"] == 3
        assert excluded["wait"] == 5

        total_excluded = sum(excluded.values())
        assert total_excluded == 21
        assert len(filtered) + total_excluded == total_input

    def test_valid_only_bull_ratio_matches_diagnostic_approx_0_316(self):
        """Construct synthetic cohort matching the 23 production VALID samples (6 bull, 13 bear, 4 tie).

        Diagnosed bull_ratio is 6 / (6 + 13) = 6 / 19 ≈ 0.3158 ≈ 0.316.
        Verify that balance dimension calculates bull_ratio = 0.3158, revealing bearish tilt.
        """
        valid_reports = []
        for i in range(6):
            valid_reports.append(make_report(report_id=f"v-bull-{i}", analysis_status="VALID", trade_action="BUY", winner="bull"))
        for i in range(13):
            valid_reports.append(make_report(report_id=f"v-bear-{i}", analysis_status="VALID", trade_action="SELL", winner="bear", direction="看空"))
        for i in range(4):
            valid_reports.append(make_report(report_id=f"v-tie-{i}", analysis_status="VALID", trade_action="HOLD", winner="tie", direction="中性"))

        gate_res = evaluate_h1b_system_gates(valid_reports)
        det_balance = gate_res["matrix"]["dimension_balance"]["details"]
        assert det_balance["bull_ratio"] == pytest.approx(0.3158, abs=0.001)
        assert det_balance["side_diff"] == 7
        assert gate_res["matrix"]["dimension_balance"]["passed"] is False  # 0.3158 < 0.40 min balance

    def test_dimension_n_fails_when_sample_count_insufficient(self):
        """Contract #4: When sample count drops to 23 or 8 after filtering, dimension_n fails honestly.

        Threshold requires >= 60 samples. Must NOT pass or relax thresholds to make it green.
        """
        samples = [
            make_report(report_id=f"s-{i}", symbol=f"00000{i}.SZ", analysis_status="VALID", trade_action="BUY", winner="bull")
            for i in range(23)
        ]
        gate_res = evaluate_h1b_system_gates(samples)
        dim_n = gate_res["matrix"]["dimension_n"]
        assert dim_n["passed"] is False
        assert dim_n["details"]["sample_count"] == 23
        assert dim_n["details"]["min_required"] == 60
        assert gate_res["passed"] is False
        assert gate_res["recommendation"] == "KEEP_FALSE"

    def test_gate_evaluation_includes_excluded_counts_in_report(self):
        """Contract #2: Gate evaluation output includes excluded_counts in summary and matrix/report."""
        mixed_pool = [
            make_report(report_id="v-1", analysis_status="VALID", trade_action="BUY"),
            make_report(report_id="v-2", analysis_status="VALID", trade_action="WAIT"),
            make_report(report_id="v-3", analysis_status="ABSTAIN"),
            make_report(report_id="v-4", analysis_status=None),
        ]
        filtered, excluded = filter_v2_completed_reports(mixed_pool, return_excluded_counts=True)
        gate_res = evaluate_h1b_system_gates(filtered, excluded_counts=excluded)
        assert "excluded_counts" in gate_res
        assert gate_res["excluded_counts"]["wait"] == 1
        assert gate_res["excluded_counts"]["abstain"] == 1
        assert gate_res["excluded_counts"]["legacy_null"] == 1
        assert gate_res["summary"]["excluded_counts"]["wait"] == 1
