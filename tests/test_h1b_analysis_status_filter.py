"""Tests for H1b sample pool analysis_status filtering and excluded_counts (DAV-783 / D-009 §5).

Contract Requirements:
1. Filter strictly based on analysis_status, not winner (prevents winner leakage from ABSTAIN/etc.).
2. Excluded samples must be categorized and counted into `excluded_counts`:
   legacy_null, abstain, invalid_run, data_error, no_trade, wait.
3. Legacy samples with analysis_status IS NULL must be excluded and counted as legacy_null.
4. When qualifying samples drop below 60, dimension_n fails honestly (no relaxing filter).
5. Qualifying VALID samples preserve existing statistical calculation integrity.
6. Three-stage pipeline ledger: raw -> qualifying v2 -> D-009 eligible.
   excluded_counts only tracks exclusions among v2 samples; non-v2 samples are in non_v2_excluded.
7. Strict canonical trade_action: no fallback to legacy decision or action fields (RT-1).
8. All tests use in-memory fixtures; no network calls, no production DB access in tests.
"""

from typing import Any, Mapping, Optional
import pytest

from tradingagents.agents.utils.shadow_credit import (
    PROTOCOL_VERSION_V2_STRUCTURED,
    classify_v2_report_d009_exclusion,
    evaluate_h1b_system_gates,
    extract_report_analysis_status_and_action,
    filter_v2_completed_reports,
    is_qualifying_h1b_report,
    is_qualifying_v2_report,
    is_v2_protocol_report,
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
    extra_fields: Optional[Mapping[str, Any]] = None,
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

    if extra_fields:
        rep.update(extra_fields)

    return rep


class TestH1bAnalysisStatusFilter:
    """Test suite for analysis_status filtering, excluded_counts, and red team scenarios."""

    def test_rt1_valid_without_trade_action_and_legacy_decision_is_rejected(self):
        """RT-1 (Contract D-009 §5): analysis_status=VALID with NO trade_action and legacy decision=BUY

        Must be judged unqualified, classified as no_trade, and NOT leak into the qualified pool.
        Strictly forbids fallback to legacy 'decision' or 'action' fields.
        """
        rep = make_report(
            report_id="rt1-sample",
            analysis_status="VALID",
            trade_action=None,
            winner="bull",
            extra_fields={"decision": "BUY", "action": "BUY"},
        )
        assert "trade_action" not in rep
        assert rep.get("decision") == "BUY"

        st_val, act_val = extract_report_analysis_status_and_action(rep)
        assert st_val == "VALID"
        assert act_val is None  # Must NOT fall back to 'decision' or 'action'

        assert is_v2_protocol_report(rep) is True
        assert is_qualifying_v2_report(rep) is True  # Valid completed v2 report for backfill
        assert classify_v2_report_d009_exclusion(rep) == "no_trade"
        assert is_qualifying_h1b_report(rep) is False  # Rejected by D-009 §5 for H1b pool

        filtered, excluded, ledger = filter_v2_completed_reports([rep], return_ledger=True)
        assert len(filtered) == 0
        assert ledger["qualifying_v2_count"] == 1
        assert ledger["eligible_count"] == 0
        assert excluded["no_trade"] == 1

    def test_valid_buy_qualifies(self):
        """analysis_status=VALID with trade_action=BUY qualifies and enters pool."""
        rep = make_report(analysis_status="VALID", trade_action="BUY", winner="bull")
        assert is_qualifying_v2_report(rep) is True
        assert is_qualifying_h1b_report(rep) is True
        filtered = filter_v2_completed_reports([rep])
        assert len(filtered) == 1
        assert filtered[0]["id"] == rep["id"]

    def test_valid_sell_qualifies(self):
        """analysis_status=VALID with trade_action=SELL qualifies and enters pool."""
        rep = make_report(analysis_status="VALID", trade_action="SELL", winner="bear", direction="看空")
        assert is_qualifying_v2_report(rep) is True
        assert is_qualifying_h1b_report(rep) is True
        filtered = filter_v2_completed_reports([rep])
        assert len(filtered) == 1

    def test_valid_hold_qualifies(self):
        """analysis_status=VALID with trade_action=HOLD qualifies and enters pool."""
        rep = make_report(analysis_status="VALID", trade_action="HOLD", winner="tie", direction="中性")
        assert is_qualifying_v2_report(rep) is True
        assert is_qualifying_h1b_report(rep) is True
        filtered = filter_v2_completed_reports([rep])
        assert len(filtered) == 1

    def test_valid_no_trade_excluded_and_counted(self):
        """analysis_status=VALID with trade_action=NO_TRADE must be excluded from H1b pool and counted as no_trade."""
        rep = make_report(analysis_status="VALID", trade_action="NO_TRADE", winner="bull")
        assert is_qualifying_v2_report(rep) is True  # v2 completed report (backfill eligible)
        assert is_qualifying_h1b_report(rep) is False  # D-009 §5 ineligible
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["no_trade"] == 1

    def test_valid_wait_excluded_and_counted(self):
        """analysis_status=VALID with trade_action=WAIT must be excluded from H1b pool and counted as wait."""
        rep = make_report(analysis_status="VALID", trade_action="WAIT", winner="bear")
        assert is_qualifying_v2_report(rep) is True  # v2 completed report (backfill eligible)
        assert is_qualifying_h1b_report(rep) is False  # D-009 §5 ineligible
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["wait"] == 1

    def test_abstain_excluded_and_counted(self):
        """analysis_status=ABSTAIN must be excluded from H1b pool and counted as abstain."""
        rep = make_report(analysis_status="ABSTAIN", trade_action="WAIT", winner="tie")
        assert is_qualifying_v2_report(rep) is True  # v2 completed report
        assert is_qualifying_h1b_report(rep) is False  # D-009 §5 ineligible
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["abstain"] == 1

    def test_abstain_winner_leakage_prevented_like_6decef3d(self):
        """ABSTAIN sample with manager_verdict.winner='bear' (like 6decef3d) must be excluded from H1b pool.

        It must NOT leak into the bear sample count.
        """
        rep = make_report(
            report_id="6decef3d-test",
            analysis_status="ABSTAIN",
            trade_action="NO_TRADE",
            winner="bear",
            direction="看空",
        )
        assert is_qualifying_h1b_report(rep) is False
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["abstain"] == 1

        # Even if evaluated directly in gate evaluation, excluded samples must not leak into side count
        gate_res = evaluate_h1b_system_gates(filtered)
        assert gate_res["matrix"]["dimension_side"]["details"]["bear_samples"] == 0

    def test_invalid_run_excluded_and_counted(self):
        """analysis_status=INVALID_RUN must be excluded from H1b pool and counted as invalid_run."""
        rep = make_report(analysis_status="INVALID_RUN", trade_action="NO_TRADE", winner="bull")
        assert is_qualifying_h1b_report(rep) is False
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["invalid_run"] == 1

    def test_data_error_excluded_and_counted(self):
        """analysis_status=DATA_ERROR must be excluded from H1b pool and counted as data_error."""
        rep = make_report(analysis_status="DATA_ERROR", trade_action="NO_TRADE", winner="tie")
        assert is_qualifying_h1b_report(rep) is False
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["data_error"] == 1

    def test_legacy_null_excluded_and_counted(self):
        """Pre-D-009 legacy reports with analysis_status=None must be excluded from H1b pool and counted as legacy_null."""
        rep = make_report(analysis_status=None, trade_action=None, winner="bull")
        assert is_qualifying_h1b_report(rep) is False
        filtered, excluded = filter_v2_completed_reports([rep], return_excluded_counts=True)
        assert len(filtered) == 0
        assert excluded["legacy_null"] == 1

    def test_rt2_three_stage_ledger_and_excluded_counts_separation(self):
        """RT-2: Three-stage ledger accounting: raw -> qualifying v2 -> D-009 eligible.

        Verifies that non-v2 reports (legacy v1 or no winner) are excluded at Stage 2
        and NEVER conflated into D-009 excluded_counts.
        """
        pool = [
            # 5 non-v2 reports (Stage 1 -> Stage 2 failure)
            {"id": "non-v2-1", "status": "completed", "protocol_version": "v1_legacy"},
            {"id": "non-v2-2", "status": "completed", "protocol_version": "v1_legacy"},
            {"id": "non-v2-3", "status": "failed", "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED},
            {"id": "non-v2-4", "status": "completed", "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED},  # no winner
            {"id": "non-v2-5", "status": "running"},
            # 3 qualifying v2 reports that fail D-009 §5 (Stage 2 -> Stage 3 failure)
            make_report(report_id="v2-ex-1", analysis_status=None, trade_action=None),  # legacy_null
            make_report(report_id="v2-ex-2", analysis_status="ABSTAIN", trade_action="WAIT"),  # abstain
            make_report(report_id="v2-ex-3", analysis_status="VALID", trade_action="WAIT"),  # wait
            # 2 qualifying v2 reports that pass D-009 §5 (Eligible)
            make_report(report_id="v2-el-1", analysis_status="VALID", trade_action="BUY", winner="bull"),
            make_report(report_id="v2-el-2", analysis_status="VALID", trade_action="SELL", winner="bear"),
        ]

        total_raw = len(pool)
        assert total_raw == 10

        qualifying, excluded_counts, ledger = filter_v2_completed_reports(pool, return_ledger=True)

        assert ledger["raw_count"] == 10
        assert ledger["qualifying_v2_count"] == 5
        assert ledger["non_v2_excluded"] == 5
        assert ledger["eligible_count"] == 2
        assert ledger["d009_excluded"] == 3

        # excluded_counts must ONLY contain the 3 exclusions among v2 samples!
        assert sum(excluded_counts.values()) == 3
        assert excluded_counts["legacy_null"] == 1
        assert excluded_counts["abstain"] == 1
        assert excluded_counts["wait"] == 1
        assert excluded_counts["invalid_run"] == 0
        assert excluded_counts["data_error"] == 0
        assert excluded_counts["no_trade"] == 0

        assert len(qualifying) == 2
        assert [q["id"] for q in qualifying] == ["v2-el-1", "v2-el-2"]

    def test_rt6_mixed_pool_excluded_counts_total_integrity(self):
        """RT-6: Complete mixed pool fixture verifying zero silent drops across all categories."""
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

    def test_rt5_synthetic_valid_pool_winner_distribution_and_bull_ratio(self):
        """RT-5: Valid-only pool calculation reflects exact winner counts without distortion.

        Synthetic cohort matching the 23 production VALID samples (6 bull, 13 bear, 4 tie).
        Diagnosed bull_ratio is 6 / (6 + 13) = 6 / 19 ≈ 0.3158 ≈ 0.316 (revealing bearish tilt).
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

    def test_dimension_n_fails_honestly_when_sample_count_insufficient(self):
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

    def test_gate_evaluation_includes_excluded_counts_and_pipeline_ledger(self):
        """Contract #2 & RT-2: Gate evaluation output includes excluded_counts and pipeline_ledger."""
        mixed_pool = [
            make_report(report_id="v-1", analysis_status="VALID", trade_action="BUY"),
            make_report(report_id="v-2", analysis_status="VALID", trade_action="WAIT"),
            make_report(report_id="v-3", analysis_status="ABSTAIN"),
            make_report(report_id="v-4", analysis_status=None),
        ]
        filtered, excluded, ledger = filter_v2_completed_reports(mixed_pool, return_ledger=True)
        gate_res = evaluate_h1b_system_gates(filtered, excluded_counts=excluded, pipeline_ledger=ledger)
        assert "excluded_counts" in gate_res
        assert gate_res["excluded_counts"]["wait"] == 1
        assert gate_res["excluded_counts"]["abstain"] == 1
        assert gate_res["excluded_counts"]["legacy_null"] == 1
        assert gate_res["summary"]["excluded_counts"]["wait"] == 1
        assert "pipeline_ledger" in gate_res
        assert gate_res["pipeline_ledger"]["raw_count"] == 4
        assert gate_res["pipeline_ledger"]["eligible_count"] == 1
