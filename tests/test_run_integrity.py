"""D-009 P0-1: RunIntegrity — 0/7 … 7/7 analyst failure detection."""

from __future__ import annotations

from tradingagents.agents.utils.decision_status import (
    ACTION_NO_TRADE,
    ANALYSIS_DATA_ERROR,
    ANALYSIS_INVALID_RUN,
    DIRECTION_NA,
    apply_decision_status_to_result,
    is_calibration_eligible,
)
from tradingagents.agents.utils.run_integrity import (
    DEFAULT_REQUIRED_ANALYSTS,
    evaluate_run_integrity,
    evaluate_state_integrity,
    is_failed_analyst_report,
    resolve_decision_status_for_result,
)


_OK = "市场技术报告：突破阻力位，均线多头排列，成交量温和放大。"
_FAIL_502 = "分析报告生成失败：Error code: 502"


def _seven(*, failed: int = 0, body_ok: str = _OK, body_fail: str = _FAIL_502) -> dict:
    keys = [
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "macro_report",
        "smart_money_report",
        "volume_price_report",
    ]
    reports = {}
    for i, key in enumerate(keys):
        reports[key] = body_fail if i < failed else body_ok
    return reports


def test_failure_markers_detect_502_and_empty():
    assert is_failed_analyst_report(_FAIL_502)[0] is True
    assert is_failed_analyst_report("")[0] is True
    assert is_failed_analyst_report(None)[0] is True
    assert is_failed_analyst_report("本项不可用：上游超时")[0] is True
    assert is_failed_analyst_report(_OK)[0] is False
    # Short stubs without failure keywords are not auto-failed (compat with unit fixtures).
    assert is_failed_analyst_report("M")[0] is False


def test_zero_of_seven_failed_is_not_invalid():
    integrity = evaluate_run_integrity(_seven(failed=0))
    assert integrity.failed_required_count == 0
    assert integrity.all_required_failed is False
    assert integrity.analysis_status is None
    assert integrity.decision_status is None
    assert integrity.required_count == 7
    assert set(integrity.required_analysts) == set(DEFAULT_REQUIRED_ANALYSTS)


def test_partial_failures_do_not_trigger_invalid_run():
    for n in (1, 3, 6):
        integrity = evaluate_run_integrity(_seven(failed=n))
        assert integrity.failed_required_count == n
        assert integrity.all_required_failed is False
        assert integrity.analysis_status == "PARTIAL"
        assert integrity.decision_status is not None
        assert integrity.decision_status["trade_action"] == "NO_TRADE"


def test_seven_of_seven_502_is_invalid_run_no_trade():
    integrity = evaluate_run_integrity(_seven(failed=7))
    assert integrity.all_required_failed is True
    assert integrity.failed_required_count == 7
    assert integrity.analysis_status == ANALYSIS_INVALID_RUN
    assert integrity.failure_class == ANALYSIS_DATA_ERROR
    ds = integrity.decision_status
    assert ds is not None
    assert ds["analysis_status"] == ANALYSIS_INVALID_RUN
    assert ds["failure_class"] == ANALYSIS_DATA_ERROR
    assert ds["direction"] == DIRECTION_NA
    assert ds["trade_action"] == ACTION_NO_TRADE
    assert ds["confidence"] is None
    assert ds["probability"] is None
    assert any("7_of_7" in code for code in integrity.reason_codes)


def test_all_empty_reports_are_invalid_run():
    integrity = evaluate_run_integrity(_seven(failed=7, body_fail=""))
    assert integrity.all_required_failed is True
    assert integrity.decision_status["trade_action"] == ACTION_NO_TRADE


def test_apply_status_nulls_fabricated_numbers():
    result = {
        **_seven(failed=7),
        "decision": "HOLD",
        "direction": "NEUTRAL",
        "confidence": 25,
        "probability": 0.55,
        "target_price": 28.6,
        "stop_loss_price": 24.8,
        "upside": 0.12,
        "downside": 0.08,
    }
    integrity = evaluate_run_integrity(result)
    apply_decision_status_to_result(result, integrity.decision_status)
    assert result["analysis_status"] == ANALYSIS_INVALID_RUN
    assert result["trade_action"] == ACTION_NO_TRADE
    assert result["decision"] == ACTION_NO_TRADE
    assert result["direction"] == DIRECTION_NA
    assert result["confidence"] is None
    assert result["probability"] is None
    assert result["target_price"] is None
    assert result["stop_loss_price"] is None
    assert is_calibration_eligible(result) is False


def test_resolve_decision_status_recomputes_from_reports():
    result = {
        **_seven(failed=7),
        "decision": "HOLD",
        "confidence": 25,
    }
    status = resolve_decision_status_for_result(result)
    assert status is not None
    assert status.analysis_status == ANALYSIS_INVALID_RUN
    assert status.trade_action == ACTION_NO_TRADE


def test_resolve_prefers_integrity_over_stale_buy_status():
    """7/7 failures must beat a leftover BUY decision_status on the payload."""
    result = {
        **_seven(failed=7),
        "decision_status": {
            "analysis_status": "VALID",
            "direction": "BULL",
            "trade_action": "BUY",
            "risk_status": "OK",
        },
        "decision": "BUY",
        "confidence": 80,
    }
    status = resolve_decision_status_for_result(result)
    assert status is not None
    assert status.analysis_status == ANALYSIS_INVALID_RUN
    assert status.trade_action == ACTION_NO_TRADE


def test_evaluate_state_integrity_uses_selected_analysts():
    state = {
        "selected_analysts": ["market", "news"],
        "market_report": _FAIL_502,
        "news_report": _FAIL_502,
        "sentiment_report": _OK,  # not required → ignored
    }
    integrity = evaluate_state_integrity(state)
    assert integrity.required_count == 2
    assert integrity.all_required_failed is True
    assert set(integrity.failed_required) == {"market", "news"}


def test_manifest_marks_failed_reports_not_passed():
    from tradingagents.agents.utils.debate_utils import build_debate_report_manifest

    manifest = build_debate_report_manifest(_seven(failed=7))
    assert all(item["passed"] is False for item in manifest.values())
    assert all(item["mode"] == "failed" for item in manifest.values())


# =====================================================================
# E-03c decision_status Consumption & Read-back Boundary Tests
# =====================================================================

def test_e03c_decision_status_consumes_adopt_partial_reject():
    """E-03c: Decision status consumes adopted / partial / rejected claim decisions deterministically."""
    from tradingagents.agents.utils.decision_status import (
        evaluate_confirmation_state,
        CONFIRM_CONFIRMED,
        CONFIRM_PARTIAL,
        CONFIRM_UNRESOLVED,
    )

    claims = [
        {"claim_id": "C-1", "claim": "主力流入", "evidence": ["E1"]},
        {"claim_id": "C-2", "claim": "突破均线", "evidence": ["E2"]},
    ]

    # 1. Adopted & fully verified core claims -> CONFIRMED
    summary_adopt = {
        "C-1": {"decision": "adopt", "counts": {"total": 1, "verified": 1, "contradicted": 0, "source_unavailable": 0}},
        "C-2": {"decision": "adopt", "counts": {"total": 1, "verified": 1, "contradicted": 0, "source_unavailable": 0}},
    }
    st_adopt, codes_adopt = evaluate_confirmation_state(
        focus_claim_ids=["C-1", "C-2"],
        claim_evidence_summary=summary_adopt,
        adopted_claim_ids=["C-1", "C-2"],
    )
    assert st_adopt == CONFIRM_CONFIRMED
    assert any("all_core_claims_verified" in c for c in codes_adopt)

    # 2. Partially adopted factual core claims (mixed evidence) -> PARTIAL (WAIT)
    summary_partial = {
        "C-1": {"decision": "adopt", "counts": {"total": 1, "verified": 1, "contradicted": 0, "source_unavailable": 0}},
        "C-2": {"decision": "partial", "counts": {"total": 3, "verified": 2, "unsupported": 1, "contradicted": 0, "source_unavailable": 0}},
    }
    st_partial, codes_partial = evaluate_confirmation_state(
        focus_claim_ids=["C-1", "C-2"],
        claim_evidence_summary=summary_partial,
        adopted_claim_ids=["C-1"],
        partially_adopted_claims=["C-2"],
    )
    assert st_partial == CONFIRM_PARTIAL
    assert any("partial_core_claims" in c or "partially_adopted_claims" in c for c in codes_partial)

    # 3. Rejected core claims -> UNRESOLVED (WAIT)
    summary_reject = {
        "C-1": {"decision": "reject", "counts": {"total": 1, "verified": 0, "unsupported": 1, "contradicted": 0, "source_unavailable": 0}},
        "C-2": {"decision": "reject", "counts": {"total": 1, "verified": 0, "unsupported": 1, "contradicted": 0, "source_unavailable": 0}},
    }
    st_reject, codes_reject = evaluate_confirmation_state(
        focus_claim_ids=["C-1", "C-2"],
        claim_evidence_summary=summary_reject,
        rejected_claim_ids=["C-1", "C-2"],
    )
    assert st_reject == CONFIRM_UNRESOLVED
    assert any("unverified_core_claims" in c for c in codes_reject)


def test_e03c_pit_failure_cannot_be_whitewashed_by_direction_or_adopted_status():
    """E-03c: PIT lookahead failure cannot be whitewashed by directional verdict or old adopted status."""
    from tradingagents.agents.utils.decision_status import (
        status_from_manager_verdict,
        ANALYSIS_ABSTAIN,
        ACTION_NO_TRADE,
        RISK_BLOCKED,
    )

    # Claim has PIT failure (contradicted)
    claims = [
        {"claim_id": "INV-PIT", "claim": "未来突破新高", "evidence": ["2026-09-20突破1800元"], "status": "adopted"},
    ]
    summary = {
        "INV-PIT": {
            "decision": "reject",
            "pit_failed": True,
            "counts": {"total": 1, "verified": 0, "unsupported": 0, "contradicted": 1, "source_unavailable": 0},
            "reason": "存在前视偏差/PIT失败 (pit_date=2026-09-20 > baseline=2026-09-08)",
        }
    }
    # Manager attempts to whitewash the PIT failure by adopting it and issuing BULL BUY
    manager_verdict = {
        "direction": "看多",
        "winner": "bull",
        "adopted_claim_ids": ["INV-PIT"],
        "partially_adopted_claims": [],
        "rejected_claim_ids": [],
        "consistency_check_passed": True,  # Manager claims pass, but verifier ledger knows it failed PIT
    }
    status = status_from_manager_verdict(
        manager_verdict,
        claims=claims,
        claim_evidence_summary=summary,
        focus_claim_ids=["INV-PIT"],
    )
    # Must fail closed: ABSTAIN, NO_TRADE, BLOCKED
    assert status.analysis_status == ANALYSIS_ABSTAIN
    assert status.trade_action == ACTION_NO_TRADE
    assert status.risk_status == RISK_BLOCKED
    assert any("manager_consistency_hard_gate" in c or "fatal_adopted_claims" in c for c in status.reason_codes)


def test_e03c_observation_hypotheses_do_not_unconditionally_force_wait():
    """E-03c: Observation/hypotheses stay in audited state without unconditionally forcing entire order to WAIT."""
    from tradingagents.agents.utils.decision_status import (
        evaluate_confirmation_state,
        status_from_manager_verdict,
        ANALYSIS_VALID,
        ACTION_BUY,
        ACTION_WAIT,
        CONFIRM_CONFIRMED,
        CONFIRM_UNRESOLVED,
    )

    # 1. Valid factual core claim + observation claim -> trade action is BUY (not WAIT)
    claims = [
        {"claim_id": "CLM-FACT", "claim": "主力净流入1.2亿", "evidence": ["E1"], "claim_type": "fact"},
        {"claim_id": "CLM-OBS", "claim": "【观察】Wyckoff吸筹形态初显", "evidence": ["E2"], "claim_type": "observation"},
    ]
    summary = {
        "CLM-FACT": {"decision": "adopt", "counts": {"total": 1, "verified": 1, "contradicted": 0, "source_unavailable": 0}, "is_observation_or_hypothesis": False},
        "CLM-OBS": {"decision": "partial", "counts": {"total": 1, "verified": 1, "contradicted": 0, "source_unavailable": 0}, "is_observation_or_hypothesis": True},
    }
    conf_state, conf_codes = evaluate_confirmation_state(
        focus_claim_ids=["CLM-FACT", "CLM-OBS"],
        claim_evidence_summary=summary,
        claims=claims,
        adopted_claim_ids=["CLM-FACT"],
        partially_adopted_claims=["CLM-OBS"],
    )
    assert conf_state == CONFIRM_CONFIRMED
    assert any("audited_observation_claims:CLM-OBS" in c for c in conf_codes)

    mv = {
        "direction": "看多",
        "winner": "bull",
        "adopted_claim_ids": ["CLM-FACT"],
        "partially_adopted_claims": ["CLM-OBS"],
        "consistency_check_passed": True,
        "stop_loss": "1500元",
        "entry": "1600元",
    }
    ds = status_from_manager_verdict(
        mv,
        claims=claims,
        claim_evidence_summary=summary,
        focus_claim_ids=["CLM-FACT", "CLM-OBS"],
    )
    assert ds.analysis_status == ANALYSIS_VALID
    assert ds.trade_action == ACTION_BUY

    # 2. Negative case: ONLY observation exists, NO factual verified core -> UNRESOLVED and WAIT
    obs_only_claims = [
        {"claim_id": "CLM-OBS-ONLY", "claim": "【假设】可能突破阻力位", "is_hypothesis": True},
    ]
    obs_only_summary = {
        "CLM-OBS-ONLY": {"decision": "partial", "counts": {"total": 1, "verified": 1, "contradicted": 0, "source_unavailable": 0}, "is_observation_or_hypothesis": True},
    }
    conf_state_obs, conf_codes_obs = evaluate_confirmation_state(
        focus_claim_ids=["CLM-OBS-ONLY"],
        claim_evidence_summary=obs_only_summary,
        claims=obs_only_claims,
        partially_adopted_claims=["CLM-OBS-ONLY"],
    )
    assert conf_state_obs == CONFIRM_UNRESOLVED
    ds_obs = status_from_manager_verdict(
        mv,
        claims=obs_only_claims,
        claim_evidence_summary=obs_only_summary,
        focus_claim_ids=["CLM-OBS-ONLY"],
    )
    assert ds_obs.trade_action == ACTION_WAIT


def test_e03c_decision_status_stratification_no_fabricated_numbers():
    """E-03c: Stratification between INVALID/PARTIAL/ABSTAIN vs VALID/NEUTRAL; no fabricated probability/confidence."""
    from tradingagents.agents.utils.decision_status import (
        invalid_run_status,
        abstain_status,
        partial_status,
        valid_status,
        apply_decision_status_to_result,
        ANALYSIS_INVALID_RUN,
        ANALYSIS_ABSTAIN,
        ANALYSIS_PARTIAL,
        ANALYSIS_VALID,
        DIRECTION_NA,
        DIRECTION_NEUTRAL,
        ACTION_NO_TRADE,
        ACTION_HOLD,
        ACTION_WAIT,
    )

    # 1. INVALID_RUN has no probability or confidence
    st_inv = invalid_run_status()
    assert st_inv.analysis_status == ANALYSIS_INVALID_RUN
    assert st_inv.direction == DIRECTION_NA
    assert st_inv.trade_action == ACTION_NO_TRADE
    assert st_inv.confidence is None
    assert st_inv.probability is None

    # 2. ABSTAIN has no probability or confidence
    st_abs = abstain_status()
    assert st_abs.analysis_status == ANALYSIS_ABSTAIN
    assert st_abs.direction == DIRECTION_NA
    assert st_abs.trade_action == ACTION_NO_TRADE
    assert st_abs.confidence is None
    assert st_abs.probability is None

    # 3. PARTIAL has no probability or confidence
    st_part = partial_status()
    assert st_part.analysis_status == ANALYSIS_PARTIAL
    assert st_part.direction == DIRECTION_NA
    assert st_part.trade_action == ACTION_NO_TRADE
    assert st_part.confidence is None
    assert st_part.probability is None

    # 4. NEUTRAL (VALID) is a valid market view (HOLD) distinct from DATA_ERROR/ABSTAIN
    st_neut = valid_status(direction=DIRECTION_NEUTRAL, trade_action=ACTION_HOLD)
    assert st_neut.analysis_status == ANALYSIS_VALID
    assert st_neut.direction == DIRECTION_NEUTRAL
    assert st_neut.trade_action == ACTION_HOLD

    # 5. apply_decision_status_to_result strips all fabricated target / stop-loss / ranges
    dirty_result = {
        "analysis_status": "VALID",
        "target_price": 1800.0,
        "stop_loss_price": 1400.0,
        "upside": 0.20,
        "downside": 0.10,
        "odds": 2.0,
        "confidence": 85,
        "probability": 0.75,
    }
    apply_decision_status_to_result(dirty_result, st_inv)
    assert dirty_result["analysis_status"] == ANALYSIS_INVALID_RUN
    assert dirty_result["trade_action"] == ACTION_NO_TRADE
    assert dirty_result["target_price"] is None
    assert dirty_result["stop_loss_price"] is None
    assert dirty_result["upside"] is None
    assert dirty_result["downside"] is None
    assert dirty_result["odds"] is None
    assert dirty_result["confidence"] is None
    assert dirty_result["probability"] is None


def test_e03c_nested_decision_status_roundtrip_consistency():
    """E-03c: Nested decision_status read-back round-trips with full consistency across layers."""
    from tradingagents.agents.utils.decision_status import (
        DecisionStatus,
        decision_status_from_state,
        decision_status_from_mapping,
        valid_status,
        ANALYSIS_VALID,
        DIRECTION_BULL,
        ACTION_BUY,
        RISK_OK,
        CONFIRM_CONFIRMED,
    )

    original = valid_status(
        direction=DIRECTION_BULL,
        trade_action=ACTION_BUY,
        risk_status=RISK_OK,
        confirmation_state=CONFIRM_CONFIRMED,
        confidence=80,
        probability=0.72,
        reason_codes=["code_1", "code_2"],
    )

    # 1. Direct mapping round-trip
    d_map = original.to_dict()
    recovered = decision_status_from_mapping(d_map)
    assert recovered is not None
    assert recovered.analysis_status == original.analysis_status
    assert recovered.direction == original.direction
    assert recovered.trade_action == original.trade_action
    assert recovered.risk_status == original.risk_status
    assert recovered.confirmation_state == original.confirmation_state
    assert recovered.confidence == original.confidence
    assert recovered.probability == original.probability
    assert recovered.reason_codes == original.reason_codes

    # 2. State nested in manager_verdict round-trip
    state_mv = {
        "manager_verdict": {
            "decision_status": d_map,
        }
    }
    recovered_mv = decision_status_from_state(state_mv)
    assert recovered_mv is not None
    assert recovered_mv.trade_action == ACTION_BUY
    assert recovered_mv.direction == DIRECTION_BULL

    # 3. State nested in result_data round-trip
    state_rd = {
        "result_data": {
            "decision_status": original,  # as dataclass object
        }
    }
    recovered_rd = decision_status_from_state(state_rd)
    assert recovered_rd is not None
    assert recovered_rd.trade_action == ACTION_BUY
    assert recovered_rd.confidence == 80
