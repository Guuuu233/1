"""Tests for P0-5b: confirmation_state hard gate and WAIT/NO_TRADE action routing.

Ensures that:
1. Unresolved focus/core claims with no verified evidence -> confirmation_state=UNRESOLVED, trade_action=WAIT.
2. Partial core claims verified -> confirmation_state=PARTIAL, trade_action=WAIT.
3. All core claims verified without fatal conflict -> confirmation_state=CONFIRMED, trade_action follows verdict mapping.
4. Fund flow guard / consistency hard gate take precedence over confirmation (ABSTAIN/NO_TRADE preserved).
5. Goertek minimal nail: simulation of unconfirmed core disagreement -> manager terminal status is WAIT,
   investment plan is WAIT/NO_TRADE and Trader does not generate buy execution plan.
6. winner=tie with unconfirmed claims -> WAIT (not eligible Neutral/HOLD trade action).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.agent_states import PROTOCOL_VERSION_V2_STRUCTURED
from tradingagents.agents.utils.decision_status import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_NO_TRADE,
    ACTION_SELL,
    ACTION_WAIT,
    ANALYSIS_ABSTAIN,
    ANALYSIS_INVALID_RUN,
    ANALYSIS_VALID,
    CONFIRM_CONFIRMED,
    CONFIRM_PARTIAL,
    CONFIRM_UNRESOLVED,
    DIRECTION_BEAR,
    DIRECTION_BULL,
    DIRECTION_NA,
    DIRECTION_NEUTRAL,
    DecisionStatus,
    apply_decision_status_to_result,
    evaluate_confirmation_state,
    is_calibration_eligible,
    is_non_executable_status,
    status_from_manager_verdict,
)


def _base_manager_state(**overrides):
    claims = [
        {
            "claim_id": "CLM-1",
            "speaker": "Bull Analyst",
            "speaker_key": "Bull",
            "stance": "bullish",
            "claim": "歌尔声学新产线良率已达98%且订单饱满",
            "evidence": ["产线良率报告", "订单排产表"],
            "confidence": 0.85,
            "battlefield": "fundamentals",
        },
        {
            "claim_id": "CLM-2",
            "speaker": "Bear Analyst",
            "speaker_key": "Bear",
            "stance": "bearish",
            "claim": "海外大客户三季度砍单20%导致产能闲置",
            "evidence": ["供应链传闻", "海外券商研报"],
            "confidence": 0.80,
            "battlefield": "macro_policy",
        },
    ]
    round_messages = [
        {
            "message_index": 1,
            "stage": "opening",
            "speaker": "Bull Analyst",
            "speaker_key": "Bull",
            "parse_status": "valid",
            "accepted": True,
            "responded_claim_ids": [],
            "target_claim_ids": [],
            "new_claim_ids": ["CLM-1"],
            "information_gain_score": 1.0,
        },
        {
            "message_index": 2,
            "stage": "opening",
            "speaker": "Bear Analyst",
            "speaker_key": "Bear",
            "parse_status": "valid",
            "accepted": True,
            "responded_claim_ids": [],
            "target_claim_ids": [],
            "new_claim_ids": ["CLM-2"],
            "information_gain_score": 1.0,
        },
        {
            "message_index": 3,
            "stage": "challenge",
            "speaker": "Bull Analyst",
            "speaker_key": "Bull",
            "parse_status": "valid",
            "accepted": True,
            "responded_claim_ids": ["CLM-2"],
            "target_claim_ids": ["CLM-2"],
            "new_claim_ids": [],
            "challenges": [{"target_claim_id": "CLM-2", "weakest_point": "传闻未经官方证实", "evidence": ["未收到调整通知"], "severity": "major"}],
            "information_gain_score": 0.9,
        },
        {
            "message_index": 4,
            "stage": "challenge",
            "speaker": "Bear Analyst",
            "speaker_key": "Bear",
            "parse_status": "valid",
            "accepted": True,
            "responded_claim_ids": ["CLM-1"],
            "target_claim_ids": ["CLM-1"],
            "new_claim_ids": [],
            "challenges": [{"target_claim_id": "CLM-1", "weakest_point": "爬坡期良率波动大", "evidence": ["历史良率仅85%"], "severity": "major"}],
            "information_gain_score": 0.88,
        },
    ]
    state = {
        "macro_report": "宏观经济报告：政策支持，流动性充裕。",
        "market_report": "市场技术报告：均线多头排列，量能稳步放大。",
        "sentiment_report": "市场情绪报告：情绪适度乐观，无过热迹象。",
        "news_report": "新闻舆情报告：行业订单增加，供应链恢复正常。",
        "fundamentals_report": "基本面分析报告：营收稳步增长，现金流充沛。",
        "smart_money_report": "主力资金报告：大单与超大单呈现净流入。",
        "volume_price_report": "量价分析报告：放量突破重要阻力位，形态健康。",
        "market_data_context": {
            "analysis_baseline_date": "2026-05-28",
            "source_provenance": {
                "stock_data": {"status": "available", "as_of": "2026-05-28"},
            },
        },
        "investment_debate_state": {
            "history": "多空双方围绕核心增长假设展开辩论。",
            "bull_history": "多头主张订单交付超预期。",
            "bear_history": "空头质疑产品良率与下半年砍单风险。",
            "current_speaker": "",
            "current_response": "",
            "count": 4,
            "claims": claims,
            "round_messages": round_messages,
            "challenges": [
                {"speaker_key": "Bull", "target_claim_id": "CLM-2", "message_index": 3},
                {"speaker_key": "Bear", "target_claim_id": "CLM-1", "message_index": 4},
            ],
            "focus_claim_ids": ["CLM-1", "CLM-2"],
            "open_claim_ids": ["CLM-1", "CLM-2"],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": ["CLM-1", "CLM-2"],
            "round_summary": "多空双方在良率与砍单核心分歧上未达成一致。",
            "round_goal": "核实新产线良率与砍单传闻",
            "claim_counter": 2,
            "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
            "feature_flags": {"v2_debate_enabled": True},
            "tiebreak_skipped": True,
        },
        "fund_flow_consensus_guard": {
            "blocked": False,
            "direction_allowed": True,
            "status": "consensus",
        },
        "trade_date": "2026-05-28",
        "horizon": "medium",
        "data_gaps": [],
    }
    state.update(overrides)
    return state


def _mock_llm(verdict_text: str):
    llm = MagicMock()
    calls = {"n": 0}

    async def _astream(*_a, **_k):
        calls["n"] += 1
        yield MagicMock(content=verdict_text)

    llm.astream = _astream
    return llm, calls


# ── Unit Tests for status_from_manager_verdict with confirmation ───────────────


def test_status_from_manager_verdict_unresolved_focus_claims_wait():
    """Unresolved focus claims without verified evidence -> UNRESOLVED + WAIT."""
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 60,
        "consistency_check_passed": True,
        "failed_checks": [],
    }
    # Both CLM-1 and CLM-2 are unresolved with no verified evidence (unsupported)
    claims_verification = [
        {"claim_id": "CLM-1", "status": "unsupported", "raw": "产线良率报告未获证实"},
        {"claim_id": "CLM-2", "status": "unsupported", "raw": "海外砍单传闻未获证实"},
    ]
    status = status_from_manager_verdict(
        mv,
        focus_claim_ids=["CLM-1", "CLM-2"],
        unresolved_claim_ids=["CLM-1", "CLM-2"],
        claims_verification=claims_verification,
    )
    assert status.analysis_status == ANALYSIS_VALID
    assert status.confirmation_state == CONFIRM_UNRESOLVED
    assert status.trade_action == ACTION_WAIT
    assert status.direction == DIRECTION_BULL
    assert is_non_executable_status(status) is True
    assert is_calibration_eligible(status) is False


def test_status_from_manager_verdict_partial_focus_claims_wait():
    """Partial focus claims verified, others unverified -> PARTIAL + WAIT."""
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 50,
        "consistency_check_passed": True,
        "failed_checks": [],
    }
    # CLM-1 is verified, CLM-2 is unsupported
    claims_verification = [
        {"claim_id": "CLM-1", "status": "verified", "raw": "产线良率98%已由基本面报告证实"},
        {"claim_id": "CLM-2", "status": "unsupported", "raw": "砍单传闻缺乏数据支撑"},
    ]
    claim_evidence_summary = {
        "CLM-1": {
            "counts": {"total": 1, "verified": 1, "unsupported": 0, "contradicted": 0, "source_unavailable": 0},
            "coverage": 1.0,
            "decision": "adopt",
        },
        "CLM-2": {
            "counts": {"total": 1, "verified": 0, "unsupported": 1, "contradicted": 0, "source_unavailable": 0},
            "coverage": 0.0,
            "decision": "reject",
        },
    }
    status = status_from_manager_verdict(
        mv,
        focus_claim_ids=["CLM-1", "CLM-2"],
        claims_verification=claims_verification,
        claim_evidence_summary=claim_evidence_summary,
    )
    assert status.confirmation_state == CONFIRM_PARTIAL
    assert status.trade_action == ACTION_WAIT
    assert status.direction == DIRECTION_BULL
    assert is_non_executable_status(status) is True


def test_status_from_manager_verdict_all_core_verified_confirmed():
    """All core focus claims verified without fatal conflict -> CONFIRMED + BUY."""
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 60,
        "consistency_check_passed": True,
        "failed_checks": [],
    }
    claims_verification = [
        {"claim_id": "CLM-1", "status": "verified", "raw": "良率数据已证实"},
        {"claim_id": "CLM-2", "status": "verified", "raw": "砍单传闻已被澄清辟谣公告证实"},
    ]
    claim_evidence_summary = {
        "CLM-1": {
            "counts": {"total": 1, "verified": 1, "unsupported": 0, "contradicted": 0, "source_unavailable": 0},
            "coverage": 1.0,
            "decision": "adopt",
        },
        "CLM-2": {
            "counts": {"total": 1, "verified": 1, "unsupported": 0, "contradicted": 0, "source_unavailable": 0},
            "coverage": 1.0,
            "decision": "adopt",
        },
    }
    status = status_from_manager_verdict(
        mv,
        focus_claim_ids=["CLM-1", "CLM-2"],
        claims_verification=claims_verification,
        claim_evidence_summary=claim_evidence_summary,
    )
    assert status.confirmation_state == CONFIRM_CONFIRMED
    assert status.trade_action == ACTION_BUY
    assert status.direction == DIRECTION_BULL
    assert is_non_executable_status(status) is False


def test_status_from_manager_verdict_contradicted_core_claim_unresolved():
    """Core claim has contradicted/fatal evidence -> UNRESOLVED + WAIT."""
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 60,
        "consistency_check_passed": True,
        "failed_checks": [],
    }
    claims_verification = [
        {"claim_id": "CLM-1", "status": "verified", "raw": "产线良率已证实"},
        {"claim_id": "CLM-2", "status": "contradicted", "raw": "公告显示砍单事实成立，多头假设被证伪", "is_fatal": True},
    ]
    status = status_from_manager_verdict(
        mv,
        focus_claim_ids=["CLM-1", "CLM-2"],
        claims_verification=claims_verification,
    )
    assert status.confirmation_state == CONFIRM_UNRESOLVED
    assert status.trade_action == ACTION_WAIT


def test_status_from_manager_verdict_tie_with_unresolved_claims_is_wait():
    """winner=tie with unresolved claims must be WAIT, not Neutral+HOLD."""
    mv = {
        "direction": "中性",
        "winner": "tie",
        "position_pct": 0,
        "consistency_check_passed": True,
        "failed_checks": [],
    }
    claims_verification = [
        {"claim_id": "CLM-1", "status": "unsupported", "raw": "无证据"},
    ]
    status = status_from_manager_verdict(
        mv,
        focus_claim_ids=["CLM-1"],
        claims_verification=claims_verification,
    )
    assert status.confirmation_state == CONFIRM_UNRESOLVED
    assert status.trade_action == ACTION_WAIT
    assert status.trade_action != ACTION_HOLD
    assert is_non_executable_status(status) is True


def test_status_from_manager_verdict_tie_with_all_confirmed_is_hold():
    """winner=tie with no unconfirmed disputes -> CONFIRMED + HOLD."""
    mv = {
        "direction": "中性",
        "winner": "tie",
        "position_pct": 0,
        "consistency_check_passed": True,
        "failed_checks": [],
    }
    status = status_from_manager_verdict(
        mv,
        focus_claim_ids=[],
        unresolved_claim_ids=[],
        claims_verification=[],
    )
    assert status.confirmation_state == CONFIRM_CONFIRMED
    assert status.trade_action == ACTION_HOLD


def test_prior_gates_take_precedence_over_confirmation():
    """Upstream INVALID_RUN and ABSTAIN are not downgraded to CONFIRMED."""
    # 1. Consistency failure -> ABSTAIN / NO_TRADE
    mv_inconsistent = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 60,
        "consistency_check_passed": False,
        "failed_checks": ["stop_loss_missing"],
    }
    st1 = status_from_manager_verdict(
        mv_inconsistent,
        focus_claim_ids=[],  # even if clean claims
    )
    assert st1.analysis_status == ANALYSIS_ABSTAIN
    assert st1.trade_action == ACTION_NO_TRADE
    assert st1.confirmation_state == CONFIRM_UNRESOLVED

    # 2. Upstream PARTIAL failure -> PARTIAL / NO_TRADE
    st2 = status_from_manager_verdict(
        {"direction": "看多", "winner": "bull", "consistency_check_passed": True},
        prior_analysis_status="PARTIAL",
    )
    assert st2.analysis_status == "PARTIAL"
    assert st2.trade_action == ACTION_NO_TRADE


# ── Integration: Goertek Nail & Research Manager Node ─────────────────────────


def test_goertek_nail_unconfirmed_core_disagreement_emits_wait():
    """Goertek sample: unconfirmed core disagreement -> manager emits WAIT and Trader does not buy."""
    state = _base_manager_state()

    # The LLM generates a bullish text verdict, but the debate has unverified focus claims
    verdict_text = """
### 辩论裁决与总结方案
【多空辩论五步深度裁决】
多头关于新产线良率的主张具有进攻性，空头关于砍单的质疑缺乏一手数据。
综合评定：多头胜。
建议仓位：50%，止损位：28.50元。
<!-- VERDICT: {"direction": "看多", "winner": "bull", "reason": "看好新产线交付", "position_pct": 50, "entry": "30.00", "target": "35.00", "stop_loss": "28.50", "confidence": 75, "probability": 0.70} -->
"""
    llm, calls = _mock_llm(verdict_text)
    memory = MagicMock()
    memory.get_memories.return_value = []
    manager_node = create_research_manager(llm, memory)

    manager_res = asyncio.run(manager_node(state))

    # Assertions on Research Manager output
    assert manager_res["analysis_status"] == ANALYSIS_VALID
    assert manager_res["confirmation_state"] == CONFIRM_UNRESOLVED
    assert manager_res["trade_action"] == ACTION_WAIT
    assert manager_res["decision_status"]["trade_action"] == ACTION_WAIT
    assert manager_res["decision_status"]["confirmation_state"] == CONFIRM_UNRESOLVED

    # Now pass state to Trader node: Trader must short-circuit and NOT generate buy orders
    trader_state = {
        "company_of_interest": "002241.SZ",
        "investment_plan": manager_res["investment_plan"],
        "trader_investment_plan": "",
        "market_report": state["market_report"],
        "sentiment_report": state["sentiment_report"],
        "news_report": state["news_report"],
        "fundamentals_report": state["fundamentals_report"],
        "risk_feedback_state": {},
        "fund_flow_consensus_guard": state["fund_flow_consensus_guard"],
        "decision_status": manager_res["decision_status"],
        "analysis_status": manager_res["analysis_status"],
        "trade_action": manager_res["trade_action"],
        "confirmation_state": manager_res["confirmation_state"],
        "instrument_context": {},
        "market_context": {},
        "user_context": {},
    }
    trader_llm, trader_calls = _mock_llm("次日开仓买入 50% 仓位")
    trader_node = create_trader(trader_llm, memory)
    trader_res = asyncio.run(trader_node(trader_state))

    assert trader_calls["n"] == 0, "Trader LLM must not be invoked when confirmation_state is UNRESOLVED / WAIT"
    assert "观望" in trader_res["trader_investment_plan"] or "WAIT" in trader_res["trader_investment_plan"] or "NO_TRADE" in trader_res["trader_investment_plan"]
    assert "买入 50%" not in trader_res["trader_investment_plan"]
    assert "次日开仓" not in trader_res["trader_investment_plan"]


def test_apply_decision_status_with_wait_strips_targets():
    """apply_decision_status_to_result on WAIT must strip price targets and set confirmation_state."""
    raw = {
        "target_price": 35.0,
        "stop_loss_price": 28.5,
        "confidence": 75,
        "probability": 0.7,
        "numeric_ranges": ["28.5-35.0"],
    }
    st = DecisionStatus(
        analysis_status=ANALYSIS_VALID,
        direction=DIRECTION_BULL,
        trade_action=ACTION_WAIT,
        risk_status="OK",
        confirmation_state=CONFIRM_UNRESOLVED,
    )
    result = apply_decision_status_to_result(raw, st)
    assert result["trade_action"] == ACTION_WAIT
    assert result["confirmation_state"] == CONFIRM_UNRESOLVED
    assert result["target_price"] is None
    assert result["stop_loss_price"] is None
    assert result["confidence"] is None
    assert result["probability"] is None
    assert result["numeric_ranges"] == []


# ── Card A: Confirmation Gate Claim Lifecycle Tests ─────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "decision_semantics"
MIDEA_FIXTURE_PATH = FIXTURES_DIR / "midea_confirmation_fixture.json"


def test_midea_real_fixture_non_core_rejected_contradiction_does_not_block():
    """1. Midea 000333.SZ fixture: rejected + deterministic reject of non-core INV-6 does NOT block confirmation."""
    assert MIDEA_FIXTURE_PATH.exists() is True
    with open(MIDEA_FIXTURE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    mv = data["manager_verdict"]
    inv = data["investment_debate_state"]
    ev_ver = data["evidence_verification"]
    ev_summary = inv.get("claim_evidence_summary") or mv.get("claim_evidence_summary")

    # Verify setup: INV-6 is rejected with contradiction
    assert "INV-6" in mv["rejected_claim_ids"]
    assert "INV-6" not in inv["focus_claim_ids"]
    inv6_summary = ev_summary["INV-6"]
    assert inv6_summary["decision"] == "reject"
    assert inv6_summary["counts"]["contradicted"] > 0

    # Confirmation state must be CONFIRMED (not blocked by INV-6)
    confirm_state, reason_codes = evaluate_confirmation_state(
        focus_claim_ids=inv["focus_claim_ids"],
        unresolved_claim_ids=inv["unresolved_claim_ids"],
        claims_verification=ev_ver,
        claim_evidence_summary=ev_summary,
        claims=inv["claims"],
        adopted_claim_ids=mv["adopted_claim_ids"],
        partially_adopted_claims=mv["partially_adopted_claims"],
        rejected_claim_ids=mv["rejected_claim_ids"],
    )
    assert confirm_state == CONFIRM_CONFIRMED
    assert "all_core_claims_verified:INV-1,INV-2" in reason_codes
    assert any("audited_rejected_claims:" in c and "INV-6" in c for c in reason_codes)
    assert not any("fatal_contradicted_claims" in c for c in reason_codes)

    # Manager verdict terminal status must be VALID + BUY + CONFIRMED
    status = status_from_manager_verdict(
        mv,
        investment_debate_state=inv,
        claims_verification=ev_ver,
        claim_evidence_summary=ev_summary,
        focus_claim_ids=inv["focus_claim_ids"],
        unresolved_claim_ids=inv["unresolved_claim_ids"],
        claims=inv["claims"],
        market_data_context=data.get("market_data_context"),
    )
    assert status.analysis_status == ANALYSIS_VALID
    assert status.confirmation_state == CONFIRM_CONFIRMED
    assert status.trade_action == ACTION_BUY
    assert status.direction == DIRECTION_BULL
    assert status.risk_status == "OK"
    assert is_non_executable_status(status) is False

    # apply_decision_status_to_result keeps actionable BUY and targets
    raw_result = {
        "company_of_interest": "000333.SZ",
        "target_price": 88.5,
        "stop_loss_price": 83.8,
        "confidence": 70,
        "probability": 0.75,
    }
    applied = apply_decision_status_to_result(raw_result, status)
    assert applied["decision"] == ACTION_BUY
    assert applied["trade_action"] == ACTION_BUY
    assert applied["confirmation_state"] == CONFIRM_CONFIRMED
    assert applied["target_price"] == 88.5
    assert applied["stop_loss_price"] == 83.8
    assert applied["confidence"] == 70
    assert applied["probability"] == 0.75


def test_focus_or_adopted_claim_contradicted_remains_wait():
    """2. Contradiction on focus claim, adopted claim, or partially adopted claim must force WAIT."""
    # 2a. Focus claim contradicted
    mv_base = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 50,
        "consistency_check_passed": True,
        "adopted_claim_ids": ["CLM-1"],
        "partially_adopted_claims": [],
        "rejected_claim_ids": [],
    }
    ev_summary_focus_contra = {
        "CLM-1": {
            "counts": {"total": 1, "verified": 0, "unsupported": 0, "contradicted": 1, "source_unavailable": 0},
            "coverage": 0.0,
            "decision": "reject",
        }
    }
    c_state, r_codes = evaluate_confirmation_state(
        focus_claim_ids=["CLM-1"],
        claim_evidence_summary=ev_summary_focus_contra,
        adopted_claim_ids=["CLM-1"],
    )
    assert c_state == CONFIRM_UNRESOLVED
    assert "fatal_core_claims:CLM-1" in r_codes
    st = status_from_manager_verdict(
        mv_base,
        focus_claim_ids=["CLM-1"],
        claim_evidence_summary=ev_summary_focus_contra,
    )
    assert st.confirmation_state == CONFIRM_UNRESOLVED
    assert st.trade_action == ACTION_WAIT

    # 2b. Adopted claim contradicted (outside focus)
    mv_adopted_contra = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 50,
        "consistency_check_passed": True,
        "adopted_claim_ids": ["CLM-1", "CLM-2"],
        "partially_adopted_claims": [],
        "rejected_claim_ids": [],
    }
    ev_summary_adopted_contra = {
        "CLM-1": {"counts": {"total": 1, "verified": 1, "unsupported": 0, "contradicted": 0, "source_unavailable": 0}, "coverage": 1.0, "decision": "adopt"},
        "CLM-2": {"counts": {"total": 1, "verified": 0, "unsupported": 0, "contradicted": 1, "source_unavailable": 0}, "coverage": 0.0, "decision": "reject"},
    }
    c_state2, r_codes2 = evaluate_confirmation_state(
        focus_claim_ids=["CLM-1"],
        claim_evidence_summary=ev_summary_adopted_contra,
        adopted_claim_ids=["CLM-1", "CLM-2"],
    )
    assert c_state2 == CONFIRM_UNRESOLVED
    assert "fatal_adopted_claims:CLM-2" in r_codes2
    st2 = status_from_manager_verdict(
        mv_adopted_contra,
        focus_claim_ids=["CLM-1"],
        claim_evidence_summary=ev_summary_adopted_contra,
    )
    assert st2.confirmation_state == CONFIRM_UNRESOLVED
    assert st2.trade_action == ACTION_WAIT


def test_rejected_with_deterministic_adopt_must_abstain_no_trade():
    """3. rejected + deterministic adopt -> verdict consistency failure -> ABSTAIN + NO_TRADE."""
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 60,
        "consistency_check_passed": True,
        "adopted_claim_ids": ["CLM-1"],
        "partially_adopted_claims": [],
        "rejected_claim_ids": ["CLM-2"],
    }
    # CLM-2 is rejected by manager, but evidence aggregator determined decision='adopt' (100% verified)
    claim_evidence_summary = {
        "CLM-1": {
            "counts": {"total": 1, "verified": 1, "unsupported": 0, "contradicted": 0, "source_unavailable": 0},
            "coverage": 1.0,
            "decision": "adopt",
        },
        "CLM-2": {
            "counts": {"total": 2, "verified": 2, "unsupported": 0, "contradicted": 0, "source_unavailable": 0},
            "coverage": 1.0,
            "decision": "adopt",
        },
    }
    c_state, r_codes = evaluate_confirmation_state(
        focus_claim_ids=["CLM-1"],
        claim_evidence_summary=claim_evidence_summary,
        adopted_claim_ids=["CLM-1"],
        rejected_claim_ids=["CLM-2"],
    )
    assert c_state == CONFIRM_UNRESOLVED
    assert "verdict_consistency_rejected_adopt:CLM-2" in r_codes

    status = status_from_manager_verdict(
        mv,
        focus_claim_ids=["CLM-1"],
        claim_evidence_summary=claim_evidence_summary,
    )
    assert status.analysis_status == ANALYSIS_ABSTAIN
    assert status.trade_action == ACTION_NO_TRADE
    assert status.risk_status == "BLOCKED"
    assert status.confirmation_state == CONFIRM_UNRESOLVED
    assert is_non_executable_status(status) is True
    assert is_calibration_eligible(status) is False


def test_rejected_with_partial_evidence_conservatively_waits():
    """4. rejected + partial -> must NOT be silently ignored -> conservative PARTIAL + WAIT."""
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 60,
        "consistency_check_passed": True,
        "adopted_claim_ids": ["CLM-1"],
        "partially_adopted_claims": [],
        "rejected_claim_ids": ["CLM-2"],
    }
    # CLM-2 has partial evidence (e.g. 3 of 4 verified, coverage=75%)
    claim_evidence_summary = {
        "CLM-1": {
            "counts": {"total": 2, "verified": 2, "unsupported": 0, "contradicted": 0, "source_unavailable": 0},
            "coverage": 1.0,
            "decision": "adopt",
        },
        "CLM-2": {
            "counts": {"total": 4, "verified": 3, "unsupported": 1, "contradicted": 0, "source_unavailable": 0},
            "coverage": 0.75,
            "decision": "partial",
        },
    }
    c_state, r_codes = evaluate_confirmation_state(
        focus_claim_ids=["CLM-1"],
        claim_evidence_summary=claim_evidence_summary,
        adopted_claim_ids=["CLM-1"],
        rejected_claim_ids=["CLM-2"],
    )
    assert c_state == CONFIRM_PARTIAL
    assert any("rejected_partial_claims:CLM-2" in c for c in r_codes)

    status = status_from_manager_verdict(
        mv,
        focus_claim_ids=["CLM-1"],
        claim_evidence_summary=claim_evidence_summary,
    )
    assert status.confirmation_state == CONFIRM_PARTIAL
    assert status.trade_action == ACTION_WAIT
    assert status.direction == DIRECTION_BULL
    assert is_non_executable_status(status) is True


def test_bull_bear_symmetry_lifecycle():
    """5. Bull and Bear symmetry: rules must behave identically in both directions."""
    # 5a. Bear winner with rejected bull contradiction does NOT block
    mv_bear = {
        "direction": "看空",
        "winner": "bear",
        "position_pct": 15,
        "consistency_check_passed": True,
        "adopted_claim_ids": ["BEAR-1"],
        "partially_adopted_claims": [],
        "rejected_claim_ids": ["BULL-1"],
    }
    summary_bear = {
        "BEAR-1": {"counts": {"total": 2, "verified": 2, "unsupported": 0, "contradicted": 0, "source_unavailable": 0}, "coverage": 1.0, "decision": "adopt"},
        "BULL-1": {"counts": {"total": 2, "verified": 1, "unsupported": 0, "contradicted": 1, "source_unavailable": 0}, "coverage": 0.5, "decision": "reject"},
    }
    c_state, r_codes = evaluate_confirmation_state(
        focus_claim_ids=["BEAR-1"],
        claim_evidence_summary=summary_bear,
        adopted_claim_ids=["BEAR-1"],
        rejected_claim_ids=["BULL-1"],
    )
    assert c_state == CONFIRM_CONFIRMED
    assert "all_core_claims_verified:BEAR-1" in r_codes
    assert "audited_rejected_claims:BULL-1" in r_codes

    status_bear = status_from_manager_verdict(
        mv_bear,
        focus_claim_ids=["BEAR-1"],
        claim_evidence_summary=summary_bear,
    )
    assert status_bear.confirmation_state == CONFIRM_CONFIRMED
    assert status_bear.trade_action == ACTION_SELL
    assert status_bear.direction == DIRECTION_BEAR
    assert is_non_executable_status(status_bear) is False

    # 5b. Bear winner with contradicted bear focus claim -> WAIT
    summary_bear_contra = {
        "BEAR-1": {"counts": {"total": 1, "verified": 0, "unsupported": 0, "contradicted": 1, "source_unavailable": 0}, "coverage": 0.0, "decision": "reject"},
        "BULL-1": {"counts": {"total": 1, "verified": 0, "unsupported": 1, "contradicted": 0, "source_unavailable": 0}, "coverage": 0.0, "decision": "reject"},
    }
    status_bear_contra = status_from_manager_verdict(
        mv_bear,
        focus_claim_ids=["BEAR-1"],
        claim_evidence_summary=summary_bear_contra,
    )
    assert status_bear_contra.confirmation_state == CONFIRM_UNRESOLVED
    assert status_bear_contra.trade_action == ACTION_WAIT

    # 5c. Bear winner with rejected bull claim having adopt -> ABSTAIN + NO_TRADE
    summary_bear_rej_adopt = {
        "BEAR-1": {"counts": {"total": 1, "verified": 1, "unsupported": 0, "contradicted": 0, "source_unavailable": 0}, "coverage": 1.0, "decision": "adopt"},
        "BULL-1": {"counts": {"total": 1, "verified": 1, "unsupported": 0, "contradicted": 0, "source_unavailable": 0}, "coverage": 1.0, "decision": "adopt"},
    }
    status_bear_rej_adopt = status_from_manager_verdict(
        mv_bear,
        focus_claim_ids=["BEAR-1"],
        claim_evidence_summary=summary_bear_rej_adopt,
    )
    assert status_bear_rej_adopt.analysis_status == ANALYSIS_ABSTAIN
    assert status_bear_rej_adopt.trade_action == ACTION_NO_TRADE


def test_fallback_behavior_when_no_focus_claims():
    """6. Fallback behavior when focus_claim_ids is empty or None."""
    # 6a. Fallback to unresolved_claim_ids
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 50,
        "consistency_check_passed": True,
        "adopted_claim_ids": ["U-1"],
        "rejected_claim_ids": [],
    }
    summary_ok = {
        "U-1": {"counts": {"total": 1, "verified": 1, "unsupported": 0, "contradicted": 0, "source_unavailable": 0}, "coverage": 1.0, "decision": "adopt"},
    }
    c_state_u, r_codes_u = evaluate_confirmation_state(
        focus_claim_ids=[],
        unresolved_claim_ids=["U-1"],
        claim_evidence_summary=summary_ok,
        adopted_claim_ids=["U-1"],
    )
    assert c_state_u == CONFIRM_CONFIRMED
    assert "all_core_claims_verified:U-1" in r_codes_u

    # 6b. Both focus and unresolved empty -> fallback to adopted claims
    c_state_ad, r_codes_ad = evaluate_confirmation_state(
        focus_claim_ids=[],
        unresolved_claim_ids=[],
        claim_evidence_summary=summary_ok,
        adopted_claim_ids=["U-1"],
    )
    assert c_state_ad == CONFIRM_CONFIRMED
    assert "all_core_claims_verified:U-1" in r_codes_ad

    # 6c. All empty, no fatal contradictions -> CONFIRMED
    c_state_empty, _ = evaluate_confirmation_state(
        focus_claim_ids=[],
        unresolved_claim_ids=[],
        adopted_claim_ids=[],
        rejected_claim_ids=[],
    )
    assert c_state_empty == CONFIRM_CONFIRMED


def test_unadjudicated_material_claims_integrity_check():
    """Unadjudicated claims with adopt or partial evidence must trigger integrity / consistency check."""
    # Claim CLM-OMIT was in claims and verified (adopt), but completely omitted from manager verdict
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 50,
        "consistency_check_passed": True,
        "adopted_claim_ids": ["CLM-1"],
        "partially_adopted_claims": [],
        "rejected_claim_ids": [],
    }
    summary = {
        "CLM-1": {"counts": {"total": 1, "verified": 1, "unsupported": 0, "contradicted": 0, "source_unavailable": 0}, "coverage": 1.0, "decision": "adopt"},
        "CLM-OMIT": {"counts": {"total": 1, "verified": 1, "unsupported": 0, "contradicted": 0, "source_unavailable": 0}, "coverage": 1.0, "decision": "adopt"},
    }
    claims = [
        {"claim_id": "CLM-1", "claim": "论点1"},
        {"claim_id": "CLM-OMIT", "claim": "重大遗漏论点"},
    ]
    c_state, r_codes = evaluate_confirmation_state(
        focus_claim_ids=["CLM-1"],
        claim_evidence_summary=summary,
        claims=claims,
        adopted_claim_ids=["CLM-1"],
        rejected_claim_ids=[],
    )
    assert c_state == CONFIRM_UNRESOLVED
    assert "unadjudicated_material_claims_adopt:CLM-OMIT" in r_codes

    status = status_from_manager_verdict(
        mv,
        focus_claim_ids=["CLM-1"],
        claim_evidence_summary=summary,
        claims=claims,
    )
    assert status.analysis_status == ANALYSIS_ABSTAIN
    assert status.trade_action == ACTION_NO_TRADE
    assert status.risk_status == "BLOCKED"


def test_trader_risk_and_db_consistency_contract():
    """8. Trader / Risk / DB decision consistency contract across BUY, WAIT, and NO_TRADE paths."""
    # BUY path: Trader and Risk execute, DB receives BUY
    status_buy = DecisionStatus(
        analysis_status=ANALYSIS_VALID,
        direction=DIRECTION_BULL,
        trade_action=ACTION_BUY,
        risk_status="OK",
        confirmation_state=CONFIRM_CONFIRMED,
        confidence=80,
        probability=0.75,
    )
    result_buy = apply_decision_status_to_result(
        {"target_price": 50.0, "stop_loss_price": 45.0, "confidence": 80, "probability": 0.75},
        status_buy,
    )
    assert result_buy["decision"] == ACTION_BUY
    assert result_buy["trade_action"] == ACTION_BUY
    assert result_buy["confirmation_state"] == CONFIRM_CONFIRMED
    assert result_buy["target_price"] == 50.0
    assert result_buy["stop_loss_price"] == 45.0
    assert result_buy.get("not_applicable") is not True

    # WAIT path: stubbed, targets stripped
    status_wait = DecisionStatus(
        analysis_status=ANALYSIS_VALID,
        direction=DIRECTION_BULL,
        trade_action=ACTION_WAIT,
        risk_status="OK",
        confirmation_state=CONFIRM_UNRESOLVED,
    )
    result_wait = apply_decision_status_to_result(
        {"target_price": 50.0, "stop_loss_price": 45.0, "confidence": 80, "probability": 0.75},
        status_wait,
    )
    assert result_wait["decision"] == ACTION_WAIT
    assert result_wait["trade_action"] == ACTION_WAIT
    assert result_wait["confirmation_state"] == CONFIRM_UNRESOLVED
    assert result_wait["target_price"] is None
    assert result_wait["stop_loss_price"] is None


# ── DAV-854: Red Team Scenarios (RT-1 ~ RT-6) ─────────────────────────────────

def test_dav854_rt1_ordinary_contradicted_restores_wait():
    """RT-1: Core or adopted claim contradicted without PIT failure -> UNRESOLVED + WAIT."""
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 50,
        "consistency_check_passed": True,
        "adopted_claim_ids": ["CLM-CORE"],
        "partially_adopted_claims": [],
        "rejected_claim_ids": [],
    }
    ev_summary = {
        "CLM-CORE": {
            "counts": {"total": 1, "verified": 0, "unsupported": 0, "contradicted": 1, "source_unavailable": 0},
            "coverage": 0.0,
            "decision": "reject",
            "pit_failed": False,
        }
    }
    c_state, r_codes = evaluate_confirmation_state(
        focus_claim_ids=["CLM-CORE"],
        claim_evidence_summary=ev_summary,
        adopted_claim_ids=["CLM-CORE"],
    )
    assert c_state == CONFIRM_UNRESOLVED
    assert "fatal_core_claims:CLM-CORE" in r_codes
    assert "fatal_adopted_claims:CLM-CORE" in r_codes

    status = status_from_manager_verdict(
        mv,
        focus_claim_ids=["CLM-CORE"],
        claim_evidence_summary=ev_summary,
    )
    assert status.confirmation_state == CONFIRM_UNRESOLVED
    assert status.trade_action == ACTION_WAIT
    assert status.analysis_status == ANALYSIS_VALID
    assert any("fatal_core_claims:CLM-CORE" in c for c in status.reason_codes)


def test_dav854_rt2_bull_bear_symmetry():
    """RT-2: Symmetry between bull and bear when contradicted -> both sides are WAIT."""
    # Bull side
    mv_bull = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 40,
        "consistency_check_passed": True,
        "adopted_claim_ids": ["CLM-BULL"],
        "partially_adopted_claims": [],
        "rejected_claim_ids": [],
    }
    summary_bull = {
        "CLM-BULL": {"counts": {"total": 1, "verified": 0, "contradicted": 1}, "decision": "reject", "pit_failed": False}
    }
    status_bull = status_from_manager_verdict(
        mv_bull,
        focus_claim_ids=["CLM-BULL"],
        claim_evidence_summary=summary_bull,
    )
    assert status_bull.confirmation_state == CONFIRM_UNRESOLVED
    assert status_bull.trade_action == ACTION_WAIT

    # Bear side
    mv_bear = {
        "direction": "看空",
        "winner": "bear",
        "position_pct": 40,
        "consistency_check_passed": True,
        "adopted_claim_ids": ["CLM-BEAR"],
        "partially_adopted_claims": [],
        "rejected_claim_ids": [],
    }
    summary_bear = {
        "CLM-BEAR": {"counts": {"total": 1, "verified": 0, "contradicted": 1}, "decision": "reject", "pit_failed": False}
    }
    status_bear = status_from_manager_verdict(
        mv_bear,
        focus_claim_ids=["CLM-BEAR"],
        claim_evidence_summary=summary_bear,
    )
    assert status_bear.confirmation_state == CONFIRM_UNRESOLVED
    assert status_bear.trade_action == ACTION_WAIT


def test_dav854_rt3_pit_failure_must_abstain_no_trade():
    """RT-3: Adopting claim with PIT failure must fail closed with ABSTAIN + NO_TRADE + BLOCKED."""
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 50,
        "consistency_check_passed": True,
        "adopted_claim_ids": ["CLM-PIT"],
        "partially_adopted_claims": [],
        "rejected_claim_ids": [],
    }
    ev_summary = {
        "CLM-PIT": {
            "counts": {"total": 1, "verified": 0, "contradicted": 1},
            "decision": "reject",
            "pit_failed": True,
            "reason": "存在前视偏差/PIT失败 (pit_date=2026-09-20 > baseline=2026-09-08)",
        }
    }
    status = status_from_manager_verdict(
        mv,
        focus_claim_ids=["CLM-PIT"],
        claim_evidence_summary=ev_summary,
    )
    assert status.analysis_status == ANALYSIS_ABSTAIN
    assert status.trade_action == ACTION_NO_TRADE
    assert status.risk_status == "BLOCKED"
    assert any("manager_consistency_hard_gate" in c for c in status.reason_codes)


def test_dav854_rt4_partially_adopted_contradicted_remains_wait():
    """RT-4: Contradicted claim in partially_adopted_claims -> UNRESOLVED + WAIT (same as RT-1)."""
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 50,
        "consistency_check_passed": True,
        "adopted_claim_ids": [],
        "partially_adopted_claims": ["CLM-PARTIAL-CONTRA"],
        "rejected_claim_ids": [],
    }
    ev_summary = {
        "CLM-PARTIAL-CONTRA": {
            "counts": {"total": 1, "verified": 0, "contradicted": 1},
            "decision": "reject",
            "pit_failed": False,
        }
    }
    c_state, r_codes = evaluate_confirmation_state(
        focus_claim_ids=[],
        claim_evidence_summary=ev_summary,
        partially_adopted_claims=["CLM-PARTIAL-CONTRA"],
    )
    assert c_state == CONFIRM_UNRESOLVED
    assert "fatal_partially_adopted_claims:CLM-PARTIAL-CONTRA" in r_codes

    status = status_from_manager_verdict(
        mv,
        claim_evidence_summary=ev_summary,
    )
    assert status.confirmation_state == CONFIRM_UNRESOLVED
    assert status.trade_action == ACTION_WAIT
    assert status.analysis_status == ANALYSIS_VALID


def test_dav854_rt5_observation_hypotheses_stratification():
    """RT-5: Observation/hypotheses unverified with factual verified core does not collapse to WAIT."""
    claims = [
        {"claim_id": "CLM-FACT", "claim": "主力净流入1.2亿", "evidence": ["E1"], "claim_type": "fact"},
        {"claim_id": "CLM-OBS", "claim": "【观察】形态初显", "evidence": ["E2"], "claim_type": "observation"},
    ]
    summary = {
        "CLM-FACT": {"decision": "adopt", "counts": {"total": 1, "verified": 1, "contradicted": 0, "source_unavailable": 0}, "is_observation_or_hypothesis": False},
        "CLM-OBS": {"decision": "partial", "counts": {"total": 1, "verified": 1, "contradicted": 0, "source_unavailable": 0}, "is_observation_or_hypothesis": True},
    }
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 50,
        "adopted_claim_ids": ["CLM-FACT"],
        "partially_adopted_claims": ["CLM-OBS"],
        "consistency_check_passed": True,
    }
    status = status_from_manager_verdict(
        mv,
        claims=claims,
        claim_evidence_summary=summary,
        focus_claim_ids=["CLM-FACT", "CLM-OBS"],
    )
    assert status.analysis_status == ANALYSIS_VALID
    assert status.trade_action == ACTION_BUY
    assert status.confirmation_state == CONFIRM_CONFIRMED


def test_dav854_rt6_non_executable_clears_metrics():
    """RT-6: Non-executable trade actions (WAIT/NO_TRADE) strip confidence, probability, and targets."""
    mv = {
        "direction": "看多",
        "winner": "bull",
        "position_pct": 50,
        "consistency_check_passed": True,
        "adopted_claim_ids": ["CLM-1"],
        "partially_adopted_claims": [],
        "rejected_claim_ids": [],
        "confidence": 85,
        "probability": 0.88,
    }
    ev_summary = {
        "CLM-1": {"counts": {"total": 1, "verified": 0, "contradicted": 1}, "decision": "reject"}
    }
    status = status_from_manager_verdict(
        mv,
        focus_claim_ids=["CLM-1"],
        claim_evidence_summary=ev_summary,
    )
    assert status.trade_action == ACTION_WAIT
    assert status.confidence is None
    assert status.probability is None

    result = apply_decision_status_to_result(
        {"target_price": 50.0, "stop_loss_price": 45.0, "confidence": 85, "probability": 0.88},
        status,
    )
    assert result["target_price"] is None
    assert result["stop_loss_price"] is None

