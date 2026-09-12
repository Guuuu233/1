"""Comprehensive unit and E2E regression tests for DAV-334:
Research Manager Claim Evidence Coverage & Adoption Hard Gate.

Covers:
1. Claim evidence aggregation (counts, coverage ratio, verified/unsupported/contradicted/unavailable).
2. Deterministic decision rules:
   - Full verified (coverage=1.0, 0 contradicted, 0 unavailable) -> adopt
   - Mixed evidence (coverage>=0.67, 0 contradicted, 0 unavailable) -> partial
   - Low coverage (coverage<0.67) or 100% unsupported -> reject
   - Contradicted / Source unavailable -> reject
3. Manager verdict consistency hard gate:
   - Full verified in adopted_claim_ids -> PASS
   - Mixed evidence in partially_adopted_claims + excluded_evidence recorded -> PASS
   - Mixed evidence directly in adopted_claim_ids -> FAIL (fail-closed)
   - Mixed evidence in partially_adopted_claims but prose claiming '证据充分' -> FAIL
   - Unsupported / Contradicted / Unavailable claim in adopted or partial list -> FAIL
4. Research Manager node integration and prompt injection with verification badges.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.utils.evidence_verifier import (
    DECISION_ADOPT,
    DECISION_PARTIAL,
    DECISION_REJECT,
    EvidenceFactualTruthEvaluator,
    STATUS_CONTRADICTED,
    STATUS_SOURCE_UNAVAILABLE,
    STATUS_UNSUPPORTED,
    STATUS_VERIFIED,
    aggregate_claim_evidence,
    extract_and_validate_manager_verdict,
    format_claims_with_verification_for_prompt,
)


async def _fake_stream(text: str):
    from types import SimpleNamespace
    yield SimpleNamespace(content=text)


def _make_e2e_debate_state():
    round_messages = [
        {"message_index": 1, "debate_round": 1, "speaker": "Bull Analyst", "speaker_key": "Bull", "parse_status": "valid", "accepted": True, "responded_claim_ids": [], "target_claim_ids": [], "new_claim_ids": ["INV-1"]},
        {"message_index": 2, "debate_round": 1, "speaker": "Bear Analyst", "speaker_key": "Bear", "parse_status": "valid", "accepted": True, "responded_claim_ids": ["INV-1"], "target_claim_ids": ["INV-1"], "new_claim_ids": ["INV-2"]},
        {"message_index": 3, "debate_round": 2, "speaker": "Bull Analyst", "speaker_key": "Bull", "parse_status": "valid", "accepted": True, "responded_claim_ids": ["INV-2"], "target_claim_ids": ["INV-2"], "new_claim_ids": ["INV-3"]},
        {"message_index": 4, "debate_round": 2, "speaker": "Bear Analyst", "speaker_key": "Bear", "parse_status": "valid", "accepted": True, "responded_claim_ids": ["INV-3"], "target_claim_ids": ["INV-3"], "new_claim_ids": ["INV-4"]},
        {"message_index": 5, "debate_round": 3, "speaker": "Bull Analyst", "speaker_key": "Bull", "parse_status": "valid", "accepted": True, "responded_claim_ids": ["INV-4"], "target_claim_ids": ["INV-4"], "new_claim_ids": ["INV-5"]},
        {"message_index": 6, "debate_round": 3, "speaker": "Bear Analyst", "speaker_key": "Bear", "parse_status": "valid", "accepted": True, "responded_claim_ids": ["INV-5"], "target_claim_ids": ["INV-5"], "new_claim_ids": ["INV-6"]},
    ]
    claims = [
        # INV-1: 100% verified (2 verified, 0 unverified)
        {
            "claim_id": "INV-1",
            "speaker": "Bull Analyst",
            "speaker_key": "Bull",
            "stance": "bullish",
            "claim": "在手订单充足与营收高增",
            "evidence": ["营收同比增长30%", "在手订单增长50%"],
            "confidence": 0.85,
        },
        # INV-2: source unavailable (fatal hallucination)
        {
            "claim_id": "INV-2",
            "speaker": "Bear Analyst",
            "speaker_key": "Bear",
            "stance": "bearish",
            "claim": "海外指数重挫压制估值",
            "evidence": ["global_indices重挫3%"],
            "confidence": 0.70,
        },
        # INV-3: 100% verified
        {
            "claim_id": "INV-3",
            "speaker": "Bull Analyst",
            "speaker_key": "Bull",
            "stance": "bullish",
            "claim": "均线多头排列且央行降息",
            "evidence": ["央行降息25bp", "均线多头排列"],
            "confidence": 0.88,
        },
        # INV-4: contradicted (factual contradiction)
        {
            "claim_id": "INV-4",
            "speaker": "Bear Analyst",
            "speaker_key": "Bear",
            "stance": "bearish",
            "claim": "毛利率严重下滑",
            "evidence": ["毛利率暴跌至10%"],
            "confidence": 0.75,
        },
        # INV-5: mixed evidence (2 verified, 1 unsupported -> coverage = 2/3 = 66.7% >= 67%)
        {
            "claim_id": "INV-5",
            "speaker": "Bull Analyst",
            "speaker_key": "Bull",
            "stance": "bullish",
            "claim": "主力持续增持与市场情绪高涨",
            "evidence": ["主力净流入5.2亿元", "情绪报告看多占比65%", "某机构私下调研看好翻倍"],
            "confidence": 0.90,
        },
        # INV-6: low coverage (1 verified, 2 unsupported -> coverage = 1/3 = 33.3% < 67%)
        {
            "claim_id": "INV-6",
            "speaker": "Bear Analyst",
            "speaker_key": "Bear",
            "stance": "bearish",
            "claim": "新进入者价格战加剧",
            "evidence": ["行业新政落地", "新进入者产能扩张500万吨", "行业整体出现300亿元亏损"],
            "confidence": 0.72,
        },
    ]
    return {
        "macro_report": "宏观报告：央行降息25bp，流动性维持宽松，M2增速10.5%。",
        "market_report": "市场技术报告：突破20.0元关键阻力位，均线多头排列。",
        "sentiment_report": "情绪报告：情绪报告看多占比65%，散户存在分歧。",
        "news_report": "新闻报告：行业新政落地，在手订单增长50%。",
        "fundamentals_report": "基本面报告：营收同比增长30%，毛利率达到28.5%，在手订单充足。",
        "smart_money_report": "主力资金报告：主力净流入5.2亿元，超大单积极建仓吸筹。",
        "volume_price_report": "量价报告：放量长阳突破整理平台，成交量放大1.5倍。",
        "symbol": "000001",
        "market_data_context": {
            "symbol": "000001",
            "analysis_baseline_date": "2026-08-22",
            "trade_date": "2026-08-22",
            "source_provenance": {
                "stock_data": {"status": "available", "as_of": "2026-08-22"},
                "tushare_daily": {"status": "available", "as_of": "2026-08-22"},
                "global_indices": {"status": "failed", "reason": "connection timeout"},
            },
            "data_failure_ledger": [
                {"source": "global_indices", "status": "failed", "reason": "接口超时"},
            ],
            "data_gaps": ["global_indices数据缺失"],
        },
        "investment_debate_state": {
            "history": "辩论历史记录",
            "bull_history": "多头发言",
            "bear_history": "空头发言",
            "current_speaker": "Bear",
            "current_response": "空头最后发言",
            "count": 6,
            "claims": claims,
            "round_messages": round_messages,
            "focus_claim_ids": ["INV-1"],
            "open_claim_ids": ["INV-1", "INV-2", "INV-3", "INV-4", "INV-5", "INV-6"],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": ["INV-1"],
            "round_summary": "多空激辩总结",
            "round_goal": "达成裁决",
            "claim_counter": 6,
        },
        "fund_flow_consensus_guard": {
            "blocked": False,
            "direction_allowed": True,
            "status": "consensus",
        },
        "trade_date": "2026-08-22",
        "horizon": "medium",
    }


class TestClaimEvidenceAggregation:
    """Test claim-level aggregation of evidence verification counts, coverage, and decisions."""

    def test_aggregate_all_verified_claim(self):
        claims = [
            {
                "claim_id": "INV-1",
                "evidence": ["营收同比增长30%", "在手订单增长50%"],
            }
        ]
        verifications = [
            {"claim_id": "INV-1", "raw": "营收同比增长30%", "status": STATUS_VERIFIED, "is_fatal": False},
            {"claim_id": "INV-1", "raw": "在手订单增长50%", "status": STATUS_VERIFIED, "is_fatal": False},
        ]
        summary = aggregate_claim_evidence(claims=claims, claims_verification=verifications)
        assert "INV-1" in summary
        s = summary["INV-1"]
        assert s["counts"]["total"] == 2
        assert s["counts"]["verified"] == 2
        assert s["counts"]["unsupported"] == 0
        assert s["counts"]["contradicted"] == 0
        assert s["counts"]["source_unavailable"] == 0
        assert s["coverage"] == 1.0
        assert s["decision"] == DECISION_ADOPT
        assert len(s["excluded_evidence"]) == 0
        assert len(s["verified_evidence"]) == 2

    def test_aggregate_mixed_evidence_claim_passes_threshold(self):
        claims = [
            {
                "claim_id": "INV-5",
                "evidence": ["主力净流入5.2亿元", "情绪报告看多占比65%", "某机构私下调研看好翻倍"],
            }
        ]
        verifications = [
            {"claim_id": "INV-5", "raw": "主力净流入5.2亿元", "status": STATUS_VERIFIED, "is_fatal": False},
            {"claim_id": "INV-5", "raw": "情绪报告看多占比65%", "status": STATUS_VERIFIED, "is_fatal": False},
            {"claim_id": "INV-5", "raw": "某机构私下调研看好翻倍", "status": STATUS_UNSUPPORTED, "is_fatal": False},
        ]
        summary = aggregate_claim_evidence(claims=claims, claims_verification=verifications)
        assert "INV-5" in summary
        s = summary["INV-5"]
        assert s["counts"]["total"] == 3
        assert s["counts"]["verified"] == 2
        assert s["counts"]["unsupported"] == 1
        assert pytest.approx(s["coverage"], 0.01) == 0.67
        assert s["decision"] == DECISION_PARTIAL
        assert s["excluded_evidence"] == ["某机构私下调研看好翻倍"]
        assert len(s["verified_evidence"]) == 2

    def test_aggregate_low_coverage_mixed_claim_rejected(self):
        claims = [
            {
                "claim_id": "INV-6",
                "evidence": ["行业新政落地", "新进入者降价50%", "产能严重过剩100%"],
            }
        ]
        verifications = [
            {"claim_id": "INV-6", "raw": "行业新政落地", "status": STATUS_VERIFIED, "is_fatal": False},
            {"claim_id": "INV-6", "raw": "新进入者降价50%", "status": STATUS_UNSUPPORTED, "is_fatal": False},
            {"claim_id": "INV-6", "raw": "产能严重过剩100%", "status": STATUS_UNSUPPORTED, "is_fatal": False},
        ]
        summary = aggregate_claim_evidence(claims=claims, claims_verification=verifications)
        assert "INV-6" in summary
        s = summary["INV-6"]
        assert s["counts"]["total"] == 3
        assert s["counts"]["verified"] == 1
        assert s["counts"]["unsupported"] == 2
        assert pytest.approx(s["coverage"], 0.01) == 0.33
        assert s["decision"] == DECISION_REJECT
        assert "覆盖率不足" in s["reason"]

    def test_aggregate_contradicted_claim_rejected(self):
        claims = [
            {
                "claim_id": "INV-4",
                "evidence": ["毛利率达到28.5%", "毛利率暴跌至10%"],
            }
        ]
        verifications = [
            {"claim_id": "INV-4", "raw": "毛利率达到28.5%", "status": STATUS_VERIFIED, "is_fatal": False},
            {"claim_id": "INV-4", "raw": "毛利率暴跌至10%", "status": STATUS_CONTRADICTED, "is_fatal": False},
        ]
        summary = aggregate_claim_evidence(claims=claims, claims_verification=verifications)
        assert "INV-4" in summary
        s = summary["INV-4"]
        assert s["counts"]["contradicted"] == 1
        assert s["decision"] == DECISION_REJECT
        assert "矛盾" in s["reason"] or "冲突" in s["reason"]

    def test_aggregate_unavailable_source_claim_rejected(self):
        claims = [
            {
                "claim_id": "INV-2",
                "evidence": ["global_indices重挫3%"],
            }
        ]
        verifications = [
            {"claim_id": "INV-2", "raw": "global_indices重挫3%", "status": STATUS_SOURCE_UNAVAILABLE, "is_fatal": True},
        ]
        summary = aggregate_claim_evidence(claims=claims, claims_verification=verifications)
        assert "INV-2" in summary
        s = summary["INV-2"]
        assert s["counts"]["source_unavailable"] == 1
        assert s["decision"] == DECISION_REJECT
        assert "严重幻觉" in s["reason"] or "不可用" in s["reason"]

    def test_aggregate_fully_unsupported_claim_rejected(self):
        claims = [
            {
                "claim_id": "INV-7",
                "evidence": ["传闻某客户退单"],
            }
        ]
        verifications = [
            {"claim_id": "INV-7", "raw": "传闻某客户退单", "status": STATUS_UNSUPPORTED, "is_fatal": False},
        ]
        summary = aggregate_claim_evidence(claims=claims, claims_verification=verifications)
        assert "INV-7" in summary
        s = summary["INV-7"]
        assert s["counts"]["verified"] == 0
        assert s["coverage"] == 0.0
        assert s["decision"] == DECISION_REJECT

    def test_aggregate_receives_and_evaluates_baseline_date_symbol_and_market_context(self):
        """Aggregation evaluates applicability, PIT lookahead, and invalidation conditions against market context."""
        claims = [
            # 1. Valid applicability and conditions
            {
                "claim_id": "C-VALID",
                "evidence": ["净利润增长20%"],
                "applicability": {
                    "symbol": "000001",
                    "horizon": "short",
                    "metric_basis": "vendor_qfq",
                    "pit_date": "2026-08-20",
                },
                "invalidation_conditions": [
                    {
                        "condition_id": "c1",
                        "metric": "pe",
                        "operator": ">",
                        "threshold": 40.0,
                        "unit": "ratio",
                        "period": "1d_close",
                        "source": "daily_price",
                        "pit_date": "2026-08-20",
                    }
                ],
            },
            # 2. PIT lookahead failure in applicability
            {
                "claim_id": "C-PIT-APP",
                "evidence": ["营收翻倍"],
                "applicability": {
                    "symbol": "000001",
                    "horizon": "short",
                    "metric_basis": "vendor_qfq",
                    "pit_date": "2026-08-25",  # after baseline 2026-08-22
                },
            },
            # 3. Applicability symbol mismatch
            {
                "claim_id": "C-SYM-MISMATCH",
                "evidence": ["产量大增"],
                "applicability": {
                    "symbol": "600519",  # mismatch with 000001
                    "horizon": "short",
                    "metric_basis": "vendor_qfq",
                    "pit_date": "2026-08-20",
                },
            },
            # 4. Triggered invalidation condition (pe=35 > 30)
            {
                "claim_id": "C-TRIG",
                "evidence": ["估值合理"],
                "invalidation_conditions": [
                    {
                        "condition_id": "c2",
                        "metric": "pe",
                        "operator": ">",
                        "threshold": 30.0,
                        "unit": "ratio",
                        "period": "1d_close",
                        "source": "daily_price",
                        "pit_date": "2026-08-20",
                    }
                ],
            },
            # 5. Observation / hypothesis claim
            {
                "claim_id": "C-OBS",
                "claim": "【观察】主力试盘吸筹中",
                "claim_type": "observation",
                "evidence": ["放量探底回升"],
            },
        ]
        verifications = [
            {"claim_id": "C-VALID", "raw": "净利润增长20%", "status": STATUS_VERIFIED},
            {"claim_id": "C-PIT-APP", "raw": "营收翻倍", "status": STATUS_VERIFIED},
            {"claim_id": "C-SYM-MISMATCH", "raw": "产量大增", "status": STATUS_VERIFIED},
            {"claim_id": "C-TRIG", "raw": "估值合理", "status": STATUS_VERIFIED},
            {"claim_id": "C-OBS", "raw": "放量探底回升", "status": STATUS_VERIFIED},
        ]
        market_ctx = {
            "symbol": "000001",
            "pe": 35.0,
            "trade_date": "2026-08-22",
            "analysis_baseline_date": "2026-08-22",
        }
        summary = aggregate_claim_evidence(
            claims=claims,
            claims_verification=verifications,
            analysis_baseline_date="2026-08-22",
            expected_symbol="000001",
            market_data_context=market_ctx,
        )

        # C-VALID: passes all, adopted
        assert summary["C-VALID"]["decision"] == DECISION_ADOPT
        assert summary["C-VALID"]["pit_failed"] is False
        assert summary["C-VALID"]["applicability"]["symbol"] == "000001"

        # C-PIT-APP: lookahead PIT fails closed
        assert summary["C-PIT-APP"]["decision"] == DECISION_REJECT
        assert summary["C-PIT-APP"]["pit_failed"] is True
        assert summary["C-PIT-APP"]["counts"]["contradicted"] >= 1

        # C-SYM-MISMATCH: symbol mismatch fails applicability
        assert summary["C-SYM-MISMATCH"]["decision"] == DECISION_REJECT
        assert "适用性规格校验失败" in summary["C-SYM-MISMATCH"]["reason"]

        # C-TRIG: condition triggered (pe=35 > 30) -> contradicted / reject
        assert summary["C-TRIG"]["decision"] == DECISION_REJECT
        assert "失效条件已触发证伪" in summary["C-TRIG"]["reason"]
        assert summary["C-TRIG"]["counts"]["contradicted"] >= 1

        # C-OBS: observation stays in partial state, cannot be adopt
        assert summary["C-OBS"]["is_observation_or_hypothesis"] is True
        assert summary["C-OBS"]["decision"] == DECISION_PARTIAL


class TestManagerVerdictConsistencyHardGateCoverage:
    """Test consistency hard gate on claim adoption rules."""

    def test_full_verified_adoption_and_partial_mixed_adoption_passes(self):
        state = _make_e2e_debate_state()
        evaluator = EvidenceFactualTruthEvaluator()
        claims_verification = evaluator.evaluate_claims(
            claims=state["investment_debate_state"]["claims"],
            seven_reports={
                "macro_report": state["macro_report"],
                "market_report": state["market_report"],
                "sentiment_report": state["sentiment_report"],
                "news_report": state["news_report"],
                "fundamentals_report": state["fundamentals_report"],
                "smart_money_report": state["smart_money_report"],
                "volume_price_report": state["volume_price_report"],
            },
            market_data_context=state["market_data_context"],
            analysis_baseline_date="2026-08-22",
        )

        raw_output = """【投研经理裁决报告】
第一步审查：INV-1全部核验通过，标证据充分；INV-5为混合证据，仅采纳主力净流入与看多情绪等verified子结论，标部分支持，未验证传闻予以剔除；INV-2/4/6予以驳回。
多头胜出。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "多头证据扎实", "position_pct": 60, "entry": "20.0", "target": "25.0", "stop_loss": "19.0", "upside": 25.0, "downside": 5.0, "odds": 5.0, "adopted_claim_ids": ["INV-1", "INV-3"], "partially_adopted_claims": ["INV-5"], "rejected_claim_ids": ["INV-2", "INV-4", "INV-6"], "excluded_evidence": ["某机构私下调研看好翻倍"]} -->
<!-- VERDICT: {"direction": "看多", "reason": "多头营收与资金扎实"} -->"""

        verdict = extract_and_validate_manager_verdict(
            raw_response=raw_output,
            claims_verification=claims_verification,
            claims=state["investment_debate_state"]["claims"],
        )
        assert verdict["consistency_check_passed"] is True
        assert verdict["adopted_claim_ids"] == ["INV-1", "INV-3"]
        assert verdict["partially_adopted_claims"] == ["INV-5"]
        assert "某机构私下调研看好翻倍" in verdict["excluded_evidence"]
        assert len(verdict["failed_checks"]) == 0

    def test_mixed_evidence_in_adopted_claim_ids_fails_gate(self):
        state = _make_e2e_debate_state()
        evaluator = EvidenceFactualTruthEvaluator()
        claims_verification = evaluator.evaluate_claims(
            claims=state["investment_debate_state"]["claims"],
            seven_reports={
                "macro_report": state["macro_report"],
                "market_report": state["market_report"],
                "sentiment_report": state["sentiment_report"],
                "news_report": state["news_report"],
                "fundamentals_report": state["fundamentals_report"],
                "smart_money_report": state["smart_money_report"],
                "volume_price_report": state["volume_price_report"],
            },
            market_data_context=state["market_data_context"],
            analysis_baseline_date="2026-08-22",
        )

        # Violation: INV-5 is mixed evidence, but LLM put it directly in adopted_claim_ids
        raw_output = """【投研经理裁决报告】
多头胜出。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "采纳全部多头论点", "position_pct": 60, "entry": "20.0", "target": "25.0", "stop_loss": "19.0", "adopted_claim_ids": ["INV-1", "INV-5"], "partially_adopted_claims": [], "rejected_claim_ids": ["INV-2", "INV-4", "INV-6"]} -->"""

        verdict = extract_and_validate_manager_verdict(
            raw_response=raw_output,
            claims_verification=claims_verification,
            claims=state["investment_debate_state"]["claims"],
        )
        assert verdict["consistency_check_passed"] is False
        assert any("全额采纳了含未核实混合证据的 claim: INV-5" in err for err in verdict["failed_checks"])

    def test_prose_marking_mixed_claim_as_sufficient_evidence_fails_gate(self):
        state = _make_e2e_debate_state()
        evaluator = EvidenceFactualTruthEvaluator()
        claims_verification = evaluator.evaluate_claims(
            claims=state["investment_debate_state"]["claims"],
            seven_reports={
                "macro_report": state["macro_report"],
                "market_report": state["market_report"],
                "sentiment_report": state["sentiment_report"],
                "news_report": state["news_report"],
                "fundamentals_report": state["fundamentals_report"],
                "smart_money_report": state["smart_money_report"],
                "volume_price_report": state["volume_price_report"],
            },
            market_data_context=state["market_data_context"],
            analysis_baseline_date="2026-08-22",
        )

        # Violation: Prose states INV-5 is '证据充分' even though machine block recorded partial
        raw_output = """【投研经理裁决报告】
经过逐条核验，INV-5证据充分，多头全面胜出。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "多头胜", "position_pct": 60, "entry": "20.0", "target": "25.0", "stop_loss": "19.0", "adopted_claim_ids": ["INV-1"], "partially_adopted_claims": ["INV-5"], "rejected_claim_ids": ["INV-2", "INV-4", "INV-6"]} -->"""

        verdict = extract_and_validate_manager_verdict(
            raw_response=raw_output,
            claims_verification=claims_verification,
            claims=state["investment_debate_state"]["claims"],
        )
        assert verdict["consistency_check_passed"] is False
        assert any("正文将未完全核实的 claim INV-5" in err and "证据充分" in err for err in verdict["failed_checks"])

    def test_adopting_contradicted_or_unavailable_claim_fails_gate(self):
        state = _make_e2e_debate_state()
        evaluator = EvidenceFactualTruthEvaluator()
        claims_verification = evaluator.evaluate_claims(
            claims=state["investment_debate_state"]["claims"],
            seven_reports={
                "macro_report": state["macro_report"],
                "market_report": state["market_report"],
                "sentiment_report": state["sentiment_report"],
                "news_report": state["news_report"],
                "fundamentals_report": state["fundamentals_report"],
                "smart_money_report": state["smart_money_report"],
                "volume_price_report": state["volume_price_report"],
            },
            market_data_context=state["market_data_context"],
            analysis_baseline_date="2026-08-22",
        )

        # Adopting contradicted INV-4
        raw_output = """【投研经理裁决报告】
空头胜出。
<!-- MANAGER_VERDICT: {"winner": "bear", "direction": "看空", "reason": "采纳空头毛利下滑", "position_pct": 0, "adopted_claim_ids": ["INV-4"], "rejected_claim_ids": []} -->"""

        verdict = extract_and_validate_manager_verdict(
            raw_response=raw_output,
            claims_verification=claims_verification,
            claims=state["investment_debate_state"]["claims"],
        )
        assert verdict["consistency_check_passed"] is False
        assert any("存在事实冲突" in err or "矛盾" in err for err in verdict["failed_checks"])

    def test_adopting_low_coverage_claim_fails_gate(self):
        state = _make_e2e_debate_state()
        evaluator = EvidenceFactualTruthEvaluator()
        claims_verification = evaluator.evaluate_claims(
            claims=state["investment_debate_state"]["claims"],
            seven_reports={
                "macro_report": state["macro_report"],
                "market_report": state["market_report"],
                "sentiment_report": state["sentiment_report"],
                "news_report": state["news_report"],
                "fundamentals_report": state["fundamentals_report"],
                "smart_money_report": state["smart_money_report"],
                "volume_price_report": state["volume_price_report"],
            },
            market_data_context=state["market_data_context"],
            analysis_baseline_date="2026-08-22",
        )

        # Adopting INV-6 (coverage 33.3%)
        raw_output = """【投研经理裁决报告】
空头胜出。
<!-- MANAGER_VERDICT: {"winner": "bear", "direction": "看空", "reason": "采纳空头竞争论点", "position_pct": 0, "adopted_claim_ids": ["INV-6"], "rejected_claim_ids": []} -->"""

        verdict = extract_and_validate_manager_verdict(
            raw_response=raw_output,
            claims_verification=claims_verification,
            claims=state["investment_debate_state"]["claims"],
        )
        assert verdict["consistency_check_passed"] is False
        assert any("证据覆盖率不足" in err for err in verdict["failed_checks"])

    def test_manager_verdict_cannot_whitewash_pit_failure(self):
        """Manager verdict attempting to adopt a claim with lookahead PIT failure is rejected (fail-closed)."""
        claim = {
            "claim_id": "CLM-PIT",
            "claim": "未来营收高增",
            "evidence": ["营收增长50%"],
            "applicability": {
                "symbol": "000001",
                "horizon": "short",
                "metric_basis": "vendor_qfq",
                "pit_date": "2026-09-01",  # Future vs 2026-08-22
            },
        }
        verifications = [
            {"claim_id": "CLM-PIT", "raw": "营收增长50%", "status": STATUS_VERIFIED}
        ]
        market_ctx = {"symbol": "000001", "trade_date": "2026-08-22", "analysis_baseline_date": "2026-08-22"}
        raw_output = """【投研经理裁决报告】
看多。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "采纳未来高增", "position_pct": 50, "stop_loss": "19.0", "adopted_claim_ids": ["CLM-PIT"], "partially_adopted_claims": [], "rejected_claim_ids": []} -->"""
        verdict = extract_and_validate_manager_verdict(
            raw_response=raw_output,
            claims_verification=verifications,
            claims=[claim],
            market_data_context=market_ctx,
        )
        assert verdict["consistency_check_passed"] is False
        assert any("存在事实冲突/前视偏差" in err or "矛盾" in err for err in verdict["failed_checks"])

    def test_manager_verdict_cannot_whitewash_triggered_invalidation_condition(self):
        """Manager verdict attempting to adopt a claim whose invalidation condition triggered is rejected."""
        claim = {
            "claim_id": "CLM-TRIG",
            "claim": "估值安全",
            "evidence": ["估值处于合理分位"],
            "invalidation_conditions": [
                {
                    "condition_id": "cond_pe",
                    "metric": "pe",
                    "operator": ">",
                    "threshold": 30.0,
                    "unit": "ratio",
                    "period": "1d_close",
                    "source": "daily_price",
                    "pit_date": "2026-08-22",
                }
            ],
        }
        verifications = [
            {"claim_id": "CLM-TRIG", "raw": "估值处于合理分位", "status": STATUS_VERIFIED}
        ]
        market_ctx = {"symbol": "000001", "pe": 40.0, "trade_date": "2026-08-22", "analysis_baseline_date": "2026-08-22"}
        raw_output = """【投研经理裁决报告】
看多。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "采纳估值安全", "position_pct": 50, "stop_loss": "19.0", "adopted_claim_ids": ["CLM-TRIG"], "partially_adopted_claims": [], "rejected_claim_ids": []} -->"""
        verdict = extract_and_validate_manager_verdict(
            raw_response=raw_output,
            claims_verification=verifications,
            claims=[claim],
            market_data_context=market_ctx,
        )
        assert verdict["consistency_check_passed"] is False
        assert any("存在事实冲突/前视偏差" in err or "矛盾" in err for err in verdict["failed_checks"])

    def test_manager_verdict_observation_hypothesis_cannot_be_adopted(self):
        """Adopting observation/hypothesis claim fails consistency check; partial adoption passes."""
        obs_claim = {
            "claim_id": "CLM-OBS",
            "claim": "【假设】若主力拉升则突破前高",
            "claim_type": "hypothesis",
            "evidence": ["突破阻力位"],
        }
        verifications = [
            {"claim_id": "CLM-OBS", "raw": "突破阻力位", "status": STATUS_VERIFIED}
        ]
        # 1. Putting in adopted_claim_ids -> FAILS
        raw_output_adopt = """【投研经理裁决报告】
看多。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "采纳假设", "position_pct": 50, "stop_loss": "19.0", "adopted_claim_ids": ["CLM-OBS"], "partially_adopted_claims": [], "rejected_claim_ids": []} -->"""
        verdict_bad = extract_and_validate_manager_verdict(
            raw_response=raw_output_adopt,
            claims_verification=verifications,
            claims=[obs_claim],
        )
        assert verdict_bad["consistency_check_passed"] is False
        assert any("观察/假设类" in err for err in verdict_bad["failed_checks"])

        # 2. Putting in partially_adopted_claims -> PASSES
        raw_output_partial = """【投研经理裁决报告】
看多。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "部分采纳观察", "position_pct": 50, "stop_loss": "19.0", "adopted_claim_ids": [], "partially_adopted_claims": ["CLM-OBS"], "rejected_claim_ids": []} -->"""
        verdict_ok = extract_and_validate_manager_verdict(
            raw_response=raw_output_partial,
            claims_verification=verifications,
            claims=[obs_claim],
        )
        assert verdict_ok["consistency_check_passed"] is True


class TestPromptFormatAndVerificationPresentation:
    """Test format_claims_with_verification_for_prompt produces accurate badges and text."""

    def test_format_claims_with_verification_output(self):
        claims = [
            {"claim_id": "INV-1", "speaker": "Bull", "stance": "bullish", "claim": "营收高增", "evidence": ["营收30%"]},
            {"claim_id": "INV-5", "speaker": "Bull", "stance": "bullish", "claim": "主力增持", "evidence": ["流入5亿", "产能95%", "传闻"]},
        ]
        verifications = [
            {"claim_id": "INV-1", "raw": "营收30%", "status": STATUS_VERIFIED, "matched_role": "fundamentals_report"},
            {"claim_id": "INV-5", "raw": "流入5亿", "status": STATUS_VERIFIED, "matched_role": "smart_money_report"},
            {"claim_id": "INV-5", "raw": "产能95%", "status": STATUS_VERIFIED, "matched_role": "fundamentals_report"},
            {"claim_id": "INV-5", "raw": "传闻", "status": STATUS_UNSUPPORTED, "details": "未找到支撑"},
        ]
        text = format_claims_with_verification_for_prompt(claims=claims, claims_verification=verifications)
        assert "INV-1" in text
        assert "【证据充分 / 全Verified】" in text
        assert "覆盖率=100.0%" in text
        assert "[VERIFIED / 真实核验] 营收30%" in text

        assert "INV-5" in text
        assert "【部分支持 / 混合证据(仅采纳Verified子结论)】" in text
        assert "[UNSUPPORTED / 未获支撑] 传闻" in text
        assert "严禁作为采纳依据，必须剔除" in text


class TestResearchManagerIntegrationWithEvidenceGate:
    """Integration test for create_research_manager with claim evidence gate."""

    def test_research_manager_runs_and_passes_valid_adjudication(self):
        state = _make_e2e_debate_state()

        llm_response = """【研究总监裁决报告】
各分析师观点穿透：
1. 证据审查：INV-1与INV-3全Verified，标证据充分；INV-5为混合证据，仅采纳主力增持与情绪结论，标部分支持；INV-2/4/6驳回。
2. 传导路径：降息与订单饱满支持多头。
3. 裁决结论：多头全面胜出。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "多头证据扎实", "position_pct": 60, "entry": "20.0", "target": "25.0", "stop_loss": "19.0", "upside": 25.0, "downside": 5.0, "odds": 5.0, "adopted_claim_ids": ["INV-1", "INV-3"], "partially_adopted_claims": ["INV-5"], "rejected_claim_ids": ["INV-2", "INV-4", "INV-6"], "excluded_evidence": ["某机构私下调研看好翻倍"]} -->
<!-- VERDICT: {"direction": "看多", "reason": "多头证据扎实"} -->"""

        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(side_effect=lambda prompt: _fake_stream(llm_response))
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])

        rm_node = create_research_manager(mock_llm, memory)
        result = asyncio.run(rm_node(state))

        assert result["manager_verdict"]["consistency_check_passed"] is True
        assert result["manager_verdict"]["winner"] == "bull"
        assert result["manager_verdict"]["adopted_claim_ids"] == ["INV-1", "INV-3"]
        assert result["manager_verdict"]["partially_adopted_claims"] == ["INV-5"]
        assert "某机构私下调研看好翻倍" in result["manager_verdict"]["excluded_evidence"]
        assert "INV-1" in result["manager_verdict"]["claim_evidence_summary"]
        assert result["manager_verdict"]["claim_evidence_summary"]["INV-1"]["decision"] == DECISION_ADOPT
        assert result["manager_verdict"]["claim_evidence_summary"]["INV-5"]["decision"] == DECISION_PARTIAL
        assert result["manager_verdict"]["claim_evidence_summary"]["INV-4"]["decision"] == DECISION_REJECT

        # E-03d: structural round-trip consistency across all 3 locations
        assert result["claim_evidence_summary"] == result["manager_verdict"]["claim_evidence_summary"]
        assert result["investment_debate_state"]["claim_evidence_summary"] == result["claim_evidence_summary"]
        assert result["decision_status"] == result["manager_verdict"]["decision_status"]
        assert result["evidence_verification"] == result["investment_debate_state"]["evidence_verification"]
        assert result["decision_status"]["analysis_status"] == "VALID"
        assert result["decision_status"]["direction"] == "BULL"
        assert result["decision_status"]["trade_action"] == "WAIT"
        assert result["decision_status"]["confirmation_state"] == "PARTIAL"

    def test_research_manager_pre_gate_blocked_path_structural_roundtrip(self):
        """Debate pre-gate check failure blocks before LLM, and preserves structured summary across all locations."""
        state = _make_e2e_debate_state()
        # Invalidate preconditions: empty round_messages violates message count precondition
        state["investment_debate_state"]["round_messages"] = []

        mock_llm = MagicMock()
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])

        rm_node = create_research_manager(mock_llm, memory)
        result = asyncio.run(rm_node(state))

        # LLM must never be called on pre-gate failure
        mock_llm.astream.assert_not_called()

        assert result["manager_verdict"]["consistency_check_passed"] is False
        assert result["decision_status"]["analysis_status"] == "ABSTAIN"
        assert result["decision_status"]["trade_action"] == "NO_TRADE"
        assert result["decision_status"]["risk_status"] == "BLOCKED"
        assert result["decision_status"] == result["manager_verdict"]["decision_status"]

        # Structured summary in verdict, state, and payload top-level are consistent
        assert result["claim_evidence_summary"] == result["manager_verdict"]["claim_evidence_summary"]
        assert result["investment_debate_state"]["claim_evidence_summary"] == result["claim_evidence_summary"]
        assert result["evidence_verification"] == result["investment_debate_state"]["evidence_verification"]
        assert "INV-1" in result["claim_evidence_summary"]
        assert result["claim_evidence_summary"]["INV-1"]["decision"] == DECISION_ADOPT

    def test_research_manager_pit_lookahead_fail_closed_cannot_be_whitewashed(self):
        """Manager text declaring adopt on lookahead PIT failure claim still fails closed to ABSTAIN/NO_TRADE."""
        state = _make_e2e_debate_state()
        # Modify INV-3 to have future PIT date lookahead
        claims = state["investment_debate_state"]["claims"]
        claims[2]["applicability"] = {
            "symbol": "000001",
            "horizon": "short",
            "metric_basis": "vendor_qfq",
            "pit_date": "2026-09-01",  # Future vs baseline 2026-08-22
        }

        # Manager tries to adopt INV-1 and INV-3
        llm_response = """【研究总监裁决报告】
裁决结论：多头胜出。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "采纳多头论点", "position_pct": 60, "entry": "20.0", "target": "25.0", "stop_loss": "19.0", "adopted_claim_ids": ["INV-1", "INV-3"], "partially_adopted_claims": [], "rejected_claim_ids": ["INV-2", "INV-4", "INV-5", "INV-6"]} -->"""

        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(side_effect=lambda prompt: _fake_stream(llm_response))
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])

        rm_node = create_research_manager(mock_llm, memory)
        result = asyncio.run(rm_node(state))

        assert result["manager_verdict"]["consistency_check_passed"] is False
        assert result["decision_status"]["analysis_status"] == "ABSTAIN"
        assert result["decision_status"]["trade_action"] == "NO_TRADE"
        assert result["decision_status"]["risk_status"] == "BLOCKED"
        assert result["claim_evidence_summary"]["INV-3"]["pit_failed"] is True
        assert result["claim_evidence_summary"]["INV-3"]["decision"] == DECISION_REJECT

        # Check consistency across all locations
        assert result["claim_evidence_summary"] == result["manager_verdict"]["claim_evidence_summary"]
        assert result["investment_debate_state"]["claim_evidence_summary"] == result["claim_evidence_summary"]

    def test_research_manager_invalidation_condition_triggered_fails_closed(self):
        """Manager adopting claim with triggered invalidation condition fails closed."""
        state = _make_e2e_debate_state()
        claims = state["investment_debate_state"]["claims"]
        claims[2]["invalidation_conditions"] = [
            {
                "condition_id": "break_20",
                "metric": "close_price",
                "operator": "<",
                "threshold": 20.0,
                "unit": "cny",
                "period": "1d_close",
                "source": "daily_price",
                "pit_date": "2026-08-22",
            }
        ]
        # Trigger condition: close_price = 18.0 < 20.0
        state["market_data_context"]["close_price"] = 18.0

        llm_response = """【研究总监裁决报告】
裁决结论：多头胜出。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "采纳均线", "position_pct": 60, "entry": "20.0", "target": "25.0", "stop_loss": "19.0", "adopted_claim_ids": ["INV-1", "INV-3"], "partially_adopted_claims": [], "rejected_claim_ids": ["INV-2", "INV-4", "INV-5", "INV-6"]} -->"""

        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(side_effect=lambda prompt: _fake_stream(llm_response))
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])

        rm_node = create_research_manager(mock_llm, memory)
        result = asyncio.run(rm_node(state))

        assert result["manager_verdict"]["consistency_check_passed"] is False
        assert result["decision_status"]["analysis_status"] == "ABSTAIN"
        assert result["decision_status"]["trade_action"] == "NO_TRADE"
        assert result["claim_evidence_summary"]["INV-3"]["decision"] == DECISION_REJECT
        assert "失效条件已触发证伪" in result["claim_evidence_summary"]["INV-3"]["reason"]
        assert result["claim_evidence_summary"] == result["manager_verdict"]["claim_evidence_summary"]

    def test_research_manager_observation_hypothesis_with_factual_core_retains_valid_direction(self):
        """Observation claim alongside verified factual core allows valid direction, not unconditional WAIT."""
        state = _make_e2e_debate_state()
        claims = state["investment_debate_state"]["claims"]
        # Make INV-5 an observation claim with 100% verified evidence (no unverified hearsay)
        claims[4]["claim_type"] = "observation"
        claims[4]["is_observation"] = True
        claims[4]["claim"] = "【观察】主力持续增持与市场情绪高涨"
        claims[4]["evidence"] = ["主力净流入5.2亿元", "情绪报告看多占比65%"]

        # Manager puts factual core INV-1 & INV-3 in adopted_claim_ids, and observation INV-5 in partially_adopted_claims
        llm_response = """【研究总监裁决报告】
裁决结论：多头胜出。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "多头事实充分且观察支持", "position_pct": 60, "entry": "20.0", "target": "25.0", "stop_loss": "19.0", "adopted_claim_ids": ["INV-1", "INV-3"], "partially_adopted_claims": ["INV-5"], "rejected_claim_ids": ["INV-2", "INV-4", "INV-6"]} -->"""

        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(side_effect=lambda prompt: _fake_stream(llm_response))
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])

        rm_node = create_research_manager(mock_llm, memory)
        result = asyncio.run(rm_node(state))

        assert result["manager_verdict"]["consistency_check_passed"] is True
        assert result["decision_status"]["analysis_status"] == "VALID"
        assert result["decision_status"]["trade_action"] == "BUY"
        assert result["decision_status"]["confirmation_state"] == "CONFIRMED"
        assert any("audited_observation_claims:INV-5" in code for code in result["decision_status"]["reason_codes"])
        assert result["claim_evidence_summary"]["INV-5"]["is_observation_or_hypothesis"] is True
        assert result["claim_evidence_summary"]["INV-5"]["decision"] == DECISION_PARTIAL
        assert result["claim_evidence_summary"] == result["manager_verdict"]["claim_evidence_summary"]

    def test_research_manager_observation_hypothesis_upgrade_to_adopt_fails_gate(self):
        """Observation claim upgraded to full adopt in adopted_claim_ids fails consistency gate."""
        state = _make_e2e_debate_state()
        claims = state["investment_debate_state"]["claims"]
        claims[4]["claim_type"] = "hypothesis"
        claims[4]["is_hypothesis"] = True
        claims[4]["claim"] = "【假设】主力持续增持与市场情绪高涨"

        # Violation: puts hypothesis INV-5 in adopted_claim_ids
        llm_response = """【研究总监裁决报告】
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "全额采纳假设", "position_pct": 60, "entry": "20.0", "target": "25.0", "stop_loss": "19.0", "adopted_claim_ids": ["INV-1", "INV-5"], "partially_adopted_claims": [], "rejected_claim_ids": ["INV-2", "INV-4", "INV-6"]} -->"""

        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(side_effect=lambda prompt: _fake_stream(llm_response))
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])

        rm_node = create_research_manager(mock_llm, memory)
        result = asyncio.run(rm_node(state))

        assert result["manager_verdict"]["consistency_check_passed"] is False
        assert any("观察/假设类" in err for err in result["manager_verdict"]["failed_checks"])
        assert result["decision_status"]["analysis_status"] == "ABSTAIN"
        assert result["decision_status"]["trade_action"] == "NO_TRADE"

    def test_research_manager_aggregation_receives_symbol_and_baseline_date(self):
        """Research manager passes resolved expected symbol and baseline date to claim evidence aggregation."""
        state = _make_e2e_debate_state()
        state["symbol"] = "000001"
        state["trade_date"] = "2026-08-22"
        state["market_data_context"]["symbol"] = "000001"
        state["market_data_context"]["analysis_baseline_date"] = "2026-08-22"

        # Claim specifying matching symbol and valid PIT date
        claims = state["investment_debate_state"]["claims"]
        claims[0]["applicability"] = {
            "symbol": "000001",
            "horizon": "short",
            "metric_basis": "vendor_qfq",
            "pit_date": "2026-08-20",
        }

        llm_response = """【研究总监裁决报告】
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "标的核验通过", "position_pct": 60, "entry": "20.0", "target": "25.0", "stop_loss": "19.0", "adopted_claim_ids": ["INV-1", "INV-3"], "partially_adopted_claims": ["INV-5"], "rejected_claim_ids": ["INV-2", "INV-4", "INV-6"], "excluded_evidence": ["某机构私下调研看好翻倍"]} -->"""

        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(side_effect=lambda prompt: _fake_stream(llm_response))
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])

        rm_node = create_research_manager(mock_llm, memory)
        result = asyncio.run(rm_node(state))

        assert result["manager_verdict"]["consistency_check_passed"] is True
        assert result["claim_evidence_summary"]["INV-1"]["decision"] == DECISION_ADOPT
        assert result["claim_evidence_summary"]["INV-1"]["applicability"]["symbol"] == "000001"
        assert result["claim_evidence_summary"]["INV-1"]["pit_failed"] is False

    def test_research_manager_blocks_when_llm_violates_claim_coverage_gate(self):
        state = _make_e2e_debate_state()

        # Violation: Adopts unverified/low-coverage INV-6 and mixed INV-5 in adopted_claim_ids
        bad_response = """【研究总监裁决报告】
裁决结论发生冲突。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "违规采纳", "position_pct": 60, "entry": "20.0", "target": "25.0", "stop_loss": "19.0", "adopted_claim_ids": ["INV-5", "INV-6"], "partially_adopted_claims": []} -->"""

        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(side_effect=lambda prompt: _fake_stream(bad_response))
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])

        rm_node = create_research_manager(mock_llm, memory)
        result = asyncio.run(rm_node(state))

        assert result["manager_verdict"]["consistency_check_passed"] is False
        assert "研究总监裁决自洽硬闸未通过" in result["investment_plan"]
        assert "已阻断进入 Trader 执行阶段" in result["investment_plan"]
