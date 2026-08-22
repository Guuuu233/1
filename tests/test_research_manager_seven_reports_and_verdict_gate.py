"""Comprehensive unit and integration tests for DAV-323:
Research Manager Seven Reports Input, Provenance & Gaps Context,
EvidenceFactualTruthEvaluator, Structured Manager Verdict, and Consistency Hard Gate.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.utils.evidence_summary import (
    build_dense_report_input,
    build_evidence_summary,
)
from tradingagents.agents.utils.evidence_verifier import (
    EvidenceFactualTruthEvaluator,
    STATUS_CONTRADICTED,
    STATUS_SOURCE_UNAVAILABLE,
    STATUS_UNSUPPORTED,
    STATUS_VERIFIED,
    extract_and_validate_manager_verdict,
    normalize_numeric_value,
    normalize_winner,
)
from tradingagents.agents.utils.debate_utils import build_debate_report_manifest
from tradingagents.graph.trading_graph import TradingAgentsGraph


async def _fake_stream(text: str):
    from types import SimpleNamespace
    yield SimpleNamespace(content=text)


def _make_seven_reports_state(overrides=None):
    state = {
        "macro_report": "宏观报告：央行降息25bp，流动性维持宽松，M2增速10.5%。",
        "market_report": "市场技术报告：突破20.0元关键阻力位，均线多头排列。",
        "sentiment_report": "情绪报告：市场情绪看多占比65%，散户存在分歧。",
        "news_report": "新闻报告：行业新政落地，新产品在手订单增长50%。",
        "fundamentals_report": "基本面报告：营收同比增长30%，毛利率达到28.5%，在手订单充足。",
        "smart_money_report": "主力资金报告：主力净流入5.2亿元，超大单积极建仓吸筹。",
        "volume_price_report": "量价报告：放量长阳突破整理平台，成交量放大1.5倍。",
        "market_data_context": {
            "analysis_baseline_date": "2026-08-22",
            "trade_date": "2026-08-22",
            "source_provenance": {
                "tushare_daily": {"status": "available", "as_of": "2026-08-22"},
                "global_indices": {"status": "failed", "reason": "connection timeout"},
            },
            "data_failure_ledger": [
                {"source": "global_indices", "status": "failed", "reason": "接口超时"},
                {"source": "northbound_flow", "status": "unavailable", "reason": "港交所休市未提供"},
            ],
            "data_gaps": ["global_indices数据缺失", "northbound_flow不可用"],
        },
        "investment_debate_state": {
            "history": "辩论历史",
            "bull_history": "多头历史",
            "bear_history": "空头历史",
            "current_speaker": "Bear",
            "current_response": "空头发言",
            "count": 2,
            "claims": [
                {
                    "claim_id": "INV-1",
                    "speaker": "Bull Analyst",
                    "speaker_key": "Bull",
                    "stance": "bullish",
                    "claim": "在手订单与营收高增",
                    "evidence": ["营收同比增长30%", "在手订单增长50%"],
                    "confidence": 0.85,
                },
                {
                    "claim_id": "INV-2",
                    "speaker": "Bear Analyst",
                    "speaker_key": "Bear",
                    "stance": "bearish",
                    "claim": "海外指数大跌压制估值",
                    "evidence": ["global_indices重挫3%"],
                    "confidence": 0.70,
                },
            ],
            "round_messages": [],
            "focus_claim_ids": ["INV-1"],
            "open_claim_ids": ["INV-1", "INV-2"],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": ["INV-1"],
            "round_summary": "多空激辩",
            "round_goal": "收敛分歧",
            "claim_counter": 2,
        },
        "fund_flow_consensus_guard": {
            "blocked": False,
            "direction_allowed": True,
            "status": "consensus",
        },
        "trade_date": "2026-08-22",
        "horizon": "medium",
    }
    if overrides:
        state.update(overrides)
    return state


class TestEvidenceFactualTruthEvaluator:
    """Test Contract 3 & 4: Deterministic fact matching, unit normalization, and fatal hallucination."""

    def test_unit_normalization(self):
        # 1 亿元 -> 100000000 元
        norm_yi = normalize_numeric_value("1", "亿元")
        assert norm_yi == (100_000_000.0, "元")

        # 10000 万元 -> 100000000 元
        norm_wan = normalize_numeric_value("10000", "万元")
        assert norm_wan == (100_000_000.0, "元")

        # 25bp -> 0.25%
        norm_bp = normalize_numeric_value("25", "bp")
        assert norm_bp == (0.25, "%")

        # 10.5% -> 10.5%
        norm_pct = normalize_numeric_value("10.5", "%")
        assert norm_pct == (10.5, "%")

    def test_verified_facts_in_seven_reports(self):
        state = _make_seven_reports_state()
        evaluator = EvidenceFactualTruthEvaluator()

        seven_reports = {
            "macro_report": state["macro_report"],
            "market_report": state["market_report"],
            "sentiment_report": state["sentiment_report"],
            "news_report": state["news_report"],
            "fundamentals_report": state["fundamentals_report"],
            "smart_money_report": state["smart_money_report"],
            "volume_price_report": state["volume_price_report"],
        }

        # Exact / normalized match in fundamentals
        res1 = evaluator.evaluate_single_evidence(
            raw_evidence="营收同比增长30%",
            seven_reports=seven_reports,
            market_data_context=state["market_data_context"],
            analysis_baseline_date="2026-08-22",
        )
        assert res1["status"] == STATUS_VERIFIED
        assert res1["matched_role"] == "fundamentals_report"
        assert res1["is_fatal"] is False

        # Match in smart money report (5.2亿元 = 52000万元)
        res2 = evaluator.evaluate_single_evidence(
            raw_evidence="主力净流入52000万元",
            seven_reports=seven_reports,
            market_data_context=state["market_data_context"],
            analysis_baseline_date="2026-08-22",
        )
        assert res2["status"] == STATUS_VERIFIED
        assert res2["matched_role"] == "smart_money_report"

        # Match in macro report (25bp)
        res3 = evaluator.evaluate_single_evidence(
            raw_evidence="央行降息25bp",
            seven_reports=seven_reports,
            market_data_context=state["market_data_context"],
            analysis_baseline_date="2026-08-22",
        )
        assert res3["status"] == STATUS_VERIFIED
        assert res3["matched_role"] == "macro_report"

    def test_unsupported_fact(self):
        state = _make_seven_reports_state()
        evaluator = EvidenceFactualTruthEvaluator()
        seven_reports = {
            "macro_report": state["macro_report"],
            "fundamentals_report": state["fundamentals_report"],
        }

        res = evaluator.evaluate_single_evidence(
            raw_evidence="公司获得中东主权基金50亿美元注资",
            seven_reports=seven_reports,
            market_data_context=state["market_data_context"],
            analysis_baseline_date="2026-08-22",
        )
        assert res["status"] == STATUS_UNSUPPORTED
        assert res["matched_role"] is None
        assert res["is_fatal"] is False

    def test_contradicted_number_detected(self):
        state = _make_seven_reports_state()
        evaluator = EvidenceFactualTruthEvaluator()
        seven_reports = {
            "fundamentals_report": "基本面报告：营收同比增长30%，净利润下滑10%。",
        }

        # Evidence claims 80% revenue growth when report records 30%
        res = evaluator.evaluate_single_evidence(
            raw_evidence="营收同比增长80%",
            seven_reports=seven_reports,
            market_data_context=state["market_data_context"],
            analysis_baseline_date="2026-08-22",
        )
        assert res["status"] == STATUS_CONTRADICTED
        assert res["matched_role"] == "fundamentals_report"
        assert "数据冲突" in res["details"]

    def test_unavailable_data_source_is_fatal_hallucination(self):
        state = _make_seven_reports_state()
        evaluator = EvidenceFactualTruthEvaluator()
        seven_reports = {"market_report": "常规技术面"}

        # global_indices is recorded as failed in data_failure_ledger
        res = evaluator.evaluate_single_evidence(
            raw_evidence="global_indices重挫3%引发系统性风险",
            seven_reports=seven_reports,
            market_data_context=state["market_data_context"],
            analysis_baseline_date="2026-08-22",
        )
        assert res["status"] == STATUS_SOURCE_UNAVAILABLE
        assert res["is_fatal"] is True
        assert "严重幻觉" in res["details"] or "不可用" in res["details"]

    def test_anti_lookahead_date_check(self):
        state = _make_seven_reports_state()
        evaluator = EvidenceFactualTruthEvaluator()
        seven_reports = {"news_report": "新闻事件"}

        # Future date stated as historical fact (baseline is 2026-08-22)
        res = evaluator.evaluate_single_evidence(
            raw_evidence="2026-08-28公司已完成股份回购注销",
            seven_reports=seven_reports,
            market_data_context=state["market_data_context"],
            analysis_baseline_date="2026-08-22",
        )
        assert res["status"] == STATUS_CONTRADICTED
        assert "前视偏差" in res["details"]


class TestSevenReportsInputAndManifest:
    """Test Contract 1 & 2: Seven reports manifest and dense input formatting."""

    def test_manifest_records_length_mode_and_char_counts(self):
        state = _make_seven_reports_state({
            "macro_report": "短宏观报告" * 10,
            "market_report": "长市场报告：" + "均线向上量能温和放大。" * 200,
        })
        manifest = build_debate_report_manifest(
            state,
            pass_info={
                "macro_report": ("full", len(state["macro_report"])),
                "market_report": ("structured_dense_summary_and_excerpts", 1500),
            },
        )

        assert manifest["macro_report"]["passed"] is True
        assert manifest["macro_report"]["length"] == len(state["macro_report"])
        assert manifest["macro_report"]["mode"] == "full"
        assert manifest["macro_report"]["passed_chars"] == len(state["macro_report"])

        assert manifest["market_report"]["passed"] is True
        assert manifest["market_report"]["mode"] == "structured_dense_summary_and_excerpts"
        assert manifest["market_report"]["passed_chars"] == 1500

    def test_build_dense_report_input_behavior(self):
        short_text = "短报告：营收+20%，净利润+15%。"
        inp_short, mode_short, chars_short = build_dense_report_input(short_text, max_chars=1000)
        assert mode_short == "full"
        assert chars_short == len(short_text)

        long_text = "分析师结论：看多\n" + "营收增长30%订单充足。\n" * 100
        inp_long, mode_long, chars_long = build_dense_report_input(long_text, max_chars=500)
        assert mode_long == "structured_dense_summary_and_excerpts"
        assert chars_long <= 500


class TestManagerVerdictConsistencyHardGate:
    """Test Contract 6 & 7: Structured Manager Verdict & Consistency Hard Gate."""

    def test_valid_bull_verdict_passes(self):
        raw_output = """投研经理裁决报告：
多头在手订单与长协锁价证据扎实，营收确定性高。判定多头胜出。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "多头订单证据充分", "position_pct": 60, "entry": "20.5", "target": "25.0", "stop_loss": "19.0", "upside": 22.0, "downside": 7.3, "odds": 3.01, "adopted_claim_ids": ["INV-1"], "rejected_claim_ids": []} -->"""

        verdict = extract_and_validate_manager_verdict(raw_output)
        assert verdict["consistency_check_passed"] is True
        assert verdict["winner"] == "bull"
        assert verdict["direction"] == "看多"
        assert len(verdict["failed_checks"]) == 0

    def test_valid_bear_verdict_passes(self):
        raw_output = """投研经理裁决报告：
空头指出的原材料暴涨与毛利承压逻辑成立。判定空头胜出。
<!-- MANAGER_VERDICT: {"winner": "bear", "direction": "看空", "reason": "毛利承压明显", "position_pct": 0, "entry": null, "target": null, "stop_loss": null, "upside": 0, "downside": 15.0, "odds": 0, "adopted_claim_ids": [], "rejected_claim_ids": ["INV-1"]} -->"""

        verdict = extract_and_validate_manager_verdict(raw_output)
        assert verdict["consistency_check_passed"] is True
        assert verdict["winner"] == "bear"
        assert verdict["direction"] == "看空"
        assert len(verdict["failed_checks"]) == 0

    def test_bear_winner_with_buy_or_high_position_fails(self):
        # Negative case: Bear winner but direction is BUY and position 70%
        raw_output = """裁决：空头获胜。
<!-- MANAGER_VERDICT: {"winner": "bear", "direction": "看多", "reason": "矛盾结论", "position_pct": 70, "entry": "20.0", "target": "25.0", "stop_loss": "19.0", "adopted_claim_ids": []} -->"""

        verdict = extract_and_validate_manager_verdict(raw_output)
        assert verdict["consistency_check_passed"] is False
        assert any("不得为看多" in err for err in verdict["failed_checks"])
        assert any("不得高于20%" in err for err in verdict["failed_checks"])

    def test_bull_winner_without_stop_loss_or_invalid_stop_fails(self):
        # Negative case: Bull winner without stop loss
        raw_output1 = """裁决：多头获胜。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "多头胜", "position_pct": 60, "entry": "20.0", "stop_loss": null} -->"""

        v1 = extract_and_validate_manager_verdict(raw_output1)
        assert v1["consistency_check_passed"] is False
        assert any("必须设定明确有效的止损位" in err for err in v1["failed_checks"])

        # Negative case: stop_loss >= entry
        raw_output2 = """裁决：多头获胜。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "多头胜", "position_pct": 60, "entry": "20.0", "stop_loss": "21.0"} -->"""
        v2 = extract_and_validate_manager_verdict(raw_output2)
        assert v2["consistency_check_passed"] is False
        assert any("止损位" in err and "必须严格低于入场价" in err for err in v2["failed_checks"])

    def test_prose_vs_verdict_contradiction_fails(self):
        # Negative case: Prose explicitly says 空头胜 but machine block says winner: bull
        raw_output = """经过五步深度裁决，空头胜，空头全面占优。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "多头胜", "position_pct": 60, "stop_loss": "19.0", "entry": "20.0"} -->"""

        verdict = extract_and_validate_manager_verdict(raw_output)
        assert verdict["consistency_check_passed"] is False
        assert any("正文与机读裁决严重矛盾" in err for err in verdict["failed_checks"])

    def test_adopted_fatal_hallucination_claim_fails(self):
        # Negative case: Adopts claim with fatal hallucination
        raw_output = """裁决：采纳关键空头论点。
<!-- MANAGER_VERDICT: {"winner": "bear", "direction": "看空", "reason": "采纳INV-2", "position_pct": 0, "adopted_claim_ids": ["INV-2"]} -->"""

        claims_verification = [
            {"claim_id": "INV-2", "is_fatal": True, "status": STATUS_SOURCE_UNAVAILABLE},
        ]
        verdict = extract_and_validate_manager_verdict(raw_output, claims_verification=claims_verification)
        assert verdict["consistency_check_passed"] is False
        assert any("不可用数据源的严重幻觉" in err for err in verdict["failed_checks"])


class TestResearchManagerIntegrationNode:
    """Integration test for create_research_manager execution node and state persistence."""

    def test_research_manager_executes_and_persists_verdict_and_manifest(self):
        state = _make_seven_reports_state()

        llm_response = """【研究总监裁决报告】
各分析师观点综合穿透：基本面与主力资金表现强劲，多头逻辑成立。
多头胜出。
<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "看多", "reason": "多头营收与资金扎实", "position_pct": 60, "entry": "20.0", "target": "25.0", "stop_loss": "19.0", "upside": 25.0, "downside": 5.0, "odds": 5.0, "adopted_claim_ids": ["INV-1"], "rejected_claim_ids": []} -->
<!-- VERDICT: {"direction": "看多", "reason": "多头营收与资金扎实"} -->"""

        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(side_effect=lambda prompt: _fake_stream(llm_response))
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])

        rm_node = create_research_manager(mock_llm, memory)
        result = asyncio.run(rm_node(state))

        assert "manager_verdict" in result
        assert result["manager_verdict"]["consistency_check_passed"] is True
        assert result["manager_verdict"]["winner"] == "bull"

        assert "evidence_verification" in result
        assert len(result["evidence_verification"]) >= 1

        assert "report_manifest" in result
        assert len(result["report_manifest"]) == 7

        new_inv_state = result["investment_debate_state"]
        assert "manager_verdict" in new_inv_state
        assert "evidence_verification" in new_inv_state
        assert "report_manifest" in new_inv_state

    def test_research_manager_blocks_on_consistency_failure(self):
        state = _make_seven_reports_state()

        # Inconsistent output: Bear winner with BUY direction
        bad_response = """【研究总监裁决报告】
裁决结论发生冲突。
<!-- MANAGER_VERDICT: {"winner": "bear", "direction": "看多", "reason": "冲突", "position_pct": 80} -->"""

        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(side_effect=lambda prompt: _fake_stream(bad_response))
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])

        rm_node = create_research_manager(mock_llm, memory)
        result = asyncio.run(rm_node(state))

        assert result["manager_verdict"]["consistency_check_passed"] is False
        assert "研究总监裁决自洽硬闸未通过" in result["investment_plan"]
        assert "已阻断进入 Trader 执行阶段" in result["investment_plan"]
