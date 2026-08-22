"""Unit and integration tests for DAV-320:
Seven reports input completeness, round_messages trajectory persistence,
and strict debate protocol hard-gate verification.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.utils.debate_utils import (
    build_debate_report_manifest,
    update_debate_state_with_payload,
)
from tradingagents.prompts.zh import PROMPTS as ZH_PROMPTS
from tradingagents.prompts.en import PROMPTS as EN_PROMPTS


async def _fake_stream(text: str):
    from types import SimpleNamespace
    yield SimpleNamespace(content=text)


def _make_initial_state(overrides=None):
    state = {
        "macro_report": "宏观报告内容：流动性宽松",
        "market_report": "市场技术报告内容：均线多头",
        "sentiment_report": "情绪报告内容：温和看多",
        "news_report": "新闻报告内容：政策利好频出",
        "fundamentals_report": "基本面报告内容：营收+30%",
        "smart_money_report": "主力资金报告内容：主力净流入",
        "volume_price_report": "量价报告内容：放量突破平台",
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_speaker": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
            "claims": [],
            "round_messages": [],
            "focus_claim_ids": [],
            "open_claim_ids": [],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "round_summary": "",
            "round_goal": "建立最核心的正反两方 claim",
            "claim_counter": 0,
        },
        "horizon": "medium",
        "user_intent": None,
    }
    if overrides:
        state.update(overrides)
    return state


class TestSevenReportsInputAndManifest:
    """Test Contract 1: Complete seven reports input and manifest generation."""

    def test_manifest_records_all_seven_reports_length_and_passed(self):
        state = _make_initial_state({
            "macro_report": "MACRO_CONTENT_123",
            "market_report": "MARKET_CONTENT_456",
            "sentiment_report": "SENTIMENT_CONTENT_789",
            "news_report": "NEWS_CONTENT_ABC",
            "fundamentals_report": "FUNDAMENTALS_CONTENT_DEF",
            "smart_money_report": "SMART_MONEY_CONTENT_GHI",
            "volume_price_report": "VOLUME_PRICE_CONTENT_JKL",
        })
        manifest = build_debate_report_manifest(state)

        expected_reports = [
            "macro_report",
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
            "smart_money_report",
            "volume_price_report",
        ]
        for rep in expected_reports:
            assert rep in manifest, f"Report {rep} missing in manifest"
            assert manifest[rep]["passed"] is True
            assert manifest[rep]["length"] > 0

        assert manifest["macro_report"]["length"] == len("MACRO_CONTENT_123")
        assert manifest["smart_money_report"]["length"] == len("SMART_MONEY_CONTENT_GHI")

    def test_bull_and_bear_prompts_receive_all_seven_reports_unique_tags(self):
        state = _make_initial_state({
            "macro_report": "[UNIQUE_MACRO_REPORT_TAG_777]",
            "market_report": "[UNIQUE_MARKET_REPORT_TAG_888]",
            "sentiment_report": "[UNIQUE_SENTIMENT_REPORT_TAG_999]",
            "news_report": "[UNIQUE_NEWS_REPORT_TAG_AAA]",
            "fundamentals_report": "[UNIQUE_FUNDAMENTALS_REPORT_TAG_BBB]",
            "smart_money_report": "[UNIQUE_SMART_MONEY_REPORT_TAG_CCC]",
            "volume_price_report": "[UNIQUE_VOLUME_PRICE_REPORT_TAG_DDD]",
        })

        bull_captured_prompts = []
        mock_llm_bull = MagicMock()

        def fake_bull_stream(prompt):
            bull_captured_prompts.append(prompt)
            return _fake_stream(
                '多头分析\n<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": [{"claim": "多头主张", "evidence": ["证据"], "confidence": 0.8, "target_claim_ids": []}], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], "round_summary": "多头立论", "round_goal": "立论"} -->'
            )

        mock_llm_bull.astream = MagicMock(side_effect=fake_bull_stream)
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])

        bull_node = create_bull_researcher(mock_llm_bull, memory)
        asyncio.run(bull_node(state))

        assert len(bull_captured_prompts) == 1
        bull_prompt_text = bull_captured_prompts[0]

        assert "[UNIQUE_MACRO_REPORT_TAG_777]" in bull_prompt_text
        assert "[UNIQUE_MARKET_REPORT_TAG_888]" in bull_prompt_text
        assert "[UNIQUE_SENTIMENT_REPORT_TAG_999]" in bull_prompt_text
        assert "[UNIQUE_NEWS_REPORT_TAG_AAA]" in bull_prompt_text
        assert "[UNIQUE_FUNDAMENTALS_REPORT_TAG_BBB]" in bull_prompt_text
        assert "[UNIQUE_SMART_MONEY_REPORT_TAG_CCC]" in bull_prompt_text
        assert "[UNIQUE_VOLUME_PRICE_REPORT_TAG_DDD]" in bull_prompt_text

        bear_captured_prompts = []
        mock_llm_bear = MagicMock()

        def fake_bear_stream(prompt):
            bear_captured_prompts.append(prompt)
            return _fake_stream(
                '空头分析\n<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1"], "new_claims": [{"claim": "空头主张", "evidence": ["证据"], "confidence": 0.8, "target_claim_ids": ["INV-1"]}], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], "round_summary": "空头立论", "round_goal": "立论"} -->'
            )

        mock_llm_bear.astream = MagicMock(side_effect=fake_bear_stream)
        bear_node = create_bear_researcher(mock_llm_bear, memory)
        asyncio.run(bear_node(state))

        assert len(bear_captured_prompts) == 1
        bear_prompt_text = bear_captured_prompts[0]

        assert "[UNIQUE_MACRO_REPORT_TAG_777]" in bear_prompt_text
        assert "[UNIQUE_MARKET_REPORT_TAG_888]" in bear_prompt_text
        assert "[UNIQUE_SENTIMENT_REPORT_TAG_999]" in bear_prompt_text
        assert "[UNIQUE_NEWS_REPORT_TAG_AAA]" in bear_prompt_text
        assert "[UNIQUE_FUNDAMENTALS_REPORT_TAG_BBB]" in bear_prompt_text
        assert "[UNIQUE_SMART_MONEY_REPORT_TAG_CCC]" in bear_prompt_text
        assert "[UNIQUE_VOLUME_PRICE_REPORT_TAG_DDD]" in bear_prompt_text


class TestSixMessageDebateFixture:
    """Test Contract 2, 3, 4, 5: 6-message fixture and strict protocol validation."""

    def test_full_six_message_valid_trajectory(self):
        """Simulate a valid 6-message debate (3 full rounds of Bull/Bear) and verify round_messages."""
        state = _make_initial_state()["investment_debate_state"]

        # Msg 1: Bull Round 1
        payload1 = {
            "responded_claim_ids": [],
            "new_claims": [
                {
                    "claim": "多头Round1: 宏观宽松驱动估值扩张",
                    "evidence": ["央行降息25bp", "M2增速10.5%"],
                    "confidence": 0.85,
                    "target_claim_ids": [],
                }
            ],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "next_focus_claim_ids": ["INV-1"],
            "round_summary": "多头首轮立论，建立宏观与估值 claim",
            "round_goal": "建立最核心的正反两方 claim",
        }
        msg1_raw = f"多头第一轮发言正文：宏观流动性极佳。\n<!-- DEBATE_STATE: {json.dumps(payload1, ensure_ascii=False)} -->"
        state = update_debate_state_with_payload(
            state=state,
            raw_response=msg1_raw,
            speaker_label="Bull Analyst",
            speaker_key="Bull",
            stance="bullish",
            history_key="bull_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        assert state["count"] == 1
        assert len(state["claims"]) == 1
        assert state["claims"][0]["claim_id"] == "INV-1"
        assert len(state["round_messages"]) == 1
        assert state["round_messages"][0]["parse_status"] == "valid"
        assert state["round_messages"][0]["message_index"] == 1
        assert state["round_messages"][0]["debate_round"] == 1
        assert state["round_messages"][0]["new_claim_ids"] == ["INV-1"]

        # Msg 2: Bear Round 1
        payload2 = {
            "responded_claim_ids": ["INV-1"],
            "new_claims": [
                {
                    "claim": "空头Round1: 估值已透支未来三年业绩",
                    "evidence": ["动态PE处于95%分位", "毛利率环比下滑2%"],
                    "confidence": 0.82,
                    "target_claim_ids": ["INV-1"],
                }
            ],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": ["INV-1"],
            "next_focus_claim_ids": ["INV-1", "INV-2"],
            "round_summary": "空头首轮攻击估值透支与毛利承压",
            "round_goal": "优先攻击对手最脆弱的假设",
        }
        msg2_raw = f"空头第一轮发言正文：反驳多头，估值过高。\n<!-- DEBATE_STATE: {json.dumps(payload2, ensure_ascii=False)} -->"
        state = update_debate_state_with_payload(
            state=state,
            raw_response=msg2_raw,
            speaker_label="Bear Analyst",
            speaker_key="Bear",
            stance="bearish",
            history_key="bear_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        assert state["count"] == 2
        assert len(state["claims"]) == 2
        assert state["claims"][1]["claim_id"] == "INV-2"
        assert state["claims"][0]["status"] == "unresolved"
        assert len(state["round_messages"]) == 2
        assert state["round_messages"][1]["parse_status"] == "valid"
        assert state["round_messages"][1]["message_index"] == 2
        assert state["round_messages"][1]["debate_round"] == 1
        assert state["round_messages"][1]["responded_claim_ids"] == ["INV-1"]
        assert state["round_messages"][1]["target_claim_ids"] == ["INV-1"]

        # Msg 3: Bull Round 2
        payload3 = {
            "responded_claim_ids": ["INV-2"],
            "new_claims": [
                {
                    "claim": "多头Round2: 新产品放量将大幅摊薄估值",
                    "evidence": ["在手订单+50%", "产能利用率95%"],
                    "confidence": 0.88,
                    "target_claim_ids": ["INV-2"],
                }
            ],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": ["INV-2"],
            "next_focus_claim_ids": ["INV-2", "INV-3"],
            "round_summary": "多头二轮反驳毛利下滑假设，指出新产品放量",
            "round_goal": "围绕时间窗口与触发条件",
        }
        msg3_raw = f"多头第二轮发言正文：在手订单充足，打穿空头逻辑。\n<!-- DEBATE_STATE: {json.dumps(payload3, ensure_ascii=False)} -->"
        state = update_debate_state_with_payload(
            state=state,
            raw_response=msg3_raw,
            speaker_label="Bull Analyst",
            speaker_key="Bull",
            stance="bullish",
            history_key="bull_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        assert state["count"] == 3
        assert len(state["claims"]) == 3
        assert state["claims"][2]["claim_id"] == "INV-3"
        assert len(state["round_messages"]) == 3
        assert state["round_messages"][2]["parse_status"] == "valid"
        assert state["round_messages"][2]["message_index"] == 3
        assert state["round_messages"][2]["debate_round"] == 2

        # Msg 4: Bear Round 2
        payload4 = {
            "responded_claim_ids": ["INV-3"],
            "new_claims": [
                {
                    "claim": "空头Round2: 上游原材料涨价将吞噬新产品利润",
                    "evidence": ["碳酸锂价格上涨15%", "下游议价权弱"],
                    "confidence": 0.79,
                    "target_claim_ids": ["INV-3"],
                }
            ],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": ["INV-3"],
            "next_focus_claim_ids": ["INV-3", "INV-4"],
            "round_summary": "空头二轮攻击上游成本传导风险",
            "round_goal": "围绕失败路径与失效条件",
        }
        msg4_raw = f"空头第二轮发言正文：上游涨价将导致增收不增利。\n<!-- DEBATE_STATE: {json.dumps(payload4, ensure_ascii=False)} -->"
        state = update_debate_state_with_payload(
            state=state,
            raw_response=msg4_raw,
            speaker_label="Bear Analyst",
            speaker_key="Bear",
            stance="bearish",
            history_key="bear_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        assert state["count"] == 4
        assert len(state["claims"]) == 4
        assert state["claims"][3]["claim_id"] == "INV-4"
        assert len(state["round_messages"]) == 4
        assert state["round_messages"][3]["parse_status"] == "valid"
        assert state["round_messages"][3]["message_index"] == 4
        assert state["round_messages"][3]["debate_round"] == 2

        # Msg 5: Bull Round 3
        payload5 = {
            "responded_claim_ids": ["INV-4"],
            "new_claims": [
                {
                    "claim": "多头Round3: 长期协议锁定原料成本且具备转嫁能力",
                    "evidence": ["80%长协价锁定", "转嫁弹性0.75"],
                    "confidence": 0.91,
                    "target_claim_ids": ["INV-4"],
                }
            ],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": ["INV-4"],
            "next_focus_claim_ids": ["INV-4", "INV-5"],
            "round_summary": "多头收官论证长协锁定成本与定价权",
            "round_goal": "检查剩余分歧收口",
        }
        msg5_raw = f"多头第三轮发言正文：长协锁价确保成本可控。\n<!-- DEBATE_STATE: {json.dumps(payload5, ensure_ascii=False)} -->"
        state = update_debate_state_with_payload(
            state=state,
            raw_response=msg5_raw,
            speaker_label="Bull Analyst",
            speaker_key="Bull",
            stance="bullish",
            history_key="bull_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        assert state["count"] == 5
        assert len(state["claims"]) == 5
        assert state["claims"][4]["claim_id"] == "INV-5"
        assert len(state["round_messages"]) == 5
        assert state["round_messages"][4]["parse_status"] == "valid"
        assert state["round_messages"][4]["message_index"] == 5
        assert state["round_messages"][4]["debate_round"] == 3

        # Msg 6: Bear Round 3
        payload6 = {
            "responded_claim_ids": ["INV-5"],
            "new_claims": [
                {
                    "claim": "空头Round3: 行业竞争加剧导致长协执行存在违约风险",
                    "evidence": ["行业CR3下滑5%", "价格战苗头"],
                    "confidence": 0.81,
                    "target_claim_ids": ["INV-5"],
                }
            ],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": ["INV-5"],
            "next_focus_claim_ids": ["INV-5", "INV-6"],
            "round_summary": "空头收官指出行业竞争格局恶化",
            "round_goal": "检查剩余分歧收口",
        }
        msg6_raw = f"空头第三轮发言正文：竞争恶化仍是最大隐患。\n<!-- DEBATE_STATE: {json.dumps(payload6, ensure_ascii=False)} -->"
        state = update_debate_state_with_payload(
            state=state,
            raw_response=msg6_raw,
            speaker_label="Bear Analyst",
            speaker_key="Bear",
            stance="bearish",
            history_key="bear_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        assert state["count"] == 6
        assert len(state["claims"]) == 6
        assert state["claims"][5]["claim_id"] == "INV-6"
        assert len(state["round_messages"]) == 6
        assert state["round_messages"][5]["parse_status"] == "valid"
        assert state["round_messages"][5]["message_index"] == 6
        assert state["round_messages"][5]["debate_round"] == 3

        # Verify trajectory completeness
        for idx, r_msg in enumerate(state["round_messages"], start=1):
            assert r_msg["message_index"] == idx
            assert r_msg["debate_round"] == (idx - 1) // 2 + 1
            assert r_msg["parse_status"] == "valid"
            assert "cleaned_prose" in r_msg and r_msg["cleaned_prose"]
            if idx >= 2:
                assert len(r_msg["responded_claim_ids"]) >= 1
                assert len(r_msg["target_claim_ids"]) >= 1


class TestProtocolGateNegativeCases:
    """Test Contract 3, 4, 5, 6: Negative cases must be rejected with invalid_protocol/invalid/missing."""

    def test_subsequent_round_empty_responded_is_rejected(self):
        """Bear in Round 1 (message_index=2) with empty responded_claim_ids fails protocol."""
        state = _make_initial_state()["investment_debate_state"]

        # Msg 1: Bull Round 1
        payload1 = {
            "responded_claim_ids": [],
            "new_claims": [{"claim": "多头主张", "evidence": ["e1"], "confidence": 0.8, "target_claim_ids": []}],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "next_focus_claim_ids": ["INV-1"],
            "round_summary": "多头立论",
            "round_goal": "立论",
        }
        msg1_raw = f"多头发言。\n<!-- DEBATE_STATE: {json.dumps(payload1, ensure_ascii=False)} -->"
        state = update_debate_state_with_payload(
            state=state,
            raw_response=msg1_raw,
            speaker_label="Bull Analyst",
            speaker_key="Bull",
            stance="bullish",
            history_key="bull_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        assert state["count"] == 1
        initial_claims = list(state["claims"])

        # Msg 2: Bear with empty responded
        payload2 = {
            "responded_claim_ids": [],  # Violates Rule: must respond to opponent claim
            "new_claims": [{"claim": "空头独立主张", "evidence": ["e2"], "confidence": 0.75, "target_claim_ids": ["INV-1"]}],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "next_focus_claim_ids": [],
            "round_summary": "空头无回应发言",
            "round_goal": "攻击",
        }
        msg2_raw = f"空头发言，但未回应多头。\n<!-- DEBATE_STATE: {json.dumps(payload2, ensure_ascii=False)} -->"
        res = update_debate_state_with_payload(
            state=state,
            raw_response=msg2_raw,
            speaker_label="Bear Analyst",
            speaker_key="Bear",
            stance="bearish",
            history_key="bear_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )

        assert res["count"] == 2
        assert res.get("blocked") is True
        assert res.get("parse_status") == "invalid_protocol"
        assert res["claims"] == initial_claims  # Claims unchanged
        assert len(res["round_messages"]) == 2
        assert res["round_messages"][1]["parse_status"] == "invalid_protocol"

    def test_subsequent_round_responding_to_own_claim_is_rejected(self):
        """Responding only to own camp's claims fails protocol."""
        state = _make_initial_state()["investment_debate_state"]

        # Setup: INV-1 (Bull), INV-2 (Bear)
        state["claims"] = [
            {"claim_id": "INV-1", "speaker_key": "Bull", "stance": "bullish", "status": "open", "claim": "多头观点"},
            {"claim_id": "INV-2", "speaker_key": "Bear", "stance": "bearish", "status": "open", "claim": "空头观点"},
        ]
        state["count"] = 2  # next message is index 3 (Bull)
        state["claim_counter"] = 2

        # Bull tries to respond to Bull's own claim INV-1 instead of Bear's INV-2
        payload = {
            "responded_claim_ids": ["INV-1"],  # Responding to own claim!
            "new_claims": [{"claim": "多头自我补充", "evidence": ["e"], "confidence": 0.85, "target_claim_ids": ["INV-2"]}],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "next_focus_claim_ids": [],
            "round_summary": "多头自我补充",
            "round_goal": "深化",
        }
        raw_resp = f"多头只回应自己。\n<!-- DEBATE_STATE: {json.dumps(payload, ensure_ascii=False)} -->"
        res = update_debate_state_with_payload(
            state=state,
            raw_response=raw_resp,
            speaker_label="Bull Analyst",
            speaker_key="Bull",
            stance="bullish",
            history_key="bull_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )

        assert res["count"] == 3
        assert res.get("blocked") is True
        assert res.get("parse_status") == "invalid_protocol"
        assert len(res["round_messages"]) == 1
        assert res["round_messages"][0]["parse_status"] == "invalid_protocol"

    def test_subsequent_round_new_claims_without_opponent_target_is_rejected(self):
        """New claims having target_claim_ids=[] or targeting own claims fails protocol on round >= 2."""
        state = _make_initial_state()["investment_debate_state"]

        state["claims"] = [
            {"claim_id": "INV-1", "speaker_key": "Bull", "stance": "bullish", "status": "open", "claim": "多头观点"},
        ]
        state["count"] = 1  # next message is index 2 (Bear)
        state["claim_counter"] = 1

        payload = {
            "responded_claim_ids": ["INV-1"],
            "new_claims": [
                {
                    "claim": "空头新claim但无target",
                    "evidence": ["e"],
                    "confidence": 0.8,
                    "target_claim_ids": [],  # Violates Rule: must target opponent claim
                }
            ],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "next_focus_claim_ids": [],
            "round_summary": "空头新claim无target",
            "round_goal": "攻击",
        }
        raw_resp = f"空头新claim无target。\n<!-- DEBATE_STATE: {json.dumps(payload, ensure_ascii=False)} -->"
        res = update_debate_state_with_payload(
            state=state,
            raw_response=raw_resp,
            speaker_label="Bear Analyst",
            speaker_key="Bear",
            stance="bearish",
            history_key="bear_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )

        assert res["count"] == 2
        assert res.get("blocked") is True
        assert res.get("parse_status") == "invalid_protocol"
        assert res["claims"][0]["status"] == "open"  # Not changed

    def test_unauthorized_resolve_opponent_claim_is_rejected(self):
        """One camp resolving opponent's claim is an unauthorized permission violation."""
        state = _make_initial_state()["investment_debate_state"]

        state["claims"] = [
            {"claim_id": "INV-1", "speaker_key": "Bull", "stance": "bullish", "status": "open", "claim": "多头核心主张"},
        ]
        state["count"] = 1  # next is Bear
        state["claim_counter"] = 1

        # Bear attempts to mark Bull's INV-1 as resolved
        payload = {
            "responded_claim_ids": ["INV-1"],
            "new_claims": [{"claim": "空头主张", "evidence": ["e"], "confidence": 0.8, "target_claim_ids": ["INV-1"]}],
            "resolved_claim_ids": ["INV-1"],  # Unauthorized! Bear cannot resolve Bull's claim
            "unresolved_claim_ids": [],
            "next_focus_claim_ids": [],
            "round_summary": "空头擅自解决多头claim",
            "round_goal": "解决",
        }
        raw_resp = f"空头试图单方面宣布多头已被解决。\n<!-- DEBATE_STATE: {json.dumps(payload, ensure_ascii=False)} -->"
        res = update_debate_state_with_payload(
            state=state,
            raw_response=raw_resp,
            speaker_label="Bear Analyst",
            speaker_key="Bear",
            stance="bearish",
            history_key="bear_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )

        assert res["count"] == 2
        assert res.get("blocked") is True
        assert res.get("parse_status") == "invalid_protocol"
        assert res["claims"][0]["status"] == "open"  # INV-1 remains open

    def test_malformed_and_missing_blocks_recorded_with_correct_parse_status(self):
        """Malformed and missing machine blocks record parse_status='invalid' or 'missing'."""
        state = _make_initial_state()["investment_debate_state"]

        # Missing block
        raw_missing = "纯正文，完全没有 DEBATE_STATE 块。"
        res1 = update_debate_state_with_payload(
            state=state,
            raw_response=raw_missing,
            speaker_label="Bull Analyst",
            speaker_key="Bull",
            stance="bullish",
            history_key="bull_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        assert res1["count"] == 1
        assert res1.get("blocked") is True
        assert res1.get("parse_status") == "missing"
        assert len(res1["round_messages"]) == 1
        assert res1["round_messages"][0]["parse_status"] == "missing"

        # Malformed JSON block
        raw_invalid = "多头正文。\n<!-- DEBATE_STATE: {broken json} -->"
        res2 = update_debate_state_with_payload(
            state=state,
            raw_response=raw_invalid,
            speaker_label="Bull Analyst",
            speaker_key="Bull",
            stance="bullish",
            history_key="bull_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
        )
        assert res2["count"] == 1
        assert res2.get("blocked") is True
        assert res2.get("parse_status") == "invalid"
        assert len(res2["round_messages"]) == 1
        assert res2["round_messages"][0]["parse_status"] == "invalid"


class TestRoundMessagesStatePersistence:
    """Test Contract 2: round_messages persistence through research manager and graph state."""

    def test_research_manager_preserves_round_messages(self):
        round_messages = [
            {"message_index": 1, "debate_round": 1, "speaker": "Bull Analyst", "parse_status": "valid"},
            {"message_index": 2, "debate_round": 1, "speaker": "Bear Analyst", "parse_status": "valid"},
        ]
        state = _make_initial_state({
            "investment_debate_state": {
                "history": "辩论历史",
                "bull_history": "多头历史",
                "bear_history": "空头历史",
                "current_speaker": "Bear",
                "current_response": "空头陈述",
                "count": 2,
                "claims": [{"claim_id": "INV-1", "claim": "多头主张", "status": "addressed"}],
                "round_messages": round_messages,
                "focus_claim_ids": ["INV-1"],
                "open_claim_ids": ["INV-1"],
                "resolved_claim_ids": [],
                "unresolved_claim_ids": ["INV-1"],
                "round_summary": "首轮辩论结束",
                "round_goal": "收敛",
                "claim_counter": 1,
            },
            "fund_flow_consensus_guard": {
                "blocked": False,
                "direction_allowed": True,
                "status": "consensus",
            },
        })

        mock_llm = MagicMock()
        mock_llm.astream = MagicMock(side_effect=lambda prompt: _fake_stream("投研经理裁决：多头胜出。\n<!-- VERDICT: {\"direction\": \"看多\", \"reason\": \"多头证据扎实\"} -->"))
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])

        rm_node = create_research_manager(mock_llm, memory)
        result = asyncio.run(rm_node(state))

        assert "investment_debate_state" in result
        new_inv_state = result["investment_debate_state"]
        assert "round_messages" in new_inv_state
        assert len(new_inv_state["round_messages"]) == 2
        assert new_inv_state["round_messages"] == round_messages
