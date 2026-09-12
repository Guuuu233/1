"""Tests for E-03b-1 Bull researcher claim reference boundary contracts (DAV-846).

Coverage:
1. Bull v2 Opening with no opponent claims does not hint/generate placeholder opponent IDs.
   Legal Opening passes and enters state ledger.
2. Unknown responded_claim_ids / target_claim_ids / challenge target are rejected fail-closed,
   and do not enter accepted round/state ledger.
3. Existing legal opponent claims in non-Opening stages can still be referenced.
4. Retry mechanisms: attempt 1 failure warns with error detail; if attempt 2 fixes it,
   count advances by 1 and only attempt 2 enters accepted state; if attempt 2 still fails,
   DebateProtocolError is raised.
"""
from __future__ import annotations

import asyncio
import copy
from typing import Any
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.utils.agent_states import (
    PROTOCOL_VERSION_V2_STRUCTURED,
)
from tradingagents.agents.utils.debate_utils import (
    DebateProtocolError,
)


class FakeStreamingLLM:
    """Fake streaming LLM that captures all prompts passed to astream and yields configured responses."""

    def __init__(self, responses: list[str] | None = None):
        self.captured_prompts: list[str] = []
        self.responses = responses or []
        self.call_count = 0

    async def astream(self, prompt: str):
        self.captured_prompts.append(prompt)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = (
                "默认立论回复\n"
                '<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": ['
                '{"claim": "主力大单流入", "evidence": ["流入1.2亿"], "confidence": 0.8, "battlefield": "capital_flow", "target_claim_ids": []}, '
                '{"claim": "题材情绪向好", "evidence": ["政策支持"], "confidence": 0.8, "battlefield": "sentiment_theme", "target_claim_ids": []}, '
                '{"claim": "量价多头排列", "evidence": ["突破60日线"], "confidence": 0.8, "battlefield": "price_volume", "target_claim_ids": []}'
                '], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
                '"round_summary": "多头立论", "round_goal": "建立多头核心立论"} -->'
            )
        self.call_count += 1
        for chunk in [resp[: len(resp) // 2], resp[len(resp) // 2 :]]:
            mock_chunk = MagicMock()
            mock_chunk.content = chunk
            yield mock_chunk


def _make_opening_state() -> dict[str, Any]:
    return {
        "macro_report": "宏观报告：流动性维持充裕。",
        "market_report": "市场报告：突破关键阻力位。",
        "sentiment_report": "情绪报告：做多意愿强烈。",
        "news_report": "新闻报告：行业景气度上行。",
        "fundamentals_report": "基本面报告：营收净利双增。",
        "smart_money_report": "主力资金报告：大单净流入。",
        "volume_price_report": "量价报告：放量长阳。",
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_speaker": "",
            "current_response": "",
            "count": 0,
            "claims": [],
            "round_messages": [],
            "attempts": [],
            "focus_claim_ids": [],
            "open_claim_ids": [],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "round_summary": "",
            "round_goal": "建立核心多头立论",
            "claim_counter": 0,
            "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
            "protocol_stage": "opening",
            "feature_flags": {"v2_debate_enabled": True},
        },
    }


def _make_challenge_state() -> dict[str, Any]:
    bull_claims = [
        {"claim_id": "INV-1", "speaker_key": "Bull", "speaker": "Bull Analyst", "stance": "bullish", "claim": "主力资金持续净流入", "evidence": ["主力资金净流入1.2亿"], "confidence": 0.85, "battlefield": "capital_flow", "debate_round": 1, "message_index": 1, "stage": "opening", "status": "open", "target_claim_ids": []},
        {"claim_id": "INV-2", "speaker_key": "Bull", "speaker": "Bull Analyst", "stance": "bullish", "claim": "行业题材景气度高", "evidence": ["宏观政策密集落地"], "confidence": 0.80, "battlefield": "sentiment_theme", "debate_round": 1, "message_index": 1, "stage": "opening", "status": "open", "target_claim_ids": []},
        {"claim_id": "INV-3", "speaker_key": "Bull", "speaker": "Bull Analyst", "stance": "bullish", "claim": "量价突破均线多头", "evidence": ["放量突破60日线"], "confidence": 0.78, "battlefield": "price_volume", "debate_round": 1, "message_index": 1, "stage": "opening", "status": "open", "target_claim_ids": []},
    ]
    bear_claims = [
        {"claim_id": "INV-4", "speaker_key": "Bear", "speaker": "Bear Analyst", "stance": "bearish", "claim": "应收账款恶化现金流承压", "evidence": ["经营现金流同比下滑30%"], "confidence": 0.82, "battlefield": "fundamentals", "debate_round": 1, "message_index": 2, "stage": "opening", "status": "open", "target_claim_ids": []},
        {"claim_id": "INV-5", "speaker_key": "Bear", "speaker": "Bear Analyst", "stance": "bearish", "claim": "外需降温出口面临逆风", "evidence": ["出口交货值同比下降"], "confidence": 0.75, "battlefield": "macro_policy", "debate_round": 1, "message_index": 2, "stage": "opening", "status": "open", "target_claim_ids": []},
        {"claim_id": "INV-6", "speaker_key": "Bear", "speaker": "Bear Analyst", "stance": "bearish", "claim": "高位筹码松动获利盘兑现", "evidence": ["高位换手率超过25%"], "confidence": 0.70, "battlefield": "capital_flow", "debate_round": 1, "message_index": 2, "stage": "opening", "status": "open", "target_claim_ids": []},
    ]
    all_claims = bull_claims + bear_claims
    return {
        "macro_report": "宏观报告：流动性维持充裕。",
        "market_report": "市场报告：突破关键阻力位。",
        "sentiment_report": "情绪报告：做多意愿强烈。",
        "news_report": "新闻报告：行业景气度上行。",
        "fundamentals_report": "基本面报告：营收净利双增。",
        "smart_money_report": "主力资金报告：大单净流入。",
        "volume_price_report": "量价报告：放量长阳。",
        "investment_debate_state": {
            "history": "辩论历史",
            "bull_history": "多头历史",
            "bear_history": "空头历史",
            "current_speaker": "Bear",
            "current_response": "空头立论",
            "count": 2,
            "claims": all_claims,
            "claim_counter": 6,
            "challenges": [],
            "challenge_counter": 0,
            "open_claim_ids": [f"INV-{i}" for i in range(1, 7)],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "round_messages": [
                {
                    "message_index": 1,
                    "debate_round": 1,
                    "stage": "opening",
                    "protocol_stage": "opening",
                    "speaker": "Bull Analyst",
                    "speaker_key": "Bull",
                    "new_claim_ids": ["INV-1", "INV-2", "INV-3"],
                    "responded_claim_ids": [],
                    "target_claim_ids": [],
                    "parse_status": "valid",
                    "accepted": True,
                },
                {
                    "message_index": 2,
                    "debate_round": 1,
                    "stage": "opening",
                    "protocol_stage": "opening",
                    "speaker": "Bear Analyst",
                    "speaker_key": "Bear",
                    "new_claim_ids": ["INV-4", "INV-5", "INV-6"],
                    "responded_claim_ids": [],
                    "target_claim_ids": [],
                    "parse_status": "valid",
                    "accepted": True,
                },
            ],
            "protocol_version": PROTOCOL_VERSION_V2_STRUCTURED,
            "protocol_stage": "challenge",
            "feature_flags": {"v2_debate_enabled": True},
        },
    }


def _make_bear_opening_state() -> dict[str, Any]:
    state = _make_opening_state()
    inv_state = state["investment_debate_state"]
    inv_state["count"] = 1
    inv_state["current_speaker"] = "Bull"
    inv_state["current_response"] = "多头立论发言正文"
    inv_state["bull_history"] = "Bull Analyst: 多头立论发言正文"
    inv_state["round_goal"] = "建立核心空头立论"
    inv_state["claims"] = [
        {"claim_id": "INV-1", "speaker_key": "Bull", "speaker": "Bull Analyst", "stance": "bullish", "claim": "主力资金持续净流入", "evidence": ["流入1.2亿"], "confidence": 0.85, "battlefield": "capital_flow", "debate_round": 1, "message_index": 1, "stage": "opening", "status": "open", "target_claim_ids": []},
        {"claim_id": "INV-2", "speaker_key": "Bull", "speaker": "Bull Analyst", "stance": "bullish", "claim": "行业题材景气度高", "evidence": ["政策密集落地"], "confidence": 0.80, "battlefield": "sentiment_theme", "debate_round": 1, "message_index": 1, "stage": "opening", "status": "open", "target_claim_ids": []},
        {"claim_id": "INV-3", "speaker_key": "Bull", "speaker": "Bull Analyst", "stance": "bullish", "claim": "量价突破均线多头", "evidence": ["突破60日线"], "confidence": 0.78, "battlefield": "price_volume", "debate_round": 1, "message_index": 1, "stage": "opening", "status": "open", "target_claim_ids": []},
    ]
    inv_state["claim_counter"] = 3
    inv_state["open_claim_ids"] = ["INV-1", "INV-2", "INV-3"]
    inv_state["round_messages"] = [
        {
            "message_index": 1,
            "debate_round": 1,
            "stage": "opening",
            "protocol_stage": "opening",
            "speaker": "Bull Analyst",
            "speaker_key": "Bull",
            "new_claim_ids": ["INV-1", "INV-2", "INV-3"],
            "responded_claim_ids": [],
            "target_claim_ids": [],
            "parse_status": "valid",
            "accepted": True,
        }
    ]
    return state


def _make_bear_challenge_state() -> dict[str, Any]:
    state = _make_challenge_state()
    inv_state = state["investment_debate_state"]
    inv_state["count"] = 3
    inv_state["current_speaker"] = "Bull"
    inv_state["current_response"] = "多头盘问空头"
    inv_state["challenges"] = [
        {
            "challenge_id": "CH-1",
            "target_claim_id": "INV-4",
            "challenger_speaker": "Bull Analyst",
            "challenger_stance": "bullish",
            "weakest_point": "三季度大额计提已完毕",
            "evidence": ["计提充分"],
            "severity": "major",
            "debate_round": 2,
            "message_index": 3,
        }
    ]
    inv_state["challenge_counter"] = 1
    inv_state["round_messages"].append(
        {
            "message_index": 3,
            "debate_round": 2,
            "stage": "challenge",
            "protocol_stage": "challenge",
            "speaker": "Bull Analyst",
            "speaker_key": "Bull",
            "new_claim_ids": [],
            "responded_claim_ids": ["INV-4"],
            "target_claim_ids": [],
            "challenge_ids": ["CH-1"],
            "parse_status": "valid",
            "accepted": True,
        }
    )
    return state


def _make_bear_tiebreak_state() -> dict[str, Any]:
    state = _make_bear_challenge_state()
    inv_state = state["investment_debate_state"]
    inv_state["count"] = 5
    inv_state["protocol_stage"] = "tiebreak"
    inv_state["round_messages"].extend([
        {
            "message_index": 4,
            "debate_round": 2,
            "stage": "challenge",
            "protocol_stage": "challenge",
            "speaker": "Bear Analyst",
            "speaker_key": "Bear",
            "new_claim_ids": [],
            "responded_claim_ids": ["INV-1"],
            "target_claim_ids": [],
            "challenge_ids": ["CH-2"],
            "parse_status": "valid",
            "accepted": True,
        },
        {
            "message_index": 5,
            "debate_round": 3,
            "stage": "tiebreak",
            "protocol_stage": "tiebreak",
            "speaker": "Bull Analyst",
            "speaker_key": "Bull",
            "new_claim_ids": ["INV-7"],
            "responded_claim_ids": ["INV-4"],
            "target_claim_ids": ["INV-4"],
            "parse_status": "valid",
            "accepted": True,
        },
    ])
    inv_state["claims"].append({
        "claim_id": "INV-7",
        "speaker_key": "Bull",
        "speaker": "Bull Analyst",
        "stance": "bullish",
        "claim": "多头终局决胜立论",
        "evidence": ["终局证据"],
        "confidence": 0.88,
        "battlefield": "capital_flow",
        "debate_round": 3,
        "message_index": 5,
        "stage": "tiebreak",
        "status": "open",
        "target_claim_ids": ["INV-4"],
    })
    inv_state["claim_counter"] = 7
    inv_state["open_claim_ids"].append("INV-7")
    return state


class TestBullOpeningDoubleBlind:
    """1. v2 Opening 的 Bull 首次发言没有对手命题可引用。双盲约束与占位符防范。"""

    def test_bull_v2_opening_prompt_has_no_placeholder_claim_ids(self):
        """Bull v2 Opening prompt has responded_claim_ids=[] and new_claims[].target_claim_ids=[] with no placeholder opponent IDs."""
        state = _make_opening_state()
        llm = FakeStreamingLLM()
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        asyncio.run(node(state))

        assert len(llm.captured_prompts) >= 1
        prompt = llm.captured_prompts[0]
        # In Opening prompt, responded_claim_ids must be [] and target_claim_ids must be []
        assert '"responded_claim_ids": []' in prompt
        assert '"target_claim_ids": []' in prompt
        # No placeholder opponent claim IDs
        assert "INV-1" not in prompt
        assert "OPPONENT_CLAIM_ID" not in prompt
        assert "INV-4" not in prompt

    def test_bull_v2_opening_retry_prompt_has_no_placeholder_claim_ids(self):
        """When attempt 1 fails Opening validation, retry prompt retains double-blind with no placeholder opponent IDs."""
        state = _make_opening_state()
        bad_opening = (
            "立论发言\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1"], "new_claims": ['
            '{"claim": "主力大单流入", "evidence": ["流入1.2亿"], "confidence": 0.8, "battlefield": "capital_flow", "target_claim_ids": []}, '
            '{"claim": "题材情绪向好", "evidence": ["政策支持"], "confidence": 0.8, "battlefield": "sentiment_theme", "target_claim_ids": []}, '
            '{"claim": "量价多头排列", "evidence": ["突破60日线"], "confidence": 0.8, "battlefield": "price_volume", "target_claim_ids": []}'
            '], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
            '"round_summary": "多头立论", "round_goal": "建立多头核心立论"} -->'
        )
        good_opening = (
            "合规立论发言\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": ['
            '{"claim": "主力大单流入", "evidence": ["流入1.2亿"], "confidence": 0.8, "battlefield": "capital_flow", "target_claim_ids": []}, '
            '{"claim": "题材情绪向好", "evidence": ["政策支持"], "confidence": 0.8, "battlefield": "sentiment_theme", "target_claim_ids": []}, '
            '{"claim": "量价多头排列", "evidence": ["突破60日线"], "confidence": 0.8, "battlefield": "price_volume", "target_claim_ids": []}'
            '], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
            '"round_summary": "多头立论", "round_goal": "建立多头核心立论"} -->'
        )
        llm = FakeStreamingLLM([bad_opening, good_opening])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        res = asyncio.run(node(state))

        assert len(llm.captured_prompts) == 2
        retry_prompt = llm.captured_prompts[1]
        assert "【协议重试警告 (Attempt 2)】" in retry_prompt
        assert "responded_claim_ids 必须为空数组 []" in retry_prompt
        assert "target_claim_ids 必须为空数组 []" in retry_prompt
        assert "INV-1" not in retry_prompt.split("【协议重试警告")[1].split("错误原因")[0]  # instruction itself has no INV-1

        # Res succeeds
        inv_state = res["investment_debate_state"]
        assert inv_state["count"] == 1
        accepted_msgs = [m for m in inv_state["round_messages"] if m.get("accepted") is True]
        assert len(accepted_msgs) == 1
        assert len(inv_state["claims"]) == 3

    def test_bull_v2_opening_valid_response_accepted_and_enters_ledger(self):
        """Valid v2 Opening response without opponent reference is accepted and enters ledger."""
        state = _make_opening_state()
        llm = FakeStreamingLLM()
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        res = asyncio.run(node(state))

        inv_state = res["investment_debate_state"]
        assert inv_state["count"] == 1
        assert len(inv_state["claims"]) == 3
        for c in inv_state["claims"]:
            assert c["stage"] == "opening"
            assert c["speaker_key"] == "Bull"
            assert c["target_claim_ids"] == []
        assert len(inv_state["round_messages"]) == 1
        assert inv_state["round_messages"][0]["accepted"] is True
        assert inv_state["round_messages"][0]["responded_claim_ids"] == []
        assert inv_state["round_messages"][0]["target_claim_ids"] == []

    def test_bull_v2_opening_with_placeholder_responded_claim_id_rejected(self):
        """Opening response with placeholder opponent claim ID in responded_claim_ids is rejected."""
        state = _make_opening_state()
        bad_opening = (
            "立论发言\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-4"], "new_claims": ['
            '{"claim": "主力大单流入", "evidence": ["流入1.2亿"], "confidence": 0.8, "battlefield": "capital_flow", "target_claim_ids": []}, '
            '{"claim": "题材情绪向好", "evidence": ["政策支持"], "confidence": 0.8, "battlefield": "sentiment_theme", "target_claim_ids": []}, '
            '{"claim": "量价多头排列", "evidence": ["突破60日线"], "confidence": 0.8, "battlefield": "price_volume", "target_claim_ids": []}'
            '], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
            '"round_summary": "多头立论", "round_goal": "建立多头核心立论"} -->'
        )
        llm = FakeStreamingLLM([bad_opening, bad_opening])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert exc_info.value.message_index == 1
        assert exc_info.value.speaker == "Bull Analyst"
        # All attempts rejected
        assert all(a["accepted"] is False for a in exc_info.value.attempts)
        # Original state is unchanged (never entered state ledger)
        assert state["investment_debate_state"]["count"] == 0
        assert len(state["investment_debate_state"]["claims"]) == 0

    def test_bull_v2_opening_with_placeholder_target_claim_id_rejected(self):
        """Opening response with placeholder target_claim_ids on new_claims is rejected."""
        state = _make_opening_state()
        bad_opening = (
            "立论发言\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": ['
            '{"claim": "主力大单流入", "evidence": ["流入1.2亿"], "confidence": 0.8, "battlefield": "capital_flow", "target_claim_ids": ["INV-1"]}, '
            '{"claim": "题材情绪向好", "evidence": ["政策支持"], "confidence": 0.8, "battlefield": "sentiment_theme", "target_claim_ids": []}, '
            '{"claim": "量价多头排列", "evidence": ["突破60日线"], "confidence": 0.8, "battlefield": "price_volume", "target_claim_ids": []}'
            '], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
            '"round_summary": "多头立论", "round_goal": "建立多头核心立论"} -->'
        )
        llm = FakeStreamingLLM([bad_opening, bad_opening])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert exc_info.value.message_index == 1
        assert state["investment_debate_state"]["count"] == 0

    def test_bull_v2_opening_with_challenges_rejected(self):
        """Opening response with challenges payload is rejected fail-closed."""
        state = _make_opening_state()
        bad_opening = (
            "立论发言\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": ['
            '{"claim": "主力大单流入", "evidence": ["流入1.2亿"], "confidence": 0.8, "battlefield": "capital_flow", "target_claim_ids": []}, '
            '{"claim": "题材情绪向好", "evidence": ["政策支持"], "confidence": 0.8, "battlefield": "sentiment_theme", "target_claim_ids": []}, '
            '{"claim": "量价多头排列", "evidence": ["突破60日线"], "confidence": 0.8, "battlefield": "price_volume", "target_claim_ids": []}'
            '], "challenges": [{"target_claim_id": "INV-4", "weakest_point": "假设脆弱", "evidence": ["数据"], "severity": "major"}], '
            '"resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
            '"round_summary": "多头立论", "round_goal": "建立多头核心立论"} -->'
        )
        llm = FakeStreamingLLM([bad_opening, bad_opening])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        with pytest.raises(DebateProtocolError):
            asyncio.run(node(state))

        assert state["investment_debate_state"]["count"] == 0


class TestBullUnknownClaimFailClosed:
    """2. 任一命题引用字段若带有当前 claim ledger 不存在的 ID，必须 fail-closed。"""

    def test_bull_unknown_responded_claim_id_rejected_in_challenge_stage(self):
        """Unknown responded_claim_ids in Challenge stage is rejected and does not enter accepted ledger."""
        state = _make_challenge_state()
        bad_challenge = (
            "多头盘问\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-4", "INV-999"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-4", "weakest_point": "假设存疑", "evidence": ["数据支撑"], "severity": "major"}], '
            '"self_win_prob": 0.75, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-4"], "next_focus_claim_ids": ["INV-4"], '
            '"round_summary": "盘问", "round_goal": "击穿空头"} -->'
        )
        llm = FakeStreamingLLM([bad_challenge, bad_challenge])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert exc_info.value.message_index == 3
        assert any("INV-999" in a["error_detail"] for a in exc_info.value.attempts)
        assert state["investment_debate_state"]["count"] == 2
        assert len(state["investment_debate_state"]["challenges"]) == 0

    def test_bull_unknown_challenge_target_rejected_in_challenge_stage(self):
        """Unknown challenge target_claim_id is rejected fail-closed."""
        state = _make_challenge_state()
        bad_challenge = (
            "多头盘问\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-999"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-999", "weakest_point": "假设存疑", "evidence": ["数据支撑"], "severity": "major"}], '
            '"self_win_prob": 0.75, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-4"], "next_focus_claim_ids": ["INV-4"], '
            '"round_summary": "盘问", "round_goal": "击穿空头"} -->'
        )
        llm = FakeStreamingLLM([bad_challenge, bad_challenge])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert exc_info.value.message_index == 3
        assert any("INV-999" in a["error_detail"] for a in exc_info.value.attempts)
        assert state["investment_debate_state"]["count"] == 2
        assert len(state["investment_debate_state"]["challenges"]) == 0

    def test_bull_unknown_target_claim_id_rejected_in_tiebreak_stage(self):
        """Unknown target_claim_ids in new_claims during Tiebreak stage is rejected fail-closed."""
        state = _make_challenge_state()
        inv_state = state["investment_debate_state"]
        inv_state["count"] = 4
        inv_state["protocol_stage"] = "tiebreak"
        inv_state["round_messages"].extend([
            {"message_index": 3, "debate_round": 2, "stage": "challenge", "speaker": "Bull Analyst", "speaker_key": "Bull", "accepted": True, "parse_status": "valid"},
            {"message_index": 4, "debate_round": 2, "stage": "challenge", "speaker": "Bear Analyst", "speaker_key": "Bear", "accepted": True, "parse_status": "valid"},
        ])

        bad_tiebreak = (
            "多头收官\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-4"], "new_claims": ['
            '{"claim": "全维度风险收益比具备极高吸引力", "evidence": ["新证据A股估值分位低于10%"], "confidence": 0.88, "target_claim_ids": ["INV-999"]}'
            '], "self_win_prob": 0.82, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-4"], "next_focus_claim_ids": ["INV-4"], '
            '"round_summary": "收官", "round_goal": "决胜"} -->'
        )
        llm = FakeStreamingLLM([bad_tiebreak, bad_tiebreak])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert exc_info.value.message_index == 5
        assert any("INV-999" in a["error_detail"] for a in exc_info.value.attempts)
        assert state["investment_debate_state"]["count"] == 4

    def test_bull_unknown_resolved_or_unresolved_claim_id_rejected(self):
        """Unknown claim IDs in resolved_claim_ids or unresolved_claim_ids are rejected fail-closed."""
        state = _make_challenge_state()
        bad_resolved = (
            "多头盘问\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-4"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-4", "weakest_point": "假设存疑", "evidence": ["数据支撑"], "severity": "major"}], '
            '"self_win_prob": 0.75, "resolved_claim_ids": ["INV-888"], "unresolved_claim_ids": ["INV-4"], "next_focus_claim_ids": ["INV-4"], '
            '"round_summary": "盘问", "round_goal": "击穿空头"} -->'
        )
        llm = FakeStreamingLLM([bad_resolved, bad_resolved])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert any("INV-888" in a["error_detail"] for a in exc_info.value.attempts)

    def test_bull_retry_recovery_from_unknown_claim_id(self):
        """Attempt 1 with unknown claim ID fails, attempt 2 with valid claim ID succeeds; unknown ID never enters state."""
        state = _make_challenge_state()
        bad_challenge = (
            "多头盘问（首次坏块含未知ID）\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-4", "INV-999"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-4", "weakest_point": "假设存疑", "evidence": ["数据支撑"], "severity": "major"}], '
            '"self_win_prob": 0.75, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-4"], "next_focus_claim_ids": ["INV-4"], '
            '"round_summary": "盘问", "round_goal": "击穿空头"} -->'
        )
        good_challenge = (
            "多头盘问（修正后仅引用合法INV-4）\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-4"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-4", "weakest_point": "应收账款计提已充分对冲现金流压力", "evidence": ["三季度大额计提已完毕"], "severity": "major"}], '
            '"self_win_prob": 0.75, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-4"], "next_focus_claim_ids": ["INV-4"], '
            '"round_summary": "盘问空头现金流漏洞", "round_goal": "击穿空头核心立论"} -->'
        )
        llm = FakeStreamingLLM([bad_challenge, good_challenge])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        res = asyncio.run(node(state))

        inv_state = res["investment_debate_state"]
        # count advances by exactly 1 (2 -> 3)
        assert inv_state["count"] == 3
        # Exactly 1 new accepted message
        accepted_msgs = [m for m in inv_state["round_messages"] if m.get("accepted") is True]
        assert len(accepted_msgs) == 3
        latest_msg = inv_state["round_messages"][-1]
        assert latest_msg["accepted"] is True
        assert latest_msg["responded_claim_ids"] == ["INV-4"]
        assert "INV-999" not in latest_msg["responded_claim_ids"]
        # Attempts trace tracks attempt 1 failure and attempt 2 success
        assert len(inv_state["attempts"]) == 2
        assert inv_state["attempts"][0]["accepted"] is False
        assert "INV-999" in inv_state["attempts"][0]["error_detail"]
        assert inv_state["attempts"][1]["accepted"] is True
        # challenges ledger only contains valid CH-1 targeting INV-4
        assert len(inv_state["challenges"]) == 1
        assert inv_state["challenges"][0]["target_claim_id"] == "INV-4"


class TestBullValidOpponentClaimAllowed:
    """3. 已存在的合法对手 claim 在非 Opening 阶段仍可引用。"""

    def test_bull_valid_opponent_claim_accepted_in_challenge_stage(self):
        """In Challenge stage (message 3), referencing legal opponent claim INV-4 is accepted."""
        state = _make_challenge_state()
        valid_challenge = (
            "多头合规盘问空头INV-4\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-4"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-4", "weakest_point": "空头忽略预收款和合同负债大增45%", "evidence": ["三季度预收款35亿"], "severity": "major"}], '
            '"self_win_prob": 0.75, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-4"], "next_focus_claim_ids": ["INV-4"], '
            '"round_summary": "击穿现金流逻辑", "round_goal": "击穿空头立论"} -->'
        )
        llm = FakeStreamingLLM([valid_challenge])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        res = asyncio.run(node(state))

        inv_state = res["investment_debate_state"]
        assert inv_state["count"] == 3
        msg = inv_state["round_messages"][-1]
        assert msg["accepted"] is True
        assert msg["stage"] == "challenge"
        assert msg["responded_claim_ids"] == ["INV-4"]
        assert len(inv_state["challenges"]) == 1
        assert inv_state["challenges"][0]["target_claim_id"] == "INV-4"

    def test_bull_valid_opponent_claim_accepted_in_tiebreak_stage(self):
        """In Tiebreak stage (message 5), referencing legal opponent claim INV-4 in target_claim_ids is accepted."""
        state = _make_challenge_state()
        inv_state = state["investment_debate_state"]
        inv_state["count"] = 4
        inv_state["protocol_stage"] = "tiebreak"
        inv_state["round_messages"].extend([
            {"message_index": 3, "debate_round": 2, "stage": "challenge", "speaker": "Bull Analyst", "speaker_key": "Bull", "accepted": True, "parse_status": "valid"},
            {"message_index": 4, "debate_round": 2, "stage": "challenge", "speaker": "Bear Analyst", "speaker_key": "Bear", "accepted": True, "parse_status": "valid"},
        ])

        valid_tiebreak = (
            "多头合规决胜收官\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-4"], "new_claims": ['
            '{"claim": "全维度风险收益比具备极高吸引力", "evidence": ["A股估值历史分位低于10%"], "confidence": 0.88, "target_claim_ids": ["INV-4"]}'
            '], "self_win_prob": 0.82, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-4"], "next_focus_claim_ids": ["INV-4"], '
            '"round_summary": "决胜收官", "round_goal": "多头决胜"} -->'
        )
        llm = FakeStreamingLLM([valid_tiebreak])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bull_researcher(llm, memory)
        res = asyncio.run(node(state))

        new_inv_state = res["investment_debate_state"]
        assert new_inv_state["count"] == 5
        msg = new_inv_state["round_messages"][-1]
        assert msg["accepted"] is True
        assert msg["stage"] == "tiebreak"
        assert msg["responded_claim_ids"] == ["INV-4"]
        assert msg["target_claim_ids"] == ["INV-4"]


class TestBearOpeningDoubleBlind:
    """1. v2 Opening 的 Bear 首次发言没有对手命题可引用。双盲约束与占位符防范。"""

    def test_bear_v2_opening_prompt_has_no_placeholder_claim_ids(self):
        """Bear v2 Opening prompt has responded_claim_ids=[] and new_claims[].target_claim_ids=[] with no placeholder opponent IDs."""
        state = _make_bear_opening_state()
        llm = FakeStreamingLLM()
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        asyncio.run(node(state))

        assert len(llm.captured_prompts) >= 1
        prompt = llm.captured_prompts[0]
        # In Opening prompt, responded_claim_ids must be [] and target_claim_ids must be []
        assert '"responded_claim_ids": []' in prompt
        assert '"target_claim_ids": []' in prompt
        # No placeholder opponent claim IDs
        assert "INV-1" not in prompt
        assert "OPPONENT_CLAIM_ID" not in prompt
        assert "INV-4" not in prompt

    def test_bear_v2_opening_retry_prompt_has_no_placeholder_claim_ids(self):
        """When attempt 1 fails Opening validation, retry prompt retains double-blind with no placeholder opponent IDs."""
        state = _make_bear_opening_state()
        bad_opening = (
            "空头立论发言\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1"], "new_claims": ['
            '{"claim": "应收账款恶化现金流承压", "evidence": ["经营现金流同比下滑30%"], "confidence": 0.82, "battlefield": "fundamentals", "target_claim_ids": []}, '
            '{"claim": "外需降温出口面临逆风", "evidence": ["出口交货值同比下降"], "confidence": 0.75, "battlefield": "macro_policy", "target_claim_ids": []}, '
            '{"claim": "高位筹码松动获利盘兑现", "evidence": ["高位换手率超过25%"], "confidence": 0.70, "battlefield": "capital_flow", "target_claim_ids": []}'
            '], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
            '"round_summary": "空头立论", "round_goal": "建立空头核心立论"} -->'
        )
        good_opening = (
            "合规空头立论发言\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": ['
            '{"claim": "应收账款恶化现金流承压", "evidence": ["经营现金流同比下滑30%"], "confidence": 0.82, "battlefield": "fundamentals", "target_claim_ids": []}, '
            '{"claim": "外需降温出口面临逆风", "evidence": ["出口交货值同比下降"], "confidence": 0.75, "battlefield": "macro_policy", "target_claim_ids": []}, '
            '{"claim": "高位筹码松动获利盘兑现", "evidence": ["高位换手率超过25%"], "confidence": 0.70, "battlefield": "capital_flow", "target_claim_ids": []}'
            '], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
            '"round_summary": "空头立论", "round_goal": "建立空头核心立论"} -->'
        )
        llm = FakeStreamingLLM([bad_opening, good_opening])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        res = asyncio.run(node(state))

        assert len(llm.captured_prompts) == 2
        retry_prompt = llm.captured_prompts[1]
        assert "【协议重试警告 (Attempt 2)】" in retry_prompt
        assert "responded_claim_ids 必须为空数组 []" in retry_prompt
        assert "target_claim_ids 必须为空数组 []" in retry_prompt
        assert "INV-1" not in retry_prompt.split("【协议重试警告")[1].split("错误原因")[0]  # instruction itself has no INV-1

        # Res succeeds
        inv_state = res["investment_debate_state"]
        assert inv_state["count"] == 2
        accepted_msgs = [m for m in inv_state["round_messages"] if m.get("accepted") is True]
        assert len(accepted_msgs) == 2
        assert len(inv_state["claims"]) == 6

    def test_bear_v2_opening_valid_response_accepted_and_enters_ledger(self):
        """Valid v2 Opening response without opponent reference is accepted and enters ledger."""
        state = _make_bear_opening_state()
        good_opening = (
            "合规空头立论发言\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": ['
            '{"claim": "应收账款恶化现金流承压", "evidence": ["经营现金流同比下滑30%"], "confidence": 0.82, "battlefield": "fundamentals", "target_claim_ids": []}, '
            '{"claim": "外需降温出口面临逆风", "evidence": ["出口交货值同比下降"], "confidence": 0.75, "battlefield": "macro_policy", "target_claim_ids": []}, '
            '{"claim": "高位筹码松动获利盘兑现", "evidence": ["高位换手率超过25%"], "confidence": 0.70, "battlefield": "capital_flow", "target_claim_ids": []}'
            '], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
            '"round_summary": "空头立论", "round_goal": "建立空头核心立论"} -->'
        )
        llm = FakeStreamingLLM([good_opening])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        res = asyncio.run(node(state))

        inv_state = res["investment_debate_state"]
        assert inv_state["count"] == 2
        assert len(inv_state["claims"]) == 6
        bear_claims = [c for c in inv_state["claims"] if c["speaker_key"] == "Bear"]
        assert len(bear_claims) == 3
        for c in bear_claims:
            assert c["stage"] == "opening"
            assert c["target_claim_ids"] == []
        assert len(inv_state["round_messages"]) == 2
        latest_msg = inv_state["round_messages"][-1]
        assert latest_msg["accepted"] is True
        assert latest_msg["speaker_key"] == "Bear"
        assert latest_msg["responded_claim_ids"] == []
        assert latest_msg["target_claim_ids"] == []

    def test_bear_v2_opening_with_placeholder_responded_claim_id_rejected(self):
        """Opening response with opponent claim ID in responded_claim_ids is rejected."""
        state = _make_bear_opening_state()
        bad_opening = (
            "空头立论发言（违规引用多头INV-1）\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1"], "new_claims": ['
            '{"claim": "应收账款恶化现金流承压", "evidence": ["经营现金流同比下滑30%"], "confidence": 0.82, "battlefield": "fundamentals", "target_claim_ids": []}, '
            '{"claim": "外需降温出口面临逆风", "evidence": ["出口交货值同比下降"], "confidence": 0.75, "battlefield": "macro_policy", "target_claim_ids": []}, '
            '{"claim": "高位筹码松动获利盘兑现", "evidence": ["高位换手率超过25%"], "confidence": 0.70, "battlefield": "capital_flow", "target_claim_ids": []}'
            '], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
            '"round_summary": "空头立论", "round_goal": "建立空头核心立论"} -->'
        )
        llm = FakeStreamingLLM([bad_opening, bad_opening])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert exc_info.value.message_index == 2
        assert exc_info.value.speaker == "Bear Analyst"
        assert all(a["accepted"] is False for a in exc_info.value.attempts)
        # Original state is unchanged (never entered state ledger)
        assert state["investment_debate_state"]["count"] == 1
        assert len(state["investment_debate_state"]["claims"]) == 3

    def test_bear_v2_opening_with_placeholder_target_claim_id_rejected(self):
        """Opening response with placeholder target_claim_ids on new_claims is rejected."""
        state = _make_bear_opening_state()
        bad_opening = (
            "空头立论发言（违规包含target_claim_ids）\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": ['
            '{"claim": "应收账款恶化现金流承压", "evidence": ["经营现金流同比下滑30%"], "confidence": 0.82, "battlefield": "fundamentals", "target_claim_ids": ["INV-1"]}, '
            '{"claim": "外需降温出口面临逆风", "evidence": ["出口交货值同比下降"], "confidence": 0.75, "battlefield": "macro_policy", "target_claim_ids": []}, '
            '{"claim": "高位筹码松动获利盘兑现", "evidence": ["高位换手率超过25%"], "confidence": 0.70, "battlefield": "capital_flow", "target_claim_ids": []}'
            '], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
            '"round_summary": "空头立论", "round_goal": "建立空头核心立论"} -->'
        )
        llm = FakeStreamingLLM([bad_opening, bad_opening])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert exc_info.value.message_index == 2
        assert state["investment_debate_state"]["count"] == 1
        assert len(state["investment_debate_state"]["claims"]) == 3

    def test_bear_v2_opening_with_challenges_rejected(self):
        """Opening response with challenges payload is rejected fail-closed."""
        state = _make_bear_opening_state()
        bad_opening = (
            "空头立论发言（违规包含challenges）\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": ['
            '{"claim": "应收账款恶化现金流承压", "evidence": ["经营现金流同比下滑30%"], "confidence": 0.82, "battlefield": "fundamentals", "target_claim_ids": []}, '
            '{"claim": "外需降温出口面临逆风", "evidence": ["出口交货值同比下降"], "confidence": 0.75, "battlefield": "macro_policy", "target_claim_ids": []}, '
            '{"claim": "高位筹码松动获利盘兑现", "evidence": ["高位换手率超过25%"], "confidence": 0.70, "battlefield": "capital_flow", "target_claim_ids": []}'
            '], "challenges": [{"target_claim_id": "INV-1", "weakest_point": "假设脆弱", "evidence": ["数据"], "severity": "major"}], '
            '"resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
            '"round_summary": "空头立论", "round_goal": "建立空头核心立论"} -->'
        )
        llm = FakeStreamingLLM([bad_opening, bad_opening])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        with pytest.raises(DebateProtocolError):
            asyncio.run(node(state))

        assert state["investment_debate_state"]["count"] == 1
        assert len(state["investment_debate_state"]["claims"]) == 3

    def test_bear_v2_opening_with_resolved_or_unresolved_claim_ids_rejected(self):
        """Opening response with resolved_claim_ids or unresolved_claim_ids is rejected fail-closed."""
        state = _make_bear_opening_state()
        bad_opening = (
            "空头立论发言（违规包含resolved_claim_ids）\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": [], "new_claims": ['
            '{"claim": "应收账款恶化现金流承压", "evidence": ["经营现金流同比下滑30%"], "confidence": 0.82, "battlefield": "fundamentals", "target_claim_ids": []}, '
            '{"claim": "外需降温出口面临逆风", "evidence": ["出口交货值同比下降"], "confidence": 0.75, "battlefield": "macro_policy", "target_claim_ids": []}, '
            '{"claim": "高位筹码松动获利盘兑现", "evidence": ["高位换手率超过25%"], "confidence": 0.70, "battlefield": "capital_flow", "target_claim_ids": []}'
            '], "resolved_claim_ids": ["INV-1"], "unresolved_claim_ids": [], "next_focus_claim_ids": [], '
            '"round_summary": "空头立论", "round_goal": "建立空头核心立论"} -->'
        )
        llm = FakeStreamingLLM([bad_opening, bad_opening])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        with pytest.raises(DebateProtocolError):
            asyncio.run(node(state))

        assert state["investment_debate_state"]["count"] == 1


class TestBearUnknownClaimFailClosed:
    """2. 任一命题引用字段若带有当前 claim ledger 不存在的 ID，必须 fail-closed。"""

    def test_bear_unknown_responded_claim_id_rejected_in_challenge_stage(self):
        """Unknown responded_claim_ids in Challenge stage is rejected and does not enter accepted ledger."""
        state = _make_bear_challenge_state()
        bad_challenge = (
            "空头盘问（含未知ID）\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1", "INV-999"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-1", "weakest_point": "假设存疑", "evidence": ["数据支撑"], "severity": "major"}], '
            '"self_win_prob": 0.75, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-1"], "next_focus_claim_ids": ["INV-1"], '
            '"round_summary": "盘问", "round_goal": "击穿多头"} -->'
        )
        llm = FakeStreamingLLM([bad_challenge, bad_challenge])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert exc_info.value.message_index == 4
        assert any("INV-999" in a["error_detail"] for a in exc_info.value.attempts)
        assert state["investment_debate_state"]["count"] == 3
        assert len(state["investment_debate_state"]["challenges"]) == 1

    def test_bear_unknown_challenge_target_rejected_in_challenge_stage(self):
        """Unknown challenge target_claim_id is rejected fail-closed."""
        state = _make_bear_challenge_state()
        bad_challenge = (
            "空头盘问（未知challenge目标）\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-999"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-999", "weakest_point": "假设存疑", "evidence": ["数据支撑"], "severity": "major"}], '
            '"self_win_prob": 0.75, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-1"], "next_focus_claim_ids": ["INV-1"], '
            '"round_summary": "盘问", "round_goal": "击穿多头"} -->'
        )
        llm = FakeStreamingLLM([bad_challenge, bad_challenge])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert exc_info.value.message_index == 4
        assert any("INV-999" in a["error_detail"] for a in exc_info.value.attempts)
        assert state["investment_debate_state"]["count"] == 3
        assert len(state["investment_debate_state"]["challenges"]) == 1

    def test_bear_unknown_target_claim_id_rejected_in_tiebreak_stage(self):
        """Unknown target_claim_ids in new_claims during Tiebreak stage is rejected fail-closed."""
        state = _make_bear_tiebreak_state()

        bad_tiebreak = (
            "空头收官（未知target_claim_ids）\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1"], "new_claims": ['
            '{"claim": "宏观基本面双杀风险依然极高", "evidence": ["新证据行业库存周期主动去化"], "confidence": 0.86, "target_claim_ids": ["INV-999"]}'
            '], "self_win_prob": 0.80, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-1"], "next_focus_claim_ids": ["INV-1"], '
            '"round_summary": "收官", "round_goal": "决胜"} -->'
        )
        llm = FakeStreamingLLM([bad_tiebreak, bad_tiebreak])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert exc_info.value.message_index == 6
        assert any("INV-999" in a["error_detail"] for a in exc_info.value.attempts)
        assert state["investment_debate_state"]["count"] == 5

    def test_bear_unknown_resolved_or_unresolved_claim_id_rejected(self):
        """Unknown claim IDs in resolved_claim_ids or unresolved_claim_ids are rejected fail-closed."""
        state = _make_bear_challenge_state()
        bad_resolved = (
            "空头盘问（含未知resolved_claim_ids）\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-1", "weakest_point": "假设存疑", "evidence": ["数据支撑"], "severity": "major"}], '
            '"self_win_prob": 0.75, "resolved_claim_ids": ["INV-888"], "unresolved_claim_ids": ["INV-1"], "next_focus_claim_ids": ["INV-1"], '
            '"round_summary": "盘问", "round_goal": "击穿多头"} -->'
        )
        llm = FakeStreamingLLM([bad_resolved, bad_resolved])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert any("INV-888" in a["error_detail"] for a in exc_info.value.attempts)

    def test_bear_retry_recovery_from_unknown_claim_id(self):
        """Attempt 1 with unknown claim ID fails, attempt 2 with valid claim ID succeeds; unknown ID never enters state."""
        state = _make_bear_challenge_state()
        bad_challenge = (
            "空头盘问（首次坏块含未知ID）\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1", "INV-999"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-1", "weakest_point": "假设存疑", "evidence": ["数据支撑"], "severity": "major"}], '
            '"self_win_prob": 0.75, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-1"], "next_focus_claim_ids": ["INV-1"], '
            '"round_summary": "盘问", "round_goal": "击穿多头"} -->'
        )
        good_challenge = (
            "空头盘问（修正后仅引用合法INV-1）\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-1", "weakest_point": "主力净流入缺乏持续性，尾盘集中流出", "evidence": ["尾盘大单净卖出8000万"], "severity": "major"}], '
            '"self_win_prob": 0.72, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-1"], "next_focus_claim_ids": ["INV-1"], '
            '"round_summary": "盘问多头主力资金漏洞", "round_goal": "击穿多头核心立论"} -->'
        )
        llm = FakeStreamingLLM([bad_challenge, good_challenge])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        res = asyncio.run(node(state))

        inv_state = res["investment_debate_state"]
        # count advances by exactly 1 (3 -> 4)
        assert inv_state["count"] == 4
        # Exactly 1 new accepted message
        accepted_msgs = [m for m in inv_state["round_messages"] if m.get("accepted") is True]
        assert len(accepted_msgs) == 4
        latest_msg = inv_state["round_messages"][-1]
        assert latest_msg["accepted"] is True
        assert latest_msg["responded_claim_ids"] == ["INV-1"]
        assert "INV-999" not in latest_msg["responded_claim_ids"]
        # Attempts trace tracks attempt 1 failure and attempt 2 success
        assert len(inv_state["attempts"]) == 2
        assert inv_state["attempts"][0]["accepted"] is False
        assert "INV-999" in inv_state["attempts"][0]["error_detail"]
        assert inv_state["attempts"][1]["accepted"] is True
        # challenges ledger only contains valid CH-2 targeting INV-1
        assert len(inv_state["challenges"]) == 2
        assert inv_state["challenges"][-1]["target_claim_id"] == "INV-1"

    def test_bear_consecutive_invalid_attempts_raises_debate_protocol_error(self):
        """Consecutive invalid attempts with unknown claim ID raise DebateProtocolError, preventing state mutation."""
        state = _make_bear_challenge_state()
        bad_challenge = (
            "空头盘问坏块\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-999"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-999", "weakest_point": "存疑", "evidence": ["无"], "severity": "major"}], '
            '"self_win_prob": 0.75, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-1"], "next_focus_claim_ids": ["INV-1"], '
            '"round_summary": "盘问", "round_goal": "击穿多头"} -->'
        )
        llm = FakeStreamingLLM([bad_challenge, bad_challenge])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        with pytest.raises(DebateProtocolError) as exc_info:
            asyncio.run(node(state))

        assert exc_info.value.message_index == 4
        assert exc_info.value.speaker == "Bear Analyst"
        assert len(exc_info.value.attempts) == 2
        assert all(not a["accepted"] for a in exc_info.value.attempts)
        assert state["investment_debate_state"]["count"] == 3
        assert len(state["investment_debate_state"]["challenges"]) == 1


class TestBearValidOpponentClaimAllowed:
    """3. 已存在的合法对手 Bull claim 在非 Opening 阶段仍可引用。"""

    def test_bear_valid_opponent_claim_accepted_in_challenge_stage(self):
        """In Challenge stage (message 4), referencing legal opponent claim INV-1 is accepted."""
        state = _make_bear_challenge_state()
        valid_challenge = (
            "空头合规盘问多头INV-1\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1"], "new_claims": [], '
            '"challenges": [{"target_claim_id": "INV-1", "weakest_point": "主力大单呈现假拉真出净流出隐蔽特征", "evidence": ["尾盘超大单集中抛售8000万"], "severity": "major"}], '
            '"self_win_prob": 0.72, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-1"], "next_focus_claim_ids": ["INV-1"], '
            '"round_summary": "击穿主力流入论点", "round_goal": "击穿多头立论"} -->'
        )
        llm = FakeStreamingLLM([valid_challenge])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        res = asyncio.run(node(state))

        inv_state = res["investment_debate_state"]
        assert inv_state["count"] == 4
        msg = inv_state["round_messages"][-1]
        assert msg["accepted"] is True
        assert msg["stage"] == "challenge"
        assert msg["responded_claim_ids"] == ["INV-1"]
        assert len(inv_state["challenges"]) == 2
        assert inv_state["challenges"][-1]["target_claim_id"] == "INV-1"

    def test_bear_valid_opponent_claim_accepted_in_tiebreak_stage(self):
        """In Tiebreak stage (message 6), referencing legal opponent claim INV-1 in target_claim_ids is accepted."""
        state = _make_bear_tiebreak_state()

        valid_tiebreak = (
            "空头合规决胜收官\n"
            '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1"], "new_claims": ['
            '{"claim": "基本面与宏观周期双重下行难以逆转", "evidence": ["行业库存周期仍处主动去库阶段"], "confidence": 0.85, "target_claim_ids": ["INV-1"]}'
            '], "self_win_prob": 0.78, "resolved_claim_ids": [], "unresolved_claim_ids": ["INV-1"], "next_focus_claim_ids": ["INV-1"], '
            '"round_summary": "决胜收官", "round_goal": "空头决胜"} -->'
        )
        llm = FakeStreamingLLM([valid_tiebreak])
        memory = MagicMock()
        memory.get_memories.return_value = []

        node = create_bear_researcher(llm, memory)
        res = asyncio.run(node(state))

        new_inv_state = res["investment_debate_state"]
        assert new_inv_state["count"] == 6
        msg = new_inv_state["round_messages"][-1]
        assert msg["accepted"] is True
        assert msg["stage"] == "tiebreak"
        assert msg["responded_claim_ids"] == ["INV-1"]
        assert msg["target_claim_ids"] == ["INV-1"]


class TestClaimReviewContractConsumers:
    """E-03c: Tests for evidence_verifier and decision_status consumer contracts."""

    def test_consumer_claim_with_valid_applicability_and_conditions_adopted(self):
        """Valid applicability and invalidation conditions pass consumer validation and claim is adopted."""
        from tradingagents.agents.utils.evidence_verifier import (
            aggregate_claim_evidence,
            DECISION_ADOPT,
        )

        claim = {
            "claim_id": "CLM-001",
            "speaker": "Bull Analyst",
            "speaker_key": "Bull",
            "claim": "主力资金净流入1.2亿",
            "evidence": ["主力净流入1.2亿"],
            "applicability": {
                "symbol": "600519",
                "horizon": "short",
                "metric_basis": "vendor_qfq",
                "pit_date": "2026-09-08",
                "preconditions": ["above_ma20"],
            },
            "invalidation_conditions": [
                {
                    "condition_id": "inv-1",
                    "metric": "close_price",
                    "operator": "<",
                    "threshold": 1600.0,
                    "unit": "cny",
                    "period": "1d_close",
                    "source": "daily_price",
                    "pit_date": "2026-09-08",
                }
            ],
        }
        ver_items = [
            {
                "claim_id": "CLM-001",
                "raw": "主力净流入1.2亿",
                "status": "verified",
                "matched_role": "smart_money_report",
                "matched_source": "smart_money",
            }
        ]
        summary = aggregate_claim_evidence(
            claims=[claim],
            claims_verification=ver_items,
            analysis_baseline_date="2026-09-08",
            expected_symbol="600519",
        )
        assert "CLM-001" in summary
        s = summary["CLM-001"]
        assert s["decision"] == DECISION_ADOPT
        assert s["applicability"] is not None
        assert s["applicability"]["symbol"] == "600519"
        assert len(s["invalidation_conditions"]) == 1
        assert s["invalidation_conditions"][0]["condition_id"] == "inv-1"
        assert s["pit_failed"] is False
        assert "全部证据核验通过" in s["reason"]

    def test_consumer_claim_applicability_pit_lookahead_fails_closed(self):
        """Claim applicability with PIT date later than baseline date fails closed (reject)."""
        from tradingagents.agents.utils.evidence_verifier import (
            aggregate_claim_evidence,
            DECISION_REJECT,
        )

        claim = {
            "claim_id": "CLM-002",
            "claim": "主力资金净流入1.2亿",
            "evidence": ["主力净流入1.2亿"],
            "applicability": {
                "symbol": "600519",
                "horizon": "short",
                "metric_basis": "vendor_qfq",
                "pit_date": "2026-09-15",  # Future PIT date
            },
        }
        ver_items = [
            {
                "claim_id": "CLM-002",
                "raw": "主力净流入1.2亿",
                "status": "verified",
            }
        ]
        summary = aggregate_claim_evidence(
            claims=[claim],
            claims_verification=ver_items,
            analysis_baseline_date="2026-09-08",
            expected_symbol="600519",
        )
        s = summary["CLM-002"]
        assert s["decision"] == DECISION_REJECT
        assert s["pit_failed"] is True
        assert "前视偏差/PIT失败" in s["reason"]
        assert s["counts"]["contradicted"] >= 1

    def test_consumer_claim_invalidation_condition_pit_lookahead_fails_closed(self):
        """Invalidation condition with future PIT date fails closed."""
        from tradingagents.agents.utils.evidence_verifier import (
            aggregate_claim_evidence,
            DECISION_REJECT,
        )

        claim = {
            "claim_id": "CLM-003",
            "claim": "主力资金净流入1.2亿",
            "evidence": ["主力净流入1.2亿"],
            "invalidation_conditions": [
                {
                    "condition_id": "inv-2",
                    "metric": "close_price",
                    "operator": "<",
                    "threshold": 1500.0,
                    "unit": "cny",
                    "period": "1d_close",
                    "source": "daily_price",
                    "pit_date": "2026-09-20",  # Future
                }
            ],
        }
        ver_items = [
            {
                "claim_id": "CLM-003",
                "raw": "主力净流入1.2亿",
                "status": "verified",
            }
        ]
        summary = aggregate_claim_evidence(
            claims=[claim],
            claims_verification=ver_items,
            analysis_baseline_date="2026-09-08",
        )
        s = summary["CLM-003"]
        assert s["decision"] == DECISION_REJECT
        assert s["pit_failed"] is True
        assert "前视偏差/PIT失败" in s["reason"]

    def test_consumer_claim_invalidation_condition_triggered_rejects_claim(self):
        """When an invalidation condition is triggered by market data, the claim is falsified and rejected."""
        from tradingagents.agents.utils.evidence_verifier import (
            aggregate_claim_evidence,
            DECISION_REJECT,
        )

        claim = {
            "claim_id": "CLM-004",
            "claim": "股价保持在1600元上方",
            "evidence": ["当前收盘价1620元"],
            "invalidation_conditions": [
                {
                    "condition_id": "inv-break-1600",
                    "metric": "close_price",
                    "operator": "<",
                    "threshold": 1600.0,
                    "unit": "cny",
                    "period": "1d_close",
                    "source": "daily_price",
                    "pit_date": "2026-09-08",
                }
            ],
        }
        ver_items = [
            {
                "claim_id": "CLM-004",
                "raw": "当前收盘价1620元",
                "status": "verified",
            }
        ]
        market_ctx = {
            "symbol": "600519",
            "close_price": 1580.0,
            "trade_date": "2026-09-08",
        }
        summary = aggregate_claim_evidence(
            claims=[claim],
            claims_verification=ver_items,
            market_data_context=market_ctx,
            analysis_baseline_date="2026-09-08",
        )
        s = summary["CLM-004"]
        assert s["decision"] == DECISION_REJECT
        assert "失效条件已触发证伪" in s["reason"]
        assert s["counts"]["contradicted"] >= 1

    def test_consumer_claim_valid_zero_and_decimal_zero_threshold(self):
        """Conditions with threshold=0 or Decimal(0) are accepted as valid numbers, not missing."""
        from decimal import Decimal
        from tradingagents.agents.utils.evidence_verifier import (
            aggregate_claim_evidence,
            DECISION_ADOPT,
        )

        claim = {
            "claim_id": "CLM-005",
            "claim": "净利润保持正增长",
            "evidence": ["净利润增速为0%"],
            "invalidation_conditions": [
                {
                    "condition_id": "inv-zero",
                    "metric": "profit_growth",
                    "operator": "<",
                    "threshold": Decimal("0.0"),
                    "unit": "pct",
                    "period": "quarterly",
                    "source": "financial_report",
                    "pit_date": "2026-09-08",
                }
            ],
        }
        ver_items = [
            {
                "claim_id": "CLM-005",
                "raw": "净利润增速为0%",
                "status": "verified",
            }
        ]
        market_ctx = {
            "profit_growth": 0.0,
            "trade_date": "2026-09-08",
        }
        summary = aggregate_claim_evidence(
            claims=[claim],
            claims_verification=ver_items,
            market_data_context=market_ctx,
            analysis_baseline_date="2026-09-08",
        )
        s = summary["CLM-005"]
        assert s["decision"] == DECISION_ADOPT
        assert len(s["invalidation_conditions"]) == 1
        assert s["invalidation_conditions"][0]["threshold"] == 0.0

    def test_consumer_observation_hypothesis_claim_contract_not_upgraded_to_adopt(self):
        """Observation / hypothesis claim review contract stays in audited partial state and cannot be adopted."""
        from tradingagents.agents.utils.evidence_verifier import (
            aggregate_claim_evidence,
            DECISION_PARTIAL,
            DECISION_ADOPT,
        )

        obs_claim = {
            "claim_id": "CLM-OBS-1",
            "claim": "【观察】当前处于Wyckoff吸筹阶段，主力持续试盘",
            "claim_type": "observation",
            "evidence": ["缩量回踩不破前低"],
            "applicability": {
                "symbol": "600519",
                "horizon": "short",
                "metric_basis": "vendor_qfq",
                "pit_date": "2026-09-08",
            },
        }
        ver_items = [
            {
                "claim_id": "CLM-OBS-1",
                "raw": "缩量回踩不破前低",
                "status": "verified",
            }
        ]
        summary = aggregate_claim_evidence(
            claims=[obs_claim],
            claims_verification=ver_items,
            analysis_baseline_date="2026-09-08",
        )
        s = summary["CLM-OBS-1"]
        assert s["is_observation_or_hypothesis"] is True
        assert s["decision"] == DECISION_PARTIAL
        assert s["decision"] != DECISION_ADOPT
        assert "不升级为已验证事实" in s["reason"]
