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
