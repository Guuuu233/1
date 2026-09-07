"""Tests for H-04d: Research Manager consumes research horizon.

Covers:
1. Research horizon resolution priority:
   state["horizon"] -> metadata resolved[0] -> requested[0] -> get_bound_research_horizon() -> "short"
2. Prompt visibility and horizon decoupling:
   - Medium research horizon properly injected in prompt
   - Analyst traces with observation_horizon='short' must NOT overwrite research horizon or treat run as short
   - Short research horizon properly injected in prompt
   - English prompt localization
3. Short vs Medium prompt criteria:
   - Core questions / validity period / invalidation conditions / action basis
4. Machine-readable manager_verdict fields:
   - horizon and research_horizon set correctly
5. Blocked / Fail-closed paths:
   - Explicit INVALID / ABSTAIN preserves horizon without fabricating direction
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

from tradingagents.agents.managers.research_manager import (
    create_research_manager,
    _resolve_research_horizon,
)
from tradingagents.graph.intent_parser import (
    bind_research_horizon,
    clear_research_horizon,
    get_bound_research_horizon,
)
from tradingagents.prompts import get_prompt
from tradingagents.prompts.catalog import ZH_PROMPTS, EN_PROMPTS


from tests.test_research_manager_seven_reports_and_verdict_gate import _make_seven_reports_state


@pytest.fixture(autouse=True)
def clean_horizon_binding():
    """Ensure contextvar binding is always clean before and after tests."""
    clear_research_horizon()
    yield
    clear_research_horizon()


async def _fake_stream(text: str):
    yield SimpleNamespace(content=text)


def _make_base_state(horizon: str = "medium", overrides: dict | None = None) -> dict:
    state_overrides = {
        "horizon": horizon,
        "analyst_traces": [
            {
                "agent": "volume_price_analyst",
                "horizon": horizon,
                "research_horizon": horizon,
                "observation_horizon": "short",
                "data_window": "14天",
                "verdict": "看多",
            },
            {
                "agent": "market_analyst",
                "horizon": horizon,
                "research_horizon": horizon,
                "observation_horizon": "short",
                "data_window": "30天",
                "verdict": "偏多",
            },
        ],
    }
    if overrides:
        state_overrides.update(overrides)
    return _make_seven_reports_state(state_overrides)


SAMPLE_LLM_RESPONSE = """
正式投研裁决报告：
经过对宏观板块、技术面、情绪、新闻、基本面、主力资金与量价七大视角的交叉审查，本轮多头立论充分。

1. 各分析师 Verdict 全景概览与动态加权：
1) 宏观板块：看多，权重 15%
2) 市场（技术面）：偏多，权重 10%
3) 舆情（情绪）：中性，权重 10%
4) 新闻：偏多，权重 15%
5) 基本面：看多，权重 25%
6) 主力资金：偏多，权重 15%
7) 量价：偏多，权重 10%

建议仓位：35%，止损位：18.50元。

<!-- MANAGER_VERDICT: {"winner": "bull", "direction": "偏多", "reason": "基本面营收同比增长30%获得核验且多头胜出", "position_pct": 35, "entry": "现价", "target": "25.0", "stop_loss": "18.50", "upside": 15.0, "downside": 5.0, "odds": 3.0, "adopted_claim_ids": ["INV-1"], "partially_adopted_claims": [], "rejected_claim_ids": [], "excluded_evidence": [], "dispute_map": []} -->
<!-- VERDICT: {"direction": "偏多", "reason": "基本面扎实且趋势向上"} -->
""".strip()


class TestResearchHorizonResolution:
    """Test Contract 1: Research horizon resolution priority."""

    def test_state_horizon_highest_priority(self):
        state = {
            "horizon": "medium",
            "horizon_run_metadata": {"resolved": ["short"], "requested": ["short"]},
        }
        bind_research_horizon("short")
        assert _resolve_research_horizon(state) == "medium"

    def test_metadata_resolved_priority_over_requested(self):
        state = {
            "horizon_run_metadata": {"resolved": ["medium"], "requested": ["short"]},
        }
        bind_research_horizon("short")
        assert _resolve_research_horizon(state) == "medium"

    def test_metadata_requested_priority_over_binding(self):
        state = {
            "horizon_run_metadata": {"requested": ["medium"]},
        }
        bind_research_horizon("short")
        assert _resolve_research_horizon(state) == "medium"

    def test_binding_priority_over_default(self):
        bind_research_horizon("medium")
        state = {}
        assert _resolve_research_horizon(state) == "medium"
        assert _resolve_research_horizon(None) == "medium"

    def test_unbound_fallback_short(self):
        clear_research_horizon()
        assert _resolve_research_horizon({}) == "short"
        assert _resolve_research_horizon(None) == "short"


class TestResearchManagerPromptHorizonInjection:
    """Test Contract 2 & 3: Prompt injection and decoupling from observation windows."""

    def test_medium_research_horizon_injected_in_prompt(self):
        captured_prompts = []

        async def _astream(prompt):
            captured_prompts.append(prompt)
            yield SimpleNamespace(content=SAMPLE_LLM_RESPONSE)

        mock_llm = MagicMock()
        mock_llm.astream = _astream
        memory = MagicMock()
        memory.get_memories.return_value = []

        state = _make_base_state(horizon="medium")
        node = create_research_manager(mock_llm, memory)
        asyncio.run(node(state))

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # 1. Medium research horizon must be explicitly visible
        assert "本次研究档：中线（1-3月，基本面主导）" in prompt
        assert "本节点专业观察窗：中线（1-3月，基本面主导）" in prompt

        # 2. Must NOT claim that the entire task/run is short-term
        assert "本次研究档：短线" not in prompt
        assert "当前分析维度" not in prompt

    def test_observation_window_short_does_not_override_medium_research_horizon(self):
        """Even if all analyst traces have observation_horizon='short', the research manager
        prompt must remain medium research horizon and NOT be degraded to short.
        """
        captured_prompts = []

        async def _astream(prompt):
            captured_prompts.append(prompt)
            yield SimpleNamespace(content=SAMPLE_LLM_RESPONSE)

        mock_llm = MagicMock()
        mock_llm.astream = _astream
        memory = MagicMock()
        memory.get_memories.return_value = []

        # State has analyst traces with observation_horizon="short"
        state = _make_base_state(
            horizon="medium",
            overrides={
                "analyst_traces": [
                    {
                        "agent": "volume_price_analyst",
                        "horizon": "medium",
                        "research_horizon": "medium",
                        "observation_horizon": "short",
                        "verdict": "看多",
                    },
                    {
                        "agent": "smart_money_analyst",
                        "horizon": "medium",
                        "research_horizon": "medium",
                        "observation_horizon": "short",
                        "verdict": "看多",
                    },
                    {
                        "agent": "market_analyst",
                        "horizon": "medium",
                        "research_horizon": "medium",
                        "observation_horizon": "short",
                        "verdict": "偏多",
                    },
                ]
            },
        )
        node = create_research_manager(mock_llm, memory)
        result = asyncio.run(node(state))

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # Prompt must clearly retain medium research horizon
        assert "本次研究档：中线（1-3月，基本面主导）" in prompt
        assert "本次研究档：短线" not in prompt
        assert "当前分析维度" not in prompt

        # manager_verdict must record medium
        assert result["manager_verdict"]["horizon"] == "medium"
        assert result["manager_verdict"]["research_horizon"] == "medium"

    def test_short_research_horizon_injected_in_prompt(self):
        captured_prompts = []

        async def _astream(prompt):
            captured_prompts.append(prompt)
            yield SimpleNamespace(content=SAMPLE_LLM_RESPONSE)

        mock_llm = MagicMock()
        mock_llm.astream = _astream
        memory = MagicMock()
        memory.get_memories.return_value = []

        state = _make_base_state(horizon="short")
        node = create_research_manager(mock_llm, memory)
        result = asyncio.run(node(state))

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # Short research horizon must be visible
        assert "本次研究档：短线（1-2周，技术面主导）" in prompt
        assert "本节点专业观察窗：短线（1-2周，技术面主导）" in prompt
        assert "本次研究档：中线" not in prompt

        assert result["manager_verdict"]["horizon"] == "short"
        assert result["manager_verdict"]["research_horizon"] == "short"

    def test_english_prompt_template_injection(self, monkeypatch):
        """English template properly displays research horizon."""
        from tradingagents.dataflows.config import get_config

        current_cfg = get_config()
        monkeypatch.setattr(
            "tradingagents.agents.managers.research_manager.get_config",
            lambda: {**current_cfg, "prompt_language": "en"},
        )
        monkeypatch.setattr(
            "tradingagents.graph.intent_parser.get_config",
            lambda: {**current_cfg, "prompt_language": "en"},
        )
        captured_prompts = []

        async def _astream(prompt):
            captured_prompts.append(prompt)
            yield SimpleNamespace(content=SAMPLE_LLM_RESPONSE)

        mock_llm = MagicMock()
        mock_llm.astream = _astream
        memory = MagicMock()
        memory.get_memories.return_value = []

        state = _make_base_state(horizon="medium")
        node = create_research_manager(mock_llm, memory)
        result = asyncio.run(node(state))

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "Research Horizon: Medium-term (1-3 months, fundamentals-driven)" in prompt
        assert "Node Observation Window: Medium-term (1-3 months, fundamentals-driven)" in prompt
        assert result["manager_verdict"]["horizon"] == "medium"
        assert result["manager_verdict"]["research_horizon"] == "medium"


class TestResearchManagerPromptCriteria:
    """Test Contract 3: Prompt distinguishes core questions, validity period, invalidation, and action basis."""

    def test_zh_prompt_distinguishes_short_vs_medium_criteria(self):
        prompt = ZH_PROMPTS["research_manager_prompt"]

        # 1. Short-term criteria
        assert "短线视角" in prompt
        assert "核心问题" in prompt
        assert "有效期限" in prompt
        assert "失效条件" in prompt
        assert "行动依据" in prompt
        assert "近窗交易结构" in prompt or "近窗" in prompt

        # 2. Medium-term criteria
        assert "中线视角" in prompt
        assert "传导与持有条件" in prompt or "持有条件" in prompt
        assert "产业链传导" in prompt or "传导路径" in prompt
        assert "不得替代短线交易结构裁决" in prompt or "不得" in prompt

    def test_en_prompt_distinguishes_short_vs_medium_criteria(self):
        prompt = EN_PROMPTS["research_manager_prompt"]

        assert "Short-term research horizon" in prompt
        assert "Medium-term research horizon" in prompt
        assert "Core questions" in prompt
        assert "Validity period" in prompt
        assert "Invalidation conditions" in prompt
        assert "Action basis" in prompt


class TestResearchManagerVerdictFieldsAndBlockedPaths:
    """Test Contract 4 & 5: manager_verdict fields and fail-closed paths."""

    def test_successful_verdict_contains_horizon_fields(self):
        mock_llm = MagicMock()
        mock_llm.astream = lambda prompt: _fake_stream(SAMPLE_LLM_RESPONSE)
        memory = MagicMock()
        memory.get_memories.return_value = []

        state = _make_base_state(horizon="medium")
        node = create_research_manager(mock_llm, memory)
        result = asyncio.run(node(state))

        assert result["manager_verdict"]["horizon"] == "medium"
        assert result["manager_verdict"]["research_horizon"] == "medium"
        assert result["investment_debate_state"]["manager_verdict"]["horizon"] == "medium"
        assert result["investment_debate_state"]["manager_verdict"]["research_horizon"] == "medium"

    def test_run_integrity_blocked_path_preserves_horizon(self):
        """When run integrity blocks (all required analysts failed), horizon is preserved."""
        mock_llm = MagicMock()
        mock_llm.astream = lambda prompt: _fake_stream(SAMPLE_LLM_RESPONSE)
        memory = MagicMock()
        memory.get_memories.return_value = []

        # Make all required reports degraded/failed
        fail_text = "分析报告生成失败：API timeout"
        state = _make_base_state(
            horizon="medium",
            overrides={
                "macro_report": fail_text,
                "market_report": fail_text,
                "sentiment_report": fail_text,
                "news_report": fail_text,
                "fundamentals_report": fail_text,
                "smart_money_report": fail_text,
                "volume_price_report": fail_text,
            },
        )
        node = create_research_manager(mock_llm, memory)
        result = asyncio.run(node(state))

        assert result["manager_verdict"]["horizon"] == "medium"
        assert result["manager_verdict"]["research_horizon"] == "medium"
        assert result["manager_verdict"]["direction"] == "N/A"
        assert result["trade_action"] == "NO_TRADE"
        assert result["decision_status"]["trade_action"] == "NO_TRADE"

    def test_fund_flow_guard_blocked_path_preserves_horizon(self):
        """When fund flow guard blocks, horizon is preserved and direction is abstained."""
        mock_llm = MagicMock()
        mock_llm.astream = lambda prompt: _fake_stream(SAMPLE_LLM_RESPONSE)
        memory = MagicMock()
        memory.get_memories.return_value = []

        state = _make_base_state(
            horizon="medium",
            overrides={
                "fund_flow_consensus_guard": {
                    "blocked": True,
                    "direction_allowed": False,
                    "status": "contradicted",
                },
            },
        )
        node = create_research_manager(mock_llm, memory)
        result = asyncio.run(node(state))

        assert result["manager_verdict"]["horizon"] == "medium"
        assert result["manager_verdict"]["research_horizon"] == "medium"
        assert result["decision_status"]["analysis_status"] == "ABSTAIN"
        assert result["decision_status"]["trade_action"] == "NO_TRADE"

    def test_debate_pregate_blocked_path_preserves_horizon(self):
        """When debate pre-gate checks fail, horizon is preserved and status is ABSTAIN."""
        mock_llm = MagicMock()
        mock_llm.astream = lambda prompt: _fake_stream(SAMPLE_LLM_RESPONSE)
        memory = MagicMock()
        memory.get_memories.return_value = []

        # Empty round messages causes pre-gate validation failure
        state = _make_base_state(
            horizon="medium",
            overrides={
                "investment_debate_state": {
                    "history": "",
                    "count": 0,
                    "claims": [],
                    "round_messages": [],
                },
            },
        )
        node = create_research_manager(mock_llm, memory)
        result = asyncio.run(node(state))

        assert result["manager_verdict"]["horizon"] == "medium"
        assert result["manager_verdict"]["research_horizon"] == "medium"
        assert result["decision_status"]["analysis_status"] == "ABSTAIN"
        assert result["decision_status"]["trade_action"] == "NO_TRADE"
