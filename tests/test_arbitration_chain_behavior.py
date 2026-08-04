"""Behavior-level tests for the risk-arbitration chain (裁决链).

Covers the post-trader adjudication sub-graph with frozen (golden) LLM
outputs, following the same approach as ``test_evidence_citation_density.py``:
drive the real graph nodes with mock LLMs, then assert on the wiring and on
what the arbitration stages actually reference (claims, evidence, constraints).

DAV-75 H3 §5.4-2: 裁决链行为级测试 — trading_graph.py / debator 覆盖缺口.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.reflection import Reflector
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.managers.risk_manager import create_risk_manager
from tradingagents.agents.utils.debate_utils import (
    build_empty_risk_debate_state,
    extract_risk_judge_result,
)


# ---------------------------------------------------------------------------
# Golden (frozen) LLM outputs
# ---------------------------------------------------------------------------

AGGRESSIVE_GOLDEN = """本轮风险焦点：仓位过重。最大风险是回调击穿止损导致回撤超限。
<!-- RISK_STATE: {"responded_claim_ids": [], "new_claims": [{"claim": "回调击穿止损导致回撤超限", "evidence": ["RSI 48.2", "止损 1780"], "confidence": 0.8}], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], "round_summary": "激进方提出止损回撤风险", "round_goal": "评估止损位的合理性"} -->
"""

CONSERVATIVE_GOLDEN = """本轮风险焦点：逆势追高。保守方要求更紧的止损。
<!-- RISK_STATE: {"responded_claim_ids": [], "new_claims": [{"claim": "追高后回落风险", "evidence": ["RSI 48.2"], "confidence": 0.7}], "resolved_claim_ids": [], "unresolved_claim_ids": ["RISK-1"], "next_focus_claim_ids": [], "round_summary": "保守方补充追高回落风险", "round_goal": "统一止损纪律"} -->
"""

NEUTRAL_GOLDEN = """本轮风险焦点：风险收益比。中性方认为当前止损位可接受。
<!-- RISK_STATE: {"responded_claim_ids": ["RISK-1"], "new_claims": [], "resolved_claim_ids": ["RISK-1"], "unresolved_claim_ids": [], "next_focus_claim_ids": [], "round_summary": "中性方确认止损位合理", "round_goal": "收口"} -->
"""

RISK_JUDGE_PASS_GOLDEN = """风控通过。硬约束：仓位≤20%，止损 1780。前提：RSI 48.2 不破位。触发：跌破 1780。
<!-- RISK_JUDGE: {"verdict": "pass", "revision_reason": "", "hard_constraints": ["仓位≤20%"], "soft_constraints": [], "execution_preconditions": ["RSI 48.2 不破位"], "de_risk_triggers": ["跌破 1780"]} -->
"""

RISK_JUDGE_REVISE_GOLDEN = """风控打回。仓位过高，需降至 15%。
<!-- RISK_JUDGE: {"verdict": "revise", "revision_reason": "仓位超过硬约束上限", "hard_constraints": ["仓位≤15%"], "soft_constraints": [], "execution_preconditions": [], "de_risk_triggers": ["跌破 1780"]} -->
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_llm(golden_output: str, captured: list | None = None) -> MagicMock:
    """LLM whose astream yields one golden chunk and records the prompt."""
    captured = captured if captured is not None else []

    async def _astream(prompt, **kwargs):
        captured.append(prompt)
        yield MagicMock(content=golden_output)

    llm = MagicMock()
    llm.astream = _astream
    return llm


def _state(**overrides) -> dict:
    base = {
        "company_of_interest": "600519",
        "market_report": "市场 RSI 48.2",
        "sentiment_report": "情绪中性",
        "news_report": "新闻利好",
        "fundamentals_report": "基本面 +15%",
        "trader_investment_plan": "交易员方案：买入 20% 仓位",
        "risk_debate_state": build_empty_risk_debate_state(),
        "risk_feedback_state": {
            "retry_count": 0,
            "max_retries": 1,
            "revision_required": False,
            "latest_risk_verdict": "",
            "hard_constraints": [],
            "soft_constraints": [],
            "execution_preconditions": [],
            "de_risk_triggers": [],
            "revision_reason": "",
        },
    }
    base.update(overrides)
    return base


def _message(tool_calls=None):
    """Minimal stand-in for a LangChain message carrying optional tool_calls."""
    msg = MagicMock()
    if tool_calls is None:
        msg.tool_calls = None
    else:
        msg.tool_calls = tool_calls
    return msg


# ---------------------------------------------------------------------------
# ConditionalLogic routing (chain assembly decisions)
# ---------------------------------------------------------------------------

class TestConditionalLogicRouting:
    def test_analyst_tool_call_continues(self):
        logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
        for method in ("market", "social", "news", "fundamentals", "macro", "smart_money", "volume_price"):
            state = {"messages": [_message(tool_calls=[{"name": "get_stock_data"}])]}
            assert getattr(logic, f"should_continue_{method}")(state) == "continue"

    def test_analyst_no_tool_call_done(self):
        logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
        state = {"messages": [_message()]}
        assert logic.should_continue_market(state) == "done"

    def test_invest_debate_rounds_cap_routes_to_manager(self):
        logic = ConditionalLogic(max_debate_rounds=1)
        state = {"investment_debate_state": {"count": 2, "current_speaker": "Bull"}}
        assert logic.should_continue_debate(state) == "Research Manager"

    def test_invest_debate_switches_speaker(self):
        logic = ConditionalLogic(max_debate_rounds=3)
        state = {"investment_debate_state": {"count": 1, "current_speaker": "Bull"}}
        assert logic.should_continue_debate(state) == "Bear Researcher"
        state = {"investment_debate_state": {"count": 1, "current_speaker": "Bear"}}
        assert logic.should_continue_debate(state) == "Bull Researcher"

    def test_risk_analysis_rounds_cap_routes_to_judge(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=1)
        state = {"risk_debate_state": {"count": 3, "latest_speaker": "Aggressive"}}
        assert logic.should_continue_risk_analysis(state) == "Risk Judge"

    def test_risk_analysis_rotates_speakers(self):
        logic = ConditionalLogic(max_risk_discuss_rounds=3)
        assert logic.should_continue_risk_analysis(
            {"risk_debate_state": {"count": 0, "latest_speaker": "Aggressive"}}
        ) == "Conservative Analyst"
        assert logic.should_continue_risk_analysis(
            {"risk_debate_state": {"count": 0, "latest_speaker": "Conservative"}}
        ) == "Neutral Analyst"
        # Empty / unknown speaker starts the rotation at Aggressive.
        assert logic.should_continue_risk_analysis(
            {"risk_debate_state": {"count": 0, "latest_speaker": ""}}
        ) == "Aggressive Analyst"

    def test_revise_after_judge_respects_retry_budget(self):
        logic = ConditionalLogic()
        # revision required and budget not exhausted -> back to Trader
        assert logic.should_revise_after_risk_judge(
            {"risk_feedback_state": {"revision_required": True, "retry_count": 0, "max_retries": 1}}
        ) == "Trader"
        # budget exhausted -> END
        assert logic.should_revise_after_risk_judge(
            {"risk_feedback_state": {"revision_required": True, "retry_count": 2, "max_retries": 1}}
        ) == "END"
        # no revision required -> END
        assert logic.should_revise_after_risk_judge(
            {"risk_feedback_state": {"revision_required": False, "retry_count": 0, "max_retries": 1}}
        ) == "END"


# ---------------------------------------------------------------------------
# Debator nodes: prompt assembly + arbitration references
# ---------------------------------------------------------------------------

class TestDebatorPromptAssembly:
    @pytest.mark.parametrize("factory,name,label", [
        (create_aggressive_debator, "aggressive", "Aggressive"),
        (create_conservative_debator, "conservative", "Conservative"),
        (create_neutral_debator, "neutral", "Neutral"),
    ])
    def test_prompt_carries_trader_plan_and_reports(self, factory, name, label):
        captured: list = []
        node = factory(_fake_llm("纯文字回应，无机器块。", captured))
        asyncio.run(node(_state()))

        assert len(captured) == 1
        prompt = captured[0]
        # The adjudicator must see the trader plan and analyst reports.
        assert "交易员方案：买入 20% 仓位" in prompt
        assert "市场 RSI 48.2" in prompt
        assert "基本面 +15%" in prompt

    @pytest.mark.parametrize("factory,name,label", [
        (create_aggressive_debator, "aggressive", "Aggressive"),
        (create_conservative_debator, "conservative", "Conservative"),
        (create_neutral_debator, "neutral", "Neutral"),
    ])
    def test_unstructured_response_still_records_argument(self, factory, name, label):
        """A debator response without a machine block must still advance history
        and the round counter — never silently drop the argument."""
        captured: list = []
        node = factory(_fake_llm("我不同意，风险收益比不划算。", captured))
        result = asyncio.run(node(_state()))

        rds = result["risk_debate_state"]
        assert rds["count"] == 1
        assert rds["latest_speaker"] == label
        assert label in rds["history"]
        assert f"{label} Analyst" in rds["history"]


class TestDebatorArbitrationReferences:
    def test_aggressive_claims_carry_evidence(self):
        captured: list = []
        node = create_aggressive_debator(_fake_llm(AGGRESSIVE_GOLDEN, captured))
        result = asyncio.run(node(_state()))

        rds = result["risk_debate_state"]
        assert rds["latest_speaker"] == "Aggressive"
        assert rds["count"] == 1
        assert len(rds["claims"]) == 1
        claim = rds["claims"][0]
        assert claim["claim_id"] == "RISK-1"
        assert claim["speaker"] == "Aggressive Analyst"
        assert claim["stance"] == "aggressive"
        assert "RSI 48.2" in claim["evidence"]
        assert "止损 1780" in claim["evidence"]
        assert claim["confidence"] == pytest.approx(0.8)
        assert rds["open_claim_ids"] == ["RISK-1"]
        # The cleaned response still carries the human-facing argument.
        assert "回调击穿止损" in rds["current_aggressive_response"]

    def test_conservative_marks_aggressive_claim_unresolved(self):
        """The conservative round must see the aggressive claim in its prompt
        and carry the arbitration forward (unresolved id referenced)."""
        state = _state()
        captured: list = []

        # Round 1: aggressive registers RISK-1.
        agg_node = create_aggressive_debator(_fake_llm(AGGRESSIVE_GOLDEN))
        agg_result = asyncio.run(agg_node(state))
        state["risk_debate_state"] = agg_result["risk_debate_state"]

        # Round 2: conservative responds, marking RISK-1 unresolved.
        cons_node = create_conservative_debator(_fake_llm(CONSERVATIVE_GOLDEN, captured))
        cons_result = asyncio.run(cons_node(state))
        rds = cons_result["risk_debate_state"]

        # The conservative prompt referenced the previous claim id.
        assert "RISK-1" in captured[0]
        assert rds["latest_speaker"] == "Conservative"
        assert rds["unresolved_claim_ids"] == ["RISK-1"]
        assert any(c["claim_id"] == "RISK-2" for c in rds["claims"])

    def test_neutral_resolves_disputed_claim(self):
        """The neutral round resolves the previously-unresolved claim and the
        state reflects the resolution."""
        state = _state()

        # Build a two-round chain: aggressive -> conservative.
        for factory, golden in ((create_aggressive_debator, AGGRESSIVE_GOLDEN),
                                (create_conservative_debator, CONSERVATIVE_GOLDEN)):
            result = asyncio.run(factory(_fake_llm(golden))(state))
            state["risk_debate_state"] = result["risk_debate_state"]

        # Round 3: neutral resolves RISK-1.
        neutral_result = asyncio.run(create_neutral_debator(_fake_llm(NEUTRAL_GOLDEN))(state))
        rds = neutral_result["risk_debate_state"]

        assert rds["latest_speaker"] == "Neutral"
        assert rds["resolved_claim_ids"] == ["RISK-1"]
        assert "RISK-1" not in rds["unresolved_claim_ids"]
        assert rds["count"] == 3


# ---------------------------------------------------------------------------
# Reflector (post-run reflection updates memory)
# ---------------------------------------------------------------------------

class TestReflector:
    def _make_reflector(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="反思结论：控制仓位。")
        return Reflector(llm)

    def _state_for_reflection(self, **overrides):
        state = {
            "market_report": "市场 RSI 48.2",
            "sentiment_report": "情绪中性",
            "news_report": "新闻利好",
            "fundamentals_report": "基本面 +15%",
            "trader_investment_plan": "交易员方案：买入 20% 仓位",
            "investment_debate_state": {"bull_history": "Bull 看多", "bear_history": "Bear 看空",
                                        "judge_decision": "研究经理裁决：看多"},
            "risk_debate_state": {"judge_decision": "风控通过，仓位≤20%"},
        }
        state.update(overrides)
        return state

    @pytest.mark.parametrize("method,key", [
        ("reflect_bull_researcher", "bull_history"),
        ("reflect_bear_researcher", "bear_history"),
        ("reflect_trader", "trader_investment_plan"),
        ("reflect_invest_judge", "judge_decision"),
    ])
    def test_reflection_feeds_memory(self, method, key):
        reflector = self._make_reflector()
        memory = MagicMock()
        memory.add_situations = MagicMock()
        getattr(reflector, method)(self._state_for_reflection(), -0.03, memory)
        memory.add_situations.assert_called_once()
        # The memory entry pairs the extracted situation with the frozen reflection.
        situations, results = memory.add_situations.call_args[0][0][0]
        assert "RSI 48.2" in situations
        assert "控制仓位" in results

    def test_reflect_risk_manager(self):
        reflector = self._make_reflector()
        memory = MagicMock()
        memory.add_situations = MagicMock()
        reflector.reflect_risk_manager(self._state_for_reflection(), 0.02, memory)
        memory.add_situations.assert_called_once()
        situations, results = memory.add_situations.call_args[0][0][0]
        assert "RSI 48.2" in situations
        assert "控制仓位" in results


# ---------------------------------------------------------------------------
# Risk judge (arbitration terminal)
# ---------------------------------------------------------------------------

class TestRiskJudge:
    def test_pass_verdict_clears_revision_and_sets_constraints(self):
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])
        captured: list = []
        node = create_risk_manager(_fake_llm(RISK_JUDGE_PASS_GOLDEN, captured), memory)
        result = asyncio.run(node(_state()))

        assert captured, "risk manager must receive a prompt"
        feedback = result["risk_feedback_state"]
        assert feedback["latest_risk_verdict"] == "pass"
        assert feedback["revision_required"] is False
        assert feedback["hard_constraints"] == ["仓位≤20%"]
        assert feedback["de_risk_triggers"] == ["跌破 1780"]
        assert "风控通过" in result["final_trade_decision"]

    def test_revise_verdict_increments_retry_and_requests_rework(self):
        memory = MagicMock()
        memory.get_memories = MagicMock(return_value=[])
        captured: list = []
        node = create_risk_manager(_fake_llm(RISK_JUDGE_REVISE_GOLDEN, captured), memory)
        result = asyncio.run(node(_state()))

        feedback = result["risk_feedback_state"]
        assert feedback["latest_risk_verdict"] == "revise"
        assert feedback["revision_required"] is True
        assert feedback["retry_count"] == 1
        assert feedback["revision_reason"] == "仓位超过硬约束上限"
        assert feedback["hard_constraints"] == ["仓位≤15%"]

    def test_judge_payload_parser_rejects_bad_verdict(self):
        """An unparseable RISK_JUDGE block degrades to a reject verdict, never
        an accidental pass."""
        parsed = extract_risk_judge_result("完全没有机器块。")
        assert parsed["verdict"] == "reject"
        assert parsed["parse_failed"] is True
        assert "风控裁决机读块解析失败" in parsed["cleaned_response"]


# ---------------------------------------------------------------------------
# Chain assembly: the compiled graph wires the adjudication loop
# ---------------------------------------------------------------------------

class _FakeWorkflow:
    """Minimal StateGraph stand-in capturing nodes / edges for assertion."""

    def __init__(self, *_args, **_kwargs):
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []

    def add_node(self, name, node):
        self.nodes[name] = node

    def add_edge(self, source, target):
        if isinstance(source, list):
            for s in source:
                self.edges.append((s, target))
        else:
            self.edges.append((source, target))

    def add_conditional_edges(self, source, condition, mapping):
        self.conditional_edges.append((source, condition, mapping))

    def compile(self, checkpointer=None):
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "conditional_edges": self.conditional_edges,
            "checkpointer": checkpointer,
        }


def test_risk_chain_edges_are_wired():
    """The compiled graph must connect Trader -> Aggressive -> Conservative ->
    Neutral -> Risk Judge, with a revise loop back to Trader."""
    from tradingagents.graph.setup import GraphSetup

    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    factories = {
        "create_aggressive_debator": MagicMock(return_value="aggressive_node"),
        "create_bear_researcher": MagicMock(return_value="bear_node"),
        "create_bull_researcher": MagicMock(return_value="bull_node"),
        "create_conservative_debator": MagicMock(return_value="conservative_node"),
        "create_fundamentals_analyst": MagicMock(return_value="fundamentals_node"),
        "create_macro_analyst": MagicMock(return_value="macro_node"),
        "create_market_analyst": MagicMock(return_value="market_node"),
        "create_neutral_debator": MagicMock(return_value="neutral_node"),
        "create_news_analyst": MagicMock(return_value="news_node"),
        "create_research_manager": MagicMock(return_value="research_node"),
        "create_risk_manager": MagicMock(return_value="risk_node"),
        "create_smart_money_analyst": MagicMock(return_value="smart_money_node"),
        "create_social_media_analyst": MagicMock(return_value="social_node"),
        "create_trader": MagicMock(return_value="trader_node"),
    }

    with patch("tradingagents.graph.setup._load_agent_factories", return_value=factories), \
         patch("tradingagents.graph.setup.StateGraph", _FakeWorkflow):
        setup = GraphSetup(
            object(), object(), {"market": object()},
            bull_memory=object(), bear_memory=object(), trader_memory=object(),
            invest_judge_memory=object(), risk_manager_memory=object(),
            conditional_logic=conditional_logic, data_collector=object(),
        )
        compiled = setup.setup_graph(["market"])

    assert {"Trader", "Aggressive Analyst", "Conservative Analyst",
            "Neutral Analyst", "Risk Judge"} <= set(compiled["nodes"])
    # Trader hands off directly to the aggressive risk analyst.
    assert ("Trader", "Aggressive Analyst") in compiled["edges"]
    # The three risk analysts are looped by the shared should_continue_risk_analysis.
    risk_conds = {src: mapping for src, _cond, mapping in compiled["conditional_edges"]
                  if src in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst")}
    assert risk_conds["Aggressive Analyst"] == {"Conservative Analyst": "Conservative Analyst",
                                                "Risk Judge": "Risk Judge"}
    assert risk_conds["Conservative Analyst"] == {"Neutral Analyst": "Neutral Analyst",
                                                  "Risk Judge": "Risk Judge"}
    assert risk_conds["Neutral Analyst"] == {"Aggressive Analyst": "Aggressive Analyst",
                                             "Risk Judge": "Risk Judge"}
    # Risk judge either loops the trader back for revision or ends the run.
    judge_conds = [mapping for src, _cond, mapping in compiled["conditional_edges"]
                   if src == "Risk Judge"]
    assert len(judge_conds) == 1
    assert "Trader" in judge_conds[0], "Risk Judge must be able to route back to Trader for revision"
