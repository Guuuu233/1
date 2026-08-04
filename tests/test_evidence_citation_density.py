"""Golden-output end-to-end regression for adjudicator evidence citation density.

DAV-68 M2 optimization ②: the whole adjudication chain — research_manager →
trader → risk_manager — must cite concrete evidence (numbers, prices, dates)
in its output, not just argumentation. This test drives the real graph nodes
with golden analyst reports and golden (mock-LLM) adjudicator outputs, then
measures how densely each output references the specific evidence facts that
were present in that node's input prompt.

The metric is per-hop citation density:

    cited(facts present in the node's input) / facts present in the input

This ties the output check back to the wiring: if a future change drops the
evidence summaries from research_manager (or the investment plan from trader /
trader plan from risk_manager), the available-fact count collapses and the
assertion fails — even though a mock LLM would still return the fixed golden
output. Reference: KNOWN_ISSUES #2 / DAV-68 "裁决必须基于证据强度".
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.managers.risk_manager import create_risk_manager
from tradingagents.agents.trader.trader import create_trader


# ---------------------------------------------------------------------------
# Golden fixtures
# ---------------------------------------------------------------------------

GOLDEN_REPORTS = {
    "market_report": (
        "市场技术面：RSI 48.2，股价 1835.5 站上 50 日均线。\n"
        '<!-- VERDICT: {"direction": "偏多", "reason": "趋势向上"} -->'
    ),
    "news_report": (
        "新闻：行业政策利好落地，预计增速 12%。\n"
        '<!-- VERDICT: {"direction": "偏多", "reason": "政策驱动"} -->'
    ),
    "fundamentals_report": (
        "基本面：营收同比 +15%，毛利率 45%。\n"
        '<!-- VERDICT: {"direction": "中性", "reason": "估值合理"} -->'
    ),
    "macro_report": (
        "宏观/板块：板块资金净流入 23 亿。\n"
        '<!-- VERDICT: {"direction": "偏多", "reason": "政策与资金共振"} -->'
    ),
    "sentiment_report": "情绪：中性偏热，无极端值。",
    "smart_money_report": "主力资金：净流入。",
    "volume_price_report": "量价：放量突破。",
}

# Concrete evidence facts an adjudicator must be able to cite back.
EVIDENCE_FACTS = ["RSI 48.2", "1835.5", "12%", "+15%", "45%", "23 亿"]

# Golden adjudicator outputs — each deliberately cites a dense subset of the
# facts its input carried, the way a well-behaved model is instructed to.
RESEARCH_MANAGER_GOLDEN = (
    "裁决 Buy。证据交叉核验：市场 RSI 48.2、股价 1835.5 站上 50 日均线；"
    "基本面营收同比 +15%、毛利率 45%；新闻行业增速 12%；宏观板块资金净流入 23 亿。"
    "多空辩论与证据一致，采纳上述最强证据，结论看多。\n"
    '<!-- VERDICT: {"direction": "看多", "reason": "技术+基本面共振"} -->'
)

TRADER_GOLDEN = (
    "最终交易建议：买入。依据研究经理方案：RSI 48.2 未超买、股价 1835.5 站稳均线、"
    "净利 +15% 支撑估值。仓位 20%，入场区间 1800–1835，止损 1780。\n"
    '<!-- VERDICT: {"direction": "看多", "reason": "技术+基本面共振"} -->'
)

RISK_MANAGER_GOLDEN = (
    "风控通过。硬约束：仓位≤20%，止损 1780。前提：净利 +15% 兑现、RSI 48.2 不破位。"
    "触发：股价跌破 1780。\n"
    '<!-- RISK_JUDGE: {"verdict": "pass", "revision_reason": "", '
    '"hard_constraints": ["仓位≤20%"], "soft_constraints": [], '
    '"execution_preconditions": ["净利+15%兑现"], "de_risk_triggers": ["跌破1780"]} -->'
)


# ---------------------------------------------------------------------------
# Density metric
# ---------------------------------------------------------------------------

def available_facts(input_text: str, facts: list[str]) -> list[str]:
    """Return the golden facts that actually appear in the node's input text."""
    return [f for f in facts if f in (input_text or "")]


def citation_density(output: str, available: list[str]) -> float:
    """Fraction of the available evidence facts the output actually cites.

    No available facts → 0.0 (the input carried no evidence to cite), so a
    wiring regression that starves the adjudicator of facts fails the floor
    rather than silently passing.
    """
    if not available:
        return 0.0
    cited = sum(1 for f in available if f in (output or ""))
    return cited / len(available)


# ---------------------------------------------------------------------------
# Node drivers (real graph nodes, mocked LLMs)
# ---------------------------------------------------------------------------

def _fake_llm(golden_output: str, captured: list) -> MagicMock:
    llm = MagicMock()

    async def _astream(*args, **_kwargs):
        captured.append(args[0])
        yield MagicMock(content=golden_output)

    llm.astream = _astream
    return llm


def _memory():
    m = MagicMock()
    m.get_memories = MagicMock(return_value=[])
    return m


def _debate_state(**overrides):
    base = {
        "history": "Bull: 看多\nBear: 看空",
        "bear_history": "",
        "bull_history": "",
        "current_speaker": "",
        "current_response": "",
        "judge_decision": "",
        "count": 0,
        "claims": [],
        "focus_claim_ids": [],
        "open_claim_ids": [],
        "resolved_claim_ids": [],
        "unresolved_claim_ids": [],
        "round_summary": "",
        "round_goal": "",
        "claim_counter": 0,
    }
    base.update(overrides)
    return base


def _risk_debate_state(**overrides):
    base = {
        "history": "Aggressive: 进攻\nConservative: 防守\nNeutral: 均衡",
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "latest_speaker": "",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "judge_decision": "",
        "count": 0,
        "claims": [],
        "focus_claim_ids": [],
        "open_claim_ids": [],
        "resolved_claim_ids": [],
        "unresolved_claim_ids": [],
        "round_summary": "",
        "round_goal": "",
        "claim_counter": 0,
    }
    base.update(overrides)
    return base


def _state(**overrides):
    state = {
        "company_of_interest": "600519",
        "instrument_context": {},
        "market_context": {},
        "user_context": {},
        "risk_feedback_state": {},
        "investment_debate_state": _debate_state(),
        "risk_debate_state": _risk_debate_state(),
        **GOLDEN_REPORTS,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_adjudication_chain_cites_evidence_densely():
    """Run research_manager → trader → risk_manager and assert per-hop density."""
    state = _state()

    # ── hop 1: research_manager ──────────────────────────────────────────
    rm_captured: list = []
    rm_node = create_research_manager(
        _fake_llm(RESEARCH_MANAGER_GOLDEN, rm_captured), _memory(),
        custom_prompt="", placement="after_data",
    )
    rm_result = asyncio.run(rm_node(state))

    rm_prompt = rm_captured[0]
    rm_available = available_facts(rm_prompt, EVIDENCE_FACTS)
    # The M2 wiring: the manager's prompt must carry the evidence summaries.
    assert len(rm_available) >= 4, (
        f"research_manager prompt missing evidence summaries: only "
        f"{len(rm_available)}/{len(EVIDENCE_FACTS)} facts present"
    )
    rm_density = citation_density(rm_result["investment_plan"], rm_available)
    assert rm_density >= 0.67, f"research_manager citation density too low: {rm_density:.2f}"

    # ── hop 2: trader (input = research_manager plan) ────────────────────
    state["investment_plan"] = rm_result["investment_plan"]
    state["trader_investment_plan"] = ""
    trader_captured: list = []
    trader_node = create_trader(
        _fake_llm(TRADER_GOLDEN, trader_captured), _memory(),
        custom_prompt="", placement="after_data",
    )
    trader_result = asyncio.run(trader_node(state))

    # Trader streams a [system, user] message list; the user message carries
    # the investment plan, which is where the evidence facts must live.
    trader_messages = trader_captured[0]
    trader_input = trader_messages[1]["content"]
    trader_available = available_facts(trader_input, EVIDENCE_FACTS)
    assert trader_available, "trader input carries no evidence via the investment plan"
    trader_density = citation_density(trader_result["trader_investment_plan"], trader_available)
    assert trader_density >= 0.5, f"trader citation density too low: {trader_density:.2f}"

    # ── hop 3: risk_manager (input = trader plan) ────────────────────────
    state["trader_investment_plan"] = trader_result["trader_investment_plan"]
    risk_captured: list = []
    risk_node = create_risk_manager(
        _fake_llm(RISK_MANAGER_GOLDEN, risk_captured), _memory(),
        custom_prompt="", placement="after_data",
    )
    risk_result = asyncio.run(risk_node(state))

    risk_prompt = risk_captured[0]
    risk_available = available_facts(risk_prompt, EVIDENCE_FACTS)
    assert risk_available, "risk_manager input carries no evidence via the trader plan"
    risk_density = citation_density(risk_result["final_trade_decision"], risk_available)
    assert risk_density >= 0.5, f"risk_manager citation density too low: {risk_density:.2f}"

    # The adjudication output that gets persisted must be evidence-bearing.
    assert risk_result["final_trade_decision"], "risk_manager produced no final decision"


def test_citation_density_metric_is_strict():
    """The metric must score 0 when the output ignores every available fact."""
    available = available_facts(
        "市场 RSI 48.2，净利 +15%，资金净流入 23 亿。", EVIDENCE_FACTS
    )
    assert len(available) == 3
    assert citation_density("全部依据技术分析与资金面判断，无具体数据。", available) == 0.0
    assert citation_density("引用 RSI 48.2 与 23 亿资金净流入。", available) == pytest.approx(2 / 3)
    # No available facts → 0.0, not an accidental pass.
    assert citation_density("无输入证据。", []) == 0.0
