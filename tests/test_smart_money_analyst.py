import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import HumanMessage

from tradingagents.agents.analysts.smart_money_analyst import (
    create_smart_money_analyst,
    _resolve_research_horizon,
)
from tradingagents.graph.data_collector import DataCollector
from tradingagents.graph.intent_parser import (
    bind_research_horizon,
    clear_research_horizon,
)


@pytest.fixture(autouse=True)
def clean_horizon():
    clear_research_horizon()
    yield
    clear_research_horizon()


def _make_state(horizon="short", **kwargs):
    state = {
        "trade_date": "2026-03-12",
        "company_of_interest": "600519",
        "horizon": horizon,
        "user_intent": {
            "raw_query": "test smart money analysis",
            "ticker": "600519",
            "horizons": ["short", "medium"],
            "focus_areas": ["主力资金", "大单流向"],
            "specific_questions": ["主力是否在吸筹建仓"],
        },
    }
    state.update(kwargs)
    return state


def _stub_pool(ticker="600519", trade_date="2026-03-12"):
    record = {
        "source": "tushare_eastmoney_moneyflow_dc",
        "source_family": "eastmoney",
        "algorithm_group": "new_algorithm_group",
        "status": "available",
        "symbol": ticker,
        "date": trade_date,
        "period_kind": "historical_daily",
        "time_window": "1d",
        "field": "r0_net",
        "value": "2.211971",
        "unit": "亿元",
        "field_semantics": {"r0_net": "主力净额（负值表示净流出）"},
    }
    return {
        "fund_flow_individual": "东方财富主力资金净额 2.21 亿，主力偏增持",
        "market_data_context": {
            "fund_flow_evidence": {
                "records": [record],
                "symbol": ticker,
                "requested_as_of": trade_date,
            }
        },
        "lhb": "无龙虎榜数据",
        "indicators": {"vwma": "100"},
        "_data_window": "近期可用",
        "_horizon": "short",
    }


def test_smart_money_analyst_returns_trace_short_horizon():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='主力资金报告正文\n<!-- VERDICT: {"direction": "看多", "reason": "主力大单显著净流入"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_smart_money_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("short")))

    assert "smart_money_report" in result
    assert "analyst_traces" in result
    assert len(result["analyst_traces"]) == 1

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "smart_money_analyst"
    assert trace["verdict"] == "看多"
    assert trace["research_horizon"] == "short"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "short"
    assert trace["data_window"] == "近期可用"
    assert trace["key_finding"] == "主力资金分析结论：看多"


def test_smart_money_analyst_returns_trace_medium_horizon():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='主力资金报告正文\n<!-- VERDICT: {"direction": "中性", "reason": "资金博弈平稳分歧不大"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_smart_money_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("medium")))

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "smart_money_analyst"
    assert trace["verdict"] == "中性"
    assert trace["research_horizon"] == "medium"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "medium"
    assert trace["data_window"] == "近期可用"
    assert trace["key_finding"] == "主力资金分析结论：中性"


def test_smart_money_analyst_prompt_contains_research_and_observation_horizon():
    captured_messages = []
    mock_llm = MagicMock()

    async def _astream(messages):
        captured_messages.append(messages)
        yield SimpleNamespace(
            content='report\n<!-- VERDICT: {"direction": "中性", "reason": "平稳"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_smart_money_analyst(mock_llm, collector)

    # 1. Medium research horizon: research=medium, observation=short
    asyncio.run(node(_make_state("medium")))
    assert len(captured_messages) == 1
    medium_human_msg = next(m for m in captured_messages[0] if isinstance(m, HumanMessage))
    medium_prompt = medium_human_msg.content
    assert "本次研究档：中线（1-3月，基本面主导）" in medium_prompt
    assert "本节点专业观察窗：短线（1-2周，技术面主导）" in medium_prompt
    assert "当前分析维度" not in medium_prompt

    # 2. Short research horizon: research=short, observation=short
    asyncio.run(node(_make_state("short")))
    assert len(captured_messages) == 2
    short_human_msg = next(m for m in captured_messages[1] if isinstance(m, HumanMessage))
    short_prompt = short_human_msg.content
    assert "本次研究档：短线（1-2周，技术面主导）" in short_prompt
    assert "本节点专业观察窗：短线（1-2周，技术面主导）" in short_prompt
    assert "当前分析维度" not in short_prompt


def test_smart_money_analyst_resolves_research_horizon_precedence():
    # Priority 1: state['horizon']
    state1 = {"horizon": "short", "horizon_run_metadata": {"resolved": ["medium"]}}
    assert _resolve_research_horizon(state1) == "short"

    # Priority 2: state['horizon_run_metadata']['resolved']
    state2 = {"horizon_run_metadata": {"resolved": ["medium"]}}
    assert _resolve_research_horizon(state2) == "medium"

    # Priority 3: state['horizon_run_metadata']['requested']
    state3 = {"horizon_run_metadata": {"requested": ["medium"]}}
    assert _resolve_research_horizon(state3) == "medium"

    # Priority 4: contextvar binding from H-04a
    bind_research_horizon("medium")
    assert _resolve_research_horizon({}) == "medium"
    clear_research_horizon()

    # Priority 5: default fallback
    assert _resolve_research_horizon({}) == "short"
    assert _resolve_research_horizon(None) == "short"


def test_smart_money_analyst_direct_fetch_fallback_without_collector():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='report\n<!-- VERDICT: {"direction": "看多", "reason": "资金流入良好"} -->'
        )

    mock_llm.astream = _astream

    with patch(
        "tradingagents.agents.utils.agent_utils.get_individual_fund_flow",
        MagicMock(invoke=lambda p: "mock_fund_flow"),
    ), patch(
        "tradingagents.agents.utils.agent_utils.get_lhb_detail",
        MagicMock(invoke=lambda p: "mock_lhb"),
    ), patch(
        "tradingagents.agents.utils.agent_utils.get_indicators",
        MagicMock(invoke=lambda p: "mock_volume"),
    ):
        node = create_smart_money_analyst(mock_llm, data_collector=None)

        # Test short run
        result_short = asyncio.run(node(_make_state("short")))
        trace_short = result_short["analyst_traces"][0]
        assert trace_short["research_horizon"] == "short"
        assert trace_short["observation_horizon"] == "short"
        assert trace_short["horizon"] == "short"
        assert trace_short["data_window"] == "近期可用"

        # Test medium run
        result_med = asyncio.run(node(_make_state("medium")))
        trace_med = result_med["analyst_traces"][0]
        assert trace_med["research_horizon"] == "medium"
        assert trace_med["observation_horizon"] == "short"
        assert trace_med["horizon"] == "medium"
        assert trace_med["data_window"] == "近期可用"


def test_smart_money_analyst_english_prompt(monkeypatch):
    from tradingagents.dataflows.config import get_config

    current_cfg = get_config()
    monkeypatch.setattr(
        "tradingagents.graph.intent_parser.get_config",
        lambda: {**current_cfg, "prompt_language": "en"},
    )

    captured_messages = []
    mock_llm = MagicMock()

    async def _astream(messages):
        captured_messages.append(messages)
        yield SimpleNamespace(
            content='report\n<!-- VERDICT: {"direction": "neutral", "reason": "stable"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_smart_money_analyst(mock_llm, collector)

    # Test medium research horizon in en
    asyncio.run(node(_make_state("medium")))
    human_msg = next(m for m in captured_messages[0] if isinstance(m, HumanMessage))
    content = human_msg.content
    assert "Research Horizon: Medium-term (1-3 months, fundamentals-driven)" in content
    assert "Node Observation Window: Short-term (1-2 weeks, technicals-driven)" in content
    assert "Current horizon" not in content


def test_smart_money_analyst_missing_data_explicit_reporting():
    """Verify data failures/missing data are explicitly reported in prompt and not fabricated."""
    captured_messages = []
    mock_llm = MagicMock()

    async def _astream(messages):
        captured_messages.append(messages)
        yield SimpleNamespace(
            content='report\n<!-- VERDICT: {"direction": "中性", "reason": "数据缺失中性观望"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    pool_with_missing = {
        "fund_flow_individual": "无数据",
        "market_data_context": {},
        "lhb": "无数据",
        "indicators": {},
        "_data_window": "近期可用",
        "_horizon": "short",
    }
    collector._cache["600519_2026-03-12"] = pool_with_missing
    node = create_smart_money_analyst(mock_llm, collector)

    asyncio.run(node(_make_state("medium")))
    human_msg = next(m for m in captured_messages[0] if isinstance(m, HumanMessage))
    content = human_msg.content
    assert "【资金流数据（来源、日期与口径见数据）】\n无数据" in content
    assert "【龙虎榜数据】\n无数据" in content
    assert "【成交量指标(vwma)】\n无数据" in content


def test_smart_money_analyst_preserves_fund_flow_consensus_guard():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='主力资金报告\n<!-- VERDICT: {"direction": "中性", "reason": "资金平衡"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_smart_money_analyst(mock_llm, collector)

    result = asyncio.run(node(_make_state("medium")))
    assert "fund_flow_consensus_guard" in result
    guard = result["fund_flow_consensus_guard"]
    assert "blocked" in guard
    assert "direction_allowed" in guard
    assert guard["blocked"] is False
    assert guard["direction_allowed"] is True
