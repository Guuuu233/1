import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import HumanMessage

from tradingagents.agents.analysts.fundamentals_analyst import (
    create_fundamentals_analyst,
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


@pytest.fixture(autouse=True)
def mock_heavy_contexts(monkeypatch):
    monkeypatch.setattr(
        "tradingagents.agents.analysts.fundamentals_analyst.resolve_industry_context",
        lambda **kwargs: ([], "【行业常识知识库】\nmock industry"),
    )
    monkeypatch.setattr(
        "tradingagents.agents.analysts.fundamentals_analyst.resolve_macro_event_context",
        lambda **kwargs: ([], "【宏观事件传导图谱】\nmock macro"),
    )
    monkeypatch.setattr(
        "tradingagents.agents.analysts.fundamentals_analyst.resolve_historical_cases_context",
        lambda **kwargs: ([], "【历史案例复盘】\nmock cases"),
    )


def _make_state(horizon="short", **kwargs):
    state = {
        "trade_date": "2026-03-12",
        "company_of_interest": "600519",
        "horizon": horizon,
        "user_intent": {
            "raw_query": "test",
            "ticker": "600519",
            "horizons": ["short", "medium"],
            "focus_areas": ["盈利能力", "估值"],
            "specific_questions": ["现金流是否充足"],
        },
    }
    state.update(kwargs)
    return state


def _stub_pool():
    return {
        "fundamentals": "2026Q2 营业收入 540 亿元 (+18%)，归母净利润 270 亿元 (+19%)，毛利率 91.5%",
        "balance_sheet": "总资产 3200 亿元，净资产 2800 亿元，资产负债率 12.5%，有息负债为 0",
        "cashflow": "经营活动产生的现金流量净额 260 亿元，资本开支 45 亿元，自由现金流强劲",
        "income_statement": "营业总成本 170 亿元，期间费用率 6.3%，净利率 52.2%",
        "global_indices": "标普500: +0.5%, 纳斯达克: +0.8%",
        "major_assets": "黄金: $2400/oz, 原油: $80/bbl",
        "cn_indices": "沪深300: 3600 (+0.9%)",
        "industry_linkage": None,
        "_data_window": "财报周期",
        "_horizon": "medium",
    }


def test_fundamentals_analyst_returns_trace_short_horizon():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='报告正文\n<!-- VERDICT: {"direction": "看多", "reason": "基本面扎实"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_fundamentals_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("short")))

    assert "fundamentals_report" in result
    assert "analyst_traces" in result
    assert len(result["analyst_traces"]) == 1

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "fundamentals_analyst"
    assert trace["verdict"] == "看多"
    assert trace["research_horizon"] == "short"
    assert trace["observation_horizon"] == "medium"
    assert trace["horizon"] == "short"
    assert trace["data_window"] == "财报周期"
    assert trace["key_finding"] == "基本面分析结论：看多"


def test_fundamentals_analyst_returns_trace_medium_horizon():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='报告正文\n<!-- VERDICT: {"direction": "中性", "reason": "估值合理"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_fundamentals_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("medium")))

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "fundamentals_analyst"
    assert trace["verdict"] == "中性"
    assert trace["research_horizon"] == "medium"
    assert trace["observation_horizon"] == "medium"
    assert trace["horizon"] == "medium"
    assert trace["data_window"] == "财报周期"
    assert trace["key_finding"] == "基本面分析结论：中性"


def test_fundamentals_analyst_prompt_contains_research_and_observation_horizon():
    captured_messages = []
    mock_llm = MagicMock()

    async def _astream(messages):
        captured_messages.append(messages)
        yield SimpleNamespace(content="report\n<!-- VERDICT: {\"direction\": \"中性\", \"reason\": \"平稳\"} -->")

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_fundamentals_analyst(mock_llm, collector)

    # 1. Short research horizon: research=short, observation=medium
    asyncio.run(node(_make_state("short")))
    assert len(captured_messages) == 1
    short_human_msg = next(m for m in captured_messages[0] if isinstance(m, HumanMessage))
    short_prompt = short_human_msg.content
    assert "本次研究档：短线（1-2周，技术面主导）" in short_prompt
    assert "本节点专业观察窗：中线（1-3月，基本面主导）" in short_prompt
    assert "当前分析维度" not in short_prompt

    # 2. Medium research horizon: research=medium, observation=medium
    asyncio.run(node(_make_state("medium")))
    assert len(captured_messages) == 2
    medium_human_msg = next(m for m in captured_messages[1] if isinstance(m, HumanMessage))
    medium_prompt = medium_human_msg.content
    assert "本次研究档：中线（1-3月，基本面主导）" in medium_prompt
    assert "本节点专业观察窗：中线（1-3月，基本面主导）" in medium_prompt
    assert "当前分析维度" not in medium_prompt


def test_fundamentals_analyst_resolves_research_horizon_precedence():
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


def test_fundamentals_analyst_direct_fetch_fallback_without_collector():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(content="report\n<!-- VERDICT: {\"direction\": \"偏多\", \"reason\": \"成长良好\"} -->")

    mock_llm.astream = _astream

    with patch("tradingagents.agents.utils.agent_utils.get_fundamentals", MagicMock(invoke=lambda p: "mock_fund")), \
         patch("tradingagents.agents.utils.agent_utils.get_balance_sheet", MagicMock(invoke=lambda p: "mock_bs")), \
         patch("tradingagents.agents.utils.agent_utils.get_cashflow", MagicMock(invoke=lambda p: "mock_cf")), \
         patch("tradingagents.agents.utils.agent_utils.get_income_statement", MagicMock(invoke=lambda p: "mock_is")), \
         patch("tradingagents.agents.utils.agent_utils.get_global_indices", create=True, new=MagicMock(invoke=lambda p: "mock_gidx")), \
         patch("tradingagents.agents.utils.agent_utils.get_major_assets", create=True, new=MagicMock(invoke=lambda p: "mock_masset")), \
         patch("tradingagents.agents.utils.agent_utils.get_cn_indices", create=True, new=MagicMock(invoke=lambda p: "mock_cnidx")), \
         patch("tradingagents.agents.analysts.fundamentals_analyst._map_stock_to_industry", return_value=None):

        node = create_fundamentals_analyst(mock_llm, data_collector=None)

        # Test short run
        result_short = asyncio.run(node(_make_state("short")))
        trace_short = result_short["analyst_traces"][0]
        assert trace_short["research_horizon"] == "short"
        assert trace_short["observation_horizon"] == "medium"
        assert trace_short["horizon"] == "short"
        assert trace_short["data_window"] == "财报周期"

        # Test medium run
        result_med = asyncio.run(node(_make_state("medium")))
        trace_med = result_med["analyst_traces"][0]
        assert trace_med["research_horizon"] == "medium"
        assert trace_med["observation_horizon"] == "medium"
        assert trace_med["horizon"] == "medium"
        assert trace_med["data_window"] == "财报周期"


def test_fundamentals_analyst_english_prompt(monkeypatch):
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
        yield SimpleNamespace(content="report\n<!-- VERDICT: {\"direction\": \"neutral\", \"reason\": \"stable\"} -->")

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_fundamentals_analyst(mock_llm, collector)

    asyncio.run(node(_make_state("short")))
    human_msg = next(m for m in captured_messages[0] if isinstance(m, HumanMessage))
    content = human_msg.content
    assert "Research Horizon: Short-term" in content
    assert "Node Observation Window: Medium-term" in content
    assert "Current horizon" not in content
