import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import HumanMessage

from tradingagents.agents.analysts.macro_analyst import (
    create_macro_analyst,
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
        "tradingagents.agents.analysts.macro_analyst.resolve_industry_context",
        lambda **kwargs: ([], "【行业常识知识库】\nmock industry"),
    )
    monkeypatch.setattr(
        "tradingagents.agents.analysts.macro_analyst.resolve_macro_event_context",
        lambda **kwargs: ([], "【宏观事件传导图谱】\nmock macro event"),
    )
    monkeypatch.setattr(
        "tradingagents.agents.analysts.macro_analyst.resolve_historical_cases_context",
        lambda **kwargs: ([], "【历史案例复盘】\nmock cases"),
    )


def _make_state(horizon="short", **kwargs):
    state = {
        "trade_date": "2026-03-12",
        "company_of_interest": "600519",
        "horizon": horizon,
        "user_intent": {
            "raw_query": "test macro analysis",
            "ticker": "600519",
            "horizons": ["short", "medium"],
            "focus_areas": ["宏观流动性", "行业轮动"],
            "specific_questions": ["外盘对大盘有什么影响"],
        },
    }
    state.update(kwargs)
    return state


def _stub_pool():
    return {
        "fund_flow_board": "白酒板块今日主力净流入 15.6 亿元，排名行业第一",
        "news": "消费刺激政策持续落地，扩大内需战略稳步推进",
        "global_news": "美联储维持基准利率不变，全球流动性预期边际改善",
        "global_indices": "标普500: 5100 (+0.4%), 纳斯达克: 16200 (+0.6%), 恒生指数: 17200 (+1.1%)",
        "major_assets": "美元指数: 103.2 (-0.3%), 布伦特原油: $82/bbl (+0.5%), 伦敦金: $2380/oz (+0.8%)",
        "cn_indices": "上证指数: 3080 (+0.8%), 沪深300: 3620 (+1.0%), 创业板指: 1850 (+1.3%)",
        "northbound_flow": "北向资金今日全天净买入 68 亿元，其中沪股通净买入 42 亿元",
        "industry_linkage": None,
        "_data_window": "板块数据",
        "_horizon": "medium",
    }


def test_macro_analyst_returns_trace_short_horizon():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='宏观报告正文\n<!-- VERDICT: {"direction": "看多", "reason": "宏观流动性宽松，板块资金大幅净流入"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_macro_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("short")))

    assert "macro_report" in result
    assert "analyst_traces" in result
    assert len(result["analyst_traces"]) == 1

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "macro_analyst"
    assert trace["verdict"] == "看多"
    assert trace["research_horizon"] == "short"
    assert trace["observation_horizon"] == "medium"
    assert trace["horizon"] == "short"
    assert trace["data_window"] == "板块数据"
    assert trace["key_finding"] == "宏观板块分析结论：看多"


def test_macro_analyst_returns_trace_medium_horizon():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='宏观报告正文\n<!-- VERDICT: {"direction": "中性", "reason": "宏观环境平稳，行业估值合理"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_macro_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("medium")))

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "macro_analyst"
    assert trace["verdict"] == "中性"
    assert trace["research_horizon"] == "medium"
    assert trace["observation_horizon"] == "medium"
    assert trace["horizon"] == "medium"
    assert trace["data_window"] == "板块数据"
    assert trace["key_finding"] == "宏观板块分析结论：中性"


def test_macro_analyst_prompt_contains_research_and_observation_horizon():
    captured_messages = []
    mock_llm = MagicMock()

    async def _astream(messages):
        captured_messages.append(messages)
        yield SimpleNamespace(content="report\n<!-- VERDICT: {\"direction\": \"中性\", \"reason\": \"平稳\"} -->")

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_macro_analyst(mock_llm, collector)

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


def test_macro_analyst_resolves_research_horizon_precedence():
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


def test_macro_analyst_direct_fetch_fallback_without_collector():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(content="report\n<!-- VERDICT: {\"direction\": \"看多\", \"reason\": \"外盘平稳资金流入\"} -->")

    mock_llm.astream = _astream

    with patch("tradingagents.agents.utils.agent_utils.get_board_fund_flow", MagicMock(invoke=lambda p: "mock_flow")), \
         patch("tradingagents.agents.utils.agent_utils.get_news", MagicMock(invoke=lambda p: "mock_news")), \
         patch("tradingagents.agents.utils.agent_utils.get_global_news", MagicMock(invoke=lambda p: "mock_gnews")), \
         patch("tradingagents.agents.utils.agent_utils.get_northbound_flow", MagicMock(invoke=lambda p: "mock_nb")), \
         patch("tradingagents.agents.utils.agent_utils.get_global_indices", create=True, new=MagicMock(invoke=lambda p: "mock_gidx")), \
         patch("tradingagents.agents.utils.agent_utils.get_major_assets", create=True, new=MagicMock(invoke=lambda p: "mock_masset")), \
         patch("tradingagents.agents.utils.agent_utils.get_cn_indices", create=True, new=MagicMock(invoke=lambda p: "mock_cnidx")), \
         patch("tradingagents.agents.analysts.macro_analyst._map_stock_to_industry", return_value=None):

        node = create_macro_analyst(mock_llm, data_collector=None)

        # Test short run
        result_short = asyncio.run(node(_make_state("short")))
        trace_short = result_short["analyst_traces"][0]
        assert trace_short["research_horizon"] == "short"
        assert trace_short["observation_horizon"] == "medium"
        assert trace_short["horizon"] == "short"
        assert trace_short["data_window"] == "板块数据"

        # Test medium run
        result_med = asyncio.run(node(_make_state("medium")))
        trace_med = result_med["analyst_traces"][0]
        assert trace_med["research_horizon"] == "medium"
        assert trace_med["observation_horizon"] == "medium"
        assert trace_med["horizon"] == "medium"
        assert trace_med["data_window"] == "板块数据"


def test_macro_analyst_english_prompt(monkeypatch):
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
    node = create_macro_analyst(mock_llm, collector)

    asyncio.run(node(_make_state("short")))
    human_msg = next(m for m in captured_messages[0] if isinstance(m, HumanMessage))
    content = human_msg.content
    assert "Research Horizon: Short-term" in content
    assert "Node Observation Window: Medium-term" in content
    assert "Current horizon" not in content


def test_macro_analyst_missing_data_explicit_reporting():
    """Verify data failures/missing data are explicitly reported in prompt and not fabricated."""
    captured_messages = []
    mock_llm = MagicMock()

    async def _astream(messages):
        captured_messages.append(messages)
        yield SimpleNamespace(content="report\n<!-- VERDICT: {\"direction\": \"中性\", \"reason\": \"数据缺失中性观望\"} -->")

    mock_llm.astream = _astream
    collector = DataCollector()
    pool_with_missing = {
        "fund_flow_board": "无数据",
        "news": "无数据",
        "global_news": "无数据",
        "global_indices": "无数据",
        "major_assets": "无数据",
        "cn_indices": "无数据",
        "northbound_flow": "无数据",
        "industry_linkage": None,
        "_data_window": "板块数据",
        "_horizon": "medium",
    }
    collector._cache["600519_2026-03-12"] = pool_with_missing
    node = create_macro_analyst(mock_llm, collector)

    asyncio.run(node(_make_state("short")))
    human_msg = next(m for m in captured_messages[0] if isinstance(m, HumanMessage))
    content = human_msg.content
    assert "【数据缺失】全球核心市场指数数据未获取到" in content
