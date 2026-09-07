import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from langchain_core.messages import HumanMessage

from tradingagents.agents.analysts.volume_price_analyst import (
    create_volume_price_analyst,
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
            "raw_query": "test volume price analysis",
            "ticker": "600519",
            "horizons": ["short", "medium"],
            "focus_areas": ["量价关系", "成交量异动"],
            "specific_questions": ["是否放量突破关键阻力位"],
        },
    }
    state.update(kwargs)
    return state


def _stub_pool(ticker="600519", trade_date="2026-03-12"):
    return {
        "vpa_indicators": "【量价指标】\n- 换手率：3.2%\n- 量比：1.5\n- 威科夫阶段：吸筹阶段假设",
        "stock_data": "date,open,high,low,close,volume\n2026-03-12,100,105,98,103,50000",
        "_data_window": "14天",
        "_horizon": "short",
    }


def test_volume_price_analyst_returns_trace_short_horizon():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='量价分析报告正文\n<!-- VERDICT: {"direction": "看多", "reason": "放量突破形态确认"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_volume_price_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("short")))

    assert "volume_price_report" in result
    assert "analyst_traces" in result
    assert len(result["analyst_traces"]) == 1

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "volume_price_analyst"
    assert trace["verdict"] == "看多"
    assert trace["research_horizon"] == "short"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "short"
    assert trace["data_window"] == "14天"
    assert trace["key_finding"] == "量价分析结论：看多"


def test_volume_price_analyst_uses_short_window_for_medium_request():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='量价分析报告正文\n<!-- VERDICT: {"direction": "偏多", "reason": "量价稳步配合"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()
    node = create_volume_price_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("medium")))

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "volume_price_analyst"
    assert trace["verdict"] == "偏多"
    assert trace["research_horizon"] == "medium"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "medium"
    assert trace["data_window"] == "14天"
    assert trace["key_finding"] == "量价分析结论：偏多"


def test_volume_price_analyst_prompt_contains_research_and_observation_horizon():
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
    node = create_volume_price_analyst(mock_llm, collector)

    # 1. Medium research horizon: research=medium, observation=short
    asyncio.run(node(_make_state("medium")))
    assert len(captured_messages) == 1
    medium_human_msg = next(m for m in captured_messages[0] if isinstance(m, HumanMessage))
    medium_prompt = medium_human_msg.content
    assert "本次研究档：中线（1-3月，基本面主导）" in medium_prompt
    assert "本节点专业观察窗：短线（1-2周，技术面主导）" in medium_prompt
    assert "（数据窗口：14天）" in medium_prompt
    assert "当前分析维度" not in medium_prompt

    # 2. Short research horizon: research=short, observation=short
    asyncio.run(node(_make_state("short")))
    assert len(captured_messages) == 2
    short_human_msg = next(m for m in captured_messages[1] if isinstance(m, HumanMessage))
    short_prompt = short_human_msg.content
    assert "本次研究档：短线（1-2周，技术面主导）" in short_prompt
    assert "本节点专业观察窗：短线（1-2周，技术面主导）" in short_prompt
    assert "（数据窗口：14天）" in short_prompt
    assert "当前分析维度" not in short_prompt


def test_volume_price_analyst_calls_get_window_with_short_observation():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='report\n<!-- VERDICT: {"direction": "中性", "reason": "平稳"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool()

    real_get_window = collector.get_window
    calls = []

    def wrapped_get_window(pool, horizon, trade_date):
        calls.append((horizon, trade_date))
        return real_get_window(pool, horizon, trade_date)

    collector.get_window = wrapped_get_window
    node = create_volume_price_analyst(mock_llm, collector)

    # When state is medium, get_window must still receive "short"
    asyncio.run(node(_make_state("medium")))
    assert len(calls) == 1
    assert calls[0][0] == "short"
    assert calls[0][1] == "2026-03-12"


def test_volume_price_analyst_resolves_research_horizon_precedence():
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


def test_volume_price_analyst_fallback_when_collector_none():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='report\n<!-- VERDICT: {"direction": "中性", "reason": "无数据中性"} -->'
        )

    mock_llm.astream = _astream
    node = create_volume_price_analyst(mock_llm, data_collector=None)
    result = asyncio.run(node(_make_state("medium")))

    trace = result["analyst_traces"][0]
    assert trace["research_horizon"] == "medium"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "medium"
    assert trace["data_window"] == "14天"


def test_volume_price_analyst_fallback_when_pool_none():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='report\n<!-- VERDICT: {"direction": "中性", "reason": "无数据中性"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()  # cache is empty
    node = create_volume_price_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("medium")))

    trace = result["analyst_traces"][0]
    assert trace["research_horizon"] == "medium"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "medium"
    assert trace["data_window"] == "14天"


def test_volume_price_analyst_english_prompt(monkeypatch):
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
    node = create_volume_price_analyst(mock_llm, collector)

    # Test medium research horizon in en
    asyncio.run(node(_make_state("medium")))
    human_msg = next(m for m in captured_messages[0] if isinstance(m, HumanMessage))
    content = human_msg.content
    assert "Research Horizon: Medium-term (1-3 months, fundamentals-driven)" in content
    assert "Node Observation Window: Short-term (1-2 weeks, technicals-driven)" in content
    assert "Current horizon" not in content


def test_volume_price_analyst_missing_data_explicit_reporting():
    captured_messages = []
    mock_llm = MagicMock()

    async def _astream(messages):
        captured_messages.append(messages)
        yield SimpleNamespace(
            content='report\n<!-- VERDICT: {"direction": "中性", "reason": "数据缺失中性观望"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()  # No cache -> empty pool
    node = create_volume_price_analyst(mock_llm, collector)

    asyncio.run(node(_make_state("short")))
    human_msg = next(m for m in captured_messages[0] if isinstance(m, HumanMessage))
    content = human_msg.content
    assert "无数据" in content
