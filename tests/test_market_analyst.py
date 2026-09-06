import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from tradingagents.agents.analysts.market_analyst import (
    create_market_analyst,
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
            "raw_query": "test",
            "ticker": "600519",
            "horizons": ["short", "medium"],
            "focus_areas": [],
            "specific_questions": [],
        },
    }
    state.update(kwargs)
    return state


def _stub_pool(horizon="short"):
    days = "14天" if horizon == "short" else "90天"
    return {
        "stock_data": "close,volume\n100,1000",
        "indicators": {
            k: "50"
            for k in [
                "close_50_sma",
                "close_200_sma",
                "close_10_ema",
                "rsi",
                "macd",
                "boll",
                "boll_ub",
                "boll_lb",
                "atr",
                "vwma",
            ]
        },
        "_data_window": days,
        "_horizon": horizon,
    }


def test_market_analyst_returns_trace():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='报告\n<!-- VERDICT: {"direction": "看多", "reason": "趋势向上"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool("short")
    node = create_market_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("short")))
    assert "market_report" in result
    assert "analyst_traces" in result
    assert len(result["analyst_traces"]) == 1

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "market_analyst"
    assert trace["verdict"] == "看多"
    assert trace["research_horizon"] == "short"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "short"
    assert trace["data_window"] == "14天"


def test_market_analyst_uses_short_window_for_medium_request():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(content="report")

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool("medium")
    node = create_market_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("medium")))

    trace = result["analyst_traces"][0]
    assert trace["research_horizon"] == "medium"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "medium"
    assert trace["data_window"] == "14天"


def test_market_analyst_prompt_contains_research_and_observation_horizon():
    captured_messages = []

    mock_llm = MagicMock()

    async def _astream(messages):
        captured_messages.append(messages)
        yield SimpleNamespace(content="report")

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool("short")
    node = create_market_analyst(mock_llm, collector)

    # 1. Medium research horizon
    asyncio.run(node(_make_state("medium")))
    assert len(captured_messages) == 1
    medium_prompt = captured_messages[0][1].content
    assert "本次研究档：中线（1-3月，基本面主导）" in medium_prompt
    assert "本节点专业观察窗：短线（1-2周，技术面主导）" in medium_prompt
    assert "（数据窗口：14天）" in medium_prompt

    # 2. Short research horizon
    asyncio.run(node(_make_state("short")))
    assert len(captured_messages) == 2
    short_prompt = captured_messages[1][1].content
    assert "本次研究档：短线（1-2周，技术面主导）" in short_prompt
    assert "本节点专业观察窗：短线（1-2周，技术面主导）" in short_prompt
    assert "（数据窗口：14天）" in short_prompt


def test_market_analyst_resolves_research_horizon_precedence():
    # Priority 1: state['horizon']
    state1 = {"horizon": "medium", "horizon_run_metadata": {"resolved": ["short"]}}
    assert _resolve_research_horizon(state1) == "medium"

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


def test_market_analyst_direct_fetch_fallback_uses_short_window():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(content="report")

    mock_llm.astream = _astream

    async def _fake_fetch_direct(ticker, current_date, horizon):
        assert horizon == "short"
        return "fake_stock_data", {}, "14天"

    with patch(
        "tradingagents.agents.analysts.market_analyst._fetch_direct",
        side_effect=_fake_fetch_direct,
    ):
        # collector is None -> fallback to _fetch_direct
        node = create_market_analyst(mock_llm, data_collector=None)
        result = asyncio.run(node(_make_state("medium")))

        trace = result["analyst_traces"][0]
        assert trace["research_horizon"] == "medium"
        assert trace["observation_horizon"] == "short"
        assert trace["horizon"] == "medium"
        assert trace["data_window"] == "14天"
