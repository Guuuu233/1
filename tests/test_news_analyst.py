import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from tradingagents.agents.analysts.news_analyst import (
    create_news_analyst,
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
def mock_macro_event_context():
    with patch(
        "tradingagents.agents.analysts.news_analyst.resolve_macro_event_context",
        return_value=([], ""),
    ):
        yield


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
    days = "14天"
    return {
        "news": "## 600519 新闻\n- 2026-03-10 贵州茅台发布一季度经营数据预喜公告",
        "global_news": "## 宏观资讯\n- 2026-03-11 央行引导流动性保持合理充裕",
        "_data_window": days,
        "_horizon": horizon,
    }


def test_news_analyst_returns_trace():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='新闻报告正文\n<!-- VERDICT: {"direction": "看多", "reason": "利多驱动"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool("short")
    node = create_news_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("short")))

    assert "news_report" in result
    assert "event_coverage" in result
    assert "analyst_traces" in result
    assert len(result["analyst_traces"]) == 1

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "news_analyst"
    assert trace["verdict"] == "看多"
    assert trace["research_horizon"] == "short"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "short"
    assert trace["data_window"] == "14天"


def test_news_analyst_uses_short_window_for_medium_request():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(content="report")

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool("medium")
    node = create_news_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("medium")))

    trace = result["analyst_traces"][0]
    assert trace["research_horizon"] == "medium"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "medium"
    assert trace["data_window"] == "14天"


def test_news_analyst_prompt_contains_research_and_observation_horizon():
    captured_messages = []

    mock_llm = MagicMock()

    async def _astream(messages):
        captured_messages.append(messages)
        yield SimpleNamespace(content="report")

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_pool("short")
    node = create_news_analyst(mock_llm, collector)

    # 1. Medium research horizon
    asyncio.run(node(_make_state("medium")))
    assert len(captured_messages) == 1
    medium_prompt = captured_messages[0][1].content
    assert "本次研究档：中线（1-3月，基本面主导）" in medium_prompt
    assert "本节点专业观察窗：短线（1-2周，技术面主导）" in medium_prompt
    assert "（14天）" in medium_prompt

    # 2. Short research horizon
    asyncio.run(node(_make_state("short")))
    assert len(captured_messages) == 2
    short_prompt = captured_messages[1][1].content
    assert "本次研究档：短线（1-2周，技术面主导）" in short_prompt
    assert "本节点专业观察窗：短线（1-2周，技术面主导）" in short_prompt
    assert "（14天）" in short_prompt


def test_news_analyst_resolves_research_horizon_precedence():
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


def test_news_analyst_direct_fetch_fallback_uses_short_window():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(content="report")

    mock_llm.astream = _astream

    async def _fake_fetch_direct(ticker, current_date, horizon):
        assert horizon == "short"
        return "fake_stock_news", "fake_global_news", "14天"

    with patch(
        "tradingagents.agents.analysts.news_analyst._fetch_direct",
        side_effect=_fake_fetch_direct,
    ):
        # collector is None -> fallback to _fetch_direct
        node = create_news_analyst(mock_llm, data_collector=None)
        result = asyncio.run(node(_make_state("medium")))

        trace = result["analyst_traces"][0]
        assert trace["research_horizon"] == "medium"
        assert trace["observation_horizon"] == "short"
        assert trace["horizon"] == "medium"
        assert trace["data_window"] == "14天"


def test_news_analyst_direct_fetch_calls_agent_utils_with_14_days():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(content="report")

    mock_llm.astream = _astream

    with patch("tradingagents.agents.utils.agent_utils.get_news") as mock_news, \
         patch("tradingagents.agents.utils.agent_utils.get_global_news") as mock_global_news:
        mock_news.invoke.return_value = "stock news"
        mock_global_news.invoke.return_value = "global news"

        node = create_news_analyst(mock_llm, data_collector=None)
        result = asyncio.run(node(_make_state("medium")))

        # Check that lookback was 14 days
        mock_news.invoke.assert_called_once_with({
            "ticker": "600519",
            "start_date": "2026-02-26",
            "end_date": "2026-03-12",
        })
        mock_global_news.invoke.assert_called_once_with({
            "curr_date": "2026-03-12",
            "look_back_days": 14,
            "limit": 10,
        })

        trace = result["analyst_traces"][0]
        assert trace["research_horizon"] == "medium"
        assert trace["observation_horizon"] == "short"
        assert trace["horizon"] == "medium"
        assert trace["data_window"] == "14天"
