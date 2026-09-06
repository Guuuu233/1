import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from tradingagents.agents.analysts.social_media_analyst import (
    create_social_media_analyst,
    _resolve_research_horizon,
)
from tradingagents.graph.data_collector import DataCollector
from tradingagents.graph.intent_parser import (
    bind_research_horizon,
    clear_research_horizon,
)
from tradingagents.dataflows.social.contracts import (
    SentimentBundleV1,
    SocialAttention,
    SocialDataContext,
    SocialSentiment,
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


def _stub_social_pool(symbol="600519", as_of="2026-03-12"):
    bundle = SentimentBundleV1(
        status="available",
        requested_as_of=as_of,
        cutoff_at=f"{as_of}T15:59:59Z",
        content_as_of=f"{as_of}T08:30:00Z",
        metric_as_of=f"{as_of}T15:00:00Z",
        direction_allowed=True,
        reason_codes=[],
        symbol=symbol,
        bundle_id="sha256:testbundle123456",
        social_attention=SocialAttention(
            post_count=15,
            comment_count=60,
            author_count=50,
            total_interactions=3500,
            interaction_velocity=12.5,
        ),
        social_sentiment=SocialSentiment(
            score=0.45,
            label="bullish",
            bullish_count=10,
            bearish_count=2,
            neutral_count=3,
            insufficient_count=0,
            is_calibrated_probability=False,
        ),
        evidence_samples=[
            {
                "record_id": "xhs:post:sample1",
                "platform": "xhs",
                "record_type": "post",
                "published_at": f"{as_of}T08:00:00Z",
                "title": "茅台讨论",
                "text": "茅台社交讨论样本",
                "stance_label": "bullish",
            }
        ],
        platform_breakdown={
            "xhs": {"post_count": 10, "comment_count": 40},
            "douyin": {"post_count": 5, "comment_count": 20},
        },
    )
    social_data_context: SocialDataContext = {
        "status": "available",
        "mode": "active",
        "requested_as_of": as_of,
        "direction_allowed": True,
        "reason_codes": [],
        "bundle": bundle.to_dict(),
        "source_provenance": {},
        "data_failure_ledger": [],
    }
    return {
        "social_data_context": social_data_context,
        "market_data_context": {
            "market_attention": {
                "zt_pool": {"status": "available", "as_of": as_of, "raw": "连板最高5板"},
                "hot_stocks": {"status": "available", "as_of": as_of, "raw": "雪球关注榜第一"},
            }
        },
    }


def test_social_media_analyst_returns_trace():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='舆情分析报告正文\n<!-- VERDICT: {"direction": "看多", "reason": "情绪高涨"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_social_pool()
    node = create_social_media_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("short", mode="active")))

    assert "sentiment_report" in result
    assert "analyst_traces" in result
    assert len(result["analyst_traces"]) == 1

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "social_media_analyst"
    assert trace["verdict"] == "看多"
    assert trace["research_horizon"] == "short"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "short"
    assert trace["data_window"] == "7天"


def test_social_media_analyst_uses_short_window_for_medium_request():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(content="report")

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_social_pool()
    node = create_social_media_analyst(mock_llm, collector)
    result = asyncio.run(node(_make_state("medium", mode="active")))

    trace = result["analyst_traces"][0]
    assert trace["research_horizon"] == "medium"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "medium"
    assert trace["data_window"] == "7天"


def test_social_media_analyst_prompt_contains_research_and_observation_horizon():
    captured_messages = []

    mock_llm = MagicMock()

    async def _astream(messages):
        captured_messages.append(messages)
        yield SimpleNamespace(content="report")

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_social_pool()
    node = create_social_media_analyst(mock_llm, collector)

    # 1. Medium research horizon
    asyncio.run(node(_make_state("medium", mode="active")))
    assert len(captured_messages) == 1
    # Captured messages: [SystemMessage, HumanMessage]
    medium_prompt = captured_messages[0][1].content
    assert "本次研究档：中线（1-3月，基本面主导）" in medium_prompt
    assert "本节点专业观察窗：短线（1-2周，技术面主导）" in medium_prompt

    # 2. Short research horizon
    asyncio.run(node(_make_state("short", mode="active")))
    assert len(captured_messages) == 2
    short_prompt = captured_messages[1][1].content
    assert "本次研究档：短线（1-2周，技术面主导）" in short_prompt
    assert "本节点专业观察窗：短线（1-2周，技术面主导）" in short_prompt


def test_social_media_analyst_resolves_research_horizon_precedence():
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


def test_social_media_analyst_active_mode_preserves_medium_horizon_and_7day_window():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='【正式分析报告】\n<!-- VERDICT: {"direction": "偏多", "reason": "样本偏多"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = _stub_social_pool()
    node = create_social_media_analyst(mock_llm, collector)

    result = asyncio.run(node(_make_state("medium", mode="active")))

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "social_media_analyst"
    assert trace["research_horizon"] == "medium"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "medium"
    assert trace["data_window"] == "7天"
    assert trace["source_mode"] == "active"
    assert trace["source_status"] == "available"
    assert trace["direction_allowed"] is True
    assert trace["bundle_id"] == "sha256:testbundle123456"
    assert trace["evidence_refs"] == ["xhs:post:sample1"]


def test_social_media_analyst_disabled_mode_preserves_medium_horizon_and_7day_window():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(
            content='【正式分析报告】数据不可用。\n<!-- VERDICT: {"direction": "中性", "reason": "不可用"} -->'
        )

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = {
        "social_data_context": {
            "status": "not_applicable",
            "mode": "disabled",
            "requested_as_of": "2026-03-12",
            "direction_allowed": False,
            "reason_codes": ["social_not_applicable"],
            "bundle": None,
            "source_provenance": {},
            "data_failure_ledger": [],
        },
        "market_data_context": {
            "market_attention": {
                "zt_pool": {"status": "available", "as_of": "2026-03-12", "raw": "连板3板"},
                "hot_stocks": {"status": "available", "as_of": "2026-03-12", "raw": "雪球榜"},
            }
        },
    }
    node = create_social_media_analyst(mock_llm, collector)

    result = asyncio.run(node(_make_state("medium", mode="disabled")))

    trace = result["analyst_traces"][0]
    assert trace["agent"] == "social_media_analyst"
    assert trace["research_horizon"] == "medium"
    assert trace["observation_horizon"] == "short"
    assert trace["horizon"] == "medium"
    assert trace["data_window"] == "7天"
    assert trace["source_mode"] == "disabled"
    assert trace["source_status"] == "not_applicable"
    assert trace["direction_allowed"] is False
    assert "（社交数据不可用/方向不可判断）" in trace["key_finding"]


def test_social_media_analyst_fallback_without_collector_uses_short_window():
    mock_llm = MagicMock()

    async def _astream(_messages):
        yield SimpleNamespace(content="report")

    mock_llm.astream = _astream

    with patch("tradingagents.agents.utils.agent_utils.get_zt_pool") as mock_zt, \
         patch("tradingagents.agents.utils.agent_utils.get_hot_stocks_xq") as mock_hot:
        mock_zt.invoke.return_value = "zt pool data"
        mock_hot.invoke.return_value = "hot stocks data"

        node = create_social_media_analyst(mock_llm, data_collector=None)
        result = asyncio.run(node(_make_state("medium")))

        trace = result["analyst_traces"][0]
        assert trace["agent"] == "social_media_analyst"
        assert trace["research_horizon"] == "medium"
        assert trace["observation_horizon"] == "short"
        assert trace["horizon"] == "medium"
        assert trace["data_window"] == "7天"
