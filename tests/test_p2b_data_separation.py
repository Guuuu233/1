"""Deterministic contracts for the P2B daily/realtime split."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from api.main import _build_result_payload
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.graph import data_collector
from tradingagents.graph.data_collector import DataCollector
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _context():
    return {
        "daily": {"as_of": "2026-07-28", "completeness": "completed"},
        "realtime": {
            "status": "available",
            "source": "sina",
            "quote_as_of": "2026-07-29 14:30:00",
            "retrieved_at": "2026-07-29T06:30:01+00:00",
            "error": None,
            "quote": {"price": 10.5, "volume": 1000.0},
        },
    }


def test_daily_context_uses_latest_complete_date():
    df = pd.DataFrame({"date": ["2026-07-27", "2026-07-28", "2026-07-29"]})
    build_daily_context = getattr(data_collector, "_build_daily_context", None)
    assert callable(build_daily_context)

    assert build_daily_context(df, "2026-07-29") == {
        "as_of": "2026-07-29",
        "completeness": "completed",
    }


def test_realtime_context_refuses_historical_date_without_vendor_call():
    fetch_realtime_context = getattr(data_collector, "_fetch_realtime_context", None)
    assert callable(fetch_realtime_context)
    with patch(
        "tradingagents.graph.data_collector.route_to_vendor",
        create=True,
    ) as route:
        context = fetch_realtime_context("600519", "2026-07-28")

    route.assert_not_called()
    assert context["status"] == "not_applicable"
    assert context["source"] is None
    assert context["quote_as_of"] is None
    assert context["retrieved_at"] is None
    assert context["error"] is None


def test_realtime_context_marks_empty_vendor_result_unavailable():
    fetch_realtime_context = getattr(data_collector, "_fetch_realtime_context", None)
    assert callable(fetch_realtime_context)
    with patch(
        "tradingagents.graph.data_collector.is_historical_analysis_date",
        return_value=False,
    ), patch(
        "tradingagents.graph.data_collector.route_to_vendor",
        return_value="{}",
        create=True,
    ):
        context = fetch_realtime_context("600519", "2026-07-29")

    assert context["status"] == "unavailable"
    assert context["source"] is None
    assert context["quote_as_of"] is None
    assert context["error"]


def test_realtime_context_preserves_source_quote_time_and_quote():
    fetch_realtime_context = getattr(data_collector, "_fetch_realtime_context", None)
    assert callable(fetch_realtime_context)
    payload = json.dumps(
        {
            "600519": {
                "price": 1800.0,
                "quote_time": "2026-07-29 14:30:00",
                "source": "sina",
            }
        }
    )
    with patch(
        "tradingagents.graph.data_collector.is_historical_analysis_date",
        return_value=False,
    ), patch(
        "tradingagents.graph.data_collector.route_to_vendor",
        return_value=payload,
        create=True,
    ):
        context = fetch_realtime_context("600519", "2026-07-29")

    assert context["status"] == "available"
    assert context["source"] == "sina"
    assert context["quote_as_of"] == "2026-07-29 14:30:00"
    assert context["quote"]["price"] == 1800.0
    assert context["retrieved_at"]


@pytest.mark.parametrize(
    "payload",
    [
        {"600519": {"source": "sina"}},
        {"600519": {"price": None, "source": "sina"}},
        {"600519": {"price": "1800.0", "source": "sina"}},
        {"600519": {"price": float("nan"), "source": "sina"}},
        {"600519": {"price": float("inf"), "source": "sina"}},
        {"600519": {"price": 1800.0, "source": "unknown"}},
        {"600519": {"price": 1800.0, "source": {"name": "sina"}}},
        {"600519": {"price": 1800.0, "source": "sina", "quote_time": 123}},
        {"600519": [1800.0, "sina"]},
    ],
)
def test_realtime_context_marks_source_or_quote_shape_error_unavailable(payload):
    fetch_realtime_context = getattr(data_collector, "_fetch_realtime_context", None)
    assert callable(fetch_realtime_context)
    with patch(
        "tradingagents.graph.data_collector.is_historical_analysis_date",
        return_value=False,
    ), patch(
        "tradingagents.graph.data_collector.route_to_vendor",
        return_value=json.dumps(payload),
        create=True,
    ):
        context = fetch_realtime_context("600519", "2026-07-29")

    assert context["status"] == "unavailable"
    assert context["error"]
    assert context["source"] is None
    assert context["quote"] is None


def test_market_analyst_separates_daily_and_realtime_context_in_prompt():
    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    seen_messages = []

    async def _astream(messages):
        seen_messages.extend(messages)
        yield SimpleNamespace(content="报告")

    mock_llm.astream = _astream
    collector = DataCollector()
    collector._cache["600519_2026-07-29"] = {
        "stock_data": "date,close,volume\n2026-07-28,10,1000",
        "indicators": {},
        "market_data_context": _context(),
    }
    state = {
        "trade_date": "2026-07-29",
        "company_of_interest": "600519",
        "horizon": "short",
        "user_intent": {},
    }

    result = asyncio.run(create_market_analyst(mock_llm, collector)(state))
    prompt = seen_messages[1].content

    assert "完整日线" in prompt
    assert "实时快照" in prompt
    assert result["market_data_context"] == _context()


def test_horizon_result_and_api_payload_preserve_market_data_context():
    context = _context()
    state = Propagator().create_initial_state(
        "600519",
        "2026-07-29",
        selected_analysts=["news"],
        market_data_context=context,
    )

    horizon_result = TradingAgentsGraph._build_horizon_result(object(), "short", state)
    api_result = _build_result_payload(state)

    assert horizon_result["market_data_context"] == context
    assert api_result["market_data_context"] == context
