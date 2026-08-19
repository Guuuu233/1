"""DAV-193: Test debate rounds configuration and 3-round debate flow."""

import tempfile
from unittest.mock import MagicMock, patch

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _base_config(**overrides):
    cfg = {
        "project_dir": tempfile.mkdtemp(prefix="ta-test-"),
        "llm_provider": "openai",
        "quick_think_llm": "default-quick",
        "deep_think_llm": "default-deep",
        "backend_url": "https://api.example.com/v1",
        "api_key": "test-api-key",
    }
    cfg.update(overrides)
    return cfg


def test_default_config_debate_rounds():
    """DEFAULT_CONFIG must default to 3 debate rounds and 3 risk discuss rounds."""
    assert DEFAULT_CONFIG["max_debate_rounds"] == 3
    assert DEFAULT_CONFIG["max_risk_discuss_rounds"] == 3


def test_conditional_logic_default_rounds():
    """ConditionalLogic default constructor arguments must be 3 and 3."""
    logic = ConditionalLogic()
    assert logic.max_debate_rounds == 3
    assert logic.max_risk_discuss_rounds == 3


def test_conditional_logic_3_rounds_debate_termination():
    """With max_debate_rounds=3, debate must continue until count reaches 6 (3 Bull + 3 Bear)."""
    logic = ConditionalLogic(max_debate_rounds=3)

    # Counts 0..5 continue between Bull and Bear
    assert logic.should_continue_debate({"investment_debate_state": {"count": 0, "current_speaker": "Bull"}}) == "Bear Researcher"
    assert logic.should_continue_debate({"investment_debate_state": {"count": 1, "current_speaker": "Bear"}}) == "Bull Researcher"
    assert logic.should_continue_debate({"investment_debate_state": {"count": 2, "current_speaker": "Bull"}}) == "Bear Researcher"
    assert logic.should_continue_debate({"investment_debate_state": {"count": 3, "current_speaker": "Bear"}}) == "Bull Researcher"
    assert logic.should_continue_debate({"investment_debate_state": {"count": 4, "current_speaker": "Bull"}}) == "Bear Researcher"
    assert logic.should_continue_debate({"investment_debate_state": {"count": 5, "current_speaker": "Bear"}}) == "Bull Researcher"

    # Count 6 reaches 2 * 3 -> routes to Research Manager
    assert logic.should_continue_debate({"investment_debate_state": {"count": 6, "current_speaker": "Bull"}}) == "Research Manager"


def test_trading_agents_graph_conditional_logic_wiring_default():
    """TradingAgentsGraph should initialize ConditionalLogic with 3 rounds by default when not in config."""
    with patch("tradingagents.graph.trading_graph.create_llm_client"), \
         patch("tradingagents.graph.trading_graph.FinancialSituationMemory"), \
         patch("tradingagents.graph.trading_graph.GraphSetup"), \
         patch("tradingagents.graph.trading_graph.Propagator"), \
         patch("tradingagents.graph.trading_graph.Reflector"), \
         patch("tradingagents.graph.trading_graph.SignalProcessor"), \
         patch("tradingagents.graph.trading_graph.set_config"), \
         patch("tradingagents.graph.trading_graph.ToolNode"):

        # Test with base config without explicit max_debate_rounds
        graph = TradingAgentsGraph(config=_base_config(), data_collector=MagicMock())
        assert graph.conditional_logic.max_debate_rounds == 3
        assert graph.conditional_logic.max_risk_discuss_rounds == 3


def test_trading_agents_graph_conditional_logic_wiring_custom():
    """TradingAgentsGraph should respect custom config if provided."""
    with patch("tradingagents.graph.trading_graph.create_llm_client"), \
         patch("tradingagents.graph.trading_graph.FinancialSituationMemory"), \
         patch("tradingagents.graph.trading_graph.GraphSetup"), \
         patch("tradingagents.graph.trading_graph.Propagator"), \
         patch("tradingagents.graph.trading_graph.Reflector"), \
         patch("tradingagents.graph.trading_graph.SignalProcessor"), \
         patch("tradingagents.graph.trading_graph.set_config"), \
         patch("tradingagents.graph.trading_graph.ToolNode"):

        graph = TradingAgentsGraph(
            config=_base_config(max_debate_rounds=5, max_risk_discuss_rounds=4),
            data_collector=MagicMock()
        )
        assert graph.conditional_logic.max_debate_rounds == 5
        assert graph.conditional_logic.max_risk_discuss_rounds == 4
