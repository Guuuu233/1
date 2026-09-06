"""Tests for H-04a: Research horizon and analyst observation window separation."""
import pytest

from tradingagents.graph.horizon_profile import (
    HorizonResolution,
    resolve_analysis_horizons,
)
from tradingagents.graph.intent_parser import (
    bind_research_horizon,
    build_horizon_context,
    clear_research_horizon,
    get_bound_research_horizon,
    research_horizon_context,
    reset_research_horizon,
)
from tradingagents.graph.propagation import Propagator


@pytest.fixture(autouse=True)
def clean_horizon_binding():
    """Ensure contextvar binding is always clean before and after tests."""
    clear_research_horizon()
    yield
    clear_research_horizon()


class TestHorizonAnalystContextContract:
    """Test suite for H-04a contract: research horizon vs node observation window."""

    def test_bound_medium_with_short_observation_window(self):
        """Binding medium displays medium research horizon and short observation window,
        without claiming the whole task is short-term.
        """
        bind_research_horizon("medium")
        ctx = build_horizon_context("short", ["量价关系"], ["能否突破"])

        # Both must be present and distinct
        assert "本次研究档：中线（1-3月，基本面主导）" in ctx
        assert "本节点专业观察窗：短线（1-2周，技术面主导）" in ctx
        assert "量价关系" in ctx
        assert "能否突破" in ctx

        # Prohibited legacy phrasing: cannot impersonate entire task as short-term
        assert "当前分析维度" not in ctx
        assert "当前分析维度：短线" not in ctx
        assert "短线任务" not in ctx
        assert "短线研究" not in ctx

    def test_bound_short_with_medium_observation_window(self):
        """Binding short displays short research horizon and medium observation window (symmetric)."""
        bind_research_horizon("short")
        ctx = build_horizon_context("medium", ["财务报表"], ["盈利能力"])

        # Both must be present and distinct
        assert "本次研究档：短线（1-2周，技术面主导）" in ctx
        assert "本节点专业观察窗：中线（1-3月，基本面主导）" in ctx
        assert "财务报表" in ctx
        assert "盈利能力" in ctx

        assert "当前分析维度" not in ctx
        assert "中线任务" not in ctx
        assert "中线研究" not in ctx

    def test_unbound_research_horizon_shows_explicit_unbound(self):
        """When unbound, observation window is visible but research horizon is explicitly unbound."""
        clear_research_horizon()
        ctx = build_horizon_context("short", [], [])

        assert "本次研究档：未绑定" in ctx
        assert "本节点专业观察窗：短线（1-2周，技术面主导）" in ctx
        # Must not fake research horizon using observation window
        assert "本次研究档：短线" not in ctx
        assert "当前分析维度" not in ctx

    def test_create_initial_state_medium_binds_research_horizon_for_analyst(self):
        """create_initial_state with horizon='medium' binds research horizon so
        analyst calls build_horizon_context('short', ...) without changing call signature.
        """
        res = resolve_analysis_horizons(["medium"])
        p = Propagator()
        state = p.create_initial_state(
            "600519",
            "2024-01-15",
            horizon="medium",
            horizon_resolution=res,
        )

        assert state["horizon"] == "medium"
        assert state["horizon_run_metadata"]["resolved"] == ["medium"]
        assert get_bound_research_horizon() == "medium"

        # Analyst invocation without signature modification
        analyst_ctx = build_horizon_context("short", [])
        assert "本次研究档：中线（1-3月，基本面主导）" in analyst_ctx
        assert "本节点专业观察窗：短线（1-2周，技术面主导）" in analyst_ctx
        assert "当前分析维度" not in analyst_ctx

    def test_create_initial_state_mismatched_horizon_clears_binding(self):
        """When horizon is not in resolved, create_initial_state must not bind it."""
        # Unprovided resolution defaults to resolved=["short"]
        p = Propagator()
        state = p.create_initial_state(
            "600519",
            "2024-01-15",
            horizon="medium",
            horizon_resolution=None,
        )
        assert state["horizon_run_metadata"]["resolved"] == ["short"]
        assert get_bound_research_horizon() is None

        ctx = build_horizon_context("short", [])
        assert "本次研究档：未绑定" in ctx

    def test_explicit_kwarg_priority_over_binding(self):
        """Explicit research_horizon kwarg takes precedence over thread binding."""
        bind_research_horizon("short")
        ctx = build_horizon_context("short", [], [], research_horizon="medium")
        assert "本次研究档：中线（1-3月，基本面主导）" in ctx
        assert "本节点专业观察窗：短线（1-2周，技术面主导）" in ctx

        # run_horizon alias works too
        ctx2 = build_horizon_context("medium", [], [], run_horizon="short")
        assert "本次研究档：短线（1-2周，技术面主导）" in ctx2
        assert "本节点专业观察窗：中线（1-3月，基本面主导）" in ctx2

    def test_context_manager_and_token_cleanup(self):
        """Context manager and reset_research_horizon cleanly restore previous state."""
        assert get_bound_research_horizon() is None

        token = bind_research_horizon("medium")
        assert get_bound_research_horizon() == "medium"
        reset_research_horizon(token)
        assert get_bound_research_horizon() is None

        with research_horizon_context("short"):
            assert get_bound_research_horizon() == "short"
            ctx = build_horizon_context("medium", [])
            assert "本次研究档：短线（1-2周，技术面主导）" in ctx
        assert get_bound_research_horizon() is None

    def test_bull_bear_same_medium_horizon_allowed(self):
        """When bull/bear researchers pass state['horizon']='medium' while bound to 'medium',
        both lines showing medium is allowed and valid.
        """
        bind_research_horizon("medium")
        ctx = build_horizon_context("medium", [], [], agent_type="bull")
        assert "本次研究档：中线（1-3月，基本面主导）" in ctx
        assert "本节点专业观察窗：中线（1-3月，基本面主导）" in ctx
        assert "次要" not in ctx

    def test_english_prompt_template(self, monkeypatch):
        """English template properly distinguishes research horizon and observation window."""
        from tradingagents.dataflows.config import get_config

        current_cfg = get_config()
        monkeypatch.setattr(
            "tradingagents.graph.intent_parser.get_config",
            lambda: {**current_cfg, "prompt_language": "en"},
        )

        bind_research_horizon("medium")
        ctx = build_horizon_context("short", ["volume-price"], ["target reached?"])
        assert "Research Horizon: Medium-term (1-3 months, fundamentals-driven)" in ctx
        assert "Node Observation Window: Short-term (1-2 weeks, technicals-driven)" in ctx
        assert "Current horizon" not in ctx

        clear_research_horizon()
        ctx_unbound = build_horizon_context("short", [], [])
        assert "Research Horizon: Unbound" in ctx_unbound
        assert "Node Observation Window: Short-term (1-2 weeks, technicals-driven)" in ctx_unbound
