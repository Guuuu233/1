"""DAV-89: config["user_id"] must reach TradingAgentsGraph so per-role model
bindings (resolve_all_roles) are consumed instead of silently falling back to
the default quick/deep models.

The production fix lives in api.main._build_runtime_config (it now writes
user_id into the returned config); these tests pin the graph contract that
consumes it.
"""

import tempfile
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from api.services import role_routing_service
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _base_config(**overrides):
    cfg = {
        "project_dir": tempfile.mkdtemp(prefix="ta-role-llms-"),
        "llm_provider": "openai",
        "quick_think_llm": "default-quick",
        "deep_think_llm": "default-deep",
        "backend_url": "https://api.example.com/v1",
        "api_key": "test-api-key",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "max_recur_limit": 10,
        "google_thinking_level": None,
        "openai_reasoning_effort": None,
    }
    cfg.update(overrides)
    return cfg


def _all_role_bindings():
    """Return a resolve_all_roles-style dict covering every role."""
    bindings = {}
    for role in role_routing_service.ALL_ROLES:
        bindings[role] = {
            "role_key": role,
            "resolved_via": "role_binding",
            "fallback_used": False,
            "provider_type": "openai",
            "model_name": f"{role}-bound-model",
            "base_url": "https://api.example.com/v1",
            "api_key": "test-api-key",
            "temperature": None,
            "max_tokens": None,
            "profile_id": "p1",
            "provider_id": "pr1",
            "profile_display_name": role,
            "provider_display_name": "openai",
        }
    return bindings


class _FakeDbCtx:
    """Context manager standing in for api.database.get_db_ctx."""

    def __enter__(self):
        self.session = MagicMock()
        return self.session

    def __exit__(self, *_exc):
        return False


class _ClientRecorder:
    """Records create_llm_client(model=...) calls and returns fake clients."""

    def __init__(self):
        self.calls = []

    def __call__(self, provider, model, base_url=None, **kwargs):
        self.calls.append({"provider": provider, "model": model, "base_url": base_url})
        client = MagicMock()
        client.get_llm.return_value = MagicMock()
        return client


def _install_graph_patches():
    """Context managers that keep TradingAgentsGraph.__init__ side-effect-free.

    Returns (recorder, patches) — recorder is a _ClientRecorder that also needs
    to be installed as the create_llm_client patch.
    """
    recorder = _ClientRecorder()
    return recorder, (
        patch("tradingagents.graph.trading_graph.create_llm_client", recorder),
        patch("tradingagents.graph.trading_graph.FinancialSituationMemory"),
        patch("tradingagents.graph.trading_graph.GraphSetup"),
        patch("tradingagents.graph.trading_graph.ConditionalLogic"),
        patch("tradingagents.graph.trading_graph.Propagator"),
        patch("tradingagents.graph.trading_graph.Reflector"),
        patch("tradingagents.graph.trading_graph.SignalProcessor"),
        patch("tradingagents.graph.trading_graph.set_config"),
        patch("tradingagents.graph.trading_graph.ToolNode"),
    )


def test_config_with_user_id_resolves_roles_and_builds_role_llms_from_bindings():
    bindings = _all_role_bindings()
    bindings["market"]["model_name"] = "gemini-3.6-flash-high"
    bindings["research_manager"]["model_name"] = "opencode/deep-bound"

    recorder, patches = _install_graph_patches()
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        mock_resolve = stack.enter_context(
            patch("api.services.role_routing_service.resolve_all_roles", return_value=bindings)
        )
        stack.enter_context(patch("api.database.get_db_ctx", _FakeDbCtx))
        graph = TradingAgentsGraph(config=_base_config(user_id="user-abc"), data_collector=MagicMock())

    # resolve_all_roles is called with the config's user_id + the full runtime_config.
    mock_resolve.assert_called_once()
    _, call_kwargs = mock_resolve.call_args
    assert mock_resolve.call_args.args[1] == "user-abc"
    assert call_kwargs["runtime_config"]["user_id"] == "user-abc"

    # The role_llms are built from the bound models, not the defaults.
    assert graph.role_resolved_configs["market"]["model_name"] == "gemini-3.6-flash-high"
    bound_models = [c["model"] for c in recorder.calls if c["model"] == "gemini-3.6-flash-high"]
    assert bound_models, "market role client must be created with the bound model"
    # quick_thinking_llm is the market-bound client (role_llms wins over default).
    assert graph.quick_thinking_llm is graph.role_llms["market"]


def test_config_without_user_id_skips_role_resolution_and_falls_back_to_defaults():
    recorder, patches = _install_graph_patches()
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        mock_resolve = stack.enter_context(
            patch("api.services.role_routing_service.resolve_all_roles")
        )
        stack.enter_context(patch("api.database.get_db_ctx", _FakeDbCtx))
        graph = TradingAgentsGraph(config=_base_config(), data_collector=MagicMock())

    mock_resolve.assert_not_called()
    assert graph.role_resolved_configs == {}
    # No bound models anywhere — every client uses the default quick/deep models.
    bound_models = [c["model"] for c in recorder.calls if "bound-model" in str(c["model"])]
    assert bound_models == []
    assert graph.quick_thinking_llm is not None
