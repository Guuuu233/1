# TradingAgents/graph/propagation.py

import copy
from typing import Dict, Any, List, Optional, Mapping, Union
from tradingagents.agents.utils.agent_states import (
    DEFAULT_PROTOCOL_METADATA,
    HorizonRunMetadata,
    InvestDebateState,
    PROTOCOL_VERSION_V1_LEGACY,
    PROTOCOL_VERSION_V2_STRUCTURED,
    RiskDebateState,
    is_v2_debate_enabled,
)
from tradingagents.agents.utils.context_utils import (
    build_market_context,
    infer_instrument_context,
    normalize_user_context,
    summarize_instrument_context,
    summarize_market_context,
    summarize_user_context,
)
from tradingagents.agents.utils.debate_utils import (
    build_empty_risk_debate_state,
    default_round_goal,
)
from .data_collector import default_market_data_context
from .horizon_profile import (
    HORIZON_PROFILE_V1,
    HorizonResolution,
    RESOLUTION_SOURCE_DEFAULT,
    RESOLUTION_SOURCE_EXPLICIT,
    resolve_analysis_horizons,
)
from .intent_parser import bind_research_horizon, clear_research_horizon
from tradingagents.dataflows.social.contracts import create_default_social_data_context


def default_fund_flow_consensus_guard() -> Dict[str, Any]:
    """Return the serialized fail-closed fund-flow source-selection contract."""
    return {
        "blocked": True,
        "direction_allowed": False,
        "status": "not_checked",
        "validation": {"status": "not_checked", "hard_guard": {"blocked": True}},
        "reason": "fund-flow source selection not checked",
    }


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        user_context: Optional[Mapping[str, Any]] = None,
        selected_analysts: Optional[List[str]] = None,
        request_source: str = "api",
        user_intent: Optional[Dict[str, Any]] = None,
        horizon: str = "short",
        market_data_context: Optional[Dict[str, Any]] = None,
        social_data_context: Optional[Dict[str, Any]] = None,
        runtime_config: Optional[Mapping[str, Any]] = None,
        horizon_resolution: Optional[Union[HorizonResolution, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph."""
        instrument_context = infer_instrument_context(company_name)
        market_context = build_market_context(company_name, str(trade_date))
        normalized_user_context = normalize_user_context(user_context)
        user_context_summary = summarize_user_context(normalized_user_context)
        user_prompt_context = (
            f"{summarize_instrument_context(instrument_context)}\n"
            f"{summarize_market_context(market_context)}\n"
            f"{user_context_summary}"
        )
        protocol_meta = copy.deepcopy(DEFAULT_PROTOCOL_METADATA)
        if runtime_config and is_v2_debate_enabled(runtime_config):
            protocol_meta["protocol_version"] = PROTOCOL_VERSION_V2_STRUCTURED
            protocol_meta["protocol_stage"] = "opening"
            protocol_meta["feature_flags"]["v2_debate_enabled"] = True

        # Resolve horizon run metadata according to H-02a contract
        if horizon_resolution is None:
            # Unprovided path: default short even if horizon="medium" slice was passed
            hr = resolve_analysis_horizons()
            resolved = list(hr.resolved)
            resolution_source = hr.resolution_source
            notice = hr.notice
            requested = None
        elif isinstance(horizon_resolution, HorizonResolution):
            resolved = list(horizon_resolution.resolved)
            resolution_source = horizon_resolution.resolution_source
            notice = horizon_resolution.notice
            requested_attr = getattr(horizon_resolution, "requested", None)
            if requested_attr is not None:
                requested = list(requested_attr)
            elif resolution_source == RESOLUTION_SOURCE_EXPLICIT:
                requested = list(resolved)
            else:
                requested = None
        elif isinstance(horizon_resolution, Mapping):
            resolved = list(horizon_resolution.get("resolved") or ["short"])
            resolution_source = str(
                horizon_resolution.get("resolution_source") or RESOLUTION_SOURCE_DEFAULT
            )
            notice = horizon_resolution.get("notice")
            if "requested" in horizon_resolution and horizon_resolution["requested"] is not None:
                requested = list(horizon_resolution["requested"])
            elif resolution_source == RESOLUTION_SOURCE_EXPLICIT:
                requested = list(resolved)
            else:
                requested = None
        else:
            hr = resolve_analysis_horizons(horizon_resolution)
            resolved = list(hr.resolved)
            resolution_source = hr.resolution_source
            notice = hr.notice
            requested = list(resolved) if resolution_source == RESOLUTION_SOURCE_EXPLICIT else None

        primary_eval_offsets = {
            h: HORIZON_PROFILE_V1[h]["primary_eval_offset"]
            for h in resolved
            if h in HORIZON_PROFILE_V1
        }
        cutoff = market_context.get("data_as_of") if market_context else None
        investment_horizon = normalized_user_context.get("investment_horizon") or None

        horizon_run_metadata: Dict[str, Any] = {
            "requested": requested,
            "resolved": resolved,
            "resolution_source": resolution_source,
            "profile_id": "horizon_profile_v1",
            "primary_eval_offsets": primary_eval_offsets,
            "cutoff": cutoff,
            "investment_horizon": investment_horizon,
        }
        if notice is not None:
            horizon_run_metadata["notice"] = notice

        if horizon in resolved:
            bind_research_horizon(horizon)
        else:
            clear_research_horizon()

        investment_debate_state_dict: Dict[str, Any] = {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_speaker": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
            "claims": [],
            "challenges": [],
            "round_messages": [],
            "focus_claim_ids": [],
            "open_claim_ids": [],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "round_summary": "",
            "round_goal": default_round_goal("investment", 1),
            "claim_counter": 0,
            "challenge_counter": 0,
            "attempts": [],
        }
        investment_debate_state_dict.update(protocol_meta)
        state: Dict[str, Any] = {
            "messages": [("human", user_prompt_context)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "instrument_context": instrument_context,
            "market_context": market_context,
            "market_data_context": market_data_context or default_market_data_context(),
            "social_data_context": social_data_context or create_default_social_data_context(requested_as_of=str(trade_date)),
            "fund_flow_consensus_guard": default_fund_flow_consensus_guard(),
            "user_context": normalized_user_context,
            "workflow_context": {
                "context_version": "v1",
                "request_source": request_source,
                "selected_analysts": selected_analysts or [],
                "analysis_baseline_date": str(trade_date),
                "data_as_of": market_context.get("data_as_of"),
            },
            "investment_debate_state": InvestDebateState(investment_debate_state_dict),
            "risk_debate_state": RiskDebateState(build_empty_risk_debate_state()),
            "risk_feedback_state": {
                "retry_count": 0,
                "max_retries": 1,
                "revision_required": False,
                "latest_risk_verdict": "",
                "hard_constraints": [],
                "soft_constraints": [],
                "execution_preconditions": [],
                "de_risk_triggers": [],
                "revision_reason": "",
            },
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            "macro_report": "",
            "smart_money_report": "",
            "volume_price_report": "",
            "event_coverage": {},
            "investment_plan": "",
            "trader_investment_plan": "",
            "final_trade_decision": "",
            "sender": "",
            "metadata": {},
            "analyst_traces": [],
            "horizon": horizon,
            "horizon_run_metadata": horizon_run_metadata,
            "short_term_result": None,
            "medium_term_result": None,
            # D-009 P0-1 placeholders (filled by Research Manager when applicable)
            "decision_status": None,
            "analysis_status": None,
            "trade_action": None,
            "risk_status": None,
            "run_integrity": None,
            "manager_verdict": None,
            "evidence_verification": [],
            "report_manifest": None,
            "integrity_route": None,
        }
        if user_intent is not None:
            state["user_intent"] = user_intent
        return state

    def get_graph_args(self, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
