import contextvars
import json
import operator
import re
from typing import Annotated, Any, List, Mapping, Tuple

from typing_extensions import Optional, TypedDict
from langgraph.graph import MessagesState

# ContextVar used to pass the AgentProgressTracker into async graph nodes
# without putting it in the LangGraph state (which would require serialization).
# Set by the API layer before each graph.astream() call; read by analyst nodes.
current_tracker_var: contextvars.ContextVar = contextvars.ContextVar(
    "current_tracker", default=None
)


def extract_verdict(text: str) -> Tuple[str, str]:
    """Extract VERDICT block from analyst output. Returns (direction, confidence)."""
    m = re.search(r'<!--\s*VERDICT:\s*(\{.*?\})\s*-->', text or "", re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1))
            return d.get("direction", "中性"), "中"
        except Exception:
            pass
    return "中性", "低"


class UserIntent(TypedDict, total=False):
    raw_query: str
    ticker: str
    horizons: List[str]
    focus_areas: List[str]
    specific_questions: List[str]
    user_context: "UserContext"


class TraceItem(TypedDict, total=False):
    agent: str
    horizon: str
    data_window: str
    key_finding: str
    verdict: str
    confidence: str
    source_status: str
    source_mode: str
    bundle_id: str
    direction_allowed: bool
    reason_codes: list[str]
    evidence_refs: list[str]
    financial_period_compliance: dict[str, Any]


class InstrumentContext(TypedDict, total=False):
    symbol: Annotated[str, "Normalized symbol"]
    security_name: Annotated[str, "Display name or fallback symbol"]
    market_country: Annotated[str, "Market country such as CN or US"]
    exchange: Annotated[str, "Exchange code"]
    currency: Annotated[str, "Trading currency"]
    asset_type: Annotated[str, "Asset type"]
    industry: Annotated[Optional[str], "Normalized industry or sector name"]


class MarketContext(TypedDict):
    trade_date: Annotated[str, "Requested trade date"]
    analysis_baseline_date: Annotated[str, "Analysis baseline date"]
    timezone: Annotated[str, "Market timezone"]
    market_country: Annotated[str, "Market country"]
    exchange: Annotated[str, "Exchange code"]
    market_session: Annotated[str, "Current session for the requested trade date"]
    market_is_open: Annotated[bool, "Whether the market is currently open"]
    analysis_mode: Annotated[str, "Analysis mode such as pre_market, intraday, post_market, t_plus_1"]
    data_as_of: Annotated[str, "Latest date the analysis should treat as confirmed data"]
    session_note: Annotated[str, "Explanation for the current session inference"]


class UserContext(TypedDict, total=False):
    objective: Annotated[str, "User's desired action"]
    risk_profile: Annotated[str, "User's risk profile"]
    investment_horizon: Annotated[str, "User's intended holding horizon"]
    cash_available: Annotated[float, "Available cash"]
    current_position: Annotated[float, "Current position size"]
    current_position_pct: Annotated[float, "Current position percentage"]
    average_cost: Annotated[float, "Average holding cost"]
    max_loss_pct: Annotated[float, "Maximum tolerated loss percentage"]
    constraints: Annotated[list[str], "Hard trading constraints"]
    user_notes: Annotated[str, "Additional user notes"]


class WorkflowContext(TypedDict):
    context_version: Annotated[str, "Workflow context version"]
    request_source: Annotated[str, "Request origin such as api or chat"]
    selected_analysts: Annotated[list[str], "Requested analyst roster"]
    analysis_baseline_date: Annotated[str, "Analysis baseline date"]
    data_as_of: Annotated[str | None, "Confirmed data cutoff"]


class ManagerVerdict(TypedDict, total=False):
    direction: Annotated[str, "Directional verdict"]
    winner: Annotated[str, "Debate winner: bull / bear / tie"]
    reason: Annotated[str, "Core rationale for verdict"]
    position_pct: Annotated[Optional[float | int | str], "Recommended position percentage"]
    entry: Annotated[Optional[str], "Recommended entry price range"]
    target: Annotated[Optional[str], "Target price"]
    stop_loss: Annotated[Optional[str], "Stop loss price"]
    upside: Annotated[Optional[float | int | str], "Upside percentage"]
    downside: Annotated[Optional[float | int | str], "Downside percentage"]
    odds: Annotated[Optional[float | int | str], "Odds ratio"]
    adopted_claim_ids: Annotated[list[str], "Fully adopted claim IDs"]
    partially_adopted_claims: Annotated[list[str], "Partially adopted claim IDs with verified sub-conclusions"]
    rejected_claim_ids: Annotated[list[str], "Rejected claim IDs"]
    excluded_evidence: Annotated[list[str], "Excluded unverified/contradicted evidence items"]
    claim_evidence_summary: Annotated[dict[str, Any], "Deterministic claim evidence aggregation and coverage summary"]
    consistency_check_passed: Annotated[bool, "Whether consistency check passed"]
    failed_checks: Annotated[list[str], "List of failed consistency checks"]
    decision_status: Annotated[dict[str, Any], "D-009 analysis/trade status payload"]
    analysis_status: Annotated[str, "VALID|PARTIAL|ABSTAIN|INVALID_RUN|DATA_ERROR"]
    trade_action: Annotated[str, "BUY|SELL|HOLD|WAIT|NO_TRADE"]
    risk_status: Annotated[str, "OK|ELEVATED|BLOCKED|UNKNOWN"]


class Challenge(TypedDict, total=False):
    challenge_id: Annotated[str, "Unique challenge identifier (e.g. CH-1)"]
    speaker: Annotated[str, "Speaker label"]
    speaker_key: Annotated[str, "Speaker key (Bull / Bear)"]
    stance: Annotated[str, "Speaker stance"]
    target_claim_id: Annotated[str, "Target opponent claim id being challenged"]
    weakest_point: Annotated[str, "Identified flaw or weakest point in target claim"]
    evidence: Annotated[list[str], "Factual evidence items supporting challenge"]
    severity: Annotated[str, "Severity level: fatal, major, minor"]
    status: Annotated[str, "Challenge status: open, addressed, resolved, rejected"]
    message_index: Annotated[int, "Message index where challenge was proposed"]
    debate_round: Annotated[int, "Debate round index"]
    stage: Annotated[str, "Debate stage: challenge"]
    evidence_status: Annotated[str, "Evidence verification status: verified, unsupported, contradicted"]


PROTOCOL_VERSION_V1_LEGACY = "v1_legacy"
PROTOCOL_VERSION_V2_STRUCTURED = "v2_structured_disagreement"

DEFAULT_FEATURE_FLAGS: dict[str, bool] = {
    "v2_debate_enabled": False,
    "shadow_credit_enabled": True,
    "credit_weighting_enabled": False,
}

DEFAULT_PROTOCOL_METADATA: dict[str, Any] = {
    "protocol_version": PROTOCOL_VERSION_V1_LEGACY,
    "protocol_stage": "opening",
    "tiebreak_skipped": False,
    "debate_degenerate": False,
    "data_utilization_metrics": {},
    "challenge_verification": [],
    "shadow_credit_metrics": {},
    "feature_flags": {
        "v2_debate_enabled": False,
        "shadow_credit_enabled": True,
        "credit_weighting_enabled": False,
    },
}


def get_protocol_metadata(state_or_result: Any) -> dict[str, Any]:
    """Extract and normalize protocol metadata from result_data or debate_state.

    Missing fields in legacy reports are defaulted to v1_legacy without error.
    """
    if not isinstance(state_or_result, dict):
        return {
            "protocol_version": PROTOCOL_VERSION_V1_LEGACY,
            "protocol_stage": "opening",
            "tiebreak_skipped": False,
            "debate_degenerate": False,
            "data_utilization_metrics": {},
            "challenge_verification": [],
            "shadow_credit_metrics": {},
            "feature_flags": dict(DEFAULT_FEATURE_FLAGS),
        }

    # Check root level first, then nested investment_debate_state
    inv_state = state_or_result.get("investment_debate_state")
    if not isinstance(inv_state, dict):
        inv_state = state_or_result

    raw_version = inv_state.get("protocol_version") or state_or_result.get("protocol_version")
    protocol_version = str(raw_version) if raw_version else PROTOCOL_VERSION_V1_LEGACY

    raw_stage = inv_state.get("protocol_stage") or state_or_result.get("protocol_stage")
    protocol_stage = str(raw_stage) if raw_stage else "opening"

    tiebreak_skipped = bool(
        inv_state.get("tiebreak_skipped")
        if "tiebreak_skipped" in inv_state
        else state_or_result.get("tiebreak_skipped", False)
    )

    debate_degenerate = bool(
        inv_state.get("debate_degenerate")
        if "debate_degenerate" in inv_state
        else state_or_result.get("debate_degenerate", False)
    )
    if not debate_degenerate and is_v2_debate_enabled(state_or_result):
        try:
            from tradingagents.agents.utils.debate_utils import detect_debate_degenerate
            debate_degenerate = detect_debate_degenerate(state_or_result)
        except Exception:
            pass

    data_metrics = (
        inv_state.get("data_utilization_metrics")
        if isinstance(inv_state.get("data_utilization_metrics"), dict)
        else state_or_result.get("data_utilization_metrics")
    )
    if not isinstance(data_metrics, dict):
        data_metrics = {}

    challenge_verif = (
        inv_state.get("challenge_verification")
        if isinstance(inv_state.get("challenge_verification"), list)
        else state_or_result.get("challenge_verification")
    )
    if not isinstance(challenge_verif, list):
        challenge_verif = []

    shadow_metrics = (
        inv_state.get("shadow_credit_metrics")
        if isinstance(inv_state.get("shadow_credit_metrics"), dict)
        else state_or_result.get("shadow_credit_metrics")
    )
    if not isinstance(shadow_metrics, dict):
        shadow_metrics = {}

    raw_flags = (
        inv_state.get("feature_flags")
        if isinstance(inv_state.get("feature_flags"), dict)
        else state_or_result.get("feature_flags")
    )
    feature_flags = dict(DEFAULT_FEATURE_FLAGS)
    if isinstance(raw_flags, dict):
        for k in DEFAULT_FEATURE_FLAGS:
            if k in raw_flags:
                feature_flags[k] = bool(raw_flags[k])

    return {
        "protocol_version": protocol_version,
        "protocol_stage": protocol_stage,
        "tiebreak_skipped": tiebreak_skipped,
        "debate_degenerate": debate_degenerate,
        "data_utilization_metrics": data_metrics,
        "challenge_verification": challenge_verif,
        "shadow_credit_metrics": shadow_metrics,
        "feature_flags": feature_flags,
    }


def normalize_protocol_metadata(state_or_result: Any) -> dict[str, Any]:
    """Alias for get_protocol_metadata with normalized output guarantees."""
    return get_protocol_metadata(state_or_result)


def is_v2_debate_enabled(state_or_config: Any) -> bool:
    """Return True if v2_debate_enabled is explicitly enabled in state, config, or feature_flags."""
    if not isinstance(state_or_config, (dict, Mapping)):
        return False

    # 1. Direct check in nested investment_debate_state if present
    inv_state = state_or_config.get("investment_debate_state")
    if isinstance(inv_state, (dict, Mapping)):
        flags = inv_state.get("feature_flags")
        if isinstance(flags, (dict, Mapping)) and flags.get("v2_debate_enabled") is not None:
            return bool(flags["v2_debate_enabled"])
        if inv_state.get("v2_debate_enabled") is not None:
            return bool(inv_state.get("v2_debate_enabled"))
        if inv_state.get("protocol_version") == PROTOCOL_VERSION_V2_STRUCTURED:
            return True

    # 2. Check in top-level feature_flags dict
    flags = state_or_config.get("feature_flags")
    if isinstance(flags, (dict, Mapping)) and flags.get("v2_debate_enabled") is not None:
        return bool(flags["v2_debate_enabled"])

    # 3. Direct key at top level
    if state_or_config.get("v2_debate_enabled") is not None:
        return bool(state_or_config.get("v2_debate_enabled"))

    # 4. Direct protocol_version at top level
    if state_or_config.get("protocol_version") == PROTOCOL_VERSION_V2_STRUCTURED:
        return True

    return False


def is_credit_weighting_enabled(state_or_config: Any) -> bool:
    """Return True if credit_weighting_enabled is explicitly enabled in state, config, or feature_flags."""
    if not isinstance(state_or_config, (dict, Mapping)):
        return False

    # 1. Direct check in nested investment_debate_state if present
    inv_state = state_or_config.get("investment_debate_state")
    if isinstance(inv_state, (dict, Mapping)):
        flags = inv_state.get("feature_flags")
        if isinstance(flags, (dict, Mapping)) and flags.get("credit_weighting_enabled") is not None:
            return bool(flags["credit_weighting_enabled"])
        if inv_state.get("credit_weighting_enabled") is not None:
            return bool(inv_state.get("credit_weighting_enabled"))

    # 2. Check in top-level feature_flags dict
    flags = state_or_config.get("feature_flags")
    if isinstance(flags, (dict, Mapping)) and flags.get("credit_weighting_enabled") is not None:
        return bool(flags["credit_weighting_enabled"])

    # 3. Direct key at top level
    if state_or_config.get("credit_weighting_enabled") is not None:
        return bool(state_or_config.get("credit_weighting_enabled"))

    return False


class InvestDebateState(TypedDict):
    bull_history: Annotated[str, "Bullish conversation history"]
    bear_history: Annotated[str, "Bearish conversation history"]
    history: Annotated[str, "Conversation history"]
    current_speaker: Annotated[str, "Speaker that spoke last"]
    current_response: Annotated[str, "Latest response"]

    # ── Parallel Rebuttal Fields ──────────────────────────────────────
    bull_initial: Annotated[str, "Bull's initial opening statement"]
    bear_initial: Annotated[str, "Bear's initial opening statement"]
    bull_rebuttal: Annotated[str, "Bull's rebuttal to Bear's initial"]
    bear_rebuttal: Annotated[str, "Bear's rebuttal to Bull's initial"]
    # ──────────────────────────────────────────────────────────────────

    judge_decision: Annotated[str, "Final judge decision"]
    count: Annotated[int, "Length of the current conversation"]
    claims: Annotated[list[dict[str, Any]], "Tracked research claims"]
    challenges: Annotated[list[dict[str, Any]], "Tracked debate challenges"]
    round_messages: Annotated[list[dict[str, Any]], "Tracked round-by-round messages"]
    focus_claim_ids: Annotated[list[str], "Claim ids that must be answered in the next round"]
    open_claim_ids: Annotated[list[str], "Claim ids still open"]
    resolved_claim_ids: Annotated[list[str], "Claim ids considered resolved"]
    unresolved_claim_ids: Annotated[list[str], "Claim ids still materially disputed"]
    round_summary: Annotated[str, "Summary of the latest debate round"]
    round_goal: Annotated[str, "Current round objective"]
    claim_counter: Annotated[int, "Claim counter for unique ids"]
    challenge_counter: Annotated[int, "Challenge counter for unique challenge ids"]
    manager_verdict: Annotated[dict[str, Any], "Structured manager verdict"]
    evidence_verification: Annotated[list[dict[str, Any]], "Deterministic evidence factual verification results"]
    report_manifest: Annotated[dict[str, Any], "Input report manifest for seven analysts"]
    attempts: Annotated[list[dict[str, Any]], "Tracked message attempt records including unaccepted attempts"]
    blocked: Annotated[bool, "Whether debate protocol validation failed and blocked progression"]
    parse_status: Annotated[str, "Latest debate state parse status"]
    block_reason: Annotated[str, "Reason if debate state is blocked"]

    # ── Protocol & Metrics Fields (P1-M) ──────────────────────────────
    protocol_version: Annotated[str, "Protocol version: v1_legacy or v2_structured_disagreement"]
    protocol_stage: Annotated[str, "Protocol stage: opening, challenge, tiebreak, manager"]
    tiebreak_skipped: Annotated[bool, "Whether tiebreak round was skipped"]
    debate_degenerate: Annotated[bool, "Whether debate belief trajectories degenerated"]
    data_utilization_metrics: Annotated[dict[str, Any], "Metrics on data utilization across seven reports"]
    challenge_verification: Annotated[list[dict[str, Any]], "Deterministic verification results for challenges"]
    shadow_credit_metrics: Annotated[dict[str, Any], "Shadow credit metrics collected during debate"]
    feature_flags: Annotated[dict[str, Any], "Feature flags controlling debate protocol and scoring"]


class RiskDebateState(TypedDict):
    aggressive_history: Annotated[str, "Aggressive analyst history"]
    conservative_history: Annotated[str, "Conservative analyst history"]
    neutral_history: Annotated[str, "Neutral analyst history"]
    history: Annotated[str, "Conversation history"]
    latest_speaker: Annotated[str, "Analyst that spoke last"]
    current_aggressive_response: Annotated[str, "Latest response by the aggressive analyst"]
    current_conservative_response: Annotated[str, "Latest response by the conservative analyst"]
    current_neutral_response: Annotated[str, "Latest response by the neutral analyst"]
    judge_decision: Annotated[str, "Judge decision"]
    count: Annotated[int, "Length of the current conversation"]
    claims: Annotated[list[dict[str, Any]], "Tracked risk claims"]
    focus_claim_ids: Annotated[list[str], "Risk claim ids that must be answered next"]
    open_claim_ids: Annotated[list[str], "Risk claim ids still open"]
    resolved_claim_ids: Annotated[list[str], "Risk claim ids considered resolved"]
    unresolved_claim_ids: Annotated[list[str], "Risk claim ids still materially disputed"]
    round_summary: Annotated[str, "Summary of the latest debate round"]
    round_goal: Annotated[str, "Current round objective"]
    claim_counter: Annotated[int, "Claim counter for unique ids"]


class RiskFeedbackState(TypedDict):
    retry_count: Annotated[int, "How many times the trader has been sent back for revision"]
    max_retries: Annotated[int, "Maximum number of allowed revisions"]
    revision_required: Annotated[bool, "Whether the trader must revise the plan"]
    latest_risk_verdict: Annotated[str, "Risk judge verdict such as pass, revise, reject"]
    hard_constraints: Annotated[list[str], "Non-negotiable constraints from the risk judge"]
    soft_constraints: Annotated[list[str], "Advisory constraints from the risk judge"]
    execution_preconditions: Annotated[list[str], "Conditions that must hold before execution"]
    de_risk_triggers: Annotated[list[str], "Triggers that require immediate de-risking"]
    revision_reason: Annotated[str, "Why the plan was sent back"]


class HorizonRunMetadata(TypedDict, total=False):
    requested: Annotated[Optional[List[str]], "User-requested horizon list or None if unprovided"]
    resolved: Annotated[List[str], "Resolved horizon list, e.g. ['short'] or ['short', 'medium']"]
    resolution_source: Annotated[str, "Resolution source: default, explicit, or legacy"]
    profile_id: Annotated[str, "Horizon profile contract identifier"]
    primary_eval_offsets: Annotated[dict[str, int], "Planned evaluation step offsets per resolved horizon"]
    cutoff: Annotated[Optional[str], "Confirmed data cutoff date identical to workflow_context.data_as_of"]
    investment_horizon: Annotated[Optional[str], "User's intended holding horizon from user_context"]
    notice: Annotated[Optional[str], "Optional non-blocking notice regarding horizon selection"]
    evaluation_eligible: Annotated[bool, "Whether run is eligible for evaluation (defaults to False/omitted)"]


class GameTheorySignals(TypedDict, total=False):
    board: Annotated[Optional[str], "Board context or sector flow"]
    players: Annotated[Optional[List[str]], "List of market players"]
    player_states: Annotated[Optional[dict[str, str]], "Posture of each player"]
    likely_actions: Annotated[Optional[dict[str, list[str]]], "Likely actions per player"]
    dominant_strategy: Annotated[Optional[str], "Dominant strategic recommendation"]
    fragile_equilibrium: Annotated[Optional[str], "Equilibrium fragility assessment"]
    counter_consensus_signal: Annotated[Optional[str], "Counter-consensus warning or signal"]
    confidence: Annotated[Optional[float], "Confidence score in [0.0, 1.0]"]
    data_status: Annotated[Optional[str], "available | partial | unavailable"]


class AgentState(MessagesState):
    company_of_interest: Annotated[str, "Company that we are interested in trading"]
    trade_date: Annotated[str, "What date we are trading at"]
    sender: Annotated[str, "Agent that sent this message"]

    instrument_context: Annotated[InstrumentContext, "Normalized instrument context"]
    market_context: Annotated[MarketContext, "Market session and timing context"]
    market_data_context: Annotated[dict[str, Any], "Completed daily bars and independent realtime snapshot"]
    social_data_context: Annotated[dict[str, Any], "Social media sentiment bundle and evidence context"]
    fund_flow_consensus_guard: Annotated[dict[str, Any], "Fail-closed fund-flow direction guard and validation"]
    user_context: Annotated[UserContext, "User-specific holdings and constraints"]
    workflow_context: Annotated[WorkflowContext, "Workflow metadata for the current run"]

    market_report: Annotated[str, "Report from the Market Analyst"]
    sentiment_report: Annotated[str, "Report from the Social Media Analyst"]
    news_report: Annotated[str, "Report from the News Researcher of current world affairs"]
    fundamentals_report: Annotated[str, "Report from the Fundamentals Researcher"]

    investment_debate_state: Annotated[
        InvestDebateState, "Current state of the debate on if to invest or not"
    ]
    investment_plan: Annotated[str, "Plan generated by the Analyst"]
    trader_investment_plan: Annotated[str, "Plan generated by the Trader"]

    risk_debate_state: Annotated[
        RiskDebateState, "Current state of the debate on evaluating risk"
    ]
    risk_feedback_state: Annotated[
        RiskFeedbackState, "Risk-judge feedback used for trader revision"
    ]
    final_trade_decision: Annotated[str, "Final decision made by the Risk Analysts"]

    macro_report: Annotated[str, "Report from the Macro/Sector Analyst"]
    smart_money_report: Annotated[str, "Report from the Smart Money Analyst"]
    volume_price_report: Annotated[str, "Report from the Volume Price Analyst"]
    game_theory_report: Annotated[Optional[str], "Report from the Game Theory Analyst"]
    game_theory_signals: Annotated[Optional[dict[str, Any]], "Structured game theory signals"]
    event_coverage: Annotated[dict[str, Any], "Structured news event evidence and coverage summary"]
    user_intent: Annotated[Optional[UserIntent], "Parsed user intent from natural language"]
    horizon: Annotated[str, "Current analysis horizon: short or medium"]
    horizon_run_metadata: Annotated[
        HorizonRunMetadata, "Analysis horizon run metadata and resolution profile"
    ]
    analyst_traces: Annotated[List[TraceItem], operator.add]
    manager_verdict: Annotated[dict[str, Any], "Structured research manager verdict"]
    evidence_verification: Annotated[list[dict[str, Any]], "Deterministic evidence factual verification results"]
    report_manifest: Annotated[dict[str, Any], "Input report manifest for seven analysts"]
    # D-009 P0-1: must be first-class state keys or LangGraph may drop them.
    decision_status: Annotated[dict[str, Any], "Canonical analysis/trade status payload"]
    analysis_status: Annotated[str, "VALID|PARTIAL|ABSTAIN|INVALID_RUN|DATA_ERROR"]
    trade_action: Annotated[str, "BUY|SELL|HOLD|WAIT|NO_TRADE"]
    risk_status: Annotated[str, "OK|ELEVATED|BLOCKED|UNKNOWN"]
    run_integrity: Annotated[dict[str, Any], "RunIntegrity summary for required analysts"]
    integrity_route: Annotated[str, "Post-analyst integrity route: Bull Researcher | END"]
    short_term_result: Annotated[Optional[dict], "Final short-term analysis result"]
    medium_term_result: Annotated[Optional[dict], "Final medium-term analysis result"]
    metadata: Annotated[dict[str, Any], "Optional runtime metadata"]


import logging as _logging

_degrade_logger = _logging.getLogger(__name__)

_DEGRADE_WHITESPACE_RATIO = 0.50
_DEGRADE_MAX_BYTES = 100_000


def check_llm_output_degraded(text: str, agent_name: str) -> bool:
    """Return True and log WARNING if text looks like a degenerate LLM output.

    Degraded criteria (either triggers):
    - Whitespace (space/tab/newline) ratio > 50 %
    - Total byte length > 100 KB

    Caller should replace the text with an explicit failure message when True.
    """
    n = len(text)
    if n == 0:
        return False
    ws = sum(1 for c in text if c in " \t\n\r")
    ratio = ws / n
    too_much_ws = ratio > _DEGRADE_WHITESPACE_RATIO
    too_long = len(text.encode("utf-8", errors="replace")) > _DEGRADE_MAX_BYTES
    if too_much_ws or too_long:
        _degrade_logger.warning(
            "[%s] Degenerate LLM output detected: len=%d bytes, whitespace_ratio=%.3f "
            "(limits: ws>%.0f%% or bytes>%d). Preview: %s",
            agent_name, n, ratio,
            _DEGRADE_WHITESPACE_RATIO * 100, _DEGRADE_MAX_BYTES,
            repr(text[:500]),
        )
        return True
    return False


_STREAM_DEGRADE_WINDOW = 2000


def check_stream_chunk_degraded(buffer_tail: str, agent_name: str) -> bool:
    """Return True if the last _STREAM_DEGRADE_WINDOW chars are all whitespace.

    Call this inside the astream loop with the trailing slice of full_content.
    When True, the caller should break the loop immediately.
    """
    n = len(buffer_tail)
    if n < _STREAM_DEGRADE_WINDOW:
        return False
    tail = buffer_tail[-_STREAM_DEGRADE_WINDOW:]
    if all(c in " \t\n\r" for c in tail):
        _degrade_logger.warning(
            "[%s] Stream degrade detected: last %d chars are all whitespace; aborting stream.",
            agent_name, _STREAM_DEGRADE_WINDOW,
        )
        return True
    return False
