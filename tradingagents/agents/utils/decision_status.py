"""Deterministic decision / run status vocabulary (D-009 P0-1).

Lifecycle ``reports.status`` (pending/running/completed/failed) remains the job
lifecycle. These fields describe *analysis validity* and *trade action* and must
not be collapsed into Neutral/HOLD.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, MutableMapping, Optional, Sequence

# analysis_status
ANALYSIS_VALID = "VALID"
ANALYSIS_PARTIAL = "PARTIAL"
ANALYSIS_ABSTAIN = "ABSTAIN"
ANALYSIS_INVALID_RUN = "INVALID_RUN"
ANALYSIS_DATA_ERROR = "DATA_ERROR"

ANALYSIS_STATUSES = frozenset(
    {
        ANALYSIS_VALID,
        ANALYSIS_PARTIAL,
        ANALYSIS_ABSTAIN,
        ANALYSIS_INVALID_RUN,
        ANALYSIS_DATA_ERROR,
    }
)

# direction
DIRECTION_BULL = "BULL"
DIRECTION_BEAR = "BEAR"
DIRECTION_NEUTRAL = "NEUTRAL"
DIRECTION_NA = "N/A"

DIRECTIONS = frozenset(
    {DIRECTION_BULL, DIRECTION_BEAR, DIRECTION_NEUTRAL, DIRECTION_NA}
)

# trade_action
ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_HOLD = "HOLD"
ACTION_WAIT = "WAIT"
ACTION_NO_TRADE = "NO_TRADE"

TRADE_ACTIONS = frozenset(
    {ACTION_BUY, ACTION_SELL, ACTION_HOLD, ACTION_WAIT, ACTION_NO_TRADE}
)

# risk_status
RISK_OK = "OK"
RISK_ELEVATED = "ELEVATED"
RISK_BLOCKED = "BLOCKED"
RISK_UNKNOWN = "UNKNOWN"

RISK_STATUSES = frozenset({RISK_OK, RISK_ELEVATED, RISK_BLOCKED, RISK_UNKNOWN})

# confirmation_state
CONFIRM_CONFIRMED = "CONFIRMED"
CONFIRM_PARTIAL = "PARTIAL"
CONFIRM_UNRESOLVED = "UNRESOLVED"

CONFIRMATION_STATES = frozenset(
    {CONFIRM_CONFIRMED, CONFIRM_PARTIAL, CONFIRM_UNRESOLVED}
)

MAX_REVERSAL_STAGED_POSITION_PCT = 10.0

# analysis_status values that must never enter calibration / directional backtest
NON_ELIGIBLE_ANALYSIS_STATUSES = frozenset(
    {
        ANALYSIS_INVALID_RUN,
        ANALYSIS_DATA_ERROR,
        ANALYSIS_ABSTAIN,
        ANALYSIS_PARTIAL,
    }
)

# trade actions that are not directional market views
NON_DIRECTIONAL_TRADE_ACTIONS = frozenset({ACTION_WAIT, ACTION_NO_TRADE})


@dataclass
class DecisionStatus:
    analysis_status: str
    direction: str
    trade_action: str
    risk_status: str
    confirmation_state: str = CONFIRM_UNRESOLVED
    failure_class: Optional[str] = None
    reason_codes: list[str] = field(default_factory=list)
    confidence: Optional[int] = None
    probability: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def invalid_run_status(
    *,
    failure_class: str = ANALYSIS_DATA_ERROR,
    reason_codes: Optional[list[str]] = None,
    risk_status: str = RISK_UNKNOWN,
) -> DecisionStatus:
    """Build the canonical INVALID_RUN → NO_TRADE payload.

    ``analysis_status`` is always INVALID_RUN; ``failure_class`` carries the
    diagnostic label (typically DATA_ERROR) without collapsing the run into a
    Neutral/HOLD market view.
    """
    fc = failure_class or ANALYSIS_DATA_ERROR
    return DecisionStatus(
        analysis_status=ANALYSIS_INVALID_RUN,
        direction=DIRECTION_NA,
        trade_action=ACTION_NO_TRADE,
        risk_status=risk_status,
        confirmation_state=CONFIRM_UNRESOLVED,
        failure_class=fc,
        reason_codes=list(reason_codes or []),
        confidence=None,
        probability=None,
    )


def abstain_status(
    *,
    reason_codes: Optional[list[str]] = None,
    trade_action: str = ACTION_NO_TRADE,
    risk_status: str = RISK_BLOCKED,
) -> DecisionStatus:
    """Data partially usable but direction must not be asserted as Neutral."""
    action = trade_action if trade_action in TRADE_ACTIONS else ACTION_NO_TRADE
    if action not in NON_DIRECTIONAL_TRADE_ACTIONS:
        action = ACTION_NO_TRADE
    return DecisionStatus(
        analysis_status=ANALYSIS_ABSTAIN,
        direction=DIRECTION_NA,
        trade_action=action,
        risk_status=risk_status,
        confirmation_state=CONFIRM_UNRESOLVED,
        failure_class=None,
        reason_codes=list(reason_codes or ["direction_not_decidable"]),
        confidence=None,
        probability=None,
    )


def valid_status(
    *,
    direction: str,
    trade_action: str,
    risk_status: str = RISK_OK,
    confirmation_state: str = CONFIRM_CONFIRMED,
    confidence: Optional[int] = None,
    probability: Optional[float] = None,
    reason_codes: Optional[list[str]] = None,
) -> DecisionStatus:
    dir_norm = direction if direction in DIRECTIONS else DIRECTION_NEUTRAL
    act_norm = trade_action if trade_action in TRADE_ACTIONS else ACTION_HOLD
    return DecisionStatus(
        analysis_status=ANALYSIS_VALID,
        direction=dir_norm,
        trade_action=act_norm,
        risk_status=risk_status if risk_status in RISK_STATUSES else RISK_OK,
        confirmation_state=confirmation_state
        if confirmation_state in CONFIRMATION_STATES
        else CONFIRM_CONFIRMED,
        failure_class=None,
        reason_codes=list(reason_codes or []),
        confidence=confidence,
        probability=probability,
    )


def partial_status(
    *,
    reason_codes: Optional[list[str]] = None,
    failed_analysts: Optional[list[str]] = None,
    trade_action: str = ACTION_NO_TRADE,
    risk_status: str = RISK_ELEVATED,
    direction: str = DIRECTION_NA,
) -> DecisionStatus:
    """Some required analysts failed; direction must not be treated as eligible Neutral."""
    codes = list(reason_codes or [])
    if failed_analysts:
        codes.append(f"failed_analysts:{','.join(failed_analysts)}")
    action = trade_action if trade_action in TRADE_ACTIONS else ACTION_NO_TRADE
    # PARTIAL runs are not calibration-eligible; prefer non-directional actions.
    if action not in NON_DIRECTIONAL_TRADE_ACTIONS and action != ACTION_HOLD:
        action = ACTION_NO_TRADE
    return DecisionStatus(
        analysis_status=ANALYSIS_PARTIAL,
        direction=direction if direction in DIRECTIONS else DIRECTION_NA,
        trade_action=action,
        risk_status=risk_status if risk_status in RISK_STATUSES else RISK_ELEVATED,
        confirmation_state=CONFIRM_PARTIAL,
        failure_class=None,
        reason_codes=codes,
        confidence=None,
        probability=None,
    )


_DIRECTION_FROM_VERDICT: dict[str, str] = {
    "看多": DIRECTION_BULL,
    "偏多": DIRECTION_BULL,
    "BULL": DIRECTION_BULL,
    "BULLISH": DIRECTION_BULL,
    "BUY": DIRECTION_BULL,
    "看空": DIRECTION_BEAR,
    "偏空": DIRECTION_BEAR,
    "BEAR": DIRECTION_BEAR,
    "BEARISH": DIRECTION_BEAR,
    "SELL": DIRECTION_BEAR,
    "中性": DIRECTION_NEUTRAL,
    "NEUTRAL": DIRECTION_NEUTRAL,
    "HOLD": DIRECTION_NEUTRAL,
    "观望": DIRECTION_NA,
    "N/A": DIRECTION_NA,
    "NA": DIRECTION_NA,
}


def map_verdict_direction(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    if not text:
        return DIRECTION_NA
    # Prefer exact Chinese keys via original casing first
    orig = str(raw or "").strip()
    if orig in _DIRECTION_FROM_VERDICT:
        return _DIRECTION_FROM_VERDICT[orig]
    if text in _DIRECTION_FROM_VERDICT:
        return _DIRECTION_FROM_VERDICT[text]
    return DIRECTION_NA


def map_verdict_trade_action(
    *,
    direction: str,
    winner: Any = None,
    position_pct: Any = None,
) -> str:
    """Map a successful manager verdict into BUY/SELL/HOLD (not WAIT/NO_TRADE)."""
    dir_norm = direction if direction in DIRECTIONS else map_verdict_direction(direction)
    if dir_norm == DIRECTION_BULL:
        return ACTION_BUY
    if dir_norm == DIRECTION_BEAR:
        return ACTION_SELL
    if dir_norm == DIRECTION_NEUTRAL:
        return ACTION_HOLD
    return ACTION_NO_TRADE


def resolve_staged_entry_position(
    requested_position_pct: Any,
    *,
    vpa_context: Mapping[str, Any] | None = None,
) -> float:
    """Resolve position percentage with staged entry caps for reversal candidates (D-009 P1-2)."""
    try:
        pos = float(requested_position_pct)
    except (TypeError, ValueError):
        pos = 0.0
    if not vpa_context or not isinstance(vpa_context, Mapping):
        return pos
    regime = str(vpa_context.get("volume_regime") or "").strip()
    rev_state = str(vpa_context.get("reversal_state") or "").strip()
    if (
        regime in {"capitulation_candidate", "high_volume_stagnation_candidate"}
        and rev_state == "reversal_confirmed"
    ):
        return min(pos, MAX_REVERSAL_STAGED_POSITION_PCT)
    return pos


def evaluate_confirmation_state(
    *,
    focus_claim_ids: Sequence[Any] | None = None,
    unresolved_claim_ids: Sequence[Any] | None = None,
    claims_verification: Sequence[Mapping[str, Any]] | None = None,
    claim_evidence_summary: Mapping[str, Mapping[str, Any]] | None = None,
    claims: Sequence[Mapping[str, Any]] | None = None,
    adopted_claim_ids: Sequence[Any] | None = None,
    partially_adopted_claims: Sequence[Any] | None = None,
    rejected_claim_ids: Sequence[Any] | None = None,
) -> tuple[str, list[str]]:
    """Determine confirmation_state (CONFIRMED / PARTIAL / UNRESOLVED) and diagnostic codes.

    Deterministic rules (D-009 P0-5b / Confirmation Gate Claim Lifecycle):
    - confirmation_relevant_claims = focus ∪ adopted ∪ partially_adopted ∪ rejected_but_deterministically_adopted
    - focus / adopted with reject or fatal -> UNRESOLVED -> WAIT
    - partially adopted (incomplete evidence) -> PARTIAL + WAIT
    - rejected, non-core with reject -> do NOT block confirmation; keep audited reason/log
    - rejected with partial -> conservative -> PARTIAL + WAIT
    - rejected with adopt -> verdict consistency failure -> ABSTAIN + NO_TRADE (evaluated as UNRESOLVED here)
    - unadjudicated material claims with adopt/partial -> completeness/consistency check
    """
    focus_ids = [str(x).strip() for x in (focus_claim_ids or []) if str(x).strip()]
    unresolved_ids = [str(x).strip() for x in (unresolved_claim_ids or []) if str(x).strip()]
    adopted_ids = [str(x).strip() for x in (adopted_claim_ids or []) if str(x).strip()]
    partially_adopted_ids = [str(x).strip() for x in (partially_adopted_claims or []) if str(x).strip()]
    rejected_ids = [str(x).strip() for x in (rejected_claim_ids or []) if str(x).strip()]

    adjudication_provided = (
        adopted_claim_ids is not None
        or partially_adopted_claims is not None
        or rejected_claim_ids is not None
    )

    if focus_ids:
        core_claim_ids = focus_ids
    elif unresolved_ids:
        core_claim_ids = unresolved_ids
    else:
        core_claim_ids = []

    summary_map: dict[str, Mapping[str, Any]] = {}
    if claim_evidence_summary:
        summary_map = {str(k).strip(): v for k, v in claim_evidence_summary.items() if str(k).strip()}
    elif claims_verification or claims:
        from tradingagents.agents.utils.evidence_verifier import aggregate_claim_evidence

        summary_map = aggregate_claim_evidence(claims=claims, claims_verification=claims_verification)

    ver_by_cid: dict[str, list[Mapping[str, Any]]] = {}
    for v in (claims_verification or []):
        cid = str(v.get("claim_id", "") or "").strip()
        if cid:
            ver_by_cid.setdefault(cid, []).append(v)

    known_claims = {
        str(c.get("claim_id", "") or "").strip(): c
        for c in (claims or [])
        if isinstance(c, Mapping) and str(c.get("claim_id", "") or "").strip()
    }

    def _is_claim_obs_hypo(cid: str) -> bool:
        sm = summary_map.get(cid)
        if sm and sm.get("is_observation_or_hypothesis"):
            return True
        cl = known_claims.get(cid)
        if cl:
            from tradingagents.agents.utils.evidence_verifier import is_observation_or_hypothesis_claim
            return is_observation_or_hypothesis_claim(cl)
        return False

    def _is_claim_fatal(cid: str) -> bool:
        sm = summary_map.get(cid)
        if sm:
            if sm.get("pit_failed"):
                return True
            cnt = sm.get("counts") or {}
            if cnt.get("contradicted", 0) > 0 or cnt.get("source_unavailable", 0) > 0:
                return True
        for v in ver_by_cid.get(cid, []):
            if v.get("status") in {"contradicted", "source_unavailable"} or v.get("is_fatal"):
                return True
        return False

    def _get_claim_decision(cid: str) -> str:
        if _is_claim_fatal(cid):
            return "reject"
        sm = summary_map.get(cid)
        if sm:
            dec = sm.get("decision")
            if dec in {"adopt", "partial", "reject"}:
                return str(dec)
            cnt = sm.get("counts") or {}
            total = cnt.get("total", 0)
            verified = cnt.get("verified", 0)
            cov = sm.get("coverage", (verified / total) if total > 0 else 0.0)
            if total > 0 and verified == total:
                return "adopt"
            elif verified > 0 and (cov >= 0.67 or math.isclose(cov, 2 / 3, abs_tol=1e-3)):
                return "partial"
            return "reject"
        v_list = ver_by_cid.get(cid, [])
        if not v_list:
            return "reject"
        total = len(v_list)
        verified = sum(1 for v in v_list if v.get("status") == "verified")
        if verified == total and total > 0:
            return "adopt"
        elif verified > 0 and (verified / total >= 0.67 or math.isclose(verified / total, 2 / 3, abs_tol=1e-3)):
            return "partial"
        return "reject"

    def _is_claim_verified(cid: str) -> bool:
        if _is_claim_fatal(cid):
            return False
        return _get_claim_decision(cid) == "adopt"

    # Row 5: rejected + deterministic adopt -> verdict consistency failure
    rejected_adopt_cids = [
        cid for cid in rejected_ids
        if _get_claim_decision(cid) == "adopt"
    ]
    if rejected_adopt_cids:
        return CONFIRM_UNRESOLVED, [
            f"verdict_consistency_rejected_adopt:{','.join(sorted(rejected_adopt_cids))}"
        ]

    # Row 6: Unadjudicated material claims check
    unadjudicated_adopt_cids: list[str] = []
    unadjudicated_partial_cids: list[str] = []
    if adjudication_provided:
        decided_cids = set(adopted_ids) | set(partially_adopted_ids) | set(rejected_ids)
        known_debate_cids = list(dict.fromkeys(
            [str(c.get("claim_id", "") or "").strip() for c in (claims or []) if str(c.get("claim_id", "") or "").strip()]
            + list(summary_map.keys())
            + list(ver_by_cid.keys())
            + core_claim_ids
        ))
        for cid in known_debate_cids:
            if cid not in decided_cids:
                dec = _get_claim_decision(cid)
                if dec == "adopt":
                    unadjudicated_adopt_cids.append(cid)
                elif dec == "partial":
                    unadjudicated_partial_cids.append(cid)

    if unadjudicated_adopt_cids:
        return CONFIRM_UNRESOLVED, [
            f"unadjudicated_material_claims_adopt:{','.join(sorted(unadjudicated_adopt_cids))}"
        ]

    # Row 3: rejected, non-core with reject -> audit record, does not block
    rejected_reject_cids = [
        cid for cid in rejected_ids
        if cid not in core_claim_ids and _get_claim_decision(cid) == "reject"
    ]
    audit_rejected_codes: list[str] = []
    if rejected_reject_cids:
        audit_rejected_codes.append(
            f"audited_rejected_claims:{','.join(sorted(rejected_reject_cids))}"
        )

    # Row 4: rejected with partial
    rejected_partial_cids = [
        cid for cid in rejected_ids
        if _get_claim_decision(cid) == "partial"
    ]

    # Row 1: Fatal check across core, adopted, and partially adopted claims
    core_fatal = [cid for cid in core_claim_ids if _is_claim_fatal(cid)]
    adopted_fatal = [cid for cid in adopted_ids if _is_claim_fatal(cid)]
    partially_adopted_fatal = [cid for cid in partially_adopted_ids if _is_claim_fatal(cid)]

    fatal_codes: list[str] = []
    if adopted_fatal:
        fatal_codes.append(f"fatal_adopted_claims:{','.join(sorted(adopted_fatal))}")
    if core_fatal:
        fatal_codes.append(f"fatal_core_claims:{','.join(sorted(core_fatal))}")
    if partially_adopted_fatal:
        fatal_codes.append(f"fatal_partially_adopted_claims:{','.join(sorted(partially_adopted_fatal))}")
    if fatal_codes:
        return CONFIRM_UNRESOLVED, fatal_codes

    # If neither core claims nor adopted claims exist
    if not core_claim_ids and not adopted_ids:
        unadjudicated_fatal_cids: set[str] = set()
        for cid, sm in summary_map.items():
            if _is_claim_fatal(cid):
                if cid in rejected_ids and _get_claim_decision(cid) == "reject":
                    continue
                unadjudicated_fatal_cids.add(cid)
        for v in (claims_verification or []):
            if v.get("status") in {"contradicted", "source_unavailable"} or v.get("is_fatal"):
                cid = str(v.get("claim_id", "") or "").strip()
                if cid:
                    if cid in rejected_ids and _get_claim_decision(cid) == "reject":
                        continue
                    unadjudicated_fatal_cids.add(cid)

        if unadjudicated_fatal_cids:
            return CONFIRM_UNRESOLVED, [
                f"fatal_contradicted_claims:{','.join(sorted(unadjudicated_fatal_cids))}"
            ]
        if rejected_partial_cids or unadjudicated_partial_cids:
            p_codes: list[str] = []
            if rejected_partial_cids:
                p_codes.append(f"rejected_partial_claims:{','.join(sorted(rejected_partial_cids))}")
            if unadjudicated_partial_cids:
                p_codes.append(f"unadjudicated_partial_claims:{','.join(sorted(unadjudicated_partial_cids))}")
            return CONFIRM_PARTIAL, p_codes + audit_rejected_codes
        return CONFIRM_CONFIRMED, audit_rejected_codes

    # Core claims verification
    if core_claim_ids:
        core_verified = [cid for cid in core_claim_ids if _is_claim_verified(cid)]
        core_unverified = [cid for cid in core_claim_ids if not _is_claim_verified(cid) and not _is_claim_fatal(cid)]
        if len(core_verified) == 0:
            return CONFIRM_UNRESOLVED, [f"unverified_core_claims:{','.join(core_claim_ids)}"]
        core_unverified_factual = [cid for cid in core_unverified if not _is_claim_obs_hypo(cid)]
        core_unverified_obs = [cid for cid in core_unverified if _is_claim_obs_hypo(cid)]
        core_has_partial = len(core_unverified_factual) > 0
    else:
        core_verified = []
        core_unverified = []
        core_unverified_factual = []
        core_unverified_obs = []
        core_has_partial = False

    # Adopted claims verification
    adopted_unverified = [cid for cid in adopted_ids if _get_claim_decision(cid) == "reject"]
    if adopted_unverified:
        return CONFIRM_UNRESOLVED, [f"unverified_adopted_claims:{','.join(sorted(adopted_unverified))}"]
    adopted_partial = [cid for cid in adopted_ids if _get_claim_decision(cid) == "partial"]

    # Assemble partial reasons
    partial_codes: list[str] = []
    audit_obs_codes: list[str] = []

    if core_has_partial:
        partial_codes.append(
            f"partial_core_claims:verified={','.join(core_verified)};unverified={','.join(core_unverified_factual)}"
        )
    if core_unverified_obs:
        audit_obs_codes.append(
            f"audited_observation_claims:{','.join(sorted(core_unverified_obs))}"
        )

    adopted_partial_factual = [cid for cid in adopted_partial if not _is_claim_obs_hypo(cid)]
    adopted_partial_obs = [cid for cid in adopted_partial if _is_claim_obs_hypo(cid)]
    if adopted_partial_factual:
        partial_codes.append(
            f"partial_adopted_claims:{','.join(sorted(adopted_partial_factual))}"
        )
    if adopted_partial_obs:
        audit_obs_codes.append(
            f"audited_observation_claims:{','.join(sorted(adopted_partial_obs))}"
        )

    partially_adopted_factual = [cid for cid in partially_adopted_ids if not _is_claim_obs_hypo(cid)]
    partially_adopted_obs = [cid for cid in partially_adopted_ids if _is_claim_obs_hypo(cid)]
    if partially_adopted_factual:
        partial_codes.append(
            f"partially_adopted_claims:{','.join(sorted(partially_adopted_factual))}"
        )
    if partially_adopted_obs:
        audit_obs_codes.append(
            f"audited_observation_claims:{','.join(sorted(partially_adopted_obs))}"
        )

    rejected_partial_factual = [cid for cid in rejected_partial_cids if not _is_claim_obs_hypo(cid)]
    rejected_partial_obs = [cid for cid in rejected_partial_cids if _is_claim_obs_hypo(cid)]
    if rejected_partial_factual:
        partial_codes.append(
            f"rejected_partial_claims:{','.join(sorted(rejected_partial_factual))}"
        )
    if rejected_partial_obs:
        audit_obs_codes.append(
            f"audited_observation_claims:{','.join(sorted(rejected_partial_obs))}"
        )

    unadjudicated_partial_factual = [cid for cid in unadjudicated_partial_cids if not _is_claim_obs_hypo(cid)]
    unadjudicated_partial_obs = [cid for cid in unadjudicated_partial_cids if _is_claim_obs_hypo(cid)]
    if unadjudicated_partial_factual:
        partial_codes.append(
            f"unadjudicated_partial_claims:{','.join(sorted(unadjudicated_partial_factual))}"
        )
    if unadjudicated_partial_obs:
        audit_obs_codes.append(
            f"audited_observation_claims:{','.join(sorted(unadjudicated_partial_obs))}"
        )

    # Consolidate unique observation codes
    all_obs_cids = sorted(list(set(
        core_unverified_obs + adopted_partial_obs + partially_adopted_obs + rejected_partial_obs + unadjudicated_partial_obs
    )))
    consolidated_obs_codes = [f"audited_observation_claims:{','.join(all_obs_cids)}"] if all_obs_cids else []

    if partial_codes:
        return CONFIRM_PARTIAL, partial_codes + consolidated_obs_codes + audit_rejected_codes

    # Everything confirmed
    verified_core = core_verified if core_claim_ids else [cid for cid in adopted_ids if _is_claim_verified(cid)]
    if verified_core:
        return CONFIRM_CONFIRMED, [f"all_core_claims_verified:{','.join(verified_core)}"] + consolidated_obs_codes + audit_rejected_codes
    elif all_obs_cids:
        # If ONLY observations exist and NO verified factual claims exist, cannot confirm
        return CONFIRM_UNRESOLVED, [f"observation_hypotheses_unverified_without_factual_core:{','.join(all_obs_cids)}"] + audit_rejected_codes
    return CONFIRM_CONFIRMED, audit_rejected_codes


def status_from_manager_verdict(
    manager_verdict: Mapping[str, Any] | None,
    *,
    prior_analysis_status: Optional[str] = None,
    investment_debate_state: Mapping[str, Any] | None = None,
    claims_verification: Sequence[Mapping[str, Any]] | None = None,
    claim_evidence_summary: Mapping[str, Mapping[str, Any]] | None = None,
    focus_claim_ids: Sequence[Any] | None = None,
    unresolved_claim_ids: Sequence[Any] | None = None,
    claims: Sequence[Mapping[str, Any]] | None = None,
    market_data_context: Mapping[str, Any] | None = None,
    vpa_context: Mapping[str, Any] | None = None,
) -> DecisionStatus:
    """Build canonical status for a completed Research Manager path."""
    mv = manager_verdict if isinstance(manager_verdict, Mapping) else {}
    nested = mv.get("decision_status")
    from_nested = None
    if isinstance(nested, DecisionStatus):
        from_nested = nested
    elif isinstance(nested, Mapping):
        from_nested = decision_status_from_mapping(nested)
    if from_nested is not None:
        # Nested ABSTAIN/INVALID from earlier gates wins.
        if from_nested.analysis_status in {
            ANALYSIS_INVALID_RUN,
            ANALYSIS_DATA_ERROR,
            ANALYSIS_ABSTAIN,
        }:
            return from_nested

    # Consistency hard gate: never emit VALID/BUY when the plan is blocked.
    if mv.get("consistency_check_passed") is False:
        failed = [str(x) for x in (mv.get("failed_checks") or []) if x]
        return abstain_status(
            reason_codes=["manager_consistency_hard_gate", *failed],
            trade_action=ACTION_NO_TRADE,
            risk_status=RISK_BLOCKED,
        )

    deb_state = investment_debate_state if isinstance(investment_debate_state, Mapping) else {}
    mv_in_deb = deb_state.get("manager_verdict") if isinstance(deb_state.get("manager_verdict"), Mapping) else {}

    f_ids = (
        focus_claim_ids
        if focus_claim_ids is not None
        else (deb_state.get("focus_claim_ids") or mv.get("focus_claim_ids") or mv_in_deb.get("focus_claim_ids"))
    )
    u_ids = (
        unresolved_claim_ids
        if unresolved_claim_ids is not None
        else (deb_state.get("unresolved_claim_ids") or mv.get("unresolved_claim_ids") or mv_in_deb.get("unresolved_claim_ids"))
    )
    ver = (
        claims_verification
        if claims_verification is not None
        else (deb_state.get("evidence_verification") or mv.get("evidence_verification") or mv_in_deb.get("evidence_verification"))
    )
    ev_summary = (
        claim_evidence_summary
        if claim_evidence_summary is not None
        else (deb_state.get("claim_evidence_summary") or mv.get("claim_evidence_summary") or mv_in_deb.get("claim_evidence_summary"))
    )
    cl_list = (
        claims
        if claims is not None
        else (deb_state.get("claims") or mv.get("claims") or mv_in_deb.get("claims"))
    )
    adopted_ids = (
        mv.get("adopted_claim_ids")
        if mv.get("adopted_claim_ids") is not None
        else (mv_in_deb.get("adopted_claim_ids") or deb_state.get("adopted_claim_ids"))
    )
    partially_adopted_ids = (
        mv.get("partially_adopted_claims")
        if mv.get("partially_adopted_claims") is not None
        else (mv_in_deb.get("partially_adopted_claims") or deb_state.get("partially_adopted_claims"))
    )
    rejected_ids = (
        mv.get("rejected_claim_ids")
        if mv.get("rejected_claim_ids") is not None
        else (mv_in_deb.get("rejected_claim_ids") or deb_state.get("rejected_claim_ids"))
    )

    confirmation_state, confirm_codes = evaluate_confirmation_state(
        focus_claim_ids=f_ids,
        unresolved_claim_ids=u_ids,
        claims_verification=ver,
        claim_evidence_summary=ev_summary,
        claims=cl_list,
        adopted_claim_ids=adopted_ids,
        partially_adopted_claims=partially_adopted_ids,
        rejected_claim_ids=rejected_ids,
    )

    # Consistency hard gate: rejected + adopt or unadjudicated material claim with adopt or fatal in adopted/partially adopted
    if any(
        code.startswith("verdict_consistency_rejected_adopt:")
        or code.startswith("unadjudicated_material_claims_adopt:")
        or code.startswith("fatal_adopted_claims:")
        or code.startswith("fatal_partially_adopted_claims:")
        for code in confirm_codes
    ):
        return abstain_status(
            reason_codes=["manager_consistency_hard_gate", *confirm_codes],
            trade_action=ACTION_NO_TRADE,
            risk_status=RISK_BLOCKED,
        )

    # VPA candidate / reversal feature evaluation (D-009 P1-2)
    vpa = vpa_context
    if vpa is None and isinstance(market_data_context, Mapping):
        vpa = market_data_context.get("vpa_context") or market_data_context.get("vpa_structured")
    if vpa is None and isinstance(deb_state, Mapping):
        mdc = deb_state.get("market_data_context")
        if isinstance(mdc, Mapping):
            vpa = mdc.get("vpa_context") or mdc.get("vpa_structured")
    if vpa is None and isinstance(mv, Mapping):
        vpa = mv.get("vpa_context") or mv.get("vpa_structured")

    vpa_regime = str(vpa.get("volume_regime") or "").strip() if isinstance(vpa, Mapping) else ""
    vpa_rev_state = str(vpa.get("reversal_state") or "").strip() if isinstance(vpa, Mapping) else ""
    vpa_codes: list[str] = []

    direction = map_verdict_direction(mv.get("direction"))
    if confirmation_state in {CONFIRM_UNRESOLVED, CONFIRM_PARTIAL}:
        trade_action = ACTION_WAIT
    else:
        trade_action = map_verdict_trade_action(
            direction=direction,
            winner=mv.get("winner"),
            position_pct=mv.get("position_pct"),
        )
        if trade_action == ACTION_BUY:
            if vpa_regime == "capitulation_candidate":
                if vpa_rev_state != "reversal_confirmed":
                    trade_action = ACTION_WAIT
                    vpa_codes.append("vpa_capitulation_unconfirmed_wait")
                else:
                    vpa_codes.append("vpa_reversal_confirmed_staged_entry")
            elif vpa_regime == "high_volume_stagnation_candidate":
                if vpa_rev_state != "reversal_confirmed":
                    trade_action = ACTION_WAIT
                    vpa_codes.append("vpa_stagnation_candidate_wait")
                else:
                    vpa_codes.append("vpa_reversal_confirmed_staged_entry")

    if prior_analysis_status == ANALYSIS_PARTIAL:
        return partial_status(
            reason_codes=["prior_partial_analyst_failures", *confirm_codes, *vpa_codes],
            trade_action=ACTION_NO_TRADE
            if trade_action not in NON_DIRECTIONAL_TRADE_ACTIONS
            else trade_action,
            direction=DIRECTION_NA,
        )
    if direction == DIRECTION_NA and trade_action in NON_DIRECTIONAL_TRADE_ACTIONS and confirmation_state == CONFIRM_CONFIRMED:
        return abstain_status(reason_codes=["manager_direction_na", *confirm_codes, *vpa_codes])

    raw_conf = mv.get("confidence")
    raw_prob = mv.get("probability")
    conf_val: Optional[int] = None
    prob_val: Optional[float] = None
    if confirmation_state == CONFIRM_CONFIRMED and trade_action not in NON_DIRECTIONAL_TRADE_ACTIONS:
        if raw_conf is not None:
            try:
                conf_val = int(raw_conf)
            except (ValueError, TypeError):
                conf_val = None
        if raw_prob is not None:
            try:
                prob_val = float(raw_prob)
            except (ValueError, TypeError):
                prob_val = None

    return valid_status(
        direction=direction if direction != DIRECTION_NA else DIRECTION_NEUTRAL,
        trade_action=trade_action if trade_action != ACTION_NO_TRADE else ACTION_HOLD,
        risk_status=RISK_OK,
        confirmation_state=confirmation_state,
        reason_codes=["manager_terminal", *confirm_codes, *vpa_codes],
        confidence=conf_val,
        probability=prob_val,
    )


def aggregate_horizon_decision_statuses(
    horizon_payloads: Mapping[str, Mapping[str, Any]] | None,
    *,
    requested_horizons: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Aggregate short/medium statuses into top-level canonical fields.

    Semantics:
    - all_invalid: every requested completed horizon is INVALID/DATA_ERROR/ABSTAIN
      → top INVALID_RUN + NO_TRADE
    - mixed: some eligible VALID and some non-VALID → PARTIAL + NO_TRADE
    - all_valid: every completed horizon VALID → promote primary VALID status
    """
    from tradingagents.agents.utils.decision_status import decision_status_from_state

    requested = list(requested_horizons or [])
    if not requested and isinstance(horizon_payloads, Mapping):
        requested = [k for k in ("short", "medium") if k in horizon_payloads]

    statuses: list[DecisionStatus] = []
    by_horizon: dict[str, dict[str, Any]] = {}
    for horizon in requested:
        payload = (horizon_payloads or {}).get(horizon) or {}
        if not isinstance(payload, Mapping):
            continue
        if payload.get("status") in {"failed", "not_requested"}:
            # Treat failed horizon as INVALID for aggregation.
            st = invalid_run_status(
                reason_codes=[f"horizon_{horizon}_failed"],
            )
        else:
            st = decision_status_from_state(payload) or resolve_soft(
                payload
            )
        statuses.append(st)
        by_horizon[horizon] = st.to_dict()

    if not statuses:
        top = invalid_run_status(reason_codes=["dual_horizon_no_status"])
        return {
            "aggregation": "all_invalid",
            "decision_status": top.to_dict(),
            "analysis_status": top.analysis_status,
            "trade_action": top.trade_action,
            "direction": top.direction,
            "risk_status": top.risk_status,
            "by_horizon": by_horizon,
        }

    non_exec = {
        ANALYSIS_INVALID_RUN,
        ANALYSIS_DATA_ERROR,
        ANALYSIS_ABSTAIN,
    }
    all_invalid = all(s.analysis_status in non_exec for s in statuses)
    all_valid = all(s.analysis_status == ANALYSIS_VALID for s in statuses)

    if all_invalid:
        top = invalid_run_status(reason_codes=["dual_horizon_all_invalid"])
        aggregation = "all_invalid"
    elif all_valid:
        top = statuses[0]
        aggregation = "all_valid"
    else:
        top = partial_status(
            reason_codes=["dual_horizon_mixed"],
            trade_action=ACTION_NO_TRADE,
        )
        aggregation = "mixed"

    return {
        "aggregation": aggregation,
        "decision_status": top.to_dict(),
        "analysis_status": top.analysis_status,
        "trade_action": top.trade_action,
        "direction": top.direction,
        "risk_status": top.risk_status,
        "by_horizon": by_horizon,
    }


def resolve_soft(payload: Mapping[str, Any]) -> DecisionStatus:
    """Fallback when a horizon payload lacks decision_status."""
    from tradingagents.agents.utils.run_integrity import evaluate_run_integrity
    from tradingagents.agents.utils.decision_status import (
        abstain_status,
        decision_status_from_mapping,
    )

    integrity = evaluate_run_integrity(payload)
    if integrity.decision_status:
        parsed = decision_status_from_mapping(integrity.decision_status)
        if parsed is not None:
            return parsed
    return abstain_status(
        reason_codes=["horizon_status_missing_abstain"],
        trade_action=ACTION_NO_TRADE,
        risk_status=RISK_UNKNOWN,
    )


def db_direction_from_canonical(
    canonical: Mapping[str, Any] | None,
    *,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Persist direction consistent with analysis_status / trade_action.

    D-009: BULL + Risk BLOCKED -> keep BULL direction, trade_action NO_TRADE.
    Only INVALID_RUN / DATA_ERROR / ABSTAIN collapse direction to N/A.
    """
    if not isinstance(canonical, Mapping):
        return fallback
    analysis_status = str(canonical.get("analysis_status") or "").upper()
    direction = str(canonical.get("direction") or "").upper()
    if analysis_status in {
        ANALYSIS_INVALID_RUN,
        ANALYSIS_DATA_ERROR,
        ANALYSIS_ABSTAIN,
    } or direction in {DIRECTION_NA, "N/A", "NA", ""}:
        return DIRECTION_NA
    display = {
        DIRECTION_BULL: "看多",
        DIRECTION_BEAR: "看空",
        DIRECTION_NEUTRAL: "中性",
        DIRECTION_NA: DIRECTION_NA,
    }
    if direction in display:
        return display[direction]
    if direction in {"看多", "偏多", "看空", "偏空", "中性", "N/A"}:
        return direction
    return fallback


def status_from_risk_verdict(
    *,
    upstream: Optional[DecisionStatus],
    risk_verdict: str,
    reason_codes: Optional[list[str]] = None,
    retry_count: int = 0,
    max_retries: int = 1,
    retry_exhausted: Optional[bool] = None,
) -> DecisionStatus:
    """Rewrite canonical status at Risk Judge terminal — never inherit stale BUY blindly.

    - upstream non-executable → keep non-executable, force risk BLOCKED
    - revise (retry not exhausted) → keep upstream analysis_status (VALID) & direction & trade_action, risk_status ELEVATED
    - revise (retry exhausted) → keep direction, trade_action NO_TRADE, risk_status BLOCKED
    - reject/blocked → keep upstream analysis_status & direction, trade_action NO_TRADE, risk_status BLOCKED
    - pass/approve → keep upstream direction/action if VALID; else ABSTAIN
    """
    verdict = str(risk_verdict or "").strip().lower()
    codes = list(reason_codes or [])
    codes.append(f"risk_verdict:{verdict or 'unknown'}")

    if upstream is not None and is_non_executable_status(upstream):
        return DecisionStatus(
            analysis_status=upstream.analysis_status,
            direction=upstream.direction if upstream.analysis_status == ANALYSIS_VALID else DIRECTION_NA,
            trade_action=ACTION_NO_TRADE
            if upstream.trade_action not in NON_DIRECTIONAL_TRADE_ACTIONS
            else upstream.trade_action,
            risk_status=RISK_BLOCKED,
            confirmation_state=CONFIRM_UNRESOLVED,
            failure_class=upstream.failure_class,
            reason_codes=list(upstream.reason_codes) + codes,
            confidence=None,
            probability=None,
        )

    is_exhausted = (
        retry_exhausted
        if retry_exhausted is not None
        else (retry_count > max_retries if retry_count > 0 else False)
    )

    if verdict in {"reject", "blocked", "fail", "failed"}:
        if upstream is not None and upstream.analysis_status == ANALYSIS_VALID:
            return DecisionStatus(
                analysis_status=ANALYSIS_VALID,
                direction=upstream.direction,
                trade_action=ACTION_NO_TRADE,
                risk_status=RISK_BLOCKED,
                confirmation_state=upstream.confirmation_state,
                failure_class=None,
                reason_codes=list(upstream.reason_codes) + codes,
                confidence=None,
                probability=None,
            )
        return abstain_status(
            reason_codes=codes,
            trade_action=ACTION_NO_TRADE,
            risk_status=RISK_BLOCKED,
        )

    if verdict == "revise":
        if is_exhausted:
            dir_val = upstream.direction if upstream is not None and upstream.analysis_status == ANALYSIS_VALID else DIRECTION_NA
            analysis_val = upstream.analysis_status if upstream is not None else ANALYSIS_ABSTAIN
            return DecisionStatus(
                analysis_status=analysis_val,
                direction=dir_val,
                trade_action=ACTION_NO_TRADE,
                risk_status=RISK_BLOCKED,
                confirmation_state=CONFIRM_UNRESOLVED,
                failure_class=None,
                reason_codes=list(upstream.reason_codes if upstream else []) + codes + ["risk_retries_exhausted"],
                confidence=None,
                probability=None,
            )
        # revise with retries remaining: keep upstream VALID status, direction, and trade_action proposal
        if upstream is not None and upstream.analysis_status == ANALYSIS_VALID:
            return DecisionStatus(
                analysis_status=ANALYSIS_VALID,
                direction=upstream.direction,
                trade_action=upstream.trade_action,
                risk_status=RISK_ELEVATED,
                confirmation_state=CONFIRM_PARTIAL,
                failure_class=None,
                reason_codes=list(upstream.reason_codes) + codes,
                confidence=upstream.confidence,
                probability=upstream.probability,
            )
        return DecisionStatus(
            analysis_status=ANALYSIS_ABSTAIN,
            direction=DIRECTION_NA,
            trade_action=ACTION_WAIT,
            risk_status=RISK_ELEVATED,
            confirmation_state=CONFIRM_PARTIAL,
            failure_class=None,
            reason_codes=codes,
            confidence=None,
            probability=None,
        )

    # pass / approve / empty → promote upstream VALID if present
    if upstream is not None and upstream.analysis_status == ANALYSIS_VALID:
        return DecisionStatus(
            analysis_status=ANALYSIS_VALID,
            direction=upstream.direction,
            trade_action=upstream.trade_action,
            risk_status=RISK_OK if verdict in {"pass", "approve", "approved", ""} else RISK_ELEVATED,
            confirmation_state=CONFIRM_CONFIRMED,
            failure_class=None,
            reason_codes=list(upstream.reason_codes) + codes,
            confidence=upstream.confidence,
            probability=upstream.probability,
        )

    return abstain_status(
        reason_codes=codes + ["risk_missing_valid_upstream"],
        trade_action=ACTION_NO_TRADE,
        risk_status=RISK_UNKNOWN,
    )


def is_calibration_eligible(
    result_or_row: Mapping[str, Any] | Any,
    *,
    allow_winner_only: bool = False,
) -> bool:
    """True only for explicit VALID runs with a directional trade action and probability.

    When ``allow_winner_only=True``, also admits completed qualifying v2 reports
    (A6 is_qualifying_v2_report specification) with winner in ('bull', 'bear')
    even when probability is None.
    Legacy rows with ``analysis_status=NULL`` are excluded so pre-P0 Neutral/HOLD
    pollution cannot keep entering calibration.
    """
    analysis_status, trade_action, probability = _extract_status_fields(result_or_row)
    if analysis_status is None and not allow_winner_only:
        return False
    if analysis_status is not None and analysis_status != ANALYSIS_VALID:
        return False
    if trade_action in NON_DIRECTIONAL_TRADE_ACTIONS:
        return False
    if probability is not None:
        return analysis_status == ANALYSIS_VALID

    if allow_winner_only:
        from tradingagents.agents.utils.shadow_credit import is_qualifying_v2_report

        rd = (
            result_or_row.get("result_data")
            if isinstance(result_or_row, Mapping)
            else getattr(result_or_row, "result_data", None)
        )
        sample = (
            dict(result_or_row)
            if isinstance(result_or_row, Mapping)
            else {
                "status": getattr(result_or_row, "status", None),
                "result_data": rd if isinstance(rd, dict) else {},
                "analysis_status": analysis_status,
                "trade_action": trade_action,
                "probability": probability,
            }
        )
        if not is_qualifying_v2_report(sample):
            return False

        # Extract winner
        target_rd = rd if isinstance(rd, dict) else (sample if isinstance(sample, dict) else {})
        inv_state = (
            target_rd.get("investment_debate_state")
            if isinstance(target_rd.get("investment_debate_state"), dict)
            else target_rd
        )
        mv = target_rd.get("manager_verdict") or inv_state.get("manager_verdict") or {}
        raw_winner = (
            mv.get("winner")
            or target_rd.get("debate_winner")
            or (
                inv_state.get("manager_verdict", {}).get("winner")
                if isinstance(inv_state.get("manager_verdict"), dict)
                else None
            )
        )
        if not raw_winner:
            for sub_k in ("short_term", "primary"):
                sub = target_rd.get(sub_k)
                if isinstance(sub, dict):
                    raw_winner = (
                        sub.get("manager_verdict", {}).get("winner")
                        if isinstance(sub.get("manager_verdict"), dict)
                        else sub.get("winner")
                    )
                    if raw_winner:
                        break
        winner_str = str(raw_winner or "").strip().lower()
        return winner_str in ("bull", "bear")

    return False


def _extract_status_fields(
    result_or_row: Mapping[str, Any] | Any,
) -> tuple[Optional[str], Optional[str], Optional[float]]:
    if isinstance(result_or_row, Mapping):
        analysis_status = result_or_row.get("analysis_status")
        trade_action = result_or_row.get("trade_action")
        probability = result_or_row.get("probability")
        nested = result_or_row.get("decision_status")
        if isinstance(nested, Mapping):
            analysis_status = analysis_status or nested.get("analysis_status")
            trade_action = trade_action or nested.get("trade_action")
            if probability is None:
                probability = nested.get("probability")
        # Also peek result_data if this looks like a row wrapper
        rd = result_or_row.get("result_data")
        if isinstance(rd, Mapping):
            analysis_status = analysis_status or rd.get("analysis_status")
            trade_action = trade_action or rd.get("trade_action")
            if probability is None:
                probability = rd.get("probability")
            nested2 = rd.get("decision_status")
            if isinstance(nested2, Mapping):
                analysis_status = analysis_status or nested2.get("analysis_status")
                trade_action = trade_action or nested2.get("trade_action")
        return (
            str(analysis_status).strip().upper() if analysis_status else None,
            str(trade_action).strip().upper() if trade_action else None,
            float(probability) if isinstance(probability, (int, float)) else None,
        )

    analysis_status = getattr(result_or_row, "analysis_status", None)
    trade_action = getattr(result_or_row, "trade_action", None)
    probability = getattr(result_or_row, "probability", None)
    rd = getattr(result_or_row, "result_data", None)
    if isinstance(rd, Mapping):
        analysis_status = analysis_status or rd.get("analysis_status")
        trade_action = trade_action or rd.get("trade_action")
        if probability is None:
            probability = rd.get("probability")
        nested = rd.get("decision_status")
        if isinstance(nested, Mapping):
            analysis_status = analysis_status or nested.get("analysis_status")
            trade_action = trade_action or nested.get("trade_action")
    return (
        str(analysis_status).strip().upper() if analysis_status else None,
        str(trade_action).strip().upper() if trade_action else None,
        float(probability) if isinstance(probability, (int, float)) else None,
    )


def apply_decision_status_to_result(
    result: MutableMapping[str, Any],
    status: DecisionStatus | Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Write decision_status into result_data and null unsafe numeric fields."""
    payload = status.to_dict() if isinstance(status, DecisionStatus) else dict(status)
    analysis_status = str(payload.get("analysis_status") or "").upper()
    trade_action = str(payload.get("trade_action") or "").upper()
    direction = str(payload.get("direction") or DIRECTION_NA)

    result["decision_status"] = payload
    result["analysis_status"] = analysis_status
    result["trade_action"] = trade_action
    result["risk_status"] = str(payload.get("risk_status") or RISK_UNKNOWN).upper()
    result["confirmation_state"] = str(
        payload.get("confirmation_state") or CONFIRM_UNRESOLVED
    ).upper()
    if payload.get("failure_class"):
        result["failure_class"] = payload.get("failure_class")
    if payload.get("reason_codes") is not None:
        result["reason_codes"] = list(payload.get("reason_codes") or [])

    # Compat: lifecycle decision stores trade_action (includes NO_TRADE/WAIT).
    result["decision"] = trade_action
    result["direction"] = direction

    if analysis_status in {
        ANALYSIS_INVALID_RUN,
        ANALYSIS_DATA_ERROR,
        ANALYSIS_ABSTAIN,
        ANALYSIS_PARTIAL,
    } or trade_action in NON_DIRECTIONAL_TRADE_ACTIONS:
        result["confidence"] = None
        result["probability"] = None
        result["target_price"] = None
        result["stop_loss_price"] = None
        # Strip fabricated range keys if present
        for key in ("upside", "downside", "numeric_ranges", "odds"):
            if key in result:
                result[key] = None if key != "numeric_ranges" else []

    return result


def decision_status_from_mapping(
    raw: Mapping[str, Any] | None,
) -> Optional[DecisionStatus]:
    if not isinstance(raw, Mapping):
        return None
    analysis_status = str(raw.get("analysis_status") or "").upper()
    if analysis_status not in ANALYSIS_STATUSES:
        return None
    trade_action = (
        str(raw.get("trade_action") or ACTION_NO_TRADE).upper()
        if str(raw.get("trade_action") or "").upper() in TRADE_ACTIONS
        else ACTION_NO_TRADE
    )
    raw_confirm = raw.get("confirmation_state")
    if raw_confirm and str(raw_confirm).upper() in CONFIRMATION_STATES:
        confirm_val = str(raw_confirm).upper()
    elif analysis_status == ANALYSIS_VALID and trade_action in {
        ACTION_BUY,
        ACTION_SELL,
        ACTION_HOLD,
    }:
        confirm_val = CONFIRM_CONFIRMED
    else:
        confirm_val = CONFIRM_UNRESOLVED

    raw_conf = raw.get("confidence")
    raw_prob = raw.get("probability")
    conf_val: Optional[int] = None
    prob_val: Optional[float] = None
    if analysis_status == ANALYSIS_VALID and trade_action not in NON_DIRECTIONAL_TRADE_ACTIONS:
        if raw_conf is not None and not isinstance(raw_conf, bool):
            try:
                conf_val = int(raw_conf)
            except (ValueError, TypeError):
                conf_val = None
        if raw_prob is not None and not isinstance(raw_prob, bool):
            try:
                prob_val = float(raw_prob)
            except (ValueError, TypeError):
                prob_val = None

    return DecisionStatus(
        analysis_status=analysis_status,
        direction=str(raw.get("direction") or DIRECTION_NA).upper()
        if str(raw.get("direction") or "").upper() in DIRECTIONS
        else DIRECTION_NA,
        trade_action=trade_action,
        risk_status=str(raw.get("risk_status") or RISK_UNKNOWN).upper()
        if str(raw.get("risk_status") or "").upper() in RISK_STATUSES
        else RISK_UNKNOWN,
        confirmation_state=confirm_val,
        failure_class=raw.get("failure_class"),
        reason_codes=list(raw.get("reason_codes") or []),
        confidence=conf_val,
        probability=prob_val,
    )


def is_non_executable_status(status: DecisionStatus | Mapping[str, Any] | None) -> bool:
    """True when downstream Trader/Risk must not invent a directional plan."""
    if status is None:
        return False
    if isinstance(status, DecisionStatus):
        analysis_status = status.analysis_status
        trade_action = status.trade_action
        confirmation_state = status.confirmation_state
    elif isinstance(status, Mapping):
        analysis_status = str(status.get("analysis_status") or "").upper()
        trade_action = str(status.get("trade_action") or "").upper()
        confirmation_state = str(status.get("confirmation_state") or "").upper()
    else:
        return False
    if analysis_status in {
        ANALYSIS_INVALID_RUN,
        ANALYSIS_DATA_ERROR,
        ANALYSIS_ABSTAIN,
    }:
        return True
    if confirmation_state in {CONFIRM_UNRESOLVED, "UNRESOLVED"}:
        return True
    return trade_action in NON_DIRECTIONAL_TRADE_ACTIONS


def decision_status_from_state(
    state: Mapping[str, Any] | None,
) -> Optional[DecisionStatus]:
    """Read decision_status from graph state (top-level or manager_verdict)."""
    if not isinstance(state, Mapping):
        return None
    raw_top = state.get("decision_status")
    if isinstance(raw_top, DecisionStatus):
        return raw_top
    parsed = decision_status_from_mapping(
        raw_top if isinstance(raw_top, Mapping) else None
    )
    if parsed is not None:
        return parsed

    analysis_status = state.get("analysis_status")
    trade_action = state.get("trade_action")
    if analysis_status or trade_action:
        parsed = decision_status_from_mapping(
            {
                "analysis_status": analysis_status or ANALYSIS_ABSTAIN,
                "direction": state.get("direction") or DIRECTION_NA,
                "trade_action": trade_action or ACTION_NO_TRADE,
                "risk_status": state.get("risk_status") or RISK_UNKNOWN,
                "confirmation_state": state.get("confirmation_state"),
                "failure_class": state.get("failure_class"),
                "reason_codes": state.get("reason_codes") or [],
                "confidence": state.get("confidence"),
                "probability": state.get("probability"),
            }
        )
        if parsed is not None:
            return parsed

    for container_key in ("manager_verdict", "investment_debate_state", "result_data"):
        container = state.get(container_key)
        if not isinstance(container, Mapping):
            continue
        raw_container = container.get("decision_status")
        if isinstance(raw_container, DecisionStatus):
            return raw_container
        if isinstance(raw_container, Mapping):
            parsed = decision_status_from_mapping(raw_container)
            if parsed is not None:
                return parsed
        if container_key == "investment_debate_state":
            mv = container.get("manager_verdict")
            if isinstance(mv, Mapping):
                raw_mv = mv.get("decision_status")
                if isinstance(raw_mv, DecisionStatus):
                    return raw_mv
                if isinstance(raw_mv, Mapping):
                    parsed = decision_status_from_mapping(raw_mv)
                    if parsed is not None:
                        return parsed
    return None
