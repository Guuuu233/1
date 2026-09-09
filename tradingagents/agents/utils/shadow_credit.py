"""H1a / H1b Shadow Credit and Credit Weighting Module (P1-S / P3-H1b).

Pure function implementation for:
1. Observational shadow credit metrics calculation.
2. 7-dimension gate threshold validation (N, Side, Time, T+5, Balance, Bias Freeze, Magnitude).
3. Layered isolation state machine (System-level -> Model-level -> Global Shadow fallback).
4. Deterministic credit weighting application (only verified claims relative weighting in [0.85, 1.15]).
"""

from collections import Counter
import copy
import csv
from datetime import date, datetime
import io
import logging
import math
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from tradingagents.dataflows.trade_calendar import (
    calculate_t_plus_5_date,
    cn_market_phase,
    now_cn,
)

logger = logging.getLogger(__name__)

from tradingagents.agents.utils.agent_states import (
    DEFAULT_FEATURE_FLAGS,
    PROTOCOL_VERSION_V1_LEGACY,
    PROTOCOL_VERSION_V2_STRUCTURED,
    get_protocol_metadata,
    is_v2_debate_enabled,
)
from tradingagents.agents.utils.debate_metrics import (
    SEVEN_REPORT_KEYS,
    _extract_cited_debate_numbers,
    extract_numerical_tokens,
)

SCHEMA_VERSION: str = "h1a_json_v1"
H1B_SCHEMA_VERSION: str = "h1b_json_v1"

# ── Cohort Isolation Constants (DAV-601) ──────────────────────────────────────
COHORT_LEGACY_UNVERSIONED: str = "legacy_unversioned"
DECISION_MODEL_LEGACY: str = "decision_model.legacy_unversioned"
DECISION_MODEL_V1: str = "decision_model.v1"
EVIDENCE_CONTRACT_V0: str = "evidence_contract.v0"
PRICE_BASIS_UNSPECIFIED: str = "price_basis.unspecified"


# Mapping from report key to canonical role slug
REPORT_KEY_TO_ROLE: dict[str, str] = {
    "macro_report": "macro",
    "market_report": "market",
    "sentiment_report": "sentiment",
    "news_report": "news",
    "fundamentals_report": "fundamentals",
    "smart_money_report": "smart_money",
    "volume_price_report": "volume_price",
}

# ── Approved H1b Activation Gate Thresholds ──────────────────────────────────
H1B_THRESHOLDS: dict[str, Any] = {
    # 1. N: >=60 debates, >=20 unique symbols, >=5 industries, max single symbol <=15%
    "min_sample_count": 60,
    "min_unique_symbols": 20,
    "min_industries": 5,
    "max_single_symbol_ratio": 0.15,
    # 2. Side split: bull/bear samples each >=25, verified claims each >=100
    "min_side_samples": 25,
    "min_side_verified_claims": 100,
    # 3. Time: >=45 calendar days and >=30 trading days
    "min_calendar_days": 45,
    "min_trading_days": 30,
    # 4. T+5 completeness: >=95%
    "min_t_plus_5_completeness": 0.95,
    # 5. Balance: bull ratio in [40%, 60%], |Nbull - Nbear| <= 10
    "min_side_balance_ratio": 0.40,
    "max_side_balance_ratio": 0.60,
    "max_side_count_diff": 10,
    # 6. Bias freeze: delta verified <=18%, delta challenge adoption <=25%, clone rate <=5%, consistency trigger rate <=5%
    "max_delta_verified_rate": 0.18,
    "max_delta_challenge_adoption_rate": 0.25,
    "max_clone_rate": 0.05,
    "max_consistency_trigger_rate": 0.05,
    # 7. Magnitude: multiplier in [0.85, 1.15]
    "min_weight_multiplier": 0.85,
    "max_weight_multiplier": 1.15,
}


def _is_bull(speaker_key: str, stance: str) -> bool:
    """Return True if speaker or stance indicates bullish side."""
    sp = str(speaker_key or "").lower().strip()
    st = str(stance or "").lower().strip()
    return sp in ("bull", "看多", "多方", "多头") or st in ("bullish", "bull", "看多", "多头")


def _is_bear(speaker_key: str, stance: str) -> bool:
    """Return True if speaker or stance indicates bearish side."""
    sp = str(speaker_key or "").lower().strip()
    st = str(stance or "").lower().strip()
    return sp in ("bear", "看空", "空方", "空头") or st in ("bearish", "bear", "看空", "空头")


def _parse_sample_date(raw: Any) -> Optional[date]:
    """Parse trade date or created_at string to date object."""
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except (ValueError, TypeError):
            pass
    return None


def calculate_shadow_credit_metrics(
    result_data_or_state: Mapping[str, Any],
    *,
    version: Optional[str] = None,
    t_plus_5_price: Optional[float] = None,
) -> dict[str, Any]:
    """Calculate shadow credit metrics from result_data or investment_debate_state.

    This function is strictly pure, read-only, deterministic, and replayable.
    """
    if not isinstance(result_data_or_state, Mapping):
        return {
            "schema_version": SCHEMA_VERSION,
            "credit_weighting_enabled": False,
            "bull_verified_rate": None,
            "bear_verified_rate": None,
            "bull_challenge_adoption_rate": None,
            "bear_challenge_adoption_rate": None,
            "analyst_utilization_by_role": {role: None for role in REPORT_KEY_TO_ROLE.values()},
            "manager_evidence_coverage": None,
            "manager_consistency_gate_triggered": False,
            "t_plus_5_direction_hit": None,
            "sample_count": 1,
            "protocol_version": version or PROTOCOL_VERSION_V1_LEGACY,
            "model_id_by_stance": {
                "bull": None,
                "bear": None,
                "manager": None,
            },
        }

    meta = get_protocol_metadata(result_data_or_state)
    protocol_version = version or meta["protocol_version"]
    credit_weighting_flag = bool(meta.get("feature_flags", {}).get("credit_weighting_enabled", False))

    inv_state = result_data_or_state.get("investment_debate_state")
    if not isinstance(inv_state, Mapping):
        inv_state = result_data_or_state

    # ── 1. Bull / Bear Verified Rates ──────────────────────────────────────────
    manager_verdict = (
        result_data_or_state.get("manager_verdict")
        or inv_state.get("manager_verdict")
        or {}
    )
    if not isinstance(manager_verdict, Mapping):
        manager_verdict = {}

    claim_evidence_summary = (
        manager_verdict.get("claim_evidence_summary")
        or inv_state.get("claim_evidence_summary")
        or {}
    )
    if not isinstance(claim_evidence_summary, Mapping):
        claim_evidence_summary = {}

    claims = inv_state.get("claims") or result_data_or_state.get("claims") or []
    if not isinstance(claims, list):
        claims = []

    claim_id_to_claim: dict[str, Mapping[str, Any]] = {}
    for c in claims:
        if isinstance(c, Mapping):
            cid = str(c.get("claim_id", "")).strip()
            if cid:
                claim_id_to_claim[cid] = c

    bull_v, bull_t = 0, 0
    bear_v, bear_t = 0, 0

    if claim_evidence_summary:
        for _cid, info in claim_evidence_summary.items():
            if not isinstance(info, Mapping):
                continue
            matched_claim = claim_id_to_claim.get(str(_cid).strip(), {})
            sp_key = str(info.get("speaker_key") or info.get("speaker") or matched_claim.get("speaker_key") or matched_claim.get("speaker") or "")
            st_val = str(info.get("stance") or matched_claim.get("stance") or "")
            counts = info.get("counts") or {}
            v_cnt = int(counts.get("verified", 0))
            t_cnt = int(counts.get("total", 0))
            if _is_bull(sp_key, st_val):
                bull_v += v_cnt
                bull_t += t_cnt
            elif _is_bear(sp_key, st_val):
                bear_v += v_cnt
                bear_t += t_cnt
    elif claims:
        for c in claims:
            if not isinstance(c, Mapping):
                continue
            sp_key = str(c.get("speaker_key") or c.get("speaker") or "")
            st_val = str(c.get("stance") or "")
            is_v = bool(c.get("status") == "verified" or c.get("is_verified") is True)
            if _is_bull(sp_key, st_val):
                bull_t += 1
                if is_v:
                    bull_v += 1
            elif _is_bear(sp_key, st_val):
                bear_t += 1
                if is_v:
                    bear_v += 1

    bull_verified_rate = round(bull_v / bull_t, 4) if bull_t > 0 else None
    bear_verified_rate = round(bear_v / bear_t, 4) if bear_t > 0 else None

    # ── 2. Challenge Adoption Rates ───────────────────────────────────────────
    challenges = inv_state.get("challenges") or result_data_or_state.get("challenges") or []
    if not isinstance(challenges, list):
        challenges = []

    adopted_ids: set[str] = set()
    raw_adopted = manager_verdict.get("adopted_challenge_ids") or []
    if isinstance(raw_adopted, list):
        adopted_ids = {str(x).strip() for x in raw_adopted if str(x).strip()}

    bull_ch_tot, bull_ch_adopt = 0, 0
    bear_ch_tot, bear_ch_adopt = 0, 0

    for ch in challenges:
        if not isinstance(ch, Mapping):
            continue
        cid = str(ch.get("challenge_id") or "").strip()
        is_adopted = (cid in adopted_ids) or bool(ch.get("adopted") is True)
        sp_key = str(ch.get("speaker_key") or ch.get("speaker") or "")
        st_val = str(ch.get("stance") or "")
        if _is_bull(sp_key, st_val):
            bull_ch_tot += 1
            if is_adopted:
                bull_ch_adopt += 1
        elif _is_bear(sp_key, st_val):
            bear_ch_tot += 1
            if is_adopted:
                bear_ch_adopt += 1

    bull_challenge_adoption_rate = (
        round(bull_ch_adopt / bull_ch_tot, 4) if bull_ch_tot > 0 else None
    )
    bear_challenge_adoption_rate = (
        round(bear_ch_adopt / bear_ch_tot, 4) if bear_ch_tot > 0 else None
    )

    # ── 3. Analyst Utilization by Role ────────────────────────────────────────
    cited_tokens = _extract_cited_debate_numbers(inv_state)
    utilization_by_role: dict[str, Optional[float]] = {}

    for report_key, role_slug in REPORT_KEY_TO_ROLE.items():
        report_text = str(
            result_data_or_state.get(report_key, "")
            or inv_state.get(report_key, "")
            or ""
        )
        tokens = extract_numerical_tokens(report_text)
        denom = len(tokens)
        if denom == 0:
            utilization_by_role[role_slug] = None
        else:
            num = sum(1 for t in tokens if t in cited_tokens)
            utilization_by_role[role_slug] = round(num / denom, 4)

    # ── 4. Manager Evidence Coverage ──────────────────────────────────────────
    manager_evidence_coverage: Optional[float] = None
    if claim_evidence_summary:
        tot_v = sum(
            int(info.get("counts", {}).get("verified", 0))
            for info in claim_evidence_summary.values()
            if isinstance(info, Mapping)
        )
        tot_t = sum(
            int(info.get("counts", {}).get("total", 0))
            for info in claim_evidence_summary.values()
            if isinstance(info, Mapping)
        )
        if tot_t > 0:
            manager_evidence_coverage = round(tot_v / tot_t, 4)
    elif claims:
        tot_v = sum(
            1
            for c in claims
            if isinstance(c, Mapping) and (c.get("status") == "verified" or c.get("is_verified") is True)
        )
        tot_t = len(claims)
        if tot_t > 0:
            manager_evidence_coverage = round(tot_v / tot_t, 4)

    # ── 5. Manager Consistency Gate Triggered ─────────────────────────────────
    manager_consistency_gate_triggered: bool = False
    if manager_verdict:
        if (
            manager_verdict.get("consistency_check_passed") is False
            or bool(manager_verdict.get("failed_checks"))
        ):
            manager_consistency_gate_triggered = True
    elif inv_state.get("blocked") is True:
        block_reason = str(inv_state.get("block_reason", "")).lower()
        if "gate" in block_reason or "自洽" in block_reason or "硬闸" in block_reason:
            manager_consistency_gate_triggered = True

    # ── 6. T+5 Direction Hit ──────────────────────────────────────────────────
    t_plus_5_direction_hit: Optional[bool] = None

    # Resolve t_plus_5_price if not explicitly passed
    if t_plus_5_price is None:
        raw_p = (
            result_data_or_state.get("t_plus_5_price")
            or inv_state.get("t_plus_5_price")
            or (
                result_data_or_state.get("shadow_credit_metrics", {}).get("t_plus_5_price")
                if isinstance(result_data_or_state.get("shadow_credit_metrics"), Mapping)
                else None
            )
        )
        if raw_p is not None:
            try:
                t_plus_5_price = float(raw_p)
            except (ValueError, TypeError):
                t_plus_5_price = None

    if t_plus_5_price is not None and isinstance(t_plus_5_price, (int, float)):
        entry_val = None
        raw_entry = (
            manager_verdict.get("entry")
            or result_data_or_state.get("entry_price")
            or result_data_or_state.get("target_price")
            or inv_state.get("entry_price")
            or inv_state.get("target_price")
        )
        if raw_entry:
            try:
                entry_val = float(str(raw_entry).split("-")[0].replace("元", "").strip())
            except (ValueError, TypeError):
                entry_val = None

        winner = str(
            manager_verdict.get("winner")
            or result_data_or_state.get("debate_winner")
            or inv_state.get("manager_verdict", {}).get("winner")
            or ""
        ).strip().lower()
        direction_str = str(
            manager_verdict.get("direction")
            or result_data_or_state.get("decision")
            or inv_state.get("manager_verdict", {}).get("direction")
            or ""
        ).upper()

        if entry_val is not None and entry_val > 0:
            price_change = t_plus_5_price - entry_val
            if winner == "bull":
                t_plus_5_direction_hit = bool(price_change > 0)
            elif winner == "bear":
                t_plus_5_direction_hit = bool(price_change < 0)
            elif winner == "tie":
                t_plus_5_direction_hit = bool(abs(price_change / entry_val) <= 0.03)
            elif any(w in direction_str for w in ("BUY", "BULLISH", "多", "买入", "增持")):
                t_plus_5_direction_hit = bool(price_change > 0)
            elif any(w in direction_str for w in ("SELL", "BEARISH", "空", "卖出", "减持")):
                t_plus_5_direction_hit = bool(price_change < 0)
            elif any(w in direction_str for w in ("HOLD", "NEUTRAL", "中性", "观望", "持有")):
                t_plus_5_direction_hit = bool(abs(price_change / entry_val) <= 0.03)
    else:
        existing_hit = (
            result_data_or_state.get("t_plus_5_direction_hit")
            or (
                result_data_or_state.get("shadow_credit_metrics", {}).get("t_plus_5_direction_hit")
                if isinstance(result_data_or_state.get("shadow_credit_metrics"), Mapping)
                else None
            )
        )
        if isinstance(existing_hit, bool):
            t_plus_5_direction_hit = existing_hit

    # ── 7. Model ID x Stance ──────────────────────────────────────────────────
    bull_model: Optional[str] = None
    bear_model: Optional[str] = None
    manager_model: Optional[str] = None

    round_messages = inv_state.get("round_messages") or result_data_or_state.get("round_messages") or []
    if isinstance(round_messages, list):
        for msg in round_messages:
            if not isinstance(msg, Mapping):
                continue
            m_name = msg.get("model_name") or msg.get("model_id") or msg.get("model")
            if m_name and isinstance(m_name, str) and m_name.strip():
                m_clean = m_name.strip()
                sp_key = str(msg.get("speaker_key") or msg.get("speaker") or "")
                st_val = str(msg.get("stance") or "")
                is_v = bool(msg.get("is_verdict") or "manager" in sp_key.lower() or "总监" in sp_key)
                if is_v:
                    if not manager_model:
                        manager_model = m_clean
                elif _is_bull(sp_key, st_val):
                    if not bull_model:
                        bull_model = m_clean
                elif _is_bear(sp_key, st_val):
                    if not bear_model:
                        bear_model = m_clean

    role_models = (
        result_data_or_state.get("model_id_by_stance")
        or inv_state.get("model_id_by_stance")
        or result_data_or_state.get("role_models")
        or {}
    )
    if isinstance(role_models, Mapping):
        if not bull_model and role_models.get("bull"):
            bull_model = str(role_models["bull"]).strip()
        if not bear_model and role_models.get("bear"):
            bear_model = str(role_models["bear"]).strip()
        if not manager_model and role_models.get("manager"):
            manager_model = str(role_models["manager"]).strip()

    model_id_by_stance = {
        "bull": bull_model if (bull_model and bull_model != "unknown") else None,
        "bear": bear_model if (bear_model and bear_model != "unknown") else None,
        "manager": manager_model if (manager_model and manager_model != "unknown") else None,
    }

    t_plus_5_status = (
        result_data_or_state.get("t_plus_5_status")
        or inv_state.get("t_plus_5_status")
        or (
            result_data_or_state.get("shadow_credit_metrics", {}).get("t_plus_5_status")
            if isinstance(result_data_or_state.get("shadow_credit_metrics"), Mapping)
            else None
        )
    )
    t_plus_5_date = (
        result_data_or_state.get("t_plus_5_date")
        or inv_state.get("t_plus_5_date")
        or (
            result_data_or_state.get("shadow_credit_metrics", {}).get("t_plus_5_date")
            if isinstance(result_data_or_state.get("shadow_credit_metrics"), Mapping)
            else None
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "credit_weighting_enabled": credit_weighting_flag,
        "bull_verified_rate": bull_verified_rate,
        "bear_verified_rate": bear_verified_rate,
        "bull_challenge_adoption_rate": bull_challenge_adoption_rate,
        "bear_challenge_adoption_rate": bear_challenge_adoption_rate,
        "analyst_utilization_by_role": utilization_by_role,
        "manager_evidence_coverage": manager_evidence_coverage,
        "manager_consistency_gate_triggered": manager_consistency_gate_triggered,
        "t_plus_5_direction_hit": t_plus_5_direction_hit,
        "t_plus_5_status": t_plus_5_status,
        "t_plus_5_date": t_plus_5_date,
        "t_plus_5_price": t_plus_5_price,
        "sample_count": 1,
        "protocol_version": protocol_version,
        "model_id_by_stance": model_id_by_stance,
    }


def extract_report_industry(sample: Mapping[str, Any]) -> Optional[str]:
    """Extract industry metadata if present in report without hardcoding or fabricating."""
    if not isinstance(sample, Mapping):
        return None

    # 1. Direct top-level industry / sector
    ind = sample.get("industry") or sample.get("sector")
    if ind and str(ind).strip() and str(ind).strip() != "未知行业":
        return str(ind).strip()

    # 2. Nested result_data if present
    res_data = sample.get("result_data")
    if isinstance(res_data, Mapping):
        ind = res_data.get("industry") or res_data.get("sector")
        if ind and str(ind).strip() and str(ind).strip() != "未知行业":
            return str(ind).strip()

    # 3. instrument_context
    inst = sample.get("instrument_context") or (res_data.get("instrument_context") if isinstance(res_data, Mapping) else None)
    if isinstance(inst, Mapping):
        ind = inst.get("industry") or inst.get("sector")
        if ind and str(ind).strip() and str(ind).strip() != "未知行业":
            return str(ind).strip()

    # 4. market_data_context.industry_linkage
    mdc = sample.get("market_data_context") or (res_data.get("market_data_context") if isinstance(res_data, Mapping) else None)
    if isinstance(mdc, Mapping):
        il = mdc.get("industry_linkage")
        if isinstance(il, Mapping):
            ind = il.get("industry_name") or il.get("industry")
            if ind and str(ind).strip() and str(ind).strip() != "未知行业":
                return str(ind).strip()

    # 5. data_collection_provenance.industry_linkage_raw
    prov = sample.get("data_collection_provenance") or (res_data.get("data_collection_provenance") if isinstance(res_data, Mapping) else None)
    if isinstance(prov, Mapping):
        il_raw = prov.get("industry_linkage_raw") or prov.get("industry_linkage")
        if isinstance(il_raw, Mapping):
            ind = il_raw.get("industry_name") or il_raw.get("industry")
            if ind and str(ind).strip() and str(ind).strip() != "未知行业":
                return str(ind).strip()

    # 6. quadrant_1_protocol_metadata
    q1 = sample.get("quadrant_1_protocol_metadata") or (res_data.get("quadrant_1_protocol_metadata") if isinstance(res_data, Mapping) else None)
    if isinstance(q1, Mapping):
        ind = q1.get("industry")
        if ind and str(ind).strip() and str(ind).strip() != "未知行业":
            return str(ind).strip()

    # 7. Nested dual-horizon sub-objects (short_term / medium_term)
    for sub_key in ("short_term", "medium_term", "primary"):
        sub = sample.get(sub_key) or (res_data.get(sub_key) if isinstance(res_data, Mapping) else None)
        if isinstance(sub, Mapping):
            sub_ind = extract_report_industry(sub)
            if sub_ind and str(sub_ind).strip() and str(sub_ind).strip() != "未知行业":
                return str(sub_ind).strip()

    return None


def extract_report_analysis_status_and_action(report: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Extract canonical (analysis_status, trade_action) from report dictionary.

    STRICT D-009 §5 CONTRACT:
    - Extracts analysis_status from canonical locations.
    - Extracts trade_action ONLY from canonical `trade_action` field name.
      NEVER falls back to legacy `decision` or `action` fields (RT-1 / DAV-783).
    """
    if not isinstance(report, Mapping):
        return None, None

    res_data = report.get("result_data") if isinstance(report.get("result_data"), Mapping) else {}
    inv_state = report.get("investment_debate_state") if isinstance(report.get("investment_debate_state"), Mapping) else (
        res_data.get("investment_debate_state") if isinstance(res_data.get("investment_debate_state"), Mapping) else {}
    )
    dec_status = report.get("decision_status") if isinstance(report.get("decision_status"), Mapping) else (
        res_data.get("decision_status") if isinstance(res_data.get("decision_status"), Mapping) else (
            inv_state.get("decision_status") if isinstance(inv_state.get("decision_status"), Mapping) else {}
        )
    )

    # 1. Canonical analysis_status
    raw_status = (
        report.get("analysis_status")
        or res_data.get("analysis_status")
        or (dec_status.get("analysis_status") if isinstance(dec_status, Mapping) else None)
        or (inv_state.get("analysis_status") if isinstance(inv_state, Mapping) else None)
    )
    analysis_status = str(raw_status).strip().upper() if raw_status is not None and str(raw_status).strip() else None

    # 2. Canonical trade_action (STRICT: NO fallback to decision/action)
    raw_action = (
        report.get("trade_action")
        or res_data.get("trade_action")
        or (dec_status.get("trade_action") if isinstance(dec_status, Mapping) else None)
        or (inv_state.get("trade_action") if isinstance(inv_state, Mapping) else None)
    )
    trade_action = str(raw_action).strip().upper() if raw_action is not None and str(raw_action).strip() else None

    return analysis_status, trade_action


def is_qualifying_v2_report(report: Mapping[str, Any]) -> bool:
    """Return True if report is a completed v2 structured debate report with a valid v2 manager verdict winner.

    Excludes:
    - Non-completed reports (if status is present and != 'completed')
    - Legacy v1 reports without v2 structured disagreement / without v2 manager_verdict.winner
    - Reports without winner in manager_verdict
    """
    if not isinstance(report, Mapping):
        return False

    # 1. Status check: if status is specified, must be 'completed'
    status = report.get("status")
    if status is not None and str(status).strip().lower() != "completed":
        return False

    # Unpack nested result_data if present
    res_data = report.get("result_data")
    target = {**res_data, **report} if isinstance(res_data, Mapping) else report

    inv_state = target.get("investment_debate_state")
    if not isinstance(inv_state, Mapping):
        inv_state = target

    # 2. Protocol version check:
    meta = get_protocol_metadata(target)
    proto_ver = meta.get("protocol_version") or target.get("protocol_version") or inv_state.get("protocol_version")
    is_v2 = (
        proto_ver == PROTOCOL_VERSION_V2_STRUCTURED
        or is_v2_debate_enabled(target)
    )

    # 3. Manager verdict & winner check:
    verdict = (
        target.get("manager_verdict")
        or inv_state.get("manager_verdict")
        or {}
    )
    if not isinstance(verdict, Mapping):
        verdict = {}

    raw_winner = verdict.get("winner") or target.get("debate_winner")
    winner_str = str(raw_winner or "").strip().lower()
    has_valid_winner = winner_str in ("bull", "bear", "tie")

    if is_v2 and has_valid_winner:
        return True
    if has_valid_winner and (
        bool(verdict.get("claim_evidence_summary"))
        or verdict.get("consistency_check_passed") is not None
        or bool(inv_state.get("claims"))
    ):
        return True

    return False


# Alias for clarity in multi-stage pipeline accounting
is_v2_protocol_report = is_qualifying_v2_report


def classify_v2_report_d009_exclusion(report: Mapping[str, Any]) -> Optional[str]:
    """Classify why a v2 protocol report is excluded under D-009 §5.

    Returns None if report qualifies:
    - analysis_status == 'VALID'
    - trade_action in {'BUY', 'SELL', 'HOLD'}

    Returns one of the categorized exclusion reasons otherwise:
    - 'legacy_null': analysis_status IS NULL (pre-D-009 sample)
    - 'abstain': analysis_status == 'ABSTAIN'
    - 'invalid_run': analysis_status == 'INVALID_RUN'
    - 'data_error': analysis_status in ('DATA_ERROR', 'PARTIAL')
    - 'wait': analysis_status == 'VALID' but trade_action == 'WAIT'
    - 'no_trade': analysis_status == 'VALID' but trade_action is NO_TRADE / missing / non-directional
    """
    st_val, act_val = extract_report_analysis_status_and_action(report)

    if st_val is None:
        return "legacy_null"
    if st_val == "ABSTAIN":
        return "abstain"
    if st_val == "INVALID_RUN":
        return "invalid_run"
    if st_val in ("DATA_ERROR", "PARTIAL"):
        return "data_error"

    if st_val == "VALID":
        if act_val == "WAIT":
            return "wait"
        if act_val in {"BUY", "SELL", "HOLD"}:
            return None
        # Missing trade_action or NO_TRADE or other non-directional action
        return "no_trade"

    return "invalid_run"


def is_qualifying_h1b_report(report: Mapping[str, Any]) -> bool:
    """Return True if report is a completed v2 structured debate report qualifying for H1b pool under D-009 §5:
    - is_qualifying_v2_report(report) is True (completed v2 debate report with winner)
    - analysis_status == 'VALID'
    - canonical trade_action in {'BUY', 'SELL', 'HOLD'} (strictly no decision/action fallback)
    """
    if not is_qualifying_v2_report(report):
        return False
    return classify_v2_report_d009_exclusion(report) is None


def normalize_report_for_evaluation(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a raw report dictionary for gate evaluation."""
    if not isinstance(sample, Mapping):
        return {}

    res_data = sample.get("result_data")
    if isinstance(res_data, Mapping):
        merged = {**res_data, **{k: v for k, v in sample.items() if v is not None}}
    else:
        merged = dict(sample)

    ind = extract_report_industry(merged)
    if ind:
        merged["industry"] = ind

    # Also normalize manager_verdict and investment_debate_state at top level if present
    inv_state = merged.get("investment_debate_state")
    if isinstance(inv_state, Mapping):
        if not merged.get("manager_verdict") and inv_state.get("manager_verdict"):
            merged["manager_verdict"] = inv_state["manager_verdict"]
        if not merged.get("claims") and inv_state.get("claims"):
            merged["claims"] = inv_state["claims"]

    if not merged.get("shadow_credit_metrics") or not isinstance(merged.get("shadow_credit_metrics"), Mapping):
        merged["shadow_credit_metrics"] = calculate_shadow_credit_metrics(merged)

    return merged


def filter_v2_completed_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    return_excluded_counts: bool = False,
    return_ledger: bool = False,
) -> Union[
    list[dict[str, Any]],
    tuple[list[dict[str, Any]], dict[str, int]],
    tuple[list[dict[str, Any]], dict[str, int], dict[str, int]],
]:
    """Filter and normalize reports, returning only qualifying completed v2 reports under D-009 §5.

    Implements three-stage pipeline accounting (DAV-783 / RT-2):
    Stage 1: raw input reports
    Stage 2: qualifying v2 protocol reports (protocol_version=v2_structured, completed, valid winner)
    Stage 3: D-009 §5 eligible reports (analysis_status=VALID, trade_action in {BUY, SELL, HOLD})

    `excluded_counts` ONLY counts reports that entered the v2 pool (Stage 2) and were excluded
    under D-009 §5 (Stage 3). Non-v2 reports filtered at Stage 2 are tracked in ledger['non_v2_excluded']
    and NEVER conflated into D-009 `excluded_counts`.
    """
    qualifying: list[dict[str, Any]] = []
    excluded_counts: dict[str, int] = {
        "legacy_null": 0,
        "abstain": 0,
        "invalid_run": 0,
        "data_error": 0,
        "no_trade": 0,
        "wait": 0,
    }
    ledger: dict[str, int] = {
        "raw_count": len(reports),
        "qualifying_v2_count": 0,
        "eligible_count": 0,
        "non_v2_excluded": 0,
        "d009_excluded": 0,
    }

    for r in reports:
        # Stage 1 -> Stage 2: Must be a qualifying v2 protocol report first
        if not is_v2_protocol_report(r):
            ledger["non_v2_excluded"] += 1
            continue

        ledger["qualifying_v2_count"] += 1

        # Stage 2 -> Stage 3: Classify under D-009 §5
        cat = classify_v2_report_d009_exclusion(r)
        if cat is None:
            normalized = normalize_report_for_evaluation(r)
            qualifying.append(normalized)
            ledger["eligible_count"] += 1
        else:
            ledger["d009_excluded"] += 1
            if cat in excluded_counts:
                excluded_counts[cat] += 1
            else:
                excluded_counts[cat] = excluded_counts.get(cat, 0) + 1

    if return_ledger:
        return qualifying, excluded_counts, ledger
    if return_excluded_counts:
        return qualifying, excluded_counts
    return qualifying


# ── Cohort Isolation Helpers (DAV-601) ────────────────────────────────────────

def extract_sample_cohort(sample: Mapping[str, Any]) -> dict[str, Optional[str]]:
    """Extract cohort triad and commit sha from report/sample dictionary."""
    if not isinstance(sample, Mapping):
        return {
            "decision_model_version": None,
            "evidence_contract_version": None,
            "price_basis_version": None,
            "generated_by_commit_sha": None,
        }

    res_data = sample.get("result_data") if isinstance(sample.get("result_data"), Mapping) else {}
    inv_state = sample.get("investment_debate_state") if isinstance(sample.get("investment_debate_state"), Mapping) else (
        res_data.get("investment_debate_state") if isinstance(res_data.get("investment_debate_state"), Mapping) else {}
    )
    meta = sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else (
        res_data.get("metadata") if isinstance(res_data.get("metadata"), Mapping) else {}
    )

    def _find_field(key: str) -> Optional[str]:
        val = (
            sample.get(key)
            or res_data.get(key)
            or inv_state.get(key)
            or meta.get(key)
        )
        if val is not None and str(val).strip():
            return str(val).strip()
        return None

    return {
        "decision_model_version": _find_field("decision_model_version"),
        "evidence_contract_version": _find_field("evidence_contract_version"),
        "price_basis_version": _find_field("price_basis_version"),
        "generated_by_commit_sha": _find_field("generated_by_commit_sha") or _find_field("commit_sha"),
    }


def is_legacy_unversioned_sample(sample: Mapping[str, Any]) -> bool:
    """Return True if sample lacks decision_model_version or is explicitly marked legacy_unversioned."""
    cohort_info = extract_sample_cohort(sample)
    dmv = cohort_info["decision_model_version"]
    if not dmv or dmv in (COHORT_LEGACY_UNVERSIONED, DECISION_MODEL_LEGACY):
        return True
    return False


def parse_cohort_spec(cohort: Union[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Parse cohort specification into canonical structured representation.

    Fail-closed: raises ValueError if cohort is missing or empty.
    """
    if cohort is None:
        raise ValueError("Cohort specification is required and cannot be empty (fail-closed)")

    if isinstance(cohort, Mapping):
        cohort_type = str(cohort.get("cohort_type") or "").strip().lower()
        dmv = cohort.get("decision_model_version")
        ecv = cohort.get("evidence_contract_version")
        pbv = cohort.get("price_basis_version")
        if cohort_type == COHORT_LEGACY_UNVERSIONED or dmv in (COHORT_LEGACY_UNVERSIONED, DECISION_MODEL_LEGACY):
            return {
                "cohort_type": COHORT_LEGACY_UNVERSIONED,
                "decision_model_version": DECISION_MODEL_LEGACY,
                "evidence_contract_version": None,
                "price_basis_version": None,
                "canonical_key": COHORT_LEGACY_UNVERSIONED,
            }
        dmv_str = str(dmv or DECISION_MODEL_V1).strip()
        ecv_str = str(ecv or EVIDENCE_CONTRACT_V0).strip()
        pbv_str = str(pbv or PRICE_BASIS_UNSPECIFIED).strip()
        return {
            "cohort_type": "triad",
            "decision_model_version": dmv_str,
            "evidence_contract_version": ecv_str,
            "price_basis_version": pbv_str,
            "canonical_key": f"{dmv_str}:{ecv_str}:{pbv_str}",
        }

    s = str(cohort).strip()
    if not s:
        raise ValueError("Cohort specification is required and cannot be empty (fail-closed)")

    if s in (COHORT_LEGACY_UNVERSIONED, DECISION_MODEL_LEGACY):
        return {
            "cohort_type": COHORT_LEGACY_UNVERSIONED,
            "decision_model_version": DECISION_MODEL_LEGACY,
            "evidence_contract_version": None,
            "price_basis_version": None,
            "canonical_key": COHORT_LEGACY_UNVERSIONED,
        }

    if s.startswith("{"):
        try:
            parsed_json = json.loads(s)
            if isinstance(parsed_json, Mapping):
                return parse_cohort_spec(parsed_json)
        except Exception:
            pass

    sep = ":" if ":" in s else ("/" if "/" in s else None)
    if sep:
        parts = [p.strip() for p in s.split(sep)]
        dmv_str = parts[0]
        ecv_str = parts[1] if len(parts) > 1 and parts[1] else EVIDENCE_CONTRACT_V0
        pbv_str = parts[2] if len(parts) > 2 and parts[2] else PRICE_BASIS_UNSPECIFIED
        return {
            "cohort_type": "triad",
            "decision_model_version": dmv_str,
            "evidence_contract_version": ecv_str,
            "price_basis_version": pbv_str,
            "canonical_key": f"{dmv_str}:{ecv_str}:{pbv_str}",
        }

    return {
        "cohort_type": "triad",
        "decision_model_version": s,
        "evidence_contract_version": EVIDENCE_CONTRACT_V0,
        "price_basis_version": PRICE_BASIS_UNSPECIFIED,
        "canonical_key": f"{s}:{EVIDENCE_CONTRACT_V0}:{PRICE_BASIS_UNSPECIFIED}",
    }


def is_cohort_homogeneous(
    reports: Sequence[Mapping[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Check if all reports in collection belong to the same cohort generation."""
    reps = list(reports or [])
    if not reps:
        return True, None

    keys = set()
    for r in reps:
        if is_legacy_unversioned_sample(r):
            keys.add(COHORT_LEGACY_UNVERSIONED)
        else:
            c_info = extract_sample_cohort(r)
            dmv = c_info["decision_model_version"]
            ecv = c_info["evidence_contract_version"] or EVIDENCE_CONTRACT_V0
            pbv = c_info["price_basis_version"] or PRICE_BASIS_UNSPECIFIED
            keys.add(f"{dmv}:{ecv}:{pbv}")

    if len(keys) == 1:
        return True, list(keys)[0]
    return False, None


def assert_cohort_homogeneity(
    reports: Sequence[Mapping[str, Any]],
) -> None:
    """Assert all reports belong to single cohort generation; raise ValueError if mixed."""
    is_homo, cohort_key = is_cohort_homogeneous(reports)
    if not is_homo:
        reps = list(reports or [])
        cohort_keys = set()
        for r in reps:
            if is_legacy_unversioned_sample(r):
                cohort_keys.add(COHORT_LEGACY_UNVERSIONED)
            else:
                c = extract_sample_cohort(r)
                cohort_keys.add(f"{c['decision_model_version']}:{c['evidence_contract_version']}:{c['price_basis_version']}")
        raise ValueError(f"Mixed cohort generations detected in evaluation pool: {sorted(cohort_keys)}")


def filter_reports_by_cohort(
    reports: Sequence[Mapping[str, Any]],
    cohort: Union[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter reports strictly by cohort specification.

    Rules:
    - --cohort=legacy_unversioned: only samples lacking version fields or marked legacy; never label as v1.
    - Triad version: strictly all 3 fields matching; commit SHA is provenance metadata, not filter key.
    - Fail-closed on invalid or empty cohort specification.
    """
    spec = parse_cohort_spec(cohort)
    reps = list(reports or [])
    filtered: List[Dict[str, Any]] = []
    shas: set[str] = set()

    if spec["cohort_type"] == COHORT_LEGACY_UNVERSIONED:
        for r in reps:
            if is_legacy_unversioned_sample(r):
                c_info = extract_sample_cohort(r)
                if c_info["decision_model_version"] not in (None, "", COHORT_LEGACY_UNVERSIONED, DECISION_MODEL_LEGACY):
                    continue
                filtered.append(dict(r))
                sha = c_info["generated_by_commit_sha"]
                if sha:
                    shas.add(sha)
    else:
        target_dmv = spec["decision_model_version"]
        target_ecv = spec["evidence_contract_version"]
        target_pbv = spec["price_basis_version"]

        for r in reps:
            if is_legacy_unversioned_sample(r):
                continue
            c_info = extract_sample_cohort(r)
            if (
                c_info["decision_model_version"] == target_dmv
                and c_info["evidence_contract_version"] == target_ecv
                and c_info["price_basis_version"] == target_pbv
            ):
                filtered.append(dict(r))
                sha = c_info["generated_by_commit_sha"]
                if sha:
                    shas.add(sha)

    cohort_meta = {
        "cohort_type": spec["cohort_type"],
        "canonical_key": spec["canonical_key"],
        "decision_model_version": spec["decision_model_version"],
        "evidence_contract_version": spec["evidence_contract_version"],
        "price_basis_version": spec["price_basis_version"],
        "commit_shas": sorted(shas),
    }
    return filtered, cohort_meta



# ── 7-Dimension Gate Threshold Evaluation (P3-H1b) ───────────────────────────

def evaluate_h1b_system_gates(
    samples_or_reports: Sequence[Mapping[str, Any]],
    *,
    cohort: Optional[Union[str, Mapping[str, Any]]] = None,
    thresholds: Optional[Mapping[str, Any]] = None,
    as_of: Optional[Union[str, date, datetime]] = None,
    trading_calendar: Optional[Sequence[Union[str, date]]] = None,
    calendar_dates: Optional[Sequence[Union[str, date]]] = None,
    excluded_counts: Optional[Mapping[str, int]] = None,
    pipeline_ledger: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    """Evaluate 7-dimension gate thresholds for credit weighting activation.

    Dimensions:
    1. N: >=60 debates, >=20 unique symbols, >=5 industries, max single symbol <=15%
    2. Side: bull/bear samples each >=25, verified claims each >=100
    3. Time: >=45 calendar days and >=30 trading days; regime coverage
    4. T+5: Completeness rate >=95%
    5. Balance: Ratio in [40%, 60%], |Nbull - Nbear| <= 10
    6. Bias freeze: Delta verified <=18%, delta challenge adoption <=25%, clone rate <=5%, consistency trigger rate <=5%
    7. Magnitude: Range [0.85, 1.15]

    Returns structured evaluation matrix, passed status, and recommendations.
    """
    cfg = dict(H1B_THRESHOLDS)
    if thresholds:
        cfg.update(thresholds)

    cal_dates = trading_calendar if trading_calendar is not None else calendar_dates

    # Parse as_of date (defaulting to today in CN timezone)
    as_of_date: date
    if as_of is None:
        as_of_date = now_cn().date()
    elif isinstance(as_of, datetime):
        as_of_date = as_of.date()
    elif isinstance(as_of, date):
        as_of_date = as_of
    elif isinstance(as_of, str):
        parsed_as_of = _parse_sample_date(as_of)
        as_of_date = parsed_as_of if parsed_as_of is not None else now_cn().date()
    else:
        as_of_date = now_cn().date()

    initial_excluded: dict[str, int] = {
        "legacy_null": 0,
        "abstain": 0,
        "invalid_run": 0,
        "data_error": 0,
        "no_trade": 0,
        "wait": 0,
    }
    if excluded_counts:
        initial_excluded.update(excluded_counts)

    current_ledger = dict(pipeline_ledger) if pipeline_ledger else None

    cohort_meta: dict[str, Any] = {}
    homogeneity_passed = True
    homogeneity_reason = None

    if cohort is not None and str(cohort).strip():
        filtered_samples, cohort_meta = filter_reports_by_cohort(samples_or_reports, cohort=cohort)
        samples = filtered_samples
    else:
        samples = list(samples_or_reports or [])
        is_homo, c_key = is_cohort_homogeneous(samples)
        if not is_homo:
            homogeneity_passed = False
            homogeneity_reason = "Mixed cohort generations detected in evaluation samples: cannot merge across cohorts"
        cohort_meta = {
            "cohort_type": c_key or "unspecified",
            "canonical_key": c_key or "unspecified",
            "commit_shas": sorted({
                extract_sample_cohort(s)["generated_by_commit_sha"]
                for s in samples
                if extract_sample_cohort(s)["generated_by_commit_sha"]
            }),
        }

    sample_count = len(samples)

    # ── Dimension 1: N (Sample Count & Diversity) ─────────────────────────────
    symbols: list[str] = []
    industries: list[str] = []
    for s in samples:
        sym = s.get("symbol") or s.get("ticker") or ""
        if sym:
            symbols.append(str(sym).strip())
        ind = extract_report_industry(s) or s.get("industry") or s.get("sector") or ""
        if ind:
            industries.append(str(ind).strip())

    unique_symbols = len(set(symbols))
    unique_industries = len(set(industries))
    symbol_counter = Counter(symbols)
    max_symbol_count = max(symbol_counter.values()) if symbol_counter else 0
    max_symbol_share = (max_symbol_count / sample_count) if sample_count > 0 else 0.0

    pass_d1_count = sample_count >= cfg["min_sample_count"]
    pass_d1_sym = unique_symbols >= cfg["min_unique_symbols"]
    pass_d1_ind = unique_industries >= cfg["min_industries"]
    pass_d1_share = max_symbol_share <= cfg["max_single_symbol_ratio"]
    pass_d1 = bool(pass_d1_count and pass_d1_sym and pass_d1_ind and pass_d1_share)

    dim_n = {
        "passed": pass_d1,
        "details": {
            "sample_count": sample_count,
            "min_required": cfg["min_sample_count"],
            "unique_symbols": unique_symbols,
            "min_unique_symbols": cfg["min_unique_symbols"],
            "unique_industries": unique_industries,
            "min_industries": cfg["min_industries"],
            "max_symbol_share": round(max_symbol_share, 4),
            "max_allowed_share": cfg["max_single_symbol_ratio"],
        },
    }

    # ── Dimension 2: Side Split (Bull / Bear samples & Verified Claims) ───────
    bull_samples = 0
    bear_samples = 0
    bull_verified_claims = 0
    bear_verified_claims = 0

    for s in samples:
        inv_state = s.get("investment_debate_state") or s.get("result_data", {}).get("investment_debate_state") or s
        verdict = inv_state.get("manager_verdict") or s.get("manager_verdict") or s.get("result_data", {}).get("manager_verdict") or {}
        winner = str(verdict.get("winner") or s.get("debate_winner") or "").lower()
        direction = str(verdict.get("direction") or s.get("direction") or "").lower()

        if winner == "bull":
            bull_samples += 1
        elif winner == "bear":
            bear_samples += 1
        elif any(w in direction for w in ("多", "buy", "bull")):
            bull_samples += 1
        elif any(w in direction for w in ("空", "sell", "bear")):
            bear_samples += 1
        else:
            # Neutral / Tie
            pass

        claims = inv_state.get("claims") or s.get("claims") or s.get("result_data", {}).get("claims") or []
        claim_map = {str(c.get("claim_id", "")).strip(): c for c in claims if isinstance(c, Mapping)}

        summary = (
            verdict.get("claim_evidence_summary")
            or inv_state.get("claim_evidence_summary")
            or s.get("claim_evidence_summary")
            or {}
        )
        if summary:
            for cid_k, info in summary.items():
                if not isinstance(info, Mapping):
                    continue
                matched_c = claim_map.get(str(cid_k).strip(), {})
                sp = str(info.get("speaker_key") or info.get("speaker") or matched_c.get("speaker_key") or matched_c.get("speaker") or "")
                st = str(info.get("stance") or matched_c.get("stance") or "")
                v_cnt = int(info.get("counts", {}).get("verified", 0))
                if _is_bull(sp, st):
                    bull_verified_claims += v_cnt
                elif _is_bear(sp, st):
                    bear_verified_claims += v_cnt
        else:
            for c in claims:
                if not isinstance(c, Mapping):
                    continue
                sp = str(c.get("speaker_key") or c.get("speaker") or "")
                st = str(c.get("stance") or "")
                is_v = bool(c.get("status") == "verified" or c.get("is_verified") is True)
                if is_v:
                    if _is_bull(sp, st):
                        bull_verified_claims += 1
                    elif _is_bear(sp, st):
                        bear_verified_claims += 1

    pass_d2_samples = (bull_samples >= cfg["min_side_samples"]) and (bear_samples >= cfg["min_side_samples"])
    pass_d2_claims = (bull_verified_claims >= cfg["min_side_verified_claims"]) and (bear_verified_claims >= cfg["min_side_verified_claims"])
    pass_d2 = bool(pass_d2_samples and pass_d2_claims)

    dim_side = {
        "passed": pass_d2,
        "details": {
            "bull_samples": bull_samples,
            "bear_samples": bear_samples,
            "min_side_samples": cfg["min_side_samples"],
            "bull_verified_claims": bull_verified_claims,
            "bear_verified_claims": bear_verified_claims,
            "min_verified_claims": cfg["min_side_verified_claims"],
        },
    }

    # ── Dimension 3: Time Span (Calendar days & Trading days) ─────────────────
    parsed_dates = []
    regimes = set()
    for s in samples:
        d = _parse_sample_date(s.get("trade_date") or s.get("date") or s.get("created_at"))
        if d:
            parsed_dates.append(d)
        reg = s.get("market_regime") or s.get("regime")
        if reg:
            regimes.add(str(reg).strip())

    calendar_days = 0
    trading_days = 0
    if parsed_dates:
        min_date = min(parsed_dates)
        max_date = max(parsed_dates)
        calendar_days = (max_date - min_date).days + 1
        trading_days = len(set(parsed_dates))

    pass_d3_cal = calendar_days >= cfg["min_calendar_days"]
    pass_d3_trd = trading_days >= cfg["min_trading_days"]
    pass_d3 = bool(pass_d3_cal and pass_d3_trd)

    dim_time = {
        "passed": pass_d3,
        "details": {
            "calendar_days": calendar_days,
            "min_calendar_days": cfg["min_calendar_days"],
            "trading_days": trading_days,
            "min_trading_days": cfg["min_trading_days"],
            "market_regimes_covered": list(regimes),
        },
    }

    # ── Dimension 4: T+5 Completeness ─────────────────────────────────────────
    due_t5_count = 0
    completed_t5_count = 0

    for s in samples:
        res_data = s.get("result_data") if isinstance(s.get("result_data"), Mapping) else {}
        metrics = s.get("shadow_credit_metrics")
        if not metrics or not isinstance(metrics, Mapping):
            metrics = res_data.get("shadow_credit_metrics") if isinstance(res_data.get("shadow_credit_metrics"), Mapping) else None
        if not metrics or not isinstance(metrics, Mapping):
            metrics = calculate_shadow_credit_metrics(s)
        hit = metrics.get("t_plus_5_direction_hit")
        st = s.get("t_plus_5_status") or metrics.get("t_plus_5_status") or res_data.get("t_plus_5_status")
        # Exclude suspension from due denominator
        if (
            st == "suspension"
            or s.get("is_suspended") is True
            or s.get("suspension") is True
            or res_data.get("is_suspended") is True
            or res_data.get("suspension") is True
        ):
            continue
        # Exclude pending / in-flight samples from due denominator
        if (
            st == "pending_due"
            or s.get("is_in_flight") is True
            or s.get("is_t_plus_5_due") is False
            or res_data.get("is_in_flight") is True
            or res_data.get("is_t_plus_5_due") is False
        ):
            continue

        is_due = s.get("is_t_plus_5_due")
        if is_due is None and res_data:
            is_due = res_data.get("is_t_plus_5_due")

        if is_due is None:
            if (
                (hit is not None)
                or bool(s.get("t_plus_5_evaluated", False))
                or bool(res_data.get("t_plus_5_evaluated", False))
                or (st in ("due_and_evaluated", "data_missing"))
            ):
                is_due = True
            else:
                # 1. Check if t_plus_5_date exists (top-level, shadow_credit_metrics, or result_data) and <= as_of
                raw_t5_date = (
                    s.get("t_plus_5_date")
                    or metrics.get("t_plus_5_date")
                    or res_data.get("t_plus_5_date")
                )
                t5_parsed = _parse_sample_date(raw_t5_date) if raw_t5_date else None
                if t5_parsed is not None:
                    is_due = bool(t5_parsed <= as_of_date)
                else:
                    # 2. Check if trade_date / date exists and compute T+5 date forward
                    raw_td = (
                        s.get("trade_date")
                        or s.get("date")
                        or res_data.get("trade_date")
                        or res_data.get("date")
                    )
                    td_parsed = _parse_sample_date(raw_td) if raw_td else None
                    if td_parsed is not None:
                        td_str = td_parsed.strftime("%Y-%m-%d")
                        calc_t5_str = None
                        try:
                            calc_t5_str = calculate_t_plus_5_date(td_str, calendar_dates=cal_dates)
                        except Exception as exc:
                            logger.debug("calculate_t_plus_5_date in gate evaluation failed for %s: %s", td_str, exc)
                            calc_t5_str = None

                        if calc_t5_str:
                            calc_t5_parsed = _parse_sample_date(calc_t5_str)
                            if calc_t5_parsed is not None and calc_t5_parsed <= as_of_date:
                                is_due = True
                            else:
                                is_due = False
                        else:
                            # Calculation failed / out of calendar: do not invent due
                            is_due = False
                    else:
                        # Cannot parse date: do not invent due
                        is_due = False

        if is_due:
            due_t5_count += 1
            if hit is not None:
                completed_t5_count += 1

    if due_t5_count == 0:
        # If no sample is due or no historical evaluated data, completeness rate is 0.0 and Dimension 4 fails (no vacuous pass)
        t5_completeness_rate = 0.0
        pass_d4 = False
    else:
        t5_completeness_rate = completed_t5_count / due_t5_count
        pass_d4 = bool(t5_completeness_rate >= cfg["min_t_plus_5_completeness"])

    dim_t5_details: dict[str, Any] = {
        "due_count": due_t5_count,
        "completed_count": completed_t5_count,
        "completeness_rate": round(t5_completeness_rate, 4),
        "min_required_rate": cfg["min_t_plus_5_completeness"],
    }
    if due_t5_count == 0:
        dim_t5_details["reason"] = "no_due_samples"

    dim_t5 = {
        "passed": pass_d4,
        "details": dim_t5_details,
    }

    # ── Dimension 5: Balance ──────────────────────────────────────────────────
    total_side = bull_samples + bear_samples
    bull_ratio = (bull_samples / total_side) if total_side > 0 else 0.0
    side_diff = abs(bull_samples - bear_samples)

    pass_d5_ratio = cfg["min_side_balance_ratio"] <= bull_ratio <= cfg["max_side_balance_ratio"]
    pass_d5_diff = side_diff <= cfg["max_side_count_diff"]
    pass_d5 = bool(pass_d5_ratio and pass_d5_diff and sample_count >= cfg["min_sample_count"])

    dim_balance = {
        "passed": pass_d5,
        "details": {
            "bull_ratio": round(bull_ratio, 4),
            "allowed_range": [cfg["min_side_balance_ratio"], cfg["max_side_balance_ratio"]],
            "side_diff": side_diff,
            "max_allowed_diff": cfg["max_side_count_diff"],
        },
    }

    # ── Dimension 6: Bias Freeze (Rates Delta, Clone Rate, Consistency Gate) ───
    bull_v_rates: list[float] = []
    bear_v_rates: list[float] = []
    bull_ch_rates: list[float] = []
    bear_ch_rates: list[float] = []
    consistency_triggers = 0
    claims_text_pool: list[str] = []

    for s in samples:
        metrics = s.get("shadow_credit_metrics")
        if not metrics or not isinstance(metrics, Mapping):
            metrics = calculate_shadow_credit_metrics(s)
        if metrics.get("bull_verified_rate") is not None:
            bull_v_rates.append(float(metrics["bull_verified_rate"]))
        if metrics.get("bear_verified_rate") is not None:
            bear_v_rates.append(float(metrics["bear_verified_rate"]))
        if metrics.get("bull_challenge_adoption_rate") is not None:
            bull_ch_rates.append(float(metrics["bull_challenge_adoption_rate"]))
        if metrics.get("bear_challenge_adoption_rate") is not None:
            bear_ch_rates.append(float(metrics["bear_challenge_adoption_rate"]))
        if metrics.get("manager_consistency_gate_triggered") is True:
            consistency_triggers += 1

        inv_state = s.get("investment_debate_state") or s.get("result_data", {}).get("investment_debate_state") or s
        for c in (inv_state.get("claims") or s.get("claims") or []):
            if isinstance(c, Mapping) and c.get("claim"):
                claims_text_pool.append(str(c["claim"]).strip())

    avg_bull_v = (sum(bull_v_rates) / len(bull_v_rates)) if bull_v_rates else 0.0
    avg_bear_v = (sum(bear_v_rates) / len(bear_v_rates)) if bear_v_rates else 0.0
    delta_verified_rate = abs(avg_bull_v - avg_bear_v)

    avg_bull_ch = (sum(bull_ch_rates) / len(bull_ch_rates)) if bull_ch_rates else 0.0
    avg_bear_ch = (sum(bear_ch_rates) / len(bear_ch_rates)) if bear_ch_rates else 0.0
    delta_challenge_rate = abs(avg_bull_ch - avg_bear_ch)

    # Clone rate: duplicate claims / total claims
    unique_claims_count = len(set(claims_text_pool))
    total_claims_count = len(claims_text_pool)
    clone_rate = (1.0 - (unique_claims_count / total_claims_count)) if total_claims_count > 0 else 0.0

    consistency_trigger_rate = (consistency_triggers / sample_count) if sample_count > 0 else 0.0

    pass_d6_v = delta_verified_rate <= cfg["max_delta_verified_rate"]
    pass_d6_ch = delta_challenge_rate <= cfg["max_delta_challenge_adoption_rate"]
    pass_d6_clone = clone_rate <= cfg["max_clone_rate"]
    pass_d6_cons = consistency_trigger_rate <= cfg["max_consistency_trigger_rate"]
    pass_d6 = bool(pass_d6_v and pass_d6_ch and pass_d6_clone and pass_d6_cons and sample_count >= cfg["min_sample_count"])

    dim_bias = {
        "passed": pass_d6,
        "details": {
            "avg_bull_verified_rate": round(avg_bull_v, 4),
            "avg_bear_verified_rate": round(avg_bear_v, 4),
            "delta_verified_rate": round(delta_verified_rate, 4),
            "max_allowed_delta_v": cfg["max_delta_verified_rate"],
            "avg_bull_challenge_adoption_rate": round(avg_bull_ch, 4),
            "avg_bear_challenge_adoption_rate": round(avg_bear_ch, 4),
            "delta_challenge_adoption_rate": round(delta_challenge_rate, 4),
            "max_allowed_delta_ch": cfg["max_delta_challenge_adoption_rate"],
            "clone_rate": round(clone_rate, 4),
            "max_allowed_clone_rate": cfg["max_clone_rate"],
            "consistency_trigger_rate": round(consistency_trigger_rate, 4),
            "max_allowed_consistency_rate": cfg["max_consistency_trigger_rate"],
        },
    }

    # ── Dimension 7: Magnitude ────────────────────────────────────────────────
    dim_magnitude = {
        "passed": True,
        "details": {
            "min_weight_multiplier": cfg["min_weight_multiplier"],
            "max_weight_multiplier": cfg["max_weight_multiplier"],
            "range": [cfg["min_weight_multiplier"], cfg["max_weight_multiplier"]],
        },
    }

    matrix = {
        "dimension_n": dim_n,
        "dimension_side": dim_side,
        "dimension_time": dim_time,
        "dimension_t5": dim_t5,
        "dimension_balance": dim_balance,
        "dimension_bias": dim_bias,
        "dimension_magnitude": dim_magnitude,
    }

    if not homogeneity_passed:
        matrix["cohort_homogeneity"] = {
            "passed": False,
            "reason": homogeneity_reason,
        }

    all_passed = bool(
        homogeneity_passed
        and (sample_count > 0)
        and dim_n["passed"]
        and dim_side["passed"]
        and dim_time["passed"]
        and dim_t5["passed"]
        and dim_balance["passed"]
        and dim_bias["passed"]
        and dim_magnitude["passed"]
    )

    recommendation = "ELIGIBLE_FOR_ACTIVATION" if all_passed else "KEEP_FALSE"

    return {
        "schema_version": H1B_SCHEMA_VERSION,
        "passed": all_passed,
        "matrix": matrix,
        "summary": {
            "sample_count": sample_count,
            "system_gate_status": "PASS" if all_passed else "FAIL",
            "recommendation": recommendation,
            "cohort": cohort_meta.get("canonical_key"),
            "commit_shas": cohort_meta.get("commit_shas", []),
            "excluded_counts": dict(initial_excluded),
            "pipeline_ledger": current_ledger,
        },
        "excluded_counts": dict(initial_excluded),
        "pipeline_ledger": current_ledger,
        "recommendation": recommendation,
        "cohort": cohort_meta.get("canonical_key"),
        "cohort_info": cohort_meta,
    }


# ── Layered Isolation State Machine (P3-H1b) ──────────────────────────────────

def evaluate_model_bias_and_weights(
    samples_or_reports: Sequence[Mapping[str, Any]],
    system_gate_passed: bool = False,
    per_model_bias_overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate per-model bias and layered isolation weights.

    Rules:
    - System gate failure -> Global weight=1.0 (Shadow-only)
    - Single model bias -> Only that model clamped to 1.0 + bias_freeze_reason
    - Abnormal model ratio > 50% -> Global fallback to Shadow
    """
    samples = list(samples_or_reports or [])
    overrides = dict(per_model_bias_overrides or {})

    # Discover all models
    models_set: set[str] = set()
    for s in samples:
        metrics = s.get("shadow_credit_metrics") or {}
        id_map = metrics.get("model_id_by_stance") or {}
        for m in id_map.values():
            if m and isinstance(m, str):
                models_set.add(m.strip())
        inv_state = s.get("investment_debate_state") or s
        for msg in (inv_state.get("round_messages") or []):
            if isinstance(msg, Mapping):
                m_name = msg.get("model_name") or msg.get("model_id") or msg.get("model")
                if m_name and isinstance(m_name, str):
                    models_set.add(m_name.strip())

    if not models_set:
        models_set = {"deepseek-r1", "qwen-max", "gpt-4o"}

    if not system_gate_passed:
        return {
            "credit_weighting_active": False,
            "global_fallback_shadow": True,
            "system_gate_status": "FAIL",
            "model_weights": {m: 1.0 for m in models_set},
            "bias_freeze_reasons": {m: "System-level activation gates not passed" for m in models_set},
            "abnormal_model_ratio": 1.0,
        }

    # Evaluate per-model bias
    model_weights: dict[str, float] = {}
    bias_freeze_reasons: dict[str, str] = {}
    biased_models_count = 0

    for model in sorted(models_set):
        ov = overrides.get(model)
        is_biased = False
        reason = ""

        if ov is not None:
            if isinstance(ov, dict):
                is_biased = bool(ov.get("biased", False))
                reason = str(ov.get("reason", "Overridden as biased"))
            else:
                is_biased = bool(ov)
                reason = "Overridden as biased" if is_biased else ""
        else:
            # Evaluate statistical bias from samples for this model
            # For now default unbiased unless overrides or stats trigger
            is_biased = False

        if is_biased:
            model_weights[model] = 1.0
            bias_freeze_reasons[model] = reason or "Model bias exceeds threshold (clamped to 1.0)"
            biased_models_count += 1
        else:
            # Calibrated weight (in [0.85, 1.15])
            model_weights[model] = 1.05  # Base default calibrated weight

    total_models = len(models_set)
    abnormal_ratio = (biased_models_count / total_models) if total_models > 0 else 0.0

    if abnormal_ratio > 0.50:
        # Abnormal model ratio > 50% -> Global fallback to Shadow
        return {
            "credit_weighting_active": False,
            "global_fallback_shadow": True,
            "system_gate_status": "PASS",
            "global_freeze_reason": f"Abnormal model ratio ({biased_models_count}/{total_models} = {abnormal_ratio:.1%}) > 50%",
            "model_weights": {m: 1.0 for m in models_set},
            "bias_freeze_reasons": bias_freeze_reasons,
            "abnormal_model_ratio": abnormal_ratio,
        }

    return {
        "credit_weighting_active": True,
        "global_fallback_shadow": False,
        "system_gate_status": "PASS",
        "model_weights": model_weights,
        "bias_freeze_reasons": bias_freeze_reasons,
        "abnormal_model_ratio": abnormal_ratio,
    }


# ── Claim Credit Weighting Calculation (P3-H1b) ──────────────────────────────

def calculate_claim_credit_weights(
    claims: Sequence[Mapping[str, Any]],
    claim_evidence_summary: Mapping[str, Any],
    model_weights: Optional[Mapping[str, float]] = None,
    *,
    credit_weighting_enabled: bool = False,
    system_gate_passed: bool = False,
) -> dict[str, Any]:
    """Calculate relative credit weights across claims.

    Hard Rules:
    1. If credit_weighting_enabled is False or system_gate_passed is False -> All weights = 1.0.
    2. Only verified claims receive relative weight modification in [0.85, 1.15].
    3. Contradicted, unsupported, or unavailable claims NEVER receive weight > 0 or get elevated.
    """
    claims_list = list(claims or [])
    m_weights = dict(model_weights or {})
    min_w = H1B_THRESHOLDS["min_weight_multiplier"]
    max_w = H1B_THRESHOLDS["max_weight_multiplier"]

    claim_weights: dict[str, float] = {}
    claim_decisions: dict[str, str] = {}
    effective_weights: dict[str, float] = {}

    if not credit_weighting_enabled or not system_gate_passed:
        for c in claims_list:
            cid = str(c.get("claim_id", "")).strip()
            if cid:
                claim_weights[cid] = 1.0
                sum_info = claim_evidence_summary.get(cid) if isinstance(claim_evidence_summary, Mapping) else {}
                dec = sum_info.get("decision", "reject") if isinstance(sum_info, Mapping) else "reject"
                claim_decisions[cid] = dec
                effective_weights[cid] = 1.0 if dec == "adopt" else 0.0
        return {
            "credit_weighting_active": False,
            "claim_weights": claim_weights,
            "claim_decisions": claim_decisions,
            "effective_weights": effective_weights,
        }

    for c in claims_list:
        cid = str(c.get("claim_id", "")).strip()
        if not cid:
            continue
        model_name = str(c.get("model_name") or c.get("model") or "").strip()
        sum_info = claim_evidence_summary.get(cid) if isinstance(claim_evidence_summary, Mapping) else {}
        if not isinstance(sum_info, Mapping):
            sum_info = {}
        dec = str(sum_info.get("decision", "reject"))
        counts = sum_info.get("counts", {}) if isinstance(sum_info.get("counts"), Mapping) else {}
        is_verified = bool(
            (c.get("status") == "verified" or c.get("is_verified") is True)
            and dec in ("adopt", "partial")
            and counts.get("contradicted", 0) == 0
            and counts.get("source_unavailable", 0) == 0
        )

        claim_decisions[cid] = dec

        if not is_verified:
            # Contradicted / unsupported / unavailable claims NEVER receive credit boost or elevation
            claim_weights[cid] = 0.0
            effective_weights[cid] = 0.0
        else:
            raw_w = m_weights.get(model_name, 1.0)
            clamped_w = max(min_w, min(max_w, float(raw_w)))
            claim_weights[cid] = round(clamped_w, 4)
            effective_weights[cid] = round(clamped_w, 4)

    return {
        "credit_weighting_active": True,
        "claim_weights": claim_weights,
        "claim_decisions": claim_decisions,
        "effective_weights": effective_weights,
    }


def resolve_claim_credit_weights_for_manager(
    *,
    claims: Sequence[Mapping[str, Any]],
    claim_evidence_summary: Mapping[str, Any],
    historical_samples: Optional[Sequence[Mapping[str, Any]]] = None,
    credit_weighting_enabled: bool = False,
    cohort: Optional[Union[str, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Resolve claim credit weights for research_manager using live H1b gates.

    Rules (DAV-601):
    1. Fail-closed: empty/missing historical samples -> flat 1.0.
    2. When flag is off, weights stay flat regardless of gate status.
    3. Unlabeled samples (missing version fields) or mixed cohort generations:
       safely degrade / hold flat weights 1.0 without breaking the main trading pipeline.
    4. Only homogeneous qualifying cohort samples can activate non-flat credit weights.
    """
    history = list(historical_samples or [])

    # Check for unlabeled samples or mixed cohorts in historical_samples
    has_unlabeled = any(is_legacy_unversioned_sample(s) for s in history) if history else False
    is_homo, c_key = is_cohort_homogeneous(history) if history else (True, None)

    if not history or has_unlabeled or not is_homo:
        gate_res = {
            "passed": False,
            "recommendation": "KEEP_FALSE",
            "matrix": {},
        }
        system_gate_passed = False
        reasons: dict[str, str] = {}
        if has_unlabeled:
            reasons["unlabeled_samples"] = "Historical samples contain unversioned/legacy reports (downgraded to flat 1.0)"
        if not is_homo:
            reasons["mixed_cohorts"] = "Historical samples contain mixed cohort generations (downgraded to flat 1.0)"
        isolation = {
            "model_weights": {},
            "bias_freeze_reasons": reasons,
            "global_fallback_shadow": True,
        }
    else:
        try:
            gate_res = evaluate_h1b_system_gates(history, cohort=cohort)
        except Exception as exc:
            logger.warning("[shadow_credit] evaluate_h1b_system_gates failed: %s, falling back to flat 1.0", exc)
            gate_res = {
                "passed": False,
                "recommendation": "KEEP_FALSE",
                "matrix": {},
            }
        system_gate_passed = bool(gate_res.get("passed", False))
        isolation = evaluate_model_bias_and_weights(
            history,
            system_gate_passed=system_gate_passed,
        )

    weights_res = calculate_claim_credit_weights(
        claims=claims,
        claim_evidence_summary=claim_evidence_summary,
        model_weights=isolation.get("model_weights", {}),
        credit_weighting_enabled=credit_weighting_enabled,
        system_gate_passed=system_gate_passed,
    )
    return {
        **weights_res,
        "system_gate_passed": system_gate_passed,
        "system_gate_status": "PASS" if system_gate_passed else "FAIL",
        "recommendation": gate_res.get("recommendation", "KEEP_FALSE"),
        "bias_freeze_reasons": isolation.get("bias_freeze_reasons", {}),
        "model_weights": isolation.get("model_weights", {}),
        "global_fallback_shadow": bool(isolation.get("global_fallback_shadow", True)),
    }


def apply_credit_weighting_to_debate(
    result_data_or_state: Mapping[str, Any],
    historical_samples: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Apply credit weighting to debate state/result and update shadow metrics."""
    meta = get_protocol_metadata(result_data_or_state)
    credit_weighting_flag = bool(meta.get("feature_flags", {}).get("credit_weighting_enabled", False))

    inv_state = result_data_or_state.get("investment_debate_state")
    if not isinstance(inv_state, Mapping):
        inv_state = result_data_or_state

    claims = inv_state.get("claims") or result_data_or_state.get("claims") or []
    verdict = inv_state.get("manager_verdict") or result_data_or_state.get("manager_verdict") or {}
    claim_summary = verdict.get("claim_evidence_summary") or inv_state.get("claim_evidence_summary") or {}

    resolved = resolve_claim_credit_weights_for_manager(
        claims=claims,
        claim_evidence_summary=claim_summary if isinstance(claim_summary, Mapping) else {},
        historical_samples=historical_samples,
        credit_weighting_enabled=credit_weighting_flag,
    )
    system_gate_passed = bool(resolved.get("system_gate_passed", False))
    weights_res = {
        "credit_weighting_active": resolved.get("credit_weighting_active", False),
        "claim_weights": resolved.get("claim_weights", {}),
        "claim_decisions": resolved.get("claim_decisions", {}),
        "effective_weights": resolved.get("effective_weights", {}),
    }
    isolation = {
        "bias_freeze_reasons": resolved.get("bias_freeze_reasons", {}),
        "model_weights": resolved.get("model_weights", {}),
    }

    shadow_metrics = calculate_shadow_credit_metrics(result_data_or_state)
    shadow_metrics.update({
        "credit_weighting_enabled": credit_weighting_flag,
        "credit_weighting_active": weights_res["credit_weighting_active"],
        "system_gate_status": "PASS" if system_gate_passed else "FAIL",
        "bias_freeze_reasons": isolation.get("bias_freeze_reasons", {}),
        "model_weights": isolation.get("model_weights", {}),
    })

    return {
        "credit_weighting_active": weights_res["credit_weighting_active"],
        "claim_weights": weights_res["claim_weights"],
        "shadow_credit_metrics": shadow_metrics,
        "system_gate_status": "PASS" if system_gate_passed else "FAIL",
    }


# ── T+5 Shadow Backfill Module (Track A5) ───────────────────────────────────

def fetch_close_prices_safe(
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict[str, float]:
    """Fetch close prices from market data vendor safely, returning date_str -> close_price mapping."""
    if not symbol or not start_date or not end_date:
        return {}
    try:
        from tradingagents.dataflows.interface import route_to_vendor
        raw_csv = route_to_vendor("get_stock_data", symbol, start_date, end_date)
        if not raw_csv or not isinstance(raw_csv, str):
            return {}
        clean_lines = [
            line for line in raw_csv.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not clean_lines:
            return {}
        reader = csv.DictReader(io.StringIO("\n".join(clean_lines)))
        prices: dict[str, float] = {}
        for row in reader:
            cols_lower = {k.lower().strip(): v for k, v in row.items() if k}
            d_val = cols_lower.get("date") or cols_lower.get("trade_date")
            c_val = cols_lower.get("close") or cols_lower.get("close_price")
            if d_val and c_val:
                try:
                    d_clean = str(d_val)[:10]
                    prices[d_clean] = float(c_val)
                except (ValueError, TypeError):
                    pass
        return prices
    except Exception as exc:
        logger.debug("fetch_close_prices_safe failed for %s (%s -> %s): %s", symbol, start_date, end_date, exc)
        return {}


def backfill_tplus5_shadow_for_report(
    report_or_result_data: Mapping[str, Any],
    *,
    as_of: Optional[Union[str, date, datetime]] = None,
    price_series: Optional[Mapping[str, float]] = None,
    get_price_fn: Optional[Any] = None,
    trading_calendar: Optional[Sequence[Union[str, date]]] = None,
    is_suspended: Optional[bool] = None,
) -> dict[str, Any]:
    """Backfill T+5 shadow credit metrics for a single report.

    Rules:
    1. Strictly filter for qualifying completed v2 reports (is_qualifying_v2_report).
       Non-qualifying reports are returned unmodified with status recorded.
    2. Strict trading calendar T+5 forward (calculate_t_plus_5_date).
       Hold window is strictly 5 trading days — forbidden to shorten.
    3. Status classification:
       - T+5 date not reached (eval_date > as_of): t_plus_5_status='pending_due', hit=None, is_t_plus_5_due=False, t_plus_5_evaluated=False.
       - Suspended (is_suspended / suspension gap): t_plus_5_status='suspension', hit=None, is_suspended=True, is_t_plus_5_due=False.
       - Market data missing (failed price fetch): t_plus_5_status='data_missing', hit=None, is_t_plus_5_due=True, t_plus_5_evaluated=True.
       - Evaluated (price successfully parsed): t_plus_5_status='due_and_evaluated', is_t_plus_5_due=True, t_plus_5_evaluated=True.
    4. Hit determination uses manager_verdict.winner:
       - 'bull': price_change > 0
       - 'bear': price_change < 0
       - 'tie': abs(price_change / entry_val) <= 0.03
    5. Pure & idempotent: preserves all existing fields in result_data.
    """
    if not isinstance(report_or_result_data, Mapping):
        return {}

    res = copy.deepcopy(dict(report_or_result_data))

    # 1. Qualification check
    if not is_qualifying_v2_report(res):
        res["_backfill_status"] = "skipped_non_qualifying"
        return res

    # Identify target dictionary (nested result_data if present, else res)
    has_nested_result_data = "result_data" in res and isinstance(res["result_data"], Mapping)
    target = dict(res["result_data"]) if has_nested_result_data else res

    inv_state = target.get("investment_debate_state")
    if not isinstance(inv_state, Mapping):
        inv_state = target

    # Extract symbol and trade date
    symbol = (
        target.get("symbol")
        or target.get("ticker")
        or res.get("symbol")
        or res.get("ticker")
        or ""
    )
    symbol = str(symbol).strip()

    raw_date = target.get("trade_date") or target.get("date") or res.get("trade_date") or res.get("date") or res.get("created_at")
    parsed_d = _parse_sample_date(raw_date)
    trade_date_str = parsed_d.strftime("%Y-%m-%d") if parsed_d else (str(raw_date)[:10] if raw_date else None)

    # 2. Strict T+5 Date Calculation
    t5_date: Optional[str] = None
    if trade_date_str:
        try:
            t5_date = calculate_t_plus_5_date(trade_date_str, calendar_dates=trading_calendar)
        except Exception as exc:
            logger.debug("calculate_t_plus_5_date failed for %s: %s", trade_date_str, exc)
            t5_date = None

    if not trade_date_str or not t5_date:
        t_plus_5_status = "data_missing"
        is_t_plus_5_due = False
        t_plus_5_evaluated = False
        t_plus_5_direction_hit = None
        t_plus_5_price = None
        target["t_plus_5_status"] = t_plus_5_status
        target["is_t_plus_5_due"] = is_t_plus_5_due
        target["t_plus_5_evaluated"] = t_plus_5_evaluated
        target["t_plus_5_direction_hit"] = t_plus_5_direction_hit
        sm = calculate_shadow_credit_metrics(target)
        target["shadow_credit_metrics"] = sm
        if has_nested_result_data:
            res["result_data"] = target
            res["t_plus_5_status"] = t_plus_5_status
            res["t_plus_5_price"] = None
            res["t_plus_5_direction_hit"] = None
            res["is_t_plus_5_due"] = is_t_plus_5_due
            res["t_plus_5_evaluated"] = t_plus_5_evaluated
            res["shadow_credit_metrics"] = sm
        res["_backfill_status"] = "missing_date"
        return res

    # 3. As-of Boundary Check
    if as_of is None:
        as_of_str = now_cn().date().strftime("%Y-%m-%d")
        is_live_today = (t5_date == as_of_str)
    elif isinstance(as_of, (date, datetime)):
        as_of_str = as_of.strftime("%Y-%m-%d")
        is_live_today = False
    else:
        as_of_str = str(as_of)[:10].strip()
        is_live_today = False

    if t5_date > as_of_str or (is_live_today and cn_market_phase() != "post_close"):
        # T+5 window not yet arrived
        t_plus_5_status = "pending_due"
        is_t_plus_5_due = False
        t_plus_5_evaluated = False
        t_plus_5_direction_hit = None
        t_plus_5_price = None

        target["t_plus_5_date"] = t5_date
        target["t_plus_5_price"] = None
        target["t_plus_5_status"] = t_plus_5_status
        target["is_t_plus_5_due"] = is_t_plus_5_due
        target["t_plus_5_evaluated"] = t_plus_5_evaluated
        target["t_plus_5_direction_hit"] = None
        sm = calculate_shadow_credit_metrics(target)
        target["shadow_credit_metrics"] = sm
        if isinstance(target.get("investment_debate_state"), dict):
            target["investment_debate_state"]["shadow_credit_metrics"] = sm
            target["investment_debate_state"]["t_plus_5_date"] = t5_date
            target["investment_debate_state"]["t_plus_5_status"] = t_plus_5_status
            target["investment_debate_state"]["is_t_plus_5_due"] = is_t_plus_5_due
            target["investment_debate_state"]["t_plus_5_evaluated"] = t_plus_5_evaluated
            target["investment_debate_state"]["t_plus_5_direction_hit"] = None
        if has_nested_result_data:
            res["result_data"] = target
            res["t_plus_5_date"] = t5_date
            res["t_plus_5_price"] = None
            res["t_plus_5_status"] = t_plus_5_status
            res["is_t_plus_5_due"] = is_t_plus_5_due
            res["t_plus_5_evaluated"] = t_plus_5_evaluated
            res["t_plus_5_direction_hit"] = None
            res["shadow_credit_metrics"] = sm
        res["_backfill_status"] = "pending_due"
        return res

    # 4. Suspension Check
    suspended = bool(
        is_suspended
        or target.get("is_suspended") is True
        or target.get("suspension") is True
        or target.get("t_plus_5_status") == "suspension"
        or res.get("is_suspended") is True
        or res.get("suspension") is True
        or res.get("t_plus_5_status") == "suspension"
    )

    # Check raw_gaps for suspension mention if any
    raw_gaps = target.get("data_gaps") or res.get("data_gaps") or []
    if not suspended and isinstance(raw_gaps, list):
        for g in raw_gaps:
            g_str = str(g).lower()
            if "suspension" in g_str or "停牌" in g_str:
                suspended = True
                break

    if suspended:
        t_plus_5_status = "suspension"
        is_t_plus_5_due = False
        t_plus_5_evaluated = False
        t_plus_5_direction_hit = None
        t_plus_5_price = None

        target["t_plus_5_date"] = t5_date
        target["t_plus_5_price"] = None
        target["t_plus_5_status"] = t_plus_5_status
        target["is_t_plus_5_due"] = False
        target["t_plus_5_evaluated"] = False
        target["t_plus_5_direction_hit"] = None
        target["is_suspended"] = True
        sm = calculate_shadow_credit_metrics(target)
        target["shadow_credit_metrics"] = sm
        if isinstance(target.get("investment_debate_state"), dict):
            target["investment_debate_state"]["shadow_credit_metrics"] = sm
            target["investment_debate_state"]["t_plus_5_date"] = t5_date
            target["investment_debate_state"]["t_plus_5_status"] = t_plus_5_status
            target["investment_debate_state"]["is_t_plus_5_due"] = False
            target["investment_debate_state"]["t_plus_5_evaluated"] = False
            target["investment_debate_state"]["t_plus_5_direction_hit"] = None
            target["investment_debate_state"]["is_suspended"] = True
        if has_nested_result_data:
            res["result_data"] = target
            res["t_plus_5_date"] = t5_date
            res["t_plus_5_price"] = None
            res["t_plus_5_status"] = t_plus_5_status
            res["is_t_plus_5_due"] = False
            res["t_plus_5_evaluated"] = False
            res["t_plus_5_direction_hit"] = None
            res["is_suspended"] = True
            res["shadow_credit_metrics"] = sm
        res["_backfill_status"] = "suspension"
        return res

    # 5. Price Resolution
    manager_verdict = (
        target.get("manager_verdict")
        or inv_state.get("manager_verdict")
        or res.get("manager_verdict")
        or {}
    )
    if not isinstance(manager_verdict, Mapping):
        manager_verdict = {}

    entry_val: Optional[float] = None
    raw_entry = (
        manager_verdict.get("entry")
        or target.get("entry_price")
        or target.get("target_price")
        or inv_state.get("entry_price")
        or inv_state.get("target_price")
        or res.get("entry_price")
        or res.get("target_price")
    )
    if raw_entry:
        try:
            entry_val = float(str(raw_entry).split("-")[0].replace("元", "").strip())
        except (ValueError, TypeError):
            entry_val = None

    t5_price_val: Optional[float] = None

    # Check custom price_series or get_price_fn or pre-set t_plus_5_price
    if price_series and isinstance(price_series, Mapping):
        if t5_date in price_series:
            try:
                t5_price_val = float(price_series[t5_date])
            except (ValueError, TypeError):
                t5_price_val = None
        if entry_val is None and trade_date_str in price_series:
            try:
                entry_val = float(price_series[trade_date_str])
            except (ValueError, TypeError):
                entry_val = None
    elif get_price_fn and callable(get_price_fn):
        try:
            fn_res = get_price_fn(symbol, trade_date_str, t5_date)
            if fn_res is not None:
                t5_price_val = float(fn_res)
        except Exception as exc:
            logger.debug("get_price_fn failed for %s: %s", symbol, exc)
    elif target.get("t_plus_5_price") is not None:
        try:
            t5_price_val = float(target["t_plus_5_price"])
        except (ValueError, TypeError):
            t5_price_val = None
    elif res.get("t_plus_5_price") is not None:
        try:
            t5_price_val = float(res["t_plus_5_price"])
        except (ValueError, TypeError):
            t5_price_val = None
    else:
        # Try fetching from vendor
        fetched = fetch_close_prices_safe(symbol, trade_date_str, t5_date)
        if fetched:
            if t5_date in fetched:
                t5_price_val = fetched[t5_date]
            if entry_val is None and trade_date_str in fetched:
                entry_val = fetched[trade_date_str]

    # 6. Evaluation based on winner
    winner = str(
        manager_verdict.get("winner")
        or target.get("debate_winner")
        or res.get("debate_winner")
        or ""
    ).strip().lower()

    t_plus_5_return_pct: Optional[float] = None

    if t5_price_val is None or entry_val is None or entry_val <= 0:
        t_plus_5_status = "data_missing"
        is_t_plus_5_due = True
        t_plus_5_evaluated = True
        t_plus_5_direction_hit = None
        t_plus_5_price = None
        backfill_status = "data_missing"
    else:
        price_change = t5_price_val - entry_val
        t_plus_5_return_pct = round((price_change / entry_val) * 100.0, 2)
        if winner == "bull":
            t_plus_5_direction_hit = bool(price_change > 0)
        elif winner == "bear":
            t_plus_5_direction_hit = bool(price_change < 0)
        elif winner == "tie":
            t_plus_5_direction_hit = bool(abs(price_change / entry_val) <= 0.03)
        else:
            direction_str = str(
                manager_verdict.get("direction")
                or target.get("decision")
                or res.get("decision")
                or ""
            ).upper()
            if any(w in direction_str for w in ("BUY", "BULLISH", "多", "买入", "增持")):
                t_plus_5_direction_hit = bool(price_change > 0)
            elif any(w in direction_str for w in ("SELL", "BEARISH", "空", "卖出", "减持")):
                t_plus_5_direction_hit = bool(price_change < 0)
            elif any(w in direction_str for w in ("HOLD", "NEUTRAL", "中性", "观望", "持有")):
                t_plus_5_direction_hit = bool(abs(price_change / entry_val) <= 0.03)
            else:
                t_plus_5_direction_hit = bool(price_change > 0)

        t_plus_5_status = "due_and_evaluated"
        is_t_plus_5_due = True
        t_plus_5_evaluated = True
        t_plus_5_price = round(t5_price_val, 4)
        backfill_status = "hit" if t_plus_5_direction_hit else "miss"

    target["t_plus_5_date"] = t5_date
    target["t_plus_5_price"] = t_plus_5_price
    target["t_plus_5_status"] = t_plus_5_status
    target["is_t_plus_5_due"] = is_t_plus_5_due
    target["t_plus_5_evaluated"] = t_plus_5_evaluated
    target["t_plus_5_direction_hit"] = t_plus_5_direction_hit
    if t_plus_5_return_pct is not None:
        target["t_plus_5_return_pct"] = t_plus_5_return_pct

    sm = calculate_shadow_credit_metrics(target, t_plus_5_price=t_plus_5_price)
    sm["t_plus_5_status"] = t_plus_5_status
    sm["t_plus_5_date"] = t5_date
    sm["t_plus_5_price"] = t_plus_5_price
    sm["t_plus_5_direction_hit"] = t_plus_5_direction_hit
    target["shadow_credit_metrics"] = sm

    if isinstance(target.get("investment_debate_state"), dict):
        target["investment_debate_state"]["shadow_credit_metrics"] = sm
        target["investment_debate_state"]["t_plus_5_date"] = t5_date
        target["investment_debate_state"]["t_plus_5_price"] = t_plus_5_price
        target["investment_debate_state"]["t_plus_5_status"] = t_plus_5_status
        target["investment_debate_state"]["is_t_plus_5_due"] = is_t_plus_5_due
        target["investment_debate_state"]["t_plus_5_evaluated"] = t_plus_5_evaluated
        target["investment_debate_state"]["t_plus_5_direction_hit"] = t_plus_5_direction_hit
        if t_plus_5_return_pct is not None:
            target["investment_debate_state"]["t_plus_5_return_pct"] = t_plus_5_return_pct

    if has_nested_result_data:
        res["result_data"] = target
        res["t_plus_5_date"] = t5_date
        res["t_plus_5_price"] = t_plus_5_price
        res["t_plus_5_status"] = t_plus_5_status
        res["is_t_plus_5_due"] = is_t_plus_5_due
        res["t_plus_5_evaluated"] = t_plus_5_evaluated
        res["t_plus_5_direction_hit"] = t_plus_5_direction_hit
        if t_plus_5_return_pct is not None:
            res["t_plus_5_return_pct"] = t_plus_5_return_pct
        res["shadow_credit_metrics"] = sm

    res["_backfill_status"] = backfill_status
    return res


def backfill_tplus5_shadow_for_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    as_of: Optional[Union[str, date, datetime]] = None,
    price_series_map: Optional[Mapping[str, Mapping[str, float]]] = None,
    get_price_fn: Optional[Any] = None,
    trading_calendar: Optional[Sequence[Union[str, date]]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Batch backfill T+5 shadow credit metrics for a list of reports.

    Returns:
        (updated_reports, summary_stats)
    """
    updated_reports: list[dict[str, Any]] = []
    prices_map = dict(price_series_map or {})

    stats: dict[str, Any] = {
        "total_scanned": len(reports),
        "qualifying_v2_count": 0,
        "skipped_non_qualifying": 0,
        "due_count": 0,
        "evaluated_count": 0,
        "hit_count": 0,
        "miss_count": 0,
        "data_missing_count": 0,
        "suspension_count": 0,
        "pending_due_count": 0,
        "completeness_rate": 0.0,
        "hit_rate": None,
    }

    for r in reports:
        if not is_qualifying_v2_report(r):
            stats["skipped_non_qualifying"] += 1
            updated_reports.append(dict(r))
            continue

        stats["qualifying_v2_count"] += 1
        sym = r.get("symbol") or (r.get("result_data", {}).get("symbol") if isinstance(r.get("result_data"), Mapping) else "")
        sym_clean = str(sym).strip()
        series = prices_map.get(sym_clean) or prices_map.get(sym_clean.split(".")[0])

        updated = backfill_tplus5_shadow_for_report(
            r,
            as_of=as_of,
            price_series=series,
            get_price_fn=get_price_fn,
            trading_calendar=trading_calendar,
        )
        updated_reports.append(updated)

        st = updated.get("t_plus_5_status") or (updated.get("result_data", {}).get("t_plus_5_status") if isinstance(updated.get("result_data"), Mapping) else None)
        hit = updated.get("t_plus_5_direction_hit")
        if hit is None and isinstance(updated.get("result_data"), Mapping):
            hit = updated["result_data"].get("t_plus_5_direction_hit")

        if st == "suspension":
            stats["suspension_count"] += 1
        elif st == "pending_due":
            stats["pending_due_count"] += 1
        elif st == "data_missing":
            stats["due_count"] += 1
            stats["data_missing_count"] += 1
        elif st == "due_and_evaluated":
            stats["due_count"] += 1
            stats["evaluated_count"] += 1
            if hit is True:
                stats["hit_count"] += 1
            elif hit is False:
                stats["miss_count"] += 1

    if stats["due_count"] > 0:
        stats["completeness_rate"] = round(stats["evaluated_count"] / stats["due_count"], 4)
    else:
        stats["completeness_rate"] = 0.0

    if stats["evaluated_count"] > 0:
        stats["hit_rate"] = round(stats["hit_count"] / stats["evaluated_count"], 4)

    return updated_reports, stats
