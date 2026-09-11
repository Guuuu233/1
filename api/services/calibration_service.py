"""
Calibration service — measures how honest the system's probability forecasts are.

Given the historical reports table, it buckets completed reports by their
structured ``probability`` (0–1) into fixed ranges and compares each bucket's
predicted rise probability with the actual rise rate observed over a hold
window.  It also reports a Brier score, the standard proper-scoring rule for
binary probability forecasts.

Design: mirrors ``api/services/backtest_service.py`` — a pure, non-invasive
service.  It reads only the reports table plus snapshot JSON already stored on
each report (``custom_prompt_snapshot`` / ``model_config_snapshot``), so every
statistic is attributable to the prompt version and model that produced it.

Resource model: calibration resolves a price window per report (I/O-heavy), so
the service bounds the evaluated set, refuses to run unboundedly many concurrent
computations, and caches identical requests by filter key.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from api.database import ReportDB
from api.services.backtest_service import (
    _get_price_on,
    PRICE_BASIS_VENDOR_QFQ,
    PRICE_BASIS_UNSPECIFIED,
)
from tradingagents.dataflows.return_labels import (
    HORIZON_PROFILE_ID_V1,
    OutcomeStatus,
    PRIMARY_EVAL_OFFSET_MEDIUM,
    PRIMARY_EVAL_OFFSET_SHORT,
)
from tradingagents.graph.horizon_profile import (
    HORIZON_MEDIUM,
    HORIZON_PROFILE_V1,
    HORIZON_SHORT,
    SUPPORTED_HORIZONS,
    T_PLUS_10,
    T_PLUS_40,
)

logger = logging.getLogger(__name__)

# Price basis semantics (DAV-606)
PRICE_BASIS_VENDOR_QFQ: str = PRICE_BASIS_VENDOR_QFQ
PRICE_BASIS_UNSPECIFIED: str = PRICE_BASIS_UNSPECIFIED

# Multi-horizon profile routing constants (V-02)
SUPPORTED_HORIZON_PROFILES: frozenset[str] = frozenset({
    HORIZON_PROFILE_ID_V1,
    "horizon_profile_v1",
})

# Bounded evaluation: fetching hold-window prices is I/O-heavy, so cap how many
# reports a single calibration run resolves.  Mirrors backtest_service's
# env-var-bounded worker pool design.
MAX_CALIBRATION_REPORTS = max(1, int(os.getenv("CALIBRATION_MAX_REPORTS", "200")))
DEFAULT_CALIBRATION_LIMIT = max(1, int(os.getenv("CALIBRATION_DEFAULT_LIMIT", "50")))
MAX_CALIBRATION_LIMIT = max(DEFAULT_CALIBRATION_LIMIT, MAX_CALIBRATION_REPORTS)
# When prompt/model snapshot filters are active, the pre-filter candidate scan
# is bounded by this cap; if the scan reaches the cap the response reports
# ``truncated_before_filter`` so callers know the result is a biased sample.
MAX_CALIBRATION_FILTER_SCAN = max(1, int(os.getenv("CALIBRATION_FILTER_SCAN", "5000")))
# Per-filter-key result cache: identical requests within the TTL skip re-fetching
# prices entirely.
CALIBRATION_CACHE_TTL_SECONDS = max(0, int(os.getenv("CALIBRATION_CACHE_TTL", "300")))
# Hard cap on concurrent calibration computations (each can hold a worker thread
# for minutes while fetching prices).
CALIBRATION_MAX_CONCURRENT = max(1, int(os.getenv("CALIBRATION_MAX_CONCURRENT", "2")))
MAX_CALIBRATION_CACHE_ENTRIES = 50
# Calendar factor used to pre-reject obviously-incomplete hold windows before
# spending I/O on price fetches; the authoritative check is the row-count guard
# inside ``_get_price_after_strict``.
HOLD_CALENDAR_FACTOR = 1.6

# Fixed reliability-curve buckets.  ``probability`` is stored as a 0–1 fraction
# on the reports table, so each bucket is expressed in both percent label and
# raw probability bounds.
_BUCKETS: List[Tuple[str, float, float]] = [
    ("0-50%", 0.0, 0.5),
    ("50-60%", 0.5, 0.6),
    ("60-70%", 0.6, 0.7),
    ("70-80%", 0.7, 0.8),
    ("80+%", 0.8, 1.0),
]

DEFAULT_HOLD_DAYS = 5

# Minimum evaluated sample size required before calibration metrics
# (Brier score and per-bucket rise rates) are presented to callers.
#
# Rationale (AGENTS.md §5 - no magic values, explicit basis):
# 1. Statistical Validity (Central Limit Theorem & Binomial Estimation):
#    Evaluating calibration divides reports into 5 probability buckets
#    (0-50%, 50-60%, 60-70%, 70-80%, 80+%). With N=30, each bucket expects
#    ~6 samples on average. In binomial proportion estimation, N >= 30 is the
#    classic rule-of-thumb lower bound required for the sampling distribution of
#    proportions to approximate normality and yield meaningful confidence intervals.
# 2. Fail-Closed Discipline (L1 fail-closed in L3 calibration, DAV-755 / DAV-758):
#    With N < 30 (e.g. N=1, 2, or 3), a single binary outcome shifts an entire
#    bucket's observed rate between 0% and 100%, creating an illusion of precision
#    and self-deception ("已校准" on random noise).
# 3. Decision Consensus:
#    Confirmed in DAV-755 diagnosis (H3) and scheduled as a hard gate in DAV-758.
DEFAULT_CALIBRATION_MIN_SAMPLE_SIZE_CONSTANT = 30


def _parse_min_sample_size_env(env_val: Optional[str] = None) -> int:
    """Safely parse CALIBRATION_MIN_SAMPLE_SIZE from environment.

    Accepts only reasonable positive integers in [1, 10000].
    Malformed or out-of-range values (empty string, non-digit, float,
    negative, zero, excessive magnitude) fail-closed to the named safe default 30
    and log a warning, completely preventing unhandled import exceptions.
    """
    if env_val is None:
        env_val = os.getenv("CALIBRATION_MIN_SAMPLE_SIZE")
    if env_val is None:
        return DEFAULT_CALIBRATION_MIN_SAMPLE_SIZE_CONSTANT
    raw = env_val.strip()
    if not raw:
        return DEFAULT_CALIBRATION_MIN_SAMPLE_SIZE_CONSTANT
    try:
        val = int(raw)
        if 1 <= val <= 10_000:
            return val
        logger.warning(
            "CALIBRATION_MIN_SAMPLE_SIZE=%r out of valid range [1, 10000]; falling back to safe default %d",
            env_val,
            DEFAULT_CALIBRATION_MIN_SAMPLE_SIZE_CONSTANT,
        )
        return DEFAULT_CALIBRATION_MIN_SAMPLE_SIZE_CONSTANT
    except (ValueError, TypeError):
        logger.warning(
            "Invalid CALIBRATION_MIN_SAMPLE_SIZE=%r (expected integer); falling back to safe default %d",
            env_val,
            DEFAULT_CALIBRATION_MIN_SAMPLE_SIZE_CONSTANT,
        )
        return DEFAULT_CALIBRATION_MIN_SAMPLE_SIZE_CONSTANT


DEFAULT_MIN_CALIBRATION_SAMPLE_SIZE = _parse_min_sample_size_env()
MIN_CALIBRATION_SAMPLE_SIZE = DEFAULT_MIN_CALIBRATION_SAMPLE_SIZE


class CalibrationBusyError(RuntimeError):
    """Raised when the calibration concurrency cap is already reached."""


# ──────────────────────────────────────────────────────────────────────────────
# Resource guard + per-key cache
# ──────────────────────────────────────────────────────────────────────────────

_calibration_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_cache_lock = threading.Lock()
_active_calibrations = 0
_guard_lock = threading.Lock()


def _cache_key(
    user_id: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    symbol: Optional[str],
    prompt_version: Optional[str],
    model: Optional[str],
    hold_days: int,
    limit: int,
    min_sample_size: Optional[int] = None,
    horizon: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> str:
    return "|".join(
        str(part) if part is not None else ""
        for part in (
            user_id,
            start_date,
            end_date,
            symbol,
            prompt_version,
            model,
            hold_days,
            limit,
            min_sample_size,
            horizon,
            profile_id,
        )
    )


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    if CALIBRATION_CACHE_TTL_SECONDS <= 0:
        return None
    with _cache_lock:
        item = _calibration_cache.get(key)
        if item is None:
            return None
        if time.monotonic() - item[0] > CALIBRATION_CACHE_TTL_SECONDS:
            _calibration_cache.pop(key, None)
            return None
        return item[1]


def _cache_put(key: str, result: Dict[str, Any]) -> None:
    with _cache_lock:
        _calibration_cache[key] = (time.monotonic(), result)
        if len(_calibration_cache) > MAX_CALIBRATION_CACHE_ENTRIES:
            oldest_key = min(_calibration_cache, key=lambda k: _calibration_cache[k][0])
            _calibration_cache.pop(oldest_key, None)


def _acquire_slot() -> bool:
    global _active_calibrations
    with _guard_lock:
        if _active_calibrations >= CALIBRATION_MAX_CONCURRENT:
            return False
        _active_calibrations += 1
        return True


def _release_slot() -> None:
    global _active_calibrations
    with _guard_lock:
        _active_calibrations = max(0, _active_calibrations - 1)


# ──────────────────────────────────────────────────────────────────────────────
# Bucket helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bucket_for(probability: float) -> Optional[Tuple[str, float, float]]:
    """Return the bucket whose half-open range contains ``probability``.

    The final bucket is closed on the right so an exact 1.0 lands in ``80+%``.
    """
    for label, low, high in _BUCKETS:
        if low <= probability < high:
            return label, low, high
    if probability == 1.0:
        return _BUCKETS[-1]
    return None


def _report_prompt_versions(report: ReportDB) -> List[str]:
    """Return the prompt-version hashes frozen onto the report.

    The snapshot lives under ``result_data.custom_prompt_snapshot.roles`` as a
    role-key -> {resolved_hash, ...} map.  Reports without a snapshot yield an
    empty list so they are excluded when a ``prompt_version`` filter is set.
    """
    result_data = report.result_data
    if not isinstance(result_data, dict):
        return []
    snapshot = result_data.get("custom_prompt_snapshot")
    if not isinstance(snapshot, dict):
        return []
    roles = snapshot.get("roles")
    if not isinstance(roles, dict):
        return []
    versions: List[str] = []
    for role in roles.values():
        if isinstance(role, dict):
            resolved_hash = role.get("resolved_hash")
            if isinstance(resolved_hash, str) and resolved_hash:
                versions.append(resolved_hash)
    return versions


def _report_model_names(report: ReportDB) -> List[str]:
    """Return the model names frozen onto the report.

    The snapshot lives under ``result_data.model_config_snapshot`` as a
    role-key -> {model_name, ...} map.
    """
    result_data = report.result_data
    if not isinstance(result_data, dict):
        return []
    snapshot = result_data.get("model_config_snapshot")
    if not isinstance(snapshot, dict):
        return []
    names: List[str] = []
    for role in snapshot.values():
        if isinstance(role, dict):
            model_name = role.get("model_name")
            if isinstance(model_name, str) and model_name:
                names.append(model_name)
    return names


def _matches_filter(values: List[str], needle: Optional[str]) -> bool:
    """Match a report attribute against a substring filter.

    A ``None`` filter matches everything.  Substring matching keeps the filter
    usable with model names and truncated prompt hashes alike.
    """
    if not needle:
        return True
    lowered = needle.strip().lower()
    if not lowered:
        return True
    return any(lowered in str(value).lower() for value in values)


def _normalize_symbol(raw: str) -> str:
    """Normalize a stock symbol for filtering.

    Reuses the pure-code branch of ``api.main._normalize_symbol`` (uppercase +
    6-digit CN suffix normalization) so ``600519.sh`` matches ``600519.SH``.
    The Chinese-name map fallback is intentionally omitted to keep this service
    free of network/stock-map loading.
    """
    s = (raw or "").strip().upper()
    m = re.search(r"(\d{6})(?:\.(SH|SZ|SS))?", s)
    if m:
        code = m.group(1)
        suffix = m.group(2)
        if suffix:
            if suffix == "SS":
                return f"{code}.SH"
            return f"{code}.{suffix}"
        market = "SH" if code.startswith(("5", "6", "9")) else "SZ"
        return f"{code}.{market}"
    m2 = re.search(r"([A-Z]{1,6}(?:\.[A-Z]{1,3})?)", s)
    if m2:
        return m2.group(1)
    return s


def _extract_report_probability(
    report: ReportDB,
    target_horizon: Optional[str] = None,
) -> Optional[float]:
    """Extract explicit numerical probability (0–1) from report DB column or result_data.

    Invariants (V-02 / AGENTS.md):
    1. NEVER converts confidence to probability (zero confidence-to-probability leakage).
    2. Respects target_horizon slice priority for dual-horizon reports.
    """
    rd = report.result_data if isinstance(report.result_data, dict) else {}

    # Target-horizon slice priority
    if target_horizon == HORIZON_SHORT and isinstance(rd.get("short_term"), dict):
        prob = rd["short_term"].get("probability")
        if prob is not None:
            try:
                p = float(prob)
                if 0.0 <= p <= 1.0:
                    return p
            except (ValueError, TypeError):
                pass
    elif target_horizon == HORIZON_MEDIUM and isinstance(rd.get("medium_term"), dict):
        prob = rd["medium_term"].get("probability")
        if prob is not None:
            try:
                p = float(prob)
                if 0.0 <= p <= 1.0:
                    return p
            except (ValueError, TypeError):
                pass

    if report.probability is not None:
        try:
            p = float(report.probability)
            if 0.0 <= p <= 1.0:
                return p
        except (ValueError, TypeError):
            pass

    if isinstance(rd, dict):
        prob = rd.get("probability")
        if prob is not None:
            try:
                p = float(prob)
                if 0.0 <= p <= 1.0:
                    return p
            except (ValueError, TypeError):
                pass
        dec_st = rd.get("decision_status")
        if isinstance(dec_st, dict):
            prob = dec_st.get("probability")
            if prob is not None:
                try:
                    p = float(prob)
                    if 0.0 <= p <= 1.0:
                        return p
                except (ValueError, TypeError):
                    pass
    return None


def _extract_report_winner(
    report: ReportDB,
    target_horizon: Optional[str] = None,
) -> Optional[str]:
    """Extract winner ('bull' or 'bear') from report."""
    rd = report.result_data
    if not isinstance(rd, dict):
        return None

    containers: List[Any] = []
    if target_horizon == HORIZON_SHORT and isinstance(rd.get("short_term"), dict):
        containers.append(rd["short_term"])
    elif target_horizon == HORIZON_MEDIUM and isinstance(rd.get("medium_term"), dict):
        containers.append(rd["medium_term"])

    containers.extend([
        rd,
        rd.get("investment_debate_state"),
        rd.get("short_term"),
        rd.get("primary"),
        rd.get("medium_term"),
    ])
    for c in containers:
        if not isinstance(c, dict):
            continue
        w = c.get("winner") or c.get("debate_winner")
        if w and str(w).strip().lower() in ("bull", "bear"):
            return str(w).strip().lower()
        mv = c.get("manager_verdict")
        if isinstance(mv, dict):
            w = mv.get("winner")
            if w and str(w).strip().lower() in ("bull", "bear"):
                return str(w).strip().lower()
        inv = c.get("investment_debate_state")
        if isinstance(inv, dict):
            mv2 = inv.get("manager_verdict")
            if isinstance(mv2, dict):
                w = mv2.get("winner")
                if w and str(w).strip().lower() in ("bull", "bear"):
                    return str(w).strip().lower()
            w = inv.get("winner") or inv.get("debate_winner")
            if w and str(w).strip().lower() in ("bull", "bear"):
                return str(w).strip().lower()
    return None


def _extract_report_directional_lean(
    report: ReportDB,
    target_horizon: Optional[str] = None,
) -> Optional[str]:
    """Extract directional lean ('bull' or 'bear') for WAIT diagnostic analysis."""
    winner = _extract_report_winner(report, target_horizon=target_horizon)
    if winner in ("bull", "bear"):
        return winner
    prob = _extract_report_probability(report, target_horizon=target_horizon)
    if prob is not None:
        if prob > 0.5:
            return "bull"
        elif prob < 0.5:
            return "bear"
    d = str(report.direction or "").upper()
    if any(k in d for k in ["BULL", "看多", "BUY"]):
        return "bull"
    if any(k in d for k in ["BEAR", "看空", "SELL"]):
        return "bear"
    rd = report.result_data if isinstance(report.result_data, dict) else {}
    direct = str(rd.get("direction") or "").upper()
    if any(k in direct for k in ["BULL", "看多"]):
        return "bull"
    if any(k in direct for k in ["BEAR", "看空"]):
        return "bear"
    return None


def _extract_report_horizon_info(report: ReportDB) -> Dict[str, Any]:
    """Extract horizon, profile_id, and profile validity from ReportDB."""
    rd = report.result_data if isinstance(report.result_data, dict) else {}

    raw_horizon = rd.get("horizon")
    profile_id = rd.get("profile_id")
    is_dual = bool(
        rd.get("mode") == "dual_horizon"
        or "short_term" in rd
        or "medium_term" in rd
    )

    available_horizons: List[str] = []
    resolution_source: Optional[str] = None
    meta = rd.get("horizon_run_metadata")
    if isinstance(meta, dict):
        if not profile_id:
            profile_id = meta.get("profile_id")
        resolved = meta.get("resolved")
        if isinstance(resolved, list):
            available_horizons.extend(str(h).lower() for h in resolved if isinstance(h, str))
        resolution_source = meta.get("resolution_source")

    if not resolution_source:
        resolution_source = rd.get("horizons_resolution_source")
        if not resolution_source:
            if rd.get("horizons_explicit"):
                resolution_source = "explicit"
            elif raw_horizon or meta:
                resolution_source = "default"
            else:
                resolution_source = "legacy"

    if not available_horizons:
        if is_dual:
            available_horizons = [HORIZON_SHORT, HORIZON_MEDIUM]
        elif raw_horizon and isinstance(raw_horizon, str):
            available_horizons = [raw_horizon.lower()]

    horizon: Optional[str] = None
    if raw_horizon and isinstance(raw_horizon, str):
        horizon = raw_horizon.lower()
    elif len(available_horizons) == 1:
        horizon = available_horizons[0]

    # Validate profile_id if explicitly specified
    is_valid_profile = True
    if profile_id is not None:
        if not isinstance(profile_id, str) or profile_id not in SUPPORTED_HORIZON_PROFILES:
            is_valid_profile = False

    # Validate horizons if explicitly populated
    for h in available_horizons:
        if h not in SUPPORTED_HORIZONS:
            is_valid_profile = False

    if horizon is not None and horizon not in SUPPORTED_HORIZONS:
        is_valid_profile = False

    return {
        "horizon": horizon if is_valid_profile and horizon in SUPPORTED_HORIZONS else horizon,
        "profile_id": profile_id,
        "is_valid_profile": is_valid_profile,
        "is_dual_horizon": is_dual,
        "available_horizons": available_horizons,
        "resolution_source": resolution_source,
    }


def _is_admissible_calibration_report(
    report: ReportDB,
    target_horizon: Optional[str] = None,
) -> Tuple[bool, bool, Optional[float], Optional[str]]:
    """Determine if report is eligible for calibration evaluation.

    Returns:
        (is_admissible, is_winner_only, probability, winner)

    Enforces V-02 pool isolation:
    - Rejects invalid profiles and corrupted horizon records.
    - Explicit target_horizon accepts only matching horizon records.
    - Default legacy call (target_horizon is None) preserves legacy/default T+5 samples,
      excluding explicit T+40 medium records or explicitly requested T+10 short records.
    - Dual-horizon reports are admitted under their respective slices or default.
    """
    from tradingagents.agents.utils.decision_status import (
        NON_DIRECTIONAL_TRADE_ACTIONS,
        NON_ELIGIBLE_ANALYSIS_STATUSES,
        is_calibration_eligible,
    )
    from tradingagents.agents.utils.shadow_credit import is_qualifying_v2_report

    if report.status != "completed":
        return False, False, None, None

    h_info = _extract_report_horizon_info(report)
    if not h_info["is_valid_profile"]:
        return False, False, None, None

    if target_horizon is not None:
        if (
            target_horizon not in h_info["available_horizons"]
            and h_info["horizon"] != target_horizon
        ):
            return False, False, None, None
    else:
        # Default legacy call: no horizon requested (T+5 evaluation)
        # "兼容旧样本，但默认不得混池"
        # Explicit single-horizon medium reports must never mix into default T+5
        if h_info["horizon"] == HORIZON_MEDIUM and not h_info["is_dual_horizon"]:
            return False, False, None, None
        # Explicitly requested single-horizon short (T+10) must also not mix into default T+5
        if (
            h_info["horizon"] == HORIZON_SHORT
            and not h_info["is_dual_horizon"]
            and h_info["resolution_source"] == "explicit"
        ):
            return False, False, None, None

    # Path 1: Explicit probability
    prob = _extract_report_probability(report, target_horizon=target_horizon)
    if prob is not None:
        if is_calibration_eligible(report):
            return True, False, prob, None

    # Path 2: Winner-only v2 report
    if report.analysis_status in NON_ELIGIBLE_ANALYSIS_STATUSES:
        return False, False, None, None
    if report.trade_action in NON_DIRECTIONAL_TRADE_ACTIONS:
        return False, False, None, None

    rd = report.result_data if isinstance(report.result_data, dict) else {}
    sample_dict = {
        "status": report.status,
        "result_data": rd,
        "analysis_status": report.analysis_status,
        "trade_action": report.trade_action,
        "probability": report.probability,
    }
    if not is_qualifying_v2_report(sample_dict):
        return False, False, None, None

    winner = _extract_report_winner(report, target_horizon=target_horizon)
    if winner in ("bull", "bear"):
        return True, True, None, winner

    return False, False, None, None


def _is_admissible_wait_report(
    report: ReportDB,
    target_horizon: Optional[str] = None,
) -> bool:
    """Determine if a report is eligible for WAIT directional diagnostic."""
    if report.status != "completed":
        return False
    if report.analysis_status != "VALID":
        return False
    if report.trade_action != "WAIT":
        return False
    h_info = _extract_report_horizon_info(report)
    if not h_info["is_valid_profile"]:
        return False
    if target_horizon is not None:
        if (
            target_horizon not in h_info["available_horizons"]
            and h_info["horizon"] != target_horizon
        ):
            return False
    else:
        if h_info["horizon"] == HORIZON_MEDIUM and not h_info["is_dual_horizon"]:
            return False
        if (
            h_info["horizon"] == HORIZON_SHORT
            and not h_info["is_dual_horizon"]
            and h_info["resolution_source"] == "explicit"
        ):
            return False
    return True


def _query_reports(
    db: Session,
    *,
    user_id: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    symbol: Optional[str],
    prompt_version: Optional[str],
    model: Optional[str],
    limit: int,
    hold_days: int,
    horizon: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> Tuple[List[ReportDB], bool, int, Dict[str, int], List[ReportDB]]:
    """Load completed reports that carry a probability or qualifying v2 winner, applying filters.

    Date/symbol/user filters run in SQL.  Hold-window completeness also runs in
    SQL, BEFORE the ``limit`` truncation, so the newest reports (whose hold
    window has not yet elapsed) are excluded from selection rather than being
    silently skipped after truncation; they are counted separately and reported
    as ``skipped_no_outcome`` so the UI can distinguish "hold window not over"
    from "no report".  Prompt-version, model, and horizon filters run in Python because
    they inspect the snapshot JSON and horizon metadata nested in ``result_data``.

    Returns ``(rows, truncated_before_filter, skipped_incomplete_window, exclusion_stats, wait_rows)``.
    ``truncated_before_filter`` is True only when the pre-filter candidate scan
    actually hit its cap (checked by fetching one extra row).
    """
    from tradingagents.agents.utils.decision_status import (
        ANALYSIS_VALID,
        NON_DIRECTIONAL_TRADE_ACTIONS,
    )

    base_query = db.query(ReportDB).filter(
        ReportDB.status == "completed",
    )
    if user_id:
        base_query = base_query.filter(ReportDB.user_id == user_id)
    if symbol:
        base_query = base_query.filter(ReportDB.symbol == _normalize_symbol(symbol))
    if start_date:
        base_query = base_query.filter(ReportDB.trade_date >= start_date)
    if end_date:
        base_query = base_query.filter(ReportDB.trade_date <= end_date)

    # Count exclusions in the filtered scope (D-009 P0-1: legacy null, invalid, abstain, non-directional)
    excluded_null = base_query.filter(ReportDB.analysis_status.is_(None)).count()
    excluded_invalid = base_query.filter(
        ReportDB.analysis_status.in_(["INVALID_RUN", "DATA_ERROR"])
    ).count()
    excluded_abstain = base_query.filter(
        ReportDB.analysis_status.in_(["ABSTAIN", "PARTIAL"])
    ).count()
    excluded_no_trade = base_query.filter(
        ReportDB.analysis_status == ANALYSIS_VALID,
        ReportDB.trade_action.in_(list(NON_DIRECTIONAL_TRADE_ACTIONS)),
    ).count()
    excluded_wait = base_query.filter(
        ReportDB.analysis_status == ANALYSIS_VALID,
        ReportDB.trade_action == "WAIT",
    ).count()

    # Only explicit VALID directional rows or candidate v2 rows are calibration-eligible
    query = base_query.filter(
        (
            (ReportDB.analysis_status == ANALYSIS_VALID)
            & (
                (ReportDB.trade_action.is_(None))
                | (~ReportDB.trade_action.in_(list(NON_DIRECTIONAL_TRADE_ACTIONS)))
            )
        )
        | (ReportDB.analysis_status.is_(None))
    )

    # Hold-window completeness before truncation: only reports whose window has
    # elapsed are eligible; count the too-recent ones so callers know the view
    # was non-empty but unevaluable yet.
    skipped_incomplete = 0
    cutoff = _hold_window_cutoff(hold_days)
    if cutoff:
        skipped_incomplete = query.filter(ReportDB.trade_date > cutoff).count()
        query = query.filter(ReportDB.trade_date <= cutoff)

    has_snapshot_filters = bool(prompt_version or model)
    has_horizon_filter = horizon is not None
    scan_cap = max(limit, MAX_CALIBRATION_FILTER_SCAN) if (has_snapshot_filters or has_horizon_filter) else limit

    candidate_rows = query.order_by(ReportDB.created_at.desc()).limit(scan_cap + 1).all()
    truncated_before_filter = (len(candidate_rows) > scan_cap) if (has_snapshot_filters or has_horizon_filter) else False
    candidate_rows = candidate_rows[:scan_cap]

    excluded_invalid_profile = 0
    excluded_mismatched_horizon = 0
    admissible_rows: List[ReportDB] = []

    for row in candidate_rows:
        h_info = _extract_report_horizon_info(row)
        if not h_info["is_valid_profile"]:
            excluded_invalid_profile += 1
            continue

        if horizon is not None:
            if (
                horizon not in h_info["available_horizons"]
                and h_info["horizon"] != horizon
            ):
                excluded_mismatched_horizon += 1
                continue
        else:
            # Default legacy T+5 call: explicit single-horizon medium or explicitly requested short cannot enter legacy pool
            if (
                (h_info["horizon"] == HORIZON_MEDIUM and not h_info["is_dual_horizon"])
                or (
                    h_info["horizon"] == HORIZON_SHORT
                    and not h_info["is_dual_horizon"]
                    and h_info["resolution_source"] == "explicit"
                )
            ):
                excluded_mismatched_horizon += 1
                continue

        if prompt_version and not _matches_filter(_report_prompt_versions(row), prompt_version):
            continue
        if model and not _matches_filter(_report_model_names(row), model):
            continue

        is_adm, _, _, _ = _is_admissible_calibration_report(row, target_horizon=horizon)
        if is_adm:
            admissible_rows.append(row)
            if len(admissible_rows) >= limit:
                break

    # Separately query VALID WAIT reports within scope for directional diagnostics (V-02)
    wait_query = base_query.filter(
        ReportDB.analysis_status == ANALYSIS_VALID,
        ReportDB.trade_action == "WAIT",
    )
    if cutoff:
        wait_candidates = (
            wait_query.filter(ReportDB.trade_date <= cutoff)
            .order_by(ReportDB.created_at.desc())
            .limit(scan_cap)
            .all()
        )
    else:
        wait_candidates = (
            wait_query.order_by(ReportDB.created_at.desc())
            .limit(scan_cap)
            .all()
        )

    wait_rows: List[ReportDB] = []
    for w in wait_candidates:
        if not _is_admissible_wait_report(w, target_horizon=horizon):
            continue
        if prompt_version and not _matches_filter(_report_prompt_versions(w), prompt_version):
            continue
        if model and not _matches_filter(_report_model_names(w), model):
            continue
        wait_rows.append(w)
        if len(wait_rows) >= limit:
            break

    excluded_total = (
        excluded_null
        + excluded_invalid
        + excluded_abstain
        + excluded_no_trade
        + excluded_invalid_profile
        + excluded_mismatched_horizon
    )
    exclusion_stats = {
        "excluded_null": excluded_null,
        "excluded_invalid": excluded_invalid,
        "excluded_abstain": excluded_abstain,
        "excluded_no_trade": excluded_no_trade,
        "excluded_wait": excluded_wait,
        "excluded_invalid_profile": excluded_invalid_profile,
        "excluded_mismatched_horizon": excluded_mismatched_horizon,
        "excluded_total": excluded_total,
    }

    return admissible_rows, truncated_before_filter, skipped_incomplete, exclusion_stats, wait_rows


# ──────────────────────────────────────────────────────────────────────────────
# Outcome resolution (hold-window integrity)
# ──────────────────────────────────────────────────────────────────────────────

def _today() -> Any:
    """UTC calendar date used for hold-window completeness checks."""
    return datetime.now(timezone.utc).date()


def _hold_window_complete(trade_date: Optional[str], hold_days: int) -> bool:
    """Return False when the report date is too recent to have a full hold window.

    Conservative calendar pre-check: at least ``hold_days * HOLD_CALENDAR_FACTOR``
    calendar days must have elapsed, so a truncated window is never used to
    conclude a rise/fall.  The authoritative check is the row-count guard inside
    ``_get_price_after_strict``.
    """
    if not trade_date:
        return False
    try:
        report_day = datetime.strptime(trade_date, "%Y-%m-%d").date()
    except ValueError:
        return False
    days_since = (_today() - report_day).days
    return days_since >= max(0, int(hold_days * HOLD_CALENDAR_FACTOR))


def _hold_window_cutoff(hold_days: int) -> Optional[str]:
    """Earliest ``trade_date`` eligible for evaluation (inclusive).

    Mirrors ``_hold_window_complete`` so the report-selection stage can exclude
    too-recent reports BEFORE the ``limit`` truncation — otherwise an active
    account's newest ``limit`` reports would all be skipped as incomplete and the
    default view would come back empty.
    """
    required_days = max(0, int(hold_days * HOLD_CALENDAR_FACTOR))
    return (_today() - timedelta(days=required_days)).strftime("%Y-%m-%d")


def _get_price_after_strict(symbol: str, base_date: str, hold_days: int) -> Optional[float]:
    """Close price ``hold_days`` trading rows after ``base_date``, or None when
    the fetched series does not contain a full hold window.

    Unlike ``backtest_service._get_price_after`` (which collapses the window
    when the series is short: ``if len(df) < hold_days: hold_days = len(df)-1``),
    this version refuses to conclude on a truncated window so recent reports are
    never given premature rise/fall outcomes.
    """
    try:
        import pandas as pd

        from tradingagents.dataflows.interface import route_to_vendor

        fmt = "%Y-%m-%d"
        start_dt = datetime.strptime(base_date, fmt)
        fetch_start = (start_dt + timedelta(days=1)).strftime(fmt)
        fetch_end = (start_dt + timedelta(days=hold_days + 30)).strftime(fmt)

        csv_data = route_to_vendor("get_stock_data", symbol, fetch_start, fetch_end)
        if not csv_data:
            return None

        df = pd.read_csv(pd.io.common.StringIO(csv_data))
        close_cols = [c for c in df.columns if "close" in c.lower() or "收盘" in c]
        date_cols = [c for c in df.columns if "date" in c.lower() or "日期" in c or "time" in c.lower()]
        if not close_cols or not date_cols:
            return None

        date_col = date_cols[0]
        close_col = close_cols[0]
        df[date_col] = df[date_col].astype(str).str[:10]
        df = df.sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True)

        from tradingagents.dataflows.trade_calendar import trading_days_forward

        df_dates = sorted(df[date_col].unique())
        try:
            target_days = trading_days_forward(base_date, hold_days, calendar_dates=df_dates)
            if len(target_days) < hold_days:
                return None
            target_date = target_days[hold_days - 1]
        except Exception:
            return None

        match = df[df[date_col] == target_date]
        if not match.empty:
            val = float(match.iloc[0][close_col])
            if val > 0:
                return val
        return None
    except Exception:
        return None


def _resolve_outcome(
    report: ReportDB,
    hold_days: int,
    *,
    price_after: Optional[Callable[[str, str, int], Optional[float]]] = None,
    price_on: Optional[Callable[[str, str], Optional[float]]] = None,
) -> Optional[bool]:
    """Resolve whether the report's horizon actually saw a price rise.

    Returns True when the close price ``hold_days`` trading days after the
    report date is strictly above the close price on/near the report date,
    False when below, and None when the outcome is unknown — either because the
    hold window is not yet complete (no premature conclusion) or because prices
    are unavailable.  The price fetchers default to the module-level helpers
    (looked up at call time so tests can ``patch.object`` them) and are
    injectable directly.
    """
    if not _hold_window_complete(report.trade_date, hold_days):
        return None
    price_after = price_after or _get_price_after_strict
    price_on = price_on or _get_price_on
    try:
        entry = price_on(report.symbol, report.trade_date)
        exit_ = price_after(report.symbol, report.trade_date, hold_days)
    except Exception:  # network/data provider hiccup — treat as unknown
        logger.warning(
            "calibration: price fetch failed for report %s (%s @ %s)",
            report.id,
            report.symbol,
            report.trade_date,
        )
        return None
    if entry is None or exit_ is None or entry <= 0 or exit_ <= 0:
        return None
    return exit_ > entry


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_calibration_unlocked(
    db: Session,
    *,
    user_id: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    symbol: Optional[str],
    prompt_version: Optional[str],
    model: Optional[str],
    hold_days: int,
    limit: int,
    outcome_resolver: Optional[Callable[[ReportDB], Optional[bool]]],
    min_sample_size: Optional[int] = None,
    horizon: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the reliability curve + Brier score for historical reports.

    Each report contributes its predicted rise probability and the observed
    binary outcome (resolved via ``outcome_resolver``, or the default price
    window).  Reports whose outcome cannot be resolved are counted separately
    and excluded from the curve and Brier score.

    For qualifying v2 debate reports without probability (winner-only), we admit
    them into the evaluable sample set without fabricating probabilities from
    confidence or text. They contribute to winner direction hit metrics
    (winner_only_admitted, winner_only_hits, winner_only_hit_rate), while
    probability reliability curve buckets and Brier score are strictly derived
    from reports with explicit probabilities.
    """
    reports, truncated_before_filter, skipped_incomplete, exclusion_stats, wait_rows = _query_reports(
        db,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
        prompt_version=prompt_version,
        model=model,
        limit=limit,
        hold_days=hold_days,
        horizon=horizon,
        profile_id=profile_id,
    )

    prob_samples: List[Tuple[float, bool]] = []
    winner_only_admitted = 0
    winner_only_hits = 0
    winner_bull_count = 0
    winner_bull_hits = 0
    winner_bear_count = 0
    winner_bear_hits = 0

    # V-02: Track samples for grouped statistics by (horizon, profile_id, model, prompt_version)
    groups_map: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    # Too-recent reports (hold window not yet elapsed) are excluded at selection
    # time; they count as skipped alongside reports whose price is unavailable.
    skipped_no_outcome = skipped_incomplete
    resolve = outcome_resolver or (lambda row: _resolve_outcome(row, hold_days))

    for report in reports:
        is_admissible, is_winner_only, probability, winner = _is_admissible_calibration_report(
            report, target_horizon=horizon
        )
        if not is_admissible:
            continue

        outcome = resolve(report)
        if outcome is None:
            skipped_no_outcome += 1
            continue

        # Extract group metadata for grouped statistics (V-02)
        h_info = _extract_report_horizon_info(report)
        grp_horizon = h_info["horizon"] or (horizon or "legacy_t5")
        grp_profile = h_info["profile_id"] or (profile_id or ("horizon_profile_v1" if horizon in SUPPORTED_HORIZONS else "legacy"))
        m_names = _report_model_names(report)
        grp_model = m_names[0] if m_names else "default"
        p_vers = _report_prompt_versions(report)
        grp_prompt = p_vers[0] if p_vers else "default"
        grp_key = (grp_horizon, grp_profile, grp_model, grp_prompt)

        if grp_key not in groups_map:
            groups_map[grp_key] = {
                "horizon": grp_horizon,
                "profile_id": grp_profile,
                "model": grp_model,
                "prompt_version": grp_prompt,
                "prob_samples": [],
                "winner_only_admitted": 0,
                "winner_only_hits": 0,
            }

        if is_winner_only:
            winner_only_admitted += 1
            hit = False
            # Direction hit evaluation:
            # - 'bull' expects rise (outcome is True -> hit)
            # - 'bear' expects fall (outcome is False -> hit)
            if winner == "bull":
                winner_bull_count += 1
                hit = bool(outcome is True)
                if hit:
                    winner_bull_hits += 1
            else:  # winner == "bear"
                winner_bear_count += 1
                hit = bool(outcome is False)
                if hit:
                    winner_bear_hits += 1
            if hit:
                winner_only_hits += 1

            groups_map[grp_key]["winner_only_admitted"] += 1
            if hit:
                groups_map[grp_key]["winner_only_hits"] += 1
        else:
            if probability is not None:
                prob_samples.append((probability, outcome))
                groups_map[grp_key]["prob_samples"].append((probability, outcome))

    buckets = [_empty_bucket(label, low, high) for label, low, high in _BUCKETS]
    for probability, outcome in prob_samples:
        bucket = _bucket_for(probability)
        if bucket is None:
            continue
        entry = next(
            item for item in buckets if item["bucket"] == bucket[0]
        )
        entry["count"] += 1
        entry["rise_count"] += 1 if outcome else 0
        entry["prob_sum"] += probability

    for entry in buckets:
        count = entry.pop("count", 0)
        prob_sum = entry.pop("prob_sum", 0.0)
        rise_count = entry.pop("rise_count", 0)
        entry["count"] = count
        entry["rise_count"] = rise_count
        entry["rise_rate"] = round(rise_count / count * 100, 1) if count else None
        entry["avg_probability"] = round(prob_sum / count, 3) if count else None

    winner_only_hit_rate = (
        round(winner_only_hits / winner_only_admitted * 100, 1)
        if winner_only_admitted > 0
        else None
    )

    total_sample_size = len(prob_samples) + winner_only_admitted
    prob_sample_size = len(prob_samples)

    effective_min_sample_size = (
        min_sample_size
        if min_sample_size is not None and min_sample_size >= 0
        else DEFAULT_MIN_CALIBRATION_SAMPLE_SIZE
    )

    sample_sufficient = prob_sample_size >= effective_min_sample_size and prob_sample_size > 0

    if sample_sufficient:
        brier = _brier_score(prob_samples)
        insufficient_reason = None
    else:
        brier = None
        if total_sample_size == 0:
            insufficient_reason = "当前筛选条件下暂无带概率的历史报告，调整日期范围或过滤条件后重试。"
        else:
            insufficient_reason = (
                f"样本量不足以支撑校准结论 (当前有效样本 {prob_sample_size} 份，低于最小阈值 {effective_min_sample_size} 份)"
            )
        # 契约 3: 样本不足时：brier_score 与各分桶的 rise_rate 必须为 null，前端不得绘制柱体，不得显示任何精确数值。
        for entry in buckets:
            entry["rise_rate"] = None

    # V-02: Compute grouped statistics per (horizon, profile_id, model, prompt_version)
    grouped_stats = []
    for g_key, g_val in groups_map.items():
        g_prob_samples = g_val["prob_samples"]
        g_w_admitted = g_val["winner_only_admitted"]
        g_w_hits = g_val["winner_only_hits"]
        g_prob_count = len(g_prob_samples)
        g_total_count = g_prob_count + g_w_admitted
        g_sufficient = g_prob_count >= effective_min_sample_size and g_prob_count > 0
        g_brier = _brier_score(g_prob_samples) if g_sufficient else None

        g_buckets = [_empty_bucket(label, low, high) for label, low, high in _BUCKETS]
        for prob, out in g_prob_samples:
            b = _bucket_for(prob)
            if b is None:
                continue
            item = next(it for it in g_buckets if it["bucket"] == b[0])
            item["count"] += 1
            item["rise_count"] += 1 if out else 0
            item["prob_sum"] += prob
        for it in g_buckets:
            c = it.pop("count", 0)
            ps = it.pop("prob_sum", 0.0)
            rc = it.pop("rise_count", 0)
            it["count"] = c
            it["rise_count"] = rc
            it["rise_rate"] = round(rc / c * 100, 1) if (c and g_sufficient) else None
            it["avg_probability"] = round(ps / c, 3) if c else None

        grouped_stats.append({
            "horizon": g_val["horizon"],
            "profile_id": g_val["profile_id"],
            "model": g_val["model"],
            "prompt_version": g_val["prompt_version"],
            "sample_size": g_total_count,
            "probability_sample_size": g_prob_count,
            "sample_sufficient": g_sufficient,
            "brier_score": g_brier,
            "winner_only_admitted": g_w_admitted,
            "winner_only_hits": g_w_hits,
            "winner_only_hit_rate": (
                round(g_w_hits / g_w_admitted * 100, 1) if g_w_admitted > 0 else None
            ),
            "buckets": g_buckets,
        })

    # V-02: WAIT direction diagnostic (strictly separated from trading performance)
    wait_evaluable = 0
    wait_rise_count = 0
    wait_fall_count = 0
    wait_bull_lean = 0
    wait_bull_hits = 0
    wait_bear_lean = 0
    wait_bear_hits = 0

    for w_rep in wait_rows:
        w_outcome = resolve(w_rep)
        if w_outcome is None:
            continue
        wait_evaluable += 1
        if w_outcome is True:
            wait_rise_count += 1
        else:
            wait_fall_count += 1

        lean = _extract_report_directional_lean(w_rep, target_horizon=horizon)
        if lean == "bull":
            wait_bull_lean += 1
            if w_outcome is True:
                wait_bull_hits += 1
        elif lean == "bear":
            wait_bear_lean += 1
            if w_outcome is False:
                wait_bear_hits += 1

    total_wait_leans = wait_bull_lean + wait_bear_lean
    total_wait_hits = wait_bull_hits + wait_bear_hits
    wait_diagnostic = {
        "total_wait_count": len(wait_rows),
        "evaluable_wait_count": wait_evaluable,
        "rise_count": wait_rise_count,
        "fall_count": wait_fall_count,
        "rise_rate": (
            round(wait_rise_count / wait_evaluable * 100, 1)
            if wait_evaluable > 0
            else None
        ),
        "directional_lean_count": total_wait_leans,
        "directional_hits": total_wait_hits,
        "directional_hit_rate": (
            round(total_wait_hits / total_wait_leans * 100, 1)
            if total_wait_leans > 0
            else None
        ),
        "bull_lean_count": wait_bull_lean,
        "bull_hits": wait_bull_hits,
        "bear_lean_count": wait_bear_lean,
        "bear_hits": wait_bear_hits,
    }

    return {
        "brier_score": brier,
        "sample_size": total_sample_size,
        "probability_sample_size": prob_sample_size,
        "sample_sufficient": sample_sufficient,
        "min_sample_size": effective_min_sample_size,
        "insufficient_reason": insufficient_reason,
        "winner_only_admitted": winner_only_admitted,
        "winner_only_hits": winner_only_hits,
        "winner_only_hit_rate": winner_only_hit_rate,
        "winner_only_stats": {
            "admitted": winner_only_admitted,
            "hits": winner_only_hits,
            "hit_rate": winner_only_hit_rate,
            "bull_count": winner_bull_count,
            "bull_hits": winner_bull_hits,
            "bear_count": winner_bear_count,
            "bear_hits": winner_bear_hits,
        },
        "wait_diagnostic": wait_diagnostic,
        "grouped_stats": grouped_stats,
        "horizon": horizon,
        "profile_id": profile_id,
        "skipped_no_outcome": skipped_no_outcome,
        "truncated_before_filter": truncated_before_filter,
        "buckets": buckets,
        "excluded_null": exclusion_stats["excluded_null"],
        "excluded_invalid": exclusion_stats["excluded_invalid"],
        "excluded_abstain": exclusion_stats["excluded_abstain"],
        "excluded_no_trade": exclusion_stats["excluded_no_trade"],
        "excluded_wait": exclusion_stats["excluded_wait"],
        "excluded_invalid_profile": exclusion_stats["excluded_invalid_profile"],
        "excluded_mismatched_horizon": exclusion_stats["excluded_mismatched_horizon"],
        "excluded_incomplete_outcome": skipped_no_outcome,
        "excluded_total": exclusion_stats["excluded_total"] + skipped_no_outcome,
        "excluded_counts": {
            "legacy_null": exclusion_stats["excluded_null"],
            "invalid": exclusion_stats["excluded_invalid"],
            "abstain": exclusion_stats["excluded_abstain"],
            "no_trade": exclusion_stats["excluded_no_trade"],
            "wait": exclusion_stats["excluded_wait"],
            "incomplete_outcome": skipped_no_outcome,
            "incomplete": skipped_no_outcome,
            "invalid_profile": exclusion_stats["excluded_invalid_profile"],
            "mismatched_horizon": exclusion_stats["excluded_mismatched_horizon"],
            "total": exclusion_stats["excluded_total"] + skipped_no_outcome,
        },
        "price_basis": PRICE_BASIS_VENDOR_QFQ,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "symbol": symbol,
            "prompt_version": prompt_version,
            "model": model,
            "hold_days": hold_days,
            "limit": limit,
            "horizon": horizon,
            "profile_id": profile_id,
        },
    }


def compute_calibration(
    db: Session,
    *,
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol: Optional[str] = None,
    prompt_version: Optional[str] = None,
    model: Optional[str] = None,
    hold_days: int = DEFAULT_HOLD_DAYS,
    limit: Optional[int] = None,
    outcome_resolver: Optional[Callable[[ReportDB], Optional[bool]]] = None,
    min_sample_size: Optional[int] = None,
    horizon: Optional[str] = None,
    profile_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the reliability curve + Brier score, guarded by cache + concurrency.

    ``outcome_resolver`` bypasses the result cache (used by tests to inject
    deterministic outcomes); production callers leave it unset.

    Multi-horizon routing (V-02):
    - When horizon is omitted: defaults to legacy T+5 evaluation (hold_days=5).
    - When horizon is 'short': routes to canonical T+10 (hold_days=10).
    - When horizon is 'medium': routes to canonical T+40 (hold_days=40).
    """
    requested = limit or DEFAULT_CALIBRATION_LIMIT
    effective_limit = min(max(1, requested), MAX_CALIBRATION_LIMIT)

    effective_horizon: Optional[str] = None
    if horizon is not None:
        if not isinstance(horizon, str) or horizon.lower() not in SUPPORTED_HORIZONS:
            raise ValueError(
                f"Unsupported horizon {horizon!r}. Supported horizons: {list(SUPPORTED_HORIZONS)}"
            )
        effective_horizon = horizon.lower()
        if profile_id is not None and profile_id not in SUPPORTED_HORIZON_PROFILES:
            raise ValueError(f"Unsupported profile_id {profile_id!r}")
        effective_profile_id = profile_id or HORIZON_PROFILE_ID_V1
        if hold_days == DEFAULT_HOLD_DAYS:
            if effective_horizon == HORIZON_SHORT:
                effective_hold_days = PRIMARY_EVAL_OFFSET_SHORT  # 10
            elif effective_horizon == HORIZON_MEDIUM:
                effective_hold_days = PRIMARY_EVAL_OFFSET_MEDIUM  # 40
            else:
                effective_hold_days = hold_days
        else:
            effective_hold_days = hold_days
    else:
        if profile_id is not None and profile_id not in SUPPORTED_HORIZON_PROFILES:
            raise ValueError(f"Unsupported profile_id {profile_id!r}")
        effective_profile_id = profile_id
        effective_hold_days = hold_days

    key = _cache_key(
        user_id,
        start_date,
        end_date,
        symbol,
        prompt_version,
        model,
        effective_hold_days,
        effective_limit,
        min_sample_size,
        effective_horizon,
        effective_profile_id,
    )
    if outcome_resolver is None:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    if not _acquire_slot():
        raise CalibrationBusyError("校准度计算繁忙，请稍后重试")

    try:
        result = _compute_calibration_unlocked(
            db,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            prompt_version=prompt_version,
            model=model,
            hold_days=effective_hold_days,
            limit=effective_limit,
            outcome_resolver=outcome_resolver,
            min_sample_size=min_sample_size,
            horizon=effective_horizon,
            profile_id=effective_profile_id,
        )
    finally:
        _release_slot()

    if outcome_resolver is None:
        _cache_put(key, result)
    return result


def _empty_bucket(label: str, low: float, high: float) -> Dict[str, Any]:
    return {
        "bucket": label,
        "probability_min": low,
        "probability_max": high,
        "count": 0,
        "rise_count": 0,
        "rise_rate": None,
        "avg_probability": None,
        "prob_sum": 0.0,
    }


def _brier_score(samples: List[Tuple[float, bool]]) -> Optional[float]:
    """Brier score = mean((predicted - observed) ** 2) over evaluated samples.

    Lower is better; 0 = perfect, 1 = worst.  Requires at least one evaluated
    sample.
    """
    if not samples:
        return None
    total = 0.0
    for probability, outcome in samples:
        total += (probability - (1.0 if outcome else 0.0)) ** 2
    return round(total / len(samples), 4)
