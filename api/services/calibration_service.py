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
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from api.database import ReportDB
from api.services.backtest_service import _get_price_after, _get_price_on

logger = logging.getLogger(__name__)

# Bounded evaluation: fetching hold-window prices is I/O-heavy, so cap how many
# reports a single calibration run resolves.  Mirrors backtest_service's
# env-var-bounded worker pool design.
MAX_CALIBRATION_REPORTS = max(1, int(os.getenv("CALIBRATION_MAX_REPORTS", "200")))

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
) -> List[ReportDB]:
    """Load completed reports that carry a probability, applying filters.

    Date/symbol/user filters run in SQL; prompt-version and model filters run
    in Python because they inspect the snapshot JSON nested in ``result_data``
    (SQLite JSON queries are unreliable across backends).
    """
    query = db.query(ReportDB).filter(
        ReportDB.status == "completed",
        ReportDB.probability.isnot(None),
    )
    if user_id:
        query = query.filter(ReportDB.user_id == user_id)
    if symbol:
        query = query.filter(ReportDB.symbol == symbol)
    if start_date:
        query = query.filter(ReportDB.trade_date >= start_date)
    if end_date:
        query = query.filter(ReportDB.trade_date <= end_date)

    rows = query.order_by(ReportDB.created_at.desc()).limit(limit).all()

    if prompt_version or model:
        rows = [
            row
            for row in rows
            if _matches_filter(_report_prompt_versions(row), prompt_version)
            and _matches_filter(_report_model_names(row), model)
        ]
    return rows


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
    False when below, and None when prices are unavailable (outcome unknown).
    The price fetchers default to the module-level helpers (looked up at call
    time so tests can ``patch.object`` them) and are injectable directly.
    """
    price_after = price_after or _get_price_after
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
    if entry is None or exit_ is None or entry <= 0:
        return None
    return exit_ > entry


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
) -> Dict[str, Any]:
    """Compute the reliability curve + Brier score for historical reports.

    Each report contributes its predicted rise probability and the observed
    binary outcome (resolved via ``outcome_resolver``, or the default price
    window).  Reports whose outcome cannot be resolved are counted separately
    and excluded from the curve and Brier score.
    """
    effective_limit = max(1, limit or MAX_CALIBRATION_REPORTS)
    reports = _query_reports(
        db,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
        prompt_version=prompt_version,
        model=model,
        limit=effective_limit,
    )

    samples: List[Tuple[float, bool]] = []
    skipped_no_outcome = 0
    resolve = outcome_resolver or (lambda row: _resolve_outcome(row, hold_days))

    for report in reports:
        probability = float(report.probability)
        outcome = resolve(report)
        if outcome is None:
            skipped_no_outcome += 1
            continue
        samples.append((probability, outcome))

    buckets = [_empty_bucket(label, low, high) for label, low, high in _BUCKETS]
    for probability, outcome in samples:
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

    brier_score = _brier_score(samples)

    return {
        "brier_score": brier_score,
        "sample_size": len(samples),
        "skipped_no_outcome": skipped_no_outcome,
        "buckets": buckets,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "symbol": symbol,
            "prompt_version": prompt_version,
            "model": model,
            "hold_days": hold_days,
            "limit": effective_limit,
        },
    }


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
