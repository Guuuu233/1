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
from api.services.backtest_service import _get_price_on

logger = logging.getLogger(__name__)

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
) -> Tuple[List[ReportDB], bool]:
    """Load completed reports that carry a probability, applying filters.

    Date/symbol/user filters run in SQL.  Prompt-version and model filters run
    in Python because they inspect the snapshot JSON nested in ``result_data``
    (SQLite JSON queries are unreliable across backends); to avoid the sampling
    bias of filtering after a tight SQL limit, snapshot filtering happens on a
    wide candidate scan BEFORE the final ``limit`` truncation.  The second
    return value reports whether that candidate scan hit its cap, i.e. whether
    the filter result may be a biased sample.
    """
    query = db.query(ReportDB).filter(
        ReportDB.status == "completed",
        ReportDB.probability.isnot(None),
    )
    if user_id:
        query = query.filter(ReportDB.user_id == user_id)
    if symbol:
        query = query.filter(ReportDB.symbol == _normalize_symbol(symbol))
    if start_date:
        query = query.filter(ReportDB.trade_date >= start_date)
    if end_date:
        query = query.filter(ReportDB.trade_date <= end_date)

    has_snapshot_filters = bool(prompt_version or model)
    if has_snapshot_filters:
        scan_cap = max(limit, MAX_CALIBRATION_FILTER_SCAN)
        rows = query.order_by(ReportDB.created_at.desc()).limit(scan_cap).all()
        truncated_before_filter = len(rows) >= scan_cap
        rows = [
            row
            for row in rows
            if _matches_filter(_report_prompt_versions(row), prompt_version)
            and _matches_filter(_report_model_names(row), model)
        ]
        rows = rows[:limit]
    else:
        rows = query.order_by(ReportDB.created_at.desc()).limit(limit).all()
        truncated_before_filter = False
    return rows, truncated_before_filter


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
        df = df.sort_values(date_cols[0]).reset_index(drop=True)
        if len(df) < max(1, hold_days):
            return None
        return float(df[close_cols[0]].iloc[hold_days - 1])
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
) -> Dict[str, Any]:
    """Compute the reliability curve + Brier score for historical reports.

    Each report contributes its predicted rise probability and the observed
    binary outcome (resolved via ``outcome_resolver``, or the default price
    window).  Reports whose outcome cannot be resolved are counted separately
    and excluded from the curve and Brier score.
    """
    reports, truncated_before_filter = _query_reports(
        db,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
        prompt_version=prompt_version,
        model=model,
        limit=limit,
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

    return {
        "brier_score": _brier_score(samples),
        "sample_size": len(samples),
        "skipped_no_outcome": skipped_no_outcome,
        "truncated_before_filter": truncated_before_filter,
        "buckets": buckets,
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "symbol": symbol,
            "prompt_version": prompt_version,
            "model": model,
            "hold_days": hold_days,
            "limit": limit,
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
) -> Dict[str, Any]:
    """Compute the reliability curve + Brier score, guarded by cache + concurrency.

    ``outcome_resolver`` bypasses the result cache (used by tests to inject
    deterministic outcomes); production callers leave it unset.
    """
    requested = limit or DEFAULT_CALIBRATION_LIMIT
    effective_limit = min(max(1, requested), MAX_CALIBRATION_LIMIT)

    key = _cache_key(
        user_id,
        start_date,
        end_date,
        symbol,
        prompt_version,
        model,
        hold_days,
        effective_limit,
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
            hold_days=hold_days,
            limit=effective_limit,
            outcome_resolver=outcome_resolver,
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
