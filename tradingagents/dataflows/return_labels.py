"""Multi-horizon return labels and trading calendar window contracts (V-01-1).

This module defines the pure type contracts and immutable calendar window resolution
for multi-horizon performance evaluation (short T+10, medium T+40).

Strict V-01-1 boundaries:
1. Pure functions and data structures only: zero network, zero side effects, zero DB access.
2. Does not fetch prices, does not calculate returns, does not evaluate suspensions or limit moves.
3. Reads evaluation offsets directly from canonical HORIZON_PROFILE_V1.
4. HorizonCalendarWindow outputs strict boolean is_due; it does not claim evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple, TypedDict

from tradingagents.graph.horizon_profile import (
    HORIZON_MEDIUM,
    HORIZON_PROFILE_V1,
    HORIZON_SHORT,
    SUPPORTED_HORIZONS,
)

# ---------------------------------------------------------------------------
# Outcome & Return Types
# ---------------------------------------------------------------------------


class OutcomeStatus(str, Enum):
    """V-01 outcome status enumeration (7 mutually exclusive terminal/interim states).

    Defined here as a shared type contract for downstream evaluation engines (V-01-2+).
    """

    PENDING_DUE = "pending_due"
    SUSPENSION = "suspension"
    DATA_MISSING = "data_missing"
    PROVIDER_FAILURE = "provider_failure"
    UNSUPPORTED_PRICE_BASIS = "unsupported_price_basis"
    UNEXECUTABLE_ENTRY = "unexecutable_entry"
    EVALUATED_OK = "evaluated_ok"


class ReturnType(str, Enum):
    """V-01 return computation type."""

    PRICE_RETURN = "price_return"
    TOTAL_RETURN = "total_return"


# ---------------------------------------------------------------------------
# Profile Constants & Max-Roll Mapping
# ---------------------------------------------------------------------------

HORIZON_PROFILE_ID_V1: str = "horizon_profile_v1"

# Primary eval offsets read directly from canonical HORIZON_PROFILE_V1 truth source
PRIMARY_EVAL_OFFSET_SHORT: int = int(HORIZON_PROFILE_V1[HORIZON_SHORT]["primary_eval_offset"])
PRIMARY_EVAL_OFFSET_MEDIUM: int = int(HORIZON_PROFILE_V1[HORIZON_MEDIUM]["primary_eval_offset"])

# Immutable max-roll mapping (short: 2 days, medium: 5 days)
HORIZON_MAX_ROLL_DAYS: Mapping[str, int] = {
    HORIZON_SHORT: 2,
    HORIZON_MEDIUM: 5,
}

# Module-level consistency assertions against canonical HORIZON_PROFILE_V1
assert PRIMARY_EVAL_OFFSET_SHORT == 10, "HORIZON_PROFILE_V1 short offset drift detected"
assert PRIMARY_EVAL_OFFSET_MEDIUM == 40, "HORIZON_PROFILE_V1 medium offset drift detected"
assert set(HORIZON_MAX_ROLL_DAYS.keys()) == set(SUPPORTED_HORIZONS), (
    "HORIZON_MAX_ROLL_DAYS keys must match SUPPORTED_HORIZONS"
)


# ---------------------------------------------------------------------------
# Result TypedDict Contract (V-01 Shared Contract)
# ---------------------------------------------------------------------------


class HorizonReturnResult(TypedDict):
    """Full-pipeline return label result contract.

    Defined in V-01-1 as a pure type contract only; values are computed and populated in V-01-2+.
    """

    symbol: str
    horizon: str
    profile_id: str
    price_basis: str
    return_type: str
    entry_date: str
    executable_entry_date: Optional[str]
    target_calendar_date: Optional[str]
    actual_exit_date: Optional[str]
    roll_days_used: int
    entry_signal_price: Optional[float]
    entry_executable_price: Optional[float]
    entry_price: Optional[float]
    exit_price: Optional[float]
    cash_dividend_total: float
    split_ratio_total: float
    return_pct: Optional[float]
    outcome_status: str
    is_direction_hit: Optional[bool]
    evaluation_eligible: bool


# ---------------------------------------------------------------------------
# Exceptions & Calendar Window Dataclass
# ---------------------------------------------------------------------------


class InsufficientTradingCalendarError(ValueError):
    """Raised when trading_days sequence does not sufficiently cover T+N+max_roll."""

    pass


@dataclass(frozen=True)
class HorizonCalendarWindow:
    """Pure calendar window contract for multi-horizon analysis.

    Does not compute prices or returns.
    Does not output OutcomeStatus.EVALUATED_OK and does not claim evaluation.
    Only outputs strict boolean is_due.
    """

    signal_date: str
    horizon: str
    eval_offset: int
    max_roll_days: int
    executable_entry_date: str
    target_calendar_date: str
    roll_candidate_dates: Tuple[str, ...]
    holding_trading_days: Tuple[str, ...]
    is_due: bool


# ---------------------------------------------------------------------------
# Validation Helpers & Pure Resolution Function
# ---------------------------------------------------------------------------

_ISO_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_iso_date(d: Any, param_name: str) -> str:
    """Validate that d is a strict ISO date string YYYY-MM-DD representing a real calendar date."""
    if not isinstance(d, str):
        raise ValueError(f"{param_name} must be a string, got {type(d).__name__}: {d!r}")
    if not _ISO_DATE_REGEX.match(d):
        raise ValueError(f"{param_name} must be in strict ISO format YYYY-MM-DD, got {d!r}")
    try:
        parts = d.split("-")
        date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, TypeError) as e:
        raise ValueError(f"{param_name} is not a valid calendar date ({d!r}): {e}") from e
    return d


def resolve_horizon_calendar_window(
    *,
    signal_date: str,
    horizon: str,
    trading_days: Sequence[str],
    as_of: str,
    as_of_market_closed: bool = True,
) -> HorizonCalendarWindow:
    """Resolve a multi-horizon trading calendar window based strictly on an explicit trading days sequence.

    Invariants:
    1. Zero network requests, zero provider calls.
    2. Zero side effects: does not mutate, sort, or deduplicate the input trading_days sequence.
    3. Strict input validation: non-string Sequence, strictly sorted, duplicate-free real ISO dates.
    4. Exact signal_date presence: no bisect backwards/forwards shifting.
    5. Fails closed with ValueError (or InsufficientTradingCalendarError) if calendar coverage is less than T+N+max_roll.
    6. Does not compute prices or returns; does not output OutcomeStatus.EVALUATED_OK.
    """
    # 1. Validate horizon
    if not isinstance(horizon, str) or horizon not in SUPPORTED_HORIZONS:
        raise ValueError(
            f"Unsupported horizon {horizon!r}. Supported horizons: {list(SUPPORTED_HORIZONS)}"
        )

    # 2. Validate signal_date
    _validate_iso_date(signal_date, "signal_date")

    # 3. Validate as_of
    _validate_iso_date(as_of, "as_of")

    # 4. Validate as_of_market_closed: must be strict bool
    if not isinstance(as_of_market_closed, bool):
        raise ValueError(
            f"as_of_market_closed must be a strict bool, got {type(as_of_market_closed).__name__}: {as_of_market_closed!r}"
        )

    # 5. Validate trading_days
    if isinstance(trading_days, (str, bytes)):
        raise ValueError("trading_days must be a non-string Sequence, got string/bytes")
    if not isinstance(trading_days, Sequence):
        raise ValueError(f"trading_days must be a Sequence, got {type(trading_days).__name__}")
    if len(trading_days) == 0:
        raise ValueError("trading_days sequence cannot be empty")

    # Validate elements, ordering, and uniqueness without modifying input
    prev_day: Optional[str] = None
    for i, day in enumerate(trading_days):
        _validate_iso_date(day, f"trading_days[{i}]")
        if prev_day is not None:
            if day == prev_day:
                raise ValueError(f"trading_days contains duplicate date at index {i}: {day!r}")
            if day < prev_day:
                raise ValueError(
                    f"trading_days is not strictly sorted in ascending order at index {i}: {prev_day!r} >= {day!r}"
                )
        prev_day = day

    # 6. Locate signal_date (exact match, no bisect drifting)
    try:
        signal_idx = trading_days.index(signal_date)
    except ValueError:
        raise ValueError(f"signal_date {signal_date!r} not found in trading_days")

    # 7. Eval offset & max_roll_days
    eval_offset = int(HORIZON_PROFILE_V1[horizon]["primary_eval_offset"])
    max_roll_days = HORIZON_MAX_ROLL_DAYS[horizon]

    # 8. Calendar coverage check (must cover up to T+N+max_roll)
    required_index = signal_idx + eval_offset + max_roll_days
    if len(trading_days) <= required_index:
        raise InsufficientTradingCalendarError(
            f"trading_days does not cover up to T+{eval_offset}+{max_roll_days}: "
            f"needs index {required_index}, but length is {len(trading_days)} "
            f"(signal_idx={signal_idx}, offset={eval_offset}, max_roll={max_roll_days})"
        )

    # 9. Extract window components
    executable_entry_date = trading_days[signal_idx + 1]
    target_calendar_date = trading_days[signal_idx + eval_offset]
    holding_trading_days = tuple(trading_days[signal_idx + 1 : signal_idx + eval_offset + 1])
    roll_candidate_dates = tuple(
        trading_days[signal_idx + eval_offset + 1 : signal_idx + eval_offset + max_roll_days + 1]
    )

    # 10. Due calculation
    if target_calendar_date < as_of:
        is_due = True
    elif target_calendar_date == as_of:
        is_due = (as_of_market_closed is True)
    else:
        is_due = False

    return HorizonCalendarWindow(
        signal_date=signal_date,
        horizon=horizon,
        eval_offset=eval_offset,
        max_roll_days=max_roll_days,
        executable_entry_date=executable_entry_date,
        target_calendar_date=target_calendar_date,
        roll_candidate_dates=roll_candidate_dates,
        holding_trading_days=holding_trading_days,
        is_due=is_due,
    )
