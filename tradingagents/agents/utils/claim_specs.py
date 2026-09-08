"""Pure contract definitions and validators for claim applicability and invalidation specs (E-03a).

This module implements the minimal pure data contract for proposition applicability
and invalidation conditions (DAV-736 / DAV-742 Path B / DAV-746).

Future Reserved:
    These specifications are future-reserved for E-03b / E-03c.
    This module has ZERO producers, ZERO consumers, ZERO execution semantics,
    and ZERO weight / voting / probability semantics.
    Run horizon and claim applicability horizon must not overwrite each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
import math
import re
from typing import Any, Mapping, Sequence

# Module metadata / future reserved markers
FUTURE_RESERVED_FOR_E03B_E03C: bool = True
FUTURE_RESERVED_DISCLAIMER: str = (
    "Pure contract definitions for E-03a. Zero producers, zero consumers, "
    "zero execution semantics, zero weight/vote/probability semantics. "
    "Run horizon and claim applicability horizon must not overwrite each other."
)


# =====================================================================
# Frozen Enums (DAV-736 / DAV-742 / DAV-746)
# =====================================================================

class ApplicabilityHorizon(str, Enum):
    """Permitted horizons for claim applicability (strictly short or medium)."""
    SHORT = "short"
    MEDIUM = "medium"


class MetricBasis(str, Enum):
    """Permitted metric bases / calculation pipelines."""
    VENDOR_QFQ = "vendor_qfq"
    RAW = "raw"
    DIVIDEND_ADJUSTED = "dividend_adjusted"
    FINANCIAL_STATEMENT = "financial_statement"
    FUND_FLOW_SCALED = "fund_flow_scaled"
    INDEX_RELATIVE = "index_relative"


class ConditionOperator(str, Enum):
    """Permitted binary comparison operators for machine-checked invalidation."""
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="


class ConditionUnit(str, Enum):
    """Permitted metric units for invalidation thresholds."""
    PCT = "pct"
    CNY = "cny"
    RATIO = "ratio"
    SHARES = "shares"
    DAYS = "days"


class ConditionPeriod(str, Enum):
    """Permitted observation windows/periods for invalidation metrics."""
    CLOSE_1D = "1d_close"
    INTRADAY_TOUCH = "intraday_touch"
    CUMULATIVE_3D = "3d_cumulative"
    MA_5D = "5d_ma"
    QUARTERLY = "quarterly"


class ConditionSource(str, Enum):
    """Permitted data ingestion sources for invalidation metrics."""
    DAILY_PRICE = "daily_price"
    SMART_MONEY = "smart_money"
    FINANCIAL_REPORT = "financial_report"
    TRADE_CALENDAR = "trade_calendar"


# Frozen sets for fast O(1) membership testing
FROZEN_APPLICABILITY_HORIZONS: frozenset[str] = frozenset(
    item.value for item in ApplicabilityHorizon
)
FROZEN_METRIC_BASES: frozenset[str] = frozenset(
    item.value for item in MetricBasis
)
FROZEN_CONDITION_OPERATORS: frozenset[str] = frozenset(
    item.value for item in ConditionOperator
)
FROZEN_CONDITION_UNITS: frozenset[str] = frozenset(
    item.value for item in ConditionUnit
)
FROZEN_CONDITION_PERIODS: frozenset[str] = frozenset(
    item.value for item in ConditionPeriod
)
FROZEN_CONDITION_SOURCES: frozenset[str] = frozenset(
    item.value for item in ConditionSource
)


# =====================================================================
# Stable Error Codes (DAV-736 / DAV-742 / DAV-746)
# =====================================================================

class ClaimSpecErrorCode(str, Enum):
    # Applicability errors
    ERR_SPEC_MISSING_CLAIM = "ERR_SPEC_MISSING_CLAIM"
    ERR_SPEC_MISSING_APPLICABILITY = "ERR_SPEC_MISSING_APPLICABILITY"
    ERR_SPEC_MISSING_SYMBOL = "ERR_SPEC_MISSING_SYMBOL"
    ERR_SPEC_INVALID_SYMBOL = "ERR_SPEC_INVALID_SYMBOL"
    ERR_SPEC_SYMBOL_MISMATCH = "ERR_SPEC_SYMBOL_MISMATCH"
    ERR_SPEC_MISSING_HORIZON = "ERR_SPEC_MISSING_HORIZON"
    ERR_SPEC_INVALID_HORIZON = "ERR_SPEC_INVALID_HORIZON"
    ERR_SPEC_MISSING_METRIC_BASIS = "ERR_SPEC_MISSING_METRIC_BASIS"
    ERR_SPEC_INVALID_METRIC_BASIS = "ERR_SPEC_INVALID_METRIC_BASIS"
    ERR_SPEC_MISSING_PIT_DATE = "ERR_SPEC_MISSING_PIT_DATE"
    ERR_SPEC_INVALID_CALENDAR_DATE = "ERR_SPEC_INVALID_CALENDAR_DATE"
    ERR_SPEC_LOOKAHEAD_PIT = "ERR_SPEC_LOOKAHEAD_PIT"
    ERR_SPEC_INVALID_PRECONDITIONS = "ERR_SPEC_INVALID_PRECONDITIONS"

    # Invalidation condition errors
    ERR_SPEC_MISSING_INVALIDATION_CONDITIONS = "ERR_SPEC_MISSING_INVALIDATION_CONDITIONS"
    ERR_SPEC_EMPTY_INVALIDATION_CONDITIONS = "ERR_SPEC_EMPTY_INVALIDATION_CONDITIONS"
    ERR_SPEC_INVALID_CONDITION_STRUCTURE = "ERR_SPEC_INVALID_CONDITION_STRUCTURE"
    ERR_SPEC_MISSING_CONDITION_ID = "ERR_SPEC_MISSING_CONDITION_ID"
    ERR_SPEC_INVALID_CONDITION_ID = "ERR_SPEC_INVALID_CONDITION_ID"
    ERR_SPEC_DUPLICATE_CONDITION_ID = "ERR_SPEC_DUPLICATE_CONDITION_ID"
    ERR_SPEC_MISSING_CONDITION_FIELDS = "ERR_SPEC_MISSING_CONDITION_FIELDS"
    ERR_SPEC_INVALID_METRIC = "ERR_SPEC_INVALID_METRIC"
    ERR_SPEC_MISSING_OPERATOR = "ERR_SPEC_MISSING_OPERATOR"
    ERR_SPEC_INVALID_OPERATOR = "ERR_SPEC_INVALID_OPERATOR"
    ERR_SPEC_MISSING_THRESHOLD = "ERR_SPEC_MISSING_THRESHOLD"
    ERR_SPEC_INVALID_THRESHOLD = "ERR_SPEC_INVALID_THRESHOLD"
    ERR_SPEC_MISSING_UNIT = "ERR_SPEC_MISSING_UNIT"
    ERR_SPEC_INVALID_UNIT = "ERR_SPEC_INVALID_UNIT"
    ERR_SPEC_MISSING_PERIOD = "ERR_SPEC_MISSING_PERIOD"
    ERR_SPEC_INVALID_PERIOD = "ERR_SPEC_INVALID_PERIOD"
    ERR_SPEC_INVALID_SOURCE = "ERR_SPEC_INVALID_SOURCE"

    # Contract level errors
    ERR_SPEC_INVALID_CLAIM_ID = "ERR_SPEC_INVALID_CLAIM_ID"


# Module-level string constants for direct equality / membership
ERR_SPEC_MISSING_CLAIM: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_CLAIM.value
ERR_SPEC_MISSING_APPLICABILITY: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_APPLICABILITY.value
ERR_SPEC_MISSING_SYMBOL: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_SYMBOL.value
ERR_SPEC_INVALID_SYMBOL: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_SYMBOL.value
ERR_SPEC_SYMBOL_MISMATCH: str = ClaimSpecErrorCode.ERR_SPEC_SYMBOL_MISMATCH.value
ERR_SPEC_MISSING_HORIZON: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_HORIZON.value
ERR_SPEC_INVALID_HORIZON: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_HORIZON.value
ERR_SPEC_MISSING_METRIC_BASIS: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_METRIC_BASIS.value
ERR_SPEC_INVALID_METRIC_BASIS: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_METRIC_BASIS.value
ERR_SPEC_MISSING_PIT_DATE: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_PIT_DATE.value
ERR_SPEC_INVALID_CALENDAR_DATE: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_CALENDAR_DATE.value
ERR_SPEC_LOOKAHEAD_PIT: str = ClaimSpecErrorCode.ERR_SPEC_LOOKAHEAD_PIT.value
ERR_SPEC_INVALID_PRECONDITIONS: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_PRECONDITIONS.value

ERR_SPEC_MISSING_INVALIDATION_CONDITIONS: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_INVALIDATION_CONDITIONS.value
ERR_SPEC_EMPTY_INVALIDATION_CONDITIONS: str = ClaimSpecErrorCode.ERR_SPEC_EMPTY_INVALIDATION_CONDITIONS.value
ERR_SPEC_INVALID_CONDITION_STRUCTURE: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_CONDITION_STRUCTURE.value
ERR_SPEC_MISSING_CONDITION_ID: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_CONDITION_ID.value
ERR_SPEC_INVALID_CONDITION_ID: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_CONDITION_ID.value
ERR_SPEC_DUPLICATE_CONDITION_ID: str = ClaimSpecErrorCode.ERR_SPEC_DUPLICATE_CONDITION_ID.value
ERR_SPEC_MISSING_CONDITION_FIELDS: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_CONDITION_FIELDS.value
ERR_SPEC_INVALID_METRIC: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_METRIC.value
ERR_SPEC_MISSING_OPERATOR: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_OPERATOR.value
ERR_SPEC_INVALID_OPERATOR: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_OPERATOR.value
ERR_SPEC_MISSING_THRESHOLD: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_THRESHOLD.value
ERR_SPEC_INVALID_THRESHOLD: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_THRESHOLD.value
ERR_SPEC_MISSING_UNIT: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_UNIT.value
ERR_SPEC_INVALID_UNIT: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_UNIT.value
ERR_SPEC_MISSING_PERIOD: str = ClaimSpecErrorCode.ERR_SPEC_MISSING_PERIOD.value
ERR_SPEC_INVALID_PERIOD: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_PERIOD.value
ERR_SPEC_INVALID_SOURCE: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_SOURCE.value
ERR_SPEC_INVALID_CLAIM_ID: str = ClaimSpecErrorCode.ERR_SPEC_INVALID_CLAIM_ID.value

# Backward-compatible short aliases matching DAV-736 text
MISSING_SYMBOL: str = ERR_SPEC_MISSING_SYMBOL
SYMBOL_MISMATCH: str = ERR_SPEC_SYMBOL_MISMATCH
MISSING_PIT_DATE: str = ERR_SPEC_MISSING_PIT_DATE
LOOKAHEAD_PIT_DATE: str = ERR_SPEC_LOOKAHEAD_PIT
INVALID_HORIZON: str = ERR_SPEC_INVALID_HORIZON
INVALID_METRIC_BASIS: str = ERR_SPEC_INVALID_METRIC_BASIS
EMPTY_INVALIDATION_CONDITIONS: str = ERR_SPEC_EMPTY_INVALIDATION_CONDITIONS
INVALID_OPERATOR: str = ERR_SPEC_INVALID_OPERATOR
MISSING_CONDITION_FIELDS: str = ERR_SPEC_MISSING_CONDITION_FIELDS
MISSING_UNIT: str = ERR_SPEC_MISSING_UNIT
INVALID_UNIT: str = ERR_SPEC_INVALID_UNIT
MISSING_PERIOD: str = ERR_SPEC_MISSING_PERIOD
INVALID_PERIOD: str = ERR_SPEC_INVALID_PERIOD


# =====================================================================
# Pure Dataclasses (Modular Spec Dataclasses)
# =====================================================================

@dataclass(frozen=True)
class ClaimApplicability:
    """Applicability boundary specifying where and when a claim holds."""
    symbol: str
    horizon: ApplicabilityHorizon
    metric_basis: MetricBasis
    pit_date: str
    preconditions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, ApplicabilityHorizon):
            raise ValueError(
                f"horizon must be an instance of ApplicabilityHorizon, got {type(self.horizon).__name__}"
            )
        if not isinstance(self.metric_basis, MetricBasis):
            raise ValueError(
                f"metric_basis must be an instance of MetricBasis, got {type(self.metric_basis).__name__}"
            )
        ok, err, norm = validate_applicability(self.to_dict())
        if not ok:
            raise ValueError(f"Invalid applicability dataclass: {err}")
        object.__setattr__(self, "symbol", norm["symbol"])
        object.__setattr__(self, "horizon", ApplicabilityHorizon(norm["horizon"]))
        object.__setattr__(self, "metric_basis", MetricBasis(norm["metric_basis"]))
        object.__setattr__(self, "pit_date", norm["pit_date"])
        object.__setattr__(self, "preconditions", tuple(norm["preconditions"]))

    def to_dict(self) -> dict[str, Any]:
        preconds: Any
        if isinstance(self.preconditions, (list, tuple)):
            preconds = list(self.preconditions)
        else:
            preconds = self.preconditions
        return {
            "symbol": self.symbol,
            "horizon": self.horizon.value if isinstance(self.horizon, Enum) else self.horizon,
            "metric_basis": self.metric_basis.value if isinstance(self.metric_basis, Enum) else self.metric_basis,
            "pit_date": self.pit_date,
            "preconditions": preconds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClaimApplicability:
        ok, err, norm = validate_applicability(data)
        if not ok:
            raise ValueError(f"Invalid applicability specification: {err}")
        return cls(
            symbol=norm["symbol"],
            horizon=ApplicabilityHorizon(norm["horizon"]),
            metric_basis=MetricBasis(norm["metric_basis"]),
            pit_date=norm["pit_date"],
            preconditions=tuple(norm["preconditions"]),
        )


@dataclass(frozen=True)
class ClaimInvalidationCondition:
    """Machine-checkable invalidation condition for prospective falsification."""
    condition_id: str
    metric: str
    operator: ConditionOperator
    threshold: float | int
    unit: ConditionUnit
    period: ConditionPeriod
    source: ConditionSource
    pit_date: str

    def __post_init__(self) -> None:
        if not isinstance(self.operator, ConditionOperator):
            raise ValueError(
                f"operator must be an instance of ConditionOperator, got {type(self.operator).__name__}"
            )
        if not isinstance(self.unit, ConditionUnit):
            raise ValueError(
                f"unit must be an instance of ConditionUnit, got {type(self.unit).__name__}"
            )
        if not isinstance(self.period, ConditionPeriod):
            raise ValueError(
                f"period must be an instance of ConditionPeriod, got {type(self.period).__name__}"
            )
        if not isinstance(self.source, ConditionSource):
            raise ValueError(
                f"source must be an instance of ConditionSource, got {type(self.source).__name__}"
            )
        ok, err, norm = validate_invalidation_condition(self.to_dict())
        if not ok:
            raise ValueError(f"Invalid condition dataclass: {err}")
        object.__setattr__(self, "condition_id", norm["condition_id"])
        object.__setattr__(self, "metric", norm["metric"])
        object.__setattr__(self, "operator", ConditionOperator(norm["operator"]))
        object.__setattr__(self, "threshold", norm["threshold"])
        object.__setattr__(self, "unit", ConditionUnit(norm["unit"]))
        object.__setattr__(self, "period", ConditionPeriod(norm["period"]))
        object.__setattr__(self, "source", ConditionSource(norm["source"]))
        object.__setattr__(self, "pit_date", norm["pit_date"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "metric": self.metric,
            "operator": self.operator.value if isinstance(self.operator, Enum) else self.operator,
            "threshold": self.threshold,
            "unit": self.unit.value if isinstance(self.unit, Enum) else self.unit,
            "period": self.period.value if isinstance(self.period, Enum) else self.period,
            "source": self.source.value if isinstance(self.source, Enum) else self.source,
            "pit_date": self.pit_date,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClaimInvalidationCondition:
        ok, err, norm = validate_invalidation_condition(data)
        if not ok:
            raise ValueError(f"Invalid invalidation condition specification: {err}")
        return cls(
            condition_id=norm["condition_id"],
            metric=norm["metric"],
            operator=ConditionOperator(norm["operator"]),
            threshold=norm["threshold"],
            unit=ConditionUnit(norm["unit"]),
            period=ConditionPeriod(norm["period"]),
            source=ConditionSource(norm["source"]),
            pit_date=norm["pit_date"],
        )


@dataclass(frozen=True)
class ClaimReviewContract:
    """Container wrapping applicability and invalidation conditions for review."""
    applicability: ClaimApplicability
    invalidation_conditions: tuple[ClaimInvalidationCondition, ...]
    claim_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.applicability, ClaimApplicability):
            raise ValueError(f"applicability must be an instance of ClaimApplicability, got {type(self.applicability).__name__}")
        if not isinstance(self.invalidation_conditions, (list, tuple)):
            raise ValueError(f"invalidation_conditions must be a tuple or list, got {type(self.invalidation_conditions).__name__}")
        for idx, cond in enumerate(self.invalidation_conditions):
            if not isinstance(cond, ClaimInvalidationCondition):
                raise ValueError(f"invalidation_conditions item at index {idx} must be a ClaimInvalidationCondition, got {type(cond).__name__}")
        object.__setattr__(self, "invalidation_conditions", tuple(self.invalidation_conditions))
        if self.claim_id is not None:
            if not isinstance(self.claim_id, str) or not self.claim_id.strip():
                raise ValueError("claim_id must be a non-empty string when provided")
            object.__setattr__(self, "claim_id", self.claim_id.strip())
        ok, errors, _ = validate_claim_review_contract(self.to_dict())
        if not ok:
            raise ValueError(f"Invalid claim review contract: {errors}")

    def to_dict(self) -> dict[str, Any]:
        app_dict = (
            self.applicability.to_dict()
            if hasattr(self.applicability, "to_dict")
            else self.applicability
        )
        cond_dicts: Any
        if isinstance(self.invalidation_conditions, (list, tuple)):
            cond_dicts = [
                c.to_dict() if hasattr(c, "to_dict") else c
                for c in self.invalidation_conditions
            ]
        else:
            cond_dicts = self.invalidation_conditions

        res: dict[str, Any] = {
            "applicability": app_dict,
            "invalidation_conditions": cond_dicts,
        }
        if self.claim_id is not None:
            res["claim_id"] = self.claim_id
        return res

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        expected_symbol: str | None = None,
        current_trade_date: str | None = None,
    ) -> ClaimReviewContract:
        ok, errors, norm = validate_claim_review_contract(
            data,
            expected_symbol=expected_symbol,
            current_trade_date=current_trade_date,
            strict_fail_closed=True,
        )
        if not ok:
            raise ValueError(f"Invalid claim review contract: {errors}")
        app = ClaimApplicability(
            symbol=norm["applicability"]["symbol"],
            horizon=ApplicabilityHorizon(norm["applicability"]["horizon"]),
            metric_basis=MetricBasis(norm["applicability"]["metric_basis"]),
            pit_date=norm["applicability"]["pit_date"],
            preconditions=tuple(norm["applicability"]["preconditions"]),
        )
        conds = tuple(
            ClaimInvalidationCondition(
                condition_id=c["condition_id"],
                metric=c["metric"],
                operator=ConditionOperator(c["operator"]),
                threshold=c["threshold"],
                unit=ConditionUnit(c["unit"]),
                period=ConditionPeriod(c["period"]),
                source=ConditionSource(c["source"]),
                pit_date=c["pit_date"],
            )
            for c in norm["invalidation_conditions"]
        )
        return cls(
            applicability=app,
            invalidation_conditions=conds,
            claim_id=norm.get("claim_id"),
        )


# =====================================================================
# Internal Helpers
# =====================================================================

def _validate_iso_calendar_date(val: Any) -> tuple[bool, str | None, str | None]:
    """Validate that val is a real ISO YYYY-MM-DD calendar date.

    Returns:
        (is_valid, error_code, normalized_date_str)
    """
    if not isinstance(val, str):
        return False, ERR_SPEC_INVALID_CALENDAR_DATE, None
    raw = val.strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return False, ERR_SPEC_INVALID_CALENDAR_DATE, None
    try:
        dt = datetime.strptime(raw, "%Y-%m-%d").date()
        if dt.strftime("%Y-%m-%d") != raw:
            return False, ERR_SPEC_INVALID_CALENDAR_DATE, None
        return True, None, raw
    except ValueError:
        return False, ERR_SPEC_INVALID_CALENDAR_DATE, None


# =====================================================================
# Pure Validation Functions (DAV-742 / DAV-746 Signatures)
# =====================================================================

def validate_applicability(
    applicability: Mapping[str, Any] | None,
    *,
    expected_symbol: str | None = None,
    current_trade_date: str | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Validate claim applicability specification.

    Args:
        applicability: Mapping containing symbol, horizon, metric_basis, pit_date, preconditions.
        expected_symbol: Optional expected stock symbol (e.g. current ticker context).
        current_trade_date: Optional current trading date cutoff (PIT ceiling).

    Returns:
        (is_valid, error_code, normalized_dict)
    """
    if applicability is None or not isinstance(applicability, Mapping):
        return False, ERR_SPEC_MISSING_APPLICABILITY, {}

    # 1. symbol: strictly 6 digits string
    if "symbol" not in applicability or applicability["symbol"] is None:
        return False, ERR_SPEC_MISSING_SYMBOL, {}
    raw_symbol = applicability["symbol"]
    if not isinstance(raw_symbol, str):
        return False, ERR_SPEC_INVALID_SYMBOL, {}
    symbol_str = raw_symbol.strip()
    if not re.match(r"^\d{6}$", symbol_str):
        return False, ERR_SPEC_INVALID_SYMBOL, {}
    if expected_symbol is not None:
        exp_clean = str(expected_symbol).strip()
        if symbol_str != exp_clean:
            return False, ERR_SPEC_SYMBOL_MISMATCH, {}

    # 2. horizon: strictly short or medium
    if "horizon" not in applicability or applicability["horizon"] is None:
        return False, ERR_SPEC_MISSING_HORIZON, {}
    raw_horizon = applicability["horizon"]
    horizon_str = raw_horizon.value if isinstance(raw_horizon, Enum) else str(raw_horizon).strip()
    if horizon_str not in FROZEN_APPLICABILITY_HORIZONS:
        return False, ERR_SPEC_INVALID_HORIZON, {}

    # 3. metric_basis: frozen enum
    if "metric_basis" not in applicability or applicability["metric_basis"] is None:
        return False, ERR_SPEC_MISSING_METRIC_BASIS, {}
    raw_mb = applicability["metric_basis"]
    mb_str = raw_mb.value if isinstance(raw_mb, Enum) else str(raw_mb).strip()
    if mb_str not in FROZEN_METRIC_BASES:
        return False, ERR_SPEC_INVALID_METRIC_BASIS, {}

    # 4. pit_date: real ISO calendar date <= current_trade_date
    if "pit_date" not in applicability or applicability["pit_date"] is None:
        return False, ERR_SPEC_MISSING_PIT_DATE, {}
    ok_date, err_date, norm_pit = _validate_iso_calendar_date(applicability["pit_date"])
    if not ok_date:
        return False, err_date, {}
    if current_trade_date is not None:
        ok_curr, err_curr, norm_curr = _validate_iso_calendar_date(current_trade_date)
        if not ok_curr:
            return False, err_curr, {}
        if norm_pit > norm_curr:
            return False, ERR_SPEC_LOOKAHEAD_PIT, {}

    # 5. preconditions: sequence of non-empty strings (no single string, no empty elements)
    normalized_preconditions: list[str] = []
    if "preconditions" in applicability:
        raw_pre = applicability["preconditions"]
        if raw_pre is None:
            return False, ERR_SPEC_INVALID_PRECONDITIONS, {}
        # Single string / bytes disguised as sequence must be explicitly rejected
        if isinstance(raw_pre, (str, bytes)):
            return False, ERR_SPEC_INVALID_PRECONDITIONS, {}
        if isinstance(raw_pre, (bool, int, float, dict)):
            return False, ERR_SPEC_INVALID_PRECONDITIONS, {}
        if not isinstance(raw_pre, (list, tuple, Sequence)):
            return False, ERR_SPEC_INVALID_PRECONDITIONS, {}
        for item in raw_pre:
            if not isinstance(item, str):
                return False, ERR_SPEC_INVALID_PRECONDITIONS, {}
            item_clean = item.strip()
            if not item_clean:
                return False, ERR_SPEC_INVALID_PRECONDITIONS, {}
            normalized_preconditions.append(item_clean)

    normalized: dict[str, Any] = {
        "symbol": symbol_str,
        "horizon": horizon_str,
        "metric_basis": mb_str,
        "pit_date": norm_pit,
        "preconditions": normalized_preconditions,
    }
    return True, None, normalized


def validate_invalidation_condition(
    condition: Mapping[str, Any] | None,
    *,
    index: int = 1,
    current_trade_date: str | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Validate a single invalidation condition specification.

    Args:
        condition: Mapping containing condition_id, metric, operator, threshold,
                   unit, period, source, pit_date.
        index: Condition index for diagnostic reporting.
        current_trade_date: Optional current trading date cutoff (PIT ceiling).

    Returns:
        (is_valid, error_code, normalized_dict)
    """
    if condition is None or not isinstance(condition, Mapping):
        return False, ERR_SPEC_INVALID_CONDITION_STRUCTURE, {}

    # 1. condition_id: non-empty string
    if "condition_id" not in condition or condition["condition_id"] is None:
        return False, ERR_SPEC_MISSING_CONDITION_ID, {}
    raw_cid = condition["condition_id"]
    if not isinstance(raw_cid, str) or not raw_cid.strip():
        return False, ERR_SPEC_INVALID_CONDITION_ID, {}
    cid_str = raw_cid.strip()

    # 2. metric: non-empty string
    if "metric" not in condition or condition["metric"] is None:
        return False, ERR_SPEC_MISSING_CONDITION_FIELDS, {}
    raw_metric = condition["metric"]
    if not isinstance(raw_metric, str) or not raw_metric.strip():
        return False, ERR_SPEC_INVALID_METRIC, {}
    metric_str = raw_metric.strip()

    # 3. operator: strictly in {<, <=, >, >=, ==, !=}
    if "operator" not in condition or condition["operator"] is None:
        return False, ERR_SPEC_MISSING_OPERATOR, {}
    raw_op = condition["operator"]
    op_str = raw_op.value if isinstance(raw_op, Enum) else str(raw_op).strip()
    if op_str not in FROZEN_CONDITION_OPERATORS:
        return False, ERR_SPEC_INVALID_OPERATOR, {}

    # 4. threshold: finite int / float / Decimal; explicitly reject bool, str, NaN, Inf
    if "threshold" not in condition or condition["threshold"] is None:
        return False, ERR_SPEC_MISSING_THRESHOLD, {}
    raw_th = condition["threshold"]
    if isinstance(raw_th, bool):
        return False, ERR_SPEC_INVALID_THRESHOLD, {}
    if isinstance(raw_th, (str, bytes)):
        return False, ERR_SPEC_INVALID_THRESHOLD, {}
    if not isinstance(raw_th, (int, float, Decimal)):
        return False, ERR_SPEC_INVALID_THRESHOLD, {}

    if isinstance(raw_th, Decimal):
        if not raw_th.is_finite():
            return False, ERR_SPEC_INVALID_THRESHOLD, {}
        try:
            norm_th: float | int = float(raw_th)
        except (OverflowError, ValueError):
            return False, ERR_SPEC_INVALID_THRESHOLD, {}
        if not math.isfinite(norm_th):
            return False, ERR_SPEC_INVALID_THRESHOLD, {}
    elif isinstance(raw_th, float):
        if not math.isfinite(raw_th):
            return False, ERR_SPEC_INVALID_THRESHOLD, {}
        norm_th = raw_th
    elif isinstance(raw_th, int):
        norm_th = raw_th
    else:
        return False, ERR_SPEC_INVALID_THRESHOLD, {}

    # 5. unit: strictly in {pct, cny, ratio, shares, days}
    if "unit" not in condition or condition["unit"] is None:
        return False, ERR_SPEC_MISSING_UNIT, {}
    raw_unit = condition["unit"]
    unit_str = raw_unit.value if isinstance(raw_unit, Enum) else str(raw_unit).strip().lower()
    if unit_str not in FROZEN_CONDITION_UNITS:
        return False, ERR_SPEC_INVALID_UNIT, {}

    # 6. period: strictly in {1d_close, intraday_touch, 3d_cumulative, 5d_ma, quarterly}
    if "period" not in condition or condition["period"] is None:
        return False, ERR_SPEC_MISSING_PERIOD, {}
    raw_period = condition["period"]
    period_str = raw_period.value if isinstance(raw_period, Enum) else str(raw_period).strip().lower()
    if period_str not in FROZEN_CONDITION_PERIODS:
        return False, ERR_SPEC_INVALID_PERIOD, {}

    # 7. source: strictly in {daily_price, smart_money, financial_report, trade_calendar}
    if "source" not in condition or condition["source"] is None:
        return False, ERR_SPEC_MISSING_CONDITION_FIELDS, {}
    raw_source = condition["source"]
    source_str = raw_source.value if isinstance(raw_source, Enum) else str(raw_source).strip().lower()
    if source_str not in FROZEN_CONDITION_SOURCES:
        return False, ERR_SPEC_INVALID_SOURCE, {}

    # 8. pit_date: real ISO calendar date <= current_trade_date
    if "pit_date" not in condition or condition["pit_date"] is None:
        return False, ERR_SPEC_MISSING_PIT_DATE, {}
    ok_date, err_date, norm_cond_pit = _validate_iso_calendar_date(condition["pit_date"])
    if not ok_date:
        return False, err_date, {}
    if current_trade_date is not None:
        ok_curr, err_curr, norm_curr = _validate_iso_calendar_date(current_trade_date)
        if not ok_curr:
            return False, err_curr, {}
        if norm_cond_pit > norm_curr:
            return False, ERR_SPEC_LOOKAHEAD_PIT, {}

    normalized: dict[str, Any] = {
        "condition_id": cid_str,
        "metric": metric_str,
        "operator": op_str,
        "threshold": norm_th,
        "unit": unit_str,
        "period": period_str,
        "source": source_str,
        "pit_date": norm_cond_pit,
    }
    return True, None, normalized


def validate_claim_review_contract(
    claim_dict: Mapping[str, Any],
    *,
    expected_symbol: str | None = None,
    current_trade_date: str | None = None,
    strict_fail_closed: bool = True,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate a claim review contract containing applicability and invalidation conditions.

    Fail-closed invariant:
        - If any field is invalid or missing, is_valid is False and normalized is {}.
        - strict_fail_closed=False only alters error aggregation (collects all errors vs stops on first);
          it NEVER makes invalid data valid, and NEVER imputes probability or confidence.

    Args:
        claim_dict: Mapping with applicability and invalidation_conditions.
        expected_symbol: Optional expected stock symbol.
        current_trade_date: Optional current trading date cutoff.
        strict_fail_closed: If True, halts on the first encountered error code;
                            if False, aggregates all error codes across the structure.

    Returns:
        (is_valid, list_of_error_codes, normalized_dict)
    """
    errors: list[str] = []

    if claim_dict is None or not isinstance(claim_dict, Mapping):
        return False, [ERR_SPEC_MISSING_CLAIM], {}

    # A. Validate applicability
    norm_applicability: dict[str, Any] = {}
    if "applicability" not in claim_dict or claim_dict["applicability"] is None:
        errors.append(ERR_SPEC_MISSING_APPLICABILITY)
        if strict_fail_closed:
            return False, errors, {}
    else:
        app_input = claim_dict["applicability"]
        if isinstance(app_input, ClaimApplicability):
            app_input = app_input.to_dict()
        ok_app, err_app, norm_app = validate_applicability(
            app_input,
            expected_symbol=expected_symbol,
            current_trade_date=current_trade_date,
        )
        if not ok_app:
            if err_app:
                errors.append(err_app)
            if strict_fail_closed:
                return False, errors, {}
        else:
            norm_applicability = norm_app

    # B. Validate invalidation_conditions
    norm_conditions: list[dict[str, Any]] = []
    if "invalidation_conditions" not in claim_dict or claim_dict["invalidation_conditions"] is None:
        errors.append(ERR_SPEC_MISSING_INVALIDATION_CONDITIONS)
        if strict_fail_closed:
            return False, errors, {}
    else:
        conds_input = claim_dict["invalidation_conditions"]
        if isinstance(conds_input, (str, bytes, Mapping)) or not isinstance(conds_input, (list, tuple, Sequence)):
            errors.append(ERR_SPEC_INVALID_CONDITION_STRUCTURE)
            if strict_fail_closed:
                return False, errors, {}
        elif len(conds_input) == 0:
            errors.append(ERR_SPEC_EMPTY_INVALIDATION_CONDITIONS)
            if strict_fail_closed:
                return False, errors, {}
        else:
            # Anchor PIT for conditions: use explicit current_trade_date, or applicability's validated PIT
            effective_trade_date = current_trade_date or norm_applicability.get("pit_date")
            seen_condition_ids: set[str] = set()

            for idx, cond in enumerate(conds_input, start=1):
                if isinstance(cond, ClaimInvalidationCondition):
                    cond = cond.to_dict()
                ok_cond, err_cond, norm_c = validate_invalidation_condition(
                    cond,
                    index=idx,
                    current_trade_date=effective_trade_date,
                )
                if not ok_cond:
                    if err_cond:
                        errors.append(err_cond)
                    if strict_fail_closed:
                        return False, errors, {}
                else:
                    cid = norm_c["condition_id"]
                    if cid in seen_condition_ids:
                        errors.append(ERR_SPEC_DUPLICATE_CONDITION_ID)
                        if strict_fail_closed:
                            return False, errors, {}
                    seen_condition_ids.add(cid)
                    norm_conditions.append(norm_c)

    # C. Validate claim_id (if explicitly provided)
    norm_claim_id: str | None = None
    if "claim_id" in claim_dict:
        raw_cid = claim_dict["claim_id"]
        if isinstance(raw_cid, str) and not isinstance(raw_cid, bool) and raw_cid.strip():
            norm_claim_id = raw_cid.strip()
        else:
            errors.append(ERR_SPEC_INVALID_CLAIM_ID)
            if strict_fail_closed:
                return False, errors, {}

    if errors:
        return False, errors, {}

    normalized_contract: dict[str, Any] = {
        "applicability": norm_applicability,
        "invalidation_conditions": norm_conditions,
    }
    if norm_claim_id is not None:
        normalized_contract["claim_id"] = norm_claim_id

    return True, [], normalized_contract
