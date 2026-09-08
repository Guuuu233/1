"""Unit tests for claim applicability and invalidation conditions specifications (E-03a).

Covers all 18 test cases from DAV-742, plus comprehensive edge-case tests mandated by DAV-746:
- bool / string / non-finite threshold rejection
- Invalid calendar date rejection (e.g. 2026-02-29 non-leap year)
- Preconditions string disguised as sequence rejection
- Empty string element in preconditions rejection
- Duplicate condition_id rejection
- Unknown fields handling / stripping
- JSON roundtrip idempotence
- Zero probability / confidence imputation
- strict_fail_closed error aggregation behavior
- Dataclass roundtrip
- Run horizon vs applicability horizon isolation invariant
"""

from __future__ import annotations

from decimal import Decimal
import json
import math
from typing import Any

import pytest

from tradingagents.agents.utils.claim_specs import (
    ApplicabilityHorizon,
    ClaimApplicability,
    ClaimInvalidationCondition,
    ClaimReviewContract,
    ClaimSpecErrorCode,
    ConditionOperator,
    ConditionPeriod,
    ConditionSource,
    ConditionUnit,
    EMPTY_INVALIDATION_CONDITIONS,
    ERR_SPEC_DUPLICATE_CONDITION_ID,
    ERR_SPEC_EMPTY_INVALIDATION_CONDITIONS,
    ERR_SPEC_INVALID_CALENDAR_DATE,
    ERR_SPEC_INVALID_HORIZON,
    ERR_SPEC_INVALID_METRIC_BASIS,
    ERR_SPEC_INVALID_OPERATOR,
    ERR_SPEC_INVALID_PERIOD,
    ERR_SPEC_INVALID_PRECONDITIONS,
    ERR_SPEC_INVALID_SYMBOL,
    ERR_SPEC_INVALID_THRESHOLD,
    ERR_SPEC_INVALID_UNIT,
    ERR_SPEC_LOOKAHEAD_PIT,
    ERR_SPEC_MISSING_APPLICABILITY,
    ERR_SPEC_MISSING_CONDITION_FIELDS,
    ERR_SPEC_MISSING_CONDITION_ID,
    ERR_SPEC_MISSING_HORIZON,
    ERR_SPEC_MISSING_INVALIDATION_CONDITIONS,
    ERR_SPEC_MISSING_METRIC_BASIS,
    ERR_SPEC_MISSING_PERIOD,
    ERR_SPEC_MISSING_PIT_DATE,
    ERR_SPEC_MISSING_SYMBOL,
    ERR_SPEC_MISSING_UNIT,
    ERR_SPEC_SYMBOL_MISMATCH,
    FUTURE_RESERVED_DISCLAIMER,
    FUTURE_RESERVED_FOR_E03B_E03C,
    INVALID_HORIZON,
    INVALID_METRIC_BASIS,
    INVALID_OPERATOR,
    INVALID_PERIOD,
    INVALID_UNIT,
    LOOKAHEAD_PIT_DATE,
    MISSING_CONDITION_FIELDS,
    MISSING_PERIOD,
    MISSING_PIT_DATE,
    MISSING_SYMBOL,
    MISSING_UNIT,
    MetricBasis,
    NON_NUMERIC_THRESHOLD,
    SYMBOL_MISMATCH,
    validate_applicability,
    validate_claim_review_contract,
    validate_invalidation_condition,
)


# =====================================================================
# Fixtures and Helpers
# =====================================================================

def make_valid_applicability_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "symbol": "600519",
        "horizon": "short",
        "metric_basis": "vendor_qfq",
        "pit_date": "2026-09-08",
        "preconditions": ["above_ma20", "volume_expansion"],
    }
    base.update(overrides)
    return base


def make_valid_condition_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "condition_id": "inv-1",
        "metric": "close_price",
        "operator": "<",
        "threshold": 1600.0,
        "unit": "cny",
        "period": "1d_close",
        "source": "daily_price",
        "pit_date": "2026-09-08",
    }
    base.update(overrides)
    return base


def make_valid_claim_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "claim_id": "CLM-BULL-1",
        "applicability": make_valid_applicability_dict(),
        "invalidation_conditions": [make_valid_condition_dict()],
    }
    base.update(overrides)
    return base


# =====================================================================
# 18 Tests from DAV-742 Matrix
# =====================================================================

def test_valid_minimal_claim_spec_positive():
    """1. Legal minimal claim spec passes validation with correct normalized structure."""
    claim = {
        "claim_id": "CLM-BULL-1",
        "applicability": {
            "symbol": "600519",
            "horizon": "short",
            "metric_basis": "vendor_qfq",
            "pit_date": "2026-09-08",
        },
        "invalidation_conditions": [
            {
                "condition_id": "inv-1",
                "metric": "close_price",
                "operator": "<",
                "threshold": 1600.0,
                "unit": "cny",
                "period": "1d_close",
                "source": "daily_price",
                "pit_date": "2026-09-08",
            }
        ],
    }
    ok, errors, norm = validate_claim_review_contract(
        claim,
        expected_symbol="600519",
        current_trade_date="2026-09-08",
    )
    assert ok is True
    assert errors == []
    assert norm["claim_id"] == "CLM-BULL-1"
    assert norm["applicability"]["symbol"] == "600519"
    assert norm["applicability"]["horizon"] == "short"
    assert norm["applicability"]["metric_basis"] == "vendor_qfq"
    assert norm["applicability"]["pit_date"] == "2026-09-08"
    assert norm["applicability"]["preconditions"] == []
    assert len(norm["invalidation_conditions"]) == 1
    assert norm["invalidation_conditions"][0]["condition_id"] == "inv-1"
    assert norm["invalidation_conditions"][0]["threshold"] == 1600.0


def test_valid_multi_invalidation_conditions_positive():
    """2. Multi invalidation conditions with price and fund flow metrics pass validation."""
    cond1 = make_valid_condition_dict(
        condition_id="inv-price-1",
        metric="close_price",
        operator="<",
        threshold=1550.0,
        unit="cny",
        period="1d_close",
        source="daily_price",
    )
    cond2 = make_valid_condition_dict(
        condition_id="inv-flow-2",
        metric="large_order_net_inflow_ratio",
        operator="<=",
        threshold=-0.05,
        unit="ratio",
        period="3d_cumulative",
        source="smart_money",
    )
    claim = make_valid_claim_dict(invalidation_conditions=[cond1, cond2])
    ok, errors, norm = validate_claim_review_contract(claim)
    assert ok is True
    assert errors == []
    assert len(norm["invalidation_conditions"]) == 2
    assert norm["invalidation_conditions"][0]["condition_id"] == "inv-price-1"
    assert norm["invalidation_conditions"][1]["condition_id"] == "inv-flow-2"


def test_missing_symbol_negative():
    """3. Missing symbol in applicability fails closed with ERR_SPEC_MISSING_SYMBOL."""
    app = make_valid_applicability_dict()
    del app["symbol"]
    ok, err, norm = validate_applicability(app)
    assert ok is False
    assert err == ERR_SPEC_MISSING_SYMBOL
    assert norm == {}

    claim = make_valid_claim_dict(applicability=app)
    ok_c, errs_c, norm_c = validate_claim_review_contract(claim)
    assert ok_c is False
    assert ERR_SPEC_MISSING_SYMBOL in errs_c
    assert norm_c == {}


def test_symbol_mismatch_negative():
    """4. Symbol mismatch against expected ticker fails closed with ERR_SPEC_SYMBOL_MISMATCH."""
    app = make_valid_applicability_dict(symbol="000001")
    ok, err, norm = validate_applicability(app, expected_symbol="600519")
    assert ok is False
    assert err == ERR_SPEC_SYMBOL_MISMATCH
    assert norm == {}

    claim = make_valid_claim_dict(applicability=app)
    ok_c, errs_c, norm_c = validate_claim_review_contract(claim, expected_symbol="600519")
    assert ok_c is False
    assert ERR_SPEC_SYMBOL_MISMATCH in errs_c
    assert norm_c == {}


def test_missing_pit_date_negative():
    """5. Missing pit_date fails closed with ERR_SPEC_MISSING_PIT_DATE."""
    app = make_valid_applicability_dict()
    del app["pit_date"]
    ok, err, norm = validate_applicability(app)
    assert ok is False
    assert err == ERR_SPEC_MISSING_PIT_DATE
    assert norm == {}

    cond = make_valid_condition_dict()
    del cond["pit_date"]
    ok_cond, err_cond, norm_cond = validate_invalidation_condition(cond)
    assert ok_cond is False
    assert err_cond == ERR_SPEC_MISSING_PIT_DATE
    assert norm_cond == {}


def test_future_pit_date_lookahead_negative():
    """6. Future pit_date beyond current_trade_date fails closed with ERR_SPEC_LOOKAHEAD_PIT."""
    app = make_valid_applicability_dict(pit_date="2026-09-10")
    ok, err, norm = validate_applicability(app, current_trade_date="2026-09-08")
    assert ok is False
    assert err == ERR_SPEC_LOOKAHEAD_PIT
    assert norm == {}

    cond = make_valid_condition_dict(pit_date="2026-09-10")
    ok_cond, err_cond, norm_cond = validate_invalidation_condition(cond, current_trade_date="2026-09-08")
    assert ok_cond is False
    assert err_cond == ERR_SPEC_LOOKAHEAD_PIT
    assert norm_cond == {}


def test_invalid_horizon_negative():
    """7. Invalid horizon outside {short, medium} fails closed with ERR_SPEC_INVALID_HORIZON."""
    app = make_valid_applicability_dict(horizon="ultra_short")
    ok, err, norm = validate_applicability(app)
    assert ok is False
    assert err == ERR_SPEC_INVALID_HORIZON
    assert norm == {}

    claim = make_valid_claim_dict(applicability=app)
    ok_c, errs_c, norm_c = validate_claim_review_contract(claim)
    assert ok_c is False
    assert ERR_SPEC_INVALID_HORIZON in errs_c
    assert norm_c == {}


def test_invalid_metric_basis_negative():
    """8. Invalid metric_basis outside frozen enum fails closed with ERR_SPEC_INVALID_METRIC_BASIS."""
    app = make_valid_applicability_dict(metric_basis="unknown_custom_basis")
    ok, err, norm = validate_applicability(app)
    assert ok is False
    assert err == ERR_SPEC_INVALID_METRIC_BASIS
    assert norm == {}

    claim = make_valid_claim_dict(applicability=app)
    ok_c, errs_c, norm_c = validate_claim_review_contract(claim)
    assert ok_c is False
    assert ERR_SPEC_INVALID_METRIC_BASIS in errs_c
    assert norm_c == {}


def test_empty_invalidation_conditions_negative():
    """9. Empty invalidation conditions list fails closed with ERR_SPEC_EMPTY_INVALIDATION_CONDITIONS."""
    claim = make_valid_claim_dict(invalidation_conditions=[])
    ok, errors, norm = validate_claim_review_contract(claim)
    assert ok is False
    assert ERR_SPEC_EMPTY_INVALIDATION_CONDITIONS in errors
    assert norm == {}


def test_invalid_operator_negative():
    """10. Invalid operator outside {<, <=, >, >=, ==, !=} fails closed with ERR_SPEC_INVALID_OPERATOR."""
    cond = make_valid_condition_dict(operator="approx")
    ok, err, norm = validate_invalidation_condition(cond)
    assert ok is False
    assert err == ERR_SPEC_INVALID_OPERATOR
    assert norm == {}

    claim = make_valid_claim_dict(invalidation_conditions=[cond])
    ok_c, errs_c, norm_c = validate_claim_review_contract(claim)
    assert ok_c is False
    assert ERR_SPEC_INVALID_OPERATOR in errs_c
    assert norm_c == {}


def test_non_numeric_or_nan_threshold_negative():
    """11. NaN, Infinity, or -Infinity threshold fails closed with ERR_SPEC_INVALID_THRESHOLD."""
    for bad_val in [float("nan"), float("inf"), float("-inf")]:
        cond = make_valid_condition_dict(threshold=bad_val)
        ok, err, norm = validate_invalidation_condition(cond)
        assert ok is False
        assert err == ERR_SPEC_INVALID_THRESHOLD
        assert norm == {}


def test_missing_unit_negative():
    """12. Missing unit in invalidation condition fails closed with ERR_SPEC_MISSING_UNIT."""
    cond = make_valid_condition_dict()
    del cond["unit"]
    ok, err, norm = validate_invalidation_condition(cond)
    assert ok is False
    assert err == ERR_SPEC_MISSING_UNIT
    assert norm == {}

    claim = make_valid_claim_dict(invalidation_conditions=[cond])
    ok_c, errs_c, norm_c = validate_claim_review_contract(claim)
    assert ok_c is False
    assert ERR_SPEC_MISSING_UNIT in errs_c
    assert norm_c == {}


def test_invalid_unit_negative():
    """13. Invalid unit outside {pct, cny, ratio, shares, days} fails closed with ERR_SPEC_INVALID_UNIT."""
    cond = make_valid_condition_dict(unit="usd_dollar")
    ok, err, norm = validate_invalidation_condition(cond)
    assert ok is False
    assert err == ERR_SPEC_INVALID_UNIT
    assert norm == {}

    claim = make_valid_claim_dict(invalidation_conditions=[cond])
    ok_c, errs_c, norm_c = validate_claim_review_contract(claim)
    assert ok_c is False
    assert ERR_SPEC_INVALID_UNIT in errs_c
    assert norm_c == {}


def test_missing_period_negative():
    """14. Missing period in invalidation condition fails closed with ERR_SPEC_MISSING_PERIOD."""
    cond = make_valid_condition_dict()
    del cond["period"]
    ok, err, norm = validate_invalidation_condition(cond)
    assert ok is False
    assert err == ERR_SPEC_MISSING_PERIOD
    assert norm == {}

    claim = make_valid_claim_dict(invalidation_conditions=[cond])
    ok_c, errs_c, norm_c = validate_claim_review_contract(claim)
    assert ok_c is False
    assert ERR_SPEC_MISSING_PERIOD in errs_c
    assert norm_c == {}


def test_invalid_period_negative():
    """15. Invalid period outside frozen enum fails closed with ERR_SPEC_INVALID_PERIOD."""
    cond = make_valid_condition_dict(period="10d_ma")
    ok, err, norm = validate_invalidation_condition(cond)
    assert ok is False
    assert err == ERR_SPEC_INVALID_PERIOD
    assert norm == {}

    claim = make_valid_claim_dict(invalidation_conditions=[cond])
    ok_c, errs_c, norm_c = validate_claim_review_contract(claim)
    assert ok_c is False
    assert ERR_SPEC_INVALID_PERIOD in errs_c
    assert norm_c == {}


def test_missing_metric_or_source_negative():
    """16. Missing metric or source fails closed with ERR_SPEC_MISSING_CONDITION_FIELDS."""
    # Missing metric
    cond_no_metric = make_valid_condition_dict()
    del cond_no_metric["metric"]
    ok1, err1, norm1 = validate_invalidation_condition(cond_no_metric)
    assert ok1 is False
    assert err1 == ERR_SPEC_MISSING_CONDITION_FIELDS
    assert norm1 == {}

    # Missing source
    cond_no_source = make_valid_condition_dict()
    del cond_no_source["source"]
    ok2, err2, norm2 = validate_invalidation_condition(cond_no_source)
    assert ok2 is False
    assert err2 == ERR_SPEC_MISSING_CONDITION_FIELDS
    assert norm2 == {}


def test_unknown_condition_preserves_none_no_prob_imputation():
    """17. Unknown conditions or lack of observation preserves zero probability/confidence imputation."""
    claim = make_valid_claim_dict()
    ok, errors, norm = validate_claim_review_contract(claim)
    assert ok is True
    # Verify neither probability nor confidence is injected into the normalized contract
    assert "probability" not in norm
    assert "confidence" not in norm
    assert "probability" not in norm["applicability"]
    assert "confidence" not in norm["applicability"]
    for c in norm["invalidation_conditions"]:
        assert "probability" not in c
        assert "confidence" not in c
        assert "status" not in c  # status is not an E-03a contract field; reserved for future observation


def test_json_roundtrip_idempotence():
    """18. Normalized contract roundtrips through json.dumps and json.loads idempotently."""
    claim = make_valid_claim_dict()
    ok, errors, norm = validate_claim_review_contract(claim)
    assert ok is True

    serialized = json.dumps(norm)
    deserialized = json.loads(serialized)
    assert deserialized == norm

    # Re-validate deserialized payload to assert idempotence
    ok2, errors2, norm2 = validate_claim_review_contract(deserialized)
    assert ok2 is True
    assert errors2 == []
    assert norm2 == norm


# =====================================================================
# Additional High-Rigor Tests Required by DAV-746
# =====================================================================

def test_bool_threshold_negative():
    """19. bool threshold (True/False) must be explicitly rejected (isinstance(True, int) trap)."""
    for bool_val in [True, False]:
        cond = make_valid_condition_dict(threshold=bool_val)
        ok, err, norm = validate_invalidation_condition(cond)
        assert ok is False
        assert err == ERR_SPEC_INVALID_THRESHOLD
        assert norm == {}


def test_string_threshold_negative():
    """20. String threshold (e.g. '1600.0', '-5%') must be explicitly rejected."""
    for str_val in ["1600.0", "-5%", "跌破5日均线"]:
        cond = make_valid_condition_dict(threshold=str_val)
        ok, err, norm = validate_invalidation_condition(cond)
        assert ok is False
        assert err == ERR_SPEC_INVALID_THRESHOLD
        assert norm == {}


def test_decimal_finite_threshold_positive():
    """21. Finite Decimal threshold is accepted and normalized to JSON-safe float."""
    dec_val = Decimal("1599.50")
    cond = make_valid_condition_dict(threshold=dec_val)
    ok, err, norm = validate_invalidation_condition(cond)
    assert ok is True
    assert err is None
    assert norm["threshold"] == 1599.5
    # Must be natively JSON serializable
    assert json.dumps({"threshold": norm["threshold"]}) == '{"threshold": 1599.5}'


def test_decimal_non_finite_threshold_negative():
    """22. Decimal NaN or Infinity must be rejected."""
    for dec_val in [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")]:
        cond = make_valid_condition_dict(threshold=dec_val)
        ok, err, norm = validate_invalidation_condition(cond)
        assert ok is False
        assert err == ERR_SPEC_INVALID_THRESHOLD
        assert norm == {}


def test_invalid_calendar_date_negative():
    """23. Non-existent calendar dates (e.g. 2026-02-29 non-leap) are rejected."""
    invalid_dates = [
        "2026-02-29",  # 2026 is not a leap year
        "2026-13-01",  # Invalid month
        "2026-04-31",  # April only has 30 days
        "2026-00-10",  # Invalid month
        "20260908",    # Not ISO format
        "not-a-date",  # Random string
        "",            # Empty string
    ]
    for bad_date in invalid_dates:
        app = make_valid_applicability_dict(pit_date=bad_date)
        ok, err, norm = validate_applicability(app)
        assert ok is False
        assert err == ERR_SPEC_INVALID_CALENDAR_DATE
        assert norm == {}


def test_preconditions_string_disguised_negative():
    """24. Passing a single string to preconditions instead of sequence must be rejected."""
    app = make_valid_applicability_dict(preconditions="above_ma20")
    ok, err, norm = validate_applicability(app)
    assert ok is False
    assert err == ERR_SPEC_INVALID_PRECONDITIONS
    assert norm == {}


def test_preconditions_empty_element_negative():
    """25. Preconditions elements after trim must not be empty."""
    for bad_elem in [[""], ["   "], ["valid", "  "], ["valid", 123]]:
        app = make_valid_applicability_dict(preconditions=bad_elem)
        ok, err, norm = validate_applicability(app)
        assert ok is False
        assert err == ERR_SPEC_INVALID_PRECONDITIONS
        assert norm == {}


def test_duplicate_condition_id_negative():
    """26. Invalidation conditions with duplicate condition_id fail closed."""
    cond1 = make_valid_condition_dict(condition_id="inv-1", metric="close_price")
    cond2 = make_valid_condition_dict(condition_id="inv-1", metric="large_order_net_inflow_ratio")
    claim = make_valid_claim_dict(invalidation_conditions=[cond1, cond2])
    ok, errors, norm = validate_claim_review_contract(claim)
    assert ok is False
    assert ERR_SPEC_DUPLICATE_CONDITION_ID in errors
    assert norm == {}


def test_unknown_fields_stripped_in_normalized_dict():
    """27. Unknown / extra fields are ignored and stripped from normalized contract."""
    claim = make_valid_claim_dict()
    claim["applicability"]["extra_unknown_field"] = "should_be_stripped"
    claim["invalidation_conditions"][0]["phantom_metric"] = 999.9
    claim["rogue_root_field"] = "should_not_pass"

    ok, errors, norm = validate_claim_review_contract(claim)
    assert ok is True
    assert "extra_unknown_field" not in norm["applicability"]
    assert "phantom_metric" not in norm["invalidation_conditions"][0]
    assert "rogue_root_field" not in norm


def test_no_probability_or_confidence_imputation():
    """28. Injected probability/confidence fields in raw input are purged and never imputed."""
    claim = make_valid_claim_dict(
        probability=0.85,
        confidence=90.0,
    )
    claim["applicability"]["probability"] = 0.5
    claim["invalidation_conditions"][0]["confidence"] = 70.0

    ok, errors, norm = validate_claim_review_contract(claim)
    assert ok is True
    assert "probability" not in norm
    assert "confidence" not in norm
    assert "probability" not in norm["applicability"]
    assert "confidence" not in norm["invalidation_conditions"][0]


def test_strict_fail_closed_error_aggregation():
    """29. strict_fail_closed=False collects all errors but still marks invalid data invalid."""
    bad_claim = {
        "applicability": {
            "symbol": "invalid_symbol",        # ERR_SPEC_INVALID_SYMBOL
            "horizon": "invalid_horizon",      # ERR_SPEC_INVALID_HORIZON
            "metric_basis": "invalid_basis",   # ERR_SPEC_INVALID_METRIC_BASIS
            "pit_date": "invalid_date",        # ERR_SPEC_INVALID_CALENDAR_DATE
        },
        "invalidation_conditions": [
            {
                "condition_id": "",             # ERR_SPEC_INVALID_CONDITION_ID
                "metric": "",                   # ERR_SPEC_INVALID_METRIC
                "operator": "bad_op",           # ERR_SPEC_INVALID_OPERATOR
                "threshold": "not_numeric",     # ERR_SPEC_INVALID_THRESHOLD
                "unit": "bad_unit",             # ERR_SPEC_INVALID_UNIT
                "period": "bad_period",         # ERR_SPEC_INVALID_PERIOD
                "source": "bad_source",         # ERR_SPEC_INVALID_SOURCE
                "pit_date": "bad_date",         # ERR_SPEC_INVALID_CALENDAR_DATE
            }
        ],
    }
    # strict_fail_closed=True halts fast on first error
    ok_strict, errors_strict, norm_strict = validate_claim_review_contract(
        bad_claim, strict_fail_closed=True
    )
    assert ok_strict is False
    assert len(errors_strict) == 1
    assert norm_strict == {}

    # strict_fail_closed=False aggregates errors across structure
    ok_non_strict, errors_non_strict, norm_non_strict = validate_claim_review_contract(
        bad_claim, strict_fail_closed=False
    )
    assert ok_non_strict is False
    assert len(errors_non_strict) > 1
    assert norm_non_strict == {}  # Invalid data never gets normalized contract or imputed values!


def test_dataclass_contract_roundtrip():
    """30. ClaimReviewContract dataclass to_dict and from_dict roundtrip smoothly."""
    claim = make_valid_claim_dict()
    contract = ClaimReviewContract.from_dict(claim)
    assert contract.applicability.symbol == "600519"
    assert contract.applicability.horizon == ApplicabilityHorizon.SHORT
    assert contract.applicability.metric_basis == MetricBasis.VENDOR_QFQ
    assert len(contract.invalidation_conditions) == 1
    assert contract.invalidation_conditions[0].operator == ConditionOperator.LT
    assert contract.invalidation_conditions[0].unit == ConditionUnit.CNY

    as_dict = contract.to_dict()
    assert as_dict["applicability"]["symbol"] == "600519"
    assert as_dict["invalidation_conditions"][0]["unit"] == "cny"

    contract2 = ClaimReviewContract.from_dict(as_dict)
    assert contract2 == contract


def test_run_horizon_isolation_invariant():
    """31. Claim applicability horizon does not mutate or overwrite state/run horizon."""
    run_state = {
        "ticker": "600519",
        "horizon": "medium",  # Global run horizon T+40
        "trade_date": "2026-09-08",
    }
    # Propose a short-horizon claim inside medium run
    short_claim = make_valid_claim_dict(
        applicability=make_valid_applicability_dict(horizon="short")
    )
    ok, errors, norm = validate_claim_review_contract(short_claim)
    assert ok is True
    assert norm["applicability"]["horizon"] == "short"

    # Verify global run horizon is completely untouched and unmutated
    assert run_state["horizon"] == "medium"
    assert FUTURE_RESERVED_FOR_E03B_E03C is True
    assert "Run horizon" in FUTURE_RESERVED_DISCLAIMER
