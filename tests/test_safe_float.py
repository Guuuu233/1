"""Unit tests for safe_float (shared dataflows conversion helper)."""

from __future__ import annotations

import math

from tradingagents.dataflows.utils import safe_float


def test_safe_float_converts_numeric_inputs():
    assert safe_float("3.14") == 3.14
    assert safe_float(42) == 42.0
    assert safe_float("1e3") == 1000.0


def test_safe_float_none_and_empty_are_none():
    assert safe_float(None) is None
    assert safe_float("") is None


def test_safe_float_nan_becomes_none():
    assert safe_float(float("nan")) is None
    assert safe_float("nan") is None
    assert safe_float(math.nan) is None


def test_safe_float_non_numeric_becomes_none():
    assert safe_float("abc") is None
    assert safe_float([]) is None
    assert safe_float({}) is None
    assert safe_float(object()) is None


def test_safe_float_round_to():
    assert safe_float("3.14159", round_to=4) == 3.1416
    assert safe_float(1234.5678, round_to=2) == 1234.57
    assert safe_float("2.5", round_to=0) == 2.0


def test_safe_float_round_to_still_returns_none_for_invalid():
    assert safe_float("not-a-number", round_to=4) is None
    assert safe_float(float("nan"), round_to=4) is None
