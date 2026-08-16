"""Shared fund-flow guard fixtures for normal graph-path tests."""
from __future__ import annotations

from typing import Any


def valid_fund_flow_consensus_guard() -> dict[str, Any]:
    """Return a complete, explicitly unblocked fund-flow guard contract."""
    return {
        "blocked": False,
        "direction_allowed": True,
        "status": "consensus",
        "consensus": {
            "status": "consensus",
            "direction": "inflow",
            "direction_allowed": True,
            "data_conflict": False,
            "hard_guard": {
                "blocked": False,
                "direction_allowed": True,
                "reason": "test fixture consensus",
            },
        },
        "validation": {
            "status": "matched",
            "hard_guard": {
                "blocked": False,
                "reason": "test fixture validation",
            },
            "structured": {},
            "model": {},
            "mismatches": [],
            "tolerance": "0.01",
        },
        "reason": "test fixture consensus",
    }
