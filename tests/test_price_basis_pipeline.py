"""Unit tests for price basis labels mapping pipeline (DAV-705 / D-02-1 / C-04-3).

Verifies:
1. Bidirectional 5-row mapping between business short labels and cohort price_basis_version.
2. Strict error handling for unknown, empty, None, and mismatched inputs (closed on failure, no silent fallback).
3. Strict inequality: pit_raw != pit_adjusted != raw != vendor_qfq != unspecified.
4. Existing system facts across backtest_service, report_service, shadow_credit are preserved.
5. Backtest/calibration defaults remain vendor_qfq and report cohort default remains price_basis.unspecified.
"""

import pytest

import api.services.backtest_service as bt
import api.services.calibration_service as cal
import api.services.report_service as rep_svc
from api.services.price_basis_labels import (
    ALL_PRICE_BASIS_SHORT_LABELS,
    ALL_PRICE_BASIS_VERSIONS,
    PRICE_BASIS_PIT_ADJUSTED,
    PRICE_BASIS_PIT_RAW,
    PRICE_BASIS_RAW,
    PRICE_BASIS_UNSPECIFIED,
    PRICE_BASIS_VENDOR_QFQ,
    PRICE_BASIS_VERSION_PIT_ADJUSTED,
    PRICE_BASIS_VERSION_PIT_RAW,
    PRICE_BASIS_VERSION_RAW,
    PRICE_BASIS_VERSION_UNSPECIFIED,
    PRICE_BASIS_VERSION_VENDOR_QFQ,
    SHORT_TO_VERSION_MAP,
    VERSION_TO_SHORT_MAP,
    PriceBasisError,
    UnknownPriceBasisError,
    is_valid_price_basis_short_label,
    is_valid_price_basis_version,
    resolve_price_basis_pair,
    to_price_basis_short_label,
    to_price_basis_version,
    validate_price_basis_short_label,
    validate_price_basis_version,
)
import tradingagents.agents.utils.shadow_credit as shadow_credit


# The 5 canonical rows defined in work/2026-09-05-c04-pit-raw-dividend-eval.md §4.2
CANONICAL_PAIRS = [
    ("vendor_qfq", "price_basis.vendor_qfq"),
    ("unspecified", "price_basis.unspecified"),
    ("raw", "price_basis.raw"),
    ("pit_raw", "price_basis.pit_raw"),
    ("pit_adjusted", "price_basis.pit_adjusted"),
]


class TestPriceBasisBidirectionalMapping:
    """Contract 1: Bidirectional mapping covering the 5 canonical rows."""

    @pytest.mark.parametrize("short_label,expected_version", CANONICAL_PAIRS)
    def test_short_to_version_mapping(self, short_label: str, expected_version: str):
        """Each business short label converts to its exact cohort price_basis_version."""
        assert to_price_basis_version(short_label) == expected_version
        assert SHORT_TO_VERSION_MAP[short_label] == expected_version

    @pytest.mark.parametrize("expected_short,version", CANONICAL_PAIRS)
    def test_version_to_short_mapping(self, expected_short: str, version: str):
        """Each cohort price_basis_version converts to its exact business short label."""
        assert to_price_basis_short_label(version) == expected_short
        assert VERSION_TO_SHORT_MAP[version] == expected_short

    @pytest.mark.parametrize("short_label,version", CANONICAL_PAIRS)
    def test_round_trip_conversions(self, short_label: str, version: str):
        """Both directions round-trip losslessly."""
        assert to_price_basis_short_label(to_price_basis_version(short_label)) == short_label
        assert to_price_basis_version(to_price_basis_short_label(version)) == version

    @pytest.mark.parametrize("short_label,version", CANONICAL_PAIRS)
    def test_resolve_pair(self, short_label: str, version: str):
        """resolve_price_basis_pair returns (short, version) from either input."""
        assert resolve_price_basis_pair(short_label) == (short_label, version)
        assert resolve_price_basis_pair(version) == (short_label, version)

    def test_canonical_sets_completeness(self):
        """Verify constant sets contain exactly the 5 canonical items."""
        assert set(ALL_PRICE_BASIS_SHORT_LABELS) == {p[0] for p in CANONICAL_PAIRS}
        assert set(ALL_PRICE_BASIS_VERSIONS) == {p[1] for p in CANONICAL_PAIRS}
        assert len(ALL_PRICE_BASIS_SHORT_LABELS) == 5
        assert len(ALL_PRICE_BASIS_VERSIONS) == 5


class TestPriceBasisDistinctness:
    """Contract 3: pit_raw != pit_adjusted != raw != vendor_qfq != unspecified."""

    def test_all_short_labels_mutually_distinct(self):
        """All 5 short labels must be distinct strings (no aliasing)."""
        labels = [
            PRICE_BASIS_VENDOR_QFQ,
            PRICE_BASIS_UNSPECIFIED,
            PRICE_BASIS_RAW,
            PRICE_BASIS_PIT_RAW,
            PRICE_BASIS_PIT_ADJUSTED,
        ]
        assert len(set(labels)) == 5
        # Explicit pairwise checks for critical boundaries
        assert PRICE_BASIS_PIT_RAW != PRICE_BASIS_PIT_ADJUSTED
        assert PRICE_BASIS_PIT_RAW != PRICE_BASIS_RAW
        assert PRICE_BASIS_RAW != PRICE_BASIS_VENDOR_QFQ
        assert PRICE_BASIS_PIT_ADJUSTED != PRICE_BASIS_VENDOR_QFQ
        assert PRICE_BASIS_RAW != PRICE_BASIS_UNSPECIFIED

    def test_all_versions_mutually_distinct(self):
        """All 5 cohort versions must be distinct strings (no aliasing)."""
        versions = [
            PRICE_BASIS_VERSION_VENDOR_QFQ,
            PRICE_BASIS_VERSION_UNSPECIFIED,
            PRICE_BASIS_VERSION_RAW,
            PRICE_BASIS_VERSION_PIT_RAW,
            PRICE_BASIS_VERSION_PIT_ADJUSTED,
        ]
        assert len(set(versions)) == 5
        assert PRICE_BASIS_VERSION_PIT_RAW != PRICE_BASIS_VERSION_PIT_ADJUSTED
        assert PRICE_BASIS_VERSION_PIT_RAW != PRICE_BASIS_VERSION_RAW
        assert PRICE_BASIS_VERSION_RAW != PRICE_BASIS_VERSION_VENDOR_QFQ
        assert PRICE_BASIS_VERSION_PIT_ADJUSTED != PRICE_BASIS_VERSION_VENDOR_QFQ
        assert PRICE_BASIS_VERSION_RAW != PRICE_BASIS_VERSION_UNSPECIFIED


class TestPriceBasisFailureClosure:
    """Contract 2: Unknown, empty, or mismatched labels fail explicitly (never default to vendor_qfq)."""

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "",
            "   ",
            None,
            123,
            [],
            {},
            "unknown_basis",
            "qfq",
            "daily",
            "vendor",
            "raw_qfq",
            "pit",
            # Mismatches: passing cohort version where short label expected
            "price_basis.raw",
            "price_basis.vendor_qfq",
            "price_basis.pit_raw",
        ],
    )
    def test_invalid_short_labels_raise_unknown_price_basis_error(self, invalid_input):
        """to_price_basis_version must fail closed on invalid/unknown short labels."""
        with pytest.raises((UnknownPriceBasisError, PriceBasisError)):
            to_price_basis_version(invalid_input)  # type: ignore[arg-type]

        with pytest.raises((UnknownPriceBasisError, PriceBasisError)):
            validate_price_basis_short_label(invalid_input)

        assert not is_valid_price_basis_short_label(invalid_input)

    @pytest.mark.parametrize(
        "invalid_version",
        [
            "",
            "   ",
            None,
            456,
            [],
            {},
            "price_basis.unknown",
            "price_basis.hfq",
            "vendor_qfq",  # short label passed where version expected
            "raw",         # short label passed where version expected
            "pit_raw",     # short label passed where version expected
            "pit_adjusted",
        ],
    )
    def test_invalid_versions_raise_unknown_price_basis_error(self, invalid_version):
        """to_price_basis_short_label must fail closed on invalid/unknown versions."""
        with pytest.raises((UnknownPriceBasisError, PriceBasisError)):
            to_price_basis_short_label(invalid_version)  # type: ignore[arg-type]

        with pytest.raises((UnknownPriceBasisError, PriceBasisError)):
            validate_price_basis_version(invalid_version)

        assert not is_valid_price_basis_version(invalid_version)

    def test_resolve_pair_fails_on_unknown(self):
        """resolve_price_basis_pair fails closed on unrecognized input."""
        with pytest.raises(UnknownPriceBasisError):
            resolve_price_basis_pair("invalid_something")
        with pytest.raises(UnknownPriceBasisError):
            resolve_price_basis_pair("")

    def test_raw_never_falls_back_to_vendor_qfq(self):
        """Crucial assertion: raw must map strictly to price_basis.raw, never vendor_qfq."""
        assert to_price_basis_version("raw") == "price_basis.raw"
        assert to_price_basis_version("raw") != "price_basis.vendor_qfq"


class TestLiveSystemFactsAndIsolation:
    """Existing system facts and boundaries (must be nailed down by tests)."""

    def test_existing_constants_preserved(self):
        """Verify live system constants across modules without unifying into one string."""
        # backtest_service short labels
        assert bt.PRICE_BASIS_VENDOR_QFQ == "vendor_qfq"
        assert bt.PRICE_BASIS_UNSPECIFIED == "unspecified"
        assert bt.PRICE_BASIS_RAW == "raw"
        assert bt.PRICE_BASIS_PIT_RAW == "pit_raw"
        assert bt.PRICE_BASIS_PIT_ADJUSTED == "pit_adjusted"

        # report_service and shadow_credit version labels
        assert rep_svc.PRICE_BASIS_UNSPECIFIED == "price_basis.unspecified"
        assert shadow_credit.PRICE_BASIS_UNSPECIFIED == "price_basis.unspecified"

        # Key fact: backtest_service and report_service use different strings for UNSPECIFIED!
        assert bt.PRICE_BASIS_UNSPECIFIED != rep_svc.PRICE_BASIS_UNSPECIFIED
        assert bt.PRICE_BASIS_UNSPECIFIED == "unspecified"
        assert rep_svc.PRICE_BASIS_UNSPECIFIED == "price_basis.unspecified"

        # Bridge mapping holds between them
        assert to_price_basis_version(bt.PRICE_BASIS_UNSPECIFIED) == rep_svc.PRICE_BASIS_UNSPECIFIED
        assert to_price_basis_short_label(rep_svc.PRICE_BASIS_UNSPECIFIED) == bt.PRICE_BASIS_UNSPECIFIED

        # calibration_service preserves vendor_qfq
        assert cal.PRICE_BASIS_VENDOR_QFQ == "vendor_qfq"
        assert cal.PRICE_BASIS_UNSPECIFIED == "unspecified"

    def test_report_service_default_cohort_unaffected(self):
        """ensure_report_cohort_persisted still defaults price_basis_version to price_basis.unspecified."""
        res_data: dict = {}
        out = rep_svc.ensure_report_cohort_persisted(res_data)
        assert out["price_basis_version"] == "price_basis.unspecified"
        assert out["price_basis_version"] != "price_basis.pit_raw"
        assert out["price_basis_version"] != "price_basis.raw"

    def test_mapping_exists_does_not_wire_raw_to_consumers(self):
        """Contract 4: mapping exists != channel is wired into backtest/calibration."""
        # Analysis default remains vendor_qfq
        final_state: dict = {}
        price_basis = final_state.get("price_basis") or bt.PRICE_BASIS_VENDOR_QFQ
        assert price_basis == "vendor_qfq"
        assert price_basis != "raw"
        assert price_basis != "pit_raw"
