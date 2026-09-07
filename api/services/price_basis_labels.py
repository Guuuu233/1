"""Price basis label definitions and bidirectional mapping.

Freeze compatibility mappings between business short labels (used in backtest_service)
and cohort price_basis_version labels (used in report_service / cohort triads).

Specification reference:
- work/2026-09-05-c04-pit-raw-dividend-eval.md §4.2
- D-02-1 / C-04-3 (DAV-705)
"""

from typing import Any, Dict, Tuple


# ==============================================================================
# Business Short Labels (price_basis)
# ==============================================================================
PRICE_BASIS_VENDOR_QFQ: str = "vendor_qfq"
PRICE_BASIS_UNSPECIFIED: str = "unspecified"
PRICE_BASIS_RAW: str = "raw"
PRICE_BASIS_PIT_RAW: str = "pit_raw"
PRICE_BASIS_PIT_ADJUSTED: str = "pit_adjusted"

ALL_PRICE_BASIS_SHORT_LABELS: Tuple[str, ...] = (
    PRICE_BASIS_VENDOR_QFQ,
    PRICE_BASIS_UNSPECIFIED,
    PRICE_BASIS_RAW,
    PRICE_BASIS_PIT_RAW,
    PRICE_BASIS_PIT_ADJUSTED,
)


# ==============================================================================
# Cohort Triad Labels (price_basis_version)
# ==============================================================================
PRICE_BASIS_VERSION_VENDOR_QFQ: str = "price_basis.vendor_qfq"
PRICE_BASIS_VERSION_UNSPECIFIED: str = "price_basis.unspecified"
PRICE_BASIS_VERSION_RAW: str = "price_basis.raw"
PRICE_BASIS_VERSION_PIT_RAW: str = "price_basis.pit_raw"
PRICE_BASIS_VERSION_PIT_ADJUSTED: str = "price_basis.pit_adjusted"

ALL_PRICE_BASIS_VERSIONS: Tuple[str, ...] = (
    PRICE_BASIS_VERSION_VENDOR_QFQ,
    PRICE_BASIS_VERSION_UNSPECIFIED,
    PRICE_BASIS_VERSION_RAW,
    PRICE_BASIS_VERSION_PIT_RAW,
    PRICE_BASIS_VERSION_PIT_ADJUSTED,
)


# ==============================================================================
# Exceptions
# ==============================================================================
class PriceBasisError(ValueError):
    """Base exception for price basis label and mapping operations."""


class UnknownPriceBasisError(PriceBasisError):
    """Raised when an unknown, invalid, or empty price basis label/version is supplied."""


class InvalidPriceBasisMappingError(PriceBasisError):
    """Raised when a price basis label cannot be mapped to the expected target domain."""


# ==============================================================================
# Canonical Bidirectional Mapping Tables
# ==============================================================================
# 5 distinct mappings:
# vendor_qfq    <-> price_basis.vendor_qfq
# unspecified   <-> price_basis.unspecified
# raw           <-> price_basis.raw
# pit_raw       <-> price_basis.pit_raw
# pit_adjusted  <-> price_basis.pit_adjusted

SHORT_TO_VERSION_MAP: Dict[str, str] = {
    PRICE_BASIS_VENDOR_QFQ: PRICE_BASIS_VERSION_VENDOR_QFQ,
    PRICE_BASIS_UNSPECIFIED: PRICE_BASIS_VERSION_UNSPECIFIED,
    PRICE_BASIS_RAW: PRICE_BASIS_VERSION_RAW,
    PRICE_BASIS_PIT_RAW: PRICE_BASIS_VERSION_PIT_RAW,
    PRICE_BASIS_PIT_ADJUSTED: PRICE_BASIS_VERSION_PIT_ADJUSTED,
}

VERSION_TO_SHORT_MAP: Dict[str, str] = {
    PRICE_BASIS_VERSION_VENDOR_QFQ: PRICE_BASIS_VENDOR_QFQ,
    PRICE_BASIS_VERSION_UNSPECIFIED: PRICE_BASIS_UNSPECIFIED,
    PRICE_BASIS_VERSION_RAW: PRICE_BASIS_RAW,
    PRICE_BASIS_VERSION_PIT_RAW: PRICE_BASIS_PIT_RAW,
    PRICE_BASIS_VERSION_PIT_ADJUSTED: PRICE_BASIS_PIT_ADJUSTED,
}


# ==============================================================================
# Validation & Conversion Functions
# ==============================================================================
def is_valid_price_basis_short_label(label: Any) -> bool:
    """Return True if label is a valid business short label."""
    return isinstance(label, str) and label in SHORT_TO_VERSION_MAP


def is_valid_price_basis_version(version: Any) -> bool:
    """Return True if version is a valid cohort price_basis_version."""
    return isinstance(version, str) and version in VERSION_TO_SHORT_MAP


def validate_price_basis_short_label(label: Any) -> str:
    """Validate that label is a known short label and return it, else raise UnknownPriceBasisError."""
    if not is_valid_price_basis_short_label(label):
        raise UnknownPriceBasisError(
            f"Invalid price basis short label: {label!r}. "
            f"Allowed labels: {list(SHORT_TO_VERSION_MAP.keys())}"
        )
    return str(label)


def validate_price_basis_version(version: Any) -> str:
    """Validate that version is a known version label and return it, else raise UnknownPriceBasisError."""
    if not is_valid_price_basis_version(version):
        raise UnknownPriceBasisError(
            f"Invalid price basis version: {version!r}. "
            f"Allowed versions: {list(VERSION_TO_SHORT_MAP.keys())}"
        )
    return str(version)


def to_price_basis_version(short_label: str) -> str:
    """Convert a business short label (price_basis) to cohort price_basis_version.

    Args:
        short_label: e.g. "vendor_qfq", "unspecified", "raw", "pit_raw", "pit_adjusted".

    Returns:
        The corresponding cohort version string (e.g. "price_basis.vendor_qfq").

    Raises:
        UnknownPriceBasisError: If short_label is empty, None, non-string, or unknown.
            Never silently falls back to vendor_qfq or any default.
    """
    if not isinstance(short_label, str) or not short_label.strip():
        raise UnknownPriceBasisError(
            f"Price basis short label must be a non-empty string, got: {short_label!r}"
        )
    normalized = short_label.strip()
    if normalized not in SHORT_TO_VERSION_MAP:
        raise UnknownPriceBasisError(
            f"Unknown price basis short label: {short_label!r}. "
            f"Allowed labels: {list(SHORT_TO_VERSION_MAP.keys())}"
        )
    return SHORT_TO_VERSION_MAP[normalized]


def to_price_basis_short_label(version: str) -> str:
    """Convert a cohort price_basis_version to business short label (price_basis).

    Args:
        version: e.g. "price_basis.vendor_qfq", "price_basis.unspecified", etc.

    Returns:
        The corresponding business short label (e.g. "vendor_qfq").

    Raises:
        UnknownPriceBasisError: If version is empty, None, non-string, or unknown.
            Never silently falls back to vendor_qfq or any default.
    """
    if not isinstance(version, str) or not version.strip():
        raise UnknownPriceBasisError(
            f"Price basis version must be a non-empty string, got: {version!r}"
        )
    normalized = version.strip()
    if normalized not in VERSION_TO_SHORT_MAP:
        raise UnknownPriceBasisError(
            f"Unknown price basis version: {version!r}. "
            f"Allowed versions: {list(VERSION_TO_SHORT_MAP.keys())}"
        )
    return VERSION_TO_SHORT_MAP[normalized]


def resolve_price_basis_pair(label_or_version: str) -> Tuple[str, str]:
    """Given either a short label or a version string, resolve both (short_label, version).

    Returns:
        (short_label, version) tuple.

    Raises:
        UnknownPriceBasisError: If the input does not match any valid short label or version.
    """
    if not isinstance(label_or_version, str) or not label_or_version.strip():
        raise UnknownPriceBasisError(
            f"Input must be a non-empty string, got: {label_or_version!r}"
        )
    normalized = label_or_version.strip()
    if normalized in SHORT_TO_VERSION_MAP:
        return normalized, SHORT_TO_VERSION_MAP[normalized]
    if normalized in VERSION_TO_SHORT_MAP:
        return VERSION_TO_SHORT_MAP[normalized], normalized
    raise UnknownPriceBasisError(
        f"Input {label_or_version!r} is neither a recognized short label nor a recognized version."
    )
