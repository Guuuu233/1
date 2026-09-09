"""Canonical Symbol Normalization, Quarantine Isolation, and Collision Detection.

Implements standard stock symbol canonicalization for V-03 / A-share evaluation:
- Canonical format: 6-digit code + exchange suffix (.SZ / .SH)
- Deterministic prefix mapping:
  * 000, 001, 002, 003, 300, 301 -> .SZ
  * 600, 601, 603, 605, 688, 689 -> .SH
  * 8xx, 4xx, 920 (Beijing Stock Exchange / BSE) -> .BJ, excluded by stock pool rule
    (not silently dropped; explicitly marked, isolated, and counted)
- Bare codes: automatically suffixed (e.g. 000001 -> 000001.SZ, 600519 -> 600519.SH)
- Already canonical codes: preserved unchanged (idempotent, e.g. 000001.SZ -> 000001.SZ)
- Unmappable symbols: empty, non-numeric/non-6-digit, or unknown prefix -> isolated & counted
- Explicit collision and duplicate checking: surfaces collisions without silent join or drop.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class CanonicalStatus(str, Enum):
    """Classification status of a processed symbol."""

    CANONICAL = "canonical"  # Valid A-share stock symbol (.SZ / .SH) eligible for trading pool
    EXCLUDED_BSE = "excluded_bse"  # Beijing Stock Exchange (.BJ) excluded under stock pool rule
    UNMAPPABLE = "unmappable"  # Empty, non-numeric, or unmappable prefix


class UnmappableReason(str, Enum):
    """Specific reason why a symbol cannot be mapped to a canonical format."""

    EMPTY = "empty"  # None, empty string, or whitespace-only
    INVALID_FORMAT = "invalid_format"  # Non-6-digit, non-numeric (e.g. 'AGENT', 'AUUSDO')
    UNKNOWN_PREFIX = "unknown_prefix"  # 6 digits but prefix not recognized in A-share prefix rules
    SUFFIX_MISMATCH = "suffix_mismatch"  # Suffix contradicts deterministic prefix mapping


# Deterministic prefix sets
SZ_PREFIXES: Tuple[str, ...] = ("000", "001", "002", "003", "300", "301")
SH_PREFIXES: Tuple[str, ...] = ("600", "601", "603", "605", "688", "689")

# BSE prefixes: 8xx, 4xx, 920
BSE_PREFIX_STARTS: Tuple[str, ...] = ("8", "4")
BSE_PREFIX_3DIGIT: Tuple[str, ...] = ("920",)


def resolve_exchange_from_prefix(code: str) -> Optional[str]:
    """Deterministically resolve exchange from 6-digit stock code prefix.

    Returns:
        'SZ' for Shenzhen prefixes (000/001/002/003/300/301)
        'SH' for Shanghai prefixes (600/601/603/605/688/689)
        'BJ' for Beijing prefixes (8xx/4xx/920)
        None if prefix is not recognized.
    """
    if not code or len(code) != 6 or not code.isdigit():
        return None

    if code.startswith(SZ_PREFIXES):
        return "SZ"
    if code.startswith(SH_PREFIXES):
        return "SH"
    if code.startswith(BSE_PREFIX_STARTS) or code.startswith(BSE_PREFIX_3DIGIT):
        return "BJ"
    return None


@dataclass(frozen=True)
class CanonicalSymbolResult:
    """Result of single symbol canonicalization."""

    raw_symbol: Optional[str]
    canonical_symbol: Optional[str]
    status: CanonicalStatus
    reason: Optional[str] = None
    code: Optional[str] = None
    exchange: Optional[str] = None

    @property
    def is_canonical(self) -> bool:
        return self.status == CanonicalStatus.CANONICAL

    @property
    def is_excluded_bse(self) -> bool:
        return self.status == CanonicalStatus.EXCLUDED_BSE

    @property
    def is_unmappable(self) -> bool:
        return self.status == CanonicalStatus.UNMAPPABLE


def canonicalize_symbol(symbol: Any) -> CanonicalSymbolResult:
    """Canonicalize a single stock symbol string.

    Rules:
    1. Empty/None/whitespace -> UNMAPPABLE (reason: EMPTY)
    2. Normalize Unicode NFKC, strip whitespace, uppercase.
    3. Parse suffix notation (e.g. '000001.SZ', '600519.SH', '600519.SS', '830001.BJ'),
       prefix notation (e.g. 'SZ000001', 'SH600519', 'BJ830001'), or bare code ('000001').
    4. Non-6-digit / non-numeric code -> UNMAPPABLE (reason: INVALID_FORMAT).
    5. Resolve exchange:
       - If code prefix has deterministic exchange:
         * If explicit suffix was provided and contradicts deterministic exchange -> UNMAPPABLE (reason: SUFFIX_MISMATCH)
         * If exchange is 'BJ' -> EXCLUDED_BSE (reason: 'bse_excluded')
         * If exchange is 'SZ' or 'SH' -> CANONICAL
       - If code prefix is unknown:
         * If explicit valid suffix (.SZ / .SH) was already present -> CANONICAL (preserve existing canonical)
         * If explicit .BJ was present -> EXCLUDED_BSE (reason: 'bse_excluded')
         * Otherwise -> UNMAPPABLE (reason: UNKNOWN_PREFIX)

    Returns:
        CanonicalSymbolResult
    """
    if symbol is None:
        return CanonicalSymbolResult(
            raw_symbol=None,
            canonical_symbol=None,
            status=CanonicalStatus.UNMAPPABLE,
            reason=UnmappableReason.EMPTY.value,
        )

    if not isinstance(symbol, str):
        symbol = str(symbol)

    cleaned = unicodedata.normalize("NFKC", symbol).strip().upper()
    if not cleaned:
        return CanonicalSymbolResult(
            raw_symbol=symbol,
            canonical_symbol=None,
            status=CanonicalStatus.UNMAPPABLE,
            reason=UnmappableReason.EMPTY.value,
        )

    # 1. Suffix notation: e.g. 600519.SH, 000001.SZ, 830001.BJ, 600519.SS
    m_suffix = re.match(r"^(\d{6})\.(SH|SZ|BJ|SS)$", cleaned)
    if m_suffix:
        code = m_suffix.group(1)
        suffix = m_suffix.group(2)
        if suffix == "SS":
            suffix = "SH"
        prefix_ex = resolve_exchange_from_prefix(code)
        if prefix_ex is not None and prefix_ex != suffix:
            return CanonicalSymbolResult(
                raw_symbol=symbol,
                canonical_symbol=None,
                status=CanonicalStatus.UNMAPPABLE,
                reason=UnmappableReason.SUFFIX_MISMATCH.value,
                code=code,
                exchange=suffix,
            )
        effective_ex = prefix_ex or suffix
        canonical = f"{code}.{effective_ex}"
        if effective_ex == "BJ":
            return CanonicalSymbolResult(
                raw_symbol=symbol,
                canonical_symbol=canonical,
                status=CanonicalStatus.EXCLUDED_BSE,
                reason="bse_excluded",
                code=code,
                exchange="BJ",
            )
        return CanonicalSymbolResult(
            raw_symbol=symbol,
            canonical_symbol=canonical,
            status=CanonicalStatus.CANONICAL,
            code=code,
            exchange=effective_ex,
        )

    # 2. Prefix notation: e.g. SH600519, SZ000001, BJ830001
    m_prefix = re.match(r"^(SH|SZ|BJ)(\d{6})$", cleaned)
    if m_prefix:
        prefix = m_prefix.group(1)
        code = m_prefix.group(2)
        prefix_ex = resolve_exchange_from_prefix(code)
        if prefix_ex is not None and prefix_ex != prefix:
            return CanonicalSymbolResult(
                raw_symbol=symbol,
                canonical_symbol=None,
                status=CanonicalStatus.UNMAPPABLE,
                reason=UnmappableReason.SUFFIX_MISMATCH.value,
                code=code,
                exchange=prefix,
            )
        effective_ex = prefix_ex or prefix
        canonical = f"{code}.{effective_ex}"
        if effective_ex == "BJ":
            return CanonicalSymbolResult(
                raw_symbol=symbol,
                canonical_symbol=canonical,
                status=CanonicalStatus.EXCLUDED_BSE,
                reason="bse_excluded",
                code=code,
                exchange="BJ",
            )
        return CanonicalSymbolResult(
            raw_symbol=symbol,
            canonical_symbol=canonical,
            status=CanonicalStatus.CANONICAL,
            code=code,
            exchange=effective_ex,
        )

    # 3. Pure 6-digit bare code: e.g. 000001, 600519, 603259, 830001
    if re.match(r"^\d{6}$", cleaned):
        code = cleaned
        prefix_ex = resolve_exchange_from_prefix(code)
        if prefix_ex is None:
            return CanonicalSymbolResult(
                raw_symbol=symbol,
                canonical_symbol=None,
                status=CanonicalStatus.UNMAPPABLE,
                reason=UnmappableReason.UNKNOWN_PREFIX.value,
                code=code,
            )
        canonical = f"{code}.{prefix_ex}"
        if prefix_ex == "BJ":
            return CanonicalSymbolResult(
                raw_symbol=symbol,
                canonical_symbol=canonical,
                status=CanonicalStatus.EXCLUDED_BSE,
                reason="bse_excluded",
                code=code,
                exchange="BJ",
            )
        return CanonicalSymbolResult(
            raw_symbol=symbol,
            canonical_symbol=canonical,
            status=CanonicalStatus.CANONICAL,
            code=code,
            exchange=prefix_ex,
        )

    # 4. Any other format (non-numeric, wrong length, etc.)
    return CanonicalSymbolResult(
        raw_symbol=symbol,
        canonical_symbol=None,
        status=CanonicalStatus.UNMAPPABLE,
        reason=UnmappableReason.INVALID_FORMAT.value,
    )


def normalize_symbol(symbol: Any) -> Optional[str]:
    """Convenience helper returning canonical symbol string if valid, else None."""
    res = canonicalize_symbol(symbol)
    return res.canonical_symbol if res.is_canonical else None


def find_symbol_collisions(symbols: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    """Detect collisions where multiple distinct raw symbol representations map to the same canonical symbol.

    Returns:
        Dict[canonical_symbol, {
            'canonical_symbol': str,
            'raw_symbols': List[str],
            'raw_counts': Dict[str, int],
            'total_occurrences': int
        }]
    """
    raw_counts: Dict[str, Counter] = defaultdict(Counter)

    for sym in symbols:
        res = canonicalize_symbol(sym)
        if res.canonical_symbol:
            raw_key = "" if sym is None else str(sym)
            raw_counts[res.canonical_symbol][raw_key] += 1

    collisions: Dict[str, Dict[str, Any]] = {}
    for canon, counts in raw_counts.items():
        if len(counts) > 1:
            collisions[canon] = {
                "canonical_symbol": canon,
                "raw_symbols": sorted(counts.keys()),
                "raw_counts": dict(counts),
                "total_occurrences": sum(counts.values()),
            }

    return collisions


@dataclass
class CanonicalDatasetResult:
    """Result of batch dataset processing with explicit counts and isolation."""

    total_count: int
    canonical_count: int
    bse_excluded_count: int
    unmappable_count: int
    canonical_records: List[Dict[str, Any]] = field(default_factory=list)
    bse_excluded_records: List[Dict[str, Any]] = field(default_factory=list)
    unmappable_records: List[Dict[str, Any]] = field(default_factory=list)
    collisions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    collision_count: int = 0
    unmappable_by_reason: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_count": self.total_count,
            "canonical_count": self.canonical_count,
            "bse_excluded_count": self.bse_excluded_count,
            "unmappable_count": self.unmappable_count,
            "collision_count": self.collision_count,
            "unmappable_by_reason": self.unmappable_by_reason,
            "collisions": self.collisions,
        }


def process_symbols_dataset(
    records: Iterable[Dict[str, Any]],
    symbol_key: str = "symbol",
    copy_records: bool = True,
) -> CanonicalDatasetResult:
    """Process an iterable of record dictionaries, canonicalizing symbols and isolating non-canonical rows.

    Invariants guaranteed:
    - Never silently drops records: total_count == canonical_count + bse_excluded_count + unmappable_count
    - Explicit quarantine buckets with reasons
    - Surfaces all collisions between raw variants
    - Attaches '_canonical_symbol', '_canonical_status', '_canonical_reason' metadata to processed records

    Args:
        records: Iterable of dicts containing symbol data
        symbol_key: Key in record dict pointing to symbol
        copy_records: Whether to create copies of dicts to prevent mutating caller's data

    Returns:
        CanonicalDatasetResult
    """
    canonical_records: List[Dict[str, Any]] = []
    bse_excluded_records: List[Dict[str, Any]] = []
    unmappable_records: List[Dict[str, Any]] = []
    unmappable_counter: Counter = Counter()

    # Track raw symbols per canonical symbol for collision analysis
    raw_symbol_tracking: Dict[str, Counter] = defaultdict(Counter)

    total = 0
    for r in records:
        total += 1
        rec = dict(r) if copy_records else r
        raw_val = rec.get(symbol_key)
        res = canonicalize_symbol(raw_val)

        rec["_canonical_symbol"] = res.canonical_symbol
        rec["_canonical_status"] = res.status.value
        rec["_canonical_reason"] = res.reason

        raw_str = "" if raw_val is None else str(raw_val)

        if res.status == CanonicalStatus.CANONICAL:
            canonical_records.append(rec)
            raw_symbol_tracking[res.canonical_symbol][raw_str] += 1
        elif res.status == CanonicalStatus.EXCLUDED_BSE:
            bse_excluded_records.append(rec)
            raw_symbol_tracking[res.canonical_symbol][raw_str] += 1
        else:  # UNMAPPABLE
            unmappable_records.append(rec)
            reason_str = res.reason or "unknown"
            unmappable_counter[reason_str] += 1

    # Surface collisions
    collisions: Dict[str, Dict[str, Any]] = {}
    for canon, counts in raw_symbol_tracking.items():
        if len(counts) > 1:
            collisions[canon] = {
                "canonical_symbol": canon,
                "raw_symbols": sorted(counts.keys()),
                "raw_counts": dict(counts),
                "total_occurrences": sum(counts.values()),
            }

    return CanonicalDatasetResult(
        total_count=total,
        canonical_count=len(canonical_records),
        bse_excluded_count=len(bse_excluded_records),
        unmappable_count=len(unmappable_records),
        canonical_records=canonical_records,
        bse_excluded_records=bse_excluded_records,
        unmappable_records=unmappable_records,
        collisions=collisions,
        collision_count=len(collisions),
        unmappable_by_reason=dict(unmappable_counter),
    )


def check_collision_and_duplicates(
    records: Iterable[Dict[str, Any]],
    symbol_key: str = "symbol",
    id_key: str = "id",
) -> Dict[str, Any]:
    """Check for symbol collisions and record ID duplicates in a dataset.

    Returns diagnostic summary dictionary.
    """
    id_counter: Counter = Counter()
    processed = process_symbols_dataset(records, symbol_key=symbol_key, copy_records=False)

    for r in processed.canonical_records + processed.bse_excluded_records + processed.unmappable_records:
        rec_id = r.get(id_key)
        if rec_id:
            id_counter[rec_id] += 1

    duplicate_ids = {k: v for k, v in id_counter.items() if v > 1}

    return {
        "has_collisions": processed.collision_count > 0,
        "collision_count": processed.collision_count,
        "collisions": processed.collisions,
        "has_duplicate_ids": len(duplicate_ids) > 0,
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids,
        "dataset_summary": processed.summary(),
    }
