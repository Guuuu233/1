"""Tests for Canonical Symbol Normalization, Quarantine Isolation, and Collision Checking.

Covers Red Team Scenarios RT-1 to RT-7 (D-012 §5b):
- RT-1: Collision handling (000001 & 000001.SZ groupby merge without split)
- RT-2: Empty symbol '' quarantine and counting (no silent join, no silent drop)
- RT-3: Illegal values AGENT / AUUSDO quarantine with reason
- RT-4: Canonical symbol idempotency (000001.SZ unchanged)
- RT-5: Prefix mapping (300xxx->.SZ, 688xxx->.SH, 600519->.SH)
- RT-6: BSE (Beijing Stock Exchange) 8xxxxx/4xxxxx/920xxx marked .BJ and pool-excluded
- RT-7: Collision and duplicate checking actively surfaced without silence
"""

from collections import Counter
import pytest

from tradingagents.agents.utils.symbol_canonical import (
    CanonicalDatasetResult,
    CanonicalStatus,
    CanonicalSymbolResult,
    UnmappableReason,
    canonicalize_symbol,
    check_collision_and_duplicates,
    find_symbol_collisions,
    normalize_symbol,
    process_symbols_dataset,
    resolve_exchange_from_prefix,
)


# ============================================================================
# RT-1: Collision Core (000001 and 000001.SZ Merge Groupby)
# ============================================================================

def test_rt1_collision_groupby_merge():
    """RT-1: 000001 and 000001.SZ merge into single canonical 000001.SZ.

    Groupby count merges correctly (2 + 57 = 59) rather than splitting.
    """
    records = []
    # 2 rows of bare code 000001
    for i in range(2):
        records.append({"id": f"bare_000001_{i}", "symbol": "000001", "val": 1})
    # 57 rows of canonical 000001.SZ
    for i in range(57):
        records.append({"id": f"canon_000001_{i}", "symbol": "000001.SZ", "val": 1})

    result = process_symbols_dataset(records)

    # Invariants
    assert result.total_count == 59
    assert result.canonical_count == 59
    assert result.unmappable_count == 0
    assert result.bse_excluded_count == 0

    # Groupby on canonical symbol
    counts = Counter(r["_canonical_symbol"] for r in result.canonical_records)
    assert counts["000001.SZ"] == 59
    assert len(counts) == 1, "Must merge into 1 group, not split into 2"

    # Collision actively detected
    assert result.collision_count == 1
    assert "000001.SZ" in result.collisions
    col_info = result.collisions["000001.SZ"]
    assert col_info["raw_symbols"] == ["000001", "000001.SZ"]
    assert col_info["raw_counts"] == {"000001": 2, "000001.SZ": 57}
    assert col_info["total_occurrences"] == 59


def test_rt1_multiple_collisions_merge_cleanly():
    """RT-1 extended: verify all 3 real-world collision pairs from reports table."""
    records = [
        {"id": "r1", "symbol": "000001"},
        {"id": "r2", "symbol": "000001.SZ"},
        {"id": "r3", "symbol": "600519"},
        {"id": "r4", "symbol": "600519.SH"},
        {"id": "r5", "symbol": "603259"},
        {"id": "r6", "symbol": "603259.SH"},
    ]

    result = process_symbols_dataset(records)
    assert result.collision_count == 3
    assert set(result.collisions.keys()) == {"000001.SZ", "600519.SH", "603259.SH"}

    grouped = Counter(r["_canonical_symbol"] for r in result.canonical_records)
    assert grouped["000001.SZ"] == 2
    assert grouped["600519.SH"] == 2
    assert grouped["603259.SH"] == 2


# ============================================================================
# RT-2: Empty Symbol Quarantine and Counting
# ============================================================================

def test_rt2_empty_symbol_isolated_and_counted():
    """RT-2: Empty string '' (mimicking 32 reports) is marked UNMAPPABLE, isolated, and counted.

    No silent drop: every record is accounted for.
    """
    records = [{"id": f"empty_{i}", "symbol": ""} for i in range(32)]
    # Add some valid records
    records.append({"id": "valid_1", "symbol": "600519.SH"})

    result = process_symbols_dataset(records)

    assert result.total_count == 33
    assert result.canonical_count == 1
    assert result.unmappable_count == 32
    assert len(result.unmappable_records) == 32

    # Quarantine verification
    for r in result.unmappable_records:
        assert r["_canonical_symbol"] is None
        assert r["_canonical_status"] == CanonicalStatus.UNMAPPABLE.value
        assert r["_canonical_reason"] == UnmappableReason.EMPTY.value

    assert result.unmappable_by_reason == {UnmappableReason.EMPTY.value: 32}

    # Strict conservation invariant: no silent drop
    assert result.total_count == (
        result.canonical_count + result.bse_excluded_count + result.unmappable_count
    )


def test_rt2_none_and_whitespace_variants():
    """RT-2: None and whitespace-only strings are also safely marked empty."""
    res_none = canonicalize_symbol(None)
    assert res_none.status == CanonicalStatus.UNMAPPABLE
    assert res_none.reason == UnmappableReason.EMPTY.value
    assert res_none.canonical_symbol is None

    res_ws = canonicalize_symbol("   \t\n")
    assert res_ws.status == CanonicalStatus.UNMAPPABLE
    assert res_ws.reason == UnmappableReason.EMPTY.value
    assert res_ws.canonical_symbol is None


# ============================================================================
# RT-3: Illegal Values Quarantine (AGENT / AUUSDO)
# ============================================================================

def test_rt3_illegal_values_quarantined_with_reasons():
    """RT-3: Non-code values AGENT and AUUSDO are marked UNMAPPABLE with reasons and isolated."""
    res_agent = canonicalize_symbol("AGENT")
    assert res_agent.status == CanonicalStatus.UNMAPPABLE
    assert res_agent.reason == UnmappableReason.INVALID_FORMAT.value
    assert res_agent.canonical_symbol is None

    res_auusdo = canonicalize_symbol("AUUSDO")
    assert res_auusdo.status == CanonicalStatus.UNMAPPABLE
    assert res_auusdo.reason == UnmappableReason.INVALID_FORMAT.value
    assert res_auusdo.canonical_symbol is None

    # In dataset processing
    records = [
        {"id": "r_agent", "symbol": "AGENT"},
        {"id": "r_auusdo", "symbol": "AUUSDO"},
        {"id": "r_valid", "symbol": "600519.SH"},
    ]
    result = process_symbols_dataset(records)
    assert result.unmappable_count == 2
    assert result.unmappable_by_reason == {UnmappableReason.INVALID_FORMAT.value: 2}
    assert [r["id"] for r in result.unmappable_records] == ["r_agent", "r_auusdo"]


def test_rt3_other_invalid_formats():
    """RT-3: Numbers with wrong lengths or non-numeric tokens."""
    for bad in ["12345", "1234567", "600519.", ".SZ", "AAPL", "600519.XYZ"]:
        res = canonicalize_symbol(bad)
        assert res.status == CanonicalStatus.UNMAPPABLE
        assert res.canonical_symbol is None


# ============================================================================
# RT-4: Idempotency of Already Canonical Symbols
# ============================================================================

def test_rt4_already_canonical_idempotent():
    """RT-4: 000001.SZ is unchanged and idempotent."""
    res = canonicalize_symbol("000001.SZ")
    assert res.canonical_symbol == "000001.SZ"
    assert res.status == CanonicalStatus.CANONICAL
    assert res.is_canonical is True

    # Idempotent re-application
    res2 = canonicalize_symbol(res.canonical_symbol)
    assert res2.canonical_symbol == "000001.SZ"
    assert res2.status == CanonicalStatus.CANONICAL

    # Case insensitivity
    res_lower = canonicalize_symbol("000001.sz")
    assert res_lower.canonical_symbol == "000001.SZ"


def test_rt4_shanghai_ss_alias_normalization():
    """RT-4: 600519.SS maps to canonical 600519.SH idempotently."""
    res = canonicalize_symbol("600519.SS")
    assert res.canonical_symbol == "600519.SH"
    assert res.status == CanonicalStatus.CANONICAL

    res2 = canonicalize_symbol(res.canonical_symbol)
    assert res2.canonical_symbol == "600519.SH"


# ============================================================================
# RT-5: Deterministic Prefix Mapping
# ============================================================================

def test_rt5_prefix_mapping():
    """RT-5: Prefix mapping rules:

    300xxx -> .SZ
    688xxx -> .SH
    600519 -> .SH
    """
    # 300xxx -> .SZ
    assert canonicalize_symbol("300750").canonical_symbol == "300750.SZ"
    # 688xxx -> .SH
    assert canonicalize_symbol("688981").canonical_symbol == "688981.SH"
    # 600519 -> .SH
    assert canonicalize_symbol("600519").canonical_symbol == "600519.SH"


def test_rt5_full_sz_prefixes():
    """RT-5: All specified SZ prefixes: 000/001/002/003/300/301 -> .SZ."""
    cases = [
        ("000001", "000001.SZ"),
        ("001234", "001234.SZ"),
        ("002273", "002273.SZ"),
        ("003001", "003001.SZ"),
        ("300015", "300015.SZ"),
        ("301001", "301001.SZ"),
    ]
    for raw, expected in cases:
        res = canonicalize_symbol(raw)
        assert res.canonical_symbol == expected, f"Failed for {raw}"
        assert res.status == CanonicalStatus.CANONICAL


def test_rt5_full_sh_prefixes():
    """RT-5: All specified SH prefixes: 600/601/603/605/688/689 -> .SH."""
    cases = [
        ("600036", "600036.SH"),
        ("601398", "601398.SH"),
        ("603259", "603259.SH"),
        ("605001", "605001.SH"),
        ("688012", "688012.SH"),
        ("689009", "689009.SH"),
    ]
    for raw, expected in cases:
        res = canonicalize_symbol(raw)
        assert res.canonical_symbol == expected, f"Failed for {raw}"
        assert res.status == CanonicalStatus.CANONICAL


def test_rt5_prefix_notation_support():
    """RT-5: Prefix notation SH600519 / SZ000001 normalized to suffix canonical."""
    assert canonicalize_symbol("SH600519").canonical_symbol == "600519.SH"
    assert canonicalize_symbol("SZ000001").canonical_symbol == "000001.SZ"


def test_rt5_unknown_prefix_quarantined():
    """RT-5: 6-digit code with unmappable prefix is marked UNMAPPABLE."""
    res = canonicalize_symbol("123456")
    assert res.status == CanonicalStatus.UNMAPPABLE
    assert res.reason == UnmappableReason.UNKNOWN_PREFIX.value
    assert res.canonical_symbol is None


# ============================================================================
# RT-6: Beijing Stock Exchange (BSE) Pool Exclusion
# ============================================================================

def test_rt6_bse_8xxxxx_marked_bj_and_excluded():
    """RT-6: 8xxxxx marked .BJ and excluded by pool rule (counted, not silent drop)."""
    res = canonicalize_symbol("830001")
    assert res.canonical_symbol == "830001.BJ"
    assert res.status == CanonicalStatus.EXCLUDED_BSE
    assert res.reason == "bse_excluded"
    assert res.is_excluded_bse is True
    assert res.is_canonical is False

    # Already suffixed .BJ
    res_suffixed = canonicalize_symbol("830001.BJ")
    assert res_suffixed.canonical_symbol == "830001.BJ"
    assert res_suffixed.status == CanonicalStatus.EXCLUDED_BSE


def test_rt6_bse_4xxxxx_and_920xxx():
    """RT-6: 4xxxxx and 920xxx marked .BJ and excluded."""
    res_4 = canonicalize_symbol("430002")
    assert res_4.canonical_symbol == "430002.BJ"
    assert res_4.status == CanonicalStatus.EXCLUDED_BSE

    res_920 = canonicalize_symbol("920001")
    assert res_920.canonical_symbol == "920001.BJ"
    assert res_920.status == CanonicalStatus.EXCLUDED_BSE


def test_rt6_bse_dataset_isolation_and_counting():
    """RT-6: In dataset processing, BSE stocks land in bse_excluded_records with counts."""
    records = [
        {"id": "bse_1", "symbol": "830001"},
        {"id": "bse_2", "symbol": "430002.BJ"},
        {"id": "sz_1", "symbol": "000001.SZ"},
    ]
    result = process_symbols_dataset(records)
    assert result.total_count == 3
    assert result.canonical_count == 1
    assert result.bse_excluded_count == 2
    assert result.unmappable_count == 0

    assert [r["id"] for r in result.bse_excluded_records] == ["bse_1", "bse_2"]
    for r in result.bse_excluded_records:
        assert r["_canonical_status"] == CanonicalStatus.EXCLUDED_BSE.value
        assert r["_canonical_reason"] == "bse_excluded"


# ============================================================================
# RT-7: Collision and Duplicate Checking (Surface Actively)
# ============================================================================

def test_rt7_collision_actively_surfaced():
    """RT-7: Actively surface conflict count and details, not silent."""
    symbols = ["000001", "000001.SZ", "600519", "600519.SH", "000858.SZ"]
    collisions = find_symbol_collisions(symbols)

    assert len(collisions) == 2
    assert "000001.SZ" in collisions
    assert "600519.SH" in collisions

    info = collisions["000001.SZ"]
    assert info["canonical_symbol"] == "000001.SZ"
    assert set(info["raw_symbols"]) == {"000001", "000001.SZ"}
    assert info["total_occurrences"] == 2


def test_rt7_check_collision_and_duplicates_diagnostic():
    """RT-7: check_collision_and_duplicates reports both symbol collisions and duplicate IDs."""
    records = [
        {"id": "rec_dup", "symbol": "000001"},
        {"id": "rec_dup", "symbol": "000001.SZ"},
        {"id": "rec_unique", "symbol": "600519.SH"},
    ]
    diag = check_collision_and_duplicates(records)
    assert diag["has_collisions"] is True
    assert diag["collision_count"] == 1
    assert "000001.SZ" in diag["collisions"]

    assert diag["has_duplicate_ids"] is True
    assert diag["duplicate_id_count"] == 1
    assert diag["duplicate_ids"] == {"rec_dup": 2}


# ============================================================================
# Edge Cases & Invariant Verifications
# ============================================================================

def test_suffix_prefix_mismatch_quarantined():
    """If suffix contradicts deterministic prefix (e.g. 600519.SZ), quarantine as mismatch."""
    res = canonicalize_symbol("600519.SZ")
    assert res.status == CanonicalStatus.UNMAPPABLE
    assert res.reason == UnmappableReason.SUFFIX_MISMATCH.value
    assert res.canonical_symbol is None

    res_sz = canonicalize_symbol("000001.SH")
    assert res_sz.status == CanonicalStatus.UNMAPPABLE
    assert res_sz.reason == UnmappableReason.SUFFIX_MISMATCH.value


def test_normalize_symbol_helper():
    """normalize_symbol returns canonical string for CANONICAL and None for others."""
    assert normalize_symbol("600519") == "600519.SH"
    assert normalize_symbol("000001") == "000001.SZ"
    assert normalize_symbol("830001") is None  # BSE excluded from trading pool
    assert normalize_symbol("") is None
    assert normalize_symbol("AGENT") is None


def test_full_reproduction_of_1408_reports_scenario():
    """Recreate the exact composition of the 1408 reports table to prove complete alignment.

    - 32 empty ''
    - 2 illegal: 'AGENT', 'AUUSDO'
    - 3 collisions: 000001 (2) & 000001.SZ (57)
                    600519 (1) & 600519.SH (739)
                    603259 (1) & 603259.SH (11)
    - Total = 32 + 2 + 2 + 57 + 1 + 739 + 1 + 11 = 845 rows in this subset
    """
    records = []
    # 32 empty
    for i in range(32):
        records.append({"id": f"empty_{i}", "symbol": ""})
    # 2 illegal
    records.append({"id": "agent", "symbol": "AGENT"})
    records.append({"id": "auusdo", "symbol": "AUUSDO"})
    # Collision 1: 000001
    for i in range(2):
        records.append({"id": f"c1_bare_{i}", "symbol": "000001"})
    for i in range(57):
        records.append({"id": f"c1_canon_{i}", "symbol": "000001.SZ"})
    # Collision 2: 600519
    records.append({"id": "c2_bare", "symbol": "600519"})
    for i in range(739):
        records.append({"id": f"c2_canon_{i}", "symbol": "600519.SH"})
    # Collision 3: 603259
    records.append({"id": "c3_bare", "symbol": "603259"})
    for i in range(11):
        records.append({"id": f"c3_canon_{i}", "symbol": "603259.SH"})

    assert len(records) == 845

    result = process_symbols_dataset(records)

    assert result.total_count == 845
    assert result.unmappable_count == 34
    assert result.unmappable_by_reason == {
        UnmappableReason.EMPTY.value: 32,
        UnmappableReason.INVALID_FORMAT.value: 2,
    }
    assert result.bse_excluded_count == 0
    assert result.canonical_count == 811
    assert result.collision_count == 3
    assert set(result.collisions.keys()) == {"000001.SZ", "600519.SH", "603259.SH"}

    assert result.collisions["000001.SZ"]["raw_counts"] == {"000001": 2, "000001.SZ": 57}
    assert result.collisions["600519.SH"]["raw_counts"] == {"600519": 1, "600519.SH": 739}
    assert result.collisions["603259.SH"]["raw_counts"] == {"603259": 1, "603259.SH": 11}
