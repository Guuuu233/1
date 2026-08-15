"""Regression coverage for CN stock-map cache "failure freeze" (DAV-92).

Root cause (Hermes, 2026-08-06): when AkShare rate-limits the service on
startup, ``_load_cn_stock_map`` writes an empty ``{}`` placeholder into the
global cache and the failure path never records a retry deadline. While the
placeholder stands, ``_get_reverse_stock_map_cached_only`` serves ``{}`` to
list pages (Chinese names disappear) and every explicit load call re-hammers
the rate-limited AkShare endpoint.

Fix: a failed load records ``_cn_stock_map_last_failure_at`` and is NOT treated
as a valid cache; the next real fetch is deferred by
``_STOCK_MAP_FAILURE_RETRY_INTERVAL`` (default 30 min, configurable via
``TA_STOCK_MAP_RETRY_INTERVAL``) instead of being retried on every call or
frozen for the full 7-day TTL. A successful load starts the 7-day TTL and
clears the failure marker.
"""

import sys
import threading
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from api import main


# ── helpers ────────────────────────────────────────────────────────────────


def _reset_cold():
    """Clear every stock-map global to the cold-start state."""
    main._cn_stock_map = None
    main._cn_stock_reverse_map = None
    main._cn_stock_map_norm = None
    main._cn_stock_map_norm_src = None
    main._cn_stock_map_loaded_at = 0
    main._cn_stock_map_last_failure_at = 0


@pytest.fixture(autouse=True)
def _isolate_stock_map_globals():
    """Snapshot/restore stock-map globals so cache mutations never leak."""
    saved = (
        main._cn_stock_map,
        main._cn_stock_reverse_map,
        main._cn_stock_map_norm,
        main._cn_stock_map_norm_src,
        main._cn_stock_map_loaded_at,
        getattr(main, "_cn_stock_map_last_failure_at", 0),
    )
    yield
    (
        main._cn_stock_map,
        main._cn_stock_reverse_map,
        main._cn_stock_map_norm,
        main._cn_stock_map_norm_src,
        main._cn_stock_map_loaded_at,
        main._cn_stock_map_last_failure_at,
    ) = saved


def _akshare_stub(stock_behavior, fund_behavior=None):
    """Akshare stub; a DataFrame is a successful return, otherwise the behavior
    is wired through MagicMock.side_effect (exception or list of results)."""
    ak = MagicMock()
    if isinstance(stock_behavior, pd.DataFrame):
        ak.stock_info_a_code_name.return_value = stock_behavior
    else:
        ak.stock_info_a_code_name.side_effect = stock_behavior
    if fund_behavior is None:
        fund_behavior = RuntimeError("no fund data")
    if isinstance(fund_behavior, pd.DataFrame):
        ak.fund_name_em.return_value = fund_behavior
    else:
        ak.fund_name_em.side_effect = fund_behavior
    return ak


def _stock_df(*rows):
    return pd.DataFrame(list(rows), columns=["name", "code"])


# ── failure retry (requirement 1) ──────────────────────────────────────────


def test_failed_load_returns_empty_but_retries_after_interval_not_ttl():
    """Failure returns {} + records a retry deadline; a later call retries
    after the short interval instead of being frozen for the 7-day TTL."""
    _reset_cold()
    ak = _akshare_stub(RuntimeError("rate limited"))
    with patch.dict(sys.modules, {"akshare": ak}):
        assert main._load_cn_stock_map() == {}
        assert main._cn_stock_map_last_failure_at > 0
        assert main._cn_stock_map_loaded_at == 0  # failure must NOT start the TTL

        # Within the retry window, akshare is NOT hammered again.
        ak.stock_info_a_code_name.reset_mock()
        assert main._load_cn_stock_map() == {}
        assert ak.stock_info_a_code_name.call_count == 0

        # After the retry interval, a real retry happens (not a 7-day freeze).
        main._cn_stock_map_last_failure_at -= main._STOCK_MAP_FAILURE_RETRY_INTERVAL + 1
        assert main._load_cn_stock_map() == {}
        assert ak.stock_info_a_code_name.call_count >= 1


def test_empty_provider_results_use_failure_backoff_then_recover():
    """Empty stock and fund responses are failures, not a successful 7-day cache."""
    _reset_cold()
    empty_stock = pd.DataFrame(columns=["name", "code"])
    empty_fund = pd.DataFrame(columns=["基金代码", "基金简称"])
    recovered_stock = _stock_df(("贵州茅台", "600519"))
    recovered_fund = pd.DataFrame([{"基金代码": "159915", "基金简称": "创业板ETF"}])
    ak = _akshare_stub(
        stock_behavior=[empty_stock, recovered_stock],
        fund_behavior=[empty_fund, recovered_fund],
    )

    with patch.dict(sys.modules, {"akshare": ak}):
        assert main._load_cn_stock_map() == {}
        assert main._cn_stock_map_loaded_at == 0
        assert main._cn_stock_map_last_failure_at > 0

        # The empty response is subject to the same short retry backoff as an
        # exception; it must not call either provider again immediately.
        ak.stock_info_a_code_name.reset_mock()
        ak.fund_name_em.reset_mock()
        assert main._load_cn_stock_map() == {}
        assert ak.stock_info_a_code_name.call_count == 0
        assert ak.fund_name_em.call_count == 0

        # Once the backoff expires, non-empty provider responses recover the
        # cache and start the success TTL.
        main._cn_stock_map_last_failure_at -= main._STOCK_MAP_FAILURE_RETRY_INTERVAL + 1
        result = main._load_cn_stock_map()
        assert result == {
            "贵州茅台": "600519.SH",
            "创业板ETF": "159915.SZ",
        }
        assert main._cn_stock_reverse_map == {
            "600519.SH": "贵州茅台",
            "159915.SZ": "创业板ETF",
        }
        assert main._cn_stock_map_loaded_at > 0
        assert main._cn_stock_map_last_failure_at == 0


def test_empty_stock_source_with_fund_rows_still_uses_failure_backoff():
    """A fund-only partial response must not start the success TTL."""
    _reset_cold()
    empty_stock = pd.DataFrame(columns=["name", "code"])
    fund_only = pd.DataFrame([{"基金代码": "159915", "基金简称": "创业板ETF"}])
    recovered_stock = _stock_df(("贵州茅台", "600519"))
    recovered_fund = pd.DataFrame([{"基金代码": "159915", "基金简称": "创业板ETF"}])
    ak = _akshare_stub(
        stock_behavior=[empty_stock, recovered_stock],
        fund_behavior=[fund_only, recovered_fund],
    )

    with patch.dict(sys.modules, {"akshare": ak}):
        assert main._load_cn_stock_map() == {}
        assert main._cn_stock_map_loaded_at == 0
        assert main._cn_stock_map_last_failure_at > 0

        ak.stock_info_a_code_name.reset_mock()
        ak.fund_name_em.reset_mock()
        assert main._load_cn_stock_map() == {}
        assert ak.stock_info_a_code_name.call_count == 0
        assert ak.fund_name_em.call_count == 0

        main._cn_stock_map_last_failure_at -= main._STOCK_MAP_FAILURE_RETRY_INTERVAL + 1
        assert main._load_cn_stock_map() == {
            "贵州茅台": "600519.SH",
            "创业板ETF": "159915.SZ",
        }
        assert main._cn_stock_map_loaded_at > 0
        assert main._cn_stock_map_last_failure_at == 0


def test_failed_cold_cache_cached_only_returns_empty():
    """Requirement 4: after a cold failure, cached-only list lookups fast-return
    empty (no blocking load)."""
    _reset_cold()
    ak = _akshare_stub(RuntimeError("rate limited"))
    with patch.dict(sys.modules, {"akshare": ak}):
        main._load_cn_stock_map()
        assert main._get_reverse_stock_map_cached_only() == {}


# ── fail then success (requirement 2) ──────────────────────────────────────


def test_failure_then_success_loads_and_fills_reverse_map():
    """Akshare recovers -> the retry fills the reverse map and clears the
    failure marker, so the 7-day TTL starts only after a real success."""
    _reset_cold()
    ak = _akshare_stub(
        stock_behavior=[
            RuntimeError("rate limited"),
            _stock_df(("贵州茅台", "600519")),
        ]
    )
    with patch.dict(sys.modules, {"akshare": ak}):
        assert main._load_cn_stock_map() == {}
        assert main._cn_stock_reverse_map == {}

        main._cn_stock_map_last_failure_at -= main._STOCK_MAP_FAILURE_RETRY_INTERVAL + 1
        result = main._load_cn_stock_map()
        assert result == {"贵州茅台": "600519.SH"}
        assert main._cn_stock_reverse_map == {"600519.SH": "贵州茅台"}
        assert main._cn_stock_map_last_failure_at == 0
        assert main._cn_stock_map_loaded_at > 0


# ── success TTL regression (requirement 3) ─────────────────────────────────


def test_success_caches_for_ttl_then_refetches():
    """Successful load is cached; within the TTL no refetch; after TTL expiry a
    refetch happens (existing 7-day TTL behavior must not regress)."""
    _reset_cold()
    ak = _akshare_stub(stock_behavior=_stock_df(("贵州茅台", "600519")))
    with patch.dict(sys.modules, {"akshare": ak}):
        assert main._load_cn_stock_map() == {"贵州茅台": "600519.SH"}

        ak.stock_info_a_code_name.reset_mock()
        assert main._load_cn_stock_map() == {"贵州茅台": "600519.SH"}
        assert ak.stock_info_a_code_name.call_count == 0

        # Force TTL expiry: the next call must refetch from akshare.
        main._cn_stock_map_loaded_at -= main._STOCK_MAP_TTL + 1
        assert main._load_cn_stock_map() == {"贵州茅台": "600519.SH"}
        assert ak.stock_info_a_code_name.call_count == 1


# ── concurrency (requirement 4) ────────────────────────────────────────────


def test_concurrent_cold_loads_fetch_only_once():
    """Concurrent cold calls trigger a single real load (double-checked locking
    must not regress)."""
    _reset_cold()
    ak = _akshare_stub(stock_behavior=_stock_df(("贵州茅台", "600519")))
    results = []
    errors = []
    barrier = threading.Barrier(5)

    def _load():
        try:
            barrier.wait(timeout=10)
            results.append(main._load_cn_stock_map())
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    with patch.dict(sys.modules, {"akshare": ak}):
        threads = [threading.Thread(target=_load) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

    assert not errors
    assert ak.stock_info_a_code_name.call_count == 1
    assert all(r == {"贵州茅台": "600519.SH"} for r in results)
    assert len(results) == 5


# ── derived normalized view (requirement 5) ────────────────────────────────


def test_normalized_view_rebuilds_after_failure_recovery():
    """``_cn_stock_map_norm`` is a derived view keyed to the source map object;
    after a failure it must not keep serving a stale empty map once a retry
    succeeds."""
    _reset_cold()
    ak = _akshare_stub(
        stock_behavior=[
            RuntimeError("rate limited"),
            _stock_df(("京东方Ａ", "000725")),
        ]
    )
    with patch.dict(sys.modules, {"akshare": ak}):
        assert main._get_normalized_stock_map() == {}

        main._cn_stock_map_last_failure_at -= main._STOCK_MAP_FAILURE_RETRY_INTERVAL + 1
        norm = main._get_normalized_stock_map()
        # NFKC normalization converts the full-width share-class letter (京东方Ａ)
        # to half-width (京东方A) — that is the point of the derived view.
        assert norm == {"京东方A": "000725.SZ"}
