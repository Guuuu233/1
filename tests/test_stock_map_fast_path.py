import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pandas as pd


def test_reverse_stock_map_cached_only_does_not_trigger_cold_load():
    from api import main as main_mod

    original_map = main_mod._cn_stock_map
    try:
        main_mod._cn_stock_map = None
        with patch.object(main_mod, "_load_cn_stock_map", side_effect=AssertionError("slow load should not run")):
            assert main_mod._get_reverse_stock_map_cached_only() == {}
    finally:
        main_mod._cn_stock_map = original_map


def test_reverse_stock_map_cached_only_uses_existing_cache():
    from api import main as main_mod

    original_map = main_mod._cn_stock_map
    try:
        main_mod._cn_stock_map = {
            "贵州茅台": "600519.SH",
            "宁德时代": "300750.SZ",
        }
        assert main_mod._get_reverse_stock_map_cached_only() == {
            "600519.SH": "贵州茅台",
            "300750.SZ": "宁德时代",
        }
    finally:
        main_mod._cn_stock_map = original_map


def test_report_refresh_does_not_wait_for_blocked_provider():
    """A report lookup returns while its background provider is blocked."""
    from api import main as main_mod

    saved = (
        main_mod._cn_stock_map,
        main_mod._cn_stock_reverse_map,
        main_mod._cn_stock_map_norm,
        main_mod._cn_stock_map_norm_src,
        main_mod._cn_stock_map_loaded_at,
        main_mod._cn_stock_map_last_failure_at,
        main_mod._cn_stock_map_refresh_inflight,
        getattr(main_mod, "_cn_stock_map_refresh_event", None),
    )
    provider_started = threading.Event()
    release_provider = threading.Event()
    provider_returned = threading.Event()
    request_finished = threading.Event()
    request_results = []
    request_thread = None

    ak = MagicMock()

    def _blocked_stock_source():
        provider_started.set()
        release_provider.wait(timeout=5)
        provider_returned.set()
        return pd.DataFrame([("贵州茅台", "600519")], columns=["name", "code"])

    ak.stock_info_a_code_name.side_effect = _blocked_stock_source
    ak.fund_name_em.return_value = pd.DataFrame(columns=["基金代码", "基金简称"])

    try:
        main_mod._cn_stock_map = None
        main_mod._cn_stock_reverse_map = None
        main_mod._cn_stock_map_norm = None
        main_mod._cn_stock_map_norm_src = None
        main_mod._cn_stock_map_loaded_at = 0
        main_mod._cn_stock_map_last_failure_at = 0
        main_mod._cn_stock_map_refresh_inflight = False
        if hasattr(main_mod, "_cn_stock_map_refresh_event"):
            main_mod._cn_stock_map_refresh_event = threading.Event()
            main_mod._cn_stock_map_refresh_event.set()

        with patch.dict(sys.modules, {"akshare": ak}):
            assert main_mod._get_report_reverse_stock_map() == {}
            assert provider_started.wait(timeout=5)

            def _request_report_names():
                request_results.append(main_mod._get_report_reverse_stock_map())
                request_finished.set()

            request_thread = threading.Thread(target=_request_report_names)
            request_thread.start()
            assert request_finished.wait(timeout=1), (
                "report lookup waited for the provider-held cache lock"
            )
            assert request_results == [{}]
    finally:
        release_provider.set()
        if request_thread is not None:
            request_thread.join(timeout=5)
        provider_returned.wait(timeout=5)
        if hasattr(main_mod, "_cn_stock_map_refresh_inflight"):
            for _ in range(500):
                if not main_mod._cn_stock_map_refresh_inflight:
                    break
                time.sleep(0.01)
        (
            main_mod._cn_stock_map,
            main_mod._cn_stock_reverse_map,
            main_mod._cn_stock_map_norm,
            main_mod._cn_stock_map_norm_src,
            main_mod._cn_stock_map_loaded_at,
            main_mod._cn_stock_map_last_failure_at,
            main_mod._cn_stock_map_refresh_inflight,
            saved_event,
        ) = saved
        if saved_event is not None:
            main_mod._cn_stock_map_refresh_event = saved_event
