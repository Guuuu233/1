import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.providers.yfinance_provider import YFinanceProvider
from tradingagents.graph.data_collector import (
    DataCollector,
    _build_data_failure_ledger,
    _build_source_provenance,
    _fetch_all,
)


class TestMacroMarketDataflows(unittest.TestCase):

    def setUp(self):
        super().setUp()
        CnAkshareProvider.clear_macro_cache()

    def tearDown(self):
        super().tearDown()
        CnAkshareProvider.clear_macro_cache()

    def test_akshare_get_cn_indices_success_and_anti_lookahead(self):
        provider = CnAkshareProvider()
        mock_ak = MagicMock()

        # Mock daily index history with bars before and after curr_date
        dates = ["2026-08-01", "2026-08-05", "2026-08-10", "2026-08-15"]
        prices = [3000.0, 3050.0, 3100.0, 3200.0]
        df_hist = pd.DataFrame({"日期": dates, "开盘": prices, "收盘": prices, "最高": prices, "最低": prices, "成交量": [1000]*4})

        mock_ak.index_zh_a_hist.return_value = df_hist

        with patch.object(provider, "_ak", return_value=mock_ak):
            result = provider.get_cn_indices(curr_date="2026-08-10", look_back_days=30)

        assert "## 国内核心大盘指数行情" in result
        assert "【数据日期】2026-08-10" in result
        assert "3100.00" in result
        # 2026-08-15 price (3200.00) must NOT appear in output because it is past 2026-08-10
        assert "3200.00" not in result

    def test_akshare_get_cn_indices_failure_handling(self):
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        mock_ak.index_zh_a_hist.side_effect = RuntimeError("Connection timed out")
        mock_ak.stock_zh_index_daily_em.side_effect = RuntimeError("EM API error")
        mock_ak.stock_zh_index_daily_tx.side_effect = RuntimeError("TX API error")

        with patch.object(provider, "_ak", return_value=mock_ak):
            result = provider.get_cn_indices(curr_date="2026-08-10")

        assert result.startswith("【数据获取失败】国内核心大盘指数")
        assert "所有国内指数接口调用失败" in result

    def test_akshare_get_global_indices_success(self):
        provider = CnAkshareProvider()
        mock_ak = MagicMock()

        dates = ["2026-08-08", "2026-08-09", "2026-08-10"]
        df_hist = pd.DataFrame({"date": dates, "close": [5400.0, 5420.0, 5450.0], "open": [5400.0]*3, "high": [5460.0]*3, "low": [5390.0]*3, "volume": [1000]*3})
        mock_ak.index_global_hist_em.return_value = df_hist

        with patch.object(provider, "_ak", return_value=mock_ak):
            result = provider.get_global_indices(curr_date="2026-08-10")

        assert "## 全球核心市场指数行情" in result
        assert "【数据日期】2026-08-10" in result
        assert "5450.00" in result

    def test_akshare_get_major_assets_success(self):
        provider = CnAkshareProvider()
        mock_ak = MagicMock()

        dates = ["2026-08-08", "2026-08-09", "2026-08-10"]
        df_futures = pd.DataFrame({"date": dates, "close": [2400.0, 2420.0, 2450.0], "open": [2400.0]*3, "high": [2460.0]*3, "low": [2390.0]*3, "volume": [100]*3})
        mock_ak.futures_foreign_hist.return_value = df_futures

        df_bond = pd.DataFrame({"日期": dates, "美国国债收益率10年": [4.30, 4.28, 4.25]})
        mock_ak.bond_zh_us_rate.return_value = df_bond

        with patch.object(provider, "_ak", return_value=mock_ak):
            result = provider.get_major_assets(curr_date="2026-08-10")

        assert "## 全球大类资产与宏观大宗商品" in result
        assert "【数据日期】2026-08-10" in result
        assert "2450.00" in result
        assert "4.250%" in result

    def test_yfinance_macro_methods(self):
        provider = YFinanceProvider()

        dates = pd.to_datetime(["2026-08-08", "2026-08-09", "2026-08-10"])
        df_hist = pd.DataFrame(
            {"Open": [100.0]*3, "High": [105.0]*3, "Low": [95.0]*3, "Close": [100.0, 102.0, 105.0], "Volume": [500]*3},
            index=dates,
        )
        df_hist.index.name = "Date"

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df_hist

        with patch("yfinance.Ticker", return_value=mock_ticker):
            cn_res = provider.get_cn_indices(curr_date="2026-08-10")
            gl_res = provider.get_global_indices(curr_date="2026-08-10")
            ma_res = provider.get_major_assets(curr_date="2026-08-10")

        assert "## 国内核心大盘指数行情" in cn_res
        assert "105.00" in cn_res
        assert "## 全球核心市场指数行情" in gl_res
        assert "## 全球大类资产与宏观大宗商品" in ma_res

    def test_route_to_vendor_refuses_invalid_or_future_as_of(self):
        # Missing date
        res = route_to_vendor("get_cn_indices", None)
        assert "【数据获取失败】" in res or "未指定" in res or "不可用" in res

        # Future date
        res = route_to_vendor("get_global_indices", "2099-01-01")
        assert "【数据获取失败】" in res or "未来日期" in res or "不可用" in res

    def test_collector_integrates_macro_views_and_failure_ledger(self):
        results = {
            "stock_data": "Date,Open,High,Low,Close,Volume\n2026-08-10,10,10,10,10,100",
            "cn_indices": "## 国内核心大盘指数行情（数据基准日：2026-08-10，来源：cn_akshare）\n【数据日期】2026-08-10",
            "global_indices": "【数据获取失败】全球核心指数 — 原因：所有全球指数接口调用失败 (来源: cn_akshare)",
            "major_assets": "## 全球大类资产与宏观大宗商品（数据基准日：2026-08-10，来源：cn_akshare）\n【数据日期】2026-08-10",
        }

        ledger = _build_data_failure_ledger(results)
        # global_indices failed, so it should be in failure ledger
        ledger_sources = [entry["source"] for entry in ledger]
        assert "global_indices" in ledger_sources

        # cn_indices and major_assets succeeded, so they should NOT be in failure ledger
        assert "cn_indices" not in ledger_sources
        assert "major_assets" not in ledger_sources

        provenance = _build_source_provenance(results, "2026-08-10", daily_as_of="2026-08-10")
        assert provenance["cn_indices"]["as_of"] == "2026-08-10"
        assert provenance["cn_indices"]["status"] == "available"
        assert provenance["major_assets"]["as_of"] == "2026-08-10"
        assert provenance["global_indices"]["status"] == "failed"

    def test_akshare_macro_cache_hit_and_ttl(self):
        provider = CnAkshareProvider()
        mock_ak = MagicMock()

        dates = ["2026-08-01", "2026-08-05", "2026-08-10"]
        prices = [3000.0, 3050.0, 3100.0]
        df_hist = pd.DataFrame({"日期": dates, "开盘": prices, "收盘": prices, "最高": prices, "最低": prices, "成交量": [1000]*3})
        mock_ak.index_zh_a_hist.return_value = df_hist

        with patch.object(provider, "_ak", return_value=mock_ak):
            # First call: populates cache
            res1 = provider.get_cn_indices(curr_date="2026-08-10", look_back_days=30)
            assert "## 国内核心大盘指数行情" in res1
            call_count_1 = mock_ak.index_zh_a_hist.call_count
            assert call_count_1 > 0

            # Second call: should hit cache without calling _ak
            res2 = provider.get_cn_indices(curr_date="2026-08-10", look_back_days=30)
            assert res1 == res2
            assert mock_ak.index_zh_a_hist.call_count == call_count_1

        # Test TTL expiration
        import time as _time
        real_monotonic = _time.monotonic
        with patch.object(provider, "_ak", return_value=mock_ak):
            # Advance time by 901s (past 900s / 15-min TTL)
            with patch("time.monotonic", side_effect=lambda: real_monotonic() + 901.0):
                res3 = provider.get_cn_indices(curr_date="2026-08-10", look_back_days=30)
                assert res3 == res1
                # Should have fetched again
                assert mock_ak.index_zh_a_hist.call_count > call_count_1

    def test_akshare_macro_cache_does_not_cache_failure(self):
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        mock_ak.index_zh_a_hist.side_effect = RuntimeError("network error")
        mock_ak.stock_zh_index_daily_em.side_effect = RuntimeError("network error")

        with patch.object(provider, "_ak", return_value=mock_ak):
            res_fail = provider.get_cn_indices(curr_date="2026-08-10")
            assert res_fail.startswith("【数据获取失败】")

        # Now mock succeeds
        dates = ["2026-08-01", "2026-08-05", "2026-08-10"]
        prices = [3000.0, 3050.0, 3100.0]
        df_hist = pd.DataFrame({"日期": dates, "开盘": prices, "收盘": prices, "最高": prices, "最低": prices, "成交量": [1000]*3})
        mock_ak.index_zh_a_hist.side_effect = None
        mock_ak.index_zh_a_hist.return_value = df_hist

        with patch.object(provider, "_ak", return_value=mock_ak):
            res_success = provider.get_cn_indices(curr_date="2026-08-10")
            assert "## 国内核心大盘指数行情" in res_success

    def test_akshare_get_cn_indices_windowed_request(self):
        """腾讯源（stock_zh_index_daily_tx）作为首选接口，验证 7 个指数均调用一次。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()

        dates = ["2026-08-01", "2026-08-05", "2026-08-10"]
        prices = [3000.0, 3050.0, 3100.0]
        df_hist = pd.DataFrame({"日期": dates, "开盘": prices, "收盘": prices, "最高": prices, "最低": prices, "成交量": [1000]*3})
        mock_ak.stock_zh_index_daily_tx.return_value = df_hist

        with patch.object(provider, "_ak", return_value=mock_ak):
            res = provider.get_cn_indices(curr_date="2026-08-10", look_back_days=30)

        # 首选腾讯源应被调用 7 次（每个指数一次）
        assert mock_ak.stock_zh_index_daily_tx.call_count == 7
        assert "## 国内核心大盘指数行情" in res

    def test_akshare_get_cn_indices_fallback_windowed_request(self):
        """腾讯源和东财历史接口都失败时，三级备选 stock_zh_index_daily_em 应带 1 年窗口参数。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        # 腾讯源和东财历史接口均失败
        mock_ak.stock_zh_index_daily_tx.return_value = None
        mock_ak.index_zh_a_hist.return_value = None

        dates = ["2026-08-01", "2026-08-05", "2026-08-10"]
        prices = [3000.0, 3050.0, 3100.0]
        df_hist = pd.DataFrame({"日期": dates, "开盘": prices, "收盘": prices, "最高": prices, "最低": prices, "成交量": [1000]*3})
        mock_ak.stock_zh_index_daily_em.return_value = df_hist

        with patch.object(provider, "_ak", return_value=mock_ak):
            res = provider.get_cn_indices(curr_date="2026-08-10", look_back_days=30)

        assert "## 国内核心大盘指数行情" in res
        assert mock_ak.stock_zh_index_daily_em.call_count == 7
        # 窗口为 max(look_back_days=30, 365) = 365 天，start = 2026-08-10 - 365 = 2025-08-10
        for call_args in mock_ak.stock_zh_index_daily_em.call_args_list:
            _, kwargs = call_args
            assert kwargs["start_date"] == "20250810"
            assert kwargs["end_date"] == "20260810"

    def test_akshare_macro_lock_granularity(self):
        """验证锁粒度为单次网络请求级别：7 个指数 × 首选接口 = 7 次 lock enter/exit。"""
        from tradingagents.dataflows.providers.cn_akshare_provider import AKSHARE_CALL_LOCK

        provider = CnAkshareProvider()
        mock_ak = MagicMock()

        dates = ["2026-08-01", "2026-08-05", "2026-08-10"]
        prices = [3000.0, 3050.0, 3100.0]
        df_hist = pd.DataFrame({"日期": dates, "开盘": prices, "收盘": prices, "最高": prices, "最低": prices, "成交量": [1000]*3})
        # 腾讯源首选接口成功返回，每个指数只进一次锁
        mock_ak.stock_zh_index_daily_tx.return_value = df_hist

        enter_count = 0
        exit_count = 0
        orig_enter = AKSHARE_CALL_LOCK.__class__.__enter__
        orig_exit = AKSHARE_CALL_LOCK.__class__.__exit__

        def tracked_enter(self_lock):
            nonlocal enter_count
            enter_count += 1
            return orig_enter(self_lock)

        def tracked_exit(self_lock, *args):
            nonlocal exit_count
            exit_count += 1
            return orig_exit(self_lock, *args)

        with patch.object(AKSHARE_CALL_LOCK.__class__, "__enter__", tracked_enter), \
             patch.object(AKSHARE_CALL_LOCK.__class__, "__exit__", tracked_exit), \
             patch.object(provider, "_ak", return_value=mock_ak):
            provider.get_cn_indices(curr_date="2026-08-10", look_back_days=30)

        # 首选腾讯源成功：7 个指数各进锁一次，不超过 7
        assert enter_count == 7
        assert exit_count == 7

    def test_akshare_macro_concurrent_threads_and_cache_sharing(self):
        import concurrent.futures

        provider = CnAkshareProvider()
        mock_ak = MagicMock()

        dates = ["2026-08-01", "2026-08-05", "2026-08-10"]
        prices = [3000.0, 3050.0, 3100.0]
        df_hist = pd.DataFrame({"日期": dates, "开盘": prices, "收盘": prices, "最高": prices, "最低": prices, "成交量": [1000]*3})
        mock_ak.index_zh_a_hist.return_value = df_hist
        mock_ak.index_global_hist_em.return_value = pd.DataFrame({"date": dates, "close": prices})
        mock_ak.futures_foreign_hist.return_value = pd.DataFrame({"date": dates, "close": prices})
        mock_ak.bond_zh_us_rate.return_value = pd.DataFrame({"日期": dates, "美国国债收益率10年": [4.3, 4.28, 4.25]})

        with patch.object(CnAkshareProvider, "_ak", return_value=mock_ak):
            # Run 10 parallel queries across 3 methods
            def _fetch_all_macro():
                p = CnAkshareProvider()
                return (
                    p.get_cn_indices(curr_date="2026-08-10", look_back_days=30),
                    p.get_global_indices(curr_date="2026-08-10", look_back_days=30),
                    p.get_major_assets(curr_date="2026-08-10", look_back_days=30),
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                futures = [pool.submit(_fetch_all_macro) for _ in range(10)]
                results = [f.result() for f in futures]

            # All 10 results should be identical and successful
            for cn, gl, ma in results:
                assert "## 国内核心大盘指数行情" in cn
                assert "## 全球核心市场指数行情" in gl
                assert "## 全球大类资产与宏观大宗商品" in ma
                assert cn == results[0][0]
                assert gl == results[0][1]
                assert ma == results[0][2]
