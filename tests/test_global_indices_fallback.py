import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.providers.yfinance_provider import YFinanceProvider
from tradingagents.dataflows.macro_market_utils import (
    calculate_series_metrics,
    build_global_indices_markdown,
)


class TestGlobalIndicesFallback(unittest.TestCase):

    def setUp(self):
        super().setUp()
        CnAkshareProvider.clear_macro_cache()

    def tearDown(self):
        super().tearDown()
        CnAkshareProvider.clear_macro_cache()

    def test_mock_eastmoney_hist_fail_and_ulist_success(self):
        """Mock 东财 hist 失败 + ulist 成功 → 部分/全部指数有真值。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        mock_ak.index_global_hist_em.side_effect = RuntimeError("RemoteDisconnected")
        mock_ak.stock_hk_index_daily_em.side_effect = RuntimeError("RemoteDisconnected")

        mock_ulist_data = {
            "标普500": {
                "name": "标普500",
                "code": "SPX",
                "latest_close": 5600.50,
                "change_1d_pct": 0.75,
                "change_5d_pct": None,
                "change_20d_pct": None,
                "as_of": "2026-08-21",
                "period_kind": "session_snapshot",
                "retrieved_at": "2026-08-21T16:00:00Z",
                "source": "eastmoney_ulist",
                "trend_desc": "上涨反弹",
            },
            "纳斯达克100": {
                "name": "纳斯达克100",
                "code": "NDX",
                "latest_close": 19800.20,
                "change_1d_pct": 1.10,
                "change_5d_pct": None,
                "change_20d_pct": None,
                "as_of": "2026-08-21",
                "period_kind": "session_snapshot",
                "retrieved_at": "2026-08-21T16:00:00Z",
                "source": "eastmoney_ulist",
                "trend_desc": "上涨反弹",
            },
            "韩国KOSPI": {
                "name": "韩国KOSPI",
                "code": "KS11",
                "latest_close": 2750.80,
                "change_1d_pct": 0.45,
                "change_5d_pct": None,
                "change_20d_pct": None,
                "as_of": "2026-08-21",
                "period_kind": "session_snapshot",
                "retrieved_at": "2026-08-21T16:00:00Z",
                "source": "eastmoney_ulist",
                "trend_desc": "平稳震荡",
            },
            "德国DAX": {
                "name": "德国DAX",
                "code": "GDAXI",
                "latest_close": 18500.00,
                "change_1d_pct": -0.30,
                "change_5d_pct": None,
                "change_20d_pct": None,
                "as_of": "2026-08-21",
                "period_kind": "session_snapshot",
                "retrieved_at": "2026-08-21T16:00:00Z",
                "source": "eastmoney_ulist",
                "trend_desc": "平稳震荡",
            },
        }

        with patch.object(provider, "_fetch_global_indices_em_ulist", return_value=mock_ulist_data), \
             patch.object(provider, "_fetch_global_indices_sina_hq", return_value={}), \
             patch.object(provider, "_ak", return_value=mock_ak):
            result = provider.get_global_indices(curr_date="2026-08-21")

        assert "## 全球核心市场指数行情" in result
        assert "【数据日期】2026-08-21" in result
        assert "标普500" in result
        assert "5600.50" in result
        assert "纳斯达克100" in result
        assert "19800.20" in result
        assert "韩国KOSPI" in result
        assert "2750.80" in result
        assert "德国DAX" in result
        assert "18500.00" in result
        assert "跨市场宏观联动观察" in result

    def test_mock_sina_hq_fallback_when_ulist_fails(self):
        """Mock 东财 ulist 失败 + 新浪 hq 成功。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        mock_ak.index_global_hist_em.side_effect = RuntimeError("RemoteDisconnected")
        mock_ak.stock_hk_index_daily_em.side_effect = RuntimeError("RemoteDisconnected")

        mock_sina_data = {
            "标普500": {
                "name": "标普500",
                "code": "^GSPC",
                "latest_close": 5580.00,
                "change_1d_pct": 0.50,
                "change_5d_pct": None,
                "change_20d_pct": None,
                "as_of": "2026-08-21",
                "period_kind": "session_snapshot",
                "retrieved_at": "2026-08-21T16:00:00Z",
                "source": "sina_hq",
                "trend_desc": "上涨反弹",
            },
            "纳斯达克综合": {
                "name": "纳斯达克综合",
                "code": "^IXIC",
                "latest_close": 17800.00,
                "change_1d_pct": 0.65,
                "change_5d_pct": None,
                "change_20d_pct": None,
                "as_of": "2026-08-21",
                "period_kind": "session_snapshot",
                "retrieved_at": "2026-08-21T16:00:00Z",
                "source": "sina_hq",
                "trend_desc": "上涨反弹",
            },
            "恒生指数": {
                "name": "恒生指数",
                "code": "HSI",
                "latest_close": 17600.00,
                "change_1d_pct": 1.20,
                "change_5d_pct": None,
                "change_20d_pct": None,
                "as_of": "2026-08-21",
                "period_kind": "session_snapshot",
                "retrieved_at": "2026-08-21T16:00:00Z",
                "source": "sina_hq",
                "trend_desc": "上涨反弹",
            },
            "恒生科技指数": {
                "name": "恒生科技指数",
                "code": "HSTECH",
                "latest_close": 3600.00,
                "change_1d_pct": 1.80,
                "change_5d_pct": None,
                "change_20d_pct": None,
                "as_of": "2026-08-21",
                "period_kind": "session_snapshot",
                "retrieved_at": "2026-08-21T16:00:00Z",
                "source": "sina_hq",
                "trend_desc": "上涨反弹",
            },
            "日经225": {
                "name": "日经225",
                "code": "N225",
                "latest_close": 38000.00,
                "change_1d_pct": -0.80,
                "change_5d_pct": None,
                "change_20d_pct": None,
                "as_of": "2026-08-21",
                "period_kind": "session_snapshot",
                "retrieved_at": "2026-08-21T16:00:00Z",
                "source": "sina_hq",
                "trend_desc": "回调下跌",
            },
            "韩国KOSPI": {
                "name": "韩国KOSPI",
                "code": "KS11",
                "latest_close": 2720.00,
                "change_1d_pct": 0.35,
                "change_5d_pct": None,
                "change_20d_pct": None,
                "as_of": "2026-08-21",
                "period_kind": "session_snapshot",
                "retrieved_at": "2026-08-21T16:00:00Z",
                "source": "sina_hq",
                "trend_desc": "平稳震荡",
            },
        }

        with patch.object(provider, "_fetch_global_indices_em_ulist", return_value={}), \
             patch.object(provider, "_fetch_global_indices_sina_hq", return_value=mock_sina_data), \
             patch.object(provider, "_ak", return_value=mock_ak):
            result = provider.get_global_indices(curr_date="2026-08-21")

        assert "## 全球核心市场指数行情" in result
        assert "纳斯达克综合" in result
        assert "恒生科技指数" in result
        assert "韩国KOSPI" in result
        assert "亚太市场温度" in result

    def test_mock_all_fail_explicit_failure_text_no_hallucination(self):
        """Mock 全部失败 → 显式失败文案且不含臆造点位。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        mock_ak.index_global_hist_em.side_effect = RuntimeError("all failed")
        mock_ak.stock_hk_index_daily_em.side_effect = RuntimeError("all failed")

        with patch.object(provider, "_fetch_global_indices_em_ulist", return_value={}), \
             patch.object(provider, "_fetch_global_indices_sina_hq", return_value={}), \
             patch.object(provider, "_ak", return_value=mock_ak):
            result = provider.get_global_indices(curr_date="2026-08-21")

        assert result.startswith("【数据获取失败】全球核心指数")
        assert "所有全球指数接口调用失败" in result
        # 绝不出现臆造点位
        assert "5400" not in result
        assert "26000" not in result

    def test_anti_lookahead_rejects_snapshot_with_future_as_of(self):
        """as_of > trade_date 的快照被拒绝。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        mock_ak.index_global_hist_em.return_value = None

        # 模拟快照日期为 2026-08-22，但分析日请求 2026-08-20
        future_snapshot = {
            "标普500": {
                "name": "标普500",
                "code": "SPX",
                "latest_close": 5600.0,
                "change_1d_pct": 0.5,
                "as_of": "2026-08-22",
                "period_kind": "session_snapshot",
                "source": "eastmoney_ulist",
            }
        }

        with patch.object(provider, "_fetch_global_indices_em_ulist", return_value=future_snapshot), \
             patch.object(provider, "_fetch_global_indices_sina_hq", return_value={}), \
             patch.object(provider, "_ak", return_value=mock_ak):
            result = provider.get_global_indices(curr_date="2026-08-20")

        # 2026-08-22 的快照必须被丢弃，全失败时返回显式获取失败
        assert result.startswith("【数据获取失败】全球核心指数")
        assert "5600.0" not in result

    def test_kospi_in_success_and_missing_list(self):
        """KOSPI 出现在成功或缺失清单中。"""
        items = {
            "标普500": {
                "code": "^GSPC",
                "as_of": "2026-08-21",
                "latest_close": 5500.00,
                "change_1d_pct": 0.5,
                "change_5d_pct": None,
                "change_20d_pct": None,
            },
            "韩国KOSPI": {
                "code": "KS11",
                "as_of": "2026-08-21",
                "latest_close": 2700.00,
                "change_1d_pct": 0.8,
                "change_5d_pct": None,
                "change_20d_pct": None,
            },
            "法国CAC40": {
                "code": "FCHI",
                "latest_close": None,
            }
        }

        md = build_global_indices_markdown(items, "2026-08-21", source="cn_akshare")
        assert "韩国KOSPI" in md
        assert "2700.00" in md
        assert "法国CAC40" in md
        assert "【数据缺失】" in md
        assert "亚太市场温度" in md

    def test_nasdaq_naming_discipline(self):
        """纳指须标明“纳斯达克综合”或“纳斯达克100”，禁止模糊混称。"""
        items_ndx = {
            "纳斯达克": {
                "code": "NDX",
                "as_of": "2026-08-21",
                "latest_close": 19000.0,
                "change_1d_pct": 1.0,
            }
        }
        md_ndx = build_global_indices_markdown(items_ndx, "2026-08-21")
        assert "纳斯达克100" in md_ndx

        items_ixic = {
            "纳斯达克": {
                "code": "^IXIC",
                "as_of": "2026-08-21",
                "latest_close": 17500.0,
                "change_1d_pct": 0.5,
            }
        }
        md_ixic = build_global_indices_markdown(items_ixic, "2026-08-21")
        assert "纳斯达克综合" in md_ixic

    def test_sina_hq_int_symbols_live_format_mock(self):
        """Mock 新浪 HQ 接口返回 int_dji, int_nasdaq, int_sp500 等文本格式解析。"""
        provider = CnAkshareProvider()
        mock_response_text = (
            'var hq_str_int_dji="道琼斯,46247.29,299.97,0.65";\n'
            'var hq_str_int_nasdaq="纳斯达克,22484.07,99.37,0.44";\n'
            'var hq_str_int_sp500="标普指数,6643.70,38.98,0.59";\n'
            'var hq_str_int_nikkei="日经指数,44946.64,-408.35,-0.90";\n'
            'var hq_str_b_KOSPI="韩国KOSPI指数,6912.9500,60.37,0.88,2:27 AM,14:27:00,2026-08-21,14:32:50,6759.9500,6852.5800,6954.1200,6742.4400,0";\n'
            'var hq_str_b_DAX="德国DAX指数,26136.5605,153.52,0.59,9/26/2025,2025-09-26,2026-08-22,00:00:35,25998.9492,25983.0391,26166.2500,25969.5195,0";\n'
            'var hq_str_b_FTSE="富时100指数,10816.5600,68.40,0.64,,,2026-08-21,23:35:00,10748.0700,10748.1600,10831.6400,10734.9200,0";\n'
            'var hq_str_b_CAC="法国CAC40指数,8484.4297,31.34,0.37,9/26/2025,2025-09-26,2026-08-21,23:50:30,8445.6494,8453.0898,8493.5498,8441.1094,0";\n'
            'var hq_str_rt_hkHSI="HSI,恒生指数,25807.610,25698.490,26009.460,25807.610,26009.459,310.970,1.210,0.000,0.000,257277433.846,13796988552,0.000,0.000,28056.100,22518.000,2026/08/21,16:09:01,,,,,,";\n'
            'var hq_str_rt_hkHSTECH="HSTECH,恒生科技指数,4710.410,4700.530,4767.790,4697.600,4766.160,65.630,1.400,0.000,0.000,75189571.304,1494093575,0.000,0.000,6715.460,4229.940,2026/08/21,16:08:30,,,,,,";\n'
        )

        mock_resp = MagicMock()
        mock_resp.text = mock_response_text
        mock_resp.encoding = "gbk"

        with patch("requests.get", return_value=mock_resp):
            snapshots = provider._fetch_global_indices_sina_hq(curr_date="2026-08-21")

        assert "标普500" in snapshots
        assert snapshots["标普500"]["latest_close"] == 6643.70
        assert snapshots["标普500"]["change_1d_pct"] == 0.59
        assert snapshots["标普500"]["source"] == "sina_hq"

        assert "纳斯达克综合" in snapshots
        assert snapshots["纳斯达克综合"]["latest_close"] == 22484.07
        assert snapshots["纳斯达克综合"]["change_1d_pct"] == 0.44

        assert "道琼斯" in snapshots
        assert snapshots["道琼斯"]["latest_close"] == 46247.29
        assert snapshots["道琼斯"]["change_1d_pct"] == 0.65

        assert "德国DAX" in snapshots
        assert snapshots["德国DAX"]["as_of"] == "2026-08-21"

    def test_sina_priority_over_eastmoney_ulist(self):
        """验证新浪 int_* 优先于东财 ulist，东财 ulist 作为第二源补齐缺失项。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        mock_ak.index_global_hist_em.side_effect = RuntimeError("fail")
        mock_ak.stock_hk_index_daily_em.side_effect = RuntimeError("fail")

        # 新浪返回道指和标普，缺少纳指
        mock_sina_data = {
            "标普500": {
                "name": "标普500",
                "code": ".INX",
                "latest_close": 6600.0,
                "change_1d_pct": 0.5,
                "as_of": "2026-08-21",
                "source": "sina_hq",
            },
            "道琼斯": {
                "name": "道琼斯",
                "code": ".DJI",
                "latest_close": 46000.0,
                "change_1d_pct": 0.6,
                "as_of": "2026-08-21",
                "source": "sina_hq",
            },
        }

        # 东财 ulist 包含标普（不同点位）和纳斯达克100
        mock_ulist_data = {
            "标普500": {
                "name": "标普500",
                "code": "SPX",
                "latest_close": 6555.0,  # 新浪优先，应保留 6600.0
                "change_1d_pct": 0.3,
                "as_of": "2026-08-21",
                "source": "eastmoney_ulist",
            },
            "纳斯达克100": {
                "name": "纳斯达克100",
                "code": "NDX",
                "latest_close": 19500.0,  # 东财第二源补齐
                "change_1d_pct": 0.8,
                "as_of": "2026-08-21",
                "source": "eastmoney_ulist",
            },
        }

        with patch.object(provider, "_fetch_global_indices_sina_hq", return_value=mock_sina_data), \
             patch.object(provider, "_fetch_global_indices_em_ulist", return_value=mock_ulist_data), \
             patch.object(provider, "_ak", return_value=mock_ak):
            result = provider.get_global_indices(curr_date="2026-08-21")

        assert "6600.00" in result  # 新浪优先保留
        assert "6555.00" not in result
        assert "46000.00" in result
        assert "纳斯达克100" in result
        assert "19500.00" in result

    def test_us_indices_partial_success_2_of_3(self):
        """美股 2/3 成功（标普+道指可用，纳指缺失）即可 partial/available，不报错。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        mock_ak.index_global_hist_em.side_effect = RuntimeError("fail")
        mock_ak.stock_hk_index_daily_em.side_effect = RuntimeError("fail")

        mock_sina_data = {
            "标普500": {
                "name": "标普500",
                "code": ".INX",
                "latest_close": 6640.0,
                "change_1d_pct": 0.5,
                "as_of": "2026-08-21",
                "source": "sina_hq",
            },
            "道琼斯": {
                "name": "道琼斯",
                "code": ".DJI",
                "latest_close": 46200.0,
                "change_1d_pct": 0.7,
                "as_of": "2026-08-21",
                "source": "sina_hq",
            },
        }

        with patch.object(provider, "_fetch_global_indices_sina_hq", return_value=mock_sina_data), \
             patch.object(provider, "_fetch_global_indices_em_ulist", return_value={}), \
             patch.object(provider, "_ak", return_value=mock_ak):
            result = provider.get_global_indices(curr_date="2026-08-21")

        assert "## 全球核心市场指数行情" in result
        assert "标普500" in result
        assert "6640.00" in result
        assert "道琼斯" in result
        assert "46200.00" in result
        assert "跨市场宏观联动观察" in result
        assert not result.startswith("【数据获取失败】")

    def test_eastmoney_ulist_timezone_as_of_conversion(self):
        """东财 ulist 的美股时间戳按美股交易日（US/Eastern）计算 as_of。"""
        provider = CnAkshareProvider()
        # 模拟 2026-08-22 00:30 UTC（即美东时间 2026-08-21 20:30 EDT）
        # ts = 1787358600
        mock_payload = {
            "rc": 0,
            "data": {
                "diff": [
                    {
                        "f12": "SPX",
                        "f2": 5600.5,
                        "f3": 0.75,
                        "f124": 1787358600,
                    }
                ]
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_payload

        with patch("requests.get", return_value=mock_resp):
            snapshots = provider._fetch_global_indices_em_ulist(curr_date="2026-08-21")

        assert "标普500" in snapshots
        assert snapshots["标普500"]["as_of"] == "2026-08-21"
        assert snapshots["标普500"]["latest_close"] == 5600.5