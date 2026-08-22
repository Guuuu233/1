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
