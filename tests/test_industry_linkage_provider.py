"""针对 IndustryLinkageProvider 的完备确定性单元测试 (DAV-201 M2)。"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from tradingagents.dataflows.industry_linkage import IndustryLinkageIndicator
from tradingagents.dataflows.providers.industry_linkage_provider import (
    IndustryLinkageProvider,
)


@pytest.fixture
def mock_copper_dataframe():
    """构造包含 70 个交易日的合成 LME 铜价 DataFrame。"""
    dates = pd.date_range("2026-05-01", periods=70, freq="B")
    # 模拟铜价从 8500 稳步上涨到 9123.50
    prices = [8500.0 + (i * 9.0) for i in range(69)] + [9123.50]
    return pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": [p + 50.0 for p in prices],
        "low": [p - 50.0 for p in prices],
        "close": prices,
        "volume": [10000] * 70,
        "position": [0] * 70,
        "s": [0] * 70,
    })


@pytest.fixture
def mock_samsung_dataframe():
    """构造包含 70 个交易日的合成三星电子股价 DataFrame。"""
    dates = pd.date_range("2026-05-01", periods=70, freq="B")
    # 模拟三星股价从 58000 震荡下行至 52000.0
    prices = [58000.0 - (i * 85.0) for i in range(69)] + [52000.0]
    df = pd.DataFrame({
        "Open": prices,
        "High": [p + 200.0 for p in prices],
        "Low": [p - 200.0 for p in prices],
        "Close": prices,
        "Volume": [500000] * 70,
    }, index=dates)
    df.index.name = "Date"
    return df


class TestIndustryLinkageProvider:
    """测试 IndustryLinkageProvider 核心功能与指标采集。"""

    def test_consumer_electronics_data_fetch_success(
        self, mock_copper_dataframe, mock_samsung_dataframe
    ):
        """测试消费电子行业全维度指标采集与计算成功场景。"""
        provider = IndustryLinkageProvider()

        with patch("akshare.futures_foreign_hist", return_value=mock_copper_dataframe), \
             patch("yfinance.Ticker") as mock_yf:
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.history.return_value = mock_samsung_dataframe
            mock_yf.return_value = mock_ticker_instance

            data = provider.get_industry_linkage("消费电子", use_cache=False)

            assert data is not None
            assert data["industry_name"] == "消费电子/半导体显示"
            assert "消费品以旧换新" in data["policy_catalysts"]

            # 验证上游成本指标 (LME铜价)
            upstream_list = data["upstream_cost"]
            assert len(upstream_list) == 1
            copper = upstream_list[0]
            assert copper["name"] == "LME铜价"
            assert copper["current_value"] == 9123.5
            assert copper["unit"] == "美元/吨"
            assert copper["mom_change"] is not None and copper["mom_change"] > 0
            assert copper["trend"] == "上升"
            assert copper["confidence"] == "高"
            assert copper["status"] == "active"

            # 验证下游需求指标 (全球智能手机出货量)
            downstream_list = data["downstream_demand"]
            assert len(downstream_list) == 1
            phone = downstream_list[0]
            assert phone["name"] == "全球智能手机出货量"
            assert phone["current_value"] is None
            assert phone["trend"] == "数据缺失"
            assert phone["note"] == "手动"
            assert phone["status"] == "manual"

            # 验证国际对标指标 (三星电子股价)
            benchmark_list = data["international_benchmark"]
            assert len(benchmark_list) == 1
            samsung = benchmark_list[0]
            assert samsung["name"] == "三星电子股价"
            assert samsung["current_value"] == 52000.0
            assert samsung["unit"] == "韩元"
            assert samsung["mom_change"] is not None and samsung["mom_change"] < 0
            assert samsung["trend"] == "下降"
            assert samsung["confidence"] == "高"
            assert samsung["status"] == "active"

    def test_new_energy_vehicle_data_fetch_success(self):
        """测试新能源车行业数据采集（包含待接入API指标与手动指标）。"""
        provider = IndustryLinkageProvider()
        data = provider.get_industry_linkage("新能源车", use_cache=False)

        assert data is not None
        assert data["industry_name"] == "新能源车/动力电池"
        assert "新能源汽车购置税减免" in data["policy_catalysts"]

        # 上游成本端：碳酸锂价格（待接入API）
        upstream_list = data["upstream_cost"]
        assert len(upstream_list) == 1
        lithium = upstream_list[0]
        assert lithium["name"] == "碳酸锂价格"
        assert lithium["current_value"] is None
        assert lithium["trend"] == "数据缺失"
        assert lithium["confidence"] == "低（待接入API）"
        assert lithium["note"] == "待接入API"
        assert lithium["status"] == "pending_api"

        # 下游需求端：新能源车渗透率
        downstream_list = data["downstream_demand"]
        assert len(downstream_list) == 1
        nev_rate = downstream_list[0]
        assert nev_rate["name"] == "新能源车渗透率"
        assert nev_rate["current_value"] is None
        assert nev_rate["trend"] == "数据缺失"
        assert nev_rate["note"] == "手动"

        # 国际对标：特斯拉交付量
        benchmark_list = data["international_benchmark"]
        assert len(benchmark_list) == 1
        tesla = benchmark_list[0]
        assert tesla["name"] == "特斯拉交付量"
        assert tesla["current_value"] is None
        assert tesla["trend"] == "数据缺失"
        assert tesla["note"] == "手动"

    def test_caching_and_clear_cache(self, mock_copper_dataframe):
        """测试 1 小时内存缓存命中与缓存清理逻辑。"""
        provider = IndustryLinkageProvider(cache_ttl=3600)

        with patch("akshare.futures_foreign_hist", return_value=mock_copper_dataframe), \
             patch("yfinance.Ticker") as mock_yf:
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.history.side_effect = Exception("Network offline")
            mock_yf.return_value = mock_ticker_instance

            # 首次调用
            res1 = provider.get_industry_linkage("消费电子")
            assert res1 is not None
            cached_at_1 = res1["cached_at"]

            # 第二次调用（验证从缓存返回且时间戳一致）
            res2 = provider.get_industry_linkage("消费电子")
            assert res2 is not None
            assert res2["cached_at"] == cached_at_1
            assert res2["upstream_cost"][0]["current_value"] == 9123.5

            # 清理缓存
            provider.clear_cache()
            assert len(provider._cache) == 0

    def test_as_of_filtering_prevents_lookahead(self, mock_copper_dataframe):
        """测试 as_of 参数过滤生效，严格遵守防前视纪律。"""
        provider = IndustryLinkageProvider()

        with patch("akshare.futures_foreign_hist", return_value=mock_copper_dataframe):
            # 将 as_of 设定在第 30 个交易日
            cutoff_date = mock_copper_dataframe.iloc[30]["date"].strftime("%Y-%m-%d")
            expected_price = float(mock_copper_dataframe.iloc[30]["close"])

            ind = IndustryLinkageIndicator(
                name="LME铜价",
                source="akshare",
                symbol="铜",
            )
            result = provider._fetch_indicator(ind, as_of=cutoff_date)

            assert result["current_value"] == expected_price
            assert result["confidence"] == "高"

    def test_graceful_degradation_on_exceptions(self):
        """测试当外部数据源发生超时、网络故障或异常时优雅降级，不抛出异常。"""
        provider = IndustryLinkageProvider()

        with patch("akshare.futures_foreign_hist", side_effect=TimeoutError("Connection timed out")), \
             patch("yfinance.Ticker", side_effect=Exception("Rate limited 429")):

            data = provider.get_industry_linkage("消费电子", use_cache=False)

            assert data is not None
            assert data["industry_name"] == "消费电子/半导体显示"

            # 异常时返回结构化缺失状态，不中断分析
            copper = data["upstream_cost"][0]
            assert copper["current_value"] is None
            assert copper["trend"] == "数据缺失"
            assert copper["confidence"] == "低（接口异常）"
            assert "Connection timed out" in copper["note"]

            samsung = data["international_benchmark"][0]
            assert samsung["current_value"] is None
            assert samsung["trend"] == "数据缺失"
            assert samsung["confidence"] == "低（接口异常）"
            assert "Rate limited" in samsung["note"]

    def test_as_of_historical_beyond_3_months(self):
        """测试超过3个月的超长历史 as_of 请求能正确构建历史查询窗口并计算指标。"""
        provider = IndustryLinkageProvider()

        # 构造 2024 年 6 月至 10 月的历史数据（上涨趋势）
        hist_dates = pd.date_range("2024-06-01", "2024-10-15", freq="B")
        hist_prices = [45000.0 + (i * 50.0) for i in range(len(hist_dates))]
        mock_df = pd.DataFrame({"Close": hist_prices}, index=hist_dates)
        mock_df.index.name = "Date"

        with patch("yfinance.Ticker") as mock_yf:
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.history.return_value = mock_df
            mock_yf.return_value = mock_ticker_instance

            ind = IndustryLinkageIndicator(
                name="三星电子股价",
                source="yfinance",
                symbol="005930.KS",
            )
            result = provider._fetch_indicator(ind, as_of="2024-10-15")

            # 验证 yfinance.history 被调用时带有明确的历史 start 与 end 日期
            mock_ticker_instance.history.assert_called_once()
            call_kwargs = mock_ticker_instance.history.call_args.kwargs
            assert "start" in call_kwargs and "end" in call_kwargs
            assert "2024" in call_kwargs["start"]
            assert "2024" in call_kwargs["end"]

            assert result["current_value"] == hist_prices[-1]
            assert result["confidence"] == "高"
            assert result["trend"] == "上升"

    def test_as_of_future_rows_strictly_discarded(self, mock_copper_dataframe):
        """测试未来日期数据被严格截断丢弃，杜绝前视偏差。"""
        provider = IndustryLinkageProvider()

        with patch("akshare.futures_foreign_hist", return_value=mock_copper_dataframe):
            # 将 as_of 设定在第 15 个交易日（后面有 50+ 个未来交易日数据）
            cutoff_date = mock_copper_dataframe.iloc[15]["date"].strftime("%Y-%m-%d")
            cutoff_price = float(mock_copper_dataframe.iloc[15]["close"])
            future_price = float(mock_copper_dataframe.iloc[-1]["close"])
            assert cutoff_price != future_price

            ind = IndustryLinkageIndicator(
                name="LME铜价",
                source="akshare",
                symbol="铜",
            )
            result = provider._fetch_indicator(ind, as_of=cutoff_date)

            assert result["current_value"] == cutoff_price
            assert result["current_value"] != future_price

    def test_empty_dataframe_or_no_data_before_as_of(self, mock_copper_dataframe):
        """测试当 as_of 处于数据源起始日期之前时，安全返回数据缺失而不崩溃。"""
        provider = IndustryLinkageProvider()

        with patch("akshare.futures_foreign_hist", return_value=mock_copper_dataframe):
            # as_of 早于 mock 数据集的起始日期 2026-05-01
            ind = IndustryLinkageIndicator(
                name="LME铜价",
                source="akshare",
                symbol="铜",
            )
            result = provider._fetch_indicator(ind, as_of="2020-01-01")

            assert result["current_value"] is None
            assert result["trend"] == "数据缺失"
            assert result["confidence"] == "低（有效数据不足）"

    def test_unknown_industry_returns_none(self):
        """测试查询未配置的未知行业时返回 None。"""
        provider = IndustryLinkageProvider()
        assert provider.get_industry_linkage("未知行业") is None
        assert provider.get_industry_linkage("") is None
        assert provider.get_industry_linkage(None) is None  # type: ignore
