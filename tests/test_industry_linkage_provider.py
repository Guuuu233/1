"""针对 IndustryLinkageProvider 的完备确定性单元测试 (DAV-201 / DAV-274)。"""

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


@pytest.fixture
def mock_tushare_lc_response():
    """构造 Tushare fut_daily LC.GFE (碳酸锂主力) 70 个交易日合成数据。"""
    dates = pd.date_range("2026-05-01", periods=70, freq="B")
    items = []
    for i, dt in enumerate(dates):
        d_str = dt.strftime("%Y%m%d")
        # 模拟从 140000 稳步上涨到 158680.0
        close = 140000.0 + (i * 270.0) if i < 69 else 158680.0
        items.append(["LC.GFE", d_str, close - 500, close + 500, close - 800, close, 15000])
    return {
        "code": 0,
        "msg": None,
        "data": {
            "fields": ["ts_code", "trade_date", "open", "high", "low", "close", "vol"],
            "items": list(reversed(items)),  # Tushare 倒序返回
        },
    }


@pytest.fixture
def mock_tushare_spx_response():
    """构造 Tushare index_global SPX 70 个交易日合成数据。"""
    dates = pd.date_range("2026-05-01", periods=70, freq="B")
    items = []
    for i, dt in enumerate(dates):
        d_str = dt.strftime("%Y%m%d")
        close = 5000.0 + (i * 10.0)
        items.append(["SPX", d_str, close - 10, close + 15, close - 15, close, 1000000])
    return {
        "code": 0,
        "msg": None,
        "data": {
            "fields": ["ts_code", "trade_date", "open", "high", "low", "close", "vol"],
            "items": list(reversed(items)),
        },
    }


@pytest.fixture
def mock_tushare_cu_shf_response():
    """构造 Tushare fut_daily CU.SHF (沪铜主力) 70 个交易日合成数据。"""
    dates = pd.date_range("2026-05-01", periods=70, freq="B")
    items = []
    for i, dt in enumerate(dates):
        d_str = dt.strftime("%Y%m%d")
        close = 72000.0 + (i * 50.0)
        items.append(["CU.SHF", d_str, close - 100, close + 200, close - 200, close, 50000])
    return {
        "code": 0,
        "msg": None,
        "data": {
            "fields": ["ts_code", "trade_date", "open", "high", "low", "close", "vol"],
            "items": list(reversed(items)),
        },
    }


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
            assert data["industry_name"] == "消费电子与智能终端"
            assert any("以旧换新" in cat for cat in data["policy_catalysts"])

            # 验证上游成本指标 (LME铜价)
            upstream_list = data["upstream_cost"]
            assert len(upstream_list) >= 1
            copper = [u for u in upstream_list if u["name"] == "LME铜价"][0]
            assert copper["current_value"] == 9123.5
            assert copper["unit"] == "美元/吨"
            assert copper["mom_change"] is not None and copper["mom_change"] > 0
            assert copper["trend"] == "上升"
            assert copper["confidence"] == "高"
            assert copper["status"] == "active"
            assert copper["requested_as_of"] is None
            assert copper["actual_as_of"] == "2026-08-06"
            assert copper["retrieved_at"] is not None
            assert copper["transport_provider"] == "akshare"
            assert copper["api_name"] == "futures_foreign_hist"

            # 验证下游需求指标 (全球智能手机出货量)
            downstream_list = data["downstream_demand"]
            assert len(downstream_list) >= 1
            phone = [d for d in downstream_list if d["name"] == "全球智能手机出货量"][0]
            assert phone["current_value"] is None
            assert phone["trend"] == "数据缺失"
            assert phone["note"] == "手动"
            assert phone["status"] == "manual"
            assert phone["actual_as_of"] is None

            # 验证国际对标指标 (三星电子股价)
            benchmark_list = data["international_benchmark"]
            assert len(benchmark_list) >= 1
            samsung = [b for b in benchmark_list if "三星电子" in b["name"]][0]
            assert samsung["current_value"] == 52000.0
            assert samsung["unit"] == "韩元"
            assert samsung["mom_change"] is not None and samsung["mom_change"] < 0
            assert samsung["trend"] == "下降"
            assert samsung["confidence"] == "高"
            assert samsung["status"] == "active"
            assert samsung["requested_as_of"] is None
            assert samsung["actual_as_of"] == "2026-08-06"
            assert samsung["retrieved_at"] is not None
            assert samsung["transport_provider"] == "yfinance"
            assert samsung["api_name"] == "history"

    def test_new_energy_vehicle_data_fetch_success(
        self, monkeypatch, mock_tushare_lc_response
    ):
        """测试新能源车行业数据采集（包含 Tushare 碳酸锂已付费源与手动指标）。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_configured")
        provider = IndustryLinkageProvider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_tushare_lc_response

        with patch("requests.post", return_value=mock_resp), \
             patch("yfinance.Ticker", side_effect=Exception("Offline test")):
            data = provider.get_industry_linkage("新能源车", use_cache=False)

            assert data is not None
            assert data["industry_name"] == "新能源汽车与智能汽车"
            assert any("购置税减免" in cat for cat in data["policy_catalysts"])

            # 上游成本端：碳酸锂价格 (Tushare fut_daily LC.GFE)
            upstream_list = data["upstream_cost"]
            assert len(upstream_list) >= 1
            lithium = [u for u in upstream_list if "碳酸锂" in u["name"]][0]
            assert lithium["source"] == "tushare"
            assert lithium["symbol"] == "LC.GFE"
            assert lithium["current_value"] == 158680.0
            assert lithium["unit"] == "元/吨"
            assert lithium["trend"] == "上升"
            assert lithium["confidence"] == "高"
            assert lithium["status"] == "active"
            assert lithium["mom_change"] is not None and lithium["mom_change"] > 0
            assert "动力电池正极核心原材料成本传导" in (lithium.get("transmission_logic") or "")
            # 验证 Tushare Provenance 证据链
            assert lithium["requested_as_of"] is None
            assert lithium["actual_as_of"] == "2026-08-06"
            assert lithium["retrieved_at"] is not None
            assert lithium["transport_provider"] == "tushare"
            assert lithium["api_name"] == "fut_daily"

            # 下游需求端：新能源车渗透率
            downstream_list = data["downstream_demand"]
            assert len(downstream_list) >= 1
            nev_rate = [d for d in downstream_list if "渗透率" in d["name"]][0]
            assert nev_rate["current_value"] is None
            assert nev_rate["trend"] == "数据缺失"
            assert nev_rate["note"] == "手动"

            # 国际对标：特斯拉交付量
            benchmark_list = data["international_benchmark"]
            assert len(benchmark_list) >= 1
            tesla = [b for b in benchmark_list if b["name"] == "特斯拉交付量"][0]
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
            assert result["status"] == "active"
            assert result["requested_as_of"] == cutoff_date
            assert result["actual_as_of"] == cutoff_date
            assert result["actual_as_of"] <= result["requested_as_of"]
            assert result["transport_provider"] == "akshare"
            assert result["api_name"] == "futures_foreign_hist"

    def test_graceful_degradation_on_exceptions(self, monkeypatch):
        """测试当外部数据源发生超时、网络故障或异常时优雅降级，不抛出异常，状态为 unavailable。"""
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
        provider = IndustryLinkageProvider()

        with patch("akshare.futures_foreign_hist", side_effect=TimeoutError("Connection timed out")), \
             patch("yfinance.Ticker", side_effect=Exception("Rate limited 429")):

            data = provider.get_industry_linkage("消费电子", use_cache=False)

            assert data is not None
            assert data["industry_name"] == "消费电子与智能终端"

            # 异常时返回结构化缺失状态，不中断分析
            copper = data["upstream_cost"][0]
            assert copper["status"] == "unavailable"
            assert copper["current_value"] is None
            assert copper["actual_as_of"] is None
            assert copper["trend"] == "数据缺失"
            assert copper["confidence"] == "低（接口异常）"
            assert "Connection timed out" in copper["note"]

            samsung = [b for b in data["international_benchmark"] if "三星电子" in b["name"]][0]
            assert samsung["status"] == "unavailable"
            assert samsung["current_value"] is None
            assert samsung["actual_as_of"] is None
            assert samsung["trend"] == "数据缺失"
            assert samsung["confidence"] == "低（接口异常）"
            assert "Rate limited" in samsung["note"]

    def test_as_of_historical_beyond_3_months(self):
        """测试超过3个月的超长历史 as_of 请求能正确构建历史查询窗口并计算指标与 Provenance。"""
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
            assert result["status"] == "active"
            assert result["trend"] == "上升"
            assert result["requested_as_of"] == "2024-10-15"
            assert result["actual_as_of"] == "2024-10-15"
            assert result["actual_as_of"] <= result["requested_as_of"]
            assert result["transport_provider"] == "yfinance"
            assert result["api_name"] == "history"

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
            assert result["status"] == "active"
            assert result["requested_as_of"] == cutoff_date
            assert result["actual_as_of"] == cutoff_date
            assert result["actual_as_of"] <= result["requested_as_of"]

    def test_empty_dataframe_or_no_data_before_as_of(self, monkeypatch, mock_copper_dataframe):
        """测试当 as_of 处于数据源起始日期之前时，安全返回数据缺失与 unavailable 状态。"""
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
        provider = IndustryLinkageProvider()

        with patch("akshare.futures_foreign_hist", return_value=mock_copper_dataframe):
            # as_of 早于 mock 数据集的起始日期 2026-05-01
            ind = IndustryLinkageIndicator(
                name="LME铜价",
                source="akshare",
                symbol="铜",
            )
            result = provider._fetch_indicator(ind, as_of="2020-01-01")

            assert result["status"] == "unavailable"
            assert result["current_value"] is None
            assert result["actual_as_of"] is None
            assert result["trend"] == "数据缺失"
            assert result["confidence"] == "低（有效数据不足）"
            assert result["category"] == "empty_rows"

    def test_unknown_industry_returns_none(self):
        """测试查询未配置的未知行业时返回 None。"""
        provider = IndustryLinkageProvider()
        assert provider.get_industry_linkage("未知行业") is None
        assert provider.get_industry_linkage("") is None
        assert provider.get_industry_linkage(None) is None  # type: ignore

    def test_tushare_fut_daily_lc_gfe_success(
        self, monkeypatch, mock_tushare_lc_response
    ):
        """测试 Tushare fut_daily 碳酸锂 LC.GFE 指标拉取与趋势计算成功。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_configured")
        provider = IndustryLinkageProvider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_tushare_lc_response

        ind = IndustryLinkageIndicator(
            name="碳酸锂价格",
            source="tushare",
            symbol="LC.GFE",
            unit="元/吨",
        )

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = provider._fetch_indicator(ind)

            assert mock_post.call_count == 1
            call_json = mock_post.call_args.kwargs.get("json", {})
            assert call_json.get("api_name") == "fut_daily"
            assert call_json.get("params", {}).get("ts_code") == "LC.GFE"

            assert result["current_value"] == 158680.0
            assert result["unit"] == "元/吨"
            assert result["trend"] == "上升"
            assert result["confidence"] == "高"
            assert result["status"] == "active"
            assert result["mom_change"] is not None and result["mom_change"] > 0
            assert "tushare" in result["note"]
            assert result["requested_as_of"] is None
            assert result["actual_as_of"] == "2026-08-06"
            assert result["retrieved_at"] is not None
            assert result["transport_provider"] == "tushare"
            assert result["api_name"] == "fut_daily"

    def test_tushare_index_global_spx_success(
        self, monkeypatch, mock_tushare_spx_response
    ):
        """测试 Tushare index_global 标普500 SPX 指标拉取与趋势计算成功。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_configured")
        provider = IndustryLinkageProvider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_tushare_spx_response

        ind = IndustryLinkageIndicator(
            name="标普500指数",
            source="tushare",
            symbol="SPX",
            unit="点",
        )

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = provider._fetch_indicator(ind)

            assert mock_post.call_count == 1
            call_json = mock_post.call_args.kwargs.get("json", {})
            assert call_json.get("api_name") == "index_global"
            assert call_json.get("params", {}).get("ts_code") == "SPX"

            assert result["current_value"] == 5690.0
            assert result["trend"] == "上升"
            assert result["confidence"] == "高"
            assert result["status"] == "active"
            assert result["requested_as_of"] is None
            assert result["actual_as_of"] == "2026-08-06"
            assert result["retrieved_at"] is not None
            assert result["transport_provider"] == "tushare"
            assert result["api_name"] == "index_global"

    def test_tushare_token_missing_categorization(self, monkeypatch):
        """测试 TUSHARE_TOKEN 缺失时准确分类为「Token缺失」，绝不伪装为 pending_api。"""
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
        provider = IndustryLinkageProvider()

        ind = IndustryLinkageIndicator(
            name="碳酸锂价格",
            source="tushare",
            symbol="LC.GFE",
        )
        result = provider._fetch_indicator(ind)

        assert result["current_value"] is None
        assert result["trend"] == "数据缺失"
        assert result["confidence"] == "低（Token缺失）"
        assert "TUSHARE_TOKEN missing" in result["note"]
        assert result["requested_as_of"] is None
        assert result["actual_as_of"] is None
        assert result["transport_provider"] == "tushare"
        assert result["api_name"] == "fut_daily"
        assert result["category"] == "token"

    def test_tushare_permission_denied_403_categorization(self, monkeypatch):
        """测试 Tushare 接口 403 / 权限不足时准确分类为「无权限403」。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_configured")
        provider = IndustryLinkageProvider()

        # 1. Tushare code 40101 无权限
        mock_resp1 = MagicMock()
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = {"code": 40101, "msg": "您没有访问该接口的权限"}

        ind = IndustryLinkageIndicator(
            name="美股NVDA股价",
            source="tushare",
            symbol="NVDA",
            metadata={"api_name": "us_daily"},
        )

        with patch("requests.post", return_value=mock_resp1):
            res1 = provider._fetch_indicator(ind)
            assert res1["current_value"] is None
            assert res1["trend"] == "数据缺失"
            assert res1["confidence"] == "低（无权限403）"
            assert "权限不足" in res1["note"]
            assert res1["requested_as_of"] is None
            assert res1["actual_as_of"] is None
            assert res1["transport_provider"] == "tushare"
            assert res1["api_name"] == "us_daily"
            assert res1["category"] == "403"

        # 2. HTTP 403 状态码
        mock_resp2 = MagicMock()
        mock_resp2.status_code = 403

        with patch("requests.post", return_value=mock_resp2):
            res2 = provider._fetch_indicator(ind)
            assert res2["current_value"] is None
            assert res2["trend"] == "数据缺失"
            assert res2["confidence"] == "低（无权限403）"
            assert res2["requested_as_of"] is None
            assert res2["actual_as_of"] is None
            assert res2["transport_provider"] == "tushare"
            assert res2["api_name"] == "us_daily"
            assert res2["category"] == "403"

    def test_tushare_empty_rows_categorization(self, monkeypatch):
        """测试 Tushare 返回空数据或空行时准确分类为「数据源为空」。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_configured")
        provider = IndustryLinkageProvider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "code": 0,
            "msg": None,
            "data": {
                "fields": ["ts_code", "trade_date", "close"],
                "items": [],
            },
        }

        ind = IndustryLinkageIndicator(
            name="碳酸锂价格",
            source="tushare",
            symbol="LC.GFE",
        )

        with patch("requests.post", return_value=mock_resp):
            result = provider._fetch_indicator(ind)
            assert result["current_value"] is None
            assert result["trend"] == "数据缺失"
            assert result["confidence"] == "低（数据源为空）"
            assert "空行" in result["note"]
            assert result["requested_as_of"] is None
            assert result["actual_as_of"] is None
            assert result["transport_provider"] == "tushare"
            assert result["api_name"] == "fut_daily"
            assert result["category"] == "empty_rows"

    def test_tushare_rate_limit_categorization(self, monkeypatch):
        """测试 Tushare 限频 (429/40203) 时准确分类为「频率限制」。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_configured")
        provider = IndustryLinkageProvider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 40203, "msg": "每分钟最多访问该接口200次"}

        ind = IndustryLinkageIndicator(
            name="碳酸锂价格",
            source="tushare",
            symbol="LC.GFE",
        )

        with patch("requests.post", return_value=mock_resp):
            result = provider._fetch_indicator(ind)
            assert result["current_value"] is None
            assert result["trend"] == "数据缺失"
            assert result["confidence"] == "低（频率限制）"
            assert "限频" in result["note"]
            assert result["requested_as_of"] is None
            assert result["actual_as_of"] is None
            assert result["transport_provider"] == "tushare"
            assert result["api_name"] == "fut_daily"
            assert result["category"] == "rate_limited"

    def test_tushare_as_of_lookahead_truncation(
        self, monkeypatch, mock_tushare_lc_response
    ):
        """测试 Tushare 指标严格执行 as_of 防前视截断，绝不泄露未来行情。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_configured")
        provider = IndustryLinkageProvider()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_tushare_lc_response

        # mock 数据集中第 20 个交易日（升序排列）：索引 20
        # items 是倒序排列的
        dates = pd.date_range("2026-05-01", periods=70, freq="B")
        cutoff_date = dates[20].strftime("%Y-%m-%d")
        expected_cutoff_price = 140000.0 + (20 * 270.0)

        ind = IndustryLinkageIndicator(
            name="碳酸锂价格",
            source="tushare",
            symbol="LC.GFE",
            unit="元/吨",
        )

        with patch("requests.post", return_value=mock_resp):
            result = provider._fetch_indicator(ind, as_of=cutoff_date)

            assert result["current_value"] == expected_cutoff_price
            assert result["current_value"] != 158680.0
            assert result["confidence"] == "高"
            assert result["requested_as_of"] == cutoff_date
            assert result["actual_as_of"] == cutoff_date
            assert result["actual_as_of"] <= result["requested_as_of"]
            assert result["transport_provider"] == "tushare"
            assert result["api_name"] == "fut_daily"

    def test_lme_copper_akshare_fails_falls_back_to_tushare_cu_shf(
        self, monkeypatch, mock_tushare_cu_shf_response
    ):
        """测试 LME铜价 在 akshare 失败时自动平滑回退到 Tushare 沪铜 CU.SHF 备用数据源。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_configured")
        provider = IndustryLinkageProvider()

        mock_ts_resp = MagicMock()
        mock_ts_resp.status_code = 200
        mock_ts_resp.json.return_value = mock_tushare_cu_shf_response

        ind = IndustryLinkageIndicator(
            name="LME铜价",
            source="akshare",
            symbol="铜",
            unit="美元/吨",
        )

        with patch("akshare.futures_foreign_hist", side_effect=TimeoutError("LME timeout")), \
             patch("requests.post", return_value=mock_ts_resp) as mock_post:

            result = provider._fetch_indicator(ind)

            assert mock_post.call_count == 1
            call_json = mock_post.call_args.kwargs.get("json", {})
            assert call_json.get("api_name") == "fut_daily"
            assert call_json.get("params", {}).get("ts_code") == "CU.SHF"

            # 验证成功使用 Tushare 备源结果
            expected_price = 72000.0 + (69 * 50.0)
            assert result["current_value"] == expected_price
            assert result["confidence"] == "高"
            assert result["status"] == "active"
            assert "CU.SHF" in result["note"]
            assert result["requested_as_of"] is None
            assert result["actual_as_of"] == "2026-08-06"
            assert result["retrieved_at"] is not None
            assert result["transport_provider"] == "tushare"
            assert result["api_name"] == "fut_daily"

    def test_lme_copper_both_sources_fail_gracefully(self, monkeypatch):
        """测试 LME铜价 在 akshare 与 Tushare 均异常时优雅降级为数据缺失与 unavailable 状态。"""
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
        provider = IndustryLinkageProvider()

        ind = IndustryLinkageIndicator(
            name="LME铜价",
            source="akshare",
            symbol="铜",
            unit="美元/吨",
        )

        with patch("akshare.futures_foreign_hist", side_effect=TimeoutError("LME akshare timeout")):
            result = provider._fetch_indicator(ind)

            assert result["status"] == "unavailable"
            assert result["current_value"] is None
            assert result["actual_as_of"] is None
            assert result["trend"] == "数据缺失"
            assert result["confidence"] == "低（接口异常）"
            assert "akshare 失败" in result["note"]
            assert "Tushare 备源失败" in result["note"]
            assert result["transport_provider"] == "akshare"
            assert result["api_name"] == "futures_foreign_hist"
            assert result["category"] == "api_error"

    def test_tushare_fail_closed_on_lookahead_violation(self, monkeypatch):
        """测试当 Tushare 返回数据存在未来日期违背防前视纪律时，必须 fail-closed。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_configured")
        provider = IndustryLinkageProvider()

        # 构造全部晚于 requested_as_of 的未来数据
        future_data = {
            "code": 0,
            "msg": None,
            "data": {
                "fields": ["ts_code", "trade_date", "open", "high", "low", "close", "vol"],
                "items": [
                    ["LC.GFE", "20260820", 160000, 161000, 159000, 160500, 20000],
                    ["LC.GFE", "20260819", 158000, 159000, 157000, 158500, 18000],
                ],
            },
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = future_data

        ind = IndustryLinkageIndicator(
            name="碳酸锂价格",
            source="tushare",
            symbol="LC.GFE",
            unit="元/吨",
        )

        # 请求基准日早于数据中的实际日期 (2026-08-10 < 2026-08-19)
        with patch("requests.post", return_value=mock_resp):
            result = provider._fetch_indicator(ind, as_of="2026-08-10")

            # 验证 fail-closed: 不返回虚假行情，保留 requested_as_of 与类型化 category
            assert result["status"] == "unavailable"
            assert result["current_value"] is None
            assert result["mom_change"] is None
            assert result["qoq_change"] is None
            assert result["trend"] == "数据缺失"
            assert result["requested_as_of"] == "2026-08-10"
            assert result["actual_as_of"] is None
            assert result["transport_provider"] == "tushare"
            assert result["api_name"] == "fut_daily"
            assert result["category"] in ("empty_rows", "lookahead_violation")

    def test_tushare_token_safety_and_provenance_integrity(self, monkeypatch):
        """测试 Tushare 调用在成功与失败时严禁打印/泄漏 Token，且完整保留 Provenance 元数据。"""
        secret_token = "secret_tushare_token_abc_xyz_98765"
        monkeypatch.setenv("TUSHARE_TOKEN", secret_token)
        provider = IndustryLinkageProvider()

        # 1. 模拟网络异常场景
        ind = IndustryLinkageIndicator(
            name="碳酸锂价格",
            source="tushare",
            symbol="LC.GFE",
            unit="元/吨",
        )
        with patch("requests.post", side_effect=Exception(f"Connection error to API with token")):
            result_err = provider._fetch_indicator(ind, as_of="2026-08-15")

            assert result_err["status"] == "unavailable"
            assert result_err["current_value"] is None
            assert result_err["requested_as_of"] == "2026-08-15"
            assert result_err["actual_as_of"] is None
            assert result_err["transport_provider"] == "tushare"
            assert result_err["api_name"] == "fut_daily"
            assert result_err["category"] in ("network_error", "api_error")
            assert secret_token not in str(result_err)

    def test_tushare_missing_symbol_provenance(self, monkeypatch):
        """测试缺少 symbol 时返回类型化分类与 Provenance 元数据。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_configured")
        provider = IndustryLinkageProvider()

        ind = IndustryLinkageIndicator(
            name="碳酸锂价格",
            source="tushare",
            symbol="",
        )
        result = provider._fetch_indicator(ind, as_of="2026-08-15")

        assert result["status"] == "unavailable"
        assert result["current_value"] is None
        assert result["trend"] == "数据缺失"
        assert result["confidence"] == "低（代码缺失）"
        assert result["requested_as_of"] == "2026-08-15"
        assert result["actual_as_of"] is None
        assert result["transport_provider"] == "tushare"
        assert result["category"] == "symbol_missing"

    def test_akshare_lme_copper_date_penetration_and_provenance(self, mock_copper_dataframe):
        """测试 AkShare LME 铜价成功路径穿透 actual_as_of 并包含完整 Provenance。"""
        provider = IndustryLinkageProvider()

        with patch("akshare.futures_foreign_hist", return_value=mock_copper_dataframe):
            ind = IndustryLinkageIndicator(
                name="LME铜价",
                source="akshare",
                symbol="铜",
                unit="美元/吨",
            )
            result = provider._fetch_indicator(ind, as_of="2026-08-05")

            assert result["status"] == "active"
            assert result["current_value"] is not None
            assert result["requested_as_of"] == "2026-08-05"
            assert result["actual_as_of"] == "2026-08-05"
            assert result["actual_as_of"] <= result["requested_as_of"]
            assert result["retrieved_at"] is not None
            assert result["transport_provider"] == "akshare"
            assert result["api_name"] == "futures_foreign_hist"
            assert result["confidence"] == "高"

    def test_yfinance_indicator_date_penetration_and_provenance(self, mock_samsung_dataframe):
        """测试 yfinance 成功路径穿透 actual_as_of 并包含完整 Provenance。"""
        provider = IndustryLinkageProvider()

        with patch("yfinance.Ticker") as mock_yf:
            mock_ticker_instance = MagicMock()
            mock_ticker_instance.history.return_value = mock_samsung_dataframe
            mock_yf.return_value = mock_ticker_instance

            ind = IndustryLinkageIndicator(
                name="三星电子股价",
                source="yfinance",
                symbol="005930.KS",
                unit="韩元",
            )
            result = provider._fetch_indicator(ind, as_of="2026-08-05")

            assert result["status"] == "active"
            assert result["current_value"] is not None
            assert result["requested_as_of"] == "2026-08-05"
            assert result["actual_as_of"] == "2026-08-05"
            assert result["actual_as_of"] <= result["requested_as_of"]
            assert result["retrieved_at"] is not None
            assert result["transport_provider"] == "yfinance"
            assert result["api_name"] == "history"
            assert result["confidence"] == "高"

    def test_yfinance_failures_status_unavailable(self):
        """测试 yfinance 在空数据、异常、缺少 symbol 与前视违规时均返回 status=unavailable。"""
        provider = IndustryLinkageProvider()

        # 1. yfinance 返回空 DataFrame
        with patch("yfinance.Ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_yf.return_value = mock_ticker

            ind = IndustryLinkageIndicator(name="苹果公司股价", source="yfinance", symbol="AAPL")
            res_empty = provider._fetch_indicator(ind, as_of="2026-08-20")

            assert res_empty["status"] == "unavailable"
            assert res_empty["current_value"] is None
            assert res_empty["actual_as_of"] is None
            assert res_empty["requested_as_of"] == "2026-08-20"
            assert res_empty["transport_provider"] == "yfinance"
            assert res_empty["api_name"] == "history"
            assert res_empty["category"] == "empty_rows"

        # 2. yfinance 抛出异常
        with patch("yfinance.Ticker", side_effect=Exception("Connection timeout")):
            ind = IndustryLinkageIndicator(name="苹果公司股价", source="yfinance", symbol="AAPL")
            res_err = provider._fetch_indicator(ind, as_of="2026-08-20")

            assert res_err["status"] == "unavailable"
            assert res_err["current_value"] is None
            assert res_err["actual_as_of"] is None
            assert res_err["requested_as_of"] == "2026-08-20"
            assert res_err["transport_provider"] == "yfinance"
            assert res_err["api_name"] == "history"
            assert res_err["category"] == "api_error"

        # 3. yfinance 缺少 symbol
        ind_no_sym = IndustryLinkageIndicator(name="未知标的", source="yfinance", symbol="")
        res_no_sym = provider._fetch_indicator(ind_no_sym, as_of="2026-08-20")
        assert res_no_sym["status"] == "unavailable"
        assert res_no_sym["current_value"] is None
        assert res_no_sym["actual_as_of"] is None
        assert res_no_sym["transport_provider"] == "yfinance"
        assert res_no_sym["category"] == "symbol_missing"

        # 4. yfinance 前视违规 fail-closed
        future_df = pd.DataFrame(
            {"Close": [100.0, 105.0]},
            index=pd.to_datetime(["2026-08-25", "2026-08-26"]),
        )
        future_df.index.name = "Date"
        with patch("yfinance.Ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = future_df
            mock_yf.return_value = mock_ticker

            ind = IndustryLinkageIndicator(name="苹果公司股价", source="yfinance", symbol="AAPL")
            # 请求 2026-08-20 但数据全部在 2026-08-25 之后
            res_future = provider._fetch_indicator(ind, as_of="2026-08-20")
            assert res_future["status"] == "unavailable"
            assert res_future["current_value"] is None
            assert res_future["actual_as_of"] is None
            assert res_future["category"] in ("empty_rows", "lookahead_violation")

    def test_weekend_requested_as_of_anti_lookahead(self):
        """测试周末请求（如周六/周日）返回上周五实际日期，绝不允许返回未来实际日期。"""
        provider = IndustryLinkageProvider()

        # 构造截至 2026-08-21 (周五) 的数据
        dates = pd.date_range("2026-05-01", "2026-08-21", freq="B")
        prices = [100.0 + i for i in range(len(dates))]

        df_copper = pd.DataFrame({
            "date": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1000] * len(dates),
        })

        df_samsung = pd.DataFrame(
            {"Close": prices, "Open": prices, "High": prices, "Low": prices, "Volume": [1000] * len(dates)},
            index=dates,
        )
        df_samsung.index.name = "Date"

        # 请求日期为 2026-08-22（周六）
        weekend_as_of = "2026-08-22"

        # 1. AkShare LME 铜价
        with patch("akshare.futures_foreign_hist", return_value=df_copper):
            ind_copper = IndustryLinkageIndicator(name="LME铜价", source="akshare", symbol="铜")
            res_copper = provider._fetch_indicator(ind_copper, as_of=weekend_as_of)

            assert res_copper["status"] == "active"
            assert res_copper["requested_as_of"] == "2026-08-22"
            assert res_copper["actual_as_of"] == "2026-08-21"
            assert res_copper["actual_as_of"] <= res_copper["requested_as_of"]

        # 2. yfinance 三星电子
        with patch("yfinance.Ticker") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = df_samsung
            mock_yf.return_value = mock_ticker

            ind_samsung = IndustryLinkageIndicator(name="三星电子股价", source="yfinance", symbol="005930.KS")
            res_samsung = provider._fetch_indicator(ind_samsung, as_of=weekend_as_of)

            assert res_samsung["status"] == "active"
            assert res_samsung["requested_as_of"] == "2026-08-22"
            assert res_samsung["actual_as_of"] == "2026-08-21"
            assert res_samsung["actual_as_of"] <= res_samsung["requested_as_of"]

    def test_pending_api_and_manual_status_preserved(self):
        """测试 pending_api 与 manual 原有状态严格保持，不把它们伪装成 unavailable 或 active。"""
        provider = IndustryLinkageProvider()

        # 1. pending_api
        ind_pending = IndustryLinkageIndicator(
            name="半导体硅片价格",
            source="pending_api",
            status="pending_api",
            note="待接入API",
        )
        res_pending = provider._fetch_indicator(ind_pending, as_of="2026-08-20")
        assert res_pending["status"] == "pending_api"
        assert res_pending["current_value"] is None
        assert res_pending["actual_as_of"] is None
        assert res_pending["trend"] == "数据缺失"
        assert res_pending["confidence"] == "低（待接入API）"

        # 2. manual
        ind_manual = IndustryLinkageIndicator(
            name="全球智能手机出货量",
            source="manual",
            status="manual",
            note="手动",
        )
        res_manual = provider._fetch_indicator(ind_manual, as_of="2026-08-20")
        assert res_manual["status"] == "manual"
        assert res_manual["current_value"] is None
        assert res_manual["actual_as_of"] is None
        assert res_manual["trend"] == "数据缺失"
        assert res_manual["confidence"] == "低（待手动录入）"

    def test_unimplemented_indicator_status_unavailable(self):
        """测试未实现指标返回 status=unavailable 与 category=not_implemented。"""
        provider = IndustryLinkageProvider()

        ind_unimpl = IndustryLinkageIndicator(
            name="自定义未实现指标",
            source="custom_unknown",
            status="active",
        )
        res = provider._fetch_indicator(ind_unimpl, as_of="2026-08-20")
        assert res["status"] == "unavailable"
        assert res["current_value"] is None
        assert res["actual_as_of"] is None
        assert res["trend"] == "数据缺失"
        assert res["confidence"] == "低（待实现）"
        assert res["category"] == "not_implemented"
