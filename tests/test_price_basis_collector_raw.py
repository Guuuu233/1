"""Unit tests for collector explicit raw daily channel wiring (D-02-3 / C-04-3 / DAV-711).

Contracts verified:
1. Default behavior (omitted price_basis or "vendor_qfq"):
   - Calls existing get_stock_data vendor chain (agent_utils tool).
   - Never calls cn_akshare.get_stock_data with price_basis="raw".
   - Result text and DataFrame metadata explicitly identify as vendor_qfq.
   - Never labeled as raw.
2. Explicit price_basis="raw":
   - Directly calls cn_akshare.get_stock_data(..., price_basis="raw").
   - Existing get_stock_data tool is not invoked.
   - Result text and DataFrame metadata explicitly identify as raw.
   - Preserves StockDataText with price_basis="raw".
3. Raw failure handling:
   - D-01 failure types (token_missing, date_exceeds_as_of, no_rows, missing_field,
     permission_denied, transport_error, etc.) are preserved as explicit failure strings.
   - Missing cn_akshare in registry produces explicit provider_unavailable failure.
   - Never falls back to vendor_qfq on raw failures.
   - Recorded in data_failure_ledger and source_provenance.
4. Forbidden channels and unknown labels:
   - price_basis="pit_raw", "pit_adjusted", "unspecified" are forbidden
     (raise UnsupportedPriceBasisError).
   - Unknown/invalid labels (e.g. "unknown", "", None, 123, "price_basis.raw") fail closed
     (raise UnknownPriceBasisError).
   - Never silently fall back to vendor_qfq or raw.
5. Cache isolation (no cross-talk / 不串口径):
   - make_cache_key incorporates price_basis for raw ("{ticker}_{date}_raw"),
     while default/vendor_qfq retains "{ticker}_{date}".
   - Sequential collect/get/evict for raw and vendor_qfq on the same ticker+date
     remain completely isolated in cache and do not overwrite each other.
6. Zero real network traffic:
   - All tests use mocked providers/tools, preventing real gateway calls.
"""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from tradingagents.dataflows.cninfo_disclosure import (
    CninfoDisclosureEnvelope,
    STATUS_OK,
    SOURCE_TYPE_ANNOUNCEMENT,
    SOURCE_TYPE_IR_SURVEY,
)
from tradingagents.dataflows.interface import _registry
from tradingagents.dataflows.providers.cn_akshare_provider import (
    PRICE_BASIS_PIT_ADJUSTED,
    PRICE_BASIS_PIT_RAW,
    PRICE_BASIS_RAW,
    PRICE_BASIS_UNSPECIFIED,
    PRICE_BASIS_VENDOR_QFQ,
    PriceBasisError,
    UnknownPriceBasisError,
    UnsupportedPriceBasisError,
    RawDailyFetchError,
    StockDataText,
)
from tradingagents.graph.data_collector import (
    DataCollector,
    make_cache_key,
    _fetch_all,
    _fetch_raw_stock_data,
)


def _make_sample_qfq_csv(symbol: str = "600519") -> str:
    """Sample CSV simulating vendor_qfq daily bar output."""
    return (
        f"# Stock data for {symbol}\n"
        f"# price_basis: {PRICE_BASIS_VENDOR_QFQ}\n"
        f"# range: 2026-08-12 ~ 2026-08-14\n"
        "Date,Open,High,Low,Close,Volume\n"
        "2026-08-12,1800.0,1820.0,1795.0,1810.0,20000.0\n"
        "2026-08-13,1810.0,1835.0,1805.0,1830.0,22000.0\n"
        "2026-08-14,1830.0,1860.0,1820.0,1850.5,28500.0\n"
    )


def _make_sample_raw_csv(symbol: str = "600519") -> str:
    """Sample CSV simulating unadjusted raw daily bar output from CnAkshareProvider."""
    return StockDataText(
        f"# Stock data for {symbol}\n"
        f"# price_basis: {PRICE_BASIS_RAW}\n"
        f"# range: 2026-08-12 ~ 2026-08-14\n"
        "Date,Open,High,Low,Close,Volume\n"
        "2026-08-12,1800.0,1820.0,1795.0,1810.0,20000.0\n"
        "2026-08-13,1810.0,1835.0,1805.0,1830.0,22000.0\n"
        "2026-08-14,1830.0,1860.0,1820.0,1850.5,28500.0\n",
        price_basis=PRICE_BASIS_RAW,
    )


def _make_mock_tool(return_value=None):
    """Construct a mock tool without invoke method so it acts as standard callable."""
    tool = MagicMock()
    del tool.invoke
    tool.return_value = return_value if return_value is not None else _make_sample_qfq_csv()
    return tool


def _make_mock_raw_provider(raw_csv_or_side_effect=None):
    """Construct a mock cn_akshare provider with all collateral methods stubbed properly."""
    prov = MagicMock()
    if isinstance(raw_csv_or_side_effect, Exception) or (
        isinstance(raw_csv_or_side_effect, type)
        and issubclass(raw_csv_or_side_effect, Exception)
    ):
        err = raw_csv_or_side_effect
        prov.get_stock_data.side_effect = lambda *a, **kw: (_ for _ in ()).throw(err)
    elif callable(raw_csv_or_side_effect):
        prov.get_stock_data.side_effect = raw_csv_or_side_effect
    elif raw_csv_or_side_effect is not None:
        prov.get_stock_data.return_value = raw_csv_or_side_effect
    else:
        prov.get_stock_data.return_value = _make_sample_raw_csv()

    prov._fetch_tushare_forecast.return_value = ([], None, None)
    prov._fetch_tushare_repurchase.return_value = ([], None, None)
    prov._fetch_tushare_disclosure_date.return_value = ([], None, None)
    prov.get_cninfo_announcements.return_value = CninfoDisclosureEnvelope(
        status=STATUS_OK, records=[], source_type=SOURCE_TYPE_ANNOUNCEMENT
    )
    prov.get_cninfo_ir_surveys.return_value = CninfoDisclosureEnvelope(
        status=STATUS_OK, records=[], source_type=SOURCE_TYPE_IR_SURVEY
    )
    return prov


# ==============================================================================
# 1. 契约 1：缺省与显式 vendor_qfq 行为一致，走原有工具链，绝不标 raw
# ==============================================================================

class TestDefaultAndVendorQfqPath:
    """验证缺省调用与显式 vendor_qfq 严格走现有工具链，绝不走 raw，绝不标 raw。"""

    def test_default_price_basis_is_vendor_qfq(self):
        """collect(symbol, trade_date) 缺省使用 vendor_qfq，文本与元数据标明 vendor_qfq。"""
        collector = DataCollector()
        mock_raw_provider = _make_mock_raw_provider()
        mock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_raw_provider}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14")

        # 现有工具链被调用
        mock_tool.assert_called_once()
        # cn_akshare 的 raw get_stock_data 绝未被调用
        mock_raw_provider.get_stock_data.assert_not_called()

        # 返回文本与元数据检查
        stock_data = pool["stock_data"]
        assert isinstance(stock_data, str)
        assert isinstance(stock_data, StockDataText)
        assert stock_data.price_basis == PRICE_BASIS_VENDOR_QFQ
        assert "# price_basis: vendor_qfq" in stock_data
        assert "# price_basis: raw" not in stock_data

        # pool 与 context 标注检查
        assert pool["price_basis"] == PRICE_BASIS_VENDOR_QFQ
        assert pool["market_data_context"]["price_basis"] == PRICE_BASIS_VENDOR_QFQ

    def test_explicit_vendor_qfq_kwarg(self):
        """显式传入 price_basis='vendor_qfq' 行为与缺省一致。"""
        collector = DataCollector()
        mock_raw_provider = _make_mock_raw_provider()
        mock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_raw_provider}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14", price_basis=PRICE_BASIS_VENDOR_QFQ)

        mock_tool.assert_called_once()
        mock_raw_provider.get_stock_data.assert_not_called()
        assert pool["stock_data"].price_basis == PRICE_BASIS_VENDOR_QFQ
        assert "# price_basis: vendor_qfq" in pool["stock_data"]
        assert "# price_basis: raw" not in pool["stock_data"]

    def test_explicit_vendor_qfq_positional(self):
        """位置参数传入 price_basis='vendor_qfq' 正常支持。"""
        collector = DataCollector()
        mock_raw_provider = _make_mock_raw_provider()
        mock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_raw_provider}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14", None, "vendor_qfq")

        mock_tool.assert_called_once()
        mock_raw_provider.get_stock_data.assert_not_called()
        assert pool["stock_data"].price_basis == PRICE_BASIS_VENDOR_QFQ

    def test_make_cache_key_default_and_vendor_qfq_identical(self):
        """缺省与显式 vendor_qfq 的缓存键严格一致且向后兼容。"""
        key_default = make_cache_key("600519", "2026-08-14")
        key_explicit = make_cache_key("600519", "2026-08-14", "vendor_qfq")
        assert key_default == "600519_2026-08-14"
        assert key_explicit == "600519_2026-08-14"


# ==============================================================================
# 2. 契约 2：显式 raw 走 cn_akshare provider 直调，标明 raw
# ==============================================================================

class TestExplicitRawPath:
    """验证显式 price_basis='raw' 直调 cn_akshare.get_stock_data(..., price_basis='raw')。"""

    def test_explicit_raw_direct_calls_provider_with_raw_basis(self):
        """显式 price_basis='raw' 直调 provider 并带 price_basis='raw'，不触碰现有 tool。"""
        collector = DataCollector()
        mock_raw_provider = _make_mock_raw_provider()
        mock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_raw_provider}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14", price_basis=PRICE_BASIS_RAW)

        # 验证 cn_akshare.get_stock_data 被直接调用且带 price_basis='raw'
        mock_raw_provider.get_stock_data.assert_called_once()
        call_kwargs = mock_raw_provider.get_stock_data.call_args[1]
        assert call_kwargs.get("price_basis") == PRICE_BASIS_RAW
        assert call_kwargs.get("symbol") == "600519"
        assert call_kwargs.get("as_of") == "2026-08-14"

        # 验证现有 get_stock_data 工具链绝未被调用
        mock_tool.assert_not_called()

        # 验证返回文本与元数据标明 raw
        stock_data = pool["stock_data"]
        assert isinstance(stock_data, str)
        assert isinstance(stock_data, StockDataText)
        assert stock_data.price_basis == PRICE_BASIS_RAW
        assert "# price_basis: raw" in stock_data
        assert "# price_basis: vendor_qfq" not in stock_data

        # 验证 pool 与 context 标注
        assert pool["price_basis"] == PRICE_BASIS_RAW
        assert pool["market_data_context"]["price_basis"] == PRICE_BASIS_RAW

    def test_explicit_raw_positional(self):
        """位置参数传入 price_basis='raw' 亦能正确直调 raw 通道。"""
        collector = DataCollector()
        mock_raw_provider = _make_mock_raw_provider()
        mock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_raw_provider}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14", None, "raw")

        mock_raw_provider.get_stock_data.assert_called_once()
        mock_tool.assert_not_called()
        assert pool["stock_data"].price_basis == PRICE_BASIS_RAW
        assert "# price_basis: raw" in pool["stock_data"]

    def test_raw_daily_computes_indicators_and_vpa(self):
        """raw 数据能够正常被解析并用于指标与 VPA 计算。"""
        collector = DataCollector()
        # 构造包含 30 条 K 线的 raw CSV
        dates = pd.date_range("2026-07-01", "2026-08-14", freq="B")[-30:]
        rows = [
            f"{d.strftime('%Y-%m-%d')},1800.0,1820.0,1790.0,1810.0,25000.0"
            for d in dates
        ]
        raw_csv_long = StockDataText(
            "# Stock data for 600519\n"
            "# price_basis: raw\n"
            "Date,Open,High,Low,Close,Volume\n" + "\n".join(rows) + "\n",
            price_basis="raw",
        )
        mock_raw_provider = _make_mock_raw_provider(raw_csv_long)

        with patch.dict(_registry._providers, {"cn_akshare": mock_raw_provider}), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14", price_basis=PRICE_BASIS_RAW)

        assert pool["stock_data"].price_basis == PRICE_BASIS_RAW
        assert "indicators" in pool
        assert "vpa_indicators" in pool
        assert pool["indicators"].get("close_10_ema") != "无数据"


# ==============================================================================
# 3. 契约 2 & 4：raw 失败沿用 D-01/D-02-2 失败类型，禁止回退，禁止声称 raw
# ==============================================================================

class TestRawDailyFailuresFailClosed:
    """验证 raw 通道失败时显式报错，绝不回退至 vendor_qfq，绝不声称 raw 成功。"""

    @pytest.mark.parametrize(
        ("category", "error_msg"),
        [
            ("token_missing", "tushare.daily:token_missing"),
            ("date_exceeds_as_of", "tushare.daily:date_exceeds_as_of"),
            ("no_rows", "tushare.daily:no_rows"),
            ("missing_field", "tushare.daily:missing_field:close"),
            ("json_shape", "tushare.daily:json_shape"),
            ("permission_denied", "tushare.daily:permission_denied"),
            ("transport_error", "tushare.daily:transport_error"),
        ],
    )
    def test_raw_provider_exceptions_fail_closed_no_qfq_fallback(self, category, error_msg):
        """raw 抛出 RawDailyFetchError 各种类型时，必须显式失败字符串，绝不回退到 qfq。"""
        collector = DataCollector()
        mock_raw_provider = _make_mock_raw_provider(RawDailyFetchError(error_msg, category))
        mock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_raw_provider}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14", price_basis=PRICE_BASIS_RAW)

        # 验证绝不回退调用现有 vendor qfq 工具链
        mock_tool.assert_not_called()

        # 验证返回明确的失败字符串
        stock_data = pool["stock_data"]
        assert isinstance(stock_data, str)
        assert "【数据获取失败】" in stock_data
        assert category in stock_data or error_msg in stock_data
        # 绝不得被标成 raw 成功
        assert not (isinstance(stock_data, StockDataText) and stock_data.price_basis == PRICE_BASIS_RAW)

        # 验证记入 failure ledger
        ledger = pool["market_data_context"]["data_failure_ledger"]
        assert any(entry.get("source") == "stock_data" for entry in ledger)

    def test_raw_cn_akshare_provider_missing_in_registry_fails_closed(self):
        """_registry 中缺失 cn_akshare provider 时，显式报告不可用，绝不回退。"""
        collector = DataCollector()
        mock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": None}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14", price_basis=PRICE_BASIS_RAW)

        mock_tool.assert_not_called()
        stock_data = pool["stock_data"]
        assert "【数据获取失败】" in stock_data
        assert "provider" in stock_data

    def test_fetch_raw_stock_data_helper_direct(self):
        """直接调用 _fetch_raw_stock_data 验证错误捕获。"""
        mock_prov = MagicMock()
        mock_prov.get_stock_data.side_effect = RawDailyFetchError("tushare.daily:token_missing", "token_missing")

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}):
            res = _fetch_raw_stock_data("600519", "2026-08-12", "2026-08-14")

        assert "【数据获取失败】" in res
        assert "token_missing" in res


# ==============================================================================
# 4. 契约 3：禁用 pit_raw / pit_adjusted 通道；未知标签失败闭合
# ==============================================================================

class TestForbiddenChannelsAndUnknownLabels:
    """验证 pit_raw 与 pit_adjusted 本卡禁止作为可用通道，未知标签严格失败闭合。"""

    @pytest.mark.parametrize("forbidden_basis", [PRICE_BASIS_PIT_RAW, PRICE_BASIS_PIT_ADJUSTED, PRICE_BASIS_UNSPECIFIED])
    def test_pit_and_unspecified_channels_forbidden(self, forbidden_basis):
        """pit_raw, pit_adjusted, unspecified 禁止作为可用通道，抛出明确异常。"""
        collector = DataCollector()
        mock_tool = _make_mock_tool()

        with patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.dict(_registry._providers, {"cn_akshare": _make_mock_raw_provider()}):
            with pytest.raises((UnsupportedPriceBasisError, PriceBasisError)):
                collector.collect("600519", "2026-08-14", price_basis=forbidden_basis)

            with pytest.raises((UnsupportedPriceBasisError, PriceBasisError)):
                collector.get("600519", "2026-08-14", price_basis=forbidden_basis)

            with pytest.raises((UnsupportedPriceBasisError, PriceBasisError)):
                make_cache_key("600519", "2026-08-14", price_basis=forbidden_basis)

            with pytest.raises((UnsupportedPriceBasisError, PriceBasisError)):
                _fetch_all("600519", "2026-08-14", price_basis=forbidden_basis)

        mock_tool.assert_not_called()

    @pytest.mark.parametrize(
        "invalid_label",
        [
            "unknown",
            "foo",
            "",
            "   ",
            None,
            123,
            "price_basis.raw",
            "price_basis.vendor_qfq",
            "pit",
            "raw_qfq",
        ],
    )
    def test_unknown_and_invalid_labels_fail_closed(self, invalid_label):
        """未知标签/非法输入必须通过 UnknownPriceBasisError 失败闭合，不得默默 qfq。"""
        collector = DataCollector()
        mock_tool = _make_mock_tool()

        with patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.dict(_registry._providers, {"cn_akshare": _make_mock_raw_provider()}):
            with pytest.raises((UnknownPriceBasisError, PriceBasisError)):
                collector.collect("600519", "2026-08-14", price_basis=invalid_label)  # type: ignore[arg-type]

            with pytest.raises((UnknownPriceBasisError, PriceBasisError)):
                collector.get("600519", "2026-08-14", price_basis=invalid_label)  # type: ignore[arg-type]

            with pytest.raises((UnknownPriceBasisError, PriceBasisError)):
                make_cache_key("600519", "2026-08-14", price_basis=invalid_label)  # type: ignore[arg-type]

            with pytest.raises((UnknownPriceBasisError, PriceBasisError)):
                _fetch_all("600519", "2026-08-14", price_basis=invalid_label)  # type: ignore[arg-type]

        mock_tool.assert_not_called()


# ==============================================================================
# 5. 契约 4：缓存隔离（禁止串口径）
# ==============================================================================

class TestCacheIsolationAndNoCrosstalk:
    """验证 raw 与 vendor_qfq 缓存键隔离，禁止串口径。"""

    def test_cache_keys_are_isolated(self):
        """raw 与 vendor_qfq 具有互不相同的缓存键。"""
        key_qfq = make_cache_key("600519", "2026-08-14", "vendor_qfq")
        key_raw = make_cache_key("600519", "2026-08-14", "raw")
        key_default = make_cache_key("600519", "2026-08-14")

        assert key_qfq == "600519_2026-08-14"
        assert key_default == "600519_2026-08-14"
        assert key_raw == "600519_2026-08-14_raw"
        assert key_raw != key_qfq
        assert key_raw != key_default

    def test_cache_no_cross_talk_between_qfq_and_raw(self):
        """同一实例先后拉取 vendor_qfq 与 raw，两个缓存项独立存在，不得混用。"""
        collector = DataCollector()
        mock_raw_provider = _make_mock_raw_provider()
        mock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_raw_provider}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):

            # 1. 先抓取缺省/qfq
            pool_qfq = collector.collect("600519", "2026-08-14")
            # 2. 再抓取显式 raw
            pool_raw = collector.collect("600519", "2026-08-14", price_basis=PRICE_BASIS_RAW)

        # 验证两者 stock_data 独立且各自保真
        assert pool_qfq["stock_data"].price_basis == PRICE_BASIS_VENDOR_QFQ
        assert "# price_basis: vendor_qfq" in pool_qfq["stock_data"]
        assert "# price_basis: raw" not in pool_qfq["stock_data"]

        assert pool_raw["stock_data"].price_basis == PRICE_BASIS_RAW
        assert "# price_basis: raw" in pool_raw["stock_data"]
        assert "# price_basis: vendor_qfq" not in pool_raw["stock_data"]

        # 验证从 cache.get 查回也是各自独立的
        cached_default = collector.get("600519", "2026-08-14")
        cached_qfq = collector.get("600519", "2026-08-14", price_basis="vendor_qfq")
        cached_raw = collector.get("600519", "2026-08-14", price_basis="raw")

        assert cached_default is not None
        assert cached_qfq is not None
        assert cached_raw is not None

        assert cached_default["stock_data"].price_basis == PRICE_BASIS_VENDOR_QFQ
        assert cached_qfq["stock_data"].price_basis == PRICE_BASIS_VENDOR_QFQ
        assert cached_raw["stock_data"].price_basis == PRICE_BASIS_RAW

        # 3. 淘汰 raw，验证 qfq 仍存活
        collector.evict("600519", "2026-08-14", price_basis="raw")
        assert collector.get("600519", "2026-08-14", price_basis="raw") is None
        assert collector.get("600519", "2026-08-14") is not None
        assert collector.get("600519", "2026-08-14", price_basis="vendor_qfq") is not None

        # 4. 淘汰 qfq，验证两者皆清空
        collector.evict("600519", "2026-08-14")
        assert collector.get("600519", "2026-08-14") is None

    def test_cache_hits_do_not_invoke_network_or_tool(self):
        """二次读取命中缓存时，绝不调用任何工具或 provider。"""
        collector = DataCollector()
        mock_raw_provider = _make_mock_raw_provider()

        with patch.dict(_registry._providers, {"cn_akshare": mock_raw_provider}), \
             patch.object(collector, "_fetch_social_context", return_value=None):

            # 第一次调用
            collector.collect("600519", "2026-08-14", price_basis=PRICE_BASIS_RAW)
            assert mock_raw_provider.get_stock_data.call_count == 1

            # 第二次调用（缓存命中）
            collector.collect("600519", "2026-08-14", price_basis=PRICE_BASIS_RAW)
            assert mock_raw_provider.get_stock_data.call_count == 1


# ==============================================================================
# 6. 契约 5：回测与分析入口缺省行为不变，零真实网络
# ==============================================================================

class TestAnalysisEntryContract:
    """验证分析与回测缺省行为保持与合入前完全一致，不触发真实网络。"""

    def test_run_single_analysis_default_stays_vendor_qfq(self):
        """api.services.backtest_service 中的 _run_single_analysis 缺省保持 vendor_qfq，绝不标 raw。"""
        import api.services.backtest_service as bt
        with patch("tradingagents.graph.trading_graph.TradingAgentsGraph") as mock_graph_cls:
            mock_graph = MagicMock()
            mock_graph.propagate.return_value = ({"final_trade_decision": "BUY"}, {})
            mock_graph.process_signal.return_value = "BUY"
            mock_graph_cls.return_value = mock_graph

            res = bt._run_single_analysis("600519.SH", "2024-01-02", ["market"], {})
            assert res["price_basis"] == PRICE_BASIS_VENDOR_QFQ
            assert res["price_basis"] != PRICE_BASIS_RAW

    def test_zero_real_network_guard(self):
        """测试套件执行期间拦截真实 requests.post，防误发外部网关流量。"""
        collector = DataCollector()
        mock_raw_provider = _make_mock_raw_provider()

        with patch.dict(_registry._providers, {"cn_akshare": mock_raw_provider}), \
             patch("requests.post") as mock_post, \
             patch.object(collector, "_fetch_social_context", return_value=None):
            collector.collect("600519", "2026-08-14", price_basis=PRICE_BASIS_RAW)

        assert mock_post.call_count == 0
