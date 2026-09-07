"""Unit tests for DataCollector dividend collateral (公司行动分红旁证) wiring (D-02-5 / C-04-3 / DAV-715).

Contracts verified:
1. Provider invocation & as_of parameter:
   - Accesses CnAkshareProvider via _registry.get("cn_akshare").
   - Calls _fetch_tushare_dividend(symbol, as_of=trade_date).
   - as_of is strictly historical trade_date (never today).
   - Missing provider in registry -> explicit failure string, never pretends no dividend.
2. Normal dividend records compression:
   - Formats fields by column names (ann_date, ex_date, cash_div, stk_div, etc.).
   - Compact text output, never raw list[dict] or raw JSON dump.
   - Structured small object / DividendEvidenceText retains .records and .as_of.
3. Empty table (no_rows):
   - Explicitly reports "空表，不得据此判断无分红".
   - Strictly forbids stating "该公司不分红" or "无分红".
   - Not treated as operational data failure.
4. Typed error categories (token_missing, transport, permission_denied, date_exceeds_as_of, missing_field):
   - Formatted as "【数据获取失败】分红旁证 — 原因：…该项不可用。".
   - D-01 error category preserved.
   - Recorded in data_failure_ledger and source_provenance.
5. Price & basis integrity (断言收盘价序列未被分红改写):
   - Dividend collateral NEVER modifies stock_data prices or price_basis.
   - Both vendor_qfq and raw collection paths hang dividend_evidence.
   - Close price sequence and OHLCV DataFrame are strictly identical with and without dividend.
   - Never claims PIT is landed.
6. Zero real network traffic:
   - All tests use mocked providers/tools with zero external gateway calls.
"""
from __future__ import annotations

import copy
import json
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
    PRICE_BASIS_RAW,
    PRICE_BASIS_VENDOR_QFQ,
    StockDataText,
)
from tradingagents.graph.data_collector import (
    DataCollector,
    DividendEvidenceText,
    _fetch_all,
    _fetch_dividend_evidence,
    _format_dividend_compact,
    _format_dividend_row_compact,
)


@pytest.fixture(autouse=True)
def clean_env_tokens(monkeypatch):
    """Ensure tests never hit real network or read real tokens."""
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_API_URL", raising=False)


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


def _make_sample_raw_csv(symbol: str = "600519") -> StockDataText:
    """Sample CSV simulating raw daily bar output."""
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


def _make_sample_dividend_records(symbol: str = "600519") -> list[dict]:
    """Sample realistic dividend records as returned by CnAkshareProvider._fetch_tushare_dividend."""
    return [
        {
            "ts_code": f"{symbol}.SH",
            "end_date": "20231231",
            "ann_date": "2024-03-26",
            "div_proc": "实施",
            "stk_div": 0.0,
            "stk_bo_rate": 0.0,
            "cash_div": 30.876,
            "cash_div_tax": 30.876,
            "record_date": "2024-06-18",
            "ex_date": "2024-06-19",
            "pay_date": "2024-06-19",
            "imp_ann_date": "2024-06-12",
            "symbol": symbol,
            "source_type": "tushare_dividend",
            "canonical_event_id": None,
        },
        {
            "ts_code": f"{symbol}.SH",
            "end_date": "20221231",
            "ann_date": "2023-03-30",
            "div_proc": "实施",
            "stk_div": 0.0,
            "stk_bo_rate": 0.0,
            "cash_div": 25.911,
            "cash_div_tax": 25.911,
            "record_date": "2023-06-29",
            "ex_date": "2023-06-30",
            "pay_date": "2023-06-30",
            "imp_ann_date": "2023-06-21",
            "symbol": symbol,
            "source_type": "tushare_dividend",
            "canonical_event_id": None,
        },
    ]


def _make_mock_provider(dividend_result=None, raw_csv=None):
    """Construct a mock provider with all required collector collateral methods stubbed."""
    prov = MagicMock()
    if dividend_result is not None:
        prov._fetch_tushare_dividend.return_value = dividend_result
    else:
        prov._fetch_tushare_dividend.return_value = (_make_sample_dividend_records(), None, None)

    prov.get_stock_data.return_value = raw_csv if raw_csv is not None else _make_sample_raw_csv()
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


def _make_mock_tool(return_value=None):
    """Construct a mock tool without invoke method."""
    tool = MagicMock()
    del tool.invoke
    tool.return_value = return_value if return_value is not None else _make_sample_qfq_csv()
    return tool


# ==============================================================================
# 1. 契约 1 & 2：正常返回并压缩文本（正常压缩，按列名，绝不 dump 原始 JSON）
# ==============================================================================

class TestDividendNormalCompression:
    """验证分红旁证成功拉取时的列名压缩文本契约。"""

    def test_fetch_dividend_evidence_compact_formatting(self):
        """验证 _fetch_dividend_evidence 返回紧凑文本，包含关键列名，禁止 dump 原始 list[dict]/JSON。"""
        recs = _make_sample_dividend_records("600519")
        mock_prov = _make_mock_provider(dividend_result=(recs, None, None))

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}):
            result = _fetch_dividend_evidence("600519", as_of="2026-08-14")

        # 验证返回为 DividendEvidenceText / str
        assert isinstance(result, str)
        assert isinstance(result, DividendEvidenceText)
        assert result.status == "ok"
        assert result.as_of == "2026-08-14"
        assert len(result.records) == 2

        # 验证包含列名信息
        assert "ann_date" in result
        assert "ex_date" in result
        assert "cash_div" in result
        assert "stk_div" in result
        assert "2024-03-26" in result
        assert "2024-06-19" in result
        assert "30.876" in result

        # 验证禁止原始 list[dict] 或 JSON dump 塞入
        assert not result.startswith("[{")
        assert not result.startswith("list[dict]")
        assert json.dumps(recs) not in result

    def test_as_of_passed_strictly_historical(self):
        """验证调用 _fetch_tushare_dividend 时 as_of 严格使用本次 trade_date（只向前，不填今天）。"""
        mock_prov = _make_mock_provider()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}):
            _fetch_dividend_evidence("600519", as_of="2024-06-18")

        mock_prov._fetch_tushare_dividend.assert_called_once_with(
            symbol="600519", as_of="2024-06-18"
        )

    def test_collect_vendor_qfq_includes_dividend_evidence(self):
        """验证 DataCollector.collect 在缺省/vendor_qfq 模式下挂载 dividend_evidence。"""
        collector = DataCollector()
        mock_prov = _make_mock_provider()
        mock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14", price_basis="vendor_qfq")

        # 检查 pool 与 market_data_context 中的挂载
        assert "dividend_evidence" in pool
        assert "dividend_evidence" in pool["market_data_context"]
        div_ev = pool["dividend_evidence"]
        assert isinstance(div_ev, str)
        assert "ann_date" in div_ev
        assert "cash_div" in div_ev
        assert "2024-03-26" in div_ev

    def test_collect_raw_includes_dividend_evidence(self):
        """验证 DataCollector.collect 在显式 raw 模式下同样挂载 dividend_evidence。"""
        collector = DataCollector()
        mock_prov = _make_mock_provider()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14", price_basis="raw")

        assert "dividend_evidence" in pool
        assert "dividend_evidence" in pool["market_data_context"]
        div_ev = pool["dividend_evidence"]
        assert isinstance(div_ev, str)
        assert "30.876" in div_ev


# ==============================================================================
# 2. 契约 3：空表上报（空表≠无分红；禁止写成「该公司不分红」）
# ==============================================================================

class TestDividendEmptyTable:
    """验证空表上报契约：必须显式说明空表不得据此判断无分红，不得写成无分红。"""

    def test_empty_rows_explicit_report(self):
        """no_rows 返回时，显式「空表，不得据此判断无分红」，不得写成「该公司不分红」。"""
        mock_prov = _make_mock_provider(dividend_result=(None, "tushare.dividend:no_rows", "no_rows"))

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}):
            result = _fetch_dividend_evidence("600519", as_of="2026-08-14")

        assert isinstance(result, str)
        assert "空表，不得据此判断无分红" in result
        assert "该公司不分红" not in result
        assert result.status == "empty"

    def test_empty_list_records_explicit_report(self):
        """当 records 为空列表时同样显式「空表，不得据此判断无分红」。"""
        mock_prov = _make_mock_provider(dividend_result=([], None, None))

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}):
            result = _fetch_dividend_evidence("600519", as_of="2026-08-14")

        assert "空表，不得据此判断无分红" in result
        assert "该公司不分红" not in result

    def test_all_rows_exceed_as_of_reported_as_empty_table(self):
        """前视行全部被 as_of 过滤时返回的 no_rows 显式报告空表。"""
        mock_prov = _make_mock_provider(
            dividend_result=([], "tushare.dividend:no_rows(all_rows_exceed_as_of)", "no_rows")
        )

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}):
            result = _fetch_dividend_evidence("600519", as_of="2010-01-01")

        assert "空表，不得据此判断无分红" in result
        assert "该公司不分红" not in result

    def test_empty_table_not_in_data_failure_ledger(self):
        """空表是有效状态，不得作为数据源拉取失败写入 data_failure_ledger。"""
        collector = DataCollector()
        mock_prov = _make_mock_provider(dividend_result=(None, "tushare.dividend:no_rows", "no_rows"))
        mock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14")

        ledger = pool["market_data_context"].get("data_failure_ledger", [])
        assert not any(
            isinstance(entry, dict) and entry.get("source") == "dividend_evidence"
            for entry in ledger
        )


# ==============================================================================
# 3. 契约 4：类型化错误分类（token_missing / transport / permission_denied 等）
# ==============================================================================

class TestDividendTypedErrors:
    """验证 D-01 失败类型的透传与记账契约。"""

    def test_token_missing_error(self):
        """token_missing: 零网络流量，写成【数据获取失败】分红旁证 — 原因：…该项不可用。"""
        mock_prov = _make_mock_provider(
            dividend_result=(None, "tushare.dividend:token_missing", "token_missing")
        )

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}):
            result = _fetch_dividend_evidence("600519", as_of="2026-08-14")

        assert "【数据获取失败】分红旁证 — 原因：" in result
        assert "token_missing" in result
        assert "该项不可用" in result
        assert result.status == "failed"

    def test_token_missing_recorded_in_failure_ledger(self):
        """token_missing 必须记入 data_failure_ledger。"""
        collector = DataCollector()
        mock_prov = _make_mock_provider(
            dividend_result=(None, "tushare.dividend:token_missing", "token_missing")
        )
        mock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14")

        ledger = pool["market_data_context"].get("data_failure_ledger", [])
        div_failures = [
            entry for entry in ledger
            if isinstance(entry, dict) and entry.get("source") == "dividend_evidence"
        ]
        assert len(div_failures) == 1
        assert div_failures[0]["status"] == "failed"

    def test_transport_error(self):
        """transport 错误正确分类并格式化。"""
        mock_prov = _make_mock_provider(
            dividend_result=(None, "tushare.dividend:transport(ConnectionError)", "transport")
        )

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}):
            result = _fetch_dividend_evidence("600519", as_of="2026-08-14")

        assert "【数据获取失败】分红旁证 — 原因：" in result
        assert "transport" in result
        assert "该项不可用" in result

    def test_permission_denied_error(self):
        """permission_denied 错误正确分类并格式化。"""
        mock_prov = _make_mock_provider(
            dividend_result=(None, "tushare.dividend:permission_denied(code=402)", "permission_denied")
        )

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}):
            result = _fetch_dividend_evidence("600519", as_of="2026-08-14")

        assert "【数据获取失败】分红旁证 — 原因：" in result
        assert "permission_denied" in result
        assert "该项不可用" in result

    def test_missing_field_error(self):
        """missing_field 错误正确分类并格式化。"""
        mock_prov = _make_mock_provider(
            dividend_result=(None, "tushare.dividend:missing_field(ex_date)", "missing_field")
        )

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}):
            result = _fetch_dividend_evidence("600519", as_of="2026-08-14")

        assert "【数据获取失败】分红旁证 — 原因：" in result
        assert "missing_field" in result
        assert "该项不可用" in result

    def test_date_exceeds_as_of_error(self):
        """date_exceeds_as_of 错误正确分类并格式化。"""
        mock_prov = _make_mock_provider(
            dividend_result=(None, "tushare.dividend:date_exceeds_as_of", "date_exceeds_as_of")
        )

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}):
            result = _fetch_dividend_evidence("600519", as_of="2026-08-14")

        assert "【数据获取失败】分红旁证 — 原因：" in result
        assert "date_exceeds_as_of" in result
        assert "该项不可用" in result


# ==============================================================================
# 4. 契约 1：无 provider 场景（显式失败字符串，不得假装无分红）
# ==============================================================================

class TestDividendMissingProvider:
    """验证 registry 中缺少 cn_akshare provider 时显式失败字符串。"""

    def test_missing_provider_returns_explicit_failure(self):
        """无 provider 时返回显式失败字符串，不得假装无分红。"""
        with patch.dict(_registry._providers, {"cn_akshare": None}):
            result = _fetch_dividend_evidence("600519", as_of="2026-08-14")

        assert "【数据获取失败】分红旁证 — 原因：" in result
        assert "provider_unavailable" in result
        assert "该项不可用" in result
        assert "空表" not in result
        assert "该公司不分红" not in result

    def test_provider_without_method_returns_explicit_failure(self):
        """provider 对象缺失 _fetch_tushare_dividend 时显式失败。"""
        fake_prov = object()  # 无 _fetch_tushare_dividend
        with patch.dict(_registry._providers, {"cn_akshare": fake_prov}):
            result = _fetch_dividend_evidence("600519", as_of="2026-08-14")

        assert "【数据获取失败】分红旁证 — 原因：" in result
        assert "provider_unavailable" in result


# ==============================================================================
# 5. 契约 5 & 6：禁止改写收盘价序列与口径（断言收盘价序列未被分红改写）
# ==============================================================================

class TestPriceBasisUnchangedByDividend:
    """验证分红旁证绝不改写日线收盘价序列或 price_basis，禁止落地 PIT。"""

    def test_vendor_qfq_close_prices_unmodified_by_dividend(self):
        """vendor_qfq 模式下，无论是否有分红旁证，stock_data 的收盘价序列严格保持完全一致。"""
        collector = DataCollector()
        mock_tool = _make_mock_tool()

        # 运行 1：带分红旁证
        mock_prov_with_div = _make_mock_provider(
            dividend_result=(_make_sample_dividend_records(), None, None)
        )
        with patch.dict(_registry._providers, {"cn_akshare": mock_prov_with_div}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool_with_div = collector.collect("600519", "2026-08-14", price_basis="vendor_qfq")

        # 运行 2：无分红旁证（空表）
        collector_no_div = DataCollector()
        mock_prov_no_div = _make_mock_provider(
            dividend_result=(None, "tushare.dividend:no_rows", "no_rows")
        )
        with patch.dict(_registry._providers, {"cn_akshare": mock_prov_no_div}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector_no_div, "_fetch_social_context", return_value=None):
            pool_no_div = collector_no_div.collect("600519", "2026-08-14", price_basis="vendor_qfq")

        stock_data_with = pool_with_div["stock_data"]
        stock_data_no = pool_no_div["stock_data"]

        # 断言日线文本中的数据行完全一致
        lines_with = [l for l in stock_data_with.splitlines() if not l.startswith("#")]
        lines_no = [l for l in stock_data_no.splitlines() if not l.startswith("#")]
        assert lines_with == lines_no

        # 解析 DataFrame 对比 close 列
        df_with = pd.read_csv(pd.io.common.StringIO("\n".join(lines_with)))
        df_no = pd.read_csv(pd.io.common.StringIO("\n".join(lines_no)))
        close_col = "close" if "close" in df_with.columns else "Close"
        vol_col = "volume" if "volume" in df_with.columns else "Volume"
        pd.testing.assert_series_equal(df_with[close_col], df_no[close_col])
        pd.testing.assert_series_equal(df_with[vol_col], df_no[vol_col])

        # 断言 price_basis 保持 vendor_qfq
        assert pool_with_div["price_basis"] == PRICE_BASIS_VENDOR_QFQ
        assert pool_with_div["market_data_context"]["price_basis"] == PRICE_BASIS_VENDOR_QFQ
        assert stock_data_with.price_basis == PRICE_BASIS_VENDOR_QFQ

    def test_raw_close_prices_unmodified_by_dividend(self):
        """raw 模式下，无论分红如何，收盘价序列与 raw daily 原文保持完全一致，严禁计算 PIT。"""
        collector = DataCollector()
        raw_csv = _make_sample_raw_csv("600519")

        # 带巨额分红数据（cash_div=100.0）
        huge_div = [
            {
                "ts_code": "600519.SH",
                "end_date": "20231231",
                "ann_date": "2024-03-26",
                "div_proc": "实施",
                "stk_div": 1.0,
                "stk_bo_rate": 0.0,
                "cash_div": 100.0,
                "cash_div_tax": 100.0,
                "record_date": "2024-06-18",
                "ex_date": "2024-06-19",
                "pay_date": "2024-06-19",
                "imp_ann_date": "2024-06-12",
                "symbol": "600519",
                "source_type": "tushare_dividend",
                "canonical_event_id": None,
            }
        ]
        mock_prov = _make_mock_provider(dividend_result=(huge_div, None, None), raw_csv=raw_csv)

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14", price_basis="raw")

        stock_data = pool["stock_data"]
        assert stock_data.price_basis == PRICE_BASIS_RAW
        assert "# price_basis: raw" in stock_data
        assert "# price_basis: vendor_qfq" not in stock_data

        # 检查收盘价依然是 1850.5，没有被 100 元分红或送转折算改写
        lines = [l for l in stock_data.splitlines() if not l.startswith("#")]
        df = pd.read_csv(pd.io.common.StringIO("\n".join(lines)))
        close_col = "close" if "close" in df.columns else "Close"
        assert float(df[close_col].iloc[-1]) == 1850.5

        # 检查旁证已正确附加
        assert "dividend_evidence" in pool
        assert "100.0元" in pool["dividend_evidence"]

    def test_pit_flags_forbidden_in_collector(self):
        """禁止宣称 pit_raw / pit_adjusted 已落地（抛出 UnsupportedPriceBasisError）。"""
        collector = DataCollector()
        from tradingagents.dataflows.providers.cn_akshare_provider import UnsupportedPriceBasisError

        with pytest.raises(UnsupportedPriceBasisError):
            collector.collect("600519", "2026-08-14", price_basis="pit_raw")

        with pytest.raises(UnsupportedPriceBasisError):
            collector.collect("600519", "2026-08-14", price_basis="pit_adjusted")


# ==============================================================================
# 6. 辅助方法与边界测试
# ==============================================================================

class TestDividendHelpersAndEdgeCases:
    """验证辅助格式化方法与特殊边界。"""

    def test_format_dividend_row_with_pit_gap(self):
        """当存在 pit_implementation_gap 时，格式化文本中保留 PIT 说明。"""
        row = {
            "ann_date": "2024-03-26",
            "end_date": "20231231",
            "div_proc": "预案",
            "cash_div": 10.0,
            "stk_div": 0.0,
            "stk_bo_rate": 0.0,
            "ex_date": None,
            "record_date": None,
            "pay_date": None,
            "imp_ann_date": None,
            "pit_implementation_gap": "imp_ann_date(2024-06-12)>2024-04-01",
        }
        text = _format_dividend_row_compact(row)
        assert "PIT边界说明: imp_ann_date(2024-06-12)>2024-04-01" in text

    def test_get_window_preserves_dividend_evidence(self):
        """DataCollector.get_window 返回的 pool 副本保留 dividend_evidence。"""
        collector = DataCollector()
        mock_prov = _make_mock_provider()
        mock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_tool), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14")

        window_pool = collector.get_window(pool, "short", "2026-08-14")
        assert "dividend_evidence" in window_pool
        assert window_pool["dividend_evidence"] == pool["dividend_evidence"]
