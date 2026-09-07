"""Unit tests for DataCollector fund flow scale metrics normalization (资金规模归一接线) (D-03-2 / C-09-3 / DAV-721).

Contracts verified:
1. Provider invocation & daily_basic denominator:
   - Accesses CnAkshareProvider via _registry.get("cn_akshare").
   - Calls _fetch_tushare_daily_basic(symbol, trade_date, as_of=trade_date).
   - Missing provider / token_missing / provider failure:
     scale_metrics records explicit gap, NEVER fills 0 ratio.
2. Units contract:
   - circ_mv official unit is 万元 (tushare.pro/document/2?doc_id=32).
   - amount must NOT have a default unit. Without explicit amount_unit, only
     net_to_circ_mv is calculated, and turnover ratio gap is recorded.
3. Net amount forwarding & cross-date/cross-security rejection:
   - Takes selected net amount + unit from fund_flow_evidence.
   - Net amount 0 or Decimal("0") is legally preserved and passed to pure function (0 / circ_mv = 0).
   - Missing net amount is passed as None with gap recorded.
   - Cross trade-date or cross security is rejected by pure function.
4. Output contract:
   - Hung under market_data_context["fund_flow_evidence"]["scale_metrics"],
     market_data_context["scale_metrics"], and results["scale_metrics"].
   - Key name is stable: scale_metrics.
   - Strictly neutral: NO "主力强度", NO "强度排名", NO cross-stock ranking.
5. Zero real network traffic:
   - All tests use mocked providers/tools with zero external gateway calls.
"""
from __future__ import annotations

import copy
from decimal import Decimal
import json
from unittest.mock import MagicMock, patch
import pytest

from sqlalchemy import JSON
from sqlalchemy.dialects import sqlite

from api.services.report_service import canonicalize_report_result_data
from tradingagents.dataflows.cninfo_disclosure import (
    CninfoDisclosureEnvelope,
    STATUS_OK,
    SOURCE_TYPE_ANNOUNCEMENT,
    SOURCE_TYPE_IR_SURVEY,
)
from tradingagents.dataflows.fund_flow_evidence import (
    FundFlowText,
    calculate_fund_flow_scale_metrics,
)
from tradingagents.dataflows.interface import _registry
from tradingagents.dataflows.providers.cn_akshare_provider import (
    PRICE_BASIS_RAW,
    PRICE_BASIS_VENDOR_QFQ,
    StockDataText,
)
from tradingagents.graph.data_collector import (
    DataCollector,
    default_market_data_context,
    _fetch_all,
    _serialize_scale_metrics_for_json,
)
from tradingagents.llm_clients.thinking_cleaner import clean_report_result_data


@pytest.fixture(autouse=True)
def clean_env_tokens(monkeypatch):
    """Ensure tests never hit real network or read real tokens."""
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TUSHARE_API_URL", raising=False)


def _make_sample_qfq_csv(symbol: str = "600519") -> str:
    return (
        f"# Stock data for {symbol}\n"
        f"# price_basis: {PRICE_BASIS_VENDOR_QFQ}\n"
        f"# range: 2026-08-12 ~ 2026-08-14\n"
        "Date,Open,High,Low,Close,Volume\n"
        "2026-08-12,1800.0,1820.0,1795.0,1810.0,20000.0\n"
        "2026-08-13,1810.0,1835.0,1805.0,1830.0,22000.0\n"
        "2026-08-14,1830.0,1860.0,1820.0,1850.5,28500.0\n"
    )


def _make_sample_fund_flow_value(
    selected_value: Any = 1.5,
    selected_unit: str = "亿元",
    selected_field: str = "r0_net",
    selected_as_of: str = "2026-08-14",
    symbol: str = "600519",
) -> FundFlowText:
    meta = {
        "status": "selected",
        "selected_source": "ths_instant_snapshot",
        "selected_field": selected_field,
        "selected_value": selected_value,
        "selected_unit": selected_unit,
        "selected_as_of": selected_as_of,
        "selected_direction": "inflow" if selected_value and selected_value > 0 else "neutral",
        "direction_allowed": True,
        "field": selected_field,
        "value": selected_value,
        "unit": selected_unit,
        "as_of": selected_as_of,
        "symbol": symbol,
    }
    records = [
        {
            "date": selected_as_of,
            "measurement_date": selected_as_of,
            "as_of": selected_as_of,
            "symbol": symbol,
            "source": "ths_instant_snapshot",
            "field": selected_field,
            "value": str(selected_value),
            "unit": selected_unit,
        }
    ]
    return FundFlowText("主力净流入约1.5亿", evidence=records, evidence_meta=meta)


def _make_mock_provider(daily_basic_result=None):
    prov = MagicMock()
    if daily_basic_result is not None:
        prov._fetch_tushare_daily_basic.return_value = daily_basic_result
    else:
        prov._fetch_tushare_daily_basic.return_value = (
            {
                "ts_code": "600519.SH",
                "trade_date": "20260814",
                "close": 1850.5,
                "circ_mv": 2220600.0,
                "total_mv": 2220600.0,
                "amount": 550000.0,
            },
            None,
            None,
        )
    prov._fetch_tushare_dividend.return_value = ([], None, None)
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
    tool = MagicMock()
    val = return_value if return_value is not None else _make_sample_qfq_csv()
    tool.invoke.return_value = val
    tool.return_value = val
    return tool


class TestFundFlowScaleCollectorNormalCircMv:
    """1. 契约 1 & 4：正常流通市值占比计算与挂载。"""

    def test_normal_circ_mv_ratio_calculated_and_hung(self):
        """正常 daily_basic 返回 circ_mv（万元），正确计算 net_to_circ_mv 并挂在三处稳定键。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 2220600.0,
                    "amount": 550000.0,
                },
                None,
                None,
            )
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        # 验证挂载位置
        assert "scale_metrics" in results
        assert "scale_metrics" in results["market_data_context"]
        assert "scale_metrics" in results["market_data_context"]["fund_flow_evidence"]
        scale_metrics = results["market_data_context"]["scale_metrics"]

        # 验证数值计算与序列化契约：1.5亿元 (150,000,000元) / 2220600万元 (22,206,000,000元)
        expected_ratio = Decimal("150000000") / Decimal("22206000000")
        assert scale_metrics["net_to_circ_mv"] is not None
        assert scale_metrics["net_to_circ_mv"] == str(expected_ratio)
        assert Decimal(scale_metrics["net_to_circ_mv"]) == expected_ratio
        assert scale_metrics["circ_mv"] == "2220600.0"
        assert Decimal(scale_metrics["circ_mv"]) == Decimal("2220600.0")
        assert scale_metrics["circ_mv_unit"] == "万元"
        assert scale_metrics["circ_mv_source"] == "tushare.daily_basic"
        assert scale_metrics["denominator_source"] == "tushare.daily_basic"

        # 验证参数传递正确性
        mock_prov._fetch_tushare_daily_basic.assert_called_once()
        call_kwargs = mock_prov._fetch_tushare_daily_basic.call_args.kwargs
        call_args = mock_prov._fetch_tushare_daily_basic.call_args.args
        assert call_kwargs.get("symbol") == "600519" or (call_args and call_args[0] == "600519")
        assert call_kwargs.get("trade_date") == "2026-08-14" or (len(call_args) > 1 and call_args[1] == "2026-08-14")
        assert call_kwargs.get("as_of") == "2026-08-14" or (len(call_args) > 2 and call_args[2] == "2026-08-14")


class TestFundFlowScaleCollectorAmountUnitContract:
    """2. 契约 2：amount 不得默认单位契约。"""

    def test_amount_without_unit_omits_turnover_ratio_and_records_gap(self):
        """daily_basic 返回的 amount 无显式单位，成交额占比拒算并记录缺口，但市值占比保留。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 2220600.0,
                    "amount": 550000.0,
                    # 未提供 amount_unit
                },
                None,
                None,
            )
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        assert scale_metrics["net_to_circ_mv"] is not None
        assert scale_metrics["net_to_amount"] is None
        assert any("成交额单位 (amount_unit) 缺失" in gap for gap in scale_metrics["gaps"])
        assert scale_metrics["status"] == "partial"

    def test_amount_with_explicit_unit_calculates_both_ratios(self):
        """若数据源显式给出了 amount_unit，则成交额占比与市值占比均可正常计算。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 2220600.0,
                    "amount": 550000.0,
                    "amount_unit": "万元",
                },
                None,
                None,
            )
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        expected_amt_ratio = Decimal("150000000") / Decimal("5500000000")
        assert scale_metrics["net_to_circ_mv"] is not None
        assert scale_metrics["net_to_amount"] is not None
        assert scale_metrics["net_to_amount"] == str(expected_amt_ratio)
        assert Decimal(scale_metrics["net_to_amount"]) == expected_amt_ratio
        assert scale_metrics["status"] == "available"


class TestFundFlowScaleCollectorDailyBasicFailures:
    """3. 契约 1：daily_basic 失败/缺失时显式缺口，不得填 0 比率。"""

    def test_daily_basic_token_missing_explicit_gap_not_zero_ratio(self):
        """token_missing 时，scale_metrics 显式缺口，比率全为 None，不得填 0。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(None, "tushare.daily_basic:token_missing", "token_missing")
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        assert scale_metrics["net_to_circ_mv"] is None
        assert scale_metrics["net_to_amount"] is None
        assert scale_metrics["net_to_circ_mv"] != 0
        assert scale_metrics["net_to_amount"] != 0
        assert scale_metrics["status"] == "unavailable"
        assert any("token_missing" in gap for gap in scale_metrics["gaps"])
        assert any("缺少流通市值 (circ_mv) 数据" in gap for gap in scale_metrics["gaps"])

    def test_daily_basic_permission_denied_explicit_gap(self):
        """permission_denied 失败时，显式记录错误与缺口，不得填 0 比率。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(None, "tushare.daily_basic:permission_denied(code=40203)", "permission_denied")
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        assert scale_metrics["net_to_circ_mv"] is None
        assert scale_metrics["net_to_circ_mv"] != 0
        assert any("permission_denied" in gap for gap in scale_metrics["gaps"])

    def test_daily_basic_provider_unavailable_in_registry(self):
        """当 registry 中无 cn_akshare provider 时，显式报告不可用，不得填 0。"""
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": None}, clear=False), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        assert scale_metrics["net_to_circ_mv"] is None
        assert scale_metrics["net_to_circ_mv"] != 0
        assert scale_metrics["status"] == "unavailable"
        assert any("cn_akshare provider 不可用" in gap for gap in scale_metrics["gaps"])

    def test_daily_basic_exception_gracefully_handled(self):
        """当 _fetch_tushare_daily_basic 抛出异常时，捕获并记录缺口，不得导致整轮抓取崩盘。"""
        mock_prov = _make_mock_provider()
        mock_prov._fetch_tushare_daily_basic.side_effect = RuntimeError("network connection reset")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        assert scale_metrics["net_to_circ_mv"] is None
        assert any("daily_basic 调用异常" in gap for gap in scale_metrics["gaps"])


class TestFundFlowScaleCollectorZeroNetAmount:
    """4. 契约 3：净额 0 是合法净额，必须原样交给纯函数。"""

    def test_net_amount_zero_int_computes_zero_ratio(self):
        """净额为 0 时，合法计算出 net_to_circ_mv == 0，不得当作缺失或缺口。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 2220600.0,
                },
                None,
                None,
            )
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=0, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        assert scale_metrics["net_amount"] == "0"
        assert Decimal(scale_metrics["net_amount"]) == 0
        assert scale_metrics["net_to_circ_mv"] == "0"
        assert Decimal(scale_metrics["net_to_circ_mv"]) == 0
        assert not any("资金净额 (net_amount) 缺失" in gap for gap in scale_metrics["gaps"])

    def test_net_amount_decimal_zero_computes_zero_ratio(self):
        """净额为 Decimal('0') 时，合法计算出 net_to_circ_mv == 0。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 2220600.0,
                },
                None,
                None,
            )
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=Decimal("0"), selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        assert scale_metrics["net_amount"] == "0"
        assert Decimal(scale_metrics["net_amount"]) == 0
        assert scale_metrics["net_to_circ_mv"] == "0"
        assert Decimal(scale_metrics["net_to_circ_mv"]) == 0

    def test_net_amount_missing_records_gap(self):
        """当 fund_flow_individual 完全不可用（无净额）时，净额记为缺失缺口。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 2220600.0,
                },
                None,
                None,
            )
        )
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(None)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        assert scale_metrics["net_to_circ_mv"] is None
        assert scale_metrics["net_to_circ_mv"] != 0
        assert any("资金净额 (net_amount) 缺失" in gap for gap in scale_metrics["gaps"])


class TestFundFlowScaleCollectorCrossDateAndSecurity:
    """5. 契约 3：跨日/跨证券由纯函数拒算。"""

    def test_cross_trade_date_rejected_by_pure_function(self):
        """分母交易日与资金流证据交易日不一致时，纯函数拒算。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 2220600.0,
                },
                None,
                None,
            )
        )
        # 资金流证据日期为 2026-08-13，与 daily_basic 20260814 跨日
        mock_ff = _make_sample_fund_flow_value(
            selected_value=1.5,
            selected_unit="亿元",
            selected_as_of="2026-08-13",
        )
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        assert scale_metrics["net_to_circ_mv"] is None
        assert any("跨交易日拒算" in gap for gap in scale_metrics["gaps"])

    def test_cross_security_rejected_by_pure_function(self):
        """分母股票代码与证据代码不匹配时，纯函数拒算。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 2220600.0,
                },
                None,
                None,
            )
        )
        # 资金流证据代码为 000001.SZ，与 daily_basic 600519.SH 跨股
        mock_ff = _make_sample_fund_flow_value(
            selected_value=1.5,
            selected_unit="亿元",
            symbol="000001.SZ",
        )
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        assert scale_metrics["net_to_circ_mv"] is None
        assert any("跨证券拒算" in gap for gap in scale_metrics["gaps"])


class TestFundFlowScaleCollectorDenominatorSanity:
    """6. 零分母或负分母拒算。"""

    def test_zero_circ_mv_rejected(self):
        """流通市值为 0 时拒算。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 0.0,
                },
                None,
                None,
            )
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        assert scale_metrics["net_to_circ_mv"] is None
        assert any("流通市值分母非正" in gap for gap in scale_metrics["gaps"])

    def test_negative_circ_mv_rejected(self):
        """流通市值为负数时拒算。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": -100.0,
                },
                None,
                None,
            )
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        assert scale_metrics["net_to_circ_mv"] is None
        assert any("流通市值分母非正" in gap for gap in scale_metrics["gaps"])


class TestFundFlowScaleCollectorNeutralityAndIntegration:
    """7. 中立性无强度排名与 DataCollector.collect 端到端集成。"""

    def test_neutrality_no_strength_ranking_in_output(self):
        """严禁任何『主力强度』、『强度排名』等非中立词汇。"""
        mock_prov = _make_mock_provider()
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        scale_metrics = results["market_data_context"]["scale_metrics"]
        forbidden = ["主力强度", "强度排名", "跨股排名", "加权排名", "rank"]
        for key, value in scale_metrics.items():
            for f in forbidden:
                assert f not in str(key).lower()
                if isinstance(value, str):
                    assert f not in value.lower()

    def test_data_collector_class_collect_integration(self):
        """验证通过 DataCollector.collect 实例方法获取的 pool 包含 scale_metrics。"""
        collector = DataCollector()
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 2220600.0,
                },
                None,
                None,
            )
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)), \
             patch.object(collector, "_fetch_social_context", return_value=None):
            pool = collector.collect("600519", "2026-08-14")

        assert "market_data_context" in pool
        assert "scale_metrics" in pool["market_data_context"]
        assert "scale_metrics" in pool
        sm = pool["market_data_context"]["scale_metrics"]
        expected_ratio = Decimal("150000000") / Decimal("22206000000")
        assert sm["net_to_circ_mv"] == str(expected_ratio)
        assert Decimal(sm["net_to_circ_mv"]) == expected_ratio
        assert sm["circ_mv_unit"] == "万元"
        assert sm["denominator_source"] == "tushare.daily_basic"


class TestFundFlowScaleCollectorJsonSerialization:
    """8. 契约 8：真实 collector 输出必须可 JSON 序列化并支持 SQLAlchemy SQLite JSON bind processor。"""

    def test_normal_collector_output_json_serializable_and_sqlite_bindable(self):
        """正常非零输入下，真实 collector 输出的所有 scale_metrics 挂载点必须支持标准 JSON 序列化及 SQLite JSON bind processor。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 2220600.0,
                    "amount": 550000.0,
                    "amount_unit": "万元",
                },
                None,
                None,
            )
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        # 1. 验证三处挂载
        sm_top = results["scale_metrics"]
        sm_ctx = results["market_data_context"]["scale_metrics"]
        sm_ff = results["market_data_context"]["fund_flow_evidence"]["scale_metrics"]

        # 2. 标准 json.dumps 必须无异常抛出
        dumped_top = json.dumps(sm_top)
        dumped_ctx = json.dumps(sm_ctx)
        dumped_ff = json.dumps(sm_ff)
        assert dumped_top is not None
        assert dumped_ctx is not None
        assert dumped_ff is not None

        # 3. SQLAlchemy SQLite JSON bind processor 必须无异常抛出
        bp = JSON().bind_processor(sqlite.dialect())
        bound_top = bp(sm_top)
        bound_ctx_scale = bp(sm_ctx)
        bound_ff_scale = bp(sm_ff)
        assert bound_top is not None
        assert bound_ctx_scale is not None
        assert bound_ff_scale is not None

        # 4. 报告持久化全路径验证：canonicalize + clean + SQLite bind processor
        report_result_data = {
            "market_data_context": {
                "scale_metrics": sm_ctx,
            },
            "scale_metrics": sm_top,
        }
        canonical = canonicalize_report_result_data(report_result_data)
        cleaned = clean_report_result_data(canonical)
        bound_report = bp(cleaned)
        assert bound_report is not None

        # 5. 反序列化往返验证：数值无损保留，精确十进制字符串契约
        parsed = json.loads(bound_report)
        scale_in_report = parsed["market_data_context"]["scale_metrics"]
        expected_ratio = Decimal("150000000") / Decimal("22206000000")
        expected_amt_ratio = Decimal("150000000") / Decimal("5500000000")
        assert scale_in_report["net_to_circ_mv"] == str(expected_ratio)
        assert Decimal(scale_in_report["net_to_circ_mv"]) == expected_ratio
        assert scale_in_report["net_to_amount"] == str(expected_amt_ratio)
        assert Decimal(scale_in_report["net_to_amount"]) == expected_amt_ratio
        assert scale_in_report["net_amount"] == "1.5"
        assert Decimal(scale_in_report["net_amount"]) == Decimal("1.5")
        assert scale_in_report["circ_mv"] == "2220600.0"
        assert Decimal(scale_in_report["circ_mv"]) == Decimal("2220600.0")
        assert scale_in_report["amount"] == "550000.0"
        assert Decimal(scale_in_report["amount"]) == Decimal("550000.0")

    def test_zero_net_amount_json_serializable_and_sqlite_bindable(self):
        """零净额输入下，0 原样保留且支持标准 JSON 序列化及 SQLite JSON bind processor。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 2220600.0,
                },
                None,
                None,
            )
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=0, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        sm = results["market_data_context"]["scale_metrics"]
        assert json.dumps(sm) is not None

        bp = JSON().bind_processor(sqlite.dialect())
        bound = bp(sm)
        assert bound is not None

        report_result_data = {
            "market_data_context": {"scale_metrics": sm},
            "scale_metrics": results["scale_metrics"],
        }
        canonical = canonicalize_report_result_data(report_result_data)
        cleaned = clean_report_result_data(canonical)
        bound_report = bp(cleaned)
        assert bound_report is not None

        parsed = json.loads(bound_report)
        sm_parsed = parsed["market_data_context"]["scale_metrics"]
        assert sm_parsed["net_amount"] == "0"
        assert Decimal(sm_parsed["net_amount"]) == 0
        assert sm_parsed["net_to_circ_mv"] == "0"
        assert Decimal(sm_parsed["net_to_circ_mv"]) == 0

    def test_missing_denominator_json_serializable_and_sqlite_bindable(self):
        """缺失分母（如 token_missing）导致比率全 None 时，缺口保留且能通过 JSON 序列化。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(None, "tushare.daily_basic:token_missing", "token_missing")
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        sm = results["market_data_context"]["scale_metrics"]
        assert json.dumps(sm) is not None

        bp = JSON().bind_processor(sqlite.dialect())
        bound = bp(sm)
        assert bound is not None

        report_result_data = {
            "market_data_context": {"scale_metrics": sm},
            "scale_metrics": results["scale_metrics"],
        }
        canonical = canonicalize_report_result_data(report_result_data)
        cleaned = clean_report_result_data(canonical)
        bound_report = bp(cleaned)
        assert bound_report is not None

        parsed = json.loads(bound_report)
        sm_parsed = parsed["market_data_context"]["scale_metrics"]
        assert sm_parsed["net_to_circ_mv"] is None
        assert sm_parsed["status"] == "unavailable"
        assert any("token_missing" in g for g in sm_parsed["gaps"])

    def test_missing_amount_unit_json_serializable_and_sqlite_bindable(self):
        """成交额单位缺失时，仅 circ_mv 占比计算，成交额占比为 None，缺口与结果均可序列化。"""
        mock_prov = _make_mock_provider(
            daily_basic_result=(
                {
                    "ts_code": "600519.SH",
                    "trade_date": "20260814",
                    "circ_mv": 2220600.0,
                    "amount": 550000.0,
                },
                None,
                None,
            )
        )
        mock_ff = _make_sample_fund_flow_value(selected_value=1.5, selected_unit="亿元")
        mock_stock_tool = _make_mock_tool()

        with patch.dict(_registry._providers, {"cn_akshare": mock_prov}), \
             patch("tradingagents.graph.data_collector.get_stock_data", mock_stock_tool), \
             patch("tradingagents.graph.data_collector.get_individual_fund_flow", _make_mock_tool(mock_ff)):
            results = _fetch_all("600519", "2026-08-14")

        sm = results["market_data_context"]["scale_metrics"]
        assert json.dumps(sm) is not None

        bp = JSON().bind_processor(sqlite.dialect())
        bound = bp(sm)
        assert bound is not None

        report_result_data = {
            "market_data_context": {"scale_metrics": sm},
            "scale_metrics": results["scale_metrics"],
        }
        canonical = canonicalize_report_result_data(report_result_data)
        cleaned = clean_report_result_data(canonical)
        bound_report = bp(cleaned)
        assert bound_report is not None

        parsed = json.loads(bound_report)
        sm_parsed = parsed["market_data_context"]["scale_metrics"]
        assert sm_parsed["net_to_circ_mv"] is not None
        assert sm_parsed["net_to_amount"] is None
        assert sm_parsed["status"] == "partial"
        assert any("成交额单位 (amount_unit) 缺失" in g for g in sm_parsed["gaps"])

    def test_default_market_data_context_json_serializable(self):
        """default_market_data_context 必须原生可 JSON 序列化。"""
        ctx = default_market_data_context()
        assert json.dumps(ctx) is not None
        bp = JSON().bind_processor(sqlite.dialect())
        bound = bp(ctx)
        assert bound is not None
        parsed = json.loads(bound)
        assert parsed["scale_metrics"]["status"] == "unavailable"
        assert parsed["scale_metrics"]["net_to_circ_mv"] is None
        assert any("未提供市场数据上下文" in g for g in parsed["scale_metrics"]["gaps"])

    def test_unknown_type_raises_type_error_no_default_str_masking(self):
        """严禁使用 default=str 粗暴掩盖未知对象；遇到非 JSON 原生且非 Decimal 类型必须显式抛出 TypeError。"""
        class DummyUnknown:
            pass

        with pytest.raises(TypeError, match="not JSON serializable"):
            _serialize_scale_metrics_for_json({"unknown": DummyUnknown()})

