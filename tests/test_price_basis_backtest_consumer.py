"""Unit tests for price_basis backtest consumer (D-02-4 / C-04-3 / DAV-713).

Contracts verified:
1. Default / vendor_qfq:
   - _get_price_after and _get_price_on default to vendor_qfq.
   - Routes to existing route_to_vendor("get_stock_data", ...).
   - Never calls cn_akshare.get_stock_data with price_basis="raw".
   - Result text/records not labeled as raw.
2. Explicit raw:
   - Explicit price_basis="raw" routes to cn_akshare.get_stock_data(..., price_basis="raw").
   - route_to_vendor is not called.
   - Raw failure (exception, empty string, refusal, insufficient hold days) returns None.
   - Prohibits falling back to vendor_qfq while claiming raw.
   - Backtest run with explicit raw calculates returns based on raw prices.
3. Fail-closed on unsupported / unknown price_basis:
   - pit_raw, pit_adjusted, unspecified, and unknown labels return None.
   - Neither route_to_vendor nor cn_akshare raw fetch is invoked.
   - Backtest run marks outcome_status="incomplete" without falling back to qfq.
4. Non-truncation of hold_days:
   - D-009 hold_days strictness preserved for both vendor_qfq and raw paths.
5. Calling shapes:
   - Supports both keyword argument price_basis="raw" and positional argument.
6. Zero real network traffic:
   - All tests use mocked providers / route_to_vendor.
"""

from unittest.mock import MagicMock, patch

import pytest

from api.services import backtest_service as bt
from tradingagents.dataflows.interface import _registry
from tradingagents.dataflows.providers.cn_akshare_provider import (
    PRICE_BASIS_PIT_ADJUSTED,
    PRICE_BASIS_PIT_RAW,
    PRICE_BASIS_RAW,
    PRICE_BASIS_UNSPECIFIED,
    PRICE_BASIS_VENDOR_QFQ,
    RawDailyFetchError,
)

SAMPLE_QFQ_CSV = """Date,Open,High,Low,Close,Volume
2024-01-02,10.0,10.5,9.8,10.0,1000
2024-01-03,10.1,10.6,9.9,10.2,1100
2024-01-04,10.2,10.7,10.0,10.4,1200
2024-01-05,10.3,10.8,10.1,10.6,1300
2024-01-08,10.5,11.0,10.3,10.8,1400
2024-01-09,10.6,11.1,10.4,11.0,1500
"""

SAMPLE_RAW_CSV = """# Stock data for 600519.SH from 2024-01-01 to 2024-02-15
# price_basis: raw
# Total records: 6

Date,Open,High,Low,Close,Volume,Dividends,Stock Splits
2024-01-02,100.0,105.0,98.0,100.0,1000,0.0,0.0
2024-01-03,101.0,106.0,99.0,102.0,1100,0.0,0.0
2024-01-04,102.0,107.0,100.0,104.0,1200,0.0,0.0
2024-01-05,103.0,108.0,101.0,106.0,1300,0.0,0.0
2024-01-08,105.0,110.0,103.0,108.0,1400,0.0,0.0
2024-01-09,106.0,111.0,104.0,110.0,1500,0.0,0.0
"""


def _make_mock_akshare_provider(csv_data_or_exc=SAMPLE_RAW_CSV):
    mock_prov = MagicMock()
    if isinstance(csv_data_or_exc, Exception) or (
        isinstance(csv_data_or_exc, type) and issubclass(csv_data_or_exc, Exception)
    ):
        mock_prov.get_stock_data.side_effect = csv_data_or_exc
    elif callable(csv_data_or_exc):
        mock_prov.get_stock_data.side_effect = csv_data_or_exc
    else:
        mock_prov.get_stock_data.return_value = csv_data_or_exc
    return mock_prov


# ==============================================================================
# 1. 契约 1：缺省与显式 vendor_qfq 仍走现有 vendor 工具链，不得标为 raw
# ==============================================================================

class TestDefaultAndVendorQfqPath:
    def test_get_price_on_default_routes_to_vendor_qfq(self):
        """_get_price_on 缺省走 route_to_vendor，不调用 cn_akshare raw。"""
        mock_prov = _make_mock_akshare_provider()
        with (
            patch("tradingagents.dataflows.interface.route_to_vendor", return_value=SAMPLE_QFQ_CSV) as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            price = bt._get_price_on("600519.SH", "2024-01-05")

        assert price == 11.0
        mock_route.assert_called_once()
        mock_prov.get_stock_data.assert_not_called()

    def test_get_price_after_default_routes_to_vendor_qfq(self):
        """_get_price_after 缺省走 route_to_vendor，不调用 cn_akshare raw。"""
        mock_prov = _make_mock_akshare_provider()
        with (
            patch("tradingagents.dataflows.interface.route_to_vendor", return_value=SAMPLE_QFQ_CSV) as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            price = bt._get_price_after("600519.SH", "2024-01-01", 3)

        assert price == 10.4
        mock_route.assert_called_once()
        mock_prov.get_stock_data.assert_not_called()

    def test_get_price_on_explicit_vendor_qfq(self):
        """显式 price_basis='vendor_qfq' 走 route_to_vendor，不调用 cn_akshare raw。"""
        mock_prov = _make_mock_akshare_provider()
        with (
            patch("tradingagents.dataflows.interface.route_to_vendor", return_value=SAMPLE_QFQ_CSV) as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            price = bt._get_price_on("600519.SH", "2024-01-05", price_basis=PRICE_BASIS_VENDOR_QFQ)

        assert price == 11.0
        mock_route.assert_called_once()
        mock_prov.get_stock_data.assert_not_called()

    def test_get_price_after_explicit_vendor_qfq(self):
        """显式 price_basis='vendor_qfq' 走 route_to_vendor，不调用 cn_akshare raw。"""
        mock_prov = _make_mock_akshare_provider()
        with (
            patch("tradingagents.dataflows.interface.route_to_vendor", return_value=SAMPLE_QFQ_CSV) as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            price = bt._get_price_after("600519.SH", "2024-01-01", 5, price_basis=PRICE_BASIS_VENDOR_QFQ)

        assert price == 10.8
        mock_route.assert_called_once()
        mock_prov.get_stock_data.assert_not_called()

    def test_backtest_run_default_vendor_qfq_uses_vendor_prices(self):
        """回测任务缺省按 vendor_qfq 取价，不得标成 raw。"""
        job_id = "test-job-default-vendor-qfq"
        bt._create_job(job_id=job_id, user_id="u1", status="pending")

        analysis_mock = {
            "decision": "BUY",
            "final_trade_decision": "BUY",
            "analysis_status": "VALID",
            "trade_action": "BUY",
        }

        mock_prov = _make_mock_akshare_provider()
        with (
            patch.object(bt, "_get_trading_dates", return_value=["2024-01-02"]),
            patch.object(bt, "_run_single_analysis", return_value=analysis_mock),
            patch("tradingagents.dataflows.interface.route_to_vendor", return_value=SAMPLE_QFQ_CSV) as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            bt._run_backtest(
                job_id=job_id,
                symbol="600519.SH",
                start_date="2024-01-02",
                end_date="2024-01-02",
                selected_analysts=["market"],
                hold_days=3,
                sample_interval=1,
                config={},
            )

        job = bt.get_job(job_id, "u1")
        assert job is not None
        record = job["records"][0]
        assert record["price_basis"] == "vendor_qfq"
        assert record["price_basis"] != "raw"
        assert record["entry_price"] == 11.0
        assert record["exit_price"] == 10.4
        assert record["outcome_status"] == "ok"
        assert mock_route.call_count == 2
        mock_prov.get_stock_data.assert_not_called()


# ==============================================================================
# 2. 契约 2：显式 raw 走 cn_akshare get_stock_data(..., price_basis="raw")
# ==============================================================================

class TestExplicitRawPath:
    def test_get_price_on_explicit_raw_routes_to_provider(self):
        """_get_price_on 显式 raw 调用 cn_akshare 并带 price_basis='raw'。"""
        mock_prov = _make_mock_akshare_provider()
        with (
            patch("tradingagents.dataflows.interface.route_to_vendor") as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            price = bt._get_price_on("600519.SH", "2024-01-05", price_basis=PRICE_BASIS_RAW)

        assert price == 110.0
        mock_route.assert_not_called()
        mock_prov.get_stock_data.assert_called_once()
        kwargs = mock_prov.get_stock_data.call_args[1]
        assert kwargs.get("price_basis") == PRICE_BASIS_RAW
        assert kwargs.get("symbol") == "600519.SH"

    def test_get_price_after_explicit_raw_routes_to_provider(self):
        """_get_price_after 显式 raw 调用 cn_akshare 并带 price_basis='raw'。"""
        mock_prov = _make_mock_akshare_provider()
        with (
            patch("tradingagents.dataflows.interface.route_to_vendor") as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            price = bt._get_price_after("600519.SH", "2024-01-01", 3, price_basis=PRICE_BASIS_RAW)

        assert price == 104.0
        mock_route.assert_not_called()
        mock_prov.get_stock_data.assert_called_once()
        kwargs = mock_prov.get_stock_data.call_args[1]
        assert kwargs.get("price_basis") == PRICE_BASIS_RAW
        assert kwargs.get("symbol") == "600519.SH"

    def test_backtest_run_explicit_raw_calculates_returns_from_raw(self):
        """回测任务显式 raw 按 raw 价格计算收益，且不调用 vendor qfq。"""
        job_id = "test-job-explicit-raw-success"
        bt._create_job(job_id=job_id, user_id="u1", status="pending")

        analysis_mock = {
            "decision": "BUY",
            "final_trade_decision": "BUY",
            "analysis_status": "VALID",
            "trade_action": "BUY",
            "price_basis": "raw",
        }

        mock_prov = _make_mock_akshare_provider()
        with (
            patch.object(bt, "_get_trading_dates", return_value=["2024-01-02"]),
            patch.object(bt, "_run_single_analysis", return_value=analysis_mock),
            patch("tradingagents.dataflows.interface.route_to_vendor") as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            bt._run_backtest(
                job_id=job_id,
                symbol="600519.SH",
                start_date="2024-01-02",
                end_date="2024-01-02",
                selected_analysts=["market"],
                hold_days=3,
                sample_interval=1,
                config={},
            )

        job = bt.get_job(job_id, "u1")
        assert job is not None
        record = job["records"][0]
        assert record["price_basis"] == "raw"
        assert record["entry_price"] == 110.0
        assert record["exit_price"] == 104.0
        # raw return: (104.0 - 110.0) / 110.0 * 100 = -5.45%
        assert record["return_pct"] == -5.45
        assert record["outcome_status"] == "ok"
        mock_route.assert_not_called()
        assert mock_prov.get_stock_data.call_count == 2


# ==============================================================================
# 3. 契约 2 & 5：raw 失败禁止回退 qfq 还声称 raw，必须进出场价不可用
# ==============================================================================

class TestRawFailureNoFallback:
    def test_raw_fetch_error_returns_none_no_qfq(self):
        """raw 抛出 RawDailyFetchError 时返回 None，绝不回退至 route_to_vendor。"""
        mock_prov = _make_mock_akshare_provider(
            RawDailyFetchError("tushare.daily:token_missing", "token_missing")
        )
        with (
            patch("tradingagents.dataflows.interface.route_to_vendor") as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            assert bt._get_price_on("600519.SH", "2024-01-05", price_basis=PRICE_BASIS_RAW) is None
            assert bt._get_price_after("600519.SH", "2024-01-01", 3, price_basis=PRICE_BASIS_RAW) is None

        mock_route.assert_not_called()

    def test_raw_empty_or_refusal_string_returns_none(self):
        """raw 返回空串或拒绝提示时返回 None，绝不回退至 route_to_vendor。"""
        for bad_data in ["", "【数据获取失败】stock_data: provider error", "No data found for symbol '600519'"]:
            mock_prov = _make_mock_akshare_provider(bad_data)
            with (
                patch("tradingagents.dataflows.interface.route_to_vendor") as mock_route,
                patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
            ):
                assert bt._get_price_on("600519.SH", "2024-01-05", price_basis=PRICE_BASIS_RAW) is None
                assert bt._get_price_after("600519.SH", "2024-01-01", 3, price_basis=PRICE_BASIS_RAW) is None
            mock_route.assert_not_called()

    def test_raw_insufficient_hold_days_returns_none_without_truncating(self):
        """raw 样本不足 hold_days 时返回 None（严格遵守 D-009），绝不回退至 qfq。"""
        short_raw_csv = """# Stock data for 600519.SH
# price_basis: raw

Date,Open,High,Low,Close,Volume,Dividends,Stock Splits
2024-01-02,100.0,105.0,98.0,100.0,1000,0.0,0.0
2024-01-03,101.0,106.0,99.0,102.0,1100,0.0,0.0
"""
        mock_prov = _make_mock_akshare_provider(short_raw_csv)
        with (
            patch("tradingagents.dataflows.interface.route_to_vendor") as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            # hold_days=5 but only 2 rows
            assert bt._get_price_after("600519.SH", "2024-01-01", 5, price_basis=PRICE_BASIS_RAW) is None

        mock_route.assert_not_called()

    def test_backtest_run_raw_failure_marks_incomplete_and_no_fallback(self):
        """回测任务 raw 失败时必须标记 outcome_status=incomplete，禁止借用 qfq 价格。"""
        job_id = "test-job-raw-fail-no-fallback"
        bt._create_job(job_id=job_id, user_id="u1", status="pending")

        analysis_mock = {
            "decision": "BUY",
            "final_trade_decision": "BUY",
            "analysis_status": "VALID",
            "trade_action": "BUY",
            "price_basis": "raw",
        }

        mock_prov = _make_mock_akshare_provider(
            RawDailyFetchError("tushare.daily:rate_limited", "rate_limited")
        )
        with (
            patch.object(bt, "_get_trading_dates", return_value=["2024-01-02"]),
            patch.object(bt, "_run_single_analysis", return_value=analysis_mock),
            patch("tradingagents.dataflows.interface.route_to_vendor", return_value=SAMPLE_QFQ_CSV) as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            bt._run_backtest(
                job_id=job_id,
                symbol="600519.SH",
                start_date="2024-01-02",
                end_date="2024-01-02",
                selected_analysts=["market"],
                hold_days=3,
                sample_interval=1,
                config={},
            )

        job = bt.get_job(job_id, "u1")
        assert job is not None
        record = job["records"][0]
        assert record["price_basis"] == "raw"
        assert record["entry_price"] is None
        assert record["exit_price"] is None
        assert record["return_pct"] is None
        assert record["outcome_status"] == "incomplete"
        # 验证绝无回退调用 vendor qfq
        mock_route.assert_not_called()


# ==============================================================================
# 4. 契约 3：pit_raw / pit_adjusted / 未知口径失败闭合
# ==============================================================================

class TestUnsupportedAndUnknownBasisFailClosed:
    @pytest.mark.parametrize(
        "unsupported_basis",
        [
            PRICE_BASIS_PIT_RAW,
            PRICE_BASIS_PIT_ADJUSTED,
            PRICE_BASIS_UNSPECIFIED,
            "price_basis.raw",
            "unknown_basis",
            "invalid",
        ],
    )
    def test_unsupported_and_unknown_price_basis_returns_none(self, unsupported_basis):
        """未知或未支持口径进出场价必须失败闭合，返回 None，禁止打网与静默 qfq。"""
        mock_prov = _make_mock_akshare_provider()
        with (
            patch("tradingagents.dataflows.interface.route_to_vendor") as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            assert bt._get_price_on("600519.SH", "2024-01-05", price_basis=unsupported_basis) is None
            assert bt._get_price_after("600519.SH", "2024-01-01", 3, price_basis=unsupported_basis) is None

        mock_route.assert_not_called()
        mock_prov.get_stock_data.assert_not_called()

    def test_backtest_run_unspecified_marks_incomplete_without_qfq(self):
        """分析声明 price_basis='unspecified' 时，进出场价不可用，outcome_status 记为 incomplete。"""
        job_id = "test-job-unspecified-fail-closed"
        bt._create_job(job_id=job_id, user_id="u1", status="pending")

        analysis_mock = {
            "decision": "BUY",
            "final_trade_decision": "BUY",
            "analysis_status": "VALID",
            "trade_action": "BUY",
            "price_basis": "unspecified",
        }

        mock_prov = _make_mock_akshare_provider()
        with (
            patch.object(bt, "_get_trading_dates", return_value=["2024-01-02"]),
            patch.object(bt, "_run_single_analysis", return_value=analysis_mock),
            patch("tradingagents.dataflows.interface.route_to_vendor", return_value=SAMPLE_QFQ_CSV) as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            bt._run_backtest(
                job_id=job_id,
                symbol="600519.SH",
                start_date="2024-01-02",
                end_date="2024-01-02",
                selected_analysts=["market"],
                hold_days=3,
                sample_interval=1,
                config={},
            )

        job = bt.get_job(job_id, "u1")
        assert job is not None
        record = job["records"][0]
        assert record["price_basis"] == "unspecified"
        assert record["entry_price"] is None
        assert record["exit_price"] is None
        assert record["return_pct"] is None
        assert record["outcome_status"] == "incomplete"
        mock_route.assert_not_called()
        mock_prov.get_stock_data.assert_not_called()


# ==============================================================================
# 5. 调用形状与位置参数兼容
# ==============================================================================

class TestCallingShapesAndSignatures:
    def test_positional_price_basis_argument(self):
        """支持位置参数传递 price_basis。"""
        mock_prov = _make_mock_akshare_provider()
        with (
            patch("tradingagents.dataflows.interface.route_to_vendor") as mock_route,
            patch.dict(_registry._providers, {"cn_akshare": mock_prov}),
        ):
            price_on = bt._get_price_on("600519.SH", "2024-01-05", "raw")
            price_after = bt._get_price_after("600519.SH", "2024-01-01", 3, "raw")

        assert price_on == 110.0
        assert price_after == 104.0
        mock_route.assert_not_called()
        assert mock_prov.get_stock_data.call_count == 2
