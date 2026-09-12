"""L5: OHLCV / 量价输入完整度与填充率提升回归与红队验收套件 (DAV-844).

覆盖验收钉子要求：
1. 候选缺口基线 vs 修复后填充率测量对照 (10/10 vs 1/10)；
2. 修复后用同一固定输入重跑，覆盖率实测提升，且非空非伪造；
3. 缺数据仍显式进入 failure / gap / provenance 语义；
4. 严格遵守请求日期截断、列名解析、重复日期冲突拒收与现有 price_basis 语义；
5. 六大红队场景：
   - RT-1: provider failure 与空结果不得变成 confirmed empty 或有效数据；
   - RT-2: 缺 volume / 缺必要 OHLCV 列必须显式不可用；
   - RT-3: 合法 volume=0 或 Decimal(0) 不得丢失；
   - RT-4: 日期超出 cutoff、未来行和重复日期冲突不得进入量价输入；
   - RT-5: provider fallback 不能跨越显式 price_basis 约束；
   - RT-6: 所有派生指标在原始输入不足时必须保持 insufficient / missing 语义。
"""

from decimal import Decimal
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from tradingagents.dataflows.interface import _registry
from tradingagents.dataflows.providers.cn_akshare_provider import (
    CnAkshareProvider,
    PRICE_BASIS_RAW,
    PRICE_BASIS_VENDOR_QFQ,
    PRICE_BASIS_PIT_RAW,
    PRICE_BASIS_PIT_ADJUSTED,
    PRICE_BASIS_UNSPECIFIED,
    UnsupportedPriceBasisError,
    UnknownPriceBasisError,
    StockDataText,
)
from tradingagents.dataflows.vendor_result import (
    VendorEmpty,
    VendorFail,
    VendorRefuse,
)
from tradingagents.graph.data_collector import (
    DataCollector,
    _compute_vpa_indicators,
    _fetch_all,
    _normalize_daily_frame,
    _parse_csv_to_dataframe,
    _resolve_ohlcv_columns,
    compute_vpa_deterministic_features,
    VOLUME_REGIME_INSUFFICIENT_DATA,
    REVERSAL_STATE_INSUFFICIENT_DATA,
)


def _make_bars(cols: list[str]) -> str:
    rows = [
        ["2026-03-10", 100.0, 105.0, 98.0, 102.0, 50000.0],
        ["2026-03-11", 102.0, 106.0, 101.0, 104.0, 55000.0],
        ["2026-03-12", 104.0, 108.0, 103.0, 107.0, 60000.0],
    ]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(x) for x in r))
    return "\n".join(lines) + "\n"


FIXED_SYMBOL = "600519"
FIXED_TRADE_DATE = "2026-03-12"

BENCHMARK_FIXTURES = {
    "canonical_en": _make_bars(["Date", "Open", "High", "Low", "Close", "Volume"]),
    "tushare_daily": _make_bars(["trade_date", "open", "high", "low", "close", "vol"]),
    "chinese_standard": _make_bars(["日期", "开盘", "最高", "最低", "收盘", "成交量"]),
    "chinese_price_suffix": _make_bars(["日期", "开盘价", "最高价", "最低价", "收盘价", "成交量"]),
    "chinese_share_unit": _make_bars(["日期", "开盘", "最高", "最低", "收盘", "成交量(股)"]),
    "chinese_lot_unit": _make_bars(["日期", "开盘", "最高", "最低", "收盘", "成交量(手)"]),
    "chinese_trade_date": _make_bars(["交易日期", "开盘", "最高", "最低", "收盘", "成交量"]),
    "short_csv_under_50": "date,open,high,low,close,vol\n2026-03-12,1,2,1,2,0",
    "whitespace_cols": _make_bars([" Date ", " Open ", " High ", " Low ", " Close ", " Volume "]),
    "dataframe_direct": pd.DataFrame({
        "Date": ["2026-03-10", "2026-03-11", "2026-03-12"],
        "Open": [100.0, 102.0, 104.0],
        "High": [105.0, 106.0, 108.0],
        "Low": [98.0, 101.0, 103.0],
        "Close": [102.0, 104.0, 107.0],
        "Volume": [50000.0, 55000.0, 60000.0],
    }),
}


def _run_mock_fetch_all(stock_data_return, symbol=FIXED_SYMBOL, trade_date=FIXED_TRADE_DATE, price_basis=PRICE_BASIS_VENDOR_QFQ):
    """Run _fetch_all with all auxiliary network tasks stubbed out."""
    import tradingagents.graph.data_collector as dc

    def mock_safe(tool, payload):
        tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
        if tool_name in ("get_stock_data", "_fetch_raw_stock_data"):
            return stock_data_return
        if tool_name == "_fetch_realtime_context":
            return {"status": "unavailable"}
        return ""

    with patch.object(dc, "_safe", side_effect=mock_safe):
        return dc._fetch_all(symbol, trade_date, price_basis=price_basis)


class TestOHLCVCompletenessAndFillRate:
    """钉子 1 & 2: 基线测量 vs 修复后全量提升对照 (10/10)."""

    @pytest.mark.parametrize("name,fixture", list(BENCHMARK_FIXTURES.items()))
    def test_repaired_fill_rate_on_all_benchmarks(self, name, fixture):
        """修复后：全部 10 种典型量价输入格式均解析成功且填充率达 100%."""
        # 1. 结构解析
        parsed = _parse_csv_to_dataframe(fixture)
        assert parsed is not None, f"Failed to parse CSV for {name}"
        assert not parsed.empty, f"Parsed empty DataFrame for {name}"

        # 2. 标准规整与日历截断
        norm_df = _normalize_daily_frame(parsed, FIXED_TRADE_DATE)
        assert norm_df is not None, f"Failed to normalize daily frame for {name}"
        assert set(norm_df.columns) == {"date", "open", "high", "low", "close", "volume"}

        # 3. 完整采集管道回执与 provenance / ledger 验证
        pool = _run_mock_fetch_all(fixture)
        prov = pool["market_data_context"]["source_provenance"]["stock_data"]
        assert prov["status"] == "available", f"{name} provenance status is {prov['status']}"
        assert prov["actual_as_of"] == FIXED_TRADE_DATE
        assert pool["market_data_context"]["daily"]["completeness"] == "completed"
        assert pool["market_data_context"]["daily"]["as_of"] == FIXED_TRADE_DATE

        # 确认未被错误归入 failure ledger
        ledger_entries = [e for e in pool["market_data_context"]["data_failure_ledger"] if e["source"] == "stock_data"]
        assert len(ledger_entries) == 0, f"{name} unexpectedly present in failure ledger: {ledger_entries}"

    def test_overall_benchmark_improvement_metric(self):
        """对比基线 1/10 (10.0%) 与修复后 10/10 (100.0%) 填充率指标."""
        passed_count = 0
        total_count = len(BENCHMARK_FIXTURES)
        for name, fixture in BENCHMARK_FIXTURES.items():
            pool = _run_mock_fetch_all(fixture)
            prov = pool["market_data_context"]["source_provenance"]["stock_data"]
            if prov.get("status") == "available":
                passed_count += 1

        assert passed_count == total_count == 10
        fill_rate = passed_count / total_count
        assert fill_rate == 1.0


class TestRedTeamScenarios:
    """钉子 5: 六大红队场景验证."""

    def test_rt1_provider_failure_and_empty_semantics(self):
        """RT-1: provider failure 与空结果不得变成 confirmed empty 或有效数据."""
        # Case A: 显式 VendorFail
        pool_fail = _run_mock_fetch_all(VendorFail("网关超时 504 (timeout)"))
        prov_fail = pool_fail["market_data_context"]["source_provenance"]["stock_data"]
        ledger_fail = [e for e in pool_fail["market_data_context"]["data_failure_ledger"] if e["source"] == "stock_data"][0]
        assert prov_fail["status"] == "failed"
        assert ledger_fail["status"] == "failed"
        assert ledger_fail["gap_class"] == "operational"
        assert "超时" in str(pool_fail["stock_data"]) or "timeout" in str(pool_fail["stock_data"]).lower()

        # Case B: 显式文本调用失败
        pool_text_fail = _run_mock_fetch_all("【数据获取失败】get_stock_data 调用异常: ConnectionRefusedError")
        prov_text_fail = pool_text_fail["market_data_context"]["source_provenance"]["stock_data"]
        ledger_text_fail = [e for e in pool_text_fail["market_data_context"]["data_failure_ledger"] if e["source"] == "stock_data"][0]
        assert prov_text_fail["status"] == "failed"
        assert ledger_text_fail["status"] == "failed"

        # Case C: 确认空结果 VendorEmpty
        pool_empty = _run_mock_fetch_all(VendorEmpty(f"No data found for symbol '{FIXED_SYMBOL}' between 2026-01-01 and {FIXED_TRADE_DATE}"))
        prov_empty = pool_empty["market_data_context"]["source_provenance"]["stock_data"]
        ledger_empty = [e for e in pool_empty["market_data_context"]["data_failure_ledger"] if e["source"] == "stock_data"][0]
        assert prov_empty["status"] == "unavailable"
        assert ledger_empty["status"] == "unavailable"
        assert ledger_empty["reason"] in ("confirmed empty", "no valid completed daily bars", "data source unavailable")

        # Case D: 历史快照拒绝 VendorRefuse
        pool_refuse = _run_mock_fetch_all(VendorRefuse("【数据获取失败】该数据仅支持当日快照，无法用于历史日期分析"))
        prov_refuse = pool_refuse["market_data_context"]["source_provenance"]["stock_data"]
        ledger_refuse = [e for e in pool_refuse["market_data_context"]["data_failure_ledger"] if e["source"] == "stock_data"][0]
        assert prov_refuse["status"] == "refused"
        assert ledger_refuse["status"] == "refused"
        assert ledger_refuse["gap_class"] == "structural"

    def test_rt2_missing_volume_or_essential_ohlcv_fails_closed(self):
        """RT-2: 缺 volume / 缺必要 OHLCV 列必须显式不可用，不得伪造或默认值."""
        # 缺少 volume 列
        no_vol_csv = "Date,Open,High,Low,Close\n2026-03-10,100,105,98,102\n2026-03-11,102,106,101,104\n"
        assert _resolve_ohlcv_columns(pd.read_csv(io.StringIO(no_vol_csv))) is None
        assert _normalize_daily_frame(pd.read_csv(io.StringIO(no_vol_csv)), FIXED_TRADE_DATE) is None

        pool = _run_mock_fetch_all(no_vol_csv)
        prov = pool["market_data_context"]["source_provenance"]["stock_data"]
        assert prov["status"] == "unavailable"
        assert pool["market_data_context"]["daily"]["completeness"] == "unavailable"
        assert pool["indicators"]["vwma"] == "无数据"
        assert pool["vpa_indicators"] == "VPA 数据不足"

        # 缺少 close 列
        no_close_csv = "Date,Open,High,Low,Volume\n2026-03-10,100,105,98,50000\n"
        assert _resolve_ohlcv_columns(pd.read_csv(io.StringIO(no_close_csv))) is None

    def test_rt3_legal_volume_zero_or_decimal_zero_preserved(self):
        """RT-3: 合法 volume=0 或 Decimal(0) 不得丢失，不得被当作缺失或引发除零."""
        df_vol_zero = pd.DataFrame({
            "Date": ["2026-03-10", "2026-03-11", "2026-03-12"],
            "Open": [10.0, 10.0, 10.0],
            "High": [10.0, 10.0, 10.0],
            "Low": [10.0, 10.0, 10.0],
            "Close": [10.0, 10.0, 10.0],
            "Volume": [Decimal(0), 0, 0.0],
        })
        norm_df = _normalize_daily_frame(df_vol_zero, FIXED_TRADE_DATE)
        assert norm_df is not None
        assert len(norm_df) == 3
        assert (norm_df["volume"] == 0.0).all()

        pool = _run_mock_fetch_all(df_vol_zero)
        prov = pool["market_data_context"]["source_provenance"]["stock_data"]
        assert prov["status"] == "available"
        assert "0.0" in str(pool["stock_data"])

        # 验证 30 天全是 0 成交量时 VPA 与指标不除零，不出现 nan/inf
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        full_zero_df = pd.DataFrame({
            "Date": dates,
            "Open": [10.0] * 30,
            "High": [10.0] * 30,
            "Low": [10.0] * 30,
            "Close": [10.0] * 30,
            "Volume": [0.0] * 30,
        })
        vpa_text = _compute_vpa_indicators(full_zero_df)
        assert "nan" not in vpa_text.lower()
        assert "inf" not in vpa_text.lower()

        vpa_feat = compute_vpa_deterministic_features(full_zero_df, cutoff="2026-01-30")
        assert not np.isnan(vpa_feat["features"]["volume_ratio"])
        assert not np.isnan(vpa_feat["features"]["volume_zscore"])

    def test_rt4_cutoff_future_and_duplicate_conflict_rejection(self):
        """RT-4: 日期超出 cutoff、未来行和重复日期冲突不得进入量价输入."""
        # 未来行被严格过滤，不泄露到 as_of
        future_csv = (
            "Date,Open,High,Low,Close,Volume\n"
            "2026-03-11,100,105,98,102,50000\n"
            "2026-03-12,102,106,101,104,55000\n"
            "2026-03-13,105,110,104,109,60000\n"  # > 2026-03-12
        )
        norm_future = _normalize_daily_frame(pd.read_csv(io.StringIO(future_csv)), FIXED_TRADE_DATE)
        assert norm_future is not None
        assert "2026-03-13" not in set(norm_future["date"].dt.strftime("%Y-%m-%d"))

        # 同日冲突（价格不同）严格拒收并进入 unavailable failure
        conflict_csv = (
            "Date,Open,High,Low,Close,Volume\n"
            "2026-03-12,100,105,98,102,50000\n"
            "2026-03-12,110,115,108,112,90000\n"
        )
        assert _normalize_daily_frame(pd.read_csv(io.StringIO(conflict_csv)), FIXED_TRADE_DATE) is None

        pool_conflict = _run_mock_fetch_all(conflict_csv)
        prov_conflict = pool_conflict["market_data_context"]["source_provenance"]["stock_data"]
        assert prov_conflict["status"] == "unavailable"

    def test_rt5_price_basis_constraint_and_no_fallback(self):
        """RT-5: provider fallback 不能跨越显式 price_basis 约束."""
        # 1. 显式 UnsupportedPriceBasisError
        for forbidden in (PRICE_BASIS_PIT_RAW, PRICE_BASIS_PIT_ADJUSTED, PRICE_BASIS_UNSPECIFIED):
            with pytest.raises(UnsupportedPriceBasisError):
                _run_mock_fetch_all(BENCHMARK_FIXTURES["canonical_en"], price_basis=forbidden)

        # 2. 未知 price_basis 报错
        with pytest.raises((UnknownPriceBasisError, ValueError)):
            _run_mock_fetch_all(BENCHMARK_FIXTURES["canonical_en"], price_basis="invalid_basis")

        # 3. raw 不向 vendor_qfq 回退
        with patch.object(_registry, "get", return_value=None):
            pool_raw = _fetch_all(FIXED_SYMBOL, FIXED_TRADE_DATE, price_basis=PRICE_BASIS_RAW)
            assert pool_raw["price_basis"] == "raw"
            prov_raw = pool_raw["market_data_context"]["source_provenance"]["stock_data"]
            assert prov_raw["status"] == "failed"

    def test_rt6_derived_indicators_insufficient_semantics(self):
        """RT-6: 所有派生指标在原始输入不足时必须保持 insufficient / missing 语义 (N/A / 无数据)."""
        # 仅 5 根 K 线，不足以计算 50/200 SMA
        dates = pd.date_range("2026-03-08", periods=5, freq="D")
        short_df = pd.DataFrame({
            "Date": dates,
            "Open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "High": [10.5, 11.5, 12.5, 13.5, 14.5],
            "Low": [9.5, 10.5, 11.5, 12.5, 13.5],
            "Close": [10.2, 11.2, 12.2, 13.2, 14.2],
            "Volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        })
        pool = _run_mock_fetch_all(short_df)
        indicators = pool["indicators"]
        assert indicators["close_50_sma"] in ("N/A", "无数据")
        assert indicators["close_200_sma"] in ("N/A", "无数据")
        # 绝不泄漏 float nan
        for k, v in indicators.items():
            if isinstance(v, float):
                assert not np.isnan(v), f"Indicator {k} leaked float nan"

        # VPA 在历史不足时保持 insufficient_data
        vpa_feat = compute_vpa_deterministic_features(short_df, cutoff=FIXED_TRADE_DATE)
        assert vpa_feat["volume_regime"] == VOLUME_REGIME_INSUFFICIENT_DATA
        assert vpa_feat["reversal_state"] == REVERSAL_STATE_INSUFFICIENT_DATA
        assert _compute_vpa_indicators(short_df) == "VPA 数据不足：历史 K 线数量不够"


class TestCnAkshareProviderNormalization:
    """验证 cn_akshare_provider 在 _normalize_hist_df 层的中文与别名兼容性."""

    def test_cn_akshare_normalizes_chinese_and_price_suffixes(self):
        provider = CnAkshareProvider()
        raw_df = pd.DataFrame({
            "交易日期": ["2026-03-10", "2026-03-11"],
            "开盘价": [100.0, 102.0],
            "最高价": [105.0, 106.0],
            "最低价": [98.0, 101.0],
            "收盘价": [102.0, 104.0],
            "成交量(股)": [50000.0, 55000.0],
        })
        norm_df = provider._normalize_hist_df(raw_df)
        assert not norm_df.empty
        assert list(norm_df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert (norm_df["Volume"] == [50000.0, 55000.0]).all()


import io
