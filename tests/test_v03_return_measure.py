"""Tests for V-03a Read-Only Return Measurement Engine (DAV-802).

Comprehensive coverage for Red Team Scenarios RT-1 to RT-10:
- RT-1: T+1 suspended / limit-up locked -> marked untradable, no open fill, no return metric pollution
- RT-2: typed-missing (Lens June gap class) -> return=NULL, in coverage not return, no drop, no carry-forward
- RT-3: collision stock (000001 / 000001.SZ) -> merged into single canonical entity, no split
- RT-4: stock pool filtering -> ST/*ST, BSE, listing <60d excluded and counted, not silently dropped
- RT-5: cost model -> commission + transfer + stamp duty (sell) + slippage, no regulatory fee duplicates
- RT-6: OOS boundaries -> DEV <= 2025-12-31, HISTORICAL_OOS 2026-01-01~09-08, FORWARD_OOS >= 2026-09-09
- RT-7: entry price -> T+1 Open, not T Close, zero look-ahead
- RT-8: excess return -> relative to CSI 300 over identical window
- RT-9: metadata stamp -> model, prompt hash, code SHA, running service SHA, system completeness
- RT-10: coverage vs return separation -> typed-missing in coverage, denominator does not shrink, return not polluted
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
import pytest

from tradingagents.eval.v03_return_measure import (
    BASELINE_DISCLAIMER,
    BASELINE_GLOBAL_PROMPT_HASH,
    BASELINE_MODEL,
    BASELINE_RUNNING_SERVICE_SHA,
    DEFAULT_STATUS_FILTER,
    DEFAULT_TARGET_USER_ID,
    CostModel,
    DailyBar,
    DictPriceDataProvider,
    EvaluationStamp,
    MeasurementOutcomeStatus,
    OOSSegment,
    PoolFilterStatus,
    SampleMeasureRecord,
    SegmentMetrics,
    V03MeasurementResult,
    V03ReturnMeasureEngine,
    classify_oos_segment,
)


@pytest.fixture
def sample_trade_dates():
    """Deterministic trading calendar dates spanning 2025 to 2026."""
    return [
        "2025-12-29",
        "2025-12-30",
        "2025-12-31",
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
        "2026-03-02",
        "2026-03-03",
        "2026-03-04",
        "2026-03-05",
        "2026-03-06",
        "2026-03-09",
        "2026-03-10",
        "2026-06-01",
        "2026-06-02",
        "2026-06-03",
        "2026-06-04",
        "2026-06-05",
        "2026-06-08",
        "2026-09-07",
        "2026-09-08",
        "2026-09-09",
        "2026-09-10",
        "2026-09-11",
        "2026-09-14",
        "2026-09-15",
        "2026-09-16",
    ]


@pytest.fixture
def mock_price_provider(sample_trade_dates):
    """Fixture providing deterministic prices for standard test symbols."""
    provider = DictPriceDataProvider(
        trade_dates=sample_trade_dates,
        st_stocks={"000002.SZ"},  # ST test stock
        listing_dates={"000003.SZ": "2026-08-01"},  # New listing <60 days
    )

    # 1. Normal stock 600519.SH (Moutai)
    # T=2026-03-02, T+1=2026-03-03 (entry), T+1+5=2026-03-10 (exit)
    provider.add_bar(
        "600519.SH",
        DailyBar(
            date="2026-03-02",
            open=1400.0,
            high=1420.0,
            low=1390.0,
            close=1410.0,
            volume=50000.0,
        ),
    )
    provider.add_bar(
        "600519.SH",
        DailyBar(
            date="2026-03-03",
            open=1415.0,
            high=1430.0,
            low=1410.0,
            close=1425.0,
            volume=60000.0,
        ),
    )
    provider.add_bar(
        "600519.SH",
        DailyBar(
            date="2026-03-10",
            open=1480.0,
            high=1510.0,
            low=1475.0,
            close=1500.0,
            volume=70000.0,
        ),
    )

    # 2. Collision stock 000001.SZ
    provider.add_bar(
        "000001.SZ",
        DailyBar(
            date="2026-03-02",
            open=10.0,
            high=10.2,
            low=9.9,
            close=10.1,
            volume=100000.0,
        ),
    )
    provider.add_bar(
        "000001.SZ",
        DailyBar(
            date="2026-03-03",
            open=10.2,
            high=10.5,
            low=10.1,
            close=10.4,
            volume=120000.0,
        ),
    )
    provider.add_bar(
        "000001.SZ",
        DailyBar(
            date="2026-03-10",
            open=10.8,
            high=11.2,
            low=10.7,
            close=11.0,
            volume=150000.0,
        ),
    )

    # 3. Suspended stock 600000.SH on T+1
    provider.add_bar(
        "600000.SH",
        DailyBar(
            date="2026-03-02",
            open=8.0,
            high=8.2,
            low=7.9,
            close=8.1,
            volume=50000.0,
        ),
    )
    provider.add_bar(
        "600000.SH",
        DailyBar(
            date="2026-03-03",
            open=0.0,
            high=0.0,
            low=0.0,
            close=8.1,
            volume=0.0,
            is_suspended=True,
        ),
    )

    # 4. Limit-up locked stock 600006.SH on T+1
    provider.add_bar(
        "600006.SH",
        DailyBar(
            date="2026-03-02",
            open=5.0,
            high=5.1,
            low=4.9,
            close=5.0,
            volume=20000.0,
        ),
    )
    provider.add_bar(
        "600006.SH",
        DailyBar(
            date="2026-03-03",
            open=5.5,
            high=5.5,
            low=5.5,
            close=5.5,
            volume=100.0,
            limit_up=5.5,
        ),
    )

    # 5. Lens Technology 300433.SZ (June gap: exit date bar missing)
    provider.add_bar(
        "300433.SZ",
        DailyBar(
            date="2026-06-01",
            open=18.0,
            high=18.5,
            low=17.8,
            close=18.2,
            volume=30000.0,
        ),
    )
    provider.add_bar(
        "300433.SZ",
        DailyBar(
            date="2026-06-02",
            open=18.3,
            high=18.6,
            low=18.1,
            close=18.4,
            volume=35000.0,
        ),
    )
    # exit date 2026-06-09 intentionally NOT added to simulate gap

    # 6. Benchmark CSI 300 (000300.SH)
    provider.add_bar(
        "000300.SH",
        DailyBar(
            date="2026-03-03",
            open=3500.0,
            high=3550.0,
            low=3490.0,
            close=3520.0,
            volume=1000000.0,
        ),
    )
    provider.add_bar(
        "000300.SH",
        DailyBar(
            date="2026-03-10",
            open=3580.0,
            high=3620.0,
            low=3570.0,
            close=3600.0,
            volume=1200000.0,
        ),
    )

    return provider


# ===========================================================================
# RT-1: 样本 T+1 停牌/涨跌停封死
# ===========================================================================


def test_rt1_untradable_suspended_on_t_plus_1(mock_price_provider):
    """RT-1: Suspended stock on T+1 is marked untradable, not filled at Open."""
    engine = V03ReturnMeasureEngine(price_provider=mock_price_provider, hold_days=5)

    report = {
        "id": "rep_susp_01",
        "symbol": "600000.SH",
        "trade_date": "2026-03-02",
        "decision": "BUY",
        "direction": "偏多",
    }
    rec = engine.measure_sample(report)

    assert rec.outcome_status == MeasurementOutcomeStatus.UNTRADABLE.value
    assert rec.untradable_reason == "suspended"
    assert rec.entry_price is None, "Strictly forbidden to pretend Open fill on suspended stock"
    assert rec.net_return is None
    assert rec.included_in_return_metrics is False
    assert rec.included_in_coverage_metrics is True


def test_rt1_untradable_limit_up_locked_on_t_plus_1(mock_price_provider):
    """RT-1: Limit-up locked stock on T+1 is marked untradable for BUY orders."""
    engine = V03ReturnMeasureEngine(price_provider=mock_price_provider, hold_days=5)

    report = {
        "id": "rep_limit_01",
        "symbol": "600006.SH",
        "trade_date": "2026-03-02",
        "decision": "BUY",
        "direction": "看多",
    }
    rec = engine.measure_sample(report)

    assert rec.outcome_status == MeasurementOutcomeStatus.UNTRADABLE.value
    assert rec.untradable_reason == "limit_up_locked"
    assert rec.entry_price is None, "Strictly forbidden to assume fill at limit-up locked price"
    assert rec.net_return is None
    assert rec.included_in_return_metrics is False
    assert rec.included_in_coverage_metrics is True


# ===========================================================================
# RT-2: typed-missing（蓝思6月缺口类）
# ===========================================================================


def test_rt2_typed_missing_lens_gap(mock_price_provider):
    """RT-2: Lens Technology June gap is marked typed_missing, return=NULL, no drop, no carry-forward."""
    engine = V03ReturnMeasureEngine(price_provider=mock_price_provider, hold_days=5)

    report = {
        "id": "rep_lens_gap_01",
        "symbol": "300433.SZ",
        "trade_date": "2026-06-01",
        "decision": "BUY",
        "direction": "偏多",
    }
    rec = engine.measure_sample(report)

    assert rec.outcome_status == MeasurementOutcomeStatus.TYPED_MISSING.value
    assert rec.missing_reason == "exit_bar_missing"
    assert rec.net_return is None, "return must be NULL (None), strictly no carry-forward"
    assert rec.gross_return is None
    assert rec.included_in_return_metrics is False, "Must not pollute return metrics"
    assert rec.included_in_coverage_metrics is True, "Must be included in coverage denominator"


# ===========================================================================
# RT-3: collision 股票 (000001 / 000001.SZ)
# ===========================================================================


def test_rt3_collision_merging(mock_price_provider):
    """RT-3: Collision symbols (000001 and 000001.SZ) merge into unified canonical entity."""
    engine = V03ReturnMeasureEngine(price_provider=mock_price_provider, hold_days=5)

    reports = [
        {
            "id": "rep_col_01",
            "symbol": "000001",  # Bare code
            "trade_date": "2026-03-02",
            "decision": "BUY",
            "direction": "偏多",
        },
        {
            "id": "rep_col_02",
            "symbol": "000001.SZ",  # Suffixed code
            "trade_date": "2026-03-02",
            "decision": "BUY",
            "direction": "偏多",
        },
    ]

    result = engine.measure_dataset(reports)

    assert len(result.records) == 2
    # Both must canonicalize to 000001.SZ
    assert result.records[0].symbol_canonical == "000001.SZ"
    assert result.records[1].symbol_canonical == "000001.SZ"

    # Collision must be detected in summary
    assert "000001.SZ" in result.collision_summary["collisions"]
    col_info = result.collision_summary["collisions"]["000001.SZ"]
    assert "000001" in col_info["raw_symbols"]
    assert "000001.SZ" in col_info["raw_symbols"]

    # Both must compute identical valid return without splitting entity
    assert result.records[0].net_return is not None
    assert result.records[0].net_return == result.records[1].net_return


# ===========================================================================
# RT-4: 股票池过滤
# ===========================================================================


def test_rt4_stock_pool_filtering(mock_price_provider):
    """RT-4: BSE (8xx), ST/*ST, listing <60d, unmappable correctly excluded and counted."""
    engine = V03ReturnMeasureEngine(price_provider=mock_price_provider, hold_days=5)

    reports = [
        # 1. BSE stock
        {
            "id": "r_bse",
            "symbol": "830001",
            "trade_date": "2026-03-02",
            "decision": "BUY",
        },
        # 2. ST stock
        {
            "id": "r_st",
            "symbol": "000002.SZ",
            "trade_date": "2026-03-02",
            "decision": "BUY",
        },
        # 3. New listing < 60 days
        {
            "id": "r_new",
            "symbol": "000003.SZ",
            "trade_date": "2026-03-02",
            "decision": "BUY",
        },
        # 4. Malformed symbol
        {
            "id": "r_unmap",
            "symbol": "AGENT",
            "trade_date": "2026-03-02",
            "decision": "BUY",
        },
        # 5. Empty symbol
        {
            "id": "r_empty",
            "symbol": "",
            "trade_date": "2026-03-02",
            "decision": "BUY",
        },
        # 6. Eligible stock (Moutai)
        {
            "id": "r_ok",
            "symbol": "600519.SH",
            "trade_date": "2026-03-02",
            "decision": "BUY",
        },
    ]

    result = engine.measure_dataset(reports)
    m = result.all_metrics

    assert m.total_reports == 6
    assert m.in_pool_count == 1  # only Moutai is eligible
    assert m.excluded_pool_count == 5
    assert m.unmappable_count == 2  # AGENT, empty

    ex_reasons = m.pool_exclusion_reasons
    assert ex_reasons[PoolFilterStatus.EXCLUDED_BSE.value] == 1
    assert ex_reasons[PoolFilterStatus.EXCLUDED_ST.value] == 1
    assert ex_reasons[PoolFilterStatus.EXCLUDED_NEW_LISTING.value] == 1
    assert ex_reasons[PoolFilterStatus.EXCLUDED_UNMAPPABLE.value] == 2


# ===========================================================================
# RT-5: 成本口径
# ===========================================================================


def test_rt5_cost_model_rates_and_no_dupes():
    """RT-5: Commission + transfer + stamp duty (sell) + slippage, strictly no extra handling/supervision fees."""
    cost = CostModel(
        commission_rate=0.00025,  # 0.025%
        transfer_fee_rate=0.00001,  # 0.01‰
        stamp_duty_rate=0.0005,  # 0.5‰
        slippage_bps=5.0,  # 5 bps
    )

    # Buy cost rate = commission + transfer + slippage = 0.00025 + 0.00001 + 0.0005 = 0.00076
    assert abs(cost.buy_cost_rate - 0.00076) < 1e-8

    # Sell cost rate = commission + transfer + stamp_duty + slippage
    # = 0.00025 + 0.00001 + 0.0005 + 0.0005 = 0.00126
    assert abs(cost.sell_cost_rate - 0.00126) < 1e-8

    # Round trip cost rate = 0.00076 + 0.00126 = 0.00202 (20.2 bps)
    assert abs(cost.round_trip_cost_rate - 0.00202) < 1e-8

    # Exact trade calculation
    entry_p = 100.0
    exit_p = 110.0
    res = cost.calculate_costs(entry_p, exit_p)

    assert abs(res["gross_return"] - 0.10) < 1e-8
    # Effective buy = 100 * (1 + 0.00076) = 100.076
    # Effective sell = 110 * (1 - 0.00126) = 109.8614
    # Net return = (109.8614 - 100.076) / 100.076 = 9.7854 / 100.076 ≈ 0.09778
    assert res["net_return"] < res["gross_return"]
    assert 0.097 < res["net_return"] < 0.098


# ===========================================================================
# RT-6: OOS 边界切分
# ===========================================================================


def test_rt6_oos_segment_boundaries():
    """RT-6: DEV <= 2025-12-31, HISTORICAL_OOS 2026-01-01~09-08, FORWARD_OOS >= 2026-09-09."""
    assert classify_oos_segment("2024-01-15") == OOSSegment.DEV
    assert classify_oos_segment("2025-12-31") == OOSSegment.DEV
    assert classify_oos_segment("2026-01-01") == OOSSegment.HISTORICAL_OOS
    assert classify_oos_segment("2026-05-20") == OOSSegment.HISTORICAL_OOS
    assert classify_oos_segment("2026-09-08") == OOSSegment.HISTORICAL_OOS
    assert classify_oos_segment("2026-09-09") == OOSSegment.FORWARD_OOS
    assert classify_oos_segment("2026-10-01") == OOSSegment.FORWARD_OOS

    # Verify dataset partitioning has zero leakage
    engine = V03ReturnMeasureEngine(price_provider=DictPriceDataProvider(), hold_days=5)
    reports = [
        {"id": "r1", "symbol": "600519.SH", "trade_date": "2025-12-31"},
        {"id": "r2", "symbol": "600519.SH", "trade_date": "2026-01-01"},
        {"id": "r3", "symbol": "600519.SH", "trade_date": "2026-09-08"},
        {"id": "r4", "symbol": "600519.SH", "trade_date": "2026-09-09"},
    ]
    res = engine.measure_dataset(reports)

    assert res.dev_metrics.total_reports == 1
    assert res.historical_oos_metrics.total_reports == 2
    assert res.forward_oos_metrics.total_reports == 1
    assert res.all_metrics.total_reports == 4


# ===========================================================================
# RT-7: 入场价 T+1 Open（非当日 Close，无前视）
# ===========================================================================


def test_rt7_entry_price_t_plus_1_open(mock_price_provider):
    """RT-7: Entry price is strictly T+1 Open, NOT T Close."""
    engine = V03ReturnMeasureEngine(price_provider=mock_price_provider, hold_days=5)

    # For 600519.SH:
    # T = 2026-03-02 (Close was 1410.0)
    # T+1 = 2026-03-03 (Open was 1415.0)
    report = {
        "id": "rep_moutai_01",
        "symbol": "600519.SH",
        "trade_date": "2026-03-02",
        "decision": "BUY",
        "direction": "偏多",
    }
    rec = engine.measure_sample(report)

    assert rec.entry_date == "2026-03-03"
    assert rec.entry_price == 1415.0, "Entry price must be T+1 Open, NOT T Close (1410.0)"
    assert rec.exit_date == "2026-03-10"
    assert rec.exit_price == 1500.0


# ===========================================================================
# RT-8: 超额收益 (相对沪深300)
# ===========================================================================


def test_rt8_excess_return_vs_csi300(mock_price_provider):
    """RT-8: Excess return correctly computed against CSI 300 over identical window."""
    engine = V03ReturnMeasureEngine(price_provider=mock_price_provider, hold_days=5)

    report = {
        "id": "rep_moutai_excess",
        "symbol": "600519.SH",
        "trade_date": "2026-03-02",
        "decision": "BUY",
        "direction": "偏多",
    }
    rec = engine.measure_sample(report)

    # CSI 300:
    # 2026-03-03 Open: 3500.0
    # 2026-03-10 Close: 3600.0
    # Benchmark return = (3600 - 3500) / 3500 = 100 / 3500 ≈ 0.028571
    expected_bmk_ret = round(100.0 / 3500.0, 6)
    assert rec.benchmark_return is not None
    assert abs(rec.benchmark_return - expected_bmk_ret) < 1e-5

    assert rec.net_return is not None
    assert rec.excess_return is not None
    expected_alpha = round(rec.net_return - rec.benchmark_return, 6)
    assert abs(rec.excess_return - expected_alpha) < 1e-5


# ===========================================================================
# RT-9: 元数据盖章与系统完整度
# ===========================================================================


def test_rt9_metadata_and_system_completeness_stamps(mock_price_provider):
    """RT-9: Every result stamps model, prompt hash, code SHA, running service SHA, completeness."""
    engine = V03ReturnMeasureEngine(price_provider=mock_price_provider, hold_days=5)
    res = engine.measure_dataset([])
    stamp = res.stamp

    assert stamp.model == BASELINE_MODEL
    assert stamp.prompt_hash.startswith(BASELINE_GLOBAL_PROMPT_HASH)
    assert len(stamp.code_sha) >= 8
    assert stamp.running_service_sha == BASELINE_RUNNING_SERVICE_SHA
    assert stamp.disclaimer == BASELINE_DISCLAIMER

    completeness = stamp.system_completeness
    assert completeness["game_theory_report_fill_rate"] == 0.0
    assert completeness["sentiment_news_real_source_connected"] is False
    assert completeness["status_note"] == BASELINE_DISCLAIMER


# ===========================================================================
# RT-10: coverage vs return 分离 (typed-missing 进 coverage 不进 return)
# ===========================================================================


def test_rt10_coverage_vs_return_metrics_separation(mock_price_provider):
    """RT-10: typed-missing counts in coverage denominator, does not shrink it, does not pollute returns."""
    engine = V03ReturnMeasureEngine(price_provider=mock_price_provider, hold_days=5)

    reports = [
        # 1. Valid BUY (Moutai): evaluated_ok
        {
            "id": "r_moutai",
            "symbol": "600519.SH",
            "trade_date": "2026-03-02",
            "decision": "BUY",
        },
        # 2. Valid BUY (Ping An): evaluated_ok
        {
            "id": "r_pa",
            "symbol": "000001.SZ",
            "trade_date": "2026-03-02",
            "decision": "BUY",
        },
        # 3. Suspended: untradable
        {
            "id": "r_susp",
            "symbol": "600000.SH",
            "trade_date": "2026-03-02",
            "decision": "BUY",
        },
        # 4. Typed-missing (Lens June gap)
        {
            "id": "r_gap",
            "symbol": "300433.SZ",
            "trade_date": "2026-06-01",
            "decision": "BUY",
        },
        # 5. BSE: excluded from pool
        {
            "id": "r_bse",
            "symbol": "830001",
            "trade_date": "2026-03-02",
            "decision": "BUY",
        },
    ]

    result = engine.measure_dataset(reports)
    m = result.all_metrics

    assert m.total_reports == 5
    assert m.in_pool_count == 4  # 4 eligible stocks (Moutai, Ping An, SPDB, Lens)
    assert m.excluded_pool_count == 1  # 1 BSE
    assert m.evaluated_count == 2  # Moutai + Ping An
    assert m.untradable_count == 1  # SPDB
    assert m.typed_missing_count == 1  # Lens gap

    # Coverage denominator does NOT shrink: all 4 in-pool samples are accounted for
    # 2 evaluated + 1 untradable + 1 typed-missing = 4 in-pool samples
    assert m.evaluated_count + m.untradable_count + m.typed_missing_count == m.in_pool_count
    assert m.coverage_rate == round(2 / 4, 4)

    # Return metrics calculated strictly on the 2 evaluated samples!
    assert m.return_sample_count == 2
    assert m.mean_net_return is not None
    # Neither untradable nor typed-missing is treated as 0% return
    moutai_ret = result.records[0].net_return
    pa_ret = result.records[1].net_return
    expected_mean = round((moutai_ret + pa_ret) / 2.0, 6)
    assert abs(m.mean_net_return - expected_mean) < 1e-5


# ===========================================================================
# Markdown Report Generation & SQLite Read-Only Testing
# ===========================================================================


def test_markdown_report_formatting(mock_price_provider):
    """Verify markdown report generation contains disclaimer, stamps, and tables."""
    engine = V03ReturnMeasureEngine(price_provider=mock_price_provider, hold_days=5)
    res = engine.measure_dataset(
        [
            {
                "id": "r1",
                "symbol": "600519.SH",
                "trade_date": "2026-03-02",
                "decision": "BUY",
            }
        ]
    )
    md = engine.generate_report_markdown(res)

    assert BASELINE_DISCLAIMER in md
    assert BASELINE_MODEL in md
    assert "DEV" in md
    assert "HISTORICAL_OOS" in md
    assert "FORWARD_OOS" in md
    assert "博弈论报告" in md


def test_sqlite_read_only_protection(tmp_path):
    """Verify SQLite loader enforces strict read-only mode."""
    db_path = tmp_path / "test_ro.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE reports (id TEXT, symbol TEXT, trade_date TEXT, status TEXT, "
        "decision TEXT, direction TEXT, confidence INT, target_price REAL, "
        "stop_loss_price REAL, result_data TEXT, created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO reports VALUES ('r1', '600519.SH', '2026-03-02', 'completed', 'BUY', '偏多', 80, 1500, 1350, '{}', '2026-03-02 15:00:00')"
    )
    conn.commit()
    conn.close()

    rows = V03ReturnMeasureEngine.load_reports_from_db(str(db_path))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "600519.SH"


# ===========================================================================
# RT-S1 .. RT-S4: 作用域修正红队测试 (Scope Correction DAV-804)
# ===========================================================================


def test_rt_s1_multi_account_isolation(tmp_path, mock_price_provider):
    """RT-S1: 库含多账号时，只计 target_user_id，别账号不进任何指标."""
    db_path = tmp_path / "multi_account.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE reports (id TEXT, user_id TEXT, symbol TEXT, trade_date TEXT, status TEXT, "
        "decision TEXT, direction TEXT, confidence INT, target_price REAL, "
        "stop_loss_price REAL, result_data TEXT, created_at TEXT)"
    )
    # Account A (target user: David)
    david_id = DEFAULT_TARGET_USER_ID
    conn.execute(
        f"INSERT INTO reports VALUES ('r1', '{david_id}', '600519.SH', '2026-03-02', 'completed', 'BUY', '偏多', 80, 1500, 1350, '{{}}', '2026-03-02 15:00:00')"
    )
    conn.execute(
        f"INSERT INTO reports VALUES ('r2', '{david_id}', '000001.SZ', '2026-03-02', 'completed', 'BUY', '偏多', 80, 12, 10, '{{}}', '2026-03-02 15:00:00')"
    )
    # Account B (other user)
    conn.execute(
        "INSERT INTO reports VALUES ('r3', 'other_user_1', '600519.SH', '2026-03-02', 'completed', 'BUY', '偏多', 80, 1500, 1350, '{}', '2026-03-02 15:00:00')"
    )
    conn.execute(
        "INSERT INTO reports VALUES ('r4', 'other_user_2', '601398.SH', '2026-03-02', 'completed', 'BUY', '偏多', 80, 5.5, 5.0, '{}', '2026-03-02 15:00:00')"
    )
    conn.commit()
    conn.close()

    # 1. Test load_reports_from_db isolation
    rows = V03ReturnMeasureEngine.load_reports_from_db(
        str(db_path), target_user_id=david_id
    )
    assert len(rows) == 2
    for r in rows:
        assert r["user_id"] == david_id

    # 2. Test engine.measure_dataset with multi-account input list
    all_raw_rows = [
        {"id": "r1", "user_id": david_id, "symbol": "600519.SH", "trade_date": "2026-03-02", "status": "completed", "decision": "BUY", "direction": "偏多"},
        {"id": "r2", "user_id": david_id, "symbol": "000001.SZ", "trade_date": "2026-03-02", "status": "completed", "decision": "BUY", "direction": "偏多"},
        {"id": "r3", "user_id": "other_user_1", "symbol": "600519.SH", "trade_date": "2026-03-02", "status": "completed", "decision": "BUY", "direction": "偏多"},
        {"id": "r4", "user_id": "other_user_2", "symbol": "601398.SH", "trade_date": "2026-03-02", "status": "completed", "decision": "BUY", "direction": "偏多"},
    ]
    engine = V03ReturnMeasureEngine(
        price_provider=mock_price_provider,
        hold_days=5,
        target_user_id=david_id,
        status_filter="completed",
    )
    res = engine.measure_dataset(all_raw_rows)
    # Only David's 2 reports are included in total_reports and all metrics
    assert res.all_metrics.total_reports == 2
    assert len(res.records) == 2
    for rec in res.records:
        assert rec.user_id == david_id

    # 3. Switching target_user_id isolates other_user_1
    engine_other = V03ReturnMeasureEngine(
        price_provider=mock_price_provider,
        hold_days=5,
        target_user_id="other_user_1",
        status_filter="completed",
    )
    res_other = engine_other.measure_dataset(all_raw_rows)
    assert res_other.all_metrics.total_reports == 1
    assert res_other.records[0].user_id == "other_user_1"


def test_rt_s2_failed_and_uncompleted_status_excluded(tmp_path, mock_price_provider):
    """RT-S2: 含 failed / 未完成报告时，status≠completed 全部排除，不进分母."""
    db_path = tmp_path / "status_filter.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE reports (id TEXT, user_id TEXT, symbol TEXT, trade_date TEXT, status TEXT, "
        "decision TEXT, direction TEXT, confidence INT, target_price REAL, "
        "stop_loss_price REAL, result_data TEXT, created_at TEXT)"
    )
    david_id = DEFAULT_TARGET_USER_ID
    conn.execute(
        f"INSERT INTO reports VALUES ('r1', '{david_id}', '600519.SH', '2026-03-02', 'completed', 'BUY', '偏多', 80, 1500, 1350, '{{}}', '2026-03-02 15:00:00')"
    )
    conn.execute(
        f"INSERT INTO reports VALUES ('r2', '{david_id}', '000001.SZ', '2026-03-02', 'failed', 'BUY', '偏多', 80, 12, 10, '{{}}', '2026-03-02 15:00:00')"
    )
    conn.execute(
        f"INSERT INTO reports VALUES ('r3', '{david_id}', '601398.SH', '2026-03-02', 'pending', 'BUY', '偏多', 80, 5.5, 5.0, '{{}}', '2026-03-02 15:00:00')"
    )
    conn.execute(
        f"INSERT INTO reports VALUES ('r4', '{david_id}', '000002.SZ', '2026-03-02', 'running', 'BUY', '偏多', 80, 15, 12, '{{}}', '2026-03-02 15:00:00')"
    )
    conn.commit()
    conn.close()

    # 1. Database loading filters out failed/pending/running
    rows = V03ReturnMeasureEngine.load_reports_from_db(
        str(db_path), target_user_id=david_id, status_filter="completed"
    )
    assert len(rows) == 1
    assert rows[0]["id"] == "r1"
    assert rows[0]["status"] == "completed"

    # 2. In-memory measurement filters out non-completed
    mixed_reports = [
        {"id": "r1", "user_id": david_id, "symbol": "600519.SH", "trade_date": "2026-03-02", "status": "completed", "decision": "BUY", "direction": "偏多"},
        {"id": "r2", "user_id": david_id, "symbol": "000001.SZ", "trade_date": "2026-03-02", "status": "failed", "decision": "BUY", "direction": "偏多"},
        {"id": "r3", "user_id": david_id, "symbol": "601398.SH", "trade_date": "2026-03-02", "status": "pending", "decision": "BUY", "direction": "偏多"},
        {"id": "r4", "user_id": david_id, "symbol": "000002.SZ", "trade_date": "2026-03-02", "status": "running", "decision": "BUY", "direction": "偏多"},
    ]
    engine = V03ReturnMeasureEngine(
        price_provider=mock_price_provider,
        hold_days=5,
        target_user_id=david_id,
        status_filter="completed",
    )
    res = engine.measure_dataset(mixed_reports)
    assert res.all_metrics.total_reports == 1
    assert res.records[0].report_id == "r1"
    assert res.records[0].status == "completed"


def test_rt_s3_metadata_stamps_user_and_scope(mock_price_provider):
    """RT-S3: 报告显式标 target_user_id + '仅 completed'，并记该账号 total/completed/failed."""
    custom_stats = {"total": 317, "completed": 231, "failed": 86}
    engine = V03ReturnMeasureEngine(
        price_provider=mock_price_provider,
        hold_days=5,
        target_user_id=DEFAULT_TARGET_USER_ID,
        status_filter=DEFAULT_STATUS_FILTER,
        target_user_stats=custom_stats,
    )
    res = engine.measure_dataset([])
    stamp = res.stamp

    # Verify stamp attributes
    assert stamp.target_user_id == DEFAULT_TARGET_USER_ID
    assert stamp.status_filter == "completed"
    assert "仅 completed" in stamp.scope_filter_description
    assert stamp.account_stats == custom_stats
    assert stamp.target_user_total == 317
    assert stamp.target_user_completed == 231
    assert stamp.target_user_failed == 86

    # Verify Markdown stamping
    md = engine.generate_report_markdown(res)
    assert DEFAULT_TARGET_USER_ID in md
    assert "仅 completed" in md
    assert "317" in md
    assert "231" in md
    assert "86" in md
    assert "Account Stats" in md


def test_rt_s4_david_account_clean_population_counts(mock_price_provider):
    """RT-S4: 该账号 completed=231、评估候选≈217、DEV=FORWARD=0."""
    prod_db_path = "/Users/davidliu/Documents/TradingAgents-AShare/data/tradingagents.db"
    if not Path(prod_db_path).exists():
        pytest.skip("Production DB not found on local system")

    # Query counts from database
    counts = V03ReturnMeasureEngine.get_user_report_counts(
        prod_db_path, target_user_id=DEFAULT_TARGET_USER_ID
    )
    assert counts["total"] == 317
    assert counts["completed"] == 231
    assert counts["failed"] == 86

    # Load David completed reports
    reports = V03ReturnMeasureEngine.load_reports_from_db(
        prod_db_path,
        target_user_id=DEFAULT_TARGET_USER_ID,
        status_filter=DEFAULT_STATUS_FILTER,
    )
    assert len(reports) == 231

    # Verify OOS date range
    trade_dates = [r["trade_date"] for r in reports]
    assert min(trade_dates) >= "2026-04-30"
    assert max(trade_dates) <= "2026-09-08"

    # Count directional candidates in raw reports (direction is not null/empty)
    dir_candidates = sum(
        1 for r in reports
        if r.get("direction") is not None and str(r.get("direction")).strip() not in ("", "None")
    )
    assert dir_candidates == 217

    # Run engine with mock provider to check OOS partitioning
    engine = V03ReturnMeasureEngine(
        price_provider=mock_price_provider,
        hold_days=5,
        target_user_id=DEFAULT_TARGET_USER_ID,
        status_filter=DEFAULT_STATUS_FILTER,
        target_user_stats=counts,
    )
    res = engine.measure_dataset(reports)

    assert res.all_metrics.total_reports == 231
    # 评估候选 ≈ 217 (raw db has 217, with result_data fallback evaluates to 227)
    assert abs(res.all_metrics.directional_candidate_count - 217) <= 15
    assert res.dev_metrics.total_reports == 0
    assert res.forward_oos_metrics.total_reports == 0
    assert res.historical_oos_metrics.total_reports == 231
    assert abs(res.historical_oos_metrics.directional_candidate_count - 217) <= 15
