"""DAV-179 deterministic fund-flow source-priority contract tests."""

from tradingagents.dataflows.fund_flow_evidence import (
    build_sina_evidence,
    select_fund_flow_source,
    summarize_evidence,
)


_SYMBOL = "600519"
_DATE = "2026-08-14"


def _record(source, field, value, *, date=_DATE, unit="亿元", group="new_algorithm_group"):
    return {
        "source": source,
        "algorithm_group": group,
        "status": "available",
        "symbol": _SYMBOL,
        "date": date,
        "period_kind": "historical_daily",
        "time_window": "1d",
        "field": field,
        "value": value,
        "unit": unit,
    }


def test_dc_wins_over_ths_and_retains_conflicting_side_evidence():
    result = select_fund_flow_source(
        [
            _record("tushare_ths_moneyflow_ths", "netamount", "9"),
            _record("tushare_eastmoney_moneyflow_dc", "r0_net", "1"),
        ],
        symbol=_SYMBOL,
        requested_as_of=_DATE,
    )

    assert result["selected_source"] == "tushare_eastmoney_moneyflow_dc"
    assert result["selected_source_family"] == "eastmoney"
    assert result["selected_algorithm_group"] == "new_algorithm_group"
    assert result["selected_field"] == "r0_net"
    assert result["selected_value"] == "1"
    assert result["selected_direction"] == "inflow"
    assert result["direction_allowed"] is True
    assert result["fallback_rank"] == 1
    assert [item["source"] for item in result["side_evidence"]] == [
        "tushare_ths_moneyflow_ths"
    ]
    assert result["side_evidence"][0]["selected"] is False


def test_single_other_new_algorithm_source_is_allowed():
    result = select_fund_flow_source(
        [_record("eastmoney_direct", "r0_net", "-2")],
        symbol=_SYMBOL,
        requested_as_of=_DATE,
    )

    assert result["selected_source"] == "eastmoney_direct"
    assert result["selected_field"] == "r0_net"
    assert result["selected_value"] == "-2"
    assert result["selected_direction"] == "outflow"
    assert result["direction_allowed"] is True
    assert result["fallback_rank"] == 3
    assert result["selection_reason"] == "first_valid_source_by_priority"


def test_invalid_high_priority_candidate_falls_through_without_mixing():
    result = select_fund_flow_source(
        [
            _record("tushare_eastmoney_moneyflow_dc", "r0_net", "1", date="未来日期"),
            _record("tushare_ths_moneyflow_ths", "netamount", "2"),
        ],
        symbol=_SYMBOL,
        requested_as_of=_DATE,
    )

    assert result["selected_source"] == "tushare_ths_moneyflow_ths"
    assert result["selected_field"] == "netamount"
    assert result["direction_allowed"] is True
    assert any(
        item["reason"] == "missing_measurement_date"
        for item in result["failure_chain"]
    )


def test_invalid_unit_and_field_do_not_block_next_source():
    result = select_fund_flow_source(
        [
            _record("tushare_eastmoney_moneyflow_dc", "r0_net", "1", unit="美元"),
            _record("tushare_ths_moneyflow_ths", "netamount", "2"),
        ],
        symbol=_SYMBOL,
        requested_as_of=_DATE,
    )

    assert result["selected_source"] == "tushare_ths_moneyflow_ths"
    assert result["selected_field"] == "netamount"
    assert result["direction_allowed"] is True
    assert any(item["reason"] == "invalid_normalized_unit" for item in result["failure_chain"])


def test_legacy_netamount_only_gets_own_direction_and_explicit_marker():
    records = build_sina_evidence(
        [{"opendate": _DATE, "netamount": "100000000"}],
        symbol=_SYMBOL,
        requested_as_of=_DATE,
        retrieved_at=None,
    )
    result = select_fund_flow_source(records, symbol=_SYMBOL, requested_as_of=_DATE)

    assert result["selected_source"] == "sina_historical"
    assert result["selected_field"] == "netamount"
    assert result["selected_value"] == "1"
    assert result["selected_direction"] == "inflow"
    assert result["direction_allowed"] is True
    assert result["legacy_reference"] is True
    assert result["legacy_web_algorithm"] is True
    assert result["selection_reason"] == "no_new_algorithm_source_legacy_fallback"
    assert result["fallback_rank"] == 4


def test_all_sources_invalid_are_blocked_with_failure_chain():
    result = select_fund_flow_source(
        [
            _record("tushare_eastmoney_moneyflow_dc", "r0_net", "not-a-number"),
            _record("tushare_ths_moneyflow_ths", "netamount", "2", date="2099-01-02"),
        ],
        symbol=_SYMBOL,
        requested_as_of=_DATE,
    )

    assert result["selected_source"] is None
    assert result["direction_allowed"] is False
    assert result["hard_guard"]["blocked"] is True
    assert result["failure_chain"]


def test_summary_uses_selected_source_and_does_not_conflict_with_side_evidence():
    records = [
        _record("tushare_eastmoney_moneyflow_dc", "r0_net", "1"),
        _record("tushare_ths_moneyflow_ths", "netamount", "20"),
    ]
    summary = summarize_evidence(
        records,
        selected_source="tushare_eastmoney_moneyflow_dc",
        selected_field="r0_net",
        requested_as_of=_DATE,
    )

    assert summary["status"] == "partial"
    assert summary["r0_net"] == "1"
    assert summary["netamount"] is None
