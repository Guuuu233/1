from decimal import Decimal

from tradingagents.dataflows.fund_flow_evidence import (
    build_sina_evidence,
    extract_model_totals,
    select_fund_flow_source,
    summarize_evidence,
    validate_model_summary,
)


_002167_ROWS = [
    {"opendate": "2026-08-13", "netamount": "-83709519.0900", "r0_net": "51607694.4100"},
    {"opendate": "2026-08-12", "netamount": "-26187171.1100", "r0_net": "-3954474.1400"},
    {"opendate": "2026-08-11", "netamount": "-78153483.6500", "r0_net": "20254086.5900"},
    {"opendate": "2026-08-10", "netamount": "116209487.9800", "r0_net": "89672105.0500"},
    {"opendate": "2026-08-07", "netamount": "-74060079.7400", "r0_net": "-192457.4800"},
]


def test_sina_evidence_preserves_semantics_sign_and_exact_yi_conversion():
    evidence = build_sina_evidence(
        _002167_ROWS,
        symbol="002167.SZ",
        requested_as_of="2026-08-13",
        retrieved_at="2026-08-13T12:00:00+00:00",
    )

    assert evidence[0]["unit"] == "亿元"
    assert evidence[0]["raw_unit"] == "元"
    assert evidence[0]["netamount"] == "-0.8370951909"
    assert evidence[0]["r0_net"] == "0.5160769441"
    assert evidence[0]["netamount_semantics"].startswith("总净额")
    assert evidence[0]["r0_net_semantics"].startswith("主力净额")
    assert evidence[0]["requested_as_of"] == "2026-08-13"
    assert evidence[0]["retrieved_at"] == "2026-08-13T12:00:00+00:00"


def test_002167_five_day_sum_does_not_round_or_shift_decimal():
    evidence = build_sina_evidence(
        _002167_ROWS,
        symbol="002167.SZ",
        requested_as_of="2026-08-13",
        retrieved_at=None,
    )
    summary = summarize_evidence(evidence)

    assert summary["status"] == "available"
    assert summary["record_count"] == 5
    assert summary["netamount"] == "-1.4590076561"
    assert summary["r0_net"] == "1.5738695443"
    assert Decimal(summary["netamount"]).quantize(Decimal("0.0001")) == Decimal("-1.4590")
    assert Decimal(summary["r0_net"]).quantize(Decimal("0.0001")) == Decimal("1.5739")


def test_model_summary_mismatch_is_explicit():
    evidence = build_sina_evidence(
        _002167_ROWS,
        symbol="002167.SZ",
        requested_as_of="2026-08-13",
        retrieved_at=None,
    )
    result = validate_model_summary(
        evidence,
        "5日累计：主力净流入 1.46 亿，总净流入额 -0.1459 亿。",
    )

    assert result["status"] == "mismatch"
    assert {item["field"] for item in result["mismatches"]} == {"r0_net", "netamount"}
    assert any(item["structured"] == "-1.4590076561" for item in result["mismatches"])


def test_model_total_parser_keeps_field_semantics_separate():
    assert extract_model_totals("主力净流入累计 1.5756 亿；总净流入额累计 -0.1459 亿") == {
        "r0_net": "1.5756",
        "netamount": "-0.1459",
    }


def _selection_record(
    source: str,
    value: str,
    *,
    field: str = "r0_net",
    date: str = "2026-08-14",
    group: str = "new_algorithm_group",
    unit: str = "亿元",
) -> dict:
    semantics = (
        "主力净额（负值表示净流出）"
        if field == "r0_net"
        else "总净额（负值表示净流出）"
    )
    return {
        "source": source,
        "algorithm_group": group,
        "status": "available",
        "symbol": "600519",
        "date": date,
        "period_kind": "historical_daily",
        "time_window": "1d",
        "field": field,
        "value": value,
        "unit": unit,
        "field_semantics": {field: semantics},
    }


def test_single_eastmoney_source_is_selected_without_consensus_gate():
    result = select_fund_flow_source(
        [_selection_record("eastmoney_direct", "1.25")],
        symbol="600519",
        requested_as_of="2026-08-14",
    )

    assert result["selected_source"] == "eastmoney_direct"
    assert result["selected_source_family"] == "eastmoney"
    assert result["selected_algorithm_group"] == "new_algorithm_group"
    assert result["selected_field"] == "r0_net"
    assert result["selected_value"] == "1.25"
    assert result["selected_direction"] == "inflow"
    assert result["selection_reason"] == "new_algorithm_source_priority"
    assert result["fallback_rank"] == 1
    assert result["legacy_reference"] is False
    assert result["direction_allowed"] is True


def test_invalid_eastmoney_then_ths_source_is_selected():
    result = select_fund_flow_source(
        [
            _selection_record("eastmoney_direct", "not-a-number"),
            _selection_record(
                "tushare_ths_moneyflow_ths",
                "-2.5",
                field="netamount",
            ),
        ],
        symbol="600519",
        requested_as_of="2026-08-14",
    )

    assert result["selected_source"] == "tushare_ths_moneyflow_ths"
    assert result["selected_field"] == "netamount"
    assert result["selected_direction"] == "outflow"
    assert result["direction_allowed"] is True
    assert any(item["source"] == "eastmoney_direct" for item in result["rejected_sources"])


def test_tushare_date_mismatch_is_skipped_before_lower_priority_source():
    result = select_fund_flow_source(
        [
            _selection_record(
                "tushare_eastmoney_moneyflow_dc",
                "1.0",
                date="2026-08-13",
            ),
            _selection_record(
                "tushare_ths_moneyflow_ths",
                "2.0",
                field="netamount",
            ),
        ],
        symbol="600519",
        requested_as_of="2026-08-14",
    )

    assert result["selected_source"] == "tushare_ths_moneyflow_ths"
    assert any(
        item["source"] == "tushare_eastmoney_moneyflow_dc"
        and item["reason"] == "date_mismatch"
        for item in result["rejected_sources"]
    )


def test_conflicting_ths_side_value_never_overrides_eastmoney_priority():
    result = select_fund_flow_source(
        [
            _selection_record("eastmoney_direct", "1.0"),
            _selection_record(
                "tushare_ths_moneyflow_ths",
                "-9.0",
                field="netamount",
            ),
        ],
        symbol="600519",
        requested_as_of="2026-08-14",
    )

    assert result["selected_source"] == "eastmoney_direct"
    assert result["selected_value"] == "1"
    assert result["selected_direction"] == "inflow"
    assert result["alternative_sources"][0]["source"] == "tushare_ths_moneyflow_ths"
    assert result["alternative_sources"][0]["value"] == "-9"


def test_legacy_is_allowed_only_as_explicit_fallback():
    result = select_fund_flow_source(
        [_selection_record("sina_historical", "-2.0", group="legacy_web_algorithm")],
        symbol="600519",
        requested_as_of="2026-08-14",
    )

    assert result["selected_source"] == "sina_historical"
    assert result["selected_algorithm_group"] == "legacy_web_algorithm"
    assert result["legacy_reference"] is True
    assert result["legacy_web_algorithm"] is True
    assert result["selection_reason"] == "no_new_algorithm_source_legacy_fallback"
    assert result["direction_allowed"] is True
    assert "legacy" in result["legacy_warning"]


def test_all_invalid_sources_are_blocked_with_rejection_chain():
    result = select_fund_flow_source(
        [
            _selection_record("eastmoney_direct", "1.0", date="2026-08-15"),
            _selection_record("tushare_ths_moneyflow_ths", "2.0", field="netamount", unit="unknown"),
        ],
        symbol="600519",
        requested_as_of="2026-08-14",
    )

    assert result["selected_source"] is None
    assert result["selection_reason"] == "all_sources_unavailable"
    assert result["direction_allowed"] is False
    assert result["hard_guard"]["blocked"] is True
    assert {item["reason"] for item in result["rejected_sources"]} >= {
        "future_date",
        "field_semantics_or_value_invalid",
    }


def test_selected_field_validation_does_not_require_missing_complementary_field():
    record = _selection_record("eastmoney_direct", "1.25")
    validation = validate_model_summary(
        [record],
        model_text=None,
        selected_field="r0_net",
    )

    assert validation["status"] == "not_checked"
    assert validation["hard_guard"]["blocked"] is False
    assert validation["structured"]["selected_field"] == "r0_net"
    assert validation["structured"]["r0_net"] == "1.25"
    assert validation["structured"]["netamount"] is None


def test_validation_uses_selected_source_only_when_side_evidence_is_present():
    dc = _selection_record("tushare_eastmoney_moneyflow_dc", "1.25")
    ths = _selection_record(
        "tushare_ths_moneyflow_ths",
        "-3.0",
        field="netamount",
    )
    validation = validate_model_summary(
        [dc, ths],
        model_text=None,
        selected_field="r0_net",
        selected_source="tushare_eastmoney_moneyflow_dc",
        requested_as_of="2026-08-14",
    )

    assert validation["status"] == "not_checked"
    assert validation["structured"]["status"] == "partial"
    assert validation["structured"]["r0_net"] == "1.25"


def test_missing_measurement_date_is_not_fabricated_from_as_of():
    record = _selection_record("eastmoney_direct", "1.0")
    record.pop("date")
    record["as_of"] = "2026-08-14"
    result = select_fund_flow_source(
        [record], symbol="600519", requested_as_of="2026-08-14"
    )

    assert result["direction_allowed"] is False
    assert result["rejected_sources"][0]["reason"] == "date_missing_or_invalid"


def test_extreme_finite_decimal_is_rejected_and_does_not_abort_selection():
    result = select_fund_flow_source(
        [
            _selection_record("eastmoney_direct", "1e999999999"),
            _selection_record("tushare_ths_moneyflow_ths", "1.0", field="netamount"),
        ],
        symbol="600519",
        requested_as_of="2026-08-14",
    )

    assert result["selected_source"] == "tushare_ths_moneyflow_ths"
    assert any(item["source"] == "eastmoney_direct" for item in result["rejected_sources"])


def test_component_only_field_cannot_authorize_direction():
    record = _selection_record("eastmoney_direct", "1.0")
    record["field"] = "large_net"
    record["value"] = "1.0"
    record["field_semantics"] = {"large_net": "大单净额（主力组成项）"}
    result = select_fund_flow_source(
        [record], symbol="600519", requested_as_of="2026-08-14"
    )

    assert result["direction_allowed"] is False


def test_model_parser_does_not_duplicate_main_force_net_as_total_net():
    assert extract_model_totals("主力资金净额累计 1.25 亿") == {"r0_net": "1.25"}


def test_five_eastmoney_daily_records_selects_target_day_1d_value_and_direction():
    records = [
        _selection_record("eastmoney_direct", "-1.0", date="2026-08-16"),
        _selection_record("eastmoney_direct", "-1.0", date="2026-08-17"),
        _selection_record("eastmoney_direct", "-0.5", date="2026-08-18"),
        _selection_record("eastmoney_direct", "-0.4280142", date="2026-08-19"),
        _selection_record("eastmoney_direct", "2.21197136", date="2026-08-20"),
    ]
    result = select_fund_flow_source(
        records,
        symbol="600519",
        requested_as_of="2026-08-20",
    )

    assert result["selected_source"] == "eastmoney_direct"
    assert result["selected_field"] == "r0_net"
    assert result["selected_value"] == "2.21197136"
    assert result["selected_direction"] == "inflow"
    assert result["selected_time_window"] == "1d"
    assert result["selected_window_days"] == 1
    assert result["selected_as_of"] == "2026-08-20"
    assert result["direction_allowed"] is True
    assert result["hard_guard"]["blocked"] is False


def test_independent_five_day_summary_is_separate_and_does_not_pollute_1d_selection():
    records = [
        _selection_record("eastmoney_direct", "-1.0", date="2026-08-16"),
        _selection_record("eastmoney_direct", "-1.0", date="2026-08-17"),
        _selection_record("eastmoney_direct", "-0.5", date="2026-08-18"),
        _selection_record("eastmoney_direct", "-0.4280142", date="2026-08-19"),
        _selection_record("eastmoney_direct", "2.21197136", date="2026-08-20"),
    ]
    result = select_fund_flow_source(
        records,
        symbol="600519",
        requested_as_of="2026-08-20",
    )

    # 1d selection is pure 1d
    assert result["selected_value"] == "2.21197136"
    assert result["selected_time_window"] == "1d"
    assert result["selected_window_days"] == 1

    # 5d summary if present must be labeled 5d and have 5d cumulative value
    if "five_day_summary" in result:
        five_d = result["five_day_summary"]
        assert five_d["time_window"] == "5d"
        assert five_d["window_days"] == 5
        assert Decimal(five_d["value"]) == Decimal("-0.71604284")
        assert five_d["direction"] == "outflow"


def test_single_tushare_dc_source_allows_direction_despite_insufficient_sources_consensus_audit():
    records = [
        _selection_record(
            "tushare_eastmoney_moneyflow_dc",
            "2.21197136",
            date="2026-08-20",
        )
    ]
    result = select_fund_flow_source(
        records,
        symbol="600519",
        requested_as_of="2026-08-20",
    )

    assert result["selected_source"] == "tushare_eastmoney_moneyflow_dc"
    assert result["selected_field"] == "r0_net"
    assert result["selected_value"] == "2.21197136"
    assert result["selected_direction"] == "inflow"
    assert result["direction_allowed"] is True
    assert result["hard_guard"]["blocked"] is False


def test_dc_r0_net_and_ths_netamount_opposite_directions_selects_dc_without_cross_field_conflict():
    records = [
        _selection_record(
            "tushare_eastmoney_moneyflow_dc",
            "2.21197136",
            date="2026-08-20",
            field="r0_net",
        ),
        _selection_record(
            "tushare_ths_moneyflow_ths",
            "-1.928246",
            date="2026-08-20",
            field="netamount",
        ),
    ]
    result = select_fund_flow_source(
        records,
        symbol="600519",
        requested_as_of="2026-08-20",
    )

    assert result["selected_source"] == "tushare_eastmoney_moneyflow_dc"
    assert result["selected_field"] == "r0_net"
    assert result["selected_value"] == "2.21197136"
    assert result["selected_direction"] == "inflow"
    assert result["direction_allowed"] is True
    assert len(result["alternative_sources"]) == 1
    assert result["alternative_sources"][0]["source"] == "tushare_ths_moneyflow_ths"
    assert result["alternative_sources"][0]["field"] == "netamount"
    assert result["alternative_sources"][0]["value"] == "-1.928246"
    assert result["alternative_sources"][0]["direction"] == "outflow"


def test_ths_netamount_only_allows_total_fund_flow_direction_summary():
    records = [
        _selection_record(
            "tushare_ths_moneyflow_ths",
            "2.0",
            date="2026-08-20",
            field="netamount",
        )
    ]
    result = select_fund_flow_source(
        records,
        symbol="600519",
        requested_as_of="2026-08-20",
    )

    assert result["selected_source"] == "tushare_ths_moneyflow_ths"
    assert result["selected_field"] == "netamount"
    assert result["selected_value"] == "2"
    assert result["selected_direction"] == "inflow"
    assert result["direction_summary"] == "总资金（非主力口径）偏流入"


def test_model_text_mistaking_five_day_sum_as_daily_value_is_blocked_by_validation():
    records = [
        _selection_record("eastmoney_direct", "-1.0", date="2026-08-16"),
        _selection_record("eastmoney_direct", "-1.0", date="2026-08-17"),
        _selection_record("eastmoney_direct", "-0.5", date="2026-08-18"),
        _selection_record("eastmoney_direct", "-0.4280142", date="2026-08-19"),
        _selection_record("eastmoney_direct", "2.21197136", date="2026-08-20"),
    ]
    # Model text incorrectly quotes the 5-day sum (-0.716 亿) as the daily value
    validation = validate_model_summary(
        records,
        "当日主力资金净额 -0.716 亿",
        selected_field="r0_net",
        selected_source="eastmoney_direct",
        requested_as_of="2026-08-20",
        window_days=1,
    )

    assert validation["status"] == "mismatch"
    assert validation["hard_guard"]["blocked"] is True

