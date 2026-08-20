import pytest

from api.services.report_service import canonicalize_report_result_data


def _record(date: str, value: str, field: str = "r0_net") -> dict:
    return {
        "source": "sina_historical",
        "algorithm_group": "legacy_web_algorithm",
        "status": "available",
        "symbol": "600036.SH",
        "date": date,
        "as_of": date,
        "period_kind": "historical_daily",
        "time_window": "1d",
        "field": field,
        "value": value,
        "unit": "亿元",
        "field_semantics": {field: "主力净额（负值表示净流出）"},
    }


def test_single_day_selection_ignores_other_historical_rows_in_persistence_validation():
    evidence = {
        "unit": "亿元",
        "records": [
            _record("2026-08-18", "1.0"),
            _record("2026-08-19", "-2.0"),
            _record("2026-08-20", "-3.9614887917"),
        ],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.9614887917",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 1,
            "selected_time_window": "1d",
        },
    }
    result = {
        "market_data_context": {"fund_flow_evidence": evidence},
    }

    canonicalize_report_result_data(result)


def test_single_day_selection_without_target_date_is_rejected():
    evidence = {
        "unit": "亿元",
        "records": [_record("2026-08-19", "-3.9614887917")],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.9614887917",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 1,
            "selected_time_window": "1d",
        },
    }

    with pytest.raises(ValueError, match="window not present"):
        canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


def test_multi_day_selection_still_sums_its_declared_window():
    values = ["1.0", "-2.0", "0.5", "-0.25", "-3.9614887917", "9.0"]
    dates = ["2026-08-15", "2026-08-16", "2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20"]
    evidence = {
        "unit": "亿元",
        "records": [_record(date, value) for date, value in zip(dates, values)],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-4.7114887917",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-19",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 5,
            "selected_time_window": "5d",
        },
    }

    canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


# ─── TDD 1: selected_window_days validation ──────────────────────────────────

@pytest.mark.parametrize("invalid_window", [0, "0", -1, "-1", -5, "abc", "1.5", 1.5, True, False, [], {}])
def test_invalid_selected_window_days_is_rejected(invalid_window):
    evidence = {
        "unit": "亿元",
        "records": [_record("2026-08-20", "-3.9614887917")],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.9614887917",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": invalid_window,
            "selected_time_window": "1d",
        },
    }

    with pytest.raises(ValueError, match="selected fund-flow window is invalid"):
        canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


def test_none_selected_window_days_defaults_to_1_and_passes():
    evidence = {
        "unit": "亿元",
        "records": [
            _record("2026-08-19", "-2.0"),
            _record("2026-08-20", "-3.9614887917"),
        ],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.9614887917",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": None,
        },
    }

    canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


def test_omitted_selected_window_days_defaults_to_1_and_passes():
    evidence = {
        "unit": "亿元",
        "records": [
            _record("2026-08-20", "-3.9614887917"),
        ],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.9614887917",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
        },
    }

    canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


# ─── TDD 2: Multi-day window distinct dates count ────────────────────────────

def test_multi_day_window_insufficient_distinct_dates_is_rejected():
    # Declared 5 days, but only 3 distinct dates exist up to selected_as_of
    values = ["1.0", "-2.0", "-3.9614887917"]
    dates = ["2026-08-18", "2026-08-19", "2026-08-20"]
    evidence = {
        "unit": "亿元",
        "records": [_record(date, value) for date, value in zip(dates, values)],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-4.9614887917",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 5,
            "selected_time_window": "5d",
        },
    }

    with pytest.raises(ValueError, match="window not present"):
        canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


# ─── TDD 3: Duplicate records deduplication & conflict rejection ─────────────

def test_duplicate_records_same_date_identical_values_are_collapsed_without_duplicate_sum():
    # Two identical records on 2026-08-20. Total must be -3.9614887917, NOT -7.9229775834.
    evidence = {
        "unit": "亿元",
        "records": [
            _record("2026-08-20", "-3.9614887917"),
            _record("2026-08-20", "-3.9614887917"),
        ],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.9614887917",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 1,
            "selected_time_window": "1d",
        },
    }

    canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


def test_duplicate_records_same_date_conflicting_values_are_rejected():
    # Two conflicting records on 2026-08-20. Must be rejected fail-closed.
    evidence = {
        "unit": "亿元",
        "records": [
            _record("2026-08-20", "-3.9614887917"),
            _record("2026-08-20", "1.5"),
        ],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.9614887917",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 1,
            "selected_time_window": "1d",
        },
    }

    with pytest.raises(ValueError, match="conflicting"):
        canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


def test_multi_day_with_identical_duplicates_sums_each_date_once():
    # 5 dates, with some dates duplicated with identical values
    evidence = {
        "unit": "亿元",
        "records": [
            _record("2026-08-16", "1.0"),
            _record("2026-08-16", "1.0"),  # identical duplicate
            _record("2026-08-17", "-2.0"),
            _record("2026-08-18", "0.5"),
            _record("2026-08-18", "0.5"),  # identical duplicate
            _record("2026-08-19", "-0.25"),
            _record("2026-08-20", "-3.0"),
        ],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.75",  # 1.0 - 2.0 + 0.5 - 0.25 - 3.0 = -3.75
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 5,
            "selected_time_window": "5d",
        },
    }

    canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


def test_multi_day_with_conflicting_duplicates_is_rejected():
    evidence = {
        "unit": "亿元",
        "records": [
            _record("2026-08-16", "1.0"),
            _record("2026-08-17", "-2.0"),
            _record("2026-08-18", "0.5"),
            _record("2026-08-18", "99.0"),  # conflict on 2026-08-18
            _record("2026-08-19", "-0.25"),
            _record("2026-08-20", "-3.0"),
        ],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.75",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 5,
            "selected_time_window": "5d",
        },
    }

    with pytest.raises(ValueError, match="conflicting"):
        canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


# ─── TDD 4: Non-ISO / unparseable date rejection ─────────────────────────────

@pytest.mark.parametrize("invalid_date", ["2026/08/20", "2026-8-20", "20260820", "2026-02-31", "invalid_date", ""])
def test_non_iso_selected_as_of_date_is_rejected(invalid_date):
    evidence = {
        "unit": "亿元",
        "records": [_record("2026-08-20", "-3.9614887917")],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.9614887917",
            "selected_unit": "亿元",
            "selected_as_of": invalid_date,
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 1,
        },
    }

    with pytest.raises(ValueError):
        canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


@pytest.mark.parametrize("invalid_record_date", ["2026/08/20", "2026-8-20", "20260820", "2026-02-31", "invalid_date"])
def test_non_iso_record_date_is_rejected(invalid_record_date):
    evidence = {
        "unit": "亿元",
        "records": [_record(invalid_record_date, "-3.9614887917")],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.9614887917",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 1,
        },
    }

    with pytest.raises(ValueError):
        canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


def test_general_records_validation_rejects_non_iso_date_without_selection():
    evidence = {
        "unit": "亿元",
        "records": [_record("2026/08/20", "-3.9614887917")],
    }
    with pytest.raises(ValueError, match="record date is invalid"):
        canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


# ─── TDD 5: Window matching and future rows exclusion ────────────────────────

def test_future_rows_do_not_enter_window_and_1d_only_matches_selected_as_of():
    evidence = {
        "unit": "亿元",
        "records": [
            _record("2026-08-19", "-2.0"),
            _record("2026-08-20", "-3.9614887917"),
            _record("2026-08-21", "10.0"),  # future relative to selected_as_of
        ],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.9614887917",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 1,
            "selected_time_window": "1d",
        },
    }

    canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


def test_5d_window_excludes_future_rows_and_picks_latest_5_before_cutoff():
    values = ["1.0", "-2.0", "0.5", "-0.25", "-3.9614887917", "100.0", "200.0"]
    dates = ["2026-08-15", "2026-08-16", "2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21"]
    evidence = {
        "unit": "亿元",
        "records": [_record(date, value) for date, value in zip(dates, values)],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-4.7114887917",  # sum of 08-15 through 08-19
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-19",  # cutoff is 08-19; 08-20 and 08-21 are future rows
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 5,
            "selected_time_window": "5d",
        },
    }

    canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


# ─── TDD 6: Direction validation ─────────────────────────────────────────────

def test_direction_validated_against_actual_window_total():
    evidence = {
        "unit": "亿元",
        "records": [_record("2026-08-20", "-3.9614887917")],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "-3.9614887917",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "inflow",  # Incorrect direction (value is negative r0_net, should be outflow)
            "selected_window_days": 1,
            "selected_time_window": "1d",
        },
    }

    with pytest.raises(ValueError, match="direction does not match"):
        canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


def test_r0_out_direction_positive_is_outflow():
    evidence = {
        "unit": "亿元",
        "records": [_record("2026-08-20", "5.0", field="r0_out")],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_out",
            "selected_value": "5.0",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "outflow",
            "selected_window_days": 1,
            "selected_time_window": "1d",
        },
    }

    canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


def test_r0_out_direction_negative_is_inflow():
    evidence = {
        "unit": "亿元",
        "records": [_record("2026-08-20", "-5.0", field="r0_out")],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_out",
            "selected_value": "-5.0",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "inflow",
            "selected_window_days": 1,
            "selected_time_window": "1d",
        },
    }

    canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})


def test_zero_total_direction_is_neutral():
    evidence = {
        "unit": "亿元",
        "records": [_record("2026-08-20", "0.0")],
        "selection": {
            "status": "selected",
            "direction_allowed": True,
            "hard_guard": {"blocked": False},
            "selected_source": "sina_historical",
            "selected_field": "r0_net",
            "selected_value": "0.0",
            "selected_unit": "亿元",
            "selected_as_of": "2026-08-20",
            "selected_algorithm_group": "legacy_web_algorithm",
            "legacy_reference": True,
            "fallback_rank": 7,
            "selected_direction": "neutral",
            "selected_window_days": 1,
            "selected_time_window": "1d",
        },
    }

    canonicalize_report_result_data({"market_data_context": {"fund_flow_evidence": evidence}})
