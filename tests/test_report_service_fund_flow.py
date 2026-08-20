import pytest

from api.services.report_service import canonicalize_report_result_data


def _record(date: str, value: str) -> dict:
    return {
        "source": "sina_historical",
        "algorithm_group": "legacy_web_algorithm",
        "status": "available",
        "symbol": "600036.SH",
        "date": date,
        "as_of": date,
        "period_kind": "historical_daily",
        "time_window": "1d",
        "field": "r0_net",
        "value": value,
        "unit": "亿元",
        "field_semantics": {"r0_net": "主力净额（负值表示净流出）"},
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
