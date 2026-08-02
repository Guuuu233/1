"""Deterministic regression coverage for DAV-37 Stage 16 fixes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from api import main
from api.services import report_service
from tradingagents.agents.utils import debate_utils
from tradingagents.graph import data_collector


def test_structured_report_accepts_canonical_numerics_only():
    report = report_service.StructuredReport(
        decision="BUY",
        probability=0.7,
        confidence=75.0,
        target_price=12.5,
        stop_loss_price=10,
    )

    assert report.probability == pytest.approx(0.7)
    assert report.confidence == 75
    assert report.target_price == 12.5
    assert report.stop_loss_price == 10.0


def test_structured_report_accepts_zero_or_positive_prices():
    report = report_service.StructuredReport(
        decision="HOLD",
        target_price=0.0,
        stop_loss_price=1.25,
    )

    assert report.target_price == 0.0
    assert report.stop_loss_price == 1.25


@pytest.mark.parametrize(
    "value",
    [
        "12.5",
        True,
        False,
        [12.5, 11.0],
        -12.5,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_structured_report_rejects_strings_bools_lists_and_non_finite_prices(value):
    report = report_service.StructuredReport(
        decision="HOLD",
        target_price=value,
        stop_loss_price=value,
    )

    assert report.target_price is None
    assert report.stop_loss_price is None


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "0.5",
        [0.5],
        float("nan"),
        float("inf"),
        -0.01,
        1.01,
    ],
)
def test_api_strict_unit_interval_rejects_non_real_or_out_of_range(value):
    with pytest.raises(ValueError, match="finite number in \\[0, 1\\]"):
        main._strict_unit_interval(value, "probability")


def test_api_strict_unit_interval_returns_canonical_float():
    assert main._strict_unit_interval(None, "probability") is None
    assert main._strict_unit_interval(0, "probability") == 0.0
    assert main._strict_unit_interval(1, "probability") == 1.0
    assert main._strict_report_probability(0.5) == 0.5


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "75",
        75.9,
        -1,
        101,
        float("nan"),
        float("inf"),
    ],
)
def test_api_strict_report_confidence_rejects_non_integer_or_out_of_range(value):
    with pytest.raises(ValueError, match="confidence must be"):
        main._strict_report_confidence(value)


def test_api_strict_report_confidence_returns_integer():
    assert main._strict_report_confidence(None) is None
    assert main._strict_report_confidence(75.0) == 75


def test_apply_structured_fields_merges_and_preserves_graph_semantics():
    structured = SimpleNamespace(
        decision=None,
        probability=None,
        data_gaps=[],
        falsification_conditions=["structured condition"],
        not_applicable=False,
    )
    result = {
        "data_gaps": ["graph gap"],
        "falsification_conditions": ["graph condition"],
        "not_applicable": True,
    }
    resolved = {
        "confidence": 80,
        "target_price": 1.2,
        "stop_loss_price": 1.0,
    }

    decision = main._apply_structured_report_fields(
        result,
        structured=structured,
        graph_decision="BUY",
        resolved=resolved,
    )

    assert decision == "BUY"
    assert result["falsification_conditions"] == ["graph condition", "structured condition"]
    assert result["data_gaps"] == ["graph gap"]
    assert result["not_applicable"] is True


def test_apply_structured_fields_keeps_existing_semantics_when_structured_is_none():
    result = {
        "falsification_conditions": ["graph condition"],
        "not_applicable": True,
    }

    decision = main._apply_structured_report_fields(
        result,
        structured=None,
        graph_decision="HOLD",
        resolved={},
    )

    assert decision == "HOLD"
    assert result["falsification_conditions"] == ["graph condition"]
    assert result["not_applicable"] is True


def test_structured_not_applicable_still_overrides_existing_false():
    structured = SimpleNamespace(
        decision="BUY",
        probability=0.6,
        data_gaps=[],
        falsification_conditions=[],
        not_applicable=True,
    )
    result = {"not_applicable": False}

    main._apply_structured_report_fields(
        result,
        structured=structured,
        graph_decision="HOLD",
        resolved={},
    )

    assert result["not_applicable"] is True


def test_data_collector_classifies_new_chinese_failure_markers():
    raw = {
        "news": "【数据获取失败】新闻接口异常",
        "global_news": "新浪财经快讯获取失败：HTTPError",
        "fund_flow_board": "板块资金流向数据暂时不可用（akshare 接口问题）",
        "fundamentals": "接口请求失败",
        "zt_pool": "返回格式异常",
        "shareholder_count": "数据暂不可用",
        "balance_sheet": "本项不可用",
        "hot_stocks": "正常无重大新闻",
    }

    ledger = data_collector._build_data_failure_ledger(raw)
    sources = {entry["source"] for entry in ledger}

    assert sources == {
        "news",
        "global_news",
        "fund_flow_board",
        "fundamentals",
        "zt_pool",
        "shareholder_count",
        "balance_sheet",
    }
    assert all(entry["status"] == "failed" for entry in ledger)


def test_data_collector_failure_ledger_treats_none_empty_and_not_applicable_as_non_failure():
    raw = {
        "news": None,
        "global_news": "",
        "zt_pool": "   ",
        "fundamentals": "本周期无评估事件，不适用。",
        "realtime": {"status": "not_applicable", "error": "no event"},
        "hot_stocks": {"status": "available", "data": []},
    }

    assert data_collector._build_data_failure_ledger(raw) == []


def test_merge_data_gaps_consumes_collector_ledger_for_new_markers():
    raw = {
        "news": "接口请求失败",
        "global_news": "新浪财经快讯获取失败：HTTPError",
        "zt_pool": "返回格式异常",
    }
    ledger = data_collector._build_data_failure_ledger(raw)
    result_data = {
        "market_data_context": {"data_failure_ledger": ledger},
    }

    gaps = report_service.merge_data_gaps(result_data)

    assert gaps == [
        "【数据获取失败】news：provider call failed",
        "【数据获取失败】global_news：provider call failed",
        "【数据获取失败】zt_pool：provider call failed",
    ]


def test_aggregate_horizon_metadata_all_not_applicable_is_true_when_all_completed():
    result = report_service.aggregate_horizon_metadata(
        [
            ("short", {"status": "completed", "not_applicable": True, "falsification_conditions": []}),
            ("medium", {"status": "completed", "not_applicable": True, "falsification_conditions": []}),
        ],
        requested_horizons=["short", "medium"],
    )

    assert result["not_applicable"] is True
    assert result["not_applicable_by_horizon"] == {"short": True, "medium": True}


def test_aggregate_horizon_metadata_not_applicable_requires_completed_horizons():
    result = report_service.aggregate_horizon_metadata(
        [
            ("short", {"status": "completed", "not_applicable": True, "falsification_conditions": []}),
            ("medium", {"status": "not_requested", "not_applicable": True, "falsification_conditions": []}),
        ],
        requested_horizons=["short", "medium"],
    )

    assert result["not_applicable"] is False
    assert result["not_applicable_by_horizon"] == {"short": True, "medium": None}


def test_extract_and_strip_strict_machine_block():
    payload = {"verdict": "pass", "hard_constraints": [], "soft_constraints": []}
    text = f"固定正文\n<!-- RISK_JUDGE: {json.dumps(payload, ensure_ascii=False)} -->"

    assert debate_utils.extract_tagged_json(text, "RISK_JUDGE") == payload
    assert debate_utils.strip_tagged_json(text, "RISK_JUDGE") == "固定正文"


def test_machine_block_close_inside_json_string_is_ignored():
    payload = {"verdict": "pass", "reason": "contains --> inside"}
    text = f"<!-- RISK_JUDGE: {json.dumps(payload, ensure_ascii=False)} -->"

    parsed = debate_utils.extract_tagged_json(text, "RISK_JUDGE")

    assert parsed.get("reason") == "contains --> inside"
    assert "RISK_JUDGE" not in debate_utils.strip_tagged_json(text, "RISK_JUDGE")


def test_malformed_machine_block_is_rejected_and_not_stripped():
    text = "固定正文\n<!-- RISK_JUDGE: {\"verdict\": [} -->"

    assert debate_utils.extract_tagged_json(text, "RISK_JUDGE") == {}
    assert debate_utils.strip_tagged_json(text, "RISK_JUDGE") == text.strip()


def test_trailing_prose_after_machine_block_is_rejected_and_not_stripped():
    payload = {"verdict": "pass", "hard_constraints": []}
    block = f"<!-- RISK_JUDGE: {json.dumps(payload, ensure_ascii=False)} -->"
    text = f"{block}\n后续正文"

    assert debate_utils.extract_tagged_json(text, "RISK_JUDGE") == {}
    assert debate_utils.strip_tagged_json(text, "RISK_JUDGE") == text.strip()


def test_machine_block_before_coherent_other_block_remains_allowed():
    payload = {"verdict": "pass", "hard_constraints": []}
    block = f"<!-- RISK_JUDGE: {json.dumps(payload, ensure_ascii=False)} -->"
    text = f"{block}\n<!-- OTHER: {{\"ok\": true}} -->"

    assert debate_utils.extract_tagged_json(text, "RISK_JUDGE") == payload

    stripped = debate_utils.strip_tagged_json(text, "RISK_JUDGE")
    assert "RISK_JUDGE" not in stripped
    assert "OTHER" in stripped
