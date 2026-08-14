from datetime import date
from decimal import Decimal

import pytest

from tradingagents.dataflows import trade_calendar
from tradingagents.dataflows.fund_flow_evidence import (
    build_sina_evidence,
    extract_model_totals,
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


@pytest.fixture(autouse=True)
def _stub_trade_calendar(monkeypatch):
    dates = [
        date(2026, 8, 3),
        date(2026, 8, 4),
        date(2026, 8, 5),
        date(2026, 8, 6),
        date(2026, 8, 7),
        date(2026, 8, 10),
        date(2026, 8, 11),
        date(2026, 8, 12),
        date(2026, 8, 13),
    ]
    monkeypatch.setattr(
        trade_calendar,
        "require_cn_trade_dates",
        lambda: (dates, set(dates)),
    )


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


def test_summarize_rejects_duplicate_dates_before_selecting_window():
    evidence = build_sina_evidence(
        _002167_ROWS,
        symbol="002167.SZ",
        requested_as_of="2026-08-13",
        retrieved_at=None,
    )
    summary = summarize_evidence([*evidence, dict(evidence[0])])

    assert summary["status"] == "data_conflict"
    assert summary["netamount"] is None
    assert summary["r0_net"] is None
    assert summary["required_window_days"] == 5
    assert summary["dates"].count("2026-08-13") == 2
    assert "重复日期" in summary["reason"]


def test_summarize_rejects_trading_day_gap_without_silent_sum():
    rows = [row for row in _002167_ROWS if row["opendate"] != "2026-08-10"]
    evidence = build_sina_evidence(
        rows,
        symbol="002167.SZ",
        requested_as_of="2026-08-13",
        retrieved_at=None,
    )
    summary = summarize_evidence(evidence)

    assert summary["status"] == "data_conflict"
    assert summary["netamount"] is None
    assert summary["r0_net"] is None
    assert summary["required_window_days"] == 5
    assert summary["dates"] == ["2026-08-07", "2026-08-11", "2026-08-12", "2026-08-13"]
    assert "断档" in summary["reason"]
    assert summary["missing_dates"] == ["2026-08-10"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("period_kind", "five_day_cumulative"),
        ("window", "5d"),
        ("raw_unit", "亿元"),
    ],
)
def test_summarize_rejects_daily_contract_mismatch(key, value):
    evidence = build_sina_evidence(
        _002167_ROWS,
        symbol="002167.SZ",
        requested_as_of="2026-08-13",
        retrieved_at=None,
    )
    mutated = [dict(record) for record in evidence]
    mutated[0][key] = value
    if key == "window":
        mutated[0]["time_window"] = value

    summary = summarize_evidence(mutated)

    assert summary["status"] == "data_conflict"
    assert summary["netamount"] is None
    assert summary["r0_net"] is None
    assert summary["required_window_days"] == 5
    assert summary["dates"]
    assert "禁止累计" in summary["reason"]


def test_summarize_marks_short_contiguous_window_partial_without_future_fill():
    evidence = build_sina_evidence(
        _002167_ROWS[:-1],
        symbol="002167.SZ",
        requested_as_of="2026-08-13",
        retrieved_at=None,
    )
    summary = summarize_evidence(evidence)

    assert summary["status"] == "partial"
    assert summary["data_conflict"] is False
    assert summary["record_count"] == 4
    assert summary["required_window_days"] == 5
    assert summary["reason"] == "记录少于完整累计窗口，未补齐缺失或未来交易日"
    assert summary["dates"] == ["2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13"]


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
