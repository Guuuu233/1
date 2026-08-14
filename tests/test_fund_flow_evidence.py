from decimal import Decimal

from tradingagents.dataflows import fund_flow_evidence as fund_flow_evidence_module
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


def _patch_002167_trade_calendar(monkeypatch):
    monkeypatch.setattr(
        fund_flow_evidence_module,
        "trading_days_back",
        lambda _date_str, count: [
            "2026-08-13",
            "2026-08-12",
            "2026-08-11",
            "2026-08-10",
            "2026-08-07",
        ][:count],
    )


def test_002167_five_day_sum_does_not_round_or_shift_decimal(monkeypatch):
    _patch_002167_trade_calendar(monkeypatch)
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


def test_duplicate_date_anywhere_in_full_evidence_is_conflict():
    rows = [
        *_002167_ROWS,
        {"opendate": "2026-08-06", "netamount": "1", "r0_net": "2"},
        {"opendate": "2026-08-06", "netamount": "1", "r0_net": "2"},
    ]
    summary = summarize_evidence(
        build_sina_evidence(
            rows,
            symbol="002167.SZ",
            requested_as_of="2026-08-13",
            retrieved_at=None,
        )
    )

    assert summary["status"] == "data_conflict"
    assert "重复日期" in summary["reason"]
    assert summary["required_window_days"] == 5
    assert summary["dates"].count("2026-08-06") == 2
    assert summary["netamount"] is None
    assert summary["r0_net"] is None


def test_missing_or_non_trading_day_breaks_window(monkeypatch):
    _patch_002167_trade_calendar(monkeypatch)
    rows = [dict(row) for row in _002167_ROWS]
    rows[-1]["opendate"] = "2026-08-06"
    summary = summarize_evidence(
        build_sina_evidence(
            rows,
            symbol="002167.SZ",
            requested_as_of="2026-08-13",
            retrieved_at=None,
        )
    )

    assert summary["status"] == "data_conflict"
    assert "断档" in summary["reason"]
    assert summary["dates"] == [
        "2026-08-06",
        "2026-08-10",
        "2026-08-11",
        "2026-08-12",
        "2026-08-13",
    ]
    assert summary["required_window_days"] == 5
    assert summary["netamount"] is None
    assert summary["r0_net"] is None


def test_unavailable_trade_calendar_is_structured_conflict(monkeypatch):
    def unavailable(_date_str, _count):
        raise RuntimeError("calendar offline")

    monkeypatch.setattr(fund_flow_evidence_module, "trading_days_back", unavailable)
    summary = summarize_evidence(
        build_sina_evidence(
            _002167_ROWS,
            symbol="002167.SZ",
            requested_as_of="2026-08-13",
            retrieved_at=None,
        )
    )

    assert summary["status"] == "data_conflict"
    assert "交易日历核验失败" in summary["reason"]
    assert summary["dates"] == [
        "2026-08-07",
        "2026-08-10",
        "2026-08-11",
        "2026-08-12",
        "2026-08-13",
    ]
    assert summary["required_window_days"] == 5
    assert summary["netamount"] is None
    assert summary["r0_net"] is None


def test_short_window_is_structured_partial():
    summary = summarize_evidence(
        build_sina_evidence(
            _002167_ROWS[:4],
            symbol="002167.SZ",
            requested_as_of="2026-08-13",
            retrieved_at=None,
        )
    )

    assert summary["status"] == "partial"
    assert "记录不足" in summary["reason"]
    assert summary["dates"] == [
        "2026-08-10",
        "2026-08-11",
        "2026-08-12",
        "2026-08-13",
    ]
    assert summary["required_window_days"] == 5


def test_period_window_and_raw_unit_mismatch_are_conflicts():
    cases = (
        {"period_kind": "five_day_cumulative", "window": "5d", "time_window": "5d"},
        {"time_window": "5d"},
        {"raw_unit": "亿元"},
    )
    for updates in cases:
        evidence = build_sina_evidence(
            _002167_ROWS,
            symbol="002167.SZ",
            requested_as_of="2026-08-13",
            retrieved_at=None,
        )
        evidence[0].update(updates)
        summary = summarize_evidence(evidence)

        assert summary["status"] == "data_conflict"
        assert summary["required_window_days"] == 5
        assert summary["dates"]
        assert summary["reason"]
        assert summary["netamount"] is None
        assert summary["r0_net"] is None


def test_model_summary_mismatch_is_explicit(monkeypatch):
    _patch_002167_trade_calendar(monkeypatch)
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
