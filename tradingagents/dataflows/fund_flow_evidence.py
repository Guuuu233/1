"""Structured evidence and arithmetic checks for individual fund-flow reports."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Iterable, Mapping


YI = Decimal("100000000")
DEFAULT_WINDOW_DAYS = 5


class FundFlowText(str):
    """Prompt-compatible text carrying structured fund-flow evidence."""

    def __new__(
        cls,
        value: str,
        *,
        evidence: Iterable[Mapping[str, Any]] = (),
        evidence_meta: Mapping[str, Any] | None = None,
    ):
        obj = super().__new__(cls, value)
        obj.fund_flow_evidence = [dict(item) for item in evidence]
        obj.fund_flow_evidence_meta = dict(evidence_meta or {})
        return obj


def decimal_value(value: Any) -> Decimal | None:
    """Parse a finite decimal from provider/model input."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


def _as_yi(value: Any) -> Decimal | None:
    parsed = decimal_value(value)
    return None if parsed is None else parsed / YI


def build_em_evidence(
    frame: Any,
    *,
    symbol: str,
    requested_as_of: str,
    retrieved_at: str | None,
    source: str = "eastmoney_individual_fund_flow",
) -> list[dict[str, Any]]:
    """Build evidence from Eastmoney's main-force-only daily series.

    Eastmoney's ``主力净流入-净额`` is r0_net semantics.  It is deliberately
    not copied into netamount, whose total-net semantics are unavailable here.
    """
    if frame is None or not hasattr(frame, "iterrows"):
        return []
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        date = str(row.get("日期", "")).strip()
        if not date:
            continue
        raw_r0 = decimal_value(row.get("主力净流入-净额"))
        if raw_r0 is None:
            continue
        records.append(
            {
                "date": date,
                "netamount": None,
                "r0_net": _decimal_text(raw_r0 / YI),
                "netamount_raw": None,
                "r0_net_raw": _decimal_text(raw_r0),
                "raw_unit": "元",
                "unit": "亿元",
                "source": source,
                "symbol": symbol,
                "requested_as_of": requested_as_of,
                "as_of": date,
                "retrieved_at": retrieved_at,
                "status": "available",
                "netamount_semantics": "总净额（当前来源未提供）",
                "r0_net_semantics": "主力净额（负值表示净流出）",
            }
        )
    return records


def build_sina_evidence(
    rows: Iterable[Mapping[str, Any]],
    *,
    symbol: str,
    requested_as_of: str,
    retrieved_at: str | None,
    source: str = "sina_historical",
) -> list[dict[str, Any]]:
    """Build one lossless, serializable record per Sina daily row.

    Sina's historical endpoint exposes these core amounts in yuan.  Keep the
    raw values alongside exact Decimal亿元 values so downstream arithmetic can
    never be reconstructed from rounded display text.
    """
    records: list[dict[str, Any]] = []
    for row in rows:
        date = str(row.get("opendate", "")).strip()
        if not date:
            continue
        netamount_raw = decimal_value(row.get("netamount"))
        r0_net_raw = decimal_value(row.get("r0_net"))
        if netamount_raw is None and r0_net_raw is None:
            continue
        records.append(
            {
                "date": date,
                "netamount": _decimal_text(netamount_raw / YI) if netamount_raw is not None else None,
                "r0_net": _decimal_text(r0_net_raw / YI) if r0_net_raw is not None else None,
                "netamount_raw": _decimal_text(netamount_raw),
                "r0_net_raw": _decimal_text(r0_net_raw),
                "raw_unit": "元",
                "unit": "亿元",
                "source": source,
                "symbol": symbol,
                "requested_as_of": requested_as_of,
                "as_of": date,
                "retrieved_at": retrieved_at,
                "status": "available",
                "netamount_semantics": "总净额（负值表示净流出）",
                "r0_net_semantics": "主力净额（负值表示净流出）",
            }
        )
    return records


def build_provider_text(
    text: str,
    *,
    symbol: str,
    requested_as_of: str,
    source: str,
    reason: str,
    retrieved_at: str | None = None,
) -> FundFlowText:
    """Keep a formatted provider report while exposing an explicit evidence gap."""
    return FundFlowText(
        text,
        evidence=[],
        evidence_meta=build_gap_meta(
            symbol=symbol,
            requested_as_of=requested_as_of,
            source=source,
            status="unavailable",
            reason=reason,
            retrieved_at=retrieved_at,
        ),
    )


def build_gap_meta(
    *,
    symbol: str,
    requested_as_of: str,
    source: str,
    status: str,
    reason: str,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    """Build explicit evidence-gap metadata without fabricating daily values."""
    return {
        "symbol": symbol,
        "requested_as_of": requested_as_of,
        "source": source,
        "unit": "亿元",
        "status": status,
        "reason": reason,
        "gap": f"【数据获取失败】资金流 evidence：{reason}",
        "retrieved_at": retrieved_at,
        "as_of": None,
    }


def _sum_field(records: Iterable[Mapping[str, Any]], field: str) -> Decimal | None:
    values = [decimal_value(record.get(field)) for record in records]
    usable = [value for value in values if value is not None]
    return sum(usable, Decimal("0")) if usable else None


def summarize_evidence(
    records: Iterable[Mapping[str, Any]],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> dict[str, Any]:
    """Compute exact five-row totals from structured亿元 values."""
    usable = [
        dict(record)
        for record in records
        if isinstance(record, Mapping) and record.get("status") == "available"
    ]
    selected = usable[-window_days:] if len(usable) > window_days else usable
    netamount = _sum_field(selected, "netamount")
    r0_net = _sum_field(selected, "r0_net")
    status = "available" if len(selected) == window_days and netamount is not None and r0_net is not None else "partial"
    return {
        "window_days": window_days,
        "record_count": len(selected),
        "status": status,
        "dates": [str(record.get("date")) for record in selected],
        "netamount": _decimal_text(netamount),
        "r0_net": _decimal_text(r0_net),
        "unit": "亿元",
        "semantics": {
            "netamount": "总净额（负值表示净流出）",
            "r0_net": "主力净额（负值表示净流出）",
        },
    }


_MODEL_TOTAL_PATTERNS = {
    "r0_net": (
        re.compile(r"主力(?:资金)?净(?:流入|额)[^\n。；;]{0,40}?(?:累计|合计|总计)[^\n。；;]{0,20}?([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*亿"),
        re.compile(r"(?:累计|合计|总计)[^\n。；;]{0,20}?主力(?:资金)?净(?:流入|额)[^\n。；;]{0,20}?([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*亿"),
    ),
    "netamount": (
        re.compile(r"(?<!主力)(?:总)?净(?:流入额|流入|额)[^\n。；;]{0,40}?(?:累计|合计|总计)[^\n。；;]{0,20}?([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*亿"),
        re.compile(r"(?:累计|合计|总计)[^\n。；;]{0,20}?(?<!主力)(?:总)?净(?:流入额|流入|额)[^\n。；;]{0,20}?([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*亿"),
    ),
}


def extract_model_totals(text: str | None) -> dict[str, str]:
    """Extract only explicitly labelled cumulative亿元 values from model text."""
    if not isinstance(text, str) or not text.strip():
        return {}
    found: dict[str, str] = {}
    for field, patterns in _MODEL_TOTAL_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                value = decimal_value(match.group(1))
                if value is not None:
                    found[field] = _decimal_text(value) or ""
                    break
    return found


def validate_model_summary(
    records: Iterable[Mapping[str, Any]],
    model_text: str | None,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    tolerance: Decimal = Decimal("0.01"),
) -> dict[str, Any]:
    """Mark model cumulative totals that disagree with exact structured totals."""
    structured = summarize_evidence(records, window_days=window_days)
    model = extract_model_totals(model_text)
    mismatches: list[dict[str, str]] = []
    for field, model_value_text in model.items():
        structured_value = decimal_value(structured.get(field))
        model_value = decimal_value(model_value_text)
        if structured_value is None or model_value is None:
            continue
        if abs(structured_value - model_value) > tolerance:
            mismatches.append(
                {
                    "field": field,
                    "structured": _decimal_text(structured_value) or "",
                    "model": _decimal_text(model_value) or "",
                    "unit": "亿元",
                    "reason": "model cumulative total differs from structured evidence",
                }
            )
    status = "mismatch" if mismatches else ("matched" if model else "not_checked")
    return {
        "status": status,
        "structured": structured,
        "model": model,
        "mismatches": mismatches,
        "tolerance": _decimal_text(tolerance),
    }
