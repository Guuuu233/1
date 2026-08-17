"""Structured fund-flow evidence, source alignment, and arithmetic checks."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Iterable, Mapping


YI = Decimal("100000000")
DEFAULT_WINDOW_DAYS = 5
NEW_ALGORITHM_GROUP = "new_algorithm_group"
LEGACY_WEB_ALGORITHM = "legacy_web_algorithm"
UNKNOWN_ALGORITHM_GROUP = "unknown_algorithm_group"
DEFAULT_RELATIVE_DISPERSION = Decimal("0.20")
_EPSILON = Decimal("0.0000000001")

_MAIN_FORCE_FIELDS = ("r0_in", "r0_out", "r0", "r0_net")
_FIELD_ORDER = ("r0_net", "r0", "r0_in", "r0_out", "netamount")

# The selection contract is intentionally source-first, not consensus-first.
# ``fallback_rank`` is a stable category rank retained in every selected and
# side-evidence record; sources in the same category use the fixed order below.
_SOURCE_PRIORITY = {
    "tushare_eastmoney_moneyflow_dc": 1,
    "tushare_dc": 1,
    "tushare_moneyflow_dc": 1,
    "moneyflow_dc": 1,
    "tushare_ths_moneyflow_ths": 2,
    "tushare_ths": 2,
    "tushare_moneyflow_ths": 2,
    "moneyflow_ths": 2,
    "eastmoney_direct": 3,
    "eastmoney_individual_fund_flow": 3,
    "ths_instant_snapshot": 3,
    "sina_app_manual_calibration": 3,
    "sina_historical": 4,
}
_SOURCE_PRIORITY_ORDER = (
    "tushare_eastmoney_moneyflow_dc",
    "tushare_ths_moneyflow_ths",
    "eastmoney_direct",
    "eastmoney_individual_fund_flow",
    "ths_instant_snapshot",
    "sina_app_manual_calibration",
    "sina_historical",
)

_FIELD_ALIASES = {
    "主力净流入-净额": "r0_net",
    "主力净流入净额": "r0_net",
    "主力净流入": "r0_net",
    "主力净额": "r0_net",
    "主力净额(元)": "r0_net",
    "主力流入": "r0_in",
    "主力流入额": "r0_in",
    "主力资金流入": "r0_in",
    "主力流出": "r0_out",
    "主力流出额": "r0_out",
    "主力资金流出": "r0_out",
    "净额": "netamount",
    "净流入额": "netamount",
    "总净额": "netamount",
    "总净流入": "netamount",
}
_COMPONENT_ALIASES = {
    "特大单净流入": "super_large_net",
    "特大单": "super_large_net",
    "超大单净流入": "super_large_net",
    "超大单": "super_large_net",
    "大单净流入": "large_net",
    "大单": "large_net",
}
_FIELD_SEMANTICS = {
    "r0_in": "主力流入（官方主力口径）",
    "r0_out": "主力流出（官方主力口径）",
    "r0": "主力资金值（官方主力口径）",
    "r0_net": "主力净额（负值表示净流出）",
    "netamount": "总净额（负值表示净流出）",
}
_COMPONENT_SEMANTICS = {
    "super_large_net": "特大单净额（主力组成项，负值表示净流出）",
    "large_net": "大单净额（主力组成项，负值表示净流出）",
}
_FIELD_CATEGORIES = {
    **{field: "main_force" for field in _MAIN_FORCE_FIELDS},
    "netamount": "total",
}
_AMOUNT_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"(万亿|亿元|万元|亿|万|元)?\s*$"
)


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
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _as_yi(value: Any) -> Decimal | None:
    parsed = decimal_value(value)
    return None if parsed is None else parsed / YI


def _normalise_source_text(source: Any) -> str:
    return str(source or "").strip().lower().replace(" ", "_")


def infer_algorithm_group(
    source: Any,
    algorithm_group: Any = None,
) -> str:
    """Classify source identity first; metadata cannot relabel legacy/new feeds."""
    label = _normalise_source_text(source)
    source_group = UNKNOWN_ALGORITHM_GROUP
    if any(
        token in label
        for token in (
            "sina_app",
            "sina_mobile",
            "sinaapp",
            "新浪财经_app",
            "新浪财经app",
            "eastmoney",
            "em_",
            "_em",
            "tushare_dc",
            "moneyflow_dc",
            "dongfangcaifu",
            "东方财富",
            "ths",
            "tushare_ths",
            "moneyflow_ths",
            "tonghuashun",
            "同花顺",
        )
    ):
        source_group = NEW_ALGORITHM_GROUP
    elif any(token in label for token in ("legacy", "web", "sina_historical")):
        source_group = LEGACY_WEB_ALGORITHM
    elif "historical" in label and "sina" in label:
        source_group = LEGACY_WEB_ALGORITHM
    elif label in {"sina", "sina_web", "sinafinance", "新浪", "新浪财经"}:
        source_group = LEGACY_WEB_ALGORITHM
    if source_group != UNKNOWN_ALGORITHM_GROUP:
        return source_group
    explicit = str(algorithm_group or "").strip()
    if explicit in {NEW_ALGORITHM_GROUP, "new_algorithm", "new"}:
        return NEW_ALGORITHM_GROUP
    if explicit in {LEGACY_WEB_ALGORITHM, "legacy_web", "legacy"}:
        return LEGACY_WEB_ALGORITHM
    return UNKNOWN_ALGORITHM_GROUP


def source_family(source: Any) -> str:
    """Return a stable family label useful in redacted evidence."""
    label = _normalise_source_text(source)
    if "sina" in label or "新浪" in label:
        return "sina_app" if any(token in label for token in ("app", "mobile")) else "sina_web"
    if any(token in label for token in ("eastmoney", "em_", "_em", "tushare_dc", "moneyflow_dc", "东方财富", "dongfangcaifu")):
        return "eastmoney"
    if any(token in label for token in ("ths", "tushare_ths", "moneyflow_ths", "tonghuashun", "同花顺")):
        return "ths"
    return label or "unknown_source"


def _unit_name(unit: Any) -> str:
    text = str(unit or "").strip().lower()
    aliases = {
        "rmb": "元",
        "cny": "元",
        "yuan": "元",
        "元": "元",
        "万": "万",
        "万元": "万元",
        "亿": "亿",
        "亿元": "亿元",
        "万亿": "万亿",
    }
    return aliases.get(text, str(unit or "").strip())


def _amount_to_yi(value: Any, unit: Any = "元") -> tuple[Decimal | None, str]:
    """Parse a number and convert it to exact 亿元 while retaining source unit."""
    text = str(value).strip() if isinstance(value, str) else ""
    match = _AMOUNT_RE.fullmatch(text) if text else None
    if match:
        number = decimal_value(match.group(1))
        source_unit = match.group(2) or _unit_name(unit)
    else:
        number = decimal_value(value)
        source_unit = _unit_name(unit)
    if number is None:
        return None, source_unit
    if not source_unit:
        return None, source_unit
    multiplier = {
        "元": Decimal("1") / YI,
        "万": Decimal("10000") / YI,
        "万元": Decimal("10000") / YI,
        "亿": Decimal("1"),
        "亿元": Decimal("1"),
        "万亿": Decimal("10000"),
    }.get(source_unit)
    if multiplier is None:
        # Unknown units cannot safely be compared or summed.
        return None, source_unit
    return number * multiplier, source_unit


def _row_value(row: Mapping[str, Any], field: str) -> tuple[Any, str | None]:
    """Find a canonical field in a provider row and return value plus key."""
    if field in row:
        return row.get(field), field
    for alias, canonical in _FIELD_ALIASES.items():
        if canonical == field and alias in row:
            return row.get(alias), alias
    return None, None


def _iter_rows(rows: Any) -> Iterable[Mapping[str, Any]]:
    if rows is None:
        return ()
    if hasattr(rows, "iterrows"):
        return (row.to_dict() for _, row in rows.iterrows())
    if isinstance(rows, Mapping):
        return (rows,)
    return (row for row in rows if isinstance(row, Mapping))


def _period_kind(row: Mapping[str, Any], source: Any, default: str | None) -> str:
    value = row.get("period_kind") or row.get("window_kind") or row.get("period")
    if value:
        return str(value)
    if row.get("realtime") is True or row.get("is_realtime") is True:
        return "realtime_single_day"
    label = _normalise_source_text(source)
    if "instant" in label or "snapshot" in label or "即时" in label:
        return "realtime_single_day"
    return default or "historical_daily"


def _normalise_date_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # Provider frames commonly stringify a date as ``YYYY-MM-DD 00:00:00``.
    # Keep the calendar date only so equivalent source rows align.
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        text = text[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def _date_value(row: Mapping[str, Any]) -> str | None:
    for key in ("measurement_date", "date", "日期", "opendate", "trade_date", "交易日期", "as_of"):
        value = _normalise_date_text(row.get(key))
        if value:
            return value
    return None


def build_source_evidence(
    rows: Any,
    *,
    symbol: str,
    requested_as_of: str,
    retrieved_at: str | None,
    source: str,
    raw_unit: str = "元",
    algorithm_group: str | None = None,
    period_kind: str | None = None,
    window: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize a source's raw fields without collapsing official semantics.

    This accepts both AkShare Chinese column names and canonical fields. Values
    are converted to exact 亿元, while the original value and unit remain in
    each record. It intentionally does not turn ``netamount`` into ``r0_net``.
    """
    group = infer_algorithm_group(source, algorithm_group)
    family = source_family(source)
    records: list[dict[str, Any]] = []
    for row in _iter_rows(rows):
        date = _date_value(row)
        # Never manufacture a measurement date from the request.  An undated
        # row may remain in the audit trail, but source selection must reject it.
        as_of_value = row.get("as_of") or date
        as_of = str(as_of_value).strip() if as_of_value is not None else None
        effective_period = _period_kind(row, source, period_kind)
        effective_window = str(
            row.get("time_window") or row.get("window") or window or
            ("1d" if effective_period != "five_day_cumulative" else "5d")
        )
        record: dict[str, Any] = {
            "date": date,
            "measurement_date": date,
            "as_of": as_of,
            "requested_as_of": requested_as_of,
            "symbol": symbol,
            "source": source,
            "source_family": family,
            "algorithm_group": group,
            "source_group": group,
            "algorithm_generation": "new" if group == NEW_ALGORITHM_GROUP else group,
            "legacy_web_algorithm": group == LEGACY_WEB_ALGORITHM,
            "period_kind": effective_period,
            "time_window": effective_window,
            "window": effective_window,
            "raw_unit": _unit_name(row.get("raw_unit") or raw_unit),
            "unit": "亿元",
            "retrieved_at": retrieved_at,
            "status": "available",
        }
        field_semantics: dict[str, str] = {}
        field_categories: dict[str, str] = {}
        for field in _FIELD_ORDER:
            raw_value, raw_key = _row_value(row, field)
            if raw_key is None or raw_value is None:
                continue
            field_unit = row.get(f"{field}_unit") or row.get("unit") or raw_unit
            parsed, parsed_unit = _amount_to_yi(raw_value, field_unit)
            if parsed is None:
                continue
            record[field] = _decimal_text(parsed)
            record[f"{field}_raw"] = str(raw_value)
            record[f"{field}_raw_unit"] = parsed_unit
            field_semantics[field] = _FIELD_SEMANTICS.get(field, _COMPONENT_SEMANTICS.get(field, field))
            field_categories[field] = _FIELD_CATEGORIES.get(field, "main_force_component")
        for alias, component in _COMPONENT_ALIASES.items():
            if alias not in row or row.get(alias) is None:
                continue
            parsed, parsed_unit = _amount_to_yi(row.get(alias), row.get("unit") or raw_unit)
            if parsed is None:
                continue
            record[component] = _decimal_text(parsed)
            record[f"{component}_raw"] = str(row.get(alias))
            record[f"{component}_raw_unit"] = parsed_unit
            record.setdefault("components", {})[component] = _decimal_text(parsed)
            record.setdefault("component_semantics", {})[component] = _COMPONENT_SEMANTICS[component]
        components = record.get("components", {})
        if "r0_net" not in record and {"super_large_net", "large_net"}.issubset(components):
            derived = (
                decimal_value(components["super_large_net"]) or Decimal("0")
            ) + (decimal_value(components["large_net"]) or Decimal("0"))
            record["r0_net"] = _decimal_text(derived)
            record["r0_net_raw"] = f"{record['super_large_net_raw']} + {record['large_net_raw']}"
            record["r0_net_raw_unit"] = "亿元"
            field_semantics["r0_net"] = _FIELD_SEMANTICS["r0_net"]
            field_categories["r0_net"] = _FIELD_CATEGORIES["r0_net"]
            record["derived_fields"] = {"r0_net": "super_large_net + large_net"}
        # Some feeds expose one canonical field/value pair instead of columns.
        explicit_field = row.get("field") or row.get("字段")
        if explicit_field and row.get("value") is not None:
            canonical = _FIELD_ALIASES.get(str(explicit_field)) or _COMPONENT_ALIASES.get(str(explicit_field)) or str(explicit_field)
            if canonical in _FIELD_SEMANTICS:
                parsed, parsed_unit = _amount_to_yi(
                    row.get("value"), row.get("value_unit") or row.get("unit") or raw_unit
                )
                if parsed is not None:
                    record[canonical] = _decimal_text(parsed)
                    record[f"{canonical}_raw"] = str(row.get("value"))
                    record[f"{canonical}_raw_unit"] = parsed_unit
                    field_semantics[canonical] = _FIELD_SEMANTICS[canonical]
                    field_categories[canonical] = _FIELD_CATEGORIES.get(canonical, "main_force_component")
            elif canonical in _COMPONENT_SEMANTICS:
                parsed, parsed_unit = _amount_to_yi(
                    row.get("value"), row.get("value_unit") or row.get("unit") or raw_unit
                )
                if parsed is not None:
                    record[canonical] = _decimal_text(parsed)
                    record[f"{canonical}_raw"] = str(row.get("value"))
                    record[f"{canonical}_raw_unit"] = parsed_unit
                    record.setdefault("components", {})[canonical] = _decimal_text(parsed)
                    record.setdefault("component_semantics", {})[canonical] = _COMPONENT_SEMANTICS[canonical]
        if not field_semantics:
            continue
        record["field_semantics"] = field_semantics
        record["field_categories"] = field_categories
        records.append(record)
    return records


def build_ths_evidence(
    rows: Any,
    *,
    symbol: str,
    requested_as_of: str,
    retrieved_at: str | None,
    source: str = "ths_instant_snapshot",
    period_kind: str = "realtime_single_day",
) -> list[dict[str, Any]]:
    """Build new-algorithm Tonghuashun evidence, keeping ``净额`` as total net."""
    return build_source_evidence(
        rows,
        symbol=symbol,
        requested_as_of=requested_as_of,
        retrieved_at=retrieved_at,
        source=source,
        raw_unit="亿元",
        algorithm_group=NEW_ALGORITHM_GROUP,
        period_kind=period_kind,
    )


def build_sina_app_evidence(
    rows: Any,
    *,
    symbol: str,
    requested_as_of: str,
    retrieved_at: str | None,
    source: str = "sina_app_manual_calibration",
    raw_unit: str = "元",
    period_kind: str = "realtime_single_day",
) -> list[dict[str, Any]]:
    """Keep screenshot/manual App observations typed; never treat them as auto evidence."""
    records = build_source_evidence(
        rows,
        symbol=symbol,
        requested_as_of=requested_as_of,
        retrieved_at=retrieved_at,
        source=source,
        raw_unit=raw_unit,
        algorithm_group=NEW_ALGORITHM_GROUP,
        period_kind=period_kind,
    )
    for record in records:
        record["status"] = "manual_observation"
        record["manual_calibration"] = True
        record["automated_consensus_eligible"] = False
    return records


def build_em_evidence(
    frame: Any,
    *,
    symbol: str,
    requested_as_of: str,
    retrieved_at: str | None,
    source: str = "eastmoney_individual_fund_flow",
) -> list[dict[str, Any]]:
    """Build evidence from Eastmoney's main-force-only daily series.

    Eastmoney's ``主力净流入-净额`` is r0_net semantics. It is deliberately not
    copied into netamount, whose total-net semantics are unavailable here.
    """
    records = build_source_evidence(
        frame,
        symbol=symbol,
        requested_as_of=requested_as_of,
        retrieved_at=retrieved_at,
        source=source,
        raw_unit="元",
        algorithm_group=NEW_ALGORITHM_GROUP,
        period_kind="historical_daily",
        window="1d",
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
    """Build lossless records from the legacy Sina Web historical endpoint."""
    records: list[dict[str, Any]] = []
    for row in rows:
        date = str(row.get("opendate", "")).strip()
        if not date:
            continue
        netamount_raw = decimal_value(row.get("netamount"))
        r0_net_raw = decimal_value(row.get("r0_net"))
        if netamount_raw is None and r0_net_raw is None:
            continue
        record = {
            "date": date,
            "measurement_date": date,
            "as_of": date,
            "netamount": _decimal_text(netamount_raw / YI) if netamount_raw is not None else None,
            "r0_net": _decimal_text(r0_net_raw / YI) if r0_net_raw is not None else None,
            "netamount_raw": _decimal_text(netamount_raw),
            "netamount_raw_unit": "元",
            "r0_net_raw": _decimal_text(r0_net_raw),
            "r0_net_raw_unit": "元",
            "raw_unit": "元",
            "unit": "亿元",
            "source": source,
            "source_family": "sina_web",
            "symbol": symbol,
            "requested_as_of": requested_as_of,
            "retrieved_at": retrieved_at,
            "status": "available",
            "algorithm_group": LEGACY_WEB_ALGORITHM,
            "source_group": LEGACY_WEB_ALGORITHM,
            "algorithm_generation": LEGACY_WEB_ALGORITHM,
            "legacy_web_algorithm": True,
            "period_kind": "historical_daily",
            "time_window": "1d",
            "window": "1d",
            "field_semantics": {
                "netamount": _FIELD_SEMANTICS["netamount"],
                "r0_net": _FIELD_SEMANTICS["r0_net"],
            },
            "field_categories": {
                "netamount": _FIELD_CATEGORIES["netamount"],
                "r0_net": _FIELD_CATEGORIES["r0_net"],
            },
            "netamount_semantics": _FIELD_SEMANTICS["netamount"],
            "r0_net_semantics": _FIELD_SEMANTICS["r0_net"],
        }
        records.append(record)
    return records


def build_provider_text(
    text: str,
    *,
    symbol: str,
    requested_as_of: str | None,
    source: str,
    reason: str,
    retrieved_at: str | None = None,
    algorithm_group: str | None = None,
    period_kind: str | None = None,
    field: str | None = "r0_net",
    raw_unit: str | None = "元",
    actual_as_of: str | None = None,
    failure_category: str = "source_unavailable",
) -> FundFlowText:
    """Keep formatted provider text while exposing an explicit evidence gap."""
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
            algorithm_group=algorithm_group,
            period_kind=period_kind,
            field=field,
            raw_unit=raw_unit,
            actual_as_of=actual_as_of,
            failure_category=failure_category,
        ),
    )


def build_gap_meta(
    *,
    symbol: str,
    requested_as_of: str | None,
    source: str,
    status: str,
    reason: str,
    retrieved_at: str | None = None,
    algorithm_group: str | None = None,
    period_kind: str | None = None,
    field: str | None = "r0_net",
    raw_unit: str | None = "元",
    actual_as_of: str | None = None,
    failure_category: str = "source_unavailable",
) -> dict[str, Any]:
    """Build explicit evidence-gap metadata without fabricating daily values."""
    group = infer_algorithm_group(source, algorithm_group)
    return {
        "symbol": symbol,
        "requested_as_of": requested_as_of,
        "source": source,
        "source_family": source_family(source),
        "algorithm_group": group,
        "source_group": group,
        "legacy_web_algorithm": group == LEGACY_WEB_ALGORITHM,
        "period_kind": period_kind,
        "field": field,
        "raw_unit": raw_unit,
        "unit": "亿元",
        "status": status,
        "direction": "blocked",
        "direction_allowed": False,
        "reason": reason,
        "failure_category": failure_category,
        "failure_categories": [failure_category],
        "attempted_sources": [],
        "fallback_errors": [],
        "failure_chain": [],
        "final_source": "unavailable",
        "last_attempted_source": None,
        "gap": f"【数据获取失败】资金流 evidence：{reason}",
        "retrieved_at": retrieved_at,
        "actual_as_of": actual_as_of,
        "as_of": actual_as_of,
    }


def _sum_field(records: Iterable[Mapping[str, Any]], field: str) -> Decimal | None:
    values = [decimal_value(record.get(field)) for record in records]
    usable = [value for value in values if value is not None]
    return sum(usable, Decimal("0")) if usable else None


def _normalise_summary_record(record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize raw amounts and dates before any cumulative arithmetic."""
    date = _date_value(record)
    if date is None:
        return None, "unparseable_date"
    normalized = dict(record)
    normalized["date"] = date
    normalized["period_kind"] = str(
        record.get("period_kind") or record.get("window_kind") or record.get("period") or "historical_daily"
    )
    normalized["window"] = str(record.get("window") or record.get("time_window") or "1d")
    explicit_field = _canonical_field(record.get("field") or record.get("字段"))
    if explicit_field and record.get("value") is not None and explicit_field not in normalized:
        raw_value = record.get("value")
        raw_unit = record.get("value_unit") or record.get("unit") or record.get("raw_unit")
        amount, parsed_unit = _amount_to_yi(raw_value, raw_unit)
        if amount is None:
            return None, f"invalid_{explicit_field}_unit_or_amount"
        normalized[explicit_field] = _decimal_text(amount)
        normalized[f"{explicit_field}_raw_unit"] = parsed_unit
    for field in ("netamount", "r0_net", "r0_in", "r0_out", "r0"):
        if field not in record or record.get(field) is None:
            continue
        raw_value = record.get(f"{field}_raw", record.get(field))
        raw_unit = record.get(f"{field}_raw_unit", record.get("raw_unit", record.get("unit")))
        amount, parsed_unit = _amount_to_yi(raw_value, raw_unit)
        if amount is None:
            return None, f"invalid_{field}_unit_or_amount"
        normalized[field] = _decimal_text(amount)
        normalized[f"{field}_raw_unit"] = parsed_unit
    return normalized, None


def _summary_conflict(
    usable: list[dict[str, Any]],
    *,
    window_days: int,
    reason: str,
    invalid_records: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "window_days": window_days,
        "record_count": len(usable),
        "required_window_days": window_days,
        "status": "data_conflict",
        "data_conflict": True,
        "reason": reason,
        "invalid_records": list(invalid_records or []),
        "dates": [str(record.get("date")) for record in usable],
        "netamount": None,
        "r0_net": None,
        "unit": "亿元",
        "semantics": {
            "netamount": _FIELD_SEMANTICS["netamount"],
            "r0_net": _FIELD_SEMANTICS["r0_net"],
        },
    }


def summarize_evidence(
    records: Iterable[Mapping[str, Any]],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    selected_source: str | None = None,
    selected_field: str | None = None,
    requested_as_of: str | None = None,
) -> dict[str, Any]:
    """Compute exact totals for one selected source/field only.

    When selection metadata is supplied, lower-ranked side evidence is ignored
    for arithmetic while remaining available to the report provenance layer.
    """
    usable: list[dict[str, Any]] = []
    invalid_records: list[str] = []
    request_date = _normalise_date_text(requested_as_of)
    for record in records:
        if not isinstance(record, Mapping) or record.get("status") != "available":
            continue
        if selected_source and str(record.get("source") or "") != str(selected_source):
            continue
        record_date = _date_value(record)
        if request_date and record_date and record_date > request_date:
            continue
        candidate = dict(record)
        if selected_field:
            # Keep arithmetic tied to one official field; other fields remain
            # in the original evidence record for side-by-side audit.
            for other_field in ("netamount", "r0_net", "r0_in", "r0_out", "r0"):
                if other_field != selected_field:
                    candidate.pop(other_field, None)
                    candidate.pop(f"{other_field}_raw", None)
                    candidate.pop(f"{other_field}_raw_unit", None)
        normalized, error = _normalise_summary_record(candidate)
        if normalized is None:
            invalid_records.append(error or "invalid_record")
        else:
            usable.append(normalized)
    if invalid_records:
        return _summary_conflict(
            usable,
            window_days=window_days,
            reason="结构化 evidence 含无效日期、单位或金额，禁止累计",
            invalid_records=invalid_records,
        )
    usable.sort(key=lambda record: str(record.get("date")))
    period_kinds = {str(record.get("period_kind")) for record in usable if record.get("period_kind")}
    windows = {str(record.get("window") or record.get("time_window")) for record in usable if record.get("window") or record.get("time_window")}
    source_ids = {str(record.get("source") or "") for record in usable}
    mixed_periods = len(period_kinds) > 1 or ("five_day_cumulative" in period_kinds and len(usable) > 1)
    invalid_windows = [
        record for record in usable
        if str(record.get("window") or "1d") != "1d"
        or str(record.get("period_kind")) in {"five_day_cumulative", "five_day_aggregate"}
        or (str(record.get("period_kind")) == "realtime_single_day" and len(usable) > 1)
    ]
    if invalid_windows:
        return _summary_conflict(
            usable,
            window_days=window_days,
            reason="逐日累计窗口包含实时多日或累计 period/window，禁止跨区间相加",
        )
    if len(source_ids) > 1:
        return {
            "window_days": window_days,
            "record_count": len(usable),
            "status": "data_conflict",
            "data_conflict": True,
            "reason": "多来源记录不能直接相加，必须先按字段/日期/窗口做新算法组共识",
            "dates": [str(record.get("date")) for record in usable],
            "netamount": None,
            "r0_net": None,
            "unit": "亿元",
            "semantics": {
                "netamount": _FIELD_SEMANTICS["netamount"],
                "r0_net": _FIELD_SEMANTICS["r0_net"],
            },
        }
    if mixed_periods:
        return {
            "window_days": window_days,
            "record_count": len(usable),
            "status": "data_conflict",
            "data_conflict": True,
            "reason": "records mix real-time, historical-daily, or cumulative windows",
            "dates": [str(record.get("date")) for record in usable],
            "netamount": None,
            "r0_net": None,
            "unit": "亿元",
            "semantics": {
                "netamount": _FIELD_SEMANTICS["netamount"],
                "r0_net": _FIELD_SEMANTICS["r0_net"],
            },
        }
    selected = usable[-window_days:] if len(usable) > window_days else usable
    selected_dates = [str(record.get("date")) for record in selected]
    if len(set(selected_dates)) != len(selected_dates) or any(
        record.get("period_kind") == "five_day_cumulative" for record in selected
    ):
        return {
            "window_days": window_days,
            "record_count": len(selected),
            "status": "data_conflict",
            "data_conflict": True,
            "reason": "累计值或重复日期不能按逐日记录相加",
            "dates": selected_dates,
            "netamount": None,
            "r0_net": None,
            "unit": "亿元",
            "semantics": {
                "netamount": _FIELD_SEMANTICS["netamount"],
                "r0_net": _FIELD_SEMANTICS["r0_net"],
            },
        }
    netamount = _sum_field(selected, "netamount")
    r0_net = _sum_field(selected, "r0_net")
    status = "available" if len(selected) == window_days and netamount is not None and r0_net is not None else "partial"
    return {
        "window_days": window_days,
        "record_count": len(selected),
        "required_window_days": window_days,
        "status": status,
        "data_conflict": False,
        "dates": [str(record.get("date")) for record in selected],
        "netamount": _decimal_text(netamount),
        "r0_net": _decimal_text(r0_net),
        "unit": "亿元",
        "windows": sorted(windows),
        "semantics": {
            "netamount": _FIELD_SEMANTICS["netamount"],
            "r0_net": _FIELD_SEMANTICS["r0_net"],
        },
    }


def _canonical_field(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in _FIELD_SEMANTICS or text in _COMPONENT_SEMANTICS:
        return text
    return _FIELD_ALIASES.get(text) or _COMPONENT_ALIASES.get(text)


def _observation_period(record: Mapping[str, Any]) -> str:
    period = record.get("period_kind") or record.get("window_kind") or record.get("period")
    if period:
        return str(period)
    if record.get("realtime") is True or record.get("is_realtime") is True:
        return "realtime_single_day"
    return "historical_daily"


def _observation_window(record: Mapping[str, Any], period: str) -> str:
    value = record.get("time_window") or record.get("window")
    if value:
        return str(value)
    return "5d" if period == "five_day_cumulative" else "1d"


def _observation_date(record: Mapping[str, Any]) -> str | None:
    value = _date_value(record)
    return value


def _record_observations(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = str(record.get("source") or "unknown_source")
    group = infer_algorithm_group(source, record.get("algorithm_group"))
    period = _observation_period(record)
    window = _observation_window(record, period)
    date = _observation_date(record)
    symbol = str(record.get("symbol") or "").strip() or None
    unit = _unit_name(record.get("unit") or record.get("raw_unit") or "亿元")
    observations: list[dict[str, Any]] = []
    explicit_field = _canonical_field(record.get("field") or record.get("字段"))
    if explicit_field is not None and record.get("value") is not None:
        fields = [(explicit_field, record.get("value"), record.get("value_unit") or unit)]
    else:
        fields = []
        for field in _FIELD_ORDER:
            value = record.get(field)
            if value is not None:
                fields.append((field, value, record.get(f"{field}_unit") or unit))
        components = record.get("components") if isinstance(record.get("components"), Mapping) else {}
        for component in _COMPONENT_SEMANTICS:
            value = record.get(component, components.get(component))
            if value is not None:
                fields.append((component, value, record.get(f"{component}_unit") or unit))
    for field, value, value_unit in fields:
        parsed, parsed_unit = _amount_to_yi(value, value_unit)
        if parsed is None:
            continue
        observations.append(
            {
                "source": source,
                "source_family": source_family(source),
                "algorithm_group": group,
                "legacy_web_algorithm": group == LEGACY_WEB_ALGORITHM,
                "symbol": symbol,
                "date": date,
                "measurement_time": record.get("measurement_time") or record.get("timestamp") or record.get("时间"),
                "period_kind": period,
                "time_window": window,
                "field": field,
                "field_category": _FIELD_CATEGORIES.get(field, "main_force_component"),
                "value": parsed,
                "value_text": _decimal_text(parsed),
                "raw_value": str(value),
                "raw_unit": record.get(f"{field}_raw_unit") or parsed_unit,
                "unit": "亿元",
                "record": dict(record),
            }
        )
    return observations


def _source_fallback_rank(source: Any) -> int:
    """Return the stable priority category for a source label."""
    label = str(source or "").strip()
    if label in _SOURCE_PRIORITY:
        return _SOURCE_PRIORITY[label]
    group = infer_algorithm_group(label)
    return 4 if group == LEGACY_WEB_ALGORITHM else 3 if group == NEW_ALGORITHM_GROUP else 99


def _source_order_index(source: Any) -> int:
    label = str(source or "").strip()
    try:
        return _SOURCE_PRIORITY_ORDER.index(label)
    except ValueError:
        return len(_SOURCE_PRIORITY_ORDER)


def _field_semantics_are_valid(record: Mapping[str, Any], field: str) -> bool:
    """Reject a contradictory provider label without requiring a label."""
    expected = _FIELD_SEMANTICS.get(field) or _COMPONENT_SEMANTICS.get(field)
    if expected is None:
        return False
    semantics = record.get("field_semantics")
    if isinstance(semantics, Mapping):
        supplied = semantics.get(field)
    else:
        supplied = record.get(f"{field}_semantics")
    if supplied is None:
        supplied = record.get("upstream_field_semantics") if field == "r0_net" else None
    if supplied is None or not str(supplied).strip():
        return True
    text = str(supplied).strip().lower()
    if field in {"r0_net", "netamount", "super_large_net", "large_net"}:
        if "净" not in text:
            return False
        if field == "r0_net" and "主力" not in text:
            return False
        if field == "netamount" and "主力" in text and "总" not in text:
            return False
    elif field == "r0_in":
        if "流入" not in text or "净" in text:
            return False
    elif field == "r0_out":
        if "流出" not in text or "净" in text:
            return False
    elif field == "r0" and "主力" not in text:
        return False
    return True


def _observation_raw_value(item: Mapping[str, Any]) -> str:
    record = item.get("record")
    if isinstance(record, Mapping):
        raw = record.get(f"{item.get('field')}_raw")
        if raw is not None:
            return str(raw)
    return str(item.get("raw_value") or item.get("value_text") or "")


def _observation_audit(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": item["source"],
        "source_family": item["source_family"],
        "algorithm_group": item["algorithm_group"],
        "legacy_web_algorithm": item["legacy_web_algorithm"],
        "value": item["value_text"],
        "selected_value": item["value_text"],
        "raw_value": _observation_raw_value(item),
        "raw_unit": item["raw_unit"],
        "unit": item["unit"],
        "date": item["date"],
        "period_kind": item["period_kind"],
        "time_window": item["time_window"],
        "field": item["field"],
        "field_category": item["field_category"],
        "transport_provider": (item.get("record") or {}).get("transport_provider"),
    }


def _selection_direction_summary(field: str, direction: str) -> str:
    scope = "主力" if field in _MAIN_FORCE_FIELDS else "总资金" if field == "netamount" else "订单组成项"
    if direction == "inflow":
        return f"{scope}偏流入"
    if direction == "outflow":
        return f"{scope}偏流出"
    return f"{scope}接近平衡"


def select_fund_flow_source(
    records: Iterable[Mapping[str, Any]],
    *,
    symbol: str | None = None,
    requested_as_of: str | None = None,
    field: str | None = None,
    failure_chain: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select the first valid source without requiring cross-source consensus.

    Sources are ranked ``Tushare DC -> Tushare THS -> other new algorithms ->
    Sina legacy``.  A candidate is valid when at least one field from the same
    source has a real measurement date, a known unit, a finite value, and a
    non-contradictory field semantic.  Values are aggregated only within that
    source, field, date, period, and window; lower-ranked valid observations are
    retained as side evidence and never mixed into the selected value.
    """
    request_date = _normalise_date_text(requested_as_of)
    requested_symbol = str(symbol or "").strip() or None
    all_records: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            invalid.append({"source": "unknown_source", "reason": "record_not_object"})
            continue
        record = dict(raw_record)
        source = str(record.get("source") or "unknown_source").strip() or "unknown_source"
        group = infer_algorithm_group(source, record.get("algorithm_group"))
        rank = _source_fallback_rank(source)
        record.setdefault("source_family", source_family(source))
        record.setdefault("algorithm_group", group)
        record.setdefault("legacy_web_algorithm", group == LEGACY_WEB_ALGORITHM)
        record.setdefault("fallback_rank", rank)
        all_records.append(record)
        if record.get("status", "available") != "available":
            invalid.append({"source": source, "fallback_rank": rank, "reason": "status_unavailable"})
            continue
        if group not in {NEW_ALGORITHM_GROUP, LEGACY_WEB_ALGORITHM}:
            invalid.append({"source": source, "fallback_rank": rank, "reason": "unknown_algorithm_group"})
            continue
        record_symbol = str(record.get("symbol") or "").strip() or None
        if requested_symbol and record_symbol and record_symbol != requested_symbol:
            invalid.append({"source": source, "fallback_rank": rank, "reason": "symbol_mismatch"})
            continue
        date = _observation_date(record)
        if date is None:
            invalid.append({"source": source, "fallback_rank": rank, "reason": "missing_measurement_date"})
            continue
        if request_date and date > request_date:
            invalid.append({
                "source": source,
                "fallback_rank": rank,
                "date": date,
                "reason": "future_measurement_date",
            })
            continue
        unit = _unit_name(record.get("unit") or "")
        if unit != "亿元":
            invalid.append({
                "source": source,
                "fallback_rank": rank,
                "reason": "invalid_normalized_unit",
                "unit": unit,
            })
            continue
        record_observations = _record_observations(record)
        if not record_observations:
            invalid.append({"source": source, "fallback_rank": rank, "reason": "no_usable_field"})
            continue
        accepted = False
        for item in record_observations:
            if field and item["field"] != field:
                continue
            raw_unit = _unit_name(
                record.get(f"{item['field']}_raw_unit")
                or record.get("raw_unit")
                or item.get("raw_unit")
                or item.get("unit")
            )
            if raw_unit not in {"元", "万", "万元", "亿", "亿元", "万亿"}:
                invalid.append({
                    "source": source,
                    "fallback_rank": rank,
                    "field": item["field"],
                    "reason": "invalid_raw_unit",
                    "unit": raw_unit,
                })
                continue
            if not _field_semantics_are_valid(record, item["field"]):
                invalid.append({
                    "source": source,
                    "fallback_rank": rank,
                    "field": item["field"],
                    "reason": "field_semantics_invalid",
                })
                continue
            item = dict(item)
            item["record"] = record
            item["fallback_rank"] = rank
            observations.append(item)
            accepted = True
        if not accepted:
            invalid.append({
                "source": source,
                "fallback_rank": rank,
                "reason": "requested_field_unavailable" if field else "no_semantically_valid_field",
            })

    raw_values = [_observation_audit(item) for item in observations]
    base: dict[str, Any] = {
        "algorithm_group": NEW_ALGORITHM_GROUP,
        "group_priority": "tushare_dc_then_ths_then_other_new_then_legacy",
        "selection_mode": "priority",
        "consensus_required": False,
        "status": "data_conflict",
        "data_conflict": True,
        "direction": "blocked",
        "direction_allowed": False,
        "selected_source": None,
        "selected_source_family": None,
        "selected_algorithm_group": None,
        "selected_field": None,
        "selected_value": None,
        "selected_direction": None,
        "fallback_rank": None,
        "legacy_reference": False,
        "legacy_web_algorithm": False,
        "raw_values": raw_values,
        "all_records": all_records,
        "legacy_sources": [item for item in raw_values if item["algorithm_group"] == LEGACY_WEB_ALGORITHM],
        "unknown_sources": [],
        "new_algorithm_sources": [item for item in raw_values if item["algorithm_group"] == NEW_ALGORITHM_GROUP],
        "side_evidence": [],
        "field_results": {},
        "failure_chain": [dict(item) for item in (failure_chain or ())] + invalid,
        "invalid_records": invalid,
        "relative_dispersion_threshold": _decimal_text(DEFAULT_RELATIVE_DISPERSION),
        "legacy_web_is_corroboration_only": any(
            item["algorithm_group"] == LEGACY_WEB_ALGORITHM for item in raw_values
        ),
    }
    if not observations:
        base.update({
            "reason_code": "no_valid_source",
            "selection_reason": "no_valid_fund_flow_source",
            "reason": "所有资金流来源均缺少日期、字段、单位或有效值",
            "hard_guard": {
                "blocked": True,
                "direction_allowed": False,
                "reason": "所有资金流来源均不可用",
            },
            "legacy_reference": False,
            "fallback_rank": None,
        })
        return base

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for item in observations:
        key = (
            item["source"],
            item["field"],
            item["period_kind"],
            item["time_window"],
        )
        grouped.setdefault(key, []).append(item)

    field_rank = {name: index for index, name in enumerate(_FIELD_ORDER)}
    candidates: list[dict[str, Any]] = []
    for (source, candidate_field, period, window), items in grouped.items():
        items = sorted(items, key=lambda item: str(item.get("date") or ""))
        by_date: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            by_date.setdefault(str(item["date"]), []).append(item)
        duplicate_conflict = any(
            len({entry["value"] for entry in same_date}) > 1
            for same_date in by_date.values()
        )
        rank = _source_fallback_rank(source)
        if duplicate_conflict:
            invalid.append({
                "source": source,
                "fallback_rank": rank,
                "field": candidate_field,
                "reason": "duplicate_date_conflict",
            })
            continue
        deduped = [same_date[0] for same_date in by_date.values()]
        deduped.sort(key=lambda item: str(item.get("date") or ""))
        selected_items = deduped[-DEFAULT_WINDOW_DAYS:]
        if period in {"five_day_cumulative", "five_day_aggregate"} or window != "1d":
            value = selected_items[-1]["value"]
        else:
            value = sum((item["value"] for item in selected_items), Decimal("0"))
        latest = selected_items[-1]
        candidates.append({
            "source": source,
            "field": candidate_field,
            "period_kind": period,
            "time_window": window,
            "fallback_rank": rank,
            "source_order": _source_order_index(source),
            "field_rank": field_rank.get(candidate_field, len(field_rank)),
            "items": selected_items,
            "latest": latest,
            "value": value,
        })

    if not candidates:
        base.update({
            "reason_code": "no_valid_source",
            "selection_reason": "no_valid_fund_flow_source",
            "reason": "资金流来源存在记录但没有同源、同字段、同日期和同窗口的有效候选",
            "failure_chain": [dict(item) for item in (failure_chain or ())] + invalid,
            "invalid_records": invalid,
            "hard_guard": {
                "blocked": True,
                "direction_allowed": False,
                "reason": "没有满足字段、日期、单位和窗口校验的资金流来源",
            },
            "legacy_reference": False,
            "fallback_rank": None,
        })
        return base

    def _candidate_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        latest_date = str(item["latest"].get("date") or "")
        try:
            latest_number = int(latest_date.replace("-", ""))
        except ValueError:
            latest_number = 0
        period_rank = 0 if item["period_kind"] == "historical_daily" else 1
        return (
            item["fallback_rank"],
            item["source_order"],
            item["field_rank"],
            period_rank,
            -latest_number,
        )

    candidates.sort(key=_candidate_sort_key)
    selected_candidate = candidates[0]
    selected_source = selected_candidate["source"]
    selected_field = selected_candidate["field"]
    selected_items = selected_candidate["items"]
    selected_value = selected_candidate["value"]
    selected_latest = selected_candidate["latest"]
    selected_group = infer_algorithm_group(selected_source)
    selected_direction = _direction_for_field(selected_field, selected_value)
    field_results: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        candidate_direction = _direction_for_field(
            candidate["field"], candidate["value"]
        )
        field_results.setdefault(candidate["field"], []).append({
            "source": candidate["source"],
            "source_family": source_family(candidate["source"]),
            "algorithm_group": infer_algorithm_group(candidate["source"]),
            "fallback_rank": candidate["fallback_rank"],
            "field": candidate["field"],
            "period_kind": candidate["period_kind"],
            "time_window": candidate["time_window"],
            "dates": [item["date"] for item in candidate["items"]],
            "selected_value": _decimal_text(candidate["value"]),
            "direction": candidate_direction,
            "selected": candidate is selected_candidate,
        })
    base["field_results"] = field_results

    # Mark the selected source/field while retaining every valid lower-ranked
    # record for audit and side-by-side reporting.
    selected_audit = []
    side_audit = []
    selected_dates = {str(item.get("date")) for item in selected_items}
    for item in raw_values:
        is_selected = (
            item["source"] == selected_source
            and item["field"] == selected_field
            and str(item.get("date")) in selected_dates
        )
        item["selected"] = is_selected
        item["fallback_rank"] = _source_fallback_rank(item["source"])
        if is_selected:
            selected_audit.append(item)
        else:
            side_audit.append(item)

    selected_reason = (
        "no_new_algorithm_source_legacy_fallback"
        if selected_group == LEGACY_WEB_ALGORITHM
        else "first_valid_source_by_priority"
    )
    selected_record = selected_latest.get("record") or {}
    transport_provider = selected_record.get("transport_provider")
    if transport_provider is None:
        transport_provider = {
            "eastmoney": "eastmoney",
            "ths": "ths",
            "sina_web": "sina_http",
            "sina_app": "sina_app",
        }.get(source_family(selected_source), source_family(selected_source))
    selected_as_of = str(selected_latest.get("date"))
    base.update({
        "status": "consensus",  # compatibility envelope; selection_mode explains the semantics.
        "data_conflict": False,
        "reason_code": "legacy_fallback" if selected_group == LEGACY_WEB_ALGORITHM else "priority_source_selected",
        "selection_reason": selected_reason,
        "reason": (
            "新算法来源均不可用，使用新浪 legacy 自身字段方向；仅供参考"
            if selected_group == LEGACY_WEB_ALGORITHM
            else "按固定来源优先级选择首个字段、日期、单位和有效值均合格的来源"
        ),
        "selected_source": selected_source,
        "selected_source_family": source_family(selected_source),
        "selected_algorithm_group": selected_group,
        "selected_field": selected_field,
        "field": selected_field,
        "field_category": _FIELD_CATEGORIES.get(selected_field, "main_force_component"),
        "selected_value": _decimal_text(selected_value),
        "selected_value_unit": "亿元",
        "selected_direction": selected_direction,
        "direction": selected_direction,
        "direction_allowed": True,
        "direction_summary": _selection_direction_summary(selected_field, selected_direction),
        "selected_as_of": selected_as_of,
        "actual_as_of": selected_as_of,
        "as_of": selected_as_of,
        "transport_provider": transport_provider,
        "fallback_rank": selected_candidate["fallback_rank"],
        "legacy_reference": selected_group == LEGACY_WEB_ALGORITHM,
        "legacy_reference_only": selected_group == LEGACY_WEB_ALGORITHM,
        "legacy_web_algorithm": selected_group == LEGACY_WEB_ALGORITHM,
        "selected_evidence": selected_audit,
        "side_evidence": side_audit,
        "new_algorithm_sources": [item for item in raw_values if item["algorithm_group"] == NEW_ALGORITHM_GROUP],
        "legacy_sources": [item for item in raw_values if item["algorithm_group"] == LEGACY_WEB_ALGORITHM],
        "contributing_sources": [selected_source],
        "consensus_value": _decimal_text(selected_value),
        "median": _decimal_text(selected_value),
        "mad": "0",
        "relative_dispersion": "0",
        "hard_guard": {
            "blocked": False,
            "direction_allowed": True,
            "reason": (
                "legacy Web 自身方向可展示但必须醒目标注旧算法"
                if selected_group == LEGACY_WEB_ALGORITHM
                else "首个优先级来源字段、日期、单位和有效值校验通过"
            ),
        },
    })
    return base


def _median(values: list[Decimal]) -> Decimal:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal("2")


def _dispersion(values: list[Decimal]) -> tuple[Decimal, Decimal, Decimal]:
    median = _median(values)
    deviations = [abs(value - median) for value in values]
    mad = _median(deviations)
    relative = mad if abs(median) <= _EPSILON else mad / abs(median)
    return median, mad, relative


def _evaluate_observation_group(
    observations: list[dict[str, Any]],
    *,
    relative_dispersion_threshold: Decimal,
) -> dict[str, Any]:
    raw_values = [
        {
            "source": item["source"],
            "source_family": item["source_family"],
            "value": item["value_text"],
            "raw_value": item["raw_value"],
            "raw_unit": item["raw_unit"],
            "unit": item["unit"],
            "date": item["date"],
            "period_kind": item["period_kind"],
            "time_window": item["time_window"],
            "field": item["field"],
        }
        for item in observations
    ]
    by_source: dict[str, list[Decimal]] = {}
    source_families: dict[str, set[str]] = {}
    for item in observations:
        family = str(item.get("source_family") or source_family(item["source"]))
        source_families.setdefault(family, set()).add(item["source"])
        by_source.setdefault(family, []).append(item["value"])
    duplicate_source_conflict = any(
        len({value for value in values}) > 1 for values in by_source.values()
    )
    unique_values: list[tuple[str, Decimal]] = [
        (family, values[0])
        for family, values in by_source.items()
        if values
    ]
    if duplicate_source_conflict:
        return {
            "status": "data_conflict",
            "data_conflict": True,
            "reason_code": "duplicate_source_conflict",
            "reason": "同一来源在同一日期/窗口/字段返回互相冲突的值",
            "raw_values": raw_values,
            "source_count": len(unique_values),
            "source_family_count": len(source_families),
            "source_families": {family: sorted(sources) for family, sources in source_families.items()},
            "source_values": {family: _decimal_text(values[0]) for family, values in by_source.items()},
            "direction": "blocked",
            "direction_allowed": False,
        }
    if len(unique_values) < 2:
        return {
            "status": "data_conflict",
            "data_conflict": True,
            "reason_code": "insufficient_sources",
            "reason": "新算法组有效且可比来源不足，无法形成共识",
            "raw_values": raw_values,
            "source_count": len(unique_values),
            "source_family_count": len(source_families),
            "source_families": {family: sorted(sources) for family, sources in source_families.items()},
            "source_values": {source: _decimal_text(value) for source, value in unique_values},
            "direction": "blocked",
            "direction_allowed": False,
        }

    working = list(unique_values)
    outliers: list[dict[str, Any]] = []
    median, mad, relative = _dispersion([value for _, value in working])
    # A source is called an outlier only when the remaining >=2 sources form a
    # low-dispersion cluster. Otherwise the disagreement remains unexplained.
    if len(working) >= 3:
        candidate_inliers = [
            (source, value)
            for source, value in working
            if abs(value - median)
            <= max(Decimal("3") * mad, relative_dispersion_threshold * max(abs(median), _EPSILON))
        ]
        if len(candidate_inliers) >= 2:
            candidate_median, candidate_mad, candidate_relative = _dispersion(
                [value for _, value in candidate_inliers]
            )
            if candidate_relative <= relative_dispersion_threshold and len(candidate_inliers) < len(working):
                outlier_sources = {source for source, _ in working} - {
                    source for source, _ in candidate_inliers
                }
                outliers = [item for item in raw_values if item["source"] in outlier_sources]
                working = candidate_inliers
                median, mad, relative = candidate_median, candidate_mad, candidate_relative

    if relative > relative_dispersion_threshold:
        return {
            "status": "data_conflict",
            "data_conflict": True,
            "reason_code": "unexplained_dispersion",
            "reason": "新算法组同字段已对齐，但离散度超过阈值且无法解释",
            "raw_values": raw_values,
            "source_count": len(unique_values),
            "contributing_sources": [source for source, _ in working],
            "source_values": {source: _decimal_text(value) for source, value in unique_values},
            "median": _decimal_text(median),
            "mad": _decimal_text(mad),
            "relative_dispersion": _decimal_text(relative),
            "relative_dispersion_threshold": _decimal_text(relative_dispersion_threshold),
            "outliers": outliers,
            "direction": "blocked",
            "direction_allowed": False,
        }

    direction = "neutral"
    if field := observations[0].get("field"):
        if field == "r0_out":
            # r0_out is an amount flowing out: a positive consensus is outflow.
            if median > 0:
                direction = "outflow"
            elif median < 0:
                direction = "inflow"
        elif median > 0:
            direction = "inflow"
        elif median < 0:
            direction = "outflow"
    elif median > 0:
        direction = "inflow"
    elif median < 0:
        direction = "outflow"
    return {
        "status": "consensus",
        "data_conflict": False,
        "reason_code": "low_dispersion_consensus",
        "reason": "新算法组同字段、同日期/窗口已对齐并形成低离散度共识",
        "raw_values": raw_values,
        "source_count": len(unique_values),
        "source_family_count": len(source_families),
        "source_families": {family: sorted(sources) for family, sources in source_families.items()},
        "contributing_sources": [source for source, _ in working],
        "source_values": {source: _decimal_text(value) for source, value in unique_values},
        "median": _decimal_text(median),
        "mad": _decimal_text(mad),
        "relative_dispersion": _decimal_text(relative),
        "relative_dispersion_threshold": _decimal_text(relative_dispersion_threshold),
        "outliers": outliers,
        "consensus_value": _decimal_text(median),
        "direction": direction,
        "direction_allowed": True,
    }


def _direction_for_field(field: str, value: Decimal) -> str:
    """Map a signed amount to a flow direction using the field's semantics."""
    if field == "r0_out":
        # r0_out is an outflow amount, unlike net fields where a negative value
        # denotes outflow.
        return "outflow" if value > 0 else "inflow" if value < 0 else "neutral"
    return "inflow" if value > 0 else "outflow" if value < 0 else "neutral"


def _aggregate_daily_field_results(
    field: str,
    grouped: list[tuple[tuple[Any, ...], list[dict[str, Any]]]],
    *,
    relative_dispersion_threshold: Decimal,
    max_days: int = DEFAULT_WINDOW_DAYS,
) -> dict[str, Any]:
    """Consensus each date first, then aggregate the latest daily values."""
    by_date: dict[str, list[tuple[tuple[Any, ...], list[dict[str, Any]]]]] = {}
    for key, observations in grouped:
        by_date.setdefault(str(key[1]), []).append((key, observations))
    dates = sorted(by_date)[-max_days:]
    daily_results: dict[str, Any] = {}
    all_raw_values: list[dict[str, Any]] = []
    for date in dates:
        date_groups = by_date[date]
        if len(date_groups) != 1:
            daily_results[date] = {
                "status": "data_conflict",
                "data_conflict": True,
                "reason_code": "incomparable_alignment",
                "reason": "同一日期存在不同时间窗口、单位或字段记录，无法形成日共识",
                "direction": "blocked",
                "direction_allowed": False,
                "raw_values": [
                    {
                        "source": item["source"],
                        "value": item["value_text"],
                        "date": item["date"],
                        "period_kind": item["period_kind"],
                        "time_window": item["time_window"],
                        "field": item["field"],
                    }
                    for _, source_group in date_groups for item in source_group
                ],
            }
        else:
            daily_results[date] = _evaluate_observation_group(
                date_groups[0][1],
                relative_dispersion_threshold=relative_dispersion_threshold,
            )
        all_raw_values.extend(daily_results[date].get("raw_values", []))

    if not daily_results:
        return {
            "status": "data_conflict",
            "data_conflict": True,
            "reason_code": "no_daily_observation",
            "reason": "没有可用于日共识的记录",
            "direction": "blocked",
            "direction_allowed": False,
            "daily_consensus": {},
        }
    blocked_dates = [date for date, result in daily_results.items() if result.get("status") != "consensus"]
    if blocked_dates:
        return {
            "status": "data_conflict",
            "data_conflict": True,
            "reason_code": "daily_consensus_conflict",
            "reason": "至少一个交易日的新算法组无法形成可解释共识",
            "dates": dates,
            "blocked_dates": blocked_dates,
            "daily_consensus": daily_results,
            "raw_values": all_raw_values,
            "direction": "blocked",
            "direction_allowed": False,
        }

    daily_values = [
        decimal_value(result.get("consensus_value"))
        for result in daily_results.values()
    ]
    if any(value is None for value in daily_values):
        return {
            "status": "data_conflict",
            "data_conflict": True,
            "reason_code": "daily_value_missing",
            "reason": "日共识缺少可聚合的数值",
            "dates": dates,
            "daily_consensus": daily_results,
            "raw_values": all_raw_values,
            "direction": "blocked",
            "direction_allowed": False,
        }
    aggregate = sum((value for value in daily_values if value is not None), Decimal("0"))
    daily_mads = [
        decimal_value(result.get("mad")) or Decimal("0")
        for result in daily_results.values()
    ]
    aggregate_mad = _median(daily_mads) if daily_mads else Decimal("0")
    relative = aggregate_mad if abs(aggregate) <= _EPSILON else aggregate_mad / abs(aggregate)
    direction = _direction_for_field(field, aggregate)
    return {
        "status": "consensus",
        "data_conflict": False,
        "reason_code": "daily_consensus_then_window_aggregate",
        "reason": "新算法组按交易日分别共识后聚合最新交易日窗口",
        "dates": dates,
        "window_days": len(dates),
        "period_kind": "five_day_aggregate" if len(dates) > 1 else "daily_consensus",
        "daily_consensus": daily_results,
        "raw_values": all_raw_values,
        "consensus_value": _decimal_text(aggregate),
        "aggregate_value": _decimal_text(aggregate),
        "median": _decimal_text(aggregate),
        "mad": _decimal_text(aggregate_mad),
        "relative_dispersion": _decimal_text(relative),
        "relative_dispersion_threshold": _decimal_text(relative_dispersion_threshold),
        "direction": direction,
        "direction_allowed": True,
        "contributing_sources": sorted({
            source
            for result in daily_results.values()
            for source in result.get("contributing_sources", [])
        }),
        "outliers": [
            item
            for result in daily_results.values()
            for item in result.get("outliers", [])
        ],
    }


def _build_consensus_evidence_legacy(
    records: Iterable[Mapping[str, Any]],
    *,
    symbol: str | None = None,
    requested_as_of: str | None = None,
    field: str | None = None,
    relative_dispersion_threshold: Decimal = DEFAULT_RELATIVE_DISPERSION,
) -> dict[str, Any]:
    """Align new-algorithm sources and calculate a guarded median/MAD consensus.

    Legacy Sina Web observations are retained in ``legacy_sources`` but never
    enter the median, MAD, outlier detection, or direction decision. A result
    marked ``data_conflict`` must not be used for buy/sell/accumulation text.
    """
    observations: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, Mapping):
            observations.extend(_record_observations(record))
    if symbol:
        observations = [item for item in observations if item["symbol"] in {None, symbol}]
    if requested_as_of:
        # requested_as_of is a filter only when a measurement date is present;
        # missing dates remain conflicts rather than being silently backfilled.
        observations = [
            item for item in observations
            if item["date"] is None or str(item["date"]) <= str(requested_as_of)
        ]

    new_observations = [
        item
        for item in observations
        if item["algorithm_group"] == NEW_ALGORITHM_GROUP
        and item["record"].get("automated_consensus_eligible", True)
        and item["record"].get("status") == "available"
    ]
    legacy_observations = [
        item for item in observations if item["algorithm_group"] == LEGACY_WEB_ALGORITHM
    ]
    unknown_observations = [
        item for item in observations if item["algorithm_group"] == UNKNOWN_ALGORITHM_GROUP
    ]
    raw_all = [
        {
            "source": item["source"],
            "algorithm_group": item["algorithm_group"],
            "field": item["field"],
            "value": item["value_text"],
            "raw_value": item["raw_value"],
            "raw_unit": item["raw_unit"],
            "unit": item["unit"],
            "date": item["date"],
            "period_kind": item["period_kind"],
            "time_window": item["time_window"],
        }
        for item in observations
    ]
    base: dict[str, Any] = {
        "algorithm_group": NEW_ALGORITHM_GROUP,
        "group_priority": "new_algorithm_over_legacy_web",
        "status": "data_conflict",
        "data_conflict": True,
        "direction": "blocked",
        "direction_allowed": False,
        "raw_values": raw_all,
        "legacy_sources": [item for item in raw_all if item["algorithm_group"] == LEGACY_WEB_ALGORITHM],
        "unknown_sources": [item for item in raw_all if item["algorithm_group"] == UNKNOWN_ALGORITHM_GROUP],
        "relative_dispersion_threshold": _decimal_text(relative_dispersion_threshold),
        "field_results": {},
    }
    if not new_observations:
        base.update({
            "reason_code": "no_new_algorithm_source",
            "reason": "没有可用于主结论的新算法组来源；legacy Web 仅作旁证",
        })
        return base

    missing_alignment = [
        item for item in new_observations
        if not item["symbol"] or not item["date"] or not item["period_kind"]
        or not item["time_window"] or not item["field"]
    ]
    if missing_alignment:
        base.update({
            "reason_code": "missing_alignment_key",
            "reason": "新算法组来源缺少股票、日期、时间窗口或字段分类，禁止跨源比较",
            "raw_values": raw_all,
        })
        return base

    # Compare each field only against the same field. Date, timestamp/window,
    # unit, and field category are part of the key, so cross-window fill is impossible.
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in new_observations:
        if field and item["field"] != field:
            continue
        key = (
            item["symbol"],
            item["date"],
            item["measurement_time"],
            item["period_kind"],
            item["time_window"],
            item["field"],
            item["field_category"],
            item["unit"],
        )
        groups.setdefault(key, []).append(item)

    if not groups:
        base.update({
            "reason_code": "incomparable_fields",
            "reason": "新算法组来源的字段不可比，未进行跨字段平均或回填",
        })
        return base

    source_keys = {
        (item["source"], item["field"], item["date"], item["period_kind"], item["time_window"], item["unit"])
        for item in new_observations
    }
    # Different official fields (for example ``netamount`` vs ``r0_net``) are
    # retained in separate field results and are never averaged together.
    grouped_by_field: dict[str, list[tuple[tuple[Any, ...], list[dict[str, Any]]]]] = {}
    for key, group in groups.items():
        grouped_by_field.setdefault(str(key[5]), []).append((key, group))
    # Prefer the main-force net field for a direction conclusion, then retain
    # all other field results for auditability.
    preferred_fields = [field] if field else list(_FIELD_ORDER)
    preferred_fields = [name for name in preferred_fields if name]
    selected_field = next(
        (name for name in preferred_fields if name in grouped_by_field),
        next(iter(grouped_by_field)),
    )
    field_results: dict[str, Any] = {}
    for field_name, field_groups in grouped_by_field.items():
        field_results[field_name] = _aggregate_daily_field_results(
            field_name,
            field_groups,
            relative_dispersion_threshold=relative_dispersion_threshold,
            max_days=DEFAULT_WINDOW_DAYS,
        )
    selected = field_results[selected_field]
    base.update(selected)
    base["field"] = selected_field
    base["field_category"] = _FIELD_CATEGORIES.get(selected_field, "main_force_component")
    base["field_results"] = field_results
    base["new_algorithm_sources"] = [
        item for item in raw_all if item["algorithm_group"] == NEW_ALGORITHM_GROUP
    ]
    base["legacy_web_is_corroboration_only"] = bool(legacy_observations)
    if base.get("status") == "consensus" and selected_field not in _MAIN_FORCE_FIELDS:
        # A total-net or order-component consensus is valid evidence but cannot
        # be labelled as a main-force buy/sell/accumulation direction.
        base["direction"] = "blocked"
        base["direction_allowed"] = False
        base["reason_code"] = "non_main_force_direction"
        base["reason"] = "仅形成非主力 r0 净额共识，不能替代主力口径方向"
    base["hard_guard"] = {
        "blocked": not bool(base.get("direction_allowed")) or base.get("status") != "consensus",
        "direction_allowed": bool(base.get("direction_allowed")) and base.get("status") == "consensus",
        "reason": base.get("reason") or "consensus available",
    }
    if base.get("direction") == "outflow":
        base["direction_summary"] = "主力偏减持/大额资金偏流出"
    elif base.get("direction") == "inflow":
        base["direction_summary"] = "主力偏增持/大额资金偏流入"
    elif base.get("direction") == "neutral":
        base["direction_summary"] = "主力资金接近平衡"
    else:
        base["direction_summary"] = "方向结论已阻断"
    return base


def build_consensus_evidence(
    records: Iterable[Mapping[str, Any]],
    *,
    symbol: str | None = None,
    requested_as_of: str | None = None,
    field: str | None = None,
    relative_dispersion_threshold: Decimal = DEFAULT_RELATIVE_DISPERSION,
) -> dict[str, Any]:
    """Compatibility envelope for deterministic priority selection.

    The public name remains for callers and serialized reports, but the
    direction gate no longer depends on a median/MAD consensus or source count.
    ``relative_dispersion_threshold`` is accepted for API compatibility and is
    retained in the result as audit metadata.
    """
    result = select_fund_flow_source(
        records,
        symbol=symbol,
        requested_as_of=requested_as_of,
        field=field,
    )
    result["relative_dispersion_threshold"] = _decimal_text(relative_dispersion_threshold)
    return result


# Public aliases keep callers concise and make the source-selection contract
# discoverable without changing provider-facing legacy names.
summarize_source_consensus = build_consensus_evidence
build_consensus = build_consensus_evidence


def consensus_prompt_instruction(consensus: Mapping[str, Any] | None) -> str:
    """Render source-selection rules for the smart-money analyst prompt."""
    if not isinstance(consensus, Mapping):
        return (
            "未提供可验证资金流来源；不得跨日期、窗口、字段或单位回填，"
            "也不得把未经选择的旁证值写成方向结论。"
        )
    if consensus.get("direction_allowed"):
        source = consensus.get("selected_source") or consensus.get("source") or "unknown"
        field = consensus.get("selected_field") or consensus.get("field") or "unknown"
        rank = consensus.get("fallback_rank")
        legacy_note = (
            "该值来自 Sina legacy Web，方向只能按其自身字段展示，并必须醒目标注 legacy/旧算法/仅供参考。"
            if consensus.get("legacy_reference") or consensus.get("legacy_web_algorithm")
            else "较低优先级来源只能作为旁证，不得改写已选来源方向。"
        )
        return (
            f"已按固定优先级选择来源 {source}（fallback_rank={rank}）、字段 {field}；"
            "单一字段/日期/单位合格即可允许该来源自身方向，不要求多源共识。"
            f"{legacy_note} 不得跨字段、日期、窗口或单位混算。"
        )
    return (
        "没有字段、日期、单位和有效值均合格的来源；必须保留失败链和各源原值，"
        "禁止输出增持、减持、吸筹等方向摘要。"
    )


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
    selected_source: str | None = None,
    selected_field: str | None = None,
    requested_as_of: str | None = None,
) -> dict[str, Any]:
    """Validate model totals against the selected source, never side evidence."""
    structured = summarize_evidence(
        records,
        window_days=window_days,
        selected_source=selected_source,
        selected_field=selected_field,
        requested_as_of=requested_as_of,
    )
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
    if structured.get("status") in {"data_conflict", "partial"}:
        status = "blocked"
    return {
        "status": status,
        "hard_guard": {
            "blocked": status not in {"matched", "not_checked"},
            "reason": "模型累计值与结构化 evidence 不一致或结构化窗口不可用"
            if status == "blocked" or status == "mismatch"
            else "no explicit model total",
        },
        "structured": structured,
        "model": model,
        "mismatches": mismatches,
        "tolerance": _decimal_text(tolerance),
    }
