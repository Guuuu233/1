"""Structured fund-flow evidence, source alignment, and arithmetic checks."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, DecimalException, InvalidOperation
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
            "dongfangcaifu",
            "东方财富",
            "ths",
            "moneyflow_dc",
            "moneyflow_ths",
            "tushare",
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
    if any(token in label for token in ("eastmoney", "em_", "_em", "东方财富", "dongfangcaifu")):
        return "eastmoney"
    if any(token in label for token in ("ths", "tonghuashun", "同花顺")):
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
    try:
        converted = number * multiplier
    except (DecimalException, OverflowError, ValueError):
        return None, source_unit
    if not converted.is_finite():
        return None, source_unit
    return converted, source_unit


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
    # Keep the calendar date only so equivalent source rows align. Tushare also
    # uses compact ``YYYYMMDD`` dates, which must compare as calendar values
    # rather than lexicographic strings (especially for future-date guards).
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        text = text[:10]
    for date_format in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            continue
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
        as_of = str(row.get("as_of") or date or "").strip() or None
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
            "r0_net_raw": _decimal_text(r0_net_raw),
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
        "gap": f"【数据获取失败】资金流 evidence：{reason}",
        "retrieved_at": retrieved_at,
        "actual_as_of": actual_as_of,
        "as_of": actual_as_of,
    }


def _sum_field(records: Iterable[Mapping[str, Any]], field: str) -> Decimal | None:
    values = [decimal_value(record.get(field)) for record in records]
    usable = [value for value in values if value is not None]
    return sum(usable, Decimal("0")) if usable else None


def _normalise_summary_record(
    record: Mapping[str, Any],
    field: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize raw amounts and dates before any cumulative arithmetic."""
    date = _date_value(record)
    if date is None:
        return None, "unparseable_date"
    normalized = dict(record)
    normalized["date"] = date
    normalized["period_kind"] = _observation_period(record)
    normalized["window"] = _observation_window(record, normalized["period_kind"])
    fields = [field] if field else ("netamount", "r0_net", "r0_in", "r0_out", "r0")
    if field and field not in record:
        explicit = _canonical_field(record.get("field") or record.get("字段"))
        if explicit == field and record.get("value") is not None:
            normalized[field] = record.get("value")
            normalized[f"{field}_unit"] = record.get("value_unit") or record.get("unit") or record.get("raw_unit")
    for candidate in fields:
        if candidate not in normalized or normalized.get(candidate) is None:
            continue
        raw_value = record.get(
            f"{candidate}_raw",
            normalized.get(candidate),
        )
        raw_unit = record.get(
            f"{candidate}_raw_unit",
            record.get(
                f"{candidate}_unit",
                normalized.get(f"{candidate}_unit", record.get("raw_unit", record.get("unit"))),
            ),
        )
        amount, parsed_unit = _amount_to_yi(raw_value, raw_unit)
        if amount is None:
            return None, f"invalid_{candidate}_unit_or_amount"
        normalized[candidate] = _decimal_text(amount)
        normalized[f"{candidate}_raw_unit"] = parsed_unit
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
    field: str | None = None,
    source: str | None = None,
    requested_as_of: str | None = None,
) -> dict[str, Any]:
    """Compute exact daily totals without mixing sources, units, or windows."""
    usable: list[dict[str, Any]] = []
    invalid_records: list[str] = []
    requested_date = _normalise_date_text(requested_as_of) if requested_as_of else None
    for record in records:
        if not isinstance(record, Mapping) or record.get("status") != "available":
            continue
        if source is not None and str(record.get("source") or "") != str(source):
            continue
        normalized, error = _normalise_summary_record(record, field=field)
        if normalized is None:
            invalid_records.append(error or "invalid_record")
        elif requested_date and str(normalized.get("date")) > requested_date:
            # Future rows are rejected from arithmetic rather than silently
            # becoming the selected as-of for a historical report.
            continue
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
    if field is not None:
        canonical_field = _canonical_field(field) or str(field)
        netamount = _sum_field(selected, "netamount") if canonical_field == "netamount" else None
        r0_net = _sum_field(selected, "r0_net") if canonical_field == "r0_net" else None
        selected_value = _sum_field(selected, canonical_field)
        required_values_available = selected_value is not None
    else:
        canonical_field = None
        selected_value = None
        netamount = _sum_field(selected, "netamount")
        r0_net = _sum_field(selected, "r0_net")
        required_values_available = netamount is not None and r0_net is not None
    status = (
        "available"
        if len(selected) == window_days and required_values_available
        else "partial"
    )
    result = {
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
    if canonical_field is not None:
        result["selected_field"] = canonical_field
        result["selected_value"] = _decimal_text(selected_value)
        result["selected_semantics"] = _FIELD_SEMANTICS.get(canonical_field)
    return result


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
        normalized = str(period).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"5d", "5_day", "five_day", "five_days", "five_day_total"}:
            return "five_day_cumulative"
        if normalized in {"daily", "day", "1d", "one_day"}:
            return "historical_daily"
        if normalized in {"realtime", "intraday", "intraday_snapshot", "snapshot"}:
            return "realtime_single_day"
        return normalized
    if record.get("realtime") is True or record.get("is_realtime") is True:
        return "realtime_single_day"
    return "historical_daily"


def _observation_window(record: Mapping[str, Any], period: str) -> str:
    value = record.get("time_window") or record.get("window")
    if value:
        return _normalise_window(value)
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


# Lower numbers are higher priority.  The order is deliberately explicit so
# adding a provider cannot silently change which source drives a direction.
_SOURCE_PRIORITY = {
    "eastmoney_direct": 1,
    "tushare_eastmoney_moneyflow_dc": 2,
    "tushare.moneyflow_dc": 2,
    "moneyflow_dc": 2,
    "eastmoney_individual_fund_flow": 3,
    "tushare_ths_moneyflow_ths": 4,
    "tushare.moneyflow_ths": 4,
    "moneyflow_ths": 4,
    "ths_instant_snapshot": 5,
}


def _source_priority(source: Any, algorithm_group: str) -> int:
    """Return a stable source rank while keeping legacy last."""
    label = _normalise_source_text(source)
    for name, rank in _SOURCE_PRIORITY.items():
        if label == name or label.endswith(f"_{name}") or name in label:
            return rank
    if algorithm_group == LEGACY_WEB_ALGORITHM:
        return 7
    if algorithm_group == NEW_ALGORITHM_GROUP:
        return 6
    return 99


def _normalise_window(value: Any) -> str:
    text = str(value or "1d").strip().lower().replace(" ", "")
    return {
        "1day": "1d",
        "1_day": "1d",
        "daily": "1d",
        "day": "1d",
        "5day": "5d",
        "5_day": "5d",
    }.get(text, text)


def _field_semantics_are_valid(record: Mapping[str, Any], field: str) -> bool:
    """Reject an explicitly mislabeled field, but accept canonical fields."""
    semantics = record.get("field_semantics")
    explicit = semantics.get(field) if isinstance(semantics, Mapping) else None
    if explicit is None:
        explicit = record.get(f"{field}_semantics")
    if explicit is None:
        # Direction selection requires an auditable semantic declaration; raw
        # field names alone are not proof that a vendor uses the same metric.
        explicit = record.get("upstream_field_semantics")
        if explicit is None:
            return False
    text = str(explicit).strip().lower()
    if not text:
        return False
    if field == "netamount":
        if any(token in text for token in ("净利润", "净资产", "净负债", "净利")):
            return False
        return (
            "净" in text
            and ("资金" in text or "总" in text)
            and ("流入" in text or "流出" in text or "净额" in text)
            and "主力" not in text
        )
    if field == "r0_net":
        return ("主力" in text or "大单" in text) and "净" in text and (
            "流入" in text or "流出" in text or "净额" in text
        )
    if field == "r0_in":
        return "主力" in text and "流入" in text and "净" not in text
    if field == "r0_out":
        return "主力" in text and "流出" in text and "净" not in text
    if field == "r0":
        return "主力" in text and ("资金" in text or "值" in text)
    return False


def _selection_fields(
    record: Mapping[str, Any], field: str | None
) -> list[tuple[str, Any, str | None]]:
    """Extract canonical candidate fields without combining their semantics."""
    requested_field = _canonical_field(field) if field else None
    explicit_field = _canonical_field(record.get("field") or record.get("字段"))
    if explicit_field is not None and record.get("value") is not None:
        if requested_field and explicit_field != requested_field:
            return []
        return [
            (
                explicit_field,
                record.get("value"),
                record.get("value_unit")
                or record.get(f"{explicit_field}_unit")
                or record.get("unit")
                or record.get("raw_unit"),
            )
        ]
    if field and requested_field is None:
        return []
    ordered = [requested_field] if requested_field else list(_FIELD_ORDER)
    candidates: list[tuple[str, Any, str | None]] = []
    for candidate in ordered:
        if not candidate:
            continue
        value, _ = _row_value(record, candidate)
        if value is not None:
            # ``unit`` describes normalized provider values. A field-specific
            # unit is only used when one was explicitly supplied for that value.
            value_unit = record.get(f"{candidate}_unit") or record.get("unit") or record.get("raw_unit")
            candidates.append((candidate, value, value_unit))
    return candidates


def _selection_candidate(
    record: Mapping[str, Any],
    *,
    field: str | None,
    symbol: str | None,
    requested_as_of: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one structured record for source-priority selection."""
    if not isinstance(record, Mapping):
        return None, "record_not_object"
    source = str(record.get("source") or "").strip()
    if not source:
        return None, "source_missing"
    group = infer_algorithm_group(source, record.get("algorithm_group"))
    if group not in {NEW_ALGORITHM_GROUP, LEGACY_WEB_ALGORITHM}:
        return None, "algorithm_group_unknown"
    if record.get("status") != "available":
        return None, f"status_{record.get('status') or 'missing'}"
    if record.get("automated_consensus_eligible", True) is False or record.get("manual_calibration"):
        return None, "manual_or_ineligible"
    record_symbol = str(record.get("symbol") or "").strip()
    if symbol and record_symbol.upper() != str(symbol).strip().upper():
        return None, "symbol_mismatch"
    if not record_symbol:
        return None, "symbol_missing"
    measurement_keys = (
        "measurement_date",
        "date",
        "日期",
        "opendate",
        "trade_date",
        "交易日期",
    )
    if not any(record.get(key) not in (None, "") for key in measurement_keys):
        return None, "date_missing_or_invalid"
    date = _date_value(record)
    if not date:
        return None, "date_missing_or_invalid"
    requested = _normalise_date_text(requested_as_of) if requested_as_of else None
    if requested_as_of and not requested:
        return None, "requested_as_of_invalid"
    if requested and date > requested:
        return None, "future_date"
    source_label = _normalise_source_text(source)
    if requested and (
        source_label.startswith("tushare")
        or source_label in {"moneyflow_dc", "moneyflow_ths"}
        or source_label == "ths_instant_snapshot"
    ) and date != requested:
        return None, "date_mismatch"
    period = _observation_period(record)
    window = _normalise_window(_observation_window(record, period))
    if period not in {"historical_daily", "realtime_single_day"}:
        return None, "period_not_daily"
    if window != "1d":
        return None, "window_not_daily"

    candidates: list[dict[str, Any]] = []
    for candidate_field, value, value_unit in _selection_fields(record, field):
        if candidate_field not in _FIELD_SEMANTICS:
            # Components remain in raw evidence, but cannot independently
            # authorize a main-force direction without a canonical net field.
            continue
        if not _field_semantics_are_valid(record, candidate_field):
            continue
        amount, parsed_unit = _amount_to_yi(value, value_unit)
        if amount is None:
            continue
        candidates.append(
            {
                "source": source,
                "source_family": source_family(source),
                "algorithm_group": group,
                "requested_as_of": requested,
                "legacy_web_algorithm": group == LEGACY_WEB_ALGORITHM,
                "symbol": record_symbol,
                "date": date,
                "period_kind": period,
                "time_window": window,
                "field": candidate_field,
                "field_category": _FIELD_CATEGORIES.get(candidate_field, "main_force_component"),
                "value": amount,
                "value_text": _decimal_text(amount),
                "raw_value": str(record.get(f"{candidate_field}_raw", value)),
                "raw_unit": (
                    record.get(f"{candidate_field}_raw_unit")
                    or record.get("raw_unit")
                    or parsed_unit
                ),
                "unit": "亿元",
                "record": dict(record),
            }
        )
    if not candidates:
        # Distinguish an invalid field/unit from a structurally valid record so
        # the rejected-source chain remains useful to operators.
        return None, "field_semantics_or_value_invalid"
    candidate_names = {candidate["field"] for candidate in candidates}
    if {"r0_in", "r0_out"}.issubset(candidate_names) and not (
        {"r0_net", "r0", "netamount"} & candidate_names
    ):
        return None, "inflow_outflow_not_net"
    return candidates[0], None


def _selection_group_summary(
    candidates: list[dict[str, Any]],
    *,
    rank: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one source/field group and select requested/latest daily row."""
    if not candidates:
        return None, "no_candidates"
    source = candidates[0]["source"]
    field = candidates[0]["field"]
    periods = {item["period_kind"] for item in candidates}
    windows = {item["time_window"] for item in candidates}
    if len(periods) != 1 or len(windows) != 1:
        return None, "mixed_period_or_window"
    by_date: dict[str, dict[str, Any]] = {}
    for item in candidates:
        previous = by_date.get(item["date"])
        if previous is not None and previous["value"] != item["value"]:
            return None, "duplicate_date_conflict"
        by_date[item["date"]] = item
    ordered = [by_date[key] for key in sorted(by_date)][-DEFAULT_WINDOW_DAYS:]
    if candidates[0]["period_kind"] == "realtime_single_day" and len(ordered) != 1:
        return None, "realtime_multiple_dates"

    selected = ordered[-1]
    day_value = selected["value"]
    day_direction = _direction_for_field(field, day_value)

    five_day_summary = None
    if len(ordered) > 1:
        try:
            cum_val = sum((item["value"] for item in ordered), Decimal("0"))
            five_day_summary = {
                "value": _decimal_text(cum_val),
                "direction": _direction_for_field(field, cum_val),
                "window_days": len(ordered),
                "time_window": f"{len(ordered)}d" if len(ordered) != 5 else "5d",
                "period_kind": "five_day_aggregate" if len(ordered) == 5 else f"{len(ordered)}_day_aggregate",
                "dates": [item["date"] for item in ordered],
                "field": field,
                "unit": "亿元",
            }
        except (DecimalException, OverflowError, ValueError):
            five_day_summary = None

    selected_records = [
        {
            "source": item["source"],
            "source_family": item["source_family"],
            "algorithm_group": item["algorithm_group"],
            "field": item["field"],
            "value": item["value_text"],
            "raw_value": item["raw_value"],
            "raw_unit": item["raw_unit"],
            "unit": item["unit"],
            "date": item["date"],
            "as_of": item["date"],
            "requested_as_of": item["record"].get("requested_as_of"),
            "retrieved_at": item["record"].get("retrieved_at"),
            "symbol": item["symbol"],
            "status": "available",
            "period_kind": item["period_kind"],
            "time_window": item["time_window"],
        }
        for item in ordered
    ]
    summary_data = {
        "source": source,
        "source_family": selected["source_family"],
        "algorithm_group": selected["algorithm_group"],
        "legacy_web_algorithm": bool(selected["legacy_web_algorithm"]),
        "field": field,
        "value": _decimal_text(day_value),
        "direction": day_direction,
        "fallback_rank": rank,
        "as_of": selected["date"],
        "window_days": 1,
        "period_kind": selected["period_kind"],
        "time_window": "1d",
        "records": selected_records,
        "source_values": {item["date"]: item["value_text"] for item in ordered},
    }
    if five_day_summary is not None:
        summary_data["five_day_summary"] = five_day_summary
        summary_data["summary_5d"] = five_day_summary
    return summary_data, None


def score_large_order_reference_credibility(
    records: Iterable[Mapping[str, Any]],
    price_change: Decimal | float | str | None = None,
) -> dict[str, Any]:
    """Score large-order / main-force reference credibility across multiple sources.

    Platform main force is a statistical large-order proxy, not account-level truth.
    - >=2 sources concordant in direction -> high credibility (偏高)
    - single source -> medium-low credibility (中等偏低) + single_source=True
    - >=2 sources divergent in direction -> low credibility (偏低) + divergence=True + divergence details
    - weak weighting with price change (optional, never absolute)
    """
    r0_items: list[Mapping[str, Any]] = []
    for r in records or ():
        if not isinstance(r, Mapping):
            continue
        val = r.get("r0_net")
        if val is None and r.get("field") == "r0_net":
            val = r.get("value")
        if val is not None and decimal_value(val) is not None:
            r0_items.append(r)

    sources_seen: set[str] = set()
    unique_sources: list[dict[str, Any]] = []
    for item in r0_items:
        src = str(item.get("source") or item.get("source_family") or "unknown")
        val = decimal_value(item.get("r0_net") if item.get("r0_net") is not None else item.get("value"))
        direction = item.get("direction")
        if direction not in {"inflow", "outflow", "neutral"}:
            if val is not None:
                direction = "inflow" if val > 0 else ("outflow" if val < 0 else "neutral")
            else:
                direction = "neutral"
        if src not in sources_seen:
            sources_seen.add(src)
            unique_sources.append({
                "source": src,
                "source_family": item.get("source_family"),
                "value": _decimal_text(val),
                "direction": direction,
                "raw_item": item,
            })

    source_count = len(unique_sources)
    source_directions = {s["source"]: s["direction"] for s in unique_sources}
    source_values = {s["source"]: s["value"] for s in unique_sources}

    pc_val = decimal_value(price_change) if price_change is not None else None

    if source_count == 0:
        return {
            "credibility": "low",
            "credibility_score": 0.20,
            "credibility_level": "偏低",
            "credibility_reason": "无有效大单/主力统计口径数据，仅作旁证参考",
            "single_source": True,
            "divergence": False,
            "reference_only": True,
            "source_count": 0,
            "source_directions": {},
            "source_values": {},
            "price_aligned": None,
        }

    if source_count == 1:
        base_score = Decimal("0.50")
        reason = "单数据源提供大单/主力统计口径，缺乏交叉印证，参考可信度中等偏低"
        aligned = None
        s0_dir = unique_sources[0]["direction"]
        if pc_val is not None and pc_val != 0:
            if (s0_dir == "inflow" and pc_val > 0) or (s0_dir == "outflow" and pc_val < 0):
                base_score += Decimal("0.05")
                reason += "；与当日行情同向弱共振"
                aligned = True
            else:
                base_score -= Decimal("0.05")
                reason += "；与当日行情存在背离"
                aligned = False
        return {
            "credibility": "medium_low",
            "credibility_score": float(base_score),
            "credibility_level": "中等偏低",
            "credibility_reason": reason,
            "single_source": True,
            "divergence": False,
            "reference_only": True,
            "source_count": 1,
            "source_directions": source_directions,
            "source_values": source_values,
            "price_aligned": aligned,
        }

    # >= 2 sources
    directions = [s["direction"] for s in unique_sources if s["direction"] in {"inflow", "outflow"}]
    has_inflow = "inflow" in directions
    has_outflow = "outflow" in directions

    if has_inflow and has_outflow:
        base_score = Decimal("0.30")
        details = "，".join(f"{s['source']}:{s['value']}亿({s['direction']})" for s in unique_sources)
        reason = f"多源大单/主力口径存在方向分歧（{details}），参考可信度偏低，方向仅作弱参考，请保留分源原值"
        return {
            "credibility": "low",
            "credibility_score": float(base_score),
            "credibility_level": "偏低",
            "credibility_reason": reason,
            "single_source": False,
            "divergence": True,
            "reference_only": True,
            "source_count": source_count,
            "source_directions": source_directions,
            "source_values": source_values,
            "price_aligned": None,
        }
    else:
        base_score = Decimal("0.80")
        common_dir = directions[0] if directions else "neutral"
        dir_label = "流入" if common_dir == "inflow" else ("流出" if common_dir == "outflow" else "平衡")
        reason = f"多源大单/主力统计口径同向{dir_label}，具较高参考价值（非绝对主力身份定性）"
        aligned = None
        if pc_val is not None and pc_val != 0:
            if (common_dir == "inflow" and pc_val > 0) or (common_dir == "outflow" and pc_val < 0):
                base_score += Decimal("0.05")
                reason += "；与当日行情变动同向弱共振"
                aligned = True
            else:
                base_score -= Decimal("0.05")
                reason += "；与当日行情走势背离，需结合量价综合参考"
                aligned = False
        return {
            "credibility": "high",
            "credibility_score": float(base_score),
            "credibility_level": "偏高",
            "credibility_reason": reason,
            "single_source": False,
            "divergence": False,
            "reference_only": True,
            "source_count": source_count,
            "source_directions": source_directions,
            "source_values": source_values,
            "price_aligned": aligned,
        }


def select_fund_flow_source(
    records: Iterable[Mapping[str, Any]],
    *,
    symbol: str | None = None,
    requested_as_of: str | None = None,
    field: str | None = None,
    price_change: Decimal | float | str | None = None,
) -> dict[str, Any]:
    """Select the first valid source instead of requiring multi-source consensus.

    Validation remains strict for date, period/window, field semantics, units,
    finite values, and symbol identity. New-algorithm sources are ranked
    deterministically; lower-ranked valid sources are retained as side evidence.
    Sina Web may provide its own direction only after every new source fails and
    is always marked as a legacy reference.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records or ():
        candidate, reason = _selection_candidate(
            record,
            field=field,
            symbol=symbol,
            requested_as_of=requested_as_of,
        )
        if candidate is None:
            source = str(record.get("source") or "unknown_source") if isinstance(record, Mapping) else "unknown_source"
            rejected.append({"source": source, "reason": reason or "invalid_record"})
        else:
            accepted.append(candidate)

    if not symbol and len({candidate["symbol"] for candidate in accepted}) > 1:
        rejected.extend(
            {"source": candidate["source"], "reason": "mixed_symbol_identity"}
            for candidate in accepted
        )
        accepted = []

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for candidate in accepted:
        grouped.setdefault(
            (
                candidate["source"],
                candidate["field"],
                candidate["period_kind"],
                candidate["time_window"],
                candidate["algorithm_group"],
            ),
            [],
        ).append(candidate)
    valid_groups: list[dict[str, Any]] = []
    for key, candidates in grouped.items():
        group = candidates[0]["algorithm_group"]
        rank = _source_priority(candidates[0]["source"], group)
        summary, reason = _selection_group_summary(candidates, rank=rank)
        if summary is None:
            rejected.append({"source": key[0], "field": key[1], "reason": reason or "invalid_source_group"})
        else:
            valid_groups.append(summary)

    field_rank = {name: index for index, name in enumerate(_FIELD_ORDER)}
    valid_groups.sort(
        key=lambda item: (
            item["fallback_rank"],
            -int(item["as_of"].replace("-", "")),
            field_rank.get(item["field"], len(field_rank)),
            item["source"],
        )
    )
    selected = valid_groups[0] if valid_groups else None
    legacy_groups = [item for item in valid_groups if item["legacy_web_algorithm"]]
    new_groups = [item for item in valid_groups if not item["legacy_web_algorithm"]]
    if selected is not None and new_groups:
        # A legacy group must never outrank a valid new source even if a future
        # provider accidentally reports a lower numeric rank.
        selected = new_groups[0]
    if selected is None and legacy_groups:
        selected = legacy_groups[0]

    raw_values = [
        {
            "source": item["source"],
            "source_family": item["source_family"],
            "algorithm_group": item["algorithm_group"],
            "field": item["field"],
            "value": item["value"],
            "direction": item["direction"],
            "date": item["as_of"],
            "period_kind": item["period_kind"],
            "time_window": item["time_window"],
        }
        for item in valid_groups
    ]
    if selected is None:
        result: dict[str, Any] = {
            "status": "blocked",
            "selection_status": "blocked",
            "data_conflict": False,
            "reason_code": "all_sources_unavailable",
            "selection_reason": "all_sources_unavailable",
            "reason": "所有新算法来源均失败、日期不匹配或字段/单位不可用",
            "requested_as_of": _normalise_date_text(requested_as_of) if requested_as_of else None,
            "algorithm_group": NEW_ALGORITHM_GROUP,
            "selected_source": None,
            "selected_source_family": None,
            "selected_algorithm_group": None,
            "selected_field": field,
            "selected_value": None,
            "selected_unit": "亿元",
            "selected_direction": "blocked",
            "fallback_rank": None,
            "legacy_reference": False,
            "legacy_web_algorithm": False,
            "direction": "blocked",
            "direction_allowed": False,
            "raw_values": raw_values,
            "alternative_sources": [],
            "rejected_sources": rejected,
            "hard_guard": {
                "blocked": True,
                "direction_allowed": False,
                "reason": "all_sources_unavailable",
            },
            "credibility": "none",
            "credibility_score": 0.0,
            "credibility_level": "无",
            "credibility_reason": "所有来源均不可用",
            "single_source": True,
            "divergence": False,
            "reference_only": True,
            "large_order_credibility": {},
        }
        return result

    is_legacy = bool(selected["legacy_web_algorithm"])
    new_fields = {item["field"] for item in new_groups}
    has_r0_net = "r0_net" in new_fields
    if has_r0_net and selected.get("field") == "r0_net":
        # Having r0_net means netamount acts as total funds side evidence (旁证)
        # It must not trigger incomparable_field_semantics to block r0_net direction
        non_side_fields = {f for f in new_fields if f != "netamount"}
        incomparable_fields = len(non_side_fields) >= 2
    else:
        incomparable_fields = len(new_fields) >= 2
    alternatives = [
        item
        for item in valid_groups
        if (
            item["source"],
            item["field"],
            item["period_kind"],
            item["time_window"],
        ) != (
            selected["source"],
            selected["field"],
            selected["period_kind"],
            selected["time_window"],
        )
    ]
    if incomparable_fields:
        selection_reason = "incomparable_field_semantics"
        reason = "新算法组存在多个不同字段同时有效，字段语义不可比，禁止放行方向"
        direction_allowed = False
        hard_guard_blocked = True
        status = "data_conflict"
        direction = "blocked"
        direction_summary = "不可比字段同时有效，方向结论已阻断"
    elif is_legacy:
        selection_reason = "no_new_algorithm_source_legacy_fallback"
        reason = "所有新算法来源不可用后使用新浪 legacy Web 参考值；仅展示该来源自身方向"
        direction_allowed = True
        hard_guard_blocked = False
        status = "selected"
        direction = selected["direction"]
        if selected["field"] == "netamount":
            label = "总资金（非主力口径）"
        else:
            label = "主力资金"
        if selected["direction"] == "outflow":
            direction_summary = f"{label}偏流出"
        elif selected["direction"] == "inflow":
            direction_summary = f"{label}偏流入"
        else:
            direction_summary = f"{label}接近平衡"
    else:
        selection_reason = "new_algorithm_source_priority"
        reason = "按固定来源优先级选择首个日期、字段、单位和数值均有效的来源"
        direction_allowed = True
        hard_guard_blocked = False
        status = "selected"
        direction = selected["direction"]
        if selected["field"] == "netamount":
            label = "总资金（非主力口径）"
        else:
            label = "主力资金"
        if selected["direction"] == "outflow":
            direction_summary = f"{label}偏流出"
        elif selected["direction"] == "inflow":
            direction_summary = f"{label}偏流入"
        else:
            direction_summary = f"{label}接近平衡"

    r0_candidates = [item for item in valid_groups if item.get("field") == "r0_net"]
    cred_info = score_large_order_reference_credibility(
        r0_candidates,
        price_change=price_change,
    )

    result = {
        "status": status,
        "selection_status": status,
        "data_conflict": incomparable_fields,
        "reason_code": selection_reason,
        "selection_reason": selection_reason,
        "reason": reason,
        "requested_as_of": _normalise_date_text(requested_as_of) if requested_as_of else None,
        "algorithm_group": selected["algorithm_group"],
        "selected_source": selected["source"],
        "selected_source_family": selected["source_family"],
        "selected_algorithm_group": selected["algorithm_group"],
        "selected_field": selected["field"],
        "selected_value": selected["value"],
        "selected_unit": "亿元",
        "selected_direction": selected["direction"],
        "selected_as_of": selected["as_of"],
        "selected_period_kind": selected["period_kind"],
        "selected_time_window": selected["time_window"],
        "selected_window_days": selected["window_days"],
        "fallback_rank": selected["fallback_rank"],
        "legacy_reference": is_legacy,
        "legacy_web_algorithm": is_legacy,
        "direction": direction,
        "direction_allowed": direction_allowed,
        "field": selected["field"],
        "value": selected["value"],
        "source": selected["source"],
        "source_family": selected["source_family"],
        "as_of": selected["as_of"],
        "records": selected["records"],
        "raw_values": raw_values,
        "alternative_sources": alternatives,
        "legacy_sources": [item for item in valid_groups if item["legacy_web_algorithm"]],
        "new_algorithm_sources": [item for item in valid_groups if not item["legacy_web_algorithm"]],
        "rejected_sources": rejected,
        "hard_guard": {
            "blocked": hard_guard_blocked,
            "direction_allowed": direction_allowed,
            "reason": selection_reason,
        },
        "direction_summary": direction_summary,
        "credibility": cred_info["credibility"],
        "credibility_score": cred_info["credibility_score"],
        "credibility_level": cred_info["credibility_level"],
        "credibility_reason": cred_info["credibility_reason"],
        "single_source": cred_info["single_source"],
        "divergence": cred_info["divergence"],
        "reference_only": True,
        "large_order_credibility": cred_info,
    }
    if is_legacy:
        result["legacy_warning"] = "legacy_web_algorithm：新浪旧 Web，仅供参考，不得冒充新算法来源"
    return result


# Public aliases make the migration discoverable to provider and graph callers.
build_source_selection = select_fund_flow_source
select_source_by_priority = select_fund_flow_source


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


def build_consensus_evidence(
    records: Iterable[Mapping[str, Any]],
    *,
    symbol: str | None = None,
    requested_as_of: str | None = None,
    field: str | None = None,
    relative_dispersion_threshold: Decimal = DEFAULT_RELATIVE_DISPERSION,
) -> dict[str, Any]:
    """Build the legacy median/MAD comparison as audit evidence only.

    ``select_fund_flow_source`` is the direction gate. This function remains
    available for historical diagnostics: legacy Sina Web observations never
    enter the median, MAD, outlier detection, or direction decision here, and
    a ``data_conflict`` result must not be used for buy/sell/accumulation text.
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
        requested_date = _normalise_date_text(requested_as_of)
        observations = [
            item for item in observations
            if item["date"] is None
            or (requested_date is not None and str(item["date"]) <= requested_date)
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


# Public aliases retain the historical audit API; direction callers use the
# explicit ``select_fund_flow_source`` contract above.
same_field_consensus_audit = build_consensus_evidence
build_same_field_consensus_audit = build_consensus_evidence
consensus_audit = build_consensus_evidence
summarize_source_consensus = build_consensus_evidence
build_consensus = build_consensus_evidence


def consensus_prompt_instruction(consensus: Mapping[str, Any] | None) -> str:
    """Render source-selection and legacy-label instructions for the analyst."""
    if not isinstance(consensus, Mapping):
        return (
            "未提供资金流来源选择 evidence；不得把任一旧 Web 值冒充新算法主力方向，"
            "也不得跨日期、窗口、单位或字段语义回填。"
        )
    if consensus.get("status") in {"selected", "consensus"} and consensus.get("direction_allowed"):
        source = consensus.get("selected_source") or consensus.get("source") or "已选来源"
        field = consensus.get("selected_field") or consensus.get("field") or "未知字段"
        value = consensus.get("selected_value")
        if value is None:
            value = consensus.get("value")
        credibility_level = consensus.get("credibility_level") or (
            "偏高" if consensus.get("credibility") == "high"
            else ("偏低" if consensus.get("credibility") == "low" else "中等偏低")
        )
        credibility_reason = consensus.get("credibility_reason") or ""
        cred_text = f"【参考可信度：{credibility_level}】{credibility_reason}；" if credibility_reason else f"【参考可信度：{credibility_level}】；"
        if consensus.get("legacy_reference") or consensus.get("legacy_web_algorithm"):
            return (
                f"仅有新浪 legacy_web_algorithm 可用，已选择 {source}/{field}={value}；"
                "可以展示该来源自身方向，但必须醒目标注 legacy/旧算法/仅供参考，"
                "不得称为 Eastmoney/THS 新算法，也不得与新算法平均。"
                "平台资金流仅具统计口径参考价值，非主力真实身份或资金流向最终结论。"
            )
        return (
            f"已按固定优先级选择 {source}/{field}={value}，方向来自该来源自身；"
            f"{cred_text}"
            "单一有效新算法来源即可允许方向；其他来源仅作旁证/差异说明，不得跨字段、日期、窗口或单位平均。"
            "【强制约束】平台「主力资金/大单」仅为统计口径代理参考，具有参考价值，"
            "严禁断言为账户级主力身份真相或资金流向最终确定结论；"
            "若同花顺大单与东财主力同向，参考可信度偏高；若存在分歧，必须如实陈述分歧并将方向仅作弱参考。"
        )
    return (
        "资金流来源选择不可用（全部来源失败、日期/字段/单位/窗口不合格）；"
        "必须保留各源原值，禁止输出增持、减持、吸筹等方向摘要。"
    )


_FIELD_VALUE_PATTERNS = {
    "r0_net": (
        re.compile(r"主力(?:资金)?(?:净)?(?:流入额?|流出额?|流入|流出|额|资金净额)[^\n。；;，,]{0,20}?(?:为|达|为约|约|：|:)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*亿"),
    ),
    "netamount": (
        re.compile(r"(?<!主力)(?:总资金|全市场总资金|全市场资金|全市场|总)?(?:净)?(?:流入额?|流出额?|流入|流出|净额)[^\n。；;，,]{0,20}?(?:为|达|为约|约|：|:)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*亿"),
    ),
}

_CUMULATIVE_KEYWORDS = ("累计", "合计", "总计", "5日", "五日", "近5", "近五")


def extract_model_totals(text: str | None) -> dict[str, str]:
    """Extract only explicitly labelled cumulative亿元 values from model text."""
    if not isinstance(text, str) or not text.strip():
        return {}
    found: dict[str, str] = {}
    for field, patterns in _FIELD_VALUE_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                sent_start = max((text.rfind(m, 0, match.start()) for m in ("。", "；", ";", "\n")), default=-1) + 1
                sent_end = min((pos for m in ("。", "；", ";", "\n") if (pos := text.find(m, match.end())) != -1), default=len(text))
                sentence = text[sent_start:sent_end]

                clause_start = max((text.rfind(m, 0, match.start()) for m in ("。", "；", ";", "，", ",", "\n")), default=-1) + 1
                clause_end = min((pos for m in ("。", "；", ";", "，", ",", "\n") if (pos := text.find(m, match.end())) != -1), default=len(text))
                clause = text[clause_start:clause_end]

                if not any(kw in sentence for kw in _CUMULATIVE_KEYWORDS):
                    continue
                if field == "netamount" and "主力" in clause:
                    continue
                matched_prefix = text[match.start():match.end()]
                num_str = match.groups()[-1]
                value = decimal_value(num_str)
                if value is not None:
                    if "流出" in matched_prefix and value > 0 and not num_str.startswith("+") and not num_str.startswith("-"):
                        value = -value
                    found[field] = _decimal_text(value) or ""
                    break
            if field in found:
                break
    return found


def extract_model_daily_values(text: str | None) -> dict[str, str]:
    """Extract daily (single-day) 亿元 values from model text."""
    if not isinstance(text, str) or not text.strip():
        return {}
    found: dict[str, str] = {}
    for field, patterns in _FIELD_VALUE_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                sent_start = max((text.rfind(m, 0, match.start()) for m in ("。", "；", ";", "\n")), default=-1) + 1
                sent_end = min((pos for m in ("。", "；", ";", "\n") if (pos := text.find(m, match.end())) != -1), default=len(text))
                sentence = text[sent_start:sent_end]

                clause_start = max((text.rfind(m, 0, match.start()) for m in ("。", "；", ";", "，", ",", "\n")), default=-1) + 1
                clause_end = min((pos for m in ("。", "；", ";", "，", ",", "\n") if (pos := text.find(m, match.end())) != -1), default=len(text))
                clause = text[clause_start:clause_end]

                if any(kw in sentence for kw in _CUMULATIVE_KEYWORDS):
                    continue
                if field == "netamount" and "主力" in clause:
                    continue
                matched_prefix = text[match.start():match.end()]
                num_str = match.groups()[-1]
                value = decimal_value(num_str)
                if value is not None:
                    if "流出" in matched_prefix and value > 0 and not num_str.startswith("+") and not num_str.startswith("-"):
                        value = -value
                    found[field] = _decimal_text(value) or ""
                    break
            if field in found:
                break
    return found


def validate_model_summary(
    records: Iterable[Mapping[str, Any]],
    model_text: str | None,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    tolerance: Decimal = Decimal("0.01"),
    selected_field: str | None = None,
    selected_source: str | None = None,
    requested_as_of: str | None = None,
) -> dict[str, Any]:
    """Mark model totals and daily values against structured evidence."""
    structured_cum = summarize_evidence(
        records,
        window_days=window_days if window_days > 1 else DEFAULT_WINDOW_DAYS,
        field=selected_field,
        source=selected_source,
        requested_as_of=requested_as_of,
    )
    structured_daily = summarize_evidence(
        records,
        window_days=1,
        field=selected_field,
        source=selected_source,
        requested_as_of=requested_as_of,
    )

    model_totals = extract_model_totals(model_text)
    model_daily = extract_model_daily_values(model_text)

    mismatches: list[dict[str, str]] = []
    unverifiable: list[str] = []
    matched_fields: list[str] = []

    for model_field, model_value_text in model_totals.items():
        structured_value = decimal_value(structured_cum.get(model_field))
        model_value = decimal_value(model_value_text)
        if structured_value is None or model_value is None:
            if structured_value is None:
                unverifiable.append(model_field)
            continue
        if abs(structured_value - model_value) > tolerance:
            mismatches.append(
                {
                    "field": model_field,
                    "structured": _decimal_text(structured_value) or "",
                    "model": _decimal_text(model_value) or "",
                    "unit": "亿元",
                    "reason": "model cumulative total differs from structured evidence",
                }
            )
        else:
            matched_fields.append(model_field)

    for model_field, model_value_text in model_daily.items():
        structured_value = decimal_value(structured_daily.get(model_field))
        model_value = decimal_value(model_value_text)
        if structured_value is None or model_value is None:
            if structured_value is None:
                unverifiable.append(model_field)
            continue
        if abs(structured_value - model_value) > tolerance:
            mismatches.append(
                {
                    "field": model_field,
                    "structured": _decimal_text(structured_value) or "",
                    "model": _decimal_text(model_value) or "",
                    "unit": "亿元",
                    "reason": "model daily value differs from structured evidence",
                }
            )
        else:
            matched_fields.append(model_field)

    combined_model = {**model_daily, **model_totals}
    primary_structured = structured_daily if window_days == 1 else structured_cum

    if primary_structured.get("status") == "data_conflict":
        status = "blocked"
    elif mismatches:
        status = "mismatch"
    elif selected_field:
        canonical_selected = _canonical_field(selected_field) or str(selected_field)
        if canonical_selected in (model_daily.keys() | model_totals.keys()):
            if canonical_selected in unverifiable or primary_structured.get("status") == "partial":
                status = "blocked"
            else:
                status = "matched"
        elif matched_fields:
            status = "blocked" if primary_structured.get("status") == "partial" else "matched"
        else:
            status = "not_checked"
    elif matched_fields:
        status = "blocked" if primary_structured.get("status") == "partial" else "matched"
    elif combined_model and (primary_structured.get("status") == "partial" or unverifiable):
        status = "blocked"
    elif combined_model:
        status = "matched"
    else:
        status = "not_checked"

    return {
        "status": status,
        "hard_guard": {
            "blocked": status not in {"matched", "not_checked"},
            "reason": "模型数值（单日或累计）与结构化 evidence 不一致或结构化窗口不可用"
            if status in {"blocked", "mismatch"}
            else "no explicit model total",
        },
        "structured": primary_structured,
        "selected_field": selected_field,
        "selected_source": selected_source,
        "requested_as_of": requested_as_of,
        "model": combined_model,
        "model_totals": model_totals,
        "model_daily": model_daily,
        "unverifiable_fields": unverifiable,
        "mismatches": mismatches,
        "tolerance": _decimal_text(tolerance),
    }


# ---------------------------------------------------------------------------
# Scale metrics normalization (D-03-1 / C-09-3)
# Pure arithmetic: net_to_circ_mv, net_to_amount
# ---------------------------------------------------------------------------


class DecimalRatio(Decimal):
    """Decimal representing a scale ratio with seamless float/approx comparisons."""

    __hash__ = Decimal.__hash__

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, float):
            return float(self) - other
        return super().__sub__(other)

    def __rsub__(self, other: Any) -> Any:
        if isinstance(other, float):
            return other - float(self)
        return super().__rsub__(other)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, float):
            return float(self) == other
        return super().__eq__(other)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, float):
            return float(self) < other
        return super().__lt__(other)

    def __le__(self, other: Any) -> bool:
        if isinstance(other, float):
            return float(self) <= other
        return super().__le__(other)

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, float):
            return float(self) > other
        return super().__gt__(other)

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, float):
            return float(self) >= other
        return super().__ge__(other)


SCALE_METRIC_VALID_UNITS: dict[str, Decimal] = {
    "元": Decimal("1"),
    "rmb": Decimal("1"),
    "cny": Decimal("1"),
    "yuan": Decimal("1"),
    "万": Decimal("10000"),
    "万元": Decimal("10000"),
    "亿": Decimal("100000000"),
    "亿元": Decimal("100000000"),
}


def _normalize_scale_metric_unit(unit: Any) -> tuple[Decimal | None, str | None]:
    """Validate unit against explicit scale metric conversion table (万元/亿元/元).

    Returns:
        (multiplier_to_yuan, canonical_unit_name) or (None, None) if invalid/unsupported.
    """
    if unit is None:
        return None, None
    text = str(unit).strip().lower()
    if not text:
        return None, None
    multiplier = SCALE_METRIC_VALID_UNITS.get(text)
    if multiplier is None:
        return None, None
    if text in {"元", "rmb", "cny", "yuan"}:
        canonical = "元"
    elif text in {"万", "万元"}:
        canonical = "万元"
    elif text in {"亿", "亿元"}:
        canonical = "亿元"
    else:
        canonical = text
    return multiplier, canonical


def _normalize_symbol_code(symbol: Any) -> str | None:
    """Normalize A-share stock code/symbol to standard 6-digit or 6-digit.MARKET format."""
    if symbol is None:
        return None
    text = str(symbol).strip().upper()
    if not text:
        return None
    match = re.search(r"(\d{6})", text)
    if not match:
        return text
    code = match.group(1)
    if text.endswith((".SH", ".SZ", ".BJ")):
        return f"{code}.{text[-2:]}"
    if text.startswith(("SH", "SZ", "BJ")):
        return f"{code}.{text[:2]}"
    market = (
        "SH"
        if code.startswith(("5", "6", "9"))
        else "BJ"
        if code.startswith(("4", "8"))
        else "SZ"
    )
    return f"{code}.{market}"


def _to_yuan_decimal(
    value: Any, unit: Any, field_name: str
) -> tuple[Decimal | None, str | None, str | None]:
    """Convert an amount to exact Decimal in 元.

    Returns:
        (amount_in_yuan, canonical_unit, gap_error)
    """
    if value is None:
        return None, None, f"{field_name}缺失"
    dec_val = decimal_value(value)
    if dec_val is None:
        return None, None, f"{field_name} ({value!r}) 无法解析为有效有限数值"
    if unit is None or not str(unit).strip():
        return None, None, f"{field_name}单位缺失，不得默认单位"
    multiplier, canonical_unit = _normalize_scale_metric_unit(unit)
    if multiplier is None:
        return (
            None,
            None,
            f"{field_name}单位 '{unit}' 无效或不支持（仅允许显式换算表：万元/亿元/元）",
        )
    try:
        yuan_val = dec_val * multiplier
    except (DecimalException, OverflowError, ValueError) as exc:
        return None, None, f"{field_name} 换算溢出或异常: {exc}"
    if not yuan_val.is_finite():
        return None, None, f"{field_name} 换算后非有限数值"
    return yuan_val, canonical_unit, None


def calculate_fund_flow_scale_metrics(
    ts_code: Any = None,
    trade_date: Any = None,
    net_amount: Any = None,
    net_amount_unit: Any = None,
    circ_mv: Any = None,
    circ_mv_unit: Any = None,
    amount: Any = None,
    amount_unit: Any = None,
    *,
    circ_mv_source: Any = None,
    circ_mv_trade_date: Any = None,
    circ_mv_ts_code: Any = None,
    amount_source: Any = None,
    amount_trade_date: Any = None,
    amount_ts_code: Any = None,
    denominator_source: Any = None,
    denominators: Mapping[str, Any] | None = None,
    daily_basic: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Pure function calculating fund flow scale metrics: net_to_circ_mv and net_to_amount.

    Contracts:
    1. Input: at least ts_code, trade_date, net_amount+unit, denominator (circ_mv and/or amount)+unit.
       Must be same-day and same-security.
    2. Convert to same currency unit before division. Reject invalid units (explicit table: 万元/亿元/元).
       No float default to 0.
    3. Denominator <= 0 rejected with gap recorded. Absolute net amount preserved. Missing market cap
       allows turnover ratio only; the two metrics never substitute for each other.
    4. Outputs named fields net_to_circ_mv, net_to_amount, denominator_source, unit, gaps.
       No ranking or main force strength ranking.
    """
    # 1. Unpack evidence if passed as first argument mapping
    if isinstance(ts_code, Mapping):
        evidence = ts_code
        extracted_code = evidence.get("ts_code")
        if extracted_code is None or not str(extracted_code).strip():
            extracted_code = evidence.get("symbol")
        ts_code = extracted_code

        if trade_date is None:
            for key in ("trade_date", "date", "measurement_date", "as_of"):
                dt = evidence.get(key)
                if dt is not None and str(dt).strip():
                    trade_date = dt
                    break

        if net_amount is None:
            for key in ("net_amount", "selected_value", "value", "r0_net", "netamount"):
                if key in evidence:
                    val = evidence.get(key)
                    if val is not None:
                        net_amount = val
                        break

        if net_amount_unit is None:
            for key in ("net_amount_unit", "selected_unit", "unit", "raw_unit"):
                if key in evidence:
                    val = evidence.get(key)
                    if val is not None and str(val).strip():
                        net_amount_unit = val
                        break

        if circ_mv is None and "circ_mv" in evidence and evidence.get("circ_mv") is not None:
            circ_mv = evidence.get("circ_mv")
        if circ_mv_unit is None and "circ_mv_unit" in evidence and evidence.get("circ_mv_unit") is not None:
            circ_mv_unit = evidence.get("circ_mv_unit")
        if circ_mv_source is None and "circ_mv_source" in evidence and evidence.get("circ_mv_source") is not None:
            circ_mv_source = evidence.get("circ_mv_source")
        if circ_mv_trade_date is None and "circ_mv_trade_date" in evidence and evidence.get("circ_mv_trade_date") is not None:
            circ_mv_trade_date = evidence.get("circ_mv_trade_date")
        if circ_mv_ts_code is None and "circ_mv_ts_code" in evidence and evidence.get("circ_mv_ts_code") is not None:
            circ_mv_ts_code = evidence.get("circ_mv_ts_code")

        if amount is None and "amount" in evidence and evidence.get("amount") is not None:
            amount = evidence.get("amount")
        if amount_unit is None and "amount_unit" in evidence and evidence.get("amount_unit") is not None:
            amount_unit = evidence.get("amount_unit")
        if amount_source is None and "amount_source" in evidence and evidence.get("amount_source") is not None:
            amount_source = evidence.get("amount_source")
        if amount_trade_date is None and "amount_trade_date" in evidence and evidence.get("amount_trade_date") is not None:
            amount_trade_date = evidence.get("amount_trade_date")
        if amount_ts_code is None and "amount_ts_code" in evidence and evidence.get("amount_ts_code") is not None:
            amount_ts_code = evidence.get("amount_ts_code")

        if denominator_source is None and "denominator_source" in evidence and evidence.get("denominator_source") is not None:
            denominator_source = evidence.get("denominator_source")

    # 2. Argument aliases
    if ts_code is None:
        for key in ("symbol", "code"):
            val = kwargs.get(key)
            if val is not None and str(val).strip():
                ts_code = val
                break

    if trade_date is None:
        for key in ("date", "as_of"):
            val = kwargs.get(key)
            if val is not None and str(val).strip():
                trade_date = val
                break

    if net_amount is None:
        for key in ("net_value", "r0_net", "netamount", "net_amount"):
            if key in kwargs:
                val = kwargs.get(key)
                if val is not None:
                    net_amount = val
                    break

    if net_amount_unit is None:
        for key in ("net_unit", "unit", "raw_unit", "net_amount_unit"):
            if key in kwargs:
                val = kwargs.get(key)
                if val is not None and str(val).strip():
                    net_amount_unit = val
                    break

    # 3. Unpack denominators or daily_basic dictionaries
    denom_dict: dict[str, Any] = {}
    if isinstance(denominators, Mapping):
        denom_dict.update(denominators)
    if isinstance(daily_basic, Mapping):
        denom_dict.update(daily_basic)
        if (
            denominator_source is None
            and "source" not in denom_dict
            and "denominator_source" not in denom_dict
        ):
            denominator_source = "tushare.daily_basic"

    if denom_dict:
        if circ_mv is None:
            circ_mv = denom_dict.get("circ_mv")
        if circ_mv_unit is None:
            circ_mv_unit = denom_dict.get("circ_mv_unit")
        if circ_mv_source is None:
            circ_mv_source = denom_dict.get("circ_mv_source") or denom_dict.get("source")
        if circ_mv_trade_date is None:
            circ_mv_trade_date = (
                denom_dict.get("circ_mv_trade_date")
                or denom_dict.get("trade_date")
                or denom_dict.get("date")
            )
        if circ_mv_ts_code is None:
            circ_mv_ts_code = (
                denom_dict.get("circ_mv_ts_code")
                or denom_dict.get("ts_code")
                or denom_dict.get("symbol")
            )

        if amount is None:
            amount = denom_dict.get("amount")
        if amount_unit is None:
            amount_unit = denom_dict.get("amount_unit")
        if amount_source is None:
            amount_source = denom_dict.get("amount_source") or denom_dict.get("source")
        if amount_trade_date is None:
            amount_trade_date = (
                denom_dict.get("amount_trade_date")
                or denom_dict.get("trade_date")
                or denom_dict.get("date")
            )
        if amount_ts_code is None:
            amount_ts_code = (
                denom_dict.get("amount_ts_code")
                or denom_dict.get("ts_code")
                or denom_dict.get("symbol")
            )

        if denominator_source is None:
            denominator_source = denom_dict.get("denominator_source") or denom_dict.get("source")

    # Resolve sources cascade
    if denominator_source is None:
        if circ_mv_source and amount_source and circ_mv_source == amount_source:
            denominator_source = circ_mv_source
        elif circ_mv_source:
            denominator_source = circ_mv_source
        elif amount_source:
            denominator_source = amount_source

    if circ_mv_source is None and denominator_source:
        circ_mv_source = denominator_source
    if amount_source is None and denominator_source:
        amount_source = denominator_source

    gaps: list[str] = []

    # 4. Validate ts_code and trade_date
    norm_ts_code = _normalize_symbol_code(ts_code)
    if not norm_ts_code:
        gaps.append(f"证券代码 ts_code={ts_code!r} 缺失或无效，拒算")

    norm_trade_date = _normalise_date_text(trade_date)
    if not norm_trade_date:
        gaps.append(f"交易日期 trade_date={trade_date!r} 缺失或无法解析为有效日期，拒算")

    # 5. Parse net_amount
    net_dec = decimal_value(net_amount)
    net_in_yuan: Decimal | None = None
    canonical_net_unit: str | None = None
    if net_amount is None:
        gaps.append("资金净额 (net_amount) 缺失，拒算")
    elif net_dec is None:
        gaps.append(f"资金净额 net_amount={net_amount!r} 无法解析为有效数值，拒算")
    elif net_amount_unit is None or not str(net_amount_unit).strip():
        gaps.append("资金净额单位 (net_amount_unit) 缺失，拒算")
    else:
        net_in_yuan, canonical_net_unit, net_err = _to_yuan_decimal(
            net_amount, net_amount_unit, "资金净额"
        )
        if net_err:
            gaps.append(net_err)

    # 6. Calculate net_to_circ_mv
    net_to_circ_mv: DecimalRatio | None = None
    circ_mv_dec = decimal_value(circ_mv)
    if circ_mv is None:
        gaps.append("缺少流通市值 (circ_mv) 数据，流通市值占比 (net_to_circ_mv) 缺失")
    elif circ_mv_dec is None:
        gaps.append(f"流通市值 circ_mv={circ_mv!r} 无法解析为有效数值，拒算流通市值占比")
    elif norm_trade_date is None or norm_ts_code is None or net_in_yuan is None:
        # Prerequisites not met, error already in gaps
        pass
    else:
        circ_date = _normalise_date_text(circ_mv_trade_date) if circ_mv_trade_date else None
        circ_code = _normalize_symbol_code(circ_mv_ts_code) if circ_mv_ts_code else None
        if circ_mv_trade_date and not circ_date:
            gaps.append(
                f"分母交易日 circ_mv_trade_date='{circ_mv_trade_date}' 无法解析为有效日期，拒算"
            )
        elif circ_date and circ_date != norm_trade_date:
            gaps.append(
                f"跨交易日拒算: circ_mv 交易日 '{circ_mv_trade_date}' 与请求交易日 '{trade_date}' 不一致"
            )
        elif circ_mv_ts_code and not circ_code:
            gaps.append(
                f"分母证券代码 circ_mv_ts_code='{circ_mv_ts_code}' 无法解析为有效代码，拒算"
            )
        elif circ_code and circ_code != norm_ts_code:
            gaps.append(
                f"跨证券拒算: circ_mv 证券代码 '{circ_mv_ts_code}' 与请求证券代码 '{ts_code}' 不一致"
            )
        elif circ_mv_unit is None or not str(circ_mv_unit).strip():
            gaps.append("流通市值单位 (circ_mv_unit) 缺失，不得默认单位，流通市值占比拒算")
        else:
            circ_in_yuan, canonical_circ_unit, circ_err = _to_yuan_decimal(
                circ_mv, circ_mv_unit, "流通市值"
            )
            if circ_err:
                gaps.append(circ_err)
            elif circ_in_yuan <= Decimal("0"):
                gaps.append(f"流通市值分母非正 (circ_mv={circ_mv_dec})，零或负分母拒算")
            else:
                net_to_circ_mv = DecimalRatio(net_in_yuan / circ_in_yuan)

    # 7. Calculate net_to_amount
    net_to_amount: DecimalRatio | None = None
    amount_dec = decimal_value(amount)
    if amount is None:
        gaps.append("缺少成交额 (amount) 数据，成交额占比 (net_to_amount) 缺失")
    elif amount_dec is None:
        gaps.append(f"成交额 amount={amount!r} 无法解析为有效数值，拒算成交额占比")
    elif norm_trade_date is None or norm_ts_code is None or net_in_yuan is None:
        # Prerequisites not met, error already in gaps
        pass
    else:
        amt_date = _normalise_date_text(amount_trade_date) if amount_trade_date else None
        amt_code = _normalize_symbol_code(amount_ts_code) if amount_ts_code else None
        if amount_trade_date and not amt_date:
            gaps.append(
                f"分母交易日 amount_trade_date='{amount_trade_date}' 无法解析为有效日期，拒算"
            )
        elif amt_date and amt_date != norm_trade_date:
            gaps.append(
                f"跨交易日拒算: amount 交易日 '{amount_trade_date}' 与请求交易日 '{trade_date}' 不一致"
            )
        elif amount_ts_code and not amt_code:
            gaps.append(
                f"分母证券代码 amount_ts_code='{amount_ts_code}' 无法解析为有效代码，拒算"
            )
        elif amt_code and amt_code != norm_ts_code:
            gaps.append(
                f"跨证券拒算: amount 证券代码 '{amount_ts_code}' 与请求证券代码 '{ts_code}' 不一致"
            )
        elif amount_unit is None or not str(amount_unit).strip():
            gaps.append("成交额单位 (amount_unit) 缺失，不得默认单位，成交额占比拒算")
        else:
            amt_in_yuan, canonical_amt_unit, amt_err = _to_yuan_decimal(
                amount, amount_unit, "成交额"
            )
            if amt_err:
                gaps.append(amt_err)
            elif amt_in_yuan <= Decimal("0"):
                gaps.append(f"成交额分母非正 (amount={amount_dec})，零或负分母拒算")
            else:
                net_to_amount = DecimalRatio(net_in_yuan / amt_in_yuan)

    # 8. Status determination
    if net_to_circ_mv is not None and net_to_amount is not None:
        status = "available"
    elif net_to_circ_mv is not None or net_to_amount is not None:
        status = "partial"
    else:
        status = "unavailable"

    canonical_circ_unit = (
        _normalize_scale_metric_unit(circ_mv_unit)[1] if circ_mv_unit else None
    )
    canonical_amt_unit = (
        _normalize_scale_metric_unit(amount_unit)[1] if amount_unit else None
    )

    return {
        "ts_code": norm_ts_code or str(ts_code or ""),
        "trade_date": norm_trade_date or str(trade_date or ""),
        "net_amount": net_dec,
        "net_amount_raw": str(net_amount) if net_amount is not None else None,
        "net_amount_unit": canonical_net_unit or (str(net_amount_unit) if net_amount_unit else None),
        "unit": canonical_net_unit or (str(net_amount_unit) if net_amount_unit else "亿元"),
        "net_to_circ_mv": net_to_circ_mv,
        "net_to_circ_mv_text": _decimal_text(net_to_circ_mv) if net_to_circ_mv is not None else None,
        "net_to_amount": net_to_amount,
        "net_to_amount_text": _decimal_text(net_to_amount) if net_to_amount is not None else None,
        "net_circ_mv_ratio": net_to_circ_mv,
        "net_amount_ratio": net_to_amount,
        "circ_mv": circ_mv_dec,
        "circ_mv_unit": canonical_circ_unit or (str(circ_mv_unit) if circ_mv_unit else None),
        "circ_mv_source": circ_mv_source,
        "amount": amount_dec,
        "amount_unit": canonical_amt_unit or (str(amount_unit) if amount_unit else None),
        "amount_source": amount_source,
        "denominator_source": denominator_source,
        "denominator_sources": {
            "circ_mv": circ_mv_source,
            "amount": amount_source,
        },
        "denominator_units": {
            "circ_mv": canonical_circ_unit or (str(circ_mv_unit) if circ_mv_unit else None),
            "amount": canonical_amt_unit or (str(amount_unit) if amount_unit else None),
        },
        "status": status,
        "gaps": gaps,
        "gap_list": gaps,
    }


# Public alias
compute_fund_flow_scale_metrics = calculate_fund_flow_scale_metrics

