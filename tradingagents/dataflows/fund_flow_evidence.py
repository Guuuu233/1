"""Structured fund-flow evidence, source alignment, and arithmetic checks."""
from __future__ import annotations

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
    return format(value, "f")


def _as_yi(value: Any) -> Decimal | None:
    parsed = decimal_value(value)
    return None if parsed is None else parsed / YI


def _normalise_source_text(source: Any) -> str:
    return str(source or "").strip().lower().replace(" ", "_")


def infer_algorithm_group(
    source: Any,
    algorithm_group: Any = None,
) -> str:
    """Classify a source without allowing legacy Sina Web data into new consensus."""
    explicit = str(algorithm_group or "").strip()
    if explicit in {NEW_ALGORITHM_GROUP, "new_algorithm", "new"}:
        return NEW_ALGORITHM_GROUP
    if explicit in {LEGACY_WEB_ALGORITHM, "legacy_web", "legacy"}:
        return LEGACY_WEB_ALGORITHM

    label = _normalise_source_text(source)
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
            "tonghuashun",
            "同花顺",
        )
    ):
        return NEW_ALGORITHM_GROUP
    if any(token in label for token in ("legacy", "web", "sina_historical")):
        return LEGACY_WEB_ALGORITHM
    if "historical" in label and "sina" in label:
        return LEGACY_WEB_ALGORITHM
    if label in {"sina", "sina_web", "sinafinance", "新浪", "新浪财经"}:
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
    return aliases.get(text, str(unit or "元").strip() or "元")


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
        return text[:10]
    return text


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
        as_of = str(row.get("as_of") or date or requested_as_of).strip() or requested_as_of
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
    source: str = "sina_app",
    raw_unit: str = "元",
    period_kind: str = "realtime_single_day",
) -> list[dict[str, Any]]:
    """Build evidence for the newer Sina App product, not the legacy Web feed."""
    return build_source_evidence(
        rows,
        symbol=symbol,
        requested_as_of=requested_as_of,
        retrieved_at=retrieved_at,
        source=source,
        raw_unit=raw_unit,
        algorithm_group=NEW_ALGORITHM_GROUP,
        period_kind=period_kind,
    )


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
    requested_as_of: str,
    source: str,
    reason: str,
    retrieved_at: str | None = None,
    algorithm_group: str | None = None,
    period_kind: str | None = None,
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
    algorithm_group: str | None = None,
    period_kind: str | None = None,
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
    """Compute exact daily totals without mixing sources, units, or windows."""
    usable = [
        dict(record)
        for record in records
        if isinstance(record, Mapping) and record.get("status") == "available"
    ]
    period_kinds = {str(record.get("period_kind")) for record in usable if record.get("period_kind")}
    windows = {str(record.get("window") or record.get("time_window")) for record in usable if record.get("window") or record.get("time_window")}
    source_ids = {str(record.get("source") or "") for record in usable}
    mixed_periods = len(period_kinds) > 1 or ("five_day_cumulative" in period_kinds and len(usable) > 1)
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
    netamount = _sum_field(selected, "netamount")
    r0_net = _sum_field(selected, "r0_net")
    status = "available" if len(selected) == window_days and netamount is not None and r0_net is not None else "partial"
    return {
        "window_days": window_days,
        "record_count": len(selected),
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
                "source_family": record.get("source_family") or source_family(source),
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
    for item in observations:
        by_source.setdefault(item["source"], []).append(item["value"])
    duplicate_source_conflict = any(
        len({value for value in values}) > 1 for values in by_source.values()
    )
    unique_values: list[tuple[str, Decimal]] = [
        (source, values[0])
        for source, values in by_source.items()
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
            "source_values": {source: _decimal_text(values[0]) for source, values in by_source.items()},
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
    if median > 0:
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


def build_consensus_evidence(
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
        item for item in observations if item["algorithm_group"] == NEW_ALGORITHM_GROUP
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
    source_fields = {item["field"] for item in new_observations}
    if len(source_fields) > 1 and field is None:
        # Different official fields (for example ``netamount`` vs ``r0_net``)
        # are retained separately but are never averaged into one direction.
        base.update({
            "reason_code": "incomparable_fields",
            "reason": "新算法组来源混合主力与总净额字段，必须按字段分类分别对齐",
            "field_results": {},
        })
        return base

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
        if len(field_groups) != 1:
            field_results[field_name] = {
                "status": "data_conflict",
                "data_conflict": True,
                "reason_code": "incomparable_alignment",
                "reason": "同一字段存在不同股票/日期/时间窗口/单位记录，无法对齐",
                "raw_values": [
                    {
                        "source": item["source"],
                        "value": item["value_text"],
                        "date": item["date"],
                        "period_kind": item["period_kind"],
                        "time_window": item["time_window"],
                        "field": item["field"],
                    }
                    for _, group in field_groups for item in group
                ],
                "direction": "blocked",
                "direction_allowed": False,
            }
        else:
            field_results[field_name] = _evaluate_observation_group(
                field_groups[0][1],
                relative_dispersion_threshold=relative_dispersion_threshold,
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
    if base.get("direction") == "outflow":
        base["direction_summary"] = "主力偏减持/大额资金偏流出"
    elif base.get("direction") == "inflow":
        base["direction_summary"] = "主力偏增持/大额资金偏流入"
    elif base.get("direction") == "neutral":
        base["direction_summary"] = "主力资金接近平衡"
    else:
        base["direction_summary"] = "方向结论已阻断"
    return base


# Public aliases keep callers concise and make the source-consensus contract
# discoverable without changing the provider-facing legacy names.
summarize_source_consensus = build_consensus_evidence
build_consensus = build_consensus_evidence


def consensus_prompt_instruction(consensus: Mapping[str, Any] | None) -> str:
    """Render a deterministic instruction for the smart-money analyst prompt."""
    if not isinstance(consensus, Mapping):
        return (
            "未提供新算法组共识 evidence；不得把任一旧 Web 值写成主力方向，"
            "也不得跨日期、窗口或字段回填。"
        )
    if consensus.get("status") == "consensus" and consensus.get("direction_allowed"):
        return (
            "新算法组已形成可解释低离散度共识；主结论优先采用新算法组 median，"
            f"MAD={consensus.get('mad')}、相对离散度={consensus.get('relative_dispersion')}。"
            "legacy_web_algorithm 仅作旁证，不得覆盖新算法组结论。"
        )
    return (
        "新算法组 evidence 标记 data_conflict（来源不足、字段/日期/窗口不可比或离散度无法解释）；"
        "必须保留各源原值，禁止输出增持、减持、吸筹等方向摘要。"
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
