"""Deterministic financial period compliance checker.

Pure deterministic utility module for validating financial period semantics in
fundamentals reports against actual financial statement inputs (P-2, DAV-809 / DAV-819).

Prohibits LLMs, network, providers, database, and prompts.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from tradingagents.dataflows.financial_announce import classify_financial_period_kind

logger = logging.getLogger(__name__)

# ── Status constants ─────────────────────────────────────────────────────────

COMPLIANCE_STATUS_CHECKED_CLEAN = "checked_clean"
COMPLIANCE_STATUS_VIOLATIONS_FOUND = "violations_found"
COMPLIANCE_STATUS_NOT_CHECKED = "not_checked"

# ── Violation kinds ──────────────────────────────────────────────────────────

KIND_CUMULATIVE_LABELED_AS_SINGLE_QUARTER = "cumulative_labeled_as_single_quarter"
KIND_CUMULATIVE_DOUBLE_COUNTED = "cumulative_double_counted"
KIND_SINGLE_QUARTER_WITHOUT_DERIVATION = "single_quarter_without_derivation"

# ── Numerical parsing & tolerance constants ───────────────────────────────────

# 1% relative tolerance accommodates common rounding in reports (e.g. 11.3464 -> 11.35)
RELATIVE_TOLERANCE: float = 0.01
# Absolute tolerance in Yuan for small amounts
ABSOLUTE_TOLERANCE: float = 1000.0

UNIT_MULTIPLIERS: dict[str, float] = {
    "万亿": 1e12,
    "万亿元": 1e12,
    "亿": 1e8,
    "亿元": 1e8,
    "万": 1e4,
    "万元": 1e4,
    "元": 1.0,
}

# Failure markers indicating upstream provider or fetch failures
DATA_FAILURE_MARKERS: tuple[str, ...] = (
    "【数据获取失败】",
    "未获取到报表数据",
    "接口返回的报表行不可解析",
    "调用失败",
    "数据源异常",
    "本项不可用",
)

# Canonical field aliases and mappings
CANONICAL_FIELD_ALIASES: dict[str, dict[str, list[str]]] = {
    "cashflow": {
        "购建固定资产、无形资产和其他长期资产所支付的现金": [
            "购建固定资产、无形资产和其他长期资产所支付的现金",
            "购建固定资产、无形资产和其他长期资产支付的现金",
            "购建固定资产支付现金",
            "构建固定资产支付现金",
            "构建固定资产所支付的现金",
            "购建固定资产",
            "构建固定资产",
            "资本开支",
            "Capex",
            "capex",
            "pay_fixed_assets_etc_cash",
        ],
        "经营活动产生的现金流量净额": [
            "经营活动产生的现金流量净额",
            "经营活动现金流量净额",
            "经营活动现金流净额",
            "经营活动现金流量",
            "经营现金流净额",
            "经营现金流",
            "经营性现金流",
            "act_cash_flow_net",
        ],
        "投资活动产生的现金流量净额": [
            "投资活动产生的现金流量净额",
            "投资活动现金流量净额",
            "投资现金流净额",
            "投资活动现金流",
            "投资现金流",
            "invest_cash_flow_net",
        ],
        "筹资活动产生的现金流量净额": [
            "筹资活动产生的现金流量净额",
            "筹资活动现金流量净额",
            "筹资现金流净额",
            "筹资活动现金流",
            "筹资现金流",
            "financing_cash_flow_net",
        ],
        "现金及现金等价物净增加额": [
            "现金及现金等价物净增加额",
            "现金净增加额",
            "cash_net_increase",
        ],
    },
    "income_statement": {
        "营业收入": [
            "营业总收入",
            "营业收入",
            "主营业务收入",
            "营收",
            "operating_income",
            "operating_revenue",
        ],
        "营业成本": [
            "营业总成本",
            "营业成本",
            "operating_costs",
            "operating_expenses",
        ],
        "销售费用": ["销售费用", "sales_fee", "selling_expenses"],
        "管理费用": ["管理费用", "manage_fee", "management_expenses"],
        "财务费用": ["财务费用", "financial_expenses", "finance_fee"],
        "营业利润": ["营业利润", "operating_profit"],
        "利润总额": ["利润总额", "total_profit"],
        "净利润": [
            "归属于母公司所有者的净利润",
            "归属于母公司股东的净利润",
            "归母净利润",
            "归母净利",
            "净利润",
            "net_profit",
            "parent_net_profit",
            "net_profit_parent",
        ],
    },
    "balance_sheet": {
        "总资产": ["总资产", "资产总计", "资产总额", "assets_total"],
        "总负债": ["总负债", "负债合计", "负债总额", "total_debt"],
    },
}


def _match_canonical_field(statement: str, text: str) -> Optional[str]:
    """Find the canonical field mentioned in a text snippet."""
    stmt_map = CANONICAL_FIELD_ALIASES.get(statement, {})
    for canonical, aliases in stmt_map.items():
        for alias in aliases:
            if alias in text:
                return canonical
    return None


def _is_approx_equal(val1: float, val2: float) -> bool:
    """Compare two monetary amounts with relative and absolute tolerances."""
    diff = abs(val1 - val2)
    denom = max(abs(val1), abs(val2))
    if denom == 0:
        return diff <= ABSOLUTE_TOLERANCE
    return diff <= max(denom * RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE)


def _safe_float(val: Any) -> Optional[float]:
    """Convert input value to float, handling scientific notation and strings."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_str = val.strip().replace(",", "")
        try:
            return float(val_str)
        except ValueError:
            return None
    return None


# ── Markdown table & input parsing ───────────────────────────────────────────


def _parse_markdown_table(table_text: str) -> list[dict[str, str]]:
    """Parse a markdown table into list of row dictionaries."""
    rows: list[dict[str, str]] = []
    lines = [line.strip() for line in table_text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return rows

    headers = [col.strip() for col in lines[0].split("|")[1:-1]]
    if not headers:
        return rows

    # Skip lines[1] which is separator |:---|---:|
    for line in lines[2:]:
        cols = [col.strip() for col in line.split("|")[1:-1]]
        if len(cols) == len(headers):
            rows.append(dict(zip(headers, cols)))
        elif len(cols) > len(headers):
            rows.append(dict(zip(headers, cols[:len(headers)])))
        else:
            padded = cols + [""] * (len(headers) - len(cols))
            rows.append(dict(zip(headers, padded)))
    return rows


def parse_financial_statement_inputs(
    financial_inputs: dict[str, Any]
) -> Tuple[bool, Optional[str], dict[str, Any]]:
    """Parse financial inputs into normalized rows, metadata, and derivations.

    Returns:
        (is_valid, failure_reason, parsed_data)
    """
    if not financial_inputs or not isinstance(financial_inputs, dict):
        return False, "财务输入为空或数据结构不正确", {}

    parsed: dict[str, Any] = {
        "statements": {},
        "q2_derivation": {},
    }

    found_any_valid_table = False

    for stmt_key in ("cashflow", "income_statement", "balance_sheet"):
        raw_val = financial_inputs.get(stmt_key)
        if raw_val is None or raw_val == "无数据":
            continue

        raw_str = str(raw_val)

        # Check for provider failure markers
        for marker in DATA_FAILURE_MARKERS:
            if marker in raw_str:
                return False, f"报表数据获取失败（{stmt_key}包含失败标记: {marker}）", {}

        # Parse Q2 derivation block if present
        if "【Q2单季派生数据" in raw_str:
            if "【Q2单季派生数据（不可用）】" in raw_str:
                parsed["q2_derivation"][stmt_key] = {
                    "available": False,
                    "reason": "unavailable_block",
                    "values": {},
                }
            elif "single_quarter_derived" in raw_str or "公式 H1-Q1" in raw_str:
                derived_values = {}
                m_vals = re.search(r"派生金额:\s*([^；\n]+)", raw_str)
                if m_vals:
                    val_items = m_vals.group(1).split(",")
                    for item in val_items:
                        if "=" in item:
                            k, v = item.split("=", 1)
                            f_val = _safe_float(v)
                            if f_val is not None:
                                canonical_k = _match_canonical_field(stmt_key, k.strip()) or k.strip()
                                derived_values[canonical_k] = f_val
                parsed["q2_derivation"][stmt_key] = {
                    "available": True,
                    "reason": "ok",
                    "values": derived_values,
                }

        # Parse table rows
        rows = _parse_markdown_table(raw_str)
        if not rows:
            # Check if input is list of dicts directly
            if isinstance(raw_val, list) and all(isinstance(r, dict) for r in raw_val):
                rows = [dict(r) for r in raw_val]

        if not rows:
            continue

        # Extract rows and classify period
        normalized_rows = []
        for r in rows:
            norm_r = dict(r)
            period_end = (
                r.get("period_end")
                or r.get("报告日")
                or r.get("report_date")
                or ""
            ).replace("-", "")
            period_kind = r.get("period_kind")
            reported_label = r.get("reported_period_label")

            if not period_kind or period_kind == "unknown":
                if period_end and len(period_end) >= 8:
                    classified = classify_financial_period_kind(
                        period_end[:8],
                        "balance" if stmt_key == "balance_sheet" else (
                            "income" if stmt_key == "income_statement" else "cashflow"
                        )
                    )
                    period_kind = classified.period_kind
                    reported_label = reported_label or classified.reported_period_label

            if not period_kind:
                if stmt_key == "balance_sheet":
                    period_kind = "period_end_stock"
                else:
                    period_kind = "unknown"

            norm_r["period_kind"] = period_kind
            norm_r["reported_period_label"] = reported_label or "unknown"
            norm_r["period_end"] = period_end

            # Parse numeric fields
            numeric_fields = {}
            for col, val_str in r.items():
                if col in (
                    "report_date",
                    "period_end",
                    "fiscal_period",
                    "reported_period_label",
                    "period_kind",
                    "derivation_formula",
                    "report_date_status",
                    "thscode",
                    "ticker",
                    "period",
                    "fiscal_year",
                    "currency",
                    "报告日",
                ):
                    continue
                num_val = _safe_float(val_str)
                if num_val is not None:
                    # Check if column name specifies unit such as （万元）
                    multiplier = 1.0
                    if "万元" in col or "(万元)" in col or "（万元）" in col:
                        multiplier = 1e4
                    elif "亿元" in col or "(亿元)" in col or "（亿元）" in col:
                        multiplier = 1e8
                    num_val_yuan = num_val * multiplier

                    canonical = _match_canonical_field(stmt_key, col)
                    if canonical:
                        numeric_fields[canonical] = num_val_yuan
                    numeric_fields[col] = num_val_yuan

            norm_r["_numeric_fields"] = numeric_fields
            normalized_rows.append(norm_r)

        if normalized_rows:
            parsed["statements"][stmt_key] = normalized_rows
            found_any_valid_table = True

    if not found_any_valid_table:
        return False, "未能从输入中解析出任何有效财务报表行", {}

    return True, None, parsed


# ── Text analysis & reported amount extraction ───────────────────────────────


def extract_reported_amounts(clause: str) -> list[Tuple[float, str]]:
    """Extract monetary amounts and their raw substrings from a clause.

    Filters out percentages, dates, stock codes, multiples, and non-monetary counts.
    Returns list of (amount_in_yuan, raw_match_str).
    """
    amounts: list[Tuple[float, str]] = []

    # Pattern for numbers with Chinese currency units
    # e.g. 11.35亿元, 16.57 亿元, 122.35 亿元, 5000 万元, 1134640000元, -7.46亿元
    pattern = re.compile(
        r"(?<![0-9A-Za-z])"
        r"([+-]?\s*\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
        r"\s*"
        r"(万亿元|万亿|亿元|亿|万元|万|元)?"
        r"(?![0-9A-Za-z%％倍个天年月季度分])"
    )

    for m in pattern.finditer(clause):
        raw_num_str = m.group(1).replace(" ", "")
        unit = m.group(2) or "元"
        full_match = m.group(0)

        # Exclude years like 2024, 2025, 2026 when without unit
        try:
            val = float(raw_num_str)
        except ValueError:
            continue

        if not m.group(2) and (1990 <= val <= 2035 or val < 10000):
            # Bare 4-digit year or small number without unit
            continue

        multiplier = UNIT_MULTIPLIERS.get(unit, 1.0)
        amount_yuan = val * multiplier
        amounts.append((amount_yuan, full_match.strip()))

    return amounts


def _extract_clause_period_label(clause: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract period semantic category and raw period label from a clause.

    Returns:
        (period_type, raw_label)
        period_type in ("single_quarter", "cumulative", "stock", None)
    """
    # 1. Single quarter patterns
    # e.g. 2026Q2单季, 2024Q3单季, 2024Q4单季, Q2单季, 单季Q2, 二季度单季
    m = re.search(r"((?:20\d{2})?(?:Q[1-4]|第[一二三四1-4]季度?)\s*单季(?:度)?)", clause)
    if m:
        return "single_quarter", m.group(1).strip()

    m = re.search(r"(单季(?:度)?\s*(?:20\d{2})?Q[1-4])", clause)
    if m:
        return "single_quarter", m.group(1).strip()

    m = re.search(r"((?:20\d{2})?Q[2-4])(?![A-Za-z0-9])", clause)
    if m:
        return "single_quarter", m.group(1).strip()

    # 2. Cumulative patterns
    # e.g. 2026H1累计, 2026H1, H1累计, 半年度累计, 2024前三季度, Q3累计, 全年, 年度, 年报
    m = re.search(r"((?:20\d{2})?H1\s*(?:累计)?)", clause)
    if m:
        return "cumulative", m.group(1).strip()

    m = re.search(r"((?:20\d{2})?半年度\s*(?:累计)?)", clause)
    if m:
        return "cumulative", m.group(1).strip()

    m = re.search(r"((?:20\d{2})?中报\s*(?:累计)?)", clause)
    if m:
        return "cumulative", m.group(1).strip()

    m = re.search(r"((?:20\d{2})?前[二三23]季度\s*(?:累计)?)", clause)
    if m:
        return "cumulative", m.group(1).strip()

    m = re.search(r"((?:20\d{2})?Q3\s*累计)", clause)
    if m:
        return "cumulative", m.group(1).strip()

    m = re.search(r"((?:20\d{2})?(?:全年|年度)\s*(?:累计)?)", clause)
    if m:
        return "cumulative", m.group(1).strip()

    m = re.search(r"((?:20\d{2})?Q4\s*累计)", clause)
    if m:
        return "cumulative", m.group(1).strip()

    m = re.search(r"(累计(?:资本开支|经营现金流|营业收入|净利润|金额)?)", clause)
    if m:
        return "cumulative", m.group(1).strip()

    # 3. Balance sheet stock patterns
    m = re.search(r"(\d{1,2}月末|期末|月末)", clause)
    if m:
        return "stock", m.group(1).strip()

    return None, None


# ── Core verification algorithm ──────────────────────────────────────────────


def check_financial_period_compliance(
    report_text: str, financial_inputs: dict[str, Any]
) -> dict[str, Any]:
    """Verify financial period compliance in fundamentals report.

    Args:
        report_text: Text of the fundamentals_report
        financial_inputs: dict with keys "fundamentals", "balance_sheet",
                          "cashflow", "income_statement"

    Returns:
        Structured result dict matching the required contract:
        - status: "checked_clean" | "violations_found" | "not_checked"
        - not_checked_reason: str or None
        - violations: list of violation dicts
    """
    # 1. Parse and validate inputs
    is_valid, fail_reason, parsed_data = parse_financial_statement_inputs(financial_inputs)
    if not is_valid:
        return {
            "status": COMPLIANCE_STATUS_NOT_CHECKED,
            "not_checked_reason": fail_reason or "财报输入数据不可用或未提供足够元数据",
            "violations": [],
        }

    if not report_text or not report_text.strip():
        return {
            "status": COMPLIANCE_STATUS_CHECKED_CLEAN,
            "not_checked_reason": None,
            "violations": [],
        }

    violations: list[dict[str, Any]] = []

    # Map statements for quick lookup
    stmts = parsed_data["statements"]
    q2_derivations = parsed_data["q2_derivation"]

    # Build input value index for flow statements (cashflow, income_statement)
    # entry: (statement, field, period_kind, reported_label, value)
    cumulative_entries: list[dict[str, Any]] = []
    # double counted candidates: (statement, field, cumulative_period, cumulative_val, q1_val, double_sum)
    double_counted_candidates: list[dict[str, Any]] = []

    for stmt_name in ("cashflow", "income_statement"):
        rows = stmts.get(stmt_name, [])
        if not rows:
            continue

        # Look for Q1 row and cumulative rows
        q1_row = next((r for r in rows if r.get("period_kind") == "first_quarter"), None)

        for r in rows:
            period_kind = r.get("period_kind", "unknown")
            period_label = r.get("reported_period_label", "unknown")
            num_fields = r.get("_numeric_fields", {})

            if period_kind in (
                "half_year_cumulative",
                "nine_month_cumulative",
                "annual_cumulative",
            ):
                for field_name, val in num_fields.items():
                    if val is None or val == 0:
                        continue
                    cumulative_entries.append({
                        "statement": stmt_name,
                        "field": field_name,
                        "period_kind": period_kind,
                        "period_label": period_label,
                        "value": float(val),
                    })

                    # If Q1 row exists with same field, calculate erroneous double counted sum
                    if q1_row:
                        q1_val = q1_row.get("_numeric_fields", {}).get(field_name)
                        if q1_val is not None:
                            double_counted_candidates.append({
                                "statement": stmt_name,
                                "field": field_name,
                                "cumulative_period_label": period_label,
                                "cumulative_period_kind": period_kind,
                                "cumulative_value": float(val),
                                "q1_value": float(q1_val),
                                "double_sum": float(val + q1_val),
                            })

    # Break report into sentences and clauses
    sentences = re.split(r"[。\n\r]+", report_text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Split sentence into clauses
        clauses = [c.strip() for c in re.split(r"[，,；;]+", sentence) if c.strip()]
        active_period_type = None
        active_period_label = None

        for clause in clauses:
            p_type, p_label = _extract_clause_period_label(clause)
            if p_type is not None:
                active_period_type = p_type
                active_period_label = p_label

            # Extract amounts in clause
            extracted_amounts = extract_reported_amounts(clause)
            if not extracted_amounts:
                continue

            # ── Check Violation 1: cumulative_labeled_as_single_quarter ──
            if active_period_type == "single_quarter":
                for amount_val, _ in extracted_amounts:
                    # Check if this amount matches a valid derived single-quarter amount
                    matched_as_derived = False
                    for stmt_name in ("cashflow", "income_statement"):
                        deriv_info = q2_derivations.get(stmt_name, {})
                        if deriv_info.get("available"):
                            for d_field, d_val in deriv_info.get("values", {}).items():
                                if _is_approx_equal(amount_val, d_val):
                                    matched_as_derived = True
                                    break
                        if matched_as_derived:
                            break

                    if matched_as_derived:
                        # Valid derived Q2 amount, not a violation
                        continue

                    # Check if this matches an input cumulative value
                    for entry in cumulative_entries:
                        stmt_name = entry["statement"]
                        canonical_field = _match_canonical_field(stmt_name, clause)

                        # Match if clause names the field or alias, or if field matches entry
                        field_matches = False
                        if canonical_field and canonical_field == entry["field"]:
                            field_matches = True
                        elif entry["field"] in clause:
                            field_matches = True
                        elif not canonical_field:
                            # If no specific field name is mentioned in clause,
                            # match entry if value matches
                            field_matches = True

                        if field_matches and _is_approx_equal(amount_val, entry["value"]):
                            # Avoid flagging balance sheet stock items
                            if stmt_name == "balance_sheet":
                                continue

                            # Check single quarter label appropriateness
                            # e.g. Q3 single quarter with nine_month_cumulative, Q4 with annual
                            label_str = active_period_label or "单季"
                            is_target_violation = False
                            if entry["period_kind"] == "half_year_cumulative" and (
                                "Q2" in label_str or "单季" in label_str or "二季度" in label_str
                            ):
                                is_target_violation = True
                            elif entry["period_kind"] == "nine_month_cumulative" and (
                                "Q3" in label_str or "三季度" in label_str or "单季" in label_str
                            ):
                                is_target_violation = True
                            elif entry["period_kind"] == "annual_cumulative" and (
                                "Q4" in label_str or "四季度" in label_str or "单季" in label_str
                            ):
                                is_target_violation = True

                            if is_target_violation:
                                violations.append({
                                    "kind": KIND_CUMULATIVE_LABELED_AS_SINGLE_QUARTER,
                                    "quoted_text": clause,
                                    "statement": stmt_name,
                                    "field": entry["field"],
                                    "reported_label": label_str,
                                    "input_period_kind": entry["period_kind"],
                                    "input_value": float(entry["value"]),
                                    "expected_label": entry["period_label"],
                                })
                                break

            # ── Check Violation 2: cumulative_double_counted ─────────────
            if active_period_type == "cumulative":
                for amount_val, _ in extracted_amounts:
                    for cand in double_counted_candidates:
                        stmt_name = cand["statement"]
                        canonical_field = _match_canonical_field(stmt_name, clause)

                        field_matches = False
                        if canonical_field and canonical_field == cand["field"]:
                            field_matches = True
                        elif cand["field"] in clause:
                            field_matches = True
                        elif not canonical_field:
                            field_matches = True

                        if field_matches and _is_approx_equal(amount_val, cand["double_sum"]):
                            label_str = active_period_label or cand["cumulative_period_label"]
                            violations.append({
                                "kind": KIND_CUMULATIVE_DOUBLE_COUNTED,
                                "quoted_text": clause,
                                "statement": stmt_name,
                                "field": cand["field"],
                                "reported_label": label_str,
                                "input_period_kind": cand["cumulative_period_kind"],
                                "input_value": float(cand["cumulative_value"]),
                                "expected_label": cand["cumulative_period_label"],
                            })
                            break

            # ── Check Violation 3: single_quarter_without_derivation ──────
            if active_period_type == "single_quarter":
                # Only check if single quarter mentions Q2
                label_str = active_period_label or ""
                if "Q2" in label_str or "二季度" in label_str:
                    for stmt_name in ("cashflow", "income_statement"):
                        deriv_info = q2_derivations.get(stmt_name)
                        # Check if derivation block is explicitly unavailable or missing
                        if deriv_info is not None and not deriv_info.get("available", False):
                            canonical_field = _match_canonical_field(stmt_name, clause)
                            if canonical_field:
                                for amount_val, _ in extracted_amounts:
                                    if amount_val != 0:
                                        violations.append({
                                            "kind": KIND_SINGLE_QUARTER_WITHOUT_DERIVATION,
                                            "quoted_text": clause,
                                            "statement": stmt_name,
                                            "field": canonical_field,
                                            "reported_label": label_str,
                                            "input_period_kind": "unknown",
                                            "input_value": float(amount_val),
                                            "expected_label": "not_available",
                                        })
                                        break

    # Deduplicate violations while preserving order
    unique_violations: list[dict[str, Any]] = []
    seen = set()
    for v in violations:
        key = (v["kind"], v["statement"], v["field"], v["reported_label"], round(v["input_value"], 2))
        if key not in seen:
            seen.add(key)
            unique_violations.append(v)

    if unique_violations:
        return {
            "status": COMPLIANCE_STATUS_VIOLATIONS_FOUND,
            "not_checked_reason": None,
            "violations": unique_violations,
        }

    return {
        "status": COMPLIANCE_STATUS_CHECKED_CLEAN,
        "not_checked_reason": None,
        "violations": [],
    }
