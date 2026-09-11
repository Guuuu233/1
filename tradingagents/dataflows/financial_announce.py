"""A-share financial-report announcement effective-date resolution.

Sina ``stock_financial_report_sina`` exposes ``公告日期``, but for income and
cash-flow statements that column is frequently rewritten when a later filing
reprints the same period as a year-over-year comparable. Balance-sheet
quarterly rows are usually stable; annual BS rows can still be refreshed as
"opening balances" in the next year's filings.

Because of that vendor semantics, raw ``公告日期`` alone is not a safe
as-of cutoff. Strategy A4 (windowed):

1. Compute the statutory disclosure deadline for report period T.
2. Collect the three statements' ``公告日期`` for T.
3. Drop values later than ``statutory_deadline + LATE_FILING_GRACE_DAYS``
   (treated as YoY-comparable refresh — typically ~360 days late).
   Real late filers (days–weeks past the deadline) stay inside the window.
4. If any remain → effective announce date = max(remaining)
   (path ``max_within_window``).
5. If none remain:
   - had only YoY-late values → path ``dropped_yoy_refresh``,
     effective = statutory deadline (conservative floor; original first
     filing date was overwritten by the vendor);
   - no parseable announce dates at all → path ``statutory_fallback``,
     effective = statutory deadline.
6. Keep period T only when effective date <= analysis ``curr_date``.

``LATE_FILING_GRACE_DAYS`` is chosen from the empirical bimodal gap between
real filings (cluster near 0, late tail ≤ ~90d) and YoY refresh (cluster
~301–400d) across a multi-stock probe; see constant docstring.

Statutory / exchange deadlines used as the hard upper bound (实务口径):

- Annual report: within 4 months after fiscal year end → Apr 30
- Semi-annual (interim / 中期) report: within 2 months after H1 end → Aug 31
  Source: CSRC Order No. 226 (2025) 《上市公司信息披露管理办法》第十三条
  (annual ≤ 4 months; interim ≤ 2 months). Order No. 226 superseded Order
  No. 182; Art. 20 of Order No. 226 is about non-standard audit opinions,
  not periodic-report deadlines.

- Quarterly reports (Q1 / Q3): within 1 month after period end → Apr 30 / Oct 31
  Source: Shanghai / Shenzhen Stock Exchange stock listing rules (交易所股票
  上市规则) on quarterly-report disclosure windows. Exact article numbers
  vary by board edition; verify against the currently effective listing rules
  before changing these day-of-month constants. Quarterly deadlines are no
  longer spelled out in the CSRC Measures after the 2021 revision narrowed
  that regulation to annual + interim reports.

Note: these constants are operational cutoffs for anti-lookahead, not a
substitute for legal advice. Regulatory text is revised over time — re-check
the current CSRC order and exchange listing rules before editing.

Historical note (no functional impact today): CSRC Order No. 40 (2007) Art. 20
also required that Q1 disclosure not precede the prior annual report; both
deadlines still land on Apr 30 under the mapping above.

KNOWN LIMITATIONS
-----------------
When every available ``公告日期`` for period T falls beyond the grace window
(path ``dropped_yoy_refresh``), the vendor has overwritten the original first
filing date with a later YoY-comparable reprint. We no longer know the true
first-public date, so the effective date falls back to the statutory deadline.

That residual uncertainty is **source-data loss**, not a misclassification by
the window. It can still under-estimate visibility by days–weeks *only if* the
issuer actually filed late *and* the first filing date was erased *and* the
analysis date sits inside that late-filing gap. Probability is low and bounded;
rejecting the whole period would drop ~20% of periods, often the one nearest
the analysis date, which is worse for historical research.

If a future vendor (e.g. CNINFO / exchange official announce calendar) exposes
authoritative first-filing dates, replace the statutory floor with that field
and this limitation goes away. Until then, prompt text must disclose the
estimate when ``dropped_yoy_refresh`` is used (see ``financial_cutoff_header``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_REPORT_PERIOD_RE = re.compile(r"^\d{8}$")


# Grace window past the statutory deadline that still counts as a real first
# filing, not a YoY-column rewrite. Empirically (8 symbols, 1678 table-rows):
# announce_date - statutory_deadline is bimodal —
#   near cluster: ≤ ~90d (on-time + real late filers, e.g. STAR 8–15d late)
#   valley:       ~91–300d nearly empty
#   YoY cluster:  ~301–400d (median ~359d)
# N=120 sits in the valley: rescues real late periods without admitting YoY.
LATE_FILING_GRACE_DAYS = 120

# Resolution paths (also emitted in DEBUG logs):
PATH_MAX_WITHIN_WINDOW = "max_within_window"
PATH_STATUTORY_FALLBACK = "statutory_fallback"
PATH_DROPPED_YOY_REFRESH = "dropped_yoy_refresh"

REPORT_COL_CANDIDATES: tuple[str, ...] = (
    "报告日",
    "end_date",
    "REPORT_DATE",
    "报告期",
    "report_date",
    "end_dt",
)
ANNOUNCE_COL_CANDIDATES: tuple[str, ...] = (
    "公告日期",
    "实际公告日",
    "f_ann_date",
    "ann_date",
    "actual_ann_date",
    "NOTICE_DATE",
    "ann_dt",
)


@dataclass(frozen=True)
class EffectiveAnnounceDate:
    """Resolved as-of announce date for one report period."""

    report_period: str  # YYYYMMDD
    effective_date: date
    statutory_deadline: date
    path: str  # max_within_window | statutory_fallback | dropped_yoy_refresh
    kept_announce_dates: tuple[date, ...]
    discarded_announce_dates: tuple[date, ...]
    grace_days: int = LATE_FILING_GRACE_DAYS

    @property
    def effective_date_str(self) -> str:
        return self.effective_date.isoformat()

    @property
    def report_period_label(self) -> str:
        return format_report_period_label(self.report_period)


@dataclass(frozen=True)
class FinancialPeriodKind:
    """Classified financial report period kind and derivation metadata."""

    reported_period_label: str
    period_kind: str  # first_quarter | half_year_cumulative | nine_month_cumulative | annual_cumulative | period_end_stock | unknown
    derivation_formula: str = "not_derived"


@dataclass(frozen=True)
class Q2DerivationResult:
    """Result of deriving Q2 single quarter from H1 cumulative minus Q1."""

    reported_period_label: str  # e.g. "2024Q2" or "unknown"
    period_kind: str  # "single_quarter_derived" | "unknown"
    derivation_formula: str  # "H1-Q1" | "not_derived"
    h1_period: str  # "YYYYMMDD" or ""
    q1_period: str  # "YYYYMMDD" or ""
    values: dict[str, float]  # {field_name: diff_value}
    missing: tuple[str, ...]  # whitelist columns missing or skipped
    reason: str  # "ok" / "" on success, or rejection code


INCOME_DERIVATION_WHITELIST: tuple[str, ...] = (
    "营业总收入",
    "营业收入",
    "营业总成本",
    "营业成本",
    "税金及附加",
    "营业税金及附加",
    "销售费用",
    "管理费用",
    "研发费用",
    "财务费用",
    "营业利润",
    "利润总额",
    "所得税费用",
    "净利润",
    "归属于母公司所有者的净利润",
)

CASHFLOW_DERIVATION_WHITELIST: tuple[str, ...] = (
    "销售商品、提供劳务收到的现金",
    "经营活动现金流入小计",
    "购买商品、接受劳务支付的现金",
    "支付给职工以及为职工支付的现金",
    "支付的各项税费",
    "经营活动现金流出小计",
    "经营活动产生的现金流量净额",
    "购建固定资产、无形资产和其他长期资产所支付的现金",
    "投资活动产生的现金流量净额",
    "吸收投资收到的现金",
    "取得借款收到的现金",
    "分配股利、利润或偿付利息支付的现金",
    "筹资活动产生的现金流量净额",
    "现金及现金等价物净增加额",
)

SCOPE_COMPARISON_COL_KEYWORDS: tuple[str, ...] = (
    "合并范围",
    "币种",
    "单位",
    "会计口径",
)

# Operational / entity subjects that end with metadata keywords (e.g. "营业单位")
# but represent business operating segments / accounting subjects rather than
# measurement or reporting units. Explicit exclusion prevents false-positive scope mismatches.
SCOPE_COMPARISON_EXCLUDED_KEYWORDS: tuple[str, ...] = (
    "营业单位",
)

# Trailing annotations on metadata columns:
# 1. Colons: ASCII ':' or full-width '：', followed by comment/value (e.g. '单位：元', '单位: 万元')
# 2. Brackets: full-width '（...）', '【...】' or ASCII '(...)', '[...]' (e.g. '币种（CNY）', '币种(CNY)')
_SCOPE_ANNOTATION_RE = re.compile(
    r"(?:\s*[:：].*|\s*(?:[（(][^）)]*[）)]|[【\[][^】\]]*[】\]]))\s*$"
)

PERCENTAGE_COL_KEYWORDS: tuple[str, ...] = (
    "同比",
    "环比",
    "%",
    "百分比",
    "增长率",
)

CASHFLOW_STOCK_COL_KEYWORDS: tuple[str, ...] = (
    "期末",
    "期初",
)

PER_SHARE_COL_KEYWORDS: tuple[str, ...] = (
    "每股",
)


def _is_scope_comparison_column(column: object) -> bool:
    """Return whether a column is dedicated comparability metadata.

    Recognizes base keywords (e.g. '币种', '单位', '合并范围', '会计口径'),
    valid prefix/suffix variants (e.g. '报表币种', '报表单位', '本期会计口径'),
    and annotated variants (e.g. '单位：元', '币种（CNY）', '币种(CNY)').

    Excludes accounting/amount subjects such as '营业单位' (business/operating
    units) and lines like '处置子公司及其他营业单位收到的现金净额', '按币种折算金额',
    avoiding arbitrary substring matching.
    """
    if not isinstance(column, str):
        return False
    name = column.strip()
    if not name:
        return False

    # Strip trailing annotations (colons and parenthetical comments)
    base_name = _SCOPE_ANNOTATION_RE.sub("", name).strip()
    if not base_name:
        return False

    # Explicit exclusion for business/operating units ending with "单位"
    if any(
        base_name == excluded or base_name.endswith(excluded)
        for excluded in SCOPE_COMPARISON_EXCLUDED_KEYWORDS
    ):
        return False

    return any(
        base_name == keyword or base_name.endswith(keyword)
        for keyword in SCOPE_COMPARISON_COL_KEYWORDS
    )


def parse_yyyymmdd(value) -> Optional[date]:
    """Parse common vendor date forms into a ``date``; invalid → None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.date()

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat", "null"}:
        return None

    # Prefer pure 8-digit first (Sina 报告日 / 公告日期).
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) >= 8:
        try:
            return datetime.strptime(digits[:8], "%Y%m%d").date()
        except ValueError:
            pass

    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def normalize_report_period(value) -> Optional[str]:
    """Normalize a report-period token to YYYYMMDD."""
    d = parse_yyyymmdd(value)
    if d is None:
        return None
    token = d.strftime("%Y%m%d")
    if token[4:] not in {"0331", "0630", "0930", "1231"}:
        # Still allow non-standard period ends; statutory helper will reject.
        return token
    return token


def format_report_period_label(report_period: str) -> str:
    """Human-readable period label, e.g. 2026Q1 / 2025H1 / 2025A."""
    token = normalize_report_period(report_period)
    if not token:
        return str(report_period)
    year = token[:4]
    md = token[4:]
    if md == "0331":
        return f"{year}Q1"
    if md == "0630":
        return f"{year}H1"
    if md == "0930":
        return f"{year}Q3"
    if md == "1231":
        return f"{year}A"
    return token


def classify_financial_period_kind(
    report_period, statement_kind: Optional[str] = None
) -> FinancialPeriodKind:
    """Classify financial statement period kind without single-quarter derivation.

    Maps (report_period, statement_kind) into an unambiguous period_kind:
    - income / cashflow + 0331 -> first_quarter
    - income / cashflow + 0630 -> half_year_cumulative
    - income / cashflow + 0930 -> nine_month_cumulative
    - income / cashflow + 1231 -> annual_cumulative
    - balance + (0331/0630/0930/1231) -> period_end_stock
    - other -> unknown
    """
    token = normalize_report_period(report_period)
    if not token or len(token) != 8:
        label = format_report_period_label(report_period) if report_period is not None else ""
        return FinancialPeriodKind(
            reported_period_label=str(label),
            period_kind="unknown",
            derivation_formula="not_derived",
        )

    label = format_report_period_label(token)
    if statement_kind not in ("income", "cashflow", "balance"):
        return FinancialPeriodKind(
            reported_period_label=label,
            period_kind="unknown",
            derivation_formula="not_derived",
        )

    md = token[4:]
    if statement_kind in ("income", "cashflow"):
        if md == "0331":
            kind = "first_quarter"
        elif md == "0630":
            kind = "half_year_cumulative"
        elif md == "0930":
            kind = "nine_month_cumulative"
        elif md == "1231":
            kind = "annual_cumulative"
        else:
            kind = "unknown"
    elif statement_kind == "balance":
        if md in ("0331", "0630", "0930", "1231"):
            kind = "period_end_stock"
        else:
            kind = "unknown"
    else:
        kind = "unknown"

    return FinancialPeriodKind(
        reported_period_label=label,
        period_kind=kind,
        derivation_formula="not_derived",
    )


def derive_q2_from_h1_q1(
    statement_kind: str,
    df: pd.DataFrame,
    effective_map: Optional[dict[str, EffectiveAnnounceDate]] = None,
) -> Q2DerivationResult:
    """Derive Q2 single quarter (H1 - Q1) for income and cashflow statements.

    Refuses with explicit reason if:
    - statement_kind is balance ("balance_is_stock")
    - statement_kind is not income or cashflow ("invalid_statement_kind")
    - df is empty or missing report-period column ("no_eligible_fields")
    - latest report period is not YYYY0630 ("not_h1_latest")
    - same-year Q1 (YYYY0331) is not in the filtered df ("missing_q1")
    - comparability / scope columns differ ("scope_mismatch")
    - all candidate fields are percentage-only ("percent_only")
    - no computable whitelist fields found ("no_eligible_fields")
    """
    _ = effective_map
    if statement_kind == "balance":
        return Q2DerivationResult(
            reported_period_label="unknown",
            period_kind="unknown",
            derivation_formula="not_derived",
            h1_period="",
            q1_period="",
            values={},
            missing=(),
            reason="balance_is_stock",
        )

    if statement_kind not in ("income", "cashflow"):
        return Q2DerivationResult(
            reported_period_label="unknown",
            period_kind="unknown",
            derivation_formula="not_derived",
            h1_period="",
            q1_period="",
            values={},
            missing=(),
            reason="invalid_statement_kind",
        )

    if df is None or df.empty:
        return Q2DerivationResult(
            reported_period_label="unknown",
            period_kind="unknown",
            derivation_formula="not_derived",
            h1_period="",
            q1_period="",
            values={},
            missing=(),
            reason="no_eligible_fields",
        )

    rep_col = next((c for c in REPORT_COL_CANDIDATES if c in df.columns), None)
    if rep_col is None:
        return Q2DerivationResult(
            reported_period_label="unknown",
            period_kind="unknown",
            derivation_formula="not_derived",
            h1_period="",
            q1_period="",
            values={},
            missing=(),
            reason="no_eligible_fields",
        )

    parsed_periods: list[tuple[str, pd.Series]] = []
    for _, row in df.iterrows():
        token = normalize_report_period(row.get(rep_col))
        if token and len(token) == 8:
            parsed_periods.append((token, row))

    if not parsed_periods:
        return Q2DerivationResult(
            reported_period_label="unknown",
            period_kind="unknown",
            derivation_formula="not_derived",
            h1_period="",
            q1_period="",
            values={},
            missing=(),
            reason="no_eligible_fields",
        )

    parsed_periods.sort(key=lambda x: x[0], reverse=True)
    latest_period, _ = parsed_periods[0]

    if not latest_period.endswith("0630"):
        return Q2DerivationResult(
            reported_period_label="unknown",
            period_kind="unknown",
            derivation_formula="not_derived",
            h1_period="",
            q1_period="",
            values={},
            missing=(),
            reason="not_h1_latest",
        )

    year = latest_period[:4]
    h1_period = f"{year}0630"
    q1_period = f"{year}0331"
    reported_period_label = f"{year}Q2"

    h1_rows = [row for p, row in parsed_periods if p == h1_period]
    q1_rows = [row for p, row in parsed_periods if p == q1_period]

    if not h1_rows:
        return Q2DerivationResult(
            reported_period_label=reported_period_label,
            period_kind="unknown",
            derivation_formula="not_derived",
            h1_period=h1_period,
            q1_period=q1_period,
            values={},
            missing=(),
            reason="not_h1_latest",
        )

    if not q1_rows:
        return Q2DerivationResult(
            reported_period_label=reported_period_label,
            period_kind="unknown",
            derivation_formula="not_derived",
            h1_period=h1_period,
            q1_period=q1_period,
            values={},
            missing=(),
            reason="missing_q1",
        )

    h1_row = h1_rows[0]
    q1_row = q1_rows[0]

    # Scope / comparability column check
    scope_cols = [
        c
        for c in df.columns
        if _is_scope_comparison_column(c)
    ]
    for sc in scope_cols:
        h1_val = h1_row.get(sc)
        q1_val = q1_row.get(sc)
        h1_has = (
            h1_val is not None
            and not (isinstance(h1_val, float) and pd.isna(h1_val))
            and str(h1_val).strip() not in ("", "nan", "None", "NAT", "null")
        )
        q1_has = (
            q1_val is not None
            and not (isinstance(q1_val, float) and pd.isna(q1_val))
            and str(q1_val).strip() not in ("", "nan", "None", "NAT", "null")
        )
        if h1_has != q1_has:
            return Q2DerivationResult(
                reported_period_label=reported_period_label,
                period_kind="unknown",
                derivation_formula="not_derived",
                h1_period=h1_period,
                q1_period=q1_period,
                values={},
                missing=(),
                reason="scope_mismatch",
            )
        if h1_has and q1_has:
            if str(h1_val).strip() != str(q1_val).strip():
                return Q2DerivationResult(
                    reported_period_label=reported_period_label,
                    period_kind="unknown",
                    derivation_formula="not_derived",
                    h1_period=h1_period,
                    q1_period=q1_period,
                    values={},
                    missing=(),
                    reason="scope_mismatch",
                )

    if statement_kind == "income":
        whitelist = INCOME_DERIVATION_WHITELIST
    else:
        whitelist = CASHFLOW_DERIVATION_WHITELIST

    values: dict[str, float] = {}
    missing: list[str] = []

    for field in whitelist:
        if field not in df.columns:
            missing.append(field)
            continue

        if any(k in field for k in PER_SHARE_COL_KEYWORDS):
            missing.append(field)
            continue

        if any(p in field for p in PERCENTAGE_COL_KEYWORDS):
            continue
        if statement_kind == "cashflow" and any(
            s in field for s in CASHFLOW_STOCK_COL_KEYWORDS
        ):
            continue

        h1_val = h1_row.get(field)
        q1_val = q1_row.get(field)

        h1_num = pd.to_numeric(h1_val, errors="coerce")
        q1_num = pd.to_numeric(q1_val, errors="coerce")

        if pd.isna(h1_num) or pd.isna(q1_num):
            missing.append(field)
            continue

        try:
            h1_f = float(h1_num)
            q1_f = float(q1_num)
            if not (pd.isna(h1_f) or pd.isna(q1_f)):
                diff = round(h1_f - q1_f, 6)
                values[field] = diff
            else:
                missing.append(field)
        except (ValueError, TypeError, OverflowError):
            missing.append(field)

    if not values:
        candidate_cols = [
            c
            for c in df.columns
            if c not in REPORT_COL_CANDIDATES
            and c not in ANNOUNCE_COL_CANDIDATES
            and not _is_scope_comparison_column(c)
        ]
        if candidate_cols and all(
            any(p in c for p in PERCENTAGE_COL_KEYWORDS) for c in candidate_cols
        ):
            return Q2DerivationResult(
                reported_period_label=reported_period_label,
                period_kind="unknown",
                derivation_formula="not_derived",
                h1_period=h1_period,
                q1_period=q1_period,
                values={},
                missing=tuple(missing),
                reason="percent_only",
            )
        return Q2DerivationResult(
            reported_period_label=reported_period_label,
            period_kind="unknown",
            derivation_formula="not_derived",
            h1_period=h1_period,
            q1_period=q1_period,
            values={},
            missing=tuple(missing),
            reason="no_eligible_fields",
        )

    return Q2DerivationResult(
        reported_period_label=reported_period_label,
        period_kind="single_quarter_derived",
        derivation_formula="H1-Q1",
        h1_period=h1_period,
        q1_period=q1_period,
        values=values,
        missing=tuple(missing),
        reason="ok",
    )


def format_q2_derivation_block(
    result: Q2DerivationResult,
    effective_map: Optional[dict[str, EffectiveAnnounceDate]] = None,
) -> str:
    """Format Q2 derivation result into prompt-visible injection block."""
    if result.period_kind == "single_quarter_derived":
        h1_eff_str = ""
        q1_eff_str = ""
        if effective_map:
            h1_eff = effective_map.get(result.h1_period)
            q1_eff = effective_map.get(result.q1_period)
            if h1_eff and q1_eff:
                h1_eff_str = f"（生效公告日 {h1_eff.effective_date_str}）"
                q1_eff_str = f"（生效公告日 {q1_eff.effective_date_str}）"

        val_lines = []
        for k, v in result.values.items():
            if isinstance(v, float) and v.is_integer():
                v_str = str(int(v))
            else:
                v_str = str(v)
            val_lines.append(f"{k}={v_str}")
        val_block = ", ".join(val_lines)
        return (
            f"【Q2单季派生数据（{result.reported_period_label}，公式 {result.derivation_formula}）】\n"
            f"（reported_period_label={result.reported_period_label}, "
            f"period_kind={result.period_kind}, "
            f"derivation_formula={result.derivation_formula}；"
            f"基准报告期: H1={result.h1_period}{h1_eff_str}, Q1={result.q1_period}{q1_eff_str}；"
            f"派生金额: {val_block}；"
            "这是 H1 累计减 Q1 得到的 Q2 单季，不是报表原始 Q2 行）"
        )
    else:
        reason_explanations = {
            "not_h1_latest": "最新已公开报告期非半年度（0630）",
            "balance_is_stock": "资产负债表为期末时点存量指标，不可做单季流量派生",
            "missing_q1": "同年度一季度（0331）财报未在当前分析日之前公开",
            "q1_not_public": "同年度一季度（0331）财报未在当前分析日之前公开",
            "scope_mismatch": "半年度与一季度合并范围/币种/单位/会计口径不一致",
            "no_eligible_fields": "无可用的同口径可减金额字段",
            "percent_only": "候选字段均为百分比/比率字段，不可做差值派生",
        }
        explanation = reason_explanations.get(result.reason, "未满足派生条件")
        return (
            f"【Q2单季派生数据（不可用）】\n"
            f"（Q2_single_quarter=N/A, period_kind=unknown, reason={result.reason}；"
            f"原因: {explanation}；"
            "禁止把 H1 累计当作 Q2 单季使用，不得将 H1 累计金额误作单季数据）"
        )



def statutory_disclosure_deadline(report_period) -> date:
    """Return the last day a report period is guaranteed public under A4.

    Mapping (实务口径，修改前请核对现行规则):
    - 1231 → next Apr 30  (CSRC Order No. 226 Art. 13: annual ≤ 4 months)
    - 0630 → Aug 31       (CSRC Order No. 226 Art. 13: interim ≤ 2 months)
    - 0331 → Apr 30       (exchange listing rules: quarterly ≤ 1 month)
    - 0930 → Oct 31       (exchange listing rules: quarterly ≤ 1 month)

    Quarterly article numbers differ across SSE/SZSE board editions; see
    module docstring. Do not cite CSRC Order No. 182 Art. 20 for these
    deadlines (that article is not the periodic-report deadline provision
    under the current Measures).
    """
    token = normalize_report_period(report_period)
    if not token or not _REPORT_PERIOD_RE.match(token):
        raise ValueError(f"unsupported report period: {report_period!r}")

    year = int(token[:4])
    md = token[4:]
    if md == "1231":
        return date(year + 1, 4, 30)
    if md == "0630":
        return date(year, 8, 31)
    if md == "0331":
        return date(year, 4, 30)
    if md == "0930":
        return date(year, 10, 31)
    raise ValueError(
        f"unsupported report period month-day {md!r} for {report_period!r}; "
        "expected 0331/0630/0930/1231"
    )


def resolve_effective_announce_date(
    report_period,
    announce_dates: Sequence,
    *,
    grace_days: int = LATE_FILING_GRACE_DAYS,
) -> EffectiveAnnounceDate:
    """Resolve the effective public announce date for one report period (A4).

    Hard drop only beyond ``statutory_deadline + grace_days``. Short real
    late filings stay; ~1y YoY rewrites are discarded. Falling back to the
    statutory deadline is used only when nothing remains inside the window.
    """
    token = normalize_report_period(report_period)
    if not token:
        raise ValueError(f"invalid report period: {report_period!r}")
    if grace_days < 0:
        raise ValueError(f"grace_days must be >= 0, got {grace_days!r}")

    deadline = statutory_disclosure_deadline(token)
    window_end = deadline + timedelta(days=grace_days)
    parsed: list[date] = []
    for raw in announce_dates:
        d = parse_yyyymmdd(raw)
        if d is not None:
            parsed.append(d)

    # Keep real on-time and real late filings; drop YoY rewrites (~360d).
    kept = tuple(sorted({d for d in parsed if d <= window_end}))
    discarded = tuple(sorted({d for d in parsed if d > window_end}))

    if kept:
        effective = max(kept)
        path = PATH_MAX_WITHIN_WINDOW
    elif discarded:
        # All announce dates look like YoY refresh; original first-filing
        # date is gone. Statutory deadline is the non-leaky floor we still have.
        effective = deadline
        path = PATH_DROPPED_YOY_REFRESH
    else:
        effective = deadline
        path = PATH_STATUTORY_FALLBACK

    result = EffectiveAnnounceDate(
        report_period=token,
        effective_date=effective,
        statutory_deadline=deadline,
        path=path,
        kept_announce_dates=kept,
        discarded_announce_dates=discarded,
        grace_days=grace_days,
    )
    logger.debug(
        "effective announce period=%s effective=%s path=%s statutory=%s "
        "window_end=%s grace_days=%s kept=%s discarded=%s",
        result.report_period,
        result.effective_date_str,
        result.path,
        result.statutory_deadline.isoformat(),
        window_end.isoformat(),
        grace_days,
        [d.isoformat() for d in result.kept_announce_dates],
        [d.isoformat() for d in result.discarded_announce_dates],
    )
    return result


def build_effective_announce_map(
    tables: dict[str, pd.DataFrame],
    report_col: str = "报告日",
    announce_col: str = "公告日期",
) -> dict[str, EffectiveAnnounceDate]:
    """Build period → effective announce date from up to three statement frames.

    ``tables`` keys are free-form labels (e.g. 资产负债表/利润表/现金流量表).
    Missing tables or missing periods simply contribute fewer announce dates.
    """
    period_to_anns: dict[str, list] = {}
    for _name, df in tables.items():
        if df is None or getattr(df, "empty", True):
            continue
        rep_col = report_col if report_col in df.columns else next(
            (c for c in REPORT_COL_CANDIDATES if c in df.columns), None
        )
        if rep_col is None:
            continue

        ann_cols = [
            c
            for c in (announce_col, *ANNOUNCE_COL_CANDIDATES)
            if c in df.columns
        ]
        seen_ann = set()
        unique_ann_cols = [
            c for c in ann_cols if not (c in seen_ann or seen_ann.add(c))
        ]
        if not unique_ann_cols:
            continue

        for _, row in df.iterrows():
            period = normalize_report_period(row.get(rep_col))
            if not period:
                continue
            for c in unique_ann_cols:
                val = row.get(c)
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    period_to_anns.setdefault(period, []).append(val)

    out: dict[str, EffectiveAnnounceDate] = {}
    for period, anns in period_to_anns.items():
        try:
            out[period] = resolve_effective_announce_date(period, anns)
        except ValueError:
            logger.warning("skip report period with unsupported shape: %s", period)
    return out


def filter_financial_df_by_effective_announce(
    df: pd.DataFrame,
    effective_map: dict[str, EffectiveAnnounceDate],
    curr_date: str,
    report_col: str = "报告日",
) -> tuple[pd.DataFrame, Optional[EffectiveAnnounceDate]]:
    """Keep rows whose resolved effective announce date is on/before curr_date.

    Returns (filtered_df, latest_kept_effective).
    """
    if df is None or df.empty:
        return df, None
    rep_col = report_col if report_col in df.columns else next(
        (c for c in REPORT_COL_CANDIDATES if c in df.columns), None
    )
    if rep_col is None:
        raise ValueError(f"missing report column {report_col!r}")

    cutoff = parse_yyyymmdd(curr_date)
    if cutoff is None:
        raise ValueError(f"invalid curr_date: {curr_date!r}")

    keep_mask = []
    latest: Optional[EffectiveAnnounceDate] = None
    for _, row in df.iterrows():
        period = normalize_report_period(row.get(rep_col))
        eff = effective_map.get(period) if period else None
        if eff is None and period:
            # Period only present on this table — resolve with its own ann if any.
            ann_cols = [
                c
                for c in (
                    "公告日期",
                    "实际公告日",
                    "f_ann_date",
                    "ann_date",
                    "NOTICE_DATE",
                    *ANNOUNCE_COL_CANDIDATES,
                )
                if c in df.columns
            ]
            seen_c = set()
            unique_c = [c for c in ann_cols if not (c in seen_c or seen_c.add(c))]
            anns = [
                row.get(c)
                for c in unique_c
                if row.get(c) is not None
                and not (isinstance(row.get(c), float) and pd.isna(row.get(c)))
            ]
            if anns:
                try:
                    eff = resolve_effective_announce_date(period, anns)
                except ValueError:
                    eff = None
        if eff is None:
            keep_mask.append(False)
            continue
        ok = eff.effective_date <= cutoff
        keep_mask.append(ok)
        if ok and (latest is None or eff.effective_date > latest.effective_date or (
            eff.effective_date == latest.effective_date and eff.report_period > latest.report_period
        )):
            latest = eff

    filtered = df.loc[keep_mask].copy()
    return filtered, latest


def filter_abstract_period_columns(
    abstract_df: pd.DataFrame,
    effective_map: dict[str, EffectiveAnnounceDate],
    curr_date: str,
    id_cols: Iterable[str] = ("选项", "指标"),
) -> tuple[pd.DataFrame, Optional[EffectiveAnnounceDate]]:
    """Filter financial-abstract wide period columns by effective announce date."""
    if abstract_df is None or abstract_df.empty:
        return abstract_df, None

    cutoff = parse_yyyymmdd(curr_date)
    if cutoff is None:
        raise ValueError(f"invalid curr_date: {curr_date!r}")

    id_col_set = {c for c in id_cols if c in abstract_df.columns}
    kept_period_cols: list[str] = []
    latest: Optional[EffectiveAnnounceDate] = None

    for col in abstract_df.columns:
        if col in id_col_set:
            continue
        period = normalize_report_period(col)
        if not period:
            # Non-period metric column — drop rather than risk undated values.
            continue
        eff = effective_map.get(period)
        if eff is None:
            # No sina mapping → cannot prove public-by-curr_date; drop.
            continue
        if eff.effective_date <= cutoff:
            kept_period_cols.append(col)
            if latest is None or eff.effective_date > latest.effective_date or (
                eff.effective_date == latest.effective_date
                and eff.report_period > latest.report_period
            ):
                latest = eff

    cols = [c for c in abstract_df.columns if c in id_col_set] + kept_period_cols
    return abstract_df.loc[:, cols].copy(), latest


# Shown to the model when a kept period used the statutory floor after YoY wipe.
DROPPED_YOY_PROMPT_NOTE = (
    "（该期首次公告日在数据源中已被后续财报刷新覆盖，"
    "此处按法定披露截止日估计，若公司当期实际逾期披露，"
    "可见时点可能晚于此估计数日至数周）"
)


def _period_kind_chinese_note(
    statement_kind: Optional[str], period_kind: str, report_period: str
) -> str:
    """Chinese semantic guidance note for period_kind."""
    token = normalize_report_period(report_period)
    md = token[4:] if token and len(token) >= 8 else ""
    if statement_kind in ("income", "cashflow"):
        if period_kind == "half_year_cumulative" or md == "0630":
            return "利润表/现金流量表半年度（0630）是1–6月累计值，禁止当作Q2单季使用（不是Q2单季）"
        elif period_kind == "nine_month_cumulative" or md == "0930":
            return "利润表/现金流量表三季度（0930）是1–9月累计值，禁止当作Q3单季使用（不是Q3单季）"
        elif period_kind == "annual_cumulative" or md == "1231":
            return "利润表/现金流量表年度（1231）是1–12月全年累计值，禁止当作Q4单季使用（不是Q4单季）"
        elif period_kind == "first_quarter" or md == "0331":
            return "利润表/现金流量表一季度（0331）是1–3月累计值（一季度单季）"
        return "利润表/现金流量表非标准报告期"
    elif statement_kind == "balance":
        if period_kind == "period_end_stock":
            return "资产负债表各报告期是期末点值（时点存量指标），不适用累计或单季流量口径"
        return "资产负债表非标准报告期"
    return "未知财务报表类型或报告期"


def financial_cutoff_header(
    latest: Optional[EffectiveAnnounceDate],
    curr_date: str,
    *,
    yoy_disclaimer: Optional[bool] = None,
    statement_kind: Optional[str] = None,
) -> str:
    """Prompt-visible cutoff line for financial injections.

    When the latest kept period (or an explicit ``yoy_disclaimer=True``) used
    path ``dropped_yoy_refresh``, append a clear estimate caveat so the model
    does not treat the statutory floor as an observed announce date.
    """
    if latest is None:
        return f"【财务数据】在 {curr_date} 及之前无已公开报告期"

    if statement_kind is not None:
        info = classify_financial_period_kind(latest.report_period, statement_kind)
        chinese_note = _period_kind_chinese_note(
            statement_kind, info.period_kind, latest.report_period
        )
        line = (
            f"【财务数据截至 {latest.report_period_label}】"
            f"（生效公告日 {latest.effective_date_str}，分析日 {curr_date}；"
            f"reported_period_label={info.reported_period_label}, "
            f"period_kind={info.period_kind}, "
            f"derivation_formula={info.derivation_formula}；"
            f"{chinese_note}）"
        )
    else:
        line = (
            f"【财务数据截至 {latest.report_period_label}】"
            f"（生效公告日 {latest.effective_date_str}，分析日 {curr_date}）"
        )

    show_note = (
        yoy_disclaimer
        if yoy_disclaimer is not None
        else latest.path == PATH_DROPPED_YOY_REFRESH
    )
    if show_note:
        line = f"{line}\n{DROPPED_YOY_PROMPT_NOTE}"
    return line



def periods_used_dropped_yoy(
    effective_map: dict[str, EffectiveAnnounceDate],
    report_periods: Iterable,
) -> bool:
    """True if any of the given report periods resolved via dropped_yoy_refresh."""
    for raw in report_periods:
        period = normalize_report_period(raw)
        if not period:
            continue
        eff = effective_map.get(period)
        if eff is not None and eff.path == PATH_DROPPED_YOY_REFRESH:
            return True
    return False


# ---------------------------------------------------------------------------
# Earnings-forecast (业绩预告) report-period selection for Eastmoney pools
# ---------------------------------------------------------------------------
#
# ``ak.stock_yjyg_em(date=YYYYMMDD)`` indexes by report-period end, not by
# calendar day. Querying the wrong period (e.g. always prior-year annual)
# yields "no forecast" when the live window is actually H1/Q1/Q3.
#
# Closed-window rule (实务口径; verify against current SSE/SZSE rules before
# changing). A period's Eastmoney 业绩预告 pool is treated as the active
# query target only after its forecast-disclosure window has closed:
#
# - Annual (YYYY1231): window closes Jan 31 of YYYY+1
#   (common board practice: annual performance forecast within ~1 month
#   after fiscal year end when required).
# - Q1 (YYYY0331): window closes Apr 15
# - H1 (YYYY0630): window closes Jul 15
# - Q3 (YYYY0930): window closes Oct 15
#
# Given analysis day D, pick the latest period whose window closed on or
# before D. Example: 2026-07-29 → 20260630 (not year-1 annual 20251231).

_FORECAST_WINDOW_SPECS: tuple[tuple[str, int, int, int], ...] = (
    # (period_md, close_month, close_day, year_offset_for_close)
    # year_offset_for_close: 0 = same calendar year as period year;
    # 1 = next calendar year (annual only).
    ("0331", 4, 15, 0),
    ("0630", 7, 15, 0),
    ("0930", 10, 15, 0),
    ("1231", 1, 31, 1),
)


def resolve_earnings_forecast_report_period(curr_date) -> str:
    """Return YYYYMMDD report period for the latest closed 业绩预告 window.

    Raises ValueError if ``curr_date`` cannot be parsed.
    """
    d = parse_yyyymmdd(curr_date)
    if d is None:
        raise ValueError(f"unparseable curr_date for earnings forecast: {curr_date!r}")

    candidates: list[tuple[date, str]] = []
    # Search a few surrounding years so Jan windows resolve correctly.
    for year in range(d.year - 2, d.year + 2):
        for period_md, cm, cd, y_off in _FORECAST_WINDOW_SPECS:
            period = f"{year}{period_md}"
            close = date(year + y_off, cm, cd)
            if close <= d:
                candidates.append((close, period))
    if not candidates:
        raise ValueError(
            f"no closed earnings-forecast window on or before {d.isoformat()}"
        )
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][1]

