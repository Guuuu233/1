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
        if report_col not in df.columns or announce_col not in df.columns:
            continue
        for _, row in df.iterrows():
            period = normalize_report_period(row.get(report_col))
            if not period:
                continue
            period_to_anns.setdefault(period, []).append(row.get(announce_col))

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
    if report_col not in df.columns:
        raise ValueError(f"missing report column {report_col!r}")

    cutoff = parse_yyyymmdd(curr_date)
    if cutoff is None:
        raise ValueError(f"invalid curr_date: {curr_date!r}")

    keep_mask = []
    latest: Optional[EffectiveAnnounceDate] = None
    for _, row in df.iterrows():
        period = normalize_report_period(row.get(report_col))
        eff = effective_map.get(period) if period else None
        if eff is None and period:
            # Period only present on this table — resolve with its own ann if any.
            ann = row.get("公告日期") if "公告日期" in df.columns else None
            try:
                eff = resolve_effective_announce_date(period, [ann] if ann is not None else [])
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


def financial_cutoff_header(
    latest: Optional[EffectiveAnnounceDate],
    curr_date: str,
    *,
    yoy_disclaimer: Optional[bool] = None,
) -> str:
    """Prompt-visible cutoff line for financial injections.

    When the latest kept period (or an explicit ``yoy_disclaimer=True``) used
    path ``dropped_yoy_refresh``, append a clear estimate caveat so the model
    does not treat the statutory floor as an observed announce date.
    """
    if latest is None:
        return f"【财务数据】在 {curr_date} 及之前无已公开报告期"
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
