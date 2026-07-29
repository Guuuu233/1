import os
import json
import pandas as pd
from datetime import date, timedelta, datetime
from typing import Annotated

SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]

def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    if save_path:
        data.to_csv(save_path)
        print(f"{tag} saved to {save_path}")


def get_current_date():
    return date.today().strftime("%Y-%m-%d")


def decorate_all_methods(decorator):
    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


def get_next_weekday(date):

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date


def take_latest(df: "pd.DataFrame", date_col: str, n: int = 1) -> "pd.DataFrame":
    """Return the latest ``n`` rows by ``date_col`` regardless of input order.

    External vendors may return ascending, descending, or unsorted frames.
    Always convert to datetime, sort descending, then take the first ``n`` rows.
    Rows with unparseable dates are dropped.

    Selection order is newest-first. For LLM-facing text, pass the result through
    :func:`chronological` so "later rows = later events".
    """
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return df
    if date_col not in df.columns:
        raise KeyError(
            f"take_latest: column {date_col!r} not in DataFrame columns {list(df.columns)}"
        )
    if n is None or int(n) <= 0:
        raise ValueError(f"take_latest: n must be positive, got {n!r}")

    out = df.copy()
    sort_key = "__take_latest_dt"
    out[sort_key] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[sort_key])
    if out.empty:
        return out.drop(columns=[sort_key], errors="ignore")
    out = out.sort_values(sort_key, ascending=False, kind="mergesort")
    out = out.drop(columns=[sort_key])
    return out.head(int(n)).reset_index(drop=True)


def chronological(df: "pd.DataFrame", date_col: str) -> "pd.DataFrame":
    """Sort rows ascending by ``date_col`` for prompt/display (later = more recent)."""
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return df
    if date_col not in df.columns:
        raise KeyError(
            f"chronological: column {date_col!r} not in DataFrame columns {list(df.columns)}"
        )
    out = df.copy()
    sort_key = "__chrono_dt"
    out[sort_key] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[sort_key])
    if out.empty:
        return out.drop(columns=[sort_key], errors="ignore")
    out = out.sort_values(sort_key, ascending=True, kind="mergesort")
    out = out.drop(columns=[sort_key])
    return out.reset_index(drop=True)


# ── Financial table prompt cleaning ──────────────────────────────────────────
# External financial frames are wide, sparse, and untrusted in column order.
# Always select by name, drop empty/near-empty columns, and never emit bare NaN.

MISSING_VALUE_MARKER = "【缺失】"

# Core four fields for statement quality gate (aliases → canonical label).
CORE_FINANCIAL_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "总资产": ("资产总计", "总资产", "资产合计"),
    "总负债": ("负债合计", "总负债", "负债总计"),
    "净资产": (
        "归属于母公司股东权益合计",
        "归属于母公司所有者权益合计",
        "所有者权益(或股东权益)合计",
        "所有者权益合计",
        "股东权益合计",
        "净资产",
    ),
    "归母净利润": (
        "归属于母公司所有者的净利润",
        "归属于母公司股东的净利润",
        "归属于母公司净利润",
        "归母净利润",
        "净利润(归母)",
    ),
}

# Preferred column order for LLM injection (name-based; never positional).
FINANCIAL_COLUMN_PRIORITIES: dict[str, tuple[str, ...]] = {
    "balance": (
        "报告日",
        "公告日期",
        "货币资金",
        "应收账款",
        "存货",
        "流动资产合计",
        "非流动资产合计",
        "资产总计",
        "短期借款",
        "应付账款",
        "流动负债合计",
        "非流动负债合计",
        "负债合计",
        "归属于母公司股东权益合计",
        "所有者权益(或股东权益)合计",
        "负债和所有者权益(或股东权益)总计",
    ),
    "income": (
        "报告日",
        "公告日期",
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
        "基本每股收益",
        "稀释每股收益",
    ),
    "cashflow": (
        "报告日",
        "公告日期",
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
        "期末现金及现金等价物余额",
    ),
    "abstract": ("选项", "指标"),
    "generic": (),
}

# ~2000 tokens for CJK/number-heavy markdown ≈ 3000-4000 chars.
DEFAULT_TABLE_PROMPT_MAX_CHARS = 3500
_NULL_RATIO_DROP = 0.80


def _is_nullish(value) -> bool:
    import pandas as pd

    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in {"nan", "none", "null", "<na>", "nat"}


def _series_null_ratio(series) -> float:
    if series is None or len(series) == 0:
        return 1.0
    nulls = sum(1 for value in series.tolist() if _is_nullish(value))
    return nulls / float(len(series))


def _resolve_column(columns, candidates: tuple[str, ...] | list[str]) -> str | None:
    col_set = {str(c): c for c in columns}
    for name in candidates:
        if name in col_set:
            return col_set[name]
    # Case-insensitive fallback.
    lower_map = {str(c).lower(): c for c in columns}
    for name in candidates:
        hit = lower_map.get(str(name).lower())
        if hit is not None:
            return hit
    return None


def resolve_core_financial_columns(columns) -> dict[str, str | None]:
    """Map canonical core labels to actual column names present in ``columns``."""
    return {
        label: _resolve_column(columns, aliases)
        for label, aliases in CORE_FINANCIAL_FIELD_ALIASES.items()
    }


def missing_core_financial_fields(columns) -> list[str]:
    resolved = resolve_core_financial_columns(columns)
    return [label for label, col in resolved.items() if col is None]


def _priority_column_order(columns, table_kind: str | None) -> list:
    cols = list(columns)
    kind = (table_kind or "generic").strip().lower()
    preferred = list(FINANCIAL_COLUMN_PRIORITIES.get(kind, ()))
    # Always pin identity-like columns first when present.
    for pin in ("报告日", "公告日期", "选项", "指标", "日期", "code", "symbol"):
        if pin not in preferred:
            preferred.insert(0, pin)

    ordered: list = []
    seen = set()
    for name in preferred:
        hit = _resolve_column(cols, (name,))
        if hit is not None and hit not in seen:
            ordered.append(hit)
            seen.add(hit)
    # Keep remaining columns by original name order (stable, not positional slice of values).
    for col in cols:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


def _drop_sparse_columns(
    df: "pd.DataFrame",
    null_ratio_drop: float = _NULL_RATIO_DROP,
    protect_cols: list | None = None,
):
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return df
    protected = set(protect_cols or [])
    keep = []
    for col in df.columns:
        if col in protected:
            keep.append(col)
            continue
        ratio = _series_null_ratio(df[col])
        if ratio >= 1.0:
            continue  # fully empty
        if ratio > null_ratio_drop:
            continue
        keep.append(col)
    if not keep:
        # Fall back to non-fully-empty columns so callers can still surface something
        # before the core-field gate decides failure.
        keep = [c for c in df.columns if _series_null_ratio(df[c]) < 1.0 or c in protected]
    if not keep:
        return df.iloc[0:0]
    return df.loc[:, keep]


def _replace_nullish_with_marker(df: "pd.DataFrame", marker: str = MISSING_VALUE_MARKER):
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    for col in out.columns:
        out[col] = [
            marker if _is_nullish(v) else v
            for v in out[col].tolist()
        ]
    return out


def _table_to_markdown(df: "pd.DataFrame") -> str:
    if df is None or getattr(df, "empty", True):
        return ""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _fit_markdown_budget(
    df: "pd.DataFrame",
    *,
    max_chars: int,
    protected_cols: list,
) -> tuple["pd.DataFrame", bool]:
    """Drop lowest-priority (rightmost non-protected) columns until under budget."""
    if df is None or getattr(df, "empty", True):
        return df, False
    work = df
    text = _table_to_markdown(work)
    if len(text) <= max_chars:
        return work, False

    protected = set(protected_cols or [])
    # Prefer dropping trailing non-protected columns first.
    cols = list(work.columns)
    droppable = [c for c in reversed(cols) if c not in protected]
    truncated = False
    for col in droppable:
        if len(text) <= max_chars:
            break
        cols = [c for c in cols if c != col]
        if not cols:
            break
        work = work.loc[:, cols]
        text = _table_to_markdown(work)
        truncated = True

    # If still over budget, drop oldest rows (assume row 0 is newest for financials).
    while len(text) > max_chars and len(work) > 1:
        work = work.iloc[:-1, :]
        text = _table_to_markdown(work)
        truncated = True

    return work, truncated


def shrink_table(
    df: "pd.DataFrame",
    *,
    max_rows: int = 12,
    table_kind: str | None = "generic",
    require_core_fields: bool = False,
    max_prompt_chars: int = DEFAULT_TABLE_PROMPT_MAX_CHARS,
    null_ratio_drop: float = _NULL_RATIO_DROP,
) -> str:
    """Clean a vendor DataFrame for LLM injection.

    Rules:
    - Select columns by name (priority list + remaining names); never positional iloc slice.
    - Drop all-empty columns and columns with null ratio > ``null_ratio_drop``.
    - Replace residual null/NaN with an explicit missing marker (no bare nan).
    - If ``require_core_fields`` and ≥2 of the four core financial fields are absent,
      return an explicit failure string instead of a partial table.
    - Enforce a prompt size budget; if truncated, append an explicit notice.
    """
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return "【数据获取失败】表格为空，本项不可用。"

    work = df.copy()
    # Normalize column labels to strings for stable name selection.
    work.columns = [str(c) for c in work.columns]

    if max_rows is not None and int(max_rows) > 0:
        work = work.head(int(max_rows))

    # Protect identity + core columns from sparse-drop so residual nulls become
    # explicit markers instead of silent column deletion.
    pre_protected: list = []
    for pin in ("报告日", "公告日期", "选项", "指标"):
        hit = _resolve_column(work.columns, (pin,))
        if hit is not None:
            pre_protected.append(hit)
    for col in resolve_core_financial_columns(work.columns).values():
        if col is not None and col not in pre_protected:
            pre_protected.append(col)

    work = _drop_sparse_columns(
        work, null_ratio_drop=null_ratio_drop, protect_cols=pre_protected
    )
    if work is None or work.empty:
        return "【数据获取失败】表格在清洗后无可用列，本项不可用。"

    ordered = _priority_column_order(work.columns, table_kind)
    work = work.loc[:, [c for c in ordered if c in work.columns]]

    if require_core_fields:
        missing = missing_core_financial_fields(work.columns)
        if len(missing) >= 2:
            return (
                "【数据获取失败】关键财务字段缺失："
                + "、".join(missing)
                + "。表格不可用，不得据此判断财务健康度。"
            )

    protected = list(pre_protected)

    work = _replace_nullish_with_marker(work)
    fitted, truncated = _fit_markdown_budget(
        work, max_chars=int(max_prompt_chars), protected_cols=protected
    )
    text = _table_to_markdown(fitted)
    if not text.strip():
        return "【数据获取失败】表格渲染结果为空，本项不可用。"
    if truncated:
        text = (
            text
            + "\n\n【已截断，保留核心字段】表格超过注入长度上限，"
            + "已按列名优先级保留核心字段，省略低优先级列/较旧报告期。"
        )
    # Final safety: never leak bare nan tokens after marker substitution.
    if "nan" in text.lower():
        import re as _re

        text = _re.sub(r"(?i)\bnan\b", MISSING_VALUE_MARKER, text)
    return text
