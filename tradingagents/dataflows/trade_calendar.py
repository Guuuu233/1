from __future__ import annotations

import bisect
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

CN_TZ = ZoneInfo("Asia/Shanghai")

logger = logging.getLogger(__name__)

# If the newest cached trade day is older than the request by more than this
# many calendar days, treat the calendar as unavailable (do not silently
# snap far into the past).
MAX_CALENDAR_STALENESS_DAYS = 10

# Cache TTL for the sina trade-date table (seconds). Refresh at most once per day.
_TRADE_DATE_TTL_SECONDS = 24 * 60 * 60
_TRADE_DATES_CACHE: dict[str, Any] = {
    "loaded_at": 0.0,
    "dates": None,  # list[date] ascending
    "dates_set": None,  # set[date]
}


class TradeCalendarUnavailableError(RuntimeError):
    """Raised when the CN trade calendar cannot be loaded or has no usable dates."""


class DateDataUnavailable(Exception):
    """Raised by a date-scoped fetch when that specific day has no usable data yet."""


class DuplicateBarConflictError(ValueError):
    """Raised when a daily series has same-date rows with conflicting OHLCV/Volume.

    Subclasses ``ValueError`` so provider-level callers that treat normalization
    failure as a ``ValueError`` keep that contract, while the router can
    distinguish this data-integrity refusal from generic runtime errors.
    """


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def cn_today_str() -> str:
    return now_cn().date().strftime("%Y-%m-%d")


SNAPSHOT_ONLY_REFUSAL = (
    "该数据源仅提供当前快照，无法用于历史日期分析，本项不可用"
)


def is_historical_analysis_date(curr_date: str | None) -> bool:
    """True when curr_date is a calendar day strictly before Asia/Shanghai today.

    Missing / unparseable curr_date is treated as non-historical (live path).
    Callers that must not run undated on historical analyses should require
    curr_date at the API boundary separately (commit 3c).
    """
    if not curr_date:
        return False
    try:
        d = _parse_date(str(curr_date))
    except Exception:
        return False
    return d < now_cn().date()


def unavailable_analysis_date_reason(curr_date: str | None) -> str | None:
    """Return an explicit refusal when the as-of date is missing/unparseable/future.

    None means the date is usable as an analysis as-of. Callers must refuse
    (never fall back to a live provider) when a reason is returned.
    """
    if curr_date is None or not str(curr_date).strip():
        return "【数据获取失败】缺少分析日期（as-of），不得回退到 live 数据源，本项不可用。"
    try:
        d = _parse_date(str(curr_date))
    except (TypeError, ValueError) as exc:
        logger.warning("Unparseable analysis date %r: %s", curr_date, exc)
        return f"【数据获取失败】分析日期无法解析：{curr_date!r}，本项不可用。"
    if d > now_cn().date():
        return f"【数据获取失败】分析日期 {_format_date(d)} 晚于当前日期，拒绝未来数据，本项不可用。"
    return None


def drop_incomplete_today_bar(
    df: pd.DataFrame,
    date_col: str,
    end_date: str,
) -> pd.DataFrame:
    """Remove today's intraday bar from a completed daily series.

    Shared by AkShare / Investoday / BaoStock so the completed-bar rule is
    consistent at the provider/router convergence. Only drops when ``end_date``
    is today and the CN market has not closed yet.
    """
    if df is None or getattr(df, "empty", True) or date_col not in df.columns:
        return df
    end_dt = pd.to_datetime(end_date, errors="coerce")
    today = pd.to_datetime(cn_today_str(), errors="coerce")
    if pd.isna(end_dt) or pd.isna(today) or end_dt.normalize() != today.normalize():
        return df
    if cn_market_phase() not in ("pre_open", "in_session", "lunch_break"):
        return df
    out = df.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    return out.loc[dates.dt.normalize() != today.normalize()].reset_index(drop=True)


def dedupe_daily_bars(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """Deterministic duplicate policy for completed daily bars.

    Identical duplicate rows (same date and same OHLCV values) are collapsed —
    the choice is order-independent because the rows are identical. A date with
    two rows whose values differ is rejected with ``ValueError``: without a
    timestamp or quality basis there is no deterministic way to choose, so
    callers must fail explicitly instead of silently picking vendor row order.
    """
    if df is None or getattr(df, "empty", True) or date_col not in df.columns:
        return df
    cols = [date_col, *value_cols]
    out = df.copy()
    out = out.sort_values(date_col, kind="stable").reset_index(drop=True)
    out = out.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)
    conflicts = out[out.duplicated(subset=[date_col], keep=False)]
    if not conflicts.empty:
        dates = sorted(
            {pd.Timestamp(d).strftime("%Y-%m-%d") for d in conflicts[date_col]}
        )
        raise DuplicateBarConflictError(
            "duplicate daily bars with conflicting OHLCV, cannot choose "
            f"deterministically for date(s): {', '.join(dates)}"
        )
    return out


def snapshot_historical_refusal(
    curr_date: str | None,
    *,
    source_label: str = "",
) -> str | None:
    """If analysis date is historical, return a fixed refusal string; else None.

    Prompt text is stable so models and e2e scanners can detect snapshot refuse.
    """
    if not is_historical_analysis_date(curr_date):
        return None
    label = f"{source_label}：" if source_label else ""
    return f"【数据获取失败】{label}{SNAPSHOT_ONLY_REFUSAL}"


def _parse_date(date_str: str) -> date:
    text = str(date_str).strip()
    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return datetime.strptime(text, "%Y-%m-%d").date()


def _format_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def clear_cn_trade_date_cache() -> None:
    """Test helper: drop the in-process trade-date cache."""
    _TRADE_DATES_CACHE["loaded_at"] = 0.0
    _TRADE_DATES_CACHE["dates"] = None
    _TRADE_DATES_CACHE["dates_set"] = None


def _fetch_cn_trade_dates_from_akshare() -> list[date]:
    import akshare as ak  # type: ignore

    df = ak.tool_trade_date_hist_sina()
    if df is None or df.empty or "trade_date" not in df.columns:
        raise TradeCalendarUnavailableError(
            "交易日历不可用：akshare.tool_trade_date_hist_sina 返回空表"
        )
    dates = sorted(
        {
            pd_dt.date()
            for pd_dt in pd.to_datetime(df["trade_date"], errors="coerce")
            if str(pd_dt) != "NaT"
        }
    )
    if not dates:
        raise TradeCalendarUnavailableError(
            "交易日历不可用：akshare.tool_trade_date_hist_sina 无有效日期"
        )
    return dates


def _fetch_cn_trade_dates_from_fuyao() -> list[date]:
    """同花顺 fuyao 交易日历（近一年）作 akshare 失败后的在线对照/备用源。

    仅在配置了 ``FUYAO_API_KEY`` 时尝试；失败抛异常，由调用方决定兜底。
    近一年窗口不足以覆盖所有历史查询，因此仅作 fallback，不替代主源。
    """
    api_key = os.getenv("FUYAO_API_KEY", "").strip()
    if not api_key:
        raise TradeCalendarUnavailableError(
            "交易日历不可用：未配置 FUYAO_API_KEY（无 fuyao 在线对照源）"
        )
    from .providers.cn_fuyao_provider import fetch_trading_days_ths

    raw = fetch_trading_days_ths(api_key)
    dates = sorted(
        {
            datetime.strptime(d, "%Y%m%d").date()
            for d in raw
            if re.fullmatch(r"\d{8}", d)
        }
    )
    if not dates:
        raise TradeCalendarUnavailableError(
            "交易日历不可用：fuyao 交易日历无有效日期"
        )
    return dates


def _load_cn_trade_dates() -> tuple[list[date], set[date]]:
    """Load/cached CN trading dates. On total failure returns empty containers.

    Callers that must not silently degrade (normalize / fallback) must check for
    empty and raise :class:`TradeCalendarUnavailableError` themselves, or use
    :func:`require_cn_trade_dates`.
    """
    now = time.time()
    cached_dates = _TRADE_DATES_CACHE["dates"]
    loaded_at = float(_TRADE_DATES_CACHE["loaded_at"] or 0.0)
    if cached_dates is not None and (now - loaded_at) < _TRADE_DATE_TTL_SECONDS:
        return cached_dates, _TRADE_DATES_CACHE["dates_set"]

    try:
        dates = _fetch_cn_trade_dates_from_akshare()
    except Exception:
        try:
            dates = _fetch_cn_trade_dates_from_fuyao()
        except Exception as exc:
            logger.debug("fuyao trading-days fallback unavailable: %s", exc)
            # Keep serving a previously successful calendar past TTL if present.
            if cached_dates is not None:
                return cached_dates, _TRADE_DATES_CACHE["dates_set"]
            return [], set()

    dates_set = set(dates)
    _TRADE_DATES_CACHE["dates"] = dates
    _TRADE_DATES_CACHE["dates_set"] = dates_set
    _TRADE_DATES_CACHE["loaded_at"] = now
    return dates, dates_set


def require_cn_trade_dates() -> tuple[list[date], set[date]]:
    """Return the CN trade calendar or raise an explicit unavailable error."""
    dates, dates_set = _load_cn_trade_dates()
    if not dates:
        raise TradeCalendarUnavailableError(
            "交易日历不可用：无法从 akshare.tool_trade_date_hist_sina 加载有效交易日"
        )
    return dates, dates_set


def is_cn_symbol(symbol: str) -> bool:
    s = symbol.strip().upper()
    return bool(re.match(r"^\d{6}(\.(SH|SZ|SS))?$", s))


def is_cn_trading_day(date_str: str, allow_weekday_fallback: bool = False) -> bool:
    """Return whether ``date_str`` is a CN trading day.

    When the trade calendar cannot be loaded:
    - ``allow_weekday_fallback=False`` (default): hard-fail with
      :class:`TradeCalendarUnavailableError`. Date-query paths must not silently
      treat Mon–Fri as trading days (Spring Festival / National Day gaps).
    - ``allow_weekday_fallback=True``: degrade to weekday rule and emit a WARNING.
      Call sites that opt in must pass True explicitly so the choice is visible.
    """
    d = _parse_date(date_str)
    dates, dates_set = _load_cn_trade_dates()
    if dates:
        return d in dates_set
    if allow_weekday_fallback:
        logger.warning(
            "CN trade calendar unavailable; is_cn_trading_day(%s) using Mon-Fri fallback",
            date_str,
        )
        return d.weekday() < 5
    raise TradeCalendarUnavailableError(
        "交易日历不可用：无法判断是否为交易日（is_cn_trading_day）"
    )


def previous_cn_trading_day(
    date_str: str, allow_weekday_fallback: bool = False
) -> str:
    """Return the latest trading day strictly before ``date_str``.

    Soft Mon–Fri rollback only when ``allow_weekday_fallback=True`` (explicit).
    """
    d = _parse_date(date_str)
    dates, _ = _load_cn_trade_dates()
    if dates:
        idx = bisect.bisect_left(dates, d) - 1
        if idx >= 0:
            return _format_date(dates[idx])
        if not allow_weekday_fallback:
            raise TradeCalendarUnavailableError(
                f"交易日历不可用：无不早于 {_format_date(d)} 之前的交易日"
            )
    elif not allow_weekday_fallback:
        raise TradeCalendarUnavailableError(
            "交易日历不可用：无法计算前一交易日（previous_cn_trading_day）"
        )

    logger.warning(
        "CN trade calendar unavailable or exhausted; previous_cn_trading_day(%s) using Mon-Fri fallback",
        date_str,
    )
    cur = d
    while True:
        cur = date.fromordinal(cur.toordinal() - 1)
        if cur.weekday() < 5:
            return _format_date(cur)


def _ensure_calendar_not_stale_for(request_day: date, dates: list[date]) -> None:
    """Reject a calendar whose newest day is too far before the request date."""
    if not dates:
        raise TradeCalendarUnavailableError("交易日历不可用：无有效交易日")
    newest = dates[-1]
    lag = (request_day - newest).days
    if lag > MAX_CALENDAR_STALENESS_DAYS:
        raise TradeCalendarUnavailableError(
            "交易日历不可用：缓存最新交易日为 "
            f"{_format_date(newest)}，早于请求日 {_format_date(request_day)} "
            f"{lag} 个自然日（阈值 {MAX_CALENDAR_STALENESS_DAYS}）"
        )


def normalize_to_trading_day(date_str: str) -> str:
    """Snap ``date_str`` to the latest trading day on or before it.

    Never rounds forward (would leak future data into historical backtests).
    Raises :class:`TradeCalendarUnavailableError` if the calendar cannot be loaded,
    or if the newest cached trade day is more than
    :data:`MAX_CALENDAR_STALENESS_DAYS` before the request date.
    """
    d = _parse_date(date_str)
    dates, dates_set = require_cn_trade_dates()
    _ensure_calendar_not_stale_for(d, dates)
    if d in dates_set:
        result = d
    else:
        idx = bisect.bisect_right(dates, d) - 1
        if idx < 0:
            raise TradeCalendarUnavailableError(
                f"交易日历不可用：无不晚于 {_format_date(d)} 的交易日"
            )
        result = dates[idx]

    if result > d:
        raise AssertionError(
            f"normalize_to_trading_day must not round forward: input={_format_date(d)} result={_format_date(result)}"
        )
    assert result <= d, "normalize_to_trading_day invariant: result <= input"
    return _format_date(result)


def trading_days_back(
    date_str: str,
    count: int,
    *,
    start_offset: int = 0,
) -> list[str]:
    """Return ``count`` trading days ending at normalize(date) - start_offset, newest first."""
    if count <= 0:
        raise ValueError(f"trading_days_back: count must be positive, got {count!r}")
    if start_offset < 0:
        raise ValueError(f"trading_days_back: start_offset must be >= 0, got {start_offset!r}")

    dates, _ = require_cn_trade_dates()
    base = _parse_date(normalize_to_trading_day(date_str))
    end_idx = bisect.bisect_left(dates, base)
    if end_idx >= len(dates) or dates[end_idx] != base:
        # base must be in the calendar after normalize
        end_idx = bisect.bisect_right(dates, base) - 1
    end_idx -= start_offset
    if end_idx < 0:
        raise TradeCalendarUnavailableError(
            f"交易日历不可用：从 {date_str} 回退 start_offset={start_offset} 后无交易日"
        )
    start_idx = max(0, end_idx - count + 1)
    window = dates[start_idx : end_idx + 1]
    return [_format_date(x) for x in reversed(window)]


@dataclass
class DateFetchResult:
    ok: bool
    data: Any = None
    as_of: Optional[str] = None
    request_date: Optional[str] = None
    attempted: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def date_header(self) -> str:
        """LLM-visible actual data date, including rollback note when needed."""
        if not self.as_of:
            return ""
        req = self.request_date or self.as_of
        if self.as_of == req:
            return f"【数据日期】{self.as_of}"
        return (
            f"【数据日期】{self.as_of}"
            f"（请求 {req}，该日数据尚未发布，已回退）"
        )


def fetch_with_date_fallback(
    fetch_fn: Callable[[str], Any],
    date_str: str,
    *,
    max_back: int = 5,
    start_offset: int = 0,
) -> DateFetchResult:
    """Try ``fetch_fn(day)`` over a backward trading-day window.

    ``fetch_fn`` should return data on success and raise
    :class:`DateDataUnavailable` (or any Exception) when that day should be
    skipped. All failures produce an error that includes the attempted range.
    """
    request_date = _format_date(_parse_date(date_str))
    try:
        candidates = trading_days_back(
            request_date, max_back, start_offset=start_offset
        )
    except TradeCalendarUnavailableError as exc:
        return DateFetchResult(
            ok=False,
            request_date=request_date,
            error=str(exc),
        )
    except Exception as exc:
        return DateFetchResult(
            ok=False,
            request_date=request_date,
            error=f"交易日历不可用：{type(exc).__name__}: {exc}",
        )

    last_err: Optional[str] = None
    attempted: list[str] = []
    for day in candidates:
        attempted.append(day)
        try:
            data = fetch_fn(day)
            return DateFetchResult(
                ok=True,
                data=data,
                as_of=day,
                request_date=request_date,
                attempted=attempted,
            )
        except DateDataUnavailable as exc:
            last_err = str(exc) or type(exc).__name__
            continue
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            continue

    first = attempted[0] if attempted else request_date
    last = attempted[-1] if attempted else request_date
    detail = f"：{last_err}" if last_err else ""
    return DateFetchResult(
        ok=False,
        request_date=request_date,
        attempted=attempted,
        error=(
            f"已尝试 {first} 至 {last} 共 {len(attempted)} 个交易日，均无数据{detail}"
        ),
    )


def cn_market_phase(now: datetime | None = None) -> str:
    now_dt = now or now_cn()
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=CN_TZ)
    else:
        now_dt = now_dt.astimezone(CN_TZ)

    today = now_dt.date().strftime("%Y-%m-%d")
    if not is_cn_trading_day(today, allow_weekday_fallback=True):
        return "closed"

    t = now_dt.time()
    if t < dt_time(9, 30):
        return "pre_open"
    if dt_time(9, 30) <= t < dt_time(11, 30):
        return "in_session"
    if dt_time(11, 30) <= t < dt_time(13, 0):
        return "lunch_break"
    if dt_time(13, 0) <= t < dt_time(15, 0):
        return "in_session"
    return "post_close"


def resolve_cn_analysis_date(
    requested_date: str | None = None,
    *,
    explicit: bool = False,
    now: datetime | None = None,
) -> str:
    """Resolve an analysis date without treating an unfinished day as closed data.

    An explicitly supplied date is only normalized for weekends/holidays; a
    trading-day request remains that date even during pre-open or intraday
    analysis so the caller can report the requested day's data gap.  When no
    date was supplied, the CN market phase determines the default: before the
    close, use the latest completed trading day strictly before today; after
    the close, use today's trading day.  Every result is sourced from the
    trading calendar and never rounded forward.
    """
    now_dt = now or now_cn()
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=CN_TZ)
    else:
        now_dt = now_dt.astimezone(CN_TZ)

    raw = str(requested_date or "").strip()
    if explicit and raw:
        return normalize_to_trading_day(raw)

    today = _format_date(now_dt.date())
    phase = cn_market_phase(now_dt)
    if phase in {"pre_open", "in_session", "lunch_break"}:
        return previous_cn_trading_day(today)
    return normalize_to_trading_day(today)


def cn_no_data_reason(date_str: str) -> str:
    if not is_cn_trading_day(date_str, allow_weekday_fallback=True):
        return "N/A：非交易日（A股休市）"

    today = cn_today_str()
    if date_str == today:
        phase = cn_market_phase()
        if phase == "pre_open":
            return "N/A：今日尚未开盘"
        if phase in ("in_session", "lunch_break"):
            return "N/A：今日盘中，日线未收盘（可参考实时价）"
        if phase == "post_close":
            return "N/A：今日已收盘，数据源尚未更新"

    return "N/A：该交易日暂无数据（可能停牌或数据延迟）"
