import logging
import math
import re
import time
import threading
import contextvars
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

import pandas as pd
from stockstats import wrap

from .base import BaseMarketDataProvider, DataResult
from ..trade_calendar import (
    DateDataUnavailable,
    DuplicateBarConflictError,
    cn_no_data_reason,
    cn_today_str,
    dedupe_daily_bars,
    drop_incomplete_today_bar,
    fetch_with_date_fallback,
    is_cn_trading_day,
    is_historical_analysis_date,
    snapshot_historical_refusal,
)
from ..utils import (
    chronological,
    format_hist_csv,
    safe_float,
    shrink_table,
    slice_hist_df,
    take_latest,
)
from ..vendor_result import (
    VendorEmpty,
    VendorFail,
    VendorRefuse,
    result_to_prompt,
)
from ..fund_flow_evidence import (
    FundFlowText,
    build_consensus_evidence,
    build_em_evidence,
    build_gap_meta,
    build_provider_text,
    build_sina_evidence,
    build_ths_evidence,
)
from ..financial_announce import (
    build_effective_announce_map,
    filter_abstract_period_columns,
    filter_financial_df_by_effective_announce,
    financial_cutoff_header,
    format_report_period_label,
    parse_yyyymmdd,
    periods_used_dropped_yoy,
    resolve_earnings_forecast_report_period,
)

_provider_logger = logging.getLogger(__name__)


# ── akshare 并发控制 ──
# 总并发上限 5（防反爬 + akshare 全局状态安全）
# 定时任务最多占 3 个槽位，保证前端至少有 2 个槽位可用
#
# 关键设计：僵尸线程回收
# _run_job 超时后不会 cancel 内部线程（避免 cancel 卡在 to_thread），
# 导致僵尸线程可能永远持有 semaphore permit。_AkshareLock 通过追踪每个
# permit 的持有时间，在超过 STALE_TIMEOUT 后自动回收，防止锁被耗尽。

_is_scheduled_task: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "is_scheduled_task", default=False,
)


def set_scheduled_task_context(value: bool = True) -> contextvars.Token:
    """标记当前上下文为定时任务（会通过 asyncio.to_thread 自动传播到工作线程）"""
    return _is_scheduled_task.set(value)


import logging as _logging

_lock_logger = _logging.getLogger(__name__)


class _AkshareLock:
    """akshare 并发锁：前端优先 + 僵尸线程自动回收。

    - 总并发上限 ``total``（防反爬）
    - 定时任务额外受 ``scheduled_max`` 限制，为前端保留带宽
    - 持锁超过 ``stale_timeout`` 秒的线程视为僵尸，permit 被自动回收
    - 僵尸线程最终退出 ``with`` 块时不会 double-release（已被回收）
    """

    ACQUIRE_TIMEOUT = 60   # 等待 slot 的最大秒数
    STALE_TIMEOUT = 120    # 单次 akshare 调用不应超过 2 分钟，超过视为僵尸

    def __init__(self, total: int = 5, scheduled_max: int = 3):
        self._total = threading.Semaphore(total)
        self._scheduled = threading.Semaphore(scheduled_max)
        self._holders: dict[int, tuple[float, bool]] = {}   # tid -> (mono_time, is_scheduled)
        self._mu = threading.Lock()

    # ── 僵尸回收 ──

    def _reclaim_stale(self) -> int:
        """回收超时持有者的 permit，返回回收数量。"""
        now = time.monotonic()
        reclaimed = 0
        with self._mu:
            stale = [
                (tid, is_sched)
                for tid, (t, is_sched) in self._holders.items()
                if now - t > self.STALE_TIMEOUT
            ]
            for tid, is_sched in stale:
                del self._holders[tid]
                self._total.release()
                if is_sched:
                    self._scheduled.release()
                reclaimed += 1
        if reclaimed:
            _lock_logger.warning("[AkshareLock] reclaimed %d stale permits from zombie threads", reclaimed)
        return reclaimed

    # ── context manager ──

    def _acquire_or_reclaim(self, sem: threading.Semaphore, label: str) -> None:
        """尝试获取 semaphore，超时后回收僵尸再重试一次。"""
        if sem.acquire(timeout=self.ACQUIRE_TIMEOUT):
            return
        self._reclaim_stale()
        if sem.acquire(timeout=10):
            return
        raise TimeoutError(f"akshare {label} slot acquire timeout after reclaim")

    def __enter__(self):
        is_scheduled = _is_scheduled_task.get(False)
        try:
            if is_scheduled:
                self._acquire_or_reclaim(self._scheduled, "scheduled")
                try:
                    self._acquire_or_reclaim(self._total, "total")
                except BaseException:
                    self._scheduled.release()
                    raise
            else:
                self._acquire_or_reclaim(self._total, "total")
        except TimeoutError:
            _lock_logger.error("[AkshareLock] acquire timeout (is_scheduled=%s)", is_scheduled)
            raise
        with self._mu:
            self._holders[threading.get_ident()] = (time.monotonic(), is_scheduled)
        return self

    def __exit__(self, *_exc_info):
        tid = threading.get_ident()
        with self._mu:
            info = self._holders.pop(tid, None)
        if info is not None:
            _, is_scheduled = info
            self._total.release()
            if is_scheduled:
                self._scheduled.release()
        # info is None → permit 已被 _reclaim_stale 回收，不 double-release


AKSHARE_CALL_LOCK = _AkshareLock(total=5, scheduled_max=3)


# ── 新浪历史资金流（Source 2.5）─────────────────────────────────────────────
# akshare 无此接口封装，直调新浪 quotes_service JSON API。历史分析日东财被限流时
# 用它提供逐日资金流；opendate <= curr_date 过滤，防前视纪律不变。
_SINA_HIST_FUND_FLOW_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "MoneyFlow.ssl_qsfx_zjlrqs?page=1&num={num}&sort=opendate&asc=0&daima={daima}"
)
_SINA_HIST_FUND_FLOW_HEADERS = {
    "Referer": "https://finance.sina.com.cn/",
    "User-Agent": "Mozilla/5.0",
}
_SINA_HIST_FUND_FLOW_TIMEOUT = 10  # 秒
_SINA_HIST_FUND_FLOW_FETCH = 20  # 请求行数：取足够多，再按 curr_date 截断
_SINA_HIST_FUND_FLOW_SHOW = 5  # 展示最近 N 日（与东财版“近5日”对齐）
_SINA_HIST_CORE_AMOUNT_FIELDS = ("netamount", "r0_net")

# ── 东方财富直连历史资金流（Source 2）────────────────────────────────────────
# AkShare wraps this same family of data, but its request path can fail on
# Python TLS/fingerprint issues.  Keep the direct adapter deliberately narrow:
# f51 is the measurement date and f52 is the only verified canonical amount.
# f53-f56 are retained as raw discovery values when the vendor returns them;
# trailing fields are not required and are never given fabricated semantics.
_EASTMONEY_DIRECT_FUND_FLOW_URL = (
    "https://push2his.eastmoney.com/api/qt/stock/fflow/kline/get"
)
_EASTMONEY_DIRECT_FUND_FLOW_HEADERS = {
    "Referer": "https://data.eastmoney.com/",
    "User-Agent": "Mozilla/5.0",
}
_EASTMONEY_DIRECT_FUND_FLOW_TIMEOUT = 10
_EASTMONEY_DIRECT_FUND_FLOW_FETCH = 120
_EASTMONEY_DIRECT_FUND_FLOW_FIELDS1 = "f1,f2,f3,f7"
_EASTMONEY_DIRECT_FUND_FLOW_FIELDS2 = (
    "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63"
)
_EASTMONEY_DIRECT_FIELD_MAPPING = {
    "f51": "measurement_date",
    "f52": "r0_net",
    "f53": "raw_discovery_only",
    "f54": "raw_discovery_only",
    "f55": "raw_discovery_only",
    "f56": "raw_discovery_only",
}
_EASTMONEY_DIRECT_DISCOVERY_FIELDS = tuple(
    f"f{field_number}" for field_number in range(53, 57)
)
_FUND_AMOUNT_TEXT_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*(?:万亿|亿元|万元|亿|万)?$"
)


def _sina_decimal(value) -> Decimal | None:
    """Parse a finite provider amount without introducing binary-float error."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _sina_amount_yi_decimal(value) -> Decimal | None:
    """Convert a raw-yuan Sina amount to exact 亿元."""
    parsed = _sina_decimal(value)
    return None if parsed is None else parsed / Decimal("100000000")


def _sina_amount_yi(value) -> str:
    """Format a raw-yuan amount as 亿元 (2 dp); empty/invalid → ''."""
    amount = _sina_amount_yi_decimal(value)
    if amount is None:
        return ""
    return f"{amount.quantize(Decimal('0.01')):.2f}"


def _sina_ratio_pct(value) -> str:
    """Format a ratio fraction as a percent string; empty/invalid → ''."""
    f = safe_float(value)
    if f is None:
        return ""
    return f"{round(f * 100, 2):.2f}%"


def _usable_fund_amount_text(value) -> str | None:
    """Return a usable fund amount while preserving legal unit-bearing text."""
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    numeric = safe_float(value)
    if numeric is not None and math.isfinite(numeric):
        return text
    if _FUND_AMOUNT_TEXT_RE.fullmatch(text):
        return text
    return None


def _fund_flow_failure_category(error: object) -> str:
    """Classify fallback errors without exposing provider exception payloads."""
    text = str(error or "").lower()
    if any(
        token in text
        for token in (
            "http_status",
            "timeout",
            "request:",
            "connectionerror",
            "remotedisconnected",
        )
    ):
        return "transport"
    if any(
        token in text
        for token in (
            "json_decode",
            "json_shape",
            "rc_",
            "data_missing",
            "klines_",
        )
    ):
        return "envelope"
    if any(
        token in text
        for token in (
            "invalid_f52",
            "malformed",
            "field_count",
            "no_usable_rows",
            "no_current_day_row",
            "duplicate_date",
            "curr_date_invalid",
            "non_trading_date",
            "curr_date_not_cn_trading_day",
            "missing",
            "not parseable",
        )
    ):
        return "validation"
    if any(token in text for token in ("empty", "unavailable", "no rows")):
        return "availability"
    return "provider"


class CnAkshareProvider(BaseMarketDataProvider):
    """A-share provider backed by AkShare."""

    INDICATOR_DESCRIPTIONS = {
        "close_50_sma": (
            "50 日均线（SMA）：中期趋势指标。"
            "用途：识别趋势方向，并作为动态支撑/阻力参考。"
        ),
        "close_200_sma": (
            "200 日均线（SMA）：长期趋势基准。"
            "用途：确认大级别趋势，并辅助识别金叉/死叉结构。"
        ),
        "close_10_ema": (
            "10 日指数均线（EMA）：短期响应更快。"
            "用途：捕捉短线动量变化与潜在入场时机。"
        ),
        "macd": "MACD：趋势与动量综合指标。",
        "macds": "MACD 信号线（Signal）。",
        "macdh": "MACD 柱状图（Histogram）。",
        "rsi": "RSI：衡量超买/超卖的动量指标。",
        "boll": "布林中轨（20 日均线）。",
        "boll_ub": "布林上轨。",
        "boll_lb": "布林下轨。",
        "atr": "ATR：真实波动幅度均值，用于波动与风控。",
        "vwma": "VWMA：成交量加权均线。",
        "mfi": "MFI：资金流量指标。",
    }

    @property
    def name(self) -> str:
        return "cn_akshare"

    def _ak(self):
        try:
            import akshare as ak  # type: ignore
        except ImportError as exc:
            raise NotImplementedError(
                "cn_akshare requires 'akshare'. Install it with: pip install akshare"
            ) from exc
        return ak

    def _normalize_symbol(self, symbol: str) -> str:
        s = symbol.strip().lower()
        m = re.search(r"(\d{6})", s)
        if not m:
            raise NotImplementedError(
                f"cn_akshare only supports A-share 6-digit symbols, got: {symbol}"
            )
        return m.group(1)

    def _sina_symbol(self, symbol: str) -> str:
        code = self._normalize_symbol(symbol)
        if code.startswith(("5", "6", "9")):
            return f"sh{code}"
        return f"sz{code}"

    def _xq_symbol(self, symbol: str) -> str:
        code = self._normalize_symbol(symbol)
        if code.startswith(("5", "6", "9")):
            return f"SH{code}"
        return f"SZ{code}"

    def _is_likely_etf_symbol(self, symbol: str) -> bool:
        code = self._normalize_symbol(symbol)
        # 常见 A 股 ETF 代码段：5xxxxx(沪市) / 15xxxx,16xxxx,18xxxx(深市)
        return code.startswith(("5", "15", "16", "18"))

    def _normalize_hist_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        col_map = {
            "日期": "Date",
            "date": "Date",
            "Date": "Date",
            "开盘": "Open",
            "open": "Open",
            "Open": "Open",
            "最高": "High",
            "high": "High",
            "High": "High",
            "最低": "Low",
            "low": "Low",
            "Low": "Low",
            "收盘": "Close",
            "close": "Close",
            "Close": "Close",
            "成交量": "Volume",
            "volume": "Volume",
            "Volume": "Volume",
            "成交额": "Amount",
            "amount": "Amount",
            "Amount": "Amount",
        }
        df = raw_df.rename(columns=col_map).copy()
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"hist dataframe missing columns: {missing}")

        out = df[required].copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"])
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        out = dedupe_daily_bars(
            out, "Date", ["Open", "High", "Low", "Close", "Volume"]
        )
        out["Volume"] = out["Volume"].astype(float)

        return out

    def _format_ak_hist(self, df: pd.DataFrame, symbol: str, start: str, end: str) -> str:
        if df is None or df.empty:
            return f"No data found for symbol '{symbol}' between {start} and {end}"
        out = self._normalize_hist_df(df)
        return format_hist_csv(out, symbol, start, end)

    @staticmethod
    def _slice_hist_df(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        return slice_hist_df(df, start_date, end_date)

    def _drop_incomplete_today_bar(
        self, hist_df: pd.DataFrame, end_date: str
    ) -> pd.DataFrame:
        """Keep incomplete intraday prices out of the completed daily series."""
        return drop_incomplete_today_bar(hist_df, "Date", end_date)

    @staticmethod
    def _shrink_table(
        df: pd.DataFrame,
        max_rows: int = 8,
        max_cols: int = 14,
        *,
        table_kind: str | None = "generic",
        require_core_fields: bool = False,
        max_prompt_chars: int | None = None,
    ) -> str:
        """Clean and render a vendor table for LLM injection.

        ``max_cols`` is retained for call-site compatibility but ignored:
        column selection is name-based only (no positional iloc slice).
        """
        kwargs = {
            "max_rows": max_rows,
            "table_kind": table_kind,
            "require_core_fields": require_core_fields,
        }
        if max_prompt_chars is not None:
            kwargs["max_prompt_chars"] = max_prompt_chars
        # max_cols intentionally unused — positional column cuts are forbidden.
        _ = max_cols
        return shrink_table(df, **kwargs)

    def _fetch_hist_df(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        with AKSHARE_CALL_LOCK:
            ak = self._ak()
            code = self._normalize_symbol(symbol)
            symbol_with_market = self._sina_symbol(symbol)
            start_yyyymmdd = start_date.replace("-", "")
            end_yyyymmdd = end_date.replace("-", "")

            # ETF 优先：Sina 历史接口稳定且不依赖东财
            if self._is_likely_etf_symbol(symbol):
                etf_errors = []
                try:
                    df = ak.fund_etf_hist_sina(symbol=symbol_with_market)
                    out = self._normalize_hist_df(df)
                    out = self._slice_hist_df(out, start_date, end_date)
                    if not out.empty:
                        return self._drop_incomplete_today_bar(out, end_date)
                    etf_errors.append("fund_etf_hist_sina: empty after date filter")
                except DuplicateBarConflictError:
                    # Data-integrity refusal: do not silently switch to another
                    # source whose row order is equally arbitrary.
                    raise
                except Exception as exc:
                    etf_errors.append(f"fund_etf_hist_sina: {type(exc).__name__}")

                try:
                    df = ak.fund_etf_hist_em(
                        symbol=code,
                        period="daily",
                        start_date=start_yyyymmdd,
                        end_date=end_yyyymmdd,
                        adjust="qfq",
                    )
                    out = self._normalize_hist_df(df)
                    if not out.empty:
                        return self._drop_incomplete_today_bar(out, end_date)
                    etf_errors.append("fund_etf_hist_em: empty dataframe")
                except DuplicateBarConflictError:
                    raise
                except Exception as exc:
                    etf_errors.append(f"fund_etf_hist_em: {type(exc).__name__}")

            # Source 1: Eastmoney (default)
            em_last_exc = None
            for i in range(2):
                try:
                    df = ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=start_yyyymmdd,
                        end_date=end_yyyymmdd,
                        adjust="qfq",
                    )
                    out = self._normalize_hist_df(df)
                    out = self._slice_hist_df(out, start_date, end_date)
                    return self._drop_incomplete_today_bar(out, end_date)
                except DuplicateBarConflictError:
                    raise
                except Exception as exc:
                    em_last_exc = exc
                    if i < 1:
                        time.sleep(0.6 * (i + 1))

            # Source 2: Sina
            try:
                df = ak.stock_zh_a_daily(
                    symbol=symbol_with_market,
                    start_date=start_yyyymmdd,
                    end_date=end_yyyymmdd,
                    adjust="qfq",
                )
                out = self._normalize_hist_df(df)
                out = self._slice_hist_df(out, start_date, end_date)
                return self._drop_incomplete_today_bar(out, end_date)
            except DuplicateBarConflictError:
                raise
            except Exception:
                pass

            # Source 3: Tencent
            try:
                df = ak.stock_zh_a_hist_tx(
                    symbol=symbol_with_market,
                    start_date=start_yyyymmdd,
                    end_date=end_yyyymmdd,
                    adjust="qfq",
                )
                out = self._normalize_hist_df(df)
                out = self._slice_hist_df(out, start_date, end_date)
                return self._drop_incomplete_today_bar(out, end_date)
            except DuplicateBarConflictError:
                raise
            except Exception:
                pass

            raise NotImplementedError(
                f"cn_akshare is temporarily unavailable for price history (eastmoney/sina/tencent all failed): {em_last_exc}"
            ) from em_last_exc

    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> str:
        df = self._fetch_hist_df(symbol, start_date, end_date)
        return self._format_ak_hist(df, symbol, start_date, end_date)

    def get_indicators(
        self, symbol: str, indicator: str, curr_date: str, look_back_days: int
    ) -> str:
        if indicator not in self.INDICATOR_DESCRIPTIONS:
            raise ValueError(
                f"Indicator {indicator} is not supported. "
                f"Please choose from: {list(self.INDICATOR_DESCRIPTIONS.keys())}"
            )

        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = curr_dt - timedelta(days=max(look_back_days, 260))
        df = self._fetch_hist_df(symbol, start_dt.strftime("%Y-%m-%d"), curr_date)
        if df is None or df.empty:
            return f"No data found for {symbol} for indicator {indicator}"

        ind_df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )[["date", "open", "high", "low", "close", "volume"]].copy()
        ind_df["date"] = pd.to_datetime(ind_df["date"], errors="coerce")
        ind_df = ind_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        ss = wrap(ind_df)
        indicator_series = ss[indicator]

        values_by_date = {}
        for idx, dt_val in enumerate(ind_df["date"]):
            date_str = pd.to_datetime(dt_val).strftime("%Y-%m-%d")
            val = indicator_series.iloc[idx]
            values_by_date[date_str] = "N/A" if pd.isna(val) else str(val)

        begin = curr_dt - timedelta(days=look_back_days)
        lines = []
        d = curr_dt
        while d >= begin:
            key = d.strftime("%Y-%m-%d")
            if key in values_by_date:
                value = values_by_date[key]
                if value == "N/A":
                    value = cn_no_data_reason(key)
            else:
                value = cn_no_data_reason(key)
            lines.append(f"{key}: {value}")
            d -= timedelta(days=1)

        result = (
            f"## {indicator} 指标值（{begin.strftime('%Y-%m-%d')} 至 {curr_date}）：\n\n"
            + "\n".join(lines)
            + "\n\n"
            + self.INDICATOR_DESCRIPTIONS[indicator]
        )
        return result

    def _fetch_company_info_em_fallback(self, code: str) -> pd.DataFrame:
        try:
            secid = f"1.{code}" if code.startswith(("5", "6", "9")) else f"0.{code}"
            url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f57,f58,f84,f85,f116,f117,f127"
            import requests
            res = requests.get(url, timeout=3).json()
            data = res.get("data") or {}
            if data:
                info_list = [
                    {"item": "股票代码", "value": str(data.get("f57") or code)},
                    {"item": "股票简称", "value": str(data.get("f58") or "未知")},
                    {"item": "行业", "value": str(data.get("f127") or "半导体/科技")},
                    {"item": "总股本", "value": str(data.get("f84") or "")},
                    {"item": "流通股", "value": str(data.get("f85") or "")},
                    {"item": "总市值", "value": str(data.get("f116") or "")},
                    {"item": "流通市值", "value": str(data.get("f117") or "")},
                ]
                return pd.DataFrame(info_list)
        except Exception:
            pass
        return pd.DataFrame()

    def get_fundamentals(self, ticker: str, curr_date: str = None) -> str:
        """Company profile (snapshot) + financial abstract (period-mapped cutoff).

        Company Profile is a live market snapshot: on historical analysis dates
        it is refused. Financial Abstract remains available with A4 period cutoff.
        """
        with AKSHARE_CALL_LOCK:
            ak = self._ak()
            code = self._normalize_symbol(ticker)
            errors = []

            info_df = None
            try:
                info_df = ak.stock_individual_info_em(symbol=code)
            except Exception as exc:
                errors.append(f"stock_individual_info_em: {type(exc).__name__}")

            if info_df is None or info_df.empty:
                try:
                    info_df = ak.stock_individual_basic_info_xq(symbol=self._xq_symbol(ticker))
                    if not info_df.empty and set(info_df.columns) >= {"item", "value"}:
                        info_df = info_df.rename(columns={"item": "item", "value": "value"})
                except Exception as exc:
                    errors.append(f"stock_individual_basic_info_xq: {type(exc).__name__}")

            if info_df is None or info_df.empty:
                try:
                    info_df = self._fetch_company_info_em_fallback(code)
                except Exception as exc:
                    errors.append(f"_fetch_company_info_em_fallback: {type(exc).__name__}")

            stock_name = ""
            if info_df is not None and not info_df.empty and "item" in info_df.columns and "value" in info_df.columns:
                name_row = info_df[info_df["item"].astype(str).str.contains("简称|名称")]
                if not name_row.empty:
                    stock_name = str(name_row.iloc[0]["value"])

            abstract_df = None
            try:
                abstract_df = ak.stock_financial_abstract(symbol=code)
            except Exception as exc:
                errors.append(f"stock_financial_abstract: {type(exc).__name__}")

            parts = [f"## Fundamentals for {ticker} ({stock_name})"] if stock_name else [f"## Fundamentals for {ticker}"]

            # Company Profile is a live snapshot — refuse on historical dates.
            if is_historical_analysis_date(curr_date):
                parts.append("### Company Profile")
                parts.append(
                    snapshot_historical_refusal(
                        curr_date, source_label="Company Profile（总市值/PE/个股信息）"
                    )
                )
            elif info_df is not None and not info_df.empty:
                for c in info_df.columns:
                    info_df[c] = info_df[c].astype(str).str.slice(0, 220)
                parts.append("### Company Profile")
                parts.append(info_df.head(40).to_markdown(index=False))

            if abstract_df is not None and not abstract_df.empty:
                if curr_date:
                    try:
                        eff_map = self._sina_effective_announce_map(ticker, assume_locked=True)
                        filtered, latest = filter_abstract_period_columns(
                            abstract_df, eff_map, curr_date
                        )
                        parts.append("### Financial Abstract")
                        yoy_note = False
                        if filtered is not None and not filtered.empty:
                            period_cols = [
                                c for c in filtered.columns if c not in ("选项", "指标")
                            ]
                            yoy_note = periods_used_dropped_yoy(eff_map, period_cols)
                        parts.append(
                            financial_cutoff_header(
                                latest, curr_date, yoy_disclaimer=yoy_note
                            )
                        )
                        if latest is None or filtered is None or filtered.empty:
                            parts.append(
                                f"【数据获取失败】财务摘要在 {curr_date} 及之前无已公开报告期列。"
                            )
                        else:
                            metric_cols = [c for c in filtered.columns if c not in ("选项", "指标")]
                            # Prefer newest periods for display (column order often newest-first).
                            top_cols = metric_cols[:8]
                            cols = [c for c in ("选项", "指标") if c in filtered.columns] + top_cols
                            parts.append(
                                self._shrink_table(
                                    filtered[cols],
                                    max_rows=20,
                                    max_cols=10,
                                    table_kind="abstract",
                                )
                            )
                    except Exception as exc:
                        _provider_logger.warning(
                            "financial abstract cutoff failed for %s: %s", ticker, exc
                        )
                        parts.append("### Financial Abstract")
                        parts.append(
                            "【数据获取失败】财务摘要无法按公告生效日截断"
                            f"（{type(exc).__name__}: {exc}），本项不可用。"
                        )
                else:
                    # No analysis date → cannot prove periods are public; refuse abstract.
                    parts.append("### Financial Abstract")
                    parts.append(
                        "【数据获取失败】财务摘要缺少 curr_date，无法做公告日截断，本项不可用。"
                    )

            if len(parts) > 1:
                return "\n\n".join(parts)

            raise NotImplementedError(
                "cn_akshare is temporarily unavailable for fundamentals: "
                + "; ".join(errors)
            )

    def _load_sina_financial_tables(self, ticker: str, assume_locked: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch raw sina balance/income/cashflow frames (no truncation).

        Short-lived per-ticker cache of **raw uncut DataFrames** only.
        Key is ticker code alone (NOT curr_date). Truncation by effective
        announce date always happens after this cache returns, so two
        analyses with different curr_date in the same 120s window cannot
        share a post-cutoff result or leak future periods.
        """
        code = self._normalize_symbol(ticker)
        cache = getattr(self, "_sina_fin_tables_cache", None)
        if cache is None:
            self._sina_fin_tables_cache = {}
            cache = self._sina_fin_tables_cache
        hit = cache.get(code)
        if hit is not None:
            loaded_at, tables = hit
            # Raw tables only; safe to reuse across curr_date values.
            if time.monotonic() - loaded_at < 120 and tables:
                return tables

        ak = self._ak()
        symbol = self._sina_symbol(ticker)
        names = ("资产负债表", "利润表", "现金流量表")
        out: dict[str, pd.DataFrame] = {}

        def _one(report_name: str) -> pd.DataFrame:
            df = ak.stock_financial_report_sina(stock=symbol, symbol=report_name)
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            return df

        def _fill() -> None:
            for name in names:
                try:
                    out[name] = _one(name)
                except Exception as exc:
                    _provider_logger.warning(
                        "sina financial table %s failed for %s: %s", name, ticker, exc
                    )

        if assume_locked:
            _fill()
        else:
            with AKSHARE_CALL_LOCK:
                _fill()

        cache[code] = (time.monotonic(), out)
        return out

    def _sina_effective_announce_map(self, ticker: str, assume_locked: bool = False):
        tables = self._load_sina_financial_tables(ticker, assume_locked=assume_locked)
        return build_effective_announce_map(tables)

    def _financial_report_sina(
        self, ticker: str, report_name: str, curr_date: str = None
    ) -> str:
        """Return one sina financial statement markdown, truncated by A4 effective announce date.

        Historical-date analysis refuses the THS abstract fallback because it has
        no announcement-date field and cannot be proven public-by-curr_date.
        """
        if not curr_date:
            return (
                "【数据获取失败】财务报表缺少 curr_date，无法按公告生效日截断，"
                f"{report_name} 本项不可用。"
            )
        with AKSHARE_CALL_LOCK:
            ak = self._ak()
            symbol = self._sina_symbol(ticker)
            errors: list[str] = []
            today = cn_today_str()
            is_historical = (
                parse_yyyymmdd(curr_date) is not None
                and parse_yyyymmdd(curr_date) < parse_yyyymmdd(today)
            )

            tables = self._load_sina_financial_tables(ticker, assume_locked=True)
            sina_df = tables.get(report_name)
            if sina_df is None:
                errors.append(f"stock_financial_report_sina: missing {report_name}")

            if sina_df is not None:
                if "报告日" not in sina_df.columns or "公告日期" not in sina_df.columns:
                    return (
                        f"【数据获取失败】{report_name} 缺少 报告日/公告日期 列，"
                        "无法做历史截断，本项不可用。"
                    )
                try:
                    # Map uses all three tables so IS/CF YoY refresh is capped by
                    # statutory deadline and cross-checked with BS announce dates.
                    eff_map = build_effective_announce_map(tables)
                    filtered, latest = filter_financial_df_by_effective_announce(
                        sina_df, eff_map, curr_date
                    )
                except Exception as exc:
                    return (
                        f"【数据获取失败】{report_name} 公告生效日截断失败"
                        f"（{type(exc).__name__}: {exc}），本项不可用。"
                    )
                if filtered is None or filtered.empty or latest is None:
                    header = financial_cutoff_header(latest, curr_date)
                    return (
                        f"{header}\n"
                        f"【数据获取失败】{report_name} 在 {curr_date} 及之前无已公开报告期。"
                    )
                # Newest first for LLM: sort by report period descending.
                work = filtered.copy()
                work["__period"] = work["报告日"].map(lambda x: parse_yyyymmdd(x))
                work = work.dropna(subset=["__period"]).sort_values("__period", ascending=False)
                work = work.drop(columns=["__period"])
                yoy_note = periods_used_dropped_yoy(eff_map, work["报告日"])
                header = financial_cutoff_header(
                    latest, curr_date, yoy_disclaimer=yoy_note
                )
                kind_map = {
                    "资产负债表": "balance",
                    "利润表": "income",
                    "现金流量表": "cashflow",
                }
                table = self._shrink_table(
                    work,
                    max_rows=12,
                    max_cols=18,
                    table_kind=kind_map.get(report_name, "generic"),
                    # Core-four gate only applies to balance sheet: income/cashflow
                    # never contain 总资产/总负债/净资产 together.
                    require_core_fields=(report_name == "资产负债表"),
                )
                if table.startswith("【数据获取失败】"):
                    return f"{header}\n\n{table}"
                return f"{header}\n\n{table}"

            # Sina failed → THS fallback only allowed for same-day (non-historical) analysis.
            if is_historical:
                return (
                    f"【数据获取失败】主数据源新浪财报不可用，备用源同花顺摘要无公告日字段，"
                    f"历史日期分析（{curr_date}）下 {report_name} 不可用。"
                    + (f" 原因：{'; '.join(errors)}" if errors else "")
                )

            code = self._normalize_symbol(ticker)
            indicator = "按报告期"
            try:
                df = ak.stock_financial_abstract_new_ths(symbol=code, indicator=indicator)
                if df is None or df.empty:
                    raise ValueError("empty dataframe")
                table = self._shrink_table(
                    df,
                    max_rows=12,
                    max_cols=18,
                    table_kind="abstract",
                    require_core_fields=False,
                )
                return (
                    f"【备用数据源】同花顺财务摘要（无公告日字段，仅当日分析可用）\n\n{table}"
                )
            except Exception as exc:
                errors.append(f"stock_financial_abstract_new_ths: {type(exc).__name__}")

            raise NotImplementedError(
                f"cn_akshare is temporarily unavailable for {report_name}: {'; '.join(errors)}"
            )

    def get_balance_sheet(
        self, ticker: str, freq: str = "quarterly", curr_date: str = None
    ) -> str:
        table = self._financial_report_sina(ticker, "资产负债表", curr_date=curr_date)
        return f"## Balance Sheet ({ticker})\n\n{table}"

    def get_cashflow(
        self, ticker: str, freq: str = "quarterly", curr_date: str = None
    ) -> str:
        table = self._financial_report_sina(ticker, "现金流量表", curr_date=curr_date)
        return f"## Cashflow ({ticker})\n\n{table}"

    def get_income_statement(
        self, ticker: str, freq: str = "quarterly", curr_date: str = None
    ) -> str:
        table = self._financial_report_sina(ticker, "利润表", curr_date=curr_date)
        return f"## Income Statement ({ticker})\n\n{table}"

    def get_news(self, ticker: str, start_date: str, end_date: str) -> str:
        """Return only rows with parseable publication timestamps in the requested window."""
        with AKSHARE_CALL_LOCK:
            ak = self._ak()
            code = self._normalize_symbol(ticker)
            try:
                df = ak.stock_news_em(symbol=code)
                if df is None or df.empty:
                    return VendorEmpty(f"No news found for {ticker}")

                date_col = next(
                    (
                        name
                        for name in (
                            "发布时间",
                            "published_at",
                            "publishedAt",
                            "发布时间",
                            "date",
                            "新闻时间",
                        )
                        if name in df.columns
                    ),
                    None,
                )
                if date_col is None:
                    return VendorFail(
                        f"{ticker} 新闻结果缺少可验证发布时间字段，历史日期不可用"
                    )

                parsed_dates = pd.to_datetime(
                    df[date_col], errors="coerce", format="mixed"
                )
                if parsed_dates.isna().any():
                    return VendorFail(
                        f"{ticker} 新闻结果包含缺失或无法解析的发布时间，历史日期不可验证"
                    )
                df = df.copy()
                df[date_col] = parsed_dates

                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
                df = df[(df[date_col] >= start_dt) & (df[date_col] < end_dt)]
                if df.empty:
                    return VendorEmpty(
                        f"No news found for {ticker} between {start_date} and {end_date}"
                    )

                df = chronological(take_latest(df, date_col, 20), date_col)
                latest_dt = pd.to_datetime(df[date_col], errors="coerce").max()
                latest_label = (
                    latest_dt.strftime("%Y-%m-%d %H:%M:%S")
                    if pd.notna(latest_dt)
                    else end_date
                )

                rows = []
                for _, row in df.iterrows():
                    published_at = pd.to_datetime(row[date_col]).strftime("%Y-%m-%d %H:%M:%S")
                    title = str(row.get("新闻标题", row.get("标题", "No title")))
                    src = str(row.get("文章来源", row.get("来源", "Unknown")))
                    summary = str(row.get("新闻内容", row.get("内容", "")))
                    link = str(row.get("新闻链接", row.get("链接", "")))
                    rows.append(f"### {title} [发布时间：{published_at}] (source: {src})")
                    if summary and summary != "nan":
                        rows.append(summary[:400])
                    if link and link != "nan":
                        rows.append(f"Link: {link}")
                    rows.append("")

                return (
                    f"## {ticker} 新闻（{start_date} 至 {end_date}；"
                    f"最新发布时间：{latest_label}）：\n\n"
                    + "\n".join(rows)
                )
            except Exception as exc:
                raise NotImplementedError(
                    f"cn_akshare is temporarily unavailable for news: {exc}"
                ) from exc

    def get_global_news(
        self, curr_date: str, look_back_days: int = 7, limit: int = 50
    ) -> str:
        result = self.get_sina_global_news(page="1", page_size="100", tag_id="1,4,7")
        # get_sina_global_news 异常时返回 "新浪财经快讯获取失败：..." 字符串（truthy），需显式检查
        if result and result.startswith("## "):
            return result
        if result and result.startswith("新浪财经快讯获取失败"):
            # 源失败（网络/接口异常）：这是 VendorFail，链路应换到下一个 vendor
            # （如 yfinance），而不是当作“确认无新闻”停止。
            return VendorFail(result)
        # 新浪接口成功返回但无快讯条目：确认空，停止链路并如实上报。
        return VendorEmpty(f"{curr_date} 未获取到全球市场新闻")

    def get_insider_transactions(self, symbol: str, curr_date: str = None) -> str:
        """股东持股/内部人相关（主路径为当前截面，非历史增减持序列）。

        curr_date 必填：内部层不得默认今天。历史日期拒绝主路径快照；
        新闻降级窗口也必须以分析日为终点。
        """
        if not curr_date:
            return (
                "【数据获取失败】股东持股结构缺少 curr_date，"
                "内部层不得默认今天，本项不可用。"
            )
        refusal = snapshot_historical_refusal(
            curr_date, source_label="股东持股结构（当前快照）"
        )
        if refusal:
            return refusal
        ak = self._ak()
        code = self._normalize_symbol(symbol)
        errors = []
        try:
            # stock_ggcg_em 不支持按个股代码查询，默认全市场数据量较大
            with AKSHARE_CALL_LOCK:
                df = ak.stock_main_stock_holder(stock=code)
            if df is not None and not df.empty:
                return (
                    f"## Insider Transactions for {symbol}\n\n"
                    f"{df.head(20).to_markdown(index=False)}"
                )
            errors.append("stock_main_stock_holder: empty dataframe")
        except Exception as exc:
            errors.append(f"stock_main_stock_holder: {type(exc).__name__}")

        try:
            # 退化为分析日近两周相关新闻（不得用 wall-clock now）
            end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
            end_date = end_dt.strftime("%Y-%m-%d")
            start_date = (end_dt - timedelta(days=14)).strftime("%Y-%m-%d")
            news = result_to_prompt(self.get_news(symbol, start_date, end_date))
            return (
                f"## Insider Transactions for {symbol}\n\n"
                f"未获取到股东交易明细，降级返回近两周公司相关新闻：\n\n{news}"
            )
        except Exception as exc:
            errors.append(f"news_fallback: {type(exc).__name__}")

        raise NotImplementedError(
            f"cn_akshare is temporarily unavailable for insider transactions: {'; '.join(errors)}"
        )

    def get_sina_global_news(
        self, page: str = "1", page_size: str = "20", zhibo_id: str = "152", tag_id: str = "0"
    ) -> str:
        """获取新浪财经全球快讯（支持参数）

        Args:
            page: 页码，默认 "1"
            page_size: 每页数量，默认 "20"
            zhibo_id: 直播ID，默认 "152"（财经）
            tag_id: 标签ID，默认 "0"（全部）

        Returns:
            格式化的新闻文本
        """
        with AKSHARE_CALL_LOCK:
            try:
                import requests
                import re as _re

                url = "https://zhibo.sina.com.cn/api/zhibo/feed"
                params = {
                    "page": page,
                    "page_size": page_size,
                    "zhibo_id": zhibo_id,
                    "tag_id": tag_id,
                    "dire": "f",
                    "dpc": "1",
                    "pagesize": page_size,
                    "type": "1",
                }

                r = requests.get(url, params=params, timeout=10)
                data_json = r.json()

                time_list = [
                    item["create_time"] for item in data_json["result"]["data"]["feed"]["list"]
                ]
                text_list = [
                    item["rich_text"] for item in data_json["result"]["data"]["feed"]["list"]
                ]

                if not text_list:
                    return "未获取到新浪财经快讯"

                rows = []
                for time_str, content in zip(time_list, text_list):
                    if not content or content == "nan":
                        continue
                    m = _re.match(r"^【(.+?)】(.*)", content, _re.DOTALL)
                    if m:
                        title, body = m.group(1), m.group(2).strip()
                        rows.append(f"### [{time_str}] {title}")
                        if body:
                            rows.append(body[:300])
                        rows.append("")

                # 每条新闻占3行（标题、正文可选、空行），计算实际输出的新闻数
                actual_count = len([r for r in rows if r.startswith("###")])
                return f"## 新浪财经快讯（第{page}页，共{actual_count}条）：\n\n" + "\n".join(rows)

            except Exception as exc:
                return f"新浪财经快讯获取失败：{type(exc).__name__}: {exc}"

    # TTL cache for stock_zh_a_spot_em to avoid hammering Eastmoney under concurrent load
    _spot_cache: "pd.DataFrame | None" = None
    _spot_cache_ts: float = 0.0
    _SPOT_CACHE_TTL: float = 8.0  # seconds

    def get_realtime_quotes(self, symbols: list[str], curr_date: str = None) -> str:
        """Fetch real-time A-share quotes. Snapshot-only: refuse historical analysis dates."""
        refusal = snapshot_historical_refusal(
            curr_date, source_label="实时行情"
        )
        if refusal:
            return refusal
        import json
        import time as _time
        import logging

        logger = logging.getLogger(__name__)

        # Build normalized code → original symbol map
        code_to_original: dict[str, str] = {}
        for s in symbols:
            if not s or not s.strip():
                continue
            try:
                code = self._normalize_symbol(s)
            except NotImplementedError:
                continue
            if code and code not in code_to_original:
                code_to_original[code] = s.strip().upper()

        if not code_to_original:
            return json.dumps({})

        last_error = None

        # Try Sina first (lightweight, rarely blocked)
        try:
            result = self._fetch_quotes_sina(code_to_original)
            if result and result != "{}":
                return result
        except Exception as exc:
            logger.debug("[realtime-quotes] Sina failed, falling back to Eastmoney: %s", exc)
            last_error = exc

        # Fallback: Eastmoney via akshare (cached)
        now = _time.time()
        df = None
        if (
            CnAkshareProvider._spot_cache is not None
            and (now - CnAkshareProvider._spot_cache_ts) < CnAkshareProvider._SPOT_CACHE_TTL
        ):
            df = CnAkshareProvider._spot_cache
        else:
            try:
                with AKSHARE_CALL_LOCK:
                    ak = self._ak()
                    df = ak.stock_zh_a_spot_em()
            except TimeoutError as exc:
                _lock_logger.warning("[realtime-quotes] Eastmoney slot timeout: %s", exc)
                last_error = exc
            except Exception as exc:
                _lock_logger.warning("[realtime-quotes] Eastmoney fetch failed: %s", exc)
                last_error = exc
            else:
                CnAkshareProvider._spot_cache = df
                CnAkshareProvider._spot_cache_ts = now

        if df is not None and not df.empty:
            result = self._build_quotes_from_em(df, code_to_original)
            if result != "{}":
                return result
            last_error = ValueError("Eastmoney returned no requested quotes")

        raise NotImplementedError(
            "cn_akshare realtime quote sources unavailable"
        ) from last_error

    def _build_quotes_from_em(self, df: "pd.DataFrame", code_to_original: dict[str, str]) -> str:
        import json
        normalized = list(code_to_original.keys())
        df = df[df["代码"].isin(normalized)]
        result: dict[str, dict] = {}
        for _, row in df.iterrows():
            code = str(row.get("代码", ""))
            original = code_to_original.get(code)
            if not original:
                continue
            price = self._safe_float(row.get("最新价"))
            prev_close = self._safe_float(row.get("昨收"))
            change = round(price - prev_close, 4) if price is not None and prev_close else None
            change_pct = round(change / prev_close * 100, 4) if change is not None and prev_close else None
            result[original] = {
                "price": price,
                "open": self._safe_float(row.get("今开")),
                "high": self._safe_float(row.get("最高")),
                "low": self._safe_float(row.get("最低")),
                "previous_close": prev_close,
                "change": change,
                "change_pct": change_pct,
                "volume": self._safe_float(row.get("成交量")),
                "amount": self._safe_float(row.get("成交额")),
                "source": "eastmoney",
            }
        return json.dumps(result, ensure_ascii=False)

    def _fetch_quotes_sina(self, code_to_original: dict[str, str]) -> str:
        """Fetch quotes from Sina Finance hq.sinajs.cn as fallback."""
        import json
        import requests as _requests

        sina_codes = []
        sina_to_original: dict[str, str] = {}
        for code, original in code_to_original.items():
            prefix = "sh" if code.startswith(("5", "6", "9")) else "bj" if code.startswith(("4", "8")) else "sz"
            sina_code = f"{prefix}{code}"
            sina_codes.append(sina_code)
            sina_to_original[sina_code] = original

        if not sina_codes:
            return json.dumps({})

        try:
            resp = _requests.get(
                "https://hq.sinajs.cn/list=" + ",".join(sina_codes),
                headers={"Referer": "https://finance.sina.com.cn/", "User-Agent": "Mozilla/5.0"},
                timeout=5,
            )
            resp.encoding = "gbk"
        except Exception:
            return json.dumps({})

        result: dict[str, dict] = {}
        for line in resp.text.splitlines():
            line = line.strip()
            if not line or '="' not in line:
                continue
            try:
                var_part, data_part = line.split('="', 1)
                sina_code = var_part.split("_")[-1]
                fields = data_part.rstrip('";').split(",")
                if len(fields) < 10:
                    continue
                original = sina_to_original.get(sina_code)
                if not original:
                    continue
                price = self._safe_float(fields[3])
                prev_close = self._safe_float(fields[2])
                change = round(price - prev_close, 4) if price is not None and prev_close else None
                change_pct = round(change / prev_close * 100, 4) if change is not None and prev_close else None
                # Sina fields[30]=date, fields[31]=time
                quote_time = None
                if len(fields) > 31 and fields[30] and fields[31]:
                    quote_time = f"{fields[30]} {fields[31]}"
                result[original] = {
                    "price": price,
                    "open": self._safe_float(fields[1]),
                    "high": self._safe_float(fields[4]),
                    "low": self._safe_float(fields[5]),
                    "previous_close": prev_close,
                    "change": change,
                    "change_pct": change_pct,
                    "volume": self._safe_float(fields[8]),
                    "amount": self._safe_float(fields[9]),
                    "quote_time": quote_time,
                    "source": "sina",
                }
            except (ValueError, IndexError):
                continue
        return json.dumps(result, ensure_ascii=False)

    @staticmethod
    def _safe_float(val) -> float | None:
        return safe_float(val)

    def get_board_fund_flow(self, curr_date: str = None) -> str:
        """获取行业板块资金流向排名（即时快照）。

        东财 ``stock_fund_flow_industry`` 对当前 IP 间歇不可达
        （RemoteDisconnected），失败时回退到同花顺
        ``stock_board_industry_summary_ths``（新浪无板块资金流接口）。
        历史日期分析直接拒绝：接口无历史截面。
        """
        refusal = snapshot_historical_refusal(
            curr_date, source_label="板块资金流向（即时）"
        )
        if refusal:
            return refusal

        errors: list[str] = []

        # Source 1: 东方财富（板块资金流）
        try:
            ak = self._ak()
            with AKSHARE_CALL_LOCK:
                df = ak.stock_fund_flow_industry(symbol="即时")
            if df is not None and not df.empty:
                return self._format_board_fund_flow(df)
            errors.append("stock_fund_flow_industry: empty dataframe")
        except Exception as exc:
            errors.append(f"stock_fund_flow_industry: {type(exc).__name__}")

        # Source 2: 同花顺（板块净流入快照）
        try:
            ak = self._ak()
            with AKSHARE_CALL_LOCK:
                df = ak.stock_board_industry_summary_ths()
            if df is not None and not df.empty:
                return self._format_board_fund_flow(
                    df, source_label="同花顺", net_col="净流入"
                )
            errors.append("stock_board_industry_summary_ths: empty dataframe")
        except Exception as exc:
            errors.append(f"stock_board_industry_summary_ths: {type(exc).__name__}")

        return (
            f"板块资金流向数据暂时不可用（东财/同花顺均失败："
            f"{'；'.join(errors)}）"
        )

    @staticmethod
    def _format_board_fund_flow(
        df: "pd.DataFrame",
        source_label: str = "东方财富",
        net_col: str | None = None,
    ) -> str:
        """Format an industry-board fund-flow frame, ranked by net inflow desc."""
        work = df.copy()
        if net_col is None:
            for cand in ("今日主力净流入-净额", "净额", "主力净流入-净额"):
                if cand in work.columns:
                    net_col = cand
                    break
        if net_col in work.columns:
            work = work.sort_values(net_col, ascending=False).reset_index(drop=True)
        else:
            work = work.reset_index(drop=True)
        work.insert(0, "排名", range(1, len(work) + 1))
        total = len(work)
        result = work.head(10).to_string(index=False)
        if source_label and source_label != "东方财富":
            return (
                f"【备用数据源：{source_label}】板块资金流向排名"
                f"（共{total}个板块，前10名）：\n{result}"
            )
        return f"板块资金流向排名（共{total}个板块，前10名）：\n{result}"

    def get_individual_fund_flow(self, symbol: str, curr_date: str = None) -> str:
        """获取个股近期主力资金净流向，并按 curr_date 截断。

        资金流路径按“AkShare EM → 东财公开直连 → 新浪历史 legacy Web →
        当前日同花顺总净额快照”的顺序尝试。东财直连只接入已核验的 f51/f52
        契约，并把 f52 保持为 ``r0_net``；f53-f56 仅作原始发现值保留，缺失的
        尾部字段不会伪装成语义值。每个成功或失败的 ``FundFlowText`` 都携带完整尝试链。
        """
        errors: list[str] = []
        attempted_sources: list[str] = []
        em_typed_gap = ""

        def _gap_reason(base_reason: str) -> str:
            return f"{base_reason}；{'；'.join(errors)}" if errors else base_reason

        def _attach_chain(value: str, final_source: str) -> FundFlowText:
            evidence = list(getattr(value, "fund_flow_evidence", []) or [])
            metadata = dict(getattr(value, "fund_flow_evidence_meta", {}) or {})
            if metadata.get("as_of") is None and evidence:
                observed_dates = sorted(
                    {
                        str(record.get("as_of") or record.get("date"))
                        for record in evidence
                        if record.get("as_of") or record.get("date")
                    }
                )
                if observed_dates:
                    metadata["as_of"] = observed_dates[-1]
            metadata.setdefault("actual_as_of", metadata.get("as_of"))
            if metadata.get("field") is None:
                fields = {
                    field
                    for record in evidence
                    for field in ("r0_net", "netamount")
                    if record.get(field) is not None
                }
                if "r0_net" in fields:
                    metadata["field"] = "r0_net"
                elif "netamount" in fields:
                    metadata["field"] = "netamount"
            consensus = metadata.get("consensus")
            if isinstance(consensus, dict):
                consensus_status = consensus.get("status")
                direction_allowed = (
                    bool(consensus.get("direction_allowed"))
                    and consensus_status == "consensus"
                    and metadata.get("algorithm_group") != "legacy_web_algorithm"
                )
                metadata["status"] = consensus_status or metadata.get("status")
                metadata["direction"] = (
                    consensus.get("direction") if direction_allowed else "blocked"
                )
                metadata["direction_allowed"] = direction_allowed
                metadata["hard_guard"] = {
                    "blocked": not direction_allowed,
                    "direction_allowed": direction_allowed,
                    "reason": consensus.get("reason")
                    or "consensus unavailable or direction is not permitted",
                }
            metadata.update(
                {
                    "attempted_sources": list(attempted_sources),
                    "fallback_errors": list(errors),
                    "failure_categories": sorted(
                        {
                            _fund_flow_failure_category(error)
                            for error in errors
                        }
                    ),
                    "em_typed_gap": em_typed_gap,
                    "final_source": final_source,
                    "last_attempted_source": attempted_sources[-1]
                    if attempted_sources
                    else None,
                }
            )
            return FundFlowText(
                str(value),
                evidence=evidence,
                evidence_meta=metadata,
            )

        if not curr_date:
            errors.append("fund_flow_individual: curr_date_missing")
            gap = build_provider_text(
                f"【数据获取失败】个股资金流向缺少 curr_date，无法做日期截断，"
                f"{symbol} 本项不可用。",
                symbol=symbol,
                requested_as_of=curr_date,
                source="fund_flow_individual",
                reason="curr_date_missing；不得在缺少分析日期时回退到 live 数据源",
                field="r0_net",
                raw_unit="元",
                failure_category="validation",
            )
            return _attach_chain(gap, "unavailable")

        cutoff = parse_yyyymmdd(curr_date)
        if cutoff is None:
            errors.append(f"fund_flow_individual: curr_date_invalid:{curr_date!r}")
            gap = build_provider_text(
                f"【数据获取失败】个股资金流向 curr_date 无法解析：{curr_date!r}",
                symbol=symbol,
                requested_as_of=curr_date,
                source="fund_flow_individual",
                reason=f"curr_date_invalid:{curr_date!r}",
                field="r0_net",
                raw_unit="元",
                failure_category="validation",
            )
            return _attach_chain(gap, "unavailable")

        code = self._normalize_symbol(symbol)
        is_historical = is_historical_analysis_date(curr_date)

        # Source 1: AkShare's Eastmoney wrapper (近 120 交易日逐日序列).
        attempted_sources.append("akshare.stock_individual_fund_flow")
        ak = None
        try:
            ak = self._ak()
        except Exception as exc:
            errors.append(f"akshare provider unavailable: {type(exc).__name__}")
        if ak is None:
            errors.append("stock_individual_fund_flow: akshare unavailable")
        else:
            try:
                # 沪市：以 5、6、9 开头；其余为深市
                market = "sh" if code[:1] in ("5", "6", "9") else "sz"
                with AKSHARE_CALL_LOCK:
                    df = ak.stock_individual_fund_flow(stock=code, market=market)
                if df is None or df.empty:
                    errors.append("stock_individual_fund_flow: empty dataframe")
                else:
                    # Invalid or out-of-range data must not terminate the chain.
                    em_text = self._format_individual_fund_flow_em(
                        df, symbol, curr_date, cutoff
                    )
                    em_evidence = getattr(em_text, "fund_flow_evidence", None)
                    if (
                        em_text is not None
                        and isinstance(em_evidence, list)
                        and em_evidence
                    ):
                        return _attach_chain(
                            em_text, "eastmoney_individual_fund_flow"
                        )
                    em_meta = getattr(em_text, "fund_flow_evidence_meta", None) or {}
                    if em_meta.get("reason"):
                        errors.append(
                            "stock_individual_fund_flow: formatter reason: "
                            f"{em_meta['reason']}"
                        )
                    if em_text:
                        errors.append(
                            "stock_individual_fund_flow: formatter failure: "
                            f"{em_text}"
                        )
                    errors.append(
                        "stock_individual_fund_flow: structured evidence unavailable"
                    )
                    errors.append(
                        "stock_individual_fund_flow: invalid or empty usable rows"
                    )
            except Exception as exc:
                errors.append(f"stock_individual_fund_flow: {type(exc).__name__}")

        em_failures = [
            error
            for error in errors
            if error.startswith("stock_individual_fund_flow:")
            or error.startswith("akshare provider unavailable:")
        ]
        em_typed_gap = "；".join(em_failures) or (
            "stock_individual_fund_flow: structured evidence unavailable"
        )

        # Source 2: direct Eastmoney endpoint.  This is a new-algorithm source,
        # but it must still fall through when the response is not auditable.
        attempted_sources.append("eastmoney_direct")
        try:
            direct_text, direct_error = self._fetch_eastmoney_direct_fund_flow(
                symbol,
                curr_date,
                cutoff,
                require_curr_date=True,
            )
            if direct_text is not None:
                return _attach_chain(direct_text, "eastmoney_direct")
            if direct_error:
                errors.append(direct_error)
        except Exception as exc:
            errors.append(f"eastmoney_direct: {type(exc).__name__}")

        # Source 2.5: Sina Web is legacy reference only. Keep the typed
        # response for auditability, but it never drives main-force direction.
        attempted_sources.append("sina_historical")
        try:
            hist_text = self._fetch_sina_historical_fund_flow(
                symbol,
                curr_date,
                cutoff,
                require_curr_date=not is_historical,
            )
            if hist_text is not None:
                metadata = dict(
                    getattr(hist_text, "fund_flow_evidence_meta", {}) or {}
                )
                metadata.update(
                    {
                        "legacy_web_algorithm": True,
                        "legacy_web_reference_only": True,
                        "direction_allowed": False,
                        "reason": "新浪旧 Web 参考值，不驱动主力方向",
                    }
                )
                hist_value = FundFlowText(
                    f"{hist_text}\n（新浪旧 Web 参考值：仅作降级参考，不驱动方向）",
                    evidence=getattr(hist_text, "fund_flow_evidence", []),
                    evidence_meta=metadata,
                )
                return _attach_chain(hist_value, "sina_historical")
            if is_historical:
                errors.append(
                    "sina historical fund flow: no rows on or before curr_date"
                )
            else:
                errors.append("sina historical fund flow: no current-day close row")
        except Exception as exc:
            errors.append(f"sina historical fund flow: {type(exc).__name__}")

        if is_historical:
            gap = build_provider_text(
                f"【数据获取失败】历史日期 {curr_date} 新算法与新浪历史/legacy Web 资金流均不可用，"
                f"{symbol} 本项不可用。（{'；'.join(errors)}）",
                symbol=symbol,
                requested_as_of=curr_date,
                source="fund_flow_individual",
                reason=_gap_reason(
                    "historical new-algorithm evidence unavailable; legacy Web reference unavailable"
                ),
                field="r0_net",
                raw_unit="元",
                failure_category="source_unavailable",
            )
            return _attach_chain(gap, "unavailable")

        # Source 3: 同花顺即时资金流净额快照（历史接口尚无当日收盘行时）。
        attempted_sources.append("ths_instant_snapshot")
        try:
            if ak is None:
                ak = self._ak()
            if ak is None:
                raise RuntimeError("akshare unavailable")
            with AKSHARE_CALL_LOCK:
                df = ak.stock_fund_flow_individual(symbol="即时")
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            stock_df = df[df["股票代码"].astype(str).str.zfill(6) == code.zfill(6)]
            if stock_df.empty:
                gap = build_provider_text(
                    f"【数据获取失败】{symbol} 同花顺即时资金流净额快照无记录"
                    f"（{'；'.join(errors)}）",
                    symbol=symbol,
                    requested_as_of=curr_date,
                    source="ths_instant_snapshot",
                    reason=_gap_reason(
                        "同花顺即时资金流净额快照无记录；该源不提供新浪 netamount/r0_net evidence"
                    ),
                    field="netamount",
                    raw_unit="亿元",
                    failure_category="validation_failure",
                )
                return _attach_chain(gap, "unavailable")
            row = stock_df.iloc[0]
            if "净额" not in stock_df.columns:
                gap = build_provider_text(
                    f"【数据获取失败】{symbol} 同花顺即时资金流净额快照缺少净额字段"
                    f"（{'；'.join(errors)}）",
                    symbol=symbol,
                    requested_as_of=curr_date,
                    source="ths_instant_snapshot",
                    reason=_gap_reason(
                        "同花顺即时资金流净额快照缺少净额字段；该源不提供新浪 netamount/r0_net evidence"
                    ),
                    field="netamount",
                    raw_unit="亿元",
                    failure_category="validation_failure",
                )
                return _attach_chain(gap, "unavailable")
            net_amount = _usable_fund_amount_text(row["净额"])
            if net_amount is None:
                gap = build_provider_text(
                    f"【数据获取失败】{symbol} 同花顺即时资金流净额快照净额缺失或不可解析"
                    f"（{'；'.join(errors)}）",
                    symbol=symbol,
                    requested_as_of=curr_date,
                    source="ths_instant_snapshot",
                    reason=_gap_reason(
                        "同花顺即时资金流净额快照净额缺失或不可解析；该源不提供新浪 netamount/r0_net evidence"
                    ),
                    field="netamount",
                    raw_unit="亿元",
                    failure_category="validation_failure",
                )
                return _attach_chain(gap, "unavailable")

            def _v(col: str) -> str:
                if col not in stock_df.columns:
                    return ""
                val = row[col]
                return "" if pd.isna(val) else str(val)

            row_payload = {
                "股票代码": code,
                "日期": curr_date,
                "净额": row.get("净额"),
                "单位": "亿元",
                "period_kind": "realtime_single_day",
                "window": "1d",
            }
            evidence = build_ths_evidence(
                [row_payload],
                symbol=symbol,
                requested_as_of=curr_date,
                retrieved_at=self._sina_retrieved_at(),
            )
            consensus = build_consensus_evidence(
                evidence,
                symbol=symbol,
                requested_as_of=curr_date,
                field="netamount",
            )
            snapshot = FundFlowText(
                (
                    f"【备用数据源：同花顺即时资金流净额快照】{symbol} 当日资金流净额快照"
                    f"（{curr_date}，最新价 {_v('最新价')}，涨跌幅 {_v('涨跌幅')}）：\n"
                    f"资金净额: {net_amount} | 流入资金: {_v('流入资金')} | "
                    f"流出资金: {_v('流出资金')} | 换手率: {_v('换手率')}\n"
                    "（该快照不是新浪历史 netamount/r0_net 同口径主力序列；"
                    "属于同花顺新算法组总净额，仍不得视为 r0_net 主力序列）"
                ),
                evidence=evidence,
                evidence_meta={
                    "symbol": symbol,
                    "requested_as_of": curr_date,
                    "retrieved_at": self._sina_retrieved_at(),
                    "source": "ths_instant_snapshot",
                    "source_family": "ths",
                    "algorithm_group": "new_algorithm_group",
                    "period_kind": "realtime_single_day",
                    "field": "netamount",
                    "raw_unit": "亿元",
                    "unit": "亿元",
                    "as_of": curr_date,
                    "actual_as_of": curr_date,
                    "status": "available",
                    "consensus": consensus,
                    "reason": "同花顺即时资金流净额是总净额，未将其等同于新浪历史 r0_net 主力序列",
                },
            )
            return _attach_chain(snapshot, "ths_instant_snapshot")
        except Exception as exc:
            errors.append(f"stock_fund_flow_individual: {type(exc).__name__}")

        gap_reason = "东财 AkShare/东财直连/新浪历史/同花顺即时资金流净额快照均失败"
        if errors:
            gap_reason = f"{gap_reason}；{'；'.join(errors)}"
        gap = build_provider_text(
            f"【数据获取失败】个股资金流向数据获取失败（东财 AkShare/东财直连/新浪历史/同花顺即时资金流净额快照均失败：{'；'.join(errors)}）",
            symbol=symbol,
            requested_as_of=curr_date,
            source="fund_flow_individual",
            reason=gap_reason,
            field="r0_net",
            raw_unit="元",
            failure_category="source_unavailable",
        )
        return _attach_chain(gap, "unavailable")

    def _fetch_eastmoney_direct_fund_flow(
        self,
        symbol: str,
        curr_date: str,
        cutoff,
        *,
        require_curr_date: bool = False,
    ) -> tuple[FundFlowText | None, str | None]:
        """Fetch verified daily fields from Eastmoney's public endpoint.

        The endpoint returns comma-separated ``f51`` onward values.  The
        provider contract requires only f51 (date) and f52 (finite main-force
        net amount).  f53-f56 remain raw/discovery-only values when present;
        unknown or missing trailing fields are ignored without fabricated
        semantics.  When the requested date is required, a prior close is not
        an acceptable as-of; the response must contain a valid row dated
        exactly ``curr_date``.
        """
        import json
        import requests as _requests

        code = self._normalize_symbol(symbol)
        secid = f"1.{code}" if code.startswith(("5", "6", "9")) else f"0.{code}"
        params = {
            "secid": secid,
            "lmt": str(_EASTMONEY_DIRECT_FUND_FLOW_FETCH),
            "klt": "101",
            "fields1": _EASTMONEY_DIRECT_FUND_FLOW_FIELDS1,
            "fields2": _EASTMONEY_DIRECT_FUND_FLOW_FIELDS2,
            "ut": "b2884a393a59ad64002292a3e90d46a5",
        }

        try:
            if not is_cn_trading_day(curr_date):
                return None, "eastmoney_direct: curr_date_not_cn_trading_day"
        except Exception as exc:
            return None, f"eastmoney_direct: trade_calendar: {type(exc).__name__}"

        try:
            response = _requests.get(
                _EASTMONEY_DIRECT_FUND_FLOW_URL,
                params=params,
                headers=_EASTMONEY_DIRECT_FUND_FLOW_HEADERS,
                timeout=_EASTMONEY_DIRECT_FUND_FLOW_TIMEOUT,
            )
        except _requests.Timeout as exc:
            return None, f"eastmoney_direct: timeout: {type(exc).__name__}"
        except _requests.RequestException as exc:
            return None, f"eastmoney_direct: request: {type(exc).__name__}"
        except Exception as exc:
            return None, f"eastmoney_direct: request: {type(exc).__name__}"

        status_code = getattr(response, "status_code", None)
        try:
            if status_code is not None and int(status_code) >= 400:
                return None, f"eastmoney_direct: http_status: {status_code}"
        except (TypeError, ValueError):
            pass
        try:
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
        except Exception as exc:
            return None, f"eastmoney_direct: http_status: {type(exc).__name__}"

        try:
            raw_text = getattr(response, "text", None)
            if raw_text is not None and str(raw_text).strip():
                payload = json.loads(raw_text)
            elif callable(getattr(response, "json", None)):
                payload = response.json()
            else:
                payload = json.loads(raw_text or "")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            return None, f"eastmoney_direct: json_decode: {type(exc).__name__}"
        except Exception as exc:
            return None, f"eastmoney_direct: json_decode: {type(exc).__name__}"

        if not isinstance(payload, dict):
            return None, "eastmoney_direct: json_shape: root is not an object"
        if "rc" not in payload:
            return None, "eastmoney_direct: rc_missing"
        if str(payload.get("rc")).strip() not in {"0", "0.0"}:
            return None, f"eastmoney_direct: rc={payload.get('rc')!r}"
        data = payload.get("data")
        if not isinstance(data, dict):
            return None, "eastmoney_direct: data_missing_or_invalid"
        klines = data.get("klines")
        if not isinstance(klines, (list, tuple)):
            return None, "eastmoney_direct: klines_missing_or_invalid"
        if not klines:
            return None, "eastmoney_direct: klines_empty"

        cutoff_date = cutoff
        parsed_rows: list[dict[str, str]] = []
        discovery_by_date: dict[str, dict[str, str]] = {}
        duplicate_dates: list[str] = []
        warnings: list[str] = []
        malformed_rows: list[str] = []
        for row_index, raw_row in enumerate(klines):
            if not isinstance(raw_row, str):
                warning = f"row {row_index}: kline is not text"
                warnings.append(warning)
                malformed_rows.append(warning)
                continue
            parts = [part.strip() for part in raw_row.split(",")]
            day_ts = (
                pd.to_datetime(parts[0], errors="coerce")
                if parts
                else pd.NaT
            )
            if pd.isna(day_ts):
                warning = f"row {row_index}: invalid_date"
                warnings.append(warning)
                malformed_rows.append(warning)
                continue
            day = day_ts.date()
            if day > cutoff_date:
                # Future rows are deliberately ignored, not rendered or used.
                continue
            if len(parts) < 2:
                warning = (
                    f"row {row_index}: field_count={len(parts)}, need_at_least=2"
                )
                warnings.append(warning)
                malformed_rows.append(warning)
                continue
            try:
                is_trade_day = is_cn_trading_day(day.isoformat())
            except Exception as exc:
                warning = (
                    f"row {row_index}: trade_calendar: {type(exc).__name__}"
                )
                warnings.append(warning)
                malformed_rows.append(warning)
                continue
            if not is_trade_day:
                warning = f"row {row_index}: non_trading_date={day.isoformat()}"
                warnings.append(warning)
                malformed_rows.append(warning)
                continue
            if _sina_decimal(parts[1]) is None:
                warning = f"row {row_index}: invalid_f52"
                warnings.append(warning)
                malformed_rows.append(warning)
                continue

            day_text = day.isoformat()
            if day_text in discovery_by_date:
                warning = f"row {row_index}: duplicate_date={day_text}"
                warnings.append(warning)
                duplicate_dates.append(day_text)
                continue
            discovery_fields: dict[str, str] = {}
            for field in _EASTMONEY_DIRECT_DISCOVERY_FIELDS:
                field_index = int(field[1:]) - 51
                if field_index < len(parts):
                    discovery_fields[field] = parts[field_index]
            discovery_by_date[day_text] = discovery_fields
            parsed_rows.append(
                {
                    "日期": day_text,
                    "主力净流入-净额": parts[1],
                    **{
                        f"{field}_raw": raw_value
                        for field, raw_value in discovery_fields.items()
                    },
                }
            )

        if duplicate_dates:
            detail = "; ".join(sorted(set(duplicate_dates))[:5])
            return None, f"eastmoney_direct: duplicate_date: {detail}"
        if require_curr_date and parsed_rows:
            requested_day = cutoff_date.isoformat()
            if not any(row.get("日期") == requested_day for row in parsed_rows):
                available = ", ".join(
                    sorted({str(row.get("日期")) for row in parsed_rows})[-5:]
                )
                reason = (
                    "no_requested_date_row"
                    if is_historical_analysis_date(curr_date)
                    else "no_current_day_row"
                )
                return None, (
                    f"eastmoney_direct: {reason} "
                    f"(requested={requested_day}; available={available})"
                )
        if malformed_rows:
            detail = "; ".join(malformed_rows[:5])
            if not parsed_rows:
                return None, (
                    "eastmoney_direct: no_usable_rows_on_or_before_curr_date "
                    f"(malformed_kline_rows: {detail})"
                )
            return None, (
                "eastmoney_direct: malformed_kline_rows_on_or_before_curr_date "
                f"({detail})"
            )
        if not parsed_rows:
            detail = f" ({'; '.join(warnings[:5])})" if warnings else ""
            return None, f"eastmoney_direct: no_usable_rows_on_or_before_curr_date{detail}"

        frame = pd.DataFrame(parsed_rows)
        formatted = self._format_individual_fund_flow_em(
            frame,
            symbol,
            curr_date,
            cutoff,
            source="eastmoney_direct",
        )
        evidence = getattr(formatted, "fund_flow_evidence", None)
        if formatted is None or not isinstance(evidence, list) or not evidence:
            return None, "eastmoney_direct: structured_evidence_unavailable"

        evidence = [dict(record) for record in evidence]
        for record in evidence:
            raw_fields = dict(discovery_by_date.get(record.get("date"), {}))
            record["vendor_raw_fields"] = raw_fields
            record["vendor_raw_field_status"] = "discovery_only"
            record["vendor_raw_field_units"] = {
                field: None for field in raw_fields
            }

        metadata = dict(getattr(formatted, "fund_flow_evidence_meta", {}) or {})
        metadata.update(
            {
                "endpoint": _EASTMONEY_DIRECT_FUND_FLOW_URL,
                "field_mapping": dict(_EASTMONEY_DIRECT_FIELD_MAPPING),
                "field_semantics_verified": {
                    "f51": "measurement_date",
                    "f52": "r0_net",
                },
                "discovery_only_fields": list(
                    _EASTMONEY_DIRECT_DISCOVERY_FIELDS
                ),
                "discovery_field_unit_policy": "raw preserved; no normalization",
                "status": "available",
            }
        )
        if warnings:
            metadata["parse_warnings"] = warnings[:20]
        return (
            FundFlowText(
                str(formatted),
                evidence=evidence,
                evidence_meta=metadata,
            ),
            None,
        )

    def _fetch_sina_historical_fund_flow(
        self,
        symbol: str,
        curr_date: str,
        cutoff,
        *,
        require_curr_date: bool = False,
    ) -> str | None:
        """Source 2.5: fetch and render the Sina historical per-day money flow.

        Direct requests call (akshare has no wrapper for this endpoint) with the
        required Referer/User-Agent and a 10s timeout. Rows are filtered to
        ``opendate <= curr_date`` (anti-lookahead unchanged) and the latest N
        days are rendered EM-style. Rows without at least one finite
        ``netamount`` or ``r0_net`` are discarded before date selection; numeric
        zero is valid, while non-empty invalid values and infinities are not.
        When ``require_curr_date`` is true, a prior close is not enough: the
        caller must fall back to the current snapshot path until the historical
        endpoint exposes the requested day's close.
        Returns ``None`` when nothing usable remains on/before ``curr_date`` (or
        when the required current-day row is absent); raises on network/HTTP/parse
        failure so the caller records an explicit error.
        """
        import json
        import requests as _requests

        url = _SINA_HIST_FUND_FLOW_URL.format(
            num=_SINA_HIST_FUND_FLOW_FETCH,
            daima=self._sina_symbol(symbol),
        )
        resp = _requests.get(
            url,
            headers=_SINA_HIST_FUND_FLOW_HEADERS,
            timeout=_SINA_HIST_FUND_FLOW_TIMEOUT,
        )
        resp.raise_for_status()
        payload = json.loads(resp.text or "[]")
        if not isinstance(payload, list):
            return None
        kept: list[dict] = []
        cutoff_ts = pd.Timestamp(cutoff).normalize()
        has_curr_date = False
        for row in payload:
            if not isinstance(row, dict):
                continue
            day = str(row.get("opendate", "")).strip()
            if not day:
                continue
            try:
                day_ts = pd.Timestamp(day)
            except Exception:
                continue
            core_amounts: list[float] = []
            invalid_core_amount = False
            for field in _SINA_HIST_CORE_AMOUNT_FIELDS:
                raw_value = row.get(field)
                if raw_value is None or (
                    isinstance(raw_value, str) and not raw_value.strip()
                ):
                    continue
                value = safe_float(raw_value)
                if value is None or not math.isfinite(value):
                    invalid_core_amount = True
                    break
                core_amounts.append(value)
            if invalid_core_amount or not core_amounts:
                continue
            if pd.notna(day_ts) and day_ts.normalize() <= cutoff_ts:
                kept.append(row)
                has_curr_date = has_curr_date or day_ts.normalize() == cutoff_ts
        if not kept or (require_curr_date and not has_curr_date):
            return None
        kept.sort(key=lambda r: str(r.get("opendate", "")))
        return self._format_sina_historical_fund_flow(kept, symbol, curr_date)

    def _sina_retrieved_at(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _format_sina_historical_fund_flow(
        self, rows: list[dict], symbol: str, curr_date: str
    ) -> str | None:
        """Render Sina historical rows as an Eastmoney-aligned per-day table.

        Maps netamount/r0_net/ratioamount (plus r1_net..r4_net when present) to
        the labels the Eastmoney table uses; amounts are shown in 亿元. When the
        interface lacks the sub-order breakdown, that is stated explicitly.
        """
        records: list[dict] = []
        has_sub_orders = False
        retrieved_at = self._sina_retrieved_at()
        evidence = build_sina_evidence(
            rows,
            symbol=symbol,
            requested_as_of=curr_date,
            retrieved_at=retrieved_at,
        )
        for row in rows:
            date = str(row.get("opendate", "")).strip()
            if not date:
                continue
            rec = {
                "日期": date,
                "净流入额(亿)": _sina_amount_yi(row.get("netamount")),
                "主力净流入(亿)": _sina_amount_yi(row.get("r0_net")),
                "净占比": _sina_ratio_pct(row.get("ratioamount")),
                "超大单净流入(亿)": "",
                "大单净流入(亿)": "",
                "中单净流入(亿)": "",
                "小单净流入(亿)": "",
            }
            for key, label in (
                ("r1_net", "超大单净流入(亿)"),
                ("r2_net", "大单净流入(亿)"),
                ("r3_net", "中单净流入(亿)"),
                ("r4_net", "小单净流入(亿)"),
            ):
                val = row.get(key)
                if val is not None and str(val).strip():
                    rec[label] = _sina_amount_yi(val)
                    has_sub_orders = True
            records.append(rec)
        if not records:
            return None
        df = pd.DataFrame(records)
        if not has_sub_orders:
            df = df.drop(
                columns=[
                    "超大单净流入(亿)",
                    "大单净流入(亿)",
                    "中单净流入(亿)",
                    "小单净流入(亿)",
                ]
            )
        df_recent = chronological(
            take_latest(df, "日期", _SINA_HIST_FUND_FLOW_SHOW), "日期"
        )
        if df_recent is None or df_recent.empty:
            return None
        latest_day = pd.to_datetime(df_recent["日期"], errors="coerce").max()
        latest_str = latest_day.date().isoformat() if pd.notna(latest_day) else curr_date
        header = (
            f"【备用数据源：新浪历史/收盘数据】{symbol} 近{len(df_recent)}日主力资金净流向"
            f"（截至于 {curr_date}，最新数据日 {latest_str}，单位：亿元）：\n"
            f"{df_recent.to_string(index=False)}"
        )
        if not has_sub_orders:
            header += "\n（新浪历史接口未提供超大单/大单/中单/小单明细）"
        consensus = build_consensus_evidence(
            evidence,
            symbol=symbol,
            requested_as_of=curr_date,
            field="r0_net",
        )
        return FundFlowText(
            header,
            evidence=evidence,
            evidence_meta={
                "symbol": symbol,
                "requested_as_of": curr_date,
                "retrieved_at": retrieved_at,
                "source": "sina_historical",
                "algorithm_group": "legacy_web_algorithm",
                "source_family": "sina_web",
                "unit": "亿元",
                "status": "available" if len(evidence) >= _SINA_HIST_FUND_FLOW_SHOW else "partial",
                "consensus": consensus,
            },
        )

    def _augment_new_algorithm_sources(
        self,
        value: FundFlowText,
        *,
        ak,
        symbol: str,
        curr_date: str,
        code: str,
        is_historical: bool,
    ) -> FundFlowText:
        """Attach optional THS same-day evidence without changing EM fallback semantics."""
        evidence = list(getattr(value, "fund_flow_evidence", []) or [])
        metadata = dict(getattr(value, "fund_flow_evidence_meta", {}) or {})
        if not evidence:
            metadata["manual_calibration_gap"] = build_gap_meta(
                symbol=symbol,
                requested_as_of=curr_date,
                source="sina_app_manual_calibration",
                status="blocked",
                reason="新浪 App 无可验证公开接口，人工截图不能写入自动 evidence",
                retrieved_at=self._sina_retrieved_at(),
                algorithm_group="new_algorithm_group",
                period_kind="realtime_single_day",
            )
            return FundFlowText(str(value), evidence=evidence, evidence_meta=metadata)
        if is_historical:
            return value
        try:
            with AKSHARE_CALL_LOCK:
                snapshot = ak.stock_fund_flow_individual(symbol="即时")
            if snapshot is None or snapshot.empty or "股票代码" not in snapshot.columns:
                metadata["manual_calibration_gap"] = build_gap_meta(
                    symbol=symbol,
                    requested_as_of=curr_date,
                    source="sina_app_manual_calibration",
                    status="blocked",
                    reason="新浪 App 没有可验证公开资金流接口；截图仅作人工校准，未生成自动 evidence",
                    retrieved_at=self._sina_retrieved_at(),
                    algorithm_group="new_algorithm_group",
                    period_kind="realtime_single_day",
                )
                return FundFlowText(str(value), evidence=evidence, evidence_meta=metadata)
            matched = snapshot[snapshot["股票代码"].astype(str).str.zfill(6) == code.zfill(6)]
            if matched.empty or "净额" not in matched.columns:
                metadata["manual_calibration_gap"] = build_gap_meta(
                    symbol=symbol,
                    requested_as_of=curr_date,
                    source="sina_app_manual_calibration",
                    status="blocked",
                    reason="新浪 App 截图无法由可验证公开接口复现；未将人工截图写入自动共识",
                    retrieved_at=self._sina_retrieved_at(),
                    algorithm_group="new_algorithm_group",
                    period_kind="realtime_single_day",
                )
                return FundFlowText(str(value), evidence=evidence, evidence_meta=metadata)
            ths_row = matched.iloc[0]
            ths_records = build_ths_evidence(
                [{
                    "股票代码": code,
                    "日期": curr_date,
                    "净额": ths_row.get("净额"),
                    "单位": "亿元",
                    "period_kind": "realtime_single_day",
                    "window": "1d",
                }],
                symbol=symbol,
                requested_as_of=curr_date,
                retrieved_at=self._sina_retrieved_at(),
            )
            if not ths_records:
                metadata["manual_calibration_gap"] = build_gap_meta(
                    symbol=symbol,
                    requested_as_of=curr_date,
                    source="sina_app_manual_calibration",
                    status="blocked",
                    reason="同花顺快照未提供可比主力字段；新浪 App 截图仅作人工校准",
                    retrieved_at=self._sina_retrieved_at(),
                    algorithm_group="new_algorithm_group",
                    period_kind="realtime_single_day",
                )
                return FundFlowText(str(value), evidence=evidence, evidence_meta=metadata)
            # THS instant ``净额`` is total-net, while EM's value is r0_net;
            # retain both raw sources but never place them in one consensus field.
            all_records = evidence + ths_records
            metadata["consensus"] = build_consensus_evidence(
                evidence,
                symbol=symbol,
                requested_as_of=curr_date,
                field="r0_net",
            )
            metadata["total_net_consensus"] = build_consensus_evidence(
                ths_records,
                symbol=symbol,
                requested_as_of=curr_date,
                field="netamount",
            )
            metadata["new_algorithm_sources"] = [
                "eastmoney_individual_fund_flow",
                "ths_instant_snapshot",
            ]
            return FundFlowText(str(value), evidence=all_records, evidence_meta=metadata)
        except Exception as exc:
            metadata["consensus_source_warning"] = f"ths_instant_snapshot: {type(exc).__name__}"
            return FundFlowText(str(value), evidence=evidence, evidence_meta=metadata)

    def _format_individual_fund_flow_em(
        self,
        df: "pd.DataFrame",
        symbol: str,
        curr_date: str,
        cutoff,
        *,
        source: str = "eastmoney_individual_fund_flow",
    ) -> str | None:
        """Format the Eastmoney per-day fund-flow series truncated to curr_date.

        Returns evidence-bearing ``FundFlowText`` only when usable records
        remain on or before ``curr_date``.  When nothing usable remains
        (``curr_date`` is outside the ~120-trading-day window or dates/amounts
        are unusable), it may return an ordinary failure string or ``None``;
        the caller must keep the failure detail and continue its fallback chain.
        """
        date_col = "日期" if "日期" in df.columns else None
        value_col = "主力净流入-净额" if "主力净流入-净额" in df.columns else None
        if date_col is None or value_col is None:
            return None

        dates = pd.to_datetime(df[date_col], errors="coerce")

        def _raw_amount_text(value) -> str:
            if value is None:
                return ""
            try:
                if pd.isna(value):
                    return ""
            except (TypeError, ValueError):
                return ""
            return str(value).strip()

        raw_values = df[value_col].map(_raw_amount_text)
        values = raw_values.map(_sina_decimal)
        valid = dates.notna() & values.map(
            lambda value: value is not None and value.is_finite()
        )
        df = df.loc[valid].copy()
        df[date_col] = dates[valid]
        # Keep the vendor text beside the date-filtering frame.  The evidence
        # builder must receive this text, not a float64 conversion of f52.
        df["__r0_net_raw"] = raw_values[valid]
        df = df[df[date_col] <= pd.Timestamp(cutoff)]
        if df.empty:
            return (
                f"【数据获取失败】资金流数据仅覆盖最近约 120 个交易日，"
                f"{curr_date} 超出可得范围，{symbol} 本项不可用。"
            )

        df_recent = chronological(take_latest(df, date_col, 5), date_col)
        if df_recent is None or df_recent.empty:
            return None
        latest_day = pd.to_datetime(df_recent[date_col], errors="coerce").max()
        latest_str = latest_day.date().isoformat() if pd.notna(latest_day) else curr_date
        retrieved_at = self._sina_retrieved_at()
        evidence_frame = df_recent.copy()
        evidence_frame[value_col] = evidence_frame["__r0_net_raw"]
        evidence_frame = evidence_frame.drop(columns=["__r0_net_raw"])
        display_frame = df_recent.drop(columns=["__r0_net_raw"])
        evidence = build_em_evidence(
            evidence_frame,
            symbol=symbol,
            requested_as_of=curr_date,
            retrieved_at=retrieved_at,
            source=source,
        )
        consensus = build_consensus_evidence(
            evidence,
            symbol=symbol,
            requested_as_of=curr_date,
            field="r0_net",
        )
        source_prefix = (
            "【备用数据源：东方财富直连】" if source == "eastmoney_direct" else ""
        )
        reason = (
            "东方财富直连仅将 f52 映射为主力净额 r0_net；"
            "f53-f56 仅保留原始发现值，未将其等同于总净额 netamount"
            if source == "eastmoney_direct"
            else "东方财富来源仅提供主力净额；未将其等同于总净额 netamount"
        )
        return FundFlowText(
            (
                f"{source_prefix}{symbol} 近5日主力资金净流向"
                f"（截至于 {curr_date}，最新数据日 {latest_str}）：\n"
                f"{display_frame.to_string(index=False)}"
            ),
            evidence=evidence,
            evidence_meta={
                "symbol": symbol,
                "requested_as_of": curr_date,
                "retrieved_at": retrieved_at,
                "source": source,
                "algorithm_group": "new_algorithm_group",
                "source_family": "eastmoney",
                "raw_unit": "元",
                "unit": "亿元",
                "field": "r0_net",
                "period_kind": "historical_daily",
                "window": "1d",
                "as_of": latest_str,
                "actual_as_of": latest_str,
                "status": "available" if source == "eastmoney_direct" else "partial",
                "consensus": consensus,
                "reason": reason,
            },
        )

    def get_lhb_detail(self, symbol: str, date: str) -> str:
        """获取龙虎榜数据，非异动日返回空提示（属正常）。

        注意：此接口依赖 akshare，可能因东方财富 API 变化而暂时不可用。
        查询日先规整到交易日，并向前回退以覆盖发布延迟。
        """
        source_name = "akshare.stock_lhb_detail_em"
        title = "龙虎榜明细"
        if not date:
            res = DataResult(
                ok=False,
                data=None,
                error="缺少 date/curr_date，内部层不得默认今天",
                source=source_name,
                title=title,
            )
            return res.to_prompt()
        request_date = date
        code = self._normalize_symbol(symbol)
        ak = self._ak()

        def _fetch_one(day: str):
            date_fmt = day.replace("-", "")
            try:
                with AKSHARE_CALL_LOCK:
                    df = ak.stock_lhb_detail_em(start_date=date_fmt, end_date=date_fmt)
            except TypeError as exc:
                # akshare 当日数据未更新时，data_json["result"] 为 None
                raise DateDataUnavailable(f"{day} 龙虎榜数据尚未更新") from exc
            if df is None or df.empty:
                raise DateDataUnavailable(f"{day} 全市场无龙虎榜数据")
            if "代码" in df.columns:
                stock_df = df[df["代码"].astype(str).str.zfill(6) == code.zfill(6)]
            else:
                stock_df = df
            if stock_df is None or stock_df.empty:
                # 有全市场榜但该票未上榜：属正常，不继续回退
                return f"{symbol} 在 {day} 无龙虎榜数据（非异动日属正常）。"
            return (
                f"{symbol} 龙虎榜明细（{day}）：\n"
                f"{stock_df.head(20).to_string(index=False)}"
            )

        result = fetch_with_date_fallback(
            _fetch_one, request_date, max_back=5, start_offset=0
        )
        if not result.ok:
            return self._lhb_sina_fallback(symbol, code, request_date, result.error)

        body = str(result.data)
        header = result.date_header()
        msg = f"{header}\n{body}" if header else body
        res = DataResult(
            ok=True,
            data=msg,
            source=source_name,
            title=title,
            as_of=result.as_of,
        )
        return res.to_prompt()

    def _lhb_sina_fallback(self, symbol: str, code: str, request_date: str, em_error: str) -> str:
        """东财龙虎榜失败时的新浪备用源（``stock_lhb_detail_daily_sina``）。"""
        source_name = "akshare.stock_lhb_detail_daily_sina"
        title = "龙虎榜明细"
        try:
            ak = self._ak()
            date_fmt = request_date.replace("-", "")
            with AKSHARE_CALL_LOCK:
                df = ak.stock_lhb_detail_daily_sina(date=date_fmt)
            if df is None or df.empty:
                raise DateDataUnavailable(f"{request_date} 新浪龙虎榜无数据")
            if "股票代码" in df.columns:
                stock_df = df[df["股票代码"].astype(str).str.zfill(6) == code.zfill(6)]
            else:
                stock_df = df
            if stock_df is None or stock_df.empty:
                res = DataResult(
                    ok=True,
                    data=f"{symbol} 在 {request_date} 无龙虎榜数据（非异动日属正常）。",
                    source=source_name,
                    title=title,
                    as_of=request_date,
                )
                return res.to_prompt()
            res = DataResult(
                ok=True,
                data=(
                    f"{symbol} 龙虎榜明细（{request_date}，新浪备用源）：\n"
                    f"{stock_df.head(20).to_string(index=False)}"
                ),
                source=source_name,
                title=title,
                as_of=request_date,
            )
            return res.to_prompt()
        except Exception as exc:
            # 东财 + 新浪备用源均失败：显式 VendorFail，链路继续到备用 vendor
            # （如 cn_fuyao），而不是用纯字符串把失败伪装成成功命中。
            return VendorFail(
                f"龙虎榜数据获取失败：{em_error}；"
                f"新浪备用源失败：{type(exc).__name__}: {exc}"
            )

    def get_zt_pool(self, date: str) -> str:
        """获取涨停板情绪池，反映市场整体情绪温度。

        ``stock_zt_pool_em`` 仅保留近窗截面（探测：约 15 个交易日，更早全空），
        不是可回测的历史序列。历史日期分析直接拒绝，避免 5 日回退空结果
        被当成「当日无涨停」的情绪信号。
        当日分析仍可规整交易日 + 发布延迟回退，并写明实际数据日。
        """
        source_name = "akshare.stock_zt_pool_em"
        title = "涨停板情绪池"
        if not date:
            res = DataResult(
                ok=False,
                data=None,
                error="缺少 date/curr_date，内部层不得默认今天",
                source=source_name,
                title=title,
            )
            return res.to_prompt()
        refusal = snapshot_historical_refusal(
            date,
            source_label="涨停板情绪池（仅提供近窗，非全历史）",
        )
        if refusal:
            return VendorRefuse(refusal, allow_peers=("cn_fuyao",))
        request_date = date
        ak = self._ak()

        def _fetch_one(day: str):
            try:
                with AKSHARE_CALL_LOCK:
                    df = ak.stock_zt_pool_em(date=day.replace("-", ""))
            except Exception as exc:
                raise DateDataUnavailable(f"{type(exc).__name__}: {exc}") from exc
            if df is None or df.empty:
                raise DateDataUnavailable(f"{day} 涨停板情绪池暂无数据")
            count = len(df)
            body = f"{day} 涨停家数：{count}\n"
            if "连板数" in df.columns:
                lianban = df["连板数"].value_counts().sort_index()
                body += f"连板分布：\n{lianban.head(10).to_string()}"
            return body

        result = fetch_with_date_fallback(
            _fetch_one, request_date, max_back=5, start_offset=0
        )
        if not result.ok:
            # 东财（及内部备用）整体失败：显式 VendorFail，链路继续到备用 vendor
            # （如 cn_fuyao），而不是用纯字符串把失败伪装成成功命中。
            return VendorFail(f"涨停板情绪池数据获取失败：{result.error}")

        header = result.date_header()
        msg = f"{header}\n{result.data}" if header else str(result.data)
        res = DataResult(
            ok=True,
            data=msg,
            source=source_name,
            title=title,
            as_of=result.as_of,
        )
        return res.to_prompt()

    def get_hot_stocks_xq(self, curr_date: str = None) -> str:
        """获取雪球热搜股票（当前热度快照）。

        历史日期分析直接拒绝：接口无历史截面。
        """
        refusal = snapshot_historical_refusal(
            curr_date, source_label="雪球热搜"
        )
        if refusal:
            return refusal
        try:
            ak = self._ak()
            with AKSHARE_CALL_LOCK:
                df = ak.stock_hot_follow_xq(symbol="本周新增")
            if df is None or df.empty:
                return "雪球热搜数据暂不可用。"
            return f"雪球热搜前20：\n{df.head(20).to_string(index=False)}"
        except Exception as exc:
            return f"雪球热搜数据获取失败：{type(exc).__name__}: {exc}"

    # --- Data Source Extensions (Institutional Risk, Chip & Fund Flow) ---

    def get_restricted_release(self, symbol: str, curr_date: str = None) -> str:
        """获取限售股解禁数据与近期解禁风险。"""
        source_name = "akshare.stock_restricted_release_detail_em"
        title = "限售股解禁风险"
        if not curr_date:
            res = DataResult(
                ok=False,
                data=None,
                error="缺少 curr_date，内部层不得默认今天",
                source=source_name,
                title=title,
            )
            return res.to_prompt()
        try:
            code = self._normalize_symbol(symbol)
            ak = self._ak()

            # Start: 30 days ago, End: 60 days ahead
            dt_curr = datetime.strptime(curr_date, "%Y-%m-%d")
            start_str = (dt_curr - timedelta(days=30)).strftime("%Y%m%d")
            end_str = (dt_curr + timedelta(days=60)).strftime("%Y%m%d")

            with AKSHARE_CALL_LOCK:
                df = ak.stock_restricted_release_detail_em(start_date=start_str, end_date=end_str)

            if df is None or df.empty:
                res = DataResult(ok=True, data=None, source=source_name, title=title)
                return res.to_prompt()

            # Filter for specific stock code
            stock_df = df[df["股票代码"].astype(str).str.zfill(6) == code.zfill(6)]
            if stock_df.empty:
                res = DataResult(ok=True, data="【解禁排查】距当前分析日期前后60日内无限售股解禁记录，无重大解禁冲击风险。", source=source_name, title=title)
                return res.to_prompt()

            summary_lines = [f"【限售解禁风险预警】找到 {len(stock_df)} 条近期解禁记录："]
            for _, row in stock_df.iterrows():
                rel_date = row.get("解禁时间", "未知日期")
                rel_ratio = row.get("占解禁前流通市值比例", "未知")
                rel_type = row.get("限售股类型", "限售股")
                summary_lines.append(f"- 解禁日期: {rel_date} | 类型: {rel_type} | 占比流通市值: {rel_ratio}%")

            res = DataResult(ok=True, data="\n".join(summary_lines), source=source_name, title=title)
            return res.to_prompt()
        except Exception as exc:
            res = DataResult(ok=False, data=None, error=f"{type(exc).__name__}: {exc}", source=source_name, title=title)
            return res.to_prompt()

    def get_share_pledge(self, symbol: str, curr_date: str = None) -> str:
        """获取大股东股权质押比例与质押风险（全市场快照）。

        历史日期分析直接拒绝：接口无 date 参数，返回的是当前质押截面。
        """
        source_name = "akshare.stock_gpzy_pledge_ratio_em"
        title = "股权质押风险"
        refusal = snapshot_historical_refusal(
            curr_date, source_label="股权质押（全市场快照）"
        )
        if refusal:
            res = DataResult(
                ok=False,
                data=None,
                error=refusal.replace("【数据获取失败】", "", 1).strip()
                if refusal.startswith("【数据获取失败】")
                else refusal,
                source=source_name,
                title=title,
            )
            # Keep the fixed phrase in the prompt body for scanners / models.
            return refusal if refusal.startswith("【数据获取失败】") else res.to_prompt()
        try:
            code = self._normalize_symbol(symbol)
            ak = self._ak()

            with AKSHARE_CALL_LOCK:
                df = ak.stock_gpzy_pledge_ratio_em()

            if df is None or df.empty:
                res = DataResult(ok=True, data=None, source=source_name, title=title)
                return res.to_prompt()

            stock_df = df[df["股票代码"].astype(str).str.zfill(6) == code.zfill(6)]
            if stock_df.empty:
                res = DataResult(ok=True, data="【股权质押排查】无大股东高比例质押记录，质押风险处于安全水平。", source=source_name, title=title)
                return res.to_prompt()

            row = stock_df.iloc[0]

            def _field(col: str):
                if col not in stock_df.columns:
                    return None
                val = row[col]
                if pd.isna(val):
                    return None
                text = str(val).strip()
                return text if text != "" else None

            ratio = _field("质押比例")
            count = _field("质押笔数")
            industry = _field("所属行业")

            missing = [name for name, val in (("质押比例", ratio), ("质押笔数", count)) if val is None]
            if missing:
                res = DataResult(
                    ok=False,
                    data=None,
                    error=f"{'、'.join(missing)}字段缺失，质押风险未排查",
                    source=source_name,
                    title=title,
                )
                return res.to_prompt()

            msg = (
                f"【股权质押排查】整体质押比例：{ratio}% "
                f"(质押笔数: {count} 笔, 行业: {industry or '未知'})"
            )
            try:
                ratio_val = float(str(ratio).replace("%", ""))
            except (TypeError, ValueError):
                res = DataResult(
                    ok=False,
                    data=None,
                    error=f"质押比例字段不可解析（raw={ratio!r}），质押风险未排查",
                    source=source_name,
                    title=title,
                )
                return res.to_prompt()

            if ratio_val > 30:
                msg += " ⚠️ [高风险警示] 该股票大股东质押比例超30%，需高度警惕平仓与流动性风险。"

            res = DataResult(ok=True, data=msg, source=source_name, title=title)
            return res.to_prompt()
        except Exception as exc:
            res = DataResult(ok=False, data=None, error=f"{type(exc).__name__}: {exc}", source=source_name, title=title)
            return res.to_prompt()

    def get_earnings_forecast(self, symbol: str, curr_date: str = None) -> str:
        """获取上市公司业绩预告与业绩快报。

        报告期按分析日前最近一个**已关闭**的预告披露窗口选取（非 year-1 年报硬编码）。
        文案标明「查询报告期 = …」，并区分「当期无预告」与「查询失败/未知」。
        """
        source_name = "akshare.stock_yjyg_em"
        title = "业绩预告与快报"
        if not curr_date:
            res = DataResult(
                ok=False,
                data=None,
                error="缺少 curr_date，内部层不得默认今天",
                source=source_name,
                title=title,
            )
            return res.to_prompt()
        try:
            code = self._normalize_symbol(symbol)
            ak = self._ak()

            try:
                date_param = resolve_earnings_forecast_report_period(curr_date)
            except ValueError as exc:
                res = DataResult(
                    ok=False,
                    data=None,
                    error=f"无法推导业绩预告报告期：{exc}",
                    source=source_name,
                    title=title,
                )
                return res.to_prompt()

            period_label = format_report_period_label(date_param)

            with AKSHARE_CALL_LOCK:
                df = ak.stock_yjyg_em(date=date_param)

            header = f"查询报告期 = {date_param}（{period_label}）"

            if df is None or df.empty:
                # Market-wide empty for a standard period is treated as query failure /
                # unknown — not "confirmed no forecast for this ticker".
                res = DataResult(
                    ok=False,
                    data=None,
                    error=(
                        f"{header}；全市场业绩预告池为空或接口无返回，"
                        "预告情况未知，不得据此判断无预告"
                    ),
                    source=source_name,
                    title=title,
                )
                return res.to_prompt()

            stock_df = df[df["股票代码"].astype(str).str.zfill(6) == code.zfill(6)]
            if stock_df.empty:
                res = DataResult(
                    ok=True,
                    data=(
                        f"【业绩预告排查】{header}。该标的在本报告期暂无业绩预警/预增公告"
                        "（查询成功，确认无预告）。"
                    ),
                    source=source_name,
                    title=title,
                )
                return res.to_prompt()

            cutoff = pd.to_datetime(curr_date, errors="coerce")
            kept_lines: list[str] = []
            for _, row in stock_df.iterrows():
                tp = row.get("预告类型", "")
                chg = row.get("业绩变动", "")
                reason = row.get("业绩变动原因", "")
                ann_date = row.get("公告日期", "")
                ann_dt = pd.to_datetime(ann_date, errors="coerce")
                if pd.isna(ann_dt):
                    _provider_logger.warning(
                        "get_earnings_forecast: unparseable 公告日期=%r symbol=%s; skip row",
                        ann_date,
                        symbol,
                    )
                    continue
                if pd.notna(cutoff) and ann_dt.normalize() > cutoff.normalize():
                    continue  # Historical date truncation (datetime, not string)
                kept_lines.append(
                    f"- 公告日: {ann_date} | 类型: {tp} | 变动: {chg}\n  原因摘要: {str(reason)[:100]}"
                )
            if not kept_lines:
                lines = [
                    f"【业绩预告排查】{header}。在分析日截断后无可用预告记录"
                    "（公告日均晚于分析日或无法解析）。"
                ]
            else:
                lines = [
                    f"【业绩预告/快报】{header}。找到 {len(kept_lines)} 条预告记录："
                ] + kept_lines

            res = DataResult(ok=True, data="\n".join(lines), source=source_name, title=title)
            return res.to_prompt()
        except Exception as exc:
            res = DataResult(ok=False, data=None, error=f"{type(exc).__name__}: {exc}", source=source_name, title=title)
            return res.to_prompt()

    def get_shareholder_count(self, symbol: str, curr_date: str = None) -> str:
        """获取股东户数变动与筹码集中度。

        curr_date 必填：缺参时若不过滤会直接 take_latest 最新 4 期，造成历史分析前视。
        """
        source_name = "akshare.stock_zh_a_gdhs_detail_em"
        title = "股东户数与筹码集中度"
        try:
            if not curr_date:
                res = DataResult(
                    ok=False,
                    data=None,
                    error="缺少 curr_date，拒绝返回未截断的最新股东户数（防止历史分析前视）",
                    source=source_name,
                    title=title,
                )
                return res.to_prompt()

            code = self._normalize_symbol(symbol)
            ak = self._ak()

            with AKSHARE_CALL_LOCK:
                df = ak.stock_zh_a_gdhs_detail_em(symbol=code)

            if df is None or df.empty:
                res = DataResult(ok=True, data=None, source=source_name, title=title)
                return res.to_prompt()

            # Truncate by curr_date (datetime compare, not string)
            date_col = "股东户数公告日期" if "股东户数公告日期" in df.columns else None
            if date_col:
                cutoff = pd.to_datetime(curr_date, errors="coerce")
                if pd.isna(cutoff):
                    res = DataResult(
                        ok=False,
                        data=None,
                        error=f"curr_date 无法解析：{curr_date!r}",
                        source=source_name,
                        title=title,
                    )
                    return res.to_prompt()
                ann = pd.to_datetime(df[date_col], errors="coerce")
                df = df[ann.notna() & (ann <= cutoff)]

            if df.empty:
                res = DataResult(ok=True, data=None, source=source_name, title=title)
                return res.to_prompt()

            if not date_col:
                res = DataResult(
                    ok=False,
                    data=None,
                    error="缺少股东户数公告日期列，无法取最新记录",
                    source=source_name,
                    title=title,
                )
                return res.to_prompt()

            # select latest N, then render oldest→newest for trend readability
            recent_df = chronological(take_latest(df, date_col, 4), date_col)
            if recent_df is None or recent_df.empty:
                res = DataResult(ok=True, data=None, source=source_name, title=title)
                return res.to_prompt()
            lines = [f"【股东户数与筹码集中度】最近 {len(recent_df)} 期户数变动："]
            for _, row in recent_df.iterrows():
                dt = row.get("股东户数统计截止日", "")
                cnt = row.get("股东户数-本次", "")
                chg_ratio = row.get("股东户数-增减比例", "")
                avg_val = row.get("户均持股市值", "")
                lines.append(f"- 截止日: {dt} | 股东户数: {cnt} | 较上期变动: {chg_ratio}% | 户均市值: {avg_val} 元")

            res = DataResult(ok=True, data="\n".join(lines), source=source_name, title=title)
            return res.to_prompt()
        except Exception as exc:
            res = DataResult(ok=False, data=None, error=f"{type(exc).__name__}: {exc}", source=source_name, title=title)
            return res.to_prompt()

    def get_margin_trading(self, symbol: str, curr_date: str = None) -> str:
        """获取融资融券交易明细。

        查询日先规整到交易日；融资融券明细有发布延迟，默认至少回退 1 个交易日
        起查，并在窗口内继续向前尝试。实际数据日写入 as_of 与正文【数据日期】。
        """
        source_name = "akshare.stock_margin_detail_sse/szse"
        title = "融资融券交易"
        if not curr_date:
            res = DataResult(
                ok=False,
                data=None,
                error="缺少 curr_date，内部层不得默认今天",
                source=source_name,
                title=title,
            )
            return res.to_prompt()
        request_date = curr_date
        code = self._normalize_symbol(symbol)
        ak = self._ak()

        def _fetch_one(day: str):
            date_fmt = day.replace("-", "")
            try:
                with AKSHARE_CALL_LOCK:
                    if code.startswith("6"):
                        df = ak.stock_margin_detail_sse(date=date_fmt)
                    else:
                        df = ak.stock_margin_detail_szse(date=date_fmt)
            except Exception as exc:
                # 空表赋列名等 akshare 内部 ValueError 视为该日无数据
                raise DateDataUnavailable(f"{type(exc).__name__}: {exc}") from exc

            if df is None or df.empty:
                raise DateDataUnavailable(f"{day} 融资融券明细为空")

            code_col = None
            for cand in ("标的证券代码", "证券代码", "股票代码", "代码"):
                if cand in df.columns:
                    code_col = cand
                    break
            if code_col is None:
                raise DateDataUnavailable(f"{day} 融资融券明细缺少证券代码列")

            stock_df = df[df[code_col].astype(str).str.zfill(6) == code.zfill(6)]
            if stock_df is None or stock_df.empty:
                # 全市场有表但该票无明细：视为该日已发布、标的无记录，停止回退
                return f"【融资融券】{day} 暂无该标的融资融券明细。"

            row = stock_df.iloc[0]

            def _margin_field(col: str):
                if col not in stock_df.columns:
                    return None
                val = row[col]
                if pd.isna(val):
                    return None
                text = str(val).strip()
                return text if text != "" else None

            rzye = _margin_field("融资余额")
            rzbuy = _margin_field("融资买入额")
            rqyl = _margin_field("融券余量")
            missing = [
                name
                for name, val in (
                    ("融资余额", rzye),
                    ("融资买入额", rzbuy),
                    ("融券余量", rqyl),
                )
                if val is None
            ]
            if missing:
                return (
                    f"【融资融券】{day} 关键字段缺失（{'、'.join(missing)}），"
                    f"融资融券风险未排查"
                )
            return (
                f"【融资融券数据】日期: {day} | 融资余额: {rzye} 元"
                f" | 融资买入额: {rzbuy} 元 | 融券余量: {rqyl}"
            )

        # 从规整后的交易日起查；融资融券常有 1 日发布延迟，窗口内向前回退
        result = fetch_with_date_fallback(
            _fetch_one, request_date, max_back=5, start_offset=0
        )
        if not result.ok:
            res = DataResult(
                ok=False,
                data=None,
                error=f"融资融券数据获取失败：{result.error}",
                source=source_name,
                title=title,
            )
            return res.to_prompt()

        header = result.date_header()
        msg = f"{header}\n{result.data}" if header else str(result.data)
        res = DataResult(
            ok=True,
            data=msg,
            source=source_name,
            title=title,
            as_of=result.as_of,
        )
        return res.to_prompt()

    def get_northbound_flow(self, symbol: str, curr_date: str = None) -> str:
        """北向/陆股通个股每日持股明细已制度性停更，不再请求网络。

        2024 年 8 月起沪深港通个股持股由每日披露改为季度披露；对 600519/000001/
        300750/688981 实测 stock_hsgt_individual_em 的 max(持股日期) 均为 2024-08-16。
        继续调用只会浪费约 12s 并返回过期日频数据。季度持股源另议，不在此接口复活。
        """
        source_name = "akshare.stock_hsgt_individual_em"
        title = "北向资金持股变动"
        res = DataResult(
            ok=False,
            data=None,
            error=(
                "沪深港通个股每日持股明细自 2024 年 8 月起停止披露，本项不可用。"
                "如需北向数据请使用季度持股口径，注意频率为季度而非每日。"
            ),
            source=source_name,
            title=title,
            as_of="2024-08-16",
        )
        return res.to_prompt()
