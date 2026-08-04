import logging
import re
import time
import threading
import contextvars
from datetime import datetime, timedelta

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
    is_historical_analysis_date,
    snapshot_historical_refusal,
)
from ..utils import chronological, shrink_table, take_latest
from ..vendor_result import (
    VendorEmpty,
    VendorFail,
    VendorRefuse,
    result_to_prompt,
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

    def __exit__(self, *exc_info):
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
        out["Dividends"] = 0.0
        out["Stock Splits"] = 0.0
        out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")

        header = f"# Stock data for {symbol} from {start} to {end}\n"
        header += f"# Total records: {len(out)}\n\n"
        return header + out.to_csv(index=False)

    @staticmethod
    def _slice_hist_df(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        start_dt = pd.to_datetime(start_date, errors="coerce")
        end_dt = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(start_dt) or pd.isna(end_dt):
            return df
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"])
        out = out[(out["Date"] >= start_dt) & (out["Date"] <= end_dt)]
        return out.sort_values("Date").reset_index(drop=True)

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
        with AKSHARE_CALL_LOCK:
            ak = self._ak()
            code = self._normalize_symbol(ticker)
            try:
                df = ak.stock_news_em(symbol=code)
                if df is None or df.empty:
                    return VendorEmpty(f"No news found for {ticker}")

                date_col = "发布时间" if "发布时间" in df.columns else None
                if date_col is not None:
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
                    df = df[(df[date_col] >= start_dt) & (df[date_col] < end_dt)]

                if df.empty:
                    return VendorEmpty(
                        f"No news found for {ticker} between {start_date} and {end_date}"
                    )

                if date_col is not None:
                    df = chronological(take_latest(df, date_col, 20), date_col)
                else:
                    df = df.head(20)

                rows = []
                for _, row in df.iterrows():
                    title = str(row.get("新闻标题", row.get("标题", "No title")))
                    src = str(row.get("文章来源", row.get("来源", "Unknown")))
                    summary = str(row.get("新闻内容", row.get("内容", "")))
                    link = str(row.get("新闻链接", row.get("链接", "")))
                    rows.append(f"### {title} (source: {src})")
                    if summary and summary != "nan":
                        rows.append(summary[:400])
                    if link and link != "nan":
                        rows.append(f"Link: {link}")
                    rows.append("")

                return f"## {ticker} 新闻（{start_date} 至 {end_date}）：\n\n" + "\n".join(rows)
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
        if val is None:
            return None
        try:
            f = float(val)
            return f if not pd.isna(f) else None
        except (ValueError, TypeError):
            return None

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

        东财 ``stock_individual_fund_flow`` 对当前 IP 间歇不可达
        （RemoteDisconnected），失败时回退到新浪全市场即时截面
        ``stock_fund_flow_individual``。新浪仅提供当日快照（无历史序列），
        因此只在非历史分析日作为备用源。
        """
        if not curr_date:
            return (
                f"【数据获取失败】个股资金流向缺少 curr_date，无法做日期截断，"
                f"{symbol} 本项不可用。"
            )
        cutoff = parse_yyyymmdd(curr_date)
        if cutoff is None:
            return f"【数据获取失败】个股资金流向 curr_date 无法解析：{curr_date!r}"

        ak = self._ak()
        code = self._normalize_symbol(symbol)
        errors: list[str] = []

        # Source 1: 东财（近 120 交易日逐日序列，可按 curr_date 截断）
        try:
            # 沪市：以 5、6、9 开头；其余为深市
            market = "sh" if code[:1] in ("5", "6", "9") else "sz"
            with AKSHARE_CALL_LOCK:
                df = ak.stock_individual_fund_flow(stock=code, market=market)
            if df is None or df.empty:
                errors.append("stock_individual_fund_flow: empty dataframe")
            else:
                # _format_individual_fund_flow_em never returns None: it returns
                # a formatted report or an explicit 【数据获取失败】 refusal (out of
                # range / unparseable dates), and must not fall back to the
                # same-day Sina snapshot for historical dates (lookahead risk).
                return self._format_individual_fund_flow_em(
                    df, symbol, curr_date, cutoff
                )
        except Exception as exc:
            errors.append(f"stock_individual_fund_flow: {type(exc).__name__}")

        # Source 2: 新浪即时截面（当日快照，无历史序列）
        if is_historical_analysis_date(curr_date):
            return (
                f"【数据获取失败】个股资金流向东财接口失败且新浪备用源为当日快照，"
                f"历史日期 {curr_date} 下无法截断，{symbol} 本项不可用。"
                f"（{'；'.join(errors)}）"
            )
        try:
            with AKSHARE_CALL_LOCK:
                df = ak.stock_fund_flow_individual(symbol="即时")
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            stock_df = df[df["股票代码"].astype(str).str.zfill(6) == code.zfill(6)]
            if stock_df.empty:
                return (
                    f"【备用数据源：新浪】{symbol} 当日资金流向快照无记录"
                    f"（{'；'.join(errors)}）"
                )
            row = stock_df.iloc[0]

            def _v(col: str) -> str:
                if col not in stock_df.columns:
                    return ""
                val = row[col]
                return "" if pd.isna(val) else str(val)

            return (
                f"【备用数据源：新浪】{symbol} 当日主力资金净流向快照"
                f"（{curr_date}，最新价 {_v('最新价')}，涨跌幅 {_v('涨跌幅')}）：\n"
                f"净额: {_v('净额')} | 流入资金: {_v('流入资金')} | "
                f"流出资金: {_v('流出资金')} | 换手率: {_v('换手率')}"
            )
        except Exception as exc:
            errors.append(f"stock_fund_flow_individual: {type(exc).__name__}")

        return f"个股资金流向数据获取失败（东财/新浪均失败：{'；'.join(errors)}）"

    def _format_individual_fund_flow_em(
        self,
        df: "pd.DataFrame",
        symbol: str,
        curr_date: str,
        cutoff,
    ) -> str:
        """Format the Eastmoney per-day fund-flow series truncated to curr_date.

        Returns a formatted report string when usable records remain on or
        before ``curr_date``.  When nothing usable remains (``curr_date`` is
        outside the ~120-trading-day window or dates are unparseable), it
        returns an explicit 【数据获取失败】 refusal string instead — it never
        returns ``None`` and never falls back to a same-day snapshot, so a
        historical-date query is surfaced as a refusal rather than risking
        lookahead bias.  The caller therefore returns this value directly.
        """
        date_col = "日期" if "日期" in df.columns else None
        if date_col is None:
            return f"{symbol} 近期主力资金流向数据缺少日期列，无法判定最新记录。"

        dates = pd.to_datetime(df[date_col], errors="coerce")
        df = df.loc[dates.notna()].copy()
        df[date_col] = dates[dates.notna()]
        df = df[df[date_col] <= pd.Timestamp(cutoff)]
        if df.empty:
            return (
                f"【数据获取失败】资金流数据仅覆盖最近约 120 个交易日，"
                f"{curr_date} 超出可得范围，{symbol} 本项不可用。"
            )

        df_recent = chronological(take_latest(df, date_col, 5), date_col)
        if df_recent is None or df_recent.empty:
            return f"{symbol} 近期主力资金流向数据日期不可解析。"
        latest_day = pd.to_datetime(df_recent[date_col], errors="coerce").max()
        latest_str = latest_day.date().isoformat() if pd.notna(latest_day) else curr_date
        return (
            f"{symbol} 近5日主力资金净流向（截至于 {curr_date}，最新数据日 {latest_str}）：\n"
            f"{df_recent.to_string(index=False)}"
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
            res = DataResult(
                ok=False,
                data=None,
                error=(
                    f"龙虎榜数据获取失败：{em_error}；"
                    f"新浪备用源失败：{type(exc).__name__}: {exc}"
                ),
                source=source_name,
                title=title,
            )
            return res.to_prompt()

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
            return refusal
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
            res = DataResult(
                ok=False,
                data=None,
                error=f"涨停板情绪池数据获取失败：{result.error}",
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
