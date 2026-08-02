"""DataCollector: fetch all data once, serve windowed views to analyst agents."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
import copy
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Optional
import json
import os
import threading
import time
import pandas as pd
from stockstats import wrap
import io

from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_global_news,
    get_insider_transactions,
    get_board_fund_flow,
    get_individual_fund_flow,
    get_lhb_detail,
    get_zt_pool,
    get_hot_stocks_xq,
    get_restricted_release,
    get_share_pledge,
    get_earnings_forecast,
    get_shareholder_count,
    get_margin_trading,
    get_northbound_flow,
)
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.trade_calendar import (
    dedupe_daily_bars,
    is_historical_analysis_date,
)

INDICATORS = [
    "close_50_sma", "close_200_sma", "close_10_ema",
    "rsi", "macd", "boll", "boll_ub", "boll_lb", "atr", "vwma",
]
SHORT_DAYS = 14
LONG_DAYS = 90

# 网络丢包时单个数据源可能永久卡死（SSL 握手/读无超时），必须给整轮抓取
# 设硬上限，否则卡死线程会拿着 per-key 锁把后续同标的分析全部拖死，
# 并逐渐占满 asyncio 默认线程池（生产事故：64/64 全部僵死 → 前端 524）。
FETCH_ALL_TIMEOUT = int(os.getenv("TA_DATA_FETCH_TIMEOUT", "300"))
FETCH_MAX_WORKERS = int(os.getenv("TA_DATA_FETCH_MAX_WORKERS", "10"))
# 超时后只再等一个很短的有界窗口，避免 shutdown(wait=True) 被卡死 worker 拖到无界。
FETCH_ALL_SHUTDOWN_GRACE_SECONDS = float(
    os.getenv("TA_DATA_FETCH_SHUTDOWN_GRACE_SECONDS", "1")
)

import numpy as np

_OHLCV_COLS = ["date", "open", "high", "low", "close", "volume"]


def _normalize_daily_frame(df: Optional[pd.DataFrame], trade_date: str) -> Optional[pd.DataFrame]:
    """Normalize an OHLCV frame to completed bars <= trade_date.

    Column-name based parsing, invalid-date/bad-row removal, dedupe by date,
    and ascending sort happen here so indicators/VPA/prompt never see a
    look-ahead, unparseable, or duplicated bar.

    Returns None when nothing usable remains (missing columns, all rows bad,
    empty after the date filter, or conflicting duplicate dates) so callers can
    surface an explicit unavailable instead of forwarding raw vendor CSV.
    """
    if df is None or df.empty:
        return None
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset({str(c).lower() for c in df.columns}):
        return None
    cols_map = {str(c).lower(): c for c in df.columns}
    out = df.rename(columns={cols_map[t]: t for t in required}).copy()
    out = out[list(required)].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    end_dt = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(end_dt):
        return None
    out = out[out["date"] <= end_dt]
    out = out.sort_values("date")
    try:
        out = dedupe_daily_bars(
            out, "date", ["open", "high", "low", "close", "volume"]
        )
    except ValueError:
        # Conflicting same-date rows: no deterministic choice, refuse the field.
        return None
    return out if not out.empty else None


def _csv_comment_lines(raw_csv: str) -> list[str]:
    """Return the source-metadata comment lines a provider prepended to CSV."""
    if not isinstance(raw_csv, str):
        return []
    return [
        line.rstrip("\r\n")
        for line in raw_csv.splitlines()
        if line.startswith("#")
    ]


def _parse_csv_to_dataframe(raw_csv: str) -> Optional[pd.DataFrame]:
    """Parse raw CSV string into a normalized OHLCV DataFrame.

    Returns None if parsing fails or the CSV is too short/empty.
    """
    if not isinstance(raw_csv, str) or len(raw_csv) <= 50:
        return None
    try:
        df = pd.read_csv(io.StringIO(raw_csv), on_bad_lines='skip', comment='#')
    except Exception:
        return None
    if df.empty:
        return None
    cols_map = {c.lower(): c for c in df.columns}
    rename_dict = {}
    for target in _OHLCV_COLS:
        if target in cols_map:
            rename_dict[cols_map[target]] = target
    df = df.rename(columns=rename_dict)
    return df


# ── VPA (Volume Price Analysis) 预计算 ──────────────────────────


def _compute_vpa_indicators(df: pd.DataFrame, window: int = 20) -> str:
    """Pre-compute Volume Price Analysis indicators from OHLCV DataFrame.

    Returns a human-readable text block for the VPA analyst agent.
    All numerical comparisons are done here so the LLM only needs to
    interpret the results, not do arithmetic.
    """
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        return "VPA 数据不足：缺少 OHLCV 列"

    df = df.copy()
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    if len(df) < window + 5:
        return "VPA 数据不足：历史 K 线数量不够"

    # ── 派生指标 ──
    df["vol_ma"] = df["volume"].rolling(window).mean()
    df["volume_ratio"] = df["volume"] / df["vol_ma"]

    hl_range = df["high"] - df["low"]
    df["bar_spread"] = hl_range / df["close"]  # 实体相对大小
    df["close_position"] = np.where(
        hl_range > 0,
        (df["close"] - df["low"]) / hl_range,
        0.5,
    )
    df["bar_type"] = np.where(
        df["close"] > df["open"], "阳线",
        np.where(df["close"] < df["open"], "阴线", "十字星"),
    )

    # 上下影线比例
    df["upper_shadow"] = np.where(
        hl_range > 0,
        (df["high"] - np.maximum(df["open"], df["close"])) / hl_range,
        0.0,
    )
    df["lower_shadow"] = np.where(
        hl_range > 0,
        (np.minimum(df["open"], df["close"]) - df["low"]) / hl_range,
        0.0,
    )

    # 价格变化率
    df["pct_change"] = df["close"].pct_change()

    # 量能趋势 (5日均量 vs 20日均量)
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_trend_ratio"] = df["vol_ma5"] / df["vol_ma"]

    # 量价一致性
    df["vp_harmony"] = np.where(
        (df["pct_change"] > 0) & (df["volume_ratio"] > 1.0), "一致(涨+放量)",
        np.where(
            (df["pct_change"] < 0) & (df["volume_ratio"] > 1.0), "一致(跌+放量)",
            np.where(
                (df["pct_change"] > 0) & (df["volume_ratio"] < 0.8), "背离(涨+缩量)",
                np.where(
                    (df["pct_change"] < 0) & (df["volume_ratio"] < 0.8), "背离(跌+缩量)",
                    "中性",
                ),
            ),
        ),
    )

    # OBV (On Balance Volume) 简易趋势 — vectorized
    close_diff = df["close"].diff()
    obv_sign = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
    obv_sign[0] = 0
    df["obv"] = (obv_sign * df["volume"].values).cumsum()
    obv_ma = df["obv"].rolling(10).mean()
    obv_trend = "上升" if len(obv_ma.dropna()) >= 2 and obv_ma.iloc[-1] > obv_ma.iloc[-5] else "下降"

    # ── 格式化输出（取最近 N 天）──
    output_days = min(30, len(df) - window)
    recent = df.tail(output_days).copy()

    lines = []
    lines.append(f"## VPA 预计算指标（基于 {window} 日均量基准）\n")
    lines.append(f"**OBV 趋势（10日）**: {obv_trend}")

    # 量能概况
    last = recent.iloc[-1]
    vol_5d = recent["volume"].tail(5).mean()
    vol_20d = last["vol_ma"] if pd.notna(last["vol_ma"]) else 0
    vol_summary = "放量" if vol_5d > vol_20d * 1.2 else ("缩量" if vol_5d < vol_20d * 0.8 else "平稳")
    lines.append(f"**近5日量能趋势**: {vol_summary}（5日均量/20日均量 = {last.get('vol_trend_ratio', 0):.2f}）\n")

    lines.append("### 逐日量价数据\n")
    lines.append("| 日期 | 类型 | 涨跌幅 | 实体大小 | 收盘位置 | 上影线 | 下影线 | 量比 | 量价关系 |")
    lines.append("|------|------|--------|----------|----------|--------|--------|------|----------|")

    for _, row in recent.iterrows():
        dt = row.get("date", "")
        if hasattr(dt, "strftime"):
            dt = dt.strftime("%m-%d")
        else:
            dt = str(dt)[-5:]

        pct = row["pct_change"] * 100 if pd.notna(row["pct_change"]) else 0
        spread_label = "宽" if row["bar_spread"] > 0.03 else ("窄" if row["bar_spread"] < 0.015 else "中")
        cp = row["close_position"]
        cp_label = "高位" if cp > 0.7 else ("低位" if cp < 0.3 else "中位")
        vr = row["volume_ratio"] if pd.notna(row["volume_ratio"]) else 0
        vr_label = f"{vr:.1f}"
        if vr > 2.0:
            vr_label += "(巨量)"
        elif vr > 1.5:
            vr_label += "(明显放量)"
        elif vr > 1.0:
            vr_label += "(温和放量)"
        elif vr < 0.5:
            vr_label += "(极度缩量)"
        elif vr < 0.8:
            vr_label += "(缩量)"

        lines.append(
            f"| {dt} | {row['bar_type']} | {pct:+.1f}% | {spread_label}({row['bar_spread']:.3f}) "
            f"| {cp_label}({cp:.2f}) | {row['upper_shadow']:.2f} | {row['lower_shadow']:.2f} "
            f"| {vr_label} | {row['vp_harmony']} |"
        )

    # ── 关键模式识别 ──
    lines.append("\n### 关键量价模式识别\n")

    # 量价背离检测（近5天）
    last5 = recent.tail(5)
    price_up = (last5["close"].iloc[-1] > last5["close"].iloc[0])
    vol_down = (last5["volume"].iloc[-1] < last5["volume"].iloc[0])
    price_down = (last5["close"].iloc[-1] < last5["close"].iloc[0])
    vol_up = (last5["volume"].iloc[-1] > last5["volume"].iloc[0])

    if price_up and vol_down:
        lines.append("- **⚠ 顶部背离信号**: 近5日价格上涨但成交量递减，上涨动能可能衰竭")
    if price_down and vol_up:
        lines.append("- **⚠ 底部放量信号**: 近5日价格下跌但成交量递增，可能是恐慌抛售或换手")
    if price_down and vol_down:
        lines.append("- **卖压衰竭信号**: 近5日价格下跌且成交量递减，空方力量可能枯竭")
    if price_up and vol_up:
        lines.append("- **健康上涨信号**: 近5日价格上涨且成交量配合递增")

    # Selling climax 检测
    for i in range(-3, 0):
        if i < -len(recent):
            continue
        row = recent.iloc[i]
        if (row.get("volume_ratio", 0) > 2.0
                and row.get("pct_change", 0) < -0.03
                and row.get("close_position", 0.5) > 0.5):
            lines.append(f"- **卖出高潮(Selling Climax)**: {str(row.get('date', ''))[-5:]} 急跌巨量但收盘收回过半，可能是恐慌见底")

    # 高位放量滞涨
    for i in range(-3, 0):
        if i < -len(recent):
            continue
        row = recent.iloc[i]
        if (row.get("volume_ratio", 0) > 1.8
                and abs(row.get("pct_change", 0)) < 0.01
                and row.get("bar_spread", 0) < 0.015):
            lines.append(f"- **放量滞涨**: {str(row.get('date', ''))[-5:]} 巨量但价格几乎不动（窄实体），多空分歧大")

    if not any("**" in l for l in lines[-5:]):
        lines.append("- 近期无显著量价异常模式")

    return "\n".join(lines)


def make_cache_key(ticker: str, trade_date: str) -> str:
    return f"{ticker}_{trade_date}"


def _safe(tool, payload: dict) -> Any:
    start_t = time.time()
    try:
        if hasattr(tool, "invoke"):
            res = tool.invoke(payload)
        else:
            res = tool(**payload)
        duration = time.time() - start_t
        # 仅在耗时较长时输出
        if duration > 0.5:
            print(f"  [Timer] {getattr(tool, 'name', str(tool))} took {duration:.2f}s")
        return res
    except Exception as exc:
        return f"{getattr(tool, 'name', str(tool))} 调用失败：{type(exc).__name__}: {exc}"


def _build_daily_context(df: Optional[pd.DataFrame], trade_date: str) -> Dict[str, Any]:
    """Describe the latest complete daily bar available to the analysis."""
    unavailable = {"as_of": None, "completeness": "unavailable"}
    if df is None or df.empty or "date" not in df.columns:
        return unavailable

    end_dt = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(end_dt):
        return unavailable

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    dates = dates[dates <= end_dt]
    if dates.empty:
        return unavailable

    return {
        "as_of": dates.max().strftime("%Y-%m-%d"),
        "completeness": "completed",
    }


def _unavailable_realtime_context(retrieved_at: Optional[str], error: str) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "source": None,
        "quote_as_of": None,
        "retrieved_at": retrieved_at,
        "error": error,
        "quote": None,
    }


def default_market_data_context() -> Dict[str, Any]:
    """Return a safe context when collection did not provide one."""
    return {
        "daily": {"as_of": None, "completeness": "unavailable"},
        "realtime": {
            "status": "unavailable",
            "source": None,
            "quote_as_of": None,
            "retrieved_at": None,
            "error": "实时行情上下文不可用",
            "quote": None,
        },
        "data_failure_ledger": [],
    }


def _fetch_realtime_context(ticker: str, trade_date: str) -> Dict[str, Any]:
    """Fetch a standalone quote snapshot without changing the daily series."""
    if is_historical_analysis_date(trade_date):
        return {
            "status": "not_applicable",
            "source": None,
            "quote_as_of": None,
            "retrieved_at": None,
            "error": None,
            "quote": None,
        }

    retrieved_at = datetime.now(timezone.utc).isoformat()
    try:
        raw = route_to_vendor("get_realtime_quotes", [ticker], curr_date=trade_date)
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, dict):
            return _unavailable_realtime_context(
                retrieved_at, "实时行情源返回结构异常"
            )

        ticker_code = str(ticker).split(".", 1)[0].upper()
        quote = payload.get(ticker) or payload.get(str(ticker).upper())
        if quote is None:
            for key, value in payload.items():
                if str(key).split(".", 1)[0].upper() == ticker_code:
                    quote = value
                    break
        if not isinstance(quote, dict):
            return _unavailable_realtime_context(
                retrieved_at, "实时行情源未返回目标标的快照"
            )

        source = quote.get("source")
        if source not in {"sina", "eastmoney", "investoday"}:
            return _unavailable_realtime_context(
                retrieved_at, "实时行情源返回 source 字段结构异常"
            )
        price = quote.get("price")
        try:
            valid_price = (
                not isinstance(price, bool)
                and isinstance(price, (int, float))
                and math.isfinite(float(price))
            )
        except (TypeError, ValueError, OverflowError):
            valid_price = False
        if not valid_price:
            return _unavailable_realtime_context(
                retrieved_at, "实时行情源返回 price 字段结构异常"
            )
        quote_as_of = quote.get("quote_time") or quote.get("quote_as_of")
        if quote_as_of is not None and not isinstance(quote_as_of, str):
            return _unavailable_realtime_context(
                retrieved_at, "实时行情源返回 quote_time 字段结构异常"
            )
        return {
            "status": "available",
            "source": source,
            "quote_as_of": quote_as_of if isinstance(quote_as_of, str) else None,
            "retrieved_at": retrieved_at,
            "error": None,
            "quote": quote,
        }
    except Exception as exc:
        return _unavailable_realtime_context(
            retrieved_at, f"实时行情源不可用：{type(exc).__name__}"
        )


_DATA_FAILURE_SOURCE_ORDER = (
    "stock_data",
    "news",
    "global_news",
    "fund_flow_board",
    "fund_flow_individual",
    "lhb",
    "insider_transactions",
    "zt_pool",
    "hot_stocks",
    "restricted_release",
    "share_pledge",
    "earnings_forecast",
    "shareholder_count",
    "margin_trading",
    "northbound_flow",
    "fundamentals",
    "balance_sheet",
    "cashflow",
    "income_statement",
    "realtime",
)
_DATA_FAILURE_MARKERS = (
    "【数据获取失败】",
    "获取失败",
    "调用失败",
    "调用异常",
    "数据拉取超时",
    "拉取失败",
    "抓取失败",
    "接口请求失败",
    "请求失败",
    "数据源不可用",
    "接口不可用",
    "返回结构异常",
    "返回格式异常",
    "服务不可用",
    "服务异常",
    "数据暂不可用",
    "暂时不可用",
    "本项不可用",
    "访问被拒绝",
    "请求被拒绝",
    "连接失败",
    "provider unavailable",
    "provider timeout",
)


def _compact_failure_reason(status: str) -> str:
    """Keep the ledger useful without persisting provider payloads or traces."""
    if status == "timeout":
        return "provider timeout"
    if status == "unavailable":
        return "data source unavailable"
    if status == "refused":
        return "data source refused"
    if status == "failed":
        return "provider call failed"
    return "data source error"


def _classify_failure_value(value: Any) -> Optional[str]:
    """Classify only explicit failures; None/empty/not_applicable stay non-failure."""
    if isinstance(value, dict):
        status = str(value.get("status") or "").strip().lower()
        if status in {"available", "not_applicable", "ok", "completed"}:
            return None
        if status in {"failed", "timeout", "unavailable", "refused", "error"}:
            return status
        return None
    if not isinstance(value, str):
        return None

    normalized = value.strip()
    lowered = normalized.lower()
    if any(marker.lower() in lowered for marker in ("调用失败", "调用异常", "拉取失败", "抓取失败")):
        return "failed"
    if "数据拉取超时" in normalized or "timeout" in lowered or "超时" in normalized:
        return "timeout"
    if any(marker.lower() in lowered for marker in _DATA_FAILURE_MARKERS):
        return "failed"
    return None


def _build_data_failure_ledger(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build stable, serializable failure evidence for the report boundary."""
    if not isinstance(results, dict):
        return []

    entries: list[tuple[int, str, Dict[str, str]]] = []
    source_rank = {source: index for index, source in enumerate(_DATA_FAILURE_SOURCE_ORDER)}
    for source, value in results.items():
        source_name = str(source).strip()
        if not source_name:
            continue
        classified = _classify_failure_value(value)
        if classified is None:
            continue
        status = classified

        reason = _compact_failure_reason(status)
        entries.append(
            (
                source_rank.get(source_name, len(_DATA_FAILURE_SOURCE_ORDER)),
                source_name,
                {
                    "source": source_name,
                    "status": status,
                    "reason": reason,
                    "gap": f"【数据获取失败】{source_name}：{reason}",
                },
            )
        )

    entries.sort(key=lambda item: (item[0], item[1]))
    return [entry for _rank, _source, entry in entries]


def _fetch_all(ticker: str, trade_date: str) -> Dict[str, Any]:
    """Fetch all data sources in parallel.

    Always fetches full data including financial statements, regardless of horizon.
    The horizon only affects the analysis window, not data collection.
    """
    lookback = LONG_DAYS
    end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    # 为了计算指标准确（如 200 SMA），需要比分析窗口更长的历史数据
    fetch_lookback = 365
    start_str = (end_dt - timedelta(days=fetch_lookback)).strftime("%Y-%m-%d")

    tasks: Dict[str, tuple] = {
        "stock_data": (get_stock_data, {"symbol": ticker, "start_date": start_str, "end_date": trade_date}),
        "realtime": (_fetch_realtime_context, {"ticker": ticker, "trade_date": trade_date}),
        "news": (get_news, {"ticker": ticker, "start_date": (end_dt - timedelta(days=lookback)).strftime("%Y-%m-%d"), "end_date": trade_date}),
        "global_news": (get_global_news, {"curr_date": trade_date, "look_back_days": lookback, "limit": 30}),
        "fund_flow_board": (get_board_fund_flow, {"curr_date": trade_date}),
        "fund_flow_individual": (get_individual_fund_flow, {"symbol": ticker, "curr_date": trade_date}),
        "lhb": (get_lhb_detail, {"symbol": ticker, "date": trade_date}),
        "insider_transactions": (get_insider_transactions, {"ticker": ticker, "curr_date": trade_date}),
        "zt_pool": (get_zt_pool, {"date": trade_date}),
        "hot_stocks": (get_hot_stocks_xq, {"curr_date": trade_date}),
        "restricted_release": (get_restricted_release, {"symbol": ticker, "curr_date": trade_date}),
        "share_pledge": (get_share_pledge, {"symbol": ticker, "curr_date": trade_date}),
        "earnings_forecast": (get_earnings_forecast, {"symbol": ticker, "curr_date": trade_date}),
        "shareholder_count": (get_shareholder_count, {"symbol": ticker, "curr_date": trade_date}),
        "margin_trading": (get_margin_trading, {"symbol": ticker, "curr_date": trade_date}),
        "northbound_flow": (get_northbound_flow, {"symbol": ticker, "curr_date": trade_date}),
    }

    # 财务报表类数据始终拉取，Research Manager 根据 horizon 自行判断权重
    tasks.update({
        "fundamentals": (get_fundamentals, {"ticker": ticker, "curr_date": trade_date}),
        "balance_sheet": (get_balance_sheet, {"ticker": ticker, "freq": "quarterly", "curr_date": trade_date}),
        "cashflow": (get_cashflow, {"ticker": ticker, "freq": "quarterly", "curr_date": trade_date}),
        "income_statement": (get_income_statement, {"ticker": ticker, "freq": "quarterly", "curr_date": trade_date}),
    })

    results: Dict[str, Any] = {}
    fetch_start = time.time()
    # 减少并发池大小，避免被反爬
    executor = ThreadPoolExecutor(max_workers=min(FETCH_MAX_WORKERS, len(tasks)))
    try:
        future_to_key = {executor.submit(_safe, tool, payload): key for key, (tool, payload) in tasks.items()}
        done, not_done = futures_wait(set(future_to_key), timeout=FETCH_ALL_TIMEOUT)
        if not_done:
            # cancel_futures 只能移除排队任务，无法中断已运行的 worker；
            # 这里给 shutdown 前的收尾等待一个显式上界，保证 _fetch_all 有界返回。
            _, not_done = futures_wait(
                not_done, timeout=FETCH_ALL_SHUTDOWN_GRACE_SECONDS
            )
        for future in done:
            results[future_to_key[future]] = future.result()
        for future in not_done:
            key = future_to_key[future]
            results[key] = f"{key} 数据拉取超时（>{FETCH_ALL_TIMEOUT}s），本次分析跳过该数据源"
            print(f"  [Warning] {key} fetch timed out after {FETCH_ALL_TIMEOUT}s, skipped")
    finally:
        # 已超时 future 不再等待：wait=False 是这里唯一的硬上界，卡死线程
        # 无法被 Python 线程池强杀，但不能继续占用本次 fetch 的锁等待预算。
        executor.shutdown(wait=False, cancel_futures=True)

    data_failure_ledger = _build_data_failure_ledger(results)

    # ── Parse CSV once, reuse for indicators and VPA ──────────────────
    raw_csv = results.get("stock_data", "")
    df = _parse_csv_to_dataframe(raw_csv)
    df = _normalize_daily_frame(df, trade_date)
    if df is not None:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        provenance = _csv_comment_lines(raw_csv)
        provenance.append(f"# as-of: {trade_date}")
        provenance.append("# normalized: sorted, deduped, date<=as-of, OHLCV columns")
        results["stock_data"] = "\n".join(provenance) + "\n" + out.to_csv(index=False)
    else:
        results["stock_data"] = (
            f"【数据获取失败】{ticker} 在 {trade_date} 无有效完整日线数据"
            "（缺列/非法日期/全部行无效/重复冲突），本项不可用。"
        )
    daily_context = _build_daily_context(df, trade_date)
    realtime_context = results.pop("realtime", None)
    if not isinstance(realtime_context, dict) or realtime_context.get("status") not in {
        "available",
        "unavailable",
        "not_applicable",
    }:
        realtime_context = _unavailable_realtime_context(
            datetime.now(timezone.utc).isoformat(),
            "实时行情抓取未完成",
        )
    results["market_data_context"] = {
        "daily": daily_context,
        "realtime": realtime_context,
        "data_failure_ledger": data_failure_ledger,
    }

    # ── 核心加速：本地计算所有技术指标 ──────────────────
    indicators_res = {}
    try:
        if df is not None and "close" in df.columns:
            ss = wrap(df.copy())

            calc_map = {
                "close_50_sma": "close_50_sma",
                "close_200_sma": "close_200_sma",
                "close_10_ema": "close_10_ema",
                "rsi": "rsi_14",
                "macd": "macd",
                "boll": "close_20_sma",
                "boll_ub": "boll_ub",
                "boll_lb": "boll_lb",
                "atr": "atr",
                "vwma": "vwma"
            }

            for key, ss_key in calc_map.items():
                try:
                    val = ss[ss_key].iloc[-1]
                    indicators_res[key] = round(float(val), 2) if isinstance(val, (int, float)) else str(val)
                except Exception:
                    indicators_res[key] = "N/A"
        else:
            print(f"  [Warning] No valid stock_data for indicator calculation.")
    except Exception as e:
        print(f"  [Error] Local indicator calculation failed: {e}")

    for ind in INDICATORS:
        if ind not in indicators_res:
            indicators_res[ind] = "无数据"

    results["indicators"] = indicators_res

    # ── VPA 预计算指标 ──────────────────────────────
    try:
        if df is not None:
            results["vpa_indicators"] = _compute_vpa_indicators(df.copy())
        else:
            results["vpa_indicators"] = "VPA 数据不足"
    except Exception as e:
        results["vpa_indicators"] = f"VPA 计算失败：{e}"

    print(f"[Timer] Total Data Collection for {ticker} took {time.time() - fetch_start:.2f}s")
    return results


class DataCollector:
    """Collect and cache data, thread-safe and shareable across jobs."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()
        self._refcounts: Dict[str, int] = {}

    def _get_key_lock(self, key: str) -> threading.Lock:
        with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def collect(self, ticker: str, trade_date: str, horizons: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fetch all data and store in cache.

        Thread-safe: concurrent calls for the same ticker+date will block
        on a per-key lock, so data is fetched only once.
        """
        key = make_cache_key(ticker, trade_date)
        key_lock = self._get_key_lock(key)
        # 带超时的 acquire：即使持锁的抓取意外卡死，排队者也能在有限时间内
        # 报错退出，而不是把线程池 worker 一个个吸进来陪葬
        if not key_lock.acquire(timeout=FETCH_ALL_TIMEOUT + 60):
            raise TimeoutError(
                f"等待 {key} 数据抓取锁超时（>{FETCH_ALL_TIMEOUT + 60}s），"
                "可能存在卡死的抓取任务，本次分析中止"
            )
        try:
            if key not in self._cache:
                self._cache[key] = _fetch_all(ticker, trade_date)
            return copy.deepcopy(self._cache[key])
        finally:
            key_lock.release()

    def get(self, ticker: str, trade_date: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached pool, or None if not collected yet."""
        cached = self._cache.get(make_cache_key(ticker, trade_date))
        return None if cached is None else copy.deepcopy(cached)

    def get_window(
        self,
        pool: Dict[str, Any],
        horizon: str,
        trade_date: str,
    ) -> Dict[str, Any]:
        """Return pool copy annotated with horizon window metadata."""
        days = SHORT_DAYS if horizon == "short" else LONG_DAYS
        result = copy.deepcopy(pool)
        result["_data_window"] = f"{days}天"
        result["_horizon"] = horizon
        return result

    def ref(self, ticker: str, trade_date: str) -> None:
        """Increment reference count (call before using cached data)."""
        key = make_cache_key(ticker, trade_date)
        with self._meta_lock:
            self._refcounts[key] = self._refcounts.get(key, 0) + 1

    def evict(self, ticker: str, trade_date: str) -> None:
        """Decrement refcount and remove cached data when no one needs it."""
        key = make_cache_key(ticker, trade_date)
        with self._meta_lock:
            count = self._refcounts.get(key, 1) - 1
            if count <= 0:
                self._cache.pop(key, None)
                self._refcounts.pop(key, None)
                # 不删除 _locks[key]：其他线程可能仍持有该锁的引用，
                # 删除会导致新 collect() 创建新锁，破坏互斥。
                # 锁对象很轻量，留着不影响内存。
            else:
                self._refcounts[key] = count
