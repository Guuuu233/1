"""DataCollector: fetch all data once, serve windowed views to analyst agents."""
from __future__ import annotations

import logging
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
import re

from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_cn_indices,
    get_global_indices,
    get_major_assets,
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
from tradingagents.dataflows.fund_flow_evidence import (
    build_gap_meta,
    build_provider_text,
    summarize_evidence,
)
from tradingagents.dataflows.trade_calendar import (
    dedupe_daily_bars,
    is_historical_analysis_date,
)
from tradingagents.dataflows.providers.industry_linkage_provider import (
    IndustryLinkageProvider,
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

import numpy as np

logger = logging.getLogger(__name__)

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
            logger.debug("  [Timer] %s took %.2fs", getattr(tool, "name", str(tool)), duration)
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
        "analysis_baseline_date": None,
        "fund_flow_evidence": {
            "status": "unavailable",
            "unit": "亿元",
            "records": [],
            "summary": summarize_evidence([], window_days=5),
            "validation": {"status": "not_checked", "mismatches": []},
            "gap": "【数据获取失败】资金流 evidence：未返回结构化逐日记录",
        },
        "daily": {"as_of": None, "completeness": "unavailable"},
        "realtime": {
            "status": "unavailable",
            "source": None,
            "quote_as_of": None,
            "retrieved_at": None,
            "error": "实时行情上下文不可用",
            "quote": None,
        },
        "global_indices": None,
        "cn_indices": None,
        "major_assets": None,
        "industry_linkage": None,
        "source_provenance": {},
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
        if source not in {"sina", "eastmoney", "investoday", "fuyao"}:
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
    "cn_indices",
    "global_indices",
    "major_assets",
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
    "industry_linkage",
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


_SOURCE_AS_OF_PATTERNS = (
    r"最新数据日\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"最新发布时间\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"【数据日期】\s*(20\d{2}-\d{2}-\d{2})",
    r"实际数据日期\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"数据基准日\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"数据日期[】：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"日期\s*[：:]\s*(20\d{2}-\d{2}-\d{2})",
    r"龙虎榜明细[（(]\s*(20\d{2}-\d{2}-\d{2})",
    r"涨停池[（(]\s*(20\d{2}-\d{2}-\d{2})",
    r"(20\d{2}-\d{2}-\d{2})\s+涨停家数",
    r"\[(20\d{2}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\]",
)


def _extract_source_as_of(value: Any, requested_as_of: str) -> Optional[str]:
    """Extract the latest explicitly reported source date, excluding request windows."""
    if isinstance(value, dict):
        for key in ("as_of", "actual_as_of", "quote_as_of", "data_as_of"):
            candidate = value.get(key)
            match = re.search(r"20\d{2}-\d{2}-\d{2}", str(candidate or ""))
            if match and match.group(0) <= requested_as_of:
                return match.group(0)
        cached_at = value.get("cached_at")
        if isinstance(cached_at, (int, float)):
            try:
                dt_str = datetime.fromtimestamp(cached_at).strftime("%Y-%m-%d")
                if dt_str <= requested_as_of:
                    return dt_str
            except Exception:
                pass
        elif isinstance(cached_at, str):
            match = re.search(r"20\d{2}-\d{2}-\d{2}", cached_at)
            if match and match.group(0) <= requested_as_of:
                return match.group(0)
        return None

    text = value if isinstance(value, str) else str(value or "")
    candidates: list[str] = []
    for pattern in _SOURCE_AS_OF_PATTERNS:
        candidates.extend(match.group(1) for match in re.finditer(pattern, text))
    dates = [item for item in candidates if item <= requested_as_of]
    return max(dates) if dates else None


def _determine_industry_linkage_status(value: Any) -> Tuple[str, Optional[str]]:
    """Determine status (available, partial, unavailable) and failure reason for industry_linkage."""
    if value is None:
        return "unavailable", "未映射行业或数据源不可用"
    if isinstance(value, str):
        return "unavailable", value or "接口调用失败"
    if not isinstance(value, dict):
        return "unavailable", "返回结构异常"

    if "status" in value and value["status"] in ("available", "partial", "unavailable", "failed"):
        st = "unavailable" if value["status"] == "failed" else value["status"]
        return st, value.get("gap") or value.get("reason")

    all_inds: list[dict] = []
    for key in ("upstream_cost", "downstream_demand", "international_benchmark"):
        inds = value.get(key)
        if isinstance(inds, list):
            for ind in inds:
                if isinstance(ind, dict):
                    all_inds.append(ind)
                elif hasattr(ind, "model_dump"):
                    all_inds.append(ind.model_dump())

    if not all_inds:
        if value.get("industry_name"):
            return "partial", None
        return "unavailable", "未包含有效产业链指标数据"

    valid_count = 0
    for ind in all_inds:
        if ind.get("current_value") is not None:
            valid_count += 1
        elif ind.get("status") == "active" and ind.get("trend") not in ("数据缺失", None, ""):
            valid_count += 1

    if valid_count == len(all_inds):
        return "available", None
    elif valid_count > 0:
        return "partial", None
    else:
        if value.get("industry_name"):
            return "partial", None
        return "unavailable", "所有指标均无有效数据"


def _extract_industry_linkage_actual_as_of(value: Any, requested_as_of: str) -> Optional[str]:
    """Recursively collect valid actual_as_of from underlying indicators within industry_linkage.

    Never use top-level requested as_of or cached_at as the real data date.
    Only dates <= requested_as_of from indicators with valid data or explicit actual_as_of are considered.
    """
    if not isinstance(value, dict):
        return None

    actual_dates: list[str] = []

    def _collect_dates(obj: Any):
        if isinstance(obj, dict):
            val = obj.get("current_value")
            trend = obj.get("trend")
            has_valid_data = val is not None or (
                obj.get("status") == "active" and trend not in ("数据缺失", None, "")
            )

            act_date = obj.get("actual_as_of")
            if act_date is not None:
                match = re.search(r"20\d{2}-\d{2}-\d{2}", str(act_date))
                if match and match.group(0) <= requested_as_of:
                    actual_dates.append(match.group(0))
            elif has_valid_data:
                for k in ("data_as_of", "quote_as_of", "date", "trade_date", "as_of"):
                    cand = obj.get(k)
                    if cand is not None:
                        match = re.search(r"20\d{2}-\d{2}-\d{2}", str(cand))
                        if match and match.group(0) <= requested_as_of:
                            actual_dates.append(match.group(0))

                note = obj.get("note")
                if note and isinstance(note, str):
                    for m in re.finditer(r"20\d{2}-\d{2}-\d{2}", note):
                        if m.group(0) <= requested_as_of:
                            actual_dates.append(m.group(0))

            for k, v in obj.items():
                if k not in ("as_of", "cached_at", "requested_as_of") and isinstance(v, (dict, list)):
                    _collect_dates(v)
        elif isinstance(obj, list):
            for item in obj:
                _collect_dates(item)

    for section in ("upstream_cost", "downstream_demand", "international_benchmark"):
        section_items = value.get(section)
        if section_items:
            _collect_dates(section_items)

    for k, v in value.items():
        if (
            k not in ("as_of", "cached_at", "requested_as_of", "upstream_cost", "downstream_demand", "international_benchmark")
            and isinstance(v, (dict, list))
        ):
            _collect_dates(v)

    valid_dates = [d for d in actual_dates if d <= requested_as_of]
    return max(valid_dates) if valid_dates else None


def _build_source_provenance(
    results: Dict[str, Any],
    requested_as_of: str,
    daily_as_of: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """Persist per-source cutoff evidence beside the compact failure ledger."""
    provenance: Dict[str, Dict[str, Any]] = {}
    for source, value in results.items():
        if source == "industry_linkage":
            status, reason = _determine_industry_linkage_status(value)
            actual_as_of = _extract_industry_linkage_actual_as_of(value, requested_as_of)
            entry: Dict[str, Any] = {
                "requested_as_of": requested_as_of,
                "actual_as_of": actual_as_of,
                "as_of": actual_as_of,
                "status": status,
            }
            if status == "unavailable":
                entry["gap"] = f"【数据获取失败】industry_linkage：{reason or '数据源不可用'}"
            elif actual_as_of is None:
                entry["gap"] = "【数据获取失败】industry_linkage：未返回可验证数据日期"
            provenance[str(source)] = entry
            continue

        status = _classify_failure_value(value) or "available"
        as_of = _extract_source_as_of(value, requested_as_of)
        if source == "stock_data":
            as_of = daily_as_of
        entry = {
            "requested_as_of": requested_as_of,
            "actual_as_of": as_of,
            "as_of": as_of,
            "status": status,
        }
        if source == "stock_data" and as_of and as_of < requested_as_of:
            entry["gap"] = (
                f"【数据获取失败】stock_data：实际最新数据日 {as_of} "
                f"早于请求日期 {requested_as_of}"
            )
        if status != "available":
            entry["gap"] = f"【数据获取失败】{source}：{_compact_failure_reason(status)}"
        elif as_of is None and source != "realtime":
            entry["gap"] = f"【数据获取失败】{source}：未返回可验证数据日期"
        provenance[str(source)] = entry
    return provenance


def _map_stock_to_industry(ticker: Optional[str]) -> Optional[str]:
    """根据股票代码映射到核心行业（覆盖全部 27 个权威行业，保持原有 5 行业与 6 只标的向后兼容）。

    Args:
        ticker: 股票代码，如 "000725.SZ", "688981.SH", "601857.SH", "600036.SH", "600519.SH", "600030.SH" 等

    Returns:
        匹配到的行业名称或别名，未配置或不识别则返回 None
    """
    if not ticker or not isinstance(ticker, str):
        return None

    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        return None

    # 完整映射表（覆盖全部 27 个权威行业，每个行业至少配置 1 只代表 A 股，原有映射完全兼容）
    industry_map: Dict[str, str] = {
        # 1. 消费电子与智能终端 ("消费电子")
        "000725.SZ": "消费电子",  # 京东方A
        "000725": "消费电子",
        "000100.SZ": "消费电子",  # TCL科技
        "000100": "消费电子",
        "002475.SZ": "消费电子",  # 立讯精密
        "002475": "消费电子",
        "002241.SZ": "消费电子",  # 歌尔股份
        "002241": "消费电子",
        "300433.SZ": "消费电子",  # 蓝思科技
        "300433": "消费电子",
        "688036.SH": "消费电子与智能终端",  # 传音控股
        "688036": "消费电子与智能终端",

        # 2. 新能源汽车与智能汽车 ("新能源车")
        "300750.SZ": "新能源车",  # 宁德时代
        "300750": "新能源车",
        "002594.SZ": "新能源车",  # 比亚迪
        "002594": "新能源车",
        "601633.SH": "新能源车",  # 长城汽车
        "601633": "新能源车",
        "002460.SZ": "新能源车",  # 赣锋锂业
        "002460": "新能源车",
        "002466.SZ": "新能源车",  # 天齐锂业
        "002466": "新能源车",
        "601238.SH": "新能源汽车与智能汽车",  # 广汽集团
        "601238": "新能源汽车与智能汽车",

        # 3. 半导体与集成电路 ("半导体")
        "688981.SH": "半导体",  # 中芯国际
        "688981": "半导体",
        "603501.SH": "半导体",  # 韦尔股份
        "603501": "半导体",
        "002049.SZ": "半导体",  # 紫光国微
        "002049": "半导体",
        "600584.SH": "半导体",  # 长电科技
        "600584": "半导体",
        "688012.SH": "半导体",  # 中微公司
        "688012": "半导体",
        "002371.SZ": "半导体",  # 北方华创
        "002371": "半导体",
        "600703.SH": "半导体",  # 三安光电
        "600703": "半导体",

        # 4. 石油石化与基础化工 ("石油化工")
        "601857.SH": "石油化工",  # 中国石油
        "601857": "石油化工",
        "600309.SH": "石油化工",  # 万华化学
        "600309": "石油化工",
        "600028.SH": "石油化工",  # 中国石化
        "600028": "石油化工",
        "600938.SH": "石油化工",  # 中国海油
        "600938": "石油化工",
        "000301.SZ": "石油化工",  # 东方盛虹
        "000301": "石油化工",
        "601233.SH": "石油化工",  # 桐昆股份
        "601233": "石油化工",
        "002493.SZ": "石油化工",  # 荣盛石化
        "002493": "石油化工",
        "000703.SZ": "石油化工",  # 恒逸石化
        "000703": "石油化工",
        "600426.SH": "石油石化与基础化工",  # 华鲁恒升
        "600426": "石油石化与基础化工",

        # 5. 金融地产 / 商业银行与房地产
        "600036.SH": "金融地产",  # 招商银行
        "600036": "金融地产",
        "000002.SZ": "金融地产",  # 万科A
        "000002": "金融地产",
        "601398.SH": "金融地产",  # 工商银行
        "601398": "金融地产",
        "601288.SH": "金融地产",  # 农业银行
        "601288": "金融地产",
        "601939.SH": "金融地产",  # 建设银行
        "601939": "金融地产",
        "600000.SH": "金融地产",  # 浦发银行
        "600000": "金融地产",
        "000001.SZ": "金融地产",  # 平安银行
        "000001": "金融地产",
        "600048.SH": "金融地产",  # 保利发展
        "600048": "金融地产",
        "001979.SZ": "金融地产",  # 招商蛇口
        "001979": "金融地产",

        # 6. 人工智能与算力服务
        "300308.SZ": "人工智能与算力服务",  # 中际旭创
        "300308": "人工智能与算力服务",
        "002230.SZ": "人工智能与算力服务",  # 科大讯飞
        "002230": "人工智能与算力服务",
        "601138.SH": "人工智能与算力服务",  # 工业富联
        "601138": "人工智能与算力服务",
        "000977.SZ": "人工智能与算力服务",  # 浪潮信息
        "000977": "人工智能与算力服务",
        "688256.SH": "人工智能与算力服务",  # 寒武纪
        "688256": "人工智能与算力服务",

        # 7. 光伏与储能系统
        "601012.SH": "光伏与储能系统",  # 隆基绿能
        "601012": "光伏与储能系统",
        "600438.SH": "光伏与储能系统",  # 通威股份
        "600438": "光伏与储能系统",
        "300274.SZ": "光伏与储能系统",  # 阳光电源
        "300274": "光伏与储能系统",
        "688599.SH": "光伏与储能系统",  # 天合光能
        "688599": "光伏与储能系统",

        # 8. 动力电池与储能电池材料
        "002709.SZ": "动力电池与储能电池材料",  # 天赐材料
        "002709": "动力电池与储能电池材料",
        "002812.SZ": "动力电池与储能电池材料",  # 恩捷股份
        "002812": "动力电池与储能电池材料",
        "603659.SH": "动力电池与储能电池材料",  # 璞泰来
        "603659": "动力电池与储能电池材料",
        "300014.SZ": "动力电池与储能电池材料",  # 亿纬锂能
        "300014": "动力电池与储能电池材料",

        # 9. 医药生物与创新药
        "600276.SH": "医药生物与创新药",  # 恒瑞医药
        "600276": "医药生物与创新药",
        "603259.SH": "医药生物与创新药",  # 药明康德
        "603259": "医药生物与创新药",
        "300122.SZ": "医药生物与创新药",  # 智飞生物
        "300122": "医药生物与创新药",
        "688180.SH": "医药生物与创新药",  # 君实生物
        "688180": "医药生物与创新药",
        "688235.SH": "医药生物与创新药",  # 百济神州
        "688235": "医药生物与创新药",
        "000538.SZ": "医药生物与创新药",  # 云南白药
        "000538": "医药生物与创新药",
        "600436.SH": "医药生物与创新药",  # 片仔癀
        "600436": "医药生物与创新药",

        # 10. 医疗器械与医疗服务
        "300760.SZ": "医疗器械与医疗服务",  # 迈瑞医疗
        "300760": "医疗器械与医疗服务",
        "688271.SH": "医疗器械与医疗服务",  # 联影医疗
        "688271": "医疗器械与医疗服务",
        "300015.SZ": "医疗器械与医疗服务",  # 爱尔眼科
        "300015": "医疗器械与医疗服务",
        "300003.SZ": "医疗器械与医疗服务",  # 乐普医疗
        "300003": "医疗器械与医疗服务",
        "688617.SH": "医疗器械与医疗服务",  # 惠泰医疗
        "688617": "医疗器械与医疗服务",

        # 11. 白酒与精制茶酒
        "600519.SH": "白酒与精制茶酒",  # 贵州茅台
        "600519": "白酒与精制茶酒",
        "000858.SZ": "白酒与精制茶酒",  # 五粮液
        "000858": "白酒与精制茶酒",
        "000568.SZ": "白酒与精制茶酒",  # 泸州老窖
        "000568": "白酒与精制茶酒",
        "600809.SH": "白酒与精制茶酒",  # 山西汾酒
        "600809": "白酒与精制茶酒",
        "002304.SZ": "白酒与精制茶酒",  # 洋河股份
        "002304": "白酒与精制茶酒",

        # 12. 大众食品与饮料
        "603288.SH": "大众食品与饮料",  # 海天味业
        "603288": "大众食品与饮料",
        "600887.SH": "大众食品与饮料",  # 伊利股份
        "600887": "大众食品与饮料",
        "600600.SH": "大众食品与饮料",  # 青岛啤酒
        "600600": "大众食品与饮料",
        "002557.SZ": "大众食品与饮料",  # 洽洽食品
        "002557": "大众食品与饮料",
        "603345.SH": "大众食品与饮料",  # 安井食品
        "603345": "大众食品与饮料",
        "605499.SH": "大众食品与饮料",  # 东鹏饮料
        "605499": "大众食品与饮料",

        # 13. 家用电器与智能家居
        "000333.SZ": "家用电器与智能家居",  # 美的集团
        "000333": "家用电器与智能家居",
        "000651.SZ": "家用电器与智能家居",  # 格力电器
        "000651": "家用电器与智能家居",
        "600690.SH": "家用电器与智能家居",  # 海尔智家
        "600690": "家用电器与智能家居",
        "688169.SH": "家用电器与智能家居",  # 石头科技
        "688169": "家用电器与智能家居",
        "002032.SZ": "家用电器与智能家居",  # 苏泊尔
        "002032": "家用电器与智能家居",

        # 14. 商业银行与信贷
        "601166.SH": "商业银行与信贷",  # 兴业银行
        "601166": "商业银行与信贷",
        "600919.SH": "商业银行与信贷",  # 江苏银行
        "600919": "商业银行与信贷",
        "601009.SH": "商业银行与信贷",  # 南京银行
        "601009": "商业银行与信贷",
        "600926.SH": "商业银行与信贷",  # 杭州银行
        "600926": "商业银行与信贷",

        # 15. 证券公司与资本市场
        "600030.SH": "证券公司与资本市场",  # 中信证券
        "600030": "证券公司与资本市场",
        "601688.SH": "证券公司与资本市场",  # 华泰证券
        "601688": "证券公司与资本市场",
        "600958.SH": "证券公司与资本市场",  # 东方证券
        "600958": "证券公司与资本市场",
        "600999.SH": "证券公司与资本市场",  # 招商证券
        "600999": "证券公司与资本市场",
        "300059.SZ": "证券公司与资本市场",  # 东方财富
        "300059": "证券公司与资本市场",
        "601211.SH": "证券公司与资本市场",  # 国泰君安
        "601211": "证券公司与资本市场",

        # 16. 保险与多元金融
        "601318.SH": "保险与多元金融",  # 中国平安
        "601318": "保险与多元金融",
        "601628.SH": "保险与多元金融",  # 中国人寿
        "601628": "保险与多元金融",
        "601601.SH": "保险与多元金融",  # 中国太保
        "601601": "保险与多元金融",
        "601336.SH": "保险与多元金融",  # 新华保险
        "601336": "保险与多元金融",
        "601319.SH": "保险与多元金融",  # 中国人保
        "601319": "保险与多元金融",

        # 17. 钢铁与黑色金属
        "600019.SH": "钢铁与黑色金属",  # 宝钢股份
        "600019": "钢铁与黑色金属",
        "000932.SZ": "钢铁与黑色金属",  # 华菱钢铁
        "000932": "钢铁与黑色金属",
        "600782.SH": "钢铁与黑色金属",  # 新钢股份
        "600782": "钢铁与黑色金属",
        "600282.SH": "钢铁与黑色金属",  # 南钢股份
        "600282": "钢铁与黑色金属",
        "000825.SZ": "钢铁与黑色金属",  # 太钢不锈
        "000825": "钢铁与黑色金属",

        # 18. 有色金属与工业金属
        "601899.SH": "有色金属与工业金属",  # 紫金矿业
        "601899": "有色金属与工业金属",
        "601600.SH": "有色金属与工业金属",  # 中国铝业
        "601600": "有色金属与工业金属",
        "600362.SH": "有色金属与工业金属",  # 江西铜业
        "600362": "有色金属与工业金属",
        "600219.SH": "有色金属与工业金属",  # 南山铝业
        "600219": "有色金属与工业金属",
        "603993.SH": "有色金属与工业金属",  # 洛阳钼业
        "603993": "有色金属与工业金属",
        "000807.SZ": "有色金属与工业金属",  # 云铝股份
        "000807": "有色金属与工业金属",

        # 19. 贵金属与稀缺资源
        "600547.SH": "贵金属与稀缺资源",  # 山东黄金
        "600547": "贵金属与稀缺资源",
        "601069.SH": "贵金属与稀缺资源",  # 西部黄金
        "601069": "贵金属与稀缺资源",
        "600489.SH": "贵金属与稀缺资源",  # 中金黄金
        "600489": "贵金属与稀缺资源",
        "002155.SZ": "贵金属与稀缺资源",  # 湖南黄金
        "002155": "贵金属与稀缺资源",
        "600988.SH": "贵金属与稀缺资源",  # 赤峰黄金
        "600988": "贵金属与稀缺资源",
        "600111.SH": "贵金属与稀缺资源",  # 北方稀土
        "600111": "贵金属与稀缺资源",

        # 20. 煤炭与传统化石能源
        "601088.SH": "煤炭与传统化石能源",  # 中国神华
        "601088": "煤炭与传统化石能源",
        "601225.SH": "煤炭与传统化石能源",  # 陕西煤业
        "601225": "煤炭与传统化石能源",
        "600188.SH": "煤炭与传统化石能源",  # 兖矿能源
        "600188": "煤炭与传统化石能源",
        "600985.SH": "煤炭与传统化石能源",  # 淮北矿业
        "600985": "煤炭与传统化石能源",
        "000983.SZ": "煤炭与传统化石能源",  # 山西焦煤
        "000983": "煤炭与传统化石能源",
        "601699.SH": "煤炭与传统化石能源",  # 潞安环能
        "601699": "煤炭与传统化石能源",

        # 21. 电力与公用事业
        "600900.SH": "电力与公用事业",  # 长江电力
        "600900": "电力与公用事业",
        "601985.SH": "电力与公用事业",  # 中国核电
        "601985": "电力与公用事业",
        "600011.SH": "电力与公用事业",  # 华能国际
        "600011": "电力与公用事业",
        "600027.SH": "电力与公用事业",  # 华电国际
        "600027": "电力与公用事业",
        "600025.SH": "电力与公用事业",  # 华能水电
        "600025": "电力与公用事业",
        "003816.SZ": "电力与公用事业",  # 中国广核
        "003816": "电力与公用事业",
        "600905.SH": "电力与公用事业",  # 三峡能源
        "600905": "电力与公用事业",

        # 22. 房地产开发与运营
        "600383.SH": "房地产开发与运营",  # 金地集团
        "600383": "房地产开发与运营",
        "600266.SH": "房地产开发与运营",  # 北京城建
        "600266": "房地产开发与运营",
        "600325.SH": "房地产开发与运营",  # 华发股份
        "600325": "房地产开发与运营",
        "000656.SZ": "房地产开发与运营",  # 金科股份
        "000656": "房地产开发与运营",
        "000069.SZ": "房地产开发与运营",  # 华侨城A
        "000069": "房地产开发与运营",

        # 23. 建筑装饰与基础设施工程
        "601668.SH": "建筑装饰与基础设施工程",  # 中国建筑
        "601668": "建筑装饰与基础设施工程",
        "601390.SH": "建筑装饰与基础设施工程",  # 中国中铁
        "601390": "建筑装饰与基础设施工程",
        "601186.SH": "建筑装饰与基础设施工程",  # 中国铁建
        "601186": "建筑装饰与基础设施工程",
        "601800.SH": "建筑装饰与基础设施工程",  # 中国交建
        "601800": "建筑装饰与基础设施工程",
        "601669.SH": "建筑装饰与基础设施工程",  # 中国电建
        "601669": "建筑装饰与基础设施工程",
        "600585.SH": "建筑装饰与基础设施工程",  # 海螺水泥
        "600585": "建筑装饰与基础设施工程",
        "002271.SZ": "建筑装饰与基础设施工程",  # 东方雨虹
        "002271": "建筑装饰与基础设施工程",

        # 24. 机械设备与工业母机
        "600031.SH": "机械设备与工业母机",  # 三一重工
        "600031": "机械设备与工业母机",
        "000425.SZ": "机械设备与工业母机",  # 徐工机械
        "000425": "机械设备与工业母机",
        "000157.SZ": "机械设备与工业母机",  # 中联重科
        "000157": "机械设备与工业母机",
        "688305.SH": "机械设备与工业母机",  # 科德数控
        "688305": "机械设备与工业母机",
        "300161.SZ": "机械设备与工业母机",  # 华中数控
        "300161": "机械设备与工业母机",
        "601100.SH": "机械设备与工业母机",  # 恒立液压
        "601100": "机械设备与工业母机",
        "300124.SZ": "机械设备与工业母机",  # 汇川技术
        "300124": "机械设备与工业母机",

        # 25. 国防军工与航天装备
        "600893.SH": "国防军工与航天装备",  # 航发动力
        "600893": "国防军工与航天装备",
        "600760.SH": "国防军工与航天装备",  # 中航沈飞
        "600760": "国防军工与航天装备",
        "000768.SZ": "国防军工与航天装备",  # 中航西飞
        "000768": "国防军工与航天装备",
        "600316.SH": "国防军工与航天装备",  # 洪都航空
        "600316": "国防军工与航天装备",
        "002179.SZ": "国防军工与航天装备",  # 中航光电
        "002179": "国防军工与航天装备",
        "600150.SH": "国防军工与航天装备",  # 中国船舶
        "600150": "国防军工与航天装备",
        "600118.SH": "国防军工与航天装备",  # 中国卫星
        "600118": "国防军工与航天装备",

        # 26. 交通运输与航运港口
        "601919.SH": "交通运输与航运港口",  # 中远海控
        "601919": "交通运输与航运港口",
        "600026.SH": "交通运输与航运港口",  # 中远海能
        "600026": "交通运输与航运港口",
        "601872.SH": "交通运输与航运港口",  # 招商轮船
        "601872": "交通运输与航运港口",
        "002352.SZ": "交通运输与航运港口",  # 顺丰控股
        "002352": "交通运输与航运港口",
        "600018.SH": "交通运输与航运港口",  # 上港集团
        "600018": "交通运输与航运港口",
        "601006.SH": "交通运输与航运港口",  # 大秦铁路
        "601006": "交通运输与航运港口",
        "601111.SH": "交通运输与航运港口",  # 中国国航
        "601111": "交通运输与航运港口",
        "600009.SH": "交通运输与航运港口",  # 上海机场
        "600009": "交通运输与航运港口",

        # 27. 通信网络与光通信
        "600941.SH": "通信网络与光通信",  # 中国移动
        "600941": "通信网络与光通信",
        "601728.SH": "通信网络与光通信",  # 中国电信
        "601728": "通信网络与光通信",
        "600050.SH": "通信网络与光通信",  # 中国联通
        "600050": "通信网络与光通信",
        "000063.SZ": "通信网络与光通信",  # 中兴通讯
        "000063": "通信网络与光通信",
        "300502.SZ": "通信网络与光通信",  # 新易盛
        "300502": "通信网络与光通信",
        "300394.SZ": "通信网络与光通信",  # 天孚通信
        "300394": "通信网络与光通信",
        "600487.SH": "通信网络与光通信",  # 亨通光电
        "600487": "通信网络与光通信",
        "600522.SH": "通信网络与光通信",  # 中天科技
        "600522": "通信网络与光通信",

        # 28. 农林牧渔与生猪养殖
        "002714.SZ": "农林牧渔与生猪养殖",  # 牧原股份
        "002714": "农林牧渔与生猪养殖",
        "300498.SZ": "农林牧渔与生猪养殖",  # 温氏股份
        "300498": "农林牧渔与生猪养殖",
        "000876.SZ": "农林牧渔与生猪养殖",  # 新希望
        "000876": "农林牧渔与生猪养殖",
        "002385.SZ": "农林牧渔与生猪养殖",  # 大北农
        "002385": "农林牧渔与生猪养殖",
        "600313.SH": "农林牧渔与生猪养殖",  # 农发种业
        "600313": "农林牧渔与生猪养殖",
        "002041.SZ": "农林牧渔与生猪养殖",  # 登海种业
        "002041": "农林牧渔与生猪养殖",
        "002299.SZ": "农林牧渔与生猪养殖",  # 圣农发展
        "002299": "农林牧渔与生猪养殖",
    }

    return industry_map.get(clean_ticker)


_DEFAULT_INDUSTRY_LINKAGE_PROVIDER = IndustryLinkageProvider()


def _fetch_all(
    ticker: str,
    trade_date: str,
    industry_provider: Optional[IndustryLinkageProvider] = None,
) -> Dict[str, Any]:
    """Fetch all data sources in parallel.

    Always fetches full data including financial statements, regardless of horizon.
    The horizon only affects the analysis window, not data collection.
    """
    lookback = LONG_DAYS
    if "-" in trade_date:
        end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
        norm_trade_date = trade_date
    elif len(trade_date) == 8 and trade_date.isdigit():
        end_dt = datetime.strptime(trade_date, "%Y%m%d")
        norm_trade_date = end_dt.strftime("%Y-%m-%d")
    else:
        dt = pd.to_datetime(trade_date, errors="coerce")
        if pd.isna(dt):
            end_dt = datetime.now()
            norm_trade_date = end_dt.strftime("%Y-%m-%d")
        else:
            end_dt = dt.to_pydatetime()
            norm_trade_date = end_dt.strftime("%Y-%m-%d")

    # 为了计算指标准确（如 200 SMA），需要比分析窗口更长的历史数据
    fetch_lookback = 365
    start_str = (end_dt - timedelta(days=fetch_lookback)).strftime("%Y-%m-%d")

    tasks: Dict[str, tuple] = {
        "stock_data": (get_stock_data, {"symbol": ticker, "start_date": start_str, "end_date": norm_trade_date}),
        "cn_indices": (get_cn_indices, {"curr_date": norm_trade_date, "look_back_days": lookback}),
        "global_indices": (get_global_indices, {"curr_date": norm_trade_date, "look_back_days": lookback}),
        "major_assets": (get_major_assets, {"curr_date": norm_trade_date, "look_back_days": lookback}),
        "realtime": (_fetch_realtime_context, {"ticker": ticker, "trade_date": norm_trade_date}),
        "news": (get_news, {"ticker": ticker, "start_date": (end_dt - timedelta(days=lookback)).strftime("%Y-%m-%d"), "end_date": norm_trade_date}),
        "global_news": (get_global_news, {"curr_date": norm_trade_date, "look_back_days": lookback, "limit": 30}),
        "fund_flow_board": (get_board_fund_flow, {"curr_date": norm_trade_date}),
        "fund_flow_individual": (get_individual_fund_flow, {"symbol": ticker, "curr_date": norm_trade_date}),
        "lhb": (get_lhb_detail, {"symbol": ticker, "date": norm_trade_date}),
        "insider_transactions": (get_insider_transactions, {"ticker": ticker, "curr_date": norm_trade_date}),
        "zt_pool": (get_zt_pool, {"date": norm_trade_date}),
        "hot_stocks": (get_hot_stocks_xq, {"curr_date": norm_trade_date}),
        "restricted_release": (get_restricted_release, {"symbol": ticker, "curr_date": norm_trade_date}),
        "share_pledge": (get_share_pledge, {"symbol": ticker, "curr_date": norm_trade_date}),
        "earnings_forecast": (get_earnings_forecast, {"symbol": ticker, "curr_date": norm_trade_date}),
        "shareholder_count": (get_shareholder_count, {"symbol": ticker, "curr_date": norm_trade_date}),
        "margin_trading": (get_margin_trading, {"symbol": ticker, "curr_date": norm_trade_date}),
        "northbound_flow": (get_northbound_flow, {"symbol": ticker, "curr_date": norm_trade_date}),
    }

    # 财务报表类数据始终拉取，Research Manager 根据 horizon 自行判断权重
    tasks.update({
        "fundamentals": (get_fundamentals, {"ticker": ticker, "curr_date": norm_trade_date}),
        "balance_sheet": (get_balance_sheet, {"ticker": ticker, "freq": "quarterly", "curr_date": norm_trade_date}),
        "cashflow": (get_cashflow, {"ticker": ticker, "freq": "quarterly", "curr_date": norm_trade_date}),
        "income_statement": (get_income_statement, {"ticker": ticker, "freq": "quarterly", "curr_date": norm_trade_date}),
    })

    results: Dict[str, Any] = {}
    fetch_start = time.time()
    # 减少并发池大小，避免被反爬
    executor = ThreadPoolExecutor(max_workers=min(FETCH_MAX_WORKERS, len(tasks)))
    try:
        future_to_key = {executor.submit(_safe, tool, payload): key for key, (tool, payload) in tasks.items()}
        done, not_done = futures_wait(set(future_to_key), timeout=FETCH_ALL_TIMEOUT)
        for future in done:
            results[future_to_key[future]] = future.result()
        for future in not_done:
            key = future_to_key[future]
            results[key] = f"{key} 数据拉取超时（>{FETCH_ALL_TIMEOUT}s），本次分析跳过该数据源"
            logger.warning("  [Warning] %s fetch timed out after %ss, skipped", key, FETCH_ALL_TIMEOUT)
    finally:
        # Provider routing has bounded timeouts, so completing the executor here
        # is finite and keeps stuck worker/socket threads from outliving the job.
        executor.shutdown(wait=True, cancel_futures=True)

    # ── 产业链数据层采集 (MVP: 消费电子 / 新能源车 / 27 行业) ─────────────
    industry = _map_stock_to_industry(ticker)
    if industry:
        provider = industry_provider or _DEFAULT_INDUSTRY_LINKAGE_PROVIDER
        try:
            results["industry_linkage"] = provider.get_industry_linkage(
                industry, as_of=norm_trade_date
            )
        except Exception as exc:
            logger.warning("  [Warning] 产业链数据采集异常 (%s, %s): %s", ticker, industry, exc)
            results["industry_linkage"] = None
    else:
        results["industry_linkage"] = None

    data_failure_ledger = _build_data_failure_ledger(results)

    fund_flow_value = results.get("fund_flow_individual")
    fund_flow_evidence = getattr(fund_flow_value, "fund_flow_evidence", None)
    fund_flow_evidence_meta = getattr(fund_flow_value, "fund_flow_evidence_meta", None)
    if isinstance(fund_flow_evidence, list) and fund_flow_evidence:
        selected_field = (
            fund_flow_evidence_meta.get("selected_field")
            or fund_flow_evidence_meta.get("field")
            if isinstance(fund_flow_evidence_meta, dict)
            else None
        )
        selected_source = (
            fund_flow_evidence_meta.get("selected_source")
            if isinstance(fund_flow_evidence_meta, dict)
            else None
        )
        summary = summarize_evidence(
            fund_flow_evidence,
            window_days=5,
            field=selected_field,
            source=selected_source,
            requested_as_of=trade_date,
        )
        fund_flow_context = {
            **dict(fund_flow_evidence_meta or {}),
            "records": copy.deepcopy(fund_flow_evidence),
            "summary": summary,
            "validation": {"status": "not_checked", "mismatches": []},
        }
        if summary.get("status") == "available":
            fund_flow_context["status"] = "available"
        elif fund_flow_evidence_meta and fund_flow_evidence_meta.get("status") in ("selected", "consensus"):
            fund_flow_context["status"] = fund_flow_evidence_meta["status"]
        else:
            fund_flow_context["status"] = "partial"
    else:
        generic_gap = build_gap_meta(
            symbol=ticker,
            requested_as_of=trade_date,
            source="fund_flow_individual",
            status="unavailable",
            reason="未返回结构化逐日 netamount/r0_net evidence",
        )
        fund_flow_context = (
            copy.deepcopy(dict(fund_flow_evidence_meta))
            if isinstance(fund_flow_evidence_meta, dict)
            else {}
        )
        for key, value in generic_gap.items():
            fund_flow_context.setdefault(key, value)
        fund_flow_context.setdefault("records", [])
        fund_flow_context.setdefault(
            "summary", summarize_evidence([], window_days=5)
        )
        fund_flow_context.setdefault(
            "validation", {"status": "not_checked", "mismatches": []}
        )
        if not any(entry.get("source") == "fund_flow_individual" for entry in data_failure_ledger):
            data_failure_ledger.append({
                "source": "fund_flow_individual",
                "status": "unavailable",
                "reason": "structured evidence unavailable",
                "gap": fund_flow_context["gap"],
            })

    # ── Parse CSV once, reuse for indicators and VPA ──────────────────
    raw_csv = results.get("stock_data", "")
    df = _parse_csv_to_dataframe(raw_csv)
    df = _normalize_daily_frame(df, trade_date)
    if df is not None:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        provenance = _csv_comment_lines(raw_csv)
        actual_daily_as_of = pd.to_datetime(out["date"], errors="coerce").max().strftime("%Y-%m-%d")
        provenance.append(f"# requested-as-of: {trade_date}")
        provenance.append(f"# as-of: {actual_daily_as_of}")
        provenance.append("# normalized: sorted, deduped, date<=as-of, OHLCV columns")
        results["stock_data"] = "\n".join(provenance) + "\n" + out.to_csv(index=False)
    else:
        results["stock_data"] = (
            f"【数据获取失败】{ticker} 在 {trade_date} 无有效完整日线数据"
            "（缺列/非法日期/全部行无效/重复冲突），本项不可用。"
        )
        if not any(
            isinstance(entry, dict) and entry.get("source") == "stock_data"
            for entry in data_failure_ledger
        ):
            data_failure_ledger.append(
                {
                    "source": "stock_data",
                    "status": "unavailable",
                    "reason": "no valid completed daily bars",
                    "gap": (
                        f"【数据获取失败】stock_data：{ticker} 在 {trade_date} "
                        "无有效完整日线数据"
                    ),
                }
            )
    daily_context = _build_daily_context(df, trade_date)
    source_provenance = _build_source_provenance(
        results,
        trade_date,
        daily_context.get("as_of"),
    )
    ledger_sources = {
        str(entry.get("source"))
        for entry in data_failure_ledger
        if isinstance(entry, dict)
    }
    for source, provenance in source_provenance.items():
        if not isinstance(provenance, dict):
            continue
        gap = provenance.get("gap")
        if gap and source not in ledger_sources:
            data_failure_ledger.append(
                {
                    "source": str(source),
                    "status": "unavailable",
                    "reason": "unverified as-of",
                    "gap": str(gap),
                }
            )
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
        "analysis_baseline_date": trade_date,
        "fund_flow_evidence": fund_flow_context,
        "daily": daily_context,
        "realtime": realtime_context,
        "global_indices": results.get("global_indices"),
        "cn_indices": results.get("cn_indices"),
        "major_assets": results.get("major_assets"),
        "industry_linkage": results.get("industry_linkage"),
        "source_provenance": source_provenance,
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
            logger.warning("  [Warning] No valid stock_data for indicator calculation.")
    except Exception as e:
        logger.error("  [Error] Local indicator calculation failed: %s", e)

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

    logger.debug("[Timer] Total Data Collection for %s took %.2fs", ticker, time.time() - fetch_start)
    return results


class DataCollector:
    """Collect and cache data, thread-safe and shareable across jobs."""

    def __init__(self, industry_linkage_provider: Optional[IndustryLinkageProvider] = None):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()
        self._refcounts: Dict[str, int] = {}
        self.industry_linkage_provider: IndustryLinkageProvider = (
            industry_linkage_provider or IndustryLinkageProvider()
        )

    _map_stock_to_industry = staticmethod(_map_stock_to_industry)

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
                self._cache[key] = _fetch_all(
                    ticker,
                    trade_date,
                    industry_provider=self.industry_linkage_provider,
                )
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
