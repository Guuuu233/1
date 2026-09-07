"""DataCollector: fetch all data once, serve windowed views to analyst agents."""
from __future__ import annotations

import logging
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    wait as futures_wait,
)
import copy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import math
from typing import Any, Dict, List, Optional, Mapping
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
from tradingagents.dataflows.interface import _registry, route_to_vendor
from tradingagents.dataflows.providers.cn_akshare_provider import (
    PRICE_BASIS_PIT_ADJUSTED,
    PRICE_BASIS_PIT_RAW,
    PRICE_BASIS_RAW,
    PRICE_BASIS_UNSPECIFIED,
    PRICE_BASIS_VENDOR_QFQ,
    PriceBasisError,
    UnknownPriceBasisError,
    UnsupportedPriceBasisError,
    RawDailyFetchError,
    StockDataText,
    validate_price_basis_short_label,
)
from tradingagents.dataflows.cninfo_disclosure import (
    CninfoDisclosureEnvelope,
    CONTENT_STATUS_UNAVAILABLE,
    STATUS_OK,
    STATUS_PROVIDER_FAILURE,
    SOURCE_TYPE_ANNOUNCEMENT,
    SOURCE_TYPE_IR_SURVEY,
    qualify_cninfo_content,
)
from tradingagents.dataflows.fund_flow_evidence import (
    build_gap_meta,
    build_provider_text,
    calculate_fund_flow_scale_metrics,
    summarize_evidence,
)
from tradingagents.dataflows.news_event_evidence import (
    NewsEvidence,
    build_news_event_coverage,
    cninfo_record_to_evidence,
    parse_news_markdown_to_evidences,
)
from tradingagents.dataflows.trade_calendar import (
    dedupe_daily_bars,
    is_historical_analysis_date,
)
from tradingagents.dataflows.providers.industry_linkage_provider import (
    IndustryLinkageProvider,
)
from tradingagents.dataflows.social.collector import (
    SocialDataCollector,
    build_social_failure_ledger,
)
from tradingagents.dataflows.social.contracts import (
    REASON_SOCIAL_ARCHIVE_LOCKED,
    REASON_SOCIAL_ARCHIVE_MISSING,
    REASON_SOCIAL_NOT_APPLICABLE,
    SocialDataContext,
    SocialStatus,
    create_default_social_data_context,
    create_empty_sentiment_bundle,
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

# Volume Regimes (D-009 P1-2)
VOLUME_REGIME_NORMAL = "normal"
VOLUME_REGIME_HIGH_VOLUME_STAGNATION = "high_volume_stagnation_candidate"
VOLUME_REGIME_CAPITULATION = "capitulation_candidate"
VOLUME_REGIME_INSUFFICIENT_DATA = "insufficient_data"

VOLUME_REGIMES = frozenset({
    VOLUME_REGIME_NORMAL,
    VOLUME_REGIME_HIGH_VOLUME_STAGNATION,
    VOLUME_REGIME_CAPITULATION,
    VOLUME_REGIME_INSUFFICIENT_DATA,
})

# Reversal States (D-009 P1-2)
REVERSAL_STATE_UNCONFIRMED = "reversal_unconfirmed"
REVERSAL_STATE_CONFIRMED = "reversal_confirmed"
REVERSAL_STATE_INSUFFICIENT_DATA = "insufficient_data"
REVERSAL_STATE_NONE = "none"

REVERSAL_STATES = frozenset({
    REVERSAL_STATE_UNCONFIRMED,
    REVERSAL_STATE_CONFIRMED,
    REVERSAL_STATE_INSUFFICIENT_DATA,
    REVERSAL_STATE_NONE,
})

VPA_DEFAULT_WINDOW = 20
VPA_MIN_REQUIRED_BARS = 25  # window + 5
VPA_CAPITULATION_ZSCORE_THRESHOLD = 2.0
VPA_CAPITULATION_VOLUME_RATIO_THRESHOLD = 2.0
VPA_CAPITULATION_DROP_PCT_THRESHOLD = -0.03
VPA_CAPITULATION_CONSECUTIVE_DOWN_DAYS = 3
VPA_STAGNATION_ZSCORE_THRESHOLD = 1.5
VPA_STAGNATION_VOLUME_RATIO_THRESHOLD = 1.8
VPA_STAGNATION_MAX_ABS_PCT_CHANGE = 0.015
VPA_STAGNATION_MAX_SPREAD = 0.02
MAX_REVERSAL_STAGED_POSITION_PCT = 10.0


def compute_vpa_deterministic_features(
    df: Optional[pd.DataFrame],
    cutoff: Optional[str] = None,
    window: int = VPA_DEFAULT_WINDOW,
) -> Dict[str, Any]:
    """Compute deterministic VPA regime and reversal candidate features (D-009 P1-2).

    Strictly no forward-looking / T+1 leakage:
    Bars are truncated strictly to date <= cutoff before evaluating patterns and follow-through.
    """
    cutoff_str = str(cutoff).strip() if cutoff is not None and str(cutoff).strip() else None

    insufficient_result = {
        "volume_regime": VOLUME_REGIME_INSUFFICIENT_DATA,
        "reversal_state": REVERSAL_STATE_INSUFFICIENT_DATA,
        "features": {},
        "as_of": cutoff_str or "",
        "cutoff": cutoff_str or "",
    }

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return insufficient_result

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        return insufficient_result

    df = df.copy()

    # If date column is available, sort and truncate strictly <= cutoff
    if "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date_dt"]).sort_values("date_dt")
        if cutoff_str:
            cutoff_dt = pd.to_datetime(cutoff_str, errors="coerce")
            if pd.notna(cutoff_dt):
                df = df[df["date_dt"] <= cutoff_dt]
        df = df.drop(columns=["date_dt"])

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    if len(df) < window + 5:
        return insufficient_result

    # ── Derive Rolling & Bar Features ──
    vol_mean = df["volume"].rolling(window).mean()
    vol_std = df["volume"].rolling(window).std(ddof=0)
    df["volume_zscore"] = np.where(vol_std > 0, (df["volume"] - vol_mean) / vol_std, 0.0)
    df["volume_ratio"] = np.where(vol_mean > 0, df["volume"] / vol_mean, 1.0)

    hl_range = df["high"] - df["low"]
    df["bar_spread"] = np.where(df["close"] > 0, hl_range / df["close"], 0.0)
    df["close_position"] = np.where(
        hl_range > 0,
        (df["close"] - df["low"]) / hl_range,
        0.5,
    )
    df["pct_change"] = df["close"].pct_change().fillna(0.0)

    # Consecutive down days up to the bar
    down_series = (df["pct_change"] < 0) | (df["close"] < df["open"])
    consec_down = []
    curr_down = 0
    for is_down in down_series:
        if is_down:
            curr_down += 1
        else:
            curr_down = 0
        consec_down.append(curr_down)
    df["consecutive_down_days"] = consec_down

    last_bar = df.iloc[-1]
    as_of = str(last_bar.get("date", cutoff_str or ""))

    # ── Regime Detection (look at recent bars up to cutoff, e.g. last 5 bars) ──
    recent_len = min(5, len(df))
    recent_bars = df.tail(recent_len).reset_index(drop=True)

    capitulation_idx = None
    for i in range(len(recent_bars) - 1, -1, -1):
        row = recent_bars.iloc[i]
        vol_extreme = (
            row["volume_zscore"] >= VPA_CAPITULATION_ZSCORE_THRESHOLD
            or row["volume_ratio"] >= VPA_CAPITULATION_VOLUME_RATIO_THRESHOLD
        )
        drop_extreme = (
            row["pct_change"] <= VPA_CAPITULATION_DROP_PCT_THRESHOLD
            or row["consecutive_down_days"] >= VPA_CAPITULATION_CONSECUTIVE_DOWN_DAYS
        )
        if vol_extreme and drop_extreme:
            capitulation_idx = i
            break

    volume_regime = VOLUME_REGIME_NORMAL
    reversal_state = REVERSAL_STATE_NONE

    if capitulation_idx is not None:
        volume_regime = VOLUME_REGIME_CAPITULATION
        # Follow-through bars occur strictly AFTER capitulation_idx within recent_bars (all <= cutoff)
        follow_through_bars = recent_bars.iloc[capitulation_idx + 1 :]
        if len(follow_through_bars) == 0:
            reversal_state = REVERSAL_STATE_UNCONFIRMED
        else:
            cap_close = recent_bars.iloc[capitulation_idx]["close"]
            confirmed = False
            for _, ft_row in follow_through_bars.iterrows():
                if ft_row["close"] > cap_close or ft_row["pct_change"] > 0:
                    confirmed = True
                    break
            reversal_state = REVERSAL_STATE_CONFIRMED if confirmed else REVERSAL_STATE_UNCONFIRMED
    else:
        # Check for high volume stagnation candidate
        stagnation_idx = None
        for i in range(len(recent_bars) - 1, -1, -1):
            row = recent_bars.iloc[i]
            vol_high = (
                row["volume_ratio"] >= VPA_STAGNATION_VOLUME_RATIO_THRESHOLD
                or row["volume_zscore"] >= VPA_STAGNATION_ZSCORE_THRESHOLD
            )
            move_narrow = (
                abs(row["pct_change"]) <= VPA_STAGNATION_MAX_ABS_PCT_CHANGE
                or row["bar_spread"] <= VPA_STAGNATION_MAX_SPREAD
            )
            if vol_high and move_narrow:
                stagnation_idx = i
                break

        if stagnation_idx is not None:
            volume_regime = VOLUME_REGIME_HIGH_VOLUME_STAGNATION
            reversal_state = REVERSAL_STATE_UNCONFIRMED
        else:
            volume_regime = VOLUME_REGIME_NORMAL
            reversal_state = REVERSAL_STATE_NONE

    features = {
        "volume_zscore": round(float(last_bar.get("volume_zscore", 0.0)), 2),
        "volume_ratio": round(float(last_bar.get("volume_ratio", 1.0)), 2),
        "bar_spread": round(float(last_bar.get("bar_spread", 0.0)), 4),
        "close_position": round(float(last_bar.get("close_position", 0.5)), 2),
        "pct_change": round(float(last_bar.get("pct_change", 0.0)), 4),
        "consecutive_down_days": int(last_bar.get("consecutive_down_days", 0)),
        "follow_through_count": int(len(recent_bars) - 1 - capitulation_idx) if capitulation_idx is not None else 0,
    }

    return {
        "volume_regime": volume_regime,
        "reversal_state": reversal_state,
        "features": features,
        "as_of": as_of,
        "cutoff": cutoff_str or as_of,
    }


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


def make_cache_key(
    ticker: str,
    trade_date: str,
    price_basis: str = PRICE_BASIS_VENDOR_QFQ,
) -> str:
    norm_price_basis = validate_price_basis_short_label(price_basis)
    if norm_price_basis in (
        PRICE_BASIS_PIT_RAW,
        PRICE_BASIS_PIT_ADJUSTED,
        PRICE_BASIS_UNSPECIFIED,
    ):
        raise UnsupportedPriceBasisError(
            f"price_basis={norm_price_basis!r} is not implemented as an available channel"
        )
    if norm_price_basis == PRICE_BASIS_RAW:
        return f"{ticker}_{trade_date}_{PRICE_BASIS_RAW}"
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


def _serialize_scale_metrics_for_json(val: Any) -> Any:
    """Convert scale_metrics dictionary to JSON-native types for storage and API boundaries.

    Specifically:
    - Decimal (including DecimalRatio) is converted to a decimal string to retain exact precision.
    - Preserves None, bool, int, float, str, list, dict.
    - Preserves 0, None, gaps, and nested dictionary structures.
    - Raises TypeError for unhandled/unknown objects (no default=str masking).
    """
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, Decimal):
        return "0" if val == 0 else format(val, "f")
    if isinstance(val, Mapping):
        return {str(k): _serialize_scale_metrics_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_scale_metrics_for_json(item) for item in val]
    raise TypeError(f"Object of type {type(val).__name__} in scale_metrics is not JSON serializable")


def default_market_data_context() -> Dict[str, Any]:
    """Return a safe context when collection did not provide one."""
    empty_vpa = compute_vpa_deterministic_features(None)
    default_scale = _serialize_scale_metrics_for_json({
        "status": "unavailable",
        "net_to_circ_mv": None,
        "net_to_amount": None,
        "gaps": ["未提供市场数据上下文，资金规模归一指标不可用"],
    })
    return {
        "analysis_baseline_date": None,
        "fund_flow_evidence": {
            "status": "unavailable",
            "unit": "亿元",
            "records": [],
            "summary": summarize_evidence([], window_days=5),
            "validation": {"status": "not_checked", "mismatches": []},
            "gap": "【数据获取失败】资金流 evidence：未返回结构化逐日记录",
            "scale_metrics": default_scale,
        },
        "scale_metrics": default_scale,
        "vpa_context": empty_vpa,
        "vpa_structured": empty_vpa,
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
    "dividend_evidence",
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
        return "数据源调用失败"
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
    if "【财务数据截至" in normalized or "生效公告日" in normalized:
        return None
    if "停止披露" in normalized or "披露停止" in normalized:
        return "unavailable"
    if "仅提供当前快照" in normalized or "仅支持当日快照" in normalized:
        return "refused"
    if any(marker.lower() in lowered for marker in ("调用失败", "调用异常", "拉取失败", "抓取失败")):
        return "failed"
    if "数据拉取超时" in normalized or "timeout" in lowered or "超时" in normalized:
        return "timeout"
    if any(marker.lower() in lowered for marker in _DATA_FAILURE_MARKERS):
        return "failed"
    return None


def _determine_gap_class(source: str, value: Any, status: str) -> str:
    """Classify a data failure/gap as 'structural' or 'operational'.

    Structural:
      - Northbound institutional stoppage (e.g. daily holdings discontinued from Aug 2024).
      - Historical snapshot refusal (snapshot_historical_refusal, e.g. share_pledge, fund_flow_board, hot_stocks).
      - Explicit status 'refused' or refusal markers.

    Operational:
      - Transient network, transport, token, timeout, connection failures.
      - Empty table/dataframe, format/parse errors.
      - Unverified as-of, missing completed daily bars.
      - Quality gate / calculation failures.
    """
    if isinstance(value, dict):
        explicit_class = value.get("gap_class")
        if explicit_class in ("structural", "operational"):
            return explicit_class
        if str(value.get("status") or "").strip().lower() == "refused":
            return "structural"
        reason = str(value.get("reason") or "")
        gap = str(value.get("gap") or "")
        combined_meta = f"{reason} {gap}"
        if any(marker in combined_meta for marker in ("停止披露", "披露停止", "制度性停更", "仅提供当前快照", "仅支持当日快照", "无法用于历史日期分析", "快照拒绝")):
            return "structural"

    text = value if isinstance(value, str) else str(value or "")

    # 1. Structural Northbound stoppage
    if source == "northbound_flow" or any(marker in text for marker in ("停止披露", "披露停止", "制度性停更", "沪深港通个股每日持股明细自 2024 年 8 月起停止披露")):
        return "structural"

    # 2. Structural historical snapshot refusal
    if status == "refused" or any(marker in text for marker in (
        "仅提供当前快照",
        "仅支持当日快照",
        "无法用于历史日期分析",
        "快照拒绝",
        "仅提供近窗，非全历史",
        "当前热度快照",
        "全市场快照",
    )):
        return "structural"

    # 3. Default to operational
    return "operational"


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
        gap_class = _determine_gap_class(source_name, value, status)
        entries.append(
            (
                source_rank.get(source_name, len(_DATA_FAILURE_SOURCE_ORDER)),
                source_name,
                {
                    "source": source_name,
                    "status": status,
                    "reason": reason,
                    "gap": f"【数据获取失败】{source_name}：{reason}",
                    "gap_class": gap_class,
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
    r"排查基准日\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"数据日期[】：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"生效公告日\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"截止日(?:期)?\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"统计截止日\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"公告日(?:期)?\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"报告日(?:期)?\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"报告期(?:截止日|日)?\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"解禁日(?:期)?\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"变动日(?:期)?\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"交易日(?:期)?\s*[：:]?\s*(20\d{2}-\d{2}-\d{2})",
    r"截至于\s*(20\d{2}-\d{2}-\d{2})",
    r"日期\s*[：:]\s*(20\d{2}-\d{2}-\d{2})",
    r"龙虎榜明细[（(]\s*(20\d{2}-\d{2}-\d{2})",
    r"涨停池[（(]\s*(20\d{2}-\d{2}-\d{2})",
    r"(20\d{2}-\d{2}-\d{2})\s+涨停家数",
    r"\[(20\d{2}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\]",
)

_SOURCE_AS_OF_YYYYMMDD_PATTERNS = (
    r"查询报告期\s*=\s*(20\d{2})(\d{2})(\d{2})",
    r"报告期\s*[：:=]?\s*(20\d{2})(\d{2})(\d{2})",
)


def _extract_source_as_of(value: Any, requested_as_of: str) -> Optional[str]:
    """Extract the latest explicitly reported source date, excluding request windows."""
    if isinstance(value, dict):
        for key in ("as_of", "actual_as_of", "quote_as_of", "data_as_of"):
            candidate = value.get(key)
            if candidate is not None:
                match = re.search(r"20\d{2}-\d{2}-\d{2}", str(candidate))
                if match:
                    if match.group(0) <= requested_as_of:
                        return match.group(0)
                else:
                    logger.warning("Explicit date field %s='%s' cannot be parsed", key, candidate)
        return None

    if hasattr(value, "fund_flow_evidence_meta") and isinstance(value.fund_flow_evidence_meta, dict):
        for key in ("as_of", "actual_as_of", "quote_as_of", "data_as_of"):
            candidate = value.fund_flow_evidence_meta.get(key)
            if candidate is not None:
                match = re.search(r"20\d{2}-\d{2}-\d{2}", str(candidate))
                if match:
                    if match.group(0) <= requested_as_of:
                        return match.group(0)
                else:
                    logger.warning("fund_flow_evidence_meta date field %s='%s' cannot be parsed", key, candidate)

    if hasattr(value, "as_of") and getattr(value, "as_of"):
        candidate = getattr(value, "as_of")
        if candidate is not None:
            match = re.search(r"20\d{2}-\d{2}-\d{2}", str(candidate))
            if match:
                if match.group(0) <= requested_as_of:
                    return match.group(0)
            else:
                logger.warning("Explicit as_of attribute '%s' cannot be parsed", candidate)

    text = value if isinstance(value, str) else str(value or "")
    candidates: list[str] = []
    for pattern in _SOURCE_AS_OF_PATTERNS:
        candidates.extend(match.group(1) for match in re.finditer(pattern, text))
    for pattern in _SOURCE_AS_OF_YYYYMMDD_PATTERNS:
        for match in re.finditer(pattern, text):
            candidates.append(f"{match.group(1)}-{match.group(2)}-{match.group(3)}")
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
        if source in ("cninfo_announcements", "cninfo_records", "cninfo_ir_surveys"):
            continue
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
                entry["gap_class"] = _determine_gap_class(str(source), value, status)
                entry["provenance_status"] = "refused"
            elif actual_as_of is None:
                entry["status"] = "unavailable"
                entry["gap"] = "【数据获取失败】industry_linkage：未返回可验证数据日期"
                entry["gap_class"] = "operational"
                entry["provenance_status"] = "refused"
            elif actual_as_of > requested_as_of:
                entry["status"] = "future"
                entry["gap"] = f"【数据获取失败】industry_linkage：数据日期 {actual_as_of} 晚于请求日期 {requested_as_of}"
                entry["gap_class"] = "operational"
                entry["provenance_status"] = "future"
            else:
                entry["provenance_status"] = "verified"
            provenance[str(source)] = entry
            continue

        classified_status = _classify_failure_value(value)
        as_of = _extract_source_as_of(value, requested_as_of)
        if source == "stock_data":
            as_of = daily_as_of

        if classified_status is not None:
            status = classified_status
            gap_class = _determine_gap_class(str(source), value, status)
            entry = {
                "requested_as_of": requested_as_of,
                "actual_as_of": as_of,
                "as_of": as_of,
                "status": status,
                "gap_class": gap_class,
                "gap": f"【数据获取失败】{source}：{_compact_failure_reason(status)}",
                "provenance_status": "refused",
            }
        else:
            if source == "stock_data" and as_of and as_of < requested_as_of:
                status = "unavailable"
                gap_class = "operational"
                entry = {
                    "requested_as_of": requested_as_of,
                    "actual_as_of": as_of,
                    "as_of": as_of,
                    "status": status,
                    "gap_class": gap_class,
                    "gap": (
                        f"【数据获取失败】stock_data：实际最新数据日 {as_of} "
                        f"早于请求日期 {requested_as_of}"
                    ),
                    "provenance_status": "refused",
                }
            elif as_of is None and source != "realtime":
                if (
                    source in _FINANCIAL_PROVENANCE_SOURCES
                    and _has_financial_field_value_pair(value)
                ):
                    # 有可解析财务字段+数值但抽不出 ISO as_of：账本假缺口，不是没拉到表。
                    entry = {
                        "requested_as_of": requested_as_of,
                        "actual_as_of": None,
                        "as_of": None,
                        "status": "available_unverified_as_of",
                        "provenance_status": "unverified",
                        "note": "有可解析财务字段与数值但缺少可验证 ISO 数据日期",
                    }
                else:
                    status = "unavailable"
                    gap_class = "operational"
                    entry = {
                        "requested_as_of": requested_as_of,
                        "actual_as_of": None,
                        "as_of": None,
                        "status": status,
                        "gap_class": gap_class,
                        "gap": f"【数据获取失败】{source}：未返回可验证数据日期",
                        "provenance_status": "refused",
                    }
            else:
                if as_of is not None and as_of > requested_as_of:
                    entry = {
                        "requested_as_of": requested_as_of,
                        "actual_as_of": as_of,
                        "as_of": as_of,
                        "status": "future",
                        "gap_class": "operational",
                        "gap": f"【数据获取失败】{source}：实际最新数据日 {as_of} 晚于请求日期 {requested_as_of}",
                        "provenance_status": "future",
                    }
                elif as_of is None:
                    entry = {
                        "requested_as_of": requested_as_of,
                        "actual_as_of": None,
                        "as_of": None,
                        "status": "available",
                        "provenance_status": "unverified",
                    }
                else:
                    entry = {
                        "requested_as_of": requested_as_of,
                        "actual_as_of": as_of,
                        "as_of": as_of,
                        "status": "available",
                        "provenance_status": "verified",
                    }

        provenance[str(source)] = entry
    return provenance


def _build_market_attention(
    results: Dict[str, Any],
    source_provenance: Dict[str, Dict[str, Any]],
    requested_as_of: str,
) -> Dict[str, Any]:
    """Extract market attention context from zt_pool and hot_stocks (Task 8 / §1 / D-008).

    Guarantees:
    - Retains status and verified as_of per source.
    - Explicit failure/gap metadata when source fails or is refused; never silent empty dict.
    - Preserves raw source payload for downstream consumption without inferring retail sentiment scores.
    """
    attention: Dict[str, Any] = {}
    for key in ("zt_pool", "hot_stocks"):
        raw_val = results.get(key)
        prov = source_provenance.get(key) if isinstance(source_provenance, dict) else None

        if prov and isinstance(prov, dict):
            status = prov.get("status", "available")
            as_of = prov.get("as_of") or prov.get("actual_as_of")
            gap = prov.get("gap")
            gap_class = prov.get("gap_class")
        else:
            classified = _classify_failure_value(raw_val)
            if classified is not None:
                status = classified
                gap_class = _determine_gap_class(key, raw_val, status)
                reason = _compact_failure_reason(status)
                gap = f"【数据获取失败】{key}：{reason}"
                as_of = None
            elif raw_val is None or raw_val == "":
                status = "unavailable"
                gap_class = "operational"
                gap = f"【数据获取失败】{key}：无数据"
                as_of = None
            else:
                status = "available"
                as_of = _extract_source_as_of(raw_val, requested_as_of)
                gap = None
                gap_class = None

        entry: Dict[str, Any] = {
            "status": status,
            "as_of": as_of,
            "requested_as_of": requested_as_of,
            "raw": raw_val,
        }
        if gap:
            entry["gap"] = gap
        if gap_class:
            entry["gap_class"] = gap_class

        attention[key] = entry
    return attention


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



_FINANCIAL_PROVENANCE_SOURCES = frozenset(
    {"fundamentals", "balance_sheet", "income_statement", "cashflow"}
)
# Field names that must appear paired with a numeric value — codes/years alone do not count.
_FINANCIAL_FIELD_NAMES = (
    "总资产",
    "总负债",
    "净资产",
    "资产总计",
    "负债合计",
    "所有者权益合计",
    "所有者权益",
    "货币资金",
    "应收账款",
    "存货",
    "营业总收入",
    "营业收入",
    "营业总成本",
    "净利润",
    "归属于母公司所有者的净利润",
    "归属于母公司",
    "每股收益",
    "毛利率",
    "经营活动产生的现金流量净额",
    "经营活动现金流入小计",
    "销售商品、提供劳务收到的现金",
)


def _has_financial_field_value_pair(value: Any) -> bool:
    """True only when a known financial field name is paired with a numeric value.

    Stock codes (e.g. 688981.SH) and bare report years must not count as data.
    """
    text = value if isinstance(value, str) else str(value or "")
    if not text.strip():
        return False
    for field in _FINANCIAL_FIELD_NAMES:
        # 「总资产 123.45」/ 「总资产：123」/ 「总资产=123」
        if re.search(rf"{re.escape(field)}\s*[:：=]?\s*-?\d", text):
            return True
        # table-ish 「123.45 总资产」
        if re.search(rf"-?\d[\d,.]*(?:\s+|\|){re.escape(field)}", text):
            return True
    return False


_COLLECTOR_AS_OF_ISO_RE = re.compile(r"^(20\d{2})-(\d{2})-(\d{2})$")
_COLLECTOR_AS_OF_COMPACT_RE = re.compile(r"^(20\d{2})(\d{2})(\d{2})$")


def _parse_collector_as_of(trade_date: Any) -> tuple[datetime, str]:
    """Parse collector as-of; only YYYY-MM-DD or YYYYMMDD. Never fall back to now."""
    if not isinstance(trade_date, str):
        raise TypeError(
            f"非法分析日期 as-of: {trade_date!r}；仅接受 YYYY-MM-DD 或 YYYYMMDD"
        )
    raw = trade_date.strip()
    match = _COLLECTOR_AS_OF_ISO_RE.fullmatch(raw)
    if match:
        year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        try:
            end_dt = datetime(year, month, day)
        except ValueError as exc:
            raise ValueError(
                f"非法分析日期 as-of: {trade_date!r}；仅接受 YYYY-MM-DD 或 YYYYMMDD"
            ) from exc
        return end_dt, end_dt.strftime("%Y-%m-%d")
    match = _COLLECTOR_AS_OF_COMPACT_RE.fullmatch(raw)
    if match:
        year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        try:
            end_dt = datetime(year, month, day)
        except ValueError as exc:
            raise ValueError(
                f"非法分析日期 as-of: {trade_date!r}；仅接受 YYYY-MM-DD 或 YYYYMMDD"
            ) from exc
        return end_dt, end_dt.strftime("%Y-%m-%d")
    raise ValueError(
        f"非法分析日期 as-of: {trade_date!r}；仅接受 YYYY-MM-DD 或 YYYYMMDD"
    )


def _fetch_raw_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    as_of: Optional[str] = None,
) -> str:
    """Fetch unadjusted (raw) daily bars directly from CnAkshareProvider in registry.

    Fails closed if cn_akshare provider is missing or if raw daily fetch fails.
    Never falls back to vendor qfq.
    """
    prov = _registry.get("cn_akshare")
    if prov is None or not hasattr(prov, "get_stock_data"):
        return (
            "【数据获取失败】stock_data: cn_akshare provider 不可用，无法获取 raw 未复权日线行情 "
            "(provider_unavailable)"
        )
    try:
        return prov.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            price_basis=PRICE_BASIS_RAW,
            as_of=as_of or end_date,
        )
    except RawDailyFetchError as exc:
        return f"【数据获取失败】stock_data raw 失败: {exc.error} ({exc.category})"
    except Exception as exc:
        return f"【数据获取失败】stock_data raw 调用异常: {type(exc).__name__}: {exc}"


class DividendEvidenceText(str):
    """Compact text representation of corporate action dividend collateral evidence."""

    def __new__(
        cls,
        content: str,
        records: Optional[List[dict]] = None,
        status: str = "ok",
        as_of: Optional[str] = None,
    ):
        obj = super().__new__(cls, content)
        obj.records = list(records) if records else []
        obj.status = status
        obj.as_of = as_of
        return obj


def _format_dividend_row_compact(r: dict) -> str:
    """Format one dividend record into a compact token-efficient line."""
    ann_date = r.get("ann_date") or "-"
    end_date = r.get("end_date") or "-"
    div_proc = r.get("div_proc") or "-"
    cash_div = r.get("cash_div")
    stk_div = r.get("stk_div")
    stk_bo_rate = r.get("stk_bo_rate")
    ex_date = r.get("ex_date") or "-"
    record_date = r.get("record_date") or "-"
    pay_date = r.get("pay_date") or "-"
    imp_ann_date = r.get("imp_ann_date") or "-"

    cash_str = f"{cash_div}元" if cash_div is not None else "-"
    stk_str = str(stk_div) if stk_div is not None else "-"
    bo_str = str(stk_bo_rate) if stk_bo_rate is not None else "-"

    parts = [
        f"预案公告日(ann_date): {ann_date}",
        f"分红年度(end_date): {end_date}",
        f"方案进度(div_proc): {div_proc}",
        f"每股派现(cash_div): {cash_str}",
        f"每股送股(stk_div): {stk_str}",
        f"每股转增(stk_bo_rate): {bo_str}",
        f"除权除息日(ex_date): {ex_date}",
        f"股权登记日(record_date): {record_date}",
    ]
    if pay_date and pay_date != "-":
        parts.append(f"派息日(pay_date): {pay_date}")
    if imp_ann_date and imp_ann_date != "-":
        parts.append(f"实施公告日(imp_ann_date): {imp_ann_date}")
    gap = r.get("pit_implementation_gap")
    if gap:
        parts.append(f"PIT边界说明: {gap}")
    return " | ".join(parts)


def _format_dividend_compact(records: list[dict], as_of: str) -> str:
    """Format dividend records into compact text representation without raw JSON dumping."""
    lines = [f"【分红送转旁证（公司行动）】（数据基准日：{as_of}，共 {len(records)} 条历史记录）："]
    for r in records:
        lines.append(f"- {_format_dividend_row_compact(r)}")
    return "\n".join(lines)


def _fetch_dividend_evidence(symbol: str, as_of: str) -> DividendEvidenceText:
    """Fetch corporate action dividend collateral via CnAkshareProvider._fetch_tushare_dividend.

    Enforces contract (D-02-5 / C-04-3 / DAV-715):
    1. Accesses cn_akshare provider through _registry.get("cn_akshare").
       Calls _fetch_tushare_dividend(symbol, as_of=as_of) where as_of is historical trade_date.
       No provider -> explicit failure string (never pretend no dividend).
    2. Success -> compact formatted text by column names (ann_date, ex_date, cash_div, stk_div, etc.),
       never raw list[dict] or JSON dump.
    3. Empty table (no_rows) -> explicit '空表，不得据此判断无分红', never '该公司不分红' or '无分红'.
    4. D-01 typed failures (token_missing, transport, permission_denied, date_exceeds_as_of, missing_field) ->
       '【数据获取失败】分红旁证 — 原因：{reason}，该项不可用。'
    5. Never modifies stock_data prices or price_basis.
    """
    prov = _registry.get("cn_akshare")
    if prov is None or not hasattr(prov, "_fetch_tushare_dividend"):
        return DividendEvidenceText(
            "【数据获取失败】分红旁证 — 原因：cn_akshare provider 不可用 (provider_unavailable)，该项不可用。",
            status="failed",
            as_of=as_of,
        )

    try:
        res = prov._fetch_tushare_dividend(symbol=symbol, as_of=as_of)
    except Exception as exc:
        return DividendEvidenceText(
            f"【数据获取失败】分红旁证 — 原因：调用异常 {type(exc).__name__}: {exc}，该项不可用。",
            status="failed",
            as_of=as_of,
        )

    if isinstance(res, (list, tuple)) and len(res) == 3:
        records, error, category = res
    else:
        # Fallback for untyped mock in other tests without explicit return_value
        return DividendEvidenceText(
            f"【分红送转旁证（公司行动）】（数据基准日：{as_of}）：空表，不得据此判断无分红。",
            status="empty",
            as_of=as_of,
        )

    # 1. Empty table: no_rows
    if category == "no_rows" or (records is not None and len(records) == 0 and not error):
        return DividendEvidenceText(
            f"【分红送转旁证（公司行动）】（数据基准日：{as_of}）：空表，不得据此判断无分红。",
            records=[],
            status="empty",
            as_of=as_of,
        )

    # 2. Typed errors (token_missing, transport, permission_denied, date_exceeds_as_of, missing_field, etc.)
    if error or category:
        detail = (
            f"{category} ({error})"
            if error and str(error).strip() != str(category).strip()
            else str(error or category or "未知错误")
        )
        return DividendEvidenceText(
            f"【数据获取失败】分红旁证 — 原因：{detail}，该项不可用。",
            status="failed",
            as_of=as_of,
        )

    # 3. Successful records
    if isinstance(records, list) and records:
        content = _format_dividend_compact(records, as_of)
        return DividendEvidenceText(
            content,
            records=records,
            status="ok",
            as_of=as_of,
        )

    # Fallback to empty table
    return DividendEvidenceText(
        f"【分红送转旁证（公司行动）】（数据基准日：{as_of}）：空表，不得据此判断无分红。",
        status="empty",
        as_of=as_of,
    )


def _fetch_all(
    ticker: str,
    trade_date: str,
    industry_provider: Optional[IndustryLinkageProvider] = None,
    price_basis: str = PRICE_BASIS_VENDOR_QFQ,
) -> Dict[str, Any]:
    """Fetch all data sources in parallel.

    Always fetches full data including financial statements, regardless of horizon.
    The horizon only affects the analysis window, not data collection.
    """
    norm_price_basis = validate_price_basis_short_label(price_basis)
    if norm_price_basis in (
        PRICE_BASIS_PIT_RAW,
        PRICE_BASIS_PIT_ADJUSTED,
        PRICE_BASIS_UNSPECIFIED,
    ):
        raise UnsupportedPriceBasisError(
            f"price_basis={norm_price_basis!r} is not implemented as an available channel in DataCollector"
        )

    lookback = LONG_DAYS
    end_dt, trade_date = _parse_collector_as_of(trade_date)
    norm_trade_date = trade_date

    # 为了计算指标准确（如 200 SMA），需要比分析窗口更长的历史数据
    fetch_lookback = 365
    start_str = (end_dt - timedelta(days=fetch_lookback)).strftime("%Y-%m-%d")
    news_start_date = (end_dt - timedelta(days=lookback)).strftime("%Y-%m-%d")

    cn_provider = _registry.get("cn_akshare")

    if norm_price_basis == PRICE_BASIS_RAW:
        stock_data_task = (
            _fetch_raw_stock_data,
            {
                "symbol": ticker,
                "start_date": start_str,
                "end_date": norm_trade_date,
                "as_of": norm_trade_date,
            },
        )
    else:
        stock_data_task = (
            get_stock_data,
            {"symbol": ticker, "start_date": start_str, "end_date": norm_trade_date},
        )

    tasks: Dict[str, tuple] = {
        "stock_data": stock_data_task,
        "cn_indices": (get_cn_indices, {"curr_date": norm_trade_date, "look_back_days": lookback}),
        "global_indices": (get_global_indices, {"curr_date": norm_trade_date, "look_back_days": lookback}),
        "major_assets": (get_major_assets, {"curr_date": norm_trade_date, "look_back_days": lookback}),
        "realtime": (_fetch_realtime_context, {"ticker": ticker, "trade_date": norm_trade_date}),
        "news": (get_news, {"ticker": ticker, "start_date": news_start_date, "end_date": norm_trade_date}),
        "cninfo_announcements": (
            cn_provider.get_cninfo_announcements
            if cn_provider and hasattr(cn_provider, "get_cninfo_announcements")
            else lambda **kw: CninfoDisclosureEnvelope(
                status=STATUS_PROVIDER_FAILURE,
                records=[],
                error="cn_akshare provider unavailable",
                source_type=SOURCE_TYPE_ANNOUNCEMENT,
            ),
            {
                "symbol": ticker,
                "start_date": news_start_date,
                "end_date": norm_trade_date,
                "cutoff": norm_trade_date,
            },
        ),
        "cninfo_ir_surveys": (
            cn_provider.get_cninfo_ir_surveys
            if cn_provider and hasattr(cn_provider, "get_cninfo_ir_surveys")
            else lambda **kw: CninfoDisclosureEnvelope(
                status=STATUS_PROVIDER_FAILURE,
                records=[],
                error="cn_akshare provider unavailable",
                source_type=SOURCE_TYPE_IR_SURVEY,
            ),
            {
                "symbol": ticker,
                "start_date": news_start_date,
                "end_date": norm_trade_date,
                "cutoff": norm_trade_date,
            },
        ),
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
        "dividend_evidence": (
            _fetch_dividend_evidence,
            {"symbol": ticker, "as_of": norm_trade_date},
        ),
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

    # 保证 cninfo_announcements 始终为 envelope
    ann_val = results.get("cninfo_announcements")
    if ann_val is None or (isinstance(ann_val, str) and ("调用失败" in ann_val or "超时" in ann_val)):
        results["cninfo_announcements"] = CninfoDisclosureEnvelope(
            status=STATUS_PROVIDER_FAILURE,
            records=[],
            error=str(ann_val) if ann_val else "cninfo_announcements unavailable",
            source_type=SOURCE_TYPE_ANNOUNCEMENT,
        )
    elif ann_val == "":
        results["cninfo_announcements"] = CninfoDisclosureEnvelope(
            status=STATUS_OK,
            records=[],
            source_type=SOURCE_TYPE_ANNOUNCEMENT,
        )

    # 保证 cninfo_ir_surveys 始终为 envelope
    ir_val = results.get("cninfo_ir_surveys")
    if (
        ir_val is None
        or isinstance(ir_val, str)
        or not (hasattr(ir_val, "status") and hasattr(ir_val, "records"))
    ):
        results["cninfo_ir_surveys"] = CninfoDisclosureEnvelope(
            status=STATUS_PROVIDER_FAILURE,
            records=[],
            error=str(ir_val) if ir_val else "cninfo_ir_surveys unavailable",
            source_type=SOURCE_TYPE_IR_SURVEY,
        )

    # 保证 dividend_evidence 始终有效并符合契约
    div_val = results.get("dividend_evidence")
    if div_val is None or div_val == "" or not isinstance(div_val, (str, DividendEvidenceText)):
        results["dividend_evidence"] = _fetch_dividend_evidence(ticker, as_of=norm_trade_date)
    elif isinstance(div_val, str) and "超时" in div_val and not div_val.startswith("【数据获取失败】"):
        results["dividend_evidence"] = DividendEvidenceText(
            f"【数据获取失败】分红旁证 — 原因：数据拉取超时 ({div_val})，该项不可用。",
            status="timeout",
            as_of=norm_trade_date,
        )

    # ── 对巨潮 envelope 调用 qualify_cninfo_content (C-05 Slice 9) ──
    cn_provider = cn_provider or _registry.get("cn_akshare")
    for key in ("cninfo_announcements", "cninfo_ir_surveys"):
        env = results.get(key)
        if (
            env is not None
            and hasattr(env, "status")
            and env.status != STATUS_PROVIDER_FAILURE
            and hasattr(env, "records")
            and isinstance(env.records, list)
        ):
            qualify_fn = (
                getattr(cn_provider, "qualify_cninfo_content", None)
                if cn_provider is not None
                else None
            ) or qualify_cninfo_content
            for i, rec in enumerate(env.records):
                try:
                    qualified = qualify_fn(rec, cutoff=norm_trade_date)
                    if qualified is not None:
                        env.records[i] = qualified
                except Exception as exc:
                    logger.warning("qualify_cninfo_content 异常 (%s): %s", key, exc)
                    if hasattr(rec, "content_status"):
                        rec.content_status = CONTENT_STATUS_UNAVAILABLE
                    if hasattr(rec, "content_sha256"):
                        rec.content_sha256 = None

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
                "gap_class": "operational",
            })

    # ── 资金规模归一接入 (D-03-2 / C-09-3) ──────────────────────────
    prov = _registry.get("cn_akshare")
    basic_row: Optional[Dict[str, Any]] = None
    basic_err: Optional[str] = None
    basic_cat: Optional[str] = None
    if prov is not None and hasattr(prov, "_fetch_tushare_daily_basic"):
        try:
            basic_row, basic_err, basic_cat = prov._fetch_tushare_daily_basic(
                symbol=ticker,
                trade_date=norm_trade_date,
                as_of=norm_trade_date,
            )
        except TypeError:
            try:
                basic_row, basic_err, basic_cat = prov._fetch_tushare_daily_basic(
                    ticker,
                    norm_trade_date,
                    norm_trade_date,
                )
            except Exception as exc:
                basic_row = None
                basic_err = f"daily_basic 调用异常: {type(exc).__name__}: {exc}"
                basic_cat = "exception"
        except Exception as exc:
            basic_row = None
            basic_err = f"daily_basic 调用异常: {type(exc).__name__}: {exc}"
            basic_cat = "exception"
    else:
        basic_err = "cn_akshare provider 不可用或缺少 _fetch_tushare_daily_basic"
        basic_cat = "provider_unavailable"

    net_val = None
    if isinstance(fund_flow_context, Mapping):
        for k in ("selected_value", "net_amount", "value", "r0_net", "netamount"):
            if fund_flow_context.get(k) is not None:
                net_val = fund_flow_context[k]
                break

    net_unit = None
    if isinstance(fund_flow_context, Mapping):
        for k in ("selected_unit", "net_amount_unit", "unit", "raw_unit"):
            val = fund_flow_context.get(k)
            if val is not None and str(val).strip():
                net_unit = str(val).strip()
                break

    evidence_trade_date = None
    if isinstance(fund_flow_context, Mapping):
        for k in ("selected_as_of", "as_of", "trade_date", "date", "measurement_date"):
            val = fund_flow_context.get(k)
            if val is not None and str(val).strip():
                evidence_trade_date = str(val).strip()
                break

    evidence_symbol = None
    if isinstance(fund_flow_context, Mapping):
        for k in ("ts_code", "symbol", "code"):
            val = fund_flow_context.get(k)
            if val is not None and str(val).strip():
                evidence_symbol = str(val).strip()
                break

    scale_metrics = calculate_fund_flow_scale_metrics(
        ts_code=evidence_symbol or ticker,
        trade_date=evidence_trade_date or norm_trade_date,
        net_amount=net_val,
        net_amount_unit=net_unit,
        daily_basic=basic_row,
        circ_mv_unit="万元" if basic_row else None,
        amount_unit=basic_row.get("amount_unit") if isinstance(basic_row, Mapping) else None,
        denominator_source="tushare.daily_basic",
    )
    if basic_err:
        gap_msg = f"daily_basic分母数据获取失败: {basic_err}"
        if gap_msg not in scale_metrics["gaps"]:
            scale_metrics["gaps"].append(gap_msg)
        scale_metrics["error"] = basic_err
        scale_metrics["error_category"] = basic_cat

    scale_metrics = _serialize_scale_metrics_for_json(scale_metrics)

    fund_flow_context["scale_metrics"] = scale_metrics
    results["scale_metrics"] = scale_metrics

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
        if norm_price_basis == PRICE_BASIS_RAW:
            provenance = [line for line in provenance if line != f"# price_basis: {PRICE_BASIS_VENDOR_QFQ}"]
            if not any(line.startswith("# price_basis:") for line in provenance):
                provenance.insert(0, f"# price_basis: {PRICE_BASIS_RAW}")
            formatted = "\n".join(provenance) + "\n" + out.to_csv(index=False)
            results["stock_data"] = StockDataText(formatted, price_basis=PRICE_BASIS_RAW)
        else:
            provenance = [line for line in provenance if line != f"# price_basis: {PRICE_BASIS_RAW}"]
            if not any(line.startswith("# price_basis:") for line in provenance):
                provenance.insert(0, f"# price_basis: {PRICE_BASIS_VENDOR_QFQ}")
            formatted = "\n".join(provenance) + "\n" + out.to_csv(index=False)
            results["stock_data"] = StockDataText(formatted, price_basis=PRICE_BASIS_VENDOR_QFQ)
    else:
        if norm_price_basis == PRICE_BASIS_RAW:
            err_details = (
                str(raw_csv).strip()
                if isinstance(raw_csv, str) and ("失败" in raw_csv or "tushare" in raw_csv or "provider" in raw_csv or "error" in raw_csv.lower())
                else f"{ticker} 在 {trade_date} 无有效完整未复权日线数据（缺列/非法日期/全部行无效/重复冲突），本项不可用。"
            )
            if not err_details.startswith("【数据获取失败】"):
                err_details = f"【数据获取失败】stock_data raw 失败: {err_details}"
            results["stock_data"] = err_details
            if not any(
                isinstance(entry, dict) and entry.get("source") == "stock_data"
                for entry in data_failure_ledger
            ):
                data_failure_ledger.append(
                    {
                        "source": "stock_data",
                        "status": "failed",
                        "reason": "raw daily fetch failed",
                        "gap": err_details,
                        "gap_class": "operational",
                    }
                )
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
                        "gap_class": "operational",
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
                    "status": str(provenance.get("status") or "unavailable"),
                    "reason": "unverified as-of",
                    "gap": str(gap),
                    "gap_class": str(provenance.get("gap_class") or "operational"),
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
    # ── 结构化新闻事件覆盖度计算 ──────────────────
    news_text = results.get("news", "")
    global_news_text = results.get("global_news", "")
    stock_evs, stock_unp = parse_news_markdown_to_evidences(news_text, default_entity=ticker)
    glob_evs, glob_unp = parse_news_markdown_to_evidences(global_news_text, default_entity="宏观/行业")
    cninfo_evs: list[NewsEvidence] = []
    cninfo_envelopes: list[Any] = []
    for key in ("cninfo_records", "cninfo_announcements", "cninfo_ir_surveys"):
        val = results.get(key)
        if not val:
            continue
        if hasattr(val, "status") or (isinstance(val, dict) and "status" in val):
            cninfo_envelopes.append(val)
        records = getattr(val, "records", None) if hasattr(val, "records") else (val if isinstance(val, (list, tuple)) else [])
        for rec in records:
            ev = cninfo_record_to_evidence(rec, default_entity=ticker)
            if ev is not None:
                cninfo_evs.append(ev)

    cn_provider = cn_provider or _registry.get("cn_akshare")
    event_cov = build_news_event_coverage(
        stock_evs + glob_evs + cninfo_evs + stock_unp + glob_unp,
        requested_themes=None,
        cutoff=trade_date,
        window=f"{lookback}天",
        default_entity=ticker,
        cninfo_envelopes=cninfo_envelopes,
        provider=cn_provider,
    )
    results["event_coverage"] = event_cov

    market_attention = _build_market_attention(
        results, source_provenance, trade_date
    )

    results["market_data_context"] = {
        "analysis_baseline_date": trade_date,
        "fund_flow_evidence": fund_flow_context,
        "scale_metrics": scale_metrics,
        "dividend_evidence": results.get("dividend_evidence"),
        "event_coverage": event_cov,
        "daily": daily_context,
        "realtime": realtime_context,
        "global_indices": results.get("global_indices"),
        "cn_indices": results.get("cn_indices"),
        "major_assets": results.get("major_assets"),
        "industry_linkage": results.get("industry_linkage"),
        "market_attention": market_attention,
        "source_provenance": source_provenance,
        "data_failure_ledger": data_failure_ledger,
        "price_basis": norm_price_basis,
    }
    results["price_basis"] = norm_price_basis

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
            results["vpa_structured"] = compute_vpa_deterministic_features(df.copy(), cutoff=trade_date)
        else:
            results["vpa_indicators"] = "VPA 数据不足"
            results["vpa_structured"] = compute_vpa_deterministic_features(None, cutoff=trade_date)
    except Exception as e:
        results["vpa_indicators"] = f"VPA 计算失败：{e}"
        results["vpa_structured"] = compute_vpa_deterministic_features(None, cutoff=trade_date)

    results["vpa_context"] = results["vpa_structured"]
    results["market_data_context"]["vpa_context"] = results["vpa_structured"]
    results["market_data_context"]["vpa_structured"] = results["vpa_structured"]

    logger.debug("[Timer] Total Data Collection for %s took %.2fs", ticker, time.time() - fetch_start)
    return results


class DataCollector:
    """Collect and cache data, thread-safe and shareable across jobs."""

    def __init__(
        self,
        industry_linkage_provider: Optional[IndustryLinkageProvider] = None,
        social_collector: Optional[SocialDataCollector] = None,
    ):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()
        self._refcounts: Dict[str, int] = {}
        self.industry_linkage_provider: IndustryLinkageProvider = (
            industry_linkage_provider or IndustryLinkageProvider()
        )
        self.social_collector: SocialDataCollector = (
            social_collector if social_collector is not None else SocialDataCollector()
        )

    _map_stock_to_industry = staticmethod(_map_stock_to_industry)

    def _get_key_lock(self, key: str) -> threading.Lock:
        with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _fetch_social_context(self, ticker: str, trade_date: str) -> Dict[str, Any]:
        """Fetch social sentiment context after market collection with independent short timeout (Task 8 / §1 / D-008)."""
        collector = self.social_collector
        if collector is None:
            collector = SocialDataCollector()

        mode = getattr(collector, "mode", os.getenv("TA_SOCIAL_MODE", "disabled"))
        timeout = float(
            os.getenv("TA_SOCIAL_FETCH_TIMEOUT")
            or getattr(collector, "fetch_timeout", 5.0)
            or 5.0
        )

        if not hasattr(collector, "collect") or not callable(collector.collect):
            bundle = create_empty_sentiment_bundle(
                status=SocialStatus.FAILED.value,
                requested_as_of=trade_date,
                cutoff_at="",
                reason_codes=[REASON_SOCIAL_ARCHIVE_MISSING],
                symbol=ticker,
            )
            return create_default_social_data_context(
                status=SocialStatus.FAILED.value,
                mode=str(mode),
                requested_as_of=trade_date,
                reason_codes=[REASON_SOCIAL_ARCHIVE_MISSING],
                bundle=bundle,
                source_provenance={
                    "social_archive": {
                        "status": SocialStatus.FAILED.value,
                        "provider": getattr(collector, "provider_name", "archive_sqlite"),
                    }
                },
                data_failure_ledger=build_social_failure_ledger(
                    SocialStatus.FAILED.value, [REASON_SOCIAL_ARCHIVE_MISSING]
                ),
            )

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(collector.collect, ticker, trade_date)
            context = future.result(timeout=timeout)
            if isinstance(context, dict):
                return context
            if hasattr(context, "to_dict"):
                return context.to_dict()
            return dict(context)
        except (TimeoutError, FuturesTimeoutError):
            logger.warning(
                "Social data fetch timed out after %ss for %s on %s",
                timeout,
                ticker,
                trade_date,
            )
            bundle = create_empty_sentiment_bundle(
                status=SocialStatus.TIMEOUT.value,
                requested_as_of=trade_date,
                cutoff_at="",
                reason_codes=[REASON_SOCIAL_ARCHIVE_LOCKED],
                symbol=ticker,
            )
            return create_default_social_data_context(
                status=SocialStatus.TIMEOUT.value,
                mode=str(mode),
                requested_as_of=trade_date,
                reason_codes=[REASON_SOCIAL_ARCHIVE_LOCKED],
                bundle=bundle,
                source_provenance={
                    "social_archive": {
                        "status": SocialStatus.TIMEOUT.value,
                        "provider": getattr(collector, "provider_name", "archive_sqlite"),
                    }
                },
                data_failure_ledger=build_social_failure_ledger(
                    SocialStatus.TIMEOUT.value, [REASON_SOCIAL_ARCHIVE_LOCKED]
                ),
            )
        except Exception as exc:
            logger.warning(
                "Social data collection failed unexpectedly for %s on %s: %s",
                ticker,
                trade_date,
                exc,
            )
            bundle = create_empty_sentiment_bundle(
                status=SocialStatus.FAILED.value,
                requested_as_of=trade_date,
                cutoff_at="",
                reason_codes=[REASON_SOCIAL_ARCHIVE_MISSING],
                symbol=ticker,
            )
            return create_default_social_data_context(
                status=SocialStatus.FAILED.value,
                mode=str(mode),
                requested_as_of=trade_date,
                reason_codes=[REASON_SOCIAL_ARCHIVE_MISSING],
                bundle=bundle,
                source_provenance={
                    "social_archive": {
                        "status": SocialStatus.FAILED.value,
                        "provider": getattr(collector, "provider_name", "archive_sqlite"),
                    }
                },
                data_failure_ledger=build_social_failure_ledger(
                    SocialStatus.FAILED.value, [REASON_SOCIAL_ARCHIVE_MISSING]
                ),
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def collect(
        self,
        ticker: str,
        trade_date: str,
        horizons: Optional[List[str]] = None,
        price_basis: str = PRICE_BASIS_VENDOR_QFQ,
    ) -> Dict[str, Any]:
        """Fetch all data and store in cache.

        Thread-safe: concurrent calls for the same ticker+date will block
        on a per-key lock, so data is fetched only once.
        """
        norm_price_basis = validate_price_basis_short_label(price_basis)
        if norm_price_basis in (
            PRICE_BASIS_PIT_RAW,
            PRICE_BASIS_PIT_ADJUSTED,
            PRICE_BASIS_UNSPECIFIED,
        ):
            raise UnsupportedPriceBasisError(
                f"price_basis={norm_price_basis!r} is not implemented as an available channel in DataCollector"
            )

        key = make_cache_key(ticker, trade_date, price_basis=norm_price_basis)
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
                pool = _fetch_all(
                    ticker,
                    trade_date,
                    industry_provider=self.industry_linkage_provider,
                    price_basis=norm_price_basis,
                )
                pool["social_data_context"] = self._fetch_social_context(
                    ticker, trade_date
                )
                self._cache[key] = pool
            return copy.deepcopy(self._cache[key])
        finally:
            key_lock.release()

    def get(
        self,
        ticker: str,
        trade_date: str,
        price_basis: str = PRICE_BASIS_VENDOR_QFQ,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached pool, or None if not collected yet."""
        key = make_cache_key(ticker, trade_date, price_basis=price_basis)
        cached = self._cache.get(key)
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

    def ref(
        self,
        ticker: str,
        trade_date: str,
        price_basis: str = PRICE_BASIS_VENDOR_QFQ,
    ) -> None:
        """Increment reference count (call before using cached data)."""
        key = make_cache_key(ticker, trade_date, price_basis=price_basis)
        with self._meta_lock:
            self._refcounts[key] = self._refcounts.get(key, 0) + 1

    def evict(
        self,
        ticker: str,
        trade_date: str,
        price_basis: str = PRICE_BASIS_VENDOR_QFQ,
    ) -> None:
        """Decrement refcount and remove cached data when no one needs it."""
        key = make_cache_key(ticker, trade_date, price_basis=price_basis)
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
