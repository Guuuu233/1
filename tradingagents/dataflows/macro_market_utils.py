"""Macro market utilities for calculating indicators, trends, and formatting macro views."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def calculate_series_metrics(
    df: Optional[pd.DataFrame],
    as_of: str,
    price_col: str = "close",
) -> Optional[Dict[str, Any]]:
    """Normalize a time series frame and calculate key returns, MAs, and trend status.

    Strictly filters rows where date <= as_of to prevent lookahead bias.
    Uses column-name based parsing and drops invalid rows.
    """
    if df is None or df.empty:
        return None

    df_work = df.copy()
    # Normalize column names to lowercase
    cols_map = {str(c).lower(): c for c in df_work.columns}

    # Identify date column
    date_col = None
    for cand in ("date", "日期", "trade_date", "datetime", "time", "index"):
        if cand in cols_map:
            date_col = cols_map[cand]
            break
    if date_col is None:
        return None

    # Identify price column
    target_price_col = None
    for cand in (price_col.lower(), "close", "收盘", "收盘价", "price", "val", "rate", "value", "latest"):
        if cand in cols_map:
            target_price_col = cols_map[cand]
            break
    if target_price_col is None:
        return None

    # Rename to standard names
    df_work = df_work.rename(columns={date_col: "date", target_price_col: "close"})
    df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce")
    df_work["close"] = pd.to_numeric(df_work["close"], errors="coerce")
    df_work = df_work.dropna(subset=["date", "close"])

    if df_work.empty:
        return None

    end_dt = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(end_dt):
        return None

    # Anti-lookahead cutoff
    df_work = df_work[df_work["date"] <= end_dt]
    if df_work.empty:
        return None

    df_work = df_work.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if df_work.empty:
        return None

    n = len(df_work)
    last_row = df_work.iloc[-1]
    actual_as_of = last_row["date"].strftime("%Y-%m-%d")
    latest_close = float(last_row["close"])

    prev_close = float(df_work["close"].iloc[-2]) if n >= 2 else latest_close
    change_1d_pct = ((latest_close - prev_close) / prev_close * 100) if prev_close != 0 else 0.0

    close_5d_ago = float(df_work["close"].iloc[-min(6, n)]) if n >= 2 else latest_close
    change_5d_pct = ((latest_close - close_5d_ago) / close_5d_ago * 100) if close_5d_ago != 0 else 0.0

    close_20d_ago = float(df_work["close"].iloc[-min(21, n)]) if n >= 2 else latest_close
    change_20d_pct = ((latest_close - close_20d_ago) / close_20d_ago * 100) if close_20d_ago != 0 else 0.0

    ma5 = float(df_work["close"].tail(5).mean()) if n >= 5 else None
    ma20 = float(df_work["close"].tail(20).mean()) if n >= 20 else None
    ma60 = float(df_work["close"].tail(60).mean()) if n >= 60 else None

    # Determine trend description
    if ma5 is not None and ma20 is not None:
        if ma60 is not None and latest_close > ma5 > ma20 > ma60:
            trend_desc = "多头排列(强势上涨)"
        elif latest_close > ma5 > ma20:
            trend_desc = "短期偏强(均线上行)"
        elif ma60 is not None and latest_close < ma5 < ma20 < ma60:
            trend_desc = "空头排列(弱势下跌)"
        elif latest_close < ma5 < ma20:
            trend_desc = "短期偏弱(均线下行)"
        elif latest_close > ma20:
            trend_desc = "震荡偏强"
        else:
            trend_desc = "震荡整理"
    else:
        if change_1d_pct > 0.5:
            trend_desc = "上涨反弹"
        elif change_1d_pct < -0.5:
            trend_desc = "回调下跌"
        else:
            trend_desc = "平稳震荡"

    return {
        "as_of": actual_as_of,
        "latest_close": latest_close,
        "change_1d_pct": change_1d_pct,
        "change_5d_pct": change_5d_pct,
        "change_20d_pct": change_20d_pct,
        "ma5": ma5,
        "ma20": ma20,
        "ma60": ma60,
        "trend_desc": trend_desc,
        "bars_count": n,
    }


def build_cn_indices_markdown(
    items: Dict[str, Dict[str, Any]],
    requested_as_of: str,
    source: str = "cn_akshare",
) -> str:
    """Build standardized Markdown view for China core indices."""
    if not items:
        return f"【数据获取失败】国内核心大盘指数 — 原因：无有效大盘指数数据 (来源: {source})"

    # Find the maximum actual as_of date among successfully parsed items
    valid_dates = [m["as_of"] for m in items.values() if isinstance(m, dict) and "as_of" in m]
    actual_as_of = max(valid_dates) if valid_dates else requested_as_of

    lines = [
        f"## 国内核心大盘指数行情（数据基准日：{actual_as_of}，来源：{source}）\n",
        f"【数据日期】{actual_as_of}",
        "| 指数名称 | 代码 | 最新点位 | 单日涨跌幅 | 5日涨跌幅 | 20日涨跌幅 | 均线与趋势特征 |",
        "|---|---|---|---|---|---|---|",
    ]

    for name, data in items.items():
        if not isinstance(data, dict) or "latest_close" not in data:
            lines.append(f"| {name} | - | 【数据获取失败】 | - | - | - | - |")
            continue

        code = data.get("code", "-")
        close_val = f"{data['latest_close']:.2f}"
        d1 = f"{data['change_1d_pct']:+.2f}%"
        d5 = f"{data['change_5d_pct']:+.2f}%"
        d20 = f"{data['change_20d_pct']:+.2f}%"
        trend = data.get("trend_desc", "震荡整理")
        lines.append(f"| {name} | {code} | {close_val} | {d1} | {d5} | {d20} | {trend} |")

    lines.append("\n### 市场综合环境与大盘广度分析")
    # Quick synthesis summary
    up_count = sum(1 for m in items.values() if isinstance(m, dict) and m.get("change_1d_pct", 0) > 0)
    total_count = len(items)
    if up_count >= total_count * 0.7:
        breadth_summary = "大盘各主要指数普涨，市场风险偏好处于高位。"
    elif up_count <= total_count * 0.3:
        breadth_summary = "主要大盘指数多数调整，市场情绪偏谨慎防御。"
    else:
        breadth_summary = "大盘各指数分化震荡，结构性行情特征明显。"
    lines.append(f"- **指数联动与广度**: {breadth_summary}（{up_count}/{total_count} 主要指数收涨）")

    return "\n".join(lines)


def build_global_indices_markdown(
    items: Dict[str, Dict[str, Any]],
    requested_as_of: str,
    source: str = "yfinance",
) -> str:
    """Build standardized Markdown view for Global core indices."""
    if not items:
        return f"【数据获取失败】全球核心指数 — 原因：无有效全球指数数据 (来源: {source})"

    valid_items = {
        k: v for k, v in items.items()
        if isinstance(v, dict) and v.get("latest_close") is not None
    }
    if not valid_items:
        return f"【数据获取失败】全球核心指数 — 原因：无有效全球指数数据 (来源: {source})"

    valid_dates = [m["as_of"] for m in valid_items.values() if isinstance(m, dict) and "as_of" in m]
    actual_as_of = max(valid_dates) if valid_dates else requested_as_of

    lines = [
        f"## 全球核心市场指数行情（数据基准日：{actual_as_of}，来源：{source}）\n",
        f"【数据日期】{actual_as_of}",
        "| 市场/指数 | 代码 | 最新收盘 | 单日涨跌幅 | 5日涨跌幅 | 20日涨跌幅 | 趋势状态 |",
        "|---|---|---|---|---|---|---|",
    ]

    for name, data in items.items():
        if not isinstance(data, dict) or data.get("latest_close") is None:
            code = data.get("code", "-") if isinstance(data, dict) else "-"
            lines.append(f"| {name} | {code} | 【数据缺失】 | - | - | - | - |")
            continue

        display_name = name
        code = data.get("code", "-")
        if display_name == "纳斯达克":
            if "100" in str(code) or "NDX" in str(code):
                display_name = "纳斯达克100"
            else:
                display_name = "纳斯达克综合"

        close_val = f"{data['latest_close']:.2f}"
        d1 = f"{data['change_1d_pct']:+.2f}%" if data.get("change_1d_pct") is not None else "【数据缺失】"
        d5 = f"{data['change_5d_pct']:+.2f}%" if data.get("change_5d_pct") is not None else "【数据缺失】"
        d20 = f"{data['change_20d_pct']:+.2f}%" if data.get("change_20d_pct") is not None else "【数据缺失】"
        trend = data.get("trend_desc", "平稳")
        lines.append(f"| {display_name} | {code} | {close_val} | {d1} | {d5} | {d20} | {trend} |")

    lines.append("\n### 跨市场宏观联动观察")
    # US markets check
    sp500 = items.get("标普500") or items.get("S&P 500")
    nasdaq = items.get("纳斯达克综合") or items.get("纳斯达克100") or items.get("纳斯达克") or items.get("Nasdaq")
    dji = items.get("道琼斯") or items.get("DJIA") or items.get("Dow Jones")

    sp_chg = sp500.get("change_1d_pct") if isinstance(sp500, dict) else None
    nasdaq_chg = nasdaq.get("change_1d_pct") if isinstance(nasdaq, dict) else None
    dji_chg = dji.get("change_1d_pct") if isinstance(dji, dict) else None

    nasdaq_label = "纳斯达克100" if (nasdaq and ("100" in str(nasdaq.get("code", "")) or "NDX" in str(nasdaq.get("code", "")))) else "纳斯达克综合"

    if sp_chg is not None and nasdaq_chg is not None:
        if sp_chg > 0 and nasdaq_chg > 0:
            lines.append(f"- **美股外盘氛围**: 标普500 ({sp_chg:+.2f}%) 与{nasdaq_label} ({nasdaq_chg:+.2f}%) 上扬，科技股表现活跃，为全球风险资产提供正面情绪支撑。")
        elif sp_chg < 0 and nasdaq_chg < 0:
            lines.append(f"- **美股外盘氛围**: 美股核心指数（标普500 {sp_chg:+.2f}%、{nasdaq_label} {nasdaq_chg:+.2f}%）走弱调整，需关注全球流动性收紧与外生风险传导。")
        else:
            lines.append(f"- **美股外盘氛围**: 美股价值与成长板块分化（标普500 {sp_chg:+.2f}%，{nasdaq_label} {nasdaq_chg:+.2f}%）。")
    elif sp_chg is not None:
        lines.append(f"- **美股外盘氛围**: 标普500单日涨跌幅为 {sp_chg:+.2f}%。")
    elif nasdaq_chg is not None:
        lines.append(f"- **美股外盘氛围**: {nasdaq_label}单日涨跌幅为 {nasdaq_chg:+.2f}%。")

    # HK markets check
    hsi = items.get("恒生指数") or items.get("Hang Seng")
    hstech = items.get("恒生科技指数") or items.get("恒生科技") or items.get("HSTECH")
    hsi_chg = hsi.get("change_1d_pct") if isinstance(hsi, dict) else None
    hstech_chg = hstech.get("change_1d_pct") if isinstance(hstech, dict) else None

    if hsi_chg is not None and hstech_chg is not None:
        lines.append(f"- **港股联动纽带**: 恒生指数单日涨跌幅为 {hsi_chg:+.2f}%，恒生科技指数为 {hstech_chg:+.2f}%，反映离岸中国资产与海外科技流动性定价反应。")
    elif hsi_chg is not None:
        lines.append(f"- **港股联动纽带**: 恒生指数单日涨跌幅为 {hsi_chg:+.2f}%，反映离岸中国资产对外部宏观及海外利率的定价反应。")
    elif hstech_chg is not None:
        lines.append(f"- **港股联动纽带**: 恒生科技指数单日涨跌幅为 {hstech_chg:+.2f}%。")

    # Asia-Pacific check (Nikkei 225 & KOSPI)
    nikkei = items.get("日经225") or items.get("Nikkei 225") or items.get("日经指数")
    kospi = items.get("韩国KOSPI") or items.get("KOSPI") or items.get("首尔综合指数")
    nikkei_chg = nikkei.get("change_1d_pct") if isinstance(nikkei, dict) else None
    kospi_chg = kospi.get("change_1d_pct") if isinstance(kospi, dict) else None

    if nikkei_chg is not None and kospi_chg is not None:
        lines.append(f"- **亚太市场温度**: 日经225 ({nikkei_chg:+.2f}%) 与韩国KOSPI ({kospi_chg:+.2f}%) 呈现联动，作为亚太半导体供应链与制造业风险温度计。")
    elif nikkei_chg is not None:
        lines.append(f"- **亚太市场温度**: 日经225单日涨跌幅为 {nikkei_chg:+.2f}%，映射亚太区域市场风险偏好。")
    elif kospi_chg is not None:
        lines.append(f"- **亚太市场温度**: 韩国KOSPI单日涨跌幅为 {kospi_chg:+.2f}%，反映韩国科技与出口供应链景气。")

    # Europe check (DAX, FTSE 100, CAC 40)
    dax = items.get("德国DAX") or items.get("德国DAX30") or items.get("DAX")
    ftse = items.get("英国富时100") or items.get("富时100") or items.get("FTSE")
    cac = items.get("法国CAC40") or items.get("CAC40") or items.get("法国CAC")
    dax_chg = dax.get("change_1d_pct") if isinstance(dax, dict) else None
    ftse_chg = ftse.get("change_1d_pct") if isinstance(ftse, dict) else None
    cac_chg = cac.get("change_1d_pct") if isinstance(cac, dict) else None

    euro_parts = []
    if dax_chg is not None:
        euro_parts.append(f"德国DAX {dax_chg:+.2f}%")
    if ftse_chg is not None:
        euro_parts.append(f"英国富时100 {ftse_chg:+.2f}%")
    if cac_chg is not None:
        euro_parts.append(f"法国CAC40 {cac_chg:+.2f}%")

    if euro_parts:
        lines.append(f"- **欧洲外围环境**: 欧洲核心市场（{' / '.join(euro_parts)}）表现，映射欧洲央行政策预期与欧洲经济增长动能。")

    return "\n".join(lines)


def build_major_assets_markdown(
    items: Dict[str, Dict[str, Any]],
    requested_as_of: str,
    source: str = "yfinance",
) -> str:
    """Build standardized Markdown view for Major Macro Assets & Commodities."""
    if not items:
        return f"【数据获取失败】全球大类资产 — 原因：无有效大类资产数据 (来源: {source})"

    valid_dates = [m["as_of"] for m in items.values() if isinstance(m, dict) and "as_of" in m]
    actual_as_of = max(valid_dates) if valid_dates else requested_as_of

    lines = [
        f"## 全球大类资产与宏观大宗商品（数据基准日：{actual_as_of}，来源：{source}）\n",
        f"【数据日期】{actual_as_of}",
        "| 资产类别 | 标的名称 | 代码 | 最新价格/点位 | 单日涨跌幅 | 5日涨跌幅 | 20日涨跌幅 | 宏观传导与风险指示 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for name, data in items.items():
        if not isinstance(data, dict) or "latest_close" not in data:
            lines.append(f"| - | {name} | - | 【数据获取失败】 | - | - | - | - |")
            continue

        category = data.get("category", "大类资产")
        code = data.get("code", "-")
        # Format yield specially if 10Y Treasury
        if "收益率" in name or "%" in str(data.get("unit", "")):
            close_val = f"{data['latest_close']:.3f}%"
            d1 = f"{data['change_1d_pct']:+.2f}%"
        else:
            close_val = f"{data['latest_close']:.2f}"
            d1 = f"{data['change_1d_pct']:+.2f}%"

        d5 = f"{data['change_5d_pct']:+.2f}%"
        d20 = f"{data['change_20d_pct']:+.2f}%"
        signal = data.get("macro_signal", data.get("trend_desc", "宏观中性"))
        lines.append(f"| {category} | {name} | {code} | {close_val} | {d1} | {d5} | {d20} | {signal} |")

    lines.append("\n### 宏观大类资产传导机制与情景评估")
    gold = items.get("COMEX黄金") or items.get("伦敦金") or items.get("黄金")
    oil = items.get("WTI原油") or items.get("布伦特原油") or items.get("原油")
    us10y = items.get("美债10年期收益率") or items.get("US10Y") or items.get("10Y美债")
    dxy = items.get("美元指数") or items.get("DXY")

    if gold and isinstance(gold, dict):
        g_d1 = gold.get("change_1d_pct", 0)
        lines.append(f"- **黄金/贵金属 ({g_d1:+.2f}%)**: 映射全球地缘避险买盘与真实利率预期。")
    if oil and isinstance(oil, dict):
        o_d1 = oil.get("change_1d_pct", 0)
        lines.append(f"- **原油/能源 ({o_d1:+.2f}%)**: 映射全球通胀预期、地缘供给扰动与中下游制造成本。")
    if us10y and isinstance(us10y, dict):
        u_val = us10y.get("latest_close", 0)
        lines.append(f"- **美债10年期收益率 ({u_val:.3f}%)**: 全球无风险资产定价之锚，直接影响成长股估值折现率与资金跨境流动。")
    if dxy and isinstance(dxy, dict):
        d_d1 = dxy.get("change_1d_pct", 0)
        lines.append(f"- **美元指数 ({d_d1:+.2f}%)**: 汇率强弱指标，影响人民币汇率稳定性与北向外资配置意愿。")

    return "\n".join(lines)
