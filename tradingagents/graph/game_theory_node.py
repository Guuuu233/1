# tradingagents/graph/game_theory_node.py
"""Game Theory Analysis Node for TradingAgents (DAV-829 / L2).

Orchestrates multi-agent counterparty game theory analysis across:
1. 主力机构 (Main Force / Institutional Capital)
2. 北向资金 (Northbound / Cross-border Capital)
3. 杠杆资金 (Leveraged Margin Capital)
4. 散户群体 (Retail Investors & Chip Concentration)

Contracts & Disciplines (AGENTS.md §3.4, §3.5, §4, §5):
- Pure deterministic calculations for all numeric signals and strategy derivation.
- No LLM hallucination of numerical indicators (RT-6).
- Explicit unavailability marking for missing/failed sources; never return empty string or fake zeros (RT-2).
- Fail-safe execution: Node exceptions never crash the workflow; leaves traceable logs and audit trails (RT-3).
- Graph reachability: executed cleanly between Research Manager and Trader (RT-4).
- State & persistence: game_theory_report and game_theory_signals written to state, persisted to ReportDB,
  and verified on readback (RT-1, RT-5).
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from typing import Any, Mapping, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph

from tradingagents.agents.utils.agent_states import (
    AgentState,
    GameTheorySignals,
    TraceItem,
    check_llm_output_degraded,
    current_tracker_var,
)
from tradingagents.agents.utils.game_theory_tools import (
    fetch_board_fund_flow,
    fetch_hot_stocks_xq,
    fetch_individual_fund_flow,
    fetch_lhb_detail,
    fetch_margin_trading,
    fetch_northbound_flow,
    fetch_shareholder_count,
    fetch_zt_pool,
)
from tradingagents.prompts import get_prompt

logger = logging.getLogger(__name__)

NODE_NAME: str = "Game Theory"
AGENT_NAME: str = "game_theory_analyst"
REPORT_KEY: str = "game_theory_report"
SIGNALS_KEY: str = "game_theory_signals"

FAILURE_MARKERS: tuple[str, ...] = (
    "【数据获取失败】",
    "未获取",
    "超时",
    "失败",
    "异常",
    "不可用",
    "无数据",
    "停更",
)


def _is_failed_text(val: Any) -> bool:
    """Return True if a data source text indicates failure or unavailability."""
    if val is None:
        return True
    s = str(val).strip()
    if not s or s == "无数据":
        return True
    return any(marker in s for marker in FAILURE_MARKERS)


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return f if f == f else None  # Filter NaN
    except (ValueError, TypeError):
        return None


def _extract_fund_flow_info(
    fund_flow_raw: Any,
    fund_flow_evidence: Any,
    scale_metrics: Any,
) -> dict[str, Any]:
    """Deterministically parse main force fund flow metrics."""
    result: dict[str, Any] = {
        "status": "unavailable",
        "net_amount": None,
        "direction": 0,  # +1 inflow, -1 outflow, 0 neutral
        "direction_label": "中性",
        "description": "资金流数据不可用",
    }

    # 1. Check fund_flow_evidence mapping if present
    if isinstance(fund_flow_evidence, Mapping):
        sel = fund_flow_evidence.get("selection")
        if isinstance(sel, Mapping):
            status = sel.get("selection_status")
            val = _safe_float(sel.get("selected_value"))
            if val is not None:
                result["net_amount"] = val
                result["status"] = "available"
                unit = sel.get("selected_unit", "元")
                if val > 0:
                    result["direction"] = 1
                    result["direction_label"] = "主力净流入"
                    result["description"] = f"主力资金净流入 {val:.2f} {unit}"
                elif val < 0:
                    result["direction"] = -1
                    result["direction_label"] = "主力净流出"
                    result["description"] = f"主力资金净流出 {abs(val):.2f} {unit}"
                else:
                    result["direction"] = 0
                    result["direction_label"] = "主力中性"
                    result["description"] = f"主力资金净额平衡 (0 {unit})"
                return result

    # 2. Check scale_metrics
    if isinstance(scale_metrics, Mapping):
        net_val = _safe_float(scale_metrics.get("net_amount"))
        if net_val is not None:
            result["net_amount"] = net_val
            result["status"] = "available"
            if net_val > 0:
                result["direction"] = 1
                result["direction_label"] = "主力净流入"
                result["description"] = f"主力资金净流入 {net_val:.2f} 万元"
            elif net_val < 0:
                result["direction"] = -1
                result["direction_label"] = "主力净流出"
                result["description"] = f"主力资金净流出 {abs(net_val):.2f} 万元"
            else:
                result["direction"] = 0
                result["direction_label"] = "主力中性"
                result["description"] = "主力资金净额平衡 (0 万元)"
            return result

    # 3. Fallback: Parse raw string from individual fund flow
    if fund_flow_raw and not _is_failed_text(fund_flow_raw):
        text = str(fund_flow_raw)
        # Match e.g. "净额: 1234.56" or "净流入: -567.89"
        m = re.search(r'(?:净流入|净额)[^\d\-+]*([+\-]?\d+(?:\.\d+)?)', text)
        if m:
            val = _safe_float(m.group(1))
            if val is not None:
                result["net_amount"] = val
                result["status"] = "available"
                if val > 0:
                    result["direction"] = 1
                    result["direction_label"] = "主力净流入"
                    result["description"] = f"主力净额呈现流入 ({val:.2f})"
                elif val < 0:
                    result["direction"] = -1
                    result["direction_label"] = "主力净流出"
                    result["description"] = f"主力净额呈现流出 ({val:.2f})"
                else:
                    result["direction"] = 0
                    result["direction_label"] = "主力中性"
                    result["description"] = "主力净额基本持平"
                return result

        result["status"] = "partial"
        result["description"] = text[:120].strip()
        return result

    result["description"] = "【数据获取失败】个股主力资金流向数据不可用"
    return result


def _extract_margin_info(margin_raw: Any) -> dict[str, Any]:
    """Deterministically parse margin trading (leveraged capital) metrics."""
    result: dict[str, Any] = {
        "status": "unavailable",
        "direction": 0,
        "direction_label": "中性",
        "description": "融资融券数据不可用",
    }
    if not margin_raw or _is_failed_text(margin_raw):
        result["description"] = "【数据获取失败】融资融券明细数据不可用"
        return result

    text = str(margin_raw)
    result["status"] = "available"

    # Match 融资买入额 vs 融资偿还额
    buy_m = re.search(r'融资买入额[^\d]*(\d+(?:\.\d+)?)', text)
    repay_m = re.search(r'融资偿还额[^\d]*(\d+(?:\.\d+)?)', text)
    balance_m = re.search(r'融资余额[^\d]*(\d+(?:\.\d+)?)', text)

    buy_amt = _safe_float(buy_m.group(1)) if buy_m else None
    repay_amt = _safe_float(repay_m.group(1)) if repay_m else None
    bal_amt = _safe_float(balance_m.group(1)) if balance_m else None

    if buy_amt is not None and repay_amt is not None:
        diff = buy_amt - repay_amt
        if diff > 0:
            result["direction"] = 1
            result["direction_label"] = "杠杆做多"
            result["description"] = f"融资净买入 {diff / 1e4:.2f} 万元（做多意愿积极）"
        elif diff < 0:
            result["direction"] = -1
            result["direction_label"] = "杠杆去化"
            result["description"] = f"融资净偿还 {abs(diff) / 1e4:.2f} 万元（去杠杆避险）"
        else:
            result["direction"] = 0
            result["direction_label"] = "杠杆平衡"
            result["description"] = "融资买入与偿还额基本均衡"
    elif bal_amt is not None:
        result["direction"] = 0
        result["direction_label"] = "杠杆平稳"
        result["description"] = f"融资余额 {bal_amt / 1e8:.2f} 亿元"
    else:
        result["direction"] = 0
        result["description"] = text[:100].strip()

    return result


def _extract_shareholder_info(shareholder_raw: Any) -> dict[str, Any]:
    """Deterministically parse shareholder count and chip concentration (retail)."""
    result: dict[str, Any] = {
        "status": "unavailable",
        "direction": 0,
        "direction_label": "中性",
        "description": "股东户数数据不可用",
    }
    if not shareholder_raw or _is_failed_text(shareholder_raw):
        result["description"] = "【数据获取失败】股东户数与筹码集中度数据不可用"
        return result

    text = str(shareholder_raw)
    result["status"] = "available"

    # Match change percent: 较上期变动: -3.5%
    m = re.search(r'较上期变动[^\d\-+]*([+\-]?\d+(?:\.\d+)?)%?', text)
    count_m = re.search(r'股东户数[^\d]*(\d+)', text)

    chg_pct = _safe_float(m.group(1)) if m else None
    count_val = _safe_float(count_m.group(1)) if count_m else None

    if chg_pct is not None:
        if chg_pct < -1.0:
            result["direction"] = 1  # Chip concentration is bullish
            result["direction_label"] = "筹码集中"
            result["description"] = f"股东户数较上期减少 {abs(chg_pct):.2f}%（筹码持续集中，散户离场/主力锁仓）"
        elif chg_pct > 1.0:
            result["direction"] = -1  # Chip dispersion is bearish
            result["direction_label"] = "筹码分散"
            result["description"] = f"股东户数较上期增加 {chg_pct:.2f}%（筹码趋向分散，散户涌入/主力派发）"
        else:
            result["direction"] = 0
            result["direction_label"] = "筹码稳定"
            result["description"] = f"股东户数较上期微幅变动 {chg_pct:.2f}%（筹码分布相对稳定）"
    elif count_val is not None:
        result["direction"] = 0
        result["direction_label"] = "筹码平稳"
        result["description"] = f"最新披露股东户数 {int(count_val)} 户"
    else:
        result["direction"] = 0
        result["description"] = text[:100].strip()

    return result


def _extract_northbound_info(northbound_raw: Any) -> dict[str, Any]:
    """Deterministically parse northbound foreign capital metrics."""
    result: dict[str, Any] = {
        "status": "unavailable",
        "direction": 0,
        "direction_label": "中性",
        "description": "北向资金数据不可用",
    }
    if not northbound_raw or _is_failed_text(northbound_raw):
        if northbound_raw and "自 2024 年 8 月起停止披露" in str(northbound_raw):
            result["description"] = "【数据制度性停更】沪深港通个股每日持股明细自2024年8月起停止披露，本项不可用。"
        else:
            result["description"] = "【数据获取失败】北向资金持股变动数据不可用"
        return result

    text = str(northbound_raw)
    result["status"] = "available"
    if "增加" in text or "净买入" in text or "增持" in text:
        result["direction"] = 1
        result["direction_label"] = "北向增持"
        result["description"] = "北向资金持股呈净增持态势"
    elif "减少" in text or "净卖出" in text or "减持" in text:
        result["direction"] = -1
        result["direction_label"] = "北向减持"
        result["description"] = "北向资金持股呈净减持态势"
    else:
        result["direction"] = 0
        result["direction_label"] = "北向平稳"
        result["description"] = text[:100].strip()

    return result


def _extract_board_flow_info(board_raw: Any) -> dict[str, Any]:
    """Deterministically parse sector/board fund flow context."""
    if not board_raw or _is_failed_text(board_raw):
        return {
            "status": "unavailable",
            "board": "板块资金流向数据不可用",
            "description": "【数据获取失败】行业板块资金流向数据不可用",
        }
    text = str(board_raw).strip()
    return {
        "status": "available",
        "board": text[:120].strip(),
        "description": text[:200].strip(),
    }


def compute_game_theory_signals(
    ticker: str,
    trade_date: str,
    raw_data: Mapping[str, Any],
    consensus_direction: Optional[str] = None,
) -> tuple[str, GameTheorySignals]:
    """Pure deterministic computation of game theory signals and markdown report.

    Strict zero hallucination (RT-6): all values, player states, dominant strategies,
    and equilibrium evaluations are determined in Python from raw evidence.
    """
    fund_flow_raw = raw_data.get("fund_flow_individual")
    fund_flow_evidence = raw_data.get("fund_flow_evidence")
    scale_metrics = raw_data.get("scale_metrics")
    margin_raw = raw_data.get("margin_trading")
    shareholder_raw = raw_data.get("shareholder_count")
    northbound_raw = raw_data.get("northbound_flow")
    board_raw = raw_data.get("fund_flow_board")
    lhb_raw = raw_data.get("lhb")

    # 1. Parse individual players
    ff_info = _extract_fund_flow_info(fund_flow_raw, fund_flow_evidence, scale_metrics)
    margin_info = _extract_margin_info(margin_raw)
    sh_info = _extract_shareholder_info(shareholder_raw)
    nb_info = _extract_northbound_info(northbound_raw)
    board_info = _extract_board_flow_info(board_raw)

    # 2. Track availability
    dimensions = {
        "主力资金": ff_info["status"],
        "杠杆资金": margin_info["status"],
        "散户筹码": sh_info["status"],
        "北向资金": nb_info["status"],
        "行业板块": board_info["status"],
    }
    available_count = sum(1 for st in dimensions.values() if st == "available")
    total_count = len(dimensions)

    # RT-2: All opponent data missing/failed -> fail closed
    if available_count == 0:
        report_text = (
            f"## 博弈论与对手盘分析报告（{ticker} | {trade_date}）\n\n"
            "【数据获取失败】博弈论对手方数据源（个股主力资金流、融资融券、股东户数、北向资金等）全部缺失或不可用。"
            "根据 AGENTS.md §3.4 规范，本项显式标注不可用，严禁伪造默认值，不得据此判断无博弈风险。\n\n"
            "<!-- VERDICT: {\"direction\": \"中性\", \"confidence\": \"低\", \"reason\": \"对手方数据全部缺失\"} -->"
        )
        signals: GameTheorySignals = {
            "board": "板块数据不可用",
            "players": ["主力机构", "北向资金", "杠杆资金", "散户群体"],
            "player_states": {k: "数据不可用" for k in ["主力机构", "北向资金", "杠杆资金", "散户群体"]},
            "likely_actions": {k: ["动作未知（数据缺失）"] for k in ["主力机构", "北向资金", "杠杆资金", "散户群体"]},
            "dominant_strategy": "数据缺失/保持观望",
            "fragile_equilibrium": "对手方数据缺失，无法判定博弈均衡状态。",
            "counter_consensus_signal": "数据不可用，无反共识信号",
            "confidence": 0.0,
            "data_status": "unavailable",
        }
        return report_text, signals

    data_status = "available" if available_count == total_count else "partial"
    confidence_score = round(available_count / total_count, 2)

    # 3. Assemble player states and likely actions
    player_states: dict[str, str] = {
        "主力机构": ff_info["direction_label"],
        "北向资金": nb_info["direction_label"],
        "杠杆资金": margin_info["direction_label"],
        "散户群体": sh_info["direction_label"],
    }

    likely_actions: dict[str, list[str]] = {}
    if ff_info["direction"] == 1:
        likely_actions["主力机构"] = ["分批拉升吸筹", "高位震荡洗盘"]
    elif ff_info["direction"] == -1:
        likely_actions["主力机构"] = ["压单分步派发", "逢反弹减仓"]
    else:
        likely_actions["主力机构"] = ["存量观望", "按兵不动"]

    if nb_info["direction"] == 1:
        likely_actions["北向资金"] = ["核心资产配置", "顺势增持"]
    elif nb_info["direction"] == -1:
        likely_actions["北向资金"] = ["流动性套现", "逢高流出"]
    else:
        likely_actions["北向资金"] = ["配置平稳或停更"]

    if margin_info["direction"] == 1:
        likely_actions["杠杆资金"] = ["利用融资杠杆追逐弹性", "顺势做多"]
    elif margin_info["direction"] == -1:
        likely_actions["杠杆资金"] = ["被动或主动降低融资负债", "防守避险"]
    else:
        likely_actions["杠杆资金"] = ["杠杆仓位保持稳定"]

    if sh_info["direction"] == 1:
        likely_actions["散户群体"] = ["筹码集中持仓", "浮筹清洗完毕"]
    elif sh_info["direction"] == -1:
        likely_actions["散户群体"] = ["跟风买入", "追涨追高", "高位被动承接"]
    else:
        likely_actions["散户群体"] = ["观望等待"]

    # 4. Deterministic Dominant Strategy derivation
    main_dir = ff_info["direction"]
    sh_dir = sh_info["direction"]
    margin_dir = margin_info["direction"]

    if main_dir == 1 and sh_dir >= 0:
        dominant_strategy = "顺势进攻：主力资金净流入且筹码集中度良好，多头占优，建议跟随主力做多"
        overall_dir = "偏多"
    elif main_dir == 1 and sh_dir == -1:
        dominant_strategy = "分歧拉升：主力资金净买入但散户筹码分散，短线急拉后警惕获利盘兑现，建议顺势波段交易"
        overall_dir = "偏多"
    elif main_dir == -1 and sh_dir <= 0:
        dominant_strategy = "严格防守：主力资金净流出且筹码分散，严禁盲目抄底接盘，建议观望或逢反弹减磅"
        overall_dir = "偏空"
    elif main_dir == -1 and sh_dir == 1:
        dominant_strategy = "防御观察：主力微幅兑现但底仓筹码集中度未破，观察关键技术均线与支撑有效性"
        overall_dir = "中性"
    else:
        dominant_strategy = "中性博弈：多空对手盘分歧明显，未见明确主导方，建议保持仓位克制与观望"
        overall_dir = "中性"

    # 5. Deterministic Fragile Equilibrium derivation
    if main_dir == -1 and (margin_dir == 1 or sh_dir == -1):
        fragile_equilibrium = (
            "极端脆弱平衡：主力资金净流出派发，全靠散户跟风或杠杆资金承接，"
            "一旦增量资金边际衰竭极易引发踩踏与多杀多急跌。"
        )
    elif main_dir == 1 and (margin_dir == 1 or nb_info["direction"] == 1):
        fragile_equilibrium = (
            "稳固多头平衡：主力资金与杠杆/机构合力共振做多，空方抛压已被充分吸收，"
            "多头掌握绝对定价权。"
        )
    elif main_dir != 0 and margin_dir != 0 and main_dir != margin_dir:
        fragile_equilibrium = (
            "多空分歧弱平衡：主力资金与杠杆资金方向发生背离，各方力量对峙，"
            "短期将在筹码密集区反复拉锯试探。"
        )
    else:
        fragile_equilibrium = "多空态势相对平稳，未见极端失衡或脆弱多杀多形态。"

    # 6. Deterministic Counter-Consensus Signal derivation
    norm_consensus = str(consensus_direction or "").strip().upper()
    is_bull_consensus = any(w in norm_consensus for w in ("BUY", "BULL", "看多", "偏多"))
    is_bear_consensus = any(w in norm_consensus for w in ("SELL", "BEAR", "看空", "偏空"))

    if is_bull_consensus and main_dir == -1:
        counter_consensus_signal = (
            "反共识预警：研究观点偏多，但主力资金与对手盘出现确定性净流出派发，"
            "警惕'诱多出货/假突破'陷阱。"
        )
    elif is_bear_consensus and main_dir == 1:
        counter_consensus_signal = (
            "反共识信号：市场预期偏空，但核心主力资金逆势吸筹，存在潜在底背离反转机会。"
        )
    else:
        counter_consensus_signal = "无显著反共识信号：资金博弈动向与研判逻辑基本一致。"

    signals: GameTheorySignals = {
        "board": board_info["board"],
        "players": ["主力机构", "北向资金", "杠杆资金", "散户群体"],
        "player_states": player_states,
        "likely_actions": likely_actions,
        "dominant_strategy": dominant_strategy,
        "fragile_equilibrium": fragile_equilibrium,
        "counter_consensus_signal": counter_consensus_signal,
        "confidence": confidence_score,
        "data_status": data_status,
    }

    # 7. Construct Markdown report
    conf_label = "高" if confidence_score >= 0.8 else ("中" if confidence_score >= 0.4 else "低")
    lhb_snippet = ""
    if lhb_raw and not _is_failed_text(lhb_raw):
        lhb_snippet = f"- **龙虎榜席位事实**：{str(lhb_raw)[:150].strip()}\n"

    report_text = (
        f"## 博弈论与对手盘分析报告（{ticker} | {trade_date}）\n\n"
        f"**数据覆盖度与置信度**：{available_count}/{total_count} 项可用（置信度: {confidence_score:.2f} | {conf_label}）\n\n"
        f"### 1. 市场参与主体画像与立场解构\n"
        f"- **主力机构**：{ff_info['description']}\n"
        f"{lhb_snippet}"
        f"- **北向资金**：{nb_info['description']}\n"
        f"- **杠杆资金**：{margin_info['description']}\n"
        f"- **散户群体**：{sh_info['description']}\n"
        f"- **行业板块**：{board_info['description']}\n\n"
        f"### 2. 筹码与流动性博弈矩阵\n"
        f"| 参与主体 | 博弈立场 | 概率动作推演 |\n"
        f"|---|---|---|\n"
        f"| 主力机构 | {player_states['主力机构']} | {'、'.join(likely_actions['主力机构'])} |\n"
        f"| 北向资金 | {player_states['北向资金']} | {'、'.join(likely_actions['北向资金'])} |\n"
        f"| 杠杆资金 | {player_states['杠杆资金']} | {'、'.join(likely_actions['杠杆资金'])} |\n"
        f"| 散户群体 | {player_states['散户群体']} | {'、'.join(likely_actions['散户群体'])} |\n\n"
        f"### 3. 占优策略与脆弱平衡研判\n"
        f"- **占优策略推导**：{dominant_strategy}\n"
        f"- **博弈平衡状态**：{fragile_equilibrium}\n\n"
        f"### 4. 反共识预警与交易执行指引\n"
        f"- **反共识预警**：{counter_consensus_signal}\n"
        f"- **策略执行含义**：基于博弈矩阵，方向判定为【{overall_dir}】，需严格执行仓位约束与动态防守。\n\n"
        f"<!-- VERDICT: {{\"direction\": \"{overall_dir}\", \"confidence\": \"{conf_label}\", \"reason\": \"{dominant_strategy[:20]}\"}} -->"
    )

    return report_text, signals


def _gather_game_theory_raw_data(
    ticker: str,
    trade_date: str,
    state: Mapping[str, Any],
    data_collector: Any,
) -> dict[str, Any]:
    """Collect data from pool or fallback functions with non-blocking error handling."""
    pool: Optional[dict[str, Any]] = None
    if data_collector is not None and hasattr(data_collector, "get"):
        try:
            pool = data_collector.get(ticker, trade_date)
        except Exception as exc:
            logger.warning("[GameTheoryNode] data_collector.get failed for %s: %s", ticker, exc)

    market_data_ctx = state.get("market_data_context") or {}
    if not isinstance(market_data_ctx, Mapping):
        market_data_ctx = {}

    pool_market_ctx = pool.get("market_data_context") if isinstance(pool, dict) else None
    if not isinstance(pool_market_ctx, Mapping):
        pool_market_ctx = market_data_ctx

    # Extract or fallback for each item
    def _fetch_or_fallback(key: str, fallback_fn, *args, **kwargs) -> Any:
        if pool and key in pool and pool[key] is not None:
            return pool[key]
        try:
            return fallback_fn(*args, **kwargs)
        except Exception as exc:
            return f"【数据获取失败】{key} — 原因：{exc}。该项不可用。"

    raw_data: dict[str, Any] = {
        "fund_flow_evidence": pool_market_ctx.get("fund_flow_evidence") or market_data_ctx.get("fund_flow_evidence"),
        "scale_metrics": pool_market_ctx.get("scale_metrics") or market_data_ctx.get("scale_metrics"),
        "fund_flow_individual": _fetch_or_fallback("fund_flow_individual", fetch_individual_fund_flow, ticker, trade_date),
        "fund_flow_board": _fetch_or_fallback("fund_flow_board", fetch_board_fund_flow, trade_date),
        "lhb": _fetch_or_fallback("lhb", fetch_lhb_detail, ticker, trade_date),
        "margin_trading": _fetch_or_fallback("margin_trading", fetch_margin_trading, ticker, curr_date=trade_date),
        "northbound_flow": _fetch_or_fallback("northbound_flow", fetch_northbound_flow, ticker, curr_date=trade_date),
        "shareholder_count": _fetch_or_fallback("shareholder_count", fetch_shareholder_count, ticker, curr_date=trade_date),
        "zt_pool": _fetch_or_fallback("zt_pool", fetch_zt_pool, trade_date),
        "hot_stocks": _fetch_or_fallback("hot_stocks", fetch_hot_stocks_xq, curr_date=trade_date),
    }
    return raw_data


def create_game_theory_node(
    llm: Any = None,
    data_collector: Any = None,
) -> RunnableLambda:
    """Create the Game Theory LangGraph node runnable with sync & async compatibility."""

    def _execute_node(state: AgentState) -> dict[str, Any]:
        """Core node execution logic with full exception isolation (RT-3)."""
        ticker = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")
        horizon = state.get("horizon") or "short"

        logger.info("[GameTheoryNode] START game theory analysis for %s on %s (%s)", ticker, trade_date, horizon)

        try:
            # 1. Gather raw data
            raw_data = _gather_game_theory_raw_data(ticker, trade_date, state, data_collector)

            # 2. Extract consensus from upstream manager verdict or investment plan
            mgr_verdict = state.get("manager_verdict")
            consensus_dir = None
            if isinstance(mgr_verdict, Mapping):
                consensus_dir = mgr_verdict.get("direction")
            if not consensus_dir:
                plan = state.get("investment_plan", "")
                if "BUY" in plan or "买入" in plan or "看多" in plan:
                    consensus_dir = "BUY"
                elif "SELL" in plan or "卖出" in plan or "看空" in plan:
                    consensus_dir = "SELL"

            # 3. Deterministic calculation (RT-6)
            report_text, signals = compute_game_theory_signals(
                ticker=ticker,
                trade_date=trade_date,
                raw_data=raw_data,
                consensus_direction=consensus_dir,
            )

            # 4. Extract verdict for trace
            m = re.search(r'<!--\s*VERDICT:\s*(\{.*?\})\s*-->', report_text, re.DOTALL)
            verdict_dir = "中性"
            confidence_str = "中"
            if m:
                try:
                    vd = json.loads(m.group(1))
                    verdict_dir = vd.get("direction", "中性")
                    confidence_str = vd.get("confidence", "中")
                except Exception:
                    pass

            # 5. Build TraceItem for analyst_traces (RT-4 audit trail)
            source_status = signals.get("data_status") or "available"
            trace_item: TraceItem = {
                "agent": AGENT_NAME,
                "horizon": horizon,
                "data_window": "短期博弈",
                "key_finding": str(signals.get("dominant_strategy") or "")[:80],
                "verdict": verdict_dir,
                "confidence": confidence_str,
                "source_status": source_status,
                "source_mode": "deterministic_game_theory",
                "bundle_id": "game_theory_v1",
                "direction_allowed": (source_status != "unavailable"),
                "reason_codes": [signals.get("dominant_strategy", "")[:30]],
                "evidence_refs": [k for k, v in raw_data.items() if not _is_failed_text(v)],
                "financial_period_compliance": {},
            }

            # 6. Update progress tracker if attached
            tracker = current_tracker_var.get()
            if tracker is not None:
                if hasattr(tracker, "report_sections") and isinstance(tracker.report_sections, dict):
                    tracker.report_sections[REPORT_KEY] = report_text
                if hasattr(tracker, "_emit_report_chunked"):
                    try:
                        tracker._emit_report_chunked(tracker.job_id, REPORT_KEY, report_text)
                    except Exception as t_err:
                        logger.debug("[GameTheoryNode] tracker emit failed: %s", t_err)

            logger.info(
                "[GameTheoryNode] COMPLETED analysis for %s: dominant_strategy=%s, confidence=%.2f",
                ticker,
                signals.get("dominant_strategy", "")[:30],
                signals.get("confidence", 0.0),
            )

            return {
                REPORT_KEY: report_text,
                SIGNALS_KEY: signals,
                "analyst_traces": [trace_item],
            }

        except Exception as exc:
            # RT-3: Never crash the graph on node failure; leave explicit gap and traceable record
            logger.error("[GameTheoryNode] Execution failed: %s", exc, exc_info=True)
            fail_msg = f"【数据获取失败】博弈论分析节点异常 — 原因：{type(exc).__name__}: {exc}。本项不可用。"
            fail_signals: GameTheorySignals = {
                "board": "节点异常不可用",
                "players": ["主力机构", "北向资金", "杠杆资金", "散户群体"],
                "player_states": {k: "节点异常不可用" for k in ["主力机构", "北向资金", "杠杆资金", "散户群体"]},
                "likely_actions": {k: ["节点异常不可用"] for k in ["主力机构", "北向资金", "杠杆资金", "散户群体"]},
                "dominant_strategy": "节点异常/不可用",
                "fragile_equilibrium": "节点异常，无法判定博弈均衡状态。",
                "counter_consensus_signal": "节点异常，无反共识信号",
                "confidence": 0.0,
                "data_status": "unavailable",
            }
            fail_trace: TraceItem = {
                "agent": AGENT_NAME,
                "horizon": horizon,
                "data_window": "短期博弈",
                "key_finding": f"博弈论节点异常: {type(exc).__name__}",
                "verdict": "中性",
                "confidence": "低",
                "source_status": "failed",
                "source_mode": "deterministic_game_theory",
                "bundle_id": "game_theory_v1",
                "direction_allowed": False,
                "reason_codes": [f"node_failure_{type(exc).__name__}"],
                "evidence_refs": [],
                "financial_period_compliance": {},
            }
            return {
                REPORT_KEY: fail_msg,
                SIGNALS_KEY: fail_signals,
                "analyst_traces": [fail_trace],
            }

    async def _async_node(state: AgentState) -> dict[str, Any]:
        # Run execution logic directly in async loop
        return await asyncio.to_thread(_execute_node, state)

    return RunnableLambda(_execute_node, afunc=_async_node)


def wire_game_theory_node(
    workflow: StateGraph,
    llm: Any = None,
    data_collector: Any = None,
) -> None:
    """Wire the Game Theory node cleanly between Research Manager and Trader in a StateGraph.

    Ensures zero topology breakage:
    Replaces ('Research Manager', 'Trader') with:
    ('Research Manager', 'Game Theory') -> ('Game Theory', 'Trader').
    """
    if NODE_NAME in workflow.nodes:
        return

    node_runnable = create_game_theory_node(llm=llm, data_collector=data_collector)
    workflow.add_node(NODE_NAME, node_runnable)

    # Check if edge ('Research Manager', 'Trader') exists and rewire
    if hasattr(workflow, "edges"):
        edge_pair = ("Research Manager", "Trader")
        if edge_pair in workflow.edges:
            workflow.edges.remove(edge_pair)
            workflow.add_edge("Research Manager", NODE_NAME)
            workflow.add_edge(NODE_NAME, "Trader")
            logger.info("[GameTheoryNode] Successfully wired between Research Manager and Trader")
            return

    # Fallback wiring if Research Manager or Trader edge wasn't found directly
    workflow.add_edge(NODE_NAME, "Trader")
