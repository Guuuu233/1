# tradingagents/graph/game_theory_node.py
"""Game Theory Analysis Node for TradingAgents (DAV-829 / L2).

Orchestrates multi-agent counterparty game theory analysis across:
1. 主力机构 (Main Force / Institutional Capital)
2. 北向资金 (Northbound / Cross-border Capital)
3. 杠杆资金 (Leveraged Margin Capital)
4. 散户群体 (Retail Investors & Chip Concentration)

Contracts & Disciplines:
- AGENTS.md §3.4, §3.5, §4, §5: Explicit unavailability marking; no fabricated metrics.
- RT-6: Pure deterministic calculations for all numeric signals and strategy derivation; no LLM hallucination.
- RT-8: Atomicity & semantic consistency:
  - When degraded/failed: {"game_theory_report": "【博弈论分析不可用】原因：...", "game_theory_signals": None}
  - When successful: report and signals both complete, never mismatched.
  - Strict JSON safety: no NaN, Inf, -Inf, numpy scalars or ndarrays.
- RT-9: Historical backtest mode & asset-specific normal absence of data:
  - Distinguishes service failures from normal business states (non-margin stocks, non-connect stocks, non-abnormal days).
  - Snapshot tools in historical dates identified as rule-based unavailability, not service outages.
  - Remaining available indicators calculated normally without throwing.
- RT-3 / RT-4: Safe execution (never crashes graph) and audit traceability via analyst_traces.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import re
from typing import Any, Mapping, Optional

from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph

from tradingagents.agents.utils.agent_states import (
    AgentState,
    GameTheorySignals,
    TraceItem,
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
from tradingagents.dataflows.trade_calendar import (
    SNAPSHOT_ONLY_REFUSAL,
    is_historical_analysis_date,
)

logger = logging.getLogger(__name__)

NODE_NAME: str = "Game Theory"
AGENT_NAME: str = "game_theory_analyst"
REPORT_KEY: str = "game_theory_report"
SIGNALS_KEY: str = "game_theory_signals"

# Service failure indicators (server error, timeout, crash)
SERVICE_FAILURE_MARKERS: tuple[str, ...] = (
    "【数据获取失败】",
    "接口超时",
    "请求超时",
    "调用失败",
    "服务异常",
    "接口异常",
    "网络错误",
    "ConnectionError",
    "TimeoutError",
    "HTTP 500",
    "HTTP 502",
    "HTTP 504",
)

# Normal business absence indicators (e.g. non-margin, non-connect, non-abnormal day, snapshot limitation)
NORMAL_ABSENCE_MARKERS: tuple[str, ...] = (
    "非两融标的",
    "未纳入融资融券",
    "暂无两融",
    "无融资融券",
    "非陆股通标的",
    "未纳入陆股通",
    "自 2024 年 8 月起停止披露",
    "非异动日",
    "未上榜",
    "无龙虎榜",
    "无上榜记录",
    "该数据源仅提供当前快照",
    "无法用于历史日期分析",
)


def ensure_json_safe(obj: Any) -> Any:
    """Recursively sanitize objects to guarantee strict JSON safety (RT-8).

    - NaN / Inf / -Inf converted to None
    - numpy scalars (float64, int64, bool_) converted to native Python types
    - numpy ndarray converted to list
    - dict keys converted to str
    - Validated with json.dumps(..., allow_nan=False)
    """
    if obj is None:
        return None

    # Check numpy types without hard dependency
    type_name = type(obj).__name__
    module_name = type(obj).__module__

    if "numpy" in module_name:
        if hasattr(obj, "shape") and getattr(obj, "shape") != ():
            return [ensure_json_safe(i) for i in obj.tolist()]
        elif hasattr(obj, "item"):
            try:
                obj = obj.item()
            except Exception:
                pass

    if isinstance(obj, (float,)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): ensure_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [ensure_json_safe(i) for i in obj]

    return str(obj)


def _is_service_failure(val: Any) -> bool:
    """Return True if text indicates an unexpected external service failure or timeout."""
    if val is None:
        return False
    s = str(val).strip()
    return any(marker in s for marker in SERVICE_FAILURE_MARKERS)


def _is_normal_absence(val: Any) -> bool:
    """Return True if data indicates a normal business absence (non-margin, non-connect, non-LHB, etc.)."""
    if val is None:
        return True
    s = str(val).strip()
    if not s or s == "无数据" or s == "[]" or s == "{}":
        return True
    return any(marker in s for marker in NORMAL_ABSENCE_MARKERS)


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return f if (f == f and not math.isinf(f)) else None
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
        "direction": 0,
        "direction_label": "数据不可用",
        "description": "个股主力资金流数据不可用",
    }

    # 1. fund_flow_evidence mapping
    if isinstance(fund_flow_evidence, Mapping):
        sel = fund_flow_evidence.get("selection")
        if isinstance(sel, Mapping):
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

    # 2. scale_metrics mapping
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
                result["description"] = "主力资金净额基本持平"
            return result

    # 3. Raw individual fund flow text
    if fund_flow_raw and not _is_service_failure(fund_flow_raw):
        text = str(fund_flow_raw)
        m = re.search(r'(?:净流入|净额)[^\d\-+]*([+\-]?\d+(?:\.\d+)?)', text)
        if m:
            val = _safe_float(m.group(1))
            if val is not None:
                result["net_amount"] = val
                result["status"] = "available"
                if val > 0:
                    result["direction"] = 1
                    result["direction_label"] = "主力净流入"
                    result["description"] = f"主力资金净流入 ({val:.2f})"
                elif val < 0:
                    result["direction"] = -1
                    result["direction_label"] = "主力净流出"
                    result["description"] = f"主力资金净流出 ({val:.2f})"
                else:
                    result["direction"] = 0
                    result["direction_label"] = "主力中性"
                    result["description"] = "主力资金净额基本平衡"
                return result
        if len(text) > 10:
            result["status"] = "partial"
            result["description"] = text[:120].strip()
            return result

    if _is_service_failure(fund_flow_raw):
        result["status"] = "failed"
        result["description"] = f"【数据获取失败】主力资金流接口异常：{str(fund_flow_raw)[:80]}"
    else:
        result["description"] = "【数据缺失】个股主力资金流数据未提供，本项不可用"
    return result


def _extract_margin_info(margin_raw: Any) -> dict[str, Any]:
    """Deterministically parse margin trading (leveraged capital) metrics (RT-9)."""
    result: dict[str, Any] = {
        "status": "unavailable",
        "direction": 0,
        "direction_label": "数据不可用",
        "description": "融资融券数据不可用",
    }

    # Case 1: Service failure
    if _is_service_failure(margin_raw):
        result["status"] = "failed"
        result["direction_label"] = "数据获取失败"
        result["description"] = "【数据获取失败】融资融券明细接口超时或服务异常，该项不可用。"
        return result

    # Case 2: Normal business absence (RT-9: 非两融标的)
    if _is_normal_absence(margin_raw):
        result["status"] = "not_applicable_normal"
        result["direction"] = 0
        result["direction_label"] = "非两融标的"
        result["description"] = "标的未纳入融资融券标的范围（非两融标的），无杠杆资金数据属正常业务状态。"
        return result

    text = str(margin_raw)
    result["status"] = "available"

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
        "direction_label": "数据未披露",
        "description": "股东户数数据不可用",
    }

    if _is_service_failure(shareholder_raw):
        result["status"] = "failed"
        result["direction_label"] = "数据获取失败"
        result["description"] = "【数据获取失败】股东户数与筹码集中度接口异常或超时，该项不可用。"
        return result

    if not shareholder_raw or str(shareholder_raw).strip() in ("", "无数据", "None"):
        result["status"] = "unavailable"
        result["direction_label"] = "数据未披露"
        result["description"] = "【数据缺失】股东户数未披露，该项不可用。"
        return result

    text = str(shareholder_raw)
    result["status"] = "available"

    m = re.search(r'较上期变动[^\d\-+]*([+\-]?\d+(?:\.\d+)?)%?', text)
    count_m = re.search(r'股东户数[^\d]*(\d+)', text)

    chg_pct = _safe_float(m.group(1)) if m else None
    count_val = _safe_float(count_m.group(1)) if count_m else None

    if chg_pct is not None:
        if chg_pct < -1.0:
            result["direction"] = 1
            result["direction_label"] = "筹码集中"
            result["description"] = f"股东户数较上期减少 {abs(chg_pct):.2f}%（筹码持续集中，散户离场/主力锁仓）"
        elif chg_pct > 1.0:
            result["direction"] = -1
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
    """Deterministically parse northbound foreign capital metrics (RT-9)."""
    result: dict[str, Any] = {
        "status": "unavailable",
        "direction": 0,
        "direction_label": "数据不可用",
        "description": "北向资金数据不可用",
    }

    # Case 1: Service failure
    if _is_service_failure(northbound_raw) and "停止披露" not in str(northbound_raw):
        result["status"] = "failed"
        result["direction_label"] = "数据获取失败"
        result["description"] = "【数据获取失败】北向资金持股变动接口异常或超时，该项不可用。"
        return result

    # Case 2: Normal business absence / institutional cessation (RT-9: 非陆股通标的或停更)
    if _is_normal_absence(northbound_raw) or (northbound_raw and "自 2024 年 8 月起停止披露" in str(northbound_raw)):
        result["status"] = "not_applicable_normal"
        result["direction"] = 0
        result["direction_label"] = "非陆股通标的/停更"
        result["description"] = "标的未纳入陆股通范围或每日持股明细自2024年8月起制度性停更，无北向数据属正常业务状态。"
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


def _extract_snapshot_tool_info(
    tool_label: str,
    raw_val: Any,
    trade_date: str,
) -> dict[str, Any]:
    """Handle real-time snapshot tools with historical backtest awareness (RT-9)."""
    is_historical = is_historical_analysis_date(trade_date)
    raw_str = str(raw_val or "").strip()

    if is_historical or SNAPSHOT_ONLY_REFUSAL in raw_str or "历史日拒绝" in raw_str:
        return {
            "status": "snapshot_historical_rule",
            "board": f"{tool_label}历史不可用（快照类规则）",
            "description": f"【数据不可用】{tool_label}仅提供当前实时快照，历史回测分析（{trade_date}）按规则不予提供（正常业务约束）。",
        }

    if _is_service_failure(raw_val):
        return {
            "status": "failed",
            "board": f"{tool_label}接口失败",
            "description": f"【数据获取失败】{tool_label}服务异常或超时，该项不可用。",
        }

    if not raw_str or raw_str == "无数据":
        return {
            "status": "unavailable",
            "board": f"{tool_label}暂无数据",
            "description": f"【数据缺失】{tool_label}暂无数据。",
        }

    return {
        "status": "available",
        "board": raw_str[:120].strip(),
        "description": raw_str[:200].strip(),
    }


def compute_game_theory_signals(
    ticker: str,
    trade_date: str,
    raw_data: Mapping[str, Any],
    consensus_direction: Optional[str] = None,
) -> tuple[str, Optional[GameTheorySignals]]:
    """Pure deterministic computation of game theory signals and markdown report.

    Strict zero hallucination (RT-6): all values, player states, dominant strategies,
    and equilibrium evaluations are determined in Python from raw evidence.

    Degraded state (RT-8):
    When all opponent data is genuinely failed / unavailable, returns:
    ("【博弈论分析不可用】原因：...", None)
    """
    fund_flow_raw = raw_data.get("fund_flow_individual")
    fund_flow_evidence = raw_data.get("fund_flow_evidence")
    scale_metrics = raw_data.get("scale_metrics")
    margin_raw = raw_data.get("margin_trading")
    shareholder_raw = raw_data.get("shareholder_count")
    northbound_raw = raw_data.get("northbound_flow")
    board_raw = raw_data.get("fund_flow_board")
    lhb_raw = raw_data.get("lhb")

    # 1. Parse individual players with RT-9 awareness
    ff_info = _extract_fund_flow_info(fund_flow_raw, fund_flow_evidence, scale_metrics)
    margin_info = _extract_margin_info(margin_raw)
    sh_info = _extract_shareholder_info(shareholder_raw)
    nb_info = _extract_northbound_info(northbound_raw)
    board_info = _extract_snapshot_tool_info("行业板块资金流向", board_raw, trade_date)

    # 2. Track availability vs normal absence (RT-9)
    # Genuine active data sources that provide directional trading evidence
    core_statuses = [ff_info["status"], margin_info["status"], sh_info["status"], nb_info["status"]]
    has_active_data = any(st in ("available", "partial") for st in core_statuses)

    # Total failure / Degraded condition (RT-8 convention)
    # If all core dimensions have 0 active data
    if not has_active_data:
        degraded_report = (
            f"【博弈论分析不可用】原因：标的 {ticker} 在分析日 {trade_date} 的对手方核心数据源"
            "（个股主力资金流、融资融券、股东户数、北向资金等）全部缺失或接口调用失败，"
            "无法建立确定性博弈矩阵与占优策略推导。本项不可用，严禁编造默认指标。"
        )
        return degraded_report, None

    # Calculate confidence based on available / normally accounted dimensions
    # Available = available, normal_absence counts towards accounted completeness
    valid_count = sum(1 for st in core_statuses if st in ("available", "partial"))
    accounted_count = sum(1 for st in core_statuses if st in ("available", "partial", "not_applicable_normal"))
    total_count = len(core_statuses)

    confidence_score = round(valid_count / total_count, 2)
    data_status = "available" if accounted_count == total_count and valid_count >= 2 else "partial"

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

    if nb_info["status"] == "not_applicable_normal":
        likely_actions["北向资金"] = ["非陆股通标的/无北向交易"]
    elif nb_info["direction"] == 1:
        likely_actions["北向资金"] = ["核心资产配置", "顺势增持"]
    elif nb_info["direction"] == -1:
        likely_actions["北向资金"] = ["流动性套现", "逢高流出"]
    else:
        likely_actions["北向资金"] = ["配置平稳或停更"]

    if margin_info["status"] == "not_applicable_normal":
        likely_actions["杠杆资金"] = ["非两融标的/无杠杆交易"]
    elif margin_info["direction"] == 1:
        likely_actions["杠杆资金"] = ["利用融资杠杆追逐弹性", "顺势做多"]
    elif margin_info["direction"] == -1:
        likely_actions["杠杆资金"] = ["降低融资负债", "防守避险"]
    else:
        likely_actions["杠杆资金"] = ["杠杆仓位保持稳定"]

    if sh_info["direction"] == 1:
        likely_actions["散户群体"] = ["筹码集中持仓", "浮筹清洗完毕"]
    elif sh_info["direction"] == -1:
        likely_actions["散户群体"] = ["跟风买入", "追涨追高", "高位被动承接"]
    else:
        likely_actions["散户群体"] = ["观望等待"]

    # 4. Deterministic Dominant Strategy derivation (RT-6)
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
    elif main_dir == 1:
        dominant_strategy = "顺势跟随：主力资金呈增持做多意愿，适度参与多头博弈"
        overall_dir = "偏多"
    elif main_dir == -1:
        dominant_strategy = "谨慎避险：主力资金流出减仓，建议以防守观望为主"
        overall_dir = "偏空"
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

    # 7. Construct and sanitize structured signals (RT-8 JSON safety)
    raw_signals: dict[str, Any] = {
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

    # Strict JSON sanitization (no NaN, Inf, numpy types)
    signals: GameTheorySignals = ensure_json_safe(raw_signals)
    json.dumps(signals, allow_nan=False)  # Assert strict JSON serialization

    # 8. Construct Markdown report
    conf_label = "高" if confidence_score >= 0.75 else ("中" if confidence_score >= 0.35 else "低")

    # LHB section (RT-9: 非异动日正常处理)
    lhb_str = str(lhb_raw or "").strip()
    if _is_normal_absence(lhb_str) or not lhb_str or lhb_str == "无数据":
        lhb_snippet = "- **龙虎榜席位事实**：当日无龙虎榜上榜记录（非异动日），属正常业务状态。\n"
    elif _is_service_failure(lhb_str):
        lhb_snippet = f"- **龙虎榜席位事实**：【数据获取失败】龙虎榜接口异常，该项不可用。\n"
    else:
        lhb_snippet = f"- **龙虎榜席位事实**：{lhb_str[:150].strip()}\n"

    report_text = (
        f"## 博弈论与对手盘分析报告（{ticker} | {trade_date}）\n\n"
        f"**数据覆盖与置信度**：{valid_count}/{total_count} 项核心数据有效参与计算（置信度: {confidence_score:.2f} | {conf_label}）\n\n"
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
    """Collect data from pool or fallback functions across all 8 tools with non-blocking safety."""
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

    def _fetch_or_fallback(key: str, fallback_fn, *args, **kwargs) -> Any:
        if pool and key in pool and pool[key] is not None:
            return pool[key]
        try:
            return fallback_fn(*args, **kwargs)
        except Exception as exc:
            return f"【数据获取失败】{key} — 原因：{exc}。该项不可用。"

    # Gather all 8 tool inputs enumerated in game_theory_tools.py
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
        """Core node execution logic with atomicity & fail-safe guarantees (RT-3, RT-8)."""
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

            # 3. Deterministic calculation (RT-6, RT-8, RT-9)
            report_text, signals = compute_game_theory_signals(
                ticker=ticker,
                trade_date=trade_date,
                raw_data=raw_data,
                consensus_direction=consensus_dir,
            )

            # Semantic consistency & atomicity validation (RT-8):
            # If signals is None or report is broken/unavailable, both must atomically degrade.
            if signals is None or not report_text or str(report_text).startswith("【博弈论分析不可用】"):
                degraded_report = (
                    str(report_text)
                    if (report_text and str(report_text).startswith("【博弈论分析不可用】"))
                    else f"【博弈论分析不可用】原因：标的 {ticker} 在分析日 {trade_date} 信号计算或报告生成不完整，保持原子性降级。"
                )
                fail_trace: TraceItem = {
                    "agent": AGENT_NAME,
                    "horizon": horizon,
                    "data_window": "短期博弈",
                    "key_finding": "博弈论分析不可用（数据缺失或降级状态）",
                    "verdict": "中性",
                    "confidence": "低",
                    "source_status": "unavailable",
                    "source_mode": "deterministic_game_theory",
                    "bundle_id": "game_theory_v1",
                    "direction_allowed": False,
                    "reason_codes": ["data_unavailable_or_degraded"],
                    "evidence_refs": [],
                    "financial_period_compliance": {},
                }
                return {
                    REPORT_KEY: degraded_report,
                    SIGNALS_KEY: None,
                    "analyst_traces": [fail_trace],
                }

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
                "evidence_refs": [k for k, v in raw_data.items() if not _is_service_failure(v)],
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

            # Return atomically with guaranteed JSON safety (RT-8)
            return {
                REPORT_KEY: report_text,
                SIGNALS_KEY: signals,
                "analyst_traces": [trace_item],
            }

        except Exception as exc:
            # RT-3 / RT-8: Fail-closed atomic degradation:
            # {"game_theory_report": "【博弈论分析不可用】原因：...", "game_theory_signals": None}
            logger.error("[GameTheoryNode] Execution failed: %s", exc, exc_info=True)
            degraded_msg = f"【博弈论分析不可用】原因：节点执行严重异常（{type(exc).__name__}: {exc}），该项不可用。"
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
                REPORT_KEY: degraded_msg,
                SIGNALS_KEY: None,
                "analyst_traces": [fail_trace],
            }

    async def _async_node(state: AgentState) -> dict[str, Any]:
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

    if hasattr(workflow, "edges"):
        edge_pair = ("Research Manager", "Trader")
        if edge_pair in workflow.edges:
            workflow.edges.remove(edge_pair)
            workflow.add_edge("Research Manager", NODE_NAME)
            workflow.add_edge(NODE_NAME, "Trader")
            logger.info("[GameTheoryNode] Successfully wired between Research Manager and Trader")
            return

    workflow.add_edge(NODE_NAME, "Trader")
