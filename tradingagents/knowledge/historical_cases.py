"""Historical Cases Knowledge & Learning Loop (历史案例学习与复盘闭环).

本模块实现分析完成后的「预测 vs 实际」案例归档与下次分析的动态历史案例检索注入：
1. 案例落库 (Record Case):
   - 在分析 completed 时提取 symbol、trade_date、decision/direction、关键 claims 及运行 Git SHA；
   - 对比 trade_date 之后的下一交易日（T+1）收盘价，计算实际涨跌幅；
   - 严禁前视偏差（as_of <= 评估日）；日历或行情缺失时写入【数据缺失】，严禁填 0 或今天；
   - 严禁另起 LLM 编造预测，保持完全幂等落库；
2. 案例检索与注入 (Retrieve & Format):
   - 支持同一行业（industry）或同一标的（symbol）检索最多 N 条相似历史案例；
   - 未命中统一返回【历史案例未命中】；
   - 零新增 pip 依赖，与现有 RAG 格式化风格无缝对齐。
"""

from __future__ import annotations

import bisect
import io
import json
import logging
import os
import re
import subprocess
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from sqlalchemy.orm import Session

from api.database import HistoricalCaseDB, ReportDB, get_db_ctx
from tradingagents.dataflows.trade_calendar import (
    TradeCalendarUnavailableError,
    _load_cn_trade_dates,
    _parse_date,
    cn_market_phase,
    is_cn_trading_day,
    now_cn,
)

logger = logging.getLogger(__name__)

# 统一常量定义
DATA_MISSING_PLACEHOLDER: str = "【数据缺失】"
HISTORICAL_CASE_MISSING_FALLBACK: str = "【历史案例未命中】"
HISTORICAL_CASE_MISSING_BLOCK: str = "【历史案例复盘】\n【历史案例未命中】"

_BASELINE_FALLBACK_SHA = "dcc871dff13878803881bdbb9aed55f7cc10dbeb"


def get_current_run_sha() -> str:
    """获取当前运行环境的精确 Git commit SHA。"""
    env_sha = (
        os.getenv("TA_RUN_SHA")
        or os.getenv("GIT_COMMIT_SHA")
        or os.getenv("COMMIT_SHA")
    )
    if env_sha and len(env_sha.strip()) >= 7:
        return env_sha.strip()

    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception as exc:
        logger.debug("Failed to get git sha via subprocess: %s", exc)

    return _BASELINE_FALLBACK_SHA


def get_next_cn_trading_day(date_str: str) -> Optional[str]:
    """获取给定日期之后的严格下一交易日 YYYY-MM-DD（若日历不可用或无后续交易日返回 None）。"""
    if not date_str or not isinstance(date_str, str):
        return None

    try:
        d = _parse_date(date_str)
    except Exception:
        return None

    dates, _ = _load_cn_trade_dates()
    if not dates:
        return None

    idx = bisect.bisect_right(dates, d)
    if idx < len(dates):
        return dates[idx].strftime("%Y-%m-%d")
    return None


def _parse_prices_from_stock_data(data_str: str) -> Dict[str, float]:
    """从 get_stock_data 返回的 CSV 文本中解析日期到收盘价的映射。"""
    if not data_str or not isinstance(data_str, str):
        return {}

    clean_lines = [
        line for line in data_str.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not clean_lines:
        return {}

    try:
        df = pd.read_csv(io.StringIO("\n".join(clean_lines)))
        if df.empty:
            return {}

        cols_map = {str(c).lower().strip(): c for c in df.columns}
        date_col = cols_map.get("date")
        close_col = cols_map.get("close")
        if not date_col or not close_col:
            return {}

        df["_p_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["_p_close"] = pd.to_numeric(df[close_col], errors="coerce")
        df = df.dropna(subset=["_p_date", "_p_close"])

        prices: Dict[str, float] = {}
        for _, row in df.iterrows():
            d_str = row["_p_date"].strftime("%Y-%m-%d")
            prices[d_str] = float(row["_p_close"])
        return prices
    except Exception as exc:
        logger.debug("Failed to parse stock data CSV: %s", exc)
        return {}


def calculate_t1_return(
    symbol: str,
    trade_date: str,
    eval_date: Optional[str] = None,
) -> Tuple[Optional[str], Optional[float], str]:
    """对比 trade_date 之后的下一交易日（T+1 收盘）涨跌。

    严格遵循契约：
    1. as_of 必须 <= 评估日；
    2. 日历缺失、未来未到日、行情获取失败统一写【数据缺失】，禁止填 0 或今天。

    返回三元组：(eval_date, actual_change_pct, actual_outcome_str)
    """
    if not symbol or not trade_date:
        return None, None, DATA_MISSING_PLACEHOLDER

    # 1. 确定评估日（下一交易日）
    target_eval_date = eval_date or get_next_cn_trading_day(trade_date)
    if not target_eval_date:
        return None, None, DATA_MISSING_PLACEHOLDER

    try:
        eval_d = _parse_date(target_eval_date)
    except Exception:
        return None, None, DATA_MISSING_PLACEHOLDER

    today_cn = now_cn().date()
    # 若评估日晚于今日，则未来数据不可知
    if eval_d > today_cn:
        return target_eval_date, None, DATA_MISSING_PLACEHOLDER

    # 若评估日恰为今日，需检查今日是否已经收盘
    if eval_d == today_cn:
        phase = cn_market_phase()
        if phase != "post_close":
            return target_eval_date, None, DATA_MISSING_PLACEHOLDER

    # 2. 调用已有行情接口获取价格数据
    try:
        from tradingagents.dataflows.interface import route_to_vendor

        raw_csv = route_to_vendor(
            "get_stock_data",
            symbol,
            trade_date,
            target_eval_date,
        )
        prices = _parse_prices_from_stock_data(raw_csv)
    except Exception as exc:
        logger.warning(
            "calculate_t1_return failed for %s (%s -> %s): %s",
            symbol,
            trade_date,
            target_eval_date,
            exc,
        )
        return target_eval_date, None, DATA_MISSING_PLACEHOLDER

    p_t0 = prices.get(trade_date)
    p_t1 = prices.get(target_eval_date)

    if p_t0 is None or p_t1 is None or p_t0 <= 0:
        return target_eval_date, None, DATA_MISSING_PLACEHOLDER

    change_pct = round(((p_t1 - p_t0) / p_t0) * 100, 2)
    outcome_str = f"{'+' if change_pct > 0 else ''}{change_pct:.2f}%"
    return target_eval_date, change_pct, outcome_str


def extract_claims_from_report(
    result_data: Optional[Dict[str, Any]],
    texts: Optional[Sequence[Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """从辩论/裁决或报告机读块中抽取关键 Claims 列表。若无则返回空列表 []。"""
    claims_list: List[Dict[str, Any]] = []
    seen_texts: set[str] = set()

    def _add_claim(c: Any) -> None:
        if not c:
            return
        if isinstance(c, dict):
            claim_text = str(c.get("claim") or c.get("text") or "").strip()
            if claim_text and claim_text not in seen_texts:
                seen_texts.add(claim_text)
                claims_list.append({
                    "claim_id": str(c.get("claim_id") or "").strip(),
                    "claim": claim_text,
                    "confidence": c.get("confidence"),
                })
        elif isinstance(c, str) and c.strip():
            claim_text = c.strip()
            if claim_text not in seen_texts:
                seen_texts.add(claim_text)
                claims_list.append({"claim": claim_text})

    if isinstance(result_data, dict):
        # 1. 顶层 debate state 中的 claims
        inv_state = result_data.get("investment_debate_state")
        if isinstance(inv_state, dict):
            for item in inv_state.get("claims") or []:
                _add_claim(item)

        risk_state = result_data.get("risk_debate_state")
        if isinstance(risk_state, dict):
            for item in risk_state.get("claims") or []:
                _add_claim(item)

        # 2. 短/中双周期 nested 结构
        for h_key in ("short", "medium"):
            h_data = (result_data.get("horizons") or {}).get(h_key) or result_data.get(h_key)
            if isinstance(h_data, dict):
                for sub_key in ("investment_debate_state", "risk_debate_state"):
                    sub_state = h_data.get(sub_key)
                    if isinstance(sub_state, dict):
                        for item in sub_state.get("claims") or []:
                            _add_claim(item)

    # 3. 从文本机读块解析 <!-- DEBATE_STATE: ... --> 或 <!-- RISK_STATE: ... -->
    search_texts = list(texts or [])
    if isinstance(result_data, dict):
        for k in ("final_trade_decision", "investment_plan", "trader_investment_plan"):
            val = result_data.get(k)
            if isinstance(val, str):
                search_texts.append(val)

    for txt in search_texts:
        if not txt or not isinstance(txt, str):
            continue
        for tag in ("DEBATE_STATE", "RISK_STATE"):
            for m in re.finditer(rf"<!--\s*{tag}\s*:\s*(\{{.*?\}})\s*-->", txt, re.DOTALL):
                try:
                    payload = json.loads(m.group(1))
                    for raw_claim in payload.get("new_claims") or []:
                        _add_claim(raw_claim)
                except Exception:
                    pass

    return claims_list


def evaluate_prediction_error(
    decision: Optional[str],
    direction: Optional[str],
    change_pct: Optional[float],
) -> Optional[bool]:
    """根据决策方向与 T+1 实际涨跌对比判定是否出现预测偏差 (is_error)。"""
    if change_pct is None:
        return None

    d_upper = str(decision or "").strip().upper()
    dir_str = str(direction or "").strip()

    is_bull = (
        d_upper in {"BUY", "买入", "增持"}
        or "多" in dir_str
        or "买" in dir_str
        or "增持" in dir_str
    )
    is_bear = (
        d_upper in {"SELL", "卖出", "减持"}
        or "空" in dir_str
        or "卖" in dir_str
        or "减持" in dir_str
    )
    is_neutral = (
        d_upper in {"HOLD", "持有", "中性"}
        or "持有" in dir_str
        or "中性" in dir_str
    )

    if is_bull:
        return change_pct < 0.0
    if is_bear:
        return change_pct > 0.0
    if is_neutral:
        # 中性时涨跌幅过大视为偏差
        return abs(change_pct) >= 3.0

    return None


def record_historical_case(
    db: Session,
    report: Union[ReportDB, Mapping[str, Any]],
    commit_sha: Optional[str] = None,
) -> Optional[HistoricalCaseDB]:
    """在分析 completed 后落库一条案例（严格幂等，只在 completed 落库）。"""
    if report is None:
        return None

    status = getattr(report, "status", None) or (report.get("status") if isinstance(report, Mapping) else None)
    if str(status or "").lower() != "completed":
        logger.debug("Skip recording historical case: report status is not completed (%s)", status)
        return None

    report_id = getattr(report, "id", None) or (report.get("id") if isinstance(report, Mapping) else None)
    symbol = getattr(report, "symbol", None) or (report.get("symbol") if isinstance(report, Mapping) else None)
    trade_date = getattr(report, "trade_date", None) or (report.get("trade_date") if isinstance(report, Mapping) else None)

    if not symbol or not trade_date:
        logger.warning("Skip recording historical case: missing symbol or trade_date")
        return None

    symbol = str(symbol).strip().upper()
    trade_date = str(trade_date).strip()

    decision = getattr(report, "decision", None) or (report.get("decision") if isinstance(report, Mapping) else None)
    direction = getattr(report, "direction", None) or (report.get("direction") if isinstance(report, Mapping) else None)
    confidence = getattr(report, "confidence", None) or (report.get("confidence") if isinstance(report, Mapping) else None)
    result_data = getattr(report, "result_data", None) or (report.get("result_data") if isinstance(report, Mapping) else None)

    # 1. 抽取行业
    from tradingagents.agents.utils.knowledge_context import resolve_industry_profile
    profile = resolve_industry_profile(ticker=symbol, extra_text=str(result_data or ""))
    industry_id = profile.industry_id if profile else None

    # 2. 抽取 Claims 列表
    final_trade_decision = getattr(report, "final_trade_decision", None) or (
        report.get("final_trade_decision") if isinstance(report, Mapping) else None
    )
    investment_plan = getattr(report, "investment_plan", None) or (
        report.get("investment_plan") if isinstance(report, Mapping) else None
    )
    claims = extract_claims_from_report(
        result_data=result_data,
        texts=[final_trade_decision, investment_plan],
    )

    # 3. 运行 SHA
    run_sha = commit_sha or get_current_run_sha()

    # 4. 计算 T+1 实际表现
    eval_date, change_pct, outcome_str = calculate_t1_return(symbol, trade_date)
    is_error = evaluate_prediction_error(decision, direction, change_pct)

    # 5. 幂等检查与保存
    existing = None
    if report_id:
        existing = db.query(HistoricalCaseDB).filter(HistoricalCaseDB.report_id == str(report_id)).first()
    if not existing:
        existing = (
            db.query(HistoricalCaseDB)
            .filter(
                HistoricalCaseDB.symbol == symbol,
                HistoricalCaseDB.trade_date == trade_date,
            )
            .first()
        )

    now = datetime.now(timezone.utc)
    if existing:
        existing.report_id = str(report_id) if report_id else existing.report_id
        existing.symbol = symbol
        existing.industry = industry_id or existing.industry
        existing.trade_date = trade_date
        existing.decision = decision
        existing.direction = direction
        existing.confidence = confidence
        existing.claims = claims
        existing.run_sha = run_sha
        existing.eval_date = eval_date
        existing.actual_change_pct = change_pct
        existing.actual_outcome = outcome_str
        existing.is_error = is_error
        existing.updated_at = now
        case_obj = existing
    else:
        from uuid import uuid4
        case_obj = HistoricalCaseDB(
            id=str(uuid4()),
            report_id=str(report_id) if report_id else None,
            symbol=symbol,
            industry=industry_id,
            trade_date=trade_date,
            decision=decision,
            direction=direction,
            confidence=confidence,
            claims=claims,
            run_sha=run_sha,
            eval_date=eval_date,
            actual_change_pct=change_pct,
            actual_outcome=outcome_str,
            is_error=is_error,
            created_at=now,
            updated_at=now,
        )
        db.add(case_obj)

    try:
        db.commit()
        db.refresh(case_obj)
        logger.info(
            "[historical_cases] Recorded case for %s on %s (outcome: %s, error: %s)",
            symbol,
            trade_date,
            outcome_str,
            is_error,
        )
    except Exception as exc:
        db.rollback()
        logger.error("Failed to commit historical case: %s", exc)
        raise

    return case_obj


def retrieve_similar_historical_cases(
    symbol: str,
    industry: Optional[str] = None,
    before_date: Optional[str] = None,
    max_cases: int = 3,
    errors_only: bool = False,
    db: Optional[Session] = None,
) -> List[HistoricalCaseDB]:
    """按同一标的或同一行业检索最多 max_cases 条历史案例（严格防前视偏差）。"""
    if max_cases <= 0:
        return []

    def _query_with_session(session: Session) -> List[HistoricalCaseDB]:
        results: List[HistoricalCaseDB] = []
        seen_ids: set[str] = set()

        # 1. 优先同一标的 (symbol)
        q_sym = session.query(HistoricalCaseDB).filter(HistoricalCaseDB.symbol == symbol.strip().upper())
        if before_date:
            q_sym = q_sym.filter(HistoricalCaseDB.trade_date < before_date.strip())
        if errors_only:
            q_sym = q_sym.filter(HistoricalCaseDB.is_error.is_(True))

        for row in q_sym.order_by(HistoricalCaseDB.trade_date.desc()).limit(max_cases).all():
            if row.id not in seen_ids:
                seen_ids.add(row.id)
                results.append(row)

        # 2. 若数量不足且提供了行业，补充同一行业 (industry)
        if len(results) < max_cases and industry and str(industry).strip():
            remaining = max_cases - len(results)
            q_ind = (
                session.query(HistoricalCaseDB)
                .filter(HistoricalCaseDB.industry == str(industry).strip())
                .filter(HistoricalCaseDB.symbol != symbol.strip().upper())
            )
            if before_date:
                q_ind = q_ind.filter(HistoricalCaseDB.trade_date < before_date.strip())
            if errors_only:
                q_ind = q_ind.filter(HistoricalCaseDB.is_error.is_(True))

            for row in q_ind.order_by(HistoricalCaseDB.trade_date.desc()).limit(remaining).all():
                if row.id not in seen_ids:
                    seen_ids.add(row.id)
                    results.append(row)

        return results

    if db is not None:
        return _query_with_session(db)

    with get_db_ctx() as ctx_db:
        return _query_with_session(ctx_db)


def _format_single_case_block(index: int, case: Any) -> str:
    """格式化单个历史案例详情。"""
    c_dict = case.to_dict() if hasattr(case, "to_dict") else dict(case)

    sym = c_dict.get("symbol") or "未知标的"
    ind = c_dict.get("industry") or "未知行业"
    t_date = c_dict.get("trade_date") or "未知日期"
    dec = c_dict.get("decision") or "无"
    dir_str = c_dict.get("direction") or ""
    conf = c_dict.get("confidence")
    conf_str = f"（置信度：{conf}%）" if conf is not None else ""

    dec_display = f"{dec}" + (f" / {dir_str}" if dir_str and dir_str != dec else "") + conf_str

    claims_list = c_dict.get("claims") or []
    if isinstance(claims_list, list) and claims_list:
        claim_lines = []
        for c in claims_list[:3]:  # 最多呈现3条核心论据
            if isinstance(c, dict):
                c_txt = c.get("claim") or c.get("text") or ""
            else:
                c_txt = str(c)
            if c_txt.strip():
                claim_lines.append(f"  * {c_txt.strip()}")
        claims_block = "\n".join(claim_lines) if claim_lines else "  * 无记录"
    else:
        claims_block = "  * 无记录"

    outcome = c_dict.get("actual_outcome") or DATA_MISSING_PLACEHOLDER
    eval_d = c_dict.get("eval_date")
    eval_str = f"（评估日 {eval_d}）" if eval_d else ""
    is_err = c_dict.get("is_error")

    if outcome == DATA_MISSING_PLACEHOLDER:
        review_note = "实际行情未到或数据缺失，待后续评估。"
    elif is_err is True:
        review_note = "【偏差复盘】预测方向与次日实际走势相悖，需重点核验假设漏洞与反转风险。"
    elif is_err is False:
        review_note = "【验证一致】预测方向与次日实际走势一致，逻辑与催化传导有效。"
    else:
        review_note = "行情已记录，需结合中长期周期持续跟踪。"

    return (
        f"[案例 {index}] 标的：{sym}（行业：{ind}）| 历史分析日：{t_date}\n"
        f"- 历史研判：{dec_display}\n"
        f"- 核心论据 (Claims)：\n{claims_block}\n"
        f"- T+1 实际表现：{outcome}{eval_str}\n"
        f"- 案例启示：{review_note}"
    )


def format_historical_cases_context(
    cases_or_query: Any,
    symbol: str = "",
    industry: Optional[str] = None,
    before_date: Optional[str] = None,
    max_cases: int = 3,
    fallback_on_miss: bool = True,
    db: Optional[Session] = None,
) -> str:
    """将检索到的历史案例格式化为 Prompt 注入文本。

    若未命中且 fallback_on_miss=True，返回 '【历史案例复盘】\\n【历史案例未命中】'；
    若 fallback_on_miss=False，返回空字符串。
    """
    cases: List[Any] = []

    if isinstance(cases_or_query, (list, tuple)):
        cases = list(cases_or_query)
    elif symbol and str(symbol).strip():
        cases = retrieve_similar_historical_cases(
            symbol=symbol,
            industry=industry,
            before_date=before_date,
            max_cases=max_cases,
            db=db,
        )

    if not cases:
        return HISTORICAL_CASE_MISSING_BLOCK if fallback_on_miss else ""

    formatted_cases = [
        _format_single_case_block(i, c)
        for i, c in enumerate(cases[:max_cases], start=1)
    ]

    header = "【历史案例复盘】（基于历史同标的/同行业预测与实际表现总结，严禁重复已证伪误判）"
    return header + "\n\n" + "\n\n".join(formatted_cases)
