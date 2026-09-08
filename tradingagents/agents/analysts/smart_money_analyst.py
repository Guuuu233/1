import logging
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
import math
from typing import Any
from tradingagents.agents.utils.context_utils import get_cn_stock_name, format_phase1_reports
import asyncio
import json

from tradingagents.dataflows.fund_flow_evidence import (
    consensus_prompt_instruction,
    select_fund_flow_source,
    validate_model_summary,
)

from langchain_core.messages import HumanMessage, SystemMessage
from tradingagents.dataflows.config import get_config
from tradingagents.prompts import get_prompt
from tradingagents.graph.intent_parser import (
    build_horizon_context,
    get_bound_research_horizon,
)
from tradingagents.agents.utils.agent_states import current_tracker_var, extract_verdict, check_llm_output_degraded, check_stream_chunk_degraded
from api.database import log_llm_call

logger = logging.getLogger(__name__)


def _resolve_research_horizon(state: dict | None) -> str:
    """Resolve the active research horizon for the current run.

    Priority:
    1. state["horizon"] if present and truthy
    2. state["horizon_run_metadata"]["resolved"][0] if present
    3. state["horizon_run_metadata"]["requested"][0] if present
    4. get_bound_research_horizon() from H-04a thread binding
    5. fallback to "short"
    """
    if state:
        if state.get("horizon"):
            return state["horizon"]
        metadata = state.get("horizon_run_metadata")
        if isinstance(metadata, dict):
            resolved = metadata.get("resolved")
            if resolved and isinstance(resolved, list) and len(resolved) > 0:
                return resolved[0]
            requested = metadata.get("requested")
            if requested and isinstance(requested, list) and len(requested) > 0:
                return requested[0]
    bound = get_bound_research_horizon()
    if bound:
        return bound
    return "short"


def format_fund_flow_scale_metrics_prompt(
    scale_metrics: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None,
) -> str:
    """Format deterministic scale_metrics evidence snippet for LLM prompt.

    Contracts:
    1. Read real fields only from scale_metrics.
    2. Read selected_algorithm_group and reference_only ONLY from selection (never from scale_metrics).
    3. reference_only displays True/False ONLY when selection contains a strict boolean;
       if missing, non-Mapping, missing key, or non-bool, display
       '未知/缺少 selection，按 reference_only 纪律处理', never default to False or coerce.
    4. Only accept known status (available/partial/unavailable). Unknown/missing status fails closed to unavailable.
       Cross-check status with usable ratios:
       - available requires both ratios with required denominator source & unit, otherwise downgrade to partial or unavailable with contract contradiction gap.
       - partial only presents structurally complete usable ratio; if neither is usable, downgrade to unavailable.
       - unavailable is never presented as usable, regardless of values carried.
    5. ts_code or trade_date missing fails closed to unavailable.
    6. Denominator source/unit read from corresponding fields, allowing denominator_sources / denominator_units
       Mapping as exact fallback; if still missing, ratio is unusable (no '未指定' to keep available).
    7. gaps/gap_list as str kept as single item; list/tuple preserved per item; malformed types add explicit contract gap.
    8. Discipline constraints appended. Input objects not modified in place.
    """
    discipline_text = (
        "【纪律约束】相对规模比率仅作为同标的、同交易日的统计参考证据；"
        "严禁据此识别机构或散户账户身份，严禁进行跨股票横向排名，严禁据此生成新评分、权重、概率或交易执行信号。"
        "不得在分母缺失时退回绝对净额作规模结论。"
    )

    # 1. Read metadata strictly and ONLY from selection
    selected_algorithm_group = None
    ref_only_repr = "未知/缺少 selection，按 reference_only 纪律处理"
    selected_source = None
    if isinstance(selection, Mapping):
        selected_algorithm_group = selection.get("selected_algorithm_group")
        if "reference_only" in selection:
            raw_ref = selection["reference_only"]
            if isinstance(raw_ref, bool):
                ref_only_repr = str(raw_ref)
        selected_source = selection.get("selected_source")

    # 2. Handle missing or empty scale_metrics -> fail closed
    if not isinstance(scale_metrics, Mapping) or not scale_metrics:
        return (
            "【资金流相对规模证据（同标的同日相对参考）】\n"
            "- 状态: unavailable (相对规模不可用/不得据绝对净额替代)\n"
            "- 缺口说明: 缺少 scale_metrics 相对规模对象\n"
            f"- {discipline_text}"
        )

    # 3. Parse gaps / gap_list
    raw_gaps = scale_metrics.get("gaps")
    if raw_gaps is None and "gap_list" in scale_metrics:
        raw_gaps = scale_metrics.get("gap_list")

    gaps: list[str] = []
    if raw_gaps is not None:
        if isinstance(raw_gaps, str):
            if raw_gaps.strip():
                gaps.append(raw_gaps.strip())
        elif isinstance(raw_gaps, (list, tuple)):
            for g in raw_gaps:
                if g is not None and str(g).strip():
                    gaps.append(str(g).strip())
        else:
            gaps.append(f"契约异常: gaps 包含畸形类型 ({type(raw_gaps).__name__}: {raw_gaps})")

    # 4. Read real identification fields
    raw_ts_code = scale_metrics.get("ts_code")
    raw_trade_date = scale_metrics.get("trade_date")
    ts_code = str(raw_ts_code).strip() if raw_ts_code is not None else ""
    trade_date = str(raw_trade_date).strip() if raw_trade_date is not None else ""

    missing_id_gaps: list[str] = []
    if not ts_code:
        missing_id_gaps.append("契约异常: 标的代码 (ts_code) 缺失，整体相对规模不可用")
    if not trade_date:
        missing_id_gaps.append("契约异常: 交易日期 (trade_date) 缺失，整体相对规模不可用")

    # 5. Read status strictly
    raw_status = scale_metrics.get("status")
    if raw_status is None:
        gaps.append("契约异常: 缺少 status 状态字段 (fail-closed 置为 unavailable)")
        status = "unavailable"
    elif not isinstance(raw_status, str):
        gaps.append(f"契约异常: 状态字段为未知类型 ({type(raw_status).__name__}: {raw_status}) (fail-closed 置为 unavailable)")
        status = "unavailable"
    else:
        norm_status = raw_status.strip().lower()
        if norm_status in {"available", "partial", "unavailable"}:
            status = norm_status
        else:
            gaps.append(f"契约异常: 未知状态 '{raw_status}' (fail-closed 置为 unavailable)")
            status = "unavailable"

    # 6. Read ratio values and sources/units with fallback
    denominator_sources = scale_metrics.get("denominator_sources")
    denominator_units = scale_metrics.get("denominator_units")
    denominator_source = scale_metrics.get("denominator_source")

    net_to_circ_mv = scale_metrics.get("net_to_circ_mv")
    net_to_circ_mv_text = scale_metrics.get("net_to_circ_mv_text")
    net_to_amount = scale_metrics.get("net_to_amount")
    net_to_amount_text = scale_metrics.get("net_to_amount_text")

    # circ_mv source & unit resolution
    circ_mv_source = scale_metrics.get("circ_mv_source")
    if not (circ_mv_source and str(circ_mv_source).strip()):
        if isinstance(denominator_sources, Mapping):
            fb_src = denominator_sources.get("circ_mv")
            circ_mv_source = str(fb_src).strip() if fb_src and str(fb_src).strip() else None
        else:
            circ_mv_source = None
    else:
        circ_mv_source = str(circ_mv_source).strip()

    circ_mv_unit = scale_metrics.get("circ_mv_unit")
    if not (circ_mv_unit and str(circ_mv_unit).strip()):
        if isinstance(denominator_units, Mapping):
            fb_unit = denominator_units.get("circ_mv")
            circ_mv_unit = str(fb_unit).strip() if fb_unit and str(fb_unit).strip() else None
        else:
            circ_mv_unit = None
    else:
        circ_mv_unit = str(circ_mv_unit).strip()

    # amount source & unit resolution
    amount_source = scale_metrics.get("amount_source")
    if not (amount_source and str(amount_source).strip()):
        if isinstance(denominator_sources, Mapping):
            fb_src = denominator_sources.get("amount")
            amount_source = str(fb_src).strip() if fb_src and str(fb_src).strip() else None
        else:
            amount_source = None
    else:
        amount_source = str(amount_source).strip()

    amount_unit = scale_metrics.get("amount_unit")
    if not (amount_unit and str(amount_unit).strip()):
        if isinstance(denominator_units, Mapping):
            fb_unit = denominator_units.get("amount")
            amount_unit = str(fb_unit).strip() if fb_unit and str(fb_unit).strip() else None
        else:
            amount_unit = None
    else:
        amount_unit = str(amount_unit).strip()

    # Validate ratios
    def _is_valid_num(v: Any) -> bool:
        if v is None or isinstance(v, bool):
            return False
        if not isinstance(v, (int, float, Decimal, str)):
            return False
        try:
            if isinstance(v, float) and not math.isfinite(v):
                return False
            stripped_or_str = str(v).strip()
            if not stripped_or_str:
                return False
            d = Decimal(stripped_or_str)
            if not d.is_finite():
                return False
            fv = float(d)
            if not math.isfinite(fv):
                return False
            return True
        except (InvalidOperation, TypeError, ValueError, OverflowError):
            return False

    has_circ_val = _is_valid_num(net_to_circ_mv)
    has_amt_val = _is_valid_num(net_to_amount)

    # String representations
    if has_circ_val:
        if net_to_circ_mv_text is not None and str(net_to_circ_mv_text).strip():
            circ_str = str(net_to_circ_mv_text).strip()
        else:
            circ_str = "0" if net_to_circ_mv == 0 else str(net_to_circ_mv).strip()
    else:
        circ_str = None

    if has_amt_val:
        if net_to_amount_text is not None and str(net_to_amount_text).strip():
            amt_str = str(net_to_amount_text).strip()
        else:
            amt_str = "0" if net_to_amount == 0 else str(net_to_amount).strip()
    else:
        amt_str = None

    # Check structural completeness of each ratio
    circ_usable = has_circ_val and bool(circ_mv_source) and bool(circ_mv_unit)
    amt_usable = has_amt_val and bool(amount_source) and bool(amount_unit)

    if has_circ_val and not circ_usable:
        if not circ_mv_source and not circ_mv_unit:
            gaps.append("契约异常: net_to_circ_mv 存在但分母来源与单位均缺失，该比率不可用")
        elif not circ_mv_source:
            gaps.append("契约异常: net_to_circ_mv 存在但分母来源缺失，该比率不可用")
        else:
            gaps.append("契约异常: net_to_circ_mv 存在但分母单位缺失，该比率不可用")

    if has_amt_val and not amt_usable:
        if not amount_source and not amount_unit:
            gaps.append("契约异常: net_to_amount 存在但分母来源与单位均缺失，该比率不可用")
        elif not amount_source:
            gaps.append("契约异常: net_to_amount 存在但分母来源缺失，该比率不可用")
        else:
            gaps.append("契约异常: net_to_amount 存在但分母单位缺失，该比率不可用")

    # 7. Check ts_code and trade_date
    if missing_id_gaps:
        status = "unavailable"
        gaps.extend(missing_id_gaps)

    # 8. Cross-check status with usable ratios
    if status == "available":
        if circ_usable and amt_usable:
            pass
        elif circ_usable or amt_usable:
            status = "partial"
            gaps.append("契约矛盾: status 声明为 available，但仅有一个比率完整可用，降级为 partial")
        else:
            status = "unavailable"
            gaps.append("契约矛盾: status 声明为 available，但两个比率均不可用，降级为 unavailable")
    elif status == "partial":
        if not circ_usable and not amt_usable:
            status = "unavailable"
            gaps.append("契约矛盾: status 声明为 partial，但无任何可用比率，降级为 unavailable")
    elif status == "unavailable":
        if has_circ_val or has_amt_val:
            gaps.append("契约矛盾: status 声明为 unavailable 但携带比率数值，按 unavailable 纪律不予呈现")

    # 9. Handle unavailable scenario
    if status == "unavailable":
        gap_lines = "\n".join(f"  * {g}" for g in gaps) if gaps else "  * 相对规模不可用或分母缺失"
        return (
            "【资金流相对规模证据（同标的同日相对参考）】\n"
            "- 状态: unavailable (相对规模不可用/不得据绝对净额替代)\n"
            f"- 标的代码: {ts_code if ts_code else '缺失'}\n"
            f"- 交易日期: {trade_date if trade_date else '缺失'}\n"
            f"- 缺口说明:\n{gap_lines}\n"
            f"- {discipline_text}"
        )

    # 10. Available and partial output
    source_display = selected_source or denominator_source or "未指定"
    lines = [
        "【资金流相对规模证据（同标的同日相对参考）】",
        f"- 状态: {status} ({'完整可用' if status == 'available' else '部分可用'})",
        f"- 标的代码: {ts_code}",
        f"- 交易日期: {trade_date}",
        f"- 资金来源: {source_display}",
        f"- 算法组: {selected_algorithm_group or '未指定'}",
        f"- 参考属性: reference_only={ref_only_repr}",
    ]

    if circ_usable:
        lines.append(
            f"- 净额占流通市值比 (net_to_circ_mv): {circ_str} "
            f"(分母来源: {circ_mv_source}, 分母单位: {circ_mv_unit})"
        )
    else:
        lines.append("- 净额占流通市值比 (net_to_circ_mv): 缺失/不可用")

    if amt_usable:
        lines.append(
            f"- 净额占成交额比 (net_to_amount): {amt_str} "
            f"(分母来源: {amount_source}, 分母单位: {amount_unit})"
        )
    else:
        lines.append("- 净额占成交额比 (net_to_amount): 缺失/不可用")

    if gaps:
        lines.append("- 缺口说明:")
        for g in gaps:
            lines.append(f"  * {g}")

    lines.append(f"- {discipline_text}")
    return "\n".join(lines)


def create_smart_money_analyst(llm, data_collector=None):
    async def _safe(tool, payload):
        try:
            return await asyncio.to_thread(tool.invoke, payload)
        except Exception as exc:
            return f"调用失败：{exc}"

    async def smart_money_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        stock_name = get_cn_stock_name(ticker)

        ticker_display = f"{ticker} ({stock_name})" if stock_name and stock_name != ticker else ticker
        logger.debug("[Smart Money Analyst] START %s %s", ticker_display, current_date)
        observation_horizon = "short"  # 资金面专业观察窗固定为短期
        research_horizon = _resolve_research_horizon(state)
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])

        config = get_config()
        system_message = get_prompt("smart_money_system_message", config=config) or ""
        horizon_ctx = build_horizon_context(
            observation_horizon,
            focus_areas,
            specific_questions,
            agent_type="smart_money",
            research_horizon=research_horizon,
        )

        pool = data_collector.get(ticker, current_date) if data_collector else None
        state_market_data_context = state.get("market_data_context")

        pool_context = None
        if pool is not None:
            fund_flow = pool.get("fund_flow_individual", "无数据")
            pool_context = pool.get("market_data_context")
            if not isinstance(pool_context, dict) and isinstance(state_market_data_context, dict):
                pool_context = state_market_data_context
            fund_flow_evidence = (
                pool_context.get("fund_flow_evidence", {})
                if isinstance(pool_context, dict)
                else {}
            )
            lhb = pool.get("lhb", "无数据")
            volume = pool.get("indicators", {}).get("vwma", "无数据")
        else:
            from tradingagents.agents.utils.agent_utils import (
                get_individual_fund_flow, get_lhb_detail, get_indicators,
            )

            # Parallelize fallback fetches
            results = await asyncio.gather(
                _safe(get_individual_fund_flow, {"symbol": ticker, "curr_date": current_date}),
                _safe(get_lhb_detail, {"symbol": ticker, "date": current_date}),
                _safe(get_indicators, {
                    "symbol": ticker, "indicator": "volume",
                    "curr_date": current_date, "look_back_days": 20,
                })
            )
            fund_flow, lhb, volume = results
            fund_flow_evidence = (
                state_market_data_context.get("fund_flow_evidence", {})
                if isinstance(state_market_data_context, dict)
                else {}
            )

        selection: dict = {}
        if isinstance(fund_flow_evidence, dict):
            selection = fund_flow_evidence.get("selection") or {}
        if not isinstance(selection, dict) or "selected_source" not in selection:
            records = fund_flow_evidence.get("records") or [] if isinstance(fund_flow_evidence, dict) else []
            selection = select_fund_flow_source(
                records,
                symbol=ticker,
                requested_as_of=current_date,
            )
            if isinstance(fund_flow_evidence, dict):
                fund_flow_evidence["selection"] = selection
        if isinstance(fund_flow_evidence, Mapping):
            evidence_for_display = dict(fund_flow_evidence)
            evidence_for_display.pop("scale_metrics", None)
        else:
            evidence_for_display = fund_flow_evidence
        evidence_text = json.dumps(evidence_for_display, ensure_ascii=False, sort_keys=True)
        consensus_instruction = consensus_prompt_instruction(selection)

        scale_metrics = None
        if isinstance(fund_flow_evidence, Mapping):
            scale_metrics = fund_flow_evidence.get("scale_metrics")

        scale_metrics_prompt = format_fund_flow_scale_metrics_prompt(scale_metrics, selection)
        validation = (
            fund_flow_evidence.get("validation", {})
            if isinstance(fund_flow_evidence, dict)
            else {}
        )
        selection_allowed = bool(
            isinstance(selection, dict)
            and selection.get("status") in {"selected", "consensus"}
            and selection.get("direction_allowed")
            and selection.get("selected_source")
            and selection.get("selected_field")
            and selection.get("selected_value") is not None
            and isinstance(selection.get("hard_guard"), dict)
            and not selection.get("hard_guard", {}).get("blocked")
        )
        consensus_blocked = bool(
            not selection_allowed
            or validation.get("status") in {"blocked", "mismatch"}
            or validation.get("hard_guard", {}).get("blocked")
        )
        consensus_guard = {
            "blocked": consensus_blocked,
            "direction_allowed": not consensus_blocked,
            "status": selection.get("status", "not_checked") if isinstance(selection, dict) else "not_checked",
            "selection": selection,
            "validation": validation,
            "reason": (validation or {}).get("hard_guard", {}).get("reason")
            or (selection or {}).get("reason", "fund-flow source selection unavailable"),
        }
        phase1_reports_text = format_phase1_reports(state)
        messages = [
            SystemMessage(content=(
                system_message
                + "\n\n请严格基于提供的量化数据输出分析，全程使用中文。"
            )),
            HumanMessage(content=(
                horizon_ctx + "\n"
                f"请分析 {ticker_display} 在 {current_date} 的资金流数据。若来源为同花顺即时资金流净额快照，"
                "不得将其视为新浪历史 netamount/r0_net 同口径的主力序列。\n\n"
                f"{phase1_reports_text}\n\n"
                f"【资金流数据（来源、日期与口径见数据）】\n{fund_flow}\n\n"
                f"【资金流结构化 evidence（仅用于精确累计，不得从展示文本反推）】\n{evidence_text}\n\n"
                f"【资金流来源选择与方向规则】\n{consensus_instruction}\n\n"
                f"{scale_metrics_prompt}\n\n"
                f"【龙虎榜数据】\n{lhb}\n\n"
                f"【成交量指标(vwma)】\n{volume}"
            )),
        ]

        # ── 实现 Token 级流式输出（含降级保障） ──────────────────


        tracker = current_tracker_var.get()


        import time as _time
        full_content = ""
        _last_chunk = None
        _t0 = _time.monotonic()


        try:


            async for chunk in llm.astream(messages):
                _last_chunk = chunk
                content = chunk.content if hasattr(chunk, "content") else str(chunk)


                full_content += content
                if check_stream_chunk_degraded(full_content, "Smart Money Analyst"):
                    break


                # Hold all content until the structured guard is finalized.
                # Directional SSE tokens must never precede a conflict/mismatch guard.


        except Exception as exc:


            logger.debug("[Smart Money Analyst] Stream error: %s", exc)



        if not full_content.strip():


            logger.debug("[Smart Money Analyst] Stream yielded empty text, attempting invoke fallback...")


            try:


                res = await asyncio.to_thread(llm.invoke, messages)


                full_content = res.content if hasattr(res, "content") else str(res)


                # Emit only after final validation below, so blocked analysis
                # cannot leak directional content through the stream.


            except Exception as exc:


                full_content = f"分析报告生成失败：{exc}"

        logger.debug("[Smart Money Analyst] DONE %s, report length=%s", ticker_display, len(full_content))
        market_data_context = state_market_data_context
        if not isinstance(market_data_context, dict) and isinstance(pool, dict):
            market_data_context = pool.get("market_data_context")
        if isinstance(market_data_context, dict):
            fund_flow_evidence = market_data_context.get("fund_flow_evidence", fund_flow_evidence)
        if isinstance(fund_flow_evidence, dict) and fund_flow_evidence.get("records"):
            current_selection = fund_flow_evidence.get("selection")
            if not isinstance(current_selection, dict) or "selected_source" not in current_selection:
                current_selection = select_fund_flow_source(
                    fund_flow_evidence.get("records", []),
                    symbol=ticker,
                    requested_as_of=current_date,
                )
                fund_flow_evidence["selection"] = current_selection
            selection = current_selection
            selected_field = selection.get("selected_field")
            selected_source = selection.get("selected_source")
            validation_window = int(selection.get("selected_window_days") or 1)
            fund_flow_evidence["validation"] = validate_model_summary(
                fund_flow_evidence.get("records", []),
                full_content,
                window_days=validation_window,
                selected_field=selected_field,
                selected_source=selected_source,
                requested_as_of=current_date,
            )
            fund_flow_evidence["consensus"] = current_selection
            if isinstance(market_data_context, dict):
                market_data_context["fund_flow_evidence"] = fund_flow_evidence
            selection = current_selection
            consensus = selection
            validation = fund_flow_evidence.get("validation", validation)
            selection_allowed = bool(
                isinstance(selection, dict)
                and selection.get("status") in {"selected", "consensus"}
                and selection.get("direction_allowed")
                and selection.get("selected_source")
                and selection.get("selected_field")
                and selection.get("selected_value") is not None
                and isinstance(selection.get("hard_guard"), dict)
                and not selection.get("hard_guard", {}).get("blocked")
            )
            if selection_allowed and selected_field == "netamount":
                non_main_force_violations = [
                    kw for kw in (
                        "主力吸筹", "主力建仓", "主力增持", "主力减持", "主力派发",
                        "主力悄然吸筹", "主力大幅增持", "主力大幅减持",
                        "主力资金吸筹", "主力资金建仓", "主力资金增持", "主力资金减持", "主力资金派发",
                    )
                    if kw in full_content
                ]
                if non_main_force_violations:
                    if isinstance(validation, dict):
                        validation["hard_guard"] = {
                            "blocked": True,
                            "reason": f"仅有总资金净额(netamount)，严禁表述为主力吸筹/增持/减持（违规词：{', '.join(non_main_force_violations)}）",
                        }
                        validation["status"] = "blocked"
            consensus_blocked = bool(
                not selection_allowed
                or validation.get("status") in {"blocked", "mismatch"}
                or validation.get("hard_guard", {}).get("blocked")
            )
        if check_llm_output_degraded(full_content, "Smart Money Analyst"):
            full_content = "主力资金分析生成异常（输出退化），本项不可用"
        if consensus_blocked:
            full_content = (
                "资金流来源选择不可用或结构化累计存在冲突；已阻断增持、减持、吸筹方向摘要。"
                "请保留各来源原值，待日期、窗口、单位和字段语义校验通过后复核。"
            )
        elif isinstance(selection, dict) and selection.get("legacy_reference"):
            full_content = (
                "⚠️ legacy_web_algorithm：以下方向仅来自新浪旧 Web 算法，"
                "仅供参考，不得视为 Eastmoney/THS 新算法结论。\n"
                + full_content
            )
        elif isinstance(selection, dict) and selection.get("selected_field") == "r0_net":
            credibility_level = selection.get("credibility_level") or (
                "偏高" if selection.get("credibility") == "high"
                else ("偏低" if selection.get("credibility") == "low" else "中等偏低")
            )
            credibility_reason = selection.get("credibility_reason") or "统计口径参考"
            header = (
                f"【大单/主力资金统计口径参考（参考可信度：{credibility_level}）】"
                f"说明：平台主力/大单指标系公开统计口径代理参考，具参考价值，非账户级主力真实身份与资金流向终局定性。"
                f"{credibility_reason}\n\n"
            )
            full_content = header + full_content
        _elapsed = _time.monotonic() - _t0
        _meta = getattr(_last_chunk, "response_metadata", {}) or {}
        _usage = _meta.get("token_usage") or _meta.get("usage") or {}
        log_llm_call(
            agent_name="Smart Money Analyst",
            model_name=getattr(llm, "model_name", None) or getattr(llm, "model", None),
            finish_reason=_meta.get("finish_reason"),
            prompt_tokens=_usage.get("prompt_tokens"),
            completion_tokens=_usage.get("completion_tokens"),
            total_tokens=_usage.get("total_tokens"),
            elapsed_seconds=round(_elapsed, 2),
            response_chars=len(full_content),
            degraded=full_content.endswith("本项不可用"),
        )
        verdict, confidence = extract_verdict(full_content)
        consensus_guard.update({
            "blocked": consensus_blocked,
            "direction_allowed": not consensus_blocked,
            "status": selection.get("status", "not_checked") if isinstance(selection, dict) else "not_checked",
            "validation": validation,
            "selection": selection,
        })
        consensus_guard.pop("consensus", None)
        if isinstance(selection, dict):
            for key in (
                "selected_source",
                "selected_source_family",
                "selected_algorithm_group",
                "selected_field",
                "selected_value",
                "selected_unit",
                "selected_direction",
                "selected_as_of",
                "selected_period_kind",
                "selected_time_window",
                "selected_window_days",
                "fallback_rank",
                "legacy_reference",
                "legacy_web_algorithm",
                "selection_reason",
                "credibility",
                "credibility_score",
                "credibility_level",
                "credibility_reason",
                "single_source",
                "divergence",
                "reference_only",
                "large_order_credibility",
            ):
                if key in selection:
                    consensus_guard[key] = selection[key]
        return {
            "smart_money_report": full_content,
            "fund_flow_consensus_guard": consensus_guard,
            "analyst_traces": [{
                "agent": "smart_money_analyst",
                "horizon": research_horizon,
                "research_horizon": research_horizon,
                "observation_horizon": observation_horizon,
                "data_window": "近期可用",
                "key_finding": f"主力资金分析结论：{verdict}",
                "verdict": verdict,
                "confidence": confidence,
            }],
        }

    return smart_money_analyst_node
