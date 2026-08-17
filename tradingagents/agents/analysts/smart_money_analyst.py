import logging
from tradingagents.agents.utils.context_utils import get_cn_stock_name
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
from tradingagents.graph.intent_parser import build_horizon_context
from tradingagents.agents.utils.agent_states import current_tracker_var, extract_verdict, check_llm_output_degraded, check_stream_chunk_degraded
from api.database import log_llm_call

logger = logging.getLogger(__name__)


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
        horizon = "short"  # 资金面固定短期视角
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])

        config = get_config()
        system_message = get_prompt("smart_money_system_message", config=config) or ""
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, agent_type="smart_money")

        pool = data_collector.get(ticker, current_date) if data_collector else None
        state_market_data_context = state.get("market_data_context")

        if pool is not None:
            fund_flow = pool.get("fund_flow_individual", "无数据")
            pool_context = pool.get("market_data_context")
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
            fund_flow_evidence = {}

        selection: dict = {}
        if isinstance(fund_flow_evidence, dict):
            selection = fund_flow_evidence.get("selection") or {}
            if not isinstance(selection, dict) or "selected_source" not in selection:
                legacy_consensus = fund_flow_evidence.get("consensus") or {}
                if isinstance(legacy_consensus, dict) and "selected_source" in legacy_consensus:
                    selection = legacy_consensus
        if not isinstance(selection, dict) or "selected_source" not in selection:
            records = fund_flow_evidence.get("records") or [] if isinstance(fund_flow_evidence, dict) else []
            selection = select_fund_flow_source(
                records,
                symbol=ticker,
                requested_as_of=current_date,
            )
            if isinstance(fund_flow_evidence, dict):
                fund_flow_evidence["selection"] = selection
                fund_flow_evidence["consensus"] = selection
        consensus = selection
        evidence_text = json.dumps(fund_flow_evidence, ensure_ascii=False, sort_keys=True)
        consensus_instruction = consensus_prompt_instruction(selection)
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
            "consensus": selection,
            "validation": validation,
            "reason": (validation or {}).get("hard_guard", {}).get("reason")
            or (selection or {}).get("reason", "fund-flow source selection unavailable"),
        }
        messages = [
            SystemMessage(content=(
                system_message
                + "\n\n请严格基于提供的量化数据输出分析，全程使用中文。"
            )),
            HumanMessage(content=(
                horizon_ctx + "\n"
                f"请分析 {ticker_display} 在 {current_date} 的资金流数据。若来源为同花顺即时资金流净额快照，"
                "不得将其视为新浪历史 netamount/r0_net 同口径的主力序列。\n\n"
                f"【资金流数据（来源、日期与口径见数据）】\n{fund_flow}\n\n"
                f"【资金流结构化 evidence（仅用于精确累计，不得从展示文本反推）】\n{evidence_text}\n\n"
                f"【资金流来源选择与方向规则】\n{consensus_instruction}\n\n"
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
            fund_flow_evidence["validation"] = validate_model_summary(
                fund_flow_evidence.get("records", []),
                full_content,
                window_days=5,
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
                and not selection.get("hard_guard", {}).get("blocked")
            )
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
            "consensus": consensus,
        })
        if isinstance(selection, dict):
            for key in (
                "selected_source",
                "selected_source_family",
                "selected_algorithm_group",
                "selected_field",
                "selected_value",
                "selected_unit",
                "selected_direction",
                "fallback_rank",
                "legacy_reference",
                "legacy_web_algorithm",
                "selection_reason",
            ):
                if key in selection:
                    consensus_guard[key] = selection[key]
        return {
            "smart_money_report": full_content,
            "fund_flow_consensus_guard": consensus_guard,
            "analyst_traces": [{
                "agent": "smart_money_analyst",
                "horizon": horizon,
                "data_window": "近期可用",
                "key_finding": f"主力资金分析结论：{verdict}",
                "verdict": verdict,
                "confidence": confidence,
            }],
        }

    return smart_money_analyst_node
