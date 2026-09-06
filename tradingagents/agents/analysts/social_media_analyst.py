import asyncio
import logging
import time as _time

from langchain_core.messages import HumanMessage, SystemMessage
from tradingagents.dataflows.config import get_config
from tradingagents.prompts import get_prompt
from tradingagents.graph.intent_parser import (
    build_horizon_context,
    get_bound_research_horizon,
)
from tradingagents.agents.utils.context_utils import get_cn_stock_name
from tradingagents.agents.utils.agent_states import (
    TraceItem,
    current_tracker_var,
    extract_verdict,
    check_llm_output_degraded,
    check_stream_chunk_degraded,
)
from tradingagents.dataflows.social.analyst_adapter import (
    resolve_social_analyst_inputs,
    resolve_social_mode,
)
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


def create_social_media_analyst(llm, data_collector=None):
    async def _safe(tool, payload):
        try:
            if hasattr(tool, "invoke"):
                return await asyncio.to_thread(tool.invoke, payload)
            elif callable(tool):
                return await asyncio.to_thread(tool, **payload)
            return str(tool)
        except Exception as exc:
            return f"调用失败：{exc}"

    async def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        stock_name = get_cn_stock_name(ticker)
        ticker_display = f"{ticker} ({stock_name})" if stock_name and stock_name != ticker else ticker
        observation_horizon = "short"  # 舆情面专业观察窗固定为短期
        research_horizon = _resolve_research_horizon(state)
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])

        config = get_config()
        system_message = get_prompt("social_system_message", config=config)
        horizon_ctx = build_horizon_context(
            observation_horizon,
            focus_areas,
            specific_questions,
            agent_type="social",
            research_horizon=research_horizon,
        )

        pool = data_collector.get(ticker, current_date) if data_collector else None

        social_data_context = state.get("social_data_context") or (
            pool.get("social_data_context") if isinstance(pool, dict) else None
        )
        market_data_context = state.get("market_data_context") or (
            pool.get("market_data_context") if isinstance(pool, dict) else None
        )

        mode = resolve_social_mode(
            mode=state.get("mode"),
            social_data_context=social_data_context,
            config=config,
        )

        market_attention_fallback = None

        # Check if market_attention is present in context/pool
        has_market_attention = False
        if market_data_context and isinstance(market_data_context, dict) and "market_attention" in market_data_context:
            has_market_attention = True
        elif pool and isinstance(pool, dict) and (
            "market_attention" in pool
            or ("market_data_context" in pool and isinstance(pool["market_data_context"], dict) and "market_attention" in pool["market_data_context"])
        ):
            has_market_attention = True

        if not has_market_attention and pool is None:
            from tradingagents.agents.utils.agent_utils import get_zt_pool, get_hot_stocks_xq
            zt_res, hot_res = await asyncio.gather(
                _safe(get_zt_pool, {"date": current_date}),
                _safe(get_hot_stocks_xq, {"curr_date": current_date}),
            )
            market_attention_fallback = {
                "zt_pool": {"status": "available", "as_of": current_date, "requested_as_of": current_date, "raw": zt_res},
                "hot_stocks": {"status": "available", "as_of": current_date, "requested_as_of": current_date, "raw": hot_res},
            }

        resolved = resolve_social_analyst_inputs(
            mode=mode,
            social_data_context=social_data_context,
            market_data_context=market_data_context,
            pool=pool,
            market_attention=market_attention_fallback,
            ticker=ticker,
            current_date=current_date,
            ticker_display=ticker_display,
            horizon_ctx=horizon_ctx,
            state=state,
            config=config,
        )

        messages = [
            SystemMessage(content=(
                system_message
                + "\n\n请严格基于提供的舆情数据输出报告，全程使用中文。"
            )),
            HumanMessage(content=resolved.human_content),
        ]

        # ── 实现 Token 级流式输出（含降级保障） ──────────────────
        tracker = current_tracker_var.get()
        full_content = ""
        _last_chunk = None
        _t0 = _time.monotonic()

        try:
            async for chunk in llm.astream(messages):
                _last_chunk = chunk
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_content += content
                if check_stream_chunk_degraded(full_content, "Social Analyst"):
                    break
                if tracker:
                    tracker._emit_token("Social Analyst", "sentiment_report", content)
        except Exception as exc:
            logger.debug("[Social Analyst] Stream error: %s", exc)

        if not full_content.strip():
            logger.debug("[Social Analyst] Stream yielded empty text, attempting invoke fallback...")
            try:
                res = await asyncio.to_thread(llm.invoke, messages)
                full_content = res.content if hasattr(res, "content") else str(res)
                if tracker:
                    tracker._emit_token("Social Analyst", "sentiment_report", full_content)
            except Exception as exc:
                full_content = f"分析报告生成失败：{exc}"

        if check_llm_output_degraded(full_content, "Social Analyst"):
            full_content = "舆情分析生成异常（输出退化），本项不可用"

        _elapsed = _time.monotonic() - _t0
        _meta = getattr(_last_chunk, "response_metadata", {}) or {}
        _usage = _meta.get("token_usage") or _meta.get("usage") or {}
        log_llm_call(
            agent_name="Social Analyst",
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
        key_finding = (
            f"舆情分析结论：{verdict}（社交数据不可用/方向不可判断）"
            if not resolved.direction_allowed
            else f"舆情分析结论：{verdict}"
        )

        trace_item: TraceItem = {
            "agent": "social_media_analyst",
            "horizon": research_horizon,
            "research_horizon": research_horizon,
            "observation_horizon": observation_horizon,
            "data_window": "7天",
            "key_finding": key_finding,
            "verdict": verdict,
            "confidence": confidence,
            "source_status": resolved.source_status,
            "source_mode": resolved.source_mode,
            "bundle_id": resolved.bundle_id or "",
            "direction_allowed": resolved.direction_allowed,
            "reason_codes": resolved.reason_codes,
            "evidence_refs": resolved.evidence_refs,
        }

        return {
            "sentiment_report": full_content,
            "analyst_traces": [trace_item],
        }

    return social_media_analyst_node
