import logging
from tradingagents.agents.utils.context_utils import get_cn_stock_name, format_phase1_reports
import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from tradingagents.dataflows.config import get_config
from tradingagents.prompts import get_prompt
from tradingagents.graph.intent_parser import build_horizon_context
from tradingagents.agents.utils.agent_states import (
    current_tracker_var,
    extract_verdict,
    check_llm_output_degraded,
    check_stream_chunk_degraded,
)
from tradingagents.agents.utils.knowledge_context import resolve_macro_event_context
from api.database import log_llm_call

logger = logging.getLogger(__name__)


def create_news_analyst(llm, data_collector=None):
    async def _safe(tool, payload):
        try:
            if hasattr(tool, "invoke"):
                return await asyncio.to_thread(tool.invoke, payload)
            elif callable(tool):
                return await asyncio.to_thread(tool, **payload)
            return str(tool)
        except Exception as exc:
            return f"调用失败：{exc}"

    async def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        stock_name = get_cn_stock_name(ticker)

        ticker_display = f"{ticker} ({stock_name})" if stock_name and stock_name != ticker else ticker
        horizon = "short"  # 新闻面固定短期视角
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])

        config = get_config()
        system_message = get_prompt("news_system_message", config=config) or ""
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, agent_type="news")

        pool = data_collector.get(ticker, current_date) if data_collector else None

        if pool is not None:
            data_window = pool.get("_data_window", "14天" if horizon == "short" else "90天")
            stock_news = pool.get("news", "无数据")
            global_news = pool.get("global_news", "无数据")
        else:
            from datetime import datetime, timedelta
            from tradingagents.agents.utils.agent_utils import get_news, get_global_news
            days = 14 if horizon == "short" else 30
            end_dt = datetime.strptime(current_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=days)

            # Parallelize fallback fetches
            results = await asyncio.gather(
                _safe(get_news, {
                    "ticker": ticker, "start_date": start_dt.strftime("%Y-%m-%d"), "end_date": current_date,
                }),
                _safe(get_global_news, {
                    "curr_date": current_date, "look_back_days": days, "limit": 10,
                })
            )
            stock_news, global_news = results
            data_window = f"{days}天"

        # ── 宏观事件情景图谱挂载 ──────────────────
        extra_event_text = f"{stock_news}\n{global_news}"
        macro_report = state.get("macro_report", "")
        if macro_report and macro_report != "无数据":
            extra_event_text += f"\n{macro_report}"
        _, macro_event_ctx = resolve_macro_event_context(
            text=extra_event_text,
            max_events=2,
        )

        phase1_reports_text = format_phase1_reports(state)

        human_content_blocks = [
            horizon_ctx + "\n" + f"以下是 {ticker_display} 在 {current_date} 的新闻资料（{data_window}）。",
            phase1_reports_text,
            f"【get_news】\n{stock_news}",
            f"【get_global_news】\n{global_news}",
        ]
        if macro_event_ctx:
            human_content_blocks.append(f"【宏观事件传导图谱】\n{macro_event_ctx}")

        messages = [
            SystemMessage(content=(
                system_message
                + "\n\n请严格基于提供的新闻资料输出报告，全程使用中文。"
            )),
            HumanMessage(content="\n\n".join(human_content_blocks)),
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
                if check_stream_chunk_degraded(full_content, "News Analyst"):
                    break
                if tracker:
                    tracker._emit_token("News Analyst", "news_report", content)
        except Exception as exc:
            logger.debug("[News Analyst] Stream error: %s", exc)

        if not full_content.strip():
            logger.debug("[News Analyst] Stream yielded empty text, attempting invoke fallback...")
            try:
                res = await asyncio.to_thread(llm.invoke, messages)
                full_content = res.content if hasattr(res, "content") else str(res)
                if tracker:
                    tracker._emit_token("News Analyst", "news_report", full_content)
            except Exception as exc:
                full_content = f"分析报告生成失败：{exc}"

        if check_llm_output_degraded(full_content, "News Analyst"):
            full_content = "新闻分析生成异常（输出退化），本项不可用"
        _elapsed = _time.monotonic() - _t0
        _meta = getattr(_last_chunk, "response_metadata", {}) or {}
        _usage = _meta.get("token_usage") or _meta.get("usage") or {}
        log_llm_call(
            agent_name="News Analyst",
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
        return {
            "news_report": full_content,
            "analyst_traces": [{
                "agent": "news_analyst",
                "horizon": horizon,
                "data_window": data_window,
                "key_finding": f"新闻分析结论：{verdict}",
                "verdict": verdict,
                "confidence": confidence,
            }],
        }

    return news_analyst_node
