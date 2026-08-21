import logging
import asyncio
import time as _time
from datetime import datetime, timedelta

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
from tradingagents.agents.utils.context_utils import get_cn_stock_name
from tradingagents.agents.utils.knowledge_context import (
    resolve_industry_context,
    resolve_macro_event_context,
)
from tradingagents.dataflows.industry_linkage import (
    format_industry_linkage_for_prompt,
)
from tradingagents.graph.data_collector import _map_stock_to_industry
from api.database import log_llm_call

logger = logging.getLogger(__name__)


def create_macro_analyst(llm, data_collector=None):
    async def _safe(tool, payload):
        try:
            if hasattr(tool, "invoke"):
                return await asyncio.to_thread(tool.invoke, payload)
            elif callable(tool):
                return await asyncio.to_thread(tool, **payload)
            return str(tool)
        except Exception as exc:
            return f"调用失败：{exc}"

    async def _fetch_optional_tool(tool_name: str, payload: dict) -> str:
        try:
            from tradingagents.agents.utils import agent_utils
            tool = getattr(agent_utils, tool_name, None)
            if tool is None:
                from tradingagents.dataflows import interface
                tool = getattr(interface, tool_name, None)
            if tool is not None:
                return await _safe(tool, payload)
        except Exception as exc:
            return f"调用失败：{exc}"
        return "无数据"

    async def macro_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        stock_name = get_cn_stock_name(ticker)

        ticker_display = f"{ticker} ({stock_name})" if stock_name and stock_name != ticker else ticker
        logger.debug("[Macro Analyst] START %s %s", ticker_display, current_date)
        horizon = "medium"  # 宏观面固定中长期视角
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])

        config = get_config()
        system_message = get_prompt("macro_system_message", config=config) or ""
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, agent_type="macro")

        pool = data_collector.get(ticker, current_date) if data_collector else None

        if pool is not None:
            board_flow = pool.get("fund_flow_board", "无数据")
            recent_news = pool.get("news", "无数据")
            global_news = pool.get("global_news", "无数据")
            global_indices = pool.get("global_indices", "无数据")
            major_assets = pool.get("major_assets", "无数据")
            cn_indices = pool.get("cn_indices", "无数据")
            northbound_flow = pool.get("northbound_flow", "无数据")
            industry_linkage_data = pool.get("industry_linkage")
        else:
            days = 7
            end_dt = datetime.strptime(current_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=days)
            from tradingagents.agents.utils.agent_utils import (
                get_board_fund_flow,
                get_news,
                get_global_news,
                get_northbound_flow,
            )

            # Parallelize fallback fetches
            results = await asyncio.gather(
                _safe(get_board_fund_flow, {"curr_date": current_date}),
                _safe(get_news, {
                    "ticker": ticker, "start_date": start_dt.strftime("%Y-%m-%d"), "end_date": current_date,
                }),
                _safe(get_global_news, {
                    "curr_date": current_date, "look_back_days": 14, "limit": 15,
                }),
                _safe(get_northbound_flow, {"symbol": ticker, "curr_date": current_date}),
                _fetch_optional_tool("get_global_indices", {"curr_date": current_date}),
                _fetch_optional_tool("get_major_assets", {"curr_date": current_date}),
                _fetch_optional_tool("get_cn_indices", {"curr_date": current_date}),
            )
            (
                board_flow,
                recent_news,
                global_news,
                northbound_flow,
                global_indices,
                major_assets,
                cn_indices,
            ) = results

            # Fallback 获取产业链数据
            mapped_ind = _map_stock_to_industry(ticker)
            if mapped_ind:
                try:
                    from tradingagents.dataflows.providers.industry_linkage_provider import (
                        IndustryLinkageProvider,
                    )
                    _provider = IndustryLinkageProvider()
                    industry_linkage_data = _provider.get_industry_linkage(mapped_ind, as_of=current_date)
                except Exception as exc:
                    logger.warning("[Macro Analyst] 获取产业链数据异常: %s", exc)
                    industry_linkage_data = None
            else:
                industry_linkage_data = None

        # ── 知识库与宏观情景图谱挂载 ──────────────────
        combined_text = f"{board_flow}\n{recent_news}\n{global_news}"
        _, industry_ctx = resolve_industry_context(
            ticker=ticker,
            stock_name=stock_name,
            extra_text=combined_text,
            state=state,
            fallback_on_miss=False,
        )
        _, macro_event_ctx = resolve_macro_event_context(
            text=f"{recent_news}\n{global_news}",
            max_events=2,
            fallback_on_miss=False,
        )

        # ── 产业链联想数据段落 ──────────────────────
        industry_linkage_text = format_industry_linkage_for_prompt(industry_linkage_data)

        # ── 组装 HumanMessage ────────────────────────
        human_content_lines = [
            horizon_ctx + "\n" + f"请分析 {ticker_display} 在 {current_date} 的宏观与板块环境。",
            f"【今日行业板块资金流向】\n{board_flow}",
            f"【近期相关新闻】\n{recent_news}",
        ]

        if industry_linkage_text:
            human_content_lines.append(f"{industry_linkage_text}")

        if global_indices != "无数据" or major_assets != "无数据" or cn_indices != "无数据":
            macro_view_blocks = []
            if global_indices != "无数据":
                macro_view_blocks.append(f"【全球核心指数】\n{global_indices}")
            if major_assets != "无数据":
                macro_view_blocks.append(f"【大类资产与宏观商品】\n{major_assets}")
            if cn_indices != "无数据":
                macro_view_blocks.append(f"【国内大盘核心指数】\n{cn_indices}")
            if macro_view_blocks:
                human_content_lines.append("\n\n".join(macro_view_blocks))

        if northbound_flow != "无数据":
            human_content_lines.append(f"【北向资金与跨市场流动性】\n{northbound_flow}")

        if global_news != "无数据":
            human_content_lines.append(f"【全球与宏观要闻】\n{global_news}")

        if industry_ctx:
            human_content_lines.append(f"{industry_ctx}")
        else:
            human_content_lines.append("【行业常识知识库】\n【知识库未命中】")

        if macro_event_ctx:
            human_content_lines.append(f"{macro_event_ctx}")
        else:
            human_content_lines.append("【宏观事件传导图谱】\n【知识库未命中】")

        messages = [
            SystemMessage(content=(
                system_message
                + "\n\n请严格基于提供的数据输出报告，全程使用中文。"
            )),
            HumanMessage(content="\n\n".join(human_content_lines)),
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
                if check_stream_chunk_degraded(full_content, "Macro Analyst"):
                    break
                if tracker:
                    tracker._emit_token("Macro Analyst", "macro_report", content)
        except Exception as exc:
            logger.debug("[Macro Analyst] Stream error: %s", exc)

        if not full_content.strip():
            logger.debug("[Macro Analyst] Stream yielded empty text, attempting invoke fallback...")
            try:
                res = await asyncio.to_thread(llm.invoke, messages)
                full_content = res.content if hasattr(res, "content") else str(res)
                if tracker:
                    tracker._emit_token("Macro Analyst", "macro_report", full_content)
            except Exception as exc:
                full_content = f"分析报告生成失败：{exc}"

        logger.debug("[Macro Analyst] DONE %s, report length=%s", ticker_display, len(full_content))
        if check_llm_output_degraded(full_content, "Macro Analyst"):
            full_content = "宏观板块分析生成异常（输出退化），本项不可用"
        _elapsed = _time.monotonic() - _t0
        _meta = getattr(_last_chunk, "response_metadata", {}) or {}
        _usage = _meta.get("token_usage") or _meta.get("usage") or {}
        log_llm_call(
            agent_name="Macro Analyst",
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
            "macro_report": full_content,
            "analyst_traces": [{
                "agent": "macro_analyst",
                "horizon": horizon,
                "data_window": "板块数据",
                "key_finding": f"宏观板块分析结论：{verdict}",
                "verdict": verdict,
                "confidence": confidence,
            }],
        }

    return macro_analyst_node
