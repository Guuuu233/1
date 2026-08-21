import logging
import asyncio
import time as _time

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


def create_fundamentals_analyst(llm, data_collector=None):
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

    async def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        stock_name = get_cn_stock_name(ticker)

        ticker_display = f"{ticker} ({stock_name})" if stock_name and stock_name != ticker else ticker
        logger.debug("[Fundamentals Analyst] START %s %s", ticker_display, current_date)
        horizon = "medium"  # 基本面固定中长期视角
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])

        config = get_config()
        system_message = get_prompt("fundamentals_system_message", config=config) or ""
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, agent_type="fundamentals")

        pool = data_collector.get(ticker, current_date) if data_collector else None

        if pool is not None:
            outputs = {
                k: pool.get(k, "无数据")
                for k in ["fundamentals", "balance_sheet", "cashflow", "income_statement"]
            }
            global_indices = pool.get("global_indices", "无数据")
            major_assets = pool.get("major_assets", "无数据")
            cn_indices = pool.get("cn_indices", "无数据")
            industry_linkage_data = pool.get("industry_linkage")
        else:
            from tradingagents.agents.utils.agent_utils import (
                get_fundamentals,
                get_balance_sheet,
                get_cashflow,
                get_income_statement,
            )
            tasks = {
                "fundamentals": _safe(get_fundamentals, {"ticker": ticker, "curr_date": current_date}),
                "balance_sheet": _safe(get_balance_sheet, {"ticker": ticker, "freq": "quarterly", "curr_date": current_date}),
                "cashflow": _safe(get_cashflow, {"ticker": ticker, "freq": "quarterly", "curr_date": current_date}),
                "income_statement": _safe(get_income_statement, {"ticker": ticker, "freq": "quarterly", "curr_date": current_date}),
                "global_indices": _fetch_optional_tool("get_global_indices", {"curr_date": current_date}),
                "major_assets": _fetch_optional_tool("get_major_assets", {"curr_date": current_date}),
                "cn_indices": _fetch_optional_tool("get_cn_indices", {"curr_date": current_date}),
            }
            keys = list(tasks.keys())
            results = await asyncio.gather(*[tasks[k] for k in keys])
            res_dict = dict(zip(keys, results))
            outputs = {k: res_dict[k] for k in ["fundamentals", "balance_sheet", "cashflow", "income_statement"]}
            global_indices = res_dict.get("global_indices", "无数据")
            major_assets = res_dict.get("major_assets", "无数据")
            cn_indices = res_dict.get("cn_indices", "无数据")

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
                    logger.warning("[Fundamentals Analyst] 获取产业链数据异常: %s", exc)
                    industry_linkage_data = None
            else:
                industry_linkage_data = None

        # ── 行业常识知识库与宏观情景挂载 ──────────────────
        combined_text = "\n".join(
            str(v) for v in outputs.values() if v and v != "无数据"
        )
        _, industry_ctx = resolve_industry_context(
            ticker=ticker,
            stock_name=stock_name,
            extra_text=combined_text,
            state=state,
        )
        _, macro_event_ctx = resolve_macro_event_context(
            text=combined_text,
            max_events=1,
        )

        # ── 产业链联想数据段落 ──────────────────────
        industry_linkage_text = format_industry_linkage_for_prompt(industry_linkage_data)

        human_content_blocks = [
            horizon_ctx + "\n" + f"以下是 {ticker_display} 在 {current_date} 的基本面资料与产业链/宏观背景。",
        ]

        if industry_linkage_text:
            human_content_blocks.append(f"{industry_linkage_text}")

        if industry_ctx:
            human_content_blocks.append(f"{industry_ctx}")

        if global_indices != "无数据" or major_assets != "无数据" or cn_indices != "无数据":
            macro_blocks = []
            if major_assets != "无数据":
                macro_blocks.append(f"大类资产与商品（成本端/通胀参考）：\n{major_assets}")
            if cn_indices != "无数据":
                macro_blocks.append(f"国内大盘核心指数：\n{cn_indices}")
            if global_indices != "无数据":
                macro_blocks.append(f"全球市场核心指数：\n{global_indices}")
            if macro_blocks:
                human_content_blocks.append("【大类资产与宏观大盘背景】\n" + "\n\n".join(macro_blocks))

        if macro_event_ctx:
            human_content_blocks.append(f"{macro_event_ctx}")

        human_content_blocks.extend([
            f"【get_fundamentals】\n{outputs['fundamentals']}",
            f"【get_balance_sheet】\n{outputs['balance_sheet']}",
            f"【get_cashflow】\n{outputs['cashflow']}",
            f"【get_income_statement】\n{outputs['income_statement']}",
        ])

        messages = [
            SystemMessage(content=(
                system_message
                + "\n\n请严格基于提供的数据输出报告，全程使用中文。"
            )),
            HumanMessage(content="\n\n".join(human_content_blocks)),
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
                if check_stream_chunk_degraded(full_content, "Fundamentals Analyst"):
                    break
                if tracker:
                    tracker._emit_token("Fundamentals Analyst", "fundamentals_report", content)
        except Exception as exc:
            logger.debug("[Fundamentals Analyst] Stream error: %s", exc)

        if not full_content.strip():
            logger.debug("[Fundamentals Analyst] Stream yielded empty text, attempting invoke fallback...")
            try:
                res = await asyncio.to_thread(llm.invoke, messages)
                full_content = res.content if hasattr(res, "content") else str(res)
                if tracker:
                    tracker._emit_token("Fundamentals Analyst", "fundamentals_report", full_content)
            except Exception as exc:
                full_content = f"分析报告生成失败：{exc}"

        logger.debug("[Fundamentals Analyst] DONE %s, report length=%s", ticker_display, len(full_content))
        if check_llm_output_degraded(full_content, "Fundamentals Analyst"):
            full_content = "基本面分析生成异常（输出退化），本项不可用"
        _elapsed = _time.monotonic() - _t0
        _meta = getattr(_last_chunk, "response_metadata", {}) or {}
        _usage = _meta.get("token_usage") or _meta.get("usage") or {}
        log_llm_call(
            agent_name="Fundamentals Analyst",
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
            "fundamentals_report": full_content,
            "analyst_traces": [{
                "agent": "fundamentals_analyst",
                "horizon": horizon,
                "data_window": "财报周期",
                "key_finding": f"基本面分析结论：{verdict}",
                "verdict": verdict,
                "confidence": confidence,
            }],
        }

    return fundamentals_analyst_node
