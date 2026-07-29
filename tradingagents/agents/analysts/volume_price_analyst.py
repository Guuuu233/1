from tradingagents.agents.utils.context_utils import get_cn_stock_name
from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.dataflows.config import get_config
from tradingagents.prompts import get_prompt
from tradingagents.graph.intent_parser import build_horizon_context
from tradingagents.agents.utils.agent_states import current_tracker_var, extract_verdict, check_llm_output_degraded
from api.database import log_llm_call


def create_volume_price_analyst(llm, data_collector=None):
    async def volume_price_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        stock_name = get_cn_stock_name(ticker)

        ticker_display = f"{ticker} ({stock_name})" if stock_name and stock_name != ticker else ticker
        horizon = "short"
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])

        config = get_config()
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, agent_type="volume_price")
        system_message = get_prompt("volume_price_system_message", config=config)

        if data_collector is not None:
            pool = data_collector.get(ticker, current_date)
            if pool is not None:
                windowed = data_collector.get_window(pool, horizon, current_date)
                vpa_data = windowed.get("vpa_indicators", "无数据")
                stock_data = windowed.get("stock_data", "无数据")
                data_window = windowed.get("_data_window", "14天")
            else:
                vpa_data, stock_data, data_window = "无数据", "无数据", "14天"
        else:
            vpa_data, stock_data, data_window = "无数据", "无数据", "14天"

        messages = [
            SystemMessage(content=horizon_ctx + system_message + "\n\n请全程使用中文。"),
            HumanMessage(content=(
                f"以下是 {ticker_display} 在 {current_date} 的量价分析预计算数据（数据窗口：{data_window}）。\n\n"
                f"{vpa_data}\n\n"
                f"【原始 K 线数据参考】\n{stock_data}"
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
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_content += content
                if tracker:
                    tracker._emit_token("Volume Price Analyst", "volume_price_report", content)
        except Exception as exc:
            print(f"[Volume Price Analyst] Stream error: {exc}")

        if not full_content.strip():
            print("[Volume Price Analyst] Stream yielded empty text, attempting invoke fallback...")
            try:
                res = await asyncio.to_thread(llm.invoke, messages)
                full_content = res.content if hasattr(res, "content") else str(res)
                if tracker:
                    tracker._emit_token("Volume Price Analyst", "volume_price_report", full_content)
            except Exception as exc:
                full_content = f"分析报告生成失败：{exc}"

        if check_llm_output_degraded(full_content, "Volume Price Analyst"):
            full_content = "量价分析生成异常（输出退化），本项不可用"
        _elapsed = _time.monotonic() - _t0
        _meta = getattr(_last_chunk, "response_metadata", {}) or {}
        _usage = _meta.get("token_usage") or _meta.get("usage") or {}
        log_llm_call(
            agent_name="Volume Price Analyst",
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
            "volume_price_report": full_content,
            "analyst_traces": [{
                "agent": "volume_price_analyst",
                "horizon": horizon,
                "data_window": data_window,
                "key_finding": f"量价分析结论：{verdict}",
                "verdict": verdict,
                "confidence": confidence,
            }],
        }

    return volume_price_analyst_node
