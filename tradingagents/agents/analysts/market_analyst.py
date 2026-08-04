import logging
from tradingagents.agents.utils.context_utils import get_cn_stock_name
import asyncio
import json
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.dataflows.config import get_config
from tradingagents.prompts import get_prompt
from tradingagents.graph.intent_parser import build_horizon_context
from tradingagents.agents.utils.agent_states import current_tracker_var, extract_verdict, check_llm_output_degraded, check_stream_chunk_degraded
from api.database import log_llm_call

logger = logging.getLogger(__name__)

# List of technical indicators to retrieve
MARKET_INDICATORS = [
    "close_50_sma",
    "close_200_sma",
    "close_10_ema",
    "rsi",
    "macd",
    "boll",
    "boll_ub",
    "boll_lb",
    "atr",
    "vwma",
]


def _default_market_data_context() -> dict:
    return {
        "daily": {"as_of": None, "completeness": "unavailable"},
        "realtime": {
            "status": "unavailable",
            "source": None,
            "quote_as_of": None,
            "retrieved_at": None,
            "error": "实时行情上下文不可用",
            "quote": None,
        },
    }


def _format_market_data_context(context: dict) -> str:
    daily = context.get("daily") or {}
    realtime = context.get("realtime") or {}
    daily_as_of = daily.get("as_of") or "不可用"
    daily_completeness = daily.get("completeness") or "unavailable"
    return (
        f"【完整日线】截至 {daily_as_of}，完整性：{daily_completeness}。"
        "以下 K 线和指标不含盘中实时快照。\n"
        "【实时快照】\n"
        f"{json.dumps(realtime, ensure_ascii=False, sort_keys=True)}"
    )


def create_market_analyst(llm, data_collector=None):
    async def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        stock_name = get_cn_stock_name(ticker)

        ticker_display = f"{ticker} ({stock_name})" if stock_name and stock_name != ticker else ticker
        horizon = "short"  # 技术面固定短期视角
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])
        
        config = get_config()
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, agent_type="market")
        system_message = get_prompt("market_system_message", config=config)

        if data_collector is not None:
            pool = data_collector.get(ticker, current_date)
            if pool is not None:
                windowed = data_collector.get_window(pool, horizon, current_date)
                stock_data = windowed.get("stock_data", "无数据")
                indicators = windowed.get("indicators", {})
                data_window = windowed.get("_data_window", "14天")
                market_data_context = windowed.get(
                    "market_data_context", _default_market_data_context()
                )
            else:
                stock_data, indicators, data_window = await _fetch_direct(ticker, current_date, horizon)
                market_data_context = _default_market_data_context()
        else:
            stock_data, indicators, data_window = await _fetch_direct(ticker, current_date, horizon)
            market_data_context = _default_market_data_context()

        indicator_blocks = [
            f"【{ind}】\n{indicators.get(ind, '无数据')}"
            for ind in MARKET_INDICATORS
        ]

        messages = [
            SystemMessage(content=system_message + "\n\n请全程使用中文。"),
            HumanMessage(content=(
                horizon_ctx + "\n"
                f"以下是 {ticker_display} 在 {current_date} 的市场数据（数据窗口：{data_window}）。\n\n"
                f"{_format_market_data_context(market_data_context)}\n\n"
                f"【完整日线 OHLCV】\n{stock_data}\n\n"
                + "\n\n".join(indicator_blocks)
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
                if check_stream_chunk_degraded(full_content, "Market Analyst"):
                    break


                if tracker:


                    tracker._emit_token("Market Analyst", "market_report", content)


        except Exception as exc:


            logger.debug("[Market Analyst] Stream error: %s", exc)



        if not full_content.strip():


            logger.debug("[Market Analyst] Stream yielded empty text, attempting invoke fallback...")


            try:


                res = await asyncio.to_thread(llm.invoke, messages)


                full_content = res.content if hasattr(res, "content") else str(res)


                if tracker:


                    tracker._emit_token("Market Analyst", "market_report", full_content)


            except Exception as exc:


                full_content = f"分析报告生成失败：{exc}"
        
        if check_llm_output_degraded(full_content, "Market Analyst"):
            full_content = "市场技术分析生成异常（输出退化），本项不可用"
        _elapsed = _time.monotonic() - _t0
        _meta = getattr(_last_chunk, "response_metadata", {}) or {}
        _usage = _meta.get("token_usage") or _meta.get("usage") or {}
        log_llm_call(
            agent_name="Market Analyst",
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
            "market_report": full_content,
            "market_data_context": market_data_context,
            "analyst_traces": [{
                "agent": "market_analyst",
                "horizon": horizon,
                "data_window": data_window,
                "key_finding": f"市场技术面结论：{verdict}",
                "verdict": verdict,
                "confidence": confidence,
            }],
        }

    return market_analyst_node


async def _fetch_direct(ticker, current_date, horizon):
    from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators

    async def _safe(tool, payload):
        try:
            return await asyncio.to_thread(tool.invoke, payload)
        except Exception as exc:
            return f"调用失败：{exc}"

    days = 14 if horizon == "short" else 90
    end_dt = datetime.strptime(current_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=days)
    
    # Run stock data fetch and all indicator fetches in parallel
    tasks = {
        "stock_data": _safe(get_stock_data, {
            "symbol": ticker, "start_date": start_dt.strftime("%Y-%m-%d"), "end_date": current_date,
        })
    }
    for ind in MARKET_INDICATORS:
        tasks[ind] = _safe(get_indicators, {
            "symbol": ticker, "indicator": ind, "curr_date": current_date, "look_back_days": days,
        })
    
    keys = list(tasks.keys())
    results = await asyncio.gather(*[tasks[k] for k in keys])
    res_map = dict(zip(keys, results))
    
    stock_data = res_map.pop("stock_data")
    return stock_data, res_map, f"{days}天"
