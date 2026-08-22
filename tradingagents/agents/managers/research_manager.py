import logging
import time

from tradingagents.dataflows.config import get_config
from tradingagents.prompts import get_prompt
from tradingagents.prompts.catalog import _resolve_language
from tradingagents.agents.utils.agent_states import current_tracker_var
from tradingagents.agents.utils.debate_utils import (
    format_claim_subset_for_prompt,
    format_claims_for_prompt,
)
from tradingagents.agents.utils.evidence_summary import build_evidence_summary
from tradingagents.agents.utils.prompt_injection import build_injection_slots, Placement, DEFAULT_PLACEMENT

_logger = logging.getLogger(__name__)


def create_research_manager(llm, memory, custom_prompt: str = "", placement: Placement = DEFAULT_PLACEMENT):
    async def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        smart_money_report = state.get("smart_money_report", "")
        volume_price_report = state.get("volume_price_report", "")
        fund_flow_guard = state.get("fund_flow_consensus_guard") or {
            "blocked": True,
            "direction_allowed": False,
            "status": "not_checked",
        }

        investment_debate_state = state["investment_debate_state"]
        claims = investment_debate_state.get("claims", [])
        unresolved_claim_ids = investment_debate_state.get("unresolved_claim_ids", [])
        round_summary = investment_debate_state.get("round_summary", "")

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        claims_text = format_claims_for_prompt(claims)
        unresolved_claims_text = format_claim_subset_for_prompt(claims, unresolved_claim_ids)
        round_summary_text = round_summary or "暂无轮次摘要。"

        # First-hand evidence access (KNOWN_ISSUES #2): adjudicators get compact
        # fact-dense excerpts of the reports they currently only use for memory
        # retrieval, so the verdict can be anchored to evidence strength. The
        # macro analyst's report was previously consumed by nobody (git log shows
        # it was added with the other analysts but never wired into a template).
        market_evidence_summary = build_evidence_summary(market_research_report)
        news_evidence_summary = build_evidence_summary(news_report)
        fundamentals_evidence_summary = build_evidence_summary(fundamentals_report)
        macro_evidence_summary = build_evidence_summary(state.get("macro_report", ""))

        # Omit the macro/sector evidence line entirely when the macro analyst
        # did not run (empty report -> empty summary). Injecting a placeholder
        # would read as "macro concluded no data" instead of "analyst not run".
        macro_evidence_line = ""
        if macro_evidence_summary:
            label = (
                "宏观/板块证据摘要："
                if _resolve_language(get_config()) == "zh"
                else "Macro/sector evidence summary: "
            )
            macro_evidence_line = f"{label}{macro_evidence_summary}"

        if fund_flow_guard.get("blocked") or not fund_flow_guard.get("direction_allowed"):
            blocked_plan = "资金流来源选择 guard 已阻断：不得输出增持、减持、吸筹或其他方向性投资计划。"
            return {
                "fund_flow_consensus_guard": fund_flow_guard,
                "investment_plan": blocked_plan,
                "investment_debate_state": {
                    **investment_debate_state,
                    "judge_decision": blocked_plan,
                    "current_response": blocked_plan,
                },
            }

        injection_slots = build_injection_slots(custom_prompt, placement, role_key="research_manager")
        prompt = get_prompt("research_manager_prompt", config=get_config()).format(
            past_memory_str=past_memory_str,
            history=history,
            smart_money_report=smart_money_report,
            volume_price_report=volume_price_report,
            sentiment_report=sentiment_report,
            market_evidence_summary=market_evidence_summary,
            news_evidence_summary=news_evidence_summary,
            fundamentals_evidence_summary=fundamentals_evidence_summary,
            macro_evidence_line=macro_evidence_line,
            claims_text=claims_text,
            unresolved_claims_text=unresolved_claims_text,
            round_summary=round_summary_text,
            **injection_slots,
        )

        _logger.info(
            "[research_manager] prompt size: total=%d chars | "
            "history=%d, smart_money=%d, volume_price=%d, sentiment=%d, "
            "evidence(market/news/fund/macro)=%d/%d/%d/%d, "
            "memory=%d, claims=%d, unresolved=%d, round_summary=%d",
            len(prompt),
            len(history or ""),
            len(smart_money_report or ""),
            len(volume_price_report or ""),
            len(sentiment_report or ""),
            len(market_evidence_summary),
            len(news_evidence_summary),
            len(fundamentals_evidence_summary),
            len(macro_evidence_summary),
            len(past_memory_str or ""),
            len(claims_text or ""),
            len(unresolved_claims_text or ""),
            len(round_summary_text or ""),
        )

        # ── 实现 Token 级流式输出 ──────────────────
        tracker = current_tracker_var.get()
        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
        full_content = ""
        reasoning_buf: list[str] = []
        first_token_at: float | None = None
        first_reasoning_at: float | None = None
        start = time.monotonic()

        async for chunk in llm.astream(prompt):
            now = time.monotonic()
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_content += content

            # reasoning_content (thinking 模型) 仅做 server 端日志，不发前端
            reasoning = None
            extra = getattr(chunk, "additional_kwargs", None) or {}
            if isinstance(extra, dict):
                reasoning = extra.get("reasoning_content")
            if reasoning:
                if first_reasoning_at is None:
                    first_reasoning_at = now
                reasoning_buf.append(reasoning)

            if content:
                if first_token_at is None:
                    first_token_at = now
                if tracker:
                    tracker._emit_token("Research Manager", "investment_plan", content)
                    tracker.emit_debate_token(
                        debate="research", agent="Research Manager",
                        round_num=-1, token=content,
                    )

        total_elapsed = time.monotonic() - start
        reasoning_text = "".join(reasoning_buf)
        _logger.info(
            "[research_manager] streaming done: total_elapsed=%.2fs | "
            "ttft_reasoning=%.2fs ttft_content=%.2fs | "
            "reasoning_chars=%d content_chars=%d",
            total_elapsed,
            (first_reasoning_at - start) if first_reasoning_at else -1,
            (first_token_at - start) if first_token_at else -1,
            len(reasoning_text),
            len(full_content),
        )
        if reasoning_text:
            _logger.debug(
                "[research_manager] reasoning preview (%d chars): %s",
                len(reasoning_text),
                reasoning_text[:1500],
            )

        # ── 推送辩论裁决（标记流式结束）──
        if tracker:
            tracker.emit_debate_message(
                debate="research", agent="Research Manager",
                round_num=-1, content=full_content, is_verdict=True,
            )

        new_investment_debate_state = {
            "judge_decision": full_content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_speaker": investment_debate_state.get("current_speaker", ""),
            "current_response": full_content,
            "count": investment_debate_state["count"],
            "claims": claims,
            "round_messages": investment_debate_state.get("round_messages", []),
            "focus_claim_ids": investment_debate_state.get("focus_claim_ids", []),
            "open_claim_ids": investment_debate_state.get("open_claim_ids", []),
            "resolved_claim_ids": investment_debate_state.get("resolved_claim_ids", []),
            "unresolved_claim_ids": unresolved_claim_ids,
            "round_summary": round_summary,
            "round_goal": investment_debate_state.get("round_goal", ""),
            "claim_counter": investment_debate_state.get("claim_counter", 0),
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": full_content,
        }

    return research_manager_node
