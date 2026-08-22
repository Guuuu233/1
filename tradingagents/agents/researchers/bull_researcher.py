import logging
from tradingagents.dataflows.config import get_config
from tradingagents.prompts import get_prompt
from tradingagents.graph.intent_parser import build_horizon_context
from tradingagents.agents.utils.agent_states import current_tracker_var
from tradingagents.agents.utils.debate_utils import (
    build_debate_report_manifest,
    format_claim_subset_for_prompt,
    format_claims_for_prompt,
    update_debate_state_with_payload,
)
from tradingagents.agents.utils.prompt_injection import build_injection_slots, Placement, DEFAULT_PLACEMENT

_logger = logging.getLogger(__name__)


def create_bull_researcher(llm, memory, custom_prompt: str = "", placement: Placement = DEFAULT_PLACEMENT):
    async def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        current_response = investment_debate_state.get("current_response", "")
        macro_report = state.get("macro_report", "")
        market_research_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        smart_money_report = state.get("smart_money_report", "")
        volume_price_report = state.get("volume_price_report", "")

        report_manifest = build_debate_report_manifest(state)
        _logger.info("[bull_researcher] report input manifest: %s", report_manifest)

        claims = investment_debate_state.get("claims", [])
        focus_claim_ids = investment_debate_state.get("focus_claim_ids", [])
        unresolved_claim_ids = investment_debate_state.get("unresolved_claim_ids", [])
        round_summary = investment_debate_state.get("round_summary", "")
        round_goal = investment_debate_state.get("round_goal", "")

        horizon = state.get("horizon", "medium")
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])
        horizon_ctx = build_horizon_context(horizon, focus_areas, specific_questions, agent_type="bull")

        curr_situation = (
            f"{macro_report}\n\n"
            f"{market_research_report}\n\n"
            f"{sentiment_report}\n\n"
            f"{news_report}\n\n"
            f"{fundamentals_report}\n\n"
            f"{smart_money_report}\n\n"
            f"{volume_price_report}"
        )
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        injection_slots = build_injection_slots(custom_prompt, placement, role_key="bull_researcher")
        prompt = horizon_ctx + get_prompt("bull_prompt", config=get_config()).format(
            macro_report=macro_report,
            market_research_report=market_research_report,
            sentiment_report=sentiment_report,
            news_report=news_report,
            fundamentals_report=fundamentals_report,
            smart_money_report=smart_money_report,
            volume_price_report=volume_price_report,
            history=history,
            current_response=current_response,
            past_memory_str=past_memory_str,
            focus_claims_text=format_claim_subset_for_prompt(claims, focus_claim_ids),
            unresolved_claims_text=format_claim_subset_for_prompt(claims, unresolved_claim_ids),
            claims_text=format_claims_for_prompt(claims),
            round_summary=round_summary or "暂无轮次摘要，请先建立核心多头 claim。",
            round_goal=round_goal,
            **injection_slots,
        )

        # ── 实现 Token 级流式输出 ──────────────────
        tracker = current_tracker_var.get()
        try:
            debate_round = int(investment_debate_state.get("count", 0) or 0) // 2 + 1
        except (ValueError, TypeError):
            debate_round = 1
        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
        full_content = ""
        async for chunk in llm.astream(prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_content += content
            if tracker:
                tracker._emit_token("Bull Researcher", "investment_debate_state", content)
                tracker.emit_debate_token(
                    debate="research", agent="Bull Researcher",
                    round_num=debate_round, token=content, model_name=model_name,
                )

        # ── 推送辩论完整消息（标记流式结束）──
        if tracker:
            tracker.emit_debate_message(
                debate="research", agent="Bull Researcher",
                round_num=debate_round, content=full_content, model_name=model_name,
            )

        new_investment_debate_state = update_debate_state_with_payload(
            state=investment_debate_state,
            raw_response=full_content,
            speaker_label="Bull Analyst",
            speaker_key="Bull",
            stance="bullish",
            history_key="bull_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
            model_name=model_name,
        )

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node

    return bull_node
