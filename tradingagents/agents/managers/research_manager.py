import logging
import time

from tradingagents.dataflows.config import get_config
from tradingagents.prompts import get_prompt
from tradingagents.prompts.catalog import _resolve_language
from tradingagents.agents.utils.agent_states import current_tracker_var
from tradingagents.agents.utils.debate_utils import (
    build_debate_report_manifest,
    format_claim_subset_for_prompt,
    format_claims_for_prompt,
)
from tradingagents.agents.utils.evidence_summary import (
    build_dense_report_input,
    build_evidence_summary,
)
from tradingagents.agents.utils.evidence_verifier import (
    EvidenceFactualTruthEvaluator,
    extract_and_validate_manager_verdict,
)
from tradingagents.agents.utils.prompt_injection import build_injection_slots, Placement, DEFAULT_PLACEMENT

_logger = logging.getLogger(__name__)


def create_research_manager(llm, memory, custom_prompt: str = "", placement: Placement = DEFAULT_PLACEMENT):
    async def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        macro_report = state.get("macro_report", "")
        market_research_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        smart_money_report = state.get("smart_money_report", "")
        volume_price_report = state.get("volume_price_report", "")
        fund_flow_guard = state.get("fund_flow_consensus_guard") or {
            "blocked": True,
            "direction_allowed": False,
            "status": "not_checked",
        }
        market_data_context = state.get("market_data_context") or {}
        analysis_baseline_date = (
            state.get("trade_date")
            or state.get("analysis_baseline_date")
            or (market_data_context.get("analysis_baseline_date") if isinstance(market_data_context, dict) else "")
            or (market_data_context.get("trade_date") if isinstance(market_data_context, dict) else "")
            or (market_data_context.get("data_as_of") if isinstance(market_data_context, dict) else "")
            or ""
        )
        data_gaps = state.get("data_gaps") or (market_data_context.get("data_gaps") if isinstance(market_data_context, dict) else []) or []

        investment_debate_state = state["investment_debate_state"]
        claims = investment_debate_state.get("claims", [])
        unresolved_claim_ids = investment_debate_state.get("unresolved_claim_ids", [])
        round_summary = investment_debate_state.get("round_summary", "")

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

        claims_text = format_claims_for_prompt(claims)
        unresolved_claims_text = format_claim_subset_for_prompt(claims, unresolved_claim_ids)
        round_summary_text = round_summary or "暂无轮次摘要。"

        # ── Seven Reports Input Extraction & Manifest ──────────────────────
        macro_input, macro_mode, macro_chars = build_dense_report_input(macro_report, max_chars=1800, role_name="macro")
        market_input, market_mode, market_chars = build_dense_report_input(market_research_report, max_chars=1800, role_name="market")
        sentiment_input, sentiment_mode, sentiment_chars = build_dense_report_input(sentiment_report, max_chars=1800, role_name="sentiment")
        news_input, news_mode, news_chars = build_dense_report_input(news_report, max_chars=1800, role_name="news")
        fundamentals_input, fundamentals_mode, fundamentals_chars = build_dense_report_input(fundamentals_report, max_chars=1800, role_name="fundamentals")
        smart_money_input, smart_money_mode, smart_money_chars = build_dense_report_input(smart_money_report, max_chars=1800, role_name="smart_money")
        volume_price_input, volume_price_mode, volume_price_chars = build_dense_report_input(volume_price_report, max_chars=1800, role_name="volume_price")

        seven_reports = {
            "macro_report": macro_report,
            "market_report": market_research_report,
            "sentiment_report": sentiment_report,
            "news_report": news_report,
            "fundamentals_report": fundamentals_report,
            "smart_money_report": smart_money_report,
            "volume_price_report": volume_price_report,
        }

        pass_info = {
            "macro_report": (macro_mode, macro_chars),
            "market_report": (market_mode, market_chars),
            "sentiment_report": (sentiment_mode, sentiment_chars),
            "news_report": (news_mode, news_chars),
            "fundamentals_report": (fundamentals_mode, fundamentals_chars),
            "smart_money_report": (smart_money_mode, smart_money_chars),
            "volume_price_report": (volume_price_mode, volume_price_chars),
        }
        report_manifest = build_debate_report_manifest(seven_reports, pass_info=pass_info)

        # ── Provenance & Data Failure Context ──────────────────────────────
        prov_lines = [f"- 基准分析日期 (analysis_baseline_date): {analysis_baseline_date or '未明确指定'}"]
        source_provenance = market_data_context.get("source_provenance") if isinstance(market_data_context, dict) else {}
        if isinstance(source_provenance, dict) and source_provenance:
            prov_lines.append("- 数据源可用状态 (source_provenance):")
            for src, info in source_provenance.items():
                st = info.get("status", "unknown") if isinstance(info, dict) else str(info)
                prov_lines.append(f"  * {src}: {st}")
        failure_ledger = market_data_context.get("data_failure_ledger") if isinstance(market_data_context, dict) else []
        if isinstance(failure_ledger, list) and failure_ledger:
            prov_lines.append("- 数据失败账本 (data_failure_ledger - 严禁采纳以下不可用/失败指标):")
            for entry in failure_ledger:
                if isinstance(entry, dict):
                    prov_lines.append(f"  * [UNAVAILABLE] {entry.get('source', '')}: {entry.get('reason', entry.get('status', 'failed'))}")
        if data_gaps:
            prov_lines.append(f"- 已知数据缺口 (data_gaps): {', '.join(str(g) for g in data_gaps)}")
        provenance_context = "\n".join(prov_lines)

        market_evidence_summary = build_evidence_summary(market_research_report)
        news_evidence_summary = build_evidence_summary(news_report)
        fundamentals_evidence_summary = build_evidence_summary(fundamentals_report)
        macro_evidence_summary = build_evidence_summary(macro_report)

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
            manager_verdict = {
                "direction": "中性",
                "winner": "tie",
                "reason": "资金流来源选择 guard 已阻断",
                "position_pct": 0,
                "entry": None,
                "target": None,
                "stop_loss": None,
                "upside": None,
                "downside": None,
                "odds": None,
                "adopted_claim_ids": [],
                "rejected_claim_ids": [],
                "consistency_check_passed": True,
                "failed_checks": [],
            }
            return {
                "fund_flow_consensus_guard": fund_flow_guard,
                "investment_plan": blocked_plan,
                "manager_verdict": manager_verdict,
                "evidence_verification": [],
                "report_manifest": report_manifest,
                "investment_debate_state": {
                    **investment_debate_state,
                    "judge_decision": blocked_plan,
                    "current_response": blocked_plan,
                    "manager_verdict": manager_verdict,
                    "evidence_verification": [],
                    "report_manifest": report_manifest,
                },
            }

        injection_slots = build_injection_slots(custom_prompt, placement, role_key="research_manager")
        prompt = get_prompt("research_manager_prompt", config=get_config()).format(
            past_memory_str=past_memory_str,
            provenance_context=provenance_context,
            history=history,
            smart_money_report=smart_money_input,
            volume_price_report=volume_price_input,
            sentiment_report=sentiment_input,
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
            "provenance=%d, memory=%d, claims=%d, unresolved=%d, round_summary=%d",
            len(prompt),
            len(history or ""),
            len(smart_money_input or ""),
            len(volume_price_input or ""),
            len(sentiment_input or ""),
            len(market_evidence_summary),
            len(news_evidence_summary),
            len(fundamentals_evidence_summary),
            len(macro_evidence_summary),
            len(provenance_context),
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

        # ── 事实核验与裁决自洽硬闸 ──────────────────
        truth_evaluator = EvidenceFactualTruthEvaluator()
        claims_verification = truth_evaluator.evaluate_claims(
            claims=claims,
            seven_reports=seven_reports,
            market_data_context=market_data_context,
            analysis_baseline_date=analysis_baseline_date,
        )

        manager_verdict = extract_and_validate_manager_verdict(
            raw_response=full_content,
            claims_verification=claims_verification,
        )

        if not manager_verdict["consistency_check_passed"]:
            failed_reasons = "; ".join(manager_verdict["failed_checks"])
            _logger.warning("[research_manager] consistency check failed: %s", failed_reasons)
            blocked_plan = f"研究总监裁决自洽硬闸未通过：{failed_reasons}。已阻断进入 Trader 执行阶段。"
            final_plan = blocked_plan
            final_decision = f"{full_content}\n\n[系统硬闸告警] 裁决自洽硬闸未通过：{failed_reasons}，已阻断后续交易。"
        else:
            final_plan = full_content
            final_decision = full_content

        # ── 推送辩论裁决（标记流式结束）──
        if tracker:
            tracker.emit_debate_message(
                debate="research", agent="Research Manager",
                round_num=-1, content=final_decision, is_verdict=True,
            )

        new_investment_debate_state = {
            "judge_decision": final_decision,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_speaker": investment_debate_state.get("current_speaker", ""),
            "current_response": final_decision,
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
            "manager_verdict": manager_verdict,
            "evidence_verification": claims_verification,
            "report_manifest": report_manifest,
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": final_plan,
            "manager_verdict": manager_verdict,
            "evidence_verification": claims_verification,
            "report_manifest": report_manifest,
        }

    return research_manager_node
