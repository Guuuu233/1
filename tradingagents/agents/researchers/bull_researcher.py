import logging
from typing import Any, Mapping
from tradingagents.dataflows.config import get_config
from tradingagents.prompts import get_prompt
from tradingagents.graph.intent_parser import build_horizon_context
from tradingagents.agents.utils.agent_states import current_tracker_var, is_v2_debate_enabled
from tradingagents.agents.utils.debate_utils import (
    DebateProtocolError,
    _string_list,
    build_debate_report_manifest,
    format_claim_subset_for_prompt,
    format_claims_for_prompt,
    render_debate_prompt,
    safe_int,
    update_debate_state_with_payload,
    validate_debate_response,
)
from tradingagents.agents.utils.prompt_injection import build_injection_slots, Placement, DEFAULT_PLACEMENT

_logger = logging.getLogger(__name__)


def _validate_bull_claim_references(
    state: Mapping[str, Any],
    payload: Mapping[str, Any],
    is_opening_stage: bool,
) -> tuple[bool, str, str]:
    """Validate claim reference boundaries for Bull Analyst (E-03b-1).

    Contract:
    1. v2 Opening: Bull first speaker has no opponent claims to reference.
       Must maintain double-blind: responded_claim_ids=[], new_claims[].target_claim_ids=[],
       challenges=[], resolved_claim_ids=[], unresolved_claim_ids=[], next_focus_claim_ids=[].
       No placeholder or invented opponent claim IDs.
    2. All stages: Any claim reference field (responded_claim_ids, target_claim_ids,
       challenges.target_claim_id, resolved_claim_ids, unresolved_claim_ids, next_focus_claim_ids)
       containing an ID not present in the current claim ledger must fail-closed.
    """
    known_claims = state.get("claims", []) or []
    known_claim_ids = {
        str(c.get("claim_id", "")).strip()
        for c in known_claims
        if isinstance(c, Mapping) and str(c.get("claim_id", "")).strip()
    }

    if is_opening_stage:
        raw_responded = _string_list(payload.get("responded_claim_ids"))
        if raw_responded:
            return (
                False,
                "invalid_protocol",
                f"Opening 阶段 (Bull) 必须为独立双盲立论，responded_claim_ids 必须为空数组 [] "
                f"(当前包含: {raw_responded})，严禁生成或提示占位的对手 claim ID",
            )

        for idx, nc in enumerate(payload.get("new_claims") or [], start=1):
            if isinstance(nc, Mapping):
                t_ids = _string_list(nc.get("target_claim_ids"))
                if t_ids:
                    return (
                        False,
                        "invalid_protocol",
                        f"Opening 阶段 (Bull) 第 {idx} 条 claim 的 target_claim_ids 必须为空数组 [] "
                        f"(当前包含: {t_ids})，严禁生成或提示占位的对手 claim ID",
                    )
                single_tid = nc.get("target_claim_id")
                if single_tid and str(single_tid).strip():
                    return (
                        False,
                        "invalid_protocol",
                        f"Opening 阶段 (Bull) 第 {idx} 条 claim 的 target_claim_id 必须为空 "
                        f"(当前包含: {single_tid})，严禁生成或提示占位的对手 claim ID",
                    )

        raw_challenges = payload.get("challenges") or []
        if raw_challenges:
            return (
                False,
                "invalid_protocol",
                f"Opening 阶段 (Bull) 为立论阶段，禁止提交 challenges 盘问载荷 "
                f"(当前包含 {len(raw_challenges)} 条 challenge)",
            )

        raw_resolved = _string_list(payload.get("resolved_claim_ids"))
        if raw_resolved:
            return (
                False,
                "invalid_protocol",
                f"Opening 阶段 (Bull) resolved_claim_ids 必须为空数组 [] (当前包含: {raw_resolved})",
            )

        raw_unresolved = _string_list(payload.get("unresolved_claim_ids"))
        if raw_unresolved:
            return (
                False,
                "invalid_protocol",
                f"Opening 阶段 (Bull) unresolved_claim_ids 必须为空数组 [] (当前包含: {raw_unresolved})",
            )

        raw_next_focus = _string_list(payload.get("next_focus_claim_ids"))
        if raw_next_focus:
            return (
                False,
                "invalid_protocol",
                f"Opening 阶段 (Bull) next_focus_claim_ids 必须为空数组 [] (当前包含: {raw_next_focus})",
            )

    # All stages: check for unknown claim IDs against the current claim ledger
    referenced_ids: list[str] = []
    referenced_ids.extend(_string_list(payload.get("responded_claim_ids")))
    for nc in payload.get("new_claims") or []:
        if isinstance(nc, Mapping):
            referenced_ids.extend(_string_list(nc.get("target_claim_ids")))
            single_tid = nc.get("target_claim_id")
            if single_tid and str(single_tid).strip():
                referenced_ids.append(str(single_tid).strip())
    for ch in payload.get("challenges") or []:
        if isinstance(ch, Mapping):
            t_id = str(ch.get("target_claim_id") or "").strip()
            if t_id:
                referenced_ids.append(t_id)
    referenced_ids.extend(_string_list(payload.get("resolved_claim_ids")))
    referenced_ids.extend(_string_list(payload.get("unresolved_claim_ids")))
    referenced_ids.extend(_string_list(payload.get("next_focus_claim_ids")))

    unknown_ids = sorted({cid for cid in referenced_ids if cid not in known_claim_ids})
    if unknown_ids:
        return (
            False,
            "invalid_protocol",
            f"Bull Analyst 命题引用包含不存在于当前 claim 账本的未知 claim ID: {unknown_ids} "
            f"(当前账本现有 claim: {sorted(known_claim_ids)})，违反 fail-closed 契约",
        )

    return True, "valid", ""


def create_bull_researcher(llm, memory, custom_prompt: str = "", placement: Placement = DEFAULT_PLACEMENT):
    async def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        current_count = safe_int(investment_debate_state.get("count", 0), 0)
        message_index = current_count + 1
        v2_enabled = is_v2_debate_enabled(state)
        current_stage = str(investment_debate_state.get("protocol_stage") or "opening").strip().lower()
        is_opening_stage = v2_enabled and current_stage == "opening" and message_index <= 2
        is_challenge_stage = v2_enabled and (
            current_stage == "challenge" or (not is_opening_stage and message_index in (3, 4))
        )

        macro_report = state.get("macro_report", "")
        market_research_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        smart_money_report = state.get("smart_money_report", "")
        volume_price_report = state.get("volume_price_report", "")

        report_manifest = build_debate_report_manifest(state)
        _logger.info("[bull_researcher] report input manifest: %s", report_manifest)

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

        if is_opening_stage:
            history = ""
            current_response = ""
            claims = []
            focus_claim_ids = []
            unresolved_claim_ids = []
            round_summary = ""
            round_goal = "建立多头核心立论并覆盖3个不同战场"
            past_memories = memory.get_memories(curr_situation, n_matches=0)
            past_memory_str = ""
        else:
            history = investment_debate_state.get("history", "")
            current_response = investment_debate_state.get("current_response", "")
            claims = investment_debate_state.get("claims", [])
            focus_claim_ids = investment_debate_state.get("focus_claim_ids", [])
            unresolved_claim_ids = investment_debate_state.get("unresolved_claim_ids", [])
            round_summary = investment_debate_state.get("round_summary", "")
            round_goal = investment_debate_state.get("round_goal", "")
            past_memories = memory.get_memories(curr_situation, n_matches=2)
            past_memory_str = ""
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"

        injection_slots = build_injection_slots(custom_prompt, placement, role_key="bull_researcher")
        cfg = get_config()
        raw_prompt_template = get_prompt("bull_prompt", config=cfg)
        prompt_language = cfg.get("prompt_language", "zh") if cfg else "zh"
        rendered_template = render_debate_prompt(
            raw_prompt_template,
            is_opening_stage=is_opening_stage,
            is_challenge_stage=is_challenge_stage,
            language=prompt_language,
        )
        prompt = horizon_ctx + rendered_template.format(
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

        # ── 实现 Token 级流式输出与单轮重试机制 ──────────────────
        tracker = current_tracker_var.get()
        debate_round = 1 if is_opening_stage else ((message_index - 1) // 2 + 1)
        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)

        attempts_trace = []
        max_attempts = 2
        last_error_detail = ""

        for attempt_num in range(1, max_attempts + 1):
            if attempt_num == 1:
                attempt_prompt = prompt
            else:
                if is_opening_stage:
                    retry_instruction = (
                        f"\n\n【协议重试警告 (Attempt {attempt_num})】：\n"
                        f"你上一次输出的 DEBATE_STATE 机器块未通过 Opening 阶段协议校验，错误原因：{last_error_detail}。\n"
                        f"请在保持专业立论正文的同时，重新严格按 Opening 契约格式在输出末尾输出 <!-- DEBATE_STATE: ... --> 机器块。\n"
                        f"- 当前 Opening 立论发言要求：\n"
                        f"  1. responded_claim_ids 必须为空数组 []。\n"
                        f"  2. new_claims 必须恰好包含 3 条 Claim，且必须覆盖 3 个不同合法战场 (capital_flow / sentiment_theme / price_volume / macro_policy / fundamentals)。\n"
                        f"  3. 每一项 new_claims 必须包含 battlefield 字段（属于五大战场之一），且 target_claim_ids 必须为空数组 []。\n"
                        f"  4. confidence 必须是 0.00-1.00 之间的有限数值，严禁百分比。\n"
                        f"  5. resolved_claim_ids 必须为空数组 []。\n"
                        f"  6. 独立双盲立论严禁引用或占位任何对手 claim ID，challenges、resolved_claim_ids、unresolved_claim_ids、next_focus_claim_ids 均必须为空数组 []。\n"
                        f"请立即修正并重新输出完整发言及合规机器块！"
                    )
                elif is_challenge_stage:
                    opponent_open_claims = [
                        c["claim_id"] for c in claims
                        if (c.get("speaker_key") == "Bear" or (c.get("stance") and c.get("stance") != "bullish"))
                        and c.get("status") != "resolved"
                    ]
                    opponent_all_claims = [
                        c["claim_id"] for c in claims
                        if (c.get("speaker_key") == "Bear" or (c.get("stance") and c.get("stance") != "bullish"))
                    ]
                    legal_targets = opponent_open_claims or opponent_all_claims
                    retry_instruction = (
                        f"\n\n【协议重试警告 (Attempt {attempt_num})】：\n"
                        f"你上一次输出的 DEBATE_STATE 机器块未通过 Challenge 阶段协议校验，错误原因：{last_error_detail}。\n"
                        f"请在保持专业盘问正文的同时，重新严格按 Challenge 契约格式在输出末尾输出 <!-- DEBATE_STATE: ... --> 机器块。\n"
                        f"- 当前 Challenge 发言要求：\n"
                        f"  1. new_claims 必须严格为空数组 []。禁止提出新 Claim。\n"
                        f"  2. challenges 至少包含 1 条；每条必须包含 target_claim_id、weakest_point、至少 1 条非空 evidence、severity（fatal/major/minor）。\n"
                        f"     当前可选合法对手 Claim: {legal_targets}。\n"
                        f"  3. self_win_prob 必须是 0.0-1.0 之间的有限数值。\n"
                        f"  4. responded_claim_ids 必须包含所有被 challenge 的 target_claim_id。\n"
                        f"  5. 严禁擅自 resolve 对手的 Claim。\n"
                        f"请立即修正并重新输出完整发言及合规机器块！"
                    )
                else:
                    opponent_open_claims = [
                        c["claim_id"] for c in claims
                        if (c.get("speaker_key") == "Bear" or (c.get("stance") and c.get("stance") != "bullish"))
                        and c.get("status") != "resolved"
                    ]
                    opponent_all_claims = [
                        c["claim_id"] for c in claims
                        if (c.get("speaker_key") == "Bear" or (c.get("stance") and c.get("stance") != "bullish"))
                    ]
                    same_side_prev_claims = [
                        c["claim"] for c in claims
                        if (c.get("speaker_key") == "Bull" or c.get("stance") == "bullish")
                    ]
                    prev_claims_hint = (
                        f"已提出的历史多头观点（严禁重复或同义改写，相似度须 < 0.82）: {same_side_prev_claims}。"
                        if same_side_prev_claims
                        else ""
                    )
                    target_hint = (
                        f" (如 target_claim_ids: {opponent_all_claims[:1]})"
                        if opponent_all_claims
                        else ""
                    )
                    retry_instruction = (
                        f"\n\n【协议重试警告 (Attempt {attempt_num})】：\n"
                        f"你上一次输出的 DEBATE_STATE 机器块未通过协议校验，错误原因：{last_error_detail}。\n"
                        f"请在保持专业论证正文的同时，重新严格按契约格式在输出末尾输出 <!-- DEBATE_STATE: ... --> 机器块。\n"
                        f"- 当前第 {message_index} 次发言要求：\n"
                        f"  1. 信息增量硬闸：本轮必须提出至少一条具有实质信息增量的新 Claim，必须包含历史未出现过的具体数值/新证据实体/新因果链，严禁复读或轻微改写前几轮观点。\n"
                        f"     {prev_claims_hint}\n"
                        f"  2. responded_claim_ids 必须包含至少一条对手未解决 Claim ID。当前可选合法未解决对手 Claim: {opponent_open_claims or opponent_all_claims}。\n"
                        f"  3. new_claims 中的每一项必须包含 target_claim_ids 字段，且 target_claim_ids 必须指定至少一条对手 Claim ID{target_hint}。\n"
                        f"  4. confidence 必须是 0.00-1.00 之间的有限数值，严禁百分比。\n"
                        f"  5. 严禁擅自 resolve 对手的 Claim。\n"
                        f"请立即修正并重新输出完整发言及合规机器块！"
                    )
                attempt_prompt = prompt + retry_instruction

            full_content = ""
            async for chunk in llm.astream(attempt_prompt):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_content += content
                if tracker:
                    tracker._emit_token("Bull Researcher", "investment_debate_state", content)
                    tracker.emit_debate_token(
                        debate="research", agent="Bull Researcher",
                        round_num=debate_round, token=content, model_name=model_name,
                    )

            is_valid, parse_status, error_detail, parsed_payload = validate_debate_response(
                state=investment_debate_state,
                raw_response=full_content,
                speaker_key="Bull",
                stance="bullish",
                marker="DEBATE_STATE",
                domain="investment",
            )

            if is_valid and parsed_payload is not None:
                is_valid, parse_status, error_detail = _validate_bull_claim_references(
                    state=investment_debate_state,
                    payload=parsed_payload,
                    is_opening_stage=is_opening_stage,
                )

            attempt_record = {
                "attempt_index": attempt_num,
                "message_index": message_index,
                "debate_round": debate_round,
                "speaker": "Bull Analyst",
                "speaker_key": "Bull",
                "parse_status": parse_status,
                "error_detail": error_detail,
                "raw_response": full_content,
                "accepted": is_valid,
            }
            attempts_trace.append(attempt_record)

            if is_valid:
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
                    attempts=attempts_trace,
                )

                return {"investment_debate_state": new_investment_debate_state}
            else:
                last_error_detail = error_detail
                _logger.warning(
                    "[bull_researcher] Attempt %d at message_index=%d failed protocol validation: %s (parse_status=%s)",
                    attempt_num, message_index, error_detail, parse_status
                )

        _logger.error(
            "[bull_researcher] All %d attempts failed debate protocol at message_index=%d. Raising DebateProtocolError.",
            max_attempts, message_index
        )
        raise DebateProtocolError(
            f"Debate protocol validation failed for Bull Analyst at message_index={message_index} after {max_attempts} attempts: {last_error_detail}",
            message_index=message_index,
            speaker="Bull Analyst",
            details=last_error_detail,
            attempts=attempts_trace,
        )

    return bull_node
