import logging
import re
import time
from typing import Any, Mapping, Sequence

from tradingagents.dataflows.config import get_config
from tradingagents.prompts import get_prompt
from tradingagents.prompts.catalog import _resolve_language
from tradingagents.agents.utils.agent_states import (
    current_tracker_var,
    is_credit_weighting_enabled,
)
from tradingagents.agents.utils.debate_utils import (
    build_debate_report_manifest,
    format_claim_subset_for_prompt,
    format_claims_for_prompt,
    safe_int,
    validate_debate_preconditions,
)
from tradingagents.agents.utils.evidence_summary import (
    build_dense_report_input,
    build_evidence_summary,
)
from tradingagents.agents.utils.evidence_verifier import (
    EvidenceFactualTruthEvaluator,
    extract_and_validate_manager_verdict,
    format_battlefield_coverage,
    format_challenge_verification_summary,
    format_challenges_for_prompt,
    format_claims_with_verification_for_prompt,
)
from tradingagents.agents.utils.claim_cluster import (
    RELATION_GRAPH_STATUS_AVAILABLE,
    RELATION_GRAPH_STATUS_INVALID,
    RELATION_GRAPH_STATUS_PENDING,
    cluster_claims,
    format_claim_cluster_summary_for_prompt,
    tally_cluster_votes,
)
from tradingagents.agents.utils.prompt_injection import build_injection_slots, Placement, DEFAULT_PLACEMENT
from tradingagents.agents.utils.decision_status import (
    ACTION_NO_TRADE,
    DIRECTION_NA,
    status_from_manager_verdict,
)
from tradingagents.agents.utils.run_integrity import (
    evaluate_state_integrity,
    fund_flow_guard_abstain_status,
)
from tradingagents.graph.intent_parser import (
    build_horizon_context,
    get_bound_research_horizon,
)

_logger = logging.getLogger(__name__)

_RELATION_GRAPH_KEYS = ("evidence_relation_graph", "evidence_relations")


class RelationPromptGuardError(ValueError):
    """The research manager prompt template no longer matches the E-02 prompt guard."""


# E-02: the manager always passes an explicit relation status, so the legacy
# keyword-cluster tally and analyst weighting instructions are rewritten into
# relation-only contribution rules.  Every legacy segment must match exactly the
# expected number of times and no weighting instruction may survive, so template
# drift fails closed instead of silently restoring keyword voting.
_RELATION_PROMPT_REWRITES: dict[str, tuple[tuple[str, str, int], ...]] = {
    "zh": (
        (
            "1. **各分析师 Verdict 全景概览与动态加权**：",
            "1. **各分析师 Verdict 全景概览与证据贡献状态**：",
            1,
        ),
        (
            "与动态加权权重：",
            "与证据贡献状态（引用上文 E-02 关系折叠结果；无显式 E-01 关系时写 UNKNOWN，不得给出百分比或分值）：",
            1,
        ),
        ("：verdict 与权重", "：verdict 与证据贡献状态", 7),
        (
            "   - 计票按 cluster_id 去重计票：analyst_count 发言人数仅作解释，不得当作独立权重；"
            "同属价格冲击（收盘价/成交量/当日涨跌幅）的技术面、量价与主力资金只计一票 independent_cluster_count。",
            "   - E-02 仅使用上文显式 E-01 关系图折叠结果约束贡献；禁止根据关键词簇、分析师人数、"
            "供应商/来源差异或核验状态推断独立支持。PENDING/UNKNOWN claim 不得增加方向性支持。",
            1,
        ),
        (
            "需根据分析视角与本次研究档（horizon）与市场环境动态赋予权重，并按研究档明确区分核心裁决依据：",
            "不得为分析师或 claim 设定百分比、分值或加总比例，只按本次研究档（horizon）明确区分核心裁决依据"
            "（下列“侧重”仅表示裁决关注点，不产生方向性支持）：",
            1,
        ),
        ("技术面、资金面、情绪面高权重，", "裁决侧重技术面、资金面、情绪面，", 1),
        ("基本面、宏观与产业链权重高，", "裁决侧重基本面、宏观与产业链，", 1),
        (
            "趋势行情加权技术与资金，震荡市加权基本面与情绪，宏观大转折期加权宏观与产业链。",
            "趋势行情侧重技术与资金，震荡市侧重基本面与情绪，宏观大转折期侧重宏观与产业链"
            "（仅调整关注重点，不产生方向性支持）。",
            1,
        ),
    ),
    "en": (
        (
            "1) Tally independent evidence clusters (deduplicating claims by cluster_id) and compute "
            "cluster-based directional weight; analyst list serves as explanatory context only "
            "(analyst_count must not be used directly as independent voting weight).",
            "1) Use only the explicit E-01 relation reduction shown above for contribution grouping; "
            "do not infer independent support from keyword clusters, analyst count, provider/source "
            "differences, or verifier status. Pending/unknown claims must not add directional support.",
            1,
        ),
        (
            "2) Dynamically weight perspectives and distinguish core adjudication criteria by research horizon:",
            "2) Do not assign percentages or scores to analysts or claims; distinguish core adjudication "
            "criteria by research horizon (the focus below marks what is decisive and adds no directional support):",
            1,
        ),
        ("Primary weight on ", "Primary adjudication focus on ", 2),
    ),
}
_RELATION_PROMPT_FORBIDDEN = {
    "zh": re.compile(r"权重|加权|计票|cluster_id"),
    "en": re.compile(r"\bweight|\btally\b|\bvot(?:e|ing)|cluster_id", re.IGNORECASE),
}


def _resolve_relation_graph_context(
    state: Mapping[str, Any],
    investment_debate_state: Mapping[str, Any],
) -> tuple[Any, str, str]:
    """Locate an explicitly supplied E-01 graph, or mark the relation contribution as pending.

    The current graph pipeline has no relation producer.  This function therefore
    never derives edges from claim text, reports, event coverage, or verifier
    status; it only forwards a named E-01 graph payload.  Deserialization, E-01
    validation and the rejection audit (raw payload + structured error) happen in
    ``tally_cluster_votes`` so every manager path records the same evidence.

    Exactly one location may carry the graph.  More than one supplied payload
    (an explicit null included) is ambiguous: it is marked invalid with every
    payload kept for audit, instead of silently using whichever location is
    scanned first.
    """
    containers = (
        ("state", state),
        ("investment_debate_state", investment_debate_state),
        ("market_data_context", state.get("market_data_context")),
        ("event_coverage", state.get("event_coverage")),
    )
    supplied: list[tuple[str, Any]] = []
    for container_name, container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in _RELATION_GRAPH_KEYS:
            if key in container:
                supplied.append((f"{container_name}.{key}", container.get(key)))

    if len(supplied) > 1:
        sources = [source for source, _ in supplied]
        return (
            dict(supplied),
            RELATION_GRAPH_STATUS_INVALID,
            f"multiple E-01 relation graph payloads supplied at {sources}; ambiguous relation input "
            "is not folded and claim contribution remains pending/unknown",
        )
    if supplied:
        source, raw_graph = supplied[0]
        if raw_graph is None:
            return (
                [],
                RELATION_GRAPH_STATUS_PENDING,
                f"E-01 relation graph at {source} is null; claim contribution remains pending/unknown",
            )
        return (
            raw_graph,
            RELATION_GRAPH_STATUS_AVAILABLE,
            f"explicit E-01 relation graph supplied at {source}",
        )

    return (
        [],
        RELATION_GRAPH_STATUS_PENDING,
        "E-01 relation graph is not produced in the current pipeline; claim contribution remains pending/unknown",
    )


def _apply_relation_prompt_guard(prompt_template: str, language: str) -> str:
    """Rewrite legacy keyword-cluster weighting instructions into E-02 contribution rules.

    Raises ``RelationPromptGuardError`` when the template drifts: each legacy segment
    must match the expected number of times and no weighting/tally instruction may
    survive the rewrite.
    """
    lang = "en" if language == "en" else "zh"
    guarded = prompt_template
    for legacy, replacement, expected in _RELATION_PROMPT_REWRITES[lang]:
        found = guarded.count(legacy)
        if found != expected:
            raise RelationPromptGuardError(
                f"{lang} legacy segment matched {found} time(s), expected {expected}: {legacy[:48]!r}"
            )
        guarded = guarded.replace(legacy, replacement)
    survivors = [
        line.strip()
        for line in guarded.splitlines()
        if _RELATION_PROMPT_FORBIDDEN[lang].search(line)
    ]
    if survivors:
        raise RelationPromptGuardError(
            f"{lang} weighting/tally instruction survived E-02 rewrite: {survivors[:3]}"
        )
    return guarded


def _format_relation_prompt_guard(relation_graph_status: str, language: str) -> str:
    """Final E-02 status line; the status comes from the reduction seam, not graph discovery."""
    status = str(relation_graph_status or RELATION_GRAPH_STATUS_PENDING).upper()
    if language == "en":
        return (
            "E-02 relation guard: status=" + status + ". Treat independence as UNKNOWN; "
            "the global contribution cap is conservative and no unconnected claim is an independent vote."
        )
    return (
        "E-02 关系贡献硬闸：状态=" + status + "。独立性恒为 UNKNOWN；遵守保守全局贡献上限，"
        "未被显式关系连接的 claim 不得作为独立票。"
    )


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
        val = state.get("horizon")
        if val and isinstance(val, str) and val.strip():
            return val.strip()
        metadata = state.get("horizon_run_metadata")
        if isinstance(metadata, dict):
            resolved = metadata.get("resolved")
            if resolved and isinstance(resolved, list) and len(resolved) > 0 and resolved[0]:
                return str(resolved[0]).strip()
            requested = metadata.get("requested")
            if requested and isinstance(requested, list) and len(requested) > 0 and requested[0]:
                return str(requested[0]).strip()
    bound = get_bound_research_horizon()
    if bound and isinstance(bound, str) and bound.strip():
        return bound.strip()
    return "short"


def _blocked_manager_payload(
    *,
    investment_debate_state: dict,
    report_manifest: dict,
    fund_flow_guard: dict,
    decision_status: dict,
    blocked_plan: str,
    manager_reason: str,
    run_integrity: dict | None = None,
    claim_evidence_summary: dict | None = None,
    consistency_check_passed: bool = True,
    failed_checks: list | None = None,
    evidence_verification: list | None = None,
    claim_cluster_metrics: dict | None = None,
    research_horizon: str = "short",
    reports: Mapping[str, str] | None = None,
    claims_verification: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    relation_graph: Any = None,
    relation_graph_status: str | None = None,
    relation_graph_reason: str | None = None,
) -> dict:
    """Shared early-return shape for INVALID/ABSTAIN manager short-circuits."""
    if claim_cluster_metrics is None:
        claim_cluster_metrics = tally_cluster_votes(
            claims=investment_debate_state.get("claims", []),
            reports=reports,
            claims_verification=claims_verification,
            relation_graph=relation_graph,
            relation_graph_status=relation_graph_status,
            relation_graph_reason=relation_graph_reason,
        )
    summary_dict = dict(claim_evidence_summary) if isinstance(claim_evidence_summary, dict) else {}
    if not summary_dict and isinstance(investment_debate_state.get("claim_evidence_summary"), dict):
        summary_dict = dict(investment_debate_state["claim_evidence_summary"])
    ev_list = list(evidence_verification if evidence_verification is not None else (claims_verification or []))
    manager_verdict = {
        "direction": DIRECTION_NA,
        "winner": "tie",
        "reason": manager_reason,
        "position_pct": 0,
        "entry": None,
        "target": None,
        "stop_loss": None,
        "upside": None,
        "downside": None,
        "odds": None,
        "adopted_claim_ids": [],
        "partially_adopted_claims": [],
        "rejected_claim_ids": [],
        "excluded_evidence": [],
        "claim_evidence_summary": summary_dict,
        "consistency_check_passed": consistency_check_passed,
        "failed_checks": list(failed_checks or []),
        "decision_status": decision_status,
        "analysis_status": decision_status.get("analysis_status"),
        "trade_action": decision_status.get("trade_action", ACTION_NO_TRADE),
        "risk_status": decision_status.get("risk_status"),
        "confirmation_state": decision_status.get("confirmation_state", "UNRESOLVED"),
        "horizon": research_horizon,
        "research_horizon": research_horizon,
        "evidence_relation_status": claim_cluster_metrics.get("relation_graph_status", "pending"),
        "evidence_relation_reason": claim_cluster_metrics.get("relation_graph_reason", ""),
    }
    payload = {
        "fund_flow_consensus_guard": fund_flow_guard,
        "investment_plan": blocked_plan,
        "manager_verdict": manager_verdict,
        "evidence_verification": ev_list,
        "claim_evidence_summary": summary_dict,
        "report_manifest": report_manifest,
        "decision_status": decision_status,
        "analysis_status": decision_status.get("analysis_status"),
        "trade_action": decision_status.get("trade_action", ACTION_NO_TRADE),
        "risk_status": decision_status.get("risk_status"),
        "confirmation_state": decision_status.get("confirmation_state", "UNRESOLVED"),
        "final_trade_decision": blocked_plan,
        "investment_debate_state": {
            **investment_debate_state,
            "judge_decision": blocked_plan,
            "current_response": blocked_plan,
            "manager_verdict": manager_verdict,
            "evidence_verification": ev_list,
            "claim_evidence_summary": summary_dict,
            "report_manifest": report_manifest,
            "claim_cluster_metrics": claim_cluster_metrics,
            "evidence_relation_status": claim_cluster_metrics.get("relation_graph_status", "pending"),
            "evidence_relation_reduction": claim_cluster_metrics.get("relation_audit", {}),
            "independent_cluster_count": claim_cluster_metrics.get("independent_cluster_count", 0),
            "analyst_count": claim_cluster_metrics.get("analyst_count", 0),
            "verified_evidence_count": claim_cluster_metrics.get("verified_evidence_count", 0),
        },
    }
    if run_integrity is not None:
        payload["run_integrity"] = run_integrity
    return payload


def create_research_manager(llm, memory, custom_prompt: str = "", placement: Placement = DEFAULT_PLACEMENT):
    async def research_manager_node(state) -> dict:
        research_horizon = _resolve_research_horizon(state)
        user_intent = state.get("user_intent") or {}
        focus_areas = user_intent.get("focus_areas", [])
        specific_questions = user_intent.get("specific_questions", [])

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
        expected_symbol = (
            (market_data_context.get("symbol") if isinstance(market_data_context, dict) else None)
            or state.get("symbol")
            or state.get("ticker")
        )
        symbol_val = expected_symbol
        effective_market_data_context = dict(market_data_context) if isinstance(market_data_context, dict) else {}
        if analysis_baseline_date and not effective_market_data_context.get("analysis_baseline_date"):
            effective_market_data_context["analysis_baseline_date"] = analysis_baseline_date
        if expected_symbol and not effective_market_data_context.get("symbol"):
            effective_market_data_context["symbol"] = expected_symbol
        data_gaps = state.get("data_gaps") or (market_data_context.get("data_gaps") if isinstance(market_data_context, dict) else []) or []

        investment_debate_state = state["investment_debate_state"]
        claims = investment_debate_state.get("claims", [])
        unresolved_claim_ids = investment_debate_state.get("unresolved_claim_ids", [])
        round_summary = investment_debate_state.get("round_summary", "")
        relation_graph, relation_graph_status, relation_graph_reason = _resolve_relation_graph_context(
            state,
            investment_debate_state,
        )

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

        # ── P0-1: Run integrity before any Neutral/HOLD collapse ────────────
        run_integrity = evaluate_state_integrity(state)
        if run_integrity.all_required_failed and run_integrity.decision_status:
            blocked_plan = (
                "运行完整性判定：必需分析师报告全部失败（INVALID_RUN/DATA_ERROR）。"
                "不得输出方向、概率、百分比区间或 Neutral/HOLD 观点；执行动作 NO_TRADE。"
                f" 失败角色={','.join(run_integrity.failed_required)}"
            )
            _logger.warning(
                "[research_manager] run integrity INVALID: %s",
                run_integrity.reason_codes,
            )
            return _blocked_manager_payload(
                investment_debate_state=investment_debate_state,
                report_manifest=report_manifest,
                fund_flow_guard=fund_flow_guard,
                decision_status=run_integrity.decision_status,
                blocked_plan=blocked_plan,
                manager_reason="必需分析师上游全部失败",
                run_integrity=run_integrity.to_dict(),
                consistency_check_passed=False,
                failed_checks=list(run_integrity.reason_codes),
                research_horizon=research_horizon,
                reports=seven_reports,
                relation_graph=relation_graph,
                relation_graph_status=relation_graph_status,
                relation_graph_reason=relation_graph_reason,
            )

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

        social_data_context = state.get("social_data_context") or {}
        if isinstance(social_data_context, dict) and social_data_context:
            social_mode = social_data_context.get("mode", "disabled")
            social_status = social_data_context.get("status", "unknown")
            social_dir_allowed = bool(social_data_context.get("direction_allowed", False))
            social_reasons = social_data_context.get("reason_codes", [])
            bundle = social_data_context.get("bundle") if isinstance(social_data_context.get("bundle"), dict) else {}
            bundle_id = (bundle.get("bundle_id") if isinstance(bundle, dict) else None) or social_data_context.get("bundle_id") or "none"

            reasons_str = f" ({', '.join(str(r) for r in social_reasons)})" if social_reasons else ""
            prov_lines.append(
                f"- 社交数据状态 (social_data_context): mode={social_mode}, status={social_status}, "
                f"direction_allowed={social_dir_allowed}, bundle_id={bundle_id}{reasons_str}"
            )
            if not social_dir_allowed:
                prov_lines.append(
                    f"  * [DIRECTION_GUARD] 明确禁止：当前社交数据 direction_allowed=false（状态: {social_status}{reasons_str}），"
                    f"严禁将社交分数、散户情绪得分或讨论热度作为多空方向证据或交易依据！"
                )

            social_ledger = social_data_context.get("data_failure_ledger")
            if isinstance(social_ledger, list) and social_ledger:
                for entry in social_ledger:
                    if isinstance(entry, dict):
                        prov_lines.append(f"  * [UNAVAILABLE] {entry.get('source', 'social_archive')}: {entry.get('reason', entry.get('status', 'failed'))}")

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
            decision_status = fund_flow_guard_abstain_status(fund_flow_guard).to_dict()
            blocked_plan = (
                "资金流来源选择 guard 已阻断：不得输出增持、减持、吸筹或其他方向性投资计划。"
                "状态=ABSTAIN，动作=NO_TRADE（不是 Neutral/HOLD 观点）。"
            )
            return _blocked_manager_payload(
                investment_debate_state=investment_debate_state,
                report_manifest=report_manifest,
                fund_flow_guard=fund_flow_guard,
                decision_status=decision_status,
                blocked_plan=blocked_plan,
                manager_reason="资金流来源选择 guard 已阻断",
                run_integrity=run_integrity.to_dict(),
                consistency_check_passed=True,
                research_horizon=research_horizon,
                reports=seven_reports,
                relation_graph=relation_graph,
                relation_graph_status=relation_graph_status,
                relation_graph_reason=relation_graph_reason,
            )

        # ── 辩论前置硬闸检查 (Debate Pre-Gate Hard Gate - fail-closed before LLM) ──
        gate_errors = validate_debate_preconditions(investment_debate_state, claims=claims)
        if gate_errors:
            failed_reasons = "; ".join(f"辩论前置硬闸未通过: {err}" for err in gate_errors)
            _logger.warning("[research_manager] debate pre-gate check failed: %s", failed_reasons)
            blocked_plan = (
                f"研究总监裁决自洽硬闸未通过：{failed_reasons}。"
                "状态=ABSTAIN，动作=NO_TRADE；已阻断进入 Trader 执行阶段。"
            )
            truth_evaluator = EvidenceFactualTruthEvaluator()
            claims_verification = truth_evaluator.evaluate_claims(
                claims=claims,
                seven_reports=seven_reports,
                market_data_context=effective_market_data_context,
                analysis_baseline_date=analysis_baseline_date,
                social_data_context=social_data_context,
            )
            claim_evidence_summary = truth_evaluator.aggregate_claim_evidence(
                claims=claims,
                claims_verification=claims_verification,
                analysis_baseline_date=analysis_baseline_date or None,
                expected_symbol=expected_symbol or None,
                market_data_context=effective_market_data_context,
            )
            from tradingagents.agents.utils.decision_status import abstain_status

            decision_status = abstain_status(
                reason_codes=[f"debate_pre_gate:{err}" for err in gate_errors],
                trade_action="NO_TRADE",
                risk_status="BLOCKED",
            ).to_dict()
            tracker = current_tracker_var.get()
            if tracker:
                tracker.emit_debate_message(
                    debate="research", agent="Research Manager",
                    round_num=-1, content=blocked_plan, is_verdict=True,
                )
            payload = _blocked_manager_payload(
                investment_debate_state=investment_debate_state,
                report_manifest=report_manifest,
                fund_flow_guard=fund_flow_guard,
                decision_status=decision_status,
                blocked_plan=blocked_plan,
                manager_reason=f"辩论前置硬闸未通过: {failed_reasons}",
                run_integrity=run_integrity.to_dict(),
                claim_evidence_summary=claim_evidence_summary,
                consistency_check_passed=False,
                failed_checks=[f"辩论前置硬闸未通过: {err}" for err in gate_errors],
                evidence_verification=claims_verification,
                research_horizon=research_horizon,
                reports=seven_reports,
                relation_graph=relation_graph,
                relation_graph_status=relation_graph_status,
                relation_graph_reason=relation_graph_reason,
            )
            # Preserve pre-gate debate bookkeeping fields
            debate_state = payload["investment_debate_state"]
            debate_state.update(
                {
                    "history": investment_debate_state.get("history", ""),
                    "bear_history": investment_debate_state.get("bear_history", ""),
                    "bull_history": investment_debate_state.get("bull_history", ""),
                    "current_speaker": investment_debate_state.get("current_speaker", ""),
                    "count": investment_debate_state.get("count", 0),
                    "claims": claims,
                    "round_messages": investment_debate_state.get("round_messages", []),
                    "focus_claim_ids": investment_debate_state.get("focus_claim_ids", []),
                    "open_claim_ids": investment_debate_state.get("open_claim_ids", []),
                    "resolved_claim_ids": investment_debate_state.get("resolved_claim_ids", []),
                    "unresolved_claim_ids": unresolved_claim_ids,
                    "round_summary": round_summary,
                    "round_goal": investment_debate_state.get("round_goal", ""),
                    "claim_counter": investment_debate_state.get("claim_counter", 0),
                    "claim_evidence_summary": claim_evidence_summary,
                    "evidence_verification": claims_verification,
                }
            )
            return payload

        # ── 事实核验与 Claim 证据链聚合 (Fact Checking & Claim Evidence Aggregation) ──
        truth_evaluator = EvidenceFactualTruthEvaluator()
        claims_verification = truth_evaluator.evaluate_claims(
            claims=claims,
            seven_reports=seven_reports,
            market_data_context=effective_market_data_context,
            analysis_baseline_date=analysis_baseline_date,
            social_data_context=social_data_context,
        )
        claim_evidence_summary = truth_evaluator.aggregate_claim_evidence(
            claims=claims,
            claims_verification=claims_verification,
            analysis_baseline_date=analysis_baseline_date or None,
            expected_symbol=expected_symbol or None,
            market_data_context=effective_market_data_context,
        )

        claims = cluster_claims(
            claims,
            symbol=symbol_val,
            date=analysis_baseline_date,
            claims_verification=claims_verification,
        )
        claim_cluster_metrics = tally_cluster_votes(
            claims=claims,
            reports=seven_reports,
            claims_verification=claims_verification,
            symbol=symbol_val,
            trade_date=analysis_baseline_date,
            relation_graph=relation_graph,
            relation_graph_status=relation_graph_status,
            relation_graph_reason=relation_graph_reason,
        )

        claims_text = format_claims_with_verification_for_prompt(
            claims=claims,
            claims_verification=claims_verification,
            claim_evidence_summary=claim_evidence_summary,
        )
        cluster_summary = format_claim_cluster_summary_for_prompt(
            claim_cluster_metrics,
            language=_resolve_language(get_config()),
        )
        if cluster_summary:
            claims_text = f"{cluster_summary}\n\n{claims_text}"
        unresolved_claims_subset = [c for c in claims if str(c.get("claim_id", "")).strip() in set(unresolved_claim_ids)]
        unresolved_claims_text = format_claims_with_verification_for_prompt(
            claims=unresolved_claims_subset,
            claims_verification=claims_verification,
            claim_evidence_summary=claim_evidence_summary,
            empty_message="当前没有未决 claim。",
        )

        # ── Parameterize actual messages, stages, challenges, and battlefield coverage ──
        round_messages = investment_debate_state.get("round_messages", [])
        actual_message_count = len(round_messages) if round_messages else safe_int(investment_debate_state.get("count", 0), 0)

        stages_list = [
            str(m.get("stage") or m.get("protocol_stage") or "").strip()
            for m in round_messages
            if m.get("stage") or m.get("protocol_stage")
        ]
        unique_stages = list(dict.fromkeys([s for s in stages_list if s]))
        if not unique_stages:
            unique_stages = ["opening", "challenge"]
        actual_stages_desc = f"覆盖阶段: {', '.join(unique_stages)}"

        is_tb_skipped = bool(investment_debate_state.get("tiebreak_skipped", False))
        if is_tb_skipped:
            tiebreak_status_desc = "已跳过加赛(证据足以裁决)"
        elif "tiebreak" in unique_stages:
            tiebreak_status_desc = "已执行加赛"
        else:
            tiebreak_status_desc = "标准流程"

        challenges = investment_debate_state.get("challenges", [])
        challenges_verification = truth_evaluator.evaluate_challenges(
            challenges=challenges,
            seven_reports=seven_reports,
            market_data_context=effective_market_data_context,
            analysis_baseline_date=analysis_baseline_date,
            social_data_context=social_data_context,
        )
        challenges_text = format_challenges_for_prompt(
            challenges=challenges,
            challenge_verification=challenges_verification,
        )
        challenge_verification_text = format_challenge_verification_summary(
            challenges=challenges,
            challenge_verification=challenges_verification,
        )
        battlefield_coverage_text = format_battlefield_coverage(claims)

        config = get_config()
        horizon_ctx = build_horizon_context(
            research_horizon,
            focus_areas,
            specific_questions,
            agent_type="research_manager",
            research_horizon=research_horizon,
        )

        injection_slots = build_injection_slots(custom_prompt, placement, role_key="research_manager")
        prompt_language = _resolve_language(config)
        try:
            prompt_template = _apply_relation_prompt_guard(
                get_prompt("research_manager_prompt", config=config),
                language=prompt_language,
            )
        except RelationPromptGuardError as exc:
            guard_failure = f"E-02 关系提示词守卫未通过：{exc}"
            _logger.error("[research_manager] %s", guard_failure)
            from tradingagents.agents.utils.decision_status import abstain_status

            return _blocked_manager_payload(
                investment_debate_state=investment_debate_state,
                report_manifest=report_manifest,
                fund_flow_guard=fund_flow_guard,
                decision_status=abstain_status(
                    reason_codes=["e02_relation_prompt_guard_failed"],
                    trade_action="NO_TRADE",
                    risk_status="BLOCKED",
                ).to_dict(),
                blocked_plan=f"{guard_failure}。状态=ABSTAIN，动作=NO_TRADE；已阻断进入 Trader 执行阶段。",
                manager_reason=guard_failure,
                run_integrity=run_integrity.to_dict(),
                claim_evidence_summary=claim_evidence_summary,
                consistency_check_passed=False,
                failed_checks=[guard_failure],
                evidence_verification=claims_verification,
                claim_cluster_metrics=claim_cluster_metrics,
                research_horizon=research_horizon,
            )
        base_prompt = prompt_template.format(
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
            actual_message_count=actual_message_count,
            actual_stages_desc=actual_stages_desc,
            tiebreak_status_desc=tiebreak_status_desc,
            challenges_text=challenges_text,
            challenge_verification_text=challenge_verification_text,
            battlefield_coverage_text=battlefield_coverage_text,
            **injection_slots,
        )
        relation_guard = _format_relation_prompt_guard(
            claim_cluster_metrics.get("relation_graph_status", RELATION_GRAPH_STATUS_PENDING),
            prompt_language,
        )
        base_prompt = f"{base_prompt}\n\n{relation_guard}"
        prompt = f"{horizon_ctx}\n\n{base_prompt}"

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
            market_data_context=effective_market_data_context,
            analysis_baseline_date=analysis_baseline_date,
            social_data_context=social_data_context,
        )

        manager_verdict = extract_and_validate_manager_verdict(
            raw_response=full_content,
            claims_verification=claims_verification,
            claims=claims,
            challenges=challenges,
            challenges_verification=challenges_verification,
            market_data_context=effective_market_data_context,
        )
        claim_evidence_summary = truth_evaluator.aggregate_claim_evidence(
            claims=claims,
            claims_verification=claims_verification,
            analysis_baseline_date=analysis_baseline_date or None,
            expected_symbol=expected_symbol or None,
            market_data_context=effective_market_data_context,
        )
        manager_verdict["claim_evidence_summary"] = claim_evidence_summary
        manager_verdict["horizon"] = research_horizon
        manager_verdict["research_horizon"] = research_horizon
        manager_verdict["evidence_relation_status"] = claim_cluster_metrics.get("relation_graph_status", "pending")
        manager_verdict["evidence_relation_reason"] = claim_cluster_metrics.get("relation_graph_reason", "")

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

        claim_weights = None
        credit_weight_audit = None
        if is_credit_weighting_enabled(investment_debate_state) or is_credit_weighting_enabled(state):
            # Live gate evaluation (fail-closed without h1b_gate_samples).
            # Do not hardcode system_gate_passed=False — that made flag-on still flat.
            from tradingagents.agents.utils.shadow_credit import (
                resolve_claim_credit_weights_for_manager,
            )

            historical_samples = state.get("h1b_gate_samples") or []
            if not isinstance(historical_samples, list):
                historical_samples = []
            weights_res = resolve_claim_credit_weights_for_manager(
                claims=claims,
                claim_evidence_summary=claim_evidence_summary,
                historical_samples=historical_samples,
                credit_weighting_enabled=True,
            )
            claim_weights = weights_res.get("claim_weights")
            credit_weight_audit = {
                "credit_weighting_active": bool(weights_res.get("credit_weighting_active", False)),
                "system_gate_passed": bool(weights_res.get("system_gate_passed", False)),
                "system_gate_status": weights_res.get("system_gate_status", "FAIL"),
                "recommendation": weights_res.get("recommendation", "KEEP_FALSE"),
                "bias_freeze_reasons": weights_res.get("bias_freeze_reasons") or {},
                "model_weights": weights_res.get("model_weights") or {},
                "global_fallback_shadow": bool(weights_res.get("global_fallback_shadow", True)),
            }
            if not credit_weight_audit["system_gate_passed"]:
                _logger.info(
                    "[research_manager] credit weighting stay flat: system_gate=%s recommendation=%s",
                    credit_weight_audit["system_gate_status"],
                    credit_weight_audit["recommendation"],
                )

        new_investment_debate_state = {
            **investment_debate_state,
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
            "claim_evidence_summary": claim_evidence_summary,
            "challenge_verification": challenges_verification,
            "report_manifest": report_manifest,
            "claim_cluster_metrics": claim_cluster_metrics,
            "evidence_relation_status": claim_cluster_metrics.get("relation_graph_status", "pending"),
            "evidence_relation_reduction": claim_cluster_metrics.get("relation_audit", {}),
            "independent_cluster_count": claim_cluster_metrics.get("independent_cluster_count", 0),
            "analyst_count": claim_cluster_metrics.get("analyst_count", 0),
            "verified_evidence_count": claim_cluster_metrics.get("verified_evidence_count", 0),
        }
        if claim_weights is not None:
            new_investment_debate_state["claim_weights"] = claim_weights
        if credit_weight_audit is not None:
            new_investment_debate_state["credit_weight_audit"] = credit_weight_audit

        # D-009 P0-1/P0-5b/P1-2: every successful terminal path emits canonical decision_status.
        vpa_ctx = effective_market_data_context.get("vpa_context") if isinstance(effective_market_data_context, dict) else None
        if (
            manager_verdict.get("position_pct") is not None
            and isinstance(vpa_ctx, dict)
            and vpa_ctx.get("reversal_state") == "reversal_confirmed"
        ):
            from tradingagents.agents.utils.decision_status import resolve_staged_entry_position
            manager_verdict["position_pct"] = resolve_staged_entry_position(
                manager_verdict["position_pct"],
                vpa_context=vpa_ctx,
            )

        terminal_status = status_from_manager_verdict(
            manager_verdict,
            prior_analysis_status=state.get("analysis_status"),
            investment_debate_state=new_investment_debate_state,
            claims_verification=claims_verification,
            claim_evidence_summary=claim_evidence_summary,
            focus_claim_ids=new_investment_debate_state.get("focus_claim_ids"),
            unresolved_claim_ids=unresolved_claim_ids,
            claims=claims,
            market_data_context=effective_market_data_context,
            vpa_context=vpa_ctx,
        )
        status_dict = terminal_status.to_dict()
        manager_verdict = {
            **manager_verdict,
            "decision_status": status_dict,
            "analysis_status": status_dict["analysis_status"],
            "trade_action": status_dict["trade_action"],
            "risk_status": status_dict["risk_status"],
            "confirmation_state": status_dict["confirmation_state"],
            "horizon": research_horizon,
            "research_horizon": research_horizon,
        }
        new_investment_debate_state["manager_verdict"] = manager_verdict

        payload = {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": final_plan,
            "manager_verdict": manager_verdict,
            "evidence_verification": claims_verification,
            "claim_evidence_summary": claim_evidence_summary,
            "challenge_verification": challenges_verification,
            "report_manifest": report_manifest,
            "decision_status": status_dict,
            "analysis_status": status_dict["analysis_status"],
            "trade_action": status_dict["trade_action"],
            "risk_status": status_dict["risk_status"],
            "confirmation_state": status_dict["confirmation_state"],
            "run_integrity": run_integrity.to_dict(),
        }
        return payload

    return research_manager_node
