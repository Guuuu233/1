"""IntentParser: parse natural language query into structured trading intent."""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.agents.utils.context_utils import normalize_user_context
from tradingagents.prompts import get_prompt
from tradingagents.prompts.catalog import _resolve_language
from tradingagents.dataflows.config import get_config

_RESEARCH_HORIZON_VAR: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_RESEARCH_HORIZON_VAR", default=None
)


def bind_research_horizon(horizon: Optional[str]) -> contextvars.Token[Optional[str]]:
    """Bind the research horizon for the current context/thread."""
    return _RESEARCH_HORIZON_VAR.set(horizon)


def reset_research_horizon(token: contextvars.Token[Optional[str]]) -> None:
    """Reset the research horizon binding using a token."""
    _RESEARCH_HORIZON_VAR.reset(token)


def clear_research_horizon() -> None:
    """Clear the research horizon binding for the current context/thread."""
    _RESEARCH_HORIZON_VAR.set(None)


def get_bound_research_horizon() -> Optional[str]:
    """Get the currently bound research horizon, if any."""
    return _RESEARCH_HORIZON_VAR.get()


@contextmanager
def research_horizon_context(horizon: Optional[str]):
    """Context manager for temporary research horizon binding."""
    token = bind_research_horizon(horizon)
    try:
        yield
    finally:
        reset_research_horizon(token)


_HORIZON_LABELS_ZH: Dict[str, str] = {
    "short": "短线（1-2周，技术面主导）",
    "medium": "中线（1-3月，基本面主导）",
}

_HORIZON_LABELS_EN: Dict[str, str] = {
    "short": "Short-term (1-2 weeks, technicals-driven)",
    "medium": "Medium-term (1-3 months, fundamentals-driven)",
}

_HORIZON_LABELS = _HORIZON_LABELS_ZH

_UNBOUND_LABEL_ZH = "未绑定"
_UNBOUND_LABEL_EN = "Unbound"


def parse_intent(
    query: str,
    llm,
    fallback_ticker: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse natural language query into structured intent dict.

    Returns dict with keys: ticker, horizons, focus_areas, specific_questions, user_context, raw_query.
    Falls back gracefully to defaults if LLM output is unparseable.
    """
    config = get_config()
    system_msg = get_prompt("intent_parser_system", config=config)
    fallback_user_context = _extract_user_context_fallback(query)

    try:
        result = llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=query),
        ])
        raw = result.content.strip()
        # Clean markdown code fences more robustly (handle potential whitespace/newlines)
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
        
        # Simple cleanup for common LLM JSON errors
        raw = re.sub(r",\s*([\]}])", r"\1", raw)
        
        parsed = json.loads(raw) or {}
        parsed_user_context = normalize_user_context(parsed.get("user_context") or {})
        return {
            "raw_query": query,
            "ticker": parsed.get("ticker") or fallback_ticker or "",
            "horizons": ["short"],  # 固定单次运行，每个分析师用自己的自然时间窗口
            "focus_areas": parsed.get("focus_areas") if isinstance(parsed.get("focus_areas"), list) else [],
            "specific_questions": parsed.get("specific_questions") if isinstance(parsed.get("specific_questions"), list) else [],
            "user_context": _merge_inferred_user_context(parsed_user_context, fallback_user_context),
        }
    except Exception:
        return {
            "raw_query": query,
            "ticker": fallback_ticker or "",
            "horizons": ["short"],
            "focus_areas": [],
            "specific_questions": [],
            "user_context": fallback_user_context,
        }


def build_horizon_context(
    horizon: str,
    focus_areas: Optional[List[str]] = None,
    specific_questions: Optional[List[str]] = None,
    agent_type: Optional[str] = None,
    *,
    research_horizon: Optional[str] = None,
    run_horizon: Optional[str] = None,
) -> str:
    """Build the horizon context block to prepend to any agent's system prompt.

    Args:
        horizon: The node's observation window (专业观察窗).
        focus_areas: Specific analysis dimensions requested by user.
        specific_questions: Concrete questions requested by user.
        agent_type: Optional analyst/researcher type identifier.
        research_horizon: Explicit research horizon override (优先级: kwarg > binding > 未绑定).
        run_horizon: Alias for research_horizon.
    """
    config = get_config()
    template = get_prompt("horizon_context_block", config=config)
    lang = _resolve_language(config)

    focus_list = focus_areas if focus_areas is not None else []
    questions_list = specific_questions if specific_questions is not None else []

    # Resolution priority: explicit kwarg > thread binding > unbound
    target_research = research_horizon if research_horizon is not None else run_horizon
    if target_research is None:
        target_research = get_bound_research_horizon()

    if lang == "zh":
        labels = _HORIZON_LABELS_ZH
        unbound_label = _UNBOUND_LABEL_ZH
        focus_str = "、".join(focus_list) if focus_list else "无特殊关注"
        questions_str = "；".join(questions_list) if questions_list else "无"
    else:
        labels = _HORIZON_LABELS_EN
        unbound_label = _UNBOUND_LABEL_EN
        focus_str = ", ".join(focus_list) if focus_list else "None"
        questions_str = "; ".join(questions_list) if questions_list else "None"

    observation_horizon_label = labels.get(horizon, horizon)
    research_horizon_label = (
        labels.get(target_research, target_research)
        if target_research is not None
        else unbound_label
    )

    return template.format(
        research_horizon_label=research_horizon_label,
        observation_horizon_label=observation_horizon_label,
        research_horizon=research_horizon_label,
        observation_horizon=observation_horizon_label,
        horizon_label=observation_horizon_label,
        focus_areas_str=focus_str,
        specific_questions_str=questions_str,
        weight_hint="",
    )


def _merge_inferred_user_context(
    parsed_context: Dict[str, Any],
    fallback_context: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(parsed_context)
    for key, value in fallback_context.items():
        if key in {"cash_available", "current_position", "current_position_pct", "average_cost", "max_loss_pct"}:
            merged[key] = value
            continue
        if key == "constraints":
            existing = [str(item).strip() for item in merged.get("constraints", []) if str(item).strip()]
            for item in value:
                text = str(item).strip()
                if text and text not in existing:
                    existing.append(text)
            if existing:
                merged["constraints"] = existing
            continue
        if key not in merged or merged.get(key) in (None, "", []):
            merged[key] = value
    return normalize_user_context(merged)


def _extract_user_context_fallback(query: str) -> Dict[str, Any]:
    text = (query or "").strip()
    if not text:
        return {}

    context: Dict[str, Any] = {}

    objective_patterns = [
        (r"(想|准备|打算|计划).*建仓|想建仓|准备建仓|打算建仓", "建仓"),
        (r"(想|准备|打算|计划|考虑).*加仓|想加仓|准备加仓|考虑加仓", "加仓"),
        (r"(想|准备|打算|计划|考虑).*减仓|想减仓|准备减仓|考虑减仓", "减仓"),
        (r"(想|准备|打算|计划|考虑).*止损|想止损|准备止损|考虑止损", "止损"),
        (r"继续拿着|继续持有|拿着不动|持有中|被套|套牢", "持有处理"),
        (r"先观察|先观望|继续观察|先看看|观望", "观察"),
    ]
    for pattern, label in objective_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            context["objective"] = label
            break

    risk_keywords = {
        "保守": "保守",
        "稳健": "保守",
        "平衡": "平衡",
        "激进": "激进",
        "高风险": "激进",
    }
    for keyword, label in risk_keywords.items():
        if keyword in text:
            context["risk_profile"] = label
            break

    horizon_keywords = {
        "短线": "短线",
        "短期": "短线",
        "波段": "波段",
        "中线": "中线",
        "中期": "中线",
        "长期": "长期",
    }
    for keyword, label in horizon_keywords.items():
        if keyword in text:
            context["investment_horizon"] = label
            break

    position_keywords = {
        "满仓": 100.0,
        "重仓": 80.0,
        "半仓": 50.0,
        "轻仓": 20.0,
        "空仓": 0.0,
    }
    for keyword, pct in position_keywords.items():
        if keyword in text:
            context["current_position_pct"] = pct
            break

    cash_match = re.search(r"(?:可用资金|现金|仓位资金)[^\d]{0,8}(\d+(?:\.\d+)?)(万|亿)?", text, re.IGNORECASE)
    if cash_match:
        amount = cash_match.group(1)
        unit = cash_match.group(2) or ""
        context["cash_available"] = f"{amount}{unit}"

    patterns = {
        "average_cost": r"(?:成本价?|均价|持仓成本|买入价|在高位)\D{0,6}(\d+(?:\.\d+)?)",
        "max_loss_pct": r"(?:最大(?:亏损|回撤)|容忍亏损|止损(?:位)?|最多(?:只能)?亏)[^\d]{0,8}(\d+(?:\.\d+)?)\s*%",
        "current_position": r"(?:持有|现有|目前有)[^\d]{0,8}(\d+(?:\.\d+)?)\s*股",
        "current_position_pct": r"(?:仓位|持仓占比)[^\d]{0,8}(\d+(?:\.\d+)?)\s*%",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            context[key] = match.group(1)

    constraints: List[str] = []
    constraint_keywords = {
        "不加杠杆": "不加杠杆",
        "不融资": "不融资",
        "不追高": "不追高",
        "只做t+1": "只做T+1",
        "只做T+1": "只做T+1",
        "不能补仓": "不能补仓",
        "不接受隔夜": "不接受隔夜",
    }
    lowered = text.lower()
    for keyword, label in constraint_keywords.items():
        if keyword.lower() in lowered and label not in constraints:
            constraints.append(label)
    if constraints:
        context["constraints"] = constraints

    return normalize_user_context(context)
