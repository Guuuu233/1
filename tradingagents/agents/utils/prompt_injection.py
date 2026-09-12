"""Shared utility for custom-prompt injection into agent prompts.

Phase C scope: build_injection_slots() is the single owner of slot-filling logic.
Three agent factories (bull_researcher, bear_researcher, research_manager) call it;
they must not duplicate placement conditionals or separator logic.

D-015 E-02 prompt guard scope:
lint_custom_prompt() is the single authoritative pure deterministic linter shared
across all three defense layers:
  Layer 1: Save / migration entry validation (custom_prompt_service)
  Layer 2: Job launch resolved bundle freeze (_resolve_and_freeze_custom_prompts in api/main.py)
  Layer 3: Research manager assembly fail-closed (research_manager.py)

Placement semantics (anchored in zh.py template markers):
- "before_data"  : slot fires between role/priority block and the first data field.
- "after_data"   : slot fires between last data field and built-in output requirements.

When custom_prompt is empty (switch off, or role has no text) both slots return "",
which means the zh.py template expands to the same bytes as before this feature
existed — T2 (byte-identity when switch off) is guaranteed here, not in the callers.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)

Placement = Literal["before_data", "after_data"]
_VALID_PLACEMENTS: tuple[str, ...] = ("before_data", "after_data")

# Single authoritative default — all callers must import and use this constant
# rather than hard-coding a string, so a future placement change is one-line.
DEFAULT_PLACEMENT: Placement = "after_data"

# The 5 roles eligible for custom prompt injection under D-015
INJECTABLE_ROLES: tuple[str, ...] = (
    "bull_researcher",
    "bear_researcher",
    "research_manager",
    "trader",
    "risk_manager",
)


class PromptGuardVerdict(str, Enum):
    VIOLATION = "VIOLATION"
    SAFE_NEGATION_OR_UNRELATED = "SAFE_NEGATION_OR_UNRELATED"
    AMBIGUOUS = "AMBIGUOUS"


REASON_MACHINE_IDENTIFIER = "machine_identifier_zero_tolerance"
REASON_EVIDENCE_INDEPENDENCE_VIOLATION = "e02_evidence_independence_violation"
REASON_AMBIGUOUS_WEIGHTING = "ambiguous_weighting_instruction"
REASON_SAFE = "safe_negation_or_unrelated"
REASON_EMPTY = "empty"


@dataclass(frozen=True)
class PromptGuardResult:
    verdict: PromptGuardVerdict
    reason_code: str
    detail: str


# --- E-02 Guard Deterministic Regular Expressions ---

# 1. Machine-readable identifiers (Zero tolerance under D-015 §4)
RE_MACHINE_ID = re.compile(r"cluster_id|independent_cluster_count", re.IGNORECASE)

# 2. Safe domain weight usages (Financial position, index, factor, market cap, volatility weights)
RE_SAFE_DOMAIN = re.compile(
    r"(?:"
    r"(?:仓位|持仓|资产|组合|投资组合|头寸|资金)\s*(?:权重|加权|比例)|"
    r"(?:指数|沪深300|中证\s*\d+|上证\s*\d+|标普|纳斯达克)\s*(?:权重|加权|成份)|"
    r"(?:行业|板块|赛道)\s*(?:权重|加权|配置)|"
    r"(?:因子|多因子|风格因子|Alpha|动量|价值|成长|质量|低波动|基本面因子)\s*(?:权重|加权|模型)|"
    r"(?:市值|流通市值)\s*(?:权重|加权)|"
    r"(?:等权重|等权)|"
    r"(?:风险|波动率|VaR|流动性)\s*(?:权重|加权|预算)|"
    r"(?:position|portfolio|asset|holding|index|factor|multi-factor|market\s*cap|cap-weighted|risk|volatility|equal)\s*(?:weight|weighted|weighting|weights)"
    r")",
    re.IGNORECASE,
)

# 3. Explicit negation patterns (D-015 §4)
RE_NEGATION_CLAUSE = re.compile(
    r"(?:"
    r"(?:不要|禁止|严禁|不得|切勿|绝不|无需|不宜|不可|不能|不按|不以|免于|不应)\s*(?:对|给|为|按|根据|以)?.*?(?:加权|计票|汇总|排序|赋权|投票|权重)|"
    r"(?:do\s+not|never|prohibit(?:ed)?|forbid(?:den)?|must\s+not|should\s+not|cannot|no)\s+.*?(?:weight|tally|vote|rank|aggregate)"
    r")",
    re.IGNORECASE,
)

# 4. Direct violation tally keywords
RE_VIOLATION_TALLY = re.compile(
    r"(?:计票|独立票|去重计票|按权重计票|按分析师计票|按证据簇计票|按命题计票|"
    r"tally\b|independent\s+vote)",
    re.IGNORECASE,
)

# 5. Action + Object violation combinations
RE_VIOLATION_ACTION_OBJECT = re.compile(
    r"(?:"
    r"(?:按|给|对|根据|以)?\s*(?:各|所有)?\s*(?:分析师|命题|证据簇|证据组|claim[s]?|cluster[s]?)\s*(?:进行)?\s*(?:权重|加权|赋权|计权|计票|投票|排序|汇总)|"
    r"(?:按|根据|以)\s*(?:分析师|命题|证据簇|claim[s]?|cluster[s]?)\s*(?:权重|排序|汇总|计票)|"
    r"(?:加权|计票|赋权|计权|汇总|排序|投票)\s*(?:各|所有)?\s*(?:分析师|命题|证据簇|claim[s]?|cluster[s]?)|"
    r"(?:发言人数|发言者人数|analyst_count)\s*(?:作为|当作|用于|计算)?\s*(?:权重|计票|独立)|"
    r"(?:weight|tally|vote|rank|aggregate)\s+(?:each|all|the)?\s*(?:analyst[s]?|claim[s]?|cluster[s]?|independent\s+vote[s]?)|"
    r"(?:analyst[s]?|claim[s]?|cluster[s]?)\s+(?:weight|weights|weighted|weighting|tally|votes)"
    r")",
    re.IGNORECASE,
)

# 6. Ambiguous combinations (vague weighting of opinions without explicit evidence voting)
RE_AMBIGUOUS = re.compile(
    r"(?:"
    r"(?:各|所有)?\s*(?:分析师|研究员|各方|各角色)?\s*(?:观点|意见|结论|建议)\s*(?:进行)?\s*(?:加权|综合加权|加权研判|加权分析|加权评估|权重参考|排序|汇总)|"
    r"(?:加权|综合加权|加权研判|加权分析|加权评估|排序|汇总)\s*(?:各|所有)?\s*(?:分析师|研究员|各方|各角色)?\s*(?:观点|意见|结论|建议)|"
    r"(?:综合各分析师观点进行加权研判)|"
    r"(?:weight|weighting|rank)\s+(?:each|all|the)?\s*(?:analyst[s]?|researcher[s]?)?\s*(?:opinion[s]?|view[s]?|perspective[s]?)"
    r")",
    re.IGNORECASE,
)


def lint_custom_prompt(prompt_text: str) -> PromptGuardResult:
    """Pure deterministic linter for E-02 prompt independence constraints.

    Returns a PromptGuardResult with three-state verdict:
      - VIOLATION: Unambiguous violation of evidence independence rules or machine identifier.
      - SAFE_NEGATION_OR_UNRELATED: Explicit negation, domain-safe weighting, or unrelated text.
      - AMBIGUOUS: Vague weighting description requiring user clarification before save / fail-closed at runtime.
    """
    text = (prompt_text or "").strip()
    if not text:
        return PromptGuardResult(
            PromptGuardVerdict.SAFE_NEGATION_OR_UNRELATED,
            REASON_EMPTY,
            "提示词为空",
        )

    # 1. Zero tolerance on machine-readable identifiers (D-015 §4)
    if RE_MACHINE_ID.search(text):
        return PromptGuardResult(
            PromptGuardVerdict.VIOLATION,
            REASON_MACHINE_IDENTIFIER,
            "检测到机读标识 (cluster_id / independent_cluster_count)",
        )

    # Split into sentences/clauses
    sentences = re.split(r"[\r\n。！？!?；;]+", text)
    ambiguous_found: PromptGuardResult | None = None

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        clauses = re.split(r"[,，、]+", sentence)

        # If sentence as a whole has explicit negation
        if RE_NEGATION_CLAUSE.search(sentence):
            has_affirmative_violation = False
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                if RE_NEGATION_CLAUSE.search(clause):
                    continue
                sanitized_clause = RE_SAFE_DOMAIN.sub("", clause)
                if RE_VIOLATION_TALLY.search(sanitized_clause) or RE_VIOLATION_ACTION_OBJECT.search(sanitized_clause):
                    has_affirmative_violation = True
                    break
            if not has_affirmative_violation:
                continue

        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue

            if RE_NEGATION_CLAUSE.search(clause):
                continue

            sanitized = RE_SAFE_DOMAIN.sub("", clause)

            if RE_VIOLATION_TALLY.search(sanitized) or RE_VIOLATION_ACTION_OBJECT.search(sanitized):
                return PromptGuardResult(
                    PromptGuardVerdict.VIOLATION,
                    REASON_EVIDENCE_INDEPENDENCE_VIOLATION,
                    "检测到违规动宾组合（如分析师加权/计票/汇总/排序）",
                )

            if RE_AMBIGUOUS.search(sanitized):
                ambiguous_found = PromptGuardResult(
                    PromptGuardVerdict.AMBIGUOUS,
                    REASON_AMBIGUOUS_WEIGHTING,
                    "提示词语义不确定（包含分析师观点与加权/排序描述），无法判定是否涉及计票",
                )

    if ambiguous_found:
        return ambiguous_found

    return PromptGuardResult(
        PromptGuardVerdict.SAFE_NEGATION_OR_UNRELATED,
        REASON_SAFE,
        "未检测到违规项",
    )


def _render_slot(custom_prompt: str) -> str:
    """Return '' for empty text; 'text\\n\\n' otherwise.

    The trailing double-newline is the only separator this module emits.
    Callers must not add their own separators.
    """
    if not custom_prompt:
        return ""
    return f"{custom_prompt}\n\n"


def build_injection_slots(
    custom_prompt: str,
    placement: Placement,
    role_key: str = "",
) -> dict[str, str]:
    """Return the two zh.py slot values for a given prompt + placement.

    Always returns both keys so callers can unconditionally unpack into
    .format(). The slot that is NOT active always gets "".

    Args:
        custom_prompt: Resolved text for this role (empty string = no injection).
        placement: "before_data" or "after_data".
        role_key: Agent role identifier used only for logging (no effect on slots).

    Returns:
        {"custom_prompt_before_data": str, "custom_prompt_after_data": str}

    Raises:
        ValueError: If placement is not one of the two valid values.
    """
    if placement not in _VALID_PLACEMENTS:
        raise ValueError(
            f"[prompt_injection] Unknown placement {placement!r}; "
            f"must be one of {_VALID_PLACEMENTS}"
        )

    rendered = _render_slot(custom_prompt)

    if placement == "before_data":
        slots = {
            "custom_prompt_before_data": rendered,
            "custom_prompt_after_data": "",
        }
    else:  # after_data
        slots = {
            "custom_prompt_before_data": "",
            "custom_prompt_after_data": rendered,
        }

    # Log every call so each debate round is traceable; suppress prompt body.
    logger.info(
        "[prompt_injection] role=%s placement=%s injected=%s length=%d hash=%s",
        role_key or "unknown",
        placement,
        bool(custom_prompt),
        len(custom_prompt),
        hashlib.sha256(custom_prompt.encode("utf-8")).hexdigest()[:12] if custom_prompt else "none",
    )

    return slots
