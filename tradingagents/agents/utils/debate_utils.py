from __future__ import annotations

import collections
import difflib
import json
import logging
import math
import re
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

from tradingagents.agents.utils.agent_states import is_v2_debate_enabled


logger = logging.getLogger(__name__)


class DebateProtocolError(RuntimeError):
    """Raised when debate protocol validation fails after retry."""

    def __init__(
        self,
        message: str,
        *,
        message_index: int | None = None,
        speaker: str | None = None,
        details: str | None = None,
        attempts: list[dict[str, Any]] | None = None,
    ):
        super().__init__(message)
        self.message_index = message_index
        self.speaker = speaker
        self.details = details
        self.attempts = attempts or []


VALID_BATTLEFIELDS: frozenset[str] = frozenset({
    "capital_flow",
    "sentiment_theme",
    "price_volume",
    "macro_policy",
    "fundamentals",
})


_MACHINE_LIST_FIELDS = (
    "responded_claim_ids",
    "new_claims",
    "resolved_claim_ids",
    "unresolved_claim_ids",
    "next_focus_claim_ids",
)
_MACHINE_TEXT_FIELDS = ("round_summary", "round_goal")
_MACHINE_FIELDS = frozenset((*_MACHINE_LIST_FIELDS, *_MACHINE_TEXT_FIELDS, "challenges", "self_win_prob"))
_MACHINE_CLAIM_FIELDS = frozenset(("claim", "evidence", "confidence", "target_claim_ids", "battlefield", "evidence_ids", "cluster_id", "canonical_event_id", "evidence_id"))
_MACHINE_CHALLENGE_FIELDS = frozenset(("challenge_id", "target_claim_id", "weakest_point", "evidence", "severity"))


def normalize_text(text: str) -> str:
    """Normalize text by stripping whitespace and punctuation and converting to lowercase."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^\w一-鿿]+", " ", text.lower()).strip()


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into Chinese characters and alphanumeric words."""
    if not isinstance(text, str):
        return []
    s = text.lower()
    return re.findall(r"[一-鿿]|[a-z0-9_]+", s)


def _token_ngrams(tokens: list[str], n: int = 2) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return [tuple(tokens)] if tokens else []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_claim_similarity(t1: str, t2: str) -> float:
    """Compute normalized n-gram and sequence similarity between two texts.

    Returns a float in [0.0, 1.0].
    Threshold >= 0.82 indicates duplicate / high-similarity paraphrase.
    """
    toks1 = tokenize_text(t1)
    toks2 = tokenize_text(t2)
    if not toks1 and not toks2:
        return 1.0
    if not toks1 or not toks2:
        return 0.0
    if toks1 == toks2:
        return 1.0

    s1 = "".join(toks1)
    s2 = "".join(toks2)
    seq_ratio = difflib.SequenceMatcher(None, s1, s2).ratio()

    # 1-gram & 2-gram Dice on tokens
    c1_1, c2_1 = collections.Counter(_token_ngrams(toks1, 1)), collections.Counter(_token_ngrams(toks2, 1))
    total_1 = sum(c1_1.values()) + sum(c2_1.values())
    dice_1 = (2.0 * sum((c1_1 & c2_1).values())) / total_1 if total_1 > 0 else 0.0

    c1_2, c2_2 = collections.Counter(_token_ngrams(toks1, 2)), collections.Counter(_token_ngrams(toks2, 2))
    total_2 = sum(c1_2.values()) + sum(c2_2.values())
    dice_2 = (2.0 * sum((c1_2 & c2_2).values())) / total_2 if total_2 > 0 else 0.0

    token_dice = 0.4 * dice_1 + 0.6 * dice_2
    return round(float(max(seq_ratio, dice_2, token_dice)), 4)


def extract_new_evidence_count(
    new_evidence_list: Iterable[Any] | None,
    prev_evidence_list: Iterable[Any] | None,
    threshold: float = 0.82,
) -> int:
    """Count how many evidence items in new_evidence_list are not duplicates of prev_evidence_list."""
    new_items = [str(item).strip() for item in (new_evidence_list or []) if str(item).strip()]
    prev_items = [str(item).strip() for item in (prev_evidence_list or []) if str(item).strip()]
    if not new_items:
        return 0
    if not prev_items:
        return len(new_items)

    new_count = 0
    for item in new_items:
        max_sim = max([compute_claim_similarity(item, prev) for prev in prev_items], default=0.0)
        if max_sim < threshold:
            new_count += 1
    return new_count


def _evaluate_claim_duplication(
    claim_payload: Mapping[str, Any],
    same_side_claims: Sequence[Mapping[str, Any]],
    prev_ev_list: Sequence[Any],
) -> tuple[bool, float, int, str | None]:
    """Evaluate whether a candidate new claim is a duplicate of same-side historical claims.

    Returns:
        (is_duplicate, max_similarity, new_evidence_count, matched_claim_id)
    """
    claim_text = str(claim_payload.get("claim", "")).strip()
    claim_ev = claim_payload.get("evidence") or []

    max_sim = 0.0
    matched_id = None
    for pc in same_side_claims:
        s = compute_claim_similarity(claim_text, str(pc.get("claim", "")))
        if s > max_sim:
            max_sim = s
            matched_id = str(pc.get("claim_id", "")).strip() or None

    new_ev_count = extract_new_evidence_count(claim_ev, prev_ev_list)

    is_duplicate = False
    if max_sim >= 0.82:
        is_duplicate = True
    elif max_sim >= 0.70 and new_ev_count == 0 and prev_ev_list:
        is_duplicate = True
    elif prev_ev_list and new_ev_count == 0 and all(
        max([compute_claim_similarity(str(e), str(pe)) for pe in prev_ev_list], default=0.0) >= 0.82
        for e in claim_ev
    ):
        is_duplicate = True

    return is_duplicate, max_sim, new_ev_count, matched_id


def _tagged_openings(text: str, tag: str) -> list[re.Match[str]]:
    if not isinstance(text, str):
        return []
    pattern = rf"<!--\s*{re.escape(tag)}\s*:"
    return list(re.finditer(pattern, text, flags=re.DOTALL))


def _tagged_occurrences(text: str, tag: str) -> list[re.Match[str]]:
    """Count same-tag machine block labels before validating their delimiter."""
    if not isinstance(text, str):
        return []
    pattern = rf"<!--\s*{re.escape(tag)}\b"
    return list(re.finditer(pattern, text, flags=re.DOTALL))


def _find_machine_block_close(text: str, start: int) -> int:
    """Return the first ``-->`` that is not inside a JSON string literal."""
    in_string = False
    escaped = False
    index = start
    while index < len(text):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif text.startswith("-->", index):
                return index
        index += 1
    return -1


def _machine_block_spans(text: str, tag: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    if not isinstance(text, str):
        return spans
    for opening in _tagged_openings(text, tag):
        close = _find_machine_block_close(text, opening.end())
        if close >= 0:
            spans.append((opening.start(), close + 3))
    return spans


def _looks_like_machine_block(text: str) -> bool:
    return bool(
        re.match(
            r"<!--\s*[A-Za-z_][A-Za-z0-9_-]*\s*:",
            text,
            flags=re.DOTALL,
        )
    )


def _get_marker_parse_status(text: str, tag: str, *, warn: bool) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(text, str):
        return None, "missing"
    occurrences = _tagged_occurrences(text, tag)
    openings = _tagged_openings(text, tag)
    if not openings:
        if warn:
            tag_pattern = rf"<!--\s*{re.escape(tag)}\b"
            category = "truncated" if re.search(tag_pattern, text or "", flags=re.DOTALL) else "missing"
            logger.warning("[debate_utils] %s parse warning (%s): machine block not accepted", tag, category)
        tag_pattern = rf"<!--\s*{re.escape(tag)}\b"
        has_marker = bool(re.search(tag_pattern, text or "", flags=re.DOTALL))
        return None, "invalid" if has_marker else "missing"
    if len(occurrences) > 1:
        if warn:
            category = "duplicate_malformed" if len(occurrences) != len(openings) else "duplicate"
            logger.warning(
                "[debate_utils] %s parse warning (%s): %d same-tag machine block labels found; rejecting all",
                tag,
                category,
                len(occurrences),
            )
        return None, "invalid"

    for opening in openings:
        payload_start = opening.end()
        closing_index = _find_machine_block_close(text, payload_start)
        if closing_index < 0:
            continue
        payload_text = text[payload_start:closing_index].strip()
        try:
            payload = json.loads(payload_text)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        remainder = text[closing_index + 3:].lstrip()
        if remainder and not _looks_like_machine_block(remainder):
            continue
        return payload, "valid"

    if warn:
        if not any(_find_machine_block_close(text, opening.end()) >= 0 for opening in openings):
            logger.warning(
                "[debate_utils] %s parse warning (truncated): closing marker is missing",
                tag,
            )
        else:
            logger.warning(
                "[debate_utils] %s parse warning (invalid_or_trailing_prose): machine block not accepted",
                tag,
            )
    return None, "invalid"


def _parse_tagged_json(text: str, tag: str, *, warn: bool) -> dict[str, Any] | None:
    payload, _ = _get_marker_parse_status(text, tag, warn=warn)
    return payload


def extract_tagged_json(text: str, tag: str) -> dict[str, Any]:
    if tag in {"DEBATE_STATE", "RISK_STATE", "RISK_JUDGE"}:
        return _parse_tagged_json(text, tag, warn=True) or {}
    pattern = rf"<!--\s*{re.escape(tag)}:\s*(\{{.*?\}})\s*-->"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def strip_tagged_json(text: str, tag: str) -> str:
    if not isinstance(text, str):
        return text
    if _parse_tagged_json(text, tag, warn=False) is None:
        return text.strip()
    for start, end in reversed(_machine_block_spans(text, tag)):
        text = text[:start] + text[end:]
    return text.strip()


def _warn_machine_validation(tag: str, category: str, detail: str) -> None:
    logger.warning("[debate_utils] %s validation warning (%s): %s", tag, category, detail)


def _normalize_machine_string_list(value: Any, tag: str, field_name: str, claim_index: int | None = None) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        location = f"claim {claim_index} field {field_name}" if claim_index is not None else field_name
        _warn_machine_validation(tag, "invalid_schema", f"{location} must be an array")
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _claim_confidence(value: Any, tag: str, claim_index: int) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        _warn_machine_validation(
            tag,
            "claim_confidence",
            f"claim {claim_index} confidence must be a finite number in [0, 1]",
        )
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError):
        _warn_machine_validation(
            tag,
            "claim_confidence",
            f"claim {claim_index} confidence must be a finite number in [0, 1]",
        )
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        _warn_machine_validation(
            tag,
            "claim_confidence",
            f"claim {claim_index} confidence must be a finite number in [0, 1]",
        )
        return None
    return confidence


def _sanitize_machine_payload(payload: Mapping[str, Any], tag: str) -> dict[str, Any] | None:
    unknown_fields = sorted(str(key) for key in payload if key not in _MACHINE_FIELDS)
    if unknown_fields:
        _warn_machine_validation(
            tag,
            "unknown_fields",
            f"unknown structured fields ignored: {', '.join(unknown_fields)}",
        )

    normalized: dict[str, Any] = {}
    for field_name in _MACHINE_LIST_FIELDS:
        value = payload.get(field_name)
        if value is None:
            normalized[field_name] = []
        elif not isinstance(value, list):
            _warn_machine_validation(
                tag,
                "invalid_schema",
                f"{field_name} must be an array",
            )
            return None
        else:
            normalized[field_name] = value

    for field_name in _MACHINE_TEXT_FIELDS:
        value = payload.get(field_name)
        if value is None:
            normalized[field_name] = ""
        elif not isinstance(value, str):
            _warn_machine_validation(
                tag,
                "invalid_schema",
                f"{field_name} must be a string",
            )
            return None
        else:
            normalized[field_name] = value

    raw_challenges = payload.get("challenges")
    if raw_challenges is None:
        normalized["challenges"] = []
    elif not isinstance(raw_challenges, list):
        _warn_machine_validation(
            tag,
            "invalid_schema",
            "challenges must be an array",
        )
        return None
    else:
        challenges: list[dict[str, Any]] = []
        for ch_index, raw_ch in enumerate(raw_challenges, start=1):
            if not isinstance(raw_ch, Mapping):
                _warn_machine_validation(
                    tag,
                    "invalid_schema",
                    f"challenge {ch_index} must be an object",
                )
                return None
            unknown_ch_fields = sorted(str(key) for key in raw_ch if key not in _MACHINE_CHALLENGE_FIELDS)
            if unknown_ch_fields:
                _warn_machine_validation(
                    tag,
                    "unknown_fields",
                    f"challenge {ch_index} unknown structured fields ignored: {', '.join(unknown_ch_fields)}",
                )

            ch_dict: dict[str, Any] = {}
            if "challenge_id" in raw_ch and raw_ch["challenge_id"] is not None:
                if isinstance(raw_ch["challenge_id"], str) and raw_ch["challenge_id"].strip():
                    ch_dict["challenge_id"] = raw_ch["challenge_id"].strip()
                else:
                    _warn_machine_validation(
                        tag,
                        "invalid_schema",
                        f"challenge {ch_index} challenge_id must be a non-empty string",
                    )
                    return None

            target_claim_id = raw_ch.get("target_claim_id")
            if target_claim_id is not None:
                if isinstance(target_claim_id, str):
                    ch_dict["target_claim_id"] = target_claim_id.strip()
                else:
                    _warn_machine_validation(
                        tag,
                        "invalid_schema",
                        f"challenge {ch_index} target_claim_id must be a string",
                    )
                    return None
            else:
                ch_dict["target_claim_id"] = ""

            weakest_point = raw_ch.get("weakest_point")
            if weakest_point is not None:
                if isinstance(weakest_point, str):
                    ch_dict["weakest_point"] = weakest_point.strip()
                else:
                    _warn_machine_validation(
                        tag,
                        "invalid_schema",
                        f"challenge {ch_index} weakest_point must be a string",
                    )
                    return None
            else:
                ch_dict["weakest_point"] = ""

            raw_ev = raw_ch.get("evidence")
            if raw_ev is None:
                ch_dict["evidence"] = []
            elif isinstance(raw_ev, list):
                ch_dict["evidence"] = [str(item).strip() for item in raw_ev if str(item).strip()]
            elif isinstance(raw_ev, str):
                s = raw_ev.strip()
                ch_dict["evidence"] = [s] if s else []
            else:
                _warn_machine_validation(
                    tag,
                    "invalid_schema",
                    f"challenge {ch_index} evidence must be an array or string",
                )
                return None

            severity = raw_ch.get("severity")
            if severity is not None:
                if isinstance(severity, str):
                    ch_dict["severity"] = severity.strip()
                else:
                    _warn_machine_validation(
                        tag,
                        "invalid_schema",
                        f"challenge {ch_index} severity must be a string",
                    )
                    return None
            else:
                ch_dict["severity"] = ""

            challenges.append(ch_dict)
        normalized["challenges"] = challenges

    if "self_win_prob" in payload and payload.get("self_win_prob") is not None:
        raw_self_win_prob = payload.get("self_win_prob")
        if isinstance(raw_self_win_prob, bool) or not isinstance(raw_self_win_prob, (int, float, Real)):
            _warn_machine_validation(
                tag,
                "invalid_self_win_prob",
                "self_win_prob must be a finite number in [0.0, 1.0]",
            )
            return None
        try:
            prob_float = float(raw_self_win_prob)
        except (TypeError, ValueError, OverflowError):
            _warn_machine_validation(
                tag,
                "invalid_self_win_prob",
                "self_win_prob must be a finite number in [0.0, 1.0]",
            )
            return None
        if not math.isfinite(prob_float) or not (0.0 <= prob_float <= 1.0):
            _warn_machine_validation(
                tag,
                "invalid_self_win_prob",
                "self_win_prob must be a finite number in [0.0, 1.0]",
            )
            return None
        normalized["self_win_prob"] = prob_float

    claims: list[dict[str, Any]] = []
    for claim_index, raw_claim in enumerate(normalized["new_claims"], start=1):
        if not isinstance(raw_claim, Mapping):
            _warn_machine_validation(
                tag,
                "invalid_claim",
                f"claim {claim_index} must be an object and was dropped",
            )
            continue
        unknown_claim_fields = sorted(str(key) for key in raw_claim if key not in _MACHINE_CLAIM_FIELDS)
        if unknown_claim_fields:
            _warn_machine_validation(
                tag,
                "unknown_fields",
                f"claim {claim_index} unknown structured fields ignored: {', '.join(unknown_claim_fields)}",
            )

        claim_text = raw_claim.get("claim")
        if not isinstance(claim_text, str) or not claim_text.strip():
            _warn_machine_validation(
                tag,
                "invalid_claim",
                f"claim {claim_index} needs non-empty claim text and was dropped",
            )
            continue
        confidence = _claim_confidence(raw_claim.get("confidence"), tag, claim_index)
        if confidence is None:
            continue
        claim_dict: dict[str, Any] = {
            "claim": claim_text.strip(),
            "evidence": _normalize_machine_string_list(
                raw_claim.get("evidence"), tag, "evidence", claim_index
            ),
            "confidence": confidence,
            "target_claim_ids": _normalize_machine_string_list(
                raw_claim.get("target_claim_ids"), tag, "target_claim_ids", claim_index
            ),
        }
        raw_battlefield = raw_claim.get("battlefield")
        if raw_battlefield is not None:
            claim_dict["battlefield"] = str(raw_battlefield).strip()
        raw_evidence_ids = raw_claim.get("evidence_ids")
        if raw_evidence_ids is not None:
            claim_dict["evidence_ids"] = _normalize_machine_string_list(
                raw_evidence_ids, tag, "evidence_ids", claim_index
            )
        raw_cluster_id = raw_claim.get("cluster_id")
        if raw_cluster_id is not None:
            claim_dict["cluster_id"] = str(raw_cluster_id).strip()
        raw_canonical = raw_claim.get("canonical_event_id")
        if raw_canonical is not None:
            claim_dict["canonical_event_id"] = str(raw_canonical).strip() if isinstance(raw_canonical, str) else raw_canonical
        raw_evidence_id = raw_claim.get("evidence_id")
        if raw_evidence_id is not None:
            claim_dict["evidence_id"] = str(raw_evidence_id).strip() if isinstance(raw_evidence_id, str) else raw_evidence_id
        claims.append(claim_dict)
    normalized["new_claims"] = claims
    return normalized


def safe_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default


def extract_risk_judge_result(text: str) -> dict[str, Any]:
    judge_payload = extract_tagged_json(text, "RISK_JUDGE")
    cleaned_response = strip_tagged_json(text, "RISK_JUDGE")
    parse_failed = not bool(judge_payload)

    verdict = str(judge_payload.get("verdict", "")).strip().lower()
    if verdict not in {"pass", "revise", "reject"}:
        parse_failed = True
        verdict = "reject"

    hard_constraints = [str(item).strip() for item in (judge_payload.get("hard_constraints") or []) if str(item).strip()]
    soft_constraints = [str(item).strip() for item in (judge_payload.get("soft_constraints") or []) if str(item).strip()]
    execution_preconditions = [
        str(item).strip() for item in (judge_payload.get("execution_preconditions") or []) if str(item).strip()
    ]
    de_risk_triggers = [str(item).strip() for item in (judge_payload.get("de_risk_triggers") or []) if str(item).strip()]
    revision_reason = str(judge_payload.get("revision_reason", "")).strip()

    if parse_failed:
        revision_reason = revision_reason or "风控裁决机读块解析失败，按拒绝处理"
        if cleaned_response:
            cleaned_response = f"{cleaned_response}\n\n[系统说明] 风控裁决机读块解析失败，已按拒绝处理。"
        else:
            cleaned_response = "风控裁决机读块解析失败，已按拒绝处理。"

    return {
        "judge_payload": judge_payload,
        "cleaned_response": cleaned_response,
        "verdict": verdict,
        "hard_constraints": hard_constraints,
        "soft_constraints": soft_constraints,
        "execution_preconditions": execution_preconditions,
        "de_risk_triggers": de_risk_triggers,
        "revision_reason": revision_reason,
        "parse_failed": parse_failed,
    }


def format_claims_for_prompt(
    claims: Iterable[Mapping[str, Any]] | None,
    focus_claim_ids: Iterable[str] | None = None,
    empty_message: str = "当前没有已登记 claim，本轮请先提出 1 到 2 条最关键 claim。",
) -> str:
    claim_list = list(claims or [])
    if not claim_list:
        return empty_message

    focus_set = {str(item) for item in (focus_claim_ids or []) if str(item).strip()}
    lines: list[str] = []
    for claim in claim_list:
        claim_id = str(claim.get("claim_id", "")).strip()
        status = str(claim.get("status", "open")).strip() or "open"
        speaker = str(claim.get("speaker", "")).strip() or "Unknown"
        summary = str(claim.get("claim", "")).strip() or "未提供 claim 文本"
        evidence = claim.get("evidence") or []
        evidence_text = "；".join(str(item).strip() for item in evidence if str(item).strip()) or "无明确证据"
        prefix = "* " if claim_id in focus_set else "- "
        lines.append(
            f"{prefix}{claim_id} [{status}] {speaker}: {summary} | 证据: {evidence_text}"
        )
    return "\n".join(lines)


def format_claim_subset_for_prompt(
    claims: Iterable[Mapping[str, Any]] | None,
    claim_ids: Iterable[str] | None,
    empty_message: str = "当前没有未解决 claim。",
) -> str:
    claim_id_set = {str(item) for item in (claim_ids or []) if str(item).strip()}
    if not claim_id_set:
        return empty_message
    subset = [claim for claim in (claims or []) if str(claim.get("claim_id", "")) in claim_id_set]
    return format_claims_for_prompt(subset, focus_claim_ids=claim_id_set, empty_message=empty_message)



def summarize_risk_feedback(feedback: Mapping[str, Any] | None) -> str:
    payload = feedback or {}
    verdict = str(payload.get("latest_risk_verdict", "")).strip()
    if not verdict:
        return "当前没有待处理的风控回退要求。"

    hard_constraints = payload.get("hard_constraints") or []
    soft_constraints = payload.get("soft_constraints") or []
    preconditions = payload.get("execution_preconditions") or []
    de_risk_triggers = payload.get("de_risk_triggers") or []

    return "\n".join(
        [
            f"风控裁决: {verdict}",
            f"是否要求重做: {'是' if payload.get('revision_required') else '否'}",
            f"打回原因: {payload.get('revision_reason', '未提供')}",
            f"硬约束: {'; '.join(str(item) for item in hard_constraints) if hard_constraints else '无'}",
            f"软约束: {'; '.join(str(item) for item in soft_constraints) if soft_constraints else '无'}",
            f"执行前提: {'; '.join(str(item) for item in preconditions) if preconditions else '无'}",
            f"降风险触发器: {'; '.join(str(item) for item in de_risk_triggers) if de_risk_triggers else '无'}",
        ]
    )


def default_round_goal(domain: str, next_count: int) -> str:
    goals = {
        "investment": [
            "建立最核心的正反两方 claim，并明确为何是现在。",
            "优先攻击对手最脆弱的假设，不要扩散议题。",
            "围绕时间窗口与触发条件，判断交易时机是否成立。",
            "围绕失败路径与失效条件，判断谁低估了回撤风险。",
            "检查剩余分歧是否仍有信息增量，否则准备收口。",
        ],
        "risk": [
            "建立最关键的执行风险 claim，明确风险预算冲突点。",
            "围绕仓位、止损、流动性约束，攻击对手最薄弱一环。",
            "判断哪些风险是可接受波动，哪些风险是硬性红线。",
            "逼迫双方给出可执行替代方案，而不是抽象立场。",
            "检查是否还存在未解决的高影响执行风险，否则准备收口。",
        ],
    }
    domain_key = domain if domain in goals else "investment"
    goal_list = goals[domain_key]
    index = min(max(next_count - 1, 0), len(goal_list) - 1)
    return goal_list[index]



def _message_is_accepted_valid(message: Mapping[str, Any]) -> bool:
    accepted = message.get("accepted")
    if accepted is None:
        accepted = message.get("accepted", True)
    status = message.get("parse_status") or message.get("parse_status")
    return bool(accepted) and status == "valid"


def _v2_message_targets_opponent(
    message: Mapping[str, Any],
    debate_state: Mapping[str, Any],
    claim_map: Mapping[str, Mapping[str, Any]],
    *,
    is_bull_msg: bool,
) -> bool:
    opponent_key = "Bear" if is_bull_msg else "Bull"
    opponent_stance = "bearish" if is_bull_msg else "bullish"

    def _is_opponent(cid: Any) -> bool:
        claim = claim_map.get(str(cid), {})
        return claim.get("speaker_key") == opponent_key or claim.get("stance") == opponent_stance

    responded = list(message.get("responded_claim_ids") or [])
    targets = list(message.get("target_claim_ids") or [])
    if any(_is_opponent(cid) for cid in responded + targets):
        return True

    msg_challenges = list(message.get("challenges") or [])
    message_index = message.get("message_index")
    speaker_key = "Bull" if is_bull_msg else "Bear"
    state_challenges = [
        ch for ch in (debate_state.get("challenges") or [])
        if isinstance(ch, Mapping)
        and (
            ch.get("message_index") == message_index
            or ch.get("speaker_key") == speaker_key
        )
    ]
    for ch in msg_challenges + state_challenges:
        if not isinstance(ch, Mapping):
            continue
        target_id = ch.get("target_claim_id") or ch.get("target_claim_id")
        if _is_opponent(target_id):
            return True
    return False


def _v2_manager_pre_gate_errors(
    debate_state: Mapping[str, Any],
    accepted_valid_messages: Sequence[Mapping[str, Any]],
    claim_list: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    skipped = bool(debate_state.get("tiebreak_skipped"))
    expected_count = 4 if skipped else 6
    expected_each = 2 if skipped else 3
    count = safe_int(debate_state.get("count", 0), 0)
    if count != expected_count or len(accepted_valid_messages) != expected_count:
        errors.append(
            f"v2 有效辩论轮次应为{expected_count}次 (当前count={count}, accepted valid={len(accepted_valid_messages)}, tiebreak_skipped={skipped})"
        )

    bull_msgs = [
        m for m in accepted_valid_messages
        if "Bull" in str(m.get("speaker", "")) or str(m.get("speaker_key", "")) == "Bull"
    ]
    bear_msgs = [
        m for m in accepted_valid_messages
        if "Bear" in str(m.get("speaker", "")) or str(m.get("speaker_key", "")) == "Bear"
    ]
    if len(bull_msgs) != expected_each or len(bear_msgs) != expected_each:
        errors.append(
            f"v2 多空双方有效发言次数不均等 (多头={len(bull_msgs)}, 空头={len(bear_msgs)}, 预期各{expected_each}次, tiebreak_skipped={skipped})"
        )

    claim_map = {str(c.get("claim_id", "")): c for c in claim_list if str(c.get("claim_id", ""))}
    challenge_errors = 0
    for idx, msg in enumerate(accepted_valid_messages):
        m_idx = int(msg.get("message_index", idx + 1) or (idx + 1))
        stage = str(msg.get("stage") or msg.get("protocol_stage") or "").strip().lower()
        is_opening = stage == "opening" or m_idx <= 2
        if is_opening:
            continue
        is_bull_msg = "Bull" in str(msg.get("speaker", "")) or str(msg.get("speaker_key", "")) == "Bull"
        if not _v2_message_targets_opponent(msg, debate_state, claim_map, is_bull_msg=is_bull_msg):
            challenge_errors += 1
            errors.append(
                f"v2 第 {m_idx} 次发言缺少针对对手的 challenge/responded_claim_ids"
            )
    if challenge_errors:
        errors.append("v2 Challenge 阶段必须包含至少一条针对对手 claim 的盘问")
    return errors


def validate_debate_preconditions(
    investment_debate_state: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Validate debate preconditions before Research Manager invocation.

    Rules (Contract 5):
    - count == 6 and accepted valid messages == 6.
    - Bull and Bear each have exactly 3 valid messages.
    - Subsequent messages (index 2..6) have non-empty responded_claim_ids targeting opponent claim and target_claim_ids targeting opponent claim.
    - Both Bull and Bear have claims in the claim ledger.

    Returns a list of error strings (empty if valid).
    """
    gate_errors: list[str] = []
    count = safe_int(investment_debate_state.get("count", 0), 0)
    round_messages = investment_debate_state.get("round_messages", []) or []
    accepted_valid_messages = [
        m for m in round_messages
        if _message_is_accepted_valid(m)
    ]
    claim_list = list(claims if claims is not None else investment_debate_state.get("claims", []) or [])
    v2_enabled = is_v2_debate_enabled(investment_debate_state)

    if v2_enabled:
        gate_errors.extend(
            _v2_manager_pre_gate_errors(
                investment_debate_state,
                accepted_valid_messages,
                claim_list,
            )
        )
    elif count != 6 or len(accepted_valid_messages) != 6:
        gate_errors.append(
            f"有效辩论轮次不足6次 (当前count={count}, accepted valid={len(accepted_valid_messages)})"
        )

    if not v2_enabled:
        bull_msgs = [
            m for m in accepted_valid_messages
            if "Bull" in str(m.get("speaker", "")) or str(m.get("speaker_key", "")) == "Bull"
        ]
        bear_msgs = [
            m for m in accepted_valid_messages
            if "Bear" in str(m.get("speaker", "")) or str(m.get("speaker_key", "")) == "Bear"
        ]
        if len(bull_msgs) != 3 or len(bear_msgs) != 3:
            gate_errors.append(
                f"多空双方有效发言次数不均等 (多头={len(bull_msgs)}, 空头={len(bear_msgs)}, 预期各3次)"
            )

    claim_map = {str(c.get("claim_id", "")): c for c in claim_list if str(c.get("claim_id", ""))}
    if not v2_enabled:
        for idx, msg in enumerate(accepted_valid_messages):
            m_idx = msg.get("message_index", idx + 1)
            if m_idx >= 2:
                responded = msg.get("responded_claim_ids") or []
                targets = msg.get("target_claim_ids") or []
                is_bull_msg = "Bull" in str(msg.get("speaker", "")) or str(msg.get("speaker_key", "")) == "Bull"
                opponent_key = "Bear" if is_bull_msg else "Bull"
                opponent_stance = "bearish" if is_bull_msg else "bullish"

                has_opp_responded = any(
                    cid in claim_map
                    and (claim_map[cid].get("speaker_key") == opponent_key or claim_map[cid].get("stance") == opponent_stance)
                    for cid in responded
                )
                if not has_opp_responded:
                    gate_errors.append(f"第 {m_idx} 次发言未在 responded_claim_ids 中回应对手 claim (responded: {responded})")

                has_opp_target = any(
                    cid in claim_map
                    and (claim_map[cid].get("speaker_key") == opponent_key or claim_map[cid].get("stance") == opponent_stance)
                    for cid in targets
                )
                if not has_opp_target:
                    gate_errors.append(f"第 {m_idx} 次发言未在 target_claim_ids 中针对对手 claim (targets: {targets})")

    has_bull_claims = any(c.get("speaker_key") == "Bull" or c.get("stance") == "bullish" for c in claim_list)
    has_bear_claims = any(c.get("speaker_key") == "Bear" or c.get("stance") == "bearish" for c in claim_list)
    if not has_bull_claims or not has_bear_claims:
        gate_errors.append(f"辩论 claim 账本缺失单方或双方论据 (多头claims={has_bull_claims}, 空头claims={has_bear_claims})")

    # Check E: Information gain / anti-repetition hard gate across same-side rounds
    for side_name, side_key, side_stance in (("多头", "Bull", "bullish"), ("空头", "Bear", "bearish")):
        side_claims = [
            c for c in claim_list
            if (c.get("speaker_key") == side_key or c.get("stance") == side_stance)
        ]
        for i in range(len(side_claims)):
            for j in range(i + 1, len(side_claims)):
                c1 = side_claims[i]
                c2 = side_claims[j]
                r1 = c1.get("round_index")
                r2 = c2.get("round_index")
                if r1 and r2 and r1 == r2:
                    continue
                if c1.get("claim_id") == c2.get("claim_id"):
                    continue
                sim = compute_claim_similarity(str(c1.get("claim", "")), str(c2.get("claim", "")))
                if sim >= 0.82:
                    gate_errors.append(
                        f"{side_name}发言存在跨轮重复观点 (最高相似度 {sim:.2f} >= 0.82: '{c1.get('claim', '')}' 与 '{c2.get('claim', '')}')，违反信息增量硬闸"
                    )

    for msg in accepted_valid_messages:
        m_idx = msg.get("message_index", 0)
        gain_score = msg.get("information_gain_score")
        if gain_score is not None and isinstance(gain_score, (int, float)) and gain_score <= 0.0 and m_idx >= 3:
            gate_errors.append(f"第 {m_idx} 次发言信息增量分数为 0，存在重复内容")

    return gate_errors


def _quarantine_rejected_machine_blocks(
    text: str,
    tags: str | Iterable[str] = ("DEBATE_STATE", "RISK_STATE"),
) -> str:
    """Isolate rejected machine blocks from transcript history while keeping prose."""
    if not isinstance(text, str):
        return text

    tag_list = (tags,) if isinstance(tags, str) else tuple(tags)

    spans: list[tuple[int, int]] = []
    for tag in tag_list:
        for match in _tagged_occurrences(text, tag):
            start = match.start()
            close_idx = _find_machine_block_close(text, match.end())
            if close_idx >= 0:
                end = close_idx + 3
            else:
                raw_close = text.find("-->", match.end())
                if raw_close >= 0:
                    end = raw_close + 3
                else:
                    end = len(text)
            spans.append((start, end))

    if not spans:
        return text.strip()

    spans.sort(key=lambda item: item[0])
    merged_spans: list[list[int]] = []
    for s, e in spans:
        if not merged_spans or s > merged_spans[-1][1]:
            merged_spans.append([s, e])
        else:
            merged_spans[-1][1] = max(merged_spans[-1][1], e)

    cleaned = text
    for s, e in reversed(merged_spans):
        cleaned = cleaned[:s] + cleaned[e:]

    return cleaned.strip()


def sanitize_debate_response(
    text: str,
    tags: str | Iterable[str] = ("DEBATE_STATE", "RISK_STATE"),
) -> str:
    """Isolate rejected or malformed machine blocks from debate responses while preserving prose."""
    return _quarantine_rejected_machine_blocks(text, tags)


def build_debate_report_manifest(
    reports_or_state: Mapping[str, Any],
    pass_info: Mapping[str, tuple[str, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build report input manifest for the 7 analyst inputs.

    ``passed`` is False when the report body is empty/failed/degraded so
    run-integrity and manager gates can see real failures (D-009 P0-1).
    """
    from tradingagents.agents.utils.run_integrity import is_failed_analyst_report

    report_keys = (
        ("macro_report", "macro_report"),
        ("market_report", "market_report"),
        ("sentiment_report", "sentiment_report"),
        ("news_report", "news_report"),
        ("fundamentals_report", "fundamentals_report"),
        ("smart_money_report", "smart_money_report"),
        ("volume_price_report", "volume_price_report"),
    )
    manifest: dict[str, dict[str, Any]] = {}
    pass_info = pass_info or {}
    for manifest_key, state_key in report_keys:
        content = reports_or_state.get(state_key, "")
        text = str(content or "")
        raw_length = len(text)
        failed, fail_reason = is_failed_analyst_report(text)
        info = pass_info.get(manifest_key)
        if info:
            mode, passed_chars = info
        else:
            mode = "full" if raw_length > 0 and not failed else ("failed" if failed else "empty")
            passed_chars = 0 if failed else raw_length
        if failed:
            mode = "failed"
            passed_chars = 0
        manifest[manifest_key] = {
            "length": raw_length,
            "raw_length": raw_length,
            "passed": not failed,
            "mode": mode,
            "pass_mode": mode,
            "passed_chars": passed_chars,
            "char_count": passed_chars,
            "fail_reason": fail_reason,
        }
    return manifest


def validate_debate_response(
    *,
    state: Mapping[str, Any],
    raw_response: str,
    speaker_key: str,
    stance: str,
    marker: str = "DEBATE_STATE",
    domain: str = "investment",
) -> tuple[bool, str, str, dict[str, Any] | None]:
    """Validate debate machine block format and protocol rules.

    Returns:
        (is_valid, parse_status, error_detail, sanitized_payload)
    """
    parsed_payload, raw_parse_status = _get_marker_parse_status(raw_response, marker, warn=True)
    if parsed_payload is None:
        if raw_parse_status == "missing":
            detail = f"未找到 <!-- {marker}: ... --> 机器块"
        elif raw_parse_status == "invalid":
            detail = f"{marker} 机器块 JSON 解析失败或格式不合法"
        else:
            detail = f"{marker} 机器块状态为 {raw_parse_status}"
        return False, raw_parse_status, detail, None

    payload = _sanitize_machine_payload(parsed_payload, marker)
    if payload is None:
        return False, "invalid", f"{marker} 机器块字段类型不符合规范", None

    current_count = safe_int(state.get("count", 0), 0)
    message_index = current_count + 1

    claims = [dict(item) for item in (state.get("claims", []) or []) if isinstance(item, Mapping)]
    claim_map = {
        str(item.get("claim_id", "")).strip(): item
        for item in claims
        if str(item.get("claim_id", "")).strip()
    }

    if domain == "investment":
        v2_enabled = is_v2_debate_enabled(state)
        explicit_stage = state.get("protocol_stage")
        if explicit_stage is not None and str(explicit_stage).strip():
            current_stage = str(explicit_stage).strip().lower()
        else:
            if message_index <= 2:
                current_stage = "opening"
            elif message_index in (3, 4):
                current_stage = "challenge"
            else:
                current_stage = "tiebreak"

        is_opening_stage = v2_enabled and current_stage == "opening"
        is_challenge_stage = v2_enabled and current_stage == "challenge"
        is_tiebreak_stage = v2_enabled and current_stage == "tiebreak"

        # Check A: Camp permission for resolved_claim_ids (cannot resolve opponent's claims)
        raw_resolved = _string_list(payload.get("resolved_claim_ids"))
        unauthorized = []
        for cid in raw_resolved:
            if cid in claim_map:
                c = claim_map[cid]
                is_opponent = (c.get("speaker_key") and c.get("speaker_key") != speaker_key) or (
                    c.get("stance") and c.get("stance") != stance
                )
                if is_opponent:
                    unauthorized.append(cid)
        if unauthorized:
            detail = f"发言人 {speaker_key} 试图单方面将对手 claim {unauthorized} 标记为 resolved，违反阵营权限契约"
            logger.warning("[debate_utils] protocol violation: %s", detail)
            return False, "invalid_protocol", detail, payload

        if is_opening_stage:
            # ── O2 Opening 阶段专属契约与五战场硬闸 ───────────────────────
            # 1. responded_claim_ids 必须为空数组 []
            raw_responded = _string_list(payload.get("responded_claim_ids"))
            if raw_responded:
                detail = (
                    f"Opening 阶段 ({speaker_key}) 必须为独立双盲立论，responded_claim_ids 必须为空数组 [] "
                    f"(当前 responded_claim_ids: {raw_responded})"
                )
                logger.warning("[debate_utils] protocol violation: %s", detail)
                return False, "invalid_protocol", detail, payload

            # 2. new_claims 必须恰好为 3 条
            new_claims_list = payload.get("new_claims") or []
            if len(new_claims_list) != 3:
                detail = (
                    f"Opening 阶段 ({speaker_key}) new_claims 数量必须恰好为 3 条 (当前数量: {len(new_claims_list)})"
                )
                logger.warning("[debate_utils] protocol violation: %s", detail)
                return False, "invalid_protocol", detail, payload

            # 3. 每一项 new_claim 的 target_claim_ids 必须为空数组 []
            for nc_idx, nc in enumerate(new_claims_list, start=1):
                t_ids = _string_list(nc.get("target_claim_ids"))
                if t_ids:
                    detail = (
                        f"Opening 阶段 ({speaker_key}) 第 {nc_idx} 条 claim 的 target_claim_ids 必须为空数组 [] "
                        f"(当前 target_claim_ids: {t_ids})"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

            # 4. 每一项 new_claim 必须指定属于 VALID_BATTLEFIELDS 的合法战场
            covered_battlefields = set()
            for nc_idx, nc in enumerate(new_claims_list, start=1):
                bf = nc.get("battlefield")
                if not bf or not isinstance(bf, str) or bf.strip() not in VALID_BATTLEFIELDS:
                    detail = (
                        f"Opening 阶段 ({speaker_key}) 第 {nc_idx} 条 claim 必须指定合法的 battlefield "
                        f"(有效战场: {sorted(VALID_BATTLEFIELDS)}, 当前: {repr(bf)})"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload
                covered_battlefields.add(bf.strip())

            # 5. 必须覆盖至少 3 个不同战场
            if len(covered_battlefields) < 3:
                detail = (
                    f"Opening 阶段 ({speaker_key}) new_claims 必须覆盖至少 3 个不同战场 "
                    f"(当前仅覆盖 {len(covered_battlefields)} 个战场: {sorted(covered_battlefields)})"
                )
                logger.warning("[debate_utils] protocol violation: %s", detail)
                return False, "invalid_protocol", detail, payload

        elif is_challenge_stage:
            # ── C2 Challenge 阶段专属契约与硬闸 ──────────────────────────
            # 1. new_claims 必须严格为空数组 []
            new_claims_list = payload.get("new_claims") or []
            if new_claims_list:
                detail = (
                    f"Challenge 阶段 ({speaker_key}) 必须聚焦反驳对手已有立论，禁止提出 new_claims，"
                    f"new_claims 必须严格为空数组 [] (当前 new_claims 数量: {len(new_claims_list)})"
                )
                logger.warning("[debate_utils] protocol violation: %s", detail)
                return False, "invalid_protocol", detail, payload

            # 2. challenges 至少包含 1 条
            raw_challenges = payload.get("challenges") or []
            if not raw_challenges:
                detail = (
                    f"Challenge 阶段 ({speaker_key}) 必须包含至少 1 条 challenges "
                    f"(当前 challenges 数量: {len(raw_challenges)})"
                )
                logger.warning("[debate_utils] protocol violation: %s", detail)
                return False, "invalid_protocol", detail, payload

            # 3. self_win_prob 必须存在且为有限 0..1
            if "self_win_prob" not in payload or payload.get("self_win_prob") is None:
                detail = (
                    f"Challenge 阶段 ({speaker_key}) 必须显式提供 self_win_prob (0.0 到 1.0 的有限数值)"
                )
                logger.warning("[debate_utils] protocol violation: %s", detail)
                return False, "invalid_protocol", detail, payload

            # 4. 每条 challenge 的字段校验 (target_claim_id, weakest_point, evidence, severity)
            target_ids_in_challenges = set()
            for ch_idx, ch in enumerate(raw_challenges, start=1):
                target_id = str(ch.get("target_claim_id") or "").strip()
                if not target_id:
                    detail = f"Challenge 阶段 ({speaker_key}) 第 {ch_idx} 条 challenge 的 target_claim_id 不能为空"
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

                if target_id not in claim_map:
                    detail = (
                        f"Challenge 阶段 ({speaker_key}) 第 {ch_idx} 条 challenge 的 target_claim_id '{target_id}' "
                        f"不存在于 claim 账本中 (现有 claim: {list(claim_map.keys())})"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

                target_claim = claim_map[target_id]
                is_opponent = (
                    target_claim.get("speaker_key") and target_claim.get("speaker_key") != speaker_key
                ) or (target_claim.get("stance") and target_claim.get("stance") != stance)
                if not is_opponent:
                    detail = (
                        f"Challenge 阶段 ({speaker_key}) 第 {ch_idx} 条 challenge 的 target_claim_id '{target_id}' "
                        f"不属于对手 claim (属于同侧 {speaker_key})"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

                if target_claim.get("status") == "resolved":
                    detail = (
                        f"Challenge 阶段 ({speaker_key}) 第 {ch_idx} 条 challenge 的 target_claim_id '{target_id}' "
                        f"状态已为 resolved，不可作为 challenge 目标"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

                weakest_point = str(ch.get("weakest_point") or "").strip()
                if not weakest_point:
                    detail = f"Challenge 阶段 ({speaker_key}) 第 {ch_idx} 条 challenge 的 weakest_point 不能为空"
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload
                if len(weakest_point) > 500:
                    detail = (
                        f"Challenge 阶段 ({speaker_key}) 第 {ch_idx} 条 challenge 的 weakest_point "
                        f"长度超出上限 500 字符 (当前长度: {len(weakest_point)})"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

                ch_ev = [str(item).strip() for item in (ch.get("evidence") or []) if str(item).strip()]
                if not ch_ev:
                    detail = (
                        f"Challenge 阶段 ({speaker_key}) 第 {ch_idx} 条 challenge 的 evidence 必须提供至少 1 条非空证据"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

                severity = str(ch.get("severity") or "").strip().lower()
                if severity not in {"fatal", "major", "minor"}:
                    detail = (
                        f"Challenge 阶段 ({speaker_key}) 第 {ch_idx} 条 challenge 的 severity "
                        f"必须为 fatal、major 或 minor 之一 (当前: {repr(ch.get('severity'))})"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

                target_ids_in_challenges.add(target_id)

            # 5. responded_claim_ids 必须包含所有被 challenge 的 target_claim_id
            raw_responded = set(_string_list(payload.get("responded_claim_ids")))
            if not target_ids_in_challenges.issubset(raw_responded):
                missing_targets = sorted(target_ids_in_challenges - raw_responded)
                detail = (
                    f"Challenge 阶段 ({speaker_key}) responded_claim_ids 必须包含所有被 challenge 的 target_claim_id "
                    f"(缺失: {missing_targets}, 当前 responded_claim_ids: {sorted(raw_responded)})"
                )
                logger.warning("[debate_utils] protocol violation: %s", detail)
                return False, "invalid_protocol", detail, payload

            # 6. duplicate 校验: 同speaker + 同target + (normalize_text(weakest_point)完全相同 或 similarity >= 0.82)
            historical_challenges = [
                c for c in (state.get("challenges", []) or [])
                if isinstance(c, Mapping)
                and (
                    (c.get("speaker_key") and c.get("speaker_key") == speaker_key)
                    or (c.get("speaker") and c.get("speaker") == speaker_key)
                )
            ]
            seen_challenges: list[dict[str, Any]] = list(historical_challenges)

            for ch_idx, ch in enumerate(raw_challenges, start=1):
                target_id = str(ch.get("target_claim_id") or "").strip()
                wp = str(ch.get("weakest_point") or "").strip()
                norm_wp = normalize_text(wp)

                for prev_ch in seen_challenges:
                    prev_target = str(prev_ch.get("target_claim_id") or "").strip()
                    if prev_target != target_id:
                        continue
                    prev_wp = str(prev_ch.get("weakest_point") or "").strip()
                    prev_norm_wp = normalize_text(prev_wp)

                    is_dup = False
                    sim = 0.0
                    if norm_wp and prev_norm_wp and norm_wp == prev_norm_wp:
                        is_dup = True
                        sim = 1.0
                    else:
                        sim = compute_claim_similarity(wp, prev_wp)
                        if sim >= 0.82:
                            is_dup = True

                    if is_dup:
                        detail = (
                            f"Challenge 阶段 ({speaker_key}) 第 {ch_idx} 条 challenge 针对同一目标 '{target_id}' "
                            f"存在重复弱点观点 (最高相似度 {sim:.2f} >= 0.82 或规范化文本完全相同: '{wp}' 与 '{prev_wp}')"
                        )
                        logger.warning("[debate_utils] protocol violation: %s", detail)
                        return False, "invalid_protocol", detail, payload

                seen_challenges.append(ch)

        elif is_tiebreak_stage:
            # ── B3 Tiebreak 阶段专属契约与硬闸 ──────────────────────────
            # 1. self_win_prob 必须存在且为有限 0..1
            if "self_win_prob" not in payload or payload.get("self_win_prob") is None:
                detail = (
                    f"Tiebreak 阶段 ({speaker_key}) 必须显式提供 self_win_prob (0.0 到 1.0 的有限数值)"
                )
                logger.warning("[debate_utils] protocol violation: %s", detail)
                return False, "invalid_protocol", detail, payload

            # 2. Tiebreak 阶段禁止提交纯 challenge 动作载荷 (challenges 属于 challenge 阶段)
            raw_challenges = payload.get("challenges") or []
            new_claims_list = payload.get("new_claims") or []
            if raw_challenges and not new_claims_list:
                detail = (
                    f"Tiebreak 阶段 ({speaker_key}) 不支持纯 challenge 载荷 (new_claims 为空且包含 challenges)；"
                    f"challenge 仅在 challenge 阶段提交"
                )
                logger.warning("[debate_utils] protocol violation: %s", detail)
                return False, "invalid_protocol", detail, payload

            # 3. duplicate claim 检查 (如果提供了 new_claims)
            same_side_claims = [
                c
                for c in claims
                if (c.get("speaker_key") and c.get("speaker_key") == speaker_key)
                or (c.get("stance") and c.get("stance") == stance)
            ]
            prev_ev_list = [
                ev
                for pc in same_side_claims
                for ev in (pc.get("evidence") or [])
                if str(ev).strip()
            ]
            if new_claims_list and same_side_claims:
                duplicate_claims = []
                for nc in new_claims_list:
                    is_dup, sim, _, _ = _evaluate_claim_duplication(
                        nc, same_side_claims, prev_ev_list
                    )
                    if is_dup:
                        duplicate_claims.append(str(nc.get("claim", "")).strip())
                if duplicate_claims:
                    detail = (
                        f"Tiebreak 阶段 ({speaker_key}) 观点与同侧历史观点重复: {duplicate_claims}"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

        else:
            # ── Legacy 规则 (及 v1 非 opening 逻辑) ───────────────────────
            # Check B: message_index >= 2 must respond to at least one un-resolved opponent claim
            if message_index >= 2:
                raw_responded = _string_list(payload.get("responded_claim_ids"))
                valid_opponent_responded = [
                    cid
                    for cid in raw_responded
                    if cid in claim_map
                    and (
                        (claim_map[cid].get("speaker_key") and claim_map[cid].get("speaker_key") != speaker_key)
                        or (claim_map[cid].get("stance") and claim_map[cid].get("stance") != stance)
                    )
                    and claim_map[cid].get("status") != "resolved"
                ]
                if not valid_opponent_responded:
                    opponent_open_ids = [
                        c["claim_id"]
                        for c in claims
                        if (
                            (c.get("speaker_key") and c.get("speaker_key") != speaker_key)
                            or (c.get("stance") and c.get("stance") != stance)
                        )
                        and c.get("status") != "resolved"
                    ]
                    detail = (
                        f"第 {message_index} 次发言 ({speaker_key}) 必须在 responded_claim_ids 中回应至少一条未解决的对手 claim ID "
                        f"(当前 responded: {raw_responded}, 合法对手未解决 claim: {opponent_open_ids})"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

            # Check C: message_index >= 2 must have at least one new claim targeting an opponent claim
            if message_index >= 2:
                new_claims_list = payload.get("new_claims") or []
                has_opponent_target = False
                for nc in new_claims_list:
                    t_ids = _string_list(nc.get("target_claim_ids"))
                    for tid in t_ids:
                        if tid in claim_map:
                            c = claim_map[tid]
                            is_opponent = (
                                c.get("speaker_key") and c.get("speaker_key") != speaker_key
                            ) or (c.get("stance") and c.get("stance") != stance)
                            if is_opponent:
                                has_opponent_target = True
                                break
                    if has_opponent_target:
                        break
                if not has_opponent_target:
                    opponent_all_ids = [
                        c["claim_id"]
                        for c in claims
                        if (c.get("speaker_key") and c.get("speaker_key") != speaker_key)
                        or (c.get("stance") and c.get("stance") != stance)
                    ]
                    detail = (
                        f"第 {message_index} 次发言 ({speaker_key}) 必须在 new_claims[].target_claim_ids 中指定至少一条对手 claim ID 作为反驳目标 "
                        f"(合法对手 claim: {opponent_all_ids})"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

            # Check D: message_index >= 3 (or same-side claims exist) must provide genuine information gain (anti-repetition hard gate)
            same_side_claims = [
                c
                for c in claims
                if (c.get("speaker_key") and c.get("speaker_key") == speaker_key)
                or (c.get("stance") and c.get("stance") == stance)
            ]
            if same_side_claims:
                new_claims_list = payload.get("new_claims") or []
                if not new_claims_list:
                    detail = (
                        f"第 {message_index} 次发言 ({speaker_key}) 必须在 new_claims 中提供至少一条具有信息增量的新观点"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

                prev_ev_list = [
                    ev
                    for pc in same_side_claims
                    for ev in (pc.get("evidence") or [])
                    if str(ev).strip()
                ]

                duplicate_claims = []
                valid_new_claims = []
                max_sims = []

                for nc in new_claims_list:
                    is_duplicate, sim, new_ev_count, _ = _evaluate_claim_duplication(
                        nc, same_side_claims, prev_ev_list
                    )
                    max_sims.append(sim)

                    if is_duplicate:
                        duplicate_claims.append(str(nc.get("claim", "")).strip())
                    else:
                        valid_new_claims.append(nc)

                if not valid_new_claims:
                    max_sim = max(max_sims) if max_sims else 0.0
                    detail = (
                        f"第 {message_index} 次发言 ({speaker_key}) 未提供有效信息增量：所有 new_claims 均与同侧历史观点重复或复用相同证据 "
                        f"(最高相似度 {max_sim:.2f} >= 0.82 或缺乏新证据，重复观点: {duplicate_claims})"
                    )
                    logger.warning("[debate_utils] protocol violation: %s", detail)
                    return False, "invalid_protocol", detail, payload

    return True, "valid", "", payload


def _record_unstructured_response(
    *,
    state: Mapping[str, Any],
    raw_response: str,
    speaker_label: str,
    speaker_key: str,
    history_key: str,
    speaker_field: str,
    store_current_response: bool,
    current_response_key: str | None = None,
    parse_status: str = "missing",
    error_detail: str = "",
    model_name: str | None = None,
    domain: str = "investment",
) -> dict[str, Any]:
    """Record an unaccepted attempt trace without advancing valid count or mutating valid transcript history."""
    cleaned_response = _quarantine_rejected_machine_blocks(raw_response)
    new_state = dict(state)
    current_count = safe_int(state.get("count", 0), 0)
    message_index = current_count + 1
    v2_enabled = is_v2_debate_enabled(state)
    explicit_stage = state.get("protocol_stage")
    if explicit_stage is not None and str(explicit_stage).strip():
        current_stage = str(explicit_stage).strip().lower()
    else:
        if message_index <= 2:
            current_stage = "opening"
        elif message_index in (3, 4):
            current_stage = "challenge"
        else:
            current_stage = "tiebreak"

    if v2_enabled and current_stage == "opening":
        debate_round = 1
    elif v2_enabled and current_stage == "challenge":
        debate_round = 2
    elif v2_enabled and current_stage == "tiebreak":
        debate_round = 3
    else:
        debate_round = (message_index - 1) // 2 + 1 if domain == "investment" else (message_index - 1) // 3 + 1

    if domain == "investment":
        attempt_record: dict[str, Any] = {
            "message_index": message_index,
            "debate_round": debate_round,
            "stage": current_stage,
            "protocol_stage": current_stage,
            "speaker": speaker_label,
            "speaker_key": speaker_key,
            "cleaned_prose": cleaned_response,
            "parse_status": parse_status,
            "accepted": False,
            "error_detail": error_detail,
            "responded_claim_ids": [],
            "new_claim_ids": [],
            "target_claim_ids": [],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "resolved": [],
            "unresolved": [],
            "round_summary": _fallback_summary(cleaned_response),
            "round_goal": state.get("round_goal") or default_round_goal(domain, message_index),
            "information_gain_score": 0.0,
            "duplicate_claim_ids": [],
            "duplicate_claims": [],
            "new_evidence_count": 0,
            "max_similarity": 1.0 if ("重复" in error_detail or "信息增量" in error_detail) else 0.0,
        }
        if model_name is not None:
            attempt_record["model_name"] = model_name

        round_messages = [dict(m) for m in (state.get("round_messages", []) or [])]
        round_messages.append(attempt_record)

        attempts = [dict(a) for a in (state.get("attempts", []) or [])]
        attempts.append(attempt_record)

        updates = {
            "current_speaker": speaker_key,
            speaker_field: speaker_key,
            "count": current_count,
            "round_messages": round_messages,
            "attempts": attempts,
            "blocked": True,
            "parse_status": parse_status,
            "block_reason": error_detail,
        }
        new_state.update(updates)
        return new_state
    else:
        argument = f"{speaker_label}: {cleaned_response}"
        round_messages = [dict(m) for m in (state.get("round_messages", []) or [])]
        round_msg: dict[str, Any] = {
            "message_index": message_index,
            "debate_round": debate_round,
            "speaker": speaker_label,
            "cleaned_prose": cleaned_response,
            "parse_status": parse_status,
            "accepted": True,
            "responded_claim_ids": [],
            "new_claim_ids": [],
            "target_claim_ids": [],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": [],
            "resolved": [],
            "unresolved": [],
            "round_summary": _fallback_summary(cleaned_response),
            "round_goal": state.get("round_goal") or default_round_goal(domain, message_index),
        }
        if model_name is not None:
            round_msg["model_name"] = model_name
        round_messages.append(round_msg)

        updates = {
            "history": _append_history(state.get("history", ""), argument),
            history_key: _append_history(state.get(history_key, ""), argument),
            "current_speaker": speaker_key,
            speaker_field: speaker_key,
            "count": message_index,
            "round_messages": round_messages,
        }
        if parse_status != "valid":
            updates["parse_status"] = parse_status
            updates["blocked"] = True

        if current_response_key:
            updates[current_response_key] = argument
        if store_current_response and current_response_key != "current_response":
            updates["current_response"] = argument
        new_state.update(updates)
        return new_state


def build_debate_claim_relation_graph(
    claims: Sequence[Mapping[str, Any]],
    *,
    prior_graph: Any = None,
    claim_canonical_anchors: Mapping[str, str] | None = None,
    analysis_baseline_date: str | None = None,
) -> Any:
    """Produce an immutable E-01 EvidenceRelationGraph from debate claims (Claim-to-Claim path).

    Contract:
    - SOURCE_REPETITION edges are produced between claims that share identical non-empty
      canonical_event_id and distinct non-empty evidence_ids (via build_canonical_source_repetition).
    - DERIVED_OBSERVATION edges are produced from explicit target_claim_ids on claims.
    - Endpoints are strictly claim_ids, matching the downstream closed-universe node universe.
    - Preserves any prior explicit relations without deriving edges from title similarity,
      overlapping keywords, verifier status, or LLM inference.
    """
    from tradingagents.agents.utils.evidence_relations import (
        EvidenceRelation,
        EvidenceRelationGraph,
        RelationType,
        build_canonical_source_repetition,
    )

    relations: list[EvidenceRelation] = []

    # 1. Ingest prior_graph if provided
    if prior_graph is not None:
        if isinstance(prior_graph, EvidenceRelationGraph):
            relations.extend(prior_graph.relations)
        elif isinstance(prior_graph, Mapping) and "relations" in prior_graph:
            raw_list = prior_graph.get("relations")
            if isinstance(raw_list, (list, tuple)):
                for r in raw_list:
                    if isinstance(r, EvidenceRelation):
                        relations.append(r)
                    elif isinstance(r, Mapping):
                        try:
                            relations.append(EvidenceRelation.from_dict(r))
                        except Exception:
                            pass
        elif isinstance(prior_graph, (list, tuple)):
            for r in prior_graph:
                if isinstance(r, EvidenceRelation):
                    relations.append(r)
                elif isinstance(r, Mapping):
                    try:
                        relations.append(EvidenceRelation.from_dict(r))
                    except Exception:
                        pass

    claim_list: list[dict[str, Any]] = []
    for c in claims:
        if isinstance(c, Mapping):
            cid = str(c.get("claim_id") or "").strip()
            if cid:
                entry = dict(c)
                if claim_canonical_anchors and cid in claim_canonical_anchors and not entry.get("canonical_event_id"):
                    entry["canonical_event_id"] = claim_canonical_anchors[cid]
                claim_list.append(entry)

    # 2. Derive DERIVED_OBSERVATION edges from explicit target_claim_ids
    for c in claim_list:
        src_id = str(c.get("claim_id") or "").strip()
        raw_targets = c.get("target_claim_ids")
        targets: list[str] = []
        if isinstance(raw_targets, (list, tuple)):
            for t in raw_targets:
                if t is not None:
                    s_t = str(t).strip()
                    if s_t:
                        targets.append(s_t)
        elif isinstance(raw_targets, str) and raw_targets.strip():
            targets.append(raw_targets.strip())

        raw_single = c.get("target_claim_id")
        if isinstance(raw_single, str) and raw_single.strip():
            s_single = raw_single.strip()
            if s_single not in targets:
                targets.append(s_single)

        for tgt_id in targets:
            meta: dict[str, Any] = {
                "producer": "debate_protocol_v2",
                "derivation_source": "target_claim_ids",
            }
            if c.get("stage"):
                meta["derivation_stage"] = str(c.get("stage"))
            if analysis_baseline_date:
                meta["timestamp"] = str(analysis_baseline_date)
            relations.append(
                EvidenceRelation(
                    source_id=src_id,
                    relation_type=RelationType.DERIVED_OBSERVATION,
                    target_id=tgt_id,
                    metadata=meta,
                )
            )

    # 3. Produce SOURCE_REPETITION edges from shared official canonical_event_id
    n = len(claim_list)
    for i in range(n):
        c1 = claim_list[i]
        for j in range(i + 1, n):
            c2 = claim_list[j]
            rep_rel = build_canonical_source_repetition(c1, c2)
            if rep_rel is not None:
                relations.append(rep_rel)

    return EvidenceRelationGraph.from_relations(relations)


def update_debate_state_with_payload(
    *,
    state: Mapping[str, Any],
    raw_response: str,
    speaker_label: str,
    speaker_key: str,
    stance: str,
    history_key: str,
    marker: str,
    claim_prefix: str,
    domain: str,
    speaker_field: str,
    store_current_response: bool = True,
    current_response_key: str | None = None,
    model_name: str | None = None,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    is_valid, parse_status, error_detail, payload = validate_debate_response(
        state=state,
        raw_response=raw_response,
        speaker_key=speaker_key,
        stance=stance,
        marker=marker,
        domain=domain,
    )
    if not is_valid or payload is None:
        return _record_unstructured_response(
            state=state,
            raw_response=raw_response,
            speaker_label=speaker_label,
            speaker_key=speaker_key,
            history_key=history_key,
            speaker_field=speaker_field,
            store_current_response=store_current_response,
            current_response_key=current_response_key,
            parse_status=parse_status,
            error_detail=error_detail,
            model_name=model_name,
            domain=domain,
        )

    current_count = safe_int(state.get("count", 0), 0)
    message_index = current_count + 1
    v2_enabled = is_v2_debate_enabled(state)
    explicit_stage = state.get("protocol_stage")
    if explicit_stage is not None and str(explicit_stage).strip():
        current_stage = str(explicit_stage).strip().lower()
    else:
        if message_index <= 2:
            current_stage = "opening"
        elif message_index in (3, 4):
            current_stage = "challenge"
        else:
            current_stage = "tiebreak"

    is_challenge_stage = v2_enabled and current_stage == "challenge"
    is_tiebreak_stage = v2_enabled and current_stage == "tiebreak"
    if v2_enabled and current_stage == "opening":
        debate_round = 1
    elif is_challenge_stage:
        debate_round = 2
    elif is_tiebreak_stage:
        debate_round = 3
    else:
        debate_round = (message_index - 1) // 2 + 1 if domain == "investment" else (message_index - 1) // 3 + 1

    claims = [dict(item) for item in (state.get("claims", []) or []) if isinstance(item, Mapping)]
    claim_map = {
        str(item.get("claim_id", "")).strip(): item
        for item in claims
        if str(item.get("claim_id", "")).strip()
    }

    cleaned_response = strip_tagged_json(raw_response, marker)
    claim_counter = safe_int(state.get("claim_counter", 0), 0)
    challenge_counter = safe_int(state.get("challenge_counter", 0), 0)
    challenges = [dict(item) for item in (state.get("challenges", []) or []) if isinstance(item, Mapping)]
    new_challenge_ids: list[str] = []

    responded_claim_ids = _filter_known_claim_ids(payload.get("responded_claim_ids"), claim_map)
    resolved_claim_ids = _filter_known_claim_ids(payload.get("resolved_claim_ids"), claim_map)
    unresolved_claim_ids = _filter_known_claim_ids(payload.get("unresolved_claim_ids"), claim_map)

    open_claim_ids = set(_string_list(state.get("open_claim_ids")))
    resolved_set = set(_string_list(state.get("resolved_claim_ids")))
    unresolved_set = set(_string_list(state.get("unresolved_claim_ids")))

    for claim_id in responded_claim_ids:
        if claim_id in claim_map and claim_map[claim_id].get("status") == "open":
            claim_map[claim_id]["status"] = "addressed"

    for claim_id in resolved_claim_ids:
        if claim_id in claim_map:
            claim_map[claim_id]["status"] = "resolved"
        open_claim_ids.discard(claim_id)
        unresolved_set.discard(claim_id)
        resolved_set.add(claim_id)

    for claim_id in unresolved_claim_ids:
        if claim_id in claim_map:
            claim_map[claim_id]["status"] = "unresolved"
        open_claim_ids.add(claim_id)
        unresolved_set.add(claim_id)
        resolved_set.discard(claim_id)

    # Process challenges in challenge stage
    if is_challenge_stage:
        for ch_payload in payload.get("challenges", []) or []:
            target_id = str(ch_payload.get("target_claim_id") or "").strip()
            wp = str(ch_payload.get("weakest_point") or "").strip()
            ev = [str(item).strip() for item in (ch_payload.get("evidence") or []) if str(item).strip()]
            sev = str(ch_payload.get("severity") or "major").strip().lower()
            if sev not in {"fatal", "major", "minor"}:
                sev = "major"
            challenge_counter += 1
            ch_id = f"CH-{challenge_counter}"
            challenge_entry = {
                "challenge_id": ch_id,
                "speaker": speaker_label,
                "speaker_key": speaker_key,
                "stance": stance,
                "target_claim_id": target_id,
                "weakest_point": wp,
                "evidence": ev,
                "severity": sev,
                "status": "open",
                "evidence_status": "unverified",
                "message_index": message_index,
                "debate_round": debate_round,
                "stage": current_stage,
            }
            challenges.append(challenge_entry)
            new_challenge_ids.append(ch_id)

    # Process new claims with per-claim duplicate rejection
    same_side_claims = [
        c
        for c in claims
        if (c.get("speaker_key") and c.get("speaker_key") == speaker_key)
        or (c.get("stance") and c.get("stance") == stance)
    ]
    prev_ev_list = [
        ev
        for pc in same_side_claims
        for ev in (pc.get("evidence") or [])
        if str(ev).strip()
    ]

    new_claim_ids = []
    all_target_claim_ids = []
    if is_challenge_stage:
        for ch in challenges:
            if ch.get("message_index") == message_index:
                t_id = ch.get("target_claim_id")
                if t_id and t_id in claim_map:
                    all_target_claim_ids.append(t_id)
    duplicate_claim_ids = []
    duplicate_claims = []
    valid_max_sims = []
    all_max_sims = []
    total_new_ev = 0

    for claim_payload in payload.get("new_claims", []) or []:
        claim_text = str(claim_payload.get("claim", "")).strip()
        if not claim_text:
            continue
        claim_counter += 1
        claim_id = f"{claim_prefix}-{claim_counter}"

        is_dup, sim, ev_count, matched_id = _evaluate_claim_duplication(
            claim_payload, same_side_claims, prev_ev_list
        )
        all_max_sims.append(sim)

        if is_dup:
            duplicate_claim_ids.append(claim_id)
            duplicate_claims.append(claim_text)
        else:
            evidence = [
                str(item).strip()
                for item in (claim_payload.get("evidence") or [])[:3]
                if str(item).strip()
            ]
            target_claim_ids = _filter_known_claim_ids(claim_payload.get("target_claim_ids"), claim_map)
            raw_target_claim_ids = _string_list(claim_payload.get("target_claim_ids"))
            claim_entry = {
                "claim_id": claim_id,
                "speaker": speaker_label,
                "speaker_key": speaker_key,
                "stance": stance,
                "claim": claim_text,
                "evidence": evidence,
                "confidence": claim_payload["confidence"],
                "status": "open",
                "target_claim_ids": raw_target_claim_ids,
                "round_index": message_index,
                "debate_round": debate_round,
                "message_index": message_index,
                "stage": current_stage,
            }
            if claim_payload.get("battlefield"):
                claim_entry["battlefield"] = claim_payload["battlefield"]
            if claim_payload.get("evidence_ids"):
                claim_entry["evidence_ids"] = list(claim_payload["evidence_ids"])
            if claim_payload.get("cluster_id"):
                claim_entry["cluster_id"] = claim_payload["cluster_id"]
            if claim_payload.get("canonical_event_id"):
                claim_entry["canonical_event_id"] = claim_payload["canonical_event_id"]
            if claim_payload.get("evidence_id"):
                claim_entry["evidence_id"] = claim_payload["evidence_id"]

            # Inline import to avoid circular dependency with claim_cluster (which imports normalize_text from debate_utils)
            from tradingagents.agents.utils.claim_cluster import assign_claim_cluster
            claim_entry = assign_claim_cluster(claim_entry)

            claims.append(claim_entry)
            claim_map[claim_id] = claim_entry
            open_claim_ids.add(claim_id)
            new_claim_ids.append(claim_id)
            all_target_claim_ids.extend(target_claim_ids)
            valid_max_sims.append(sim)
            total_new_ev += ev_count

    all_target_claim_ids = list(dict.fromkeys(all_target_claim_ids))

    next_focus_claim_ids = _filter_known_claim_ids(payload.get("next_focus_claim_ids"), claim_map)
    if not next_focus_claim_ids:
        preferred_ids = list(unresolved_set) + [cid for cid in open_claim_ids if cid not in unresolved_set]
        next_focus_claim_ids = preferred_ids[:2]

    summary = str(payload.get("round_summary", "")).strip() or _fallback_summary(cleaned_response)
    round_goal = str(payload.get("round_goal", "")).strip() or default_round_goal(
        domain, message_index
    )

    # Compute information gain metrics for round_messages
    if not same_side_claims:
        information_gain_score = 1.0
        max_similarity = 0.0
        new_evidence_count = sum(len(nc.get("evidence") or []) for nc in payload.get("new_claims", []) or [])
    else:
        max_similarity = max(valid_max_sims) if valid_max_sims else (max(all_max_sims) if all_max_sims else 0.0)
        information_gain_score = max(0.0, min(1.0, round(1.0 - max_similarity, 4)))
        new_evidence_count = total_new_ev

    round_messages = [dict(m) for m in (state.get("round_messages", []) or [])]
    round_msg = {
        "message_index": message_index,
        "debate_round": debate_round,
        "stage": current_stage,
        "protocol_stage": current_stage,
        "speaker": speaker_label,
        "speaker_key": speaker_key,
        "cleaned_prose": cleaned_response,
        "parse_status": "valid",
        "accepted": True,
        "responded_claim_ids": responded_claim_ids,
        "new_claim_ids": new_claim_ids,
        "target_claim_ids": all_target_claim_ids,
        "resolved_claim_ids": resolved_claim_ids,
        "unresolved_claim_ids": unresolved_claim_ids,
        "resolved": resolved_claim_ids,
        "unresolved": unresolved_claim_ids,
        "round_summary": summary,
        "round_goal": round_goal,
        "information_gain_score": information_gain_score,
        "duplicate_claim_ids": duplicate_claim_ids,
        "duplicate_claims": duplicate_claims,
        "new_evidence_count": new_evidence_count,
        "max_similarity": max_similarity,
    }
    if is_challenge_stage:
        round_msg["challenge_ids"] = new_challenge_ids
    if payload.get("self_win_prob") is not None:
        round_msg["self_win_prob"] = payload["self_win_prob"]
    if model_name is not None:
        round_msg["model_name"] = model_name
    if attempts:
        round_msg["attempts"] = [dict(a) for a in attempts]
    round_messages.append(round_msg)

    state_attempts = [dict(a) for a in (state.get("attempts", []) or [])]
    if attempts:
        for a in attempts:
            if not any(
                sa.get("attempt_index") == a.get("attempt_index")
                and sa.get("message_index") == a.get("message_index")
                for sa in state_attempts
            ):
                state_attempts.append(dict(a))
    else:
        state_attempts.append({
            "attempt_index": 1,
            "message_index": message_index,
            "debate_round": debate_round,
            "speaker": speaker_label,
            "parse_status": "valid",
            "accepted": True,
            "error_detail": "",
            "raw_response": raw_response,
        })

    argument = f"{speaker_label}: {cleaned_response}"
    new_state = dict(state)
    next_protocol_stage = current_stage
    if v2_enabled and domain == "investment":
        if current_stage == "opening":
            if message_index == 1:
                next_protocol_stage = "opening"
            elif message_index >= 2:
                next_protocol_stage = "challenge"
        elif current_stage == "challenge":
            if message_index <= 3:
                next_protocol_stage = "challenge"
            elif message_index >= 4:
                next_protocol_stage = "tiebreak"
        elif current_stage == "tiebreak":
            if message_index <= 5:
                next_protocol_stage = "tiebreak"
            elif message_index >= 6:
                next_protocol_stage = "manager"
        else:
            next_protocol_stage = state.get("protocol_stage", current_stage)

    trajectory_entry = {
        "stage": current_stage,
        "message_index": message_index,
        "speaker": speaker_label,
        "speaker_key": speaker_key,
        "stance": stance,
        "self_win_prob": float(payload["self_win_prob"]) if payload.get("self_win_prob") is not None else None,
        "debate_round": debate_round,
    }
    prior_trajectory = [dict(e) for e in (state.get("belief_trajectory") or []) if isinstance(e, Mapping)]
    belief_trajectory = prior_trajectory + [trajectory_entry]

    updates = {
        "history": _append_history(state.get("history", ""), argument),
        history_key: _append_history(state.get(history_key, ""), argument),
        "current_speaker": speaker_key,
        speaker_field: speaker_key,
        "count": message_index,
        "claims": claims,
        "claim_counter": claim_counter,
        "challenges": challenges,
        "challenge_counter": challenge_counter,
        "open_claim_ids": sorted(open_claim_ids),
        "resolved_claim_ids": sorted(resolved_set),
        "unresolved_claim_ids": sorted(unresolved_set),
        "focus_claim_ids": next_focus_claim_ids,
        "round_summary": summary,
        "round_goal": round_goal,
        "round_messages": round_messages,
        "attempts": state_attempts,
        "belief_trajectory": belief_trajectory,
    }
    if v2_enabled and domain == "investment":
        updates["protocol_stage"] = next_protocol_stage
    if "blocked" in new_state:
        del new_state["blocked"]
    if "parse_status" in new_state:
        del new_state["parse_status"]
    if "block_reason" in new_state:
        del new_state["block_reason"]
    if current_response_key:
        updates[current_response_key] = argument
    if store_current_response and current_response_key != "current_response":
        updates["current_response"] = argument

    prior_rel_graph = state.get("evidence_relation_graph")
    rel_graph = build_debate_claim_relation_graph(claims, prior_graph=prior_rel_graph)
    if rel_graph.relations:
        updates["evidence_relation_graph"] = rel_graph.to_dict()
    else:
        if prior_rel_graph is not None and not isinstance(prior_rel_graph, (dict, list, tuple)):
            updates["evidence_relation_graph"] = prior_rel_graph
        elif isinstance(prior_rel_graph, dict) and prior_rel_graph.get("relations"):
            updates["evidence_relation_graph"] = prior_rel_graph
        else:
            updates["evidence_relation_graph"] = None

    temp_state = dict(state)
    temp_state.update(updates)
    updates["debate_degenerate"] = detect_debate_degenerate(temp_state)
    new_state.update(updates)
    return new_state


def detect_debate_degenerate(state_or_inv_state: Mapping[str, Any]) -> bool:
    """Detect whether debate has degenerated into fixed dogmatic stances.

    Returns True when both sides maintain stationary self_win_prob across stages
    despite cross-examination and challenge stage progression.
    """
    if not isinstance(state_or_inv_state, Mapping):
        return False
    if not is_v2_debate_enabled(state_or_inv_state):
        return False

    inv_state = state_or_inv_state.get("investment_debate_state")
    if not isinstance(inv_state, Mapping):
        inv_state = state_or_inv_state

    # Extract belief trajectory or round messages
    trajectory = inv_state.get("belief_trajectory") or []
    if not trajectory:
        round_messages = inv_state.get("round_messages") or []
        trajectory = [
            {
                "speaker_key": m.get("speaker_key") or m.get("speaker"),
                "self_win_prob": m.get("self_win_prob"),
                "stage": m.get("stage") or m.get("protocol_stage"),
                "message_index": m.get("message_index"),
            }
            for m in round_messages
            if isinstance(m, Mapping) and m.get("self_win_prob") is not None
        ]

    bull_probs = [
        float(e["self_win_prob"])
        for e in trajectory
        if isinstance(e, Mapping)
        and (e.get("speaker_key") == "Bull" or str(e.get("speaker", "")).startswith("Bull"))
        and e.get("self_win_prob") is not None
    ]
    bear_probs = [
        float(e["self_win_prob"])
        for e in trajectory
        if isinstance(e, Mapping)
        and (e.get("speaker_key") == "Bear" or str(e.get("speaker", "")).startswith("Bear"))
        and e.get("self_win_prob") is not None
    ]

    # Need at least 2 probability samples per side (e.g. opening + challenge)
    if len(bull_probs) >= 2 and len(bear_probs) >= 2:
        bull_delta = max(bull_probs) - min(bull_probs)
        bear_delta = max(bear_probs) - min(bear_probs)
        if bull_delta < 0.001 and bear_delta < 0.001:
            # Check if challenges or evidence updates occurred
            challenges = inv_state.get("challenges") or []
            if challenges or len(trajectory) >= 4:
                return True

    return False


def _append_history(history: Any, argument: str) -> str:
    existing = str(history or "").strip()
    if not existing:
        return argument
    return f"{existing}\n{argument}"


def _filter_known_claim_ids(values: Any, claim_map: Mapping[str, Any]) -> list[str]:
    result = []
    for item in _string_list(values):
        if item in claim_map:
            result.append(item)
    return result


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for item in values:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _fallback_summary(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return "本轮未提取到有效摘要。"
    return compact[:120]


def build_empty_risk_debate_state() -> dict[str, Any]:
    return {
        "history": "",
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "latest_speaker": "",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "judge_decision": "",
        "count": 0,
        "claims": [],
        "focus_claim_ids": [],
        "open_claim_ids": [],
        "resolved_claim_ids": [],
        "unresolved_claim_ids": [],
        "round_summary": "",
        "round_goal": default_round_goal("risk", 1),
        "claim_counter": 0,
    }


def render_debate_prompt(
    template: str,
    *,
    is_opening_stage: bool,
    language: str = "zh",
    is_challenge_stage: bool = False,
) -> str:
    """Render debate prompt by handling stage framework and output contract sections.

    In legacy / non-opening mode, preserves the legacy 3-round framework and original machine block.
    In v2 Opening mode, removes legacy-only sections and injects the Opening double-blind contract
    (exactly 3 claims across 3 distinct valid battlefields, responded/target/resolved as empty arrays,
    clean machine block example with 0 INV IDs).
    In v2 Challenge mode, forbids new_claims and requires challenges + self_win_prob.
    """
    if not isinstance(template, str):
        return template

    if is_opening_stage:
        if language == "en":
            opening_framework = (
                "【Opening Stage Independent Double-Blind Opening Contract】:\n"
                "- Independent Opening: State exactly 3 core claims (new_claims count must be exactly 3), covering 3 distinct valid battlefields (capital_flow / sentiment_theme / price_volume / macro_policy / fundamentals).\n"
                "- Double-Blind Zero-Rebuttal: This stage is independent double-blind opening without rebuttals. responded_claim_ids must be [], new_claims[].target_claim_ids must be [], and resolved_claim_ids must be [].\n"
                "- Hard Data & Confidence: Each claim must be based on hard data and exact evidence sources. Claim confidence must be a finite number in 0.00-1.00, never a percentage."
            )
            opening_output_contract = (
                "At the very end append this machine-readable block:\n"
                '<!-- DEBATE_STATE: {{"responded_claim_ids": [], "new_claims": [{{"claim": "under 18 words", "evidence": ["evidence 1", "evidence 2"], "confidence": 0.72, "battlefield": "capital_flow", "target_claim_ids": []}}, {{"claim": "under 18 words", "evidence": ["evidence 1", "evidence 2"], "confidence": 0.75, "battlefield": "sentiment_theme", "target_claim_ids": []}}, {{"claim": "under 18 words", "evidence": ["evidence 1", "evidence 2"], "confidence": 0.80, "battlefield": "price_volume", "target_claim_ids": []}}], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], "round_summary": "under 30 words", "round_goal": "under 20 words"}} -->\n'
                "Output rules:\n"
                "- Opening Stage: responded_claim_ids must be [], resolved_claim_ids must be [];\n"
                "- new_claims must contain exactly 3 claims covering 3 distinct valid battlefields (capital_flow / sentiment_theme / price_volume / macro_policy / fundamentals);\n"
                "- Each new_claim must have a valid battlefield field and target_claim_ids must be [];\n"
                "- If an item is empty, return an empty array."
            )
        else:
            opening_framework = (
                "【Opening 阶段独立双盲立论契约】：\n"
                "- 独立立论：提出恰好3条核心立论（new_claims 数量必须恰好为3条），分别覆盖3个不同合法战场（capital_flow / sentiment_theme / price_volume / macro_policy / fundamentals）。\n"
                "- 双盲无反驳：本阶段为独立双盲立论，禁止反驳或提及对手观点。responded_claim_ids 必须为空数组 []，new_claims[].target_claim_ids 必须为空数组 []，resolved_claim_ids 必须为空数组 []。\n"
                "- 硬数据与置信度：每项 claim 必须基于硬数据与精确证据来源，claim confidence 为 0.00-1.00 的有限数值，严禁使用百分比。"
            )
            opening_output_contract = (
                "在正文末尾追加机读块（固定格式）：\n"
                '<!-- DEBATE_STATE: {{"responded_claim_ids": [], "new_claims": [{{"claim": "不超过28字", "evidence": ["证据1", "证据2"], "confidence": 0.72, "battlefield": "capital_flow", "target_claim_ids": []}}, {{"claim": "不超过28字", "evidence": ["证据1", "证据2"], "confidence": 0.75, "battlefield": "sentiment_theme", "target_claim_ids": []}}, {{"claim": "不超过28字", "evidence": ["证据1", "证据2"], "confidence": 0.80, "battlefield": "price_volume", "target_claim_ids": []}}], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], "round_summary": "不超过50字", "round_goal": "不超过30字"}} -->\n'
                "输出规则：\n"
                "- Opening阶段独立双盲立论：responded_claim_ids 必须为空数组 []，resolved_claim_ids 必须为空数组 []；\n"
                "- new_claims 必须恰好包含3条 Claim，且必须覆盖3个不同合法战场（capital_flow / sentiment_theme / price_volume / macro_policy / fundamentals）；\n"
                "- 每条 new_claim 必须指定合法 battlefield 字段，且 target_claim_ids 必须为空数组 []；\n"
                "- 若没有对应项，返回空数组。"
            )

        rendered = re.sub(
            r"<!--\s*STAGE_FRAMEWORK_START\s*-->.*?<!--\s*STAGE_FRAMEWORK_END\s*-->",
            opening_framework,
            template,
            flags=re.DOTALL,
        )
        rendered = re.sub(
            r"<!--\s*STAGE_OUTPUT_CONTRACT_START\s*-->.*?<!--\s*STAGE_OUTPUT_CONTRACT_END\s*-->",
            opening_output_contract,
            rendered,
            flags=re.DOTALL,
        )
        return rendered

    if is_challenge_stage:
        if language == "en":
            challenge_framework = (
                "【Challenge Stage Evidence Cross-Examination Contract】:\n"
                "- No new claims: Do not propose new_claims. new_claims must be strictly an empty array [].\n"
                "- Cross-examine opponent claims: challenges must contain at least 1 item. Each item must include target_claim_id (an unresolved opponent claim), weakest_point, at least 1 non-empty evidence item, and severity (fatal / major / minor).\n"
                "- Self assessment: self_win_prob must be a finite number in 0.0-1.0.\n"
                "- responded_claim_ids must include every target_claim_id that appears in challenges."
            )
            challenge_output_contract = (
                "At the very end append this machine-readable block:\n"
                '<!-- DEBATE_STATE: {{"responded_claim_ids": ["OPPONENT_CLAIM_ID"], "new_claims": [], "challenges": [{{"target_claim_id": "OPPONENT_CLAIM_ID", "weakest_point": "the weakest assumption", "evidence": ["hard evidence"], "severity": "major"}}], "self_win_prob": 0.55, "resolved_claim_ids": [], "unresolved_claim_ids": ["OPPONENT_CLAIM_ID"], "next_focus_claim_ids": ["OPPONENT_CLAIM_ID"], "round_summary": "under 30 words", "round_goal": "under 20 words"}} -->\n'
                "Output rules:\n"
                "- Challenge Stage: new_claims must be strictly an empty array [];\n"
                "- challenges must contain at least 1 item with target_claim_id, weakest_point, evidence, and severity (fatal/major/minor);\n"
                "- self_win_prob must be a finite number in 0.0-1.0;\n"
                "- responded_claim_ids must include every challenged target_claim_id."
            )
        else:
            challenge_framework = (
                "【Challenge 阶段证据盘问契约】：\n"
                "- 禁止立新：本阶段禁止提出新 Claim，new_claims 必须严格为空数组 []。\n"
                "- 必须盘问：challenges 至少包含 1 条；每条必须包含 target_claim_id（对手未解决 Claim）、weakest_point、至少 1 条非空 evidence、severity（fatal / major / minor）。\n"
                "- 必须自评：self_win_prob 必须是 0.0 到 1.0 的有限数值。\n"
                "- responded_claim_ids 必须包含所有被 challenge 的 target_claim_id。"
            )
            challenge_output_contract = (
                "在正文末尾追加机读块（固定格式）：\n"
                '<!-- DEBATE_STATE: {{"responded_claim_ids": ["OPPONENT_CLAIM_ID"], "new_claims": [], "challenges": [{{"target_claim_id": "OPPONENT_CLAIM_ID", "weakest_point": "不超过80字的最弱环节", "evidence": ["硬证据"], "severity": "major"}}], "self_win_prob": 0.55, "resolved_claim_ids": [], "unresolved_claim_ids": ["OPPONENT_CLAIM_ID"], "next_focus_claim_ids": ["OPPONENT_CLAIM_ID"], "round_summary": "不超过50字", "round_goal": "不超过30字"}} -->\n'
                "输出规则：\n"
                "- Challenge 阶段：new_claims 必须严格为空数组 []；\n"
                "- challenges 至少包含 1 条，且必须包含 target_claim_id、weakest_point、evidence、severity（fatal/major/minor）；\n"
                "- self_win_prob 必须是 0.0 到 1.0 的有限数值；\n"
                "- responded_claim_ids 必须包含所有被 challenge 的 target_claim_id。"
            )

        rendered = re.sub(
            r"<!--\s*STAGE_FRAMEWORK_START\s*-->.*?<!--\s*STAGE_FRAMEWORK_END\s*-->",
            challenge_framework,
            template,
            flags=re.DOTALL,
        )
        rendered = re.sub(
            r"<!--\s*STAGE_OUTPUT_CONTRACT_START\s*-->.*?<!--\s*STAGE_OUTPUT_CONTRACT_END\s*-->",
            challenge_output_contract,
            rendered,
            flags=re.DOTALL,
        )
        return rendered

    # Strip the stage marker comments while preserving the exact inner legacy text
    cleaned = re.sub(r"<!--\s*STAGE_FRAMEWORK_START\s*-->\n?", "", template)
    cleaned = re.sub(r"<!--\s*STAGE_FRAMEWORK_END\s*-->\n?", "", cleaned)
    cleaned = re.sub(r"<!--\s*STAGE_OUTPUT_CONTRACT_START\s*-->", "", cleaned)
    cleaned = re.sub(r"<!--\s*STAGE_OUTPUT_CONTRACT_END\s*-->", "", cleaned)
    return cleaned
