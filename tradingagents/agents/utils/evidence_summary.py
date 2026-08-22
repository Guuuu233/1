"""Bounded, fact-dense evidence summaries of analyst reports.

Adjudicators receive compact first-hand excerpts of analyst reports rather than
the full reports, so the final verdict can be anchored to evidence strength
(KNOWN_ISSUES #2 / DAV-68 M2) without blowing up context. ``build_evidence_summary``
is the single owner of that extraction.

The extractor is deterministic — it never invokes an LLM — so a given report
always maps to the same summary and the behaviour is unit-testable. It follows
the KNOWN_ISSUES #2 suggested fix: evidence summaries keep verifiable facts
(numbers, dates, named events) and drop argumentation. The analyst's own
VERDICT direction is preserved as a *labeled* fact, because adjudicators are
asked to tally analyst verdicts but do not receive the full reports.
"""

from __future__ import annotations

import re

from tradingagents.agents.utils.debate_utils import extract_tagged_json, strip_tagged_json

DEFAULT_MAX_CHARS = 300
DEFAULT_DENSE_INPUT_MAX_CHARS = 2000
_ELLIPSIS = "…"

# Machine-readable blocks that are output protocol, not evidence.
_MACHINE_TAGS = ("VERDICT", "DEBATE_STATE", "RISK_STATE", "RISK_JUDGE", "MANAGER_VERDICT")

# A line counts as evidence-bearing when it carries a number, a percentage, a
# currency/quantifier unit, or a date-like token — the raw material an
# adjudicator needs to cross-check a claim against the underlying data.
_EVIDENCE_RE = re.compile(
    r"\d|%|％|亿|万|同比|环比|净流入|净流出|上涨|下跌|增速|毛利率|营收|利润|"
    r"元|港元|美元|日期|风险等级|结论倾向"
)
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")


def extract_verdict_direction(report: str) -> str:
    """Return the analyst's own direction from the VERDICT machine block, if any."""
    if not report:
        return ""
    payload = extract_tagged_json(report, "VERDICT")
    if not payload:
        payload = extract_tagged_json(report, "MANAGER_VERDICT")
    return str(payload.get("direction", "")).strip()


def strip_machine_blocks(report: str) -> str:
    """Remove every machine-readable block (VERDICT / DEBATE_STATE / RISK_* / MANAGER_VERDICT)."""
    text = report or ""
    for tag in _MACHINE_TAGS:
        text = strip_tagged_json(text, tag)
    return text


def _normalize_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    if not line:
        return ""
    # Flatten markdown table cells: "| a | b |" -> "a | b"
    if line.startswith("|") and line.endswith("|"):
        line = line.strip("|").strip()
    return line


def _is_evidence_line(line: str) -> bool:
    if len(line) < 4:
        return False
    return bool(_EVIDENCE_RE.search(line))


def build_dense_report_input(
    report: str,
    max_chars: int = DEFAULT_DENSE_INPUT_MAX_CHARS,
    role_name: str = "",
) -> tuple[str, str, int]:
    """Return structured high-density summary and key evidence excerpts for a report.

    Args:
        report: Full report text.
        max_chars: Maximum character limit for the extracted input.
        role_name: Name of the analyst role for labeling context.

    Returns:
        tuple of (input_text, mode, char_count)
        where mode is "full", "structured_dense_summary_and_excerpts", or "empty".
    """
    if not report or not str(report).strip():
        return "", "empty", 0

    raw_text = str(report).strip()
    raw_len = len(raw_text)
    if raw_len <= max_chars:
        return raw_text, "full", raw_len

    direction = extract_verdict_direction(raw_text)
    body = strip_machine_blocks(raw_text)

    lines: list[str] = []
    for raw in body.splitlines():
        line = _normalize_line(raw)
        if not line or _TABLE_SEPARATOR_RE.match(line):
            continue
        lines.append(line)

    evidence_lines = [ln for ln in lines if _is_evidence_line(ln)]
    chosen = evidence_lines if evidence_lines else lines

    seen: set[str] = set()
    selected_parts: list[str] = []
    current_length = 0

    prefix = f"[分析师结论：{direction}] " if direction else ""
    current_length += len(prefix)

    for line in chosen:
        if line in seen:
            continue
        seen.add(line)
        if current_length + len(line) + 2 > max_chars:
            remaining_budget = max_chars - current_length - len(_ELLIPSIS) - 2
            if remaining_budget > 20:
                selected_parts.append(line[:remaining_budget] + _ELLIPSIS)
            break
        selected_parts.append(line)
        current_length += len(line) + 1

    extracted_body = "\n".join(selected_parts) if selected_parts else body[:max_chars]
    final_text = f"{prefix}{extracted_body}".strip()
    return final_text, "structured_dense_summary_and_excerpts", len(final_text)


def build_evidence_summary(report: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Return a bounded, fact-dense evidence summary of one analyst report.

    The summary is composed of the report's evidence-bearing lines (those
    containing numbers / percentages / quantitative keywords), preserving
    document order, and is prefixed with the analyst's own direction when one
    is present so adjudicators can tally verdicts without the full report.

    Args:
        report: Full analyst report text (may be empty).
        max_chars: Hard cap on the returned summary length (exclusive of the
            direction prefix).

    Returns:
        A compact, single-paragraph evidence summary, or an empty string when
        the report is empty (callers may then omit the summary line entirely
        rather than inject a misleading placeholder).
    """
    if not report or not report.strip():
        return ""

    direction = extract_verdict_direction(report)
    body = strip_machine_blocks(report)

    lines: list[str] = []
    for raw in body.splitlines():
        line = _normalize_line(raw)
        if not line:
            continue
        if _TABLE_SEPARATOR_RE.match(line):
            continue
        lines.append(line)

    evidence_lines = [ln for ln in lines if _is_evidence_line(ln)]
    # If the report carries almost no numbers, fall back to a small leading
    # excerpt so the adjudicator still sees the analyst's core content.
    chosen = evidence_lines if evidence_lines else lines[:6]

    seen: set[str] = set()
    parts: list[str] = []
    for line in chosen:
        if line in seen:
            continue
        seen.add(line)
        parts.append(line)

    summary = "；".join(parts)
    if len(summary) > max_chars:
        summary = summary[: max_chars - len(_ELLIPSIS)].rstrip("；，、 ") + _ELLIPSIS

    if direction:
        return f"[分析师结论：{direction}] {summary}"
    return summary
