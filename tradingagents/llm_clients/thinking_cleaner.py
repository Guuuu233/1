"""Strip AI inner-monologue ("thinking") traces from report text.

Some models emit their private reasoning as plain sentences inside the
content stream on long tasks (e.g. "Let me think...", "Hmm, wait,",
"I think ..."). These are not report content. This module detects such
lines and removes them (or trims the thinking prefix) before a report is
persisted, while leaving formal markdown, prose and Chinese content intact.
"""

from __future__ import annotations

import re

# Meta-process narration → the whole line is dropped ("let me think",
# "I need to reconsider", "让我想想").
_PROCESS_NARRATION_RE = re.compile(
    r"""
    ^(?:
        let\s+me\s+(?:think|reconsider|check|see|look|figure|decide|start)\b
      | i\s+need\s+to\s+(?:think|reconsider|check|see|figure)\b
      | (?:now\s+)?let's\s+(?:think|reconsider|check|see|look)\b
      | 让我(?:想想|看看|看一下|先|再想|考虑|复盘|重新考虑)
      | 先(?:看看|看一下|想想)
      | 等一下
      | 我再想(?:想)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Hedges / interjections that may prefix a substantive sentence → trim the
# prefix and keep the content (or drop the line if only filler remains).
_HEDGE_MARKER_RE = re.compile(
    r"""
    ^(?:
        i'?d\s+say\b
      | i\s+would\s+say\b
      | i\s+guess\b
      | i\s+think\b
      | i\s+reckon\b
      | h+m+(?=[,.\s]|$)
      | um+(?=[,.\s]|$)
      | uh+(?=[,.\s]|$)
      | erm(?=[,.\s]|$)
      | wait\b
      | ok(?:ay)?\b
      | so\s*(?:,|$)
      | well\s*(?:,|$)
      | actually\b
      | anyway\b
      | alright\b
      | 嗯+
      | 唔+
      | 呃+
      | 唉
      | 说实话
      | 好吧
      | 总之
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Leading whitespace plus markdown list/quote/heading markers. The markers are
# kept separate from the content so a cleaned line can keep its structure.
_MARKDOWN_PREFIX_RE = re.compile(r"^([ \t]*(?:[#>*\-+]\s+)*)(.*)$")

# English particles and CJK interjections that add no report content. A line
# whose only residue after stripping markers is these words is a pure
# interjection and is dropped entirely.
_FILLER_WORDS = frozenset(
    "about this that here first so then now one more again please ok okay fine "
    "alright anyway well yeah right hmm ah".split()
)
_FILLER_CJK = frozenset("嗯啊哦呢吧唉呃")


def clean_thinking_traces(text: str) -> str:
    """Return *text* with AI thinking-monologue lines removed or trimmed."""
    if not isinstance(text, str) or not text:
        return text
    lines = text.split("\n")
    cleaned = [_clean_line(line) for line in lines]
    return "\n".join(_collapse_blank_lines(cleaned))


def clean_report_result_data(result_data):
    """Return a cleaned copy of a report ``result_data`` dict.

    Cleans every top-level report section as well as the per-horizon nested
    sections (``short_term`` / ``medium_term`` / ``horizons``). Non-text
    fields are passed through unchanged.
    """
    if not isinstance(result_data, dict):
        return result_data
    cleaned = dict(result_data)
    for key in _REPORT_TEXT_KEYS:
        value = cleaned.get(key)
        if isinstance(value, str):
            cleaned[key] = clean_thinking_traces(value)
    for hkey in ("short_term", "medium_term"):
        horizon = cleaned.get(hkey)
        if isinstance(horizon, dict):
            cleaned[hkey] = clean_report_result_data(horizon)
    horizons = cleaned.get("horizons")
    if isinstance(horizons, dict):
        cleaned["horizons"] = {
            key: clean_report_result_data(value) if isinstance(value, dict) else value
            for key, value in horizons.items()
        }
    return cleaned


# Report sections that may carry free-form model prose.
_REPORT_TEXT_KEYS = (
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "macro_report",
    "smart_money_report",
    "volume_price_report",
    "game_theory_report",
    "investment_plan",
    "trader_investment_plan",
    "final_trade_decision",
)


def _clean_line(line: str) -> str:
    m = _MARKDOWN_PREFIX_RE.match(line)
    if not m:
        return line
    md_prefix, content = m.group(1), m.group(2)
    if _PROCESS_NARRATION_RE.match(content):
        return ""  # "let me think / 让我想想" narration → drop the line
    if not _HEDGE_MARKER_RE.match(content):
        return line
    rest = _strip_hedges(content)
    if _PROCESS_NARRATION_RE.match(rest):
        return ""  # "wait, I need to reconsider ..." → still narration
    if _is_filler_only(rest):
        return ""  # pure interjection ("Hmm", "wait,", "OK.")
    return md_prefix + rest


def _strip_hedges(content: str) -> str:
    rest = content
    while True:
        m = _HEDGE_MARKER_RE.match(rest)
        if not m:
            break
        rest = rest[m.end():].lstrip(" ,，:：;；")
    return rest.strip()


def _is_filler_only(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+|[一-鿿]", text)
    if not tokens:
        return True
    return all(t.lower() in _FILLER_WORDS or t in _FILLER_CJK for t in tokens)


def _collapse_blank_lines(lines) -> list[str]:
    result: list[str] = []
    prev_blank = False
    for line in lines:
        if not line.strip():
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False
    while result and not result[0].strip():
        result.pop(0)
    while result and not result[-1].strip():
        result.pop()
    return result
