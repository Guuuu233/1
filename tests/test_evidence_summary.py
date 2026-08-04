"""Unit tests for the deterministic analyst-report evidence summary extractor.

DAV-68 M2: adjudicators must be able to anchor verdicts on evidence strength.
``build_evidence_summary`` produces the bounded, fact-dense first-hand excerpts
that research_manager receives instead of full analyst reports.
"""
from __future__ import annotations

from tradingagents.agents.utils.evidence_summary import (
    DEFAULT_MAX_CHARS,
    build_evidence_summary,
    extract_verdict_direction,
    strip_machine_blocks,
)


def test_empty_report_returns_empty_string():
    # Empty input must produce an empty summary (not a placeholder), so callers
    # can conditionally omit the evidence line instead of injecting a
    # misleading "no content" marker for an analyst that did not run.
    assert build_evidence_summary("") == ""
    assert build_evidence_summary(None) == ""
    assert build_evidence_summary("   \n  ") == ""


def test_direction_extracted_from_verdict_block():
    report = (
        "一些叙述。\n"
        '<!-- VERDICT: {"direction": "偏空", "reason": "资金流出"} -->'
    )
    assert extract_verdict_direction(report) == "偏空"


def test_direction_missing_when_no_verdict():
    assert extract_verdict_direction("没有机读块") == ""


def test_strip_machine_blocks_removes_protocol_blocks():
    report = (
        "正文内容。\n"
        '<!-- VERDICT: {"direction": "看多", "reason": "x"} -->\n'
        '<!-- DEBATE_STATE: {"new_claims": []} -->\n'
    )
    cleaned = strip_machine_blocks(report)
    assert "VERDICT" not in cleaned
    assert "DEBATE_STATE" not in cleaned
    assert "正文内容" in cleaned


def test_summary_keeps_numbers_and_drops_boilerplate():
    report = (
        "# 分析报告标题\n"
        "本文基于公开数据对公司进行基本面分析。\n"
        "净利润同比增长 15%，毛利率 45%。\n"
        "经营现金流为正，资产负债率 38%。\n"
        '<!-- VERDICT: {"direction": "中性", "reason": "估值合理"} -->'
    )
    summary = build_evidence_summary(report)
    assert "净利润同比增长 15%" in summary
    assert "毛利率 45%" in summary
    assert "资产负债率 38%" in summary
    # Boilerplate prose without numbers is not evidence.
    assert "本文基于公开数据" not in summary
    # Direction is prefixed as a labeled fact.
    assert "[分析师结论：中性]" in summary
    # Machine block is gone.
    assert "VERDICT" not in summary


def test_markdown_table_rows_are_flattened():
    report = (
        "指标 | 当前信号 | 交易含义\n"
        "---|--- |--- \n"
        "RSI | 48.2 | 中性\n"
        "MACD | 金叉 | 偏多\n"
        '<!-- VERDICT: {"direction": "偏多", "reason": "技术向好"} -->'
    )
    summary = build_evidence_summary(report)
    assert "RSI" in summary
    assert "48.2" in summary
    # The header row and separator row are dropped, and non-numeric table rows
    # (pure qualitative labels) are not treated as evidence.
    assert "当前信号" not in summary
    assert "---|---" not in summary
    assert "MACD" not in summary
    assert "[分析师结论：偏多]" in summary


def test_summary_respects_char_cap():
    report = "数字填充。" + ("业绩数据 12% 增长 34% 波动 56% 扩张 78% 回落。") * 20
    summary = build_evidence_summary(report)
    assert len(summary) <= DEFAULT_MAX_CHARS + len("[分析师结论：]")
    assert summary.endswith("…")


def test_fallback_excerpt_for_numberless_report():
    report = (
        "该标的缺乏量化数据。\n"
        "分析师观察到情绪偏暖但无法量化。\n"
        '<!-- VERDICT: {"direction": "看多", "reason": "定性看好"} -->'
    )
    summary = build_evidence_summary(report)
    # Even with no numbers, the adjudicator sees the analyst's core content.
    assert "缺乏量化数据" in summary or "情绪偏暖" in summary
    assert "[分析师结论：看多]" in summary


def test_duplicate_lines_deduped():
    report = (
        "净利润同比增长 15%。\n"
        "净利润同比增长 15%。\n"
        '<!-- VERDICT: {"direction": "中性", "reason": "x"} -->'
    )
    summary = build_evidence_summary(report)
    assert summary.count("净利润同比增长 15%") == 1
