"""Regression coverage for report confidence extraction (Bug C, DAV-88).

Hermes root cause in api/services/report_service.py:
1. ``_extract_confidence_regex`` only matched "置信度：55%" percent format; the
   LLM actually emits "置信度：62/75" (x/75 cap format) → regex missed it.
2. ``resolve_report_fields`` read confidence only from ``final_trade_decision``;
   the LLM often writes it in ``trader_investment_plan`` (600206.SH proof) → None.
   target_price / stop_loss already fall back to the plan; confidence did not.
3. The LLM structured-extraction prompt never explained the x/75 format.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from api.services import report_service


# ── _extract_confidence_regex unit tests ─────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("置信度：62/75", 62),  # x/75 format, full-width colon
        ("置信度:62/75", 62),  # x/75 format, half-width colon
        ("置信度：62／75", 62),  # x/75 format, full-width slash
        ("置信度：55%", 55),  # percent format (existing behavior)
        ("confidence: 60/75", 60),  # English x/75
        ("confidence: 60/100", 60),  # English x/100
        ("confidence: 55%", 55),  # English percent (existing behavior)
        ("置信度：75/75", 75),
        ("置信度：0/75", 0),
    ],
)
def test_extract_confidence_regex_handles_x_over_75_and_percent(text, expected):
    assert report_service._extract_confidence_regex(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "今天天气不错",
        "置信度：abc",
        "置信度：-1",
        "confidence: none",
    ],
)
def test_extract_confidence_regex_returns_none_for_no_confidence(text):
    assert report_service._extract_confidence_regex(text) is None


# ── resolve_report_fields fallback to trader_investment_plan ─────────────────


def test_confidence_extracted_from_plan_when_decision_has_none():
    """600206.SH repro: confidence lives in trader_investment_plan, not the decision."""
    result = report_service.resolve_report_fields(
        result_data={
            "final_trade_decision": "建议持仓观望，短线偏谨慎。",
            "trader_investment_plan": "置信度：62/75。计划分批建仓。",
        }
    )
    assert result["confidence"] == 62


def test_confidence_english_x_over_75_from_plan():
    result = report_service.resolve_report_fields(
        result_data={
            "final_trade_decision": "Hold with caution.",
            "trader_investment_plan": "confidence: 60/75",
        }
    )
    assert result["confidence"] == 60


def test_confidence_decision_wins_over_plan():
    """When both sources carry a confidence, final_trade_decision takes priority."""
    result = report_service.resolve_report_fields(
        result_data={
            "final_trade_decision": "置信度：55%。",
            "trader_investment_plan": "置信度：62/75。",
        }
    )
    assert result["confidence"] == 55


def test_no_confidence_anywhere_returns_none():
    result = report_service.resolve_report_fields(
        result_data={
            "final_trade_decision": "建议持仓观望。",
            "trader_investment_plan": "分批建仓。",
        }
    )
    assert result["confidence"] is None


def test_confidence_override_still_wins():
    result = report_service.resolve_report_fields(
        result_data={
            "final_trade_decision": "置信度：62/75。",
        },
        confidence_override=88,
    )
    assert result["confidence"] == 88


# ── LLM structured-extraction prompt documents the x/75 format ───────────────


class _RecordingInvokeLLM:
    def __init__(self):
        self.captured_prompt = None

    def invoke(self, messages):
        self.captured_prompt = messages[0].content
        return SimpleNamespace(content='{"decision": "HOLD", "confidence": 62}')


def test_structured_extraction_prompt_documents_x_over_75_format():
    llm = _RecordingInvokeLLM()
    with patch(
        "tradingagents.llm_clients.create_llm_client",
        return_value=SimpleNamespace(get_llm=lambda: llm),
    ):
        structured = report_service.extract_structured_data(
            final_trade_decision="持仓观望。",
            config={"llm_provider": "fake", "quick_think_llm": "fake"},
        )
    assert structured is not None
    assert llm.captured_prompt is not None
    # The confidence instruction must document the x/75 upper-bound format and
    # that the numerator is taken (62 from "置信度：62/75").
    assert "x/75" in llm.captured_prompt or "上限格式" in llm.captured_prompt
