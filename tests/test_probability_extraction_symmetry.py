"""Regression coverage for probability extraction symmetry (DAV-759, ref DAV-755).

Verifies that resolve_report_fields scans final_trade_decision and
trader_investment_plan for probability symmetrically with confidence,
recovering valid probabilities previously dropped silently.
"""

import copy
import pytest

from api.services import report_service


# ── Real production fixtures from DAV-755 ──────────────────────────────────────

SAMPLE_601766_FTD = (
    "- **盲区铁律**：**6.17–6.34 区间绝对禁入**。严禁挂条件单，触价自动撤单。"
    "保守派提供的“突破回踩胜率38%、期望值为负”的统计证据充分，不予条件准入。"
)

SAMPLE_603019_FTD = (
    "2. **置信度与概率检查**：交易员给出的 Confidence 为 65（严格低于75的硬上限），"
    "Probability 为 0.40。在6项关键数据缺失的背景下，该置信度定价客观，未出现过度自信的偏差。"
)
SAMPLE_603019_TIP = (
    "- Probability: 0.40 for price > 88.65 ✓\n"
    "## 九、概率与置信度\n"
    "- **Probability（周期结束时 > 88.65 的上涨概率）**：**0.40**"
)

SAMPLE_159763_FTD = (
    "**调整理由**：置信度65%、上涨概率0.55、盈亏比仅约2:1（保守方计算），"
    "5%仓位过于激进；4%上限在保留趋势参与权的同时，将最大损失（含跳空尾部）控制在组合0.15%以内。"
)
SAMPLE_159763_TIP = "- **上涨概率**：0.55"


# ── 1. DAV-755 Real sample regression tests ───────────────────────────────────


def test_probability_extracted_from_final_trade_decision_601766():
    """601766.SH repro: win rate 38% resides in final_trade_decision."""
    result_data = {
        "final_trade_decision": SAMPLE_601766_FTD,
        "trader_investment_plan": "计划分批建仓。",
    }
    resolved = report_service.resolve_report_fields(result_data)
    assert resolved["probability"] == pytest.approx(0.38)


def test_probability_extracted_from_trader_investment_plan_603019():
    """603019.SH repro: Probability: 0.40 resides in trader_investment_plan."""
    result_data = {
        "final_trade_decision": SAMPLE_603019_FTD,
        "trader_investment_plan": SAMPLE_603019_TIP,
    }
    resolved = report_service.resolve_report_fields(result_data)
    assert resolved["probability"] == pytest.approx(0.40)


def test_probability_extracted_from_final_trade_decision_159763():
    """159763.SZ repro: 上涨概率0.55 resides in final_trade_decision and plan."""
    result_data = {
        "final_trade_decision": SAMPLE_159763_FTD,
        "trader_investment_plan": SAMPLE_159763_TIP,
    }
    resolved = report_service.resolve_report_fields(result_data)
    assert resolved["probability"] == pytest.approx(0.55)


# ── 2. Fallback hierarchy and symmetry tests ───────────────────────────────────


def test_probability_decision_wins_over_plan():
    """Symmetry with test_confidence_decision_wins_over_plan:
    When both final_trade_decision and trader_investment_plan carry probability,
    final_trade_decision takes priority.
    """
    result_data = {
        "final_trade_decision": "综合评估短线上涨概率为 0.60。",
        "trader_investment_plan": "交易计划预估上涨概率为 0.50。",
    }
    resolved = report_service.resolve_report_fields(result_data)
    assert resolved["probability"] == pytest.approx(0.60)


def test_probability_plan_wins_over_judge_decision():
    """When final_trade_decision lacks probability, trader_investment_plan
    takes priority over upstream judge_decision.
    """
    result_data = {
        "final_trade_decision": "建议逢低买入，未特别指定概率。",
        "trader_investment_plan": "交易计划中给出的上涨概率为 0.55。",
        "investment_debate_state": {
            "judge_decision": "辩论阶段预估短线上涨概率为 0.45。"
        },
    }
    resolved = report_service.resolve_report_fields(result_data)
    assert resolved["probability"] == pytest.approx(0.55)


def test_probability_fallback_to_judge_decision_when_decision_and_plan_lack_it():
    """When neither final_trade_decision nor trader_investment_plan has probability,
    fallback to judge_decision preserved.
    """
    result_data = {
        "final_trade_decision": "建议维持中性观望。",
        "trader_investment_plan": "按兵不动。",
        "investment_debate_state": {
            "judge_decision": "双方辩论，裁决上涨概率为 0.45。"
        },
    }
    resolved = report_service.resolve_report_fields(result_data)
    assert resolved["probability"] == pytest.approx(0.45)


def test_probability_explicit_result_data_takes_precedence():
    """Explicit / structured result_data['probability'] has highest priority."""
    result_data = {
        "probability": 0.70,
        "final_trade_decision": "上涨概率为 0.60。",
        "trader_investment_plan": "上涨概率为 0.50。",
    }
    resolved = report_service.resolve_report_fields(result_data)
    assert resolved["probability"] == pytest.approx(0.70)


def test_probability_none_when_all_sources_lack_it():
    """When no source provides probability, result is None and extraction_note marks it."""
    result_data = {
        "final_trade_decision": "维持观望。置信度：60/75。",
        "trader_investment_plan": "暂无计划。",
    }
    resolved = report_service.resolve_report_fields(result_data)
    assert resolved["probability"] is None
    assert resolved["extraction_note"] is not None
    assert "概率未提供/未提取" in resolved["extraction_note"]


def test_resolve_report_fields_does_not_mutate_input():
    """Contract 4: resolve_report_fields must not mutate input result_data."""
    input_data = {
        "final_trade_decision": "上涨概率为 0.55。",
        "trader_investment_plan": "交易计划。",
        "nested": {"key": "value"},
    }
    snapshot = copy.deepcopy(input_data)
    resolved = report_service.resolve_report_fields(input_data)
    assert input_data == snapshot
    assert resolved["probability"] == pytest.approx(0.55)
