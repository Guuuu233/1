"""Unit tests for DAV-190 T9: Deep reasoning & 1:1 mirror-symmetric debate prompt enhancements.

Verifies deterministic semantic requirements, deep reasoning frameworks
(Top-down macro confirmation/headwind, supply chain upstream/downstream game,
international peer benchmarking, extreme scenarios attack/defense),
1:1 mirror symmetry, DEBATE_STATE contracts, and researcher node execution.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from tradingagents.prompts.zh import PROMPTS as ZH_PROMPTS
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher


DEBATE_PROMPT_KEYS = ["bull_prompt", "bear_prompt"]


def test_no_parallel_v2_debate_prompt_keys():
    """Ensure no _v2, _new, or parallel keys were introduced in zh.py for debate prompts."""
    forbidden_suffixes = ("_v2", "_new", "_fixed", "_enhanced", "_deep")
    for key in ZH_PROMPTS.keys():
        for suffix in forbidden_suffixes:
            assert not key.endswith(suffix), f"Parallel key found: {key}"


@pytest.mark.parametrize("key", DEBATE_PROMPT_KEYS)
def test_debate_state_machine_contract_is_intact(key):
    """Every debate prompt must preserve the exact DEBATE_STATE contract at the end."""
    prompt = ZH_PROMPTS[key]

    # Check DEBATE_STATE comment structure
    assert '<!-- DEBATE_STATE: {{"responded_claim_ids": ["INV-1"], "new_claims": [{{"claim": "不超过28字", "evidence": ["证据1", "证据2"], "confidence": 0.72}}], "resolved_claim_ids": ["INV-2"], "unresolved_claim_ids": ["INV-3"], "next_focus_claim_ids": ["INV-3"], "round_summary": "不超过50字", "round_goal": "不超过30字"}} -->' in prompt
    assert "若没有对应项，返回空数组。" in prompt
    assert "口径约束（不新增正文级 canonical 字段）：" in prompt
    assert "DEBATE_STATE 中每个 new_claims[].confidence 是 claim confidence，只能是有限的 0.00–1.00 数值，不得使用百分比。" in prompt
    assert "Bear 与 Bull 共同遵守同一 probability 口径：Bear 的 probability 不得改成下跌概率，不得反转，不做 1-p。" in prompt


def test_bull_prompt_deep_framework():
    """T9: bull_prompt must include top-down macro confirmation, supply chain bargaining power,

    international peer benchmarking, and extreme scenario anti-fragility.
    """
    prompt = ZH_PROMPTS["bull_prompt"]

    # 1. 自上而下宏观印证与共振催化
    assert "自上而下" in prompt
    assert any(term in prompt for term in ("全球宏观", "流动性", "央行政策", "大宗商品", "产业政策", "共振"))

    # 2. 产业链上下游博弈与议价权
    assert any(term in prompt for term in ("产业链", "议价权", "定价权", "竞争壁垒", "毛利率"))

    # 3. 国际同业对标与估值溢价
    assert any(term in prompt for term in ("国际同业对标", "同业龙头", "海外同业", "估值溢价", "估值重估"))

    # 4. 极端情景攻防与反脆弱底线
    assert any(term in prompt for term in ("极端情景", "反脆弱", "底线支撑"))
    assert any(term in prompt for term in ("AI泡沫破裂", "地缘断供", "宏观滞胀"))

    # 5. 焦点 claim 与硬证据
    assert "焦点 claim" in prompt
    assert any(term in prompt for term in ("硬证据", "价格", "成交量", "资金流", "财务报表"))

    # 6. 风险收益比与赔率量化
    assert any(term in prompt for term in ("风险收益比", "赔率", "上涨目标", "回撤风险"))

    # 7. 市场情绪预期差识别
    assert any(term in prompt for term in ("情绪预期差", "过度悲观", "反转机会"))

    # 8. 失败条件与失效纠错机制
    assert any(term in prompt for term in ("失败条件", "失效纠错", "边界条件"))


def test_bear_prompt_deep_framework():
    """T9: bear_prompt must include top-down macro headwind, supply chain vulnerability,

    international peer ceiling, and extreme scenario downside stress.
    """
    prompt = ZH_PROMPTS["bear_prompt"]

    # 1. 自上而下宏观逆风与估值挤压
    assert "自上而下" in prompt
    assert any(term in prompt for term in ("全球宏观", "流动性收紧", "地缘风险", "大宗商品成本", "宏观逆风", "估值挤压"))

    # 2. 产业链上下游博弈与脆弱点穿透
    assert any(term in prompt for term in ("产业链", "脆弱点", "议价劣势", "毛利坍塌", "价格战"))

    # 3. 国际同业对标与估值天花板
    assert any(term in prompt for term in ("国际同业对标", "同业龙头", "估值天花板", "下行映射"))

    # 4. 极端情景攻防与下行压力推演
    assert any(term in prompt for term in ("极端情景", "下行压力", "脆弱性放大", "估值杀跌"))
    assert any(term in prompt for term in ("AI泡沫破裂", "地缘断供", "宏观滞胀"))

    # 5. 焦点 claim 与硬证据
    assert "焦点 claim" in prompt
    assert any(term in prompt for term in ("硬证据", "价格破位", "量价背离", "资金流出", "财务瑕疵"))

    # 6. 风险收益比与赔率量化
    assert any(term in prompt for term in ("风险收益比", "赔率", "回撤空间", "反弹阻力"))

    # 7. 市场情绪预期差识别
    assert any(term in prompt for term in ("情绪预期差", "极度乐观", "狂热追高", "见顶回落"))

    # 8. 失败条件与失效纠错机制
    assert any(term in prompt for term in ("失败条件", "失效纠错", "边界条件"))


def test_debate_prompts_strict_mirror_symmetry():
    """T9: bull_prompt and bear_prompt must be 1:1 mirror symmetric across all dimensions."""
    bull = ZH_PROMPTS["bull_prompt"]
    bear = ZH_PROMPTS["bear_prompt"]

    # Required placeholder symmetry
    common_placeholders = [
        "{custom_prompt_before_data}",
        "{market_research_report}",
        "{sentiment_report}",
        "{news_report}",
        "{fundamentals_report}",
        "{volume_price_report}",
        "{history}",
        "{current_response}",
        "{claims_text}",
        "{focus_claims_text}",
        "{unresolved_claims_text}",
        "{round_summary}",
        "{round_goal}",
        "{past_memory_str}",
        "{custom_prompt_after_data}",
    ]
    for ph in common_placeholders:
        assert ph in bull, f"Placeholder {ph} missing in bull_prompt"
        assert ph in bear, f"Placeholder {ph} missing in bear_prompt"

    # Both must have 9 writing requirements numbered 1 to 9
    for i in range(1, 10):
        assert f"{i}. " in bull, f"bull_prompt missing requirement {i}"
        assert f"{i}. " in bear, f"bear_prompt missing requirement {i}"

    # Verify 1:1 thematic alignment
    # Dimension 1: Macro
    assert "宏观" in bull and "宏观" in bear
    # Dimension 2: Supply chain
    assert "产业链" in bull and "产业链" in bear
    # Dimension 3: International peers
    assert "国际同业对标" in bull and "国际同业对标" in bear
    # Dimension 4: Extreme scenarios
    for scenario in ("AI泡沫破裂", "地缘断供", "宏观滞胀"):
        assert scenario in bull, f"Scenario {scenario} missing in bull_prompt"
        assert scenario in bear, f"Scenario {scenario} missing in bear_prompt"
    # Dimension 5: Focus claims & evidence
    assert "焦点 claim" in bull and "焦点 claim" in bear
    # Dimension 6: Risk-reward
    assert "风险收益比" in bull and "风险收益比" in bear
    # Dimension 7: Sentiment expectation gap
    assert "情绪预期差" in bull and "情绪预期差" in bear
    # Dimension 8: Invalidation / failure boundary
    assert "失效纠错机制" in bull and "失效纠错机制" in bear
    # Dimension 9: Machine readable DEBATE_STATE block
    assert "DEBATE_STATE" in bull and "DEBATE_STATE" in bear


def test_three_round_progressive_framework_symmetry():
    """DAV-197: Both bull_prompt and bear_prompt must strictly define the 3-round progressive framework."""
    for key in DEBATE_PROMPT_KEYS:
        prompt = ZH_PROMPTS[key]
        assert "【辩论三轮递进推进框架】" in prompt, f"3-round header missing in {key}"
        assert "第1轮（开场立论）" in prompt, f"Round 1 missing in {key}"
        assert "第2轮（攻防回应）" in prompt, f"Round 2 missing in {key}"
        assert "第3轮（收官深化）" in prompt, f"Round 3 missing in {key}"
        assert "responded_claim_ids不得为空" in prompt, f"responded_claim_ids constraint missing in {key}"
        assert "极端情景推演" in prompt, f"Extreme scenario simulation missing in {key}"
        assert "0.00-1.00" in prompt, f"Confidence range missing in {key}"
        assert "量化风险收益比" in prompt, f"Risk-reward quantification missing in {key}"


def test_evidence_citation_and_conflict_resolution_constraints():
    """DAV-197: Both bull_prompt and bear_prompt must strictly enforce citation format and conflict weighting."""
    for key in DEBATE_PROMPT_KEYS:
        prompt = ZH_PROMPTS[key]
        assert "【论证与引用纪律】" in prompt, f"Citation header missing in {key}"
        assert '根据XX分析师报告，YY数据显示ZZ' in prompt, f"Citation template missing in {key}"
        assert "禁止机械重复分析师已陈述的表面内容" in prompt, f"Anti-repetition rule missing in {key}"
        assert "二阶深度推理" in prompt, f"Second-order reasoning rule missing in {key}"
        assert "当多个分析师结论矛盾时" in prompt, f"Analyst conflict rule missing in {key}"
        assert "明确说明如何权衡取舍并给出权重依据" in prompt, f"Weight explanation rule missing in {key}"


def test_researcher_nodes_end_to_end_execution():
    """Verify that create_bull_researcher and create_bear_researcher execute cleanly with new prompts."""
    debate_payload = '<!-- DEBATE_STATE: {"responded_claim_ids": ["INV-1"], "new_claims": [{"claim": "宏观与产业链博弈论证", "evidence": ["宏观流动性宽松", "上下游定价权稳固"], "confidence": 0.85}], "resolved_claim_ids": [], "unresolved_claim_ids": [], "next_focus_claim_ids": [], "round_summary": "完成宏观与产业链论证", "round_goal": "下轮验证极端情景"} -->'
    bull_response = f"【多头深度研报】\n基于自上而下宏观印证与产业链博弈分析。\n{debate_payload}"
    bear_response = f"【空头深度研报】\n基于自上而下宏观逆风与产业链脆弱点分析。\n{debate_payload}"

    state = {
        "market_report": "市场技术面趋势向好，RSI 55",
        "sentiment_report": "情绪面温和，分歧显现",
        "news_report": "新闻面行业政策持续加码支持",
        "fundamentals_report": "基本面营收+20%，毛利率稳健",
        "volume_price_report": "量价温和放量突破",
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_speaker": "",
            "current_response": "上轮观点",
            "count": 0,
            "claims": [{"claim_id": "INV-1", "claim": "旧 claim", "status": "open"}],
            "focus_claim_ids": ["INV-1"],
            "open_claim_ids": ["INV-1"],
            "resolved_claim_ids": [],
            "unresolved_claim_ids": ["INV-1"],
            "round_summary": "首轮辩论",
            "round_goal": "展开宏观与产业链博弈",
            "claim_counter": 1,
            "judge_decision": "",
        },
        "horizon": "medium",
        "user_intent": None,
    }

    mock_llm_bull = MagicMock()
    mock_llm_bull.astream = MagicMock(side_effect=lambda prompt: _fake_stream(bull_response))
    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    bull_node = create_bull_researcher(mock_llm_bull, memory, custom_prompt="关注海外对标", placement="after_data")
    bull_result = asyncio.run(bull_node(state))

    assert "investment_debate_state" in bull_result
    new_debate = bull_result["investment_debate_state"]
    assert len(new_debate["claims"]) >= 2
    assert new_debate["round_summary"] == "完成宏观与产业链论证"

    mock_llm_bear = MagicMock()
    mock_llm_bear.astream = MagicMock(side_effect=lambda prompt: _fake_stream(bear_response))
    bear_node = create_bear_researcher(mock_llm_bear, memory, custom_prompt="关注地缘断供", placement="after_data")
    bear_result = asyncio.run(bear_node(state))

    assert "investment_debate_state" in bear_result
    new_bear_debate = bear_result["investment_debate_state"]
    assert len(new_bear_debate["claims"]) >= 2


async def _fake_stream(text: str):
    from types import SimpleNamespace
    yield SimpleNamespace(content=text)
