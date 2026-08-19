"""Unit tests for DAV-190 T10: Deep reasoning prompt enhancements across adjudication and risk management layers.

Verifies deterministic semantic requirements, macro & supply chain penetration,
global asset volatility linkage (gold/treasuries/crude oil/FX), dynamic risk control,
VERDICT and RISK_STATE / RISK_JUDGE contracts, output discipline, and regression safety.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tradingagents.prompts import get_prompt
from tradingagents.prompts.zh import PROMPTS as ZH_PROMPTS
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.managers.risk_manager import create_risk_manager
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.utils.debate_utils import extract_risk_judge_result, strip_tagged_json


ADJUDICATION_AND_RISK_KEYS = [
    "research_manager_prompt",
    "trader_system_prompt",
    "risk_manager_prompt",
    "aggressive_prompt",
    "conservative_prompt",
    "neutral_prompt",
]


def test_no_parallel_v2_prompt_keys():
    """Ensure no _v2, _new, or parallel keys were introduced in zh.py."""
    forbidden_suffixes = ("_v2", "_new", "_fixed", "_enhanced", "_deep")
    for key in ZH_PROMPTS.keys():
        for suffix in forbidden_suffixes:
            assert not key.endswith(suffix), f"Parallel key found: {key}"


def test_research_manager_prompt_deep_framework():
    """T10: research_manager_prompt must include macro/supply chain cross-penetration,

    vulnerability puncture, expectation gap analysis, and dynamic weighting.
    """
    prompt = ZH_PROMPTS["research_manager_prompt"]

    # 1. 深度穿透与脆弱点审查
    assert any(term in prompt for term in ("穿透审查", "脆弱点", "宏观叙事", "逻辑链条"))
    assert any(term in prompt for term in ("产业链", "供需", "议价权", "库存周期"))

    # 2. 预期差与多维共振
    assert "预期差" in prompt
    assert any(term in prompt for term in ("主力资金", "散户情绪", "逆势建仓", "派发"))

    # 3. 动态加权与决策收敛
    assert any(term in prompt for term in ("动态加权", "分析视角", "短线视角", "中线视角"))
    assert any(term in prompt for term in ("入场区间", "止损位", "止盈", "失效条件"))

    # 4. Hold 限制与机会成本
    assert "机会成本" in prompt
    assert "观望的明确验证信号" in prompt or "观望的验证信号" in prompt

    # 5. VERDICT 与输出纪律
    assert '<!-- VERDICT: {{"direction": "看多", "reason": "不超过20字的一句话核心结论"}} -->' in prompt
    assert "【输出纪律】只输出正式报告正文" in prompt


def test_research_manager_five_step_framework_and_output_discipline():
    """DAV-198 (DAV-192-T6): Verify research_manager_prompt covers five-step adjudication framework and output discipline."""
    prompt = ZH_PROMPTS["research_manager_prompt"]

    # 五步深度裁决框架
    # 第一步：证据链完整性审查 - 标注"证据充分/薄弱/缺失"
    assert "第一步：证据链完整性审查" in prompt
    assert any(term in prompt for term in ("证据充分/薄弱/缺失", "证据充分", "证据薄弱", "证据缺失"))

    # 第二步：传导路径验证 - 检查时滞/量化/历史验证是否合理
    assert "第二步：传导路径验证" in prompt
    assert "时滞" in prompt
    assert "量化" in prompt
    assert "历史验证" in prompt

    # 第三步：焦点分歧裁决 - 逐个裁决哪方证据更强、逻辑更严密，明确判定"多头胜/空头胜/势均力敌"
    assert "第三步：焦点分歧裁决" in prompt
    assert "多头胜/空头胜/势均力敌" in prompt

    # 第四步：极端情景测试 - 推演多空双方的极端情景是否合理
    assert "第四步：极端情景测试" in prompt
    assert "极端情景" in prompt

    # 第五步：风险收益比量化 - 综合给出上涨空间/下跌风险/赔率
    assert "第五步：风险收益比量化" in prompt
    assert "上涨空间" in prompt
    assert any(term in prompt for term in ("下跌风险", "回撤风险"))
    assert "赔率" in prompt

    # 输出纪律
    # 1. 总字数800-1200字
    assert any(term in prompt for term in ("800-1200", "800-1200字"))
    # 2. 必须有明确的"多头胜/空头胜/势均力敌"结论
    assert "多头胜/空头胜/势均力敌" in prompt
    # 3. 必须给出"建议仓位"（0-100%）和"止损位"（具体价格/跌幅）
    assert "建议仓位" in prompt
    assert "0-100%" in prompt
    assert "止损位" in prompt
    assert "具体价格/跌幅" in prompt or ("具体价格" in prompt and "跌幅" in prompt)


def test_trader_system_prompt_deep_framework():
    """T10: trader_system_prompt must enforce direction anchoring, global macro calibration,

    supply chain sensitivity, and strict HOLD boundary conditions.
    """
    prompt = ZH_PROMPTS["trader_system_prompt"]

    # 1. 宏观联动与动态风控
    assert any(term in prompt for term in ("全球宏观", "大类资产", "美债", "美元", "原油", "黄金", "汇率"))
    assert any(term in prompt for term in ("产业链", "成本冲击", "关键技术位"))

    # 2. 方向锚定与执行铁律
    assert "研究经理的结论一致" in prompt
    assert any(term in prompt for term in ("新建仓位", "顺势加仓", "分批减仓", "清仓止损"))

    # 3. 严格限制 HOLD
    assert "观望（HOLD）限制条件" in prompt or "HOLD 不是默认选项" in prompt
    assert "技术面无明确趋势" in prompt
    assert "资金面无明确方向" in prompt

    # 4. VERDICT 与输出纪律
    assert '<!-- VERDICT: {{"direction": "看多", "reason": "不超过20字的一句话核心结论"}} -->' in prompt
    assert "【输出纪律】只输出正式报告正文" in prompt
    assert "最终交易建议：买入 / 卖出 / 观望" in prompt


def test_aggressive_debator_prompt_framework():
    """T10: aggressive_prompt must advocate macro tailwinds, supply chain expansion,

    and dynamic risk control (trailing stop, position scaling).
    """
    prompt = ZH_PROMPTS["aggressive_prompt"]

    # 1. 宏观顺风与产业链景气扩张
    assert any(term in prompt for term in ("全球宏观", "产业政策", "产业链", "新质生产力", "景气度"))
    assert any(term in prompt for term in ("收益弹性", "主升浪", "赔率", "胜率"))

    # 2. 动态风控替代机械防守
    assert any(term in prompt for term in ("仓位梯度", "移动止损", "Trailing Stop", "分批止盈"))
    assert "过度保守" in prompt or "静态风险厌恶" in prompt

    # 3. RISK_STATE 契约
    assert "<!-- RISK_STATE:" in prompt
    assert "responded_claim_ids" in prompt
    assert "new_claims" in prompt


def test_conservative_debator_prompt_framework():
    """T10: conservative_prompt must stress-test macro vulnerabilities (bonds/gold/oil/FX),

    supply chain margin squeeze, and tail risk protection.
    """
    prompt = ZH_PROMPTS["conservative_prompt"]

    # 1. 宏观脆弱点与全球大类资产波动
    assert any(term in prompt for term in ("全球大类资产", "美债", "美元", "原油", "黄金", "大盘流动性"))
    assert any(term in prompt for term in ("极端尾部风险", "脆弱点", "黑天鹅", "流动性踩踏"))

    # 2. 产业链断供与上下游挤压
    assert any(term in prompt for term in ("原材料涨价", "毛利坍塌", "产业链", "去库压力", "集中度"))

    # 3. 防守型风控与反驳
    assert any(term in prompt for term in ("压降仓位", "右侧确认", "止损线", "撤退预案"))

    # 4. RISK_STATE 契约
    assert "<!-- RISK_STATE:" in prompt
    assert "responded_claim_ids" in prompt
    assert "new_claims" in prompt


def test_neutral_debator_prompt_framework():
    """T10: neutral_prompt must discern information increments, optimize risk-reward ratio,

    and build adaptive risk budget switching.
    """
    prompt = ZH_PROMPTS["neutral_prompt"]

    # 1. 甄别信息增量与平衡宏观波动
    assert any(term in prompt for term in ("信息增量", "大类资产", "美债", "汇率", "原油", "黄金"))
    assert any(term in prompt for term in ("风险收益比", "Risk-Reward", "盈亏比"))

    # 2. 自适应风控折中方案
    assert any(term in prompt for term in ("自适应切换", "自适应", "仓位预算", "触发条件"))

    # 3. RISK_STATE 契约
    assert "<!-- RISK_STATE:" in prompt
    assert "responded_claim_ids" in prompt
    assert "new_claims" in prompt


def test_risk_manager_prompt_framework():
    """T10: risk_manager_prompt must enforce multi-layer constraints (hard/soft/preconditions/de-risk triggers)

    and macro/supply chain pressure testing.
    """
    prompt = ZH_PROMPTS["risk_manager_prompt"]

    # 1. 宏观与产业链压力测试
    assert any(term in prompt for term in ("全球大类资产", "美债", "汇率", "原油", "黄金", "产业链"))
    assert any(term in prompt for term in ("穿透审查", "动态风控", "压力测试"))

    # 2. 四层风控体系
    assert "hard_constraints" in prompt or "硬约束" in prompt
    assert "soft_constraints" in prompt or "软约束" in prompt
    assert "execution_preconditions" in prompt or "允许执行的前提条件" in prompt
    assert "de_risk_triggers" in prompt or "立即降风险的触发条件" in prompt

    # 3. 契约机读块
    assert "<!-- RISK_JUDGE:" in prompt
    assert "verdict 只可填：pass / revise / reject" in prompt
    assert '<!-- VERDICT: {{"direction": "看多", "reason": "不超过20字的一句话核心结论"}} -->' in prompt


def test_node_execution_smoke():
    """Verify all 6 enhanced node factories initialize and format prompts without KeyError."""
    from tradingagents.agents.utils.debate_utils import build_empty_risk_debate_state
    llm = MagicMock()
    async def fake_astream(prompt, **kwargs):
        if isinstance(prompt, list):
            yield SimpleNamespace(content="最终交易建议：买入\n<!-- VERDICT: {\"direction\": \"看多\", \"reason\": \"趋势确认\"} -->")
        else:
            yield SimpleNamespace(content="测试输出正文\n<!-- VERDICT: {\"direction\": \"看多\", \"reason\": \"趋势确认\"} -->\n<!-- RISK_JUDGE: {\"verdict\": \"pass\", \"revision_reason\": \"\", \"hard_constraints\": [], \"soft_constraints\": [], \"execution_preconditions\": [], \"de_risk_triggers\": []} -->")

    llm.astream = fake_astream
    memory = MagicMock()
    memory.get_memories = MagicMock(return_value=[])

    # 1. Research Manager
    rm_node = create_research_manager(llm, memory)
    rm_state = {
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "smart_money_report": "smart_money",
        "volume_price_report": "volume_price",
        "investment_debate_state": {"count": 1},
        "fund_flow_consensus_guard": {"blocked": False, "direction_allowed": True},
    }
    rm_res = asyncio.run(rm_node(rm_state))
    assert "investment_plan" in rm_res

    # 2. Trader
    trader_node = create_trader(llm, memory)
    trader_state = {
        "company_of_interest": "600519",
        "investment_plan": "plan",
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "fund_flow_consensus_guard": {"blocked": False, "direction_allowed": True},
    }
    trader_res = asyncio.run(trader_node(trader_state))
    assert "trader_investment_plan" in trader_res

    # 3. Aggressive Debator
    agg_node = create_aggressive_debator(llm)
    agg_state = {
        "risk_debate_state": build_empty_risk_debate_state(),
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "trader_investment_plan": "trader plan",
    }
    agg_res = asyncio.run(agg_node(agg_state))
    assert "risk_debate_state" in agg_res

    # 4. Conservative Debator
    cons_node = create_conservative_debator(llm)
    cons_res = asyncio.run(cons_node(agg_state))
    assert "risk_debate_state" in cons_res

    # 5. Neutral Debator
    neu_node = create_neutral_debator(llm)
    neu_res = asyncio.run(neu_node(agg_state))
    assert "risk_debate_state" in neu_res

    # 6. Risk Manager
    risk_node = create_risk_manager(llm, memory)
    risk_state = {
        "company_of_interest": "600519",
        "risk_debate_state": build_empty_risk_debate_state(),
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "trader_investment_plan": "trader plan",
        "fund_flow_consensus_guard": {"blocked": False, "direction_allowed": True},
    }
    risk_res = asyncio.run(risk_node(risk_state))
    assert "final_trade_decision" in risk_res
