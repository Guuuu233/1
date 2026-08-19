"""Unit tests for DAV-188: Deep reasoning prompt enhancements across analysts.

Verifies deterministic semantic requirements, deep reasoning frameworks
(Top-down macro, supply chain linkage, stress testing, What/Why/SoWhat/WhatNext),
VERDICT contracts, output discipline, and regression safety.
"""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tradingagents.prompts import get_prompt
from tradingagents.prompts.zh import PROMPTS as ZH_PROMPTS
from tradingagents.agents.utils.agent_states import extract_verdict
from tradingagents.agents.analysts.macro_analyst import create_macro_analyst
from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.social_media_analyst import create_social_media_analyst
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.smart_money_analyst import create_smart_money_analyst
from tradingagents.graph.data_collector import DataCollector


ANALYST_PROMPT_KEYS = [
    "macro_system_message",
    "fundamentals_system_message",
    "news_system_message",
    "social_system_message",
    "market_system_message",
    "smart_money_system_message",
]


def test_no_parallel_v2_prompt_keys():
    """Ensure no _v2, _new, or parallel keys were introduced in zh.py."""
    forbidden_suffixes = ("_v2", "_new", "_fixed", "_enhanced", "_deep")
    for key in ZH_PROMPTS.keys():
        for suffix in forbidden_suffixes:
            assert not key.endswith(suffix), f"Parallel key found: {key}"


@pytest.mark.parametrize("key", ANALYST_PROMPT_KEYS)
def test_analyst_verdict_contract_is_intact(key):
    """Every analyst prompt must preserve the exact VERDICT contract at the end."""
    prompt = ZH_PROMPTS[key]

    # Check verdict comment structure
    assert '<!-- VERDICT: {"direction": "看多", "reason": "不超过20字的一句话核心结论"} -->' in prompt
    assert "direction 只可填：看多 / 偏多 / 中性 / 偏空 / 看空" in prompt
    assert "（数据有方向倾向时必须选偏多或偏空，仅数据确实不足时可选中性）" in prompt


@pytest.mark.parametrize("key", ANALYST_PROMPT_KEYS)
def test_analyst_output_discipline_is_present(key):
    """Every analyst prompt must mandate strict output discipline (no thinking leaks)."""
    prompt = ZH_PROMPTS[key]
    assert "【输出纪律】只输出正式报告正文" in prompt
    assert "禁止在报告中写下思考过程、内心独白或推理草稿" in prompt
    assert "所有思考在内部完成，不要写进报告" in prompt


def test_macro_system_message_deep_framework():
    """T2: macro_system_message must include global macro, domestic liquidity,

    industry transmission, sector rotation, and scenario analysis.
    """
    prompt = ZH_PROMPTS["macro_system_message"]

    # 1. Top-down体系与全球宏观
    assert "自上而下" in prompt or "Top-Down" in prompt
    assert any(term in prompt for term in ("全球宏观", "美联储", "美元指数", "美债", "大宗商品", "外溢效应"))

    # 2. 国内宏观与大盘流动性
    assert any(term in prompt for term in ("中国货币政策", "流动性", "降准", "降息", "LPR", "社融", "大盘"))

    # 3. 产业政策与产业链传导
    assert any(term in prompt for term in ("产业政策", "产业链", "新质生产力", "设备更新", "传导路径"))

    # 4. 板块轮动与资金结构
    assert "板块资金流向" in prompt or "行业板块资金" in prompt
    assert any(term in prompt for term in ("板块轮动", "风格", "成长", "价值"))

    # 5. 多维情景推演
    assert any(term in prompt for term in ("情景推演", "Scenario Analysis", "基准情景", "乐观情景", "悲观情景"))


def test_fundamentals_system_message_deep_framework():
    """T3: fundamentals_system_message must include supply chain bargaining power,

    industry cycle, macro sensitivity, and stress testing.
    """
    prompt = ZH_PROMPTS["fundamentals_system_message"]

    # 1. 产业链议价权与竞争壁垒（A. 上游议价权与成本传导、B. 下游需求与订单质量）
    assert any(term in prompt for term in ("产业链", "议价权", "供应商", "客户", "定价权"))
    assert "上游议价权与成本传导" in prompt or "上游议价权" in prompt
    assert "下游需求与订单质量" in prompt or "下游需求" in prompt
    assert "国际对标" in prompt

    # 2. 行业周期与供需定位
    assert any(term in prompt for term in ("产能周期", "库存周期", "供需", "Capex", "资本开支"))

    # 3. 财务穿透与现金流
    assert any(term in prompt for term in ("盈利质量", "现金流", "自由现金流", "FCF", "资产负债"))

    # 4. 宏观敏感度与极端情景压力测试
    assert any(term in prompt for term in ("敏感度", "敏感性", "极端情景", "压力测试", "Stress Test"))

    # 5. 输出强制要求（数据缺失、对比、议价权定性、敏感性）
    assert "【输出强制要求】" in prompt or "输出强制要求" in prompt
    assert "数据缺失" in prompt
    assert any(term in prompt for term in ("置信度下降", "明确指出是哪个报表", "【数据缺失】"))
    assert "同比/环比" in prompt or "同比" in prompt
    assert "成本敏感性" in prompt or "敏感性分析" in prompt

    # 6. 项目符号列表要求
    assert "项目符号列表" in prompt
    assert "不要用 Markdown 表格" in prompt


@pytest.mark.parametrize("key", ["news_system_message", "social_system_message", "market_system_message", "smart_money_system_message"])
def test_what_why_sowhat_whatnext_framework_present(key):
    """T4: news, social, market, and smart_money must inject the What/Why/SoWhat/WhatNext framework."""
    prompt = ZH_PROMPTS[key]
    assert "What" in prompt
    assert "Why" in prompt
    assert "So What" in prompt or "SoWhat" in prompt
    assert "What Next" in prompt or "WhatNext" in prompt


def test_news_system_message_deep_framework():
    """T4: news_system_message must include cross-market and supply chain linkage, and expected value gap."""
    prompt = ZH_PROMPTS["news_system_message"]
    assert any(term in prompt for term in ("跨市场", "产业链联动", "传导机制", "预期差", "定价充分性"))
    assert "明确区分\"事实\"与\"推断\"" in prompt or "明确区分'事实'与'推断'" in prompt


def test_social_system_message_deep_framework():
    """T4: social_system_message must include sentiment quantification, sentiment life cycle, and reflexivity risk."""
    prompt = ZH_PROMPTS["social_system_message"]
    assert any(term in prompt for term in ("涨停板情绪池", "涨停池", "雪球"))
    assert any(term in prompt for term in ("反身性", "极值", "极度贪婪", "极度恐惧"))
    assert any(term in prompt for term in ("持续性判断", "扩散路径", "生命周期"))


def test_market_system_message_deep_framework():
    """T4: market_system_message must maintain allowed indicators and inject index resonance."""
    prompt = ZH_PROMPTS["market_system_message"]
    assert "close_50_sma, close_200_sma, close_10_ema, macd, macds, macdh, rsi, boll, boll_ub, boll_lb, atr, vwma, mfi" in prompt
    assert any(term in prompt for term in ("大盘/行业共振", "共振", "Beta", "Alpha", "相对强弱"))
    assert any(term in prompt for term in ("演化推演", "演化路径", "多空情景"))


def test_smart_money_system_message_deep_framework():
    """T4: smart_money_system_message must include institutional vs hot money seats, intention analysis, and capital synergy."""
    prompt = ZH_PROMPTS["smart_money_system_message"]
    assert any(term in prompt for term in ("龙虎榜", "机构专用席位", "游资席位"))
    assert any(term in prompt for term in ("建仓", "派发", "洗盘", "震仓"))
    assert any(term in prompt for term in ("协同", "抽血", "筹码", "预期差"))


def _make_analyst_state(ticker="600519", trade_date="2026-07-31", horizon="short"):
    return {
        "trade_date": trade_date,
        "company_of_interest": ticker,
        "horizon": horizon,
        "user_intent": {
            "raw_query": "宏观与个股全景分析",
            "ticker": ticker,
            "horizons": ["short", "medium"],
            "focus_areas": ["宏观联动", "产业链地位", "资金流向"],
            "specific_questions": [],
        },
    }


def _stub_collector_pool():
    return {
        "stock_data": "date,open,high,low,close,volume\n2026-07-31,100,105,98,103,10000",
        "indicators": {
            "close_50_sma": "100", "close_200_sma": "95", "close_10_ema": "102",
            "rsi": "58", "macd": "0.5", "boll": "100", "boll_ub": "108", "boll_lb": "92",
            "atr": "2.5", "vwma": "101", "mfi": "55",
        },
        "fund_flow_board": "行业板块资金流入排名前三：半导体、电力设备、有色金属",
        "fund_flow_individual": "主力净流入：+5000万元，超大单流入：+3000万元",
        "market_data_context": {
            "fund_flow_evidence": {
                "selection": {
                    "status": "selected",
                    "direction_allowed": True,
                    "selected_source": "eastmoney",
                    "selected_field": "r0_net",
                    "selected_value": "1.5",
                    "hard_guard": {"blocked": False},
                },
                "consensus": {
                    "status": "selected",
                    "direction_allowed": True,
                    "selected_source": "eastmoney",
                    "selected_field": "r0_net",
                    "selected_value": "1.5",
                    "hard_guard": {"blocked": False},
                },
                "validation": {
                    "status": "not_checked",
                    "hard_guard": {"blocked": False},
                },
                "records": [],
            }
        },
        "news": "2026-07-30 公司签订战略合作框架协议，拓展海外高端市场",
        "global_news": "2026-07-30 央行宣布精准支持科技创新与设备更新结构性货币政策工具",
        "zt_pool": "今日涨停家数 45 家，连板最高 4 板，炸板率 20%",
        "hot_stocks": "雪球热搜前五：茅台、宁德、中芯、比亚迪、赛力斯",
        "lhb": "龙虎榜买入前五中包含 2 家机构专用席位，合计净买入 1.2 亿元",
        "fundamentals": "2026Q2 净利润同比增长 18%，毛利率维持在 48%",
        "balance_sheet": "资产负债率 35%，货币资金充裕，有息负债率低于 5%",
        "cashflow": "经营活动现金流净额 15 亿元，自由现金流充沛",
        "income_statement": "营业收入 100 亿元，归母净利润 25 亿元",
        "_data_window": "14天",
        "_horizon": "short",
    }


@pytest.mark.parametrize(
    "analyst_factory, report_key, expected_agent",
    [
        (create_macro_analyst, "macro_report", "macro_analyst"),
        (create_fundamentals_analyst, "fundamentals_report", "fundamentals_analyst"),
        (create_news_analyst, "news_report", "news_analyst"),
        (create_social_media_analyst, "sentiment_report", "social_media_analyst"),
        (create_market_analyst, "market_report", "market_analyst"),
        (create_smart_money_analyst, "smart_money_report", "smart_money_analyst"),
    ],
)
def test_analyst_nodes_end_to_end_with_new_prompts(analyst_factory, report_key, expected_agent):
    """Verify that each analyst node loads the updated prompt from zh.py, executes, and extracts verdict."""
    received_messages = []
    sample_verdict = '<!-- VERDICT: {"direction": "偏多", "reason": "深度思考推演结论偏多"} -->'
    sample_response = f"【正式深度分析报告】\n基于系统指令与多层联动分析。\n{sample_verdict}"

    mock_llm = MagicMock()

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream

    collector = DataCollector()
    collector._cache["600519_2026-07-31"] = _stub_collector_pool()

    node = analyst_factory(mock_llm, collector)
    state = _make_analyst_state()
    result = asyncio.run(node(state))

    assert report_key in result
    assert sample_verdict in result[report_key]
    assert "analyst_traces" in result
    assert len(result["analyst_traces"]) == 1
    trace = result["analyst_traces"][0]
    assert trace["agent"] == expected_agent
    assert trace["verdict"] == "偏多"

    # Verify system message came from enhanced zh prompt
    system_msgs = [m for m in received_messages if hasattr(m, "content") and "【输出纪律】" in m.content]
    assert len(system_msgs) >= 1
