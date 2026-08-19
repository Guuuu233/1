"""Unit tests for DAV-189 T7: Knowledge context mounting and global/macro views in analyst nodes.

Verifies:
1. macro_analyst and fundamentals_analyst correctly mount knowledge base context
   (format_industry_deep_context, format_macro_event_context) and global/domestic views.
2. Graceful behavior in both collector pool mode and no-collector fallback mode.
3. Accurate industry resolution and macro event scenario extraction.
4. Output integrity, verdict contracts, degradation handling, and trace generation.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from tradingagents.agents.analysts.macro_analyst import create_macro_analyst
from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.utils.knowledge_context import (
    format_macro_market_view,
    resolve_industry_context,
    resolve_industry_profile,
    resolve_macro_event_context,
)
from tradingagents.graph.data_collector import DataCollector
from tradingagents.knowledge.industry_linkage import INDUSTRY_PROFILES


# ─────────────────────────────────────────────────────────────────────────────
# 1. 行业解析与知识库挂载测试 (Industry Resolution Tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_industry_profile_by_state():
    """State 中的 explicit industry 优先级最高。"""
    state = {"industry": "白酒与精制茶酒"}
    p = resolve_industry_profile(ticker="000001", stock_name="平安银行", state=state)
    assert p is not None
    assert p.industry_id == "liquor_beverage"


def test_resolve_industry_profile_by_stock_name_dict():
    """典型股票名称字典直接精确映射。"""
    cases = [
        ("600519", "贵州茅台", "liquor_beverage"),
        ("300750", "宁德时代", "lithium_battery"),
        ("688981", "中芯国际", "semiconductor"),
        ("002594", "比亚迪", "nev_auto"),
        ("600036", "招商银行", "banking"),
        ("601318", "中国平安", "insurance_financials"),
        ("601088", "中国神华", "coal_energy"),
        ("600900", "长江电力", "power_utilities"),
        ("601899", "紫金矿业", "precious_metals"),
        ("600276", "恒瑞医药", "biopharma"),
        ("300760", "迈瑞医疗", "medical_devices"),
        ("002475", "立讯精密", "consumer_electronics"),
        ("002230", "科大讯飞", "ai_computing"),
        ("601012", "隆基绿能", "photovoltaic_storage"),
        ("600030", "中信证券", "securities"),
        ("600019", "宝钢股份", "steel_ferrous"),
        ("601857", "中国石油", "petrochemicals"),
        ("000002", "万科A", "real_estate"),
        ("600585", "海螺水泥", "construction_materials"),
        ("600031", "三一重工", "industrial_machinery"),
        ("600760", "中航沈飞", "defense_military"),
        ("601919", "中远海控", "logistics_shipping"),
        ("000063", "中兴通讯", "telecom_optical"),
        ("002714", "牧原股份", "agriculture_breeding"),
        ("000333", "美的集团", "home_appliances"),
        ("600887", "伊利股份", "food_beverage"),
    ]
    for ticker, name, expected_id in cases:
        p = resolve_industry_profile(ticker=ticker, stock_name=name)
        assert p is not None, f"Failed for {ticker} {name}"
        assert p.industry_id == expected_id, f"Expected {expected_id}, got {p.industry_id} for {name}"


def test_resolve_industry_profile_by_generic_keywords():
    """通用行业后缀与关键词规则匹配。"""
    cases = [
        ("华夏银行", "banking"),
        ("东吴证券", "securities"),
        ("太平洋保险", "insurance_financials"),
        ("天士力制药", "biopharma"),
        ("鱼跃医疗", "medical_devices"),
        ("欣旺达动力电池", "lithium_battery"),
        ("晶澳光伏", "photovoltaic_storage"),
        ("赛力斯智能汽车", "nev_auto"),
        ("长电芯片", "semiconductor"),
        ("拓维算力", "ai_computing"),
        ("歌尔智能终端", "consumer_electronics"),
        ("首钢钢铁", "steel_ferrous"),
        ("中金黄金", "precious_metals"),
        ("华鲁化工", "petrochemicals"),
        ("山西焦煤", "coal_energy"),
        ("国投水电", "power_utilities"),
        ("保利地产", "real_estate"),
        ("天山水泥", "construction_materials"),
        ("华中数控机床", "industrial_machinery"),
        ("成飞航空装备", "defense_military"),
        ("宁波港口", "logistics_shipping"),
        ("天孚光模块", "telecom_optical"),
        ("新希望生猪养殖", "agriculture_breeding"),
        ("海信家电", "home_appliances"),
        ("安井食品", "food_beverage"),
    ]
    for name, expected_id in cases:
        p = resolve_industry_profile(stock_name=name)
        assert p is not None, f"Failed for {name}"
        assert p.industry_id == expected_id, f"Expected {expected_id}, got {p.industry_id} for {name}"


def test_resolve_industry_context_formatting():
    """验证 resolve_industry_context 成功生成结构化知识库文本。"""
    p, ctx = resolve_industry_context(ticker="600519", stock_name="贵州茅台")
    assert p is not None
    assert p.industry_id == "liquor_beverage"
    assert "【行业常识知识库 - 白酒与精制茶酒 (大消费)】" in ctx
    assert "1. 产业链上下游穿透：" in ctx
    assert "2. 宏观与周期敏感度：" in ctx
    assert "3. 风险矩阵与监控指标：" in ctx
    assert "定价权与传导：" in ctx


def test_resolve_industry_context_unknown_fallback():
    """无法识别的标的返回 None 和空字符串，不产生幻觉。"""
    p, ctx = resolve_industry_context(ticker="999999", stock_name="未知公司XYZ")
    assert p is None
    assert ctx == ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. 宏观事件情景图谱测试 (Macro Event Resolution Tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_macro_event_context_hit():
    """从新闻文本中识别宏观事件情景并生成三级传导链路。"""
    news = "央行今日宣布下调存款准备金率0.5个百分点，实施全面降准降息并增加流动性供给。"
    scenarios, ctx = resolve_macro_event_context(news, max_events=2)
    assert len(scenarios) >= 1
    assert any(s.event_id == "monetary_easing" for s in scenarios)
    assert "【宏观事件传导图谱 - 央行降息降准与流动性宽松" in ctx
    assert "1. 核心事件定义与直接冲击：" in ctx
    assert "2. 三级传导机制推演：" in ctx
    assert "3. 行业结构性分化影响：" in ctx
    assert "4. 跨市场联动与高频监测：" in ctx


def test_resolve_macro_event_context_no_hit():
    """无宏观事件的新闻返回空列表和空字符串。"""
    news = "公司发布例行日常经营与员工培训公告。"
    scenarios, ctx = resolve_macro_event_context(news)
    assert scenarios == []
    assert ctx == ""


def test_format_macro_market_view():
    """格式化大盘与全球市场视图。"""
    view = format_macro_market_view(
        global_indices="标普500: +0.8%, 纳斯达克: +1.2%, 恒生指数: +1.5%",
        major_assets="COMEX黄金: $2450/oz, 布伦特原油: $82/bbl, 美债10Y: 4.15%",
        cn_indices="沪深300: +1.1%, 上证指数: 3100 (+0.9%), 创业板指: +1.8%",
        northbound_flow="北向资金今日净流入 +85.2 亿元",
    )
    assert "【全球核心指数】" in view
    assert "【大类资产与宏观商品】" in view
    assert "【国内核心大盘指数】" in view
    assert "【北向资金与跨市场流动性】" in view


# ─────────────────────────────────────────────────────────────────────────────
# 3. Macro Analyst 端到端测试 (Macro Analyst Node Tests)
# ─────────────────────────────────────────────────────────────────────────────

def _make_macro_collector_pool():
    return {
        "global_indices": "标普500: 5500 (+0.6%), 纳斯达克: 17500 (+0.9%), 恒生指数: 17800 (+1.2%)",
        "major_assets": "黄金: $2420/oz, 原油: $80.5/bbl, 美债10年期: 4.12%, 美元指数: 103.8",
        "cn_indices": "上证指数: 3120 (+0.8%), 沪深300: 3650 (+1.0%), 创业板指: 1880 (+1.5%)",
        "northbound_flow": "北向资金单日净买入 65.4 亿元，重点加仓白酒与电新",
        "fund_flow_board": "行业板块资金净流入前三：半导体 (+35亿)、白酒 (+28亿)、电力设备 (+22亿)",
        "news": "2026-07-31 贵州茅台发布半年报业绩预增公告，海外渠道拓展加速",
        "global_news": "2026-07-31 央行宣布实施降准降息，进一步释放中长期流动性",
        "_data_window": "14天",
        "_horizon": "medium",
    }


def test_macro_analyst_node_with_collector_pool():
    """验证 Macro Analyst 在 DataCollector pool 下完整挂载知识库与全球大盘视图。"""
    received_messages = []
    sample_verdict = '<!-- VERDICT: {"direction": "看多", "reason": "宏观宽松且板块资金强劲流入"} -->'
    sample_response = f"【宏观与板块深度分析报告】\n基于自上而下全景分析框架。\n{sample_verdict}"

    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream

    collector = DataCollector()
    collector._cache["600519_2026-07-31"] = _make_macro_collector_pool()

    node = create_macro_analyst(mock_llm, collector)
    state = {
        "trade_date": "2026-07-31",
        "company_of_interest": "600519",
        "user_intent": {"focus_areas": ["宏观流动性", "板块资金"], "specific_questions": []},
    }

    result = asyncio.run(node(state))

    assert "macro_report" in result
    assert sample_verdict in result["macro_report"]
    assert "analyst_traces" in result
    assert result["analyst_traces"][0]["verdict"] == "看多"

    # 检查 HumanMessage 中注入的内容
    human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
    content = human_msg.content

    # 1. 全球宏观与大盘指数
    assert "【全球核心指数】" in content
    assert "标普500" in content
    assert "【大类资产与宏观商品】" in content
    assert "黄金" in content
    assert "【国内大盘核心指数】" in content
    assert "沪深300" in content
    assert "【北向资金与跨市场流动性】" in content

    # 2. 板块资金与新闻
    assert "【今日行业板块资金流向】" in content
    assert "【近期相关新闻】" in content

    # 3. 知识库与宏观事件传导图谱
    assert "【行业常识知识库 - 白酒与精制茶酒" in content
    assert "产业链上下游穿透" in content
    assert "【宏观事件传导图谱 - 央行降息降准与流动性宽松" in content


def test_macro_analyst_node_fallback_without_collector():
    """验证 Macro Analyst 在无 DataCollector 时走异步回退并挂载知识库。"""
    received_messages = []
    sample_verdict = '<!-- VERDICT: {"direction": "偏多", "reason": "板块资金与新闻面共振"} -->'
    sample_response = f"【宏观与板块分析报告】\n{sample_verdict}"

    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream

    # 模拟 tool 的返回值
    with patch("tradingagents.agents.utils.agent_utils.get_board_fund_flow") as mock_board, \
         patch("tradingagents.agents.utils.agent_utils.get_news") as mock_news, \
         patch("tradingagents.agents.utils.agent_utils.get_global_news") as mock_gnews, \
         patch("tradingagents.agents.utils.agent_utils.get_northbound_flow") as mock_nb:

        mock_board.invoke.return_value = "板块资金净流入：半导体 +10亿"
        mock_news.invoke.return_value = "2026-07-31 688981 中芯国际 产能利用率大幅提升"
        mock_gnews.invoke.return_value = "2026-07-31 国家大力支持集成电路与半导体自主可控战略"
        mock_nb.invoke.return_value = "北向资金净买入 +2000万元"

        node = create_macro_analyst(mock_llm, data_collector=None)
        state = {
            "trade_date": "2026-07-31",
            "company_of_interest": "688981",
        }

        result = asyncio.run(node(state))

        assert "macro_report" in result
        assert sample_verdict in result["macro_report"]

        human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
        content = human_msg.content

        # 验证知识库挂载了半导体行业
        assert "【行业常识知识库 - 半导体与集成电路" in content
        assert "【今日行业板块资金流向】" in content
        assert "【近期相关新闻】" in content


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fundamentals Analyst 端到端测试 (Fundamentals Analyst Node Tests)
# ─────────────────────────────────────────────────────────────────────────────

def _make_fundamentals_collector_pool():
    return {
        "fundamentals": "2026Q2 营业收入 540 亿元 (+18%)，归母净利润 270 亿元 (+19%)，毛利率 91.5%",
        "balance_sheet": "总资产 3200 亿元，净资产 2800 亿元，资产负债率 12.5%，有息负债为 0",
        "cashflow": "经营活动产生的现金流量净额 260 亿元，资本开支 45 亿元，自由现金流强劲",
        "income_statement": "营业总成本 170 亿元，期间费用率 6.3%，净利率 52.2%",
        "global_indices": "标普500: +0.5%, 纳斯达克: +0.8%",
        "major_assets": "黄金: $2400/oz, 原油: $80/bbl, 工业金属: 偏强",
        "cn_indices": "沪深300: 3600 (+0.9%), 上证指数: 3100 (+0.7%)",
        "_data_window": "90天",
        "_horizon": "medium",
    }


def test_fundamentals_analyst_node_with_collector_pool():
    """验证 Fundamentals Analyst 在 DataCollector pool 下完整挂载行业知识库与大盘宏观视图。"""
    received_messages = []
    sample_verdict = '<!-- VERDICT: {"direction": "看多", "reason": "高毛利与强劲现金流，上下游议价权极强"} -->'
    sample_response = f"【基本面穿透分析报告】\n基于产业链地位与财务质量全面解构。\n{sample_verdict}"

    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream

    collector = DataCollector()
    collector._cache["600519_2026-07-31"] = _make_fundamentals_collector_pool()

    node = create_fundamentals_analyst(mock_llm, collector)
    state = {
        "trade_date": "2026-07-31",
        "company_of_interest": "600519",
        "user_intent": {"focus_areas": ["产业链议价权", "财务质量"], "specific_questions": []},
    }

    result = asyncio.run(node(state))

    assert "fundamentals_report" in result
    assert sample_verdict in result["fundamentals_report"]
    assert "analyst_traces" in result
    assert result["analyst_traces"][0]["agent"] == "fundamentals_analyst"
    assert result["analyst_traces"][0]["verdict"] == "看多"

    human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
    content = human_msg.content

    # 1. 行业常识知识库挂载
    assert "【行业常识知识库 - 白酒与精制茶酒" in content
    assert "产业链上下游穿透" in content
    assert "定价权与传导" in content
    assert "风险矩阵与监控指标" in content

    # 2. 宏观大盘与大类资产背景
    assert "【大类资产与宏观大盘背景】" in content
    assert "大类资产与商品（成本端/通胀参考）" in content
    assert "国内大盘核心指数" in content

    # 3. 四大财报原始数据
    assert "【get_fundamentals】" in content
    assert "【get_balance_sheet】" in content
    assert "【get_cashflow】" in content
    assert "【get_income_statement】" in content


def test_fundamentals_analyst_node_fallback_without_collector():
    """验证 Fundamentals Analyst 在无 DataCollector 时走异步回退并挂载知识库。"""
    received_messages = []
    sample_verdict = '<!-- VERDICT: {"direction": "偏多", "reason": "动力电池出货高增，毛利率环比改善"} -->'
    sample_response = f"【基本面分析报告】\n{sample_verdict}"

    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream

    with patch("tradingagents.agents.utils.agent_utils.get_fundamentals") as mock_fund, \
         patch("tradingagents.agents.utils.agent_utils.get_balance_sheet") as mock_bs, \
         patch("tradingagents.agents.utils.agent_utils.get_cashflow") as mock_cf, \
         patch("tradingagents.agents.utils.agent_utils.get_income_statement") as mock_is:

        mock_fund.invoke.return_value = "2026Q2 动力电池与储能电池营收高增 25%"
        mock_bs.invoke.return_value = "资产负债率 55%，流动比率 1.6"
        mock_cf.invoke.return_value = "经营现金流净额 120 亿元"
        mock_is.invoke.return_value = "净利润 130 亿元"

        node = create_fundamentals_analyst(mock_llm, data_collector=None)
        state = {
            "trade_date": "2026-07-31",
            "company_of_interest": "300750",
        }

        result = asyncio.run(node(state))

        assert "fundamentals_report" in result
        assert sample_verdict in result["fundamentals_report"]

        human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
        content = human_msg.content

        # 验证动力电池知识库成功挂载
        assert "【行业常识知识库 - 动力电池与储能电池材料" in content
        assert "【get_fundamentals】" in content
        assert "【get_balance_sheet】" in content


# ─────────────────────────────────────────────────────────────────────────────
# 5. News Analyst 宏观情景图谱挂载测试
# ─────────────────────────────────────────────────────────────────────────────

def test_news_analyst_node_mounts_macro_event():
    """验证 News Analyst 在包含宏观事件的新闻下自动挂载宏观事件传导图谱。"""
    received_messages = []
    sample_verdict = '<!-- VERDICT: {"direction": "看多", "reason": "宏观降准释放流动性，政策利好显著"} -->'
    sample_response = f"【新闻分析报告】\n{sample_verdict}"

    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream

    pool = {
        "news": "2026-07-31 公司获得央行碳减排支持工具低息贷款支持",
        "global_news": "2026-07-31 央行决定全面降准降息以支持实体经济",
        "_data_window": "14天",
    }
    collector = DataCollector()
    collector._cache["600519_2026-07-31"] = pool

    node = create_news_analyst(mock_llm, collector)
    state = {
        "trade_date": "2026-07-31",
        "company_of_interest": "600519",
    }

    result = asyncio.run(node(state))

    assert "news_report" in result
    human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
    assert "【宏观事件传导图谱 - 央行降息降准与流动性宽松" in human_msg.content
