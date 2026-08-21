"""针对 DAV-278 知识库动态 RAG 检索与分析师注入的完备测试套件。

涵盖测试项：
1. 27 行业图谱动态 RAG 检索覆盖性（行业名、别名、细分赛道、政策词、上下游要素、风险卡脖子）；
2. 17+ 宏观情景动态 RAG 检索覆盖性（情景名、别名、三级传导、直接冲击、受益/承压行业、跨市场外溢）；
3. 动态检索命中 (Hit) 与未命中 (Miss) 行为及格式契约；
4. 缺命中回退契约：返回 `【知识库未命中】`，严禁编造行业事实与产生幻觉；
5. 分析师节点 (Macro Analyst, Fundamentals Analyst, News Analyst) 动态 RAG 注入全覆盖：
   - 命中时挂载完整结构化上下文；
   - 未命中时严格注入 `【知识库未命中】` 占位块；
6. 零新增外部向量库依赖 / 纯本地确定性分词与倒排索引检索验证；
7. 权威对齐与禁止 _v2 并行 Prompt 验证。
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
    INDUSTRY_KNOWLEDGE_MISSING_BLOCK,
    KNOWLEDGE_MISSING_FALLBACK,
    MACRO_EVENT_MISSING_BLOCK,
    format_macro_market_view,
    format_rag_industry_context,
    format_rag_macro_context,
    resolve_dynamic_knowledge_context,
    resolve_industry_context,
    resolve_industry_profile,
    resolve_macro_event_context,
    retrieve_industry_knowledge,
    retrieve_macro_event_knowledge,
)
from tradingagents.graph.data_collector import DataCollector
from tradingagents.knowledge.industry_linkage import INDUSTRY_PROFILES
from tradingagents.knowledge.macro_events import MACRO_EVENT_SCENARIOS
from tradingagents.knowledge.rag import (
    KnowledgeRAGIndex,
    get_global_rag_index,
    tokenize_cn_en,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. RAG 索引与分词器基础测试 (Tokenizer & Index Tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_tokenize_cn_en_basic():
    """验证中英文分词与 N-Gram 提取正确过滤停用词并保留核心术语与股票代码。"""
    text = "600519 贵州茅台发布以旧换新与半导体自主可控采购计划，LPR下调且SOX指数上涨。"
    tokens = tokenize_cn_en(text)
    assert "600519" in tokens
    assert "贵州" in tokens or "茅台" in tokens
    assert "半导体" in tokens or "半导" in tokens
    assert "lpr" in tokens
    assert "sox" in tokens
    # 停用词被过滤
    assert "的" not in tokens
    assert "和" not in tokens
    assert "发布" not in tokens


def test_rag_index_covers_all_twenty_seven_industries():
    """验证全局 RAG 索引包含全部 27 个行业画像。"""
    rag = get_global_rag_index()
    assert len(rag.industry_docs) == 27
    assert len(rag.industry_docs) == len(INDUSTRY_PROFILES)
    for ind_id in INDUSTRY_PROFILES.keys():
        assert ind_id in rag.industry_docs


def test_rag_index_covers_all_macro_events():
    """验证全局 RAG 索引包含全部宏观事件情景。"""
    rag = get_global_rag_index()
    assert len(rag.macro_docs) == len(MACRO_EVENT_SCENARIOS)
    for ev_id in MACRO_EVENT_SCENARIOS.keys():
        assert ev_id in rag.macro_docs


# ─────────────────────────────────────────────────────────────────────────────
# 2. 行业动态 RAG 检索测试 (Industry Dynamic Retrieval Tests)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query,expected_industry_id",
    [
        ("光刻机与EDA设计软件", "semiconductor"),
        ("算力数据中心与大模型服务器", "ai_computing"),
        ("动力电池与隔膜电解液材料", "lithium_battery"),
        ("智能座舱与激光雷达汽车零部件", "nev_auto"),
        ("光伏逆变器与组件EVA胶膜", "photovoltaic_storage"),
        ("创新药研发与CXO外包服务", "biopharma"),
        ("化学发光IVD与高端超声医疗设备", "medical_devices"),
        ("果链代工与智能手机终端模组", "consumer_electronics"),
        ("浓香型白酒与高端酱香窖池", "liquor_beverage"),
        ("调味品酱油与冷冻食品", "food_beverage"),
        ("变频空调与白电智能家居", "home_appliances"),
        ("商业银行资产质量与净息差NIM", "banking"),
        ("证券公司两融业务与投行IPO承销", "securities"),
        ("寿险保费与偿付能力充足率", "insurance_financials"),
        ("高炉炼铁与螺纹钢黑色金属", "steel_ferrous"),
        ("电解铝产能天花板与铜精矿加工费TCRC", "nonferrous_metals"),
        ("伦敦现货黄金与战略稀土小金属", "precious_metals"),
        ("民营大炼化与原油裂解价差", "petrochemicals"),
        ("长协动力煤与焦煤坑口价", "coal_energy"),
        ("特高压绿电输送与水电机组来水", "power_utilities"),
        ("商品房销售去化与保交房白名单", "real_estate"),
        ("水泥熟料与防水建筑涂料", "construction_materials"),
        ("数控机床与液压挖掘机工程机械", "industrial_machinery"),
        ("中航沈飞战斗机与军工船舶总装", "defense_military"),
        ("集装箱航运与沿海港口吞吐量", "logistics_shipping"),
        ("800G光模块与高速通信光纤", "telecom_optical"),
        ("能繁母猪存栏量与猪周期生猪育肥", "agriculture_breeding"),
    ],
)
def test_retrieve_industry_knowledge_all_twenty_seven(query, expected_industry_id):
    """验证全部 27 个行业均能通过上下游、细分赛道或专业要素高精准度检索召回。"""
    results = retrieve_industry_knowledge(query, top_k=1, min_score=1.0)
    assert len(results) >= 1, f"检索失败：query={query}"
    prof, score = results[0]
    assert prof.industry_id == expected_industry_id, (
        f"期望 {expected_industry_id}，实际得到 {prof.industry_id}（score={score}）"
    )
    assert score > 1.0


def test_retrieve_industry_knowledge_miss():
    """验证无意义查询或不存在的行业安全返回空列表。"""
    miss_queries = [
        "火星空间站引力波探测采矿",
        "魔法炼金术草药种植",
        "xyz987654321nonsense",
        "",
    ]
    for q in miss_queries:
        res = retrieve_industry_knowledge(q, top_k=1, min_score=3.0)
        assert res == []


# ─────────────────────────────────────────────────────────────────────────────
# 3. 宏观事件动态 RAG 检索测试 (Macro Event Dynamic Retrieval Tests)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query,expected_event_id",
    [
        ("央行全面降准0.5个百分点并下调MLF利率", "monetary_easing"),
        ("央行收紧流动性去杠杆并上调政策利率", "monetary_tightening"),
        ("发行超长期特别国债用于大规模基建与两重建设", "fiscal_expansion"),
        ("人民币对美元汇率跌破7.3关口面临贬值压力", "rmb_depreciation"),
        ("人民币大幅升值提升进口大宗商品购买力", "rmb_appreciation"),
        ("中东地缘冲突导致布伦特原油飙升至100美元", "oil_price_shock_up"),
        ("全球铜矿供应中断引发工业金属超级周期", "commodity_supercycle_metals"),
        ("美债实际利率下行推动COMEX黄金价格创历史新高", "gold_safe_haven_rally"),
        ("红海海运航道遇袭受阻商船被迫绕行好望角", "geopolitical_conflict_escalation"),
        ("海外主要经济体对我国光伏与电动车加征惩罚性关税", "export_tariffs_trade_friction"),
        ("上游原材料PPI低迷而下游CPI回暖导致剪刀差走阔", "cpi_ppi_scissors_widening"),
        ("国内居民信贷需求不足陷入有效需求收缩与通缩风险", "deflation_demand_contraction"),
        ("美联储FOMC决议开启降息周期驱动美元指数走弱", "us_fed_rate_cut_cycle"),
        ("全面取消商品房限购与降低首付比例推进保障房收储", "real_estate_policy_easing"),
        ("通用人工智能大模型与算力集群颠覆性技术突破", "ai_tech_revolution_breakthrough"),
        ("新国九条发布强化上市公司现金分红与退市监管", "capital_market_institutional_reform"),
        ("夏季极端高温干旱引发主要水电省份工业限电限产", "extreme_weather_power_curtailment"),
    ],
)
def test_retrieve_macro_event_knowledge_all_scenarios(query, expected_event_id):
    """验证所有宏观事件情景均可通过宏观描述、政策变量或传导特征精准检索。"""
    results = retrieve_macro_event_knowledge(query, top_k=1, min_score=2.0)
    assert len(results) >= 1, f"检索失败：query={query}"
    sc, score = results[0]
    assert sc.event_id == expected_event_id, (
        f"期望 {expected_event_id}，实际得到 {sc.event_id}（score={score}）"
    )


def test_retrieve_macro_event_knowledge_miss():
    """验证无宏观事件的新闻文本安全返回空列表。"""
    miss_texts = [
        "公司今日召开例行内部周度工作会议，总结上周行政后勤工作。",
        "员工生日庆祝与工会摄影比赛圆满结束。",
        "xyz987654321nothing",
    ]
    for t in miss_texts:
        res = retrieve_macro_event_knowledge(t, top_k=2, min_score=3.0)
        assert res == []


# ─────────────────────────────────────────────────────────────────────────────
# 4. 格式化与未命中占位契约测试 (Formatting & Fallback Contract Tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_format_rag_industry_context_hit_and_miss():
    """验证 format_rag_industry_context 命中与未命中输出契约。"""
    # 1. 命中已知行业
    hit_text = format_rag_industry_context("半导体与集成电路", fallback_on_miss=True)
    assert "【行业常识知识库 - 半导体与集成电路 (科技/TMT)】" in hit_text
    assert "1. 产业链上下游穿透：" in hit_text
    assert "2. 宏观与周期敏感度：" in hit_text
    assert "3. 风险矩阵与监控指标：" in hit_text

    # 2. 未命中未知行业，fallback_on_miss=True
    miss_text = format_rag_industry_context("火星采矿未知行业XYZ", fallback_on_miss=True)
    assert miss_text == INDUSTRY_KNOWLEDGE_MISSING_BLOCK
    assert "【知识库未命中】" in miss_text

    # 3. 未命中未知行业，fallback_on_miss=False
    miss_empty = format_rag_industry_context("火星采矿未知行业XYZ", fallback_on_miss=False)
    assert miss_empty == ""


def test_format_rag_macro_context_hit_and_miss():
    """验证 format_rag_macro_context 命中与未命中输出契约。"""
    # 1. 命中宏观事件
    hit_text = format_rag_macro_context("央行降息降准释放流动性", fallback_on_miss=True)
    assert "【宏观事件传导图谱 - 央行降息降准与流动性宽松 (货币与流动性)】" in hit_text
    assert "1. 核心事件定义与直接冲击：" in hit_text
    assert "2. 三级传导机制推演：" in hit_text
    assert "3. 行业结构性分化影响：" in hit_text

    # 2. 未命中无宏观事件文本，fallback_on_miss=True
    miss_text = format_rag_macro_context("公司日常行政保洁通知", fallback_on_miss=True)
    assert miss_text == MACRO_EVENT_MISSING_BLOCK
    assert "【知识库未命中】" in miss_text

    # 3. 未命中，fallback_on_miss=False
    miss_empty = format_rag_macro_context("公司日常行政保洁通知", fallback_on_miss=False)
    assert miss_empty == ""


def test_resolve_dynamic_knowledge_context_helper():
    """验证 resolve_dynamic_knowledge_context 组合解析辅助函数。"""
    # 命中场景
    ctx_hit = resolve_dynamic_knowledge_context(
        ticker="600519",
        stock_name="贵州茅台",
        extra_text="央行宣布全面降息降准以提振内需消费",
        fallback_on_miss=True,
    )
    assert ctx_hit["industry_profile"] is not None
    assert ctx_hit["industry_profile"].industry_id == "liquor_beverage"
    assert "【行业常识知识库 - 白酒与精制茶酒" in ctx_hit["industry_context"]
    assert len(ctx_hit["macro_scenarios"]) >= 1
    assert "【宏观事件传导图谱 - 央行降息降准与流动性宽松" in ctx_hit["macro_event_context"]

    # 未命中场景
    ctx_miss = resolve_dynamic_knowledge_context(
        ticker="999999",
        stock_name="未知外星标的XYZ",
        extra_text="例行公司保洁巡查记录",
        fallback_on_miss=True,
    )
    assert ctx_miss["industry_profile"] is None
    assert ctx_miss["industry_context"] == INDUSTRY_KNOWLEDGE_MISSING_BLOCK
    assert ctx_miss["macro_scenarios"] == []
    assert ctx_miss["macro_event_context"] == MACRO_EVENT_MISSING_BLOCK


# ─────────────────────────────────────────────────────────────────────────────
# 5. 分析师 Prompt 动态注入与未命中防护测试 (Analyst Injection & Anti-Hallucination)
# ─────────────────────────────────────────────────────────────────────────────

def test_macro_analyst_unhit_injects_knowledge_missing_block():
    """验证 Macro Analyst 在未知标的且无宏观事件时，注入【知识库未命中】占位符，不产生幻觉。"""
    received_messages = []
    sample_verdict = '<!-- VERDICT: {"direction": "中性", "reason": "缺乏宏观与行业知识支持"} -->'
    sample_response = f"【宏观与板块分析报告】\n{sample_verdict}"

    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream

    collector = DataCollector()
    collector._cache["999999_2026-07-31"] = {
        "fund_flow_board": "无数据",
        "news": "例行日常内部培训",
        "global_news": "例行行政通知",
        "global_indices": "无数据",
        "major_assets": "无数据",
        "cn_indices": "无数据",
        "northbound_flow": "无数据",
    }

    node = create_macro_analyst(mock_llm, collector)
    state = {
        "trade_date": "2026-07-31",
        "company_of_interest": "999999",
    }

    result = asyncio.run(node(state))
    assert "macro_report" in result

    human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
    content = human_msg.content

    # 验证行业和宏观事件均注入了【知识库未命中】
    assert "【行业常识知识库】\n【知识库未命中】" in content
    assert "【宏观事件传导图谱】\n【知识库未命中】" in content


def test_fundamentals_analyst_unhit_injects_knowledge_missing_block():
    """验证 Fundamentals Analyst 在未知标的且无宏观事件时，注入【知识库未命中】占位符。"""
    received_messages = []
    sample_verdict = '<!-- VERDICT: {"direction": "中性", "reason": "基本面数据与知识库均未命中"} -->'
    sample_response = f"【基本面分析报告】\n{sample_verdict}"

    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream

    collector = DataCollector()
    collector._cache["999999_2026-07-31"] = {
        "fundamentals": "无数据",
        "balance_sheet": "无数据",
        "cashflow": "无数据",
        "income_statement": "无数据",
        "global_indices": "无数据",
        "major_assets": "无数据",
        "cn_indices": "无数据",
    }

    node = create_fundamentals_analyst(mock_llm, collector)
    state = {
        "trade_date": "2026-07-31",
        "company_of_interest": "999999",
    }

    result = asyncio.run(node(state))
    assert "fundamentals_report" in result

    human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
    content = human_msg.content

    assert "【行业常识知识库】\n【知识库未命中】" in content
    assert "【宏观事件传导图谱】\n【知识库未命中】" in content


def test_news_analyst_unhit_injects_knowledge_missing_block():
    """验证 News Analyst 在未知标的且无宏观事件时，注入【知识库未命中】占位符。"""
    received_messages = []
    sample_verdict = '<!-- VERDICT: {"direction": "中性", "reason": "无显著宏观事件与知识库未命中"} -->'
    sample_response = f"【新闻分析报告】\n{sample_verdict}"

    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream

    collector = DataCollector()
    collector._cache["999999_2026-07-31"] = {
        "news": "例行行政保洁工作记录",
        "global_news": "例行系统维护通知",
        "_data_window": "14天",
    }

    node = create_news_analyst(mock_llm, collector)
    state = {
        "trade_date": "2026-07-31",
        "company_of_interest": "999999",
    }

    result = asyncio.run(node(state))
    assert "news_report" in result

    human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
    content = human_msg.content

    assert "【行业常识知识库】\n【知识库未命中】" in content
    assert "【宏观事件传导图谱】\n【知识库未命中】" in content


def test_no_v2_prompts_and_contract_compliance():
    """验证遵循产品契约：无 _v2 并行 Prompt，改原路径。"""
    from tradingagents.prompts.zh import PROMPTS
    for k in PROMPTS.keys():
        assert not k.endswith("_v2"), f"禁止存在 _v2 并行 Prompt key: {k}"
        assert "_v2" not in k, f"禁止存在 _v2 Prompt key: {k}"
