import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START

from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.analysts.smart_money_analyst import create_smart_money_analyst
from tradingagents.agents.analysts.volume_price_analyst import create_volume_price_analyst
from tradingagents.agents.utils.context_utils import format_phase1_reports
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.data_collector import DataCollector
from tradingagents.graph.setup import GraphSetup


class _TopologyRecordingWorkflow:
    """Mock workflow recording nodes, edges, and conditional edges."""

    def __init__(self, state_cls=None):
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []

    def add_node(self, name, node):
        self.nodes[name] = node

    def add_edge(self, source, target):
        if isinstance(source, list):
            self.edges.append((tuple(source), target))
        else:
            self.edges.append((source, target))

    def add_conditional_edges(self, source, condition, mapping):
        self.conditional_edges.append((source, condition, mapping))

    def compile(self, checkpointer=None):
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "conditional_edges": self.conditional_edges,
            "checkpointer": checkpointer,
        }


def _make_mock_factories():
    dummy_node = lambda s: {}
    return {
        "create_aggressive_debator": MagicMock(return_value=dummy_node),
        "create_bear_researcher": MagicMock(return_value=dummy_node),
        "create_bull_researcher": MagicMock(return_value=dummy_node),
        "create_conservative_debator": MagicMock(return_value=dummy_node),
        "create_fundamentals_analyst": MagicMock(return_value=dummy_node),
        "create_macro_analyst": MagicMock(return_value=dummy_node),
        "create_market_analyst": MagicMock(return_value=dummy_node),
        "create_neutral_debator": MagicMock(return_value=dummy_node),
        "create_news_analyst": MagicMock(return_value=dummy_node),
        "create_research_manager": MagicMock(return_value=dummy_node),
        "create_risk_manager": MagicMock(return_value=dummy_node),
        "create_smart_money_analyst": MagicMock(return_value=dummy_node),
        "create_social_media_analyst": MagicMock(return_value=dummy_node),
        "create_volume_price_analyst": MagicMock(return_value=dummy_node),
        "create_trader": MagicMock(return_value=dummy_node),
    }


def _make_graph_setup():
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    tool_nodes = {
        "market": MagicMock(),
        "social": MagicMock(),
        "news": MagicMock(),
        "fundamentals": MagicMock(),
        "macro": MagicMock(),
        "smart_money": MagicMock(),
        "volume_price": MagicMock(),
    }
    return GraphSetup(
        quick_thinking_llm=MagicMock(),
        deep_thinking_llm=MagicMock(),
        tool_nodes=tool_nodes,
        bull_memory=MagicMock(),
        bear_memory=MagicMock(),
        trader_memory=MagicMock(),
        invest_judge_memory=MagicMock(),
        risk_manager_memory=MagicMock(),
        conditional_logic=conditional_logic,
        data_collector=MagicMock(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. 拓扑与连边测试：全 7 分析师两阶段拓扑
# ─────────────────────────────────────────────────────────────────────────────

def test_full_7_analysts_two_stage_topology():
    """验证 7 分析师拓扑：
    - 阶段一（Macro/Market/Social）由 START 并行出发
    - 阶段一完成汇合（Barrier），广播至阶段二（Fundamentals/News/Smart Money/Volume Price）
    - 阶段二完成汇合，进入 Bull Researcher
    """
    setup = _make_graph_setup()
    factories = _make_mock_factories()

    all_analysts = ["macro", "market", "social", "fundamentals", "news", "smart_money", "volume_price"]

    with patch("tradingagents.graph.setup._load_agent_factories", return_value=factories), \
         patch("tradingagents.graph.setup.StateGraph", _TopologyRecordingWorkflow):
        compiled = setup.setup_graph(all_analysts)

    edges = compiled["edges"]

    # 1. 阶段一节点直接由 START 出发
    assert (START, "Macro Analyst") in edges
    assert (START, "Market Analyst") in edges
    assert (START, "Social Analyst") in edges

    # 阶段二节点禁止由 START 出发
    assert (START, "Fundamentals Analyst") not in edges
    assert (START, "News Analyst") not in edges
    assert (START, "Smart Money Analyst") not in edges
    assert (START, "Volume Price Analyst") not in edges

    # 2. 阶段一 Done 汇合广播到阶段二全部节点
    phase1_dones = ("Macro Analyst Done", "Market Analyst Done", "Social Analyst Done")
    assert (phase1_dones, "Fundamentals Analyst") in edges
    assert (phase1_dones, "News Analyst") in edges
    assert (phase1_dones, "Smart Money Analyst") in edges
    assert (phase1_dones, "Volume Price Analyst") in edges

    # 3. 阶段二 Done 汇合进入 Bull Researcher
    phase2_dones = (
        "Fundamentals Analyst Done",
        "News Analyst Done",
        "Smart Money Analyst Done",
        "Volume Price Analyst Done",
    )
    assert (phase2_dones, "Bull Researcher") in edges

    # 阶段一 Done 不得直接进入 Bull Researcher
    assert (phase1_dones, "Bull Researcher") not in edges


# ─────────────────────────────────────────────────────────────────────────────
# 2. 子集拓扑测试：仅阶段一 / 仅阶段二 / 部分混合
# ─────────────────────────────────────────────────────────────────────────────

def test_phase1_only_topology():
    """当仅选择阶段一分析师时，阶段一直接汇合进入 Bull Researcher。"""
    setup = _make_graph_setup()
    factories = _make_mock_factories()

    with patch("tradingagents.graph.setup._load_agent_factories", return_value=factories), \
         patch("tradingagents.graph.setup.StateGraph", _TopologyRecordingWorkflow):
        compiled = setup.setup_graph(["macro", "market"])

    edges = compiled["edges"]
    assert (START, "Macro Analyst") in edges
    assert (START, "Market Analyst") in edges
    assert (("Macro Analyst Done", "Market Analyst Done"), "Bull Researcher") in edges


def test_phase2_only_topology():
    """当仅选择阶段二分析师时，阶段二由 START 并行出发，汇合进入 Bull Researcher。"""
    setup = _make_graph_setup()
    factories = _make_mock_factories()

    with patch("tradingagents.graph.setup._load_agent_factories", return_value=factories), \
         patch("tradingagents.graph.setup.StateGraph", _TopologyRecordingWorkflow):
        compiled = setup.setup_graph(["fundamentals", "news"])

    edges = compiled["edges"]
    assert (START, "Fundamentals Analyst") in edges
    assert (START, "News Analyst") in edges
    assert (("Fundamentals Analyst Done", "News Analyst Done"), "Bull Researcher") in edges


def test_mixed_subset_topology():
    """阶段一选 1 个，阶段二选 2 个：正确两阶段串联。"""
    setup = _make_graph_setup()
    factories = _make_mock_factories()

    with patch("tradingagents.graph.setup._load_agent_factories", return_value=factories), \
         patch("tradingagents.graph.setup.StateGraph", _TopologyRecordingWorkflow):
        compiled = setup.setup_graph(["market", "fundamentals", "smart_money"])

    edges = compiled["edges"]
    assert (START, "Market Analyst") in edges
    assert (("Market Analyst Done",), "Fundamentals Analyst") in edges
    assert (("Market Analyst Done",), "Smart Money Analyst") in edges
    assert (("Fundamentals Analyst Done", "Smart Money Analyst Done"), "Bull Researcher") in edges


# ─────────────────────────────────────────────────────────────────────────────
# 3. 阶段一产物格式化与【数据缺失】规范测试
# ─────────────────────────────────────────────────────────────────────────────

def test_format_phase1_reports_with_complete_data():
    """阶段一三份报告均有时，格式化文本包含三份报告内容。"""
    state = {
        "macro_report": "宏观报告：经济温和复苏，半导体行业处于主动补库周期。",
        "market_report": "市场技术面报告：突破 20 日均线，MACD 金叉。",
        "sentiment_report": "情绪报告：社交媒体情绪偏多，涨停连板效应好。",
    }
    formatted = format_phase1_reports(state)

    assert "【阶段一分析师产物（宏观/大盘/情绪）】" in formatted
    assert "【宏观与行业板块结论（阶段一）】" in formatted
    assert "半导体行业处于主动补库周期" in formatted
    assert "【大盘与市场技术面结论（阶段一）】" in formatted
    assert "突破 20 日均线" in formatted
    assert "【市场情绪与舆情结论（阶段一）】" in formatted
    assert "社交媒体情绪偏多" in formatted
    assert "【数据缺失】" not in formatted


def test_format_phase1_reports_with_missing_data():
    """缺数据时必须显式标注【数据缺失】，严禁伪造数据。"""
    # 全空 state
    state = {}
    formatted = format_phase1_reports(state)
    assert "【阶段一分析师产物（宏观/大盘/情绪）】" in formatted
    assert "【数据缺失】宏观板块分析报告缺失" in formatted
    assert "【数据缺失】大盘市场技术分析报告缺失" in formatted
    assert "【数据缺失】市场情绪舆情分析报告缺失" in formatted

    # 部分缺失 state
    state_partial = {
        "macro_report": "宏观面偏多",
        "market_report": "",
        "sentiment_report": "无数据",
    }
    formatted_partial = format_phase1_reports(state_partial)
    assert "宏观面偏多" in formatted_partial
    assert "【数据缺失】大盘市场技术分析报告缺失" in formatted_partial
    assert "【数据缺失】市场情绪舆情分析报告缺失" in formatted_partial


# ─────────────────────────────────────────────────────────────────────────────
# 4. 阶段二 4 个分析师 prompt 读取阶段一产物测试
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_collector():
    collector = DataCollector()
    collector._cache["600519_2026-07-31"] = {
        "fundamentals": "营收稳健增长 15%",
        "balance_sheet": "资产负债率 12.5%",
        "cashflow": "经营现金流 260 亿元",
        "income_statement": "净利润 400 亿元",
        "news": "公司签订重大战略合作协议",
        "global_news": "全球主要指数稳定",
        "fund_flow_individual": "主力净流入 +5.2 亿元",
        "lhb": "机构席位净买入",
        "indicators": {"vwma": "1520.00"},
        "vpa_indicators": "量增价涨，放量突破阶段高点",
        "stock_data": "2026-07-31 close: 1550.00 vol: 25000",
        "market_data_context": {
            "analysis_baseline_date": "2026-07-31",
            "fund_flow_evidence": {
                "records": [{
                    "source": "eastmoney_direct",
                    "as_of": "2026-07-31",
                    "r0_net": 52000.0,
                    "unit": "万元",
                }],
            },
        },
        "_data_window": "14天",
    }
    return collector


def test_fundamentals_analyst_receives_phase1_reports():
    """基本面分析师 prompt 能读到阶段一产物（宏观/大盘/情绪）。"""
    received_messages = []
    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    sample_verdict = '<!-- VERDICT: {"direction": "看多", "reason": "基本面扎实"} -->'
    sample_response = f"【基本面报告】\n{sample_verdict}"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream

    collector = _make_mock_collector()

    with patch("tradingagents.agents.analysts.fundamentals_analyst.get_cn_stock_name", return_value="贵州茅台"):
        node = create_fundamentals_analyst(mock_llm, data_collector=collector)
        state = {
            "trade_date": "2026-07-31",
            "company_of_interest": "600519",
            "macro_report": "宏观结论：白酒消费预期向好",
            "market_report": "大盘结论：指数震荡筑底",
            "sentiment_report": "情绪结论：市场情绪温和",
        }

        result = asyncio.run(node(state))

        assert "fundamentals_report" in result
        human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
        assert "【阶段一分析师产物（宏观/大盘/情绪）】" in human_msg.content
        assert "白酒消费预期向好" in human_msg.content
        assert "指数震荡筑底" in human_msg.content
        assert "市场情绪温和" in human_msg.content


def test_fundamentals_analyst_missing_phase1_reports_shows_missing():
    """基本面分析师未提供阶段一产物时，prompt 显式标注【数据缺失】。"""
    received_messages = []
    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content='<!-- VERDICT: {"direction": "中性", "reason": "缺数据"} -->')

    mock_llm.astream = _mock_astream
    collector = _make_mock_collector()

    with patch("tradingagents.agents.analysts.fundamentals_analyst.get_cn_stock_name", return_value="贵州茅台"):
        node = create_fundamentals_analyst(mock_llm, data_collector=collector)
        state = {
            "trade_date": "2026-07-31",
            "company_of_interest": "600519",
        }

        asyncio.run(node(state))

        human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
        assert "【数据缺失】宏观板块分析报告缺失" in human_msg.content
        assert "【数据缺失】大盘市场技术分析报告缺失" in human_msg.content
        assert "【数据缺失】市场情绪舆情分析报告缺失" in human_msg.content


def test_news_analyst_receives_phase1_reports():
    """新闻分析师 prompt 能读到阶段一产物。"""
    received_messages = []
    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    sample_verdict = '<!-- VERDICT: {"direction": "看多", "reason": "利好频发"} -->'
    sample_response = f"【新闻报告】\n{sample_verdict}"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream
    collector = _make_mock_collector()

    with patch("tradingagents.agents.analysts.news_analyst.get_cn_stock_name", return_value="贵州茅台"):
        node = create_news_analyst(mock_llm, data_collector=collector)
        state = {
            "trade_date": "2026-07-31",
            "company_of_interest": "600519",
            "macro_report": "宏观结论：货币政策中性偏宽",
            "market_report": "大盘结论：放量突破",
            "sentiment_report": "情绪结论：看多情绪浓厚",
        }

        result = asyncio.run(node(state))

        assert "news_report" in result
        human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
        assert "【阶段一分析师产物（宏观/大盘/情绪）】" in human_msg.content
        assert "货币政策中性偏宽" in human_msg.content
        assert "放量突破" in human_msg.content
        assert "看多情绪浓厚" in human_msg.content


def test_smart_money_analyst_receives_phase1_reports():
    """主力资金分析师 prompt 能读到阶段一产物。"""
    received_messages = []
    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    sample_verdict = '<!-- VERDICT: {"direction": "看多", "reason": "主力资金净流入"} -->'
    sample_response = f"【资金分析报告】\n{sample_verdict}"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream
    collector = _make_mock_collector()

    with patch("tradingagents.agents.analysts.smart_money_analyst.get_cn_stock_name", return_value="贵州茅台"):
        node = create_smart_money_analyst(mock_llm, data_collector=collector)
        state = {
            "trade_date": "2026-07-31",
            "company_of_interest": "600519",
            "macro_report": "宏观结论：流动性充裕",
            "market_report": "大盘结论：量价齐升",
            "sentiment_report": "情绪结论：散户情绪中性",
        }

        result = asyncio.run(node(state))

        assert "smart_money_report" in result
        human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
        assert "【阶段一分析师产物（宏观/大盘/情绪）】" in human_msg.content
        assert "流动性充裕" in human_msg.content
        assert "量价齐升" in human_msg.content
        assert "散户情绪中性" in human_msg.content


def test_volume_price_analyst_receives_phase1_reports():
    """量价分析师 prompt 能读到阶段一产物。"""
    received_messages = []
    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    sample_verdict = '<!-- VERDICT: {"direction": "看多", "reason": "放量突破压力位"} -->'
    sample_response = f"【量价分析报告】\n{sample_verdict}"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content=sample_response)

    mock_llm.astream = _mock_astream
    collector = _make_mock_collector()

    with patch("tradingagents.agents.analysts.volume_price_analyst.get_cn_stock_name", return_value="贵州茅台"):
        node = create_volume_price_analyst(mock_llm, data_collector=collector)
        state = {
            "trade_date": "2026-07-31",
            "company_of_interest": "600519",
            "macro_report": "宏观结论：消费刺激政策落地",
            "market_report": "大盘结论：上证指数上扬",
            "sentiment_report": "情绪结论：市场热情回暖",
        }

        result = asyncio.run(node(state))

        assert "volume_price_report" in result
        human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
        assert "【阶段一分析师产物（宏观/大盘/情绪）】" in human_msg.content
        assert "消费刺激政策落地" in human_msg.content
        assert "上证指数上扬" in human_msg.content
        assert "市场热情回暖" in human_msg.content


def test_volume_price_analyst_missing_phase1_reports_shows_missing():
    """量价分析师未提供阶段一产物时，prompt 显式标注【数据缺失】。"""
    received_messages = []
    mock_llm = MagicMock()
    mock_llm.model_name = "test_model"

    async def _mock_astream(messages):
        received_messages.extend(messages)
        yield SimpleNamespace(content='<!-- VERDICT: {"direction": "中性", "reason": "缺数据"} -->')

    mock_llm.astream = _mock_astream
    collector = _make_mock_collector()

    with patch("tradingagents.agents.analysts.volume_price_analyst.get_cn_stock_name", return_value="贵州茅台"):
        node = create_volume_price_analyst(mock_llm, data_collector=collector)
        state = {
            "trade_date": "2026-07-31",
            "company_of_interest": "600519",
        }

        asyncio.run(node(state))

        human_msg = next(m for m in received_messages if isinstance(m, HumanMessage))
        assert "【数据缺失】宏观板块分析报告缺失" in human_msg.content
        assert "【数据缺失】大盘市场技术分析报告缺失" in human_msg.content
        assert "【数据缺失】市场情绪舆情分析报告缺失" in human_msg.content


# ─────────────────────────────────────────────────────────────────────────────
# 5. 端到端执行顺序测试（真实编译图）
# ─────────────────────────────────────────────────────────────────────────────

def test_real_compiled_graph_execution_order_and_state_visibility():
    """在真实 StateGraph 上验证两阶段拓扑执行：
    1. 阶段一节点并行执行并写回 state
    2. 阶段二节点必须在阶段一全部完成后才启动，并能读取到阶段一写入的报告
    3. Bull Researcher 必须在阶段二全部完成后才启动
    """
    execution_timeline = []

    def make_analyst_mock_node(analyst_name, write_field):
        async def node(state):
            # 记录启动时刻及当时可见的阶段一 state
            seen_macro = state.get("macro_report", "")
            seen_market = state.get("market_report", "")
            seen_sentiment = state.get("sentiment_report", "")
            execution_timeline.append({
                "name": analyst_name,
                "event": "start",
                "seen_macro": seen_macro,
                "seen_market": seen_market,
                "seen_sentiment": seen_sentiment,
            })
            await asyncio.sleep(0.01)
            execution_timeline.append({
                "name": analyst_name,
                "event": "finish",
            })
            return {
                write_field: f"{analyst_name}_output",
                "messages": [AIMessage(content=f"{analyst_name} done")],
            }
        return node

    factories = _make_mock_factories()
    factories["create_macro_analyst"] = lambda *args, **kwargs: make_analyst_mock_node("Macro Analyst", "macro_report")
    factories["create_market_analyst"] = lambda *args, **kwargs: make_analyst_mock_node("Market Analyst", "market_report")
    factories["create_social_media_analyst"] = lambda *args, **kwargs: make_analyst_mock_node("Social Analyst", "sentiment_report")
    factories["create_fundamentals_analyst"] = lambda *args, **kwargs: make_analyst_mock_node("Fundamentals Analyst", "fundamentals_report")
    factories["create_news_analyst"] = lambda *args, **kwargs: make_analyst_mock_node("News Analyst", "news_report")
    factories["create_smart_money_analyst"] = lambda *args, **kwargs: make_analyst_mock_node("Smart Money Analyst", "smart_money_report")
    factories["create_volume_price_analyst"] = lambda *args, **kwargs: make_analyst_mock_node("Volume Price Analyst", "volume_price_report")

    async def mock_bull(state):
        execution_timeline.append({"name": "Bull Researcher", "event": "start"})
        return {
            "messages": [AIMessage(content="bull done")],
            "investment_debate_state": {
                "count": 100,  # 结束辩论，让 should_continue_debate 路由至 Research Manager
                "current_speaker": "Bull",
                "history": "",
            },
            "risk_debate_state": {
                "count": 100,  # 结束风控辩论，让 should_continue_risk_analysis 路由至 Risk Judge
                "latest_speaker": "Aggressive",
                "history": "",
            },
            "risk_feedback_state": {
                "revision_required": False,  # 结束图执行，路由至 END
            },
        }
    factories["create_bull_researcher"] = lambda *args, **kwargs: mock_bull

    setup = _make_graph_setup()

    with patch("tradingagents.graph.setup._load_agent_factories", return_value=factories):
        compiled_graph = setup.setup_graph(
            ["macro", "market", "social", "fundamentals", "news", "smart_money", "volume_price"]
        )

    initial_state = {
        "messages": [HumanMessage(content="start analysis")],
        "company_of_interest": "600519",
        "trade_date": "2026-07-31",
        "macro_report": "",
        "market_report": "",
        "sentiment_report": "",
        "fundamentals_report": "",
        "news_report": "",
        "smart_money_report": "",
        "volume_price_report": "",
        "investment_debate_state": {"count": 0, "current_speaker": "", "history": ""},
        "risk_debate_state": {"count": 0, "latest_speaker": "", "history": ""},
        "risk_feedback_state": {"retry_count": 0, "max_retries": 1, "revision_required": False},
    }

    final_state = asyncio.run(compiled_graph.ainvoke(initial_state))

    # 验证最终报告全量产出
    assert final_state["macro_report"] == "Macro Analyst_output"
    assert final_state["market_report"] == "Market Analyst_output"
    assert final_state["sentiment_report"] == "Social Analyst_output"
    assert final_state["fundamentals_report"] == "Fundamentals Analyst_output"
    assert final_state["news_report"] == "News Analyst_output"
    assert final_state["smart_money_report"] == "Smart Money Analyst_output"
    assert final_state["volume_price_report"] == "Volume Price Analyst_output"

    # 验证时间线顺序：
    # 所有阶段一 finish 必须在阶段二任何 start 之前
    phase1_names = {"Macro Analyst", "Market Analyst", "Social Analyst"}
    phase2_names = {"Fundamentals Analyst", "News Analyst", "Smart Money Analyst", "Volume Price Analyst"}

    phase1_finishes = [
        i for i, item in enumerate(execution_timeline)
        if item["name"] in phase1_names and item["event"] == "finish"
    ]
    phase2_starts = [
        i for i, item in enumerate(execution_timeline)
        if item["name"] in phase2_names and item["event"] == "start"
    ]
    phase2_finishes = [
        i for i, item in enumerate(execution_timeline)
        if item["name"] in phase2_names and item["event"] == "finish"
    ]
    bull_starts = [
        i for i, item in enumerate(execution_timeline)
        if item["name"] == "Bull Researcher" and item["event"] == "start"
    ]

    assert len(phase1_finishes) == 3
    assert len(phase2_starts) == 4
    assert len(phase2_finishes) == 4
    assert len(bull_starts) >= 1

    last_phase1_finish = max(phase1_finishes)
    first_phase2_start = min(phase2_starts)
    last_phase2_finish = max(phase2_finishes)
    first_bull_start = min(bull_starts)

    # 严格顺序：阶段一全完成 < 阶段二启动
    assert last_phase1_finish < first_phase2_start, (
        f"阶段一完成 (index {last_phase1_finish}) 必须在阶段二启动 (index {first_phase2_start}) 之前！"
    )

    # 严格顺序：阶段二全完成 < Bull 辩论启动
    assert last_phase2_finish < first_bull_start, (
        f"阶段二完成 (index {last_phase2_finish}) 必须在 Bull 启动 (index {first_bull_start}) 之前！"
    )

    # 验证阶段二启动时均已看到阶段一写入的 3 份报告
    for item in execution_timeline:
        if item["name"] in phase2_names and item["event"] == "start":
            assert item["seen_macro"] == "Macro Analyst_output", f"{item['name']} 未看到 macro_report"
            assert item["seen_market"] == "Market Analyst_output", f"{item['name']} 未看到 market_report"
            assert item["seen_sentiment"] == "Social Analyst_output", f"{item['name']} 未看到 sentiment_report"
