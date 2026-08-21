"""针对分析师 Prompt 产业链数据注入 (DAV-201 / DAV-274) 的单元测试。

测试覆盖：
1. `format_industry_linkage_for_prompt` 格式化函数的完整输出、缺失标注与边界兼容；
2. 宏观分析师 (`macro_analyst`) 中产业链数据的正确挂载与 Prompt 注入；
3. 基本面分析师 (`fundamentals_analyst`) 中产业链数据的正确挂载与 Prompt 注入；
4. 未映射股票（如 999999.SH）的安全隔离（不出现产业链数据段落）；
5. 无 DataCollector 时的异步回退路径中的产业链数据获取与注入。
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import HumanMessage

from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
from tradingagents.agents.analysts.macro_analyst import create_macro_analyst
from tradingagents.dataflows.industry_linkage import (
    INDUSTRY_LINKAGE_MAP,
    format_industry_linkage_for_prompt,
)
from tradingagents.graph.data_collector import DataCollector


class TestFormatIndustryLinkageForPrompt:
    """测试 format_industry_linkage_for_prompt 文本格式化逻辑。"""

    def test_empty_or_none_returns_empty_string(self):
        """测试 None、空字典或无效输入返回空字符串。"""
        assert format_industry_linkage_for_prompt(None) == ""
        assert format_industry_linkage_for_prompt({}) == ""
        assert format_industry_linkage_for_prompt({"industry_name": ""}) == ""
        assert format_industry_linkage_for_prompt("invalid_str") == ""  # type: ignore

    def test_consumer_electronics_formatting_with_active_and_missing_data(self):
        """测试消费电子行业数据格式化（包含活跃数据、缺失标注与政策催化）。"""
        payload = {
            "industry_name": "消费电子与智能终端",
            "upstream_cost": [
                {
                    "name": "LME铜价",
                    "current_value": 14008.5,
                    "unit": "美元/吨",
                    "mom_change": 0.74,
                    "qoq_change": 1.20,
                    "trend": "平稳",
                    "confidence": "高",
                    "transmission_logic": "核心导电、引线框架与连接件原材料成本传导",
                }
            ],
            "downstream_demand": [
                {
                    "name": "全球智能手机出货量",
                    "current_value": None,
                    "unit": "万部",
                    "trend": "数据缺失",
                    "confidence": "低（待手动录入）",
                    "note": "手动",
                    "transmission_logic": "下游终端消费电子需求与换机周期景气度验证",
                }
            ],
            "international_benchmark": [
                {
                    "name": "三星电子股价",
                    "current_value": 52000.0,
                    "unit": "韩元",
                    "mom_change": -1.50,
                    "qoq_change": -3.20,
                    "trend": "下降",
                    "confidence": "高",
                    "transmission_logic": "全球消费电子、存储半导体与显示面板龙头估值与景气度对标",
                }
            ],
            "policy_catalysts": [
                "消费品以旧换新补贴政策",
                "超高清视频产业发展规划",
            ],
        }

        formatted = format_industry_linkage_for_prompt(payload)

        # 核心段落标题
        assert "【产业链联想数据】：消费电子与智能终端" in formatted
        # 上游成本指标
        assert "- 上游成本端核心指标：" in formatted
        assert "LME铜价：14008.50 美元/吨" in formatted
        assert "月环比 +0.74%" in formatted
        assert "季度环比 +1.20%" in formatted
        assert "趋势：平稳" in formatted
        assert "传导逻辑：核心导电、引线框架与连接件原材料成本传导" in formatted
        # 下游缺失需求指标
        assert "- 下游需求端核心指标：" in formatted
        assert "【数据缺失】全球智能手机出货量：手动" in formatted
        # 国际对标
        assert "- 国际对标核心标的/指标：" in formatted
        assert "三星电子股价：52000.00 韩元" in formatted
        assert "月环比 -1.50%" in formatted
        assert "趋势：下降" in formatted
        # 政策催化
        assert "- 行业政策催化关键词：消费品以旧换新补贴政策、超高清视频产业发展规划" in formatted

    def test_new_energy_vehicle_formatting_with_pending_api(self):
        """测试新能源车行业数据格式化（碳酸锂待接入 API 标注）。"""
        payload = {
            "industry_name": "新能源汽车与智能汽车",
            "upstream_cost": [
                {
                    "name": "碳酸锂价格",
                    "current_value": None,
                    "unit": "万元/吨",
                    "trend": "数据缺失",
                    "confidence": "低（待接入API）",
                    "note": "待接入API",
                    "transmission_logic": "动力电池正极核心原材料成本传导",
                }
            ],
            "downstream_demand": [
                {
                    "name": "新能源车渗透率",
                    "current_value": None,
                    "trend": "数据缺失",
                    "confidence": "低（待手动录入）",
                    "note": "手动",
                    "transmission_logic": "终端新能源汽车市场渗透水平与消费端销量景气度",
                }
            ],
            "international_benchmark": [
                {
                    "name": "特斯拉交付量",
                    "current_value": None,
                    "trend": "数据缺失",
                    "confidence": "低（待手动录入）",
                    "note": "手动",
                    "transmission_logic": "全球新能源汽车领军企业产销与需求风向标",
                }
            ],
            "policy_catalysts": [
                "新能源汽车购置税减免",
                "车路云一体化试点",
            ],
        }

        formatted = format_industry_linkage_for_prompt(payload)

        assert "【产业链联想数据】：新能源汽车与智能汽车" in formatted
        assert "【数据缺失】碳酸锂价格：待接入API" in formatted
        assert "【数据缺失】新能源车渗透率：手动" in formatted
        assert "【数据缺失】特斯拉交付量：手动" in formatted
        assert "行业政策催化关键词：新能源汽车购置税减免、车路云一体化试点" in formatted

    def test_pydantic_model_input_direct_support(self):
        """测试直接传入 IndustryLinkage Pydantic 模型对象。"""
        model = INDUSTRY_LINKAGE_MAP["消费电子"]
        formatted = format_industry_linkage_for_prompt(model)
        assert "【产业链联想数据】：消费电子与智能终端" in formatted
        assert "LME铜价" in formatted
        assert "三星电子股价" in formatted


class TestAnalystPromptInjectionIntegration:
    """测试宏观分析师与基本面分析师 Prompt 注入产业链数据。"""

    def _create_mock_llm(self, sample_verdict):
        mock_llm = MagicMock()
        mock_llm.model_name = "test_llm"
        mock_llm.model = "test_llm"

        received = []

        async def _mock_astream(messages):
            received.extend(messages)
            yield SimpleNamespace(content=f"分析报告正文\n{sample_verdict}")

        mock_llm.astream = _mock_astream
        return mock_llm, received

    def test_macro_analyst_injects_consumer_electronics_linkage(self):
        """测试宏观分析师为京东方A (000725.SZ) 注入消费电子产业链数据。"""
        sample_verdict = '<!-- VERDICT: {"direction": "看多", "reason": "产业链成本平稳"} -->'
        mock_llm, received = self._create_mock_llm(sample_verdict)

        collector = DataCollector()
        collector._cache["000725.SZ_2026-08-20"] = {
            "fund_flow_board": "电子板块主力净流入 +5亿",
            "news": "京东方A 柔性OLED面板出货创历史新高",
            "global_news": "全球消费电子旺季备货开启",
            "global_indices": "标普500: +0.5%",
            "major_assets": "LME铜: 9123美元/吨",
            "cn_indices": "沪深300: +0.8%",
            "northbound_flow": "北向净买入 +5000万",
            "industry_linkage": {
                "industry_name": "消费电子与智能终端",
                "upstream_cost": [{
                    "name": "LME铜价", "current_value": 9123.5, "unit": "美元/吨",
                    "mom_change": 2.3, "trend": "上升", "confidence": "高",
                    "transmission_logic": "原材料成本传导",
                }],
                "downstream_demand": [{
                    "name": "全球智能手机出货量", "current_value": None,
                    "trend": "数据缺失", "note": "手动",
                }],
                "international_benchmark": [{
                    "name": "三星电子股价", "current_value": 52000.0, "unit": "韩元",
                    "mom_change": -1.5, "trend": "下降", "confidence": "高",
                }],
                "policy_catalysts": ["消费品以旧换新补贴政策"],
            },
        }

        node = create_macro_analyst(mock_llm, collector)
        state = {
            "trade_date": "2026-08-20",
            "company_of_interest": "000725.SZ",
            "user_intent": {},
        }

        result = asyncio.run(node(state))
        assert "macro_report" in result

        human_msg = next(m for m in received if isinstance(m, HumanMessage))
        prompt_content = human_msg.content

        # 核心断言：Prompt 中成功注入产业链数据
        assert "【产业链联想数据】：消费电子与智能终端" in prompt_content
        assert "LME铜价：9123.50 美元/吨" in prompt_content
        assert "月环比 +2.30%" in prompt_content
        assert "三星电子股价：52000.00 韩元" in prompt_content
        assert "消费品以旧换新补贴政策" in prompt_content

    def test_fundamentals_analyst_injects_new_energy_linkage(self):
        """测试基本面分析师为宁德时代 (300750.SZ) 注入新能源车产业链数据。"""
        sample_verdict = '<!-- VERDICT: {"direction": "看多", "reason": "基本面稳健"} -->'
        mock_llm, received = self._create_mock_llm(sample_verdict)

        collector = DataCollector()
        collector._cache["300750.SZ_2026-08-20"] = {
            "fundamentals": "营业收入 1000亿，同比增长 25%",
            "balance_sheet": "资产负债率 55%",
            "cashflow": "经营活动现金流净额 150亿",
            "income_statement": "净利润 120亿",
            "global_indices": "无数据",
            "major_assets": "无数据",
            "cn_indices": "无数据",
            "industry_linkage": {
                "industry_name": "新能源汽车与智能汽车",
                "upstream_cost": [{
                    "name": "碳酸锂价格", "current_value": None,
                    "trend": "数据缺失", "note": "待接入API",
                    "transmission_logic": "动力电池正极核心原材料成本传导",
                }],
                "downstream_demand": [{
                    "name": "新能源车渗透率", "current_value": None,
                    "trend": "数据缺失", "note": "手动",
                }],
                "international_benchmark": [{
                    "name": "特斯拉交付量", "current_value": None,
                    "trend": "数据缺失", "note": "手动",
                }],
                "policy_catalysts": ["新能源汽车购置税减免"],
            },
        }

        node = create_fundamentals_analyst(mock_llm, collector)
        state = {
            "trade_date": "2026-08-20",
            "company_of_interest": "300750.SZ",
            "user_intent": {},
        }

        result = asyncio.run(node(state))
        assert "fundamentals_report" in result

        human_msg = next(m for m in received if isinstance(m, HumanMessage))
        prompt_content = human_msg.content

        # 核心断言：Prompt 中成功注入新能源车产业链数据及缺失标注
        assert "【产业链联想数据】：新能源汽车与智能汽车" in prompt_content
        assert "【数据缺失】碳酸锂价格：待接入API" in prompt_content
        assert "动力电池正极核心原材料成本传导" in prompt_content
        assert "新能源汽车购置税减免" in prompt_content

    def test_analyst_unmapped_stock_does_not_inject_linkage(self):
        """测试未映射股票（如 999999.SH）不注入产业链联想数据段落。"""
        sample_verdict = '<!-- VERDICT: {"direction": "中性", "reason": "业务稳健"} -->'
        mock_llm, received = self._create_mock_llm(sample_verdict)

        collector = DataCollector()
        collector._cache["999999.SH_2026-08-20"] = {
            "fund_flow_board": "板块流向平稳",
            "news": "发布半年度业绩快报",
            "global_news": "无数据",
            "global_indices": "无数据",
            "major_assets": "无数据",
            "cn_indices": "无数据",
            "northbound_flow": "无数据",
            "industry_linkage": None,
        }

        node = create_macro_analyst(mock_llm, collector)
        state = {
            "trade_date": "2026-08-20",
            "company_of_interest": "999999.SH",
            "user_intent": {},
        }

        asyncio.run(node(state))

        human_msg = next(m for m in received if isinstance(m, HumanMessage))
        assert "【产业链联想数据】" not in human_msg.content

    def test_analyst_fallback_mode_fetches_and_injects_linkage(self):
        """测试在无 DataCollector 时的 fallback 回退分支能正确获取并注入产业链数据。"""
        sample_verdict = '<!-- VERDICT: {"direction": "偏多", "reason": "回退模式下产业链数据注入成功"} -->'
        mock_llm, received = self._create_mock_llm(sample_verdict)

        with patch("tradingagents.agents.analysts.macro_analyst.get_cn_stock_name", return_value="京东方A"), \
             patch("tradingagents.agents.utils.agent_utils.get_board_fund_flow") as mock_board, \
             patch("tradingagents.agents.utils.agent_utils.get_news") as mock_news, \
             patch("tradingagents.agents.utils.agent_utils.get_global_news") as mock_gnews, \
             patch("tradingagents.agents.utils.agent_utils.get_northbound_flow") as mock_nb, \
             patch("tradingagents.agents.utils.agent_utils.get_global_indices", create=True) as mock_gidx, \
             patch("tradingagents.agents.utils.agent_utils.get_major_assets", create=True) as mock_masset, \
             patch("tradingagents.agents.utils.agent_utils.get_cn_indices", create=True) as mock_cnidx, \
             patch("tradingagents.dataflows.providers.industry_linkage_provider.IndustryLinkageProvider.get_industry_linkage") as mock_linkage:

            mock_board.invoke.return_value = "资金流"
            mock_news.invoke.return_value = "新闻"
            mock_gnews.invoke.return_value = "宏观新闻"
            mock_nb.invoke.return_value = "北向"
            mock_gidx.invoke.return_value = "全球指数"
            mock_masset.invoke.return_value = "大类资产"
            mock_cnidx.invoke.return_value = "国内指数"
            mock_linkage.return_value = {
                "industry_name": "消费电子与智能终端",
                "upstream_cost": [{"name": "LME铜价", "current_value": 9123.5, "unit": "美元/吨", "trend": "平稳"}],
            }

            node = create_macro_analyst(mock_llm, data_collector=None)
            state = {
                "trade_date": "2026-08-20",
                "company_of_interest": "000725.SZ",
            }

            asyncio.run(node(state))

            human_msg = next(m for m in received if isinstance(m, HumanMessage))
            assert "【产业链联想数据】：消费电子与智能终端" in human_msg.content
            assert "LME铜价：9123.50 美元/吨" in human_msg.content
