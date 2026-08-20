import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from tradingagents.agents.analysts.smart_money_analyst import create_smart_money_analyst


class _RecordingLLM:
    def __init__(self):
        self.messages = None

    async def astream(self, messages):
        self.messages = messages
        yield SimpleNamespace(content="固定分析输出")


class _FundFlowCollector:
    def get(self, ticker, curr_date):
        assert ticker == "600519"
        assert curr_date == "2026-08-10"
        return {
            "fund_flow_individual": (
                "【备用数据源：同花顺即时资金流净额快照】600519 当日资金流净额快照\n"
                "资金净额: 5.60亿\n"
                "（该快照不是新浪历史 netamount/r0_net 同口径主力序列）"
            ),
            "lhb": "无龙虎榜数据",
            "indicators": {"vwma": "100"},
        }


def test_smart_money_prompt_preserves_fund_flow_source_semantics():
    llm = _RecordingLLM()
    module = __import__(
        "tradingagents.agents.analysts.smart_money_analyst",
        fromlist=["smart_money_analyst"],
    )
    state = {
        "trade_date": "2026-08-10",
        "company_of_interest": "600519",
        "user_intent": {"focus_areas": [], "specific_questions": []},
    }

    with (
        patch.object(module, "get_cn_stock_name", return_value="测试股票"),
        patch.object(module, "get_config", return_value={}),
        patch.object(module, "get_prompt", return_value="固定系统提示"),
        patch.object(module, "build_horizon_context", return_value="固定上下文"),
        patch.object(module, "log_llm_call"),
    ):
        result = asyncio.run(
            create_smart_money_analyst(llm, _FundFlowCollector())(state)
        )

    assert "smart_money_report" in result
    human_prompt = llm.messages[1].content
    assert "【资金流数据（来源、日期与口径见数据）】" in human_prompt
    assert "不得将其视为新浪历史 netamount/r0_net 同口径的主力序列" in human_prompt
    assert "【近5日主力资金净流向】" not in human_prompt


class _YieldingLLM:
    def __init__(self, content: str):
        self.content = content

    async def astream(self, messages):
        yield SimpleNamespace(content=self.content)


class _StructuredTHSCollector:
    def get(self, ticker, curr_date):
        record = {
            "source": "ths_instant_snapshot",
            "source_family": "ths",
            "algorithm_group": "new_algorithm_group",
            "status": "available",
            "symbol": ticker,
            "date": curr_date,
            "period_kind": "realtime_single_day",
            "time_window": "1d",
            "field": "netamount",
            "value": "5.60",
            "unit": "亿元",
            "field_semantics": {"netamount": "总净额（负值表示净流出）"},
        }
        return {
            "fund_flow_individual": "同花顺即时资金流净额 5.60 亿",
            "market_data_context": {
                "fund_flow_evidence": {
                    "records": [record],
                    "symbol": ticker,
                    "requested_as_of": curr_date,
                }
            },
            "lhb": "无龙虎榜数据",
            "indicators": {"vwma": "100"},
        }


def test_smart_money_ths_netamount_allowed_for_total_flow_direction():
    llm = _YieldingLLM("全市场总资金偏流入，资金面整体稳定。")
    module = __import__(
        "tradingagents.agents.analysts.smart_money_analyst",
        fromlist=["smart_money_analyst"],
    )
    state = {
        "trade_date": "2026-08-10",
        "company_of_interest": "600519",
        "user_intent": {"focus_areas": [], "specific_questions": []},
    }

    with (
        patch.object(module, "get_cn_stock_name", return_value="测试股票"),
        patch.object(module, "get_config", return_value={}),
        patch.object(module, "get_prompt", return_value="固定系统提示"),
        patch.object(module, "build_horizon_context", return_value="固定上下文"),
        patch.object(module, "log_llm_call"),
    ):
        result = asyncio.run(
            create_smart_money_analyst(llm, _StructuredTHSCollector())(state)
        )

    assert "全市场总资金偏流入" in result["smart_money_report"]
    guard = result["fund_flow_consensus_guard"]
    assert guard["blocked"] is False
    assert guard["direction_allowed"] is True
    assert guard["selected_field"] == "netamount"


def test_smart_money_ths_netamount_blocks_main_force_accumulation_claims():
    llm = _YieldingLLM("资金面显示主力资金积极吸筹建仓，主力大幅增持。")
    module = __import__(
        "tradingagents.agents.analysts.smart_money_analyst",
        fromlist=["smart_money_analyst"],
    )
    state = {
        "trade_date": "2026-08-10",
        "company_of_interest": "600519",
        "user_intent": {"focus_areas": [], "specific_questions": []},
    }

    with (
        patch.object(module, "get_cn_stock_name", return_value="测试股票"),
        patch.object(module, "get_config", return_value={}),
        patch.object(module, "get_prompt", return_value="固定系统提示"),
        patch.object(module, "build_horizon_context", return_value="固定上下文"),
        patch.object(module, "log_llm_call"),
    ):
        result = asyncio.run(
            create_smart_money_analyst(llm, _StructuredTHSCollector())(state)
        )

    guard = result["fund_flow_consensus_guard"]
    assert guard["blocked"] is True
    assert guard["direction_allowed"] is False
    assert "已阻断增持、减持、吸筹方向摘要" in result["smart_money_report"]

