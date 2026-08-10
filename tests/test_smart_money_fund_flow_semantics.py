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
