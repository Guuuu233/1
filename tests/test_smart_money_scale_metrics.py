"""Unit tests for smart money analyst consumption of scale_metrics (D-03-3).

Covers 10 acceptance criteria:
1. Both ratios available: captures prompt sent to fake LLM and verifies numerical text,
   date, target ticker, source, and units.
2. net_to_circ_mv == 0 and net_to_amount == 0 are presented as valid zero values.
3. partial status: only available ratio presented, missing ratio not manufactured, gap preserved.
4. unavailable status: outputs explicit discipline "相对规模不可用/不得据绝对净额替代", no fallback to net amount.
5. Completely missing scale_metrics fails closed with explicit unavailable discipline.
6. selected_algorithm_group and reference_only read only from selection; fake fields in scale_metrics not trusted.
7. Prompt discipline strictly forbids account identity inference, cross-stock ranking, new scores/weights/probabilities/signals.
8. No new provider calls; existing direction/validation guard and SSE delayed semantics preserved.
9. Compliance negative statements in LLM output are not killed by naive keyword scanning.
10. Input objects are not modified in-place, and node outputs are strictly JSON-serializable.
"""

import asyncio
import copy
from decimal import Decimal
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tradingagents.agents.analysts.smart_money_analyst import create_smart_money_analyst


@pytest.fixture(autouse=True)
def guard_no_network_calls():
    """Ensure no real network calls can be made in this test suite."""
    with patch(
        "socket.socket.connect",
        side_effect=RuntimeError("Network access forbidden in offline tests"),
    ), patch(
        "requests.post",
        side_effect=RuntimeError("requests.post forbidden in offline tests"),
    ), patch(
        "requests.get",
        side_effect=RuntimeError("requests.get forbidden in offline tests"),
    ):
        yield


class _RecordingLLM:
    def __init__(self, content: str = "固定分析输出"):
        self.messages = None
        self.content = content

    async def astream(self, messages):
        self.messages = messages
        yield SimpleNamespace(content=self.content)

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.content)


class _MockCollector:
    def __init__(self, fund_flow_evidence: dict | None = None, market_data_context: dict | None = None):
        self.fund_flow_evidence = fund_flow_evidence or {}
        self.market_data_context = market_data_context

    def get(self, ticker: str, curr_date: str):
        mdc = self.market_data_context or {
            "fund_flow_evidence": self.fund_flow_evidence,
        }
        return {
            "fund_flow_individual": "同花顺即时资金流净额 5.60 亿",
            "market_data_context": mdc,
            "lhb": "无龙虎榜数据",
            "indicators": {"vwma": "100"},
        }


def _run_analyst_node(llm, collector, state=None):
    module = __import__(
        "tradingagents.agents.analysts.smart_money_analyst",
        fromlist=["smart_money_analyst"],
    )
    if state is None:
        state = {
            "trade_date": "2026-08-14",
            "company_of_interest": "600519",
            "user_intent": {"focus_areas": [], "specific_questions": []},
        }

    with (
        patch.object(module, "get_cn_stock_name", return_value="贵州茅台"),
        patch.object(module, "get_config", return_value={}),
        patch.object(module, "get_prompt", return_value="固定系统提示"),
        patch.object(module, "build_horizon_context", return_value="固定上下文"),
        patch.object(module, "log_llm_call"),
    ):
        return asyncio.run(create_smart_money_analyst(llm, collector)(state))


def test_both_ratios_available_prompt_contains_exact_metrics():
    """1. 两个比率均可用，捕获发给 fake LLM 的消息并核对数值文本、日期、标的、来源与单位。"""
    scale_metrics = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "net_amount": Decimal("15000.0"),
        "net_amount_raw": "15000.0",
        "net_amount_unit": "万元",
        "unit": "万元",
        "net_to_circ_mv": Decimal("0.006755"),
        "net_to_circ_mv_text": "0.006755",
        "net_to_amount": Decimal("0.027273"),
        "net_to_amount_text": "0.027273",
        "circ_mv": Decimal("2220600.0"),
        "circ_mv_unit": "万元",
        "circ_mv_source": "tushare.daily_basic",
        "amount": Decimal("550000.0"),
        "amount_unit": "万元",
        "amount_source": "tushare.daily_basic",
        "denominator_source": "tushare.daily_basic",
        "denominator_sources": {
            "circ_mv": "tushare.daily_basic",
            "amount": "tushare.daily_basic",
        },
        "denominator_units": {
            "circ_mv": "万元",
            "amount": "万元",
        },
        "status": "available",
        "gaps": [],
        "gap_list": [],
    }
    selection = {
        "selected_source": "tushare_eastmoney_moneyflow_dc",
        "selected_algorithm_group": "new_algorithm_group",
        "reference_only": True,
    }
    fund_flow_evidence = {
        "scale_metrics": scale_metrics,
        "selection": selection,
        "records": [],
    }

    llm = _RecordingLLM()
    collector = _MockCollector(fund_flow_evidence=fund_flow_evidence)
    result = _run_analyst_node(llm, collector)

    assert "smart_money_report" in result
    human_prompt = llm.messages[1].content

    # 核对节标题
    assert "相对规模证据" in human_prompt
    # 核对标的与日期
    assert "600519.SH" in human_prompt
    assert "2026-08-14" in human_prompt
    # 核对比率数值文本（不得修改精度）
    assert "0.006755" in human_prompt
    assert "0.027273" in human_prompt
    # 核对来源与单位
    assert "tushare.daily_basic" in human_prompt
    assert "万元" in human_prompt
    # 核对状态
    assert "available" in human_prompt


def test_zero_ratios_presented_as_valid_zero():
    """2. net_to_circ_mv == 0 与 net_to_amount == 0 仍被呈现为有效零值。"""
    scale_metrics = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "net_amount": Decimal("0"),
        "net_amount_raw": "0",
        "net_amount_unit": "万元",
        "unit": "万元",
        "net_to_circ_mv": Decimal("0"),
        "net_to_circ_mv_text": "0",
        "net_to_amount": Decimal("0"),
        "net_to_amount_text": "0",
        "circ_mv": Decimal("2220600.0"),
        "circ_mv_unit": "万元",
        "circ_mv_source": "tushare.daily_basic",
        "amount": Decimal("550000.0"),
        "amount_unit": "万元",
        "amount_source": "tushare.daily_basic",
        "denominator_source": "tushare.daily_basic",
        "status": "available",
        "gaps": [],
        "gap_list": [],
    }
    selection = {
        "selected_source": "tushare_eastmoney_moneyflow_dc",
        "selected_algorithm_group": "new_algorithm_group",
        "reference_only": True,
    }
    fund_flow_evidence = {
        "scale_metrics": scale_metrics,
        "selection": selection,
        "records": [],
    }

    llm = _RecordingLLM()
    collector = _MockCollector(fund_flow_evidence=fund_flow_evidence)
    result = _run_analyst_node(llm, collector)

    assert "smart_money_report" in result
    human_prompt = llm.messages[1].content

    # 零值必须被正常呈现，绝不得被判定为缺少或 None
    assert "net_to_circ_mv" in human_prompt
    assert "net_to_amount" in human_prompt
    assert " 0" in human_prompt or ": 0" in human_prompt
    assert "available" in human_prompt


def test_partial_preserves_single_ratio_and_gaps():
    """3. 只有一个比率可用的 partial 不制造另一个比率，并保留对应 gap。"""
    scale_metrics = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "net_amount": Decimal("15000.0"),
        "net_amount_raw": "15000.0",
        "net_amount_unit": "万元",
        "unit": "万元",
        "net_to_circ_mv": Decimal("0.006755"),
        "net_to_circ_mv_text": "0.006755",
        "net_to_amount": None,
        "net_to_amount_text": None,
        "circ_mv": Decimal("2220600.0"),
        "circ_mv_unit": "万元",
        "circ_mv_source": "tushare.daily_basic",
        "amount": None,
        "amount_unit": None,
        "amount_source": None,
        "denominator_source": "tushare.daily_basic",
        "status": "partial",
        "gaps": ["成交额单位 (amount_unit) 缺失，不得默认单位，成交额占比拒算"],
        "gap_list": ["成交额单位 (amount_unit) 缺失，不得默认单位，成交额占比拒算"],
    }
    selection = {
        "selected_source": "tushare_eastmoney_moneyflow_dc",
        "selected_algorithm_group": "new_algorithm_group",
        "reference_only": True,
    }
    fund_flow_evidence = {
        "scale_metrics": scale_metrics,
        "selection": selection,
        "records": [],
    }

    llm = _RecordingLLM()
    collector = _MockCollector(fund_flow_evidence=fund_flow_evidence)
    result = _run_analyst_node(llm, collector)

    assert "smart_money_report" in result
    human_prompt = llm.messages[1].content

    # 呈现可用比率
    assert "0.006755" in human_prompt
    # 状态为 partial
    assert "partial" in human_prompt
    # 缺失比率保留缺口
    assert "成交额单位 (amount_unit) 缺失" in human_prompt
    # 不制造另一个比率
    assert "net_to_amount" in human_prompt
    assert "不可用" in human_prompt or "缺失" in human_prompt or "None" in human_prompt


def test_unavailable_outputs_explicit_discipline_no_fallback():
    """4. unavailable 明确输出不可用纪律，不回退为绝对净额规模结论。"""
    scale_metrics = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "net_amount": Decimal("1.5"),
        "net_amount_raw": "1.5",
        "net_amount_unit": "亿元",
        "unit": "亿元",
        "net_to_circ_mv": None,
        "net_to_circ_mv_text": None,
        "net_to_amount": None,
        "net_to_amount_text": None,
        "circ_mv": None,
        "circ_mv_unit": None,
        "circ_mv_source": None,
        "amount": None,
        "amount_unit": None,
        "amount_source": None,
        "denominator_source": None,
        "status": "unavailable",
        "gaps": ["流通市值分母缺失", "成交额分母缺失"],
        "gap_list": ["流通市值分母缺失", "成交额分母缺失"],
    }
    fund_flow_evidence = {
        "scale_metrics": scale_metrics,
        "selection": {
            "selected_source": "tushare_eastmoney_moneyflow_dc",
            "selected_algorithm_group": "new_algorithm_group",
            "reference_only": True,
        },
        "records": [],
    }

    llm = _RecordingLLM()
    collector = _MockCollector(fund_flow_evidence=fund_flow_evidence)
    result = _run_analyst_node(llm, collector)

    assert "smart_money_report" in result
    human_prompt = llm.messages[1].content

    # unavailable 必须明确写“相对规模不可用/不得据绝对净额替代”
    assert "相对规模不可用" in human_prompt
    assert "不得据绝对净额替代" in human_prompt
    assert "流通市值分母缺失" in human_prompt


def test_missing_scale_metrics_fails_closed():
    """5. 完全缺少 scale_metrics 时同样显式 fail-closed。"""
    fund_flow_evidence = {
        "selection": {
            "selected_source": "tushare_eastmoney_moneyflow_dc",
            "selected_algorithm_group": "new_algorithm_group",
            "reference_only": True,
        },
        "records": [],
    }

    llm = _RecordingLLM()
    collector = _MockCollector(fund_flow_evidence=fund_flow_evidence)
    result = _run_analyst_node(llm, collector)

    assert "smart_money_report" in result
    human_prompt = llm.messages[1].content

    # 完全缺少 scale_metrics 时显式输出不可用纪律，而不是静默省略
    assert "相对规模证据" in human_prompt
    assert "相对规模不可用" in human_prompt
    assert "不得据绝对净额替代" in human_prompt


def test_algorithm_group_and_reference_only_read_only_from_selection():
    """6. selected_algorithm_group / reference_only 只从 selection 读取；scale 对象中的同名伪字段不得被信任。"""
    scale_metrics = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "net_to_circ_mv": Decimal("0.006755"),
        "net_to_circ_mv_text": "0.006755",
        "net_to_amount": Decimal("0.027273"),
        "net_to_amount_text": "0.027273",
        "status": "available",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "amount_source": "tushare.daily_basic",
        "amount_unit": "万元",
        "gaps": [],
        # 故意注入伪字段到 scale_metrics，契约规定 scale_metrics 没有这些字段
        "algorithm_group": "fake_scale_algo_group",
        "selected_algorithm_group": "fake_scale_algo_group",
        "reference_only": False,
    }
    selection = {
        "selected_source": "tushare_eastmoney_moneyflow_dc",
        "selected_algorithm_group": "real_selection_algo_group",
        "reference_only": True,
    }
    fund_flow_evidence = {
        "scale_metrics": scale_metrics,
        "selection": selection,
        "records": [],
    }

    llm = _RecordingLLM()
    collector = _MockCollector(fund_flow_evidence=fund_flow_evidence)
    result = _run_analyst_node(llm, collector)

    human_prompt = llm.messages[1].content

    # 检查整个 HumanMessage：真实算法组来自 selection，伪字段与伪 reference_only 绝不存在
    assert "real_selection_algo_group" in human_prompt
    assert "fake_scale_algo_group" not in human_prompt
    assert "reference_only=True" in human_prompt
    assert "reference_only=False" not in human_prompt

    # 直接验证纯函数 format_fund_flow_scale_metrics_prompt 的契约行为
    from tradingagents.agents.analysts.smart_money_analyst import format_fund_flow_scale_metrics_prompt
    pure_prompt = format_fund_flow_scale_metrics_prompt(scale_metrics, selection)
    assert "real_selection_algo_group" in pure_prompt
    assert "fake_scale_algo_group" not in pure_prompt
    assert "reference_only=True" in pure_prompt
    assert "reference_only=False" not in pure_prompt

    # 当 selection 中无算法组时，即便 scale_metrics 中有伪字段，也绝不读取伪字段
    pure_prompt_no_sel = format_fund_flow_scale_metrics_prompt(scale_metrics, {})
    assert "fake_scale_algo_group" not in pure_prompt_no_sel
    assert "未指定" in pure_prompt_no_sel


def test_prompt_discipline_forbids_account_identity_ranking_and_signals():
    """7. 消息纪律明确禁止账户身份、跨股票排名、新评分/权重/概率/执行信号。"""
    scale_metrics = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "net_to_circ_mv": Decimal("0.006755"),
        "net_to_circ_mv_text": "0.006755",
        "net_to_amount": Decimal("0.027273"),
        "net_to_amount_text": "0.027273",
        "status": "available",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "amount_source": "tushare.daily_basic",
        "amount_unit": "万元",
        "gaps": [],
    }
    fund_flow_evidence = {
        "scale_metrics": scale_metrics,
        "selection": {
            "selected_source": "tushare_eastmoney_moneyflow_dc",
            "selected_algorithm_group": "new_algorithm_group",
            "reference_only": True,
        },
        "records": [],
    }

    llm = _RecordingLLM()
    collector = _MockCollector(fund_flow_evidence=fund_flow_evidence)
    result = _run_analyst_node(llm, collector)

    human_prompt = llm.messages[1].content

    # 4条纪律约束必须在提示词中显式存在
    assert "同标的" in human_prompt and "统计参考证据" in human_prompt
    assert "账户" in human_prompt and ("不得" in human_prompt or "严禁" in human_prompt)
    assert "跨股票排名" in human_prompt or "横向排名" in human_prompt
    assert "评分" in human_prompt or "权重" in human_prompt or "执行信号" in human_prompt


def test_no_provider_calls_and_existing_guard_behavior_preserved():
    """8. 不新增 provider 调用，既有 direction/validation guard 行为与 SSE 延后语义不回归。"""
    record = {
        "source": "ths_instant_snapshot",
        "source_family": "ths",
        "algorithm_group": "new_algorithm_group",
        "status": "available",
        "symbol": "600519",
        "date": "2026-08-14",
        "period_kind": "realtime_single_day",
        "time_window": "1d",
        "field": "netamount",
        "value": "5.60",
        "unit": "亿元",
        "field_semantics": {"netamount": "总净额（负值表示净流出）"},
    }
    scale_metrics = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "net_to_circ_mv": Decimal("0.006755"),
        "net_to_circ_mv_text": "0.006755",
        "net_to_amount": Decimal("0.027273"),
        "net_to_amount_text": "0.027273",
        "status": "available",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "amount_source": "tushare.daily_basic",
        "amount_unit": "万元",
        "gaps": [],
    }
    fund_flow_evidence = {
        "scale_metrics": scale_metrics,
        "records": [record],
        "symbol": "600519",
        "requested_as_of": "2026-08-14",
    }

    # 当模型输出违规主力建仓词汇时，既有 direction guard 必须生效阻断
    llm_violation = _RecordingLLM("主力资金积极吸筹建仓，主力大幅增持。")
    collector = _MockCollector(fund_flow_evidence=fund_flow_evidence)
    result = _run_analyst_node(llm_violation, collector)

    guard = result["fund_flow_consensus_guard"]
    assert guard["blocked"] is True
    assert guard["direction_allowed"] is False
    assert "已阻断增持、减持、吸筹方向摘要" in result["smart_money_report"]

    # 当模型合规总资金描述时，不阻断
    llm_valid = _RecordingLLM("全市场总资金偏流入，资金面整体稳定。")
    result_valid = _run_analyst_node(llm_valid, collector)
    guard_valid = result_valid["fund_flow_consensus_guard"]
    assert guard_valid["blocked"] is False
    assert guard_valid["direction_allowed"] is True


def test_compliance_negative_sentences_not_blocked():
    """9. 测试合规否定句，证明未引入朴素关键词误杀。"""
    record = {
        "source": "tushare_eastmoney_moneyflow_dc",
        "source_family": "eastmoney",
        "algorithm_group": "new_algorithm_group",
        "status": "available",
        "symbol": "600036",
        "date": "2026-08-20",
        "period_kind": "historical_daily",
        "time_window": "1d",
        "field": "r0_net",
        "value": "2.211971",
        "unit": "亿元",
        "field_semantics": {"r0_net": "主力净额（负值表示净流出）"},
    }
    scale_metrics = {
        "ts_code": "600036.SH",
        "trade_date": "2026-08-20",
        "net_to_circ_mv": Decimal("0.002"),
        "net_to_circ_mv_text": "0.002",
        "net_to_amount": Decimal("0.05"),
        "net_to_amount_text": "0.05",
        "status": "available",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "amount_source": "tushare.daily_basic",
        "amount_unit": "万元",
        "gaps": [],
    }
    fund_flow_evidence = {
        "scale_metrics": scale_metrics,
        "records": [record],
        "symbol": "600036",
        "requested_as_of": "2026-08-20",
    }

    # 合规否定句：包含“机构账户”、“排名”、“评分”、“交易信号”等词，但为否定句免责声明
    compliance_report = (
        "当日主力资金净额为 2.21 亿，主力偏增持。\n"
        "【合规声明】本相对规模数据仅供同标的同日统计参考，不得据此识别机构或散户账户身份，"
        "亦不得用于跨股票横向排名，不生成任何评分、权重或交易执行信号。"
    )
    llm = _RecordingLLM(compliance_report)
    collector = _MockCollector(fund_flow_evidence=fund_flow_evidence)
    state = {
        "trade_date": "2026-08-20",
        "company_of_interest": "600036",
        "user_intent": {"focus_areas": [], "specific_questions": []},
    }
    result = _run_analyst_node(llm, collector, state=state)

    # 验证未被误杀阻断
    guard = result["fund_flow_consensus_guard"]
    assert guard["blocked"] is False
    assert guard["direction_allowed"] is True
    assert "【合规声明】" in result["smart_money_report"]


def test_immutability_and_json_serializability():
    """10. 输入对象不被原地修改，严格 JSON 序列化（无 default=str）。"""
    scale_metrics = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "net_amount": "15000.0",
        "net_amount_raw": "15000.0",
        "net_amount_unit": "万元",
        "unit": "万元",
        "net_to_circ_mv": "0.006755",
        "net_to_circ_mv_text": "0.006755",
        "net_to_amount": "0.027273",
        "net_to_amount_text": "0.027273",
        "circ_mv": "2220600.0",
        "circ_mv_unit": "万元",
        "circ_mv_source": "tushare.daily_basic",
        "amount": "550000.0",
        "amount_unit": "万元",
        "amount_source": "tushare.daily_basic",
        "denominator_source": "tushare.daily_basic",
        "denominator_sources": {
            "circ_mv": "tushare.daily_basic",
            "amount": "tushare.daily_basic",
        },
        "denominator_units": {
            "circ_mv": "万元",
            "amount": "万元",
        },
        "status": "available",
        "gaps": ["无缺口"],
        "gap_list": ["无缺口"],
    }
    selection = {
        "selected_source": "tushare_eastmoney_moneyflow_dc",
        "selected_algorithm_group": "new_algorithm_group",
        "reference_only": True,
    }
    fund_flow_evidence = {
        "scale_metrics": scale_metrics,
        "selection": selection,
        "records": [],
    }

    scale_metrics_snapshot = copy.deepcopy(scale_metrics)
    selection_snapshot = copy.deepcopy(selection)

    state = {
        "trade_date": "2026-08-14",
        "company_of_interest": "600519",
        "user_intent": {"focus_areas": [], "specific_questions": []},
        "market_data_context": {
            "fund_flow_evidence": fund_flow_evidence,
        },
    }
    state_snapshot = copy.deepcopy(state)

    llm = _RecordingLLM()
    collector = _MockCollector(market_data_context=state["market_data_context"])
    result = _run_analyst_node(llm, collector, state=state)

    # 验证输入对象未被原地修改
    assert scale_metrics == scale_metrics_snapshot
    assert selection == selection_snapshot
    assert state == state_snapshot

    # 验证 node 返回结果可严格 JSON 序列化（不得使用 default=str 掩盖）
    serialized = json.dumps(result, ensure_ascii=False)
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert "smart_money_report" in deserialized
    assert "fund_flow_consensus_guard" in deserialized
    assert "analyst_traces" in deserialized


def test_reference_only_strict_boolean_and_missing_selection():
    """11. reference_only 仅在 selection 中存在严格布尔值时展示 True/False；缺失、非 Mapping、缺键或非 bool 时显式处理。"""
    from tradingagents.agents.analysts.smart_money_analyst import format_fund_flow_scale_metrics_prompt

    valid_scale = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "status": "available",
        "net_to_circ_mv": "0.006755",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "net_to_amount": "0.027273",
        "amount_source": "tushare.daily_basic",
        "amount_unit": "万元",
    }
    unknown_marker = "reference_only=未知/缺少 selection，按 reference_only 纪律处理"

    # 缺少 selection (None)
    out_none = format_fund_flow_scale_metrics_prompt(valid_scale, None)
    assert unknown_marker in out_none
    assert "reference_only=False" not in out_none
    assert "reference_only=True" not in out_none

    # selection 为空 dict (缺键)
    out_empty = format_fund_flow_scale_metrics_prompt(valid_scale, {})
    assert unknown_marker in out_empty
    assert "reference_only=False" not in out_empty

    # selection 中 reference_only 为字符串 "false" (非 bool，禁止 bool("false")==True 强转)
    out_str_false = format_fund_flow_scale_metrics_prompt(valid_scale, {"reference_only": "false"})
    assert unknown_marker in out_str_false
    assert "reference_only=False" not in out_str_false
    assert "reference_only=True" not in out_str_false

    # selection 中 reference_only 为字符串 "true" (非 bool)
    out_str_true = format_fund_flow_scale_metrics_prompt(valid_scale, {"reference_only": "true"})
    assert unknown_marker in out_str_true

    # selection 中 reference_only 为整型 0 / 1 (非 bool)
    out_int_0 = format_fund_flow_scale_metrics_prompt(valid_scale, {"reference_only": 0})
    assert unknown_marker in out_int_0
    out_int_1 = format_fund_flow_scale_metrics_prompt(valid_scale, {"reference_only": 1})
    assert unknown_marker in out_int_1

    # 严格布尔值 True / False
    out_bool_false = format_fund_flow_scale_metrics_prompt(valid_scale, {"reference_only": False})
    assert "reference_only=False" in out_bool_false
    assert unknown_marker not in out_bool_false

    out_bool_true = format_fund_flow_scale_metrics_prompt(valid_scale, {"reference_only": True})
    assert "reference_only=True" in out_bool_true
    assert unknown_marker not in out_bool_true

    # 端到端 node 运行验证非 bool 时整个 HumanMessage 均无 reference_only=False
    fund_flow_evidence = {
        "scale_metrics": valid_scale,
        "selection": {"reference_only": "false", "selected_source": "ths"},
        "records": [],
    }
    llm = _RecordingLLM()
    collector = _MockCollector(fund_flow_evidence=fund_flow_evidence)
    _run_analyst_node(llm, collector)
    human_msg = llm.messages[1].content
    assert unknown_marker in human_msg
    assert "reference_only=False" not in human_msg


def test_available_status_downgrade_when_single_ratio():
    """12. 声明 available 但仅一个比率完整可用，必须降级为 partial，并增加契约矛盾 gap。"""
    from tradingagents.agents.analysts.smart_money_analyst import format_fund_flow_scale_metrics_prompt

    scale_metrics = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "status": "available",  # 声明为 available
        "net_to_circ_mv": "0.006755",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "net_to_amount": None,  # 但缺失另一个比率
        "amount_source": None,
        "amount_unit": None,
        "gaps": [],
    }
    selection = {"reference_only": True, "selected_source": "tushare"}

    out = format_fund_flow_scale_metrics_prompt(scale_metrics, selection)
    assert "- 状态: partial (部分可用)" in out
    assert "0.006755" in out
    assert "- 净额占成交额比 (net_to_amount): 缺失/不可用" in out
    assert "契约矛盾: status 声明为 available，但仅有一个比率完整可用，降级为 partial" in out


def test_unknown_and_missing_status_fail_closed():
    """13. 未知或缺失状态直接 fail-closed 为 unavailable，并保留原始异常说明。"""
    from tradingagents.agents.analysts.smart_money_analyst import format_fund_flow_scale_metrics_prompt

    base_scale = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "net_to_circ_mv": "0.006755",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "net_to_amount": "0.027273",
        "amount_source": "tushare.daily_basic",
        "amount_unit": "万元",
    }

    # 1. 缺失 status 键
    scale_no_status = dict(base_scale)
    out_no_status = format_fund_flow_scale_metrics_prompt(scale_no_status, {"reference_only": True})
    assert "- 状态: unavailable (相对规模不可用/不得据绝对净额替代)" in out_no_status
    assert "缺少 status 状态字段" in out_no_status

    # 2. status 为 unknown
    scale_unknown = dict(base_scale, status="unknown")
    out_unknown = format_fund_flow_scale_metrics_prompt(scale_unknown, {"reference_only": True})
    assert "- 状态: unavailable (相对规模不可用/不得据绝对净额替代)" in out_unknown
    assert "未知状态 'unknown'" in out_unknown

    # 3. status 为 error / invalid
    scale_err = dict(base_scale, status="calc_error")
    out_err = format_fund_flow_scale_metrics_prompt(scale_err, {"reference_only": True})
    assert "- 状态: unavailable (相对规模不可用/不得据绝对净额替代)" in out_err
    assert "未知状态 'calc_error'" in out_err

    # 4. status 为非字符串类型
    scale_num = dict(base_scale, status=500)
    out_num = format_fund_flow_scale_metrics_prompt(scale_num, {"reference_only": True})
    assert "- 状态: unavailable (相对规模不可用/不得据绝对净额替代)" in out_num
    assert "状态字段为未知类型" in out_num


def test_unavailable_status_with_ratios_never_presented_as_available():
    """14. unavailable 无论携带何值都不得呈现为可用。"""
    from tradingagents.agents.analysts.smart_money_analyst import format_fund_flow_scale_metrics_prompt

    scale_metrics = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "status": "unavailable",
        "net_to_circ_mv": "0.006755",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "net_to_amount": "0.027273",
        "amount_source": "tushare.daily_basic",
        "amount_unit": "万元",
        "gaps": ["上游强制判定不可用"],
    }
    selection = {"reference_only": True}

    out = format_fund_flow_scale_metrics_prompt(scale_metrics, selection)
    assert "- 状态: unavailable (相对规模不可用/不得据绝对净额替代)" in out
    # 虽携带数值，但绝不得作为可用比率呈现
    assert "0.006755" not in out
    assert "0.027273" not in out
    assert "上游强制判定不可用" in out
    assert "契约矛盾: status 声明为 unavailable 但携带比率数值，按 unavailable 纪律不予呈现" in out


def test_gaps_as_string_and_malformed_types():
    """15. gaps/gap_list 为字符串时保留为单项；为 list/tuple 时逐项保留；其他畸形类型增加显式契约 gap。"""
    from tradingagents.agents.analysts.smart_money_analyst import format_fund_flow_scale_metrics_prompt

    base_scale = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "status": "partial",
        "net_to_circ_mv": "0.006755",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "net_to_amount": None,
    }

    # 字符串形式的 gap
    scale_str = dict(base_scale, gaps="成交额数据缺失无法计算占比")
    out_str = format_fund_flow_scale_metrics_prompt(scale_str, {"reference_only": True})
    assert "* 成交额数据缺失无法计算占比" in out_str

    # 列表形式的 gaps
    scale_list = dict(base_scale, gaps=["缺口一", "缺口二"])
    out_list = format_fund_flow_scale_metrics_prompt(scale_list, {"reference_only": True})
    assert "* 缺口一" in out_list
    assert "* 缺口二" in out_list

    # 畸形对象：int
    scale_int = dict(base_scale, gaps=9999)
    out_int = format_fund_flow_scale_metrics_prompt(scale_int, {"reference_only": True})
    assert "契约异常: gaps 包含畸形类型 (int: 9999)" in out_int

    # 畸形对象：dict
    scale_dict = dict(base_scale, gaps={"invalid": "data"})
    out_dict = format_fund_flow_scale_metrics_prompt(scale_dict, {"reference_only": True})
    assert "契约异常: gaps 包含畸形类型 (dict:" in out_dict


def test_missing_ts_code_or_trade_date_fails_closed():
    """16. ts_code 或 trade_date 缺失时整体相对规模 unavailable；不得输出空标的/空日期却称完整可用。"""
    from tradingagents.agents.analysts.smart_money_analyst import format_fund_flow_scale_metrics_prompt

    base_scale = {
        "status": "available",
        "net_to_circ_mv": "0.006755",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "net_to_amount": "0.027273",
        "amount_source": "tushare.daily_basic",
        "amount_unit": "万元",
    }

    # 缺失 ts_code
    scale_no_code = dict(base_scale, ts_code="", trade_date="2026-08-14")
    out_no_code = format_fund_flow_scale_metrics_prompt(scale_no_code, {"reference_only": True})
    assert "- 状态: unavailable (相对规模不可用/不得据绝对净额替代)" in out_no_code
    assert "标的代码 (ts_code) 缺失" in out_no_code

    # 缺失 trade_date
    scale_no_date = dict(base_scale, ts_code="600519.SH", trade_date="")
    out_no_date = format_fund_flow_scale_metrics_prompt(scale_no_date, {"reference_only": True})
    assert "- 状态: unavailable (相对规模不可用/不得据绝对净额替代)" in out_no_date
    assert "交易日期 (trade_date) 缺失" in out_no_date


def test_missing_denominator_source_or_unit_makes_ratio_unusable():
    """17. 对应分母来源或单位缺失时该比率不可用，不得用“未指定”维持 available。"""
    from tradingagents.agents.analysts.smart_money_analyst import format_fund_flow_scale_metrics_prompt

    # 1. net_to_circ_mv 缺单位
    scale_no_unit = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "status": "available",
        "net_to_circ_mv": "0.006755",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": None,  # 缺单位
        "net_to_amount": "0.027273",
        "amount_source": "tushare.daily_basic",
        "amount_unit": "万元",
    }
    out_no_unit = format_fund_flow_scale_metrics_prompt(scale_no_unit, {"reference_only": True})
    assert "- 状态: partial (部分可用)" in out_no_unit
    assert "- 净额占流通市值比 (net_to_circ_mv): 缺失/不可用" in out_no_unit
    assert "- 净额占成交额比 (net_to_amount): 0.027273" in out_no_unit
    assert "分母单位缺失" in out_no_unit

    # 2. net_to_amount 缺来源
    scale_no_src = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "status": "available",
        "net_to_circ_mv": "0.006755",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "net_to_amount": "0.027273",
        "amount_source": None,  # 缺来源
        "amount_unit": "万元",
    }
    out_no_src = format_fund_flow_scale_metrics_prompt(scale_no_src, {"reference_only": True})
    assert "- 状态: partial (部分可用)" in out_no_src
    assert "- 净额占成交额比 (net_to_amount): 缺失/不可用" in out_no_src
    assert "分母来源缺失" in out_no_src

    # 3. 两者分母均缺来源/单位 -> unavailable
    scale_no_both = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "status": "available",
        "net_to_circ_mv": "0.006755",
        "circ_mv_source": None,
        "circ_mv_unit": None,
        "net_to_amount": "0.027273",
        "amount_source": None,
        "amount_unit": None,
    }
    out_no_both = format_fund_flow_scale_metrics_prompt(scale_no_both, {"reference_only": True})
    assert "- 状态: unavailable (相对规模不可用/不得据绝对净额替代)" in out_no_both


def test_denominator_mapping_fallback():
    """18. 允许真实 denominator_sources / denominator_units Mapping 作为精确 fallback。"""
    from tradingagents.agents.analysts.smart_money_analyst import format_fund_flow_scale_metrics_prompt

    scale_metrics = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "status": "available",
        "net_to_circ_mv": "0.006755",
        "circ_mv_source": None,  # 顶层为空
        "circ_mv_unit": None,
        "net_to_amount": "0.027273",
        "amount_source": None,
        "amount_unit": None,
        "denominator_sources": {
            "circ_mv": "tushare.daily_basic",
            "amount": "tushare.daily_basic",
        },
        "denominator_units": {
            "circ_mv": "万元",
            "amount": "万元",
        },
        "gaps": [],
    }
    selection = {"reference_only": True, "selected_source": "tushare"}

    out = format_fund_flow_scale_metrics_prompt(scale_metrics, selection)
    assert "- 状态: available (完整可用)" in out
    assert "- 净额占流通市值比 (net_to_circ_mv): 0.006755 (分母来源: tushare.daily_basic, 分母单位: 万元)" in out
    assert "- 净额占成交额比 (net_to_amount): 0.027273 (分母来源: tushare.daily_basic, 分母单位: 万元)" in out


def test_scale_pseudo_fields_invisible_in_entire_human_message():
    """19. 原始 scale 伪字段在整个 HumanMessage 中均不存在（封死 HIGH 级旁路）。"""
    scale_metrics = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "status": "available",
        "net_to_circ_mv": "0.006755",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "net_to_amount": "0.027273",
        "amount_source": "tushare.daily_basic",
        "amount_unit": "万元",
        # 伪字段
        "algorithm_group": "fake_scale_algo_group",
        "selected_algorithm_group": "fake_scale_algo_group",
        "reference_only": False,
    }
    selection = {
        "selected_source": "ths",
        "selected_algorithm_group": "real_selection_algo_group",
        "reference_only": True,
    }
    fund_flow_evidence = {
        "scale_metrics": scale_metrics,
        "selection": selection,
        "records": [],
    }

    llm = _RecordingLLM()
    collector = _MockCollector(fund_flow_evidence=fund_flow_evidence)
    _run_analyst_node(llm, collector)

    # 检查整个 HumanMessage
    human_msg = llm.messages[1].content
    assert "fake_scale_algo_group" not in human_msg
    assert "reference_only=False" not in human_msg
    assert "real_selection_algo_group" in human_msg
    assert "reference_only=True" in human_msg


def test_top_level_scale_bypass_rejected_and_nested_state_only_consumed():
    """20. 顶层错误路径 scale 不被消费，嵌套 state-only 正确消费。"""
    valid_scale = {
        "ts_code": "600519.SH",
        "trade_date": "2026-08-14",
        "status": "available",
        "net_to_circ_mv": "0.006755",
        "circ_mv_source": "tushare.daily_basic",
        "circ_mv_unit": "万元",
        "net_to_amount": "0.027273",
        "amount_source": "tushare.daily_basic",
        "amount_unit": "万元",
    }
    valid_sel = {
        "selected_source": "tushare",
        "selected_algorithm_group": "new_algorithm_group",
        "reference_only": True,
    }

    # 1. 顶层错误路径：market_data_context["scale_metrics"] 旁路不被消费
    llm1 = _RecordingLLM()
    pool_data = {
        "fund_flow_individual": "无数据",
        "market_data_context": {
            "scale_metrics": valid_scale,  # 顶层旁路
            # 没有 fund_flow_evidence
        },
        "lhb": "无数据",
        "indicators": {"vwma": "无数据"},
    }
    class _TopLevelCollector:
        def get(self, ticker, curr_date):
            return pool_data

    _run_analyst_node(llm1, _TopLevelCollector())
    human_msg1 = llm1.messages[1].content
    # 顶层旁路不得被消费，fail-closed 输出缺少 scale_metrics
    assert "缺少 scale_metrics 相对规模对象" in human_msg1
    assert "0.006755" not in human_msg1

    # 2. 嵌套 state-only 正常消费，无 UnboundLocalError
    llm2 = _RecordingLLM()
    state = {
        "trade_date": "2026-08-14",
        "company_of_interest": "600519",
        "user_intent": {"focus_areas": [], "specific_questions": []},
        "market_data_context": {
            "fund_flow_evidence": {
                "scale_metrics": valid_scale,
                "selection": valid_sel,
                "records": [],
            },
        },
    }
    # data_collector 为 None 触发 fallback 分支（state-only 路径）
    _run_analyst_node(llm2, None, state=state)
    human_msg2 = llm2.messages[1].content
    assert "- 状态: available (完整可用)" in human_msg2
    assert "0.006755" in human_msg2
    assert "0.027273" in human_msg2


def test_unserializable_evidence_raises_type_error():
    """21. 未知不可序列化 evidence 对象严格抛 TypeError。"""
    class _DummyUnknown:
        pass

    fund_flow_evidence = {
        "records": [],
        "unknown_obj": _DummyUnknown(),
    }
    llm = _RecordingLLM()
    collector = _MockCollector(fund_flow_evidence=fund_flow_evidence)

    with pytest.raises(TypeError):
        _run_analyst_node(llm, collector)
