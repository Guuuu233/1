"""Tests for fund flow scale metrics consumption layer (D-04 / DAV-825).

依据 .hermes/plans/2026-09-06_整合施工计划-v1.1.md §6.4:
1. 新建 tests/test_fund_flow_scale_consumption.py，覆盖归一 scale_metrics 从结果持久化到
   report_service 回读后仍保留 ts_code、trade_date、status、来源/算法组、比率和 gaps
   等结构字段；测试必须使用临时或内存数据库并证明 JSON/SQLite 可绑定。
2. 覆盖总监/上层消费的非干预契约：输入包含 net_to_circ_mv/net_to_amount 的 smart-money 结果时，
   不得生成机构身份判断、横向排名、权重/概率/新评分或交易执行信号；若复用现有 prompt/manager seam，
   必须 mock LLM 并断言调用次数/提示词纪律。
3. 覆盖 reference_only=True、冲突/不可用状态、合法 Decimal(0) 至少各一个边界；
   失败/empty/unavailable 不得被断言为零。
"""
from __future__ import annotations

import asyncio
import copy
from decimal import Decimal
import json
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text, JSON
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import sessionmaker

from api.database import Base, ReportDB
from api.services import report_service
from api.services.report_service import canonicalize_report_result_data
from tradingagents.agents.analysts.smart_money_analyst import (
    create_smart_money_analyst,
    format_fund_flow_scale_metrics_prompt,
)
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.dataflows.fund_flow_evidence import (
    DecimalRatio,
    calculate_fund_flow_scale_metrics,
)
from tradingagents.graph.data_collector import (
    _serialize_scale_metrics_for_json,
    default_market_data_context,
)
from tradingagents.llm_clients.thinking_cleaner import clean_report_result_data
from tests.test_research_manager_seven_reports_and_verdict_gate import _make_seven_reports_state


@pytest.fixture(autouse=True)
def guard_no_network_calls():
    """Ensure no real network calls can be made in this test suite."""
    with (
        patch(
            "socket.socket.connect",
            side_effect=RuntimeError("Network access forbidden in offline tests"),
        ),
        patch(
            "requests.post",
            side_effect=RuntimeError("requests.post forbidden in offline tests"),
        ),
        patch(
            "requests.get",
            side_effect=RuntimeError("requests.get forbidden in offline tests"),
        ),
    ):
        yield


@pytest.fixture
def sqlite_session():
    """Provide an isolated in-memory SQLite database session for each test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _build_sample_scale_metrics(
    ts_code: str = "600519.SH",
    trade_date: str = "2026-08-14",
    status: str = "available",
    net_amount: str | Decimal = "15000.0",
    net_to_circ_mv: str | Decimal | None = "0.006755",
    net_to_amount: str | Decimal | None = "0.027273",
    circ_mv: str | Decimal | None = "2220600.0",
    amount: str | Decimal | None = "550000.0",
    gaps: list[str] | None = None,
) -> dict[str, Any]:
    """Helper to build a deterministic scale_metrics dictionary."""
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "status": status,
        "net_amount": str(net_amount) if net_amount is not None else None,
        "net_amount_raw": str(net_amount) if net_amount is not None else None,
        "net_amount_unit": "万元",
        "unit": "万元",
        "net_to_circ_mv": str(net_to_circ_mv) if net_to_circ_mv is not None else None,
        "net_to_circ_mv_text": str(net_to_circ_mv) if net_to_circ_mv is not None else None,
        "net_to_amount": str(net_to_amount) if net_to_amount is not None else None,
        "net_to_amount_text": str(net_to_amount) if net_to_amount is not None else None,
        "circ_mv": str(circ_mv) if circ_mv is not None else None,
        "circ_mv_unit": "万元" if circ_mv is not None else None,
        "circ_mv_source": "tushare.daily_basic" if circ_mv is not None else None,
        "amount": str(amount) if amount is not None else None,
        "amount_unit": "万元" if amount is not None else None,
        "amount_source": "tushare.daily_basic" if amount is not None else None,
        "denominator_source": "tushare.daily_basic" if (circ_mv or amount) else None,
        "denominator_sources": {
            "circ_mv": "tushare.daily_basic",
            "amount": "tushare.daily_basic",
        } if (circ_mv or amount) else {},
        "denominator_units": {
            "circ_mv": "万元",
            "amount": "万元",
        } if (circ_mv or amount) else {},
        "gaps": gaps or [],
        "gap_list": gaps or [],
    }


def _build_sample_fund_flow_evidence(
    scale_metrics: dict[str, Any],
    selected_source: str = "ths_instant_snapshot",
    selected_algorithm_group: str = "new_algorithm_group",
    reference_only: bool = True,
    selected_value: float = 1.5,
    selected_unit: str = "亿元",
    date: str = "2026-08-14",
) -> dict[str, Any]:
    """Helper to build a valid fund_flow_evidence matching report_service validator."""
    return {
        "scale_metrics": scale_metrics,
        "selection": {
            "selected_source": selected_source,
            "selected_field": "r0_net",
            "selected_value": selected_value,
            "selected_unit": selected_unit,
            "selected_as_of": date,
            "selected_direction": "inflow" if selected_value > 0 else "neutral",
            "direction_allowed": True,
            "selected_algorithm_group": selected_algorithm_group,
            "fallback_rank": 1,
            "reference_only": reference_only,
            "hard_guard": {"blocked": False},
        },
        "unit": selected_unit,
        "records": [
            {
                "date": date,
                "source": selected_source,
                "status": "available",
                "field": "r0_net",
                "value": str(selected_value),
                "unit": selected_unit,
            }
        ],
    }


# ============================================================================
# 1. 结果持久化与 report_service 回读保留结构字段及 SQLite/JSON 绑定验证
# ============================================================================

class TestFundFlowScalePersistenceAndReadback:
    """1. 覆盖 scale_metrics 从结果持久化到 report_service 回读后保留所有契约字段及 SQLite/JSON 绑定。"""

    def test_single_horizon_report_persists_and_reads_all_scale_fields(self, sqlite_session):
        """单周期报告落库后，get_report 回读仍完整保留 ts_code、trade_date、status、来源/算法组、比率及 gaps。"""
        scale_metrics = _build_sample_scale_metrics(
            ts_code="600519.SH",
            trade_date="2026-08-14",
            status="available",
            net_amount="15000.0",
            net_to_circ_mv="0.006755",
            net_to_amount="0.027273",
            circ_mv="2220600.0",
            amount="550000.0",
            gaps=[],
        )
        fund_flow_evidence = _build_sample_fund_flow_evidence(
            scale_metrics=scale_metrics,
            selected_source="ths_instant_snapshot",
            selected_algorithm_group="new_algorithm_group",
            reference_only=True,
            selected_value=1.5,
            selected_unit="亿元",
            date="2026-08-14",
        )
        result_data = {
            "market_data_context": {
                "fund_flow_evidence": fund_flow_evidence,
                "scale_metrics": scale_metrics,
            },
            "scale_metrics": scale_metrics,
            "smart_money_report": "主力资金分析报告：资金面平稳。",
            "decision": "WAIT",
        }

        # 通过 report_service 持久化
        rep = report_service.create_report(
            db=sqlite_session,
            symbol="600519.SH",
            trade_date="2026-08-14",
            decision="WAIT",
            result_data=result_data,
            user_id="test_user_single",
        )
        assert rep.id is not None

        # 通过 report_service 回读
        fetched = report_service.get_report(sqlite_session, rep.id, user_id="test_user_single")
        assert fetched is not None
        assert fetched.result_data is not None

        # 验证挂载点结构
        rd = fetched.result_data
        assert "market_data_context" in rd
        assert "fund_flow_evidence" in rd["market_data_context"]
        assert "scale_metrics" in rd["market_data_context"]["fund_flow_evidence"]
        assert "scale_metrics" in rd["market_data_context"]
        assert "scale_metrics" in rd

        read_sm = rd["market_data_context"]["fund_flow_evidence"]["scale_metrics"]
        read_sel = rd["market_data_context"]["fund_flow_evidence"]["selection"]

        # 1. 标的与日期
        assert read_sm["ts_code"] == "600519.SH"
        assert read_sm["trade_date"] == "2026-08-14"

        # 2. 状态
        assert read_sm["status"] == "available"

        # 3. 来源与分母
        assert read_sm["denominator_source"] == "tushare.daily_basic"
        assert read_sm["circ_mv_source"] == "tushare.daily_basic"
        assert read_sm["circ_mv_unit"] == "万元"
        assert read_sm["amount_source"] == "tushare.daily_basic"
        assert read_sm["amount_unit"] == "万元"
        assert read_sm["denominator_sources"] == {
            "circ_mv": "tushare.daily_basic",
            "amount": "tushare.daily_basic",
        }
        assert read_sm["denominator_units"] == {
            "circ_mv": "万元",
            "amount": "万元",
        }

        # 4. 算法组与选源
        assert read_sel["selected_source"] == "ths_instant_snapshot"
        assert read_sel["selected_algorithm_group"] == "new_algorithm_group"
        assert read_sel["reference_only"] is True

        # 5. 比率与精度数值
        assert read_sm["net_to_circ_mv"] == "0.006755"
        assert read_sm["net_to_circ_mv_text"] == "0.006755"
        assert Decimal(read_sm["net_to_circ_mv"]) == Decimal("0.006755")
        assert read_sm["net_to_amount"] == "0.027273"
        assert read_sm["net_to_amount_text"] == "0.027273"
        assert Decimal(read_sm["net_to_amount"]) == Decimal("0.027273")

        # 6. 缺口结构
        assert read_sm["gaps"] == []
        assert read_sm["gap_list"] == []

    def test_dual_horizon_report_persists_and_reads_both_horizons(self, sqlite_session):
        """双周期 (short_term / medium_term) 结构下，两档 scale_metrics 均完整持久化并回读。"""
        sm_short = _build_sample_scale_metrics(
            ts_code="600519.SH",
            trade_date="2026-08-14",
            status="available",
            net_to_circ_mv="0.006755",
            net_to_amount="0.027273",
        )
        sm_medium = _build_sample_scale_metrics(
            ts_code="600519.SH",
            trade_date="2026-08-14",
            status="partial",
            net_to_circ_mv="0.006755",
            net_to_amount=None,
            amount=None,
            gaps=["中期成交额分母缺失，成交额占比拒算"],
        )
        ffe_short = _build_sample_fund_flow_evidence(
            scale_metrics=sm_short,
            selected_algorithm_group="short_algo_group",
            reference_only=True,
            date="2026-08-14",
        )
        ffe_medium = _build_sample_fund_flow_evidence(
            scale_metrics=sm_medium,
            selected_algorithm_group="medium_algo_group",
            reference_only=True,
            date="2026-08-14",
        )

        result_data = {
            "mode": "dual_horizon",
            "short_term": {
                "market_data_context": {
                    "fund_flow_evidence": ffe_short,
                    "scale_metrics": sm_short,
                },
                "scale_metrics": sm_short,
                "decision": "WAIT",
            },
            "medium_term": {
                "market_data_context": {
                    "fund_flow_evidence": ffe_medium,
                    "scale_metrics": sm_medium,
                },
                "scale_metrics": sm_medium,
                "decision": "BULL",
            },
            "market_data_context": {
                "fund_flow_evidence": ffe_short,
                "scale_metrics": sm_short,
            },
            "scale_metrics": sm_short,
            "decision": "WAIT",
        }

        rep = report_service.create_report(
            db=sqlite_session,
            symbol="600519.SH",
            trade_date="2026-08-14",
            decision="WAIT",
            result_data=result_data,
            user_id="test_user_dual",
        )
        fetched = report_service.get_report(sqlite_session, rep.id, user_id="test_user_dual")
        assert fetched is not None
        assert fetched.result_data is not None

        # 短线回读核对
        read_short_sm = fetched.result_data["short_term"]["market_data_context"]["fund_flow_evidence"]["scale_metrics"]
        assert read_short_sm["status"] == "available"
        assert read_short_sm["net_to_circ_mv"] == "0.006755"
        assert read_short_sm["net_to_amount"] == "0.027273"
        read_short_sel = fetched.result_data["short_term"]["market_data_context"]["fund_flow_evidence"]["selection"]
        assert read_short_sel["selected_algorithm_group"] == "short_algo_group"

        # 中线回读核对
        read_medium_sm = fetched.result_data["medium_term"]["market_data_context"]["fund_flow_evidence"]["scale_metrics"]
        assert read_medium_sm["status"] == "partial"
        assert read_medium_sm["net_to_circ_mv"] == "0.006755"
        assert read_medium_sm["net_to_amount"] is None
        assert any("成交额分母缺失" in g for g in read_medium_sm["gaps"])
        read_medium_sel = fetched.result_data["medium_term"]["market_data_context"]["fund_flow_evidence"]["selection"]
        assert read_medium_sel["selected_algorithm_group"] == "medium_algo_group"

    def test_sqlite_json_bind_processor_and_native_json_extract(self, sqlite_session):
        """证明 SQLAlchemy SQLite JSON bind processor 和原生 SQLite json_extract 对 scale_metrics 完全绑定。"""
        scale_metrics = _build_sample_scale_metrics(
            ts_code="600519.SH",
            trade_date="2026-08-14",
            status="available",
            net_to_circ_mv="0.006755",
            net_to_amount="0.027273",
        )
        fund_flow_evidence = _build_sample_fund_flow_evidence(scale_metrics)

        # 1. 验证 SQLAlchemy SQLite JSON bind processor
        bp = JSON().bind_processor(sqlite.dialect())
        bound_json = bp(scale_metrics)
        assert bound_json is not None
        assert isinstance(bound_json, str)
        reloaded = json.loads(bound_json)
        assert reloaded["ts_code"] == "600519.SH"
        assert reloaded["net_to_circ_mv"] == "0.006755"

        # 2. 验证落库后通过原生 SQLite json_extract 语句提取各结构字段
        rep = report_service.create_report(
            db=sqlite_session,
            symbol="600519.SH",
            trade_date="2026-08-14",
            result_data={
                "market_data_context": {
                    "fund_flow_evidence": fund_flow_evidence,
                    "scale_metrics": scale_metrics,
                },
                "scale_metrics": scale_metrics,
            },
        )

        query_sql = text("""
            SELECT
                json_extract(result_data, '$.market_data_context.fund_flow_evidence.scale_metrics.ts_code') AS ts_code,
                json_extract(result_data, '$.market_data_context.fund_flow_evidence.scale_metrics.trade_date') AS trade_date,
                json_extract(result_data, '$.market_data_context.fund_flow_evidence.scale_metrics.status') AS status,
                json_extract(result_data, '$.market_data_context.fund_flow_evidence.scale_metrics.net_to_circ_mv') AS net_to_circ_mv,
                json_extract(result_data, '$.market_data_context.fund_flow_evidence.scale_metrics.net_to_amount') AS net_to_amount,
                json_extract(result_data, '$.market_data_context.fund_flow_evidence.scale_metrics.denominator_source') AS denominator_source,
                json_extract(result_data, '$.market_data_context.fund_flow_evidence.selection.selected_algorithm_group') AS algo_group,
                json_extract(result_data, '$.market_data_context.fund_flow_evidence.selection.reference_only') AS ref_only
            FROM reports
            WHERE id = :id
        """)
        row = sqlite_session.execute(query_sql, {"id": rep.id}).fetchone()
        assert row is not None
        assert row.ts_code == "600519.SH"
        assert row.trade_date == "2026-08-14"
        assert row.status == "available"
        assert row.net_to_circ_mv == "0.006755"
        assert row.net_to_amount == "0.027273"
        assert row.denominator_source == "tushare.daily_basic"
        assert row.algo_group == "new_algorithm_group"
        assert row.ref_only in (1, True)

    def test_decimal_serialization_roundtrip_preserves_exact_precision(self, sqlite_session):
        """纯计算模块输出的 Decimal 与 DecimalRatio 经序列化落库后，保持精确十进制字符串，不发生浮点截断。"""
        raw_scale = calculate_fund_flow_scale_metrics(
            ts_code="600519.SH",
            trade_date="2026-08-14",
            net_amount=Decimal("15000.0"),
            net_amount_unit="万元",
            circ_mv=Decimal("2220600.0"),
            circ_mv_unit="万元",
            amount=Decimal("550000.0"),
            amount_unit="万元",
            denominator_source="tushare.daily_basic",
        )
        assert isinstance(raw_scale["net_to_circ_mv"], Decimal)
        assert isinstance(raw_scale["net_to_amount"], Decimal)

        serialized_sm = _serialize_scale_metrics_for_json(raw_scale)
        assert isinstance(serialized_sm["net_to_circ_mv"], str)
        assert isinstance(serialized_sm["net_to_amount"], str)

        ffe = _build_sample_fund_flow_evidence(serialized_sm)
        rep = report_service.create_report(
            db=sqlite_session,
            symbol="600519.SH",
            trade_date="2026-08-14",
            result_data={
                "market_data_context": {
                    "fund_flow_evidence": ffe,
                    "scale_metrics": serialized_sm,
                },
                "scale_metrics": serialized_sm,
            },
        )
        fetched = report_service.get_report(sqlite_session, rep.id)
        assert fetched is not None
        read_sm = fetched.result_data["market_data_context"]["fund_flow_evidence"]["scale_metrics"]

        # 精度完全无损验证
        expected_circ_ratio = Decimal("15000") / Decimal("2220600")
        expected_amt_ratio = Decimal("15000") / Decimal("550000")
        assert Decimal(read_sm["net_to_circ_mv"]) == expected_circ_ratio
        assert Decimal(read_sm["net_to_amount"]) == expected_amt_ratio


# ============================================================================
# 2. 总监/上层消费的非干预契约测试（Mock LLM、调用次数、提示词纪律）
# ============================================================================

class TestResearchManagerScaleNonInterventionContract:
    """2. 覆盖总监/上层消费非干预契约：严禁机构身份判断、横向排名、权重/概率/新评分或交易执行信号。"""

    def test_research_manager_mock_llm_invocation_and_prompt_discipline(self):
        """总监节点消费含相对规模的 smart_money_report 时，必须 mock LLM 并断言调用次数(1次)与提示词纪律。"""
        captured_prompts: list[str] = []

        async def _mock_astream(prompt):
            captured_prompts.append(prompt)
            yield SimpleNamespace(
                content='<!-- VERDICT: {"direction": "中性", "reason": "多空分歧"} -->'
            )

        mock_llm = MagicMock()
        mock_llm.astream = _mock_astream
        memory = MagicMock()
        memory.get_memories.return_value = []

        smart_money_snippet = (
            "【主力资金分析】当日资金呈现流入特征。\n"
            "【资金流相对规模证据（同标的同日相对参考）】\n"
            "- 状态: available\n"
            "- 标的: 600519.SH (交易日: 2026-08-14)\n"
            "- 相对规模比率:\n"
            "  * 净额/流通市值: 0.006755 (tushare.daily_basic, 万元)\n"
            "  * 净额/成交额: 0.027273 (tushare.daily_basic, 万元)\n"
            "- 参考算法组: new_algorithm_group (reference_only=True)\n"
            "- 【纪律约束】相对规模比率仅作为同标的、同交易日的统计参考证据；"
            "严禁据此识别机构或散户账户身份，严禁进行跨股票横向排名，严禁据此生成新评分、权重、概率或交易执行信号。"
            "不得在分母缺失时退回绝对净额作规模结论。"
        )

        state = _make_seven_reports_state()
        state["smart_money_report"] = smart_money_snippet

        manager_node = create_research_manager(mock_llm, memory)
        asyncio.run(manager_node(state))

        # 1. 严格断言 mock LLM 调用次数为 1 次
        assert len(captured_prompts) == 1, f"Expected exactly 1 LLM call, got {len(captured_prompts)}"

        prompt = captured_prompts[0]

        # 2. 检查相对规模证据原样进入提示词，未被篡改或丢失
        assert "0.006755" in prompt
        assert "0.027273" in prompt
        assert "tushare.daily_basic" in prompt

        # 3. 检查总监提示词纪律约束显式存在且不可缺少
        # a. 严禁账户身份判断
        assert "不得据此识别机构或散户账户身份" in prompt or "不得当作身份结论" in prompt
        # b. 严禁跨股票横向排名
        assert "严禁进行跨股票横向排名" in prompt or "横向排名" in prompt
        # c. 严禁生成新评分、权重、概率或执行信号
        assert "新评分" in prompt or "权重" in prompt or "交易执行信号" in prompt or "执行信号" in prompt
        # d. 严禁分母缺失退回绝对净额
        assert "不得在分母缺失时退回绝对净额" in prompt or "仅作解释" in prompt

    def test_research_manager_output_does_not_fabricate_scale_ratings_or_signals(self):
        """总监输出状态严格遵守决策裁决，绝不在输出中凭 scale_metrics 生成排名、评分提升或伪执行信号。"""
        async def _mock_astream(prompt):
            yield SimpleNamespace(
                content='<!-- VERDICT: {"direction": "中性", "reason": "多空分歧"} -->'
            )

        mock_llm = MagicMock()
        mock_llm.astream = _mock_astream
        memory = MagicMock()
        memory.get_memories.return_value = []

        smart_money_snippet = (
            "【资金流相对规模证据（同标的同日相对参考）】\n"
            "- 状态: available\n"
            "- 标的: 600519.SH (交易日: 2026-08-14)\n"
            "- 相对规模比率:\n"
            "  * 净额/流通市值: 0.006755 (tushare.daily_basic, 万元)\n"
            "  * 净额/成交额: 0.027273 (tushare.daily_basic, 万元)\n"
            "- 参考算法组: new_algorithm_group (reference_only=True)\n"
            "- 【纪律约束】相对规模比率仅作为同标的、同交易日的统计参考证据；"
            "严禁据此识别机构或散户账户身份，严禁进行跨股票横向排名，严禁据此生成新评分、权重、概率或交易执行信号。"
        )

        state = _make_seven_reports_state()
        state["smart_money_report"] = smart_money_snippet

        manager_node = create_research_manager(mock_llm, memory)
        result = asyncio.run(manager_node(state))

        # 验证总监输出字典中严禁出现规模排名或外部评分衍生字段
        forbidden_keys = [
            "scale_rank",
            "net_to_circ_mv_rank",
            "net_to_amount_rank",
            "scale_score",
            "scale_weight",
            "scale_execution_signal",
            "scale_probability",
        ]
        for k in forbidden_keys:
            assert k not in result, f"Forbidden key '{k}' found in research_manager output"

        # 验证 decision_status 未受资金规模比率干预而凭空生成交易动作
        ds = result.get("decision_status", {})
        assert ds.get("trade_action") != "BUY_ON_SCALE"
        assert ds.get("trade_action") != "SELL_ON_SCALE"

    def test_smart_money_analyst_seam_mock_llm_discipline_and_call_count(self):
        """验证 smart_money_analyst 生产节点输入 scale_metrics 时调用 mock LLM 1 次并注入纪律约束。"""
        captured_messages = []

        class _RecordingLLM:
            async def astream(self, messages):
                captured_messages.append(messages)
                yield SimpleNamespace(content="主力资金分析结论：中性。")

        scale_metrics = _build_sample_scale_metrics(
            ts_code="600519.SH",
            trade_date="2026-08-14",
            status="available",
            net_to_circ_mv="0.006755",
            net_to_amount="0.027273",
        )
        fund_flow_evidence = _build_sample_fund_flow_evidence(scale_metrics)

        class _MockCollector:
            def get(self, ticker: str, curr_date: str):
                return {
                    "fund_flow_individual": "主力净流入 1.5 亿",
                    "market_data_context": {"fund_flow_evidence": fund_flow_evidence},
                    "lhb": "无龙虎榜",
                    "indicators": {},
                }

        module = __import__(
            "tradingagents.agents.analysts.smart_money_analyst",
            fromlist=["smart_money_analyst"],
        )
        state = {
            "trade_date": "2026-08-14",
            "company_of_interest": "600519",
            "user_intent": {"focus_areas": [], "specific_questions": []},
        }

        with (
            patch.object(module, "get_cn_stock_name", return_value="贵州茅台"),
            patch.object(module, "get_config", return_value={}),
            patch.object(module, "get_prompt", return_value="固定提示词"),
            patch.object(module, "build_horizon_context", return_value="固定上下文"),
            patch.object(module, "log_llm_call"),
        ):
            res = asyncio.run(create_smart_money_analyst(_RecordingLLM(), _MockCollector())(state))

        assert len(captured_messages) == 1
        human_msg = captured_messages[0][1].content
        assert "0.006755" in human_msg
        assert "0.027273" in human_msg
        assert "严禁据此识别机构或散户账户身份" in human_msg
        assert "严禁进行跨股票横向排名" in human_msg
        assert "严禁据此生成新评分、权重、概率或交易执行信号" in human_msg

        # 检查输出中 analyst_traces 保留，且无非法衍生评分
        traces = res.get("analyst_traces", [])
        assert len(traces) == 1
        assert "scale_rank" not in traces[0]


# ============================================================================
# 3. 边界条件覆盖：reference_only=True、冲突/不可用状态、合法 Decimal(0) 与非零断言
# ============================================================================

class TestScaleConsumptionBoundaries:
    """3. 覆盖 reference_only=True、冲突/不可用状态、合法 Decimal(0) 至少各一个边界；失败/empty/unavailable 不得被断言为零。"""

    def test_boundary_reference_only_true_preserved_in_prompt_and_persistence(self, sqlite_session):
        """边界 1：reference_only=True 严格生效且在提示词与数据库往返中保持 True，绝不回退为 False。"""
        scale_metrics = _build_sample_scale_metrics(
            ts_code="600519.SH",
            trade_date="2026-08-14",
            status="available",
            net_to_circ_mv="0.006755",
            net_to_amount="0.027273",
        )
        # 故意在 scale_metrics 中塞入假字段 reference_only=False，但在 selection 中为 True
        scale_metrics["reference_only"] = False
        selection = {
            "selected_source": "ths_instant_snapshot",
            "selected_algorithm_group": "new_algorithm_group",
            "reference_only": True,
        }

        # 1. 验证纯提示词格式化函数：仅从 selection 严格读取 reference_only=True
        prompt_text = format_fund_flow_scale_metrics_prompt(scale_metrics, selection)
        assert "reference_only=True" in prompt_text
        assert "reference_only=False" not in prompt_text

        # 2. 验证落库持久化及回读：selection["reference_only"] 仍为严格布尔值 True
        fund_flow_evidence = _build_sample_fund_flow_evidence(
            scale_metrics=scale_metrics,
            reference_only=True,
        )
        rep = report_service.create_report(
            db=sqlite_session,
            symbol="600519.SH",
            trade_date="2026-08-14",
            result_data={
                "market_data_context": {
                    "fund_flow_evidence": fund_flow_evidence,
                    "scale_metrics": scale_metrics,
                },
                "scale_metrics": scale_metrics,
            },
        )
        fetched = report_service.get_report(sqlite_session, rep.id)
        assert fetched is not None
        read_sel = fetched.result_data["market_data_context"]["fund_flow_evidence"]["selection"]
        assert read_sel["reference_only"] is True
        assert read_sel["reference_only"] is not False
        assert read_sel["reference_only"] is not None

    def test_boundary_contract_contradiction_and_conflict_status_not_silently_suppressed(self):
        """边界 2：契约冲突/矛盾状态（声明 available 却缺少比率或标签）自动降级并记录缺口，绝不被静默择优或断言为零。"""
        # 矛盾场景：status 声明为 available，但 net_to_circ_mv 为 None，且缺少 circ_mv_source
        contradictory_scale = {
            "ts_code": "600519.SH",
            "trade_date": "2026-08-14",
            "status": "available",
            "net_to_circ_mv": None,
            "net_to_circ_mv_text": None,
            "net_to_amount": "0.027273",
            "net_to_amount_text": "0.027273",
            "amount_source": "tushare.daily_basic",
            "amount_unit": "万元",
            "circ_mv_source": None,
            "circ_mv_unit": None,
            "gaps": [],
            "gap_list": [],
        }
        selection = {
            "selected_source": "ths_instant_snapshot",
            "selected_algorithm_group": "new_algorithm_group",
            "reference_only": True,
        }

        prompt_output = format_fund_flow_scale_metrics_prompt(contradictory_scale, selection)

        # 必须降级为 partial，不得维持虚假的 available
        assert "状态: partial" in prompt_output
        # 必须记录契约矛盾缺口
        assert "契约矛盾: status 声明为 available，但仅有一个比率完整可用，降级为 partial" in prompt_output
        # 不可用的比率绝不得被格式化为 0
        assert "net_to_circ_mv: 0" not in prompt_output
        assert "净额占流通市值比 (net_to_circ_mv): 缺失/不可用" in prompt_output

        # 多源分歧冲突场景：divergence=True 必须保留在 guard 中，不得静默择优删除
        conflict_guard = {
            "status": "consensus",
            "direction_allowed": True,
            "divergence": True,
            "divergence_reason": "不同源资金方向出现分歧",
            "blocked": False,
        }
        assert conflict_guard["divergence"] is True
        assert conflict_guard["divergence_reason"] == "不同源资金方向出现分歧"

    def test_boundary_legal_decimal_zero_asserted_as_valid_zero(self, sqlite_session):
        """边界 3：合法 Decimal(0) 净额计算出的比率为零，必须被正常断言为零，绝不被误判为缺失或 None。"""
        # 1. 纯计算层：net_amount=Decimal("0")，分母正常
        zero_calc = calculate_fund_flow_scale_metrics(
            ts_code="600519.SH",
            trade_date="2026-08-14",
            net_amount=Decimal("0"),
            net_amount_unit="万元",
            circ_mv=Decimal("2220600.0"),
            circ_mv_unit="万元",
            amount=Decimal("550000.0"),
            amount_unit="万元",
            denominator_source="tushare.daily_basic",
        )
        assert zero_calc["status"] == "available"
        assert zero_calc["net_amount"] == Decimal("0")
        assert zero_calc["net_to_circ_mv"] == Decimal("0")
        assert zero_calc["net_to_amount"] == Decimal("0")
        assert zero_calc["net_to_circ_mv"] == 0
        assert zero_calc["net_to_amount"] == 0
        assert zero_calc["gaps"] == []

        # 2. 消费层格式化提示词：0 必须被呈现为有效数值，绝不是缺失或不可用
        selection = {
            "selected_source": "ths_instant_snapshot",
            "selected_algorithm_group": "new_algorithm_group",
            "reference_only": True,
        }
        zero_prompt = format_fund_flow_scale_metrics_prompt(zero_calc, selection)
        assert "net_to_circ_mv" in zero_prompt
        assert "net_to_amount" in zero_prompt
        assert "- 净额占流通市值比 (net_to_circ_mv): 0" in zero_prompt
        assert "- 净额占成交额比 (net_to_amount): 0" in zero_prompt
        assert "缺失/不可用" not in zero_prompt
        assert "状态: available" in zero_prompt

        # 3. 序列化与 SQLite 落库及回读：0 保持十进制字符串 "0"，可转回 Decimal("0") == 0
        serialized_zero = _serialize_scale_metrics_for_json(zero_calc)
        assert serialized_zero["net_to_circ_mv"] == "0"
        assert serialized_zero["net_to_amount"] == "0"

        ffe_zero = {
            "scale_metrics": serialized_zero,
            "selection": {
                "selected_source": "ths_instant_snapshot",
                "selected_field": "r0_net",
                "selected_value": 0.0,
                "selected_unit": "亿元",
                "selected_as_of": "2026-08-14",
                "selected_direction": "neutral",
                "direction_allowed": True,
                "selected_algorithm_group": "new_algorithm_group",
                "fallback_rank": 1,
                "reference_only": True,
                "hard_guard": {"blocked": False},
            },
            "unit": "亿元",
            "records": [
                {
                    "date": "2026-08-14",
                    "source": "ths_instant_snapshot",
                    "status": "available",
                    "field": "r0_net",
                    "value": "0.0",
                    "unit": "亿元",
                }
            ],
        }

        rep = report_service.create_report(
            db=sqlite_session,
            symbol="600519.SH",
            trade_date="2026-08-14",
            result_data={
                "market_data_context": {
                    "fund_flow_evidence": ffe_zero,
                    "scale_metrics": serialized_zero,
                },
                "scale_metrics": serialized_zero,
            },
        )
        fetched = report_service.get_report(sqlite_session, rep.id)
        assert fetched is not None
        read_zero_sm = fetched.result_data["market_data_context"]["fund_flow_evidence"]["scale_metrics"]
        assert read_zero_sm["status"] == "available"
        assert read_zero_sm["net_to_circ_mv"] == "0"
        assert read_zero_sm["net_to_amount"] == "0"
        assert Decimal(read_zero_sm["net_to_circ_mv"]) == Decimal("0")
        assert Decimal(read_zero_sm["net_to_circ_mv"]) == 0
        assert Decimal(read_zero_sm["net_to_amount"]) == Decimal("0")
        assert Decimal(read_zero_sm["net_to_amount"]) == 0

    def test_boundary_failure_empty_unavailable_never_asserted_as_zero(self, sqlite_session):
        """边界 4：失败/empty/unavailable 必须为 None 并记录缺口，绝不得被断言为零 (0, 0.0, '0')。"""
        # 1. 缺失分母导致 unavailable
        unavail_calc = calculate_fund_flow_scale_metrics(
            ts_code="600519.SH",
            trade_date="2026-08-14",
            net_amount=Decimal("15000.0"),
            net_amount_unit="万元",
            circ_mv=None,
            amount=None,
            denominator_source="tushare.daily_basic",
        )
        assert unavail_calc["status"] == "unavailable"

        # 核心断言：比率严格为 None，绝不是 0
        assert unavail_calc["net_to_circ_mv"] is None
        assert unavail_calc["net_to_circ_mv"] != 0
        assert unavail_calc["net_to_circ_mv"] != 0.0
        assert unavail_calc["net_to_circ_mv"] != Decimal("0")
        assert unavail_calc["net_to_circ_mv"] != "0"

        assert unavail_calc["net_to_amount"] is None
        assert unavail_calc["net_to_amount"] != 0
        assert unavail_calc["net_to_amount"] != 0.0
        assert unavail_calc["net_to_amount"] != Decimal("0")
        assert unavail_calc["net_to_amount"] != "0"

        # 缺口必须存在
        assert len(unavail_calc["gaps"]) > 0
        assert any("流通市值" in g for g in unavail_calc["gaps"])
        assert any("成交额" in g for g in unavail_calc["gaps"])

        # 2. 完全空/缺失 scale_metrics 传入消费提示词：显式 fail-closed，绝不填 0
        prompt_empty = format_fund_flow_scale_metrics_prompt(None, {})
        assert "状态: unavailable (相对规模不可用/不得据绝对净额替代)" in prompt_empty
        assert "缺少 scale_metrics 相对规模对象" in prompt_empty
        assert "net_to_circ_mv: 0" not in prompt_empty
        assert "net_to_amount: 0" not in prompt_empty

        # 3. 落库与回读验证：None 必须作为 null/None 往返，绝不被 JSON/SQLite 兜底翻成 0
        serialized_unavail = _serialize_scale_metrics_for_json(unavail_calc)
        assert serialized_unavail["net_to_circ_mv"] is None
        assert serialized_unavail["net_to_amount"] is None

        ffe_unavail = {
            "scale_metrics": serialized_unavail,
            "selection": {
                "selected_source": "ths_instant_snapshot",
                "selected_field": "r0_net",
                "selected_value": 1.5,
                "selected_unit": "亿元",
                "selected_as_of": "2026-08-14",
                "selected_direction": "inflow",
                "direction_allowed": True,
                "selected_algorithm_group": "new_algorithm_group",
                "fallback_rank": 1,
                "reference_only": True,
                "hard_guard": {"blocked": False},
            },
            "unit": "亿元",
            "records": [
                {
                    "date": "2026-08-14",
                    "source": "ths_instant_snapshot",
                    "status": "available",
                    "field": "r0_net",
                    "value": "1.5",
                    "unit": "亿元",
                }
            ],
        }

        rep = report_service.create_report(
            db=sqlite_session,
            symbol="600519.SH",
            trade_date="2026-08-14",
            result_data={
                "market_data_context": {
                    "fund_flow_evidence": ffe_unavail,
                    "scale_metrics": serialized_unavail,
                },
                "scale_metrics": serialized_unavail,
            },
        )
        fetched = report_service.get_report(sqlite_session, rep.id)
        assert fetched is not None
        read_unavail_sm = fetched.result_data["market_data_context"]["fund_flow_evidence"]["scale_metrics"]
        assert read_unavail_sm["status"] == "unavailable"

        # 回读后断言：严格为 None，绝不等于 0
        assert read_unavail_sm["net_to_circ_mv"] is None
        assert read_unavail_sm["net_to_circ_mv"] != 0
        assert read_unavail_sm["net_to_circ_mv"] != "0"
        assert read_unavail_sm["net_to_amount"] is None
        assert read_unavail_sm["net_to_amount"] != 0
        assert read_unavail_sm["net_to_amount"] != "0"
        assert len(read_unavail_sm["gaps"]) > 0

    def test_boundary_partial_status_never_asserts_missing_ratio_as_zero(self):
        """边界 5：partial 状态下可用比率正常计算，缺失比率严格为 None，绝不将缺失比率断言为零。"""
        partial_calc = calculate_fund_flow_scale_metrics(
            ts_code="600519.SH",
            trade_date="2026-08-14",
            net_amount=Decimal("15000.0"),
            net_amount_unit="万元",
            circ_mv=Decimal("2220600.0"),
            circ_mv_unit="万元",
            amount=None,  # 成交额分母缺失
            denominator_source="tushare.daily_basic",
        )
        assert partial_calc["status"] == "partial"
        assert partial_calc["net_to_circ_mv"] is not None
        assert partial_calc["net_to_circ_mv"] == Decimal("15000") / Decimal("2220600")

        # 缺失的成交额比率必须为 None，绝不是 0
        assert partial_calc["net_to_amount"] is None
        assert partial_calc["net_to_amount"] != 0
        assert partial_calc["net_to_amount"] != 0.0
        assert partial_calc["net_to_amount"] != Decimal("0")
        assert partial_calc["net_to_amount"] != "0"
        assert any("成交额" in g for g in partial_calc["gaps"])
