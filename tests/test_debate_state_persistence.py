"""Tests for debate state and risk feedback state persistence and hoisting (DAV-205, DAV-210)."""

import asyncio
import json
from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from api import main
from api.job_store import InMemoryJobStore
from api.services import report_service
from tradingagents.agents.utils import debate_utils
from tradingagents.graph.trading_graph import TradingAgentsGraph


class TestDebateStatePayloadExtraction:
    """Test API level payload extraction for debate states in _build_result_payload."""

    def test_build_result_payload_includes_debate_states(self):
        final_state = {
            "company_of_interest": "600519.SH",
            "horizon": "short",
            "trade_date": "2026-08-20",
            "investment_plan": "测试投资计划",
            "trader_investment_plan": "测试交易计划",
            "final_trade_decision": "买入",
            "investment_debate_state": {
                "bull_history": "看多观点",
                "bear_history": "看空观点",
                "judge_decision": "多头胜出",
            },
            "risk_debate_state": {
                "aggressive_history": "激进风控观点",
                "conservative_history": "保守风控观点",
                "neutral_history": "中性风控观点",
                "judge_decision": "通过交易",
            },
            "risk_feedback_state": {
                "latest_risk_verdict": "pass",
                "retry_count": 0,
                "max_retries": 1,
            },
        }
        payload = main._build_result_payload(final_state)

        assert payload["investment_debate_state"] == final_state["investment_debate_state"]
        assert payload["risk_debate_state"] == final_state["risk_debate_state"]
        assert payload["risk_feedback_state"] == final_state["risk_feedback_state"]

    def test_build_result_payload_handles_none_debate_states(self):
        final_state = {
            "company_of_interest": "600519.SH",
            "horizon": "short",
            "trade_date": "2026-08-20",
            "final_trade_decision": "买入",
        }
        payload = main._build_result_payload(final_state)
        assert payload.get("investment_debate_state") is None
        assert payload.get("risk_debate_state") is None
        assert payload.get("risk_feedback_state") is None


class TestHorizonResultDebateStates:
    """Test Graph core _build_horizon_result includes debate states."""

    def _make_mock_graph(self):
        with patch("tradingagents.graph.trading_graph.create_llm_client"), \
             patch("tradingagents.graph.trading_graph.FinancialSituationMemory"), \
             patch("tradingagents.graph.trading_graph.GraphSetup"), \
             patch("tradingagents.graph.trading_graph.ConditionalLogic"), \
             patch("tradingagents.graph.trading_graph.Propagator"), \
             patch("tradingagents.graph.trading_graph.Reflector"), \
             patch("tradingagents.graph.trading_graph.SignalProcessor"), \
             patch("tradingagents.graph.trading_graph.set_config"):
            ta = TradingAgentsGraph.__new__(TradingAgentsGraph)
            ta.debug = False
            ta.config = {}
            ta.callbacks = []
            ta.ticker = None
            ta.log_states_dict = {}
            ta.quick_thinking_llm = MagicMock()
            return ta

    def test_build_horizon_result_includes_debate_and_risk_states(self):
        ta = self._make_mock_graph()
        final_state = {
            "company_of_interest": "600519.SH",
            "horizon": "short",
            "trade_date": "2026-08-20",
            "investment_plan": "短期投资计划",
            "trader_investment_plan": "短期交易计划",
            "final_trade_decision": "买入",
            "investment_debate_state": {
                "bull_history": "短线多头观点",
                "bear_history": "短线空头观点",
                "judge_decision": "短线多头胜",
            },
            "risk_debate_state": {
                "aggressive_history": "短线激进观点",
                "conservative_history": "短线保守观点",
                "neutral_history": "短线中性观点",
                "judge_decision": "短线风控通过",
            },
            "risk_feedback_state": {
                "latest_risk_verdict": "pass",
                "retry_count": 0,
                "max_retries": 1,
            },
        }

        result = ta._build_horizon_result("short", final_state)

        assert result["investment_debate_state"] == final_state["investment_debate_state"]
        assert result["risk_debate_state"] == final_state["risk_debate_state"]
        assert result["risk_feedback_state"] == final_state["risk_feedback_state"]

    def test_build_horizon_result_handles_none_debate_states(self):
        ta = self._make_mock_graph()
        result = ta._build_horizon_result("medium", {})
        assert result.get("investment_debate_state") is None
        assert result.get("risk_debate_state") is None
        assert result.get("risk_feedback_state") is None


class _FakePropagator:
    def __init__(self, horizon):
        self.horizon = horizon

    def get_graph_args(self):
        return {"config": {"configurable": {}}}

    def create_initial_state(self, *_args, **kwargs):
        return {
            "horizon": kwargs.get("horizon", "short"),
            "market_data_context": kwargs.get("market_data_context"),
        }


class _FakeGraphStream:
    @staticmethod
    def _state(init_state):
        horizon = init_state.get("horizon", "short")
        return {
            "company_of_interest": "600519.SH",
            "trade_date": "2026-08-20",
            "horizon": horizon,
            "news_report": f"{horizon} news",
            "final_trade_decision": f"{horizon} decision 买入",
            "investment_plan": f"{horizon} 投资计划",
            "trader_investment_plan": f"{horizon} 交易计划",
            "investment_debate_state": {
                "bull_history": f"{horizon} 多头观点",
                "bear_history": f"{horizon} 空头观点",
                "judge_decision": f"{horizon} 多头胜",
            },
            "risk_debate_state": {
                "aggressive_history": f"{horizon} 激进观点",
                "conservative_history": f"{horizon} 保守观点",
                "neutral_history": f"{horizon} 中性观点",
                "judge_decision": f"{horizon} 风控通过",
            },
            "risk_feedback_state": {
                "latest_risk_verdict": "pass",
                "retry_count": 0,
                "max_retries": 1,
            },
            "market_data_context": init_state.get("market_data_context"),
            "analyst_traces": [{"horizon": horizon}],
        }

    async def astream(self, init_state, **_kwargs):
        yield self._state(init_state)


class _FakeTradingGraphForJob:
    def __init__(self, selected_analysts, data_collector, **_kwargs):
        self.data_collector = data_collector
        self.propagator = _FakePropagator("unknown")
        self.graph = _FakeGraphStream()
        self.role_resolved_configs = {}
        self.quick_thinking_llm = object()

    def process_signal(self, decision):
        return "BUY"

    def _build_horizon_result(self, horizon, state):
        res = dict(state)
        res["horizon"] = horizon
        return res


class TestJobExecutionDebatePersistence:
    """Test full job execution persists debate states to ReportDB.result_data."""

    def test_dual_horizon_persists_debate_states_to_result_data(self):
        job_id = f"job-{uuid4().hex}"
        store = InMemoryJobStore()
        collector = MagicMock()
        collector.collect.return_value = {"market_data_context": {"daily": {"as_of": "2026-08-20"}}}
        saved_reports = []
        db = MagicMock()

        request = main.AnalyzeRequest(
            symbol="600519.SH",
            trade_date="2026-08-20",
            horizons=["short", "medium"],
            selected_analysts=[],
        )

        def capture_create_report(**kwargs):
            saved_reports.append(kwargs)

        async def run_job():
            stream = main._stream_job_events(job_id)
            await stream.__anext__()
            task = asyncio.create_task(
                main._run_job_inner(job_id, request, stream_events=False, save_report=True)
            )
            async for chunk in stream:
                if "event: done" in chunk:
                    break
            await task
            await stream.aclose()

        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "_shared_data_collector", collector),
            patch.object(main, "TradingAgentsGraph", _FakeTradingGraphForJob),
            patch.object(main, "_build_runtime_config", return_value={}),
            patch.object(main, "_resolve_and_freeze_custom_prompts", return_value=({}, False)),
            patch.object(main, "get_db_ctx", return_value=nullcontext(db)),
            patch.object(report_service, "init_report"),
            patch.object(report_service, "update_report_partial"),
            patch.object(report_service, "extract_structured_data", return_value=None),
            patch.object(report_service, "create_report", side_effect=capture_create_report),
        ):
            asyncio.run(run_job())

        job = store.get_job(job_id)
        assert job["status"] == "completed"
        result_data = job["result"]

        # Check short_term and medium_term in result_data have debate states
        assert result_data["short_term"]["investment_debate_state"]["judge_decision"] == "short 多头胜"
        assert result_data["short_term"]["risk_debate_state"]["judge_decision"] == "short 风控通过"
        assert result_data["short_term"]["risk_feedback_state"]["latest_risk_verdict"] == "pass"

        assert result_data["medium_term"]["investment_debate_state"]["judge_decision"] == "medium 多头胜"
        assert result_data["medium_term"]["risk_debate_state"]["judge_decision"] == "medium 风控通过"
        assert result_data["medium_term"]["risk_feedback_state"]["latest_risk_verdict"] == "pass"

        # Check saved report in ReportDB
        assert len(saved_reports) == 1
        saved = saved_reports[0]["result_data"]
        assert saved["short_term"]["investment_debate_state"]["judge_decision"] == "short 多头胜"
        assert saved["medium_term"]["risk_debate_state"]["judge_decision"] == "medium 风控通过"
        assert saved["short_term"]["risk_feedback_state"]["latest_risk_verdict"] == "pass"
        assert saved["medium_term"]["risk_feedback_state"]["latest_risk_verdict"] == "pass"

    def test_intent_hoist_persists_debate_states_to_result_data(self):
        job_id = f"job-{uuid4().hex}"
        store = InMemoryJobStore()
        collector = MagicMock()
        collector.collect.return_value = {"market_data_context": {"daily": {"as_of": "2026-08-20"}}}
        saved_reports = []
        db = MagicMock()

        # query with single horizon invokes the hoist path
        request = main.AnalyzeRequest(
            symbol="600519.SH",
            trade_date="2026-08-20",
            horizons=["short"],
            query="分析 600519.SH 短线走势",
            selected_analysts=[],
        )

        def capture_create_report(**kwargs):
            saved_reports.append(kwargs)

        async def run_job():
            stream = main._stream_job_events(job_id)
            await stream.__anext__()
            task = asyncio.create_task(
                main._run_job_inner(job_id, request, stream_events=False, save_report=True)
            )
            async for chunk in stream:
                if "event: done" in chunk:
                    break
            await task
            await stream.aclose()

        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "_shared_data_collector", collector),
            patch.object(main, "TradingAgentsGraph", _FakeTradingGraphForJob),
            patch.object(main, "_build_runtime_config", return_value={}),
            patch.object(main, "_resolve_and_freeze_custom_prompts", return_value=({}, False)),
            patch.object(main, "get_db_ctx", return_value=nullcontext(db)),
            patch.object(report_service, "init_report"),
            patch.object(report_service, "update_report_partial"),
            patch.object(report_service, "extract_structured_data", return_value=None),
            patch.object(report_service, "create_report", side_effect=capture_create_report),
        ):
            asyncio.run(run_job())

        job = store.get_job(job_id)
        assert job["status"] == "completed"
        result_data = job["result"]

        # Check top-level hoisted fields
        assert result_data["investment_debate_state"]["judge_decision"] == "short 多头胜"
        assert result_data["risk_debate_state"]["judge_decision"] == "short 风控通过"

        # Check saved report in ReportDB
        assert len(saved_reports) == 1
        saved = saved_reports[0]["result_data"]
        assert saved["investment_debate_state"]["judge_decision"] == "short 多头胜"
        assert saved["risk_debate_state"]["judge_decision"] == "short 风控通过"

    def test_single_horizon_persists_debate_states_to_result_data(self):
        job_id = f"job-{uuid4().hex}"
        store = InMemoryJobStore()
        collector = MagicMock()
        collector.collect.return_value = {"market_data_context": {"daily": {"as_of": "2026-08-20"}}}
        saved_reports = []
        db = MagicMock()

        request = main.AnalyzeRequest(
            symbol="600519.SH",
            trade_date="2026-08-20",
            horizons=["short"],
            selected_analysts=[],
        )

        def capture_create_report(**kwargs):
            saved_reports.append(kwargs)

        async def run_job():
            stream = main._stream_job_events(job_id)
            await stream.__anext__()
            task = asyncio.create_task(
                main._run_job_inner(job_id, request, stream_events=True, save_report=True)
            )
            async for chunk in stream:
                if "event: done" in chunk:
                    break
            await task
            await stream.aclose()

        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "_shared_data_collector", collector),
            patch.object(main, "TradingAgentsGraph", _FakeTradingGraphForJob),
            patch.object(main, "_build_runtime_config", return_value={}),
            patch.object(main, "_resolve_and_freeze_custom_prompts", return_value=({}, False)),
            patch.object(main, "get_db_ctx", return_value=nullcontext(db)),
            patch.object(report_service, "init_report"),
            patch.object(report_service, "update_report_partial"),
            patch.object(report_service, "extract_structured_data", return_value=None),
            patch.object(report_service, "create_report", side_effect=capture_create_report),
        ):
            asyncio.run(run_job())

        job = store.get_job(job_id)
        assert job["status"] == "completed"
        result_data = job["result"]

        assert result_data["investment_debate_state"]["judge_decision"] == "short 多头胜"
        assert result_data["risk_debate_state"]["judge_decision"] == "short 风控通过"
        assert result_data["risk_feedback_state"]["latest_risk_verdict"] == "pass"

        assert len(saved_reports) == 1
        saved = saved_reports[0]["result_data"]
        assert saved["investment_debate_state"]["judge_decision"] == "short 多头胜"
        assert saved["risk_debate_state"]["judge_decision"] == "short 风控通过"
        assert saved["risk_feedback_state"]["latest_risk_verdict"] == "pass"


# ============================================================================
# DAV-210: Isolating rejected machine blocks from transcript history
# ============================================================================


def _make_investment_debate_state() -> dict:
    return {
        "history": "",
        "bull_history": "",
        "bear_history": "",
        "current_speaker": "",
        "current_response": "",
        "count": 0,
        "claims": [],
        "focus_claim_ids": [],
        "open_claim_ids": [],
        "resolved_claim_ids": [],
        "unresolved_claim_ids": [],
        "round_summary": "",
        "round_goal": debate_utils.default_round_goal("investment", 1),
        "claim_counter": 0,
    }


def _make_risk_debate_state() -> dict:
    return debate_utils.build_empty_risk_debate_state()


def _apply_debate_response(state: dict, raw_response: str, marker: str = "DEBATE_STATE") -> dict:
    if marker == "DEBATE_STATE":
        return debate_utils.update_debate_state_with_payload(
            state=state,
            raw_response=raw_response,
            speaker_label="Bull Analyst",
            speaker_key="Bull",
            stance="bullish",
            history_key="bull_history",
            marker="DEBATE_STATE",
            claim_prefix="INV",
            domain="investment",
            speaker_field="current_speaker",
            store_current_response=True,
        )
    else:
        return debate_utils.update_debate_state_with_payload(
            state=state,
            raw_response=raw_response,
            speaker_label="Aggressive Analyst",
            speaker_key="Aggressive",
            stance="aggressive",
            history_key="aggressive_history",
            marker="RISK_STATE",
            claim_prefix="RISK",
            domain="risk",
            speaker_field="latest_speaker",
            store_current_response=True,
        )


def test_1_malformed_json_block_quarantined_and_prose_kept_and_report_valid():
    """1. 合法正文 + malformed JSON块：正文保留，标签完整移除，count+1，claims不变，最终validate_report_machine_blocks()通过"""
    initial_state = _make_investment_debate_state()
    initial_claims = deepcopy(initial_state["claims"])
    raw_response = "多头论点：看好后市结构性行情。\n<!-- DEBATE_STATE: {\"new_claims\": [invalid json} -->"

    result = _apply_debate_response(initial_state, raw_response, "DEBATE_STATE")

    assert result["count"] == 1
    assert result["claims"] == initial_claims
    assert "DEBATE_STATE" not in result["history"]
    assert "<!--" not in result["history"]
    assert "多头论点：看好后市结构性行情。" in result["history"]
    assert "Bull Analyst: 多头论点：看好后市结构性行情。" in result["history"]
    assert "DEBATE_STATE" not in result["bull_history"]
    assert "DEBATE_STATE" not in result["current_response"]

    mock_report = {
        "final_trade_decision": "BUY",
        "investment_debate_state": result,
    }
    report_service.validate_report_machine_blocks(mock_report)


def test_2_trailing_prose_quarantined_and_both_prose_kept_and_payload_not_accepted():
    """2. 合法JSON块后有尾随正文（触发invalid_or_trailing_prose）：正文与尾随正文保留，机读注释删除，结构化payload不采纳"""
    initial_state = _make_investment_debate_state()
    payload = {
        "new_claims": [
            {
                "claim": "此claim不应被采纳",
                "evidence": ["无效证据"],
                "confidence": 0.8,
                "target_claim_ids": [],
            }
        ]
    }
    block = f"<!-- DEBATE_STATE: {json.dumps(payload, ensure_ascii=False)} -->"
    raw_response = f"前导分析：支撑位明确。\n{block}\n尾随正文：补充量能不足的风险分析。"

    result = _apply_debate_response(initial_state, raw_response, "DEBATE_STATE")

    assert result["count"] == 1
    assert result["claims"] == []
    assert "DEBATE_STATE" not in result["history"]
    assert "<!--" not in result["history"]
    assert "前导分析：支撑位明确。" in result["history"]
    assert "尾随正文：补充量能不足的风险分析。" in result["history"]
    assert "此claim不应被采纳" not in json.dumps(result["claims"], ensure_ascii=False)

    mock_report = {"investment_debate_state": result}
    report_service.validate_report_machine_blocks(mock_report)


def test_3_duplicate_same_tag_blocks_quarantined_and_not_in_history():
    """3. 重复同标签块：全部同标签注释隔离，不进入history"""
    initial_state = _make_investment_debate_state()
    payload1 = {"new_claims": [{"claim": "claim 1", "evidence": [], "confidence": 0.5, "target_claim_ids": []}]}
    payload2 = {"new_claims": [{"claim": "claim 2", "evidence": [], "confidence": 0.6, "target_claim_ids": []}]}
    block1 = f"<!-- DEBATE_STATE: {json.dumps(payload1, ensure_ascii=False)} -->"
    block2 = f"<!-- DEBATE_STATE: {json.dumps(payload2, ensure_ascii=False)} -->"
    raw_response = f"多方陈述。\n{block1}\n中间补充论点。\n{block2}\n总结。"

    result = _apply_debate_response(initial_state, raw_response, "DEBATE_STATE")

    assert result["count"] == 1
    assert result["claims"] == []
    assert "DEBATE_STATE" not in result["history"]
    assert "<!--" not in result["history"]
    assert "多方陈述。" in result["history"]
    assert "中间补充论点。" in result["history"]
    assert "总结。" in result["history"]

    mock_report = {"investment_debate_state": result}
    report_service.validate_report_machine_blocks(mock_report)


def test_4_missing_colon_quarantined_and_prose_kept():
    """4. 标签缺冒号"""
    initial_state = _make_investment_debate_state()
    raw_response = "多方分析。\n<!-- DEBATE_STATE {\"new_claims\": []} -->\n后续观点。"

    result = _apply_debate_response(initial_state, raw_response, "DEBATE_STATE")

    assert result["count"] == 1
    assert result["claims"] == []
    assert "DEBATE_STATE" not in result["history"]
    assert "<!--" not in result["history"]
    assert "多方分析。" in result["history"]
    assert "后续观点。" in result["history"]

    mock_report = {"investment_debate_state": result}
    report_service.validate_report_machine_blocks(mock_report)


def test_5_truncated_block_quarantined_and_preceding_prose_kept():
    """5. 截断块（无-->）：从标签起至末尾隔离，标签前正文保留"""
    initial_state = _make_investment_debate_state()
    raw_response = "多方核心论点已陈述完毕。\n<!-- DEBATE_STATE: {\"new_claims\": [{\"claim\": \"截断未完\""

    result = _apply_debate_response(initial_state, raw_response, "DEBATE_STATE")

    assert result["count"] == 1
    assert result["claims"] == []
    assert "DEBATE_STATE" not in result["history"]
    assert "<!--" not in result["history"]
    assert "多方核心论点已陈述完毕。" in result["history"]
    assert "截断未完" not in result["history"]

    mock_report = {"investment_debate_state": result}
    report_service.validate_report_machine_blocks(mock_report)


def test_6_risk_state_handled_with_identical_quarantine():
    """6. RISK_STATE同样处理"""
    initial_state = _make_risk_debate_state()
    raw_response = "风控分析：激进方认为下行风险有限。\n<!-- RISK_STATE: {\"new_claims\": [malformed json} -->"

    result = _apply_debate_response(initial_state, raw_response, "RISK_STATE")

    assert result["count"] == 1
    assert result["claims"] == []
    assert "RISK_STATE" not in result["history"]
    assert "<!--" not in result["history"]
    assert "风控分析：激进方认为下行风险有限。" in result["history"]
    assert "Aggressive Analyst: 风控分析：激进方认为下行风险有限。" in result["history"]
    assert "RISK_STATE" not in result["aggressive_history"]
    assert "RISK_STATE" not in result["current_response"]

    mock_report = {"risk_debate_state": result}
    report_service.validate_report_machine_blocks(mock_report)


def test_7_valid_machine_block_path_not_regressed():
    """7. 合法机读块现有路径不回归：正常解析claims/responded/resolved，history不含注释"""
    initial_state = _make_investment_debate_state()
    payload = {
        "new_claims": [
            {
                "claim": "消费升级主线持续验证",
                "evidence": ["三季报净利润增长超预期"],
                "confidence": 0.85,
                "target_claim_ids": [],
            }
        ],
        "responded_claim_ids": [],
        "resolved_claim_ids": [],
        "unresolved_claim_ids": [],
        "next_focus_claim_ids": ["INV-1"],
        "round_summary": "首轮多头建立核心 claim",
        "round_goal": "建立最核心的正反两方 claim",
    }
    block = f"<!-- DEBATE_STATE: {json.dumps(payload, ensure_ascii=False)} -->"
    raw_response = f"详细论证消费升级逻辑。\n{block}"

    result = _apply_debate_response(initial_state, raw_response, "DEBATE_STATE")

    assert result["count"] == 1
    assert len(result["claims"]) == 1
    assert result["claims"][0]["claim"] == "消费升级主线持续验证"
    assert result["claims"][0]["confidence"] == 0.85
    assert result["claims"][0]["claim_id"] == "INV-1"
    assert result["claim_counter"] == 1
    assert result["open_claim_ids"] == ["INV-1"]
    assert "DEBATE_STATE" not in result["history"]
    assert "<!--" not in result["history"]
    assert "详细论证消费升级逻辑。" in result["history"]

    mock_report = {"investment_debate_state": result}
    report_service.validate_report_machine_blocks(mock_report)


def test_8_direct_report_with_bad_machine_blocks_fails_closed():
    """8. 直接报告正文包含坏块时，report_service.validate_report_machine_blocks()仍必须抛错（fail-closed不变）"""
    bad_json_report = {
        "final_trade_decision": "决策正文\n<!-- DEBATE_STATE: {\"new_claims\": [bad json} -->",
    }
    with pytest.raises(ValueError, match="DEBATE_STATE machine block contains invalid JSON"):
        report_service.validate_report_machine_blocks(bad_json_report)

    missing_colon_report = {
        "final_trade_decision": "决策正文\n<!-- DEBATE_STATE {\"new_claims\": []} -->",
    }
    with pytest.raises(ValueError, match="DEBATE_STATE machine block must use ':' after the marker"):
        report_service.validate_report_machine_blocks(missing_colon_report)

    truncated_report = {
        "final_trade_decision": "决策正文\n<!-- DEBATE_STATE: {\"new_claims\": []",
    }
    with pytest.raises(ValueError, match="DEBATE_STATE machine block is truncated"):
        report_service.validate_report_machine_blocks(truncated_report)

    duplicate_report = {
        "final_trade_decision": "决策正文\n<!-- DEBATE_STATE: {} -->\n<!-- DEBATE_STATE: {} -->",
    }
    with pytest.raises(ValueError, match="DEBATE_STATE machine block must not be duplicated"):
        report_service.validate_report_machine_blocks(duplicate_report)

    risk_bad_report = {
        "risk_assessment": "风控报告\n<!-- RISK_STATE: {invalid} -->",
    }
    with pytest.raises(ValueError, match="RISK_STATE machine block contains invalid JSON"):
        report_service.validate_report_machine_blocks(risk_bad_report)
