"""Tests for debate state and risk feedback state persistence and hoisting (DAV-205)."""

import asyncio
from contextlib import nullcontext
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from api import main
from api.job_store import InMemoryJobStore
from api.services import report_service
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
