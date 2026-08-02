"""Focused regression tests for the DAV-8 dual-horizon save contract."""

import asyncio
from contextlib import nullcontext
from unittest.mock import MagicMock, patch
from uuid import uuid4

from api import main
from api.job_store import InMemoryJobStore
from api.services import report_service


class _FakePropagator:
    def get_graph_args(self):
        return {}

    def create_initial_state(self, *_args, **kwargs):
        return {"horizon": kwargs.get("horizon", "short")}


class _FakeGraphStream:
    async def astream(self, init_state, **_kwargs):
        horizon = init_state["horizon"]
        yield {
            "company_of_interest": "600519.SH",
            "trade_date": "2026-07-31",
            "horizon": horizon,
            "final_trade_decision": f"{horizon} 结论：持有",
        }


class _FakeTradingGraph:
    def __init__(self, *_args, data_collector=None, **_kwargs):
        self.data_collector = data_collector
        self.propagator = _FakePropagator()
        self.graph = _FakeGraphStream()
        self.role_resolved_configs = {}
        self.quick_thinking_llm = object()

    def process_signal(self, _decision):
        return "HOLD"

    def _build_horizon_result(self, _horizon, state):
        return dict(state)


def _run_dual_save(structured_by_horizon):
    job_id = f"dav8-{uuid4().hex}"
    store = InMemoryJobStore()
    collector = MagicMock()
    collector.collect.return_value = {"market_data_context": {"source": "fixture"}}
    db = MagicMock()
    saved_reports = []
    request = main.AnalyzeRequest(
        symbol="600519.SH",
        trade_date="2026-07-31",
        horizons=["short", "medium"],
        selected_analysts=[],
    )

    def capture_create_report(**kwargs):
        saved_reports.append(kwargs)

    def structured_for(final_trade_decision, *_args, **_kwargs):
        horizon = "medium" if "medium" in final_trade_decision else "short"
        return structured_by_horizon[horizon]

    async def run_with_sse():
        stream = main._stream_job_events(job_id)
        assert (await stream.__anext__()).startswith("event: job.ready")
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
        patch.object(main, "TradingAgentsGraph", _FakeTradingGraph),
        patch.object(main, "_build_runtime_config", return_value={}),
        patch.object(main, "_resolve_and_freeze_custom_prompts", return_value=({}, False)),
        patch.object(main, "get_db_ctx", return_value=nullcontext(db)),
        patch.object(report_service, "init_report"),
        patch.object(report_service, "update_report_partial"),
        patch.object(report_service, "extract_structured_data", side_effect=structured_for),
        patch.object(report_service, "create_report", side_effect=capture_create_report),
    ):
        asyncio.run(run_with_sse())

    return store.get_job(job_id), saved_reports


def test_dual_horizon_save_propagates_aggregated_structured_fields():
    job, saved_reports = _run_dual_save(
        {
            "short": report_service.StructuredReport(
                data_gaps=[],
                not_applicable=False,
                falsification_conditions=["条件A：业绩不达预期"],
            ),
            "medium": report_service.StructuredReport(
                data_gaps=[],
                not_applicable=True,
                falsification_conditions=["条件A：业绩不达预期", "条件B：政策风险"],
            ),
        }
    )

    assert job["status"] == "completed"
    assert job["result"]["not_applicable"] is False
    assert job["result"]["not_applicable_by_horizon"] == {
        "short": False,
        "medium": True,
    }
    assert job["result"]["falsification_conditions"] == [
        "条件A：业绩不达预期",
        "条件B：政策风险",
    ]
    assert len(saved_reports) == 1
    assert saved_reports[0]["not_applicable"] is False
    assert saved_reports[0]["falsification_conditions"] == job["result"]["falsification_conditions"]


def test_dual_horizon_save_uses_safe_defaults_when_structured_fields_are_empty():
    job, saved_reports = _run_dual_save(
        {
            "short": report_service.StructuredReport(data_gaps=[]),
            "medium": report_service.StructuredReport(data_gaps=[]),
        }
    )

    assert job["status"] == "completed"
    assert job["result"]["not_applicable"] is False
    assert job["result"]["falsification_conditions"] == []
    assert len(saved_reports) == 1
    assert saved_reports[0]["not_applicable"] is False
    assert saved_reports[0]["falsification_conditions"] == []
