import asyncio
import inspect
from contextlib import nullcontext
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api import main
from api.database import Base, ReportDB
from api.job_store import InMemoryJobStore
from api.services import report_service


def test_normalize_analysis_horizons_preserves_explicit_dual_and_infers_only_strong_dual_language():
    assert main._normalize_analysis_horizons(["medium", "short", "medium"]) == ["short", "medium"]
    assert main._normalize_analysis_horizons(["short"], query="分析 600519.SH 短线和中线机会") == ["short", "medium"]
    assert main._normalize_analysis_horizons(["short"], query="分析 600519.SH 的短线机会") == ["short"]
    assert main._normalize_analysis_horizons(["unknown"], query=None) == ["short"]


def test_chat_extraction_contract_allows_explicit_dual_horizon():
    source = inspect.getsource(main._ai_extract_symbol_and_date)
    streaming_source = inspect.getsource(main._ai_extract_symbol_and_date_streaming)
    assert "短线与中线" in source or "short and medium" in source
    assert "短线与中线" in streaming_source or "short and medium" in streaming_source


class _FakePropagator:
    def __init__(self, horizon):
        self.horizon = horizon

    def get_graph_args(self):
        return {}

    def create_initial_state(self, *_args, **kwargs):
        state = {
            "horizon": kwargs.get("horizon", "short"),
            "market_data_context": kwargs.get("market_data_context"),
        }
        _FakeTradingGraph.initial_states.append(state)
        return state


class _FakeGraphStream:
    @staticmethod
    def _state(init_state):
        horizon = init_state["horizon"]
        return {
            "company_of_interest": "600519.SH",
            "trade_date": "2026-07-31",
            "horizon": horizon,
            "news_report": f"{horizon} report",
            "final_trade_decision": f"{horizon} decision",
            "market_data_context": init_state["market_data_context"],
            "analyst_traces": [{"horizon": horizon}],
        }

    def invoke(self, init_state, **_kwargs):
        horizon = init_state["horizon"]
        if horizon in _FakeTradingGraph.fail_horizons:
            raise RuntimeError("medium provider unavailable")
        return self._state(init_state)

    async def astream(self, init_state, **_kwargs):
        horizon = init_state["horizon"]
        _FakeTradingGraph.thread_ids.append(
            _kwargs["config"]["configurable"]["thread_id"]
        )
        if horizon in _FakeTradingGraph.fail_horizons:
            raise RuntimeError("medium provider unavailable")
        yield self._state(init_state)


class _FakeTradingGraph:
    collector = None
    fail_horizons = set()
    thread_ids = []
    initial_states = []

    def __init__(self, selected_analysts, data_collector, **_kwargs):
        self.data_collector = data_collector
        self.propagator = _FakePropagator("unknown")
        self.graph = _FakeGraphStream()
        self.role_resolved_configs = {}
        self.quick_thinking_llm = object()

    def process_signal(self, decision):
        return "BUY" if "short" in decision else "SELL"

    def _build_horizon_result(self, _horizon, state):
        return dict(state)


def _run_dual_job(
    *,
    fail_horizons=(),
    horizons=("short", "medium"),
    query=None,
    user_intent=None,
    stream_events=False,
):
    job_id = f"dual-{uuid4().hex}"
    store = InMemoryJobStore()
    collector = MagicMock()
    collector.collect.return_value = {"market_data_context": {"source": "fixture"}}
    saved_reports = []
    db = MagicMock()
    _FakeTradingGraph.collector = collector
    _FakeTradingGraph.fail_horizons = set(fail_horizons)
    _FakeTradingGraph.thread_ids = []
    _FakeTradingGraph.initial_states = []
    request = main.AnalyzeRequest(
        symbol="600519.SH",
        trade_date="2026-07-31",
        horizons=list(horizons),
        selected_analysts=[],
        query=query,
        user_intent=user_intent,
    )

    partial_updates = MagicMock()

    def capture_create_report(**kwargs):
        saved_reports.append(kwargs)

    def structured_for(final_trade_decision, *_args, **_kwargs):
        horizon = "short" if "short" in final_trade_decision else "medium"
        return report_service.StructuredReport(
            decision="BUY" if horizon == "short" else "SELL",
            confidence=80 if horizon == "short" else 60,
            probability=0.8 if horizon == "short" else 0.4,
            data_gaps=[f"{horizon} LLM gap"],
            not_applicable=horizon == "medium",
        )

    async def run_job():
        stream = main._stream_job_events(job_id)
        assert (await stream.__anext__()).startswith("event: job.ready")
        task = asyncio.create_task(
            main._run_job_inner(job_id, request, stream_events=stream_events, save_report=True)
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
            if "event: done" in chunk:
                break
        await task
        await stream.aclose()
        return chunks

    with (
        patch.object(main, "_job_store_instance", store),
        patch.object(main, "_shared_data_collector", collector),
        patch.object(main, "TradingAgentsGraph", _FakeTradingGraph),
        patch.object(main, "_build_runtime_config", return_value={}),
        patch.object(main, "_resolve_and_freeze_custom_prompts", return_value=({}, False)),
        patch.object(main, "get_db_ctx", return_value=nullcontext(db)),
        patch.object(report_service, "init_report"),
        patch.object(report_service, "update_report_partial", side_effect=partial_updates),
        patch.object(report_service, "extract_structured_data", side_effect=structured_for),
        patch.object(report_service, "create_report", side_effect=capture_create_report),
    ):
        chunks = asyncio.run(run_job())

    return store.get_job(job_id), saved_reports, chunks, collector, partial_updates


def test_dual_job_keeps_horizon_scoped_fields_and_propagates_result_sse_and_reportdb():
    job, saved_reports, chunks, collector, _partial_updates = _run_dual_job()

    assert job["status"] == "completed"
    result = job["result"]
    assert result["status"] == "completed"
    assert result["horizon_status"] == {"short": "completed", "medium": "completed"}
    assert result["short_term"]["direction"] == "BUY"
    assert result["medium_term"]["direction"] == "SELL"
    assert result["short_term"]["decision"] == "BUY"
    assert result["medium_term"]["decision"] == "SELL"
    assert result["short_term"]["not_applicable"] is False
    assert result["medium_term"]["not_applicable"] is True
    assert "direction" not in result
    assert "confidence" not in result
    assert "decision" not in result
    assert result["data_gaps"] == ["short LLM gap", "medium LLM gap"]
    assert result["short_term"]["market_data_context"] == {"source": "fixture"}
    assert result["medium_term"]["market_data_context"] == {"source": "fixture"}
    assert result["market_data_context"] == {
        "short": {"source": "fixture"},
        "medium": {"source": "fixture"},
    }
    assert len(_FakeTradingGraph.thread_ids) == 2
    assert {thread_id.rsplit("_", 1)[-1] for thread_id in _FakeTradingGraph.thread_ids} == {
        "short",
        "medium",
    }
    assert all(thread_id.startswith("dual-") for thread_id in _FakeTradingGraph.thread_ids)
    assert saved_reports[0]["result_data"] == result
    assert saved_reports[0]["decision"] is None
    assert saved_reports[0]["data_gaps"] == result["data_gaps"]
    assert saved_reports[0]["not_applicable"] is False
    assert collector.collect.call_args.kwargs["horizons"] == ["short", "medium"]
    assert any('"mode": "dual_horizon"' in chunk for chunk in chunks)
    assert any('"data_gaps": ["short LLM gap", "medium LLM gap"]' in chunk for chunk in chunks)


def test_dual_job_allows_partial_result_with_failed_horizon_status_and_impact():
    job, saved_reports, chunks, _collector, _partial_updates = _run_dual_job(fail_horizons=("medium",))

    assert job["status"] == "completed"
    result = job["result"]
    assert result["status"] == "partial"
    assert result["horizon_status"] == {"short": "completed", "medium": "failed"}
    assert result["failed_horizons"] == ["medium"]
    failed = result["medium_term"]
    assert failed["status"] == "failed"
    assert failed["horizon"] == "medium"
    assert failed["not_applicable"] is None
    assert failed["impact"]
    assert "medium" in result["data_gaps"][0] or result["data_gaps"]
    assert saved_reports[0]["result_data"]["medium_term"]["status"] == "failed"
    assert saved_reports[0]["decision"] is None
    assert saved_reports[0]["not_applicable"] is False
    assert any('"status": "partial"' in chunk for chunk in chunks)
    assert any("agent.horizon_failed" in chunk for chunk in chunks)


@pytest.mark.parametrize("stream_events", [False, True])
def test_structured_direct_medium_only_passes_medium_horizon_and_context(stream_events):
    job, saved_reports, _chunks, _collector, partial_updates = _run_dual_job(
        horizons=("medium",),
        stream_events=stream_events,
    )

    assert job["status"] == "completed"
    assert _FakeTradingGraph.initial_states[0]["horizon"] == "medium"
    assert _FakeTradingGraph.initial_states[0]["market_data_context"] == {"source": "fixture"}
    assert saved_reports[0]["result_data"]["market_data_context"] == {"source": "fixture"}


def test_chat_single_horizon_restores_incremental_reportdb_updates():
    user_intent = {
        "ticker": "600519.SH",
        "horizons": ["short"],
        "focus_areas": [],
        "specific_questions": [],
    }
    job, _saved_reports, _chunks, _collector, partial_updates = _run_dual_job(
        horizons=("short",),
        query="分析 600519.SH 短线机会",
        user_intent=user_intent,
    )

    assert job["status"] == "completed"
    chunk_calls = [
        call
        for call in partial_updates.call_args_list
        if "news_report" in call.kwargs or "final_trade_decision" in call.kwargs
    ]
    assert chunk_calls
    assert all("status" not in call.kwargs for call in chunk_calls)


def test_dual_result_round_trips_nested_horizons_through_reportdb():
    _job, saved_reports, _chunks, _collector, _partial_updates = _run_dual_job()
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = session_factory()
    try:
        payload = dict(saved_reports[0])
        payload.pop("db", None)
        payload.pop("report_id", None)
        report = report_service.create_report(db=db, **payload)
        persisted = db.query(ReportDB).filter(ReportDB.id == report.id).one()
        assert persisted.result_data["short_term"]["status"] == "completed"
        assert persisted.result_data["medium_term"]["status"] == "completed"
        assert persisted.result_data["short_term"]["not_applicable"] is False
        assert persisted.result_data["medium_term"]["not_applicable"] is True
        assert persisted.decision is None
        assert persisted.direction is None
        assert persisted.confidence is None
    finally:
        db.close()
        engine.dispose()


def test_http_analyze_jobs_and_reports_round_trip_partial_contract():
    client = TestClient(main.app, raise_server_exceptions=False)
    store = InMemoryJobStore()
    captured_requests = []
    result = {
        "mode": "dual_horizon",
        "status": "partial",
        "horizon_status": {"short": "completed", "medium": "failed"},
        "failed_horizons": ["medium"],
        "short_term": {
            "horizon": "short",
            "status": "completed",
            "direction": "BUY",
            "confidence": 80,
            "decision": "BUY",
            "not_applicable": False,
            "market_data_context": {"source": "fixture"},
        },
        "medium_term": {
            "horizon": "medium",
            "status": "failed",
            "error": "medium provider unavailable",
            "impact": "medium horizon is unavailable",
            "not_applicable": None,
        },
        "market_data_context": {"short": {"source": "fixture"}},
        "data_gaps": ["medium unavailable"],
    }

    async def fake_run_job(job_id, analyze_request, *_args, **_kwargs):
        captured_requests.append(analyze_request)
        main._set_job(
            job_id,
            status="completed",
            result=result,
            decision=None,
            error=None,
            finished_at=main._utcnow_iso(),
        )
        main._emit_job_event(job_id, "job.completed", {"job_id": job_id, "result": result})

    with (
        patch.object(main, "_job_store_instance", store),
        patch.object(main, "_compose_analysis_user_context", return_value={}),
        patch.object(main, "_run_job", side_effect=fake_run_job),
        patch.object(main, "_get_reverse_stock_map", return_value={}),
    ):
        analyze_response = client.post(
            "/v1/analyze",
            json={
                "symbol": "600519.SH",
                "trade_date": "2026-07-31",
                "horizons": ["short", "medium"],
                "dry_run": True,
            },
        )
        assert analyze_response.status_code == 200
        job_id = analyze_response.json()["job_id"]
        assert captured_requests[0].horizons == ["short", "medium"]

        status_response = client.get(f"/v1/jobs/{job_id}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "completed"

        result_response = client.get(f"/v1/jobs/{job_id}/result")
        assert result_response.status_code == 200
        assert result_response.json()["status"] == "completed"
        assert result_response.json()["result"]["status"] == "partial"
        assert result_response.json()["result"]["horizon_status"]["medium"] == "failed"
        assert result_response.json()["result"]["failed_horizons"] == ["medium"]

        report_response = client.post(
            "/v1/reports",
            json={
                "symbol": "600519.SH",
                "trade_date": "2026-07-31",
                "result_data": result,
                "data_gaps": result["data_gaps"],
            },
        )
        assert report_response.status_code == 200
        report_id = report_response.json()["id"]

        detail_response = client.get(f"/v1/reports/{report_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["result_data"]["status"] == "partial"
        assert detail["result_data"]["short_term"]["market_data_context"] == {"source": "fixture"}
        assert detail["result_data"]["failed_horizons"] == ["medium"]


@pytest.mark.parametrize("stream", [False, True])
def test_chat_natural_language_dual_horizon_is_forwarded_for_stream_and_nonstream(stream):
    captured = []
    user = MagicMock(id="user-1")
    request = main.ChatCompletionRequest(
        messages=[{"role": "user", "content": "分析 600519.SH 短线和中线机会"}],
        stream=stream,
        dry_run=True,
    )

    async def fake_run_job(job_id, analyze_request, *_args, **_kwargs):
        captured.append(analyze_request)
        main._set_job(job_id, status="completed", decision="DRY_RUN", result={})
        main._emit_job_event(job_id, "job.completed", {"job_id": job_id, "result": {}})

    extraction = ("600519.SH", "2026-07-31", ["short", "medium"], [], [], {})

    async def run():
        with (
            patch.object(main, "_build_runtime_config", return_value={}),
            patch.object(main, "_compose_analysis_user_context", return_value={}),
            patch.object(main, "_job_store_instance", InMemoryJobStore()),
            patch.object(main, "_ai_extract_symbol_and_date", return_value=extraction),
            patch.object(main, "_ai_extract_symbol_and_date_streaming", return_value=extraction),
            patch.object(main, "_run_job", side_effect=fake_run_job),
        ):
            response = await main.chat_completions(request, current_user=user)
            if stream:
                body = "".join([chunk async for chunk in response.body_iterator])
                assert "job.completed" in body
            else:
                assert response["choices"][0]["finish_reason"] == "stop"

    asyncio.run(run())
    assert captured and captured[0].horizons == ["short", "medium"]
