from __future__ import annotations

import asyncio
from contextlib import nullcontext
import threading
from unittest.mock import MagicMock, patch
from uuid import uuid4

from api.job_store import InMemoryJobStore
from api import main
from api.services import report_service


def _request() -> main.AnalyzeRequest:
    return main.AnalyzeRequest(symbol="600519.SH", trade_date="2026-07-10")


def test_soft_timeout_emits_overtime_then_allows_completion():
    store = InMemoryJobStore()
    events: list[tuple[str, dict]] = []
    store.set_job("job-1", status="running", error="stale error")

    def capture_event(job_id: str, event: str, data: dict) -> None:
        events.append((event, data))
        store.emit_event(job_id, event, data)

    async def scenario() -> None:
        release_inner = asyncio.Event()

        async def fake_inner(job_id, *_args, **_kwargs):
            await release_inner.wait()
            main._set_job(
                job_id,
                status="completed",
                result={"decision": "BUY"},
                error=None,
                overtime=False,
                overtime_at=None,
                finished_at=main._utcnow_iso(),
            )
            main._emit_job_event(job_id, "job.completed", {"job_id": job_id})

        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "_JOB_TIMEOUT", 0.01),
            patch.object(main, "_JOB_HARD_TIMEOUT", 1),
            patch.object(main, "_run_job_inner", side_effect=fake_inner),
            patch.object(main, "_emit_job_event", side_effect=capture_event),
        ):
            wrapper = asyncio.create_task(main._run_job("job-1", _request()))
            await asyncio.sleep(0.02)
            assert not wrapper.done(), "soft deadline must not release the wrapper/scheduler slot"
            assert [event for event, _ in events] == ["job.overtime"]
            assert store.get_job("job-1")["status"] == "running"
            release_inner.set()
            await wrapper

    asyncio.run(scenario())

    job = store.get_job("job-1")
    assert [event for event, _ in events] == ["job.overtime", "job.completed"]
    assert job["status"] == "completed"
    assert job["error"] is None
    assert job["overtime"] is False
    assert events[0][1]["elapsed_seconds"] >= 0.01
    assert events[0][1]["soft_timeout_seconds"] == 0.01


def test_job_finishing_before_deadline_does_not_emit_overtime():
    store = InMemoryJobStore()
    events: list[str] = []

    async def fake_inner(job_id, *_args, **_kwargs):
        main._set_job(job_id, status="completed", error=None, overtime=False)
        main._emit_job_event(job_id, "job.completed", {"job_id": job_id})

    async def scenario() -> None:
        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "_JOB_TIMEOUT", 1),
            patch.object(main, "_JOB_HARD_TIMEOUT", 2),
            patch.object(main, "_run_job_inner", side_effect=fake_inner),
            patch.object(main, "_emit_job_event", side_effect=lambda _j, event, _d: events.append(event)),
        ):
            await main._run_job("job-fast", _request())

    asyncio.run(scenario())
    assert events == ["job.completed"]


def test_completed_task_is_not_overwritten_by_stale_wait_snapshot():
    store = InMemoryJobStore()
    events: list[str] = []

    async def fake_inner(job_id, *_args, **_kwargs):
        main._set_job(job_id, status="completed", error=None, overtime=False)
        main._emit_job_event(job_id, "job.completed", {"job_id": job_id})

    async def stale_wait_snapshot(tasks, **_kwargs):
        task = next(iter(tasks))
        await task
        return set(), {task}

    async def scenario() -> None:
        # main.asyncio is the process-wide asyncio module.  This patch is
        # intentionally scoped to the context so it cannot leak to other tests.
        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "_JOB_TIMEOUT", 1),
            patch.object(main, "_JOB_HARD_TIMEOUT", 2),
            patch.object(main, "_run_job_inner", side_effect=fake_inner),
            patch.object(main.asyncio, "wait", side_effect=stale_wait_snapshot),
            patch.object(main, "_emit_job_event", side_effect=lambda _j, event, _d: events.append(event)),
        ):
            await main._run_job("job-stale-wait", _request())

    asyncio.run(scenario())
    assert events == ["job.completed"]
    assert store.get_job("job-stale-wait")["status"] == "completed"


def test_zero_timeout_disables_warning_but_still_waits_for_completion():
    store = InMemoryJobStore()
    events: list[str] = []

    async def fake_inner(job_id, *_args, **_kwargs):
        await asyncio.sleep(0.01)
        main._set_job(job_id, status="completed", error=None, overtime=False)
        main._emit_job_event(job_id, "job.completed", {"job_id": job_id})

    async def scenario() -> None:
        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "_JOB_TIMEOUT", 0),
            patch.object(main, "_JOB_HARD_TIMEOUT", 1),
            patch.object(main, "_run_job_inner", side_effect=fake_inner),
            patch.object(main, "_emit_job_event", side_effect=lambda _j, event, _d: events.append(event)),
        ):
            await main._run_job("job-no-warning", _request())

    asyncio.run(scenario())
    assert store.get_job("job-no-warning")["status"] == "completed"
    assert events == ["job.completed"]


def test_unexpected_inner_exception_marks_job_failed():
    store = InMemoryJobStore()
    events: list[tuple[str, dict]] = []
    store.set_job("job-2", status="running", error=None)

    async def failing_inner(*_args, **_kwargs):
        raise RuntimeError("model unavailable")

    def capture_event(_job_id: str, event: str, data: dict) -> None:
        events.append((event, data))

    async def scenario() -> None:
        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "_JOB_TIMEOUT", 1),
            patch.object(main, "_JOB_HARD_TIMEOUT", 2),
            patch.object(main, "_run_job_inner", side_effect=failing_inner),
            patch.object(main, "_emit_job_event", side_effect=capture_event),
            patch.object(main, "get_db_ctx", side_effect=RuntimeError("db disabled")),
        ):
            await main._run_job("job-2", _request())

    asyncio.run(scenario())

    job = store.get_job("job-2")
    assert [event for event, _ in events] == ["job.failed"]
    assert job["status"] == "failed"
    assert job["error"] == "RuntimeError: model unavailable"
    assert job["overtime"] is False


def test_overtime_job_that_later_raises_finishes_as_failed():
    store = InMemoryJobStore()
    events: list[str] = []

    async def failing_inner(*_args, **_kwargs):
        await asyncio.sleep(0.02)
        raise RuntimeError("late failure")

    async def scenario() -> None:
        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "_JOB_TIMEOUT", 0.01),
            patch.object(main, "_JOB_HARD_TIMEOUT", 1),
            patch.object(main, "_run_job_inner", side_effect=failing_inner),
            patch.object(main, "_emit_job_event", side_effect=lambda _j, event, _d: events.append(event)),
            patch.object(main, "get_db_ctx", side_effect=RuntimeError("db disabled")),
        ):
            await main._run_job("job-late-failure", _request())

    asyncio.run(scenario())
    job = store.get_job("job-late-failure")
    assert events == ["job.overtime", "job.failed"]
    assert job["status"] == "failed"
    assert job["error"] == "RuntimeError: late failure"
    assert job["overtime"] is False
    assert job["overtime_at"] is None


def test_hard_timeout_releases_wrapper_and_ignores_late_thread_result():
    store = InMemoryJobStore()
    events: list[tuple[str, dict]] = []
    release_thread = threading.Event()

    async def blocked_inner(job_id, *_args, **_kwargs):
        await asyncio.to_thread(release_thread.wait)
        main._set_job(job_id, status="completed", error=None, overtime=False)
        main._emit_job_event(job_id, "job.completed", {"job_id": job_id})

    async def scenario() -> None:
        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "_JOB_TIMEOUT", 0.01),
            patch.object(main, "_JOB_HARD_TIMEOUT", 0.03),
            patch.object(main, "_run_job_inner", side_effect=blocked_inner),
            patch.object(main, "_emit_job_event", side_effect=lambda _j, event, data: events.append((event, data))),
            patch.object(main, "get_db_ctx", side_effect=RuntimeError("db disabled")),
        ):
            await main._run_job("job-hard-timeout", _request())
            assert store.get_job("job-hard-timeout")["status"] == "failed"
            release_thread.set()
            await asyncio.sleep(0.02)

    try:
        asyncio.run(scenario())
    finally:
        release_thread.set()

    job = store.get_job("job-hard-timeout")
    assert [event for event, _ in events] == ["job.overtime", "job.failed"]
    assert job["status"] == "failed"
    assert "硬性运行上限" in job["error"]
    assert job["overtime"] is False
    assert events[-1][1]["hard_timeout_seconds"] == 0.03


def test_report_save_failure_is_raised_to_the_job_failure_boundary():
    def failing_save():
        raise OSError("disk full")

    async def scenario() -> None:
        try:
            await main._save_report_or_raise("job-db", failing_save, stage="save")
        except RuntimeError as exc:
            assert "Failed to save report for job job-db" in str(exc)
            assert isinstance(exc.__cause__, OSError)
        else:
            raise AssertionError("report persistence errors must not be swallowed")

    asyncio.run(scenario())

class _FakePropagator:
    def create_initial_state(self, *args, **kwargs):
        return {
            "company_of_interest": "600519.SH",
            "trade_date": "2026-08-20",
            "horizon": "short",
            "final_trade_decision": "BUY",
        }

    def get_graph_args(self):
        return {}


class _FakeGraphStream:
    def __init__(self, exc: Exception | None = None):
        self.exc = exc

    async def astream(self, init_state, **_kwargs):
        yield {
            "company_of_interest": "600519.SH",
            "trade_date": "2026-08-20",
            "horizon": "short",
            "market_report": "market report content",
        }
        if self.exc is not None:
            raise self.exc
        yield {
            "final_trade_decision": "BUY",
            "investment_debate_state": {"judge_decision": "多头胜", "count": 6},
            "risk_debate_state": {"judge_decision": "风控通过", "count": 9},
        }


class _FakeTradingGraphForStream:
    def __init__(self, exc: Exception | None = None):
        self.data_collector = MagicMock()
        self.data_collector.collect.return_value = {}
        self.propagator = _FakePropagator()
        self.graph = _FakeGraphStream(exc)
        self.role_resolved_configs = {}
        self.quick_thinking_llm = object()
        self.config = {}

    def process_signal(self, decision):
        return "BUY"


def test_single_horizon_astream_quota_error_fails_job_and_report():
    store = InMemoryJobStore()
    events: list[tuple[str, dict]] = []
    job_id = f"job-{uuid4().hex}"
    db = MagicMock()
    marked_failed: list[tuple[str, str]] = []

    def capture_mark_failed(_db, rep_id, err):
        marked_failed.append((rep_id, err))

    def capture_event(j_id: str, event: str, data: dict) -> None:
        events.append((event, data))
        store.emit_event(j_id, event, data)

    fake_graph = _FakeTradingGraphForStream(RuntimeError("insufficient_quota: 403 Forbidden"))
    mock_create_report = MagicMock()
    request = main.AnalyzeRequest(
        symbol="600519.SH",
        trade_date="2026-08-20",
        horizons=["short"],
        selected_analysts=["market"],
    )

    async def scenario():
        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "TradingAgentsGraph", return_value=fake_graph),
            patch.object(main, "_resolve_and_freeze_custom_prompts", return_value=({}, False)),
            patch.object(main, "get_db_ctx", return_value=nullcontext(db)),
            patch.object(report_service, "init_report"),
            patch.object(report_service, "update_report_partial"),
            patch.object(report_service, "mark_report_failed", side_effect=capture_mark_failed),
            patch.object(report_service, "create_report", mock_create_report),
            patch.object(main, "_emit_job_event", side_effect=capture_event),
        ):
            await main._run_job_inner(job_id, request, stream_events=True, save_report=True)

    asyncio.run(scenario())

    job = store.get_job(job_id)
    assert job["status"] == "failed"
    assert "insufficient_quota: 403 Forbidden" in job["error"]
    assert "RuntimeError" in job["error"]
    assert any(event == "job.failed" for event, _ in events)
    assert not any(event == "job.completed" for event, _ in events)
    assert len(marked_failed) == 1
    assert marked_failed[0][0] == job_id
    assert "insufficient_quota: 403 Forbidden" in marked_failed[0][1]
    assert "RuntimeError" in marked_failed[0][1]
    mock_create_report.assert_not_called()


def test_single_horizon_astream_connection_error_fails_job_and_report():
    store = InMemoryJobStore()
    events: list[tuple[str, dict]] = []
    job_id = f"job-{uuid4().hex}"
    db = MagicMock()
    marked_failed: list[tuple[str, str]] = []

    def capture_mark_failed(_db, rep_id, err):
        marked_failed.append((rep_id, err))

    def capture_event(j_id: str, event: str, data: dict) -> None:
        events.append((event, data))
        store.emit_event(j_id, event, data)

    fake_graph = _FakeTradingGraphForStream(ConnectionError("connection reset by peer"))
    mock_create_report = MagicMock()
    request = main.AnalyzeRequest(
        symbol="600519.SH",
        trade_date="2026-08-20",
        horizons=["short"],
        selected_analysts=["market"],
    )

    async def scenario():
        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "TradingAgentsGraph", return_value=fake_graph),
            patch.object(main, "_resolve_and_freeze_custom_prompts", return_value=({}, False)),
            patch.object(main, "get_db_ctx", return_value=nullcontext(db)),
            patch.object(report_service, "init_report"),
            patch.object(report_service, "update_report_partial"),
            patch.object(report_service, "mark_report_failed", side_effect=capture_mark_failed),
            patch.object(report_service, "create_report", mock_create_report),
            patch.object(main, "_emit_job_event", side_effect=capture_event),
        ):
            await main._run_job_inner(job_id, request, stream_events=True, save_report=True)

    asyncio.run(scenario())

    job = store.get_job(job_id)
    assert job["status"] == "failed"
    assert "connection reset by peer" in job["error"]
    assert "ConnectionError" in job["error"]
    assert any(event == "job.failed" for event, _ in events)
    assert not any(event == "job.completed" for event, _ in events)
    assert len(marked_failed) == 1
    assert marked_failed[0][0] == job_id
    assert "connection reset by peer" in marked_failed[0][1]
    assert "ConnectionError" in marked_failed[0][1]
    mock_create_report.assert_not_called()


def test_single_horizon_astream_success_path_completes_normally():
    store = InMemoryJobStore()
    events: list[tuple[str, dict]] = []
    job_id = f"job-{uuid4().hex}"
    db = MagicMock()
    saved_reports: list[dict] = []

    def capture_create_report(**kwargs):
        saved_reports.append(kwargs)

    def capture_event(j_id: str, event: str, data: dict) -> None:
        events.append((event, data))
        store.emit_event(j_id, event, data)

    fake_graph = _FakeTradingGraphForStream(exc=None)
    mock_create_report = MagicMock()
    request = main.AnalyzeRequest(
        symbol="600519.SH",
        trade_date="2026-08-20",
        horizons=["short"],
        selected_analysts=["market"],
    )

    async def scenario():
        with (
            patch.object(main, "_job_store_instance", store),
            patch.object(main, "TradingAgentsGraph", return_value=fake_graph),
            patch.object(main, "_resolve_and_freeze_custom_prompts", return_value=({}, False)),
            patch.object(main, "get_db_ctx", return_value=nullcontext(db)),
            patch.object(report_service, "init_report"),
            patch.object(report_service, "update_report_partial"),
            patch.object(report_service, "extract_structured_data", return_value=None),
            patch.object(report_service, "create_report", side_effect=capture_create_report),
            patch.object(main, "_emit_job_event", side_effect=capture_event),
        ):
            await main._run_job_inner(job_id, request, stream_events=True, save_report=True)

    asyncio.run(scenario())

    job = store.get_job(job_id)
    assert job["status"] == "completed"
    assert job["error"] is None
    assert any(event == "job.completed" for event, _ in events)
    assert not any(event == "job.failed" for event, _ in events)
    assert len(saved_reports) == 1
    assert saved_reports[0]["symbol"] == "600519.SH"
