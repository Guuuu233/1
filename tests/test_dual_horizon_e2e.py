"""Dual-horizon E2E integration tests for the full propagation chain (DAV-9).

Covers the three required scenarios for ``request.horizons=["short", "medium"]``:

1. Full-chain propagation: job result -> ``job.completed`` SSE -> ReportDB save.
2. Partial failure: one horizon raises while the other horizon's result and
   data_gaps still propagate correctly (job completes with status "partial").
3. All horizons fail: the ``RuntimeError`` raised inside ``_run_job_inner``
   surfaces as a failed job with the aggregated error message.

Assertions follow the DAV-38 contract: the dual-horizon result carries
aggregated top-level ``not_applicable`` (all completed horizons) and
``falsification_conditions`` (deduped union), and ``_save_dual_report_sync``
propagates them to ``create_report``.
"""
import asyncio
from contextlib import nullcontext
from unittest.mock import MagicMock, patch
from uuid import uuid4

from api import main
from api.job_store import InMemoryJobStore
from api.services import report_service


class _FakePropagator:
    """Minimal propagator that threads horizon + market_data_context through."""

    def get_graph_args(self):
        return {}

    def create_initial_state(self, *_args, **kwargs):
        return {
            "horizon": kwargs.get("horizon", "short"),
            "market_data_context": kwargs.get("market_data_context"),
        }


class _FakeGraphStream:
    """Yields a per-horizon final state; raises when the horizon is marked failed."""

    fail_horizons = set()
    empty_horizons = set()

    @staticmethod
    def _state(init_state):
        horizon = init_state["horizon"]
        return {
            "company_of_interest": "600519.SH",
            "trade_date": "2026-07-31",
            "horizon": horizon,
            "news_report": (
                "正常无重大新闻。\n"
                f"- 【数据获取失败】{horizon} horizon 新闻接口超时"
            ),
            "final_trade_decision": f"{horizon} 结论：持有",
            "market_data_context": init_state["market_data_context"],
            "analyst_traces": [{"horizon": horizon}],
        }

    async def astream(self, init_state, **_kwargs):
        horizon = init_state["horizon"]
        if horizon in _FakeGraphStream.fail_horizons:
            raise RuntimeError(f"{horizon} provider unavailable")
        if horizon in _FakeGraphStream.empty_horizons:
            return
        yield self._state(init_state)


class _FakeTradingGraph:
    """Fake TradingAgentsGraph used for both the outer graph and per-horizon graphs."""

    fail_horizons = set()
    empty_horizons = set()
    build_fail_horizons = set()

    def __init__(self, *_args, data_collector=None, **_kwargs):
        self.data_collector = data_collector
        self.propagator = _FakePropagator()
        self.graph = _FakeGraphStream()
        self.role_resolved_configs = {}
        self.quick_thinking_llm = object()

    def process_signal(self, _decision):
        return "HOLD"

    def _build_horizon_result(self, horizon, state):
        if horizon in _FakeTradingGraph.build_fail_horizons:
            raise RuntimeError(f"{horizon} build error")
        return dict(state)


def _run_dual_horizon_job(*, fail_horizons=(), empty_horizons=(), build_fail_horizons=(), horizons=("short", "medium")):
    """Run the full dual-horizon job path with fakes, returning observable state."""
    job_id = f"dual-e2e-{uuid4().hex}"
    store = InMemoryJobStore()
    collector = MagicMock()
    collector.collect.return_value = {"market_data_context": {"source": "fixture"}}
    saved_reports = []
    db = MagicMock()
    _FakeTradingGraph.fail_horizons = set(fail_horizons)
    _FakeGraphStream.fail_horizons = set(fail_horizons)
    _FakeTradingGraph.empty_horizons = set(empty_horizons)
    _FakeGraphStream.empty_horizons = set(empty_horizons)
    _FakeTradingGraph.build_fail_horizons = set(build_fail_horizons)

    request = main.AnalyzeRequest(
        symbol="600519.SH",
        trade_date="2026-07-31",
        horizons=list(horizons),
        selected_analysts=[],
        query=None,
        user_intent=None,
    )

    def capture_create_report(**kwargs):
        saved_reports.append(kwargs)

    def structured_for(final_trade_decision, *_args, **_kwargs):
        horizon = next((h for h in horizons if h in final_trade_decision), "short")
        return report_service.StructuredReport(
            decision="BUY" if horizon == "short" else "SELL",
            confidence=80 if horizon == "short" else 60,
            probability=0.8 if horizon == "short" else 0.4,
            data_gaps=[f"模型补充：{horizon} 数据不完整"],
            falsification_conditions=[f"条件：{horizon} 业绩不达预期"],
            not_applicable=(horizon == "medium"),
        )

    async def run_with_sse():
        stream = main._stream_job_events(job_id)
        assert (await stream.__anext__()).startswith("event: job.ready")
        task = asyncio.create_task(
            main._run_job_inner(job_id, request, stream_events=False, save_report=True)
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
        patch.object(
            main,
            "TradingAgentsGraph",
            side_effect=lambda **_kwargs: _FakeTradingGraph(**_kwargs),
        ),
        patch.object(main, "_build_runtime_config", return_value={}),
        patch.object(main, "_resolve_and_freeze_custom_prompts", return_value=({}, False)),
        patch.object(main, "get_db_ctx", return_value=nullcontext(db)),
        patch.object(report_service, "init_report"),
        patch.object(report_service, "update_report_partial"),
        patch.object(report_service, "extract_structured_data", side_effect=structured_for),
        patch.object(report_service, "create_report", side_effect=capture_create_report),
    ):
        chunks = asyncio.run(run_with_sse())

    return store.get_job(job_id), saved_reports, chunks


def test_dual_horizon_propagates_both_horizons_to_result_sse_and_reportdb():
    """Scenario 1: full dual-horizon chain propagates result, SSE and ReportDB."""
    job, saved_reports, chunks = _run_dual_horizon_job()

    assert job["status"] == "completed"
    result = job["result"]
    assert result["mode"] == "dual_horizon"
    assert result["status"] == "completed"
    assert result["requested_horizons"] == ["short", "medium"]
    assert result["horizon_status"] == {"short": "completed", "medium": "completed"}
    assert result["short_term"]["status"] == "completed"
    assert result["medium_term"]["status"] == "completed"

    expected_gaps = [
        "【数据获取失败】short horizon 新闻接口超时",
        "模型补充：short 数据不完整",
        "【数据获取失败】medium horizon 新闻接口超时",
        "模型补充：medium 数据不完整",
    ]
    assert result["data_gaps"] == expected_gaps
    # DAV-38 contract: report-level not_applicable requires all completed
    # horizons to be not_applicable, while per-horizon values stay visible.
    assert result["not_applicable"] is False
    assert result["not_applicable_by_horizon"] == {"short": False, "medium": True}
    assert result["falsification_conditions"] == [
        "条件：short 业绩不达预期",
        "条件：medium 业绩不达预期",
    ]
    assert result["short_term"]["not_applicable"] is False
    assert result["medium_term"]["not_applicable"] is True

    assert len(saved_reports) == 1
    assert saved_reports[0]["result_data"] == result
    assert saved_reports[0]["data_gaps"] == expected_gaps
    assert saved_reports[0]["not_applicable"] is False
    assert saved_reports[0]["falsification_conditions"] == result["falsification_conditions"]

    assert any('"mode": "dual_horizon"' in chunk for chunk in chunks)
    assert any('"data_gaps": ["【数据获取失败】short horizon 新闻接口超时"' in chunk for chunk in chunks)
    assert any('"data_gaps": ["【数据获取失败】medium horizon 新闻接口超时"' in chunk for chunk in chunks)


def test_dual_horizon_partial_failure_keeps_other_horizon_result_and_gaps():
    """Scenario 2: one horizon fails; the other horizon's result/gaps survive."""
    job, saved_reports, chunks = _run_dual_horizon_job(fail_horizons=("medium",))

    assert job["status"] == "completed"
    result = job["result"]
    assert result["mode"] == "dual_horizon"
    assert result["status"] == "partial"
    assert result["horizon_status"] == {"short": "completed", "medium": "failed"}
    assert result["failed_horizons"] == ["medium"]

    failed = result["medium_term"]
    assert failed["status"] == "failed"
    assert failed["horizon"] == "medium"
    assert failed["error"]
    assert failed["not_applicable"] is None

    short = result["short_term"]
    assert short["status"] == "completed"
    assert short["data_gaps"] == [
        "【数据获取失败】short horizon 新闻接口超时",
        "模型补充：short 数据不完整",
    ]
    assert result["data_gaps"] == short["data_gaps"] + [
        "【数据获取失败】medium horizon：medium provider unavailable"
    ]
    assert result["not_applicable"] is False
    assert result["falsification_conditions"] == ["条件：short 业绩不达预期"]

    assert len(saved_reports) == 1
    assert saved_reports[0]["result_data"]["medium_term"]["status"] == "failed"
    assert saved_reports[0]["data_gaps"] == result["data_gaps"]
    assert saved_reports[0]["result_data"]["short_term"]["status"] == "completed"
    assert saved_reports[0]["data_gaps"] == result["data_gaps"]
    assert any('"status": "partial"' in chunk for chunk in chunks)
    assert any("agent.horizon_failed" in chunk for chunk in chunks)


def test_dual_horizon_all_failures_surface_as_failed_job():
    """Scenario 3: all horizons raise -> RuntimeError surfaces as a failed job."""
    job, saved_reports, chunks = _run_dual_horizon_job(fail_horizons=("short", "medium"))

    assert job["status"] == "failed"
    assert "RuntimeError" in job["error"]
    assert "All requested horizons failed" in job["error"]
    assert "All requested horizons failed" in job["traceback"]
    assert "short: short provider unavailable" in job["error"]
    assert "medium: medium provider unavailable" in job["error"]
    # No dual-horizon report is saved when every horizon failed.
    assert saved_reports == []
    assert any("event: job.failed" in chunk for chunk in chunks)
    assert any("event: done" in chunk for chunk in chunks)
    assert not any('"mode": "dual_horizon"' in chunk for chunk in chunks)


def test_dual_horizon_short_only_failure_keeps_other_horizon_result_and_gaps():
    """Scenario 4: short horizon fails; medium horizon's result/gaps survive."""
    job, saved_reports, chunks = _run_dual_horizon_job(fail_horizons=("short",))

    assert job["status"] == "completed"
    result = job["result"]
    assert result["mode"] == "dual_horizon"
    assert result["status"] == "partial"
    assert result["horizon_status"] == {"short": "failed", "medium": "completed"}
    assert result["failed_horizons"] == ["short"]

    failed = result["short_term"]
    assert failed["status"] == "failed"
    assert failed["horizon"] == "short"
    assert failed["error"]
    assert failed["not_applicable"] is None

    medium = result["medium_term"]
    assert medium["status"] == "completed"
    assert medium["data_gaps"] == [
        "【数据获取失败】medium horizon 新闻接口超时",
        "模型补充：medium 数据不完整",
    ]
    assert result["data_gaps"] == [
        "【数据获取失败】short horizon：short provider unavailable",
        "【数据获取失败】medium horizon 新闻接口超时",
        "模型补充：medium 数据不完整",
    ]
    assert result["not_applicable"] is False
    assert result["falsification_conditions"] == ["条件：medium 业绩不达预期"]

    assert len(saved_reports) == 1
    assert saved_reports[0]["result_data"]["short_term"]["status"] == "failed"
    assert saved_reports[0]["result_data"]["medium_term"]["status"] == "completed"
    assert saved_reports[0]["data_gaps"] == result["data_gaps"]
    assert any('"status": "partial"' in chunk for chunk in chunks)
    assert any("agent.horizon_failed" in chunk for chunk in chunks)


def test_dual_horizon_real_db_all_failures_marks_report_failed():
    """Scenario 5: all horizons fail with a real SQLite DB -> ReportDB marked status='failed'."""
    from contextlib import contextmanager
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from api.database import Base, ReportDB

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine)

    @contextmanager
    def _fake_db_ctx():
        sess = TestingSession()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    job_id = f"dual-db-{uuid4().hex}"
    store = InMemoryJobStore()
    collector = MagicMock()
    collector.collect.return_value = {"market_data_context": {"source": "fixture"}}
    _FakeTradingGraph.fail_horizons = {"short", "medium"}
    _FakeGraphStream.fail_horizons = {"short", "medium"}

    request = main.AnalyzeRequest(
        symbol="600519.SH",
        trade_date="2026-07-31",
        horizons=["short", "medium"],
        selected_analysts=[],
    )

    db = TestingSession()
    report_service.init_report(db, job_id, "600519.SH", "2026-07-31")

    with (
        patch.object(main, "_job_store_instance", store),
        patch.object(main, "_shared_data_collector", collector),
        patch.object(main, "TradingAgentsGraph", side_effect=lambda **_kwargs: _FakeTradingGraph(**_kwargs)),
        patch.object(main, "_build_runtime_config", return_value={}),
        patch.object(main, "_resolve_and_freeze_custom_prompts", return_value=({}, False)),
        patch.object(main, "get_db_ctx", _fake_db_ctx),
    ):
        asyncio.run(main._run_job_inner(job_id, request, stream_events=False, save_report=True))

    db.expire_all()
    db_report = db.query(ReportDB).filter(ReportDB.id == job_id).first()
    assert db_report is not None
    assert db_report.status == "failed"
    assert db_report.error is not None
    assert "All requested horizons failed" in db_report.error
    assert "short" in db_report.error
    assert "medium" in db_report.error

    job = store.get_job(job_id)
    assert job["status"] == "failed"
    assert "All requested horizons failed" in job["error"]


def test_dual_horizon_empty_stream_fails_horizon_and_survives_partial():
    """Scenario 6: empty stream on one horizon treated as failed; other horizon completes."""
    job, saved_reports, chunks = _run_dual_horizon_job(empty_horizons=("medium",))

    assert job["status"] == "completed"
    result = job["result"]
    assert result["mode"] == "dual_horizon"
    assert result["status"] == "partial"
    assert result["horizon_status"] == {"short": "completed", "medium": "failed"}
    assert result["failed_horizons"] == ["medium"]
    assert result["short_term"]["status"] == "completed"
    assert result["medium_term"]["status"] == "failed"
    assert len(saved_reports) == 1


def test_dual_horizon_all_empty_streams_fails_job():
    """Scenario 7: all horizons return empty stream -> job fails."""
    job, saved_reports, chunks = _run_dual_horizon_job(empty_horizons=("short", "medium"))

    assert job["status"] == "failed"
    assert "All requested horizons failed" in job["error"]
    assert saved_reports == []


def test_dual_horizon_post_processing_failure_records_failed_horizon():
    """Scenario 8: horizon fails in post-processing build -> marked failed horizon."""
    job, saved_reports, chunks = _run_dual_horizon_job(build_fail_horizons=("medium",))

    assert job["status"] == "completed"
    result = job["result"]
    assert result["mode"] == "dual_horizon"
    assert result["status"] == "partial"
    assert result["horizon_status"] == {"short": "completed", "medium": "failed"}
    assert result["failed_horizons"] == ["medium"]
    assert len(saved_reports) == 1


