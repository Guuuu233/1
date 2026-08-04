"""Focused deterministic regression coverage for DAV-27.

These tests use only fixed payloads and temporary SQLite.  They must not call
real LLMs or market-data providers.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, ReportDB
from api import main
from api.job_store import InMemoryJobStore
from api.services import report_service
from tradingagents.graph import data_collector


def _make_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return session_factory(), engine


@pytest.mark.parametrize("decision", ["BUY", "SELL", "HOLD"])
def test_structured_report_keeps_legal_decisions_and_missing_fields_unknown(decision):
    structured = report_service.StructuredReport(decision=decision)

    assert structured.decision == decision
    assert structured.confidence is None
    assert structured.probability is None


def test_structured_report_does_not_turn_empty_or_illegal_decisions_into_hold(caplog):
    with caplog.at_level(logging.WARNING, logger="api.services.report_service"):
        empty = report_service.StructuredReport()
        illegal = report_service.StructuredReport(decision="MAYBE")

    assert empty.decision is None
    assert illegal.decision is None
    assert empty.confidence is None
    assert any("decision" in record.getMessage().lower() for record in caplog.records)


def test_structured_extraction_requires_explicit_confidence_semantics():
    prompts = []

    def invoke(messages):
        prompts.append(messages[0].content)
        return SimpleNamespace(content=json.dumps({"decision": "BUY"}, ensure_ascii=False))

    fake_llm = SimpleNamespace(invoke=invoke)
    fake_client = SimpleNamespace(get_llm=lambda: fake_llm)

    with patch("tradingagents.llm_clients.create_llm_client", return_value=fake_client):
        structured = report_service.extract_structured_data(
            final_trade_decision="结论语气很强，但没有明确置信度数字。",
            config={"llm_provider": "fake", "quick_think_llm": "fake"},
        )

    assert structured is not None
    assert structured.confidence is None
    assert "根据语气" not in prompts[0]
    assert "明确给出则为 null" in prompts[0]


def test_report_service_rejects_malformed_machine_blocks_before_create():
    db, engine = _make_session()
    try:
        malformed = "固定正文\n<!-- DEBATE_STATE: {\"new_claims\": [} -->"
        with pytest.raises(ValueError, match="DEBATE_STATE"):
            report_service.create_report(
                db=db,
                symbol="600519.SH",
                trade_date="2026-07-31",
                result_data={"final_trade_decision": malformed},
            )

        assert db.query(ReportDB).count() == 0
    finally:
        db.close()
        engine.dispose()


def test_report_service_rejects_malformed_machine_blocks_before_partial_commit():
    db, engine = _make_session()
    try:
        report = report_service.init_report(
            db=db,
            report_id="dav27-partial",
            symbol="600519.SH",
            trade_date="2026-07-31",
        )
        malformed = "固定正文\n<!-- RISK_STATE: {\"new_claims\": [} -->"

        with pytest.raises(ValueError, match="RISK_STATE"):
            report_service.update_report_partial(
                db,
                report.id,
                result_data={"final_trade_decision": malformed},
            )

        db.expire_all()
        persisted = db.query(ReportDB).filter(ReportDB.id == report.id).one()
        assert persisted.result_data is None
        assert persisted.status == "pending"
    finally:
        db.close()
        engine.dispose()


def test_unknown_structured_fields_are_warned_and_not_persisted(caplog):
    db, engine = _make_session()
    try:
        payload = {
            "decision": "BUY",
            "confidence": 80,
            "unknown_root_field": "discard me",
        }
        with caplog.at_level(logging.WARNING, logger="api.services.report_service"):
            report = report_service.create_report(
                db=db,
                symbol="600519.SH",
                trade_date="2026-07-31",
                result_data={"structured": payload, "final_trade_decision": "结论：买入"},
            )

        assert "unknown_root_field" not in report.result_data["structured"]
        assert any("unknown structured fields" in record.getMessage() for record in caplog.records)
    finally:
        db.close()
        engine.dispose()


def test_collector_failure_ledger_is_ordered_and_excludes_normal_empty_results():
    raw_results = {
        "stock_data": "正常行情数据",
        "news": "news 调用失败：TimeoutError: provider timeout",
        "global_news": "global_news 数据拉取超时（>300s），本次分析跳过该数据源",
        "hot_stocks": "正常无重大新闻",
        "fundamentals": "无数据：本周期没有可用财报",
        "realtime": {
            "status": "unavailable",
            "error": "实时行情源不可用：TimeoutError",
        },
    }

    ledger = data_collector._build_data_failure_ledger(raw_results)

    assert [entry["source"] for entry in ledger] == ["news", "global_news", "realtime"]
    assert [entry["status"] for entry in ledger] == ["failed", "timeout", "unavailable"]
    assert all(entry["gap"].startswith("【数据获取失败】") for entry in ledger)
    assert "hot_stocks" not in {entry["source"] for entry in ledger}
    assert "fundamentals" not in {entry["source"] for entry in ledger}


def test_merge_data_gaps_consumes_known_collector_ledger_before_report_and_llm():
    result_data = {
        "market_data_context": {
            "data_failure_ledger": [
                {
                    "source": "news",
                    "status": "failed",
                    "reason": "provider timeout",
                },
                {
                    "source": "not_an_applicable_source",
                    "status": "not_applicable",
                    "reason": "no event",
                },
            ]
        },
        "news_report": "【数据获取失败】新闻摘要源超时",
    }

    assert report_service.merge_data_gaps(
        result_data,
        llm_data_gaps=["模型识别：新闻数据不完整"],
    ) == [
        "【数据获取失败】news：provider timeout",
        "【数据获取失败】新闻摘要源超时",
        "模型识别：新闻数据不完整",
    ]


def test_dual_horizon_metadata_is_source_keyed_and_mixed_not_applicable_is_false():
    result = report_service.aggregate_horizon_metadata(
        [
            ("short", {"status": "completed", "not_applicable": False, "falsification_conditions": ["短线条件"]}),
            ("medium", {"status": "completed", "not_applicable": True, "falsification_conditions": ["中线条件"]}),
        ],
        requested_horizons=["short", "medium"],
    )

    assert result["not_applicable"] is False
    assert result["not_applicable_by_horizon"] == {"short": False, "medium": True}
    assert result["falsification_conditions_by_horizon"] == {
        "short": ["短线条件"],
        "medium": ["中线条件"],
    }
    assert result["falsification_conditions"] == ["短线条件", "中线条件"]


class _OrdinaryGraphEngine:
    def __init__(self, state):
        self.state = state

    async def astream(self, _init_state, **_kwargs):
        yield dict(self.state)


class _OrdinaryGraph:
    state = {
        "company_of_interest": "600519.SH",
        "trade_date": "2026-07-31",
        "news_report": "【数据获取失败】新闻摘要源超时",
        "fundamentals_report": "固定基本面报告",
        "final_trade_decision": "固定图信号结论",
        "market_data_context": {
            "data_failure_ledger": [
                {"source": "news", "status": "timeout", "reason": "provider timeout"}
            ]
        },
    }

    def __init__(self, data_collector, **_kwargs):
        self.data_collector = data_collector
        self.quick_thinking_llm = object()
        self.role_resolved_configs = {}
        self.propagator = SimpleNamespace(
            create_initial_state=lambda *args, **kwargs: dict(self.state),
            get_graph_args=lambda: {},
        )
        self.graph = _OrdinaryGraphEngine(self.state)

    def process_signal(self, _decision):
        return "BUY"

    def propagate(self, *_args, **_kwargs):
        return dict(self.state), None


@pytest.mark.parametrize("stream_events", [False, True])
def test_ordinary_and_streaming_paths_merge_gaps_and_keep_graph_fallback(stream_events):
    job_id = f"dav27-ordinary-{uuid4().hex}"
    store = InMemoryJobStore()
    collector = MagicMock()
    collector.collect.return_value = {"market_data_context": _OrdinaryGraph.state["market_data_context"]}
    saved_reports = []
    db = MagicMock()

    request = main.AnalyzeRequest(
        symbol="600519.SH",
        trade_date="2026-07-31",
        horizons=["short"],
        selected_analysts=[],
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

    def capture_create_report(**kwargs):
        saved_reports.append(kwargs)

    with (
        patch.object(main, "_job_store_instance", store),
        patch.object(main, "_shared_data_collector", collector),
        patch.object(main, "TradingAgentsGraph", _OrdinaryGraph),
        patch.object(main, "_build_runtime_config", return_value={}),
        patch.object(main, "_resolve_and_freeze_custom_prompts", return_value=({}, False)),
        patch.object(main, "get_db_ctx", return_value=nullcontext(db)),
        patch.object(report_service, "init_report"),
        patch.object(report_service, "update_report_partial"),
        patch.object(report_service, "extract_structured_data", return_value=report_service.StructuredReport()),
        patch.object(report_service, "create_report", side_effect=capture_create_report),
    ):
        chunks = asyncio.run(run_job())

    job = store.get_job(job_id)
    expected_gaps = [
        "【数据获取失败】news：provider timeout",
        "【数据获取失败】新闻摘要源超时",
    ]
    assert job["status"] == "completed"
    assert job["decision"] == "BUY"
    assert job["result"]["confidence"] is None
    assert job["result"]["data_gaps"] == expected_gaps
    assert saved_reports[0]["data_gaps"] == expected_gaps
    assert saved_reports[0]["result_data"]["data_gaps"] == expected_gaps
    assert any(expected_gaps[0] in chunk for chunk in chunks)
