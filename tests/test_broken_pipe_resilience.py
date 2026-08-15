from __future__ import annotations

import asyncio
import logging
import sys
from unittest.mock import MagicMock, patch

from api import main
from api.job_store import InMemoryJobStore
from tradingagents.dataflows import interface
from tradingagents.llm_clients import openai_client


class _ClosedPipe:
    def write(self, _text: str) -> int:
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self) -> None:
        raise BrokenPipeError(32, "Broken pipe")


def test_provider_trace_does_not_turn_closed_stdout_into_analysis_failure(monkeypatch, caplog):
    monkeypatch.setenv("TA_TRACE", "1")
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())

    with caplog.at_level(logging.INFO, logger=interface._logger.name):
        interface._trace("provider request")

    assert "[provider-trace] provider request" in caplog.text


def test_openai_client_initialization_does_not_write_to_stdout(monkeypatch, caplog):
    monkeypatch.setattr(sys, "stdout", _ClosedPipe())
    fake_llm = MagicMock()
    fake_llm._is_reasoning_model.return_value = False
    fake_llm.return_value = {"model": "test-model"}
    monkeypatch.setattr(openai_client, "UnifiedChatOpenAI", fake_llm)
    client = openai_client.OpenAIClient("test-model", max_retries=0)

    with caplog.at_level(logging.INFO, logger=openai_client._logger.name):
        llm_kwargs = client.get_llm()

    assert llm_kwargs["model"] == "test-model"
    assert "[LLM Client] Init" in caplog.text


def test_sse_disconnect_does_not_cancel_background_job():
    async def scenario() -> None:
        job_id = "sse-disconnect-job"
        store = InMemoryJobStore()
        store.set_job(job_id, status="running")
        release = asyncio.Event()

        async def background_job() -> None:
            await release.wait()
            main._set_job(
                job_id,
                status="completed",
                result={"decision": "HOLD"},
                decision="HOLD",
                error=None,
                finished_at=main._utcnow_iso(),
            )
            main._emit_job_event(job_id, "job.completed", {"job_id": job_id})

        with patch.object(main, "_job_store_instance", store):
            task = asyncio.create_task(background_job())
            stream = main._stream_job_events(job_id)
            assert (await stream.__anext__()).startswith("event: job.ready")

            # Closing the SSE consumer must not cancel the independently tracked
            # analysis task or change its eventual terminal state.
            await stream.aclose()
            release.set()
            await task

        assert store.get_job(job_id)["status"] == "completed"
        assert store.get_job(job_id)["result"] == {"decision": "HOLD"}

    asyncio.run(scenario())
