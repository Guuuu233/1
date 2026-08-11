"""Regression coverage for chat symbol extraction boundaries."""

import asyncio
from contextlib import ExitStack, nullcontext
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api import main
from tradingagents.dataflows import trade_calendar as tc


_EMPTY_EXTRACTION = (
    '{"stock_name": null, "date": null, "horizons": ["short"], '
    '"focus_areas": [], "specific_questions": [], "user_context": {}}'
)
_CUSTOM_REQUIREMENTS = "\r\n\r\n  [分析要求] 请让 Agent 团队重点关注量价和 AAPL。"


class _FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return MagicMock(content=_EMPTY_EXTRACTION)

    async def astream(self, prompt):
        self.prompts.append(prompt)
        yield MagicMock(content=_EMPTY_EXTRACTION)


class _FakeClient:
    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm


def _patch_extraction_dependencies(stack, llm, name_resolver=None):
    stack.enter_context(
        patch("tradingagents.llm_clients.factory.create_llm_client", return_value=_FakeClient(llm))
    )
    stack.enter_context(
        patch.object(main, "_search_cn_stock_by_name", side_effect=name_resolver or (lambda _: None))
    )


def _extract_sync(text: str, name_resolver=None):
    llm = _FakeLLM()
    with ExitStack() as stack:
        _patch_extraction_dependencies(stack, llm, name_resolver)
        result = main._ai_extract_symbol_and_date(text, {})
    return result, llm.prompts


def _extract_streaming(text: str, name_resolver=None):
    llm = _FakeLLM()
    with ExitStack() as stack:
        _patch_extraction_dependencies(stack, llm, name_resolver)
        stack.enter_context(patch.object(main, "_emit_job_event"))
        result = asyncio.run(main._ai_extract_symbol_and_date_streaming(text, {}, "job-1"))
    return result, llm.prompts


@pytest.mark.parametrize("extract", [_extract_sync, _extract_streaming])
def test_extraction_ignores_custom_requirements_agent_token(extract):
    original_question = "请分析后市走势"

    result, prompts = extract(original_question + _CUSTOM_REQUIREMENTS)

    assert result[0] is None
    assert len(prompts) == 1
    assert original_question in prompts[0]
    assert "[分析要求]" not in prompts[0]
    assert "Agent" not in prompts[0]


@pytest.mark.parametrize("extract", [_extract_sync, _extract_streaming])
@pytest.mark.parametrize(
    ("text", "expected_symbol"),
    [("分析 600519 的短线机会", "600519.SH"), ("分析 AAPL 的短线机会", "AAPL")],
)
def test_extraction_preserves_explicit_a_share_and_us_tickers(extract, text, expected_symbol):
    result, _ = extract(text)

    assert result[0] == expected_symbol


@pytest.mark.parametrize("extract", [_extract_sync, _extract_streaming])
def test_extraction_does_not_infer_today_when_date_is_omitted(extract):
    result, _ = extract("分析 600519 的短线机会")

    assert result[1] is None


@pytest.mark.parametrize("extract", [_extract_sync, _extract_streaming])
def test_extraction_keeps_literal_requirements_marker_in_question_body(extract):
    text = "请解释 [分析要求] AAPL 在 2026-01-15 的走势"

    result, prompts = extract(text)

    assert result[0] == "AAPL"
    assert result[1] == "2026-01-15"
    assert text in prompts[0]


@pytest.mark.parametrize("extract", [_extract_sync, _extract_streaming])
def test_chinese_name_and_date_use_original_question_not_custom_requirements(extract):
    original_question = "请分析贵州茅台，截止日期 2026-01-15"
    received_name_queries = []

    def resolve_name(query):
        received_name_queries.append(query)
        return "600519.SH" if query == original_question else None

    result, prompts = extract(original_question + _CUSTOM_REQUIREMENTS, resolve_name)

    assert result[0] == "600519.SH"
    assert result[1] == "2026-01-15"
    assert received_name_queries == [original_question]
    assert original_question in prompts[0]
    assert "请让 Agent 团队重点关注量价和 AAPL。" not in prompts[0]


def _chat_request(text: str, stream: bool):
    return main.ChatCompletionRequest(
        messages=[main.ChatMessage(role="user", content=text)], stream=stream, dry_run=True
    )


async def _capture_chat_analyze_request(
    text: str,
    stream: bool,
    extraction_result=None,
):
    run_job = AsyncMock()
    extraction_result = extraction_result or (
        "600519.SH", "2026-01-15", ["short"], [], [], {}
    )
    background_tasks = []

    def track_task(coro):
        task = asyncio.create_task(coro)
        background_tasks.append(task)
        return task

    extract_path = (
        "_ai_extract_symbol_and_date_streaming" if stream else "_ai_extract_symbol_and_date"
    )
    with (
        patch.object(main, "_build_runtime_config", return_value={}),
        patch.object(main, extract_path, return_value=extraction_result),
        patch.object(main, "get_db_ctx", return_value=nullcontext(MagicMock())),
        patch.object(main, "_compose_analysis_user_context", return_value={}),
        patch.object(main, "_set_job"),
        patch.object(main, "_get_job", return_value={"status": "completed", "decision": "DRY_RUN"}),
        patch.object(main, "_emit_job_event"),
        patch.object(main, "_run_job", run_job),
        patch.object(main, "_create_tracked_task", side_effect=track_task),
    ):
        await main.chat_completions(
            _chat_request(text, stream), SimpleNamespace(id="user-1")
        )
        if background_tasks:
            await asyncio.gather(*background_tasks)

    return run_job.await_args.args[1]


@pytest.mark.parametrize("stream", [False, True])
def test_chat_preserves_full_text_for_query_and_pre_intent(stream):
    text = "分析600519\n\n[分析要求] Agent 团队只关注 AAPL"

    analyze_request = asyncio.run(_capture_chat_analyze_request(text, stream))

    assert analyze_request.query == text
    assert analyze_request.user_intent["raw_query"] == text


@pytest.mark.parametrize("stream", [False, True])
def test_chat_default_date_is_consistent_for_sync_and_streaming(stream):
    dates = [date(2026, 8, 11), date(2026, 8, 12), date(2026, 8, 13)]
    tc._TRADE_DATES_CACHE["dates"] = dates
    tc._TRADE_DATES_CACHE["dates_set"] = set(dates)
    tc._TRADE_DATES_CACHE["loaded_at"] = 1e18
    extraction = ("600519.SH", None, ["short"], [], [], {})
    frozen = datetime(2026, 8, 12, 3, 0, tzinfo=tc.CN_TZ)

    with patch.object(tc, "now_cn", return_value=frozen):
        analyze_request = asyncio.run(
            _capture_chat_analyze_request(
                "分析600519的短线机会",
                stream,
                extraction_result=extraction,
            )
        )

    assert analyze_request.trade_date == "2026-08-11"
    assert analyze_request.trade_date_explicit is False
    tc.clear_cn_trade_date_cache()
