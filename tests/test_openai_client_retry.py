"""TDD tests for OpenAIClient transient-error retry (DAV-91).

Uses httpx.MockTransport to script upstream responses and count actual HTTP
calls, exercising the real OpenAI SDK retry path returned by
``OpenAIClient.get_llm()`` — not a mocked retry wrapper.
"""

import asyncio
import json

import httpx
import pytest
from openai import AuthenticationError, InternalServerError

from tradingagents.llm_clients.openai_client import OpenAIClient

_MODEL = "gpt-4o"


def _sse(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


def _completion_chunk(delta: str = "", finish_reason=None) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": delta} if delta else {},
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


def _completion(body: str = "ok") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": body},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _make_client(handler, **client_kwargs) -> OpenAIClient:
    """Build an OpenAIClient whose HTTP layer is backed by MockTransport(handler)."""
    kwargs = {
        "api_key": "sk-test",
        "http_client": httpx.Client(transport=httpx.MockTransport(handler)),
        "http_async_client": httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    }
    kwargs.update(client_kwargs)
    return OpenAIClient(model=_MODEL, base_url="https://api.openai.com/v1", **kwargs)


# ── 重试次数解析 ──────────────────────────────────────────────


def test_default_retry_times_is_two(monkeypatch):
    monkeypatch.delenv("TA_LLM_RETRY_TIMES", raising=False)
    llm = OpenAIClient(model=_MODEL, api_key="sk-test").get_llm()
    assert llm.max_retries == 2


def test_env_var_configures_retry_times(monkeypatch):
    monkeypatch.setenv("TA_LLM_RETRY_TIMES", "1")
    llm = OpenAIClient(model=_MODEL, api_key="sk-test").get_llm()
    assert llm.max_retries == 1


def test_kwargs_max_retries_overrides_env(monkeypatch):
    monkeypatch.setenv("TA_LLM_RETRY_TIMES", "2")
    llm = OpenAIClient(model=_MODEL, api_key="sk-test", max_retries=0).get_llm()
    assert llm.max_retries == 0


def test_invalid_env_var_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("TA_LLM_RETRY_TIMES", "abc")
    llm = OpenAIClient(model=_MODEL, api_key="sk-test").get_llm()
    assert llm.max_retries == 2


# ── 瞬时故障重试（调用次数 = 1 + 重试数）─────────────────────


def test_transient_500_retries_then_succeeds(monkeypatch):
    """上游瞬时 500 后恢复：重试成功，调用次数 = 1 + 重试数。"""
    monkeypatch.setenv("TA_LLM_RETRY_TIMES", "2")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(500, json={"error": {"message": "boom"}})
        return httpx.Response(200, json=_completion())

    llm = _make_client(handler).get_llm()
    result = llm.invoke("hi")
    assert result.content == "ok"
    assert len(calls) == 3  # 1 次原始 + 2 次重试


def test_persistent_500_fails_after_max_retries(monkeypatch):
    """持续 500：最终失败且调用次数 = 1 + 重试数（不无限重试）。"""
    monkeypatch.setenv("TA_LLM_RETRY_TIMES", "2")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, json={"error": {"message": "boom"}})

    llm = _make_client(handler).get_llm()
    with pytest.raises(InternalServerError):
        llm.invoke("hi")
    assert len(calls) == 3


def test_successful_call_has_zero_retries(monkeypatch):
    """正常完成的调用零重试（调用次数 = 1）。"""
    monkeypatch.setenv("TA_LLM_RETRY_TIMES", "2")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_completion())

    llm = _make_client(handler).get_llm()
    result = llm.invoke("hi")
    assert result.content == "ok"
    assert len(calls) == 1


# ── 不可重试错误直接失败 ──────────────────────────────────────


def test_auth_error_does_not_retry(monkeypatch):
    """401 认证失败：不重试，1 次即失败。"""
    monkeypatch.setenv("TA_LLM_RETRY_TIMES", "2")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    llm = _make_client(handler).get_llm()
    with pytest.raises(AuthenticationError):
        llm.invoke("hi")
    assert len(calls) == 1


def test_zero_retries_disables_retry(monkeypatch):
    """TA_LLM_RETRY_TIMES=0 恢复完全禁用：500 也 1 次即失败。"""
    monkeypatch.setenv("TA_LLM_RETRY_TIMES", "0")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, json={"error": {"message": "boom"}})

    llm = _make_client(handler).get_llm()
    with pytest.raises(InternalServerError):
        llm.invoke("hi")
    assert len(calls) == 1


# ── 流式路径：中途断连不回归 ──────────────────────────────────


def test_stream_mid_disconnect_raises_and_does_not_retry(monkeypatch):
    """流中途断连：异常上抛（代理层 try/except 已处理），不从头发起重试。"""
    monkeypatch.setenv("TA_LLM_RETRY_TIMES", "2")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)

        async def abody():
            yield _sse(_completion_chunk(""))
            yield _sse(_completion_chunk("Hello"))
            raise httpx.ReadError("mid-stream disconnect")

        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=abody()
        )

    llm = _make_client(handler).get_llm()
    chunks = []

    async def _run():
        with pytest.raises(httpx.ReadError):
            async for chunk in llm.astream("hi"):
                chunks.append(chunk.content)

    asyncio.run(_run())
    assert len(calls) == 1  # 流建立后断连不重试整个请求
    assert "".join(chunks) == "Hello"  # 已收到的分片保留


def test_stream_initial_500_retries_before_establish(monkeypatch):
    """流建立前失败（首个请求 500）会被重试，成功后正常出流。"""
    monkeypatch.setenv("TA_LLM_RETRY_TIMES", "2")
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(500, json={"error": {"message": "boom"}})

        async def abody():
            yield _sse(_completion_chunk(""))
            yield _sse(_completion_chunk("Hello"))
            yield _sse(_completion_chunk("", "stop"))
            yield b"data: [DONE]\n\n"

        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=abody()
        )

    llm = _make_client(handler).get_llm()
    chunks = []

    async def _run():
        async for chunk in llm.astream("hi"):
            chunks.append(chunk.content)

    asyncio.run(_run())
    assert len(calls) == 3  # 1 次原始 + 2 次重试（重试只发生在流建立前）
    assert "".join(chunks) == "Hello"
