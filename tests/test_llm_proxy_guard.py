"""Tests for LLM base_url proxy reachability guard (DAV-777).

Covers:
1. 小写 no_proxy 缺字面量 IP（应拒绝，fail-closed）
2. 含字面量 IP（应放行）
3. 仅 CIDR（应拒绝——httpx 不认 CIDR）
4. 完全不设（应拒绝）
5. base_url 为 localhost/127.0.0.1（应跳过自检）
6. 大小写 no_proxy 不一致时以 httpx 实际行为为准
7. 契约格式验证：错误信息必须包含 (a) 实际 base_url, (b) 代理地址, (c) 修复命令（含小写 no_proxy 与字面量 IP）
8. 客户端构造处拦截（create_llm_client 与 OpenAIClient）
9. 境外服务/域名合法代理需求放行（Contract 3）
10. 禁止真实打网（全部用环境变量注入 + httpx 判定）
"""

import os
import pytest
from unittest.mock import patch

from tradingagents.llm_clients import (
    LLMProxyRoutingError,
    check_llm_proxy_guard,
    create_llm_client,
)
from tradingagents.llm_clients.openai_client import OpenAIClient
from tradingagents.llm_clients.proxy_guard import clear_proxy_guard_cache

TEST_PROXY = "http://127.0.0.1:7897"
TEST_CPA_URL = "http://100.65.130.33:8317/v1"
TEST_CPA_IP = "100.65.130.33"


@pytest.fixture(autouse=True)
def clean_proxy_env_and_cache(monkeypatch):
    """Ensure a clean proxy environment and clear guard cache for each test."""
    clear_proxy_guard_cache()
    # Remove all proxy env vars first
    for var in [
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    ]:
        monkeypatch.delenv(var, raising=False)
    yield
    clear_proxy_guard_cache()


def test_missing_literal_ip_in_lowercase_no_proxy_rejected(monkeypatch):
    """Case 1: 小写 no_proxy 缺字面量 IP（应拒绝）."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    monkeypatch.setenv("no_proxy", "192.168.1.1,example.com")

    with pytest.raises(LLMProxyRoutingError) as exc_info:
        check_llm_proxy_guard(TEST_CPA_URL)

    err = exc_info.value
    assert TEST_CPA_URL in str(err)
    assert TEST_PROXY in str(err)
    assert "no_proxy" in str(err)
    assert TEST_CPA_IP in str(err)


def test_contains_literal_ip_in_no_proxy_allowed(monkeypatch):
    """Case 2: 含字面量 IP（应放行）."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    monkeypatch.setenv("no_proxy", f"192.168.1.1,{TEST_CPA_IP},localhost")

    # Should not raise
    check_llm_proxy_guard(TEST_CPA_URL)


def test_cidr_in_no_proxy_rejected(monkeypatch):
    """Case 3: 仅 CIDR（应拒绝——httpx 不认 CIDR）."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    monkeypatch.setenv("no_proxy", "100.64.0.0/10")

    with pytest.raises(LLMProxyRoutingError) as exc_info:
        check_llm_proxy_guard(TEST_CPA_URL)

    assert TEST_CPA_URL in str(exc_info.value)
    assert TEST_CPA_IP in str(exc_info.value)


def test_no_proxy_completely_unset_rejected(monkeypatch):
    """Case 4: 完全不设（应拒绝）."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    # no_proxy and NO_PROXY are unset

    with pytest.raises(LLMProxyRoutingError) as exc_info:
        check_llm_proxy_guard(TEST_CPA_URL)

    assert TEST_CPA_URL in str(exc_info.value)


@pytest.mark.parametrize(
    "localhost_url",
    [
        "http://localhost:8317/v1",
        "http://127.0.0.1:8317/v1",
        "http://localhost:11434/v1",
        "http://127.0.0.1:11434/v1",
    ],
)
def test_localhost_and_loopback_skipped(monkeypatch, localhost_url):
    """Case 5: base_url 为 localhost/127.0.0.1 时跳过自检."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    # Even if proxy is set and no_proxy is empty, localhost must skip guard
    check_llm_proxy_guard(localhost_url)


def test_case_inconsistency_uppercase_no_proxy_ignored_by_httpx(monkeypatch):
    """Case 6a: 大写 NO_PROXY 含字面量 IP，但小写 no_proxy 为空或缺 IP 时，httpx 会走代理，应拒绝."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    monkeypatch.setenv("HTTP_PROXY", TEST_PROXY)
    monkeypatch.setenv("NO_PROXY", TEST_CPA_IP)
    monkeypatch.setenv("no_proxy", "")

    with pytest.raises(LLMProxyRoutingError) as exc_info:
        check_llm_proxy_guard(TEST_CPA_URL)

    assert TEST_CPA_URL in str(exc_info.value)


def test_case_inconsistency_lowercase_no_proxy_respected_by_httpx(monkeypatch):
    """Case 6b: 小写 no_proxy 含字面量 IP，大写 NO_PROXY 为空时，httpx 直连，应放行."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    monkeypatch.setenv("HTTP_PROXY", TEST_PROXY)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", TEST_CPA_IP)

    check_llm_proxy_guard(TEST_CPA_URL)


def test_error_message_has_required_contract_elements(monkeypatch):
    """Contract 2: 包含 (a) 实际 base_url, (b) 代理地址, (c) 修复命令（含小写 no_proxy 与字面量 IP）."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)

    with pytest.raises(LLMProxyRoutingError) as exc_info:
        check_llm_proxy_guard(TEST_CPA_URL)

    err = exc_info.value
    # (a) 实际 base_url
    assert TEST_CPA_URL in err.base_url
    assert TEST_CPA_URL in str(err)

    # (b) 代理地址
    assert "127.0.0.1:7897" in err.proxy_url
    assert "127.0.0.1:7897" in str(err)

    # (c) 可直接复制执行的修复命令（含小写 no_proxy 与字面量 IP）
    assert f'export no_proxy="${{no_proxy}},{TEST_CPA_IP}"' in err.repair_command
    assert f'export no_proxy="${{no_proxy}},{TEST_CPA_IP}"' in str(err)
    assert "no_proxy" in err.repair_command
    assert TEST_CPA_IP in err.repair_command


def test_client_construction_interception_openai_client(monkeypatch):
    """OpenAIClient 构造处直接 fail-closed 拦截，不静默降级."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    monkeypatch.setenv("no_proxy", "")

    with pytest.raises(LLMProxyRoutingError):
        OpenAIClient(model="gpt-4o", base_url=TEST_CPA_URL)


def test_client_construction_interception_factory(monkeypatch):
    """create_llm_client 构造处直接 fail-closed 拦截，不静默降级."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    monkeypatch.setenv("no_proxy", "")

    with pytest.raises(LLMProxyRoutingError):
        create_llm_client(provider="openai", model="gpt-4o", base_url=TEST_CPA_URL)


def test_client_construction_allowed_when_no_proxy_configured(monkeypatch):
    """配置了小写 no_proxy 时，客户端正常构造通过."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    monkeypatch.setenv("no_proxy", TEST_CPA_IP)

    client = create_llm_client(provider="openai", model="gpt-4o", base_url=TEST_CPA_URL)
    assert isinstance(client, OpenAIClient)
    assert client.base_url == TEST_CPA_URL


def test_overseas_domain_base_url_allows_proxy(monkeypatch):
    """Contract 3: 境外服务/域名合法代理需求放行，不得误拦截."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    monkeypatch.setenv("https_proxy", TEST_PROXY)
    monkeypatch.setenv("no_proxy", "")

    # Domain-based endpoints like api.openai.com are allowed to go through proxy
    check_llm_proxy_guard("https://api.openai.com/v1")
    check_llm_proxy_guard("https://openrouter.ai/api/v1")


def test_none_or_empty_base_url_skipped(monkeypatch):
    """None 或空 base_url 跳过自检."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    check_llm_proxy_guard(None)
    check_llm_proxy_guard("")


def test_url_without_scheme_handled_properly(monkeypatch):
    """未显式写 http:// 的 IP base_url 同样能正确被识别并保护."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    monkeypatch.setenv("no_proxy", "")

    with pytest.raises(LLMProxyRoutingError) as exc_info:
        check_llm_proxy_guard("100.65.130.33:8317/v1")

    assert TEST_CPA_IP in str(exc_info.value)


def test_cache_avoids_repeated_checks(monkeypatch):
    """Contract 4: 自检具有内存缓存，重复构造不会重复调用 httpx."""
    monkeypatch.setenv("http_proxy", TEST_PROXY)
    monkeypatch.setenv("no_proxy", TEST_CPA_IP)

    # First check populates cache
    check_llm_proxy_guard(TEST_CPA_URL)

    # Patch httpx.Client to ensure it is NOT called again on subsequent checks
    with patch("httpx.Client") as mock_client:
        check_llm_proxy_guard(TEST_CPA_URL)
        mock_client.assert_not_called()
