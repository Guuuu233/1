"""Deterministic offline tests for POST /v1/models/fetch SSRF policy."""

import http.client
import json
import socket
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from api import main as api_main


def _getaddrinfo_for(ip_text):
    def getaddrinfo(host, port, type=socket.SOCK_STREAM):
        if ":" in ip_text:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip_text, port, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip_text, port))]

    return getaddrinfo


def _unexpected(*args, **kwargs):
    raise AssertionError("network path should not run")


class _FakeResponse:
    def __init__(self, status=200, body=b"{}", headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    def read(self):
        return self._body


class _FakeConnection:
    def __init__(self, response=None, request_error=None):
        self._response = response or _FakeResponse()
        self.request_error = request_error
        self.requested = None
        self.getresponse_calls = 0

    def request(self, method, path, headers=None):
        self.requested = (method, path, headers)
        if self.request_error is not None:
            raise self.request_error

    def getresponse(self):
        self.getresponse_calls += 1
        return self._response

    def close(self):
        pass


@contextmanager
def _endpoint_client():
    app = api_main.app
    dummy_user = MagicMock()
    dummy_user.id = "user-1"

    def override_user():
        return dummy_user

    def override_db():
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        yield db

    app.dependency_overrides[api_main._require_api_user] = override_user
    app.dependency_overrides[api_main.get_db] = override_db
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()


def test_allowlisted_public_host_fetches_and_sorts_models(monkeypatch):
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "models.example.com:8080")
    monkeypatch.setattr(api_main, "_resolve_models_fetch_target", lambda host, port, **kwargs: "93.184.216.34")
    fake_conn = _FakeConnection(
        response=_FakeResponse(
            body=json.dumps({"data": [{"id": "z"}, {"id": "a"}, {"id": "a"}]}).encode("utf-8")
        )
    )
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", lambda *args, **kwargs: fake_conn)

    models, url = api_main._fetch_available_models(
        "http://models.example.com:8080/v1",
        "secret-key",
    )

    assert models == ["a", "z"]
    assert url == "http://models.example.com:8080/v1/models"
    assert fake_conn.requested[0] == "GET"
    assert fake_conn.requested[1] == "/v1/models"
    assert fake_conn.requested[2]["Authorization"] == "Bearer secret-key"


def test_missing_allowlist_blocks_before_resolution(monkeypatch):
    monkeypatch.delenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, raising=False)
    monkeypatch.setattr(api_main, "_resolve_models_fetch_target", _unexpected)
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", _unexpected)

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models("http://public.example.com/v1", "")


def test_invalid_allowlist_blocks_request(monkeypatch):
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "models.example.com:notaport")

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models("http://models.example.com/v1", "")


@pytest.mark.parametrize(
    "url",
    [
        "ftp://models.example.com/v1",
        "file:///etc/passwd",
        "gopher://models.example.com:70/",
        "//models.example.com/v1",
        "models.example.com/v1",
        "http://user:pass@models.example.com/v1",
        "http://models.example.com/v1?debug=1",
        "http://models.example.com/v1#fragment",
        "http://models.example.com:99999/v1",
        "http://models.example.com:abc/v1",
    ],
)
def test_unsafe_or_non_http_url_rejected(monkeypatch, url):
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "models.example.com")
    monkeypatch.setattr(api_main, "_resolve_models_fetch_target", _unexpected)
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", _unexpected)

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models(url, "")


def test_host_not_in_allowlist_is_blocked(monkeypatch):
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "allowed.example.com")
    monkeypatch.setattr(api_main, "_resolve_models_fetch_target", _unexpected)
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", _unexpected)

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models("http://other.example.com/v1", "")


def test_allowlisted_port_mismatch_is_blocked(monkeypatch):
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "models.example.com:8080")
    monkeypatch.setattr(api_main, "_resolve_models_fetch_target", _unexpected)

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models("http://models.example.com:9090/v1", "")


@pytest.mark.parametrize(
    "resolved_ip",
    [
        "10.0.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "192.168.1.1",
        "100.64.0.1",
        "fd00:ec2::254",
        "::1",
        "fe80::1",
        "::ffff:10.0.0.1",
    ],
)
def test_blocked_resolved_ip_rejected(monkeypatch, resolved_ip):
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "public.example.com")
    monkeypatch.setattr(api_main.socket, "getaddrinfo", _getaddrinfo_for(resolved_ip))
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", _unexpected)

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models("http://public.example.com/v1", "")


@pytest.mark.parametrize(
    ("allow_entry", "url"),
    [
        ("127.0.0.1", "http://127.0.0.1/v1"),
        ("169.254.169.254", "http://169.254.169.254/v1"),
        ("[::1]", "http://[::1]/v1"),
    ],
)
def test_private_ip_literal_rejected(monkeypatch, allow_entry, url):
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, allow_entry)
    parsed = api_main._parse_models_fetch_url(url)
    monkeypatch.setattr(api_main.socket, "getaddrinfo", _getaddrinfo_for(parsed.hostname))

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models(url, "")


def test_redirect_response_is_rejected(monkeypatch):
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "models.example.com")
    monkeypatch.setattr(api_main, "_resolve_models_fetch_target", lambda host, port, **kwargs: "93.184.216.34")
    fake_conn = _FakeConnection(
        response=_FakeResponse(
            status=302,
            body=b"",
            headers={"Location": "http://169.254.169.254/latest/meta-data/"},
        )
    )
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", lambda *args, **kwargs: fake_conn)

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models("http://models.example.com/v1", "")

    assert fake_conn.getresponse_calls == 1


@pytest.mark.parametrize(
    "request_error",
    [
        OSError("timed out"),
        socket.timeout("timed out"),
        http.client.HTTPException("broken pipe"),
    ],
)
def test_connection_failure_is_wrapped_generically(monkeypatch, request_error):
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "models.example.com")
    monkeypatch.setattr(api_main, "_resolve_models_fetch_target", lambda host, port, **kwargs: "93.184.216.34")
    fake_conn = _FakeConnection(request_error=request_error)
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", lambda *args, **kwargs: fake_conn)

    with pytest.raises(api_main._ModelsFetchError) as exc_info:
        api_main._fetch_available_models("http://models.example.com/v1", "")

    assert "timed out" not in str(exc_info.value)
    assert "broken pipe" not in str(exc_info.value)


def test_http_connection_pins_resolved_ip(monkeypatch):
    captured = {}

    def fake_create_connection(address, timeout, source_address=None):
        captured["address"] = address
        return MagicMock()

    monkeypatch.setattr(api_main.socket, "create_connection", fake_create_connection)
    conn = api_main._SafeHTTPConnection(
        "models.example.com",
        8080,
        timeout=1,
        safe_ip="93.184.216.34",
    )

    conn.connect()

    assert captured["address"] == ("93.184.216.34", 8080)


def test_https_connection_pins_ip_and_keeps_sni_host(monkeypatch):
    captured = {}

    def fake_create_connection(address, timeout, source_address=None):
        captured["address"] = address
        return MagicMock()

    monkeypatch.setattr(api_main.socket, "create_connection", fake_create_connection)
    conn = api_main._SafeHTTPSConnection(
        "models.example.com",
        443,
        timeout=1,
        safe_ip="93.184.216.34",
    )
    conn._context = MagicMock()
    conn._context.wrap_socket.return_value = MagicMock()

    conn.connect()

    assert captured["address"] == ("93.184.216.34", 443)
    _, kwargs = conn._context.wrap_socket.call_args
    assert kwargs["server_hostname"] == "models.example.com"


def test_endpoint_hides_rejection_details(monkeypatch):
    def blocked(base_url, api_key, **kwargs):
        raise api_main._ModelsFetchError("resolved to 10.0.0.5")

    monkeypatch.setattr(api_main, "_fetch_available_models", blocked)

    with _endpoint_client() as client:
        response = client.post(
            "/v1/models/fetch",
            json={"base_url": "http://allowed.example.com/v1"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "ok": False,
        "error": api_main._MODELS_FETCH_GENERIC_ERROR,
        "models": [],
        "count": 0,
    }


def test_endpoint_success_shape_is_preserved(monkeypatch):
    sync_calls = []
    monkeypatch.setattr(
        api_main,
        "_fetch_available_models",
        lambda base_url, api_key, **kwargs: (["a", "b"], "http://allowed.example.com/v1/models"),
    )
    monkeypatch.setattr(
        api_main.role_routing_service,
        "sync_model_profiles_from_names",
        lambda *args, **kwargs: sync_calls.append((args, kwargs)),
    )

    with _endpoint_client() as client:
        response = client.post(
            "/v1/models/fetch",
            json={"base_url": "http://allowed.example.com/v1", "api_key": "secret-key"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "models": ["a", "b"],
        "count": 2,
        "url": "http://allowed.example.com/v1/models",
    }
    assert sync_calls == []


# --- 受信本地默认（host.docker.internal / 回环）+ 白名单 + 钉端口 ---

def test_trusted_local_host_docker_gateway_allowed_with_pinned_port(monkeypatch):
    """host.docker.internal 在白名单中且钉死端口时，放行解析出的私网 IP。"""
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "host.docker.internal:8317")
    monkeypatch.setattr(api_main.socket, "getaddrinfo", _getaddrinfo_for("172.17.0.1"))
    fake_conn = _FakeConnection(
        response=_FakeResponse(body=json.dumps({"data": [{"id": "a"}]}).encode("utf-8"))
    )
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", lambda *args, **kwargs: fake_conn)

    models, url = api_main._fetch_available_models("http://host.docker.internal:8317/v1", "")

    assert models == ["a"]


def test_trusted_local_host_loopback_allowed_with_pinned_port(monkeypatch):
    """裸机场景：127.0.0.1:8317 白名单放行回环。"""
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "127.0.0.1:8317")
    monkeypatch.setattr(api_main.socket, "getaddrinfo", _getaddrinfo_for("127.0.0.1"))
    fake_conn = _FakeConnection(
        response=_FakeResponse(body=json.dumps({"data": [{"id": "b"}]}).encode("utf-8"))
    )
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", lambda *args, **kwargs: fake_conn)

    models, url = api_main._fetch_available_models("http://127.0.0.1:8317/v1", "")

    assert models == ["b"]


def test_trusted_local_host_bare_hostname_still_rejected(monkeypatch):
    """受信本地主机必须钉死端口：裸 host 形式即使在白名单也拒绝。"""
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "host.docker.internal")
    monkeypatch.setattr(api_main.socket, "getaddrinfo", _getaddrinfo_for("172.17.0.1"))
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", _unexpected)

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models("http://host.docker.internal:8317/v1", "")


def test_trusted_local_host_wrong_port_rejected(monkeypatch):
    """白名单端口与实际端口不匹配时拒绝。"""
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "127.0.0.1:8317")
    monkeypatch.setattr(api_main.socket, "getaddrinfo", _getaddrinfo_for("127.0.0.1"))
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", _unexpected)

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models("http://127.0.0.1:9999/v1", "")


def test_trusted_local_host_metadata_ip_never_allowed(monkeypatch):
    """即便受信本地主机 + 白名单，云元数据地址仍无条件拦截。"""
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "host.docker.internal:8317")
    monkeypatch.setattr(api_main.socket, "getaddrinfo", _getaddrinfo_for("169.254.169.254"))
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", _unexpected)

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models("http://host.docker.internal:8317/v1", "")


def test_non_trusted_host_private_ip_still_rejected(monkeypatch):
    """非受信主机解析到私网地址仍被拦（fail-closed 回归）。"""
    monkeypatch.setenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, "public.example.com:8080")
    monkeypatch.setattr(api_main.socket, "getaddrinfo", _getaddrinfo_for("10.0.0.5"))
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", _unexpected)

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models("http://public.example.com:8080/v1", "")


# --- B：匿名回退仅限回环来源 ---

def _default_local_user():
    user = MagicMock()
    user.id = api_main._DEFAULT_LOCAL_USER_ID
    return user


@contextmanager
def _endpoint_client_with_user(user, client_addr=("testclient", 50000)):
    app = api_main.app

    def override_user():
        return user

    def override_db():
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        yield db

    app.dependency_overrides[api_main._require_api_user] = override_user
    app.dependency_overrides[api_main.get_db] = override_db
    try:
        yield TestClient(app, raise_server_exceptions=False, client=client_addr)
    finally:
        app.dependency_overrides.clear()


def test_anonymous_fetch_from_non_loopback_rejected():
    """默认本地账号从非回环来源调用 fetch → 401。"""
    with _endpoint_client_with_user(
        _default_local_user(), client_addr=("192.168.1.5", 50000)
    ) as client:
        resp = client.post("/v1/models/fetch", json={"base_url": "http://public.example.com/v1"})

    assert resp.status_code == 401


def test_anonymous_fetch_from_loopback_allowed(monkeypatch):
    """默认本地账号从回环调用时可继续，但不获得用户 URL 绕过权。"""
    captured = {}

    def fake_fetch(base_url, api_key, *, allow_user_url=False):
        captured.update(
            base_url=base_url,
            api_key=api_key,
            allow_user_url=allow_user_url,
        )
        return ["model-a"], "http://public.example.com/v1/models"

    monkeypatch.setattr(api_main, "_fetch_available_models", fake_fetch)
    with _endpoint_client_with_user(
        _default_local_user(), client_addr=("127.0.0.1", 50000)
    ) as client:
        resp = client.post("/v1/models/fetch", json={"base_url": "http://public.example.com/v1"})

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert captured == {
        "base_url": "http://public.example.com/v1",
        "api_key": "",
        "allow_user_url": False,
    }


def test_authenticated_fetch_from_non_loopback_uses_user_url_policy(monkeypatch):
    """真实用户从非回环来源调用时，HTTP 路由启用用户 URL 政策。"""
    captured = {}

    def fake_fetch(base_url, api_key, *, allow_user_url=False):
        captured.update(
            base_url=base_url,
            api_key=api_key,
            allow_user_url=allow_user_url,
        )
        return ["model-a"], "http://public.example.com/v1/models"

    monkeypatch.setattr(api_main, "_fetch_available_models", fake_fetch)
    user = MagicMock()
    user.id = "real-user-1"
    with _endpoint_client_with_user(user, client_addr=("192.168.1.5", 50000)) as client:
        resp = client.post("/v1/models/fetch", json={"base_url": "http://public.example.com/v1"})

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert captured == {
        "base_url": "http://public.example.com/v1",
        "api_key": "",
        "allow_user_url": True,
    }


def test_authenticated_user_cgnat_url_bypasses_allowlist(monkeypatch):
    """已登录用户的自有 Tailscale URL 不应被主机白名单拒绝。"""
    monkeypatch.delenv(api_main._MODELS_FETCH_ALLOWLIST_ENV, raising=False)
    monkeypatch.setattr(
        api_main.socket,
        "getaddrinfo",
        _getaddrinfo_for("100.65.130.33"),
    )
    fake_conn = _FakeConnection(
        response=_FakeResponse(body=json.dumps({"data": [{"id": "model-a"}]}).encode("utf-8"))
    )
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", lambda *args, **kwargs: fake_conn)

    models, url = api_main._fetch_available_models(
        "http://100.65.130.33:8317/v1",
        "",
        allow_user_url=True,
    )

    assert models == ["model-a"]
    assert url == "http://100.65.130.33:8317/v1/models"


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest",
        "http://[fd00:ec2::254]/latest",
        "http://[::ffff:169.254.169.254]/latest",
    ],
)
def test_authenticated_user_url_still_rejects_cloud_metadata(monkeypatch, url):
    """用户 URL 放行不能覆盖云元数据硬拦截。"""
    parsed = api_main._parse_models_fetch_url(url)
    monkeypatch.setattr(
        api_main.socket,
        "getaddrinfo",
        _getaddrinfo_for(parsed.hostname),
    )
    monkeypatch.setattr(api_main, "_build_models_fetch_connection", _unexpected)

    with pytest.raises(api_main._ModelsFetchError):
        api_main._fetch_available_models(url, "", allow_user_url=True)


def _call_authenticated_fetch_endpoint(monkeypatch, *, payload, user_cfg=None, provider=None):
    captured = {}
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = provider
    user = SimpleNamespace(id="real-user-1")
    request = Request({"type": "http", "client": ("192.168.1.5", 50000)})

    monkeypatch.setattr(
        api_main.auth_service,
        "get_user_llm_config",
        lambda db_arg, user_id: user_cfg,
    )
    monkeypatch.setattr(
        api_main.auth_service,
        "decrypt_secret",
        lambda value: f"decrypted:{value}" if value else None,
    )

    def fake_fetch(base_url, api_key, *, allow_user_url=False):
        captured.update(
            base_url=base_url,
            api_key=api_key,
            allow_user_url=allow_user_url,
        )
        return ["model-a"], f"{base_url.rstrip('/')}/models"

    monkeypatch.setattr(api_main, "_fetch_available_models", fake_fetch)
    monkeypatch.setattr(
        api_main.role_routing_service,
        "sync_model_profiles_from_names",
        lambda *args, **kwargs: None,
    )

    response = api_main.fetch_available_models(
        api_main.FetchModelsRequest(**payload),
        request,
        db,
        user,
    )
    assert response["ok"] is True
    return captured


def test_request_body_url_wins_over_saved_and_provider_urls(monkeypatch):
    """设置页未保存的输入值必须是本次拉取的目标。"""
    user_cfg = SimpleNamespace(
        backend_url="http://saved.example.com/v1",
        api_key_encrypted="user-key",
    )
    provider = SimpleNamespace(
        base_url="http://provider.example.com/v1",
        api_key_encrypted="provider-key",
    )

    captured = _call_authenticated_fetch_endpoint(
        monkeypatch,
        payload={
            "base_url": "http://input.example.com/v1",
            "provider_id": "provider-1",
        },
        user_cfg=user_cfg,
        provider=provider,
    )

    assert captured == {
        "base_url": "http://input.example.com/v1",
        "api_key": "decrypted:provider-key",
        "allow_user_url": True,
    }


def test_saved_backend_url_precedes_provider_url_and_uses_real_key_fields(monkeypatch):
    """请求体为空时先用用户 backend_url，Key 仍优先选中 provider。"""
    user_cfg = SimpleNamespace(
        backend_url="http://saved.example.com/v1",
        api_key_encrypted="user-key",
    )
    provider = SimpleNamespace(
        base_url="http://provider.example.com/v1",
        api_key_encrypted="provider-key",
    )

    captured = _call_authenticated_fetch_endpoint(
        monkeypatch,
        payload={"provider_id": "provider-1"},
        user_cfg=user_cfg,
        provider=provider,
    )

    assert captured == {
        "base_url": "http://saved.example.com/v1",
        "api_key": "decrypted:provider-key",
        "allow_user_url": True,
    }


def test_user_key_then_environment_key_fallback(monkeypatch):
    """无 provider Key 时读 api_key_encrypted，再回退 TA_API_KEY。"""
    user_cfg = SimpleNamespace(
        backend_url="http://saved.example.com/v1",
        api_key_encrypted="user-key",
    )
    captured = _call_authenticated_fetch_endpoint(
        monkeypatch,
        payload={},
        user_cfg=user_cfg,
    )
    assert captured["api_key"] == "decrypted:user-key"

    monkeypatch.setenv("TA_API_KEY", "environment-key")
    captured = _call_authenticated_fetch_endpoint(
        monkeypatch,
        payload={"base_url": "http://input.example.com/v1"},
        user_cfg=None,
    )
    assert captured["api_key"] == "environment-key"


def test_authenticated_fetch_without_user_url_uses_localhost_default(monkeypatch):
    """已登录用户没有任何自有 URL 时使用可解析的 localhost 默认值。"""
    captured = _call_authenticated_fetch_endpoint(
        monkeypatch,
        payload={},
        user_cfg=None,
        provider=None,
    )

    assert captured["base_url"] == "http://localhost:8317/v1"
    assert captured["allow_user_url"] is False
