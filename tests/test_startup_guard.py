"""Startup guard tests (DAV-66).

The API server and the scheduler must refuse to start when TA_APP_SECRET_KEY is
unset, unless the operator explicitly opted into the insecure built-in default
key for local development with ``TA_ALLOW_DEFAULT_SECRET=1``.

conftest.py sets TA_ALLOW_DEFAULT_SECRET=1 for the offline suite, so every test
here manipulates the env explicitly with monkeypatch to exercise each branch.
"""
from __future__ import annotations

import pytest

from api.services import auth_service

TRUTHY = ["1", "true", "TRUE", "yes", "on"]
FALSY = ["", "0", "false", "no", "off", "anything-else"]


def _unset_key(monkeypatch) -> None:
    monkeypatch.delenv("TA_APP_SECRET_KEY", raising=False)


def _unset_optout(monkeypatch) -> None:
    monkeypatch.delenv(auth_service.ALLOW_DEFAULT_SECRET_ENV, raising=False)


class TestEnsureSecureSecretConfigured:
    def test_refuses_when_key_missing_and_no_optout(self, monkeypatch):
        _unset_key(monkeypatch)
        _unset_optout(monkeypatch)
        with pytest.raises(RuntimeError, match="TA_APP_SECRET_KEY"):
            auth_service.ensure_secure_secret_configured()

    def test_allows_when_key_is_set(self, monkeypatch):
        monkeypatch.setenv("TA_APP_SECRET_KEY", "super-secret")
        _unset_optout(monkeypatch)
        auth_service.ensure_secure_secret_configured()  # must not raise

    @pytest.mark.parametrize("value", TRUTHY)
    def test_allows_when_optout_is_truthy(self, monkeypatch, value):
        _unset_key(monkeypatch)
        monkeypatch.setenv(auth_service.ALLOW_DEFAULT_SECRET_ENV, value)
        auth_service.ensure_secure_secret_configured()  # must not raise

    @pytest.mark.parametrize("value", FALSY)
    def test_refuses_when_optout_is_falsy(self, monkeypatch, value):
        _unset_key(monkeypatch)
        monkeypatch.setenv(auth_service.ALLOW_DEFAULT_SECRET_ENV, value)
        with pytest.raises(RuntimeError, match="TA_APP_SECRET_KEY"):
            auth_service.ensure_secure_secret_configured()


class TestAllowDefaultSecretParsing:
    @pytest.mark.parametrize("value", TRUTHY)
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv(auth_service.ALLOW_DEFAULT_SECRET_ENV, value)
        assert auth_service.allow_default_secret() is True

    @pytest.mark.parametrize("value", FALSY)
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv(auth_service.ALLOW_DEFAULT_SECRET_ENV, value)
        assert auth_service.allow_default_secret() is False

    def test_unset_defaults_to_false(self, monkeypatch):
        _unset_optout(monkeypatch)
        assert auth_service.allow_default_secret() is False


class TestLifespanRefusal:
    """End-to-end: FastAPI lifespan (and thus server startup) is refused."""

    def _client(self):
        from fastapi.testclient import TestClient
        from api import main as main_mod

        return TestClient(main_mod.app, raise_server_exceptions=False)

    def test_server_refuses_to_start_without_key_or_optout(self, monkeypatch):
        _unset_key(monkeypatch)
        _unset_optout(monkeypatch)
        with pytest.raises(RuntimeError, match="TA_APP_SECRET_KEY"):
            with self._client():
                pass  # pragma: no cover - lifespan must not start

    def test_server_starts_when_key_is_set(self, monkeypatch):
        monkeypatch.setenv("TA_APP_SECRET_KEY", "super-secret")
        _unset_optout(monkeypatch)
        with self._client() as client:
            resp = client.get("/healthz")
            assert resp.status_code == 200

    def test_server_starts_with_explicit_dev_optout(self, monkeypatch):
        _unset_key(monkeypatch)
        monkeypatch.setenv(auth_service.ALLOW_DEFAULT_SECRET_ENV, "1")
        with self._client() as client:
            resp = client.get("/healthz")
            assert resp.status_code == 200
