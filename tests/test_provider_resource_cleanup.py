"""Offline tests for provider timeout, retry, and concurrency cleanup."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from unittest.mock import patch

from tradingagents.dataflows import interface as iface
from tradingagents.dataflows.providers.base import ProviderResourcePolicy
from tradingagents.dataflows.providers.registry import (
    DataProviderRegistry,
    build_default_registry,
)


class _FakeProvider:
    def __init__(self, name, func):
        self.name = name
        self.is_placeholder = False
        self._func = func

    def get_stock_data(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class _FakeRegistry:
    def __init__(self, providers, policy):
        self._providers = providers
        self._policy = policy

    def list_names(self):
        return list(self._providers)

    def get(self, name):
        return self._providers.get(name)

    def resource_policy(self, name):
        return self._policy


def test_default_registry_exposes_valid_resource_policies():
    registry = build_default_registry()
    policies = registry.list_resource_policies()

    assert set(policies) == set(registry.list_names())
    for policy in policies.values():
        assert policy.timeout_seconds > 0
        assert policy.max_retries >= 0
        assert policy.max_concurrency >= 1


def test_registry_accepts_explicit_resource_policy():
    registry = DataProviderRegistry()
    provider = _FakeProvider("custom", lambda *args, **kwargs: "ok")
    registry.register(
        provider,
        ProviderResourcePolicy(timeout_seconds=2.0, max_retries=3, max_concurrency=1),
    )

    policy = registry.resource_policy("custom")
    assert policy.timeout_seconds == 2.0
    assert policy.max_retries == 3
    assert policy.max_concurrency == 1


def test_route_timeout_falls_back_to_next_provider():
    started = threading.Event()
    done = threading.Event()

    def slow(*args, **kwargs):
        started.set()
        try:
            time.sleep(0.05)
            return "slow"
        finally:
            done.set()

    fast = _FakeProvider("yfinance", lambda *args, **kwargs: "fast")
    providers = {
        "cn_akshare": _FakeProvider("cn_akshare", slow),
        "yfinance": fast,
    }
    registry = _FakeRegistry(
        providers,
        ProviderResourcePolicy(timeout_seconds=0.01, max_retries=0, max_concurrency=2),
    )

    with patch.object(iface, "_registry", registry), \
         patch.object(iface, "get_vendor", return_value="cn_akshare,yfinance"):
        out = iface.route_to_vendor(
            "get_stock_data", "600519", "2025-01-01", "2025-01-31"
        )

    assert out == "fast"
    assert started.is_set()
    assert done.wait(0.2)


def test_route_timeout_retries_then_falls_back():
    calls = 0
    done = threading.Event()

    def slow(*args, **kwargs):
        nonlocal calls
        calls += 1
        try:
            time.sleep(0.05)
            return None
        finally:
            done.set()

    fast = _FakeProvider("yfinance", lambda *args, **kwargs: "fast")
    providers = {
        "cn_akshare": _FakeProvider("cn_akshare", slow),
        "yfinance": fast,
    }
    registry = _FakeRegistry(
        providers,
        ProviderResourcePolicy(timeout_seconds=0.01, max_retries=1, max_concurrency=2),
    )

    with patch.object(iface, "_registry", registry), \
         patch.object(iface, "get_vendor", return_value="cn_akshare,yfinance"):
        out = iface.route_to_vendor(
            "get_stock_data", "600519", "2025-01-01", "2025-01-31"
        )

    assert out == "fast"
    assert calls == 2
    assert done.wait(0.2)


def test_route_provider_error_retries_then_falls_back():
    calls = 0

    def flaky(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider down")

    fast = _FakeProvider("yfinance", lambda *args, **kwargs: "fast")
    providers = {
        "cn_akshare": _FakeProvider("cn_akshare", flaky),
        "yfinance": fast,
    }
    registry = _FakeRegistry(
        providers,
        ProviderResourcePolicy(timeout_seconds=1.0, max_retries=1, max_concurrency=2),
    )

    with patch.object(iface, "_registry", registry), \
         patch.object(iface, "get_vendor", return_value="cn_akshare,yfinance"):
        out = iface.route_to_vendor(
            "get_stock_data", "600519", "2025-01-01", "2025-01-31"
        )

    assert out == "fast"
    assert calls == 2


def test_route_enforces_per_provider_concurrency_bound():
    lock = threading.Lock()
    active = 0
    max_active = 0

    def work(*args, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.03)
            return "ok"
        finally:
            with lock:
                active -= 1

    provider = _FakeProvider("cn_akshare", work)
    registry = _FakeRegistry(
        {"cn_akshare": provider},
        ProviderResourcePolicy(timeout_seconds=1.0, max_retries=0, max_concurrency=2),
    )
    executor = ThreadPoolExecutor(max_workers=8)
    try:
        with patch.object(iface, "_registry", registry), \
             patch.object(iface, "get_vendor", return_value="cn_akshare"), \
             patch.object(iface, "_PROVIDER_CALL_EXECUTOR", executor), \
             patch.object(iface, "_PROVIDER_SEMAPHORES", {}):
            with ThreadPoolExecutor(max_workers=8) as caller_pool:
                futures = [
                    caller_pool.submit(
                        lambda: iface.route_to_vendor(
                            "get_stock_data",
                            "600519",
                            "2025-01-01",
                            "2025-01-31",
                        )
                    )
                    for _ in range(8)
                ]
                results = [future.result(timeout=3) for future in futures]
    finally:
        executor.shutdown(wait=True)

    assert results == ["ok"] * 8
    assert 0 < max_active <= 2


def test_provider_call_executor_registers_atexit_shutdown():
    import atexit
    import importlib

    # Reload in a controlled way so we can observe the registration call while
    # avoiding private atexit internals. The old executor is shut down first so
    # it cannot leave idle worker threads behind.
    old_executor = iface._PROVIDER_CALL_EXECUTOR
    old_executor.shutdown(wait=True, cancel_futures=True)
    with patch("atexit.register") as register:
        reloaded = importlib.reload(iface)
        registered = register.call_args.args[0]

    assert registered.__name__ == "_shutdown_provider_call_executor"
    assert register.call_args.kwargs == {}
    assert reloaded._PROVIDER_CALL_EXECUTOR is not old_executor
    # The reload above ran under a mocked atexit, so register the replacement
    # executor for real as well; otherwise later tests would leave it unmanaged.
    atexit.register(reloaded._shutdown_provider_call_executor)
