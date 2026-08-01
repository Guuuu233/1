"""Tests for the provider smoke script's exit/failure semantics (DAV-29).

The script must fail (non-zero exit) whenever any data item fails, even when
other items pass, and must report the specific failing item and error instead
of silently swallowing exceptions. Live-mode failures include falsy/error-shaped
results such as {"ok": false} and "No data found", not just raised exceptions.
"""

import importlib.util
import io
import os
import sys
import types

import pytest


SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "smoke_providers.py",
)


class _Provider:
    def __init__(self, name, methods):
        self.name = name
        self.is_placeholder = False
        for method in methods:
            setattr(self, method, lambda *a, **k: {"ok": True})


class _FakeRegistry:
    def __init__(self, providers):
        self._providers = {p.name: p for p in providers}

    def get(self, name):
        return self._providers.get(name)

    def list_names(self):
        return list(self._providers.keys())


@pytest.fixture()
def smoke():
    """Load scripts/smoke_providers.py against a stub dataflows interface."""
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "tradingagents" or name.startswith("tradingagents.")
    }
    original_path = list(sys.path)
    registry_holder = {}
    stub_interface = types.ModuleType("tradingagents.dataflows.interface")
    stub_interface._registry = _FakeRegistry([])
    registry_holder["registry"] = stub_interface._registry
    stub_interface._registry_holder = registry_holder

    def _route_to_vendor(method, *args, **kwargs):
        registry = registry_holder["registry"]
        for provider in registry.list_names():
            fn = getattr(registry.get(provider), method, None)
            if fn is not None:
                return fn(*args, **kwargs)
        return {"ok": False}

    stub_interface.route_to_vendor = _route_to_vendor
    sys.modules["tradingagents.dataflows.interface"] = stub_interface

    spec = importlib.util.spec_from_file_location("smoke_providers", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        module._registry_holder = registry_holder
        yield module
    finally:
        for name in list(sys.modules):
            if (
                (name == "tradingagents" or name.startswith("tradingagents."))
                and name not in original_modules
            ):
                sys.modules.pop(name, None)
        for name, module in original_modules.items():
            sys.modules[name] = module
        sys.modules.pop("smoke_providers", None)
        sys.path[:] = original_path


@pytest.fixture()
def offline_env(monkeypatch):
    monkeypatch.delenv("LIVE", raising=False)


@pytest.fixture()
def live_env(monkeypatch):
    monkeypatch.setenv("LIVE", "1")


def _captured_run(module):
    """Capture stdout and SystemExit(rc) as (rc, output)."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        try:
            module.run_smoke()
            rc = 0
        except SystemExit as exc:
            rc = exc.code
    finally:
        sys.stdout = old_stdout
    return rc, buf.getvalue()


def _registry_with(module, methods_by_provider):
    registry = _FakeRegistry(
        [
            _Provider(name, methods)
            for name, methods in methods_by_provider.items()
        ]
    )
    module._registry = registry
    module._registry_holder["registry"] = registry


def test_all_items_pass_in_offline_mode_exits_zero(smoke, offline_env):
    _registry_with(
        smoke,
        {"stub": [method_name for _, method_name, _ in smoke.METHODS]},
    )

    rc, output = _captured_run(smoke)

    assert rc == 0
    assert output.count("✅ 正常") == len(smoke.METHODS)
    assert "smoke-result: PASS" in output


def test_partial_failure_is_failure_and_exits_nonzero(smoke, offline_env):
    # Provide every method except get_news (新闻资讯): 13 pass, 1 fails.
    missing = "get_news"
    present = [m for _, m, _ in smoke.METHODS if m != missing]
    _registry_with(smoke, {"stub": present})

    rc, output = _captured_run(smoke)

    assert rc == 1
    assert "smoke-result: FAIL" in output
    assert "❌ 失败" in output
    assert "新闻资讯/get_news" in output
    assert "registry 无 provider 实现该方法" in output
    # The pass count is still reported per item; the status must not say OK
    # merely because some symbols passed.
    assert output.count("✅ 正常") == len(smoke.METHODS) - 1


def test_exception_is_reported_and_exits_nonzero(smoke, live_env):
    # LIVE mode: every method is provided, but get_stock_data raises; the
    # failure must be surfaced with the specific symbol/error, not swallowed,
    # while the other 13 items still pass.
    def boom(*args, **kwargs):
        raise RuntimeError("provider boom")

    all_methods = [method_name for _, method_name, _ in smoke.METHODS]
    _registry_with(smoke, {"stub": all_methods})
    smoke._registry.get("stub").get_stock_data = boom
    rc, output = _captured_run(smoke)

    assert rc == 1
    assert "smoke-result: FAIL" in output
    assert "K线行情/get_stock_data" in output
    assert "RuntimeError: provider boom" in output
    assert output.count("❌ 失败") == 1
    assert output.count("✅ 正常") == len(smoke.METHODS) - 1


def test_live_dict_ok_false_is_failure_and_exits_nonzero(smoke, live_env):
    # LIVE mode: get_news returns an explicit {"ok": false}; it must be treated
    # as a failure even though no exception was raised.
    all_methods = [method_name for _, method_name, _ in smoke.METHODS]
    _registry_with(smoke, {"stub": all_methods})
    smoke._registry.get("stub").get_news = lambda *a, **k: {"ok": False}

    rc, output = _captured_run(smoke)

    assert rc == 1
    assert "smoke-result: FAIL" in output
    assert "新闻资讯/get_news" in output
    assert "返回失败结果" in output
    assert "{'ok': False}" in output
    assert output.count("❌ 失败") == 1
    assert output.count("✅ 正常") == len(smoke.METHODS) - 1


def test_live_no_data_found_string_is_failure_and_exits_nonzero(smoke, live_env):
    # LIVE mode: a "No data found" string is a failed result, not success.
    all_methods = [method_name for _, method_name, _ in smoke.METHODS]
    _registry_with(smoke, {"stub": all_methods})
    smoke._registry.get("stub").get_news = lambda *a, **k: "No data found"

    rc, output = _captured_run(smoke)

    assert rc == 1
    assert "smoke-result: FAIL" in output
    assert "新闻资讯/get_news" in output
    assert "No data found" in output
    assert output.count("❌ 失败") == 1
    assert output.count("✅ 正常") == len(smoke.METHODS) - 1


def test_live_success_results_exit_zero(smoke, live_env):
    # LIVE mode: success-shaped results ({"ok": true}) must not be misreported
    # as failures.
    all_methods = [method_name for _, method_name, _ in smoke.METHODS]
    _registry_with(smoke, {"stub": all_methods})

    rc, output = _captured_run(smoke)

    assert rc == 0
    assert output.count("✅ 正常") == len(smoke.METHODS)
    assert "smoke-result: PASS" in output
    assert "❌ 失败" not in output
