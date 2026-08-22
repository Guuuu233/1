"""KNOWN_ISSUES #1 — vendor chain typed-result semantics + EM backup sources.

Covers the three-outcome / two-behavior collapse fix:

- ``VendorRefuse``  -> chain stops (no silent fallthrough to a date-blind vendor)
- ``VendorFail``    -> chain falls through to the next vendor
- ``VendorEmpty``   -> confirmed empty, chain stops
- ``VendorOk``      -> explicit success
- plain string      -> backward-compatible success hit

And the Eastmoney backup-source work (get_board_fund_flow / get_individual_fund_flow /
get_lhb_detail fall back to THS / Sina when the EM interface fails).
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.dataflows import interface as iface
from tradingagents.dataflows.providers.base import ProviderResourcePolicy
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.trade_calendar import CN_TZ, cn_today_str
from tradingagents.dataflows.vendor_result import (
    VendorEmpty,
    VendorFail,
    VendorOk,
    VendorRefuse,
    result_to_prompt,
)

FAST_POLICY = ProviderResourcePolicy(
    timeout_seconds=1.0, max_retries=0, max_concurrency=2
)


class _FakeProvider:
    def __init__(self, name, func, *, placeholder: bool = False, method: str = "get_stock_data"):
        self.name = name
        self.is_placeholder = placeholder
        self._func = func
        self._method = method

    def __getattr__(self, attr: str):
        if attr == self._method:
            return self._func
        raise AttributeError(attr)


class _FakeRegistry:
    def __init__(self, providers):
        self._providers = providers

    def list_names(self):
        return list(self._providers)

    def get(self, name):
        return self._providers.get(name)

    def resource_policy(self, name):
        return FAST_POLICY


_ROUTER_SAMPLE_ARGS = {
    "get_stock_data": ("600519", "2026-01-01", "2026-01-31"),
}


def _route(chain: dict[str, object], configured: str = "p1,p2", method: str = "get_stock_data"):
    """Run route_to_vendor against an in-memory registry of fake providers."""
    registry = _FakeRegistry(chain)
    args = _ROUTER_SAMPLE_ARGS[method]
    with patch.object(iface, "_registry", registry), \
         patch.object(iface, "get_vendor", return_value=configured):
        return iface.route_to_vendor(method, *args)


# ── Router: typed result semantics ────────────────────────────────────


def test_vendor_refuse_stops_chain_without_fallthrough():
    refused = _FakeProvider("p1", lambda *a, **k: VendorRefuse("snapshot-only"))
    second = _FakeProvider(
        "p2",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    out = _route({"p1": refused, "p2": second}, "p1,p2")
    assert out == "snapshot-only"


def test_vendor_fail_falls_through_to_next_vendor():
    failing = _FakeProvider("p1", lambda *a, **k: VendorFail("push2his down"))
    ok = _FakeProvider("p2", lambda *a, **k: "fallback csv")
    out = _route({"p1": failing, "p2": ok}, "p1,p2")
    assert out == "fallback csv"


def test_vendor_empty_stops_chain_and_reports_confirmed_none():
    empty = _FakeProvider("p1", lambda *a, **k: VendorEmpty("No news found for 600519"))
    second = _FakeProvider(
        "p2",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    out = _route({"p1": empty, "p2": second}, "p1,p2")
    assert out == "No news found for 600519"


def test_vendor_ok_returns_payload():
    ok = _FakeProvider("p1", lambda *a, **k: VendorOk("## live news"))
    out = _route({"p1": ok}, "p1")
    assert out == "## live news"


def test_plain_string_is_backward_compatible_hit():
    hit = _FakeProvider("p1", lambda *a, **k: "plain csv")
    second = _FakeProvider(
        "p2",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    out = _route({"p1": hit, "p2": second}, "p1,p2")
    assert out == "plain csv"


def test_vendor_refuse_with_allow_peers_continues_only_through_peers():
    refused = _FakeProvider(
        "p1",
        lambda *a, **k: VendorRefuse(
            "near-window only", allow_peers=("p3",)
        ),
    )
    p2 = _FakeProvider(
        "p2",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("p2 must be skipped")),
    )
    p3 = _FakeProvider("p3", lambda *a, **k: "## historical news")
    out = _route({"p1": refused, "p2": p2, "p3": p3}, "p1,p2,p3")
    assert out == "## historical news"


def test_vendor_refuse_with_allow_peers_returns_refusal_when_peers_fail():
    refused = _FakeProvider(
        "p1",
        lambda *a, **k: VendorRefuse(
            "near-window only", allow_peers=("p3",)
        ),
    )
    p2 = _FakeProvider(
        "p2",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("p2 must be skipped")),
    )
    p3 = _FakeProvider("p3", lambda *a, **k: VendorFail("peer also down"))
    out = _route({"p1": refused, "p2": p2, "p3": p3}, "p1,p2,p3")
    assert out == "near-window only"


def test_vendor_fail_at_chain_end_still_raises_runtime_error():
    failing = _FakeProvider("p1", lambda *a, **k: VendorFail("all down"))
    with pytest.raises(RuntimeError, match="No available vendor"):
        _route({"p1": failing}, "p1")


# ── Router: exception fallback preserved ──────────────────────────────


def test_exception_still_falls_back_to_next_vendor():
    broken = _FakeProvider("p1", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    ok = _FakeProvider("p2", lambda *a, **k: "fallback csv")
    out = _route({"p1": broken, "p2": ok}, "p1,p2")
    assert out == "fallback csv"


# ── Provider: cn_akshare get_global_news typed semantics ──────────────


def _provider_with_sina(result: str):
    p = CnAkshareProvider()
    p.get_sina_global_news = MagicMock(return_value=result)
    return p


def test_akshare_global_news_sina_failure_is_vendor_fail(frozen_trade_date):
    p = _provider_with_sina("新浪财经快讯获取失败：ConnectionError: boom")
    out = p.get_global_news(frozen_trade_date)
    assert isinstance(out, VendorFail)
    assert "新浪财经快讯获取失败" in out.error


def test_akshare_global_news_sina_empty_is_vendor_empty(frozen_trade_date):
    p = _provider_with_sina("未获取到新浪财经快讯")
    today = frozen_trade_date
    out = p.get_global_news(today)
    assert isinstance(out, VendorEmpty)
    assert "未获取到全球市场新闻" in out.message


def test_akshare_global_news_sina_hit_returns_string(frozen_trade_date):
    p = _provider_with_sina("## 新浪财经快讯（第1页，共3条）：\n### [10:00] 标题")
    out = p.get_global_news(frozen_trade_date)
    assert isinstance(out, str)
    assert out.startswith("## ")


def test_akshare_global_news_signature_accepts_look_back_and_limit(frozen_trade_date):
    """Signature must match the base/router call shape (curr_date, look_back_days, limit)."""
    p = _provider_with_sina("## ok")
    out = p.get_global_news(frozen_trade_date, look_back_days=7, limit=50)
    assert out.startswith("## ")


def test_akshare_get_news_empty_is_vendor_empty():
    class _EmptyAk:
        def stock_news_em(self, symbol):
            return pd.DataFrame()

    p = CnAkshareProvider()
    p._ak = lambda: _EmptyAk()
    out = p.get_news("600519", "2026-08-01", "2026-08-04")
    assert isinstance(out, VendorEmpty)
    assert "No news found" in out.message


def test_akshare_historical_news_requires_parseable_timestamps():
    class _NoDateAk:
        def stock_news_em(self, symbol):
            return pd.DataFrame(
                {
                    "新闻标题": ["无时间"],
                    "新闻内容": ["body"],
                }
            )

    p = CnAkshareProvider()
    p._ak = lambda: _NoDateAk()
    out = p.get_news("600519", "2026-08-01", "2026-08-04")
    assert isinstance(out, VendorFail)
    assert "发布时间" in out.error


def test_akshare_historical_news_filters_future_timestamps():
    class _NewsAk:
        def stock_news_em(self, symbol):
            return pd.DataFrame(
                {
                    "发布时间": ["2026-08-04 12:00:00", "2026-08-05 12:00:00"],
                    "新闻标题": ["kept", "future"],
                    "新闻内容": ["y", "z"],
                }
            )

    p = CnAkshareProvider()
    p._ak = lambda: _NewsAk()
    out = p.get_news("600519", "2026-08-01", "2026-08-04")
    assert "kept" in out
    assert "future" not in out
    assert "2026-08-04 12:00:00" in out


def test_akshare_historical_news_rejects_invalid_timestamps():
    class _NewsAk:
        def stock_news_em(self, symbol):
            return pd.DataFrame(
                {
                    "发布时间": ["bad", "2026-08-04 12:00:00"],
                    "新闻标题": ["invalid", "kept"],
                    "新闻内容": ["x", "y"],
                }
            )

    p = CnAkshareProvider()
    p._ak = lambda: _NewsAk()
    out = p.get_news("600519", "2026-08-01", "2026-08-04")
    assert isinstance(out, VendorFail)
    assert "发布时间" in out.error


# ── Provider: yfinance typed semantics ────────────────────────────────


def test_yfinance_news_error_string_is_vendor_fail():
    from tradingagents.dataflows.providers.yfinance_provider import _classify_text_result

    out = _classify_text_result(
        "Error fetching news for 600519.SS: timeout", empty_prefixes=("No news found",), fail_prefixes=("Error fetching news",)
    )
    assert isinstance(out, VendorFail)
    assert "timeout" in out.error


def test_yfinance_news_empty_string_is_vendor_empty():
    from tradingagents.dataflows.providers.yfinance_provider import _classify_text_result

    out = _classify_text_result(
        "No news found for 600519.SS", empty_prefixes=("No news found",), fail_prefixes=("Error fetching news",)
    )
    assert isinstance(out, VendorEmpty)


def test_yfinance_news_success_passthrough():
    from tradingagents.dataflows.providers.yfinance_provider import _classify_text_result

    out = _classify_text_result(
        "## 600519.SS News, from 2026-08-01 to 2026-08-04:\n\n### headline",
        empty_prefixes=("No news found",),
        fail_prefixes=("Error fetching news",),
    )
    assert out.startswith("## ")


def test_yfinance_get_insider_transactions_error_is_vendor_fail():
    from tradingagents.dataflows.providers.yfinance_provider import YFinanceProvider

    p = YFinanceProvider()
    with patch(
        "tradingagents.dataflows.providers.yfinance_provider.get_yfinance_insider_transactions",
        return_value="Error retrieving insider transactions for 600519.SS: boom",
    ):
        out = p.get_insider_transactions("600519")
    assert isinstance(out, VendorFail)


# ── Router integration: akshare fail -> yfinance serves global news ───


def test_router_akshare_global_news_fail_falls_to_yfinance(frozen_trade_date):
    """The classic KNOWN_ISSUES case: akshare sina failure must not look like
    'confirmed no news' and block the next vendor."""
    ak = _FakeProvider(
        "cn_akshare",
        lambda *a, **k: VendorFail("新浪财经快讯获取失败：ConnectionError"),
        method="get_global_news",
    )
    yf = _FakeProvider(
        "yfinance",
        lambda *a, **k: "## yfinance global news",
        method="get_global_news",
    )
    registry = _FakeRegistry({"cn_akshare": ak, "yfinance": yf})
    with patch.object(iface, "_registry", registry), \
         patch.object(iface, "get_vendor", return_value="cn_akshare,yfinance"):
        out = iface.route_to_vendor("get_global_news", frozen_trade_date, 7, 10)
    assert out == "## yfinance global news"


def test_router_akshare_global_news_empty_stops_chain(frozen_trade_date):
    """Confirmed empty from akshare stops the chain (does not fall to yfinance)."""
    ak = _FakeProvider(
        "cn_akshare",
        lambda *a, **k: VendorEmpty("未获取到全球市场新闻"),
        method="get_global_news",
    )
    yf = _FakeProvider(
        "yfinance",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")),
        method="get_global_news",
    )
    registry = _FakeRegistry({"cn_akshare": ak, "yfinance": yf})
    with patch.object(iface, "_registry", registry), \
         patch.object(iface, "get_vendor", return_value="cn_akshare,yfinance"):
        out = iface.route_to_vendor("get_global_news", frozen_trade_date, 7, 10)
    assert out == "未获取到全球市场新闻"


def test_router_global_news_fallback_with_dynamic_today():
    """Deterministic regression for the two 2026-08-04 date bombs above.

    The two router tests originally routed get_global_news with a hardcoded
    calendar date standing in for 'today'. Once the real date passed it, the
    router began refusing the call as historical near-window news and the
    fall-through assertions failed. This test freezes the clock and routes
    with cn_today_str(), so the as-of date is guaranteed to equal the frozen
    today and the vendor chain semantics are exercised forever.
    """
    ak = _FakeProvider(
        "cn_akshare",
        lambda *a, **k: VendorFail("新浪财经快讯获取失败：ConnectionError"),
        method="get_global_news",
    )
    yf = _FakeProvider(
        "yfinance",
        lambda *a, **k: "## yfinance global news",
        method="get_global_news",
    )
    registry = _FakeRegistry({"cn_akshare": ak, "yfinance": yf})
    frozen_today = datetime(2026, 8, 5, 0, 1, tzinfo=CN_TZ)
    with patch.object(iface, "_registry", registry), \
         patch.object(iface, "get_vendor", return_value="cn_akshare,yfinance"), \
         patch("tradingagents.dataflows.trade_calendar.now_cn", return_value=frozen_today):
        out = iface.route_to_vendor("get_global_news", cn_today_str(), 7, 10)
    assert out == "## yfinance global news"


# ── result_to_prompt helper ───────────────────────────────────────────


def test_result_to_prompt_unwraps_typed_results():
    assert result_to_prompt("plain") == "plain"
    assert result_to_prompt(123) == "123"
    assert result_to_prompt(VendorOk("payload")) == "payload"
    assert result_to_prompt(VendorEmpty("empty")) == "empty"
    assert result_to_prompt(VendorRefuse("refuse")) == "refuse"
    assert result_to_prompt(VendorFail("fail")) == "fail"
