"""Historical news routes only through providers with verifiable as-of windows."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.dataflows import interface as iface
from tradingagents.dataflows.providers.cn_cls_provider import CnClsProvider
from tradingagents.dataflows.trade_calendar import cn_today_str, now_cn
from tradingagents.dataflows.vendor_result import VendorFail, VendorRefuse


@pytest.fixture
def past_date() -> str:
    return (now_cn().date() - timedelta(days=90)).isoformat()


def test_global_news_historical_routes_only_to_investoday():
    akshare = MagicMock()
    akshare.get_global_news.side_effect = AssertionError(
        "historical requests must not call the live Sina provider"
    )
    investoday = MagicMock()
    investoday.get_global_news.return_value = (
        "## 全球市场新闻（来源：今日投资；数据窗口：2026-07-28 至 2026-08-11）：\n"
        "### 历史宏观条目"
    )
    yfinance = MagicMock()
    yfinance.get_global_news.side_effect = AssertionError(
        "historical requests must not call yfinance current news"
    )

    with patch(
        "tradingagents.dataflows.trade_calendar.now_cn",
        return_value=datetime(2026, 8, 12, 12, 0, 0),
    ), patch.object(iface, "_registry") as reg:
        reg.list_names.return_value = ["cn_akshare", "cn_investoday", "yfinance"]
        reg.get.side_effect = {
            "cn_akshare": akshare,
            "cn_investoday": investoday,
            "yfinance": yfinance,
        }.get
        with patch.object(iface, "get_vendor", return_value="cn_akshare,yfinance"):
            out = iface.route_to_vendor(
                "get_global_news", "2026-08-11", look_back_days=14, limit=5
            )

    assert "来源：今日投资" in out
    assert "数据窗口：2026-07-28 至 2026-08-11" in out
    investoday.get_global_news.assert_called_once_with(
        "2026-08-11", look_back_days=14, limit=5
    )
    akshare.get_global_news.assert_not_called()
    yfinance.get_global_news.assert_not_called()


def test_global_news_historical_investoday_failure_is_explicit_without_fallback(
    past_date,
):
    akshare = MagicMock()
    akshare.get_global_news.side_effect = AssertionError(
        "live fallback must not be called"
    )
    investoday = MagicMock()
    investoday.get_global_news.side_effect = NotImplementedError(
        "cn_investoday needs API Key"
    )

    with patch.object(iface, "_registry") as reg:
        reg.list_names.return_value = ["cn_akshare", "cn_investoday"]
        reg.get.side_effect = {
            "cn_akshare": akshare,
            "cn_investoday": investoday,
        }.get
        with patch.object(iface, "get_vendor", return_value="cn_akshare"):
            out = iface.route_to_vendor(
                "get_global_news", past_date, look_back_days=14, limit=5
            )

    assert out.startswith("【数据获取失败】")
    assert "不得回退到实时新闻源" in out
    investoday.get_global_news.assert_called_once_with(
        past_date, look_back_days=14, limit=5
    )
    akshare.get_global_news.assert_not_called()


def test_global_news_historical_malformed_or_network_failure_is_explicit(
    past_date,
):
    investoday = MagicMock()
    investoday.get_global_news.side_effect = OSError("network down")

    with patch.object(iface, "_registry") as reg:
        reg.list_names.return_value = ["cn_investoday", "yfinance"]
        reg.get.side_effect = {
            "cn_investoday": investoday,
            "yfinance": MagicMock(),
        }.get
        with patch.object(iface, "get_vendor", return_value="cn_investoday,yfinance"):
            out = iface.route_to_vendor(
                "get_global_news", past_date, look_back_days=14, limit=5
            )

    assert out.startswith("【数据获取失败】")
    assert "可验证数据" in out
    assert investoday.get_global_news.call_args_list == [
        ((past_date,), {"look_back_days": 14, "limit": 5}),
        ((past_date,), {"look_back_days": 14, "limit": 5}),
    ]


def test_news_historical_routes_only_to_verified_sources(past_date):
    akshare = MagicMock()
    akshare.get_news.return_value = "## 600519 历史新闻（2026-04-30 至 2026-05-14）："
    investoday = MagicMock()
    investoday.get_news.side_effect = AssertionError("must stop after verified hit")
    yfinance = MagicMock()
    yfinance.get_news.side_effect = AssertionError("live news must not receive historical as-of")

    start_date = (now_cn().date() - timedelta(days=104)).isoformat()
    with patch.object(iface, "_registry") as reg:
        reg.list_names.return_value = ["cn_akshare", "cn_investoday", "yfinance"]
        reg.get.side_effect = {
            "cn_akshare": akshare,
            "cn_investoday": investoday,
            "yfinance": yfinance,
        }.get
        with patch.object(iface, "get_vendor", return_value="cn_akshare,cn_investoday,yfinance"):
            out = iface.route_to_vendor("get_news", "600519", start_date, past_date)

    assert "历史新闻" in out
    akshare.get_news.assert_called_once_with("600519", start_date, past_date)
    investoday.get_news.assert_not_called()
    yfinance.get_news.assert_not_called()


def test_cn_cls_unsupported_stock_news_allows_peer_fallback():
    cls = MagicMock()
    cls.get_news.return_value = VendorRefuse(
        "cn_cls unsupported ticker", allow_peers=("cn_akshare", "cn_investoday")
    )
    akshare = MagicMock()
    akshare.get_news.return_value = "## verified stock news"
    with patch.object(iface, "_registry") as reg:
        reg.list_names.return_value = ["cn_cls", "cn_akshare", "cn_investoday"]
        reg.get.side_effect = {"cn_cls": cls, "cn_akshare": akshare}.get
        with patch.object(iface, "get_vendor", return_value="cn_cls,cn_akshare"):
            out = iface.route_to_vendor(
                "get_news", "600519", "2026-01-01", "2026-01-02"
            )
    assert out == "## verified stock news"
    cls.get_news.assert_called_once()
    akshare.get_news.assert_called_once()


def test_cn_cls_historical_vendor_fail_is_not_success_hit(past_date):
    cls = MagicMock()
    cls.get_global_news.return_value = VendorFail("coverage incomplete")
    investoday = MagicMock()
    investoday.get_global_news.return_value = "## verified historical news"
    with patch.object(iface, "_registry") as reg:
        reg.list_names.return_value = ["cn_cls", "cn_investoday"]
        reg.get.side_effect = {"cn_cls": cls, "cn_investoday": investoday}.get
        with patch.object(iface, "get_vendor", return_value="cn_cls,cn_investoday"):
            out = iface.route_to_vendor(
                "get_global_news", past_date, look_back_days=7, limit=5
            )
    assert out == "## verified historical news"
    cls.get_global_news.assert_called_once()
    investoday.get_global_news.assert_called_once()


def test_cn_cls_latest_routes_as_live_near_window_hit_without_peer_call():
    today = cn_today_str()
    provider = CnClsProvider(snapshot_dir="/tmp/cls-route-test")
    latest_text = "## latest"
    with patch.object(provider, "_fetch_latest", return_value=([], {"data": {"roll_data": []}}, "https://www.cls.cn/api/cache?name=telegraph", None)), patch.object(provider, "_write_page_snapshot", return_value="/tmp/latest.json"), patch.object(iface, "_registry") as reg:
        peer = MagicMock()
        reg.list_names.return_value = ["cn_cls", "cn_investoday"]
        reg.get.side_effect = {"cn_cls": provider, "cn_investoday": peer}.get
        with patch.object(iface, "get_vendor", return_value="cn_cls,cn_investoday"):
            with patch.object(iface, "_trace") as trace:
                out = iface.route_to_vendor("get_global_news", today, look_back_days=14, limit=1)
    assert "live/near-window 最新缓存，仅供近实时使用" in out
    assert "coverage_complete=false" in out
    assert "look_back_days=14 未被视为完整历史回溯" in out
    peer.get_global_news.assert_not_called()
    assert any("status=hit" in call.args[0] and "cn_cls" in call.args[0] for call in trace.call_args_list)


def test_global_news_today_still_routes_to_provider():
    today = cn_today_str()
    provider = MagicMock()
    provider.get_global_news.return_value = "## live global news"

    with patch.object(iface, "_registry") as reg:
        reg.list_names.return_value = ["cn_akshare"]
        reg.get.return_value = provider
        with patch.object(iface, "get_vendor", return_value="cn_akshare"):
            out = iface.route_to_vendor("get_global_news", today, 7, 5)

    assert out == "## live global news"
    provider.get_global_news.assert_called_once()


def test_news_today_still_routes_to_provider():
    today = cn_today_str()
    start = (now_cn().date() - timedelta(days=7)).isoformat()
    provider = MagicMock()
    provider.get_news.return_value = "## live stock news"

    with patch.object(iface, "_registry") as reg:
        reg.list_names.return_value = ["cn_akshare"]
        reg.get.return_value = provider
        with patch.object(iface, "get_vendor", return_value="cn_akshare"):
            out = iface.route_to_vendor("get_news", "600519", start, today)

    assert out == "## live stock news"
    provider.get_news.assert_called_once_with("600519", start, today)


def test_news_kwarg_end_date_historical_failure_is_explicit(past_date):
    provider = MagicMock()
    with patch.object(iface, "_registry") as reg:
        reg.list_names.return_value = ["cn_akshare"]
        reg.get.return_value = provider
        with patch.object(iface, "get_vendor", return_value="cn_akshare"):
            out = iface.route_to_vendor(
                "get_news",
                "600519",
                start_date="2026-01-01",
                end_date=past_date,
            )
    assert out.startswith("【数据获取失败】")
    assert "历史个股新闻" in out
    provider.get_news.assert_called_once()
