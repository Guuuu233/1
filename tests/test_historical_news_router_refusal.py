"""3b-hotfix: near-window news refused at route_to_vendor for historical dates."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.dataflows import interface as iface
from tradingagents.dataflows.trade_calendar import cn_today_str, now_cn


STOCK_REASON = "个股新闻源仅覆盖近期，历史日期不可用"


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


def test_news_historical_refuses_with_zero_provider_hits(past_date, capsys):
    provider = MagicMock()
    provider.get_news.side_effect = AssertionError("provider must not be called")

    with patch.object(iface, "_registry") as reg:
        reg.list_names.return_value = ["cn_akshare", "cn_investoday", "yfinance"]
        reg.get.return_value = provider
        with patch.object(iface, "get_vendor", return_value="cn_akshare,cn_investoday,yfinance"):
            out = iface.route_to_vendor(
                "get_news",
                "600519",
                (now_cn().date() - timedelta(days=104)).isoformat(),
                past_date,
            )

    assert out.startswith("【数据获取失败】")
    assert STOCK_REASON in out
    assert "No news found" not in out
    provider.get_news.assert_not_called()
    reg.get.assert_not_called()
    captured = capsys.readouterr().out
    assert "status=historical-refuse" in captured
    assert "providers_hit=0" in captured


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


def test_news_kwarg_end_date_historical_refuses(past_date):
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
    assert STOCK_REASON in out
    provider.get_news.assert_not_called()
