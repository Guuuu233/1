"""3b-hotfix: near-window news refused at route_to_vendor for historical dates."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.dataflows import interface as iface
from tradingagents.dataflows.trade_calendar import cn_today_str, now_cn


GLOBAL_REASON = "全球快讯为实时直播流，不提供可复现的历史切片，历史日期分析下本项不可用"
STOCK_REASON = "个股新闻源仅覆盖近期，历史日期不可用"


@pytest.fixture
def past_date() -> str:
    return (now_cn().date() - timedelta(days=90)).isoformat()


def test_global_news_historical_refuses_with_zero_provider_hits(past_date, capsys):
    provider = MagicMock()
    provider.get_global_news.side_effect = AssertionError("provider must not be called")

    with patch.object(iface, "_registry") as reg:
        reg.list_names.return_value = ["cn_akshare", "yfinance"]
        reg.get.return_value = provider
        with patch.object(iface, "get_vendor", return_value="cn_akshare,yfinance"):
            out = iface.route_to_vendor(
                "get_global_news", past_date, look_back_days=14, limit=10
            )

    assert out.startswith("【数据获取失败】")
    assert GLOBAL_REASON in out
    provider.get_global_news.assert_not_called()
    reg.get.assert_not_called()
    captured = capsys.readouterr().out
    assert "status=historical-refuse" in captured
    assert "providers_hit=0" in captured


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
