"""Tests for 同花顺 (THS) fuyao.aicubes.cn provider + route wiring."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.dataflows import interface as iface
from tradingagents.dataflows.providers.base import ProviderResourcePolicy
from tradingagents.dataflows.providers.cn_fuyao_provider import (
    CnFuyaoProvider,
    FuyaoApiError,
)
from tradingagents.dataflows.trade_calendar import CN_TZ
from tradingagents.dataflows.vendor_result import VendorEmpty, VendorFail

FAST_POLICY = ProviderResourcePolicy(timeout_seconds=1.0, max_retries=0, max_concurrency=2)


def _mock_json_response(body: dict):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = body
    return mock_resp


def _ok_payload(item=None, extra_data=None):
    data = dict(extra_data or {})
    if item is not None:
        data["item"] = item
    return {"code": 0, "message": "success", "request_id": "r1", "data": data}


# ── thscode normalization ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("600519", "600519.SH"),
        ("600519.SH", "600519.SH"),
        ("600519.SS", "600519.SH"),
        ("SH600519", "600519.SH"),
        ("sh600519", "600519.SH"),
        ("000001.SZ", "000001.SZ"),
        ("300033", "300033.SZ"),
        ("430047", "430047.BJ"),
        ("688981", "688981.SH"),
        ("", None),
        ("INVALID", None),
        ("AAPL", None),
    ],
)
def test_normalize_thscode(raw, expected):
    assert CnFuyaoProvider._normalize_thscode(raw) == expected


# ── 行情快照 ──────────────────────────────────────────────────────────


def test_get_realtime_quotes_maps_fields_and_batches():
    body = _ok_payload(
        item=[
            {
                "thscode": "600519.SH",
                "ticker": "600519",
                "volume": 3098875,
                "turnover": 3937375200,
                "last_price": 1277.8,
                "price_change": 21.8,
                "price_change_ratio_pct": 1.735669,
                "open_price": 1252.08,
                "high_price": 1282,
                "low_price": 1250.21,
                "prev_price": 1256,
            }
        ],
        extra_data={"timestamp": 1784275991000},
    )
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="test-key"), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             return_value=_mock_json_response(body),
         ) as mock_get:
        out = provider.get_realtime_quotes(["600519.SH", "600519.SS"])

    mock_get.assert_called_once()
    call_kw = mock_get.call_args[1]
    assert call_kw["headers"] == {"X-api-key": "test-key"}
    assert call_kw["params"]["thscodes"] == "600519.SH"

    data = json.loads(out)
    assert "600519.SH" in data
    q = data["600519.SH"]
    assert q["price"] == 1277.8
    assert q["previous_close"] == 1256
    assert q["change"] == pytest.approx(21.8)
    assert q["change_pct"] == pytest.approx(1.735669)
    assert q["open"] == 1252.08
    assert q["high"] == 1282
    assert q["low"] == 1250.21
    assert q["volume"] == 3098875.0
    assert q["amount"] == 3937375200.0
    assert q["quote_time"] == "2026-07-17"
    assert q["source"] == "fuyao"


def test_get_realtime_quotes_no_api_key():
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value=""):
        with pytest.raises(NotImplementedError, match="API Key"):
            provider.get_realtime_quotes(["600519.SH"])


def test_get_realtime_quotes_empty_symbols():
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"):
        assert json.loads(provider.get_realtime_quotes([])) == {}
        assert json.loads(provider.get_realtime_quotes(["", "INVALID"])) == {}


def test_get_realtime_quotes_key_error_maps_to_not_implemented():
    body = {"code": 2001, "message": "invalid key", "data": None}
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="bad"), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             return_value=_mock_json_response(body),
         ):
        with pytest.raises(NotImplementedError, match="API Key"):
            provider.get_realtime_quotes(["600519.SH"])


# ── 历史 K 线 ─────────────────────────────────────────────────────────


def test_get_stock_data_csv_from_historical():
    body = _ok_payload(
        item=[
            {
                "date_ms": CnFuyaoProvider._date_to_ms("2025-01-02"),
                "open_price": 10.0,
                "high_price": 11.0,
                "low_price": 9.5,
                "close_price": 10.5,
                "volume": 10000.0,
                "turnover": 105000.0,
            },
            {
                "date_ms": CnFuyaoProvider._date_to_ms("2025-01-03"),
                "open_price": 10.5,
                "high_price": 12.0,
                "low_price": 10.4,
                "close_price": 11.0,
                "volume": 12000.0,
                "turnover": 130000.0,
            },
        ]
    )
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             return_value=_mock_json_response(body),
         ) as mock_get:
        out = provider.get_stock_data("600519.SH", "2025-01-02", "2025-01-03")

    assert "# Stock data for 600519.SH" in out
    assert "2025-01-02" in out
    assert "Open,High,Low,Close,Volume" in out
    call_kw = mock_get.call_args[1]
    assert call_kw["params"]["thscode"] == "600519.SH"
    assert call_kw["params"]["interval"] == "1d"
    assert call_kw["params"]["adjust"] == "forward"


def test_get_stock_data_10y_window_guard():
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"):
        with pytest.raises(ValueError, match="10 年"):
            provider.get_stock_data("600519.SH", "2010-01-01", "2026-01-01")


def test_get_stock_data_3001_maps_to_vendor_empty():
    body = {"code": 3001, "message": "标的不存在", "data": None}
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             return_value=_mock_json_response(body),
         ):
        out = provider.get_stock_data("999999.SH", "2025-01-01", "2025-01-31")
    assert isinstance(out, VendorEmpty)
    assert "标的不存在" in out.message


# ── 三大报表 / 财务指标 ───────────────────────────────────────────────


def test_get_income_statement_markdown():
    body = _ok_payload(
        item=[
            {
                "thscode": "600519.SH",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "operating_income": 174144000000,
                "net_profit": 93000000000,
                "basic_eps": 68.50,
            }
        ]
    )
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             return_value=_mock_json_response(body),
         ) as mock_get:
        out = provider.get_income_statement("600519.SH", "annual", "2026-08-05")

    assert "利润表" in out
    assert "同花顺 fuyao" in out
    assert "operating_income" in out or "174144000000" in out
    call_kw = mock_get.call_args[1]
    assert call_kw["params"]["thscode"] == "600519.SH"
    assert call_kw["params"]["period"] == "annual"
    assert call_kw["params"]["start"] < call_kw["params"]["end"]


def test_financial_missing_curr_date_refuses():
    provider = CnFuyaoProvider()
    out = provider.get_income_statement("600519.SH", "annual", None)
    assert "缺少 curr_date" in out


def test_get_fundamentals_indicators():
    body = {
        "code": 0,
        "message": "success",
        "request_id": "r1",
        "data": {
            "thscode": "600519.SH",
            "report": "2026-1",
            "abilities": [
                {
                    "ability": "growth",
                    "indicators": [
                        {"index_id": "total_assets_growth_ratio", "value": "-16.0031"}
                    ],
                },
                {
                    "ability": "profitability",
                    "indicators": [
                        {"index_id": "sale_gross_margin", "value": "89.12"},
                        {"index_id": "earned_interest_multiple", "value": None},
                    ],
                },
            ],
        },
    }
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             return_value=_mock_json_response(body),
         ) as mock_get:
        out = provider.get_fundamentals("600519.SH", "2026-08-05")

    assert "Fundamentals" in out
    assert "成长能力" in out
    assert "盈利能力" in out
    assert "total_assets_growth_ratio=-16.0031" in out
    assert "earned_interest_multiple=缺失" in out
    assert mock_get.call_args[1]["params"]["report"] == "2026-1"


# ── 错误码映射 ────────────────────────────────────────────────────────


def test_1002_param_error_raises_value_error():
    body = {"code": 1002, "message": "参数格式错误", "data": None}
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             return_value=_mock_json_response(body),
         ):
        with pytest.raises(ValueError, match="参数错误"):
            provider.get_stock_data("600519.SH", "2025-01-01", "2025-01-31")


def test_5001_server_error_maps_to_vendor_fail():
    body = {"code": 5001, "message": "服务内部错误", "data": None}
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             return_value=_mock_json_response(body),
         ):
        out = provider.get_stock_data("600519.SH", "2025-01-01", "2025-01-31")
    assert isinstance(out, VendorFail)
    assert "服务端错误" in out.error


def test_4001_rate_limit_retries_then_vendor_fail():
    rate_limited = _mock_json_response({"code": 4001, "message": "频率超限", "data": None})
    server_error = _mock_json_response({"code": 5003, "message": "上游不可用", "data": None})
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"), \
         patch.object(provider, "_RATE_LIMIT_RETRIES", 1), \
         patch.object(provider, "_RATE_LIMIT_BACKOFF_SECONDS", 0), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             side_effect=[rate_limited, server_error],
         ) as mock_get:
        out = provider.get_stock_data("600519.SH", "2025-01-01", "2025-01-31")

    assert mock_get.call_count == 2
    assert isinstance(out, VendorFail)


# ── 涨跌停池（分页） / 龙虎榜 ─────────────────────────────────────────


def test_get_zt_pool_paginates_and_formats():
    page1 = _ok_payload(
        item=[
            {
                "thscode": "603986.SH",
                "name": "兆易创新",
                "continue_day_text": "2连板",
                "continue_day_cnt": 2,
                "price_change_ratio_pct": 10.0,
                "limit_up_time": "09:34",
                "limit_up_reason": "存储芯片",
            }
        ],
        extra_data={
            "pagination": {"total": 3, "pages": 2, "size": 200, "page": 1},
            "timestamp": 1748102400000,
        },
    )
    page2 = _ok_payload(
        item=[
            {
                "thscode": "000001.SZ",
                "name": "平安银行",
                "continue_day_text": "首板",
                "continue_day_cnt": 1,
                "price_change_ratio_pct": 9.9,
                "limit_up_time": "10:01",
                "limit_up_reason": "低估值",
            }
        ],
        extra_data={
            "pagination": {"total": 3, "pages": 2, "size": 200, "page": 2},
            "timestamp": 1748102400000,
        },
    )
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             side_effect=[_mock_json_response(page1), _mock_json_response(page2)],
         ) as mock_get:
        out = provider.get_zt_pool("2026-08-04")

    assert "共 3 只" in out
    assert "兆易创新" in out
    assert "平安银行" in out
    assert "2连板 1只" in out
    assert mock_get.call_count == 2
    # 第二页分页参数正确
    page2_params = mock_get.call_args_list[1][1]["params"]
    assert page2_params["page"] == 2
    assert page2_params["date_ms"] == CnFuyaoProvider._date_to_ms("2026-08-04")


def test_get_zt_pool_missing_date_refuses():
    provider = CnFuyaoProvider()
    out = provider.get_zt_pool(None)
    assert "缺少 date" in out


def test_get_lhb_detail_filters_by_symbol():
    from tradingagents.dataflows import trade_calendar as tc

    tc.clear_cn_trade_date_cache()
    tc._TRADE_DATES_CACHE["dates"] = [pd.Timestamp("2026-07-01").date()]
    tc._TRADE_DATES_CACHE["dates_set"] = {pd.Timestamp("2026-07-01").date()}
    tc._TRADE_DATES_CACHE["loaded_at"] = 1e18

    body = {
        "code": 0,
        "message": "success",
        "request_id": "r1",
        "data": {
            "timestamp": 1782921600000,
            "board_type": "all",
            "trade_date": "2026-07-01",
            "count": 1,
            "stock_count": 1,
            "stock_items": [
                {
                    "thscode": "002407.SZ",
                    "ticker": "002407",
                    "name": "多氟多",
                    "change": 0.09994,
                    "net_value": 1786253128.23,
                    "net_rate": 0.119,
                    "buy_value": 2674755016.05,
                    "sell_value": 888501887.82,
                    "limit_reason": "六氟磷酸锂涨价",
                    "range_days": 3,
                }
            ],
            "hot_money_items": [],
        },
    }
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             return_value=_mock_json_response(body),
         ) as mock_get:
        out = provider.get_lhb_detail("002407", "2026-07-01")

    assert "龙虎榜明细" in out
    assert "多氟多" in out
    assert mock_get.call_args[1]["params"]["date"] == "2026-07-01"
    assert mock_get.call_args[1]["params"]["board_type"] == "all"
    tc.clear_cn_trade_date_cache()


def test_get_lhb_detail_not_on_board_is_normal_empty():
    from tradingagents.dataflows import trade_calendar as tc

    tc.clear_cn_trade_date_cache()
    tc._TRADE_DATES_CACHE["dates"] = [pd.Timestamp("2026-07-01").date()]
    tc._TRADE_DATES_CACHE["dates_set"] = {pd.Timestamp("2026-07-01").date()}
    tc._TRADE_DATES_CACHE["loaded_at"] = 1e18

    body = {
        "code": 0,
        "message": "success",
        "request_id": "r1",
        "data": {
            "trade_date": "2026-07-01",
            "stock_items": [{"thscode": "002407.SZ", "name": "多氟多"}],
            "hot_money_items": [],
        },
    }
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             return_value=_mock_json_response(body),
         ):
        out = provider.get_lhb_detail("600519", "2026-07-01")
    assert "无龙虎榜数据" in out
    tc.clear_cn_trade_date_cache()


# ── 交易日历 ──────────────────────────────────────────────────────────


def test_get_trading_days_returns_dates():
    body = _ok_payload(
        item=[
            {"date_ms": 1716566400000, "date": "20250525"},
            {"date_ms": 1716652800000, "date": "20250526"},
        ]
    )
    provider = CnFuyaoProvider()
    with patch.object(provider, "_resolve_api_key", return_value="k"), \
         patch(
             "tradingagents.dataflows.providers.cn_fuyao_provider.requests.get",
             return_value=_mock_json_response(body),
         ):
        out = provider.get_trading_days()
    assert "20250525" in out
    assert "20250526" in out
    assert "共 2 个交易日" in out


# ── registry / config ─────────────────────────────────────────────────


def test_build_default_registry_includes_cn_fuyao():
    from tradingagents.dataflows.providers.registry import build_default_registry

    reg = build_default_registry()
    assert "cn_fuyao" in reg.list_names()
    assert reg.get("cn_fuyao") is not None


def test_default_config_routes_fundamentals_to_fuyao_primary():
    from tradingagents.dataflows.interface import get_vendor

    chain = get_vendor("fundamental_data")
    assert chain.split(",")[0] == "cn_fuyao"
    assert "cn_akshare" in chain


def test_default_config_zt_lhb_route_akshare_then_fuyao():
    from tradingagents.dataflows.interface import get_vendor

    assert get_vendor("cn_market_data", "get_zt_pool") == "cn_akshare,cn_fuyao"
    assert get_vendor("cn_market_data", "get_lhb_detail") == "cn_akshare,cn_fuyao"


# ── 不支持方法显式回退 ────────────────────────────────────────────────


def test_unsupported_methods_raise_not_implemented():
    provider = CnFuyaoProvider()
    with pytest.raises(NotImplementedError):
        provider.get_indicators("600519.SH", "rsi", "2026-08-05", 14)
    with pytest.raises(NotImplementedError):
        provider.get_news("600519.SH", "2026-01-01", "2026-01-31")
    with pytest.raises(NotImplementedError):
        provider.get_insider_transactions("600519.SH")


# ── route 接线 ────────────────────────────────────────────────────────


class _FakeProvider:
    def __init__(self, name, func, method):
        self.name = name
        self._func = func
        self._method = method

    def __getattr__(self, attr):
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


def _route(chain: dict[str, object], configured: str, method: str, *args):
    registry = _FakeRegistry(chain)
    with patch.object(iface, "_registry", registry), \
         patch.object(iface, "get_vendor", return_value=configured):
        return iface.route_to_vendor(method, *args)


def test_route_fundamentals_uses_fuyao_primary_before_akshare():
    fuyao = _FakeProvider(
        "cn_fuyao",
        lambda *a, **k: "## 利润表（同花顺 fuyao）",
        method="get_income_statement",
    )
    akshare = _FakeProvider(
        "cn_akshare",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("akshare must not be called")),
        method="get_income_statement",
    )
    out = _route(
        {"cn_fuyao": fuyao, "cn_akshare": akshare},
        "cn_fuyao,cn_akshare",
        "get_income_statement",
        "600519.SH",
        "annual",
        "2026-08-05",
    )
    assert out == "## 利润表（同花顺 fuyao）"


def test_route_zt_pool_falls_back_to_fuyao_when_akshare_vendor_fail():
    akshare = _FakeProvider(
        "cn_akshare",
        lambda *a, **k: VendorFail("东财接口失败"),
        method="get_zt_pool",
    )
    fuyao = _FakeProvider(
        "cn_fuyao",
        lambda *a, **k: "涨停池（2026-08-04，同花顺 fuyao）：共 3 只",
        method="get_zt_pool",
    )
    out = _route(
        {"cn_akshare": akshare, "cn_fuyao": fuyao},
        "cn_akshare,cn_fuyao",
        "get_zt_pool",
        "2026-08-04",
    )
    assert "同花顺 fuyao" in out


def test_route_lhb_falls_back_to_fuyao_when_akshare_vendor_fail():
    akshare = _FakeProvider(
        "cn_akshare",
        lambda *a, **k: VendorFail("东财龙虎榜失败"),
        method="get_lhb_detail",
    )
    fuyao = _FakeProvider(
        "cn_fuyao",
        lambda *a, **k: "600519 龙虎榜明细（2026-08-04，同花顺 fuyao）",
        method="get_lhb_detail",
    )
    out = _route(
        {"cn_akshare": akshare, "cn_fuyao": fuyao},
        "cn_akshare,cn_fuyao",
        "get_lhb_detail",
        "600519.SH",
        "2026-08-04",
    )
    assert "同花顺 fuyao" in out


# ── akshare 东财失败 → VendorFail（保证链路可切到 fuyao）────────────


def test_akshare_get_zt_pool_failure_is_vendor_fail():
    from tradingagents.dataflows import trade_calendar as tc
    from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider

    tc.clear_cn_trade_date_cache()
    today = pd.Timestamp(tc.cn_today_str()).date()
    tc._TRADE_DATES_CACHE["dates"] = [today]
    tc._TRADE_DATES_CACHE["dates_set"] = {today}
    tc._TRADE_DATES_CACHE["loaded_at"] = 1e18

    ak = MagicMock()
    ak.stock_zt_pool_em.side_effect = ConnectionError("RemoteDisconnected")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    out = p.get_zt_pool(tc.cn_today_str())
    assert isinstance(out, VendorFail)
    assert "涨停板情绪池数据获取失败" in out.error
    tc.clear_cn_trade_date_cache()


def test_akshare_get_lhb_detail_double_failure_is_vendor_fail():
    from tradingagents.dataflows import trade_calendar as tc
    from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider

    tc.clear_cn_trade_date_cache()
    tc._TRADE_DATES_CACHE["dates"] = [pd.Timestamp("2026-08-03").date()]
    tc._TRADE_DATES_CACHE["dates_set"] = {pd.Timestamp("2026-08-03").date()}
    tc._TRADE_DATES_CACHE["loaded_at"] = 1e18

    ak = MagicMock()
    ak.stock_lhb_detail_em.side_effect = ConnectionError("RemoteDisconnected")
    ak.stock_lhb_detail_daily_sina.side_effect = ConnectionError("sina down")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    out = p.get_lhb_detail("600519", "2026-08-03")
    assert isinstance(out, VendorFail)
    assert "龙虎榜数据获取失败" in out.error
    tc.clear_cn_trade_date_cache()
