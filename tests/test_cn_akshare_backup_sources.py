"""DAV-69 — Eastmoney backup sources for intermittently-unreachable interfaces.

The Eastmoney push2his endpoints (fund flow, LHB) intermittently drop the
connection (RemoteDisconnected) on the current IP. Each affected method now
falls back to an alternative source inside the provider:

- get_board_fund_flow        EM stock_fund_flow_industry  -> THS stock_board_industry_summary_ths
- get_individual_fund_flow   EM stock_individual_fund_flow -> Tushare DC/THS -> Sina historical close API
  (DAV-88 Bug E) for dated rows; Tonghuashun stock_fund_flow_individual for the
  current-day generic funds net-flow snapshot when the close row is unavailable
  (not a same-semantic Sina netamount/r0_net main-force series)
- get_lhb_detail             EM stock_lhb_detail_em        -> Sina stock_lhb_detail_daily_sina
"""

from __future__ import annotations

import json
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import tradingagents.dataflows.providers.cn_akshare_provider as cn_akshare_provider
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.trade_calendar import cn_today_str


@pytest.fixture
def trading_day(monkeypatch):
    """Inject a deterministic trading-day calendar only for tests that need it."""
    monkeypatch.setattr(cn_akshare_provider, "is_cn_trading_day", lambda _date: True)


# ── Task 2: Eastmoney backup sources ──────────────────────────────────


class _EastmoneyResponse:
    def __init__(self, payload=None, *, status_code=200, text=None):
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# The live /fflow/kline/get response currently returns six fields, not the
# full requested f51-f63 projection: f51 date, f52 r0_net, f53-f56 raw only.
_DIRECT_KLINE = "2026-08-14,120000000,-10000000,-20000000,50000000,70000000"
_DIRECT_KLINE_WITH_UNKNOWN_TAIL = f"{_DIRECT_KLINE},0.12,-0.01,12.34,unknown-tail"


def _direct_payload(*klines, rc=0):
    return {"rc": rc, "data": {"klines": list(klines)}}


def _current_day_ths():
    return pd.DataFrame(
        {
            "股票代码": ["600519"],
            "股票简称": ["贵州茅台"],
            "最新价": [1358.98],
            "涨跌幅": ["0.62%"],
            "流入资金": ["26.30亿"],
            "流出资金": ["22.69亿"],
            "净额": ["3.61亿"],
            "换手率": ["0.29%"],
        }
    )


class _TushareResponse:
    def __init__(self, payload, *, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


class _BrokenTushareResponse:
    status_code = 200
    text = "not-json"

    def json(self):
        raise ValueError("invalid JSON fixture")


def _tushare_payload(
    api_name: str,
    *,
    trade_date: str = "20260814",
    net_amount: str = "12000",
    net_d5_amount: str = "56000",
    code: int = 0,
    include_net_amount: bool = True,
    ts_code: str = "600519.SH",
    include_ts_code: bool = True,
):
    fields = ["trade_date"]
    if include_ts_code:
        fields.insert(0, "ts_code")
    if include_net_amount:
        fields.append("net_amount")
    if api_name == "moneyflow_ths":
        fields.append("net_d5_amount")
    if api_name == "moneyflow_dc":
        fields.extend(
            [
                "buy_sm_amount",
                "buy_md_amount",
                "buy_lg_amount",
                "buy_elg_amount",
            ]
        )
    else:
        fields.extend(["buy_sm_amount", "buy_md_amount", "buy_lg_amount"])
    values = {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "net_amount": net_amount,
        "net_d5_amount": net_d5_amount,
        "buy_sm_amount": "100",
        "sell_sm_amount": "80",
        "buy_md_amount": "200",
        "sell_md_amount": "150",
        "buy_lg_amount": "300",
        "sell_lg_amount": "250",
        "buy_elg_amount": "400",
        "sell_elg_amount": "350",
    }
    return {
        "code": code,
        "data": {"fields": fields, "items": [[values[field] for field in fields]]},
    }


def test_tushare_dc_ths_success_keeps_transport_and_field_semantics(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "configured")
    provider = CnAkshareProvider()
    with patch(
        "requests.post",
        side_effect=[
            _TushareResponse(_tushare_payload("moneyflow_dc")),
            _TushareResponse(_tushare_payload("moneyflow_ths")),
        ],
    ) as mock_post:
        out, errors, meta = provider._fetch_tushare_fund_flow(
            "600519", "2026-08-14"
        )

    assert errors == []
    assert mock_post.call_count == 2
    assert mock_post.call_args_list[0].kwargs["json"]["api_name"] == "moneyflow_dc"
    first_request = mock_post.call_args_list[0].kwargs["json"]
    assert first_request["params"] == {
        "ts_code": "600519.SH",
        "trade_date": "20260814",
    }
    assert first_request["fields"].split(",") == [
        "ts_code",
        "trade_date",
        "net_amount",
        "buy_sm_amount",
        "buy_md_amount",
        "buy_lg_amount",
        "buy_elg_amount",
    ]
    second_request = mock_post.call_args_list[1].kwargs["json"]
    assert second_request["fields"].split(",") == [
        "ts_code",
        "trade_date",
        "net_amount",
        "buy_sm_amount",
        "buy_md_amount",
        "buy_lg_amount",
    ]
    assert out is not None
    assert len(out.fund_flow_evidence) == 2
    dc_record, ths_record = out.fund_flow_evidence
    assert dc_record["transport_provider"] == "tushare"
    assert dc_record["date"] == "2026-08-14"
    assert dc_record["source_family"] == "eastmoney"
    assert dc_record["upstream_field"] == "net_amount"
    assert dc_record["upstream_field_semantics"] == "今日主力净流入额（万元）"
    assert dc_record["r0_net"] == "1.2"
    assert dc_record["vendor_raw_fields"]["buy_sm_amount"] == "100"
    assert dc_record["vendor_normalized_fields"]["buy_sm_amount"] == "0.01"
    assert ths_record["source_family"] == "ths"
    assert ths_record["netamount"] == "1.2"
    assert ths_record["net_d5_amount"] == "5.6"
    assert ths_record["net_d5_amount_raw"] == "56000"
    assert ths_record["net_d5_amount_period_kind"] == "five_day_cumulative"
    assert meta["transport_provider"] == "tushare"
    assert meta["consensus"]["selected_source"] == "tushare_eastmoney_moneyflow_dc"
    assert meta["consensus"]["selected_field"] == "r0_net"
    assert meta["consensus"]["direction_allowed"] is True
    assert meta["consensus"]["alternative_sources"][0]["source"] == "tushare_ths_moneyflow_ths"
    assert meta["consensus_audit"]["reason_code"] == "incomparable_field_semantics"


def test_tushare_token_missing_is_typed_and_does_not_call_network(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    provider = CnAkshareProvider()
    with patch("requests.post") as mock_post:
        out, errors, meta = provider._fetch_tushare_fund_flow(
            "600519", "2026-08-14"
        )

    assert out is None
    assert mock_post.call_count == 0
    assert len(errors) == 2
    assert meta["transport_provider"] == "tushare"
    assert meta["attempted_sources"] == []
    assert meta["gated_sources"] == ["moneyflow_dc", "moneyflow_ths"]
    assert meta["failure_categories"] == ["token_missing"]
    assert all("token_missing" in error for error in errors)


@pytest.mark.parametrize(
    "payload, expected_category",
    [
        (_tushare_payload("moneyflow_dc", code=2002), "permission_denied"),
        (_tushare_payload("moneyflow_dc", code=40203), "rate_limited"),
        (_tushare_payload("moneyflow_dc", code=12345), "api_code"),
        (_tushare_payload("moneyflow_dc", include_net_amount=False), "missing_field"),
        (_tushare_payload("moneyflow_dc", trade_date="20260813"), "date_mismatch"),
        (_tushare_payload("moneyflow_dc", ts_code="000001.SZ"), "symbol_mismatch"),
        (_tushare_payload("moneyflow_dc", ts_code="bad"), "invalid_identity"),
        (_tushare_payload("moneyflow_dc", include_ts_code=False), "missing_field"),
    ],
)
def test_tushare_typed_api_and_validation_gaps(
    monkeypatch, payload, expected_category
):
    monkeypatch.setenv("TUSHARE_TOKEN", "configured")
    provider = CnAkshareProvider()
    with patch("requests.post", return_value=_TushareResponse(payload)):
        out, errors, meta = provider._fetch_tushare_fund_flow(
            "600519", "2026-08-14"
        )

    assert out is None
    assert len(errors) == 2
    assert meta["status"] == "unavailable"
    assert expected_category in [
        failure["category"] for failure in meta["tushare_failures"]
    ]


@pytest.mark.parametrize(
    "response, expected_category",
    [
        (_TushareResponse({}, status_code=503), "http_error"),
        (_BrokenTushareResponse(), "json_error"),
    ],
)
def test_tushare_http_and_json_failures_are_typed(
    monkeypatch, response, expected_category
):
    monkeypatch.setenv("TUSHARE_TOKEN", "configured")
    provider = CnAkshareProvider()
    with patch("requests.post", return_value=response):
        out, errors, meta = provider._fetch_tushare_fund_flow(
            "600519", "2026-08-14"
        )

    assert out is None
    assert len(errors) == 2
    assert all(
        failure["category"] == expected_category
        for failure in meta["tushare_failures"]
    )


def test_individual_fund_flow_uses_tushare_before_legacy_when_configured(
    monkeypatch, trading_day
):
    monkeypatch.setenv("TUSHARE_TOKEN", "configured")
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    provider = CnAkshareProvider()
    provider._ak = lambda: ak
    with patch("requests.get", side_effect=ConnectionError("EM direct down")), patch(
        "requests.post",
        side_effect=[
            _TushareResponse(_tushare_payload("moneyflow_dc")),
            _TushareResponse(_tushare_payload("moneyflow_ths")),
        ],
    ) as mock_post:
        out = provider.get_individual_fund_flow(
            "600519", curr_date="20260814"
        )

    assert (
        mock_post.call_args_list[0].kwargs["json"]["params"]["trade_date"]
        == "20260814"
    )
    assert "\\n" not in str(out)
    assert "\n" in str(out)
    meta = out.fund_flow_evidence_meta
    assert meta["final_source"] == "tushare"
    assert meta["requested_as_of"] == "20260814"
    assert meta["actual_as_of"] == "2026-08-14"
    assert meta["transport_provider"] == "tushare"
    assert "sina_historical" not in meta["attempted_sources"]
    assert "Tushare Pro" in out


def test_tushare_failure_then_legacy_reference_is_explicit(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "configured")
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    provider = CnAkshareProvider()
    provider._ak = lambda: ak
    legacy_rows = [
        {
            "opendate": "2026-08-14",
            "netamount": "100000000",
            "r0_net": "50000000",
        }
    ]
    with patch(
        "requests.get",
        side_effect=[
            ConnectionError("eastmoney direct down"),
            _EastmoneyResponse(legacy_rows),
        ],
    ), patch(
        "requests.post",
        side_effect=[
            ConnectionError("Tushare unavailable"),
            ConnectionError("Tushare unavailable"),
            ConnectionError("Tushare unavailable"),
            ConnectionError("Tushare unavailable"),
        ],
    ):
        out = provider.get_individual_fund_flow(
            "600519", curr_date="2026-08-14"
        )

    assert out.fund_flow_evidence_meta["final_source"] == "sina_historical"
    assert out.fund_flow_evidence_meta["legacy_web_reference_only"] is True
    tushare_meta = out.fund_flow_evidence_meta["tushare_provider"]
    assert tushare_meta["transport_provider"] == "tushare"
    assert all(
        failure["category"] == "transport_error"
        for failure in tushare_meta["tushare_failures"]
    )
    assert "新浪旧 Web 参考值" in out
    assert out.fund_flow_evidence_meta["selected_source"] == "sina_historical"
    assert out.fund_flow_evidence_meta["selection_reason"] == "no_new_algorithm_source_legacy_fallback"
    assert out.fund_flow_evidence_meta["legacy_reference"] is True
    assert out.fund_flow_evidence_meta["direction_allowed"] is True


def test_individual_fund_flow_direct_eastmoney_success_is_structured_and_typed(
    trading_day,
):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("TLS fingerprint")
    ak.stock_fund_flow_individual.return_value = _current_day_ths()
    p = CnAkshareProvider()
    p._ak = lambda: ak

    with patch(
        "requests.get",
        return_value=_EastmoneyResponse(_direct_payload(_DIRECT_KLINE)),
    ) as mock_get:
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    assert "东方财富直连" in out
    assert mock_get.call_count == 1
    request_kwargs = mock_get.call_args.kwargs
    assert (
        mock_get.call_args.args[0]
        == "https://push2his.eastmoney.com/api/qt/stock/fflow/kline/get"
    )
    assert request_kwargs["params"]["secid"] == "1.600519"
    assert request_kwargs["params"]["klt"] == "101"
    assert request_kwargs["params"]["lmt"] == "120"
    assert request_kwargs["params"]["fields1"] == "f1,f2,f3,f7"
    assert request_kwargs["headers"] == {
        "Referer": "https://data.eastmoney.com/",
        "User-Agent": "Mozilla/5.0",
    }
    assert request_kwargs["params"]["fields2"].startswith(
        "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63"
    )
    assert "f64" not in request_kwargs["params"]["fields2"]
    assert "f65" not in request_kwargs["params"]["fields2"]
    assert request_kwargs["timeout"] == 10
    evidence = out.fund_flow_evidence
    assert len(evidence) == 1
    record = evidence[0]
    assert record["source"] == "eastmoney_direct"
    assert record["source_family"] == "eastmoney"
    assert record["algorithm_group"] == "new_algorithm_group"
    assert record["date"] == "2026-08-14"
    assert record["as_of"] == "2026-08-14"
    assert record["requested_as_of"] == "2026-08-14"
    assert record["raw_unit"] == "元"
    assert record["unit"] == "亿元"
    assert record["r0_net"] == "1.2"
    assert record["r0_net_raw"] == "120000000"
    assert "large_net" not in record
    assert "super_large_net" not in record
    assert "components" not in record
    assert "netamount" not in record
    assert record["field_semantics"]["r0_net"].startswith("主力净额")
    assert record["vendor_raw_field_status"] == "discovery_only"
    assert record["vendor_raw_fields_missing"] == []
    assert record["vendor_raw_fields"] == {
        "f53": "-10000000",
        "f54": "-20000000",
        "f55": "50000000",
        "f56": "70000000",
    }
    assert record["vendor_raw_field_units"] == {
        "f53": None,
        "f54": None,
        "f55": None,
        "f56": None,
    }
    assert "f53_raw" not in out
    assert "f54_raw" not in out

    meta = out.fund_flow_evidence_meta
    assert meta["source"] == "eastmoney_direct"
    assert meta["final_source"] == "eastmoney_direct"
    assert meta["status"] == "selected"
    assert meta["selected_source"] == "eastmoney_direct"
    assert meta["selected_field"] == "r0_net"
    assert meta["direction"] == "inflow"
    assert meta["direction_allowed"] is True
    assert meta["hard_guard"]["blocked"] is False
    assert meta["attempted_sources"] == [
        "akshare.stock_individual_fund_flow",
        "eastmoney_direct",
    ]
    assert "stock_individual_fund_flow: ConnectionError" in meta["em_typed_gap"]
    assert meta["fallback_errors"] == [
        "stock_individual_fund_flow: ConnectionError"
    ]
    assert not ak.stock_fund_flow_individual.called
    assert meta["field_mapping"]["f52"] == "r0_net"
    assert meta["field_mapping"]["f55"] == "raw_discovery_only"
    assert meta["discovery_only_fields"] == ["f53", "f54", "f55", "f56"]
    assert "f57" not in meta["field_mapping"]
    assert "f57" not in record["vendor_raw_fields"]
    assert meta["discovery_field_unit_policy"] == "raw preserved; no normalization"


def test_direct_accepts_minimum_two_field_contract_without_fabricated_discovery(
    trading_day,
):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    minimal_kline = "2026-08-14,0"

    with patch(
        "requests.get",
        return_value=_EastmoneyResponse(_direct_payload(minimal_kline)),
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    assert out.fund_flow_evidence_meta["final_source"] == "eastmoney_direct"
    record = out.fund_flow_evidence[0]
    assert record["r0_net"] == "0"
    assert record["r0_net_raw"] == "0"
    assert record["vendor_raw_fields"] == {}
    assert record["vendor_raw_field_status"] == "not_returned"
    assert record["vendor_raw_fields_missing"] == ["f53", "f54", "f55", "f56"]
    assert record["vendor_raw_field_units"] == {}
    assert out.fund_flow_evidence_meta["discovery_field_status_policy"].startswith(
        "per-record"
    )
    assert "f57" not in out.fund_flow_evidence_meta["field_mapping"]


def test_direct_drops_unverified_trailing_fields_from_evidence_and_prompt(
    trading_day,
):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    p = CnAkshareProvider()
    p._ak = lambda: ak

    with patch(
        "requests.get",
        return_value=_EastmoneyResponse(
            _direct_payload(_DIRECT_KLINE_WITH_UNKNOWN_TAIL)
        ),
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    record = out.fund_flow_evidence[0]
    assert record["r0_net"] == "1.2"
    assert record["vendor_raw_fields"] == {
        "f53": "-10000000",
        "f54": "-20000000",
        "f55": "50000000",
        "f56": "70000000",
    }
    assert "f57" not in record
    assert "unknown-tail" not in out


@pytest.mark.parametrize(
    "symbol, expected_secid",
    [("601398.SH", "1.601398"), ("002167.SZ", "0.002167")],
)
def test_direct_fixed_historical_symbols_preserve_evidence_contract(
    symbol, expected_secid, trading_day
):
    """The fixed DAV-167 symbols use the audited market-specific secid path."""
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    p = CnAkshareProvider()
    p._ak = lambda: ak

    with patch(
        "requests.get",
        return_value=_EastmoneyResponse(_direct_payload(_DIRECT_KLINE)),
    ) as mock_get:
        out = p.get_individual_fund_flow(symbol, curr_date="2026-08-14")

    assert mock_get.call_args.kwargs["params"]["secid"] == expected_secid
    assert out.fund_flow_evidence_meta["final_source"] == "eastmoney_direct"
    assert out.fund_flow_evidence_meta["requested_as_of"] == "2026-08-14"
    assert out.fund_flow_evidence
    record = out.fund_flow_evidence[0]
    assert record["symbol"] == symbol
    assert record["source"] == "eastmoney_direct"
    assert record["algorithm_group"] == "new_algorithm_group"
    assert record["date"] == "2026-08-14"
    assert record["as_of"] == "2026-08-14"
    assert record["requested_as_of"] == "2026-08-14"
    assert record["field_semantics"]["r0_net"].startswith("主力净额")
    assert record["r0_net"] == "1.2"
    assert record["r0_net_raw"] == "120000000"
    assert record["raw_unit"] == "元"
    assert record["unit"] == "亿元"
    assert "netamount" not in record
    assert out.fund_flow_evidence_meta["field_mapping"]["f52"] == "r0_net"


def test_individual_fund_flow_direct_filters_future_rows_without_lookahead(
    trading_day,
):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    future = _DIRECT_KLINE.replace("2026-08-14", "2026-08-15")

    with patch(
        "requests.get",
        return_value=_EastmoneyResponse(_direct_payload(future, _DIRECT_KLINE)),
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    assert out.fund_flow_evidence
    assert [row["date"] for row in out.fund_flow_evidence] == ["2026-08-14"]
    assert "2026-08-15" not in out
    assert out.fund_flow_evidence_meta["final_source"] == "eastmoney_direct"


def test_direct_historical_requires_requested_date_before_sina_fallback(
    trading_day,
):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    older = _DIRECT_KLINE.replace("2026-08-14", "2026-08-12")
    sina_rows = [
        {
            "opendate": "2026-08-12",
            "netamount": "100000000",
            "r0_net": "50000000",
        }
    ]

    with patch(
        "requests.get",
        side_effect=[
            _EastmoneyResponse(_direct_payload(older)),
            _EastmoneyResponse(sina_rows),
        ],
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    meta = out.fund_flow_evidence_meta
    assert meta["final_source"] == "sina_historical"
    assert any(
        "eastmoney_direct: no_requested_date_row" in error
        for error in meta["fallback_errors"]
    )
    assert "validation" in meta["failure_categories"]
    assert [row["date"] for row in out.fund_flow_evidence] == ["2026-08-12"]


def test_direct_malformed_requested_row_keeps_validation_detail(trading_day):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    older = _DIRECT_KLINE.replace("2026-08-14", "2026-08-12")
    invalid_requested = "2026-08-14,NaN"
    sina_rows = [
        {
            "opendate": "2026-08-12",
            "netamount": "100000000",
            "r0_net": "50000000",
        }
    ]

    with patch(
        "requests.get",
        side_effect=[
            _EastmoneyResponse(_direct_payload(older, invalid_requested)),
            _EastmoneyResponse(sina_rows),
        ],
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    assert out.fund_flow_evidence_meta["final_source"] == "sina_historical"
    assert any(
        "eastmoney_direct: malformed_kline_rows_on_or_before_curr_date" in error
        and "invalid_f52" in error
        for error in out.fund_flow_evidence_meta["fallback_errors"]
    )


def test_direct_current_day_requires_exact_as_of_before_ths_fallback(trading_day):
    today = cn_today_str()
    stale = _DIRECT_KLINE.replace(
        "2026-08-14",
        (pd.Timestamp(today) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    ak.stock_fund_flow_individual.return_value = _current_day_ths()
    p = CnAkshareProvider()
    p._ak = lambda: ak

    with patch(
        "requests.get",
        side_effect=[
            _EastmoneyResponse(_direct_payload(stale)),
            ConnectionError("Sina down"),
        ],
    ):
        out = p.get_individual_fund_flow("600519", curr_date=today)

    meta = out.fund_flow_evidence_meta
    assert meta["final_source"] == "ths_instant_snapshot"
    assert any(
        "eastmoney_direct: no_current_day_row" in error
        for error in meta["fallback_errors"]
    )
    assert meta["requested_as_of"] == today
    assert meta["actual_as_of"] == today
    assert all(record["date"] == today for record in out.fund_flow_evidence)
    ak.stock_fund_flow_individual.assert_called_once_with(symbol="即时")


def test_direct_invalid_date_fails_closed_before_sina_fallback(trading_day):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    sina_rows = [
        {
            "opendate": "2026-08-14",
            "netamount": "100000000",
            "r0_net": "50000000",
        }
    ]

    with patch(
        "requests.get",
        side_effect=[
            _EastmoneyResponse(_direct_payload("not-a-date,120000000")),
            _EastmoneyResponse(sina_rows),
        ],
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    assert out.fund_flow_evidence_meta["final_source"] == "sina_historical"
    assert any(
        "eastmoney_direct: no_usable_rows_on_or_before_curr_date" in error
        for error in out.fund_flow_evidence_meta["fallback_errors"]
    )


def test_direct_rejects_timezone_bearing_date_input(trading_day):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    sina_rows = [
        {
            "opendate": "2026-08-14",
            "netamount": "100000000",
            "r0_net": "50000000",
        }
    ]

    with patch(
        "requests.get",
        side_effect=[
            _EastmoneyResponse(
                _direct_payload("2026-08-14T23:30:00-05:00,120000000")
            ),
            _EastmoneyResponse(sina_rows),
        ],
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    assert out.fund_flow_evidence_meta["final_source"] == "sina_historical"
    assert any(
        "invalid_date" in error
        for error in out.fund_flow_evidence_meta["fallback_errors"]
    )


def test_direct_duplicate_normalized_date_is_typed_validation_gap(trading_day):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    duplicate = _DIRECT_KLINE.replace("120000000", "120000001")
    sina_rows = [
        {
            "opendate": "2026-08-14",
            "netamount": "100000000",
            "r0_net": "50000000",
        }
    ]

    with patch(
        "requests.get",
        side_effect=[
            _EastmoneyResponse(_direct_payload(_DIRECT_KLINE, duplicate)),
            _EastmoneyResponse(sina_rows),
        ],
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    meta = out.fund_flow_evidence_meta
    assert meta["final_source"] == "sina_historical"
    assert "eastmoney_direct: duplicate_date: 2026-08-14" in meta["fallback_errors"]
    assert all(record["source"] == "sina_historical" for record in out.fund_flow_evidence)


def test_direct_rejects_non_trading_curr_date_and_continues_fallback(trading_day):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    ak.stock_fund_flow_individual.return_value = _current_day_ths()
    p = CnAkshareProvider()
    p._ak = lambda: ak

    with patch.object(
        cn_akshare_provider, "is_cn_trading_day", return_value=False
    ), patch("requests.get", side_effect=ConnectionError("Sina down")):
        out = p.get_individual_fund_flow("600519", curr_date=cn_today_str())

    assert "【备用数据源：同花顺即时资金流净额快照】" not in out
    assert out.fund_flow_evidence_meta["final_source"] == "unavailable"
    assert any(
        "eastmoney_direct: curr_date_not_cn_trading_day" in error
        for error in out.fund_flow_evidence_meta["fallback_errors"]
    )
    assert any(
        "ths_instant_snapshot: curr_date_not_cn_trading_day" in error
        for error in out.fund_flow_evidence_meta["fallback_errors"]
    )


def test_direct_mixed_valid_and_corrupt_rows_continue_to_sina_fallback(
    trading_day,
):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    corrupt = "2026-08-13"
    sina_rows = [
        {
            "opendate": "2026-08-13",
            "netamount": "100000000",
            "r0_net": "50000000",
        }
    ]

    with patch(
        "requests.get",
        side_effect=[
            _EastmoneyResponse(_direct_payload(_DIRECT_KLINE, corrupt)),
            _EastmoneyResponse(sina_rows),
        ],
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    assert "新浪历史/收盘数据" in out
    assert out.fund_flow_evidence_meta["final_source"] == "sina_historical"
    assert any(
        "eastmoney_direct: malformed_kline_rows_on_or_before_curr_date" in error
        for error in out.fund_flow_evidence_meta["fallback_errors"]
    )
    assert all(row["source"] == "sina_historical" for row in out.fund_flow_evidence)


def test_direct_preserves_f52_raw_decimal_text_without_float_rounding(trading_day):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    precise_kline = _DIRECT_KLINE.replace("120000000", "12345678901234567891")

    with patch(
        "requests.get",
        return_value=_EastmoneyResponse(_direct_payload(precise_kline)),
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    record = out.fund_flow_evidence[0]
    assert record["r0_net_raw"] == "12345678901234567891"
    assert record["r0_net"] == "123456789012.34567891"


@pytest.mark.parametrize(
    "response, expected_error, expected_category",
    [
        (
            _EastmoneyResponse({}, status_code=503),
            "eastmoney_direct: http_status: 503",
            "transport",
        ),
        (
            _EastmoneyResponse({}, status_code=200),
            "eastmoney_direct: rc_missing",
            "envelope",
        ),
        (
            _EastmoneyResponse({"rc": 0}, status_code=200),
            "eastmoney_direct: data_missing_or_invalid",
            "envelope",
        ),
        (
            _EastmoneyResponse({"rc": 0, "data": {}}, status_code=200),
            "eastmoney_direct: klines_missing_or_invalid",
            "envelope",
        ),
        (
            _EastmoneyResponse(text="not-json"),
            "eastmoney_direct: json_decode:",
            "envelope",
        ),
        (
            _EastmoneyResponse(_direct_payload(_DIRECT_KLINE, rc=-1), status_code=200),
            "eastmoney_direct: rc=-1",
            "envelope",
        ),
        (
            _EastmoneyResponse(
                _direct_payload("2026-08-14"),
                status_code=200,
            ),
            "eastmoney_direct: no_usable_rows_on_or_before_curr_date",
            "validation",
        ),
    ],
)
def test_direct_failure_keeps_chain_and_falls_back_to_ths(
    response, expected_error, expected_category, trading_day, monkeypatch
):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    ak.stock_fund_flow_individual.return_value = _current_day_ths()
    p = CnAkshareProvider()
    p._ak = lambda: ak

    with patch("requests.get", side_effect=[response, ConnectionError("Sina down")]):
        out = p.get_individual_fund_flow("600519", curr_date=cn_today_str())

    assert "同花顺即时资金流净额快照" in out
    meta = out.fund_flow_evidence_meta
    assert meta["final_source"] == "ths_instant_snapshot"
    assert meta["attempted_sources"] == [
        "akshare.stock_individual_fund_flow",
        "eastmoney_direct",
        "ths_instant_snapshot",
    ]
    assert expected_error in meta["fallback_errors"] or any(
        expected_error in error for error in meta["fallback_errors"]
    )
    assert expected_category in meta["failure_categories"]
    assert "sina_historical" not in meta["attempted_sources"]
    assert "stock_fund_flow_individual" not in meta["fallback_errors"]
    assert "stock_individual_fund_flow: ConnectionError" in meta["em_typed_gap"]


def test_direct_date_mismatch_is_typed_gap_and_continues_to_current_fallback(
    trading_day,
):
    """A response containing only future rows cannot satisfy the requested as-of."""
    today = cn_today_str()
    future = (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    ak.stock_fund_flow_individual.return_value = _current_day_ths()
    p = CnAkshareProvider()
    p._ak = lambda: ak

    with patch(
        "requests.get",
        side_effect=[
            _EastmoneyResponse(_direct_payload(_DIRECT_KLINE.replace("2026-08-14", future))),
            ConnectionError("Sina down"),
        ],
    ):
        out = p.get_individual_fund_flow("600519", curr_date=today)

    assert "同花顺即时资金流净额快照" in out
    assert out.fund_flow_evidence_meta["final_source"] == "ths_instant_snapshot"
    assert any(
        "eastmoney_direct: no_usable_rows_on_or_before_curr_date" in error
        for error in out.fund_flow_evidence_meta["fallback_errors"]
    )
    assert all(record["date"] == today for record in out.fund_flow_evidence)



@pytest.mark.parametrize(
    "invalid_f52", ["", "NaN", "Infinity", "-Infinity", "1e309"]
)
def test_direct_invalid_f52_does_not_derive_r0_net_from_components(
    invalid_f52, trading_day
):
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    ak.stock_fund_flow_individual.return_value = _current_day_ths()
    p = CnAkshareProvider()
    p._ak = lambda: ak
    fields = _DIRECT_KLINE.split(",")
    fields[1] = invalid_f52
    missing_main_force = ",".join(fields)

    with patch(
        "requests.get",
        side_effect=[
            _EastmoneyResponse(_direct_payload(missing_main_force)),
            ConnectionError("Sina down"),
        ],
    ):
        out = p.get_individual_fund_flow("600519", curr_date=cn_today_str())

    assert "同花顺即时资金流净额快照" in out
    assert out.fund_flow_evidence_meta["final_source"] == "ths_instant_snapshot"
    assert any(
        "invalid_f52" in error
        for error in out.fund_flow_evidence_meta["fallback_errors"]
    )
    assert all("eastmoney_direct" not in row.get("source", "") for row in out.fund_flow_evidence)


def test_direct_failure_then_sina_success_preserves_both_sources(
    trading_day, monkeypatch
):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    sina_rows = [
        {
            "opendate": "2026-08-13",
            "netamount": "100000000",
            "r0_net": "50000000",
        }
    ]

    with patch(
        "requests.get",
        side_effect=[
            _EastmoneyResponse(_direct_payload(_DIRECT_KLINE, rc=1)),
            _EastmoneyResponse(sina_rows),
        ],
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    assert "新浪历史/收盘数据" in out
    meta = out.fund_flow_evidence_meta
    assert meta["final_source"] == "sina_historical"
    assert meta["attempted_sources"] == [
        "akshare.stock_individual_fund_flow",
        "eastmoney_direct",
        "sina_historical",
    ]
    assert "eastmoney_direct: rc=1" in meta["fallback_errors"]
    assert "stock_individual_fund_flow: ConnectionError" in meta["em_typed_gap"]
    assert all(row["source"] == "sina_historical" for row in out.fund_flow_evidence)


@pytest.mark.parametrize("curr_date", [None, "not-a-date"])
def test_invalid_curr_date_returns_structured_provider_gap(curr_date):
    out = CnAkshareProvider().get_individual_fund_flow("600519", curr_date=curr_date)

    meta = out.fund_flow_evidence_meta
    assert out.fund_flow_evidence == []
    assert meta["requested_as_of"] == curr_date
    assert meta["actual_as_of"] is None
    assert meta["as_of"] is None
    assert meta["field"] == "r0_net"
    assert meta["raw_unit"] == "元"
    assert meta["unit"] == "亿元"
    assert meta["failure_category"] == "validation"
    assert "validation" in meta["failure_categories"]
    assert meta["attempted_sources"] == []
    assert meta["direction"] == "blocked"
    assert meta["direction_allowed"] is False


def test_stale_akshare_current_row_falls_through_to_current_ths_snapshot(
    monkeypatch, trading_day
):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    today = cn_today_str()
    stale = (pd.Timestamp(today) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ak = MagicMock()
    ak.stock_individual_fund_flow.return_value = pd.DataFrame(
        {"日期": [stale], "主力净流入-净额": ["100000000"]}
    )
    ak.stock_fund_flow_individual.return_value = _current_day_ths()
    provider = CnAkshareProvider()
    provider._ak = lambda: ak

    with patch("requests.get", side_effect=ConnectionError("direct down")):
        out = provider.get_individual_fund_flow("600519", curr_date=today)

    assert out.fund_flow_evidence_meta["final_source"] == "ths_instant_snapshot"
    assert out.fund_flow_evidence_meta["selected_as_of"] == today
    assert "eastmoney_individual_fund_flow" not in out.fund_flow_evidence_meta["attempted_sources"]


def test_future_fund_flow_date_is_rejected_before_live_sources(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    provider = CnAkshareProvider()
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        out = provider.get_individual_fund_flow("600519", curr_date="2099-01-01")

    assert out.fund_flow_evidence_meta["failure_category"] == "validation"
    assert out.fund_flow_evidence_meta["direction_allowed"] is False
    assert mock_get.call_count == 0
    assert mock_post.call_count == 0


def test_legacy_netamount_only_row_keeps_its_own_direction(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    provider = CnAkshareProvider()
    provider._ak = lambda: ak
    legacy_rows = [{"opendate": "2026-08-14", "netamount": "-200000000"}]
    with patch(
        "requests.get",
        side_effect=[ConnectionError("direct down"), _EastmoneyResponse(legacy_rows)],
    ):
        out = provider.get_individual_fund_flow("600519", curr_date="2026-08-14")

    assert out.fund_flow_evidence_meta["final_source"] == "sina_historical"
    assert out.fund_flow_evidence_meta["selected_field"] == "netamount"
    assert out.fund_flow_evidence_meta["direction"] == "outflow"
    assert out.fund_flow_evidence_meta["direction_allowed"] is True
    assert out.fund_flow_evidence_meta["legacy_reference"] is True


def test_tushare_ths_daily_row_does_not_require_optional_d5(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "configured")
    payload = _tushare_payload("moneyflow_ths")
    d5_index = payload["data"]["fields"].index("net_d5_amount")
    payload["data"]["fields"].pop(d5_index)
    payload["data"]["items"][0].pop(d5_index)
    provider = CnAkshareProvider()
    with patch("requests.post", return_value=_TushareResponse(payload)):
        out, errors, _meta = provider._fetch_tushare_fund_flow(
            "600519", "2026-08-14"
        )

    assert errors == []
    assert out is not None
    ths_record = next(record for record in out.fund_flow_evidence if "netamount" in record)
    assert ths_record["netamount"] == "1.2"
    assert "net_d5_amount" not in ths_record


def test_direct_data_code_mismatch_is_rejected(trading_day):
    provider = CnAkshareProvider()
    payload = _direct_payload(_DIRECT_KLINE)
    payload["data"]["code"] = "000001"
    response = _EastmoneyResponse(payload)
    with patch("requests.get", return_value=response):
        _text, error = provider._fetch_eastmoney_direct_fund_flow(
            "600519", "2026-08-14", pd.Timestamp("2026-08-14").date(), require_curr_date=True
        )

    assert error is not None
    assert "symbol_mismatch" in error


def test_board_fund_flow_falls_back_to_ths_when_em_fails():
    ths_df = pd.DataFrame(
        {
            "板块": ["电力", "银行"],
            "涨跌幅": [2.28, 0.79],
            "净流入": [32.11, 23.38],
            "领涨股": ["乐山电力", "瑞丰银行"],
        }
    )
    ak = MagicMock()
    ak.stock_fund_flow_industry.side_effect = ConnectionError("RemoteDisconnected")
    ak.stock_board_industry_summary_ths.return_value = ths_df
    p = CnAkshareProvider()
    p._ak = lambda: ak
    out = p.get_board_fund_flow(curr_date=cn_today_str())
    assert "同花顺" in out
    assert "电力" in out
    assert "净流入" in out


def test_board_fund_flow_em_success_keeps_primary_format():
    em_df = pd.DataFrame(
        {
            "行业": ["电力", "银行"],
            "行业-涨跌幅": [2.28, 0.79],
            "净额": [21.61, 23.38],
        }
    )
    ak = MagicMock()
    ak.stock_fund_flow_industry.return_value = em_df
    p = CnAkshareProvider()
    p._ak = lambda: ak
    out = p.get_board_fund_flow(curr_date=cn_today_str())
    assert "同花顺" not in out
    assert "电力" in out


def test_individual_fund_flow_falls_back_to_ths_when_em_fails(trading_day):
    sina_df = pd.DataFrame(
        {
            "股票代码": ["600519", "000001"],
            "股票简称": ["贵州茅台", "平安银行"],
            "最新价": [1358.98, 11.0],
            "涨跌幅": ["0.62%", "-0.5%"],
            "流入资金": ["26.30亿", "1.0亿"],
            "流出资金": ["22.69亿", "1.2亿"],
            "净额": ["3.61亿", "-0.2亿"],
            "换手率": ["0.29%", "0.5%"],
        }
    )
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("RemoteDisconnected")
    ak.stock_fund_flow_individual.return_value = sina_df
    p = CnAkshareProvider()
    p._ak = lambda: ak
    with patch("requests.get", side_effect=ConnectionError("Sina history unavailable")):
        out = p.get_individual_fund_flow("600519", curr_date=cn_today_str())
    assert "同花顺即时资金流净额快照" in out
    assert "新浪历史/收盘数据" not in out
    assert "资金净额: 3.61亿" in out
    assert "不是新浪历史 netamount/r0_net 同口径主力序列" in out
    assert "600519" in out


def test_primary_historical_requires_requested_date_before_fallback(
    trading_day, monkeypatch
):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    ak = MagicMock()
    ak.stock_individual_fund_flow.return_value = pd.DataFrame(
        {
            "日期": ["2026-08-13"],
            "主力净流入-净额": ["100000000"],
        }
    )
    p = CnAkshareProvider()
    p._ak = lambda: ak
    sina_rows = [
        {
            "opendate": "2026-08-14",
            "netamount": "100000000",
            "r0_net": "50000000",
        }
    ]

    with patch(
        "requests.get",
        side_effect=[
            ConnectionError("Direct down"),
            _EastmoneyResponse(sina_rows),
        ],
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    meta = out.fund_flow_evidence_meta
    assert meta["final_source"] == "sina_historical"
    assert meta["attempted_sources"] == [
        "akshare.stock_individual_fund_flow",
        "eastmoney_direct",
        "sina_historical",
    ]
    assert any(
        "stock_individual_fund_flow: formatter reason:" in error
        and "no_requested_date_row" in error
        for error in meta["fallback_errors"]
    )
    assert [row["date"] for row in out.fund_flow_evidence] == ["2026-08-14"]


@pytest.mark.parametrize(
    "case, dates, amounts, expected_reason",
    [
        (
            "invalid_amount",
            ["2026-08-14", "2026-08-13"],
            ["100000000", "NaN"],
            "invalid_f52",
        ),
        (
            "duplicate_date",
            ["2026-08-14", "2026-08-14"],
            ["100000000", "100000001"],
            "duplicate_date",
        ),
        (
            "non_trading_date",
            ["2026-08-14", "2026-08-13"],
            ["100000000", "100000001"],
            "non_trading_date=2026-08-13",
        ),
    ],
)
def test_primary_malformed_rows_fail_closed_before_fallback(
    case, dates, amounts, expected_reason, trading_day
):
    del case
    ak = MagicMock()
    ak.stock_individual_fund_flow.return_value = pd.DataFrame(
        {"日期": dates, "主力净流入-净额": amounts}
    )
    p = CnAkshareProvider()
    p._ak = lambda: ak
    sina_rows = [
        {
            "opendate": "2026-08-14",
            "netamount": "100000000",
            "r0_net": "50000000",
        }
    ]
    calendar = (
        lambda day: day != "2026-08-13"
        if expected_reason.startswith("non_trading_date")
        else True
    )

    with patch.object(
        cn_akshare_provider, "is_cn_trading_day", side_effect=calendar
    ), patch(
        "requests.get",
        side_effect=[
            ConnectionError("Direct down"),
            _EastmoneyResponse(sina_rows),
        ],
    ):
        out = p.get_individual_fund_flow("600519", curr_date="2026-08-14")

    meta = out.fund_flow_evidence_meta
    assert meta["final_source"] == "sina_historical"
    assert any(
        "stock_individual_fund_flow: formatter reason:" in error
        and expected_reason in error
        for error in meta["fallback_errors"]
    )


def test_individual_fund_flow_nonempty_invalid_em_falls_back_with_typed_ths_evidence(
    trading_day,
):
    curr_date = cn_today_str()
    em_df = pd.DataFrame(
        {
            "日期": [curr_date],
            "主力净流入-净额": ["not-a-number"],
        }
    )
    ths_df = pd.DataFrame(
        {
            "股票代码": ["600519"],
            "股票简称": ["贵州茅台"],
            "最新价": [1358.98],
            "涨跌幅": ["0.62%"],
            "流入资金": ["26.30亿"],
            "流出资金": ["22.69亿"],
            "净额": ["3.61亿"],
            "换手率": ["0.29%"],
        }
    )
    ak = MagicMock()
    ak.stock_individual_fund_flow.return_value = em_df
    ak.stock_fund_flow_individual.return_value = ths_df
    p = CnAkshareProvider()
    p._ak = lambda: ak
    with patch("requests.get", side_effect=ConnectionError("Sina history unavailable")):
        out = p.get_individual_fund_flow("600519", curr_date=curr_date)

    assert "同花顺即时资金流净额快照" in out
    assert out.fund_flow_evidence
    assert out.fund_flow_evidence[0]["source"] == "ths_instant_snapshot"
    assert out.fund_flow_evidence_meta["source"] == "ths_instant_snapshot"
    assert out.fund_flow_evidence_meta["status"] == "selected"
    assert out.fund_flow_evidence_meta["selected_source"] == "ths_instant_snapshot"
    assert out.fund_flow_evidence_meta["selected_field"] == "netamount"
    assert out.fund_flow_evidence_meta["direction"] == "inflow"
    assert out.fund_flow_evidence_meta["direction_allowed"] is True
    assert out.fund_flow_evidence_meta["hard_guard"]["blocked"] is False
    ak.stock_fund_flow_individual.assert_called_once_with(symbol="即时")


def test_individual_fund_flow_sina_refuses_historical_date(trading_day):
    """Historical date: the THS instant snapshot must never leak (anti-lookahead).

    For past dates the Sina historical API (Source 2.5) is tried first; when it
    also fails the result is an explicit refusal — never the current-day THS
    instant snapshot.
    """
    ak = MagicMock()
    ak.stock_individual_fund_flow.return_value = pd.DataFrame(
        {"日期": [cn_today_str()], "主力净流入-净额": ["not-a-number"]}
    )
    ak.stock_fund_flow_individual.return_value = pd.DataFrame(
        {"股票代码": ["600519"], "净额": ["3.61亿"]}
    )
    p = CnAkshareProvider()
    p._ak = lambda: ak
    past = (pd.Timestamp(cn_today_str()) - timedelta(days=90)).strftime("%Y-%m-%d")
    with patch("requests.get", side_effect=ConnectionError("RemoteDisconnected")):
        out = p.get_individual_fund_flow("600519", curr_date=past)
    meta = out.fund_flow_evidence_meta
    required = (
        "stock_individual_fund_flow: formatter failure:",
        "sina historical fund flow: ConnectionError",
    )
    assert all(token in meta[field] for field in ("reason", "gap") for token in required)
    assert "历史日期" in out
    assert "不可用" in out
    assert "同花顺即时资金流净额快照" not in out
    assert "3.61亿" not in out
    assert meta["requested_as_of"] == past
    assert meta["actual_as_of"] is None
    assert meta["as_of"] is None
    assert meta["field"] == "r0_net"
    assert meta["raw_unit"] == "元"
    assert meta["unit"] == "亿元"
    assert meta["failure_category"] == "source_unavailable"
    assert meta["direction"] == "blocked"
    assert meta["direction_allowed"] is False
    assert "transport" in meta["failure_categories"]


def test_lhb_detail_falls_back_to_sina_when_em_fails():
    from tradingagents.dataflows import trade_calendar as tc

    tc.clear_cn_trade_date_cache()
    tc._TRADE_DATES_CACHE["dates"] = [pd.Timestamp("2026-08-03").date()]
    tc._TRADE_DATES_CACHE["dates_set"] = {pd.Timestamp("2026-08-03").date()}
    tc._TRADE_DATES_CACHE["loaded_at"] = 1e18

    sina_df = pd.DataFrame(
        {
            "序号": [1],
            "股票代码": ["000533"],
            "股票名称": ["顺钠股份"],
            "收盘价": [11.45],
            "对应值": [10.33],
            "成交量": [11559.9585],
            "成交额": [126090.1211],
            "指标": ["涨幅偏离值达7%的证券"],
        }
    )
    ak = MagicMock()
    ak.stock_lhb_detail_em.side_effect = ConnectionError("RemoteDisconnected")
    ak.stock_lhb_detail_daily_sina.return_value = sina_df
    p = CnAkshareProvider()
    p._ak = lambda: ak
    out = p.get_lhb_detail("000533", "2026-08-03")
    assert "新浪备用源" in out
    assert "顺钠股份" in out
    tc.clear_cn_trade_date_cache()


def test_lhb_detail_confirmed_empty_via_sina_is_normal():
    from tradingagents.dataflows import trade_calendar as tc

    tc.clear_cn_trade_date_cache()
    tc._TRADE_DATES_CACHE["dates"] = [pd.Timestamp("2026-08-03").date()]
    tc._TRADE_DATES_CACHE["dates_set"] = {pd.Timestamp("2026-08-03").date()}
    tc._TRADE_DATES_CACHE["loaded_at"] = 1e18

    sina_df = pd.DataFrame(
        {"股票代码": ["000533"], "股票名称": ["顺钠股份"]}
    )
    ak = MagicMock()
    ak.stock_lhb_detail_em.side_effect = ConnectionError("RemoteDisconnected")
    ak.stock_lhb_detail_daily_sina.return_value = sina_df
    p = CnAkshareProvider()
    p._ak = lambda: ak
    out = p.get_lhb_detail("600519", "2026-08-03")
    assert "无龙虎榜数据" in out
    assert "非异动日属正常" in out
    tc.clear_cn_trade_date_cache()
