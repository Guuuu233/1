"""DAV-69 — Eastmoney backup sources for intermittently-unreachable interfaces.

The Eastmoney push2his endpoints (fund flow, LHB) intermittently drop the
connection (RemoteDisconnected) on the current IP. Each affected method now
falls back to an alternative source inside the provider:

- get_board_fund_flow        EM stock_fund_flow_industry  -> THS stock_board_industry_summary_ths
- get_individual_fund_flow   EM stock_individual_fund_flow -> Sina historical close API
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

from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.trade_calendar import cn_today_str

# ── Task 2: Eastmoney backup sources ──────────────────────────────────


class _EastmoneyResponse:
    def __init__(self, payload=None, *, status_code=200, text=None):
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


_DIRECT_KLINE = (
    "2026-08-14,120000000,-10000000,-20000000,50000000,70000000,"
    "0.12,-0.01,-0.02,0.05,0.07,12.34,1.20,unknown-a,unknown-b"
)


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


def test_individual_fund_flow_direct_eastmoney_success_is_structured_and_typed():
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
    assert mock_get.call_args.args[0].endswith(
        "/api/qt/stock/fflow/daykline/get"
    )
    assert request_kwargs["params"]["secid"] == "1.600519"
    assert request_kwargs["params"]["klt"] == "101"
    assert "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61" in request_kwargs["params"]["fields2"]
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
    assert record["large_net"] == "0.5"
    assert record["super_large_net"] == "0.7"
    assert "netamount" not in record
    assert record["field_semantics"]["r0_net"].startswith("主力净额")
    assert record["component_semantics"]["large_net"].startswith("大单净额")

    meta = out.fund_flow_evidence_meta
    assert meta["source"] == "eastmoney_direct"
    assert meta["final_source"] == "eastmoney_direct"
    assert meta["attempted_sources"] == [
        "akshare.stock_individual_fund_flow",
        "eastmoney_direct",
    ]
    assert "stock_individual_fund_flow: ConnectionError" in meta["em_typed_gap"]
    assert meta["fallback_errors"] == [
        "stock_individual_fund_flow: ConnectionError"
    ]
    assert not ak.stock_fund_flow_individual.called
    assert meta["field_mapping"]["f52"].endswith("r0_net)")


def test_individual_fund_flow_direct_filters_future_rows_without_lookahead():
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


@pytest.mark.parametrize(
    "response, expected_error",
    [
        (
            _EastmoneyResponse({}, status_code=503),
            "eastmoney_direct: http_status: 503",
        ),
        (
            _EastmoneyResponse(text="not-json"),
            "eastmoney_direct: json_decode:",
        ),
        (
            _EastmoneyResponse(_direct_payload(_DIRECT_KLINE, rc=-1), status_code=200),
            "eastmoney_direct: rc=-1",
        ),
        (
            _EastmoneyResponse(
                _direct_payload("2026-08-14,1,2,3,4,5,6,7,8,9"),
                status_code=200,
            ),
            "eastmoney_direct: no_usable_rows_on_or_before_curr_date",
        ),
    ],
)
def test_direct_failure_keeps_chain_and_falls_back_to_ths(response, expected_error):
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
        "sina_historical",
        "ths_instant_snapshot",
    ]
    assert expected_error in meta["fallback_errors"] or any(
        expected_error in error for error in meta["fallback_errors"]
    )
    assert "sina historical fund flow: ConnectionError" in meta["fallback_errors"]
    assert "stock_individual_fund_flow: ConnectionError" in meta["em_typed_gap"]


def test_direct_missing_f52_does_not_derive_r0_net_from_components():
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError("EM unavailable")
    ak.stock_fund_flow_individual.return_value = _current_day_ths()
    p = CnAkshareProvider()
    p._ak = lambda: ak
    fields = _DIRECT_KLINE.split(",")
    fields[1] = ""
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


def test_direct_failure_then_sina_success_preserves_both_sources():
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


def test_individual_fund_flow_falls_back_to_ths_when_em_fails():
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


def test_individual_fund_flow_nonempty_invalid_em_falls_back_with_typed_ths_evidence():
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
    assert out.fund_flow_evidence_meta["status"] == "available"
    ak.stock_fund_flow_individual.assert_called_once_with(symbol="即时")


def test_individual_fund_flow_sina_refuses_historical_date():
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
