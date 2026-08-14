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

from tradingagents.dataflows.fund_flow_evidence import FundFlowText
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.trade_calendar import cn_today_str

# ── Task 2: Eastmoney backup sources ──────────────────────────────────


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
            "日期": [cn_today_str(), cn_today_str()],
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
            "日期": [curr_date],
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
    assert out.fund_flow_evidence[0]["date"] == curr_date
    assert out.fund_flow_evidence[0]["netamount"] == "3.61"
    assert "r0_net" not in out.fund_flow_evidence[0]
    assert out.fund_flow_evidence_meta["source"] == "ths_instant_snapshot"
    assert out.fund_flow_evidence_meta["status"] == "available"
    meta = out.fund_flow_evidence_meta
    assert [item["source"] for item in meta["attempted_sources"]] == [
        "eastmoney_individual_fund_flow",
        "sina_historical",
        "ths_instant_snapshot",
    ]
    assert [item["status"] for item in meta["attempted_sources"]] == [
        "failed",
        "failed",
        "success",
    ]
    assert meta["final_source"] == "ths_instant_snapshot"
    assert meta["em_typed_gap"]["source"] == "eastmoney_individual_fund_flow"
    assert meta["em_typed_gap"]["status"] == "unavailable"
    assert [item["source"] for item in meta["fallback_errors"]] == [
        "eastmoney_individual_fund_flow",
        "sina_historical",
    ]
    ak.stock_fund_flow_individual.assert_called_once_with(symbol="即时")


def test_individual_fund_flow_all_failures_keep_typed_gap_and_redact_errors():
    curr_date = cn_today_str()
    em_df = pd.DataFrame(
        {
            "日期": [curr_date],
            "主力净流入-净额": ["not-a-number"],
        }
    )
    ak = MagicMock()
    ak.stock_individual_fund_flow.return_value = em_df
    ak.stock_fund_flow_individual.side_effect = RuntimeError(
        "cookie=secret-cookie-value"
    )
    p = CnAkshareProvider()
    p._ak = lambda: ak
    with patch(
        "requests.get",
        side_effect=ConnectionError(
            "https://example.invalid/feed?token=secret-token"
        ),
    ):
        out = p.get_individual_fund_flow("600519", curr_date=curr_date)

    meta = out.fund_flow_evidence_meta
    assert "【数据获取失败】" in out
    assert [item["source"] for item in meta["attempted_sources"]] == [
        "eastmoney_individual_fund_flow",
        "sina_historical",
        "ths_instant_snapshot",
    ]
    assert all(item["status"] == "failed" for item in meta["attempted_sources"])
    assert meta["final_source"] == "unavailable_gap"
    assert meta["em_typed_gap"]["source"] == "eastmoney_individual_fund_flow"
    assert meta["em_typed_gap"]["status"] == "unavailable"
    assert [item["source"] for item in meta["fallback_errors"]] == [
        "eastmoney_individual_fund_flow",
        "sina_historical",
        "ths_instant_snapshot",
    ]
    assert [item["error_type"] for item in meta["fallback_errors"]] == [
        "formatter_failure",
        "ConnectionError",
        "RuntimeError",
    ]
    serialized = json.dumps(meta, ensure_ascii=False)
    assert "secret-cookie-value" not in serialized
    assert "secret-token" not in serialized
    assert "https://example.invalid" not in serialized


def test_ths_missing_source_date_is_typed_gap_without_as_of():
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
            "净额": ["3.61亿"],
        }
    )
    ak = MagicMock()
    ak.stock_individual_fund_flow.return_value = em_df
    ak.stock_fund_flow_individual.return_value = ths_df
    provider = CnAkshareProvider()
    provider._ak = lambda: ak
    with patch("requests.get", side_effect=ConnectionError("Sina unavailable")):
        out = provider.get_individual_fund_flow("600519", curr_date=curr_date)

    meta = out.fund_flow_evidence_meta
    assert "【备用数据源：同花顺即时资金流净额快照】" not in out
    assert not out.fund_flow_evidence
    assert meta["final_source"] == "unavailable_gap"
    assert meta["as_of"] is None
    assert meta["attempted_sources"][-1]["source"] == "ths_instant_snapshot"
    assert meta["attempted_sources"][-1]["status"] == "failed"
    assert meta["fallback_errors"][-1]["error_type"] == "missing_source_date"


def test_ths_future_source_date_is_rejected():
    curr_date = cn_today_str()
    future_source_date = (
        pd.Timestamp(curr_date) + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    em_df = pd.DataFrame(
        {
            "日期": [curr_date],
            "主力净流入-净额": ["not-a-number"],
        }
    )
    ths_df = pd.DataFrame(
        {
            "股票代码": ["600519"],
            "日期": [future_source_date],
            "净额": ["3.61亿"],
        }
    )
    ak = MagicMock()
    ak.stock_individual_fund_flow.return_value = em_df
    ak.stock_fund_flow_individual.return_value = ths_df
    provider = CnAkshareProvider()
    provider._ak = lambda: ak
    with patch("requests.get", side_effect=ConnectionError("Sina unavailable")):
        out = provider.get_individual_fund_flow("600519", curr_date=curr_date)

    meta = out.fund_flow_evidence_meta
    assert "【备用数据源：同花顺即时资金流净额快照】" not in out
    assert not out.fund_flow_evidence
    assert meta["final_source"] == "unavailable_gap"
    assert meta["as_of"] is None
    assert meta["fallback_errors"][-1]["error_type"] == "future_source_date"
    assert future_source_date in meta["fallback_errors"][-1]["reason"]


def test_future_analysis_date_fails_closed_before_provider_calls():
    curr_date = cn_today_str()
    future_date = (pd.Timestamp(curr_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    provider = CnAkshareProvider()
    provider._ak = lambda: (_ for _ in ()).throw(
        AssertionError("provider must not be contacted for a future analysis date")
    )

    out = provider.get_individual_fund_flow("600519", curr_date=future_date)

    meta = out.fund_flow_evidence_meta
    assert "【数据获取失败】" in out
    assert not out.fund_flow_evidence
    assert meta["final_source"] == "unavailable_gap"
    assert meta["as_of"] is None
    assert meta["attempted_sources"] == [
        {
            "source": "analysis_date_guard",
            "status": "failed",
            "reason": "analysis date is in the future",
        }
    ]
    assert meta["fallback_errors"] == [
        {
            "source": "analysis_date_guard",
            "error_type": "future_analysis_date",
            "reason": "analysis date is in the future",
        }
    ]


def test_nested_typed_gap_metadata_is_whitelisted_and_redacted():
    curr_date = cn_today_str()
    em_df = pd.DataFrame(
        {
            "日期": [curr_date],
            "主力净流入-净额": ["not-a-number"],
        }
    )
    ak = MagicMock()
    ak.stock_individual_fund_flow.return_value = em_df
    ak.stock_fund_flow_individual.side_effect = RuntimeError(
        "cookie=secret-ths-cookie"
    )
    provider = CnAkshareProvider()
    provider._ak = lambda: ak
    provider._format_individual_fund_flow_em = lambda *args: FundFlowText(
        "typed gap",
        evidence=[],
        evidence_meta={
            "source": "eastmoney_individual_fund_flow",
            "status": "unavailable",
            "reason": {
                "detail": "cookie=secret-cookie",
                "nested": {"Authorization": "Bearer secret-auth"},
            },
            "gap": {
                "url": "https://example.invalid/feed?token=secret-token&signature=secret-signature",
                "detail": "api_key=secret-key",
            },
            "date": curr_date,
            "as_of": curr_date,
            "untrusted_metadata": "must be dropped",
        },
    )
    with patch(
        "requests.get",
        side_effect=ConnectionError(
            "https://example.invalid/feed?token=secret-sina-token"
        ),
    ):
        out = provider.get_individual_fund_flow("600519", curr_date=curr_date)

    meta = out.fund_flow_evidence_meta
    typed_gap = meta["em_typed_gap"]
    serialized = json.dumps(meta, ensure_ascii=False)
    for secret in (
        "secret-cookie",
        "secret-auth",
        "secret-token",
        "secret-signature",
        "secret-key",
        "https://example.invalid",
    ):
        assert secret not in serialized
    assert "untrusted_metadata" not in typed_gap
    assert "date" not in typed_gap
    assert typed_gap["as_of"] is None
    assert all(
        "untrusted_metadata" not in item
        for item in meta["fallback_errors"]
    )


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
