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

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

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


def test_individual_fund_flow_typed_em_gap_to_ths_preserves_attempt_chain():
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

    metadata = out.fund_flow_evidence_meta
    assert metadata["requested_as_of"] == curr_date
    assert metadata["as_of"] == curr_date
    assert metadata["status"] == "available"
    assert metadata["final_source"] == "ths_instant_snapshot"
    assert metadata["attempted_sources"] == [
        {
            "source": "eastmoney_individual_fund_flow",
            "status": "unavailable",
            "reason": "no finite r0_net rows on or before curr_date",
        },
        {
            "source": "sina_historical",
            "status": "failed",
            "reason": "ConnectionError",
        },
        {
            "source": "ths_instant_snapshot",
            "status": "available",
            "reason": "同花顺即时资金流净额是总净额，未将其等同于新浪历史 r0_net 主力序列",
        },
    ]
    assert metadata["fallback_errors"] == metadata["attempted_sources"][:2]
    assert metadata["em_typed_gap"] == {
        "symbol": "600519",
        "requested_as_of": curr_date,
        "source": "eastmoney_individual_fund_flow",
        "source_family": "eastmoney",
        "algorithm_group": "new_algorithm_group",
        "source_group": "new_algorithm_group",
        "legacy_web_algorithm": False,
        "period_kind": "historical_daily",
        "unit": "亿元",
        "status": "unavailable",
        "reason": "no finite r0_net rows on or before curr_date",
        "gap": "【数据获取失败】资金流 evidence：no finite r0_net rows on or before curr_date",
        "retrieved_at": metadata["em_typed_gap"]["retrieved_at"],
        "as_of": None,
    }
    assert metadata["em_typed_gap"]["retrieved_at"]
    assert out.fund_flow_evidence[0]["netamount"] == "3.61"
    assert "r0_net" not in out.fund_flow_evidence[0]


def test_individual_fund_flow_all_sources_failed_retains_typed_failures():
    curr_date = cn_today_str()
    ak = MagicMock()
    ak.stock_individual_fund_flow.side_effect = ConnectionError(
        "https://vendor.invalid?sign=secret&token=secret"
    )
    ak.stock_fund_flow_individual.side_effect = TimeoutError("provider timeout")
    p = CnAkshareProvider()
    p._ak = lambda: ak
    with patch("requests.get", side_effect=ConnectionError("Sina ?key=secret")):
        out = p.get_individual_fund_flow("600519", curr_date=curr_date)

    metadata = out.fund_flow_evidence_meta
    assert metadata["requested_as_of"] == curr_date
    assert metadata["as_of"] is None
    assert metadata["status"] == "unavailable"
    assert metadata["final_source"] is None
    assert [item["source"] for item in metadata["attempted_sources"]] == [
        "eastmoney_individual_fund_flow",
        "sina_historical",
        "ths_instant_snapshot",
    ]
    assert [item["status"] for item in metadata["attempted_sources"]] == [
        "failed",
        "failed",
        "failed",
    ]
    assert metadata["fallback_errors"] == metadata["attempted_sources"]
    assert metadata["em_typed_gap"]["reason"] == "ConnectionError"
    assert "secret" not in repr(metadata)
    for token in (
        "stock_individual_fund_flow: ConnectionError",
        "sina historical fund flow: ConnectionError",
        "stock_fund_flow_individual: TimeoutError",
    ):
        assert token in metadata["reason"]


def test_individual_fund_flow_em_success_has_no_fallback_errors():
    curr_date = cn_today_str()
    previous_date = (pd.Timestamp(curr_date) - timedelta(days=1)).strftime("%Y-%m-%d")
    em_df = pd.DataFrame(
        {
            "日期": [previous_date, curr_date],
            "主力净流入-净额": [100000000, -200000000],
        }
    )
    ak = MagicMock()
    ak.stock_individual_fund_flow.return_value = em_df
    p = CnAkshareProvider()
    p._ak = lambda: ak

    out = p.get_individual_fund_flow("600519", curr_date=curr_date)

    metadata = out.fund_flow_evidence_meta
    assert metadata["requested_as_of"] == curr_date
    assert metadata["as_of"] == curr_date
    assert metadata["status"] == "partial"
    assert metadata["final_source"] == "eastmoney_individual_fund_flow"
    assert metadata["attempted_sources"] == [
        {
            "source": "eastmoney_individual_fund_flow",
            "status": "partial",
            "reason": "东方财富来源仅提供主力净额；未将其等同于总净额 netamount",
        }
    ]
    assert metadata["fallback_errors"] == []
    assert metadata["em_typed_gap"] is None
    assert ak.stock_fund_flow_individual.call_count == 0


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
