"""Regression coverage for Sina historical per-day fund flow (Bug E, DAV-88).

Bug E: 个股资金流向历史日期获取不到. When Eastmoney's
``stock_individual_fund_flow`` is rate-limited and the analysis date is
historical, the old Sina backup was a same-day snapshot and had to be refused
(anti-lookahead) → no backup source for historical dates.

Fix: add the Sina historical money-flow endpoint as Source 2.5 (direct requests,
Referer/UA, 10s timeout; akshare has no wrapper). Rows are filtered to
``opendate <= curr_date`` and rendered like the Eastmoney table.
"""

import json
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.trade_calendar import cn_today_str

_SINA_HIST_ROWS = [
    {
        "opendate": "2026-07-22",
        "trade": "1306.92",
        "netamount": "-872073609.06",
        "ratioamount": "-0.158967",
        "r0_net": "-704366610.82",
        "r0_ratio": "-0.1283",
        "r1_net": "-123000000.0",
        "r2_net": "-581366610.82",
        "r3_net": "10000000.0",
        "r4_net": "20000000.0",
    },
    {
        "opendate": "2026-07-23",
        "trade": "1310.11",
        "netamount": "123456789.0",
        "ratioamount": "0.03",
        "r0_net": "98765432.1",
        "r0_ratio": "0.02",
        "r1_net": "50000000.0",
        "r2_net": "48765432.1",
        "r3_net": "10000000.0",
        "r4_net": "20000000.0",
    },
    {
        "opendate": "2026-07-28",  # == curr_date, kept
        "trade": "1315.22",
        "netamount": "500000000.0",
        "ratioamount": "0.12",
        "r0_net": "300000000.0",
        "r0_ratio": "0.08",
        "r1_net": "150000000.0",
        "r2_net": "150000000.0",
        "r3_net": "100000000.0",
        "r4_net": "100000000.0",
    },
    {
        "opendate": "2026-07-29",  # > curr_date, must be excluded (lookahead)
        "trade": "1320.0",
        "netamount": "999999999.0",
        "ratioamount": "0.9",
        "r0_net": "888888888.0",
        "r0_ratio": "0.8",
        "r1_net": "777777777.0",
        "r2_net": "111111111.0",
        "r3_net": "555555555.0",
        "r4_net": "666666666.0",
    },
]


class _FakeResp:
    def __init__(self, payload):
        self.text = json.dumps(payload, ensure_ascii=False)

    def raise_for_status(self):
        return None


class _SinaHistFixtureProvider(CnAkshareProvider):
    """Eastmoney fails; historical date must route to the Sina history endpoint."""

    def _ak(self):
        class _Ak:
            def stock_individual_fund_flow(self, stock, market):
                raise RuntimeError("eastmoney rate-limited")

            def stock_fund_flow_individual(self, symbol="即时"):
                raise AssertionError("current-day snapshot must not run for a historical date")

        return _Ak()


class _CurrentDaySnapshotProvider(CnAkshareProvider):
    """Eastmoney fails on a current date: the instant snapshot remains available."""

    def __init__(self, snapshot_error=None):
        self._snapshot_error = snapshot_error
        self._row = {
            "股票代码": "600519",
            "最新价": "1700.00",
            "涨跌幅": "1.23",
            "净额": "5.60亿",
            "流入资金": "10.00亿",
            "流出资金": "4.40亿",
            "换手率": "0.50",
        }

    def _ak(self):
        class _Ak:
            def __init__(self, row, snapshot_error):
                self._row = row
                self._snapshot_error = snapshot_error

            def stock_individual_fund_flow(self, stock, market):
                raise RuntimeError("eastmoney rate-limited")

            def stock_fund_flow_individual(self, symbol="即时"):
                if self._snapshot_error is not None:
                    raise self._snapshot_error
                return pd.DataFrame([self._row])

        return _Ak(self._row, self._snapshot_error)


def test_sina_historical_fund_flow_serves_historical_date():
    """Mocked Sina history → a historical analysis date gets a fund-flow table."""
    provider = _SinaHistFixtureProvider()
    with patch("requests.get", return_value=_FakeResp(_SINA_HIST_ROWS)):
        text = provider.get_individual_fund_flow("600519", curr_date="2026-07-28")

    assert "新浪历史" in text
    assert "截至于 2026-07-28" in text
    # Rows on/before curr_date are rendered...
    assert "2026-07-22" in text
    assert "2026-07-28" in text
    # ...the row after curr_date is NOT (anti-lookahead unchanged).
    assert "2026-07-29" not in text
    # Field mapping labels are present.
    assert "净流入额" in text
    assert "主力净流入" in text
    assert "净占比" in text


def test_sina_history_request_uses_referer_ua_timeout_and_daima():
    provider = _SinaHistFixtureProvider()
    with patch("requests.get", return_value=_FakeResp(_SINA_HIST_ROWS)) as mock_get:
        provider.get_individual_fund_flow("600519", curr_date="2026-07-28")

    args = mock_get.call_args
    url = args.args[0] if args.args else ""
    kwargs = args.kwargs
    assert "MoneyFlow.ssl_qsfx_zjlrqs" in url
    assert "daima=sh600519" in url
    assert kwargs["headers"]["Referer"] == "https://finance.sina.com.cn/"
    assert kwargs["timeout"] == 10
    assert "User-Agent" in kwargs["headers"]


def test_sina_history_failure_reports_explicitly():
    provider = _SinaHistFixtureProvider()
    with patch("requests.get", side_effect=RuntimeError("connection reset")):
        text = provider.get_individual_fund_flow("600519", curr_date="2026-07-28")

    assert "【数据获取失败】" in text
    assert "新浪历史" in text
    assert "RuntimeError" in text
    assert "不可用" in text


def test_sina_history_empty_rows_reports_explicitly():
    """Endpoint OK but nothing on/before curr_date → explicit refusal, not blank."""
    provider = _SinaHistFixtureProvider()
    with patch("requests.get", return_value=_FakeResp([])):
        text = provider.get_individual_fund_flow("600519", curr_date="2026-07-28")

    assert "【数据获取失败】" in text
    assert "新浪历史" in text
    assert "不可用" in text


def test_current_day_history_close_precedes_snapshot_after_eastmoney_failure():
    """A same-day Sina close row is preferred over the live snapshot path."""
    today = cn_today_str()
    rows = [
        *_SINA_HIST_ROWS,
        {
            "opendate": today,
            "trade": "72.80",
            "netamount": "-286620931.5",
            "ratioamount": "-0.1331",
            "r0_net": "-381071910.46",
        },
    ]
    provider = _CurrentDaySnapshotProvider()
    with patch("requests.get", return_value=_FakeResp(rows)):
        text = provider.get_individual_fund_flow("600519", curr_date=today)

    assert "新浪历史/收盘数据" in text
    assert today in text
    assert "-2.87" in text
    assert "-3.81" in text
    assert "当日主力资金净流向快照" not in text


def test_current_day_without_close_row_still_tries_snapshot():
    """Before the Sina close arrives, the existing instant snapshot is tried."""
    today = cn_today_str()
    provider = _CurrentDaySnapshotProvider()
    with patch("requests.get", return_value=_FakeResp(_SINA_HIST_ROWS)) as mock_get:
        text = provider.get_individual_fund_flow("600519", curr_date=today)

    assert mock_get.called
    assert "当日主力资金净流向快照" in text
    assert "净额: 5.60亿" in text
    assert "最新价 1700.00" in text


def test_current_day_without_close_or_snapshot_reports_data_gap():
    """A missing close plus failed snapshot is an explicit data gap."""
    today = cn_today_str()
    provider = _CurrentDaySnapshotProvider(
        snapshot_error=AttributeError("stock_fund_flow_individual unavailable")
    )
    with patch("requests.get", return_value=_FakeResp(_SINA_HIST_ROWS)):
        text = provider.get_individual_fund_flow("600519", curr_date=today)

    assert "【数据获取失败】" in text
    assert "stock_fund_flow_individual: AttributeError" in text
    assert "当日主力资金净流向快照" not in text
