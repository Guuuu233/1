"""Commit 3b: snapshot-only sources refuse historical analysis dates."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.dataflows.trade_calendar import (
    SNAPSHOT_ONLY_REFUSAL,
    cn_today_str,
    is_historical_analysis_date,
    now_cn,
    snapshot_historical_refusal,
)
from tradingagents.dataflows.vendor_result import result_to_prompt
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider


def test_is_historical_true_for_past_false_for_today_and_missing():
    today = cn_today_str()
    past = (now_cn().date() - timedelta(days=90)).isoformat()
    assert is_historical_analysis_date(past) is True
    assert is_historical_analysis_date(today) is False
    assert is_historical_analysis_date(None) is False
    assert is_historical_analysis_date("") is False


def test_snapshot_refusal_message_stable():
    past = (now_cn().date() - timedelta(days=10)).isoformat()
    msg = snapshot_historical_refusal(past, source_label="股权质押（全市场快照）")
    assert msg is not None
    assert msg.startswith("【数据获取失败】")
    assert SNAPSHOT_ONLY_REFUSAL in msg
    assert "股权质押" in msg
    assert snapshot_historical_refusal(cn_today_str()) is None


@pytest.fixture
def provider():
    return CnAkshareProvider()


def test_share_pledge_refuses_historical(provider):
    past = (now_cn().date() - timedelta(days=90)).isoformat()
    with patch.object(provider, "_ak", side_effect=AssertionError("must not call network")):
        out = provider.get_share_pledge("600519", curr_date=past)
    assert SNAPSHOT_ONLY_REFUSAL in out
    assert out.startswith("【数据获取失败】")


def test_board_fund_flow_refuses_historical(provider):
    past = (now_cn().date() - timedelta(days=90)).isoformat()
    with patch.object(provider, "_ak", side_effect=AssertionError("must not call network")):
        out = provider.get_board_fund_flow(curr_date=past)
    assert SNAPSHOT_ONLY_REFUSAL in out


def test_hot_stocks_refuses_historical(provider):
    past = (now_cn().date() - timedelta(days=90)).isoformat()
    with patch.object(provider, "_ak", side_effect=AssertionError("must not call network")):
        out = provider.get_hot_stocks_xq(curr_date=past)
    assert SNAPSHOT_ONLY_REFUSAL in out


def test_insider_refuses_historical(provider):
    past = (now_cn().date() - timedelta(days=90)).isoformat()
    with patch.object(provider, "_ak", side_effect=AssertionError("must not call network")):
        out = provider.get_insider_transactions("600519", curr_date=past)
    assert SNAPSHOT_ONLY_REFUSAL in out


def test_realtime_quotes_refuses_historical(provider):
    past = (now_cn().date() - timedelta(days=90)).isoformat()
    out = provider.get_realtime_quotes(["600519"], curr_date=past)
    assert SNAPSHOT_ONLY_REFUSAL in out


def test_fundamentals_profile_refused_abstract_may_remain(provider):
    """Company Profile refused on historical; abstract path still attempted with cutoff."""
    past = (now_cn().date() - timedelta(days=90)).isoformat()

    empty = MagicMock()
    empty.empty = True

    # Avoid network: empty info + empty abstract → may raise or return partial.
    # We only assert profile section is the snapshot refusal when profile path runs.
    with patch.object(provider, "_ak") as ak_mock:
        ak = MagicMock()
        ak_mock.return_value = ak
        # individual info returns a non-empty frame so profile branch is hit
        import pandas as pd

        info = pd.DataFrame({"item": ["股票简称"], "value": ["贵州茅台"]})
        ak.stock_individual_info_em.return_value = info
        ak.stock_financial_abstract.return_value = pd.DataFrame()
        # sina map not needed if abstract empty
        with patch.object(
            provider,
            "_sina_effective_announce_map",
            side_effect=AssertionError("abstract empty should not map"),
        ):
            out = provider.get_fundamentals("600519", curr_date=past)

    assert "### Company Profile" in out
    assert SNAPSHOT_ONLY_REFUSAL in out
    # Must not embed raw profile table rows as if historical
    assert "股票简称" not in out.split("### Company Profile", 1)[1].split("###", 1)[0]


def test_same_day_board_does_not_auto_refuse(provider):
    """Today is not historical; refusal helper returns None (network still may fail)."""
    today = cn_today_str()
    assert snapshot_historical_refusal(today, source_label="板块") is None


def test_zt_pool_refuses_historical(provider):
    """stock_zt_pool_em is near-window only; historical analysis must refuse."""
    past = (now_cn().date() - timedelta(days=90)).isoformat()
    with patch.object(provider, "_ak", side_effect=AssertionError("must not call network")):
        out = provider.get_zt_pool(past)
    prompt = result_to_prompt(out)
    assert SNAPSHOT_ONLY_REFUSAL in prompt
    assert prompt.startswith("【数据获取失败】")


def test_zt_pool_same_day_does_not_auto_refuse(provider):
    today = cn_today_str()
    assert snapshot_historical_refusal(today, source_label="涨停") is None

