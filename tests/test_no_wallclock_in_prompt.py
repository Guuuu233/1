"""Prompt text must not leak wall-clock retrieval timestamps or future dates."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows.trade_calendar import now_cn
from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows import y_finance as yf_mod


DATE_RE = re.compile(r"(?<!\d)(20\d{2}-\d{1,2}-\d{1,2})(?!\d)")
WALLCLOCK_MARKERS = (
    "Data retrieved on",
    "data retrieved on",
    "拉取时间",
    "retrieved on:",
)


def _norm(s: str) -> str:
    y, m, d = s.split("-")
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


def _offline_collected_pool(analysis: str) -> dict[str, object]:
    """Deterministic _fetch_all-shaped pool with no wall-clock or future dates."""
    analysis_day = datetime.strptime(analysis, "%Y-%m-%d").date()
    previous = (analysis_day - timedelta(days=1)).isoformat()
    week_ago = (analysis_day - timedelta(days=7)).isoformat()
    return {
        "stock_data": (
            "# Stock data for 600519\n"
            f"# as-of: {analysis}\n"
            "date,open,high,low,close,volume\n"
            f"{week_ago},1.0,1.1,0.9,1.1,100\n"
            f"{previous},1.2,1.3,1.1,1.2,120\n"
        ),
        "market_data_context": {
            "daily": {"as_of": analysis, "completeness": "completed"},
            "realtime": {"status": "not_applicable"},
        },
        "indicators": {"close_50_sma": 10.5, "rsi": 55.2},
        "news": f"Company news through {analysis}: stable operations.",
        "global_news": f"Macro context through {analysis}: no wall-clock marker.",
    }


def test_akshare_hist_header_has_no_wallclock():
    provider = CnAkshareProvider()
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-04-28", "2026-04-29", "2026-04-30"]),
            "Open": [1.0, 1.1, 1.2],
            "High": [1.1, 1.2, 1.3],
            "Low": [0.9, 1.0, 1.1],
            "Close": [1.05, 1.15, 1.25],
            "Volume": [100.0, 110.0, 120.0],
        }
    )
    text = provider._format_ak_hist(df, "600519", "2026-04-28", "2026-04-30")
    for marker in WALLCLOCK_MARKERS:
        assert marker not in text
    # Must not inject wall-clock day either
    today = now_cn().date().isoformat()
    assert today not in text
    assert "2026-04-30" in text  # analysis range stays


def test_yfinance_stock_header_has_no_wallclock():
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.1],
            "Low": [0.9],
            "Close": [1.05],
            "Adj Close": [1.05],
            "Volume": [100],
        },
        index=pd.to_datetime(["2026-04-30"]),
    )
    with patch.object(yf_mod.yf, "download", return_value=df):
        # get_YFin_data_online may have different name; call internal path if needed
        text = yf_mod.get_YFin_data_online("AAPL", "2026-04-01", "2026-04-30")
    for marker in WALLCLOCK_MARKERS:
        assert marker not in text
    assert now_cn().date().isoformat() not in text


def test_full_collect_no_date_after_analysis():
    """Collected pool fixture: no date string in any pool text after analysis day."""
    from tradingagents.graph import data_collector as dc

    analysis = (now_cn().date() - timedelta(days=90)).isoformat()
    pool = _offline_collected_pool(analysis)
    with patch.object(dc, "_fetch_all", return_value=pool) as mock_fetch:
        pool = dc._fetch_all("600519", analysis)
    mock_fetch.assert_called_once_with("600519", analysis)
    violations = {}
    for key, val in pool.items():
        text = val if isinstance(val, str) else str(val)
        for marker in WALLCLOCK_MARKERS:
            assert marker not in text, f"{key} still has wall-clock marker {marker}"
        futures = sorted({_norm(m) for m in DATE_RE.findall(text) if _norm(m) > analysis})
        if futures:
            violations[key] = futures
    assert violations == {}, f"future dates in prompt: {violations}"
