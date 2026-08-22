from datetime import timedelta

import pandas as pd

from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider
from tradingagents.dataflows.trade_calendar import cn_today_str, now_cn


class _EmptyAkshareClient:
    def stock_main_stock_holder(self, stock):
        return pd.DataFrame()


class _NewsFallbackProvider(CnAkshareProvider):
    def __init__(self):
        self.news_calls = []

    def _ak(self):
        return _EmptyAkshareClient()

    def get_news(self, ticker: str, start_date: str, end_date: str) -> str:
        self.news_calls.append((ticker, start_date, end_date))
        return "fallback news"


def test_insider_transactions_fallback_uses_analysis_date_window(frozen_trade_date):
    provider = _NewsFallbackProvider()

    end_date = frozen_trade_date
    start_date = (now_cn().date() - timedelta(days=14)).isoformat()
    result = provider.get_insider_transactions("600519.SH", curr_date=end_date)

    assert "fallback news" in result
    assert provider.news_calls == [("600519.SH", start_date, end_date)]
