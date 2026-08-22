from .base import BaseMarketDataProvider
from ..y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from ..yfinance_news import get_news_yfinance, get_global_news_yfinance
from ..vendor_result import VendorEmpty, VendorFail


def _classify_text_result(
    result,
    *,
    empty_prefixes: tuple[str, ...],
    fail_prefixes: tuple[str, ...],
):
    """Tag yfinance string results with typed vendor semantics.

    yfinance helpers fold failures into an "Error ..." string and confirmed-empty
    into a "No ... found" string.  Before the typed-result refactor (KNOWN_ISSUES
    #1) both were ordinary strings, so an "Error fetching news" message stopped
    the vendor chain exactly like a successful hit instead of falling through to
    the next provider.
    """
    if not isinstance(result, str):
        return result
    if result.startswith(empty_prefixes):
        return VendorEmpty(result)
    if result.startswith(fail_prefixes):
        return VendorFail(result)
    return result


class YFinanceProvider(BaseMarketDataProvider):
    @property
    def name(self) -> str:
        return "yfinance"

    def _normalize_symbol(self, symbol: str) -> str:
        s = symbol.strip().upper()
        # yfinance uses .SS for Shanghai and .SZ for Shenzhen.
        if s.endswith(".SH"):
            return s[:-3] + ".SS"
        return s

    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> str:
        return get_YFin_data_online(self._normalize_symbol(symbol), start_date, end_date)

    def get_indicators(
        self, symbol: str, indicator: str, curr_date: str, look_back_days: int
    ) -> str:
        return get_stock_stats_indicators_window(
            self._normalize_symbol(symbol), indicator, curr_date, look_back_days
        )

    def get_fundamentals(self, ticker: str, curr_date: str = None) -> str:
        return get_yfinance_fundamentals(self._normalize_symbol(ticker), curr_date)

    def get_balance_sheet(
        self, ticker: str, freq: str = "quarterly", curr_date: str = None
    ) -> str:
        return get_yfinance_balance_sheet(self._normalize_symbol(ticker), freq, curr_date)

    def get_cashflow(
        self, ticker: str, freq: str = "quarterly", curr_date: str = None
    ) -> str:
        return get_yfinance_cashflow(self._normalize_symbol(ticker), freq, curr_date)

    def get_income_statement(
        self, ticker: str, freq: str = "quarterly", curr_date: str = None
    ) -> str:
        return get_yfinance_income_statement(self._normalize_symbol(ticker), freq, curr_date)

    def get_news(self, ticker: str, start_date: str, end_date: str) -> str:
        result = get_news_yfinance(self._normalize_symbol(ticker), start_date, end_date)
        return _classify_text_result(
            result,
            empty_prefixes=("No news found",),
            fail_prefixes=("Error fetching news",),
        )

    def get_global_news(
        self, curr_date: str, look_back_days: int = 7, limit: int = 50
    ) -> str:
        result = get_global_news_yfinance(curr_date, look_back_days, limit)
        return _classify_text_result(
            result,
            empty_prefixes=("No global news found",),
            fail_prefixes=("Error fetching global news",),
        )

    def get_insider_transactions(self, symbol: str, curr_date: str = None) -> str:
        result = get_yfinance_insider_transactions(self._normalize_symbol(symbol))
        return _classify_text_result(
            result,
            empty_prefixes=("No insider transactions data found",),
            fail_prefixes=("Error retrieving insider transactions",),
        )

    def get_global_indices(
        self, curr_date: str = None, look_back_days: int = 30
    ) -> str:
        if curr_date is None:
            return "【数据获取失败】全球核心指数 — 原因：缺少分析基准日期 (来源: yfinance)"
        import yfinance as yf
        from datetime import datetime, timedelta
        from ..macro_market_utils import calculate_series_metrics, build_global_indices_markdown

        try:
            end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        except Exception:
            return f"【数据获取失败】全球核心指数 — 原因：非法日期格式 {curr_date} (来源: yfinance)"

        start_dt = end_dt - timedelta(days=max(look_back_days * 2, 90))
        start_str = start_dt.strftime("%Y-%m-%d")
        end_inclusive = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        indices_map = {
            "标普500": {"symbol": "^GSPC", "code": "^GSPC"},
            "纳斯达克综合": {"symbol": "^IXIC", "code": "^IXIC"},
            "道琼斯": {"symbol": "^DJI", "code": "^DJI"},
            "恒生指数": {"symbol": "^HSI", "code": "^HSI"},
            "恒生科技指数": {"symbol": "^HSTECH", "code": "^HSTECH"},
            "日经225": {"symbol": "^N225", "code": "^N225"},
            "韩国KOSPI": {"symbol": "^KS11", "code": "^KS11"},
            "德国DAX": {"symbol": "^GDAXI", "code": "^GDAXI"},
            "法国CAC40": {"symbol": "^FCHI", "code": "^FCHI"},
            "英国富时100": {"symbol": "^FTSE", "code": "^FTSE"},
        }

        results = {}
        for name, meta in indices_map.items():
            sym = meta["symbol"]
            try:
                ticker = yf.Ticker(sym)
                data = ticker.history(start=start_str, end=end_inclusive)
                if data is not None and not data.empty:
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    df = data.reset_index()
                    metrics = calculate_series_metrics(df, curr_date, price_col="Close")
                    if metrics:
                        metrics["code"] = meta["code"]
                        results[name] = metrics
            except Exception:
                continue

        if not results:
            return "【数据获取失败】全球核心指数 — 原因：yfinance 接口无可用数据 (来源: yfinance)"
        return build_global_indices_markdown(results, curr_date, source="yfinance")

    def get_major_assets(
        self, curr_date: str = None, look_back_days: int = 30
    ) -> str:
        if curr_date is None:
            return "【数据获取失败】全球大类资产 — 原因：缺少分析基准日期 (来源: yfinance)"
        import yfinance as yf
        from datetime import datetime, timedelta
        from ..macro_market_utils import calculate_series_metrics, build_major_assets_markdown

        try:
            end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        except Exception:
            return f"【数据获取失败】全球大类资产 — 原因：非法日期格式 {curr_date} (来源: yfinance)"

        start_dt = end_dt - timedelta(days=max(look_back_days * 2, 90))
        start_str = start_dt.strftime("%Y-%m-%d")
        end_inclusive = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        assets_map = {
            "COMEX黄金": {"symbol": "GC=F", "code": "GC=F", "category": "贵金属", "macro_signal": "避险情绪 / 实际利率映射"},
            "WTI原油": {"symbol": "CL=F", "code": "CL=F", "category": "能源商品", "macro_signal": "通胀预期 / 工业能源基准"},
            "布伦特原油": {"symbol": "BZ=F", "code": "BZ=F", "category": "能源商品", "macro_signal": "全球原油基准价格"},
            "美债10年期收益率": {"symbol": "^TNX", "code": "^TNX", "category": "主权债券", "unit": "%", "macro_signal": "无风险利率锚 / 资产折现率"},
            "美元指数": {"symbol": "DX-Y.NYB", "code": "DXY", "category": "外汇货币", "macro_signal": "全球流动性与非美汇率压力"},
            "COMEX铜": {"symbol": "HG=F", "code": "HG=F", "category": "工业金属", "macro_signal": "全球制造业需求晴雨表"},
            "比特币": {"symbol": "BTC-USD", "code": "BTC-USD", "category": "数字资产", "macro_signal": "全球高贝塔流动性风险偏好"},
        }

        results = {}
        for name, meta in assets_map.items():
            sym = meta["symbol"]
            try:
                ticker = yf.Ticker(sym)
                data = ticker.history(start=start_str, end=end_inclusive)
                if data is not None and not data.empty:
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    df = data.reset_index()
                    metrics = calculate_series_metrics(df, curr_date, price_col="Close")
                    if metrics:
                        metrics["code"] = meta["code"]
                        metrics["category"] = meta["category"]
                        metrics["macro_signal"] = meta.get("macro_signal", "")
                        if "unit" in meta:
                            metrics["unit"] = meta["unit"]
                        results[name] = metrics
            except Exception:
                continue

        if not results:
            return "【数据获取失败】全球大类资产 — 原因：yfinance 接口无可用数据 (来源: yfinance)"
        return build_major_assets_markdown(results, curr_date, source="yfinance")

    def get_cn_indices(
        self, curr_date: str = None, look_back_days: int = 30
    ) -> str:
        if curr_date is None:
            return "【数据获取失败】国内核心大盘指数 — 原因：缺少分析基准日期 (来源: yfinance)"
        import yfinance as yf
        from datetime import datetime, timedelta
        from ..macro_market_utils import calculate_series_metrics, build_cn_indices_markdown

        try:
            end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        except Exception:
            return f"【数据获取失败】国内核心大盘指数 — 原因：非法日期格式 {curr_date} (来源: yfinance)"

        start_dt = end_dt - timedelta(days=max(look_back_days * 2, 90))
        start_str = start_dt.strftime("%Y-%m-%d")
        end_inclusive = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        cn_map = {
            "上证指数": {"symbol": "000001.SS", "code": "000001.SH"},
            "深证成指": {"symbol": "399001.SZ", "code": "399001.SZ"},
            "沪深300": {"symbol": "000300.SS", "code": "000300.SH"},
            "创业板指": {"symbol": "399006.SZ", "code": "399006.SZ"},
            "科创50": {"symbol": "000688.SS", "code": "000688.SH"},
        }

        results = {}
        for name, meta in cn_map.items():
            sym = meta["symbol"]
            try:
                ticker = yf.Ticker(sym)
                data = ticker.history(start=start_str, end=end_inclusive)
                if data is not None and not data.empty:
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    df = data.reset_index()
                    metrics = calculate_series_metrics(df, curr_date, price_col="Close")
                    if metrics:
                        metrics["code"] = meta["code"]
                        results[name] = metrics
            except Exception:
                continue

        if not results:
            return "【数据获取失败】国内核心大盘指数 — 原因：yfinance 接口无可用数据 (来源: yfinance)"
        return build_cn_indices_markdown(results, curr_date, source="yfinance")
