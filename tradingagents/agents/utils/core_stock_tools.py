from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.
    Uses the configured core_stock_apis vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    return route_to_vendor("get_stock_data", symbol, start_date, end_date)


@tool
def get_cn_indices(
    curr_date: Annotated[str | None, "分析日期 YYYY-MM-DD，严格过滤 <= trade_date 防前视偏差"] = None,
    look_back_days: Annotated[int, "回溯天数，默认 30 天"] = 30,
) -> str:
    """获取国内核心大盘指数（上证指数、深证成指、沪深300、创业板指、科创50等）历史行情与趋势。"""
    return route_to_vendor("get_cn_indices", curr_date, look_back_days=look_back_days)


@tool
def get_global_indices(
    curr_date: Annotated[str | None, "分析日期 YYYY-MM-DD，严格过滤 <= trade_date 防前视偏差"] = None,
    look_back_days: Annotated[int, "回溯天数，默认 30 天"] = 30,
) -> str:
    """获取全球核心市场指数（标普500、纳斯达克、道琼斯、恒生指数、日经225等）历史行情与跨市场联动。"""
    return route_to_vendor("get_global_indices", curr_date, look_back_days=look_back_days)


@tool
def get_major_assets(
    curr_date: Annotated[str | None, "分析日期 YYYY-MM-DD，严格过滤 <= trade_date 防前视偏差"] = None,
    look_back_days: Annotated[int, "回溯天数，默认 30 天"] = 30,
) -> str:
    """获取全球大类资产与大宗商品（黄金、原油、美债10年期收益率、美元指数、铜等）历史行情与宏观传导信号。"""
    return route_to_vendor("get_major_assets", curr_date, look_back_days=look_back_days)

