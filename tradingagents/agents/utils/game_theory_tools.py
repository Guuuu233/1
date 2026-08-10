from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_board_fund_flow(
    curr_date: Annotated[str | None, "分析日期 YYYY-MM-DD；历史日拒绝即时板块资金流快照"] = None,
) -> str:
    """获取今日行业板块资金流向排名（即时快照）。历史日期分析不可用。"""
    return route_to_vendor("get_board_fund_flow", curr_date)


@tool
def get_individual_fund_flow(
    symbol: Annotated[str, "股票代码，格式如 600519.SH"],
    curr_date: Annotated[
        str | None,
        "分析日期 YYYY-MM-DD，用于截断资金流历史；同花顺即时资金流净额快照不是新浪历史主力序列",
    ] = None,
) -> str:
    """获取个股资金流数据；同花顺即时资金流净额快照不是新浪历史主力序列。"""
    return route_to_vendor("get_individual_fund_flow", symbol, curr_date)


@tool
def get_lhb_detail(
    symbol: Annotated[str, "股票代码，格式如 600519.SH"],
    date: Annotated[str, "日期，格式 YYYY-MM-DD"],
) -> str:
    """获取个股龙虎榜数据，非异动日无数据属正常。symbol 格式如 600519.SH，date 格式 YYYY-MM-DD。"""
    return route_to_vendor("get_lhb_detail", symbol, date)


@tool
def get_zt_pool(
    date: Annotated[str, "日期，格式 YYYY-MM-DD"],
) -> str:
    """获取市场涨停板情绪池，反映市场整体情绪温度，date 格式 YYYY-MM-DD。"""
    return route_to_vendor("get_zt_pool", date)


@tool
def get_hot_stocks_xq(
    curr_date: Annotated[str | None, "分析日期 YYYY-MM-DD；历史日拒绝雪球热搜快照"] = None,
) -> str:
    """获取雪球热搜股票列表（当前热度快照）。历史日期分析不可用。"""
    return route_to_vendor("get_hot_stocks_xq", curr_date)

@tool
def get_shareholder_count(
    symbol: Annotated[str, "股票代码，格式如 600519"],
    curr_date: Annotated[str | None, "当前分析日期 YYYY-MM-DD"] = None,
) -> str:
    """获取股东户数变动与筹码集中度。"""
    return route_to_vendor("get_shareholder_count", symbol, curr_date=curr_date)

@tool
def get_margin_trading(
    symbol: Annotated[str, "股票代码，格式如 600519"],
    curr_date: Annotated[str | None, "当前分析日期 YYYY-MM-DD"] = None,
) -> str:
    """获取融资融券交易明细。"""
    return route_to_vendor("get_margin_trading", symbol, curr_date=curr_date)

@tool
def get_northbound_flow(
    symbol: Annotated[str, "股票代码，格式如 600519"],
    curr_date: Annotated[str | None, "当前分析日期 YYYY-MM-DD"] = None,
) -> str:
    """获取北向资金（陆股通）持股变动。"""
    return route_to_vendor("get_northbound_flow", symbol, curr_date=curr_date)
