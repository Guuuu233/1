from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_board_fund_flow() -> str:
    """获取今日行业板块资金流向排名，用于判断板块轮动信号和个股所在板块的资金吸引力。"""
    return route_to_vendor("get_board_fund_flow")


@tool
def get_individual_fund_flow(
    symbol: Annotated[str, "股票代码，格式如 600519.SH"],
    curr_date: Annotated[str | None, "分析日期 YYYY-MM-DD，用于截断资金流历史"] = None,
) -> str:
    """获取个股近5日主力资金净流向，判断机构资金进出方向。symbol 格式如 600519.SH。"""
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
def get_hot_stocks_xq() -> str:
    """获取雪球热搜股票列表，反映散户当前关注热点。"""
    return route_to_vendor("get_hot_stocks_xq")

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
