#!/usr/bin/env python3
"""Smoke test script for all TradingAgents data providers."""

import sys
import time
from datetime import datetime
from tradingagents.dataflows.interface import route_to_vendor

TEST_SYMBOLS = ["600519", "000001", "300750"]
METHODS = [
    ("K线行情", "get_stock_data", lambda s, d: (s, "2025-01-01", d)),
    ("新闻资讯", "get_news", lambda s, d: (s, "2025-01-01", d)),
    ("高管增减持", "get_insider_transactions", lambda s, d: (s, d)),
    ("板块资金流", "get_board_fund_flow", lambda s, d: ()),
    ("个股资金流", "get_individual_fund_flow", lambda s, d: (s,)),
    ("龙虎榜", "get_lhb_detail", lambda s, d: (s, d)),
    ("涨停池", "get_zt_pool", lambda s, d: (d,)),
    ("雪球热股", "get_hot_stocks_xq", lambda s, d: ()),
    ("解禁风险", "get_restricted_release", lambda s, d: (s, d)),
    ("股权质押", "get_share_pledge", lambda s, d: (s, d)),
    ("业绩预告", "get_earnings_forecast", lambda s, d: (s, d)),
    ("股东户数", "get_shareholder_count", lambda s, d: (s, d)),
    ("融资融券", "get_margin_trading", lambda s, d: (s, d)),
    ("北向资金", "get_northbound_flow", lambda s, d: (s, d)),
]


def run_smoke():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Starting Provider Smoke Test ({today}) ===")
    results = []

    for name, method_name, arg_builder in METHODS:
        passed = 0
        failed = 0
        total_time = 0.0

        for symbol in TEST_SYMBOLS:
            args = arg_builder(symbol, today)
            start_t = time.time()
            try:
                res = route_to_vendor(method_name, *args)
                elapsed = time.time() - start_t
                total_time += elapsed
                if res and "失败" not in str(res)[:30]:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1

        avg_time = total_time / len(TEST_SYMBOLS)
        status = "✅ 正常" if passed > 0 else "❌ 失败"
        results.append((name, method_name, f"{passed}/{len(TEST_SYMBOLS)}", f"{avg_time:.2f}s", status))

    print("\n" + "=" * 60)
    print(f"{'数据项':<12} | {'接口方法':<24} | {'成功率':<8} | {'平均耗时':<8} | {'状态'}")
    print("=" * 60)
    for name, method, rate, avg_t, st in results:
        print(f"{name:<12} | {method:<24} | {rate:<8} | {avg_t:<8} | {st}")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke()
