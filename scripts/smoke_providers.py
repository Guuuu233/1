#!/usr/bin/env python3
"""Smoke test script for TradingAgents data providers.

Offline by default: run provider *dispatch fixture checks* only and do not call
real data sources. Set LIVE=1 to exercise real providers.

Exit semantics: a data item passes only when every probe symbol passes; any
failure (missing method, non-falsy/失败 response, or exception) makes the item
fail, is reported with its symbol and error, and makes the script exit non-zero.
"""

import os
import sys
import time
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tradingagents.dataflows.interface import _registry, route_to_vendor

TEST_SYMBOLS = ["600519", "000001", "300750"]
METHODS = [
    ("K线行情", "get_stock_data", lambda s, d: (s, "2025-01-01", d)),
    ("新闻资讯", "get_news", lambda s, d: (s, "2025-01-01", d)),
    ("高管增减持", "get_insider_transactions", lambda s, d: (s, d)),
    ("板块资金流", "get_board_fund_flow", lambda s, d: (d,)),
    ("个股资金流", "get_individual_fund_flow", lambda s, d: (s, d)),
    ("龙虎榜", "get_lhb_detail", lambda s, d: (s, d)),
    ("涨停池", "get_zt_pool", lambda s, d: (d,)),
    ("雪球热股", "get_hot_stocks_xq", lambda s, d: (d,)),
    ("解禁风险", "get_restricted_release", lambda s, d: (s, d)),
    ("股权质押", "get_share_pledge", lambda s, d: (s, d)),
    ("业绩预告", "get_earnings_forecast", lambda s, d: (s, d)),
    ("股东户数", "get_shareholder_count", lambda s, d: (s, d)),
    ("融资融券", "get_margin_trading", lambda s, d: (s, d)),
    ("北向资金", "get_northbound_flow", lambda s, d: (s, d)),
]


def _is_offline() -> bool:
    return os.environ.get("LIVE", "").strip().lower() not in ("1", "true", "yes")


def _method_available(method_name: str) -> bool:
    return any(
        hasattr(_registry.get(name), method_name)
        for name in _registry.list_names()
    )


_FAILURE_MARKERS = (
    "失败",
    "error",
    "exception",
    "no data found",
    "nodata",
    "无数据",
    "数据获取失败",
)


def _is_failure_result(res) -> bool:
    """Unified live-result failure predicate.

    Treats falsy results (None, "", [], {}), explicit {"ok": false} /
    {"success": false}, error-shaped dicts, and strings carrying common
    failure markers as failures so "No data found" and {"ok": false} cannot
    be reported as success.
    """
    if res is None or res is False or res == "":
        return True
    if isinstance(res, dict):
        if not res:
            return True
        if "ok" in res and not res["ok"]:
            return True
        if "success" in res and not res["success"]:
            return True
        if any(k in res for k in ("error", "error_code", "errmsg", "message")):
            value = res.get("error") or res.get("error_code") or res.get("errmsg") or res.get("message")
            if value:
                return True
        if not any(res.values()):
            return True
        return False
    if isinstance(res, list):
        return not res
    text = str(res).strip().lower()
    return not text or any(marker in text for marker in _FAILURE_MARKERS)


def run_checks():
    """Run every method x probe symbol and return per-item result dicts.

    Each dict: label, method_name, passed, failed, avg_time, errors.
    An item passes only when passed == len(TEST_SYMBOLS) and failed == 0.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    offline = _is_offline()
    results = []

    for label, method_name, arg_builder in METHODS:
        passed = 0
        failed = 0
        total_time = 0.0
        errors = []

        for symbol in TEST_SYMBOLS:
            args = arg_builder(symbol, today)
            start_t = time.time()
            try:
                if offline:
                    if _method_available(method_name):
                        passed += 1
                    else:
                        failed += 1
                        errors.append(f"{symbol}: registry 无 provider 实现该方法")
                else:
                    res = route_to_vendor(method_name, *args)
                    elapsed = time.time() - start_t
                    total_time += elapsed
                    if _is_failure_result(res):
                        failed += 1
                        errors.append(f"{symbol}: 返回失败结果 {str(res)[:120]!r}")
                    else:
                        passed += 1
            except Exception as e:
                failed += 1
                errors.append(f"{symbol}: {type(e).__name__}: {e}")

        results.append({
            "label": label,
            "method_name": method_name,
            "passed": passed,
            "failed": failed,
            "avg_time": (total_time / len(TEST_SYMBOLS)) if TEST_SYMBOLS else 0.0,
            "errors": errors,
        })

    return results


def format_summary(results) -> str:
    lines = [
        "\n" + "=" * 60,
        f"{'数据项':<12} | {'接口方法':<24} | {'成功率':<8} | {'平均耗时':<8} | {'状态'}",
        "=" * 60,
    ]
    for item in results:
        status = "✅ 正常" if item["failed"] == 0 else "❌ 失败"
        lines.append(
            f"{item['label']:<12} | {item['method_name']:<24} | "
            f"{item['passed']}/{len(TEST_SYMBOLS)} | "
            f"{item['avg_time']:.2f}s | {status}"
        )
    lines.append("=" * 60)

    failed_items = [item for item in results if item["failed"] > 0]
    if failed_items:
        lines.append("失败明细:")
        for item in failed_items:
            for err in item["errors"]:
                lines.append(f"  [{item['label']}/{item['method_name']}] {err}")
    return "\n".join(lines)


def run_smoke():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"=== Starting Provider Smoke Test ({today}) ===")
    print(f"mode={'offline-fixture' if _is_offline() else 'live'}")

    results = run_checks()
    print(format_summary(results))

    failed_any = any(item["failed"] > 0 for item in results)
    print("smoke-result: " + ("FAIL" if failed_any else "PASS"))
    if failed_any:
        raise SystemExit(1)


if __name__ == "__main__":
    run_smoke()
