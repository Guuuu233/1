import copy
import json
import threading
from unittest.mock import patch

from tradingagents.dataflows.fund_flow_evidence import FundFlowText

from tradingagents.graph.data_collector import (
    DataCollector,
    _build_source_provenance,
    _fetch_all,
    make_cache_key,
)


def test_make_cache_key():
    assert make_cache_key("600519", "2026-03-12") == "600519_2026-03-12"


def test_source_provenance_keeps_actual_as_of_and_explicit_gap():
    results = {
        "news": "## 600519 新闻（2026-08-05 至 2026-08-11；最新发布时间：2026-08-11 15:00:00）：",
        "global_news": "【数据获取失败】历史宏观新闻不可用",
        "zt_pool": "【数据获取失败】涨停板情绪池：无可验证数据日期",
    }
    provenance = _build_source_provenance(
        results,
        "2026-08-11",
        daily_as_of="2026-08-11",
    )

    assert provenance["news"]["requested_as_of"] == "2026-08-11"
    assert provenance["news"]["as_of"] == "2026-08-11"
    assert "gap" not in provenance["news"]
    assert provenance["global_news"]["status"] == "failed"
    assert "gap" in provenance["global_news"]
    assert provenance["zt_pool"]["status"] == "failed"


def test_source_provenance_uses_pool_actual_date_not_request_window():
    provenance = _build_source_provenance(
        {"zt_pool": "涨停池（2026-08-04，同花顺 fuyao）：共 2 只"},
        "2026-08-11",
        daily_as_of="2026-08-11",
    )
    assert provenance["zt_pool"]["as_of"] == "2026-08-04"
    assert "gap" not in provenance["zt_pool"]


def test_source_provenance_does_not_infer_news_window_end_as_actual_date():
    provenance = _build_source_provenance(
        {"news": "## 新闻（2026-08-05 至 2026-08-11）："},
        "2026-08-11",
        daily_as_of="2026-08-11",
    )
    assert provenance["news"]["as_of"] is None
    assert "gap" in provenance["news"]


def test_source_provenance_extracts_global_news_latest_publication_date():
    provenance = _build_source_provenance(
        {"global_news": "## 全球市场新闻（最新发布时间：2026-08-10 15:00:00）"},
        "2026-08-11",
        daily_as_of="2026-08-11",
    )
    assert provenance["global_news"]["as_of"] == "2026-08-10"
    assert "gap" not in provenance["global_news"]


def test_source_provenance_extracts_latest_publication_date():
    provenance = _build_source_provenance(
        {"news": "## 新闻（最新发布时间：2026-08-10 15:00:00）"},
        "2026-08-11",
        daily_as_of="2026-08-11",
    )
    assert provenance["news"]["as_of"] == "2026-08-10"


def test_collect_populates_required_keys():
    collector = DataCollector()
    stub_pool = {
        "stock_data": "data", "indicators": {}, "news": "n", "global_news": "gn",
        "fundamentals": "f", "balance_sheet": "bs", "cashflow": "cf",
        "income_statement": "is", "fund_flow_board": "ffb",
        "fund_flow_individual": "ffi", "lhb": "lhb",
        "insider_transactions": "it", "zt_pool": "zt", "hot_stocks": "hs",
    }
    with patch("tradingagents.graph.data_collector._fetch_all", return_value=stub_pool):
        result = collector.collect("600519", "2026-03-12")
    assert "stock_data" in result
    assert "lhb" in result
    assert "zt_pool" in result


def test_collect_uses_cache_on_second_call():
    collector = DataCollector()
    stub_pool = {"stock_data": "x", "indicators": {}}
    with patch("tradingagents.graph.data_collector._fetch_all", return_value=stub_pool) as mock_fetch:
        collector.collect("600519", "2026-03-12")
        collector.collect("600519", "2026-03-12")
    assert mock_fetch.call_count == 1


def test_evict_removes_from_cache():
    collector = DataCollector()
    collector._cache["600519_2026-03-12"] = {"stock_data": "x"}
    collector.evict("600519", "2026-03-12")
    assert "600519_2026-03-12" not in collector._cache


def test_get_window_short_returns_14_day_window():
    collector = DataCollector()
    pool = {"stock_data": "x", "indicators": {}}
    sliced = collector.get_window(pool, horizon="short", trade_date="2026-03-12")
    assert sliced["_data_window"] == "14天"
    assert sliced["_horizon"] == "short"


def test_get_window_medium_returns_90_day_window():
    collector = DataCollector()
    pool = {"stock_data": "x", "indicators": {}}
    sliced = collector.get_window(pool, horizon="medium", trade_date="2026-03-12")
    assert sliced["_data_window"] == "90天"
    assert sliced["_horizon"] == "medium"


def test_collect_returns_defensive_copy_of_cache():
    collector = DataCollector()
    stub_pool = {
        "stock_data": "data",
        "indicators": {},
        "details": {"tags": ["a"], "quote": {"price": 1.0}},
    }
    with patch("tradingagents.graph.data_collector._fetch_all", return_value=stub_pool):
        result = collector.collect("600519", "2026-03-12")

    result["indicators"]["close"] = 100
    result["details"]["tags"].append("mutated")
    result["details"]["quote"]["price"] = 999

    cached = collector._cache["600519_2026-03-12"]
    assert cached["indicators"] == {}
    assert cached["details"]["tags"] == ["a"]
    assert cached["details"]["quote"]["price"] == 1.0
    assert result is not cached


def test_get_returns_defensive_copy_of_cache():
    collector = DataCollector()
    key = make_cache_key("600519", "2026-03-12")
    collector._cache[key] = {"items": [{"v": 1}]}

    result = collector.get("600519", "2026-03-12")
    result["items"].append({"v": 2})

    assert collector._cache[key] == {"items": [{"v": 1}]}
    assert collector.get("600519", "2026-03-12") == {"items": [{"v": 1}]}


def test_get_window_does_not_mutate_source_pool():
    collector = DataCollector()
    pool = {"stock_data": "x", "items": [{"v": 1}]}

    sliced = collector.get_window(pool, horizon="short", trade_date="2026-03-12")
    sliced["stock_data"] = "mutated"
    sliced["items"].append({"v": 2})

    assert pool == {"stock_data": "x", "items": [{"v": 1}]}
    assert sliced is not pool


def test_evict_clears_cache_and_refcount_but_retains_lock_then_refetches():
    collector = DataCollector()
    key = make_cache_key("600519", "2026-03-12")
    collector._cache[key] = {"stock_data": "old"}
    collector._locks[key] = threading.Lock()
    collector._refcounts[key] = 1

    collector.evict("600519", "2026-03-12")

    assert key not in collector._cache
    # 锁对象必须保留：其他线程可能仍持有该锁的引用，删除会导致新 collect()
    # 创建新锁、破坏 per-key 互斥。
    assert key in collector._locks
    assert key not in collector._refcounts

    with patch(
        "tradingagents.graph.data_collector._fetch_all",
        return_value={"stock_data": "new"},
    ) as mock_fetch:
        result = collector.collect("600519", "2026-03-12")
    assert result["stock_data"] == "new"
    assert mock_fetch.call_count == 1


def test_normalized_daily_csv_separates_requested_and_actual_as_of():
    from tradingagents.graph import data_collector

    raw = "# vendor: fixture\nDate,Open,High,Low,Close,Volume\n2026-08-10,1,1,1,1,1\n"
    with patch.object(data_collector, "_safe", return_value=raw), \
         patch.object(data_collector, "FETCH_ALL_TIMEOUT", 1):
        result = data_collector._fetch_all("600519", "2026-08-11")
    csv = result["stock_data"]
    assert "# requested-as-of: 2026-08-11" in csv
    assert "# as-of: 2026-08-10" in csv
    assert result["market_data_context"]["daily"]["as_of"] == "2026-08-10"


def test_stale_daily_as_of_enters_provenance_gap():
    from tradingagents.graph import data_collector

    provenance = data_collector._build_source_provenance(
        {"stock_data": "Date,Open,High,Low,Close,Volume\n2026-08-10,1,1,1,1,1"},
        "2026-08-11",
        daily_as_of="2026-08-10",
    )
    assert provenance["stock_data"]["requested_as_of"] == "2026-08-11"
    assert provenance["stock_data"]["as_of"] == "2026-08-10"
    assert "早于请求日期" in provenance["stock_data"]["gap"]


def test_failed_stock_data_enters_ledger_and_gap():
    from tradingagents.graph import data_collector

    with patch.object(data_collector, "_safe", return_value=""), \
         patch.object(data_collector, "FETCH_ALL_TIMEOUT", 1):
        result = data_collector._fetch_all("600519", "2026-08-11")

    ledger = result["market_data_context"]["data_failure_ledger"]
    stock_entries = [entry for entry in ledger if entry["source"] == "stock_data"]
    assert stock_entries
    assert "无有效完整日线数据" in stock_entries[0]["gap"]
    assert result["market_data_context"]["source_provenance"]["stock_data"]["as_of"] is None


def test_empty_fund_flow_evidence_preserves_provider_gap_at_report_boundary():
    from tradingagents.graph import data_collector

    provider_meta = {
        "symbol": "601398.SH",
        "requested_as_of": "2026-08-14",
        "actual_as_of": None,
        "as_of": None,
        "source": "fund_flow_individual",
        "algorithm_group": "unknown_algorithm_group",
        "legacy_web_algorithm": False,
        "field": "r0_net",
        "raw_unit": "元",
        "unit": "亿元",
        "status": "unavailable",
        "direction": "blocked",
        "direction_allowed": False,
        "hard_guard": {
            "blocked": True,
            "direction_allowed": False,
            "reason": "all provider sources unavailable",
        },
        "reason": "all provider sources unavailable",
        "gap": "【数据获取失败】资金流 evidence：all provider sources unavailable",
        "failure_category": "transport",
        "attempted_sources": [
            "akshare.stock_individual_fund_flow",
            "eastmoney_direct",
            "sina_historical",
        ],
        "fallback_errors": [
            "stock_individual_fund_flow: SSLError",
            "eastmoney_direct: request: SSLError",
            "sina historical fund flow: ConnectionError",
        ],
        "failure_categories": ["transport"],
        "final_source": "unavailable",
        "last_attempted_source": "sina_historical",
    }
    provider_gap = FundFlowText(
        "【数据获取失败】all fund-flow sources unavailable",
        evidence=[],
        evidence_meta=provider_meta,
    )

    def fake_safe(tool, _payload):
        if tool is data_collector.get_individual_fund_flow:
            return provider_gap
        return ""

    with patch.object(data_collector, "_safe", side_effect=fake_safe), \
         patch.object(data_collector, "FETCH_ALL_TIMEOUT", 1):
        result = data_collector._fetch_all("601398.SH", "2026-08-14")

    serialized = json.loads(
        json.dumps(result["market_data_context"], ensure_ascii=False)
    )
    fund_flow = serialized["fund_flow_evidence"]

    assert fund_flow["records"] == []
    assert fund_flow["requested_as_of"] == "2026-08-14"
    assert fund_flow["actual_as_of"] is None
    assert fund_flow["as_of"] is None
    assert fund_flow["field"] == "r0_net"
    assert fund_flow["raw_unit"] == "元"
    assert fund_flow["unit"] == "亿元"
    assert fund_flow["failure_category"] == "transport"
    assert fund_flow["attempted_sources"] == provider_meta["attempted_sources"]
    assert fund_flow["fallback_errors"] == provider_meta["fallback_errors"]
    assert fund_flow["failure_categories"] == ["transport"]
    assert fund_flow["final_source"] == "unavailable"
    assert fund_flow["last_attempted_source"] == "sina_historical"
    assert fund_flow["reason"] == "all provider sources unavailable"
    assert fund_flow["gap"] == provider_meta["gap"]
    assert fund_flow["direction"] == "blocked"
    assert fund_flow["direction_allowed"] is False
    assert fund_flow["hard_guard"]["blocked"] is True
    assert fund_flow["algorithm_group"] == "unknown_algorithm_group"
    assert fund_flow["legacy_web_algorithm"] is False
    assert fund_flow["status"] == "unavailable"
    assert "new_algorithm_sources" not in fund_flow
    assert fund_flow["source_family"] == "fund_flow_individual"
    assert fund_flow["summary"]["status"] == "partial"
    assert fund_flow["validation"]["status"] == "not_checked"


def test_empty_fund_flow_evidence_preserves_provider_records_at_report_boundary():
    from tradingagents.graph import data_collector

    provider_meta = {
        "status": "unavailable",
        "reason": "provider-specific failure chain",
        "records": [
            {
                "source": "provider_failure_chain",
                "failure_chain": [
                    {
                        "source": "akshare.stock_individual_fund_flow",
                        "failure_category": "transport",
                    },
                    {
                        "source": "eastmoney_direct",
                        "failure_category": "transport",
                    },
                ],
            }
        ],
        "summary": {"status": "provider_unavailable", "attempt_count": 2},
        "validation": {
            "status": "provider_checked",
            "mismatches": ["provider-owned marker"],
        },
        "source_family": "provider_fund_flow_chain",
    }
    provider_snapshot = copy.deepcopy(provider_meta)
    provider_gap = FundFlowText(
        "【数据获取失败】provider records must survive",
        evidence=[],
        evidence_meta=provider_meta,
    )

    def fake_safe(tool, _payload):
        if tool is data_collector.get_individual_fund_flow:
            return provider_gap
        return ""

    with patch.object(data_collector, "_safe", side_effect=fake_safe), \
         patch.object(data_collector, "FETCH_ALL_TIMEOUT", 1):
        result = data_collector._fetch_all("601398.SH", "2026-08-14")

    fund_flow = result["market_data_context"]["fund_flow_evidence"]
    serialized = json.loads(
        json.dumps(result["market_data_context"], ensure_ascii=False)
    )["fund_flow_evidence"]

    assert fund_flow["records"] == provider_snapshot["records"]
    assert serialized["records"] == provider_snapshot["records"]
    assert serialized["summary"] == provider_snapshot["summary"]
    assert serialized["validation"] == provider_snapshot["validation"]
    assert serialized["source_family"] == provider_snapshot["source_family"]
    assert fund_flow["records"] is not provider_meta["records"]
    assert fund_flow["records"] is not provider_gap.fund_flow_evidence_meta["records"]
    assert fund_flow["records"][0] is not provider_meta["records"][0]
    assert (
        fund_flow["records"][0]["failure_chain"]
        is not provider_meta["records"][0]["failure_chain"]
    )

    fund_flow["records"][0]["failure_chain"].append({"source": "collector mutation"})
    assert provider_meta == provider_snapshot
    assert provider_gap.fund_flow_evidence_meta == provider_snapshot


def test_fetch_all_completes_executor_with_fast_tools():
    with patch("tradingagents.graph.data_collector._safe", return_value=""), \
         patch("tradingagents.graph.data_collector.FETCH_ALL_TIMEOUT", 1):
        result = _fetch_all("600519", "2025-01-02")

    assert "stock_data" in result
    assert "market_data_context" in result
