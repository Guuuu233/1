import asyncio
import json
import threading
from types import SimpleNamespace
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


def test_fetch_all_propagates_manual_calibration_gap_to_ledger_and_provenance():
    from tradingagents.graph import data_collector

    trade_date = "2026-08-11"
    manual_gap = {
        "source": "sina_app_manual_calibration",
        "status": "blocked",
        "reason": "新浪 App 无可验证公开接口；截图仅作人工校准",
        "retrieved_at": "2026-08-11T12:00:00+00:00",
        "as_of": None,
        "requested_as_of": "2026-08-09",
        "gap": "【数据获取失败】资金流 evidence：新浪 App 无可验证公开接口；截图仅作人工校准",
    }
    evidence = [
        {
            "date": f"2026-08-{day:02d}",
            "source": "eastmoney_individual_fund_flow",
            "status": "available",
            "period_kind": "historical_daily",
            "window": "1d",
            "unit": "亿元",
            "netamount": "1.0",
            "r0_net": "0.5",
        }
        for day in (5, 6, 7, 10, 11)
    ]
    fund_flow = FundFlowText(
        "东财结构化资金流 evidence",
        evidence=evidence,
        evidence_meta={
            "source": "eastmoney_individual_fund_flow",
            "status": "available",
            "requested_as_of": trade_date,
            "manual_calibration_gap": manual_gap,
        },
    )
    raw_csv = "# vendor: fixture\nDate,Open,High,Low,Close,Volume\n2026-08-11,1,1,1,1,1\n"

    def fake_safe(tool, payload):
        if tool is data_collector.get_stock_data:
            return raw_csv
        if tool is data_collector.get_individual_fund_flow:
            return fund_flow
        if tool is data_collector._fetch_realtime_context:
            return {
                "status": "not_applicable",
                "source": None,
                "quote_as_of": None,
                "retrieved_at": None,
                "error": None,
                "quote": None,
            }
        return ""

    with patch.object(data_collector, "_safe", side_effect=fake_safe), \
         patch.object(data_collector, "FETCH_ALL_TIMEOUT", 1):
        result = data_collector._fetch_all("600519", trade_date)

    context = result["market_data_context"]
    assert context["fund_flow_evidence"]["status"] == "available"
    assert context["fund_flow_evidence"]["manual_calibration_gap"] == manual_gap

    provenance = context["source_provenance"]["fund_flow_individual"]
    assert provenance["status"] == "available"
    provenance_gap = provenance["manual_calibration_gap"]
    assert {key: provenance_gap[key] for key in manual_gap} == manual_gap
    assert provenance_gap["gap_type"] == "manual_calibration_gap"
    assert provenance_gap["blocking"] is False
    assert provenance_gap["non_blocking"] is True

    manual_entries = [
        entry
        for entry in context["data_failure_ledger"]
        if entry.get("gap_type") == "manual_calibration_gap"
    ]
    assert len(manual_entries) == 1
    entry = manual_entries[0]
    assert entry["source"] == manual_gap["source"]
    assert entry["status"] == manual_gap["status"]
    assert entry["reason"] == manual_gap["reason"]
    assert entry["manual_calibration_gap"] == manual_gap
    assert entry["blocking"] is False
    assert entry["non_blocking"] is True

    data_collector._append_manual_calibration_gap(context["data_failure_ledger"], manual_gap)
    assert len(
        [
            item
            for item in context["data_failure_ledger"]
            if item.get("gap_type") == "manual_calibration_gap"
        ]
    ) == 1


def test_fetch_all_without_manual_calibration_gap_keeps_ledger_and_provenance_clean():
    from tradingagents.graph import data_collector

    fund_flow = FundFlowText(
        "东财结构化资金流 evidence",
        evidence=[
            {
                "date": f"2026-08-{day:02d}",
                "source": "eastmoney_individual_fund_flow",
                "status": "available",
                "period_kind": "historical_daily",
                "window": "1d",
                "unit": "亿元",
                "netamount": "1.0",
                "r0_net": "0.5",
            }
            for day in range(7, 12)
        ],
        evidence_meta={
            "source": "eastmoney_individual_fund_flow",
            "status": "available",
        },
    )
    raw_csv = "Date,Open,High,Low,Close,Volume\n2026-08-11,1,1,1,1,1\n"

    def fake_safe(tool, payload):
        if tool is data_collector.get_stock_data:
            return raw_csv
        if tool is data_collector.get_individual_fund_flow:
            return fund_flow
        if tool is data_collector._fetch_realtime_context:
            return {
                "status": "not_applicable",
                "source": None,
                "quote_as_of": None,
                "retrieved_at": None,
                "error": None,
                "quote": None,
            }
        return ""

    with patch.object(data_collector, "_safe", side_effect=fake_safe), \
         patch.object(data_collector, "FETCH_ALL_TIMEOUT", 1):
        result = data_collector._fetch_all("600519", "2026-08-11")

    context = result["market_data_context"]
    assert "manual_calibration_gap" not in context["source_provenance"]["fund_flow_individual"]
    assert not any(
        entry.get("gap_type") == "manual_calibration_gap"
        for entry in context["data_failure_ledger"]
    )


def test_fetch_all_completes_executor_with_fast_tools():
    with patch("tradingagents.graph.data_collector._safe", return_value=""), \
         patch("tradingagents.graph.data_collector.FETCH_ALL_TIMEOUT", 1):
        result = _fetch_all("600519", "2025-01-02")

    assert "stock_data" in result
    assert "market_data_context" in result


def test_empty_fund_flow_evidence_preserves_typed_metadata_and_ledger_reason():
    from tradingagents.graph import data_collector
    from tradingagents.dataflows.fund_flow_evidence import FundFlowText

    typed_reason = (
        "historical new-algorithm evidence unavailable; legacy Web reference unavailable；"
        "stock_individual_fund_flow: formatter reason: 接口调用失败"
    )
    typed_gap = f"【数据获取失败】资金流 evidence：{typed_reason}"
    fund_flow = FundFlowText(
        "【数据获取失败】历史日期 2026-08-11 新算法与新浪历史/legacy Web 资金流均不可用，"
        "600519 本项不可用。",
        evidence=[],
        evidence_meta={
            "symbol": "600519",
            "requested_as_of": "2026-08-11",
            "source": "fund_flow_individual",
            "source_family": "akshare",
            "algorithm_group": "new_algorithm_group",
            "unit": "亿元",
            "status": "unavailable",
            "reason": typed_reason,
            "gap": typed_gap,
            "attempted_sources": ["em", "sina_historical", "ths_instant_snapshot"],
            "fallback_errors": [
                "stock_individual_fund_flow: formatter reason: 接口调用失败",
                "sina historical fund flow: no current-day close row",
            ],
            "em_typed_gap": {"status": "unavailable", "reason": "formatter failure detail"},
            "final_source": "ths_instant_snapshot",
        },
    )

    def _fake_safe(tool, payload):
        if tool is data_collector.get_individual_fund_flow:
            return fund_flow
        return ""

    with (
        patch.object(data_collector, "_safe", side_effect=_fake_safe),
        patch.object(data_collector, "FETCH_ALL_TIMEOUT", 1),
    ):
        result = data_collector._fetch_all("600519", "2026-08-11")

    fund_flow_evidence = result["market_data_context"]["fund_flow_evidence"]
    assert fund_flow_evidence["status"] == "unavailable"
    assert fund_flow_evidence["reason"] == typed_reason
    assert fund_flow_evidence["gap"] == typed_gap
    assert fund_flow_evidence["source"] == "fund_flow_individual"
    assert fund_flow_evidence["requested_as_of"] == "2026-08-11"
    assert "未返回结构化逐日" not in fund_flow_evidence["reason"]
    assert fund_flow_evidence["attempted_sources"] == [
        "em", "sina_historical", "ths_instant_snapshot",
    ]
    assert fund_flow_evidence["fallback_errors"] == [
        "stock_individual_fund_flow: formatter reason: 接口调用失败",
        "sina historical fund flow: no current-day close row",
    ]
    assert fund_flow_evidence["em_typed_gap"] == {
        "status": "unavailable", "reason": "formatter failure detail",
    }
    assert fund_flow_evidence["final_source"] == "ths_instant_snapshot"
    assert fund_flow_evidence["records"] == []
    assert fund_flow_evidence["validation"] == {"status": "not_checked", "mismatches": []}

    ledger = result["market_data_context"]["data_failure_ledger"]
    ff_entries = [entry for entry in ledger if entry.get("source") == "fund_flow_individual"]
    assert len(ff_entries) == 1
    assert ff_entries[0]["status"] == "unavailable"
    assert ff_entries[0]["reason"] == typed_reason
    assert ff_entries[0]["gap"] == typed_gap
    assert "未返回结构化逐日" not in ff_entries[0]["reason"]
    assert ff_entries[0]["attempted_sources"] == [
        "em", "sina_historical", "ths_instant_snapshot",
    ]
    assert ff_entries[0]["fallback_errors"] == [
        "stock_individual_fund_flow: formatter reason: 接口调用失败",
        "sina historical fund flow: no current-day close row",
    ]
    assert ff_entries[0]["em_typed_gap"] == {
        "status": "unavailable", "reason": "formatter failure detail",
    }
    assert ff_entries[0]["final_source"] == "ths_instant_snapshot"


class _RecordingLLM:
    def __init__(self):
        self.messages = None

    async def astream(self, messages):
        self.messages = messages
        yield SimpleNamespace(content="固定分析输出")


def test_smart_money_no_collector_fallback_preserves_typed_evidence_metadata():
    import tradingagents.agents.utils.agent_utils as agent_utils
    from tradingagents.agents.analysts.smart_money_analyst import create_smart_money_analyst
    from tradingagents.dataflows.fund_flow_evidence import FundFlowText

    typed_reason = (
        "historical new-algorithm evidence unavailable; legacy Web reference unavailable；"
        "stock_individual_fund_flow: formatter reason: 接口调用失败"
    )
    typed_gap = f"【数据获取失败】资金流 evidence：{typed_reason}"
    fund_flow = FundFlowText(
        "【数据获取失败】历史日期 2026-08-10 新算法与新浪历史/legacy Web 资金流均不可用，"
        "600519 本项不可用。",
        evidence=[],
        evidence_meta={
            "symbol": "600519",
            "requested_as_of": "2026-08-10",
            "source": "fund_flow_individual",
            "status": "unavailable",
            "reason": typed_reason,
            "gap": typed_gap,
            "attempted_sources": ["em", "sina_historical", "ths_instant_snapshot"],
            "fallback_errors": ["stock_individual_fund_flow: formatter reason: 接口调用失败"],
            "em_typed_gap": {"status": "unavailable", "reason": "formatter failure detail"},
            "final_source": "ths_instant_snapshot",
        },
    )

    llm = _RecordingLLM()
    module = __import__(
        "tradingagents.agents.analysts.smart_money_analyst",
        fromlist=["smart_money_analyst"],
    )
    state = {
        "trade_date": "2026-08-10",
        "company_of_interest": "600519",
        "user_intent": {"focus_areas": [], "specific_questions": []},
    }

    fund_flow_tool = SimpleNamespace(invoke=lambda payload: fund_flow)
    lhb_tool = SimpleNamespace(invoke=lambda payload: "无龙虎榜数据")
    indicator_tool = SimpleNamespace(invoke=lambda payload: "100")

    with (
        patch.object(module, "get_cn_stock_name", return_value="测试股票"),
        patch.object(module, "get_config", return_value={}),
        patch.object(module, "get_prompt", return_value="固定系统提示"),
        patch.object(module, "build_horizon_context", return_value="固定上下文"),
        patch.object(module, "log_llm_call"),
        patch.object(agent_utils, "get_individual_fund_flow", fund_flow_tool),
        patch.object(agent_utils, "get_lhb_detail", lhb_tool),
        patch.object(agent_utils, "get_indicators", indicator_tool),
    ):
        result = asyncio.run(create_smart_money_analyst(llm, None)(state))

    assert "smart_money_report" in result
    human_prompt = llm.messages[1].content
    marker = "【资金流结构化 evidence（仅用于精确累计，不得从展示文本反推）】"
    evidence_block = human_prompt.split(marker, 1)[1]
    evidence_json = evidence_block.split("\n\n【新算法组共识规则】", 1)[0].strip()
    evidence = json.loads(evidence_json)

    assert evidence["status"] == "unavailable"
    assert evidence["reason"] == typed_reason
    assert evidence["gap"] == typed_gap
    assert "未返回结构化逐日" not in evidence["reason"]
    assert evidence["attempted_sources"] == ["em", "sina_historical", "ths_instant_snapshot"]
    assert evidence["fallback_errors"] == [
        "stock_individual_fund_flow: formatter reason: 接口调用失败",
    ]
    assert evidence["em_typed_gap"] == {
        "status": "unavailable", "reason": "formatter failure detail",
    }
    assert evidence["final_source"] == "ths_instant_snapshot"
