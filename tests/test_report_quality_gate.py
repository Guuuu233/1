import pytest
from unittest.mock import MagicMock

from tradingagents.graph.report_quality_gate import (
    check_report_keywords,
    check_global_indices_compliance,
    check_report_quality,
    apply_report_quality_gate,
    is_global_indices_failed_or_partial,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph


class TestReportQualityGate:
    """Test suite for report quality gate contract (§6)."""

    def test_qualified_macro_report_passes(self):
        text = (
            "【全球宏观与跨市场联动】\n"
            "美股标普500指数上涨0.8%，恒生指数上涨1.2%。"
            "海外科技股表现强势，通过情绪与产业链渠道向A股科技成长板块形成积极传导。"
            "跨市场联动效应明显，国内宏观流动性充裕。"
        )
        market_data_context = {
            "global_indices": "标普500: +0.8%, 恒生指数: +1.2%",
            "data_failure_ledger": [],
        }

        passed, reasons = check_report_quality(
            macro_report=text,
            market_data_context=market_data_context,
        )
        assert passed is True
        assert reasons == []

    def test_missing_transmission_keyword_records_failure(self):
        # Has 联动, but missing 传导
        text = (
            "【全球宏观与跨市场联动】\n"
            "美股标普500指数上涨0.8%，恒生指数上涨1.2%。"
            "海外科技板块与A股相关板块存在显著的联动效应。"
        )
        passed, reasons = check_report_quality(macro_report=text)
        assert passed is False
        assert any("传导" in r for r in reasons)

        # Apply to state
        state = {
            "macro_report": text,
            "market_data_context": {"data_failure_ledger": []},
        }
        gate_passed = apply_report_quality_gate(state)
        assert gate_passed is False
        ledger = state["market_data_context"]["data_failure_ledger"]
        assert len(ledger) >= 1
        assert ledger[0]["source"] == "report_quality_gate"
        assert ledger[0]["status"] == "failed"
        assert "传导" in ledger[0]["reason"]

    def test_missing_linkage_spillover_lag_keywords_records_failure(self):
        # Has 传导, but missing 联动 / 外溢 / 时滞
        text = (
            "【全球宏观】\n"
            "海外主要央行货币政策变动通过利率渠道对国内资产价格产生结构性传导。"
        )
        passed, reasons = check_report_quality(macro_report=text)
        assert passed is False
        assert any("联动" in r or "外溢" in r or "时滞" in r for r in reasons)

    def test_global_indices_failed_with_forbidden_smoothing_records_failure(self):
        # global_indices is failed, but report only writes "外围平稳" without index/missing markers
        text = (
            "【宏观环境与传导机制】\n"
            "近期外围平稳，海外宏观流动性对国内市场形成温和传导与联动。"
        )
        market_data_context = {
            "global_indices": "【数据获取失败】全球核心指数 — 原因：所有全球指数接口调用失败",
            "data_failure_ledger": [
                {
                    "source": "global_indices",
                    "status": "failed",
                    "reason": "所有全球指数接口调用失败",
                    "gap": "【数据获取失败】global_indices：所有全球指数接口调用失败",
                }
            ],
        }

        passed, reasons = check_report_quality(
            macro_report=text,
            market_data_context=market_data_context,
        )
        assert passed is False
        assert any("外围平稳" in r or "缺失" in r for r in reasons)

        state = {
            "macro_report": text,
            "market_data_context": market_data_context,
        }
        gate_passed = apply_report_quality_gate(state)
        assert gate_passed is False
        gate_entries = [
            e for e in state["market_data_context"]["data_failure_ledger"]
            if e.get("source") == "report_quality_gate"
        ]
        assert len(gate_entries) >= 1
        assert gate_entries[0]["status"] == "failed"

    def test_global_indices_failed_with_explicit_missing_annotation_passes(self):
        # global_indices is failed, but report explicitly notes 【数据缺失】
        text = (
            "【全球宏观与跨市场联动】\n"
            "【数据缺失】全球核心指数数据未获取到。"
            "海外政策对国内市场的传导存在时滞，需防范潜在外溢风险。"
        )
        market_data_context = {
            "global_indices": "无数据",
            "data_failure_ledger": [
                {
                    "source": "global_indices",
                    "status": "unavailable",
                    "reason": "无数据",
                }
            ],
        }

        passed, reasons = check_report_quality(
            macro_report=text,
            market_data_context=market_data_context,
        )
        assert passed is True
        assert reasons == []

    def test_is_global_indices_failed_or_partial_detection(self):
        assert is_global_indices_failed_or_partial(None) is False
        assert is_global_indices_failed_or_partial({}) is False

        # From data_failure_ledger
        ctx1 = {"data_failure_ledger": [{"source": "global_indices", "status": "failed"}]}
        assert is_global_indices_failed_or_partial(ctx1) is True

        # From source_provenance
        ctx2 = {"source_provenance": {"global_indices": {"status": "unavailable"}}}
        assert is_global_indices_failed_or_partial(ctx2) is True

        # From string failure marker
        ctx3 = {"global_indices": "【数据获取失败】接口调用超时"}
        assert is_global_indices_failed_or_partial(ctx3) is True

        # From "无数据"
        ctx4 = {"global_indices": "无数据"}
        assert is_global_indices_failed_or_partial(ctx4) is True

        # Normal valid data
        ctx5 = {"global_indices": "标普500: 5500 (+0.5%)", "data_failure_ledger": []}
        assert is_global_indices_failed_or_partial(ctx5) is False

    def test_fundamentals_fallback_when_macro_empty(self):
        text = (
            "基本面分析：产业链政策传导效应显著，行业上下游联动紧密，"
            "外溢效应带动公司营收增长。"
        )
        passed, reasons = check_report_quality(
            macro_report="",
            fundamentals_report=text,
        )
        assert passed is True
        assert reasons == []

    def test_idempotency_does_not_duplicate_ledger_entries(self):
        state = {
            "macro_report": "缺少关键字的报告正文",
            "market_data_context": {"data_failure_ledger": []},
        }
        apply_report_quality_gate(state)
        apply_report_quality_gate(state)
        ledger = state["market_data_context"]["data_failure_ledger"]
        gate_entries = [e for e in ledger if e.get("source") == "report_quality_gate"]
        assert len(gate_entries) == len(set(e["reason"] for e in gate_entries))

    def test_linkage_alternatives_pass(self):
        # Using 外溢 without 联动
        text_spillover = (
            "全球宏观传导分析：海外央行紧缩政策的外溢效应对新兴市场汇率构成压力。"
        )
        passed, reasons = check_report_quality(macro_report=text_spillover)
        assert passed is True
        assert reasons == []

        # Using 时滞 without 联动
        text_lag = (
            "宏观政策传导机制分析：降息政策对实体经济的提振作用存在一定的政策时滞。"
        )
        passed, reasons = check_report_quality(macro_report=text_lag)
        assert passed is True
        assert reasons == []

    def test_global_indices_failed_without_missing_or_indices_fails(self):
        # global_indices is failed, text has keywords but lacks any index or missing marker
        text = "宏观政策传导分析：国内流动性充裕，跨市场联动机制保持畅通。"
        ctx = {"global_indices": "failed", "data_failure_ledger": []}
        passed, reasons = check_report_quality(macro_report=text, market_data_context=ctx)
        assert passed is False
        assert any("正文必须出现【数据缺失】或全球核心指数/标普/恒生之一" in r for r in reasons)

    def test_global_indices_with_points_and_smoothing_passes(self):
        # global_indices is failed, but report explicitly references specific index points
        text = (
            "【全球宏观】虽然外围整体平稳，但标普500指数下跌0.5%，恒生指数上涨1.2%，"
            "外盘动能对国内成长板块形成传导与联动效应。"
        )
        ctx = {"global_indices": "【数据获取失败】", "data_failure_ledger": []}
        passed, reasons = check_report_quality(macro_report=text, market_data_context=ctx)
        assert passed is True
        assert reasons == []

    def test_trading_graph_propagate_invokes_quality_gate(self, tmp_path):
        graph = TradingAgentsGraph(
            selected_analysts=["macro"],
            config={
                "project_dir": str(tmp_path),
                "quick_think_llm": "mock",
                "deep_think_llm": "mock",
                "llm_provider": "openai",
                "api_key": "test-key",
            },
            data_collector=MagicMock(),
        )
        mock_final_state = {
            "company_of_interest": "000001",
            "trade_date": "2026-08-21",
            "macro_report": "美股平稳，具备一定联动效应。",
            "market_data_context": {"data_failure_ledger": []},
            "final_trade_decision": "HOLD",
        }
        graph.graph = MagicMock()
        graph.graph.invoke.return_value = mock_final_state
        graph.process_signal = MagicMock(return_value="HOLD")
        graph._log_state = MagicMock()
        graph.data_collector.collect.return_value = {"market_data_context": {"data_failure_ledger": []}}

        final_state, signal = graph.propagate("000001", "2026-08-21")
        ledger = final_state["market_data_context"]["data_failure_ledger"]
        assert any(e.get("source") == "report_quality_gate" for e in ledger)
        assert signal == "HOLD"
        graph = TradingAgentsGraph(
            selected_analysts=["macro"],
            config={
                "project_dir": str(tmp_path),
                "quick_think_llm": "mock",
                "deep_think_llm": "mock",
                "llm_provider": "openai",
                "api_key": "test-key",
            },
            data_collector=MagicMock(),
        )
        # Mock final_state with missing transmission keywords
        mock_final_state = {
            "company_of_interest": "000001",
            "trade_date": "2026-08-21",
            "macro_report": "美股表现良好，形成一定联动效应。",
            "market_data_context": {"data_failure_ledger": []},
            "final_trade_decision": "HOLD",
        }
        result = graph._build_horizon_result("short", mock_final_state)

        # Verify quality gate ran and recorded ledger entry into data_gaps without blocking
        assert "data_gaps" in result
        gate_gaps = [g for g in result["data_gaps"] if "report_quality_gate" in g]
        assert len(gate_gaps) >= 1
        assert "传导" in gate_gaps[0]
        # Completed horizon result is still returned normally
        assert result["horizon"] == "short"
        assert result["final_trade_decision"] == "HOLD"
