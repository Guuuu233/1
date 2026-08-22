import pytest
from unittest.mock import MagicMock

from tradingagents.graph.report_quality_gate import (
    evaluate_macro_depth,
    evaluate_fundamentals_depth,
    evaluate_news_depth,
    evaluate_volume_price_depth,
    evaluate_role_depth,
    check_analyst_depth_quality,
    apply_report_quality_gate,
    check_report_quality,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph


class TestAnalystDepthQualityGate:
    """Comprehensive test suite for analyst depth quality gate contract (DAV-321)."""

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Macro Analyst Depth Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_macro_qualified_report_passes(self):
        text = (
            "【全球宏观与跨市场传导】\n"
            "标普500指数上涨0.8%，恒生指数上涨1.2%。海外流动性宽松通过风险偏好与汇率渠道"
            "向A股科技成长板块形成积极传导，带动国内半导体板块估值修复。"
            "宏观政策落地对产业端的传导时滞预计在1-2个月内体现。"
        )
        passed, score, failed_dims, reason = evaluate_macro_depth(text)
        assert passed is True
        assert score == 1.0
        assert failed_dims == []
        assert reason == ""

    def test_macro_empty_talk_fails(self):
        # Negative test 1: Empty talk slogan "宏观政策传导与市场联动值得关注"
        text = "宏观政策传导与市场联动值得关注。"
        passed, score, failed_dims, reason = evaluate_macro_depth(text)
        assert passed is False
        assert score < 1.0
        assert "entities_or_metrics" in failed_dims
        assert "causal_chain" in failed_dims
        assert "lag_or_missing" in failed_dims
        assert "macro深度不足" in reason

    def test_macro_piled_numbers_without_causal_chain_fails(self):
        # Negative test 2: Piled numbers without causal chain
        text = "标普500上涨0.8%，恒生指数上涨1.2%，纳斯达克上涨1.5%，上证指数3100点。"
        passed, score, failed_dims, reason = evaluate_macro_depth(text)
        assert passed is False
        assert "causal_chain" in failed_dims
        assert "lag_or_missing" in failed_dims

    def test_macro_causal_without_entities_or_metrics_fails(self):
        # Negative test 3: Causal chain without numbers/entities/missing marker
        text = (
            "海外央行货币政策变动直接导致国内市场流动性收紧，压制高估值成长板块估值表现，"
            "时滞大约为一个季度。"
        )
        passed, score, failed_dims, reason = evaluate_macro_depth(text)
        assert passed is False
        assert "entities_or_metrics" in failed_dims

    def test_macro_honest_missing_passes(self):
        # Positive test: Honest missing annotations
        text = (
            "【全球宏观】\n"
            "【数据缺失】全球核心市场指数与大类资产数据获取失败。"
            "海外政策对国内产业链的传导机制与时滞分析暂缺乏量化数据支持。"
        )
        passed, score, failed_dims, reason = evaluate_macro_depth(text)
        assert passed is True
        assert score == 1.0
        assert failed_dims == []

    def test_macro_arrow_style_transmission_passes(self):
        text = (
            "美联储降息25bp -> 美元指数回落至103.5点 -> 人民币汇率走强 -> 驱动外资流入A股核心资产，"
            "该传导效应预计存在2周左右的时滞窗口。"
        )
        passed, score, failed_dims, reason = evaluate_macro_depth(text)
        assert passed is True
        assert score == 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Fundamentals Analyst Depth Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_fundamentals_qualified_report_passes(self):
        text = (
            "【基本面深度分析】\n"
            "公司2026年上半年实现营业收入50亿元，同比+15%，净利润6.2亿元，毛利率32.5%。"
            "在光伏逆变器产业链中处于核心中游地位，具备较强的议价权与品牌壁垒。\n"
            "敏感性测算：上游IGBT器件成本上涨5%将导致公司综合毛利率承压下降约0.8个百分点；"
            "若海外出货量增长20%，可增厚净利润12%。"
        )
        passed, score, failed_dims, reason = evaluate_fundamentals_depth(text)
        assert passed is True
        assert score == 1.0
        assert failed_dims == []
        assert reason == ""

    def test_fundamentals_slogan_without_sensitivity_fails(self):
        # Negative: Financial numbers + chain but missing sensitivity
        text = (
            "公司2026年实现营收50亿元，净利润5亿元。在产业链中具备较高议价权。"
        )
        passed, score, failed_dims, reason = evaluate_fundamentals_depth(text)
        assert passed is False
        assert "sensitivity_relationship" in failed_dims
        assert "fundamentals深度不足" in reason

    def test_fundamentals_missing_industry_chain_fails(self):
        # Negative: Financial numbers + sensitivity but missing industry chain/pricing power
        text = (
            "公司2026年实现营业收入50亿元，净利润6亿元。成本上涨5%将导致净利润下降8%。"
        )
        passed, score, failed_dims, reason = evaluate_fundamentals_depth(text)
        assert passed is False
        assert "industry_chain_or_pricing_power" in failed_dims

    def test_fundamentals_missing_financial_numbers_fails(self):
        # Negative: Chain + sensitivity but missing financial numbers
        text = (
            "公司处于产业链核心地位，具备强大议价权。成本上涨将导致毛利率承压。"
        )
        passed, score, failed_dims, reason = evaluate_fundamentals_depth(text)
        assert passed is False
        assert "financial_metrics" in failed_dims

    def test_fundamentals_honest_missing_passes(self):
        # Positive: Honest missing annotation
        text = (
            "【基本面分析】\n"
            "【数据缺失】公司最新季度财报及敏感性测算数据未获取到。"
        )
        passed, score, failed_dims, reason = evaluate_fundamentals_depth(text)
        assert passed is True
        assert score == 1.0
        assert failed_dims == []

    # ─────────────────────────────────────────────────────────────────────────
    # 3. News Analyst Depth Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_news_qualified_report_passes(self):
        text = (
            "【重大新闻与产业事件分析】\n"
            "据证券时报报道，国家发改委等部门发布关于推动新型储能产业高质量发展的指导意见。\n"
            "直接影响：政策直接利好公司工商业储能核心业务订单放量；\n"
            "间接传导：通过产业链扩散至上游电芯供应商与同行竞对，促进行业集中度提升；\n"
            "时滞与验证：政策效应落地与业绩兑现的验证节点预计在第四季度财报体现。"
        )
        passed, score, failed_dims, reason = evaluate_news_depth(text)
        assert passed is True
        assert score == 1.0
        assert failed_dims == []
        assert reason == ""

    def test_news_missing_indirect_transmission_fails(self):
        # Negative: Has facts, direct impact, verification, but missing indirect chain
        text = (
            "据证券时报报道，国家发改委发布最新指引。该政策直接利好公司储能业务，"
            "业绩兑现的验证节点在第四季度。"
        )
        passed, score, failed_dims, reason = evaluate_news_depth(text)
        assert passed is False
        assert "indirect_transmission" in failed_dims

    def test_news_missing_direct_impact_fails(self):
        # Negative: Has facts, indirect, verification, but missing direct impact
        text = (
            "据新华社报道，行业监管政策发布。间接带动上下游产业链景气度提升，"
            "观察期预计持续至年底。"
        )
        passed, score, failed_dims, reason = evaluate_news_depth(text)
        assert passed is False
        assert "direct_impact" in failed_dims

    def test_news_missing_lag_verification_fails(self):
        # Negative: Has facts, direct, indirect, but missing lag/verification
        text = (
            "据财联社报道，公司与大客户签约合作。直接利好公司营收，同时通过供应链间接带动同行协同发展。"
        )
        passed, score, failed_dims, reason = evaluate_news_depth(text)
        assert passed is False
        assert "lag_or_verification" in failed_dims

    def test_news_honest_missing_passes(self):
        # Positive: Honest missing annotation
        text = "【新闻舆情】暂无重大新闻，【数据缺失】未检索到近期相关重大政策或公告。"
        passed, score, failed_dims, reason = evaluate_news_depth(text)
        assert passed is True
        assert score == 1.0
        assert failed_dims == []

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Volume Price Analyst Depth Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_volume_price_qualified_report_passes(self):
        text = (
            "【量价时空与技术形态深度分析】\n"
            "2026-08-21标的放量突破60日均线，收盘价28.50元，涨幅+5.2%，成交额12.8亿元，换手率4.5%。\n"
            "量价配合：呈现典型的量价齐升形态，放量突破前期平台阻力位；\n"
            "供需与阶段：筹码结构显示主力资金在底部蓄势完成后启动买盘主导，脱离震荡筑底阶段；\n"
            "后续验证：需关注后续3个交易日能否放量站稳28.00元防守位，若跌破则警惕假突破风险；\n"
            "跨维度印证：结合阶段一宏观流动性充裕及板块情绪高涨的上下文，量价突破有效性得到确认。"
        )
        passed, score, failed_dims, reason = evaluate_volume_price_depth(text)
        assert passed is True
        assert score == 1.0
        assert failed_dims == []
        assert reason == ""

    def test_volume_price_missing_phase1_cross_reference_fails(self):
        # Negative: Has metrics, anomaly, stage, verification, but completely ignores Phase 1 context without missing note
        text = (
            "2026-08-21标的收盘价28.50元(+5.2%)，成交额12.8亿元。呈现量价齐升与放量突破形态；"
            "筹码结构显示主力资金在底部蓄势后启动买盘主导；后续验证条件：需关注能否站稳28.00元防守位。"
        )
        passed, score, failed_dims, reason = evaluate_volume_price_depth(text)
        assert passed is False
        assert "cross_dimension_reference" in failed_dims

    def test_volume_price_with_explicit_no_context_passes(self):
        # Positive: Has metrics, anomaly, stage, verification, and explicitly notes missing Phase 1 context
        text = (
            "2026-08-21标的收盘价28.50元(+5.2%)，成交额12.8亿元。呈现量价齐升与放量突破形态；"
            "筹码结构显示主力资金在底部蓄势后启动买盘主导；后续验证条件：需关注能否站稳28.00元防守位。"
            "【数据缺失】无可用上下文：阶段一宏观与舆情报告缺失，本次分析独立基于K线量价数据展开。"
        )
        passed, score, failed_dims, reason = evaluate_volume_price_depth(text)
        assert passed is True
        assert score == 1.0
        assert failed_dims == []

    def test_volume_price_missing_verification_conditions_fails(self):
        # Negative: Missing verification conditions
        text = (
            "2026-08-21标的收盘价28.50元(+5.2%)，成交额12.8亿元。呈现量价齐升形态，主力资金在底部蓄势；"
            "结合阶段一宏观流动性充裕的上下文，量价形态偏多。"
        )
        passed, score, failed_dims, reason = evaluate_volume_price_depth(text)
        assert passed is False
        assert "verification_conditions" in failed_dims

    def test_volume_price_honest_missing_passes(self):
        text = "【量价分析】\n【数据缺失】行情K线及量价指标数据未获取到。"
        passed, score, failed_dims, reason = evaluate_volume_price_depth(text)
        assert passed is True
        assert score == 1.0
        assert failed_dims == []

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Ledger Writing & Idempotency & Graph Integration Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_evaluate_role_depth_dispatch(self):
        # Macro
        ok, score, _, _ = evaluate_role_depth("macro", "宏观政策传导与市场联动值得关注")
        assert ok is False
        # Unknown role passes
        ok_unk, score_unk, _, _ = evaluate_role_depth("unknown_role", "anything")
        assert ok_unk is True
        assert score_unk == 1.0

    def test_check_analyst_depth_quality_multi_role(self):
        reports = {
            "macro": "宏观政策传导与市场联动值得关注",
            "fundamentals": "基本面分析：营收增长，具备议价权。",
        }
        all_passed, results = check_analyst_depth_quality(reports)
        assert all_passed is False
        assert results["macro"]["passed"] is False
        assert results["fundamentals"]["passed"] is False

    def test_apply_quality_gate_structured_ledger_recording(self):
        state = {
            "macro_report": "宏观政策传导与市场联动值得关注",
            "fundamentals_report": "公司营收稳步增长，现金流充裕。",
            "news_report": "暂无重大新闻，【数据缺失】。",
            "volume_price_report": "2026-08-21收盘28元，【数据缺失】。",
            "market_data_context": {"data_failure_ledger": []},
            "data_gaps": [],
        }
        gate_passed = apply_report_quality_gate(state)
        assert gate_passed is False
        ledger = state["market_data_context"]["data_failure_ledger"]
        assert len(ledger) >= 1

        macro_entries = [e for e in ledger if e.get("role") == "macro"]
        assert len(macro_entries) == 1
        assert macro_entries[0]["source"] == "report_quality_gate"
        assert macro_entries[0]["status"] == "failed"
        assert isinstance(macro_entries[0]["score"], float)
        assert isinstance(macro_entries[0]["failed_dimensions"], list)
        assert "causal_chain" in macro_entries[0]["failed_dimensions"]
        assert "macro深度不足" in macro_entries[0]["reason"]

        fundamentals_entries = [e for e in ledger if e.get("role") == "fundamentals"]
        assert len(fundamentals_entries) == 1
        assert "sensitivity_relationship" in fundamentals_entries[0]["failed_dimensions"]

    def test_apply_quality_gate_idempotent_no_duplicates(self):
        state = {
            "macro_report": "宏观政策传导与市场联动值得关注",
            "market_data_context": {"data_failure_ledger": []},
            "data_gaps": [],
        }
        apply_report_quality_gate(state)
        first_len = len(state["market_data_context"]["data_failure_ledger"])
        first_gaps_len = len(state["data_gaps"])

        # Second run
        apply_report_quality_gate(state)
        assert len(state["market_data_context"]["data_failure_ledger"]) == first_len
        assert len(state["data_gaps"]) == first_gaps_len

    def test_trading_graph_integration_with_depth_quality_gate(self, tmp_path):
        graph = TradingAgentsGraph(
            selected_analysts=["macro", "fundamentals", "news", "volume_price"],
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
            "macro_report": "宏观政策传导与市场联动值得关注",  # Fails depth
            "fundamentals_report": "营收稳步增长。",  # Fails depth
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
        assert any(e.get("role") == "macro" for e in ledger)
        assert any(e.get("role") == "fundamentals" for e in ledger)
        assert signal == "HOLD"
