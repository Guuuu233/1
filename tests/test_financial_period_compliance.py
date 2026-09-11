"""Tests for financial period compliance verification (P-2, DAV-809 / DAV-819).

Pure deterministic compliance check for financial period semantics in
fundamentals_report against actual financial statement inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tradingagents.agents.utils.financial_period_compliance import (
    COMPLIANCE_STATUS_CHECKED_CLEAN,
    COMPLIANCE_STATUS_NOT_CHECKED,
    COMPLIANCE_STATUS_VIOLATIONS_FOUND,
    KIND_CUMULATIVE_LABELED_AS_SINGLE_QUARTER,
    KIND_CUMULATIVE_DOUBLE_COUNTED,
    KIND_SINGLE_QUARTER_WITHOUT_DERIVATION,
    RELATIVE_TOLERANCE,
    check_financial_period_compliance,
    extract_reported_amounts,
    parse_financial_statement_inputs,
)
from tradingagents.agents.utils.agent_states import TraceItem
from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "financial_period_compliance"


def _load_rt13_fixtures():
    report_file = FIXTURE_DIR / "report_6350b744_fundamentals_report.txt"
    report_text = report_file.read_text(encoding="utf-8")

    fuyao_file = FIXTURE_DIR / "600873_sh_fuyao_financial_reports.json"
    fuyao_data = json.loads(fuyao_file.read_text(encoding="utf-8"))
    return report_text, fuyao_data


# ── RT-5: 累计标为单季 ───────────────────────────────────────────────


def test_rt5_cumulative_labeled_as_single_quarter():
    """RT-5: 输入 H1 累计资本开支 1.13464e+09，报告写“2026Q2单季……11.35亿元”，
    命中 cumulative_labeled_as_single_quarter。
    """
    cashflow_md = """
| report_date | period_end | fiscal_period | reported_period_label | period_kind | pay_fixed_assets_etc_cash |
|:---|:---|:---|:---|:---|---:|
| 2026-08-15 | 2026-06-30 | Q2 | 2026H1 | half_year_cumulative | 1134640000 |
| 2026-04-22 | 2026-03-31 | Q1 | 2026Q1 | first_quarter | 522610000 |

【Q2单季派生数据（2026Q2，公式 H1-Q1）】
（reported_period_label=2026Q2, period_kind=single_quarter_derived, derivation_formula=H1-Q1；基准报告期: H1=20260630, Q1=20260331；派生金额: 购建固定资产、无形资产和其他长期资产所支付的现金=612030000；这是 H1 累计减 Q1 得到的 Q2 单季，不是报表原始 Q2 行）
"""
    financial_inputs = {
        "balance_sheet": "无数据",
        "income_statement": "无数据",
        "cashflow": cashflow_md,
        "fundamentals": "无数据",
    }
    report_text = "公司2026Q2单季“购建固定资产、无形资产和其他长期资产支付的现金”为11.35亿元。"

    res = check_financial_period_compliance(report_text, financial_inputs)
    assert res["status"] == COMPLIANCE_STATUS_VIOLATIONS_FOUND
    assert res["not_checked_reason"] is None
    assert len(res["violations"]) >= 1

    v = next(
        v for v in res["violations"] if v["kind"] == "cumulative_labeled_as_single_quarter"
    )
    assert "2026Q2" in v["reported_label"] or "单季" in v["reported_label"]
    assert v["input_period_kind"] == "half_year_cumulative"
    assert v["input_value"] == pytest.approx(1134640000.0)
    assert "2026H1" in v["expected_label"]
    assert v["statement"] == "cashflow"


# ── RT-6: 累计双计 ───────────────────────────────────────────────────


def test_rt6_cumulative_double_counted():
    """RT-6: 同一字段同时有 Q1 5.2261e+08 与 H1 1.13464e+09，
    报告写“2026H1累计资本开支达16.57亿元”，命中 cumulative_double_counted。
    """
    cashflow_md = """
| report_date | period_end | fiscal_period | reported_period_label | period_kind | pay_fixed_assets_etc_cash |
|:---|:---|:---|:---|:---|---:|
| 2026-08-15 | 2026-06-30 | Q2 | 2026H1 | half_year_cumulative | 1134640000 |
| 2026-04-22 | 2026-03-31 | Q1 | 2026Q1 | first_quarter | 522610000 |
"""
    financial_inputs = {
        "balance_sheet": "无数据",
        "income_statement": "无数据",
        "cashflow": cashflow_md,
        "fundamentals": "无数据",
    }
    report_text = "2026H1累计资本开支达16.57亿元，较去年同期维持刚性高位。"

    res = check_financial_period_compliance(report_text, financial_inputs)
    assert res["status"] == COMPLIANCE_STATUS_VIOLATIONS_FOUND
    v = next(v for v in res["violations"] if v["kind"] == "cumulative_double_counted")
    assert "H1" in v["reported_label"] or "累计" in v["reported_label"]
    assert v["input_period_kind"] == "half_year_cumulative"
    assert v["input_value"] == pytest.approx(1134640000.0)
    assert "16.57" in v["quoted_text"]


# ── RT-7: 正确用法不误报 ─────────────────────────────────────────────


def test_rt7_valid_cumulative_and_derived_not_flagged():
    """RT-7: 报告写“2026H1累计资本开支11.35亿元”，派生可用时写
    “2026Q2单季（H1−Q1）6.12亿元”，不产生违规。
    """
    cashflow_md = """
| report_date | period_end | fiscal_period | reported_period_label | period_kind | pay_fixed_assets_etc_cash |
|:---|:---|:---|:---|:---|---:|
| 2026-08-15 | 2026-06-30 | Q2 | 2026H1 | half_year_cumulative | 1134640000 |
| 2026-04-22 | 2026-03-31 | Q1 | 2026Q1 | first_quarter | 522610000 |

【Q2单季派生数据（2026Q2，公式 H1-Q1）】
（reported_period_label=2026Q2, period_kind=single_quarter_derived, derivation_formula=H1-Q1；基准报告期: H1=20260630, Q1=20260331；派生金额: 购建固定资产、无形资产和其他长期资产所支付的现金=612030000；这是 H1 累计减 Q1 得到的 Q2 单季，不是报表原始 Q2 行）
"""
    financial_inputs = {
        "balance_sheet": "无数据",
        "income_statement": "无数据",
        "cashflow": cashflow_md,
        "fundamentals": "无数据",
    }
    report_text = "公司2026H1累计资本开支11.35亿元；派生可用时，2026Q2单季（H1−Q1）资本开支为6.12亿元。"

    res = check_financial_period_compliance(report_text, financial_inputs)
    assert res["status"] == COMPLIANCE_STATUS_CHECKED_CLEAN
    assert res["not_checked_reason"] is None
    assert res["violations"] == []


# ── RT-8: 派生不可用却给单季 ─────────────────────────────────────────


def test_rt8_single_quarter_without_derivation_when_unavailable():
    """RT-8: Q2 派生明确不可用，报告出现有金额的“Q2单季”，
    命中 single_quarter_without_derivation。
    """
    cashflow_md = """
| report_date | period_end | fiscal_period | reported_period_label | period_kind | pay_fixed_assets_etc_cash |
|:---|:---|:---|:---|:---|---:|
| 2026-08-15 | 2026-06-30 | Q2 | 2026H1 | half_year_cumulative | 1134640000 |

【Q2单季派生数据（不可用）】
（Q2_single_quarter=N/A, period_kind=unknown, reason=missing_q1；原因: 同年度一季度财报未在当前分析日之前公开；禁止把 H1 累计当作 Q2 单季使用）
"""
    financial_inputs = {
        "balance_sheet": "无数据",
        "income_statement": "无数据",
        "cashflow": cashflow_md,
        "fundamentals": "无数据",
    }
    report_text = "2026Q2单季资本开支达6.12亿元。"

    res = check_financial_period_compliance(report_text, financial_inputs)
    assert res["status"] == COMPLIANCE_STATUS_VIOLATIONS_FOUND
    v = next(
        v for v in res["violations"] if v["kind"] == "single_quarter_without_derivation"
    )
    assert "Q2" in v["reported_label"] or "单季" in v["reported_label"]
    assert v["statement"] == "cashflow"


# ── RT-9: 0930 累计标为 Q3 单季与 1231 累计标为 Q4 单季 ──────────────


def test_rt9_q3_and_q4_cumulative_labeled_as_single_quarter():
    """RT-9: 0930 累计值写成 Q3 单季、1231 年度值写成 Q4 单季，均命中对应违规。"""
    income_md = """
| 报告日 | 营业收入 | 净利润 |
|:---|---:|---:|
| 20241231 | 4500000000 | 800000000 |
| 20240930 | 3050000000 | 550000000 |
| 20240630 | 2000000000 | 350000000 |
| 20240331 | 1000000000 | 180000000 |
"""
    financial_inputs = {
        "balance_sheet": "无数据",
        "income_statement": income_md,
        "cashflow": "无数据",
        "fundamentals": "无数据",
    }

    # Case A: Q3 cumulative written as Q3 single quarter
    report_text_q3 = "公司2024Q3单季营收为30.50亿元，净利润5.50亿元。"
    res_q3 = check_financial_period_compliance(report_text_q3, financial_inputs)
    assert res_q3["status"] == COMPLIANCE_STATUS_VIOLATIONS_FOUND
    v_q3 = next(
        v for v in res_q3["violations"] if v["kind"] == "cumulative_labeled_as_single_quarter"
    )
    assert v_q3["input_period_kind"] == "nine_month_cumulative"
    assert v_q3["input_value"] == pytest.approx(3050000000.0)
    assert "Q3" in v_q3["reported_label"]

    # Case B: Q4 annual cumulative written as Q4 single quarter
    report_text_q4 = "公司2024Q4单季营收为45.00亿元。"
    res_q4 = check_financial_period_compliance(report_text_q4, financial_inputs)
    assert res_q4["status"] == COMPLIANCE_STATUS_VIOLATIONS_FOUND
    v_q4 = next(
        v for v in res_q4["violations"] if v["kind"] == "cumulative_labeled_as_single_quarter"
    )
    assert v_q4["input_period_kind"] == "annual_cumulative"
    assert v_q4["input_value"] == pytest.approx(4500000000.0)
    assert "Q4" in v_q4["reported_label"]


# ── RT-10: 时点存量不误报 ────────────────────────────────────────────


def test_rt10_point_in_time_balance_stock_not_flagged():
    """RT-10: 资产负债表“总负债111.15亿元”“6月末”等时点存量表述不误报。"""
    balance_md = """
| report_date | period_end | fiscal_period | reported_period_label | period_kind | total_debt | assets_total |
|:---|:---|:---|:---|:---|---:|---:|
| 2026-08-15 | 2026-06-30 | Q2 | 2026H1 | period_end_stock | 11114500000 | 26785400000 |
"""
    financial_inputs = {
        "balance_sheet": balance_md,
        "income_statement": "无数据",
        "cashflow": "无数据",
        "fundamentals": "无数据",
    }
    report_text = "公司6月末资产负债表保持稳健，总负债111.15亿元，资产总额267.85亿元。"

    res = check_financial_period_compliance(report_text, financial_inputs)
    assert res["status"] == COMPLIANCE_STATUS_CHECKED_CLEAN
    assert res["violations"] == []


# ── RT-11: 单位、记数法与容差行为 ───────────────────────────────────


def test_rt11_unit_conversion_scientific_notation_and_tolerances():
    """RT-11: 输入使用 e 记数法或万元，报告使用亿元；数值匹配与容差行为有具体断言。"""
    # 1. Scientific notation e+09 input vs Chinese 亿元 in report
    cashflow_md_e = """
| report_date | period_end | fiscal_period | reported_period_label | period_kind | pay_fixed_assets_etc_cash |
|:---|:---|:---|:---|:---|---:|
| 2026-08-15 | 2026-06-30 | Q2 | 2026H1 | half_year_cumulative | 1.13464e+09 |
"""
    inputs_e = {
        "balance_sheet": "无数据",
        "income_statement": "无数据",
        "cashflow": cashflow_md_e,
        "fundamentals": "无数据",
    }
    report_e = "2026Q2单季资本开支11.35亿元。"
    res_e = check_financial_period_compliance(report_e, inputs_e)
    assert res_e["status"] == COMPLIANCE_STATUS_VIOLATIONS_FOUND
    assert res_e["violations"][0]["input_value"] == pytest.approx(1134640000.0)

    # 2. 万元 input vs 亿元 in report
    income_md_wan = """
| 报告日 | 营业收入（万元） | 净利润（万元） |
|:---|---:|---:|
| 20260630 | 1223509.53 | 66186.27 |
"""
    inputs_wan = {
        "balance_sheet": "无数据",
        "income_statement": income_md_wan,
        "cashflow": "无数据",
        "fundamentals": "无数据",
    }
    report_wan = "2026Q2单季营业收入122.35亿元。"
    res_wan = check_financial_period_compliance(report_wan, inputs_wan)
    assert res_wan["status"] == COMPLIANCE_STATUS_VIOLATIONS_FOUND

    # 3. Tolerance: within 1% matches, beyond tolerance does not match
    report_out_of_tolerance = "2026Q2单季资本开支15.00亿元。"
    res_out = check_financial_period_compliance(report_out_of_tolerance, inputs_e)
    assert not any(
        v["kind"] == "cumulative_labeled_as_single_quarter" for v in res_out["violations"]
    )


# ── RT-12: 无期间标签或新闻口径不误报 ────────────────────────────────


def test_rt12_no_period_or_news_attribution_not_misreported():
    """RT-12: 报告无期间标签或仅引用新闻“中报净利润6.62亿元”，不得误报。"""
    income_md = """
| 报告日 | 营业收入 | 净利润 |
|:---|---:|---:|
| 20260630 | 12235095339.54 | 661862706.23 |
| 20260331 | 5985630000 | 117920000 |
"""
    financial_inputs = {
        "balance_sheet": "无数据",
        "income_statement": income_md,
        "cashflow": "无数据",
        "fundamentals": "无数据",
    }

    # Case A: news reference with 中报
    report_news = "根据公司公告与新闻信息，中报净利润6.62亿元，符合市场此前预期。"
    res_news = check_financial_period_compliance(report_news, financial_inputs)
    assert res_news["status"] == COMPLIANCE_STATUS_CHECKED_CLEAN
    assert res_news["violations"] == []

    # Case B: general statement without period labels
    report_general = "公司持续推进研发投入与生产基地建设，综合毛利率保持在合理区间。"
    res_gen = check_financial_period_compliance(report_general, financial_inputs)
    assert res_gen["status"] == COMPLIANCE_STATUS_CHECKED_CLEAN
    assert res_gen["violations"] == []


# ── RT-13: 样本 6350b744 端到端校验 ──────────────────────────────────


def test_rt13_frozen_sample_6350b744_e2e_violations():
    """RT-13: 冻结 6350b744 基本面报告原文 + RT-1 输入，
    结果为 violations_found，至少包含 RT-5、RT-6 两条结构化违规。
    """
    report_text, fuyao_data = _load_rt13_fixtures()

    # Reconstruct Fuyao markdown tables from fixture
    from tradingagents.dataflows.providers.cn_fuyao_provider import CnFuyaoProvider

    provider = CnFuyaoProvider()
    cashflow_df = provider._annotate_financial_rows(
        fuyao_data["financials"]["cashflow"], "cashflow", "2026-09-10"
    )
    cashflow_table = provider._shrink_table(
        provider._sanitize_future_rows(cashflow_df), max_rows=12, max_cols=18, table_kind="generic"
    )
    cashflow_derivation = provider._q2_derivation_block(cashflow_df, "cashflow", "2026-09-10")
    cashflow_md = f"## 现金流量表\n\n{cashflow_table}\n\n{cashflow_derivation}"

    income_df = provider._annotate_financial_rows(
        fuyao_data["financials"]["income"], "income", "2026-09-10"
    )
    income_table = provider._shrink_table(
        provider._sanitize_future_rows(income_df), max_rows=12, max_cols=18, table_kind="generic"
    )
    income_derivation = provider._q2_derivation_block(income_df, "income", "2026-09-10")
    income_md = f"## 利润表\n\n{income_table}\n\n{income_derivation}"

    balance_df = provider._annotate_financial_rows(
        fuyao_data["financials"]["balance"], "balance", "2026-09-10"
    )
    balance_table = provider._shrink_table(
        provider._sanitize_future_rows(balance_df), max_rows=12, max_cols=18, table_kind="generic"
    )
    balance_md = f"## 资产负债表\n\n{balance_table}"

    financial_inputs = {
        "balance_sheet": balance_md,
        "income_statement": income_md,
        "cashflow": cashflow_md,
        "fundamentals": "无数据",
    }

    res = check_financial_period_compliance(report_text, financial_inputs)
    assert res["status"] == COMPLIANCE_STATUS_VIOLATIONS_FOUND
    kinds = [v["kind"] for v in res["violations"]]
    assert "cumulative_labeled_as_single_quarter" in kinds  # RT-5
    assert "cumulative_double_counted" in kinds  # RT-6


# ── RT-14: 入库路径与单/双期限序列化 ─────────────────────────────────


def test_rt14_storage_payload_and_dual_horizon_preservation():
    """RT-14: 通过 api.main._build_result_payload 的单期限结果，
    以及 _run_job_inner 的 short/medium 双期限组装，
    验证结构化对象存在于最终 result_data，字段完整且可 json.dumps。
    """
    from api.main import _build_result_payload

    compliance_obj = {
        "status": "violations_found",
        "not_checked_reason": None,
        "violations": [
            {
                "kind": "cumulative_labeled_as_single_quarter",
                "quoted_text": "公司2026Q2单季资本开支11.35亿元",
                "statement": "cashflow",
                "field": "购建固定资产、无形资产和其他长期资产所支付的现金",
                "reported_label": "2026Q2单季",
                "input_period_kind": "half_year_cumulative",
                "input_value": 1134640000.0,
                "expected_label": "2026H1",
            }
        ],
    }

    fund_trace = {
        "agent": "fundamentals_analyst",
        "horizon": "short",
        "research_horizon": "short",
        "observation_horizon": "medium",
        "data_window": "财报周期",
        "key_finding": "基本面分析结论：看空",
        "verdict": "看空",
        "confidence": "中",
        "financial_period_compliance": compliance_obj,
    }

    final_state_single = {
        "trade_date": "2026-09-10",
        "company_of_interest": "600873.SH",
        "analyst_traces": [fund_trace],
        "fundamentals_report": "测试报告内容",
    }

    # 1. Single horizon payload
    single_payload = _build_result_payload(final_state_single)
    assert "analyst_traces" in single_payload
    traces = single_payload["analyst_traces"]
    assert len(traces) == 1
    assert "financial_period_compliance" in traces[0]
    assert traces[0]["financial_period_compliance"]["status"] == "violations_found"

    # Verify JSON serializability
    dumped_single = json.dumps(single_payload)
    loaded_single = json.loads(dumped_single)
    assert loaded_single["analyst_traces"][0]["financial_period_compliance"]["status"] == "violations_found"

    # 2. Dual horizon assembly
    short_r = {"analyst_traces": [fund_trace]}
    medium_fund_trace = dict(fund_trace, horizon="medium", research_horizon="medium")
    medium_r = {"analyst_traces": [medium_fund_trace]}

    dual_traces = short_r.get("analyst_traces", []) + medium_r.get("analyst_traces", [])
    dual_result = {"analyst_traces": dual_traces}

    dumped_dual = json.dumps(dual_result)
    loaded_dual = json.loads(dumped_dual)
    assert len(loaded_dual["analyst_traces"]) == 2
    for t in loaded_dual["analyst_traces"]:
        assert t["financial_period_compliance"]["status"] == "violations_found"
        assert t["financial_period_compliance"]["violations"][0]["kind"] == "cumulative_labeled_as_single_quarter"


# ── RT-15: 干净与未检查区分 ──────────────────────────────────────────


def test_rt15_distinguish_clean_from_not_checked():
    """RT-15: 完整且正常的输入/报告为 checked_clean；
    失败标记或不可解析表格为 not_checked，不得混淆。
    """
    normal_income_md = """
| 报告日 | 营业收入 | 净利润 |
|:---|---:|---:|
| 20260630 | 12235095339.54 | 661862706.23 |
"""
    clean_inputs = {
        "balance_sheet": "无数据",
        "income_statement": normal_income_md,
        "cashflow": "无数据",
        "fundamentals": "无数据",
    }
    clean_report = "2026H1营业收入122.35亿元，净利润6.62亿元。"

    # (a) Normal clean
    res_clean = check_financial_period_compliance(clean_report, clean_inputs)
    assert res_clean["status"] == COMPLIANCE_STATUS_CHECKED_CLEAN
    assert res_clean["not_checked_reason"] is None
    assert res_clean["violations"] == []

    # (b) Failure markers: 【数据获取失败】
    failed_inputs_1 = {
        "balance_sheet": "【数据获取失败】资产负债表 接口超时",
        "income_statement": "【数据获取失败】利润表",
        "cashflow": "【数据获取失败】现金流量表",
        "fundamentals": "无数据",
    }
    res_failed_1 = check_financial_period_compliance(clean_report, failed_inputs_1)
    assert res_failed_1["status"] == COMPLIANCE_STATUS_NOT_CHECKED
    assert res_failed_1["not_checked_reason"] is not None
    assert "失败" in res_failed_1["not_checked_reason"]
    assert res_failed_1["violations"] == []

    # (c) Unparseable table
    unparseable_inputs = {
        "balance_sheet": "random text without any table",
        "income_statement": "invalid content",
        "cashflow": "corrupted data",
        "fundamentals": "无数据",
    }
    res_unparseable = check_financial_period_compliance(clean_report, unparseable_inputs)
    assert res_unparseable["status"] == COMPLIANCE_STATUS_NOT_CHECKED
    assert res_unparseable["not_checked_reason"] is not None
    assert res_unparseable["violations"] == []


# ── RT-16: 不阻断（只记录违规，不改分析状态） ─────────────────────────


def test_rt16_non_blocking_does_not_alter_analysis_status():
    """RT-16: 存在违规的正常运行仍保持原 analysis_status、trade_action 不变，
    违规仅被记录于 analyst_traces。
    """
    import asyncio

    mock_llm = MagicMock()
    mock_llm.model_name = "test-model"
    # LLM outputs a report with RT-5 violation
    violation_report = (
        "公司2026Q2单季“购建固定资产、无形资产和其他长期资产支付的现金”为11.35亿元。\n\n"
        '<!-- VERDICT: {"direction": "看空"} -->'
    )

    async def _mock_astream(messages):
        yield MagicMock(content=violation_report)

    mock_llm.astream = _mock_astream

    cashflow_md = """
| report_date | period_end | fiscal_period | reported_period_label | period_kind | pay_fixed_assets_etc_cash |
|:---|:---|:---|:---|:---|---:|
| 2026-08-15 | 2026-06-30 | Q2 | 2026H1 | half_year_cumulative | 1134640000 |
| 2026-04-22 | 2026-03-31 | Q1 | 2026Q1 | first_quarter | 522610000 |
"""
    collector = MagicMock()
    collector.get.return_value = {
        "fundamentals": "无数据",
        "balance_sheet": "无数据",
        "cashflow": cashflow_md,
        "income_statement": "无数据",
    }

    state = {
        "trade_date": "2026-09-10",
        "company_of_interest": "600873.SH",
        "analysis_status": "NORMAL",
        "trade_action": "BUY",
        "decision_status": "PENDING",
    }

    node = create_fundamentals_analyst(mock_llm, collector)
    result = asyncio.run(node(state))

    # 1. fundamentals_report text is NOT modified
    assert result["fundamentals_report"] == violation_report

    # 2. TraceItem contains financial_period_compliance
    assert "analyst_traces" in result
    trace = result["analyst_traces"][0]
    assert "financial_period_compliance" in trace
    compliance = trace["financial_period_compliance"]
    assert compliance["status"] == COMPLIANCE_STATUS_VIOLATIONS_FOUND
    assert len(compliance["violations"]) >= 1

    # 3. State fields analysis_status / trade_action are untouched
    assert "analysis_status" not in result
    assert "trade_action" not in result
    assert "decision_status" not in result


# ── Schema 严格性与原生 JSON 类型测试 ────────────────────────────────────


def test_violations_schema_strictness_and_native_json_types():
    """验证所有违规条目严格包含 8 个必选字段，且 input_value 为 JSON 原生类型。"""
    cashflow_md = """
| report_date | period_end | fiscal_period | reported_period_label | period_kind | pay_fixed_assets_etc_cash |
|:---|:---|:---|:---|:---|---:|
| 2026-08-15 | 2026-06-30 | Q2 | 2026H1 | half_year_cumulative | 1134640000 |
| 2026-04-22 | 2026-03-31 | Q1 | 2026Q1 | first_quarter | 522610000 |
"""
    financial_inputs = {
        "balance_sheet": "无数据",
        "income_statement": "无数据",
        "cashflow": cashflow_md,
        "fundamentals": "无数据",
    }
    report_text = (
        "公司2026Q2单季资本开支为11.35亿元，2026H1累计资本开支达16.57亿元。"
    )

    res = check_financial_period_compliance(report_text, financial_inputs)
    assert res["status"] == COMPLIANCE_STATUS_VIOLATIONS_FOUND
    assert len(res["violations"]) >= 2

    required_keys = {
        "kind",
        "quoted_text",
        "statement",
        "field",
        "reported_label",
        "input_period_kind",
        "input_value",
        "expected_label",
    }

    for v in res["violations"]:
        assert required_keys.issubset(v.keys()), f"Missing keys in violation: {v}"
        # Assert type is strictly JSON native (int or float)
        assert type(v["input_value"]) in (int, float), f"input_value must be int/float, got {type(v['input_value'])}"
        assert type(v["kind"]) is str
        assert type(v["quoted_text"]) is str
        assert type(v["statement"]) is str
        assert type(v["field"]) is str
        assert type(v["reported_label"]) is str
        assert type(v["input_period_kind"]) is str
        assert type(v["expected_label"]) is str

    # Direct json.dumps test
    dumped = json.dumps(res)
    loaded = json.loads(dumped)
    assert loaded["status"] == COMPLIANCE_STATUS_VIOLATIONS_FOUND


# ── AkShare 降级表格格式支持测试 ─────────────────────────────────────────


def test_akshare_format_support_and_classification():
    """测试 AkShare 降级链路格式的表格输入（使用 报告日 与 中文列名）。"""
    cashflow_akshare_md = """
【财务数据截至 2026H1（生效公告日 2026-08-15）】（reported_period_label=2026H1, period_kind=half_year_cumulative, derivation_formula=not_derived；注意：半年度报告包含 1-6 月累计金额，不是 Q2 单季度数据，禁止当作 Q2 单季使用）

| 报告日 | 经营活动产生的现金流量净额 | 购建固定资产、无形资产和其他长期资产所支付的现金 |
|:---|---:|---:|
| 20260630 | 389140000 | 1134640000 |
| 20260331 | -1155560000 | 522610000 |

【Q2单季派生数据（2026Q2，公式 H1-Q1）】
（reported_period_label=2026Q2, period_kind=single_quarter_derived, derivation_formula=H1-Q1；基准报告期: H1=20260630, Q1=20260331；派生金额: 购建固定资产、无形资产和其他长期资产所支付的现金=612030000；这是 H1 累计减 Q1 得到的 Q2 单季，不是报表原始 Q2 行）
"""
    financial_inputs = {
        "balance_sheet": "无数据",
        "income_statement": "无数据",
        "cashflow": cashflow_akshare_md,
        "fundamentals": "无数据",
    }

    # 1. Valid derivation usage -> clean
    clean_report = "2026Q2单季（H1-Q1）资本开支为6.12亿元。"
    res_clean = check_financial_period_compliance(clean_report, financial_inputs)
    assert res_clean["status"] == COMPLIANCE_STATUS_CHECKED_CLEAN
    assert res_clean["violations"] == []

    # 2. Mislabeling cumulative as Q2 -> violation
    violation_report = "2026Q2单季购建固定资产、无形资产和其他长期资产所支付的现金达11.35亿元。"
    res_vio = check_financial_period_compliance(violation_report, financial_inputs)
    assert res_vio["status"] == COMPLIANCE_STATUS_VIOLATIONS_FOUND
    assert res_vio["violations"][0]["kind"] == KIND_CUMULATIVE_LABELED_AS_SINGLE_QUARTER


# ── 金额抽取与单位/符号过滤测试 ──────────────────────────────────────────


def test_amount_extraction_units_and_exclusions():
    """测试金额抽取函数对中文单位、负数、百分比/日期排除的正确性。"""
    sample = "2026Q2单季自由现金流为-7.46亿元，毛利率12.93%，PE约15倍，投资达5000万元，基准日2026-09-10。"
    amounts = extract_reported_amounts(sample)
    amount_vals = [a[0] for a in amounts]

    # -7.46亿元 -> -746000000.0
    assert pytest.approx(-7.46e8) in amount_vals
    # 5000万元 -> 50000000.0
    assert pytest.approx(50000000.0) in amount_vals

    # Exclusions: 12.93%, 15倍, 2026-09-10 must NOT appear as monetary values
    assert 12.93 not in amount_vals
    assert 15.0 not in amount_vals
    assert 2026.0 not in amount_vals


# ── 边界与异常入参鲁棒性测试 ─────────────────────────────────────────────


def test_graceful_handling_of_edge_cases():
    """测试异常入参（None、空字典、空字符串、非字典结构）的防御性处理。"""
    # None inputs
    res1 = check_financial_period_compliance("报告内容", None)
    assert res1["status"] == COMPLIANCE_STATUS_NOT_CHECKED
    assert res1["not_checked_reason"] is not None

    # Empty dict inputs
    res2 = check_financial_period_compliance("报告内容", {})
    assert res2["status"] == COMPLIANCE_STATUS_NOT_CHECKED

    # Empty report with valid inputs
    cashflow_md = """
| report_date | period_end | fiscal_period | reported_period_label | period_kind | pay_fixed_assets_etc_cash |
|:---|:---|:---|:---|:---|---:|
| 2026-08-15 | 2026-06-30 | Q2 | 2026H1 | half_year_cumulative | 1134640000 |
"""
    valid_inputs = {
        "balance_sheet": "无数据",
        "income_statement": "无数据",
        "cashflow": cashflow_md,
        "fundamentals": "无数据",
    }
    res3 = check_financial_period_compliance("", valid_inputs)
    assert res3["status"] == COMPLIANCE_STATUS_CHECKED_CLEAN
    assert res3["violations"] == []

    # Report with pure symbols
    res4 = check_financial_period_compliance("---...###!!!", valid_inputs)
    assert res4["status"] == COMPLIANCE_STATUS_CHECKED_CLEAN
    assert res4["violations"] == []