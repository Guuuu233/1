"""Tests for financial report period_kind classification and cutoff header integration.

D-009 / audit plan §P0-3a contract tests.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from tradingagents.dataflows.financial_announce import (
    FinancialPeriodKind,
    Q2DerivationResult,
    classify_financial_period_kind,
    derive_q2_from_h1_q1,
    financial_cutoff_header,
    format_q2_derivation_block,
    resolve_effective_announce_date,
)
import pandas as pd


# ── classify_financial_period_kind unit tests ────────────────────────


@pytest.mark.parametrize(
    "period, expected_kind, expected_label",
    [
        ("20240331", "first_quarter", "2024Q1"),
        ("20240630", "half_year_cumulative", "2024H1"),
        ("20240930", "nine_month_cumulative", "2024Q3"),
        ("20241231", "annual_cumulative", "2024A"),
    ],
)
def test_classify_income_periods(period, expected_kind, expected_label):
    res = classify_financial_period_kind(period, "income")
    assert isinstance(res, FinancialPeriodKind)
    assert res.period_kind == expected_kind
    assert res.reported_period_label == expected_label
    assert res.derivation_formula == "not_derived"
    assert "Q2" not in res.reported_period_label
    assert "Q2" not in res.period_kind


@pytest.mark.parametrize(
    "period, expected_kind, expected_label",
    [
        ("20240331", "first_quarter", "2024Q1"),
        ("20240630", "half_year_cumulative", "2024H1"),
        ("20240930", "nine_month_cumulative", "2024Q3"),
        ("20241231", "annual_cumulative", "2024A"),
    ],
)
def test_classify_cashflow_periods(period, expected_kind, expected_label):
    res = classify_financial_period_kind(period, "cashflow")
    assert isinstance(res, FinancialPeriodKind)
    assert res.period_kind == expected_kind
    assert res.reported_period_label == expected_label
    assert res.derivation_formula == "not_derived"
    assert "Q2" not in res.reported_period_label
    assert "Q2" not in res.period_kind


@pytest.mark.parametrize(
    "period, expected_label",
    [
        ("20240331", "2024Q1"),
        ("20240630", "2024H1"),
        ("20240930", "2024Q3"),
        ("20241231", "2024A"),
    ],
)
def test_classify_balance_periods(period, expected_label):
    res = classify_financial_period_kind(period, "balance")
    assert isinstance(res, FinancialPeriodKind)
    assert res.period_kind == "period_end_stock"
    assert res.reported_period_label == expected_label
    assert res.derivation_formula == "not_derived"
    assert "Q2" not in res.reported_period_label
    assert "cumulative" not in res.period_kind


def test_classify_invalid_inputs():
    res1 = classify_financial_period_kind("20240501", "income")
    assert res1.period_kind == "unknown"
    assert res1.derivation_formula == "not_derived"

    res2 = classify_financial_period_kind("20240630", "invalid_statement")
    assert res2.period_kind == "unknown"
    assert res2.derivation_formula == "not_derived"

    res3 = classify_financial_period_kind(None, "income")
    assert res3.period_kind == "unknown"
    assert res3.derivation_formula == "not_derived"

    res4 = classify_financial_period_kind("invalid_date", "income")
    assert res4.period_kind == "unknown"
    assert res4.derivation_formula == "not_derived"


# ── financial_cutoff_header tests with statement_kind ─────────────────


def test_financial_cutoff_header_preserves_legacy_when_statement_kind_none():
    eff = resolve_effective_announce_date("20240630", ["20240809"])
    header = financial_cutoff_header(eff, "2024-08-20")
    assert "财务数据截至 2024H1" in header
    assert "生效公告日 2024-08-09" in header
    assert "period_kind=" not in header


def test_financial_cutoff_header_with_statement_kind_income():
    eff = resolve_effective_announce_date("20240630", ["20240809"])
    header = financial_cutoff_header(eff, "2024-08-20", statement_kind="income")
    assert "财务数据截至 2024H1" in header
    assert "生效公告日 2024-08-09" in header
    assert "reported_period_label=2024H1" in header
    assert "period_kind=half_year_cumulative" in header
    assert "derivation_formula=not_derived" in header
    assert ("不是 Q2" in header or "不是Q2" in header or "禁止当作 Q2" in header or "禁止当作Q2" in header)
    assert "2024Q2" not in header


def test_financial_cutoff_header_with_statement_kind_balance():
    eff = resolve_effective_announce_date("20240630", ["20240809"])
    header = financial_cutoff_header(eff, "2024-08-20", statement_kind="balance")
    assert "财务数据截至 2024H1" in header
    assert "period_kind=period_end_stock" in header
    assert "reported_period_label=2024H1" in header
    assert "derivation_formula=not_derived" in header
    assert "期末点值" in header


def test_financial_cutoff_header_with_statement_kind_cashflow():
    eff = resolve_effective_announce_date("20240630", ["20240809"])
    header = financial_cutoff_header(eff, "2024-08-20", statement_kind="cashflow")
    assert "财务数据截至 2024H1" in header
    assert "period_kind=half_year_cumulative" in header
    assert "reported_period_label=2024H1" in header
    assert "derivation_formula=not_derived" in header
    assert ("不是 Q2" in header or "不是Q2" in header or "禁止当作 Q2" in header or "禁止当作Q2" in header)
    assert "2024Q2" not in header


def test_financial_cutoff_header_none_latest_with_statement_kind():
    header = financial_cutoff_header(None, "2024-08-20", statement_kind="income")
    assert header == "【财务数据】在 2024-08-20 及之前无已公开报告期"


# ── Q2 derivation unit tests (P0-3b) ─────────────────────────────────


def test_derive_q2_income_statement_success():
    """H1 net profit 250, Q1 net profit 100 -> Q2 derived 150."""
    df = pd.DataFrame(
        {
            "报告日": ["20240630", "20240331", "20231231"],
            "净利润": [250.0, 100.0, 400.0],
            "归属于母公司所有者的净利润": [200.0, 80.0, 350.0],
            "营业总收入": [1000.0, 400.0, 1800.0],
            "营业收入": [950.0, 380.0, 1700.0],
        }
    )
    res = derive_q2_from_h1_q1("income", df)
    assert isinstance(res, Q2DerivationResult)
    assert res.period_kind == "single_quarter_derived"
    assert res.reported_period_label == "2024Q2"
    assert res.derivation_formula == "H1-Q1"
    assert res.h1_period == "20240630"
    assert res.q1_period == "20240331"
    assert res.reason in ("ok", "")
    assert res.values["净利润"] == 150.0
    assert res.values["归属于母公司所有者的净利润"] == 120.0
    assert res.values["营业总收入"] == 600.0
    assert res.values["营业收入"] == 570.0
    assert "基本每股收益" not in res.values
    assert "稀释每股收益" not in res.values


def test_derive_q2_income_statement_excludes_eps():
    """EPS is non-additive and must not be subtracted or written to Q2 values."""
    df = pd.DataFrame(
        {
            "报告日": ["20240630", "20240331"],
            "净利润": [250.0, 100.0],
            "基本每股收益": [2.50, 1.00],
            "稀释每股收益": [2.40, 0.90],
        }
    )
    res = derive_q2_from_h1_q1("income", df)
    assert res.period_kind == "single_quarter_derived"
    assert res.reported_period_label == "2024Q2"
    assert res.derivation_formula == "H1-Q1"
    assert res.values["净利润"] == 150.0
    assert "基本每股收益" not in res.values
    assert "稀释每股收益" not in res.values


def test_derive_q2_defends_against_per_share_columns_even_if_in_whitelist(monkeypatch):
    """Defense: any column name containing '每股' must never be subtracted into values."""
    from tradingagents.dataflows import financial_announce

    monkeypatch.setattr(
        financial_announce,
        "INCOME_DERIVATION_WHITELIST",
        (*financial_announce.INCOME_DERIVATION_WHITELIST, "基本每股收益", "每股收益", "扣非每股收益"),
    )
    df = pd.DataFrame(
        {
            "报告日": ["20240630", "20240331"],
            "净利润": [250.0, 100.0],
            "基本每股收益": [2.50, 1.00],
            "每股收益": [2.50, 1.00],
            "扣非每股收益": [2.20, 0.90],
        }
    )
    res = derive_q2_from_h1_q1("income", df)
    assert res.period_kind == "single_quarter_derived"
    assert res.values["净利润"] == 150.0
    assert "基本每股收益" not in res.values
    assert "每股收益" not in res.values
    assert "扣非每股收益" not in res.values
    assert "基本每股收益" in res.missing
    assert "每股收益" in res.missing
    assert "扣非每股收益" in res.missing


def test_derive_q2_cashflow_statement_success_and_excludes_stock():
    """Cashflow flow fields derived; period-end stock balance excluded."""
    df = pd.DataFrame(
        {
            "报告日": ["20240630", "20240331"],
            "经营活动产生的现金流量净额": [500.0, 200.0],
            "现金及现金等价物净增加额": [100.0, 40.0],
            "期末现金及现金等价物余额": [1000.0, 900.0],
            "期初现金及现金等价物余额": [900.0, 860.0],
        }
    )
    res = derive_q2_from_h1_q1("cashflow", df)
    assert res.period_kind == "single_quarter_derived"
    assert res.reported_period_label == "2024Q2"
    assert res.derivation_formula == "H1-Q1"
    assert res.values["经营活动产生的现金流量净额"] == 300.0
    assert res.values["现金及现金等价物净增加额"] == 60.0
    # Must NOT include stock balances
    assert "期末现金及现金等价物余额" not in res.values
    assert "期初现金及现金等价物余额" not in res.values


def test_derive_q2_cashflow_sina_fixture_ignores_amount_columns_with_scope_words():
    """600873 cashflow rows must not treat an amount field containing ``单位`` as scope."""
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "financial_period_kind"
        / "600873_sh_financial_reports.json"
    )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    income_df = pd.DataFrame(payload["reports"]["income_statement"])
    df = pd.DataFrame(payload["reports"]["cashflow"])

    income_res = derive_q2_from_h1_q1("income", income_df)
    res = derive_q2_from_h1_q1("cashflow", df)

    assert income_res.period_kind == "single_quarter_derived"
    assert income_res.reason == "ok"
    assert income_res.values["净利润"] == pytest.approx(543941742.16)
    assert res.period_kind == "single_quarter_derived"
    assert res.reported_period_label == "2026Q2"
    assert res.reason == "ok"
    assert res.values["经营活动产生的现金流量净额"] == pytest.approx(1544697572.57)
    assert res.values["购建固定资产、无形资产和其他长期资产所支付的现金"] == pytest.approx(612034279.74)


def test_derive_q2_ignores_scope_words_inside_amount_columns():
    """Only dedicated metadata columns may trigger a scope mismatch."""
    df = pd.DataFrame(
        {
            "报告日": ["20240630", "20240331"],
            "处置子公司及其他营业单位收到的现金净额": [322188026.4, None],
            "按币种折算金额": [100.0, None],
            "经营活动产生的现金流量净额": [500.0, 200.0],
            "购建固定资产、无形资产和其他长期资产所支付的现金": [1000.0, 400.0],
        }
    )

    res = derive_q2_from_h1_q1("cashflow", df)

    assert res.period_kind == "single_quarter_derived"
    assert res.reason == "ok"
    assert res.values["经营活动产生的现金流量净额"] == 300.0
    assert res.values["购建固定资产、无形资产和其他长期资产所支付的现金"] == 600.0


def test_derive_q2_missing_q1():
    """When Q1 is missing from filtered df, returns missing_q1 without values."""
    df = pd.DataFrame(
        {
            "报告日": ["20240630", "20231231", "20230930"],
            "净利润": [250.0, 400.0, 300.0],
        }
    )
    res = derive_q2_from_h1_q1("income", df)
    assert res.period_kind == "unknown"
    assert res.reason == "missing_q1"
    assert len(res.values) == 0


def test_derive_q2_not_h1_latest():
    """When latest period is not 0630, derivation is skipped."""
    df = pd.DataFrame(
        {
            "报告日": ["20240930", "20240630", "20240331"],
            "净利润": [350.0, 250.0, 100.0],
        }
    )
    res = derive_q2_from_h1_q1("income", df)
    assert res.period_kind == "unknown"
    assert res.reason == "not_h1_latest"
    assert len(res.values) == 0


@pytest.mark.parametrize(
    "scope_col, h1_val, q1_val",
    [
        ("合并范围", "合并报表", "母公司报表"),
        ("币种", "CNY", "USD"),
        ("单位", "千元", "元"),
        ("会计口径", "新准则", "旧准则"),
        ("报表币种", "CNY", "USD"),
        ("报表单位", "千元", "元"),
        ("本期会计口径", "新准则", "旧准则"),
        ("合并范围", "合并", None),
    ],
)
def test_derive_q2_scope_mismatch(scope_col, h1_val, q1_val):
    """When scope / currency / unit columns differ or only one side present, refuse."""
    df = pd.DataFrame(
        {
            "报告日": ["20240630", "20240331"],
            scope_col: [h1_val, q1_val],
            "净利润": [250.0, 100.0],
        }
    )
    res = derive_q2_from_h1_q1("income", df)
    assert res.period_kind == "unknown"
    assert res.reason == "scope_mismatch"
    assert len(res.values) == 0


def test_derive_q2_percent_only():
    """When all candidate fields are YoY / QoQ / percentage, refuse with percent_only."""
    df = pd.DataFrame(
        {
            "报告日": ["20240630", "20240331"],
            "净利润同比增长率": [10.0, 8.0],
            "营业收入环比增长(%)": [5.0, 3.0],
        }
    )
    res = derive_q2_from_h1_q1("income", df)
    assert res.period_kind == "unknown"
    assert res.reason == "percent_only"
    assert len(res.values) == 0


def test_derive_q2_balance_sheet_refused():
    """Balance sheet is point-in-time stock and cannot derive single-quarter flow."""
    df = pd.DataFrame(
        {
            "报告日": ["20240630", "20240331"],
            "货币资金": [500.0, 300.0],
            "总资产": [2000.0, 1800.0],
        }
    )
    res = derive_q2_from_h1_q1("balance", df)
    assert res.period_kind == "unknown"
    assert res.reason == "balance_is_stock"
    assert len(res.values) == 0


def test_format_q2_derivation_block_success():
    """Successful derivation formats structured text with required machine-readable tokens."""
    res = Q2DerivationResult(
        reported_period_label="2024Q2",
        period_kind="single_quarter_derived",
        derivation_formula="H1-Q1",
        h1_period="20240630",
        q1_period="20240331",
        values={"净利润": 150.0, "归属于母公司所有者的净利润": 120.0},
        missing=("基本每股收益",),
        reason="ok",
    )
    block = format_q2_derivation_block(res)
    assert "period_kind=single_quarter_derived" in block
    assert "reported_period_label=2024Q2" in block
    assert "derivation_formula=H1-Q1" in block
    assert "净利润" in block and "150" in block
    assert "这是 H1 累计减 Q1 得到的 Q2 单季，不是报表原始 Q2 行" in block


def test_format_q2_derivation_block_failure():
    """Failed derivation formats explicit N/A notice with reason and prohibition."""
    res = Q2DerivationResult(
        reported_period_label="2024Q2",
        period_kind="unknown",
        derivation_formula="not_derived",
        h1_period="20240630",
        q1_period="20240331",
        values={},
        missing=(),
        reason="missing_q1",
    )
    block = format_q2_derivation_block(res)
    assert ("Q2_single_quarter=N/A" in block or "period_kind=unknown" in block)
    assert "reason=missing_q1" in block
    assert ("禁止把 H1 累计当作 Q2 单季" in block or "禁止把H1累计当作Q2单季" in block)
    assert "150" not in block  # no fabricated amounts
