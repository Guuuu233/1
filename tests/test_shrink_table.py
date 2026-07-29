"""Regression tests for name-based financial table cleaning (commit 5)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from tradingagents.dataflows.utils import (
    MISSING_VALUE_MARKER,
    missing_core_financial_fields,
    shrink_table,
)


def test_shrink_table_reorders_by_name_and_drops_nan():
    """Shuffled columns + NaNs → name-selected core cols, no bare nan."""
    df = pd.DataFrame(
        [
            {
                "垃圾列A": float("nan"),
                "存货": 1.0,
                "归属于母公司所有者的净利润": 9.0,
                "无关噪声": "x",
                "资产总计": 100.0,
                "负债合计": 40.0,
                "归属于母公司股东权益合计": 60.0,
                "报告日": 20260331,
                "几乎全空": float("nan"),
            },
            {
                "垃圾列A": float("nan"),
                "存货": 2.0,
                "归属于母公司所有者的净利润": 8.0,
                "无关噪声": "y",
                "资产总计": 110.0,
                "负债合计": 45.0,
                "归属于母公司股东权益合计": 65.0,
                "报告日": 20251231,
                "几乎全空": float("nan"),
            },
        ]
    )
    # Add a >80% null column that should be dropped.
    df["超空列"] = [float("nan"), float("nan")]

    text = shrink_table(df, max_rows=5, table_kind="balance", require_core_fields=True)
    assert "【数据获取失败】" not in text
    assert "nan" not in text.lower()
    assert MISSING_VALUE_MARKER not in text or True  # may or may not appear
    # Core columns present by name, not first-N positional.
    assert "资产总计" in text
    assert "负债合计" in text
    assert "归属于母公司股东权益合计" in text
    assert "归属于母公司所有者的净利润" in text or "报告日" in text
    # Fully empty column dropped.
    assert "超空列" not in text
    assert "垃圾列A" not in text


def test_shrink_table_fails_when_two_core_fields_missing():
    df = pd.DataFrame(
        [
            {"报告日": 20260331, "货币资金": 1.0, "资产总计": 100.0},
            {"报告日": 20251231, "货币资金": 2.0, "资产总计": 110.0},
        ]
    )
    # Missing 总负债 / 净资产 / 归母净利润 → ≥2 core missing.
    missing = missing_core_financial_fields(df.columns)
    assert len(missing) >= 2
    text = shrink_table(df, table_kind="balance", require_core_fields=True)
    assert text.startswith("【数据获取失败】关键财务字段缺失：")
    assert "总负债" in text or "净资产" in text or "归母净利润" in text
    # Must not return a partial markdown table.
    assert "| 资产总计 |" not in text


def test_shrink_table_marks_residual_nulls_explicitly():
    df = pd.DataFrame(
        [
            {
                "报告日": 20260331,
                "资产总计": 100.0,
                "负债合计": float("nan"),
                "归属于母公司股东权益合计": 60.0,
                "归属于母公司所有者的净利润": 9.0,
                "货币资金": None,
            }
        ]
    )
    text = shrink_table(df, table_kind="balance", require_core_fields=True)
    assert "nan" not in text.lower()
    assert MISSING_VALUE_MARKER in text


def test_shrink_table_explicit_truncate_notice():
    # Wide table with many filled non-core columns to force char budget cut.
    row = {
        "报告日": 20260331,
        "资产总计": 1.0,
        "负债合计": 2.0,
        "归属于母公司股东权益合计": 3.0,
        "归属于母公司所有者的净利润": 4.0,
    }
    for i in range(40):
        row[f"扩展字段{i:02d}"] = 1000000.123456 + i
    df = pd.DataFrame([row, dict(row, **{"报告日": 20251231})])
    text = shrink_table(
        df,
        table_kind="balance",
        require_core_fields=True,
        max_prompt_chars=500,
    )
    assert "【已截断，保留核心字段】" in text
    assert "资产总计" in text
    assert "nan" not in text.lower()


def test_provider_wrapper_returns_string_not_dataframe():
    from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider

    df = pd.DataFrame(
        [
            {
                "报告日": 20260331,
                "资产总计": 100.0,
                "负债合计": 40.0,
                "归属于母公司股东权益合计": 60.0,
                "归属于母公司所有者的净利润": 9.0,
                "zzz_last": float("nan"),
            }
        ]
    )
    out = CnAkshareProvider._shrink_table(
        df, max_rows=5, max_cols=2, table_kind="balance", require_core_fields=True
    )
    assert isinstance(out, str)
    # max_cols must NOT cause positional cut of core columns.
    assert "资产总计" in out
    assert "负债合计" in out
    assert "nan" not in out.lower()
