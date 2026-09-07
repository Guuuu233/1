"""Unit tests for fund flow scale metrics pure calculation (D-03-1 / C-09-3).

Covers:
1. Same unit recalculation and cross-unit mathematical consistency (万元/亿元/元).
2. Wrong unit rejection (explicit table: 万元/亿元/元 only).
3. Cross-date and cross-security mismatch rejection.
4. Zero or negative denominator rejection, preserving absolute net amount.
5. Missing amount unit: turnover ratio rejected while market cap ratio computed.
6. Missing market cap: turnover ratio computed without substituting for market cap.
7. No float default to 0.0 behavior.
8. Output contract fields compliance and absence of ranking terms.
9. DecimalRatio compatibility with Decimal and pytest.approx.
10. Fixture-only execution with zero network traffic.
"""

from decimal import Decimal
import socket
from unittest.mock import patch

import pytest

from tradingagents.dataflows.fund_flow_evidence import (
    DecimalRatio,
    SCALE_METRIC_VALID_UNITS,
    calculate_fund_flow_scale_metrics,
    compute_fund_flow_scale_metrics,
)


@pytest.fixture(autouse=True)
def guard_no_network_calls():
    """Ensure no real network calls can be made in this test suite."""
    with patch("socket.socket.connect", side_effect=RuntimeError("Network access forbidden in pure calculation tests")), \
         patch("requests.post", side_effect=RuntimeError("requests.post forbidden in pure calculation tests")), \
         patch("requests.get", side_effect=RuntimeError("requests.get forbidden in pure calculation tests")):
        yield


def test_same_unit_recalculation_and_cross_unit_consistency():
    """1. 同单位可复算：同一货币单位以及不同货币单位间显式换算相除，结果严格数学一致。"""
    # Case 1.1: 双方均为万元
    res_wan = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=100.0,
        net_amount_unit="万元",
        circ_mv=10000.0,
        circ_mv_unit="万元",
        amount=5000.0,
        amount_unit="万元",
        denominator_source="fixture_test",
    )
    assert res_wan["status"] == "available"
    assert res_wan["net_to_circ_mv"] == Decimal("0.01")
    assert res_wan["net_to_amount"] == Decimal("0.02")

    # Case 1.2: 双方均为亿元
    res_yi = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.0,
        net_amount_unit="亿元",
        circ_mv=100.0,
        circ_mv_unit="亿元",
        amount=50.0,
        amount_unit="亿元",
        denominator_source="fixture_test",
    )
    assert res_yi["status"] == "available"
    assert res_yi["net_to_circ_mv"] == Decimal("0.01")
    assert res_yi["net_to_amount"] == Decimal("0.02")

    # Case 1.3: 双方均为元
    res_yuan = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=100000000,
        net_amount_unit="元",
        circ_mv=10000000000,
        circ_mv_unit="元",
        amount=5000000000,
        amount_unit="元",
        denominator_source="fixture_test",
    )
    assert res_yuan["status"] == "available"
    assert res_yuan["net_to_circ_mv"] == Decimal("0.01")
    assert res_yuan["net_to_amount"] == Decimal("0.02")

    # Case 1.4: 跨单位换算：净额亿元（上游常见口径 1.5 亿），市值万元（Tushare 官方口径 2220600 万元）
    # 1.5 亿元 = 15000 万元，15000 / 2220600 = 1.5 / 222.06
    res_cross = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=2220600.0,
        circ_mv_unit="万元",
        amount=550000.0,
        amount_unit="万元",
        denominator_source="tushare.daily_basic",
    )
    assert res_cross["status"] == "available"
    expected_circ_ratio = Decimal("150000000") / Decimal("22206000000")
    expected_amt_ratio = Decimal("150000000") / Decimal("5500000000")
    assert res_cross["net_to_circ_mv"] == expected_circ_ratio
    assert res_cross["net_to_amount"] == expected_amt_ratio
    assert res_cross["net_to_circ_mv"] == pytest.approx(float(expected_circ_ratio))
    assert res_cross["net_to_amount"] == pytest.approx(float(expected_amt_ratio))

    # 验证与纯万元输入完全等价
    res_cross_wan = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=15000.0,
        net_amount_unit="万元",
        circ_mv=2220600.0,
        circ_mv_unit="万元",
        amount=550000.0,
        amount_unit="万元",
    )
    assert res_cross["net_to_circ_mv"] == res_cross_wan["net_to_circ_mv"]
    assert res_cross["net_to_amount"] == res_cross_wan["net_to_amount"]


def test_wrong_unit_rejected():
    """2. 错单位拒绝：不在显式换算表（万元/亿元/元）的单位必须被拒绝并记入缺口。"""
    # Case 2.1: 净额单位非法（千元、万亿、美元、股、None、空字符串）
    for bad_unit in ("千元", "万亿", "usd", "shares", "股", None, "", "   "):
        res = calculate_fund_flow_scale_metrics(
            ts_code="600519.SH",
            trade_date="2026-08-14",
            net_amount=1.5,
            net_amount_unit=bad_unit,
            circ_mv=2220600.0,
            circ_mv_unit="万元",
            amount=550000.0,
            amount_unit="万元",
        )
        assert res["net_to_circ_mv"] is None
        assert res["net_to_amount"] is None
        assert res["status"] == "unavailable"
        assert any("单位" in gap for gap in res["gaps"])

    # Case 2.2: 流通市值单位非法（千元、未知单位），成交额单位合法
    res_bad_circ_unit = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=2220600.0,
        circ_mv_unit="千元",
        amount=550000.0,
        amount_unit="万元",
    )
    assert res_bad_circ_unit["net_to_circ_mv"] is None
    assert res_bad_circ_unit["net_to_amount"] is not None
    assert any("流通市值单位 '千元'" in gap for gap in res_bad_circ_unit["gaps"])

    # Case 2.3: 成交额单位非法，流通市值单位合法
    res_bad_amt_unit = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=2220600.0,
        circ_mv_unit="万元",
        amount=550000.0,
        amount_unit="未知单位",
    )
    assert res_bad_amt_unit["net_to_circ_mv"] is not None
    assert res_bad_amt_unit["net_to_amount"] is None
    assert any("成交额单位 '未知单位'" in gap for gap in res_bad_amt_unit["gaps"])


def test_cross_date_and_cross_security_rejected():
    """3. 跨日/跨证券拒绝：同日同证券才可算，不一致必须拒绝相除。"""
    # Case 3.1: 跨日拒算（circ_mv 交易日与请求日不同）
    res_cross_date_circ = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=2220600.0,
        circ_mv_unit="万元",
        circ_mv_trade_date="2026-08-13",
        amount=550000.0,
        amount_unit="万元",
        amount_trade_date="2026-08-14",
    )
    assert res_cross_date_circ["net_to_circ_mv"] is None
    assert res_cross_date_circ["net_to_amount"] is not None
    assert any("跨交易日拒算" in gap and "circ_mv" in gap for gap in res_cross_date_circ["gaps"])

    # Case 3.2: 跨日拒算（amount 交易日不同）
    res_cross_date_amt = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=2220600.0,
        circ_mv_unit="万元",
        circ_mv_trade_date="2026-08-14",
        amount=550000.0,
        amount_unit="万元",
        amount_trade_date="2026-08-15",
    )
    assert res_cross_date_amt["net_to_circ_mv"] is not None
    assert res_cross_date_amt["net_to_amount"] is None
    assert any("跨交易日拒算" in gap and "amount" in gap for gap in res_cross_date_amt["gaps"])

    # Case 3.3: 跨证券拒算（circ_mv 股票代码不同）
    res_cross_sec_circ = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=2220600.0,
        circ_mv_unit="万元",
        circ_mv_ts_code="000001.SZ",
    )
    assert res_cross_sec_circ["net_to_circ_mv"] is None
    assert any("跨证券拒算" in gap and "circ_mv" in gap for gap in res_cross_sec_circ["gaps"])

    # Case 3.4: 跨证券拒算（amount 股票代码不同）
    res_cross_sec_amt = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        amount=550000.0,
        amount_unit="万元",
        amount_ts_code="000858.SZ",
    )
    assert res_cross_sec_amt["net_to_amount"] is None
    assert any("跨证券拒算" in gap and "amount" in gap for gap in res_cross_sec_amt["gaps"])

    # Case 3.5: 等价日期格式（YYYYMMDD 与 YYYY-MM-DD）应正确识别为同日同证券，正常计算
    res_date_norm = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=2220600.0,
        circ_mv_unit="万元",
        circ_mv_trade_date="20260814",
        circ_mv_ts_code="600519",
    )
    assert res_date_norm["net_to_circ_mv"] is not None
    assert not any("跨交易日" in gap for gap in res_date_norm["gaps"])
    assert not any("跨证券" in gap for gap in res_date_norm["gaps"])


def test_zero_and_negative_denominator_rejected():
    """4. 零分母拒算：分母为 0 或负数时比率拒算并记录缺口，但必须保留绝对净额。"""
    # Case 4.1: circ_mv 为 0
    res_zero_circ = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=0.0,
        circ_mv_unit="万元",
    )
    assert res_zero_circ["net_to_circ_mv"] is None
    assert res_zero_circ["net_amount"] == Decimal("1.5")
    assert res_zero_circ["net_amount_unit"] == "亿元"
    assert any("流通市值分母非正" in gap for gap in res_zero_circ["gaps"])

    # Case 4.2: circ_mv 为负数
    res_neg_circ = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=-500.0,
        circ_mv_unit="万元",
    )
    assert res_neg_circ["net_to_circ_mv"] is None
    assert res_neg_circ["net_amount"] == Decimal("1.5")
    assert any("流通市值分母非正" in gap for gap in res_neg_circ["gaps"])

    # Case 4.3: amount 为 0
    res_zero_amt = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        amount=0.0,
        amount_unit="万元",
    )
    assert res_zero_amt["net_to_amount"] is None
    assert res_zero_amt["net_amount"] == Decimal("1.5")
    assert any("成交额分母非正" in gap for gap in res_zero_amt["gaps"])

    # Case 4.4: amount 为负数
    res_neg_amt = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        amount=-1000.0,
        amount_unit="万元",
    )
    assert res_neg_amt["net_to_amount"] is None
    assert any("成交额分母非正" in gap for gap in res_neg_amt["gaps"])

    # Case 4.5: 净额为负（净流出），分母为正时，比率正常计算为负值
    res_outflow = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=-1.5,
        net_amount_unit="亿元",
        circ_mv=2220600.0,
        circ_mv_unit="万元",
        amount=550000.0,
        amount_unit="万元",
    )
    assert res_outflow["net_to_circ_mv"] is not None
    assert res_outflow["net_to_circ_mv"] < 0
    assert res_outflow["net_to_amount"] is not None
    assert res_outflow["net_to_amount"] < 0


def test_missing_amount_unit_turnover_ratio_rejected_circ_mv_computed():
    """5. 缺 amount 单位时成交额占比缺失但市值占比仍在（官方 daily_basic 未列出 amount 单位，不得脑补）。"""
    # 模拟 daily_basic 返回的结构（包含 circ_mv 与 amount，但 amount 单位未由官方定义）
    daily_basic_row = {
        "ts_code": "600519.SH",
        "trade_date": "20260814",
        "circ_mv": 2220600.0,
        "amount": 550000.0,
    }

    # 调用方显式传入 circ_mv_unit="万元"，但未传入 amount_unit
    res = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        daily_basic=daily_basic_row,
        circ_mv_unit="万元",
        # amount_unit 显式留空/缺失
    )

    # 市值占比仍在
    assert res["net_to_circ_mv"] is not None
    assert res["net_to_circ_mv"] == Decimal("150000000") / Decimal("22206000000")
    assert res["circ_mv_unit"] == "万元"
    assert res["circ_mv_source"] == "tushare.daily_basic"

    # 成交额占比拒算并记录缺口
    assert res["net_to_amount"] is None
    assert any("成交额单位 (amount_unit) 缺失" in gap for gap in res["gaps"])
    assert res["status"] == "partial"
    assert res["denominator_source"] == "tushare.daily_basic"


def test_missing_market_cap_turnover_ratio_still_computed():
    """6. 缺市值时允许只出成交额占比，二者不得当同一量纲互相替代。"""
    res = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=None,  # 缺市值
        amount=550000.0,
        amount_unit="万元",
        amount_source="vendor_quote",
    )

    # 成交额占比正常产出
    assert res["net_to_amount"] is not None
    assert res["net_to_amount"] == Decimal("150000000") / Decimal("5500000000")
    assert res["amount_unit"] == "万元"
    assert res["amount_source"] == "vendor_quote"

    # 市值占比缺失，记录缺口，绝不可用成交额占比回填市值占比
    assert res["net_to_circ_mv"] is None
    assert any("缺少流通市值 (circ_mv)" in gap for gap in res["gaps"])
    assert res["status"] == "partial"


def test_no_float_default_zero_behavior():
    """7. 禁止 float 默认填 0：分母缺失时绝不能默认填 0.0，必须保留 None 并记录缺口。"""
    res = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=None,
        amount=None,
    )
    assert res["circ_mv"] is None
    assert res["amount"] is None
    assert res["net_to_circ_mv"] is None
    assert res["net_to_amount"] is None
    assert res["status"] == "unavailable"
    assert any("流通市值" in gap for gap in res["gaps"])
    assert any("成交额" in gap for gap in res["gaps"])
    # 绝对净额保留
    assert res["net_amount"] == Decimal("1.5")


def test_output_contract_fields_and_no_ranking_vocabulary():
    """8. 契约输出具名字段验证，且严禁出现「主力强度排名」等跨股排名词汇。"""
    res = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=2220600.0,
        circ_mv_unit="万元",
        amount=550000.0,
        amount_unit="万元",
        denominator_source="tushare.daily_basic",
    )

    # 契约必出字段
    assert "net_to_circ_mv" in res
    assert "net_to_amount" in res
    assert "denominator_source" in res
    assert "unit" in res
    assert "gaps" in res
    assert isinstance(res["gaps"], list)
    assert "gap_list" in res

    # 严禁出现「主力强度排名」或跨股排序字段
    forbidden_tokens = ["主力强度排名", "强度排名", "rank", "ranking"]
    for key in res.keys():
        for token in forbidden_tokens:
            assert token not in key.lower()

    for gap in res["gaps"]:
        for token in forbidden_tokens:
            assert token not in gap.lower()


def test_decimal_ratio_approx_and_decimal_compatibility():
    """9. DecimalRatio 验证：既是真正的 Decimal，又可与 float/pytest.approx 无缝比较。"""
    r = DecimalRatio("0.00675493")
    assert isinstance(r, Decimal)
    assert r == Decimal("0.00675493")
    assert r == pytest.approx(0.00675493)
    assert r > 0.005
    assert r < 0.01
    assert float(r) == pytest.approx(0.00675493)


def test_public_alias_compute_fund_flow_scale_metrics():
    """10. 验证公共别名 compute_fund_flow_scale_metrics 与 calculate_fund_flow_scale_metrics 等价。"""
    assert compute_fund_flow_scale_metrics is calculate_fund_flow_scale_metrics


def test_evidence_mapping_as_first_argument():
    """11. 验证支持将 evidence 结构化 dict 作为首参传入。"""
    evidence_dict = {
        "symbol": "600519",
        "date": "2026-08-14",
        "selected_value": Decimal("1.5"),
        "selected_unit": "亿元",
    }
    res = calculate_fund_flow_scale_metrics(
        evidence_dict,
        circ_mv=2220600.0,
        circ_mv_unit="万元",
        denominator_source="tushare.daily_basic",
    )
    assert res["ts_code"] == "600519.SH"
    assert res["trade_date"] == "2026-08-14"
    assert res["net_amount"] == Decimal("1.5")
    assert res["net_to_circ_mv"] is not None


def test_mapping_zero_net_amount_not_swallowed_or_overwritten():
    """12. 验证 mapping 首参中 net_amount 为 0 或 Decimal('0') 时不被误判为缺失或被 selected_value 覆盖。"""
    # Case 12.1: net_amount 为 int 0，计算出 net_to_circ_mv == 0
    res_zero = calculate_fund_flow_scale_metrics(
        {
            "ts_code": "600519.SH",
            "trade_date": "2026-08-14",
            "net_amount": 0,
            "net_amount_unit": "亿元",
        },
        circ_mv=100.0,
        circ_mv_unit="亿元",
        denominator_source="fixture_test",
    )
    assert res_zero["net_amount"] == Decimal("0")
    assert res_zero["net_to_circ_mv"] == 0
    assert res_zero["net_to_circ_mv"] == Decimal("0")
    assert not any("资金净额 (net_amount) 缺失" in gap for gap in res_zero["gaps"])

    # Case 12.2: net_amount 为 Decimal("0")，计算出 net_to_circ_mv == 0
    res_dec_zero = calculate_fund_flow_scale_metrics(
        {
            "ts_code": "600519.SH",
            "trade_date": "2026-08-14",
            "net_amount": Decimal("0"),
            "net_amount_unit": "亿元",
        },
        circ_mv=100.0,
        circ_mv_unit="亿元",
        denominator_source="fixture_test",
    )
    assert res_dec_zero["net_amount"] == Decimal("0")
    assert res_dec_zero["net_to_circ_mv"] == 0
    assert res_dec_zero["net_to_circ_mv"] == Decimal("0")
    assert not any("资金净额 (net_amount) 缺失" in gap for gap in res_dec_zero["gaps"])

    # Case 12.3: net_amount=0 且存在非零 selected_value=50 时，仍使用 0，绝不能被覆盖成 50
    res_override_int = calculate_fund_flow_scale_metrics(
        {
            "ts_code": "600519.SH",
            "trade_date": "2026-08-14",
            "net_amount": 0,
            "selected_value": 50,
            "net_amount_unit": "亿元",
        },
        circ_mv=100.0,
        circ_mv_unit="亿元",
        denominator_source="fixture_test",
    )
    assert res_override_int["net_amount"] == Decimal("0")
    assert res_override_int["net_to_circ_mv"] == 0
    assert res_override_int["net_amount"] != 50

    # Case 12.4: net_amount=Decimal("0") 且存在非零 selected_value=50 时，仍使用 Decimal("0")，绝不能被覆盖成 50
    res_override_dec = calculate_fund_flow_scale_metrics(
        {
            "ts_code": "600519.SH",
            "trade_date": "2026-08-14",
            "net_amount": Decimal("0"),
            "selected_value": 50,
            "net_amount_unit": "亿元",
        },
        circ_mv=100.0,
        circ_mv_unit="亿元",
        denominator_source="fixture_test",
    )
    assert res_override_dec["net_amount"] == Decimal("0")
    assert res_override_dec["net_to_circ_mv"] == 0
    assert res_override_dec["net_amount"] != 50


def test_decimal_ratio_hashable():
    """13. 验证 DecimalRatio 显式实现 __hash__，支持哈希和集合操作，并与数值等价 float/Decimal 保持哈希一致。"""
    r = DecimalRatio("0.01")
    h = hash(r)
    assert isinstance(h, int)
    assert hash(r) == hash(Decimal("0.01"))
    assert hash(DecimalRatio("0.5")) == hash(0.5)

    s = {r, DecimalRatio("0.02")}
    assert DecimalRatio("0.01") in s
    assert Decimal("0.01") in s


def test_invalid_denominator_date_rejected():
    """14. 验证分母日期为非法字符串时（如无法解析为有效日期），必须拒算对应比率并记录缺口。"""
    res_circ = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=100.0,
        circ_mv_unit="亿元",
        circ_mv_trade_date="invalid-date",
        amount=50.0,
        amount_unit="亿元",
        amount_trade_date="2026-08-14",
    )
    assert res_circ["net_to_circ_mv"] is None
    assert res_circ["net_to_amount"] is not None
    assert any("分母交易日非法" in gap and "circ_mv" in gap for gap in res_circ["gaps"])

    res_amt = calculate_fund_flow_scale_metrics(
        ts_code="600519.SH",
        trade_date="2026-08-14",
        net_amount=1.5,
        net_amount_unit="亿元",
        circ_mv=100.0,
        circ_mv_unit="亿元",
        circ_mv_trade_date="2026-08-14",
        amount=50.0,
        amount_unit="亿元",
        amount_trade_date="not_a_date_2026",
    )
    assert res_amt["net_to_circ_mv"] is not None
    assert res_amt["net_to_amount"] is None
    assert any("分母交易日非法" in gap and "amount" in gap for gap in res_amt["gaps"])

