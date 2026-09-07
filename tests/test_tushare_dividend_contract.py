"""Unit tests for Tushare Pro dividend (分红送转) reader interface contract (D-01 / C-04-2 / DAV-703).

Covers:
- Normal multi-row fetching and column access by name (no iloc).
- 12 required fields coverage (ts_code, end_date, ann_date, div_proc, stk_div, stk_bo_rate,
  cash_div, cash_div_tax, record_date, ex_date, pay_date, imp_ann_date).
- Empty table explicit error reporting (no_rows, never interpreted as "no dividend").
- PIT boundary guard: rows with ann_date > as_of are discarded as lookahead rows and recorded as gaps.
- Future implementation dates masked when imp_ann_date > as_of.
- Token missing handling without network traffic (token_missing).
- Missing required fields reporting (missing_field).
- Typed error classifications: permission_denied, rate_limited, json_shape, transport.
- URL routing via TUSHARE_API_URL / TUSHARE_BASE_URL with official default fallback.
"""

from unittest.mock import patch

import pytest
import requests

from tradingagents.dataflows.providers.cn_akshare_provider import (
    _TUSHARE_DIVIDEND_API,
    _TUSHARE_DIVIDEND_REQUIRED_FIELDS,
    CnAkshareProvider,
)


class _MockResponse:
    def __init__(self, data: dict, status_code: int = 200, text: str = ""):
        self._data = data
        self.status_code = status_code
        self.text = text

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def _make_dividend_payload(
    *,
    code: int = 0,
    msg: str = "",
    rows: list[dict] | None = None,
    fields: list[str] | None = None,
    empty_items: bool = False,
    data: dict | None = None,
):
    if data is not None:
        return {"code": code, "msg": msg, "data": data}

    if fields is None:
        fields = list(_TUSHARE_DIVIDEND_REQUIRED_FIELDS)

    if empty_items or rows is None:
        items = []
    else:
        items = [[row.get(f) for f in fields] for row in rows]

    return {
        "code": code,
        "msg": msg,
        "data": {
            "fields": fields,
            "items": items,
        },
    }


def test_dividend_success_records(monkeypatch):
    """1. 正常返回：验证返回列表按列名取数，包含所有 12 个必需字段，无 iloc，无 token 泄露。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_dividend_token_abc")
    provider = CnAkshareProvider()

    row_data = {
        "ts_code": "600519.SH",
        "end_date": "20231231",
        "ann_date": "20240326",
        "div_proc": "实施",
        "stk_div": 0.0,
        "stk_bo_rate": 0.0,
        "cash_div": 30.876,
        "cash_div_tax": 30.876,
        "record_date": "20240618",
        "ex_date": "20240619",
        "pay_date": "20240619",
        "imp_ann_date": "20240612",
    }
    payload = _make_dividend_payload(rows=[row_data])

    with patch("requests.post", return_value=_MockResponse(payload)) as mock_post:
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-07-01"
        )

    assert error is None
    assert category is None
    assert isinstance(records, list)
    assert len(records) == 1

    rec = records[0]
    # 验证按名字取数 (无 iloc)
    assert rec["ts_code"] == "600519.SH"
    assert rec["end_date"] == "20231231"
    assert rec["ann_date"] == "2024-03-26"
    assert rec["div_proc"] == "实施"
    assert rec["stk_div"] == 0.0
    assert rec["stk_bo_rate"] == 0.0
    assert rec["cash_div"] == 30.876
    assert rec["cash_div_tax"] == 30.876
    assert rec["record_date"] == "20240618"
    assert rec["ex_date"] == "20240619"
    assert rec["pay_date"] == "20240619"
    assert rec["imp_ann_date"] == "20240612"

    assert rec["symbol"] == "600519"
    assert rec["source_type"] == "tushare_dividend"
    assert rec["canonical_event_id"] is None

    # 验证向网关传递的参数
    assert mock_post.call_count == 1
    call_json = mock_post.call_args[1]["json"]
    assert call_json["api_name"] == _TUSHARE_DIVIDEND_API
    assert call_json["api_name"] == "dividend"
    assert call_json["params"]["ts_code"] == "600519.SH"
    assert call_json["token"] == "mock_dividend_token_abc"
    for field in _TUSHARE_DIVIDEND_REQUIRED_FIELDS:
        assert field in call_json["fields"]


def test_dividend_no_iloc_column_reordering(monkeypatch):
    """验证按名字取数：返回字段次序颠倒打乱时，依然能够按列名准确取数，不依赖 iloc 索引。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_dividend_token_abc")
    provider = CnAkshareProvider()

    scrambled_fields = [
        "imp_ann_date",
        "cash_div",
        "div_proc",
        "stk_div",
        "record_date",
        "end_date",
        "ts_code",
        "pay_date",
        "ex_date",
        "ann_date",
        "stk_bo_rate",
        "cash_div_tax",
    ]
    row_data = {
        "ts_code": "600519.SH",
        "end_date": "20231231",
        "ann_date": "20240326",
        "div_proc": "实施",
        "stk_div": 0.1,
        "stk_bo_rate": 0.2,
        "cash_div": 25.5,
        "cash_div_tax": 25.5,
        "record_date": "20240618",
        "ex_date": "20240619",
        "pay_date": "20240619",
        "imp_ann_date": "20240612",
    }
    payload = _make_dividend_payload(fields=scrambled_fields, rows=[row_data])

    with patch("requests.post", return_value=_MockResponse(payload)):
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-07-01"
        )

    assert error is None
    assert category is None
    assert isinstance(records, list)
    rec = records[0]
    assert rec["ts_code"] == "600519.SH"
    assert rec["cash_div"] == 25.5
    assert rec["stk_div"] == 0.1
    assert rec["stk_bo_rate"] == 0.2
    assert rec["div_proc"] == "实施"
    assert rec["ex_date"] == "20240619"


def test_dividend_empty_table_explicit_no_rows(monkeypatch):
    """3. 空表：空表必须显式上报为 no_rows，严禁解释为无分红或返回成功空列表。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_dividend_token_abc")
    provider = CnAkshareProvider()

    # Case 3.1: 接口返回成功但 items 为空列表
    empty_payload = _make_dividend_payload(empty_items=True)
    with patch("requests.post", return_value=_MockResponse(empty_payload)):
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-07-01"
        )

    assert records is None
    assert category == "no_rows"
    assert error == "tushare.dividend:no_rows"

    # Case 3.2: data 为 None
    none_data_payload = {"code": 0, "msg": "", "data": None}
    with patch("requests.post", return_value=_MockResponse(none_data_payload)):
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-07-01"
        )

    assert records is None
    assert category == "no_rows"
    assert error == "tushare.dividend:no_rows"


def test_dividend_missing_columns(monkeypatch):
    """4. 缺列：当 fields 缺少必需字段时上报 missing_field 类型化错误。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_dividend_token_abc")
    provider = CnAkshareProvider()

    for missing_col in (
        "ex_date",
        "cash_div",
        "stk_div",
        "ann_date",
        "record_date",
        "imp_ann_date",
    ):
        fields_subset = [
            f for f in _TUSHARE_DIVIDEND_REQUIRED_FIELDS if f != missing_col
        ]
        row_data = {
            f: "val" if "date" in f or f in ("ts_code", "div_proc") else 0.0
            for f in fields_subset
        }
        payload = _make_dividend_payload(fields=fields_subset, rows=[row_data])
        with patch("requests.post", return_value=_MockResponse(payload)):
            records, error, category = provider._fetch_tushare_dividend(
                "600519", as_of="2024-07-01"
            )

        assert records is None
        assert category == "missing_field"
        assert error == f"tushare.dividend:missing_field({missing_col})"


def test_dividend_token_missing(monkeypatch):
    """5. Token 缺失：未配置 TUSHARE_TOKEN 时直接返回 token_missing，禁止外发网络请求。"""
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    provider = CnAkshareProvider()

    with patch("requests.post") as mock_post:
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-07-01"
        )

    assert records is None
    assert category == "token_missing"
    assert error == "tushare.dividend:token_missing"
    assert mock_post.call_count == 0


def test_dividend_pit_lookahead_rows_dropped_and_gaps_recorded(monkeypatch):
    """6. PIT 契约：ann_date > as_of 的前视行必须剔除并记缺口，不得把未来预案喂给历史 as_of。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_dividend_token_abc")
    provider = CnAkshareProvider()

    row_prior = {
        "ts_code": "600519.SH",
        "end_date": "20231231",
        "ann_date": "20240326",
        "div_proc": "实施",
        "stk_div": 0.0,
        "stk_bo_rate": 0.0,
        "cash_div": 30.876,
        "cash_div_tax": 30.876,
        "record_date": "20240618",
        "ex_date": "20240619",
        "pay_date": "20240619",
        "imp_ann_date": "20240612",
    }
    row_future = {
        "ts_code": "600519.SH",
        "end_date": "20241231",
        "ann_date": "20250325",
        "div_proc": "预案",
        "stk_div": 0.0,
        "stk_bo_rate": 0.0,
        "cash_div": 35.0,
        "cash_div_tax": 35.0,
        "record_date": "",
        "ex_date": "",
        "pay_date": "",
        "imp_ann_date": "",
    }
    payload = _make_dividend_payload(rows=[row_prior, row_future])

    # as_of = "2024-07-01" 时，row_future (ann_date="20250325") 必须被剔除并记缺口，仅保留 row_prior
    with patch("requests.post", return_value=_MockResponse(payload)):
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-07-01"
        )

    assert error is None
    assert category is None
    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["ann_date"] == "2024-03-26"
    assert records[0]["end_date"] == "20231231"

    # 验证记缺口：剔除的前视行记录在 lookahead_gaps 中
    assert "lookahead_gaps" in records[0]
    gaps = records[0]["lookahead_gaps"]
    assert len(gaps) == 1
    assert gaps[0]["reason"] == "ann_date_exceeds_as_of"
    assert gaps[0]["ann_date"] == "2025-03-25"
    assert gaps[0]["as_of"] == "2024-07-01"
    assert gaps[0]["end_date"] == "20241231"


def test_dividend_pit_implementation_future_date_masked(monkeypatch):
    """7. PIT 契约：当 ann_date <= as_of 但 imp_ann_date > as_of 时，实施日期必须遮蔽并记实施缺口。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_dividend_token_abc")
    provider = CnAkshareProvider()

    # 预案公布于 2024-03-26，实施公告于 2024-06-12。
    # 如果站在 2024-04-15 评估，市场只知道预案，不知道未来的实施公告与除权日。
    row = {
        "ts_code": "600519.SH",
        "end_date": "20231231",
        "ann_date": "20240326",
        "div_proc": "实施",
        "stk_div": 0.0,
        "stk_bo_rate": 0.0,
        "cash_div": 30.876,
        "cash_div_tax": 30.876,
        "record_date": "20240618",
        "ex_date": "20240619",
        "pay_date": "20240619",
        "imp_ann_date": "20240612",
    }
    payload = _make_dividend_payload(rows=[row])

    with patch("requests.post", return_value=_MockResponse(payload)):
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-04-15"
        )

    assert error is None
    assert category is None
    assert isinstance(records, list)
    assert len(records) == 1
    rec = records[0]
    assert rec["ann_date"] == "2024-03-26"
    # 实施相关日期在 2024-04-15 尚未发生，必须被遮蔽为 None
    assert rec["imp_ann_date"] is None
    assert rec["record_date"] is None
    assert rec["ex_date"] is None
    assert rec["pay_date"] is None
    assert "pit_implementation_gap" in rec
    assert "imp_ann_date(2024-06-12)>2024-04-15" in rec["pit_implementation_gap"]


def test_dividend_all_rows_exceed_as_of_reported_as_no_rows(monkeypatch):
    """8. 当网关返回的所有行均晚于 as_of 时，全部行被剔除，上报 no_rows(all_rows_exceed_as_of)。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_dividend_token_abc")
    provider = CnAkshareProvider()

    row_future = {
        "ts_code": "600519.SH",
        "end_date": "20241231",
        "ann_date": "20250325",
        "div_proc": "预案",
        "stk_div": 0.0,
        "stk_bo_rate": 0.0,
        "cash_div": 35.0,
        "cash_div_tax": 35.0,
        "record_date": "",
        "ex_date": "",
        "pay_date": "",
        "imp_ann_date": "",
    }
    payload = _make_dividend_payload(rows=[row_future])

    with patch("requests.post", return_value=_MockResponse(payload)):
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-01-01"
        )

    assert records == []
    assert category == "no_rows"
    assert "all_rows_exceed_as_of" in error


def test_dividend_query_date_exceeds_as_of_rejected_before_network(monkeypatch):
    """9. 查询参数越界：显式传入的 ann_date > as_of 在发网前立即拒绝，不产生网络请求。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_dividend_token_abc")
    provider = CnAkshareProvider()

    with patch("requests.post") as mock_post:
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-05-01", ann_date="2024-06-01"
        )

    assert records is None
    assert category == "date_exceeds_as_of"
    assert "2024-06-01>2024-05-01" in str(error)
    assert mock_post.call_count == 0


@pytest.mark.parametrize(
    "status_code, resp_payload, expected_cat",
    [
        (200, {"code": 2002, "msg": "权限不足", "data": None}, "permission_denied"),
        (200, {"code": 40101, "msg": "未授权凭证", "data": None}, "permission_denied"),
        (403, {"code": 0, "data": None}, "permission_denied"),
        (200, {"code": 40203, "msg": "访问频次超限", "data": None}, "rate_limited"),
        (429, {"code": 0, "data": None}, "rate_limited"),
        (200, {"code": 99999, "msg": "其它服务端未知错误", "data": None}, "api_code"),
        (200, {"code": "invalid"}, "api_code_invalid"),
        (200, {"code": 0}, "json_shape"),
        (200, {"code": 0, "data": "not-a-dict"}, "json_shape"),
        (200, {"code": 0, "data": {"fields": "not-a-list", "items": []}}, "json_shape"),
        (200, {"code": 0, "data": {"fields": [], "items": "not-a-list"}}, "json_shape"),
    ],
)
def test_dividend_error_classifications(
    monkeypatch, status_code, resp_payload, expected_cat
):
    """验证错误码精细化分类：permission_denied / rate_limited / json_shape 等。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_dividend_token_abc")
    provider = CnAkshareProvider()

    with patch("requests.post", return_value=_MockResponse(resp_payload, status_code=status_code)):
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-07-01"
        )

    assert records is None
    assert category == expected_cat
    assert f"tushare.dividend:{expected_cat}" in error


def test_dividend_transport_timeout_and_error(monkeypatch):
    """验证传输超时与网络错误处理。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_dividend_token_abc")
    provider = CnAkshareProvider()

    with patch("requests.post", side_effect=requests.Timeout("gateway timeout")):
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-07-01"
        )
    assert records is None
    assert category == "transport_timeout"
    assert error == "tushare.dividend:transport_timeout"

    with patch("requests.post", side_effect=requests.ConnectionError("connection failed")):
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-07-01"
        )
    assert records is None
    assert category == "transport_error"
    assert error == "tushare.dividend:transport_error"


def test_dividend_url_environment_cascade(monkeypatch):
    """验证网关环境变量路由：TUSHARE_API_URL > TUSHARE_BASE_URL > 官方兜底。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_dividend_token_abc")
    provider = CnAkshareProvider()
    row_data = {
        "ts_code": "600519.SH",
        "end_date": "20231231",
        "ann_date": "20240326",
        "div_proc": "实施",
        "stk_div": 0.0,
        "stk_bo_rate": 0.0,
        "cash_div": 30.876,
        "cash_div_tax": 30.876,
        "record_date": "20240618",
        "ex_date": "20240619",
        "pay_date": "20240619",
        "imp_ann_date": "20240612",
    }
    payload = _make_dividend_payload(rows=[row_data])

    # 1. TUSHARE_API_URL 优先
    monkeypatch.setenv("TUSHARE_API_URL", "http://gateway.internal:9000/v1")
    monkeypatch.setenv("TUSHARE_BASE_URL", "http://backup.internal:9000/v1")
    with patch("requests.post", return_value=_MockResponse(payload)) as mock_post:
        provider._fetch_tushare_dividend("600519", as_of="2024-07-01")
    assert mock_post.call_args[0][0] == "http://gateway.internal:9000/v1"

    # 2. TUSHARE_BASE_URL 备选
    monkeypatch.delenv("TUSHARE_API_URL", raising=False)
    with patch("requests.post", return_value=_MockResponse(payload)) as mock_post:
        provider._fetch_tushare_dividend("600519", as_of="2024-07-01")
    assert mock_post.call_args[0][0] == "http://backup.internal:9000/v1"

    # 3. 环境变量均未配置时，默认回落至官方端点 https://api.tushare.pro
    monkeypatch.delenv("TUSHARE_BASE_URL", raising=False)
    with patch("requests.post", return_value=_MockResponse(payload)) as mock_post:
        provider._fetch_tushare_dividend("600519", as_of="2024-07-01")
    assert mock_post.call_args[0][0] == "https://api.tushare.pro"


def test_dividend_input_validation(monkeypatch):
    """验证非法股票代码和非法日期输入。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_dividend_token_abc")
    provider = CnAkshareProvider()

    with patch("requests.post") as mock_post:
        # 非法 symbol
        records, error, category = provider._fetch_tushare_dividend(
            "bad_symbol", as_of="2024-07-01"
        )
        assert records is None
        assert category == "validation"
        assert error == "tushare.dividend:validation(symbol)"

        # 非法 as_of
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="invalid_as_of"
        )
        assert records is None
        assert category == "validation"
        assert error == "tushare.dividend:validation(as_of)"

        # 非法 ann_date
        records, error, category = provider._fetch_tushare_dividend(
            "600519", as_of="2024-07-01", ann_date="not-a-date"
        )
        assert records is None
        assert category == "validation"
        assert error == "tushare.dividend:validation(ann_date)"

    assert mock_post.call_count == 0
