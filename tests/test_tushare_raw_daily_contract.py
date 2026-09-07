"""Unit tests for Tushare Pro raw daily reader interface contract (D-01 / C-04-2 / DAV-703).

Covers:
- Normal single-row fetching and column access by name (no iloc).
- Strictly unadjusted raw price contract (no adj / adj_factor params).
- Token missing handling without network traffic.
- Empty table / no rows handling (suspension, non-trading days).
- Missing required fields reporting (open, high, low, close, pre_close, vol, amount, etc.).
- PIT date boundary guard (trade_date > as_of rejected before network call).
- Typed error classifications: permission_denied, rate_limited, json_shape, transport.
- URL routing via TUSHARE_API_URL / TUSHARE_BASE_URL with official default fallback.
"""

from unittest.mock import patch

import pytest
import requests

from tradingagents.dataflows.providers.cn_akshare_provider import (
    _TUSHARE_DAILY_API,
    _TUSHARE_DAILY_REQUIRED_FIELDS,
    _TUSHARE_RAW_DAILY_API,
    _TUSHARE_RAW_DAILY_REQUIRED_FIELDS,
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


def _make_daily_payload(
    *,
    code: int = 0,
    msg: str = "",
    ts_code: str = "600519.SH",
    trade_date: str = "20260814",
    open_p: float = 1800.0,
    high: float = 1860.0,
    low: float = 1795.0,
    close: float = 1850.5,
    pre_close: float = 1810.0,
    vol: float = 28500.0,
    amount: float = 521400.12,
    fields: list[str] | None = None,
    empty_items: bool = False,
    data: dict | None = None,
):
    if data is not None:
        return {"code": code, "msg": msg, "data": data}

    if fields is None:
        fields = list(_TUSHARE_DAILY_REQUIRED_FIELDS)

    field_val_map = {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "pre_close": pre_close,
        "vol": vol,
        "amount": amount,
    }

    if empty_items:
        items = []
    else:
        items = [[field_val_map.get(f, 0.0) for f in fields]]

    return {
        "code": code,
        "msg": msg,
        "data": {
            "fields": fields,
            "items": items,
        },
    }


def test_raw_daily_success_row(monkeypatch):
    """1. 正常一行：验证返回字典按列名取数，包含所有必需列，无 iloc，无 token 泄露，不传复权参数。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_daily_token_12345")
    provider = CnAkshareProvider()

    payload = _make_daily_payload(
        ts_code="600519.SH",
        trade_date="20260814",
        open_p=1810.0,
        high=1865.0,
        low=1805.0,
        close=1850.5,
        pre_close=1800.0,
        vol=32000.0,
        amount=580000.0,
    )

    with patch("requests.post", return_value=_MockResponse(payload)) as mock_post:
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "2026-08-14", as_of="2026-08-14"
        )

    assert error is None
    assert category is None
    assert row is not None
    assert isinstance(row, dict)

    # 验证按名字取数 (无 iloc)
    assert row["ts_code"] == "600519.SH"
    assert row["trade_date"] == "20260814"
    assert row["open"] == 1810.0
    assert row["high"] == 1865.0
    assert row["low"] == 1805.0
    assert row["close"] == 1850.5
    assert row["pre_close"] == 1800.0
    assert row["vol"] == 32000.0
    assert row["amount"] == 580000.0

    # 验证向网关传递的参数规范：严禁复权参数，走 daily
    assert mock_post.call_count == 1
    call_json = mock_post.call_args[1]["json"]
    assert call_json["api_name"] == _TUSHARE_DAILY_API
    assert call_json["api_name"] == "daily"
    assert call_json["params"] == {"ts_code": "600519.SH", "trade_date": "20260814"}
    assert "adj" not in call_json["params"]
    assert "adj_factor" not in call_json["params"]
    assert call_json["token"] == "mock_daily_token_12345"
    for col in _TUSHARE_DAILY_REQUIRED_FIELDS:
        assert col in call_json["fields"]

    # 验证别名 _fetch_tushare_daily 同样有效
    assert provider._fetch_tushare_daily == provider._fetch_tushare_raw_daily


def test_raw_daily_no_iloc_column_reordering(monkeypatch):
    """验证按名字取数：返回字段次序颠倒打乱时，依然能够按列名准确取数，不依赖 iloc 索引。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_daily_token_12345")
    provider = CnAkshareProvider()

    scrambled_fields = [
        "amount",
        "close",
        "pre_close",
        "trade_date",
        "vol",
        "low",
        "ts_code",
        "open",
        "high",
    ]
    payload = _make_daily_payload(
        fields=scrambled_fields,
        ts_code="600519.SH",
        trade_date="20260814",
        open_p=1820.0,
        high=1870.0,
        low=1815.0,
        close=1860.0,
        pre_close=1810.0,
        vol=29000.0,
        amount=540000.0,
    )

    with patch("requests.post", return_value=_MockResponse(payload)):
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "2026-08-14"
        )

    assert error is None
    assert category is None
    assert row is not None
    assert row["open"] == 1820.0
    assert row["high"] == 1870.0
    assert row["low"] == 1815.0
    assert row["close"] == 1860.0
    assert row["pre_close"] == 1810.0
    assert row["vol"] == 29000.0
    assert row["amount"] == 540000.0
    assert row["ts_code"] == "600519.SH"
    assert row["trade_date"] == "20260814"


def test_raw_daily_token_missing(monkeypatch):
    """2. Token 缺失：未配置 TUSHARE_TOKEN 时直接返回 token_missing，禁止外发网络请求。"""
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    provider = CnAkshareProvider()

    with patch("requests.post") as mock_post:
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "2026-08-14"
        )

    assert row is None
    assert category == "token_missing"
    assert error == "tushare.daily:token_missing"
    assert mock_post.call_count == 0


def test_raw_daily_empty_table_no_rows(monkeypatch):
    """3. 空表：网关返回 items=[] 或无匹配交易日，返回类型化 no_rows 错误。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_daily_token_12345")
    provider = CnAkshareProvider()

    # Case 3.1: 接口返回成功但 items 为空（停牌或非交易日）
    empty_payload = _make_daily_payload(empty_items=True)
    with patch("requests.post", return_value=_MockResponse(empty_payload)):
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "2026-08-14"
        )

    assert row is None
    assert category == "no_rows"
    assert error == "tushare.daily:no_rows"

    # Case 3.2: data 为 None
    none_data_payload = {"code": 0, "msg": "", "data": None}
    with patch("requests.post", return_value=_MockResponse(none_data_payload)):
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "2026-08-14"
        )

    assert row is None
    assert category == "no_rows"
    assert error == "tushare.daily:no_rows"

    # Case 3.3: items 中日期不匹配请求日期
    mismatch_payload = _make_daily_payload(trade_date="20260813")
    with patch("requests.post", return_value=_MockResponse(mismatch_payload)):
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "2026-08-14"
        )

    assert row is None
    assert category == "no_rows"
    assert error == "tushare.daily:no_rows"


def test_raw_daily_missing_columns(monkeypatch):
    """4. 缺列：当 fields 缺少必需字段时上报 missing_field 类型化错误。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_daily_token_12345")
    provider = CnAkshareProvider()

    for missing_col in ("open", "close", "high", "low", "pre_close", "vol", "amount"):
        fields_subset = [
            f for f in _TUSHARE_DAILY_REQUIRED_FIELDS if f != missing_col
        ]
        payload = _make_daily_payload(fields=fields_subset)
        with patch("requests.post", return_value=_MockResponse(payload)):
            row, error, category = provider._fetch_tushare_raw_daily(
                "600519", "2026-08-14"
            )

        assert row is None
        assert category == "missing_field"
        assert error == f"tushare.daily:missing_field({missing_col})"


def test_raw_daily_pit_date_exceeds_as_of(monkeypatch):
    """5. 日期越界：trade_date > as_of 必须严格在发网前拒绝，不得发出网关请求。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_daily_token_12345")
    provider = CnAkshareProvider()

    with patch("requests.post") as mock_post:
        # trade_date (2026-08-15) > as_of (2026-08-14)
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "2026-08-15", as_of="2026-08-14"
        )

    assert row is None
    assert category == "date_exceeds_as_of"
    assert "date_exceeds_as_of" in str(error)
    assert "2026-08-15>2026-08-14" in str(error)
    assert mock_post.call_count == 0

    # 跨格式日期校验 (YYYYMMDD 与 YYYY-MM-DD 混合)
    with patch("requests.post") as mock_post2:
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "20260815", as_of="2026-08-14"
        )

    assert row is None
    assert category == "date_exceeds_as_of"
    assert mock_post2.call_count == 0


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
def test_raw_daily_error_classifications(
    monkeypatch, status_code, resp_payload, expected_cat
):
    """验证错误码精细化分类：permission_denied / rate_limited / json_shape 等。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_daily_token_12345")
    provider = CnAkshareProvider()

    with patch("requests.post", return_value=_MockResponse(resp_payload, status_code=status_code)):
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "2026-08-14"
        )

    assert row is None
    assert category == expected_cat
    assert f"tushare.daily:{expected_cat}" in error


def test_raw_daily_transport_timeout_and_error(monkeypatch):
    """验证传输超时与网络错误处理。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_daily_token_12345")
    provider = CnAkshareProvider()

    with patch("requests.post", side_effect=requests.Timeout("gateway timeout")):
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "2026-08-14"
        )
    assert row is None
    assert category == "transport_timeout"
    assert error == "tushare.daily:transport_timeout"

    with patch("requests.post", side_effect=requests.ConnectionError("connection failed")):
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "2026-08-14"
        )
    assert row is None
    assert category == "transport_error"
    assert error == "tushare.daily:transport_error"


def test_raw_daily_url_environment_cascade(monkeypatch):
    """验证网关环境变量路由：TUSHARE_API_URL > TUSHARE_BASE_URL > 官方兜底。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_daily_token_12345")
    provider = CnAkshareProvider()
    payload = _make_daily_payload()

    # 1. TUSHARE_API_URL 优先
    monkeypatch.setenv("TUSHARE_API_URL", "http://gateway.internal:9000/v1")
    monkeypatch.setenv("TUSHARE_BASE_URL", "http://backup.internal:9000/v1")
    with patch("requests.post", return_value=_MockResponse(payload)) as mock_post:
        provider._fetch_tushare_raw_daily("600519", "2026-08-14")
    assert mock_post.call_args[0][0] == "http://gateway.internal:9000/v1"

    # 2. TUSHARE_BASE_URL 备选
    monkeypatch.delenv("TUSHARE_API_URL", raising=False)
    with patch("requests.post", return_value=_MockResponse(payload)) as mock_post:
        provider._fetch_tushare_raw_daily("600519", "2026-08-14")
    assert mock_post.call_args[0][0] == "http://backup.internal:9000/v1"

    # 3. 环境变量均未配置时，默认回落至官方端点 https://api.tushare.pro
    monkeypatch.delenv("TUSHARE_BASE_URL", raising=False)
    with patch("requests.post", return_value=_MockResponse(payload)) as mock_post:
        provider._fetch_tushare_raw_daily("600519", "2026-08-14")
    assert mock_post.call_args[0][0] == "https://api.tushare.pro"


def test_raw_daily_input_validation(monkeypatch):
    """验证非法股票代码和非法日期输入。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_daily_token_12345")
    provider = CnAkshareProvider()

    with patch("requests.post") as mock_post:
        # 非法 symbol
        row, error, category = provider._fetch_tushare_raw_daily(
            "bad_symbol", "2026-08-14"
        )
        assert row is None
        assert category == "validation"
        assert error == "tushare.daily:validation(symbol)"

        # 非法 trade_date
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "not-a-date"
        )
        assert row is None
        assert category == "validation"
        assert error == "tushare.daily:validation(trade_date)"

        # 非法 as_of
        row, error, category = provider._fetch_tushare_raw_daily(
            "600519", "2026-08-14", as_of="invalid_as_of"
        )
        assert row is None
        assert category == "validation"
        assert error == "tushare.daily:validation(as_of)"

    assert mock_post.call_count == 0


def test_raw_daily_range_query(monkeypatch):
    """验证日期区间查询及 as_of 截断。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_daily_token_12345")
    provider = CnAkshareProvider()

    fields = list(_TUSHARE_DAILY_REQUIRED_FIELDS)
    items = [
        ["600519.SH", "20260812", 1790.0, 1810.0, 1785.0, 1800.0, 1780.0, 25000.0, 450000.0],
        ["600519.SH", "20260813", 1805.0, 1830.0, 1800.0, 1820.0, 1800.0, 27000.0, 490000.0],
        ["600519.SH", "20260814", 1820.0, 1855.0, 1815.0, 1850.0, 1820.0, 31000.0, 570000.0],
    ]
    payload = {
        "code": 0,
        "msg": "",
        "data": {"fields": fields, "items": items},
    }

    with patch("requests.post", return_value=_MockResponse(payload)) as mock_post:
        rows, error, category = provider._fetch_tushare_raw_daily(
            "600519", start_date="2026-08-12", end_date="2026-08-13", as_of="2026-08-13"
        )

    assert error is None
    assert category is None
    assert isinstance(rows, list)
    # 20260814 > as_of(2026-08-13) 必须被过滤掉，仅返回 20260812 和 20260813
    assert len(rows) == 2
    assert rows[0]["trade_date"] == "20260812"
    assert rows[1]["trade_date"] == "20260813"
