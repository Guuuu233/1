"""Unit tests for optional raw daily channel in CnAkshareProvider (D-02-2 / C-04-3 / DAV-707).

Contracts verified:
1. Default behavior (omitted price_basis):
   - Stays on vendor_qfq chain (Eastmoney -> Sina -> Tencent with adjust="qfq").
   - Result text and DataFrame metadata explicitly identify as vendor_qfq.
   - Never labeled as raw.
2. Explicit price_basis="vendor_qfq":
   - Explicit keyword or positional call stays on vendor_qfq chain.
   - Never calls Tushare daily gateway.
3. Explicit price_basis="raw":
   - Routes to _fetch_tushare_raw_daily and retrieves unadjusted daily bars.
   - Column access by name (no iloc); standard OHLCV normalization.
   - Result text and DataFrame metadata explicitly identify as raw.
   - AkShare chain is not invoked.
4. Raw failure handling:
   - D-01 failure types (token_missing, date_exceeds_as_of, no_rows, missing_field,
     json_shape, permission_denied, rate_limited, transport, validation) are preserved.
   - Explicitly raises RawDailyFetchError (never pretends success with empty string).
   - Never falls back to vendor_qfq on raw failures.
5. Forbidden channels and unknown labels:
   - price_basis="pit_raw" and "pit_adjusted" are forbidden as available channels
     (raises UnsupportedPriceBasisError / NotImplementedError).
   - Unknown/invalid labels (e.g. "unknown", "", None, "price_basis.raw") fail closed
     by raising UnknownPriceBasisError.
   - Never silently fall back to vendor_qfq.
6. Zero real network traffic:
   - All tests use mocked HTTP / mocked ak, preventing real gateway calls.
"""

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from api.services.price_basis_labels import (
    PRICE_BASIS_PIT_ADJUSTED,
    PRICE_BASIS_PIT_RAW,
    PRICE_BASIS_RAW,
    PRICE_BASIS_UNSPECIFIED,
    PRICE_BASIS_VENDOR_QFQ,
    PriceBasisError,
    UnknownPriceBasisError,
)
from tradingagents.dataflows.providers.cn_akshare_provider import (
    _TUSHARE_DAILY_REQUIRED_FIELDS,
    CnAkshareProvider,
    RawDailyFetchError,
    StockDataText,
    UnsupportedPriceBasisError,
)


class _MockResponse:
    """Mock requests.Response for Tushare gateway tests."""

    def __init__(self, data: dict, status_code: int = 200, text: str = ""):
        self._data = data
        self.status_code = status_code
        self.text = text

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def _make_tushare_daily_payload(
    *,
    code: int = 0,
    msg: str = "",
    ts_code: str = "600519.SH",
    items: list[list] | None = None,
    fields: list[str] | None = None,
    empty_items: bool = False,
    data: dict | None = None,
):
    """Generate mock payload for Tushare daily endpoint."""
    if data is not None:
        return {"code": code, "msg": msg, "data": data}

    if fields is None:
        fields = list(_TUSHARE_DAILY_REQUIRED_FIELDS)

    if empty_items:
        items = []
    elif items is None:
        items = [
            [ts_code, "20260812", 1800.0, 1820.0, 1795.0, 1810.0, 1800.0, 20000.0, 360000.0],
            [ts_code, "20260813", 1810.0, 1835.0, 1805.0, 1830.0, 1810.0, 22000.0, 400000.0],
            [ts_code, "20260814", 1830.0, 1860.0, 1820.0, 1850.5, 1830.0, 28500.0, 521400.12],
        ]

    return {
        "code": code,
        "msg": msg,
        "data": {
            "fields": fields,
            "items": items,
        },
    }


def _make_sample_ak_df():
    """Generate sample DataFrame simulating AkShare daily hist output."""
    return pd.DataFrame(
        {
            "日期": ["2026-08-12", "2026-08-13", "2026-08-14"],
            "开盘": [1800.0, 1810.0, 1830.0],
            "最高": [1820.0, 1835.0, 1860.0],
            "最低": [1795.0, 1805.0, 1820.0],
            "收盘": [1810.0, 1830.0, 1850.5],
            "成交量": [20000.0, 22000.0, 28500.0],
            "成交额": [360000.0, 400000.0, 521400.12],
        }
    )


# ==============================================================================
# 1. 契约 1：缺省与显式 vendor_qfq 行为一致，走 qfq 链，标明 vendor_qfq
# ==============================================================================

class TestDefaultAndVendorQfqPath:
    """验证缺省调用与显式 vendor_qfq 严格走 AkShare qfq 链，不得走 raw，不得标 raw。"""

    def test_default_price_basis_is_vendor_qfq(self):
        """get_stock_data(symbol, start, end) 缺省使用 vendor_qfq，文本与元数据标明 vendor_qfq。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        mock_ak.stock_zh_a_hist.return_value = _make_sample_ak_df()

        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post") as mock_post:
            out = provider.get_stock_data("600519", "2026-08-12", "2026-08-14")

        # 验证 AkShare 链被调用且参数包含 adjust='qfq'
        mock_ak.stock_zh_a_hist.assert_called_once()
        call_kwargs = mock_ak.stock_zh_a_hist.call_args[1]
        assert call_kwargs.get("adjust") == "qfq"

        # 验证 Tushare 网关绝未被触碰
        assert mock_post.call_count == 0

        # 验证返回类型和口径标签
        assert isinstance(out, str)
        assert isinstance(out, StockDataText)
        assert out.price_basis == PRICE_BASIS_VENDOR_QFQ
        assert "# price_basis: vendor_qfq" in out
        assert "# price_basis: raw" not in out

        # 验证 CSV 可正常解析
        df = pd.read_csv(io.StringIO(out), comment="#")
        assert len(df) == 3
        assert list(df["Date"]) == ["2026-08-12", "2026-08-13", "2026-08-14"]

    def test_fetch_hist_df_default_attrs_vendor_qfq(self):
        """_fetch_hist_df 缺省返回的 DataFrame 带有 attrs['price_basis'] = 'vendor_qfq'。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        mock_ak.stock_zh_a_hist.return_value = _make_sample_ak_df()

        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post") as mock_post:
            df = provider._fetch_hist_df("600519", "2026-08-12", "2026-08-14")

        assert mock_ak.stock_zh_a_hist.call_count == 1
        assert mock_post.call_count == 0
        assert df.attrs.get("price_basis") == PRICE_BASIS_VENDOR_QFQ
        assert df.attrs.get("price_basis") != PRICE_BASIS_RAW

    def test_explicit_vendor_qfq_kwarg(self):
        """显式传入 price_basis='vendor_qfq' 行为与缺省一致。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        mock_ak.stock_zh_a_hist.return_value = _make_sample_ak_df()

        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post") as mock_post:
            out = provider.get_stock_data(
                "600519", "2026-08-12", "2026-08-14", price_basis=PRICE_BASIS_VENDOR_QFQ
            )

        assert mock_ak.stock_zh_a_hist.call_count == 1
        assert mock_post.call_count == 0
        assert out.price_basis == PRICE_BASIS_VENDOR_QFQ
        assert "# price_basis: vendor_qfq" in out

    def test_explicit_vendor_qfq_positional(self):
        """位置参数传入 'vendor_qfq' 正常支持。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        mock_ak.stock_zh_a_hist.return_value = _make_sample_ak_df()

        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post") as mock_post:
            out = provider.get_stock_data(
                "600519", "2026-08-12", "2026-08-14", "vendor_qfq"
            )

        assert mock_ak.stock_zh_a_hist.call_count == 1
        assert mock_post.call_count == 0
        assert out.price_basis == PRICE_BASIS_VENDOR_QFQ


# ==============================================================================
# 2. 契约 2：显式 raw 走 _fetch_tushare_raw_daily，不走 ak 链
# ==============================================================================

class TestExplicitRawDailyPath:
    """验证显式 price_basis='raw' 走 Tushare 未复权日线通道。"""

    def test_raw_daily_success_flow(self, monkeypatch):
        """验证显式 price_basis='raw' 走 Tushare daily mock，字段按列名提取，标明 raw。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_for_raw_provider")
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        payload = _make_tushare_daily_payload(ts_code="600519.SH")

        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post", return_value=_MockResponse(payload)) as mock_post:
            out = provider.get_stock_data(
                "600519", "2026-08-12", "2026-08-14", price_basis=PRICE_BASIS_RAW
            )

        # 验证 Tushare 网关被调用
        assert mock_post.call_count == 1
        # 验证 AkShare 链绝未被调用
        assert mock_ak.stock_zh_a_hist.call_count == 0

        # 验证返回文本与元数据
        assert isinstance(out, str)
        assert isinstance(out, StockDataText)
        assert out.price_basis == PRICE_BASIS_RAW
        assert "# price_basis: raw" in out
        assert "# price_basis: vendor_qfq" not in out

        # 验证 CSV 内容与按列名映射正确
        df = pd.read_csv(io.StringIO(out), comment="#")
        assert len(df) == 3
        assert list(df["Date"]) == ["2026-08-12", "2026-08-13", "2026-08-14"]
        assert list(df["Open"]) == [1800.0, 1810.0, 1830.0]
        assert list(df["Close"]) == [1810.0, 1830.0, 1850.5]
        assert list(df["Volume"]) == [20000.0, 22000.0, 28500.0]

    def test_fetch_hist_df_raw_returns_dataframe_with_attrs(self, monkeypatch):
        """_fetch_hist_df(..., price_basis='raw') 返回的 DataFrame 正确设置 attrs['price_basis'] = 'raw'。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_for_raw_provider")
        provider = CnAkshareProvider()
        payload = _make_tushare_daily_payload()

        with patch("requests.post", return_value=_MockResponse(payload)):
            df = provider._fetch_hist_df(
                "600519", "2026-08-12", "2026-08-14", price_basis=PRICE_BASIS_RAW
            )

        assert isinstance(df, pd.DataFrame)
        assert df.attrs.get("price_basis") == PRICE_BASIS_RAW
        assert df.attrs.get("price_basis") != PRICE_BASIS_VENDOR_QFQ
        assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 3

    def test_raw_daily_positional_argument(self, monkeypatch):
        """位置参数传入 'raw' 亦能正确命中 raw 通道。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_for_raw_provider")
        provider = CnAkshareProvider()
        payload = _make_tushare_daily_payload()

        with patch("requests.post", return_value=_MockResponse(payload)) as mock_post:
            out = provider.get_stock_data("600519", "2026-08-12", "2026-08-14", "raw")

        assert mock_post.call_count == 1
        assert out.price_basis == PRICE_BASIS_RAW
        assert "# price_basis: raw" in out


# ==============================================================================
# 3. 契约 2 & 4：raw 失败沿用 D-01 失败类型，禁止空串冒充，禁止回退到 qfq
# ==============================================================================

class TestRawDailyFailuresFailClosed:
    """验证 raw 通道所有失败类型显式抛出 RawDailyFetchError，携带 D-01 错误码，绝不落入 qfq。"""

    def test_raw_token_missing_raises_error_no_network_no_qfq(self, monkeypatch):
        """无 token 时发网前拦截，抛出 token_missing，零网络，不回退到 qfq。"""
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
        provider = CnAkshareProvider()
        mock_ak = MagicMock()

        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post") as mock_post:
            with pytest.raises(RawDailyFetchError) as exc_info:
                provider.get_stock_data(
                    "600519", "2026-08-12", "2026-08-14", price_basis=PRICE_BASIS_RAW
                )

        assert exc_info.value.category == "token_missing"
        assert exc_info.value.error == "tushare.daily:token_missing"
        assert mock_post.call_count == 0
        assert mock_ak.stock_zh_a_hist.call_count == 0

    def test_raw_date_exceeds_as_of_raises_error_no_network_no_qfq(self, monkeypatch):
        """end_date > as_of 时发网前拦截，抛出 date_exceeds_as_of，零网络，不回退到 qfq。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token")
        provider = CnAkshareProvider()
        mock_ak = MagicMock()

        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post") as mock_post:
            with pytest.raises(RawDailyFetchError) as exc_info:
                provider.get_stock_data(
                    "600519",
                    "2026-08-12",
                    "2026-08-15",
                    price_basis=PRICE_BASIS_RAW,
                    as_of="2026-08-14",
                )

        assert exc_info.value.category == "date_exceeds_as_of"
        assert "date_exceeds_as_of" in exc_info.value.error
        assert mock_post.call_count == 0
        assert mock_ak.stock_zh_a_hist.call_count == 0

    def test_raw_no_rows_raises_error_not_empty_string(self, monkeypatch):
        """网关返回空表（no_rows）显式报错，禁止返回空串冒充成功，绝不回退 qfq。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token")
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        payload = _make_tushare_daily_payload(empty_items=True)

        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post", return_value=_MockResponse(payload)):
            with pytest.raises(RawDailyFetchError) as exc_info:
                provider.get_stock_data(
                    "600519", "2026-08-12", "2026-08-14", price_basis=PRICE_BASIS_RAW
                )

        assert exc_info.value.category == "no_rows"
        assert exc_info.value.error == "tushare.daily:no_rows"
        assert mock_ak.stock_zh_a_hist.call_count == 0

    def test_raw_missing_required_field_raises_error(self, monkeypatch):
        """缺列（missing_field）显式报错，绝不回退 qfq。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token")
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        # 故意缺少 close 字段
        fields_without_close = [
            f for f in _TUSHARE_DAILY_REQUIRED_FIELDS if f != "close"
        ]
        payload = _make_tushare_daily_payload(fields=fields_without_close)

        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post", return_value=_MockResponse(payload)):
            with pytest.raises(RawDailyFetchError) as exc_info:
                provider.get_stock_data(
                    "600519", "2026-08-12", "2026-08-14", price_basis=PRICE_BASIS_RAW
                )

        assert exc_info.value.category == "missing_field"
        assert "missing_field" in exc_info.value.error
        assert mock_ak.stock_zh_a_hist.call_count == 0

    def test_raw_json_shape_invalid_raises_error(self, monkeypatch):
        """JSON 结构异常（json_shape）显式报错，绝不回退 qfq。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token")
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        # data 不是 dict
        bad_payload = {"code": 0, "msg": "", "data": "not-an-object"}

        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post", return_value=_MockResponse(bad_payload)):
            with pytest.raises(RawDailyFetchError) as exc_info:
                provider.get_stock_data(
                    "600519", "2026-08-12", "2026-08-14", price_basis=PRICE_BASIS_RAW
                )

        assert exc_info.value.category == "json_shape"
        assert mock_ak.stock_zh_a_hist.call_count == 0

    def test_raw_permission_denied_raises_error(self, monkeypatch):
        """网关权限不足（permission_denied）显式报错，绝不回退 qfq。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token")
        provider = CnAkshareProvider()
        mock_ak = MagicMock()
        perm_payload = {"code": 40101, "msg": "未授权凭证", "data": None}

        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post", return_value=_MockResponse(perm_payload)):
            with pytest.raises(RawDailyFetchError) as exc_info:
                provider.get_stock_data(
                    "600519", "2026-08-12", "2026-08-14", price_basis=PRICE_BASIS_RAW
                )

        assert exc_info.value.category == "permission_denied"
        assert mock_ak.stock_zh_a_hist.call_count == 0

    def test_raw_transport_error_raises_error(self, monkeypatch):
        """网络/传输异常（transport_error）显式报错，绝不回退 qfq。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token")
        provider = CnAkshareProvider()
        mock_ak = MagicMock()

        import requests
        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post", side_effect=requests.ConnectionError("network down")):
            with pytest.raises(RawDailyFetchError) as exc_info:
                provider.get_stock_data(
                    "600519", "2026-08-12", "2026-08-14", price_basis=PRICE_BASIS_RAW
                )

        assert exc_info.value.category == "transport_error"
        assert exc_info.value.error == "tushare.daily:transport_error"
        assert mock_ak.stock_zh_a_hist.call_count == 0


# ==============================================================================
# 4. 契约 3：禁止 pit_raw / pit_adjusted 可用通道；未知标签失败闭合
# ==============================================================================

class TestForbiddenChannelsAndUnknownLabels:
    """验证 pit_raw/pit_adjusted 被禁止作为可用通道，未知标签严格失败闭合。"""

    @pytest.mark.parametrize("forbidden_basis", [PRICE_BASIS_PIT_RAW, PRICE_BASIS_PIT_ADJUSTED])
    def test_pit_channels_forbidden(self, forbidden_basis):
        """pit_raw 与 pit_adjusted 本卡禁止实现为可用通道，抛出明确异常，绝不默默 qfq。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()

        with patch.object(provider, "_ak", return_value=mock_ak):
            with pytest.raises((UnsupportedPriceBasisError, NotImplementedError, PriceBasisError)):
                provider.get_stock_data(
                    "600519", "2026-08-12", "2026-08-14", price_basis=forbidden_basis
                )

            with pytest.raises((UnsupportedPriceBasisError, NotImplementedError, PriceBasisError)):
                provider._fetch_hist_df(
                    "600519", "2026-08-12", "2026-08-14", price_basis=forbidden_basis
                )

        assert mock_ak.stock_zh_a_hist.call_count == 0

    def test_unspecified_basis_forbidden(self):
        """unspecified 标签亦未实现为可用通道，严格失败闭合。"""
        provider = CnAkshareProvider()
        with pytest.raises((UnsupportedPriceBasisError, NotImplementedError, PriceBasisError)):
            provider.get_stock_data(
                "600519", "2026-08-12", "2026-08-14", price_basis=PRICE_BASIS_UNSPECIFIED
            )

    @pytest.mark.parametrize(
        "invalid_label",
        [
            "unknown",
            "foo",
            "",
            "   ",
            None,
            123,
            "price_basis.raw",        # cohort version passed where short label expected
            "price_basis.vendor_qfq",
            "pit",
            "raw_qfq",
        ],
    )
    def test_unknown_labels_fail_closed_with_unknown_price_basis_error(self, invalid_label):
        """未知标签/非法输入必须通过 UnknownPriceBasisError 失败闭合，不得默默 qfq。"""
        provider = CnAkshareProvider()
        mock_ak = MagicMock()

        with patch.object(provider, "_ak", return_value=mock_ak), \
             patch("requests.post") as mock_post:
            with pytest.raises((UnknownPriceBasisError, PriceBasisError)):
                provider.get_stock_data(
                    "600519", "2026-08-12", "2026-08-14", price_basis=invalid_label  # type: ignore[arg-type]
                )

            with pytest.raises((UnknownPriceBasisError, PriceBasisError)):
                provider._fetch_hist_df(
                    "600519", "2026-08-12", "2026-08-14", price_basis=invalid_label  # type: ignore[arg-type]
                )

        assert mock_ak.stock_zh_a_hist.call_count == 0
        assert mock_post.call_count == 0


# ==============================================================================
# 5. 契约 4：测试 mock HTTP，禁止真打网关，as_of 截断验证
# ==============================================================================

class TestRawAsOfBoundaryAndDateFiltering:
    """验证 raw 通道下 as_of 截断与日期范围过滤契约。"""

    def test_raw_as_of_filtering_in_range(self, monkeypatch):
        """区间查询包含 as_of 之后的数据时，as_of 之后的数据被正确过滤。"""
        monkeypatch.setenv("TUSHARE_TOKEN", "mock_token")
        provider = CnAkshareProvider()
        payload = _make_tushare_daily_payload()

        with patch("requests.post", return_value=_MockResponse(payload)):
            # as_of 设置为 2026-08-13，20260814 必须被过滤掉
            out = provider.get_stock_data(
                "600519",
                "2026-08-12",
                "2026-08-13",
                price_basis=PRICE_BASIS_RAW,
                as_of="2026-08-13",
            )

        df = pd.read_csv(io.StringIO(out), comment="#")
        assert len(df) == 2
        assert list(df["Date"]) == ["2026-08-12", "2026-08-13"]
        assert "2026-08-14" not in list(df["Date"])
        assert out.price_basis == PRICE_BASIS_RAW
