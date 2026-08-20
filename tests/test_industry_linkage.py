"""产业链数据层核心综合单元测试套件 (DAV-201 M5 / DAV-196).

本模块按照施工计划 M5 阶段要求，为产业链数据层 (Industry Linkage MVP) 提供综合确定性单元测试：
1. `test_get_industry_linkage_consumer_electronics`: 消费电子行业数据采集、指标完整性与 Prompt 渲染验证；
2. `test_get_industry_linkage_new_energy`: 新能源车行业数据采集、缺失与手动标注指标验证；
3. `test_cache`: 1 小时内存 TTL 缓存机制、缓存清理、防御性拷贝与多线程并发安全；
4. `test_unknown_industry`: 未配置行业安全返回 None、网络异常优雅降级及空值边界容错；
5. `test_data_collector_integration`: DataCollector 股票到行业映射、数据注入与全流程采集缓存集成。

设计规范：
- 离线确定性：所有涉及外部网络与数据接口 (akshare / yfinance / DataCollector) 均具备完备 Mock，确保 CI 环境 100% 通过；
- 防前视纪律：验证 as_of 参数过滤机制与截止日期截断；
- 零虚构原则：严格核查未接入指标与手动指标的结构化缺失标注。
"""

import concurrent.futures
import copy
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.dataflows.industry_linkage import (
    INDUSTRY_LINKAGE_MAP,
    IndustryLinkage,
    IndustryLinkageIndicator,
    format_industry_linkage_for_prompt,
    get_industry_linkage_config,
    list_supported_industries,
)
from tradingagents.dataflows.providers.industry_linkage_provider import (
    IndustryLinkageProvider,
)
from tradingagents.graph.data_collector import (
    DataCollector,
    _fetch_all,
    _map_stock_to_industry,
)


# ---------------------------------------------------------------------------
# 测试 Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_copper_dataframe() -> pd.DataFrame:
    """构造包含 70 个交易日的合成 LME 铜价日行情 DataFrame。"""
    dates = pd.date_range("2026-05-01", periods=70, freq="B")
    # 模拟铜价从 8500 稳步上涨到 9123.50
    prices = [8500.0 + (i * 9.0) for i in range(69)] + [9123.50]
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": [p + 50.0 for p in prices],
            "low": [p - 50.0 for p in prices],
            "close": prices,
            "volume": [10000] * 70,
            "position": [0] * 70,
            "s": [0] * 70,
        }
    )


@pytest.fixture
def mock_samsung_dataframe() -> pd.DataFrame:
    """构造包含 70 个交易日的合成三星电子股价 DataFrame。"""
    dates = pd.date_range("2026-05-01", periods=70, freq="B")
    # 模拟三星电子股价从 58000 震荡下行至 52000.0
    prices = [58000.0 - (i * 85.0) for i in range(69)] + [52000.0]
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p + 200.0 for p in prices],
            "Low": [p - 200.0 for p in prices],
            "Close": prices,
            "Volume": [500000] * 70,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# 5 个核心测试用例 (Mandatory Test Cases for DAV-201 M5)
# ---------------------------------------------------------------------------


def test_get_industry_linkage_consumer_electronics(
    mock_copper_dataframe: pd.DataFrame, mock_samsung_dataframe: pd.DataFrame
):
    """测试消费电子行业数据采集与指标完整性 (M5 核心用例 1).

    验证点：
    1. 成功匹配并获取 '消费电子/半导体显示' 配置；
    2. 上游成本端：LME铜价实时采集、最新值 (9123.50)、月环比 (MoM) 与季度环比 (QoQ)、趋势 (上升)、置信度 (高)；
    3. 下游需求端：全球智能手机出货量标注为 '手动'，当前值为 None，趋势为 '数据缺失'；
    4. 国际对标：三星电子股价实时采集 (52000.00 韩元)、趋势 (下降)、置信度 (高)；
    5. 行业政策催化关键词包含 '消费品以旧换新'、'超高清视频产业发展'、'新型显示产业支持政策'；
    6. 验证 format_industry_linkage_for_prompt 能正确将该结构渲染为 Prompt 文本。
    """
    provider = IndustryLinkageProvider()

    with patch("akshare.futures_foreign_hist", return_value=mock_copper_dataframe), \
         patch("yfinance.Ticker") as mock_yf:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_samsung_dataframe
        mock_yf.return_value = mock_ticker_instance

        data = provider.get_industry_linkage("消费电子", as_of="2026-08-20", use_cache=False)

        assert data is not None, "消费电子产业链数据采集结果不应为 None"
        assert data["industry_name"] == "消费电子/半导体显示"
        assert data["as_of"] == "2026-08-20"
        assert "消费品以旧换新" in data["policy_catalysts"]
        assert "超高清视频产业发展" in data["policy_catalysts"]
        assert "新型显示产业支持政策" in data["policy_catalysts"]

        # 1. 验证上游成本端：LME铜价
        upstream_list = data["upstream_cost"]
        assert len(upstream_list) == 1
        copper = upstream_list[0]
        assert copper["name"] == "LME铜价"
        assert copper["source"] == "akshare"
        assert copper["current_value"] == 9123.5
        assert copper["unit"] == "美元/吨"
        assert copper["role"] == "upstream"
        assert copper["status"] == "active"
        assert copper["confidence"] == "高"
        assert copper["trend"] == "上升"
        assert copper["mom_change"] is not None and copper["mom_change"] > 0
        assert copper["qoq_change"] is not None and copper["qoq_change"] > 0
        assert "原材料成本传导" in (copper.get("transmission_logic") or "")

        # 2. 验证下游需求端：全球智能手机出货量 (手动标注)
        downstream_list = data["downstream_demand"]
        assert len(downstream_list) == 1
        phone = downstream_list[0]
        assert phone["name"] == "全球智能手机出货量"
        assert phone["source"] == "manual"
        assert phone["current_value"] is None
        assert phone["trend"] == "数据缺失"
        assert phone["note"] == "手动"
        assert phone["status"] == "manual"
        assert phone["confidence"] == "低（待手动录入）"
        assert "景气度验证" in (phone.get("transmission_logic") or "")

        # 3. 验证国际对标：三星电子股价
        benchmark_list = data["international_benchmark"]
        assert len(benchmark_list) == 1
        samsung = benchmark_list[0]
        assert samsung["name"] == "三星电子股价"
        assert samsung["source"] == "yfinance"
        assert samsung["symbol"] == "005930.KS"
        assert samsung["current_value"] == 52000.0
        assert samsung["unit"] == "韩元"
        assert samsung["role"] == "benchmark"
        assert samsung["status"] == "active"
        assert samsung["confidence"] == "高"
        assert samsung["trend"] == "下降"
        assert samsung["mom_change"] is not None and samsung["mom_change"] < 0
        assert "龙头估值与景气度对标" in (samsung.get("transmission_logic") or "")

        # 4. 验证 Prompt 渲染
        prompt_text = format_industry_linkage_for_prompt(data)
        assert "【产业链联想数据】：消费电子/半导体显示" in prompt_text
        assert "LME铜价：9123.50 美元/吨" in prompt_text
        assert "三星电子股价：52000.00 韩元" in prompt_text
        assert "【数据缺失】全球智能手机出货量：手动" in prompt_text
        assert "消费品以旧换新" in prompt_text


def test_get_industry_linkage_new_energy():
    """测试新能源车行业数据采集与缺失/手动标注 (M5 核心用例 2).

    验证点：
    1. 成功匹配并获取 '新能源车/动力电池' 配置；
    2. 上游成本端：碳酸锂价格明确标注为 '待接入API' (pending_api)，current_value 为 None，置信度 '低（待接入API）'，不产生异常；
    3. 下游需求端：新能源车渗透率标注为 '手动'，current_value 为 None，趋势 '数据缺失'；
    4. 国际对标：特斯拉交付量标注为 '手动'，current_value 为 None，趋势 '数据缺失'；
    5. 行业政策催化关键词包含 '新能源汽车购置税减免'、'车路云一体化试点'、'充换电基础设施建设支持'；
    6. 验证 Prompt 渲染中所有未接入/手动指标均带有显式【数据缺失】标识，严禁杜撰虚构。
    """
    provider = IndustryLinkageProvider()
    data = provider.get_industry_linkage("新能源车", as_of="2026-08-20", use_cache=False)

    assert data is not None, "新能源车产业链数据采集结果不应为 None"
    assert data["industry_name"] == "新能源车/动力电池"
    assert data["as_of"] == "2026-08-20"
    assert "新能源汽车购置税减免" in data["policy_catalysts"]
    assert "车路云一体化试点" in data["policy_catalysts"]
    assert "充换电基础设施建设支持" in data["policy_catalysts"]

    # 1. 验证上游成本端：碳酸锂价格（待接入API）
    upstream_list = data["upstream_cost"]
    assert len(upstream_list) == 1
    lithium = upstream_list[0]
    assert lithium["name"] == "碳酸锂价格"
    assert lithium["source"] == "pending_api"
    assert lithium["current_value"] is None
    assert lithium["unit"] == "万元/吨"
    assert lithium["trend"] == "数据缺失"
    assert lithium["confidence"] == "低（待接入API）"
    assert lithium["note"] == "待接入API"
    assert lithium["status"] == "pending_api"
    assert "动力电池正极核心原材料成本传导" in (lithium.get("transmission_logic") or "")

    # 2. 验证下游需求端：新能源车渗透率（手动标注）
    downstream_list = data["downstream_demand"]
    assert len(downstream_list) == 1
    nev_penetration = downstream_list[0]
    assert nev_penetration["name"] == "新能源车渗透率"
    assert nev_penetration["source"] == "manual"
    assert nev_penetration["current_value"] is None
    assert nev_penetration["unit"] == "%"
    assert nev_penetration["trend"] == "数据缺失"
    assert nev_penetration["confidence"] == "低（待手动录入）"
    assert nev_penetration["note"] == "手动"
    assert nev_penetration["status"] == "manual"

    # 3. 验证国际对标：特斯拉交付量（手动标注）
    benchmark_list = data["international_benchmark"]
    assert len(benchmark_list) == 1
    tesla = benchmark_list[0]
    assert tesla["name"] == "特斯拉交付量"
    assert tesla["symbol"] == "TSLA"
    assert tesla["current_value"] is None
    assert tesla["unit"] == "辆"
    assert tesla["trend"] == "数据缺失"
    assert tesla["confidence"] == "低（待手动录入）"
    assert tesla["note"] == "手动"
    assert tesla["status"] == "manual"

    # 4. 验证 Prompt 渲染文本
    prompt_text = format_industry_linkage_for_prompt(data)
    assert "【产业链联想数据】：新能源车/动力电池" in prompt_text
    assert "【数据缺失】碳酸锂价格：待接入API" in prompt_text
    assert "【数据缺失】新能源车渗透率：手动" in prompt_text
    assert "【数据缺失】特斯拉交付量：手动" in prompt_text
    assert "新能源汽车购置税减免" in prompt_text


def test_cache(mock_copper_dataframe: pd.DataFrame):
    """测试 1 小时内存 TTL 缓存机制与并发安全 (M5 核心用例 3).

    验证点：
    1. 首次调用执行实际采集并建立缓存；
    2. 第二次相同参数调用直接命中缓存，不重复请求底层数据源；
    3. 返回对象具备深拷贝隔离性，外部修改不污染缓存内部数据；
    4. clear_cache() 能正确清空缓存；
    5. 多线程并发请求下无死锁、无竞态条件，所有线程均获得完整数据；
    6. TTL 超时后缓存自动失效并重新拉取。
    """
    provider = IndustryLinkageProvider(cache_ttl=3600)

    with patch("akshare.futures_foreign_hist", return_value=mock_copper_dataframe) as mock_ak, \
         patch("yfinance.Ticker") as mock_yf:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = Exception("Offline in test")
        mock_yf.return_value = mock_ticker_instance

        # 1. 首次调用
        res1 = provider.get_industry_linkage("消费电子", as_of="2026-08-20")
        assert res1 is not None
        assert mock_ak.call_count == 1
        cached_ts = res1["cached_at"]

        # 2. 第二次调用（验证命中缓存）
        res2 = provider.get_industry_linkage("消费电子", as_of="2026-08-20")
        assert res2 is not None
        assert mock_ak.call_count == 1  # 未发生二次调用
        assert res2["cached_at"] == cached_ts
        assert res2["upstream_cost"][0]["current_value"] == 9123.5

        # 3. 验证深拷贝隔离性
        res2["upstream_cost"][0]["current_value"] = 99999.99
        res3 = provider.get_industry_linkage("消费电子", as_of="2026-08-20")
        assert res3["upstream_cost"][0]["current_value"] == 9123.5  # 缓存未被篡改

        # 4. 验证 clear_cache()
        provider.clear_cache()
        assert len(provider._cache) == 0
        assert len(provider._cache_timestamps) == 0

        # 5. 清空后再次调用将重新触发数据采集
        res4 = provider.get_industry_linkage("消费电子", as_of="2026-08-20")
        assert res4 is not None
        assert mock_ak.call_count == 2

    # 6. 验证短 TTL 超时失效机制
    short_ttl_provider = IndustryLinkageProvider(cache_ttl=1)  # 1秒 TTL
    with patch("akshare.futures_foreign_hist", return_value=mock_copper_dataframe) as mock_ak_short, \
         patch("yfinance.Ticker") as mock_yf_short:
        mock_yf_short.return_value.history.side_effect = Exception("Offline")
        r1 = short_ttl_provider.get_industry_linkage("消费电子")
        assert mock_ak_short.call_count == 1

        # 手动将缓存时间戳往前拨 2 秒模拟过期
        for k in list(short_ttl_provider._cache_timestamps.keys()):
            short_ttl_provider._cache_timestamps[k] -= 2.0

        r2 = short_ttl_provider.get_industry_linkage("消费电子")
        assert mock_ak_short.call_count == 2  # TTL 过期重新拉取

    # 7. 验证多线程高并发安全性
    concurrent_provider = IndustryLinkageProvider(cache_ttl=3600)
    with patch("akshare.futures_foreign_hist", return_value=mock_copper_dataframe), \
         patch("yfinance.Ticker") as mock_yf_conc:
        mock_yf_conc.return_value.history.side_effect = Exception("Offline")

        def worker(industry_name: str):
            return concurrent_provider.get_industry_linkage(industry_name, as_of="2026-08-20")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            targets = ["消费电子", "新能源车"] * 10
            futures = [executor.submit(worker, ind) for ind in targets]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        assert all(r is not None for r in results)
        assert len(concurrent_provider._cache) == 2


def test_unknown_industry():
    """测试未配置行业安全返回与异常优雅降级 (M5 核心用例 4).

    验证点：
    1. 查询未配置行业（如 '未知行业'、'房地产'、'银行业'）安全返回 None，不抛出异常；
    2. 空字符串、纯空白、非法类型 (None, 数字) 安全返回 None；
    3. 外部数据源（akshare / yfinance）网络超时、429 报错或接口崩溃时，Provider 自动捕获并降级为结构化缺失状态；
    4. format_industry_linkage_for_prompt 面对空数据、None 与非预期输入安全返回空字符串。
    """
    provider = IndustryLinkageProvider()

    # 1. 未配置或非法行业输入
    assert provider.get_industry_linkage("未知行业") is None
    assert provider.get_industry_linkage("房地产") is None
    assert provider.get_industry_linkage("银行业") is None
    assert provider.get_industry_linkage("") is None
    assert provider.get_industry_linkage("   ") is None
    assert provider.get_industry_linkage(None) is None  # type: ignore
    assert provider.get_industry_linkage(12345) is None  # type: ignore

    # 2. 外部接口全面抛出异常时的优雅降级保护
    with patch("akshare.futures_foreign_hist", side_effect=TimeoutError("Connection timed out (mock)")), \
         patch("yfinance.Ticker", side_effect=Exception("Rate limited 429 (mock)")):

        data = provider.get_industry_linkage("消费电子", use_cache=False)

        assert data is not None
        assert data["industry_name"] == "消费电子/半导体显示"

        # LME铜价在异常时安全降级
        copper = data["upstream_cost"][0]
        assert copper["current_value"] is None
        assert copper["trend"] == "数据缺失"
        assert copper["confidence"] == "低（接口异常）"
        assert "Connection timed out" in (copper.get("note") or "")

        # 三星电子股价在异常时安全降级
        samsung = data["international_benchmark"][0]
        assert samsung["current_value"] is None
        assert samsung["trend"] == "数据缺失"
        assert samsung["confidence"] == "低（接口异常）"
        assert "Rate limited" in (samsung.get("note") or "")

        # 降级后的数据格式化依旧能安全输出
        prompt_text = format_industry_linkage_for_prompt(data)
        assert "【产业链联想数据】：消费电子/半导体显示" in prompt_text
        assert "【数据缺失】LME铜价" in prompt_text
        assert "【数据缺失】三星电子股价" in prompt_text

    # 3. 边界输入格式化防护
    assert format_industry_linkage_for_prompt(None) == ""
    assert format_industry_linkage_for_prompt({}) == ""
    assert format_industry_linkage_for_prompt({"industry_name": ""}) == ""
    assert format_industry_linkage_for_prompt("invalid_type") == ""  # type: ignore


def test_data_collector_integration():
    """测试 DataCollector 股票映射与产业链数据注入全流程 (M5 核心用例 5).

    验证点：
    1. _map_stock_to_industry 映射逻辑（消费电子标的、新能源车标的、未映射银行白酒标的、大小写与代码后缀容错）；
    2. DataCollector 实例初始化与 industry_linkage_provider 依赖注入；
    3. _fetch_all 针对已映射标的（京东方A、宁德时代）正确注入 industry_linkage；
    4. _fetch_all 针对未映射标的（招商银行）将 industry_linkage 置为 None 且不发起多余调用；
    5. _fetch_all 兼容 YYYY-MM-DD 与 YYYYMMDD 两种日期格式；
    6. DataCollector.collect() 具备全局缓存与深拷贝安全。
    """
    # 1. 股票代码映射验证
    # 消费电子
    assert _map_stock_to_industry("000725.SZ") == "消费电子"  # 京东方A
    assert _map_stock_to_industry("000725") == "消费电子"
    assert _map_stock_to_industry("000725.sz") == "消费电子"
    assert _map_stock_to_industry("000100.SZ") == "消费电子"  # TCL科技
    assert _map_stock_to_industry("002475.SZ") == "消费电子"  # 立讯精密
    assert _map_stock_to_industry("002241.SZ") == "消费电子"  # 歌尔股份
    assert _map_stock_to_industry("300433.SZ") == "消费电子"  # 蓝思科技

    # 新能源车
    assert _map_stock_to_industry("300750.SZ") == "新能源车"  # 宁德时代
    assert _map_stock_to_industry("300750") == "新能源车"
    assert _map_stock_to_industry("300750.sz") == "新能源车"
    assert _map_stock_to_industry("002594.SZ") == "新能源车"  # 比亚迪
    assert _map_stock_to_industry("601633.SH") == "新能源车"  # 长城汽车
    assert _map_stock_to_industry("002460.SZ") == "新能源车"  # 赣锋锂业
    assert _map_stock_to_industry("002466.SZ") == "新能源车"  # 天齐锂业

    # 未映射股票与非法输入
    assert _map_stock_to_industry("600036.SH") is None  # 招商银行
    assert _map_stock_to_industry("600519.SH") is None  # 贵州茅台
    assert _map_stock_to_industry("000001.SZ") is None  # 平安银行
    assert _map_stock_to_industry(None) is None
    assert _map_stock_to_industry("") is None
    assert _map_stock_to_industry(12345) is None  # type: ignore

    # 2. DataCollector 初始化与依赖注入
    collector = DataCollector()
    assert hasattr(collector, "industry_linkage_provider")
    assert isinstance(collector.industry_linkage_provider, IndustryLinkageProvider)

    custom_provider = IndustryLinkageProvider(cache_ttl=1800)
    injected_collector = DataCollector(industry_linkage_provider=custom_provider)
    assert injected_collector.industry_linkage_provider is custom_provider

    # 3. _fetch_all 对消费电子标的采集注入
    mock_linkage_payload = {
        "industry_name": "消费电子/半导体显示",
        "upstream_cost": [{"name": "LME铜价", "current_value": 9123.5}],
        "downstream_demand": [{"name": "全球智能手机出货量", "trend": "数据缺失"}],
        "international_benchmark": [{"name": "三星电子股价", "current_value": 52000.0}],
        "policy_catalysts": ["消费品以旧换新"],
        "description": "消费电子产业链",
        "as_of": "2026-08-20",
    }

    mock_provider = MagicMock(spec=IndustryLinkageProvider)
    mock_provider.get_industry_linkage.return_value = mock_linkage_payload

    with patch("tradingagents.graph.data_collector._safe", return_value="dummy_data"):
        # 消费电子 (京东方A)
        res_boe = _fetch_all("000725.SZ", "2026-08-20", industry_provider=mock_provider)
        assert "industry_linkage" in res_boe
        assert res_boe["industry_linkage"] is not None
        assert res_boe["industry_linkage"]["industry_name"] == "消费电子/半导体显示"
        mock_provider.get_industry_linkage.assert_called_with("消费电子", as_of="2026-08-20")

        # 4. _fetch_all 对未映射标的 (招商银行)
        mock_provider.reset_mock()
        res_cmb = _fetch_all("600036.SH", "2026-08-20", industry_provider=mock_provider)
        assert "industry_linkage" in res_cmb
        assert res_cmb["industry_linkage"] is None
        mock_provider.get_industry_linkage.assert_not_called()

        # 5. 日期格式归一化 (YYYYMMDD -> 2026-08-20)
        mock_provider.reset_mock()
        res_date_norm = _fetch_all("000725.SZ", "20260820", industry_provider=mock_provider)
        assert res_date_norm["industry_linkage"] is not None
        mock_provider.get_industry_linkage.assert_called_with("消费电子", as_of="2026-08-20")

    # 6. DataCollector.collect() 缓存与深拷贝
    collector_instance = DataCollector()
    stub_pool = {
        "stock_data": "mock_stock",
        "indicators": {},
        "industry_linkage": copy.deepcopy(mock_linkage_payload),
    }

    with patch("tradingagents.graph.data_collector._fetch_all", return_value=stub_pool) as mock_fetch:
        coll_res1 = collector_instance.collect("000725.SZ", "2026-08-20")
        assert coll_res1["industry_linkage"]["industry_name"] == "消费电子/半导体显示"

        # 修改返回值验证深拷贝防护
        coll_res1["industry_linkage"]["upstream_cost"][0]["current_value"] = 88888.88

        coll_res2 = collector_instance.collect("000725.SZ", "2026-08-20")
        assert mock_fetch.call_count == 1  # 命中内存缓存
        assert coll_res2["industry_linkage"]["upstream_cost"][0]["current_value"] == 9123.5


# ---------------------------------------------------------------------------
# 补充测试类 (Structural & Boundary Checks)
# ---------------------------------------------------------------------------


class TestIndustryLinkageSuite:
    """产业链数据层完整性与边界回归测试类。"""

    def test_linkage_map_and_helper_functions(self):
        """测试 INDUSTRY_LINKAGE_MAP 配置结构与辅助查询函数。"""
        supported = list_supported_industries()
        assert "消费电子" in supported
        assert "新能源车" in supported

        ce_config = get_industry_linkage_config("消费电子")
        assert ce_config is not None
        assert ce_config.industry_name == "消费电子/半导体显示"

        # 模糊匹配测试
        ce_fuzzy = get_industry_linkage_config("消费电子/半导体显示")
        assert ce_fuzzy is not None
        assert ce_fuzzy.industry_name == "消费电子/半导体显示"

        unknown_config = get_industry_linkage_config("未知赛道")
        assert unknown_config is None

    def test_as_of_lookahead_filtering(self, mock_copper_dataframe: pd.DataFrame):
        """测试 as_of 截止日期过滤，防止未来数据泄露。"""
        provider = IndustryLinkageProvider()
        cutoff_date = mock_copper_dataframe.iloc[25]["date"].strftime("%Y-%m-%d")
        expected_price = float(mock_copper_dataframe.iloc[25]["close"])

        ind = IndustryLinkageIndicator(
            name="LME铜价",
            source="akshare",
            symbol="铜",
        )

        with patch("akshare.futures_foreign_hist", return_value=mock_copper_dataframe):
            res = provider._fetch_indicator(ind, as_of=cutoff_date)
            assert res["current_value"] == expected_price
            assert res["confidence"] == "高"

    def test_pydantic_model_format_prompt(self):
        """测试直接传入 Pydantic IndustryLinkage 对象给 Prompt 格式化函数。"""
        config = get_industry_linkage_config("消费电子")
        assert config is not None
        text = format_industry_linkage_for_prompt(config)
        assert "【产业链联想数据】：消费电子/半导体显示" in text
        assert "LME铜价" in text

    def test_calculate_series_metrics_variations(self):
        """测试 _calculate_series_metrics 在各种时序样本、平稳趋势与缺失列下的健壮性。"""
        provider = IndustryLinkageProvider()

        # 1. 空 DataFrame 或缺少列
        assert provider._calculate_series_metrics(None) is None  # type: ignore
        assert provider._calculate_series_metrics(pd.DataFrame()) is None
        assert provider._calculate_series_metrics(pd.DataFrame({"invalid_col": [1, 2, 3]})) is None
        assert provider._calculate_series_metrics(pd.DataFrame({"date": ["2026-08-01"]})) is None

        # 2. 平稳趋势判定 (变动在 -1.0% ~ 1.0% 之间)
        dates = pd.date_range("2026-05-01", periods=30, freq="B")
        flat_prices = [100.0 + (0.01 * i) for i in range(30)]  # 涨幅极小 (+0.29%)
        flat_df = pd.DataFrame({"date": dates, "close": flat_prices})
        flat_metrics = provider._calculate_series_metrics(flat_df)
        assert flat_metrics is not None
        assert flat_metrics["trend"] == "平稳"

        # 3. 样本量较小 (< 22 但 >= 2)
        short_dates = pd.date_range("2026-08-10", periods=5, freq="B")
        short_prices = [100.0, 102.0, 104.0, 106.0, 110.0]  # +10.0%
        short_df = pd.DataFrame({"date": short_dates, "close": short_prices})
        short_metrics = provider._calculate_series_metrics(short_df)
        assert short_metrics is not None
        assert short_metrics["mom_change"] == 10.0
        assert short_metrics["trend"] == "上升"

        # 4. as_of 过滤致空或异常解析
        assert provider._calculate_series_metrics(short_df, as_of="2020-01-01") is None
        res_bad_date = provider._calculate_series_metrics(short_df, as_of="invalid-date-format")
        assert res_bad_date is not None

    def test_provider_handles_empty_dataframe_returns(self):
        """测试外部接口返回空 DataFrame 时指标优雅降级。"""
        provider = IndustryLinkageProvider()

        # akshare 返回空 DataFrame
        with patch("akshare.futures_foreign_hist", return_value=pd.DataFrame()):
            copper_res = provider._fetch_indicator(
                IndustryLinkageIndicator(name="LME铜价", source="akshare", symbol="铜")
            )
            assert copper_res["current_value"] is None
            assert copper_res["trend"] == "数据缺失"
            assert copper_res["confidence"] == "低（数据源为空）"

        # yfinance 返回空 DataFrame
        with patch("yfinance.Ticker") as mock_yf:
            mock_yf.return_value.history.return_value = pd.DataFrame()
            samsung_res = provider._fetch_indicator(
                IndustryLinkageIndicator(name="三星电子股价", source="yfinance", symbol="005930.KS")
            )
            assert samsung_res["current_value"] is None
            assert samsung_res["trend"] == "数据缺失"
            assert samsung_res["confidence"] == "低（数据源为空）"

    def test_fetch_indicator_unsupported_config_fallback(self):
        """测试自定义未支持指标配置的安全降级逻辑。"""
        provider = IndustryLinkageProvider()
        unknown_ind = IndustryLinkageIndicator(
            name="稀土永磁指数",
            source="custom_vendor",
            status="active",
        )
        res = provider._fetch_indicator(unknown_ind)
        assert res["current_value"] is None
        assert res["trend"] == "数据缺失"
        assert res["confidence"] == "低（待实现）"

