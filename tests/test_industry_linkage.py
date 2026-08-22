"""产业链数据层核心综合单元测试套件 (DAV-201 / DAV-256 / DAV-274).

本模块为产业链数据层 (Industry Linkage 27 行业全景) 提供综合确定性单元测试：
1. `test_get_industry_linkage_consumer_electronics`: 消费电子行业数据采集、指标完整性与 Prompt 渲染验证；
2. `test_get_industry_linkage_new_energy`: 新能源汽车行业数据采集、缺失与手动标注指标验证；
3. `test_cache`: 1 小时内存 TTL 缓存机制、缓存清理、防御性拷贝与多线程并发安全；
4. `test_unknown_industry`: 未配置行业安全返回 None、网络异常优雅降级及空值边界容错；
5. `test_data_collector_integration`: DataCollector 股票到行业映射、数据注入与全流程采集缓存集成；
6. `test_all_twenty_seven_industries_return_non_empty_linkage`: 验证 27 个行业全部返回非空结构，无 API 指标显式缺失，零虚构值；
7. `test_data_collector_maps_required_six_stocks`: 验证原 6 只核心股票映射严格不回归；
8. `test_data_collector_maps_all_twenty_seven_industries`: 验证 27 个行业均有至少 1 只 A 股代表股票映射。
"""

import concurrent.futures
import copy
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.dataflows.industry_linkage import (
    DEFAULT_INDUSTRY_LINKAGE_MISSING_PROMPT,
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
from tradingagents.knowledge.industry_linkage import (
    get_all_industry_names,
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
# 5 个核心测试用例
# ---------------------------------------------------------------------------


def test_get_industry_linkage_consumer_electronics(
    mock_copper_dataframe: pd.DataFrame, mock_samsung_dataframe: pd.DataFrame
):
    """测试消费电子行业数据采集与指标完整性 (核心用例 1).

    验证点：
    1. 成功匹配并获取 '消费电子与智能终端' 配置；
    2. 上游成本端：LME铜价实时采集、最新值 (9123.50)、月环比 (MoM) 与季度环比 (QoQ)、趋势 (上升)、置信度 (高)；
    3. 下游需求端：全球智能手机出货量标注为 '手动'，当前值为 None，趋势为 '数据缺失'；
    4. 国际对标：三星电子股价实时采集 (52000.00 韩元)、趋势 (下降)、置信度 (高)；
    5. 行业政策催化关键词包含 '消费品以旧换新补贴政策'；
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
        assert data["industry_name"] == "消费电子与智能终端"
        assert data["as_of"] == "2026-08-20"
        assert any("以旧换新" in cat for cat in data["policy_catalysts"])

        # 1. 验证上游成本端：LME铜价
        upstream_list = data["upstream_cost"]
        assert len(upstream_list) >= 1
        copper = [u for u in upstream_list if u["name"] == "LME铜价"][0]
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
        assert len(downstream_list) >= 1
        phone = [d for d in downstream_list if d["name"] == "全球智能手机出货量"][0]
        assert phone["source"] == "manual"
        assert phone["current_value"] is None
        assert phone["trend"] == "数据缺失"
        assert phone["note"] == "手动"
        assert phone["status"] == "manual"
        assert phone["confidence"] == "低（待手动录入）"
        assert "景气度验证" in (phone.get("transmission_logic") or "")

        # 3. 验证国际对标：三星电子股价
        benchmark_list = data["international_benchmark"]
        assert len(benchmark_list) >= 1
        samsung = [b for b in benchmark_list if "三星电子" in b["name"]][0]
        assert samsung["source"] == "yfinance"
        assert samsung["symbol"] == "005930.KS"
        assert samsung["current_value"] == 52000.0
        assert samsung["unit"] == "韩元"
        assert samsung["role"] == "benchmark"
        assert samsung["status"] == "active"
        assert samsung["confidence"] == "高"
        assert samsung["trend"] == "下降"
        assert samsung["mom_change"] is not None and samsung["mom_change"] < 0
        assert "对标" in (samsung.get("transmission_logic") or "")

        # 4. 验证 Prompt 渲染
        prompt_text = format_industry_linkage_for_prompt(data)
        assert "【产业链联想数据】：消费电子与智能终端" in prompt_text
        assert "LME铜价：9123.50 美元/吨" in prompt_text
        assert "三星电子股价：52000.00 韩元" in prompt_text
        assert "【数据缺失】全球智能手机出货量：手动" in prompt_text


def test_get_industry_linkage_new_energy(monkeypatch):
    """测试新能源汽车行业数据采集与缺失/手动标注 (核心用例 2).

    验证点：
    1. 成功匹配并获取 '新能源汽车与智能汽车' 配置；
    2. 上游成本端：碳酸锂价格对接已付费 Tushare (LC.GFE)，返回真实采集数值 158680.00 元/吨 与置信度 '高'；
    3. 下游需求端：新能源车渗透率标注为 '手动'，current_value 为 None，趋势 '数据缺失'；
    4. 国际对标：特斯拉交付量标注为 '手动'，current_value 为 None，趋势 '数据缺失'；
    5. 行业政策催化关键词包含 '新能源汽车购置税减免'；
    6. 验证 Prompt 渲染中碳酸锂包含准确价格与单位，手动指标带有显式【数据缺失】标识。
    """
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_configured")
    provider = IndustryLinkageProvider()

    dates = pd.date_range("2026-05-01", periods=70, freq="B")
    items = []
    for i, dt in enumerate(dates):
        d_str = dt.strftime("%Y%m%d")
        close = 140000.0 + (i * 270.0) if i < 69 else 158680.0
        items.append(["LC.GFE", d_str, close - 500, close + 500, close - 800, close, 15000])

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "code": 0,
        "msg": None,
        "data": {
            "fields": ["ts_code", "trade_date", "open", "high", "low", "close", "vol"],
            "items": list(reversed(items)),
        },
    }

    with patch("requests.post", return_value=mock_resp), \
         patch("yfinance.Ticker", side_effect=Exception("Offline test")):
        data = provider.get_industry_linkage("新能源车", as_of="2026-08-20", use_cache=False)

        assert data is not None, "新能源汽车产业链数据采集结果不应为 None"
        assert data["industry_name"] == "新能源汽车与智能汽车"
        assert data["as_of"] == "2026-08-20"
        assert any("购置税减免" in cat for cat in data["policy_catalysts"])

        # 1. 验证上游成本端：碳酸锂价格（Tushare 已付费源）
        upstream_list = data["upstream_cost"]
        assert len(upstream_list) >= 1
        lithium = [u for u in upstream_list if "碳酸锂" in u["name"]][0]
        assert lithium["source"] == "tushare"
        assert lithium["symbol"] == "LC.GFE"
        assert lithium["current_value"] == 158680.0
        assert lithium["unit"] == "元/吨"
        assert lithium["trend"] == "上升"
        assert lithium["confidence"] == "高"
        assert lithium["status"] == "active"
        assert "动力电池正极核心原材料成本传导" in (lithium.get("transmission_logic") or "")

        # 2. 验证下游需求端：新能源车渗透率（手动标注）
        downstream_list = data["downstream_demand"]
        assert len(downstream_list) >= 1
        nev_penetration = [d for d in downstream_list if "渗透率" in d["name"]][0]
        assert nev_penetration["source"] == "manual"
        assert nev_penetration["current_value"] is None
        assert nev_penetration["unit"] == "%"
        assert nev_penetration["trend"] == "数据缺失"
        assert nev_penetration["confidence"] == "低（待手动录入）"
        assert nev_penetration["note"] == "手动"
        assert nev_penetration["status"] == "manual"

        # 3. 验证国际对标：特斯拉交付量（手动标注）
        benchmark_list = data["international_benchmark"]
        assert len(benchmark_list) >= 1
        tesla_delivery = [b for b in benchmark_list if b["name"] == "特斯拉交付量"][0]
        assert tesla_delivery["symbol"] == "TSLA"
        assert tesla_delivery["current_value"] is None
        assert tesla_delivery["unit"] == "辆"
        assert tesla_delivery["trend"] == "数据缺失"
        assert tesla_delivery["confidence"] == "低（待手动录入）"
        assert tesla_delivery["note"] == "手动"
        assert tesla_delivery["status"] == "manual"

        # 4. 验证 Prompt 渲染文本
        prompt_text = format_industry_linkage_for_prompt(data)
        assert "【产业链联想数据】：新能源汽车与智能汽车" in prompt_text
        assert "碳酸锂价格：158680.00 元/吨" in prompt_text
        assert "【数据缺失】新能源车渗透率：手动" in prompt_text
        assert "【数据缺失】特斯拉交付量：手动" in prompt_text


def test_get_industry_linkage_commercial_bank(monkeypatch):
    """测试商业银行与信贷行业数据采集、Tushare宏观利率与 Prompt 渲染。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "mock_token_configured")
    provider = IndustryLinkageProvider()

    # 构造 shibor 数据
    dates_shibor = pd.date_range("2026-05-01", periods=70, freq="B")
    items_shibor = []
    for i, dt in enumerate(dates_shibor):
        d_str = dt.strftime("%Y%m%d")
        rate_3m = 1.60 + (i * 0.005) if i < 69 else 1.95
        items_shibor.append([d_str, 1.20, 1.35, 1.45, 1.50, rate_3m, 1.75, 1.85, 1.95])
    mock_shibor_resp = {
        "code": 0,
        "msg": None,
        "data": {
            "fields": ["date", "on", "1w", "2w", "1m", "3m", "6m", "9m", "1y"],
            "items": list(reversed(items_shibor)),
        },
    }

    # 构造 lpr 数据 (每月 20 日)
    dates_lpr = [pd.Timestamp("2024-09-20") + pd.DateOffset(months=i) for i in range(24)]
    items_lpr = []
    for i, dt in enumerate(dates_lpr):
        d_str = dt.strftime("%Y%m%d")
        rate_1y = 3.45 - (i * 0.005) if i < 23 else 3.35
        items_lpr.append([d_str, rate_1y, 3.85])
    mock_lpr_resp = {
        "code": 0,
        "msg": None,
        "data": {
            "fields": ["date", "1y", "5y"],
            "items": list(reversed(items_lpr)),
        },
    }

    def mock_post(url, json=None, **kwargs):
        mock_r = MagicMock()
        mock_r.status_code = 200
        if json and json.get("api_name") == "shibor":
            mock_r.json.return_value = mock_shibor_resp
        elif json and json.get("api_name") == "shibor_lpr":
            mock_r.json.return_value = mock_lpr_resp
        else:
            mock_r.json.return_value = {"code": 0, "msg": None, "data": {"fields": [], "items": []}}
        return mock_r

    with patch("requests.post", side_effect=mock_post), \
         patch("yfinance.Ticker", side_effect=Exception("Offline test")):

        data = provider.get_industry_linkage("商业银行与信贷", as_of="2026-08-20", use_cache=False)

        assert data is not None
        assert data["industry_name"] == "商业银行与信贷"

        # 1. 验证 Shibor 3M
        shibor = [u for u in data["upstream_cost"] if "Shibor" in u["name"]][0]
        assert shibor["source"] == "tushare"
        assert shibor["status"] == "active"
        assert shibor["current_value"] == 1.95
        assert shibor["unit"] == "%"
        assert shibor["trend"] == "上升"
        assert shibor["confidence"] == "高"
        assert shibor["transport_provider"] == "tushare"
        assert shibor["api_name"] == "shibor"
        assert shibor["value_field"] == "3m"

        # 2. 验证 LPR 1Y
        lpr = [d for d in data["downstream_demand"] if "LPR" in d["name"]][0]
        assert lpr["source"] == "tushare"
        assert lpr["status"] == "active"
        assert lpr["current_value"] == 3.35
        assert lpr["unit"] == "%"
        assert lpr["trend"] == "下降"
        assert lpr["confidence"] == "高"
        assert lpr["transport_provider"] == "tushare"
        assert lpr["api_name"] == "shibor_lpr"
        assert lpr["value_field"] == "1y"

        # 3. 验证 Prompt 渲染
        prompt_text = format_industry_linkage_for_prompt(data)
        assert "【产业链联想数据】：商业银行与信贷" in prompt_text
        assert "银行间同业拆借利率Shibor：1.95 %" in prompt_text
        assert "贷款市场报价利率LPR_1Y：3.35 %" in prompt_text


def test_cache(mock_copper_dataframe: pd.DataFrame):
    """测试 1 小时内存 TTL 缓存机制与并发安全 (核心用例 3).

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
    """测试未配置行业安全返回与异常优雅降级 (核心用例 4).

    验证点：
    1. 查询未配置行业（如 '未知行业XYZ'、'采掘服务'、'虚拟游戏'）安全返回 None，不抛出异常；
    2. 空字符串、纯空白、非法类型 (None, 数字) 安全返回 None；
    3. 外部数据源（akshare / yfinance）网络超时、429 报错或接口崩溃时，Provider 自动捕获并降级为结构化缺失状态；
    4. format_industry_linkage_for_prompt 面对空数据、None 与非预期输入安全返回空字符串。
    """
    provider = IndustryLinkageProvider()

    # 1. 未配置或非法行业输入
    assert provider.get_industry_linkage("未知行业XYZ") is None
    assert provider.get_industry_linkage("采掘服务") is None
    assert provider.get_industry_linkage("虚拟游戏") is None
    assert provider.get_industry_linkage("") is None
    assert provider.get_industry_linkage("   ") is None
    assert provider.get_industry_linkage(None) is None  # type: ignore
    assert provider.get_industry_linkage(12345) is None  # type: ignore

    # 2. 外部接口全面抛出异常时的优雅降级保护
    with patch("akshare.futures_foreign_hist", side_effect=TimeoutError("Connection timed out (mock)")), \
         patch("yfinance.Ticker", side_effect=Exception("Rate limited 429 (mock)")):

        data = provider.get_industry_linkage("消费电子", use_cache=False)

        assert data is not None
        assert data["industry_name"] == "消费电子与智能终端"

        # LME铜价在异常时安全降级
        copper = data["upstream_cost"][0]
        assert copper["current_value"] is None
        assert copper["trend"] == "数据缺失"
        assert copper["confidence"] == "低（接口异常）"
        assert "Connection timed out" in (copper.get("note") or "")

        # 三星电子股价在异常时安全降级
        samsung = [b for b in data["international_benchmark"] if "三星电子" in b["name"]][0]
        assert samsung["current_value"] is None
        assert samsung["trend"] == "数据缺失"
        assert samsung["confidence"] == "低（接口异常）"
        assert "Rate limited" in (samsung.get("note") or "")

        # 降级后的数据格式化依旧能安全输出
        prompt_text = format_industry_linkage_for_prompt(data)
        assert "【产业链联想数据】：消费电子与智能终端" in prompt_text
        assert "【数据缺失】LME铜价" in prompt_text
        assert "【数据缺失】三星电子股价" in prompt_text

    # 3. 边界输入格式化防护 (fail-closed 契约)
    assert format_industry_linkage_for_prompt(None) == DEFAULT_INDUSTRY_LINKAGE_MISSING_PROMPT
    assert format_industry_linkage_for_prompt({}) == DEFAULT_INDUSTRY_LINKAGE_MISSING_PROMPT
    assert format_industry_linkage_for_prompt({"industry_name": ""}) == DEFAULT_INDUSTRY_LINKAGE_MISSING_PROMPT
    assert format_industry_linkage_for_prompt("invalid_type") == DEFAULT_INDUSTRY_LINKAGE_MISSING_PROMPT


def test_data_collector_integration():
    """测试 DataCollector 股票映射与产业链数据注入全流程 (核心用例 5).

    验证点：
    1. _map_stock_to_industry 映射逻辑（包含原有 6 只标的、新行业标的及未映射标的）；
    2. DataCollector 实例初始化与 industry_linkage_provider 依赖注入；
    3. _fetch_all 针对已映射标的正确注入 industry_linkage；
    4. _fetch_all 针对未映射标的将 industry_linkage 置为 None 且不发起多余调用；
    5. _fetch_all 兼容 YYYY-MM-DD 与 YYYYMMDD 两种日期格式；
    6. DataCollector.collect() 具备全局缓存与深拷贝安全。
    """
    # 1. 股票代码映射验证
    # 原有 6 只核心标的严格验证
    assert _map_stock_to_industry("688981.SH") == "半导体"  # 中芯国际
    assert _map_stock_to_industry("603501.SH") == "半导体"  # 韦尔股份
    assert _map_stock_to_industry("601857.SH") == "石油化工"  # 中国石油
    assert _map_stock_to_industry("600309.SH") == "石油化工"  # 万华化学
    assert _map_stock_to_industry("600036.SH") == "金融地产"  # 招商银行
    assert _map_stock_to_industry("000002.SZ") == "金融地产"  # 万科A

    # 消费电子 & 新能源车标的
    assert _map_stock_to_industry("000725.SZ") == "消费电子"  # 京东方A
    assert _map_stock_to_industry("300750.SZ") == "新能源车"  # 宁德时代

    # 新增行业标的
    assert _map_stock_to_industry("600519.SH") == "白酒与精制茶酒"  # 贵州茅台
    assert _map_stock_to_industry("600030.SH") == "证券公司与资本市场"  # 中信证券
    assert _map_stock_to_industry("601318.SH") == "保险与多元金融"  # 中国平安
    assert _map_stock_to_industry("601088.SH") == "煤炭与传统化石能源"  # 中国神华
    assert _map_stock_to_industry("600900.SH") == "电力与公用事业"  # 长江电力
    assert _map_stock_to_industry("600019.SH") == "钢铁与黑色金属"  # 宝钢股份
    assert _map_stock_to_industry("601899.SH") == "有色金属与工业金属"  # 紫金矿业
    assert _map_stock_to_industry("600547.SH") == "贵金属与稀缺资源"  # 山东黄金
    assert _map_stock_to_industry("002714.SZ") == "农林牧渔与生猪养殖"  # 牧原股份

    # 未映射股票与非法输入
    assert _map_stock_to_industry("UNKNOWN_999999") is None
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
        "industry_name": "消费电子与智能终端",
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
        assert res_boe["industry_linkage"]["industry_name"] == "消费电子与智能终端"
        mock_provider.get_industry_linkage.assert_called_with("消费电子", as_of="2026-08-20")

        # 4. _fetch_all 对未映射标的
        mock_provider.reset_mock()
        res_unmapped = _fetch_all("999999.SH", "2026-08-20", industry_provider=mock_provider)
        assert "industry_linkage" in res_unmapped
        assert res_unmapped["industry_linkage"] is None
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
        assert coll_res1["industry_linkage"]["industry_name"] == "消费电子与智能终端"

        # 修改返回值验证深拷贝防护
        coll_res1["industry_linkage"]["upstream_cost"][0]["current_value"] = 88888.88

        coll_res2 = collector_instance.collect("000725.SZ", "2026-08-20")
        assert mock_fetch.call_count == 1  # 命中内存缓存
        assert coll_res2["industry_linkage"]["upstream_cost"][0]["current_value"] == 9123.5


# ---------------------------------------------------------------------------
# 补充测试类 (DAV-274: 27 行业全覆盖与验收用例)
# ---------------------------------------------------------------------------


class TestIndustryLinkageSuite:
    """产业链数据层完整性与 27 行业覆盖验收测试类。"""

    def test_linkage_map_and_helper_functions(self):
        """测试 INDUSTRY_LINKAGE_MAP 覆盖全部 27 个行业且支持历史 5 个行业快捷解析。"""
        supported = list_supported_industries()
        kb_names = get_all_industry_names()

        assert len(supported) == 27
        assert len(kb_names) == 27
        assert set(supported) == set(kb_names)

        # 历史 5 行业别名查询验证
        for legacy_key, expected_target in [
            ("消费电子", "消费电子与智能终端"),
            ("新能源车", "新能源汽车与智能汽车"),
            ("半导体", "半导体与集成电路"),
            ("石油化工", "石油石化与基础化工"),
            ("金融地产", "商业银行与信贷"),
        ]:
            cfg = get_industry_linkage_config(legacy_key)
            assert cfg is not None, f"历史行业 {legacy_key} 必须能解析"
            assert cfg.industry_name == expected_target

        # 验证字典索引能力
        assert INDUSTRY_LINKAGE_MAP["消费电子"].industry_name == "消费电子与智能终端"
        assert INDUSTRY_LINKAGE_MAP["新能源车"].industry_name == "新能源汽车与智能汽车"
        assert INDUSTRY_LINKAGE_MAP["半导体"].industry_name == "半导体与集成电路"
        assert INDUSTRY_LINKAGE_MAP["石油化工"].industry_name == "石油石化与基础化工"
        assert INDUSTRY_LINKAGE_MAP["金融地产"].industry_name == "商业银行与信贷"

        unknown_config = get_industry_linkage_config("未知赛道XYZ")
        assert unknown_config is None

    def test_commercial_bank_linkage_shibor_and_lpr_config(self):
        """测试商业银行与信贷行业图谱配置中 Shibor 与 LPR 为 Tushare A类真值源。"""
        bank_cfg = get_industry_linkage_config("商业银行与信贷")
        assert bank_cfg is not None

        # 1. 验证 Shibor 3M 配置
        shibor_ind = [u for u in bank_cfg.upstream_cost if "Shibor" in u.name][0]
        assert shibor_ind.source == "tushare"
        assert shibor_ind.status == "active"
        assert shibor_ind.unit == "%"
        assert shibor_ind.metadata.get("api_name") == "shibor"
        assert shibor_ind.metadata.get("value_field") == "3m"
        assert shibor_ind.metadata.get("is_price") is False

        # 2. 验证 LPR 1Y 配置
        lpr_ind = [d for d in bank_cfg.downstream_demand if "LPR" in d.name][0]
        assert lpr_ind.source == "tushare"
        assert lpr_ind.status == "active"
        assert lpr_ind.unit == "%"
        assert lpr_ind.metadata.get("api_name") == "shibor_lpr"
        assert lpr_ind.metadata.get("value_field") == "1y"
        assert lpr_ind.metadata.get("is_price") is False

    def test_all_twenty_seven_industries_return_non_empty_linkage(self, monkeypatch):
        """验收标准 1 & 2：27 个行业都能 get_industry_linkage 返回非空结构；缺指标显式「数据缺失」，禁止臆造数值。"""
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
        provider = IndustryLinkageProvider()
        all_27_names = get_all_industry_names()
        assert len(all_27_names) == 27

        with patch("akshare.futures_foreign_hist", return_value=pd.DataFrame()), \
             patch("yfinance.Ticker", side_effect=Exception("Offline test")), \
             patch("requests.post", side_effect=Exception("Offline test")):
            for ind_name in all_27_names:
                data = provider.get_industry_linkage(ind_name, as_of="2026-08-20", use_cache=False)
                assert data is not None, f"行业 {ind_name} 应返回非空产业链结构"
                assert data["industry_name"] == ind_name
                assert "upstream_cost" in data and isinstance(data["upstream_cost"], list)
                assert len(data["upstream_cost"]) >= 1, f"行业 {ind_name} 必须有上游成本指标"
                assert "downstream_demand" in data and isinstance(data["downstream_demand"], list)
                assert len(data["downstream_demand"]) >= 1, f"行业 {ind_name} 必须有下游需求指标"
                assert "international_benchmark" in data and isinstance(data["international_benchmark"], list)
                assert len(data["international_benchmark"]) >= 1, f"行业 {ind_name} 必须有国际对标指标"
                assert "policy_catalysts" in data and len(data["policy_catalysts"]) >= 1, f"行业 {ind_name} 必须有政策催化"
                assert "description" in data

                # 验证离线/异常时所有指标均为「数据缺失」，零臆造数值
                all_indicators = (
                    data["upstream_cost"] + data["downstream_demand"] + data["international_benchmark"]
                )
                assert len(all_indicators) > 0
                for ind in all_indicators:
                    assert ind["current_value"] is None, f"{ind['name']} 异常/离线时 current_value 必须为 None"
                    assert ind["trend"] == "数据缺失", f"{ind['name']} trend 必须为「数据缺失」"

                prompt = format_industry_linkage_for_prompt(data)
                assert f"【产业链联想数据】：{ind_name}" in prompt

    def test_data_collector_maps_required_six_stocks(self):
        """验收标准 3：DataCollector 能把原有 6 只股票严格映射到对应行业，绝不回归。"""
        assert _map_stock_to_industry("688981.SH") == "半导体"  # 中芯国际
        assert _map_stock_to_industry("603501.SH") == "半导体"  # 韦尔股份
        assert _map_stock_to_industry("601857.SH") == "石油化工"  # 中国石油
        assert _map_stock_to_industry("600309.SH") == "石油化工"  # 万华化学
        assert _map_stock_to_industry("600036.SH") == "金融地产"  # 招商银行
        assert _map_stock_to_industry("000002.SZ") == "金融地产"  # 万科A

    def test_data_collector_maps_all_twenty_seven_industries(self):
        """验收标准：27 个行业均有至少 1 只 A 股代表股票映射，且全部可解析。"""
        representative_stocks = {
            "半导体与集成电路": "688981.SH",  # 中芯国际
            "人工智能与算力服务": "300308.SZ",  # 中际旭创
            "新能源汽车与智能汽车": "300750.SZ",  # 宁德时代
            "光伏与储能系统": "601012.SH",  # 隆基绿能
            "动力电池与储能电池材料": "002709.SZ",  # 天赐材料
            "医药生物与创新药": "600276.SH",  # 恒瑞医药
            "医疗器械与医疗服务": "300760.SZ",  # 迈瑞医疗
            "消费电子与智能终端": "000725.SZ",  # 京东方A
            "白酒与精制茶酒": "600519.SH",  # 贵州茅台
            "大众食品与饮料": "603288.SH",  # 海天味业
            "家用电器与智能家居": "000333.SZ",  # 美的集团
            "商业银行与信贷": "601166.SH",  # 兴业银行
            "证券公司与资本市场": "600030.SH",  # 中信证券
            "保险与多元金融": "601318.SH",  # 中国平安
            "钢铁与黑色金属": "600019.SH",  # 宝钢股份
            "有色金属与工业金属": "601899.SH",  # 紫金矿业
            "贵金属与稀缺资源": "600547.SH",  # 山东黄金
            "石油石化与基础化工": "601857.SH",  # 中国石油
            "煤炭与传统化石能源": "601088.SH",  # 中国神华
            "电力与公用事业": "600900.SH",  # 长江电力
            "房地产开发与运营": "600383.SH",  # 金地集团
            "建筑装饰与基础设施工程": "601668.SH",  # 中国建筑
            "机械设备与工业母机": "600031.SH",  # 三一重工
            "国防军工与航天装备": "600893.SH",  # 航发动力
            "交通运输与航运港口": "601919.SH",  # 中远海控
            "通信网络与光通信": "600941.SH",  # 中国移动
            "农林牧渔与生猪养殖": "002714.SZ",  # 牧原股份
        }

        assert len(representative_stocks) == 27
        for expected_ind_name, stock in representative_stocks.items():
            mapped_ind = _map_stock_to_industry(stock)
            assert mapped_ind is not None, f"股票 {stock} ({expected_ind_name}) 未能映射"
            cfg = get_industry_linkage_config(mapped_ind)
            assert cfg is not None, f"股票 {stock} 映射行业 {mapped_ind} 无法解析到配置"
            assert cfg.industry_name == expected_ind_name, f"股票 {stock} 映射到 {cfg.industry_name}，预期 {expected_ind_name}"

    def test_free_sources_typed_gap_on_failure(self):
        """验收标准：免费源（Yahoo/LME 等）失败必须 typed gap，不得假成功。"""
        provider = IndustryLinkageProvider()

        # akshare 失败
        with patch("akshare.futures_foreign_hist", side_effect=Exception("LME connection failed")):
            ind_lme = IndustryLinkageIndicator(name="LME铜价", source="akshare", symbol="铜")
            res_lme = provider._fetch_indicator(ind_lme)
            assert res_lme["current_value"] is None
            assert res_lme["trend"] == "数据缺失"
            assert res_lme["confidence"] == "低（接口异常）"
            assert "LME connection failed" in res_lme["note"]

        # yfinance 失败
        with patch("yfinance.Ticker", side_effect=Exception("Yahoo 429 Too Many Requests")):
            ind_yf = IndustryLinkageIndicator(name="台积电股价", source="yfinance", symbol="TSM")
            res_yf = provider._fetch_indicator(ind_yf)
            assert res_yf["current_value"] is None
            assert res_yf["trend"] == "数据缺失"
            assert res_yf["confidence"] == "低（接口异常）"
            assert "Yahoo 429" in res_yf["note"]

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
        assert "【产业链联想数据】：消费电子与智能终端" in text
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
