"""针对 DataCollector 产业链数据层集成 (DAV-201 M3) 的单元测试。

测试覆盖：
1. 股票代码到核心行业硬编码映射 (_map_stock_to_industry)；
2. DataCollector 实例初始化与 IndustryLinkageProvider 依赖注入；
3. _fetch_all 中产业链数据的正确拉取、字段写入与未映射股票安全返回 None；
4. 日期格式兼容性 (YYYY-MM-DD 与 YYYYMMDD) 与异常优雅降级；
5. DataCollector.collect 缓存与防御性拷贝。
"""

from unittest.mock import MagicMock, patch
import pytest

from tradingagents.dataflows.providers.industry_linkage_provider import (
    IndustryLinkageProvider,
)
from tradingagents.graph.data_collector import (
    DataCollector,
    _fetch_all,
    _map_stock_to_industry,
)


class TestMapStockToIndustry:
    """测试股票代码到产业链核心行业的映射逻辑。"""

    def test_consumer_electronics_stocks(self):
        """测试消费电子赛道核心标的映射。"""
        # 京东方A
        assert _map_stock_to_industry("000725.SZ") == "消费电子"
        assert _map_stock_to_industry("000725") == "消费电子"
        assert _map_stock_to_industry("000725.sz") == "消费电子"
        # TCL科技
        assert _map_stock_to_industry("000100.SZ") == "消费电子"
        assert _map_stock_to_industry("000100") == "消费电子"
        # 立讯精密、歌尔股份、蓝思科技
        assert _map_stock_to_industry("002475.SZ") == "消费电子"
        assert _map_stock_to_industry("002241.SZ") == "消费电子"
        assert _map_stock_to_industry("300433.SZ") == "消费电子"

    def test_new_energy_vehicle_stocks(self):
        """测试新能源车赛道核心标的映射。"""
        # 宁德时代
        assert _map_stock_to_industry("300750.SZ") == "新能源车"
        assert _map_stock_to_industry("300750") == "新能源车"
        assert _map_stock_to_industry("300750.sz") == "新能源车"
        # 比亚迪
        assert _map_stock_to_industry("002594.SZ") == "新能源车"
        assert _map_stock_to_industry("002594") == "新能源车"
        # 长城汽车、赣锋锂业、天齐锂业
        assert _map_stock_to_industry("601633.SH") == "新能源车"
        assert _map_stock_to_industry("002460.SZ") == "新能源车"
        assert _map_stock_to_industry("002466.SZ") == "新能源车"

    def test_semiconductor_stocks(self):
        """测试半导体赛道核心标的映射 (DAV-256)."""
        # 中芯国际
        assert _map_stock_to_industry("688981.SH") == "半导体"
        assert _map_stock_to_industry("688981") == "半导体"
        assert _map_stock_to_industry("688981.sh") == "半导体"
        # 韦尔股份
        assert _map_stock_to_industry("603501.SH") == "半导体"
        assert _map_stock_to_industry("603501") == "半导体"
        # 其他半导体标的
        assert _map_stock_to_industry("002049.SZ") == "半导体"
        assert _map_stock_to_industry("600584.SH") == "半导体"
        assert _map_stock_to_industry("688012.SH") == "半导体"
        assert _map_stock_to_industry("002371.SZ") == "半导体"

    def test_petrochemical_stocks(self):
        """测试石油化工赛道核心标的映射 (DAV-256)."""
        # 中国石油
        assert _map_stock_to_industry("601857.SH") == "石油化工"
        assert _map_stock_to_industry("601857") == "石油化工"
        assert _map_stock_to_industry("601857.sh") == "石油化工"
        # 万华化学
        assert _map_stock_to_industry("600309.SH") == "石油化工"
        assert _map_stock_to_industry("600309") == "石油化工"
        # 其他石化标的
        assert _map_stock_to_industry("600028.SH") == "石油化工"
        assert _map_stock_to_industry("600938.SH") == "石油化工"
        assert _map_stock_to_industry("000301.SZ") == "石油化工"

    def test_finance_real_estate_stocks(self):
        """测试金融地产/银行赛道核心标的映射 (DAV-256)."""
        # 招商银行
        assert _map_stock_to_industry("600036.SH") == "金融地产"
        assert _map_stock_to_industry("600036") == "金融地产"
        assert _map_stock_to_industry("600036.sh") == "金融地产"
        # 万科A
        assert _map_stock_to_industry("000002.SZ") == "金融地产"
        assert _map_stock_to_industry("000002") == "金融地产"
        assert _map_stock_to_industry("000002.sz") == "金融地产"
        # 其他银行地产标的
        assert _map_stock_to_industry("601398.SH") == "金融地产"
        assert _map_stock_to_industry("600048.SH") == "金融地产"

    def test_unmapped_and_invalid_inputs(self):
        """测试未映射标的及非法输入返回 None。"""
        # 未映射股票
        assert _map_stock_to_industry("600519.SH") is None  # 贵州茅台
        assert _map_stock_to_industry("601318.SH") is None  # 中国平安
        assert _map_stock_to_industry("000858.SZ") is None  # 五粮液
        assert _map_stock_to_industry("UNKNOWN") is None

        # 空值与非法类型
        assert _map_stock_to_industry(None) is None
        assert _map_stock_to_industry("") is None
        assert _map_stock_to_industry("   ") is None
        assert _map_stock_to_industry(12345) is None  # type: ignore

    def test_class_level_and_instance_binding(self):
        """测试 DataCollector 类及实例能直接访问 _map_stock_to_industry。"""
        assert DataCollector._map_stock_to_industry("000725.SZ") == "消费电子"
        collector = DataCollector()
        assert collector._map_stock_to_industry("300750.SZ") == "新能源车"
        assert collector._map_stock_to_industry("688981.SH") == "半导体"
        assert collector._map_stock_to_industry("601857.SH") == "石油化工"
        assert collector._map_stock_to_industry("600036.SH") == "金融地产"
        assert collector._map_stock_to_industry("600519.SH") is None


class TestDataCollectorIndustryLinkageIntegration:
    """测试 DataCollector 与 IndustryLinkageProvider 的集成逻辑。"""

    def test_data_collector_initialization_and_provider_injection(self):
        """测试 DataCollector 初始化具有 industry_linkage_provider 属性且支持注入。"""
        collector = DataCollector()
        assert hasattr(collector, "industry_linkage_provider")
        assert isinstance(collector.industry_linkage_provider, IndustryLinkageProvider)

        custom_provider = IndustryLinkageProvider(cache_ttl=1800)
        custom_collector = DataCollector(industry_linkage_provider=custom_provider)
        assert custom_collector.industry_linkage_provider is custom_provider

    @patch("tradingagents.graph.data_collector._safe", return_value="dummy_data")
    @patch("tradingagents.graph.data_collector.IndustryLinkageProvider.get_industry_linkage")
    def test_fetch_all_mapped_stock_consumer_electronics(self, mock_get_linkage, mock_safe):
        """测试 _fetch_all 对消费电子标的（如京东方A 000725.SZ）调用 Provider 并填充 industry_linkage。"""
        mock_get_linkage.return_value = {
            "industry_name": "消费电子/半导体显示",
            "upstream_cost": [{"name": "LME铜价", "current_value": 9123.5}],
            "downstream_demand": [{"name": "全球智能手机出货量", "trend": "数据缺失"}],
            "international_benchmark": [{"name": "三星电子股价", "current_value": 52000.0}],
            "policy_catalysts": ["消费品以旧换新"],
            "description": "消费电子产业链",
            "as_of": "2026-08-20",
        }

        result = _fetch_all("000725.SZ", "2026-08-20")

        assert "industry_linkage" in result
        linkage = result["industry_linkage"]
        assert linkage is not None
        assert linkage["industry_name"] == "消费电子/半导体显示"
        assert linkage["upstream_cost"][0]["name"] == "LME铜价"
        mock_get_linkage.assert_called_once_with("消费电子", as_of="2026-08-20")

    @patch("tradingagents.graph.data_collector._safe", return_value="dummy_data")
    @patch("tradingagents.graph.data_collector.IndustryLinkageProvider.get_industry_linkage")
    def test_fetch_all_mapped_stock_new_energy(self, mock_get_linkage, mock_safe):
        """测试 _fetch_all 对新能源车标的（如宁德时代 300750.SZ）调用 Provider 并填充 industry_linkage。"""
        mock_get_linkage.return_value = {
            "industry_name": "新能源车/动力电池",
            "upstream_cost": [{"name": "碳酸锂价格", "trend": "数据缺失"}],
            "downstream_demand": [{"name": "新能源车渗透率", "trend": "数据缺失"}],
            "international_benchmark": [{"name": "特斯拉交付量", "trend": "数据缺失"}],
            "policy_catalysts": ["新能源汽车购置税减免"],
            "description": "新能源车产业链",
            "as_of": "2026-08-20",
        }

        result = _fetch_all("300750.SZ", "2026-08-20")

        assert "industry_linkage" in result
        linkage = result["industry_linkage"]
        assert linkage is not None
        assert linkage["industry_name"] == "新能源车/动力电池"
        mock_get_linkage.assert_called_once_with("新能源车", as_of="2026-08-20")

    @patch("tradingagents.graph.data_collector._safe", return_value="dummy_data")
    @patch("tradingagents.graph.data_collector.IndustryLinkageProvider.get_industry_linkage")
    def test_fetch_all_unmapped_stock_returns_none(self, mock_get_linkage, mock_safe):
        """测试 _fetch_all 对未映射标的（如贵州茅台 600519.SH）返回 None 且不调用 Provider。"""
        result = _fetch_all("600519.SH", "2026-08-20")

        assert "industry_linkage" in result
        assert result["industry_linkage"] is None
        mock_get_linkage.assert_not_called()

    @patch("tradingagents.graph.data_collector._safe", return_value="dummy_data")
    def test_fetch_all_date_format_normalization(self, mock_safe):
        """测试 _fetch_all 能兼容 YYYYMMDD 与 YYYY-MM-DD 两种日期输入格式。"""
        mock_provider = MagicMock(spec=IndustryLinkageProvider)
        mock_provider.get_industry_linkage.return_value = {
            "industry_name": "消费电子/半导体显示",
            "as_of": "2026-08-20",
        }

        # 格式 1: YYYY-MM-DD
        res1 = _fetch_all("000725.SZ", "2026-08-20", industry_provider=mock_provider)
        assert res1["industry_linkage"] is not None
        mock_provider.get_industry_linkage.assert_called_with("消费电子", as_of="2026-08-20")

        # 格式 2: YYYYMMDD
        res2 = _fetch_all("000725.SZ", "20260820", industry_provider=mock_provider)
        assert res2["industry_linkage"] is not None
        mock_provider.get_industry_linkage.assert_called_with("消费电子", as_of="2026-08-20")

    @patch("tradingagents.graph.data_collector._safe", return_value="dummy_data")
    def test_fetch_all_provider_exception_graceful_degradation(self, mock_safe):
        """测试当 Provider 发生异常时，_fetch_all 优雅捕获并将 industry_linkage 置为 None，不中断整体采集。"""
        faulty_provider = MagicMock(spec=IndustryLinkageProvider)
        faulty_provider.get_industry_linkage.side_effect = RuntimeError("网络不可达")

        result = _fetch_all("000725.SZ", "2026-08-20", industry_provider=faulty_provider)

        assert "industry_linkage" in result
        assert result["industry_linkage"] is None

    def test_data_collector_collect_caches_and_defensive_copies_industry_linkage(self):
        """测试 DataCollector.collect 返回深拷贝且支持缓存。"""
        collector = DataCollector()
        fake_linkage = {
            "industry_name": "消费电子/半导体显示",
            "upstream_cost": [{"name": "LME铜价", "current_value": 9123.5}],
        }
        stub_pool = {
            "stock_data": "dummy",
            "indicators": {},
            "industry_linkage": fake_linkage,
        }

        with patch("tradingagents.graph.data_collector._fetch_all", return_value=stub_pool) as mock_fetch:
            res1 = collector.collect("000725.SZ", "2026-08-20")
            assert res1["industry_linkage"]["industry_name"] == "消费电子/半导体显示"

            # 验证深拷贝：修改返回值不污染缓存
            res1["industry_linkage"]["upstream_cost"][0]["current_value"] = 99999.0

            res2 = collector.collect("000725.SZ", "2026-08-20")
            assert mock_fetch.call_count == 1  # 命中缓存
            assert res2["industry_linkage"]["upstream_cost"][0]["current_value"] == 9123.5
