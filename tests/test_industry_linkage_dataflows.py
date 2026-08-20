"""针对 tradingagents.dataflows.industry_linkage 数据结构定义的确定性单元测试 (DAV-201 M1)。"""

import pytest
from pydantic import ValidationError

from tradingagents.dataflows.industry_linkage import (
    INDUSTRY_LINKAGE_MAP,
    IndustryLinkage,
    IndustryLinkageIndicator,
    get_industry_linkage_config,
    list_supported_industries,
)


class TestIndustryLinkageDataStructures:
    """测试产业链数据层数据模型与验证逻辑。"""

    def test_indicator_model_defaults_and_validation(self):
        """验证 IndustryLinkageIndicator 基础字段与默认值。"""
        ind = IndustryLinkageIndicator(
            name="LME铜价",
            source="akshare",
            symbol="铜",
            unit="美元/吨",
            role="upstream",
            status="active",
            transmission_logic="导电与连接件成本",
        )
        assert ind.name == "LME铜价"
        assert ind.source == "akshare"
        assert ind.symbol == "铜"
        assert ind.unit == "美元/吨"
        assert ind.role == "upstream"
        assert ind.status == "active"
        assert ind.frequency == "daily"
        assert ind.transmission_logic == "导电与连接件成本"
        # 静态配置阶段不得包含伪造的实时数值
        assert ind.current_value is None
        assert ind.mom_change is None
        assert ind.qoq_change is None
        assert ind.yoy_change is None
        assert ind.trend is None
        assert ind.confidence is None
        assert ind.note is None
        assert isinstance(ind.metadata, dict)

    def test_indicator_runtime_assignment_and_dump(self):
        """验证运行时采集器注入数据后的序列化与反序列化。"""
        ind = IndustryLinkageIndicator(
            name="LME铜价",
            source="akshare",
            symbol="铜",
            current_value=9123.5,
            mom_change=2.3,
            qoq_change=5.1,
            yoy_change=-1.2,
            trend="上升",
            confidence="高",
        )
        assert ind.current_value == 9123.5
        assert ind.mom_change == 2.3
        assert ind.trend == "上升"

        dumped = ind.model_dump()
        assert dumped["name"] == "LME铜价"
        assert dumped["current_value"] == 9123.5
        assert dumped["mom_change"] == 2.3

    def test_indicator_missing_required_fields_raises(self):
        """缺少必填字段时必须抛出 ValidationError。"""
        with pytest.raises(ValidationError):
            IndustryLinkageIndicator()  # type: ignore

        with pytest.raises(ValidationError):
            IndustryLinkageIndicator(name="LME铜价")  # type: ignore

    def test_industry_linkage_model_defaults(self):
        """验证 IndustryLinkage 整体容器默认行为与结构。"""
        linkage = IndustryLinkage(industry_name="测试行业")
        assert linkage.industry_name == "测试行业"
        assert linkage.upstream_cost == []
        assert linkage.downstream_demand == []
        assert linkage.international_benchmark == []
        assert linkage.policy_catalysts == []
        assert linkage.description is None
        assert isinstance(linkage.metadata, dict)


class TestIndustryLinkageMap:
    """测试 INDUSTRY_LINKAGE_MAP 预设映射的完整性与准确性。"""

    def test_map_contains_two_mvp_industries(self):
        """验证 MVP 阶段准确包含 '消费电子' 和 '新能源车' 两个核心行业。"""
        keys = list(INDUSTRY_LINKAGE_MAP.keys())
        assert set(keys) == {"消费电子", "新能源车"}
        assert len(keys) == 2

    def test_consumer_electronics_configuration(self):
        """验证消费电子行业配置的指标定义、上下游传导与对标。"""
        config = INDUSTRY_LINKAGE_MAP["消费电子"]
        assert isinstance(config, IndustryLinkage)
        assert config.industry_name == "消费电子/半导体显示"
        assert len(config.upstream_cost) >= 1
        assert len(config.downstream_demand) >= 1
        assert len(config.international_benchmark) >= 1
        assert len(config.policy_catalysts) >= 1

        # 上游成本端：LME铜价
        upstream = config.upstream_cost[0]
        assert upstream.name == "LME铜价"
        assert upstream.source == "akshare"
        assert upstream.symbol == "铜"
        assert upstream.unit == "美元/吨"
        assert upstream.role == "upstream"
        assert upstream.status == "active"
        assert upstream.current_value is None  # 禁止伪造静态数值

        # 下游需求端：全球智能手机出货量（标注手动）
        downstream = config.downstream_demand[0]
        assert downstream.name == "全球智能手机出货量"
        assert downstream.source == "manual"
        assert downstream.note == "手动"
        assert downstream.status == "manual"
        assert downstream.role == "downstream"

        # 国际对标：三星电子股价
        benchmark = config.international_benchmark[0]
        assert benchmark.name == "三星电子股价"
        assert benchmark.source == "yfinance"
        assert benchmark.symbol == "005930.KS"
        assert benchmark.unit == "韩元"
        assert benchmark.role == "benchmark"
        assert benchmark.status == "active"

        # 政策催化
        assert "消费品以旧换新" in config.policy_catalysts

    def test_new_energy_vehicle_configuration(self):
        """验证新能源车行业配置的指标定义、上下游传导与对标。"""
        config = INDUSTRY_LINKAGE_MAP["新能源车"]
        assert isinstance(config, IndustryLinkage)
        assert config.industry_name == "新能源车/动力电池"
        assert len(config.upstream_cost) >= 1
        assert len(config.downstream_demand) >= 1
        assert len(config.international_benchmark) >= 1
        assert len(config.policy_catalysts) >= 1

        # 上游成本端：碳酸锂价格（标注待接入API）
        upstream = config.upstream_cost[0]
        assert upstream.name == "碳酸锂价格"
        assert upstream.source == "pending_api"
        assert upstream.note == "待接入API"
        assert upstream.status == "pending_api"
        assert upstream.role == "upstream"
        assert upstream.current_value is None

        # 下游需求端：新能源车渗透率（标注手动）
        downstream = config.downstream_demand[0]
        assert downstream.name == "新能源车渗透率"
        assert downstream.source == "manual"
        assert downstream.note == "手动"
        assert downstream.status == "manual"
        assert downstream.role == "downstream"

        # 国际对标：特斯拉交付量（标注手动）
        benchmark = config.international_benchmark[0]
        assert benchmark.name == "特斯拉交付量"
        assert benchmark.source == "manual"
        assert benchmark.note == "手动"
        assert benchmark.status == "manual"
        assert benchmark.role == "benchmark"

        # 政策催化
        assert "新能源汽车购置税减免" in config.policy_catalysts

    def test_helper_get_industry_linkage_config(self):
        """验证配置查询辅助函数与模糊匹配逻辑。"""
        ce = get_industry_linkage_config("消费电子")
        assert ce is not None
        assert ce.industry_name == "消费电子/半导体显示"

        ce_full = get_industry_linkage_config("消费电子/半导体显示")
        assert ce_full is not None
        assert ce_full.industry_name == "消费电子/半导体显示"

        nev = get_industry_linkage_config("新能源车")
        assert nev is not None
        assert nev.industry_name == "新能源车/动力电池"

        nev_full = get_industry_linkage_config("新能源车/动力电池")
        assert nev_full is not None
        assert nev_full.industry_name == "新能源车/动力电池"

        unknown = get_industry_linkage_config("未知行业XYZ")
        assert unknown is None

    def test_helper_list_supported_industries(self):
        """验证支持行业列表辅助函数。"""
        industries = list_supported_industries()
        assert isinstance(industries, list)
        assert set(industries) == {"消费电子", "新能源车"}
