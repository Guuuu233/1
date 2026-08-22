"""针对 tradingagents.dataflows.industry_linkage 数据结构定义的确定性单元测试 (DAV-201 / DAV-274)。"""

import pytest
from pydantic import ValidationError

from tradingagents.dataflows.industry_linkage import (
    INDUSTRY_LINKAGE_MAP,
    IndustryLinkage,
    IndustryLinkageIndicator,
    get_industry_linkage_config,
    list_supported_industries,
)
from tradingagents.knowledge.industry_linkage import (
    get_all_industry_names,
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
    """测试 INDUSTRY_LINKAGE_MAP 预设映射的完整性与准确性 (DAV-274 覆盖 27 行业)。"""

    def test_map_contains_twenty_seven_industries(self):
        """验证准确包含知识库 27 个权威行业标准全称。"""
        keys = list(INDUSTRY_LINKAGE_MAP.keys())
        kb_names = get_all_industry_names()
        assert len(keys) == 27
        assert len(kb_names) == 27
        assert set(keys) == set(kb_names)

    def test_consumer_electronics_configuration(self):
        """验证消费电子行业配置的指标定义、上下游传导与对标。"""
        config = INDUSTRY_LINKAGE_MAP["消费电子"]
        assert isinstance(config, IndustryLinkage)
        assert config.industry_name == "消费电子与智能终端"
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
        samsung = [b for b in config.international_benchmark if "三星电子" in b.name][0]
        assert samsung.source == "yfinance"
        assert samsung.symbol == "005930.KS"
        assert samsung.unit == "韩元"
        assert samsung.role == "benchmark"
        assert samsung.status == "active"

        # 政策催化
        assert any("以旧换新" in cat for cat in config.policy_catalysts)

    def test_new_energy_vehicle_configuration(self):
        """验证新能源车行业配置的指标定义、上下游传导与对标。"""
        config = INDUSTRY_LINKAGE_MAP["新能源车"]
        assert isinstance(config, IndustryLinkage)
        assert config.industry_name == "新能源汽车与智能汽车"
        assert len(config.upstream_cost) >= 1
        assert len(config.downstream_demand) >= 1
        assert len(config.international_benchmark) >= 1
        assert len(config.policy_catalysts) >= 1

        # 上游成本端：碳酸锂价格（已付费 Tushare fut_daily LC.GFE）
        upstream = config.upstream_cost[0]
        assert "碳酸锂" in upstream.name
        assert upstream.source == "tushare"
        assert upstream.symbol == "LC.GFE"
        assert upstream.status == "active"
        assert upstream.role == "upstream"

        # 下游需求端：新能源车渗透率（标注手动）
        downstream = config.downstream_demand[0]
        assert downstream.name == "新能源车渗透率"
        assert downstream.source == "manual"
        assert downstream.note == "手动"
        assert downstream.status == "manual"
        assert downstream.role == "downstream"

        # 国际对标：特斯拉交付量（标注手动）
        tesla_del = [b for b in config.international_benchmark if b.name == "特斯拉交付量"][0]
        assert tesla_del.source == "manual"
        assert tesla_del.note == "手动"
        assert tesla_del.status == "manual"
        assert tesla_del.role == "benchmark"

        # 政策催化
        assert any("购置税减免" in cat for cat in config.policy_catalysts)

    def test_semiconductor_configuration(self):
        """验证半导体行业配置的指标定义、上下游传导与对标。"""
        config = INDUSTRY_LINKAGE_MAP["半导体"]
        assert isinstance(config, IndustryLinkage)
        assert config.industry_name == "半导体与集成电路"
        assert len(config.upstream_cost) >= 1
        assert len(config.downstream_demand) >= 1
        assert len(config.international_benchmark) >= 2
        assert len(config.policy_catalysts) >= 1

        # 上游：半导体硅片价格（pending_api）
        upstream = config.upstream_cost[0]
        assert upstream.name == "半导体硅片价格"
        assert upstream.source == "pending_api"
        assert upstream.current_value is None

        # 国际对标：费城半导体指数SOX与台积电TSM
        symbols = [b.symbol for b in config.international_benchmark]
        assert "^SOX" in symbols
        assert "TSM" in symbols

        # 政策催化
        assert any("大基金" in cat for cat in config.policy_catalysts)

    def test_petrochemical_configuration(self):
        """验证石油化工行业配置的指标定义、上下游传导与对标。"""
        config = INDUSTRY_LINKAGE_MAP["石油化工"]
        assert isinstance(config, IndustryLinkage)
        assert config.industry_name == "石油石化与基础化工"
        assert len(config.upstream_cost) >= 1
        assert len(config.downstream_demand) >= 1
        assert len(config.international_benchmark) >= 1
        assert len(config.policy_catalysts) >= 1

        # 上游：布伦特原油价格
        upstream = config.upstream_cost[0]
        assert upstream.name == "布伦特原油价格"
        assert upstream.source == "yfinance"
        assert upstream.symbol == "BZ=F"
        assert upstream.current_value is None

        # 国际对标：埃克森美孚XOM
        benchmark = config.international_benchmark[0]
        assert benchmark.name == "埃克森美孚股价"
        assert benchmark.source == "yfinance"
        assert benchmark.symbol == "XOM"

        # 政策催化
        assert any("碳排放" in cat or "能耗" in cat for cat in config.policy_catalysts)

    def test_finance_real_estate_configuration(self):
        """验证金融地产行业配置的指标定义、上下游传导与对标。"""
        config = INDUSTRY_LINKAGE_MAP["金融地产"]
        assert isinstance(config, IndustryLinkage)
        assert config.industry_name == "商业银行与信贷"
        assert len(config.upstream_cost) >= 1
        assert len(config.downstream_demand) >= 1
        assert len(config.international_benchmark) >= 1
        assert len(config.policy_catalysts) >= 1

        # 国际对标：摩根大通JPM
        symbols = [b.symbol for b in config.international_benchmark]
        assert "JPM" in symbols

        # 政策催化
        assert any("房贷利率" in cat or "债务" in cat for cat in config.policy_catalysts)

    def test_helper_get_industry_linkage_config(self):
        """验证配置查询辅助函数与模糊/别名匹配逻辑。"""
        ce = get_industry_linkage_config("消费电子")
        assert ce is not None
        assert ce.industry_name == "消费电子与智能终端"

        ce_full = get_industry_linkage_config("消费电子与智能终端")
        assert ce_full is not None
        assert ce_full.industry_name == "消费电子与智能终端"

        nev = get_industry_linkage_config("新能源车")
        assert nev is not None
        assert nev.industry_name == "新能源汽车与智能汽车"

        semi = get_industry_linkage_config("半导体")
        assert semi is not None
        assert semi.industry_name == "半导体与集成电路"

        petro = get_industry_linkage_config("石油化工")
        assert petro is not None
        assert petro.industry_name == "石油石化与基础化工"

        fin = get_industry_linkage_config("金融地产")
        assert fin is not None
        assert fin.industry_name == "商业银行与信贷"

        bank = get_industry_linkage_config("银行")
        assert bank is not None
        assert bank.industry_name == "商业银行与信贷"

        re_conf = get_industry_linkage_config("房地产")
        assert re_conf is not None
        assert re_conf.industry_name == "房地产开发与运营"

        unknown = get_industry_linkage_config("未知行业XYZ")
        assert unknown is None

    def test_helper_list_supported_industries(self):
        """验证支持行业列表辅助函数。"""
        industries = list_supported_industries()
        assert isinstance(industries, list)
        assert len(industries) == 27
        assert "半导体与集成电路" in industries
        assert "消费电子与智能终端" in industries
        assert "新能源汽车与智能汽车" in industries


class TestIndustryLinkageMarketDataContextFlows:
    """测试产业链数据进入 market_data_context、source_provenance 与 failure_ledger 的数据流 (DAV-311)。"""

    def test_default_market_data_context_contains_industry_linkage(self):
        from tradingagents.graph.data_collector import default_market_data_context

        ctx = default_market_data_context()
        assert "industry_linkage" in ctx
        assert ctx["industry_linkage"] is None

    def test_source_provenance_and_ledger_for_available_and_partial_linkage(self):
        from tradingagents.graph.data_collector import _build_source_provenance, _build_data_failure_ledger

        # 1. 结构化部分可用产业链数据
        partial_linkage = {
            "industry_name": "消费电子与智能终端",
            "upstream_cost": [{"name": "LME铜价", "current_value": 9123.5, "status": "active"}],
            "downstream_demand": [{"name": "全球智能手机出货量", "current_value": None, "status": "manual"}],
            "international_benchmark": [{"name": "三星电子股价", "current_value": 52000.0, "status": "active"}],
            "as_of": "2026-08-20",
        }
        results = {"industry_linkage": partial_linkage}
        provenance = _build_source_provenance(results, "2026-08-20", daily_as_of="2026-08-20")

        assert "industry_linkage" in provenance
        prov = provenance["industry_linkage"]
        assert prov["requested_as_of"] == "2026-08-20"
        assert prov["actual_as_of"] == "2026-08-20"
        assert prov["as_of"] == "2026-08-20"
        assert prov["status"] == "partial"
        assert "gap" not in prov

        ledger = _build_data_failure_ledger(results)
        assert not any(entry.get("source") == "industry_linkage" for entry in ledger)

    def test_source_provenance_and_ledger_for_unavailable_linkage(self):
        from tradingagents.graph.data_collector import _build_source_provenance, _build_data_failure_ledger

        # 完全失败时 (None 或 error string)
        results = {"industry_linkage": None}
        provenance = _build_source_provenance(results, "2026-08-20", daily_as_of="2026-08-20")

        assert "industry_linkage" in provenance
        prov = provenance["industry_linkage"]
        assert prov["requested_as_of"] == "2026-08-20"
        assert prov["actual_as_of"] is None
        assert prov["status"] == "unavailable"
        assert "gap" in prov

    def test_anti_lookahead_discipline_for_linkage_date(self):
        from tradingagents.graph.data_collector import _build_source_provenance

        # 当产业链数据日期超过 analysis_baseline_date 时，不得前视未来日期
        future_linkage = {
            "industry_name": "消费电子与智能终端",
            "upstream_cost": [{"name": "LME铜价", "current_value": 9123.5}],
            "as_of": "2026-08-25",  # 晚于 requested_as_of 2026-08-20
        }
        results = {"industry_linkage": future_linkage}
        provenance = _build_source_provenance(results, "2026-08-20", daily_as_of="2026-08-20")
        assert provenance["industry_linkage"]["actual_as_of"] is None
        assert provenance["industry_linkage"]["as_of"] is None

    def test_collector_fetch_all_populates_both_top_level_and_context(self):
        from unittest.mock import patch
        from tradingagents.graph import data_collector

        mock_linkage = {
            "industry_name": "消费电子与智能终端",
            "upstream_cost": [{"name": "LME铜价", "current_value": 9123.5, "status": "active"}],
            "downstream_demand": [{"name": "出货量", "current_value": None, "status": "manual"}],
            "international_benchmark": [{"name": "三星电子", "current_value": 50000.0, "status": "active"}],
            "as_of": "2026-08-20",
        }

        mock_provider = data_collector.IndustryLinkageProvider()
        with patch.object(data_collector, "_safe", return_value="dummy"), \
             patch.object(data_collector, "FETCH_ALL_TIMEOUT", 1), \
             patch.object(mock_provider, "get_industry_linkage", return_value=mock_linkage):
            res = data_collector._fetch_all("000725.SZ", "2026-08-20", industry_provider=mock_provider)

        # 1. 顶层保留 results["industry_linkage"]
        assert "industry_linkage" in res
        assert res["industry_linkage"] == mock_linkage

        # 2. context 写入 market_data_context.industry_linkage
        ctx = res["market_data_context"]
        assert "industry_linkage" in ctx
        assert ctx["industry_linkage"] == mock_linkage

        # 3. source_provenance 正确
        prov = ctx["source_provenance"]["industry_linkage"]
        assert prov["requested_as_of"] == "2026-08-20"
        assert prov["actual_as_of"] == "2026-08-20"
        assert prov["status"] == "partial"

        # 4. failure_ledger 中无 industry_linkage 错误（因为已成功获取部分有效数据）
        assert not any(e.get("source") == "industry_linkage" for e in ctx["data_failure_ledger"])

