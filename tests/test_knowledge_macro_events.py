"""针对 tradingagents.knowledge.macro_events 的完备单元测试。"""

import pytest
from tradingagents.knowledge.macro_events import (
    MACRO_EVENT_SCENARIOS,
    MacroEventScenario,
    SectorImpact,
    get_all_macro_event_ids,
    get_all_macro_event_names,
    get_macro_event_scenario,
    search_macro_events,
    match_events_from_text,
    get_transmission_path,
    get_sector_macro_exposure,
    format_macro_event_context,
)


def test_macro_scenarios_coverage():
    """验证宏观情景库覆盖主流宏观场景（货币、财政、外汇、大宗商品、地缘、通胀周期等）。"""
    ids = get_all_macro_event_ids()
    assert len(ids) >= 12, f"宏观事件情景数量不足，当前为 {len(ids)}"
    assert len(MACRO_EVENT_SCENARIOS) >= 12


def test_every_macro_scenario_completeness():
    """验证每个宏观情景结构完整、三级传导链路健全且具备明确的受益与受损行业。"""
    for event_id, sc in MACRO_EVENT_SCENARIOS.items():
        assert sc.event_id == event_id
        assert isinstance(sc.event_name, str) and len(sc.event_name) > 0
        assert isinstance(sc.category, str) and len(sc.category) > 0
        assert isinstance(sc.aliases, list) and len(sc.aliases) > 0
        assert isinstance(sc.description, str) and len(sc.description) > 10

        # 三级传导机制
        assert isinstance(sc.transmission_mechanism, list) and len(sc.transmission_mechanism) == 3, f"{event_id} 传导链必须为 3 级"
        assert "一级" in sc.transmission_mechanism[0] or "Step 1" in sc.transmission_mechanism[0]
        assert "二级" in sc.transmission_mechanism[1] or "Step 2" in sc.transmission_mechanism[1]
        assert "三级" in sc.transmission_mechanism[2] or "Step 3" in sc.transmission_mechanism[2]

        # 直接冲击与跨市场外溢
        assert isinstance(sc.direct_impact, list) and len(sc.direct_impact) >= 2
        assert isinstance(sc.cross_market_spillovers, dict) and len(sc.cross_market_spillovers) >= 3

        # 受益与受损行业
        assert isinstance(sc.beneficiary_sectors, list) and len(sc.beneficiary_sectors) >= 1, f"{event_id} 缺少受益行业"
        for b in sc.beneficiary_sectors:
            assert isinstance(b, SectorImpact)
            assert len(b.sector) > 0
            assert len(b.transmission_logic) > 10
            assert b.impact_level in ("极高", "高", "中高", "中", "低")

        assert isinstance(sc.adversely_affected_sectors, list) and len(sc.adversely_affected_sectors) >= 1, f"{event_id} 缺少受损行业"
        for a in sc.adversely_affected_sectors:
            assert isinstance(a, SectorImpact)
            assert len(a.sector) > 0
            assert len(a.transmission_logic) > 10

        # 高频监测与历史复盘
        assert isinstance(sc.key_monitoring_indicators, list) and len(sc.key_monitoring_indicators) >= 2
        assert isinstance(sc.historical_reference_cases, list) and len(sc.historical_reference_cases) >= 1


def test_get_all_macro_event_names():
    """验证获取宏观事件名称列表。"""
    names = get_all_macro_event_names()
    assert len(names) == len(MACRO_EVENT_SCENARIOS)
    assert "央行降息降准与流动性宽松" in names
    assert "原油与能源价格暴涨" in names
    assert "国际地缘冲突与海峡航道受阻" in names
    assert "黄金与贵金属避险暴涨/实际利率下行" in names


@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("monetary_easing", "monetary_easing"),
        ("降息", "monetary_easing"),
        ("降准", "monetary_easing"),
        ("LPR下调", "monetary_easing"),
        ("monetary_tightening", "monetary_tightening"),
        ("加息", "monetary_tightening"),
        ("金融去杠杆", "monetary_tightening"),
        ("fiscal_expansion", "fiscal_expansion"),
        ("特别国债", "fiscal_expansion"),
        ("基建刺激", "fiscal_expansion"),
        ("rmb_depreciation", "rmb_depreciation"),
        ("人民币贬值", "rmb_depreciation"),
        ("汇率破7", "rmb_depreciation"),
        ("rmb_appreciation", "rmb_appreciation"),
        ("人民币升值", "rmb_appreciation"),
        ("oil_price_shock_up", "oil_price_shock_up"),
        ("油价暴涨", "oil_price_shock_up"),
        ("原油飙升", "oil_price_shock_up"),
        ("commodity_supercycle_metals", "commodity_supercycle_metals"),
        ("铜价暴涨", "commodity_supercycle_metals"),
        ("工业金属超级周期", "commodity_supercycle_metals"),
        ("gold_safe_haven_rally", "gold_safe_haven_rally"),
        ("金价暴涨", "gold_safe_haven_rally"),
        ("央行购金", "gold_safe_haven_rally"),
        ("geopolitical_conflict_escalation", "geopolitical_conflict_escalation"),
        ("地缘冲突", "geopolitical_conflict_escalation"),
        ("红海危机", "geopolitical_conflict_escalation"),
        ("export_tariffs_trade_friction", "export_tariffs_trade_friction"),
        ("加征关税", "export_tariffs_trade_friction"),
        ("贸易摩擦", "export_tariffs_trade_friction"),
        ("cpi_ppi_scissors_widening", "cpi_ppi_scissors_widening"),
        ("剪刀差", "cpi_ppi_scissors_widening"),
        ("CPI-PPI剪刀差", "cpi_ppi_scissors_widening"),
        ("deflation_demand_contraction", "deflation_demand_contraction"),
        ("通缩", "deflation_demand_contraction"),
        ("资产负债表衰退", "deflation_demand_contraction"),
        ("us_fed_rate_cut_cycle", "us_fed_rate_cut_cycle"),
        ("美联储降息", "us_fed_rate_cut_cycle"),
        ("real_estate_policy_easing", "real_estate_policy_easing"),
        ("地产松绑", "real_estate_policy_easing"),
        ("取消限购", "real_estate_policy_easing"),
        ("ai_tech_revolution_breakthrough", "ai_tech_revolution_breakthrough"),
        ("大模型升级", "ai_tech_revolution_breakthrough"),
        ("算力爆发", "ai_tech_revolution_breakthrough"),
        ("capital_market_institutional_reform", "capital_market_institutional_reform"),
        ("新国九条", "capital_market_institutional_reform"),
        ("分红新规", "capital_market_institutional_reform"),
        ("extreme_weather_power_curtailment", "extreme_weather_power_curtailment"),
        ("高温限电", "extreme_weather_power_curtailment"),
        ("限电", "extreme_weather_power_curtailment"),
    ],
)
def test_get_macro_event_scenario_matching(query, expected_id):
    """验证宏观事件ID、标准名及别名的精确与模糊匹配正确性。"""
    sc = get_macro_event_scenario(query)
    assert sc is not None, f"未能匹配到宏观事件: {query}"
    assert sc.event_id == expected_id


def test_get_macro_event_scenario_invalid():
    """验证无效宏观事件查询安全返回 None。"""
    assert get_macro_event_scenario("") is None
    assert get_macro_event_scenario("   ") is None
    assert get_macro_event_scenario(None) is None  # type: ignore
    assert get_macro_event_scenario(999) is None  # type: ignore
    assert get_macro_event_scenario("不存在的火星陨石撞击事件abc") is None


def test_search_macro_events():
    """验证宏观事件综合搜索。"""
    # 搜索 "红海" 应当命中地缘冲突
    results = search_macro_events("红海")
    ids = [r.event_id for r in results]
    assert "geopolitical_conflict_escalation" in ids

    # 搜索 "LPR" 应当命中降息宽松
    results_lpr = search_macro_events("LPR")
    ids_lpr = [r.event_id for r in results_lpr]
    assert "monetary_easing" in ids_lpr

    # 搜索 "分红" 应当命中资本市场改革
    results_div = search_macro_events("分红")
    ids_div = [r.event_id for r in results_div]
    assert "capital_market_institutional_reform" in ids_div

    # 异常输入
    assert search_macro_events("") == []
    assert search_macro_events(None) == []  # type: ignore


def test_match_events_from_text():
    """验证从长文本或新闻摘要中自动识别宏观事件。"""
    news_sample = "央行今日宣布降准0.5个百分点，同时由于中东红海危机导致油价暴涨，国际大宗商品大幅波动。"
    matched = match_events_from_text(news_sample)
    matched_ids = [m.event_id for m in matched]
    assert "monetary_easing" in matched_ids
    assert "geopolitical_conflict_escalation" in matched_ids
    assert "oil_price_shock_up" in matched_ids

    # 空文本测试
    assert match_events_from_text("") == []
    assert match_events_from_text(None) == []  # type: ignore


def test_get_transmission_path():
    """验证提取指定事件的三级传导链路。"""
    path = get_transmission_path("美联储降息")
    assert path is not None
    assert len(path) == 3
    assert "一级" in path[0] or "Step 1" in path[0]
    assert "二级" in path[1] or "Step 2" in path[1]
    assert "三级" in path[2] or "Step 3" in path[2]

    assert get_transmission_path("invalid_event_xyz") is None


def test_get_sector_macro_exposure():
    """验证行业在各类宏观事件下的收益/受损反向暴露索引。"""
    # 查询 "券商"
    exposure = get_sector_macro_exposure("券商")
    b_events = [item["event_id"] for item in exposure["beneficiary_in"]]
    a_events = [item["event_id"] for item in exposure["adversely_affected_in"]]

    # 降息和资本市场改革中券商受益，紧缩中券商受损
    assert "monetary_easing" in b_events
    assert "capital_market_institutional_reform" in b_events
    assert "monetary_tightening" in a_events

    # 查询 "航空"
    exposure_airline = get_sector_macro_exposure("航空")
    a_events_airline = [item["event_id"] for item in exposure_airline["adversely_affected_in"]]
    b_events_airline = [item["event_id"] for item in exposure_airline["beneficiary_in"]]
    assert "rmb_depreciation" in a_events_airline
    assert "oil_price_shock_up" in a_events_airline
    assert "rmb_appreciation" in b_events_airline

    # 空输入安全
    assert get_sector_macro_exposure("") == {"beneficiary_in": [], "adversely_affected_in": []}


def test_format_macro_event_context():
    """验证宏观事件 Prompt 注入上下文格式化生成。"""
    ctx = format_macro_event_context("油价暴涨")
    assert "【宏观事件传导图谱 - 原油与能源价格暴涨 (大宗商品与能源)】" in ctx
    assert "三级传导机制推演" in ctx
    assert "明确受益行业" in ctx
    assert "明确受损承压行业" in ctx
    assert "三桶油" in ctx or "油气勘探" in ctx
    assert "民航客运" in ctx

    # 空或未知查询返回空字符串
    assert format_macro_event_context("未知虚构宏观事件xyz") == ""
    assert format_macro_event_context("") == ""
