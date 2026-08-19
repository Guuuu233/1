"""针对 tradingagents.knowledge.industry_linkage 的完备单元测试。"""

import pytest
from tradingagents.knowledge.industry_linkage import (
    INDUSTRY_PROFILES,
    IndustryProfile,
    MacroSensitivity,
    CycleProfile,
    RiskMatrix,
    get_all_industries,
    get_all_industry_names,
    get_industry_profile,
    get_upstream_downstream_chain,
    get_macro_sensitivity_matrix,
    get_industry_risk_profile,
    format_industry_deep_context,
    search_industries,
)


def test_industry_coverage_at_least_20():
    """验证行业图谱覆盖量达到并超过 20+ 个核心行业。"""
    industries = get_all_industries()
    assert len(industries) >= 20, f"行业数量不足 20 个，当前为 {len(industries)}"
    assert len(INDUSTRY_PROFILES) >= 20


def test_every_industry_has_complete_fields():
    """验证每一个行业画像的所有维度与字段均完整、非空且结构合规。"""
    for ind_id, profile in INDUSTRY_PROFILES.items():
        assert profile.industry_id == ind_id
        assert isinstance(profile.industry_name, str) and len(profile.industry_name) > 0
        assert isinstance(profile.category, str) and len(profile.category) > 0
        assert isinstance(profile.aliases, list) and len(profile.aliases) > 0

        # 产业链上下游
        assert isinstance(profile.upstream, list) and len(profile.upstream) >= 2, f"{ind_id} 上游环节不足"
        assert isinstance(profile.downstream, list) and len(profile.downstream) >= 2, f"{ind_id} 下游应用不足"
        assert isinstance(profile.core_inputs, list) and len(profile.core_inputs) >= 2, f"{ind_id} 核心要素不足"
        assert isinstance(profile.pricing_power, str) and len(profile.pricing_power) > 10, f"{ind_id} 议价权描述过短"

        # 宏观敏感度
        ms = profile.macro_sensitivity
        assert isinstance(ms, MacroSensitivity)
        assert len(ms.interest_rate) > 0
        assert len(ms.fx_rate) > 0
        assert len(ms.commodity_inflation) > 0
        assert len(ms.liquidity) > 0
        assert isinstance(ms.policy_drivers, list) and len(ms.policy_drivers) >= 2
        assert len(ms.global_macro_linkage) > 5

        # 周期属性
        cp = profile.cycle_profile
        assert isinstance(cp, CycleProfile)
        assert len(cp.cycle_type) > 0
        assert len(cp.typical_length) > 0
        assert len(cp.capacity_lag) > 0
        assert isinstance(cp.key_cycle_indicators, list) and len(cp.key_cycle_indicators) >= 2

        # 风险矩阵
        rk = profile.risks
        assert isinstance(rk, RiskMatrix)
        assert len(rk.geopolitical) >= 1
        assert len(rk.supply_chain_bottlenecks) >= 1
        assert len(rk.technology_substitution) >= 1
        assert len(rk.policy_regulatory) >= 1
        assert len(rk.demand_cliff) >= 1

        # 跟踪指标与细分赛道
        assert isinstance(profile.key_metrics, list) and len(profile.key_metrics) >= 3
        assert isinstance(profile.representative_segments, list) and len(profile.representative_segments) >= 2


def test_get_industry_names_list():
    """验证获取所有行业名称列表。"""
    names = get_all_industry_names()
    assert len(names) == len(INDUSTRY_PROFILES)
    assert "半导体与集成电路" in names
    assert "人工智能与算力服务" in names
    assert "新能源汽车与智能汽车" in names
    assert "商业银行与信贷" in names
    assert "贵金属与稀缺资源" in names


@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("semiconductor", "semiconductor"),
        ("半导体与集成电路", "semiconductor"),
        ("芯片", "semiconductor"),
        ("晶圆", "semiconductor"),
        ("IC", "semiconductor"),
        ("ai_computing", "ai_computing"),
        ("人工智能", "ai_computing"),
        ("大模型", "ai_computing"),
        ("算力", "ai_computing"),
        ("nev_auto", "nev_auto"),
        ("新能源汽车", "nev_auto"),
        ("电动车", "nev_auto"),
        ("photovoltaic_storage", "photovoltaic_storage"),
        ("光伏", "photovoltaic_storage"),
        ("储能", "photovoltaic_storage"),
        ("lithium_battery", "lithium_battery"),
        ("碳酸锂", "lithium_battery"),
        ("动力电池", "lithium_battery"),
        ("biopharma", "biopharma"),
        ("创新药", "biopharma"),
        ("CXO", "biopharma"),
        ("medical_devices", "medical_devices"),
        ("医疗器械", "medical_devices"),
        ("IVD", "medical_devices"),
        ("consumer_electronics", "consumer_electronics"),
        ("果链", "consumer_electronics"),
        ("苹果产业链", "consumer_electronics"),
        ("liquor_beverage", "liquor_beverage"),
        ("白酒", "liquor_beverage"),
        ("茅台", "liquor_beverage"),
        ("food_beverage", "food_beverage"),
        ("调味品", "food_beverage"),
        ("乳制品", "food_beverage"),
        ("home_appliances", "home_appliances"),
        ("家电", "home_appliances"),
        ("空调", "home_appliances"),
        ("banking", "banking"),
        ("银行", "banking"),
        ("国有大行", "banking"),
        ("securities", "securities"),
        ("券商", "securities"),
        ("投行", "securities"),
        ("insurance_financials", "insurance_financials"),
        ("保险", "insurance_financials"),
        ("寿险", "insurance_financials"),
        ("steel_ferrous", "steel_ferrous"),
        ("钢铁", "steel_ferrous"),
        ("螺纹钢", "steel_ferrous"),
        ("nonferrous_metals", "nonferrous_metals"),
        ("铜", "nonferrous_metals"),
        ("电解铝", "nonferrous_metals"),
        ("precious_metals", "precious_metals"),
        ("黄金", "precious_metals"),
        ("稀土", "precious_metals"),
        ("petrochemicals", "petrochemicals"),
        ("石油", "petrochemicals"),
        ("炼化", "petrochemicals"),
        ("coal_energy", "coal_energy"),
        ("煤炭", "coal_energy"),
        ("动力煤", "coal_energy"),
        ("power_utilities", "power_utilities"),
        ("电力", "power_utilities"),
        ("水电", "power_utilities"),
        ("核电", "power_utilities"),
        ("real_estate", "real_estate"),
        ("房地产", "real_estate"),
        ("地产", "real_estate"),
        ("construction_materials", "construction_materials"),
        ("基建", "construction_materials"),
        ("水泥", "construction_materials"),
        ("industrial_machinery", "industrial_machinery"),
        ("工业母机", "industrial_machinery"),
        ("机床", "industrial_machinery"),
        ("defense_military", "defense_military"),
        ("军工", "defense_military"),
        ("航天", "defense_military"),
        ("logistics_shipping", "logistics_shipping"),
        ("航运", "logistics_shipping"),
        ("集运", "logistics_shipping"),
        ("telecom_optical", "telecom_optical"),
        ("光模块", "telecom_optical"),
        ("运营商", "telecom_optical"),
        ("agriculture_breeding", "agriculture_breeding"),
        ("生猪养殖", "agriculture_breeding"),
        ("养猪", "agriculture_breeding"),
    ],
)
def test_get_industry_profile_matching(query, expected_id):
    """验证行业ID、中文全名及多类别名的精确与模糊匹配正确性。"""
    profile = get_industry_profile(query)
    assert profile is not None, f"未能匹配到行业: {query}"
    assert profile.industry_id == expected_id


def test_get_industry_profile_invalid():
    """验证无效行业查询安全返回 None。"""
    assert get_industry_profile("") is None
    assert get_industry_profile("   ") is None
    assert get_industry_profile(None) is None  # type: ignore
    assert get_industry_profile(12345) is None  # type: ignore
    assert get_industry_profile("不存在的火星采矿行业xyz999") is None


def test_search_industries():
    """验证行业关键词多字段联合搜索。"""
    # 搜索光刻机应命中半导体
    matches = search_industries("光刻机")
    ids = [m.industry_id for m in matches]
    assert "semiconductor" in ids

    # 搜索算力应命中 AI 与光通信
    matches_ai = search_industries("算力")
    ids_ai = [m.industry_id for m in matches_ai]
    assert "ai_computing" in ids_ai

    # 搜索铜应命中有色金属
    matches_copper = search_industries("铜")
    ids_copper = [m.industry_id for m in matches_copper]
    assert "nonferrous_metals" in ids_copper

    # 异常输入
    assert search_industries("") == []
    assert search_industries(None) == []  # type: ignore


def test_get_upstream_downstream_chain():
    """验证上下游结构化数据提取函数。"""
    chain = get_upstream_downstream_chain("半导体")
    assert chain is not None
    assert chain["industry_id"] == "semiconductor"
    assert "光刻机/刻蚀机/薄膜沉积/离子注入等半导体设备" in chain["upstream"]
    assert "AI算力数据中心/服务器" in chain["downstream"]
    assert "高纯电子化学品" in chain["core_inputs"]
    assert "设计端龙头" in chain["pricing_power"]

    assert get_upstream_downstream_chain("invalid_xyz") is None


def test_get_macro_sensitivity_matrix():
    """验证宏观敏感度矩阵提取函数。"""
    matrix = get_macro_sensitivity_matrix("新能源汽车")
    assert matrix is not None
    assert matrix["industry_id"] == "nev_auto"
    assert "大宗可选消费" in matrix["interest_rate_sensitivity"]
    assert "碳酸锂" in matrix["commodity_inflation_sensitivity"]
    assert len(matrix["policy_drivers"]) >= 2
    assert "反补贴" in matrix["global_macro_linkage"]

    assert get_macro_sensitivity_matrix("invalid_xyz") is None


def test_get_industry_risk_profile():
    """验证风险图谱提取函数。"""
    risk_info = get_industry_risk_profile("创新药")
    assert risk_info is not None
    assert risk_info["industry_id"] == "biopharma"
    assert any("生物安全法案" in r for r in risk_info["geopolitical_risks"])
    assert any("培养基" in r for r in risk_info["supply_chain_bottlenecks"])
    assert any("集采" in r for r in risk_info["policy_regulatory"])

    assert get_industry_risk_profile("invalid_xyz") is None


def test_format_industry_deep_context():
    """验证注入 LLM Prompt 的上下文文本生成。"""
    ctx = format_industry_deep_context("光模块")
    assert "【行业常识知识库 - 通信网络与光通信 (科技/TMT)】" in ctx
    assert "产业链上下游穿透" in ctx
    assert "宏观与周期敏感度" in ctx
    assert "风险矩阵与监控指标" in ctx
    assert "800G/1.6T" in ctx

    # 空查询与未知行业返回空字符串
    assert format_industry_deep_context("未知虚构行业999") == ""
    assert format_industry_deep_context("") == ""
