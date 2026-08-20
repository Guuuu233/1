"""产业链数据层核心数据结构与行业指标配置映射 (Industry Linkage Data Models).

本模块定义产业链数据层 MVP (DAV-196 / DAV-201 M1) 所需的核心 Pydantic 数据模型与行业指标配置映射：
1. `IndustryLinkageIndicator`: 单个产业链高频/核心指标定义与数据载体；
2. `IndustryLinkage`: 行业维度的完整上下游、对标与政策催化配置；
3. `INDUSTRY_LINKAGE_MAP`: 行业指标配置字典，支持消费电子、新能源车等核心赛道。

设计原则：
- 类型严谨：基于 Pydantic BaseModel，所有字段均具备显式类型注解；
- 零虚构值：配置阶段默认 `current_value` 为 None，严禁填入虚假静态数值，由运行期 Provider 实时采集或标注状态；
- 容错降级：显式标注数据源状态 (如 active, manual, pending_api 等)，支持缺失与手动数据标识。
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IndustryLinkageIndicator(BaseModel):
    """单个产业链指标定义与运行时数据载体。

    用于定义产业链上下游成本、需求、国际对标等指标的元数据配置及采集后的数据结构。
    """

    name: str = Field(
        ...,
        description="指标名称，如 'LME铜价'、'三星电子股价'、'碳酸锂价格'",
    )
    source: str = Field(
        ...,
        description="数据源标识，如 'akshare'、'yfinance'、'manual'、'pending_api'",
    )
    symbol: Optional[str] = Field(
        default=None,
        description="数据源查询代码/符号，如 '铜'、'005930.KS'、'TSLA'",
    )
    frequency: str = Field(
        default="daily",
        description="指标更新频率，如 'daily'、'monthly'、'quarterly'、'annual'",
    )
    unit: Optional[str] = Field(
        default=None,
        description="指标单位，如 '美元/吨'、'韩元'、'万元/吨'、'万部'、'%'、'辆'",
    )
    role: str = Field(
        default="upstream",
        description="指标在产业链中的角色，如 'upstream' (上游成本)、'downstream' (下游需求)、'benchmark' (国际对标)",
    )
    status: str = Field(
        default="active",
        description="指标数据状态，如 'active' (正常自动接入)、'manual' (手动录入/标注)、'pending_api' (待接入API)",
    )
    transmission_logic: Optional[str] = Field(
        default=None,
        description="产业链价格或景气度传导逻辑说明",
    )
    current_value: Optional[float] = Field(
        default=None,
        description="指标最新采集数值（配置阶段为 None，禁止硬编码虚构值，由采集器实时注入）",
    )
    mom_change: Optional[float] = Field(
        default=None,
        description="月环比变动率 (%)",
    )
    qoq_change: Optional[float] = Field(
        default=None,
        description="季度环比变动率 (%)",
    )
    yoy_change: Optional[float] = Field(
        default=None,
        description="同比变动率 (%)",
    )
    trend: Optional[str] = Field(
        default=None,
        description="趋势判断描述，如 '上升'、'平稳'、'下降'、'数据缺失'",
    )
    confidence: Optional[str] = Field(
        default=None,
        description="数据置信度评级，如 '高'、'中'、'低（待接入API）'、'低（待实现）'",
    )
    note: Optional[str] = Field(
        default=None,
        description="状态或数据获取备注说明，如 '手动'、'待接入API'",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="额外元数据扩展字典",
    )


class IndustryLinkage(BaseModel):
    """某个行业的完整产业链指标映射与配置。

    包含上游成本端、下游需求端、国际对标以及政策催化关键词。
    """

    industry_name: str = Field(
        ...,
        description="行业标准全称，如 '消费电子/半导体显示'、'新能源车/动力电池'",
    )
    upstream_cost: List[IndustryLinkageIndicator] = Field(
        default_factory=list,
        description="上游成本端核心指标列表",
    )
    downstream_demand: List[IndustryLinkageIndicator] = Field(
        default_factory=list,
        description="下游需求端核心指标列表",
    )
    international_benchmark: List[IndustryLinkageIndicator] = Field(
        default_factory=list,
        description="国际对标核心标的或指标列表",
    )
    policy_catalysts: List[str] = Field(
        default_factory=list,
        description="行业政策催化与导向关键词列表",
    )
    description: Optional[str] = Field(
        default=None,
        description="行业产业链结构与传导机制简要描述",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="行业级扩展元数据",
    )


# ---------------------------------------------------------------------------
# 行业产业链指标配置映射 (MVP 阶段支持核心 2 个行业)
# ---------------------------------------------------------------------------

INDUSTRY_LINKAGE_MAP: Dict[str, IndustryLinkage] = {
    "消费电子": IndustryLinkage(
        industry_name="消费电子/半导体显示",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="LME铜价",
                source="akshare",
                symbol="铜",
                frequency="daily",
                unit="美元/吨",
                role="upstream",
                status="active",
                transmission_logic="核心导电、引线框架与连接件原材料成本传导",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="全球智能手机出货量",
                source="manual",
                symbol=None,
                frequency="quarterly",
                unit="万部",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="下游终端消费电子需求与换机周期景气度验证",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="三星电子股价",
                source="yfinance",
                symbol="005930.KS",
                frequency="daily",
                unit="韩元",
                role="benchmark",
                status="active",
                transmission_logic="全球消费电子、存储半导体与显示面板龙头估值与景气度对标",
            ),
        ],
        policy_catalysts=[
            "消费品以旧换新",
            "超高清视频产业发展",
            "新型显示产业支持政策",
        ],
        description="消费电子/半导体显示行业产业链指标映射（上游铜价成本、下游智能手机出货量、国际对标三星电子）",
    ),
    "新能源车": IndustryLinkage(
        industry_name="新能源车/动力电池",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="碳酸锂价格",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="万元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="动力电池正极核心原材料成本传导",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="新能源车渗透率",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="%",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="终端新能源汽车市场渗透水平与消费端销量景气度",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="特斯拉交付量",
                source="manual",
                symbol="TSLA",
                frequency="quarterly",
                unit="辆",
                role="benchmark",
                status="manual",
                note="手动",
                transmission_logic="全球新能源汽车领军企业产销与需求风向标",
            ),
        ],
        policy_catalysts=[
            "新能源汽车购置税减免",
            "车路云一体化试点",
            "充换电基础设施建设支持",
        ],
        description="新能源车/动力电池行业产业链指标映射（上游碳酸锂成本、下游新能源车渗透率、国际对标特斯拉交付量）",
    ),
}


def get_industry_linkage_config(industry: str) -> Optional[IndustryLinkage]:
    """获取指定行业的产业链指标配置对象。

    Args:
        industry: 行业名称或行业关键词 (如 "消费电子", "新能源车")

    Returns:
        匹配到的 IndustryLinkage 配置对象，未匹配则返回 None
    """
    if industry in INDUSTRY_LINKAGE_MAP:
        return INDUSTRY_LINKAGE_MAP[industry]

    # 支持模糊匹配（如 "消费电子/半导体显示" -> "消费电子"）
    for key, config in INDUSTRY_LINKAGE_MAP.items():
        if key in industry or industry in config.industry_name:
            return config

    return None


def list_supported_industries() -> List[str]:
    """返回当前已支持配置的行业列表。"""
    return list(INDUSTRY_LINKAGE_MAP.keys())
