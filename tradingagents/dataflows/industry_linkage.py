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

from typing import Any, Dict, List, Optional, Union
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
# 行业产业链指标配置映射 (支持核心 5 个行业: 消费电子、新能源车、半导体、石油化工、金融地产)
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
    "半导体": IndustryLinkage(
        industry_name="半导体/集成电路",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="半导体硅片价格",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="美元/片",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="晶圆制造核心大硅片衬底原材料成本传导",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="全球半导体销售额",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="亿美元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="SIA月度全球半导体产业销售总额与终端景气度验证",
            ),
            IndustryLinkageIndicator(
                name="DRAM存储芯片现货价",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="美元",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="存储芯片现货价格走势与半导体周期供需拐点风向标",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="费城半导体指数",
                source="yfinance",
                symbol="^SOX",
                frequency="daily",
                unit="点",
                role="benchmark",
                status="active",
                transmission_logic="全球半导体行业景气度、估值体系与技术周期核心风向标",
            ),
            IndustryLinkageIndicator(
                name="台积电股价",
                source="yfinance",
                symbol="TSM",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球先进制程晶圆代工龙头业绩、产能利用率与资本开支对标",
            ),
        ],
        policy_catalysts=[
            "国家大基金产业投资",
            "集成电路重大专项支持",
            "半导体关键设备与材料国产替代",
            "先进制程自主可控政策",
        ],
        description="半导体/集成电路行业产业链指标映射（上游硅片成本、下游全球半导体销售额与存储现货价、国际对标费城半导体指数SOX与台积电TSM）",
    ),
    "石油化工": IndustryLinkage(
        industry_name="石油化工/基础化工",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="布伦特原油价格",
                source="yfinance",
                symbol="BZ=F",
                frequency="daily",
                unit="美元/桶",
                role="upstream",
                status="active",
                transmission_logic="石油化工产业链最源头大宗原油原材料成本传导",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="国内成品油消费量",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="万吨",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="下游交通运输与工业基础燃料终端消费需求景气度",
            ),
            IndustryLinkageIndicator(
                name="聚酯长丝开工率",
                source="pending_api",
                symbol=None,
                frequency="weekly",
                unit="%",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="下游纺织服装与聚酯化纤织造端开工与补库需求验证",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="埃克森美孚股价",
                source="yfinance",
                symbol="XOM",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球综合性一体化石油石化龙头估值、油气开采与炼化盈利对标",
            ),
        ],
        policy_catalysts=[
            "能耗双控向碳排放双控转变",
            "成品油出口配额优化",
            "石化产业布局规划方案",
            "绿色石化与高端新材料支持",
        ],
        description="石油化工/基础化工行业产业链指标映射（上游布伦特原油成本、下游成品油消费与聚酯开工率、国际对标埃克森美孚XOM）",
    ),
    "金融地产": IndustryLinkage(
        industry_name="金融地产/商业银行与房地产",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="银行间同业拆借利率",
                source="pending_api",
                symbol="Shibor_3M",
                frequency="daily",
                unit="%",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="商业银行批发性资金获取与同业负债综合成本传导",
            ),
            IndustryLinkageIndicator(
                name="房企综合融资成本",
                source="manual",
                symbol=None,
                frequency="quarterly",
                unit="%",
                role="upstream",
                status="manual",
                note="手动",
                transmission_logic="房地产企业境内外债务融资利率、利息资本化与现金流压力",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="30大中城市商品房成交面积",
                source="pending_api",
                symbol=None,
                frequency="weekly",
                unit="万平方米",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="终端商品房销售高频周度景气度与居民部门购房加杠杆意愿",
            ),
            IndustryLinkageIndicator(
                name="社会融资规模增量",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="万亿元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="实体经济全社会信贷投放与有效融资需求扩张验证",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="摩根大通股价",
                source="yfinance",
                symbol="JPM",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球系统重要性商业银行龙头估值中枢与净息差对标",
            ),
            IndustryLinkageIndicator(
                name="标普500金融行业指数",
                source="yfinance",
                symbol="XLF",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="海外成熟市场多元金融与银行地产板块整体周期景气度风向标",
            ),
        ],
        policy_catalysts=[
            "存量房贷利率调降政策",
            "地方政府隐性债务化解置换",
            "房地产保交房与收储支持政策",
            "降准降息与结构性货币政策工具支持",
        ],
        description="金融地产/商业银行与房地产行业产业链指标映射（资金端同业负债与融资成本、资产端社融与地产销售、国际对标摩根大通JPM与金融ETF XLF）",
    ),
}


def get_industry_linkage_config(industry: str) -> Optional[IndustryLinkage]:
    """获取指定行业的产业链指标配置对象。

    Args:
        industry: 行业名称或行业关键词 (如 "消费电子", "新能源车", "半导体", "石油化工", "金融地产", "银行", "房地产")

    Returns:
        匹配到的 IndustryLinkage 配置对象，未匹配则返回 None
    """
    if not industry or not isinstance(industry, str):
        return None

    clean_industry = industry.strip()
    if not clean_industry:
        return None

    if clean_industry in INDUSTRY_LINKAGE_MAP:
        return INDUSTRY_LINKAGE_MAP[clean_industry]

    # 支持模糊匹配（如 "消费电子/半导体显示" -> "消费电子"，"银行" -> "金融地产"）
    for key, config in INDUSTRY_LINKAGE_MAP.items():
        if key in clean_industry or clean_industry in key or clean_industry in config.industry_name or config.industry_name in clean_industry:
            return config

    return None


def list_supported_industries() -> List[str]:
    """返回当前已支持配置的行业列表。"""
    return list(INDUSTRY_LINKAGE_MAP.keys())


def _format_indicator_item(ind: Dict[str, Any]) -> str:
    """格式化单个产业链指标为 Markdown 行。"""
    name = ind.get("name", "未命名指标")
    val = ind.get("current_value")
    unit = ind.get("unit") or ""
    trend = ind.get("trend")
    confidence = ind.get("confidence")
    note = ind.get("note")
    logic = ind.get("transmission_logic")
    mom = ind.get("mom_change")
    qoq = ind.get("qoq_change")

    if val is not None and trend != "数据缺失":
        val_str = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
        unit_str = f" {unit}" if unit else ""
        mom_str = f"，月环比 {mom:+.2f}%" if isinstance(mom, (int, float)) else ""
        qoq_str = f"，季度环比 {qoq:+.2f}%" if isinstance(qoq, (int, float)) else ""
        trend_str = f"，趋势：{trend}" if trend else ""
        conf_str = f"（置信度：{confidence}）" if confidence else ""
        line = f"  * {name}：{val_str}{unit_str}{mom_str}{qoq_str}{trend_str}{conf_str}"
    else:
        reason = note or confidence or "数据缺失"
        line = f"  * 【数据缺失】{name}：{reason}"

    if logic:
        line += f"\n    - 传导逻辑：{logic}"
    return line


def format_industry_linkage_for_prompt(
    industry_linkage: Optional[Union[Dict[str, Any], IndustryLinkage]]
) -> str:
    """将产业链联动数据格式化为可直接注入 Prompt 的结构化文本。

    Args:
        industry_linkage: 产业链数据字典或 IndustryLinkage 模型实例，若为 None 则返回空字符串

    Returns:
        结构化 Markdown 文本段落；若无数据或未映射则返回空字符串
    """
    if not industry_linkage:
        return ""

    if hasattr(industry_linkage, "model_dump"):
        data = industry_linkage.model_dump()
    elif isinstance(industry_linkage, dict):
        data = industry_linkage
    else:
        return ""

    industry_name = data.get("industry_name")
    if not industry_name:
        return ""

    lines = [f"【产业链联想数据】：{industry_name}"]

    upstream = data.get("upstream_cost") or []
    if upstream:
        lines.append("- 上游成本端核心指标：")
        for ind in upstream:
            ind_dict = ind if isinstance(ind, dict) else ind.model_dump()
            lines.append(_format_indicator_item(ind_dict))

    downstream = data.get("downstream_demand") or []
    if downstream:
        lines.append("- 下游需求端核心指标：")
        for ind in downstream:
            ind_dict = ind if isinstance(ind, dict) else ind.model_dump()
            lines.append(_format_indicator_item(ind_dict))

    benchmark = data.get("international_benchmark") or []
    if benchmark:
        lines.append("- 国际对标核心标的/指标：")
        for ind in benchmark:
            ind_dict = ind if isinstance(ind, dict) else ind.model_dump()
            lines.append(_format_indicator_item(ind_dict))

    catalysts = data.get("policy_catalysts") or []
    if catalysts:
        cat_str = "、".join(str(c) for c in catalysts)
        lines.append(f"- 行业政策催化关键词：{cat_str}")

    return "\n".join(lines)
