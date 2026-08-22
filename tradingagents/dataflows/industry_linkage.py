"""产业链数据层核心数据结构与行业指标配置映射 (Industry Linkage Data Models).

本模块定义产业链数据层 (DAV-196 / DAV-201 / DAV-274) 所需的核心 Pydantic 数据模型与行业指标配置映射：
1. `IndustryLinkageIndicator`: 单个产业链高频/核心指标定义与数据载体；
2. `IndustryLinkage`: 行业维度的完整上下游、对标与政策催化配置；
3. `INDUSTRY_LINKAGE_MAP`: 行业指标配置字典，覆盖知识库全部 27 个行业。

设计原则：
- 类型严谨：基于 Pydantic BaseModel，所有字段均具备显式类型注解；
- 零虚构值：配置阶段默认 `current_value` 为 None，严禁填入虚假静态数值，由运行期 Provider 实时采集或标注状态；
- 容错降级：显式标注数据源状态 (如 active, manual, pending_api 等)，支持缺失与手动数据标识；
- 权威对齐：以 `tradingagents/knowledge/industry_linkage.py` 现有 27 个 `industry_name` 为权威清单，原地扩展。
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from tradingagents.knowledge.industry_linkage import (
    get_all_industry_names,
    get_industry_profile,
)


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
        description="行业标准全称，与知识库 27 行业权威命名对齐",
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
# 行业产业链指标配置映射 (涵盖知识库全部 27 个行业，对齐权威 industry_name)
# ---------------------------------------------------------------------------

_BASE_INDUSTRY_LINKAGE_MAP: Dict[str, IndustryLinkage] = {
    # 1. 半导体与集成电路 (科技/TMT)
    "半导体与集成电路": IndustryLinkage(
        industry_name="半导体与集成电路",
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
            IndustryLinkageIndicator(
                name="半导体设备采购指数",
                source="pending_api",
                symbol=None,
                frequency="quarterly",
                unit="点",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="前道光刻/刻蚀/薄膜沉积设备资本开支成本",
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
        description="半导体与集成电路行业产业链指标映射（上游大硅片成本、下游全球半导体销售额与存储现货价、国际对标费城半导体指数SOX与台积电TSM）",
    ),

    # 2. 人工智能与算力服务 (科技/TMT)
    "人工智能与算力服务": IndustryLinkage(
        industry_name="人工智能与算力服务",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="HBM高带宽内存价格",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="美元/GB",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="高端AI加速算力卡核心存储BOM成本传导",
            ),
            IndustryLinkageIndicator(
                name="IDC机房平均电价",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="元/千瓦时",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="智算中心与数据中心核心运营能耗成本",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="全球大模型算力需求规模",
                source="manual",
                symbol=None,
                frequency="quarterly",
                unit="EFLOPS",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="主流AI大模型训练与推理集群Token消耗需求景气度",
            ),
            IndustryLinkageIndicator(
                name="AI服务器季度出货量",
                source="pending_api",
                symbol=None,
                frequency="quarterly",
                unit="万台",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="下游数据中心与云厂商AI算力基础设施采购规模",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="英伟达股价",
                source="yfinance",
                symbol="NVDA",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球AI算力芯片与GPU基础设施领军龙头估值风向标",
            ),
            IndustryLinkageIndicator(
                name="微软股价",
                source="yfinance",
                symbol="MSFT",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球大模型云端商业化变现与AI软件基础设施对标",
            ),
        ],
        policy_catalysts=[
            "全国一体化算力网络东数西算工程",
            "人工智能+行动意见",
            "自主可控算力底座支持政策",
            "数据要素市场化配置改革",
        ],
        description="人工智能与算力服务产业链指标映射（上游HBM与算力能耗成本、下游算力需求与AI服务器出货、国际对标英伟达NVDA与微软MSFT）",
    ),

    # 3. 新能源汽车与智能汽车 (先进制造)
    "新能源汽车与智能汽车": IndustryLinkage(
        industry_name="新能源汽车与智能汽车",
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
            IndustryLinkageIndicator(
                name="汽车用冷轧板价格",
                source="pending_api",
                symbol=None,
                frequency="weekly",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="汽车车身冲压与制造基础钢材成本传导",
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
            IndustryLinkageIndicator(
                name="乘联会乘用车月度批发销量",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="万辆",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="全国乘用车厂商批发与终端零售动销景气度",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="特斯拉股价",
                source="yfinance",
                symbol="TSLA",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球新能源汽车领军企业估值、产销与价格调整风向标",
            ),
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
            "汽车以旧换新置换补贴",
        ],
        description="新能源汽车与智能汽车产业链指标映射（上游碳酸锂成本、下游新能源车渗透率与销量、国际对标特斯拉TSLA）",
    ),

    # 4. 光伏与储能系统 (绿色能源)
    "光伏与储能系统": IndustryLinkage(
        industry_name="光伏与储能系统",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="多晶硅致密料价格",
                source="pending_api",
                symbol=None,
                frequency="weekly",
                unit="元/千克",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="光伏全产业链最源头硅料成本与供需博弈传导",
            ),
            IndustryLinkageIndicator(
                name="光伏级白银价格",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="元/千克",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="电池片正银背银电极金属浆料核心成本传导",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="国内月度新增光伏装机量",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="GW",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="国家能源局月度集中式与分布式光伏新增并网装机需求",
            ),
            IndustryLinkageIndicator(
                name="光伏组件单月出口金额",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="亿美元",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="海外欧洲/中东/新兴市场光伏地面电站装机采购景气度",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="第一太阳能股价",
                source="yfinance",
                symbol="FSLR",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="美国本土薄膜光伏组件领军企业估值与海外贸易政策对标",
            ),
            IndustryLinkageIndicator(
                name="Enphase能源股价",
                source="yfinance",
                symbol="ENPH",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球微型逆变器与户用储能系统领军企业景气度对标",
            ),
        ],
        policy_catalysts=[
            "全球碳中和目标与可再生能源配额",
            "国内电网消纳与配储政策",
            "绿电绿证交易市场化改革",
            "超长期特别国债支持新能源大基地",
        ],
        description="光伏与储能系统产业链指标映射（上游多晶硅料成本、下游光伏装机量与组件出口、国际对标First Solar与Enphase）",
    ),

    # 5. 动力电池与储能电池材料 (绿色能源)
    "动力电池与储能电池材料": IndustryLinkage(
        industry_name="动力电池与储能电池材料",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="电池级碳酸锂价格",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="万元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="动力电池正极材料与电芯最核心金属原材料成本传导",
            ),
            IndustryLinkageIndicator(
                name="六氟磷酸锂价格",
                source="pending_api",
                symbol=None,
                frequency="weekly",
                unit="万元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="电解液核心溶质成本传导与产能供需风向标",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="动力电池月度装车量",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="GWh",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="动力电池产业创新联盟月度动力电池装车与产销数据",
            ),
            IndustryLinkageIndicator(
                name="全球新型储能新增装机量",
                source="pending_api",
                symbol=None,
                frequency="quarterly",
                unit="GWh",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="电网侧与工商业储能专用电芯需求景气度",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="LG新能源股价",
                source="yfinance",
                symbol="373220.KS",
                frequency="daily",
                unit="韩元",
                role="benchmark",
                status="active",
                transmission_logic="全球动力电池装车量第二大企业估值与海外欧美市场份额对标",
            ),
            IndustryLinkageIndicator(
                name="雅宝公司股价",
                source="yfinance",
                symbol="ALB",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球最大锂矿锂盐生产商估值与全球锂资源周期对标",
            ),
        ],
        policy_catalysts=[
            "新能源汽车动力电池产业发展规划",
            "新型储能高质量发展行动方案",
            "欧盟新电池法案碳足迹要求",
            "固态电池国家重点研发专项",
        ],
        description="动力电池与储能电池材料产业链指标映射（上游碳酸锂与六氟磷酸锂成本、下游动力电池装车量、国际对标LG新能源与雅宝ALB）",
    ),

    # 6. 医药生物与创新药 (医药健康)
    "医药生物与创新药": IndustryLinkage(
        industry_name="医药生物与创新药",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="医药中间体价格指数",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="点",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="化学原料药与制剂上游基础精细化工原材料成本传导",
            ),
            IndustryLinkageIndicator(
                name="实验动物价格",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="万元/只",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="临床前药效与安全性评价实验模型核心要素成本",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="全球Biotech医疗健康投融资金额",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="亿美元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="全球生物医药一级市场投融资景气度与CXO新签订单先导指标",
            ),
            IndustryLinkageIndicator(
                name="中国创新药海外授权License-out总额",
                source="pending_api",
                symbol=None,
                frequency="quarterly",
                unit="亿美元",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="国内创新药出海管线商业化价值与首付款里程碑验证",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="纳斯达克生物科技指数",
                source="yfinance",
                symbol="^NBI",
                frequency="daily",
                unit="点",
                role="benchmark",
                status="active",
                transmission_logic="全球创新药与Biotech板块估值中枢与风险偏好核心风向标",
            ),
            IndustryLinkageIndicator(
                name="礼来公司股价",
                source="yfinance",
                symbol="LLY",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球GLP-1减肥药与重磅创新药领军药企市值与研发对标",
            ),
        ],
        policy_catalysts=[
            "国家医保药品目录动态调整谈判",
            "仿制药一致性评价与国家组织集采",
            "全链条支持创新药发展政策",
            "中医药振兴发展重大工程",
        ],
        description="医药生物与创新药产业链指标映射（上游中间体与实验成本、下游投融资与出海授权、国际对标纳斯达克NBI与礼来LLY）",
    ),

    # 7. 医疗器械与医疗服务 (医药健康)
    "医疗器械与医疗服务": IndustryLinkage(
        industry_name="医疗器械与医疗服务",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="医用级钛合金价格",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="万元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="骨科植入与高值耗材核心医用金属原材料成本传导",
            ),
            IndustryLinkageIndicator(
                name="高精度传感器元器件成本",
                source="pending_api",
                symbol=None,
                frequency="quarterly",
                unit="点",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="医学影像设备与监护仪器核心传感器及芯片采购成本",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="公立医院医疗设备招投标金额",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="亿元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="各级公立医院医疗设备更新采购招标与入院装机景气度",
            ),
            IndustryLinkageIndicator(
                name="高值医用耗材集采采购量",
                source="pending_api",
                symbol=None,
                frequency="quarterly",
                unit="万套",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="国家与省际联盟集采报量执行与公立医院采购意愿",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="美敦力股价",
                source="yfinance",
                symbol="MDT",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球综合性医疗器械与高值耗材领军巨头估值对标",
            ),
            IndustryLinkageIndicator(
                name="直觉外科股价",
                source="yfinance",
                symbol="ISRG",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球达芬奇手术机器人与创新微创医疗设备对标",
            ),
        ],
        policy_catalysts=[
            "医疗领域大规模设备更新财政贴息贷款",
            "高值医用耗材国家集中带量采购",
            "千县工程与县域医疗中心新基建",
            "DRG/DIP医保支付方式改革",
        ],
        description="医疗器械与医疗服务产业链指标映射（上游医用材料成本、下游医院设备招标采购、国际对标美敦力MDT与直觉外科ISRG）",
    ),

    # 8. 消费电子与智能终端 (科技/TMT)
    "消费电子与智能终端": IndustryLinkage(
        industry_name="消费电子与智能终端",
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
            IndustryLinkageIndicator(
                name="显示面板主流尺寸报价",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="美元/片",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="智能手机与PC显示屏模组BOM成本传导",
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
            IndustryLinkageIndicator(
                name="全球PC出货量增速",
                source="pending_api",
                symbol=None,
                frequency="quarterly",
                unit="%",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="IDC/Canalys季度个人电脑与笔记本出货景气度",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="苹果公司股价",
                source="yfinance",
                symbol="AAPL",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球消费电子终端创新生态与果链供应链景气度核心风向标",
            ),
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
            "消费品以旧换新补贴政策",
            "超高清视频产业发展规划",
            "新型显示产业支持政策",
            "AI端侧智能硬件推广支持",
        ],
        description="消费电子与智能终端产业链指标映射（上游铜价与面板成本、下游智能手机出货量、国际对标苹果AAPL与三星电子005930.KS）",
    ),

    # 9. 白酒与精制茶酒 (大消费)
    "白酒与精制茶酒": IndustryLinkage(
        industry_name="白酒与精制茶酒",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="酿酒高粱原粮收购均价",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="元/公斤",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="优质红缨子高粱等酿造核心原粮成本传导",
            ),
            IndustryLinkageIndicator(
                name="白酒包材纸箱玻璃成本指数",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="点",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="陶瓷酒瓶、玻璃瓶与外包装纸盒物料成本传导",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="飞天茅台一批价",
                source="manual",
                symbol=None,
                frequency="daily",
                unit="元/瓶",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="高端白酒渠道库存蓄水池与商务社交消费景气度核心风向标",
            ),
            IndustryLinkageIndicator(
                name="烟酒店与商超白酒渠道动销率",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="%",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="传统渠道经销商动销周转与终端开瓶率跟踪",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="帝亚吉欧股价",
                source="yfinance",
                symbol="DEO",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球最大烈酒洋酒巨头估值中枢与全球高端酒水消费景气对标",
            ),
            IndustryLinkageIndicator(
                name="保乐力加股价",
                source="yfinance",
                symbol="RI.PA",
                frequency="daily",
                unit="欧元",
                role="benchmark",
                status="active",
                transmission_logic="全球高端烈酒与葡萄酒品牌集团估值与渠道去库周期对标",
            ),
        ],
        policy_catalysts=[
            "消费税改革与征收环节后移预期",
            "商务接待与公务用餐规范政策",
            "扩大内需与促进消费政策",
            "白酒产业高质量发展指导意见",
        ],
        description="白酒与精制茶酒产业链指标映射（上游原粮包材成本、下游茅台批价与渠道动销、国际对标帝亚吉欧DEO与保乐力加RI.PA）",
    ),

    # 10. 大众食品与饮料 (大消费)
    "大众食品与饮料": IndustryLinkage(
        industry_name="大众食品与饮料",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="生鲜乳主产区收购均价",
                source="pending_api",
                symbol=None,
                frequency="weekly",
                unit="元/公斤",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="乳制品加工企业原奶核心采购成本与喷粉周期传导",
            ),
            IndustryLinkageIndicator(
                name="白糖大宗现货价格",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="软饮料、调味品与烘焙休闲食品核心糖类成本传导",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="全国餐饮收入月度同比增速",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="%",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="国家统计局月度餐饮消费总额与调味品B端餐饮需求景气度",
            ),
            IndustryLinkageIndicator(
                name="量贩零食渠道月度出货额",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="亿元",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="休闲食品量贩店与即时零售渠道终端动销增速",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="雀巢公司股价",
                source="yfinance",
                symbol="NESN.SW",
                frequency="daily",
                unit="瑞士法郎",
                role="benchmark",
                status="active",
                transmission_logic="全球最大综合包装食品与饮料巨头估值与毛利率韧性对标",
            ),
            IndustryLinkageIndicator(
                name="可口可乐股价",
                source="yfinance",
                symbol="KO",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球软饮料领军企业渠道掌控力与成本转嫁定价权对标",
            ),
        ],
        policy_catalysts=[
            "食品安全国家标准与全过程溯源体系建设",
            "农村电商与县域商业体系建设",
            "健康中国减糖减油减盐引导政策",
            "促进餐饮业高质量发展指导意见",
        ],
        description="大众食品与饮料产业链指标映射（上游生鲜乳与白糖成本、下游餐饮收入与量贩出货、国际对标雀巢NESN与可口可乐KO）",
    ),

    # 11. 家用电器与智能家居 (大消费)
    "家用电器与智能家居": IndustryLinkage(
        industry_name="家用电器与智能家居",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="LME铜价",
                source="akshare",
                symbol="铜",
                frequency="daily",
                unit="美元/吨",
                role="upstream",
                status="active",
                transmission_logic="空调制冷铜管与电机核心金属原材料成本传导",
            ),
            IndustryLinkageIndicator(
                name="ABS塑料颗粒现货价",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="家电外壳注塑与结构件化工塑料原材料成本传导",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="产业在线空调月度内销排产",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="万台",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="白色家电核心空调品类月度出货与渠道铺货景气度",
            ),
            IndustryLinkageIndicator(
                name="家电产品月度出口金额增速",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="%",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="海外欧美与新兴市场耐用消费品出口外需景气度",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="惠而浦股价",
                source="yfinance",
                symbol="WHR",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="欧美白电龙头企业估值中枢与海外家电消费需求对标",
            ),
            IndustryLinkageIndicator(
                name="大金工业股价",
                source="yfinance",
                symbol="6367.T",
                frequency="daily",
                unit="日元",
                role="benchmark",
                status="active",
                transmission_logic="全球暖通空调与商用氟化工领军企业技术与盈利对标",
            ),
        ],
        policy_catalysts=[
            "消费品以旧换新绿色智能家电补贴政策",
            "保交楼与商品房竣工交付支持政策",
            "家电能效新国标实施",
            "推动绿色智能家电下乡",
        ],
        description="家用电器与智能家居产业链指标映射（上游铜价与ABS塑料成本、下游空调排产与家电出口、国际对标惠而浦WHR与大金工业6367.T）",
    ),

    # 12. 商业银行与信贷 (金融地产)
    "商业银行与信贷": IndustryLinkage(
        industry_name="商业银行与信贷",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="银行间同业拆借利率Shibor",
                source="pending_api",
                symbol="Shibor_3M",
                frequency="daily",
                unit="%",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="商业银行同业负债与批发性资金综合获取成本传导",
            ),
            IndustryLinkageIndicator(
                name="国有大行1年期定期存款挂牌利率",
                source="manual",
                symbol=None,
                frequency="quarterly",
                unit="%",
                role="upstream",
                status="manual",
                note="手动",
                transmission_logic="商业银行核心负债端存款付息成本与挂牌调降节奏",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="每月新增人民币贷款总额",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="万亿元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="央行月度信贷投放规模与实体经济融资有效需求验证",
            ),
            IndustryLinkageIndicator(
                name="贷款市场报价利率LPR_1Y",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="%",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="资产端企业贷款与零售贷款基准定价中枢",
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
                name="汇丰控股股价",
                source="yfinance",
                symbol="HSBC",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="跨国综合性商业银行巨头资产质量与跨境财富管理对标",
            ),
        ],
        policy_catalysts=[
            "存量房贷利率调降政策",
            "地方政府隐性债务置换化解",
            "央行降准降息与结构性货币工具",
            "普惠小微与科技创新再贷款支持",
        ],
        description="商业银行与信贷产业链指标映射（资金端同业负债与存款成本、资产端信贷投放与LPR定价、国际对标摩根大通JPM与汇丰HSBC）",
    ),

    # 13. 证券公司与资本市场 (金融地产)
    "证券公司与资本市场": IndustryLinkage(
        industry_name="证券公司与资本市场",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="转融通融券利率中枢",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="%",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="券商两融业务转融通拆借资金与证券获取成本",
            ),
            IndustryLinkageIndicator(
                name="券商短融发债票面利率",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="%",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="券商公开发行短期融资券与次级债重资本融资成本",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="A股全市场单日成交金额",
                source="manual",
                symbol=None,
                frequency="daily",
                unit="亿元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="沪深北交易所单日总成交额与经纪业务佣金弹性核心驱动",
            ),
            IndustryLinkageIndicator(
                name="全市场融资融券余额",
                source="manual",
                symbol=None,
                frequency="daily",
                unit="亿元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="杠杆资金入市活跃度与券商信用业务利息收入规模",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="高盛集团股价",
                source="yfinance",
                symbol="GS",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球顶级投资银行与机构做市交易龙头估值中枢对标",
            ),
            IndustryLinkageIndicator(
                name="嘉信理财股价",
                source="yfinance",
                symbol="SCHW",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球财富管理与互联网零售券商模式标杆企业对标",
            ),
        ],
        policy_catalysts=[
            "建设一流投资银行与头部券商并购重组",
            "全面注册制改革深化与逆周期调节",
            "公募基金与证券行业费率改革",
            "证券基金保险互换便利SFISF支持",
        ],
        description="证券公司与资本市场产业链指标映射（资金端拆借成本、交投端全市场成交量与两融余额、国际对标高盛GS与嘉信SCHW）",
    ),

    # 14. 保险与多元金融 (金融地产)
    "保险与多元金融": IndustryLinkage(
        industry_name="保险与多元金融",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="中国10年期国债到期收益率",
                source="pending_api",
                symbol="CN10Y",
                frequency="daily",
                unit="%",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="险资长久期资产配置收益率中枢与防范利差损核心锚",
            ),
            IndustryLinkageIndicator(
                name="人身险产品法定预定利率上限",
                source="manual",
                symbol=None,
                frequency="quarterly",
                unit="%",
                role="upstream",
                status="manual",
                note="手动",
                transmission_logic="寿险与年金负债端保单刚性负债成本调控基准",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="寿险行业新单原保险保费收入",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="亿元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="居民端养老储蓄与长期保障型保单承保需求景气度",
            ),
            IndustryLinkageIndicator(
                name="财险车险单月保费增速",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="%",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="汽车保有量增长与财险承保保费规模扩张验证",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="伯克希尔哈撒韦股价",
                source="yfinance",
                symbol="BRK-B",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球财险浮存金投资与长期复利价值投资标杆对标",
            ),
            IndustryLinkageIndicator(
                name="保德信金融股价",
                source="yfinance",
                symbol="PRU",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球人寿保险与资产管理巨头估值及利率敏感度对标",
            ),
        ],
        policy_catalysts=[
            "下调人身险预定利率防范利差损",
            "报行合一严控寿险与车险渠道费用",
            "养老金融第三支柱个人养老金政策",
            "长周期考核引导险资中长期入市",
        ],
        description="保险与多元金融产业链指标映射（资产端长端国债利率、负债端保费收入与预定利率、国际对标伯克希尔BRK-B与保德信PRU）",
    ),

    # 15. 钢铁与黑色金属 (周期大宗)
    "钢铁与黑色金属": IndustryLinkage(
        industry_name="钢铁与黑色金属",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="普氏铁矿石价格指数",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="美元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="高炉长流程炼钢最核心进口铁矿石原料成本传导",
            ),
            IndustryLinkageIndicator(
                name="主焦煤港口平仓价",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="焦化炼焦与高炉还原剂核心燃料成本传导",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="全国高炉开工率",
                source="manual",
                symbol=None,
                frequency="weekly",
                unit="%",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="钢厂实际生产负荷与粗钢日均产量供给验证",
            ),
            IndustryLinkageIndicator(
                name="五大品种钢材社会与钢厂总库存",
                source="pending_api",
                symbol=None,
                frequency="weekly",
                unit="万吨",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="建筑与制造业钢材供需缺口与季节性去库景气度",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="安赛乐米塔尔股价",
                source="yfinance",
                symbol="MT",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球最大跨国钢铁联合企业估值与全球用钢需求对标",
            ),
            IndustryLinkageIndicator(
                name="纽柯钢铁股价",
                source="yfinance",
                symbol="NUE",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球短流程电炉炼钢龙头企业盈利能力与吨钢利润对标",
            ),
        ],
        policy_catalysts=[
            "粗钢产量平控限产政策",
            "超低排放改造与全国碳市场扩容",
            "钢铁行业兼并重组提高产业集中度",
            "特钢关键材料攻坚与国产替代",
        ],
        description="钢铁与黑色金属产业链指标映射（上游铁矿石与焦炭成本、下游高炉开工与钢材库存、国际对标安赛乐米塔尔MT与纽柯NUE）",
    ),

    # 16. 有色金属与工业金属 (周期大宗)
    "有色金属与工业金属": IndustryLinkage(
        industry_name="有色金属与工业金属",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="铜精矿现货加工费TC",
                source="pending_api",
                symbol=None,
                frequency="weekly",
                unit="美元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="铜冶炼厂加工费收益与全球铜矿供给紧张程度核心指标",
            ),
            IndustryLinkageIndicator(
                name="氧化铝现货价格",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="电解铝最核心直接原材料成本传导",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="国家电网月度投资完成额",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="亿元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="电力特高压与电网智能化建设用铜第一大户需求景气度",
            ),
            IndustryLinkageIndicator(
                name="新能源汽车铝合金单车用量",
                source="pending_api",
                symbol=None,
                frequency="quarterly",
                unit="公斤/辆",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="汽车轻量化一体化压铸铝材消费增量验证",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="自由港麦克莫兰股价",
                source="yfinance",
                symbol="FCX",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球最大铜矿上市公司估值与全球铜博士景气度对标",
            ),
            IndustryLinkageIndicator(
                name="美国铝业股价",
                source="yfinance",
                symbol="AA",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球铝土矿氧化铝电解铝一体化跨国龙头对标",
            ),
        ],
        policy_catalysts=[
            "电解铝4500万吨合规产能红线硬约束",
            "特高压与配电网大规模建设规划",
            "大规模设备更新改造用铜需求",
            "战略金属收储与出口管制政策",
        ],
        description="有色金属与工业金属产业链指标映射（上游铜精矿加工费与氧化铝成本、下游电网投资与轻量化铝需求、国际对标自由港FCX与美铝AA）",
    ),

    # 17. 贵金属与稀缺资源 (周期大宗)
    "贵金属与稀缺资源": IndustryLinkage(
        industry_name="贵金属与稀缺资源",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="克金综合维持成本AISC",
                source="manual",
                symbol=None,
                frequency="quarterly",
                unit="美元/盎司",
                role="upstream",
                status="manual",
                note="手动",
                transmission_logic="全球主要黄金矿业公司开采完全生产边际现金成本",
            ),
            IndustryLinkageIndicator(
                name="稀土开采总量控制指标",
                source="manual",
                symbol=None,
                frequency="semi-annual",
                unit="万吨",
                role="upstream",
                status="manual",
                note="手动",
                transmission_logic="工信部与自然资源部稀土开采冶炼分离配额供给刚性约束",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="世界黄金协会全球央行季度净购金量",
                source="manual",
                symbol=None,
                frequency="quarterly",
                unit="吨",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="去美元化背景下全球官方外汇储备增持黄金刚性需求",
            ),
            IndustryLinkageIndicator(
                name="SPDR黄金ETF持仓量",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="吨",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="全球最大黄金ETF实物持仓与机构避险投机资金流向",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="纽蒙特黄金股价",
                source="yfinance",
                symbol="NEM",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球最大黄金开采矿业巨头估值与黄金现货价格对标",
            ),
            IndustryLinkageIndicator(
                name="巴里克黄金股价",
                source="yfinance",
                symbol="GOLD",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球第二大黄金矿企克金成本与资源储量对标",
            ),
        ],
        policy_catalysts=[
            "国家战略性矿产资源开采总量控制指标",
            "关键稀缺战略金属出口管制政策",
            "全球央行去美元化储备多元化趋势",
            "稀土行业整合重组提高话语权",
        ],
        description="贵金属与稀缺资源产业链指标映射（上游生产成本与开采配额、下游央行购金与ETF持仓、国际对标纽蒙特NEM与巴里克GOLD）",
    ),

    # 18. 石油石化与基础化工 (周期大宗)
    "石油石化与基础化工": IndustryLinkage(
        industry_name="石油石化与基础化工",
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
            IndustryLinkageIndicator(
                name="动力煤坑口价",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="煤化工乙二醇/甲醇/烯烃路线源头煤炭原料成本",
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
            IndustryLinkageIndicator(
                name="雪佛龙股价",
                source="yfinance",
                symbol="CVX",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球跨国能源巨头油气勘探开采与炼油销售盈利对标",
            ),
        ],
        policy_catalysts=[
            "能耗双控向碳排放双控转变",
            "成品油出口配额优化发放",
            "石化产业绿色低碳布局方案",
            "高端化工新材料自主创新支持",
        ],
        description="石油石化与基础化工产业链指标映射（上游布伦特原油成本、下游成品油消费与聚酯开工率、国际对标埃克森美孚XOM与雪佛龙CVX）",
    ),

    # 19. 煤炭与传统化石能源 (周期大宗)
    "煤炭与传统化石能源": IndustryLinkage(
        industry_name="煤炭与传统化石能源",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="煤矿智能化采掘设备投入",
                source="pending_api",
                symbol=None,
                frequency="annual",
                unit="亿元",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="现代化大型矿井安全生产与综采设备更新资本开支",
            ),
            IndustryLinkageIndicator(
                name="大秦线煤炭铁路运费",
                source="manual",
                symbol=None,
                frequency="quarterly",
                unit="元/吨",
                role="upstream",
                status="manual",
                note="手动",
                transmission_logic="晋陕蒙西煤东运铁路干线运价与港口中转运费成本",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="沿海八省电厂日耗煤量",
                source="manual",
                symbol=None,
                frequency="daily",
                unit="万吨",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="火力发电旺季与淡季电厂煤炭消耗刚性需求核心指标",
            ),
            IndustryLinkageIndicator(
                name="秦皇岛港动力煤平仓价",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="元/吨",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="国内5500大卡动力煤现货市场价格中枢风向标",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="皮博迪能源股价",
                source="yfinance",
                symbol="BTU",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="美国最大私营煤炭生产商估值与全球海运煤炭供需对标",
            ),
            IndustryLinkageIndicator(
                name="嘉能可股价",
                source="yfinance",
                symbol="GLEN.L",
                frequency="daily",
                unit="便士",
                role="benchmark",
                status="active",
                transmission_logic="全球大宗商品贸易与动力煤海运出口巨头盈利对标",
            ),
        ],
        policy_catalysts=[
            "煤炭保供稳价与长协合同履约率考核",
            "矿山安全生产大检查与产能核定",
            "超长期特别国债支持煤炭清洁高效利用",
            "煤电容量电价支撑长协煤消纳",
        ],
        description="煤炭与传统化石能源产业链指标映射（上游采掘与铁路运费成本、下游电厂日耗与港口煤价、国际对标皮博迪BTU与嘉能可GLEN）",
    ),

    # 20. 电力与公用事业 (公用基建)
    "电力与公用事业": IndustryLinkage(
        industry_name="电力与公用事业",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="电煤长协入厂标煤单价",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="火电企业营业成本70%以上之动力煤燃料成本传导",
            ),
            IndustryLinkageIndicator(
                name="天然铀现货价格",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="美元/磅",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="核电发电机组核燃料组件制造源头原材料成本",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="全社会月度用电量同比增速",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="%",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="国家能源局月度全社会及三大产业用电需求景气度",
            ),
            IndustryLinkageIndicator(
                name="长江三峡入库流量月度均值",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="立方米/秒",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="水电流域丰水枯水期来水情况与大型大坝水电发电量",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="新纪元能源股价",
                source="yfinance",
                symbol="NEE",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球最大公用事业与新能源绿电运营龙头估值对标",
            ),
            IndustryLinkageIndicator(
                name="南方电力公司股价",
                source="yfinance",
                symbol="SO",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="美国大型传统电力与核电公用事业龙头高股息对标",
            ),
        ],
        policy_catalysts=[
            "煤电容量电价机制全面落地",
            "电力现货市场与辅助服务市场建设",
            "可再生能源消纳权重与绿电交易",
            "核电重大项目常态化核准建设",
        ],
        description="电力与公用事业产业链指标映射（上游电煤与天然铀成本、下游用电量与水电流域流量、国际对标新纪元能源NEE与南方电力SO）",
    ),

    # 21. 房地产开发与运营 (金融地产)
    "房地产开发与运营": IndustryLinkage(
        industry_name="房地产开发与运营",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="全国重点城市土地成交溢价率",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="%",
                role="upstream",
                status="manual",
                note="手动",
                transmission_logic="房企土地储备购置土地出让金成本与地市热度",
            ),
            IndustryLinkageIndicator(
                name="房企境内外债券平均发行票息",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="%",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="房地产企业境内公募债与离岸美元债债务融资成本",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="30大中城市商品房单周成交面积",
                source="pending_api",
                symbol=None,
                frequency="weekly",
                unit="万平方米",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="终端商品房销售高频周度景气度与居民部门购房意愿",
            ),
            IndustryLinkageIndicator(
                name="百强房企单月全口径销售金额",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="亿元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="克而瑞百强房企单月销售回款与市场份额集中度",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="西蒙地产股价",
                source="yfinance",
                symbol="SPG",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球最大商业零售地产REITs持有运营与租金收益对标",
            ),
            IndustryLinkageIndicator(
                name="霍顿房屋股价",
                source="yfinance",
                symbol="DHI",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="美国最大住宅建筑开发商销售周转与毛利对标",
            ),
        ],
        policy_catalysts=[
            "取消限购限售限价与调降首付比例",
            "房贷利率下限取消与公积金利率调降",
            "保交房白名单贷款支持与城中村改造",
            "收购存量商品房用作保障性住房收储",
        ],
        description="房地产开发与运营产业链指标映射（上游土地与融资成本、下游30城商品房销售与百强房企回款、国际对标西蒙地产SPG与霍顿房屋DHI）",
    ),

    # 22. 建筑装饰与基础设施工程 (公用基建)
    "建筑装饰与基础设施工程": IndustryLinkage(
        industry_name="建筑装饰与基础设施工程",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="全国水泥均价PO42.5散装",
                source="pending_api",
                symbol=None,
                frequency="weekly",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="基础设施与建筑工程现场施工核心基础胶凝材料成本",
            ),
            IndustryLinkageIndicator(
                name="重交沥青全国出厂均价",
                source="pending_api",
                symbol=None,
                frequency="weekly",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="公路交通与市政道路路面摊铺施工核心建材成本",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="地方政府新增专项债每月发行规模",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="亿元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="基建项目资金到位与重大工程实物工作量资金保障核心先导指标",
            ),
            IndustryLinkageIndicator(
                name="八大建筑央企新签合同额季度增速",
                source="manual",
                symbol=None,
                frequency="quarterly",
                unit="%",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="铁路/公路/水利/房建等重大总包工程在手订单饱满度",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="万喜集团股价",
                source="yfinance",
                symbol="DG.PA",
                frequency="daily",
                unit="欧元",
                role="benchmark",
                status="active",
                transmission_logic="全球最大综合工程承包与特许经营基建巨头估值对标",
            ),
            IndustryLinkageIndicator(
                name="卡特彼勒股价",
                source="yfinance",
                symbol="CAT",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球工程机械与施工设备领军龙头景气度对标",
            ),
        ],
        policy_catalysts=[
            "超长期特别国债两重建设支持",
            "地方政府化债与清理拖欠企业账款",
            "城市地下管网改造与城市更新",
            "高质量共建一带一路重大基建项目落地",
        ],
        description="建筑装饰与基础设施工程产业链指标映射（上游水泥与沥青成本、下游专项债发行与央企新签合同、国际对标万喜DG.PA与卡特彼勒CAT）",
    ),

    # 23. 机械设备与工业母机 (先进制造)
    "机械设备与工业母机": IndustryLinkage(
        industry_name="机械设备与工业母机",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="数控系统与高精度伺服电机价格指数",
                source="pending_api",
                symbol=None,
                frequency="quarterly",
                unit="点",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="高档数控机床与自动化设备核心控制大脑与驱动部件成本",
            ),
            IndustryLinkageIndicator(
                name="机械铸件与特种生铁价格",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="大型机床床身、注塑机机架与工程机械铸件原材料成本",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="日本机床工业会JMTBA对华订单额",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="亿日元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="全球制造业资本开支与高端机床采购景气度先行指标",
            ),
            IndustryLinkageIndicator(
                name="国内工业机器人月度产量",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="台",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="汽车与3C等制造业自动化产线设备更新改造活跃度",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="发那科股价",
                source="yfinance",
                symbol="6954.T",
                frequency="daily",
                unit="日元",
                role="benchmark",
                status="active",
                transmission_logic="全球数控系统与工业机器人龙头估值中枢与订单对标",
            ),
            IndustryLinkageIndicator(
                name="德马吉森精机股价",
                source="yfinance",
                symbol="6141.T",
                frequency="daily",
                unit="日元",
                role="benchmark",
                status="active",
                transmission_logic="全球高端五轴加工中心与车铣复合工业母机标杆对标",
            ),
        ],
        policy_catalysts=[
            "推动大规模设备更新和消费品以旧换新行动方案",
            "工业母机高质量发展税收优惠与研发补贴",
            "智能制造示范工厂与制造业单项冠军支持",
            "工业母机首台套重大技术装备保险补偿机制",
        ],
        description="机械设备与工业母机产业链指标映射（上游数控系统与铸件成本、下游机床订单与工业机器人产量、国际对标发那科6954.T与德马吉森精机6141.T）",
    ),

    # 24. 国防军工与航天装备 (先进制造)
    "国防军工与航天装备": IndustryLinkage(
        industry_name="国防军工与航天装备",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="军工级高温合金与钛合金价格",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="万元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="航空发动机、导弹机体与舰船特种耐高温结构材料成本",
            ),
            IndustryLinkageIndicator(
                name="宇航级高可靠电子元器件价格指数",
                source="pending_api",
                symbol=None,
                frequency="quarterly",
                unit="点",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="军工雷达、卫星通信与武器制导高可靠芯片元器件采购成本",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="全国国防支出预算年度增长率",
                source="manual",
                symbol=None,
                frequency="annual",
                unit="%",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="国家财政预算国防装备建设与练兵备战采购刚性支出",
            ),
            IndustryLinkageIndicator(
                name="军工主机厂合同负债与大额预付款",
                source="manual",
                symbol=None,
                frequency="quarterly",
                unit="亿元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="军方五年规划重点型号装备下发订单与批产交付景气度",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="洛克希德马丁股价",
                source="yfinance",
                symbol="LMT",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球最大国防军工与先进战机承包商估值与军备采购对标",
            ),
            IndustryLinkageIndicator(
                name="雷神技术股价",
                source="yfinance",
                symbol="RTX",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球航空发动机与防空导弹雷达系统巨头对标",
            ),
        ],
        policy_catalysts=[
            "五年规划武器装备采购中期调整与终期冲刺订单落地",
            "建设世界一流军队与全军练兵备战战略需求",
            "商业航天与卫星互联网纳入战略性新兴产业",
            "军工央企中长期股权激励与资产证券化",
        ],
        description="国防军工与航天装备产业链指标映射（上游高温合金与军工电子成本、下游国防预算与主机厂合同负债、国际对标洛克希德马丁LMT与雷神RTX）",
    ),

    # 25. 交通运输与航运港口 (公用基建)
    "交通运输与航运港口": IndustryLinkage(
        industry_name="交通运输与航运港口",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="新加坡保税低硫船用燃料油现货价",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="美元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="国际远洋集运与油运集装箱班轮核心航行燃料动力成本",
            ),
            IndustryLinkageIndicator(
                name="国内航空煤油出厂价",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="民航客运与航空货运航司最核心燃油营运成本传导",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="上海出口集装箱运价指数SCFI",
                source="manual",
                symbol=None,
                frequency="weekly",
                unit="点",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="中国出口欧洲美西等主力航线集装箱海运即期运价核心晴雨表",
            ),
            IndustryLinkageIndicator(
                name="全国主要沿海港口集装箱吞吐量",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="万TEU",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="外贸进出口与跨境电商实物商品海运装卸总吞吐需求",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="马士基股价",
                source="yfinance",
                symbol="MAERSK-B.CO",
                frequency="daily",
                unit="丹麦克朗",
                role="benchmark",
                status="active",
                transmission_logic="全球综合航运物流领军集团估值中枢与全球集运周期对标",
            ),
            IndustryLinkageIndicator(
                name="联合包裹UPS股价",
                source="yfinance",
                symbol="UPS",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球快递供应链与陆空物流网络巨头景气度对标",
            ),
        ],
        policy_catalysts=[
            "国际海事组织IMO船舶碳排放环保新规",
            "快递行业反内卷与保障基层权益政策",
            "国家综合立体交通网与物流枢纽建设",
            "丝路海运与中欧班列高质量发展",
        ],
        description="交通运输与航运港口产业链指标映射（上游低硫船燃与航油成本、下游SCFI运价指数与港口吞吐量、国际对标马士基MAERSK与UPS）",
    ),

    # 26. 通信网络与光通信 (科技/TMT)
    "通信网络与光通信": IndustryLinkage(
        industry_name="通信网络与光通信",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="高端高速光芯片价格指数",
                source="pending_api",
                symbol=None,
                frequency="quarterly",
                unit="点",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="800G/1.6T高速光模块内部EML/VCSEL光芯片与DSP电芯片采购成本",
            ),
            IndustryLinkageIndicator(
                name="光纤预制棒与光缆集采单价",
                source="pending_api",
                symbol=None,
                frequency="monthly",
                unit="元/芯公里",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="千兆宽带网络与骨干网铺设基础光纤线缆成本",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="三大运营商年度5G与算力资本开支",
                source="manual",
                symbol=None,
                frequency="annual",
                unit="亿元",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="中国移动/电信/联通通信网络基站与智算中心集采预算",
            ),
            IndustryLinkageIndicator(
                name="全球800G及以上高速光模块季度出货量",
                source="pending_api",
                symbol=None,
                frequency="quarterly",
                unit="万只",
                role="downstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="北美云厂商与国内AI大模型数据中心算力互联网络出货景气度",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="Coherent高意股价",
                source="yfinance",
                symbol="COHR",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球光芯片与光通信收发模块材料领军企业对标",
            ),
            IndustryLinkageIndicator(
                name="思科系统股价",
                source="yfinance",
                symbol="CSCO",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球企业级网络交换机、路由器与电信设备龙头对标",
            ),
        ],
        policy_catalysts=[
            "双千兆协同发展行动计划与万兆光网试点",
            "算力网络国家枢纽节点直连网络建设",
            "电信运营商分红率提升与市值管理考核",
            "6G技术前沿研发与国际标准推进",
        ],
        description="通信网络与光通信产业链指标映射（上游光芯片与光纤成本、下游运营商资本开支与800G光模块出货、国际对标Coherent高意与思科CSCO）",
    ),

    # 27. 农林牧渔与生猪养殖 (大消费)
    "农林牧渔与生猪养殖": IndustryLinkage(
        industry_name="农林牧渔与生猪养殖",
        upstream_cost=[
            IndustryLinkageIndicator(
                name="国内玉米现货均价",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="生猪与家禽配合饲料核心能量原料采购成本传导",
            ),
            IndustryLinkageIndicator(
                name="全国豆粕现货均价",
                source="pending_api",
                symbol=None,
                frequency="daily",
                unit="元/吨",
                role="upstream",
                status="pending_api",
                note="待接入API",
                transmission_logic="养殖饲料核心蛋白原料成本传导（受大豆进口汇率与期货主导）",
            ),
        ],
        downstream_demand=[
            IndustryLinkageIndicator(
                name="全国生猪出栏均价",
                source="manual",
                symbol=None,
                frequency="daily",
                unit="元/公斤",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="全国生猪现货出栏交易价格与猪周期景气度核心晴雨表",
            ),
            IndustryLinkageIndicator(
                name="全国能繁母猪存栏量",
                source="manual",
                symbol=None,
                frequency="monthly",
                unit="万头",
                role="downstream",
                status="manual",
                note="手动",
                transmission_logic="农业农村部生猪基础产能与未来10个月肉猪供给先行指标",
            ),
        ],
        international_benchmark=[
            IndustryLinkageIndicator(
                name="泰森食品股价",
                source="yfinance",
                symbol="TSN",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球最大禽肉与猪肉蛋白质食品加工跨国企业估值对标",
            ),
            IndustryLinkageIndicator(
                name="阿彻丹尼尔斯米德兰股价",
                source="yfinance",
                symbol="ADM",
                frequency="daily",
                unit="美元",
                role="benchmark",
                status="active",
                transmission_logic="全球四大粮商之一大豆玉米农产品贸易与压榨加工对标",
            ),
        ],
        policy_catalysts=[
            "生猪产能调控实施方案能繁母猪正常保有量调控",
            "国家储备肉中央冻猪肉收储与投放调节机制",
            "转基因玉米大豆产业化应用与种业振兴行动",
            "现代设施农业建设规划与动保防疫支持",
        ],
        description="农林牧渔与生猪养殖产业链指标映射（上游玉米与豆粕成本、下游生猪均价与能繁母猪存栏、国际对标泰森食品TSN与ADM）",
    ),
}


class _IndustryLinkageMap(dict):
    """行业产业链指标配置字典，支持 27 个权威行业标准名称及历史/别名索引。"""

    def __getitem__(self, key: str) -> IndustryLinkage:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        cfg = get_industry_linkage_config(key)
        if cfg is not None:
            return cfg
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if dict.__contains__(self, key):
            return True
        if isinstance(key, str):
            return get_industry_linkage_config(key) is not None
        return False

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if dict.__contains__(self, key):
            return dict.get(self, key)
        cfg = get_industry_linkage_config(key)
        if cfg is not None:
            return cfg
        return default


INDUSTRY_LINKAGE_MAP: Dict[str, IndustryLinkage] = _IndustryLinkageMap(_BASE_INDUSTRY_LINKAGE_MAP)


# 历史 5 核心行业与权威行业名称的稳定别名映射表
_LEGACY_INDUSTRY_ALIAS_MAP: Dict[str, str] = {
    "消费电子": "消费电子与智能终端",
    "消费电子/半导体显示": "消费电子与智能终端",
    "新能源车": "新能源汽车与智能汽车",
    "新能源汽车": "新能源汽车与智能汽车",
    "新能源车/动力电池": "新能源汽车与智能汽车",
    "半导体": "半导体与集成电路",
    "半导体/集成电路": "半导体与集成电路",
    "石油化工": "石油石化与基础化工",
    "石油化工/基础化工": "石油石化与基础化工",
    "金融地产": "商业银行与信贷",
    "金融地产/商业银行与房地产": "商业银行与信贷",
    "银行": "商业银行与信贷",
    "商业银行": "商业银行与信贷",
    "房地产": "房地产开发与运营",
    "地产": "房地产开发与运营",
    "证券": "证券公司与资本市场",
    "券商": "证券公司与资本市场",
    "保险": "保险与多元金融",
    "光伏": "光伏与储能系统",
    "储能": "光伏与储能系统",
    "光伏储能": "光伏与储能系统",
    "锂电池": "动力电池与储能电池材料",
    "动力电池": "动力电池与储能电池材料",
    "医药": "医药生物与创新药",
    "生物医药": "医药生物与创新药",
    "创新药": "医药生物与创新药",
    "医疗器械": "医疗器械与医疗服务",
    "白酒": "白酒与精制茶酒",
    "食品饮料": "大众食品与饮料",
    "大众食品": "大众食品与饮料",
    "家电": "家用电器与智能家居",
    "白色家电": "家用电器与智能家居",
    "钢铁": "钢铁与黑色金属",
    "黑色金属": "钢铁与黑色金属",
    "有色金属": "有色金属与工业金属",
    "工业金属": "有色金属与工业金属",
    "贵金属": "贵金属与稀缺资源",
    "煤炭": "煤炭与传统化石能源",
    "电力": "电力与公用事业",
    "公用事业": "电力与公用事业",
    "建筑": "建筑装饰与基础设施工程",
    "基建": "建筑装饰与基础设施工程",
    "建材": "建筑装饰与基础设施工程",
    "建筑装饰": "建筑装饰与基础设施工程",
    "机械": "机械设备与工业母机",
    "机械设备": "机械设备与工业母机",
    "工业母机": "机械设备与工业母机",
    "军工": "国防军工与航天装备",
    "国防军工": "国防军工与航天装备",
    "交运": "交通运输与航运港口",
    "交通运输": "交通运输与航运港口",
    "航运": "交通运输与航运港口",
    "通信": "通信网络与光通信",
    "光通信": "通信网络与光通信",
    "通信网络": "通信网络与光通信",
    "农业": "农林牧渔与生猪养殖",
    "农林牧渔": "农林牧渔与生猪养殖",
    "生猪养殖": "农林牧渔与生猪养殖",
    "AI": "人工智能与算力服务",
    "人工智能": "人工智能与算力服务",
    "算力": "人工智能与算力服务",
}


def get_industry_linkage_config(industry: str) -> Optional[IndustryLinkage]:
    """获取指定行业的产业链指标配置对象。

    Args:
        industry: 行业标准全称、别名或行业关键词 (如 "消费电子", "新能源车", "半导体", "石油化工", "金融地产", "白酒", "银行" 等)

    Returns:
        匹配到的 IndustryLinkage 配置对象，未匹配则返回 None
    """
    if not industry or not isinstance(industry, str):
        return None

    clean_industry = industry.strip()
    if not clean_industry:
        return None

    # 1. 尝试直接在底层基础字典中精确命中
    if clean_industry in _BASE_INDUSTRY_LINKAGE_MAP:
        return _BASE_INDUSTRY_LINKAGE_MAP[clean_industry]

    # 2. 尝试通过快捷别名表映射
    if clean_industry in _LEGACY_INDUSTRY_ALIAS_MAP:
        target_name = _LEGACY_INDUSTRY_ALIAS_MAP[clean_industry]
        if target_name in _BASE_INDUSTRY_LINKAGE_MAP:
            return _BASE_INDUSTRY_LINKAGE_MAP[target_name]

    # 3. 借助静态知识库的权威画像匹配 (支持 industry_id, industry_name, aliases, 子串)
    profile = get_industry_profile(clean_industry)
    if profile and profile.industry_name in _BASE_INDUSTRY_LINKAGE_MAP:
        return _BASE_INDUSTRY_LINKAGE_MAP[profile.industry_name]

    # 4. 遍历所有配置项进行子串包含匹配
    for name, config in _BASE_INDUSTRY_LINKAGE_MAP.items():
        if clean_industry in name or name in clean_industry:
            return config

    return None


def list_supported_industries() -> List[str]:
    """返回当前已支持配置的 27 个权威行业标准全称列表。"""
    return list(_BASE_INDUSTRY_LINKAGE_MAP.keys())


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


DEFAULT_INDUSTRY_LINKAGE_MISSING_PROMPT = (
    "【产业链联想数据】：【数据缺失】（未映射行业或采集失败，不得据此推断景气中性）"
)


def format_industry_linkage_for_prompt(
    industry_linkage: Optional[Union[Dict[str, Any], IndustryLinkage]]
) -> str:
    """将产业链联动数据格式化为可直接注入 Prompt 的结构化文本。

    永远返回非空文本。若无数据或未映射行业，返回 fail-closed 的【数据缺失】提示段落。

    Args:
        industry_linkage: 产业链数据字典或 IndustryLinkage 模型实例

    Returns:
        结构化 Markdown 文本段落；若无数据或未映射则返回 fail-closed 缺省提示段落
    """
    if not industry_linkage:
        return DEFAULT_INDUSTRY_LINKAGE_MISSING_PROMPT

    if hasattr(industry_linkage, "model_dump"):
        data = industry_linkage.model_dump()
    elif isinstance(industry_linkage, dict):
        data = industry_linkage
    else:
        return DEFAULT_INDUSTRY_LINKAGE_MISSING_PROMPT

    industry_name = data.get("industry_name")
    if not industry_name:
        return DEFAULT_INDUSTRY_LINKAGE_MISSING_PROMPT

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
