"""行业上下游产业链、宏观驱动、周期与风险图谱知识库 (Industry Linkage & Risk Graph).

提供 20+ 主流行业全景静态常识图谱，支持：
1. 产业链上下游 (Upstream / Downstream) 穿透；
2. 核心要素与成本项、产业链议价权与利润分配分析；
3. 宏观敏感度矩阵 (利率、汇率、大宗商品/通胀、流动性、产业政策、全球联动)；
4. 行业周期属性 (周期类型、典型周期长度、产能扩张滞后期、高频周期指标)；
5. 综合风险图谱 (地缘政治敏感度、卡脖子瓶颈、替代技术、监管政策、需求悬崖)；
6. 核心高频与财务跟踪指标、代表性细分赛道；
7. 供 LLM Prompt 注入的结构化紧凑上下文组装函数。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import re


@dataclass(frozen=True)
class MacroSensitivity:
    """宏观敏感度因子画像。"""
    interest_rate: str  # 高 / 中 / 低 / 逆向 (附简要逻辑)
    fx_rate: str  # 高 / 中 / 低 (附简要逻辑)
    commodity_inflation: str  # 高 / 中 / 低 (附成本挤压或转嫁逻辑)
    liquidity: str  # 高 / 中 / 低 (流动性与估值弹性)
    policy_drivers: List[str] = field(default_factory=list)  # 核心产业政策驱动
    global_macro_linkage: str = ""  # 全球宏观/海外市场联动机制


@dataclass(frozen=True)
class CycleProfile:
    """行业周期属性画像。"""
    cycle_type: str  # 强周期 / 成长周期 / 稳定弱周期 / 防御逆周期 / 政策订单周期
    typical_length: str  # 典型周期跨度 (如 3-4年硅周期, 3-5年猪周期)
    capacity_lag: str  # 产能建设与投放滞后期 (如 1.5-2年)
    key_cycle_indicators: List[str] = field(default_factory=list)  # 周期位置判断指标


@dataclass(frozen=True)
class RiskMatrix:
    """多维风险矩阵。"""
    geopolitical: List[str] = field(default_factory=list)  # 地缘政治与出口限制
    supply_chain_bottlenecks: List[str] = field(default_factory=list)  # 卡脖子关键原材料/设备
    technology_substitution: List[str] = field(default_factory=list)  # 技术路线颠覆与替代风险
    policy_regulatory: List[str] = field(default_factory=list)  # 政策与行业监管风险
    demand_cliff: List[str] = field(default_factory=list)  # 下游需求骤降/断崖风险


@dataclass(frozen=True)
class IndustryProfile:
    """行业全维画像与产业链图谱。"""
    industry_id: str
    industry_name: str
    category: str  # 科技/TMT, 先进制造, 绿色能源, 大消费, 周期大宗, 金融地产, 公用基建, 医药健康
    aliases: List[str]
    upstream: List[str]  # 上游主要环节/供应商行业
    downstream: List[str]  # 下游应用领域/主要采购方
    core_inputs: List[str]  # 核心原材料与生产要素
    pricing_power: str  # 产业链议价权特征与毛利转嫁机制
    macro_sensitivity: MacroSensitivity
    cycle_profile: CycleProfile
    risks: RiskMatrix
    key_metrics: List[str]  # 跟踪监控与财报核心指标
    representative_segments: List[str]  # 核心细分赛道/代表领域


# ─────────────────────────────────────────────────────────────────────────────
# 25+ 行业全景产业链图谱静态知识库
# ─────────────────────────────────────────────────────────────────────────────

INDUSTRY_PROFILES: Dict[str, IndustryProfile] = {
    "semiconductor": IndustryProfile(
        industry_id="semiconductor",
        industry_name="半导体与集成电路",
        category="科技/TMT",
        aliases=["半导体", "集成电路", "芯片", "晶圆", "封测", "semiconductor", "IC", "存储芯片", "功率半导体"],
        upstream=[
            "电子级多晶硅/大硅片",
            "光刻机/刻蚀机/薄膜沉积/离子注入等半导体设备",
            "光刻胶/电子特气/高纯化学品/靶材等半导体材料",
            "EDA设计软件/IP核授权",
        ],
        downstream=[
            "智能手机与消费电子终端",
            "AI算力数据中心/服务器",
            "新能源汽车与智能座舱/智能驾驶",
            "通信基站与网络通信设备",
            "工业控制与自动化",
            "军工航天高可靠芯片",
        ],
        core_inputs=["高纯电子化学品", "先进设备折旧", "高素质研发人员薪酬", "洁净室电力能耗"],
        pricing_power="设计端龙头依靠架构与IP壁垒具备强定价权；先进制程晶圆制造具有重资产高集中度极强议价权；成熟制程与封测环节竞争激烈，议价权中等偏弱。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="高（高估值科技成长属性，DCF对折现率极度敏感；资本开支对融资利率敏感）",
            fx_rate="高（原材料/设备依赖美元日元欧元进口，海外销售直接受美元汇率波动影响）",
            commodity_inflation="中（金线/铜靶材/化学品成本有一定影响，但研发与折旧占主导）",
            liquidity="高（流动性宽裕期估值弹性大，易产生板块性估值扩张）",
            policy_drivers=["国家大基金扶持", "集成电路重大专项", "国产替代自主可控采购倾斜", "研发费用加计扣除与税收优惠"],
            global_macro_linkage="高度联动全球费城半导体指数(SOX)、台积电月度营收/资本开支、全球半导体销售额(SIA)及海外出口管制法案。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="强周期成长型（硅周期）",
            typical_length="3~4年（经历主动去库、被动去库、主动补库、被动补库阶段）",
            capacity_lag="1.5~2.5年（新建晶圆厂从土建到设备调优量产周期长）",
            key_cycle_indicators=["全球半导体月度销售额同比", "DRAM/NAND存储芯片现货与合约价格", "晶圆代工厂产能利用率", "行业存货周转天数", "BB值(书单出货比)"],
        ),
        risks=RiskMatrix(
            geopolitical=["美国出口管制实体清单扩容", "先进制程设备(EUV/高阶ArFi)及EDA工具禁运", "跨国并购审查受阻"],
            supply_chain_bottlenecks=["高端光刻机", "先进制程光刻胶/高纯电子化学品", "高端EDA软件", "CPU/GPU底层IP架构"],
            technology_substitution=["先进封装(Chiplet/CoWoS)对传统封装挤压", "新材料(GaN/SiC)对硅基功率器件替代"],
            policy_regulatory=["海外技术管制升级", "补贴政策退坡或资金分配不均"],
            demand_cliff=["下游智能手机/PC换机周期拉长", "云厂商资本开支周期性削减"],
        ),
        key_metrics=["毛利率/净利率", "研发费用占营收比重", "存货周转天数与存货减值准备", "产能利用率", "在手订单与预收款"],
        representative_segments=["Fabless芯片设计", "Foundry晶圆代工", "OSAT芯片封测", "半导体前道/后道设备", "半导体关键材料(大硅片/光刻胶等)"],
    ),

    "ai_computing": IndustryProfile(
        industry_id="ai_computing",
        industry_name="人工智能与算力服务",
        category="科技/TMT",
        aliases=["AI", "人工智能", "算力", "数据中心", "云计算", "大模型", "软件服务", "SaaS", "服务器"],
        upstream=[
            "AI芯片/GPU/NPU/ASIC",
            "高带宽内存(HBM)/DDR5",
            "高速光模块(800G/1.6T)/光芯片",
            "服务器PCB/高速铜缆/散热液冷系统",
            "算力数据中心(IDC机房/供电系统)",
        ],
        downstream=[
            "大模型研发厂商",
            "互联网与短视频/电商推荐",
            "自动驾驶与具身智能",
            "金融科技与智能量化",
            "工业互联网与企业级软件",
            "政务与智慧城市",
        ],
        core_inputs=["高性能GPU/算力板卡", "数据中心绿电能耗", "光模块/网络交换机", "算法研发工程师薪酬"],
        pricing_power="顶级AI芯片与大模型核心技术方掌握定价权；算力租赁与集成环节受算力供需格局直接影响；通用软件服务同质化竞争议价偏弱。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="高（高远期现金流折现，全球降息周期大幅提振科技估值与风险偏好）",
            fx_rate="中高（高端算力卡以美元计价，汇率贬值推高国内算力采购成本）",
            commodity_inflation="低（主要成本在硬件投入与研发，铜/铝等金属对PCB成本有微弱传导）",
            liquidity="极高（科技主题炒作与机构重仓对市场流动性极为敏感）",
            policy_drivers=["全国一体化算力网络/东数西算工程", "人工智能+赋能实体经济政策", "自主可控信创工程推进"],
            global_macro_linkage="紧密联动美股纳斯达克100、英伟达/微软等全球科技巨头资本开支(Capex)及财报指引。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="技术驱动爆发性成长周期",
            typical_length="5~8年（技术范式革命驱动的主升浪，穿插1~2年硬件过剩调整）",
            capacity_lag="6~18个月（算力集群搭建与数据中心绿电审批周期）",
            key_cycle_indicators=["北美云巨头资本开支增速", "高端光模块出货量与迭代速度", "IDC机房上架率", "算力租赁单位算力价格变动"],
        ),
        risks=RiskMatrix(
            geopolitical=["海外先进制程算力卡禁售与算力租赁云端受限", "开源开源生态分化"],
            supply_chain_bottlenecks=["HBM存储芯片产能受限", "CoWoS先进封装产能瓶颈", "高端光芯片自主化率低"],
            technology_substitution=["大模型轻量化/端侧模型突破削减云端算力总需求", "新型网络互联架构替代传统光模块"],
            policy_regulatory=["大模型算法备案与数据合规监管", "高能耗数据中心碳排放指标受限"],
            demand_cliff=["AI应用商业化变现不及预期导致大模型公司缩减算力投入"],
        ),
        key_metrics=["算力基础设施投入(Capex)增速", "光模块毛利率及产品结构", "SaaS企业ARR/续费率", "研发费用率", "算力利用率"],
        representative_segments=["AI加速芯片", "高速光模块与光器件", "AI服务器与液冷散热", "算力租赁与IDC运营", "行业垂类大模型与AI应用"],
    ),

    "nev_auto": IndustryProfile(
        industry_id="nev_auto",
        industry_name="新能源汽车与智能汽车",
        category="先进制造",
        aliases=["新能源汽车", "整车", "汽车零部件", "智能汽车", "自动驾驶", "电动车", "汽车", "NEV"],
        upstream=[
            "动力电池与电池材料(正极/负极/电解液/隔膜)",
            "汽车芯片(MCU/IGBT/SiC/智驾芯片)",
            "汽车钢板/铝合金压铸件/轮胎橡胶/车用塑料",
            "智能座舱与激光雷达/毫米波雷达/摄像头",
        ],
        downstream=[
            "C端个人乘用车消费者",
            "B端网约车/出租车出行市场",
            "商用物流车/重卡/客车运营方",
            "海外出口市场(欧洲/东南亚/中东/拉美)",
        ],
        core_inputs=["动力电池系统", "汽车电子与芯片", "钢材与铝压铸件", "营销与渠道网络建设"],
        pricing_power="头部强势车企具备终端定价与供应链压价权；具备核心自研技术(如智驾/三电)零部件厂具备一定议价权；通用机械加工件议价弱。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="中（汽车消费属于大宗可选消费，车贷利率下降与流动性宽裕提振购车需求）",
            fx_rate="中高（中国汽车出海提速，人民币贬值有利于出口结算毛利）",
            commodity_inflation="高（碳酸锂、铜、铝、钢材价格大幅波动直接传导至整车BOM成本）",
            liquidity="中高（乘用车消费与居民可支配收入及宏观财富效应强相关）",
            policy_drivers=["新能源车购置税减免/置换补贴(以旧换新)", "充换电基础设施建设规划", "智能网联汽车准入与L3试点"],
            global_macro_linkage="关注欧盟反补贴关税、美国关税政策及特斯拉全球价格调整策略。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="渗透率提升成长周期 + 宏观可选消费波动周期",
            typical_length="3~5年（受产品换代、补贴政策与宏观经济周期交织影响）",
            capacity_lag="1~2年（整车厂冲压焊装涂装总装四大工艺产线建设周期）",
            key_cycle_indicators=["乘联会月度狭义乘用车零售与批发销量", "新能源汽车单月渗透率", "经销商库存预警指数", "碳酸锂价格走势"],
        ),
        risks=RiskMatrix(
            geopolitical=["海外加征反补贴高额关税与本土化建厂要求", "智驾数据跨境流动受限"],
            supply_chain_bottlenecks=["车规级高端主控芯片与SiC功率器件供应紧张", "固态电池研发产业化不及预期"],
            technology_substitution=["换电与超快充路线竞争", "混动(PHEV/增程)对纯电(BEV)结构性分流"],
            policy_regulatory=["购车补贴全面退坡", "自动驾驶事故引发的法规收紧"],
            demand_cliff=["行业内部价格战白热化导致全产业链毛利率剧烈下滑", "居民大宗消费意愿阶段性疲软"],
        ),
        key_metrics=["月度交付量与市占率", "单车收入(ASP)与单车毛利率", "研发与销售费用率", "存货周转天数", "现金储备与自由现金流"],
        representative_segments=["新能源乘用车整车", "商用车与客车", "智能座舱与车载HUD", "底盘悬架与空气弹簧", "热管理系统与一体化压铸"],
    ),

    "photovoltaic_storage": IndustryProfile(
        industry_id="photovoltaic_storage",
        industry_name="光伏与储能系统",
        category="绿色能源",
        aliases=["光伏", "储能", "硅料", "硅片", "电池片", "组件", "逆变器", "光伏设备"],
        upstream=[
            "工业硅/多晶硅料",
            "高纯石英砂/坩埚",
            "银浆/EVA胶膜/POE胶膜/光伏玻璃/铝边框",
            "光伏切片机/丝网印刷机/PECVD设备",
        ],
        downstream=[
            "国内集中式大型风光电站(五大六小央国企)",
            "分布式工商业屋顶与户用光伏",
            "海外地面电站与分布式光储市场(欧美/中东/新兴市场)",
            "电网侧大储与独立储能电站",
        ],
        core_inputs=["多晶硅/工业硅", "高纯石英砂", "银浆", "光伏玻璃与胶膜", "电力成本"],
        pricing_power="供需过剩期全产业链价格踩踏，议价权微弱；技术迭代初期先进电池技术(如TOPCon/HJT/BC)享受短暂溢价；逆变器与储能系统集成具有品牌渠道壁垒。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="高（电站投资属于重资产基建，度电成本(LCOE)对融资利率极其敏感，海外降息刺激装机）",
            fx_rate="高（组件与逆变器60%以上依赖海外出口，人民币贬值提升出口收益）",
            commodity_inflation="高（工业硅、白银、纯碱/玻璃、铝等大宗商品直接影响BOM成本）",
            liquidity="中（依赖绿电投融资环境与地方国企资本开支意愿）",
            policy_drivers=["全球碳中和目标与可再生能源配额制", "国内电网消纳与配储政策", "绿电绿证交易市场化改革"],
            global_macro_linkage="紧密跟踪美国UFLPA关税政策、东南亚光伏双反调查、欧洲电价与天然气库存水平。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="典型重资产制造业产能过剩强周期 + 长期成长",
            typical_length="2~3年（产能扩张迅速，从供不应求到严重过剩仅需1-2年）",
            capacity_lag="6~12个月（电池与组件产线扩产快，硅料扩产约18个月）",
            key_cycle_indicators=["硅料/硅片/电池片/组件周度现货报价(InfoLink)", "国内月度新增光伏装机量", "组件与逆变器单月出口金额", "光伏产业链各环节开工率"],
        ),
        risks=RiskMatrix(
            geopolitical=["欧美贸易壁垒、关税加征与本地制造保护政策", "原产地穿透合规审查"],
            supply_chain_bottlenecks=["高纯内层石英砂供应波动", "贵金属白银价格飙涨推高浆料成本"],
            technology_substitution=["P型PERC被N型TOPCon淘汰", "BC电池与钙钛矿叠层电池对现有产线的潜在颠覆"],
            policy_regulatory=["强制配储利用率低引发政策调整", "电网消纳瓶颈导致弃光率上升与分时电价下调"],
            demand_cliff=["国内电网消纳承载力见顶导致集中式装机节奏放缓", "海外库存高企导致阶段性去库暂停采购"],
        ),
        key_metrics=["组件单瓦毛利与单瓦净利", "各环节产能利用率", "存货与固定资产减值计提", "在手订单排产情况", "海外销售占比"],
        representative_segments=["高纯多晶硅", "单晶硅棒/硅片", "高效光伏电池片", "光伏组件", "光伏/储能逆变器", "大储与工商储系统集成"],
    ),

    "lithium_battery": IndustryProfile(
        industry_id="lithium_battery",
        industry_name="动力电池与储能电池材料",
        category="绿色能源",
        aliases=["锂电池", "动力电池", "储能电池", "正极材料", "负极材料", "电解液", "隔膜", "碳酸锂"],
        upstream=[
            "锂矿/锂盐(碳酸锂/氢氧化锂)",
            "镍钴锰前驱体/磷酸铁/天然与人造石墨",
            "六氟磷酸锂/添加剂/溶剂",
            "基膜/氧化铝涂覆材料/铜箔/铝箔",
        ],
        downstream=[
            "新能源汽车乘用车/商用车车企",
            "储能系统集成商(户储/大储/通信基站)",
            "两轮电动车与电动船舶/eVTOL低空经济",
            "消费电子(手机/笔记本/无人机)",
        ],
        core_inputs=["碳酸锂/氢氧化锂", "石墨焦", "六氟磷酸锂", "电解铜箔/铝箔", "石墨化电费"],
        pricing_power="头部电池厂(具备规模、客户认证和专利壁垒)拥有较强定价与金属联动机制；中游四大主材产能过剩，加工费受挤压，议价权中等偏弱。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="中（关联下游新能源汽车与储能电站的融资成本）",
            fx_rate="中高（锂矿原材料大量从澳洲/南美进口，电池与材料海外出口占比逐年提升）",
            commodity_inflation="极高（碳酸锂、镍、钴、铜、铝价格直接决定营业成本与毛利）",
            liquidity="中高（资本开支扩张依赖融资，估值受成长板块情绪主导）",
            policy_drivers=["新能源汽车产业发展规划", "新型储能高质量发展行动方案", "欧盟新电池法案(碳足迹要求)"],
            global_macro_linkage="关注全球锂资源供给(南美盐湖/澳洲锂辉石/非洲锂矿)、海外电池本地化建厂法案(美国IRA/欧洲CRMA)。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="大宗商品原材料周期 + 高端制造业产能周期",
            typical_length="3~4年（受锂价起伏与中游扩产周期主导）",
            capacity_lag="1~1.5年（电池产线与正负极材料产线建设调试周期）",
            key_cycle_indicators=["电池级碳酸锂现货价格与期货主力合约", "动力电池月度装车量与产量", "正负极/电解液/隔膜加工费走势", "行业库存水位"],
        ),
        risks=RiskMatrix(
            geopolitical=["美国IRA法案对受关注外国实体(FEOC)限制", "欧盟电池碳足迹与回收追溯壁垒"],
            supply_chain_bottlenecks=["海外锂矿出口国有化与开采许可限制", "高端湿法隔膜涂覆设备依赖海外"],
            technology_substitution=["全固态电池对液态锂电颠覆", "钠离子电池在低速车和储能端替代", "磷酸锰铁锂/富锂锰基技术迭代"],
            policy_regulatory=["动力电池安全与热失控国家标准加码", "环保与能耗双控限制高耗能石墨化产能"],
            demand_cliff=["下游车企销量不及预期导致电池砍单", "海外贸易保护阻断出口通路"],
        ),
        key_metrics=["单Wh毛利与单Wh净利", "产能利用率", "存货减值准备与跌价计提", "海外收入占比", "研发费用与专利储备"],
        representative_segments=["锂离子动力电池", "储能专用电芯", "三元/磷酸铁锂正极", "人造/天然石墨负极", "电解液与六氟磷酸锂", "湿法/干法隔膜", "电池结构件"],
    ),

    "biopharma": IndustryProfile(
        industry_id="biopharma",
        industry_name="医药生物与创新药",
        category="医药健康",
        aliases=["医药", "生物医药", "创新药", "中药", "仿制药", "CXO", "疫苗", "biotech"],
        upstream=[
            "医药中间体/基础化工原料/天然药材",
            "生物反应器/色谱填料/培养基/耗材",
            "实验动物(实验猴/小鼠)",
            "临床前研究/CRO服务",
        ],
        downstream=[
            "公立医院与基层医疗卫生机构",
            "零售药店与医药电商平台",
            "跨国药企License-out授权交易",
            "疾控中心与公共卫生体系",
        ],
        core_inputs=["高端生物试剂与耗材", "临床试验受试者招募与医院伦理审查", "资深医学研发与临床研究人员薪酬"],
        pricing_power="全球独家专利创新药具有极高定价权；进入医保目录后以量换价；仿制药在集采(VBP)机制下议价权微弱；品牌中药独家品种具备自主提价权。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="高（Biotech估值完全依赖远期现金流折现，对全球降息周期反应极其敏锐；海外融资环境影响CXO订单）",
            fx_rate="中（CXO企业主要收入来自海外美元结算，创新药出海首付款以美元计价，汇率影响汇兑收益）",
            commodity_inflation="低（原材料成本占比相对较低，中药材价格受气候与供需波动影响）",
            liquidity="中高（流动性宽裕有利于创新药一级市场投融资与二级市场风险偏好提升）",
            policy_drivers=["国家医保药品目录动态调整与谈判", "仿制药一致性评价与国家集采", "国家支持全链条创新药发展政策"],
            global_macro_linkage="关注美联储利率对海外Biotech投融资景气度影响、美国生物安全法案立法进展、全球大药企(MNC)研发管线交易。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="研发与监管审批政策驱动周期 + 长期弱周期刚需",
            typical_length="8~12年（一款创新药从立项、临床I/II/III期到获批上市周期长）",
            capacity_lag="2~3年（GMP规范生物药制剂生产车间建设验证周期）",
            key_cycle_indicators=["全球与国内Biotech医疗健康领域投融资总额", "国家医保谈判降价幅度与新增纳入品种", "海外创新药License-out交易总额", "CXO在手订单与新签订单金额"],
        ),
        risks=RiskMatrix(
            geopolitical=["美国《生物安全法案》等对国内CXO企业海外业务的限制", "基因数据出境安全审查"],
            supply_chain_bottlenecks=["生物制药核心培养基与一次性生物反应袋/填料国产化率低", "高等级实验动物供应受限"],
            technology_substitution=["小核酸药物(siRNA)/ADC/细胞基因治疗(CGT)对传统抗体/小分子化药的迭代挤压"],
            policy_regulatory=["医保谈判降幅超预期", "药品带量采购常态化制度化扩面(国家集采)", "医疗反腐合规常态化对医院准入与学术推广的影响"],
            demand_cliff=["创新药临床III期数据不及预期导致研发失败终止", "专利到期遭遇仿制药专利悬崖(Patent Cliff)"],
        ),
        key_metrics=["研发管线阶段与在研产品数量", "研发费用率与研发资本化比例", "海外授权(License-out)里程碑款", "商业化产品进院数量与医保覆盖", "现金消耗率(Cash Burn Rate)"],
        representative_segments=["小分子创新药", "抗体药物与ADC", "细胞与基因治疗(CGT)", "医药研发外包(CXO/CDMO)", "中药独家品种", "疫苗与血制品"],
    ),

    "medical_devices": IndustryProfile(
        industry_id="medical_devices",
        industry_name="医疗器械与医疗服务",
        category="医药健康",
        aliases=["医疗器械", "高值耗材", "体外诊断", "IVD", "医疗服务", "医疗设备", "影像设备"],
        upstream=[
            "高精度传感器/精密光学镜片/探测器",
            "医用钛合金/高分子生物材料/硅胶",
            "精密电机/机械加工件/医用芯片",
            "体外诊断抗原抗体/酶/化学发光试剂原料",
        ],
        downstream=[
            "公立各级医院及专科医院(检验科/影像科/手术室)",
            "民营医院与体检中心",
            "第三方独立医学检验实验室(ICL)",
            "家用健康监测与养老康复消费终端",
        ],
        core_inputs=["精密核心零部件与光学/电子元器件", "医用高分子原材料", "研发工程师与专业售后工程师薪酬"],
        pricing_power="高壁垒大型影像设备(高端CT/超导MRI)与创新介入器械议价权强；常规体外诊断试剂与低值耗材受集采压制，议价权弱。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="中（与医院资本开支、医疗新基建专项债发行及设备更新贷款利率挂钩）",
            fx_rate="中（核心传感器等进口零部件受汇率影响，海外出口业务享受汇率贬值红利）",
            commodity_inflation="低（主要成本在研发、精密加工与营销服务）",
            liquidity="中（公立医疗机构采购资金来自财政拨款、医保基金及医疗专项债）",
            policy_drivers=["医疗领域大规模设备更新与财政贴息贷款", "高值医用耗材国家集采与联盟集采", "千县工程与医疗新基建"],
            global_macro_linkage="关注海外市场CE/FDA认证周期及海外反倾销/准入壁垒。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="医疗新基建采购周期 + 进口替代成长",
            typical_length="3~5年（受国家医疗卫生财政支出、医院设备折旧置换周期主导）",
            capacity_lag="1~2年（医疗器械注册证二类/三类审批周期1~3年）",
            key_cycle_indicators=["医疗专项债与超长期特别国债下达金额", "公立医院医疗设备招投标金额与台数", "高值耗材集采中标结果与报量执行率"],
        ),
        risks=RiskMatrix(
            geopolitical=["高端医疗设备核心零部件(如CT球管/超导磁体/探测器)出口限制", "海外市场准入合规门槛提高"],
            supply_chain_bottlenecks=["医用级高性能芯片", "高端光学镜片与特种涂层材料"],
            technology_substitution=["手术机器人辅助系统对传统手术器械升级", "POCT即时检验对传统大型生化免疫流水线补充分流"],
            policy_regulatory=["高值耗材与IVD集采大幅杀价", "医疗设备招投标全流程合规整顿导致采购推迟", "DRG/DIP医保支付方式改革倒逼医院控制耗材成本"],
            demand_cliff=["医疗设备更新政策落地节奏滞后导致医院观望延期招标"],
        ),
        key_metrics=["医疗器械三类注册证获批数量", "招投标中标金额与份额", "毛利率及集采后净利率韧性", "海外市场营收增速", "应收账款周转天数(医院回款账期)"],
        representative_segments=["医学影像设备(CT/MRI/超声)", "心血管/骨科高值耗材", "体外诊断(IVD)仪器与试剂", "手术机器人与内窥镜", "康复与家用医疗设备"],
    ),

    "consumer_electronics": IndustryProfile(
        industry_id="consumer_electronics",
        industry_name="消费电子与智能终端",
        category="科技/TMT",
        aliases=["消费电子", "手机产业链", "智能终端", "PC", "可穿戴设备", "VR/AR", "苹果产业链", "果链"],
        upstream=[
            "芯片与半导体元器件(SoC/存储/射频/电源管理)",
            "显示面板(OLED/LCD/MiniLED)",
            "精密结构件/玻璃盖板/金属中框/柔性电路板(FPC)",
            "光学镜头/音响电声器件/微型马达/锂聚合物电池",
        ],
        downstream=[
            "全球智能手机终端品牌商(Apple/华为/小米/三星/OPPO/vivo)",
            "PC与平板电脑品牌商(联想/戴尔/惠普/苹果)",
            "智能穿戴与智能家居(智能手表/TWS耳机/XR头显)",
            "汽车电子终端与IoT物联网设备",
        ],
        core_inputs=["集成电路元器件", "显示面板与玻璃基板", "精密五金模具与刀具", "产线组装工人工资"],
        pricing_power="终端品牌巨头掌握绝对采购定价权；具备独特精密制造工艺或关键部件独供地位的供应链企业具备一定议价权；通用组装代工毛利微薄。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="中（居民消费信贷与全球消费意愿相关）",
            fx_rate="高（外销代工占比较大，以美元结算为主，汇率波动对毛利率和汇兑损益影响显著）",
            commodity_inflation="中（铜、铝、贵金属及化工塑料价格传导至结构件与元器件）",
            liquidity="中高（消费电子行情高度依赖全球科技产品创新周期与流动性催化）",
            policy_drivers=["消费品以旧换新补贴政策(手机/平板/电脑)", "数字经济与AI终端硬件落地扶持"],
            global_macro_linkage="紧密联动苹果(Apple)秋季新品发布会、全球智能手机季度出货量(IDC/Canalys数据)及美股主要消费科技股。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="产品创新驱动周期 + 全球居民消费周期",
            typical_length="2~3年（经历新品爆发、渗透率见顶、换机周期拉长与去库存）",
            capacity_lag="6~12个月（精密制造产线改造与打样量产周期）",
            key_cycle_indicators=["全球智能手机与PC季度出货量同比", "面板价格(群智咨询/WitsView数据)", "重点代工厂月度营收数据", "元器件库存周转天数"],
        ),
        risks=RiskMatrix(
            geopolitical=["海外大客户供应链外迁(印度/越南)风险", "海外进出口关税与贸易摩擦"],
            supply_chain_bottlenecks=["高端光学传感器(CIS)", "高阶折叠屏铰链材料", "核心主控芯片供应"],
            technology_substitution=["AI手机/AI PC创新不及预期未能引发换机潮", "新交互硬件形态替代传统终端"],
            policy_regulatory=["电子产品环保与碳中和回收合规要求", "海外数据隐私与安全监管"],
            demand_cliff=["宏观经济下行导致居民消费降级，换机周期进一步拉长至36-40个月"],
        ),
        key_metrics=["大客户营收集中度(单一客户占比)", "毛利率与净利率变动", "存货减值与呆滞物料计提", "产能利用率与旺季加班工时", "折旧与摊销占营业成本比例"],
        representative_segments=["精密结构件与外观件", "面板与显示模组", "光学镜头与摄像头模组", "电声器件与振动马达", "ODM/EMS整机组装代工"],
    ),

    "liquor_beverage": IndustryProfile(
        industry_id="liquor_beverage",
        industry_name="白酒与精制茶酒",
        category="大消费",
        aliases=["白酒", "高端白酒", "次高端白酒", "白酒板块", "烈酒", "茅台", "五粮液"],
        upstream=[
            "红缨子糯高粱/小麦/大米等粮食作物",
            "陶瓷酒瓶/玻璃瓶/包装纸盒/酒标",
            "水资源与优质生态酿造产区",
            "酿造发酵曲药",
        ],
        downstream=[
            "政商务宴请与商务社交消费",
            "婚庆寿宴与居民大众聚饮消费",
            "礼品赠送与金融投资收藏属性",
            "餐饮与烟酒店/商超/电商渠道网络",
        ],
        core_inputs=["优质酿酒原粮", "水质与微生物环境", "多年陶坛储存基酒陈酿时间成本", "白酒包装物与人工"],
        pricing_power="高端白酒拥有极强的品牌壁垒、文化护城河与自主提价权；中低端与次高端受商务活动景气度及库存压力制约，议价权分化。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="低（对基准利率不直接敏感，但低利率环境下高股息与稳定现金流白酒具防御配置价值）",
            fx_rate="低（以国内消费为主，出海占比极低，受汇率直接冲击极小）",
            commodity_inflation="低（原粮与包材占营业成本比例很低，毛利率高达70%~90%，抵御通胀能力极强）",
            liquidity="中高（机构投资者重仓核心资产，流动性与外资流向对板块估值中枢影响较大）",
            policy_drivers=["消费税改革与征收环节后移预期", "商务接待与公务用餐规范政策", "扩大内需与促进消费政策"],
            global_macro_linkage="外资(北向资金)流动对核心白酒龙头持仓具有风向标意义；与国内宏观经济活跃度及固定资产投资预期正相关。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="宏观商务景气周期 + 渠道库存蓄水池周期",
            typical_length="4~6年（受房地产/基建投资、商务宴请及渠道加杠杆/去库存驱动）",
            capacity_lag="5年以上（高端酱香/浓香白酒从生产到出厂需要4-5年陈酿周期，供给刚性强）",
            key_cycle_indicators=["核心高端白酒批价(散瓶/原箱飞天茅台、普五批价)", "渠道经销商库存月数", "酒企合同负债(预收账款)环比变动", "中秋/国庆/春节旺季动销反馈"],
        ),
        risks=RiskMatrix(
            geopolitical=["无直接地缘制裁风险，主要受宏观外资进出流动性影响"],
            supply_chain_bottlenecks=["核心酿造产区独特地理与微生物环境不可复制", "陈年老酒基酒储备刚性限制"],
            technology_substitution=["年轻一代消费群体对烈酒饮用习惯变化与低度潮饮分流"],
            policy_regulatory=["白酒消费税税率调整或征收环节改为批发/零售端", "严禁违规吃喝与公务消费限制常态化"],
            demand_cliff=["宏观商务活动降温导致次高端白酒批价倒挂、渠道压货爆仓与动销停滞"],
        ),
        key_metrics=["营业收入与归母净利润增速", "毛利率与净利率", "合同负债(预收款)与现金回款质量", "应收票据及应收账款规模", "批价与建议零售价价差"],
        representative_segments=["高端白酒(千元以上)", "次高端白酒(300-800元)", "大众及区域名酒(100-300元)", "光瓶酒与低端白酒"],
    ),

    "food_beverage": IndustryProfile(
        industry_id="food_beverage",
        industry_name="大众食品与饮料",
        category="大消费",
        aliases=["食品饮料", "调味品", "乳制品", "软饮料", "休闲食品", "速冻食品", "预制菜", "啤酒"],
        upstream=[
            "大豆/小麦/白糖/生鲜乳/棕榈油/大麦等农产品",
            "PET塑料瓶胚/铝制易拉罐/无菌纸盒包装材料",
            "食品添加剂/香精香料/酵母",
            "冷链物流与仓储配送服务",
        ],
        downstream=[
            "餐饮连锁与传统餐饮门店",
            "大型连锁商超/便利店/夫妻老婆店",
            "零食量贩店/社区团购/即时零售与电商",
            "家庭日常终端消费",
        ],
        core_inputs=["农副原材料(原奶/大豆/大麦/糖)", "包装物(PET/铝罐/纸箱)", "冷链物流运费", "渠道进场费与营销推广费"],
        pricing_power="具备全国知名品牌与深度渠道掌控力的龙头可进行成本转嫁提价；同质化大众品议价权弱，易陷入买赠促销价格战。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="低（大众必需品需求刚性，对利率变化不敏感）",
            fx_rate="低中（大豆、大麦、棕榈油、白糖等部分原料依赖进口，汇率贬值推高采购成本）",
            commodity_inflation="高（大宗农产品和包材价格上涨会直接压缩毛利率，需关注成本周期拐点）",
            liquidity="低（防御性必选消费品，在流动性收缩或经济弱复苏期往往具备超额收益）",
            policy_drivers=["食品安全国家标准与溯源体系建设", "农村电商与县域商业体系建设", "减糖健康引导政策"],
            global_macro_linkage="关注CBOT大豆/小麦期货价格、全球原糖与棕榈油价格波动、国际海运费变化。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="弱周期必选消费 + 原材料成本逆周期",
            typical_length="2~3年（主要受上游大宗农产品丰歉周期与包材成本波动主导）",
            capacity_lag="6~12个月（食品饮料加工产线扩建周期短）",
            key_cycle_indicators=["生鲜乳收购均价", "大豆/白糖/大麦现货及期货价格", "铝锭/PET切片价格指数", "商超与量贩零食渠道动销走势"],
        ),
        risks=RiskMatrix(
            geopolitical=["大豆、大麦、白糖等进口农产品关税与供应链受限"],
            supply_chain_bottlenecks=["极端天气与厄尔尼诺对全球主要农产品产区的减产冲击"],
            technology_substitution=["健康无糖/低卡替代传统含糖饮品，即饮茶/现制茶饮分流包装软饮料"],
            policy_regulatory=["食品安全黑天鹅事件与舆情风险", "环保与包装物回收合规要求"],
            demand_cliff=["消费偏好向平价理性倾斜，传统高毛利单品面临量贩渠道低价冲击"],
        ),
        key_metrics=["主营业务收入与毛利率", "销售费用率(费用投放ROI)", "应收账款周转天数与存货周转天数", "单品动销月度增速", "经销商数量与单商平均提货额"],
        representative_segments=["调味发酵品(酱油/醋/复合调料)", "乳制品(常温奶/低温鲜奶/奶酪)", "啤酒(高端化精酿)", "软饮料(无糖茶/功能饮料/包装水)", "休闲食品与量贩零食", "速冻食品与预制菜"],
    ),

    "home_appliances": IndustryProfile(
        industry_id="home_appliances",
        industry_name="家用电器与智能家居",
        category="大消费",
        aliases=["家电", "白色家电", "黑色家电", "厨电", "小家电", "空调", "冰箱", "洗衣机", "电视"],
        upstream=[
            "压缩机/电机/电磁阀/智能控制器",
            "铜管/铝箔/冷轧钢板/镀锌板",
            "塑料粒子(ABS/PP/PS)/发泡料",
            "显示面板/驱动芯片/PCB板",
        ],
        downstream=[
            "国内新建商品房新装与精装修交付",
            "存量住宅家电以旧换新置换需求",
            "海外出口市场(欧洲/北美/新兴市场自主品牌与OEM代工)",
            "商业地产与公共建筑中央空调/商用冷链",
        ],
        core_inputs=["铜材/铝材/冷轧钢板", "塑料原材料", "压缩机与核心芯片", "仓储物流与渠道安装售后服务费"],
        pricing_power="白电双雄具备极高行业集中度与全产业链规模壁垒，具备较强成本转嫁能力；小家电与二三线品牌竞争激烈，议价权偏弱。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="中（联动房地产成交景气与海外耐用品消费信贷）",
            fx_rate="高（家电行业出口占比高达30%~50%，汇率贬值直接增厚出口毛利与汇兑收益）",
            commodity_inflation="极高（铜、铝、钢材、塑料占生产成本60%以上，原材料价格是盈利核心扰动项）",
            liquidity="中（地产链与宏观刺激政策敏感型品种）",
            policy_drivers=["消费品以旧换新与绿色智能家电补贴政策", "保交楼与房地产支持政策", "家电能效新国标实施"],
            global_macro_linkage="关注海外主要经济体通胀与零售库存、海运集装箱运价(CCFI/SCFI)及海外贸易关税。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="地产后周期 + 存量更新周期 + 成本大宗周期",
            typical_length="3~4年（受地产竣工、气候温度(厄尔尼诺酷暑)与原材料成本周期共振）",
            capacity_lag="6~12个月（组装产能弹性大，但核心压缩机/电机产能存在一定门槛）",
            key_cycle_indicators=["产业在线月度空调/冰箱/洗衣机内销与出口排产数据", "LME铜/沪铜期货价格与铝价走势", "全国商品房住宅竣工与销售面积", "以旧换新政策申领进度"],
        ),
        risks=RiskMatrix(
            geopolitical=["欧美对中国家电加征关税及针对海外生产基地(墨西哥/越南)的产地调查"],
            supply_chain_bottlenecks=["车规级/工业级高可靠智能主控MCU芯片", "特种制冷剂配额限制"],
            technology_substitution=["集成灶对传统烟灶分流", "热泵与全屋舒适系统对传统分体空调的场景升级"],
            policy_regulatory=["新能效等级标准提高倒逼老旧机型库存降价出清", "制冷剂环保配额削减"],
            demand_cliff=["国内房地产销售持续低迷拖累新装需求", "海外经济衰退导致外需耐用品订单收缩"],
        ),
        key_metrics=["内销与外销出货量增速", "原材料成本占比与综合毛利率", "经销商库存系数", "海外自有品牌(OBM)占比", "净资产收益率(ROE)与现金分红率"],
        representative_segments=["白电(空调/冰箱/洗衣机)", "厨电(油烟机/燃气灶/集成灶)", "小家电(扫地机器人/空气炸锅)", "黑电(智能电视/激光投影)", "中央空调与商用热泵冷链"],
    ),

    "banking": IndustryProfile(
        industry_id="banking",
        industry_name="商业银行与信贷",
        category="金融地产",
        aliases=["银行", "商业银行", "国有大行", "股份行", "城商行", "农商行", "信贷", "银行业"],
        upstream=[
            "央行基础货币投放与再贷款/MLF/PSL",
            "居民储蓄存款与企业活期/定期存款",
            "同业拆借/同业存单/金融债发行",
            "金融科技IT系统与风控大数据服务",
        ],
        downstream=[
            "实体企业贷款(制造业/基建/普惠小微/高新科技)",
            "房地产开发贷与地方政府融资平台",
            "居民个人按揭贷款/消费贷/信用卡借款",
            "债券投资(国债/地方债/金融债/信用债)",
        ],
        core_inputs=["存款利息支出与资金获取成本", "信用风险资产减值损失计提", "网点运营与金融科技人员薪酬", "监管资本充足率占用成本"],
        pricing_power="大行依靠庞大低成本网点与存款沉淀占据成本优势；在优质资产端面临全行业同质化争夺，资产端定价权受LPR下调挤压。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="极高（LPR与存款基准利率调整直接决定净息差(NIM)水平，降息往往压缩息差空间）",
            fx_rate="中（外币资产负债敞口及跨境贸易融资受汇率微调影响）",
            commodity_inflation="低（间接影响企业客户经营状况与还本付息能力）",
            liquidity="高（央行降准、公开市场操作力度决定银行超额准备金与流动性覆盖率(LCR)）",
            policy_drivers=["存量房贷利率调降政策", "地方政府隐性债务化解置换", "普惠小微与科技创新专项再贷款支持", "银行业资本监管新规"],
            global_macro_linkage="关注中美利差走势、人民币汇率预期及全球巴塞尔协议III终局方案落地。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="典型宏观经济信贷强周期 + 资产质量滞后周期",
            typical_length="5~8年（紧密绑定全社会宏观杠杆率扩张与去杠杆信用周期）",
            capacity_lag="受资本充足率(CAR)与风险加权资产(RWA)扩张节奏约束",
            key_cycle_indicators=["净息差(NIM)季度环比变动", "不良贷款率(NPL)与关注类贷款占比", "拨备覆盖率", "社融与人民币新增贷款月度数据", "存贷利差与LPR报价"],
        ),
        risks=RiskMatrix(
            geopolitical=["跨境金融交易结算合规风险与SWIFT系统相关风险"],
            supply_chain_bottlenecks=["优质生息资产荒(有效信贷需求不足导致资产端收益率下行)"],
            technology_substitution=["金融科技与第三方移动支付对传统柜面及结算中间收入的长期侵蚀"],
            policy_regulatory=["让利实体经济导向对息差的持续挤压", "房贷首付比例及存量按揭利率调整政策", "中小金融机构兼并重组化险"],
            demand_cliff=["宏观经济承压引发房地产与地方城投信用风险暴露，导致不良率上升与资产减值计提侵蚀利润"],
        ),
        key_metrics=["净息差(NIM)", "不良贷款率与不良生成率", "拨备覆盖率与拨贷比", "非利息收入占比", "核心一级资本充足率(CET1)", "股息率(高分红属性)"],
        representative_segments=["国有大型商业银行", "全国性股份制银行", "区域性城市商业银行", "农村商业银行"],
    ),

    "securities": IndustryProfile(
        industry_id="securities",
        industry_name="证券公司与资本市场",
        category="金融地产",
        aliases=["证券", "券商", "非银金融", "投行", "经纪业务", "两融", "资管", "自营业务"],
        upstream=[
            "交易所交易结算系统与金融数据终端(Wind/同花顺)",
            "银行间与交易所拆借质押资金(转融通/短期融资券)",
            "金融IT系统(恒生电子/顶点软件等核心交易系统)",
        ],
        downstream=[
            "个人散户与高净值个人投资者",
            "机构投资者(公募基金/私募基金/保险资管/外资QFI)",
            "拟上市与已上市企业客户(IPO/再融资/并购重组/债券承销)",
        ],
        core_inputs=["资金拆借利息成本", "信息技术与系统运维投入", "投行保荐/财富管理/投研专业人员薪酬"],
        pricing_power="经纪业务佣金率已降至极低水平(万分之二以下)，费率价格战见底；重资本自营与衍生品业务依赖资本实力与风控定价；投行业务向头部集中。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="高（降息降准提振市场流动性与风险偏好，降低券商自身发债融资成本）",
            fx_rate="中（人民币汇率企稳回升吸引外资流入A股，提升市场成交活跃度）",
            commodity_inflation="低（主要受二级市场行情与风险资产估值主导）",
            liquidity="极高（被称为\"牛市旗手\"，对A股成交量、两融余额、市场换手率具备极高高贝塔(Beta)弹性）",
            policy_drivers=["建设一流投资银行与头部券商并购重组政策", "全面注册制改革深化与IPO/再融资节奏逆周期调节", "公募基金与证券行业费率改革", "互换便利(SFISF)等结构性货币工具支持"],
            global_macro_linkage="紧密联动港股及全球主要股票市场指数表现、外资北向资金净买入额及中国资产风险溢价。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="典型极高贝塔(Beta)资本市场牛熊周期",
            typical_length="3~5年（完全取决于A股市场交投活跃度、股指涨跌与融资节奏）",
            capacity_lag="无明显物理产能滞后，主要受净资本风控指标与监管杠杆倍数约束",
            key_cycle_indicators=["A股全市场单日成交金额(万亿级关口)", "全市场融资融券余额", "中证全指/上证指数点位与换手率", "IPO与再融资每月发行承销规模", "权益自营投资收益率"],
        ),
        risks=RiskMatrix(
            geopolitical=["跨境资本流动管制与海外上市监管审查(如中概股审计底稿/美股IPO审查)"],
            supply_chain_bottlenecks=["重资本业务依赖净资本规模，中小券商资本补充渠道受限"],
            technology_substitution=["量化交易生态变迁与互联网券商低佣金流量冲击"],
            policy_regulatory=["IPO节奏阶段性收紧导致投行业务收入腰斩", "公募基金降费降佣政策持续压降分仓佣金收入", "监管严控高杠杆场外衍生品"],
            demand_cliff=["A股市场长期缩量阴跌导致自营投资大幅亏损与经纪两融业务收入萎缩"],
        ),
        key_metrics=["经纪业务净收入与市占率", "自营投资收益率(年化ROI)", "投资银行业务承销金额", "两融余额与利息净收入", "净资本与杠杆倍数", "加权平均ROE"],
        representative_segments=["头部综合型券商(航母级券商)", "互联网特色零售券商", "特色投行/资管精品券商", "区域特色中小券商"],
    ),

    "insurance_financials": IndustryProfile(
        industry_id="insurance_financials",
        industry_name="保险与多元金融",
        category="金融地产",
        aliases=["保险", "人身险", "寿险", "财险", "车险", "再保险", "多元金融", "信托", "租赁"],
        upstream=[
            "长久期资产供给(超长期国债/地方债/高评级非标/基础设施债权计划)",
            "精算数据系统与核保核赔系统",
            "再保险分保服务提供商",
        ],
        downstream=[
            "居民养老与重疾/医疗健康保障需求",
            "个人汽车与财产综合保险投保人",
            "企业财产/责任险/工程险/农业保险投保主体",
            "高净值客户财富传承与家族信托",
        ],
        core_inputs=["保单负债成本(预定利率与分红/万能结算利率)", "理赔支出与退保金", "代理人佣金与渠道费用", "投资端资产减值"],
        pricing_power="头部险企具备庞大代理人队伍与品牌信任度，在财险(车险/非车险)具备较强精算与承保盈利壁垒；寿险负债端定价受监管预定利率严格调控。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="极高（长端国债利率下行引发利差损风险，同时压低新增固收类资产再投资收益率）",
            fx_rate="低中（海外资产配置受QDII额度与汇率波动影响）",
            commodity_inflation="低（间接影响财产险车险理赔维修成本）",
            liquidity="高（权益投资收益率对股票市场表现敏感，直接计入利润或OCI综合收益）",
            policy_drivers=["下调人身险产品预定利率(防范利差损)", "报行合一降低寿险与车险渠道费用", "养老金融第三支柱与个人养老金税收优惠政策", "险资入市考核周期拉长支持长期投资"],
            global_macro_linkage="关注全球长端国债收益率中枢走势、新会计准则(IFRS 9 / IFRS 17)实施对报表波动的跨国映射。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="利率周期 + 资本市场资产端周期 + 代理人改革周期",
            typical_length="5~10年（资产负债长久期匹配属性，跨越完整利率与经济周期）",
            capacity_lag="受偿付能力充足率(核心/综合偿付能力)约束",
            key_cycle_indicators=["中国10年期国债到期收益率", "寿险新业务价值(NBV)增速", "财险综合成本率(COR)", "总投资收益率与净投资收益率", "核心偿付能力充足率"],
        ),
        risks=RiskMatrix(
            geopolitical=["全球地缘冲突引发的海外投资敞口减值"],
            supply_chain_bottlenecks=["长期低利率环境下长久期安全高收益资产供给匮乏(资产荒)"],
            technology_substitution=["互联网保险第三方平台对传统代理人渠道分流"],
            policy_regulatory=["下调人身险预定利率引发短期保费退潮", "报行合一监管严查导致银保与经代渠道保费承压", "新会计准则下利润表波动加剧"],
            demand_cliff=["居民当期收入预期下调导致寿险保单购买力减弱与退保率上升", "长端利率持续低迷导致利差损隐患发酵"],
        ),
        key_metrics=["新业务价值(NBV)与NBV Margin", "内含价值(EV)及增长率", "综合成本率(COR，低于100%为承保盈利)", "净投资收益率与总投资收益率", "综合偿付能力充足率"],
        representative_segments=["人身险与寿险", "财产险与车险", "健康险与养老险", "金融信托与金融租赁"],
    ),

    "steel_ferrous": IndustryProfile(
        industry_id="steel_ferrous",
        industry_name="钢铁与黑色金属",
        category="周期大宗",
        aliases=["钢铁", "普钢", "特钢", "螺纹钢", "热卷", "铁矿石", "焦炭", "黑色金属"],
        upstream=[
            "铁矿石(澳洲/巴西力拓、必和必拓、淡水河谷等进口矿)",
            "炼焦煤与冶金焦炭",
            "废钢原料",
            "高炉电力、石灰石、耐火材料",
        ],
        downstream=[
            "房地产建筑与住宅开发(螺纹钢/线材)",
            "基建工程(桥梁/铁路/公路/水利/管网)",
            "汽车制造与白色家电(冷轧板/热轧板/镀锌板)",
            "船舶制造与重型集装箱(中厚板)",
            "风电/核电/航空航天/高端制造(特钢/高温合金)",
        ],
        core_inputs=["铁矿石", "冶金焦", "废钢", "电力与转炉能耗"],
        pricing_power="上游铁矿石高度垄断(四大矿山)，下游建筑地产分散且承压；普钢深陷两头受挤压困境，议价权极弱；特种高温合金/硅钢龙头具备较高议价权。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="中（影响房地产开发商与基建投资融资成本）",
            fx_rate="中高（铁矿石全靠美元计价进口，人民币贬值直接推高原料采购成本）",
            commodity_inflation="极高（铁矿石与焦炭价格大幅上涨吞噬钢厂吨钢利润）",
            liquidity="中（与地方政府专项债发行、基建实物工作量和地产融资紧密相关）",
            policy_drivers=["钢铁行业粗钢产量平控政策", "超低排放绿色改造与碳市场扩容", "兼并重组提高产业集中度", "特钢国产化与关键材料攻关"],
            global_macro_linkage="关注全球铁矿石发运量(巴西/澳洲港口离港数据)、海外用钢需求及中国钢材出口退税与反倾销政策。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="强周期大宗工业品",
            typical_length="3~5年（受房地产投资、基建逆周期调节与环保供给侧限产主导）",
            capacity_lag="1~2年（受产能置换政策严格限制，新建高炉/电炉门槛极高）",
            key_cycle_indicators=["螺纹钢/热卷现货价格与期货主力合约", "铁矿石普氏指数(Platts)", "吨钢毛利与高炉开工率", "钢材社会库存与钢厂库存(五大品种库存)", "日均铁水产量"],
        ),
        risks=RiskMatrix(
            geopolitical=["铁矿石进口来源高度集中于澳洲巴西的地缘供应链安全风险", "欧美对中国出口钢材加征碳关税(CBAM)及反倾销税"],
            supply_chain_bottlenecks=["海外优质高品位铁矿石资源依赖度高", "特种高温合金与高等级轴承钢工艺壁垒"],
            technology_substitution=["短流程电炉炼钢(EAF)对长流程高炉转炉的低碳替代", "铝合金与复合材料在汽车轻量化中对传统钢材的替代"],
            policy_regulatory=["粗钢产量平控限产目标落空导致供给泛滥", "环保减排与能耗双控限产导致生产受限"],
            demand_cliff=["国内房地产新开工面积持续大幅下滑导致建筑用钢需求断崖式下跌"],
        ),
        key_metrics=["吨钢毛利/吨钢净利", "高炉与电炉产能利用率", "存货减值准备", "资产负债率", "特钢与高附加值产品占比"],
        representative_segments=["建筑用普钢(螺纹钢/线材)", "制造业板材(热轧/冷轧/中厚板)", "无缝钢管与焊管", "高端特钢与不锈钢", "高温合金与特种轴承钢"],
    ),

    "nonferrous_metals": IndustryProfile(
        industry_id="nonferrous_metals",
        industry_name="有色金属与工业金属",
        category="周期大宗",
        aliases=["有色金属", "铜", "铝", "锌", "铅", "锡", "工业金属", "电解铝", "铜矿"],
        upstream=[
            "铜精矿/铝土矿/锌精矿等矿山开采",
            "海外矿山采矿权与港口海运物流",
            "冶炼辅料(硫酸/烧碱/阳极炭块)",
            "电解铝火电/水电绿电能耗",
        ],
        downstream=[
            "电力电网(特高压/配电网/电线电缆——用铜第一大户)",
            "新能源汽车与传统汽车(车身轻量化铝板/动力电池铜箔铝箔/电机绕组)",
            "光伏与风电(光伏铝边框/支架/风电电缆)",
            "家电与消费电子(空调铜管/铝制散热件)",
            "建筑装饰与包装材料",
        ],
        core_inputs=["铜精矿/铝土矿/氧化铝", "电力能源(电解铝度电成本占比40%左右)", "海运费与冶炼加工费(TC/RC)"],
        pricing_power="全球大宗定价权主要由国际期货交易所(LME/COMEX)及大型跨国矿业巨头决定；电解铝受国内4500万吨产能天花板硬约束，具备较强供给刚性与定价权。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="高（铜被称为\"铜博士\"，是全球宏观经济晴雨表，对全球降息与流动性周期高度敏感）",
            fx_rate="极高（工业金属以美元计价，美元指数走弱直接推高大宗商品价格）",
            commodity_inflation="极高（自身即为通胀核心传导介质，大宗上涨带动全产业链名义价格）",
            liquidity="极高（对全球M2增速、美联储货币政策与投机资金仓位极度敏感）",
            policy_drivers=["中国电解铝4500万吨合规产能红线", "特高压与电网智能化大规模投资计划", "海外矿业资源开采税率调整与出口配额"],
            global_macro_linkage="紧密联动LME铜铝期货价格、COMEX黄金/白银、美联储利率决议及全球制造业PMI。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="强周期大宗全球定价工业品",
            typical_length="4~7年（受全球资本开支周期、矿山投产周期与宏观经济波动主导）",
            capacity_lag="矿山从勘探开发到投产需5~8年，冶炼厂建设需2~3年（供给刚性极强）",
            key_cycle_indicators=["LME与沪铜/沪铝期货主力合约价格", "铜精矿现货加工费(TC/RC)", "全球交易所库存与保税区库存(LME/SHFE库存)", "氧化铝现货价格", "中国制造业PMI及电网投资增速"],
        ),
        risks=RiskMatrix(
            geopolitical=["南美(智利/秘鲁)及非洲矿区社区罢工、环保抗议与采矿权国有化风险", "几内亚铝土矿政局波动"],
            supply_chain_bottlenecks=["海外优质高品位铜矿资源品位下滑与极端天气断供", "国内电解铝产能天花板硬性限制"],
            technology_substitution=["高导电铝合金在部分电缆领域对铜的替代(以铝节铜)"],
            policy_regulatory=["云南/内蒙等主要电解铝基地枯水期限电限产", "欧盟碳关税(CBAM)对铝制品出口限制"],
            demand_cliff=["全球宏观经济陷入深度衰退导致工业金属消费断崖下滑"],
        ),
        key_metrics=["矿产铜/电解铝单位生产现金成本(C1成本)", "自给矿比例与资源储量", "吨铝/吨铜综合毛利", "冶炼加工费(TC/RC)水平", "存货价值与资产负债率"],
        representative_segments=["铜采选与冶炼加工", "铝土矿-氧化铝-电解铝一体化", "铝加工与轻量化铝型材", "锌/铅/锡等基本金属", "铜箔/铝箔等新能源电池金属材料"],
    ),

    "precious_metals": IndustryProfile(
        industry_id="precious_metals",
        industry_name="贵金属与稀缺资源",
        category="周期大宗",
        aliases=["贵金属", "黄金", "白银", "稀土", "钨", "钼", "锑", "小金属", "战略金属"],
        upstream=[
            "金矿/银矿/稀土矿山采选",
            "选矿药剂与氰化提金试剂",
            "采矿设备与井下开采工程",
        ],
        downstream=[
            "全球央行官方外汇黄金储备增持",
            "金融避险与民间黄金首饰/金条投资",
            "工业用途(光伏银浆/电子元器件金丝引线)",
            "国防军工与硬质合金(钨/钼/锑等战略金属)",
            "永磁电机与机器人伺服(稀土钕铁硼)",
        ],
        core_inputs=["矿石品位与开采采选能耗", "环保合规与尾矿库治理成本", "人工与炸药爆破耗材"],
        pricing_power="黄金由全球货币属性与实际利率决定；战略小金属(稀土/锑/钨)受国家开采配额与出口管制主导，中国掌握核心供应话语权。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="极高（黄金价格与美国10年期实际利率(TIPS)呈高度负相关关系）",
            fx_rate="极高（黄金以美元计价，与美元指数呈显著负相关；去美元化背景下与央行购金需求强挂钩）",
            commodity_inflation="高（作为终极硬通胀抗衡工具，在滞胀与通胀失控预期下表现卓越）",
            liquidity="极高（全球流动性过剩、主权信用货币贬值预期下黄金配置需求爆发）",
            policy_drivers=["国家战略性矿产资源开采总量控制指标", "关键稀缺矿产出口管制与战略收储", "全球央行去美元化与多元化储备策略"],
            global_macro_linkage="紧密联动伦敦现货黄金(XAU/USD)、COMEX黄金期货、美国实际利率、全球地缘政治风险指数(GPR)。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="主权信用/实际利率长周期 + 避险脉冲周期",
            typical_length="8~10年以上超级货币信用周期",
            capacity_lag="金矿从发现勘探到投产需7~10年，高品位矿源日益稀缺",
            key_cycle_indicators=["美国10年期TIPS实际收益率", "美元指数(DXY)", "世界黄金协会(WGC)全球央行季度净购金量", "全球最大黄金ETF(SPDR Gold Trust)持仓量", "COMEX非商业多头持仓净头寸"],
        ),
        risks=RiskMatrix(
            geopolitical=["海外矿山所在国(非洲/中亚/南美)政局动荡与资源民族主义剥夺采矿权"],
            supply_chain_bottlenecks=["国内战略小金属资源枯竭与低品位矿开采成本攀升"],
            technology_substitution=["光伏领域去银化(铜电镀/无银浆料)对白银工业需求的潜在挤压"],
            policy_regulatory=["环保督察叫停违规尾矿库开采", "稀土整合重组与配额收紧"],
            demand_cliff=["美联储超预期大幅加息推高实际利率导致金价回落", "地缘风险溢价消退引发投机资金快速撤离"],
        ),
        key_metrics=["克金综合维持成本(AISC)", "金银矿石储量与平均品位", "自产金比例与外购合质金比例", "黄金现货均价与结算毛利率", "资源开采年限"],
        representative_segments=["金矿开采与黄金冶炼", "白银采选与工业银深加工", "稀土采选与高性能钕铁硼磁材", "钨/钼/锑等战略小金属"],
    ),

    "petrochemicals": IndustryProfile(
        industry_id="petrochemicals",
        industry_name="石油石化与基础化工",
        category="周期大宗",
        aliases=["石油", "石化", "基础化工", "炼化", "原油", "化纤", "聚酯", "塑料", "农药", "化肥"],
        upstream=[
            "原油与天然气开采(中石油/中石化/海外油气田)",
            "煤炭(煤化工甲醇/烯烃路线)",
            "原盐、纯碱、磷矿石、硫磺",
        ],
        downstream=[
            "交通运输燃料(汽油/柴油/航煤)",
            "纺织服装(涤纶长丝/锦纶/化纤织物)",
            "汽车与家电(工程塑料/合成橡胶/改性塑料)",
            "农业种植(尿素/磷肥/钾肥/农药)",
            "日用化学品与医药中间体",
        ],
        core_inputs=["原油/天然气/原料煤", "蒸汽与电力能源消耗", "催化剂与助剂", "大型炼化装置折旧"],
        pricing_power="上游原油受OPEC+与地缘格局主导；大型民营大炼化具备一体化成本优势；大宗基础化工品产能过剩竞争激烈，议价权偏弱；精细特种化学品有溢价。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="中（影响全球宏观总需求与工业制造活跃度）",
            fx_rate="极高（原油100%以美元计价进口，汇率波动直接影响炼化企业原料采购成本与库存计价）",
            commodity_inflation="极高（布伦特/WTI原油是全球大宗商品之母，油价剧烈波动直接影响产品毛利与库存损益）",
            liquidity="中高（大宗商品期货交易属性与投机资金参与度高）",
            policy_drivers=["能耗双控向碳排放双控转变政策", "成品油出口配额发放", "化工园区安全环保规范与准入门槛提高"],
            global_macro_linkage="紧密联动布伦特(Brent)与WTI原油期货价格、OPEC+部长级会议减产决议、中东地缘政治危机及美国页岩油库存(EIA数据)。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="强周期大宗工业品 + 油价成本驱动周期",
            typical_length="3~5年（受原油超级周期与下游制造业补库/去库周期共振）",
            capacity_lag="2~3年（大型百万吨乙烯/芳烃一体化炼化项目建设调试周期长）",
            key_cycle_indicators=["布伦特原油现货与期货价格", "石化产品价差(如PX-原油价差、PTA价差、聚酯长丝单吨盈利)", "炼厂开工率与炼油裂解价差", "化工品社会库存与港口库存"],
        ),
        risks=RiskMatrix(
            geopolitical=["中东霍尔木兹海峡与红海航道封锁推高油运与油价", "海外原油禁运与制裁风险"],
            supply_chain_bottlenecks=["高端聚烯烃弹性体(POE)、高端电子化学品依赖进口"],
            technology_substitution=["生物基化学品对化石基化学品的部分替代", "新能源车渗透率提升长期压制汽油消费需求"],
            policy_regulatory=["安全环保事故引发的区域性化工园区停产整顿", "严格限制新增传统炼化过剩产能"],
            demand_cliff=["全球经济衰退导致原油暴跌产生巨额原材料在途存货跌价损失", "下游纺织与房地产需求疲软"],
        ),
        key_metrics=["炼油与化工产品价差(Crack Spread)", "原油库存损益", "装置负荷率(开工率)", "固定资产投资折旧与经营性现金流", "ROE与资产负债率"],
        representative_segments=["油气开采与油服装备", "民营大炼化与聚酯化纤", "煤化工(煤制烯烃/乙二醇)", "农化(化肥/农药)", "精细化工与特种电子化学品"],
    ),

    "coal_energy": IndustryProfile(
        industry_id="coal_energy",
        industry_name="煤炭与传统化石能源",
        category="周期大宗",
        aliases=["煤炭", "动力煤", "炼焦煤", "焦煤", "焦炭", "无烟煤", "煤企", "神华"],
        upstream=[
            "煤炭资源勘探与采矿权出让",
            "煤矿采掘机械与综采支架装备",
            "露天与井工开采电力与爆破炸药",
            "大秦铁路/浩吉铁路等铁路运力与港口中转",
        ],
        downstream=[
            "火力发电厂(动力煤——消耗全社会煤炭60%以上)",
            "钢铁冶炼与焦化厂(焦煤/焦炭)",
            "水泥建材企业(窑炉燃料)",
            "煤化工企业(甲醇/煤制油/合成氨)",
        ],
        core_inputs=["采矿权摊销与安全生产费计提", "井下掘进人工与电力能耗", "铁路运费与港口港杂费"],
        pricing_power="国内煤炭供给受安全环保与安监核查约束，供给弹性极低；长协煤机制平抑中枢价格，市场现货煤具备较高议价权。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="低（煤企普遍拥有充沛现金流与极低负债率，高股息属性在低利率环境下极具吸引力）",
            fx_rate="低中（部分沿海电厂采购印尼/澳洲/蒙古进口煤，受进口汇率微调）",
            commodity_inflation="高（煤价是国内基础工业电价与制造能源成本基石）",
            liquidity="低（主要受基本面供需、长协保供与红利策略资金偏好影响）",
            policy_drivers=["煤炭保供稳价政策与长协合同履约率考核", "矿山安全生产大检查常态化", "超长期特别国债与能源安全战略储备建设"],
            global_macro_linkage="关注纽卡斯尔动力煤现货价格、蒙古焦煤通关车数及印尼煤炭出口政策。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="供给强约束下的弱周期/高股息红利资产",
            typical_length="3~5年（受宏观工业用电增速、气候气温与安监供给约束主导）",
            capacity_lag="3~5年（新建大型现代化矿井审批严格、核准周期长）",
            key_cycle_indicators=["秦皇岛港5500大卡动力煤平仓价", "北方港口煤炭调入量与合计库存", "沿海八省电厂日耗煤量与可用天数", "京唐港主焦煤库提价", "煤炭长协覆盖比例"],
        ),
        risks=RiskMatrix(
            geopolitical=["海外煤炭进口关税调整与地缘贸易配额限制"],
            supply_chain_bottlenecks=["极端雨雪天气与大秦线检修导致的铁路运力瓶颈"],
            technology_substitution=["风电/光伏/核电/储能等新能源装机加速对火电发电小时数的长期挤压"],
            policy_regulatory=["长协煤价格区间行政调控与保供履约刚性要求", "矿难事故引发的大面积区域性停产整顿"],
            demand_cliff=["宏观工业用电增速放缓或水电丰水期大发挤压火电煤炭日耗"],
        ),
        key_metrics=["自产煤吨煤完全生产成本", "长协煤与市场煤销售比例", "吨煤净利与净资产收益率(ROE)", "自由现金流与股息分红率(Dividend Yield)", "煤炭核定产能与剩余可采储量"],
        representative_segments=["动力煤龙头(高分红长协属性)", "炼焦煤企(弹性冶金煤属性)", "无烟煤与喷吹煤", "煤电一体化综合能源企"],
    ),

    "power_utilities": IndustryProfile(
        industry_id="power_utilities",
        industry_name="电力与公用事业",
        category="公用基建",
        aliases=["电力", "公用事业", "火电", "水电", "核电", "绿电", "风电", "电网", "燃气", "水务"],
        upstream=[
            "动力煤/天然气/核燃料(天然铀与燃料组件)",
            "风力发电机组/光伏组件/水轮发电机组/核岛设备",
            "输配电特高压与电网调度系统",
        ],
        downstream=[
            "高耗能工业企业(钢铁/水泥/电解铝/化工)",
            "一般工商业企业与写字楼商超",
            "居民生活用电与公共照明",
            "充电桩与数据中心等新型大负荷主体",
        ],
        core_inputs=["电煤/天然气燃料成本(火电燃料占成本70%左右)", "大坝/核电机组/发电设备折旧", "长周期银行贷款利息支出"],
        pricing_power="电价受政府管制与电力市场化交易规则约束；容量电价机制为火电托底；水电/核电度电成本极低且稳定，具备极强成本护城河。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="高（水电/核电/风光基建前期资本开支极大，负债率较高，降息显著节约财务费用）",
            fx_rate="低（以内销电量为主，仅天然铀与海外设备有少量外币敞口）",
            commodity_inflation="极高（对火电而言，动力煤价格决定盈利死活，煤价下跌即为火电业绩爆发催化剂）",
            liquidity="低（高股息类公用事业，防御属性强，在熊市或低利率环境下具备较强配置吸引力）",
            policy_drivers=["煤电容量电价机制全面落地", "电力现货市场与辅助服务市场建设", "绿电消纳与可再生能源配额制", "核电新机组常态化核准"],
            global_macro_linkage="关注全球天然气与天然铀现货价格、国际碳排放交易价格走势。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="火电呈成本逆周期；水电/核电呈稳定现金流弱周期",
            typical_length="水电核电30~60年运营长周期；火电跟随煤价呈2-3年波动周期",
            capacity_lag="水电8~10年，核电5~6年，火电/风电1~2年",
            key_cycle_indicators=["全社会用电量月度同比增速", "各省电力市场化交易电价较基准价上浮比例", "三峡及主要流域入库水流量与水电发电量", "动力煤价格与火电度电利润", "发电设备利用小时数"],
        ),
        risks=RiskMatrix(
            geopolitical=["天然铀进口受限与核电海外供应链合规风险"],
            supply_chain_bottlenecks=["特高压外送通道建设滞后导致区域性弃风弃光弃水"],
            technology_substitution=["四代高温气冷堆与可控核聚变前沿技术突破对传统能源结构的长期重塑"],
            policy_regulatory=["电力现货市场日前/实时价格大幅波动与负电价风险", "新能源绿电上网电价无序竞争下跌"],
            demand_cliff=["极端干旱枯水导致水电发电量骤降", "宏观工业用电需求萎缩导致机组利用小时数下滑"],
        ),
        key_metrics=["发电设备平均利用小时数", "市场化交易电量比例与度电综合售价", "度电燃料成本(火电)", "资产负债率与财务费用率", "自由现金流与分红比例"],
        representative_segments=["水力发电龙头(长江电力等金沙江流域大水电)", "核电运营", "煤电及煤电联营企业", "新能源风光绿电运营", "燃气与城市供水"],
    ),

    "real_estate": IndustryProfile(
        industry_id="real_estate",
        industry_name="房地产开发与运营",
        category="金融地产",
        aliases=["房地产", "地产", "物业管理", "房企", "商业地产", "住宅开发", "二手房", "土地市场"],
        upstream=[
            "地方政府土地出让与国土规划",
            "建筑总包工程与建安施工企业",
            "水泥/钢铁/玻璃/防水涂料/电梯等建材与设备",
            "银行开发贷/信托资金/境内外发债融资",
        ],
        downstream=[
            "城镇居民刚性与改善型购房需求",
            "企业办公写字楼与商铺租售",
            "产业园区与物流仓储运营方",
            "存量物业持有运营与社区生活服务",
        ],
        core_inputs=["土地购置出让金", "建安工程施工与原材料成本", "开发贷款利息资本化与财务费用", "营销渠道分销佣金"],
        pricing_power="供求关系逆转后，除一线城市核心地段高端改善盘外，绝大多数城市新房无定价溢价，房企以价换量去库存。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="极高（房贷利率是购房者核心成本，融资利率直接决定房企债务存续生死）",
            fx_rate="中高（部分出险房企发行大量美元离岸债，汇率贬值加剧偿债负担）",
            commodity_inflation="中（建安成本受钢筋、水泥影响，但地价与资金利息占主导）",
            liquidity="极高（与全社会信贷总量、M1/M2剪刀差及居民杠杆率高度共振）",
            policy_drivers=["取消限购限售限价政策与降低首付比例", "房贷利率下限取消与公积金利率调降", "保交房白名单贷款支持与城中村改造", "收购存量商品房用作保障性住房(收储政策)"],
            global_macro_linkage="关注离岸中资美元地产债违约率与重组进展、全球核心城市房地产资本化率(Cap Rate)。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="宏观人口与金融信用超级大周期",
            typical_length="大周期10~15年，短周期库存波动3~4年",
            capacity_lag="从拿地、开工、预售到竣工交付周期2~3年",
            key_cycle_indicators=["30大中城市商品房单日/单周成交面积", "克而瑞百强房企单月全口径销售金额", "全国房地产开发投资与新开工面积同比", "居民中长期贷款月度新增额", "土地流拍率与溢价率"],
        ),
        risks=RiskMatrix(
            geopolitical=["离岸美元债违约引发的跨国诉讼与清盘呈请风险"],
            supply_chain_bottlenecks=["金融机构对非白名单民营房企融资授信的风险厌恶"],
            technology_substitution=["代建轻资产模式与保障房租赁对传统高周转重资产开发模式的冲击"],
            policy_regulatory=["预售资金监管账户严格穿透管理导致资金调用受限", "房地产税出台预期对购房预期的压制"],
            demand_cliff=["人口结构变化与城镇化放缓导致居民中长期购房意愿趋势性减弱"],
        ),
        key_metrics=["全口径与权益销售金额/面积", "销售回款率", "净负债率与现金短债比", "存货减值与投资性房地产公允价值变动", "在建与竣工交付面积"],
        representative_segments=["央国企优质信用住宅开发商", "高能级城市改善型房企", "轻资产代建与物业管理", "商业地产运营与仓储物流REITs"],
    ),

    "construction_materials": IndustryProfile(
        industry_id="construction_materials",
        industry_name="建筑装饰与基础设施工程",
        category="公用基建",
        aliases=["建筑", "基建", "建材", "水泥", "玻璃", "工程机械", "建筑装饰", "水利基建", "八大建筑央企"],
        upstream=[
            "钢铁/水泥熟料/砂石骨料/沥青/平板玻璃",
            "工程机械(挖掘机/起重机/盾构机/装载机)",
            "地方政府专项债/特别国债/政策性金融工具",
            "设计院与勘察测绘服务",
        ],
        downstream=[
            "国家与地方基础设施建设(铁路/公路/机场/港口/水利/管网)",
            "城市地下管网改造与城市更新",
            "新能源大基地基建(风电光伏大基地土建/抽水蓄能电站)",
            "工业厂房建设与出海工程(一带一路沿线国家项目)",
            "商业地产与住宅精装修",
        ],
        core_inputs=["水泥与砂石等建材", "钢材与结构件", "劳务分包工人工资", "工程机械燃油与折旧", "垫资施工资金利息"],
        pricing_power="建筑央企具备大型复杂工程总包资质与垫资融资优势；水泥建材受运输半径(200-300公里)限制具备区域定价权；普通装饰工程议价弱。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="高（基建投资极度依赖债务融资，降息显著缓解地方政府与业主的还本付息压力）",
            fx_rate="中（海外工程出海承包以美元/欧元结算，汇率贬值提升折算利润）",
            commodity_inflation="高（钢材、水泥、柴油价格上涨直接压缩施工总包合同固定单价毛利）",
            liquidity="高（依赖地方政府专项债发行节奏与银行基建配套贷款资金到位率）",
            policy_drivers=["超长期特别国债支持国家重大战略与重点领域安全能力建设(两重建设)", "地方政府化债与清理拖欠企业账款", "大规模设备更新与城市地下管网改造", "高质量共建一带一路倡议"],
            global_macro_linkage="关注海外一带一路沿线国家主权主权信用评级、国际大宗工程承包竞争态势。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="逆周期基建稳增长政策驱动 + 地产后周期",
            typical_length="3~5年（受国家财政赤字、专项债节奏与五年规划周期主导）",
            capacity_lag="水泥等建材受错峰生产与产能置换约束，工程总包受资质和垫资能力约束",
            key_cycle_indicators=["地方政府新增专项债每月发行规模", "全国固定资产投资(基建投资)同比增速", "全国水泥均价(PO42.5散装)与磨机开工率", "建筑央企新签订单季度同比增速", "沥青装置开工率与出货量"],
        ),
        risks=RiskMatrix(
            geopolitical=["海外一带一路地缘冲突与部分国家主权债务违约导致工程款无法回收"],
            supply_chain_bottlenecks=["高品质天然砂石短缺与环保开采限制"],
            technology_substitution=["装配式建筑与模块化建造对传统现浇施工工艺的替代"],
            policy_regulatory=["地方隐性债务化解对低收益基建项目的严格叫停与退库审查", "严查虚假开工与垫资合规性"],
            demand_cliff=["地方财政收支承压导致基建项目资金到位缓慢、工程停工与应收账款账期无限拉长"],
        ),
        key_metrics=["新签合同额与在手未完工订单(订单保障倍数)", "应收账款与合同资产规模(坏账计提比例)", "收现比与经营性现金流净额", "毛利率与净利率", "资产负债率与带息负债规模"],
        representative_segments=["八大基建建筑央企(中国中铁/中国铁建/中国建筑等)", "区域水泥熟料龙头", "防水材料与建筑涂料", "装配式钢结构", "国际工程承包商"],
    ),

    "industrial_machinery": IndustryProfile(
        industry_id="industrial_machinery",
        industry_name="机械设备与工业母机",
        category="先进制造",
        aliases=["机械", "工业母机", "机床", "自动化", "工业机器人", "注塑机", "激光设备", "工程机械", "智能制造"],
        upstream=[
            "数控系统(CNC/伺服驱动器/伺服电机)",
            "主轴/丝杠/导轨/减速机(RV减速器/谐波减速器)",
            "铸件/特种钢材/铝合金板材",
            "工业传感器/激光器/气动液压元件",
        ],
        downstream=[
            "汽车及新能源汽车制造(冲压/焊接/总装产线)",
            "3C消费电子与半导体封装测试精密加工",
            "航空航天与国防军工高端曲面结构件制造",
            "风电核电及重大能源装备加工",
            "通用制造业通用设备更新与自动化产线",
        ],
        core_inputs=["高精度数控系统与伺服电机", "高精密轴承与导轨丝杠", "铸件与特种钢材", "研发与装配调试高级技工薪酬"],
        pricing_power="五轴联动高档数控机床与核心精密零部件掌握技术定价权；中低端三轴/四轴机床同质化严重，价格竞争激烈。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="中（影响制造业企业设备更新的资本开支意愿与中长期贷款成本）",
            fx_rate="中高（高端核心部件从日本/德国进口，同时中国机械设备整机出口竞争力持续增强）",
            commodity_inflation="中高（生铁、废钢、铜、铝原材料价格波动传导至机械铸件成本）",
            liquidity="中（关联制造业企业固定资产投资与信贷支持力度）",
            policy_drivers=["推动大规模设备更新和消费品以旧换新行动方案", "工业母机高质量发展税收优惠与研发补贴", "制造业单项冠军与专精特新支持政策", "智能制造示范工厂建设"],
            global_macro_linkage="紧密关注日本机床工业会(JMTBA)月度机床订单额(全球制造景气先行指标)、全球制造业PMI。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="典型制造业资本开支朱格拉周期(设备更新周期)",
            typical_length="7~10年（受设备物理磨损与技术换代驱动，内嵌3~4年小周期）",
            capacity_lag="6~18个月（高端定制化机床装配调试与客户现场验收周期长）",
            key_cycle_indicators=["中国制造业PMI及生产指数", "日本对华机床订单金额月度同比", "工业机器人月度产量数据", "挖掘机国内与出口销量数据", "制造业固定资产投资增速"],
        ),
        risks=RiskMatrix(
            geopolitical=["高端数控系统与精密五轴机床出口管制及关键零部件断供"],
            supply_chain_bottlenecks=["高精度主轴轴承、高刚性滚珠丝杠与滚柱导轨依赖日本THK/NSK/德国博世力士乐"],
            technology_substitution=["增材制造(3D打印)在航空航天复杂异形件制造中对传统减材切削机床的局部替代"],
            policy_regulatory=["设备更新补贴申请流程与资金兑付进度不及预期"],
            demand_cliff=["下游汽车/3C制造业资本开支意愿骤降导致新增机床订单断崖下滑"],
        ),
        key_metrics=["新签订单与在手未交付订单金额", "综合毛利率与高档机床占比", "研发投入占比", "存货周转天数与发出商品规模", "海外出口销售额占比"],
        representative_segments=["高档五轴数控机床", "工业机器人与自动化系统集成", "减速器/丝杠/伺服等核心零部件", "激光加工装备", "注塑机与压铸机", "工程机械(挖机/起重机)"],
    ),

    "defense_military": IndustryProfile(
        industry_id="defense_military",
        industry_name="国防军工与航天装备",
        category="先进制造",
        aliases=["军工", "国防", "航空装备", "航天", "导弹", "军工电子", "舰船制造", "商业航天", "低空经济"],
        upstream=[
            "军工高温合金/钛合金/碳纤维复合材料",
            "军品级高可靠元器件/抗辐射芯片/微波射频器件",
            "特种火炸药与推进剂原料",
            "军用光电探测器/惯性导航陀螺仪",
        ],
        downstream=[
            "陆军/海军/空军/火箭军/战略支援等军种最终装备列装",
            "国家航天局与商业卫星互联网星座组网",
            "民用航空大飞机(C919/ARJ21等配套)",
            "军贸出口(中东/非洲/东南亚友好国家采购)",
            "低空经济eVTOL与无人机应用",
        ],
        core_inputs=["特种高性能金属与复合材料", "宇航级电子元器件", "重大科研攻关与试飞试验成本", "高密级军工研发人员薪酬"],
        pricing_power="军品定价主要遵循军方审价机制(以往为成本加成，现逐步转向目标价格激励约束定价)；核心系统与稀缺材料具备高壁垒。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="低（属于国家刚性战略安全支出，对市场利率与货币周期几乎完全脱敏）",
            fx_rate="低（以内销军品为主，军贸结算受特定外汇结算安排保护）",
            commodity_inflation="低（军工产品毛利率较高，且审价机制可部分消化特种原材料成本）",
            liquidity="中高（板块估值受地缘局势情绪、五年规划中期调整与军工主题资金偏好主导）",
            policy_drivers=["五年规划武器装备建设采购订单落地", "建设世界一流军队战略目标与练兵备战需求", "商业航天与低空经济纳入战略性新兴产业", "军工央企资产证券化与股权激励"],
            global_macro_linkage="关注全球地缘冲突热点(台海/南海/中东/东欧)、全球主要大国国防预算增长率。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="国家五年规划指令性订单周期 + 装备代际换装周期",
            typical_length="5年（紧随国家五年规划前松后紧、中期调整与终期冲刺节奏）",
            capacity_lag="2~4年（军工资质军标认证、定型批产与扩产保密产线建设周期长）",
            key_cycle_indicators=["全国国防预算年度增幅与决算执行", "军工板块合同负债与大额预付款变动", "五年规划中期调整订单下发节奏", "军工主机厂交付与收入确认季节性进度"],
        ),
        risks=RiskMatrix(
            geopolitical=["西方对中国军工实体清单全面制裁与全面技术封锁"],
            supply_chain_bottlenecks=["宇航级特种DSP/FPGA/ADC芯片", "极端工况高可靠传感器"],
            technology_substitution=["无人化/智能化/低成本巡飞弹对传统高昂有人装备的战术挑战"],
            policy_regulatory=["军品采购定价机制改革与降价招标压力", "人事调整与军品招标采购节奏延后"],
            demand_cliff=["型号研制推迟或军种需求结构性调整导致部分配套厂商订单断档"],
        ),
        key_metrics=["合同负债与大额预收款", "在手订单饱满度与交付周期", "研发费用率与型号定型进展", "毛利率与军品审价补差金额", "应收账款(军方与主机厂回款账期)"],
        representative_segments=["军用航空飞机与发动机", "航天装备与导弹武器", "军工电子与元器件", "特种高温合金与碳纤维材料", "商业航天与卫星遥感通信", "无人机与低空飞行器"],
    ),

    "logistics_shipping": IndustryProfile(
        industry_id="logistics_shipping",
        industry_name="交通运输与航运港口",
        category="公用基建",
        aliases=["交运", "航运", "港口", "快递", "物流", "供应链", "集运", "油运", "干散货", "航空货运"],
        upstream=[
            "集装箱船/超大型油轮(VLCC)/散货船等造船厂",
            "船用低硫重油/柴油/航空煤油",
            "集装箱制造与港口装卸吊装设备",
            "干线货运卡车与自动化物流分拣设备",
        ],
        downstream=[
            "国际贸易跨国货主与跨境电商平台",
            "国内大宗原材料采购企业(钢厂铁矿石/电厂煤炭/炼厂原油)",
            "国内电商平台(淘宝/京东/拼多多/抖音)与C端消费者",
            "制造业供应链零部件干线运输",
        ],
        core_inputs=["船用燃料油与航空燃油", "船舶与飞机租金及折旧", "港口装卸码头费与运河通行费", "货车司机与快递员劳动力成本"],
        pricing_power="国际集运油运由全球运力供需与地缘突发事件决定，具备极高运价弹性；国内电商快递深陷恶性价格战，议价权微弱；核心枢纽港口具备自然垄断属性。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="中（影响全球贸易融资与大型船舶资产购置贷款成本）",
            fx_rate="极高（国际海运费以美元结算，人民币贬值直接增加航运企业汇兑与折算收益）",
            commodity_inflation="高（国际原油及船用重油价格直接决定航运企业核心营业成本）",
            liquidity="中（与全球贸易总量、集装箱吞吐量及电商物流GMV增速相关）",
            policy_drivers=["全球海事组织(IMO)船舶碳减排环保新规", "快递行业反内卷与保障快递员合法权益政策", "国家综合立体交通网规划"],
            global_macro_linkage="紧密联动波罗的海干散货指数(BDI)、原油运输指数(BDTI)、上海出口集装箱运价指数(SCFI)、红海/苏伊士运河通行安全。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="强周期全球运力供需周期 + 地缘突发事件驱动",
            typical_length="3~5年（集运/油运运价呈现脉冲式爆发与断崖式回归特征）",
            capacity_lag="2~3年（大型集装箱船与油轮从船厂下单到交付下水周期）",
            key_cycle_indicators=["SCFI/CCFI集装箱运价指数", "BDI干散货运价指数", "VLCC等价期租租金(TCE水平)", "全球集装箱船在手订单占现有运力比例(Orderbook/Fleet)", "港口集装箱月度吞吐量"],
        ),
        risks=RiskMatrix(
            geopolitical=["红海危机/霍尔木兹海峡地缘冲突迫使船舶绕行好望角", "巴拿马运河干旱限行导致通航拥堵"],
            supply_chain_bottlenecks=["船厂船坞泊位排满导致新船交付推迟", "全球主要港口罢工与拥堵"],
            technology_substitution=["中欧班列陆路运输在特定货品上对海运的时效性补充分流"],
            policy_regulatory=["IMO国际海事组织碳排放能效指数(EEXI/CII)倒逼老旧船舶降速或拆解", "反垄断机构对班轮联盟联运航线的审查"],
            demand_cliff=["欧美贸易关税加征或全球经济衰退导致欧美补库结束、外贸货运需求骤降"],
        ),
        key_metrics=["综合运价指数与单位航次单箱毛利", "TCE(等价期租日租金)", "燃油成本占营收比例", "新造船订单占比与老旧船舶拆解量", "单票快递收入与单票毛利"],
        representative_segments=["国际集装箱航运(集运)", "国际原油与成品油运输(油运)", "全球干散货航运", "沿海枢纽港口运营", "国内快递与快运物流", "跨境电商供应链与跨境物流"],
    ),

    "telecom_optical": IndustryProfile(
        industry_id="telecom_optical",
        industry_name="通信网络与光通信",
        category="科技/TMT",
        aliases=["通信", "通信设备", "光模块", "光通信", "5G", "光纤光缆", "交换机", "运营商", "三大运营商"],
        upstream=[
            "激光器光芯片(EML/VCSEL/CW光源)/探测器芯片",
            "电芯片(DSP/TIA/Driver)/高速交换芯片",
            "光隔离器/光纤连接器/光学透镜",
            "通信基站天线/射频功放/滤波器/PCB板",
        ],
        downstream=[
            "电信运营商(中国移动/中国电信/中国联通)",
            "全球云厂商与AI大模型厂商数据中心(微软/Meta/谷歌/字节/阿里/腾讯)",
            "政企专网与智能交通通信",
            "千家万户家庭宽带与千兆光网接入",
        ],
        core_inputs=["高端光芯片与电芯片", "高频高速通信PCB板材", "精密光学组装与测试设备", "研发与测试工程师薪酬"],
        pricing_power="三大运营商对通信设备集采具有绝对买方定价权；AI高速光模块(800G/1.6T)具备技术代际壁垒，龙头厂商具备高定价权与高毛利。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="中（运营商具备高股息红利属性，对低利率敏感；光模块公司受全球科技成长估值主导）",
            fx_rate="高（光模块企业核心营收来自海外北美巨头，以美元结算为主，汇率贬值直接增厚业绩）",
            commodity_inflation="低中（铜材与光纤原材料有一定成本传导，但芯片与器件占主导）",
            liquidity="高（光模块作为AI算力核心硬件，流动性与科技板块风险偏好驱动明显）",
            policy_drivers=["双千兆协同发展行动计划与万兆光网试点", "算力网络国家枢纽节点直连网络建设", "电信运营商分红率提升与市值管理考核"],
            global_macro_linkage="紧密联动美股英伟达、北美云厂商算力网络升级架构、全球光通信行业展会(OFC)发布动态。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="技术代际演进驱动周期 (5G/6G、400G->800G->1.6T光模块)",
            typical_length="2~3年（光模块高速率产品迭代周期短，技术红利窗口期2-3年）",
            capacity_lag="6~12个月（光模块自动化测试封装产线扩产调试周期）",
            key_cycle_indicators=["北美云厂商资本开支中网络设备投入占比", "800G/1.6T光模块出货量与单价(ASP)", "电信运营商年度5G/算力资本开支预算", "光纤光缆集采中标价格"],
        ),
        risks=RiskMatrix(
            geopolitical=["海外限制中国通信设备及核心光器件在特定国家关键信息基础设施中的应用"],
            supply_chain_bottlenecks=["高端高速光芯片(200G EML)与高算力网络交换芯片高度依赖海外进口"],
            technology_substitution=["CPO(共封装光学)及硅光方案成熟对传统可插拔光模块技术路线的颠覆"],
            policy_regulatory=["电信资费提速降费政策对运营商ARPU值的潜在影响", "能耗指标限制数据中心网络交换设备功耗"],
            demand_cliff=["北美或国内云厂商AI大模型资本开支阶段性放缓导致光模块砍单"],
        ),
        key_metrics=["高速率光模块出货占比与毛利率", "电信运营商ARPU值与移动用户数", "在手订单与预收款", "研发费用占营业收入比例", "应收账款周转天数"],
        representative_segments=["三大电信运营商", "高速光模块与光器件", "通信网络主设备商(基站/路由器/交换机)", "光纤光缆与海缆", "通信射频与基站天线"],
    ),

    "agriculture_breeding": IndustryProfile(
        industry_id="agriculture_breeding",
        industry_name="农林牧渔与生猪养殖",
        category="大消费",
        aliases=["农业", "农林牧渔", "生猪养殖", "养猪", "禽养殖", "饲料", "种子", "种业", "动物疫苗", "白羽鸡"],
        upstream=[
            "玉米/豆粕/小麦/鱼粉等饲料原料",
            "种猪(原种猪/二元种猪)/种鸡等种源",
            "兽药/动物疫苗/抗生素",
            "养殖场建设(自动化猪舍/温控通风设备/环保排污系统)",
        ],
        downstream=[
            "生猪屠宰与冷鲜肉加工企业",
            "肉制品深加工企业(火腿肠/熟食/调理肉制品)",
            "农贸市场/大型商超/社区生鲜/餐饮门店",
            "居民日常肉蛋奶消费",
        ],
        core_inputs=["饲料原料(玉米与豆粕成本占养殖总成本60%左右)", "母猪折旧与仔猪成本", "疫病防控防疫支出与死淘损失", "养殖场能耗与人工"],
        pricing_power="生猪与肉禽属于完全竞争农产品，养殖场为纯价格接受者(Price Taker)；具备基因育种与全产业链成本优势的头部龙头拥有超额收益能力。",
        macro_sensitivity=MacroSensitivity(
            interest_rate="中（大型规模化猪企负债率较高，降息能减轻巨额猪舍建设贷款财务压力）",
            fx_rate="中（饲料主要原料大豆高度依赖进口，汇率贬值推高豆粕采购成本）",
            commodity_inflation="极高（玉米与豆粕期货价格直接决定完全养殖成本线；猪肉价格是国内CPI核心构成）",
            liquidity="中（猪周期底部猪企资金链紧绷，融资宽松决定产能出清节奏）",
            policy_drivers=["生猪产能调控实施方案(能繁母猪正常保有量调控目标)", "国家储备肉收储与投放调节机制", "转基因玉米大豆产业化应用与种业振兴行动"],
            global_macro_linkage="关注CBOT大豆/玉米期货价格、南美(巴西/阿根廷)大豆产量预估(USDA供需月报)及国际猪肉贸易流向。",
        ),
        cycle_profile=CycleProfile(
            cycle_type="典型蛛网模型供需错配强周期 (猪周期)",
            typical_length="3~4年（经历猪价大跌->母猪淘汰->供应断档->猪价暴涨->后备补栏->产能过剩）",
            capacity_lag="从能繁母猪受孕、分娩到仔猪育肥出栏需约10~11个月，扩繁引种需18个月以上",
            key_cycle_indicators=["全国能繁母猪存栏量", "生猪出栏均价与仔猪价格", "猪粮比价与全国生猪养殖头均盈利", "生猪出栏平均体重与屠宰场开工率", "涌益咨询/钢联农产品周度草根调研数据"],
        ),
        risks=RiskMatrix(
            geopolitical=["进口大豆关税与海外供应链中断风险"],
            supply_chain_bottlenecks=["优质种猪与白羽肉鸡祖代种鸡引种受限", "转基因种业自主育种转化率"],
            technology_substitution=["人造肉与植物蛋白在特定场景的潜在渗透(目前影响极小)"],
            policy_regulatory=["环保禁养区划定与粪污排污环保检查", "动物防疫法与重大动物疫病通报合规"],
            demand_cliff=["非洲猪瘟、禽流感等烈性疫病大面积爆发导致养殖场清栏与断崖式亏损", "极端消费淡季需求承压"],
        ),
        key_metrics=["生猪完全出栏成本(元/公斤)", "能繁母猪存栏量与PSY/MSY指标", "月度生猪出栏量与出栏均重", "资产负债率与现金流储备", "生产性生物资产规模"],
        representative_segments=["生猪养殖龙头(一体化自繁自养/公司+农户)", "白羽鸡与黄羽肉鸡养殖", "饲料加工与添加剂", "转基因玉米/水稻种业", "动物保健与兽用生物制品"],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 辅助查询与上下文生成函数
# ─────────────────────────────────────────────────────────────────────────────

def get_all_industries() -> List[IndustryProfile]:
    """获取所有行业图谱列表。"""
    return list(INDUSTRY_PROFILES.values())


def get_all_industry_names() -> List[str]:
    """获取所有行业标准名称。"""
    return [p.industry_name for p in INDUSTRY_PROFILES.values()]


def get_industry_profile(industry_or_alias: str) -> Optional[IndustryProfile]:
    """根据行业ID、标准名称或别名精确/模糊匹配行业画像。"""
    if not industry_or_alias or not isinstance(industry_or_alias, str):
        return None

    query = industry_or_alias.strip().lower()
    if not query:
        return None

    # 1. 尝试直接通过 ID 匹配
    if query in INDUSTRY_PROFILES:
        return INDUSTRY_PROFILES[query]

    # 2. 尝试标准名称或别名精确匹配
    for profile in INDUSTRY_PROFILES.values():
        if query == profile.industry_name.lower():
            return profile
        for alias in profile.aliases:
            if query == alias.lower():
                return profile

    # 3. 尝试子串包含匹配
    for profile in INDUSTRY_PROFILES.values():
        if query in profile.industry_name.lower() or profile.industry_name.lower() in query:
            return profile
        for alias in profile.aliases:
            if query in alias.lower() or alias.lower() in query:
                return profile

    return None


def search_industries(keyword: str) -> List[IndustryProfile]:
    """根据关键词搜索匹配的行业图谱列表。"""
    if not keyword or not isinstance(keyword, str):
        return []

    kw = keyword.strip().lower()
    matches: List[IndustryProfile] = []

    for profile in INDUSTRY_PROFILES.values():
        score = 0
        if kw in profile.industry_id.lower():
            score += 10
        if kw in profile.industry_name.lower():
            score += 10
        if any(kw in a.lower() for a in profile.aliases):
            score += 5
        if any(kw in u.lower() for u in profile.upstream):
            score += 3
        if any(kw in d.lower() for d in profile.downstream):
            score += 3
        if any(kw in s.lower() for s in profile.representative_segments):
            score += 4
        if any(kw in r.lower() for r in profile.risks.supply_chain_bottlenecks + profile.risks.geopolitical):
            score += 2

        if score > 0:
            matches.append(profile)

    return matches


def get_upstream_downstream_chain(industry_or_alias: str) -> Optional[Dict[str, Any]]:
    """获取指定行业的上下游产业链结构数据。"""
    profile = get_industry_profile(industry_or_alias)
    if not profile:
        return None

    return {
        "industry_id": profile.industry_id,
        "industry_name": profile.industry_name,
        "category": profile.category,
        "upstream": list(profile.upstream),
        "downstream": list(profile.downstream),
        "core_inputs": list(profile.core_inputs),
        "pricing_power": profile.pricing_power,
    }


def get_macro_sensitivity_matrix(industry_or_alias: str) -> Optional[Dict[str, Any]]:
    """获取指定行业的宏观敏感度矩阵。"""
    profile = get_industry_profile(industry_or_alias)
    if not profile:
        return None

    ms = profile.macro_sensitivity
    return {
        "industry_id": profile.industry_id,
        "industry_name": profile.industry_name,
        "interest_rate_sensitivity": ms.interest_rate,
        "fx_sensitivity": ms.fx_rate,
        "commodity_inflation_sensitivity": ms.commodity_inflation,
        "liquidity_sensitivity": ms.liquidity,
        "policy_drivers": list(ms.policy_drivers),
        "global_macro_linkage": ms.global_macro_linkage,
    }


def get_industry_risk_profile(industry_or_alias: str) -> Optional[Dict[str, Any]]:
    """获取指定行业的多维风险图谱。"""
    profile = get_industry_profile(industry_or_alias)
    if not profile:
        return None

    r = profile.risks
    return {
        "industry_id": profile.industry_id,
        "industry_name": profile.industry_name,
        "geopolitical_risks": list(r.geopolitical),
        "supply_chain_bottlenecks": list(r.supply_chain_bottlenecks),
        "technology_substitution": list(r.technology_substitution),
        "policy_regulatory": list(r.policy_regulatory),
        "demand_cliff": list(r.demand_cliff),
    }


def format_industry_deep_context(industry_or_alias: str) -> str:
    """生成紧凑、结构化的行业深度知识上下文文本，供 LLM Prompt/Agent 直接注入使用。

    若未匹配到特定行业，则返回空字符串或通用引导提示。
    """
    profile = get_industry_profile(industry_or_alias)
    if not profile:
        return ""

    ms = profile.macro_sensitivity
    cp = profile.cycle_profile
    rk = profile.risks

    lines = [
        f"【行业常识知识库 - {profile.industry_name} ({profile.category})】",
        f"1. 产业链上下游穿透：",
        f"   - 上游环节：{', '.join(profile.upstream)}",
        f"   - 下游应用：{', '.join(profile.downstream)}",
        f"   - 核心成本/要素：{', '.join(profile.core_inputs)}",
        f"   - 定价权与传导：{profile.pricing_power}",
        f"2. 宏观与周期敏感度：",
        f"   - 周期属性：{cp.cycle_type}（典型周期跨度：{cp.typical_length}，产能滞后期：{cp.capacity_lag}）",
        f"   - 宏观因子：利率[{ms.interest_rate}] | 汇率[{ms.fx_rate}] | 大宗通胀[{ms.commodity_inflation}] | 流动性[{ms.liquidity}]",
        f"   - 全球联动：{ms.global_macro_linkage}",
        f"   - 政策驱动：{'; '.join(ms.policy_drivers)}",
        f"3. 风险矩阵与监控指标：",
        f"   - 地缘/卡脖子：{'; '.join(rk.geopolitical + rk.supply_chain_bottlenecks)}",
        f"   - 替代/监管/需求：{'; '.join(rk.technology_substitution + rk.policy_regulatory + rk.demand_cliff)}",
        f"   - 关键跟踪指标：{', '.join(profile.key_metrics)}",
        f"   - 核心细分赛道：{', '.join(profile.representative_segments)}",
    ]

    return "\n".join(lines)
