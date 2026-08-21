"""宏观事件情景分析与三级传导链路图谱知识库 (Macro Events & Scenario Transmission Graph).

提供结构化的宏观事件静态常识情景库，支持：
1. 宏观事件三级传导链路 (一级直接冲击 -> 二级产业链与成本传导 -> 三级跨市场与跨行业外溢)；
2. 受益行业 (Beneficiary Sectors) 与受损行业 (Adversely Affected Sectors) 深度归因与逻辑推演；
3. 跨市场资产外溢联动 (A股、港股/中概、美债与美股、大宗商品、外汇汇率)；
4. 核心先行与验证高频监测指标；
5. 行业暴露度反向索引 (查询指定行业在各类宏观事件下的收益/风险敞口)；
6. 供 LLM Prompt 注入的紧凑、高信息密度情景推演上下文组装函数。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import re


@dataclass(frozen=True)
class SectorImpact:
    """行业受宏观事件影响的量化/定性描述。"""
    sector: str  # 行业/板块名称
    transmission_logic: str  # 传导逻辑与驱动机制
    impact_level: str  # 极高 / 高 / 中 / 低
    duration: str = "中期"  # 短期(脉冲) / 中期(季度) / 长期(跨年度趋势)
    key_drivers: List[str] = field(default_factory=list)  # 核心驱动因素或缓解对冲因素


@dataclass(frozen=True)
class MacroEventScenario:
    """宏观事件情景画像与三级传导链路。"""
    event_id: str
    event_name: str
    category: str  # 货币与流动性, 财政与基建, 外汇与国际收支, 大宗商品与能源, 地缘与国际贸易, 通胀与宏观周期, 产业技术突破, 资本市场制度
    aliases: List[str]
    description: str
    transmission_mechanism: List[str]  # 传导机制三步走：[Step 1 直接冲击, Step 2 产业链传导, Step 3 跨市场外溢]
    direct_impact: List[str]  # 对基准利率、汇率、大宗价格等直接变量的影响
    beneficiary_sectors: List[SectorImpact]  # 明确受益的行业列表及逻辑
    adversely_affected_sectors: List[SectorImpact]  # 明确承压受损的行业列表及风险点
    cross_market_spillovers: Dict[str, str]  # 跨市场联动：A股/港股/美债/大宗/汇率
    key_monitoring_indicators: List[str]  # 核心高频跟踪与验证指标
    historical_reference_cases: List[str]  # 历史典型情景常识参考


# ─────────────────────────────────────────────────────────────────────────────
# 19 大类宏观事件情景与传导链路静态知识库
# ─────────────────────────────────────────────────────────────────────────────

MACRO_EVENT_SCENARIOS: Dict[str, MacroEventScenario] = {
    "monetary_easing": MacroEventScenario(
        event_id="monetary_easing",
        event_name="央行降息降准与流动性宽松",
        category="货币与流动性",
        aliases=["降息", "降准", "货币宽松", "流动性充裕", "LPR下调", "MLF降息", "宽松货币政策"],
        description="中央银行下调政策利率(如逆回购/MLF/LPR)或降低法定存款准备金率，释放中长期流动性并降低全社会资金成本。",
        transmission_mechanism=[
            "一级直接冲击：银行间市场同业拆借利率(DR007/SHIBOR)快速下行，国债收益率曲线整体下移，银行超额准备金规模增加。",
            "二级产业链传导：实体经济与居民部门信贷融资成本下降，高负债企业财务费用显著节约，重资本开支行业(如基建、新能源、制造业)投资回报率预期提升。",
            "三级跨市场外溢：权益市场整体风险偏好回暖，分母端折现率下行驱动高估值成长股估值扩张，高股息资产比价优势提升，汇率短期承受温和贬值压力。",
        ],
        direct_impact=[
            "银行间DR007及短端国债利率下行",
            "人民币贷款市场报价利率(LPR)下调",
            "债券市场收益率下行(中长端国债价格上涨)",
            "人民币汇率短期面临利差收窄贬值压力",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="券商与非银金融",
                transmission_logic="市场交投活跃度回升、两融余额扩张，自营固收与权益投资浮盈增加。",
                impact_level="极高",
                duration="中期",
                key_drivers=["两融利率", "A股单日成交量", "自营投资收益率"],
            ),
            SectorImpact(
                sector="高估值成长科技(半导体/AI/光通信)",
                transmission_logic="DCF估值模型中折现率(WACC)下行，远期现金流价值大幅放大，估值乘数提升。",
                impact_level="高",
                duration="中长期",
                key_drivers=["无风险利率下行", "科技板块风险偏好提升"],
            ),
            SectorImpact(
                sector="重资产高负债行业(电力水务/工程基建/重工业)",
                transmission_logic="带息负债利息支出刚性下降，直接增厚税前净利润与经营现金流净额。",
                impact_level="高",
                duration="长期",
                key_drivers=["综合融资成本下降", "利息费用节约"],
            ),
            SectorImpact(
                sector="房地产开发与链条",
                transmission_logic="房贷利率下调降低居民购房门槛与月供压力，房企债务展期与再融资成本降低。",
                impact_level="中高",
                duration="中期",
                key_drivers=["首套房贷利率", "商品房成交面积"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="商业银行(尤其存款成本刚性中小行)",
                transmission_logic="资产端贷款重定价快于负债端存款降息，导致净息差(NIM)被动收窄，利息净收入承压。",
                impact_level="中高",
                duration="中长期",
                key_drivers=["净息差收窄幅度", "存款挂牌利率下调滞后"],
            ),
            SectorImpact(
                sector="人身险/寿险公司(投资端)",
                transmission_logic="固收类新增资产配置收益率中枢下移，长久期资产配置压力加大，利差损风险隐现。",
                impact_level="中",
                duration="长期",
                key_drivers=["10年期国债收益率破位下行", "保单负债预定利率成本刚性"],
            ),
        ],
        cross_market_spillovers={
            "A股": "整体估值提振，成长风格与高弹性券商领涨，高股息资产股息率利差扩大具备配置价值。",
            "港股/中概": "受南向资金增持与国内宏观流动性宽松溢出提振，恒生科技指数弹性较大。",
            "大宗商品": "国内基建与制造业投资预期改善，对螺纹钢、铜、铝等工业金属构成需求支撑。",
            "外汇汇率": "中美利差倒挂可能阶段性加深，对在岸/离岸人民币汇率带来短期贬值压力。",
        },
        key_monitoring_indicators=["DR007与公开市场操作利率偏离度", "1年期与5年期以上LPR报价", "月度新增社融与M2/M1增速", "10年期国债到期收益率"],
        historical_reference_cases=["2020年上半年全球流动性大放水行情", "2024年9月24日央行降准降息并设立互换便利(SFISF)组合拳行情"],
    ),

    "monetary_tightening": MacroEventScenario(
        event_id="monetary_tightening",
        event_name="央行加息与流动性收紧",
        category="货币与流动性",
        aliases=["加息", "收紧流动性", "提高准备金率", "金融去杠杆", "紧缩货币政策", "定向收紧"],
        description="中央银行上调政策基准利率、收紧公开市场流动性或提高存款准备金率，以遏制经济过热、资产泡沫或恶性通胀。",
        transmission_mechanism=[
            "一级直接冲击：银行间市场资金面骤紧，SHIBOR/同业存单利率飙升，长短端国债收益率快速上行。",
            "二级产业链传导：全社会企业与居民信贷成本显著攀升，企业推迟固定资产投资与扩产，居民缩减消费与房贷借借款。",
            "三级跨市场外溢：权益资产估值承受严重压制(杀估值)，高负债、现金流脆弱企业爆发信用违约风险，本币汇率相对坚挺。",
        ],
        direct_impact=[
            "货币市场利率与国债收益率全面上行",
            "商业银行新增贷款增速放缓",
            "债券市场价格整体承压走熊",
            "本币汇率获利差支撑企稳升值",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="头部大型商业银行",
                transmission_logic="资产端贷款定价迅速上浮，而活期存款沉淀占比高，净息差(NIM)实现趋势性扩张。",
                impact_level="高",
                duration="中期",
                key_drivers=["净息差扩大", "活期存款沉淀率高"],
            ),
            SectorImpact(
                sector="高现金流净现金企业(手握大量货币资金)",
                transmission_logic="货币资金利息收入显著提升，且无偿债压力，具备逆势低成本并购同行资产的能力。",
                impact_level="中",
                duration="中长期",
                key_drivers=["净负债率为负", "高现金储备利息收入"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="高估值成长科技与未盈利Biotech",
                transmission_logic="分母端折现率飙升引发估值剧烈杀跌，海外一级市场风险投资与二级市场再融资断流。",
                impact_level="极高",
                duration="中长期",
                key_drivers=["无风险利率飙升", "估值倍数严重压缩"],
            ),
            SectorImpact(
                sector="重资产高负债与房地产行业",
                transmission_logic="财务利息支出剧增，现金流断裂与债务展期违约风险呈指数级上升。",
                impact_level="极高",
                duration="长期",
                key_drivers=["带息负债利息成本暴涨", "再融资渠道闭合"],
            ),
            SectorImpact(
                sector="非银金融与券商",
                transmission_logic="二级市场交投低迷缩量，两融规模收缩，券商自营债券与股票持仓发生账面浮亏。",
                impact_level="高",
                duration="中期",
                key_drivers=["市场成交额暴跌", "自营投资收益下滑"],
            ),
        ],
        cross_market_spillovers={
            "A股": "全市场面临杀估值压力，高股息防御性资产跑赢大盘，成长板块与小盘股波动剧烈。",
            "港股/中概": "在联系汇率制或美元流动性收缩影响下，港股估值及流动性承压明显。",
            "大宗商品": "工业投资与总需求降温，原油、铜、铝等工业大宗价格普遍面临回调下行压力。",
            "外汇汇率": "利差走阔吸引外资配置本币债券，本币汇率相对坚挺升值。",
        },
        key_monitoring_indicators=["银行间拆借利率SHIBOR", "国债收益率曲线斜率", "全社会M2与信用利差", "企业债违约率"],
        historical_reference_cases=["2011年央行连续加息抗通胀行情", "2017-2018年金融去杠杆与资管新规落地期的市场调整"],
    ),

    "fiscal_expansion": MacroEventScenario(
        event_id="fiscal_expansion",
        event_name="积极财政政策与特别国债基建刺激",
        category="财政与基建",
        aliases=["积极财政", "超长期特别国债", "地方专项债发力", "基建刺激", "赤字率提升", "两重建设", "大规模设备更新"],
        description="政府扩大财政赤字规模、发行超长期特别国债或增加地方政府专项债额度，直接投向重大战略基建、新型基础设施及设备更新。",
        transmission_mechanism=[
            "一级直接冲击：政府部门大额发债筹集资金，国债发行量剧增，财政存款转化为项目直达资金与企业银行存款。",
            "二级产业链传导：重大基础设施工程与设备更新项目密集启动，总包建筑央企新签订单暴增，直接拉动水泥、钢铁、工程机械、特高压及工业母机采购订单。",
            "三级跨市场外溢：实体经济有效需求回暖，工业企业产能利用率回升，PPI同比由负转正，市场风险偏好显著改善，顺周期板块领涨。",
        ],
        direct_impact=[
            "重大基建工程与重大项目实物工作量快速形成",
            "财政支出增速与地方项目配套贷款大幅提升",
            "债券市场供给端放量可能引发短端利率扰动",
            "企业部门与地方政府流动性压力获得化解",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="大型建筑与基础设施央企",
                transmission_logic="重大战略工程与两重项目直接承建方，新签订单与营业收入大幅释放，现金流大幅改善。",
                impact_level="极高",
                duration="长期",
                key_drivers=["专项债配套资金到位率", "重大项目新签合同额增速"],
            ),
            SectorImpact(
                sector="工程机械与工业母机",
                transmission_logic="开工率提升与大规模设备更新财政补贴直接催化设备购置与更新替换需求。",
                impact_level="高",
                duration="中长期",
                key_drivers=["挖掘机开工小时数", "设备更新补贴落地"],
            ),
            SectorImpact(
                sector="水泥建材与黑色金属(钢铁)",
                transmission_logic="基建实物工作量形成带来直接采购拉动，水泥出货率与钢材表观消费量回升。",
                impact_level="高",
                duration="中期",
                key_drivers=["水泥磨机开工率", "螺纹钢表观消费量"],
            ),
            SectorImpact(
                sector="特高压与电网智能化基建",
                transmission_logic="作为国家绿色能源大基地外送与现代化电网的重点财政倾斜方向，订单景气度高企。",
                impact_level="高",
                duration="长期",
                key_drivers=["电网年度投资完成额", "特高压线路核准与招标"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="纯防御性债权固收资产",
                transmission_logic="国债海量发行形成供给冲击，加之经济复苏预期推高长端收益率，纯债资产面临资本利得回撤。",
                impact_level="中",
                duration="短期",
                key_drivers=["国债发行供给放量", "长端利率反弹"],
            ),
        ],
        cross_market_spillovers={
            "A股": "顺周期基建链、机械、材料及化债受益股迎来估值与盈利双击，市场风格从防御转向进攻。",
            "港股/中概": "内需基建与工业制造板块跟随上行，海外投资者对中国经济基本面信心修复。",
            "大宗商品": "螺纹钢、铁矿石、动力煤、沥青及工业金属现货与期货价格获强劲基本面支撑。",
            "外汇汇率": "经济基本面预期改善对人民币汇率形成坚实支撑，跨境资金回流。",
        },
        key_monitoring_indicators=["新增专项债与特别国债发行进度", "全国固定资产投资(基建)月度增速", "挖掘机国内单月销量与开工小时数", "水泥与沥青装置开工率"],
        historical_reference_cases=["2008年底四万亿基建投资计划", "2023年底增发一万亿国债支持灾后重建与水利基建行情"],
    ),

    "rmb_depreciation": MacroEventScenario(
        event_id="rmb_depreciation",
        event_name="人民币大幅贬值与汇率承压",
        category="外汇与国际收支",
        aliases=["人民币贬值", "汇率破7", "汇率贬值", "美元走强人民币走弱", "外汇贬值", "结售汇逆差", "贬值压力", "汇率承压", "人民币贬值压力"],
        description="受中美利差走阔、美元指数强势或国内经济复苏节奏影响，人民币对美元汇率发生较大幅度贬值。",
        transmission_mechanism=[
            "一级直接冲击：以美元计价的出口商品折算人民币收入增加，进口商品折算人民币采购成本同步上升。",
            "二级产业链传导：纺织服装、轻工家电、光模块、汽车零部件等外销导向型企业毛利率显著增厚，并产生丰厚汇兑净收益；进口原料依赖型行业(航空、炼化、造纸、铁矿石冶炼)成本骤升，产生汇兑净损失。",
            "三级跨市场外溢：以人民币计价的核心资产对海外外资吸引力面临汇兑风险考验，北向资金可能阶段性净流出，加剧A股大盘蓝筹短期抛压。",
        ],
        direct_impact=[
            "美元兑离岸/在岸人民币汇率上行(如破7.10/7.20/7.30关口)",
            "进口原材料与大宗商品到岸成本攀升",
            "出口制造企业结算外汇汇兑收益大增",
            "外汇储备及央行外汇风险准备金工具启动",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="电子代工与消费电子零部件(外销占比>50%)",
                transmission_logic="产品以美元报价结算，汇率贬值直接增厚人民币计价毛利，并贡献大额财务汇兑收益。",
                impact_level="极高",
                duration="中期",
                key_drivers=["外销收入占比", "美元净资产头寸与汇兑损益"],
            ),
            SectorImpact(
                sector="光通信与高速光模块",
                transmission_logic="北美云厂商订单主要以美元结算，人民币贬值直接提升毛利率与净利润率。",
                impact_level="高",
                duration="中期",
                key_drivers=["北美市场出货占比", "美元结算比例"],
            ),
            SectorImpact(
                sector="家用电器与工具五金出口商",
                transmission_logic="海外性价比优势进一步强化，外销订单增长且报表毛利增厚。",
                impact_level="高",
                duration="中期",
                key_drivers=["家电出口排产增速", "海外自主品牌定价"],
            ),
            SectorImpact(
                sector="汽车零部件出海与整车出口",
                transmission_logic="提升在欧洲、东南亚、拉美等市场的综合价格竞争力与汇兑收益。",
                impact_level="中高",
                duration="中期",
                key_drivers=["整车与零部件出口量同比"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="民航客运与航空公司",
                transmission_logic="拥有巨额以美元计价的飞机租赁负债与航油进口采购成本，汇率贬值产生巨额账面汇兑净亏损。",
                impact_level="极高",
                duration="中期",
                key_drivers=["美元负债净敞口", "汇兑损失占净利润比例"],
            ),
            SectorImpact(
                sector="石油炼化与基础化工(纯进口原油加工)",
                transmission_logic="原油100%以美元计价进口，汇率贬值直接垫高进料成本，若国内成品油调价滞后将挤压裂解价差。",
                impact_level="高",
                duration="中期",
                key_drivers=["美元进口原油成本", "炼油毛利压缩"],
            ),
            SectorImpact(
                sector="造纸与纸制品(依赖进口木浆)",
                transmission_logic="海外进口商品木浆以美元计价，人民币贬值推高原材料成本，终端纸价转嫁困难。",
                impact_level="中高",
                duration="中期",
                key_drivers=["进口木浆均价", "白卡纸/文化纸毛利率"],
            ),
            SectorImpact(
                sector="外资重仓的核心资产蓝筹(白酒/白马大盘股)",
                transmission_logic="外资担忧汇率折算亏损而阶段性减仓撤出，对重仓股票产生资金面抛压。",
                impact_level="中",
                duration="短期",
                key_drivers=["北向资金单日净流出规模", "外资持股集中度"],
            ),
        ],
        cross_market_spillovers={
            "A股": "结构分化剧烈，出口链板块(纺织/家电/电子)逆势大涨，大盘蓝筹与高负债进口链承压。",
            "港股/中概": "由于港币挂钩美元，中资港股企业的人民币计价盈利折算成港币出现缩水，恒指承压。",
            "大宗商品": "国内以内盘计价的人民币黄金(AU9999)、原油、铜期货价格往往强于外盘(内外盘溢价拉大)。",
            "外汇汇率": "央行通常通过逆周期因子、下调外汇存款准备金率或发行离岸央票进行汇率维稳引导。",
        },
        key_monitoring_indicators=["USD/CNH与USD/CNY即期与中间价", "出口集装箱货运量与海关出口金额同比", "央行结售汇差额", "北向资金净买入额"],
        historical_reference_cases=["2018年中美贸易摩擦期间汇率贬值出口链行情", "2022-2023年美联储激进加息引发的汇率承压与出口链分化"],
    ),

    "rmb_appreciation": MacroEventScenario(
        event_id="rmb_appreciation",
        event_name="人民币大幅升值与购买力提升",
        category="外汇与国际收支",
        aliases=["人民币升值", "汇率升值", "人民币走强", "外资流入", "结汇潮"],
        description="受国内经济强劲复苏、出口顺差高企或美联储降息美元指数走弱驱动，人民币汇率快速大幅升值。",
        transmission_mechanism=[
            "一级直接冲击：以人民币折算的进口原料成本大幅下降，海外大宗商品变得更便宜；出口商品折算人民币收入减少。",
            "二级产业链传导：航空、造纸、纯进口资源炼化等行业迎来原材料成本骤降与巨额汇兑收益；依赖价格战的低附加值出口代工企业毛利率承压。",
            "三级跨市场外溢：人民币资产全球吸引力飙升，外资(北向资金)汹涌净流入A股和港股市场，推升全市场估值中枢，大盘蓝筹核心资产领涨。",
        ],
        direct_impact=[
            "美元兑人民币汇率下行(如升破7.00/6.80关口)",
            "进口原材料成本显著下降",
            "外资持续大幅净买入中国资本市场资产",
            "出口制造企业面临汇兑损失与价格重谈压力",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="民航客运与航空公司",
                transmission_logic="巨额美元飞机租赁负债产生庞大汇兑收益，且航油进口成本折算人民币显著降低，业绩弹性极大。",
                impact_level="极高",
                duration="中期",
                key_drivers=["美元负债敞口", "航油人民币折算成本下降"],
            ),
            SectorImpact(
                sector="造纸与轻工业(进口木浆/废纸)",
                transmission_logic="进口木浆采购成本刚性下降，纸企综合毛利率实现显著扩张。",
                impact_level="高",
                duration="中期",
                key_drivers=["木浆进口成本下行", "造纸毛利率走阔"],
            ),
            SectorImpact(
                sector="外资偏好的大盘核心资产(白酒/医药/大金融)",
                transmission_logic="人民币升值强化海外资金持有中国核心资产意愿，外资净流入推升板块估值溢价。",
                impact_level="高",
                duration="中长期",
                key_drivers=["北向资金单月净买入额", "外资持股比例提升"],
            ),
            SectorImpact(
                sector="出境旅游与境外零售消费",
                transmission_logic="国内居民海外购买力直接增强，出境游意愿与免税消费客单价显著提升。",
                impact_level="中",
                duration="短期",
                key_drivers=["出境游人次", "免税客单价"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="低毛利纯出口代工制造(低端纺织/劳动密集型硬件)",
                transmission_logic="在手美元未结汇订单出现汇兑账面亏损，且新签订单面临降价竞争与汇率双重挤压。",
                impact_level="高",
                duration="中期",
                key_drivers=["外销毛利率极低(<15%)", "缺乏产品定价权"],
            ),
        ],
        cross_market_spillovers={
            "A股": "外资持续增持催生指数型大牛市，白酒、医药、新能源、大金融等权重股全线走强。",
            "港股/中概": "港股人民币资产折算港币盈利放大，港股迎来估值与流动性大幅重估。",
            "大宗商品": "以人民币计价的国内大宗商品现货表现相对弱于外盘美元大宗。",
            "外汇汇率": "银行代客结汇需求释放，结售汇呈现顺差，外汇储备规模稳步增加。",
        },
        key_monitoring_indicators=["USD/CNY与USD/CNH汇率走势", "北向资金(陆股通)每日净买入金额", "银行代客结售汇顺差", "中国外汇储备月度数据"],
        historical_reference_cases=["2020年下半年至2021年初人民币单边强劲升值行情", "2017年人民币汇率大幅逆转升值推动的核心资产大牛市"],
    ),

    "oil_price_shock_up": MacroEventScenario(
        event_id="oil_price_shock_up",
        event_name="原油与能源价格暴涨",
        category="大宗商品与能源",
        aliases=["油价暴涨", "原油飙升", "能源危机", "布伦特原油突破100美元", "石油危机", "OPEC减产"],
        description="受中东地缘冲突、主要产油国联合大幅减产或全球供应链中断驱动，国际布伦特/WTI原油及天然气价格暴涨。",
        transmission_mechanism=[
            "一级直接冲击：国际油价与成品油批发价格跳涨，全球航运燃油、柴油、汽油燃料价格暴增。",
            "二级产业链传导：上游油气开采与油田工程服务业绩爆发；下游交通物流、航空客运、汽车燃油成本剧增；石化中下游衍生塑料、化纤、橡胶成本全面推高，若下游转嫁困难将面临严重毛利挤压。",
            "三级跨市场外溢：推升全球CPI通胀预期，促使全球央行推迟降息或重启加息，高能耗制造板块估值承压，新能源替代(光伏/储能/电动车)经济性大幅凸显。",
        ],
        direct_impact=[
            "布伦特(Brent)与WTI原油期货暴涨突破高位",
            "国内成品油限价连续上调",
            "航空煤油与船用低硫重油成本激增",
            "全球主要经济体通胀预期(Breakeven Inflation)抬头",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="油气勘探开采与综合石油巨头(三桶油)",
                transmission_logic="原油完全开采成本相对固定，油价暴涨直接转化为超额上游油气开采暴利，现金流丰厚。",
                impact_level="极高",
                duration="中长期",
                key_drivers=["实现油价均价", "桶油作业成本", "自产油气当量产量"],
            ),
            SectorImpact(
                sector="油田技术服务与油气装备制造",
                transmission_logic="高油价驱动全球及国内油气公司大幅增加资本开支(Capex)，油服在手钻井订单量价齐升。",
                impact_level="高",
                duration="中长期",
                key_drivers=["油气勘探资本开支", "钻机日费率", "压裂设备订单"],
            ),
            SectorImpact(
                sector="煤化工(煤制烯烃/煤制乙二醇)",
                transmission_logic="油头路线成本暴涨，而国内煤炭原料价格相对可控，煤化工路线的成本比较优势大幅放大。",
                impact_level="高",
                duration="中期",
                key_drivers=["油煤比价(原油价格/动力煤价格)", "煤制聚烯烃单吨盈利"],
            ),
            SectorImpact(
                sector="新能源车与光储清洁能源",
                transmission_logic="燃油车用车成本大幅攀升，刺激消费者转向新能源车；清洁替代能源装机经济性增强。",
                impact_level="中高",
                duration="中长期",
                key_drivers=["燃油车与电动车每百公里能耗成本差", "光伏平价度电经济性"],
            ),
            SectorImpact(
                sector="原油远洋油运(VLCC油轮)",
                transmission_logic="地缘冲突拉长全球航运运距(吨海里需求增加)或引发浮仓囤油，油运运价TCE暴涨。",
                impact_level="高",
                duration="中期",
                key_drivers=["VLCC TCE日租金水平", "全球原油海运贸易路线重构"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="民航客运与航空公司",
                transmission_logic="航空煤油占航空公司总营业成本30%~40%，油价暴涨导致燃油附加费无法完全转嫁，盈利严重受损。",
                impact_level="极高",
                duration="中期",
                key_drivers=["航油出厂均价", "客座率与票价转嫁能力"],
            ),
            SectorImpact(
                sector="公路物流与货运快递",
                transmission_logic="柴油燃油成本刚性上涨，在运力过剩竞争格局下无法向货主完全提价，毛利受直接侵蚀。",
                impact_level="高",
                duration="中期",
                key_drivers=["柴油价格", "单票运费提价阻力"],
            ),
            SectorImpact(
                sector="传统燃油汽车制造",
                transmission_logic="消费者因高油价抑制燃油车购买意愿，终端销量进一步承压下滑。",
                impact_level="中高",
                duration="中期",
                key_drivers=["燃油乘用车销量同比下滑"],
            ),
            SectorImpact(
                sector="中下游塑料改性与化纤织造",
                transmission_logic="上游原料PX/PTA/PP树脂涨价，下游服装与塑料消费疲软难以接受提价，面临两头受压。",
                impact_level="中高",
                duration="中期",
                key_drivers=["原料-产品加工价差收窄", "开工率被迫下调"],
            ),
        ],
        cross_market_spillovers={
            "A股": "石油石化、煤炭、油服及煤化工板块逆势暴涨，航空、交运及中下游制造业承压震荡。",
            "港股/中概": "港股三桶油及海外能源股领涨，航空与消费股面临估值下修。",
            "大宗商品": "原油带动沥青、燃料油、PTA、甲醇等全能化大宗期货品种共振大涨。",
            "美债与外汇": "通胀预期推升美国长端美债收益率，推迟美联储降息窗口，美元指数走强。",
        },
        key_monitoring_indicators=["ICE布伦特原油与NYMEX WTI原油期货结算价", "EIA美国商业原油与汽油库存周报", "OPEC+产量合规率与配额决议", "VLCC油轮等价期租租金(TCE)"],
        historical_reference_cases=["2022年俄乌冲突爆发初期国际油价飙升至139美元/桶行情", "2007-2008年原油超级大牛市突破147美元历史高位行情"],
    ),

    "commodity_supercycle_metals": MacroEventScenario(
        event_id="commodity_supercycle_metals",
        event_name="工业金属(铜/铝)超级周期与价格暴涨",
        category="大宗商品与能源",
        aliases=["铜价暴涨", "铝价上涨", "工业金属超级周期", "大宗商品繁荣", "有色金属大涨", "铜博士暴涨", "资源为王"],
        description="受全球绿色能源转型(电网/新能源车/AI算力)、矿端资本开支长期不足及供需严重错配驱动，铜、铝等工业金属价格迎来超级大周期上涨。",
        transmission_mechanism=[
            "一级直接冲击：LME与沪铜、沪铝期货价格大幅飙涨，现货升水持续高企，上游采矿与冶炼企业利润爆发。",
            "二级产业链传导：电网电力线缆、家电空调铜管、汽车线束及新能源电池铜箔成本全面垫高；若终端无法顺价提价，中下游加工企业毛利大幅受挫，引发\"以铝节铜\"等替代替代方案加速落地。",
            "三级跨市场外溢：推升工业品出厂价格指数(PPI)，引发全球工业通胀预期，具备上游优质自备矿产资源的资源型企业获得极高重估溢价。",
        ],
        direct_impact=[
            "LME铜突破10000美元/吨以上高位，沪铝突破历史价格中枢",
            "国内有色金属现货升贴水剧烈波动",
            "工业企业原材料采购成本指数大幅上升",
            "全球主要金属交易所显性库存降至历史低位",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="高自给率铜矿采选龙头(如紫金矿业/洛阳钼业)",
                transmission_logic="自产矿成本稳定，铜价每上涨1000美元直接带来百亿级利润弹性释放，资源储量获得大幅价值重估。",
                impact_level="极高",
                duration="长期",
                key_drivers=["矿产铜产量", "自产矿比例(>80%)", "单位现金生产成本"],
            ),
            SectorImpact(
                sector="电解铝合规产能一体化龙头",
                transmission_logic="国内4500万吨产能天花板锁死供给上限，铝价上涨带来巨额吨铝盈利扩张，自备电厂/绿电优势凸显。",
                impact_level="极高",
                duration="长期",
                key_drivers=["吨铝毛利", "自备电力与氧化铝自给率"],
            ),
            SectorImpact(
                sector="废旧金属回收与再生铜铝循环利用",
                transmission_logic="高金属价格刺激废旧金属回收折价套利空间扩大，再生金属成本优势与政策扶持双重提升。",
                impact_level="高",
                duration="中期",
                key_drivers=["废旧铜铝回收利用量", "碳减排税收优惠"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="电线电缆与配电网设备中游加工",
                transmission_logic="铜材成本占电缆总成本80%以上，招投标合同多为固定单价或调价机制滞后，铜价暴涨导致大面积亏损。",
                impact_level="极高",
                duration="中期",
                key_drivers=["铜材成本占比高", "合同调价条款滞后"],
            ),
            SectorImpact(
                sector="白色家电(空调/冰箱)",
                transmission_logic="空调铜管用量大，铜铝原材料涨价直接推升整机BOM成本，压制内销毛利率。",
                impact_level="高",
                duration="中期",
                key_drivers=["单台空调耗铜量", "终端提价顺价阻力"],
            ),
            SectorImpact(
                sector="电子元器件与PCB印制电路板",
                transmission_logic="覆铜板(CCL)与电解铜箔原料涨价向下游PCB传导受阻，挤压中小型PCB板厂净利率。",
                impact_level="中高",
                duration="中期",
                key_drivers=["覆铜板采购价格", "PCB议价能力"],
            ),
        ],
        cross_market_spillovers={
            "A股": "有色金属采选、黄金及稀缺资源板块全面爆发成为市场主线，中下游加工与轻工制造业承压。",
            "港股/中概": "港股大型海外矿业巨头估值大幅拉升，高分红与资源属性吸引全球资金涌入。",
            "大宗商品": "带动锌、铅、锡、镍等其他基本工业金属全线共振上行。",
            "全球宏观": "提升全球制造业补库周期热度，同时加剧海外欧美制造业通胀粘性。",
        },
        key_monitoring_indicators=["LME与SHFE铜/铝期货主力合约价格", "全球交易所及保税区显性库存(LME/SHFE/COMEX库存)", "铜精矿现货粗炼/精炼加工费(TC/RC)", "中国电网年度投资建设进度"],
        historical_reference_cases=["2005-2007年全球工业化城镇化大宗超级牛市", "2020-2021年全球流动性泛滥与绿色能源转型共振引发的铜铝暴涨行情"],
    ),

    "gold_safe_haven_rally": MacroEventScenario(
        event_id="gold_safe_haven_rally",
        event_name="黄金与贵金属避险暴涨/实际利率下行",
        category="大宗商品与能源",
        aliases=["金价暴涨", "黄金大牛市", "贵金属暴涨", "去美元化避险", "央行购金", "实际利率下行", "黄金破历史新高"],
        description="受美国实际利率下行、全球央行去美元化持续增持黄金储备、主权债务信任危机或重大地缘政治动荡驱动，国际黄金价格持续暴涨并创历史新高。",
        transmission_mechanism=[
            "一级直接冲击：COMEX黄金期货与伦敦现货黄金(XAU/USD)价格持续攀升突破历史极值，国内沪金、金条与实物金零售价水涨船高。",
            "二级产业链传导：上游金矿采选企业单克毛利几何级数扩张，现金流极其充沛，展开全球优质金矿资产并购；下游黄金珠宝零售品牌毛利率因金价单边上涨实现存货增值，但过高金价可能阶段性抑制克重类首饰终端销量。",
            "三级跨市场外溢：全球资本对法币主权信用产生长期通胀与违约担忧，黄金ETF持仓持续增加，贵金属板块对全市场防御与避险资金产生强力虹吸效应。",
        ],
        direct_impact=[
            "伦敦现货金(XAU)与COMEX黄金期货持续大幅上行创历史新高",
            "国内实物黄金金条与品牌足金首饰克价大幅上涨",
            "全球主要央行外汇储备中黄金占比持续攀升",
            "全球避险情绪指数(VIX/地缘政治风险指数)处于高位",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="自产金矿采选龙头(如山东黄金/中金黄金/紫金矿业/赤峰黄金)",
                transmission_logic="每克黄金开采完全成本(AISC)相对固定，金价上涨部分扣除税费后近乎全额转化为税前净利润，业绩弹性巨大。",
                impact_level="极高",
                duration="长期",
                key_drivers=["矿产金年产量", "克金综合维持成本(AISC)", "金矿剩余可开采储量"],
            ),
            SectorImpact(
                sector="白银与伴生贵金属采选",
                transmission_logic="白银兼具货币金融属性与光伏工业属性，在金银比修复驱动下弹性往往超越黄金。",
                impact_level="极高",
                duration="中长期",
                key_drivers=["金银比价回归", "光伏银浆工业需求与金融属性共振"],
            ),
            SectorImpact(
                sector="黄金珠宝加盟连锁品牌(拥有大量黄金存货)",
                transmission_logic="低价黄金存货在金价上涨周期中享受巨额存货重估增值毛利，计价类与一口价古法黄金毛利高企。",
                impact_level="中高",
                duration="中期",
                key_drivers=["黄金存货规模", "加盟费与终端单店模型动销"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="光伏组件与导电银浆生产商",
                transmission_logic="白银价格暴涨导致光伏正银/背银浆料成本骤增，直接推高TOPCon/HJT电池片非硅BOM成本。",
                impact_level="高",
                duration="中期",
                key_drivers=["单瓦银浆消耗量", "银价上涨对度电成本挤压"],
            ),
            SectorImpact(
                sector="纯克重黄金首饰零售加工(缺乏定价权)",
                transmission_logic="金价短时间暴涨导致终端消费者产生恐慌观望情绪，传统克重首饰终端动销大幅失速。",
                impact_level="中",
                duration="短期",
                key_drivers=["终端黄金首饰零售量下滑", "加工费被压缩"],
            ),
        ],
        cross_market_spillovers={
            "A股": "黄金及贵金属采选股持续走出独立牛市，与大盘指数弱相关甚至负相关，成为确定性最高的避险防守进攻兼备品种。",
            "港股/美股": "全球黄金矿业巨头(如纽蒙特Newmont/巴里克Barrick)及港股黄金股获得全球主权基金增配。",
            "美债与美元": "若金价因实际利率下行而涨，通常伴随美债收益率下行；若因去美元化与地缘而涨，则可能与美元同涨。",
            "外汇汇率": "非美货币主权信用承压，全球央行加速将外汇储备多元化为实物黄金资产。",
        },
        key_monitoring_indicators=["伦敦金现(XAU/USD)与COMEX黄金期货", "美国10年期国债实际收益率(TIPS)", "世界黄金协会(WGC)全球央行购金季度数据", "全球最大黄金ETF(SPDR)持仓量", "金银比价(Gold/Silver Ratio)"],
        historical_reference_cases=["1970年代布雷顿森林体系解体后的大通胀黄金超级牛市", "2023-2024年全球地缘动荡与全球央行持续买金推动的黄金历史性大牛市"],
    ),

    "geopolitical_conflict_escalation": MacroEventScenario(
        event_id="geopolitical_conflict_escalation",
        event_name="国际地缘冲突与海峡航道受阻",
        category="地缘与国际贸易",
        aliases=["地缘冲突", "局部战争", "红海危机", "霍尔木兹海峡封锁", "海峡航运受阻", "地缘局势恶化", "战时经济"],
        description="关键地缘政治热点区域(中东/东欧/台海/红海/马六甲)爆发军事冲突或航运封锁，导致全球关键大宗供应链与国际海运航道中断。",
        transmission_mechanism=[
            "一级直接冲击：国际油轮/集装箱货船被迫绕行好望角或停航，海运运距与航行时间大幅拉长，国际原油、天然气及海运保险费率瞬间飙升。",
            "二级产业链传导：全球集装箱与油运有效运力大幅抽紧，海运运价指数(SCFI/BDI/TCE)暴涨；下游外贸企业面临货物交付延期、海运费暴增及货柜短缺困境；军工国防订单迫切性与军费预算全面调高。",
            "三级跨市场外溢：全球资本市场避险情绪瞬间达到峰值，风险资产遭抛售，黄金、原油、军工、航运逆势暴涨，跨国供应链加速转向区域化与本土化备份。",
        ],
        direct_impact=[
            "国际海运即期运价与船舶战争险费率跳涨数倍",
            "布伦特原油与欧洲天然气期货风险溢价飙升",
            "全球避险资产(黄金/美元/瑞士法郎)大幅跳空高开",
            "全球股市主要指数单日面临避险抛压",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="国际远洋集运与油运航运龙头(如中远海控/中远海能)",
                transmission_logic="船舶绕行好望角使全球航运周转效率大幅下降，运力供给出现人造结构性短缺，海运费与TCE暴涨带来暴利。",
                impact_level="极高",
                duration="中期",
                key_drivers=["绕行增加的吨海里需求", "SCFI集运运价指数", "VLCC油轮运价"],
            ),
            SectorImpact(
                sector="国防军工与武器装备(导弹/无人机/特种材料)",
                transmission_logic="地缘冲突加剧催化国防安全备战紧迫感，军方加大实弹演练与武器装备消耗采购，军贸出口订单激增。",
                impact_level="高",
                duration="长期",
                key_drivers=["国家军费预算增长", "军工在手合同负债", "实弹与无人装备消耗采购"],
            ),
            SectorImpact(
                sector="油气开采与战略大宗商品储备",
                transmission_logic="能源与大宗断供恐慌推高大宗现货价格，国内资源型央企保障能源安全战略价值凸显。",
                impact_level="高",
                duration="中期",
                key_drivers=["原油风险溢价", "战略物资收储"],
            ),
            SectorImpact(
                sector="跨境陆路物流(中欧班列)",
                transmission_logic="海运时效恶化促使高货值电子产品与汽车零部件货主转投中欧班列陆路运输，班列舱位量价齐升。",
                impact_level="中高",
                duration="中期",
                key_drivers=["中欧班列发车列数与订舱运价"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="跨境电商与低货值轻工外贸出口商",
                transmission_logic="海运集装箱运费翻倍直接侵蚀产品利润率，甚至导致运费高于货值，海外买家推迟提货。",
                impact_level="极高",
                duration="中期",
                key_drivers=["海运费占货值比例", "海外港口到货延误与退货率"],
            ),
            SectorImpact(
                sector="民航国际客运与跨国旅游",
                transmission_logic="领空关闭与航线绕飞推高航油能耗与机组飞行成本，旅客出国旅游与商务出行意愿暴跌。",
                impact_level="高",
                duration="中期",
                key_drivers=["国际航线客座率下滑", "绕飞燃油成本增加"],
            ),
        ],
        cross_market_spillovers={
            "A股": "航运、油运、军工、贵金属等战争避险受益板块领涨，外贸出口及整体大盘估值面临避险波动。",
            "港股/美股": "全球主要股指大幅震荡，国防军工巨头(洛克希德/雷神)及能源巨头跑赢大盘。",
            "大宗商品": "原油、天然气、黄金、白银、化肥等受地缘供给约束的大宗品种全面暴涨。",
            "外汇汇率": "美元指数受避险资金追捧走强，处于冲突漩涡周边的区域性货币面临贬值压力。",
        },
        key_monitoring_indicators=["红海/苏伊士运河/霍尔木兹海峡单日通航船舶数量", "SCFI与CCFI出口集装箱运价指数", "波罗的海原油运输指数(BDTI)", "地缘政治风险指数(GPR Index)"],
        historical_reference_cases=["2023年底红海危机导致全球集运价格暴涨数倍行情", "2022年俄乌冲突爆发引发的全球能源与化肥供应链震荡行情"],
    ),

    "export_tariffs_trade_friction": MacroEventScenario(
        event_id="export_tariffs_trade_friction",
        event_name="全球贸易摩擦与海外关税加征",
        category="地缘与国际贸易",
        aliases=["加征关税", "贸易摩擦", "贸易战", "反倾销调查", "实体清单", "出口管制", "301调查", "关税壁垒", "惩罚性关税", "海外关税", "关税制裁"],
        description="欧美等海外主要经济体对中国出口的高附加值产品(如新能源汽车、光伏、锂电池、钢铝制品、消费电子)加征高额惩罚性关税或设置技术贸易壁垒。",
        transmission_mechanism=[
            "一级直接冲击：目标国进口关税税率大幅调高，中国出口产品在当地市场的到岸含税售价大幅上涨，直接削弱终端价格竞争力。",
            "二级产业链传导：外销型企业海外订单流失或被迫降价承担部分关税，倒逼龙头企业加速推进海外本土化产能布局(如赴东南亚、欧洲、墨西哥、中东建厂)；国内供应链加速国产替代与自主可控。",
            "三级跨市场外溢：引发全球供应链重构与跨国贸易转移，国内出口链估值折价，而自主可控、国产替代及具备全球本地化交付能力的跨国龙头享有估值溢价。",
        ],
        direct_impact=[
            "相关行业出口至目标国的集装箱货量短期骤降",
            "外销产品综合毛利率受关税分摊直接挤压",
            "企业海外直接投资(FDI)与海外设厂资本开支加速",
            "国内自主可控信创与供应链安全政策强化",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="半导体自主可控与国产设备材料",
                transmission_logic="外部技术封锁与贸易壁垒倒逼国内芯片设计、晶圆制造与终端品牌全面采购国产半导体零部件与设备。",
                impact_level="极高",
                duration="长期",
                key_drivers=["关键零部件国产化率考核", "国家产业基金扶持"],
            ),
            SectorImpact(
                sector="具备全球全球化产能布局的跨国制造龙头",
                transmission_logic="在匈牙利、墨西哥、越南已具备量产工厂的企业能够有效规避关税壁垒，抢占无海外布局同行的市场份额。",
                impact_level="高",
                duration="中长期",
                key_drivers=["海外本土化产能占比", "全球多基地交付能力"],
            ),
            SectorImpact(
                sector="国内纯内需与必选消费(白酒/大众食品/医药)",
                transmission_logic="业务完全依赖国内大循环，完全免疫海外贸易关税摩擦，在贸易摩擦期成为资金避风港。",
                impact_level="中高",
                duration="中期",
                key_drivers=["内需消费占比100%", "无海外关税风险敞口"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="纯国内制造对美/对欧出口单一依赖型企业",
                transmission_logic="无海外工厂且利润率微薄，高额关税完全封死其出口通路，面临订单骤降与产线闲置危机。",
                impact_level="极高",
                duration="中长期",
                key_drivers=["对美欧出口收入占比", "缺乏海外建厂资金实力"],
            ),
            SectorImpact(
                sector="光伏与电池组件(涉双反与反规避调查)",
                transmission_logic="海外关税与原产地穿透合规审查导致海外高毛利市场准入受限，产能被迫回流国内加剧内卷价格战。",
                impact_level="高",
                duration="中期",
                key_drivers=["海外涉案产品关税税率", "产能过剩内卷恶化"],
            ),
        ],
        cross_market_spillovers={
            "A股": "出口依赖板块面临估值下修与业绩下调，自主可控(芯片/工业母机/军工)与内需防御板块受青睐。",
            "港股/中概": "受国际贸易局势情绪压制，外资对涉外出口制造业风险溢价要求提高。",
            "大宗商品": "若贸易摩擦波及农产品(大豆/玉米)或能源，对应进口大宗品种国内期货价格易出现脉冲式上涨。",
            "外汇汇率": "市场预期出口结汇减少，人民币汇率可能在关税落地初期面临阶段性贬值压力以对冲关税成本。",
        },
        key_monitoring_indicators=["海关总署月度主要商品分国别出口金额同比", "美国USTR与欧盟贸易救济调查裁决公告", "中国企业对外直接投资(ODI)增速", "涉案企业海外建厂投产时间表"],
        historical_reference_cases=["2018-2019年中美贸易战301关税与实体清单行情", "2024年美欧对中国电动汽车、光伏与锂电池加征关税调查"],
    ),

    "cpi_ppi_scissors_widening": MacroEventScenario(
        event_id="cpi_ppi_scissors_widening",
        event_name="CPI-PPI剪刀差走阔与利润格局重塑",
        category="通胀与宏观周期",
        aliases=["剪刀差", "CPI-PPI剪刀差", "成本型通胀", "PPI上行CPI低迷", "中下游利润挤压", "通胀剪刀差"],
        description="PPI(生产资料价格指数)大幅走高而CPI(居民消费价格指数)低迷，形成\"上游暴利、中下游严重受挤压\"的典型成本通胀剪刀差格局。",
        transmission_mechanism=[
            "一级直接冲击：上游大宗原材料(煤炭、钢铁、有色、原油、基础化工)出厂价格全面大涨，中下游制造业采购成本激增。",
            "二级产业链传导：由于下游终端居民消费需求偏弱，中下游制造与消费品企业无法顺价提价，巨额成本上涨完全由中游加工与下游消费企业自行承担，导致其毛利率断崖式下滑；上游资源与原材料行业利润暴增。",
            "三级跨市场外溢：宏观经济显现滞胀特征，全市场盈利向少数上游周期资源板块高度集中，制造业资本开支意愿受到严重压制。",
        ],
        direct_impact=[
            "PPI同比增速快速攀升至高位(如>8%)",
            "CPI同比增速维持在低位震荡(如<1.5%)",
            "工业企业利润总额向上游采矿业与原材料制造业极端集中",
            "中下游制造企业采购经理人指数(PMI)购进价格与出厂价格差距拉大",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="上游采矿与原材料采选(煤炭/有色采矿/油气/磷矿)",
                transmission_logic="产品直接对应PPI指数成分，享有大宗商品价格暴涨带来的全部增量利润，ROE大幅飙升。",
                impact_level="极高",
                duration="中长期",
                key_drivers=["PPI同比高位运行", "上游资源自给率100%"],
            ),
            SectorImpact(
                sector="拥有极强品牌壁垒的高端消费(高端白酒)",
                transmission_logic="毛利率极高(80%+)且对原材料涨价极不敏感，完全免疫PPI上涨侵蚀，保持利润稳定性。",
                impact_level="高",
                duration="中长期",
                key_drivers=["超高毛利率", "成本占比微乎其微"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="中游通用机械与汽车零部件制造",
                transmission_logic="钢铝塑料等原材料成本刚性暴涨，下游车企要求降价，中游制造夹在中间两头受气，毛利暴跌。",
                impact_level="极高",
                duration="中期",
                key_drivers=["原材料成本占BOM比重高", "缺乏对上下游议价权"],
            ),
            SectorImpact(
                sector="白色家电与消费电子组装",
                transmission_logic="铜材、塑料和面板成本大涨，终端消费疲软不敢提价，单机净利润被侵蚀殆尽。",
                impact_level="高",
                duration="中期",
                key_drivers=["终端消费降价促销", "原材料占成本超60%"],
            ),
            SectorImpact(
                sector="大众包装食品与调味品",
                transmission_logic="农产品原粮与包装包材(PET/纸箱)涨价，提价周期滞后半年以上，业绩出现阶段性阵痛。",
                impact_level="中高",
                duration="中期",
                key_drivers=["原材料成本滞后传导", "渠道动销压力"],
            ),
        ],
        cross_market_spillovers={
            "A股": "市场呈现极端的\"周期独大\"结构性行情，煤炭、有色、化纤一骑绝尘，制造业与大消费普遍阴跌。",
            "港股/中概": "上游资源类央企大涨，科技互联与制造代工受成本及消费拖累承压。",
            "大宗商品": "黑色系、有色系、能化系大宗商品期货全线处于贴水修复或现货升水结构。",
            "外汇汇率": "进口大宗付汇需求激增可能导致贸易顺差收窄，对汇率带来阶段性贬值压力。",
        },
        key_monitoring_indicators=["国家统计局月度CPI与PPI同比剪刀差值", "规模以上工业企业利润分行业增速(采矿业 vs 制造业)", "PMI原材料购进价格指数 vs 出厂价格指数", "上游大宗工业品综合价格指数"],
        historical_reference_cases=["2021年PPI高达13.5%而CPI仅1.5%的\"超级周期大年\"行情", "2017年供给侧结构性改革期间上游原材料暴利行情"],
    ),

    "deflation_demand_contraction": MacroEventScenario(
        event_id="deflation_demand_contraction",
        event_name="国内有效需求不足与通缩压力",
        category="通胀与宏观周期",
        aliases=["通缩", "需求不足", "资产负债表衰退", "居民去杠杆", "CPI低迷", "物价下行", "流动性陷阱"],
        description="受房地产深度调整、居民收入预期走弱及地方化债约束影响，全社会总需求不足，CPI/PPI持续低迷甚至负增长，企业陷入以价换量恶性竞争。",
        transmission_mechanism=[
            "一级直接冲击：消费品与工业品价格中枢持续下移，全社会通胀预期降至冰点，消费者倾向于推迟大宗消费与耐用品采购。",
            "二级产业链传导：各行业产能过剩矛盾加剧，企业为了维持现金流与开工率大打价格战，行业综合毛利率普遍被拉低；实体有效信贷需求萎缩，资金沉淀在银行体系形成\"流动性陷阱\"。",
            "三级跨市场外溢：权益市场整体估值承压，企业名义盈利增速下滑，长端国债收益率持续刷新历史新低，高股息、稳定现金流的红利类资产获得全市场绝对超额收益。",
        ],
        direct_impact=[
            "CPI与核心CPI同比长期在0%附近低位徘徊甚至为负",
            "PPI同比持续处于负增长区间(工业领域深度通缩)",
            "10年期国债收益率持续下行破位创历史新低",
            "居民储蓄率高企而新增中长期贷款意愿低迷",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="高股息红利公用事业(大水电/核电/大型煤电)",
                transmission_logic="拥有特许经营垄断壁垒与极稳定充沛现金流，股息率(5%~7%)远超长端国债利率，成为全社会资金配置避风港。",
                impact_level="极高",
                duration="长期",
                key_drivers=["股息率与国债利差走阔", "现金流不受宏观消费影响"],
            ),
            SectorImpact(
                sector="平价消费与硬折扣零售(零食量贩/平价连锁)",
                transmission_logic="消费者追求极致性价比与平替消费，硬折扣与量贩零食渠道凭借供应链效率逆势高速拓店。",
                impact_level="高",
                duration="中长期",
                key_drivers=["同店销售额增速", "单店投资回收期", "平替消费渗透率"],
            ),
            SectorImpact(
                sector="全球化出海龙头(具备海外独立造血能力)",
                transmission_logic="海外市场享有正常通胀与合理利润率，海外高毛利业务有效平抑国内内卷降价压力。",
                impact_level="高",
                duration="长期",
                key_drivers=["海外业务收入占比与毛利率稳定性"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="可选大宗耐用品(传统燃油车/高端家电/精装地产链)",
                transmission_logic="消费者推迟换车、换房与大额家电采购，需求断崖式下跌引发全行业恶性价格踩踏。",
                impact_level="极高",
                duration="中长期",
                key_drivers=["终端零售单价持续下跌", "经销商库存爆仓"],
            ),
            SectorImpact(
                sector="次高端及大众餐饮商务消费(次高端白酒/中高端餐饮)",
                transmission_logic="政商务宴请与居民大额聚餐频次骤降，渠道库存积压严重，批价持续倒挂。",
                impact_level="高",
                duration="中长期",
                key_drivers=["商务宴请动销萎缩", "产品批价倒挂"],
            ),
            SectorImpact(
                sector="商业银行(资产荒冲击)",
                transmission_logic="优质实体信贷需求极度匮乏，贷款利率不断下行以争夺有限优质客户，净息差被动持续收窄。",
                impact_level="高",
                duration="长期",
                key_drivers=["有效信贷需求不足", "净息差下行突破警戒线"],
            ),
        ],
        cross_market_spillovers={
            "A股": "红利低波策略与公用事业走出长期独立大牛市，顺周期、核心资产与成长板块面临漫长估值消化。",
            "港股/中概": "高股息央企红利股受南向资金持续抢筹，消费与科技龙头估值被压制在历史低位。",
            "大宗商品": "国内基建与地产大宗(螺纹钢/水泥/玻璃/焦炭)价格持续低迷，呈现供大于求格局。",
            "债券市场": "超长期国债(30年/50年特别国债)迎来史诗级超级大牛市，收益率屡创新低。",
        },
        key_monitoring_indicators=["全国CPI与PPI月度同比及环比数据", "中国10年期与30年期国债到期收益率", "居民部门中长期贷款月度新增额", "M1与M2剪刀差(M1-M2增速差)"],
        historical_reference_cases=["日本1990年代资产负债表衰退与失去的三十年初期行情", "2023-2024年国内有效需求不足背景下的高股息红利大牛市"],
    ),

    "us_fed_rate_cut_cycle": MacroEventScenario(
        event_id="us_fed_rate_cut_cycle",
        event_name="美联储降息周期与全球流动性外溢",
        category="货币与流动性",
        aliases=["美联储降息", "Fed降息", "美元流动性外溢", "全球降息潮", "美元指数下行", "外资回流新兴市场"],
        description="美联储开启降息周期，连续下调联邦基金目标利率区间，美元指数和美债收益率趋势性回落，全球流动性环境显著宽松。",
        transmission_mechanism=[
            "一级直接冲击：美国10年期国债收益率与美元指数(DXY)下行，中美利差倒挂幅度收窄，打开国内央行货币政策宽松操作空间。",
            "二级产业链传导：全球风险资产借贷成本下降，海外Biotech医药、算力AI及跨国高科技初创企业融资环境大幅回暖，跨国研发外包(CXO)新签订单触底反弹。",
            "三级跨市场外溢：全球套息交易平仓与外资配置资金加速回流新兴市场与中国资产，港股及A股高弹性科技成长股迎来估值与流动性双重提振。",
        ],
        direct_impact=[
            "美国联邦基金利率下调，美债长短端收益率下行",
            "美元指数(DXY)走弱下跌",
            "中美10年期国债利差倒挂显著收窄",
            "人民币汇率被动升值企稳，资本外流压力解除",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="港股恒生科技与互联网中概龙头",
                transmission_logic="港股对美元流动性极其敏感，外资回流与折现率下行直接催生科技互联网估值大级别修复。",
                impact_level="极高",
                duration="中长期",
                key_drivers=["美债10年期收益率下行", "外资配置资金净流入港股"],
            ),
            SectorImpact(
                sector="医药外包CXO与创新药Biotech",
                transmission_logic="海外生物医药一级投融资景气度与美联储利率高度负相关，降息直接刺激海外大药企与Biotech加大研发订单投放。",
                impact_level="极高",
                duration="中长期",
                key_drivers=["全球医疗健康领域月度投融资额", "海外客户新签订单增速"],
            ),
            SectorImpact(
                sector="黄金与贵金属采选",
                transmission_logic="美国实际利率(TIPS)下行直接推高无息资产黄金的吸引力，驱动金价继续上行。",
                impact_level="高",
                duration="中长期",
                key_drivers=["美债实际收益率走低", "美元指数下行"],
            ),
            SectorImpact(
                sector="A股高估值成长股(芯片/光模块/机器人)",
                transmission_logic="国内宽松空间打开叠加风险偏好回暖，成长板块估值弹性全面释放。",
                impact_level="高",
                duration="中期",
                key_drivers=["分母端估值折现率下行", "机构风险偏好提升"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="纯美元现金存款管理类产品",
                transmission_logic="美元高息存款与美元理财收益率快速下滑，高息吸储吸引力不再。",
                impact_level="中",
                duration="中长期",
                key_drivers=["美元存款利率下调"],
            ),
        ],
        cross_market_spillovers={
            "A股": "成长风格占优，医药生物、电子半导体、电力设备等高弹性赛道迎来大级别反弹。",
            "港股/中概": "港股恒生指数与恒生科技指数弹性显著优于A股，呈现流动性推动的大牛市。",
            "大宗商品": "美元走弱支撑大宗商品计价，铜、原油、黄金等以美元计价的大宗普遍获益。",
            "美股与美债": "美债全线走牛，美股大盘在软着陆预期下走高，防御性资产跑输成长科技。",
        },
        key_monitoring_indicators=["美联储FOMC利率决议与点阵图(Dot Plot)", "美国10年期国债到期收益率", "美元指数(DXY)", "美联储降息预期概率(CME FedWatch Tool)"],
        historical_reference_cases=["2019年下半年美联储预防式降息推动的全球半导体与医药科技牛市", "2024年9月美联储超预期降息50BP引发的全球风险资产狂欢行情"],
    ),

    "real_estate_policy_easing": MacroEventScenario(
        event_id="real_estate_policy_easing",
        event_name="房地产重大支持政策与化险收储",
        category="财政与基建",
        aliases=["地产松绑", "取消限购", "降低首付", "房贷利率下调", "保交楼", "地产白名单", "收储保障房", "城中村改造"],
        description="国家出台重量级房地产支持政策，包括全面取消限购限售、大幅降低首付比例与房贷利率、设立保障房收购再贷款、推进城中村改造与房企融资白名单。",
        transmission_mechanism=[
            "一级直接冲击：购房者首付门槛与月供利息支出骤降，房企合规项目获得银行信贷白名单支持，地方国企获专项资金收购存量商品房用作保障房。",
            "二级产业链传导：核心城市二手房与新房带看量、成交量脉冲式回暖，房企销售回款与现金流压力缓解；带动防水材料、建筑涂料、水泥、陶瓷、电梯、五金等后周期产业链需求企稳。",
            "三级跨市场外溢：消除金融系统对房地产引发系统性金融风险的极端悲观预期，银行与保险资产质量担忧大幅纾解，带动大金融与顺周期产业链估值修复。",
        ],
        direct_impact=[
            "全国重点城市商品房首付比例降至历史最低(如15%)",
            "存量与新增房贷利率大幅调降",
            "房企项目白名单贷款审批发放额突破数万亿",
            "高能级城市一二级市场商品房周度成交面积脉冲反弹",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="高信用央国企优质房企(如保利发展/招商蛇口/华润置地)",
                transmission_logic="在行业洗牌出清后享有极高的市占率提升空间与政策红利，低成本拿地与销售去化优势确立。",
                impact_level="极高",
                duration="中长期",
                key_drivers=["核心城市优质土储占比", "销售回款与融资成本优势"],
            ),
            SectorImpact(
                sector="地产后周期消费建材(防水/涂料/管材/五金/电梯)",
                transmission_logic="保交房交付与保障房装修拉动建材集采需求，应收账款坏账风险显著降低，现金流改善。",
                impact_level="高",
                duration="中期",
                key_drivers=["房企供应链回款改善", "建材出货量回暖"],
            ),
            SectorImpact(
                sector="家用电器与智能家居(厨电/中央空调/白电)",
                transmission_logic="商品房竣工交付与存量二手房交易活跃直接拉动家电新装与换装需求。",
                impact_level="高",
                duration="中期",
                key_drivers=["精装房交付率", "以旧换新置换需求"],
            ),
            SectorImpact(
                sector="商业银行(尤其是涉房贷款敞口大行)",
                transmission_logic="对公房地产贷款及个人按揭断供违约的系统性风险被政策托底化解，资产质量预期大幅修复。",
                impact_level="高",
                duration="中期",
                key_drivers=["房地产不良贷款生成率见顶回落", "估值PB修复"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="出险且无自救能力的严重资不抵债中小房企",
                transmission_logic="政策支持严格坚持\"保项目不保主体\"与市场化原则，壳资源与僵尸房企加速退市出清。",
                impact_level="高",
                duration="长期",
                key_drivers=["债务重组失败", "触及财务或面值退市"],
            ),
        ],
        cross_market_spillovers={
            "A股": "房地产、建材、大金融(银行/保险)迎来估值修复报复性反弹，顺周期板块领涨大盘。",
            "港股/中概": "港股内房股与物管股高弹性暴涨，中资美元离岸地产债二级市场价格大幅反弹。",
            "大宗商品": "黑色系(螺纹钢/铁矿石/玻璃/焦炭)期货价格受预期提振大幅反弹。",
            "债券市场": "经济预期向好推升风险偏好，债市长端收益率面临阶段性回调压力。",
        },
        key_monitoring_indicators=["30大中城市商品房每日网签成交面积", "百强房企单月全口径销售金额同比", "全国商品房住宅待售面积与库存去化周期", "房企融资白名单贷款投放总额"],
        historical_reference_cases=["2014-2015年\"930\"与\"330\"房地产重磅松绑大牛市", "2024年\"517\"与\"924\"中央政治局会议促止跌回稳系列组合拳行情"],
    ),

    "ai_tech_revolution_breakthrough": MacroEventScenario(
        event_id="ai_tech_revolution_breakthrough",
        event_name="颠覆性技术突破与产业智能化爆发",
        category="产业技术突破",
        aliases=["AI突破", "大模型升级", "AGI", "具身智能", "人形机器人", "算力爆发", "科技革命", "AI+应用"],
        description="全球顶级人工智能实验室发布代际跃迁的颠覆性大模型(具备深度推理、多模态、Agent自主行动或具身智能突破)，引发全球新一轮科技军备竞赛与硬件重构。",
        transmission_mechanism=[
            "一级直接冲击：全球科技云巨头与主权国家大幅上调未来数年AI算力资本开支(Capex)预算，先进制程算力卡、高速光互联及先进制程产能被抢购一空。",
            "二级产业链传导：算力硬件层(GPU/HBM/800G与1.6T光模块/液冷服务器/PCB/高速铜缆)排产爆满、单价与毛利齐升；下游人形机器人、自动驾驶、AI手机/PC、企业级AI智能体应用全面加速商用落地。",
            "三级跨市场外溢：全球资本市场风险偏好飙升，科技成长板块享有极高估值溢价，全社会生产力重构预期强烈，传统低效劳动力依赖型行业面临自动化替代变革。",
        ],
        direct_impact=[
            "北美与国内科技巨头资本开支(Capex)指引大幅上修",
            "顶级AI芯片与高速光模块交期拉长、供不应求",
            "全球科技指数(纳斯达克/恒生科技/创业板指)风险偏好飙升",
            "算力数据中心绿电能耗需求激增",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="高速光模块与光通信器件(800G/1.6T/3.2T)",
                transmission_logic="大模型参数规模与集群算力扩展对网络带宽提出指数级要求，光模块出货量与产品单价同步爆发。",
                impact_level="极高",
                duration="长期",
                key_drivers=["北美云巨头采购指引", "高速率产品良品率与出货占比"],
            ),
            SectorImpact(
                sector="AI芯片与先进封装(HBM/CoWoS/Chiplet)",
                transmission_logic="算力基础设施核心底座，享受最高的技术护城河与定价权，业绩呈爆发式增长。",
                impact_level="极高",
                duration="长期",
                key_drivers=["算力芯片出货量", "先进封装产能利用率"],
            ),
            SectorImpact(
                sector="服务器PCB、高速铜缆与液冷散热",
                transmission_logic="单台AI服务器价值量较通用服务器提升数倍，高多层PCB与液冷渗透率快速提升。",
                impact_level="高",
                duration="中长期",
                key_drivers=["液冷渗透率突破", "高频高速材料用量翻倍"],
            ),
            SectorImpact(
                sector="人形机器人与具身智能核心零部件(减速器/丝杠/传感器)",
                transmission_logic="多模态大模型赋予机器人\"大脑\"，人形机器人产业化进程提前，核心零部件量产订单呼之欲出。",
                impact_level="高",
                duration="长期",
                key_drivers=["机器人量产订单落地", "核心零部件单机价值量"],
            ),
            SectorImpact(
                sector="绿色电力与电网配套(绿电/核电/储能)",
                transmission_logic="AI的尽头是能源，巨型数据中心集群对稳定基荷电力(核电/绿电)产生庞大新增消纳需求。",
                impact_level="中高",
                duration="长期",
                key_drivers=["数据中心直供电订单", "绿电溢价采购"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="传统低附加值初级软件外包与基础代码搬运",
                transmission_logic="AI编程与代码自动生成技术大幅降低开发门槛，传统人头外包模式面临颠覆性降本压力。",
                impact_level="高",
                duration="长期",
                key_drivers=["人头单价下降", "客户自研AI工具替代外包"],
            ),
            SectorImpact(
                sector="传统通用低端服务器与低速网络设备",
                transmission_logic="企业IT预算全面向AI算力倾斜，挤占通用计算与传统存储采购预算。",
                impact_level="中",
                duration="中期",
                key_drivers=["通用服务器出货量下滑"],
            ),
        ],
        cross_market_spillovers={
            "A股": "TMT板块(通信/电子/计算机/传媒)与机器人概念掀起波澜壮阔的科技主线大牛市。",
            "港股/美股": "英伟达、微软、台积电及港股大型科技股带动全球股指创历史新高。",
            "大宗商品": "数据中心与电网建设拉动工业金属铜、铝需求，芯片制造拉动贵金属金、稀有气体等原材料消耗。",
            "全球宏观": "推动全要素生产率(TFP)长期中枢提升，重塑全球科技产业链分工格局。",
        },
        key_monitoring_indicators=["北美四大云厂商(微软/亚马逊/谷歌/Meta)季度Capex增速", "英伟达/台积电月度及季度营收指引", "高速光模块月度出货统计", "国内外通用大模型推理性能Benchmark排行榜"],
        historical_reference_cases=["2022年底ChatGPT横空出世引爆的全球AI算力浪潮", "2024年初Sora及GPT-4o发布催化的多模态与端侧AI热潮"],
    ),

    "capital_market_institutional_reform": MacroEventScenario(
        event_id="capital_market_institutional_reform",
        event_name="资本市场制度重大改革与生态重塑",
        category="资本市场制度",
        aliases=["新国九条", "分红新规", "退市常态化", "严厉打击造假", "并购重组", "市值管理", "长钱长投", "资本市场改革"],
        description="国务院与证监会发布资本市场重大顶层设计改革文件(如新\"国九条\")，严把上市入口关、畅通多元退市通道、强制提升现金分红比例、强化上市公司市值管理考核并鼓励产业并购重组。",
        transmission_mechanism=[
            "一级直接冲击：IPO与再融资审核标准大幅提高、节奏受控放缓，违规减持与财务造假处罚力度空前，常态化退市节奏加快。",
            "二级产业链传导：上市公司大幅提高年度现金分红率与股份回购注销力度，高股息资产吸引力显著提升；头部优质企业通过并购重组加速吸收合并同行业优质资产，实现做大做强；微盘股与垃圾股炒作受到强力遏制。",
            "三级跨市场外溢：A股市场投资生态发生深刻变革，从\"炒小炒差炒壳\"彻底转向\"注重基本面、看重现金回报与高分红\"的价值投资时代，中长期社保、年金、保险长线资金加速入市。",
        ],
        direct_impact=[
            "IPO过会率与新股发行家数大幅精简",
            "A股上市公司现金分红总额与分红家数创历史新高",
            "触发财务类/规范类/面值类退市公司数量大幅增加",
            "中证红利指数与高股息央企估值中枢持续抬升",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="高分红中央企业与地方国企(央企红利龙头)",
                transmission_logic="市值管理纳入央企负责人考核，央企分红意愿与分红能力最强，享受长线险资和公募长期增配溢价。",
                impact_level="极高",
                duration="长期",
                key_drivers=["股息率(>4%)", "现金分红比例(>50%)", "破净修复需求"],
            ),
            SectorImpact(
                sector="行业龙头与具备并购重组能力的平台型企业",
                transmission_logic="在监管支持硬科技与产业链协同并购重组背景下，龙头企业低成本兼并整合优质标的加速成长。",
                impact_level="高",
                duration="中长期",
                key_drivers=["在手现金储备", "产业链横向/纵向整合案例"],
            ),
            SectorImpact(
                sector="头部综合券商与专业中介机构",
                transmission_logic="并购重组与市值管理财务顾问业务迎来黄金期，投行业务向合规风控严密的大券商集中。",
                impact_level="中高",
                duration="长期",
                key_drivers=["并购重组业务收入占比", "保荐合规评级"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="微盘垃圾股与亏损壳资源公司",
                transmission_logic="退市新规全面堵死借壳炒作与财务保壳漏洞，缺乏基本面支撑的小微盘股面临流动性折价与退市清盘风险。",
                impact_level="极高",
                duration="长期",
                key_drivers=["扣非净利润连续亏损", "市值低于退市红线"],
            ),
            SectorImpact(
                sector="单纯依赖IPO保荐通道业务的中小券商",
                transmission_logic="IPO发行节奏阶段性收紧，项目储备不足的中小券商投行收入断崖式下跌。",
                impact_level="高",
                duration="中期",
                key_drivers=["IPO承销收入占比过高"],
            ),
        ],
        cross_market_spillovers={
            "A股": "市场生态两极分化，优质大盘蓝筹、高股息央企与科技龙头领跑，微盘股指数大幅波动出清。",
            "港股/中概": "境内外互联互通机制持续深化，高分红港股通标的受南向险资与散户资金抢筹。",
            "长线资金": "险资、社保基金、养老金权益投资比例与考核周期拉长，市场长期波动率趋于平缓。",
        },
        key_monitoring_indicators=["A股上市公司年度累计现金分红总额", "全市场年内退市公司总家数", "并购重组重大资产重组预案发布数量", "中证红利指数相对全A超额收益"],
        historical_reference_cases=["2004年与2014年两次\"国九条\"出台后开启的历史性制度红利行情", "2024年4月新\"国九条\"发布后高股息与绩优龙头长期领涨行情"],
    ),

    "extreme_weather_power_curtailment": MacroEventScenario(
        event_id="extreme_weather_power_curtailment",
        event_name="极端气候与高温/干旱限电限产",
        category="大宗商品与能源",
        aliases=["限电", "高温限电", "干旱限水", "拉尼娜", "厄尔尼诺", "限电限产", "枯水期", "能耗双控限产"],
        description="受厄尔尼诺/拉尼娜极端气候影响，夏季出现持续极端高温或干旱少雨，导致水电大省(四川/云南/湖北)发电量骤减同时全社会降温用电负荷破历史极值，引发区域性工业用电限电限产。",
        transmission_mechanism=[
            "一级直接冲击：水电发电量断崖式下滑，电网用电负荷缺口扩大，省内启动有序用电预案，高耗能工业企业(电解铝/工业硅/黄磷/锂盐/水泥)被迫停产或降负荷运行。",
            "二级产业链传导：局部供给骤减导致电解铝、工业硅、多晶硅、黄磷等原材料现货价格跳涨；火电开工率打满，动力煤迎峰度夏日耗见顶，储能与虚拟电厂调度需求激增；中下游组装加工企业面临断料风险与发电机租赁成本增加。",
            "三级跨市场外溢：强化对新型电力系统建设、电网跨区域特高压调度通道及抽水蓄能/工商业储能建设的紧迫感与政策倾斜力度。",
        ],
        direct_impact=[
            "主要流域入库水流量同比大幅减少50%以上",
            "电网区域性最大负荷屡创新高，启动工业企业错峰限电",
            "电解铝/工业硅等高耗能原材料现货价格短期脉冲上涨",
            "火电单日耗煤量与电厂日耗达到夏季峰值",
        ],
        beneficiary_sectors=[
            SectorImpact(
                sector="火电调峰与煤电一体化企业",
                transmission_logic="在水电出力严重不足时，火电机组作为基荷与顶峰主力全负荷满发，发电小时数与容量电费收益大增。",
                impact_level="极高",
                duration="短期",
                key_drivers=["火电利用小时数大幅上升", "现货交易电价顶格上浮"],
            ),
            SectorImpact(
                sector="高耗能大宗原材料不受限电影响的区域龙头(如电解铝/工业硅)",
                transmission_logic="限电导致行业总供给收缩价格暴涨，未受限电限产影响的省外企业享受价增量稳的超额暴利。",
                impact_level="高",
                duration="短期",
                key_drivers=["现货涨价幅度", "自备绿电不受电网限制"],
            ),
            SectorImpact(
                sector="虚拟电厂、工商业储能与柴油发电机组",
                transmission_logic="工业企业为了保供生产自购储能与备用发电机，虚拟电厂聚合商获得丰厚需求侧响应补贴。",
                impact_level="高",
                duration="中短期",
                key_drivers=["需求侧响应补贴电价", "储能系统出货量"],
            ),
            SectorImpact(
                sector="特高压外送通道与电网调度设备",
                transmission_logic="暴露出跨省跨区电力互济短板，倒逼国家加大特高压直流外送与配电网智能化改造投资。",
                impact_level="中高",
                duration="中长期",
                key_drivers=["特高压项目核准提速", "电网投资预算追加"],
            ),
        ],
        adversely_affected_sectors=[
            SectorImpact(
                sector="限电重灾区的高耗能制造企业(电解铝/硅料/芯片代工)",
                transmission_logic="产线停产导致固定资产折旧空转，且电解槽重启成本极为昂贵，当季产销量与利润严重受损。",
                impact_level="极高",
                duration="短期",
                key_drivers=["限电停产天数", "设备停机重启损失"],
            ),
            SectorImpact(
                sector="水力发电运营商(干旱流域水电企)",
                transmission_logic="入库来水严重偏枯，导致发电设备利用小时数与发电量大幅跳水，单季营收腰斩。",
                impact_level="高",
                duration="短期",
                key_drivers=["水库入库流量同比降幅", "发电量降幅"],
            ),
        ],
        cross_market_spillovers={
            "A股": "煤电、电网设备、储能、虚拟电厂及限电涨价大宗品暴涨，受限电停产区域个股短期承压。",
            "大宗商品": "动力煤现货、电解铝、工业硅期货价格呈现脉冲式暴涨。",
            "全球宏观": "凸显全球气候变暖带来的极端天气对全球制造业供应链韧性的长期挑战。",
        },
        key_monitoring_indicators=["三峡及主要流域入库水流量(立方米/秒)", "沿海八省电厂日耗煤量与存煤可用天数", "全国各省最高用电负荷读数", "电解铝与工业硅开工率变动"],
        historical_reference_cases=["2022年8月四川历史罕见极端高温干旱引发的全省工业限电大停产行情", "2021年秋季能耗双控限电限产推动的周期品暴涨行情"],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 宏观情景查询与上下文组装辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def get_all_macro_event_ids() -> List[str]:
    """获取所有宏观事件的情景ID列表。"""
    return list(MACRO_EVENT_SCENARIOS.keys())


def get_all_macro_event_names() -> List[str]:
    """获取所有宏观事件的标准中文名称列表。"""
    return [e.event_name for e in MACRO_EVENT_SCENARIOS.values()]


def get_macro_event_scenario(event_or_alias: str) -> Optional[MacroEventScenario]:
    """根据宏观事件ID、标准名称或别名精确/模糊匹配宏观事件情景。"""
    if not event_or_alias or not isinstance(event_or_alias, str):
        return None

    query = event_or_alias.strip().lower()
    if not query:
        return None

    # 1. 尝试直接通过 ID 匹配
    if query in MACRO_EVENT_SCENARIOS:
        return MACRO_EVENT_SCENARIOS[query]

    # 2. 尝试标准名称或别名精确匹配
    for scenario in MACRO_EVENT_SCENARIOS.values():
        if query == scenario.event_name.lower():
            return scenario
        for alias in scenario.aliases:
            if query == alias.lower():
                return scenario

    # 3. 尝试子串包含匹配
    for scenario in MACRO_EVENT_SCENARIOS.values():
        if query in scenario.event_name.lower() or scenario.event_name.lower() in query:
            return scenario
        for alias in scenario.aliases:
            if query in alias.lower() or alias.lower() in query:
                return scenario

    return None


def search_macro_events(keyword: str) -> List[MacroEventScenario]:
    """根据关键词搜索匹配的宏观事件情景列表。"""
    if not keyword or not isinstance(keyword, str):
        return []

    kw = keyword.strip().lower()
    matches: List[MacroEventScenario] = []

    for scenario in MACRO_EVENT_SCENARIOS.values():
        score = 0
        if kw in scenario.event_id.lower():
            score += 10
        if kw in scenario.event_name.lower():
            score += 10
        if any(kw in a.lower() for a in scenario.aliases):
            score += 5
        if kw in scenario.description.lower():
            score += 3
        if any(kw in d.lower() for d in scenario.direct_impact):
            score += 3
        if any(kw in b.sector.lower() or kw in b.transmission_logic.lower() for b in scenario.beneficiary_sectors):
            score += 4
        if any(kw in a.sector.lower() or kw in a.transmission_logic.lower() for a in scenario.adversely_affected_sectors):
            score += 4

        if score > 0:
            matches.append(scenario)

    return matches


def match_events_from_text(text: str) -> List[MacroEventScenario]:
    """从输入的新闻文本、宏观描述或分析报告中自动识别并提取相关的宏观事件情景。"""
    if not text or not isinstance(text, str):
        return []

    matched: List[MacroEventScenario] = []
    text_lower = text.lower()

    for scenario in MACRO_EVENT_SCENARIOS.values():
        is_hit = False
        if scenario.event_name.lower() in text_lower:
            is_hit = True
        else:
            for alias in scenario.aliases:
                if alias.lower() in text_lower:
                    is_hit = True
                    break

        if is_hit:
            matched.append(scenario)

    return matched


def get_transmission_path(event_or_alias: str) -> Optional[List[str]]:
    """获取指定宏观事件的三级传导链路。"""
    scenario = get_macro_event_scenario(event_or_alias)
    if not scenario:
        return None

    return list(scenario.transmission_mechanism)


def get_sector_macro_exposure(sector_name: str) -> Dict[str, Any]:
    """反向查询：获取指定行业/板块在各类宏观事件下的收益与受损敞口矩阵。"""
    if not sector_name or not isinstance(sector_name, str):
        return {"beneficiary_in": [], "adversely_affected_in": []}

    s_name = sector_name.strip().lower()
    beneficiary_in: List[Dict[str, Any]] = []
    adversely_affected_in: List[Dict[str, Any]] = []

    for scenario in MACRO_EVENT_SCENARIOS.values():
        for b in scenario.beneficiary_sectors:
            if s_name in b.sector.lower() or b.sector.lower() in s_name:
                beneficiary_in.append({
                    "event_id": scenario.event_id,
                    "event_name": scenario.event_name,
                    "transmission_logic": b.transmission_logic,
                    "impact_level": b.impact_level,
                    "duration": b.duration,
                    "key_drivers": list(b.key_drivers),
                })

        for a in scenario.adversely_affected_sectors:
            if s_name in a.sector.lower() or a.sector.lower() in s_name:
                adversely_affected_in.append({
                    "event_id": scenario.event_id,
                    "event_name": scenario.event_name,
                    "transmission_logic": a.transmission_logic,
                    "impact_level": a.impact_level,
                    "duration": a.duration,
                    "key_drivers": list(a.key_drivers),
                })

    return {
        "sector_query": sector_name,
        "beneficiary_in": beneficiary_in,
        "adversely_affected_in": adversely_affected_in,
    }


def format_macro_event_context(event_or_alias: str) -> str:
    """生成紧凑、结构化的高信息密度宏观事件传导上下文文本，供 LLM Prompt/Agent 直接注入使用。

    若未匹配到特定宏观事件，则返回空字符串。
    """
    scenario = get_macro_event_scenario(event_or_alias)
    if not scenario:
        return ""

    b_lines = [f"     * 【{b.sector}】(影响:{b.impact_level}|周期:{b.duration})：{b.transmission_logic}" for b in scenario.beneficiary_sectors]
    a_lines = [f"     * 【{a.sector}】(风险:{a.impact_level}|周期:{a.duration})：{a.transmission_logic}" for a in scenario.adversely_affected_sectors]

    cross_mkt_str = " | ".join([f"{k}:{v}" for k, v in scenario.cross_market_spillovers.items()])

    lines = [
        f"【宏观事件传导图谱 - {scenario.event_name} ({scenario.category})】",
        f"1. 核心事件定义与直接冲击：",
        f"   - 概要：{scenario.description}",
        f"   - 直接变量影响：{'; '.join(scenario.direct_impact)}",
        f"2. 三级传导机制推演：",
        f"   - Step 1: {scenario.transmission_mechanism[0]}",
        f"   - Step 2: {scenario.transmission_mechanism[1]}",
        f"   - Step 3: {scenario.transmission_mechanism[2]}",
        f"3. 行业结构性分化影响：",
        f"   - 明确受益行业：",
        *b_lines,
        f"   - 明确受损承压行业：",
        *a_lines,
        f"4. 跨市场联动与高频监测：",
        f"   - 跨市场外溢：{cross_mkt_str}",
        f"   - 核心监控指标：{', '.join(scenario.key_monitoring_indicators)}",
        f"   - 历史参考情景：{'; '.join(scenario.historical_reference_cases)}",
    ]

    return "\n".join(lines)
