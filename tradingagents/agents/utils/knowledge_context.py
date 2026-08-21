"""Knowledge context resolver and formatter for analyst agents.

Provides deterministic resolution and formatting for:
1. Industry deep context from tradingagents.knowledge.industry_linkage
2. Macro event scenario context from tradingagents.knowledge.macro_events
3. Global indices, major assets, and domestic market views
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Tuple

from tradingagents.knowledge.industry_linkage import (
    INDUSTRY_PROFILES,
    IndustryProfile,
    format_industry_deep_context,
    get_industry_profile,
    search_industries,
)
from tradingagents.knowledge.macro_events import (
    MACRO_EVENT_SCENARIOS,
    MacroEventScenario,
    format_macro_event_context,
    get_macro_event_scenario,
    match_events_from_text,
)
from tradingagents.knowledge.rag import (
    INDUSTRY_KNOWLEDGE_MISSING_BLOCK,
    KNOWLEDGE_MISSING_FALLBACK,
    MACRO_EVENT_MISSING_BLOCK,
    format_rag_industry_context,
    format_rag_macro_context,
    retrieve_industry_knowledge,
    retrieve_macro_event_knowledge,
)

logger = logging.getLogger(__name__)

# 常见 A 股代表性标的证券代码（6位代码）到标准行业 ID 的映射字典
_STOCK_CODE_INDUSTRY_MAP: dict[str, str] = {
    # 白酒与精制茶酒 (liquor_beverage)
    "600519": "liquor_beverage",  # 贵州茅台 / 茅台
    "000858": "liquor_beverage",  # 五粮液
    "000568": "liquor_beverage",  # 泸州老窖
    "600809": "liquor_beverage",  # 山西汾酒 / 汾酒
    "002304": "liquor_beverage",  # 洋河股份 / 洋河
    "000596": "liquor_beverage",  # 古井贡酒 / 古井贡
    "603369": "liquor_beverage",  # 今世缘
    "000799": "liquor_beverage",  # 酒鬼酒
    "600702": "liquor_beverage",  # 舍得酒业
    "600600": "liquor_beverage",  # 青岛啤酒
    "600132": "liquor_beverage",  # 重庆啤酒
    "000729": "liquor_beverage",  # 燕京啤酒
    # 动力电池与储能电池材料 (lithium_battery)
    "300750": "lithium_battery",  # 宁德时代
    "300014": "lithium_battery",  # 亿纬锂能
    "002074": "lithium_battery",  # 国轩高科
    "300207": "lithium_battery",  # 欣旺达
    "002812": "lithium_battery",  # 恩捷股份
    "300568": "lithium_battery",  # 星源材质
    "603659": "lithium_battery",  # 璞泰来
    "600884": "lithium_battery",  # 杉杉股份
    "300919": "lithium_battery",  # 中伟股份
    "603799": "lithium_battery",  # 华友钴业
    "002709": "lithium_battery",  # 天赐材料
    "300037": "lithium_battery",  # 新宙邦
    # 新能源汽车与智能汽车 (nev_auto)
    "002594": "nev_auto",  # 比亚迪
    "601633": "nev_auto",  # 长城汽车
    "000625": "nev_auto",  # 长安汽车
    "601127": "nev_auto",  # 赛力斯
    "601238": "nev_auto",  # 广汽集团
    "600104": "nev_auto",  # 上汽集团
    "601689": "nev_auto",  # 拓普集团
    "002050": "nev_auto",  # 三花智控
    "002920": "nev_auto",  # 德赛西威
    "002906": "nev_auto",  # 华阳集团
    "603596": "nev_auto",  # 伯特利
    # 半导体与集成电路 (semiconductor)
    "688981": "semiconductor",  # 中芯国际 / 中芯
    "688347": "semiconductor",  # 华虹公司
    "002371": "semiconductor",  # 北方华创
    "688012": "semiconductor",  # 中微公司
    "688072": "semiconductor",  # 拓荆科技
    "688082": "semiconductor",  # 盛美上海
    "688041": "semiconductor",  # 海光信息
    "688256": "semiconductor",  # 寒武纪
    "603986": "semiconductor",  # 兆易创新
    "603501": "semiconductor",  # 韦尔股份
    "300661": "semiconductor",  # 圣邦股份
    "300782": "semiconductor",  # 卓胜微
    "688008": "semiconductor",  # 澜起科技
    "600584": "semiconductor",  # 长电科技
    "002156": "semiconductor",  # 通富微电
    "002185": "semiconductor",  # 华天科技
    # 人工智能与算力服务 (ai_computing)
    "002230": "ai_computing",  # 科大讯飞
    "000977": "ai_computing",  # 浪潮信息
    "603019": "ai_computing",  # 中科曙光
    "688111": "ai_computing",  # 金山办公
    "601360": "ai_computing",  # 三六零
    "688327": "ai_computing",  # 云从科技
    "002415": "ai_computing",  # 海康威视
    "002236": "ai_computing",  # 大华股份
    "000938": "ai_computing",  # 紫光股份
    # 光伏与储能系统 (photovoltaic_storage)
    "601012": "photovoltaic_storage",  # 隆基绿能
    "300274": "photovoltaic_storage",  # 阳光电源
    "600438": "photovoltaic_storage",  # 通威股份
    "002459": "photovoltaic_storage",  # 晶澳科技
    "688599": "photovoltaic_storage",  # 天合光能
    "688223": "photovoltaic_storage",  # 晶科能源
    "300763": "photovoltaic_storage",  # 锦浪科技
    "688390": "photovoltaic_storage",  # 固德威
    "605117": "photovoltaic_storage",  # 德业股份
    "603806": "photovoltaic_storage",  # 福斯特
    "601865": "photovoltaic_storage",  # 福莱特
    # 医药生物与创新药 (biopharma)
    "600276": "biopharma",  # 恒瑞医药
    "603259": "biopharma",  # 药明康德
    "688235": "biopharma",  # 百济神州
    "688180": "biopharma",  # 君实生物
    "600196": "biopharma",  # 复星医药
    "000661": "biopharma",  # 长春高新
    "300122": "biopharma",  # 智飞生物
    "300601": "biopharma",  # 康泰生物
    "300142": "biopharma",  # 沃森生物
    "600436": "biopharma",  # 片仔癀
    "000538": "biopharma",  # 云南白药
    "600085": "biopharma",  # 同仁堂
    "000963": "biopharma",  # 华东医药
    # 医疗器械与医疗服务 (medical_devices)
    "300760": "medical_devices",  # 迈瑞医疗
    "688271": "medical_devices",  # 联影医疗
    "300832": "medical_devices",  # 新产业
    "603658": "medical_devices",  # 安图生物
    "300015": "medical_devices",  # 爱尔眼科
    "600763": "medical_devices",  # 通策医疗
    "300347": "medical_devices",  # 泰格医药
    "300759": "medical_devices",  # 康龙化成
    # 消费电子与智能终端 (consumer_electronics)
    "002475": "consumer_electronics",  # 立讯精密
    "002241": "consumer_electronics",  # 歌尔股份
    "300433": "consumer_electronics",  # 蓝思科技
    "000725": "consumer_electronics",  # 京东方A / 京东方
    "000100": "consumer_electronics",  # TCL科技
    "688036": "consumer_electronics",  # 传音控股
    "002600": "consumer_electronics",  # 领益智造
    "002938": "consumer_electronics",  # 鹏鼎控股
    # 商业银行与信贷 (banking)
    "600036": "banking",  # 招商银行
    "601398": "banking",  # 工商银行
    "601939": "banking",  # 建设银行
    "601288": "banking",  # 农业银行
    "601988": "banking",  # 中国银行
    "601328": "banking",  # 交通银行
    "601658": "banking",  # 邮储银行
    "601166": "banking",  # 兴业银行
    "600000": "banking",  # 浦发银行
    "601998": "banking",  # 中信银行
    "600016": "banking",  # 民生银行
    "601818": "banking",  # 光大银行
    "000001": "banking",  # 平安银行
    "002142": "banking",  # 宁波银行
    "600919": "banking",  # 江苏银行
    "601009": "banking",  # 南京银行
    "600926": "banking",  # 杭州银行
    "601838": "banking",  # 成都银行
    # 证券公司与资本市场 (securities)
    "600030": "securities",  # 中信证券
    "601066": "securities",  # 中信建投
    "601995": "securities",  # 中金公司
    "601688": "securities",  # 华泰证券
    "601211": "securities",  # 国泰君安
    "600837": "securities",  # 海通证券
    "000776": "securities",  # 广发证券
    "600999": "securities",  # 招商证券
    "000166": "securities",  # 申万宏源
    "601881": "securities",  # 中国银河
    "300059": "securities",  # 东方财富
    "601788": "securities",  # 光大证券
    # 保险与多元金融 (insurance_financials)
    "601318": "insurance_financials",  # 中国平安
    "601628": "insurance_financials",  # 中国人寿
    "601601": "insurance_financials",  # 中国太保
    "601336": "insurance_financials",  # 新华保险
    "601319": "insurance_financials",  # 中国人保
    # 钢铁与黑色金属 (steel_ferrous)
    "600019": "steel_ferrous",  # 宝钢股份
    "000932": "steel_ferrous",  # 华菱钢铁
    "000959": "steel_ferrous",  # 首钢股份
    "000898": "steel_ferrous",  # 鞍钢股份
    "600010": "steel_ferrous",  # 包钢股份
    "600282": "steel_ferrous",  # 南钢股份
    # 有色金属与工业金属 (nonferrous_metals)
    "601600": "nonferrous_metals",  # 中国铝业
    "000807": "nonferrous_metals",  # 云铝股份
    "000933": "nonferrous_metals",  # 神火股份
    "600362": "nonferrous_metals",  # 江西铜业
    "000630": "nonferrous_metals",  # 铜陵有色
    "000878": "nonferrous_metals",  # 云南铜业
    "603993": "nonferrous_metals",  # 洛阳钼业
    "600111": "nonferrous_metals",  # 北方稀土
    "000831": "nonferrous_metals",  # 中国稀土
    "002466": "nonferrous_metals",  # 天齐锂业
    "002460": "nonferrous_metals",  # 赣锋锂业
    # 贵金属与稀缺资源 (precious_metals)
    "601899": "precious_metals",  # 紫金矿业
    "600547": "precious_metals",  # 山东黄金
    "600489": "precious_metals",  # 中金黄金
    "600988": "precious_metals",  # 赤峰黄金
    "000975": "precious_metals",  # 银泰黄金
    "002155": "precious_metals",  # 湖南黄金
    "000603": "precious_metals",  # 盛达资源
    # 石油石化与基础化工 (petrochemicals)
    "601857": "petrochemicals",  # 中国石油
    "600028": "petrochemicals",  # 中国石化
    "600938": "petrochemicals",  # 中国海油
    "600309": "petrochemicals",  # 万华化学
    "600346": "petrochemicals",  # 恒力石化
    "002493": "petrochemicals",  # 荣盛石化
    "000301": "petrochemicals",  # 东方盛虹
    "600426": "petrochemicals",  # 华鲁恒升
    "600989": "petrochemicals",  # 宝丰能源
    "002648": "petrochemicals",  # 卫星化学
    # 煤炭与传统化石能源 (coal_energy)
    "601088": "coal_energy",  # 中国神华
    "601225": "coal_energy",  # 陕西煤业
    "600188": "coal_energy",  # 兖矿能源
    "000983": "coal_energy",  # 山西焦煤
    "601898": "coal_energy",  # 中煤能源
    "601699": "coal_energy",  # 潞安环能
    "600985": "coal_energy",  # 淮北矿业
    # 电力与公用事业 (power_utilities)
    "600900": "power_utilities",  # 长江电力
    "600011": "power_utilities",  # 华能国际
    "600795": "power_utilities",  # 国电电力
    "600905": "power_utilities",  # 三峡能源
    "001289": "power_utilities",  # 龙源电力
    "601985": "power_utilities",  # 中国核电
    "003816": "power_utilities",  # 中国广核
    "600025": "power_utilities",  # 华能水电
    "600886": "power_utilities",  # 国投电力
    "600023": "power_utilities",  # 浙能电力
    # 房地产开发与运营 (real_estate)
    "000002": "real_estate",  # 万科A / 万科
    "600048": "real_estate",  # 保利发展
    "001979": "real_estate",  # 招商蛇口
    "600383": "real_estate",  # 金地集团
    "601155": "real_estate",  # 新城控股
    "002244": "real_estate",  # 滨江集团
    # 建筑装饰与基础设施工程 (construction_materials)
    "600585": "construction_materials",  # 海螺水泥
    "600801": "construction_materials",  # 华新水泥
    "002271": "construction_materials",  # 东方雨虹
    "000786": "construction_materials",  # 北新建材
    "002372": "construction_materials",  # 伟星新材
    "601668": "construction_materials",  # 中国建筑
    "601390": "construction_materials",  # 中国中铁
    "601186": "construction_materials",  # 中国铁建
    "601800": "construction_materials",  # 中国交建
    "601669": "construction_materials",  # 中国电建
    # 机械设备与工业母机 (industrial_machinery)
    "600031": "industrial_machinery",  # 三一重工
    "000157": "industrial_machinery",  # 中联重科
    "000425": "industrial_machinery",  # 徐工机械
    "601100": "industrial_machinery",  # 恒立液压
    "300124": "industrial_machinery",  # 汇川技术
    "688017": "industrial_machinery",  # 绿的谐波
    "300161": "industrial_machinery",  # 华中数控
    "601882": "industrial_machinery",  # 海天精工
    "300083": "industrial_machinery",  # 创世纪
    # 国防军工与航天装备 (defense_military)
    "600760": "defense_military",  # 中航沈飞
    "000768": "defense_military",  # 中航西飞
    "600893": "defense_military",  # 航发动力
    "002179": "defense_military",  # 中航光电
    "600765": "defense_military",  # 中航重机
    "600150": "defense_military",  # 中国船舶
    "601989": "defense_military",  # 中国重工
    "600372": "defense_military",  # 中航电子
    "600879": "defense_military",  # 航天电子
    # 交通运输与航运港口 (logistics_shipping)
    "601919": "logistics_shipping",  # 中远海控
    "002352": "logistics_shipping",  # 顺丰控股
    "601816": "logistics_shipping",  # 京沪高铁
    "601006": "logistics_shipping",  # 大秦铁路
    "601872": "logistics_shipping",  # 招商轮船
    "600026": "logistics_shipping",  # 中远海能
    "600233": "logistics_shipping",  # 圆通速递
    "600009": "logistics_shipping",  # 上海机场
    "600004": "logistics_shipping",  # 白云机场
    "600018": "logistics_shipping",  # 上港集团
    "601018": "logistics_shipping",  # 宁波港
    # 通信网络与光通信 (telecom_optical)
    "000063": "telecom_optical",  # 中兴通讯
    "300308": "telecom_optical",  # 中际旭创
    "300502": "telecom_optical",  # 新易盛
    "300394": "telecom_optical",  # 天孚通信
    "600941": "telecom_optical",  # 中国移动
    "601728": "telecom_optical",  # 中国电信
    "600050": "telecom_optical",  # 中国联通
    "600498": "telecom_optical",  # 烽火通信
    "600487": "telecom_optical",  # 亨通光电
    "600522": "telecom_optical",  # 中天科技
    # 农林牧渔与生猪养殖 (agriculture_breeding)
    "002714": "agriculture_breeding",  # 牧原股份
    "300498": "agriculture_breeding",  # 温氏股份
    "000876": "agriculture_breeding",  # 新希望
    "002311": "agriculture_breeding",  # 海大集团
    "002299": "agriculture_breeding",  # 圣农发展
    "002385": "agriculture_breeding",  # 大北农
    "000998": "agriculture_breeding",  # 隆平高科
    "002041": "agriculture_breeding",  # 登海种业
    # 家用电器与智能家居 (home_appliances)
    "000333": "home_appliances",  # 美的集团
    "000651": "home_appliances",  # 格力电器
    "600690": "home_appliances",  # 海尔智家
    "002508": "home_appliances",  # 老板电器
    "002032": "home_appliances",  # 苏泊尔
    "688169": "home_appliances",  # 石头科技
    "603486": "home_appliances",  # 科沃斯
    "600060": "home_appliances",  # 海信视像
    "000921": "home_appliances",  # 海信家电
    # 大众食品与饮料 (food_beverage)
    "600887": "food_beverage",  # 伊利股份
    "603288": "food_beverage",  # 海天味业
    "000895": "food_beverage",  # 双汇发展
    "600872": "food_beverage",  # 中炬高新
    "603027": "food_beverage",  # 千禾味业
    "603345": "food_beverage",  # 安井食品
    "603517": "food_beverage",  # 绝味食品
    "002557": "food_beverage",  # 洽洽食品
    "605499": "food_beverage",  # 东鹏饮料
    "603156": "food_beverage",  # 养元饮品
}

# 常见 A 股代表性标的/关键词到标准行业 ID 的映射字典
_STOCK_NAME_INDUSTRY_MAP: dict[str, str] = {
    # 白酒与精制茶酒 (liquor_beverage)
    "茅台": "liquor_beverage",
    "贵州茅台": "liquor_beverage",
    "五粮液": "liquor_beverage",
    "泸州老窖": "liquor_beverage",
    "山西汾酒": "liquor_beverage",
    "汾酒": "liquor_beverage",
    "洋河股份": "liquor_beverage",
    "洋河": "liquor_beverage",
    "古井贡酒": "liquor_beverage",
    "古井贡": "liquor_beverage",
    "今世缘": "liquor_beverage",
    "酒鬼酒": "liquor_beverage",
    "舍得酒业": "liquor_beverage",
    "青岛啤酒": "liquor_beverage",
    "重庆啤酒": "liquor_beverage",
    "燕京啤酒": "liquor_beverage",
    # 动力电池与储能电池材料 (lithium_battery)
    "宁德时代": "lithium_battery",
    "亿纬锂能": "lithium_battery",
    "国轩高科": "lithium_battery",
    "欣旺达": "lithium_battery",
    "恩捷股份": "lithium_battery",
    "星源材质": "lithium_battery",
    "璞泰来": "lithium_battery",
    "杉杉股份": "lithium_battery",
    "中伟股份": "lithium_battery",
    "华友钴业": "lithium_battery",
    "天赐材料": "lithium_battery",
    "新宙邦": "lithium_battery",
    # 新能源汽车与智能汽车 (nev_auto)
    "比亚迪": "nev_auto",
    "长城汽车": "nev_auto",
    "长安汽车": "nev_auto",
    "赛力斯": "nev_auto",
    "广汽集团": "nev_auto",
    "上汽集团": "nev_auto",
    "拓普集团": "nev_auto",
    "三花智控": "nev_auto",
    "德赛西威": "nev_auto",
    "华阳集团": "nev_auto",
    "伯特利": "nev_auto",
    # 半导体与集成电路 (semiconductor)
    "中芯国际": "semiconductor",
    "中芯": "semiconductor",
    "华虹公司": "semiconductor",
    "北方华创": "semiconductor",
    "中微公司": "semiconductor",
    "拓荆科技": "semiconductor",
    "盛美上海": "semiconductor",
    "海光信息": "semiconductor",
    "寒武纪": "semiconductor",
    "兆易创新": "semiconductor",
    "韦尔股份": "semiconductor",
    "圣邦股份": "semiconductor",
    "卓胜微": "semiconductor",
    "澜起科技": "semiconductor",
    "长电科技": "semiconductor",
    "通富微电": "semiconductor",
    "华天科技": "semiconductor",
    # 人工智能与算力服务 (ai_computing)
    "科大讯飞": "ai_computing",
    "浪潮信息": "ai_computing",
    "中科曙光": "ai_computing",
    "金山办公": "ai_computing",
    "三六零": "ai_computing",
    "云从科技": "ai_computing",
    "海康威视": "ai_computing",
    "大华股份": "ai_computing",
    "紫光股份": "ai_computing",
    # 光伏与储能系统 (photovoltaic_storage)
    "隆基绿能": "photovoltaic_storage",
    "阳光电源": "photovoltaic_storage",
    "通威股份": "photovoltaic_storage",
    "晶澳科技": "photovoltaic_storage",
    "天合光能": "photovoltaic_storage",
    "晶科能源": "photovoltaic_storage",
    "锦浪科技": "photovoltaic_storage",
    "固德威": "photovoltaic_storage",
    "德业股份": "photovoltaic_storage",
    "福斯特": "photovoltaic_storage",
    "福莱特": "photovoltaic_storage",
    # 医药生物与创新药 (biopharma)
    "恒瑞医药": "biopharma",
    "药明康德": "biopharma",
    "百济神州": "biopharma",
    "信达生物": "biopharma",
    "君实生物": "biopharma",
    "复星医药": "biopharma",
    "长春高新": "biopharma",
    "智飞生物": "biopharma",
    "康泰生物": "biopharma",
    "沃森生物": "biopharma",
    "片仔癀": "biopharma",
    "云南白药": "biopharma",
    "同仁堂": "biopharma",
    "华东医药": "biopharma",
    # 医疗器械与医疗服务 (medical_devices)
    "迈瑞医疗": "medical_devices",
    "联影医疗": "medical_devices",
    "新产业": "medical_devices",
    "安图生物": "medical_devices",
    "爱尔眼科": "medical_devices",
    "通策医疗": "medical_devices",
    "泰格医药": "medical_devices",
    "康龙化成": "medical_devices",
    # 消费电子与智能终端 (consumer_electronics)
    "立讯精密": "consumer_electronics",
    "歌尔股份": "consumer_electronics",
    "蓝思科技": "consumer_electronics",
    "京东方A": "consumer_electronics",
    "京东方": "consumer_electronics",
    "TCL科技": "consumer_electronics",
    "传音控股": "consumer_electronics",
    "领益智造": "consumer_electronics",
    "鹏鼎控股": "consumer_electronics",
    "欣旺达消费": "consumer_electronics",
    # 商业银行与信贷 (banking)
    "招商银行": "banking",
    "工商银行": "banking",
    "建设银行": "banking",
    "农业银行": "banking",
    "中国银行": "banking",
    "交通银行": "banking",
    "邮储银行": "banking",
    "兴业银行": "banking",
    "浦发银行": "banking",
    "中信银行": "banking",
    "民生银行": "banking",
    "光大银行": "banking",
    "平安银行": "banking",
    "宁波银行": "banking",
    "江苏银行": "banking",
    "南京银行": "banking",
    "杭州银行": "banking",
    "成都银行": "banking",
    # 证券公司与资本市场 (securities)
    "中信证券": "securities",
    "中信建投": "securities",
    "中金公司": "securities",
    "华泰证券": "securities",
    "国泰君安": "securities",
    "海通证券": "securities",
    "广发证券": "securities",
    "招商证券": "securities",
    "申万宏源": "securities",
    "中国银河": "securities",
    "东方财富": "securities",
    "光大证券": "securities",
    # 保险与多元金融 (insurance_financials)
    "中国平安": "insurance_financials",
    "中国人寿": "insurance_financials",
    "中国太保": "insurance_financials",
    "新华保险": "insurance_financials",
    "中国人保": "insurance_financials",
    # 钢铁与黑色金属 (steel_ferrous)
    "宝钢股份": "steel_ferrous",
    "华菱钢铁": "steel_ferrous",
    "首钢股份": "steel_ferrous",
    "鞍钢股份": "steel_ferrous",
    "包钢股份": "steel_ferrous",
    "南钢股份": "steel_ferrous",
    # 有色金属与工业金属 (nonferrous_metals)
    "中国铝业": "nonferrous_metals",
    "云铝股份": "nonferrous_metals",
    "神火股份": "nonferrous_metals",
    "江西铜业": "nonferrous_metals",
    "铜陵有色": "nonferrous_metals",
    "云南铜业": "nonferrous_metals",
    "洛阳钼业": "nonferrous_metals",
    "北方稀土": "nonferrous_metals",
    "中国稀土": "nonferrous_metals",
    "天齐锂业": "nonferrous_metals",
    "赣锋锂业": "nonferrous_metals",
    # 贵金属与稀缺资源 (precious_metals)
    "紫金矿业": "precious_metals",
    "山东黄金": "precious_metals",
    "中金黄金": "precious_metals",
    "赤峰黄金": "precious_metals",
    "银泰黄金": "precious_metals",
    "湖南黄金": "precious_metals",
    "盛达资源": "precious_metals",
    # 石油石化与基础化工 (petrochemicals)
    "中国石油": "petrochemicals",
    "中国石化": "petrochemicals",
    "中国海油": "petrochemicals",
    "万华化学": "petrochemicals",
    "恒力石化": "petrochemicals",
    "荣盛石化": "petrochemicals",
    "东方盛虹": "petrochemicals",
    "华鲁恒升": "petrochemicals",
    "宝丰能源": "petrochemicals",
    "卫星化学": "petrochemicals",
    # 煤炭与传统化石能源 (coal_energy)
    "中国神华": "coal_energy",
    "陕西煤业": "coal_energy",
    "兖矿能源": "coal_energy",
    "山西焦煤": "coal_energy",
    "中煤能源": "coal_energy",
    "潞安环能": "coal_energy",
    "淮北矿业": "coal_energy",
    # 电力与公用事业 (power_utilities)
    "长江电力": "power_utilities",
    "华能国际": "power_utilities",
    "国电电力": "power_utilities",
    "三峡能源": "power_utilities",
    "龙源电力": "power_utilities",
    "中国核电": "power_utilities",
    "中国广核": "power_utilities",
    "华能水电": "power_utilities",
    "国投电力": "power_utilities",
    "浙能电力": "power_utilities",
    # 房地产开发与运营 (real_estate)
    "万科A": "real_estate",
    "万科": "real_estate",
    "保利发展": "real_estate",
    "招商蛇口": "real_estate",
    "金地集团": "real_estate",
    "新城控股": "real_estate",
    "滨江集团": "real_estate",
    # 建筑装饰与基础设施工程 (construction_materials)
    "海螺水泥": "construction_materials",
    "华新水泥": "construction_materials",
    "东方雨虹": "construction_materials",
    "北新建材": "construction_materials",
    "伟星新材": "construction_materials",
    "中国建筑": "construction_materials",
    "中国中铁": "construction_materials",
    "中国铁建": "construction_materials",
    "中国交建": "construction_materials",
    "中国电建": "construction_materials",
    # 机械设备与工业母机 (industrial_machinery)
    "三一重工": "industrial_machinery",
    "中联重科": "industrial_machinery",
    "徐工机械": "industrial_machinery",
    "恒立液压": "industrial_machinery",
    "汇川技术": "industrial_machinery",
    "绿的谐波": "industrial_machinery",
    "华中数控": "industrial_machinery",
    "海天精工": "industrial_machinery",
    "创世纪": "industrial_machinery",
    # 国防军工与航天装备 (defense_military)
    "中航沈飞": "defense_military",
    "中航西飞": "defense_military",
    "航发动力": "defense_military",
    "中航光电": "defense_military",
    "中航重机": "defense_military",
    "中国船舶": "defense_military",
    "中国重工": "defense_military",
    "中航电子": "defense_military",
    "航天电子": "defense_military",
    # 交通运输与航运港口 (logistics_shipping)
    "中远海控": "logistics_shipping",
    "顺丰控股": "logistics_shipping",
    "京沪高铁": "logistics_shipping",
    "大秦铁路": "logistics_shipping",
    "招商轮船": "logistics_shipping",
    "中远海能": "logistics_shipping",
    "圆通速递": "logistics_shipping",
    "上海机场": "logistics_shipping",
    "白云机场": "logistics_shipping",
    "上港集团": "logistics_shipping",
    "宁波港": "logistics_shipping",
    # 通信网络与光通信 (telecom_optical)
    "中兴通讯": "telecom_optical",
    "中际旭创": "telecom_optical",
    "新易盛": "telecom_optical",
    "天孚通信": "telecom_optical",
    "中国移动": "telecom_optical",
    "中国电信": "telecom_optical",
    "中国联通": "telecom_optical",
    "烽火通信": "telecom_optical",
    "亨通光电": "telecom_optical",
    "中天科技": "telecom_optical",
    # 农林牧渔与生猪养殖 (agriculture_breeding)
    "牧原股份": "agriculture_breeding",
    "温氏股份": "agriculture_breeding",
    "新希望": "agriculture_breeding",
    "海大集团": "agriculture_breeding",
    "圣农发展": "agriculture_breeding",
    "大北农": "agriculture_breeding",
    "隆平高科": "agriculture_breeding",
    "登海种业": "agriculture_breeding",
    # 家用电器与智能家居 (home_appliances)
    "美的集团": "home_appliances",
    "格力电器": "home_appliances",
    "海尔智家": "home_appliances",
    "老板电器": "home_appliances",
    "苏泊尔": "home_appliances",
    "石头科技": "home_appliances",
    "科沃斯": "home_appliances",
    "海信视像": "home_appliances",
    "海信家电": "home_appliances",
    # 大众食品与饮料 (food_beverage)
    "伊利股份": "food_beverage",
    "海天味业": "food_beverage",
    "双汇发展": "food_beverage",
    "中炬高新": "food_beverage",
    "千禾味业": "food_beverage",
    "安井食品": "food_beverage",
    "绝味食品": "food_beverage",
    "洽洽食品": "food_beverage",
    "东鹏饮料": "food_beverage",
    "养元饮品": "food_beverage",
}

# 行业后缀通用规则匹配（按优先级排列）
_GENERIC_KEYWORD_RULES: list[tuple[str, str]] = [
    ("白酒", "liquor_beverage"),
    ("啤酒", "liquor_beverage"),
    ("黄酒", "liquor_beverage"),
    ("酒业", "liquor_beverage"),
    ("银行", "banking"),
    ("券商", "securities"),
    ("证券", "securities"),
    ("保险", "insurance_financials"),
    ("创新药", "biopharma"),
    ("生物药", "biopharma"),
    ("医药", "biopharma"),
    ("制药", "biopharma"),
    ("医疗器械", "medical_devices"),
    ("医疗服务", "medical_devices"),
    ("医疗", "medical_devices"),
    ("锂电池", "lithium_battery"),
    ("动力电池", "lithium_battery"),
    ("储能电池", "lithium_battery"),
    ("电池材料", "lithium_battery"),
    ("光伏", "photovoltaic_storage"),
    ("储能", "photovoltaic_storage"),
    ("逆变器", "photovoltaic_storage"),
    ("新能源汽车", "nev_auto"),
    ("智能汽车", "nev_auto"),
    ("汽车零部件", "nev_auto"),
    ("汽车", "nev_auto"),
    ("芯片", "semiconductor"),
    ("半导体", "semiconductor"),
    ("集成电路", "semiconductor"),
    ("晶圆", "semiconductor"),
    ("人工智能", "ai_computing"),
    ("算力", "ai_computing"),
    ("大模型", "ai_computing"),
    ("消费电子", "consumer_electronics"),
    ("智能终端", "consumer_electronics"),
    ("果链", "consumer_electronics"),
    ("钢铁", "steel_ferrous"),
    ("黑色金属", "steel_ferrous"),
    ("有色金属", "nonferrous_metals"),
    ("工业金属", "nonferrous_metals"),
    ("铜业", "nonferrous_metals"),
    ("铝业", "nonferrous_metals"),
    ("黄金", "precious_metals"),
    ("贵金属", "precious_metals"),
    ("稀土", "nonferrous_metals"),
    ("石油", "petrochemicals"),
    ("石化", "petrochemicals"),
    ("化工", "petrochemicals"),
    ("煤炭", "coal_energy"),
    ("焦煤", "coal_energy"),
    ("火电", "power_utilities"),
    ("水电", "power_utilities"),
    ("绿电", "power_utilities"),
    ("核电", "power_utilities"),
    ("电力", "power_utilities"),
    ("房地产", "real_estate"),
    ("地产", "real_estate"),
    ("水泥", "construction_materials"),
    ("建材", "construction_materials"),
    ("基建工程", "construction_materials"),
    ("建筑", "construction_materials"),
    ("工业母机", "industrial_machinery"),
    ("机床", "industrial_machinery"),
    ("工程机械", "industrial_machinery"),
    ("机械", "industrial_machinery"),
    ("军工", "defense_military"),
    ("航天", "defense_military"),
    ("航空装备", "defense_military"),
    ("船舶", "defense_military"),
    ("航运", "logistics_shipping"),
    ("港口", "logistics_shipping"),
    ("物流", "logistics_shipping"),
    ("快递", "logistics_shipping"),
    ("光通信", "telecom_optical"),
    ("光模块", "telecom_optical"),
    ("通信设备", "telecom_optical"),
    ("通信", "telecom_optical"),
    ("生猪", "agriculture_breeding"),
    ("生猪养殖", "agriculture_breeding"),
    ("养殖", "agriculture_breeding"),
    ("饲料", "agriculture_breeding"),
    ("农牧", "agriculture_breeding"),
    ("家电", "home_appliances"),
    ("智能家居", "home_appliances"),
    ("调味品", "food_beverage"),
    ("乳制品", "food_beverage"),
    ("食品", "food_beverage"),
    ("饮料", "food_beverage"),
]


def resolve_industry_profile(
    ticker: str = "",
    stock_name: str = "",
    extra_text: str = "",
    state: Optional[dict] = None,
) -> Optional[IndustryProfile]:
    """解析标的或上下文对应的行业知识库画像。

    解析优先级：
    1. state 中的 industry 字段（若存在且有效）；
    2. ticker 或 stock_name 中的 6 位证券代码精准匹配代码词典；
    3. stock_name 精确匹配代表性股票名称词典；
    4. stock_name 包含子串匹配代表性股票名称词典；
    5. stock_name 匹配通用行业关键词/后缀规则；
    6. stock_name 直接匹配标准行业图谱 profile/alias；
    7. extra_text（如财报/新闻/板块信息）匹配行业关键词或 profile/alias；
    8. search_industries 关键词检索。
    """
    state_dict = state or {}

    # 1. 显式 state 中的 industry
    explicit_ind = state_dict.get("industry")
    if isinstance(explicit_ind, str) and explicit_ind.strip():
        profile = get_industry_profile(explicit_ind.strip())
        if profile:
            return profile
        searched = search_industries(explicit_ind.strip())
        if searched:
            return searched[0]

    clean_ticker = (ticker or "").strip().upper()
    if not clean_ticker and "company_of_interest" in state_dict:
        clean_ticker = str(state_dict["company_of_interest"] or "").strip().upper()

    clean_name = (stock_name or "").strip()

    # 2. 6 位证券代码匹配（优先 ticker，后备 clean_name 中包含的代码）
    for candidate in (clean_ticker, clean_name):
        if not candidate:
            continue
        if candidate in _STOCK_CODE_INDUSTRY_MAP:
            ind_id = _STOCK_CODE_INDUSTRY_MAP[candidate]
            return INDUSTRY_PROFILES.get(ind_id)
        m = re.search(r"(\d{6})", candidate)
        if m:
            code = m.group(1)
            if code in _STOCK_CODE_INDUSTRY_MAP:
                ind_id = _STOCK_CODE_INDUSTRY_MAP[code]
                return INDUSTRY_PROFILES.get(ind_id)

    # 3. 代表性股票映射词典（精确名称匹配）
    if clean_name in _STOCK_NAME_INDUSTRY_MAP:
        ind_id = _STOCK_NAME_INDUSTRY_MAP[clean_name]
        return INDUSTRY_PROFILES.get(ind_id)

    # 4. 包含子串匹配代表性股票
    for key_name, ind_id in _STOCK_NAME_INDUSTRY_MAP.items():
        if key_name and (key_name in clean_name or clean_name in key_name):
            return INDUSTRY_PROFILES.get(ind_id)

    # 5. clean_ticker 直接匹配代表性股票词典
    if clean_ticker in _STOCK_NAME_INDUSTRY_MAP:
        ind_id = _STOCK_NAME_INDUSTRY_MAP[clean_ticker]
        return INDUSTRY_PROFILES.get(ind_id)

    # 6. 通用行业关键词/后缀规则
    for kw, ind_id in _GENERIC_KEYWORD_RULES:
        if kw in clean_name:
            return INDUSTRY_PROFILES.get(ind_id)

    # 7. 直接匹配标准行业 profile 或 alias
    if clean_name:
        profile = get_industry_profile(clean_name)
        if profile:
            return profile
        searched = search_industries(clean_name)
        if searched:
            return searched[0]

    # 8. extra_text 匹配（如财报所属行业、板块资金流等）
    if extra_text and isinstance(extra_text, str):
        # 匹配 "所属行业: XXX" 或 "行业: XXX"
        m = re.search(r"(?:所属行业|行业|所属板块|板块)[：:\s]+([^\s\n,，;；|]+)", extra_text)
        if m:
            extracted_ind = m.group(1).strip()
            profile = get_industry_profile(extracted_ind)
            if profile:
                return profile
            searched = search_industries(extracted_ind)
            if searched:
                return searched[0]

        # 扫描 extra_text 是否包含明确的行业名称/别名
        for kw, ind_id in _GENERIC_KEYWORD_RULES:
            if kw in extra_text:
                return INDUSTRY_PROFILES.get(ind_id)

        for p in INDUSTRY_PROFILES.values():
            if p.industry_name in extra_text:
                return p
            for alias in p.aliases:
                if len(alias) >= 2 and alias in extra_text:
                    return p

    # 9. 动态 RAG 检索回退（基于 BM25 加权倒排索引）
    if clean_name:
        rag_matched = retrieve_industry_knowledge(clean_name, top_k=1, min_score=1.0)
        if rag_matched:
            return rag_matched[0][0]

    return None


def resolve_industry_context(
    ticker: str = "",
    stock_name: str = "",
    extra_text: str = "",
    state: Optional[dict] = None,
    fallback_on_miss: bool = False,
) -> Tuple[Optional[IndustryProfile], str]:
    """获取标的的行业深度常识图谱文本。

    若成功匹配行业，返回 (profile, formatted_industry_context)；
    若未匹配到行业：
      - fallback_on_miss=True 时返回 (None, '【行业常识知识库】\\n【知识库未命中】')；
      - fallback_on_miss=False 时返回 (None, '')。
    """
    profile = resolve_industry_profile(
        ticker=ticker,
        stock_name=stock_name,
        extra_text=extra_text,
        state=state,
    )
    if not profile:
        return None, (INDUSTRY_KNOWLEDGE_MISSING_BLOCK if fallback_on_miss else "")

    formatted = format_industry_deep_context(profile.industry_name)
    return profile, formatted


def resolve_macro_event_context(
    text: str = "",
    max_events: int = 2,
    fallback_on_miss: bool = False,
    min_score: float = 0.5,
) -> Tuple[List[MacroEventScenario], str]:
    """从输入的文本（新闻、大盘描述、宏观背景）中自动匹配宏观事件并生成三级传导图谱。

    若匹配到事件，返回 (matched_scenarios, formatted_macro_event_context)；
    若未匹配到事件：
      - fallback_on_miss=True 时返回 ([], '【宏观事件传导图谱】\\n【知识库未命中】')；
      - fallback_on_miss=False 时返回 ([], '')。
    """
    if not text or not isinstance(text, str) or not text.strip():
        return [], (MACRO_EVENT_MISSING_BLOCK if fallback_on_miss else "")

    # 1. 结合文本直接匹配与 RAG 检索
    scenarios: List[MacroEventScenario] = []
    seen: Set[str] = set()

    # 优先直接命中
    direct_matches = match_events_from_text(text)
    for s in direct_matches:
        if s.event_id not in seen:
            seen.add(s.event_id)
            scenarios.append(s)
            if len(scenarios) >= max_events:
                break

    # 若未满 max_events，使用 RAG 检索补充
    if len(scenarios) < max_events:
        rag_matches = retrieve_macro_event_knowledge(text, top_k=max_events, min_score=min_score)
        for s, _score in rag_matches:
            if s.event_id not in seen:
                seen.add(s.event_id)
                scenarios.append(s)
                if len(scenarios) >= max_events:
                    break

    if not scenarios:
        return [], (MACRO_EVENT_MISSING_BLOCK if fallback_on_miss else "")

    blocks = [format_macro_event_context(s.event_name) for s in scenarios]
    formatted = "\n\n".join(b for b in blocks if b)
    if not formatted:
        return [], (MACRO_EVENT_MISSING_BLOCK if fallback_on_miss else "")

    return scenarios, formatted


def resolve_dynamic_knowledge_context(
    ticker: str = "",
    stock_name: str = "",
    extra_text: str = "",
    state: Optional[dict] = None,
    max_macro_events: int = 2,
    fallback_on_miss: bool = True,
) -> Dict[str, Any]:
    """统一动态解析标的行业知识图谱与宏观事件传导情景。

    返回结构：
    {
        'industry_profile': Optional[IndustryProfile],
        'industry_context': str,
        'macro_scenarios': List[MacroEventScenario],
        'macro_event_context': str,
    }
    """
    profile, ind_ctx = resolve_industry_context(
        ticker=ticker,
        stock_name=stock_name,
        extra_text=extra_text,
        state=state,
        fallback_on_miss=fallback_on_miss,
    )
    scenarios, macro_ctx = resolve_macro_event_context(
        text=extra_text,
        max_events=max_macro_events,
        fallback_on_miss=fallback_on_miss,
    )
    return {
        "industry_profile": profile,
        "industry_context": ind_ctx,
        "macro_scenarios": scenarios,
        "macro_event_context": macro_ctx,
    }


def format_macro_market_view(
    global_indices: Any = None,
    major_assets: Any = None,
    cn_indices: Any = None,
    northbound_flow: Any = None,
) -> str:
    """组装全球市场、国内大盘、大类资产与跨市场流动性统一视图。"""
    sections = []

    if global_indices and global_indices != "无数据":
        sections.append(f"【全球核心指数】\n{global_indices}")

    if major_assets and major_assets != "无数据":
        sections.append(f"【大类资产与宏观商品】\n{major_assets}")

    if cn_indices and cn_indices != "无数据":
        sections.append(f"【国内核心大盘指数】\n{cn_indices}")

    if northbound_flow and northbound_flow != "无数据":
        sections.append(f"【北向资金与跨市场流动性】\n{northbound_flow}")

    return "\n\n".join(sections)
