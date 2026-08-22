# DAV-314 只读审计：65 个 pending_api 的 Tushare 可替代矩阵

> **审计基线**：`2ee1fb8`  
> **审计模式**：只读审计、0代码改动、0测试、不打印/不泄露 Token、严格脱敏探针  
> **审计目标**：针对 `tradingagents/dataflows/industry_linkage.py` 中全部 65 个 `source="pending_api"` 产业链指标，基于 Tushare 官方文档与本机实测脱敏 HTTP 探针，逐一建立替代可行性矩阵。

---

## 一、 审计结论与矩阵概览

### 1.1 核心分类分布统计

| 分类标识 | 分类定义 | 命中指标数量 | 占比 | 核心落地策略 |
| :--- | :--- | :---: | :---: | :--- |
| **A** | **可直接真值替代**（官方现货/宏观基准利率等真实数据） | **2** | 3.1% | 立即支持 Tushare 官方真实接口接入（如 Shibor、LPR） |
| **B** | **可用近似代理但需用户批准**（如期货主力合约收盘价代理现货价） | **18** | 27.7% | 严禁冒充现货真值；须在获得用户批准后以代理形式接入 |
| **C** | **Tushare无对应接口**（官方查无此接口或无细分统计） | **44** | 67.7% | 维持 `pending_api` 状态，待扩展国家统计局/行业协会/海关等专项数据源 |
| **D** | **有接口但当前权限不足**（官方有接口，当前 Token 积分/权限受限） | **1** | 1.5% | 标注权限需求（中债国债收益率曲线 `yc_cb` 需额外权限，当前返回 403） |
| **合计** | **全口径 pending_api 指标** | **65** | **100.0%** | **建立严密分级演进路径** |

### 1.2 推荐实施优先级分布

- **P1（最高优先级 / 12项）**：包含 2 个 A 类真值指标（Shibor、LPR）、1 个 D 类核心国债收益率（10年期国债）、9 个高流动性/强联动大宗商品期货代理指标（白糖、玉米、豆粕、氧化铝、低硫燃料油、重交沥青、铁矿石、主焦煤、现货白银）；
- **P2（中高优先级 / 10项）**：包含冷轧卷板、动力煤、ABS塑料、华安黄金ETF、碳酸锂等产业链强相关大宗衍生品代理指标；
- **P3（中低优先级 / 14项）**：涉及家电出口、光伏出口、汽车销量、工业机器人产量、开工率等需行业统计局/海关等第三方权威接口支持的指标；
- **P4（长期规划 / 29项）**：涉及半导体硅片、先进封装内存HBM、实验动物、创新药出海License-out、高值耗材集采等高度非标或咨询机构私有数据库指标。

---

## 二、 65 个 pending_api 完整替代矩阵表

| 序号 | 行业 | 指标名称 | 频率 | 角色/单位 | Tushare 接口与参数 | 权限探针状态 (code / rows / as_of / fields) | 分类 | 优先级 | 审计说明与代理逻辑 |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| 01 | 半导体与集成电路 | 半导体硅片价格 | monthly | upstream_cost (美元/片) | 无 | N/A (无接口) | **C** | **P4** | Tushare无半导体硅片现货/长协价格接口，需SEMI/TrendForce数据源 |
| 02 | 半导体与集成电路 | 半导体设备采购指数 | quarterly | upstream_cost (点) | 无 | N/A (无接口) | **C** | **P4** | Tushare无前道设备出货与资本开支指数，需SEMI季度出货报告 |
| 03 | 半导体与集成电路 | DRAM存储芯片现货价 | daily | downstream_demand (美元) | 无 | N/A (无接口) | **C** | **P4** | Tushare无DRAM/NAND现货报价接口，需InSpectrum/TrendForce数据源 |
| 04 | 人工智能与算力服务 | HBM高带宽内存价格 | monthly | upstream_cost (美元/GB) | 无 | N/A (无接口) | **C** | **P4** | Tushare无HBM/先进封装内存价格接口，需专业行研报告 |
| 05 | 人工智能与算力服务 | IDC机房平均电价 | monthly | upstream_cost (元/千瓦时) | 无 | N/A (无接口) | **C** | **P4** | Tushare无IDC机房电价接口，需各地发改委大工业电价与电网交易数据 |
| 06 | 人工智能与算力服务 | AI服务器季度出货量 | quarterly | downstream_demand (万台) | 无 | N/A (无接口) | **C** | **P4** | Tushare无AI服务器季度出货量接口，需IDC/Canalys季度追踪报告 |
| 07 | 新能源汽车与智能汽车 | 汽车用冷轧板价格 | weekly | upstream_cost (元/吨) | `fut_daily` (ts_code=HC.SHF, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P2** | 用上期所热轧卷板期货主力连续(HC.SHF)收盘价作为汽车钢板原材料近似代理 |
| 08 | 新能源汽车与智能汽车 | 乘联会乘用车月度批发销量 | monthly | downstream_demand (万辆) | 无 | N/A (无接口) | **C** | **P3** | Tushare无乘联会/中汽协乘用车月度销量接口，需CPCA/CAAM官方数据源 |
| 09 | 光伏与储能系统 | 光伏级白银价格 | daily | upstream_cost (元/千克) | `sge_daily` (ts_code=Ag99.99, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, close 等14列] | **B** | **P1** | 用上海黄金交易所现货白银(Ag99.99)日行情代理光伏导电银浆核心白银原料成本 |
| 10 | 光伏与储能系统 | 光伏组件单月出口金额 | monthly | downstream_demand (亿美元) | 无 | N/A (无接口) | **C** | **P3** | Tushare无海关光伏组件细分出口统计接口，需海关总署月度商品HS编码数据 |
| 11 | 动力电池与储能电池材料 | 六氟磷酸锂价格 | weekly | upstream_cost (万元/吨) | `fut_daily` (ts_code=LC.GFE, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P2** | Tushare无六氟磷酸锂细分接口，可用广期所碳酸锂期货主力(LC.GFE)代理源头锂盐成本 |
| 12 | 动力电池与储能电池材料 | 全球新型储能新增装机量 | quarterly | downstream_demand (GWh) | 无 | N/A (无接口) | **C** | **P4** | Tushare无全球新型储能装机容量统计接口，需CNESA/BNEF行业报告 |
| 13 | 医药生物与创新药 | 医药中间体价格指数 | monthly | upstream_cost (点) | 无 | N/A (无接口) | **C** | **P4** | Tushare无医药中间体及精细化工原料价格指数 |
| 14 | 医药生物与创新药 | 实验动物价格 | monthly | upstream_cost (万元/只) | 无 | N/A (无接口) | **C** | **P4** | Tushare无食蟹猴/实验小鼠等实验模型价格接口 |
| 15 | 医药生物与创新药 | 中国创新药海外授权License-out总额 | quarterly | downstream_demand (亿美元) | 无 | N/A (无接口) | **C** | **P4** | Tushare无创新药License-out交易统计接口，需医药魔方/动脉网 |
| 16 | 医疗器械与医疗服务 | 医用级钛合金价格 | monthly | upstream_cost (万元/吨) | 无 | N/A (无接口) | **C** | **P4** | Tushare无医用金属/钛合金现货价格接口 |
| 17 | 医疗器械与医疗服务 | 高精度传感器元器件成本 | quarterly | upstream_cost (点) | 无 | N/A (无接口) | **C** | **P4** | Tushare无医疗器械专用传感器及芯片成本指数 |
| 18 | 医疗器械与医疗服务 | 高值医用耗材集采采购量 | quarterly | downstream_demand (万套) | 无 | N/A (无接口) | **C** | **P4** | Tushare无国家/省际耗材集采中标量统计接口 |
| 19 | 消费电子与智能终端 | 显示面板主流尺寸报价 | monthly | upstream_cost (美元/片) | 无 | N/A (无接口) | **C** | **P3** | Tushare无LCD/OLED面板报价接口，需WitsView/群智咨询数据源 |
| 20 | 消费电子与智能终端 | 全球PC出货量增速 | quarterly | downstream_demand (%) | 无 | N/A (无接口) | **C** | **P4** | Tushare无全球PC/笔电季度出货量统计，需IDC/Canalys报告 |
| 21 | 白酒与精制茶酒 | 酿酒高粱原粮收购均价 | monthly | upstream_cost (元/公斤) | 无 | N/A (无接口) | **C** | **P4** | Tushare无专用酿酒红缨子高粱收购价接口 |
| 22 | 白酒与精制茶酒 | 白酒包材纸箱玻璃成本指数 | monthly | upstream_cost (点) | `fut_daily` (ts_code=SP.SHF, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P3** | 用上期所纸浆期货主力连续(SP.SHF)收盘价作为包装纸箱原材料成本代理 |
| 23 | 白酒与精制茶酒 | 烟酒店与商超白酒渠道动销率 | monthly | downstream_demand (%) | 无 | N/A (无接口) | **C** | **P4** | Tushare无白酒线下渠道开瓶动销率数据 |
| 24 | 大众食品与饮料 | 生鲜乳主产区收购均价 | weekly | upstream_cost (元/公斤) | 无 | N/A (无接口) | **C** | **P3** | Tushare无农业农村部主产区生鲜乳收购均价接口 |
| 25 | 大众食品与饮料 | 白糖大宗现货价格 | daily | upstream_cost (元/吨) | `fut_daily` (ts_code=SR.ZCE, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P1** | 用郑商所白糖期货主力连续(SR.ZCE)收盘价代理现货价，严禁冒充真值现货 |
| 26 | 大众食品与饮料 | 量贩零食渠道月度出货额 | monthly | downstream_demand (亿元) | 无 | N/A (无接口) | **C** | **P4** | Tushare无量贩零食/折扣超市终端出货额接口 |
| 27 | 家用电器与智能家居 | ABS塑料颗粒现货价 | daily | upstream_cost (元/吨) | `fut_daily` (ts_code=PP.DCE, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P2** | 用大商所聚丙烯期货主力连续(PP.DCE)收盘价代理家电注塑外壳塑料成本 |
| 28 | 家用电器与智能家居 | 家电产品月度出口金额增速 | monthly | downstream_demand (%) | 无 | N/A (无接口) | **C** | **P3** | Tushare无家电品类细分出口额统计接口，需海关总署月报 |
| 29 | 商业银行与信贷 | 银行间同业拆借利率Shibor | daily | upstream_cost (%) | `shibor` (start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [date, on, 1w 等9列] | **A** | **P1** | Tushare官方shibor接口直接提供全国银行间同业拆借中心官方Shibor(3M)真值 |
| 30 | 商业银行与信贷 | 贷款市场报价利率LPR_1Y | monthly | downstream_demand (%) | `shibor_lpr` (start_date=20240101, end_date=20240630) | code: 0, rows: 6, act_as_of: 20240620, fields: [date, 1y, 5y] | **A** | **P1** | Tushare官方shibor_lpr接口直接提供央行/同业拆借中心1年期LPR官方真值 |
| 31 | 证券公司与资本市场 | 转融通融券利率中枢 | daily | upstream_cost (%) | 无 | N/A (无接口) | **C** | **P3** | Tushare仅有margin/margin_detail融资融券交易量，无中证金融转融通拆借利率 |
| 32 | 证券公司与资本市场 | 券商短融发债票面利率 | monthly | upstream_cost (%) | 无 | N/A (无接口) | **C** | **P4** | Tushare无券商短融券平均发行票面利率专项统计 |
| 33 | 保险与多元金融 | 中国10年期国债到期收益率 | daily | upstream_cost (%) | `yc_cb` (ts_code=1001.CB, curve_term=10, start_date=20240101, end_date=20240110) | code: 403, msg: 请联系管理员添加此权限 | **D** | **P1** | Tushare官方有中债国债收益率曲线接口yc_cb(1001.CB/10Y)，但当前token返回code 403权限不足 |
| 34 | 保险与多元金融 | 财险车险单月保费增速 | monthly | downstream_demand (%) | 无 | N/A (无接口) | **C** | **P3** | Tushare无金融监管总局财险/车险单月保费统计接口 |
| 35 | 钢铁与黑色金属 | 普氏铁矿石价格指数 | daily | upstream_cost (美元/吨) | `fut_daily` (ts_code=I.DCE, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P1** | 用大商所铁矿石期货主力连续(I.DCE)收盘价代理铁矿石现货/普氏指数走势 |
| 36 | 钢铁与黑色金属 | 主焦煤港口平仓价 | daily | upstream_cost (元/吨) | `fut_daily` (ts_code=JM.DCE, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P1** | 用大商所焦煤期货主力连续(JM.DCE)收盘价代理主焦煤现货港口平仓价 |
| 37 | 钢铁与黑色金属 | 五大品种钢材社会与钢厂总库存 | weekly | downstream_demand (万吨) | 无 | N/A (无接口) | **C** | **P2** | Tushare无Mysteel全口径五大品种钢材社会及钢厂总库存接口，fut_wsr仅为交割仓单 |
| 38 | 有色金属与工业金属 | 铜精矿现货加工费TC | weekly | upstream_cost (美元/吨) | 无 | N/A (无接口) | **C** | **P3** | Tushare无铜精矿现货加工费TC/RC接口，需安泰科/SMM数据源 |
| 39 | 有色金属与工业金属 | 氧化铝现货价格 | daily | upstream_cost (元/吨) | `fut_daily` (ts_code=AO.SHF, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P1** | 用上期所氧化铝期货主力连续(AO.SHF)收盘价代理氧化铝现货价格走势 |
| 40 | 有色金属与工业金属 | 新能源汽车铝合金单车用量 | quarterly | downstream_demand (公斤/辆) | 无 | N/A (无接口) | **C** | **P4** | Tushare无汽车轻量化铝合金单车用量参数接口 |
| 41 | 贵金属与稀缺资源 | SPDR黄金ETF持仓量 | daily | downstream_demand (吨) | `fund_daily` (ts_code=518880.SH, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等11列] | **B** | **P2** | Tushare无海外SPDR持仓吨数，可用国内最大黄金ETF华安黄金(518880.SH)或沪金期货AU.SHF持仓代理 |
| 42 | 石油石化与基础化工 | 动力煤坑口价 | daily | upstream_cost (元/吨) | `fut_daily` (ts_code=ZC.ZCE, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P2** | 用郑商所动力煤期货(ZC.ZCE)或大商所焦煤期货(JM.DCE)代理煤化工原料煤价格 |
| 43 | 石油石化与基础化工 | 聚酯长丝开工率 | weekly | downstream_demand (%) | `fut_daily` (ts_code=TA.ZCE, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P3** | 用郑商所PTA期货主力连续(TA.ZCE)或短纤(PF.ZCE)价格与持仓走势作为聚酯化纤产业链代理 |
| 44 | 煤炭与传统化石能源 | 煤矿智能化采掘设备投入 | annual | upstream_cost (亿元) | 无 | N/A (无接口) | **C** | **P4** | Tushare无煤矿智能化改造及采掘设备投资统计接口 |
| 45 | 煤炭与传统化石能源 | 秦皇岛港动力煤平仓价 | daily | downstream_demand (元/吨) | `fut_daily` (ts_code=ZC.ZCE, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P2** | 用郑商所动力煤期货(ZC.ZCE)或大商所焦煤期货(JM.DCE)代理动力煤平仓价走势 |
| 46 | 电力与公用事业 | 电煤长协入厂标煤单价 | monthly | upstream_cost (元/吨) | `fut_daily` (ts_code=ZC.ZCE, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P2** | 用动力煤期货(ZC.ZCE)或焦煤期货(JM.DCE)代理火电燃料成本走势 |
| 47 | 电力与公用事业 | 天然铀现货价格 | monthly | upstream_cost (美元/磅) | 无 | N/A (无接口) | **C** | **P4** | Tushare无国际天然铀U3O8现货价格接口，需UxC/TradeTech报告 |
| 48 | 电力与公用事业 | 长江三峡入库流量月度均值 | daily | downstream_demand (立方米/秒) | 无 | N/A (无接口) | **C** | **P3** | Tushare无水利枢纽入库/出库流量水文接口，需长江水文网 |
| 49 | 房地产开发与运营 | 房企境内外债券平均发行票息 | monthly | upstream_cost (%) | 无 | N/A (无接口) | **C** | **P4** | Tushare无房企境内外债券平均发行票息统计接口 |
| 50 | 房地产开发与运营 | 30大中城市商品房单周成交面积 | weekly | downstream_demand (万平方米) | 无 | N/A (无接口) | **C** | **P2** | Tushare无30城高频商品房成交面积接口，需克而瑞CRIC/Wind |
| 51 | 建筑装饰与基础设施工程 | 全国水泥均价PO42.5散装 | weekly | upstream_cost (元/吨) | 无 | N/A (无接口) | **C** | **P2** | Tushare无全国/区域水泥均价接口，需数字水泥网/百川盈孚 |
| 52 | 建筑装饰与基础设施工程 | 重交沥青全国出厂均价 | weekly | upstream_cost (元/吨) | `fut_daily` (ts_code=BU.SHF, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P1** | 用上期所石油沥青期货主力连续(BU.SHF)收盘价代理全国重交沥青出厂均价 |
| 53 | 机械设备与工业母机 | 数控系统与高精度伺服电机价格指数 | quarterly | upstream_cost (点) | 无 | N/A (无接口) | **C** | **P4** | Tushare无数控系统及伺服电机成本指数 |
| 54 | 机械设备与工业母机 | 机械铸件与特种生铁价格 | monthly | upstream_cost (元/吨) | 无 | N/A (无接口) | **C** | **P4** | Tushare无特种生铁及机械铸件现货均价接口 |
| 55 | 机械设备与工业母机 | 国内工业机器人月度产量 | monthly | downstream_demand (台) | 无 | N/A (无接口) | **C** | **P3** | Tushare无国家统计局工业机器人产量专项接口，需国家统计局月度宏观报告 |
| 56 | 国防军工与航天装备 | 军工级高温合金与钛合金价格 | monthly | upstream_cost (万元/吨) | 无 | N/A (无接口) | **C** | **P4** | Tushare无军工特种高温合金现货价格 |
| 57 | 国防军工与航天装备 | 宇航级高可靠电子元器件价格指数 | quarterly | upstream_cost (点) | 无 | N/A (无接口) | **C** | **P4** | Tushare无宇航级电子元器件价格指数 |
| 58 | 交通运输与航运港口 | 新加坡保税低硫船用燃料油现货价 | daily | upstream_cost (美元/吨) | `fut_daily` (ts_code=LU.INE, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P1** | 用上海国际能源交易中心低硫燃料油期货主力连续(LU.INE)收盘价代理保税船用燃油价格 |
| 59 | 交通运输与航运港口 | 国内航空煤油出厂价 | monthly | upstream_cost (元/吨) | 无 | N/A (无接口) | **C** | **P3** | Tushare无国内航煤出厂中枢基准价接口，需发改委/金联创数据 |
| 60 | 交通运输与航运港口 | 全国主要沿海港口集装箱吞吐量 | monthly | downstream_demand (万TEU) | 无 | N/A (无接口) | **C** | **P3** | Tushare无交通运输部沿海港口集装箱月度吞吐量接口 |
| 61 | 通信网络与光通信 | 高端高速光芯片价格指数 | quarterly | upstream_cost (点) | 无 | N/A (无接口) | **C** | **P4** | Tushare无高速光芯片/电芯片成本价格指数 |
| 62 | 通信网络与光通信 | 光纤预制棒与光缆集采单价 | monthly | upstream_cost (元/芯公里) | 无 | N/A (无接口) | **C** | **P4** | Tushare无三大电信运营商光缆集采中标价接口 |
| 63 | 通信网络与光通信 | 全球800G及以上高速光模块季度出货量 | quarterly | downstream_demand (万只) | 无 | N/A (无接口) | **C** | **P4** | Tushare无高速光模块出货量统计接口，需LightCounting季度报告 |
| 64 | 农林牧渔与生猪养殖 | 国内玉米现货均价 | daily | upstream_cost (元/吨) | `fut_daily` (ts_code=C.DCE, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P1** | 用大商所玉米期货主力连续(C.DCE)收盘价代理现货均价，严禁冒充真值现货 |
| 65 | 农林牧渔与生猪养殖 | 全国豆粕现货均价 | daily | upstream_cost (元/吨) | `fut_daily` (ts_code=M.DCE, start_date=20240101, end_date=20240110) | code: 0, rows: 7, act_as_of: 20240110, fields: [ts_code, trade_date, pre_close 等15列] | **B** | **P1** | 用大商所豆粕期货主力连续(M.DCE)收盘价代理现货均价，严禁冒充真值现货 |

---

## 三、 重点核实指标专项审查

依据需求指令，对 14 个核心重点指标进行逐一审查核验：

### 3.1 宏观利率与金融市场类
1. **银行间同业拆借利率 Shibor（#29）**
   - **Tushare 接口**：`shibor`（参数 `start_date`, `end_date`）
   - **探针实测**：`code: 0, rows: 7, actual_as_of: 20240110, fields: [date, on, 1w, 2w, 1m, 3m, 6m, 9m, 1y]`
   - **定级**：**A（可直接真值替代）**
   - **说明**：提供全国银行间同业拆借中心官方 Shibor 利率各期限真值，可无缝替代。

2. **贷款市场报价利率 LPR_1Y（#30）**
   - **Tushare 接口**：`shibor_lpr`（参数 `start_date`, `end_date`）
   - **探针实测**：`code: 0, rows: 6, actual_as_of: 20240620, fields: [date, 1y, 5y]`
   - **定级**：**A（可直接真值替代）**
   - **说明**：提供央行及同业拆借中心官方公布的 1年期与5年期以上 LPR 基准利率真值。

3. **中国10年期国债到期收益率（#33）**
   - **Tushare 接口**：`yc_cb`（参数 `ts_code="1001.CB"`, `curve_term=10`, `curve_type=0`）
   - **探针实测**：`code: 403, msg: 请联系管理员添加此权限`
   - **定级**：**D（有接口但当前权限不足）**
   - **说明**：Tushare 官方具备完整中债国债收益率曲线（doc_id=201），但当前 Token 权限不足返回 403；需联系管理员升级权限或临时保留外部数据源 fallback。

### 3.2 大宗商品与工业原材料类（严守期货代理原则）
4. **白糖大宗现货价格（#25）**
   - **Tushare 接口**：`fut_daily`（参数 `ts_code="SR.ZCE"`）
   - **探针实测**：`code: 0, rows: 7, actual_as_of: 20240110, fields: [ts_code, trade_date, pre_close, pre_settle, open, high, low, close...]`
   - **定级**：**B（可用近似代理但需用户批准）**
   - **说明**：郑商所白糖期货主力连续合约收盘价可高度反映糖类供需与价格趋势，严禁冒充现货真值。

5. **国内玉米现货均价（#64）**
   - **Tushare 接口**：`fut_daily`（参数 `ts_code="C.DCE"`）
   - **探针实测**：`code: 0, rows: 7, actual_as_of: 20240110`
   - **定级**：**B（可用近似代理但需用户批准）**
   - **说明**：大商所玉米期货主力连续收盘价作为能量饲料成本代理。

6. **全国豆粕现货均价（#65）**
   - **Tushare 接口**：`fut_daily`（参数 `ts_code="M.DCE"`）
   - **探针实测**：`code: 0, rows: 7, actual_as_of: 20240110`
   - **定级**：**B（可用近似代理但需用户批准）**
   - **说明**：大商所豆粕期货主力连续收盘价作为蛋白饲料成本代理。

7. **氧化铝现货价格（#39）**
   - **Tushare 接口**：`fut_daily`（参数 `ts_code="AO.SHF"`）
   - **探针实测**：`code: 0, rows: 7, actual_as_of: 20240110`
   - **定级**：**B（可用近似代理但需用户批准）**
   - **说明**：上期所氧化铝期货主力连续合约收盘价作为电解铝上游成本代理。

8. **新加坡保税低硫船用燃料油现货价（#58）**
   - **Tushare 接口**：`fut_daily`（参数 `ts_code="LU.INE"` 或 `FU.SHF`）
   - **探针实测**：`code: 0, rows: 7, actual_as_of: 20240110`
   - **定级**：**B（可用近似代理但需用户批准）**
   - **说明**：上海国际能源交易中心低硫燃料油期货主力连续收盘价作为国际航运船用燃油成本代理。

9. **重交沥青全国出厂均价（#52）**
   - **Tushare 接口**：`fut_daily`（参数 `ts_code="BU.SHF"`）
   - **探针实测**：`code: 0, rows: 7, actual_as_of: 20240110`
   - **定级**：**B（可用近似代理但需用户批准）**
   - **说明**：上期所石油沥青期货主力连续收盘价作为道路施工沥青出厂均价代理。

10. **煤炭产业链相关指标（#36, #42, #45, #46）**
    - **指标明细**：主焦煤港口平仓价（#36）、动力煤坑口价（#42）、秦皇岛港动力煤平仓价（#45）、电煤长协入厂标煤单价（#46）
    - **Tushare 接口**：`fut_daily`（大商所焦煤 `JM.DCE`、焦炭 `J.DCE`、郑商所动力煤 `ZC.ZCE`）
    - **探针实测**：`JM.DCE` 与 `ZC.ZCE` 均正常返回 `code: 0, rows: 7`
    - **定级**：**B（可用近似代理但需用户批准）**
    - **说明**：可用煤焦期货收盘价代理煤炭产业链价格走势。

11. **钢材与铁矿石产业链指标（#07, #35, #37）**
    - **普氏铁矿石价格指数（#35）**：`fut_daily`（`I.DCE` 大商所铁矿石期货），探针 `code: 0, rows: 7`，定级 **B**；
    - **汽车用冷轧板价格（#07）**：`fut_daily`（`HC.SHF` 上期所热轧卷板期货），探针 `code: 0, rows: 7`，定级 **B**；
    - **五大品种钢材社会与钢厂总库存（#37）**：Tushare 无 Mysteel 全行业五大材总库存统计，探针 `fut_wsr` 仅为交割仓单，定级 **C（Tushare无对应接口）**。

### 3.3 宏观行业统计与出口类（Tushare缺失）
12. **乘联会乘用车月度批发销量（#08）**
    - **Tushare 状态**：无乘联会 CPCA / 中汽协 CAAM 汽车销量接口
    - **定级**：**C（Tushare无对应接口）**

13. **国内工业机器人月度产量（#55）**
    - **Tushare 状态**：无国家统计局工业机器人规上产量专门接口
    - **定级**：**C（Tushare无对应接口）**

14. **家电产品月度出口金额增速（#28）**
    - **Tushare 状态**：无海关总署细分家电出口额统计接口
    - **定级**：**C（Tushare无对应接口）**

---

## 四、 分类实施指引与系统演进建议

1. **A 类指标即刻实施方案（无缝真值升级）**：
   - 将 `商业银行与信贷` 中的 `银行间同业拆借利率Shibor` (`Shibor_3M`) 与 `贷款市场报价利率LPR_1Y` 数据源从 `pending_api` 升级为 `tushare`；
   - 依据 `IndustryLinkageProvider` 标准防前视纪律解析 `date` / `1y` / `3m` 字段，计算环比与趋势。

2. **B 类代理指标用户授权规范（明确代理属性）**：
   - 对于白糖 (`SR.ZCE`)、玉米 (`C.DCE`)、豆粕 (`M.DCE`)、氧化铝 (`AO.SHF`)、燃料油 (`LU.INE`)、沥青 (`BU.SHF`)、铁矿石 (`I.DCE`) 等 18 个 B 类指标；
   - 必须在元数据中明确标注 `is_proxy=True` 及 `proxy_target="期货主力收盘价代理现货"`，并在前端展示与报告渲染时显式向用户声明，严禁冒充真值。

3. **D 类权限申请与降级保护**：
   - 针对 `中国10年期国债到期收益率` (`yc_cb`)，建议在 Tushare 账户开通相关权限（或增补中债收益率曲线权限）；在权限开通前，保持现有 fallback 机制。

4. **C 类指标多源拓展规划**：
   - 针对汽车销量、工业机器人产量、家电/光伏出口等统计局与海关数据，建议在后续规划中引入 AkShare / 官方公开数据爬虫作为专用拓展 Provider。
