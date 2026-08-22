# 七分析师深度联想静态/契约审计报告（基线 1eb280b）

> **审计基准 Commit**: `1eb280b35f436c2ff1ece00a448ad7483c86eff9`  
> **对比权威文献**: 
> 1. 《TradingAgents-AShare 深度联想能力增强方案（2026-08-20版）》（`/Users/davidliu/Downloads/项目加强方案`）
> 2. 《TradingAgents-AShare 深度联想能力诊断报告》（`/Users/davidliu/Downloads/项目诊断` / `项目完成度评估.md`）  
> **审计对象**: `macro`（宏观）、`fundamentals`（基本面）、`news`（新闻）、`sentiment`（社交舆情）、`market`（技术分析）、`smart_money`（主力资金）、`volume_price`（量价分析）七大分析师。  
> **代码修改**: 0 行业务代码变更（纯只读静态与契约审计）。

---

## 一、两阶段拓扑架构与实质可见性审计

### 1.1 拓扑定义与 LangGraph Barrier 机制
在 `tradingagents/graph/setup.py:228-253` 中，两阶段拓扑定义如下：
- **第一阶段（Phase 1 并行组）**: `macro`（宏观）、`market`（技术）、`social`（情绪）直接连接 `START` (`setup.py:241`)。
- **阶段同步屏障（Barrier）**: `phase1_dones = ['Macro Analyst Done', 'Market Analyst Done', 'Social Media Analyst Done']`。
- **第二阶段（Phase 2 并行组）**: `fundamentals`、`news`、`smart_money`、`volume_price` 均通过 `workflow.add_edge(phase1_dones, f"{analyst_display_name(analyst_type)} Analyst")` 接入 (`setup.py:243`)。
- **进入辩论层（Debate Barrier）**: `phase2_dones` 全部完成后接入 `Bull Researcher` (`setup.py:244`)。

```
[START]
   ├──> Macro Analyst (Phase 1) ────────> Macro Analyst Done ──────┐
   ├──> Market Analyst (Phase 1) ───────> Market Analyst Done ─────┼──> [LangGraph Barrier]
   └──> Social Media Analyst (Phase 1) ─> Social Analyst Done ─────┘           │
                                                                               ├──> Fundamentals Analyst (Phase 2) ──> Done ──┐
                                                                               ├──> News Analyst (Phase 2) ──────────> Done ──┼──> Bull Researcher
                                                                               ├──> Smart Money Analyst (Phase 2) ───> Done ──┤    (Debate)
                                                                               └──> Volume Price Analyst (Phase 2) ──> Done ──┘
```

### 1.2 实质数据流与报告可见性核查
在 LangGraph 运行时，Phase 1 节点执行后将其产物写回 `AgentState`：
- `macro_analyst.py:256` 写入 `state["macro_report"]`
- `market_analyst.py:199` 写入 `state["market_report"]`
- `social_media_analyst.py:157` 写入 `state["sentiment_report"]`

Phase 2 中的四个分析师均调用了 `tradingagents/agents/utils/context_utils.py:280-297` 中的 `format_phase1_reports(state)`：
1. `fundamentals_analyst.py:149, 153`：将 `phase1_reports_text` 注入 `human_content_blocks`。
2. `news_analyst.py:94, 98`：将 `phase1_reports_text` 注入 `human_content_blocks`。
3. `smart_money_analyst.py:126, 136`：将 `phase1_reports_text` 注入 `HumanMessage`。
4. `volume_price_analyst.py:45, 50`：将 `phase1_reports_text` 注入 `HumanMessage`。

**结论**：第二阶段分析师**真实且确定性地**接收到了第一阶段三大分析师的完整产物，并非仅有 edge 名称。

### 1.3 核心架构断层与并发盲区（Defect）
1. **同阶段并发盲区（Intra-Stage Blindness）**：
   - `fundamentals_system_message`（`prompts/zh.py`）明确要求基本面分析师：“*必须主动从宏观/新闻分析师报告中提取原材料价格走势、行业政策催化与下游需求预测*”。
   - 但在真实运行时，`fundamentals` 与 `news` 同属 Phase 2 并行执行，`fundamentals` 执行时 `state["news_report"]` 尚未生成，`format_phase1_reports` 仅包含宏观、技术和情绪。
   - **结果**：基本面分析师**永远无法看到**新闻分析师的报告，提示词契约与实际执行拓扑存在矛盾。
2. **Phase 1 内部孤岛**：
   - Phase 1 内部 `macro`、`market`、`social` 完全并行，三者互不可见。宏观分析师无法感知大盘技术面，情绪分析师亦无法感知宏观数据。

---

## 二、七分析师逐角色全维度静态与契约审计矩阵

### 2.1 综合对照审计总表

| 角色 / 文件 | 阶段 | 输入数据源 | Phase 1 上下文注入 | 产业链指标 (Linkage) | RAG 行业/宏观图谱 | 历史案例复盘 (Cases) | What/Why/SoWhat/WhatNext | 量化传导与敏感性 | 时滞标注 | 数据缺失纪律 | 跨报告交叉引用 | 综合评级 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Macro Analyst** (`macro_analyst.py:55`) | P1 | 全球指数/大类资产/国内指数/北向资金/板块资金/要闻 | 无 (自身P1) | ✅ `format_industry_linkage_for_prompt` | ✅ 27行业+19宏观情景 | ✅ 3条历史案例 | ✅ 严格 4 步分层 | ✅ 5大行业强制清单与量化弹性 | ✅ 明确2-6月/1-3月时滞 | ✅ 严格【数据缺失】禁止平滑 | ❌ P1无上游报告可引 | **92分 (达标)** |
| **Fundamentals Analyst** (`fundamentals_analyst.py:54`) | P2 | 四张财务报表/宏观大盘/大类资产/产业链 | ✅ 包含宏观/技术/情绪 | ✅ `format_industry_linkage_for_prompt` | ✅ 27行业+19宏观情景 | ✅ 3条历史案例 | ⚠️ 融入7项执行要求，无显式标题 | ✅ 成本上涨X%→毛利Y%敏感性 | ✅ 议价权账期与周期时滞 | ✅ 严格【数据缺失】指明报表字段 | ⚠️ 引用宏观/情绪(实有)，新闻(并发缺失) | **85分 (良好)** |
| **News Analyst** (`news_analyst.py:35`) | P2 | 个股新闻/全球宏观新闻 | ✅ 包含宏观/技术/情绪 | ❌ 未注入 Linkage 结构化指标 | ✅ 27行业+19宏观情景 | ❌ 未注入历史案例 | ✅ 严格 4 步 + 强制三层传导 | ✅ 强制渗透率/稼动率/ASP测算 | ✅ 明确即时/1-3月/半年时滞 | ⚠️ 提示词有要求但无硬后处理守卫 | ✅ 引用宏观/大盘/情绪 | **78分 (良好)** |
| **Social Media Analyst** (`social_media_analyst.py:22`) | P1 | 涨停池/雪球热股/个股新闻 | 无 (自身P1) | ❌ 未注入 | ❌ 未注入 | ❌ 未注入 | ✅ 严格 4 步分层 | ⚠️ 情绪温度量化，无产业链量化 | ✅ 1-3天/1-2周/1月 | ❌ 提示词缺少严格缺失语法 | ❌ P1无上游报告可引 | **65分 (及格)** |
| **Market Analyst** (`market_analyst.py:67`) | P1 | 日线 OHLCV / 10大技术指标 / 实时快照 | 无 (自身P1) | ❌ 未注入 | ❌ 未注入 | ❌ 未注入 | ✅ 严格 4 步分层 | ⚠️ 多因子合成与支撑阻力，无产业链传导 | ✅ 多周期路径推演时效 | ⚠️ 依赖 market_data_context 状态 | ❌ P1无上游报告可引 | **68分 (及格)** |
| **Smart Money Analyst** (`smart_money_analyst.py:29`) | P2 | 个股资金流/龙虎榜/VWMA/资金流证据链 | ✅ 包含宏观/技术/情绪 | ❌ 未注入 | ❌ 未注入 | ❌ 未注入 | ✅ 严格 4 步分层 | ⚠️ 筹码控制与分单测算，非商品链 | ✅ 资金周期4阶段时滞 | ✅ 极其严格：`consensus_guard` 硬阻断 | ✅ 引用宏观/大盘/情绪 | **80分 (良好)** |
| **Volume Price Analyst** (`volume_price_analyst.py:17`) | P2 | VPA 预计算指标 / 原始 K 线 | ✅ 包含宏观/技术/情绪 | ❌ 未注入 | ❌ 未注入 | ❌ 未注入 | ❌ 缺失 4 步框架 (采用威科夫理论) | ⚠️ 供求定律与量价异常，无产业链传导 | ⚠️ 包含阶段演化，无量化时滞 | ❌ 提示词缺失严格缺失要求 | ⚠️ 注入但提示词未指导如何结合宏观 | **58分 (不达标)** |

---

### 2.2 逐角色深度审计详情

#### 1. 宏观与板块分析师 (`macro_analyst.py`)
- **文件与行号**: `tradingagents/agents/analysts/macro_analyst.py:55-267`，Prompt 位于 `tradingagents/prompts/zh.py:PROMPTS["macro_system_message"]`。
- **输入与上下文**:
  - 输入：板块资金流、个股新闻、全球要闻、全球核心指数、大类资产与大宗商品、国内指数、北向资金 (`macro_analyst.py:75-83`)。
  - 知识与数据注入：全维挂载 `resolve_industry_context` (`:135`)、`resolve_macro_event_context` (`:142`)、`resolve_historical_cases_context` (`:147`) 及 `format_industry_linkage_for_prompt` (`:157`)。
- **Prompt 契约与框架**:
  - What/Why/SoWhat/WhatNext 结构清晰，包含全球宏观外溢 (What/Why)、国内货币流动性 (Why/So What)、产业政策传导 (So What/What Next)。
  - **强制 5 大行业传导清单**：消费电子（铜/铝/化工→驱动IC→面板→三星/LG对标）、新能源车（碳酸锂/钴/镍→Pack成本→单车毛利→特斯拉对标）、石化（布伦特/WTI→裂解价差）、金融地产（MLF/LPR→息差）、通用框架。
  - **时滞与量化**：明确要求政策→基本面（2-6个月）、原材料→成本（1-3个月），要求传导方向（正/负）与量级（强/中/弱）。
  - **数据缺失纪律**：明确外盘数据真实性铁律，外盘缺失时必须写明 `【数据缺失】`，严禁平滑改写为“外围平稳”。
- **评分**: **92 / 100**（优秀，完全对齐原始方案与 DAV-192/194 规范）。

#### 2. 基本面分析师 (`fundamentals_analyst.py`)
- **文件与行号**: `tradingagents/agents/analysts/fundamentals_analyst.py:54-256`，Prompt 位于 `tradingagents/prompts/zh.py:PROMPTS["fundamentals_system_message"]`。
- **输入与上下文**:
  - 输入：资产负债表、现金流量表、利润表、财务指标、宏观大类资产与指数、产业链指标 (`fundamentals_analyst.py:74-82`)。
  - 注入：`resolve_industry_context` (`:125`)、`resolve_macro_event_context` (`:132`)、`resolve_historical_cases_context` (`:137`)、`format_industry_linkage_for_prompt` (`:147`) 及 Phase 1 报告 (`:149`)。
- **Prompt 契约与框架**:
  - 覆盖上游议价权（应付账款天数/集中度）、下游需求（合同负债/应收账款）、供需格局（稼动率/Capex/ASP）、国际对标（PE/PB/出货量）。
  - **敏感性测算**：强制要求“成本上涨 X% → 毛利率影响 Y%”、“销量变动 A% → 利润变动 B%”的定量 Stress Test。
  - **数据缺失纪律**：明确指出具体报表与字段缺失（`【数据缺失】XX报表YY指标无法获取`）。
- **主要缺口**:
  - Prompt 要求提取新闻分析师的下游需求预测，但拓扑上与新闻分析师并发，无法获取新闻报告。
- **评分**: **85 / 100**（良好，受同阶段并发拓扑限制）。

#### 3. 新闻与宏观事件分析师 (`news_analyst.py`)
- **文件与行号**: `tradingagents/agents/analysts/news_analyst.py:35-179`，Prompt 位于 `tradingagents/prompts/zh.py:PROMPTS["news_system_message"]`。
- **输入与上下文**:
  - 输入：个股新闻、全球新闻 (`news_analyst.py:53-56`)。
  - 注入：`resolve_macro_event_context` (`:81`)、`resolve_industry_context` (`:86`) 及 Phase 1 报告 (`:94`)。
- **Prompt 契约与框架**:
  - 完整具备 What/Why/SoWhat/WhatNext 结构与**强制三层传导框架**（第一层事件本身 What、第二层直接影响 Why+SoWhat、第三层间接传导与同行/国际映射 WhatNext）。
  - 要求量化推导链路（渗透率、订单量、稼动率、单机 ASP）、传导时滞窗口（即时/1-3月/半年）及区分事实与推断。
- **主要缺口**:
  - 未注入 `historical_cases`（历史案例）与 `industry_linkage`（结构化产业链数据）。
- **评分**: **78 / 100**（良好，三层传导逻辑完备，但缺少历史案例与量化产业链输入）。

#### 4. 社交舆情与市场情绪分析师 (`social_media_analyst.py`)
- **文件与行号**: `tradingagents/agents/analysts/social_media_analyst.py:22-168`，Prompt 位于 `tradingagents/prompts/zh.py:PROMPTS["social_system_message"]`。
- **输入与上下文**:
  - 输入：个股新闻、涨停池连板梯队数据、雪球热门股票排行 (`social_media_analyst.py:41-43`)。
  - 上下文注入：无行业图谱、无宏观情景、无历史案例、无产业链数据。
- **Prompt 契约与框架**:
  - 具备 What/Why/SoWhat/WhatNext 框架，涵盖情绪温度量化、资金性质动因、反身性与生命周期、跨题材联动与极值预警。
  - 包含 1-3天短期脉冲、1-2周波段、1月以上主线时滞。
- **主要缺口**:
  - 完全脱离产业链知识库，缺少数据缺失标准语法定义。
- **评分**: **65 / 100**（及格，专注于微观情绪，但缺少外部产业知识支撑）。

#### 5. 市场技术分析师 (`market_analyst.py`)
- **文件与行号**: `tradingagents/agents/analysts/market_analyst.py:67-211`，Prompt 位于 `tradingagents/prompts/zh.py:PROMPTS["market_system_message"]`。
- **输入与上下文**:
  - 输入：日线 OHLCV、10 个经典技术指标、`market_data_context`（含基准日、日线完整性、实时快照与数据源溯源）(`market_analyst.py:84-92`)。
- **Prompt 契约与框架**:
  - 具备 What/Why/SoWhat/WhatNext 框架，包含价格形态、指标供求博弈、大盘共振（Beta vs Alpha）及多周期交易计划（入场、止损、目标位）。
- **主要缺口**:
  - 仅关注纯行情量价指标，未注入任何行业常识与产业链知识；Prompt 未强制规定指标缺失时的 `【数据缺失】` 标注规则。
- **评分**: **68 / 100**（及格，技术面本身完整，但缺乏跨维度联想能力）。

#### 6. 机构资金行为分析师 (`smart_money_analyst.py`)
- **文件与行号**: `tradingagents/agents/analysts/smart_money_analyst.py:29-338`，Prompt 位于 `tradingagents/prompts/zh.py:PROMPTS["smart_money_system_message"]`。
- **输入与上下文**:
  - 输入：个股资金流、龙虎榜席位明细、VWMA、`fund_flow_evidence` 结构化证据 (`smart_money_analyst.py:50-58`)。
  - 注入：Phase 1 报告 (`:126`)。
- **Prompt 契约与确定性防御**:
  - 具备 What/Why/SoWhat/WhatNext 框架与资金行为经典模型（吸筹/派发/洗盘/主升）。
  - **全系统最严苛的确定性后处理守卫** (`:206-276`)：
    1. 结构化来源选择 (`select_fund_flow_source`) 与严格算法组隔离（区分 Sina 旧 Web 算法与 Eastmoney/THS 新算法）。
    2. 模型输出后置校验 (`validate_model_summary`)，一旦发现数据冲突或未通过方向守卫，硬阻断报告正文并覆写为 fail-closed 提示。
    3. 违规词拦截：若仅有 `netamount`（总净额），硬性拦截“主力吸筹/建仓/增持/减持/派发”等违规用词 (`:248-263`)。
- **主要缺口**:
  - 未挂载行业图谱与宏观情景。
- **评分**: **80 / 100**（良好，防幻觉与数据合规能力极强，但无产业链知识注入）。

#### 7. 量价分析师 (`volume_price_analyst.py`)
- **文件与行号**: `tradingagents/agents/analysts/volume_price_analyst.py:17-113`，Prompt 位于 `tradingagents/prompts/zh.py:PROMPTS["volume_price_system_message"]`。
- **输入与上下文**:
  - 输入：VPA 预计算指标、原始 OHLCV (`volume_price_analyst.py:37-38`)。
  - 注入：Phase 1 报告 (`:45`)。
- **Prompt 契约与框架**:
  - 完整实现 Anna Coulling《量价分析》理论：供求定律、因果定律、投入产出定律，三步分析法，市场循环五大阶段（吸筹/供给测试/派筹/需求测试/放量高峰），关键 K 线信号。
- **致命缺口 (Defects)**:
  1. **完全缺失 What/Why/SoWhat/WhatNext 标准框架**，与其他 6 位分析师结构不统一。
  2. **跨报告融合为零**：虽然在 HumanMessage 中注入了 Phase 1 报告 (`:50`)，但 Prompt 中没有任何一处指示模型如何将宏观/大盘/情绪与量价配合进行分析。
  3. **单元测试遗漏**：在 `test_analyst_prompts_deep_reasoning.py` 中，`ANALYST_PROMPT_KEYS` 和 `test_what_why_sowhat_whatnext_framework_present` 均将 `volume_price_system_message` 漏测。
- **评分**: **58 / 100**（不达标，独立理论完整但跨角色契约与联想严重断裂）。

---

## 三、报告质量闸（Quality Gate）漏洞与自动化验收规则

### 3.1 现有 `report_quality_gate.py` 的“关键词假通过”漏洞
在 `tradingagents/graph/report_quality_gate.py:107-122` 中：
```python
def check_report_keywords(text: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if KEYWORD_REQUIRED_CHAIN not in text:  # "传导"
        reasons.append(f"缺少核心关键词：{KEYWORD_REQUIRED_CHAIN}")
    if not any(k in text for k in KEYWORD_LINKAGE_OPTIONS):  # ("联动", "外溢", "时滞") 之一
        reasons.append(f"缺少核心关键词：{options_str}之一")
    return len(reasons) == 0, reasons
```
**严重缺陷**：
只要大模型生成一句“*宏观政策传导与市场联动效应值得关注*”（空话废话），关键词检查即可 100% 通过！这导致质量闸流于形式，无法阻断浅层复读。

### 3.2 建议重构的确定性多级自动化验收规则（Scoring Rubric）

每个分析师的自动化验收规则必须基于**结构、量化关系、实体与时滞**四重匹配，禁止关键词单独通过：

```python
# 建议落地的新型自动化验收机器评分器规范
RUBRIC_SPEC = {
    "macro": {
        "min_chars": 2000,
        "must_have_sections": ["全球宏观", "货币流动性", "产业政策", "板块轮动", "情景推演"],
        "regex_patterns": [
            r"【全球核心指数】.*?(?:标普|纳斯达克|道琼斯|恒生|日经)",
            r"(?:上涨|下跌|环比|同比)[+-]?\d+(?:\.\d+)?%",  # 必须有量化百分比
            r"时滞.*?(?:\d+[-~至]\d+个?月|\d+个?月)",          # 必须有时滞范围
            r"(?:正相关|负相关).*?(?:强|中|弱)",               # 必须有方向与量级
        ],
        "forbidden_phrases": ["外围平稳", "外盘平稳", "外围中性"],
        "industry_linkage_guard": True,  # 若 context 有 industry_linkage 必须引用
    },
    "fundamentals": {
        "min_chars": 2000,
        "must_have_sections": ["上游议价权", "下游需求", "供需格局", "财务质量", "敏感性分析"],
        "regex_patterns": [
            r"(?:毛利率|净利润|营业收入).*?(?:同比|环比)[+-]?\d+(?:\.\d+)?%",
            r"成本(?:上涨|增加|变动)\s*\d+%\s*→\s*毛利率(?:影响|承压|下降|变动)\s*[±+-]?\d+", # 敏感性推演公式
            r"议价权.*?(?:强|中|弱)",
        ],
    },
    "news": {
        "min_chars": 1500,
        "must_have_sections": ["事件本身", "直接影响", "间接传导"],
        "regex_patterns": [
            r"(?:第一层|事件本身).*?(?:发布时间|核心事实)",
            r"(?:第二层|直接影响).*?(?:基本面|市场情绪)",
            r"(?:第三层|间接传导).*?(?:上游|下游|同行|海外)",
            r"(?:时滞|时效).*?(?:即时|\d+个?月|\d+周)",
        ],
    },
    "smart_money": {
        "min_chars": 1200,
        "must_have_sections": ["资金数据透视", "资金动机", "资金阶段", "板块协同"],
        "fail_closed_guard": True, # fund_flow_consensus_guard 校验
    },
    "volume_price": {
        "min_chars": 1200,
        "must_have_sections": ["威科夫", "量价", "供求", "阶段"],
        "regex_patterns": [
            r"(?:吸筹|拉升|派发|洗盘|震仓)",
            r"(?:确认|异常).*?成交量",
        ],
    }
}
```

---

## 四、测试覆盖与测试缺口清单

### 4.1 现有测试文件清单
- `tests/test_analyst_prompts_deep_reasoning.py` (88 tests 组之一): 测试 Prompt 中的框架关键词与 VERDICT 格式。
- `tests/test_two_stage_analyst_topology.py`: 测试两阶段 Graph 拓扑构建与 Phase 1 报告注入。
- `tests/test_industry_linkage_prompt_injection.py`: 测试产业链数据格式化与 Prompt 拼接。
- `tests/test_report_quality_gate.py`: 测试质量闸关键词与外盘平滑拦截。
- `tests/test_analyst_knowledge_and_macro_views.py`: 测试 RAG 行业/宏观图谱解析与注入。

### 4.2 审计发现的测试缺口（Test Gaps）

1. **Volume Price 分析师测试缺失**:
   - `test_analyst_prompts_deep_reasoning.py:31` 的 `ANALYST_PROMPT_KEYS` 遗漏了 `volume_price_system_message`。
   - `test_what_why_sowhat_whatnext_framework_present` 遗漏了 `volume_price` 的测试。
2. **Phase 2 内部并发真实数据流断言缺失**:
   - `test_two_stage_analyst_topology.py` 仅断言了 Phase 2 分析师接收到了 Phase 1 的输出，但没有测试 Phase 2 分析师之间相互引用的断言（导致没有捕获基本面分析师引用新闻分析师失败的拓扑缺陷）。
3. **质量闸抗平滑与抗空话伪通过测试缺口**:
   - 现存 `test_report_quality_gate.py` 只测试了包含“传导”与“联动”即可返回 `True`，缺乏对“空话文本依然返回 True”的负向用例防护测试。
4. **真实 LLM 输出端到端（E2E）结构化正则断言缺口**:
   - 现有测试多为静态 Prompt 字符串 `assert "关键词" in prompt`，缺少基于 mock LLM 输出文本进行结构化传导公式正则捕获的单元测试。

---

## 五、审计结论与后续实施路线图

### 5.1 审计结论（一句话）
> **基线 `1eb280b` 已成功落地两阶段拓扑架构、全维知识库 RAG/历史案例注入及宏观/基本面/新闻的深度传导 Prompt，Phase 2 实质可见 Phase 1 产物；但存在「基本面-新闻同阶段并发盲区」、「量价分析师框架断层」、「质量闸单纯关键词易被空话绕过」三大核心缺口。**

### 5.2 优化建议与优先级（P0 - P2）

1. **P0（紧急修复）**: 
   - 补齐 `volume_price_system_message` 的 What/Why/SoWhat/WhatNext 框架与跨报告融合指引，并在 `test_analyst_prompts_deep_reasoning.py` 中补齐单测。
   - 修正 `fundamentals_system_message` 中的交叉引用描述（将其引用的新闻分析师调整为第一阶段宏观/情绪分析师，或重构为三阶段拓扑）。
2. **P1（质量闸升级）**:
   - 升级 `report_quality_gate.py`，从简单的关键词存在性检查升级为传导公式正则、时滞区间捕获与敏感性测算的多维结构体验收。
3. **P2（知识库全覆盖）**:
   - 为 `news_analyst`、`smart_money_analyst`、`volume_price_analyst` 适度开放行业图谱（`industry_context`）与历史案例（`historical_cases`）的按需检索注入。
