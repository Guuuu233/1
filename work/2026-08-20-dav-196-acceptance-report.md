# TradingAgents-AShare 阶段二：产业链数据层 MVP 阶段审计与真实验收报告 (DAV-196 / DAV-201)

> **制定日期**：2026-08-20
> **项目主管**：David Liu
> **技术总监**：Hermes Agent
> **开发责任人**：资深开发1
> **测试标的**：京东方A (`000725.SZ`)
> **候选基准分支**：`feature/dav-201-m6-repair`
> **阶段结论**：M1~M5 数据层与单元测试全部通过；M6 真实数据池与 Prompt 注入验证通过，全量 LLM 端到端调用受限于外部 API Key 处于 BLOCKED（诚实记录，拒绝任何伪造数据与文本）。

---

## 一、阶段二整体目标与里程碑验收概览

本项目旨在实现**产业链数据层 MVP**，打通消费电子与新能源车核心赛道的上下游数据感知、指标采集、Prompt 注入与量化传导分析链路，严格遵循零幻觉与数据诚实纪律。

| 里程碑 | 目标与范围 | 责任角色 | 交付产物 | 验收结果 |
| :--- | :--- | :--- | :--- | :---: |
| **M1** | 数据结构定义 (`IndustryLinkageIndicator`, `IndustryLinkage`, `INDUSTRY_LINKAGE_MAP`) | 资深开发1 | `tradingagents/dataflows/industry_linkage.py`<br>`tests/test_industry_linkage_dataflows.py` | ✅ **通过 (PASS)** |
| **M2** | 数据采集器实现 (`IndustryLinkageProvider`, 1h TTL 缓存, 历史 as_of 窗口, 容错降级) | 资深开发1 | `tradingagents/dataflows/providers/industry_linkage_provider.py`<br>`tests/test_industry_linkage_provider.py` | ✅ **通过 (PASS)** |
| **M3** | DataCollector 集成 (`_map_stock_to_industry`, 股票行业映射, 依赖注入) | 资深开发2 | `tradingagents/graph/data_collector.py`<br>`tests/test_industry_linkage_collector.py` | ✅ **通过 (PASS)** |
| **M4** | 分析师 Prompt 注入 (`format_industry_linkage_for_prompt`, 宏观与基本面挂载) | 资深开发2 | `tradingagents/agents/analysts/macro_analyst.py`<br>`tradingagents/agents/analysts/fundamentals_analyst.py` | ✅ **通过 (PASS)** |
| **M5** | 综合单元测试套件 (5 大核心场景覆盖, 离线 Mock, 并发安全) | 代码运维测试员 | `tests/test_industry_linkage.py` | ✅ **通过 (PASS)** |
| **M6** | 端到端可复现真实运行验证 (京东方A 真实数据采集, 真实溯源账本, 拒绝伪造) | 资深开发1 | `scripts/run_dav196_e2e_validation.py`<br>`work/dav196-validation-京东方A.json` | ⚠️ **数据与Prompt层 PASS / LLM真实调用 BLOCKED** |

---

## 二、M6 端到端真实运行核验证据与账本

根据 `scripts/run_dav196_e2e_validation.py` 在当前候选分支的真实执行结果（产物：`work/dav196-validation-京东方A.json`）：

### 1. 真实数据采集溯源 (Data Provenance)
- **标的代码**：京东方A (`000725.SZ`)
- **分析基准日期**：`2026-08-20`
- **行业映射**：`消费电子/半导体显示`
- **数据池总量**：`DataCollector` 成功采集 26 类数据项（包含 `industry_linkage`, `stock_data`, `fundamentals`, `indicators`, `vpa_indicators` 等）。

### 2. 数据状态与缺失账本 (Data Failure / Status Ledger)
- **LME铜价 (上游成本)**：
  - 数据源：`akshare (CAD)`
  - 最新采集值：`14019.50` 美元/吨
  - 月环比 (MoM)：`+0.82%`
  - 季度环比 (QoQ)：`+3.07%`
  - 趋势判定：`平稳`（置信度：高）
- **全球智能手机出货量 (下游需求)**：
  - 数据源：`manual`
  - 采集值：`None`
  - 状态：`【数据缺失】手动`（置信度：低（待手动录入））
- **三星电子股价 (国际对标)**：
  - 数据源：`yfinance (005930.KS)`
  - 采集值：`None`
  - 状态：`【数据缺失】`（置信度：低（接口异常），原因：`Too Many Requests. Rate limited`）

### 3. Prompt 注入真实文本 (Prompt Injection Evidence)
真实生成的 Prompt 注入片段：
```text
【产业链联想数据】：消费电子/半导体显示
- 上游成本端核心指标：
  * LME铜价：14019.50 美元/吨，月环比 +0.82%，季度环比 +3.07%，趋势：平稳（置信度：高）
    - 传导逻辑：核心导电、引线框架与连接件原材料成本传导
- 下游需求端核心指标：
  * 【数据缺失】全球智能手机出货量：手动
    - 传导逻辑：下游终端消费电子需求与换机周期景气度验证
- 国际对标核心标的/指标：
  * 【数据缺失】三星电子股价：数据获取失败: Too Many Requests. Rate limited. Try after a while.
    - 传导逻辑：全球消费电子、存储半导体与显示面板龙头估值与景气度对标
- 行业政策催化关键词：消费品以旧换新、超高清视频产业发展、新型显示产业支持政策
```

### 4. LLM 执行环境真实状态 (Honest Execution Status)
- **状态**：`BLOCKED_NO_API_KEY`
- **事实说明**：当前运行环境未注入有效外部大模型 API Key（`TA_API_KEY` / `OPENAI_API_KEY` 未配置），因此真实 LLM 节点未执行，严格禁止手工编造虚假报告文本与无来源财务数字。

---

## 三、历史 as_of 窗口修复与测试验证

1. **历史查询窗口修复**：
   - 针对三星电子股价等外部接口，当传入指定 `as_of` 时，动态构建 `start_dt = as_of - 120 days` 和 `end_dt = as_of + 1 day` 历史查询窗口，不再固定使用 `period="3mo"`；
   - 对返回数据严格按 `_std_date <= as_of` 截断，杜绝前视偏差。
2. **确定性测试覆盖**：
   - 补充近窗测试、超3个月历史测试、未来行严格截断测试、空源/异常降级测试。

---

## 四、自动化测试套件运行结果

- **产业链专项测试**：`pytest tests/test_industry_linkage*.py -v` → **48 passed (0 failed, 耗时 0.98s)**
- **知识库关联测试**：`pytest tests/test_knowledge_industry_linkage.py tests/test_knowledge_macro_events.py -v` → **153 passed (0 failed, 耗时 0.44s)**
- **代码语法编译检查**：`python -m compileall -q tradingagents tests scripts` → **通过 (0 errors)**
- **代码空白与格式检查**：`git diff --check target/codex/dav-4-p2a-trunk..HEAD` → **通过 (0 errors)**
