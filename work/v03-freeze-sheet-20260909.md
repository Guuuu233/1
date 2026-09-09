# V-03 收益对照实验 · 冻结表（2026-09-09，全部冻结）

关联 DAV-799（V-03 设计审定）。**V-03a 现被 Canonical Symbol Normalization blocker 卡阻塞**——该卡完成前不派 V-03a；H1b/收益实验不解锁。

## ✅ 已冻结（David 2026-09-09）

### 入场与成本
| 项 | 值 |
|---|---|
| 入场价 | **次日开盘 Open**（信号 T 日，T+1 开盘成交） |
| 成本口径 | **佣金含交易规费**：总成本 = 佣金 + 过户费 + 印花税(卖)，**不再另加经手费/证管费**（防重复计） |
| 滑点 | **固定单边 5bps** |
| 简单基准 | **沪深300** |

### 成本费率（Codex 从官方源核实，2026）
印花税 0.5‰ 卖方单边 / 经手费 0.0341‰ 双 / 证管费 0.002% 双 / 过户费 0.01‰ 双 / 券商佣金 ≤3‰ 最低5元（按账户协议，`0.025%` 仅作显式账户假设）。口径已按"佣金含规费"，模型中佣金外只加过户费+印花税。

### 1. OOS 三段切分（David 修正 Claude 的"全 2026 OOS"）
- `DEV ≤ 2025-12-31`
- `HISTORICAL_OOS = 2026-01-01 ~ 2026-09-08`
- `FORWARD_OOS ≥ 2026-09-09`
- **冻结后不得基于 OOS 结果调参再报告同一 OOS**。理由：Claude/agent 已看过大量 2026 结果，2026 不再是纯净未来样本，只能算 historical OOS；**唯一干净的前向验证集是今日起新增的数据（FORWARD_OOS）**。

### 2. 股票池
- **canonical symbol 先行**（见下 blocker）。目标格式 `000001.SZ / 600000.SH / 688xxx.SH / 300xxx.SZ`；空值/无法映射值 **报错或隔离，禁止静默 join**。
- 主样本：沪深 A 股普通股，含 **主板 + 创业板 + 科创板**；排除 **ST/*ST、北交所、上市 <60 交易日**。
- **执行可行性规则**：T+1 执行日停牌 / 涨跌停封死 / 实际不可交易 → 标 `untradable`，**禁止假装以 Open 成交**（否则回测偏乐观）。
- 北交所先排除（流动性/涨跌幅/样本结构不同），以后单列 robustness cohort。

### 3. 蓝思 typed-gap → `typed-missing`
- 语义：`outcome_status = typed_missing`，`return = NULL`，`included_in_return_metrics = false`，`included_in_coverage_metrics = true`，记录 `missing_reason`。
- **禁止 carry-forward**（凭空造价格=价格不变假设）；**禁止静默 drop**（分母消失→选择偏差，且无法区分"模型预测失败"与"市场数据不可观测"）。
- 收益指标不被污染；coverage/evaluability 指标反映"发出多少预测、其中多少可事后验证"。

## ⛔ V-03a 前置 blocker：Canonical Symbol Normalization

实测刻画（`data/tradingagents.db` reports，1408 条）：
- 空值 `''` **32 条**；非法值 **2 种**（`AGENT`、`AUUSDO`）——共 34 行必须显式隔离并计数
- 带后缀 1370、裸码 6；85 个 6 位核心，**3 处 collision**：`000001`/`000001.SZ`、`600519`/`600519.SH`、`603259`/`603259.SH`
- 危害：collision 使同一股票在 groupby/去重/收益对齐/股票池过滤/预测计数/agent credit/校准统计中被劈成两份——**不报错，只给"数学正常、事实错误"的结果**。

**David 定：此为独立 blocker 卡，非顺手修；V-03a 依赖其完成。**

✅ **DAV-800 交付 `5a00c753694792bde8cc213d08ae36211afd8792` 并经总控亲验（2026-09-10）**：第一父=fab99d9、纯新增 1645 行不动现有代码、守恒 1408（1370 已规范+4 裸码补齐+34 隔离，零静默丢弃）、3 collision 合并（茅台740/平安59/药明12，裸码残留0）、34 隔离带原因、21 定向测试全过、真全量 `18 failed/3950 passed` 失败集=基线**零新增**。待 David 合入 trunk。**注：合代码不改数据；实际跑迁移清库是另一步，写生产库需 David 单独授权 + 先备份。**

## 基线冻结（事实性）—— 实测核实（2026-09-09，David + 总控双验）

**模型来源 = DB `role_bindings` → `model_profiles`（权威），env 仅兜底。**（总控最初只看 env 属误；已更正。）
- 15 角色干净绑 **`gemini-3.8-flash-high`**。
- ✅ **3 角色"冲突"已解析（2026-09-10 查 created_at）**：`bear_researcher`/`bull_researcher`/`research_manager` 各有两条绑定——旧（2026-07-28：deepseek-r1 / gemini-2.5-flash / qwen-max）+ 新（2026-09-03：gemini-3.8-flash-high）。**2026-09-03 全部重绑 gemini-3.8-flash-high，旧行未删（stale）。最新生效 → 全 18 角色有效模型 = `gemini-3.8-flash-high`**（与 David 观察一致）。3 条 07-28 旧行为死数据，不影响运行，另清即可。
  - 注：2026-09-03 是一次配置整合日（重绑模型 + 更新 prompt），故 HISTORICAL_OOS 在 09-03 前后有配置边界——再次印证 FORWARD_OOS 才是完全受控。
- provider `openai`（兼容形），base `http://100.65.130.33:8317/v1`。env 里 `.env=gemini-3.7-flash-high`、`.env.local=gemini-2.5-flash`(未加载)——均被 DB 绑定覆盖。

**Prompt 两层**：
- 内置 agent prompt 在代码 `tradingagents/prompts/zh.py`/`en.py`，SHA 固定。
- DB `user_custom_prompts`：1 条 `enabled=1`、`target=global`、注入位 `after_data`、hash `5489166b`、2838 字符。**自 2026-05-06 起稳定生效**（204 条报告 result_data 带此快照；总控此前据 updated_at 推"9-03 漂移"属误，已更正——hash 早于此）。基线须纳入此 hash 或运行前禁用。

**运行参数**：temperature `0`（代码默认）；max_tokens 未传（`.env.local` 的 12000 不生效）；max_retries `5`；timeout `300s`；**seed 未进 LLM 调用链**；辩论/风险轮数各 `3`。

**可追溯性**：`reports` 无独立 model/prompt 列，靠 `result_data` 快照；仅部分报告有（此 hash 204/1408）。HISTORICAL_OOS 部分可追溯、非全部；FORWARD_OOS（配置冻结后）才是完全受控。

### 基线冻结待办（David/Codex，DAV-800 后）
- **去重 3 条冲突角色绑定**（bear/bull/research_manager），钉死单一模型
- 决定 global 自定义 prompt（hash `5489166b`）：纳入基线 or 运行前禁用
- 显式钉死 temperature=0 / max_tokens / seed(无) / 辩论轮数=3
- 代码 SHA `fab99d9` + 运行服务 SHA（现 `a6d4540`，落后一跳，需升 fab99d9）
- ⚠️ 建议另起卡：给 `reports` 增记每条 generating model/prompt hash（否则未来仍不可审计）

## 🚧 系统完整度 —— 决定 V-03 定性的关键前提（David 2026-09-10 指出）

**系统尚未施工完成**：舆情等众多数据源未正式接入，当前报告是半成品。实测填充率（2026-07 起 788 份）：
- `game_theory_report` **0%**（完全未接）
- 舆情/新闻/行情/基本面/宏观/主力 ~70%（且舆情真实数据源未接，填充≠已接）
- `volume_price_report` 55%；**约 30% 报告普遍缺项**

**结论**：对当前系统跑收益/校准，得到的是"半成品机器"的表现，**不能作为"AI 能否盈利"的定性判断**——任何"不赚钱"都可能只因输入没接齐。

**V-03 的正确定位（待 David 定）**：
- 选项A：**现在建测量工具**，当前跑作**显式标注"半成品起点/进度基线"**（记录哪些输入 off），随系统逐步接入再复测，看每接一块是否真的提升；真正定性等系统建完（FORWARD_OOS）。
- 选项B：**推迟 V-03**，等舆情/博弈论等接入后再首测，让第一个数就有意义。
- 无论哪个，基线元数据必须记录"当时系统完整度/哪些输入 off"，否则结果不可诚实解读。
