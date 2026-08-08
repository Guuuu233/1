# TradingAgents-AShare 全仓审计报告（H1 · 只读）

> 审计范围：仓库全部非 git 文件（**286 个**），含源码 / 测试 / 配置 / 文档 / 脚本。
> 审计基线：`codex/dav-4-p2a-trunk@3c339c5`（M1~M5 已合入）。
> 审计方式：pyflakes / vulture 静态扫描 + 全仓 grep + 人工逐项复核（所有自动结果均经人工验证后才登记，未直接照抄）。
> 审计纪律：**全程只读，未修改任何代码**；`git status` 仅新增本报告文档。
> 日期：2026-08-05
> 更新：2026-08-05 合并 DAV-74 H2 覆盖缺口分析（见 §五），作为 H3 施工依据。

---

## 一、文件清单分类（任务清单 #1）

| 分类 | 数量 | 说明 |
|---|---|---|
| 后端 Python 源码 | 101 | `api/` `tradingagents/` `scheduler/` `scripts/` `conftest.py` `docker-entrypoint.py` |
| 测试 | 62 | `tests/`（含 `conftest.py`） |
| 前端 TS/TSX/JS | 55 | `frontend/src/` |
| 配置文件 | 7 | `requirements.txt` `pyproject.toml` `docker-compose*.yml` `Dockerfile` `.env.example` `.gitignore` |
| 文档 | 12 | `README*` `CHANGELOG.md` `AGENTS.md` `docs/` `guide/` `skills/` `tradingagents/llm_clients/TODO.md` |
| JSON/数据 | 7 | `api/announcements.json` `frontend/public/sponsors.json` 等 |
| 前端配置/锁文件/其他 | 42 | `frontend/*`（含 `package.json` `uv.lock` 等） |
| **合计** | **286** | 与任务描述「约 286 个非 git 文件」一致 |

---

## 二、结论摘要

- **P0（必须修复）：0 条。** 未发现会导致故障/数据损坏/安全事件的问题。
- **P1（应修）：24 条。** 含 2 个运行时 bug、2 个日期炸弹测试、1 个生产路径残留 eval 落盘、1 个无用依赖、1 个打包缺失依赖、13 个确认死代码、5 个前端孤立文件、4 个文档失实。
- **P2（建议）：约 55 条。** 未用导入、重复逻辑、空函数/空文件、弱断言、调试 print、配置与文档细节等。
- **覆盖缺口（H2 合并）：** TOTAL 67%（13091 stmts / 4355 missing），详见 §五。核心低覆盖模块：`job_store_redis.py` 22%、`role_routing_service.py` 11%、`yfinance_news.py` 6%、`alpha_vantage_indicator.py` 3%、`trading_graph.py` 41%、`conditional_logic.py` 21%、`reflection.py` 31%。

---

## 三、P1 问题清单（应修）

### 3.1 运行时 Bug（真实缺陷）

#### P1-1 `api/main.py:108` — `requests` 未导入即使用（匿名版本统计静默失效）
- **位置**：`api/main.py:108`（`_report_version_stats._send()` 内 `requests.post(...)`）
- **问题**：全文件没有任何 `import requests`（pyflakes 报 `undefined name 'requests'`，人工复核确认全仓 grep 无此导入）。该函数在 app 启动时由 `api/main.py:276` 调用。
- **影响**：`NameError` 被 `except Exception: pass` 吞掉，匿名版本统计**从未真正上报过**，且无任何日志痕迹（违反 AGENTS.md「禁止静默失败」）。Telemetry 功能形同虚设。
- **建议**：文件顶部补 `import requests`，并在 `except` 中至少打一条 debug 日志。改动极小、风险低。

#### P1-2 `tradingagents/agents/analysts/volume_price_analyst.py:69` — `asyncio` 未导入（空流 fallback 永远失败）
- **位置**：`volume_price_analyst.py:69`（`res = await asyncio.to_thread(llm.invoke, messages)`）
- **问题**：文件未 `import asyncio`（pyflakes 报 `undefined name 'asyncio'`，人工复核确认同目录其余 6 个 analyst 均导入了，唯独本文件缺失）。
- **影响**：当 LLM 流式返回空文本、走 invoke fallback 时，`NameError` 被 except 捕获，导致显示「分析报告生成失败：name 'asyncio' is not defined」，fallback 分支**永远无法工作**。
- **建议**：补 `import asyncio`。

#### P1-3 `tradingagents/graph/trading_graph.py:350/411` — 生产分析路径无条件向 `eval_results/` 落盘
- **位置**：`trading_graph.py:350`（`propagate` 内无条件调用 `_log_state`）、`:411`（`propagate_async` 内无条件调用 `_log_state_dual`）、`_log_state`/`_log_state_dual` 写盘逻辑 `:462-466`/`:528-533`
- **问题**：`propagate` 由生产 API 路径调用（`api/main.py:2946`），每次单周期分析都会写 `eval_results/<ticker>/TradingAgentsStrategy_logs/full_states_log_<trade_date>.json`（含完整辩论历史、裁决等，可达数百 KB），中周期再写 `dual_horizon_*.json`。该行为**无开关、不读 `TA_TRACE`、也不读文档声明的 `TA_RESULTS_DIR`**（guide/configuration.md:80），是上一轮 eval 仿真器遗留。
- **影响**：生产服务每次分析都在工作目录产生无法清理的 JSON 垃圾，磁盘占用随任务数线性增长；路径以 ticker 为目录名，特殊字符未完全防注入。
- **建议**：删除 `_log_state`/`_log_state_dual` 的写盘（`self.log_states_dict` 内存缓存若无人消费也可一并移除），或改为仅在 `TA_TRACE=1` / 显式开关下写盘，并落盘到 `TA_RESULTS_DIR`。

### 3.2 测试日期炸弹（当前已红）

#### P1-4 `tests/test_vendor_chain_semantics.py:290,309` — 硬编码 `2026-08-04` 日期炸弹（上海时区过零点即失败）
- **位置**：`:290`（`route_to_vendor("get_global_news", "2026-08-04", 7, 10)`）、`:309`（同值），所属用例 `test_router_akshare_global_news_fail_falls_to_yfinance`（:274）与 `test_router_akshare_global_news_empty_stops_chain`（:294）
- **问题**：`2026-08-04` 是历史日期后，`interface.py:_historical_near_window_news_refusal` 会先于 fake provider 命中返回「全球快讯为实时直播流…」，`assert out == "## yfinance global news"` 与 `assert out == "未获取到全球市场新闻"` 失败。
- **影响**：全量回归 777 用例中的 2 条当前必红（与规划 §1 预置日期炸弹一致）。
- **建议**：改为动态取今日（参照 `tests/test_historical_news_router_refusal.py` 的 `now_cn().date() - timedelta(days=…)`），或删除这两条路由用例。

### 3.3 确认死代码（生产路径不可达，可安全删除）

以下均经全仓 grep + 人工复核，确认除自身定义外无任何生产调用点（FastAPI 装饰器路由、`getattr` 动态分派、`__init__` 再导出、测试引用等均已排除）：

| # | 位置 | 符号 | 证据 |
|---|---|---|---|
| P1-5 | `api/database.py:26` | `_can_use_wal` | 全仓无引用；WAL 开关逻辑被绕过（`:33` 直接读 env） |
| P1-6 | `api/main.py:1336` | `_optional_user` | 无任何 `Depends(_optional_user)` 引用 |
| P1-7 | `api/main.py:1717` | `AgentProgressTracker.apply_chunk` | 同文件其余 tracker 方法均有调用，唯独它无 |
| P1-8 | `api/main.py:4520` | `_warmup_model_targets` | 同源 `_warmup_model_names`（:4508）被使用（:4719/:5085），本函数无人调用 |
| P1-9 | `api/services/scheduled_service.py:57` | `get_scheduled_batch` | 全仓无引用；实际用 `get_scheduled`/`list_scheduled` |
| P1-10 | `api/services/report_service.py:452` | `_canonicalize_result_data` | 仅一行转调 `canonicalize_report_result_data`（:423）的私有别名，无调用者 |
| P1-11 | `tradingagents/dataflows/interface.py:81` | `VENDOR_LIST` | 定义为 `_registry.list_names()`，此后无引用 |
| P1-12 | `tradingagents/dataflows/utils.py:9,15,19,29` | `save_output` / `get_current_date` / `decorate_all_methods` / `get_next_weekday` | 全仓（含 tests/scheduler/scripts/conftest）零引用 |
| P1-13 | `tradingagents/dataflows/providers/cn_akshare_provider.py:194` | `_locked` | 从未调用；代码统一用 `with AKSHARE_CALL_LOCK:`（20+ 处） |
| P1-14 | `tradingagents/graph/trading_graph.py:537` | `reflect_and_remember` | 全仓零调用（其内部调用的 `self.reflector.reflect_*` 仍有用） |
| P1-15 | `tradingagents/graph/trading_graph.py:204` | `TradingAgentsGraph.get_state(thread_id)` | 全仓唯一出现是 def；`self.graph.get_state`（:207）是底层 langgraph，非本方法 |
| P1-16 | `api/services/custom_prompt_service.py:157` | `resolve_role_prompt` | **生产死**（仅测试 `test_custom_prompts.py`/`test_custom_prompt_injection.py` 使用）。注意 `docs/KNOWN_ISSUES.md:211` 仍建议未来 Phase C 用它。**H3 决策：保留**——`resolve_role_prompt` 不删除，标注供 Phase C 接回调用点 |

> 注：`api/database.py:372 to_dict`、`trade_calendar.py:177 clear_cn_trade_date_cache`、`trading_graph.py:355 propagate_async`、`registry.py:81 list_resource_policies` 等经查**仅测试/工具使用**，非生产调用点，属「生产死/测试活」，不单独列为 P1（在 P2-观察项列出）。

### 3.4 无用 / 缺失依赖

#### P1-17 `requirements.txt:4` + `pyproject.toml:14` — `langchain-experimental` 声明但从未被导入
- **问题**：全仓（含 tests/scripts）零处 `import langchain_experimental`（grep 复核）。`uv.lock:1183` 也含该包。
- **影响**：无功能作用，白占依赖树（还拖入 `langchain-community` 等传递依赖）。
- **建议**：从 requirements.txt、pyproject.toml 移除；如确为上游 fork 需要，加注释说明。

#### P1-18 `pyproject.toml` — 缺 `python-dotenv`（requirements.txt 有，`pip install .` 会崩）
- **位置**：`pyproject.toml` dependencies 无 `python-dotenv`；但 `api/main.py:36` 与 `scheduler/main.py:31` 均 `from dotenv import load_dotenv`。
- **问题**：requirements.txt 有 `python-dotenv`（末尾行），pyproject 漏了。若用户 `pip install .`（走 pyproject），启动即 `ModuleNotFoundError: No module named 'dotenv'`。
- **影响**：打包产物在纯净环境不可用。
- **建议**：pyproject dependencies 补 `python-dotenv`。

### 3.5 前端孤立文件（死文件，P1）

| # | 位置 | 证据 |
|---|---|---|
| P1-19 | `frontend/src/hooks/useSSE.ts` | 全 src 无任何文件导入 `useSSE`（仅自引用） |
| P1-20 | `frontend/src/hooks/useTypeWriter.ts` | 导出 `useTypeWriter`/`useStreamingSection` 均无导入者 |
| P1-21 | `frontend/src/utils/portfolioSync.ts` | `formatSyncTimestamp`/`buildPortfolioSyncSummary` 无任何导入者（含 tests） |
| P1-22 | `frontend/src/main.js` + `frontend/src/styles.css` | 旧 vanilla-JS 落地页；`index.html:29` 实际加载 `/src/main.tsx`。`main.js`（380+ 行）与 `styles.css`（435 行）仅被 `eslint.config.js:10` 注释提及 |
| P1-23 | `frontend/src/utils/reportText.ts:26` | `buildAgentSummary` 导出但全 src/tests 零引用 |

> 关联：`frontend/src/stores/analysisStore.ts:226-232` 的 `addAgentMessage`/`addAgentToolCall` 为「已移至后端」空实现，仅可达自死文件 `useSSE.ts`；删 `useSSE.ts` 后这两个 no-op 一并变成死代码。

### 3.6 文档失实（与代码不一致）

| # | 位置 | 问题 |
|---|---|---|
| P1-24 | `README.md:3,11,73` | 三处写「14 名智能体/专业 Agent」，但 `tradingagents/graph/setup.py:40-53` 注册 **15 个** `create_*` 工厂（7 分析师 + 2 研究员 + 3 风控辩手 + 研究总监 + 风控经理 + 交易员），`CHANGELOG.md:24` 也写「12-Agent → 15-Agent」。README 数字过时 |

---

## 四、P2 问题清单（建议）

### 4.1 后端重复逻辑（可合并）

| # | 位置 | 描述 |
|---|---|---|
| P2-1 | `cn_akshare_provider.py:283` = `cn_investoday_provider.py:201` | `_slice_hist_df` 逐字节相同，可提到公共模块 |
| P2-2 | `cn_akshare_provider.py:296` = `cn_investoday_provider.py:243` | `_drop_incomplete_today_bar` 相同包装（都转调 `trade_calendar.drop_incomplete_today_bar`） |
| P2-3 | `cn_akshare_provider.py:270` `_format_ak_hist` vs `cn_investoday_provider.py:214` `_format_hist_csv` | 近相同 CSV 格式化（header/records/Dividends/Stock Splits 逻辑一致） |
| P2-4 | `cn_akshare_provider.py:303` = `cn_investoday_provider.py:568` | `_shrink_table` 静态包装重复（都转调 `utils.shrink_table`） |
| P2-5 | `_safe_float`/`_to_float` **5 份近相同实现**：`tracking_board_service.py:214`、`vlm_position_parser.py:71`、`portfolio_import_service.py:215`、`cn_investoday_provider.py:105`、`cn_akshare_provider.py:1155` | 建议抽公共 `utils.safe_float` |
| P2-6 | `api/main.py:1195` = `:1210` | `serialize_token_datetimes` 两个字段序列化器函数体完全相同 |
| P2-7 | `api/main.py:758` = `report_service.py:316` | `_strict_unit_interval` 逐字节相同（另 `_strict_report_confidence`/`_strict_report_probability` vs `_strict_claim_confidence` 亦重复） |
| P2-8 | `tradingagents/graph/conditional_logic.py:15-69` | 7 个 `should_continue_*` 方法体相同（同走 tool-calls 检查），仅经 `getattr` 分派，可合并为参数化方法 |

### 4.2 前端重复/死代码

| # | 位置 | 描述 |
|---|---|---|
| P2-9 | `pages/Analysis.tsx:13`、`pages/Reports.tsx:29`、`components/DecisionCard.tsx:48`、`utils/reportText.ts:1` | 决策字符串→动作映射实现 4 份且语义不一致（Reports 把 BUY→`add`，DecisionCard 把 BUY→`buy`） |
| P2-10 | `pages/Reports.tsx:158` vs `components/ReportViewer.tsx:89` | Markdown 导出/下载逻辑重复 |
| P2-11 | `pages/Portfolio.tsx:11` vs `utils/reportDualHorizon.ts:3` | `HORIZON_LABELS = { short:'短线', medium:'中线' }` 两处声明 |
| P2-12 | `components/KlinePanel.tsx:56-69` vs `components/TrackingBoardPanel.tsx:1148-1164` | 并行 `formatNumber`/中文单位格式化 |
| P2-13 | `services/api.ts:83,139,162,175,189,225,237,379,383,396,403,410,434,441,489,500,504` | **17 个未使用的 API 方法**：`startAnalysis` `getLatestReportsBySymbols` `createReport` `getWatchlist` `getScheduled` `triggerScheduledTest` `getPortfolioImportState` `getFeedback` `getFeedbackUnreadCount` `createProvider` `updateProvider` `deleteProvider` `updateModelProfile` `deleteModelProfile` `getResolvedCustomPrompts` `getPromptInjectionSwitch` `updatePromptInjectionSwitch` |
| P2-14 | `components/RoleModelConfigSection.tsx:85` | props 接口声明 `onRefreshRequired?: () => void`，从未解构/使用 |

### 4.3 未使用导入（pyflakes 确认，均人工复核）

后端（P2-15~P2-41 为批量登记，行号见各条）：

- `scheduler/main.py:23` `Any`、`Dict`；`:93` `_new_job_store`；`:106` `_set_job`；`:108` `_emit_job_event`
- `api/main.py:40` `fastapi.status`（另 `:825/:1577/:1596` 的 `status` 重定义、`:2996` 循环变量遮蔽 `status`）；`:48` `UserLLMConfigDB`/`ReportDB`/`ImportedPortfolioPositionDB`/`ModelProfileDB`/`RoleBindingDB`
- `tradingagents/graph/data_collector.py:17` `get_indicators`；`:686` 无占位 f-string
  > 注（DAV-82 收口 · P2-15 回补）：`get_indicators` **保留原因——测试 patch 目标**。`tests/test_dav28_data_boundaries.py`
  > 以 `patch.multiple("tradingagents.graph.data_collector", **patch_targets)` 批量替换该模块内的
  > `get_indicators`/`get_stock_data` 等数据入口；删除此导入会触发 `AttributeError`，使相关离线用例失败。
- `tradingagents/graph/trading_graph.py:3,8,9,10,19` `asyncio`/`date`/`Tuple`/`sqlite3`/`AgentState`/`InvestDebateState`/`RiskDebateState`
- `tradingagents/graph/setup.py:3` `List`；`propagation.py:4` `AgentState`
- `tradingagents/dataflows/interface.py:17` `result_to_prompt`
- `tradingagents/dataflows/y_finance.py:5` `os`；`:224` `curr_date_dt` 赋值未用
- `tradingagents/dataflows/utils.py:1,2` `os`、`json`；`:270/:296/:375` 局部 `pd` 未用
- `tradingagents/dataflows/alpha_vantage.py:2-5` 大量再导出（该文件本身是 4 行兼容 shim，见 P2-44）
- `tradingagents/llm_clients/openai_client.py:3,4` `time`、`JSONDecodeError`
- `tradingagents/dataflows/providers/cn_akshare_provider.py:24` `VendorRefuse`
- `api/services/auth_service.py:15` `JWTError`
- `api/services/vlm_service.py:15` `Any`；`report_service.py:12` `Literal`；`role_routing_service.py:8` `Tuple`
- `api/services/backtest_service.py:10` `json`；`:159` 局部 `ak` 未用
- `tradingagents/agents/researchers/bull_researcher.py:1,2,3` 与 `bear_researcher.py:1,2,3` `AIMessage`/`time`/`json`
- `tradingagents/agents/risk_mgmt/aggressive_debator.py:1,2`、`conservative_debator.py:1,2`、`neutral_debator.py:1,2` `time`/`json`
- `tradingagents/agents/trader/trader.py:3,4` `time`/`json`；`managers/risk_manager.py:1,2` `time`/`json`
- `tradingagents/agents/utils/context_utils.py:8` `cn_today_str`
- 各 analyst `:107/:111/:113/:114/:147` 无占位 f-string（`f"..."` 无 `{}`）

> 说明：`tradingagents/agents/utils/agent_utils.py:4-24` 的 20 个导入经查为**有意的再导出**（analysts 与 graph 均 `from agent_utils import get_indicators/…` 消费），**不算死代码**，但若重构可直接从 `agent_utils` 删除转发以减一层间接。

测试（P2-42，15 处，全部确认）：

| 位置 | 未使用项 |
|---|---|
| `tests/test_vendor_chain_semantics.py:17` | `datetime.timedelta` |
| `tests/test_config_fallback.py:2` | `MagicMock` |
| `tests/test_shrink_table.py:5` | `math` |
| `tests/test_shrink_table.py:8` | `pytest` |
| `tests/test_provider_date_guards.py:7` | `timedelta` |
| `tests/test_provider_date_guards.py:18` | `now_cn` |
| `tests/test_email_report_service.py:8` | `pytest` |
| `tests/test_no_wallclock_in_prompt.py:10` | `pytest` |
| `tests/test_trading_graph_multi_horizon.py:6` | `pytest` |
| `tests/test_dashboard_tracking.py:6` | `ReportDB` |
| `tests/test_api_smoke.py:327` | 局部 `content` |
| `tests/test_api_smoke.py:376` | 局部 `probe` |
| `tests/test_job_store_redis.py:142` | 局部 `it` |
| `tests/test_custom_prompt_injection.py:177` | 局部 `state_graph` |
| `tests/test_financial_announce_cutoff.py:376` | 局部 `parent` |

### 4.4 测试质量

| # | 位置 | 描述 |
|---|---|---|
| P2-43 | `tests/test_cn_akshare_backup_sources.py:25` | `_fake_ak(**methods)` 死辅助函数（从未调用，全仓 1 处） |
| P2-44 | `tests/test_dav27_report_semantics.py:38` | `_valid_machine_payload()` 死辅助（与 `test_prompt_semantics.py:111` 同名不同源，后者被用） |
| P2-45 | `tests/test_vendor_chain_semantics.py:69` | `_ROUTER_SAMPLE_ARGS["get_global_news"]` 死数据（无对应 `_route(..., method="get_global_news")` 消费点） |
| P2-46 | `tests/test_scheduler_persistence.py:388` | `test_scheduler_loop_exits_when_stop_event_set` 无断言无超时，回归会以「挂起」而非「失败」暴露 |
| P2-47 | `tests/test_shrink_table.py:51` | 恒真断言 `assert MISSING_VALUE_MARKER not in text or True`（注释自认「may or may not appear」），零覆盖 |
| P2-48 | `tests/test_watchlist_scheduled.py:110` | `test_allow_boundary_times` 只验证「不抛异常」不验证行是否创建 |
| P2-49 | `conftest.py:36` | `_OFFLINE_TRADE_END = date(2027,12,31)` 固定地平线：任何用实时时钟走到 2027-12 之后的测试会开始失败（潜在，非紧急） |
| P2-50 | `scripts/smoke_custom_prompts.py:193-195,278` | `get_db_ctx`/`UserDB`/`app` 未用，且 `app` 重复定义遮蔽 |

### 4.5 生产路径调试 print / 残留物

| # | 位置 | 描述 |
|---|---|---|
| P2-51 | `tradingagents/graph/data_collector.py:325,619,686,688` | `[Timer]/[Warning]/[Error]` 直出 stdout（数据抓取生产路径） |
| P2-52 | `tradingagents/agents/analysts/volume_price_analyst.py:64,67` | 流错误/fallback 提示 print |
| P2-53 | `tradingagents/dataflows/y_finance.py:170,292`、`alpha_vantage_common.py:121`、`alpha_vantage_indicator.py:221` | provider 路径错误 print |
| P2-54 | `frontend/.vade-report` | Vercel Speed Insights 安装报告被提交进仓库（工具残留） |

### 4.6 其他后端

| # | 位置 | 描述 |
|---|---|---|
| P2-55 | `tradingagents/llm_clients/TODO.md` | 源码包内遗留任务文档（4 条未完成项：validate_model 未调用、参数未统一、base_url 被忽略、validators 未同步），建议转 issue 后删除或移到 docs/ |
| P2-56 | `api/database.py:84` `exc_val`/`exc_tb`、`scripts/smoke_custom_prompts.py:61` `exc_val`/`exc_tb`、`cn_akshare_provider.py:138` `exc_info` | `except ... as` 绑定未使用变量（可改 `except Exception:`） |
| P2-57 | `tradingagents/dataflows/alpha_vantage.py`（4 行） | **不适用**——纯再导出 shim，无真实逻辑，无需处理（grep 复核仍无任何消费者；保留零副作用，删除亦无实际收益） |
| P2-58 | `scheduler/__init__.py`、`tests/__init__.py`、`tradingagents/dataflows/__init__.py` | 空文件（包标记，正常，仅记录） |

### 4.7 前端依赖

| # | 位置 | 描述 |
|---|---|---|
| P2-59 | `frontend/package.json:15,18,19,28` | 声明但全项目零导入：`@tanstack/react-virtual`、`clsx`、`date-fns`、`tailwind-merge`（含 src/config/tests grep 复核） |

### 4.8 配置与文档细节

| # | 位置 | 描述 |
|---|---|---|
| P2-60 | `README.md:152` vs `docker-compose.yml:8` | 根 README 说 compose 起后访问 `:8000`，实际映射 `8001:8000`（`README_本地部署.md:23` 正确写 8001） |
| P2-61 | `frontend/README.md:8` | 「12 个智能体」应为 15；`:62-63` 列出的 `AgentPipeline.tsx`/`LogStream.tsx` 组件已不存在；`:144` License 写 MIT，实际 `frontend/LICENSE` 为 PolyForm Noncommercial 1.0.0 |
| P2-62 | `CHANGELOG.md:38` | v0.5.0「移除 redis」与现依赖不符（`requirements.txt`/`pyproject.toml` 均有 `redis[hiredis]`，`api/job_store_redis.py` 在用）——条目过时 |
| P2-63 | `CHANGELOG.md:5` + `pyproject.toml:9` | 版本不一致：CHANGELOG 最新 v0.5.0（2026-03-22），pyproject `version = "0.2.0"`；且 CHANGELOG 完全缺失 2026-07/08 的 M1~M5（DAV-67~71）条目 |
| P2-64 | `AGENTS.md:35` | 前端描述为「Vue + TypeScript」，实际为 React + TS（`frontend/README.md` 与代码均 React） |
| P2-65 | `docker-compose.yml:11-14` | 把 `./tests`、`./scripts` 也 bind-mount 进生产容器（开发便利项，建议只挂 `api`/`tradingagents`/`data`） |
| P2-66 | `docs/KNOWN_ISSUES.md` | 内容整体与代码同步良好（引用 DAV-68/69 均为真实实现）；唯一注意点：`:211` 建议 Phase C 用 `resolve_role_prompt()`（P1-16），但该函数当前生产死，Phase C 落地时需先接回调用点 |

### 4.9 规划文档遗留观察项（不属本审计，转 H3 参考）

规划 `work/2026-08-05-code-audit-plan.md` §1 观察项（非本次新发现，供 H3 处理）：
- `get_global_news` 参数未生效（look_back_days/limit）
- VendorFail 同 vendor 重试语义
- Redis 孤儿 job 清理
- compose `stop_grace_period` 未配置
- M2 私有函数耦合 / 静态标签悬空 / 密度阈值裕量

---

## 五、覆盖缺口分析（合并自 DAV-74 H2 交付登记）

> 数据来源：DAV-74 代码运维测试员 H2 交付汇报（comment `a9702040`）「4. 覆盖缺口分析」与「5. 质量抽查」。基线：日期炸弹修复 commit `97d76b6` 后全量回归 `780 passed / 13 skipped / 0 failed`。**本节为 H3 施工依据之一**（H2 验收标准 #4「覆盖缺口登记到审计报告」）。

### 5.1 总体覆盖

- **TOTAL 67%**（13091 stmts / 4355 missing）。按核心路径四类分组登记，行号区段为该模块未覆盖函数的大致范围。

### 5.2 分类缺口明细

| 核心路径 | 模块（行覆盖率） | 未覆盖要点（H2 登记） |
|---|---|---|
| 数据获取 | `tradingagents/dataflows/y_finance.py` 14% | 行情/财务解析主路径 |
| 数据获取 | `tradingagents/dataflows/yfinance_news.py` 6% | 新闻解析 |
| 数据获取 | `tradingagents/dataflows/alpha_vantage_indicator.py` 3% | 技术指标解析 |
| 数据获取 | `tradingagents/dataflows/alpha_vantage_common.py` 20% | 通用解析 |
| 数据获取 | `tradingagents/dataflows/providers/cn_baostock_provider.py` 53% | 备用源 |
| 数据获取 | `tradingagents/dataflows/stockstats_utils.py` 28% | 指标计算 |
| 数据获取 | `tradingagents/dataflows/providers/cn_akshare_provider.py` 65% | 缺 386–619 等数据方法 |
| 数据获取 | `tradingagents/graph/data_collector.py` 65% | 缺 144–308 VPA 格式化、505–544 |
| 裁决链 | `tradingagents/graph/trading_graph.py` 41% | 缺 69–202 主图装配、472–551 |
| 裁决链 | `tradingagents/graph/conditional_logic.py` 21% | 分支条件判定（对应本报告 P2-8 的 7 个 `should_continue_*`） |
| 裁决链 | `tradingagents/graph/reflection.py` 31% | 反思环节 |
| 裁决链 | 6 个分析师 prompt 模块 | 均 ~12–13% |
| 裁决链 | 3 个 debator（裁决辩论） | 各 16% |
| 任务持久化 | `api/job_store.py` 85% | 良好 |
| 任务持久化 | `api/job_store_redis.py` 22% | **受 12 个 Redis skip 直接影响**（无 Redis 环境） |
| 任务持久化 | `scheduler/main.py` 68% | 缺 610–675 SIGTERM 排空、391–419 认领/恢复 |
| 校准统计 | `api/services/calibration_service.py` 87% | 良好；缺 417–424 价格获取异常路径、556 busy 重试 |
| 校准统计 | `api/services/role_routing_service.py` 11% | **角色路由严重低覆盖** |
| 校准统计 | `api/services/backtest_service.py` 59% | 回测 |

### 5.3 H2 质量抽查补充（并入 P2 测试清单）

| 位置 | 描述 |
|---|---|
| `tests/test_api_smoke.py:244`、`tests/test_calibration_service.py:580` | 冒烟级弱断言（`assert ... is not None`），建议按 AGENTS.md 要求断言具体内容 |
| `tests/test_job_store.py:161` | 断言内部属性 `store._job_events`（测实现细节而非行为），建议改行为级断言 |

### 5.4 低覆盖修复建议（H2 优化 P1/P2，供 H3 落地）

1. **P1 — 恢复 Redis 覆盖**：dev 依赖引入 `fakeredis` 或 CI 起 Redis，使 12 个 `test_job_store_redis.py` 用例离线可跑（预期 `job_store_redis` 22%→85%+）。
2. **P1 — 补齐裁决链行为级测试**：`trading_graph.py` 41%、debator 16%，冻结 LLM 输出断言链路装配与裁决引用（与 H2 已交付的 `test_evidence_citation_density.py` 同思路）。
3. **P2 — yfinance/Alpha Vantage 解析层离线单测**：解析/分类逻辑不依赖网络，mock 输入即可覆盖（`y_finance.py` 14%、`alpha_vantage_indicator.py` 3%）。
4. **P2 — 加强弱断言**：`test_api_smoke.py:244`、`test_calibration_service.py:580`；`test_job_store.py:161` 改行为级断言。
5. **P3 — `.gitignore` 补 `.venv310`**；`test_provider_date_guards.py:273` 死代码守卫改相对日期（与 P2-49 `conftest.py:36` 地平线同类）。

---

## 六、文档体检结论

| 文档 | 状态 | 说明 |
|---|---|---|
| `README.md` | 🟡 失实 | Agent 数 14 vs 15；compose 端口；API 示例日期过时（2026-03-28） |
| `README_本地部署.md` | 🟢 个人运维笔记 | 与本地部署一致，端口 8001 正确；仅模型名/密钥为个人配置示例 |
| `frontend/README.md` | 🔴 显著过时 | 12 agents、已删除组件、License 错误 |
| `CHANGELOG.md` | 🔴 缺失/过时 | 无 M1~M5 条目；v0.5.0「移除 redis」失实；版本号与 pyproject 冲突 |
| `AGENTS.md` | 🟡 小失实 | 前端框架 Vue→React；其余纪律条款与代码现实一致 |
| `docs/KNOWN_ISSUES.md` | 🟢 基本同步 | 引用 DAV-68/69 均为真实代码；遗留 gap 描述准确 |
| `guide/`（README/configuration/deployment） | 🟢 高质量 | configuration.md 逐条对应当前 env 读取，含 scheduler/VLM/超时等新变量 |
| `docs/TA_MODELS_FETCH_ALLOWLIST.md` | 🟢 与代码一致 | SSRF 白名单描述与 `_models_fetch_allowlist` 实现吻合 |
| `skills/tradingagents-analysis/SKILL.md` | 🟢 辅助技能 | 与仓库分析流程一致 |

---

## 七、总体评价与修复优先顺序

**总体**：代码库卫生状况良好——无 P0，无安全漏洞，无导入即崩的坏导入，无空函数，几乎无注释掉的代码块，`_v2/_old/_new` 并行实现已按 AGENTS.md 纪律清零。主要债务集中在**两处真实运行时 bug（P1-1/P1-2）**、**生产路径 eval 落盘（P1-3）**、**日期炸弹（P1-4）**、**约 13 个确认死函数**、**前端 5 个孤立文件 + 17 个未用 API 方法 + 4 个未用 npm 包**，以及**文档与代码事实漂移**（README/CHANGELOG/AGENTS/frontend README）。

建议修复顺序（供 H3 参考）：
1. **先**：P1-1、P1-2（两行补导入，风险最低收益最高）；P1-4 日期炸弹（当前回归必红）。
2. **再**：P1-3 移除/收敛 eval_results 写盘（改一行到 `TA_TRACE` 开关，回归看文件差异）。
3. **清理死代码**：P1-5~P1-16（约 13 个符号，每条删除前在 H3 用 `grep` 复确认一次）。
4. **依赖**：P1-17 移除 langchain-experimental；P1-18 补 python-dotenv；P2-59 移除 4 个 npm 包。
5. **前端**：删除 P1-19~P1-23 孤立文件 + P2-13 的 17 个 API 方法。
6. **重复逻辑合并**：P2-1~P2-8（后端）、P2-9~P2-12（前端），按模块分批、一个 commit 一个关注点。
7. **测试清理**：P2-42~P2-48。
8. **覆盖缺口**：按 §5.4 补 Redis 覆盖（P1）、裁决链行为级测试（P1）、解析层离线单测（P2）。
9. **文档同步**：P1-24、P2-60~P2-64。

> 全部结论基于实际读到的代码（源码 + grep + pyflakes/vulture 人工复核）。存疑处已明确标注。本审计未修改任何代码。
