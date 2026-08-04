# Changelog

All notable changes to this project will be documented in this file.

## [v0.6.0] - 2026-08-05

M1~M5 里程碑（DAV-67~71，2026-07/08）竣工，随附 H1~H3 审计与清理轮记录（见 Unreleased）。

### Added
- **M1（DAV-67）**：收口 2026-08-04 十项修复——交易日历 fallback、近窗豁免、中文前置、3000 字提示词恢复、SSRF allowlist、新浪资金流备用源、数据库路径、start.sh、socksio、max_tokens。
- **M2（DAV-68）**：裁决者证据闭环——`research_manager` 接收 market/news/fundamentals/macro 一手证据摘要（≤300 字符，`build_evidence_summary`），macro 分析接线补齐；trader/risk_manager 自定义提示词注入；golden-output citation-density 回归（`tests/test_evidence_citation_density.py`）。
- **M3（DAV-69）**：数据源语义重构（`VendorResult`/`VendorRefuse`/`VendorFail`/`VendorEmpty`/`VendorOk` 类型化结果，拒绝/失败/确认空正确区分）；东财受限接口备用源；任务状态落库 + SIGTERM 优雅关停；调度心跳区分崩溃任务与弃单任务。
- **M4（DAV-70）**：校准度统计闭环——可靠性曲线（按 probability 分桶统计实际上涨率）+ Brier score + 前端校准度面板，支持按日期段/提示词版本/模型过滤。
- **M5（DAV-71）**：安全与体验收尾——`TA_APP_SECRET_KEY` 未设置拒绝启动（DAV-66）、前端死任务轮询修复、旧版英文报告「旧版报告」标识、全量回归 + 文档收口。

### Changed
- 模型配置新增推荐厂商预设与引导入口；支持从代理自动拉取模型并同步进角色模型配置。
- 角色级模型路由与数据供应商扩展；历史日期分析严格执行日期语义（快照源拒绝服务、财报按公告日期截断、按列名取数而非位置切片）。
- 七个分析师补齐 stream 退化中止保护；结构化报告新增 probability/data_gaps/falsification_conditions 等字段。
- 线程池饱和 524 修复；长时分析超过软超时后继续执行；股票识别兜底与错误信息人性化。

### Fixed
- 修复历史数据日期边界：报表/新闻/龙虎榜/资金流等近窗来源在历史分析日期被拒，财报按有效公告日截断。
- 修复风控/持仓字段默认零填充：缺失质押、融资融券风险字段显式拒绝而非填 0。
- 修复调度任务重复触发与运行中重启丢失（任务状态落库 + 启动恢复）。
- 修复信号侧否定词误判导致的 BUY 决策偏差。

### Removed
- 移除盘中概念板块异动扫描（外部数据源不稳定且有合规风险，回滚 #200）。

## Unreleased

### 审计与清理轮（H1~H3）
- **H1（DAV-73）**：全仓只读审计（286 个非 git 文件），产出 `work/code-audit-report.md`（P1 24 条施工依据）。
- **H2（DAV-74）**：测试体检 + 日期炸弹修复（P1-4），全量 780 passed / 0 failed；登记覆盖缺口（Redis fakeredis 恢复、裁决链行为级测试）。
- **H3（DAV-75）**：按审计报告逐条销号——删死代码、合并重复逻辑、清理未用导入与孤立文件、依赖体检（移除 `langchain-experimental`、补 `python-dotenv`）、文档同步（README/CHANGELOG/AGENTS/frontend README 与代码一致）。

## [v0.5.0] - 2026-03-22

### Added
- **Token 级流式输出**：全部 15 个 Agent 支持 astream Token 推送，对话框实时展示 LLM 输出过程。
- **自选股管理**：数据库持久化的自选列表（上限 50），支持股票代码/名称模糊搜索。
- **定时分析**：每个交易日在用户设定时间（20:00~08:00）自动触发分析，连续失败 3 次自动停用。
- **后台调度器**：FastAPI lifespan 内嵌 asyncio 调度协程，交易日判断、防重复触发、串行预采集。
- **多阶段状态指示器**：用户提交后即时反馈：连接中 → 识别标的 → 采集数据 → 多智能体分析。
- **股票搜索 API**：`/v1/market/stock-search` 支持代码前缀和名称模糊匹配，7 天 TTL 缓存。
- **Dependabot**：自动依赖更新（pip、npm、GitHub Actions、Docker）。

### Changed
- **对话框重构**：Agent 消息改为紧凑卡片（图标+标签+实时预览），点击展开完整内容，完成后自动转为报告卡片。
- **图标体系统一**：对话框与协作面板使用一致的 Lucide 图标与配色，覆盖全部 15 个 Agent。
- **结构化提取增强**：使用 json-repair 替代正则剥离；Pydantic 模型容忍 LLM 输出变体（数组→首元素、数字→字符串、缺失字段默认值）。
- **后端异步并行**：分析师节点全部转为 async，数据采集并行执行，意图解析不再阻塞 SSE 流返回。
- **Portfolio 页面**：从"热榜选股"重构为"自选 & 定时分析"，去掉外部热榜数据依赖。
- **日志体系**：uvicorn 日志配置文件统一时间戳格式。
- **Docker 优化**：CMD 使用 tradingagents-api 入口点；git tag 注入 VERSION build-arg。
- **登录页**：12-Agent → 15-Agent，补齐宏观分析、主力资金、博弈裁判。
- **SQLite WAL 模式**：启用 WAL 支持并发读写。

### Fixed
- 修复 mini_racer/V8 多线程 crash：启动时预加载交易日历，后续请求全走缓存。
- 修复结构化提取失败：LLM 返回 markdown 代码围栏或非标准 JSON 格式导致 Pydantic 解析错误。
- 修复定时任务重复触发：启动前标记 last_run_date，防止调度循环重复启动。
- 修复 `import re` 缺失导致 report_service 崩溃。
- 修复 `_load_cn_stock_map` 错误导入位置。
- 修复 Agent 卡片状态：job.completed 时标记所有 Agent 为已完成，不再显示"撰写中"。
- 修复意图解析 JSON 在对话框显示的问题。
- 移除协作面板流光动画。

### Removed
- 移除 11 个未使用的依赖（backtrader、chainlit、alembic、rich、typer 等）。
- 移除热榜选股功能（外部数据源不稳定且有合规风险）。
- 移除对话框中冗余的系统消息（job.created、job.running、agent.tool_call）。

## [v0.4.4] - 2026-03-18

### Fixed
- Fixed critical **SQLAlchemy TimeoutError** by unifying database session lifecycle across API endpoints and background tasks.
- Fixed **Resource/Semaphore Leakage** on shutdown by adding executor shutdown to the FastAPI lifespan.
- Improved repository structure by moving `announcements.json` to the `api/` directory and updating search paths.
- Cleaned up redundant `uv.lock.cp313` and `CLAUDE.md` files.
- Resolved **Announcement Schema Validation** errors (500) by aligning `announcements.json` with Pydantic model requirements.
- Made `/v1/announcements/latest` a public endpoint to ensure visibility before login.

## [v0.4.3] - 2026-03-16

### Added
- Added **Task Lifecycle Persistence and Recovery** (#32): Analysis jobs can now survive server restarts.
- Added **Configurable Max Workers** (#33): Job executor concurrency is now tunable via `TA_MAX_WORKERS` env var.
- Added persistent report lifecycle fields, including `status`, `error`, and richer section-level report storage.
- Added structured analyst trace persistence to support future report-side insight displays.
- Added header announcement support backed by `announcements.json` and `/v1/announcements/latest`.

### Changed
- Changed the report flow to initialize records earlier and update report content incrementally during long-running analysis jobs.
- Changed the header announcement entry to load from backend data instead of hard-coded preview text.
- Improved error messaging for failed analysis steps in the UI.

### Fixed
- Fixed report serialization gaps so newly persisted lifecycle and extended section fields can be returned consistently.
- Fixed report finalization and failure recording so completed and failed jobs leave clearer artifacts for follow-up inspection.

## [v0.4.2] - 2026-03-16

### Added
- Added user-context grounding so analysis can incorporate objective, risk preference, investment horizon, and holding constraints.
- Added local Docker one-click deployment script for easier self-hosted setup.

### Changed
- Upgraded the debate workflow to a claim-driven flow for stronger argument organization and downstream judgment.
- Improved multi-horizon analysis wording and parameter handling.

### Fixed
- Fixed structured extraction prompts by explicitly restoring missing JSON keywords that caused 400 errors.
- Removed mistakenly committed runtime artifacts such as `deploy` and `.vite` from version control.

## [v0.4.1] - 2026-03-15

### Added
- Added intent-driven multi-horizon analysis with streaming progress updates.
- Added integrated frontend-backend Docker packaging and multi-architecture CI/CD automation.
- Added restored A-share analysis skills with a hardened CI environment.

### Changed
- Re-applied missing dependency updates including `marshmallow` and `python-socketio`.

### Fixed
- Fixed review issues raised during the v0.4.1 stabilization cycle.
- Improved SKILL metadata and SEO-related presentation.

## [v0.4.0] - 2026-03-13

### Added
- Added monorepo synchronization and the new game-theory agent integration.
- Added report `direction` field and UTC timestamp serialization.
- Added frontend commit message support.
- Added skills support for using TradingAgents through reusable skill workflows.

### Fixed
- Fixed default agent settings.
- Fixed stock symbol normalization at task startup and during K-line data retrieval.

## [v0.3.0] - 2026-03-12

### Changed
- Removed the redundant `frontend_backup/` tree from the main branch to simplify the repository layout.
