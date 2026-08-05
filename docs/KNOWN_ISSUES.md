# Known Issues

## Legacy report English direction leaks on secondary surfaces

**Status:** Open (minor, display-layer only)  
**Discovered:** 2026-08-04 during M5 wrap-up

### Symptom

Pre-Chinese-localization reports store their direction in English (`BULLISH` /
`LEAN_BEARISH` / `NEUTRAL` / …). The main report surfaces now handle this — the
report list and detail view show a「旧版报告」badge, and the DecisionCard maps
English directions to Chinese via `DIRECTION_ALIAS` (`localizeDirection` in
`frontend/src/utils/reportText.ts`). Secondary surfaces that render a report's
raw `direction` field are not mapped yet:

- `TrackingBoardPanel`（跟踪看板）— renders `analysis.direction` directly
- `Portfolio`（持仓页）— renders `report.direction` in the latest-report line

So a legacy report can still display「方向：BULLISH」in those spots.

### Suggested fix

Apply the same display-layer mapping the DecisionCard uses
(`localizeDirection`) to the tracking board and portfolio list. Low risk; no
data migration involved.

---

## Vendor chain collapses three outcomes into two behaviors

**Status:** Fixed in DAV-69 (typed vendor results). The result-type redesign below is implemented in `tradingagents/dataflows/vendor_result.py` and consumed by `route_to_vendor` in `tradingagents/dataflows/interface.py`.
**Discovered:** 2026-07-29 during historical-news refusal work

### Symptom

`route_to_vendor` only advances to the next provider on **exceptions**.
A provider that returns an ordinary failure / empty / refusal **string** is treated as a successful hit and **stops the chain**.

Three logical outcomes are therefore folded into two runtime behaviors:

| Semantic outcome | Desired behavior | Current behavior |
|---|---|---|
| **Refuse** — this source cannot serve the request (snapshot-only, near-window-only, missing date field, etc.) | Stop the chain *or* skip only equivalent weak sources; never pretend success | Returned as a normal string → **chain stops** |
| **Fail** — network / timeout / parse error / temporary API breakage | Try the next vendor | Exception → **fallback** |
| **Confirmed empty** — query succeeded and there is truly no data | Stop and report “confirmed none” | Returned as a normal string → **chain stops** (same as refuse) |

### Why it matters

- A **refusal** implemented only inside one provider can be bypassed if that provider raises (signature mismatch, lock timeout, schema error) and a later vendor has **no date semantics** (classic case: `get_global_news` akshare → yfinance).
- A **near-window empty** returned as `"No news found …"` looks like “confirmed none” to the model, blocks investoday/other historical-capable sources, and is the wrong affirmative conclusion for historical analysis.
- The same shape affects any multi-provider category where some vendors are date-aware and others are live snapshots.

### Affected methods (current inventory)

High risk (multi-provider chain + mixed date semantics):

1. `get_global_news` — mitigated for historical dates by router-level refuse (3b-hotfix); live path still uses chain
2. `get_news` — same mitigation for historical dates (policy A); live path still uses chain
3. `get_insider_transactions` — provider-level snapshot refuse; exception can still fall through to yfinance/AV
4. `get_fundamentals` / three statements — date truncation on CN paths; yfinance path does not use `curr_date`
5. `get_realtime_quotes` — shorter chain; lower risk after snapshot refuse
6. `cn_market_data` / `institutional_risk` tools — config default may fall back to `yfinance` when category unset (also tracked for data_vendors commit)

### Suggested fix (implemented in DAV-69)

1. Introduce an explicit result type (or exception hierarchy), e.g.:
   - `VendorRefuse(reason)` — do not fall through to date-blind vendors; optional allow-list for same-semantics peers
   - `VendorFail(error)` — try next vendor
   - `VendorEmpty(confirmed=True)` — stop; prompt says confirmed none
   - `VendorOk(payload)`
2. Until that lands, **date-blind / near-window capabilities must be refused at `route_to_vendor` (or category policy)**, not only inside a single provider.
3. Re-enabling investoday (or any historical-capable source) for news must be an **explicit same-source whitelist for both live and historical modes**, never a silent fallback hit.

Implementation notes (DAV-69):

- `VendorResult` / `VendorOk` / `VendorRefuse` / `VendorEmpty` / `VendorFail` live in
  `tradingagents/dataflows/vendor_result.py`; `result_to_prompt()` unwraps a typed
  result back to a prompt string for direct callers.
- `route_to_vendor` now interprets: plain value or `VendorOk` = hit; `VendorRefuse` =
  stop (continue only through `allow_peers`); `VendorEmpty` = confirmed none, stop;
  `VendorFail` or exception = fall through to the next vendor.
- Providers signal the new semantics: cn_akshare `get_global_news` (sina fail →
  `VendorFail`, sina empty → `VendorEmpty`), cn_akshare `get_news` empty →
  `VendorEmpty`, yfinance news/global/insider error strings → `VendorFail` and
  "No ... found" → `VendorEmpty`. Plain strings remain backward-compatible hits.
- Regression tests: `tests/test_vendor_chain_semantics.py`.

### Regression guard

3c-3 e2e guards (missing `curr_date` hard-fail, dual historical-date upper-bound comparison, provider signature whitelist) are intended to catch silent reintroduction of undated live data on historical paths. They do **not** replace a typed vendor-result redesign.

---

## Adjudicators have no first-hand access to analyst reports

**Status:** Partially resolved in DAV-68 M2 (`research_manager` now receives
first-hand evidence summaries for market/news/fundamentals/macro; `trader` and
`risk_manager` custom-prompt injection landed; trader/risk_manager analyst-report
access remains a follow-up)  
**Discovered:** 2026-07-29 during 1.58MB output investigation

### Symptom

`research_manager`, `risk_manager`, and `trader` do **not** receive any analyst
report directly in their prompt.  They only see debate history and summary fields.
`fundamentals_report` (and others) are read from state only to construct
`curr_situation`, which is used exclusively as a memory-retrieval embedding query —
the string is never injected into the prompt template.

### Prompt template coverage map (from `tradingagents/prompts/zh.py`)

`✓(摘要)` = the report is passed as a bounded first-hand **evidence summary**
(`build_evidence_summary`, ≤300 chars) rather than the full report.

| Agent | market | sentiment | news | fundamentals | smart_money | volume_price | macro |
|---|---|---|---|---|---|---|---|
| bull_researcher | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |
| bear_researcher | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |
| aggressive_debator | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |
| conservative_debator | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |
| neutral_debator | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |
| **research_manager** | ✓(摘要) | ✓ | ✓(摘要) | ✓(摘要) | ✓ | ✓ | ✓(摘要) |
| **trader** | — | — | — | — | — | — | — |
| **risk_manager** | — | — | — | — | — | — | — |

`research_manager` receives `smart_money_report`, `volume_price_report`, and
`sentiment_report` as raw data for its "expected-value gap analysis", plus
bounded evidence summaries of market/news/fundamentals/macro for evidence-level
cross-checks.  The macro analyst's report — previously consumed by nobody — is now
wired in.  `trader` and `risk_manager` receive only structured summaries built by
`build_agent_context_view` — no analyst reports at all.

### Why it matters

Adjudicators decide direction and construct the final trade plan, but they can only
compare *how persuasively each debater argued* — they cannot independently verify
the underlying data.  In the 2026-07-29 600519 run, the fundamentals analyst gave
"中性", yet `research_manager` reached "偏空" exclusively from debate text and
volume-price signals; it had no way to cross-check the fundamentals claim.

This is an inherent structural constraint, not a bug per se, but it creates a
situation where strong rhetoric can outweigh weak evidence at the adjudication
layer.

### Fix (implemented in DAV-68 M2)

Adjudicators do **not** receive full analyst reports — that would blow up context.
`tradingagents/agents/utils/evidence_summary.py::build_evidence_summary` builds a
deterministic, bounded (≤300 chars) summary per report that keeps verifiable facts
(numbers, dates, named events), drops argumentation, strips machine-readable
blocks, and prefixes the analyst's own `VERDICT` direction as a labeled fact so the
manager can still tally verdicts.  `research_manager` receives these summaries for
market/news/fundamentals/macro.

The analyst's full report remains available in state for bull/bear to read.

**Golden-output citation-density regression (DAV-68 optimization ②).**
`tests/test_evidence_citation_density.py` drives the real adjudication chain
(research_manager → trader → risk_manager) with golden analyst reports and
golden adjudicator outputs, then asserts a per-hop **citation density** floor:
of the concrete evidence facts present in a node's input prompt, the node's
output must cite at least a minimum fraction. The metric ties the output back
to the wiring — if the evidence summaries (or the plan hand-off) are ever
dropped, the available-fact count collapses and the test fails even though a
mock LLM would still return the fixed golden output.

### Remaining gap (follow-up)

`trader` and `risk_manager` still do not receive analyst reports (only
`build_agent_context_view` summaries).  They DO now receive the 3000-char
custom-prompt injection (confidence-ceiling / falsification constraints), so the
adjudication-affecting constraints reach them, but first-hand analyst-evidence
access for those two roles is a candidate next step.

---

## Custom prompt history is not retained — old versions are unrecoverable

Status: **known gap, by design for Phase B. Must be closed in Phase C.**

### Symptom

`PATCH /v1/custom-prompts` replaces the user's whole prompt set (delete + insert in
one transaction), mirroring `update_role_bindings`.  Each row carries a
`prompt_hash` (sha256[:12]) that identifies *which version* a prompt was, but the
previous row — and therefore the previous prompt **text** — is gone after any edit.

### Why it matters

This bites the project's end goal (statistical calibration of historical-date
analyses), not just tidiness.

In Phase E's A/B runs, each report can be tagged with the prompt hash that produced
it.  Months later, when calibration is computed across a batch of reports, a report
tagged `hash=abc123` cannot be traced back to any prompt text: we will know that two
batches used *different* prompts, but not *what the older prompt said*.  Attributing
a calibration shift to a specific prompt change — which is precisely the question the
custom-prompt work exists to answer — becomes impossible.

### Suggested fix (Phase C, when injection is implemented)

Do **not** build a prompt-history table.  Instead, at injection time write the full
**resolved prompt text itself** into the report snapshot, alongside its hash and
length.  Attribution then becomes self-contained: no lookup against
`user_custom_prompts` is ever needed, and later user edits cannot invalidate the
record.  The resolved text is capped at 6000 chars
(`custom_prompt_service.RESOLVED_PROMPT_MAX_CHARS`), which is negligible next to a
report that is already hundreds of KB.

Retrieve the text via `custom_prompt_service.resolve_role_prompt()` /
`resolve_all_roles_prompts()` — do not re-concatenate global + override at the call
site, or the two implementations will drift.

---

## 同花顺（fuyao.aicubes.cn）数据源接入

**Status:** DAV-83 已接入；存在若干已知边界。  
**Discovered:** 2026-08-05

### 接入情况

- 新增 `cn_fuyao` provider：行情快照、历史日 K（前复权）、三大报表、财务指标（五类能力）、
  涨跌停池（含连板分布）、龙虎榜、交易日历。
- 路由：`fundamental_data` 以 `cn_fuyao` 为主源（失败降级现有弱源）；`get_zt_pool` /
  `get_lhb_detail` 以东财为主、`cn_fuyao` 备用；`core_stock_apis` 中 `cn_fuyao` 排第 5
  （`cn_akshare,cn_baostock,cn_investoday,yfinance,cn_fuyao`，yfinance 之后）；`realtime_data`
  中 `cn_fuyao` 提供第三备用；交易日历在 AKShare 不可用时以 fuyao 近一年日历在线对照。
- 错误码映射：`1001~1004` 参数错误显式报错；`2001/2003` Key 无效显式报错；
  `3001/3002` 标的不存在/数据未就绪、`3004` → `VendorEmpty`；**财务数据路径**
  （`get_fundamentals` / 三大报表）下 `3001` 映射为 `VendorFail`（触发降级到
  cn_akshare / cn_baostock / cn_investoday 等弱源），`3002` 仍为 `VendorEmpty`；
  `4001` 频率超限退避重试后 → `VendorFail`；`5001~5003` 服务端错误 → `VendorFail`。

### 已知边界

- **`/limit-up-ladder` 未接线**：`cn_fuyao` 未实现连板天梯端点；「连板天梯」语义仅由
  `get_zt_pool` 返回中的「连板分布」部分覆盖。文档不再宣称「连板天梯已覆盖」。
- **`3004` 映射**：`3004` 未在接口文档中单独定义，当前按「确认无数据」处理为
  `VendorEmpty`（与 `3002` 同语义）。若后续接口文档明确 3004 为「目标未覆盖/无权限」，
  应复核其是否应改为 `VendorFail`（参考财务路径对 3001 的分治处理）。
- 三大报表按 `period_end` 落在 `[curr_date-8y, curr_date]` 区间取数，未做「披露日」级
  前视剔除（与 Investoday 现有实现一致）；对临近披露窗口的极端历史回测可能存在轻微前视。
- 财务指标 `get_fundamentals` 的报告期（`yyyy-N`）按披露截止日启发式选取
  （一季报 4/30、中报 8/31、三季报 10/31、年报次年 4/30），非逐票公告日精确映射。
- **`get_fundamentals` 缺 `curr_date` 回退 `now()`**：`_latest_report_period` 在
  `curr_date=None` 时用 `datetime.now(CN_TZ)` 推算报告期。路由层
  `route_to_vendor` 对 `get_fundamentals` 做 as-of 必填拒绝，正常调用不会缺失；
  但绕过路由直接调用 provider 时存在「未显式传日期却取到当前报告期」的语义缺口。
- 龙虎榜 `date` 仅支持一年内（接口约束）；超出返回参数错误。
- **`get_lhb_detail` 首次 4001 被 `fetch_with_date_fallback` 吞掉**：`_fetch_one` 对
  `4001` 频率超限直接抛 `FuyaoApiError`，而 `fetch_with_date_fallback` 的通用
  `except Exception` 会把它当作「该日无数据」继续向前回退，首日 4001 不做退避重试，
  导致静默回退到更早日（而非重试当日）。频率超限未达退避上限时语义被弱化。
- **交易日历 fallback 仅读 env，与 provider 配置优先不一致**：`trade_calendar`
  的 `_fetch_cn_trade_dates_from_fuyao` 只读 `os.getenv("FUYAO_API_KEY")`，
  而 `CnFuyaoProvider._resolve_api_key` 先读配置 `fuyao_api_key` 再读 env。
  若仅配置了 `fuyao_api_key`（未设 env），日历 fallback 不会启用，但 provider 本身可用。
- 交易日历 fallback 依赖 `FUYAO_API_KEY`；近一年窗口不足以覆盖更早历史查询。
- 涨跌停/龙虎榜备用链依赖 `cn_akshare.get_zt_pool` / `get_lhb_detail` 在东财失败时返回
  `VendorFail`（已改为显式 VendorFail），否则纯字符串会截断 vendor 链。

