# Known Issues

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

**Status:** Open (documented; not fixing this round)  
**Discovered:** 2026-07-29 during 1.58MB output investigation

### Symptom

`research_manager`, `risk_manager`, and `trader` do **not** receive any analyst
report directly in their prompt.  They only see debate history and summary fields.
`fundamentals_report` (and others) are read from state only to construct
`curr_situation`, which is used exclusively as a memory-retrieval embedding query —
the string is never injected into the prompt template.

### Prompt template coverage map (from `tradingagents/prompts/zh.py`)

| Agent | market | sentiment | news | fundamentals | smart_money | volume_price | macro |
|---|---|---|---|---|---|---|---|
| bull_researcher | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |
| bear_researcher | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |
| aggressive_debator | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |
| conservative_debator | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |
| neutral_debator | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |
| **research_manager** | — | ✓ (smart_money only) | — | **✗** | ✓ | ✓ | — |
| **trader** | — | — | — | **✗** | — | — | — |
| **risk_manager** | — | — | — | **✗** | — | — | — |

`research_manager` receives `smart_money_report`, `volume_price_report`, and
`sentiment_report` as raw data for its "expected-value gap analysis", but none of
market/news/fundamentals/macro.  `trader` and `risk_manager` receive only
structured summaries built by `build_agent_context_view` — no analyst reports at
all.

### Why it matters

Adjudicators decide direction and construct the final trade plan, but they can only
compare *how persuasively each debater argued* — they cannot independently verify
the underlying data.  In the 2026-07-29 600519 run, the fundamentals analyst gave
"中性", yet `research_manager` reached "偏空" exclusively from debate text and
volume-price signals; it had no way to cross-check the fundamentals claim.

This is an inherent structural constraint, not a bug per se, but it creates a
situation where strong rhetoric can outweigh weak evidence at the adjudication
layer.

### Suggested fix (future; do not implement until custom-prompt injection is live)

Do **not** pass full analyst reports to adjudicators — that would blow up context.
Instead, add a second output slot to each analyst:

```python
"fundamentals_evidence_summary": "<≤300字，仅包含A级事实和数字，无论断>"
```

Adjudicators receive the evidence summaries (not full reports) and can perform
evidence-level cross-checks rather than pure rhetoric comparison.  This also
enables `research_manager` to call out "多头方引用的基本面结论与原始数据不符"
type disagreements.

Implementation note: the 300-char cap is strict.  Evidence summaries must contain
only verifiable facts (numbers, dates, named events), no interpretations.  The
analyst's full report remains available in state for bull/bear to read.

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
