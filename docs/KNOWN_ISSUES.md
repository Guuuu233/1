# Known Issues

## Vendor chain collapses three outcomes into two behaviors

**Status:** Open (documented; not fixed in 3b-hotfix)  
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

### Suggested fix (future refactor; not this commit)

1. Introduce an explicit result type (or exception hierarchy), e.g.:
   - `VendorRefuse(reason)` — do not fall through to date-blind vendors; optional allow-list for same-semantics peers
   - `VendorFail(error)` — try next vendor
   - `VendorEmpty(confirmed=True)` — stop; prompt says confirmed none
   - `VendorOk(payload)`
2. Until that lands, **date-blind / near-window capabilities must be refused at `route_to_vendor` (or category policy)**, not only inside a single provider.
3. Re-enabling investoday (or any historical-capable source) for news must be an **explicit same-source whitelist for both live and historical modes**, never a silent fallback hit.

### Regression guard

3c-3 e2e guards (missing `curr_date` hard-fail, dual historical-date upper-bound comparison, provider signature whitelist) are intended to catch silent reintroduction of undated live data on historical paths. They do **not** replace a typed vendor-result redesign.
