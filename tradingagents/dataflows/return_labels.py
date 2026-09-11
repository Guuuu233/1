"""Multi-horizon return labels and trading calendar window contracts (V-01-1 & V-01-2).

This module defines the type contracts, immutable calendar window resolution,
and real settlement pipeline execution for multi-horizon performance evaluation
(short T+10, medium T+40).

Strict V-01-1 & V-01-2 boundaries:
1. Pure functions and data structures with fail-closed semantics.
2. Reads evaluation offsets directly from canonical HORIZON_PROFILE_V1.
3. HorizonCalendarWindow outputs strict boolean is_due.
4. resolve_horizon_return_label implements real settlement with strict T+1 entry viability,
   trading calendar roll, bilateral suspension evidence, and total/price return support.
5. Dividends/splits are implemented as result fields; OutcomeStatus remains exact 7 states.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple, TypedDict

from tradingagents.graph.horizon_profile import (
    HORIZON_MEDIUM,
    HORIZON_PROFILE_V1,
    HORIZON_SHORT,
    SUPPORTED_HORIZONS,
)

# ---------------------------------------------------------------------------
# Outcome & Return Types
# ---------------------------------------------------------------------------


class OutcomeStatus(str, Enum):
    """V-01 outcome status enumeration (7 mutually exclusive terminal/interim states).

    Defined here as a shared type contract for downstream evaluation engines (V-01-2+).
    """

    PENDING_DUE = "pending_due"
    SUSPENSION = "suspension"
    DATA_MISSING = "data_missing"
    PROVIDER_FAILURE = "provider_failure"
    UNSUPPORTED_PRICE_BASIS = "unsupported_price_basis"
    UNEXECUTABLE_ENTRY = "unexecutable_entry"
    EVALUATED_OK = "evaluated_ok"


class ReturnType(str, Enum):
    """V-01 return computation type."""

    PRICE_RETURN = "price_return"
    TOTAL_RETURN = "total_return"


# ---------------------------------------------------------------------------
# Price Basis Constants (DAV-606, DAV-705, DAV-830)
# ---------------------------------------------------------------------------

PRICE_BASIS_VENDOR_QFQ: str = "vendor_qfq"
PRICE_BASIS_RAW: str = "raw"
PRICE_BASIS_UNSPECIFIED: str = "unspecified"
PRICE_BASIS_PIT_RAW: str = "pit_raw"
PRICE_BASIS_PIT_ADJUSTED: str = "pit_adjusted"

SUPPORTED_PRICE_BASES: frozenset[str] = frozenset({
    PRICE_BASIS_VENDOR_QFQ,
    PRICE_BASIS_RAW,
})

RETURN_TYPE_PRICE_RETURN: str = ReturnType.PRICE_RETURN.value
RETURN_TYPE_TOTAL_RETURN: str = ReturnType.TOTAL_RETURN.value
SUPPORTED_RETURN_TYPES: frozenset[str] = frozenset({
    RETURN_TYPE_PRICE_RETURN,
    RETURN_TYPE_TOTAL_RETURN,
})

# ---------------------------------------------------------------------------
# Authoritative Audited Label Contract Specification (v1.1 §8 V-01 / DAV-830)
# ---------------------------------------------------------------------------

LABEL_CONTRACT_V1_SPEC: str = """Multi-Horizon Return Labels Contract Specification (v1.1 §8 V-01 / DAV-830)

经审定的标签契约文本（五大核心概念严格区分）：

1. T 定义 (Signal Date / Analysis Date):
   - 信号产生或分析发布的基准交易日 (signal_date / trade_date)。
   - 严格约束：必须是真实交易日历中真实存在的有效交易日 (严格 ISO YYYY-MM-DD 校验且精确存在于 trading_days 序列中，禁止 bisect 漂移)。
   - 不变性约束：T 日收盘价不能作为 T 日盘后信号的执行成交价，严防前瞻偏差 (look-ahead bias) 与盘后偷价。

2. cutoff 资格 (Cutoff Eligibility / Maturity / is_due / 样本成熟度):
   - 样本在评估基准日 as_of 时，其持有评价窗口是否已完全走完并具备被评估资格。
   - 严格四象限判定规则：
     a. 若 target_calendar_date > as_of：未到期，is_due = False，标记 pending_due，禁止截断缩短持有天数 (如禁止将 T+40 缩短为 20 天计算)。
     b. 若 target_calendar_date == as_of 且 as_of_market_closed == False (盘中未收盘)：is_due = False，标记 pending_due，防止盘中偷取未定收盘价。
     c. 仅当 target_calendar_date < as_of，或 (target_calendar_date == as_of 且 as_of_market_closed == True) 时，样本才具备 cutoff 资格 (is_due = True)。

3. 主评价日 (Primary Evaluation Date / Target Calendar Date):
   - 依据 canonical HORIZON_PROFILE_V1 中设定的主评价偏移量 primary_eval_offset (short 为 10，medium 为 40)，在真实交易日历中严格顺延第 N 个交易日 (T+N)。
   - 严格约束：基于真实交易日历序列严格索引，严禁使用 DataFrame 行数切片 (iloc[hold_days - 1]) 或日历自然日加法。
   - 展期机制 (Roll Policy)：target_calendar_date 自身作为基准锚点不变；若目标日停牌或无法成交退出，允许在严格受限的 max_roll_days (short: 2天, medium: 5天) 候选日中顺延寻找第一个可成交退出日 (actual_exit_date)，并记录 roll_days_used；超限仍无法退出则判定为 suspension。

4. 研究基准 (Research Benchmark / Profile & Return Basis):
   - 收益衡量与属性归属的基准系统：
     a. 期限档位 (Horizon)：short (T+10, roll<=2) 与 medium (T+40, roll<=5)。
     b. 配置 profile：canonical horizon_profile_v1。
     c. 价格基准 (price_basis)：vendor_qfq (前复权) 与 raw (不复权)；未支持或未指定基准 fail-closed 标记 unsupported_price_basis。
     d. 收益类型 (return_type)：price_return (纯价格收益) 与 total_return (全收益，计入 cash_dividend_total 与 split_ratio_total)；缺分红数据不得静默按 0 计。
     e. 隔离约束：不同档位、不同 profile 的样本严格物理隔离，严禁跨档跨配置混算。

5. 可执行入场 (Executable Entry Date & Price / Execution Viability):
   - T 日生成信号后，最早可执行入场时间为 T+1 开盘 (executable_entry_date = trading_days[signal_idx + 1])。
   - 真实性与可执行性核验：若 T+1 日停牌 (is_suspended、成交量为 0、或开盘收盘为 0)，或一字涨停无法买入 (open == high == low == limit_up)，则入场不可执行，判定为 unexecutable_entry。
   - 严格禁止假装以开盘价或假想价成交；未执行入场的样本不计算收益 (return_pct = None)，evaluation_eligible = False。
"""


# ---------------------------------------------------------------------------
# Profile Constants & Max-Roll Mapping
# ---------------------------------------------------------------------------

HORIZON_PROFILE_ID_V1: str = "horizon_profile_v1"

# Primary eval offsets read directly from canonical HORIZON_PROFILE_V1 truth source
PRIMARY_EVAL_OFFSET_SHORT: int = int(HORIZON_PROFILE_V1[HORIZON_SHORT]["primary_eval_offset"])
PRIMARY_EVAL_OFFSET_MEDIUM: int = int(HORIZON_PROFILE_V1[HORIZON_MEDIUM]["primary_eval_offset"])

# Immutable max-roll mapping (short: 2 days, medium: 5 days) wrapped in MappingProxyType
HORIZON_MAX_ROLL_DAYS: Mapping[str, int] = MappingProxyType({
    HORIZON_SHORT: 2,
    HORIZON_MEDIUM: 5,
})

# Module-level consistency assertions against canonical HORIZON_PROFILE_V1
assert PRIMARY_EVAL_OFFSET_SHORT == 10, "HORIZON_PROFILE_V1 short offset drift detected"
assert PRIMARY_EVAL_OFFSET_MEDIUM == 40, "HORIZON_PROFILE_V1 medium offset drift detected"
assert set(HORIZON_MAX_ROLL_DAYS.keys()) == set(SUPPORTED_HORIZONS), (
    "HORIZON_MAX_ROLL_DAYS keys must match SUPPORTED_HORIZONS"
)


# ---------------------------------------------------------------------------
# Result TypedDict Contract (V-01 Shared Contract)
# ---------------------------------------------------------------------------


class HorizonReturnResult(TypedDict):
    """Full-pipeline return label result contract.

    Defined in V-01-1 as a pure type contract only; values are computed and populated in V-01-2+.
    """

    symbol: str
    horizon: str
    profile_id: str
    price_basis: str
    return_type: str
    entry_date: str
    executable_entry_date: Optional[str]
    target_calendar_date: Optional[str]
    actual_exit_date: Optional[str]
    roll_days_used: int
    entry_signal_price: Optional[float]
    entry_executable_price: Optional[float]
    entry_price: Optional[float]
    exit_price: Optional[float]
    cash_dividend_total: float
    split_ratio_total: float
    return_pct: Optional[float]
    outcome_status: str
    is_direction_hit: Optional[bool]
    evaluation_eligible: bool


# ---------------------------------------------------------------------------
# Exceptions & Calendar Window Dataclass
# ---------------------------------------------------------------------------


class InsufficientTradingCalendarError(ValueError):
    """Raised when trading_days sequence does not sufficiently cover T+N+max_roll."""

    pass


@dataclass(frozen=True)
class HorizonCalendarWindow:
    """Pure calendar window contract for multi-horizon analysis.

    Does not compute prices or returns.
    Does not output OutcomeStatus.EVALUATED_OK and does not claim evaluation.
    Only outputs strict boolean is_due.
    """

    signal_date: str
    horizon: str
    eval_offset: int
    max_roll_days: int
    executable_entry_date: str
    target_calendar_date: str
    roll_candidate_dates: Tuple[str, ...]
    holding_trading_days: Tuple[str, ...]
    is_due: bool


# ---------------------------------------------------------------------------
# Validation Helpers & Pure Resolution Function
# ---------------------------------------------------------------------------

_ISO_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_iso_date(d: Any, param_name: str) -> str:
    """Validate that d is a strict ISO date string YYYY-MM-DD representing a real calendar date."""
    if not isinstance(d, str):
        raise ValueError(f"{param_name} must be a string, got {type(d).__name__}: {d!r}")
    if not _ISO_DATE_REGEX.match(d):
        raise ValueError(f"{param_name} must be in strict ISO format YYYY-MM-DD, got {d!r}")
    try:
        parts = d.split("-")
        date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, TypeError) as e:
        raise ValueError(f"{param_name} is not a valid calendar date ({d!r}): {e}") from e
    return d


def resolve_horizon_calendar_window(
    *,
    signal_date: str,
    horizon: str,
    trading_days: Sequence[str],
    as_of: str,
    as_of_market_closed: bool,
) -> HorizonCalendarWindow:
    """Resolve a multi-horizon trading calendar window based strictly on an explicit trading days sequence.

    Invariants:
    1. Zero network requests, zero provider calls.
    2. Zero side effects: does not mutate, sort, or deduplicate the input trading_days sequence.
    3. Strict input validation: non-string Sequence, strictly sorted, duplicate-free real ISO dates.
    4. Exact signal_date presence: no bisect backwards/forwards shifting.
    5. Fails closed with ValueError (or InsufficientTradingCalendarError) if calendar coverage is less than T+N+max_roll.
    6. Does not compute prices or returns; does not output OutcomeStatus.EVALUATED_OK.
    """
    # 1. Validate horizon: must be native str (reject str Enum, plain Enum, stringifiable objects)
    if type(horizon) is not str or horizon not in SUPPORTED_HORIZONS:
        raise ValueError(
            f"Unsupported horizon {horizon!r}. Supported horizons: {list(SUPPORTED_HORIZONS)}"
        )

    # 2. Validate signal_date
    _validate_iso_date(signal_date, "signal_date")

    # 3. Validate as_of
    _validate_iso_date(as_of, "as_of")

    # 4. Validate as_of_market_closed: must be strict bool
    if not isinstance(as_of_market_closed, bool):
        raise ValueError(
            f"as_of_market_closed must be a strict bool, got {type(as_of_market_closed).__name__}: {as_of_market_closed!r}"
        )

    # 5. Validate trading_days
    if isinstance(trading_days, (str, bytes)):
        raise ValueError("trading_days must be a non-string Sequence, got string/bytes")
    if not isinstance(trading_days, Sequence):
        raise ValueError(f"trading_days must be a Sequence, got {type(trading_days).__name__}")
    if len(trading_days) == 0:
        raise ValueError("trading_days sequence cannot be empty")

    # Validate elements, ordering, and uniqueness without modifying input
    prev_day: Optional[str] = None
    for i, day in enumerate(trading_days):
        _validate_iso_date(day, f"trading_days[{i}]")
        if prev_day is not None:
            if day == prev_day:
                raise ValueError(f"trading_days contains duplicate date at index {i}: {day!r}")
            if day < prev_day:
                raise ValueError(
                    f"trading_days is not strictly sorted in ascending order at index {i}: {prev_day!r} >= {day!r}"
                )
        prev_day = day

    # 6. Locate signal_date (exact match, no bisect drifting)
    try:
        signal_idx = trading_days.index(signal_date)
    except ValueError:
        raise ValueError(f"signal_date {signal_date!r} not found in trading_days")

    # 7. Eval offset & max_roll_days
    eval_offset = int(HORIZON_PROFILE_V1[horizon]["primary_eval_offset"])
    max_roll_days = HORIZON_MAX_ROLL_DAYS[horizon]

    # 8. Calendar coverage check (must cover up to T+N+max_roll)
    required_index = signal_idx + eval_offset + max_roll_days
    if len(trading_days) <= required_index:
        raise InsufficientTradingCalendarError(
            f"trading_days does not cover up to T+{eval_offset}+{max_roll_days}: "
            f"needs index {required_index}, but length is {len(trading_days)} "
            f"(signal_idx={signal_idx}, offset={eval_offset}, max_roll={max_roll_days})"
        )

    # 9. Extract window components
    executable_entry_date = trading_days[signal_idx + 1]
    target_calendar_date = trading_days[signal_idx + eval_offset]
    holding_trading_days = tuple(trading_days[signal_idx + 1 : signal_idx + eval_offset + 1])
    roll_candidate_dates = tuple(
        trading_days[signal_idx + eval_offset + 1 : signal_idx + eval_offset + max_roll_days + 1]
    )

    # 10. Due calculation
    if target_calendar_date < as_of:
        is_due = True
    elif target_calendar_date == as_of:
        is_due = (as_of_market_closed is True)
    else:
        is_due = False

    return HorizonCalendarWindow(
        signal_date=signal_date,
        horizon=horizon,
        eval_offset=eval_offset,
        max_roll_days=max_roll_days,
        executable_entry_date=executable_entry_date,
        target_calendar_date=target_calendar_date,
        roll_candidate_dates=roll_candidate_dates,
        holding_trading_days=holding_trading_days,
        is_due=is_due,
    )


# ---------------------------------------------------------------------------
# Bar Field Extraction & Viability Helpers (V-01-2)
# ---------------------------------------------------------------------------


def _extract_bar_fields(
    bar: Any,
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    float,
    bool,
    Optional[float],
    Optional[float],
]:
    """Extract (open, high, low, close, volume, is_suspended, limit_up, limit_down) from bar.

    Accepts:
    - dict with standard / capitalized / Chinese column keys
    - Object / dataclass with attribute access
    - float / int directly representing close price
    """
    if bar is None:
        return None, None, None, None, 0.0, False, None, None

    if isinstance(bar, (int, float)):
        val = float(bar)
        is_susp = val <= 0.0
        return val, val, val, val, (0.0 if is_susp else 1000.0), is_susp, None, None

    open_val: Optional[float] = None
    high_val: Optional[float] = None
    low_val: Optional[float] = None
    close_val: Optional[float] = None
    volume_val: float = 0.0
    is_susp: bool = False
    limit_up_val: Optional[float] = None
    limit_down_val: Optional[float] = None

    if isinstance(bar, Mapping):
        def _get(keys: Sequence[str]) -> Optional[Any]:
            for k in keys:
                if k in bar:
                    return bar[k]
            return None

        o = _get(["open", "Open", "OPEN", "开盘", "开盘价"])
        h = _get(["high", "High", "HIGH", "最高", "最高价"])
        l = _get(["low", "Low", "LOW", "最低", "最低价"])
        c = _get(["close", "Close", "CLOSE", "收盘", "收盘价", "price", "Price"])
        v = _get(["volume", "Volume", "VOLUME", "成交量", "vol", "Vol"])
        s = _get(["is_suspended", "suspended", "is_susp", "停牌"])
        lup = _get(["limit_up", "limit_up_price", "涨停", "涨停价"])
        ldown = _get(["limit_down", "limit_down_price", "跌停", "跌停价"])

        if o is not None:
            try:
                open_val = float(o)
            except (ValueError, TypeError):
                pass
        if h is not None:
            try:
                high_val = float(h)
            except (ValueError, TypeError):
                pass
        if l is not None:
            try:
                low_val = float(l)
            except (ValueError, TypeError):
                pass
        if c is not None:
            try:
                close_val = float(c)
            except (ValueError, TypeError):
                pass
        if v is not None:
            try:
                volume_val = float(v)
            except (ValueError, TypeError):
                pass
        if s is not None:
            is_susp = bool(s)
        if lup is not None:
            try:
                limit_up_val = float(lup)
            except (ValueError, TypeError):
                pass
        if ldown is not None:
            try:
                limit_down_val = float(ldown)
            except (ValueError, TypeError):
                pass
    else:
        for attr in ("open", "Open"):
            if hasattr(bar, attr):
                try:
                    open_val = float(getattr(bar, attr))
                    break
                except (ValueError, TypeError):
                    pass
        for attr in ("high", "High"):
            if hasattr(bar, attr):
                try:
                    high_val = float(getattr(bar, attr))
                    break
                except (ValueError, TypeError):
                    pass
        for attr in ("low", "Low"):
            if hasattr(bar, attr):
                try:
                    low_val = float(getattr(bar, attr))
                    break
                except (ValueError, TypeError):
                    pass
        for attr in ("close", "Close", "price"):
            if hasattr(bar, attr):
                try:
                    close_val = float(getattr(bar, attr))
                    break
                except (ValueError, TypeError):
                    pass
        for attr in ("volume", "Volume", "vol"):
            if hasattr(bar, attr):
                try:
                    volume_val = float(getattr(bar, attr))
                    break
                except (ValueError, TypeError):
                    pass
        for attr in ("is_suspended", "suspended"):
            if hasattr(bar, attr):
                is_susp = bool(getattr(bar, attr))
                break
        for attr in ("limit_up", "limit_up_price"):
            if hasattr(bar, attr):
                try:
                    limit_up_val = float(getattr(bar, attr))
                    break
                except (ValueError, TypeError):
                    pass
        for attr in ("limit_down", "limit_down_price"):
            if hasattr(bar, attr):
                try:
                    limit_down_val = float(getattr(bar, attr))
                    break
                except (ValueError, TypeError):
                    pass

    # Zero volume or zero prices without explicit flag indicate suspension
    if not is_susp:
        if volume_val <= 0.0 and (open_val is not None or close_val is not None):
            is_susp = True
        elif open_val == 0.0 and close_val == 0.0:
            is_susp = True

    return open_val, high_val, low_val, close_val, volume_val, is_susp, limit_up_val, limit_down_val


# ---------------------------------------------------------------------------
# Real Settlement Pipeline: resolve_horizon_return_label (V-01-2)
# ---------------------------------------------------------------------------


def resolve_horizon_return_label(
    *,
    symbol: str,
    signal_date: str,
    horizon: str,
    trading_days: Sequence[str],
    as_of: str,
    as_of_market_closed: bool,
    price_basis: str = PRICE_BASIS_VENDOR_QFQ,
    return_type: str = ReturnType.PRICE_RETURN.value,
    profile_id: str = HORIZON_PROFILE_ID_V1,
    direction: Optional[str] = "BUY",
    bar_data: Optional[Mapping[str, Any]] = None,
    price_fetcher: Optional[Callable[[str, str], Any]] = None,
    dividend_data: Optional[Mapping[str, float]] = None,
    split_data: Optional[Mapping[str, float]] = None,
) -> HorizonReturnResult:
    """Resolve and evaluate multi-horizon return label with real market settlement rules (V-01-2).

    Enforces all 5 core concepts of LABEL_CONTRACT_V1_SPEC:
    1. T definition (signal_date in trading_days, no look-ahead).
    2. Cutoff eligibility (strict 4-quadrant is_due check; pending_due never truncates hold window).
    3. Primary evaluation date (canonical T+N offset from trading calendar, never row slicing).
    4. Research benchmark (strict horizon short/medium, supported price bases, total/price return).
    5. Executable entry (T+1 tradability check; suspension/limit-up locks -> unexecutable_entry; no fake fills).

    OutcomeStatus coverage (all 7 terminal states, no additions):
    - pending_due: evaluation window not matured as of as_of.
    - unexecutable_entry: T+1 suspended or locked at limit-up/limit-down.
    - unsupported_price_basis: price basis unsupported or unknown.
    - suspension: target/holding period objectively suspended with bilateral evidence.
    - data_missing: price or required dividend data missing without bilateral suspension proof.
    - provider_failure: data fetcher/provider exception or malformed return.
    - evaluated_ok: successfully settled return label.
    """
    # 1. Symbol validation
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"symbol must be a non-empty string, got {symbol!r}")
    clean_symbol = symbol.strip()

    # 2. Return type validation
    if return_type not in SUPPORTED_RETURN_TYPES:
        raise ValueError(
            f"Unsupported return_type {return_type!r}. Supported return types: {list(SUPPORTED_RETURN_TYPES)}"
        )

    # 3. Price basis fail-closed validation
    if price_basis not in SUPPORTED_PRICE_BASES:
        return HorizonReturnResult(
            symbol=clean_symbol,
            horizon=horizon if type(horizon) is str and horizon in SUPPORTED_HORIZONS else "unknown",
            profile_id=profile_id,
            price_basis=str(price_basis),
            return_type=return_type,
            entry_date=signal_date,
            executable_entry_date=None,
            target_calendar_date=None,
            actual_exit_date=None,
            roll_days_used=0,
            entry_signal_price=None,
            entry_executable_price=None,
            entry_price=None,
            exit_price=None,
            cash_dividend_total=0.0,
            split_ratio_total=1.0,
            return_pct=None,
            outcome_status=OutcomeStatus.UNSUPPORTED_PRICE_BASIS.value,
            is_direction_hit=None,
            evaluation_eligible=False,
        )

    # 4. Resolve calendar window (validates signal_date, horizon, trading_days, as_of, as_of_market_closed)
    window = resolve_horizon_calendar_window(
        signal_date=signal_date,
        horizon=horizon,
        trading_days=trading_days,
        as_of=as_of,
        as_of_market_closed=as_of_market_closed,
    )

    # 5. Check Cutoff Eligibility (Maturity)
    if not window.is_due:
        return HorizonReturnResult(
            symbol=clean_symbol,
            horizon=horizon,
            profile_id=profile_id,
            price_basis=price_basis,
            return_type=return_type,
            entry_date=signal_date,
            executable_entry_date=window.executable_entry_date,
            target_calendar_date=window.target_calendar_date,
            actual_exit_date=None,
            roll_days_used=0,
            entry_signal_price=None,
            entry_executable_price=None,
            entry_price=None,
            exit_price=None,
            cash_dividend_total=0.0,
            split_ratio_total=1.0,
            return_pct=None,
            outcome_status=OutcomeStatus.PENDING_DUE.value,
            is_direction_hit=None,
            evaluation_eligible=False,
        )

    # 6. Retrieve bars via bar_data, price_fetcher, or provider
    fetched_bars: dict[str, Any] = {}
    provider_failed = False

    def _fetch_single_bar(d: str) -> Optional[Any]:
        nonlocal provider_failed
        if bar_data is not None:
            return bar_data.get(d)
        if price_fetcher is not None:
            try:
                return price_fetcher(clean_symbol, d)
            except Exception:
                provider_failed = True
                return None
        return None

    # If neither bar_data nor price_fetcher is supplied, attempt vendor route
    if bar_data is None and price_fetcher is None:
        try:
            from tradingagents.dataflows.interface import route_to_vendor
            import pandas as pd
            import io

            start_fetch = window.signal_date
            end_fetch = window.roll_candidate_dates[-1] if window.roll_candidate_dates else window.target_calendar_date
            csv_data = route_to_vendor("get_stock_data", clean_symbol, start_fetch, end_fetch)
            if not csv_data or str(csv_data).startswith("【数据获取失败】") or str(csv_data).startswith("No data found"):
                provider_failed = True
            else:
                lines = [l for l in str(csv_data).splitlines() if not l.startswith("#") and l.strip()]
                if lines:
                    df = pd.read_csv(io.StringIO("\n".join(lines)))
                    date_cols = [c for c in df.columns if "date" in c.lower() or "日期" in c or "time" in c.lower()]
                    if date_cols:
                        d_col = date_cols[0]
                        df[d_col] = df[d_col].astype(str).str[:10]
                        for _, row in df.iterrows():
                            fetched_bars[str(row[d_col])] = dict(row)
        except Exception:
            provider_failed = True

    if provider_failed:
        return HorizonReturnResult(
            symbol=clean_symbol,
            horizon=horizon,
            profile_id=profile_id,
            price_basis=price_basis,
            return_type=return_type,
            entry_date=signal_date,
            executable_entry_date=window.executable_entry_date,
            target_calendar_date=window.target_calendar_date,
            actual_exit_date=None,
            roll_days_used=0,
            entry_signal_price=None,
            entry_executable_price=None,
            entry_price=None,
            exit_price=None,
            cash_dividend_total=0.0,
            split_ratio_total=1.0,
            return_pct=None,
            outcome_status=OutcomeStatus.PROVIDER_FAILURE.value,
            is_direction_hit=None,
            evaluation_eligible=False,
        )

    def _get_bar(d: str) -> Optional[Any]:
        if d in fetched_bars:
            return fetched_bars[d]
        b = _fetch_single_bar(d)
        if b is not None:
            fetched_bars[d] = b
        return b

    # Retrieve T (signal) bar for signal price logging
    signal_bar = _get_bar(window.signal_date)
    if provider_failed:
        return HorizonReturnResult(
            symbol=clean_symbol,
            horizon=horizon,
            profile_id=profile_id,
            price_basis=price_basis,
            return_type=return_type,
            entry_date=signal_date,
            executable_entry_date=window.executable_entry_date,
            target_calendar_date=window.target_calendar_date,
            actual_exit_date=None,
            roll_days_used=0,
            entry_signal_price=None,
            entry_executable_price=None,
            entry_price=None,
            exit_price=None,
            cash_dividend_total=0.0,
            split_ratio_total=1.0,
            return_pct=None,
            outcome_status=OutcomeStatus.PROVIDER_FAILURE.value,
            is_direction_hit=None,
            evaluation_eligible=False,
        )
    _, _, _, sig_close, _, _, _, _ = _extract_bar_fields(signal_bar)
    entry_signal_price = sig_close

    # 7. Check Executable Entry Viability at T+1
    entry_date = window.executable_entry_date
    entry_bar = _get_bar(entry_date)

    if provider_failed:
        return HorizonReturnResult(
            symbol=clean_symbol,
            horizon=horizon,
            profile_id=profile_id,
            price_basis=price_basis,
            return_type=return_type,
            entry_date=signal_date,
            executable_entry_date=entry_date,
            target_calendar_date=window.target_calendar_date,
            actual_exit_date=None,
            roll_days_used=0,
            entry_signal_price=entry_signal_price,
            entry_executable_price=None,
            entry_price=None,
            exit_price=None,
            cash_dividend_total=0.0,
            split_ratio_total=1.0,
            return_pct=None,
            outcome_status=OutcomeStatus.PROVIDER_FAILURE.value,
            is_direction_hit=None,
            evaluation_eligible=False,
        )

    if entry_bar is None:
        # Check bilateral evidence for T+1
        has_preceding = entry_signal_price is not None and entry_signal_price > 0
        has_subsequent = False
        signal_idx = trading_days.index(signal_date)
        if signal_idx + 2 < len(trading_days):
            t2_bar = _get_bar(trading_days[signal_idx + 2])
            if t2_bar is not None:
                _, _, _, t2_c, t2_v, _, _, _ = _extract_bar_fields(t2_bar)
                if t2_c is not None and t2_c > 0 and t2_v > 0:
                    has_subsequent = True
        entry_status = OutcomeStatus.SUSPENSION.value if (has_preceding and has_subsequent) else OutcomeStatus.DATA_MISSING.value

        return HorizonReturnResult(
            symbol=clean_symbol,
            horizon=horizon,
            profile_id=profile_id,
            price_basis=price_basis,
            return_type=return_type,
            entry_date=signal_date,
            executable_entry_date=entry_date,
            target_calendar_date=window.target_calendar_date,
            actual_exit_date=None,
            roll_days_used=0,
            entry_signal_price=entry_signal_price,
            entry_executable_price=None,
            entry_price=None,
            exit_price=None,
            cash_dividend_total=0.0,
            split_ratio_total=1.0,
            return_pct=None,
            outcome_status=entry_status,
            is_direction_hit=None,
            evaluation_eligible=False,
        )

    (
        e_open,
        e_high,
        e_low,
        e_close,
        e_vol,
        e_susp,
        e_lup,
        e_ldown,
    ) = _extract_bar_fields(entry_bar)

    clean_dir = str(direction or "BUY").strip().upper()
    is_bull = any(k in clean_dir for k in ("BUY", "BULL", "多", "看多", "增持"))
    is_bear = any(k in clean_dir for k in ("SELL", "BEAR", "空", "看空", "减持"))

    is_entry_unexecutable = False

    # Check suspension / zero volume / invalid price
    if e_susp or e_vol <= 0.0 or (e_open is not None and e_open <= 0.0) or (e_close is not None and e_close <= 0.0):
        is_entry_unexecutable = True

    # Check limit-up locked when buying (open == high == low == limit_up)
    if is_bull and e_lup is not None and e_open is not None and e_open >= e_lup:
        if e_high is not None and e_low is not None and e_high == e_low == e_lup:
            is_entry_unexecutable = True

    # Check limit-down locked when selling (open == high == low == limit_down)
    if is_bear and e_ldown is not None and e_open is not None and e_open <= e_ldown:
        if e_high is not None and e_low is not None and e_high == e_low == e_ldown:
            is_entry_unexecutable = True

    if is_entry_unexecutable:
        return HorizonReturnResult(
            symbol=clean_symbol,
            horizon=horizon,
            profile_id=profile_id,
            price_basis=price_basis,
            return_type=return_type,
            entry_date=signal_date,
            executable_entry_date=entry_date,
            target_calendar_date=window.target_calendar_date,
            actual_exit_date=None,
            roll_days_used=0,
            entry_signal_price=entry_signal_price,
            entry_executable_price=None,
            entry_price=None,
            exit_price=None,
            cash_dividend_total=0.0,
            split_ratio_total=1.0,
            return_pct=None,
            outcome_status=OutcomeStatus.UNEXECUTABLE_ENTRY.value,
            is_direction_hit=None,
            evaluation_eligible=False,
        )

    # Entry is executable: fill at T+1 open (or close if open missing)
    entry_executable_price = e_open if e_open is not None and e_open > 0 else e_close
    entry_price = entry_executable_price

    # 8. Check Exit Tradability and Roll Candidates
    exit_candidates = (window.target_calendar_date,) + window.roll_candidate_dates
    actual_exit_date: Optional[str] = None
    actual_exit_price: Optional[float] = None
    roll_days_used: int = 0

    for r_idx, cand_date in enumerate(exit_candidates):
        cand_bar = _get_bar(cand_date)
        if cand_bar is None:
            continue
        c_open, c_high, c_low, c_close, c_vol, c_susp, c_lup, c_ldown = _extract_bar_fields(cand_bar)
        if c_susp or c_vol <= 0.0 or c_close is None or c_close <= 0.0:
            continue
        # Exit limit lock check: if locked at limit on exit, cannot execute exit on this day
        if is_bull and c_ldown is not None and c_open is not None and c_open <= c_ldown:
            if c_high is not None and c_low is not None and c_high == c_low == c_ldown:
                continue
        if is_bear and c_lup is not None and c_open is not None and c_open >= c_lup:
            if c_high is not None and c_low is not None and c_high == c_low == c_lup:
                continue

        actual_exit_date = cand_date
        actual_exit_price = c_close
        roll_days_used = r_idx
        break

    # 9. Handle Non-Exit: Bilateral Suspension Evidence vs Data Missing
    if actual_exit_date is None or actual_exit_price is None or actual_exit_price <= 0:
        has_preceding_price = entry_price is not None and entry_price > 0
        has_subsequent_price = False
        signal_idx = trading_days.index(signal_date)
        eval_offset = window.eval_offset
        max_roll = window.max_roll_days
        subsequent_idx = signal_idx + eval_offset + max_roll + 1
        if subsequent_idx < len(trading_days):
            sub_date = trading_days[subsequent_idx]
            sub_bar = _get_bar(sub_date)
            if sub_bar is not None:
                _, _, _, s_close, s_vol, _, _, _ = _extract_bar_fields(sub_bar)
                if s_close is not None and s_close > 0 and s_vol > 0:
                    has_subsequent_price = True

        target_bar = _get_bar(window.target_calendar_date)
        target_explicit_susp = False
        if target_bar is not None:
            _, _, _, _, _, t_susp, _, _ = _extract_bar_fields(target_bar)
            target_explicit_susp = t_susp

        if (has_preceding_price and has_subsequent_price) or target_explicit_susp:
            final_status = OutcomeStatus.SUSPENSION.value
        else:
            final_status = OutcomeStatus.DATA_MISSING.value

        return HorizonReturnResult(
            symbol=clean_symbol,
            horizon=horizon,
            profile_id=profile_id,
            price_basis=price_basis,
            return_type=return_type,
            entry_date=signal_date,
            executable_entry_date=entry_date,
            target_calendar_date=window.target_calendar_date,
            actual_exit_date=None,
            roll_days_used=0,
            entry_signal_price=entry_signal_price,
            entry_executable_price=entry_executable_price,
            entry_price=entry_price,
            exit_price=None,
            cash_dividend_total=0.0,
            split_ratio_total=1.0,
            return_pct=None,
            outcome_status=final_status,
            is_direction_hit=None,
            evaluation_eligible=False,
        )

    # 10. Calculate Return (price_return vs total_return)
    assert entry_price is not None and entry_price > 0
    assert actual_exit_price is not None and actual_exit_price > 0
    assert actual_exit_date is not None

    if return_type == ReturnType.TOTAL_RETURN.value:
        # RT-5 & RT-9: Atomic completeness for total_return.
        # Both dividend_data and split_data MUST be present.
        # If either is None, fail-closed to DATA_MISSING to prevent fake crashes (-50% drop on unadjusted split).
        if dividend_data is None or split_data is None:
            return HorizonReturnResult(
                symbol=clean_symbol,
                horizon=horizon,
                profile_id=profile_id,
                price_basis=price_basis,
                return_type=return_type,
                entry_date=signal_date,
                executable_entry_date=entry_date,
                target_calendar_date=window.target_calendar_date,
                actual_exit_date=actual_exit_date,
                roll_days_used=roll_days_used,
                entry_signal_price=entry_signal_price,
                entry_executable_price=entry_executable_price,
                entry_price=entry_price,
                exit_price=actual_exit_price,
                cash_dividend_total=0.0,
                split_ratio_total=1.0,
                return_pct=None,
                outcome_status=OutcomeStatus.DATA_MISSING.value,
                is_direction_hit=None,
                evaluation_eligible=False,
            )

        cash_div_total = 0.0
        for d_str, div_val in dividend_data.items():
            if entry_date < d_str <= actual_exit_date:
                try:
                    cash_div_total += float(div_val)
                except (ValueError, TypeError):
                    pass

        split_total = 1.0
        for s_str, s_val in split_data.items():
            if entry_date < s_str <= actual_exit_date:
                try:
                    split_total *= float(s_val)
                except (ValueError, TypeError):
                    pass

        effective_exit = actual_exit_price * split_total + cash_div_total
        raw_ret = (effective_exit - entry_price) / entry_price * 100.0

        if is_bear:
            return_pct = round(-raw_ret, 4)
            is_hit = bool(effective_exit < entry_price)
        else:
            return_pct = round(raw_ret, 4)
            is_hit = bool(effective_exit > entry_price)

        return HorizonReturnResult(
            symbol=clean_symbol,
            horizon=horizon,
            profile_id=profile_id,
            price_basis=price_basis,
            return_type=return_type,
            entry_date=signal_date,
            executable_entry_date=entry_date,
            target_calendar_date=window.target_calendar_date,
            actual_exit_date=actual_exit_date,
            roll_days_used=roll_days_used,
            entry_signal_price=entry_signal_price,
            entry_executable_price=entry_executable_price,
            entry_price=entry_price,
            exit_price=actual_exit_price,
            cash_dividend_total=round(cash_div_total, 4),
            split_ratio_total=round(split_total, 4),
            return_pct=return_pct,
            outcome_status=OutcomeStatus.EVALUATED_OK.value,
            is_direction_hit=is_hit,
            evaluation_eligible=True,
        )

    # Standard price_return
    raw_ret = (actual_exit_price - entry_price) / entry_price * 100.0
    if is_bear:
        return_pct = round(-raw_ret, 4)
        is_hit = bool(actual_exit_price < entry_price)
    else:
        return_pct = round(raw_ret, 4)
        is_hit = bool(actual_exit_price > entry_price)

    return HorizonReturnResult(
        symbol=clean_symbol,
        horizon=horizon,
        profile_id=profile_id,
        price_basis=price_basis,
        return_type=return_type,
        entry_date=signal_date,
        executable_entry_date=entry_date,
        target_calendar_date=window.target_calendar_date,
        actual_exit_date=actual_exit_date,
        roll_days_used=roll_days_used,
        entry_signal_price=entry_signal_price,
        entry_executable_price=entry_executable_price,
        entry_price=entry_price,
        exit_price=actual_exit_price,
        cash_dividend_total=0.0,
        split_ratio_total=1.0,
        return_pct=return_pct,
        outcome_status=OutcomeStatus.EVALUATED_OK.value,
        is_direction_hit=is_hit,
        evaluation_eligible=True,
    )
