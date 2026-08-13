from concurrent.futures import ThreadPoolExecutor
import os
import threading

from .alpha_vantage_common import AlphaVantageRateLimitError
from .config import get_config
from .providers import (
    DEFAULT_PROVIDER_RESOURCE_POLICY,
    ProviderResourcePolicy,
    build_default_registry,
)
from .trade_calendar import (
    DuplicateBarConflictError,
    is_historical_analysis_date,
    unavailable_analysis_date_reason,
)
from .vendor_result import (
    VendorEmpty,
    VendorFail,
    VendorOk,
    VendorRefuse,
)

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": ["get_stock_data"],
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": ["get_indicators"],
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement",
        ],
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ],
    },
    "realtime_data": {
        "description": "Real-time market quotes",
        "tools": ["get_realtime_quotes"],
    },
    "cn_market_data": {
        "description": "China A-share market sentiment and fund flow data",
        "tools": [
            "get_board_fund_flow",
            "get_individual_fund_flow",
            "get_lhb_detail",
            "get_zt_pool",
            "get_hot_stocks_xq",
            "get_shareholder_count",
            "get_margin_trading",
            "get_northbound_flow",
        ],
    },
    "institutional_risk": {
        "description": "Governance, pledge, earnings forecast and restricted release risks",
        "tools": [
            "get_restricted_release",
            "get_share_pledge",
            "get_earnings_forecast",
        ],
    },
}

_registry = build_default_registry()

PROVIDER_CALL_EXECUTOR_MAX_WORKERS = int(
    os.getenv("TA_PROVIDER_MAX_WORKERS", "8")
)
_PROVIDER_CALL_EXECUTOR = ThreadPoolExecutor(
    max_workers=PROVIDER_CALL_EXECUTOR_MAX_WORKERS,
    thread_name_prefix="provider-call",
)
_PROVIDER_SEMAPHORES: dict[tuple[str, int], threading.BoundedSemaphore] = {}
_PROVIDER_SEMAPHORE_LOCK = threading.Lock()


def _is_trace_enabled() -> bool:
    env_value = os.getenv("TA_TRACE")
    if env_value is not None:
        return env_value.strip().lower() in ("1", "true", "yes", "on")

    config = get_config()
    return bool(config.get("provider_trace", True))


def _trace(msg: str) -> None:
    if _is_trace_enabled():
        print(f"[provider-trace] {msg}", flush=True)


_TRACE_KEYS = ("symbol", "ticker", "start_date", "end_date", "curr_date", "indicator")


def _summarize_args(args: tuple, kwargs: dict) -> str:
    """格式化首参数（通常是 symbol）和常见日期/指标键，用于 trace 日志定位。"""
    parts = []
    if args:
        # 约定：所有 provider 方法首参数为 symbol/ticker
        parts.append(f"symbol={args[0]!r}")
        if len(args) >= 2:
            parts.append(f"arg2={args[1]!r}")
        if len(args) >= 3:
            parts.append(f"arg3={args[2]!r}")
    for k, v in kwargs.items():
        if k in _TRACE_KEYS:
            parts.append(f"{k}={v!r}")
    return " ".join(parts)


def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")


def get_vendor(category: str, method: str = None) -> str:
    """Get configured vendor for category or tool method."""
    config = get_config()

    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    return config.get("data_vendors", {}).get(category, "yfinance")


def _resolve_vendor_chain(method: str, configured_vendor: str) -> list[str]:
    configured = [v.strip() for v in configured_vendor.split(",") if v.strip()]
    fallback = configured.copy()

    for provider_name in _registry.list_names():
        if provider_name in fallback:
            continue
        provider = _registry.get(provider_name)
        # 占位 provider（如 cn_stub）不自动追加进 fallback chain，
        # 避免污染日志和兜底链；用户显式配置仍可强制使用。
        if getattr(provider, "is_placeholder", False) is True:
            continue
        fallback.append(provider_name)

    return fallback


# Historical news must use providers that can query and filter an explicit
# publication window. Live/current sources never receive a historical as-of.
_HISTORICAL_NEAR_WINDOW_NEWS_PROVIDER_ALLOWLIST = {
    "get_news": frozenset({"cn_akshare", "cn_investoday", "cn_cls"}),
    "get_global_news": frozenset({"cn_investoday", "cn_cls"}),
}


def _extract_as_of(method: str, args: tuple, kwargs: dict) -> str | None:
    """Return the as-of/analysis date for a routed method call, if present."""
    if "end_date" in kwargs:
        val = kwargs["end_date"]
        return None if val is None else str(val)
    if "curr_date" in kwargs:
        val = kwargs["curr_date"]
        return None if val is None else str(val)
    if "date" in kwargs:
        val = kwargs["date"]
        return None if val is None else str(val)

    positional = {
        "get_stock_data": 2,
        "get_news": 2,
        "get_indicators": 2,
        "get_global_news": 0,
        "get_board_fund_flow": 0,
        "get_zt_pool": 0,
        "get_hot_stocks_xq": 0,
        "get_fundamentals": 1,
        "get_insider_transactions": 1,
        "get_restricted_release": 1,
        "get_share_pledge": 1,
        "get_earnings_forecast": 1,
        "get_shareholder_count": 1,
        "get_margin_trading": 1,
        "get_northbound_flow": 1,
        "get_individual_fund_flow": 1,
        "get_lhb_detail": 1,
        "get_balance_sheet": 2,
        "get_cashflow": 2,
        "get_income_statement": 2,
    }
    idx = positional.get(method)
    if idx is not None and len(args) > idx:
        val = args[idx]
        return None if val is None else str(val)
    return None


_DATE_REQUIRED_METHODS = {
    "get_stock_data",
    "get_indicators",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
    "get_restricted_release",
    "get_share_pledge",
    "get_earnings_forecast",
    "get_shareholder_count",
    "get_margin_trading",
    "get_northbound_flow",
    "get_board_fund_flow",
    "get_individual_fund_flow",
    "get_lhb_detail",
    "get_zt_pool",
    "get_hot_stocks_xq",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
}


def _as_of_refusal(method: str, args: tuple, kwargs: dict) -> str | None:
    """Return a prompt-safe refusal for missing/invalid/future as-of dates."""
    if method == "get_realtime_quotes":
        # Signature: (symbols[, curr_date]); a missing/explicit-None curr_date
        # keeps the live dashboard path, while empty/invalid/future is refused
        # before any provider is contacted.
        if "curr_date" in kwargs and kwargs["curr_date"] is not None:
            return unavailable_analysis_date_reason(str(kwargs["curr_date"]))
        if len(args) >= 2 and args[1] is not None:
            return unavailable_analysis_date_reason(str(args[1]))
        return None
    if method not in _DATE_REQUIRED_METHODS:
        return None
    as_of = _extract_as_of(method, args, kwargs)
    return unavailable_analysis_date_reason(as_of)


def _analysis_date_for_method(method: str, args: tuple, kwargs: dict) -> str | None:
    """Extract the analysis/as-of date used for historical near-window refusal."""
    if method == "get_global_news":
        if "curr_date" in kwargs and kwargs["curr_date"] is not None:
            return str(kwargs["curr_date"])
        if args:
            return str(args[0])
        return None
    if method == "get_news":
        # Signature: (ticker, start_date, end_date); end_date is the analysis date.
        if "end_date" in kwargs and kwargs["end_date"] is not None:
            return str(kwargs["end_date"])
        if len(args) >= 3:
            return str(args[2])
        return None
    return None


def _historical_news_provider_allowlist(
    method: str, args: tuple, kwargs: dict
) -> set[str] | None:
    """Return the only providers allowed for a historical news request."""
    configured = _HISTORICAL_NEAR_WINDOW_NEWS_PROVIDER_ALLOWLIST.get(method)
    if not configured:
        return None
    analysis_date = _analysis_date_for_method(method, args, kwargs)
    if not is_historical_analysis_date(analysis_date):
        return None
    return set(configured)


def _historical_news_failure(method: str) -> str:
    """Keep historical news failures explicit without exposing provider details."""
    if method == "get_news":
        return (
            "【数据获取失败】历史个股新闻仅允许使用具备发布时间过滤能力的"
            "东财历史新闻或今日投资接口；未获取到可验证数据，"
            "不得回退到实时新闻源，本项不可用。"
        )
    return (
        "【数据获取失败】历史宏观新闻允许使用今日投资或财联社历史接口；"
        "未获取到可验证数据（接口失败、覆盖不完整或返回结构异常），"
        "不得回退到实时新闻源，本项不可用。"
    )


def _resource_policy_for(provider_name: str) -> ProviderResourcePolicy:
    resolver = getattr(_registry, "resource_policy", None)
    if callable(resolver):
        candidate = resolver(provider_name)
        if isinstance(candidate, ProviderResourcePolicy):
            return candidate
    return DEFAULT_PROVIDER_RESOURCE_POLICY


def _provider_semaphore(
    provider_name: str, max_concurrency: int
) -> threading.BoundedSemaphore:
    with _PROVIDER_SEMAPHORE_LOCK:
        key = (provider_name, max_concurrency)
        semaphore = _PROVIDER_SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(max_concurrency)
            _PROVIDER_SEMAPHORES[key] = semaphore
        return semaphore


def _submit_provider_call(
    provider_name: str,
    policy: ProviderResourcePolicy,
    impl_func,
    args: tuple,
    kwargs: dict,
):
    """Run one provider call under a hard per-provider concurrency bound."""
    semaphore = _provider_semaphore(provider_name, policy.max_concurrency)
    if not semaphore.acquire(timeout=policy.timeout_seconds):
        raise TimeoutError(f"{provider_name} concurrency slot unavailable")
    try:
        future = _PROVIDER_CALL_EXECUTOR.submit(
            lambda: impl_func(*args, **kwargs)
        )
    except BaseException:
        semaphore.release()
        raise
    future.add_done_callback(lambda _: semaphore.release())
    return future


def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to provider implementations with fallback support.

    Provider returns are interpreted with typed vendor semantics
    (KNOWN_ISSUES #1):

    - plain value / ``VendorOk``  -> successful hit, chain stops
    - ``VendorRefuse``            -> this source cannot serve; chain stops
      (unless ``allow_peers`` names same-semantics peers to continue through)
    - ``VendorEmpty``             -> confirmed empty; chain stops
    - ``VendorFail`` / exception  -> transient failure; chain falls through
    """
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    fallback_vendors = _resolve_vendor_chain(method, vendor_config)
    historical_provider_allowlist = _historical_news_provider_allowlist(
        method, args, kwargs
    )
    if historical_provider_allowlist is not None:
        fallback_vendors = [
            vendor
            for vendor in fallback_vendors
            if vendor in historical_provider_allowlist
        ]
    args_summary = _summarize_args(args, kwargs)
    last_exc = None
    refusal_reason = None
    peer_allowlist: set[str] | None = None
    _trace(
        f"method={method} {args_summary} category={category} "
        f"configured='{vendor_config}' chain={fallback_vendors}"
    )

    as_of_refusal = _as_of_refusal(method, args, kwargs)
    if as_of_refusal is not None:
        _trace(
            f"method={method} {args_summary} status=as-of-refuse "
            f"reason=missing-invalid-future providers_hit=0"
        )
        return as_of_refusal

    for vendor in fallback_vendors:
        if peer_allowlist is not None and vendor not in peer_allowlist:
            _trace(
                f"method={method} {args_summary} vendor={vendor} "
                "status=skip reason=refuse-peer-allowlist"
            )
            continue
        provider = _registry.get(vendor)
        if provider is None:
            _trace(f"method={method} {args_summary} vendor={vendor} status=skip reason=not-registered")
            continue

        impl_func = getattr(provider, method, None)
        if impl_func is None:
            _trace(f"method={method} {args_summary} vendor={vendor} status=skip reason=not-implemented")
            continue

        policy = _resource_policy_for(vendor)
        for attempt in range(policy.max_retries + 1):
            future = None
            try:
                future = _submit_provider_call(
                    vendor, policy, impl_func, args, kwargs
                )
                result = future.result(timeout=policy.timeout_seconds)
            except (AlphaVantageRateLimitError, NotImplementedError) as exc:
                last_exc = exc
                # Try next provider for transient/routing issues or placeholder providers.
                _trace(
                    f"method={method} {args_summary} vendor={vendor} status=fallback "
                    f"reason={type(exc).__name__}: {exc}"
                )
                break
            except TimeoutError as exc:
                last_exc = exc
                if future is not None:
                    future.cancel()
                _trace(
                    f"method={method} {args_summary} vendor={vendor} "
                    f"status=provider-timeout attempt={attempt + 1}/"
                    f"{policy.max_retries + 1} reason=TimeoutError"
                )
                if attempt < policy.max_retries:
                    continue
                _trace(
                    f"method={method} {args_summary} vendor={vendor} "
                    "status=fallback reason=TimeoutError"
                )
                break
            except DuplicateBarConflictError as exc:
                # Data-integrity conflict (same date, different OHLCV/Volume): this
                # is an explicit refusal, not a transient provider failure. Do not
                # retry or fall back to another vendor, which would silently mask
                # the conflict by using a different source's row order.
                _trace(
                    f"method={method} {args_summary} vendor={vendor} "
                    "status=unavailable reason=duplicate-bar-conflict"
                )
                return f"【数据获取失败】{exc}，本项不可用。"
            except Exception as exc:
                last_exc = exc
                _trace(
                    f"method={method} {args_summary} vendor={vendor} "
                    f"status=provider-error attempt={attempt + 1}/"
                    f"{policy.max_retries + 1} reason={type(exc).__name__}: {exc}"
                )
                if attempt < policy.max_retries:
                    continue
                _trace(
                    f"method={method} {args_summary} vendor={vendor} "
                    f"status=fallback reason={type(exc).__name__}: {exc}"
                )
                break
            else:
                # Provider returned normally — interpret typed vendor semantics.
                if isinstance(result, VendorRefuse):
                    if result.allow_peers:
                        refusal_reason = result.to_prompt()
                        peer_allowlist = set(result.allow_peers)
                        _trace(
                            f"method={method} {args_summary} vendor={vendor} "
                            f"status=refuse reason=refuse-with-peers "
                            f"allow_peers={sorted(peer_allowlist)}"
                        )
                        break
                    if historical_provider_allowlist is not None:
                        return _historical_news_failure(method)
                    _trace(
                        f"method={method} {args_summary} vendor={vendor} "
                        "status=refuse reason=vendor-refuse"
                    )
                    return result.to_prompt()
                if isinstance(result, VendorEmpty):
                    if historical_provider_allowlist is not None:
                        last_exc = RuntimeError("historical news is empty")
                        _trace(
                            f"method={method} {args_summary} vendor={vendor} "
                            "status=fallback reason=historical-empty"
                        )
                        break
                    _trace(
                        f"method={method} {args_summary} vendor={vendor} "
                        "status=confirmed-empty reason=vendor-empty"
                    )
                    return result.to_prompt()
                if isinstance(result, VendorFail):
                    last_exc = RuntimeError(result.error)
                    _trace(
                        f"method={method} {args_summary} vendor={vendor} "
                        f"status=fallback reason=VendorFail: {result.error}"
                    )
                    break
                if isinstance(result, VendorOk):
                    if (
                        historical_provider_allowlist is not None
                        and not isinstance(result.payload, str)
                    ):
                        last_exc = RuntimeError(
                            "historical news returned malformed result"
                        )
                        _trace(
                            f"method={method} {args_summary} vendor={vendor} "
                            "status=fallback reason=malformed-result"
                        )
                        break
                    prompt = result.to_prompt()
                    if historical_provider_allowlist is not None and not prompt.strip():
                        last_exc = RuntimeError("historical news is empty")
                        _trace(
                            f"method={method} {args_summary} vendor={vendor} "
                            "status=fallback reason=empty-result"
                        )
                        break
                    _trace(f"method={method} {args_summary} vendor={vendor} status=hit")
                    return prompt
                if (
                    historical_provider_allowlist is not None
                    and not isinstance(result, str)
                ):
                    last_exc = RuntimeError(
                        "historical news returned malformed result"
                    )
                    _trace(
                        f"method={method} {args_summary} vendor={vendor} "
                        "status=fallback reason=malformed-result"
                    )
                    break
                if historical_provider_allowlist is not None and not result.strip():
                    last_exc = RuntimeError("historical news is empty")
                    _trace(
                        f"method={method} {args_summary} vendor={vendor} "
                        "status=fallback reason=empty-result"
                    )
                    break
                _trace(f"method={method} {args_summary} vendor={vendor} status=hit")
                return result

    if historical_provider_allowlist is not None:
        _trace(
            f"method={method} {args_summary} status=historical-failure "
            "reason=no-verified-historical-news-data"
        )
        return _historical_news_failure(method)
    if refusal_reason is not None:
        _trace(f"method={method} {args_summary} status=refuse-sticky reason=no-peer-hit")
        return refusal_reason
    _trace(f"method={method} {args_summary} status=failed reason=no-available-vendor")
    if last_exc is not None:
        if historical_provider_allowlist is not None:
            return _historical_news_failure(method)
        raise RuntimeError(
            f"No available vendor for method '{method}'. "
            f"Configured chain: {fallback_vendors}. "
            f"Last error: {type(last_exc).__name__}: {last_exc}"
        ) from last_exc
    raise RuntimeError(
        f"No available vendor for method '{method}'. "
        f"Configured chain: {fallback_vendors}"
    )
